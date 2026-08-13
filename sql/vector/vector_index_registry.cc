/* Copyright (c) 2026, Oracle and/or its affiliates.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0,
   as published by the Free Software Foundation.

   This program is designed to work with certain software (including
   but not limited to OpenSSL) that is licensed under separate terms,
   as designated in a particular file or component or in included license
   documentation.  The authors of MySQL hereby grant you an additional
   permission to link the program and your derivative works with the
   separately licensed software that they have either included with
   the program or referenced in the documentation.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

#include "sql/vector/vector_index_registry.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "my_dbug.h"
#include "sql/log.h"
#include "sql/mysqld.h"
#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_diagnostics.h"
#include "sql/vector/vector_index_identity.h"
#include "sql/vector/vector_index_registry_internal.h"
#include "sql/vector/vector_index_runtime_config.h"
#include "sql/vector/vector_index_service.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_mapped_search.h"
#include "sql/vector/vector_status.h"
#include "sql/vector/vector_trx_participant.h"

namespace vector_index_registry::detail {

std::shared_mutex g_registry_mutex;
vector_index::index_service g_index_service;
bool g_metadata_loaded = false;
bool g_pending_spill_startup_cleanup_complete = false;
registry_health_state g_registry_health = registry_health_state::kReady;
std::string g_registry_failure_reason;
truth_recovery_state g_truth_recovery_state;
std::atomic<uint64_t> g_next_txn_id{1};
uint64_t g_manifest_version = 1;
uint64_t g_manifest_metadata_checkpoint = 0;
uint64_t g_manifest_committed_checkpoint = 0;
uint64_t g_manifest_change_log_checkpoint = 0;
uint64_t g_next_change_log_sequence = 1;
std::vector<vector_index_metadata_store::change_log_row> g_change_log_rows;
std::vector<vector_index_metadata_store::prepared_change_row>
    g_prepared_change_rows;
std::vector<vector_index_metadata_store::segment_task_row> g_segment_task_rows;
std::unordered_map<uint64_t, explicit_txn_owner> g_explicit_txn_owners;
std::unordered_map<uint64_t, std::unordered_set<uint64_t>>
    g_explicit_txns_by_thd;

std::vector<recovery_action> g_recovery_actions;

std::unordered_map<uint64_t, thd_txn_context> g_thd_txn_contexts;
std::unordered_map<std::string, index_binding> g_index_bindings;
std::unordered_map<std::string, std::string> g_index_owner_schemas;

void fail_stop_registry_locked(const char *reason) {
  g_registry_health = registry_health_state::kFailed;
  g_registry_failure_reason =
      reason == nullptr ? "registry_state_restore_failed" : reason;
}

namespace {

constexpr size_t kVectorChangeLogCompactRows = 1000000;
constexpr const char *kLifecycleCreating = "creating";
constexpr const char *kLifecycleBackfilling = "backfilling";
constexpr const char *kLifecycleRecoveryRequired = "recovery_required";
constexpr uint32_t kSegmentTaskMissingArtifactError = 1;

bool is_incomplete_create_lifecycle(const std::string &state) {
  return state == kLifecycleCreating || state == kLifecycleBackfilling;
}

bool has_index_name(const std::unordered_set<std::string> &index_names,
                    const std::string &index_name) {
  return index_names.find(index_name) != index_names.end();
}

std::unordered_set<std::string> remove_incomplete_create_metadata_rows(
    std::vector<vector_index_metadata_store::metadata_row> *rows) {
  std::unordered_set<std::string> removed_index_names;
  if (rows == nullptr) return removed_index_names;

  rows->erase(std::remove_if(rows->begin(), rows->end(),
                             [&removed_index_names](const auto &row) {
                               if (!is_incomplete_create_lifecycle(
                                       row.lifecycle_state)) {
                                 return false;
                               }
                               removed_index_names.insert(row.index_name);
                               return true;
                             }),
              rows->end());
  return removed_index_names;
}

void remove_committed_rows_for_indexes(
    const std::unordered_set<std::string> &index_names,
    std::vector<vector_index_metadata_store::committed_row> *rows) {
  if (index_names.empty() || rows == nullptr) return;
  rows->erase(std::remove_if(rows->begin(), rows->end(),
                             [&index_names](const auto &row) {
                               return has_index_name(index_names,
                                                     row.index_name);
                             }),
              rows->end());
}

void remove_change_log_rows_for_indexes(
    const std::unordered_set<std::string> &index_names,
    std::vector<vector_index_metadata_store::change_log_row> *rows) {
  if (index_names.empty() || rows == nullptr) return;
  rows->erase(std::remove_if(rows->begin(), rows->end(),
                             [&index_names](const auto &row) {
                               return has_index_name(index_names,
                                                     row.index_name);
                             }),
              rows->end());
}

void remove_prepared_rows_for_indexes(
    const std::unordered_set<std::string> &index_names,
    std::vector<vector_index_metadata_store::prepared_change_row> *rows) {
  if (index_names.empty() || rows == nullptr) return;
  rows->erase(std::remove_if(rows->begin(), rows->end(),
                             [&index_names](const auto &row) {
                               return has_index_name(index_names,
                                                     row.index_name);
                             }),
              rows->end());
}

void remove_segment_task_rows_for_indexes(
    const std::unordered_set<std::string> &index_names,
    std::vector<vector_index_metadata_store::segment_task_row> *rows) {
  if (index_names.empty() || rows == nullptr) return;
  rows->erase(std::remove_if(rows->begin(), rows->end(),
                             [&index_names](const auto &row) {
                               return has_index_name(index_names,
                                                     row.index_name);
                             }),
              rows->end());
}

std::unordered_set<std::string> metadata_index_names(
    const std::vector<vector_index_metadata_store::metadata_row> &rows) {
  std::unordered_set<std::string> index_names;
  for (const auto &row : rows) {
    index_names.insert(row.index_name);
  }
  return index_names;
}

bool path_exists(const std::string &path) {
  if (path.empty()) return true;
  std::error_code ec;
  return std::filesystem::exists(path, ec) && !ec;
}

bool artifact_prefix_exists(const std::string &prefix) {
  if (prefix.empty()) return true;
  std::error_code ec;
  if (std::filesystem::exists(prefix, ec) && !ec) return true;
  if (ec) return false;

  const std::filesystem::path prefix_path(prefix);
  const std::filesystem::path parent = prefix_path.parent_path();
  const std::string filename_prefix = prefix_path.filename().string();
  if (parent.empty() || filename_prefix.empty() ||
      !std::filesystem::is_directory(parent, ec) || ec) {
    return false;
  }

  for (const auto &entry : std::filesystem::directory_iterator(
           parent, std::filesystem::directory_options::skip_permission_denied,
           ec)) {
    if (ec) return false;
    const std::string filename = entry.path().filename().string();
    if (filename.rfind(filename_prefix, 0) == 0) return true;
  }
  return false;
}

void remove_artifact_prefix_best_effort(const std::string &prefix) {
  if (prefix.empty()) return;

  std::error_code ec;
  std::filesystem::remove_all(prefix, ec);
  ec.clear();

  const std::filesystem::path prefix_path(prefix);
  const std::filesystem::path parent = prefix_path.parent_path();
  const std::string filename_prefix = prefix_path.filename().string();
  if (parent.empty() || filename_prefix.empty() ||
      !std::filesystem::is_directory(parent, ec) || ec) {
    return;
  }

  for (const auto &entry : std::filesystem::directory_iterator(
           parent, std::filesystem::directory_options::skip_permission_denied,
           ec)) {
    if (ec) return;
    const std::string filename = entry.path().filename().string();
    if (filename.rfind(filename_prefix, 0) == 0) {
      std::error_code remove_ec;
      std::filesystem::remove_all(entry.path(), remove_ec);
    }
  }
}

bool segment_task_raw_input_complete(
    const vector_index_metadata_store::segment_task_row &row) {
  return path_exists(row.vector_path) && path_exists(row.docid_path);
}

bool segment_task_ready_artifacts_complete(
    const vector_index_metadata_store::segment_task_row &row) {
  return segment_task_raw_input_complete(row) &&
         artifact_prefix_exists(row.artifact_prefix);
}

bool normalize_segment_tasks_for_recovery(
    const std::vector<vector_index_metadata_store::metadata_row> &metadata_rows,
    std::vector<vector_index_metadata_store::segment_task_row> *rows) {
  if (rows == nullptr) return false;

  bool changed = false;
  const std::unordered_set<std::string> live_indexes =
      metadata_index_names(metadata_rows);
  rows->erase(std::remove_if(rows->begin(), rows->end(),
                             [&live_indexes, &changed](const auto &row) {
                               if (!has_index_name(live_indexes,
                                                   row.index_name)) {
                                 changed = true;
                                 remove_artifact_prefix_best_effort(
                                     row.artifact_prefix);
                                 return true;
                               }
                               if (row.state == vector_index_metadata_store::
                                                    segment_task_state::
                                                        kAbandoned) {
                                 changed = true;
                                 remove_artifact_prefix_best_effort(
                                     row.artifact_prefix);
                                 return true;
                               }
                               return false;
                             }),
              rows->end());

  for (auto &row : *rows) {
    using vector_index_metadata_store::segment_task_state;
    switch (row.state) {
      case segment_task_state::kPending:
        remove_artifact_prefix_best_effort(row.artifact_prefix);
        if (!segment_task_raw_input_complete(row)) {
          row.state = segment_task_state::kFailed;
          row.last_error_code = kSegmentTaskMissingArtifactError;
          changed = true;
        }
        break;
      case segment_task_state::kBuilding:
        remove_artifact_prefix_best_effort(row.artifact_prefix);
        row.state = segment_task_raw_input_complete(row)
                        ? segment_task_state::kPending
                        : segment_task_state::kFailed;
        if (row.state == segment_task_state::kFailed) {
          row.last_error_code = kSegmentTaskMissingArtifactError;
        }
        changed = true;
        break;
      case segment_task_state::kReady:
        if (!segment_task_ready_artifacts_complete(row)) {
          row.state = segment_task_state::kFailed;
          row.last_error_code = kSegmentTaskMissingArtifactError;
          changed = true;
        }
        break;
      case segment_task_state::kFailed:
        break;
      case segment_task_state::kAbandoned:
        break;
    }
  }
  return changed;
}

uint32_t effective_hnsw_build_threads(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kHnswlib) {
    return 0;
  }
  if (options.defaults_resolved) return options.build_threads;
  if (options.build_threads_specified && options.build_threads != 0) {
    return options.build_threads;
  }
  return static_cast<uint32_t>(opt_vector_hnsw_build_threads);
}

uint32_t effective_diskann_build_threads(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kDiskAnn) {
    return 0;
  }
  if (options.defaults_resolved) return options.build_threads;
  if (options.build_threads_specified && options.build_threads != 0) {
    return options.build_threads;
  }
  return static_cast<uint32_t>(opt_vector_diskann_build_threads);
}

uint32_t effective_diskann_max_degree(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kDiskAnn) {
    return 0;
  }
  if (options.defaults_resolved) return options.diskann_max_degree;
  if (options.diskann_max_degree != 0) return options.diskann_max_degree;
  return static_cast<uint32_t>(opt_vector_diskann_max_degree);
}

uint32_t effective_diskann_build_complexity(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kDiskAnn) {
    return 0;
  }
  if (options.defaults_resolved) return options.diskann_build_complexity;
  if (options.diskann_build_complexity != 0) {
    return options.diskann_build_complexity;
  }
  return static_cast<uint32_t>(opt_vector_diskann_build_complexity);
}

uint64_t effective_diskann_pq_code_budget_size(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kDiskAnn) {
    return 0;
  }
  if (options.defaults_resolved) return options.diskann_pq_code_budget_size;
  return opt_vector_diskann_pq_code_budget_size;
}

uint32_t effective_diskann_disk_pq_dims(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kDiskAnn) {
    return 0;
  }
  if (options.defaults_resolved) return options.diskann_disk_pq_dims;
  if (options.diskann_disk_pq_dims != 0) return options.diskann_disk_pq_dims;
  return static_cast<uint32_t>(opt_vector_diskann_disk_pq_dims);
}

bool effective_diskann_accelerate_build(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kDiskAnn) {
    return false;
  }
  if (options.defaults_resolved) return options.diskann_accelerate_build;
  if (options.diskann_accelerate_build_specified) {
    return options.diskann_accelerate_build;
  }
  return opt_vector_diskann_accelerate_build;
}

bool effective_diskann_shuffle_build(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kDiskAnn) {
    return false;
  }
  if (options.defaults_resolved) return options.diskann_shuffle_build;
  if (options.diskann_shuffle_build_specified) {
    return options.diskann_shuffle_build;
  }
  return opt_vector_diskann_shuffle_build;
}

bool effective_diskann_use_bfs_cache(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kDiskAnn) {
    return false;
  }
  if (options.defaults_resolved) return options.diskann_use_bfs_cache;
  if (options.diskann_use_bfs_cache_specified) {
    return options.diskann_use_bfs_cache;
  }
  return opt_vector_diskann_use_bfs_cache;
}

uint32_t effective_diskann_search_complexity(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kDiskAnn) {
    return 0;
  }
  if (options.defaults_resolved) return options.diskann_search_complexity;
  return static_cast<uint32_t>(opt_vector_diskann_search_complexity);
}

uint32_t effective_diskann_search_beamwidth(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kDiskAnn) {
    return 0;
  }
  if (options.defaults_resolved) return options.diskann_search_beamwidth;
  return static_cast<uint32_t>(opt_vector_diskann_search_beamwidth);
}

uint32_t effective_faiss_build_threads(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kFaiss) {
    return 0;
  }
  if (options.defaults_resolved) return options.build_threads;
  if (options.build_threads_specified && options.build_threads != 0) {
    return options.build_threads;
  }
  return static_cast<uint32_t>(opt_vector_faiss_build_threads);
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
size_t g_change_log_compact_threshold_for_testing = 0;
#endif

size_t change_log_compact_threshold() {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  if (g_change_log_compact_threshold_for_testing != 0)
    return g_change_log_compact_threshold_for_testing;
#endif
  size_t threshold = kVectorChangeLogCompactRows;
  DBUG_EXECUTE_IF("vector_truth_store_compact_after_3_rows", threshold = 3;);
  return threshold;
}

bool change_log_compaction_needed_locked() {
  return !g_change_log_rows.empty() &&
         g_change_log_rows.size() >= change_log_compact_threshold();
}

bool compact_change_log_locked() {
  vector_status::record_truth_store_compact_request();
  if (!persist_metadata_locked(nullptr)) {
    vector_status::record_truth_store_compact_failure();
    return false;
  }
  const std::vector<vector_index_metadata_store::change_log_row> empty_rows;
  if (!vector_index_truth_store::get()->save_change_log(empty_rows)) {
    vector_status::record_metadata_persist_failure();
    vector_status::record_change_log_persist_failure();
    vector_status::record_truth_store_compact_failure();
    return false;
  }

  g_change_log_rows.clear();
  g_manifest_change_log_checkpoint = 0;
  return true;
}

bool prepared_rows_are_valid(
    const std::vector<vector_index_metadata_store::prepared_change_row> &rows) {
  using xid_identity = std::tuple<int64_t, int64_t, int64_t, std::string>;
  try {
    std::map<xid_identity, std::pair<uint64_t, bool>> xid_to_txn;
    std::map<uint64_t, xid_identity> txn_to_xid;
    for (const auto &row : rows) {
      if (row.txn_id == 0 ||
          !vector_index_metadata_store::valid_prepared_xid(row)) {
        return false;
      }

      const xid_identity identity{row.format_id, row.gtrid_length,
                                  row.bqual_length, row.xid_data};
      const auto [xid_it, xid_inserted] = xid_to_txn.emplace(
          identity, std::make_pair(row.txn_id, row.prepared_in_tc));
      if (!xid_inserted && (xid_it->second.first != row.txn_id ||
                            xid_it->second.second != row.prepared_in_tc)) {
        return false;
      }

      const auto [txn_it, txn_inserted] =
          txn_to_xid.emplace(row.txn_id, identity);
      if (!txn_inserted && txn_it->second != identity) return false;
    }
  } catch (...) {
    return false;
  }
  return true;
}

}  // namespace

bool describe_publication_token_locked(
    const std::string &index_name,
    vector_index_truth_store::publication_token *token) {
  if (index_name.empty() || token == nullptr) return false;
  *token = vector_index_truth_store::publication_token{};
  if (!g_index_service.index_exists(index_name)) return true;

  vector_index::index_service::index_config config;
  vector_index::index_service::index_publication_state publication;
  uint64_t lifecycle_version = 0;
  if (!g_index_service.describe_index(index_name, &config, nullptr, nullptr,
                                      nullptr, nullptr, &lifecycle_version) ||
      !g_index_service.describe_publication_state(index_name, &publication)) {
    return false;
  }

  token->exists = true;
  token->index_identity = publication.index_identity;
  token->truth_generation = publication.truth_generation;
  token->config_generation = publication.config_generation;
  token->artifact_generation = publication.artifact_generation;
  token->runtime_generation = publication.runtime_generation;
  token->lifecycle_version = lifecycle_version;
  token->source_generation =
      config.consistency_mode ==
              vector_index::index_consistency_mode::kStandalone
          ? g_index_service.standalone_source_generation(index_name)
          : 0;
  return true;
}

bool publication_tokens_equal(
    const vector_index_truth_store::publication_token &lhs,
    const vector_index_truth_store::publication_token &rhs) {
  return lhs.exists == rhs.exists &&
         lhs.index_identity == rhs.index_identity &&
         lhs.truth_generation == rhs.truth_generation &&
         lhs.config_generation == rhs.config_generation &&
         lhs.artifact_generation == rhs.artifact_generation &&
         lhs.runtime_generation == rhs.runtime_generation &&
         lhs.lifecycle_version == rhs.lifecycle_version &&
         lhs.source_generation == rhs.source_generation;
}

bool xid_matches_row(
    const XID &xid,
    const vector_index_metadata_store::prepared_change_row &row) {
  return vector_index_metadata_store::prepared_xid_matches(row, xid);
}

bool xid_from_prepared_row(
    const vector_index_metadata_store::prepared_change_row &row, XID *xid) {
  return vector_index_metadata_store::get_prepared_xid(row, xid);
}

bool find_prepared_rows_for_xid_locked(
    const XID &xid,
    std::vector<vector_index_metadata_store::prepared_change_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();
  for (const auto &row : g_prepared_change_rows) {
    if (xid_matches_row(xid, row)) rows->push_back(row);
  }
  return true;
}

bool has_prepared_xid_locked(const XID &xid) {
  return std::any_of(
      g_prepared_change_rows.begin(), g_prepared_change_rows.end(),
      [&xid](const vector_index_metadata_store::prepared_change_row &row) {
        return xid_matches_row(xid, row);
      });
}

void erase_prepared_rows_for_xid_locked(const XID &xid) {
  g_prepared_change_rows.erase(
      std::remove_if(
          g_prepared_change_rows.begin(), g_prepared_change_rows.end(),
          [&xid](const auto &row) { return xid_matches_row(xid, row); }),
      g_prepared_change_rows.end());
}

void queue_recovery_action_locked(const XID &xid, recovery_action_type type,
                                  bool conditional) {
  for (auto &action : g_recovery_actions) {
    if (action.xid.eq(&xid)) {
      action.type = type;
      action.conditional = action.conditional && conditional;
      return;
    }
  }
  g_recovery_actions.push_back(recovery_action{xid, type, conditional});
}

bool snapshot_runtime_state_locked(
    std::vector<vector_index_metadata_store::metadata_row> *metadata_before,
    vector_index::index_service::committed_state *committed_before,
    std::vector<vector_index_metadata_store::change_log_row> *change_log_before,
    std::vector<std::string> *lagging_indexes_before);
bool rollback_runtime_state_locked(
    const std::vector<vector_index_metadata_store::metadata_row>
        &metadata_before,
    const vector_index::index_service::committed_state &committed_before,
    const std::vector<vector_index_metadata_store::change_log_row>
        &change_log_before,
    const std::vector<std::string> &lagging_indexes_before);
bool persist_registry_state_locked();
bool persist_prepared_locked();

uint64_t allocate_txn_id() {
  uint64_t candidate = g_next_txn_id.load(std::memory_order_relaxed);
  while (candidate != 0) {
    const uint64_t next =
        candidate == std::numeric_limits<uint64_t>::max() ? 0 : candidate + 1;
    if (g_next_txn_id.compare_exchange_weak(
            candidate, next, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      return candidate;
    }
  }
  return 0;
}

void advance_txn_id_high_water(uint64_t txn_id) {
  if (txn_id == 0) return;

  const uint64_t next =
      txn_id == std::numeric_limits<uint64_t>::max() ? 0 : txn_id + 1;
  uint64_t current = g_next_txn_id.load(std::memory_order_relaxed);
  while (current != 0 && current <= txn_id &&
         !g_next_txn_id.compare_exchange_weak(
             current, next, std::memory_order_relaxed,
             std::memory_order_relaxed)) {
  }
}

std::string make_stmt_savepoint_name(uint64_t statement_id) {
  return "__stmt_" + std::to_string(statement_id);
}

std::string make_user_savepoint_name(const std::string &name) {
  return "user:" + name;
}

std::string make_mapped_index_name(const std::string &db_name,
                                   const std::string &table_name,
                                   const std::string &column_name) {
  return vector_index_identity::make_index_name(db_name, table_name,
                                                column_name);
}

bool parse_mapped_index_name(const std::string &index_name,
                             std::string *schema_name, std::string *table_name,
                             std::string *column_name) {
  vector_index_identity::mapped_index_identity identity;
  if (!vector_index_identity::parse_index_name(index_name, &identity)) {
    return false;
  }
  if (schema_name != nullptr) {
    *schema_name = identity.schema_name;
  }
  if (table_name != nullptr) {
    *table_name = identity.table_name;
  }
  if (column_name != nullptr) {
    *column_name = identity.column_name;
  }
  return true;
}

index_binding binding_from_name(const std::string &index_name) {
  index_binding binding;
  (void)parse_mapped_index_name(index_name, &binding.schema_name,
                                &binding.table_name, &binding.column_name);
  return binding;
}

index_binding binding_for_index_locked(const std::string &index_name) {
  const auto it = g_index_bindings.find(index_name);
  if (it != g_index_bindings.end()) {
    return it->second;
  }
  return index_binding();
}

std::string owner_schema_for_index_locked(const std::string &index_name) {
  const auto it = g_index_owner_schemas.find(index_name);
  return it == g_index_owner_schemas.end() ? std::string() : it->second;
}

bool make_diskann_artifact_identity_locked(
    const std::string &index_name,
    const vector_index::index_service::index_config &config,
    const vector_index::index_service::index_publication_state &publication,
    size_t doc_id_count,
    vector_index::diskann_artifact_identity *identity) {
  if (identity == nullptr || index_name.empty() ||
      publication.index_identity == 0 || publication.config_generation == 0 ||
      config.dimension == 0 ||
      config.dimension > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const index_binding binding = binding_for_index_locked(index_name);
  *identity = {};
  identity->index_identity = publication.index_identity;
  identity->truth_generation = publication.truth_generation;
  identity->config_generation = publication.config_generation;
  identity->doc_id_count = static_cast<uint64_t>(doc_id_count);
  identity->dimension = static_cast<uint32_t>(config.dimension);
  identity->metric = vector_index::metric_to_string(config.metric);
  identity->mode = vector_index::backend_mode_to_string(config.mode);
  identity->provider = vector_index::backend_provider_to_string(config.provider);
  identity->consistency_mode =
      vector_index::index_consistency_mode_to_string(config.consistency_mode);
  identity->schema_name = binding.schema_name;
  identity->table_name = binding.table_name;
  identity->column_name = binding.column_name;
  identity->doc_id_column_name = binding.doc_id_column_name;
  return true;
}

void set_index_binding_locked(const std::string &index_name,
                              const std::string &schema_name,
                              const std::string &table_name,
                              const std::string &column_name,
                              const std::string &doc_id_column_name) {
  g_index_bindings[index_name] =
      index_binding{schema_name, table_name, column_name, doc_id_column_name};
}

void erase_index_binding_and_owner_schema_locked(
    const std::string &index_name) {
  g_index_bindings.erase(index_name);
  g_index_owner_schemas.erase(index_name);
}

void rename_index_binding_locked(const std::string &old_index_name,
                                 const std::string &new_index_name) {
  const auto it = g_index_bindings.find(old_index_name);
  if (it == g_index_bindings.end()) return;

  index_binding binding = it->second;
  const index_binding fallback = binding_from_name(new_index_name);
  if (!fallback.schema_name.empty() || !fallback.table_name.empty() ||
      !fallback.column_name.empty()) {
    binding.schema_name = fallback.schema_name;
    binding.table_name = fallback.table_name;
    binding.column_name = fallback.column_name;
  }
  g_index_bindings.erase(it);
  g_index_bindings.emplace(new_index_name, std::move(binding));
}

void set_index_binding_and_owner_schema_locked(
    const std::string &index_name, const index_binding &binding,
    const std::string &owner_schema) {
  set_index_binding_locked(index_name, binding.schema_name, binding.table_name,
                           binding.column_name, binding.doc_id_column_name);
  g_index_owner_schemas[index_name] = owner_schema;
}

void rename_index_binding_and_owner_schema_locked(
    const std::string &old_index_name, const std::string &new_index_name,
    const std::string &owner_schema) {
  rename_index_binding_locked(old_index_name, new_index_name);
  g_index_owner_schemas.erase(old_index_name);
  g_index_owner_schemas[new_index_name] = owner_schema;
}

bool resolve_create_index_definition(
    const std::string &mode, const std::string &provider,
    const vector_index_registry::create_index_options &requested_options,
    std::string *resolved_mode, std::string *resolved_provider,
    vector_index_registry::create_index_options *resolved_options) {
  if (resolved_mode == nullptr || resolved_provider == nullptr ||
      resolved_options == nullptr) {
    return false;
  }

  vector_index::backend_provider provider_value;
  if (provider.empty()) {
    if (!vector_index::default_backend_provider(&provider_value)) return false;
  } else if (!vector_index::parse_backend_provider(provider, &provider_value) ||
             !vector_index::backend_provider_supported(provider_value)) {
    return false;
  }

  vector_index::backend_mode mode_value;
  if (mode.empty()) {
    if (!vector_index::default_backend_mode_for_provider(provider_value,
                                                        &mode_value)) {
      return false;
    }
  } else if (!vector_index::parse_backend_mode(mode, &mode_value)) {
    return false;
  }
  if (!vector_index::runtime_provider_accepts_mode(provider_value, mode_value)) {
    return false;
  }

  *resolved_provider =
      vector_index::backend_provider_to_string(provider_value);
  *resolved_mode = vector_index::backend_mode_to_string(mode_value);

  vector_index_registry::create_index_options resolved = requested_options;
  switch (provider_value) {
    case vector_index::backend_provider::kHnswlib:
      resolved.build_threads =
          effective_hnsw_build_threads(*resolved_provider, requested_options);
      break;
    case vector_index::backend_provider::kFaiss:
      resolved.build_threads =
          effective_faiss_build_threads(*resolved_provider, requested_options);
      break;
    case vector_index::backend_provider::kDiskAnn:
      resolved.build_threads =
          effective_diskann_build_threads(*resolved_provider, requested_options);
      break;
    case vector_index::backend_provider::kNative:
      resolved.build_threads = 0;
      break;
  }
  resolved.build_threads_specified = true;
  resolved.diskann_max_degree =
      effective_diskann_max_degree(*resolved_provider, requested_options);
  resolved.diskann_build_complexity =
      effective_diskann_build_complexity(*resolved_provider, requested_options);
  resolved.diskann_pq_code_budget_size =
      effective_diskann_pq_code_budget_size(*resolved_provider,
                                            requested_options);
  resolved.diskann_disk_pq_dims =
      effective_diskann_disk_pq_dims(*resolved_provider, requested_options);
  resolved.diskann_accelerate_build =
      effective_diskann_accelerate_build(*resolved_provider,
                                         requested_options);
  resolved.diskann_accelerate_build_specified = true;
  resolved.diskann_shuffle_build =
      effective_diskann_shuffle_build(*resolved_provider, requested_options);
  resolved.diskann_shuffle_build_specified = true;
  resolved.diskann_use_bfs_cache =
      effective_diskann_use_bfs_cache(*resolved_provider, requested_options);
  resolved.diskann_use_bfs_cache_specified = true;
  resolved.diskann_search_complexity =
      effective_diskann_search_complexity(*resolved_provider,
                                          requested_options);
  resolved.diskann_search_beamwidth =
      effective_diskann_search_beamwidth(*resolved_provider,
                                         requested_options);
  if (!resolved.consistency_mode_specified) {
    resolved.consistency_mode = vector_index::global_index_consistency_mode();
  }
  resolved.consistency_mode_specified = true;
  resolved.defaults_resolved = true;
  *resolved_options = std::move(resolved);
  return true;
}

bool drop_cleanup_allows_create_locked(const std::string &index_name) {
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (!truth_store->supports_publication_intents()) return true;

  std::vector<vector_index_truth_store::publication_intent> intents;
  if (!truth_store->load_publication_intents(&intents)) return false;
  return std::none_of(
      intents.begin(), intents.end(), [&](const auto &intent) {
        return intent.index_name == index_name &&
               intent.operation == vector_index_truth_store::
                                       publication_operation::kDropIndex;
      });
}

bool create_index_locked(
    const std::string &index_name, size_t dimension, const std::string &metric,
    const std::string &mode, const std::string &provider,
    const index_binding *binding, const std::string &owner_schema,
    const vector_index_registry::create_index_options &options) {
  if (!drop_cleanup_allows_create_locked(index_name)) return false;
  std::vector<vector_index_metadata_store::metadata_row> metadata_before;
  vector_index::index_service::committed_state committed_before;
  std::vector<vector_index_metadata_store::change_log_row> change_log_before;
  std::vector<std::string> lagging_indexes_before;
  if (!snapshot_runtime_state_locked(&metadata_before, &committed_before,
                                     &change_log_before,
                                     &lagging_indexes_before)) {
    return false;
  }
  if (owner_schema.empty()) return false;
  if (binding != nullptr && options.consistency_mode_specified &&
      options.consistency_mode ==
          vector_index::index_consistency_mode::kStandalone) {
    return false;
  }
  const bool ok = g_index_service.register_index_from_strings(
      index_name, dimension, metric, mode, provider);
  if (!ok) return false;
  vector_index::index_service::index_config registered_config;
  bool supports_mutations = false;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  if (!g_index_service.describe_index(index_name, &registered_config,
                                      &supports_mutations, &entry_count,
                                      &committed_entry_count)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  const std::string effective_provider =
      vector_index::backend_provider_to_string(registered_config.provider);
  vector_index::index_consistency_mode consistency_mode =
      vector_index::index_consistency_mode::kTransactional;
  if (binding == nullptr) {
    consistency_mode = options.consistency_mode_specified
                           ? options.consistency_mode
                           : vector_index::global_index_consistency_mode();
  }
  if (!g_index_service.set_index_consistency_mode(index_name,
                                                  consistency_mode)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  if (!options.initial_lifecycle_state.empty() &&
      !g_index_service.set_lifecycle_state(index_name,
                                           options.initial_lifecycle_state)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  const uint32_t hnsw_build_threads =
      effective_hnsw_build_threads(effective_provider, options);
  if (hnsw_build_threads != 0 &&
      !g_index_service.set_hnsw_build_threads(index_name, hnsw_build_threads)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  const uint32_t faiss_build_threads =
      effective_faiss_build_threads(effective_provider, options);
  if (faiss_build_threads != 0 &&
      !g_index_service.set_faiss_build_threads(index_name,
                                               faiss_build_threads)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  const uint32_t diskann_build_threads =
      effective_diskann_build_threads(effective_provider, options);
  const uint32_t diskann_max_degree =
      effective_diskann_max_degree(effective_provider, options);
  const uint32_t diskann_build_complexity =
      effective_diskann_build_complexity(effective_provider, options);
  const bool diskann_build_params_applied =
      diskann_max_degree != 0 && diskann_build_complexity != 0;
  if (diskann_build_params_applied &&
      !g_index_service.set_diskann_build_params(
          index_name, diskann_max_degree, diskann_build_complexity,
          diskann_build_threads)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  if (!diskann_build_params_applied && diskann_build_threads != 0 &&
      !g_index_service.set_diskann_build_threads(index_name,
                                                 diskann_build_threads)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  const uint64_t diskann_pq_code_budget_size =
      effective_diskann_pq_code_budget_size(effective_provider, options);
  if (diskann_pq_code_budget_size != 0 &&
      !g_index_service.set_diskann_pq_code_budget_size(
          index_name, diskann_pq_code_budget_size)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  const uint32_t diskann_disk_pq_dims =
      effective_diskann_disk_pq_dims(effective_provider, options);
  if (diskann_disk_pq_dims != 0 &&
      !g_index_service.set_diskann_disk_pq_dims(index_name,
                                                diskann_disk_pq_dims)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  const bool diskann_accelerate_build =
      effective_diskann_accelerate_build(effective_provider, options);
  if ((diskann_accelerate_build ||
       options.diskann_accelerate_build_specified) &&
      !g_index_service.set_diskann_accelerate_build(
          index_name, diskann_accelerate_build)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  const bool diskann_shuffle_build =
      effective_diskann_shuffle_build(effective_provider, options);
  if ((diskann_shuffle_build || options.diskann_shuffle_build_specified) &&
      !g_index_service.set_diskann_shuffle_build(index_name,
                                                 diskann_shuffle_build)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  const bool diskann_use_bfs_cache =
      effective_diskann_use_bfs_cache(effective_provider, options);
  if ((diskann_use_bfs_cache || options.diskann_use_bfs_cache_specified) &&
      !g_index_service.set_diskann_use_bfs_cache(index_name,
                                                 diskann_use_bfs_cache)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  const uint32_t diskann_search_complexity =
      effective_diskann_search_complexity(effective_provider, options);
  if (diskann_search_complexity != 0 &&
      !g_index_service.set_diskann_search_complexity(
          index_name, diskann_search_complexity)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  const uint32_t diskann_search_beamwidth =
      effective_diskann_search_beamwidth(effective_provider, options);
  if (diskann_search_beamwidth != 0 &&
      !g_index_service.set_diskann_search_beamwidth(index_name,
                                                    diskann_search_beamwidth)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  set_index_binding_and_owner_schema_locked(
      index_name, binding == nullptr ? index_binding{} : *binding,
      owner_schema);
  if (!persist_registry_state_locked()) {
    vector_status::record_truth_store_persist_failure();
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  vector_status::record_index_create_success();
  return true;
}

bool apply_index_tuning_locked(const std::string &index_name,
                               const vector_index_registry::index_info &info) {
  if (info.search_ef != 0 &&
      !g_index_service.set_search_ef(index_name, info.search_ef)) {
    return false;
  }
  if (info.hnsw_build_threads != 0 &&
      !g_index_service.set_hnsw_build_threads(index_name,
                                              info.hnsw_build_threads)) {
    return false;
  }
  if ((info.hnsw_m != 0 || info.hnsw_ef_construction != 0) &&
      !g_index_service.set_hnsw_build_params(index_name, info.hnsw_m,
                                             info.hnsw_ef_construction)) {
    return false;
  }
  if (info.faiss_build_threads != 0 &&
      !g_index_service.set_faiss_build_threads(index_name,
                                               info.faiss_build_threads)) {
    return false;
  }
  if ((info.faiss_nlist != 0 || info.faiss_nprobe != 0) &&
      !g_index_service.set_faiss_ivf_params(index_name, info.faiss_nlist,
                                            info.faiss_nprobe)) {
    return false;
  }
  if ((info.faiss_pq_m != 0 || info.faiss_pq_bits != 0) &&
      !g_index_service.set_faiss_ivf_pq_params(
          index_name, info.faiss_nlist, info.faiss_nprobe, info.faiss_pq_m,
          info.faiss_pq_bits)) {
    return false;
  }
  if ((info.diskann_max_degree != 0 || info.diskann_build_complexity != 0) &&
      !g_index_service.set_diskann_build_params(
          index_name, info.diskann_max_degree, info.diskann_build_complexity,
          info.diskann_build_threads)) {
    return false;
  }
  if (info.diskann_build_threads != 0 &&
      info.diskann_max_degree == 0 && info.diskann_build_complexity == 0 &&
      !g_index_service.set_diskann_build_threads(index_name,
                                                 info.diskann_build_threads)) {
    return false;
  }
  if (info.diskann_build_mode_specified &&
      !g_index_service.set_diskann_build_mode(
          index_name, info.diskann_build_mode_value)) {
    return false;
  }
  if (info.diskann_search_complexity != 0 &&
      !g_index_service.set_diskann_search_complexity(
          index_name, info.diskann_search_complexity)) {
    return false;
  }
  if (info.diskann_search_beamwidth != 0 &&
      !g_index_service.set_diskann_search_beamwidth(
          index_name, info.diskann_search_beamwidth)) {
    return false;
  }
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (vector_index::parse_backend_provider(info.provider, &provider_value) &&
      provider_value == vector_index::backend_provider::kDiskAnn &&
      !g_index_service.set_diskann_pq_code_budget_size(
          index_name, info.diskann_pq_code_budget_size)) {
    return false;
  }
  if (provider_value == vector_index::backend_provider::kDiskAnn &&
      !g_index_service.set_diskann_disk_pq_dims(
          index_name, info.diskann_disk_pq_dims)) {
    return false;
  }
  if (provider_value == vector_index::backend_provider::kDiskAnn &&
      !g_index_service.set_diskann_accelerate_build(
          index_name, info.diskann_accelerate_build)) {
    return false;
  }
  if (provider_value == vector_index::backend_provider::kDiskAnn &&
      !g_index_service.set_diskann_shuffle_build(index_name,
                                                 info.diskann_shuffle_build)) {
    return false;
  }
  if (provider_value == vector_index::backend_provider::kDiskAnn &&
      !g_index_service.set_diskann_use_bfs_cache(index_name,
                                                 info.diskann_use_bfs_cache)) {
    return false;
  }
  return true;
}

vector_index::index_service::index_config config_from_metadata_row(
    const vector_index_metadata_store::metadata_row &row) {
  vector_index::index_service::index_config config;
  config.dimension = row.dimension;
  config.metric = row.metric;
  config.mode = row.mode;
  config.provider = row.provider;
  config.consistency_mode = row.consistency_mode;
  config.search_ef = row.search_ef;
  config.hnsw_m = row.hnsw_m;
  config.hnsw_ef_construction = row.hnsw_ef_construction;
  config.hnsw_build_threads = row.hnsw_build_threads;
  config.faiss_nlist = row.faiss_nlist;
  config.faiss_nprobe = row.faiss_nprobe;
  config.faiss_pq_m = row.faiss_pq_m;
  config.faiss_pq_bits = row.faiss_pq_bits;
  config.faiss_build_threads = row.faiss_build_threads;
  config.diskann_max_degree = row.diskann_max_degree;
  config.diskann_build_complexity = row.diskann_build_complexity;
  config.diskann_build_threads = row.diskann_build_threads;
  config.diskann_build_mode_value = row.diskann_build_mode_value;
  config.diskann_build_mode_specified = row.diskann_build_mode_specified;
  config.diskann_search_complexity = row.diskann_search_complexity;
  config.diskann_search_beamwidth = row.diskann_search_beamwidth;
  config.diskann_pq_code_budget_size = row.diskann_pq_code_budget_size;
  config.diskann_disk_pq_dims = row.diskann_disk_pq_dims;
  config.diskann_accelerate_build = row.diskann_accelerate_build;
  config.diskann_shuffle_build = row.diskann_shuffle_build;
  config.diskann_use_bfs_cache = row.diskann_use_bfs_cache;
  return config;
}

void assign_config_to_metadata_row(
    const vector_index::index_service::index_config &config,
    vector_index_metadata_store::metadata_row &row) {
  row.dimension = config.dimension;
  row.metric = config.metric;
  row.mode = config.mode;
  row.provider = config.provider;
  row.consistency_mode = config.consistency_mode;
  row.search_ef = config.search_ef;
  row.hnsw_m = config.hnsw_m;
  row.hnsw_ef_construction = config.hnsw_ef_construction;
  row.hnsw_build_threads = config.hnsw_build_threads;
  row.faiss_nlist = config.faiss_nlist;
  row.faiss_nprobe = config.faiss_nprobe;
  row.faiss_pq_m = config.faiss_pq_m;
  row.faiss_pq_bits = config.faiss_pq_bits;
  row.faiss_build_threads = config.faiss_build_threads;
  row.diskann_max_degree = config.diskann_max_degree;
  row.diskann_build_complexity = config.diskann_build_complexity;
  row.diskann_build_threads = config.diskann_build_threads;
  row.diskann_build_mode_value = config.diskann_build_mode_value;
  row.diskann_build_mode_specified = config.diskann_build_mode_specified;
  row.diskann_search_complexity = config.diskann_search_complexity;
  row.diskann_search_beamwidth = config.diskann_search_beamwidth;
  row.diskann_pq_code_budget_size = config.diskann_pq_code_budget_size;
  row.diskann_disk_pq_dims = config.diskann_disk_pq_dims;
  row.diskann_accelerate_build = config.diskann_accelerate_build;
  row.diskann_shuffle_build = config.diskann_shuffle_build;
  row.diskann_use_bfs_cache = config.diskann_use_bfs_cache;
}

bool snapshot_metadata_locked(
    std::vector<vector_index_metadata_store::metadata_row> *rows) {
  if (rows == nullptr) return false;

  rows->clear();
  std::vector<std::string> index_names;
  if (!g_index_service.list_indexes(&index_names)) return false;

  rows->reserve(index_names.size());
  for (const std::string &index_name : index_names) {
    vector_index::index_service::index_config config;
    bool supports_mutations = false;
    size_t entry_count = 0;
    size_t committed_entry_count = 0;
    std::string lifecycle_state;
    uint64_t lifecycle_version = 0;
    uint32_t last_error_code = 0;
    uint64_t last_error_ts = 0;
    uint64_t recover_fallback_count = 0;
    uint64_t last_recover_fallback_ts = 0;
    vector_index::index_service::index_publication_state publication;
    if (!g_index_service.describe_index(
            index_name, &config, &supports_mutations, &entry_count,
            &committed_entry_count, &lifecycle_state, &lifecycle_version,
            &last_error_code, &last_error_ts, nullptr, &recover_fallback_count,
            &last_recover_fallback_ts, nullptr, nullptr, nullptr,
            &publication)) {
      return false;
    }

    vector_index_metadata_store::metadata_row row;
    row.index_name = index_name;
    assign_config_to_metadata_row(config, row);
    const index_binding binding = binding_for_index_locked(index_name);
    row.schema_name = binding.schema_name;
    row.table_name = binding.table_name;
    row.column_name = binding.column_name;
    row.doc_id_column_name = binding.doc_id_column_name;
    row.owner_schema = owner_schema_for_index_locked(index_name);
    row.lifecycle_state = lifecycle_state;
    row.lifecycle_version = lifecycle_version;
    row.last_error_code = last_error_code;
    row.last_error_ts = last_error_ts;
    row.recover_fallback_count = recover_fallback_count;
    row.last_recover_fallback_ts = last_recover_fallback_ts;
    row.index_identity = publication.index_identity;
    row.truth_generation = publication.truth_generation;
    row.config_generation = publication.config_generation;
    row.artifact_generation = publication.artifact_generation;
    row.runtime_generation = publication.runtime_generation;
    rows->push_back(std::move(row));
  }

  return true;
}

bool snapshot_committed_rows_locked(
    std::vector<vector_index_metadata_store::committed_row> *rows) {
  if (rows == nullptr) return false;

  rows->clear();
  vector_index::index_service::committed_state state;
  if (!g_index_service.snapshot_committed_state(&state)) return false;

  for (const auto &index_entry : state) {
    for (const auto &doc_entry : index_entry.second) {
      vector_index_metadata_store::committed_row row;
      row.index_name = index_entry.first;
      row.doc_id = doc_entry.first;
      row.vector = doc_entry.second;
      rows->push_back(std::move(row));
    }
  }
  std::sort(rows->begin(), rows->end(),
            [](const vector_index_metadata_store::committed_row &lhs,
               const vector_index_metadata_store::committed_row &rhs) {
              if (lhs.index_name != rhs.index_name)
                return lhs.index_name < rhs.index_name;
              return lhs.doc_id < rhs.doc_id;
            });
  return true;
}

bool snapshot_prepared_rows_for_txn_locked(
    uint64_t txn_id, const XID &xid,
    std::vector<vector_index_metadata_store::prepared_change_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();

  std::vector<vector_index::index_service::pending_change_snapshot>
      pending_changes;
  if (!g_index_service.snapshot_pending_changes(txn_id, &pending_changes))
    return false;

  vector_index_metadata_store::prepared_change_row xid_identity;
  if (txn_id == 0 ||
      !vector_index_metadata_store::set_prepared_xid(xid, &xid_identity)) {
    return false;
  }

  rows->reserve(pending_changes.size());
  for (const auto &change : pending_changes) {
    vector_index_metadata_store::prepared_change_row row = xid_identity;
    row.prepared_in_tc = false;
    row.txn_id = txn_id;
    row.op = change.erase ? vector_index_metadata_store::change_op::kErase
                          : vector_index_metadata_store::change_op::kUpsert;
    row.index_name = change.index_name;
    row.doc_id = change.doc_id;
    row.vector = change.vector;
    rows->push_back(std::move(row));
  }
  return true;
}

bool apply_metadata_rows_locked(
    const std::vector<vector_index_metadata_store::metadata_row> &rows) {
  g_index_service.discard_all_pending_changes();
  g_index_service = vector_index::index_service();
  g_index_bindings.clear();
  g_index_owner_schemas.clear();
  for (const auto &row : rows) {
    if (row.owner_schema.empty()) return false;
    const vector_index::index_service::index_config config =
        config_from_metadata_row(row);
    if (!g_index_service.register_index(row.index_name, config)) {
      return false;
    }
    vector_index::index_service::index_publication_state publication;
    publication.index_identity = row.index_identity;
    publication.truth_generation = row.truth_generation;
    publication.config_generation = row.config_generation;
    publication.artifact_generation = row.artifact_generation;
    publication.runtime_generation = row.runtime_generation;
    if (!g_index_service.restore_publication_state(row.index_name,
                                                   publication)) {
      return false;
    }
    set_index_binding_and_owner_schema_locked(
        row.index_name,
        index_binding{row.schema_name, row.table_name, row.column_name,
                      row.doc_id_column_name},
        row.owner_schema);
    if (!g_index_service.set_lifecycle_info(
            row.index_name, row.lifecycle_state, row.lifecycle_version,
            row.last_error_code, row.last_error_ts, row.recover_fallback_count,
            row.last_recover_fallback_ts)) {
      return false;
    }
  }
  return true;
}

bool apply_committed_rows_locked(
    const std::vector<vector_index_metadata_store::committed_row> &rows) {
  vector_index::index_service::committed_state state;
  for (const auto &row : rows) {
    state[row.index_name][row.doc_id] = row.vector;
  }
  if (!g_index_service.restore_committed_state_for_startup(state)) return false;

  std::vector<std::string> index_names;
  if (!g_index_service.list_indexes(&index_names)) return false;
  for (const std::string &index_name : index_names) {
    vector_index::index_service::index_publication_state publication;
    vector_index::index_service::index_config config;
    bool manifest_present = false;
    if (!g_index_service.describe_publication_state(index_name, &publication) ||
        !g_index_service.describe_index(
            index_name, &config, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, &manifest_present)) {
      return false;
    }
    const bool deferred_diskann_artifact =
        config.provider == vector_index::backend_provider::kDiskAnn &&
        config.mode == vector_index::backend_mode::kExternal;
    publication.runtime_generation =
        deferred_diskann_artifact ? 0 : publication.truth_generation;
    if (!deferred_diskann_artifact) {
      publication.artifact_generation =
          config.mode == vector_index::backend_mode::kExternal &&
                  manifest_present
              ? publication.truth_generation
              : 0;
    }
    if (!g_index_service.restore_publication_state(index_name, publication)) {
      return false;
    }
  }
  return true;
}

bool apply_change_log_rows_locked(
    const std::vector<vector_index_metadata_store::change_log_row> &rows) {
  if (rows.empty()) return true;

  std::vector<vector_index_metadata_store::change_log_row> ordered_rows(rows);
  std::sort(ordered_rows.begin(), ordered_rows.end(),
            [](const vector_index_metadata_store::change_log_row &lhs,
               const vector_index_metadata_store::change_log_row &rhs) {
              if (lhs.index_name != rhs.index_name)
                return lhs.index_name < rhs.index_name;
              if (lhs.index_identity != rhs.index_identity)
                return lhs.index_identity < rhs.index_identity;
              if (lhs.truth_generation != rhs.truth_generation)
                return lhs.truth_generation < rhs.truth_generation;
              if (lhs.sequence != rhs.sequence)
                return lhs.sequence < rhs.sequence;
              if (lhs.doc_id != rhs.doc_id) return lhs.doc_id < rhs.doc_id;
              return static_cast<int>(lhs.op) < static_cast<int>(rhs.op);
            });

  vector_index::index_service::committed_state state;
  if (!g_index_service.snapshot_committed_state(&state)) return false;
  for (const auto &row : ordered_rows) {
    if (row.sequence == 0 || row.index_name.empty() ||
        row.index_identity == 0 || row.publication_id == 0 ||
        row.truth_generation == 0) {
      return false;
    }
    vector_index::index_service::index_publication_state publication;
    if (!g_index_service.describe_publication_state(row.index_name,
                                                    &publication)) {
      continue;
    }
    if (row.index_identity != publication.index_identity) {
      continue;
    }
    auto index_it = state.find(row.index_name);
    if (index_it == state.end()) continue;
    if (row.op == vector_index_metadata_store::change_op::kUpsert) {
      index_it->second[row.doc_id] = row.vector;
    } else if (row.op == vector_index_metadata_store::change_op::kErase) {
      index_it->second.erase(row.doc_id);
    } else {
      return false;
    }
  }
  if (!g_index_service.restore_committed_state_for_startup(state)) return false;

  std::unordered_map<std::string, uint64_t> truth_generations;
  for (const auto &row : ordered_rows) {
    vector_index::index_service::index_publication_state publication;
    if (!g_index_service.describe_publication_state(row.index_name,
                                                    &publication) ||
        row.index_identity != publication.index_identity) {
      continue;
    }
    uint64_t &generation = truth_generations[row.index_name];
    generation = std::max(generation, row.truth_generation);
  }
  for (const auto &entry : truth_generations) {
    vector_index::index_service::index_publication_state publication;
    vector_index::index_service::index_config config;
    bool manifest_present = false;
    if (!g_index_service.describe_publication_state(entry.first,
                                                    &publication) ||
        !g_index_service.describe_index(
            entry.first, &config, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, &manifest_present)) {
      return false;
    }
    publication.truth_generation =
        std::max(publication.truth_generation, entry.second);
    const bool deferred_diskann_artifact =
        config.provider == vector_index::backend_provider::kDiskAnn &&
        config.mode == vector_index::backend_mode::kExternal;
    publication.runtime_generation =
        deferred_diskann_artifact ? 0 : publication.truth_generation;
    if (!deferred_diskann_artifact) {
      publication.artifact_generation =
          config.mode == vector_index::backend_mode::kExternal &&
                  manifest_present
              ? publication.truth_generation
              : 0;
    }
    if (!g_index_service.restore_publication_state(entry.first, publication)) {
      return false;
    }
  }
  return true;
}

void refresh_committed_snapshot_rows_locked() {
  vector_index::index_service::committed_state state;
  if (!g_index_service.snapshot_committed_state(&state)) return;

  uint64_t total_rows = 0;
  for (const auto &index_entry : state) {
    total_rows += static_cast<uint64_t>(index_entry.second.size());
  }
  vector_status::set_committed_snapshot_rows(total_rows);
}

void refresh_manifest_status_locked() {
  vector_status::set_manifest_version(g_manifest_version);
  vector_status::set_manifest_metadata_checkpoint(
      g_manifest_metadata_checkpoint);
  vector_status::set_manifest_committed_checkpoint(
      g_manifest_committed_checkpoint);
  vector_status::set_manifest_change_log_checkpoint(
      g_manifest_change_log_checkpoint);
}

bool allocate_pending_change_log_delta_locked(
    uint64_t txn_id,
    const std::vector<vector_index::index_service::pending_change_snapshot>
        &changes,
    std::vector<vector_index_metadata_store::change_log_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();
  if (changes.empty()) return true;
  const uint64_t max_sequence = std::numeric_limits<uint64_t>::max();
  if (changes.size() > max_sequence - g_next_change_log_sequence) {
    return false;
  }

  try {
    std::vector<vector_index::index_service::pending_change_snapshot> ordered(
        changes);
    std::sort(ordered.begin(), ordered.end(),
              [](const auto &lhs, const auto &rhs) {
                if (lhs.index_name != rhs.index_name)
                  return lhs.index_name < rhs.index_name;
                if (lhs.doc_id != rhs.doc_id) return lhs.doc_id < rhs.doc_id;
                return lhs.erase < rhs.erase;
              });

    std::vector<vector_index_metadata_store::change_log_row> allocated_rows;
    allocated_rows.reserve(ordered.size());
    uint64_t sequence = g_next_change_log_sequence;
    for (const auto &change : ordered) {
      vector_index_metadata_store::change_log_row row;
      row.sequence = sequence++;
      row.txn_id = txn_id;
      row.op = change.erase ? vector_index_metadata_store::change_op::kErase
                            : vector_index_metadata_store::change_op::kUpsert;
      row.index_name = change.index_name;
      vector_index::index_service::index_publication_state publication;
      if (!g_index_service.describe_publication_state(change.index_name,
                                                      &publication) ||
          publication.index_identity == 0) {
        return false;
      }
      row.index_identity = publication.index_identity;
      row.doc_id = change.doc_id;
      row.vector = change.vector;
      allocated_rows.push_back(std::move(row));
    }
    *rows = std::move(allocated_rows);
    g_next_change_log_sequence = sequence;
  } catch (...) {
    return false;
  }
  return true;
}

bool append_pending_change_log_delta_locked(
    uint64_t txn_id,
    const std::vector<vector_index::index_service::pending_change_snapshot>
        &changes) {
  std::vector<vector_index_metadata_store::change_log_row> delta;
  if (!allocate_pending_change_log_delta_locked(txn_id, changes, &delta)) {
    return false;
  }
  for (auto &row : delta) {
    vector_index::index_service::index_publication_state publication;
    if (!g_index_service.describe_publication_state(row.index_name,
                                                    &publication) ||
        publication.index_identity != row.index_identity ||
        publication.truth_generation == 0) {
      return false;
    }
    row.publication_id = publication.truth_generation;
    row.truth_generation = publication.truth_generation;
  }
  try {
    g_change_log_rows.insert(g_change_log_rows.end(),
                             std::make_move_iterator(delta.begin()),
                             std::make_move_iterator(delta.end()));
  } catch (...) {
    return false;
  }
  g_manifest_change_log_checkpoint = g_change_log_rows.size();
  return true;
}

void prune_change_log_for_index_locked(const std::string &index_name) {
  g_change_log_rows.erase(
      std::remove_if(
          g_change_log_rows.begin(), g_change_log_rows.end(),
          [&index_name](
              const vector_index_metadata_store::change_log_row &row) {
            return row.index_name == index_name;
          }),
      g_change_log_rows.end());
  g_manifest_change_log_checkpoint = g_change_log_rows.size();
}

bool capture_runtime_state_locked(runtime_state_snapshot *snapshot) {
  if (snapshot == nullptr) return false;
  return snapshot_runtime_state_locked(
      &snapshot->metadata_rows, &snapshot->committed_state,
      &snapshot->change_log_rows, &snapshot->lagging_index_names);
}

bool rollback_runtime_state_locked(const runtime_state_snapshot &snapshot) {
  return rollback_runtime_state_locked(
      snapshot.metadata_rows, snapshot.committed_state,
      snapshot.change_log_rows, snapshot.lagging_index_names);
}

bool evict_committed_cache_to_budget_locked() {
  return g_index_service.evict_committed_cache_to_budget();
}

bool persist_index_config_manifest_locked(
    const std::string &index_name,
    const vector_index::index_service::index_config &config,
    const vector_index::index_service::index_publication_state &publication) {
  vector_status::record_truth_store_persist_request();
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (!truth_store->begin_persist()) return false;

  std::vector<vector_index_metadata_store::metadata_row> rows;
  if (!snapshot_metadata_locked(&rows)) {
    truth_store->rollback_persist();
    return false;
  }

  bool found = false;
  for (auto &row : rows) {
    if (row.index_name != index_name) continue;
    assign_config_to_metadata_row(config, row);
    row.index_identity = publication.index_identity;
    row.truth_generation = publication.truth_generation;
    row.config_generation = publication.config_generation;
    row.artifact_generation = publication.artifact_generation;
    row.runtime_generation = publication.runtime_generation;
    found = true;
    break;
  }
  if (!found) {
    truth_store->rollback_persist();
    return false;
  }

  const uint64_t metadata_checkpoint_before = g_manifest_metadata_checkpoint;
  g_manifest_metadata_checkpoint = rows.size();
  const bool ok =
      truth_store->save_metadata(rows) && persist_manifest_locked();
  if (!ok) {
    g_manifest_metadata_checkpoint = metadata_checkpoint_before;
    truth_store->rollback_persist();
    vector_status::record_metadata_persist_failure();
    return false;
  }
  if (!truth_store->commit_persist()) {
    g_manifest_metadata_checkpoint = metadata_checkpoint_before;
    truth_store->rollback_persist();
    vector_status::record_metadata_persist_failure();
    vector_status::record_manifest_persist_failure();
    return false;
  }
  return true;
}

bool metadata_rows_have_recovery_source(
    const std::vector<vector_index_metadata_store::metadata_row> &rows) {
  return std::all_of(
      rows.begin(), rows.end(), [](const auto &row) {
        return row.consistency_mode ==
                   vector_index::index_consistency_mode::kTransactional &&
               !row.schema_name.empty() && !row.table_name.empty() &&
               !row.column_name.empty() && !row.doc_id_column_name.empty();
      });
}

bool require_truth_recovery_locked(
    vector_index_truth_store::truth_store *truth_store,
    const std::vector<vector_index_metadata_store::metadata_row> &rows,
    const std::string &artifact_name, const std::string &reason,
    uint64_t generation) {
  if (!metadata_rows_have_recovery_source(rows)) {
    return fail_stop_truth_artifact_locked(truth_store, artifact_name, reason,
                                           generation);
  }

  std::string quarantine_identity;
  if (truth_store == nullptr ||
      !truth_store->stage_quarantine(artifact_name, reason, generation,
                                     &quarantine_identity)) {
    g_registry_health = registry_health_state::kFailed;
    g_registry_failure_reason = reason + ":quarantine_copy_failed";
    return false;
  }

  g_truth_recovery_state.quarantine_identities[artifact_name] =
      std::move(quarantine_identity);
  for (const auto &row : rows) {
    g_truth_recovery_state.pending_index_names.insert(row.index_name);
  }
  g_registry_health = registry_health_state::kRecoveryRequired;
  g_registry_failure_reason = "truth_projection_recovery_required";
  return true;
}

void mark_metadata_recovery_required(
    std::vector<vector_index_metadata_store::metadata_row> *rows) {
  if (rows == nullptr) return;
  for (auto &row : *rows) {
    row.lifecycle_state = kLifecycleRecoveryRequired;
    ++row.lifecycle_version;
  }
}

bool ensure_metadata_available_locked() {
  // A pre-commit search can load runtime before a detached XA is resolved.
  // Keep the after-commit observer available so that resolution catches up the
  // durable outbox even when this process has no original THD transaction state.
  (void)vector_trx_participant::ensure_observer_registered();
  if (g_registry_health == registry_health_state::kFailed) return false;
  if (g_metadata_loaded) return true;

  if (!g_pending_spill_startup_cleanup_complete) {
    if (!g_index_service.cleanup_orphaned_pending_spills()) {
      g_registry_health = registry_health_state::kFailed;
      g_registry_failure_reason = "pending_spill_cleanup_failed";
      return false;
    }
    g_pending_spill_startup_cleanup_complete = true;
  }

  g_truth_recovery_state = truth_recovery_state{};

  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  bool manifest_checkpoint_mismatch = false;
  bool committed_checkpoint_mismatch = false;
  bool change_log_checkpoint_mismatch = false;

  vector_index_metadata_store::manifest_row manifest_row;
  if (!truth_store->load_manifest(&manifest_row)) {
    vector_status::record_metadata_load_failure();
    vector_status::record_manifest_load_failure();
    return fail_stop_truth_artifact_locked(
        truth_store, "manifest", "manifest_load_failed", 0);
  }
  g_manifest_version = manifest_row.version == 0 ? 1 : manifest_row.version;
  g_manifest_metadata_checkpoint = manifest_row.metadata_checkpoint;
  g_manifest_committed_checkpoint = manifest_row.committed_checkpoint;
  g_manifest_change_log_checkpoint = manifest_row.change_log_checkpoint;
  g_next_change_log_sequence = manifest_row.next_change_log_sequence;

  std::vector<vector_index_metadata_store::metadata_row> rows;
  if (!truth_store->load_metadata(&rows)) {
    vector_status::record_metadata_load_failure();
    return fail_stop_truth_artifact_locked(
        truth_store, "metadata", "metadata_load_failed",
        g_manifest_metadata_checkpoint);
  }
  const std::unordered_set<std::string> incomplete_create_indexes =
      remove_incomplete_create_metadata_rows(&rows);
  const bool cleanup_incomplete_create_state =
      !incomplete_create_indexes.empty();
  if (g_manifest_metadata_checkpoint > rows.size()) {
    manifest_checkpoint_mismatch = true;
    g_manifest_metadata_checkpoint = rows.size();
  }
  if (!apply_metadata_rows_locked(rows)) {
    return fail_stop_truth_artifact_locked(
        truth_store, "metadata", "metadata_validation_failed",
        g_manifest_metadata_checkpoint);
  }
  if (!g_index_service.restore_next_index_identity(
          std::max(manifest_row.next_index_identity,
                   g_index_service.next_index_identity()))) {
    return fail_stop_truth_artifact_locked(truth_store, "manifest",
                                           "index_identity_high_water_invalid",
                                           g_manifest_metadata_checkpoint);
  }

  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  bool truth_recovery_required = false;
  if (!truth_store->load_committed(&committed_rows)) {
    vector_status::record_committed_load_failure();
    if (!require_truth_recovery_locked(
            truth_store, rows, "committed", "committed_load_failed",
            g_manifest_committed_checkpoint)) {
      return false;
    }
    truth_recovery_required = true;
  }
  remove_committed_rows_for_indexes(incomplete_create_indexes, &committed_rows);
  if (!truth_recovery_required &&
      g_manifest_committed_checkpoint > committed_rows.size()) {
    manifest_checkpoint_mismatch = true;
    committed_checkpoint_mismatch = true;
    g_manifest_committed_checkpoint = committed_rows.size();
  }
  if (!truth_recovery_required &&
      !apply_committed_rows_locked(committed_rows)) {
    vector_status::record_committed_load_failure();
    if (!require_truth_recovery_locked(
            truth_store, rows, "committed", "committed_validation_failed",
            g_manifest_committed_checkpoint)) {
      return false;
    }
    truth_recovery_required = true;
  }

  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  if (!truth_store->load_change_log(&change_log_rows)) {
    vector_status::record_metadata_load_failure();
    vector_status::record_change_log_load_failure();
    if (!require_truth_recovery_locked(
            truth_store, rows, "changelog", "changelog_load_failed",
            g_manifest_change_log_checkpoint)) {
      return false;
    }
    truth_recovery_required = true;
  }

  remove_change_log_rows_for_indexes(incomplete_create_indexes,
                                     &change_log_rows);
  if (!truth_recovery_required) g_change_log_rows = std::move(change_log_rows);
  if (!truth_recovery_required &&
      g_manifest_change_log_checkpoint > g_change_log_rows.size()) {
    manifest_checkpoint_mismatch = true;
    change_log_checkpoint_mismatch = true;
    g_manifest_change_log_checkpoint = g_change_log_rows.size();
  }
  if (manifest_checkpoint_mismatch) {
    vector_status::record_metadata_load_failure();
    vector_status::record_manifest_load_failure();
    if (committed_checkpoint_mismatch) {
      vector_status::record_committed_load_failure();
    }
    if (change_log_checkpoint_mismatch) {
      vector_status::record_change_log_load_failure();
    }
  }
  if (!truth_recovery_required &&
      !apply_change_log_rows_locked(g_change_log_rows)) {
    vector_status::record_metadata_load_failure();
    vector_status::record_change_log_load_failure();
    vector_status::record_change_log_replay_failure();
    if (!require_truth_recovery_locked(
            truth_store, rows, "changelog", "changelog_replay_failed",
            g_manifest_change_log_checkpoint)) {
      return false;
    }
    truth_recovery_required = true;
  }
  if (truth_recovery_required) {
    mark_metadata_recovery_required(&rows);
    if (!apply_metadata_rows_locked(rows) || !apply_committed_rows_locked({})) {
      return fail_stop_truth_artifact_locked(
          truth_store, "metadata", "recovery_state_install_failed",
          g_manifest_metadata_checkpoint);
    }
    g_change_log_rows.clear();
  }
  g_next_change_log_sequence =
      std::max<uint64_t>(1, manifest_row.next_change_log_sequence);
  for (const auto &row : g_change_log_rows) {
    if (row.sequence >= g_next_change_log_sequence) {
      g_next_change_log_sequence = row.sequence + 1;
    }
  }
  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  if (!truth_store->load_prepared(&prepared_rows)) {
    vector_status::record_metadata_load_failure();
    return fail_stop_truth_artifact_locked(
        truth_store, "prepared", "prepared_load_failed",
        g_manifest_change_log_checkpoint);
  } else if (!prepared_rows_are_valid(prepared_rows)) {
    vector_status::record_metadata_load_failure();
    return fail_stop_truth_artifact_locked(
        truth_store, "prepared", "prepared_validation_failed",
        g_manifest_change_log_checkpoint);
  }
  if (truth_recovery_required && !prepared_rows.empty()) {
    g_registry_health = registry_health_state::kFailed;
    g_registry_failure_reason = "truth_recovery_has_prepared_rows";
    return false;
  }
  remove_prepared_rows_for_indexes(incomplete_create_indexes, &prepared_rows);
  g_prepared_change_rows = std::move(prepared_rows);
  for (const auto &row : g_change_log_rows) {
    advance_txn_id_high_water(row.txn_id);
  }
  for (const auto &row : g_prepared_change_rows) {
    advance_txn_id_high_water(row.txn_id);
  }

  std::vector<vector_index_metadata_store::segment_task_row> segment_task_rows;
  if (!truth_store->load_segment_tasks(&segment_task_rows)) {
    vector_status::record_metadata_load_failure();
    if (!truth_store->quarantine_segment_tasks()) return false;
    segment_task_rows.clear();
  }
  remove_segment_task_rows_for_indexes(incomplete_create_indexes,
                                       &segment_task_rows);
  const bool segment_task_recovery_changed =
      truth_recovery_required
          ? !segment_task_rows.empty()
          : normalize_segment_tasks_for_recovery(rows, &segment_task_rows);
  if (truth_recovery_required) segment_task_rows.clear();
  g_segment_task_rows = std::move(segment_task_rows);

  if (truth_recovery_required && !g_recovery_actions.empty()) {
    g_registry_health = registry_health_state::kFailed;
    g_registry_failure_reason = "truth_recovery_has_pending_xa_actions";
    return false;
  }
  if (mysqld_server_started && !g_recovery_actions.empty()) {
    const auto actions = g_recovery_actions;
    g_recovery_actions.clear();
    for (const auto &action : actions) {
      xa_status_code rc = XA_OK;
      switch (action.type) {
        case recovery_action_type::kCommit:
          rc = apply_prepared_xid_locked(action.xid, true, 0);
          break;
        case recovery_action_type::kRollback:
          rc = apply_prepared_xid_locked(action.xid, false, 0);
          break;
        case recovery_action_type::kPreparedInTc:
          rc = XAER_NOTA;
          for (auto &row : g_prepared_change_rows) {
            if (xid_matches_row(action.xid, row)) {
              row.prepared_in_tc = true;
              rc = XA_OK;
            }
          }
          if (rc == XA_OK && !persist_prepared_locked()) rc = XAER_RMERR;
          break;
      }
      if (rc == XAER_NOTA && action.conditional) continue;
      if (rc != XA_OK) {
        g_recovery_actions = actions;
        return false;
      }
    }
  }

  if (!truth_recovery_required) {
    g_manifest_metadata_checkpoint = rows.size();
    std::vector<vector_index_metadata_store::committed_row>
        effective_committed_rows;
    if (!snapshot_committed_rows_locked(&effective_committed_rows)) return false;
    g_manifest_committed_checkpoint = effective_committed_rows.size();
    g_manifest_change_log_checkpoint = g_change_log_rows.size();
  }

  vector_status::set_registered_indexes(rows.size());
  refresh_committed_snapshot_rows_locked();
  refresh_manifest_status_locked();
  if (!evict_committed_cache_to_budget_locked()) return false;
  if (!truth_recovery_required) {
    g_registry_health = registry_health_state::kReady;
    g_registry_failure_reason.clear();
  }
  g_metadata_loaded = true;
  if (!truth_recovery_required &&
      (cleanup_incomplete_create_state || segment_task_recovery_changed) &&
      !persist_registry_state_locked()) {
    vector_status::record_truth_store_persist_failure();
    g_metadata_loaded = false;
    return false;
  }
  return true;
}

bool ensure_metadata_loaded_locked() {
  return ensure_metadata_available_locked() &&
         g_registry_health == registry_health_state::kReady;
}

bool fail_stop_truth_artifact_locked(
    vector_index_truth_store::truth_store *truth_store,
    const std::string &artifact_name, const std::string &reason,
    uint64_t generation) {
  std::string quarantine_identity;
  const bool evidence_saved =
      truth_store != nullptr &&
      truth_store->stage_quarantine(artifact_name, reason, generation,
                                    &quarantine_identity);
  g_registry_health = registry_health_state::kFailed;
  g_registry_failure_reason = evidence_saved
                                  ? reason
                                  : reason + ":quarantine_copy_failed";
  return false;
}

bool persist_metadata_locked(size_t *row_count) {
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  std::vector<vector_index_metadata_store::metadata_row> rows;
  if (!snapshot_metadata_locked(&rows)) return false;
  if (row_count != nullptr) *row_count = rows.size();
  if (!truth_store->save_metadata(rows)) {
    vector_status::record_metadata_persist_failure();
    return false;
  }
  g_manifest_metadata_checkpoint = rows.size();
  return true;
}

bool persist_committed_locked(size_t *row_count) {
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  std::vector<vector_index_metadata_store::committed_row> rows;
  if (!snapshot_committed_rows_locked(&rows)) return false;
  if (row_count != nullptr) *row_count = rows.size();
  if (!truth_store->save_committed(rows)) {
    vector_status::record_committed_persist_failure();
    return false;
  }
  vector_status::set_committed_snapshot_rows(rows.size());
  g_manifest_committed_checkpoint = rows.size();
  return true;
}

bool persist_change_log_locked() {
  if (!vector_index_truth_store::get()->save_change_log(g_change_log_rows)) {
    vector_status::record_metadata_persist_failure();
    vector_status::record_change_log_persist_failure();
    return false;
  }
  g_manifest_change_log_checkpoint = g_change_log_rows.size();
  return true;
}

bool persist_committed_delta_locked(
    const std::vector<vector_index_metadata_store::change_log_row> &rows) {
  if (!vector_index_truth_store::get()->apply_committed_delta(rows)) {
    vector_status::record_committed_persist_failure();
    return false;
  }
  const size_t row_count = g_index_service.committed_entry_count();
  vector_status::set_committed_snapshot_rows(row_count);
  g_manifest_committed_checkpoint = row_count;
  return true;
}

bool persist_change_log_delta_locked(
    const std::vector<vector_index_metadata_store::change_log_row> &rows) {
  if (!vector_index_truth_store::get()->append_change_log_delta(rows)) {
    vector_status::record_metadata_persist_failure();
    vector_status::record_change_log_persist_failure();
    return false;
  }
  g_manifest_change_log_checkpoint = g_change_log_rows.size();
  return true;
}

bool persist_prepared_locked() {
  if (!vector_index_truth_store::get()->save_prepared(g_prepared_change_rows)) {
    vector_status::record_truth_store_persist_failure();
    return false;
  }
  return true;
}

bool persist_segment_tasks_locked() {
  if (!vector_index_truth_store::get()->save_segment_tasks(
          g_segment_task_rows)) {
    vector_status::record_truth_store_persist_failure();
    return false;
  }
  return true;
}

bool persist_manifest_locked() {
  if (g_manifest_version >= std::numeric_limits<uint64_t>::max() - 1) {
    vector_status::record_metadata_persist_failure();
    vector_status::record_manifest_persist_failure();
    return false;
  }
  vector_index_metadata_store::manifest_row row;
  row.state = "ready";
  row.version = g_manifest_version + 1;
  row.metadata_checkpoint = g_manifest_metadata_checkpoint;
  row.committed_checkpoint = g_manifest_committed_checkpoint;
  row.change_log_checkpoint = g_manifest_change_log_checkpoint;
  row.next_index_identity = g_index_service.next_index_identity();
  row.next_change_log_sequence = g_next_change_log_sequence;
  if (!vector_index_truth_store::get()->save_manifest(row)) {
    vector_status::record_metadata_persist_failure();
    vector_status::record_manifest_persist_failure();
    return false;
  }
  g_manifest_version = row.version;
  refresh_manifest_status_locked();
  return true;
}

bool load_persisted_commit_artifacts_snapshot_locked(
    persisted_commit_artifacts_snapshot *snapshot) {
  if (snapshot == nullptr) return false;
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (!truth_store->load_committed(&snapshot->committed_rows)) {
    return false;
  }
  if (!truth_store->load_change_log(&snapshot->change_log_rows)) {
    return false;
  }
  if (!truth_store->load_prepared(&snapshot->prepared_rows)) {
    return false;
  }
  if (!truth_store->load_segment_tasks(&snapshot->segment_task_rows)) {
    return false;
  }
  if (!truth_store->load_manifest(&snapshot->manifest_row)) {
    return false;
  }
  return true;
}

bool restore_persisted_commit_artifacts_snapshot_locked(
    const persisted_commit_artifacts_snapshot &snapshot) {
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (!truth_store->begin_persist()) return false;
  bool ok = truth_store->save_committed(snapshot.committed_rows) &&
            truth_store->save_change_log(snapshot.change_log_rows) &&
            truth_store->save_prepared(snapshot.prepared_rows) &&
            truth_store->save_segment_tasks(snapshot.segment_task_rows) &&
            truth_store->save_manifest(snapshot.manifest_row);
  if (!ok) {
    truth_store->rollback_persist();
    return false;
  }
  if (!truth_store->commit_persist()) {
    truth_store->rollback_persist();
    return false;
  }

  g_manifest_version =
      snapshot.manifest_row.version == 0 ? 1 : snapshot.manifest_row.version;
  g_manifest_metadata_checkpoint = snapshot.manifest_row.metadata_checkpoint;
  g_manifest_committed_checkpoint = snapshot.manifest_row.committed_checkpoint;
  g_manifest_change_log_checkpoint =
      snapshot.manifest_row.change_log_checkpoint;
  g_next_change_log_sequence = snapshot.manifest_row.next_change_log_sequence;
  vector_status::set_committed_snapshot_rows(snapshot.committed_rows.size());
  refresh_manifest_status_locked();
  return true;
}

bool persist_registry_state_locked() {
  vector_status::record_truth_store_persist_request();
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (!truth_store->begin_persist()) return false;
  bool ok = persist_metadata_locked(nullptr) &&
            persist_committed_locked(nullptr) && persist_change_log_locked() &&
            persist_prepared_locked() && persist_segment_tasks_locked() &&
            persist_manifest_locked();
  if (!ok) {
    truth_store->rollback_persist();
    return false;
  }
  if (!truth_store->commit_persist()) {
    truth_store->rollback_persist();
    vector_status::record_metadata_persist_failure();
    vector_status::record_manifest_persist_failure();
    return false;
  }
  return true;
}

bool persist_commit_artifacts_impl_locked(
    const std::vector<vector_index_metadata_store::change_log_row>
        *commit_delta_rows,
    bool include_prepared, bool delta_already_staged) {
  const bool diagnostics_enabled = vector_index_diagnostics::enabled();
  const auto total_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  const bool use_delta_path =
      truth_store->is_transactional() && truth_store->supports_delta_persist();
  if (delta_already_staged && !use_delta_path) {
    truth_store->rollback_persist();
    vector_status::record_persist_artifact_rollback();
    vector_status::record_truth_store_persist_failure();
    vector_status::record_truth_store_delta_persist_failure();
    return false;
  }
  const bool persist_delta = use_delta_path && commit_delta_rows != nullptr;
  persisted_commit_artifacts_snapshot persisted_before;
  uint64_t load_snapshot_ms = 0;
  if (!delta_already_staged && !persist_delta) {
    const auto load_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    if (!load_persisted_commit_artifacts_snapshot_locked(&persisted_before)) {
      vector_status::record_truth_store_persist_failure();
      if (diagnostics_enabled) {
        vector_index_diagnostics::record_event(
            "vector_persist_commit_artifacts", {{"result", "load_failed"}},
            {{"load_snapshot_ms", vector_index_diagnostics::elapsed_ms_if(
                                      diagnostics_enabled, load_start)},
             {"total_ms", vector_index_diagnostics::elapsed_ms_if(
                              diagnostics_enabled, total_start)},
             {"ok", 0}});
      }
      return false;
    }
    load_snapshot_ms = vector_index_diagnostics::elapsed_ms_if(
        diagnostics_enabled, load_start);
  }

  if (!delta_already_staged) {
    vector_status::record_truth_store_persist_request();
    if (persist_delta)
      vector_status::record_truth_store_delta_persist_request();
  }
  const auto begin_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  if (!delta_already_staged && !truth_store->begin_persist()) {
    vector_status::record_truth_store_persist_failure();
    if (persist_delta)
      vector_status::record_truth_store_delta_persist_failure();
    if (diagnostics_enabled) {
      vector_index_diagnostics::record_event(
          "vector_persist_commit_artifacts", {{"result", "begin_failed"}},
          {{"load_snapshot_ms", load_snapshot_ms},
           {"begin_persist_ms", vector_index_diagnostics::elapsed_ms_if(
                                    diagnostics_enabled, begin_start)},
           {"total_ms", vector_index_diagnostics::elapsed_ms_if(
                            diagnostics_enabled, total_start)},
           {"ok", 0}});
    }
    return false;
  }
  const uint64_t begin_persist_ms =
      delta_already_staged
          ? 0
          : vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                     begin_start);
  const auto committed_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  bool committed_ok = delta_already_staged;
  if (delta_already_staged) {
    const size_t row_count = g_index_service.committed_entry_count();
    vector_status::set_committed_snapshot_rows(row_count);
    g_manifest_committed_checkpoint = row_count;
  } else {
    committed_ok = persist_delta
                       ? persist_committed_delta_locked(*commit_delta_rows)
                       : persist_committed_locked(nullptr);
  }
  const uint64_t committed_ms = vector_index_diagnostics::elapsed_ms_if(
      diagnostics_enabled, committed_start);
  uint64_t changelog_ms = 0;
  uint64_t prepared_ms = 0;
  uint64_t manifest_ms = 0;
  uint64_t commit_persist_ms = 0;
  const auto changelog_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  bool changelog_ok = delta_already_staged;
  if (delta_already_staged) {
    g_manifest_change_log_checkpoint = g_change_log_rows.size();
  } else if (committed_ok) {
    changelog_ok = persist_delta
                       ? persist_change_log_delta_locked(*commit_delta_rows)
                       : persist_change_log_locked();
  }
  if (committed_ok)
    changelog_ms = vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                           changelog_start);
  bool compact_ok = true;
  uint64_t compact_ms = 0;
  // Staged explicit commits already hold publication ownership here. Do not
  // perform changelog table rewrites in that lock layer: attached DML can hold
  // the same hidden rows while waiting to publish its runtime state.
  if (!delta_already_staged && changelog_ok &&
      change_log_compaction_needed_locked()) {
    const auto compact_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    compact_ok = compact_change_log_locked();
    compact_ms = vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                         compact_start);
  }
  bool prepared_ok = false;
  const auto prepared_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  if (changelog_ok && compact_ok) {
    prepared_ok = !include_prepared || persist_prepared_locked();
    if (include_prepared)
      prepared_ms = vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                            prepared_start);
  }
  const auto manifest_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  const bool manifest_ok = prepared_ok ? persist_manifest_locked() : false;
  if (prepared_ok)
    manifest_ms = vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                          manifest_start);
  bool ok =
      committed_ok && changelog_ok && compact_ok && prepared_ok && manifest_ok;
  if (ok) {
    const auto commit_persist_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    ok = truth_store->commit_persist();
    commit_persist_ms = vector_index_diagnostics::elapsed_ms_if(
        diagnostics_enabled, commit_persist_start);
  }
  if (!ok) {
    truth_store->rollback_persist();
    if (persist_delta || delta_already_staged) {
      vector_status::record_persist_artifact_rollback();
    } else if (restore_persisted_commit_artifacts_snapshot_locked(
                   persisted_before)) {
      vector_status::record_persist_artifact_rollback();
    } else {
      vector_status::record_persist_artifact_rollback_failure();
      fail_stop_registry_locked("persisted_commit_artifacts_restore_failed");
    }
    vector_status::record_truth_store_persist_failure();
    if (persist_delta || delta_already_staged)
      vector_status::record_truth_store_delta_persist_failure();
    if (diagnostics_enabled) {
      vector_index_diagnostics::record_event(
          "vector_persist_commit_artifacts", {{"result", "persist_failed"}},
          {{"load_snapshot_ms", load_snapshot_ms},
           {"begin_persist_ms", begin_persist_ms},
           {"committed_ms", committed_ms},
           {"changelog_ms", changelog_ms},
           {"compact_ms", compact_ms},
           {"prepared_ms", prepared_ms},
           {"manifest_ms", manifest_ms},
           {"commit_persist_ms", commit_persist_ms},
           {"total_ms", vector_index_diagnostics::elapsed_ms_if(
                            diagnostics_enabled, total_start)},
           {"committed_ok", committed_ok ? 1U : 0U},
           {"changelog_ok", changelog_ok ? 1U : 0U},
           {"compact_ok", compact_ok ? 1U : 0U},
           {"prepared_ok", prepared_ok ? 1U : 0U},
           {"manifest_ok", manifest_ok ? 1U : 0U},
           {"delta", (persist_delta || delta_already_staged) ? 1U : 0U},
           {"delta_already_staged", delta_already_staged ? 1U : 0U},
           {"ok", 0}});
    }
    return false;
  }
  if (diagnostics_enabled) {
    vector_index_diagnostics::record_event(
        "vector_persist_commit_artifacts", {{"result", "ok"}},
        {{"load_snapshot_ms", load_snapshot_ms},
         {"begin_persist_ms", begin_persist_ms},
         {"committed_ms", committed_ms},
         {"changelog_ms", changelog_ms},
         {"compact_ms", compact_ms},
         {"prepared_ms", prepared_ms},
         {"manifest_ms", manifest_ms},
         {"commit_persist_ms", commit_persist_ms},
         {"total_ms", vector_index_diagnostics::elapsed_ms_if(
                          diagnostics_enabled, total_start)},
         {"committed_ok", committed_ok ? 1U : 0U},
         {"changelog_ok", changelog_ok ? 1U : 0U},
         {"compact_ok", compact_ok ? 1U : 0U},
         {"prepared_ok", prepared_ok ? 1U : 0U},
         {"manifest_ok", manifest_ok ? 1U : 0U},
         {"delta", (persist_delta || delta_already_staged) ? 1U : 0U},
         {"delta_already_staged", delta_already_staged ? 1U : 0U},
         {"ok", 1}});
  }
  return true;
}

bool persist_commit_artifacts_locked(
    const std::vector<vector_index_metadata_store::change_log_row>
        *commit_delta_rows,
    bool include_prepared) {
  return persist_commit_artifacts_impl_locked(commit_delta_rows,
                                              include_prepared, false);
}

bool persist_staged_commit_artifacts_locked(bool include_prepared) {
  return persist_commit_artifacts_impl_locked(nullptr, include_prepared, true);
}

bool change_log_compaction_needed_with_delta_locked(size_t delta_rows) {
  const size_t threshold = change_log_compact_threshold();
  if (g_change_log_rows.empty() && delta_rows == 0) return false;
  if (g_change_log_rows.size() >= threshold) return true;
  return delta_rows >= threshold - g_change_log_rows.size();
}

thd_txn_context *get_or_create_thd_txn_context_locked(uint64_t thd_id) {
  const auto [it, inserted] = g_thd_txn_contexts.try_emplace(thd_id);
  thd_txn_context &ctx = it->second;
  if (ctx.txn_id != 0) return &ctx;

  ctx.txn_id = allocate_txn_id();
  if (ctx.txn_id != 0) return &ctx;
  if (inserted) g_thd_txn_contexts.erase(it);
  return nullptr;
}

bool ensure_stmt_savepoint_locked(thd_txn_context *ctx, uint64_t statement_id) {
  if (ctx == nullptr) return false;
  if (ctx->stmt_savepoint_active && ctx->active_stmt_id == statement_id)
    return true;

  if (ctx->stmt_savepoint_active) {
    (void)g_index_service.release_savepoint(ctx->txn_id,
                                            ctx->stmt_savepoint_name);
    ctx->stmt_savepoint_active = false;
    ctx->active_stmt_id = 0;
    ctx->stmt_savepoint_name.clear();
  }

  ctx->stmt_savepoint_name = make_stmt_savepoint_name(statement_id);
  if (!g_index_service.savepoint(ctx->txn_id, ctx->stmt_savepoint_name))
    return false;
  ctx->stmt_savepoint_active = true;
  ctx->active_stmt_id = statement_id;
  return true;
}

void clear_stmt_savepoint_state(thd_txn_context *ctx) {
  if (ctx == nullptr) return;
  ctx->stmt_savepoint_active = false;
  ctx->active_stmt_id = 0;
  ctx->stmt_savepoint_name.clear();
}

void discard_thd_txn_context_locked(uint64_t thd_id) {
  auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end()) return;
  const size_t pending_change_count =
      g_index_service.pending_change_count(it->second.txn_id);
  g_index_service.rollback(it->second.txn_id);
  if (pending_change_count > 0) {
    vector_status::subtract_pending_txn_changes(pending_change_count);
  }
  g_thd_txn_contexts.erase(it);
}

bool snapshot_runtime_state_locked(
    std::vector<vector_index_metadata_store::metadata_row> *metadata_rows,
    vector_index::index_service::committed_state *committed_state,
    std::vector<vector_index_metadata_store::change_log_row> *change_log_rows,
    std::vector<std::string> *lagging_index_names) {
  if (metadata_rows == nullptr || committed_state == nullptr ||
      change_log_rows == nullptr || lagging_index_names == nullptr) {
    return false;
  }
  if (!snapshot_metadata_locked(metadata_rows)) return false;
  if (!g_index_service.snapshot_committed_state(committed_state)) return false;
  *change_log_rows = g_change_log_rows;

  lagging_index_names->clear();
  std::vector<std::string> index_names;
  if (!g_index_service.list_indexes(&index_names)) return false;
  lagging_index_names->reserve(index_names.size());
  for (const std::string &index_name : index_names) {
    vector_index::index_service::index_config config;
    bool supports_mutations = false;
    size_t entry_count = 0;
    size_t committed_entry_count = 0;
    if (!g_index_service.describe_index(index_name, &config,
                                        &supports_mutations, &entry_count,
                                        &committed_entry_count)) {
      return false;
    }
    // Standalone ingest intentionally leads its serving backend until rebuild.
    if (config.consistency_mode ==
        vector_index::index_consistency_mode::kStandalone) {
      continue;
    }
    if (entry_count < committed_entry_count) {
      lagging_index_names->push_back(index_name);
    }
  }
  return true;
}

bool restore_runtime_state_locked(
    const std::vector<vector_index_metadata_store::metadata_row> &metadata_rows,
    const vector_index::index_service::committed_state &committed_state,
    const std::vector<vector_index_metadata_store::change_log_row>
        &change_log_rows,
    const std::vector<std::string> &lagging_index_names) {
  DBUG_EXECUTE_IF("vector_registry_fail_restore_runtime_state", return false;);
  const uint64_t next_change_log_sequence = g_next_change_log_sequence;
  g_change_log_rows = change_log_rows;
  g_next_change_log_sequence = next_change_log_sequence;
  for (const auto &row : g_change_log_rows) {
    if (row.sequence >= g_next_change_log_sequence) {
      g_next_change_log_sequence = row.sequence + 1;
    }
  }
  if (!apply_metadata_rows_locked(metadata_rows)) return false;
  if (!g_index_service.restore_committed_state(committed_state)) return false;
  for (const std::string &index_name : lagging_index_names) {
    if (!g_index_service.rebuild_index(index_name)) return false;
  }
  g_manifest_metadata_checkpoint = metadata_rows.size();
  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  if (!snapshot_committed_rows_locked(&committed_rows)) return false;
  g_manifest_committed_checkpoint = committed_rows.size();
  g_manifest_change_log_checkpoint = g_change_log_rows.size();
  vector_status::set_registered_indexes(metadata_rows.size());
  refresh_committed_snapshot_rows_locked();
  refresh_manifest_status_locked();
  return true;
}

bool rollback_runtime_state_locked(
    const std::vector<vector_index_metadata_store::metadata_row> &metadata_rows,
    const vector_index::index_service::committed_state &committed_state,
    const std::vector<vector_index_metadata_store::change_log_row>
        &change_log_rows,
    const std::vector<std::string> &lagging_index_names) {
  if (!restore_runtime_state_locked(metadata_rows, committed_state,
                                    change_log_rows, lagging_index_names)) {
    vector_status::record_runtime_state_rollback_failure();
    return false;
  }
  vector_status::record_runtime_state_rollback();
  return true;
}

bool capture_runtime_commit_state_locked(commit_runtime_snapshot *snapshot) {
  if (snapshot == nullptr) return false;
  if (!g_index_service.snapshot_committed_state(&snapshot->committed_state)) {
    return false;
  }
  if (!g_index_service.snapshot_publication_states(
          &snapshot->publication_states)) {
    return false;
  }
  snapshot->change_log_rows = g_change_log_rows;
  snapshot->manifest_change_log_checkpoint = g_manifest_change_log_checkpoint;
  snapshot->next_change_log_sequence = g_next_change_log_sequence;
  return true;
}

bool rollback_runtime_commit_state_locked(
    const commit_runtime_snapshot &snapshot) {
  g_change_log_rows = snapshot.change_log_rows;
  g_manifest_change_log_checkpoint = snapshot.manifest_change_log_checkpoint;
  g_next_change_log_sequence = snapshot.next_change_log_sequence;
  if (!g_index_service.restore_committed_state(snapshot.committed_state)) {
    vector_status::record_runtime_state_rollback_failure();
    return false;
  }
  if (!g_index_service.restore_publication_states(
          snapshot.publication_states)) {
    vector_status::record_runtime_state_rollback_failure();
    return false;
  }
  refresh_committed_snapshot_rows_locked();
  refresh_manifest_status_locked();
  vector_status::record_runtime_state_rollback();
  return true;
}

}  // namespace vector_index_registry::detail

namespace vector_index_registry {

using namespace detail;

bool snapshot_runtime_state(registry_state_snapshot *snapshot) {
  if (snapshot == nullptr) return false;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return snapshot_runtime_state_locked(&snapshot->metadata_rows,
                                       &snapshot->committed_state,
                                       &snapshot->change_log_rows,
                                       &snapshot->lagging_index_names);
}

bool restore_runtime_state(const registry_state_snapshot &snapshot,
                           bool persist) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!restore_runtime_state_locked(snapshot.metadata_rows,
                                    snapshot.committed_state,
                                    snapshot.change_log_rows,
                                    snapshot.lagging_index_names)) {
    return false;
  }
  g_metadata_loaded = true;
  if (!persist) return true;
  if (!persist_registry_state_locked()) {
    vector_status::record_truth_store_persist_failure();
    return false;
  }
  return true;
}

bool restore_runtime_state_after_drop_rollback(
    const registry_state_snapshot &snapshot,
    const dropped_index_artifacts &artifacts) {
  if (!artifacts.valid) {
    return false;
  }
  if (!artifacts.has_cleanup_intent) {
    return restore_runtime_state(snapshot, true);
  }
  if (artifacts.cleanup_intent.operation !=
      vector_index_truth_store::publication_operation::kDropIndex) {
    return false;
  }

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  runtime_state_snapshot dropped_state;
  if (!capture_runtime_state_locked(&dropped_state)) return false;

  const uint64_t manifest_version = g_manifest_version;
  const uint64_t metadata_checkpoint = g_manifest_metadata_checkpoint;
  const uint64_t committed_checkpoint = g_manifest_committed_checkpoint;
  const uint64_t change_log_checkpoint = g_manifest_change_log_checkpoint;
  const uint64_t next_change_log_sequence = g_next_change_log_sequence;
  if (!restore_runtime_state_locked(snapshot.metadata_rows,
                                    snapshot.committed_state,
                                    snapshot.change_log_rows,
                                    snapshot.lagging_index_names)) {
    fail_stop_registry_locked("drop_rollback_state_restore_failed");
    return false;
  }

  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  vector_status::record_truth_store_persist_request();
  bool persisted = truth_store->begin_persist();
  if (persisted) {
    persisted = persist_metadata_locked(nullptr) &&
                persist_committed_locked(nullptr) &&
                persist_change_log_locked() && persist_prepared_locked() &&
                persist_segment_tasks_locked() && persist_manifest_locked() &&
                truth_store->delete_publication_intent(
                    artifacts.cleanup_intent.index_name,
                    artifacts.cleanup_intent.publication_id);
  }
  if (persisted) persisted = truth_store->commit_persist();
  if (persisted) return true;

  truth_store->rollback_persist();
  g_manifest_version = manifest_version;
  g_manifest_metadata_checkpoint = metadata_checkpoint;
  g_manifest_committed_checkpoint = committed_checkpoint;
  g_manifest_change_log_checkpoint = change_log_checkpoint;
  g_next_change_log_sequence = next_change_log_sequence;
  if (!rollback_runtime_state_locked(dropped_state)) {
    fail_stop_registry_locked("drop_rollback_state_restore_failed");
  }
  vector_status::record_truth_store_persist_failure();
  return false;
}

size_t committed_vector_memory_bytes() {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return 0;
  return g_index_service.committed_vector_memory_bytes();
}

size_t total_pending_vector_memory_bytes() {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return 0;
  return g_index_service.total_pending_vector_memory_bytes();
}

registry_health_state registry_health() {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  return g_registry_health;
}

std::string registry_failure_reason() {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  return g_registry_failure_reason;
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool parse_mapped_index_name_for_testing(const std::string &index_name,
                                         std::string *schema_name,
                                         std::string *table_name,
                                         std::string *column_name) {
  return parse_mapped_index_name(index_name, schema_name, table_name,
                                 column_name);
}

void rename_index_binding_for_testing(const std::string &old_index_name,
                                      const std::string &new_index_name) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  rename_index_binding_locked(old_index_name, new_index_name);
}

bool get_index_binding_for_testing(const std::string &index_name,
                                   std::string *schema_name,
                                   std::string *table_name,
                                   std::string *column_name,
                                   std::string *doc_id_column_name) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  const index_binding binding = binding_for_index_locked(index_name);
  if (schema_name != nullptr) *schema_name = binding.schema_name;
  if (table_name != nullptr) *table_name = binding.table_name;
  if (column_name != nullptr) *column_name = binding.column_name;
  if (doc_id_column_name != nullptr) {
    *doc_id_column_name = binding.doc_id_column_name;
  }
  return !binding.schema_name.empty() || !binding.table_name.empty() ||
         !binding.column_name.empty() || !binding.doc_id_column_name.empty();
}

bool snapshot_runtime_state_for_testing(
    std::vector<vector_index_metadata_store::metadata_row> *metadata_rows,
    vector_index::index_service::committed_state *committed_state,
    std::vector<vector_index_metadata_store::change_log_row> *change_log_rows,
    std::vector<std::string> *lagging_index_names) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  return snapshot_runtime_state_locked(metadata_rows, committed_state,
                                       change_log_rows, lagging_index_names);
}

bool restore_runtime_state_for_testing(
    const std::vector<vector_index_metadata_store::metadata_row> &metadata_rows,
    const vector_index::index_service::committed_state &committed_state,
    const std::vector<vector_index_metadata_store::change_log_row>
        &change_log_rows,
    const std::vector<std::string> &lagging_index_names) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  const bool ok = restore_runtime_state_locked(
      metadata_rows, committed_state, change_log_rows, lagging_index_names);
  if (ok) g_metadata_loaded = true;
  return ok;
}

effective_index_options_for_testing effective_options_for_testing(
    const std::string &provider, const create_index_options &options) {
  effective_index_options_for_testing effective;
  effective.hnsw_build_threads =
      effective_hnsw_build_threads(provider, options);
  effective.faiss_build_threads =
      effective_faiss_build_threads(provider, options);
  effective.diskann_build_threads =
      effective_diskann_build_threads(provider, options);
  effective.diskann_max_degree =
      effective_diskann_max_degree(provider, options);
  effective.diskann_build_complexity =
      effective_diskann_build_complexity(provider, options);
  effective.diskann_pq_code_budget_size =
      effective_diskann_pq_code_budget_size(provider, options);
  effective.diskann_disk_pq_dims =
      effective_diskann_disk_pq_dims(provider, options);
  effective.diskann_accelerate_build =
      effective_diskann_accelerate_build(provider, options);
  effective.diskann_shuffle_build =
      effective_diskann_shuffle_build(provider, options);
  effective.diskann_use_bfs_cache =
      effective_diskann_use_bfs_cache(provider, options);
  effective.diskann_search_complexity =
      effective_diskann_search_complexity(provider, options);
  effective.diskann_search_beamwidth =
      effective_diskann_search_beamwidth(provider, options);
  return effective;
}

void set_change_log_compact_threshold_for_testing(size_t threshold) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  g_change_log_compact_threshold_for_testing = threshold;
}

void reset_for_testing() {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  detail::reset_publication_intent_recovery_for_testing();
  g_index_service.discard_all_pending_changes();
  g_index_service = vector_index::index_service();
  g_metadata_loaded = false;
  g_pending_spill_startup_cleanup_complete = false;
  g_registry_health = registry_health_state::kReady;
  g_registry_failure_reason.clear();
  g_truth_recovery_state = truth_recovery_state{};
  g_next_txn_id.store(1, std::memory_order_relaxed);
  g_manifest_version = 1;
  g_manifest_metadata_checkpoint = 0;
  g_manifest_committed_checkpoint = 0;
  g_manifest_change_log_checkpoint = 0;
  g_next_change_log_sequence = 1;
  g_change_log_rows.clear();
  g_prepared_change_rows.clear();
  g_segment_task_rows.clear();
  g_explicit_txn_owners.clear();
  g_explicit_txns_by_thd.clear();
  g_thd_txn_contexts.clear();
  g_index_bindings.clear();
  g_index_owner_schemas.clear();
  g_change_log_compact_threshold_for_testing = 0;
  vector_status::set_registered_indexes(0);
  vector_status::set_committed_snapshot_rows(0);
  vector_status::set_manifest_version(1);
  vector_status::set_manifest_metadata_checkpoint(0);
  vector_status::set_manifest_committed_checkpoint(0);
  vector_status::set_manifest_change_log_checkpoint(0);
  vector_status::set_apply_latency_ms(0);
  const uint64_t pending = vector_status::pending_txn_changes();
  if (pending > 0) {
    vector_status::subtract_pending_txn_changes(static_cast<size_t>(pending));
  }
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index_registry
