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
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
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
#include "sql/vector/vector_index_service.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_mapped_search.h"
#include "sql/vector/vector_status.h"

namespace vector_index_registry::detail {

std::shared_mutex g_registry_mutex;
vector_index::index_service g_index_service;
bool g_metadata_loaded = false;
std::atomic<uint64_t> g_next_txn_id{1};
uint64_t g_manifest_version = 1;
uint64_t g_manifest_metadata_checkpoint = 0;
uint64_t g_manifest_committed_checkpoint = 0;
uint64_t g_manifest_change_log_checkpoint = 0;
uint64_t g_next_change_log_sequence = 1;
std::vector<vector_index_metadata_store::change_log_row> g_change_log_rows;
std::vector<vector_index_metadata_store::prepared_change_row>
    g_prepared_change_rows;

std::vector<recovery_action> g_recovery_actions;

std::unordered_map<uint64_t, thd_txn_context> g_thd_txn_contexts;
std::unordered_map<std::string, index_binding> g_index_bindings;

namespace {

constexpr size_t kVectorChangeLogCompactRows = 1000000;
constexpr const char *kLifecycleCreating = "creating";
constexpr const char *kLifecycleBackfilling = "backfilling";

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

uint32_t effective_hnsw_build_threads(
    const std::string &provider,
    const vector_index_registry::create_index_options &options) {
  vector_index::backend_provider provider_value =
      vector_index::backend_provider::kNative;
  if (!vector_index::parse_backend_provider(provider, &provider_value) ||
      provider_value != vector_index::backend_provider::kHnswlib) {
    return 0;
  }
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
  if (options.build_threads_specified && options.build_threads != 0) {
    return options.build_threads;
  }
  return static_cast<uint32_t>(opt_vector_diskann_build_threads);
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

}  // namespace

bool xid_matches_row(
    const XID &xid,
    const vector_index_metadata_store::prepared_change_row &row) {
  return static_cast<int64_t>(xid.get_format_id()) == row.format_id &&
         static_cast<int64_t>(xid.get_gtrid_length()) == row.gtrid_length &&
         static_cast<int64_t>(xid.get_bqual_length()) == row.bqual_length &&
         row.xid_data.size() == static_cast<size_t>(xid.get_gtrid_length() +
                                                    xid.get_bqual_length()) &&
         std::memcmp(row.xid_data.data(), xid.get_data(),
                     row.xid_data.size()) == 0;
}

void xid_from_prepared_row(
    const vector_index_metadata_store::prepared_change_row &row, XID *xid) {
  xid->reset();
  xid->set_format_id(static_cast<long>(row.format_id));
  xid->set_gtrid_length(static_cast<long>(row.gtrid_length));
  xid->set_bqual_length(static_cast<long>(row.bqual_length));
  if (!row.xid_data.empty()) {
    xid->set_data(row.xid_data.data(), static_cast<long>(row.xid_data.size()));
  }
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
bool persist_metadata_manifest_locked();
bool persist_prepared_locked();

uint64_t allocate_txn_id() {
  return g_next_txn_id.fetch_add(1, std::memory_order_relaxed);
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

void set_index_binding_locked(const std::string &index_name,
                              const std::string &schema_name,
                              const std::string &table_name,
                              const std::string &column_name,
                              const std::string &doc_id_column_name) {
  g_index_bindings[index_name] =
      index_binding{schema_name, table_name, column_name, doc_id_column_name};
}

void erase_index_binding_locked(const std::string &index_name) {
  g_index_bindings.erase(index_name);
}

void rename_index_binding_locked(const std::string &old_index_name,
                                 const std::string &new_index_name) {
  const auto it = g_index_bindings.find(old_index_name);
  if (it == g_index_bindings.end()) return;

  index_binding binding = it->second;
  const index_binding fallback = binding_from_name(new_index_name);
  if (!fallback.schema_name.empty() || !fallback.table_name.empty() ||
      !fallback.column_name.empty()) {
    binding = fallback;
  }
  g_index_bindings.erase(it);
  g_index_bindings.emplace(new_index_name, std::move(binding));
}

bool create_index_locked(const std::string &index_name, size_t dimension,
                         const std::string &metric, const std::string &mode,
                         const std::string &provider,
                         const index_binding *binding,
                         const vector_index_registry::create_index_options
                             &options) {
  std::vector<vector_index_metadata_store::metadata_row> metadata_before;
  vector_index::index_service::committed_state committed_before;
  std::vector<vector_index_metadata_store::change_log_row> change_log_before;
  std::vector<std::string> lagging_indexes_before;
  if (!snapshot_runtime_state_locked(&metadata_before, &committed_before,
                                     &change_log_before,
                                     &lagging_indexes_before)) {
    return false;
  }
  const bool ok = g_index_service.register_index_from_strings(
      index_name, dimension, metric, mode, provider);
  if (!ok) return false;
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
      effective_hnsw_build_threads(provider, options);
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
      effective_faiss_build_threads(provider, options);
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
      effective_diskann_build_threads(provider, options);
  if (diskann_build_threads != 0 &&
      !g_index_service.set_diskann_build_threads(index_name,
                                                 diskann_build_threads)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }
  if (binding != nullptr) {
    set_index_binding_locked(index_name, binding->schema_name,
                             binding->table_name, binding->column_name,
                             binding->doc_id_column_name);
  } else {
    set_index_binding_locked(index_name, "", "", "", "");
  }
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
  if (info.diskann_search_complexity != 0 &&
      !g_index_service.set_diskann_search_complexity(
          index_name, info.diskann_search_complexity)) {
    return false;
  }
  return true;
}

bool reapply_metadata_tuning_locked(
    const std::vector<vector_index_metadata_store::metadata_row> &rows) {
  for (const auto &row : rows) {
    vector_index_registry::index_info info;
    info.search_ef = row.search_ef;
    info.hnsw_m = row.hnsw_m;
    info.hnsw_ef_construction = row.hnsw_ef_construction;
    info.hnsw_build_threads = row.hnsw_build_threads;
    info.faiss_nlist = row.faiss_nlist;
    info.faiss_nprobe = row.faiss_nprobe;
    info.faiss_pq_m = row.faiss_pq_m;
    info.faiss_pq_bits = row.faiss_pq_bits;
    info.faiss_build_threads = row.faiss_build_threads;
    info.diskann_max_degree = row.diskann_max_degree;
    info.diskann_build_complexity = row.diskann_build_complexity;
    info.diskann_build_threads = row.diskann_build_threads;
    info.diskann_search_complexity = row.diskann_search_complexity;
    if (!apply_index_tuning_locked(row.index_name, info)) {
      return false;
    }
  }
  return true;
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
    if (!g_index_service.describe_index(
            index_name, &config, &supports_mutations, &entry_count,
            &committed_entry_count, &lifecycle_state, &lifecycle_version,
            &last_error_code, &last_error_ts, nullptr, &recover_fallback_count,
            &last_recover_fallback_ts)) {
      return false;
    }

    vector_index_metadata_store::metadata_row row;
    row.index_name = index_name;
    row.dimension = config.dimension;
    row.metric = config.metric;
    row.mode = config.mode;
    row.provider = config.provider;
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
    row.diskann_search_complexity = config.diskann_search_complexity;
    const index_binding binding = binding_for_index_locked(index_name);
    row.schema_name = binding.schema_name;
    row.table_name = binding.table_name;
    row.column_name = binding.column_name;
    row.doc_id_column_name = binding.doc_id_column_name;
    row.lifecycle_state = lifecycle_state;
    row.lifecycle_version = lifecycle_version;
    row.last_error_code = last_error_code;
    row.last_error_ts = last_error_ts;
    row.recover_fallback_count = recover_fallback_count;
    row.last_recover_fallback_ts = last_recover_fallback_ts;
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

  rows->reserve(pending_changes.size());
  for (const auto &change : pending_changes) {
    vector_index_metadata_store::prepared_change_row row;
    row.format_id = xid.get_format_id();
    row.gtrid_length = xid.get_gtrid_length();
    row.bqual_length = xid.get_bqual_length();
    row.xid_data.assign(
        xid.get_data(),
        xid.get_data() + xid.get_gtrid_length() + xid.get_bqual_length());
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
  g_index_service = vector_index::index_service();
  g_index_bindings.clear();
  for (const auto &row : rows) {
    vector_index::index_service::index_config config;
    config.dimension = row.dimension;
    config.metric = row.metric;
    config.mode = row.mode;
    config.provider = row.provider;
    if (!g_index_service.register_index(row.index_name, config)) {
      return false;
    }
    set_index_binding_locked(row.index_name, row.schema_name, row.table_name,
                             row.column_name, row.doc_id_column_name);
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
  return g_index_service.restore_committed_state(state);
}

bool apply_change_log_rows_locked(
    const std::vector<vector_index_metadata_store::change_log_row> &rows) {
  if (rows.empty()) return true;

  std::vector<vector_index_metadata_store::change_log_row> ordered_rows(rows);
  std::sort(ordered_rows.begin(), ordered_rows.end(),
            [](const vector_index_metadata_store::change_log_row &lhs,
               const vector_index_metadata_store::change_log_row &rhs) {
              if (lhs.sequence != rhs.sequence)
                return lhs.sequence < rhs.sequence;
              if (lhs.index_name != rhs.index_name)
                return lhs.index_name < rhs.index_name;
              if (lhs.doc_id != rhs.doc_id) return lhs.doc_id < rhs.doc_id;
              return static_cast<int>(lhs.op) < static_cast<int>(rhs.op);
            });

  vector_index::index_service::committed_state state;
  if (!g_index_service.snapshot_committed_state(&state)) return false;
  for (const auto &row : ordered_rows) {
    auto index_it = state.find(row.index_name);
    if (index_it == state.end()) return false;
    if (row.op == vector_index_metadata_store::change_op::kUpsert) {
      index_it->second[row.doc_id] = row.vector;
    } else if (row.op == vector_index_metadata_store::change_op::kErase) {
      index_it->second.erase(row.doc_id);
    } else {
      return false;
    }
  }
  return g_index_service.restore_committed_state(state);
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

void append_pending_change_log_delta_locked(
    uint64_t txn_id,
    const std::vector<vector_index::index_service::pending_change_snapshot>
        &changes) {
  std::vector<vector_index::index_service::pending_change_snapshot> ordered(
      changes);
  std::sort(ordered.begin(), ordered.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.index_name != rhs.index_name)
                return lhs.index_name < rhs.index_name;
              if (lhs.doc_id != rhs.doc_id) return lhs.doc_id < rhs.doc_id;
              return lhs.erase < rhs.erase;
            });

  for (const auto &change : ordered) {
    vector_index_metadata_store::change_log_row row;
    row.sequence = g_next_change_log_sequence++;
    row.txn_id = txn_id;
    row.op = change.erase ? vector_index_metadata_store::change_op::kErase
                          : vector_index_metadata_store::change_op::kUpsert;
    row.index_name = change.index_name;
    row.doc_id = change.doc_id;
    row.vector = change.vector;
    g_change_log_rows.push_back(std::move(row));
  }
  g_manifest_change_log_checkpoint = g_change_log_rows.size();
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

bool persist_or_rollback_runtime_state_locked(
    const runtime_state_snapshot &snapshot) {
  if (persist_registry_state_locked()) return true;
  vector_status::record_truth_store_persist_failure();
  return rollback_runtime_state_locked(snapshot);
}

bool persist_or_rollback_index_config_locked(
    const std::string &index_name,
    const vector_index::index_service::index_config &before) {
  if (persist_metadata_manifest_locked()) return true;

  vector_status::record_truth_store_persist_failure();
  vector_status::record_runtime_state_rollback();
  if (!g_index_service.restore_index_config(index_name, before)) {
    vector_status::record_runtime_state_rollback_failure();
  }
  return false;
}

bool persist_index_config_manifest_locked(
    const std::string &index_name,
    const vector_index::index_service::index_config &config) {
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
    row.dimension = config.dimension;
    row.metric = config.metric;
    row.mode = config.mode;
    row.provider = config.provider;
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
    row.diskann_search_complexity = config.diskann_search_complexity;
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

bool ensure_metadata_loaded_locked() {
  if (g_metadata_loaded) return true;

  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  bool manifest_checkpoint_mismatch = false;
  bool committed_checkpoint_mismatch = false;
  bool change_log_checkpoint_mismatch = false;

  vector_index_metadata_store::manifest_row manifest_row;
  if (!truth_store->load_manifest(&manifest_row)) {
    vector_status::record_metadata_load_failure();
    vector_status::record_manifest_load_failure();
    if (!truth_store->quarantine_manifest()) return false;
    manifest_row = vector_index_metadata_store::manifest_row();
  }
  g_manifest_version = manifest_row.version == 0 ? 1 : manifest_row.version;
  g_manifest_metadata_checkpoint = manifest_row.metadata_checkpoint;
  g_manifest_committed_checkpoint = manifest_row.committed_checkpoint;
  g_manifest_change_log_checkpoint = manifest_row.change_log_checkpoint;

  std::vector<vector_index_metadata_store::metadata_row> rows;
  if (!truth_store->load_metadata(&rows)) {
    vector_status::record_metadata_load_failure();
    if (!truth_store->quarantine_metadata()) return false;
    rows.clear();
  }
  const std::unordered_set<std::string> incomplete_create_indexes =
      remove_incomplete_create_metadata_rows(&rows);
  const bool cleanup_incomplete_create_state =
      !incomplete_create_indexes.empty();
  if (g_manifest_metadata_checkpoint > rows.size()) {
    manifest_checkpoint_mismatch = true;
    g_manifest_metadata_checkpoint = rows.size();
  }
  if (!apply_metadata_rows_locked(rows)) return false;

  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  if (!truth_store->load_committed(&committed_rows)) {
    vector_status::record_committed_load_failure();
    if (!truth_store->quarantine_committed()) return false;
    committed_rows.clear();
  }
  remove_committed_rows_for_indexes(incomplete_create_indexes, &committed_rows);
  if (g_manifest_committed_checkpoint > committed_rows.size()) {
    manifest_checkpoint_mismatch = true;
    committed_checkpoint_mismatch = true;
    g_manifest_committed_checkpoint = committed_rows.size();
  }
  if (!apply_committed_rows_locked(committed_rows)) return false;

  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  if (!truth_store->load_change_log(&change_log_rows)) {
    vector_status::record_metadata_load_failure();
    vector_status::record_change_log_load_failure();
    if (!truth_store->quarantine_change_log()) return false;
    change_log_rows.clear();
  }

  remove_change_log_rows_for_indexes(incomplete_create_indexes,
                                     &change_log_rows);
  g_change_log_rows = std::move(change_log_rows);
  if (g_manifest_change_log_checkpoint > g_change_log_rows.size()) {
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
  g_next_change_log_sequence = 1;
  for (const auto &row : g_change_log_rows) {
    if (row.sequence >= g_next_change_log_sequence) {
      g_next_change_log_sequence = row.sequence + 1;
    }
  }
  if (!apply_change_log_rows_locked(g_change_log_rows)) {
    vector_status::record_metadata_load_failure();
    vector_status::record_change_log_load_failure();
    vector_status::record_change_log_replay_failure();
    if (!truth_store->quarantine_change_log()) return false;
    g_change_log_rows.clear();
    g_next_change_log_sequence = 1;
    g_manifest_change_log_checkpoint = 0;
    if (!apply_committed_rows_locked(committed_rows)) return false;
  }
  if (!reapply_metadata_tuning_locked(rows)) return false;

  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  if (!truth_store->load_prepared(&prepared_rows)) {
    vector_status::record_metadata_load_failure();
    if (!truth_store->quarantine_prepared()) return false;
    prepared_rows.clear();
  }
  remove_prepared_rows_for_indexes(incomplete_create_indexes, &prepared_rows);
  g_prepared_change_rows = std::move(prepared_rows);

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

  g_manifest_metadata_checkpoint = rows.size();
  std::vector<vector_index_metadata_store::committed_row>
      effective_committed_rows;
  if (!snapshot_committed_rows_locked(&effective_committed_rows)) return false;
  g_manifest_committed_checkpoint = effective_committed_rows.size();
  g_manifest_change_log_checkpoint = g_change_log_rows.size();

  vector_status::set_registered_indexes(rows.size());
  refresh_committed_snapshot_rows_locked();
  refresh_manifest_status_locked();
  g_metadata_loaded = true;
  if (cleanup_incomplete_create_state && !persist_registry_state_locked()) {
    vector_status::record_truth_store_persist_failure();
    g_metadata_loaded = false;
    return false;
  }
  return true;
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

bool persist_manifest_locked() {
  vector_index_metadata_store::manifest_row row;
  row.state = "ready";
  row.version = g_manifest_version + 1;
  row.metadata_checkpoint = g_manifest_metadata_checkpoint;
  row.committed_checkpoint = g_manifest_committed_checkpoint;
  row.change_log_checkpoint = g_manifest_change_log_checkpoint;
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
            persist_prepared_locked() && persist_manifest_locked();
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

bool persist_metadata_manifest_locked() {
  vector_status::record_truth_store_persist_request();
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (!truth_store->begin_persist()) return false;
  bool ok = persist_metadata_locked(nullptr) && persist_manifest_locked();
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

bool persist_commit_artifacts_locked(
    const std::vector<vector_index_metadata_store::change_log_row>
        *commit_delta_rows,
    bool include_prepared) {
  const bool diagnostics_enabled = vector_index_diagnostics::enabled();
  const auto total_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  const bool use_delta_path = commit_delta_rows != nullptr &&
                              truth_store->is_transactional() &&
                              truth_store->supports_delta_persist();
  persisted_commit_artifacts_snapshot persisted_before;
  uint64_t load_snapshot_ms = 0;
  if (!use_delta_path) {
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

  vector_status::record_truth_store_persist_request();
  if (use_delta_path) vector_status::record_truth_store_delta_persist_request();
  const auto begin_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  if (!truth_store->begin_persist()) {
    vector_status::record_truth_store_persist_failure();
    if (use_delta_path)
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
      vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled, begin_start);
  const auto committed_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  const bool committed_ok =
      use_delta_path ? persist_committed_delta_locked(*commit_delta_rows)
                     : persist_committed_locked(nullptr);
  const uint64_t committed_ms = vector_index_diagnostics::elapsed_ms_if(
      diagnostics_enabled, committed_start);
  uint64_t changelog_ms = 0;
  uint64_t prepared_ms = 0;
  uint64_t manifest_ms = 0;
  uint64_t commit_persist_ms = 0;
  const auto changelog_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  const bool changelog_ok =
      committed_ok ? (use_delta_path
                          ? persist_change_log_delta_locked(*commit_delta_rows)
                          : persist_change_log_locked())
                   : false;
  if (committed_ok)
    changelog_ms = vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                           changelog_start);
  bool compact_ok = true;
  uint64_t compact_ms = 0;
  if (changelog_ok && change_log_compaction_needed_locked()) {
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
    if (use_delta_path) {
      vector_status::record_persist_artifact_rollback();
    } else if (restore_persisted_commit_artifacts_snapshot_locked(
                   persisted_before)) {
      vector_status::record_persist_artifact_rollback();
    } else {
      vector_status::record_persist_artifact_rollback_failure();
    }
    vector_status::record_truth_store_persist_failure();
    if (use_delta_path)
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
           {"delta", use_delta_path ? 1U : 0U},
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
         {"delta", use_delta_path ? 1U : 0U},
         {"ok", 1}});
  }
  return true;
}

thd_txn_context *get_or_create_thd_txn_context_locked(uint64_t thd_id) {
  thd_txn_context &ctx = g_thd_txn_contexts[thd_id];
  if (ctx.txn_id == 0) ctx.txn_id = allocate_txn_id();
  return &ctx;
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
  g_change_log_rows = change_log_rows;
  g_next_change_log_sequence = 1;
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
  if (!reapply_metadata_tuning_locked(metadata_rows)) return false;

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

void set_change_log_compact_threshold_for_testing(size_t threshold) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  g_change_log_compact_threshold_for_testing = threshold;
}

void reset_for_testing() {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  g_index_service = vector_index::index_service();
  g_metadata_loaded = false;
  g_next_txn_id.store(1, std::memory_order_relaxed);
  g_manifest_version = 1;
  g_manifest_metadata_checkpoint = 0;
  g_manifest_committed_checkpoint = 0;
  g_manifest_change_log_checkpoint = 0;
  g_next_change_log_sequence = 1;
  g_change_log_rows.clear();
  g_prepared_change_rows.clear();
  g_thd_txn_contexts.clear();
  g_index_bindings.clear();
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
