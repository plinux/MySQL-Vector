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
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sql/vector/vector_index_identity.h"
#include "sql/vector/vector_index_registry_internal.h"
#include "sql/vector/vector_index_service_internal.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_status.h"

namespace vector_index_registry {

using namespace detail;

namespace {

using vector_index::detail::build_backend_from_config;

constexpr size_t k_backfill_committed_persist_batch_rows = 256;

bool binding_matches_table(const index_binding &binding,
                           const std::string &schema_name,
                           const std::string &table_name) {
  return binding.schema_name == schema_name &&
         binding.table_name == table_name && !binding.column_name.empty();
}

bool binding_matches_column(const index_binding &binding,
                            const std::string &schema_name,
                            const std::string &table_name,
                            const std::string &column_name) {
  return binding_matches_table(binding, schema_name, table_name) &&
         binding.column_name == column_name;
}

bool snapshot_index_config_locked(
    const std::string &index_name,
    vector_index::index_service::index_config *config) {
  return g_index_service.describe_index(index_name, config, nullptr, nullptr,
                                        nullptr);
}

bool index_config_matches(
    const vector_index::index_service::index_config &lhs,
    const vector_index::index_service::index_config &rhs) {
  return lhs.dimension == rhs.dimension && lhs.metric == rhs.metric &&
         lhs.mode == rhs.mode && lhs.provider == rhs.provider &&
         lhs.consistency_mode == rhs.consistency_mode &&
         lhs.search_ef == rhs.search_ef && lhs.hnsw_m == rhs.hnsw_m &&
         lhs.hnsw_ef_construction == rhs.hnsw_ef_construction &&
         lhs.hnsw_build_threads == rhs.hnsw_build_threads &&
         lhs.faiss_nlist == rhs.faiss_nlist &&
         lhs.faiss_nprobe == rhs.faiss_nprobe &&
         lhs.faiss_pq_m == rhs.faiss_pq_m &&
         lhs.faiss_pq_bits == rhs.faiss_pq_bits &&
         lhs.faiss_build_threads == rhs.faiss_build_threads &&
         lhs.diskann_max_degree == rhs.diskann_max_degree &&
         lhs.diskann_build_complexity == rhs.diskann_build_complexity &&
         lhs.diskann_build_threads == rhs.diskann_build_threads &&
         lhs.diskann_build_mode_value == rhs.diskann_build_mode_value &&
         lhs.diskann_build_mode_specified ==
             rhs.diskann_build_mode_specified &&
         lhs.diskann_search_complexity == rhs.diskann_search_complexity &&
         lhs.diskann_search_beamwidth == rhs.diskann_search_beamwidth &&
         lhs.diskann_pq_code_budget_size ==
             rhs.diskann_pq_code_budget_size;
}

vector_index_metadata_store::change_log_row make_backfill_delta_row(
    const std::string &index_name, uint64_t sequence, uint64_t doc_id,
    const vector_index::vector_data &vector) {
  vector_index_metadata_store::change_log_row row;
  row.sequence = sequence;
  row.op = vector_index_metadata_store::change_op::kUpsert;
  row.index_name = index_name;
  row.doc_id = doc_id;
  row.vector = vector;
  return row;
}

vector_index_metadata_store::change_log_row make_erase_delta_row(
    const std::string &index_name, uint64_t sequence, uint64_t doc_id) {
  vector_index_metadata_store::change_log_row row;
  row.sequence = sequence;
  row.op = vector_index_metadata_store::change_op::kErase;
  row.index_name = index_name;
  row.doc_id = doc_id;
  return row;
}

bool persist_backfill_committed_batch_locked(
    const std::vector<vector_index_metadata_store::change_log_row> &rows) {
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  vector_status::record_truth_store_persist_request();
  vector_status::record_truth_store_delta_persist_request();
  if (!truth_store->begin_persist()) {
    vector_status::record_truth_store_persist_failure();
    vector_status::record_truth_store_delta_persist_failure();
    return false;
  }
  const bool ok = persist_committed_delta_locked(rows);
  if (!ok) {
    truth_store->rollback_persist();
    vector_status::record_truth_store_persist_failure();
    vector_status::record_truth_store_delta_persist_failure();
    return false;
  }
  if (!truth_store->commit_persist()) {
    truth_store->rollback_persist();
    vector_status::record_truth_store_persist_failure();
    vector_status::record_truth_store_delta_persist_failure();
    return false;
  }
  return true;
}

bool persist_metadata_manifest_locked(bool include_change_log) {
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  vector_status::record_truth_store_persist_request();
  if (!truth_store->begin_persist()) {
    vector_status::record_truth_store_persist_failure();
    return false;
  }
  bool ok = persist_metadata_locked(nullptr);
  if (ok && include_change_log) ok = persist_change_log_locked();
  if (ok) ok = persist_manifest_locked();
  if (!ok) {
    truth_store->rollback_persist();
    vector_status::record_truth_store_persist_failure();
    return false;
  }
  if (!truth_store->commit_persist()) {
    truth_store->rollback_persist();
    vector_status::record_truth_store_persist_failure();
    vector_status::record_metadata_persist_failure();
    vector_status::record_manifest_persist_failure();
    return false;
  }
  return true;
}

bool persist_backfill_committed_entries_locked(
    const std::string &index_name,
    const vector_index::index_service::committed_entries &entries) {
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (!truth_store->is_transactional() ||
      !truth_store->supports_delta_persist()) {
    return persist_registry_state_locked();
  }

  const uint64_t checkpoint_before = g_manifest_committed_checkpoint;
  if (entries.empty()) {
    g_manifest_committed_checkpoint = g_index_service.committed_entry_count();
    vector_status::set_committed_snapshot_rows(g_manifest_committed_checkpoint);
    return true;
  }

  std::vector<uint64_t> doc_ids;
  doc_ids.reserve(entries.size());
  for (const auto &entry : entries) doc_ids.push_back(entry.first);
  std::sort(doc_ids.begin(), doc_ids.end());

  std::vector<vector_index_metadata_store::change_log_row> batch;
  batch.reserve(k_backfill_committed_persist_batch_rows);
  uint64_t sequence = 1;
  for (uint64_t doc_id : doc_ids) {
    auto entry_it = entries.find(doc_id);
    if (entry_it == entries.end()) {
      g_manifest_committed_checkpoint = checkpoint_before;
      return false;
    }
    batch.push_back(
        make_backfill_delta_row(index_name, sequence++, doc_id,
                                entry_it->second));
    if (batch.size() < k_backfill_committed_persist_batch_rows) continue;
    if (!persist_backfill_committed_batch_locked(batch)) {
      g_manifest_committed_checkpoint = checkpoint_before;
      return false;
    }
    batch.clear();
  }
  if (!batch.empty() && !persist_backfill_committed_batch_locked(batch)) {
    g_manifest_committed_checkpoint = checkpoint_before;
    return false;
  }
  return true;
}

bool persist_drop_committed_entries_locked(
    const std::string &index_name,
    const vector_index::index_service::committed_entries &entries) {
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (!truth_store->is_transactional() ||
      !truth_store->supports_delta_persist()) {
    return persist_registry_state_locked();
  }

  const uint64_t checkpoint_before = g_manifest_committed_checkpoint;
  if (entries.empty()) {
    g_manifest_committed_checkpoint = g_index_service.committed_entry_count();
    vector_status::set_committed_snapshot_rows(g_manifest_committed_checkpoint);
    return true;
  }

  std::vector<uint64_t> doc_ids;
  doc_ids.reserve(entries.size());
  for (const auto &entry : entries) doc_ids.push_back(entry.first);
  std::sort(doc_ids.begin(), doc_ids.end());

  std::vector<vector_index_metadata_store::change_log_row> batch;
  batch.reserve(k_backfill_committed_persist_batch_rows);
  uint64_t sequence = 1;
  for (uint64_t doc_id : doc_ids) {
    batch.push_back(make_erase_delta_row(index_name, sequence++, doc_id));
    if (batch.size() < k_backfill_committed_persist_batch_rows) continue;
    if (!persist_backfill_committed_batch_locked(batch)) {
      g_manifest_committed_checkpoint = checkpoint_before;
      return false;
    }
    batch.clear();
  }
  if (!batch.empty() && !persist_backfill_committed_batch_locked(batch)) {
    g_manifest_committed_checkpoint = checkpoint_before;
    return false;
  }
  return true;
}

void refresh_committed_checkpoint_from_runtime_locked() {
  g_manifest_committed_checkpoint = g_index_service.committed_entry_count();
  vector_status::set_committed_snapshot_rows(g_manifest_committed_checkpoint);
}

bool rollback_runtime_state_and_fail_locked(
    const runtime_state_snapshot &snapshot) {
  if (!rollback_runtime_state_locked(snapshot)) return false;
  return false;
}

bool persist_registry_state_or_rollback_locked(
    const runtime_state_snapshot &snapshot, bool record_truth_store_failure) {
  if (persist_registry_state_locked()) return true;
  if (record_truth_store_failure)
    vector_status::record_truth_store_persist_failure();
  return rollback_runtime_state_and_fail_locked(snapshot);
}

bool persist_metadata_manifest_or_rollback_locked(
    const runtime_state_snapshot &snapshot, bool include_change_log) {
  if (persist_metadata_manifest_locked(include_change_log)) return true;
  return rollback_runtime_state_and_fail_locked(snapshot);
}

struct lifecycle_backend_plan {
  std::string index_name;
  vector_index::index_service::index_config config;
  vector_index::index_service::committed_entries entries;
  uint64_t lifecycle_version{0};
};

struct prepared_lifecycle_backend {
  lifecycle_backend_plan plan;
  std::unique_ptr<vector_index::backend> backend;
  bool used_recover_fallback{false};
};

struct drop_artifact_plan {
  std::string index_name;
  vector_index::backend_mode mode{vector_index::backend_mode::kMemory};
  vector_index::backend_provider provider{
      vector_index::backend_provider::kNative};
};

bool uses_standalone_source(const lifecycle_backend_plan &plan) {
  return plan.config.consistency_mode ==
         vector_index::index_consistency_mode::kStandalone;
}

void replace_segment_task_rows_for_index(
    const std::string &index_name,
    const std::vector<vector_index_metadata_store::segment_task_row> &new_rows) {
  if (index_name.empty()) return;
  g_segment_task_rows.erase(
      std::remove_if(g_segment_task_rows.begin(), g_segment_task_rows.end(),
                     [&index_name](const auto &row) {
                       return row.index_name == index_name;
                     }),
      g_segment_task_rows.end());
  g_segment_task_rows.insert(g_segment_task_rows.end(), new_rows.begin(),
                             new_rows.end());
}

bool refresh_segment_tasks_from_service_locked(const std::string &index_name) {
  std::vector<vector_index_metadata_store::segment_task_row> rows;
  if (!g_index_service.snapshot_segment_tasks(index_name, &rows)) return false;
  replace_segment_task_rows_for_index(index_name, rows);
  return true;
}

bool snapshot_backend_plan_from_state_locked(
    const std::string &index_name,
    const vector_index::index_service::committed_state &committed_state,
    lifecycle_backend_plan *plan) {
  if (plan == nullptr) return false;

  lifecycle_backend_plan candidate;
  candidate.index_name = index_name;
  if (!g_index_service.describe_index(
          index_name, &candidate.config, nullptr, nullptr, nullptr, nullptr,
          &candidate.lifecycle_version)) {
    return false;
  }
  if (candidate.lifecycle_version == 0 ||
      g_index_service.has_pending_changes_for_index(index_name)) {
    return false;
  }
  if (candidate.config.consistency_mode ==
      vector_index::index_consistency_mode::kStandalone) {
    *plan = std::move(candidate);
    return true;
  }

  auto state_it = committed_state.find(index_name);
  if (state_it == committed_state.end()) return false;
  candidate.entries = state_it->second;
  *plan = std::move(candidate);
  return true;
}

bool snapshot_backend_plan_locked(const std::string &index_name,
                                  lifecycle_backend_plan *plan) {
  vector_index::index_service::committed_state committed_state;
  if (!g_index_service.snapshot_committed_state(&committed_state)) return false;
  return snapshot_backend_plan_from_state_locked(index_name, committed_state,
                                                plan);
}

bool snapshot_all_backend_plans_locked(
    std::vector<lifecycle_backend_plan> *plans) {
  if (plans == nullptr) return false;

  std::vector<std::string> index_names;
  if (!g_index_service.list_indexes(&index_names)) return false;
  std::sort(index_names.begin(), index_names.end());

  vector_index::index_service::committed_state committed_state;
  if (!g_index_service.snapshot_committed_state(&committed_state)) return false;

  plans->clear();
  plans->reserve(index_names.size());
  for (const std::string &index_name : index_names) {
    lifecycle_backend_plan plan;
    if (!snapshot_backend_plan_from_state_locked(index_name, committed_state,
                                                &plan)) {
      return false;
    }
    plans->push_back(std::move(plan));
  }
  return true;
}

bool validate_backend_plan_against_state_locked(
    const lifecycle_backend_plan &plan,
    const vector_index::index_service::committed_state &current_state) {
  vector_index::index_service::index_config current_config;
  uint64_t current_lifecycle_version = 0;
  if (!g_index_service.describe_index(
          plan.index_name, &current_config, nullptr, nullptr, nullptr, nullptr,
          &current_lifecycle_version)) {
    return false;
  }
  if (!index_config_matches(plan.config, current_config) ||
      current_lifecycle_version != plan.lifecycle_version ||
      g_index_service.has_pending_changes_for_index(plan.index_name)) {
    return false;
  }
  if (uses_standalone_source(plan)) return true;

  auto current_entries_it = current_state.find(plan.index_name);
  return current_entries_it != current_state.end() &&
         current_entries_it->second == plan.entries;
}

bool validate_backend_plan_locked(const lifecycle_backend_plan &plan) {
  vector_index::index_service::committed_state current_state;
  if (!g_index_service.snapshot_committed_state(&current_state)) return false;
  return validate_backend_plan_against_state_locked(plan, current_state);
}

std::unique_ptr<vector_index::backend> build_rebuilt_backend(
    const lifecycle_backend_plan &plan) {
  if (uses_standalone_source(plan)) return nullptr;
  std::unique_ptr<vector_index::backend> rebuilt =
      build_backend_from_config(plan.index_name, plan.config);
  if (rebuilt == nullptr ||
      !rebuilt->rebuild_from_committed_entries(plan.entries)) {
    return nullptr;
  }
  return rebuilt;
}

std::unique_ptr<vector_index::backend> build_recovered_backend(
    const lifecycle_backend_plan &plan, bool *used_recover_fallback) {
  if (used_recover_fallback == nullptr) return nullptr;
  *used_recover_fallback = false;
  if (uses_standalone_source(plan)) return nullptr;

  std::unique_ptr<vector_index::backend> recovered =
      build_backend_from_config(plan.index_name, plan.config);
  if (recovered == nullptr ||
      !recovered->recover_committed_entries(plan.entries)) {
    return nullptr;
  }
  *used_recover_fallback = recovered->last_recover_used_fallback();
  return recovered;
}

bool rebuild_standalone_plan_locked(const lifecycle_backend_plan &plan) {
  vector_index::index_service::committed_state current_state;
  if (!g_index_service.snapshot_committed_state(&current_state)) return false;
  if (!validate_backend_plan_against_state_locked(plan, current_state))
    return false;

  runtime_state_snapshot snapshot;
  if (!capture_runtime_state_locked(&snapshot)) return false;
  if (!g_index_service.rebuild_index(plan.index_name)) {
    const bool has_segment_tasks =
        refresh_segment_tasks_from_service_locked(plan.index_name);
    if (has_segment_tasks) (void)persist_segment_tasks_locked();
    return rollback_runtime_state_and_fail_locked(snapshot);
  }
  if (!refresh_segment_tasks_from_service_locked(plan.index_name)) {
    return rollback_runtime_state_and_fail_locked(snapshot);
  }
  if (!persist_registry_state_or_rollback_locked(snapshot, true)) return false;
  return evict_committed_cache_to_budget_locked();
}

bool recover_standalone_plan_locked(const lifecycle_backend_plan &plan) {
  vector_index::index_service::committed_state current_state;
  if (!g_index_service.snapshot_committed_state(&current_state)) return false;
  if (!validate_backend_plan_against_state_locked(plan, current_state))
    return false;

  runtime_state_snapshot snapshot;
  if (!capture_runtime_state_locked(&snapshot)) return false;
  if (!g_index_service.recover_index(plan.index_name)) {
    return rollback_runtime_state_and_fail_locked(snapshot);
  }
  if (!persist_registry_state_or_rollback_locked(snapshot, true)) return false;
  return evict_committed_cache_to_budget_locked();
}

bool snapshot_drop_artifact_plan_locked(const std::string &index_name,
                                        drop_artifact_plan *plan) {
  if (plan == nullptr) return false;

  vector_index::index_service::index_config config;
  if (!g_index_service.describe_index(index_name, &config, nullptr, nullptr,
                                      nullptr)) {
    return false;
  }
  plan->index_name = index_name;
  plan->mode = config.mode;
  plan->provider = config.provider;
  return true;
}

void cleanup_drop_artifacts(const std::vector<drop_artifact_plan> &plans) {
  for (const drop_artifact_plan &plan : plans) {
    (void)vector_index::remove_backend_artifacts(plan.index_name, plan.mode,
                                                 plan.provider);
  }
}

template <typename Prepare, typename Apply>
bool apply_persisted_index_config_change_locked(const std::string &index_name,
                                                Prepare prepare,
                                                Apply apply) {
  vector_index::index_service::index_config before;
  if (!snapshot_index_config_locked(index_name, &before)) return false;

  vector_index::index_service::index_config candidate = before;
  prepare(&candidate);
  if (!persist_index_config_manifest_locked(index_name, candidate)) return false;
  if (apply()) return true;

  vector_status::record_runtime_state_rollback();
  if (!g_index_service.restore_index_config(index_name, before)) {
    vector_status::record_runtime_state_rollback_failure();
    return false;
  }
  if (!persist_index_config_manifest_locked(index_name, before)) {
    vector_status::record_truth_store_persist_failure();
  }
  return false;
}

}  // namespace

bool create_index(const std::string &index_name, size_t dimension,
                  const std::string &metric, const std::string &mode,
                  const std::string &provider,
                  const std::string &owner_schema) {
  return create_index(index_name, dimension, metric, mode, provider,
                      owner_schema, create_index_options{});
}

bool create_index(const std::string &index_name, size_t dimension,
                  const std::string &metric, const std::string &mode,
                  const std::string &provider, const std::string &owner_schema,
                  const create_index_options &options) {
  vector_status::record_index_create_request();
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return create_index_locked(index_name, dimension, metric, mode, provider,
                             nullptr, owner_schema, options);
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool create_index(const std::string &index_name, size_t dimension,
                  const std::string &metric, const std::string &mode,
                  const std::string &provider) {
  return create_index(index_name, dimension, metric, mode, provider,
                      "vector_test_owner");
}

bool create_index(const std::string &index_name, size_t dimension,
                  const std::string &metric, const std::string &mode,
                  const std::string &provider,
                  const create_index_options &options) {
  return create_index(index_name, dimension, metric, mode, provider,
                      "vector_test_owner", options);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

bool create_mapped_index(const std::string &index_name, size_t dimension,
                         const std::string &metric, const std::string &mode,
                         const std::string &provider,
                         const std::string &schema_name,
                         const std::string &table_name,
                         const std::string &column_name,
                         const std::string &doc_id_column_name) {
  return create_mapped_index(index_name, dimension, metric, mode, provider,
                             schema_name, table_name, column_name,
                             doc_id_column_name, create_index_options{});
}

bool create_mapped_index(const std::string &index_name, size_t dimension,
                         const std::string &metric, const std::string &mode,
                         const std::string &provider,
                         const std::string &schema_name,
                         const std::string &table_name,
                         const std::string &column_name,
                         const std::string &doc_id_column_name,
                         const create_index_options &options) {
  vector_status::record_index_create_request();
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  const index_binding binding{schema_name, table_name, column_name,
                              doc_id_column_name};
  return create_index_locked(index_name, dimension, metric, mode, provider,
                             &binding, schema_name, options);
}

bool drop_index_impl(const std::string &index_name,
                     dropped_index_artifacts *artifacts,
                     bool cleanup_artifacts) {
  vector_status::record_index_drop_request();
  if (artifacts != nullptr) *artifacts = dropped_index_artifacts{};
  std::vector<drop_artifact_plan> drop_artifacts;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    std::vector<vector_index_metadata_store::metadata_row> metadata_before;
    vector_index::index_service::committed_state committed_before;
    std::vector<vector_index_metadata_store::change_log_row> change_log_before;
    std::vector<std::string> lagging_indexes_before;
    if (!snapshot_runtime_state_locked(&metadata_before, &committed_before,
                                       &change_log_before,
                                       &lagging_indexes_before)) {
      return false;
    }
    vector_index::index_service::committed_entries dropped_entries;
    auto dropped_entries_it = committed_before.find(index_name);
    if (dropped_entries_it != committed_before.end()) {
      dropped_entries = dropped_entries_it->second;
    }
    drop_artifact_plan artifact_plan;
    if (!snapshot_drop_artifact_plan_locked(index_name, &artifact_plan))
      return false;
    if (!g_index_service.unregister_index(index_name)) return false;
    erase_index_binding_and_owner_schema_locked(index_name);
    prune_change_log_for_index_locked(index_name);
    refresh_committed_checkpoint_from_runtime_locked();
    if (!persist_metadata_manifest_locked(true)) {
      vector_status::record_truth_store_persist_failure();
      if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                         change_log_before,
                                         lagging_indexes_before)) {
        return false;
      }
      (void)persist_backfill_committed_entries_locked(index_name,
                                                      dropped_entries);
      (void)persist_metadata_manifest_locked(true);
      return false;
    }
    if (!persist_drop_committed_entries_locked(index_name, dropped_entries)) {
      /*
        The visible DROP state is already durable.  Large hidden truth-store
        cleanup can exceed tiny redo configurations, so a cleanup failure must
        not resurrect the index or make the completed DDL look rolled back.
      */
      vector_status::record_truth_store_persist_failure();
    }
    if (artifacts != nullptr) {
      artifacts->index_name = artifact_plan.index_name;
      artifacts->mode = artifact_plan.mode;
      artifacts->provider = artifact_plan.provider;
      artifacts->valid = true;
    }
    drop_artifacts.push_back(std::move(artifact_plan));
    if (cleanup_artifacts) vector_status::record_index_drop_success();
  }
  if (cleanup_artifacts) cleanup_drop_artifacts(drop_artifacts);
  return true;
}

bool drop_index(const std::string &index_name) {
  return drop_index_impl(index_name, nullptr, true);
}

bool drop_index(const std::string &index_name,
                dropped_index_artifacts *artifacts) {
  return drop_index_impl(index_name, artifacts, false);
}

void cleanup_dropped_index_artifacts(
    const dropped_index_artifacts &artifacts) {
  if (!artifacts.valid) return;
  (void)vector_index::remove_backend_artifacts(artifacts.index_name,
                                               artifacts.mode,
                                               artifacts.provider);
  vector_status::record_index_drop_success();
}

bool drop_indexes_for_table(const std::string &db_name,
                            const std::string &table_name) {
  std::vector<drop_artifact_plan> drop_artifacts;
  size_t dropped_count = 0;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!g_metadata_loaded) return true;
    if (!ensure_metadata_loaded_locked()) return false;
    std::vector<vector_index_metadata_store::metadata_row> metadata_before;
    vector_index::index_service::committed_state committed_before;
    std::vector<vector_index_metadata_store::change_log_row> change_log_before;
    std::vector<std::string> lagging_indexes_before;
    if (!snapshot_runtime_state_locked(&metadata_before, &committed_before,
                                       &change_log_before,
                                       &lagging_indexes_before)) {
      return false;
    }

    std::vector<std::string> index_names;
    if (!g_index_service.list_indexes(&index_names)) return false;

    for (const std::string &index_name : index_names) {
      if (!binding_matches_table(binding_for_index_locked(index_name), db_name,
                                 table_name)) {
        continue;
      }

      drop_artifact_plan artifact_plan;
      if (!snapshot_drop_artifact_plan_locked(index_name, &artifact_plan))
        return false;
      vector_status::record_index_drop_request();
      if (!g_index_service.unregister_index(index_name)) return false;
      erase_index_binding_and_owner_schema_locked(index_name);
      prune_change_log_for_index_locked(index_name);
      drop_artifacts.push_back(std::move(artifact_plan));
      ++dropped_count;
    }

    if (dropped_count == 0) return true;
    if (!persist_registry_state_locked()) {
      vector_status::record_truth_store_persist_failure();
      if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                         change_log_before,
                                         lagging_indexes_before)) {
        return false;
      }
      return false;
    }
    for (size_t i = 0; i < dropped_count; ++i) {
      vector_status::record_index_drop_success();
    }
  }
  cleanup_drop_artifacts(drop_artifacts);
  return true;
}

bool drop_indexes_for_database(const std::string &db_name) {
  std::vector<drop_artifact_plan> drop_artifacts;
  size_t dropped_count = 0;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!g_metadata_loaded) return true;
    if (!ensure_metadata_loaded_locked()) return false;
    std::vector<vector_index_metadata_store::metadata_row> metadata_before;
    vector_index::index_service::committed_state committed_before;
    std::vector<vector_index_metadata_store::change_log_row> change_log_before;
    std::vector<std::string> lagging_indexes_before;
    if (!snapshot_runtime_state_locked(&metadata_before, &committed_before,
                                       &change_log_before,
                                       &lagging_indexes_before)) {
      return false;
    }

    std::vector<std::string> index_names;
    if (!g_index_service.list_indexes(&index_names)) return false;

    for (const std::string &index_name : index_names) {
      if (owner_schema_for_index_locked(index_name) != db_name) {
        continue;
      }

      drop_artifact_plan artifact_plan;
      if (!snapshot_drop_artifact_plan_locked(index_name, &artifact_plan))
        return false;
      vector_status::record_index_drop_request();
      if (!g_index_service.unregister_index(index_name)) return false;
      erase_index_binding_and_owner_schema_locked(index_name);
      prune_change_log_for_index_locked(index_name);
      drop_artifacts.push_back(std::move(artifact_plan));
      ++dropped_count;
    }

    if (dropped_count == 0) return true;
    if (!persist_registry_state_locked()) {
      vector_status::record_truth_store_persist_failure();
      if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                         change_log_before,
                                         lagging_indexes_before)) {
        return false;
      }
      return false;
    }
    for (size_t i = 0; i < dropped_count; ++i) {
      vector_status::record_index_drop_success();
    }
  }
  cleanup_drop_artifacts(drop_artifacts);
  return true;
}

bool reset_mapped_indexes_for_table(const std::string &db_name,
                                    const std::string &table_name) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!g_metadata_loaded) return true;
  if (!ensure_metadata_loaded_locked()) return false;

  std::vector<std::string> index_names;
  if (!g_index_service.list_indexes(&index_names)) return false;

  std::vector<mapped_index_reset_spec> reset_specs;
  reset_specs.reserve(index_names.size());
  for (const std::string &index_name : index_names) {
    const index_binding binding = binding_for_index_locked(index_name);
    if (!binding_matches_table(binding, db_name, table_name)) {
      continue;
    }

    mapped_index_reset_spec spec;
    spec.index_name = index_name;
    spec.binding = binding;
    vector_index::index_service::index_config config;
    bool supports_mutations = false;
    size_t entry_count = 0;
    size_t committed_entry_count = 0;
    if (!g_index_service.describe_index(index_name, &config,
                                        &supports_mutations, &entry_count,
                                        &committed_entry_count)) {
      return false;
    }
    spec.info.dimension = config.dimension;
    spec.info.metric = vector_index::metric_to_string(config.metric);
    spec.info.mode = vector_index::backend_mode_to_string(config.mode);
    spec.info.provider =
        vector_index::backend_provider_to_string(config.provider);
    spec.info.search_ef = config.search_ef;
    spec.info.hnsw_m = config.hnsw_m;
    spec.info.hnsw_ef_construction = config.hnsw_ef_construction;
    spec.info.hnsw_build_threads = config.hnsw_build_threads;
    spec.info.faiss_nlist = config.faiss_nlist;
    spec.info.faiss_nprobe = config.faiss_nprobe;
    spec.info.faiss_pq_m = config.faiss_pq_m;
    spec.info.faiss_pq_bits = config.faiss_pq_bits;
    spec.info.faiss_build_threads = config.faiss_build_threads;
    spec.info.diskann_max_degree = config.diskann_max_degree;
    spec.info.diskann_build_complexity = config.diskann_build_complexity;
    spec.info.diskann_build_threads = config.diskann_build_threads;
    spec.info.diskann_build_mode_value = config.diskann_build_mode_value;
    spec.info.diskann_build_mode_specified =
        config.diskann_build_mode_specified;
    spec.info.diskann_search_complexity = config.diskann_search_complexity;
    spec.info.diskann_search_beamwidth = config.diskann_search_beamwidth;
    spec.info.diskann_pq_code_budget_size =
        config.diskann_pq_code_budget_size;
    reset_specs.push_back(std::move(spec));
  }

  if (reset_specs.empty()) return true;

  std::vector<vector_index_metadata_store::metadata_row> metadata_before;
  vector_index::index_service::committed_state committed_before;
  std::vector<vector_index_metadata_store::change_log_row> change_log_before;
  std::vector<std::string> lagging_indexes_before;
  if (!snapshot_runtime_state_locked(&metadata_before, &committed_before,
                                     &change_log_before,
                                     &lagging_indexes_before)) {
    return false;
  }

  for (const mapped_index_reset_spec &spec : reset_specs) {
    if (!g_index_service.drop_index(spec.index_name)) return false;
    erase_index_binding_and_owner_schema_locked(spec.index_name);
    if (!g_index_service.register_index_from_strings(
            spec.index_name, spec.info.dimension, spec.info.metric,
            spec.info.mode, spec.info.provider)) {
      return false;
    }
    set_index_binding_and_owner_schema_locked(
        spec.index_name, spec.binding, spec.binding.schema_name);
    if (!apply_index_tuning_locked(spec.index_name, spec.info)) return false;
  }

  refresh_committed_snapshot_rows_locked();
  if (!persist_registry_state_locked()) {
    vector_status::record_truth_store_persist_failure();
    return rollback_runtime_state_locked(metadata_before, committed_before,
                                         change_log_before,
                                         lagging_indexes_before);
  }
  return true;
}

bool rename_indexes_for_table(const std::string &old_db_name,
                              const std::string &old_table_name,
                              const std::string &new_db_name,
                              const std::string &new_table_name) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!g_metadata_loaded) return true;
  if (!ensure_metadata_loaded_locked()) return false;

  std::vector<std::string> index_names;
  if (!g_index_service.list_indexes(&index_names)) return false;

  std::unordered_map<std::string, std::string> rename_map;
  rename_map.reserve(index_names.size());
  std::unordered_set<std::string> new_index_names;
  new_index_names.reserve(index_names.size());
  bool has_renamed_index = false;

  for (const std::string &index_name : index_names) {
    std::string renamed_index_name;
    const index_binding binding = binding_for_index_locked(index_name);
    if (binding_matches_table(binding, old_db_name, old_table_name)) {
      if (!vector_index_identity::replace_table_name(
              index_name, old_db_name, old_table_name, new_db_name,
              new_table_name, &renamed_index_name)) {
        return false;
      }
    } else {
      renamed_index_name = index_name;
    }
    if (renamed_index_name != index_name) {
      has_renamed_index = true;
      rename_map.emplace(index_name, renamed_index_name);
    }
    if (!new_index_names.insert(renamed_index_name).second) return false;
  }

  if (!has_renamed_index) return true;

  std::vector<vector_index_metadata_store::metadata_row> metadata_before;
  if (!snapshot_metadata_locked(&metadata_before)) return false;

  std::vector<std::pair<std::string, std::string>> rename_pairs(
      rename_map.begin(), rename_map.end());
  std::sort(
      rename_pairs.begin(), rename_pairs.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

  std::vector<std::pair<std::string, std::string>> applied_pairs;
  applied_pairs.reserve(rename_pairs.size());
  auto rollback_applied_pairs = [&applied_pairs, &old_db_name]() {
    for (auto it = applied_pairs.rbegin(); it != applied_pairs.rend(); ++it) {
      if (!g_index_service.rename_index(it->second, it->first)) {
        vector_status::record_runtime_state_rollback_failure();
        return false;
      }
      rename_index_binding_and_owner_schema_locked(it->second, it->first,
                                                   old_db_name);
    }
    vector_status::record_runtime_state_rollback();
    return true;
  };

  for (const auto &pair : rename_pairs) {
    if (!g_index_service.rename_index(pair.first, pair.second)) {
      if (!rollback_applied_pairs()) return false;
      return false;
    }
    rename_index_binding_and_owner_schema_locked(pair.first, pair.second,
                                                 new_db_name);
    applied_pairs.push_back(pair);
  }

  const std::vector<vector_index_metadata_store::change_log_row>
      change_log_before = g_change_log_rows;
  for (auto &row : g_change_log_rows) {
    const auto it = rename_map.find(row.index_name);
    if (it != rename_map.end()) row.index_name = it->second;
  }
  g_manifest_change_log_checkpoint = g_change_log_rows.size();
  refresh_committed_snapshot_rows_locked();

  if (!persist_registry_state_locked()) {
    vector_status::record_truth_store_persist_failure();
    g_change_log_rows = change_log_before;
    g_manifest_change_log_checkpoint = g_change_log_rows.size();
    if (!rollback_applied_pairs()) return false;
    refresh_committed_snapshot_rows_locked();
    refresh_manifest_status_locked();
    return false;
  }

  for (const auto &row : metadata_before) {
    const auto it = rename_map.find(row.index_name);
    if (it == rename_map.end()) continue;
    (void)vector_index::remove_backend_artifacts(row.index_name, row.mode,
                                                 row.provider);
  }

  return true;
}

bool drop_index_for_column(const std::string &db_name,
                           const std::string &table_name,
                           const std::string &column_name) {
  const std::string index_name =
      make_mapped_index_name(db_name, table_name, column_name);

  std::vector<drop_artifact_plan> drop_artifacts;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!g_metadata_loaded) return true;
    if (!ensure_metadata_loaded_locked()) return false;

    const index_binding binding = binding_for_index_locked(index_name);
    if (!binding_matches_column(binding, db_name, table_name, column_name)) {
      return true;
    }

    drop_artifact_plan artifact_plan;
    if (!snapshot_drop_artifact_plan_locked(index_name, &artifact_plan)) {
      return true;
    }

    std::vector<vector_index_metadata_store::metadata_row> metadata_before;
    vector_index::index_service::committed_state committed_before;
    std::vector<vector_index_metadata_store::change_log_row> change_log_before;
    std::vector<std::string> lagging_indexes_before;
    if (!snapshot_runtime_state_locked(&metadata_before, &committed_before,
                                       &change_log_before,
                                       &lagging_indexes_before)) {
      return false;
    }

    vector_status::record_index_drop_request();
    if (!g_index_service.unregister_index(index_name)) return false;
    erase_index_binding_and_owner_schema_locked(index_name);
    prune_change_log_for_index_locked(index_name);

    if (!persist_registry_state_locked()) {
      vector_status::record_truth_store_persist_failure();
      if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                         change_log_before,
                                         lagging_indexes_before)) {
        return false;
      }
      return false;
    }

    vector_status::record_index_drop_success();
    drop_artifacts.push_back(std::move(artifact_plan));
  }

  cleanup_drop_artifacts(drop_artifacts);
  return true;
}

bool rename_index_for_column(const std::string &db_name,
                             const std::string &table_name,
                             const std::string &old_column_name,
                             const std::string &new_column_name) {
  const std::string old_index_name =
      make_mapped_index_name(db_name, table_name, old_column_name);
  const std::string new_index_name =
      make_mapped_index_name(db_name, table_name, new_column_name);
  if (old_index_name == new_index_name) return true;

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!g_metadata_loaded) return true;
  if (!ensure_metadata_loaded_locked()) return false;

  const index_binding binding = binding_for_index_locked(old_index_name);
  if (!binding_matches_column(binding, db_name, table_name, old_column_name)) {
    return true;
  }

  vector_index::index_service::index_config old_config;
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
          old_index_name, &old_config, &supports_mutations, &entry_count,
          &committed_entry_count, &lifecycle_state, &lifecycle_version,
          &last_error_code, &last_error_ts, nullptr, &recover_fallback_count,
          &last_recover_fallback_ts)) {
    return true;
  }

  vector_index::index_service::index_config new_config;
  if (g_index_service.describe_index(
          new_index_name, &new_config, &supports_mutations, &entry_count,
          &committed_entry_count, &lifecycle_state, &lifecycle_version,
          &last_error_code, &last_error_ts, nullptr, &recover_fallback_count,
          &last_recover_fallback_ts)) {
    return false;
  }

  std::vector<vector_index_metadata_store::metadata_row> metadata_before;
  vector_index::index_service::committed_state committed_before;
  std::vector<vector_index_metadata_store::change_log_row> change_log_before;
  std::vector<std::string> lagging_indexes_before;
  if (!snapshot_runtime_state_locked(&metadata_before, &committed_before,
                                     &change_log_before,
                                     &lagging_indexes_before)) {
    return false;
  }

  if (!g_index_service.rename_index(old_index_name, new_index_name)) {
    return false;
  }
  rename_index_binding_and_owner_schema_locked(old_index_name, new_index_name,
                                               db_name);
  for (auto &row : g_change_log_rows) {
    if (row.index_name == old_index_name) row.index_name = new_index_name;
  }
  g_manifest_change_log_checkpoint = g_change_log_rows.size();
  refresh_committed_snapshot_rows_locked();

  if (!persist_registry_state_locked()) {
    vector_status::record_truth_store_persist_failure();
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return false;
    }
    return false;
  }

  (void)vector_index::remove_backend_artifacts(old_index_name, old_config.mode,
                                               old_config.provider);
  return true;
}

bool begin_bulk_load(const std::string &index_name) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  runtime_state_snapshot snapshot;
  if (!capture_runtime_state_locked(&snapshot)) return false;

  if (!g_index_service.begin_bulk_load(index_name)) {
    return rollback_runtime_state_and_fail_locked(snapshot);
  }
  return persist_registry_state_or_rollback_locked(snapshot, true);
}

bool bulk_upsert_from_reader(
    const std::string &index_name,
    const vector_index::index_service::bulk_load_reader &reader,
    const vector_index::index_service::bulk_load_options &options,
    std::string *error) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  runtime_state_snapshot snapshot;
  if (!capture_runtime_state_locked(&snapshot)) return false;

  if (!g_index_service.bulk_upsert_from_reader(index_name, reader, options,
                                               error)) {
    return rollback_runtime_state_and_fail_locked(snapshot);
  }
  return persist_registry_state_or_rollback_locked(snapshot, true);
}

bool bulk_upsert_from_raw_files(
    const std::string &index_name, const std::string &vector_filename,
    const std::string &docid_filename,
    const vector_index::index_service::bulk_load_options &options,
    uint64_t *loaded_rows, std::string *error) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  runtime_state_snapshot snapshot;
  if (!capture_runtime_state_locked(&snapshot)) return false;

  if (!g_index_service.bulk_upsert_from_raw_files(
          index_name, vector_filename, docid_filename, options, loaded_rows,
          error)) {
    return rollback_runtime_state_and_fail_locked(snapshot);
  }
  return persist_registry_state_or_rollback_locked(snapshot, true);
}

bool bulk_build_index(const std::string &index_name) {
  return rebuild_index(index_name);
}

bool rebuild_index(const std::string &index_name) {
  vector_status::record_rebuild_request();
  lifecycle_backend_plan plan;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    if (!snapshot_backend_plan_locked(index_name, &plan)) return false;
  }
  if (uses_standalone_source(plan)) {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    return rebuild_standalone_plan_locked(plan);
  }

  std::unique_ptr<vector_index::backend> rebuilt = build_rebuilt_backend(plan);
  if (rebuilt == nullptr) return false;

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  if (!validate_backend_plan_locked(plan)) return false;

  runtime_state_snapshot snapshot;
  if (!capture_runtime_state_locked(&snapshot)) return false;
  const bool ok = g_index_service.install_rebuilt_index(
      index_name, plan.entries, std::move(rebuilt));
  if (!ok) {
    return rollback_runtime_state_and_fail_locked(snapshot);
  }
  if (!persist_registry_state_or_rollback_locked(snapshot, true)) return false;
  return evict_committed_cache_to_budget_locked();
}

bool replace_committed_entries(
    const std::string &index_name,
    const vector_index::index_service::committed_entries &entries) {
  vector_index::index_service::index_config config;
  vector_index::index_service::committed_entries before_entries;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    if (!g_index_service.describe_index(index_name, &config, nullptr, nullptr,
                                        nullptr)) {
      return false;
    }
    vector_index::index_service::committed_state snapshot_state;
    if (!g_index_service.snapshot_committed_state(&snapshot_state))
      return false;
    auto state_it = snapshot_state.find(index_name);
    if (state_it == snapshot_state.end()) {
      before_entries.clear();
    } else {
      before_entries = state_it->second;
    }
  }

  std::unique_ptr<vector_index::backend> rebuilt =
      build_backend_from_config(index_name, config);
  if (rebuilt == nullptr || !rebuilt->rebuild_from_committed_entries(entries)) {
    return false;
  }

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  const bool ok = g_index_service.install_rebuilt_index(index_name, entries,
                                                        std::move(rebuilt));
  if (!ok) return false;
  if (!persist_registry_state_locked()) {
    vector_status::record_truth_store_persist_failure();
    std::unique_ptr<vector_index::backend> rollback_backend =
        build_backend_from_config(index_name, config);
    if (rollback_backend == nullptr ||
        !rollback_backend->rebuild_from_committed_entries(before_entries) ||
        !g_index_service.install_rebuilt_index(index_name, before_entries,
                                               std::move(rollback_backend))) {
      return false;
    }
    return false;
  }
  return evict_committed_cache_to_budget_locked();
}

bool replace_committed_entries_preserve_lifecycle(
    const std::string &index_name,
    const vector_index::index_service::committed_entries &entries) {
  vector_index::index_service::committed_entries before_entries;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    vector_index::index_service::committed_state snapshot_state;
    if (!g_index_service.snapshot_committed_state(&snapshot_state))
      return false;
    auto state_it = snapshot_state.find(index_name);
    if (state_it == snapshot_state.end()) {
      before_entries.clear();
    } else {
      before_entries = state_it->second;
    }
  }

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  const bool ok = g_index_service.replace_committed_entries_preserve_lifecycle(
      index_name, entries);
  if (!ok) return false;
  if (!persist_registry_state_locked()) {
    vector_status::record_truth_store_persist_failure();
    if (!g_index_service.replace_committed_entries_preserve_lifecycle(
            index_name, before_entries)) {
      return false;
    }
    return false;
  }
  return evict_committed_cache_to_budget_locked();
}

bool replace_committed_entries_for_backfill(
    const std::string &index_name,
    const vector_index::index_service::committed_entries &entries) {
  vector_index::index_service::committed_entries before_entries;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    vector_index::index_service::committed_state snapshot_state;
    if (!g_index_service.snapshot_committed_state(&snapshot_state))
      return false;
    auto state_it = snapshot_state.find(index_name);
    if (state_it == snapshot_state.end()) {
      before_entries.clear();
    } else {
      before_entries = state_it->second;
    }
  }

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  const bool ok = g_index_service.replace_committed_entries_preserve_lifecycle(
      index_name, entries);
  if (!ok) return false;
  if (!persist_backfill_committed_entries_locked(index_name, entries)) {
    vector_status::record_truth_store_persist_failure();
    if (!g_index_service.replace_committed_entries_preserve_lifecycle(
            index_name, before_entries)) {
      return false;
    }
    return false;
  }
  return evict_committed_cache_to_budget_locked();
}

bool finish_backfill(const std::string &index_name,
                     const std::string &lifecycle_state) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  runtime_state_snapshot snapshot;
  if (!capture_runtime_state_locked(&snapshot)) return false;
  if (!g_index_service.set_lifecycle_state(index_name, lifecycle_state)) {
    return false;
  }

  if (!persist_metadata_manifest_or_rollback_locked(snapshot, false))
    return false;
  return true;
}

bool set_lifecycle_state(const std::string &index_name,
                         const std::string &lifecycle_state) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  runtime_state_snapshot snapshot;
  if (!capture_runtime_state_locked(&snapshot)) return false;
  if (!g_index_service.set_lifecycle_state(index_name, lifecycle_state)) {
    return false;
  }
  return persist_registry_state_or_rollback_locked(snapshot, true);
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool build_backend_from_config_for_testing(
    const std::string &index_name,
    const vector_index::index_service::index_config &config) {
  return build_backend_from_config(index_name, config) != nullptr;
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

bool recover_index(const std::string &index_name) {
  vector_status::record_recover_request();
  lifecycle_backend_plan plan;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    if (!snapshot_backend_plan_locked(index_name, &plan)) return false;
  }
  if (uses_standalone_source(plan)) {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    return recover_standalone_plan_locked(plan);
  }

  bool used_recover_fallback = false;
  std::unique_ptr<vector_index::backend> recovered =
      build_recovered_backend(plan, &used_recover_fallback);
  if (recovered == nullptr) return false;

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  if (!validate_backend_plan_locked(plan)) return false;

  runtime_state_snapshot snapshot;
  if (!capture_runtime_state_locked(&snapshot)) return false;
  if (!g_index_service.install_recovered_index(
          index_name, plan.entries, std::move(recovered),
          used_recover_fallback)) {
    return rollback_runtime_state_and_fail_locked(snapshot);
  }
  if (!persist_registry_state_or_rollback_locked(snapshot, true)) return false;
  return evict_committed_cache_to_budget_locked();
}

bool set_search_ef(const std::string &index_name, uint32_t search_ef) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return apply_persisted_index_config_change_locked(
      index_name,
      [&](vector_index::index_service::index_config *candidate) {
        candidate->search_ef = search_ef;
      },
      [&] { return g_index_service.set_search_ef(index_name, search_ef); });
}

bool set_hnsw_build_params(const std::string &index_name, uint32_t hnsw_m,
                           uint32_t hnsw_ef_construction) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return apply_persisted_index_config_change_locked(
      index_name,
      [&](vector_index::index_service::index_config *candidate) {
        candidate->hnsw_m = hnsw_m;
        candidate->hnsw_ef_construction = hnsw_ef_construction;
      },
      [&] {
        return g_index_service.set_hnsw_build_params(index_name, hnsw_m,
                                                     hnsw_ef_construction);
      });
}

bool set_faiss_ivf_params(const std::string &index_name, uint32_t faiss_nlist,
                          uint32_t faiss_nprobe) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return apply_persisted_index_config_change_locked(
      index_name,
      [&](vector_index::index_service::index_config *candidate) {
        candidate->faiss_nlist = faiss_nlist;
        candidate->faiss_nprobe = faiss_nprobe;
      },
      [&] {
        return g_index_service.set_faiss_ivf_params(index_name, faiss_nlist,
                                                    faiss_nprobe);
      });
}

bool set_faiss_ivf_pq_params(const std::string &index_name,
                             uint32_t faiss_nlist, uint32_t faiss_nprobe,
                             uint32_t faiss_pq_m, uint32_t faiss_pq_bits) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return apply_persisted_index_config_change_locked(
      index_name,
      [&](vector_index::index_service::index_config *candidate) {
        candidate->faiss_nlist = faiss_nlist;
        candidate->faiss_nprobe = faiss_nprobe;
        candidate->faiss_pq_m = faiss_pq_m;
        candidate->faiss_pq_bits = faiss_pq_bits;
      },
      [&] {
        return g_index_service.set_faiss_ivf_pq_params(
            index_name, faiss_nlist, faiss_nprobe, faiss_pq_m, faiss_pq_bits);
      });
}

bool set_diskann_build_params(const std::string &index_name,
                              uint32_t diskann_max_degree,
                              uint32_t diskann_build_complexity,
                              uint32_t diskann_build_threads) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return apply_persisted_index_config_change_locked(
      index_name,
      [&](vector_index::index_service::index_config *candidate) {
        candidate->diskann_max_degree = diskann_max_degree;
        candidate->diskann_build_complexity = diskann_build_complexity;
        candidate->diskann_build_threads = diskann_build_threads;
      },
      [&] {
        return g_index_service.set_diskann_build_params(
            index_name, diskann_max_degree, diskann_build_complexity,
            diskann_build_threads);
      });
}

bool set_diskann_build_threads(const std::string &index_name,
                               uint32_t diskann_build_threads) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return apply_persisted_index_config_change_locked(
      index_name,
      [&](vector_index::index_service::index_config *candidate) {
        candidate->diskann_build_threads = diskann_build_threads;
      },
      [&] {
        return g_index_service.set_diskann_build_threads(
            index_name, diskann_build_threads);
      });
}

bool set_diskann_build_mode(
    const std::string &index_name,
    vector_index::diskann_build_mode diskann_build_mode_value) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  vector_index::index_service::index_config config;
  if (!snapshot_index_config_locked(index_name, &config)) return false;
  if (config.provider != vector_index::backend_provider::kDiskAnn ||
      config.mode != vector_index::backend_mode::kExternal) {
    return diskann_build_mode_value == vector_index::diskann_build_mode::kAuto;
  }
  return apply_persisted_index_config_change_locked(
      index_name,
      [&](vector_index::index_service::index_config *candidate) {
        candidate->diskann_build_mode_value = diskann_build_mode_value;
        candidate->diskann_build_mode_specified = true;
      },
      [&] {
        return g_index_service.set_diskann_build_mode(
            index_name, diskann_build_mode_value);
      });
}

bool set_index_consistency_mode(
    const std::string &index_name,
    vector_index::index_consistency_mode consistency_mode) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  vector_index::index_service::index_config config;
  if (!snapshot_index_config_locked(index_name, &config)) return false;
  if (consistency_mode == vector_index::index_consistency_mode::kStandalone) {
    const index_binding binding = binding_for_index_locked(index_name);
    if (!binding.schema_name.empty() || !binding.table_name.empty() ||
        !binding.column_name.empty()) {
      return false;
    }
  }
  return apply_persisted_index_config_change_locked(
      index_name,
      [&](vector_index::index_service::index_config *candidate) {
        candidate->consistency_mode = consistency_mode;
      },
      [&] {
        return g_index_service.set_index_consistency_mode(index_name,
                                                          consistency_mode);
      });
}

bool set_diskann_search_complexity(const std::string &index_name,
                                   uint32_t diskann_search_complexity) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return apply_persisted_index_config_change_locked(
      index_name,
      [&](vector_index::index_service::index_config *candidate) {
        candidate->diskann_search_complexity = diskann_search_complexity;
      },
      [&] {
        return g_index_service.set_diskann_search_complexity(
            index_name, diskann_search_complexity);
      });
}

bool set_diskann_search_beamwidth(const std::string &index_name,
                                  uint32_t diskann_search_beamwidth) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return apply_persisted_index_config_change_locked(
      index_name,
      [&](vector_index::index_service::index_config *candidate) {
        candidate->diskann_search_beamwidth = diskann_search_beamwidth;
      },
      [&] {
        return g_index_service.set_diskann_search_beamwidth(
            index_name, diskann_search_beamwidth);
      });
}

bool set_diskann_pq_code_budget_size(const std::string &index_name,
                                     uint64_t diskann_pq_code_budget_size) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return apply_persisted_index_config_change_locked(
      index_name,
      [&](vector_index::index_service::index_config *candidate) {
        candidate->diskann_pq_code_budget_size =
            diskann_pq_code_budget_size;
      },
      [&] {
        return g_index_service.set_diskann_pq_code_budget_size(
            index_name, diskann_pq_code_budget_size);
      });
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool detail::index_config_matches_for_testing(
    const vector_index::index_service::index_config &lhs,
    const vector_index::index_service::index_config &rhs) {
  return index_config_matches(lhs, rhs);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

bool rebuild_all_indexes(size_t *rebuilt_count) {
  vector_status::record_rebuild_all_request();
  if (rebuilt_count == nullptr) return false;

  std::vector<lifecycle_backend_plan> plans;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    *rebuilt_count = 0;
    if (!snapshot_all_backend_plans_locked(&plans)) return false;
  }

  std::vector<lifecycle_backend_plan> standalone_plans;
  std::vector<prepared_lifecycle_backend> prepared_backends;
  prepared_backends.reserve(plans.size());
  for (const lifecycle_backend_plan &plan : plans) {
    if (uses_standalone_source(plan)) {
      standalone_plans.push_back(plan);
      continue;
    }
    prepared_lifecycle_backend prepared;
    prepared.plan = plan;
    prepared.backend = build_rebuilt_backend(plan);
    if (prepared.backend == nullptr) {
      return false;
    }
    prepared_backends.push_back(std::move(prepared));
  }

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  vector_index::index_service::committed_state current_state;
  if (!g_index_service.snapshot_committed_state(&current_state)) return false;
  for (const prepared_lifecycle_backend &prepared : prepared_backends) {
    if (!validate_backend_plan_against_state_locked(prepared.plan,
                                                   current_state)) {
      return false;
    }
  }
  for (const lifecycle_backend_plan &plan : standalone_plans) {
    if (!validate_backend_plan_against_state_locked(plan, current_state)) {
      return false;
    }
  }
  runtime_state_snapshot snapshot;
  if (!capture_runtime_state_locked(&snapshot)) return false;
  for (prepared_lifecycle_backend &prepared : prepared_backends) {
    if (!g_index_service.install_rebuilt_index(
            prepared.plan.index_name, prepared.plan.entries,
            std::move(prepared.backend))) {
      return rollback_runtime_state_and_fail_locked(snapshot);
    }
  }
  for (const lifecycle_backend_plan &plan : standalone_plans) {
    if (!g_index_service.rebuild_index(plan.index_name)) {
      const bool has_segment_tasks =
          refresh_segment_tasks_from_service_locked(plan.index_name);
      if (has_segment_tasks) (void)persist_segment_tasks_locked();
      return rollback_runtime_state_and_fail_locked(snapshot);
    }
    if (!refresh_segment_tasks_from_service_locked(plan.index_name)) {
      return rollback_runtime_state_and_fail_locked(snapshot);
    }
  }
  if (!persist_registry_state_or_rollback_locked(snapshot, true)) return false;
  if (!evict_committed_cache_to_budget_locked()) return false;
  *rebuilt_count = prepared_backends.size() + standalone_plans.size();
  return true;
}

bool recover_all_indexes(size_t *recovered_count) {
  vector_status::record_recover_all_request();
  if (recovered_count == nullptr) return false;

  std::vector<lifecycle_backend_plan> plans;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    *recovered_count = 0;
    if (!snapshot_all_backend_plans_locked(&plans)) return false;
  }

  std::vector<lifecycle_backend_plan> standalone_plans;
  std::vector<prepared_lifecycle_backend> prepared_backends;
  prepared_backends.reserve(plans.size());
  for (const lifecycle_backend_plan &plan : plans) {
    if (uses_standalone_source(plan)) {
      standalone_plans.push_back(plan);
      continue;
    }
    prepared_lifecycle_backend prepared;
    prepared.plan = plan;
    prepared.backend =
        build_recovered_backend(plan, &prepared.used_recover_fallback);
    if (prepared.backend == nullptr) {
      return false;
    }
    prepared_backends.push_back(std::move(prepared));
  }

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  vector_index::index_service::committed_state current_state;
  if (!g_index_service.snapshot_committed_state(&current_state)) return false;
  for (const prepared_lifecycle_backend &prepared : prepared_backends) {
    if (!validate_backend_plan_against_state_locked(prepared.plan,
                                                   current_state)) {
      return false;
    }
  }
  for (const lifecycle_backend_plan &plan : standalone_plans) {
    if (!validate_backend_plan_against_state_locked(plan, current_state)) {
      return false;
    }
  }
  runtime_state_snapshot snapshot;
  if (!capture_runtime_state_locked(&snapshot)) return false;
  for (prepared_lifecycle_backend &prepared : prepared_backends) {
    if (!g_index_service.install_recovered_index(
            prepared.plan.index_name, prepared.plan.entries,
            std::move(prepared.backend), prepared.used_recover_fallback)) {
      return rollback_runtime_state_and_fail_locked(snapshot);
    }
  }
  for (const lifecycle_backend_plan &plan : standalone_plans) {
    if (!g_index_service.recover_index(plan.index_name)) {
      return rollback_runtime_state_and_fail_locked(snapshot);
    }
  }
  if (!persist_registry_state_or_rollback_locked(snapshot, true)) return false;
  if (!evict_committed_cache_to_budget_locked()) return false;
  *recovered_count = prepared_backends.size() + standalone_plans.size();
  return true;
}

bool get_index_info(const std::string &index_name, index_info *info) {
  if (info == nullptr) return false;

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  vector_index::index_service::index_config config;
  bool supports_mutations = false;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  std::string lifecycle_state;
  uint64_t lifecycle_version = 0;
  uint32_t last_error_code = 0;
  uint64_t last_error_ts = 0;
  if (!g_index_service.describe_index(
          index_name, &config, &supports_mutations, &entry_count,
          &committed_entry_count, &lifecycle_state, &lifecycle_version,
          &last_error_code, &last_error_ts, &info->last_apply_latency_ms,
          &info->recover_fallback_count, &info->last_recover_fallback_ts,
          &info->external_manifest_present,
          &info->external_manifest_generation, &info->build_diagnostics)) {
    return false;
  }
  vector_index::index_service::build_pipeline_snapshot pipeline_snapshot;
  if (g_index_service.describe_build_pipeline(index_name,
                                              &pipeline_snapshot)) {
    info->build_pipeline_mode = pipeline_snapshot.mode;
    info->build_pipeline_decision = pipeline_snapshot.decision;
    info->build_pipeline_trigger = pipeline_snapshot.trigger;
    info->build_pipeline_rows = pipeline_snapshot.row_count;
    info->build_pipeline_payload_size = pipeline_snapshot.payload_size;
    info->build_pipeline_raw_segments = pipeline_snapshot.raw_segment_count;
  }

  info->dimension = config.dimension;
  info->metric = vector_index::metric_to_string(config.metric);
  info->mode = vector_index::backend_mode_to_string(config.mode);
  info->provider = vector_index::backend_provider_to_string(config.provider);
  info->consistency_mode =
      vector_index::index_consistency_mode_to_string(config.consistency_mode);
  info->truth_store_enabled =
      config.consistency_mode ==
      vector_index::index_consistency_mode::kTransactional;
  info->standalone_ingest_memory_bytes =
      g_index_service.standalone_ingest_memory_bytes(index_name);
  info->standalone_segment_count =
      g_index_service.standalone_segment_count(index_name);
  info->standalone_segment_bytes =
      g_index_service.standalone_segment_bytes(index_name);
  info->standalone_raw_segment_count =
      g_index_service.standalone_raw_segment_count(index_name);
  info->standalone_raw_segment_bytes =
      g_index_service.standalone_raw_segment_bytes(index_name);
  info->build_source = info->truth_store_enabled
                           ? "truth_store"
                           : g_index_service.standalone_build_source(index_name);
  info->backend_variant = config.backend_variant;
  info->search_ef = config.search_ef;
  info->hnsw_m = config.hnsw_m;
  info->hnsw_ef_construction = config.hnsw_ef_construction;
  info->hnsw_build_threads = config.hnsw_build_threads;
  info->faiss_nlist = config.faiss_nlist;
  info->faiss_nprobe = config.faiss_nprobe;
  info->faiss_pq_m = config.faiss_pq_m;
  info->faiss_pq_bits = config.faiss_pq_bits;
  info->faiss_build_threads = config.faiss_build_threads;
  info->diskann_max_degree = config.diskann_max_degree;
  info->diskann_build_complexity = config.diskann_build_complexity;
  info->diskann_build_threads = config.diskann_build_threads;
  info->diskann_build_mode_value = config.diskann_build_mode_value;
  info->diskann_build_mode_specified = config.diskann_build_mode_specified;
  info->diskann_search_complexity = config.diskann_search_complexity;
  info->diskann_search_beamwidth = config.diskann_search_beamwidth;
  info->diskann_pq_code_budget_size = config.diskann_pq_code_budget_size;
  const index_binding binding = binding_for_index_locked(index_name);
  info->owner_schema = owner_schema_for_index_locked(index_name);
  info->schema_name = binding.schema_name;
  info->table_name = binding.table_name;
  info->column_name = binding.column_name;
  info->lifecycle_state = lifecycle_state;
  info->lifecycle_version = lifecycle_version;
  info->last_error_code = last_error_code;
  info->last_error_ts = last_error_ts;
  info->supports_mutations = supports_mutations;
  info->entry_count = entry_count;
  info->committed_entry_count = committed_entry_count;
  return true;
}

bool list_indexes(std::vector<std::string> *index_names) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return g_index_service.list_indexes(index_names);
}

bool metadata_loaded() {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  return g_metadata_loaded;
}

}  // namespace vector_index_registry
