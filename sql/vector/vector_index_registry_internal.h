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

#ifndef SQL_VECTOR_INDEX_REGISTRY_INTERNAL_INCLUDED
#define SQL_VECTOR_INDEX_REGISTRY_INTERNAL_INCLUDED

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_diskann_generation_store.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_truth_store.h"

namespace vector_index_registry::detail {

enum class recovery_action_type { kCommit, kRollback, kPreparedInTc };

struct recovery_action {
  XID xid;
  recovery_action_type type{recovery_action_type::kCommit};
  bool conditional{false};
};

struct runtime_state_snapshot {
  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  vector_index::index_service::committed_state committed_state;
  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  std::vector<std::string> lagging_index_names;
};

struct thd_txn_context {
  uint64_t txn_id{0};
  uint64_t active_stmt_id{0};
  bool stmt_savepoint_active{false};
  bool attached_dml{false};
  bool publication_prepared{false};
  std::string stmt_savepoint_name;
  std::vector<vector_index_metadata_store::change_log_row>
      durable_change_log_rows;
  std::vector<vector_index_truth_store::publication_intent>
      publication_intents;
};

struct index_binding {
  std::string schema_name;
  std::string table_name;
  std::string column_name;
  std::string doc_id_column_name;
};

struct truth_recovery_state {
  std::unordered_set<std::string> pending_index_names;
  std::unordered_map<std::string, std::string> quarantine_identities;
};

struct mapped_index_reset_spec {
  std::string index_name;
  index_binding binding;
  vector_index_registry::index_info info;
};

struct persisted_commit_artifacts_snapshot {
  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  std::vector<vector_index_metadata_store::segment_task_row> segment_task_rows;
  vector_index_metadata_store::manifest_row manifest_row;
};

struct commit_runtime_snapshot {
  vector_index::index_service::committed_state committed_state;
  std::unordered_map<std::string,
                     vector_index::index_service::index_publication_state>
      publication_states;
  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  uint64_t manifest_change_log_checkpoint{0};
  uint64_t next_change_log_sequence{1};
};

inline constexpr size_t k_mapped_search_min_candidate_top_k = 32;

inline size_t mapped_search_candidate_limit(size_t top_k, size_t entry_count,
                                            size_t pending_count) {
  if (top_k == 0) return 0;

  const size_t total_candidates =
      vector_index::saturated_add_size(entry_count, pending_count);
  return std::min(total_candidates,
                  static_cast<size_t>(vector_index::k_max_search_top_k));
}

inline bool mapped_search_candidate_limit_is_hard(size_t entry_count,
                                                  size_t pending_count) {
  return vector_index::saturated_add_size(entry_count, pending_count) >
         static_cast<size_t>(vector_index::k_max_search_top_k);
}

inline size_t mapped_search_initial_candidate_top_k(size_t top_k,
                                                    size_t entry_count,
                                                    size_t pending_count) {
  const size_t limit =
      mapped_search_candidate_limit(top_k, entry_count, pending_count);
  if (limit == 0) return 0;

  const size_t requested =
      std::max(top_k, k_mapped_search_min_candidate_top_k);
  return std::min(requested, limit);
}

inline size_t mapped_search_next_candidate_top_k(size_t current_top_k,
                                                 size_t requested_top_k,
                                                 size_t candidate_limit) {
  if (current_top_k >= candidate_limit) return candidate_limit;

  const size_t max_value = std::numeric_limits<size_t>::max();
  const size_t doubled = (current_top_k > max_value / 2)
                             ? max_value
                             : current_top_k * 2;
  const size_t step =
      std::max(requested_top_k, k_mapped_search_min_candidate_top_k);
  const size_t stepped = (step > max_value - current_top_k)
                             ? max_value
                             : current_top_k + step;
  return std::min(candidate_limit, std::max(doubled, stepped));
}

extern std::shared_mutex g_registry_mutex;
extern vector_index::index_service g_index_service;
extern bool g_metadata_loaded;
extern registry_health_state g_registry_health;
extern std::string g_registry_failure_reason;
extern truth_recovery_state g_truth_recovery_state;
extern std::atomic<uint64_t> g_next_txn_id;
extern uint64_t g_manifest_version;
extern uint64_t g_manifest_metadata_checkpoint;
extern uint64_t g_manifest_committed_checkpoint;
extern uint64_t g_manifest_change_log_checkpoint;
extern uint64_t g_next_change_log_sequence;
extern std::vector<vector_index_metadata_store::change_log_row>
    g_change_log_rows;
extern std::vector<vector_index_metadata_store::prepared_change_row>
    g_prepared_change_rows;
extern std::vector<vector_index_metadata_store::segment_task_row>
    g_segment_task_rows;
extern std::vector<recovery_action> g_recovery_actions;
extern std::unordered_map<uint64_t, thd_txn_context> g_thd_txn_contexts;
extern std::unordered_map<std::string, index_binding> g_index_bindings;
extern std::unordered_map<std::string, std::string> g_index_owner_schemas;

void fail_stop_registry_locked(const char *reason);

xa_status_code apply_prepared_xid_locked(const XID &xid, bool commit,
                                         uint64_t thd_id_to_clear);

bool snapshot_runtime_state_locked(
    std::vector<vector_index_metadata_store::metadata_row> *metadata_rows,
    vector_index::index_service::committed_state *committed_state,
    std::vector<vector_index_metadata_store::change_log_row> *change_log_rows,
    std::vector<std::string> *lagging_index_names);
bool restore_runtime_state_locked(
    const std::vector<vector_index_metadata_store::metadata_row> &metadata_rows,
    const vector_index::index_service::committed_state &committed_state,
    const std::vector<vector_index_metadata_store::change_log_row>
        &change_log_rows,
    const std::vector<std::string> &lagging_index_names);
bool rollback_runtime_state_locked(
    const std::vector<vector_index_metadata_store::metadata_row> &metadata_rows,
    const vector_index::index_service::committed_state &committed_state,
    const std::vector<vector_index_metadata_store::change_log_row>
        &change_log_rows,
    const std::vector<std::string> &lagging_index_names);
bool rollback_runtime_state_locked(const runtime_state_snapshot &snapshot);
bool capture_runtime_state_locked(runtime_state_snapshot *snapshot);
bool evict_committed_cache_to_budget_locked();
bool persist_registry_state_locked();
bool persist_prepared_locked();
bool persist_segment_tasks_locked();
bool persist_index_config_manifest_locked(
    const std::string &index_name,
    const vector_index::index_service::index_config &config,
    const vector_index::index_service::index_publication_state &publication);
bool ensure_metadata_available_locked();
bool ensure_metadata_loaded_locked();
bool load_runtime_for_search(const std::string &index_name);
bool describe_publication_token_locked(
    const std::string &index_name,
    vector_index_truth_store::publication_token *token);
bool publication_tokens_equal(
    const vector_index_truth_store::publication_token &lhs,
    const vector_index_truth_store::publication_token &rhs);
bool fail_stop_truth_artifact_locked(
    vector_index_truth_store::truth_store *truth_store,
    const std::string &artifact_name, const std::string &reason,
    uint64_t generation);
bool persist_metadata_locked(size_t *row_count);
bool persist_committed_locked(size_t *row_count);
bool persist_committed_delta_locked(
    const std::vector<vector_index_metadata_store::change_log_row> &rows);
bool persist_change_log_locked();
bool persist_change_log_delta_locked(
    const std::vector<vector_index_metadata_store::change_log_row> &rows);
bool persist_manifest_locked();
void refresh_manifest_status_locked();
bool load_persisted_commit_artifacts_snapshot_locked(
    persisted_commit_artifacts_snapshot *snapshot);
bool restore_persisted_commit_artifacts_snapshot_locked(
    const persisted_commit_artifacts_snapshot &snapshot);
bool persist_commit_artifacts_locked(
    const std::vector<vector_index_metadata_store::change_log_row>
        *commit_delta_rows = nullptr,
    bool include_prepared = false);
bool persist_staged_commit_artifacts_locked(bool include_prepared = false);
bool change_log_compaction_needed_with_delta_locked(size_t delta_rows);
bool capture_runtime_commit_state_locked(commit_runtime_snapshot *snapshot);
bool rollback_runtime_commit_state_locked(
    const commit_runtime_snapshot &snapshot);
bool xid_matches_row(
    const XID &xid,
    const vector_index_metadata_store::prepared_change_row &row);
bool xid_from_prepared_row(
    const vector_index_metadata_store::prepared_change_row &row, XID *xid);
bool find_prepared_rows_for_xid_locked(
    const XID &xid,
    std::vector<vector_index_metadata_store::prepared_change_row> *rows);
bool has_prepared_xid_locked(const XID &xid);
void erase_prepared_rows_for_xid_locked(const XID &xid);
void queue_recovery_action_locked(const XID &xid, recovery_action_type type,
                                  bool conditional);
bool snapshot_prepared_rows_for_txn_locked(
    uint64_t txn_id, const XID &xid,
    std::vector<vector_index_metadata_store::prepared_change_row> *rows);
bool append_pending_change_log_delta_locked(
    uint64_t txn_id,
    const std::vector<vector_index::index_service::pending_change_snapshot>
        &changes);
bool allocate_pending_change_log_delta_locked(
    uint64_t txn_id,
    const std::vector<vector_index::index_service::pending_change_snapshot>
        &changes,
    std::vector<vector_index_metadata_store::change_log_row> *rows);
uint64_t allocate_txn_id();
void advance_txn_id_high_water(uint64_t txn_id);
std::string make_stmt_savepoint_name(uint64_t statement_id);
std::string make_user_savepoint_name(const std::string &name);
std::string make_mapped_index_name(const std::string &db_name,
                                   const std::string &table_name,
                                   const std::string &column_name);
bool parse_mapped_index_name(const std::string &index_name,
                             std::string *schema_name, std::string *table_name,
                             std::string *column_name);
index_binding binding_from_name(const std::string &index_name);
index_binding binding_for_index_locked(const std::string &index_name);
std::string owner_schema_for_index_locked(const std::string &index_name);
bool make_diskann_artifact_identity_locked(
    const std::string &index_name,
    const vector_index::index_service::index_config &config,
    const vector_index::index_service::index_publication_state &publication,
    size_t doc_id_count,
    vector_index::diskann_artifact_identity *identity);
void set_index_binding_locked(const std::string &index_name,
                              const std::string &schema_name,
                              const std::string &table_name,
                              const std::string &column_name,
                              const std::string &doc_id_column_name);
void erase_index_binding_and_owner_schema_locked(
    const std::string &index_name);
void rename_index_binding_locked(const std::string &old_index_name,
                                 const std::string &new_index_name);
void set_index_binding_and_owner_schema_locked(
    const std::string &index_name, const index_binding &binding,
    const std::string &owner_schema);
void rename_index_binding_and_owner_schema_locked(
    const std::string &old_index_name, const std::string &new_index_name,
    const std::string &owner_schema);
bool create_index_locked(
    const std::string &index_name, size_t dimension, const std::string &metric,
    const std::string &mode, const std::string &provider,
    const index_binding *binding, const std::string &owner_schema,
    const vector_index_registry::create_index_options &options);

bool resolve_create_index_definition(
    const std::string &mode, const std::string &provider,
    const vector_index_registry::create_index_options &requested_options,
    std::string *resolved_mode, std::string *resolved_provider,
    vector_index_registry::create_index_options *resolved_options);
bool apply_index_tuning_locked(const std::string &index_name,
                               const vector_index_registry::index_info &info);
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool publication_intent_target_matches_for_testing(
    const vector_index_truth_store::publication_intent &intent,
    const vector_index_truth_store::publication_token &current);
bool index_config_matches_for_testing(
    const vector_index::index_service::index_config &lhs,
    const vector_index::index_service::index_config &rhs);
bool bind_commit_truth_generations_for_testing(
    std::vector<vector_index_metadata_store::change_log_row> *rows,
    vector_index::index_service::commit_build_plan *plan);
bool publish_pending_runtime_for_testing(
    uint64_t txn_id,
    const std::vector<vector_index_metadata_store::change_log_row>
        &durable_rows,
    bool update_pending_status, std::string *failure_stage = nullptr);
bool index_bindings_equal_for_testing(const index_binding &lhs,
                                      const index_binding &rhs);
void set_standalone_rebuild_build_hook_for_testing(
    std::function<void()> hook);
void reset_standalone_rebuild_build_hook_for_testing();
void reset_publication_intent_recovery_for_testing();
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
bool snapshot_metadata_locked(
    std::vector<vector_index_metadata_store::metadata_row> *rows);
void refresh_committed_snapshot_rows_locked();
void refresh_manifest_status_locked();
void prune_change_log_for_index_locked(const std::string &index_name);
thd_txn_context *get_or_create_thd_txn_context_locked(uint64_t thd_id);
bool ensure_stmt_savepoint_locked(thd_txn_context *ctx, uint64_t statement_id);
void clear_stmt_savepoint_state(thd_txn_context *ctx);
void discard_thd_txn_context_locked(uint64_t thd_id);

}  // namespace vector_index_registry::detail

#endif  // SQL_VECTOR_INDEX_REGISTRY_INTERNAL_INCLUDED
