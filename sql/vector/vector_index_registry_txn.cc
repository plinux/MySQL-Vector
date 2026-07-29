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
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include "my_dbug.h"
#include "sql/vector/vector_index_diagnostics.h"
#include "sql/vector/vector_index_registry_internal.h"
#include "sql/vector/vector_index_service_internal.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_mapped_search.h"
#include "sql/vector/vector_status.h"

namespace vector_index_registry {

using namespace detail;

namespace {

bool init_empty_mod_tables(XA_recover_txn *txn, MEM_ROOT *mem_root) {
  if (txn == nullptr) return false;
  txn->mod_tables = nullptr;
  if (mem_root == nullptr) return true;

  txn->mod_tables = new (mem_root) List<st_handler_tablename>();
  return txn->mod_tables != nullptr;
}

bool persisted_prepared_xid_exists_locked(const XID &xid, bool *exists) {
  if (exists == nullptr) return false;
  *exists = has_prepared_xid_locked(xid);
  if (*exists || g_metadata_loaded) return true;

  std::vector<vector_index_metadata_store::prepared_change_row> rows;
  if (!vector_index_truth_store::get()->load_prepared(&rows)) {
    vector_status::record_metadata_load_failure();
    return false;
  }

  *exists = std::any_of(
      rows.begin(), rows.end(),
      [&xid](const vector_index_metadata_store::prepared_change_row &row) {
        return xid_matches_row(xid, row);
      });
  return true;
}

bool ensure_metadata_loaded_for_search() {
  {
    std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
    if (g_metadata_loaded) return true;
  }

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  return ensure_metadata_loaded_locked();
}

bool ensure_runtime_loaded_for_search(const std::string &index_name) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return g_index_service.ensure_runtime_loaded_for_search(index_name);
}

bool search_committed_runtime_loaded(
    const std::string &index_name, const vector_index::vector_data &query,
    size_t top_k, std::vector<vector_index::search_result> *results) {
  if (results == nullptr) return false;

  vector_index::index_service::backend_ptr runtime;
  vector_index::index_service::index_config config;
  {
    std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
    if (!g_index_service.snapshot_search_backend_loaded(index_name, query,
                                                        &runtime, &config)) {
      return false;
    }
  }

  {
    std::shared_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
    if (runtime->search(query, top_k, results)) return true;
  }
  if (!vector_index::detail::can_rebuild_after_search_failure(config))
    return false;

  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!g_index_service.rebuild_runtime_from_store_for_search(index_name)) {
      return false;
    }
    if (!g_index_service.snapshot_search_backend_loaded(index_name, query,
                                                        &runtime, nullptr)) {
      return false;
    }
  }

  std::shared_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
  return runtime->search(query, top_k, results);
}

bool search_committed_runtime_batch_loaded(
    const std::string &index_name,
    const std::vector<vector_index::vector_data> &queries, size_t top_k,
    std::vector<std::vector<vector_index::search_result>> *results) {
  if (results == nullptr || queries.empty()) return false;

  vector_index::index_service::backend_ptr runtime;
  vector_index::index_service::index_config config;
  {
    std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
    if (!g_index_service.snapshot_search_backend_loaded(index_name, queries[0],
                                                        &runtime, &config)) {
      return false;
    }
  }
  for (const vector_index::vector_data &query : queries) {
    if (query.size() != config.dimension) return false;
  }

  {
    std::shared_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
    if (runtime->search_batch(queries, top_k, results)) return true;
  }
  if (!vector_index::detail::can_rebuild_after_search_failure(config))
    return false;

  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!g_index_service.rebuild_runtime_from_store_for_search(index_name)) {
      return false;
    }
    if (!g_index_service.snapshot_search_backend_loaded(index_name, queries[0],
                                                        &runtime, nullptr)) {
      return false;
    }
  }

  std::shared_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
  return runtime->search_batch(queries, top_k, results);
}

size_t pending_change_count_for_thd_locked(uint64_t thd_id) {
  const auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end()) return 0;
  return g_index_service.pending_change_count(it->second.txn_id);
}

xa_status_code apply_prepared_xid_with_metadata_locked(const XID &xid,
                                                       bool commit,
                                                       uint64_t thd_id) {
  if (!ensure_metadata_loaded_locked()) return XAER_RMERR;
  return detail::apply_prepared_xid_locked(xid, commit, thd_id);
}

xa_status_code apply_existing_prepared_xid_locked(const XID &xid,
                                                  bool commit) {
  bool exists = false;
  if (!persisted_prepared_xid_exists_locked(xid, &exists)) return XAER_RMERR;
  if (!exists) return XAER_NOTA;
  return apply_prepared_xid_with_metadata_locked(xid, commit, 0);
}

}  // namespace

void record_commit_diagnostics(const char *scope, uint64_t txn_id,
                               size_t pending_change_count, bool ok,
                               uint64_t pending_index_names_ms,
                               uint64_t snapshot_before_ms,
                               uint64_t index_commit_ms,
                               uint64_t pending_delta_ms,
                               uint64_t changelog_delta_ms,
                               uint64_t persist_ms, uint64_t total_ms) {
  if (!vector_index_diagnostics::enabled()) return;
  vector_index_diagnostics::record_event(
      "vector_commit", {{"scope", scope == nullptr ? "" : scope}},
      {{"txn_id", txn_id},
       {"pending_changes", static_cast<uint64_t>(pending_change_count)},
       {"pending_index_names_ms", pending_index_names_ms},
       {"snapshot_before_ms", snapshot_before_ms},
       {"index_commit_ms", index_commit_ms},
       {"pending_delta_ms", pending_delta_ms},
       {"changelog_delta_ms", changelog_delta_ms},
       {"persist_ms", persist_ms},
       {"total_ms", total_ms},
       {"ok", ok ? 1U : 0U}});
}

bool commit_index_service_txn(const char *scope, uint64_t txn_id,
                              bool enable_debug_failures) {
  size_t pending_change_count = 0;
  const bool diagnostics_enabled = vector_index_diagnostics::enabled();
  const auto total_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  uint64_t pending_index_names_ms = 0;
  uint64_t snapshot_before_ms = 0;
  uint64_t index_commit_ms = 0;
  uint64_t pending_delta_ms = 0;
  uint64_t changelog_delta_ms = 0;
  uint64_t persist_ms = 0;
  std::vector<std::string> pending_index_names;

  commit_runtime_snapshot before_state;
  std::vector<vector_index::index_service::pending_change_snapshot>
      pending_delta;
  vector_index::index_service::pending_state_snapshot before_pending_state;
  vector_index::index_service::commit_build_plan build_plan;
  const auto commit_started =
      std::chrono::steady_clock::now();
  const auto index_commit_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    pending_change_count = g_index_service.pending_change_count(txn_id);
    if (pending_change_count == 0) return true;

    if (enable_debug_failures) {
      DBUG_EXECUTE_IF("vector_registry_fail_pending_change_index_names",
                      return false;);
    }
    const auto pending_names_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    if (!g_index_service.pending_change_index_names(txn_id,
                                                   &pending_index_names)) {
      record_commit_diagnostics(
          scope, txn_id, pending_change_count, false,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  pending_names_start),
          0, 0, 0, 0, 0,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  total_start));
      return false;
    }
    pending_index_names_ms =
        vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                pending_names_start);

    const auto pending_delta_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    if (!g_index_service.snapshot_commit_build_plan(txn_id, &build_plan)) {
      record_commit_diagnostics(
          scope, txn_id, pending_change_count, false, pending_index_names_ms,
          snapshot_before_ms, 0,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  pending_delta_start),
          0, 0,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  total_start));
      return false;
    }
    pending_delta_ms =
        vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                pending_delta_start);

  }

  if (!g_index_service.build_commit_backends(&build_plan)) {
    index_commit_ms =
        vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                index_commit_start);
    vector_status::record_txn_commit_failure();
    record_commit_diagnostics(
        scope, txn_id, pending_change_count, false, pending_index_names_ms,
        snapshot_before_ms, index_commit_ms, pending_delta_ms, 0, 0,
        vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                total_start));
    return false;
  }

  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    if (enable_debug_failures) {
      DBUG_EXECUTE_IF("vector_registry_fail_snapshot_committed_before_commit",
                      return false;);
    }
    const auto snapshot_before_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    if (!capture_runtime_commit_state_locked(&before_state)) {
      record_commit_diagnostics(
          scope, txn_id, pending_change_count, false, pending_index_names_ms,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  snapshot_before_start),
          index_commit_ms, pending_delta_ms, 0, 0,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  total_start));
      return false;
    }
    snapshot_before_ms =
        vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                snapshot_before_start);
    const size_t change_log_size_before = before_state.change_log_rows.size();

    const auto pending_delta_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    if (!g_index_service.snapshot_pending_change_delta(txn_id, &pending_delta)) {
      record_commit_diagnostics(
          scope, txn_id, pending_change_count, false, pending_index_names_ms,
          snapshot_before_ms, index_commit_ms,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  pending_delta_start),
          0, 0,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  total_start));
      return false;
    }
    if (!g_index_service.snapshot_pending_state(txn_id, &before_pending_state)) {
      record_commit_diagnostics(
          scope, txn_id, pending_change_count, false, pending_index_names_ms,
          snapshot_before_ms, index_commit_ms, pending_delta_ms, 0, 0,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  total_start));
      return false;
    }
    pending_delta_ms =
        vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                pending_delta_start);
    const bool ok = g_index_service.apply_commit_build_plan(txn_id, &build_plan);
    index_commit_ms =
        vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                index_commit_start);
    if (!ok) {
      vector_status::record_txn_commit_failure();
      record_commit_diagnostics(
          scope, txn_id, pending_change_count, false, pending_index_names_ms,
          snapshot_before_ms, index_commit_ms, pending_delta_ms, 0, 0,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  total_start));
      return false;
    }

    const auto changelog_delta_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    append_pending_change_log_delta_locked(txn_id, pending_delta);
    changelog_delta_ms =
        pending_delta_ms +
        vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                changelog_delta_start);
    const std::vector<vector_index_metadata_store::change_log_row>
        commit_delta_rows(g_change_log_rows.begin() + change_log_size_before,
                          g_change_log_rows.end());

    vector_status::subtract_pending_txn_changes(pending_change_count);
    const auto persist_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    if (!persist_commit_artifacts_locked(&commit_delta_rows)) {
      persist_ms =
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  persist_start);
      if (!rollback_runtime_commit_state_locked(before_state) ||
          !g_index_service.restore_pending_state(txn_id, before_pending_state)) {
        return false;
      }
      vector_status::add_pending_txn_changes(pending_change_count);
      vector_status::record_txn_commit_failure();
      record_commit_diagnostics(
          scope, txn_id, pending_change_count, false, pending_index_names_ms,
          snapshot_before_ms, index_commit_ms, pending_delta_ms,
          changelog_delta_ms, persist_ms,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  total_start));
      return false;
    }
    persist_ms =
        vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                persist_start);
    if (!evict_committed_cache_to_budget_locked()) {
      vector_status::record_runtime_state_rollback_failure();
      vector_status::record_txn_commit_failure();
      record_commit_diagnostics(
          scope, txn_id, pending_change_count, false, pending_index_names_ms,
          snapshot_before_ms, index_commit_ms, pending_delta_ms,
          changelog_delta_ms, persist_ms,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  total_start));
      return false;
    }

    uint64_t apply_latency_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - commit_started)
            .count());
    if (apply_latency_ms == 0) apply_latency_ms = 1;
    vector_status::set_apply_latency_ms(apply_latency_ms);
    for (const std::string &index_name : pending_index_names) {
      (void)g_index_service.set_last_apply_latency_ms(index_name,
                                                      apply_latency_ms);
    }
    record_commit_diagnostics(
        scope, txn_id, pending_change_count, true, pending_index_names_ms,
        snapshot_before_ms, index_commit_ms, pending_delta_ms,
        changelog_delta_ms, persist_ms,
        vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                total_start));
  }
  return true;
}

uint64_t begin_txn() { return allocate_txn_id(); }

bool commit_txn(uint64_t txn_id) {
  vector_status::record_txn_commit_request();
  return commit_index_service_txn("explicit", txn_id, false);
}

bool rollback_txn(uint64_t txn_id) {
  vector_status::record_txn_rollback_request();
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  const size_t pending_change_count = g_index_service.pending_change_count(txn_id);
  g_index_service.rollback(txn_id);
  if (pending_change_count > 0) {
    vector_status::subtract_pending_txn_changes(pending_change_count);
  }
  return true;
}

size_t pending_txn_changes(uint64_t txn_id) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return 0;
  return g_index_service.pending_change_count(txn_id);
}

bool savepoint_txn(uint64_t txn_id, const std::string &name) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return g_index_service.savepoint(txn_id, name);
}

bool rollback_to_savepoint_txn(uint64_t txn_id, const std::string &name) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  const size_t pending_before = g_index_service.pending_change_count(txn_id);
  const bool ok = g_index_service.rollback_to_savepoint(txn_id, name);
  if (!ok) return false;

  const size_t pending_after = g_index_service.pending_change_count(txn_id);
  if (pending_before > pending_after) {
    vector_status::subtract_pending_txn_changes(pending_before - pending_after);
  }
  return true;
}

bool release_savepoint_txn(uint64_t txn_id, const std::string &name) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return g_index_service.release_savepoint(txn_id, name);
}

bool stage_upsert(uint64_t txn_id, const std::string &index_name, uint64_t doc_id,
                 const vector_index::vector_data &vector) {
  vector_status::record_stage_upsert_request();
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  const bool ok = g_index_service.stage_upsert(txn_id, index_name, doc_id, vector);
  if (ok) vector_status::add_pending_txn_changes(1);
  return ok;
}

bool stage_erase(uint64_t txn_id, const std::string &index_name, uint64_t doc_id) {
  vector_status::record_stage_erase_request();
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  const bool ok = g_index_service.stage_erase(txn_id, index_name, doc_id);
  if (ok) vector_status::add_pending_txn_changes(1);
  return ok;
}

bool stage_upsert_for_thd_txn(uint64_t thd_id, uint64_t statement_id,
                          const std::string &index_name, uint64_t doc_id,
                          const vector_index::vector_data &vector) {
  vector_status::record_stage_upsert_request();
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;

  thd_txn_context *ctx = get_or_create_thd_txn_context_locked(thd_id);
  if (!ensure_stmt_savepoint_locked(ctx, statement_id)) return false;

  const bool ok =
      g_index_service.stage_upsert(ctx->txn_id, index_name, doc_id, vector);
  if (ok) vector_status::add_pending_txn_changes(1);
  return ok;
}

bool stage_erase_for_thd_txn(uint64_t thd_id, uint64_t statement_id,
                         const std::string &index_name, uint64_t doc_id) {
  vector_status::record_stage_erase_request();
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;

  thd_txn_context *ctx = get_or_create_thd_txn_context_locked(thd_id);
  if (!ensure_stmt_savepoint_locked(ctx, statement_id)) return false;

  const bool ok = g_index_service.stage_erase(ctx->txn_id, index_name, doc_id);
  if (ok) vector_status::add_pending_txn_changes(1);
  return ok;
}

bool commit_stmt_for_thd_txn(uint64_t thd_id, uint64_t statement_id) {
  if (vector_index_truth_store::internal_sql_active()) return true;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end()) return true;
  if (!ensure_metadata_loaded_locked()) return false;

  thd_txn_context &ctx = it->second;
  if (!ctx.stmt_savepoint_active || ctx.active_stmt_id != statement_id) {
    return true;
  }

  const bool ok =
      g_index_service.release_savepoint(ctx.txn_id, ctx.stmt_savepoint_name);
  if (ok) clear_stmt_savepoint_state(&ctx);
  return ok;
}

bool rollback_stmt_for_thd_txn(uint64_t thd_id, uint64_t statement_id) {
  if (vector_index_truth_store::internal_sql_active()) return true;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end()) return true;
  if (!ensure_metadata_loaded_locked()) return false;

  thd_txn_context &ctx = it->second;
  if (!ctx.stmt_savepoint_active || ctx.active_stmt_id != statement_id) {
    return true;
  }

  const size_t pending_before = g_index_service.pending_change_count(ctx.txn_id);
  if (!g_index_service.rollback_to_savepoint(ctx.txn_id, ctx.stmt_savepoint_name)) {
    return false;
  }
  const size_t pending_after = g_index_service.pending_change_count(ctx.txn_id);
  if (pending_before > pending_after) {
    vector_status::subtract_pending_txn_changes(pending_before - pending_after);
  }

  DBUG_EXECUTE_IF("vector_registry_fail_release_stmt_savepoint_after_rollback",
                  return false;);
  if (!g_index_service.release_savepoint(ctx.txn_id, ctx.stmt_savepoint_name)) {
    return false;
  }

  clear_stmt_savepoint_state(&ctx);
  return true;
}

xa_status_code detail::apply_prepared_xid_locked(const XID &xid, bool commit,
                                                 uint64_t thd_id_to_clear) {
  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  if (!find_prepared_rows_for_xid_locked(xid, &prepared_rows)) return XAER_RMERR;
  if (prepared_rows.empty()) return XAER_NOTA;

  if (!commit) {
    const auto prepared_before = g_prepared_change_rows;
    erase_prepared_rows_for_xid_locked(xid);
    if (!persist_prepared_locked()) {
      g_prepared_change_rows = prepared_before;
      return XAER_RMERR;
    }
    if (thd_id_to_clear != 0) discard_thd_txn_context_locked(thd_id_to_clear);
    return XA_OK;
  }

  std::vector<vector_index_metadata_store::metadata_row> metadata_before;
  vector_index::index_service::committed_state committed_before;
  std::vector<vector_index_metadata_store::change_log_row> change_log_before;
  std::vector<std::string> lagging_indexes_before;
  if (!snapshot_runtime_state_locked(&metadata_before, &committed_before,
                                     &change_log_before,
                                     &lagging_indexes_before)) {
    return XAER_RMERR;
  }
  const auto prepared_before = g_prepared_change_rows;
  persisted_commit_artifacts_snapshot persisted_before;
  auto *truth_store = vector_index_truth_store::get();
  const bool needs_persisted_snapshot = !truth_store->is_transactional();
  if (needs_persisted_snapshot &&
      !load_persisted_commit_artifacts_snapshot_locked(&persisted_before)) {
    return XAER_RMERR;
  }
  const size_t thd_pending_before =
      pending_change_count_for_thd_locked(thd_id_to_clear);

  std::vector<vector_index::index_service::pending_change_snapshot> changes;
  changes.reserve(prepared_rows.size());
  uint64_t replay_txn_id = 0;
  for (const auto &row : prepared_rows) {
    if (replay_txn_id == 0) replay_txn_id = row.txn_id;
    changes.push_back(vector_index::index_service::pending_change_snapshot{
        row.index_name, row.op == vector_index_metadata_store::change_op::kErase,
        row.doc_id, row.vector});
  }
  if (replay_txn_id == 0) replay_txn_id = allocate_txn_id();
  DBUG_EXECUTE_IF("vector_registry_fail_restore_pending_for_prepared_replay",
                  return XAER_RMERR;);
  if (!g_index_service.restore_pending_changes(replay_txn_id, changes)) {
    return XAER_RMERR;
  }

  DBUG_EXECUTE_IF("vector_registry_fail_snapshot_before_prepared_commit",
                  g_index_service.rollback(replay_txn_id); return XAER_RMERR;);
  std::vector<vector_index::index_service::pending_change_snapshot>
      pending_delta;
  if (!g_index_service.snapshot_pending_change_delta(replay_txn_id,
                                                     &pending_delta)) {
    g_index_service.rollback(replay_txn_id);
    return XAER_RMERR;
  }
  DBUG_EXECUTE_IF("vector_registry_fail_commit_prepared_replay",
                  g_index_service.rollback(replay_txn_id); return XAER_RMERR;);
  if (!g_index_service.commit(replay_txn_id)) {
    g_index_service.rollback(replay_txn_id);
    return XAER_RMERR;
  }

  const size_t change_log_size_before = g_change_log_rows.size();
  append_pending_change_log_delta_locked(replay_txn_id, pending_delta);
  const std::vector<vector_index_metadata_store::change_log_row>
      commit_delta_rows(g_change_log_rows.begin() + change_log_size_before,
                        g_change_log_rows.end());
  erase_prepared_rows_for_xid_locked(xid);

  bool ok = persist_commit_artifacts_locked(&commit_delta_rows, true);
  if (!ok) {
    g_prepared_change_rows = prepared_before;
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      return XAER_RMERR;
    }
    if (needs_persisted_snapshot &&
        !restore_persisted_commit_artifacts_snapshot_locked(persisted_before)) {
      return XAER_RMERR;
    }
    return XAER_RMERR;
  }

  if (thd_id_to_clear != 0) {
    const size_t thd_pending_after =
        pending_change_count_for_thd_locked(thd_id_to_clear);
    if (thd_pending_before > thd_pending_after) {
      vector_status::subtract_pending_txn_changes(thd_pending_before -
                                                  thd_pending_after);
    }
    discard_thd_txn_context_locked(thd_id_to_clear);
  }
  refresh_committed_snapshot_rows_locked();
  return XA_OK;
}

bool commit_thd_txn(uint64_t thd_id) {
  if (vector_index_truth_store::internal_sql_active()) return true;
  uint64_t txn_id = 0;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    auto it = g_thd_txn_contexts.find(thd_id);
    if (it == g_thd_txn_contexts.end()) return true;
    if (!ensure_metadata_loaded_locked()) return false;

    vector_status::record_txn_commit_request();

    thd_txn_context &ctx = it->second;
    if (ctx.stmt_savepoint_active) {
      (void)g_index_service.release_savepoint(ctx.txn_id, ctx.stmt_savepoint_name);
      clear_stmt_savepoint_state(&ctx);
    }
    txn_id = ctx.txn_id;
  }

  if (!commit_index_service_txn("thd", txn_id, true)) return false;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  g_thd_txn_contexts.erase(thd_id);
  return true;
}

bool rollback_thd_txn(uint64_t thd_id) {
  if (vector_index_truth_store::internal_sql_active()) return true;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end()) return true;
  if (!ensure_metadata_loaded_locked()) return false;

  vector_status::record_txn_rollback_request();

  thd_txn_context &ctx = it->second;
  const size_t pending_change_count = g_index_service.pending_change_count(ctx.txn_id);
  g_index_service.rollback(ctx.txn_id);
  if (pending_change_count > 0) {
    vector_status::subtract_pending_txn_changes(pending_change_count);
  }
  g_thd_txn_contexts.erase(it);
  return true;
}

void discard_empty_thd_txn(uint64_t thd_id) {
  if (vector_index_truth_store::internal_sql_active()) return;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end()) return;

  thd_txn_context &ctx = it->second;
  if (g_index_service.pending_change_count(ctx.txn_id) != 0) return;

  g_index_service.rollback(ctx.txn_id);
  g_thd_txn_contexts.erase(it);
}

bool prepare_thd_txn(uint64_t thd_id, const XID &xid) {
  if (vector_index_truth_store::internal_sql_active()) return true;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end()) return true;
  if (!ensure_metadata_loaded_locked()) return false;

  thd_txn_context &ctx = it->second;
  std::vector<vector_index_metadata_store::prepared_change_row> rows;
  DBUG_EXECUTE_IF("vector_registry_fail_snapshot_prepared_rows_for_thd_txn",
                  return false;);
  if (!snapshot_prepared_rows_for_txn_locked(ctx.txn_id, xid, &rows)) {
    return false;
  }
  if (rows.empty()) return true;

  const auto prepared_before = g_prepared_change_rows;
  erase_prepared_rows_for_xid_locked(xid);
  g_prepared_change_rows.insert(g_prepared_change_rows.end(), rows.begin(),
                                rows.end());
  if (persist_prepared_locked()) return true;
  g_prepared_change_rows = prepared_before;
  return false;
}

bool set_prepared_in_tc(const XID &xid) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;

  const auto prepared_before = g_prepared_change_rows;
  bool found = false;
  for (auto &row : g_prepared_change_rows) {
    if (!xid_matches_row(xid, row)) continue;
    row.prepared_in_tc = true;
    found = true;
  }
  if (!found) return true;
  if (persist_prepared_locked()) return true;
  g_prepared_change_rows = prepared_before;
  return false;
}

bool has_prepared_xid(const XID &xid) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  return has_prepared_xid_locked(xid);
}

xa_status_code commit_prepared_xid(const XID &xid) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  return apply_prepared_xid_with_metadata_locked(xid, true, 0);
}

xa_status_code commit_prepared_xid_if_loaded(const XID &xid) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  return apply_existing_prepared_xid_locked(xid, true);
}

xa_status_code commit_prepared_xid_for_thd(uint64_t thd_id, const XID &xid) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  return apply_prepared_xid_with_metadata_locked(xid, true, thd_id);
}

xa_status_code rollback_prepared_xid(const XID &xid) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  return apply_prepared_xid_with_metadata_locked(xid, false, 0);
}

xa_status_code rollback_prepared_xid_if_loaded(const XID &xid) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  return apply_existing_prepared_xid_locked(xid, false);
}

xa_status_code rollback_prepared_xid_for_thd(uint64_t thd_id, const XID &xid) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  return apply_prepared_xid_with_metadata_locked(xid, false, thd_id);
}

int recover_prepared_xids(XA_recover_txn *txn_list, uint len,
                          MEM_ROOT *mem_root) {
  if (txn_list == nullptr || len == 0) return 0;

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return 0;

  size_t count = 0;
  for (const auto &row : g_prepared_change_rows) {
    if (count >= len) break;
    XID xid;
    xid_from_prepared_row(row, &xid);

    bool seen = false;
    for (size_t i = 0; i < count; ++i) {
      if (txn_list[i].id.eq(&xid)) {
        seen = true;
        break;
      }
    }
    if (seen) continue;

    txn_list[count].id = xid;
    if (!init_empty_mod_tables(&txn_list[count], mem_root)) break;
    ++count;
  }
  return static_cast<int>(count);
}

int recover_prepared_in_tc(Xa_state_list &xa_list) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return 1;

  for (const auto &row : g_prepared_change_rows) {
    if (!row.prepared_in_tc) continue;

    XID xid;
    xid_from_prepared_row(row, &xid);
    if (xid.get_my_xid() != 0) continue;
    (void)xa_list.add(xid, enum_ha_recover_xa_state::PREPARED_IN_TC);
  }
  return 0;
}

void queue_recovery_commit_xid(const XID &xid) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  queue_recovery_action_locked(xid, recovery_action_type::kCommit, true);
}

void queue_recovery_rollback_xid(const XID &xid) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  queue_recovery_action_locked(xid, recovery_action_type::kRollback, true);
}

void queue_recovery_set_prepared_in_tc(const XID &xid) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  queue_recovery_action_locked(xid, recovery_action_type::kPreparedInTc, true);
}

bool savepoint_thd_txn(uint64_t thd_id, const std::string &name) {
  if (vector_index_truth_store::internal_sql_active()) return true;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (name.empty()) return false;

  if (!ensure_metadata_loaded_locked()) return false;

  thd_txn_context *ctx = get_or_create_thd_txn_context_locked(thd_id);
  return g_index_service.savepoint(ctx->txn_id,
                                   make_user_savepoint_name(name));
}

bool rollback_to_savepoint_thd_txn(uint64_t thd_id, const std::string &name) {
  if (vector_index_truth_store::internal_sql_active()) return true;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (name.empty()) return false;

  auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end()) return true;
  if (!ensure_metadata_loaded_locked()) return false;

  thd_txn_context &ctx = it->second;
  const size_t pending_before = g_index_service.pending_change_count(ctx.txn_id);
  if (!g_index_service.rollback_to_savepoint(ctx.txn_id,
                                           make_user_savepoint_name(name))) {
    return false;
  }
  const size_t pending_after = g_index_service.pending_change_count(ctx.txn_id);
  if (pending_before > pending_after) {
    vector_status::subtract_pending_txn_changes(pending_before - pending_after);
  }
  clear_stmt_savepoint_state(&ctx);
  return true;
}

bool release_savepoint_thd_txn(uint64_t thd_id, const std::string &name) {
  if (vector_index_truth_store::internal_sql_active()) return true;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (name.empty()) return false;

  auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end()) return true;
  if (!ensure_metadata_loaded_locked()) return false;

  return g_index_service.release_savepoint(it->second.txn_id,
                                          make_user_savepoint_name(name));
}

bool upsert(const std::string &index_name, uint64_t doc_id,
            const vector_index::vector_data &vector) {
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    vector_index::index_service::index_config config;
    if (!g_index_service.describe_index(index_name, &config, nullptr, nullptr,
                                        nullptr)) {
      return false;
    }
    if (config.consistency_mode ==
        vector_index::index_consistency_mode::kStandalone) {
      return g_index_service.direct_upsert(index_name, doc_id, vector);
    }
  }

  const uint64_t txn_id = allocate_txn_id();
  if (!stage_upsert(txn_id, index_name, doc_id, vector)) return false;
  if (!commit_txn(txn_id)) {
    rollback_txn(txn_id);
    return false;
  }
  return true;
}

bool erase(const std::string &index_name, uint64_t doc_id) {
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    vector_index::index_service::index_config config;
    if (!g_index_service.describe_index(index_name, &config, nullptr, nullptr,
                                        nullptr)) {
      return false;
    }
    if (config.consistency_mode ==
        vector_index::index_consistency_mode::kStandalone) {
      return g_index_service.direct_erase(index_name, doc_id);
    }
  }

  const uint64_t txn_id = allocate_txn_id();
  if (!stage_erase(txn_id, index_name, doc_id)) return false;
  if (!commit_txn(txn_id)) {
    rollback_txn(txn_id);
    return false;
  }
  return true;
}

bool search(const std::string &index_name, const vector_index::vector_data &query,
            size_t top_k, std::vector<vector_index::search_result> *results) {
  vector_status::record_search_request();
  if (results == nullptr) {
    vector_status::record_search_failure();
    return false;
  }
  if (!ensure_metadata_loaded_for_search()) {
    vector_status::record_search_failure();
    return false;
  }
  if (!ensure_runtime_loaded_for_search(index_name)) {
    vector_status::record_search_failure();
    return false;
  }
  const bool ok =
      search_committed_runtime_loaded(index_name, query, top_k, results);
  if (!ok) {
    vector_status::record_search_failure();
    return false;
  }

  if (results != nullptr && !results->empty()) {
    vector_status::record_search_results_returned(results->size());
  }
  return true;
}

bool search_for_thd_txn(THD *thd, uint64_t thd_id, const std::string &index_name,
                     const vector_index::vector_data &query, size_t top_k,
                     std::vector<vector_index::search_result> *results) {
  vector_status::record_search_request();
  if (results == nullptr) {
    vector_status::record_search_failure();
    return false;
  }
  if (!ensure_metadata_loaded_for_search()) {
    vector_status::record_search_failure();
    return false;
  }
  if (!ensure_runtime_loaded_for_search(index_name)) {
    vector_status::record_search_failure();
    return false;
  }

  vector_mapped_search::search_spec mapped_spec;
  bool use_mapped_filter = false;
  uint64_t txn_id = 0;
  size_t candidate_top_k = top_k;
  size_t candidate_limit = top_k;
  bool candidate_hard_limit = false;
  {
    std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
    auto ctx_it = g_thd_txn_contexts.find(thd_id);
    if (ctx_it != g_thd_txn_contexts.end()) txn_id = ctx_it->second.txn_id;

    vector_index::index_service::index_config config;
    bool supports_mutations = false;
    size_t entry_count = 0;
    size_t committed_entry_count = 0;
    const bool described =
        g_index_service.describe_index(index_name, &config, &supports_mutations,
                                      &entry_count, &committed_entry_count);
    if (described) {
      const index_binding binding = binding_for_index_locked(index_name);
      if (!binding.schema_name.empty() && !binding.table_name.empty() &&
          !binding.column_name.empty() && !binding.doc_id_column_name.empty() &&
          thd != nullptr) {
        use_mapped_filter = true;
        mapped_spec.schema_name = binding.schema_name;
        mapped_spec.table_name = binding.table_name;
        mapped_spec.column_name = binding.column_name;
        mapped_spec.doc_id_column_name = binding.doc_id_column_name;
        mapped_spec.metric = config.metric;
        const size_t pending_bonus =
            (txn_id != 0) ? g_index_service.pending_change_count(txn_id) : 0;
        candidate_hard_limit =
            mapped_search_candidate_limit_is_hard(entry_count, pending_bonus);
        candidate_limit =
            mapped_search_candidate_limit(top_k, entry_count, pending_bonus);
        candidate_top_k = mapped_search_initial_candidate_top_k(
            top_k, entry_count, pending_bonus);
        mapped_spec.top_k = top_k;
      }
    }
  }

  if (!use_mapped_filter) {
    bool ok = false;
    if (txn_id != 0) {
      std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
      ok = g_index_service.search_with_pending_loaded(txn_id, index_name, query,
                                                      top_k, results);
    } else {
      ok = search_committed_runtime_loaded(index_name, query, top_k, results);
    }
    if (!ok) {
      vector_status::record_search_failure();
      return false;
    }
  } else {
    if (candidate_limit == 0) {
      if (results != nullptr) results->clear();
    }

    while (candidate_limit > 0) {
      std::vector<vector_index::search_result> candidate_results;
      bool ok = false;
      if (txn_id != 0) {
        std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
        ok = g_index_service.search_with_pending_loaded(
            txn_id, index_name, query, candidate_top_k, &candidate_results);
      } else {
        ok = search_committed_runtime_loaded(index_name, query, candidate_top_k,
                                             &candidate_results);
      }
      if (!ok) {
        vector_status::record_search_failure();
        return false;
      }

      vector_status::record_search_mvcc_candidate_rows(
          candidate_results.size());
      if (!vector_mapped_search::filter_visible_results(
              thd, mapped_spec, query, &candidate_results)) {
        vector_status::record_search_failure();
        return false;
      }
      if (candidate_results.size() >= top_k ||
          candidate_top_k >= candidate_limit) {
        if (candidate_hard_limit && candidate_results.size() < top_k &&
            candidate_top_k >= candidate_limit) {
          vector_status::record_search_mvcc_limit_hit();
        }
        if (results != nullptr) *results = std::move(candidate_results);
        break;
      }

      candidate_top_k = mapped_search_next_candidate_top_k(
          candidate_top_k, top_k, candidate_limit);
      vector_status::record_search_mvcc_expansion();
    }
  }

  if (results != nullptr && !results->empty()) {
    vector_status::record_search_results_returned(results->size());
  }
  return true;
}

bool search_batch_for_thd_txn(
    THD *thd, uint64_t thd_id, const std::string &index_name,
    const std::vector<vector_index::vector_data> &queries, size_t top_k,
    std::vector<std::vector<vector_index::search_result>> *results) {
  if (results == nullptr) return false;
  results->clear();
  if (queries.empty()) return true;

  if (!ensure_metadata_loaded_for_search()) {
    for (size_t i = 0; i < queries.size(); ++i)
      vector_status::record_search_failure();
    return false;
  }
  if (!ensure_runtime_loaded_for_search(index_name)) {
    for (size_t i = 0; i < queries.size(); ++i)
      vector_status::record_search_failure();
    return false;
  }

  bool fallback_to_single_search = false;
  bool ok = false;
  {
    std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
    const auto ctx_it = g_thd_txn_contexts.find(thd_id);
    fallback_to_single_search =
        ctx_it != g_thd_txn_contexts.end() && ctx_it->second.txn_id != 0;

    vector_index::index_service::index_config config;
    bool supports_mutations = false;
    size_t entry_count = 0;
    size_t committed_entry_count = 0;
    if (!fallback_to_single_search &&
        g_index_service.describe_index(index_name, &config, &supports_mutations,
                                      &entry_count, &committed_entry_count)) {
      const index_binding binding = binding_for_index_locked(index_name);
      fallback_to_single_search =
          !binding.schema_name.empty() && !binding.table_name.empty() &&
          !binding.column_name.empty() &&
          !binding.doc_id_column_name.empty() && thd != nullptr;
    }

    if (!fallback_to_single_search) {
      for (size_t i = 0; i < queries.size(); ++i)
        vector_status::record_search_request();
    }
  }

  if (fallback_to_single_search) {
    results->resize(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
      if (!search_for_thd_txn(thd, thd_id, index_name, queries[i], top_k,
                            &(*results)[i])) {
        return false;
      }
    }
    return true;
  }

  if (!ok) {
    ok = search_committed_runtime_batch_loaded(index_name, queries, top_k,
                                               results);
  }

  if (!ok) {
    for (size_t i = 0; i < queries.size(); ++i)
      vector_status::record_search_failure();
    return false;
  }

  for (const auto &one_result : *results) {
    if (!one_result.empty())
      vector_status::record_search_results_returned(one_result.size());
  }
  return true;
}

}  // namespace vector_index_registry
