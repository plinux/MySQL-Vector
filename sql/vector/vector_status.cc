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

#include "sql/vector/vector_status.h"

#include <atomic>

namespace {

std::atomic<uint64_t> g_index_create_requests{0};
std::atomic<uint64_t> g_index_drop_requests{0};
std::atomic<uint64_t> g_metadata_load_failures{0};
std::atomic<uint64_t> g_metadata_persist_failures{0};
std::atomic<uint64_t> g_committed_load_failures{0};
std::atomic<uint64_t> g_committed_persist_failures{0};
std::atomic<uint64_t> g_manifest_load_failures{0};
std::atomic<uint64_t> g_manifest_persist_failures{0};
std::atomic<uint64_t> g_change_log_load_failures{0};
std::atomic<uint64_t> g_change_log_replay_failures{0};
std::atomic<uint64_t> g_change_log_persist_failures{0};
std::atomic<uint64_t> g_backend_recover_fallbacks{0};
std::atomic<uint64_t> g_registered_indexes{0};
std::atomic<uint64_t> g_committed_snapshot_rows{0};
std::atomic<uint64_t> g_manifest_version{1};
std::atomic<uint64_t> g_manifest_metadata_checkpoint{0};
std::atomic<uint64_t> g_manifest_committed_checkpoint{0};
std::atomic<uint64_t> g_manifest_change_log_checkpoint{0};

std::atomic<uint64_t> g_rebuild_requests{0};
std::atomic<uint64_t> g_recover_requests{0};
std::atomic<uint64_t> g_rebuild_all_requests{0};
std::atomic<uint64_t> g_recover_all_requests{0};

std::atomic<uint64_t> g_search_requests{0};
std::atomic<uint64_t> g_search_failures{0};
std::atomic<uint64_t> g_search_results_returned{0};
std::atomic<uint64_t> g_search_mvcc_candidate_rows{0};
std::atomic<uint64_t> g_search_mvcc_expansions{0};
std::atomic<uint64_t> g_search_mvcc_limit_hits{0};

std::atomic<uint64_t> g_stage_upsert_requests{0};
std::atomic<uint64_t> g_stage_erase_requests{0};

std::atomic<uint64_t> g_txn_commit_requests{0};
std::atomic<uint64_t> g_txn_commit_failures{0};
std::atomic<uint64_t> g_txn_rollback_requests{0};
std::atomic<uint64_t> g_runtime_state_rollbacks{0};
std::atomic<uint64_t> g_runtime_state_rollback_failures{0};
std::atomic<uint64_t> g_persist_artifact_rollbacks{0};
std::atomic<uint64_t> g_persist_artifact_rollback_failures{0};
std::atomic<uint64_t> g_truth_store_persist_requests{0};
std::atomic<uint64_t> g_truth_store_persist_failures{0};
std::atomic<uint64_t> g_truth_store_delta_persist_requests{0};
std::atomic<uint64_t> g_truth_store_delta_persist_failures{0};
std::atomic<uint64_t> g_truth_store_compact_requests{0};
std::atomic<uint64_t> g_truth_store_compact_failures{0};

std::atomic<uint64_t> g_pending_txn_changes{0};
std::atomic<uint64_t> g_apply_latency_ms{0};

inline void add_counter(std::atomic<uint64_t> *counter, uint64_t delta = 1) {
  counter->fetch_add(delta, std::memory_order_relaxed);
}

inline void subtract_counter(std::atomic<uint64_t> *counter, uint64_t delta = 1) {
  uint64_t current = counter->load(std::memory_order_relaxed);
  while (true) {
    const uint64_t next = (delta > current) ? 0 : (current - delta);
    if (counter->compare_exchange_weak(current, next, std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
      return;
    }
  }
}

inline uint64_t read_counter(const std::atomic<uint64_t> &counter) {
  return counter.load(std::memory_order_relaxed);
}

}  // namespace

namespace vector_status {

void read_snapshot(snapshot *snapshot) {
  if (snapshot == nullptr) return;

  snapshot->index_create_requests = index_create_requests();
  snapshot->index_drop_requests = index_drop_requests();
  snapshot->metadata_load_failures = metadata_load_failures();
  snapshot->metadata_persist_failures = metadata_persist_failures();
  snapshot->committed_load_failures = committed_load_failures();
  snapshot->committed_persist_failures = committed_persist_failures();
  snapshot->manifest_load_failures = manifest_load_failures();
  snapshot->manifest_persist_failures = manifest_persist_failures();
  snapshot->change_log_load_failures = change_log_load_failures();
  snapshot->change_log_replay_failures = change_log_replay_failures();
  snapshot->change_log_persist_failures = change_log_persist_failures();
  snapshot->backend_recover_fallbacks = backend_recover_fallbacks();
  snapshot->registered_indexes = registered_indexes();
  snapshot->committed_snapshot_rows = committed_snapshot_rows();
  snapshot->manifest_version = manifest_version();
  snapshot->manifest_metadata_checkpoint = manifest_metadata_checkpoint();
  snapshot->manifest_committed_checkpoint = manifest_committed_checkpoint();
  snapshot->manifest_change_log_checkpoint = manifest_change_log_checkpoint();

  snapshot->rebuild_requests = rebuild_requests();
  snapshot->recover_requests = recover_requests();
  snapshot->rebuild_all_requests = rebuild_all_requests();
  snapshot->recover_all_requests = recover_all_requests();

  snapshot->search_requests = search_requests();
  snapshot->search_failures = search_failures();
  snapshot->search_results_returned = search_results_returned();
  snapshot->search_mvcc_candidate_rows = search_mvcc_candidate_rows();
  snapshot->search_mvcc_expansions = search_mvcc_expansions();
  snapshot->search_mvcc_limit_hits = search_mvcc_limit_hits();

  snapshot->stage_upsert_requests = stage_upsert_requests();
  snapshot->stage_erase_requests = stage_erase_requests();
  snapshot->txn_commit_requests = txn_commit_requests();
  snapshot->txn_commit_failures = txn_commit_failures();
  snapshot->txn_rollback_requests = txn_rollback_requests();
  snapshot->runtime_state_rollbacks = runtime_state_rollbacks();
  snapshot->runtime_state_rollback_failures = runtime_state_rollback_failures();
  snapshot->persist_artifact_rollbacks = persist_artifact_rollbacks();
  snapshot->persist_artifact_rollback_failures =
      persist_artifact_rollback_failures();
  snapshot->truth_store_persist_requests = truth_store_persist_requests();
  snapshot->truth_store_persist_failures = truth_store_persist_failures();
  snapshot->truth_store_delta_persist_requests =
      truth_store_delta_persist_requests();
  snapshot->truth_store_delta_persist_failures =
      truth_store_delta_persist_failures();
  snapshot->truth_store_compact_requests = truth_store_compact_requests();
  snapshot->truth_store_compact_failures = truth_store_compact_failures();
  snapshot->pending_txn_changes = pending_txn_changes();
  snapshot->apply_latency_ms = apply_latency_ms();
}

void record_index_create_request() { add_counter(&g_index_create_requests); }

void record_index_create_success() { add_counter(&g_registered_indexes); }

void record_index_drop_request() { add_counter(&g_index_drop_requests); }

void record_index_drop_success() { subtract_counter(&g_registered_indexes); }

void set_registered_indexes(uint64_t count) {
  g_registered_indexes.store(count, std::memory_order_relaxed);
}

void record_metadata_load_failure() { add_counter(&g_metadata_load_failures); }

void record_metadata_persist_failure() { add_counter(&g_metadata_persist_failures); }

void record_committed_load_failure() { add_counter(&g_committed_load_failures); }

void record_committed_persist_failure() {
  add_counter(&g_committed_persist_failures);
}

void record_manifest_load_failure() { add_counter(&g_manifest_load_failures); }

void record_manifest_persist_failure() {
  add_counter(&g_manifest_persist_failures);
}

void record_change_log_load_failure() { add_counter(&g_change_log_load_failures); }

void record_change_log_replay_failure() {
  add_counter(&g_change_log_replay_failures);
}

void record_change_log_persist_failure() {
  add_counter(&g_change_log_persist_failures);
}

void record_backend_recover_fallback() {
  add_counter(&g_backend_recover_fallbacks);
}

void set_committed_snapshot_rows(uint64_t count) {
  g_committed_snapshot_rows.store(count, std::memory_order_relaxed);
}

void set_manifest_version(uint64_t value) {
  g_manifest_version.store(value, std::memory_order_relaxed);
}

void set_manifest_metadata_checkpoint(uint64_t value) {
  g_manifest_metadata_checkpoint.store(value, std::memory_order_relaxed);
}

void set_manifest_committed_checkpoint(uint64_t value) {
  g_manifest_committed_checkpoint.store(value, std::memory_order_relaxed);
}

void set_manifest_change_log_checkpoint(uint64_t value) {
  g_manifest_change_log_checkpoint.store(value, std::memory_order_relaxed);
}

void record_rebuild_request() { add_counter(&g_rebuild_requests); }

void record_recover_request() { add_counter(&g_recover_requests); }

void record_rebuild_all_request() { add_counter(&g_rebuild_all_requests); }

void record_recover_all_request() { add_counter(&g_recover_all_requests); }

void record_search_request() { add_counter(&g_search_requests); }

void record_search_failure() { add_counter(&g_search_failures); }

void record_search_results_returned(size_t result_count) {
  add_counter(&g_search_results_returned, result_count);
}

void record_search_mvcc_candidate_rows(size_t candidate_count) {
  add_counter(&g_search_mvcc_candidate_rows, candidate_count);
}

void record_search_mvcc_expansion() {
  add_counter(&g_search_mvcc_expansions);
}

void record_search_mvcc_limit_hit() {
  add_counter(&g_search_mvcc_limit_hits);
}

void record_stage_upsert_request() { add_counter(&g_stage_upsert_requests); }

void record_stage_erase_request() { add_counter(&g_stage_erase_requests); }

void record_txn_commit_request() { add_counter(&g_txn_commit_requests); }

void record_txn_commit_failure() { add_counter(&g_txn_commit_failures); }

void record_txn_rollback_request() { add_counter(&g_txn_rollback_requests); }

void record_runtime_state_rollback() { add_counter(&g_runtime_state_rollbacks); }

void record_runtime_state_rollback_failure() {
  add_counter(&g_runtime_state_rollback_failures);
}

void record_persist_artifact_rollback() {
  add_counter(&g_persist_artifact_rollbacks);
}

void record_persist_artifact_rollback_failure() {
  add_counter(&g_persist_artifact_rollback_failures);
}

void record_truth_store_persist_request() {
  add_counter(&g_truth_store_persist_requests);
}

void record_truth_store_persist_failure() {
  add_counter(&g_truth_store_persist_failures);
}

void record_truth_store_delta_persist_request() {
  add_counter(&g_truth_store_delta_persist_requests);
}

void record_truth_store_delta_persist_failure() {
  add_counter(&g_truth_store_delta_persist_failures);
}

void record_truth_store_compact_request() {
  add_counter(&g_truth_store_compact_requests);
}

void record_truth_store_compact_failure() {
  add_counter(&g_truth_store_compact_failures);
}

void set_apply_latency_ms(uint64_t value) {
  g_apply_latency_ms.store(value, std::memory_order_relaxed);
}

void add_pending_txn_changes(size_t count) { add_counter(&g_pending_txn_changes, count); }

void subtract_pending_txn_changes(size_t count) {
  subtract_counter(&g_pending_txn_changes, count);
}

uint64_t index_create_requests() { return read_counter(g_index_create_requests); }

uint64_t index_drop_requests() { return read_counter(g_index_drop_requests); }

uint64_t metadata_load_failures() { return read_counter(g_metadata_load_failures); }

uint64_t metadata_persist_failures() {
  return read_counter(g_metadata_persist_failures);
}

uint64_t committed_load_failures() { return read_counter(g_committed_load_failures); }

uint64_t committed_persist_failures() {
  return read_counter(g_committed_persist_failures);
}

uint64_t manifest_load_failures() { return read_counter(g_manifest_load_failures); }

uint64_t manifest_persist_failures() {
  return read_counter(g_manifest_persist_failures);
}

uint64_t change_log_load_failures() {
  return read_counter(g_change_log_load_failures);
}

uint64_t change_log_replay_failures() {
  return read_counter(g_change_log_replay_failures);
}

uint64_t change_log_persist_failures() {
  return read_counter(g_change_log_persist_failures);
}

uint64_t backend_recover_fallbacks() {
  return read_counter(g_backend_recover_fallbacks);
}

uint64_t registered_indexes() { return read_counter(g_registered_indexes); }

uint64_t committed_snapshot_rows() { return read_counter(g_committed_snapshot_rows); }

uint64_t manifest_version() { return read_counter(g_manifest_version); }

uint64_t manifest_metadata_checkpoint() {
  return read_counter(g_manifest_metadata_checkpoint);
}

uint64_t manifest_committed_checkpoint() {
  return read_counter(g_manifest_committed_checkpoint);
}

uint64_t manifest_change_log_checkpoint() {
  return read_counter(g_manifest_change_log_checkpoint);
}

uint64_t rebuild_requests() { return read_counter(g_rebuild_requests); }

uint64_t recover_requests() { return read_counter(g_recover_requests); }

uint64_t rebuild_all_requests() { return read_counter(g_rebuild_all_requests); }

uint64_t recover_all_requests() { return read_counter(g_recover_all_requests); }

uint64_t search_requests() { return read_counter(g_search_requests); }

uint64_t search_failures() { return read_counter(g_search_failures); }

uint64_t search_results_returned() { return read_counter(g_search_results_returned); }

uint64_t search_mvcc_candidate_rows() {
  return read_counter(g_search_mvcc_candidate_rows);
}

uint64_t search_mvcc_expansions() {
  return read_counter(g_search_mvcc_expansions);
}

uint64_t search_mvcc_limit_hits() {
  return read_counter(g_search_mvcc_limit_hits);
}

uint64_t stage_upsert_requests() { return read_counter(g_stage_upsert_requests); }

uint64_t stage_erase_requests() { return read_counter(g_stage_erase_requests); }

uint64_t txn_commit_requests() { return read_counter(g_txn_commit_requests); }

uint64_t txn_commit_failures() { return read_counter(g_txn_commit_failures); }

uint64_t txn_rollback_requests() { return read_counter(g_txn_rollback_requests); }

uint64_t runtime_state_rollbacks() {
  return read_counter(g_runtime_state_rollbacks);
}

uint64_t runtime_state_rollback_failures() {
  return read_counter(g_runtime_state_rollback_failures);
}

uint64_t persist_artifact_rollbacks() {
  return read_counter(g_persist_artifact_rollbacks);
}

uint64_t persist_artifact_rollback_failures() {
  return read_counter(g_persist_artifact_rollback_failures);
}

uint64_t truth_store_persist_requests() {
  return read_counter(g_truth_store_persist_requests);
}

uint64_t truth_store_persist_failures() {
  return read_counter(g_truth_store_persist_failures);
}

uint64_t truth_store_delta_persist_requests() {
  return read_counter(g_truth_store_delta_persist_requests);
}

uint64_t truth_store_delta_persist_failures() {
  return read_counter(g_truth_store_delta_persist_failures);
}

uint64_t truth_store_compact_requests() {
  return read_counter(g_truth_store_compact_requests);
}

uint64_t truth_store_compact_failures() {
  return read_counter(g_truth_store_compact_failures);
}

uint64_t pending_txn_changes() { return read_counter(g_pending_txn_changes); }

uint64_t apply_latency_ms() { return read_counter(g_apply_latency_ms); }

}  // namespace vector_status
