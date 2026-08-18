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

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "sql/vector/vector_status.h"

namespace vector_status_unittest {

TEST(VectorStatusTest, CountersAndGaugesAreUpdated) {
  /* Normalize mutable gauges to avoid cross-test interference. */
  vector_status::subtract_pending_txn_changes(
      vector_status::pending_txn_changes());
  vector_status::set_apply_latency_ms(0);

  const uint64_t create_before = vector_status::index_create_requests();
  const uint64_t drop_before = vector_status::index_drop_requests();
  const uint64_t metadata_load_fail_before = vector_status::metadata_load_failures();
  const uint64_t metadata_persist_fail_before =
      vector_status::metadata_persist_failures();
  const uint64_t manifest_load_fail_before =
      vector_status::manifest_load_failures();
  const uint64_t manifest_persist_fail_before =
      vector_status::manifest_persist_failures();
  const uint64_t change_log_load_fail_before =
      vector_status::change_log_load_failures();
  const uint64_t change_log_replay_fail_before =
      vector_status::change_log_replay_failures();
  const uint64_t change_log_persist_fail_before =
      vector_status::change_log_persist_failures();
  const uint64_t backend_recover_fallback_before =
      vector_status::backend_recover_fallbacks();
  const uint64_t registered_before = vector_status::registered_indexes();
  const uint64_t manifest_version_before = vector_status::manifest_version();
  const uint64_t manifest_meta_ckpt_before =
      vector_status::manifest_metadata_checkpoint();
  const uint64_t manifest_committed_ckpt_before =
      vector_status::manifest_committed_checkpoint();
  const uint64_t manifest_change_log_ckpt_before =
      vector_status::manifest_change_log_checkpoint();

  const uint64_t rebuild_before = vector_status::rebuild_requests();
  const uint64_t recover_before = vector_status::recover_requests();
  const uint64_t rebuild_all_before = vector_status::rebuild_all_requests();
  const uint64_t recover_all_before = vector_status::recover_all_requests();

  const uint64_t search_before = vector_status::search_requests();
  const uint64_t search_fail_before = vector_status::search_failures();
  const uint64_t search_result_before = vector_status::search_results_returned();
  const uint64_t search_mvcc_candidates_before =
      vector_status::search_mvcc_candidate_rows();
  const uint64_t search_mvcc_expansions_before =
      vector_status::search_mvcc_expansions();
  const uint64_t search_mvcc_limit_hits_before =
      vector_status::search_mvcc_limit_hits();

  const uint64_t stage_upsert_before = vector_status::stage_upsert_requests();
  const uint64_t stage_erase_before = vector_status::stage_erase_requests();
  const uint64_t commit_before = vector_status::txn_commit_requests();
  const uint64_t commit_fail_before = vector_status::txn_commit_failures();
  const uint64_t rollback_before = vector_status::txn_rollback_requests();
  const uint64_t runtime_rollback_before =
      vector_status::runtime_state_rollbacks();
  const uint64_t runtime_rollback_fail_before =
      vector_status::runtime_state_rollback_failures();
  const uint64_t artifact_rollback_before =
      vector_status::persist_artifact_rollbacks();
  const uint64_t artifact_rollback_fail_before =
      vector_status::persist_artifact_rollback_failures();
  const uint64_t truth_store_delta_persist_before =
      vector_status::truth_store_delta_persist_requests();
  const uint64_t truth_store_delta_persist_fail_before =
      vector_status::truth_store_delta_persist_failures();
  const uint64_t truth_store_compact_before =
      vector_status::truth_store_compact_requests();
  const uint64_t truth_store_compact_fail_before =
      vector_status::truth_store_compact_failures();
  const uint64_t pending_before = vector_status::pending_txn_changes();
  const uint64_t apply_latency_before = vector_status::apply_latency_ms();

  vector_status::record_index_create_request();
  vector_status::record_index_create_success();
  vector_status::record_index_drop_request();
  vector_status::record_index_drop_success();
  vector_status::record_metadata_load_failure();
  vector_status::record_metadata_persist_failure();
  vector_status::record_manifest_load_failure();
  vector_status::record_manifest_persist_failure();
  vector_status::record_change_log_load_failure();
  vector_status::record_change_log_replay_failure();
  vector_status::record_change_log_persist_failure();
  vector_status::record_backend_recover_fallback();
  vector_status::set_manifest_version(manifest_version_before + 10);
  vector_status::set_manifest_metadata_checkpoint(manifest_meta_ckpt_before + 3);
  vector_status::set_manifest_committed_checkpoint(
      manifest_committed_ckpt_before + 4);
  vector_status::set_manifest_change_log_checkpoint(
      manifest_change_log_ckpt_before + 5);

  vector_status::record_rebuild_request();
  vector_status::record_recover_request();
  vector_status::record_rebuild_all_request();
  vector_status::record_recover_all_request();

  vector_status::record_search_request();
  vector_status::record_search_failure();
  vector_status::record_search_results_returned(3);
  vector_status::record_search_mvcc_candidate_rows(7);
  vector_status::record_search_mvcc_expansion();
  vector_status::record_search_mvcc_limit_hit();

  vector_status::record_stage_upsert_request();
  vector_status::record_stage_erase_request();
  vector_status::record_txn_commit_request();
  vector_status::record_txn_commit_failure();
  vector_status::record_txn_rollback_request();
  vector_status::record_runtime_state_rollback();
  vector_status::record_runtime_state_rollback_failure();
  vector_status::record_persist_artifact_rollback();
  vector_status::record_persist_artifact_rollback_failure();
  vector_status::record_truth_store_delta_persist_request();
  vector_status::record_truth_store_delta_persist_failure();
  vector_status::record_truth_store_compact_request();
  vector_status::record_truth_store_compact_failure();
  vector_status::set_apply_latency_ms(apply_latency_before + 7);

  vector_status::add_pending_txn_changes(5);
  vector_status::subtract_pending_txn_changes(2);

  EXPECT_EQ(create_before + 1, vector_status::index_create_requests());
  EXPECT_EQ(drop_before + 1, vector_status::index_drop_requests());
  EXPECT_EQ(metadata_load_fail_before + 1, vector_status::metadata_load_failures());
  EXPECT_EQ(metadata_persist_fail_before + 1,
            vector_status::metadata_persist_failures());
  EXPECT_EQ(manifest_load_fail_before + 1,
            vector_status::manifest_load_failures());
  EXPECT_EQ(manifest_persist_fail_before + 1,
            vector_status::manifest_persist_failures());
  EXPECT_EQ(change_log_load_fail_before + 1,
            vector_status::change_log_load_failures());
  EXPECT_EQ(change_log_replay_fail_before + 1,
            vector_status::change_log_replay_failures());
  EXPECT_EQ(change_log_persist_fail_before + 1,
            vector_status::change_log_persist_failures());
  EXPECT_EQ(backend_recover_fallback_before + 1,
            vector_status::backend_recover_fallbacks());
  EXPECT_EQ(registered_before, vector_status::registered_indexes());
  EXPECT_EQ(manifest_version_before + 10, vector_status::manifest_version());
  EXPECT_EQ(manifest_meta_ckpt_before + 3,
            vector_status::manifest_metadata_checkpoint());
  EXPECT_EQ(manifest_committed_ckpt_before + 4,
            vector_status::manifest_committed_checkpoint());
  EXPECT_EQ(manifest_change_log_ckpt_before + 5,
            vector_status::manifest_change_log_checkpoint());

  EXPECT_EQ(rebuild_before + 1, vector_status::rebuild_requests());
  EXPECT_EQ(recover_before + 1, vector_status::recover_requests());
  EXPECT_EQ(rebuild_all_before + 1, vector_status::rebuild_all_requests());
  EXPECT_EQ(recover_all_before + 1, vector_status::recover_all_requests());

  EXPECT_EQ(search_before + 1, vector_status::search_requests());
  EXPECT_EQ(search_fail_before + 1, vector_status::search_failures());
  EXPECT_EQ(search_result_before + 3, vector_status::search_results_returned());
  EXPECT_EQ(search_mvcc_candidates_before + 7,
            vector_status::search_mvcc_candidate_rows());
  EXPECT_EQ(search_mvcc_expansions_before + 1,
            vector_status::search_mvcc_expansions());
  EXPECT_EQ(search_mvcc_limit_hits_before + 1,
            vector_status::search_mvcc_limit_hits());

  EXPECT_EQ(stage_upsert_before + 1, vector_status::stage_upsert_requests());
  EXPECT_EQ(stage_erase_before + 1, vector_status::stage_erase_requests());
  EXPECT_EQ(commit_before + 1, vector_status::txn_commit_requests());
  EXPECT_EQ(commit_fail_before + 1, vector_status::txn_commit_failures());
  EXPECT_EQ(rollback_before + 1, vector_status::txn_rollback_requests());
  EXPECT_EQ(runtime_rollback_before + 1, vector_status::runtime_state_rollbacks());
  EXPECT_EQ(runtime_rollback_fail_before + 1,
            vector_status::runtime_state_rollback_failures());
  EXPECT_EQ(artifact_rollback_before + 1,
            vector_status::persist_artifact_rollbacks());
  EXPECT_EQ(artifact_rollback_fail_before + 1,
            vector_status::persist_artifact_rollback_failures());
  EXPECT_EQ(truth_store_delta_persist_before + 1,
            vector_status::truth_store_delta_persist_requests());
  EXPECT_EQ(truth_store_delta_persist_fail_before + 1,
            vector_status::truth_store_delta_persist_failures());
  EXPECT_EQ(truth_store_compact_before + 1,
            vector_status::truth_store_compact_requests());
  EXPECT_EQ(truth_store_compact_fail_before + 1,
            vector_status::truth_store_compact_failures());
  EXPECT_EQ(pending_before + 3, vector_status::pending_txn_changes());
  EXPECT_EQ(apply_latency_before + 7, vector_status::apply_latency_ms());

  vector_status::snapshot snapshot;
  vector_status::read_snapshot(&snapshot);
  EXPECT_EQ(apply_latency_before + 7, snapshot.apply_latency_ms);
  EXPECT_EQ(search_mvcc_candidates_before + 7,
            snapshot.search_mvcc_candidate_rows);
  EXPECT_EQ(search_mvcc_expansions_before + 1,
            snapshot.search_mvcc_expansions);
  EXPECT_EQ(search_mvcc_limit_hits_before + 1,
            snapshot.search_mvcc_limit_hits);
  EXPECT_EQ(truth_store_delta_persist_before + 1,
            snapshot.truth_store_delta_persist_requests);
  EXPECT_EQ(truth_store_delta_persist_fail_before + 1,
            snapshot.truth_store_delta_persist_failures);
  EXPECT_EQ(truth_store_compact_before + 1,
            snapshot.truth_store_compact_requests);
  EXPECT_EQ(truth_store_compact_fail_before + 1,
            snapshot.truth_store_compact_failures);

  vector_status::subtract_pending_txn_changes(1000);
  EXPECT_EQ(0U, vector_status::pending_txn_changes());
}

TEST(VectorStatusTest, ReadSnapshotIgnoresNullTarget) {
  vector_status::read_snapshot(nullptr);
}

TEST(VectorStatusTest, PendingCounterHandlesConcurrentSubtraction) {
  vector_status::subtract_pending_txn_changes(
      vector_status::pending_txn_changes());
  constexpr size_t kThreadCount = 16;
  constexpr size_t kIterations = 2000;
  vector_status::add_pending_txn_changes(kThreadCount * kIterations);

  std::atomic<size_t> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (size_t thread = 0; thread < kThreadCount; ++thread) {
    workers.emplace_back([&] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (size_t i = 0; i < kIterations; ++i) {
        vector_status::subtract_pending_txn_changes(1);
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (std::thread &worker : workers) worker.join();
  EXPECT_EQ(0U, vector_status::pending_txn_changes());

  vector_status::subtract_pending_txn_changes(1);
  EXPECT_EQ(0U, vector_status::pending_txn_changes());
}

}  // namespace vector_status_unittest
