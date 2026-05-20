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

#ifndef SQL_VECTOR_STATUS_INCLUDED
#define SQL_VECTOR_STATUS_INCLUDED

#include <cstddef>
#include <cstdint>

namespace vector_status {

struct snapshot {
  uint64_t index_create_requests{0};
  uint64_t index_drop_requests{0};
  uint64_t metadata_load_failures{0};
  uint64_t metadata_persist_failures{0};
  uint64_t committed_load_failures{0};
  uint64_t committed_persist_failures{0};
  uint64_t manifest_load_failures{0};
  uint64_t manifest_persist_failures{0};
  uint64_t change_log_load_failures{0};
  uint64_t change_log_replay_failures{0};
  uint64_t change_log_persist_failures{0};
  uint64_t backend_recover_fallbacks{0};
  uint64_t registered_indexes{0};
  uint64_t committed_snapshot_rows{0};
  uint64_t manifest_version{0};
  uint64_t manifest_metadata_checkpoint{0};
  uint64_t manifest_committed_checkpoint{0};
  uint64_t manifest_change_log_checkpoint{0};

  uint64_t rebuild_requests{0};
  uint64_t recover_requests{0};
  uint64_t rebuild_all_requests{0};
  uint64_t recover_all_requests{0};

  uint64_t search_requests{0};
  uint64_t search_failures{0};
  uint64_t search_results_returned{0};
  uint64_t search_mvcc_candidate_rows{0};
  uint64_t search_mvcc_expansions{0};
  uint64_t search_mvcc_limit_hits{0};

  uint64_t stage_upsert_requests{0};
  uint64_t stage_erase_requests{0};
  uint64_t txn_commit_requests{0};
  uint64_t txn_commit_failures{0};
  uint64_t txn_rollback_requests{0};
  uint64_t runtime_state_rollbacks{0};
  uint64_t runtime_state_rollback_failures{0};
  uint64_t persist_artifact_rollbacks{0};
  uint64_t persist_artifact_rollback_failures{0};
  uint64_t truth_store_persist_requests{0};
  uint64_t truth_store_persist_failures{0};
  uint64_t truth_store_delta_persist_requests{0};
  uint64_t truth_store_delta_persist_failures{0};
  uint64_t truth_store_compact_requests{0};
  uint64_t truth_store_compact_failures{0};
  uint64_t pending_txn_changes{0};
  uint64_t apply_latency_ms{0};
};

void read_snapshot(snapshot *snapshot);

void record_index_create_request();
void record_index_create_success();
void record_index_drop_request();
void record_index_drop_success();
void set_registered_indexes(uint64_t count);
void record_metadata_load_failure();
void record_metadata_persist_failure();
void record_committed_load_failure();
void record_committed_persist_failure();
void record_manifest_load_failure();
void record_manifest_persist_failure();
void record_change_log_load_failure();
void record_change_log_replay_failure();
void record_change_log_persist_failure();
void record_backend_recover_fallback();
void set_committed_snapshot_rows(uint64_t count);
void set_manifest_version(uint64_t value);
void set_manifest_metadata_checkpoint(uint64_t value);
void set_manifest_committed_checkpoint(uint64_t value);
void set_manifest_change_log_checkpoint(uint64_t value);

void record_rebuild_request();
void record_recover_request();
void record_rebuild_all_request();
void record_recover_all_request();

void record_search_request();
void record_search_failure();
void record_search_results_returned(size_t result_count);
void record_search_mvcc_candidate_rows(size_t candidate_count);
void record_search_mvcc_expansion();
void record_search_mvcc_limit_hit();

void record_stage_upsert_request();
void record_stage_erase_request();

void record_txn_commit_request();
void record_txn_commit_failure();
void record_txn_rollback_request();
void record_runtime_state_rollback();
void record_runtime_state_rollback_failure();
void record_persist_artifact_rollback();
void record_persist_artifact_rollback_failure();
void record_truth_store_persist_request();
void record_truth_store_persist_failure();
void record_truth_store_delta_persist_request();
void record_truth_store_delta_persist_failure();
void record_truth_store_compact_request();
void record_truth_store_compact_failure();
void set_apply_latency_ms(uint64_t value);

void add_pending_txn_changes(size_t count);
void subtract_pending_txn_changes(size_t count);

uint64_t index_create_requests();
uint64_t index_drop_requests();
uint64_t metadata_load_failures();
uint64_t metadata_persist_failures();
uint64_t committed_load_failures();
uint64_t committed_persist_failures();
uint64_t manifest_load_failures();
uint64_t manifest_persist_failures();
uint64_t change_log_load_failures();
uint64_t change_log_replay_failures();
uint64_t change_log_persist_failures();
uint64_t backend_recover_fallbacks();
uint64_t registered_indexes();
uint64_t committed_snapshot_rows();
uint64_t manifest_version();
uint64_t manifest_metadata_checkpoint();
uint64_t manifest_committed_checkpoint();
uint64_t manifest_change_log_checkpoint();

uint64_t rebuild_requests();
uint64_t recover_requests();
uint64_t rebuild_all_requests();
uint64_t recover_all_requests();

uint64_t search_requests();
uint64_t search_failures();
uint64_t search_results_returned();
uint64_t search_mvcc_candidate_rows();
uint64_t search_mvcc_expansions();
uint64_t search_mvcc_limit_hits();

uint64_t stage_upsert_requests();
uint64_t stage_erase_requests();

uint64_t txn_commit_requests();
uint64_t txn_commit_failures();
uint64_t txn_rollback_requests();
uint64_t runtime_state_rollbacks();
uint64_t runtime_state_rollback_failures();
uint64_t persist_artifact_rollbacks();
uint64_t persist_artifact_rollback_failures();
uint64_t truth_store_persist_requests();
uint64_t truth_store_persist_failures();
uint64_t truth_store_delta_persist_requests();
uint64_t truth_store_delta_persist_failures();
uint64_t truth_store_compact_requests();
uint64_t truth_store_compact_failures();

uint64_t pending_txn_changes();
uint64_t apply_latency_ms();

}  // namespace vector_status

#endif  // SQL_VECTOR_STATUS_INCLUDED
