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
#include <cmath>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "my_byteorder.h"
#include "my_dbug.h"
#include "sql/debug_sync.h"
#include "sql/sql_class.h"
#include "sql/vector/vector_index_diagnostics.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_registry_internal.h"
#include "sql/vector/vector_index_runtime_config.h"
#include "sql/vector/vector_index_service_internal.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_load_file.h"
#include "sql/vector/vector_load_staging.h"
#include "sql/vector/vector_mapped_search.h"
#include "sql/vector/vector_statement_publication.h"
#include "sql/vector/vector_status.h"
#include "sql/vector/vector_trx_participant.h"

namespace vector_index_registry {

using namespace detail;

namespace {

std::atomic<uint64_t> g_publication_recovery_requests{1};
std::atomic<uint64_t> g_publication_recovery_completed{0};
std::mutex g_publication_recovery_mutex;
thread_local bool g_publication_recovery_active{false};
thread_local size_t g_publication_read_scope_depth{0};
constexpr uint64_t k_batch_load_payload = 1;

enum create_unsigned_field : size_t {
  kCreateDimension = 0,
  kCreateBuildThreads,
  kCreateDiskannMaxDegree,
  kCreateDiskannBuildComplexity,
  kCreateDiskannPqCodeBudgetSize,
  kCreateDiskannDiskPqDims,
  kCreateDiskannAccelerateBuild,
  kCreateDiskannShuffleBuild,
  kCreateDiskannUseBfsCache,
  kCreateDiskannSearchComplexity,
  kCreateDiskannSearchBeamwidth,
  kCreateConsistencyMode,
  kCreateUnsignedFieldCount,
};

enum create_string_field : size_t {
  kCreateMetric = 0,
  kCreateMode,
  kCreateProvider,
  kCreateOwnerSchema,
  kCreateLifecycleState,
  kCreateStringFieldCount,
};

class publication_recovery_scope {
 public:
  publication_recovery_scope() { g_publication_recovery_active = true; }
  ~publication_recovery_scope() { g_publication_recovery_active = false; }
};

void request_publication_intent_recovery() {
  g_publication_recovery_requests.fetch_add(1, std::memory_order_release);
}

bool publication_intent_recovery_pending() {
  const uint64_t requested =
      g_publication_recovery_requests.load(std::memory_order_acquire);
  return g_publication_recovery_completed.load(std::memory_order_acquire) <
         requested;
}

bool rollback_staged_statement_locked(thd_txn_context *ctx) {
  if (ctx == nullptr || !ctx->stmt_savepoint_active) return false;

  const size_t pending_before =
      g_index_service.pending_change_count(ctx->txn_id);
  if (!g_index_service.rollback_to_savepoint(ctx->txn_id,
                                             ctx->stmt_savepoint_name)) {
    return false;
  }
  const size_t pending_after =
      g_index_service.pending_change_count(ctx->txn_id);
  if (ctx->attached_dml) {
    if (pending_after > ctx->durable_change_log_rows.size()) return false;
    ctx->durable_change_log_rows.resize(pending_after);
  }
  if (pending_before > pending_after) {
    vector_status::subtract_pending_txn_changes(pending_before - pending_after);
  }
  DBUG_EXECUTE_IF("vector_registry_fail_release_stmt_savepoint_after_rollback",
                  return false;);
  if (!g_index_service.release_savepoint(ctx->txn_id,
                                         ctx->stmt_savepoint_name)) {
    return false;
  }
  clear_stmt_savepoint_state(ctx);
  return true;
}

bool append_durable_change_log_rows_locked(
    const std::vector<vector_index_metadata_store::change_log_row> &rows) {
  try {
    const size_t existing_size = g_change_log_rows.size();
    g_change_log_rows.insert(g_change_log_rows.end(), rows.begin(), rows.end());
    std::inplace_merge(
        g_change_log_rows.begin(), g_change_log_rows.begin() + existing_size,
        g_change_log_rows.end(),
        [](const auto &lhs, const auto &rhs) {
          return lhs.sequence < rhs.sequence;
        });
  } catch (...) {
    return false;
  }
  for (const auto &row : rows) {
    if (row.sequence >= g_next_change_log_sequence) {
      g_next_change_log_sequence = row.sequence + 1;
    }
  }
  g_manifest_change_log_checkpoint = g_change_log_rows.size();
  return true;
}

bool erase_durable_change_log_sequences_locked(
    const std::vector<uint64_t> &sequences) {
  if (!std::is_sorted(sequences.begin(), sequences.end()) ||
      std::adjacent_find(sequences.begin(), sequences.end()) !=
          sequences.end()) {
    return false;
  }
  const size_t size_before = g_change_log_rows.size();
  g_change_log_rows.erase(
      std::remove_if(g_change_log_rows.begin(), g_change_log_rows.end(),
                     [&](const auto &row) {
                       return std::binary_search(sequences.begin(),
                                                 sequences.end(), row.sequence);
                     }),
      g_change_log_rows.end());
  if (size_before - g_change_log_rows.size() != sequences.size()) return false;
  g_manifest_change_log_checkpoint = g_change_log_rows.size();
  return true;
}

bool snapshot_durable_change_log_sequences_locked(
    std::vector<uint64_t> *sequences) {
  if (sequences == nullptr) return false;
  try {
    sequences->clear();
    sequences->reserve(g_change_log_rows.size());
    for (const auto &row : g_change_log_rows)
      sequences->push_back(row.sequence);
  } catch (...) {
    sequences->clear();
    return false;
  }
  return std::is_sorted(sequences->begin(), sequences->end()) &&
         std::adjacent_find(sequences->begin(), sequences->end()) ==
             sequences->end();
}

bool bind_commit_truth_generations(
    std::vector<vector_index_metadata_store::change_log_row> *rows,
    vector_index::index_service::commit_build_plan *plan) {
  if (rows == nullptr || plan == nullptr) return false;
  std::unordered_map<std::string,
                     vector_index::index_service::commit_index_plan *>
      index_plans;
  for (auto &index_plan : plan->indexes) {
    if (index_plan.index_name.empty() ||
        index_plan.publication_before.index_identity == 0 ||
        index_plan.publication_before.truth_generation ==
            std::numeric_limits<uint64_t>::max()) {
      return false;
    }
    index_plan.target_truth_generation =
        index_plan.publication_before.truth_generation;
    if (!index_plans.emplace(index_plan.index_name, &index_plan).second) {
      return false;
    }
  }

  for (auto &row : *rows) {
    if (row.index_name.empty() || row.sequence == 0) return false;
    const auto plan_it = index_plans.find(row.index_name);
    if (plan_it == index_plans.end()) return false;
    const auto &publication = plan_it->second->publication_before;
    if (plan_it->second->target_truth_generation ==
        publication.truth_generation) {
      plan_it->second->target_truth_generation =
          publication.truth_generation + 1;
    }
    const uint64_t target_generation = plan_it->second->target_truth_generation;
    if (row.index_identity == 0) {
      row.index_identity = publication.index_identity;
    } else if (row.index_identity != publication.index_identity) {
      return false;
    }
    if (row.publication_id == 0) {
      row.publication_id = target_generation;
    } else if (row.publication_id != target_generation) {
      return false;
    }
    if (row.truth_generation == 0) {
      row.truth_generation = target_generation;
    } else if (row.truth_generation != target_generation) {
      return false;
    }
  }
  return true;
}

bool build_transactional_publication_intents_locked(
    uint64_t txn_id,
    const vector_index::index_service::commit_build_plan &build_plan,
    std::vector<vector_index_truth_store::publication_intent> *intents) {
  if (txn_id == 0 || intents == nullptr) return false;
  intents->clear();
  intents->reserve(build_plan.indexes.size());

  for (const auto &index_plan : build_plan.indexes) {
    if (index_plan.target_truth_generation ==
        index_plan.publication_before.truth_generation) {
      continue;
    }
    vector_index_truth_store::publication_intent intent;
    intent.index_name = index_plan.index_name;
    intent.txn_id = txn_id;
    intent.operation =
        vector_index_truth_store::publication_operation::kTransactionalDml;
    if (!describe_publication_token_locked(index_plan.index_name,
                                           &intent.expected) ||
        !intent.expected.exists ||
        intent.expected.index_identity !=
            index_plan.publication_before.index_identity ||
        intent.expected.truth_generation !=
            index_plan.publication_before.truth_generation ||
        intent.expected.config_generation !=
            index_plan.publication_before.config_generation ||
        intent.expected.artifact_generation !=
            index_plan.publication_before.artifact_generation ||
        intent.expected.runtime_generation !=
            index_plan.publication_before.runtime_generation ||
        index_plan.target_truth_generation == 0) {
      return false;
    }

    intent.target = intent.expected;
    intent.target.truth_generation = index_plan.target_truth_generation;
    std::string lifecycle_state;
    vector_index::index_service::index_config current_config;
    if (!g_index_service.describe_index(index_plan.index_name,
                                        &current_config, nullptr, nullptr,
                                        nullptr, &lifecycle_state)) {
      return false;
    }
    const bool publish_runtime =
        lifecycle_state != "bulk_loading" &&
        !index_plan.defer_runtime_rebuild;
    if (publish_runtime) {
      intent.target.runtime_generation = intent.target.truth_generation;
      // A runtime mutation advances an external artifact only when this index
      // already serves a published artifact. External mode alone does not
      // imply that an artifact exists (for example, the serial fallback).
      intent.target.artifact_generation =
          index_plan.config.mode == vector_index::backend_mode::kExternal &&
                  intent.expected.artifact_generation != 0
              ? intent.target.truth_generation
              : intent.expected.artifact_generation;
    }
    intent.publication_id = intent.target.truth_generation;
    intents->push_back(std::move(intent));
  }
  return true;
}

void set_publication_failure(std::string *failure_stage,
                             const std::string &stage) {
  if (failure_stage != nullptr) *failure_stage = stage;
}

const char *publication_token_mismatch_field(
    const vector_index_truth_store::publication_token &expected,
    const vector_index_truth_store::publication_token &current) {
  if (expected.exists != current.exists) return "exists";
  if (expected.index_identity != current.index_identity)
    return "index_identity";
  if (expected.truth_generation != current.truth_generation)
    return "truth_generation";
  if (expected.config_generation != current.config_generation)
    return "config_generation";
  if (expected.artifact_generation != current.artifact_generation)
    return "artifact_generation";
  if (expected.runtime_generation != current.runtime_generation)
    return "runtime_generation";
  if (expected.lifecycle_version != current.lifecycle_version)
    return "lifecycle_version";
  if (expected.source_generation != current.source_generation)
    return "source_generation";
  return "unknown";
}

bool publication_intent_target_matches(
    const vector_index_truth_store::publication_intent &intent,
    const vector_index_truth_store::publication_token &current) {
  if (publication_tokens_equal(current, intent.target)) return true;
  if (intent.operation !=
      vector_index_truth_store::publication_operation::kTransactionalDml) {
    return false;
  }

  const auto &target = intent.target;
  const bool stable_fields_match =
      current.exists == target.exists &&
      current.index_identity == target.index_identity &&
      current.truth_generation == target.truth_generation &&
      current.config_generation == target.config_generation &&
      current.lifecycle_version == target.lifecycle_version &&
      current.source_generation == target.source_generation;
  if (!stable_fields_match ||
      current.runtime_generation != current.truth_generation) {
    return false;
  }

  // Crash recovery may eagerly rebuild a runtime that the original commit
  // deliberately left deferred. The expected CAS token remains strict; only
  // a fully published runtime at the exact target generation is equivalent.
  // An external backend can create its first artifact, or remove the last
  // empty artifact, while that recovery build is applied.
  return current.artifact_generation == 0 ||
         current.artifact_generation == current.truth_generation;
}

bool token_advanced_from(
    const vector_index_truth_store::publication_token &expected,
    const vector_index_truth_store::publication_token &current) {
  return expected.exists && current.exists &&
         expected.index_identity == current.index_identity &&
         (current.truth_generation > expected.truth_generation ||
          current.config_generation > expected.config_generation ||
          current.artifact_generation > expected.artifact_generation ||
          current.runtime_generation > expected.runtime_generation ||
          current.lifecycle_version > expected.lifecycle_version ||
          current.source_generation > expected.source_generation);
}

bool statement_intent_can_resume_after_source_preparation(
    const vector_index_truth_store::publication_intent &intent,
    const vector_index_truth_store::publication_token &current) {
  using vector_index_truth_store::publication_operation;
  if (intent.operation != publication_operation::kRebuildIndex &&
      intent.operation != publication_operation::kBulkBuildIndex) {
    return false;
  }

  // Standalone rebuild preparation can durably rewrite equivalent input into
  // raw segments before runtime publication. The recovery gate prevents later
  // statements from mutating this index until the durable intent is replayed.
  const auto &expected = intent.expected;
  return expected.exists && current.exists &&
         expected.index_identity == current.index_identity &&
         expected.truth_generation == current.truth_generation &&
         expected.config_generation == current.config_generation &&
         expected.artifact_generation == current.artifact_generation &&
         expected.runtime_generation == current.runtime_generation &&
         expected.lifecycle_version == current.lifecycle_version &&
         current.source_generation > expected.source_generation;
}

bool decode_statement_payload(
    const vector_index_truth_store::publication_intent &intent,
    vector_statement_publication::operation_payload *payload,
    std::string *failure_stage) {
  if (vector_statement_publication::decode_operation_payload(intent.payload,
                                                             payload)) {
    return true;
  }
  set_publication_failure(failure_stage, "decode_statement_payload");
  return false;
}

bool statement_operation_has_payload(
    vector_index_truth_store::publication_operation operation) {
  using vector_index_truth_store::publication_operation;
  switch (operation) {
    case publication_operation::kStandaloneUpsert:
    case publication_operation::kStandaloneErase:
    case publication_operation::kBulkLoad:
    case publication_operation::kCreateIndex:
    case publication_operation::kUpdateConfig:
      return true;
    case publication_operation::kTransactionalDml:
    case publication_operation::kDropIndex:
    case publication_operation::kRebuildIndex:
    case publication_operation::kRecoverIndex:
    case publication_operation::kBeginBulkLoad:
    case publication_operation::kBulkBuildIndex:
      return false;
  }
  return false;
}

bool apply_config_statement(
    const vector_index_truth_store::publication_intent &intent,
    const vector_statement_publication::operation_payload &payload) {
  using vector_statement_publication::config_change;
  const auto &values = payload.unsigned_values;
  switch (payload.config) {
    case config_change::kSearchEf:
      return values.size() == 1 &&
             set_search_ef(intent.index_name,
                           static_cast<uint32_t>(values[0]));
    case config_change::kHnswBuildParams:
      return values.size() == 2 &&
             set_hnsw_build_params(intent.index_name,
                                   static_cast<uint32_t>(values[0]),
                                   static_cast<uint32_t>(values[1]));
    case config_change::kFaissIvfParams:
      return values.size() == 2 &&
             set_faiss_ivf_params(intent.index_name,
                                  static_cast<uint32_t>(values[0]),
                                  static_cast<uint32_t>(values[1]));
    case config_change::kFaissIvfPqParams:
      return values.size() == 4 &&
             set_faiss_ivf_pq_params(
                 intent.index_name, static_cast<uint32_t>(values[0]),
                 static_cast<uint32_t>(values[1]),
                 static_cast<uint32_t>(values[2]),
                 static_cast<uint32_t>(values[3]));
    case config_change::kDiskannBuildParams:
      return values.size() == 3 &&
             set_diskann_build_params(intent.index_name,
                                      static_cast<uint32_t>(values[0]),
                                      static_cast<uint32_t>(values[1]),
                                      static_cast<uint32_t>(values[2]));
    case config_change::kDiskannSearchComplexity:
      return values.size() == 1 &&
             set_diskann_search_complexity(
                 intent.index_name, static_cast<uint32_t>(values[0]));
    case config_change::kDiskannSearchBeamwidth:
      return values.size() == 1 &&
             set_diskann_search_beamwidth(
                 intent.index_name, static_cast<uint32_t>(values[0]));
    case config_change::kDiskannPqCodeBudgetSize:
      return values.size() == 1 &&
             set_diskann_pq_code_budget_size(intent.index_name, values[0]);
    case config_change::kDiskannDiskPqDims:
      return values.size() == 1 &&
             set_diskann_disk_pq_dims(intent.index_name,
                                      static_cast<uint32_t>(values[0]));
    case config_change::kDiskannAccelerateBuild:
      return values.size() == 1 && values[0] <= 1 &&
             set_diskann_accelerate_build(intent.index_name, values[0] != 0);
    case config_change::kDiskannShuffleBuild:
      return values.size() == 1 && values[0] <= 1 &&
             set_diskann_shuffle_build(intent.index_name, values[0] != 0);
    case config_change::kDiskannUseBfsCache:
      return values.size() == 1 && values[0] <= 1 &&
             set_diskann_use_bfs_cache(intent.index_name, values[0] != 0);
    case config_change::kDiskannBuildMode:
      return values.size() == 1 &&
             set_diskann_build_mode(
                 intent.index_name,
                 static_cast<vector_index::diskann_build_mode>(values[0]));
  }
  return false;
}

bool config_statement_applied(
    const vector_index_truth_store::publication_intent &intent,
    const vector_statement_publication::operation_payload &payload) {
  vector_index_registry::index_info info;
  if (!get_index_info(intent.index_name, &info)) return false;
  const auto &values = payload.unsigned_values;
  using vector_statement_publication::config_change;
  switch (payload.config) {
    case config_change::kSearchEf:
      return values.size() == 1 && info.search_ef == values[0];
    case config_change::kHnswBuildParams:
      return values.size() == 2 && info.hnsw_m == values[0] &&
             info.hnsw_ef_construction == values[1];
    case config_change::kFaissIvfParams:
      return values.size() == 2 && info.faiss_nlist == values[0] &&
             info.faiss_nprobe == values[1];
    case config_change::kFaissIvfPqParams:
      return values.size() == 4 && info.faiss_nlist == values[0] &&
             info.faiss_nprobe == values[1] && info.faiss_pq_m == values[2] &&
             info.faiss_pq_bits == values[3];
    case config_change::kDiskannBuildParams:
      return values.size() == 3 && info.diskann_max_degree == values[0] &&
             info.diskann_build_complexity == values[1] &&
             info.diskann_build_threads == values[2];
    case config_change::kDiskannSearchComplexity:
      return values.size() == 1 &&
             info.diskann_search_complexity == values[0];
    case config_change::kDiskannSearchBeamwidth:
      return values.size() == 1 && info.diskann_search_beamwidth == values[0];
    case config_change::kDiskannPqCodeBudgetSize:
      return values.size() == 1 &&
             info.diskann_pq_code_budget_size == values[0];
    case config_change::kDiskannDiskPqDims:
      return values.size() == 1 && info.diskann_disk_pq_dims == values[0];
    case config_change::kDiskannAccelerateBuild:
      return values.size() == 1 && values[0] <= 1 &&
             info.diskann_accelerate_build == (values[0] != 0);
    case config_change::kDiskannShuffleBuild:
      return values.size() == 1 && values[0] <= 1 &&
             info.diskann_shuffle_build == (values[0] != 0);
    case config_change::kDiskannUseBfsCache:
      return values.size() == 1 && values[0] <= 1 &&
             info.diskann_use_bfs_cache == (values[0] != 0);
    case config_change::kDiskannBuildMode:
      return values.size() == 1 &&
             info.diskann_build_mode_value ==
                 static_cast<vector_index::diskann_build_mode>(values[0]);
  }
  return false;
}

bool config_statement_payload_valid(
    const vector_statement_publication::operation_payload &payload) {
  using vector_statement_publication::config_change;
  const auto &values = payload.unsigned_values;
  const auto fits_uint32 = [](uint64_t value) {
    return value <= std::numeric_limits<uint32_t>::max();
  };
  const auto all_fit_uint32 = [&]() {
    return std::all_of(values.begin(), values.end(), fits_uint32);
  };

  switch (payload.config) {
    case config_change::kSearchEf:
      return values.size() == 1 && values[0] != 0 && all_fit_uint32();
    case config_change::kHnswBuildParams:
      return values.size() == 2 && values[0] != 0 && values[1] != 0 &&
             all_fit_uint32();
    case config_change::kFaissIvfParams:
      return values.size() == 2 && all_fit_uint32();
    case config_change::kFaissIvfPqParams:
      return values.size() == 4 && values[0] != 0 && values[1] != 0 &&
             values[2] != 0 && values[3] != 0 &&
             values[3] <= vector_index::k_max_faiss_pq_bits && all_fit_uint32();
    case config_change::kDiskannBuildParams:
      return values.size() == 3 && values[0] != 0 && values[1] != 0 &&
             all_fit_uint32();
    case config_change::kDiskannSearchComplexity:
    case config_change::kDiskannSearchBeamwidth:
      return values.size() == 1 && values[0] != 0 && all_fit_uint32();
    case config_change::kDiskannPqCodeBudgetSize:
      return values.size() == 1;
    case config_change::kDiskannDiskPqDims:
      return values.size() == 1 && all_fit_uint32();
    case config_change::kDiskannAccelerateBuild:
    case config_change::kDiskannShuffleBuild:
    case config_change::kDiskannUseBfsCache:
      return values.size() == 1 && values[0] <= 1;
    case config_change::kDiskannBuildMode:
      return values.size() == 1 &&
             values[0] <= static_cast<uint64_t>(
                              vector_index::diskann_build_mode::kOffline);
  }
  return false;
}

bool config_statement_supported(
    const vector_index::index_service::index_config &config,
    const vector_statement_publication::operation_payload &payload) {
  using vector_index::backend_mode;
  using vector_index::backend_provider;
  using vector_statement_publication::config_change;

  const bool supports_hnsw_tuning =
      (config.provider == backend_provider::kHnswlib &&
       config.mode == backend_mode::kMemory) ||
      (config.provider == backend_provider::kFaiss &&
       config.mode == backend_mode::kExternal);
  const bool supports_faiss_tuning =
      config.provider == backend_provider::kFaiss &&
      config.mode == backend_mode::kExternal;
  const bool supports_diskann_tuning =
      config.provider == backend_provider::kDiskAnn &&
      config.mode == backend_mode::kExternal;
  if (payload.unsigned_values.empty()) return false;

  switch (payload.config) {
    case config_change::kSearchEf:
    case config_change::kHnswBuildParams:
      return supports_hnsw_tuning;
    case config_change::kFaissIvfParams:
    case config_change::kFaissIvfPqParams:
      return supports_faiss_tuning;
    case config_change::kDiskannBuildParams:
    case config_change::kDiskannSearchComplexity:
    case config_change::kDiskannSearchBeamwidth:
    case config_change::kDiskannPqCodeBudgetSize:
      return supports_diskann_tuning;
    case config_change::kDiskannDiskPqDims:
    case config_change::kDiskannAccelerateBuild:
    case config_change::kDiskannShuffleBuild:
    case config_change::kDiskannUseBfsCache:
      return payload.unsigned_values[0] == 0 || supports_diskann_tuning;
    case config_change::kDiskannBuildMode:
      return payload.unsigned_values[0] ==
                 static_cast<uint64_t>(
                     vector_index::diskann_build_mode::kAuto) ||
             supports_diskann_tuning;
  }
  return false;
}

bool config_statement_requires_quiescent_index(
    vector_statement_publication::config_change config) {
  using vector_statement_publication::config_change;
  switch (config) {
    case config_change::kSearchEf:
    case config_change::kDiskannSearchComplexity:
    case config_change::kDiskannSearchBeamwidth:
      return false;
    case config_change::kHnswBuildParams:
    case config_change::kFaissIvfParams:
    case config_change::kFaissIvfPqParams:
    case config_change::kDiskannBuildParams:
    case config_change::kDiskannPqCodeBudgetSize:
    case config_change::kDiskannDiskPqDims:
    case config_change::kDiskannAccelerateBuild:
    case config_change::kDiskannShuffleBuild:
    case config_change::kDiskannUseBfsCache:
    case config_change::kDiskannBuildMode:
      return true;
  }
  return true;
}

bool batch_payload_valid(
    const vector_statement_publication::operation_payload &payload,
    size_t *row_count, size_t *dimension) {
  if (payload.unsigned_values.size() != 4 ||
      payload.unsigned_values[0] != 1 ||
      payload.unsigned_values[1] == 0 ||
      payload.unsigned_values[2] == 0 || payload.string_values.size() != 1) {
    return false;
  }
  const uint64_t row_count_value = payload.unsigned_values[1];
  const uint64_t dimension_value = payload.unsigned_values[2];
  if (row_count_value > std::numeric_limits<size_t>::max() ||
      dimension_value > std::numeric_limits<size_t>::max()) {
    return false;
  }
  const size_t parsed_row_count = static_cast<size_t>(row_count_value);
  const size_t parsed_dimension = static_cast<size_t>(dimension_value);
  if (parsed_row_count >
          std::numeric_limits<size_t>::max() / sizeof(uint64_t) ||
      parsed_row_count >
          std::numeric_limits<size_t>::max() / parsed_dimension ||
      parsed_row_count * parsed_dimension >
          std::numeric_limits<size_t>::max() / sizeof(float) ||
      payload.string_values[0].size() !=
          parsed_row_count * sizeof(uint64_t) ||
      payload.binary_value.size() !=
          parsed_row_count * parsed_dimension * sizeof(float)) {
    return false;
  }
  if (row_count != nullptr) *row_count = parsed_row_count;
  if (dimension != nullptr) *dimension = parsed_dimension;
  return true;
}

bool decode_batch_reader(
    vector_statement_publication::operation_payload payload,
    vector_index::index_service::bulk_load_reader *reader,
    vector_index::index_service::bulk_load_options *options) {
  size_t row_count = 0;
  size_t dimension = 0;
  if (reader == nullptr || options == nullptr ||
      !batch_payload_valid(payload, &row_count, &dimension)) {
    return false;
  }

  *reader = [docids = std::move(payload.string_values[0]),
             vectors = std::move(payload.binary_value), row_count, dimension](
                const vector_index::index_service::bulk_load_visitor &visitor,
                std::string *error) {
    vector_index::vector_data row(dimension);
    const auto *docid_bytes =
        reinterpret_cast<const uchar *>(docids.data());
    const auto *vector_bytes =
        reinterpret_cast<const uchar *>(vectors.data());
    for (size_t row_idx = 0; row_idx < row_count; ++row_idx) {
      const uint64_t doc_id =
          uint8korr(docid_bytes + row_idx * sizeof(uint64_t));
      for (size_t dim_idx = 0; dim_idx < dimension; ++dim_idx) {
        const float value = float4get(
            vector_bytes + (row_idx * dimension + dim_idx) * sizeof(float));
        if (!std::isfinite(value)) {
          if (error != nullptr) *error = "non-finite vector value";
          return false;
        }
        row[dim_idx] = value;
      }
      if (!visitor(doc_id, row.data(), row.size())) return false;
    }
    return true;
  };
  options->replace_duplicates = payload.unsigned_values[3] != 0;
  options->rebuild_after_load = false;
  options->source_format = "BINARY_BLOB";
  return true;
}

bool apply_bulk_load_statement(
    const vector_index_truth_store::publication_intent &intent,
    vector_statement_publication::operation_payload payload,
    std::string *failure_stage) {
  if (payload.unsigned_values.empty()) return false;
  if (payload.unsigned_values[0] == k_batch_load_payload) {
    vector_index::index_service::bulk_load_reader reader;
    vector_index::index_service::bulk_load_options options;
    if (!decode_batch_reader(std::move(payload), &reader, &options)) {
      set_publication_failure(failure_stage, "decode_batch_payload");
      return false;
    }
    std::string error;
    if (!bulk_upsert_from_reader(intent.index_name, reader, options, &error)) {
      set_publication_failure(
          failure_stage,
          error.empty() ? "publish_batch_payload" : error);
      return false;
    }
    return true;
  }
  const vector_index::load_staging_identity identity{intent.index_name,
                                                     intent.publication_id};
  vector_index::load_staging_artifact artifact;
  bool replace_duplicates = false;
  bool rebuild_after_load = false;
  std::string format;
  if (!vector_index::parse_load_staging_payload(
          identity, payload, &artifact, &replace_duplicates,
          &rebuild_after_load, nullptr, &format)) {
    set_publication_failure(failure_stage, "decode_load_payload");
    return false;
  }

  vector_index::load_staging_paths paths;
  std::string error;
  if (!vector_index::verify_load_artifact(identity, artifact, format, &paths,
                                          &error)) {
    set_publication_failure(
        failure_stage, error.empty() ? "verify_load_staging_artifact" : error);
    return false;
  }

  vector_index::index_service::bulk_load_options options;
  options.replace_duplicates = replace_duplicates;
  options.rebuild_after_load = rebuild_after_load;
  options.source_format = format;
  const vector_index::standalone_load_receipt receipt{
      intent.publication_id, artifact.identity, artifact.vector_checksum,
      artifact.docid_checksum};
  if (options.source_format == "FBIN") {
    uint64_t loaded_rows = 0;
    if (bulk_upsert_from_raw_files(intent.index_name, paths.vector_filename,
                                   paths.docid_filename, options, &loaded_rows,
                                   &error, &receipt)) {
      return true;
    }
  } else if (options.source_format == "CSV") {
    const vector_index::index_service::bulk_load_reader reader =
        [vector_filename = std::move(paths.vector_filename)](
            const vector_index::index_service::bulk_load_visitor &visitor,
            std::string *reader_error) {
          return vector_index::read_csv_vectors(
              vector_filename, 0, nullptr, reader_error,
              [&visitor](uint64_t doc_id, const float *values,
                         size_t dimension) {
                return visitor(doc_id, values, dimension);
              });
        };
    if (bulk_upsert_from_reader(intent.index_name, reader, options, &error,
                                &receipt)) {
      return true;
    }
  }
  set_publication_failure(failure_stage,
                          error.empty() ? "publish_load_payload" : error);
  return false;
}

bool create_statement_payload_valid(
    const vector_statement_publication::operation_payload &payload) {
  const auto &values = payload.unsigned_values;
  const auto &strings = payload.string_values;
  if (values.size() != kCreateUnsignedFieldCount ||
      strings.size() != kCreateStringFieldCount ||
      values[kCreateDimension] == 0 ||
      values[kCreateDimension] > vector_index::k_max_vector_dimension ||
      values[kCreateBuildThreads] > vector_index::k_max_build_threads ||
      values[kCreateDiskannMaxDegree] >
          std::numeric_limits<uint32_t>::max() ||
      values[kCreateDiskannBuildComplexity] >
          std::numeric_limits<uint32_t>::max() ||
      values[kCreateDiskannDiskPqDims] >
          std::numeric_limits<uint32_t>::max() ||
      values[kCreateDiskannAccelerateBuild] > 1 ||
      values[kCreateDiskannShuffleBuild] > 1 ||
      values[kCreateDiskannUseBfsCache] > 1 ||
      values[kCreateDiskannSearchComplexity] >
          std::numeric_limits<uint32_t>::max() ||
      values[kCreateDiskannSearchBeamwidth] >
          std::numeric_limits<uint32_t>::max() ||
      values[kCreateConsistencyMode] >
          static_cast<uint64_t>(
              vector_index::index_consistency_mode::kStandalone) ||
      strings[kCreateMode].empty() || strings[kCreateProvider].empty() ||
      strings[kCreateOwnerSchema].empty()) {
    return false;
  }

  vector_index::metric_type metric;
  if (!vector_index::parse_metric(strings[kCreateMetric], &metric)) {
    return false;
  }

  vector_index::backend_provider provider;
  if (!vector_index::parse_backend_provider(strings[kCreateProvider],
                                            &provider) ||
      !vector_index::backend_provider_supported(provider)) {
    return false;
  }

  vector_index::backend_mode mode;
  if (!vector_index::parse_backend_mode(strings[kCreateMode], &mode)) {
    return false;
  }
  if (!vector_index::runtime_provider_accepts_mode(provider, mode)) return false;

  const bool diskann_provider =
      provider == vector_index::backend_provider::kDiskAnn;
  if (diskann_provider) {
    return values[kCreateDiskannMaxDegree] != 0 &&
           values[kCreateDiskannBuildComplexity] != 0 &&
           values[kCreateDiskannSearchComplexity] != 0 &&
           values[kCreateDiskannSearchBeamwidth] != 0;
  }
  return values[kCreateDiskannMaxDegree] == 0 &&
         values[kCreateDiskannBuildComplexity] == 0 &&
         values[kCreateDiskannPqCodeBudgetSize] == 0 &&
         values[kCreateDiskannDiskPqDims] == 0 &&
         values[kCreateDiskannAccelerateBuild] == 0 &&
         values[kCreateDiskannShuffleBuild] == 0 &&
         values[kCreateDiskannUseBfsCache] == 0 &&
         values[kCreateDiskannSearchComplexity] == 0 &&
         values[kCreateDiskannSearchBeamwidth] == 0;
}

bool apply_create_statement(
    const vector_index_truth_store::publication_intent &intent,
    const vector_statement_publication::operation_payload &payload) {
  if (!create_statement_payload_valid(payload)) return false;

  const auto &values = payload.unsigned_values;
  const auto &strings = payload.string_values;
  create_index_options options;
  options.defaults_resolved = true;
  options.build_threads_specified = true;
  options.build_threads = static_cast<uint32_t>(values[kCreateBuildThreads]);
  options.diskann_max_degree =
      static_cast<uint32_t>(values[kCreateDiskannMaxDegree]);
  options.diskann_build_complexity =
      static_cast<uint32_t>(values[kCreateDiskannBuildComplexity]);
  options.diskann_pq_code_budget_size =
      values[kCreateDiskannPqCodeBudgetSize];
  options.diskann_disk_pq_dims =
      static_cast<uint32_t>(values[kCreateDiskannDiskPqDims]);
  options.diskann_accelerate_build_specified = true;
  options.diskann_accelerate_build =
      values[kCreateDiskannAccelerateBuild] != 0;
  options.diskann_shuffle_build_specified = true;
  options.diskann_shuffle_build = values[kCreateDiskannShuffleBuild] != 0;
  options.diskann_use_bfs_cache_specified = true;
  options.diskann_use_bfs_cache = values[kCreateDiskannUseBfsCache] != 0;
  options.diskann_search_complexity =
      static_cast<uint32_t>(values[kCreateDiskannSearchComplexity]);
  options.diskann_search_beamwidth =
      static_cast<uint32_t>(values[kCreateDiskannSearchBeamwidth]);
  options.consistency_mode_specified = true;
  options.consistency_mode =
      static_cast<vector_index::index_consistency_mode>(
          values[kCreateConsistencyMode]);
  options.initial_lifecycle_state = strings[kCreateLifecycleState];
  return create_index(
      intent.index_name, static_cast<size_t>(values[kCreateDimension]),
      strings[kCreateMetric], strings[kCreateMode], strings[kCreateProvider],
      strings[kCreateOwnerSchema], options);
}

bool apply_upsert_statement(
    const vector_index_truth_store::publication_intent &intent,
    const vector_statement_publication::operation_payload &payload) {
  if (payload.unsigned_values.size() != 2 ||
      payload.unsigned_values[1] == 0 ||
      payload.unsigned_values[1] > std::numeric_limits<size_t>::max()) {
    return false;
  }
  const size_t dimension =
      static_cast<size_t>(payload.unsigned_values[1]);
  if (dimension > std::numeric_limits<size_t>::max() / sizeof(float) ||
      payload.binary_value.size() != dimension * sizeof(float)) {
    return false;
  }
  const auto *bytes =
      reinterpret_cast<const uchar *>(payload.binary_value.data());
  vector_index::vector_data vector(dimension);
  for (size_t i = 0; i < dimension; ++i) {
    vector[i] = float4get(bytes + i * sizeof(float));
    if (!std::isfinite(vector[i])) return false;
  }
  return upsert(intent.index_name, payload.unsigned_values[0], vector);
}

bool validate_statement_operation_locked(
    const vector_index_truth_store::publication_intent &intent) {
  using vector_index_truth_store::publication_operation;
  vector_statement_publication::operation_payload payload;
  if (statement_operation_has_payload(intent.operation) &&
      !decode_statement_payload(intent, &payload, nullptr)) {
    return false;
  }

  if (intent.operation == publication_operation::kTransactionalDml) {
    return false;
  }
  if (intent.operation == publication_operation::kCreateIndex) {
    return !intent.expected.exists && create_statement_payload_valid(payload);
  }
  if (!intent.expected.exists) return false;

  vector_index::index_service::index_config config;
  if (!g_index_service.describe_index(intent.index_name, &config, nullptr,
                                      nullptr, nullptr)) {
    return false;
  }
  const bool has_pending_changes =
      g_index_service.has_pending_changes_for_index(intent.index_name);
  switch (intent.operation) {
    case publication_operation::kStandaloneUpsert: {
      if (config.consistency_mode !=
              vector_index::index_consistency_mode::kStandalone ||
          has_pending_changes || payload.unsigned_values.size() != 2 ||
          payload.unsigned_values[1] != config.dimension ||
          config.dimension >
              std::numeric_limits<size_t>::max() / sizeof(float) ||
          payload.binary_value.size() != config.dimension * sizeof(float)) {
        return false;
      }
      const auto *bytes =
          reinterpret_cast<const uchar *>(payload.binary_value.data());
      for (size_t i = 0; i < config.dimension; ++i) {
        if (!std::isfinite(float4get(bytes + i * sizeof(float)))) return false;
      }
      return true;
    }
    case publication_operation::kStandaloneErase:
      return config.consistency_mode ==
                 vector_index::index_consistency_mode::kStandalone &&
             !has_pending_changes && payload.unsigned_values.size() == 1;
    case publication_operation::kBulkLoad:
      if (config.consistency_mode !=
              vector_index::index_consistency_mode::kStandalone ||
          has_pending_changes || payload.unsigned_values.empty()) {
        return false;
      }
      if (payload.unsigned_values[0] == k_batch_load_payload) {
        size_t dimension = 0;
        return batch_payload_valid(payload, nullptr, &dimension) &&
               dimension == config.dimension;
      }
      {
        size_t dimension = 0;
        return vector_index::parse_load_staging_payload(
                   {intent.index_name, intent.publication_id}, payload, nullptr,
                   nullptr, nullptr, &dimension, nullptr) &&
               dimension == config.dimension;
      }
    case publication_operation::kUpdateConfig:
      if (!config_statement_payload_valid(payload) ||
          !config_statement_supported(config, payload) ||
          (config_statement_requires_quiescent_index(payload.config) &&
           has_pending_changes)) {
        return false;
      }
      if (payload.config ==
          vector_statement_publication::config_change::kFaissIvfPqParams) {
        const auto &values = payload.unsigned_values;
        return vector_index::valid_faiss_ivf_pq_config(
            config.dimension, static_cast<uint32_t>(values[0]),
            static_cast<uint32_t>(values[1]), static_cast<uint32_t>(values[2]),
            static_cast<uint32_t>(values[3]));
      }
      return true;
    case publication_operation::kRebuildIndex:
    case publication_operation::kRecoverIndex:
    case publication_operation::kBeginBulkLoad:
    case publication_operation::kBulkBuildIndex:
      return intent.payload.empty() && !has_pending_changes;
    case publication_operation::kDropIndex:
      return intent.payload.empty();
    case publication_operation::kTransactionalDml:
    case publication_operation::kCreateIndex:
      break;
  }
  return false;
}

bool apply_statement_operation(
    const vector_index_truth_store::publication_intent &intent,
    std::string *failure_stage) {
  using vector_index_truth_store::publication_operation;
  vector_statement_publication::operation_payload payload;
  if (statement_operation_has_payload(intent.operation) &&
      !decode_statement_payload(intent, &payload, failure_stage)) {
    return false;
  }

  bool ok = false;
  switch (intent.operation) {
    case publication_operation::kTransactionalDml:
      set_publication_failure(failure_stage,
                              "invalid_statement_transactional_intent");
      return false;
    case publication_operation::kStandaloneUpsert:
      ok = apply_upsert_statement(intent, payload);
      break;
    case publication_operation::kStandaloneErase:
      ok = payload.unsigned_values.size() == 1 &&
           erase(intent.index_name, payload.unsigned_values[0]);
      break;
    case publication_operation::kBulkLoad:
      return apply_bulk_load_statement(intent, std::move(payload),
                                       failure_stage);
    case publication_operation::kCreateIndex:
      ok = apply_create_statement(intent, payload);
      break;
    case publication_operation::kDropIndex:
      ok = drop_index(intent.index_name);
      break;
    case publication_operation::kUpdateConfig:
      ok = apply_config_statement(intent, payload);
      break;
    case publication_operation::kRebuildIndex:
      ok = rebuild_index(intent.index_name, failure_stage);
      break;
    case publication_operation::kRecoverIndex:
      ok = recover_index(intent.index_name);
      break;
    case publication_operation::kBeginBulkLoad:
      ok = begin_bulk_load(intent.index_name);
      break;
    case publication_operation::kBulkBuildIndex:
      ok = bulk_build_index(intent.index_name, failure_stage);
      break;
  }
  if (!ok && failure_stage != nullptr && failure_stage->empty()) {
    *failure_stage = "publish_statement_operation";
  }
  return ok;
}

bool statement_operation_applied(
    const vector_index_truth_store::publication_intent &intent,
    const vector_index_truth_store::publication_token &current,
    std::string *failure_stage) {
  using vector_index_truth_store::publication_operation;
  if (intent.operation == publication_operation::kCreateIndex) {
    return !intent.expected.exists && current.exists;
  }
  if (intent.operation == publication_operation::kDropIndex) {
    return intent.expected.exists && !current.exists;
  }
  if (!intent.expected.exists || !current.exists ||
      intent.expected.index_identity != current.index_identity) {
    return false;
  }

  vector_statement_publication::operation_payload payload;
  switch (intent.operation) {
    case publication_operation::kUpdateConfig:
      return decode_statement_payload(intent, &payload, failure_stage) &&
             config_statement_applied(intent, payload);
    case publication_operation::kStandaloneUpsert:
    case publication_operation::kStandaloneErase:
      return token_advanced_from(intent.expected, current);
    case publication_operation::kBulkLoad: {
      if (!decode_statement_payload(intent, &payload, failure_stage) ||
          payload.unsigned_values.empty()) {
        return false;
      }
      if (payload.unsigned_values[0] == k_batch_load_payload) {
        return token_advanced_from(intent.expected, current);
      }
      vector_index::load_staging_artifact artifact;
      if (!vector_index::parse_load_staging_payload(
              {intent.index_name, intent.publication_id}, payload, &artifact,
              nullptr, nullptr, nullptr, nullptr)) {
        return false;
      }
      const vector_index::standalone_load_receipt receipt{
          intent.publication_id, artifact.identity, artifact.vector_checksum,
          artifact.docid_checksum};
      std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
      return ensure_metadata_loaded_locked() &&
             g_index_service.standalone_load_receipt_matches(intent.index_name,
                                                             receipt);
    }
    case publication_operation::kBeginBulkLoad: {
      index_info info;
      return get_index_info(intent.index_name, &info) &&
             info.lifecycle_state == "bulk_loading";
    }
    case publication_operation::kRebuildIndex:
    case publication_operation::kBulkBuildIndex:
    case publication_operation::kRecoverIndex: {
      index_info info;
      return get_index_info(intent.index_name, &info) &&
             info.lifecycle_state == "ready" &&
             info.runtime_generation == info.truth_generation;
    }
    case publication_operation::kTransactionalDml:
    case publication_operation::kCreateIndex:
    case publication_operation::kDropIndex:
      break;
  }
  return false;
}

bool rollback_runtime_and_pending_state_locked(
    uint64_t txn_id, const commit_runtime_snapshot &runtime_snapshot,
    const vector_index::index_service::pending_state_snapshot
        &pending_snapshot) {
  const bool runtime_restored =
      rollback_runtime_commit_state_locked(runtime_snapshot);
  const bool pending_restored =
      g_index_service.restore_pending_state(txn_id, pending_snapshot);
  if (!runtime_restored) {
    fail_stop_registry_locked("runtime_commit_state_restore_failed");
  } else if (!pending_restored) {
    vector_status::record_runtime_state_rollback_failure();
    fail_stop_registry_locked("pending_state_restore_failed");
  }
  return runtime_restored && pending_restored;
}

bool rollback_runtime_commit_state_or_fail_locked(
    const commit_runtime_snapshot &snapshot) {
  if (rollback_runtime_commit_state_locked(snapshot)) return true;
  fail_stop_registry_locked("runtime_commit_state_restore_failed");
  return false;
}

bool publish_pending_runtime(
    uint64_t txn_id,
    std::vector<vector_index_metadata_store::change_log_row> durable_rows,
    bool update_pending_status, std::string *failure_stage = nullptr) {
  vector_index::index_service::commit_build_plan build_plan;
  size_t pending_change_count = 0;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) {
      set_publication_failure(failure_stage, "metadata_load_before_snapshot");
      return false;
    }
    pending_change_count = g_index_service.pending_change_count(txn_id);
    if (pending_change_count == 0) return durable_rows.empty();
    if (pending_change_count != durable_rows.size()) {
      set_publication_failure(failure_stage, "pending_change_count_mismatch");
      return false;
    }
    if (!g_index_service.snapshot_commit_build_plan(txn_id, &build_plan)) {
      set_publication_failure(failure_stage, "snapshot_commit_build_plan");
      return false;
    }
    if (!bind_commit_truth_generations(&durable_rows, &build_plan)) {
      set_publication_failure(failure_stage, "bind_commit_truth_generations");
      return false;
    }
  }

  const auto commit_started = std::chrono::steady_clock::now();
  if (!g_index_service.build_commit_backends(&build_plan)) {
    set_publication_failure(failure_stage, "build_commit_backends");
    return false;
  }

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) {
    set_publication_failure(failure_stage, "metadata_load_before_apply");
    return false;
  }
  if (g_index_service.pending_change_count(txn_id) != pending_change_count) {
    set_publication_failure(failure_stage, "pending_change_count_changed");
    return false;
  }

  commit_runtime_snapshot before_state;
  vector_index::index_service::pending_state_snapshot before_pending_state;
  if (!capture_runtime_commit_state_locked(&before_state)) {
    set_publication_failure(failure_stage, "capture_runtime_commit_state");
    return false;
  }
  if (!g_index_service.snapshot_pending_state(txn_id, &before_pending_state)) {
    set_publication_failure(failure_stage, "snapshot_pending_state");
    return false;
  }
  if (!g_index_service.apply_commit_build_plan(txn_id, &build_plan)) {
    set_publication_failure(
        failure_stage,
        build_plan.failure_stage.empty()
            ? "apply_commit_build_plan"
            : "apply_commit_build_plan:" + build_plan.failure_stage);
    vector_status::record_txn_commit_failure();
    if (!rollback_runtime_and_pending_state_locked(
            txn_id, before_state, before_pending_state) &&
        failure_stage != nullptr) {
      *failure_stage += ":rollback_incomplete";
    }
    return false;
  }
  if (!append_durable_change_log_rows_locked(durable_rows)) {
    set_publication_failure(failure_stage, "append_durable_change_log_rows");
    vector_status::record_txn_commit_failure();
    if (!rollback_runtime_and_pending_state_locked(
            txn_id, before_state, before_pending_state) &&
        failure_stage != nullptr) {
      *failure_stage += ":rollback_incomplete";
    }
    return false;
  }

  if (update_pending_status) {
    vector_status::subtract_pending_txn_changes(pending_change_count);
  }
  g_manifest_committed_checkpoint =
      g_index_service.committed_entry_count();
  refresh_committed_snapshot_rows_locked();
  refresh_manifest_status_locked();
  if (!evict_committed_cache_to_budget_locked()) {
    set_publication_failure(failure_stage, "evict_committed_cache_to_budget");
    if (update_pending_status) {
      vector_status::add_pending_txn_changes(pending_change_count);
    }
    vector_status::record_txn_commit_failure();
    if (!rollback_runtime_and_pending_state_locked(
            txn_id, before_state, before_pending_state) &&
        failure_stage != nullptr) {
      *failure_stage += ":rollback_incomplete";
    }
    return false;
  }

  uint64_t apply_latency_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - commit_started)
          .count());
  if (apply_latency_ms == 0) apply_latency_ms = 1;
  vector_status::set_apply_latency_ms(apply_latency_ms);
  for (const auto &index_plan : build_plan.indexes) {
    (void)g_index_service.set_last_apply_latency_ms(index_plan.index_name,
                                                    apply_latency_ms);
  }
  return true;
}

bool publish_unapplied_truth_rows_impl(bool acquire_publication_guard) {
  std::vector<vector_index_metadata_store::change_log_row> persisted_rows;
  if (!vector_index_truth_store::get()->load_change_log(&persisted_rows)) {
    vector_status::record_change_log_load_failure();
    return false;
  }

  std::vector<std::string> persisted_index_names;
  persisted_index_names.reserve(persisted_rows.size());
  for (const auto &row : persisted_rows) {
    persisted_index_names.push_back(row.index_name);
  }
  vector_statement_publication::publication_guard publication_guard;
  if (acquire_publication_guard && !persisted_index_names.empty() &&
      !publication_guard.lock_indexes(persisted_index_names)) {
    return false;
  }

  std::vector<vector_index_metadata_store::change_log_row> unapplied_rows;
  uint64_t replay_txn_id = 0;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;

    std::unordered_set<uint64_t> applied_sequences;
    applied_sequences.reserve(g_change_log_rows.size());
    for (const auto &row : g_change_log_rows) {
      applied_sequences.insert(row.sequence);
    }
    for (const auto &row : persisted_rows) {
      if (applied_sequences.find(row.sequence) == applied_sequences.end()) {
        unapplied_rows.push_back(row);
      }
    }
    if (unapplied_rows.empty()) return true;

    std::sort(unapplied_rows.begin(), unapplied_rows.end(),
              [](const auto &lhs, const auto &rhs) {
                if (lhs.index_name != rhs.index_name)
                  return lhs.index_name < rhs.index_name;
                if (lhs.index_identity != rhs.index_identity)
                  return lhs.index_identity < rhs.index_identity;
                if (lhs.truth_generation != rhs.truth_generation)
                  return lhs.truth_generation < rhs.truth_generation;
                return lhs.sequence < rhs.sequence;
              });
    replay_txn_id = allocate_txn_id();
    if (replay_txn_id == 0) return false;

    std::vector<vector_index::index_service::pending_change_snapshot> changes;
    changes.reserve(unapplied_rows.size());
    for (const auto &row : unapplied_rows) {
      changes.push_back({
          row.index_name,
          row.op == vector_index_metadata_store::change_op::kErase,
          row.doc_id, row.vector});
    }
    if (!g_index_service.restore_pending_changes(replay_txn_id, changes)) {
      g_index_service.rollback(replay_txn_id);
      return false;
    }
  }

  if (publish_pending_runtime(replay_txn_id, unapplied_rows, false)) {
    return true;
  }
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  g_index_service.rollback(replay_txn_id);
  return false;
}

bool acknowledge_publication_intents(
    const std::vector<vector_index_truth_store::publication_intent> &intents,
    std::string *failure_stage) {
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) {
      set_publication_failure(failure_stage,
                              "metadata_load_before_intent_ack");
      return false;
    }
    for (const auto &intent : intents) {
      vector_index_truth_store::publication_token current;
      if (!describe_publication_token_locked(intent.index_name, &current) ||
          !publication_intent_target_matches(intent, current)) {
        set_publication_failure(
            failure_stage,
            std::string("publication_intent_target_mismatch:") +
                publication_token_mismatch_field(intent.target, current));
        return false;
      }
    }
  }

  auto *truth_store = vector_index_truth_store::get();
  for (const auto &intent : intents) {
    if (!truth_store->delete_committed_publication_intent(
            intent.index_name, intent.publication_id)) {
      set_publication_failure(failure_stage,
                              "delete_publication_intent");
      return false;
    }
  }
  return true;
}

bool recover_durable_publication_intents(std::string *failure_stage) {
  auto *truth_store = vector_index_truth_store::get();
  if (!truth_store->supports_publication_intents()) return true;

  std::vector<vector_index_truth_store::publication_intent> intents;
  if (!truth_store->load_publication_intents(&intents)) {
    set_publication_failure(failure_stage, "load_publication_intents");
    return false;
  }
  if (!vector_index::cleanup_orphaned_load_staging_artifacts(intents)) {
    set_publication_failure(failure_stage,
                            "cleanup_orphaned_load_staging_artifacts");
    return false;
  }
  if (intents.empty()) return true;

  std::vector<std::string> index_names;
  index_names.reserve(intents.size());
  bool catalog_exclusive = false;
  for (const auto &intent : intents) {
    index_names.push_back(intent.index_name);
    catalog_exclusive =
        catalog_exclusive ||
        intent.operation ==
            vector_index_truth_store::publication_operation::kCreateIndex ||
        intent.operation ==
            vector_index_truth_store::publication_operation::kDropIndex;
  }
  vector_statement_publication::publication_guard publication_guard;
  const bool locked = catalog_exclusive
                          ? publication_guard.lock_catalog()
                          : publication_guard.lock_indexes(index_names);
  if (!locked) {
    set_publication_failure(failure_stage, "lock_publication_intents");
    return false;
  }

  bool replay_transactional_dml = false;
  std::vector<vector_index_truth_store::publication_intent>
      transactional_intents;
  std::vector<vector_index_truth_store::publication_intent> statement_intents;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) {
      set_publication_failure(failure_stage,
                              "metadata_load_before_intent_recovery");
      return false;
    }
    for (const auto &intent : intents) {
      if (intent.operation != vector_index_truth_store::publication_operation::
                                  kTransactionalDml) {
        statement_intents.push_back(intent);
        continue;
      }
      transactional_intents.push_back(intent);
      vector_index_truth_store::publication_token current;
      if (!describe_publication_token_locked(intent.index_name, &current)) {
        set_publication_failure(failure_stage,
                                "describe_publication_intent_target");
        return false;
      }
      if (publication_intent_target_matches(intent, current)) continue;
      if (!publication_tokens_equal(current, intent.expected)) {
        set_publication_failure(
            failure_stage,
            std::string("publication_intent_cas_mismatch:") +
                publication_token_mismatch_field(intent.expected, current));
        return false;
      }
      replay_transactional_dml = true;
    }
  }

  if (replay_transactional_dml &&
      !publish_unapplied_truth_rows_impl(false)) {
    set_publication_failure(failure_stage,
                            "replay_publication_intent_changelog");
    return false;
  }
  if (!transactional_intents.empty() &&
      !acknowledge_publication_intents(transactional_intents,
                                       failure_stage)) {
    return false;
  }
  for (const auto &intent : statement_intents) {
    if (!publish_statement_publication_intent(intent, failure_stage) ||
        !acknowledge_statement_publication_intent(intent, failure_stage)) {
      return false;
    }
  }
  return true;
}

bool recover_publication_intents_if_requested(std::string *failure_stage) {
  if (g_publication_recovery_active || g_publication_read_scope_depth != 0) {
    return true;
  }
  uint64_t requested =
      g_publication_recovery_requests.load(std::memory_order_acquire);
  if (g_publication_recovery_completed.load(std::memory_order_acquire) >=
      requested) {
    return true;
  }

  std::lock_guard<std::mutex> recovery_guard(g_publication_recovery_mutex);
  requested = g_publication_recovery_requests.load(std::memory_order_acquire);
  if (g_publication_recovery_completed.load(std::memory_order_acquire) >=
      requested) {
    return true;
  }

  publication_recovery_scope recovery_scope;
  if (!recover_durable_publication_intents(failure_stage)) return false;
  g_publication_recovery_completed.store(requested,
                                         std::memory_order_release);
  return true;
}

bool recover_publication_intents_if_requested() {
  std::string failure_stage;
  return recover_publication_intents_if_requested(&failure_stage);
}

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
  bool metadata_ready = false;
  {
    std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
    metadata_ready =
        g_metadata_loaded &&
        g_registry_health == registry_health_state::kReady;
  }

  if (!metadata_ready) {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    metadata_ready = ensure_metadata_loaded_locked();
  }
  return metadata_ready && recover_publication_intents_if_requested();
}

bool ensure_runtime_loaded_for_search(const std::string &index_name) {
  {
    std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
    if (g_metadata_loaded &&
        g_registry_health == registry_health_state::kReady &&
        g_index_service.runtime_loaded_for_search(index_name)) {
      return true;
    }
  }

  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;
    if (g_index_service.runtime_loaded_for_search(index_name)) return true;

    vector_index::index_service::index_config config;
    vector_index::index_service::index_publication_state publication;
    size_t authoritative_entry_count = 0;
    std::string lifecycle_state;
    if (!g_index_service.describe_index(
            index_name, &config, nullptr, nullptr, &authoritative_entry_count,
            &lifecycle_state) ||
        !g_index_service.describe_publication_state(index_name, &publication)) {
      return false;
    }
    if (lifecycle_state == "bulk_loading") return false;
    const bool recoverable_diskann_artifact =
        config.provider == vector_index::backend_provider::kDiskAnn &&
        config.mode == vector_index::backend_mode::kExternal &&
        publication.artifact_generation != 0 &&
        publication.artifact_generation == publication.truth_generation;
    if (recoverable_diskann_artifact) {
      vector_index::diskann_artifact_identity identity;
      if (make_diskann_artifact_identity_locked(index_name, config, publication,
                                                authoritative_entry_count,
                                                &identity) &&
          g_index_service.recover_artifact_publication(index_name, identity) &&
          g_index_service.artifact_publication_matches(index_name, identity)) {
        return true;
      }
    }
  }

  if (!detail::load_runtime_for_search(index_name)) return false;
  std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
  return g_registry_health == registry_health_state::kReady &&
         g_index_service.runtime_loaded_for_search(index_name);
}

bool rebuild_runtime_after_search_failure(const std::string &index_name) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  return ensure_metadata_loaded_locked() &&
         g_index_service.rebuild_runtime_from_store_for_search(index_name);
}

// Continuous configuration publication can invalidate every bounded optimistic
// attempt. Hold the registry read lock only on that cold path so one request can
// make progress without mixing runtime and configuration generations.
bool search_committed_runtime_serialized(
    const std::string &index_name, const vector_index::vector_data &query,
    size_t top_k, std::vector<vector_index::search_result> *results) {
  std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
  vector_index::index_service::search_runtime_snapshot snapshot;
  if (g_registry_health != registry_health_state::kReady ||
      !g_index_service.snapshot_search_runtime_loaded(
          index_name, query.size(), top_k, 1, &snapshot)) {
    return false;
  }

  std::vector<vector_index::search_result> candidates;
  {
    std::shared_lock<std::shared_mutex> runtime_guard(
        snapshot.runtime->runtime_mutex());
    if (!snapshot.runtime->search_for_rerank(
            query, top_k, snapshot.candidate_top_k, &candidates)) {
      return false;
    }
  }
  return g_index_service.search_runtime_snapshot_matches(index_name, snapshot) &&
         g_index_service.finish_search_with_exact_rerank(
             index_name, snapshot.config, query, top_k, std::move(candidates),
             results);
}

bool search_committed_runtime_batch_serialized(
    const std::string &index_name,
    const std::vector<vector_index::vector_data> &queries, size_t top_k,
    std::vector<std::vector<vector_index::search_result>> *results) {
  std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
  vector_index::index_service::search_runtime_snapshot snapshot;
  if (g_registry_health != registry_health_state::kReady ||
      !g_index_service.snapshot_search_runtime_loaded(
          index_name, queries[0].size(), top_k, queries.size(), &snapshot)) {
    return false;
  }
  for (const vector_index::vector_data &query : queries) {
    if (query.size() != snapshot.config.dimension) return false;
  }

  vector_index::batch_search_candidates candidates;
  {
    std::shared_lock<std::shared_mutex> runtime_guard(
        snapshot.runtime->runtime_mutex());
    if (!snapshot.runtime->search_batch_for_rerank(
            queries, top_k, snapshot.candidate_top_k, &candidates)) {
      return false;
    }
  }
  return g_index_service.search_runtime_snapshot_matches(index_name, snapshot) &&
         g_index_service.finish_search_batch_with_exact_rerank(
             index_name, snapshot.config, queries, top_k,
             std::move(candidates), results);
}

bool search_committed_runtime_loaded(
    const std::string &index_name, const vector_index::vector_data &query,
    size_t top_k, std::vector<vector_index::search_result> *results) {
  if (results == nullptr) return false;

  bool rebuilt_after_failure = false;
  for (size_t attempt = 0; attempt < 3; ++attempt) {
    if (!ensure_runtime_loaded_for_search(index_name)) return false;

    vector_index::index_service::search_runtime_snapshot snapshot;
    {
      std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
      if (!g_index_service.snapshot_search_runtime_loaded(
              index_name, query.size(), top_k, 1, &snapshot)) {
        continue;
      }
    }

    std::vector<vector_index::search_result> candidates;
    bool searched = false;
    {
      std::shared_lock<std::shared_mutex> runtime_guard(
          snapshot.runtime->runtime_mutex());
      searched = snapshot.runtime->search_for_rerank(
          query, top_k, snapshot.candidate_top_k, &candidates);
    }
    if (!searched) {
      if (rebuilt_after_failure ||
          !vector_index::detail::can_rebuild_after_search_failure(
              snapshot.config)) {
        return false;
      }
      if (!rebuild_runtime_after_search_failure(index_name)) return false;
      rebuilt_after_failure = true;
      continue;
    }

    bool snapshot_current = false;
    bool finished = false;
    {
      std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
      snapshot_current = g_index_service.search_runtime_snapshot_matches(
          index_name, snapshot);
      if (snapshot_current) {
        finished = g_index_service.finish_search_with_exact_rerank(
            index_name, snapshot.config, query, top_k, std::move(candidates),
            results);
      }
    }
    if (!snapshot_current) continue;
    if (finished) return true;
    if (rebuilt_after_failure ||
        !vector_index::detail::can_rebuild_after_search_failure(
            snapshot.config)) {
      return false;
    }
    if (!rebuild_runtime_after_search_failure(index_name)) return false;
    rebuilt_after_failure = true;
  }
  if (!ensure_runtime_loaded_for_search(index_name)) return false;
  return search_committed_runtime_serialized(index_name, query, top_k, results);
}

bool search_committed_runtime_batch_loaded(
    const std::string &index_name,
    const std::vector<vector_index::vector_data> &queries, size_t top_k,
    std::vector<std::vector<vector_index::search_result>> *results) {
  if (results == nullptr || queries.empty()) return false;

  bool rebuilt_after_failure = false;
  for (size_t attempt = 0; attempt < 3; ++attempt) {
    if (!ensure_runtime_loaded_for_search(index_name)) return false;

    vector_index::index_service::search_runtime_snapshot snapshot;
    {
      std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
      if (!g_index_service.snapshot_search_runtime_loaded(
              index_name, queries[0].size(), top_k, queries.size(),
              &snapshot)) {
        continue;
      }
      for (const vector_index::vector_data &query : queries) {
        if (query.size() != snapshot.config.dimension) return false;
      }
    }

    vector_index::batch_search_candidates candidates;
    bool searched = false;
    {
      std::shared_lock<std::shared_mutex> runtime_guard(
          snapshot.runtime->runtime_mutex());
      searched = snapshot.runtime->search_batch_for_rerank(
          queries, top_k, snapshot.candidate_top_k, &candidates);
    }
    if (!searched) {
      if (rebuilt_after_failure ||
          !vector_index::detail::can_rebuild_after_search_failure(
              snapshot.config)) {
        return false;
      }
      if (!rebuild_runtime_after_search_failure(index_name)) return false;
      rebuilt_after_failure = true;
      continue;
    }

    bool snapshot_current = false;
    bool finished = false;
    {
      std::shared_lock<std::shared_mutex> guard(g_registry_mutex);
      snapshot_current = g_index_service.search_runtime_snapshot_matches(
          index_name, snapshot);
      if (snapshot_current) {
        finished = g_index_service.finish_search_batch_with_exact_rerank(
            index_name, snapshot.config, queries, top_k,
            std::move(candidates), results);
      }
    }
    if (!snapshot_current) continue;
    if (finished) return true;
    if (rebuilt_after_failure ||
        !vector_index::detail::can_rebuild_after_search_failure(
            snapshot.config)) {
      return false;
    }
    if (!rebuild_runtime_after_search_failure(index_name)) return false;
    rebuilt_after_failure = true;
  }
  if (!ensure_runtime_loaded_for_search(index_name)) return false;
  return search_committed_runtime_batch_serialized(index_name, queries, top_k,
                                                    results);
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

publication_read_guard::~publication_read_guard() {
  if (!m_read_scope_active) return;
  m_guard = vector_statement_publication::read_guard();
  assert(g_publication_read_scope_depth != 0);
  --g_publication_read_scope_depth;
  m_read_scope_active = false;
}

bool publication_read_guard::lock_index(const std::string &index_name) {
  if (owns_lock() || index_name.empty()) return false;
  for (;;) {
    if (!ensure_publication_intents_recovered() ||
        !m_guard.lock_index(index_name)) {
      return false;
    }
    if (!publication_intent_recovery_pending()) break;
    m_guard = vector_statement_publication::read_guard();
  }
  ++g_publication_read_scope_depth;
  m_read_scope_active = true;
  return true;
}

bool publication_read_guard::lock_catalog() {
  if (owns_lock()) return false;
  for (;;) {
    if (!ensure_publication_intents_recovered() || !m_guard.lock_catalog()) {
      return false;
    }
    if (!publication_intent_recovery_pending()) break;
    m_guard = vector_statement_publication::read_guard();
  }
  ++g_publication_read_scope_depth;
  m_read_scope_active = true;
  return true;
}

bool publication_read_guard::owns_lock() const {
  return m_read_scope_active && m_guard.owns_lock();
}

bool ensure_publication_intents_recovered() {
  return recover_publication_intents_if_requested();
}

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

class staged_commit_delta_persist {
 public:
  staged_commit_delta_persist() = default;
  staged_commit_delta_persist(const staged_commit_delta_persist &) = delete;
  staged_commit_delta_persist &operator=(
      const staged_commit_delta_persist &) = delete;

  ~staged_commit_delta_persist() { rollback(); }

  bool stage(
      const std::vector<vector_index_metadata_store::change_log_row> &rows,
      bool compact_change_log,
      const std::vector<uint64_t> &compacted_sequences) {
    if (m_active) return false;
    vector_index_truth_store::truth_store *truth_store =
        vector_index_truth_store::get();
    if (!truth_store->is_transactional() ||
        !truth_store->supports_delta_persist()) {
      return false;
    }

    vector_status::record_truth_store_persist_request();
    vector_status::record_truth_store_delta_persist_request();
    if (!truth_store->begin_persist()) {
      vector_status::record_truth_store_persist_failure();
      vector_status::record_truth_store_delta_persist_failure();
      return false;
    }
    m_active = true;
    m_compaction_staged = compact_change_log;
    if (compact_change_log)
      vector_status::record_truth_store_compact_request();
    const bool committed_ok = truth_store->apply_committed_delta(rows);
    const bool changelog_ok =
        committed_ok &&
        (compact_change_log
             ? truth_store->erase_change_log_sequences(compacted_sequences)
             : truth_store->append_change_log_delta(rows));
    if (!committed_ok || !changelog_ok) {
      vector_status::record_truth_store_persist_failure();
      vector_status::record_truth_store_delta_persist_failure();
      rollback();
      return false;
    }
    return true;
  }

  bool finish_locked() {
    if (!m_active) return false;
    const bool ok = persist_staged_commit_artifacts_locked();
    if (!ok && m_compaction_staged)
      vector_status::record_truth_store_compact_failure();
    // The finalizer commits on success and rolls back on every failure path.
    m_active = false;
    return ok;
  }

  /** Discard staged truth rows before waiting outside the row-lock domain. */
  void cancel() { rollback(); }

 private:
  void rollback() {
    if (!m_active) return;
    vector_index_truth_store::get()->rollback_persist();
    vector_status::record_persist_artifact_rollback();
    if (m_compaction_staged)
      vector_status::record_truth_store_compact_failure();
    m_active = false;
  }

  bool m_active{false};
  bool m_compaction_staged{false};
};

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
  std::vector<vector_index_metadata_store::change_log_row> commit_delta_rows;
  std::vector<uint64_t> compacted_change_log_sequences;
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  const bool stage_delta_before_publication =
      truth_store->is_transactional() && truth_store->supports_delta_persist();
  bool compact_staged_change_log = false;
  staged_commit_delta_persist staged_delta;
  const auto commit_started = std::chrono::steady_clock::now();
  auto index_commit_start =
      vector_index_diagnostics::now_if(diagnostics_enabled);
  vector_statement_publication::publication_guard publication_guard;
  bool retrying_after_publication_contention = false;
  for (;;) {
    pending_change_count = 0;
    pending_index_names.clear();
    build_plan = vector_index::index_service::commit_build_plan{};
    pending_delta.clear();
    commit_delta_rows.clear();
    compacted_change_log_sequences.clear();
    compact_staged_change_log = false;
    pending_index_names_ms = 0;
    snapshot_before_ms = 0;
    index_commit_ms = 0;
    pending_delta_ms = 0;
    changelog_delta_ms = 0;
    persist_ms = 0;
    index_commit_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);

    {
      std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
      if (!ensure_metadata_loaded_locked()) return false;
      pending_change_count = g_index_service.pending_change_count(txn_id);
      if (pending_change_count == 0) {
        if (!retrying_after_publication_contention) return true;
        vector_status::record_txn_commit_failure();
        return false;
      }

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

      if (stage_delta_before_publication) {
        const auto pending_delta_start =
            vector_index_diagnostics::now_if(diagnostics_enabled);
        if (!g_index_service.snapshot_commit_build_plan(txn_id, &build_plan) ||
            !g_index_service.snapshot_pending_change_delta(txn_id,
                                                           &pending_delta)) {
          record_commit_diagnostics(
              scope, txn_id, pending_change_count, false,
              pending_index_names_ms, 0, 0,
              vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                      pending_delta_start),
              0, 0,
              vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                      total_start));
          return false;
        }
        pending_delta_ms = vector_index_diagnostics::elapsed_ms_if(
            diagnostics_enabled, pending_delta_start);
      }
    }

    if (!stage_delta_before_publication) {
      if (!publication_guard.lock_indexes(pending_index_names)) return false;
      std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
      if (!ensure_metadata_loaded_locked() ||
          g_index_service.pending_change_count(txn_id) !=
              pending_change_count) {
        return false;
      }
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

    if (!stage_delta_before_publication) break;

    {
      std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
      if (!ensure_metadata_loaded_locked() ||
          g_index_service.pending_change_count(txn_id) !=
              pending_change_count ||
          !allocate_pending_change_log_delta_locked(txn_id, pending_delta,
                                                    &commit_delta_rows) ||
          !bind_commit_truth_generations(&commit_delta_rows, &build_plan)) {
        vector_status::record_txn_commit_failure();
        return false;
      }
      compact_staged_change_log =
          change_log_compaction_needed_with_delta_locked(
              commit_delta_rows.size());
      if (compact_staged_change_log &&
          !snapshot_durable_change_log_sequences_locked(
              &compacted_change_log_sequences)) {
        vector_status::record_txn_commit_failure();
        return false;
      }
    }
    if (!staged_delta.stage(commit_delta_rows, compact_staged_change_log,
                            compacted_change_log_sequences)) {
      vector_status::record_txn_commit_failure();
      return false;
    }
    DEBUG_SYNC_C("vector_explicit_txn_after_truth_stage");
    if (publication_guard.try_lock_indexes(pending_index_names)) break;

    // A statement publication can wait for one of the hidden rows staged
    // above. Release every InnoDB row lock before waiting for its reservation,
    // then rebuild from a fresh registry snapshot after it completes.
    staged_delta.cancel();
    {
      vector_statement_publication::publication_guard wait_guard;
      if (!wait_guard.lock_indexes(pending_index_names)) {
        vector_status::record_txn_commit_failure();
        return false;
      }
    }
    retrying_after_publication_contention = true;
  }

  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked() ||
        g_index_service.pending_change_count(txn_id) != pending_change_count) {
      return false;
    }
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
    if (!stage_delta_before_publication &&
        !g_index_service.snapshot_pending_change_delta(txn_id,
                                                       &pending_delta)) {
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
    if (!stage_delta_before_publication) {
      pending_delta_ms = vector_index_diagnostics::elapsed_ms_if(
          diagnostics_enabled, pending_delta_start);
    }
    if (!stage_delta_before_publication &&
        (!allocate_pending_change_log_delta_locked(txn_id, pending_delta,
                                                   &commit_delta_rows) ||
         !bind_commit_truth_generations(&commit_delta_rows, &build_plan))) {
      vector_status::record_txn_commit_failure();
      if (!rollback_runtime_commit_state_or_fail_locked(before_state)) {
        return false;
      }
      return false;
    }
    const bool ok =
        g_index_service.apply_commit_build_plan(txn_id, &build_plan);
    index_commit_ms = vector_index_diagnostics::elapsed_ms_if(
        diagnostics_enabled, index_commit_start);
    if (!ok) {
      vector_status::record_txn_commit_failure();
      if (!rollback_runtime_commit_state_or_fail_locked(before_state)) {
        return false;
      }
      record_commit_diagnostics(
          scope, txn_id, pending_change_count, false, pending_index_names_ms,
          snapshot_before_ms, index_commit_ms, pending_delta_ms, 0, 0,
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  total_start));
      return false;
    }

    const auto changelog_delta_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    const bool changelog_updated =
        compact_staged_change_log
            ? erase_durable_change_log_sequences_locked(
                  compacted_change_log_sequences)
            : append_durable_change_log_rows_locked(commit_delta_rows);
    if (!changelog_updated) {
      if (!rollback_runtime_and_pending_state_locked(
              txn_id, before_state, before_pending_state)) {
        return false;
      }
      vector_status::record_txn_commit_failure();
      return false;
    }
    changelog_delta_ms =
        pending_delta_ms + vector_index_diagnostics::elapsed_ms_if(
                               diagnostics_enabled, changelog_delta_start);
    const bool change_log_size_valid =
        compact_staged_change_log
            ? compacted_change_log_sequences.size() <=
                      change_log_size_before &&
                  g_change_log_rows.size() ==
                      change_log_size_before -
                          compacted_change_log_sequences.size()
            : g_change_log_rows.size() ==
                  change_log_size_before + commit_delta_rows.size();
    if (!change_log_size_valid) {
      if (!rollback_runtime_and_pending_state_locked(
              txn_id, before_state, before_pending_state)) {
        return false;
      }
      vector_status::record_txn_commit_failure();
      return false;
    }

    vector_status::subtract_pending_txn_changes(pending_change_count);
    const auto persist_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    const bool persisted = stage_delta_before_publication
                               ? staged_delta.finish_locked()
                               : persist_commit_artifacts_locked(
                                     &commit_delta_rows);
    if (!persisted) {
      persist_ms =
          vector_index_diagnostics::elapsed_ms_if(diagnostics_enabled,
                                                  persist_start);
      if (!rollback_runtime_and_pending_state_locked(
              txn_id, before_state, before_pending_state)) {
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

namespace {

std::string account_component(LEX_CSTRING value) {
  return value.str == nullptr ? std::string()
                              : std::string(value.str, value.length);
}

explicit_txn_owner explicit_owner_for_thd(const THD *thd) {
  explicit_txn_owner owner;
  if (thd == nullptr) return owner;
  owner.thd_id = static_cast<uint64_t>(thd->thread_id());
  owner.user = account_component(thd->m_main_security_ctx.priv_user());
  owner.host = account_component(thd->m_main_security_ctx.priv_host());
  return owner;
}

bool explicit_owner_matches(const explicit_txn_owner &expected,
                            const explicit_txn_owner &caller) {
  return expected.thd_id == caller.thd_id && expected.user == caller.user &&
         expected.host == caller.host;
}

bool active_explicit_txn_owned_by_locked(
    uint64_t txn_id, const explicit_txn_owner &caller) {
  const auto it = g_explicit_txn_owners.find(txn_id);
  return it != g_explicit_txn_owners.end() &&
         it->second.state == explicit_txn_state::kActive &&
         explicit_owner_matches(it->second, caller);
}

bool erase_explicit_txn_owner_locked(uint64_t txn_id) {
  const auto owner_it = g_explicit_txn_owners.find(txn_id);
  if (owner_it == g_explicit_txn_owners.end()) return false;

  const uint64_t thd_id = owner_it->second.thd_id;
  g_explicit_txn_owners.erase(owner_it);
  auto thd_it = g_explicit_txns_by_thd.find(thd_id);
  if (thd_it == g_explicit_txns_by_thd.end()) return true;
  thd_it->second.erase(txn_id);
  if (!thd_it->second.empty()) return false;
  g_explicit_txns_by_thd.erase(thd_it);
  return true;
}

uint64_t begin_txn_for_owner(const explicit_txn_owner &owner) {
  if (owner.thd_id == 0) return 0;
  if (!ensure_publication_intents_recovered()) return 0;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return 0;

  const uint64_t txn_id = allocate_txn_id();
  if (txn_id == 0) return 0;
  g_explicit_txn_owners.emplace(txn_id, owner);
  g_explicit_txns_by_thd[owner.thd_id].insert(txn_id);
  return txn_id;
}

bool commit_txn_for_owner(const explicit_txn_owner &caller, uint64_t txn_id,
                          bool *last_for_thd) {
  if (last_for_thd != nullptr) *last_for_thd = false;
  vector_status::record_txn_commit_request();
  if (txn_id == 0) return false;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!active_explicit_txn_owned_by_locked(txn_id, caller)) return false;
    g_explicit_txn_owners[txn_id].state = explicit_txn_state::kCommitting;
  }

  const bool committed = commit_index_service_txn("explicit", txn_id, false);
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  const auto owner_it = g_explicit_txn_owners.find(txn_id);
  if (owner_it == g_explicit_txn_owners.end() ||
      !explicit_owner_matches(owner_it->second, caller)) {
    return false;
  }
  if (!committed) {
    owner_it->second.state = explicit_txn_state::kActive;
    return false;
  }
  const bool last = erase_explicit_txn_owner_locked(txn_id);
  if (last_for_thd != nullptr) *last_for_thd = last;
  return true;
}

bool rollback_txn_for_owner(const explicit_txn_owner &caller, uint64_t txn_id,
                            bool *last_for_thd) {
  if (last_for_thd != nullptr) *last_for_thd = false;
  vector_status::record_txn_rollback_request();
  if (txn_id == 0) return false;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!active_explicit_txn_owned_by_locked(txn_id, caller)) return false;

  const size_t pending_change_count =
      g_index_service.pending_change_count(txn_id);
  g_index_service.rollback(txn_id);
  if (pending_change_count > 0) {
    vector_status::subtract_pending_txn_changes(pending_change_count);
  }
  const bool last = erase_explicit_txn_owner_locked(txn_id);
  if (last_for_thd != nullptr) *last_for_thd = last;
  return true;
}

bool pending_txn_changes_for_owner(const explicit_txn_owner &caller,
                                   uint64_t txn_id, size_t *pending_count) {
  if (txn_id == 0 || pending_count == nullptr) return false;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!active_explicit_txn_owned_by_locked(txn_id, caller)) return false;
  *pending_count = g_index_service.pending_change_count(txn_id);
  return true;
}

bool savepoint_txn_for_owner(const explicit_txn_owner &caller, uint64_t txn_id,
                             const std::string &name) {
  if (txn_id == 0) return false;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!active_explicit_txn_owned_by_locked(txn_id, caller)) return false;
  return g_index_service.savepoint(txn_id, name);
}

bool rollback_to_savepoint_txn_for_owner(const explicit_txn_owner &caller,
                                         uint64_t txn_id,
                                         const std::string &name) {
  if (txn_id == 0) return false;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!active_explicit_txn_owned_by_locked(txn_id, caller)) return false;
  const size_t pending_before = g_index_service.pending_change_count(txn_id);
  if (!g_index_service.rollback_to_savepoint(txn_id, name)) return false;

  const size_t pending_after = g_index_service.pending_change_count(txn_id);
  if (pending_before > pending_after) {
    vector_status::subtract_pending_txn_changes(pending_before - pending_after);
  }
  return true;
}

bool release_savepoint_txn_for_owner(const explicit_txn_owner &caller,
                                     uint64_t txn_id,
                                     const std::string &name) {
  if (txn_id == 0) return false;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!active_explicit_txn_owned_by_locked(txn_id, caller)) return false;
  return g_index_service.release_savepoint(txn_id, name);
}

bool stage_upsert_for_owner(const explicit_txn_owner &caller, uint64_t txn_id,
                            const std::string &index_name, uint64_t doc_id,
                            const vector_index::vector_data &vector) {
  vector_status::record_stage_upsert_request();
  if (txn_id == 0) return false;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!active_explicit_txn_owned_by_locked(txn_id, caller)) return false;
  const bool ok = g_index_service.stage_upsert(txn_id, index_name, doc_id, vector);
  if (ok) vector_status::add_pending_txn_changes(1);
  return ok;
}

bool stage_erase_for_owner(const explicit_txn_owner &caller, uint64_t txn_id,
                           const std::string &index_name, uint64_t doc_id) {
  vector_status::record_stage_erase_request();
  if (txn_id == 0) return false;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!active_explicit_txn_owned_by_locked(txn_id, caller)) return false;
  const bool ok = g_index_service.stage_erase(txn_id, index_name, doc_id);
  if (ok) vector_status::add_pending_txn_changes(1);
  return ok;
}

bool stage_internal_upsert(uint64_t txn_id, const std::string &index_name,
                           uint64_t doc_id,
                           const vector_index::vector_data &vector) {
  vector_status::record_stage_upsert_request();
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  const bool ok = g_index_service.stage_upsert(txn_id, index_name, doc_id, vector);
  if (ok) vector_status::add_pending_txn_changes(1);
  return ok;
}

bool stage_internal_erase(uint64_t txn_id, const std::string &index_name,
                          uint64_t doc_id) {
  vector_status::record_stage_erase_request();
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  const bool ok = g_index_service.stage_erase(txn_id, index_name, doc_id);
  if (ok) vector_status::add_pending_txn_changes(1);
  return ok;
}

void rollback_internal_txn(uint64_t txn_id) {
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  const size_t pending_change_count =
      g_index_service.pending_change_count(txn_id);
  g_index_service.rollback(txn_id);
  if (pending_change_count > 0) {
    vector_status::subtract_pending_txn_changes(pending_change_count);
  }
}

size_t rollback_explicit_txns_for_thd_id(uint64_t thd_id) {
  if (thd_id == 0) return 0;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  const auto thd_it = g_explicit_txns_by_thd.find(thd_id);
  if (thd_it == g_explicit_txns_by_thd.end()) return 0;

  size_t rolled_back = 0;
  for (const uint64_t txn_id : thd_it->second) {
    const size_t pending_change_count =
        g_index_service.pending_change_count(txn_id);
    g_index_service.rollback(txn_id);
    if (pending_change_count > 0) {
      vector_status::subtract_pending_txn_changes(pending_change_count);
    }
    g_explicit_txn_owners.erase(txn_id);
    ++rolled_back;
  }
  g_explicit_txns_by_thd.erase(thd_it);
  return rolled_back;
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
const explicit_txn_owner &unit_test_explicit_owner() {
  static const explicit_txn_owner owner{1, "vector_unit", "localhost",
                                        explicit_txn_state::kActive};
  return owner;
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace

uint64_t begin_txn(THD *thd) {
  const explicit_txn_owner owner = explicit_owner_for_thd(thd);
  const uint64_t txn_id = begin_txn_for_owner(owner);
  if (txn_id == 0) return 0;
  if (vector_trx_participant::register_explicit_txn_owner(thd)) return txn_id;

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  (void)erase_explicit_txn_owner_locked(txn_id);
  return 0;
}

bool commit_txn(THD *thd, uint64_t txn_id) {
  bool last_for_thd = false;
  const bool ok =
      commit_txn_for_owner(explicit_owner_for_thd(thd), txn_id, &last_for_thd);
  if (last_for_thd) vector_trx_participant::release_explicit_txn_owner(thd);
  return ok;
}

bool rollback_txn(THD *thd, uint64_t txn_id) {
  bool last_for_thd = false;
  const bool ok = rollback_txn_for_owner(explicit_owner_for_thd(thd), txn_id,
                                         &last_for_thd);
  if (last_for_thd) vector_trx_participant::release_explicit_txn_owner(thd);
  return ok;
}

bool pending_txn_changes(THD *thd, uint64_t txn_id, size_t *pending_count) {
  return pending_txn_changes_for_owner(explicit_owner_for_thd(thd), txn_id,
                                       pending_count);
}

bool savepoint_txn(THD *thd, uint64_t txn_id, const std::string &name) {
  return savepoint_txn_for_owner(explicit_owner_for_thd(thd), txn_id, name);
}

bool rollback_to_savepoint_txn(THD *thd, uint64_t txn_id,
                               const std::string &name) {
  return rollback_to_savepoint_txn_for_owner(explicit_owner_for_thd(thd),
                                              txn_id, name);
}

bool release_savepoint_txn(THD *thd, uint64_t txn_id,
                           const std::string &name) {
  return release_savepoint_txn_for_owner(explicit_owner_for_thd(thd), txn_id,
                                         name);
}

bool stage_upsert(THD *thd, uint64_t txn_id, const std::string &index_name,
                  uint64_t doc_id,
                  const vector_index::vector_data &vector) {
  return stage_upsert_for_owner(explicit_owner_for_thd(thd), txn_id,
                                index_name, doc_id, vector);
}

bool stage_erase(THD *thd, uint64_t txn_id, const std::string &index_name,
                 uint64_t doc_id) {
  return stage_erase_for_owner(explicit_owner_for_thd(thd), txn_id, index_name,
                               doc_id);
}

size_t rollback_explicit_txns_for_thd(THD *thd) {
  if (thd == nullptr) return 0;
  const size_t rolled_back = rollback_explicit_txns_for_thd_id(
      static_cast<uint64_t>(thd->thread_id()));
  vector_trx_participant::release_explicit_txn_owner(thd);
  return rolled_back;
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
uint64_t begin_txn() { return begin_txn_for_owner(unit_test_explicit_owner()); }

bool commit_txn(uint64_t txn_id) {
  return commit_txn_for_owner(unit_test_explicit_owner(), txn_id, nullptr);
}

bool rollback_txn(uint64_t txn_id) {
  return rollback_txn_for_owner(unit_test_explicit_owner(), txn_id, nullptr);
}

size_t pending_txn_changes(uint64_t txn_id) {
  size_t pending_count = 0;
  return pending_txn_changes_for_owner(unit_test_explicit_owner(), txn_id,
                                       &pending_count)
             ? pending_count
             : 0;
}

bool savepoint_txn(uint64_t txn_id, const std::string &name) {
  return savepoint_txn_for_owner(unit_test_explicit_owner(), txn_id, name);
}

bool rollback_to_savepoint_txn(uint64_t txn_id, const std::string &name) {
  return rollback_to_savepoint_txn_for_owner(unit_test_explicit_owner(),
                                              txn_id, name);
}

bool release_savepoint_txn(uint64_t txn_id, const std::string &name) {
  return release_savepoint_txn_for_owner(unit_test_explicit_owner(), txn_id,
                                         name);
}

bool stage_upsert(uint64_t txn_id, const std::string &index_name,
                  uint64_t doc_id,
                  const vector_index::vector_data &vector) {
  return stage_upsert_for_owner(unit_test_explicit_owner(), txn_id, index_name,
                                doc_id, vector);
}

bool stage_erase(uint64_t txn_id, const std::string &index_name,
                 uint64_t doc_id) {
  return stage_erase_for_owner(unit_test_explicit_owner(), txn_id, index_name,
                               doc_id);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

bool stage_upsert_for_thd_txn(uint64_t thd_id, uint64_t statement_id,
                          const std::string &index_name, uint64_t doc_id,
                          const vector_index::vector_data &vector) {
  vector_status::record_stage_upsert_request();
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;

  thd_txn_context *ctx = get_or_create_thd_txn_context_locked(thd_id);
  if (ctx == nullptr) return false;
  if (ctx->attached_dml) return false;
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
  if (ctx == nullptr) return false;
  if (ctx->attached_dml) return false;
  if (!ensure_stmt_savepoint_locked(ctx, statement_id)) return false;

  const bool ok = g_index_service.stage_erase(ctx->txn_id, index_name, doc_id);
  if (ok) vector_status::add_pending_txn_changes(1);
  return ok;
}

bool stage_changes_for_thd_txn(
    THD *thd, uint64_t statement_id,
    const std::vector<vector_index::index_service::pending_change_snapshot>
        &changes) {
  if (thd == nullptr || changes.empty() ||
      !ensure_publication_intents_recovered()) {
    return false;
  }

  const uint64_t thd_id = static_cast<uint64_t>(thd->thread_id());
  uint64_t txn_id = 0;
  std::vector<vector_index_metadata_store::change_log_row> durable_rows;
  std::vector<vector_index_truth_store::publication_intent>
      publication_intents;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked()) return false;

    thd_txn_context *ctx = get_or_create_thd_txn_context_locked(thd_id);
    if (ctx == nullptr) return false;
    if (ctx->publication_prepared) return false;
    if (!ctx->attached_dml &&
        g_index_service.pending_change_count(ctx->txn_id) != 0) {
      return false;
    }
    if (!ensure_stmt_savepoint_locked(ctx, statement_id))
      return false;
    ctx->attached_dml = true;
    txn_id = ctx->txn_id;

    for (const auto &change : changes) {
      if (change.erase) {
        vector_status::record_stage_erase_request();
      } else {
        vector_status::record_stage_upsert_request();
      }
      const bool staged =
          change.erase
              ? g_index_service.stage_erase(txn_id, change.index_name,
                                            change.doc_id)
              : g_index_service.stage_upsert(txn_id, change.index_name,
                                             change.doc_id, change.vector);
      if (!staged) {
        (void)rollback_staged_statement_locked(ctx);
        return false;
      }
      vector_status::add_pending_txn_changes(1);
    }

    if (!allocate_pending_change_log_delta_locked(txn_id, changes,
                                                  &durable_rows)) {
      (void)rollback_staged_statement_locked(ctx);
      return false;
    }
  }

  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (!truth_store->supports_attached_dml() ||
      !truth_store->apply_attached_committed(thd, durable_rows)) {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    auto it = g_thd_txn_contexts.find(thd_id);
    if (it != g_thd_txn_contexts.end() && it->second.txn_id == txn_id) {
      (void)rollback_staged_statement_locked(&it->second);
    }
    return false;
  }

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end() || it->second.txn_id != txn_id) {
    return false;
  }
  try {
    it->second.durable_change_log_rows.insert(
        it->second.durable_change_log_rows.end(), durable_rows.begin(),
        durable_rows.end());
  } catch (...) {
    (void)rollback_staged_statement_locked(&it->second);
    return false;
  }
  return true;
}

bool pending_index_names_for_thd_txn(
    uint64_t thd_id, std::vector<std::string> *index_names) {
  if (index_names == nullptr) return false;
  index_names->clear();
  if (vector_index_truth_store::internal_sql_active()) return true;

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  const auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end()) return true;
  if (!ensure_metadata_loaded_locked()) return false;
  return g_index_service.pending_change_index_names(it->second.txn_id,
                                                    index_names);
}

bool prepare_thd_txn_publication(THD *thd, uint64_t thd_id) {
  if (thd == nullptr) return false;
  if (vector_index_truth_store::internal_sql_active()) return true;

  uint64_t txn_id = 0;
  std::vector<vector_index_metadata_store::change_log_row> durable_rows;
  std::vector<vector_index_truth_store::publication_intent>
      publication_intents;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    auto it = g_thd_txn_contexts.find(thd_id);
    if (it == g_thd_txn_contexts.end()) return true;
    if (!ensure_metadata_loaded_locked()) return false;

    thd_txn_context &ctx = it->second;
    if (ctx.publication_prepared) return true;
    txn_id = ctx.txn_id;
    const size_t pending_change_count =
        g_index_service.pending_change_count(txn_id);
    if (pending_change_count == 0) {
      return ctx.durable_change_log_rows.empty();
    }
    if (pending_change_count != ctx.durable_change_log_rows.size()) {
      return false;
    }

    vector_index::index_service::commit_build_plan build_plan;
    if (!g_index_service.snapshot_commit_build_plan(txn_id, &build_plan)) {
      return false;
    }
    durable_rows = ctx.durable_change_log_rows;
    if (!bind_commit_truth_generations(&durable_rows, &build_plan) ||
        !build_transactional_publication_intents_locked(
            txn_id, build_plan, &publication_intents)) {
      return false;
    }
  }

  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (!truth_store->supports_attached_dml() ||
      !truth_store->supports_publication_intents() ||
      !truth_store->prepare_attached_publication(
          thd, durable_rows, publication_intents)) {
    return false;
  }

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end() || it->second.txn_id != txn_id ||
      it->second.publication_prepared) {
    return false;
  }
  it->second.durable_change_log_rows = std::move(durable_rows);
  it->second.publication_intents = std::move(publication_intents);
  it->second.publication_prepared = true;
  return true;
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

  return rollback_staged_statement_locked(&ctx);
}

bool publish_thd_txn(uint64_t thd_id, bool recover_detached_xa,
                     std::string *failure_stage) {
  if (failure_stage != nullptr) failure_stage->clear();
  if (vector_index_truth_store::internal_sql_active()) return true;

  uint64_t txn_id = 0;
  bool has_context = false;
  std::vector<vector_index_metadata_store::change_log_row> durable_rows;
  std::vector<vector_index_truth_store::publication_intent>
      publication_intents;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    auto it = g_thd_txn_contexts.find(thd_id);
    if (it != g_thd_txn_contexts.end()) {
      if (!ensure_metadata_loaded_locked()) {
        set_publication_failure(failure_stage, "metadata_load_thd_context");
        discard_thd_txn_context_locked(thd_id);
        request_publication_intent_recovery();
        return false;
      }

      has_context = true;
      thd_txn_context &ctx = it->second;
      if (ctx.stmt_savepoint_active) {
        if (!g_index_service.release_savepoint(ctx.txn_id,
                                               ctx.stmt_savepoint_name)) {
          set_publication_failure(failure_stage, "release_statement_savepoint");
          discard_thd_txn_context_locked(thd_id);
          request_publication_intent_recovery();
          return false;
        }
        clear_stmt_savepoint_state(&ctx);
      }
      txn_id = ctx.txn_id;
      durable_rows = ctx.durable_change_log_rows;
      publication_intents = ctx.publication_intents;
    }
  }

  if (!has_context) {
    if (!recover_detached_xa) return true;
    request_publication_intent_recovery();
    return recover_publication_intents_if_requested(failure_stage);
  }

  const bool published =
      publish_pending_runtime(txn_id, durable_rows, true, failure_stage);
  if (!published) {
    {
      std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
      auto it = g_thd_txn_contexts.find(thd_id);
      if (it != g_thd_txn_contexts.end() && it->second.txn_id == txn_id) {
        discard_thd_txn_context_locked(thd_id);
      }
    }
    request_publication_intent_recovery();
    return false;
  }

  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    auto it = g_thd_txn_contexts.find(thd_id);
    if (it != g_thd_txn_contexts.end() && it->second.txn_id == txn_id) {
      g_thd_txn_contexts.erase(it);
    }
  }
  const bool acknowledged =
      acknowledge_publication_intents(publication_intents, failure_stage);
  if (!acknowledged) request_publication_intent_recovery();
  return acknowledged;
}

bool detach_thd_txn_for_prepare(uint64_t thd_id) {
  if (vector_index_truth_store::internal_sql_active()) return true;
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  auto it = g_thd_txn_contexts.find(thd_id);
  if (it == g_thd_txn_contexts.end()) return true;
  if (!ensure_metadata_loaded_locked()) return false;

  const size_t pending_change_count =
      g_index_service.pending_change_count(it->second.txn_id);
  g_index_service.rollback(it->second.txn_id);
  if (pending_change_count > 0) {
    vector_status::subtract_pending_txn_changes(pending_change_count);
  }
  g_thd_txn_contexts.erase(it);
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
  if (replay_txn_id == 0) return XAER_RMERR;
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
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      fail_stop_registry_locked("prepared_runtime_state_restore_failed");
    }
    return XAER_RMERR;
  }

  const size_t change_log_size_before = g_change_log_rows.size();
  if (!append_pending_change_log_delta_locked(replay_txn_id, pending_delta)) {
    if (!rollback_runtime_state_locked(metadata_before, committed_before,
                                       change_log_before,
                                       lagging_indexes_before)) {
      fail_stop_registry_locked("prepared_runtime_state_restore_failed");
    }
    return XAER_RMERR;
  }
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
      fail_stop_registry_locked("prepared_runtime_state_restore_failed");
      return XAER_RMERR;
    }
    if (needs_persisted_snapshot &&
        !restore_persisted_commit_artifacts_snapshot_locked(persisted_before)) {
      fail_stop_registry_locked("persisted_commit_artifacts_restore_failed");
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
    if (!xid_from_prepared_row(row, &xid)) return 0;

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
    if (!xid_from_prepared_row(row, &xid)) return 1;
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
  if (ctx == nullptr) return false;
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
  if (ctx.attached_dml) {
    if (pending_after > ctx.durable_change_log_rows.size()) return false;
    ctx.durable_change_log_rows.resize(pending_after);
  }
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

bool make_create_statement_payload(
    size_t dimension, const std::string &metric, const std::string &mode,
    const std::string &provider, const std::string &owner_schema,
    const create_index_options &options, std::string *payload) {
  if (dimension == 0 || dimension > vector_index::k_max_vector_dimension ||
      owner_schema.empty() || payload == nullptr) {
    return false;
  }

  std::string resolved_mode;
  std::string resolved_provider;
  create_index_options resolved_options;
  if (!resolve_create_index_definition(mode, provider, options, &resolved_mode,
                                       &resolved_provider, &resolved_options)) {
    return false;
  }

  vector_statement_publication::operation_payload operation_payload;
  operation_payload.unsigned_values = {
      static_cast<uint64_t>(dimension),
      resolved_options.build_threads,
      resolved_options.diskann_max_degree,
      resolved_options.diskann_build_complexity,
      resolved_options.diskann_pq_code_budget_size,
      resolved_options.diskann_disk_pq_dims,
      resolved_options.diskann_accelerate_build ? 1U : 0U,
      resolved_options.diskann_shuffle_build ? 1U : 0U,
      resolved_options.diskann_use_bfs_cache ? 1U : 0U,
      resolved_options.diskann_search_complexity,
      resolved_options.diskann_search_beamwidth,
      static_cast<uint64_t>(resolved_options.consistency_mode)};
  operation_payload.string_values = {
      metric, resolved_mode, resolved_provider, owner_schema,
      resolved_options.initial_lifecycle_state};
  return create_statement_payload_valid(operation_payload) &&
         vector_statement_publication::encode_operation_payload(
             operation_payload, payload);
}

bool make_statement_publication_intent(
    vector_index_truth_store::publication_operation operation,
    const std::string &index_name, const std::string &payload,
    vector_index_truth_store::publication_intent *intent) {
  if (index_name.empty() || intent == nullptr ||
      operation ==
          vector_index_truth_store::publication_operation::kTransactionalDml ||
      !recover_publication_intents_if_requested()) {
    return false;
  }

  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;

  vector_index_truth_store::publication_intent candidate;
  candidate.index_name = index_name;
  candidate.publication_id = allocate_txn_id();
  candidate.txn_id = candidate.publication_id;
  candidate.operation = operation;
  candidate.payload = payload;
  candidate.state = vector_index_truth_store::publication_intent_state::
      kTargetToBeObserved;
  if (candidate.publication_id == 0 ||
      !describe_publication_token_locked(index_name, &candidate.expected)) {
    return false;
  }
  const bool creates_index =
      operation == vector_index_truth_store::publication_operation::
                       kCreateIndex;
  if (creates_index == candidate.expected.exists) return false;
  candidate.target =
      operation == vector_index_truth_store::publication_operation::kDropIndex
          ? vector_index_truth_store::publication_token{}
          : candidate.expected;
  *intent = std::move(candidate);
  return true;
}

bool validate_statement_publication_intent(
    const vector_index_truth_store::publication_intent &intent) {
  if (intent.state != vector_index_truth_store::publication_intent_state::
                          kTargetToBeObserved) {
    return false;
  }
  std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
  if (!ensure_metadata_loaded_locked()) return false;
  vector_index_truth_store::publication_token current;
  return describe_publication_token_locked(intent.index_name, &current) &&
         publication_tokens_equal(current, intent.expected) &&
         validate_statement_operation_locked(intent);
}

bool publish_statement_publication_intent(
    const vector_index_truth_store::publication_intent &intent,
    std::string *failure_stage) {
  if (failure_stage != nullptr) failure_stage->clear();
  if (intent.state != vector_index_truth_store::publication_intent_state::
                          kTargetToBeObserved) {
    set_publication_failure(failure_stage, "invalid_statement_intent_state");
    return false;
  }

  vector_index_truth_store::publication_token current;
  bool expected_matches = false;
  bool can_resume_after_source_preparation = false;
  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked() ||
        !describe_publication_token_locked(intent.index_name, &current)) {
      set_publication_failure(failure_stage,
                              "describe_statement_publication_token");
      return false;
    }
    expected_matches = publication_tokens_equal(current, intent.expected);
    can_resume_after_source_preparation =
        statement_intent_can_resume_after_source_preparation(intent, current);
    if ((expected_matches || can_resume_after_source_preparation) &&
        !validate_statement_operation_locked(intent)) {
      set_publication_failure(failure_stage,
                              "statement_publication_precondition");
      return false;
    }
  }
  if (!expected_matches) {
    if (statement_operation_applied(intent, current, failure_stage)) {
      return true;
    }
    if (!can_resume_after_source_preparation) {
      set_publication_failure(failure_stage,
                              "statement_publication_cas_mismatch");
      return false;
    }
  }
  if (intent.operation ==
          vector_index_truth_store::publication_operation::kBulkLoad &&
      statement_operation_applied(intent, current, failure_stage)) {
    return true;
  }

  if (!apply_statement_operation(intent, failure_stage)) return false;

  {
    std::lock_guard<std::shared_mutex> guard(g_registry_mutex);
    if (!ensure_metadata_loaded_locked() ||
        !describe_publication_token_locked(intent.index_name, &current)) {
      set_publication_failure(failure_stage,
                              "describe_statement_publication_target");
      return false;
    }
  }
  if (!statement_operation_applied(intent, current, failure_stage)) {
    set_publication_failure(failure_stage,
                            "statement_publication_target_mismatch");
    return false;
  }
  return true;
}

bool acknowledge_statement_publication_intent(
    const vector_index_truth_store::publication_intent &intent,
    std::string *failure_stage) {
  if (intent.operation ==
          vector_index_truth_store::publication_operation::kDropIndex &&
      !cleanup_dropped_index_physical_state(intent, failure_stage)) {
    request_publication_intent_recovery();
    return false;
  }
  if (!vector_index_truth_store::get()->delete_committed_publication_intent(
          intent.index_name, intent.publication_id)) {
    set_publication_failure(failure_stage,
                            "delete_statement_publication_intent");
    return false;
  }
  if (intent.operation ==
      vector_index_truth_store::publication_operation::kBulkLoad &&
      !vector_index::remove_managed_load_staging_artifact(intent)) {
    set_publication_failure(failure_stage, "remove_load_staging_artifact");
    request_publication_intent_recovery();
    return false;
  }
  if (intent.operation ==
      vector_index_truth_store::publication_operation::kDropIndex) {
    vector_status::record_index_drop_success();
  }
  return true;
}

void schedule_publication_intent_recovery() {
  request_publication_intent_recovery();
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
  if (txn_id == 0) return false;
  if (!stage_internal_upsert(txn_id, index_name, doc_id, vector)) return false;
  vector_status::record_txn_commit_request();
  if (!commit_index_service_txn("direct_upsert", txn_id, false)) {
    rollback_internal_txn(txn_id);
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
  if (txn_id == 0) return false;
  if (!stage_internal_erase(txn_id, index_name, doc_id)) return false;
  vector_status::record_txn_commit_request();
  if (!commit_index_service_txn("direct_erase", txn_id, false)) {
    rollback_internal_txn(txn_id);
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
  DBUG_EXECUTE_IF("vector_fail_single_mapped_batch_fallback", return false;);
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
  bool use_mapped_filter = false;
  vector_mapped_search::search_spec mapped_spec;
  size_t candidate_top_k = top_k;
  size_t candidate_limit = top_k;
  bool candidate_hard_limit = false;
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
      use_mapped_filter =
          !binding.schema_name.empty() && !binding.table_name.empty() &&
          !binding.column_name.empty() &&
          !binding.doc_id_column_name.empty() && thd != nullptr;
      if (use_mapped_filter) {
        mapped_spec.schema_name = binding.schema_name;
        mapped_spec.table_name = binding.table_name;
        mapped_spec.column_name = binding.column_name;
        mapped_spec.doc_id_column_name = binding.doc_id_column_name;
        mapped_spec.metric = config.metric;
        mapped_spec.top_k = top_k;
        candidate_hard_limit =
            mapped_search_candidate_limit_is_hard(entry_count, 0);
        candidate_limit = mapped_search_candidate_limit(top_k, entry_count, 0);
        candidate_top_k =
            mapped_search_initial_candidate_top_k(top_k, entry_count, 0);
      }
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

  if (use_mapped_filter && candidate_limit == 0) {
    results->resize(queries.size());
    ok = true;
  }

  while (use_mapped_filter && !ok && candidate_limit > 0) {
    std::vector<std::vector<vector_index::search_result>> candidate_results;
    if (!search_committed_runtime_batch_loaded(
            index_name, queries, candidate_top_k, &candidate_results)) {
      break;
    }
    for (const auto &query_candidates : candidate_results) {
      vector_status::record_search_mvcc_candidate_rows(
          query_candidates.size());
    }

    std::vector<std::vector<vector_index::search_result>> filtered_results;
    if (!vector_mapped_search::filter_visible_results_batch(
            thd, mapped_spec, queries, candidate_results,
            &filtered_results)) {
      break;
    }

    size_t incomplete_query_count = 0;
    for (const auto &query_results : filtered_results) {
      if (query_results.size() < top_k) ++incomplete_query_count;
    }
    if (incomplete_query_count == 0 || candidate_top_k >= candidate_limit) {
      if (candidate_hard_limit && candidate_top_k >= candidate_limit) {
        for (const auto &query_results : filtered_results) {
          if (query_results.size() < top_k)
            vector_status::record_search_mvcc_limit_hit();
        }
      }
      *results = std::move(filtered_results);
      ok = true;
      break;
    }

    for (size_t i = 0; i < incomplete_query_count; ++i)
      vector_status::record_search_mvcc_expansion();
    candidate_top_k = mapped_search_next_candidate_top_k(
        candidate_top_k, top_k, candidate_limit);
  }

  if (!use_mapped_filter) {
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

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool detail::publication_intent_target_matches_for_testing(
    const vector_index_truth_store::publication_intent &intent,
    const vector_index_truth_store::publication_token &current) {
  return publication_intent_target_matches(intent, current);
}

bool detail::bind_commit_truth_generations_for_testing(
    std::vector<vector_index_metadata_store::change_log_row> *rows,
    vector_index::index_service::commit_build_plan *plan) {
  return bind_commit_truth_generations(rows, plan);
}

bool detail::publish_pending_runtime_for_testing(
    uint64_t txn_id,
    const std::vector<vector_index_metadata_store::change_log_row>
        &durable_rows,
    bool update_pending_status, std::string *failure_stage) {
  return publish_pending_runtime(txn_id, durable_rows, update_pending_status,
                                 failure_stage);
}

void detail::reset_publication_intent_recovery_for_testing() {
  std::lock_guard<std::mutex> recovery_guard(g_publication_recovery_mutex);
  g_publication_recovery_requests.store(1, std::memory_order_relaxed);
  g_publication_recovery_completed.store(0, std::memory_order_relaxed);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index_registry
