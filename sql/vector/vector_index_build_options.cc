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

#include "sql/vector/vector_index_build_options.h"

#include <algorithm>

#include "sql/vector/vector_build_pipeline_policy.h"
#include "sql/vector/vector_diskann_scheduler.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_runtime_thread_pool.h"
#include "sql/vector/vector_segment_runtime_scheduler.h"

ulong opt_vector_hnsw_build_threads = 0;
ulong opt_vector_faiss_build_threads = 0;
ulong opt_vector_diskann_build_threads = 0;
ulong opt_vector_diskann_build_blas_threads = 1;
ulong opt_vector_search_batch_count =
    vector_index::k_default_search_batch_count;
ulong opt_vector_search_batch_result_count =
    vector_index::k_default_search_batch_result_count;
ulong opt_vector_batch_search_threads =
    vector_index::k_default_batch_search_threads;
ulong opt_vector_hnsw_search_threads = 0;
ulong opt_vector_faiss_search_threads = 0;
ulong opt_vector_diskann_search_threads = 0;
ulong opt_vector_diskann_offline_search_threads =
    vector_index::k_default_diskann_offline_search_threads;
ulong opt_vector_diskann_search_io_limit = 0;
ulong opt_vector_diskann_cache_nodes = 0;
ulonglong opt_vector_diskann_search_cache_size = 0;
double opt_vector_diskann_search_cache_ratio = 0.1;
ulong opt_vector_diskann_search_complexity =
    vector_index::k_default_diskann_search_complexity;
ulong opt_vector_diskann_search_profile =
    static_cast<ulong>(vector_index::diskann_search_profile::kManual);
ulong opt_vector_diskann_search_beamwidth =
    vector_index::k_default_diskann_search_beamwidth;
ulong opt_vector_diskann_exact_rerank_candidates = 0;
ulonglong opt_vector_diskann_pq_code_budget_size = 0;
double opt_vector_diskann_pq_code_budget_ratio =
    vector_index::k_default_diskann_pq_code_budget_ratio;
ulong opt_vector_diskann_disk_pq_dims = 0;
bool opt_vector_diskann_accelerate_build = false;
bool opt_vector_diskann_shuffle_build = false;
bool opt_vector_diskann_use_bfs_cache = false;
bool opt_vector_diskann_segmented_serving = true;
ulong opt_vector_diskann_max_degree =
    vector_index::k_default_diskann_max_degree;
ulong opt_vector_diskann_build_complexity =
    vector_index::k_default_diskann_build_complexity;
ulong opt_vector_diskann_build_mode =
    static_cast<ulong>(vector_index::diskann_build_mode::kAuto);
ulong opt_vector_diskann_pq_runtime =
    static_cast<ulong>(vector_index::diskann_pq_runtime_mode::kNativeAuto);
ulong opt_vector_default_library =
    static_cast<ulong>(vector_index::vector_default_library::kNone);
ulong opt_vector_index_consistency_mode =
    static_cast<ulong>(vector_index::index_consistency_mode::kTransactional);
ulonglong opt_vector_entry_cache_size = 256ULL * 1024ULL * 1024ULL;
ulonglong opt_vector_pending_cache_size = 64ULL * 1024ULL * 1024ULL;
ulonglong opt_vector_build_memory_size = 0;
ulonglong opt_vector_build_memory_reserve_size = 256ULL * 1024ULL * 1024ULL;
ulong opt_vector_build_resource_wait_timeout = 0;
ulonglong opt_vector_diskann_build_memory_size =
    1024ULL * 1024ULL * 1024ULL;
ulonglong opt_vector_diskann_raw_segment_size = 256ULL * 1024ULL * 1024ULL;
ulonglong opt_vector_faiss_train_size = 0;
bool opt_vector_faiss_keep_loaded = true;
ulonglong opt_vector_hnsw_index_memory_size = 0;
bool opt_vector_lazy_external_runtime = false;
ulong opt_vector_build_pipeline_mode =
    static_cast<ulong>(vector_index::build_pipeline_mode::kAuto);
ulong opt_vector_diskann_segment_profile =
    static_cast<ulong>(vector_index::diskann_segment_profile::kManual);
ulonglong opt_vector_build_pipeline_min_rows =
    vector_index::k_default_build_pipeline_min_rows;
ulonglong opt_vector_build_pipeline_min_size =
    vector_index::k_default_build_pipeline_min_size;
ulonglong opt_vector_build_segment_max_rows =
    vector_index::k_default_build_segment_max_rows;
ulonglong opt_vector_build_segment_target_size =
    vector_index::k_default_build_segment_target_size;
ulong opt_vector_build_pipeline_max_tasks =
    vector_index::k_default_build_pipeline_max_tasks;
ulong opt_vector_build_pipeline_progress_interval =
    vector_index::k_default_build_pipeline_progress_interval;

namespace vector_index {

diskann_build_mode global_diskann_build_mode() {
  switch (static_cast<diskann_build_mode>(opt_vector_diskann_build_mode)) {
    case diskann_build_mode::kAuto:
      return diskann_build_mode::kAuto;
    case diskann_build_mode::kSerial:
      return diskann_build_mode::kSerial;
    case diskann_build_mode::kOffline:
      return diskann_build_mode::kOffline;
  }
  return diskann_build_mode::kAuto;
}

diskann_pq_runtime_mode global_diskann_pq_runtime_mode() {
  switch (static_cast<diskann_pq_runtime_mode>(
      opt_vector_diskann_pq_runtime)) {
    case diskann_pq_runtime_mode::kOfficial:
      return diskann_pq_runtime_mode::kOfficial;
    case diskann_pq_runtime_mode::kNativeAuto:
      return diskann_pq_runtime_mode::kNativeAuto;
    case diskann_pq_runtime_mode::kNativeStrict:
      return diskann_pq_runtime_mode::kNativeStrict;
  }
  return diskann_pq_runtime_mode::kNativeAuto;
}

const char *diskann_pq_runtime_mode_name(diskann_pq_runtime_mode mode) {
  switch (mode) {
    case diskann_pq_runtime_mode::kOfficial:
      return "official";
    case diskann_pq_runtime_mode::kNativeAuto:
      return "native_auto";
    case diskann_pq_runtime_mode::kNativeStrict:
      return "native_strict";
  }
  return "native_auto";
}

bool diskann_pq_runtime_uses_native(diskann_pq_runtime_mode mode) {
  return mode == diskann_pq_runtime_mode::kNativeAuto ||
         mode == diskann_pq_runtime_mode::kNativeStrict;
}

bool diskann_pq_runtime_allows_official_fallback(
    diskann_pq_runtime_mode mode) {
  return mode == diskann_pq_runtime_mode::kNativeAuto;
}

vector_default_library global_vector_default_library() {
  switch (static_cast<vector_default_library>(opt_vector_default_library)) {
    case vector_default_library::kNone:
      return vector_default_library::kNone;
    case vector_default_library::kDiskAnn:
      return vector_default_library::kDiskAnn;
    case vector_default_library::kHnsw:
      return vector_default_library::kHnsw;
    case vector_default_library::kFaiss:
      return vector_default_library::kFaiss;
  }
  return vector_default_library::kNone;
}

index_consistency_mode global_index_consistency_mode() {
  switch (static_cast<index_consistency_mode>(
      opt_vector_index_consistency_mode)) {
    case index_consistency_mode::kTransactional:
      return index_consistency_mode::kTransactional;
    case index_consistency_mode::kStandalone:
      return index_consistency_mode::kStandalone;
  }
  return index_consistency_mode::kTransactional;
}

size_t effective_batch_search_threads(size_t query_count,
                                      ulong backend_override_threads) {
  if (query_count == 0) return 0;

  ulong configured = backend_override_threads;
  if (configured == 0) configured = opt_vector_batch_search_threads;

  return effective_runtime_worker_count(
      query_count, effective_build_scheduler_thread_budget(configured));
}

size_t effective_build_scheduler_thread_budget(ulong backend_override_threads) {
  segment_scheduler_input input;
  input.segment_count = 1;
  input.requested_task_count = 1;
  input.cpu_budget = backend_override_threads == 0 ? 0 : backend_override_threads;
  input.requested_build_threads = backend_override_threads;
  input.requested_blas_threads = backend_override_threads == 0
                                      ? 1
                                      : backend_override_threads;

  segment_scheduler_plan plan;
  if (!make_segment_scheduler_plan(input, &plan)) return 1;
  return std::max<size_t>(1, plan.effective_build_threads);
}

size_t effective_build_scheduler_threads(size_t entry_count,
                                         ulong backend_override_threads) {
  if (entry_count == 0) return 0;
  return effective_runtime_worker_count(
      entry_count,
      effective_build_scheduler_thread_budget(backend_override_threads));
}

size_t effective_hnsw_search_threads(size_t query_count) {
  return effective_batch_search_threads(query_count,
                                        opt_vector_hnsw_search_threads);
}

size_t effective_hnsw_search_thread_budget() {
  ulong configured = opt_vector_hnsw_search_threads;
  if (configured == 0) configured = opt_vector_batch_search_threads;
  return effective_build_scheduler_thread_budget(configured);
}

size_t effective_faiss_search_threads(size_t query_count) {
  return effective_batch_search_threads(query_count,
                                        opt_vector_faiss_search_threads);
}

size_t effective_diskann_search_threads(size_t query_count) {
  return effective_batch_search_threads(query_count,
                                        opt_vector_diskann_search_threads);
}

}  // namespace vector_index
