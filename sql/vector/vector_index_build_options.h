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

#ifndef SQL_VECTOR_VECTOR_INDEX_BUILD_OPTIONS_H
#define SQL_VECTOR_VECTOR_INDEX_BUILD_OPTIONS_H

#include <cstddef>

#include "my_inttypes.h"
#include "sql/vector/vector_index_backend.h"

/**
  Global vector index build defaults.

  The vector module owns these option values so backend, registry and SQL
  integration commits can share one storage location while exposing each
  backend-specific sysvar at the first commit where the backend becomes usable.
*/
extern ulong opt_vector_hnsw_build_threads;
extern ulong opt_vector_faiss_build_threads;
extern ulong opt_vector_diskann_build_threads;
extern ulong opt_vector_diskann_build_blas_threads;
extern ulong opt_vector_search_batch_count;
extern ulong opt_vector_search_batch_result_count;
extern ulong opt_vector_batch_search_threads;
extern ulong opt_vector_hnsw_search_threads;
extern ulong opt_vector_faiss_search_threads;
extern ulong opt_vector_diskann_search_threads;
extern ulong opt_vector_diskann_offline_search_threads;
extern ulong opt_vector_diskann_search_io_limit;
extern ulong opt_vector_diskann_cache_nodes;
extern ulonglong opt_vector_diskann_search_cache_size;
extern double opt_vector_diskann_search_cache_ratio;
extern ulong opt_vector_diskann_search_complexity;
extern ulong opt_vector_diskann_search_beamwidth;
extern ulonglong opt_vector_diskann_pq_code_budget_size;
extern double opt_vector_diskann_pq_code_budget_ratio;
extern ulong opt_vector_diskann_max_degree;
extern ulong opt_vector_diskann_build_complexity;
extern ulong opt_vector_diskann_build_mode;
extern ulong opt_vector_default_library;
extern ulong opt_vector_index_consistency_mode;
extern ulonglong opt_vector_entry_cache_size;
extern ulonglong opt_vector_pending_cache_size;
extern ulonglong opt_vector_build_memory_size;
extern ulonglong opt_vector_diskann_build_memory_size;
extern ulonglong opt_vector_diskann_raw_segment_size;
extern ulonglong opt_vector_faiss_train_size;
extern bool opt_vector_faiss_keep_loaded;
extern ulonglong opt_vector_hnsw_index_memory_size;
extern bool opt_vector_lazy_external_runtime;
extern ulong opt_vector_build_pipeline_mode;
extern ulonglong opt_vector_build_pipeline_min_rows;
extern ulonglong opt_vector_build_pipeline_min_size;
extern ulonglong opt_vector_build_segment_max_rows;
extern ulonglong opt_vector_build_segment_target_size;
extern ulong opt_vector_build_pipeline_max_tasks;
extern ulong opt_vector_build_pipeline_progress_interval;

namespace vector_index {

enum class vector_default_library : ulong {
  kNone = 0,
  kDiskAnn = 1,
  kHnsw = 2,
  kFaiss = 3
};

/**
  Return the global DiskANN build-mode default.

  The sysvar stores an enum index as ulong; this helper keeps all backend and
  registry users on the vector enum instead of duplicating casts.
*/
diskann_build_mode global_diskann_build_mode();

/** Return the global default vector library. */
vector_default_library global_vector_default_library();

/** Return the global vector-index consistency default. */
index_consistency_mode global_index_consistency_mode();

/** Return the effective build worker count for one backend build. */
size_t effective_build_scheduler_threads(size_t entry_count,
                                         ulong backend_override_threads);

/** Return the effective OpenMP/BLAS build worker budget. */
size_t effective_build_scheduler_thread_budget(
    ulong backend_override_threads);

/** Return the effective batch search worker count after inheritance. */
size_t effective_batch_search_threads(size_t query_count,
                                      ulong backend_override_threads);

/** Return the effective hnswlib batch search worker count. */
size_t effective_hnsw_search_threads(size_t query_count);

/** Return the effective FAISS batch search worker count. */
size_t effective_faiss_search_threads(size_t query_count);

/** Return the effective DiskANN batch search worker count. */
size_t effective_diskann_search_threads(size_t query_count);

}  // namespace vector_index

#endif  // SQL_VECTOR_VECTOR_INDEX_BUILD_OPTIONS_H
