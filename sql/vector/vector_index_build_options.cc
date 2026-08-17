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

#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_runtime_thread_pool.h"

ulong opt_vector_hnsw_build_threads = 0;
ulong opt_vector_faiss_build_threads = 0;
ulong opt_vector_diskann_build_threads = 0;
ulong opt_vector_diskann_build_mode =
    static_cast<ulong>(vector_index::diskann_build_mode::kAuto);
ulong opt_vector_search_batch_count =
    vector_index::k_default_search_batch_count;
ulong opt_vector_search_batch_result_count =
    vector_index::k_default_search_batch_result_count;
ulong opt_vector_batch_search_threads =
    vector_index::k_default_batch_search_threads;
ulong opt_vector_hnsw_search_threads = 0;
ulong opt_vector_default_library =
    static_cast<ulong>(vector_index::vector_default_library::kNone);
ulong opt_vector_index_consistency_mode =
    static_cast<ulong>(vector_index::index_consistency_mode::kTransactional);
ulonglong opt_vector_entry_cache_size = 256ULL * 1024ULL * 1024ULL;
ulonglong opt_vector_pending_cache_size = 64ULL * 1024ULL * 1024ULL;
ulonglong opt_vector_build_memory_size = 1024ULL * 1024ULL * 1024ULL;
ulonglong opt_vector_diskann_build_memory_size =
    1024ULL * 1024ULL * 1024ULL;
ulonglong opt_vector_diskann_raw_segment_size = 256ULL * 1024ULL * 1024ULL;
ulonglong opt_vector_faiss_train_size = 0;
ulonglong opt_vector_hnsw_index_memory_size = 0;
bool opt_vector_lazy_external_runtime = false;

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

size_t effective_build_scheduler_threads(size_t entry_count,
                                         ulong backend_override_threads) {
  return effective_runtime_worker_count(entry_count,
                                        backend_override_threads);
}

size_t effective_hnsw_search_threads(size_t query_count) {
  if (query_count == 0) return 0;

  const ulong configured = opt_vector_hnsw_search_threads == 0
                               ? opt_vector_batch_search_threads
                               : opt_vector_hnsw_search_threads;
  return effective_runtime_worker_count(query_count, configured);
}

}  // namespace vector_index
