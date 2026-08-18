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

#ifndef SQL_VECTOR_INDEX_RUNTIME_THREAD_POOL_INCLUDED
#define SQL_VECTOR_INDEX_RUNTIME_THREAD_POOL_INCLUDED

#include <cstddef>
#include <cstdint>
#include <functional>

namespace vector_index {

using range_visitor =
    std::function<bool(size_t begin, size_t end, size_t worker_id)>;
using query_range_visitor = range_visitor;
using search_item_visitor =
    std::function<bool(size_t item_index, size_t worker_id)>;

/** Execution plan observed when a request enters the shared search pool. */
struct shared_search_execution {
  size_t worker_budget{0};
  size_t active_requests{0};
  size_t effective_workers{0};
  size_t work_items{0};
};

/**
  Return the effective number of workers for a fixed amount of work.

  A configured value of 0 inherits the machine thread count. The result is
  capped by both the work item count and k_max_build_threads.
*/
size_t effective_runtime_worker_count(size_t item_count,
                                      size_t configured_threads);

/**
  Execute independent item ranges in a process-wide bounded worker pool.

  Concurrent callers keep independent completion and failure state. Their
  ranges are interleaved across the shared workers instead of serializing an
  entire request behind another request.

  @retval true All query ranges succeeded.
  @retval false Invalid visitor or at least one worker reported failure.
*/
bool parallel_for_ranges(size_t item_count, size_t configured_threads,
                         const range_visitor &visitor);

/**
  Execute independent item ranges using one-shot scoped worker threads.

  This helper is intended for outer orchestration layers whose visitor may
  call parallel_for_ranges() again through a backend implementation. It avoids
  nesting on the shared runtime worker pool.
*/
bool parallel_for_ranges_scoped(size_t item_count, size_t configured_threads,
                                const range_visitor &visitor);

/**
  Execute query ranges in parallel.
*/
bool parallel_for_queries(size_t query_count, size_t configured_threads,
                          const query_range_visitor &visitor);

/**
  Execute independent search items in a process-wide bounded worker pool.

  Concurrent requests share one worker budget. Work from different requests is
  interleaved instead of creating one complete worker group per SQL request.

  @retval true All search items succeeded.
  @retval false Invalid visitor, worker setup failure, or item failure.
*/
bool parallel_for_shared_search_items(
    size_t item_count, size_t configured_threads,
    const search_item_visitor &visitor,
    shared_search_execution *execution = nullptr);

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
size_t runtime_worker_pool_size_for_testing();
void reset_runtime_worker_pool_for_testing();
size_t shared_search_worker_pool_size_for_testing();
void reset_shared_search_worker_pool_for_testing();
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

/**
  Temporarily override OpenMP thread count for libraries that use OpenMP/MKL.
*/
class scoped_omp_threads {
 public:
  explicit scoped_omp_threads(size_t thread_count);
  ~scoped_omp_threads();

  scoped_omp_threads(const scoped_omp_threads &) = delete;
  scoped_omp_threads &operator=(const scoped_omp_threads &) = delete;

 private:
  int m_previous_threads{0};
  int m_previous_dynamic{0};
  int m_previous_max_active_levels{1};
  bool m_active{false};
};

}  // namespace vector_index

#endif  // SQL_VECTOR_INDEX_RUNTIME_THREAD_POOL_INCLUDED
