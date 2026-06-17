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

/**
  Return the effective number of workers for a fixed amount of work.

  A configured value of 0 inherits the machine thread count. The result is
  capped by both the work item count and k_max_build_threads.
*/
size_t effective_runtime_worker_count(size_t item_count,
                                      size_t configured_threads);

/**
  Execute independent item ranges in parallel.

  @retval true All ranges succeeded.
  @retval false Invalid visitor or at least one worker reported failure.
*/
bool parallel_for_ranges(size_t item_count, size_t configured_threads,
                         const range_visitor &visitor);

/**
  Execute query ranges in parallel.

  @retval true All query ranges succeeded.
  @retval false Invalid visitor or at least one worker reported failure.
*/
bool parallel_for_queries(size_t query_count, size_t configured_threads,
                          const query_range_visitor &visitor);

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
#ifdef _OPENMP
  int m_previous_threads{0};
  bool m_active{false};
#endif
};

}  // namespace vector_index

#endif  // SQL_VECTOR_INDEX_RUNTIME_THREAD_POOL_INCLUDED
