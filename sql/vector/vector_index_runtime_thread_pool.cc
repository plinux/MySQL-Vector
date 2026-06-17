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

#include "sql/vector/vector_index_runtime_thread_pool.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sql/vector/vector_index_limits.h"

namespace vector_index {

namespace {

bool invoke_range_visitor(const range_visitor &visitor, size_t begin,
                          size_t end, size_t worker_id) {
  try {
    return visitor(begin, end, worker_id);
  } catch (...) {
    return false;
  }
}

size_t hardware_worker_count() {
  const unsigned int hardware_threads = std::thread::hardware_concurrency();
  return hardware_threads == 0 ? 1 : static_cast<size_t>(hardware_threads);
}

}  // namespace

size_t effective_runtime_worker_count(size_t item_count,
                                      size_t configured_threads) {
  if (item_count == 0) return 0;

  const size_t selected =
      configured_threads == 0 ? hardware_worker_count() : configured_threads;
  const size_t limited = std::min<size_t>(selected, k_max_build_threads);
  return std::max<size_t>(1, std::min(item_count, limited));
}

bool parallel_for_ranges(size_t item_count, size_t configured_threads,
                         const range_visitor &visitor) {
  if (!visitor) return false;

  const size_t thread_count =
      effective_runtime_worker_count(item_count, configured_threads);
  if (thread_count == 0) return true;
  if (thread_count == 1) return invoke_range_visitor(visitor, 0, item_count, 0);

  std::atomic<bool> failed{false};
  std::vector<std::thread> workers;
  std::vector<unsigned char> worker_ok(thread_count, 1);
  const size_t chunk_size = (item_count + thread_count - 1) / thread_count;

  workers.reserve(thread_count);
  for (size_t worker_id = 0; worker_id < thread_count; ++worker_id) {
    const size_t begin = worker_id * chunk_size;
    const size_t end = std::min(item_count, begin + chunk_size);
    if (begin >= end) break;

    workers.emplace_back([&, begin, end, worker_id]() {
      if (failed.load(std::memory_order_relaxed)) return;
      if (!invoke_range_visitor(visitor, begin, end, worker_id)) {
        worker_ok[worker_id] = 0;
        failed.store(true, std::memory_order_relaxed);
      }
    });
  }

  for (std::thread &worker : workers) worker.join();
  return std::all_of(worker_ok.begin(), worker_ok.end(),
                     [](unsigned char ok) { return ok != 0; });
}

bool parallel_for_queries(size_t query_count, size_t configured_threads,
                          const query_range_visitor &visitor) {
  return parallel_for_ranges(query_count, configured_threads, visitor);
}

scoped_omp_threads::scoped_omp_threads(size_t thread_count) {
#ifdef _OPENMP
  if (thread_count == 0) return;

  const size_t limited =
      std::min<size_t>(thread_count,
                       static_cast<size_t>(std::numeric_limits<int>::max()));
  m_previous_threads = omp_get_max_threads();
  omp_set_num_threads(static_cast<int>(limited));
  m_active = true;
#else
  (void)thread_count;
#endif
}

scoped_omp_threads::~scoped_omp_threads() {
#ifdef _OPENMP
  if (m_active) omp_set_num_threads(m_previous_threads);
#endif
}

}  // namespace vector_index
