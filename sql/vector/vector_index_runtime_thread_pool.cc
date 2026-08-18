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
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "my_dbug.h"
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

size_t divide_round_up(size_t value, size_t divisor) {
  return value / divisor + static_cast<size_t>(value % divisor != 0);
}

bool contains_current_thread(const std::vector<std::thread> &workers) {
  const std::thread::id caller_id = std::this_thread::get_id();
  return std::any_of(workers.begin(), workers.end(),
                     [&](const std::thread &worker) {
                       return worker.joinable() &&
                              worker.get_id() == caller_id;
                     });
}

/**
  Join caller-owned worker threads after their work has drained.

  Pool shutdown and scoped execution invoke this helper only from the owning
  caller, never from one of the workers being joined. A self-join therefore
  violates the ownership contract and must fail immediately. The function is
  noexcept so any other impossible std::thread::join() failure also terminates
  instead of leaving a joinable thread to fail later during destruction.
*/
void join_worker_threads(std::vector<std::thread> &workers) noexcept {
  const std::thread::id caller_id = std::this_thread::get_id();
  for (std::thread &worker : workers) {
    if (!worker.joinable()) continue;
    if (worker.get_id() == caller_id) std::terminate();
    worker.join();
  }
}

class runtime_worker_pool {
 public:
  ~runtime_worker_pool() { shutdown(); }

  bool run(size_t item_count, size_t thread_count,
           const range_visitor &visitor) {
    if (thread_count == 0) return true;
    if (thread_count == 1)
      return invoke_range_visitor(visitor, 0, item_count, 0);

    std::shared_ptr<runtime_job> job;
    const size_t chunk_size = divide_round_up(item_count, thread_count);
    try {
      DBUG_EXECUTE_IF("vector_runtime_worker_pool_fail_state_allocation",
                      { throw std::bad_alloc(); });
      job = std::make_shared<runtime_job>(visitor);
      job->tasks.reserve(thread_count);
      for (size_t worker_id = 0; worker_id < thread_count; ++worker_id) {
        const size_t begin = worker_id * chunk_size;
        const size_t end = std::min(item_count, begin + chunk_size);
        if (begin >= end) break;
        DBUG_EXECUTE_IF("vector_runtime_worker_pool_fail_task_staging",
                        { throw std::bad_alloc(); });
        job->tasks.push_back(runtime_task{begin, end, worker_id});
      }
    } catch (...) {
      return false;
    }
    job->remaining.store(job->tasks.size(), std::memory_order_relaxed);

    {
      std::lock_guard<std::mutex> guard(m_mutex);
      if (!m_accepting || m_active_jobs >= k_max_build_threads ||
          !ensure_workers_locked(thread_count)) {
        return false;
      }
      try {
        m_jobs.push_back(job);
      } catch (...) {
        return false;
      }
      ++m_active_jobs;
    }
    m_worker_cv.notify_all();

    {
      std::unique_lock<std::mutex> done_guard(job->done_mutex);
      job->done_cv.wait(done_guard, [&]() {
        return job->remaining.load(std::memory_order_acquire) == 0;
      });
    }

    {
      std::lock_guard<std::mutex> guard(m_mutex);
      --m_active_jobs;
      if (m_active_jobs == 0 && m_jobs.empty() && m_active_tasks == 0) {
        m_idle_cv.notify_all();
      }
    }
    return !job->failed.load(std::memory_order_relaxed);
  }

  size_t size() const {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_workers.size();
  }

  void reset() { shutdown(); }

 private:
  struct runtime_task {
    size_t begin;
    size_t end;
    size_t worker_id;
  };

  struct runtime_job {
    explicit runtime_job(range_visitor visitor_arg)
        : visitor(std::move(visitor_arg)) {}

    std::vector<runtime_task> tasks;
    size_t next_task{0};
    std::atomic<size_t> remaining{0};
    std::atomic<bool> failed{false};
    range_visitor visitor;
    std::mutex done_mutex;
    std::condition_variable done_cv;
  };

  bool ensure_workers_locked(size_t thread_count) {
    try {
      while (m_workers.size() < thread_count) {
        m_workers.emplace_back([this]() { worker_loop(); });
      }
    } catch (...) {
      return false;
    }
    return true;
  }

  void worker_loop() {
    for (;;) {
      std::shared_ptr<runtime_job> job;
      runtime_task task{};
      {
        std::unique_lock<std::mutex> guard(m_mutex);
        m_worker_cv.wait(guard, [&]() { return m_stop || !m_jobs.empty(); });
        if (m_stop && m_jobs.empty()) return;
        auto job_it = m_jobs.begin();
        job = *job_it;
        task = job->tasks[job->next_task++];
        if (job->next_task < job->tasks.size()) {
          m_jobs.splice(m_jobs.end(), m_jobs, job_it);
        } else {
          m_jobs.erase(job_it);
        }
        ++m_active_tasks;
      }

      if (!job->failed.load(std::memory_order_relaxed) &&
          !invoke_range_visitor(job->visitor, task.begin, task.end,
                                task.worker_id)) {
        job->failed.store(true, std::memory_order_relaxed);
      }
      if (job->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> done_guard(job->done_mutex);
        job->done_cv.notify_one();
      }

      {
        std::lock_guard<std::mutex> guard(m_mutex);
        --m_active_tasks;
        if (m_active_jobs == 0 && m_jobs.empty() && m_active_tasks == 0) {
          m_idle_cv.notify_all();
        }
      }
      m_worker_cv.notify_one();
    }
  }

  void shutdown() {
    std::vector<std::thread> workers;
    {
      std::unique_lock<std::mutex> guard(m_mutex);
      if (contains_current_thread(m_workers)) std::terminate();
      m_accepting = false;
      m_idle_cv.wait(guard, [&]() {
        return m_active_jobs == 0 && m_jobs.empty() && m_active_tasks == 0;
      });
      m_stop = true;
      workers.swap(m_workers);
    }
    m_worker_cv.notify_all();
    join_worker_threads(workers);
    {
      std::lock_guard<std::mutex> guard(m_mutex);
      m_stop = false;
      m_accepting = true;
    }
  }

  mutable std::mutex m_mutex;
  std::condition_variable m_worker_cv;
  std::condition_variable m_idle_cv;
  std::list<std::shared_ptr<runtime_job>> m_jobs;
  std::vector<std::thread> m_workers;
  size_t m_active_jobs{0};
  size_t m_active_tasks{0};
  bool m_stop{false};
  bool m_accepting{true};
};

runtime_worker_pool &global_runtime_worker_pool() {
  static runtime_worker_pool pool;
  return pool;
}

class shared_search_worker_pool {
 public:
  ~shared_search_worker_pool() { shutdown(); }

  bool run(size_t item_count, size_t worker_budget,
           const search_item_visitor &visitor,
           shared_search_execution *execution) {
    if (!visitor) return false;
    if (item_count == 0) {
      if (execution != nullptr) *execution = {};
      return true;
    }

    std::shared_ptr<search_job> job;
    try {
      DBUG_EXECUTE_IF("vector_shared_search_fail_job_allocation",
                      { throw std::bad_alloc(); });
      job = std::make_shared<search_job>(item_count, worker_budget, visitor);
    } catch (...) {
      return false;
    }
    size_t workers_to_wake = 0;
    {
      std::lock_guard<std::mutex> guard(m_mutex);
      if (!m_accepting) return false;
      const size_t current_budget =
          m_active_budgets.empty() ? 0 : *m_active_budgets.rbegin();
      const size_t global_budget = std::max(current_budget, worker_budget);
      if (m_pending_items >
          std::numeric_limits<size_t>::max() - m_active_workers) {
        return false;
      }
      size_t outstanding_work = m_pending_items + m_active_workers;
      if (item_count >
          std::numeric_limits<size_t>::max() - outstanding_work) {
        return false;
      }
      outstanding_work += item_count;
      const size_t desired_workers = std::min(global_budget, outstanding_work);
      if (!ensure_workers_locked(desired_workers)) return false;
      auto budget_it = m_active_budgets.end();
      try {
        budget_it = m_active_budgets.insert(worker_budget);
        DBUG_EXECUTE_IF("vector_shared_search_fail_job_enqueue",
                        { throw std::bad_alloc(); });
        m_jobs.push_back(job);
      } catch (...) {
        if (budget_it != m_active_budgets.end()) {
          m_active_budgets.erase(budget_it);
        }
        return false;
      }
      m_worker_limit = *m_active_budgets.rbegin();
      m_pending_items += item_count;
      workers_to_wake = std::min(
          item_count, m_worker_limit > m_active_workers
                          ? m_worker_limit - m_active_workers
                          : 0);
      if (execution != nullptr) {
        execution->worker_budget = worker_budget;
        execution->active_requests = m_active_budgets.size();
        execution->effective_workers = std::min(item_count, worker_budget);
        execution->work_items = item_count;
      }
    }
    for (size_t i = 0; i < workers_to_wake; ++i) {
      m_worker_cv.notify_one();
    }

    {
      std::unique_lock<std::mutex> done_guard(job->done_mutex);
      job->done_cv.wait(done_guard, [&]() {
        return job->remaining.load(std::memory_order_acquire) == 0;
      });
    }

    {
      std::lock_guard<std::mutex> guard(m_mutex);
      const auto budget_it = m_active_budgets.find(worker_budget);
      if (budget_it != m_active_budgets.end()) {
        m_active_budgets.erase(budget_it);
      }
      m_worker_limit =
          m_active_budgets.empty() ? 0 : *m_active_budgets.rbegin();
      if (m_active_budgets.empty() && m_jobs.empty() &&
          m_active_workers == 0) {
        m_idle_cv.notify_all();
      }
    }
    return !job->failed.load(std::memory_order_relaxed);
  }

  size_t size() const {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_workers.size();
  }

  void reset() { shutdown(); }

 private:
  struct search_job {
    search_job(size_t item_count_arg, size_t worker_budget_arg,
               search_item_visitor visitor_arg)
        : item_count(item_count_arg),
          worker_budget(worker_budget_arg),
          remaining(item_count_arg),
          visitor(std::move(visitor_arg)) {}

    size_t item_count;
    size_t worker_budget;
    size_t active_workers{0};
    std::atomic<size_t> next_item{0};
    std::atomic<size_t> remaining;
    std::atomic<bool> failed{false};
    search_item_visitor visitor;
    std::mutex done_mutex;
    std::condition_variable done_cv;
  };

  bool ensure_workers_locked(size_t worker_budget) {
    try {
      while (m_workers.size() < worker_budget) {
        const size_t worker_id = m_workers.size();
        m_workers.emplace_back(
            [this, worker_id]() { worker_loop(worker_id); });
      }
    } catch (...) {
      return false;
    }
    return true;
  }

  bool has_runnable_job_locked() const {
    return std::any_of(m_jobs.begin(), m_jobs.end(), [](const auto &job) {
      return job->active_workers < job->worker_budget;
    });
  }

  void worker_loop(size_t worker_id) {
    for (;;) {
      std::shared_ptr<search_job> job;
      size_t item_index = 0;
      {
        std::unique_lock<std::mutex> guard(m_mutex);
        m_worker_cv.wait(guard, [&]() {
          return m_stop ||
                 (m_active_workers < m_worker_limit &&
                  has_runnable_job_locked());
        });
        if (m_stop && m_jobs.empty()) return;

        const auto job_it = std::find_if(
            m_jobs.begin(), m_jobs.end(), [](const auto &candidate) {
              return candidate->active_workers < candidate->worker_budget;
            });
        if (job_it == m_jobs.end()) continue;
        job = *job_it;
        m_jobs.erase(job_it);
        item_index = job->next_item.fetch_add(1, std::memory_order_relaxed);
        if (item_index + 1 < job->item_count) m_jobs.push_back(job);
        --m_pending_items;
        ++job->active_workers;
        ++m_active_workers;
      }

      bool item_ok = true;
      if (!job->failed.load(std::memory_order_relaxed)) {
        try {
          item_ok = job->visitor(item_index, worker_id);
        } catch (...) {
          item_ok = false;
        }
      }
      if (!item_ok) job->failed.store(true, std::memory_order_relaxed);

      bool notify_done = false;
      {
        std::lock_guard<std::mutex> guard(m_mutex);
        --job->active_workers;
        --m_active_workers;
        notify_done =
            job->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1;
        if (m_active_budgets.empty() && m_jobs.empty() &&
            m_active_workers == 0) {
          m_idle_cv.notify_all();
        }
      }
      if (notify_done) {
        std::lock_guard<std::mutex> done_guard(job->done_mutex);
        job->done_cv.notify_one();
      }
      m_worker_cv.notify_one();
    }
  }

  void shutdown() {
    std::vector<std::thread> workers;
    {
      std::unique_lock<std::mutex> guard(m_mutex);
      if (contains_current_thread(m_workers)) std::terminate();
      m_accepting = false;
      m_idle_cv.wait(guard, [&]() {
        return m_active_budgets.empty() && m_jobs.empty() &&
               m_active_workers == 0;
      });
      m_stop = true;
      workers.swap(m_workers);
    }
    m_worker_cv.notify_all();
    join_worker_threads(workers);
    {
      std::lock_guard<std::mutex> guard(m_mutex);
      m_stop = false;
      m_accepting = true;
      m_worker_limit = 0;
      m_pending_items = 0;
    }
  }

  mutable std::mutex m_mutex;
  std::condition_variable m_worker_cv;
  std::condition_variable m_idle_cv;
  std::deque<std::shared_ptr<search_job>> m_jobs;
  std::vector<std::thread> m_workers;
  std::multiset<size_t> m_active_budgets;
  size_t m_worker_limit{0};
  size_t m_pending_items{0};
  size_t m_active_workers{0};
  bool m_stop{false};
  bool m_accepting{true};
};

shared_search_worker_pool &global_shared_search_worker_pool() {
  static shared_search_worker_pool pool;
  return pool;
}

class scoped_thread_joiner {
 public:
  explicit scoped_thread_joiner(std::vector<std::thread> *workers)
      : m_workers(workers) {}

  ~scoped_thread_joiner() { join_all(); }

  void join_all() {
    if (m_joined) return;
    m_joined = true;
    join_worker_threads(*m_workers);
  }

 private:
  std::vector<std::thread> *m_workers;
  bool m_joined{false};
};

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
  return global_runtime_worker_pool().run(item_count, thread_count, visitor);
}

bool parallel_for_ranges_scoped(size_t item_count, size_t configured_threads,
                                const range_visitor &visitor) {
  if (!visitor) return false;

  const size_t thread_count =
      effective_runtime_worker_count(item_count, configured_threads);
  if (thread_count == 0) return true;
  if (thread_count == 1)
    return invoke_range_visitor(visitor, 0, item_count, 0);

  std::atomic<bool> failed{false};
  std::vector<unsigned char> worker_ok;
  std::vector<std::thread> workers;
  scoped_thread_joiner joiner(&workers);
  const size_t chunk_size = divide_round_up(item_count, thread_count);

  try {
    DBUG_EXECUTE_IF("vector_runtime_worker_pool_fail_state_allocation",
                    { throw std::bad_alloc(); });
    worker_ok.assign(thread_count, 1);
    workers.reserve(thread_count);
    for (size_t worker_id = 0; worker_id < thread_count; ++worker_id) {
      const size_t begin = worker_id * chunk_size;
      const size_t end = std::min(item_count, begin + chunk_size);
      if (begin >= end) break;
      workers.emplace_back([&, begin, end, worker_id]() {
        if (!failed.load(std::memory_order_relaxed)) {
          const bool ok =
              invoke_range_visitor(visitor, begin, end, worker_id);
          if (!ok) worker_ok[worker_id] = 0;
          if (!ok) failed.store(true, std::memory_order_relaxed);
        }
      });
    }
  } catch (...) {
    failed.store(true, std::memory_order_relaxed);
  }
  joiner.join_all();
  return !failed.load(std::memory_order_relaxed) &&
         std::all_of(worker_ok.begin(), worker_ok.end(),
                     [](unsigned char ok) { return ok != 0; });
}

bool parallel_for_queries(size_t query_count, size_t configured_threads,
                          const query_range_visitor &visitor) {
  return parallel_for_ranges(query_count, configured_threads, visitor);
}

bool parallel_for_shared_search_items(
    size_t item_count, size_t configured_threads,
    const search_item_visitor &visitor, shared_search_execution *execution) {
  if (!visitor) return false;
  if (item_count == 0) {
    if (execution != nullptr) *execution = {};
    return true;
  }
  const size_t worker_budget = effective_runtime_worker_count(
      k_max_build_threads, configured_threads);
  return global_shared_search_worker_pool().run(item_count, worker_budget,
                                                visitor, execution);
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
size_t runtime_worker_pool_size_for_testing() {
  return global_runtime_worker_pool().size();
}

void reset_runtime_worker_pool_for_testing() {
  global_runtime_worker_pool().reset();
}

size_t shared_search_worker_pool_size_for_testing() {
  return global_shared_search_worker_pool().size();
}

void reset_shared_search_worker_pool_for_testing() {
  global_shared_search_worker_pool().reset();
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

scoped_omp_threads::scoped_omp_threads(size_t thread_count) {
#ifdef _OPENMP
  if (thread_count == 0) return;

  const size_t limited =
      std::min<size_t>(thread_count,
                       static_cast<size_t>(std::numeric_limits<int>::max()));
  m_previous_threads = omp_get_max_threads();
  m_previous_dynamic = omp_get_dynamic();
#if _OPENMP >= 200805
  m_previous_max_active_levels = omp_get_max_active_levels();
#endif
  omp_set_dynamic(0);
#if _OPENMP >= 200805
  omp_set_max_active_levels(1);
#endif
  omp_set_num_threads(static_cast<int>(limited));
  m_active = true;
#else
  (void)thread_count;
#endif
}

scoped_omp_threads::~scoped_omp_threads() {
#ifdef _OPENMP
  if (m_active) {
    omp_set_num_threads(m_previous_threads);
#if _OPENMP >= 200805
    omp_set_max_active_levels(m_previous_max_active_levels);
#endif
    omp_set_dynamic(m_previous_dynamic);
  }
#endif
}

}  // namespace vector_index
