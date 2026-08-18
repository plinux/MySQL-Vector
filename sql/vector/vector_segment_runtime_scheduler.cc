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

#include "sql/vector/vector_segment_runtime_scheduler.h"

#include <algorithm>
#include <limits>
#include <thread>

namespace vector_index {

namespace {

uint32_t auto_cpu_budget() {
  const auto concurrency = std::thread::hardware_concurrency();
  return concurrency == 0 ? 1U : concurrency;
}

uint32_t at_least_one(uint32_t value) { return value == 0 ? 1U : value; }

uint32_t auto_build_threads(uint32_t cpu_budget, uint32_t task_count) {
  const uint32_t budget_per_task = std::max(1U, cpu_budget / task_count);
  return std::max(1U, budget_per_task);
}

uint32_t memory_limited_task_count(uint32_t requested_tasks,
                                   uint32_t segment_count,
                                   uint64_t build_memory_budget,
                                   uint64_t per_segment_memory_estimate) {
  if (build_memory_budget == 0 || per_segment_memory_estimate == 0) {
    return requested_tasks;
  }
  const uint64_t memory_tasks =
      std::max<uint64_t>(1, build_memory_budget / per_segment_memory_estimate);
  const uint32_t capped_tasks =
      static_cast<uint32_t>(std::min<uint64_t>(segment_count, memory_tasks));
  return std::min(requested_tasks, capped_tasks);
}

uint32_t saturated_multiply(uint32_t lhs, uint32_t rhs) {
  const uint64_t product = static_cast<uint64_t>(lhs) * rhs;
  return product > std::numeric_limits<uint32_t>::max()
             ? std::numeric_limits<uint32_t>::max()
             : static_cast<uint32_t>(product);
}

}  // namespace

bool make_segment_scheduler_plan(const segment_scheduler_input &input,
                                 segment_scheduler_plan *plan) {
  if (plan == nullptr || input.segment_count == 0) return false;

  const uint32_t cpu_budget =
      input.cpu_budget == 0 ? auto_cpu_budget() : input.cpu_budget;

  const uint32_t requested_tasks = input.requested_task_count == 0
                                       ? input.segment_count
                                       : input.requested_task_count;
  const uint32_t requested_task_count =
      std::max(1U, std::min(input.segment_count, requested_tasks));
  const uint32_t cpu_limited_task_count =
      std::min(requested_task_count, cpu_budget);
  const uint32_t memory_limited_tasks = memory_limited_task_count(
      cpu_limited_task_count, input.segment_count, input.build_memory_budget,
      input.per_segment_memory_estimate);
  const uint32_t task_count =
      input.single_index_build ? 1U : memory_limited_tasks;
  const char *segment_parallel_reason = "unbounded";
  if (input.single_index_build) {
    segment_parallel_reason = "single_index_build";
  } else if (memory_limited_tasks < cpu_limited_task_count) {
    segment_parallel_reason = "memory_budget";
  } else if (cpu_limited_task_count < requested_task_count) {
    segment_parallel_reason = "cpu_budget";
  } else if (input.build_memory_budget != 0 &&
             input.per_segment_memory_estimate != 0) {
    segment_parallel_reason = "within_budget";
  }
  const uint32_t requested_build_threads =
      input.requested_build_threads == 0
          ? auto_build_threads(cpu_budget, task_count)
          : input.requested_build_threads;
  const uint32_t budget_per_task = std::max(1U, cpu_budget / task_count);
  const uint32_t effective_build_threads =
      std::max(1U, std::min(requested_build_threads, budget_per_task));
  const uint32_t requested_blas_threads =
      at_least_one(input.requested_blas_threads);
  const uint32_t effective_blas_threads =
      std::max(1U, std::min(requested_blas_threads, effective_build_threads));

  const uint32_t multiplier = at_least_one(input.search_candidate_multiplier);
  const uint32_t candidates =
      input.top_k == 0 ? 0 : saturated_multiply(input.top_k, multiplier);

  plan->task_count = task_count;
  plan->cpu_budget = cpu_budget;
  plan->effective_build_threads = effective_build_threads;
  plan->effective_blas_threads = effective_blas_threads;
  plan->raw_reader_threads =
      input.single_index_build
          ? std::max(1U, std::min(input.segment_count, effective_build_threads))
          : task_count;
  plan->pq_train_threads = effective_build_threads;
  plan->pq_compress_threads = effective_build_threads;
  plan->candidates_per_segment = std::max(input.top_k, candidates);
  plan->segment_memory_budget = input.build_memory_budget;
  plan->segment_memory_estimate = input.per_segment_memory_estimate;
  plan->segment_parallel_reason = segment_parallel_reason;
  plan->single_index_build = input.single_index_build;
  return true;
}

}  // namespace vector_index
