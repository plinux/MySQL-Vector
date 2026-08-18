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

#include <gtest/gtest.h>

#include "sql/vector/vector_segment_runtime_scheduler.h"

namespace vector_segment_runtime_scheduler_unittest {

TEST(VectorSegmentRuntimeSchedulerTest, ExplicitBuildThreadsShareCpuBudget) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 5;
  input.requested_task_count = 5;
  input.cpu_budget = 32;
  input.requested_build_threads = 32;
  input.requested_blas_threads = 1;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(5U, plan.task_count);
  EXPECT_EQ(6U, plan.effective_build_threads);
  EXPECT_EQ(1U, plan.effective_blas_threads);
}

TEST(VectorSegmentRuntimeSchedulerTest,
     ExplicitBuildThreadsCannotOversubscribeCpuBudget) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 5;
  input.requested_task_count = 5;
  input.cpu_budget = 32;
  input.requested_build_threads = 8;
  input.requested_blas_threads = 1;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(5U, plan.task_count);
  EXPECT_EQ(6U, plan.effective_build_threads);
  EXPECT_EQ(1U, plan.effective_blas_threads);
}

TEST(VectorSegmentRuntimeSchedulerTest, ExplicitBuildThreadsCanStayBelowBudget) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 4;
  input.requested_task_count = 4;
  input.cpu_budget = 32;
  input.requested_build_threads = 4;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(4U, plan.task_count);
  EXPECT_EQ(4U, plan.effective_build_threads);
}

TEST(VectorSegmentRuntimeSchedulerTest, AutoBuildThreadsSharesCpuBudget) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 4;
  input.requested_task_count = 4;
  input.cpu_budget = 32;
  input.requested_build_threads = 0;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(4U, plan.task_count);
  EXPECT_EQ(8U, plan.effective_build_threads);
}

TEST(VectorSegmentRuntimeSchedulerTest, CpuBudgetCapsParallelTasks) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 8;
  input.requested_task_count = 8;
  input.cpu_budget = 3;
  input.requested_build_threads = 8;
  input.requested_blas_threads = 8;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(3U, plan.task_count);
  EXPECT_EQ(1U, plan.effective_build_threads);
  EXPECT_EQ(1U, plan.effective_blas_threads);
  EXPECT_STREQ("cpu_budget", plan.segment_parallel_reason);
}

TEST(VectorSegmentRuntimeSchedulerTest, MemoryBudgetKeepsRequestedTasks) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 8;
  input.requested_task_count = 4;
  input.cpu_budget = 32;
  input.requested_build_threads = 0;
  input.build_memory_budget = 64ULL * 1024ULL * 1024ULL * 1024ULL;
  input.per_segment_memory_estimate = 8ULL * 1024ULL * 1024ULL * 1024ULL;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(4U, plan.task_count);
  EXPECT_EQ(8U, plan.effective_build_threads);
  EXPECT_EQ(input.build_memory_budget, plan.segment_memory_budget);
  EXPECT_EQ(input.per_segment_memory_estimate, plan.segment_memory_estimate);
  EXPECT_STREQ("within_budget", plan.segment_parallel_reason);
}

TEST(VectorSegmentRuntimeSchedulerTest, MemoryBudgetReducesTasks) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 8;
  input.requested_task_count = 4;
  input.cpu_budget = 32;
  input.requested_build_threads = 0;
  input.build_memory_budget = 16ULL * 1024ULL * 1024ULL * 1024ULL;
  input.per_segment_memory_estimate = 8ULL * 1024ULL * 1024ULL * 1024ULL;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(2U, plan.task_count);
  EXPECT_EQ(16U, plan.effective_build_threads);
  EXPECT_STREQ("memory_budget", plan.segment_parallel_reason);
}

TEST(VectorSegmentRuntimeSchedulerTest, MemoryBudgetAllowsOneTaskMinimum) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 8;
  input.requested_task_count = 4;
  input.cpu_budget = 32;
  input.requested_build_threads = 0;
  input.build_memory_budget = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  input.per_segment_memory_estimate = 8ULL * 1024ULL * 1024ULL * 1024ULL;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(1U, plan.task_count);
  EXPECT_EQ(32U, plan.effective_build_threads);
  EXPECT_STREQ("memory_budget", plan.segment_parallel_reason);
}

TEST(VectorSegmentRuntimeSchedulerTest, RequestedSerialIgnoresLargerBudget) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 8;
  input.requested_task_count = 1;
  input.cpu_budget = 32;
  input.requested_build_threads = 0;
  input.build_memory_budget = 64ULL * 1024ULL * 1024ULL * 1024ULL;
  input.per_segment_memory_estimate = 8ULL * 1024ULL * 1024ULL * 1024ULL;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(1U, plan.task_count);
  EXPECT_EQ(32U, plan.effective_build_threads);
  EXPECT_STREQ("within_budget", plan.segment_parallel_reason);
}

TEST(VectorSegmentRuntimeSchedulerTest, ExpandsSearchCandidatesPerSegment) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 5;
  input.top_k = 10;
  input.search_candidate_multiplier = 2;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(20U, plan.candidates_per_segment);
}

TEST(VectorSegmentRuntimeSchedulerTest, SaturatesSearchCandidatesOnOverflow) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 5;
  input.top_k = 0x80000000U;
  input.search_candidate_multiplier = 2;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(0xffffffffU, plan.candidates_per_segment);
}

TEST(VectorSegmentRuntimeSchedulerTest, RejectsZeroSegments) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 0;

  vector_index::segment_scheduler_plan plan;
  EXPECT_FALSE(vector_index::make_segment_scheduler_plan(input, &plan));
}

TEST(VectorSegmentRuntimeSchedulerTest, RejectsNullPlan) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 1;
  EXPECT_FALSE(vector_index::make_segment_scheduler_plan(input, nullptr));
}

TEST(VectorSegmentRuntimeSchedulerTest, ZeroDefaultsStayBounded) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 2;
  input.cpu_budget = 2;
  input.requested_blas_threads = 0;
  input.search_candidate_multiplier = 0;
  input.top_k = 0;
  input.build_memory_budget = 1024;
  input.per_segment_memory_estimate = 0;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));
  EXPECT_EQ(1U, plan.effective_blas_threads);
  EXPECT_EQ(0U, plan.candidates_per_segment);
  EXPECT_STREQ("unbounded", plan.segment_parallel_reason);
}

TEST(VectorSegmentRuntimeSchedulerTest, ZeroMemoryBudgetLeavesTasksUnbounded) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 4;
  input.requested_task_count = 3;
  input.cpu_budget = 12;
  input.build_memory_budget = 0;
  input.per_segment_memory_estimate = 1024;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));
  EXPECT_EQ(3U, plan.task_count);
  EXPECT_STREQ("unbounded", plan.segment_parallel_reason);
}

TEST(VectorSegmentRuntimeSchedulerTest, AutoBuildThreadsCapsBlasThreads) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 4;
  input.requested_task_count = 4;
  input.cpu_budget = 16;
  input.requested_build_threads = 0;
  input.requested_blas_threads = 8;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(4U, plan.task_count);
  EXPECT_EQ(4U, plan.effective_build_threads);
  EXPECT_EQ(4U, plan.effective_blas_threads);
}

TEST(VectorSegmentRuntimeSchedulerTest, SingleIndexBuildKeepsFullBuildThreads) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 5;
  input.requested_task_count = 5;
  input.cpu_budget = 32;
  input.requested_build_threads = 32;
  input.requested_blas_threads = 4;
  input.single_index_build = true;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_TRUE(plan.single_index_build);
  EXPECT_EQ(1U, plan.task_count);
  EXPECT_EQ(32U, plan.effective_build_threads);
  EXPECT_EQ(4U, plan.effective_blas_threads);
  EXPECT_EQ(5U, plan.raw_reader_threads);
  EXPECT_EQ(32U, plan.pq_train_threads);
  EXPECT_EQ(32U, plan.pq_compress_threads);
}

TEST(VectorSegmentRuntimeSchedulerTest, SingleIndexAutoThreadsUseCpuBudget) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 3;
  input.requested_task_count = 8;
  input.cpu_budget = 16;
  input.requested_build_threads = 0;
  input.single_index_build = true;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_TRUE(plan.single_index_build);
  EXPECT_EQ(1U, plan.task_count);
  EXPECT_EQ(16U, plan.effective_build_threads);
  EXPECT_EQ(3U, plan.raw_reader_threads);
  EXPECT_EQ(16U, plan.pq_train_threads);
  EXPECT_EQ(16U, plan.pq_compress_threads);
}

}  // namespace vector_segment_runtime_scheduler_unittest
