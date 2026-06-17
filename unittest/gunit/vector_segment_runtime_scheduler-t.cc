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

TEST(VectorSegmentRuntimeSchedulerTest, CapsBuildThreadsByCpuBudget) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 5;
  input.requested_task_count = 5;
  input.cpu_budget = 32;
  input.requested_build_threads = 32;
  input.requested_blas_threads = 1;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(5U, plan.task_count);
  EXPECT_LE(plan.effective_build_threads * plan.task_count, 32U);
  EXPECT_GE(plan.effective_build_threads, 1U);
  EXPECT_EQ(1U, plan.effective_blas_threads);
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

TEST(VectorSegmentRuntimeSchedulerTest, RejectsZeroSegments) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 0;

  vector_index::segment_scheduler_plan plan;
  EXPECT_FALSE(vector_index::make_segment_scheduler_plan(input, &plan));
}

TEST(VectorSegmentRuntimeSchedulerTest, CapsBlasThreadsByEffectiveBuildThreads) {
  vector_index::segment_scheduler_input input;
  input.segment_count = 4;
  input.requested_task_count = 4;
  input.cpu_budget = 16;
  input.requested_build_threads = 8;
  input.requested_blas_threads = 8;

  vector_index::segment_scheduler_plan plan;
  ASSERT_TRUE(vector_index::make_segment_scheduler_plan(input, &plan));

  EXPECT_EQ(4U, plan.task_count);
  EXPECT_EQ(4U, plan.effective_build_threads);
  EXPECT_EQ(4U, plan.effective_blas_threads);
}

}  // namespace vector_segment_runtime_scheduler_unittest
