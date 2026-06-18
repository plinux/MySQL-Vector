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

#include <limits>

#include "sql/vector/vector_diskann_scheduler.h"

namespace vector_diskann_scheduler_unittest {

constexpr uint64_t k_small_budget_rows = 4096;

TEST(VectorDiskAnnSchedulerTest, DerivesPerSegmentPqBudget) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 128;
  input.row_count = 262144;
  input.payload_size = 262144ULL * 128ULL * sizeof(float);
  input.pq_code_budget_size = 0;
  input.pq_code_budget_ratio = 0.125;
  input.disk_pq_dims = 0;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));

  EXPECT_EQ(64U, budget.pq_chunks);
  EXPECT_GT(budget.pq_code_budget_size, 0U);
  EXPECT_GT(budget.pq_code_budget_gb, 0.0);
}

TEST(VectorDiskAnnSchedulerTest, DerivesEffectiveDiskPqDimsForLargeNodes) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 1024;
  input.row_count = 16777216;
  input.payload_size = input.row_count * input.dimension * sizeof(float);
  input.pq_code_budget_size = 0;
  input.pq_code_budget_ratio = 0.125;
  input.disk_pq_dims = 0;
  input.max_degree = 56;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));

  EXPECT_EQ(512U, budget.disk_pq_dims);
  EXPECT_EQ(512U, budget.pq_chunks);
}

TEST(VectorDiskAnnSchedulerTest, ExplicitDiskPqDimsWinsOverRatio) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 128;
  input.row_count = 262144;
  input.payload_size = 262144ULL * 128ULL * sizeof(float);
  input.pq_code_budget_size = 0;
  input.pq_code_budget_ratio = 0.125;
  input.disk_pq_dims = 32;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));

  EXPECT_EQ(32U, budget.pq_chunks);
}

TEST(VectorDiskAnnSchedulerTest, DerivesCacheNodesFromSegmentRows) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 128;
  input.row_count = 262144;
  input.payload_size = 262144ULL * 128ULL * sizeof(float);
  input.max_degree = 56;
  input.search_cache_ratio = 0.2;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));

  EXPECT_EQ(30229U, budget.cache_nodes);
}

TEST(VectorDiskAnnSchedulerTest, ExplicitCacheSizeWinsOverRatio) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 128;
  input.row_count = 262144;
  input.payload_size = 262144ULL * 128ULL * sizeof(float);
  input.max_degree = 56;
  input.search_cache_size = 1024ULL * 1024ULL;
  input.search_cache_ratio = 0.2;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));

  EXPECT_EQ(1180U, budget.cache_nodes);
}

TEST(VectorDiskAnnSchedulerTest, DiskPqDoesNotShrinkRuntimeCacheFootprint) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 512;
  input.row_count = 1044495;
  input.payload_size = input.row_count * input.dimension * sizeof(float);
  input.disk_pq_dims = 128;
  input.max_degree = 56;
  input.search_cache_ratio = 0.25;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));

  EXPECT_EQ(128U, budget.disk_pq_dims);
  EXPECT_EQ(195804U, budget.cache_nodes);
}

TEST(VectorDiskAnnSchedulerTest, RuntimeCacheFootprintAlignsFloatCoordinates) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 129;
  input.row_count = k_small_budget_rows;
  input.payload_size = input.row_count * input.dimension * sizeof(float);
  input.max_degree = 56;
  input.search_cache_size = 1024ULL * 1024ULL;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));

  EXPECT_EQ(1131U, budget.cache_nodes);
}

TEST(VectorDiskAnnSchedulerTest, UsesConfiguredBuildMemoryBudget) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 128;
  input.row_count = 262144;
  input.payload_size = 262144ULL * 128ULL * sizeof(float);
  input.build_memory_size = 16ULL * 1024ULL * 1024ULL * 1024ULL;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));

  EXPECT_DOUBLE_EQ(16.0, budget.build_memory_gb);
}

TEST(VectorDiskAnnSchedulerTest, AutoBuildMemoryUsesAvailableMemoryBudget) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 128;
  input.row_count = 262144;
  input.payload_size = 262144ULL * 128ULL * sizeof(float);
  input.build_memory_size = 0;
  input.available_build_memory_size = 24ULL * 1024ULL * 1024ULL * 1024ULL;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));

  EXPECT_DOUBLE_EQ(24.0, budget.build_memory_gb);
}

TEST(VectorDiskAnnSchedulerTest, AutoBuildMemoryFallsBackToPayloadBudget) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 128;
  input.row_count = 262144;
  input.payload_size = 262144ULL * 128ULL * sizeof(float);
  input.build_memory_size = 0;
  input.available_build_memory_size = 0;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));

  EXPECT_DOUBLE_EQ(0.125, budget.build_memory_gb);
}

TEST(VectorDiskAnnSchedulerTest, CacheSizeRoundsUpToOneNode) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 128;
  input.row_count = 16;
  input.payload_size = 16ULL * 128ULL * sizeof(float);
  input.search_cache_size = 1;
  input.search_cache_ratio = 0.0;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));

  EXPECT_EQ(0U, budget.cache_nodes);
}

TEST(VectorDiskAnnSchedulerTest, PqBudgetUsesDiskAnnFloorSemantics) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 4;
  input.row_count = 2;
  input.payload_size = 2ULL * 4ULL * sizeof(float);
  input.pq_code_budget_size = 3;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));

  EXPECT_EQ(1U, budget.pq_chunks);
}

TEST(VectorDiskAnnSchedulerTest, ZeroRatiosKeepMinimumPqChunkWithoutCache) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 4;
  input.row_count = 2;
  input.payload_size = 2ULL * 4ULL * sizeof(float);
  input.pq_code_budget_size = 0;
  input.pq_code_budget_ratio = 0.0;
  input.search_cache_size = 0;
  input.search_cache_ratio = 0.0;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));
  EXPECT_EQ(1U, budget.pq_chunks);
  EXPECT_EQ(0U, budget.pq_code_budget_size);
  EXPECT_EQ(0U, budget.cache_nodes);
}

TEST(VectorDiskAnnSchedulerTest, ExplicitDiskPqDimsClampToDimension) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 4;
  input.row_count = 2;
  input.payload_size = 2ULL * 4ULL * sizeof(float);
  input.disk_pq_dims = 999;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));
  EXPECT_EQ(4U, budget.pq_chunks);
}

TEST(VectorDiskAnnSchedulerTest, RejectsMissingOutputAndZeroInputs) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 128;
  input.row_count = 262144;
  input.payload_size = 262144ULL * 128ULL * sizeof(float);

  vector_index::diskann_segment_budget budget;
  EXPECT_FALSE(vector_index::make_diskann_segment_budget(input, nullptr));

  auto invalid = input;
  invalid.dimension = 0;
  EXPECT_FALSE(vector_index::make_diskann_segment_budget(invalid, &budget));

  invalid = input;
  invalid.row_count = 0;
  EXPECT_FALSE(vector_index::make_diskann_segment_budget(invalid, &budget));

  invalid = input;
  invalid.payload_size = 0;
  EXPECT_FALSE(vector_index::make_diskann_segment_budget(invalid, &budget));
}

TEST(VectorDiskAnnSchedulerTest, ClampsLargeCacheNodeCount) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 1;
  input.row_count = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) +
                    1000ULL;
  input.payload_size = input.row_count * sizeof(float);
  input.pq_code_budget_size = input.row_count;
  input.disk_pq_dims = 1;
  input.max_degree = 0;
  input.search_cache_size = std::numeric_limits<uint64_t>::max();

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));
  EXPECT_EQ(std::numeric_limits<uint32_t>::max(), budget.cache_nodes);
}

TEST(VectorDiskAnnSchedulerTest, SaturatesHugePqBudgetToDimension) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 8;
  input.row_count = 1;
  input.payload_size = std::numeric_limits<uint64_t>::max();
  input.pq_code_budget_ratio = 1.0;

  vector_index::diskann_segment_budget budget;
  ASSERT_TRUE(vector_index::make_diskann_segment_budget(input, &budget));
  EXPECT_EQ(8U, budget.pq_chunks);
  EXPECT_EQ(std::numeric_limits<uint64_t>::max(),
            budget.pq_code_budget_size);
}

TEST(VectorDiskAnnSchedulerTest, RejectsInvalidRatios) {
  vector_index::diskann_segment_budget_input input;
  input.dimension = 128;
  input.row_count = 262144;
  input.payload_size = 262144ULL * 128ULL * sizeof(float);

  vector_index::diskann_segment_budget budget;
  input.pq_code_budget_ratio = -0.1;
  EXPECT_FALSE(vector_index::make_diskann_segment_budget(input, &budget));

  input.pq_code_budget_ratio = 1.1;
  EXPECT_FALSE(vector_index::make_diskann_segment_budget(input, &budget));

  input.pq_code_budget_ratio = 0.125;
  input.search_cache_ratio = 1.1;
  EXPECT_FALSE(vector_index::make_diskann_segment_budget(input, &budget));

  input.search_cache_ratio = -0.1;
  EXPECT_FALSE(vector_index::make_diskann_segment_budget(input, &budget));

  input.search_cache_ratio = 0.0;
  input.pq_code_budget_ratio = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(vector_index::make_diskann_segment_budget(input, &budget));

  input.pq_code_budget_ratio = 0.125;
  input.search_cache_ratio = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(vector_index::make_diskann_segment_budget(input, &budget));
}

}  // namespace vector_diskann_scheduler_unittest
