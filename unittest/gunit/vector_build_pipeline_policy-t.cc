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

#include "sql/vector/vector_build_pipeline_policy.h"

namespace vector_build_pipeline_policy_unittest {

using vector_index::build_input_stats;
using vector_index::build_pipeline_decision;
using vector_index::build_pipeline_mode;
using vector_index::build_pipeline_path;
using vector_index::build_pipeline_thresholds;
using vector_index::build_pipeline_trigger;

namespace {

build_pipeline_decision decide(const build_input_stats &input,
                               build_pipeline_thresholds thresholds = {}) {
  return vector_index::select_build_pipeline(input, thresholds);
}

void expect_decision(const build_pipeline_decision &decision,
                     build_pipeline_path path,
                     build_pipeline_trigger trigger) {
  EXPECT_EQ(path, decision.path);
  EXPECT_EQ(trigger, decision.trigger);
}

}  // namespace

TEST(VectorBuildPipelinePolicyTest, DefaultSmallInputUsesDirectPath) {
  const build_input_stats input{42, 1024, 1, true};

  expect_decision(decide(input), build_pipeline_path::kDirect,
                  build_pipeline_trigger::kBelowThreshold);
}

TEST(VectorBuildPipelinePolicyTest, ForcedDirectOverridesLargeInput) {
  const build_input_stats input{vector_index::k_default_build_pipeline_min_rows,
                                vector_index::k_default_build_pipeline_min_size,
                                2, true};
  build_pipeline_thresholds thresholds;
  thresholds.mode = build_pipeline_mode::kDirect;

  expect_decision(decide(input, thresholds), build_pipeline_path::kDirect,
                  build_pipeline_trigger::kForcedDirect);
}

TEST(VectorBuildPipelinePolicyTest, ForcedSegmentedOverridesSmallInput) {
  const build_input_stats input{1, 128, 0, false};
  build_pipeline_thresholds thresholds;
  thresholds.mode = build_pipeline_mode::kSegmented;

  expect_decision(decide(input, thresholds), build_pipeline_path::kSegmented,
                  build_pipeline_trigger::kForcedSegmented);
}

TEST(VectorBuildPipelinePolicyTest, MultipleRawSegmentsUseSegmentedPath) {
  const build_input_stats input{10, 1024, 2, true};

  expect_decision(decide(input), build_pipeline_path::kSegmented,
                  build_pipeline_trigger::kRawSegmentCount);
}

TEST(VectorBuildPipelinePolicyTest, SingleRawSegmentDoesNotForceSegmentedPath) {
  const build_input_stats input{10, 1024, 1, true};

  expect_decision(decide(input), build_pipeline_path::kDirect,
                  build_pipeline_trigger::kBelowThreshold);
}

TEST(VectorBuildPipelinePolicyTest, RowThresholdUsesSegmentedPath) {
  const build_input_stats input{vector_index::k_default_build_pipeline_min_rows,
                                1024, 1, true};

  expect_decision(decide(input), build_pipeline_path::kSegmented,
                  build_pipeline_trigger::kRowCount);
}

TEST(VectorBuildPipelinePolicyTest, PayloadThresholdUsesSegmentedPath) {
  const build_input_stats input{42, vector_index::k_default_build_pipeline_min_size,
                                1, true};

  expect_decision(decide(input), build_pipeline_path::kSegmented,
                  build_pipeline_trigger::kPayloadSize);
}

TEST(VectorBuildPipelinePolicyTest, ZeroThresholdsDisableAutoTriggers) {
  const build_input_stats input{vector_index::k_default_build_pipeline_min_rows,
                                vector_index::k_default_build_pipeline_min_size,
                                1, true};
  build_pipeline_thresholds thresholds;
  thresholds.min_rows = 0;
  thresholds.min_size = 0;

  expect_decision(decide(input, thresholds), build_pipeline_path::kDirect,
                  build_pipeline_trigger::kBelowThreshold);
}

TEST(VectorBuildPipelinePolicyTest, AutoTriggerPriorityIsStable) {
  const build_input_stats input{vector_index::k_default_build_pipeline_min_rows,
                                vector_index::k_default_build_pipeline_min_size,
                                2, true};

  expect_decision(decide(input), build_pipeline_path::kSegmented,
                  build_pipeline_trigger::kRawSegmentCount);
}

TEST(VectorBuildPipelinePolicyTest, DefaultSegmentRowLimitHonorsRowCap) {
  build_pipeline_thresholds thresholds;

  const uint64_t row_limit =
      vector_index::build_segment_row_limit(1024, thresholds);

  EXPECT_EQ(1048576ULL, thresholds.segment_max_rows);
  EXPECT_EQ(8ULL * 1024ULL * 1024ULL * 1024ULL,
            thresholds.segment_target_size);
  EXPECT_EQ(thresholds.segment_max_rows, row_limit);
}

TEST(VectorBuildPipelinePolicyTest, EstimatesMilvusSizedRawSegmentFanout) {
  build_pipeline_thresholds thresholds;
  thresholds.segment_max_rows = 262144;
  thresholds.segment_target_size = 128ULL * 1024ULL * 1024ULL;
  thresholds.max_tasks = 4;

  const uint64_t row_limit =
      vector_index::build_segment_row_limit(128, thresholds);
  EXPECT_LE(row_limit, thresholds.segment_max_rows);

  const build_input_stats input{
      1000000, 1000000ULL * 128ULL * sizeof(float), 4, true};
  expect_decision(decide(input, thresholds), build_pipeline_path::kSegmented,
                  build_pipeline_trigger::kRawSegmentCount);
}

TEST(VectorBuildPipelinePolicyTest, HandlesZeroRowsAndSaturatedDimensions) {
  build_pipeline_thresholds thresholds;
  thresholds.segment_max_rows = 0;
  thresholds.segment_target_size = 0;
  EXPECT_EQ(std::numeric_limits<uint64_t>::max(),
            vector_index::build_segment_row_limit(128, thresholds));

  thresholds.segment_target_size = 1024;
  EXPECT_EQ(1U, vector_index::build_segment_row_limit(
                    std::numeric_limits<uint64_t>::max(), thresholds));
}

TEST(VectorBuildPipelinePolicyTest, UnknownModeUsesAutomaticThresholds) {
  build_pipeline_thresholds thresholds;
  thresholds.mode = static_cast<build_pipeline_mode>(999);
  expect_decision(decide({1, 1, 0, false}, thresholds),
                  build_pipeline_path::kDirect,
                  build_pipeline_trigger::kBelowThreshold);
}

TEST(VectorBuildPipelinePolicyTest, NamesAreStableForStatusOutput) {
  EXPECT_STREQ("auto",
               vector_index::build_pipeline_mode_name(build_pipeline_mode::kAuto));
  EXPECT_STREQ("direct", vector_index::build_pipeline_path_name(
                             build_pipeline_path::kDirect));
  EXPECT_STREQ("segmented", vector_index::build_pipeline_path_name(
                                build_pipeline_path::kSegmented));
  EXPECT_STREQ("payload_size", vector_index::build_pipeline_trigger_name(
                                   build_pipeline_trigger::kPayloadSize));
  EXPECT_STREQ("unknown", vector_index::build_pipeline_mode_name(
                              static_cast<build_pipeline_mode>(999)));
  EXPECT_STREQ("unknown", vector_index::build_pipeline_path_name(
                              static_cast<build_pipeline_path>(999)));
  EXPECT_STREQ("unknown", vector_index::build_pipeline_trigger_name(
                              static_cast<build_pipeline_trigger>(999)));
}

}  // namespace vector_build_pipeline_policy_unittest
