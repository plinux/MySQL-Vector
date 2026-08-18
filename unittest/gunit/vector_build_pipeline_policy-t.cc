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
using vector_index::diskann_segment_profile;

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

TEST(VectorBuildPipelinePolicyTest, DefaultSegmentRowLimitPrefersByteTarget) {
  build_pipeline_thresholds thresholds;

  const uint64_t row_limit =
      vector_index::build_segment_row_limit(1024, thresholds);

  EXPECT_EQ(static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
            thresholds.segment_max_rows);
  EXPECT_EQ(2ULL * 1024ULL * 1024ULL * 1024ULL,
            thresholds.segment_target_size);
  EXPECT_EQ((2ULL * 1024ULL * 1024ULL * 1024ULL) /
                (1024ULL * sizeof(float) + sizeof(uint64_t)),
            row_limit);
}

TEST(VectorBuildPipelinePolicyTest, ManualDiskAnnSegmentProfileKeepsThresholds) {
  build_pipeline_thresholds thresholds;
  thresholds.segment_target_size = 768ULL * 1024ULL * 1024ULL;

  const auto result = vector_index::apply_diskann_segment_profile(
      1024, 2097152, 8589934592ULL, diskann_segment_profile::kManual,
      thresholds, true);

  EXPECT_FALSE(result.applied);
  EXPECT_STREQ("manual", result.reason);
  EXPECT_EQ(thresholds.segment_target_size,
            result.thresholds.segment_target_size);
  EXPECT_EQ(vector_index::build_segment_row_limit(1024, thresholds),
            vector_index::build_segment_row_limit(1024, result.thresholds));
}

TEST(VectorBuildPipelinePolicyTest, HighRecallDiskAnnSegmentProfileUses256M) {
  build_pipeline_thresholds thresholds;

  const auto result = vector_index::apply_diskann_segment_profile(
      1024, 2097152, 8589934592ULL, diskann_segment_profile::kHighRecall,
      thresholds, true);

  EXPECT_TRUE(result.applied);
  EXPECT_STREQ("high_recall_256m", result.reason);
  EXPECT_EQ(256ULL * 1024ULL * 1024ULL,
            result.thresholds.segment_target_size);
  EXPECT_EQ(65408ULL,
            vector_index::build_segment_row_limit(1024, result.thresholds));
}

TEST(VectorBuildPipelinePolicyTest, BalancedDiskAnnSegmentProfileUses512M) {
  build_pipeline_thresholds thresholds;

  const auto result = vector_index::apply_diskann_segment_profile(
      1024, 2097152, 8589934592ULL, diskann_segment_profile::kBalanced,
      thresholds, true);

  EXPECT_TRUE(result.applied);
  EXPECT_STREQ("balanced_512m", result.reason);
  EXPECT_EQ(512ULL * 1024ULL * 1024ULL,
            result.thresholds.segment_target_size);
  EXPECT_EQ(130816ULL,
            vector_index::build_segment_row_limit(1024, result.thresholds));
}

TEST(VectorBuildPipelinePolicyTest, ThroughputDiskAnnSegmentProfileUsesLargeTarget) {
  build_pipeline_thresholds thresholds;
  thresholds.segment_target_size = 256ULL * 1024ULL * 1024ULL;

  const auto result = vector_index::apply_diskann_segment_profile(
      1024, 2097152, 8589934592ULL, diskann_segment_profile::kThroughput,
      thresholds, true);

  EXPECT_TRUE(result.applied);
  EXPECT_STREQ("throughput_large_segment", result.reason);
  EXPECT_GE(result.thresholds.segment_target_size,
            vector_index::k_default_build_segment_target_size);
}

TEST(VectorBuildPipelinePolicyTest, DiskAnnSegmentProfileIgnoresIneligiblePath) {
  build_pipeline_thresholds thresholds;

  const auto result = vector_index::apply_diskann_segment_profile(
      1024, 2097152, 8589934592ULL, diskann_segment_profile::kHighRecall,
      thresholds, false);

  EXPECT_FALSE(result.applied);
  EXPECT_STREQ("not_applicable", result.reason);
  EXPECT_EQ(thresholds.segment_target_size,
            result.thresholds.segment_target_size);
}

TEST(VectorBuildPipelinePolicyTest, ForcedDirectSkipsDiskAnnSegmentProfile) {
  build_pipeline_thresholds thresholds;
  thresholds.mode = build_pipeline_mode::kDirect;

  const auto result = vector_index::apply_diskann_segment_profile(
      1024, 2097152, 8589934592ULL, diskann_segment_profile::kHighRecall,
      thresholds, true);

  EXPECT_FALSE(result.applied);
  EXPECT_STREQ("forced_direct", result.reason);
  EXPECT_EQ(thresholds.segment_target_size,
            result.thresholds.segment_target_size);
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

TEST(VectorBuildPipelinePolicyTest,
     HighRecallSmallDimensionUsesBalancedSegmentSize) {
  build_pipeline_thresholds thresholds;
  const auto result = vector_index::apply_diskann_segment_profile(
      512, 1000, 1000, diskann_segment_profile::kHighRecall, thresholds, true);
  EXPECT_TRUE(result.applied);
  EXPECT_STREQ("high_recall_512m", result.reason);
  EXPECT_EQ(512ULL * 1024ULL * 1024ULL,
            result.thresholds.segment_target_size);
}

TEST(VectorBuildPipelinePolicyTest,
     ThroughputProfileReportsNoChangeAtDefaultTarget) {
  build_pipeline_thresholds thresholds;
  thresholds.segment_target_size =
      vector_index::k_default_build_segment_target_size;
  const auto result = vector_index::apply_diskann_segment_profile(
      128, 1000, 1000, diskann_segment_profile::kThroughput, thresholds, true);
  EXPECT_FALSE(result.applied);
  EXPECT_STREQ("throughput_large_segment", result.reason);
}

TEST(VectorBuildPipelinePolicyTest, UnknownProfileLeavesManualThresholds) {
  build_pipeline_thresholds thresholds;
  const auto result = vector_index::apply_diskann_segment_profile(
      128, 1000, 1000, static_cast<diskann_segment_profile>(999), thresholds,
      true);
  EXPECT_FALSE(result.applied);
  EXPECT_EQ(thresholds.segment_target_size,
            result.thresholds.segment_target_size);
  EXPECT_STREQ("manual", result.reason);
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
  EXPECT_STREQ("high_recall", vector_index::diskann_segment_profile_name(
                                  diskann_segment_profile::kHighRecall));
  EXPECT_STREQ("manual", vector_index::diskann_segment_profile_name(
                             diskann_segment_profile::kManual));
  EXPECT_STREQ("throughput", vector_index::diskann_segment_profile_name(
                                 diskann_segment_profile::kThroughput));
  EXPECT_STREQ("balanced", vector_index::diskann_segment_profile_name(
                               diskann_segment_profile::kBalanced));
  EXPECT_STREQ("unknown", vector_index::diskann_segment_profile_name(
                              static_cast<diskann_segment_profile>(999)));
}

}  // namespace vector_build_pipeline_policy_unittest
