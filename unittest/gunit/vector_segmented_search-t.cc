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
#include <utility>
#include <vector>

#include "sql/vector/vector_diskann_scheduler.h"
#include "sql/vector/vector_segmented_search.h"

namespace vector_index {
namespace {

TEST(VectorSegmentedSearchTest, MergeRejectsNullOutput) {
  EXPECT_FALSE(merge_segment_topk({}, 1, nullptr));
}

TEST(VectorSegmentedSearchTest, CandidateTopKExpandsBySegmentCount) {
  EXPECT_EQ(50U, segmented_search_candidate_top_k(10, 5, 1000, 1, 65536));
}

TEST(VectorSegmentedSearchTest, CandidateTopKRespectsSegmentEntryCount) {
  EXPECT_EQ(7U, segmented_search_candidate_top_k(10, 5, 7, 1, 65536));
}

TEST(VectorSegmentedSearchTest, CandidateTopKRespectsBatchBudgetWhenPossible) {
  EXPECT_EQ(20U, segmented_search_candidate_top_k(10, 5, 1000, 100, 10000));
}

TEST(VectorSegmentedSearchTest, CandidateTopKKeepsOldBehaviorWhenBudgetIsTight) {
  EXPECT_EQ(10U, segmented_search_candidate_top_k(10, 5, 1000, 100, 1000));
}

TEST(VectorSegmentedSearchTest, CandidateTopKHandlesEmptyInputs) {
  EXPECT_EQ(0U, segmented_search_candidate_top_k(0, 5, 1000, 1, 65536));
  EXPECT_EQ(0U, segmented_search_candidate_top_k(10, 0, 1000, 1, 65536));
  EXPECT_EQ(0U, segmented_search_candidate_top_k(10, 5, 0, 1, 65536));
}

TEST(VectorSegmentedSearchTest, DiskAnnSearchListSlackShrinksByFanout) {
  EXPECT_EQ(63U, diskann_search_list_slack_top_k(10, 64, 1));
  EXPECT_EQ(199U, diskann_search_list_slack_top_k(10, 800, 4));
  EXPECT_EQ(24U, diskann_search_list_slack_top_k(10, 800, 32));
  EXPECT_EQ(10U, diskann_search_list_slack_top_k(10, 16, 4));
  EXPECT_EQ(10U, diskann_search_list_slack_top_k(10, 0, 4));
  EXPECT_EQ(0U, diskann_search_list_slack_top_k(0, 64, 1));
}

TEST(VectorSegmentedSearchTest, DiskAnnBalancedSearchProfileCoversFewSegments) {
  diskann_search_budget budget;
  EXPECT_TRUE(choose_diskann_search_budget(
      10, 17, 1, 130816, 800, diskann_search_profile::kBalanced, &budget));

  EXPECT_EQ(2400U, budget.search_complexity);
  EXPECT_EQ(diskann_search_profile::kBalanced, budget.profile);
  EXPECT_EQ(diskann_search_profile_reason::kBalancedFewSegments,
            budget.reason);
  EXPECT_LT(budget.candidate_count, budget.search_complexity);
}

TEST(VectorSegmentedSearchTest, DiskAnnHighRecallSearchProfileCoversFewSegments) {
  diskann_search_budget budget;
  EXPECT_TRUE(choose_diskann_search_budget(
      10, 17, 1, 130816, 800, diskann_search_profile::kHighRecall, &budget));

  EXPECT_EQ(3200U, budget.search_complexity);
  EXPECT_EQ(diskann_search_profile::kHighRecall, budget.profile);
  EXPECT_EQ(diskann_search_profile_reason::kHighRecallFewSegments,
            budget.reason);
  EXPECT_LT(budget.candidate_count, budget.search_complexity);
}

TEST(VectorSegmentedSearchTest, DiskAnnHighRecallSearchProfileKeepsManySegments) {
  diskann_search_budget budget;
  EXPECT_TRUE(choose_diskann_search_budget(
      10, 33, 1, 65408, 800, diskann_search_profile::kHighRecall, &budget));

  EXPECT_EQ(1600U, budget.search_complexity);
  EXPECT_EQ(diskann_search_profile_reason::kHighRecallManySegments,
            budget.reason);
  EXPECT_LT(budget.candidate_count, budget.search_complexity);
}

TEST(VectorSegmentedSearchTest, DiskAnnManualSearchProfileKeepsExplicitValue) {
  diskann_search_budget budget;
  EXPECT_TRUE(choose_diskann_search_budget(
      10, 17, 1, 130816, 1234, diskann_search_profile::kManual, &budget));

  EXPECT_EQ(1234U, budget.search_complexity);
  EXPECT_EQ(diskann_search_profile::kManual, budget.profile);
  EXPECT_EQ(diskann_search_profile_reason::kManual, budget.reason);
}

TEST(VectorSegmentedSearchTest, DiskAnnSearchProfileDoesNotLowerExplicitValue) {
  diskann_search_budget budget;
  EXPECT_TRUE(choose_diskann_search_budget(
      10, 17, 1, 130816, 5000, diskann_search_profile::kBalanced, &budget));

  EXPECT_EQ(5000U, budget.search_complexity);
  EXPECT_EQ(diskann_search_profile_reason::kManualHigher, budget.reason);
}

TEST(VectorSegmentedSearchTest, DiskAnnFastAndBalancedManySegmentProfiles) {
  diskann_search_budget budget;
  ASSERT_TRUE(choose_diskann_search_budget(
      10, 4, 1, 1024, 64, diskann_search_profile::kFast, &budget));
  EXPECT_EQ(800U, budget.search_complexity);
  EXPECT_EQ(diskann_search_profile_reason::kFast, budget.reason);

  ASSERT_TRUE(choose_diskann_search_budget(
      10, 32, 1, 1024, 64, diskann_search_profile::kBalanced, &budget));
  EXPECT_EQ(1200U, budget.search_complexity);
  EXPECT_EQ(diskann_search_profile_reason::kBalancedManySegments,
            budget.reason);
}

TEST(VectorSegmentedSearchTest, DiskAnnSearchBudgetRejectsInvalidInputs) {
  diskann_search_budget budget;
  EXPECT_FALSE(choose_diskann_search_budget(
      10, 1, 1, 1, 64, diskann_search_profile::kManual, nullptr));
  EXPECT_FALSE(choose_diskann_search_budget(
      0, 1, 1, 1, 64, diskann_search_profile::kManual, &budget));
  EXPECT_FALSE(choose_diskann_search_budget(
      10, 0, 1, 1, 64, diskann_search_profile::kManual, &budget));
  EXPECT_FALSE(choose_diskann_search_budget(
      10, 1, 1, 0, 64, diskann_search_profile::kManual, &budget));
}

TEST(VectorSegmentedSearchTest, DiskAnnSearchBudgetHandlesMaximumTopK) {
  diskann_search_budget budget;
  ASSERT_TRUE(choose_diskann_search_budget(
      std::numeric_limits<uint32_t>::max(), 1, 1,
      std::numeric_limits<uint32_t>::max(), 0,
      diskann_search_profile::kManual, &budget));
  EXPECT_EQ(std::numeric_limits<uint32_t>::max(), budget.search_complexity);
}

TEST(VectorSegmentedSearchTest, DiskAnnSearchProfileNamesCoverAllValues) {
  EXPECT_STREQ("manual", diskann_search_profile_name(
                             diskann_search_profile::kManual));
  EXPECT_STREQ("fast",
               diskann_search_profile_name(diskann_search_profile::kFast));
  EXPECT_STREQ("balanced", diskann_search_profile_name(
                               diskann_search_profile::kBalanced));
  EXPECT_STREQ("high_recall", diskann_search_profile_name(
                                  diskann_search_profile::kHighRecall));
  EXPECT_STREQ("manual", diskann_search_profile_name(
                             static_cast<diskann_search_profile>(999)));

  EXPECT_STREQ("manual", diskann_search_profile_reason_name(
                             diskann_search_profile_reason::kManual));
  EXPECT_STREQ("fast_800", diskann_search_profile_reason_name(
                               diskann_search_profile_reason::kFast));
  EXPECT_STREQ("balanced_many_segments_1200",
               diskann_search_profile_reason_name(
                   diskann_search_profile_reason::kBalancedManySegments));
  EXPECT_STREQ("balanced_few_segments_2400",
               diskann_search_profile_reason_name(
                   diskann_search_profile_reason::kBalancedFewSegments));
  EXPECT_STREQ("high_recall_many_segments_1600",
               diskann_search_profile_reason_name(
                   diskann_search_profile_reason::kHighRecallManySegments));
  EXPECT_STREQ("high_recall_few_segments_3200",
               diskann_search_profile_reason_name(
                   diskann_search_profile_reason::kHighRecallFewSegments));
  EXPECT_STREQ("manual_higher", diskann_search_profile_reason_name(
                                    diskann_search_profile_reason::kManualHigher));
  EXPECT_STREQ("manual", diskann_search_profile_reason_name(
                             static_cast<diskann_search_profile_reason>(999)));
}

TEST(VectorSegmentedSearchTest, MergeHandlesEmptyAndZeroTopK) {
  std::vector<search_result> merged{{1, 1.0}};
  EXPECT_TRUE(merge_segment_topk({}, 10, &merged));
  EXPECT_TRUE(merged.empty());

  EXPECT_TRUE(merge_segment_topk({{{1, 0.1}}}, 0, &merged));
  EXPECT_TRUE(merged.empty());
}

TEST(VectorSegmentedSearchTest, MergeOrdersAcrossSegmentsAndAppliesTopK) {
  std::vector<search_result> merged;
  EXPECT_TRUE(merge_segment_topk(
      {{{10, 0.30}, {30, 0.10}}, {{20, 0.20}, {40, 0.40}}}, 3,
      &merged));

  ASSERT_EQ(3U, merged.size());
  EXPECT_EQ(30U, merged[0].doc_id);
  EXPECT_EQ(20U, merged[1].doc_id);
  EXPECT_EQ(10U, merged[2].doc_id);
}

TEST(VectorSegmentedSearchTest, MergeBreaksDistanceTiesByDocId) {
  std::vector<search_result> merged;
  EXPECT_TRUE(
      merge_segment_topk({{{30, 0.10}}, {{10, 0.10}}, {{20, 0.10}}}, 10,
                         &merged));

  ASSERT_EQ(3U, merged.size());
  EXPECT_EQ(10U, merged[0].doc_id);
  EXPECT_EQ(20U, merged[1].doc_id);
  EXPECT_EQ(30U, merged[2].doc_id);
}

TEST(VectorSegmentedSearchTest, MergeKeepsBestDistanceForDuplicateDocIds) {
  std::vector<search_result> merged;
  EXPECT_TRUE(merge_segment_topk(
      {{{10, 0.50}, {20, 0.20}}, {{10, 0.10}, {30, 0.30}}}, 10,
      &merged));

  ASSERT_EQ(3U, merged.size());
  EXPECT_EQ(10U, merged[0].doc_id);
  EXPECT_DOUBLE_EQ(0.10, merged[0].distance);
  EXPECT_EQ(20U, merged[1].doc_id);
  EXPECT_EQ(30U, merged[2].doc_id);
}

TEST(VectorSegmentedSearchTest, MergeIgnoresWorseDuplicateDistance) {
  std::vector<search_result> merged;
  EXPECT_TRUE(merge_segment_topk(
      {{{10, 0.10}, {20, 0.20}}, {{10, 0.50}, {30, 0.30}}}, 10,
      &merged));

  ASSERT_EQ(3U, merged.size());
  EXPECT_EQ(10U, merged[0].doc_id);
  EXPECT_DOUBLE_EQ(0.10, merged[0].distance);
}

TEST(VectorSegmentedSearchTest, MergeUsesDocIdAtTopKDistanceBoundary) {
  std::vector<search_result> merged;
  EXPECT_TRUE(merge_segment_topk(
      {{{50, 0.10}, {10, 0.10}},
       {{40, 0.10}, {20, 0.10}},
       {{30, 0.10}}},
      3, &merged));

  ASSERT_EQ(3U, merged.size());
  EXPECT_EQ(10U, merged[0].doc_id);
  EXPECT_EQ(20U, merged[1].doc_id);
  EXPECT_EQ(30U, merged[2].doc_id);
}

TEST(VectorSegmentedSearchTest, MergeDeduplicatesBeforeTopKSelection) {
  std::vector<search_result> merged;
  EXPECT_TRUE(merge_segment_topk(
      {{{10, 10.0}, {20, 0.20}, {30, 0.30}},
       {{40, 0.40}, {10, 0.10}}},
      2, &merged));

  ASSERT_EQ(2U, merged.size());
  EXPECT_EQ(10U, merged[0].doc_id);
  EXPECT_DOUBLE_EQ(0.10, merged[0].distance);
  EXPECT_EQ(20U, merged[1].doc_id);
}

TEST(VectorSegmentedSearchTest, BatchMergeRejectsNullOutput) {
  EXPECT_FALSE(merge_segment_batch_topk({}, 1, nullptr));
}

TEST(VectorSegmentedSearchTest, BatchMergeRejectsMismatchedQueryCounts) {
  std::vector<std::vector<search_result>> merged;
  EXPECT_FALSE(merge_segment_batch_topk(
      {{{{1, 0.1}}, {{2, 0.2}}}, {{{3, 0.3}}}}, 1, &merged));
}

TEST(VectorSegmentedSearchTest, BatchMergeHandlesEmptyAndZeroTopK) {
  std::vector<std::vector<search_result>> merged{{{1, 0.1}}};
  EXPECT_TRUE(merge_segment_batch_topk({}, 10, &merged));
  EXPECT_TRUE(merged.empty());

  EXPECT_TRUE(merge_segment_batch_topk({{{{1, 0.1}}, {{2, 0.2}}}}, 0,
                                       &merged));
  ASSERT_EQ(2U, merged.size());
  EXPECT_TRUE(merged[0].empty());
  EXPECT_TRUE(merged[1].empty());
}

TEST(VectorSegmentedSearchTest, BatchMergeKeepsPerQueryTopK) {
  std::vector<std::vector<search_result>> merged;
  EXPECT_TRUE(merge_segment_batch_topk(
      {{{{1, 0.4F}, {2, 0.6F}}, {{10, 0.3F}, {11, 0.5F}}},
       {{{3, 0.2F}, {4, 0.7F}}, {{12, 0.4F}, {13, 0.6F}}}},
      2, &merged));

  ASSERT_EQ(2U, merged.size());
  ASSERT_EQ(2U, merged[0].size());
  ASSERT_EQ(2U, merged[1].size());
  EXPECT_EQ(3U, merged[0][0].doc_id);
  EXPECT_EQ(1U, merged[0][1].doc_id);
  EXPECT_EQ(10U, merged[1][0].doc_id);
  EXPECT_EQ(12U, merged[1][1].doc_id);
}

TEST(VectorSegmentedSearchTest, BatchMergeDeduplicatesEachQueryIndependently) {
  std::vector<std::vector<search_result>> merged;
  EXPECT_TRUE(merge_segment_batch_topk(
      {{{{7, 0.50F}, {8, 0.20F}}, {{30, 0.10F}}},
       {{{7, 0.10F}, {9, 0.30F}}, {{10, 0.10F}, {20, 0.10F}}}},
      3, &merged));

  ASSERT_EQ(2U, merged.size());
  ASSERT_EQ(3U, merged[0].size());
  EXPECT_EQ(7U, merged[0][0].doc_id);
  EXPECT_FLOAT_EQ(0.10F, merged[0][0].distance);
  EXPECT_EQ(8U, merged[0][1].doc_id);
  EXPECT_EQ(9U, merged[0][2].doc_id);

  ASSERT_EQ(3U, merged[1].size());
  EXPECT_EQ(10U, merged[1][0].doc_id);
  EXPECT_EQ(20U, merged[1][1].doc_id);
  EXPECT_EQ(30U, merged[1][2].doc_id);
}

TEST(VectorSegmentedSearchTest,
     BatchCandidatesShareFullCoverageAndMaterializeOnDemand) {
  batch_search_candidates candidates;
  EXPECT_EQ(0U, candidates.query_count());
  EXPECT_EQ(nullptr, candidates.for_query(0));
  EXPECT_EQ(nullptr, candidates.mutable_for_query(0));
  EXPECT_FALSE(std::move(candidates).materialize(nullptr));

  candidates.set_shared(2, {{3, 0.3F}, {1, 0.1F}});
  EXPECT_TRUE(candidates.uses_shared_candidates());
  ASSERT_NE(nullptr, candidates.for_query(0));
  EXPECT_EQ(candidates.for_query(0), candidates.for_query(1));
  EXPECT_EQ(nullptr, candidates.for_query(2));
  EXPECT_EQ(nullptr, candidates.mutable_for_query(0));

  std::vector<std::vector<search_result>> materialized;
  ASSERT_TRUE(std::move(candidates).materialize(&materialized));
  ASSERT_EQ(2U, materialized.size());
  EXPECT_EQ(2U, materialized[0].size());
  EXPECT_EQ(2U, materialized[1].size());

  candidates.prepare_per_query(2);
  EXPECT_FALSE(candidates.uses_shared_candidates());
  ASSERT_NE(nullptr, candidates.mutable_for_query(0));
  candidates.mutable_for_query(0)->push_back({7, 0.7F});
  candidates.mutable_for_query(1)->push_back({8, 0.8F});
  ASSERT_TRUE(std::move(candidates).materialize(&materialized));
  EXPECT_EQ(7U, materialized[0][0].doc_id);
  EXPECT_EQ(8U, materialized[1][0].doc_id);
}

TEST(VectorSegmentedSearchTest, BatchMergeConsumesSharedCandidateRanges) {
  batch_search_candidates first_segment;
  first_segment.set_shared(2, {{7, 0.5F}, {8, 0.2F}});
  batch_search_candidates second_segment;
  second_segment.set_per_query(
      {{{7, 0.1F}, {9, 0.3F}}, {{10, 0.1F}, {20, 0.1F}}});

  std::vector<batch_search_candidates> segments;
  segments.push_back(std::move(first_segment));
  segments.push_back(std::move(second_segment));
  std::vector<std::vector<search_result>> merged;
  ASSERT_TRUE(merge_segment_batch_candidates_topk(segments, 3, &merged));

  ASSERT_EQ(2U, merged.size());
  ASSERT_EQ(3U, merged[0].size());
  EXPECT_EQ(7U, merged[0][0].doc_id);
  EXPECT_EQ(8U, merged[0][1].doc_id);
  EXPECT_EQ(9U, merged[0][2].doc_id);
  ASSERT_EQ(3U, merged[1].size());
  EXPECT_EQ(10U, merged[1][0].doc_id);
  EXPECT_EQ(20U, merged[1][1].doc_id);
  EXPECT_EQ(8U, merged[1][2].doc_id);

  batch_search_candidates mismatched;
  mismatched.set_shared(1, {{1, 0.1F}});
  segments.push_back(std::move(mismatched));
  EXPECT_FALSE(merge_segment_batch_candidates_topk(segments, 3, &merged));
  EXPECT_FALSE(merge_segment_batch_candidates_topk(
      std::vector<batch_search_candidates>{}, 3, nullptr));
}

}  // namespace
}  // namespace vector_index
