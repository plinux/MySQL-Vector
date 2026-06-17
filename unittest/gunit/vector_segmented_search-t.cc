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

#include <vector>

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

}  // namespace
}  // namespace vector_index
