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

}  // namespace
}  // namespace vector_index
