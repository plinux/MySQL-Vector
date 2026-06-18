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

#include <array>
#include <limits>
#include <string>
#include <vector>

#include "m_ctype.h"
#include "my_byteorder.h"
#include "sql/vector/vector_mapped_search.h"
#include "sql/vector/vector_utils.h"
#include "sql_string.h"
#include "unittest/gunit/test_utils.h"

namespace vector_mapped_search_unittest {

namespace {

using my_testing::Server_initializer;

}  // namespace

TEST(VectorMappedSearchTest, RankVisibleCandidatesReordersByVisibleDistance) {
  const vector_index::vector_data query{9.0F, 9.0F};
  const std::vector<vector_mapped_search::visible_candidate> visible_rows{
      {1, {1.0F, 1.0F}}, {4, {8.0F, 8.0F}}};

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(vector_mapped_search::rank_visible_candidates(
      query, vector_index::metric_type::kEuclidean, visible_rows, 1, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(4U, results[0].doc_id);
}

TEST(VectorMappedSearchTest, RankVisibleCandidatesUsesInnerProductOrder) {
  const vector_index::vector_data query{1.0F, 1.0F};
  const std::vector<vector_mapped_search::visible_candidate> visible_rows{
      {10, {2.0F, 2.0F}}, {20, {1.0F, 0.0F}}};

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(vector_mapped_search::rank_visible_candidates(
      query, vector_index::metric_type::kInnerProduct, visible_rows, 2, &results));
  ASSERT_EQ(2U, results.size());
  EXPECT_EQ(10U, results[0].doc_id);
  EXPECT_EQ(20U, results[1].doc_id);
}

TEST(VectorMappedSearchTest, RankVisibleCandidatesCoversErrorAndTieBranches) {
  const vector_index::vector_data query{1.0F, 1.0F};
  std::vector<vector_index::search_result> results{{99, 99.0}};

  EXPECT_FALSE(vector_mapped_search::rank_visible_candidates(
      query, vector_index::metric_type::kEuclidean, {}, 1, nullptr));
  EXPECT_FALSE(vector_mapped_search::rank_visible_candidates(
      query, vector_index::metric_type::kEuclidean,
      {{1, {1.0F, 1.0F, 1.0F}}}, 1, &results));

  ASSERT_TRUE(vector_mapped_search::rank_visible_candidates(
      query, vector_index::metric_type::kEuclidean,
      {{20, {2.0F, 2.0F}}, {10, {2.0F, 2.0F}}}, 0, &results));
  EXPECT_TRUE(results.empty());

  ASSERT_TRUE(vector_mapped_search::rank_visible_candidates(
      query, vector_index::metric_type::kEuclidean,
      {{20, {2.0F, 2.0F}}, {10, {2.0F, 2.0F}}}, 2, &results));
  ASSERT_EQ(2U, results.size());
  EXPECT_EQ(10U, results[0].doc_id);
  EXPECT_EQ(20U, results[1].doc_id);
}

TEST(VectorMappedSearchTest, RankVisibleCandidatesCoversCosineZeroNorm) {
  const vector_index::vector_data query{1.0F, 0.0F};
  std::vector<vector_index::search_result> results;

  ASSERT_TRUE(vector_mapped_search::rank_visible_candidates(
      query, vector_index::metric_type::kCosine,
      {{10, {1.0F, 0.0F}}, {20, {0.0F, 0.0F}}}, 2, &results));
  ASSERT_EQ(2U, results.size());
  EXPECT_EQ(10U, results[0].doc_id);
  EXPECT_EQ(20U, results[1].doc_id);

  ASSERT_TRUE(vector_mapped_search::rank_visible_candidates(
      {0.0F, 0.0F}, vector_index::metric_type::kCosine,
      {{10, {1.0F, 0.0F}}, {20, {0.0F, 1.0F}}}, 2, &results));
  ASSERT_EQ(2U, results.size());
}

TEST(VectorMappedSearchTest, FilterVisibleResultsRejectsNullAndKeepsEmptyInput) {
  vector_mapped_search::search_spec spec;
  std::vector<vector_index::search_result> results;

  EXPECT_FALSE(vector_mapped_search::filter_visible_results(
      nullptr, spec, {1.0F, 1.0F}, nullptr));
  EXPECT_TRUE(vector_mapped_search::filter_visible_results(
      nullptr, spec, {1.0F, 1.0F}, &results));

  results.push_back({1, 0.0});
  EXPECT_FALSE(vector_mapped_search::filter_visible_results(
      nullptr, spec, {1.0F, 1.0F}, &results));
}

TEST(VectorMappedSearchTest, TestingWrappersDecodeBinaryAndTextVectors) {
  vector_index::vector_data decoded;
  EXPECT_FALSE(
      vector_mapped_search::decode_binary_vector_for_testing(nullptr, &decoded));

  String text("[1.5,2.5]", &my_charset_bin);
  EXPECT_FALSE(
      vector_mapped_search::decode_binary_vector_for_testing(&text, nullptr));
  ASSERT_TRUE(
      vector_mapped_search::decode_binary_vector_for_testing(&text, &decoded));
  ASSERT_EQ(2U, decoded.size());
  EXPECT_FLOAT_EQ(1.5F, decoded[0]);
  EXPECT_FLOAT_EQ(2.5F, decoded[1]);

  String malformed("[1.5,", &my_charset_bin);
  EXPECT_FALSE(vector_mapped_search::decode_binary_vector_for_testing(
      &malformed, &decoded));

  std::string binary(2 * vector_utils::kVectorElemSize, '\0');
  float4store(pointer_cast<uchar *>(&binary[0]), 3.5F);
  float4store(pointer_cast<uchar *>(&binary[vector_utils::kVectorElemSize]),
              4.5F);
  String binary_value(binary.data(), binary.size(), &my_charset_bin);
  ASSERT_TRUE(vector_mapped_search::decode_binary_vector_for_testing(
      &binary_value, &decoded));
  ASSERT_EQ(2U, decoded.size());
  EXPECT_FLOAT_EQ(3.5F, decoded[0]);
  EXPECT_FLOAT_EQ(4.5F, decoded[1]);
}

TEST(VectorMappedSearchTest, TestingWrappersParseDocIdColumnShapes) {
  uint64_t doc_id = 0;
  EXPECT_FALSE(vector_mapped_search::parse_null_doc_id_column_for_testing(
      &doc_id));
  EXPECT_FALSE(vector_mapped_search::parse_doc_id_column_for_testing(
      nullptr, 0, &doc_id));
  EXPECT_FALSE(vector_mapped_search::parse_doc_id_column_for_testing(
      "42", 2, nullptr));
  EXPECT_FALSE(vector_mapped_search::parse_doc_id_column_for_testing(
      "", 0, &doc_id));
  std::array<uchar, 8> bytes{};
  bytes[0] = 7;
  ASSERT_TRUE(vector_mapped_search::parse_doc_id_column_for_testing(
      pointer_cast<const char *>(bytes.data()), 1, &doc_id));
  EXPECT_EQ(7U, doc_id);

  bytes[0] = 0xFF;
  ASSERT_TRUE(vector_mapped_search::parse_doc_id_column_for_testing(
      pointer_cast<const char *>(bytes.data()), 1, &doc_id));
  EXPECT_EQ(255U, doc_id);

  int2store(bytes.data(), 300);
  ASSERT_TRUE(vector_mapped_search::parse_doc_id_column_for_testing(
      pointer_cast<const char *>(bytes.data()), 2, &doc_id));
  EXPECT_EQ(300U, doc_id);

  int3store(bytes.data(), 70000);
  ASSERT_TRUE(vector_mapped_search::parse_doc_id_column_for_testing(
      pointer_cast<const char *>(bytes.data()), 3, &doc_id));
  EXPECT_EQ(70000U, doc_id);

  int4store(bytes.data(), 700000);
  ASSERT_TRUE(vector_mapped_search::parse_doc_id_column_for_testing(
      pointer_cast<const char *>(bytes.data()), 4, &doc_id));
  EXPECT_EQ(700000U, doc_id);

  int4store(bytes.data(), 51);
  ASSERT_TRUE(vector_mapped_search::parse_doc_id_column_for_testing(
      pointer_cast<const char *>(bytes.data()), 4, &doc_id));
  EXPECT_EQ(51U, doc_id);

  int8store(bytes.data(), 7000000000LL);
  ASSERT_TRUE(vector_mapped_search::parse_doc_id_column_for_testing(
      pointer_cast<const char *>(bytes.data()), 8, &doc_id));
  EXPECT_EQ(7000000000ULL, doc_id);

  int8store(bytes.data(), std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(vector_mapped_search::parse_doc_id_column_for_testing(
      pointer_cast<const char *>(bytes.data()), 8, &doc_id));
  EXPECT_EQ(std::numeric_limits<uint64_t>::max(), doc_id);

  EXPECT_FALSE(vector_mapped_search::parse_doc_id_column_for_testing(
      pointer_cast<const char *>(bytes.data()), 5, &doc_id));
}

TEST(VectorMappedSearchTest, TestingWrappersCoverContextAndCollectGuards) {
  Server_initializer initializer;
  initializer.SetUp();

  vector_mapped_search::search_spec spec;
  std::vector<vector_index::search_result> candidates;
  std::vector<vector_mapped_search::visible_candidate> rows{{9, {9.0F}}};

  EXPECT_FALSE(vector_mapped_search::activate_scoped_context_for_testing(
      nullptr));
  EXPECT_FALSE(vector_mapped_search::collect_visible_rows_for_testing(
      initializer.thd(), spec, candidates, nullptr));
  EXPECT_TRUE(vector_mapped_search::collect_visible_rows_for_testing(
      initializer.thd(), spec, candidates, &rows));
  EXPECT_TRUE(rows.empty());

  initializer.TearDown();
}

TEST(VectorMappedSearchTest, TestingWrapperQuotesBackticksInIdentifiers) {
  EXPECT_EQ("`simple`", vector_mapped_search::quote_identifier_for_testing(
                            "simple"));
  EXPECT_EQ("`has``tick`",
            vector_mapped_search::quote_identifier_for_testing("has`tick"));
}

}  // namespace vector_mapped_search_unittest
