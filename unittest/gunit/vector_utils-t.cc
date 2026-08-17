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

#include <cstddef>
#include <initializer_list>
#include <vector>

#include "my_byteorder.h"
#include "sql/vector/vector_utils.h"
#include "sql_string.h"

namespace vector_utils_unittest {

namespace {

class binary_vector_data {
 public:
  explicit binary_vector_data(std::initializer_list<float> values)
      : m_bytes(values.size() * vector_utils::kVectorElemSize),
        m_string(reinterpret_cast<const char *>(m_bytes.data()), m_bytes.size(),
                 &my_charset_bin) {
    size_t pos = 0;
    for (const float value : values) {
      float4store(m_bytes.data() + pos, value);
      pos += vector_utils::kVectorElemSize;
    }
  }

  const String *string() const { return &m_string; }

 private:
  std::vector<uchar> m_bytes;
  String m_string;
};

}  // namespace

TEST(VectorUtilsTest, ParseTextVectorSuccess) {
  String input(" [1, -2.5, 3e1] ", &my_charset_latin1);
  std::vector<float> values;

  EXPECT_TRUE(vector_utils::parse_text_vector(&input, &values));
  ASSERT_EQ(3U, values.size());
  EXPECT_FLOAT_EQ(1.0F, values[0]);
  EXPECT_FLOAT_EQ(-2.5F, values[1]);
  EXPECT_FLOAT_EQ(30.0F, values[2]);
}

TEST(VectorUtilsTest, ParseTextVectorRejectsInvalidSyntax) {
  String input("[1,2", &my_charset_latin1);
  std::vector<float> values{42.0F};

  EXPECT_FALSE(vector_utils::parse_text_vector(&input, &values));
  EXPECT_TRUE(values.empty());
}

TEST(VectorUtilsTest, ParseTextVectorRejectsMalformedTokens) {
  const char *invalid_inputs[] = {
      "",
      "1, 2]",
      "[not-a-number]",
      "[1e1000]",
      "[1 2]",
      "[1] trailing",
      "[] trailing",
      "[1,]",
  };

  for (const char *text : invalid_inputs) {
    String input(text, &my_charset_latin1);
    std::vector<float> values{42.0F};

    EXPECT_FALSE(vector_utils::parse_text_vector(&input, &values)) << text;
    EXPECT_TRUE(values.empty()) << text;
  }
}

TEST(VectorUtilsTest, ParseBinaryVectorAndCompatibility) {
  binary_vector_data lhs({1.0F, 2.0F, 3.0F});
  binary_vector_data rhs({4.0F, 5.0F, 6.0F});
  size_t dim = 0;

  EXPECT_TRUE(vector_utils::parse_binary_vector(lhs.string(), &dim));
  EXPECT_EQ(3U, dim);
  EXPECT_TRUE(vector_utils::check_compatible_vector_inputs(
      lhs.string(), rhs.string(), &dim));
  EXPECT_EQ(3U, dim);
}

TEST(VectorUtilsTest, ParseBinaryVectorRejectsMisalignedLength) {
  const char data[] = {'a', 'b', 'c'};
  String input(data, sizeof(data), &my_charset_bin);
  size_t dim = 0;

  EXPECT_FALSE(vector_utils::parse_binary_vector(&input, &dim));
}

TEST(VectorUtilsTest, ParseDistanceMetricRecognizesAliases) {
  String l2("  L2 ", &my_charset_latin1);
  String cosine("cosine", &my_charset_latin1);
  String inner_product(" INNER_PRODUCT ", &my_charset_latin1);
  vector_utils::distance_metric metric = vector_utils::distance_metric::kCosine;

  EXPECT_TRUE(vector_utils::parse_distance_metric(&l2, &metric));
  EXPECT_EQ(vector_utils::distance_metric::kEuclidean, metric);
  EXPECT_TRUE(vector_utils::parse_distance_metric(&cosine, &metric));
  EXPECT_EQ(vector_utils::distance_metric::kCosine, metric);
  EXPECT_TRUE(vector_utils::parse_distance_metric(&inner_product, &metric));
  EXPECT_EQ(vector_utils::distance_metric::kInnerProduct, metric);
}

TEST(VectorUtilsTest, ParseDistanceMetricRejectsUnknownMetric) {
  String metric_arg("manhattan", &my_charset_latin1);
  vector_utils::distance_metric metric =
      vector_utils::distance_metric::kEuclidean;

  EXPECT_FALSE(vector_utils::parse_distance_metric(&metric_arg, &metric));
}

TEST(VectorUtilsTest, ComputeDistanceEuclidean) {
  binary_vector_data lhs({1.0F, 2.0F});
  binary_vector_data rhs({4.0F, 6.0F});
  double distance = 0.0;

  EXPECT_TRUE(vector_utils::compute_distance(
      vector_utils::distance_metric::kEuclidean, lhs.string(), rhs.string(),
      &distance));
  EXPECT_DOUBLE_EQ(5.0, distance);
}

TEST(VectorUtilsTest, ComputeDistanceCosine) {
  binary_vector_data lhs({1.0F, 0.0F});
  binary_vector_data rhs({0.0F, 1.0F});
  double distance = 0.0;

  EXPECT_TRUE(vector_utils::compute_distance(
      vector_utils::distance_metric::kCosine, lhs.string(), rhs.string(),
      &distance));
  EXPECT_DOUBLE_EQ(1.0, distance);
}

TEST(VectorUtilsTest, ComputeDistanceCosineRejectsZeroNorm) {
  binary_vector_data lhs({0.0F, 0.0F});
  binary_vector_data rhs({1.0F, 2.0F});
  double distance = 0.0;

  EXPECT_FALSE(vector_utils::compute_distance(
      vector_utils::distance_metric::kCosine, lhs.string(), rhs.string(),
      &distance));
}

TEST(VectorUtilsTest, ComputeDotProduct) {
  binary_vector_data lhs({1.0F, 2.0F});
  binary_vector_data rhs({4.0F, 6.0F});
  double dot_product = 0.0;

  EXPECT_TRUE(
      vector_utils::compute_dot_product(lhs.string(), rhs.string(), &dot_product));
  EXPECT_DOUBLE_EQ(16.0, dot_product);
}

TEST(VectorUtilsTest, ComputeDotProductRejectsDimensionMismatch) {
  binary_vector_data lhs({1.0F, 2.0F});
  binary_vector_data rhs({4.0F, 6.0F, 8.0F});
  double dot_product = 0.0;

  EXPECT_FALSE(vector_utils::compute_dot_product(lhs.string(), rhs.string(),
                                                 &dot_product));
}

TEST(VectorUtilsTest, ComputeDistanceInnerProduct) {
  binary_vector_data lhs({1.0F, 2.0F});
  binary_vector_data rhs({4.0F, 6.0F});
  double distance = 0.0;

  EXPECT_TRUE(vector_utils::compute_distance(
      vector_utils::distance_metric::kInnerProduct, lhs.string(), rhs.string(),
      &distance));
  EXPECT_DOUBLE_EQ(-16.0, distance);
}

}  // namespace vector_utils_unittest
