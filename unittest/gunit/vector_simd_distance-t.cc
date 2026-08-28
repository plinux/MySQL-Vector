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
#include <cstddef>
#include <vector>

#include "sql/vector/vector_simd_distance.h"

namespace vector_simd_distance_unittest {
namespace {

std::vector<float> make_vector(size_t dimension, float scale,
                               float alternating_offset) {
  std::vector<float> values(dimension + 1);
  values[0] = 99.0F;  // force tests to pass unaligned data() + 1 inputs
  for (size_t i = 0; i < dimension; ++i) {
    const float sign = (i % 2 == 0) ? 1.0F : -1.0F;
    values[i + 1] = sign * (scale * static_cast<float>(i + 1)) +
                    alternating_offset;
  }
  return values;
}

}  // namespace

TEST(VectorSimdDistanceTest, ScalarHandlesAllPlannedDimensions) {
  constexpr std::array<size_t, 12> dimensions = {0, 1, 2, 3, 4, 7,
                                                8, 15, 16, 31, 32, 128};
  for (const size_t dimension : dimensions) {
    const std::vector<float> lhs = make_vector(dimension, 0.25F, -3.0F);
    const std::vector<float> rhs = make_vector(dimension, 0.125F, 2.0F);

    float expected = 0.0F;
    for (size_t i = 0; i < dimension; ++i) {
      const float diff = lhs[i + 1] - rhs[i + 1];
      expected += diff * diff;
    }

    EXPECT_NEAR(expected,
                vector_index::l2_distance_scalar(lhs.data() + 1,
                                                 rhs.data() + 1, dimension),
                1e-5F)
        << "dimension=" << dimension;
  }
}

TEST(VectorSimdDistanceTest, DispatchMatchesScalarForUnalignedInput) {
  constexpr std::array<size_t, 11> dimensions = {1, 2, 3, 4, 7, 8,
                                                15, 16, 31, 32, 128};
  for (const size_t dimension : dimensions) {
    const std::vector<float> lhs = make_vector(dimension, 0.5F, 1.0F);
    const std::vector<float> rhs = make_vector(dimension, -0.25F, 0.0F);
    const float scalar = vector_index::l2_distance_scalar(
        lhs.data() + 1, rhs.data() + 1, dimension);

    vector_index::l2_distance_stats stats;
    const float dispatched = vector_index::l2_distance(
        lhs.data() + 1, rhs.data() + 1, dimension, &stats);

    EXPECT_NEAR(scalar, dispatched, 1e-5F) << "dimension=" << dimension;
    EXPECT_EQ(1U, stats.calls);
    EXPECT_NE(nullptr, vector_index::l2_distance_kernel_name(stats.kernel));
    EXPECT_NE('\0', vector_index::l2_distance_kernel_name(stats.kernel)[0]);
  }
}

TEST(VectorSimdDistanceTest, SupportsZeroVectorsAndNullStats) {
  std::vector<float> lhs(33, 0.0F);
  std::vector<float> rhs(33, 0.0F);

  EXPECT_EQ(0.0F, vector_index::l2_distance(lhs.data() + 1, rhs.data() + 1,
                                            32, nullptr));
}

TEST(VectorSimdDistanceTest, ContextSelectsKernelOnceForShortVectors) {
  constexpr size_t kDimension = 4;
  const std::vector<float> lhs = make_vector(kDimension, 0.5F, 1.0F);
  const std::vector<float> rhs = make_vector(kDimension, -0.25F, 0.0F);
  const float expected = vector_index::l2_distance_scalar(
      lhs.data() + 1, rhs.data() + 1, kDimension);

  const vector_index::l2_distance_context context =
      vector_index::make_l2_distance_context(kDimension);
  EXPECT_EQ(vector_index::l2_distance_kernel::kScalar, context.kernel);
  vector_index::l2_distance_stats stats;
  EXPECT_FLOAT_EQ(
      expected, vector_index::l2_distance_with_context(context, lhs.data() + 1,
                                                       rhs.data() + 1, &stats));
  EXPECT_FLOAT_EQ(
      expected, vector_index::l2_distance_with_context(context, lhs.data() + 1,
                                                       rhs.data() + 1, &stats));
  EXPECT_EQ(2U, stats.calls);
  EXPECT_EQ(context.kernel, stats.kernel);
}

TEST(VectorSimdDistanceTest, ContextMatchesScalarAcrossDimensions) {
  constexpr std::array<size_t, 6> dimensions = {1, 4, 8, 16, 31, 128};
  for (const size_t dimension : dimensions) {
    const std::vector<float> lhs = make_vector(dimension, 0.75F, -2.0F);
    const std::vector<float> rhs = make_vector(dimension, -0.5F, 3.0F);
    const float expected = vector_index::l2_distance_scalar(
        lhs.data() + 1, rhs.data() + 1, dimension);
    const vector_index::l2_distance_context context =
        vector_index::make_l2_distance_context(dimension);
    vector_index::l2_distance_stats stats;
    const float actual = vector_index::l2_distance_with_context(
        context, lhs.data() + 1, rhs.data() + 1, &stats);

    EXPECT_NEAR(expected, actual, 1e-5F) << "dimension=" << dimension;
    EXPECT_EQ(1U, stats.calls);
    EXPECT_EQ(context.kernel, stats.kernel);
  }
}

TEST(VectorSimdDistanceTest, KernelNamesAreStable) {
  EXPECT_STREQ("scalar", vector_index::l2_distance_kernel_name(
                             vector_index::l2_distance_kernel::kScalar));
  EXPECT_STREQ("avx2", vector_index::l2_distance_kernel_name(
                           vector_index::l2_distance_kernel::kAvx2));
  EXPECT_STREQ("avx512", vector_index::l2_distance_kernel_name(
                             vector_index::l2_distance_kernel::kAvx512));
  EXPECT_STREQ("neon", vector_index::l2_distance_kernel_name(
                           vector_index::l2_distance_kernel::kNeon));
  EXPECT_STREQ("unknown", vector_index::l2_distance_kernel_name(
                              static_cast<vector_index::l2_distance_kernel>(
                                  999)));
}

TEST(VectorSimdDistanceTest, NullContextFunctionFallsBackToScalar) {
  constexpr size_t kDimension = 4;
  const std::vector<float> lhs = make_vector(kDimension, 0.5F, 1.0F);
  const std::vector<float> rhs = make_vector(kDimension, -0.25F, 0.0F);
  const float expected = vector_index::l2_distance_scalar(
      lhs.data() + 1, rhs.data() + 1, kDimension);

  vector_index::l2_distance_context context;
  context.dimension = kDimension;
  context.function = nullptr;
  vector_index::l2_distance_stats stats;
  EXPECT_FLOAT_EQ(
      expected, vector_index::l2_distance_with_context(context, lhs.data() + 1,
                                                       rhs.data() + 1, &stats));
  EXPECT_EQ(vector_index::l2_distance_kernel::kScalar, stats.kernel);
  EXPECT_EQ(1U, stats.calls);
}

}  // namespace vector_simd_distance_unittest
