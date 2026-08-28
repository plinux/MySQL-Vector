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
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "sql/vector/vector_pq_centroid_scan.h"

namespace vector_pq_centroid_scan_unittest {
namespace {

std::vector<float> make_centers(uint32_t center_count, uint32_t dimension) {
  std::vector<float> centers(static_cast<size_t>(center_count) * dimension);
  for (uint32_t center = 0; center < center_count; ++center) {
    for (uint32_t dim = 0; dim < dimension; ++dim) {
      centers[static_cast<size_t>(center) * dimension + dim] =
          static_cast<float>((center * 17 + dim * 11) % 37) / 7.0F;
    }
  }
  return centers;
}

std::vector<float> make_unaligned_query(uint32_t dimension) {
  std::vector<float> query(dimension + 1, 0.0F);
  query[0] = 99.0F;
  for (uint32_t dim = 0; dim < dimension; ++dim) {
    query[dim + 1] = static_cast<float>((dim * 13 + 5) % 29) / 9.0F;
  }
  return query;
}

uint32_t reference_nearest(const std::vector<float> &centers,
                           uint32_t center_count, uint32_t dimension,
                           const float *query,
                           std::vector<float> *squared_distances) {
  squared_distances->assign(center_count, 0.0F);
  uint32_t nearest = 0;
  float best = std::numeric_limits<float>::infinity();
  for (uint32_t center = 0; center < center_count; ++center) {
    float distance = 0.0F;
    for (uint32_t dim = 0; dim < dimension; ++dim) {
      const float diff =
          query[dim] - centers[static_cast<size_t>(center) * dimension + dim];
      distance += diff * diff;
    }
    (*squared_distances)[center] = distance;
    if (distance < best) {
      best = distance;
      nearest = center;
    }
  }
  return nearest;
}

void expect_kernel_matches_reference(
    vector_index::pq_centroid_scan_kernel kernel) {
  constexpr std::array<uint32_t, 5> kDimensions = {1, 2, 4, 8, 16};
  constexpr std::array<uint32_t, 7> kCenterCounts = {1, 7, 8, 15, 16, 17, 31};
  for (const uint32_t dimension : kDimensions) {
    for (const uint32_t center_count : kCenterCounts) {
      const std::vector<float> centers = make_centers(center_count, dimension);
      const std::vector<float> query = make_unaligned_query(dimension);
      std::vector<float> expected_distances;
      const uint32_t expected =
          reference_nearest(centers, center_count, dimension, query.data() + 1,
                            &expected_distances);

      vector_index::pq_centroid_model model;
      std::string error;
      ASSERT_TRUE(vector_index::make_pq_centroid_model(
          centers.data(), center_count, dimension, kernel, &model, &error))
          << error;
      std::vector<float> actual_distances(center_count, 0.0F);
      uint32_t actual = center_count;
      float nearest_distance = 0.0F;
      ASSERT_TRUE(vector_index::scan_pq_centroids(model, query.data() + 1,
                                                  actual_distances.data(),
                                                  &actual, &nearest_distance));

      EXPECT_EQ(expected, actual)
          << "dimension=" << dimension << " centers=" << center_count;
      EXPECT_NEAR(expected_distances[expected], nearest_distance, 1e-5F);
      for (uint32_t center = 0; center < center_count; ++center) {
        EXPECT_NEAR(expected_distances[center], actual_distances[center], 1e-5F)
            << "dimension=" << dimension << " centers=" << center_count
            << " center=" << center;
      }
    }
  }
}

}  // namespace

TEST(VectorPqCentroidScanTest, ScalarMatchesReferenceForShortDimensions) {
  expect_kernel_matches_reference(
      vector_index::pq_centroid_scan_kernel::kScalar);
}

TEST(VectorPqCentroidScanTest, AutoDispatchMatchesReference) {
  expect_kernel_matches_reference(vector_index::pq_centroid_scan_kernel::kAuto);
}

TEST(VectorPqCentroidScanTest, Avx2MatchesReferenceWhenAvailable) {
  if (!vector_index::pq_centroid_scan_kernel_supported(
          vector_index::pq_centroid_scan_kernel::kAvx2)) {
    GTEST_SKIP() << "AVX2 is not available on this CPU";
  }
  expect_kernel_matches_reference(vector_index::pq_centroid_scan_kernel::kAvx2);
}

TEST(VectorPqCentroidScanTest, Avx512MatchesReferenceWhenAvailable) {
  if (!vector_index::pq_centroid_scan_kernel_supported(
          vector_index::pq_centroid_scan_kernel::kAvx512)) {
    GTEST_SKIP() << "AVX-512 is not available on this CPU";
  }
  expect_kernel_matches_reference(
      vector_index::pq_centroid_scan_kernel::kAvx512);
}

TEST(VectorPqCentroidScanTest, EqualDistancesSelectLowestCenter) {
  constexpr uint32_t kDimension = 4;
  const std::vector<float> centers = {1.0F, 2.0F, 3.0F, 4.0F,
                                      1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> query = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F};

  vector_index::pq_centroid_model model;
  std::string error;
  ASSERT_TRUE(vector_index::make_pq_centroid_model(
      centers.data(), 2, kDimension,
      vector_index::pq_centroid_scan_kernel::kAuto, &model, &error));
  uint32_t nearest = 2;
  ASSERT_TRUE(vector_index::scan_pq_centroids(model, query.data() + 1, nullptr,
                                              &nearest, nullptr));
  EXPECT_EQ(0U, nearest);
}

TEST(VectorPqCentroidScanTest, EuclideanDistancesMatchReference) {
  constexpr uint32_t kDimension = 4;
  constexpr uint32_t kCenterCount = 17;
  const std::vector<float> centers = make_centers(kCenterCount, kDimension);
  const std::vector<float> query = make_unaligned_query(kDimension);
  std::vector<float> expected_squared;
  const uint32_t expected = reference_nearest(
      centers, kCenterCount, kDimension, query.data() + 1, &expected_squared);

  vector_index::pq_centroid_model model;
  std::string error;
  ASSERT_TRUE(vector_index::make_pq_centroid_model(
      centers.data(), kCenterCount, kDimension,
      vector_index::pq_centroid_scan_kernel::kAuto, &model, &error));
  std::vector<float> actual(kCenterCount, 0.0F);
  uint32_t nearest = kCenterCount;
  float nearest_distance = 0.0F;
  ASSERT_TRUE(vector_index::scan_pq_centroid_distances(
      model, query.data() + 1, actual.data(), &nearest, &nearest_distance));

  EXPECT_EQ(expected, nearest);
  EXPECT_NEAR(std::sqrt(expected_squared[expected]), nearest_distance, 1e-5F);
  for (uint32_t center = 0; center < kCenterCount; ++center) {
    EXPECT_NEAR(std::sqrt(expected_squared[center]), actual[center], 1e-5F);
  }
}

TEST(VectorPqCentroidScanTest, RejectsInvalidModelInputs) {
  const float center = 0.0F;
  vector_index::pq_centroid_model model;
  std::string error;
  EXPECT_FALSE(vector_index::make_pq_centroid_model(
      nullptr, 1, 1, vector_index::pq_centroid_scan_kernel::kScalar, &model,
      &error));
  EXPECT_FALSE(vector_index::make_pq_centroid_model(
      &center, 0, 1, vector_index::pq_centroid_scan_kernel::kScalar, &model,
      &error));
  EXPECT_FALSE(vector_index::make_pq_centroid_model(
      &center, 1, 0, vector_index::pq_centroid_scan_kernel::kScalar, &model,
      &error));
  EXPECT_FALSE(vector_index::scan_pq_centroids(model, &center, nullptr, nullptr,
                                               nullptr));
}

TEST(VectorPqCentroidScanTest, RejectsNonFiniteAndUnavailableKernels) {
  const float non_finite = std::numeric_limits<float>::infinity();
  vector_index::pq_centroid_model model;
  std::string error;
  EXPECT_FALSE(vector_index::make_pq_centroid_model(
      &non_finite, 1, 1, vector_index::pq_centroid_scan_kernel::kScalar,
      &model, &error));
  EXPECT_EQ("centroid matrix contains non-finite value", error);

  const float center = 0.0F;
  const auto invalid_kernel =
      static_cast<vector_index::pq_centroid_scan_kernel>(999);
  EXPECT_FALSE(vector_index::pq_centroid_scan_kernel_supported(invalid_kernel));
  EXPECT_FALSE(vector_index::make_pq_centroid_model(
      &center, 1, 1, invalid_kernel, &model, &error));
  EXPECT_EQ("requested centroid scan kernel is unavailable", error);
  EXPECT_STREQ("unknown",
               vector_index::pq_centroid_scan_kernel_name(invalid_kernel));
  EXPECT_FALSE(vector_index::make_pq_centroid_model(
      &center, 1, 1, vector_index::pq_centroid_scan_kernel::kScalar, nullptr,
      nullptr));
}

TEST(VectorPqCentroidScanTest, ScanRejectsInvalidQueryAndOutputCombinations) {
  const float center = 0.0F;
  vector_index::pq_centroid_model model;
  std::string error;
  ASSERT_TRUE(vector_index::make_pq_centroid_model(
      &center, 1, 1, vector_index::pq_centroid_scan_kernel::kScalar, &model,
      &error));

  uint32_t nearest = 0;
  EXPECT_FALSE(vector_index::scan_pq_centroids(model, nullptr, nullptr,
                                               &nearest, nullptr));
  EXPECT_FALSE(vector_index::scan_pq_centroids(model, &center, nullptr,
                                               nullptr, nullptr));

  model.kernel = vector_index::pq_centroid_scan_kernel::kAuto;
  EXPECT_FALSE(vector_index::scan_pq_centroids(model, &center, nullptr,
                                               &nearest, nullptr));
}

TEST(VectorPqCentroidScanTest, KernelNamesAreStable) {
  EXPECT_STREQ("auto", vector_index::pq_centroid_scan_kernel_name(
                           vector_index::pq_centroid_scan_kernel::kAuto));
  EXPECT_STREQ("scalar_fallback",
               vector_index::pq_centroid_scan_kernel_name(
                   vector_index::pq_centroid_scan_kernel::kScalar));
  EXPECT_STREQ("avx2", vector_index::pq_centroid_scan_kernel_name(
                           vector_index::pq_centroid_scan_kernel::kAvx2));
  EXPECT_STREQ("avx512", vector_index::pq_centroid_scan_kernel_name(
                             vector_index::pq_centroid_scan_kernel::kAvx512));
}

}  // namespace vector_pq_centroid_scan_unittest
