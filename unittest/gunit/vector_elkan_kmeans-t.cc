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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "sql/vector/vector_elkan_kmeans.h"
#include "sql/vector/vector_simd_distance.h"

namespace vector_elkan_kmeans_unittest {
namespace {

vector_index::elkan_kmeans_config make_config(uint32_t dimension,
                                              uint32_t center_count) {
  vector_index::elkan_kmeans_config config;
  config.dimension = dimension;
  config.center_count = center_count;
  config.max_iterations = 20;
  config.seed = 0;
  config.zero_mean = false;
  return config;
}

uint32_t brute_force_assignment(const std::vector<float> &data, uint64_t row,
                                const vector_index::elkan_kmeans_config &config,
                                const vector_index::elkan_kmeans_result &result) {
  uint32_t best_center = 0;
  float best_distance = std::numeric_limits<float>::infinity();
  const float *point = data.data() + static_cast<size_t>(row) * config.dimension;
  for (uint32_t center = 0; center < config.center_count; ++center) {
    const float *center_values =
        result.centers.data() + static_cast<size_t>(center) * config.dimension;
    const float distance = vector_index::l2_distance_scalar(
        point, center_values, config.dimension);
    if (distance < best_distance) {
      best_distance = distance;
      best_center = center;
    }
  }
  return best_center;
}

}  // namespace

TEST(VectorElkanKmeansTest, ConvergesTwoObviousClusters) {
  const std::vector<float> data = {0.0F, 0.0F, 0.0F, 1.0F,
                                  10.0F, 10.0F, 10.0F, 11.0F};
  const vector_index::elkan_kmeans_config config = make_config(2, 2);

  vector_index::elkan_kmeans_result result;
  std::string error;
  ASSERT_TRUE(vector_index::run_elkan_kmeans(data.data(), 4, config, &result,
                                             &error))
      << error;
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.iterations, config.max_iterations);
  EXPECT_EQ(result.assignments[0], result.assignments[1]);
  EXPECT_EQ(result.assignments[2], result.assignments[3]);
  EXPECT_NE(result.assignments[0], result.assignments[2]);
}

TEST(VectorElkanKmeansTest, RepairsEmptyClustersWithoutNanCenters) {
  const std::vector<float> data = {0.0F, 0.0F, 0.0F, 0.0F,
                                  10.0F, 10.0F, 10.0F, 10.0F};
  vector_index::elkan_kmeans_config config = make_config(2, 3);
  config.max_iterations = 5;

  vector_index::elkan_kmeans_result result;
  std::string error;
  ASSERT_TRUE(vector_index::run_elkan_kmeans(data.data(), 4, config, &result,
                                             &error))
      << error;
  ASSERT_EQ(static_cast<size_t>(config.center_count * config.dimension),
            result.centers.size());
  for (float value : result.centers) EXPECT_TRUE(std::isfinite(value));
}

TEST(VectorElkanKmeansTest, IsDeterministicForSameSeed) {
  const std::vector<float> data = {0.0F, 0.0F, 0.0F, 1.0F, 9.0F, 9.0F,
                                  9.0F, 10.0F, 20.0F, 20.0F, 20.0F, 21.0F};
  vector_index::elkan_kmeans_config config = make_config(2, 3);
  config.seed = 7;

  vector_index::elkan_kmeans_result first;
  vector_index::elkan_kmeans_result second;
  std::string error;
  ASSERT_TRUE(vector_index::run_elkan_kmeans(data.data(), 6, config, &first,
                                             &error))
      << error;
  ASSERT_TRUE(vector_index::run_elkan_kmeans(data.data(), 6, config, &second,
                                             &error))
      << error;
  EXPECT_EQ(first.assignments, second.assignments);
  EXPECT_EQ(first.centers, second.centers);
}

TEST(VectorElkanKmeansTest, AppliesZeroMeanCentroidNormalization) {
  const std::vector<float> data = {1.0F, 2.0F, 3.0F, 4.0F,
                                  10.0F, 20.0F, 30.0F, 40.0F};
  vector_index::elkan_kmeans_config config = make_config(2, 2);
  config.zero_mean = true;

  vector_index::elkan_kmeans_result result;
  std::string error;
  ASSERT_TRUE(vector_index::run_elkan_kmeans(data.data(), 4, config, &result,
                                             &error))
      << error;
  ASSERT_EQ(2U, result.centroid.size());
  EXPECT_FLOAT_EQ(11.0F, result.centroid[0]);
  EXPECT_FLOAT_EQ(16.5F, result.centroid[1]);
  ASSERT_EQ(static_cast<size_t>(config.center_count * config.dimension),
            result.centers.size());
  for (float value : result.centers) EXPECT_TRUE(std::isfinite(value));
}

TEST(VectorElkanKmeansTest, StopsAtIterationLimitWhenAssignmentsKeepMoving) {
  const std::vector<float> data = {
      0.6982032F,  7.5431795F,  -0.10602845F, -8.598239F,
      5.57362F,    3.6338415F,  -8.811392F,   -9.580304F};
  vector_index::elkan_kmeans_config config = make_config(2, 2);
  config.max_iterations = 1;
  config.seed = 0;

  vector_index::elkan_kmeans_result result;
  std::string error;
  ASSERT_TRUE(vector_index::run_elkan_kmeans(data.data(), 4, config, &result,
                                             &error))
      << error;
  EXPECT_EQ(1U, result.iterations);
  EXPECT_FALSE(result.converged);
  EXPECT_GT(result.distance_calls, 0U);
}

TEST(VectorElkanKmeansTest, PrunesDistancesAndMatchesBruteForceAssignment) {
  const std::vector<float> data = {
      0.0F, 0.0F, 0.0F, 0.5F, 0.5F, 0.0F, 0.5F, 0.5F,
      20.0F, 20.0F, 20.0F, 20.5F, 20.5F, 20.0F, 20.5F, 20.5F};
  const vector_index::elkan_kmeans_config config = make_config(2, 2);

  vector_index::elkan_kmeans_result result;
  std::string error;
  ASSERT_TRUE(vector_index::run_elkan_kmeans(data.data(), 8, config, &result,
                                             &error))
      << error;
  EXPECT_GT(result.skipped_distance_calls, 0U);
  EXPECT_GT(result.distance_calls, 0U);
  EXPECT_EQ(8U * config.center_count, result.packed_distance_calls);
  EXPECT_NE("not_run", result.centroid_scan_kernel);
  for (uint64_t row = 0; row < 8; ++row) {
    EXPECT_EQ(brute_force_assignment(data, row, config, result),
              result.assignments[static_cast<size_t>(row)])
        << "row=" << row;
  }
}

TEST(VectorElkanKmeansTest, ThreadBudgetDoesNotChangeDeterministicResult) {
  constexpr uint32_t kRows = 64;
  constexpr uint32_t kDimension = 4;
  constexpr uint32_t kCenters = 8;
  std::vector<float> data(static_cast<size_t>(kRows) * kDimension);
  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t dim = 0; dim < kDimension; ++dim) {
      data[static_cast<size_t>(row) * kDimension + dim] =
          static_cast<float>((row * 17 + dim * 11) % 53) / 53.0F;
    }
  }

  vector_index::elkan_kmeans_config serial_config =
      make_config(kDimension, kCenters);
  serial_config.seed = 11;
  serial_config.threads = 1;
  vector_index::elkan_kmeans_config parallel_config = serial_config;
  parallel_config.threads = 8;

  vector_index::elkan_kmeans_result serial_result;
  vector_index::elkan_kmeans_result parallel_result;
  std::string error;
  ASSERT_TRUE(vector_index::run_elkan_kmeans(data.data(), kRows, serial_config,
                                             &serial_result, &error));
  ASSERT_TRUE(vector_index::run_elkan_kmeans(
      data.data(), kRows, parallel_config, &parallel_result, &error));
  EXPECT_EQ(serial_result.assignments, parallel_result.assignments);
  EXPECT_EQ(serial_result.centers, parallel_result.centers);
  EXPECT_EQ(serial_result.packed_distance_calls,
            parallel_result.packed_distance_calls);
  EXPECT_EQ(serial_result.centroid_scan_kernel,
            parallel_result.centroid_scan_kernel);
}

TEST(VectorElkanKmeansTest, RejectsTooFewRows) {
  const std::vector<float> data = {1.0F, 2.0F, 3.0F, 4.0F};
  const vector_index::elkan_kmeans_config config = make_config(2, 3);

  vector_index::elkan_kmeans_result result;
  std::string error;
  EXPECT_FALSE(vector_index::run_elkan_kmeans(data.data(), 2, config, &result,
                                              &error));
  EXPECT_EQ("row_count is smaller than center_count", error);
}

TEST(VectorElkanKmeansTest, RejectsInvalidConfigAndOutputArguments) {
  const std::vector<float> data = {1.0F, 2.0F, 3.0F, 4.0F};
  std::string error;

  EXPECT_FALSE(vector_index::run_elkan_kmeans(
      data.data(), 2, make_config(2, 1), nullptr, &error));
  EXPECT_EQ("result is null", error);

  vector_index::elkan_kmeans_result result;
  EXPECT_FALSE(vector_index::run_elkan_kmeans(
      data.data(), 2, make_config(0, 1), &result, &error));
  EXPECT_EQ("dimension is zero", error);

  EXPECT_FALSE(vector_index::run_elkan_kmeans(
      data.data(), 2, make_config(2, 0), &result, &error));
  EXPECT_EQ("center_count is zero", error);

  vector_index::elkan_kmeans_config config = make_config(2, 1);
  config.max_iterations = 0;
  EXPECT_FALSE(vector_index::run_elkan_kmeans(data.data(), 2, config, &result,
                                              &error));
  EXPECT_EQ("max_iterations is zero", error);

  EXPECT_FALSE(vector_index::run_elkan_kmeans(nullptr, 2, make_config(2, 1),
                                              &result, &error));
  EXPECT_EQ("data is null", error);
}

TEST(VectorElkanKmeansTest, RejectsEmptyInputWhenCentersAreRequested) {
  vector_index::elkan_kmeans_config config = make_config(2, 1);
  vector_index::elkan_kmeans_result result;
  std::string error;

  EXPECT_FALSE(vector_index::run_elkan_kmeans(nullptr, 0, config, &result,
                                              &error));
  EXPECT_EQ("row_count is smaller than center_count", error);
}

}  // namespace vector_elkan_kmeans_unittest
