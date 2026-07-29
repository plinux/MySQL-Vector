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

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "sql/vector/vector_diskann_pq_runtime.h"

namespace vector_diskann_pq_runtime_unittest {
namespace {

void write_fbin_file(const std::string &path, uint32_t rows, uint32_t dimension,
                     const std::vector<float> &values) {
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.is_open()) << path;
  file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char *>(&dimension), sizeof(dimension));
  file.write(reinterpret_cast<const char *>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  ASSERT_TRUE(file.good());
}

void write_text_file(const std::string &path, const std::string &payload) {
  std::ofstream file(path, std::ios::out | std::ios::trunc);
  ASSERT_TRUE(file.is_open()) << path;
  file << payload;
  ASSERT_TRUE(file.good());
}

void write_truncated_file(const std::string &path, const std::string &payload) {
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.is_open()) << path;
  file.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  ASSERT_TRUE(file.good());
}

void append_byte(const std::string &path, char value) {
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::app);
  ASSERT_TRUE(file.is_open()) << path;
  file.write(&value, 1);
  ASSERT_TRUE(file.good());
}

std::string temp_path(const char *name) {
  return std::string(testing::TempDir()) + "/" + name;
}

std::string temp_prefix(const char *name) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_pq_runtime";
  std::error_code ec;
  std::filesystem::create_directories(root, ec);
  return root + "/" + name;
}

std::vector<uint8_t> read_compressed_codes(const std::string &path,
                                           uint32_t *rows, uint32_t *columns) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  EXPECT_TRUE(file.is_open()) << path;
  file.read(reinterpret_cast<char *>(rows), sizeof(*rows));
  file.read(reinterpret_cast<char *>(columns), sizeof(*columns));
  std::vector<uint8_t> values(static_cast<size_t>(*rows) * *columns, 0);
  file.read(reinterpret_cast<char *>(values.data()),
            static_cast<std::streamsize>(values.size()));
  EXPECT_TRUE(file.good());
  return values;
}

std::vector<uint8_t> read_file_bytes(const std::string &path) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  EXPECT_TRUE(file.is_open()) << path;
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(file),
                              std::istreambuf_iterator<char>());
}

std::vector<float> read_float_artifact(const std::string &path, uint32_t rows,
                                       uint32_t columns) {
  std::vector<float> values;
  std::string error;
  EXPECT_TRUE(vector_index::read_diskann_pq_float_artifact(path, rows, columns,
                                                           &values, &error))
      << error;
  return values;
}

void write_raw_manifest(
    const std::string &path, uint32_t dimension,
    const std::vector<std::pair<std::string, uint32_t>> &segments) {
  uint64_t total_rows = 0;
  for (const auto &segment : segments) total_rows += segment.second;

  std::ofstream file(path, std::ios::out | std::ios::trunc);
  ASSERT_TRUE(file.is_open()) << path;
  file << "mysql-vector-diskann-raw-manifest-v1\n"
       << "dimension\t" << dimension << '\n'
       << "count\t" << total_rows << '\n';
  for (size_t index = 0; index < segments.size(); ++index) {
    file << "segment\tunused-" << index << ".docids\t" << segments[index].first
         << '\t' << segments[index].second << '\n';
  }
  ASSERT_TRUE(file.good());
}

std::vector<float> normalize_rows(const std::vector<float> &values,
                                  uint32_t dimension) {
  std::vector<float> normalized(values);
  for (size_t offset = 0; offset < normalized.size(); offset += dimension) {
    float norm = std::numeric_limits<float>::epsilon();
    for (uint32_t dim = 0; dim < dimension; ++dim) {
      norm += normalized[offset + dim] * normalized[offset + dim];
    }
    norm = std::sqrt(norm);
    for (uint32_t dim = 0; dim < dimension; ++dim) {
      normalized[offset + dim] /= norm;
    }
  }
  return normalized;
}

vector_index::diskann_pq_runtime_config make_config(uint32_t dimension,
                                                    uint32_t pq_chunks) {
  vector_index::diskann_pq_runtime_config config;
  config.dimension = dimension;
  config.pq_chunks = pq_chunks;
  config.threads = 2;
  config.memory_budget_size = 512ULL * 1024ULL * 1024ULL;
  return config;
}

}  // namespace

TEST(VectorDiskannPqMemoryPlanTest, RejectsInvalidAndOverflowingInputs) {
  vector_index::diskann_pq_memory_plan_input input;
  input.rows = 64;
  input.training_rows = 64;
  input.dimension = 8;
  input.pq_chunks = 4;
  input.requested_threads = 4;

  std::string error;
  EXPECT_FALSE(
      vector_index::make_diskann_pq_memory_plan(input, nullptr, &error));
  EXPECT_EQ("native_pq_memory_plan is null", error);

  const auto expect_invalid = [&](auto mutate) {
    auto invalid = input;
    mutate(&invalid);
    vector_index::diskann_pq_memory_plan plan;
    EXPECT_FALSE(
        vector_index::make_diskann_pq_memory_plan(invalid, &plan, &error));
    EXPECT_EQ("native_pq_memory_estimate_overflow", error);
  };
  expect_invalid([](vector_index::diskann_pq_memory_plan_input *value) {
    value->rows = 0;
  });
  expect_invalid([](vector_index::diskann_pq_memory_plan_input *value) {
    value->training_rows = 0;
  });
  expect_invalid([](vector_index::diskann_pq_memory_plan_input *value) {
    value->dimension = 0;
  });
  expect_invalid([](vector_index::diskann_pq_memory_plan_input *value) {
    value->pq_chunks = 0;
  });
  expect_invalid([](vector_index::diskann_pq_memory_plan_input *value) {
    value->pq_chunks = value->dimension + 1;
  });
  expect_invalid([](vector_index::diskann_pq_memory_plan_input *value) {
    value->training_rows = std::numeric_limits<uint64_t>::max();
  });
}

TEST(VectorDiskannPqMemoryPlanTest, AppliesBudgetInPriorityOrder) {
  vector_index::diskann_pq_memory_plan_input input;
  input.rows = 64;
  input.training_rows = 64;
  input.dimension = 8;
  input.pq_chunks = 4;
  input.requested_threads = 4;

  vector_index::diskann_pq_memory_plan plan;
  std::string error;
  ASSERT_TRUE(vector_index::make_diskann_pq_memory_plan(input, &plan, &error));
  EXPECT_EQ(4U, plan.worker_count);
  EXPECT_EQ(1000000U, plan.encode_block_rows);
  EXPECT_EQ("unlimited", plan.adjustment);
  EXPECT_GT(plan.estimated_size, 0U);

  input.memory_budget_size = 256ULL * 1024ULL * 1024ULL + 60ULL * 1024ULL;
  ASSERT_TRUE(vector_index::make_diskann_pq_memory_plan(input, &plan, &error));
  EXPECT_LT(plan.worker_count, input.requested_threads);
  EXPECT_EQ(1000000U, plan.encode_block_rows);
  EXPECT_EQ("reduced_threads", plan.adjustment);
  EXPECT_LE(plan.estimated_size, input.memory_budget_size);

  input.rows = 8192;
  input.training_rows = 8192;
  input.dimension = 64;
  input.pq_chunks = 8;
  input.requested_threads = 1;
  input.memory_budget_size = 256ULL * 1024ULL * 1024ULL + 12800ULL * 1024ULL;
  ASSERT_TRUE(vector_index::make_diskann_pq_memory_plan(input, &plan, &error));
  EXPECT_EQ(1U, plan.worker_count);
  EXPECT_LT(plan.encode_block_rows, 1000000U);
  EXPECT_EQ("reduced_block_rows", plan.adjustment);
  EXPECT_LE(plan.estimated_size, input.memory_budget_size);

  input.requested_threads = 4;
  ASSERT_TRUE(vector_index::make_diskann_pq_memory_plan(input, &plan, &error));
  EXPECT_EQ(1U, plan.worker_count);
  EXPECT_LT(plan.encode_block_rows, 1000000U);
  EXPECT_EQ("reduced_threads_and_block_rows", plan.adjustment);
  EXPECT_LE(plan.estimated_size, input.memory_budget_size);

  input.memory_budget_size = 1;
  EXPECT_FALSE(vector_index::make_diskann_pq_memory_plan(input, &plan, &error));
  EXPECT_EQ("budget_exceeded", plan.adjustment);
  EXPECT_EQ("native_pq_memory_budget_exceeded", error);
}

TEST(VectorDiskannPqRuntimeTest, TrainsChunksFromFbinPayload) {
  const std::string input = temp_path("vector_diskann_pq_runtime_payload.fbin");
  write_fbin_file(input, 8, 4,
                  {0.0F,  0.0F,  0.0F,  0.1F,  0.1F,  0.0F,  0.1F,  0.1F,
                   1.0F,  1.0F,  1.0F,  1.1F,  1.1F,  1.0F,  1.1F,  1.1F,
                   10.0F, 10.0F, 10.0F, 10.1F, 10.1F, 10.0F, 10.1F, 10.1F,
                   20.0F, 20.0F, 20.0F, 20.1F, 20.1F, 20.0F, 20.1F, 20.1F});

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  const std::string output_prefix = temp_prefix("single_payload_out");
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      make_config(4, 2), input.c_str(), output_prefix.c_str(), &result, &error))
      << error;
  EXPECT_EQ(8U, result.row_count);
  EXPECT_EQ(8U, result.train_rows);
  EXPECT_EQ(8U, result.compressed_rows);
  EXPECT_EQ(2U, result.pq_chunks);
  EXPECT_EQ(256U, result.centroid_count);
  EXPECT_GT(result.distance_calls, 0U);
  EXPECT_LE(result.train_ms, result.elapsed_ms);
  EXPECT_LE(result.encode_ms, result.elapsed_ms);
  EXPECT_LE(result.artifact_validation_ms, result.elapsed_ms);
  EXPECT_FALSE(result.centroid_scan_kernel.empty());
  EXPECT_TRUE(result.artifacts_written);
  EXPECT_TRUE(result.selected_path == "scalar_fallback" ||
              result.selected_path == "avx2" ||
              result.selected_path == "avx512");
  EXPECT_TRUE(result.artifacts.pivot_path.find("_pq_pivots.bin") !=
              std::string::npos);

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = result.row_count;
  metadata.dimension = 4;
  metadata.pq_chunks = result.pq_chunks;
  metadata.centroid_count = result.centroid_count;
  ASSERT_TRUE(vector_index::validate_diskann_pq_artifacts(result.artifacts,
                                                          metadata, &error))
      << error;

  uint32_t compressed_rows = 0;
  uint32_t compressed_columns = 0;
  const std::vector<uint8_t> codes = read_compressed_codes(
      result.artifacts.compressed_path, &compressed_rows, &compressed_columns);
  EXPECT_EQ(8U, compressed_rows);
  EXPECT_EQ(2U, compressed_columns);
  for (uint8_t code : codes) EXPECT_LE(code, 255U);
}

TEST(VectorDiskannPqRuntimeTest, SamplesTrainingRowsAndCompressesAllRows) {
  const std::string input =
      temp_path("vector_diskann_pq_runtime_training_sample.fbin");
  std::vector<float> values;
  constexpr uint32_t kRows = 8;
  constexpr uint32_t kDimension = 4;
  values.reserve(static_cast<size_t>(kRows) * kDimension);
  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t dim = 0; dim < kDimension; ++dim) {
      values.push_back(static_cast<float>(row * 10 + dim));
    }
  }
  write_fbin_file(input, kRows, kDimension, values);

  auto config = make_config(kDimension, 2);
  config.training_row_limit = 4;

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), temp_prefix("training_sample_out").c_str(),
      &result, &error))
      << error;

  EXPECT_EQ(kRows, result.row_count);
  EXPECT_EQ(4U, result.train_rows);
  EXPECT_EQ(kRows, result.compressed_rows);

  uint32_t compressed_rows = 0;
  uint32_t compressed_columns = 0;
  const std::vector<uint8_t> codes = read_compressed_codes(
      result.artifacts.compressed_path, &compressed_rows, &compressed_columns);
  EXPECT_EQ(kRows, compressed_rows);
  EXPECT_EQ(2U, compressed_columns);
  EXPECT_EQ(static_cast<size_t>(compressed_rows * compressed_columns),
            codes.size());
}

TEST(VectorDiskannPqRuntimeTest, EncodesRowsAfterSubtractingChunkCentroid) {
  const std::string input =
      temp_path("vector_diskann_pq_runtime_centroid_encoding.fbin");
  write_fbin_file(input, 2, 1, {100.0F, 101.0F});

  auto config = make_config(1, 1);
  config.threads = 1;

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), temp_prefix("centroid_encoding_out").c_str(),
      &result, &error))
      << error;

  uint32_t compressed_rows = 0;
  uint32_t compressed_columns = 0;
  const std::vector<uint8_t> codes = read_compressed_codes(
      result.artifacts.compressed_path, &compressed_rows, &compressed_columns);
  ASSERT_EQ(2U, compressed_rows);
  ASSERT_EQ(1U, compressed_columns);
  ASSERT_EQ(2U, codes.size());
  EXPECT_EQ(0U, codes[0]);
  EXPECT_EQ(1U, codes[1]);
}

TEST(VectorDiskannPqRuntimeTest, BuildsUncenteredArtifactsWhenRequested) {
  const std::string input =
      temp_path("vector_diskann_pq_runtime_uncentered.fbin");
  write_fbin_file(input, 2, 1, {100.0F, 101.0F});

  auto config = make_config(1, 1);
  config.threads = 1;
  config.zero_mean = false;

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), temp_prefix("uncentered_out").c_str(), &result,
      &error))
      << error;

  const std::vector<float> centroid =
      read_float_artifact(result.artifacts.centroid_path, 1, 1);
  ASSERT_EQ(1U, centroid.size());
  EXPECT_FLOAT_EQ(0.0F, centroid[0]);

  const std::vector<float> pivots =
      read_float_artifact(result.artifacts.pivot_path, 256, 1);
  ASSERT_EQ(256U, pivots.size());
  EXPECT_TRUE((pivots[0] == 100.0F && pivots[1] == 101.0F) ||
              (pivots[0] == 101.0F && pivots[1] == 100.0F));
}

TEST(VectorDiskannPqRuntimeTest,
     ReusesCodesWhenDiskPqChunksMatchMemoryPqChunks) {
  const std::string input =
      temp_path("vector_diskann_pq_runtime_disk_reuse.fbin");
  write_fbin_file(input, 2, 1, {100.0F, 101.0F});

  auto config = make_config(1, 1);
  config.threads = 1;
  config.disk_pq_chunks = 1;

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), temp_prefix("disk_reuse_out").c_str(), &result,
      &error))
      << error;
  ASSERT_TRUE(result.disk_artifacts_written);

  const std::vector<float> memory_centroid =
      read_float_artifact(result.artifacts.centroid_path, 1, 1);
  const std::vector<float> disk_centroid =
      read_float_artifact(result.disk_artifacts.centroid_path, 1, 1);
  ASSERT_EQ(1U, memory_centroid.size());
  ASSERT_EQ(1U, disk_centroid.size());
  EXPECT_FLOAT_EQ(100.5F, memory_centroid[0]);
  EXPECT_FLOAT_EQ(0.0F, disk_centroid[0]);

  const std::vector<float> memory_pivots =
      read_float_artifact(result.artifacts.pivot_path, 256, 1);
  const std::vector<float> disk_pivots =
      read_float_artifact(result.disk_artifacts.pivot_path, 256, 1);
  ASSERT_EQ(memory_pivots.size(), disk_pivots.size());
  for (size_t center = 0; center < memory_pivots.size(); ++center) {
    EXPECT_FLOAT_EQ(memory_pivots[center] + memory_centroid[0],
                    disk_pivots[center]);
  }
  EXPECT_EQ(read_file_bytes(result.artifacts.compressed_path),
            read_file_bytes(result.disk_artifacts.compressed_path));
}

TEST(VectorDiskannPqRuntimeTest,
     TrainsIndependentDiskArtifactsForDifferentChunkLayout) {
  const std::string input =
      temp_path("vector_diskann_pq_runtime_disk_independent.fbin");
  write_fbin_file(input, 4, 4,
                  {10.0F, 11.0F, 12.0F, 13.0F, 20.0F, 21.0F, 22.0F, 23.0F,
                   30.0F, 31.0F, 32.0F, 33.0F, 40.0F, 41.0F, 42.0F, 43.0F});

  auto config = make_config(4, 2);
  config.threads = 1;
  config.disk_pq_chunks = 4;

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), temp_prefix("disk_independent_out").c_str(),
      &result, &error))
      << error;
  ASSERT_TRUE(result.disk_artifacts_written);

  uint32_t rows = 0;
  uint32_t columns = 0;
  (void)read_compressed_codes(result.disk_artifacts.compressed_path, &rows,
                              &columns);
  EXPECT_EQ(4U, rows);
  EXPECT_EQ(4U, columns);

  const std::vector<float> disk_centroid =
      read_float_artifact(result.disk_artifacts.centroid_path, 4, 1);
  ASSERT_EQ(4U, disk_centroid.size());
  for (const float value : disk_centroid) EXPECT_FLOAT_EQ(0.0F, value);
}

TEST(VectorDiskannPqRuntimeTest, RejectsInputExceedingMemoryBudget) {
  const std::string input = temp_path("vector_diskann_pq_runtime_budget.fbin");
  write_fbin_file(input, 8, 4,
                  {0.0F,  0.0F,  0.0F,  0.1F,  0.1F,  0.0F,  0.1F,  0.1F,
                   1.0F,  1.0F,  1.0F,  1.1F,  1.1F,  1.0F,  1.1F,  1.1F,
                   10.0F, 10.0F, 10.0F, 10.1F, 10.1F, 10.0F, 10.1F, 10.1F,
                   20.0F, 20.0F, 20.0F, 20.1F, 20.1F, 20.0F, 20.1F, 20.1F});

  auto config = make_config(4, 2);
  config.memory_budget_size = 1;

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), temp_prefix("budget_out").c_str(), &result,
      &error));
  EXPECT_EQ("native_pq_memory_budget_exceeded", error);
  EXPECT_EQ(8U, result.row_count);
  EXPECT_FALSE(result.artifacts_written);
}

TEST(VectorDiskannPqRuntimeTest, ReducesWorkerCountInsideMemoryBudget) {
  const std::string input =
      temp_path("vector_diskann_pq_runtime_worker_budget.fbin");
  constexpr uint32_t kRows = 64;
  constexpr uint32_t kDimension = 8;
  std::vector<float> values;
  values.reserve(static_cast<size_t>(kRows) * kDimension);
  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t dim = 0; dim < kDimension; ++dim) {
      values.push_back(static_cast<float>((row * 11 + dim * 7) % 23));
    }
  }
  write_fbin_file(input, kRows, kDimension, values);

  auto config = make_config(kDimension, 4);
  config.threads = 4;
  config.memory_budget_size = 256ULL * 1024ULL * 1024ULL + 60ULL * 1024ULL;

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), temp_prefix("worker_budget_out").c_str(), &result,
      &error))
      << error << " estimate=" << result.memory_estimate_bytes;

  EXPECT_EQ(kRows, result.row_count);
  EXPECT_LT(result.effective_threads, config.threads);
  EXPECT_NE("none", result.memory_budget_adjustment);
  EXPECT_EQ("reduced_threads", result.memory_budget_adjustment);
  EXPECT_LE(result.memory_estimate_bytes, config.memory_budget_size);
  EXPECT_EQ(config.memory_budget_size, result.memory_budget_size);
  EXPECT_EQ(1000000U, result.encode_block_rows);
  EXPECT_TRUE(result.artifacts_written);
}

TEST(VectorDiskannPqRuntimeTest, ReducesBlockRowsInsideMemoryBudget) {
  const std::string input =
      temp_path("vector_diskann_pq_runtime_block_budget.fbin");
  constexpr uint32_t kRows = 8192;
  constexpr uint32_t kDimension = 64;
  std::vector<float> values;
  values.reserve(static_cast<size_t>(kRows) * kDimension);
  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t dim = 0; dim < kDimension; ++dim) {
      values.push_back(static_cast<float>((row * 13 + dim * 17) % 101) /
                       101.0F);
    }
  }
  write_fbin_file(input, kRows, kDimension, values);

  auto config = make_config(kDimension, 8);
  config.threads = 1;
  config.memory_budget_size = 256ULL * 1024ULL * 1024ULL + 12800ULL * 1024ULL;

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), temp_prefix("block_budget_out").c_str(), &result,
      &error))
      << error << " estimate=" << result.memory_estimate_bytes;

  EXPECT_EQ(kRows, result.row_count);
  EXPECT_EQ(1U, result.effective_threads);
  EXPECT_LT(result.encode_block_rows, 1000000U);
  EXPECT_EQ("reduced_block_rows", result.memory_budget_adjustment);
  EXPECT_LE(result.memory_estimate_bytes, config.memory_budget_size);
  EXPECT_TRUE(result.artifacts_written);
}

TEST(VectorDiskannPqRuntimeTest, ParallelChunkTrainingMatchesSerialArtifacts) {
  const std::string input =
      temp_path("vector_diskann_pq_runtime_parallel_equivalence.fbin");
  std::vector<float> values;
  constexpr uint32_t kRows = 32;
  constexpr uint32_t kDimension = 8;
  values.reserve(static_cast<size_t>(kRows) * kDimension);
  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t dim = 0; dim < kDimension; ++dim) {
      values.push_back(static_cast<float>((row * 17 + dim * 5) % 31) / 31.0F);
    }
  }
  write_fbin_file(input, kRows, kDimension, values);

  auto serial_config = make_config(kDimension, 4);
  serial_config.threads = 1;
  auto parallel_config = serial_config;
  parallel_config.threads = 4;

  vector_index::diskann_pq_runtime_result serial_result;
  vector_index::diskann_pq_runtime_result parallel_result;
  std::string error;
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      serial_config, input.c_str(), temp_prefix("serial_equivalence").c_str(),
      &serial_result, &error))
      << error;
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      parallel_config, input.c_str(),
      temp_prefix("parallel_equivalence").c_str(), &parallel_result, &error))
      << error;

  EXPECT_EQ(1U, serial_result.effective_threads);
  EXPECT_EQ(4U, parallel_result.effective_threads);
  EXPECT_EQ(serial_result.row_count, parallel_result.row_count);
  EXPECT_EQ(serial_result.pq_chunks, parallel_result.pq_chunks);
  EXPECT_EQ(serial_result.centroid_count, parallel_result.centroid_count);
  EXPECT_EQ(read_file_bytes(serial_result.artifacts.pivot_path),
            read_file_bytes(parallel_result.artifacts.pivot_path));
  EXPECT_EQ(read_file_bytes(serial_result.artifacts.compressed_path),
            read_file_bytes(parallel_result.artifacts.compressed_path));
  EXPECT_EQ(read_file_bytes(serial_result.artifacts.centroid_path),
            read_file_bytes(parallel_result.artifacts.centroid_path));
  EXPECT_EQ(read_file_bytes(serial_result.artifacts.chunk_offsets_path),
            read_file_bytes(parallel_result.artifacts.chunk_offsets_path));
  EXPECT_EQ(read_file_bytes(serial_result.artifacts.docid_ordinal_path),
            read_file_bytes(parallel_result.artifacts.docid_ordinal_path));
}

TEST(VectorDiskannPqRuntimeTest, TrainsContinuousArtifactsFromRawManifest) {
  const std::string first = temp_path("vector_diskann_pq_runtime_seg0.fbin");
  const std::string second = temp_path("vector_diskann_pq_runtime_seg1.fbin");
  const std::string manifest =
      temp_path("vector_diskann_pq_runtime_raw.manifest");
  write_fbin_file(first, 4, 4,
                  {0.0F, 0.0F, 0.0F, 0.1F, 0.1F, 0.0F, 0.1F, 0.1F, 1.0F, 1.0F,
                   1.0F, 1.1F, 1.1F, 1.0F, 1.1F, 1.1F});
  write_fbin_file(second, 4, 4,
                  {10.0F, 10.0F, 10.0F, 10.1F, 10.1F, 10.0F, 10.1F, 10.1F,
                   20.0F, 20.0F, 20.0F, 20.1F, 20.1F, 20.0F, 20.1F, 20.1F});
  write_raw_manifest(manifest, 4, {{first, 4}, {second, 4}});

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  const std::string output_prefix = temp_prefix("manifest_payload_out");
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      make_config(4, 2), manifest.c_str(), output_prefix.c_str(), &result,
      &error))
      << error;
  EXPECT_EQ(8U, result.row_count);
  EXPECT_EQ(8U, result.compressed_rows);

  uint32_t compressed_rows = 0;
  uint32_t compressed_columns = 0;
  const std::vector<uint8_t> codes = read_compressed_codes(
      result.artifacts.compressed_path, &compressed_rows, &compressed_columns);
  EXPECT_EQ(8U, compressed_rows);
  EXPECT_EQ(2U, compressed_columns);
  EXPECT_EQ(static_cast<size_t>(compressed_rows * compressed_columns),
            codes.size());
}

TEST(VectorDiskannPqRuntimeTest,
     CosineNormalizationMatchesPreNormalizedManifestInput) {
  constexpr uint32_t kRows = 8;
  constexpr uint32_t kDimension = 4;
  const std::vector<float> values = {
      0.0F, 0.0F, 0.0F, 0.0F, 3.0F, 4.0F, 0.0F, 0.0F, 1.0F, 2.0F, 3.0F,
      4.0F, 2.0F, 1.0F, 4.0F, 3.0F, 9.0F, 0.0F, 0.0F, 1.0F, 0.0F, 8.0F,
      1.0F, 0.0F, 5.0F, 5.0F, 5.0F, 5.0F, 7.0F, 2.0F, 6.0F, 3.0F};
  const std::string first = temp_path("cosine_runtime_segment0.fbin");
  const std::string second = temp_path("cosine_runtime_segment1.fbin");
  const std::string manifest = temp_path("cosine_runtime.manifest");
  const std::string normalized_input =
      temp_path("cosine_runtime_normalized.fbin");
  write_fbin_file(first, 4, kDimension,
                  std::vector<float>(values.begin(), values.begin() + 16));
  write_fbin_file(second, 4, kDimension,
                  std::vector<float>(values.begin() + 16, values.end()));
  write_raw_manifest(manifest, kDimension, {{first, 4}, {second, 4}});
  write_fbin_file(normalized_input, kRows, kDimension,
                  normalize_rows(values, kDimension));

  auto cosine_config = make_config(kDimension, 2);
  cosine_config.normalize_input = true;
  cosine_config.disk_pq_chunks = 1;
  auto normalized_config = make_config(kDimension, 2);
  normalized_config.disk_pq_chunks = 1;
  vector_index::diskann_pq_runtime_result cosine_result;
  vector_index::diskann_pq_runtime_result normalized_result;
  std::string error;
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      cosine_config, manifest.c_str(), temp_prefix("cosine_runtime").c_str(),
      &cosine_result, &error))
      << error;
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      normalized_config, normalized_input.c_str(),
      temp_prefix("normalized_runtime").c_str(), &normalized_result, &error))
      << error;

  EXPECT_EQ(read_file_bytes(normalized_result.artifacts.pivot_path),
            read_file_bytes(cosine_result.artifacts.pivot_path));
  EXPECT_EQ(read_file_bytes(normalized_result.artifacts.compressed_path),
            read_file_bytes(cosine_result.artifacts.compressed_path));
  EXPECT_EQ(read_file_bytes(normalized_result.artifacts.centroid_path),
            read_file_bytes(cosine_result.artifacts.centroid_path));
  EXPECT_EQ(read_file_bytes(normalized_result.artifacts.chunk_offsets_path),
            read_file_bytes(cosine_result.artifacts.chunk_offsets_path));
  EXPECT_EQ(read_file_bytes(normalized_result.artifacts.docid_ordinal_path),
            read_file_bytes(cosine_result.artifacts.docid_ordinal_path));
  EXPECT_EQ(read_file_bytes(normalized_result.disk_artifacts.pivot_path),
            read_file_bytes(cosine_result.disk_artifacts.pivot_path));
  EXPECT_EQ(read_file_bytes(normalized_result.disk_artifacts.compressed_path),
            read_file_bytes(cosine_result.disk_artifacts.compressed_path));
  EXPECT_EQ(read_file_bytes(normalized_result.disk_artifacts.centroid_path),
            read_file_bytes(cosine_result.disk_artifacts.centroid_path));
}

TEST(VectorDiskannPqRuntimeTest, TrainsUnevenChunksWithScalarPreference) {
  const std::string input =
      temp_path("vector_diskann_pq_runtime_uneven_chunks.fbin");
  write_fbin_file(
      input, 4, 3,
      {0.0F, 0.0F, 0.1F, 0.1F, 0.0F, 0.2F, 1.0F, 1.0F, 1.1F, 1.1F, 1.0F, 1.2F});

  auto config = make_config(3, 2);
  config.prefer_avx512 = false;

  vector_index::diskann_pq_runtime_result result;
  const std::string output_prefix = temp_prefix("uneven_chunks_out");
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), output_prefix.c_str(), &result, nullptr));
  EXPECT_EQ(4U, result.row_count);
  EXPECT_EQ(2U, result.pq_chunks);
  EXPECT_TRUE(result.artifacts_written);

  uint32_t compressed_rows = 0;
  uint32_t compressed_columns = 0;
  const std::vector<uint8_t> codes = read_compressed_codes(
      result.artifacts.compressed_path, &compressed_rows, &compressed_columns);
  EXPECT_EQ(4U, compressed_rows);
  EXPECT_EQ(2U, compressed_columns);
  EXPECT_EQ(static_cast<size_t>(compressed_rows * compressed_columns),
            codes.size());
}

TEST(VectorDiskannPqRuntimeTest, TrainsRawManifestWithRelativeSegmentPath) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_pq_runtime_relative";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  ASSERT_TRUE(std::filesystem::create_directories(root, ec));

  const std::string segment = root + "/relative_segment.fbin";
  const std::string manifest = root + "/relative_manifest.manifest";
  write_fbin_file(segment, 4, 4,
                  {0.0F, 0.0F, 0.0F, 0.1F, 0.1F, 0.0F, 0.1F, 0.1F, 1.0F, 1.0F,
                   1.0F, 1.1F, 1.1F, 1.0F, 1.1F, 1.1F});
  write_raw_manifest(manifest, 4, {{"relative_segment.fbin", 4}});

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  const std::string output_prefix = temp_prefix("relative_manifest_out");
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      make_config(4, 2), manifest.c_str(), output_prefix.c_str(), &result,
      &error))
      << error;
  EXPECT_EQ(4U, result.row_count);
  EXPECT_TRUE(result.artifacts_written);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorDiskannPqRuntimeTest, RejectsTooManyChunks) {
  const std::string input = temp_path("vector_diskann_pq_runtime_chunks.fbin");
  write_fbin_file(input, 1, 2, {1.0F, 2.0F});

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 3), input.c_str(), "vector_diskann_pq_runtime_bad",
      &result, &error));
  EXPECT_EQ("pq_chunks exceeds dimension", error);
}

TEST(VectorDiskannPqRuntimeTest, RejectsInvalidDiskPqConfiguration) {
  const std::string input =
      temp_path("vector_diskann_pq_runtime_disk_chunks.fbin");
  write_fbin_file(input, 1, 2, {1.0F, 2.0F});

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  auto config = make_config(2, 1);
  config.disk_pq_chunks = 3;
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), "vector_diskann_pq_runtime_bad_disk_chunks",
      &result, &error));
  EXPECT_EQ("disk_pq_chunks exceeds dimension", error);

  config.disk_pq_chunks = 1;
  config.zero_mean = false;
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), "vector_diskann_pq_runtime_nested_disk_build",
      &result, &error));
  EXPECT_EQ("disk_pq_chunks requires a zero-mean memory PQ build", error);
}

TEST(VectorDiskannPqRuntimeTest, RejectsInvalidTopLevelArguments) {
  const std::string input = temp_path("vector_diskann_pq_runtime_args.fbin");
  write_fbin_file(input, 1, 2, {1.0F, 2.0F});

  vector_index::diskann_pq_runtime_result result;
  std::string error;

  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), input.c_str(), "ignored", nullptr, &error));
  EXPECT_EQ("result is null", error);

  auto null_error_config = make_config(0, 1);
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      null_error_config, input.c_str(), "ignored", &result, nullptr));

  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), nullptr, "ignored", &result, &error));
  EXPECT_EQ("input_file is empty", error);

  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), input.c_str(), nullptr, &result, &error));
  EXPECT_EQ("output_prefix is empty", error);

  auto config = make_config(0, 1);
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), "ignored", &result, &error));
  EXPECT_EQ("dimension is zero", error);

  config = make_config(2, 0);
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, input.c_str(), "ignored", &result, &error));
  EXPECT_EQ("pq_chunks is zero", error);

  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), "", "ignored", &result, &error));
  EXPECT_EQ("input_file is empty", error);

  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), input.c_str(), "", &result, &error));
  EXPECT_EQ("output_prefix is empty", error);
}

TEST(VectorDiskannPqRuntimeTest, RejectsInvalidFbinInputs) {
  vector_index::diskann_pq_runtime_result result;
  std::string error;

  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), temp_path("missing_input.fbin").c_str(), "missing_out",
      &result, &error));
  EXPECT_EQ("input_file is not readable", error);

  const std::string truncated_header =
      temp_path("vector_diskann_pq_runtime_truncated_header.fbin");
  write_truncated_file(truncated_header, "abc");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), truncated_header.c_str(), "truncated_header_out",
      &result, &error));
  EXPECT_EQ("truncated fbin header", error);

  const std::string dimension_mismatch =
      temp_path("vector_diskann_pq_runtime_dimension_mismatch.fbin");
  write_fbin_file(dimension_mismatch, 1, 3, {1.0F, 2.0F, 3.0F});
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), dimension_mismatch.c_str(), "dimension_mismatch_out",
      &result, &error));
  EXPECT_EQ("fbin dimension mismatch", error);

  const std::string zero_rows =
      temp_path("vector_diskann_pq_runtime_zero_rows.fbin");
  write_fbin_file(zero_rows, 0, 2, {});
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), zero_rows.c_str(), "zero_rows_out", &result, &error));
  EXPECT_EQ("fbin row_count is zero", error);

  const std::string extra_payload =
      temp_path("vector_diskann_pq_runtime_extra_payload.fbin");
  write_fbin_file(extra_payload, 1, 2, {1.0F, 2.0F});
  append_byte(extra_payload, 'x');
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), extra_payload.c_str(),
      temp_prefix("extra_payload_out").c_str(), &result, &error));
  EXPECT_EQ("fbin file has extra payload", error);

  const std::string non_finite =
      temp_path("vector_diskann_pq_runtime_non_finite.fbin");
  write_fbin_file(non_finite, 1, 2,
                  {1.0F, std::numeric_limits<float>::quiet_NaN()});
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), non_finite.c_str(), "non_finite_out", &result,
      &error));
  EXPECT_EQ("fbin payload contains non-finite value", error);
}

TEST(VectorDiskannPqRuntimeTest, RejectsTruncatedPayload) {
  const std::string input =
      temp_path("vector_diskann_pq_runtime_truncated.fbin");
  write_fbin_file(input, 2, 2, {1.0F, 2.0F});

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), input.c_str(), "vector_diskann_pq_runtime_truncated",
      &result, &error));
  EXPECT_EQ("truncated fbin payload", error);
}

TEST(VectorDiskannPqRuntimeTest, RejectsInvalidRawManifests) {
  const std::string valid_segment =
      temp_path("vector_diskann_pq_runtime_manifest_segment.fbin");
  write_fbin_file(valid_segment, 1, 2, {1.0F, 2.0F});

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  const auto config = make_config(2, 1);
  const std::string header = "mysql-vector-diskann-raw-manifest-v1\n";

  const std::string bad_header =
      temp_path("vector_diskann_pq_runtime_bad_header.manifest");
  write_text_file(bad_header, "mysql-vector-diskann-raw-manifest-v1-extra\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, bad_header.c_str(), "bad_header_out", &result, &error));
  EXPECT_EQ("raw manifest header mismatch", error);

  const std::string missing_metadata =
      temp_path("vector_diskann_pq_runtime_missing_metadata.manifest");
  write_text_file(missing_metadata, header + "segment\tunused.docids\t" +
                                        valid_segment + "\t1\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, missing_metadata.c_str(), "missing_metadata_out", &result,
      &error));
  EXPECT_EQ("raw manifest is missing required metadata", error);

  const std::string dimension_mismatch =
      temp_path("vector_diskann_pq_runtime_manifest_dimension.manifest");
  write_text_file(dimension_mismatch,
                  header + "dimension\t3\ncount\t1\nsegment\tunused.docids\t" +
                      valid_segment + "\t1\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, dimension_mismatch.c_str(), "manifest_dimension_out", &result,
      &error));
  EXPECT_EQ("raw manifest dimension mismatch", error);

  const std::string invalid_count =
      temp_path("vector_diskann_pq_runtime_manifest_count.manifest");
  write_text_file(invalid_count,
                  header +
                      "dimension\t2\ncount\tNaN\nsegment\tunused.docids\t" +
                      valid_segment + "\t1\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, invalid_count.c_str(), "manifest_count_out", &result, &error));
  EXPECT_EQ("raw manifest count is invalid", error);

  const std::string invalid_dimension =
      temp_path("vector_diskann_pq_runtime_manifest_bad_dimension.manifest");
  write_text_file(invalid_dimension, header +
                                         "dimension\tNaN\ncount\t1\n"
                                         "segment\tunused.docids\t" +
                                         valid_segment + "\t1\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, invalid_dimension.c_str(), "manifest_bad_dimension_out", &result,
      &error));
  EXPECT_EQ("raw manifest dimension mismatch", error);

  const std::string trailing_dimension =
      temp_path("vector_diskann_pq_runtime_manifest_dimension_tail.manifest");
  write_text_file(trailing_dimension, header +
                                          "dimension\t2x\ncount\t1\n"
                                          "segment\tunused.docids\t" +
                                          valid_segment + "\t1\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, trailing_dimension.c_str(), "manifest_dimension_tail_out",
      &result, &error));
  EXPECT_EQ("raw manifest dimension mismatch", error);

  const std::string trailing_count =
      temp_path("vector_diskann_pq_runtime_manifest_count_tail.manifest");
  write_text_file(trailing_count, header +
                                      "dimension\t2\ncount\t1x\n"
                                      "segment\tunused.docids\t" +
                                      valid_segment + "\t1\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, trailing_count.c_str(), "manifest_count_tail_out", &result,
      &error));
  EXPECT_EQ("raw manifest count is invalid", error);

  const std::string invalid_segment_shape =
      temp_path("vector_diskann_pq_runtime_manifest_segment_shape.manifest");
  write_text_file(invalid_segment_shape,
                  header +
                      "dimension\t2\ncount\t1\nsegment\ttoo\tmany\t"
                      "fields\t1\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, invalid_segment_shape.c_str(), "manifest_segment_shape_out",
      &result, &error));
  EXPECT_EQ("raw manifest line is invalid", error);

  const std::string invalid_line =
      temp_path("vector_diskann_pq_runtime_manifest_line.manifest");
  write_text_file(invalid_line, header + "dimension\t2\ncount\t1\njunk\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, invalid_line.c_str(), "manifest_line_out", &result, &error));
  EXPECT_EQ("raw manifest line is invalid", error);

  const std::string invalid_segment_rows =
      temp_path("vector_diskann_pq_runtime_manifest_segment_rows.manifest");
  write_text_file(invalid_segment_rows,
                  header + "dimension\t2\ncount\t1\nsegment\tunused.docids\t" +
                      valid_segment + "\t0\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, invalid_segment_rows.c_str(), "manifest_segment_rows_out",
      &result, &error));
  EXPECT_EQ("raw manifest segment row count is invalid", error);

  const std::string invalid_segment_row_text =
      temp_path("vector_diskann_pq_runtime_manifest_segment_row_text.manifest");
  write_text_file(invalid_segment_row_text, header +
                                                "dimension\t2\ncount\t1\n"
                                                "segment\tunused.docids\t" +
                                                valid_segment + "\t1x\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, invalid_segment_row_text.c_str(), "manifest_segment_row_text_out",
      &result, &error));
  EXPECT_EQ("raw manifest segment row count is invalid", error);

  const std::string count_mismatch =
      temp_path("vector_diskann_pq_runtime_manifest_count_mismatch.manifest");
  write_text_file(count_mismatch,
                  header + "dimension\t2\ncount\t2\nsegment\tunused.docids\t" +
                      valid_segment + "\t1\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, count_mismatch.c_str(), "manifest_count_mismatch_out", &result,
      &error));
  EXPECT_EQ("raw manifest count mismatch", error);

  const std::string no_segment =
      temp_path("vector_diskann_pq_runtime_manifest_no_segment.manifest");
  write_text_file(no_segment, header + "dimension\t2\ncount\t0\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, no_segment.c_str(), "manifest_no_segment_out", &result, &error));
  EXPECT_EQ("raw manifest has no segment", error);

  const std::string blank_lines =
      temp_path("vector_diskann_pq_runtime_manifest_blank_lines.manifest");
  write_text_file(blank_lines, header +
                                   "\ndimension\t2\n\ncount\t1\n\n"
                                   "segment\tunused.docids\t" +
                                   valid_segment + "\t1\n\n");
  const std::string blank_lines_output =
      temp_prefix("manifest_blank_lines_out");
  ASSERT_TRUE(vector_index::build_diskann_pq_runtime(
      config, blank_lines.c_str(), blank_lines_output.c_str(), &result, &error))
      << error;
  EXPECT_EQ(1U, result.row_count);

  const std::string row_overflow =
      temp_path("vector_diskann_pq_runtime_manifest_row_overflow.manifest");
  write_text_file(row_overflow,
                  header + "dimension\t2\ncount\t" +
                      std::to_string(std::numeric_limits<uint64_t>::max()) +
                      "\nsegment\tunused0.docids\t" + valid_segment + "\t" +
                      std::to_string(std::numeric_limits<uint64_t>::max()) +
                      "\nsegment\tunused1.docids\t" + valid_segment + "\t1\n");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      config, row_overflow.c_str(), "manifest_row_overflow_out", &result,
      &error));
  EXPECT_EQ("raw manifest row count overflow", error);
}

TEST(VectorDiskannPqRuntimeTest, RejectsManifestSegmentMismatches) {
  const std::string short_segment =
      temp_path("vector_diskann_pq_runtime_manifest_short_segment.fbin");
  write_fbin_file(short_segment, 1, 2, {1.0F, 2.0F});
  const std::string short_manifest =
      temp_path("vector_diskann_pq_runtime_manifest_short_segment.manifest");
  write_raw_manifest(short_manifest, 2, {{short_segment, 2}});

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), short_manifest.c_str(), "manifest_short_out", &result,
      &error));
  EXPECT_EQ("raw manifest segment row count mismatch", error);

  const std::string wrong_dimension =
      temp_path("vector_diskann_pq_runtime_manifest_wrong_dimension.fbin");
  write_fbin_file(wrong_dimension, 1, 3, {1.0F, 2.0F, 3.0F});
  const std::string wrong_dimension_manifest =
      temp_path("vector_diskann_pq_runtime_manifest_wrong_dimension.manifest");
  write_raw_manifest(wrong_dimension_manifest, 2, {{wrong_dimension, 1}});
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), wrong_dimension_manifest.c_str(),
      "manifest_wrong_dimension_out", &result, &error));
  EXPECT_EQ("fbin dimension mismatch", error);
}

TEST(VectorDiskannPqRuntimeTest, RejectsArtifactOutputWriteFailure) {
  const std::string input =
      temp_path("vector_diskann_pq_runtime_unwritable_output.fbin");
  write_fbin_file(input, 2, 2, {1.0F, 2.0F, 3.0F, 4.0F});

  vector_index::diskann_pq_runtime_result result;
  std::string error;
  const std::string output_prefix =
      temp_path("vector_diskann_pq_runtime_missing_parent/out");
  EXPECT_FALSE(vector_index::build_diskann_pq_runtime(
      make_config(2, 1), input.c_str(), output_prefix.c_str(), &result,
      &error));
  EXPECT_EQ("pq_pivots artifact is not writable", error);
  EXPECT_FALSE(result.artifacts_written);
}

TEST(VectorDiskannPqRuntimeTest,
     FbinReaderRejectsStructuralDriftAtEveryBuildStage) {
  std::string error;
  uint64_t rows = 0;

  const std::string valid = temp_path("pq_reader_valid.fbin");
  write_fbin_file(valid, 2, 2, {1.0F, 2.0F, 3.0F, 4.0F});
  EXPECT_FALSE(vector_index::read_diskann_fbin_header_for_testing(
      valid, 2, nullptr, &error));
  ASSERT_TRUE(vector_index::read_diskann_fbin_header_for_testing(
      valid, 2, &rows, &error));
  EXPECT_EQ(2U, rows);

  const std::string truncated_header = temp_path("pq_reader_header.fbin");
  write_truncated_file(truncated_header, "abc");
  EXPECT_FALSE(vector_index::read_diskann_fbin_header_for_testing(
      truncated_header, 2, &rows, &error));
  EXPECT_EQ("truncated fbin header", error);

  const std::string wrong_dimension = temp_path("pq_reader_dimension.fbin");
  write_fbin_file(wrong_dimension, 1, 3, {1.0F, 2.0F, 3.0F});
  EXPECT_FALSE(vector_index::read_diskann_fbin_header_for_testing(
      wrong_dimension, 2, &rows, &error));
  EXPECT_EQ("fbin dimension mismatch", error);

  const std::string zero_rows = temp_path("pq_reader_zero.fbin");
  write_fbin_file(zero_rows, 0, 2, {});
  EXPECT_FALSE(vector_index::read_diskann_fbin_header_for_testing(
      zero_rows, 2, &rows, &error));
  EXPECT_EQ("fbin row_count is zero", error);

  uint64_t visited_rows = 0;
  EXPECT_FALSE(vector_index::for_each_diskann_fbin_block_for_testing(
      valid, 2, 2, 0, true, &visited_rows, &error));
  EXPECT_EQ("fbin block row count is zero", error);
  EXPECT_FALSE(vector_index::for_each_diskann_fbin_block_for_testing(
      temp_path("pq_reader_missing.fbin"), 2, 2, 1, true, &visited_rows,
      &error));
  EXPECT_EQ("input_file is not readable", error);
  EXPECT_FALSE(vector_index::for_each_diskann_fbin_block_for_testing(
      valid, 2, 3, 1, true, &visited_rows, &error));
  EXPECT_EQ("test row count mismatch", error);

  const std::string truncated_payload = temp_path("pq_reader_payload.fbin");
  write_fbin_file(truncated_payload, 2, 2, {1.0F, 2.0F});
  EXPECT_FALSE(vector_index::for_each_diskann_fbin_block_for_testing(
      truncated_payload, 2, 2, 2, true, &visited_rows, &error));
  EXPECT_EQ("truncated fbin payload", error);

  const std::string non_finite = temp_path("pq_reader_non_finite.fbin");
  write_fbin_file(non_finite, 1, 2,
                  {1.0F, std::numeric_limits<float>::infinity()});
  EXPECT_FALSE(vector_index::for_each_diskann_fbin_block_for_testing(
      non_finite, 2, 1, 1, true, &visited_rows, &error));
  EXPECT_EQ("fbin payload contains non-finite value", error);

  const std::string extra_payload = temp_path("pq_reader_extra.fbin");
  write_fbin_file(extra_payload, 1, 2, {1.0F, 2.0F});
  append_byte(extra_payload, 'x');
  EXPECT_FALSE(vector_index::for_each_diskann_fbin_block_for_testing(
      extra_payload, 2, 1, 1, true, &visited_rows, &error));
  EXPECT_EQ("fbin file has extra payload", error);

  visited_rows = 0;
  EXPECT_FALSE(vector_index::for_each_diskann_fbin_block_for_testing(
      valid, 2, 2, 1, false, &visited_rows, &error));
  EXPECT_EQ(1U, visited_rows);
  visited_rows = 0;
  ASSERT_TRUE(vector_index::for_each_diskann_fbin_block_for_testing(
      valid, 2, 2, 1, true, &visited_rows, &error));
  EXPECT_EQ(2U, visited_rows);
}

TEST(VectorDiskannPqRuntimeTest,
     TrainingSampleAndInspectionRejectInvalidInputsIndependently) {
  const std::string input = temp_path("pq_sample_valid.fbin");
  write_fbin_file(input, 2, 2, {1.0F, 2.0F, 3.0F, 4.0F});
  std::vector<float> sample;
  std::string error;

  EXPECT_FALSE(vector_index::read_diskann_raw_training_sample_for_testing(
      input, 2, 2, 1, nullptr, &error));
  EXPECT_FALSE(vector_index::read_diskann_raw_training_sample_for_testing(
      input, 2, 0, 1, &sample, &error));
  EXPECT_FALSE(vector_index::read_diskann_raw_training_sample_for_testing(
      input, 2, 2, 0, &sample, &error));
  EXPECT_FALSE(vector_index::read_diskann_raw_training_sample_for_testing(
      input, 2, 2, 3, &sample, &error));
  ASSERT_TRUE(vector_index::read_diskann_raw_training_sample_for_testing(
      input, 2, 2, 2, &sample, &error));
  EXPECT_EQ(std::vector<float>({1.0F, 2.0F, 3.0F, 4.0F}), sample);

  uint64_t rows = 0;
  EXPECT_FALSE(vector_index::inspect_diskann_raw_input_rows_for_testing(
      input, 2, nullptr, &error));
  EXPECT_FALSE(vector_index::inspect_diskann_raw_input_rows_for_testing(
      temp_path("pq_inspect_missing.fbin"), 2, &rows, &error));
  EXPECT_EQ("input_file is not readable", error);
  ASSERT_TRUE(vector_index::inspect_diskann_raw_input_rows_for_testing(
      input, 2, &rows, &error));
  EXPECT_EQ(2U, rows);
}

}  // namespace vector_diskann_pq_runtime_unittest
