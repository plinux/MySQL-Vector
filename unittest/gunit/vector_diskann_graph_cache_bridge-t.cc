/* Copyright (c) 2026, Oracle and/or its affiliates.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0,
   as published by the Free Software Foundation.

   This program is designed to work with certain software (including
   but not limited to OpenSSL) that is licensed under separate terms,
   as designated in a particular file or component or in included license
   documentation.  The authors of MySQL hereby grant you an additional
   permission to link the program and your derivative works with the
   separately licensed software that they have either included with the
   program or referenced in the documentation.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software Foundation,
   Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sql/vector/vector_diskann_graph_cache_bridge.h"

namespace vector_diskann_graph_cache_bridge_unittest {
namespace {

std::string test_prefix(const char *name) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_graph_cache_bridge";
  std::error_code ec;
  std::filesystem::create_directories(root, ec);
  return root + "/" + name;
}

void write_text_file(const std::string &path, const char *content) {
  std::ofstream file(path, std::ios::out | std::ios::trunc);
  ASSERT_TRUE(file.is_open()) << path;
  file << content;
  ASSERT_TRUE(file.good());
}

void overwrite_first_float(const std::string &path, float value) {
  std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(file.is_open()) << path;
  file.seekp(2 * sizeof(uint32_t), std::ios::beg);
  file.write(reinterpret_cast<const char *>(&value), sizeof(value));
  ASSERT_TRUE(file.good());
}

void write_valid_artifacts(
    const vector_index::diskann_pq_artifact_paths &paths,
    const vector_index::diskann_pq_artifact_metadata &metadata) {
  vector_index::diskann_pq_artifact_payload payload;
  payload.pivots.assign(
      static_cast<size_t>(metadata.centroid_count) * metadata.dimension, 1.0F);
  payload.compressed_codes.assign(
      static_cast<size_t>(metadata.row_count) * metadata.pq_chunks, 7);
  payload.centroid.assign(metadata.dimension, 0.0F);
  payload.chunk_offsets.assign(metadata.pq_chunks + 1, 0);
  for (uint32_t chunk = 0; chunk <= metadata.pq_chunks; ++chunk) {
    payload.chunk_offsets[chunk] = chunk;
  }
  payload.docid_ordinals.assign(static_cast<size_t>(metadata.row_count), 0);
  for (uint64_t row = 0; row < metadata.row_count; ++row) {
    payload.docid_ordinals[static_cast<size_t>(row)] = row;
  }
  std::string error;
  ASSERT_TRUE(vector_index::write_diskann_pq_artifacts(paths, metadata, payload,
                                                       &error))
      << error;
}

struct fake_build_context {
  bool called{false};
  bool succeed{true};
  bool write_error{true};
  uint32_t expected_dimension{4};
  uint32_t expected_pq_chunks{2};
  uint32_t expected_disk_pq_dims{0};
  std::string failure_message{"fake bridge failure"};
  std::string index_prefix;
  std::string manifest_path;
  std::string pq_pivots_path;
  std::string pq_compressed_path;
  std::string disk_pq_pivots_path;
  std::string disk_pq_compressed_path;
};

bool fake_bridge_build(const char *index_prefix, const char *manifest_path,
                       const char *pq_pivots_path,
                       const char *pq_compressed_path,
                       const char *disk_pq_pivots_path,
                       const char *disk_pq_compressed_path, uint32_t dimension,
                       int32_t metric_type, uint32_t max_degree,
                       uint32_t build_complexity, uint32_t build_threads,
                       double build_memory_gb, uint32_t pq_chunks,
                       uint32_t cache_nodes, uint32_t blas_threads,
                       uint32_t disk_pq_dims, bool accelerate_build,
                       bool shuffle_build, char *error_buffer,
                       size_t error_buffer_size, void *context) {
  fake_build_context *build_context =
      static_cast<fake_build_context *>(context);
  EXPECT_EQ(build_context->expected_dimension, dimension);
  EXPECT_EQ(0, metric_type);
  EXPECT_EQ(16U, max_degree);
  EXPECT_EQ(32U, build_complexity);
  EXPECT_EQ(2U, build_threads);
  EXPECT_EQ(0.0, build_memory_gb);
  EXPECT_EQ(build_context->expected_pq_chunks, pq_chunks);
  EXPECT_EQ(0U, cache_nodes);
  EXPECT_EQ(1U, blas_threads);
  EXPECT_EQ(build_context->expected_disk_pq_dims, disk_pq_dims);
  EXPECT_FALSE(accelerate_build);
  EXPECT_FALSE(shuffle_build);

  build_context->called = true;
  build_context->index_prefix = index_prefix;
  build_context->manifest_path = manifest_path;
  build_context->pq_pivots_path = pq_pivots_path;
  build_context->pq_compressed_path = pq_compressed_path;
  build_context->disk_pq_pivots_path =
      disk_pq_pivots_path == nullptr ? "" : disk_pq_pivots_path;
  build_context->disk_pq_compressed_path =
      disk_pq_compressed_path == nullptr ? "" : disk_pq_compressed_path;
  if (build_context->succeed) {
    vector_index::diskann_pq_artifact_paths paths;
    if (!vector_index::make_diskann_pq_artifact_paths(index_prefix, &paths)) {
      const char message[] = "fake path build failed";
      std::strncpy(error_buffer, message, error_buffer_size - 1);
      error_buffer[error_buffer_size - 1] = '\0';
      return false;
    }
    std::ofstream medoids(paths.disk_index_medoids_path,
                          std::ios::out | std::ios::binary | std::ios::trunc);
    if (!medoids.is_open()) {
      const char message[] = "fake medoids write failed";
      std::strncpy(error_buffer, message, error_buffer_size - 1);
      error_buffer[error_buffer_size - 1] = '\0';
      return false;
    }
    const uint32_t fake_medoids[] = {0, 1};
    medoids.write(reinterpret_cast<const char *>(fake_medoids),
                  sizeof(fake_medoids));
    return medoids.good();
  }
  if (build_context->write_error) {
    std::strncpy(error_buffer, build_context->failure_message.c_str(),
                 error_buffer_size - 1);
    error_buffer[error_buffer_size - 1] = '\0';
  }
  return false;
}

vector_index::diskann_graph_cache_bridge_config make_config(
    const char *name, fake_build_context *build_context = nullptr) {
  const std::string prefix = test_prefix(name);
  vector_index::diskann_graph_cache_bridge_config config;
  config.input_manifest_path = prefix + ".manifest";
  config.index_prefix = prefix;
  config.row_count = 4;
  config.dimension = 4;
  config.pq_chunks = 2;
  config.centroid_count = 4;
  config.max_degree = 16;
  config.build_complexity = 32;
  config.build_threads = 2;
  config.blas_threads = 1;
  if (build_context != nullptr) {
    config.build_fn = fake_bridge_build;
    config.build_context = build_context;
  }
  EXPECT_TRUE(vector_index::make_diskann_pq_artifact_paths(prefix,
                                                           &config.artifacts));
  write_text_file(config.input_manifest_path, "manifest\n");
  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = config.row_count;
  metadata.dimension = config.dimension;
  metadata.pq_chunks = config.pq_chunks;
  metadata.centroid_count = config.centroid_count;
  write_valid_artifacts(config.artifacts, metadata);
  return config;
}

}  // namespace

TEST(VectorDiskannGraphCacheBridgeTest, RejectsNullResult) {
  std::string error;
  EXPECT_FALSE(vector_index::build_diskann_graph_cache_from_native_pq(
      make_config("null_result"), nullptr, &error));
  EXPECT_EQ("result is null", error);
}

TEST(VectorDiskannGraphCacheBridgeTest, RejectsInvalidConfig) {
  vector_index::diskann_graph_cache_bridge_config config =
      make_config("invalid_config");
  config.index_prefix.clear();

  vector_index::diskann_graph_cache_bridge_result result;
  std::string error;
  EXPECT_FALSE(vector_index::build_diskann_graph_cache_from_native_pq(
      config, &result, &error));
  EXPECT_EQ("index prefix is empty", error);
  EXPECT_FALSE(result.artifacts_consumed);
  EXPECT_FALSE(result.official_pq_used);
}

TEST(VectorDiskannGraphCacheBridgeTest, RejectsEveryInvalidConfigField) {
  struct invalid_case {
    const char *name;
    const char *error;
    void (*mutate)(vector_index::diskann_graph_cache_bridge_config *);
  };
  const invalid_case cases[] = {
      {"empty_manifest", "input manifest path is empty",
       [](vector_index::diskann_graph_cache_bridge_config *config) {
         config->input_manifest_path.clear();
       }},
      {"zero_rows", "row_count is zero",
       [](vector_index::diskann_graph_cache_bridge_config *config) {
         config->row_count = 0;
       }},
      {"zero_dimension", "dimension is zero",
       [](vector_index::diskann_graph_cache_bridge_config *config) {
         config->dimension = 0;
       }},
      {"zero_pq_chunks", "pq_chunks is zero",
       [](vector_index::diskann_graph_cache_bridge_config *config) {
         config->pq_chunks = 0;
       }},
      {"pq_chunks_too_large", "pq_chunks exceeds dimension",
       [](vector_index::diskann_graph_cache_bridge_config *config) {
         config->pq_chunks = config->dimension + 1;
       }},
      {"disk_pq_dims_too_large", "disk_pq_dims exceeds dimension",
       [](vector_index::diskann_graph_cache_bridge_config *config) {
         config->disk_pq_dims = config->dimension + 1;
       }},
      {"zero_centroids", "centroid_count is zero",
       [](vector_index::diskann_graph_cache_bridge_config *config) {
         config->centroid_count = 0;
       }},
      {"zero_degree", "max_degree is zero",
       [](vector_index::diskann_graph_cache_bridge_config *config) {
         config->max_degree = 0;
       }},
      {"zero_complexity", "build_complexity is zero",
       [](vector_index::diskann_graph_cache_bridge_config *config) {
         config->build_complexity = 0;
       }},
  };

  for (const invalid_case &test_case : cases) {
    vector_index::diskann_graph_cache_bridge_config config =
        make_config(test_case.name);
    test_case.mutate(&config);

    vector_index::diskann_graph_cache_bridge_result result;
    std::string error;
    EXPECT_FALSE(vector_index::build_diskann_graph_cache_from_native_pq(
        config, &result, &error))
        << test_case.name;
    EXPECT_EQ(test_case.error, error) << test_case.name;
    EXPECT_FALSE(result.artifacts_consumed) << test_case.name;
    EXPECT_FALSE(result.official_pq_used) << test_case.name;
  }
}

TEST(VectorDiskannGraphCacheBridgeTest, RejectsInvalidArtifacts) {
  vector_index::diskann_graph_cache_bridge_config config =
      make_config("invalid_artifacts");
  std::filesystem::remove(config.artifacts.compressed_path);

  vector_index::diskann_graph_cache_bridge_result result;
  std::string error;
  EXPECT_FALSE(vector_index::build_diskann_graph_cache_from_native_pq(
      config, &result, &error));
  EXPECT_EQ("pq_compressed artifact is missing or truncated", error);
  EXPECT_FALSE(result.artifacts_consumed);
  EXPECT_FALSE(result.official_pq_used);
}

TEST(VectorDiskannGraphCacheBridgeTest, HandlesNullErrorOutput) {
  vector_index::diskann_graph_cache_bridge_result result;
  vector_index::diskann_graph_cache_bridge_config config =
      make_config("null_error_output");
  config.index_prefix.clear();

  EXPECT_FALSE(vector_index::build_diskann_graph_cache_from_native_pq(
      config, &result, nullptr));
  EXPECT_FALSE(result.artifacts_consumed);
}

TEST(VectorDiskannGraphCacheBridgeTest, RejectsMissingBuilder) {
  vector_index::diskann_graph_cache_bridge_result result;
  std::string error;
  EXPECT_FALSE(vector_index::build_diskann_graph_cache_from_native_pq(
      make_config("missing_builder"), &result, &error));
  EXPECT_EQ("native PQ graph/cache bridge is unavailable", error);
  EXPECT_TRUE(result.bridge_name.empty());
  EXPECT_FALSE(result.artifacts_consumed);
  EXPECT_FALSE(result.official_pq_used);
}

TEST(VectorDiskannGraphCacheBridgeTest, PropagatesBuilderFailure) {
  fake_build_context context;
  context.succeed = false;
  vector_index::diskann_graph_cache_bridge_result result;
  std::string error;
  vector_index::diskann_graph_cache_bridge_config config =
      make_config("builder_failure", &context);
  EXPECT_FALSE(vector_index::build_diskann_graph_cache_from_native_pq(
      config, &result, &error));
  EXPECT_TRUE(context.called);
  EXPECT_EQ("fake bridge failure", error);
  EXPECT_EQ("native_pq_graph_cache", result.bridge_name);
  EXPECT_FALSE(result.artifacts_consumed);
  EXPECT_FALSE(result.official_pq_used);
}

TEST(VectorDiskannGraphCacheBridgeTest, PropagatesDiskPqLayoutFailure) {
  fake_build_context context;
  context.succeed = false;
  context.failure_message = "disk_pq_layout_failed";
  vector_index::diskann_graph_cache_bridge_result result;
  std::string error;
  vector_index::diskann_graph_cache_bridge_config config =
      make_config("disk_pq_layout_failure", &context);
  EXPECT_FALSE(vector_index::build_diskann_graph_cache_from_native_pq(
      config, &result, &error));
  EXPECT_TRUE(context.called);
  EXPECT_EQ("disk_pq_layout_failed", error);
  EXPECT_EQ("native_pq_graph_cache", result.bridge_name);
  EXPECT_FALSE(result.artifacts_consumed);
  EXPECT_FALSE(result.official_pq_used);
}

TEST(VectorDiskannGraphCacheBridgeTest, UsesFallbackErrorWhenBuilderIsSilent) {
  fake_build_context context;
  context.succeed = false;
  context.write_error = false;
  vector_index::diskann_graph_cache_bridge_result result;
  std::string error;
  vector_index::diskann_graph_cache_bridge_config config =
      make_config("silent_builder_failure", &context);
  EXPECT_FALSE(vector_index::build_diskann_graph_cache_from_native_pq(
      config, &result, &error));
  EXPECT_TRUE(context.called);
  EXPECT_EQ("native PQ graph/cache bridge build failed", error);
  EXPECT_EQ("native_pq_graph_cache", result.bridge_name);
  EXPECT_FALSE(result.artifacts_consumed);
}

TEST(VectorDiskannGraphCacheBridgeTest, PassesDiskPqDimsToBuilder) {
  fake_build_context context;
  context.expected_dimension = 768;
  context.expected_pq_chunks = 128;
  context.expected_disk_pq_dims = 64;
  vector_index::diskann_graph_cache_bridge_result result;
  std::string error;
  vector_index::diskann_graph_cache_bridge_config config =
      make_config("builder_disk_pq_dims", &context);
  config.dimension = context.expected_dimension;
  config.pq_chunks = context.expected_pq_chunks;
  config.disk_pq_dims = context.expected_disk_pq_dims;

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = config.row_count;
  metadata.dimension = config.dimension;
  metadata.pq_chunks = config.pq_chunks;
  metadata.centroid_count = config.centroid_count;
  write_valid_artifacts(config.artifacts, metadata);
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      config.index_prefix + "_disk_pq", &config.disk_artifacts));
  metadata.pq_chunks = config.disk_pq_dims;
  metadata.zero_mean = false;
  write_valid_artifacts(config.disk_artifacts, metadata);

  EXPECT_TRUE(vector_index::build_diskann_graph_cache_from_native_pq(
      config, &result, &error))
      << error;
  EXPECT_TRUE(context.called);
  EXPECT_TRUE(result.artifacts_consumed);
  EXPECT_FALSE(result.official_pq_used);
  EXPECT_EQ("native_pq_graph_cache", result.bridge_name);
  EXPECT_EQ(config.disk_artifacts.pivot_path,
            context.disk_pq_pivots_path);
  EXPECT_EQ(config.disk_artifacts.compressed_path,
            context.disk_pq_compressed_path);
  vector_index::diskann_pq_bridge_manifest manifest;
  ASSERT_TRUE(vector_index::read_diskann_pq_bridge_manifest(config.artifacts,
                                                            &manifest, &error));
  EXPECT_EQ(config.pq_chunks, manifest.pq_chunks);
  EXPECT_TRUE(manifest.zero_mean);
}

TEST(VectorDiskannGraphCacheBridgeTest, RejectsMissingDiskPqArtifacts) {
  fake_build_context context;
  context.expected_disk_pq_dims = 2;
  vector_index::diskann_graph_cache_bridge_result result;
  std::string error;
  vector_index::diskann_graph_cache_bridge_config config =
      make_config("missing_disk_pq_artifacts", &context);
  config.disk_pq_dims = 2;

  EXPECT_FALSE(vector_index::build_diskann_graph_cache_from_native_pq(
      config, &result, &error));
  EXPECT_EQ("pq_pivots artifact is missing or truncated", error);
  EXPECT_FALSE(context.called);
  EXPECT_FALSE(result.artifacts_consumed);
}

TEST(VectorDiskannGraphCacheBridgeTest, RejectsCenteredDiskPqArtifacts) {
  fake_build_context context;
  context.expected_disk_pq_dims = 2;
  vector_index::diskann_graph_cache_bridge_result result;
  std::string error;
  vector_index::diskann_graph_cache_bridge_config config =
      make_config("centered_disk_pq_artifacts", &context);
  config.disk_pq_dims = 2;

  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      config.index_prefix + "_disk_pq", &config.disk_artifacts));
  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = config.row_count;
  metadata.dimension = config.dimension;
  metadata.pq_chunks = config.disk_pq_dims;
  metadata.centroid_count = config.centroid_count;
  metadata.zero_mean = false;
  write_valid_artifacts(config.disk_artifacts, metadata);
  overwrite_first_float(config.disk_artifacts.centroid_path, 1.0F);

  EXPECT_FALSE(vector_index::build_diskann_graph_cache_from_native_pq(
      config, &result, &error));
  EXPECT_EQ("disk PQ centroid is not zero", error);
  EXPECT_FALSE(context.called);
  EXPECT_FALSE(result.artifacts_consumed);
}

TEST(VectorDiskannGraphCacheBridgeTest, WritesManifestAfterBuilderSucceeds) {
  fake_build_context context;
  vector_index::diskann_graph_cache_bridge_result result;
  std::string error;
  vector_index::diskann_graph_cache_bridge_config config =
      make_config("builder_success", &context);

  EXPECT_TRUE(vector_index::build_diskann_graph_cache_from_native_pq(
      config, &result, &error))
      << error;
  EXPECT_TRUE(context.called);
  EXPECT_EQ(config.index_prefix, context.index_prefix);
  EXPECT_EQ(config.input_manifest_path, context.manifest_path);
  EXPECT_EQ(config.artifacts.pivot_path, context.pq_pivots_path);
  EXPECT_EQ(config.artifacts.compressed_path, context.pq_compressed_path);
  EXPECT_TRUE(result.artifacts_consumed);
  EXPECT_FALSE(result.official_pq_used);
  EXPECT_EQ("native_pq_graph_cache", result.bridge_name);
  EXPECT_TRUE(error.empty());

  vector_index::diskann_pq_bridge_manifest manifest;
  ASSERT_TRUE(vector_index::read_diskann_pq_bridge_manifest(config.artifacts,
                                                            &manifest, &error))
      << error;
  EXPECT_TRUE(manifest.artifacts_consumed);
  EXPECT_FALSE(manifest.official_pq_used);
  EXPECT_EQ(config.row_count, manifest.row_count);
  EXPECT_EQ(config.dimension, manifest.dimension);
  EXPECT_EQ(config.pq_chunks, manifest.pq_chunks);
}

TEST(VectorDiskannGraphCacheBridgeTest,
     AcceptsZeroMeanArtifactsForCosineGraphBuild) {
  fake_build_context context;
  vector_index::diskann_graph_cache_bridge_result result;
  std::string error;
  vector_index::diskann_graph_cache_bridge_config config =
      make_config("cosine_zero_mean", &context);
  config.metric = vector_index::metric_type::kCosine;

  EXPECT_TRUE(vector_index::build_diskann_graph_cache_from_native_pq(
      config, &result, &error))
      << error;
  EXPECT_TRUE(context.called);
  EXPECT_TRUE(result.artifacts_consumed);

  vector_index::diskann_pq_bridge_manifest manifest;
  ASSERT_TRUE(vector_index::read_diskann_pq_bridge_manifest(config.artifacts,
                                                            &manifest, &error))
      << error;
  EXPECT_TRUE(manifest.zero_mean);
}

}  // namespace vector_diskann_graph_cache_bridge_unittest
