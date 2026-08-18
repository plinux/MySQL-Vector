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

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "extra/vector/diskann-offline-adapter-cpp/mysql_vector_diskann_offline_checked_io.h"
#include "sql/vector/vector_diskann_artifact_layout.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_diskann_artifact_layout_unittest {
namespace {

TEST(VectorDiskannArtifactLayoutTest,
     OfflineAdapterPayloadLayoutRejectsOverflowBeforeAllocation) {
  mysql_vector_diskann_offline::checked_payload_layout layout;
  ASSERT_TRUE(mysql_vector_diskann_offline::make_checked_payload_layout(
      3, sizeof(uint64_t), sizeof(uint64_t), &layout));
  EXPECT_EQ(3U, layout.element_count);
  EXPECT_EQ(3U * sizeof(uint64_t), layout.payload_bytes);
  EXPECT_EQ(4U * sizeof(uint64_t), layout.file_bytes);

  EXPECT_FALSE(mysql_vector_diskann_offline::make_checked_payload_layout(
      1, 0, 0, &layout));
  EXPECT_FALSE(mysql_vector_diskann_offline::make_checked_payload_layout(
      1, sizeof(uint64_t), sizeof(uint64_t), nullptr));
  EXPECT_FALSE(mysql_vector_diskann_offline::make_checked_payload_layout(
      std::numeric_limits<uint64_t>::max(), sizeof(uint64_t), sizeof(uint64_t),
      &layout));
  EXPECT_FALSE(mysql_vector_diskann_offline::make_checked_payload_layout(
      static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) /
              sizeof(uint64_t) +
          1,
      sizeof(uint64_t), sizeof(uint64_t), &layout));
}

void write_bin_header(const std::string &path, uint32_t rows,
                      uint32_t columns) {
  std::ofstream file(path, std::ios::out | std::ios::binary);
  ASSERT_TRUE(file.is_open()) << path;
  file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char *>(&columns), sizeof(columns));
}

void truncate_file(const std::string &path) {
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.is_open()) << path;
}

void write_u32_bin_file(const std::string &path,
                        const std::vector<uint32_t> &values) {
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.is_open()) << path;
  const uint32_t rows = static_cast<uint32_t>(values.size());
  const uint32_t columns = 1;
  file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char *>(&columns), sizeof(columns));
  if (!values.empty()) {
    file.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(uint32_t)));
  }
  ASSERT_TRUE(file.good());
}

void write_manifest_text(const std::string &path, const std::string &payload) {
  std::ofstream file(path, std::ios::out | std::ios::trunc);
  ASSERT_TRUE(file.is_open()) << path;
  file << payload;
  ASSERT_TRUE(file.good());
}

std::string read_manifest_text(const std::string &path) {
  std::ifstream file(path, std::ios::in);
  EXPECT_TRUE(file.is_open()) << path;
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

std::string replace_manifest_line(std::string payload, const std::string &key,
                                  const std::string &replacement) {
  const size_t start = payload.find(key + '\t');
  if (start == std::string::npos) {
    ADD_FAILURE() << "missing manifest key: " << key;
    return payload;
  }
  const size_t end = payload.find('\n', start);
  if (end == std::string::npos) {
    ADD_FAILURE() << "unterminated manifest key: " << key;
    return payload;
  }
  payload.replace(start, end - start + 1, replacement);
  return payload;
}

void rewrite_first_payload_byte(const std::string &path, uint8_t value) {
  std::fstream file(path,
                    std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(file.is_open()) << path;
  file.seekp(sizeof(uint32_t) + sizeof(uint32_t));
  file.write(reinterpret_cast<const char *>(&value), sizeof(value));
  ASSERT_TRUE(file.good());
}

std::string test_prefix(const char *name) {
  static const vector_gunit::ScopedTempDirectory root(
      "vector_diskann_artifact_layout");
  EXPECT_TRUE(root.valid()) << root.error();
  return (root.path() / name).string();
}

void write_valid_artifacts(
    const vector_index::diskann_pq_artifact_paths &paths,
    const vector_index::diskann_pq_artifact_metadata &metadata) {
  vector_index::diskann_pq_artifact_payload payload;
  payload.pivots.assign(
      static_cast<size_t>(metadata.centroid_count) * metadata.dimension, 1.0F);
  payload.compressed_codes.assign(
      static_cast<size_t>(metadata.row_count) * metadata.pq_chunks, 7);
  payload.centroid.assign(metadata.dimension, 0.5F);
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

}  // namespace

TEST(VectorDiskannArtifactLayoutTest, MakesDiskAnnCompatiblePaths) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      "/tmp/mysql_vector_idx", &paths));

  EXPECT_EQ("/tmp/mysql_vector_idx_pq_pivots.bin", paths.pivot_path);
  EXPECT_EQ("/tmp/mysql_vector_idx_pq_compressed.bin", paths.compressed_path);
  EXPECT_EQ("/tmp/mysql_vector_idx_pq_pivots.bin_centroid.bin",
            paths.centroid_path);
  EXPECT_EQ("/tmp/mysql_vector_idx_pq_pivots.bin_chunk_offsets.bin",
            paths.chunk_offsets_path);
  EXPECT_EQ("/tmp/mysql_vector_idx_docid_ordinal.bin",
            paths.docid_ordinal_path);
  EXPECT_EQ("/tmp/mysql_vector_idx_disk.index", paths.disk_index_path);
  EXPECT_EQ("/tmp/mysql_vector_idx_disk.index_medoids.bin",
            paths.disk_index_medoids_path);
  EXPECT_EQ("/tmp/mysql_vector_idx_disk.index_centroids.bin",
            paths.disk_index_centroids_path);
  EXPECT_EQ("/tmp/mysql_vector_idx_disk.index_pq_pivots.bin",
            paths.disk_index_pq_pivots_path);
  EXPECT_EQ("/tmp/mysql_vector_idx_sample_data.bin", paths.sample_data_path);
  EXPECT_EQ("/tmp/mysql_vector_idx_cached_nodes.bin", paths.cached_nodes_path);
  EXPECT_EQ("/tmp/mysql_vector_idx_native_pq_bridge_manifest",
            paths.artifact_manifest_path);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsInvalidPathArguments) {
  vector_index::diskann_pq_artifact_paths paths;
  EXPECT_FALSE(vector_index::make_diskann_pq_artifact_paths("", &paths));
  EXPECT_FALSE(
      vector_index::make_diskann_pq_artifact_paths("/tmp/ignored", nullptr));
}

TEST(VectorDiskannArtifactLayoutTest, RejectsInvalidMetadata) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("invalid_metadata"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 1;
  metadata.dimension = 2;
  metadata.pq_chunks = 1;
  metadata.centroid_count = 2;

  std::string error;
  metadata.row_count = 0;
  EXPECT_FALSE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error));
  EXPECT_EQ("row_count is zero", error);

  metadata.row_count = 1;
  metadata.dimension = 0;
  EXPECT_FALSE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error));
  EXPECT_EQ("dimension is zero", error);

  metadata.dimension = 2;
  metadata.pq_chunks = 0;
  EXPECT_FALSE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error));
  EXPECT_EQ("pq_chunks is zero", error);

  metadata.pq_chunks = 1;
  metadata.centroid_count = 0;
  EXPECT_FALSE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error));
  EXPECT_EQ("centroid_count is zero", error);
}

TEST(VectorDiskannArtifactLayoutTest,
     RejectsOversizedRowsAndAllowsNullErrorSink) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("oversized_validate"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  metadata.dimension = 2;
  metadata.pq_chunks = 1;
  metadata.centroid_count = 2;

  EXPECT_FALSE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, nullptr));

  std::vector<float> values;
  EXPECT_FALSE(vector_index::read_diskann_pq_float_artifact(
      paths.pivot_path, std::numeric_limits<uint32_t>::max(), 2, &values,
      nullptr));
}

TEST(VectorDiskannArtifactLayoutTest, RejectsMissingArtifacts) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("missing"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 10;
  metadata.dimension = 8;
  metadata.pq_chunks = 2;

  std::string error;
  EXPECT_FALSE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error));
  EXPECT_EQ("pq_pivots artifact is missing or truncated", error);
}

TEST(VectorDiskannArtifactLayoutTest, ValidatesAllArtifactHeaders) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("valid"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 42;
  metadata.dimension = 128;
  metadata.pq_chunks = 16;
  metadata.centroid_count = 256;

  write_valid_artifacts(paths, metadata);

  std::string error;
  EXPECT_TRUE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error))
      << error;
  EXPECT_TRUE(error.empty());
}

TEST(VectorDiskannArtifactLayoutTest, StreamsCompressedArtifactBlocks) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("streaming_compressed"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 4;
  metadata.dimension = 3;
  metadata.pq_chunks = 2;
  metadata.centroid_count = 4;

  vector_index::diskann_pq_artifact_payload payload;
  payload.pivots.assign(
      static_cast<size_t>(metadata.centroid_count) * metadata.dimension, 1.0F);
  payload.centroid.assign(metadata.dimension, 0.5F);
  payload.chunk_offsets.assign({0, 1, 3});
  payload.docid_ordinals.assign({0, 1, 2, 3});

  std::string error;
  ASSERT_TRUE(vector_index::write_diskann_pq_static_artifacts(
      paths, metadata, payload, &error))
      << error;

  vector_index::diskann_compressed_artifact_writer writer;
  ASSERT_TRUE(writer.open(paths.compressed_path,
                          static_cast<uint32_t>(metadata.row_count),
                          metadata.pq_chunks, &error))
      << error;
  const std::vector<uint8_t> first_block{1, 2, 3, 4};
  const std::vector<uint8_t> second_block{5, 6, 7, 8};
  EXPECT_TRUE(writer.write_block(first_block.data(), 2, &error)) << error;
  EXPECT_TRUE(writer.write_block(second_block.data(), 2, &error)) << error;
  EXPECT_TRUE(writer.close(&error)) << error;

  EXPECT_TRUE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error))
      << error;
}

TEST(VectorDiskannArtifactLayoutTest, RejectsShortStreamingCompressedArtifact) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("streaming_short_compressed"), &paths));

  std::string error;
  vector_index::diskann_compressed_artifact_writer writer;
  ASSERT_TRUE(writer.open(paths.compressed_path, 2, 2, &error)) << error;
  const std::vector<uint8_t> block{1, 2};
  EXPECT_TRUE(writer.write_block(block.data(), 1, &error)) << error;
  EXPECT_FALSE(writer.close(&error));
  EXPECT_EQ("pq_compressed artifact row count mismatch", error);
}

TEST(VectorDiskannArtifactLayoutTest,
     StreamingWriterRejectsInvalidLifecycleAndBounds) {
  const std::string path = test_prefix("streaming_writer_bounds.bin");
  vector_index::diskann_compressed_artifact_writer writer;
  std::string error;
  const uint8_t code = 1;

  EXPECT_FALSE(writer.write_block(&code, 1, &error));
  EXPECT_EQ("pq_compressed artifact writer is not open", error);
  EXPECT_FALSE(writer.close(&error));
  EXPECT_EQ("pq_compressed artifact writer is not open", error);
  EXPECT_FALSE(writer.open("", 1, 1, &error));
  EXPECT_EQ("pq_compressed artifact path is empty", error);
  EXPECT_FALSE(writer.open(path, 0, 1, &error));
  EXPECT_EQ("pq_compressed artifact header is invalid", error);
  EXPECT_FALSE(writer.open(path, 1, 0, &error));
  EXPECT_EQ("pq_compressed artifact header is invalid", error);

  ASSERT_TRUE(writer.open(path, 1, 1, &error));
  EXPECT_FALSE(writer.open(path, 1, 1, &error));
  EXPECT_EQ("pq_compressed artifact writer is already open", error);
  EXPECT_TRUE(writer.write_block(nullptr, 0, &error));
  EXPECT_FALSE(writer.write_block(nullptr, 1, &error));
  EXPECT_EQ("pq_compressed block payload is null", error);
  EXPECT_FALSE(writer.write_block(&code, 2, &error));
  EXPECT_EQ("pq_compressed artifact row count mismatch", error);
  EXPECT_TRUE(writer.write_block(&code, 1, &error));
  EXPECT_TRUE(writer.close(&error));
  EXPECT_FALSE(writer.close(nullptr));
}

TEST(VectorDiskannArtifactLayoutTest,
     StreamingWriterRejectsOversizedBlockAndUnwritablePath) {
  const std::string path = test_prefix("streaming_writer_oversized.bin");
  vector_index::diskann_compressed_artifact_writer writer;
  const uint8_t code = 1;
  std::string error;

  EXPECT_FALSE(writer.open(test_prefix("missing_parent") + "/data.bin", 1, 1,
                           &error));
  EXPECT_EQ("pq_compressed artifact is not writable", error);

  ASSERT_TRUE(writer.open(path, std::numeric_limits<uint32_t>::max(),
                          std::numeric_limits<uint32_t>::max(), &error));
  EXPECT_FALSE(writer.write_block(&code,
                                  std::numeric_limits<uint32_t>::max(),
                                  &error));
  EXPECT_EQ("pq_compressed payload too large", error);
  EXPECT_FALSE(writer.close(&error));
}

TEST(VectorDiskannArtifactLayoutTest, RejectsPivotDimensionMismatch) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("pivot_mismatch"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 42;
  metadata.dimension = 128;
  metadata.pq_chunks = 16;
  metadata.centroid_count = 256;
  write_valid_artifacts(paths, metadata);
  write_bin_header(paths.pivot_path, metadata.centroid_count,
                   metadata.dimension + 1);

  std::string error;
  EXPECT_FALSE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error));
  EXPECT_EQ("pq_pivots artifact header mismatch", error);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsCompressedRowMismatch) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("compressed_mismatch"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 42;
  metadata.dimension = 128;
  metadata.pq_chunks = 16;
  metadata.centroid_count = 256;
  write_valid_artifacts(paths, metadata);
  write_bin_header(paths.compressed_path,
                   static_cast<uint32_t>(metadata.row_count + 1),
                   metadata.pq_chunks);

  std::string error;
  EXPECT_FALSE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error));
  EXPECT_EQ("pq_compressed artifact header mismatch", error);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsCompressedSizeMismatch) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("compressed_size_mismatch"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 42;
  metadata.dimension = 128;
  metadata.pq_chunks = 16;
  metadata.centroid_count = 256;
  write_valid_artifacts(paths, metadata);
  write_bin_header(paths.compressed_path,
                   static_cast<uint32_t>(metadata.row_count),
                   metadata.pq_chunks);

  std::string error;
  EXPECT_FALSE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error));
  EXPECT_EQ("pq_compressed artifact size mismatch", error);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsTrailingArtifactMismatches) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("trailing_mismatch"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 4;
  metadata.dimension = 3;
  metadata.pq_chunks = 2;
  metadata.centroid_count = 4;

  std::string error;
  write_valid_artifacts(paths, metadata);
  write_bin_header(paths.centroid_path, metadata.dimension + 1, 1);
  EXPECT_FALSE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error));
  EXPECT_EQ("centroid artifact header mismatch", error);

  write_valid_artifacts(paths, metadata);
  write_bin_header(paths.chunk_offsets_path, metadata.pq_chunks + 1, 1);
  EXPECT_FALSE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error));
  EXPECT_EQ("chunk_offsets artifact size mismatch", error);

  write_valid_artifacts(paths, metadata);
  write_bin_header(paths.docid_ordinal_path,
                   static_cast<uint32_t>(metadata.row_count + 1), 1);
  EXPECT_FALSE(
      vector_index::validate_diskann_pq_artifacts(paths, metadata, &error));
  EXPECT_EQ("docid_ordinal artifact header mismatch", error);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsOversizedCompressedRows) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("oversized_compressed"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  metadata.dimension = 2;
  metadata.pq_chunks = 1;
  metadata.centroid_count = 2;

  vector_index::diskann_pq_artifact_payload payload;
  payload.pivots.assign(4, 1.0F);
  payload.centroid.assign(2, 0.0F);
  payload.chunk_offsets.assign({0, 2});

  std::string error;
  EXPECT_FALSE(vector_index::write_diskann_pq_artifacts(paths, metadata,
                                                        payload, &error));
  EXPECT_EQ("pq_compressed artifact row count exceeds limit", error);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsPayloadSizeMismatches) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("payload_mismatch"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 2;
  metadata.dimension = 4;
  metadata.pq_chunks = 2;
  metadata.centroid_count = 3;

  vector_index::diskann_pq_artifact_payload payload;
  payload.pivots.assign(1, 1.0F);
  payload.compressed_codes.assign(4, 7);
  payload.centroid.assign(4, 0.5F);
  payload.chunk_offsets.assign({0, 2, 4});
  payload.docid_ordinals.assign({0, 1});

  std::string error;
  EXPECT_FALSE(vector_index::write_diskann_pq_artifacts(paths, metadata,
                                                        payload, &error));
  EXPECT_EQ("pq_pivots payload size mismatch", error);

  payload.pivots.assign(12, 1.0F);
  payload.compressed_codes.assign(1, 7);
  EXPECT_FALSE(vector_index::write_diskann_pq_artifacts(paths, metadata,
                                                        payload, &error));
  EXPECT_EQ("pq_compressed payload size mismatch", error);

  payload.compressed_codes.assign(4, 7);
  payload.centroid.assign(1, 0.5F);
  EXPECT_FALSE(vector_index::write_diskann_pq_artifacts(paths, metadata,
                                                        payload, &error));
  EXPECT_EQ("centroid payload size mismatch", error);

  payload.centroid.assign(4, 0.5F);
  payload.chunk_offsets.assign({0, 2});
  EXPECT_FALSE(vector_index::write_diskann_pq_artifacts(paths, metadata,
                                                        payload, &error));
  EXPECT_EQ("chunk_offsets payload size mismatch", error);

  payload.chunk_offsets.assign({0, 2, 4});
  payload.docid_ordinals.assign({0});
  EXPECT_FALSE(vector_index::write_diskann_pq_artifacts(paths, metadata,
                                                        payload, &error));
  EXPECT_EQ("docid_ordinal payload size mismatch", error);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsZeroRowArtifactAfterEmptyWrites) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("zero_row_write"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 0;
  metadata.dimension = 0;
  metadata.pq_chunks = 0;
  metadata.centroid_count = 0;

  vector_index::diskann_pq_artifact_payload payload;
  payload.chunk_offsets.assign({0});

  EXPECT_FALSE(vector_index::write_diskann_pq_artifacts(paths, metadata,
                                                        payload, nullptr));
}

TEST(VectorDiskannArtifactLayoutTest, RejectsUnwritableOutputPath) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("missing_parent") + "/idx", &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 1;
  metadata.dimension = 2;
  metadata.pq_chunks = 1;
  metadata.centroid_count = 1;

  vector_index::diskann_pq_artifact_payload payload;
  payload.pivots.assign(2, 1.0F);
  payload.compressed_codes.assign(1, 7);
  payload.centroid.assign(2, 0.5F);
  payload.chunk_offsets.assign({0, 2});
  payload.docid_ordinals.assign({0});

  std::string error;
  EXPECT_FALSE(vector_index::write_diskann_pq_artifacts(paths, metadata,
                                                        payload, &error));
  EXPECT_EQ("pq_pivots artifact is not writable", error);
}

TEST(VectorDiskannArtifactLayoutTest, ReadsBackFloatArtifactPayload) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("readback"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 4;
  metadata.dimension = 3;
  metadata.pq_chunks = 2;
  metadata.centroid_count = 4;
  write_valid_artifacts(paths, metadata);

  std::vector<float> values;
  std::string error;
  ASSERT_TRUE(vector_index::read_diskann_pq_float_artifact(
      paths.pivot_path, metadata.centroid_count, metadata.dimension, &values,
      &error))
      << error;
  ASSERT_EQ(static_cast<size_t>(metadata.centroid_count * metadata.dimension),
            values.size());
  EXPECT_EQ(1.0F, values[0]);
}

TEST(VectorDiskannArtifactLayoutTest, ReconstructsOneNativePqVector) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("reconstruct_vector"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 2;
  metadata.dimension = 4;
  metadata.pq_chunks = 2;
  metadata.centroid_count = 2;

  vector_index::diskann_pq_artifact_payload payload;
  payload.pivots = {1.0F, 2.0F, 10.0F, 20.0F, 3.0F, 4.0F, 30.0F, 40.0F};
  payload.compressed_codes = {0, 1, 1, 0};
  payload.centroid = {0.5F, 0.5F, 0.5F, 0.5F};
  payload.chunk_offsets = {0, 2, 4};
  payload.docid_ordinals = {0, 1};
  std::string error;
  ASSERT_TRUE(vector_index::write_diskann_pq_artifacts(paths, metadata, payload,
                                                       &error))
      << error;

  std::vector<float> reconstructed;
  ASSERT_TRUE(vector_index::reconstruct_diskann_pq_vector(
      paths, metadata, 0, &reconstructed, &error))
      << error;
  EXPECT_EQ((std::vector<float>{1.5F, 2.5F, 30.5F, 40.5F}), reconstructed);

  EXPECT_FALSE(vector_index::reconstruct_diskann_pq_vector(
      paths, metadata, metadata.row_count, &reconstructed, &error));
  EXPECT_EQ("pq_compressed ordinal is out of range", error);

  rewrite_first_payload_byte(paths.compressed_path, 7);
  EXPECT_FALSE(vector_index::reconstruct_diskann_pq_vector(
      paths, metadata, 0, &reconstructed, &error));
  EXPECT_EQ("pq_compressed code exceeds centroid count", error);
}

TEST(VectorDiskannArtifactLayoutTest, WritesAndValidatesBridgeManifest) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("bridge_manifest"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 4;
  metadata.dimension = 3;
  metadata.pq_chunks = 2;
  metadata.centroid_count = 4;
  write_valid_artifacts(paths, metadata);
  write_u32_bin_file(paths.disk_index_medoids_path, {1});

  vector_index::diskann_pq_bridge_manifest manifest;
  std::string error;
  ASSERT_TRUE(vector_index::make_diskann_pq_bridge_manifest(
      paths, metadata, 48, 128, 2, true, false, &manifest, &error))
      << error;
  EXPECT_TRUE(manifest.artifacts_consumed);
  EXPECT_FALSE(manifest.official_pq_used);
  EXPECT_NE(0U, manifest.pivots_checksum);
  EXPECT_NE(0U, manifest.compressed_checksum);
  EXPECT_NE(0U, manifest.medoids_checksum);

  ASSERT_TRUE(
      vector_index::write_diskann_pq_bridge_manifest(paths, manifest, &error))
      << error;

  vector_index::diskann_pq_bridge_manifest read_manifest;
  ASSERT_TRUE(vector_index::read_diskann_pq_bridge_manifest(
      paths, &read_manifest, &error))
      << error;
  EXPECT_TRUE(vector_index::validate_diskann_pq_bridge_manifest(
      paths, metadata, read_manifest, &error))
      << error;
}

TEST(VectorDiskannArtifactLayoutTest, RejectsInvalidBridgeManifestArguments) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("bridge_manifest_invalid_args"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 4;
  metadata.dimension = 3;
  metadata.pq_chunks = 2;
  metadata.centroid_count = 4;
  write_valid_artifacts(paths, metadata);

  vector_index::diskann_pq_bridge_manifest manifest;
  std::string error;
  EXPECT_FALSE(vector_index::make_diskann_pq_bridge_manifest(
      paths, metadata, 48, 128, 2, true, false, nullptr, &error));
  EXPECT_EQ("manifest is null", error);

  vector_index::diskann_pq_artifact_paths empty_paths = paths;
  empty_paths.artifact_manifest_path.clear();
  EXPECT_FALSE(vector_index::write_diskann_pq_bridge_manifest(empty_paths,
                                                              manifest,
                                                              &error));
  EXPECT_EQ("artifact manifest path is empty", error);

  EXPECT_FALSE(vector_index::read_diskann_pq_bridge_manifest(paths, nullptr,
                                                             &error));
  EXPECT_EQ("manifest is null", error);

  EXPECT_FALSE(vector_index::read_diskann_pq_bridge_manifest(paths, &manifest,
                                                             &error));
  EXPECT_EQ("artifact manifest is missing", error);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsConsumedManifestWithoutMedoids) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("bridge_manifest_missing_medoids"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 4;
  metadata.dimension = 3;
  metadata.pq_chunks = 2;
  metadata.centroid_count = 4;
  write_valid_artifacts(paths, metadata);

  vector_index::diskann_pq_bridge_manifest manifest;
  std::string error;
  EXPECT_FALSE(vector_index::make_diskann_pq_bridge_manifest(
      paths, metadata, 48, 128, 2, true, false, &manifest, &error));
  EXPECT_EQ("medoids artifact is required when artifacts are consumed", error);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsMalformedBridgeManifestText) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("bridge_manifest_malformed_text"), &paths));

  vector_index::diskann_pq_bridge_manifest manifest;
  std::string error;
  write_manifest_text(paths.artifact_manifest_path, "bad-header\n");
  EXPECT_FALSE(vector_index::read_diskann_pq_bridge_manifest(paths, &manifest,
                                                             &error));
  EXPECT_EQ("artifact manifest header mismatch", error);

  const std::string header = "mysql-vector-diskann-pq-bridge-manifest-v1\n";
  write_manifest_text(paths.artifact_manifest_path, header + "malformed\n");
  EXPECT_FALSE(vector_index::read_diskann_pq_bridge_manifest(paths, &manifest,
                                                             &error));
  EXPECT_EQ("artifact manifest line is malformed", error);

  write_manifest_text(paths.artifact_manifest_path,
                      header + "row_count\tNaN\n");
  EXPECT_FALSE(vector_index::read_diskann_pq_bridge_manifest(paths, &manifest,
                                                             &error));
  EXPECT_EQ("artifact manifest value is malformed", error);

  write_manifest_text(paths.artifact_manifest_path,
                      header + "zero_mean\tmaybe\n");
  EXPECT_FALSE(vector_index::read_diskann_pq_bridge_manifest(paths, &manifest,
                                                             &error));

  write_manifest_text(paths.artifact_manifest_path,
                      header + "artifacts_consumed\tmaybe\n");
  EXPECT_FALSE(vector_index::read_diskann_pq_bridge_manifest(paths, &manifest,
                                                             &error));

  write_manifest_text(paths.artifact_manifest_path,
                      header + "official_pq_used\tmaybe\n");
  EXPECT_FALSE(vector_index::read_diskann_pq_bridge_manifest(paths, &manifest,
                                                             &error));
}

TEST(VectorDiskannArtifactLayoutTest,
     RejectsIncompleteDuplicateUnknownAndNarrowedManifestFields) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("bridge_manifest_strict_fields"), &paths));

  vector_index::diskann_pq_bridge_manifest written;
  written.row_count = 4;
  written.dimension = 3;
  written.pq_chunks = 2;
  written.centroid_count = 4;
  written.max_degree = 48;
  written.build_complexity = 128;
  written.cache_nodes = 2;
  written.pivots_checksum = 11;
  written.compressed_checksum = 12;
  written.medoids_checksum = 13;
  written.zero_mean = true;
  written.artifacts_consumed = true;
  written.official_pq_used = false;
  std::string error;
  ASSERT_TRUE(
      vector_index::write_diskann_pq_bridge_manifest(paths, written, &error));
  const std::string valid = read_manifest_text(paths.artifact_manifest_path);

  vector_index::diskann_pq_bridge_manifest parsed;
  write_manifest_text(paths.artifact_manifest_path, valid + "dimension\t3\n");
  EXPECT_FALSE(
      vector_index::read_diskann_pq_bridge_manifest(paths, &parsed, &error));
  EXPECT_EQ("artifact manifest field is duplicated: dimension", error);

  write_manifest_text(paths.artifact_manifest_path,
                      replace_manifest_line(valid, "cache_nodes", ""));
  EXPECT_FALSE(
      vector_index::read_diskann_pq_bridge_manifest(paths, &parsed, &error));
  EXPECT_EQ("artifact manifest fields are incomplete", error);

  write_manifest_text(paths.artifact_manifest_path,
                      valid + "unknown_field\t1\n");
  EXPECT_FALSE(
      vector_index::read_diskann_pq_bridge_manifest(paths, &parsed, &error));
  EXPECT_EQ("artifact manifest field is unknown: unknown_field", error);

  const std::vector<std::string> u32_fields{"dimension",        "pq_chunks",
                                            "centroid_count",   "max_degree",
                                            "build_complexity", "cache_nodes"};
  for (const std::string &field : u32_fields) {
    write_manifest_text(
        paths.artifact_manifest_path,
        replace_manifest_line(valid, field, field + "\t4294967296\n"));
    EXPECT_FALSE(
        vector_index::read_diskann_pq_bridge_manifest(paths, &parsed, &error))
        << field;
    EXPECT_EQ("artifact manifest value exceeds uint32 limit", error) << field;
  }

  write_manifest_text(
      paths.artifact_manifest_path,
      replace_manifest_line(valid, "row_count", "row_count\t-1\n"));
  EXPECT_FALSE(
      vector_index::read_diskann_pq_bridge_manifest(paths, &parsed, &error));
  EXPECT_EQ("artifact manifest value is malformed", error);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsBridgeManifestChecksumMismatch) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("bridge_manifest_checksum_mismatch"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 4;
  metadata.dimension = 3;
  metadata.pq_chunks = 2;
  metadata.centroid_count = 4;
  write_valid_artifacts(paths, metadata);
  write_u32_bin_file(paths.disk_index_medoids_path, {1});

  vector_index::diskann_pq_bridge_manifest manifest;
  std::string error;
  ASSERT_TRUE(vector_index::make_diskann_pq_bridge_manifest(
      paths, metadata, 48, 128, 2, true, false, &manifest, &error))
      << error;
  rewrite_first_payload_byte(paths.compressed_path, 8);
  EXPECT_FALSE(vector_index::validate_diskann_pq_bridge_manifest(
      paths, metadata, manifest, &error));
  EXPECT_EQ("pq_compressed checksum mismatch", error);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsBridgeManifestMetadataMismatch) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("bridge_manifest_metadata_mismatch"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 4;
  metadata.dimension = 3;
  metadata.pq_chunks = 2;
  metadata.centroid_count = 4;
  metadata.zero_mean = true;
  write_valid_artifacts(paths, metadata);
  write_u32_bin_file(paths.disk_index_medoids_path, {1});

  vector_index::diskann_pq_bridge_manifest manifest;
  std::string error;
  ASSERT_TRUE(vector_index::make_diskann_pq_bridge_manifest(
      paths, metadata, 48, 128, 2, true, false, &manifest, &error))
      << error;

  auto expect_metadata_mismatch =
      [&](vector_index::diskann_pq_bridge_manifest candidate) {
        EXPECT_FALSE(vector_index::validate_diskann_pq_bridge_manifest(
            paths, metadata, candidate, &error));
        EXPECT_EQ("artifact manifest metadata mismatch", error);
      };

  vector_index::diskann_pq_bridge_manifest candidate = manifest;
  candidate.row_count++;
  expect_metadata_mismatch(candidate);

  candidate = manifest;
  candidate.dimension++;
  expect_metadata_mismatch(candidate);

  candidate = manifest;
  candidate.pq_chunks++;
  expect_metadata_mismatch(candidate);

  candidate = manifest;
  candidate.centroid_count++;
  expect_metadata_mismatch(candidate);

  candidate = manifest;
  candidate.zero_mean = !candidate.zero_mean;
  expect_metadata_mismatch(candidate);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsBridgeManifestOtherChecksums) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("bridge_manifest_other_checksum_mismatch"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 4;
  metadata.dimension = 3;
  metadata.pq_chunks = 2;
  metadata.centroid_count = 4;
  write_valid_artifacts(paths, metadata);
  write_u32_bin_file(paths.disk_index_medoids_path, {1});

  vector_index::diskann_pq_bridge_manifest manifest;
  std::string error;
  ASSERT_TRUE(vector_index::make_diskann_pq_bridge_manifest(
      paths, metadata, 48, 128, 2, true, false, &manifest, &error))
      << error;

  vector_index::diskann_pq_bridge_manifest candidate = manifest;
  candidate.pivots_checksum++;
  EXPECT_FALSE(vector_index::validate_diskann_pq_bridge_manifest(
      paths, metadata, candidate, &error));
  EXPECT_EQ("pq_pivots checksum mismatch", error);

  candidate = manifest;
  candidate.medoids_checksum++;
  EXPECT_FALSE(vector_index::validate_diskann_pq_bridge_manifest(
      paths, metadata, candidate, &error));
  EXPECT_EQ("medoids checksum mismatch", error);
}

TEST(VectorDiskannArtifactLayoutTest,
     WritesUnconsumedManifestWithoutOptionalMedoids) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("bridge_manifest_unconsumed"), &paths));

  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count = 4;
  metadata.dimension = 3;
  metadata.pq_chunks = 2;
  metadata.centroid_count = 4;
  write_valid_artifacts(paths, metadata);

  vector_index::diskann_pq_bridge_manifest manifest;
  std::string error;
  ASSERT_TRUE(vector_index::make_diskann_pq_bridge_manifest(
      paths, metadata, 48, 128, 0, false, false, &manifest, &error))
      << error;
  EXPECT_EQ(0U, manifest.medoids_checksum);
  EXPECT_FALSE(manifest.artifacts_consumed);
  ASSERT_TRUE(
      vector_index::write_diskann_pq_bridge_manifest(paths, manifest, &error))
      << error;
  EXPECT_TRUE(vector_index::validate_diskann_pq_bridge_manifest(
      paths, metadata, manifest, &error))
      << error;
}

TEST(VectorDiskannArtifactLayoutTest,
     RejectsManifestWriteAndAtomicRenameFailures) {
  vector_index::diskann_pq_bridge_manifest manifest;
  std::string error;

  vector_index::diskann_pq_artifact_paths missing_parent;
  missing_parent.artifact_manifest_path =
      test_prefix("missing_manifest_parent") + "/manifest";
  EXPECT_FALSE(vector_index::write_diskann_pq_bridge_manifest(
      missing_parent, manifest, &error));
  EXPECT_EQ("artifact manifest is not writable", error);

  vector_index::diskann_pq_artifact_paths collision;
  collision.artifact_manifest_path = test_prefix("manifest_directory");
  std::error_code ec;
  std::filesystem::remove_all(collision.artifact_manifest_path, ec);
  ASSERT_TRUE(
      std::filesystem::create_directory(collision.artifact_manifest_path, ec));
  ASSERT_FALSE(ec);
  EXPECT_FALSE(vector_index::write_diskann_pq_bridge_manifest(
      collision, manifest, &error));
  EXPECT_EQ("artifact manifest rename failed", error);
  std::filesystem::remove_all(collision.artifact_manifest_path, ec);
}

TEST(VectorDiskannArtifactLayoutTest, ReadsManifestBooleanFields) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("bridge_manifest_boolean_values"), &paths));
  const std::string header =
      "mysql-vector-diskann-pq-bridge-manifest-v1\n";
  write_manifest_text(paths.artifact_manifest_path,
                      header +
                          "row_count\t4\n"
                          "dimension\t3\n"
                          "pq_chunks\t2\n"
                          "centroid_count\t4\n"
                          "max_degree\t48\n"
                          "build_complexity\t128\n"
                          "cache_nodes\t2\n"
                          "pivots_checksum\t11\n"
                          "compressed_checksum\t12\n"
                          "medoids_checksum\t13\n"
                          "zero_mean\t0\n"
                          "artifacts_consumed\t1\n"
                          "official_pq_used\t1\n");

  vector_index::diskann_pq_bridge_manifest manifest;
  std::string error;
  ASSERT_TRUE(vector_index::read_diskann_pq_bridge_manifest(
      paths, &manifest, &error))
      << error;
  EXPECT_EQ(4U, manifest.row_count);
  EXPECT_EQ(3U, manifest.dimension);
  EXPECT_EQ(2U, manifest.pq_chunks);
  EXPECT_EQ(4U, manifest.centroid_count);
  EXPECT_EQ(48U, manifest.max_degree);
  EXPECT_EQ(128U, manifest.build_complexity);
  EXPECT_EQ(2U, manifest.cache_nodes);
  EXPECT_EQ(11U, manifest.pivots_checksum);
  EXPECT_EQ(12U, manifest.compressed_checksum);
  EXPECT_EQ(13U, manifest.medoids_checksum);
  EXPECT_FALSE(manifest.zero_mean);
  EXPECT_TRUE(manifest.artifacts_consumed);
  EXPECT_TRUE(manifest.official_pq_used);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsManifestNumericBoundaries) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("bridge_manifest_numeric_boundaries"), &paths));
  const std::string header =
      "mysql-vector-diskann-pq-bridge-manifest-v1\n";
  vector_index::diskann_pq_bridge_manifest manifest;
  std::string error;

  write_manifest_text(paths.artifact_manifest_path,
                      header + "row_count\t1x\n");
  EXPECT_FALSE(vector_index::read_diskann_pq_bridge_manifest(
      paths, &manifest, &error));
  EXPECT_EQ("artifact manifest value is malformed", error);

  write_manifest_text(paths.artifact_manifest_path,
                      header + "row_count\t18446744073709551616\n");
  EXPECT_FALSE(vector_index::read_diskann_pq_bridge_manifest(
      paths, &manifest, &error));
  EXPECT_EQ("artifact manifest value is malformed", error);
}

TEST(VectorDiskannArtifactLayoutTest, RejectsOversizedStaticArtifactRows) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("oversized_static_rows"), &paths));
  vector_index::diskann_pq_artifact_metadata metadata;
  metadata.row_count =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  metadata.dimension = 1;
  metadata.pq_chunks = 1;
  metadata.centroid_count = 1;
  vector_index::diskann_pq_artifact_payload payload;
  std::string error;

  EXPECT_FALSE(vector_index::write_diskann_pq_static_artifacts(
      paths, metadata, payload, &error));
  EXPECT_EQ("docid_ordinal artifact row count exceeds limit", error);
}

TEST(VectorDiskannArtifactLayoutTest, ReadsEmptyFloatArtifactPayload) {
  const std::string path = test_prefix("empty_float_payload.bin");
  write_bin_header(path, 0, 0);

  std::vector<float> values = {1.0F};
  std::string error;
  ASSERT_TRUE(vector_index::read_diskann_pq_float_artifact(
      path, 0, 0, &values, &error))
      << error;
  EXPECT_TRUE(values.empty());
  EXPECT_TRUE(error.empty());
}

TEST(VectorDiskannArtifactLayoutTest, RejectsInvalidFloatArtifactReads) {
  vector_index::diskann_pq_artifact_paths paths;
  ASSERT_TRUE(vector_index::make_diskann_pq_artifact_paths(
      test_prefix("read_invalid"), &paths));

  std::vector<float> values = {1.0F};
  std::string error;
  EXPECT_FALSE(vector_index::read_diskann_pq_float_artifact(
      paths.pivot_path, 1, 2, nullptr, &error));
  EXPECT_EQ("values is null", error);

  EXPECT_FALSE(vector_index::read_diskann_pq_float_artifact(
      paths.pivot_path, 1, 2, &values, &error));
  EXPECT_TRUE(values.empty());
  EXPECT_EQ("float artifact is missing or truncated", error);

  write_bin_header(paths.pivot_path, 1, 2);
  EXPECT_FALSE(vector_index::read_diskann_pq_float_artifact(
      paths.pivot_path, 1, 2, &values, &error));
  EXPECT_EQ("float artifact size mismatch", error);

  truncate_file(paths.pivot_path);
  EXPECT_FALSE(vector_index::read_diskann_pq_float_artifact(
      paths.pivot_path, 1, 2, &values, &error));
  EXPECT_EQ("float artifact is missing or truncated", error);
}

}  // namespace vector_diskann_artifact_layout_unittest
