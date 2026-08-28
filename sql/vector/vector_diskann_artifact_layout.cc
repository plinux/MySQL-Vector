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

#include "sql/vector/vector_diskann_artifact_layout.h"

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <type_traits>

namespace vector_index {
namespace {

void set_error(std::string *error, const std::string &message) {
  if (error != nullptr) *error = message;
}

bool read_bin_header(const std::string &path, uint32_t *rows,
                     uint32_t *columns) {
  if (rows == nullptr || columns == nullptr) return false;
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) return false;

  file.read(reinterpret_cast<char *>(rows), sizeof(*rows));
  file.read(reinterpret_cast<char *>(columns), sizeof(*columns));
  return file.good();
}

bool checksum_file(const std::string &path, uint64_t *checksum,
                   std::string *error) {
  if (checksum == nullptr) return false;
  *checksum = 1469598103934665603ULL;
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    set_error(error, "artifact is missing: " + path);
    return false;
  }

  char buffer[8192];
  while (file.good()) {
    file.read(buffer, sizeof(buffer));
    const std::streamsize bytes = file.gcount();
    for (std::streamsize pos = 0; pos < bytes; ++pos) {
      *checksum ^= static_cast<unsigned char>(buffer[pos]);
      *checksum *= 1099511628211ULL;
    }
  }
  if (!file.eof()) {
    set_error(error, "artifact read failed: " + path);
    return false;
  }
  return true;
}

bool optional_checksum_file(const std::string &path, uint64_t *checksum,
                            std::string *error) {
  if (checksum == nullptr) return false;
  *checksum = 0;
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) return true;
  return checksum_file(path, checksum, error);
}

bool multiply_u64(uint64_t lhs, uint64_t rhs, uint64_t *out) {
  if (out == nullptr) return false;
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    return false;
  }
  *out = lhs * rhs;
  return true;
}

bool expected_artifact_size(uint64_t rows, uint32_t columns,
                            size_t payload_element_size, uintmax_t *bytes) {
  if (bytes == nullptr) return false;
  uint64_t value_count = 0;
  if (!multiply_u64(rows, columns, &value_count)) return false;
  if (payload_element_size != 0 &&
      value_count > std::numeric_limits<uintmax_t>::max() /
                        payload_element_size) {
    return false;
  }
  const uintmax_t payload_bytes =
      static_cast<uintmax_t>(value_count) * payload_element_size;
  constexpr uintmax_t header_bytes = sizeof(uint32_t) + sizeof(uint32_t);
  if (payload_bytes > std::numeric_limits<uintmax_t>::max() - header_bytes) {
    return false;
  }
  *bytes = header_bytes + payload_bytes;
  return true;
}

bool check_file_size(const std::string &path, uint64_t rows, uint32_t columns,
                     size_t payload_element_size, const char *label,
                     std::string *error) {
  uintmax_t expected_size = 0;
  if (!expected_artifact_size(rows, columns, payload_element_size,
                              &expected_size)) {
    set_error(error, std::string(label) + " artifact size overflow");
    return false;
  }

  std::error_code ec;
  const uintmax_t actual_size = std::filesystem::file_size(path, ec);
  if (ec) {
    set_error(error, std::string(label) + " artifact is missing");
    return false;
  }
  if (actual_size != expected_size) {
    set_error(error, std::string(label) + " artifact size mismatch");
    return false;
  }
  return true;
}

bool check_header(const std::string &path, uint64_t expected_rows,
                  uint32_t expected_columns, size_t payload_element_size,
                  const char *label, std::string *error) {
  if (expected_rows > std::numeric_limits<uint32_t>::max()) {
    set_error(error, std::string(label) + " artifact row count exceeds limit");
    return false;
  }
  uint32_t rows = 0;
  uint32_t columns = 0;
  if (!read_bin_header(path, &rows, &columns)) {
    set_error(error, std::string(label) + " artifact is missing or truncated");
    return false;
  }
  if (rows != expected_rows || columns != expected_columns) {
    set_error(error, std::string(label) + " artifact header mismatch");
    return false;
  }
  return check_file_size(path, expected_rows, expected_columns,
                         payload_element_size, label, error);
}

template <typename T>
bool write_bin_artifact(const std::string &path, uint32_t rows,
                        uint32_t columns, const std::vector<T> &values,
                        const char *label, std::string *error) {
  static_assert(std::is_trivially_copyable<T>::value,
                "DiskANN artifact payload must be trivially copyable");
  const uint64_t expected_values =
      static_cast<uint64_t>(rows) * static_cast<uint64_t>(columns);
  if (values.size() != expected_values) {
    set_error(error, std::string(label) + " payload size mismatch");
    return false;
  }

  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.is_open()) {
    set_error(error, std::string(label) + " artifact is not writable");
    return false;
  }
  file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char *>(&columns), sizeof(columns));
  if (!values.empty()) {
    if (values.size() >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) /
            sizeof(T)) {
      set_error(error, std::string(label) + " payload too large");
      return false;
    }
    file.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
  }
  if (!file.good()) {
    set_error(error, std::string(label) + " artifact write failed");
    return false;
  }
  return true;
}

void write_manifest_bool(std::ostream &stream, const char *name, bool value) {
  stream << name << '\t' << (value ? 1 : 0) << '\n';
}

void write_manifest_uint(std::ostream &stream, const char *name,
                         uint64_t value) {
  stream << name << '\t' << value << '\n';
}

bool parse_manifest_bool(const std::string &value, bool *out) {
  if (out == nullptr) return false;
  if (value == "0") {
    *out = false;
    return true;
  }
  if (value == "1") {
    *out = true;
    return true;
  }
  return false;
}

bool parse_manifest_uint(const std::string &value, uint64_t *out) {
  if (out == nullptr || value.empty()) return false;
  for (char digit : value) {
    if (digit < '0' || digit > '9') return false;
  }
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (errno != 0 || end == value.c_str() || *end != '\0') return false;
  *out = static_cast<uint64_t>(parsed);
  return true;
}

enum class bridge_manifest_field : uint32_t {
  kRowCount,
  kDimension,
  kPqChunks,
  kCentroidCount,
  kMaxDegree,
  kBuildComplexity,
  kCacheNodes,
  kPivotsChecksum,
  kCompressedChecksum,
  kMedoidsChecksum,
  kZeroMean,
  kArtifactsConsumed,
  kOfficialPqUsed,
  kCount
};

bridge_manifest_field parse_bridge_manifest_field(const std::string &key) {
  if (key == "row_count") return bridge_manifest_field::kRowCount;
  if (key == "dimension") return bridge_manifest_field::kDimension;
  if (key == "pq_chunks") return bridge_manifest_field::kPqChunks;
  if (key == "centroid_count") return bridge_manifest_field::kCentroidCount;
  if (key == "max_degree") return bridge_manifest_field::kMaxDegree;
  if (key == "build_complexity") return bridge_manifest_field::kBuildComplexity;
  if (key == "cache_nodes") return bridge_manifest_field::kCacheNodes;
  if (key == "pivots_checksum") return bridge_manifest_field::kPivotsChecksum;
  if (key == "compressed_checksum")
    return bridge_manifest_field::kCompressedChecksum;
  if (key == "medoids_checksum") return bridge_manifest_field::kMedoidsChecksum;
  if (key == "zero_mean") return bridge_manifest_field::kZeroMean;
  if (key == "artifacts_consumed")
    return bridge_manifest_field::kArtifactsConsumed;
  if (key == "official_pq_used") return bridge_manifest_field::kOfficialPqUsed;
  return bridge_manifest_field::kCount;
}

bool parse_manifest_u32(const std::string &value, uint32_t *out,
                        std::string *error) {
  uint64_t parsed = 0;
  if (!parse_manifest_uint(value, &parsed)) {
    set_error(error, "artifact manifest value is malformed");
    return false;
  }
  if (parsed > std::numeric_limits<uint32_t>::max()) {
    set_error(error, "artifact manifest value exceeds uint32 limit");
    return false;
  }
  *out = static_cast<uint32_t>(parsed);
  return true;
}

}  // namespace

bool make_diskann_pq_artifact_paths(const std::string &index_prefix,
                                    diskann_pq_artifact_paths *paths) {
  if (paths == nullptr) return false;
  *paths = {};
  if (index_prefix.empty()) return false;

  paths->pivot_path = index_prefix + "_pq_pivots.bin";
  paths->compressed_path = index_prefix + "_pq_compressed.bin";
  paths->centroid_path = paths->pivot_path + "_centroid.bin";
  paths->chunk_offsets_path = paths->pivot_path + "_chunk_offsets.bin";
  paths->docid_ordinal_path = index_prefix + "_docid_ordinal.bin";
  paths->disk_index_path = index_prefix + "_disk.index";
  paths->disk_index_medoids_path = paths->disk_index_path + "_medoids.bin";
  paths->disk_index_centroids_path = paths->disk_index_path + "_centroids.bin";
  paths->disk_index_pq_pivots_path = paths->disk_index_path + "_pq_pivots.bin";
  paths->sample_data_path = index_prefix + "_sample_data.bin";
  paths->cached_nodes_path = index_prefix + "_cached_nodes.bin";
  paths->artifact_manifest_path = index_prefix + "_native_pq_bridge_manifest";
  return true;
}

bool diskann_compressed_artifact_writer::open(const std::string &path,
                                              uint32_t rows, uint32_t columns,
                                              std::string *error) {
  if (error != nullptr) error->clear();
  if (m_open) {
    set_error(error, "pq_compressed artifact writer is already open");
    return false;
  }
  if (path.empty()) {
    set_error(error, "pq_compressed artifact path is empty");
    return false;
  }
  if (rows == 0 || columns == 0) {
    set_error(error, "pq_compressed artifact header is invalid");
    return false;
  }

  m_file.open(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!m_file.is_open()) {
    set_error(error, "pq_compressed artifact is not writable");
    return false;
  }

  m_file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  m_file.write(reinterpret_cast<const char *>(&columns), sizeof(columns));
  if (!m_file.good()) {
    set_error(error, "pq_compressed artifact write failed");
    m_file.close();
    return false;
  }

  m_rows = rows;
  m_columns = columns;
  m_written_rows = 0;
  m_open = true;
  return true;
}

bool diskann_compressed_artifact_writer::write_block(const uint8_t *codes,
                                                     uint32_t block_rows,
                                                     std::string *error) {
  if (!m_open) {
    set_error(error, "pq_compressed artifact writer is not open");
    return false;
  }
  if (block_rows == 0) return true;
  if (codes == nullptr) {
    set_error(error, "pq_compressed block payload is null");
    return false;
  }
  if (m_written_rows + block_rows > m_rows) {
    set_error(error, "pq_compressed artifact row count mismatch");
    return false;
  }

  const uint64_t value_count =
      static_cast<uint64_t>(block_rows) * static_cast<uint64_t>(m_columns);
  if (value_count >
      static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    set_error(error, "pq_compressed payload too large");
    return false;
  }
  m_file.write(reinterpret_cast<const char *>(codes),
               static_cast<std::streamsize>(value_count));
  if (!m_file.good()) {
    set_error(error, "pq_compressed artifact write failed");
    return false;
  }
  m_written_rows += block_rows;
  return true;
}

bool diskann_compressed_artifact_writer::close(std::string *error) {
  if (!m_open) {
    set_error(error, "pq_compressed artifact writer is not open");
    return false;
  }
  if (m_written_rows != m_rows) {
    set_error(error, "pq_compressed artifact row count mismatch");
    m_file.close();
    m_open = false;
    return false;
  }
  m_file.close();
  if (m_file.fail()) {
    set_error(error, "pq_compressed artifact write failed");
    m_open = false;
    return false;
  }
  m_open = false;
  return true;
}

bool validate_diskann_pq_artifacts(
    const diskann_pq_artifact_paths &paths,
    const diskann_pq_artifact_metadata &metadata, std::string *error) {
  if (error != nullptr) error->clear();
  if (metadata.row_count == 0) {
    set_error(error, "row_count is zero");
    return false;
  }
  if (metadata.dimension == 0) {
    set_error(error, "dimension is zero");
    return false;
  }
  if (metadata.pq_chunks == 0) {
    set_error(error, "pq_chunks is zero");
    return false;
  }
  if (metadata.centroid_count == 0) {
    set_error(error, "centroid_count is zero");
    return false;
  }

  return check_header(paths.pivot_path, metadata.centroid_count,
                      metadata.dimension, sizeof(float), "pq_pivots", error) &&
         check_header(paths.compressed_path, metadata.row_count,
                      metadata.pq_chunks, sizeof(uint8_t), "pq_compressed",
                      error) &&
         check_header(paths.centroid_path, metadata.dimension, 1, sizeof(float),
                      "centroid", error) &&
         check_header(paths.chunk_offsets_path,
                      static_cast<uint64_t>(metadata.pq_chunks) + 1, 1,
                      sizeof(uint32_t), "chunk_offsets", error) &&
         check_header(paths.docid_ordinal_path, metadata.row_count, 1,
                      sizeof(uint64_t), "docid_ordinal", error);
}

bool write_diskann_pq_artifacts(
    const diskann_pq_artifact_paths &paths,
    const diskann_pq_artifact_metadata &metadata,
    const diskann_pq_artifact_payload &payload, std::string *error) {
  if (error != nullptr) error->clear();
  if (metadata.row_count > std::numeric_limits<uint32_t>::max()) {
    set_error(error, "pq_compressed artifact row count exceeds limit");
    return false;
  }
  if (!write_diskann_pq_static_artifacts(paths, metadata, payload, error))
    return false;
  const uint64_t expected_values =
      metadata.row_count * static_cast<uint64_t>(metadata.pq_chunks);
  if (payload.compressed_codes.size() != expected_values) {
    set_error(error, "pq_compressed payload size mismatch");
    return false;
  }

  diskann_compressed_artifact_writer writer;
  if (!writer.open(paths.compressed_path,
                   static_cast<uint32_t>(metadata.row_count),
                   metadata.pq_chunks, error)) {
    return false;
  }
  if (!writer.write_block(payload.compressed_codes.data(),
                          static_cast<uint32_t>(metadata.row_count), error)) {
    return false;
  }
  if (!writer.close(error)) {
    return false;
  }
  return validate_diskann_pq_artifacts(paths, metadata, error);
}

bool write_diskann_pq_static_artifacts(
    const diskann_pq_artifact_paths &paths,
    const diskann_pq_artifact_metadata &metadata,
    const diskann_pq_artifact_payload &payload, std::string *error) {
  if (error != nullptr) error->clear();
  if (metadata.row_count > std::numeric_limits<uint32_t>::max()) {
    set_error(error, "docid_ordinal artifact row count exceeds limit");
    return false;
  }
  if (!write_bin_artifact(paths.pivot_path, metadata.centroid_count,
                          metadata.dimension, payload.pivots, "pq_pivots",
                          error)) {
    return false;
  }
  if (!write_bin_artifact(paths.centroid_path, metadata.dimension, 1,
                          payload.centroid, "centroid", error)) {
    return false;
  }
  if (!write_bin_artifact(paths.chunk_offsets_path, metadata.pq_chunks + 1, 1,
                          payload.chunk_offsets, "chunk_offsets", error)) {
    return false;
  }
  if (!write_bin_artifact(paths.docid_ordinal_path,
                          static_cast<uint32_t>(metadata.row_count), 1,
                          payload.docid_ordinals, "docid_ordinal", error)) {
    return false;
  }
  return true;
}

bool read_diskann_pq_float_artifact(const std::string &path,
                                    uint32_t expected_rows,
                                    uint32_t expected_columns,
                                    std::vector<float> *values,
                                    std::string *error) {
  if (values == nullptr) {
    set_error(error, "values is null");
    return false;
  }
  values->clear();

  if (!check_header(path, expected_rows, expected_columns, sizeof(float),
                    "float", error)) {
    return false;
  }

  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    set_error(error, "float artifact is missing");
    return false;
  }
  uint32_t rows = 0;
  uint32_t columns = 0;
  file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char *>(&columns), sizeof(columns));
  const size_t value_count =
      static_cast<size_t>(expected_rows) * expected_columns;
  values->assign(value_count, 0.0F);
  if (value_count != 0) {
    file.read(reinterpret_cast<char *>(values->data()),
              static_cast<std::streamsize>(value_count * sizeof(float)));
  }
  if (!file.good()) {
    set_error(error, "float artifact payload is truncated");
    return false;
  }
  return true;
}

bool make_diskann_pq_bridge_manifest(
    const diskann_pq_artifact_paths &paths,
    const diskann_pq_artifact_metadata &metadata, uint32_t max_degree,
    uint32_t build_complexity, uint32_t cache_nodes, bool artifacts_consumed,
    bool official_pq_used, diskann_pq_bridge_manifest *manifest,
    std::string *error) {
  if (manifest == nullptr) {
    set_error(error, "manifest is null");
    return false;
  }
  if (!validate_diskann_pq_artifacts(paths, metadata, error)) return false;

  diskann_pq_bridge_manifest next;
  next.row_count = metadata.row_count;
  next.dimension = metadata.dimension;
  next.pq_chunks = metadata.pq_chunks;
  next.centroid_count = metadata.centroid_count;
  next.max_degree = max_degree;
  next.build_complexity = build_complexity;
  next.cache_nodes = cache_nodes;
  next.zero_mean = metadata.zero_mean;
  next.artifacts_consumed = artifacts_consumed;
  next.official_pq_used = official_pq_used;

  if (!checksum_file(paths.pivot_path, &next.pivots_checksum, error) ||
      !checksum_file(paths.compressed_path, &next.compressed_checksum, error)) {
    return false;
  }
  if (!optional_checksum_file(paths.disk_index_medoids_path,
                              &next.medoids_checksum, error)) {
    return false;
  }
  if (artifacts_consumed && next.medoids_checksum == 0) {
    set_error(error, "medoids artifact is required when artifacts are consumed");
    return false;
  }
  *manifest = next;
  if (error != nullptr) error->clear();
  return true;
}

bool write_diskann_pq_bridge_manifest(
    const diskann_pq_artifact_paths &paths,
    const diskann_pq_bridge_manifest &manifest, std::string *error) {
  if (paths.artifact_manifest_path.empty()) {
    set_error(error, "artifact manifest path is empty");
    return false;
  }

  const std::string tmp_path = paths.artifact_manifest_path + ".tmp";
  {
    std::ofstream file(tmp_path, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
      set_error(error, "artifact manifest is not writable");
      return false;
    }
    file << "mysql-vector-diskann-pq-bridge-manifest-v1\n";
    write_manifest_uint(file, "row_count", manifest.row_count);
    write_manifest_uint(file, "dimension", manifest.dimension);
    write_manifest_uint(file, "pq_chunks", manifest.pq_chunks);
    write_manifest_uint(file, "centroid_count", manifest.centroid_count);
    write_manifest_uint(file, "max_degree", manifest.max_degree);
    write_manifest_uint(file, "build_complexity", manifest.build_complexity);
    write_manifest_uint(file, "cache_nodes", manifest.cache_nodes);
    write_manifest_uint(file, "pivots_checksum", manifest.pivots_checksum);
    write_manifest_uint(file, "compressed_checksum",
                        manifest.compressed_checksum);
    write_manifest_uint(file, "medoids_checksum", manifest.medoids_checksum);
    write_manifest_bool(file, "zero_mean", manifest.zero_mean);
    write_manifest_bool(file, "artifacts_consumed",
                        manifest.artifacts_consumed);
    write_manifest_bool(file, "official_pq_used", manifest.official_pq_used);
    if (!file.good()) {
      set_error(error, "artifact manifest write failed");
      return false;
    }
  }

  std::error_code ec;
  std::filesystem::rename(tmp_path, paths.artifact_manifest_path, ec);
  if (ec) {
    std::filesystem::remove(tmp_path, ec);
    set_error(error, "artifact manifest rename failed");
    return false;
  }
  if (error != nullptr) error->clear();
  return true;
}

bool read_diskann_pq_bridge_manifest(const diskann_pq_artifact_paths &paths,
                                     diskann_pq_bridge_manifest *manifest,
                                     std::string *error) {
  if (manifest == nullptr) {
    set_error(error, "manifest is null");
    return false;
  }
  *manifest = {};
  std::ifstream file(paths.artifact_manifest_path, std::ios::in);
  if (!file.is_open()) {
    set_error(error, "artifact manifest is missing");
    return false;
  }

  std::string header;
  std::getline(file, header);
  if (header != "mysql-vector-diskann-pq-bridge-manifest-v1") {
    set_error(error, "artifact manifest header mismatch");
    return false;
  }

  constexpr uint32_t field_count =
      static_cast<uint32_t>(bridge_manifest_field::kCount);
  static_assert(field_count < std::numeric_limits<uint32_t>::digits);
  constexpr uint32_t all_fields = (uint32_t{1} << field_count) - 1;
  uint32_t seen_fields = 0;
  std::string line;
  while (std::getline(file, line)) {
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) {
      set_error(error, "artifact manifest line is malformed");
      return false;
    }
    const std::string key = line.substr(0, tab);
    const std::string value = line.substr(tab + 1);
    const bridge_manifest_field field = parse_bridge_manifest_field(key);
    if (field == bridge_manifest_field::kCount) {
      set_error(error, "artifact manifest field is unknown: " + key);
      return false;
    }
    const uint32_t field_bit = uint32_t{1} << static_cast<uint32_t>(field);
    if ((seen_fields & field_bit) != 0) {
      set_error(error, "artifact manifest field is duplicated: " + key);
      return false;
    }
    seen_fields |= field_bit;

    switch (field) {
      case bridge_manifest_field::kRowCount:
        if (!parse_manifest_uint(value, &manifest->row_count)) {
          set_error(error, "artifact manifest value is malformed");
          return false;
        }
        break;
      case bridge_manifest_field::kDimension:
        if (!parse_manifest_u32(value, &manifest->dimension, error))
          return false;
        break;
      case bridge_manifest_field::kPqChunks:
        if (!parse_manifest_u32(value, &manifest->pq_chunks, error))
          return false;
        break;
      case bridge_manifest_field::kCentroidCount:
        if (!parse_manifest_u32(value, &manifest->centroid_count, error))
          return false;
        break;
      case bridge_manifest_field::kMaxDegree:
        if (!parse_manifest_u32(value, &manifest->max_degree, error))
          return false;
        break;
      case bridge_manifest_field::kBuildComplexity:
        if (!parse_manifest_u32(value, &manifest->build_complexity, error))
          return false;
        break;
      case bridge_manifest_field::kCacheNodes:
        if (!parse_manifest_u32(value, &manifest->cache_nodes, error))
          return false;
        break;
      case bridge_manifest_field::kPivotsChecksum:
        if (!parse_manifest_uint(value, &manifest->pivots_checksum)) {
          set_error(error, "artifact manifest value is malformed");
          return false;
        }
        break;
      case bridge_manifest_field::kCompressedChecksum:
        if (!parse_manifest_uint(value, &manifest->compressed_checksum)) {
          set_error(error, "artifact manifest value is malformed");
          return false;
        }
        break;
      case bridge_manifest_field::kMedoidsChecksum:
        if (!parse_manifest_uint(value, &manifest->medoids_checksum)) {
          set_error(error, "artifact manifest value is malformed");
          return false;
        }
        break;
      case bridge_manifest_field::kZeroMean:
        if (!parse_manifest_bool(value, &manifest->zero_mean)) {
          set_error(error, "artifact manifest value is malformed");
          return false;
        }
        break;
      case bridge_manifest_field::kArtifactsConsumed:
        if (!parse_manifest_bool(value, &manifest->artifacts_consumed)) {
          set_error(error, "artifact manifest value is malformed");
          return false;
        }
        break;
      case bridge_manifest_field::kOfficialPqUsed:
        if (!parse_manifest_bool(value, &manifest->official_pq_used)) {
          set_error(error, "artifact manifest value is malformed");
          return false;
        }
        break;
      case bridge_manifest_field::kCount:
        return false;
    }
  }
  if (file.bad()) {
    set_error(error, "artifact manifest read failed");
    return false;
  }
  if (seen_fields != all_fields) {
    set_error(error, "artifact manifest fields are incomplete");
    return false;
  }
  if (error != nullptr) error->clear();
  return true;
}

bool validate_diskann_pq_bridge_manifest(
    const diskann_pq_artifact_paths &paths,
    const diskann_pq_artifact_metadata &metadata,
    const diskann_pq_bridge_manifest &manifest, std::string *error) {
  if (manifest.row_count != metadata.row_count ||
      manifest.dimension != metadata.dimension ||
      manifest.pq_chunks != metadata.pq_chunks ||
      manifest.centroid_count != metadata.centroid_count ||
      manifest.zero_mean != metadata.zero_mean) {
    set_error(error, "artifact manifest metadata mismatch");
    return false;
  }

  diskann_pq_bridge_manifest current;
  if (!make_diskann_pq_bridge_manifest(
          paths, metadata, manifest.max_degree, manifest.build_complexity,
          manifest.cache_nodes, manifest.artifacts_consumed,
          manifest.official_pq_used, &current, error)) {
    return false;
  }
  if (manifest.pivots_checksum != current.pivots_checksum) {
    set_error(error, "pq_pivots checksum mismatch");
    return false;
  }
  if (manifest.compressed_checksum != current.compressed_checksum) {
    set_error(error, "pq_compressed checksum mismatch");
    return false;
  }
  if (manifest.medoids_checksum != current.medoids_checksum) {
    set_error(error, "medoids checksum mismatch");
    return false;
  }
  if (error != nullptr) error->clear();
  return true;
}

}  // namespace vector_index
