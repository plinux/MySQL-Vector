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

#include "sql/vector/vector_diskann_pq_runtime.h"

#include "my_dbug.h"
#include "sql/vector/vector_elkan_kmeans.h"
#include "sql/vector/vector_index_runtime_config.h"
#include "sql/vector/vector_index_runtime_thread_pool.h"
#include "sql/vector/vector_pq_centroid_scan.h"
#include "sql/vector/vector_simd_distance.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <new>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace vector_index {
namespace {

using pq_runtime_clock = std::chrono::steady_clock;

constexpr const char *kDiskAnnRawManifestHeader =
    "mysql-vector-diskann-raw-manifest-v1";
constexpr uint32_t kDiskAnnPqCentroidCount = 256;
constexpr uint64_t kDiskAnnMaxPqTrainingRows = 256000;
constexpr uint64_t kDefaultDiskAnnPqEncodeBlockRows = 1000000;
constexpr uint64_t kMinDiskAnnPqEncodeBlockRows = 4096;
constexpr const char *kNativePqMemoryBudgetExceeded =
    "native_pq_memory_budget_exceeded";

struct raw_manifest_segment {
  std::string vector_path;
  uint64_t row_count{0};
};

struct pq_chunk_runtime_result {
  uint64_t distance_calls{0};
  uint64_t skipped_distance_calls{0};
  bool ok{false};
  std::string error;
};

struct pq_chunk_model {
  uint32_t begin{0};
  uint32_t end{0};
  pq_centroid_model centroids;
};

uint64_t elapsed_ms(pq_runtime_clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          pq_runtime_clock::now() - start)
          .count());
}

bool cpu_supports_avx512() {
#if defined(MYSQL_VECTOR_ENABLE_NATIVE_SIMD) &&   \
    (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
  return __builtin_cpu_supports("avx512f");
#else
  return false;
#endif
}

uint32_t effective_thread_count(uint32_t requested_threads) {
  if (requested_threads != 0) return requested_threads;
  const unsigned hardware_threads = std::thread::hardware_concurrency();
  return std::max<uint32_t>(1, static_cast<uint32_t>(hardware_threads));
}

template <typename T>
bool read_binary_value(std::ifstream &file, T *value) {
  if (value == nullptr) return false;
  file.read(reinterpret_cast<char *>(value), sizeof(*value));
  return file.good();
}

void set_error(std::string *error, const char *message) {
  if (error != nullptr) *error = message == nullptr ? "" : message;
}

bool multiply_uint64(uint64_t lhs, uint64_t rhs, uint64_t *out) {
  if (out == nullptr) return false;
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    return false;
  }
  *out = lhs * rhs;
  return true;
}

void normalize_cosine_rows(float *vectors, uint64_t rows, uint32_t dimension) {
  for (uint64_t row = 0; row < rows; ++row) {
    float *vector = vectors + row * dimension;
    float norm = std::numeric_limits<float>::epsilon();
    for (uint32_t dim = 0; dim < dimension; ++dim) {
      norm += vector[dim] * vector[dim];
    }
    norm = std::sqrt(norm);
    for (uint32_t dim = 0; dim < dimension; ++dim) vector[dim] /= norm;
  }
}

std::vector<std::string> split_tab_line(const std::string &line) {
  std::vector<std::string> fields;
  size_t begin = 0;
  while (begin <= line.size()) {
    const size_t end = line.find('\t', begin);
    if (end == std::string::npos) {
      fields.push_back(line.substr(begin));
      break;
    }
    fields.push_back(line.substr(begin, end - begin));
    begin = end + 1;
  }
  return fields;
}

bool parse_uint64_field(const std::string &text, uint64_t *value) {
  if (value == nullptr || text.empty()) return false;
  std::istringstream stream(text);
  uint64_t parsed = 0;
  stream >> parsed;
  if (!stream || !stream.eof()) return false;
  *value = parsed;
  return true;
}

bool sample_position(uint64_t sample_index, uint64_t total_count,
                     uint64_t sample_count, uint64_t *position) {
  if (position == nullptr ||
      sample_index > std::numeric_limits<size_t>::max() ||
      total_count > std::numeric_limits<size_t>::max() ||
      sample_count > std::numeric_limits<size_t>::max()) {
    return false;
  }
  size_t sampled_position = 0;
  if (!runtime_uniform_sample_position(
          static_cast<size_t>(sample_index), static_cast<size_t>(total_count),
          static_cast<size_t>(sample_count), &sampled_position)) {
    return false;
  }
  *position = sampled_position;
  return true;
}

std::string resolve_manifest_path(const std::filesystem::path &manifest_path,
                                  const std::string &path) {
  std::filesystem::path candidate(path);
  if (candidate.is_relative())
    candidate = manifest_path.parent_path() / candidate;
  return candidate.string();
}

bool read_fbin_header(std::ifstream &file, uint32_t expected_dimension,
                      uint64_t *row_count, std::string *error) {
  if (row_count == nullptr) return false;
  *row_count = 0;
  uint32_t rows = 0;
  uint32_t dimension = 0;
  if (!read_binary_value(file, &rows) || !read_binary_value(file, &dimension)) {
    set_error(error, "truncated fbin header");
    return false;
  }
  if (dimension != expected_dimension) {
    set_error(error, "fbin dimension mismatch");
    return false;
  }
  if (rows == 0) {
    set_error(error, "fbin row_count is zero");
    return false;
  }
  *row_count = rows;
  return true;
}

bool read_fbin_sample_rows(const std::string &input_file,
                           uint32_t expected_dimension,
                           uint64_t global_row_offset,
                           uint64_t expected_row_count,
                           const std::vector<uint64_t> &sample_positions,
                           size_t *next_sample_index,
                           std::vector<float> *sample, std::string *error) {
  if (next_sample_index == nullptr || sample == nullptr) return false;

  std::ifstream file(input_file, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    set_error(error, "input_file is not readable");
    return false;
  }

  uint64_t rows = 0;
  if (!read_fbin_header(file, expected_dimension, &rows, error)) return false;
  if (expected_row_count != 0 && rows != expected_row_count) {
    set_error(error, "raw manifest segment row count mismatch");
    return false;
  }
  const uint64_t segment_end = global_row_offset + rows;
  if (segment_end < global_row_offset) {
    set_error(error, "raw manifest row count overflow");
    return false;
  }

  const uint64_t row_bytes =
      static_cast<uint64_t>(expected_dimension) * sizeof(float);
  const uint64_t payload_offset = 2ULL * sizeof(uint32_t);
  while (*next_sample_index < sample_positions.size()) {
    const uint64_t position = sample_positions[*next_sample_index];
    if (position < global_row_offset) {
      set_error(error, "training sample position order is invalid");
      return false;
    }
    if (position >= segment_end) break;

    const uint64_t local_row = position - global_row_offset;
    uint64_t byte_offset = 0;
    if (!multiply_uint64(local_row, row_bytes, &byte_offset) ||
        byte_offset > std::numeric_limits<uint64_t>::max() - payload_offset ||
        byte_offset + payload_offset >
            static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
      set_error(error, "training sample offset overflow");
      return false;
    }
    file.seekg(static_cast<std::streamoff>(payload_offset + byte_offset),
               std::ios::beg);
    if (!file.good()) {
      set_error(error, "training sample seek failed");
      return false;
    }
    const size_t begin = sample->size();
    sample->resize(begin + expected_dimension);
    file.read(reinterpret_cast<char *>(sample->data() + begin),
              static_cast<std::streamsize>(row_bytes));
    if (!file.good()) {
      set_error(error, "truncated fbin payload");
      return false;
    }
    ++(*next_sample_index);
  }
  return true;
}

bool for_each_fbin_block(
    const std::string &input_file, uint32_t expected_dimension,
    uint64_t expected_row_count, uint64_t block_rows,
    const char *row_count_mismatch_error,
    const std::function<bool(float *, uint64_t)> &visitor,
    std::string *error) {
  if (block_rows == 0) {
    set_error(error, "fbin block row count is zero");
    return false;
  }

  std::ifstream file(input_file, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    set_error(error, "input_file is not readable");
    return false;
  }

  uint64_t rows = 0;
  if (!read_fbin_header(file, expected_dimension, &rows, error)) return false;
  if (expected_row_count != 0 && rows != expected_row_count) {
    set_error(error, row_count_mismatch_error);
    return false;
  }

  uint64_t remaining_rows = rows;
  std::vector<float> block;
  while (remaining_rows > 0) {
    const uint64_t current_rows = std::min(remaining_rows, block_rows);
    uint64_t value_count = 0;
    if (!multiply_uint64(current_rows, expected_dimension, &value_count) ||
        value_count > std::numeric_limits<size_t>::max()) {
      set_error(error, "fbin payload size overflow");
      return false;
    }
    if (value_count >
        static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) /
            sizeof(float)) {
      set_error(error, "fbin payload size overflow");
      return false;
    }
    block.resize(static_cast<size_t>(value_count));
    file.read(reinterpret_cast<char *>(block.data()),
              static_cast<std::streamsize>(value_count * sizeof(float)));
    if (!file.good()) {
      set_error(error, "truncated fbin payload");
      return false;
    }
    if (std::any_of(block.begin(), block.end(),
                    [](float value) { return !std::isfinite(value); })) {
      set_error(error, "fbin payload contains non-finite value");
      return false;
    }
    if (!visitor(block.data(), current_rows)) return false;
    remaining_rows -= current_rows;
  }

  char extra = 0;
  file.read(&extra, 1);
  if (!file.eof()) {
    set_error(error, "fbin file has extra payload");
    return false;
  }
  return true;
}

bool input_is_raw_manifest(const char *input_file) {
  std::ifstream file(input_file, std::ios::in | std::ios::binary);
  if (!file.is_open()) return false;
  const size_t header_size = std::strlen(kDiskAnnRawManifestHeader);
  std::string prefix(header_size, '\0');
  file.read(prefix.data(), static_cast<std::streamsize>(prefix.size()));
  return file.gcount() == static_cast<std::streamsize>(prefix.size()) &&
         std::memcmp(prefix.data(), kDiskAnnRawManifestHeader, prefix.size()) ==
             0;
}

bool read_raw_manifest(const char *input_file, uint32_t expected_dimension,
                       std::vector<raw_manifest_segment> *segments,
                       uint64_t *total_rows, std::string *error) {
  if (segments == nullptr || total_rows == nullptr) return false;
  segments->clear();
  *total_rows = 0;

  std::ifstream file(input_file, std::ios::in);
  if (!file.is_open()) {
    set_error(error, "raw manifest is not readable");
    return false;
  }

  std::string line;
  if (!std::getline(file, line) || line != kDiskAnnRawManifestHeader) {
    set_error(error, "raw manifest header mismatch");
    return false;
  }

  bool saw_dimension = false;
  bool saw_count = false;
  uint64_t manifest_count = 0;
  const std::filesystem::path manifest_path(input_file);
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> fields = split_tab_line(line);
    if (fields.size() == 2 && fields[0] == "dimension") {
      uint64_t dimension = 0;
      if (!parse_uint64_field(fields[1], &dimension) ||
          dimension != expected_dimension) {
        set_error(error, "raw manifest dimension mismatch");
        return false;
      }
      saw_dimension = true;
      continue;
    }
    if (fields.size() == 2 && fields[0] == "count") {
      if (!parse_uint64_field(fields[1], &manifest_count)) {
        set_error(error, "raw manifest count is invalid");
        return false;
      }
      saw_count = true;
      continue;
    }
    if (fields.size() == 4 && fields[0] == "segment") {
      raw_manifest_segment segment;
      segment.vector_path = resolve_manifest_path(manifest_path, fields[2]);
      if (!parse_uint64_field(fields[3], &segment.row_count) ||
          segment.row_count == 0) {
        set_error(error, "raw manifest segment row count is invalid");
        return false;
      }
      if (*total_rows >
          std::numeric_limits<uint64_t>::max() - segment.row_count) {
        set_error(error, "raw manifest row count overflow");
        return false;
      }
      *total_rows += segment.row_count;
      segments->push_back(std::move(segment));
      continue;
    }
    set_error(error, "raw manifest line is invalid");
    return false;
  }

  if (!file.eof()) {
    set_error(error, "raw manifest read failed");
    return false;
  }
  if (!saw_dimension || !saw_count) {
    set_error(error, "raw manifest is missing required metadata");
    return false;
  }
  if (manifest_count != *total_rows) {
    set_error(error, "raw manifest count mismatch");
    return false;
  }
  if (segments->empty()) {
    set_error(error, "raw manifest has no segment");
    return false;
  }
  return true;
}

bool for_each_raw_input_block(
    const char *input_file, uint32_t expected_dimension,
    uint64_t expected_row_count, uint64_t block_rows,
    const std::function<bool(float *, uint64_t)> &visitor, std::string *error) {
  if (!input_is_raw_manifest(input_file)) {
    return for_each_fbin_block(
        input_file, expected_dimension, expected_row_count, block_rows,
        "raw input row count changed during training", visitor, error);
  }

  std::vector<raw_manifest_segment> segments;
  uint64_t manifest_rows = 0;
  if (!read_raw_manifest(input_file, expected_dimension, &segments,
                         &manifest_rows, error)) {
    return false;
  }
  if (manifest_rows != expected_row_count) {
    set_error(error, "raw input row count changed during training");
    return false;
  }

  for (const raw_manifest_segment &segment : segments) {
    if (!for_each_fbin_block(segment.vector_path, expected_dimension,
                             segment.row_count, block_rows,
                             "raw manifest segment row count mismatch", visitor,
                             error)) {
      return false;
    }
  }
  return true;
}

bool read_raw_training_sample(const char *input_file,
                              uint32_t expected_dimension, uint64_t row_count,
                              uint64_t sample_rows, std::vector<float> *sample,
                              std::string *error) {
  if (sample == nullptr || row_count == 0 || sample_rows == 0 ||
      sample_rows > row_count) {
    return false;
  }
  sample->clear();

  uint64_t value_count = 0;
  if (!multiply_uint64(sample_rows, expected_dimension, &value_count) ||
      value_count > std::numeric_limits<size_t>::max()) {
    set_error(error, "training sample size overflow");
    return false;
  }

  std::vector<uint64_t> sample_positions;
  sample_positions.reserve(static_cast<size_t>(sample_rows));
  for (uint64_t sample_index = 0; sample_index < sample_rows; ++sample_index) {
    uint64_t position = 0;
    if (!sample_position(sample_index, row_count, sample_rows, &position)) {
      set_error(error, "training sample position overflow");
      return false;
    }
    sample_positions.push_back(position);
  }
  sample->reserve(static_cast<size_t>(value_count));

  size_t next_sample_index = 0;
  if (!input_is_raw_manifest(input_file)) {
    if (!read_fbin_sample_rows(input_file, expected_dimension, 0, 0,
                               sample_positions, &next_sample_index, sample,
                               error)) {
      return false;
    }
  } else {
    std::vector<raw_manifest_segment> segments;
    uint64_t manifest_rows = 0;
    if (!read_raw_manifest(input_file, expected_dimension, &segments,
                           &manifest_rows, error)) {
      return false;
    }
    if (manifest_rows != row_count) {
      set_error(error, "raw manifest count mismatch");
      return false;
    }
    uint64_t global_row_offset = 0;
    for (const raw_manifest_segment &segment : segments) {
      if (!read_fbin_sample_rows(segment.vector_path, expected_dimension,
                                 global_row_offset, segment.row_count,
                                 sample_positions, &next_sample_index, sample,
                                 error)) {
        return false;
      }
      global_row_offset += segment.row_count;
    }
  }

  if (next_sample_index != sample_positions.size() ||
      sample->size() != static_cast<size_t>(value_count)) {
    set_error(error, "training sample row count mismatch");
    return false;
  }
  if (std::any_of(sample->begin(), sample->end(),
                  [](float value) { return !std::isfinite(value); })) {
    set_error(error, "fbin payload contains non-finite value");
    return false;
  }
  return true;
}

bool inspect_raw_input_rows(const char *input_file, uint32_t expected_dimension,
                            uint64_t *row_count, std::string *error) {
  if (row_count == nullptr) return false;
  *row_count = 0;
  if (!input_is_raw_manifest(input_file)) {
    std::ifstream file(input_file, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      set_error(error, "input_file is not readable");
      return false;
    }
    uint32_t rows = 0;
    uint32_t dimension = 0;
    if (!read_binary_value(file, &rows) ||
        !read_binary_value(file, &dimension)) {
      set_error(error, "truncated fbin header");
      return false;
    }
    if (dimension != expected_dimension) {
      set_error(error, "fbin dimension mismatch");
      return false;
    }
    if (rows == 0) {
      set_error(error, "fbin row_count is zero");
      return false;
    }
    *row_count = rows;
    return true;
  }

  std::vector<raw_manifest_segment> segments;
  return read_raw_manifest(input_file, expected_dimension, &segments, row_count,
                           error);
}

bool checked_add(uint64_t value, uint64_t *total) {
  if (total == nullptr ||
      *total > std::numeric_limits<uint64_t>::max() - value) {
    return false;
  }
  *total += value;
  return true;
}

bool add_product(uint64_t lhs, uint64_t rhs, uint64_t *total) {
  uint64_t product = 0;
  return multiply_uint64(lhs, rhs, &product) && checked_add(product, total);
}

bool estimate_native_pq_memory(uint64_t rows, uint64_t training_rows,
                               uint64_t encode_block_rows, uint32_t dimension,
                               uint32_t pq_chunks, uint32_t worker_count,
                               uint64_t *estimated_size) {
  const uint64_t active_block_rows = std::min(rows, encode_block_rows);
  const uint32_t largest_chunk_dimension =
      (dimension + pq_chunks - 1) / pq_chunks;
  const uint32_t center_count = static_cast<uint32_t>(
      std::min<uint64_t>(kDiskAnnPqCentroidCount, training_rows));
  const uint32_t packed_center_count = ((center_count + 15U) / 16U) * 16U;

  uint64_t total = 0;
  if (!add_product(training_rows,
                   static_cast<uint64_t>(dimension) * sizeof(float), &total) ||
      !add_product(kDiskAnnPqCentroidCount,
                   static_cast<uint64_t>(dimension) * sizeof(float), &total) ||
      !add_product(packed_center_count,
                   static_cast<uint64_t>(dimension) * sizeof(float), &total) ||
      !add_product(dimension, sizeof(float), &total) ||
      !add_product(static_cast<uint64_t>(pq_chunks) + 1, sizeof(uint32_t),
                   &total) ||
      !add_product(rows, sizeof(uint64_t), &total) ||
      !add_product(active_block_rows,
                   static_cast<uint64_t>(dimension) * sizeof(float), &total) ||
      !add_product(active_block_rows, pq_chunks, &total)) {
    return false;
  }

  uint64_t per_worker = 0;
  if (!add_product(
          training_rows,
          static_cast<uint64_t>(largest_chunk_dimension) * sizeof(float),
          &per_worker) ||
      !add_product(
          training_rows,
          static_cast<uint64_t>(largest_chunk_dimension) * sizeof(float),
          &per_worker) ||
      !add_product(training_rows, sizeof(uint32_t), &per_worker) ||
      !add_product(training_rows, sizeof(float), &per_worker) ||
      !add_product(training_rows,
                   static_cast<uint64_t>(center_count) * sizeof(float),
                   &per_worker) ||
      !add_product(
          center_count,
          static_cast<uint64_t>(largest_chunk_dimension) * sizeof(float),
          &per_worker) ||
      !add_product(center_count,
                   static_cast<uint64_t>(center_count) * sizeof(float),
                   &per_worker)) {
    return false;
  }

  if (!add_product(worker_count, per_worker, &total) ||
      !checked_add(256ULL * 1024ULL * 1024ULL, &total)) {
    return false;
  }

  *estimated_size = total;
  return true;
}

std::vector<uint32_t> make_chunk_offsets(uint32_t dimension,
                                         uint32_t pq_chunks) {
  std::vector<uint32_t> offsets(pq_chunks + 1, 0);
  const uint32_t base_width = dimension / pq_chunks;
  const uint32_t remainder = dimension % pq_chunks;
  for (uint32_t chunk = 0; chunk < pq_chunks; ++chunk) {
    const uint32_t width = base_width + (chunk < remainder ? 1U : 0U);
    offsets[chunk + 1] = offsets[chunk] + width;
  }
  return offsets;
}

std::vector<float> extract_chunk_data(const std::vector<float> &vectors,
                                      uint64_t row_count, uint32_t dimension,
                                      uint32_t begin, uint32_t end) {
  const uint32_t chunk_dimension = end - begin;
  std::vector<float> chunk_data(static_cast<size_t>(row_count) *
                                chunk_dimension);
  for (uint64_t row = 0; row < row_count; ++row) {
    const float *source =
        vectors.data() + static_cast<size_t>(row) * dimension + begin;
    float *target =
        chunk_data.data() + static_cast<size_t>(row) * chunk_dimension;
    std::copy(source, source + chunk_dimension, target);
  }
  return chunk_data;
}

const char *runtime_selected_path_name(l2_distance_kernel kernel) {
  if (kernel == l2_distance_kernel::kScalar) return "scalar_fallback";
  return l2_distance_kernel_name(kernel);
}

void copy_chunk_centroid(const elkan_kmeans_result &kmeans_result,
                         uint32_t begin, uint32_t end,
                         std::vector<float> *centroid) {
  for (uint32_t dim = begin; dim < end; ++dim) {
    (*centroid)[dim] = kmeans_result.centroid[dim - begin];
  }
}

void copy_chunk_centers(const elkan_kmeans_result &kmeans_result,
                        uint32_t dimension, uint32_t begin, uint32_t end,
                        std::vector<float> *pivots) {
  const uint32_t chunk_dimension = end - begin;
  const uint32_t center_count =
      static_cast<uint32_t>(kmeans_result.centers.size() / chunk_dimension);
  for (uint32_t center = 0; center < center_count; ++center) {
    float *target =
        pivots->data() + static_cast<size_t>(center) * dimension + begin;
    const float *source = kmeans_result.centers.data() +
                          static_cast<size_t>(center) * chunk_dimension;
    std::copy(source, source + chunk_dimension, target);
  }
}

void pad_chunk_centers(uint32_t trained_center_count,
                       uint32_t output_center_count, uint32_t dimension,
                       uint32_t begin, uint32_t end,
                       std::vector<float> *pivots) {
  if (pivots == nullptr || trained_center_count == 0 ||
      trained_center_count >= output_center_count) {
    return;
  }

  const uint32_t chunk_dimension = end - begin;
  const float *source =
      pivots->data() +
      static_cast<size_t>(trained_center_count - 1) * dimension + begin;
  for (uint32_t center = trained_center_count; center < output_center_count;
       ++center) {
    float *target =
        pivots->data() + static_cast<size_t>(center) * dimension + begin;
    std::copy(source, source + chunk_dimension, target);
  }
}

void merge_pq_centroid_scan_stats(const pq_centroid_scan_stats &source,
                                  pq_centroid_scan_stats *target) {
  if (target == nullptr || source.distance_evaluations == 0) return;
  target->distance_evaluations += source.distance_evaluations;
  target->kernel = source.kernel;
}

bool train_pq_chunk(const std::vector<float> &training_vectors,
                    uint64_t training_rows, uint32_t dimension, uint32_t chunk,
                    uint32_t trained_center_count, uint32_t output_center_count,
                    const std::vector<uint32_t> &offsets, bool zero_mean,
                    diskann_pq_artifact_payload *payload, pq_chunk_model *model,
                    pq_chunk_runtime_result *chunk_result) {
  const uint32_t begin = offsets[chunk];
  const uint32_t end = offsets[chunk + 1];

  std::vector<float> chunk_data = extract_chunk_data(
      training_vectors, training_rows, dimension, begin, end);
  elkan_kmeans_config kmeans_config;
  kmeans_config.dimension = end - begin;
  kmeans_config.center_count = trained_center_count;
  kmeans_config.max_iterations = 20;
  // PQ chunks own the worker budget; Elkan must not create a nested pool.
  kmeans_config.threads = 1;
  kmeans_config.seed = static_cast<uint64_t>(chunk);
  kmeans_config.zero_mean = zero_mean;

  elkan_kmeans_result kmeans_result;
  if (!run_elkan_kmeans(chunk_data.data(), training_rows, kmeans_config,
                        &kmeans_result, &chunk_result->error)) {
    return false;
  }
  chunk_result->distance_calls += kmeans_result.distance_calls;
  chunk_result->skipped_distance_calls += kmeans_result.skipped_distance_calls;
  copy_chunk_centroid(kmeans_result, begin, end, &payload->centroid);
  copy_chunk_centers(kmeans_result, dimension, begin, end, &payload->pivots);
  pad_chunk_centers(trained_center_count, output_center_count, dimension, begin,
                    end, &payload->pivots);

  model->begin = begin;
  model->end = end;
  if (!make_pq_centroid_model(kmeans_result.centers.data(),
                              trained_center_count, end - begin,
                              pq_centroid_scan_kernel::kAuto, &model->centroids,
                              &chunk_result->error)) {
    return false;
  }
  chunk_result->ok = true;
  return true;
}

bool write_uncentered_disk_pq_artifacts(
    const char *output_prefix,
    const diskann_pq_artifact_metadata &memory_metadata,
    const diskann_pq_artifact_payload &memory_payload,
    const diskann_pq_artifact_paths &memory_artifacts,
    diskann_pq_artifact_paths *disk_artifacts, std::string *error) {
  if (output_prefix == nullptr || output_prefix[0] == '\0' ||
      disk_artifacts == nullptr) {
    set_error(error, "disk PQ artifact output is invalid");
    return false;
  }
  if (!make_diskann_pq_artifact_paths(output_prefix, disk_artifacts)) {
    set_error(error, "disk PQ artifact output prefix is invalid");
    return false;
  }

  diskann_pq_artifact_payload disk_payload;
  disk_payload.pivots = memory_payload.pivots;
  disk_payload.centroid = memory_payload.centroid;
  disk_payload.chunk_offsets = memory_payload.chunk_offsets;
  disk_payload.docid_ordinals = memory_payload.docid_ordinals;
  for (uint32_t center = 0; center < memory_metadata.centroid_count; ++center) {
    float *pivot = disk_payload.pivots.data() +
                   static_cast<size_t>(center) * memory_metadata.dimension;
    for (uint32_t dim = 0; dim < memory_metadata.dimension; ++dim) {
      pivot[dim] += memory_payload.centroid[dim];
    }
  }
  std::fill(disk_payload.centroid.begin(), disk_payload.centroid.end(), 0.0F);

  diskann_pq_artifact_metadata disk_metadata = memory_metadata;
  disk_metadata.zero_mean = false;
  if (!write_diskann_pq_static_artifacts(*disk_artifacts, disk_metadata,
                                         disk_payload, error)) {
    return false;
  }

  std::error_code copy_error;
  std::filesystem::copy_file(
      memory_artifacts.compressed_path, disk_artifacts->compressed_path,
      std::filesystem::copy_options::overwrite_existing, copy_error);
  if (copy_error) {
    set_error(error, "disk PQ compressed artifact copy failed");
    return false;
  }
  return validate_diskann_pq_artifacts(*disk_artifacts, disk_metadata, error);
}

bool encode_block_with_models(const float *vectors, uint64_t rows,
                              uint32_t dimension,
                              const std::vector<pq_chunk_model> &models,
                              const std::vector<float> &centroid,
                              uint32_t pq_chunks, uint32_t worker_count,
                              std::vector<uint8_t> *block_codes,
                              pq_centroid_scan_stats *scan_stats,
                              std::string *error) {
  if (vectors == nullptr || block_codes == nullptr ||
      models.size() != pq_chunks || centroid.size() != dimension) {
    set_error(error, "pq runtime model state is invalid");
    return false;
  }
  if (rows > std::numeric_limits<size_t>::max() / pq_chunks) {
    set_error(error, "pq_compressed payload size overflow");
    return false;
  }
  block_codes->assign(static_cast<size_t>(rows) * pq_chunks, 0);

  uint32_t max_chunk_dimension = 0;
  for (const pq_chunk_model &model : models) {
    if (model.end <= model.begin) {
      set_error(error, "pq runtime model state is invalid");
      return false;
    }
    max_chunk_dimension =
        std::max(max_chunk_dimension, model.end - model.begin);
  }

  const auto encode_range = [&](uint64_t begin_row, uint64_t end_row,
                                pq_centroid_scan_stats *local_stats) -> bool {
    std::vector<float> centered_values(max_chunk_dimension);
    for (uint64_t row = begin_row; row < end_row; ++row) {
      for (uint32_t chunk = 0; chunk < pq_chunks; ++chunk) {
        const pq_chunk_model &model = models[chunk];
        const uint32_t chunk_dimension = model.end - model.begin;
        const float *values =
            vectors + static_cast<size_t>(row) * dimension + model.begin;
        float *centered = centered_values.data();
        for (uint32_t dim = 0; dim < chunk_dimension; ++dim) {
          centered[dim] =
              values[dim] - centroid[static_cast<size_t>(model.begin) + dim];
        }
        uint32_t nearest_center = 0;
        if (!scan_pq_centroids(model.centroids, centered, nullptr,
                               &nearest_center, nullptr, local_stats)) {
          return false;
        }
        (*block_codes)[static_cast<size_t>(row) * pq_chunks + chunk] =
            static_cast<uint8_t>(nearest_center);
      }
    }
    return true;
  };

  const uint32_t workers = std::max<uint32_t>(
      1, rows > std::numeric_limits<uint32_t>::max()
             ? worker_count
             : std::min<uint32_t>(worker_count, static_cast<uint32_t>(rows)));
  if (workers == 1) {
    if (!encode_range(0, rows, scan_stats)) {
      set_error(error, "pq centroid scan failed");
      return false;
    }
    return true;
  }

  std::vector<pq_centroid_scan_stats> worker_stats(workers);
  bool inject_worker_failure = false;
  DBUG_EXECUTE_IF("vector_pq_runtime_fail_encode_worker",
                  { inject_worker_failure = true; });
  const bool encode_ok = parallel_for_ranges_scoped(
      static_cast<size_t>(rows), workers,
      [&](size_t begin_row, size_t end_row, size_t worker) {
        if (inject_worker_failure && worker == 0) throw std::bad_alloc();
        return encode_range(begin_row, end_row, &worker_stats[worker]);
      });
  if (!encode_ok) {
    set_error(error, "pq centroid scan failed");
    return false;
  }
  for (uint32_t worker = 0; worker < workers; ++worker) {
    merge_pq_centroid_scan_stats(worker_stats[worker], scan_stats);
  }
  return true;
}

}  // namespace

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool read_diskann_fbin_header_for_testing(const std::string &input_file,
                                          uint32_t expected_dimension,
                                          uint64_t *row_count,
                                          std::string *error) {
  std::ifstream file(input_file, std::ios::in | std::ios::binary);
  return read_fbin_header(file, expected_dimension, row_count, error);
}

bool for_each_diskann_fbin_block_for_testing(
    const std::string &input_file, uint32_t expected_dimension,
    uint64_t expected_row_count, uint64_t block_rows, bool visitor_result,
    uint64_t *visited_rows, std::string *error) {
  return for_each_fbin_block(
      input_file, expected_dimension, expected_row_count, block_rows,
      "test row count mismatch",
      [visitor_result, visited_rows](float *, uint64_t rows) {
        if (visited_rows != nullptr) *visited_rows += rows;
        return visitor_result;
      },
      error);
}

bool read_diskann_raw_training_sample_for_testing(const std::string &input_file,
                                                  uint32_t expected_dimension,
                                                  uint64_t row_count,
                                                  uint64_t sample_rows,
                                                  std::vector<float> *sample,
                                                  std::string *error) {
  return read_raw_training_sample(input_file.c_str(), expected_dimension,
                                  row_count, sample_rows, sample, error);
}

bool inspect_diskann_raw_input_rows_for_testing(const std::string &input_file,
                                                uint32_t expected_dimension,
                                                uint64_t *row_count,
                                                std::string *error) {
  return inspect_raw_input_rows(input_file.c_str(), expected_dimension,
                                row_count, error);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

bool make_diskann_pq_memory_plan(const diskann_pq_memory_plan_input &input,
                                 diskann_pq_memory_plan *plan,
                                 std::string *error) {
  if (plan == nullptr) {
    set_error(error, "native_pq_memory_plan is null");
    return false;
  }
  *plan = {};
  plan->encode_block_rows = kDefaultDiskAnnPqEncodeBlockRows;
  plan->adjustment = "none";
  if (input.rows == 0 || input.training_rows == 0 || input.dimension == 0 ||
      input.pq_chunks == 0 || input.pq_chunks > input.dimension) {
    set_error(error, "native_pq_memory_estimate_overflow");
    return false;
  }

  const uint32_t requested_workers = std::max<uint32_t>(
      1, std::min<uint32_t>(effective_thread_count(input.requested_threads),
                            input.pq_chunks));
  uint64_t current_estimate = 0;
  if (!estimate_native_pq_memory(input.rows, input.training_rows,
                                 plan->encode_block_rows, input.dimension,
                                 input.pq_chunks, requested_workers,
                                 &current_estimate)) {
    set_error(error, "native_pq_memory_estimate_overflow");
    return false;
  }
  if (input.memory_budget_size == 0 ||
      current_estimate <= input.memory_budget_size) {
    plan->worker_count = requested_workers;
    plan->estimated_size = current_estimate;
    if (input.memory_budget_size == 0) plan->adjustment = "unlimited";
    return true;
  }

  for (uint32_t workers = requested_workers; workers > 1;) {
    --workers;
    if (!estimate_native_pq_memory(
            input.rows, input.training_rows, plan->encode_block_rows,
            input.dimension, input.pq_chunks, workers, &current_estimate)) {
      set_error(error, "native_pq_memory_estimate_overflow");
      return false;
    }
    if (current_estimate <= input.memory_budget_size) {
      plan->worker_count = workers;
      plan->estimated_size = current_estimate;
      plan->adjustment = "reduced_threads";
      return true;
    }
  }

  while (plan->encode_block_rows > kMinDiskAnnPqEncodeBlockRows) {
    plan->encode_block_rows = std::max<uint64_t>(kMinDiskAnnPqEncodeBlockRows,
                                                 plan->encode_block_rows / 2);
    if (!estimate_native_pq_memory(input.rows, input.training_rows,
                                   plan->encode_block_rows, input.dimension,
                                   input.pq_chunks, 1, &current_estimate)) {
      set_error(error, "native_pq_memory_estimate_overflow");
      return false;
    }
    if (current_estimate <= input.memory_budget_size) {
      plan->worker_count = 1;
      plan->estimated_size = current_estimate;
      plan->adjustment = requested_workers > 1
                             ? "reduced_threads_and_block_rows"
                             : "reduced_block_rows";
      return true;
    }
  }

  plan->estimated_size = current_estimate;
  plan->adjustment = "budget_exceeded";
  set_error(error, kNativePqMemoryBudgetExceeded);
  return false;
}

bool build_diskann_pq_runtime(const diskann_pq_runtime_config &config,
                              const char *input_file, const char *output_prefix,
                              diskann_pq_runtime_result *result,
                              std::string *error) {
  if (result == nullptr) {
    set_error(error, "result is null");
    return false;
  }
  *result = {};
  const pq_runtime_clock::time_point start = pq_runtime_clock::now();

  if (config.dimension == 0) {
    set_error(error, "dimension is zero");
    return false;
  }
  if (config.pq_chunks == 0) {
    set_error(error, "pq_chunks is zero");
    return false;
  }
  if (config.pq_chunks > config.dimension) {
    set_error(error, "pq_chunks exceeds dimension");
    return false;
  }
  if (config.disk_pq_chunks > config.dimension) {
    set_error(error, "disk_pq_chunks exceeds dimension");
    return false;
  }
  if (!config.zero_mean && config.disk_pq_chunks != 0) {
    set_error(error, "disk_pq_chunks requires a zero-mean memory PQ build");
    return false;
  }
  if (input_file == nullptr || input_file[0] == '\0') {
    set_error(error, "input_file is empty");
    return false;
  }
  if (output_prefix == nullptr || output_prefix[0] == '\0') {
    set_error(error, "output_prefix is empty");
    return false;
  }

  const bool use_avx512 = config.prefer_avx512 && cpu_supports_avx512();
  result->selected_path = use_avx512 ? "avx512" : "scalar_fallback";
  result->pq_chunks = config.pq_chunks;
  result->memory_budget_size = config.memory_budget_size;
  result->encode_block_rows = kDefaultDiskAnnPqEncodeBlockRows;
  if (!make_diskann_pq_artifact_paths(output_prefix, &result->artifacts)) {
    set_error(error, "artifact output prefix is invalid");
    return false;
  }

  uint64_t inspected_rows = 0;
  if (!inspect_raw_input_rows(input_file, config.dimension, &inspected_rows,
                              error)) {
    return false;
  }
  result->row_count = inspected_rows;

  const uint64_t training_row_limit = config.training_row_limit == 0
                                          ? kDiskAnnMaxPqTrainingRows
                                          : config.training_row_limit;
  const uint64_t training_rows = std::min(inspected_rows, training_row_limit);
  const diskann_pq_memory_plan_input memory_plan_input{
      inspected_rows,   training_rows,    config.memory_budget_size,
      config.dimension, config.pq_chunks, config.threads};
  diskann_pq_memory_plan memory_plan;
  const bool memory_plan_ok =
      make_diskann_pq_memory_plan(memory_plan_input, &memory_plan, error);
  result->effective_threads = memory_plan.worker_count;
  result->encode_block_rows = memory_plan.encode_block_rows;
  result->memory_estimate_bytes = memory_plan.estimated_size;
  result->memory_budget_adjustment = memory_plan.adjustment;
  result->wait_cycles_hint =
      memory_plan.worker_count <= 1 ? 0 : memory_plan.worker_count - 1;
  if (!memory_plan_ok) return false;
  const uint32_t threads = memory_plan.worker_count;

  std::vector<float> training_vectors;
  const pq_runtime_clock::time_point raw_reader_start = pq_runtime_clock::now();
  if (!read_raw_training_sample(input_file, config.dimension, inspected_rows,
                                training_rows, &training_vectors, error)) {
    return false;
  }
  if (config.normalize_input) {
    normalize_cosine_rows(training_vectors.data(), training_rows,
                          config.dimension);
  }
  result->raw_reader_ms = elapsed_ms(raw_reader_start);

  l2_distance_stats path_stats;
  (void)l2_distance(training_vectors.data(), training_vectors.data(),
                    config.dimension, &path_stats);
  if (use_avx512 && path_stats.kernel == l2_distance_kernel::kAvx512) {
    result->selected_path = "avx512";
  } else {
    result->selected_path = runtime_selected_path_name(path_stats.kernel);
  }

  const std::vector<uint32_t> offsets =
      make_chunk_offsets(config.dimension, config.pq_chunks);
  const uint32_t trained_center_count = static_cast<uint32_t>(
      std::min<uint64_t>(kDiskAnnPqCentroidCount, training_rows));
  const uint32_t output_center_count = kDiskAnnPqCentroidCount;
  result->centroid_count = output_center_count;
  result->train_rows = training_rows;

  diskann_pq_artifact_payload payload;
  payload.pivots.assign(
      static_cast<size_t>(output_center_count) * config.dimension, 0.0F);
  payload.centroid.assign(config.dimension, 0.0F);
  payload.chunk_offsets = offsets;
  payload.docid_ordinals.assign(static_cast<size_t>(inspected_rows), 0);
  std::iota(payload.docid_ordinals.begin(), payload.docid_ordinals.end(), 0);

  std::vector<pq_chunk_runtime_result> chunk_results(config.pq_chunks);
  std::vector<pq_chunk_model> chunk_models(config.pq_chunks);
  const uint32_t worker_count = std::min(threads, config.pq_chunks);
  if (worker_count <= 1) {
    for (uint32_t chunk = 0; chunk < config.pq_chunks; ++chunk) {
      if (!train_pq_chunk(training_vectors, training_rows, config.dimension,
                          chunk, trained_center_count, output_center_count,
                          offsets, config.zero_mean, &payload,
                          &chunk_models[chunk], &chunk_results[chunk])) {
        set_error(error, chunk_results[chunk].error.c_str());
        return false;
      }
    }
  } else {
    std::atomic<uint32_t> next_chunk{0};
    std::atomic<bool> training_failed{false};
    bool inject_worker_failure = false;
    DBUG_EXECUTE_IF("vector_pq_runtime_fail_training_worker",
                    { inject_worker_failure = true; });
    const bool training_ok = parallel_for_ranges_scoped(
        worker_count, worker_count, [&](size_t, size_t, size_t worker) {
          if (inject_worker_failure && worker == 0) throw std::bad_alloc();
          while (!training_failed.load(std::memory_order_acquire)) {
            const uint32_t chunk =
                next_chunk.fetch_add(1, std::memory_order_relaxed);
            if (chunk >= config.pq_chunks) break;
            if (!train_pq_chunk(training_vectors, training_rows,
                                config.dimension, chunk, trained_center_count,
                                output_center_count, offsets, config.zero_mean,
                                &payload, &chunk_models[chunk],
                                &chunk_results[chunk])) {
              training_failed.store(true, std::memory_order_release);
              return false;
            }
          }
          return true;
        });
    if (!training_ok) {
      for (const pq_chunk_runtime_result &chunk_result : chunk_results) {
        if (!chunk_result.ok && !chunk_result.error.empty()) {
          set_error(error, chunk_result.error.c_str());
          return false;
        }
      }
      set_error(error, "pq chunk worker failed");
      return false;
    }
  }
  for (const pq_chunk_runtime_result &chunk_result : chunk_results) {
    result->distance_calls += chunk_result.distance_calls;
    result->skipped_distance_calls += chunk_result.skipped_distance_calls;
  }

  diskann_pq_artifact_metadata metadata;
  metadata.row_count = inspected_rows;
  metadata.dimension = config.dimension;
  metadata.pq_chunks = config.pq_chunks;
  metadata.centroid_count = output_center_count;
  metadata.zero_mean = config.zero_mean;
  if (!write_diskann_pq_static_artifacts(result->artifacts, metadata, payload,
                                         error)) {
    return false;
  }
  if (inspected_rows > std::numeric_limits<uint32_t>::max()) {
    set_error(error, "pq_compressed artifact row count exceeds limit");
    return false;
  }

  diskann_compressed_artifact_writer compressed_writer;
  if (!compressed_writer.open(result->artifacts.compressed_path,
                              static_cast<uint32_t>(inspected_rows),
                              config.pq_chunks, error)) {
    return false;
  }
  std::vector<uint8_t> block_codes;
  pq_centroid_scan_stats compression_stats;
  uint64_t compressed_rows = 0;
  const bool encoded = for_each_raw_input_block(
      input_file, config.dimension, inspected_rows, result->encode_block_rows,
      [&](float *block_vectors, uint64_t block_rows) -> bool {
        if (config.normalize_input) {
          normalize_cosine_rows(block_vectors, block_rows, config.dimension);
        }
        if (!encode_block_with_models(
                block_vectors, block_rows, config.dimension, chunk_models,
                payload.centroid, config.pq_chunks, result->effective_threads,
                &block_codes, &compression_stats, error)) {
          return false;
        }
        if (block_rows > std::numeric_limits<uint32_t>::max()) {
          set_error(error, "pq_compressed artifact row count exceeds limit");
          return false;
        }
        if (!compressed_writer.write_block(
                block_codes.data(), static_cast<uint32_t>(block_rows), error)) {
          return false;
        }
        compressed_rows += block_rows;
        return true;
      },
      error);
  if (!encoded) return false;
  if (!compressed_writer.close(error)) return false;
  result->distance_calls += compression_stats.distance_evaluations;
  result->compressed_rows = compressed_rows;
  if (compressed_rows != inspected_rows) {
    set_error(error, "raw input row count changed during training");
    return false;
  }
  if (!validate_diskann_pq_artifacts(result->artifacts, metadata, error))
    return false;
  result->artifacts_written = true;

  if (config.disk_pq_chunks != 0) {
    const std::string disk_output_prefix =
        std::string(output_prefix) + "_disk_pq";
    if (config.disk_pq_chunks == config.pq_chunks) {
      if (!write_uncentered_disk_pq_artifacts(
              disk_output_prefix.c_str(), metadata, payload, result->artifacts,
              &result->disk_artifacts, error)) {
        return false;
      }
      result->disk_artifacts_written = true;
    } else {
      diskann_pq_runtime_config disk_config = config;
      disk_config.pq_chunks = config.disk_pq_chunks;
      disk_config.disk_pq_chunks = 0;
      disk_config.zero_mean = false;
      diskann_pq_runtime_result disk_result;
      if (!build_diskann_pq_runtime(disk_config, input_file,
                                    disk_output_prefix.c_str(), &disk_result,
                                    error)) {
        return false;
      }
      result->disk_artifacts = std::move(disk_result.artifacts);
      result->disk_artifacts_written = disk_result.artifacts_written;
      result->raw_reader_ms += disk_result.raw_reader_ms;
      result->distance_calls += disk_result.distance_calls;
      result->skipped_distance_calls += disk_result.skipped_distance_calls;
      result->wait_cycles_hint += disk_result.wait_cycles_hint;
      result->memory_estimate_bytes = std::max(
          result->memory_estimate_bytes, disk_result.memory_estimate_bytes);
    }
  }

  result->elapsed_ms = elapsed_ms(start);
  if (error != nullptr) error->clear();
  return true;
}

}  // namespace vector_index
