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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char *k_raw_manifest_header =
    "mysql-vector-diskann-raw-manifest-v1";

struct raw_segment {
  std::string docid_path;
  std::string vector_path;
  uint64_t row_count{0};
};

struct fake_index {
  size_t dimension{0};
  int32_t metric_type{2};
  std::vector<uint64_t> doc_ids;
  std::vector<float> vectors;
};

std::filesystem::path raw_data_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() + "_mysql_vector_raw.fbin");
}

std::filesystem::path doc_ids_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() + "_mysql_vector_docids.bin");
}

std::filesystem::path build_options_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() +
                               "_mysql_vector_build_options.txt");
}

std::filesystem::path medoids_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() + "_disk.index_medoids.bin");
}

template <typename T>
bool read_binary_value(std::istream *file, T *value) {
  if (file == nullptr || value == nullptr) return false;
  file->read(reinterpret_cast<char *>(value), sizeof(*value));
  return file->good();
}

template <typename T>
bool write_binary_value(std::ostream *file, T value) {
  if (file == nullptr) return false;
  file->write(reinterpret_cast<const char *>(&value), sizeof(value));
  return file->good();
}

bool write_index_files(const std::filesystem::path &prefix, size_t dimension,
                       const std::vector<uint64_t> &doc_ids,
                       const std::vector<float> &vectors) {
  if (dimension == 0 || doc_ids.size() * dimension != vectors.size()) {
    return false;
  }
  if (doc_ids.size() > std::numeric_limits<uint32_t>::max()) return false;

  std::error_code ec;
  std::filesystem::create_directories(prefix.parent_path(), ec);
  if (ec) return false;

  std::ofstream raw_file(raw_data_path(prefix),
                         std::ios::out | std::ios::binary | std::ios::trunc);
  std::ofstream docid_file(doc_ids_path(prefix),
                           std::ios::out | std::ios::binary | std::ios::trunc);
  if (!raw_file.is_open() || !docid_file.is_open()) return false;

  if (!write_binary_value(&raw_file, static_cast<uint32_t>(doc_ids.size())) ||
      !write_binary_value(&raw_file, static_cast<uint32_t>(dimension)) ||
      !write_binary_value(&docid_file, static_cast<uint64_t>(doc_ids.size()))) {
    return false;
  }

  if (!vectors.empty()) {
    raw_file.write(reinterpret_cast<const char *>(vectors.data()),
                   static_cast<std::streamsize>(vectors.size() *
                                                sizeof(float)));
  }
  if (!doc_ids.empty()) {
    docid_file.write(reinterpret_cast<const char *>(doc_ids.data()),
                     static_cast<std::streamsize>(doc_ids.size() *
                                                  sizeof(uint64_t)));
  }
  return raw_file.good() && docid_file.good();
}

bool write_build_options(const std::filesystem::path &prefix,
                         uint32_t build_threads,
                         uint32_t build_blas_threads,
                         const char *build_source) {
  std::ofstream file(build_options_path(prefix),
                     std::ios::out | std::ios::trunc);
  if (!file.is_open()) return false;
  file << "build_threads=" << build_threads << '\n';
  file << "build_blas_threads=" << build_blas_threads << '\n';
  file << "build_source=" << build_source << '\n';
  return file.good();
}

bool write_fake_medoids(const std::filesystem::path &prefix,
                        const std::vector<uint64_t> &doc_ids) {
  std::ofstream file(medoids_path(prefix),
                     std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.is_open()) return false;
  const uint32_t medoid_count = doc_ids.empty() ? 0 : 1;
  const uint32_t medoid_dim = 1;
  if (!write_binary_value(&file, medoid_count) ||
      !write_binary_value(&file, medoid_dim)) {
    return false;
  }
  if (medoid_count != 0) {
    if (!write_binary_value(&file, static_cast<uint32_t>(doc_ids.front()))) {
      return false;
    }
  }
  return file.good();
}

bool read_index_files(const std::filesystem::path &prefix, fake_index *index) {
  if (index == nullptr) return false;

  std::ifstream raw_file(raw_data_path(prefix), std::ios::in | std::ios::binary);
  std::ifstream docid_file(doc_ids_path(prefix),
                           std::ios::in | std::ios::binary);
  if (!raw_file.is_open() || !docid_file.is_open()) return false;

  uint32_t rows = 0;
  uint32_t dimension = 0;
  uint64_t docid_rows = 0;
  if (!read_binary_value(&raw_file, &rows) ||
      !read_binary_value(&raw_file, &dimension) ||
      !read_binary_value(&docid_file, &docid_rows) ||
      docid_rows != rows || dimension == 0) {
    return false;
  }

  std::vector<uint64_t> doc_ids(docid_rows);
  std::vector<float> vectors(static_cast<size_t>(rows) * dimension);
  if (!doc_ids.empty()) {
    docid_file.read(reinterpret_cast<char *>(doc_ids.data()),
                    static_cast<std::streamsize>(doc_ids.size() *
                                                 sizeof(uint64_t)));
  }
  if (!vectors.empty()) {
    raw_file.read(reinterpret_cast<char *>(vectors.data()),
                  static_cast<std::streamsize>(vectors.size() *
                                               sizeof(float)));
  }
  if (!raw_file.good() || !docid_file.good()) return false;

  index->dimension = dimension;
  index->doc_ids = std::move(doc_ids);
  index->vectors = std::move(vectors);
  return true;
}

bool append_segment(const raw_segment &segment, size_t expected_dimension,
                    std::vector<uint64_t> *doc_ids,
                    std::vector<float> *vectors) {
  if (doc_ids == nullptr || vectors == nullptr || expected_dimension == 0) {
    return false;
  }

  std::ifstream raw_file(segment.vector_path, std::ios::in | std::ios::binary);
  std::ifstream docid_file(segment.docid_path, std::ios::in | std::ios::binary);
  if (!raw_file.is_open() || !docid_file.is_open()) return false;

  uint32_t rows = 0;
  uint32_t dimension = 0;
  uint64_t docid_rows = 0;
  if (!read_binary_value(&raw_file, &rows) ||
      !read_binary_value(&raw_file, &dimension) ||
      !read_binary_value(&docid_file, &docid_rows) ||
      rows != segment.row_count || docid_rows != segment.row_count ||
      dimension != expected_dimension) {
    return false;
  }

  std::vector<uint64_t> segment_doc_ids(docid_rows);
  std::vector<float> segment_vectors(static_cast<size_t>(rows) * dimension);
  if (!segment_doc_ids.empty()) {
    docid_file.read(reinterpret_cast<char *>(segment_doc_ids.data()),
                    static_cast<std::streamsize>(segment_doc_ids.size() *
                                                 sizeof(uint64_t)));
  }
  if (!segment_vectors.empty()) {
    raw_file.read(reinterpret_cast<char *>(segment_vectors.data()),
                  static_cast<std::streamsize>(segment_vectors.size() *
                                               sizeof(float)));
  }
  if (!raw_file.good() || !docid_file.good()) return false;

  doc_ids->insert(doc_ids->end(), segment_doc_ids.begin(),
                  segment_doc_ids.end());
  vectors->insert(vectors->end(), segment_vectors.begin(),
                  segment_vectors.end());
  return true;
}

bool parse_manifest(const char *manifest_path, uint32_t dimension,
                    std::vector<raw_segment> *segments,
                    uint64_t *expected_count) {
  if (manifest_path == nullptr || segments == nullptr ||
      expected_count == nullptr || dimension == 0) {
    return false;
  }

  std::ifstream manifest(manifest_path, std::ios::in);
  if (!manifest.is_open()) return false;

  std::string line;
  if (!std::getline(manifest, line) || line != k_raw_manifest_header) {
    return false;
  }

  bool saw_dimension = false;
  bool saw_count = false;
  while (std::getline(manifest, line)) {
    std::istringstream input(line);
    std::string tag;
    std::getline(input, tag, '\t');
    if (tag == "dimension") {
      uint32_t parsed_dimension = 0;
      input >> parsed_dimension;
      if (parsed_dimension != dimension) return false;
      saw_dimension = true;
    } else if (tag == "count") {
      input >> *expected_count;
      saw_count = true;
    } else if (tag == "segment") {
      raw_segment segment;
      std::string row_count_text;
      std::getline(input, segment.docid_path, '\t');
      std::getline(input, segment.vector_path, '\t');
      std::getline(input, row_count_text, '\t');
      if (segment.docid_path.empty() || segment.vector_path.empty() ||
          row_count_text.empty()) {
        return false;
      }
      segment.row_count = std::stoull(row_count_text);
      segments->push_back(std::move(segment));
    } else {
      return false;
    }
  }
  return saw_dimension && saw_count;
}

float distance_for_metric(const fake_index &index, const float *query,
                          size_t row) {
  const float *candidate = &index.vectors[row * index.dimension];
  float dot = 0.0F;
  float query_norm = 0.0F;
  float candidate_norm = 0.0F;
  float l2 = 0.0F;
  for (size_t dim = 0; dim < index.dimension; ++dim) {
    const float lhs = query[dim];
    const float rhs = candidate[dim];
    const float diff = lhs - rhs;
    l2 += diff * diff;
    dot += lhs * rhs;
    query_norm += lhs * lhs;
    candidate_norm += rhs * rhs;
  }

  if (index.metric_type == 0) {
    if (query_norm == 0.0F || candidate_norm == 0.0F) {
      return std::numeric_limits<float>::infinity();
    }
    return 1.0F - dot / (std::sqrt(query_norm) * std::sqrt(candidate_norm));
  }
  if (index.metric_type == 1) return -dot;
  return l2;
}

int32_t search_one(const fake_index *index, const float *query, size_t dimension,
                   uint32_t top_k, uint64_t *labels, float *distances) {
  if (index == nullptr || query == nullptr || labels == nullptr ||
      distances == nullptr || dimension != index->dimension) {
    return -1;
  }
  if (top_k == 0 || index->doc_ids.empty()) return 0;

  std::vector<std::pair<float, uint64_t>> ranked;
  ranked.reserve(index->doc_ids.size());
  for (size_t row = 0; row < index->doc_ids.size(); ++row) {
    ranked.push_back({distance_for_metric(*index, query, row),
                      index->doc_ids[row]});
  }
  std::sort(ranked.begin(), ranked.end(), [](const auto &lhs,
                                             const auto &rhs) {
    if (lhs.first != rhs.first) return lhs.first < rhs.first;
    return lhs.second < rhs.second;
  });

  const size_t result_count = std::min<size_t>(top_k, ranked.size());
  for (size_t idx = 0; idx < result_count; ++idx) {
    distances[idx] = ranked[idx].first;
    labels[idx] = ranked[idx].second;
  }
  return static_cast<int32_t>(result_count);
}

}  // namespace

extern "C" bool mysql_vector_diskann_offline_build(
    const char *index_prefix, const uint8_t *docid_data, size_t docid_width,
    size_t docid_stride, const uint8_t *vector_data, size_t dimension,
    size_t vector_stride, size_t row_count, int32_t, uint32_t, uint32_t,
    uint32_t build_threads, double, uint32_t, uint32_t,
    uint32_t build_blas_threads) {
  if (index_prefix == nullptr || docid_data == nullptr ||
      vector_data == nullptr || docid_width != sizeof(uint64_t) ||
      docid_stride < sizeof(uint64_t) || dimension == 0 ||
      vector_stride < dimension * sizeof(float)) {
    return false;
  }

  std::vector<uint64_t> doc_ids;
  std::vector<float> vectors;
  doc_ids.reserve(row_count);
  vectors.reserve(row_count * dimension);
  for (size_t row = 0; row < row_count; ++row) {
    uint64_t doc_id = 0;
    std::memcpy(&doc_id, docid_data + row * docid_stride, sizeof(doc_id));
    doc_ids.push_back(doc_id);
    const auto *row_vector =
        reinterpret_cast<const float *>(vector_data + row * vector_stride);
    vectors.insert(vectors.end(), row_vector, row_vector + dimension);
  }
  return write_index_files(index_prefix, dimension, doc_ids, vectors) &&
         write_build_options(index_prefix, build_threads, build_blas_threads,
                             "flat_arrays");
}

extern "C" bool mysql_vector_diskann_offline_build_from_manifest(
    const char *index_prefix, const char *manifest_path, uint32_t dimension,
    int32_t, uint32_t, uint32_t, uint32_t build_threads, double, uint32_t,
    uint32_t, uint32_t build_blas_threads) {
  std::vector<raw_segment> segments;
  uint64_t expected_count = 0;
  if (index_prefix == nullptr ||
      !parse_manifest(manifest_path, dimension, &segments, &expected_count)) {
    return false;
  }

  std::vector<uint64_t> doc_ids;
  std::vector<float> vectors;
  for (const raw_segment &segment : segments) {
    if (!append_segment(segment, dimension, &doc_ids, &vectors)) return false;
  }
  if (doc_ids.size() != expected_count) return false;
  return write_index_files(index_prefix, dimension, doc_ids, vectors) &&
         write_build_options(index_prefix, build_threads, build_blas_threads,
                             segments.size() == 1 ? "manifest_direct"
                                                  : "manifest_merge");
}

extern "C" bool mysql_vector_diskann_offline_build_from_native_pq(
    const char *index_prefix, const char *manifest_path,
    const char *pq_pivots_path, const char *pq_compressed_path,
    const char *disk_pq_pivots_path, const char *disk_pq_compressed_path,
    uint32_t dimension, int32_t, uint32_t, uint32_t, uint32_t build_threads,
    double, uint32_t, uint32_t, uint32_t build_blas_threads,
    uint32_t disk_pq_dims,
    uint32_t, uint32_t) {
  std::vector<raw_segment> segments;
  uint64_t expected_count = 0;
  if (index_prefix == nullptr || pq_pivots_path == nullptr ||
      pq_compressed_path == nullptr ||
      !std::filesystem::exists(pq_pivots_path) ||
      !std::filesystem::exists(pq_compressed_path) ||
      (disk_pq_dims != 0 &&
       (disk_pq_pivots_path == nullptr ||
        disk_pq_compressed_path == nullptr ||
        !std::filesystem::exists(disk_pq_pivots_path) ||
        !std::filesystem::exists(disk_pq_compressed_path))) ||
      !parse_manifest(manifest_path, dimension, &segments, &expected_count)) {
    return false;
  }

  std::vector<uint64_t> doc_ids;
  std::vector<float> vectors;
  for (const raw_segment &segment : segments) {
    if (!append_segment(segment, dimension, &doc_ids, &vectors)) return false;
  }
  if (doc_ids.size() != expected_count) return false;
  return write_index_files(index_prefix, dimension, doc_ids, vectors) &&
         write_fake_medoids(index_prefix, doc_ids) &&
         write_build_options(index_prefix, build_threads, build_blas_threads,
                             "native_pq_bridge");
}

extern "C" const void *mysql_vector_diskann_offline_load(
    const char *index_prefix, int32_t metric_type, uint32_t, uint32_t, uint32_t,
    uint32_t) {
  auto index = std::make_unique<fake_index>();
  if (index_prefix == nullptr || !read_index_files(index_prefix, index.get())) {
    return nullptr;
  }
  index->metric_type = metric_type;
  return index.release();
}

extern "C" int32_t mysql_vector_diskann_offline_search(
    const void *handle, const float *query, size_t dimension, uint32_t top_k,
    uint32_t, uint32_t, uint64_t *labels, float *distances) {
  return search_one(static_cast<const fake_index *>(handle), query, dimension,
                    top_k, labels, distances);
}

extern "C" int32_t mysql_vector_diskann_offline_search_batch(
    const void *handle, const float *queries, size_t query_count,
    size_t dimension, uint32_t top_k, uint32_t, uint32_t, uint32_t,
    uint64_t *labels, float *distances, uint32_t *result_counts) {
  if (queries == nullptr || labels == nullptr || distances == nullptr) {
    return -1;
  }

  for (size_t query_idx = 0; query_idx < query_count; ++query_idx) {
    const int32_t result_count = search_one(
        static_cast<const fake_index *>(handle),
        queries + query_idx * dimension, dimension, top_k,
        labels + query_idx * top_k, distances + query_idx * top_k);
    if (result_count < 0) return result_count;
    if (result_counts != nullptr) {
      result_counts[query_idx] = static_cast<uint32_t>(result_count);
    }
  }
  return static_cast<int32_t>(query_count);
}

extern "C" uint64_t mysql_vector_diskann_offline_card(const void *handle) {
  const auto *index = static_cast<const fake_index *>(handle);
  return index == nullptr ? 0 : index->doc_ids.size();
}

extern "C" void mysql_vector_diskann_offline_drop(const void *handle) {
  delete static_cast<const fake_index *>(handle);
}
