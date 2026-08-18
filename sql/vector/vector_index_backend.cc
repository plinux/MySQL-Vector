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

#include "sql/vector/vector_index_backend.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "sql/vector/vector_index_build_options.h"

#ifdef HAVE_FAISS
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIDMap.h>
#include <faiss/clone_index.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>
#endif

#ifdef HAVE_HNSWLIB
#include <hnswlib/hnswalg.h>
#include <hnswlib/space_ip.h>
#include <hnswlib/space_l2.h>
#endif

#include "sql/vector/vector_index_backend_common.h"
#include "sql/vector/vector_index_backend_internal.h"
#include "sql/vector/vector_index_runtime_config.h"
#include "sql/vector/vector_load_file.h"
#include "sql/mysqld.h"
#include "sql/vector/vector_status.h"
#include "my_dbug.h"

namespace vector_index::detail {

const char *kFaissExternalSnapshotHeader =
    "mysql-vector-faiss-external-v1";
const char *kFaissExternalManifestHeader =
    "mysql-vector-faiss-external-manifest-v1";
const char *kDiskAnnExternalSnapshotHeader =
    "mysql-vector-diskann-external-v1";
const char *kDiskAnnExternalManifestHeader =
    "mysql-vector-diskann-external-manifest-v1";
const char *kVectorIndexDirectory = "mysql_vector_index";
const char *kFaissExternalManifestFilename =
    "faiss_external.manifest.v1";
const char *kDiskAnnExternalManifestFilename =
    "diskann_external.manifest.v1";
const char *kFaissExternalSnapshotPrefix =
    "faiss_external.snapshot.";
const char *kDiskAnnExternalSnapshotPrefix =
    "diskann_external.snapshot.";
const char *kFaissExternalSnapshotSuffix = ".v1";

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
std::mutex g_faiss_external_snapshot_root_mutex;
std::string g_faiss_external_snapshot_root_override;
#endif

bool ensure_parent_directory(const std::string &path);

bool load_external_manifest_generation_from_file(const std::string &path,
                                                 const char *expected_header,
                                                 uint64_t *generation,
                                                 bool *exists) {
  DBUG_EXECUTE_IF("vector_backend_fail_load_manifest_generation",
                  return false;);
  if (generation == nullptr || exists == nullptr || expected_header == nullptr) {
    return false;
  }
  *generation = 0;
  *exists = false;

  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.good()) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) && !ec) return true;
    return false;
  }

  std::string line;
  if (!std::getline(file, line) || line != expected_header) {
    return false;
  }
  if (!std::getline(file, line)) return false;

  uint64_t parsed_generation = 0;
  if (!parse_uint64(line, &parsed_generation) || parsed_generation == 0) {
    return false;
  }
  if (file.bad()) return false;

  *generation = parsed_generation;
  *exists = true;
  return true;
}

bool save_external_manifest_generation_to_file(const std::string &path,
                                               const char *header,
                                               uint64_t generation) {
  DBUG_EXECUTE_IF("vector_backend_fail_save_manifest_generation",
                  return false;);
  if (generation == 0 || header == nullptr) return false;
  if (!ensure_parent_directory(path)) return false;

  const std::string temp_path = path + ".tmp";
  std::ofstream file(temp_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.good()) return false;
  file << header << "\n" << generation << "\n";
  file.close();
  if (!file) {
    std::remove(temp_path.c_str());
    return false;
  }
  DBUG_EXECUTE_IF("vector_backend_fail_save_manifest_generation_rename", {
    std::remove(temp_path.c_str());
    return false;
  };);
  if (std::rename(temp_path.c_str(), path.c_str()) != 0) {
    std::remove(temp_path.c_str());
    return false;
  }
  return true;
}

std::string vector_index_root_path() {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  std::lock_guard<std::mutex> guard(g_faiss_external_snapshot_root_mutex);
  if (!g_faiss_external_snapshot_root_override.empty()) {
    return g_faiss_external_snapshot_root_override;
  }
#endif

  if (mysql_real_data_home[0] == '\0') {
    return "";
  }

  std::string root(mysql_real_data_home);
  if (!root.empty()) {
    const char tail = root.back();
    if (tail != '/' && tail != '\\') root.push_back('/');
  }
  root.append(kVectorIndexDirectory);
  return root;
}

const char *external_snapshot_header(
    vector_index::external_sidecar_profile profile) {
  return profile == vector_index::external_sidecar_profile::kDiskAnn
             ? kDiskAnnExternalSnapshotHeader
             : kFaissExternalSnapshotHeader;
}

const char *external_manifest_header(
    vector_index::external_sidecar_profile profile) {
  return profile == vector_index::external_sidecar_profile::kDiskAnn
             ? kDiskAnnExternalManifestHeader
             : kFaissExternalManifestHeader;
}

const char *external_manifest_filename(
    vector_index::external_sidecar_profile profile) {
  return profile == vector_index::external_sidecar_profile::kDiskAnn
             ? kDiskAnnExternalManifestFilename
             : kFaissExternalManifestFilename;
}

const char *external_snapshot_prefix(
    vector_index::external_sidecar_profile profile) {
  return profile == vector_index::external_sidecar_profile::kDiskAnn
             ? kDiskAnnExternalSnapshotPrefix
             : kFaissExternalSnapshotPrefix;
}

std::string encoded_index_name(const std::string &index_name) {
  return encode_hex_bytes(index_name.data(), index_name.size());
}

std::string external_snapshot_directory(const std::string &index_name) {
  const std::string root = vector_index_root_path();
  if (root.empty() || index_name.empty()) return "";

  const std::string encoded = encoded_index_name(index_name);
  std::string directory = root;
  if (!directory.empty()) {
    const char tail = directory.back();
    if (tail != '/' && tail != '\\') directory.push_back('/');
  }
  directory.append(encoded);
  return directory;
}

std::string faiss_external_snapshot_directory(const std::string &index_name) {
  return external_snapshot_directory(index_name);
}

std::string external_manifest_path(
    const std::string &index_name,
    vector_index::external_sidecar_profile profile) {
  const std::string directory = external_snapshot_directory(index_name);
  if (directory.empty()) return "";
  std::string path = directory;
  path.push_back('/');
  path.append(external_manifest_filename(profile));
  return path;
}

bool ensure_parent_directory(const std::string &path) {
  const std::filesystem::path file_path(path);
  const std::filesystem::path parent = file_path.parent_path();
  if (parent.empty()) return true;

  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  return !ec;
}

bool remove_if_exists(const std::string &path) {
  DBUG_EXECUTE_IF("vector_backend_fail_remove_if_exists", return false;);
  if (path.empty()) return true;
  if (std::remove(path.c_str()) != 0 && errno != ENOENT) return false;
  return true;
}

std::string quarantine_path_for(const std::string &path) {
  if (path.empty()) return "";
  return path + ".corrupt";
}

bool quarantine_file_if_exists(const std::string &path) {
  DBUG_EXECUTE_IF("vector_backend_fail_quarantine_file_if_exists",
                  return false;);
  if (path.empty()) return true;
  const std::string quarantine_path = quarantine_path_for(path);
  std::remove(quarantine_path.c_str());
  if (std::rename(path.c_str(), quarantine_path.c_str()) == 0) return true;
  return errno == ENOENT;
}

bool remove_generated_snapshots_with_prefix(const std::string &directory,
                                            const std::string &prefix) {
  DBUG_EXECUTE_IF("vector_backend_fail_remove_generated_snapshots",
                  return false;);
  if (directory.empty()) return true;

  std::error_code ec;
  const std::filesystem::path dir(directory);
  if (!std::filesystem::exists(dir, ec)) return !ec;
  if (ec) return false;
  if (!std::filesystem::is_directory(dir, ec)) return !ec;
  if (ec) return false;

  for (const auto &entry : std::filesystem::directory_iterator(
           dir, std::filesystem::directory_options::skip_permission_denied,
           ec)) {
    if (ec) return false;
    if (!entry.is_regular_file(ec)) {
      if (ec) return false;
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (!starts_with(filename, prefix) ||
        !ends_with(filename, kFaissExternalSnapshotSuffix)) {
      continue;
    }
    if (std::remove(entry.path().string().c_str()) != 0) return false;
  }
  return true;
}

bool remove_dir_if_empty(const std::string &path) {
  DBUG_EXECUTE_IF("vector_backend_fail_remove_dir_if_empty", return false;);
  if (path.empty()) return true;
  std::error_code ec;
  const std::filesystem::path dir(path);
  if (!std::filesystem::exists(dir, ec)) return !ec;
  if (ec) return false;
  if (!std::filesystem::is_directory(dir, ec)) return !ec;
  if (ec) return false;
  if (!std::filesystem::is_empty(dir, ec)) return !ec;
  if (ec) return false;
  std::filesystem::remove(dir, ec);
  return !ec;
}

bool remove_external_sidecar_artifacts(
    const std::string &index_name, external_sidecar_profile profile,
    bool remove_snapshot_directory) {
  if (index_name.empty()) return true;

  const std::string snapshot_directory = external_snapshot_directory(index_name);
  const std::string manifest_path = external_manifest_path(index_name, profile);
  if (!remove_if_exists(manifest_path)) return false;
  if (!remove_if_exists(quarantine_path_for(manifest_path))) return false;
  if (!remove_generated_snapshots_with_prefix(
          snapshot_directory, external_snapshot_prefix(profile))) {
    return false;
  }

  if (remove_snapshot_directory && !remove_dir_if_empty(snapshot_directory))
    return false;
  return true;
}

std::string diskann_store_directory(const std::string &index_name) {
  const std::string directory = external_snapshot_directory(index_name);
  if (directory.empty()) return "";
  return directory + "/diskann_external.store";
}

bool remove_diskann_external_artifacts(const std::string &index_name,
                                       bool remove_native_store) {
  if (!remove_external_sidecar_artifacts(index_name,
                                        external_sidecar_profile::kDiskAnn,
                                        false)) {
    return false;
  }

  if (remove_native_store) {
    const std::string native_store_directory =
        diskann_store_directory(index_name);
    std::error_code ec;
    if (!native_store_directory.empty()) {
      std::filesystem::remove_all(native_store_directory, ec);
      if (ec) return false;
    }
  }

  return remove_dir_if_empty(external_snapshot_directory(index_name));
}

}  // namespace vector_index::detail

using vector_index::detail::check_dimension;
using vector_index::detail::compute_distance;
using vector_index::detail::decode_hex_bytes;
using vector_index::detail::encode_hex_bytes;
using vector_index::detail::ends_with;
using vector_index::detail::normalize_token;
using vector_index::detail::parse_uint64;
using vector_index::detail::starts_with;
using vector_index::detail::kFaissExternalSnapshotHeader;
using vector_index::detail::kFaissExternalManifestHeader;
using vector_index::detail::kDiskAnnExternalSnapshotHeader;
using vector_index::detail::kDiskAnnExternalManifestHeader;
using vector_index::detail::kVectorIndexDirectory;
using vector_index::detail::kFaissExternalManifestFilename;
using vector_index::detail::kDiskAnnExternalManifestFilename;
using vector_index::detail::kFaissExternalSnapshotPrefix;
using vector_index::detail::kDiskAnnExternalSnapshotPrefix;
using vector_index::detail::kFaissExternalSnapshotSuffix;
using vector_index::detail::ensure_parent_directory;
using vector_index::detail::load_external_manifest_generation_from_file;
using vector_index::detail::save_external_manifest_generation_to_file;
using vector_index::detail::vector_index_root_path;
using vector_index::detail::external_snapshot_header;
using vector_index::detail::external_manifest_header;
using vector_index::detail::external_manifest_filename;
using vector_index::detail::external_snapshot_prefix;
using vector_index::detail::encoded_index_name;
using vector_index::detail::external_snapshot_directory;
using vector_index::detail::faiss_external_snapshot_directory;
using vector_index::detail::external_manifest_path;
using vector_index::detail::remove_if_exists;
using vector_index::detail::quarantine_path_for;
using vector_index::detail::quarantine_file_if_exists;
using vector_index::detail::remove_generated_snapshots_with_prefix;
using vector_index::detail::remove_dir_if_empty;
using vector_index::detail::diskann_store_directory;
using vector_index::detail::remove_external_sidecar_artifacts;
using vector_index::detail::remove_diskann_external_artifacts;

namespace vector_index {

namespace {

constexpr vector_library_status k_vector_library_statuses[] = {
#ifdef HAVE_FAISS
    {"faiss", true, false, false, "Faiss backend is compiled in"},
#else
    {"faiss", false, false, false, "Faiss backend is not compiled in"},
#endif
#ifdef HAVE_DISKANN
#if defined(MYSQL_VECTOR_DISKANN_OFFLINE_STATIC_LINKED) || \
    defined(MYSQL_VECTOR_DISKANN_OFFLINE_DEFAULT_LIB)
    {"diskann", true, true, true, "DiskANN backend is compiled in"},
#else
    {"diskann", true, true, false, "DiskANN backend is compiled in"},
#endif
#else
    {"diskann", false, true, false, "DiskANN backend is not compiled in"},
#endif
#ifdef HAVE_HNSWLIB
    {"hnsw", true, false, false, "hnswlib backend is compiled in"},
#else
    {"hnsw", false, false, false, "hnswlib backend is not compiled in"},
#endif
};

constexpr bool any_vector_library_supported() {
#if defined(HAVE_FAISS) || defined(HAVE_DISKANN) || defined(HAVE_HNSWLIB)
  return true;
#else
  return false;
#endif
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
#ifndef NDEBUG
bool g_native_provider_supported_for_testing = true;
#else
bool g_native_provider_supported_for_testing = false;
#endif
#endif

bool debug_native_provider_supported() {
#ifndef NDEBUG
  return any_vector_library_supported();
#else
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  return g_native_provider_supported_for_testing &&
         any_vector_library_supported();
#else
  return false;
#endif
#endif
}

bool read_committed_entries(
    const committed_entry_reader &reader,
    std::unordered_map<uint64_t, vector_data> *entries) {
  if (!reader || entries == nullptr) return false;

  entries->clear();
  return reader([&](uint64_t doc_id, const vector_data &vector) {
    (*entries)[doc_id] = vector;
    return true;
  });
}

bool collect_full_rerank_candidates(
    const backend &source, size_t requested_candidate_count,
    std::vector<search_result> *candidates) {
  const size_t entry_count = source.entry_count();
  if (candidates == nullptr || requested_candidate_count < entry_count) {
    return false;
  }

  std::vector<uint64_t> doc_ids;
  if (!source.collect_doc_ids(&doc_ids) || doc_ids.size() != entry_count) {
    return false;
  }

  std::sort(doc_ids.begin(), doc_ids.end());
  if (std::adjacent_find(doc_ids.begin(), doc_ids.end()) != doc_ids.end()) {
    return false;
  }

  candidates->clear();
  candidates->reserve(doc_ids.size());
  for (uint64_t doc_id : doc_ids) {
    candidates->push_back({doc_id, 0.0});
  }
  return true;
}

}  // namespace

bool backend::load_committed_entries(
    const std::unordered_map<uint64_t, vector_data> &entries) {
  if (!supports_mutations()) return entries.empty();
  for (const auto &entry : entries) {
    if (!upsert(entry.first, entry.second)) return false;
  }
  return true;
}

bool backend::load_committed_entries_from_reader(
    const committed_entry_reader &reader) {
  std::unordered_map<uint64_t, vector_data> entries;
  if (!read_committed_entries(reader, &entries)) return false;

  return load_committed_entries(entries);
}

bool backend::rebuild_from_committed_entries_from_reader(
    const committed_entry_reader &reader) {
  std::unordered_map<uint64_t, vector_data> entries;
  if (!read_committed_entries(reader, &entries)) return false;

  return rebuild_from_committed_entries(entries);
}

bool backend::rebuild_from_committed_entry_source(
    const committed_entry_source &source) {
  return rebuild_from_committed_entries_from_reader(source.reader);
}

bool backend::rebuild_from_raw_segments(
    const raw_vector_segment_reader &reader) {
  if (!reader) return false;

  const committed_entry_reader entry_reader =
      [this, &reader](const committed_entry_visitor &visitor) {
        return reader([this, &visitor](const raw_vector_segment &segment) {
          if (segment.dimension != dimension()) return false;
          vector_load_file_info info;
          std::string error;
          return read_fbin_vectors(
                     segment.vector_path, segment.docid_path, dimension(),
                     &info, &error,
                     [&visitor](uint64_t doc_id, const float *values,
                                size_t row_dimension) {
                       return visitor(
                           doc_id,
                           vector_data(values, values + row_dimension));
                     }) &&
                 info.row_count == segment.row_count &&
                 info.dimension == segment.dimension;
        });
      };
  return rebuild_from_committed_entries_from_reader(entry_reader);
}

bool backend::build_segment_from_raw(const segment_build_input &input,
                                     segment_build_result *result) {
  if (result == nullptr || input.segment_id == 0 ||
      input.segment.dimension != dimension()) {
    return false;
  }

  const raw_vector_segment segment = input.segment;
  const raw_vector_segment_reader reader =
      [&segment](const raw_vector_segment_visitor &visitor) {
        if (!visitor) return false;
        return visitor(segment);
      };
  if (!rebuild_from_raw_segments(reader)) return false;

  result->segment_id = input.segment_id;
  result->generation = input.generation;
  result->row_count = input.segment.row_count;
  result->payload_size = input.segment.bytes;
  result->vector_path = input.segment.vector_path;
  result->docid_path = input.segment.docid_path;
  result->artifact_prefix = input.artifact_prefix;
  result->ready = true;
  result->diagnostics = build_diagnostics();
  return true;
}

bool backend::load_segment_handle(const segment_build_result &result) {
  return result.ready && result.segment_id != 0;
}

bool backend::recover_committed_entries_from_reader(
    const committed_entry_reader &reader) {
  std::unordered_map<uint64_t, vector_data> entries;
  if (!read_committed_entries(reader, &entries)) return false;

  return recover_committed_entries(entries);
}

bool backend::search_batch(
    const std::vector<vector_data> &queries, size_t top_k,
    std::vector<std::vector<search_result>> *results) const {
  if (results == nullptr) return false;
  results->clear();
  results->resize(queries.size());
  for (size_t i = 0; i < queries.size(); ++i) {
    if (!search(queries[i], top_k, &(*results)[i])) return false;
  }
  return true;
}

bool backend::search_with_options(
    const vector_data &query, size_t top_k,
    const backend_search_options &options [[maybe_unused]],
    std::vector<search_result> *results) const {
  return search(query, top_k, results);
}

bool backend::search_batch_with_options(
    const std::vector<vector_data> &queries, size_t top_k,
    const backend_search_options &options [[maybe_unused]],
    std::vector<std::vector<search_result>> *results) const {
  return search_batch(queries, top_k, results);
}

bool backend::search_for_rerank(
    const vector_data &query, size_t top_k, size_t candidate_top_k,
    std::vector<search_result> *results) const {
  if (results == nullptr) return false;
  const size_t requested_candidate_count = std::max(top_k, candidate_top_k);
  if (candidate_top_k > top_k &&
      collect_full_rerank_candidates(*this, requested_candidate_count,
                                     results)) {
    return true;
  }
  return search(query, requested_candidate_count, results);
}

bool backend::search_batch_for_rerank(
    const std::vector<vector_data> &queries, size_t top_k,
    size_t candidate_top_k,
    std::vector<std::vector<search_result>> *results) const {
  if (results == nullptr) return false;
  results->clear();
  if (queries.empty()) return true;

  const size_t requested_candidate_count = std::max(top_k, candidate_top_k);
  std::vector<search_result> candidates;
  if (candidate_top_k > top_k &&
      collect_full_rerank_candidates(*this, requested_candidate_count,
                                     &candidates)) {
    results->assign(queries.size(), candidates);
    return true;
  }
  return search_batch(queries, requested_candidate_count, results);
}

bool backend::collect_doc_ids(std::vector<uint64_t> *doc_ids
                              [[maybe_unused]]) const {
  return false;
}

memory_backend::memory_backend(size_t dimension, metric_type metric)
    : m_dimension(dimension), m_metric(metric) {}

bool memory_backend::upsert(uint64_t doc_id, const vector_data &vector) {
  if (!check_dimension(vector, m_dimension)) return false;
  m_entries[doc_id] = vector;
  return true;
}

bool memory_backend::erase(uint64_t doc_id) {
  m_entries.erase(doc_id);
  return true;
}

void memory_backend::reset() { m_entries.clear(); }

bool memory_backend::contains(uint64_t doc_id) const {
  return m_entries.find(doc_id) != m_entries.end();
}

bool memory_backend::snapshot_entries(
    std::unordered_map<uint64_t, vector_data> *entries) const {
  if (entries == nullptr) return false;
  *entries = m_entries;
  return true;
}

bool memory_backend::search(const vector_data &query, size_t top_k,
                           std::vector<search_result> *results) const {
  if (results == nullptr) return false;
  results->clear();
  if (top_k == 0) return true;
  if (!check_dimension(query, m_dimension)) return false;

  for (const auto &entry : m_entries) {
    double distance = 0.0;
    if (!compute_distance(m_metric, query, entry.second, &distance)) return false;
    results->push_back(search_result{entry.first, distance});
  }

  const size_t count = std::min(top_k, results->size());
  std::partial_sort(
      results->begin(), results->begin() + count, results->end(),
      [](const search_result &lhs, const search_result &rhs) {
        if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
        return lhs.doc_id < rhs.doc_id;
      });
  results->resize(count);
  return true;
}

bool external_backend::upsert(uint64_t doc_id, const vector_data &vector) {
  if (!check_dimension(vector, m_dimension)) return false;
  m_entries[doc_id] = vector;
  return true;
}

bool external_backend::erase(uint64_t doc_id) {
  m_entries.erase(doc_id);
  return true;
}

bool external_backend::search(const vector_data &query, size_t top_k,
                             std::vector<search_result> *results) const {
  if (results == nullptr) return false;
  results->clear();
  if (top_k == 0) return true;
  if (!check_dimension(query, m_dimension)) return false;

  for (const auto &entry : m_entries) {
    double distance = 0.0;
    if (!compute_distance(m_metric, query, entry.second, &distance)) return false;
    results->push_back(search_result{entry.first, distance});
  }

  const size_t count = std::min(top_k, results->size());
  std::partial_sort(
      results->begin(), results->begin() + count, results->end(),
      [](const search_result &lhs, const search_result &rhs) {
        if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
        return lhs.doc_id < rhs.doc_id;
      });
  results->resize(count);
  return true;
}

void external_backend::reset() { m_entries.clear(); }

std::unique_ptr<backend> create_backend(size_t dimension, metric_type metric,
                                       backend_mode mode,
                                       backend_provider provider,
                                       const std::string &index_name) {
  return create_backend(dimension, metric, mode, provider,
                        index_consistency_mode::kTransactional, index_name);
}

std::unique_ptr<backend> create_backend(size_t dimension, metric_type metric,
                                       backend_mode mode,
                                       backend_provider provider,
                                       index_consistency_mode consistency_mode,
                                       const std::string &index_name) {
  if (!backend_provider_supported(provider)) return nullptr;

  (void)consistency_mode;
  switch (provider) {
    case backend_provider::kNative:
      if (mode == backend_mode::kMemory)
        return std::make_unique<memory_backend>(dimension, metric);
      return std::make_unique<external_backend>(dimension, metric);

    case backend_provider::kFaiss:
      return std::make_unique<faiss_backend>(dimension, metric, mode, index_name);

    case backend_provider::kDiskAnn:
      if (mode == backend_mode::kExternal)
        return std::make_unique<diskann_backend>(dimension, metric, mode,
                                                index_name);
      return nullptr;

    case backend_provider::kHnswlib:
      if (mode == backend_mode::kMemory)
        return std::make_unique<hnswlib_backend>(dimension, metric, mode);
      return nullptr;
  }

  return nullptr;
}

const vector_library_status *vector_library_statuses(size_t *count) {
  if (count != nullptr)
    *count = sizeof(k_vector_library_statuses) /
             sizeof(k_vector_library_statuses[0]);
  return k_vector_library_statuses;
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool native_provider_supported_for_testing() {
  return g_native_provider_supported_for_testing;
}

void set_native_provider_supported_for_testing(bool supported) {
  g_native_provider_supported_for_testing = supported;
}
#endif

bool backend_provider_supported(backend_provider provider) {
  switch (provider) {
    case backend_provider::kNative:
      return debug_native_provider_supported();
    case backend_provider::kFaiss:
#ifdef HAVE_FAISS
      return true;
#else
      return false;
#endif
    case backend_provider::kDiskAnn:
#ifdef HAVE_DISKANN
      return true;
#else
      return false;
#endif
    case backend_provider::kHnswlib:
#ifdef HAVE_HNSWLIB
      return true;
#else
      return false;
#endif
  }
  return false;
}

bool default_backend_provider(backend_provider *provider) {
  if (provider == nullptr) return false;

  switch (global_vector_default_library()) {
    case vector_default_library::kNone:
      return false;
    case vector_default_library::kDiskAnn:
      *provider = backend_provider::kDiskAnn;
      return backend_provider_supported(*provider);
    case vector_default_library::kHnsw:
      *provider = backend_provider::kHnswlib;
      return backend_provider_supported(*provider);
    case vector_default_library::kFaiss:
      *provider = backend_provider::kFaiss;
      return backend_provider_supported(*provider);
  }
  return false;
}

bool default_backend_mode_for_provider(backend_provider provider,
                                       backend_mode *mode) {
  if (mode == nullptr) return false;

  switch (provider) {
    case backend_provider::kNative:
    case backend_provider::kHnswlib:
      *mode = backend_mode::kMemory;
      return true;
    case backend_provider::kFaiss:
    case backend_provider::kDiskAnn:
      *mode = backend_mode::kExternal;
      return true;
  }
  return false;
}

bool parse_metric(const std::string &value, metric_type *metric) {
  if (metric == nullptr) return false;

  const std::string token = normalize_token(value);
  if (token == "euclidean" || token == "l2") {
    *metric = metric_type::kEuclidean;
    return true;
  }
  if (token == "cosine") {
    *metric = metric_type::kCosine;
    return true;
  }
  if (token == "inner_product" || token == "ip") {
    *metric = metric_type::kInnerProduct;
    return true;
  }

  return false;
}

bool parse_backend_mode(const std::string &value, backend_mode *mode) {
  if (mode == nullptr) return false;

  const std::string token = normalize_token(value);
  if (token == "memory") {
    *mode = backend_mode::kMemory;
    return true;
  }
  if (token == "external") {
    *mode = backend_mode::kExternal;
    return true;
  }

  return false;
}

bool parse_backend_provider(const std::string &value, backend_provider *provider) {
  if (provider == nullptr) return false;

  const std::string token = normalize_token(value);
  if (token == "native") {
    *provider = backend_provider::kNative;
    return true;
  }
  if (token == "faiss") {
    *provider = backend_provider::kFaiss;
    return true;
  }
  if (token == "diskann") {
    *provider = backend_provider::kDiskAnn;
    return true;
  }
  if (token == "hnswlib" || token == "hnsw") {
    *provider = backend_provider::kHnswlib;
    return true;
  }

  return false;
}

bool parse_diskann_build_mode(const std::string &value,
                              diskann_build_mode *build_mode) {
  if (build_mode == nullptr) return false;

  const std::string token = normalize_token(value);
  if (token == "auto") {
    *build_mode = diskann_build_mode::kAuto;
    return true;
  }
  if (token == "serial") {
    *build_mode = diskann_build_mode::kSerial;
    return true;
  }
  if (token == "offline") {
    *build_mode = diskann_build_mode::kOffline;
    return true;
  }

  return false;
}

bool parse_index_consistency_mode(const std::string &value,
                                  index_consistency_mode *consistency_mode) {
  if (consistency_mode == nullptr) return false;

  const std::string token = normalize_token(value);
  if (token == "transactional") {
    *consistency_mode = index_consistency_mode::kTransactional;
    return true;
  }
  if (token == "standalone" || token == "non_transactional" ||
      token == "non-transactional") {
    *consistency_mode = index_consistency_mode::kStandalone;
    return true;
  }

  return false;
}

const char *metric_to_string(metric_type metric) {
  switch (metric) {
    case metric_type::kEuclidean:
      return "euclidean";
    case metric_type::kCosine:
      return "cosine";
    case metric_type::kInnerProduct:
      return "inner_product";
  }
  return "unknown";
}

const char *backend_mode_to_string(backend_mode mode) {
  switch (mode) {
    case backend_mode::kMemory:
      return "memory";
    case backend_mode::kExternal:
      return "external";
  }
  return "unknown";
}

const char *backend_provider_to_string(backend_provider provider) {
  switch (provider) {
    case backend_provider::kNative:
      return "native";
    case backend_provider::kFaiss:
      return "faiss";
    case backend_provider::kDiskAnn:
      return "diskann";
    case backend_provider::kHnswlib:
      return "hnswlib";
  }
  return "unknown";
}

const char *diskann_build_mode_to_string(diskann_build_mode build_mode) {
  switch (build_mode) {
    case diskann_build_mode::kAuto:
      return "auto";
    case diskann_build_mode::kSerial:
      return "serial";
    case diskann_build_mode::kOffline:
      return "offline";
  }
  return "unknown";
}

const char *index_consistency_mode_to_string(
    index_consistency_mode consistency_mode) {
  switch (consistency_mode) {
    case index_consistency_mode::kTransactional:
      return "transactional";
    case index_consistency_mode::kStandalone:
      return "standalone";
  }
  return "unknown";
}

bool remove_backend_artifacts(const std::string &index_name, backend_mode mode,
                            backend_provider provider) {
  if (index_name.empty()) return true;

  if (mode == backend_mode::kExternal && provider == backend_provider::kFaiss) {
    return remove_external_sidecar_artifacts(
        index_name, external_sidecar_profile::kFaiss, true);
  }
  if (mode == backend_mode::kExternal &&
      provider == backend_provider::kDiskAnn) {
    return remove_diskann_external_artifacts(index_name, true);
  }
  return true;
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
void set_faiss_external_snapshot_root_for_testing(const std::string &root_path) {
  std::lock_guard<std::mutex> guard(
      detail::g_faiss_external_snapshot_root_mutex);
  detail::g_faiss_external_snapshot_root_override = root_path;
}

void reset_faiss_external_snapshot_root_for_testing() {
  std::lock_guard<std::mutex> guard(
      detail::g_faiss_external_snapshot_root_mutex);
  detail::g_faiss_external_snapshot_root_override.clear();
}

bool parse_uint64_for_testing(const std::string &text, uint64_t *value) {
  return parse_uint64(text, value);
}

bool load_external_manifest_generation_for_testing(const std::string &path,
                                              const char *expected_header,
                                              uint64_t *generation,
                                              bool *exists) {
  return load_external_manifest_generation_from_file(path, expected_header,
                                                     generation, exists);
}

bool save_external_manifest_generation_for_testing(const std::string &path,
                                              const char *header,
                                              uint64_t generation) {
  return save_external_manifest_generation_to_file(path, header, generation);
}

bool quarantine_file_if_exists_for_testing(const std::string &path) {
  return quarantine_file_if_exists(path);
}

bool remove_generated_snapshots_with_prefix_for_testing(const std::string &directory,
                                                  const std::string &prefix) {
  return remove_generated_snapshots_with_prefix(directory, prefix);
}

bool remove_dir_if_empty_for_testing(const std::string &path) {
  return remove_dir_if_empty(path);
}

bool ensure_parent_directory_for_testing(const std::string &path) {
  return ensure_parent_directory(path);
}

bool ends_with_for_testing(const std::string &text, const std::string &suffix) {
  return ends_with(text, suffix);
}

bool decode_hex_bytes_for_testing(const std::string &input, std::string *output) {
  return decode_hex_bytes(input, output);
}

std::string diskann_store_directory_for_testing(const std::string &index_name) {
  return diskann_store_directory(index_name);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index
