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
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "my_dbug.h"
#include "sql/vector/vector_diskann_garnet_abi.h"
#include "sql/vector/vector_index_backend_common.h"
#include "sql/vector/vector_index_backend_internal.h"
#include "sql/vector/vector_index_limits.h"

namespace {

using vector_index::detail::encode_hex_bytes;
using vector_index::detail::ensure_parent_directory;
using vector_index::detail::compute_distance;
using vector_index::detail::external_snapshot_directory;
using vector_index::detail::remove_dir_if_empty;
using vector_index::detail::remove_diskann_external_artifacts;
using vector_index::detail::remove_if_exists;

constexpr uint64_t kDiskAnnTermBitmask = 0x7ULL;
constexpr uint32_t kDiskAnnNoQuant = 1;
constexpr uint32_t kDiskAnnBuildComplexity = 64;
constexpr uint32_t kDiskAnnMaxDegree = 32;
constexpr const char *kDiskAnnOfflineBuildSymbol =
    "mysql_vector_diskann_offline_build";
constexpr const char *kDiskAnnOfflineBuildFromManifestSymbol =
    "mysql_vector_diskann_offline_build_from_manifest";
constexpr const char *kDiskAnnOfflineBuildFromNativePqSymbol =
    "mysql_vector_diskann_offline_build_from_native_pq";
constexpr const char *kDiskAnnOfflineLoadSymbol =
    "mysql_vector_diskann_offline_load";
constexpr const char *kDiskAnnOfflineSearchSymbol =
    "mysql_vector_diskann_offline_search";
constexpr const char *kDiskAnnOfflineSearchBatchSymbol =
    "mysql_vector_diskann_offline_search_batch";
constexpr const char *kDiskAnnOfflineCardSymbol =
    "mysql_vector_diskann_offline_card";
constexpr const char *kDiskAnnOfflineDropSymbol =
    "mysql_vector_diskann_offline_drop";

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
std::string &diskann_adapter_path_for_testing() {
  static std::string path;
  return path;
}

#ifndef MYSQL_VECTOR_DISKANN_OFFLINE_STATIC_LINKED
std::string &diskann_offline_adapter_path_for_testing() {
  static std::string path;
  return path;
}
#endif
#endif

bool diskann_sidecar_supports_doc_id(uint64_t doc_id) {
#ifdef HAVE_FAISS
  // The rebuildable FAISS sidecar stores identifiers in signed idx_t values.
  return doc_id <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
#else
  (void)doc_id;
  return true;
#endif
}

bool diskann_sidecar_supports_entries(
    const std::unordered_map<uint64_t, vector_index::vector_data> &entries) {
  return std::all_of(entries.begin(), entries.end(), [](const auto &entry) {
    return diskann_sidecar_supports_doc_id(entry.first);
  });
}

bool diskann_entries_match_dimension(
    const std::unordered_map<uint64_t, vector_index::vector_data> &entries,
    size_t dimension) {
  return std::all_of(entries.begin(), entries.end(),
                     [dimension](const auto &entry) {
                       return entry.second.size() == dimension;
                     });
}

using vector_index::diskann_garnet_abi::diskann_card_fn;
using vector_index::diskann_garnet_abi::diskann_backfill_quant_vectors_fn;
using vector_index::diskann_garnet_abi::diskann_build_quant_table_fn;
using vector_index::diskann_garnet_abi::diskann_create_index_fn;
using vector_index::diskann_garnet_abi::diskann_delete_callback;
using vector_index::diskann_garnet_abi::diskann_drop_index_fn;
using vector_index::diskann_garnet_abi::diskann_filter_callback;
using vector_index::diskann_garnet_abi::diskann_insert_fn;
using vector_index::diskann_garnet_abi::diskann_log_callback;
using vector_index::diskann_garnet_abi::diskann_random_members_fn;
using vector_index::diskann_garnet_abi::diskann_read_callback;
using vector_index::diskann_garnet_abi::diskann_read_data_callback;
using vector_index::diskann_garnet_abi::diskann_read_modify_write_callback;
using vector_index::diskann_garnet_abi::diskann_remove_fn;
using vector_index::diskann_garnet_abi::diskann_rmw_data_callback;
using vector_index::diskann_garnet_abi::diskann_search_vector_fn;
using vector_index::diskann_garnet_abi::diskann_search_neighbors_fn;
using vector_index::diskann_garnet_abi::diskann_write_callback;
using diskann_offline_build_fn = bool (*)(const char *, const uint8_t *, size_t,
                                          size_t, const uint8_t *, size_t,
                                          size_t, size_t, int32_t, uint32_t,
                                          uint32_t, uint32_t, double,
                                          uint32_t, uint32_t, uint32_t,
                                          uint32_t, uint32_t, uint32_t);
using diskann_offline_build_from_manifest_fn = bool (*)(const char *,
                                                        const char *, uint32_t,
                                                        int32_t, uint32_t,
                                                        uint32_t, uint32_t,
                                                        double, uint32_t,
                                                        uint32_t, uint32_t,
                                                        uint32_t, uint32_t,
                                                        uint32_t);
using diskann_offline_build_from_native_pq_fn = bool (*)(
    const char *, const char *, const char *, const char *, const char *,
    const char *, uint32_t, int32_t, uint32_t, uint32_t, uint32_t, double,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
using diskann_offline_load_fn = const void *(*)(const char *, int32_t, uint32_t,
                                                uint32_t, uint32_t, uint32_t);
using diskann_offline_search_fn = int32_t (*)(const void *, const float *,
                                              size_t, uint32_t, uint32_t,
                                              uint32_t, uint64_t *, float *);
using diskann_offline_search_batch_fn =
    int32_t (*)(const void *, const float *, size_t, size_t, uint32_t,
                uint32_t, uint32_t, uint32_t, uint64_t *, float *, uint32_t *);
using diskann_offline_card_fn = uint64_t (*)(const void *);
using diskann_offline_drop_fn = void (*)(const void *);

#ifdef MYSQL_VECTOR_DISKANN_OFFLINE_STATIC_LINKED
extern "C" {
bool mysql_vector_diskann_offline_build(
    const char *index_prefix, const uint8_t *vectors, size_t row_count,
    size_t dimension, const uint8_t *doc_ids, size_t doc_id_stride,
    size_t doc_id_count, size_t raw_vector_bytes, int32_t metric_type,
    uint32_t max_degree, uint32_t build_complexity, uint32_t num_threads,
    double build_memory_gb, uint32_t pq_code_budget_gb,
    uint32_t num_nodes_to_cache, uint32_t build_blas_threads,
    uint32_t disk_pq_dims, uint32_t accelerate_build,
    uint32_t shuffle_build);
bool mysql_vector_diskann_offline_build_from_manifest(
    const char *index_prefix, const char *manifest_path, uint32_t dimension,
    int32_t metric_type, uint32_t max_degree, uint32_t build_complexity,
    uint32_t num_threads, double build_memory_gb,
    uint32_t pq_code_budget_gb, uint32_t num_nodes_to_cache,
    uint32_t build_blas_threads, uint32_t disk_pq_dims,
    uint32_t accelerate_build, uint32_t shuffle_build);
bool mysql_vector_diskann_offline_build_from_native_pq(
    const char *index_prefix, const char *manifest_path,
    const char *pq_pivots_path, const char *pq_compressed_path,
    const char *disk_pq_pivots_path, const char *disk_pq_compressed_path,
    uint32_t dimension, int32_t metric_type, uint32_t max_degree,
    uint32_t build_complexity, uint32_t num_threads, double build_memory_gb,
    uint32_t pq_code_budget_gb, uint32_t num_nodes_to_cache,
    uint32_t build_blas_threads, uint32_t disk_pq_dims,
    uint32_t accelerate_build, uint32_t shuffle_build);
const void *mysql_vector_diskann_offline_load(
    const char *index_prefix, int32_t metric_type, uint32_t num_threads,
    uint32_t search_io_limit, uint32_t cache_nodes, uint32_t use_bfs_cache);
int32_t mysql_vector_diskann_offline_search(
    const void *index_ptr, const float *query, size_t dimension, uint32_t top_k,
    uint32_t search_complexity, uint32_t num_threads, uint64_t *labels,
    float *distances);
int32_t mysql_vector_diskann_offline_search_batch(
    const void *index_ptr, const float *queries, size_t query_count,
    size_t dimension, uint32_t top_k, uint32_t search_complexity,
    uint32_t beamwidth, uint32_t num_threads, uint64_t *labels,
    float *distances, uint32_t *result_counts);
uint64_t mysql_vector_diskann_offline_card(const void *index_ptr);
void mysql_vector_diskann_offline_drop(const void *index_ptr);
}
#endif

struct diskann_api {
  void *handle{nullptr};
  diskann_create_index_fn create_index{nullptr};
  diskann_drop_index_fn drop_index{nullptr};
  diskann_insert_fn insert{nullptr};
  diskann_build_quant_table_fn build_quant_table{nullptr};
  diskann_backfill_quant_vectors_fn backfill_quant_vectors{nullptr};
  diskann_random_members_fn random_members{nullptr};
  diskann_search_neighbors_fn search_neighbors{nullptr};
  diskann_search_vector_fn search_vector{nullptr};
  diskann_remove_fn remove{nullptr};
  diskann_card_fn card{nullptr};

  bool load() {
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_api_load", return false;);
    if (handle != nullptr) return available();

    std::vector<std::string> candidates;
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
    if (!diskann_adapter_path_for_testing().empty()) {
      candidates.emplace_back(diskann_adapter_path_for_testing());
    }
#endif
#ifdef MYSQL_VECTOR_DISKANN_DEFAULT_LIB
    candidates.emplace_back(MYSQL_VECTOR_DISKANN_DEFAULT_LIB);
#endif
#ifdef MYSQL_VECTOR_DISKANN_DEFAULT_LIB_DIR
    {
      const std::filesystem::path lib_dir(MYSQL_VECTOR_DISKANN_DEFAULT_LIB_DIR);
      candidates.emplace_back(
          (lib_dir / "libdiskann_garnet.dylib").string());
      candidates.emplace_back((lib_dir / "libdiskann_garnet.so").string());
    }
#endif
    candidates.emplace_back("libdiskann_garnet.dylib");
    candidates.emplace_back("libdiskann_garnet.so");

    for (const std::string &candidate : candidates) {
      if (candidate.empty()) continue;
      handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (handle == nullptr) continue;
      create_index = reinterpret_cast<diskann_create_index_fn>(
          dlsym(handle, "create_index"));
      drop_index =
          reinterpret_cast<diskann_drop_index_fn>(dlsym(handle, "drop_index"));
      insert = reinterpret_cast<diskann_insert_fn>(dlsym(handle, "insert"));
      build_quant_table = reinterpret_cast<diskann_build_quant_table_fn>(
          dlsym(handle, "build_quant_table"));
      backfill_quant_vectors =
          reinterpret_cast<diskann_backfill_quant_vectors_fn>(
              dlsym(handle, "backfill_quant_vectors"));
      random_members = reinterpret_cast<diskann_random_members_fn>(
          dlsym(handle, "random_members"));
      search_neighbors = reinterpret_cast<diskann_search_neighbors_fn>(
          dlsym(handle, "search_neighbors"));
      search_vector = reinterpret_cast<diskann_search_vector_fn>(
          dlsym(handle, "search_vector"));
      remove = reinterpret_cast<diskann_remove_fn>(dlsym(handle, "remove"));
      card = reinterpret_cast<diskann_card_fn>(dlsym(handle, "card"));
      DBUG_EXECUTE_IF("vector_backend_fail_diskann_api_symbol", { card = nullptr; };);
      if (available()) return true;
      dlclose(handle);
      handle = nullptr;
      create_index = nullptr;
      drop_index = nullptr;
      insert = nullptr;
      build_quant_table = nullptr;
      backfill_quant_vectors = nullptr;
      random_members = nullptr;
      search_neighbors = nullptr;
      search_vector = nullptr;
      remove = nullptr;
      card = nullptr;
    }

    return false;
  }

  bool available() const {
    return handle != nullptr && create_index != nullptr &&
           drop_index != nullptr && insert != nullptr &&
           build_quant_table != nullptr && backfill_quant_vectors != nullptr &&
           random_members != nullptr && search_neighbors != nullptr &&
           search_vector != nullptr &&
           remove != nullptr && card != nullptr;
  }

  const void *create_index_handle(uint64_t ctx, uint32_t dimension,
                                  uint32_t reduce_dimension,
                                  uint32_t quant_type, int32_t metric_type,
                                  uint32_t build_complexity,
                                  uint32_t max_degree,
                                  diskann_read_callback read_callback,
                                  diskann_write_callback write_callback,
                                  diskann_delete_callback delete_callback,
                                  diskann_read_modify_write_callback
                                      read_modify_write_callback,
                                  diskann_filter_callback filter_callback,
                                  diskann_log_callback log_callback,
                                  bool *quantization_needed) const {
    return create_index(ctx, dimension, reduce_dimension, quant_type, metric_type,
                        build_complexity, max_degree, read_callback,
                        write_callback, delete_callback,
                        read_modify_write_callback, filter_callback,
                        log_callback,
                        quantization_needed);
  }
};

struct diskann_offline_api {
  void *handle{nullptr};
  diskann_offline_build_fn build{nullptr};
  diskann_offline_build_from_manifest_fn build_from_manifest{nullptr};
  diskann_offline_build_from_native_pq_fn build_from_native_pq{nullptr};
  diskann_offline_load_fn load_index{nullptr};
  diskann_offline_search_fn search{nullptr};
  diskann_offline_search_batch_fn search_batch{nullptr};
  diskann_offline_card_fn card{nullptr};
  diskann_offline_drop_fn drop_index{nullptr};

  bool load() {
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_offline_api_load",
                    return false;);
    if (handle != nullptr) return available();

#ifdef MYSQL_VECTOR_DISKANN_OFFLINE_STATIC_LINKED
    handle = this;
    build = mysql_vector_diskann_offline_build;
    build_from_manifest = mysql_vector_diskann_offline_build_from_manifest;
    build_from_native_pq = mysql_vector_diskann_offline_build_from_native_pq;
    load_index = mysql_vector_diskann_offline_load;
    search = mysql_vector_diskann_offline_search;
    search_batch = mysql_vector_diskann_offline_search_batch;
    card = mysql_vector_diskann_offline_card;
    drop_index = mysql_vector_diskann_offline_drop;
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_offline_api_symbol",
                    { card = nullptr; });
    return available();
#else
    std::vector<std::string> candidates;
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
    if (!diskann_offline_adapter_path_for_testing().empty()) {
      candidates.emplace_back(diskann_offline_adapter_path_for_testing());
    }
#endif
#ifdef MYSQL_VECTOR_DISKANN_OFFLINE_DEFAULT_LIB
    candidates.emplace_back(MYSQL_VECTOR_DISKANN_OFFLINE_DEFAULT_LIB);
#endif
#ifdef MYSQL_VECTOR_DISKANN_OFFLINE_DEFAULT_LIB_DIR
    {
      const std::filesystem::path lib_dir(
          MYSQL_VECTOR_DISKANN_OFFLINE_DEFAULT_LIB_DIR);
      candidates.emplace_back(
          (lib_dir / "libmysql_vector_diskann_offline_adapter.dylib").string());
      candidates.emplace_back(
          (lib_dir / "libmysql_vector_diskann_offline_adapter.so").string());
    }
#endif
    candidates.emplace_back("libmysql_vector_diskann_offline_adapter.dylib");
    candidates.emplace_back("libmysql_vector_diskann_offline_adapter.so");

    for (const std::string &candidate : candidates) {
      if (candidate.empty()) continue;
      handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (handle == nullptr) continue;
      build = reinterpret_cast<diskann_offline_build_fn>(
          dlsym(handle, kDiskAnnOfflineBuildSymbol));
      build_from_manifest =
          reinterpret_cast<diskann_offline_build_from_manifest_fn>(
              dlsym(handle, kDiskAnnOfflineBuildFromManifestSymbol));
      build_from_native_pq =
          reinterpret_cast<diskann_offline_build_from_native_pq_fn>(
              dlsym(handle, kDiskAnnOfflineBuildFromNativePqSymbol));
      load_index = reinterpret_cast<diskann_offline_load_fn>(
          dlsym(handle, kDiskAnnOfflineLoadSymbol));
      search = reinterpret_cast<diskann_offline_search_fn>(
          dlsym(handle, kDiskAnnOfflineSearchSymbol));
      search_batch = reinterpret_cast<diskann_offline_search_batch_fn>(
          dlsym(handle, kDiskAnnOfflineSearchBatchSymbol));
      card = reinterpret_cast<diskann_offline_card_fn>(
          dlsym(handle, kDiskAnnOfflineCardSymbol));
      drop_index = reinterpret_cast<diskann_offline_drop_fn>(
          dlsym(handle, kDiskAnnOfflineDropSymbol));
      DBUG_EXECUTE_IF("vector_backend_fail_diskann_offline_api_symbol",
                      { card = nullptr; });
      if (available()) return true;
      dlclose(handle);
      handle = nullptr;
      build = nullptr;
      build_from_manifest = nullptr;
      build_from_native_pq = nullptr;
      load_index = nullptr;
      search = nullptr;
      search_batch = nullptr;
      card = nullptr;
      drop_index = nullptr;
    }

    return false;
#endif
  }

  bool available() const {
    return handle != nullptr && build != nullptr &&
           build_from_manifest != nullptr && load_index != nullptr &&
           search != nullptr && card != nullptr && drop_index != nullptr;
  }

  bool manifest_build_available() const {
    DBUG_EXECUTE_IF("vector_backend_diskann_manifest_unavailable",
                    return false;);
    return available();
  }

  bool native_pq_build_available() const {
    return available() && build_from_native_pq != nullptr;
  }
};

diskann_api &get_diskann_api() {
  static diskann_api api;
  if (!api.available()) (void)api.load();
  return api;
}

diskann_offline_api &get_diskann_offline_api() {
  static diskann_offline_api api;
  if (!api.available()) (void)api.load();
  return api;
}

const char *diskann_term_name(uint64_t ctx) {
  switch (ctx & kDiskAnnTermBitmask) {
    case 0:
      return "vector";
    case 1:
      return "neighbors";
    case 2:
      return "quantized";
    case 3:
      return "attributes";
    case 4:
      return "metadata";
    case 5:
      return "intmap";
    case 6:
      return "extmap";
    default:
      return "unknown";
  }
}

int32_t diskann_metric_code(vector_index::metric_type metric) {
  switch (metric) {
    case vector_index::metric_type::kCosine:
      return 0;
    case vector_index::metric_type::kInnerProduct:
      return 1;
    case vector_index::metric_type::kEuclidean:
      return 2;
  }
  return 2;
}

std::string diskann_doc_id_bytes(uint64_t doc_id) {
  std::string bytes(sizeof(doc_id), '\0');
  std::memcpy(bytes.data(), &doc_id, sizeof(doc_id));
  return bytes;
}

bool parse_diskann_doc_id(const uint8_t *data, size_t length, uint64_t *doc_id) {
  if (doc_id == nullptr || data == nullptr || length != sizeof(uint64_t)) {
    return false;
  }
  std::memcpy(doc_id, data, sizeof(uint64_t));
  return true;
}

}  // namespace

namespace vector_index {

bool diskann_backend::search_entries_exact(
    const vector_data &query, size_t top_k,
    std::vector<search_result> *results) const {
  if (results == nullptr || query.size() != m_dimension) return false;
  results->clear();
  if (top_k == 0) return true;
  for (const auto &entry : m_entries) {
    double distance = 0.0;
    if (!compute_distance(m_metric, query, entry.second, &distance))
      return false;
    results->push_back(search_result{entry.first, distance});
  }
  const size_t count = std::min(top_k, results->size());
  std::partial_sort(results->begin(), results->begin() + count, results->end(),
                    [](const search_result &lhs, const search_result &rhs) {
                      if (lhs.distance != rhs.distance)
                        return lhs.distance < rhs.distance;
                      return lhs.doc_id < rhs.doc_id;
                    });
  results->resize(count);
  return true;
}

class diskann_native_state {
 public:
  diskann_native_state(size_t dimension, metric_type metric, const std::string &index_name,
                     uint32_t build_complexity, uint32_t max_degree)
      : m_dimension(dimension),
        m_metric(metric),
        m_index_name(index_name),
        m_store_directory(detail::diskann_store_directory(index_name)),
        m_api(&get_diskann_api()),
        m_build_complexity(build_complexity),
        m_max_degree(max_degree) {}

  ~diskann_native_state() {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    close_locked();
  }

  bool supported() const { return m_api != nullptr && m_api->available(); }
  bool active() const {
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    return m_index_handle != nullptr;
  }
  const std::string &store_directory() const { return m_store_directory; }

  bool reopen(bool reset_store) {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    return reopen_locked(reset_store);
  }

  bool rebuild_from_entries(const std::unordered_map<uint64_t, vector_data> &entries) {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    std::vector<std::pair<uint64_t, const vector_data *>> ordered_entries;
    ordered_entries.reserve(entries.size());
    for (const auto &entry : entries) {
      if (entry.second.size() != m_dimension) {
        return false;
      }
      ordered_entries.push_back({entry.first, &entry.second});
    }
    std::sort(ordered_entries.begin(), ordered_entries.end(),
              [](const auto &lhs, const auto &rhs) {
                return lhs.first < rhs.first;
              });

    begin_build_memory_store_locked();
    if (!reopen_locked(false)) {
      discard_build_memory_store_locked();
      return false;
    }

    bool rebuild_ok = true;
    for (const auto &entry : ordered_entries) {
      if (!insert_locked(entry.first, *entry.second)) {
        rebuild_ok = false;
        break;
      }
    }
    if (rebuild_ok) rebuild_ok = flush_build_memory_store_locked();
    if (!rebuild_ok) close_locked();
    discard_build_memory_store_locked();
    return rebuild_ok;
  }

  bool insert(uint64_t doc_id, const vector_data &vector) {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    return insert_locked(doc_id, vector);
  }

  bool remove(uint64_t doc_id) {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    if (!ensure_open_locked(false)) return false;
    const std::string doc_id_bytes = diskann_doc_id_bytes(doc_id);
    return m_api->remove(callback_context(), m_index_handle,
                         reinterpret_cast<const uint8_t *>(doc_id_bytes.data()),
                         doc_id_bytes.size());
  }

  bool search(const vector_data &query, size_t top_k,
              std::vector<search_result> *results) const {
    if (results == nullptr) return false;
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    results->clear();
    if (m_index_handle == nullptr) return false;
    std::string ids_buffer(top_k * (sizeof(uint32_t) + sizeof(uint64_t)), '\0');
    std::vector<float> distances(top_k, 0.0F);
    const uint32_t effective_search_complexity =
        std::max(m_search_complexity, static_cast<uint32_t>(top_k));
    const int32_t count = m_api->search_vector(
        callback_context(), m_index_handle,
        reinterpret_cast<const uint8_t *>(query.data()), query.size(), 0.0F,
        effective_search_complexity, nullptr, 0, 0,
        reinterpret_cast<uint8_t *>(ids_buffer.data()), ids_buffer.size(),
        distances.data(), distances.size(),
        vector_index::k_default_diskann_search_beamwidth, nullptr);
    if (count < 0) return false;

    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(ids_buffer.data());
    size_t remaining = ids_buffer.size();
    for (int32_t i = 0; i < count; ++i) {
      if (remaining < sizeof(uint32_t)) return false;
      uint32_t length = 0;
      std::memcpy(&length, ptr, sizeof(length));
      ptr += sizeof(length);
      remaining -= sizeof(length);
      if (remaining < length) return false;
      uint64_t doc_id = 0;
      if (!parse_diskann_doc_id(ptr, length, &doc_id)) return false;
      results->push_back({doc_id, distances[static_cast<size_t>(i)]});
      ptr += length;
      remaining -= length;
    }
    return true;
  }

  size_t entry_count() const {
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    if (m_index_handle == nullptr) return 0;
    return static_cast<size_t>(m_api->card(callback_context(), m_index_handle));
  }

  bool set_search_complexity(uint32_t search_complexity) {
    if (search_complexity == 0) return false;
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    m_search_complexity = search_complexity;
    return true;
  }

  uint32_t search_complexity() const {
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    return m_search_complexity;
  }

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  std::string term_directory_for_testing(uint64_t ctx) const {
    return term_directory(ctx);
  }

  std::string key_path_for_testing(uint64_t ctx, const uint8_t *key,
                                size_t length) const {
    return key_path(ctx, key, length);
  }

  bool load_value_for_testing(uint64_t ctx, const uint8_t *key, size_t key_length,
                           std::string *value) const {
    return load_value(ctx, key, key_length, value);
  }

  bool save_value_for_testing(uint64_t ctx, const uint8_t *key, size_t key_length,
                           const uint8_t *value, size_t value_length) {
    return save_value(ctx, key, key_length, value, value_length);
  }

  bool delete_value_for_testing(uint64_t ctx, const uint8_t *key,
                             size_t key_length) {
    return delete_value(ctx, key, key_length);
  }

  bool build_memory_store_round_trip_for_testing(
      uint64_t ctx, const uint8_t *key, size_t key_length,
      const std::string &value, std::string *loaded_before_flush,
      bool *file_exists_before_flush, std::string *loaded_after_flush) {
    if (loaded_before_flush == nullptr || file_exists_before_flush == nullptr ||
        loaded_after_flush == nullptr) {
      return false;
    }
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    const std::string path = key_path(ctx, key, key_length);
    if (path.empty()) return false;
    begin_build_memory_store_locked();
    const bool saved =
        save_value(ctx, key, key_length,
                   reinterpret_cast<const uint8_t *>(value.data()),
                   value.size());
    const bool loaded =
        load_value(ctx, key, key_length, loaded_before_flush);
    *file_exists_before_flush = std::filesystem::exists(path);
    const bool flushed = flush_build_memory_store_locked();
    discard_build_memory_store_locked();
    loaded_after_flush->clear();
    return saved && loaded && flushed &&
           load_value(ctx, key, key_length, loaded_after_flush);
  }

  bool resident_store_round_trip_for_testing(
      uint64_t ctx, const uint8_t *key, size_t key_length,
      const std::string &value, std::string *loaded_after_removing_file) {
    if (loaded_after_removing_file == nullptr) return false;
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    const std::string path = key_path(ctx, key, key_length);
    if (path.empty()) return false;
    begin_build_memory_store_locked();
    const bool saved =
        save_value(ctx, key, key_length,
                   reinterpret_cast<const uint8_t *>(value.data()),
                   value.size());
    const bool flushed = flush_build_memory_store_locked();
    discard_build_memory_store_locked();
    loaded_after_removing_file->clear();
    return saved && flushed && remove_if_exists(path) &&
           load_value(ctx, key, key_length, loaded_after_removing_file);
  }

  static bool parse_prefixed_keys_for_testing(
      const uint8_t *key_data, size_t key_length, uint32_t key_count,
      const std::function<bool(uint32_t, const uint8_t *, size_t)> &visitor) {
    return parse_prefixed_keys(key_data, key_length, key_count, visitor);
  }

  struct test_rmw_payload {
    const char *data;
    size_t length;
  };

  static void copy_rmw_payload_for_testing(void *user_data, uint8_t *data,
                                           size_t length) {
    if (user_data == nullptr || data == nullptr || length == 0) return;
    const auto *payload = static_cast<const test_rmw_payload *>(user_data);
    const size_t copy_length = std::min(length, payload->length);
    if (copy_length != 0) std::memcpy(data, payload->data, copy_length);
  }

  bool read_modify_write_round_trip_for_testing(
      uint64_t ctx, const uint8_t *key, size_t key_length,
      const std::string &initial_value, const std::string &patch_value,
      size_t write_length, std::string *persistent_value,
      std::string *build_memory_value, std::string *resident_value) {
    if (persistent_value == nullptr || build_memory_value == nullptr ||
        resident_value == nullptr) {
      return false;
    }

    test_rmw_payload payload{patch_value.data(), patch_value.size()};
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);

    bool ok = save_value(ctx, key, key_length,
                         reinterpret_cast<const uint8_t *>(initial_value.data()),
                         initial_value.size());
    ok = read_modify_write_value(ctx, key, key_length, write_length,
                                 &diskann_native_state::copy_rmw_payload_for_testing,
                                 &payload) &&
         ok;
    persistent_value->clear();
    ok = load_value(ctx, key, key_length, persistent_value) && ok;

    begin_build_memory_store_locked();
    ok = read_modify_write_value(ctx, key, key_length, 0,
                                 &diskann_native_state::copy_rmw_payload_for_testing,
                                 &payload) &&
         ok;
    build_memory_value->clear();
    ok = load_value(ctx, key, key_length, build_memory_value) && ok;
    std::string after_build_delete;
    ok = delete_value(ctx, key, key_length) && ok;
    ok = !load_value(ctx, key, key_length, &after_build_delete) && ok;
    discard_build_memory_store_locked();

    begin_build_memory_store_locked();
    ok = save_value(ctx, key, key_length,
                    reinterpret_cast<const uint8_t *>(initial_value.data()),
                    initial_value.size()) &&
         ok;
    ok = flush_build_memory_store_locked() && ok;
    discard_build_memory_store_locked();
    (void)remove_if_exists(key_path(ctx, key, key_length));
    ok = read_modify_write_value(ctx, key, key_length, write_length,
                                 &diskann_native_state::copy_rmw_payload_for_testing,
                                 &payload) &&
         ok;
    resident_value->clear();
    ok = load_value(ctx, key, key_length, resident_value) && ok;
    std::string after_resident_delete;
    ok = delete_value(ctx, key, key_length) && ok;
    ok = !load_value(ctx, key, key_length, &after_resident_delete) && ok;
    return ok;
  }
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

 private:
  bool reopen_locked(bool reset_store) {
    if (!supported()) return false;

    close_locked();
    if (m_store_directory.empty()) return false;

    std::error_code ec;
    if (!m_build_memory_store_active.load(std::memory_order_acquire)) {
      if (reset_store) {
        std::filesystem::remove_all(m_store_directory, ec);
        clear_store(&m_resident_store);
        m_resident_store_active.store(false, std::memory_order_release);
      }
      ec.clear();
      std::filesystem::create_directories(m_store_directory, ec);
      if (ec) return false;
    }

    bool quantization_needed = false;
    const void *index = m_api->create_index_handle(
        callback_context(), static_cast<uint32_t>(m_dimension), 0,
        kDiskAnnNoQuant, diskann_metric_code(m_metric), m_build_complexity,
        m_max_degree, &diskann_native_state::read_callback,
        &diskann_native_state::write_callback,
        &diskann_native_state::delete_callback,
        &diskann_native_state::read_modify_write_callback,
        &diskann_native_state::filter_callback,
        &diskann_native_state::log_callback, &quantization_needed);
    if (index == nullptr || quantization_needed) {
      if (index != nullptr) m_api->drop_index(callback_context(), index);
      return false;
    }
    m_index_handle = index;
    return true;
  }

  bool insert_locked(uint64_t doc_id, const vector_data &vector) {
    if (!ensure_open_locked(false)) return false;
    const std::string doc_id_bytes = diskann_doc_id_bytes(doc_id);
    const uint8_t status = m_api->insert(
        callback_context(), m_index_handle,
        reinterpret_cast<const uint8_t *>(doc_id_bytes.data()),
        doc_id_bytes.size(),
        reinterpret_cast<const uint8_t *>(vector.data()), vector.size(), nullptr,
        0);
    // Garnet uses two nonzero statuses for successful insert transitions.
    return status != 0;
  }

  static diskann_native_state *from_context(uint64_t ctx) {
    return reinterpret_cast<diskann_native_state *>(ctx & ~kDiskAnnTermBitmask);
  }

  uint64_t callback_context() const {
    return reinterpret_cast<uint64_t>(const_cast<diskann_native_state *>(this));
  }

  bool ensure_open_locked(bool reset_store) const {
    if (m_index_handle != nullptr) return true;
    return const_cast<diskann_native_state *>(this)->reopen_locked(reset_store);
  }

  void close_locked() {
    if (m_index_handle != nullptr && supported()) {
      m_api->drop_index(callback_context(), m_index_handle);
      m_index_handle = nullptr;
    }
  }

  std::string term_directory(uint64_t ctx) const {
    if (m_store_directory.empty()) return "";
    return m_store_directory + "/" + diskann_term_name(ctx);
  }

  std::string store_key(uint64_t ctx, const uint8_t *key, size_t length) const {
    if (key == nullptr || length == 0) return "";
    std::string relative_path = diskann_term_name(ctx);
    relative_path.push_back('/');
    relative_path.append(
        encode_hex_bytes(reinterpret_cast<const char *>(key), length));
    return relative_path;
  }

  std::string key_path_from_store_key(const std::string &root,
                                   const std::string &relative_path) const {
    if (root.empty() || relative_path.empty()) return "";
    return (std::filesystem::path(root) / relative_path).string();
  }

  std::string key_path(uint64_t ctx, const uint8_t *key, size_t length) const {
    return key_path_from_store_key(m_store_directory, store_key(ctx, key, length));
  }

  static constexpr size_t kDiskAnnStoreShardCount = 64;

  struct native_store_shard {
    mutable std::mutex mutex;
    std::unordered_map<std::string, std::string> entries;
  };

  using native_store_shards =
      std::array<native_store_shard, kDiskAnnStoreShardCount>;

  static size_t store_shard_index(const std::string &relative_path) {
    return std::hash<std::string>{}(relative_path) % kDiskAnnStoreShardCount;
  }

  static native_store_shard &store_shard(native_store_shards *store,
                                         const std::string &relative_path) {
    return (*store)[store_shard_index(relative_path)];
  }

  static const native_store_shard &store_shard(
      const native_store_shards &store, const std::string &relative_path) {
    return store[store_shard_index(relative_path)];
  }

  static void clear_store(native_store_shards *store) {
    for (native_store_shard &shard : *store) {
      std::lock_guard<std::mutex> guard(shard.mutex);
      shard.entries.clear();
    }
  }

  static bool load_from_store(const native_store_shards &store,
                              const std::string &relative_path,
                              std::string *value) {
    if (value == nullptr) return false;
    const native_store_shard &shard = store_shard(store, relative_path);
    std::lock_guard<std::mutex> guard(shard.mutex);
    const auto iter = shard.entries.find(relative_path);
    if (iter == shard.entries.end()) return false;
    *value = iter->second;
    return true;
  }

  static bool save_to_store(native_store_shards *store,
                            const std::string &relative_path,
                            std::string value) {
    native_store_shard &shard = store_shard(store, relative_path);
    std::lock_guard<std::mutex> guard(shard.mutex);
    shard.entries[relative_path] = std::move(value);
    return true;
  }

  static void delete_from_store(native_store_shards *store,
                                const std::string &relative_path) {
    native_store_shard &shard = store_shard(store, relative_path);
    std::lock_guard<std::mutex> guard(shard.mutex);
    shard.entries.erase(relative_path);
  }

  void move_build_store_to_resident_locked() {
    for (size_t i = 0; i < kDiskAnnStoreShardCount; ++i) {
      std::scoped_lock guard(m_build_memory_store[i].mutex,
                             m_resident_store[i].mutex);
      m_resident_store[i].entries =
          std::move(m_build_memory_store[i].entries);
      m_build_memory_store[i].entries.clear();
    }
    m_resident_store_active.store(true, std::memory_order_release);
    m_build_memory_store_active.store(false, std::memory_order_release);
  }

  void begin_build_memory_store_locked() {
    clear_store(&m_build_memory_store);
    m_build_memory_store_active.store(true, std::memory_order_release);
  }

  void discard_build_memory_store_locked() {
    clear_store(&m_build_memory_store);
    m_build_memory_store_active.store(false, std::memory_order_release);
  }

  bool save_persistent_value(const std::string &path,
                            const std::string &value) const {
    if (path.empty()) return false;
    if (!ensure_parent_directory(path)) return false;
    const std::string temp_path = path + ".tmp";
    std::ofstream file(temp_path,
                       std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file.good()) return false;
    file.write(value.data(), value.size());
    file.close();
    if (!file) {
      std::remove(temp_path.c_str());
      return false;
    }
    if (std::rename(temp_path.c_str(), path.c_str()) != 0) {
      std::remove(temp_path.c_str());
      return false;
    }
    return true;
  }

  bool swap_persistent_store_directory(const std::string &staging_directory) const {
    std::error_code ec;
    const std::filesystem::path store(m_store_directory);
    const std::filesystem::path staging(staging_directory);
    const std::string suffix =
        ".bulk-old-" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
    const std::filesystem::path backup(m_store_directory + suffix);

    std::filesystem::remove_all(backup, ec);
    if (ec) return false;

    bool has_existing_store = std::filesystem::exists(store, ec);
    if (ec) return false;
    if (has_existing_store) {
      std::filesystem::rename(store, backup, ec);
      if (ec) return false;
    }

    std::filesystem::rename(staging, store, ec);
    if (ec) {
      std::error_code restore_ec;
      if (has_existing_store) std::filesystem::rename(backup, store, restore_ec);
      std::filesystem::remove_all(staging, restore_ec);
      return false;
    }

    if (has_existing_store) {
      std::filesystem::remove_all(backup, ec);
      if (ec) return false;
    }
    return true;
  }

  bool flush_build_memory_store_locked() {
    if (!m_build_memory_store_active.load(std::memory_order_acquire) ||
        m_store_directory.empty()) {
      return false;
    }

    const std::string suffix =
        ".bulk-tmp-" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
    const std::filesystem::path staging(m_store_directory + suffix);
    std::error_code ec;
    std::filesystem::remove_all(staging, ec);
    if (ec) return false;
    std::filesystem::create_directories(staging, ec);
    if (ec) return false;

    for (const native_store_shard &shard : m_build_memory_store) {
      std::lock_guard<std::mutex> guard(shard.mutex);
      for (const auto &entry : shard.entries) {
        const std::string path =
            key_path_from_store_key(staging.string(), entry.first);
        if (!save_persistent_value(path, entry.second)) {
          std::filesystem::remove_all(staging, ec);
          return false;
        }
      }
    }
    if (!swap_persistent_store_directory(staging.string())) return false;
    move_build_store_to_resident_locked();
    return true;
  }

  bool load_value(uint64_t ctx, const uint8_t *key, size_t key_length,
                 std::string *value) const {
    if (value == nullptr) return false;
    const std::string relative_path = store_key(ctx, key, key_length);
    if (relative_path.empty()) return false;
    if (m_build_memory_store_active.load(std::memory_order_acquire)) {
      return load_from_store(m_build_memory_store, relative_path, value);
    }
    if (m_resident_store_active.load(std::memory_order_acquire) &&
        load_from_store(m_resident_store, relative_path, value)) {
      return true;
    }
    return load_persistent_value(relative_path, value);
  }

  bool load_persistent_value(const std::string &relative_path,
                             std::string *value) const {
    if (value == nullptr) return false;
    const std::string path = key_path_from_store_key(m_store_directory, relative_path);
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.good()) return false;
    *value = std::string(std::istreambuf_iterator<char>(file),
                         std::istreambuf_iterator<char>());
    return !file.bad();
  }

  bool save_value(uint64_t ctx, const uint8_t *key, size_t key_length,
                 const uint8_t *value, size_t value_length) {
    if (value == nullptr && value_length != 0) return false;
    const std::string relative_path = store_key(ctx, key, key_length);
    if (relative_path.empty()) return false;
    std::string payload;
    if (value_length != 0) {
      payload.assign(reinterpret_cast<const char *>(value), value_length);
    }
    if (m_build_memory_store_active.load(std::memory_order_acquire)) {
      return save_to_store(&m_build_memory_store, relative_path,
                           std::move(payload));
    }
    if (!save_persistent_value(
            key_path_from_store_key(m_store_directory, relative_path), payload)) {
      return false;
    }
    if (m_resident_store_active.load(std::memory_order_acquire)) {
      return save_to_store(&m_resident_store, relative_path,
                           std::move(payload));
    }
    return true;
  }

  bool delete_value(uint64_t ctx, const uint8_t *key, size_t key_length) {
    const std::string relative_path = store_key(ctx, key, key_length);
    if (relative_path.empty()) return false;
    if (m_build_memory_store_active.load(std::memory_order_acquire)) {
      delete_from_store(&m_build_memory_store, relative_path);
      return true;
    }
    const std::string path = key_path_from_store_key(m_store_directory, relative_path);
    if (!remove_if_exists(path)) return false;
    if (m_resident_store_active.load(std::memory_order_acquire)) {
      delete_from_store(&m_resident_store, relative_path);
    }
    return remove_dir_if_empty(std::filesystem::path(path).parent_path().string());
  }

  static bool apply_read_modify_write(const std::string &value,
                                      size_t write_length,
                                      diskann_rmw_data_callback callback,
                                      void *user_data, std::string *updated) {
    if (callback == nullptr || updated == nullptr ||
        write_length >
            std::numeric_limits<size_t>::max() - (sizeof(uint64_t) - 1) ||
        write_length > updated->max_size()) {
      return false;
    }
    const size_t aligned_words =
        (write_length + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    if (aligned_words > std::vector<uint64_t>().max_size()) return false;

    std::vector<uint64_t> aligned(aligned_words);
    if (!aligned.empty()) {
      std::memset(aligned.data(), 0, aligned.size() * sizeof(uint64_t));
      if (!value.empty()) {
        std::memcpy(aligned.data(), value.data(),
                    std::min(value.size(), write_length));
      }
    }
    callback(user_data, reinterpret_cast<uint8_t *>(aligned.data()), write_length);
    if (write_length == 0) {
      updated->clear();
    } else {
      updated->assign(reinterpret_cast<const char *>(aligned.data()),
                      write_length);
    }
    return true;
  }

  static bool read_modify_write_store_value(
      native_store_shards *store, const std::string &relative_path,
      size_t write_length, diskann_rmw_data_callback callback, void *user_data) {
    native_store_shard &shard = store_shard(store, relative_path);
    std::lock_guard<std::mutex> guard(shard.mutex);
    std::string value;
    const auto iter = shard.entries.find(relative_path);
    if (iter != shard.entries.end()) value = iter->second;
    std::string updated;
    if (!apply_read_modify_write(value, write_length, callback, user_data,
                                 &updated)) {
      return false;
    }
    shard.entries[relative_path] = std::move(updated);
    return true;
  }

  bool read_modify_write_resident_value(
      const std::string &relative_path, size_t write_length,
      diskann_rmw_data_callback callback, void *user_data) {
    native_store_shard &shard = store_shard(&m_resident_store, relative_path);
    std::lock_guard<std::mutex> guard(shard.mutex);
    std::string value;
    const auto iter = shard.entries.find(relative_path);
    if (iter != shard.entries.end()) {
      value = iter->second;
    } else {
      (void)load_persistent_value(relative_path, &value);
    }
    std::string updated;
    if (!apply_read_modify_write(value, write_length, callback, user_data,
                                 &updated)) {
      return false;
    }
    if (!save_persistent_value(
            key_path_from_store_key(m_store_directory, relative_path), updated)) {
      return false;
    }
    shard.entries[relative_path] = std::move(updated);
    return true;
  }

  bool read_modify_write_value(uint64_t ctx, const uint8_t *key,
                               size_t key_length, size_t write_length,
                               diskann_rmw_data_callback callback,
                               void *user_data) {
    const std::string relative_path = store_key(ctx, key, key_length);
    if (relative_path.empty()) return false;
    if (m_build_memory_store_active.load(std::memory_order_acquire)) {
      return read_modify_write_store_value(&m_build_memory_store, relative_path,
                                           write_length, callback, user_data);
    }
    if (m_resident_store_active.load(std::memory_order_acquire)) {
      return read_modify_write_resident_value(relative_path, write_length,
                                              callback, user_data);
    }
    std::string value;
    (void)load_persistent_value(relative_path, &value);
    std::string updated;
    if (!apply_read_modify_write(value, write_length, callback, user_data,
                                 &updated)) {
      return false;
    }
    return save_persistent_value(
        key_path_from_store_key(m_store_directory, relative_path), updated);
  }

  static bool parse_prefixed_keys(const uint8_t *key_data, size_t key_length,
                                uint32_t key_count,
                                const std::function<bool(uint32_t, const uint8_t *,
                                                         size_t)> &visitor) {
    if (key_count == 0) return true;
    if (key_data == nullptr) return false;
    const uint8_t *ptr = key_data;
    size_t remaining = key_length;
    for (uint32_t i = 0; i < key_count; ++i) {
      if (remaining < sizeof(uint32_t)) return false;
      uint32_t length = 0;
      std::memcpy(&length, ptr, sizeof(length));
      ptr += sizeof(length);
      remaining -= sizeof(length);
      if (remaining < length) return false;
      if (!visitor(i, ptr, length)) return false;
      ptr += length;
      remaining -= length;
    }
    return true;
  }

  static void read_callback(uint64_t ctx, uint32_t key_count,
                            uint32_t value_length_hint,
                            const uint8_t *key_data, size_t key_length,
                            diskann_read_data_callback callback,
                            void *user_data) {
    diskann_native_state *state = from_context(ctx);
    if (state == nullptr || callback == nullptr) return;
    (void)value_length_hint;
    (void)parse_prefixed_keys(
        key_data, key_length, key_count,
        [state, ctx, callback, user_data](uint32_t index, const uint8_t *key,
                                          size_t length) {
          std::string value;
          if (!state->load_value(ctx, key, length, &value)) return true;
          std::vector<uint64_t> aligned((value.size() + sizeof(uint64_t) - 1) /
                                        sizeof(uint64_t));
          if (!value.empty()) {
            std::memcpy(aligned.data(), value.data(), value.size());
          }
          callback(index, user_data,
                   reinterpret_cast<const uint8_t *>(aligned.data()),
                   value.size());
          return true;
        });
  }

  static bool write_callback(uint64_t ctx, const uint8_t *key, size_t key_length,
                            const uint8_t *value, size_t value_length) {
    diskann_native_state *state = from_context(ctx);
    if (state == nullptr || value == nullptr) return false;
    return state->save_value(ctx, key, key_length, value, value_length);
  }

  static bool delete_callback(uint64_t ctx, const uint8_t *key, size_t key_length) {
    diskann_native_state *state = from_context(ctx);
    if (state == nullptr) return false;
    return state->delete_value(ctx, key, key_length);
  }

  static bool read_modify_write_callback(uint64_t ctx, const uint8_t *key,
                                      size_t key_length, size_t write_length,
                                      diskann_rmw_data_callback callback,
                                      void *user_data) {
    diskann_native_state *state = from_context(ctx);
    if (state == nullptr || callback == nullptr) return false;
    return state->read_modify_write_value(ctx, key, key_length, write_length,
                                          callback, user_data);
  }

  static bool filter_callback(uint64_t, const uint8_t *, size_t) {
    // MySQL applies visibility filtering after DiskANN returns candidates.
    return true;
  }

  static void log_callback(uint64_t, const uint8_t *, size_t) {
    // MySQL exposes stable vector diagnostics through its own status surfaces.
  }

  size_t m_dimension{0};
  metric_type m_metric{metric_type::kEuclidean};
  std::string m_index_name;
  std::string m_store_directory;
  const diskann_api *m_api{nullptr};
  const void *m_index_handle{nullptr};
  uint32_t m_build_complexity{kDiskAnnBuildComplexity};
  uint32_t m_max_degree{kDiskAnnMaxDegree};
  uint32_t m_search_complexity{kDiskAnnBuildComplexity};
  mutable std::shared_mutex m_lifecycle_mutex;
  std::atomic_bool m_build_memory_store_active{false};
  native_store_shards m_build_memory_store;
  std::atomic_bool m_resident_store_active{false};
  native_store_shards m_resident_store;
};

diskann_backend::diskann_backend(size_t dimension, metric_type metric, backend_mode mode,
                               const std::string &index_name)
    : m_dimension(dimension),
      m_metric(metric),
      m_mode(mode),
      m_index_name(index_name),
      m_external_adapter(dimension, metric, mode, index_name,
                         external_sidecar_profile::kDiskAnn) {}

diskann_backend::~diskann_backend() = default;

bool diskann_backend::upsert(uint64_t doc_id, const vector_data &vector) {
  if (m_mode != backend_mode::kExternal) return false;
  if (vector.size() != m_dimension) return false;
  if (m_external_adapter_active && !diskann_sidecar_supports_doc_id(doc_id)) {
    m_external_adapter_active = false;
  }
  if (m_external_adapter_active &&
      !m_external_adapter.upsert(doc_id, vector))
    return false;
  m_entries[doc_id] = vector;
  bool native_insert_ok = true;
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    native_insert_ok = m_native_state->insert(doc_id, vector);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_insert",
                    native_insert_ok = false;);
  }
  if (m_native_runtime_enabled && m_native_state != nullptr && !native_insert_ok) {
    m_native_runtime_enabled = false;
    m_native_state.reset();
  }
  return true;
}

bool diskann_backend::erase(uint64_t doc_id) {
  if (m_mode != backend_mode::kExternal) return false;
  if (m_external_adapter_active && !m_external_adapter.erase(doc_id))
    return false;
  m_entries.erase(doc_id);
  bool native_remove_ok = true;
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    native_remove_ok = m_native_state->remove(doc_id);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_remove",
                    native_remove_ok = false;);
  }
  if (m_native_runtime_enabled && m_native_state != nullptr && !native_remove_ok) {
    m_native_runtime_enabled = false;
    m_native_state.reset();
  }
  return true;
}

bool diskann_backend::search(const vector_data &query, size_t top_k,
                            std::vector<search_result> *results) const {
  if (results == nullptr || query.size() != m_dimension) return false;
  results->clear();
  if (top_k == 0) return true;
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    bool native_search_ok = m_native_state->search(query, top_k, results);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_search",
                    native_search_ok = false; results->clear(););
    if (native_search_ok) {
      if (top_k == 0 || !results->empty() || m_entries.empty()) {
        return true;
      }

      // Native DiskANN reopen/rebuild can transiently yield an empty hit set
      // even though committed entries exist. The in-memory committed-entry
      // snapshot keeps SQL-visible search stable without requiring a sidecar.
    } else {
      results->clear();
    }
  }
  return search_entries_exact(query, top_k, results) ||
         m_external_adapter.search(query, top_k, results);
}

bool diskann_backend::load_committed_entries(
    const std::unordered_map<uint64_t, vector_data> &entries) {
  if (m_mode != backend_mode::kExternal) return false;
  const bool sidecar_compatible = diskann_sidecar_supports_entries(entries);
  if (sidecar_compatible) {
    if (!m_external_adapter.load_committed_entries(entries)) return false;
  } else if (!diskann_entries_match_dimension(entries, m_dimension)) {
    return false;
  }
  m_entries = entries;
  m_external_adapter_active = sidecar_compatible;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

bool diskann_backend::rebuild_from_committed_entries(
    const std::unordered_map<uint64_t, vector_data> &entries) {
  if (m_mode != backend_mode::kExternal) return false;
  if (m_diskann_build_mode == diskann_build_mode::kOffline) return false;
  auto rebuilt = std::make_unique<diskann_native_state>(
      m_dimension, m_metric, m_index_name, m_diskann_build_complexity,
      m_diskann_max_degree);
  bool rebuild_ok = rebuilt->rebuild_from_entries(entries);
  DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_rebuild",
                  rebuild_ok = false;);
  if (!rebuild_ok) {
    if (rebuilt->supported()) return false;
    const bool sidecar_compatible = diskann_sidecar_supports_entries(entries);
    if (sidecar_compatible) {
      if (!m_external_adapter.load_committed_entries(entries)) return false;
    } else if (!diskann_entries_match_dimension(entries, m_dimension)) {
      return false;
    }
    (void)rebuilt->set_search_complexity(m_diskann_search_complexity);
    m_entries = entries;
    m_external_adapter_active = sidecar_compatible;
    m_native_state = std::move(rebuilt);
    m_native_runtime_enabled = false;
    return true;
  }
  bool search_complexity_ok =
      rebuilt->set_search_complexity(m_diskann_search_complexity);
  DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_set_search_complexity",
                  search_complexity_ok = false;);
  if (!search_complexity_ok) return false;
  if (!remove_diskann_external_artifacts(m_index_name, false))
    return false;
  m_entries = entries;
  m_external_adapter_active = false;
  m_native_state = std::move(rebuilt);
  m_native_runtime_enabled = true;
  return true;
}

bool diskann_backend::recover() {
  if (m_mode != backend_mode::kExternal) return false;
  if (!m_external_adapter.recover()) return false;
  m_entries = m_external_adapter.external_snapshot_entries();
  m_external_adapter_active = true;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

bool diskann_backend::recover_committed_entries(
    const std::unordered_map<uint64_t, vector_data> &entries) {
  if (m_mode != backend_mode::kExternal) return false;
  /*
    The sidecar is a rebuildable serving artifact. Recovery must reconcile it
    with the committed truth rows handed in by the registry instead of trusting
    sidecar contents as authoritative.
  */
  (void)recover();
  return rebuild_from_committed_entries(entries);
}

bool diskann_backend::last_recover_used_fallback() const {
  return m_external_adapter.last_recover_used_fallback();
}

size_t diskann_backend::entry_count() const {
  return m_entries.size();
}

bool diskann_backend::external_manifest_present() const {
  return m_external_adapter_active &&
         m_external_adapter.external_manifest_present();
}

uint64_t diskann_backend::external_manifest_generation() const {
  return m_external_adapter_active
             ? m_external_adapter.external_manifest_generation()
             : 0;
}

std::string diskann_backend::backend_variant() const {
  return "diskann_garnet";
}

bool diskann_backend::set_diskann_build_params(uint32_t diskann_max_degree,
                                               uint32_t diskann_build_complexity,
                                               uint32_t diskann_build_threads) {
  if (m_mode != backend_mode::kExternal || diskann_max_degree == 0 ||
      diskann_build_complexity == 0 ||
      diskann_build_threads > vector_index::k_max_build_threads) {
    return false;
  }
  m_diskann_max_degree = diskann_max_degree;
  m_diskann_build_complexity = diskann_build_complexity;
  m_diskann_build_threads = diskann_build_threads;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

bool diskann_backend::set_diskann_build_threads(
    uint32_t diskann_build_threads) {
  if (m_mode != backend_mode::kExternal ||
      diskann_build_threads > vector_index::k_max_build_threads) {
    return false;
  }
  m_diskann_build_threads = diskann_build_threads;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

uint32_t diskann_backend::diskann_max_degree() const {
  return m_mode == backend_mode::kExternal ? m_diskann_max_degree : 0;
}

uint32_t diskann_backend::diskann_build_complexity() const {
  return m_mode == backend_mode::kExternal ? m_diskann_build_complexity : 0;
}

uint32_t diskann_backend::diskann_build_threads() const {
  return m_mode == backend_mode::kExternal ? m_diskann_build_threads : 0;
}

bool diskann_backend::set_diskann_build_blas_threads(
    uint32_t diskann_build_blas_threads) {
  if (m_mode != backend_mode::kExternal ||
      diskann_build_blas_threads > vector_index::k_max_build_threads) {
    return false;
  }
  m_diskann_build_blas_threads = diskann_build_blas_threads;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

uint32_t diskann_backend::diskann_build_blas_threads() const {
  return m_mode == backend_mode::kExternal ? m_diskann_build_blas_threads : 0;
}

bool diskann_backend::set_diskann_build_mode(
    diskann_build_mode diskann_build_mode_value) {
  if (m_mode != backend_mode::kExternal) return false;
  m_diskann_build_mode = diskann_build_mode_value;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

diskann_build_mode diskann_backend::diskann_build_mode_value() const {
  return m_mode == backend_mode::kExternal ? m_diskann_build_mode
                                           : diskann_build_mode::kAuto;
}

bool diskann_backend::set_diskann_search_complexity(
    uint32_t diskann_search_complexity) {
  if (m_mode != backend_mode::kExternal || diskann_search_complexity == 0) {
    return false;
  }
  bool native_search_complexity_ok = true;
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    native_search_complexity_ok =
        m_native_state->set_search_complexity(diskann_search_complexity);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_live_search_complexity",
                    native_search_complexity_ok = false;);
  }
  if (m_native_runtime_enabled && m_native_state != nullptr &&
      !native_search_complexity_ok) {
    return false;
  }
  m_diskann_search_complexity = diskann_search_complexity;
  return true;
}

uint32_t diskann_backend::diskann_search_complexity() const {
  return m_mode == backend_mode::kExternal ? m_diskann_search_complexity : 0;
}

bool diskann_backend::set_diskann_search_beamwidth(
    uint32_t diskann_search_beamwidth) {
  if (m_mode != backend_mode::kExternal ||
      !valid_diskann_search_beamwidth(diskann_search_beamwidth)) {
    return false;
  }
  m_diskann_search_beamwidth = diskann_search_beamwidth;
  return true;
}

uint32_t diskann_backend::diskann_search_beamwidth() const {
  return m_mode == backend_mode::kExternal ? m_diskann_search_beamwidth : 0;
}

bool diskann_backend::set_diskann_pq_code_budget_size(
    uint64_t diskann_pq_code_budget_size) {
  if (m_mode != backend_mode::kExternal) return false;
  m_diskann_pq_code_budget_size = diskann_pq_code_budget_size;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

uint64_t diskann_backend::diskann_pq_code_budget_size() const {
  return m_mode == backend_mode::kExternal ? m_diskann_pq_code_budget_size : 0;
}

bool diskann_backend::set_diskann_disk_pq_dims(uint32_t diskann_disk_pq_dims) {
  if (m_mode != backend_mode::kExternal ||
      diskann_disk_pq_dims > vector_index::k_max_diskann_disk_pq_dims) {
    return false;
  }
  m_diskann_disk_pq_dims = diskann_disk_pq_dims;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

uint32_t diskann_backend::diskann_disk_pq_dims() const {
  return m_mode == backend_mode::kExternal ? m_diskann_disk_pq_dims : 0;
}

bool diskann_backend::set_diskann_accelerate_build(
    bool diskann_accelerate_build) {
  if (m_mode != backend_mode::kExternal) return false;
  m_diskann_accelerate_build = diskann_accelerate_build;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

bool diskann_backend::diskann_accelerate_build() const {
  return m_mode == backend_mode::kExternal && m_diskann_accelerate_build;
}

bool diskann_backend::set_diskann_shuffle_build(bool diskann_shuffle_build) {
  if (m_mode != backend_mode::kExternal) return false;
  m_diskann_shuffle_build = diskann_shuffle_build;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

bool diskann_backend::diskann_shuffle_build() const {
  return m_mode == backend_mode::kExternal && m_diskann_shuffle_build;
}

bool diskann_backend::set_diskann_use_bfs_cache(bool diskann_use_bfs_cache) {
  if (m_mode != backend_mode::kExternal) return false;
  m_diskann_use_bfs_cache = diskann_use_bfs_cache;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

bool diskann_backend::diskann_use_bfs_cache() const {
  return m_mode == backend_mode::kExternal && m_diskann_use_bfs_cache;
}

uint32_t diskann_backend::diskann_offline_search_threads() const {
  return m_mode == backend_mode::kExternal ? m_diskann_offline_search_threads : 0;
}

uint32_t diskann_backend::diskann_search_io_limit() const {
  return m_mode == backend_mode::kExternal ? m_diskann_search_io_limit : 0;
}

uint32_t diskann_backend::diskann_cache_nodes() const {
  return m_mode == backend_mode::kExternal ? m_diskann_cache_nodes : 0;
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
std::string diskann_term_directory_for_testing(const std::string &index_name,
                                           uint64_t ctx) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree);
  return state.term_directory_for_testing(ctx);
}

std::string diskann_key_path_for_testing(const std::string &index_name, uint64_t ctx,
                                     const std::string &key_bytes) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree);
  if (key_bytes.empty()) return state.key_path_for_testing(ctx, nullptr, 0);
  return state.key_path_for_testing(
      ctx, reinterpret_cast<const uint8_t *>(key_bytes.data()),
      key_bytes.size());
}

bool diskann_load_value_for_testing(const std::string &index_name, uint64_t ctx,
                                const std::string &key_bytes,
                                std::string *value) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree);
  if (key_bytes.empty()) return state.load_value_for_testing(ctx, nullptr, 0, value);
  return state.load_value_for_testing(
      ctx, reinterpret_cast<const uint8_t *>(key_bytes.data()),
      key_bytes.size(), value);
}

bool diskann_save_value_for_testing(const std::string &index_name, uint64_t ctx,
                                const std::string &key_bytes,
                                const std::string &value) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree);
  const uint8_t *key_ptr = key_bytes.empty()
                               ? nullptr
                               : reinterpret_cast<const uint8_t *>(key_bytes.data());
  return state.save_value_for_testing(ctx, key_ptr, key_bytes.size(),
                                   reinterpret_cast<const uint8_t *>(value.data()),
                                   value.size());
}

bool diskann_delete_value_for_testing(const std::string &index_name, uint64_t ctx,
                                  const std::string &key_bytes) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree);
  const uint8_t *key_ptr = key_bytes.empty()
                               ? nullptr
                               : reinterpret_cast<const uint8_t *>(key_bytes.data());
  return state.delete_value_for_testing(ctx, key_ptr, key_bytes.size());
}

bool diskann_build_memory_store_round_trip_for_testing(
    const std::string &index_name, uint64_t ctx, const std::string &key_bytes,
    const std::string &value, std::string *loaded_before_flush,
    bool *file_exists_before_flush, std::string *loaded_after_flush) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree);
  const uint8_t *key_ptr = key_bytes.empty()
                               ? nullptr
                               : reinterpret_cast<const uint8_t *>(key_bytes.data());
  return state.build_memory_store_round_trip_for_testing(
      ctx, key_ptr, key_bytes.size(), value, loaded_before_flush,
      file_exists_before_flush, loaded_after_flush);
}

bool diskann_resident_store_round_trip_for_testing(
    const std::string &index_name, uint64_t ctx, const std::string &key_bytes,
    const std::string &value, std::string *loaded_after_removing_file) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree);
  const uint8_t *key_ptr = key_bytes.empty()
                               ? nullptr
                               : reinterpret_cast<const uint8_t *>(key_bytes.data());
  return state.resident_store_round_trip_for_testing(
      ctx, key_ptr, key_bytes.size(), value, loaded_after_removing_file);
}

bool diskann_read_modify_write_round_trip_for_testing(
    const std::string &index_name, uint64_t ctx, const std::string &key_bytes,
    const std::string &initial_value, const std::string &patch_value,
    size_t write_length, std::string *persistent_value,
    std::string *build_memory_value, std::string *resident_value) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree);
  const uint8_t *key_ptr = key_bytes.empty()
                               ? nullptr
                               : reinterpret_cast<const uint8_t *>(key_bytes.data());
  return state.read_modify_write_round_trip_for_testing(
      ctx, key_ptr, key_bytes.size(), initial_value, patch_value, write_length,
      persistent_value, build_memory_value, resident_value);
}

bool diskann_parse_prefixed_keys_for_testing(const std::string &payload,
                                             uint32_t key_count,
                                             std::vector<std::string> *keys,
                                             int fail_index,
                                             bool null_key_data) {
  if (keys == nullptr) return false;
  keys->clear();
  const uint8_t *key_data =
      null_key_data ? nullptr
                    : reinterpret_cast<const uint8_t *>(payload.data());
  return diskann_native_state::parse_prefixed_keys_for_testing(
      key_data, payload.size(), key_count,
      [keys, fail_index](uint32_t index, const uint8_t *key, size_t length) {
        if (static_cast<int>(index) == fail_index) return false;
        keys->emplace_back(reinterpret_cast<const char *>(key), length);
        return true;
      });
}

bool diskann_api_load_for_testing() {
  diskann_api api;
  return api.load();
}

void diskann_set_adapter_path_for_testing(const std::string &path) {
  diskann_adapter_path_for_testing() = path;
}

void diskann_reset_adapter_path_for_testing() {
  diskann_adapter_path_for_testing().clear();
}

bool diskann_api_available_for_testing(bool has_handle, bool has_create_index,
                                       bool has_drop_index, bool has_insert,
                                       bool has_search_vector, bool has_remove,
                                       bool has_card,
                                       bool has_build_quant_table,
                                       bool has_backfill_quant_vectors,
                                       bool has_random_members,
                                       bool has_search_neighbors) {
  diskann_api api;
  api.handle = has_handle ? reinterpret_cast<void *>(1) : nullptr;
  api.create_index = has_create_index
                         ? reinterpret_cast<diskann_create_index_fn>(1)
                         : nullptr;
  api.drop_index =
      has_drop_index ? reinterpret_cast<diskann_drop_index_fn>(1) : nullptr;
  api.insert = has_insert ? reinterpret_cast<diskann_insert_fn>(1) : nullptr;
  api.build_quant_table =
      has_build_quant_table
          ? reinterpret_cast<diskann_build_quant_table_fn>(1)
          : nullptr;
  api.backfill_quant_vectors =
      has_backfill_quant_vectors
          ? reinterpret_cast<diskann_backfill_quant_vectors_fn>(1)
          : nullptr;
  api.random_members =
      has_random_members ? reinterpret_cast<diskann_random_members_fn>(1)
                         : nullptr;
  api.search_neighbors =
      has_search_neighbors ? reinterpret_cast<diskann_search_neighbors_fn>(1)
                           : nullptr;
  api.search_vector = has_search_vector
                          ? reinterpret_cast<diskann_search_vector_fn>(1)
                          : nullptr;
  api.remove = has_remove ? reinterpret_cast<diskann_remove_fn>(1) : nullptr;
  api.card = has_card ? reinterpret_cast<diskann_card_fn>(1) : nullptr;
  return api.available();
}

bool diskann_offline_api_load_for_testing() {
  diskann_offline_api api;
  return api.load();
}

#ifndef MYSQL_VECTOR_DISKANN_OFFLINE_STATIC_LINKED
void diskann_set_offline_adapter_path_for_testing(const std::string &path) {
  diskann_offline_adapter_path_for_testing() = path;
}

void diskann_reset_offline_adapter_path_for_testing() {
  diskann_offline_adapter_path_for_testing().clear();
}
#endif

std::vector<std::string> diskann_offline_api_symbol_names_for_testing() {
  return {kDiskAnnOfflineBuildSymbol,
          kDiskAnnOfflineBuildFromManifestSymbol,
          kDiskAnnOfflineBuildFromNativePqSymbol,
          kDiskAnnOfflineLoadSymbol,
          kDiskAnnOfflineSearchSymbol,
          kDiskAnnOfflineSearchBatchSymbol,
          kDiskAnnOfflineCardSymbol,
          kDiskAnnOfflineDropSymbol};
}

bool diskann_offline_api_available_for_testing(bool has_handle, bool has_build,
                                               bool has_build_from_manifest,
                                               bool has_load_index,
                                               bool has_search,
                                               bool has_search_batch,
                                               bool has_card,
                                               bool has_drop_index) {
  diskann_offline_api api;
  api.handle = has_handle ? reinterpret_cast<void *>(1) : nullptr;
  api.build = has_build ? reinterpret_cast<diskann_offline_build_fn>(1)
                        : nullptr;
  api.build_from_manifest =
      has_build_from_manifest
          ? reinterpret_cast<diskann_offline_build_from_manifest_fn>(1)
          : nullptr;
  api.load_index =
      has_load_index ? reinterpret_cast<diskann_offline_load_fn>(1) : nullptr;
  api.search =
      has_search ? reinterpret_cast<diskann_offline_search_fn>(1) : nullptr;
  api.search_batch = has_search_batch
                         ? reinterpret_cast<diskann_offline_search_batch_fn>(1)
                         : nullptr;
  api.card = has_card ? reinterpret_cast<diskann_offline_card_fn>(1) : nullptr;
  api.drop_index = has_drop_index
                       ? reinterpret_cast<diskann_offline_drop_fn>(1)
                       : nullptr;
  return api.available();
}

bool diskann_offline_api_manifest_build_load_for_testing() {
  diskann_offline_api api;
  return api.load() && api.manifest_build_available();
}

bool diskann_offline_api_manifest_build_available_for_testing(
    bool has_handle, bool has_build, bool has_build_from_manifest,
    bool has_load_index, bool has_search, bool has_search_batch,
    bool has_card, bool has_drop_index) {
  diskann_offline_api api;
  api.handle = has_handle ? reinterpret_cast<void *>(1) : nullptr;
  api.build = has_build ? reinterpret_cast<diskann_offline_build_fn>(1)
                        : nullptr;
  api.build_from_manifest =
      has_build_from_manifest
          ? reinterpret_cast<diskann_offline_build_from_manifest_fn>(1)
          : nullptr;
  api.load_index =
      has_load_index ? reinterpret_cast<diskann_offline_load_fn>(1) : nullptr;
  api.search =
      has_search ? reinterpret_cast<diskann_offline_search_fn>(1) : nullptr;
  api.search_batch = has_search_batch
                         ? reinterpret_cast<diskann_offline_search_batch_fn>(1)
                         : nullptr;
  api.card = has_card ? reinterpret_cast<diskann_offline_card_fn>(1) : nullptr;
  api.drop_index = has_drop_index
                       ? reinterpret_cast<diskann_offline_drop_fn>(1)
                       : nullptr;
  return api.manifest_build_available();
}

int32_t diskann_metric_code_for_testing(metric_type metric) {
  return diskann_metric_code(metric);
}

bool parse_diskann_doc_id_for_testing(const uint8_t *data, size_t length,
                                 uint64_t *doc_id) {
  return parse_diskann_doc_id(data, length, doc_id);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index
