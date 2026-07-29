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

#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "my_dbug.h"
#include "sql/vector/vector_diskann_garnet_abi.h"
#include "sql/vector/vector_diskann_graph_cache_bridge.h"
#include "sql/vector/vector_diskann_pq_runtime.h"
#include "sql/vector/vector_diskann_scheduler.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_backend_common.h"
#include "sql/vector/vector_index_backend_internal.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_runtime_thread_pool.h"
#include "sql/vector/vector_load_file.h"

#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
#include "mysql_vector_diskann_runtime.h"
#endif

namespace {

using vector_index::detail::encode_hex_bytes;
using vector_index::detail::ensure_parent_directory;
using vector_index::detail::compute_distance;
using vector_index::detail::external_snapshot_directory;
using vector_index::detail::remove_dir_if_empty;
using vector_index::detail::remove_diskann_external_artifacts;
using vector_index::detail::remove_if_exists;
using vector_index::saturated_add_size;
using vector_index::saturated_mul_size;

constexpr uint64_t kDiskAnnTermBitmask = 0x7ULL;
constexpr uint32_t kDiskAnnNoQuant = 1;
constexpr uint32_t kDiskAnnBuildComplexity =
    vector_index::k_default_diskann_build_complexity;
constexpr uint32_t kDiskAnnMaxDegree =
    vector_index::k_default_diskann_max_degree;
#if defined(MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED) || \
    defined(EXTRA_CODE_FOR_UNIT_TESTING)
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
#endif
constexpr size_t kDiskAnnMinimumOfflineBuildRows = 4;
constexpr const char *kDiskAnnRawManifestHeader =
    "mysql-vector-diskann-raw-manifest-v1";
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
constexpr const char *kDiskAnnRuntimeOfficialCppMain = "official_cpp_main";
constexpr const char *kDiskAnnRuntimeSerial = "serial";
constexpr const char *kDiskAnnRuntimeSerialFallback = "serial_fallback";
constexpr const char *kDiskAnnRuntimeVendored = "vendored_runtime";
constexpr const char *kDiskAnnFallbackOfflineMinRows = "offline_min_rows";

struct diskann_batch_search_scratch {
  std::vector<float> query_data;
  std::vector<uint64_t> doc_ids;
  std::vector<float> distances;
  std::vector<uint32_t> result_counts;
};

enum class diskann_runtime_variant {
  kNone,
  kSerial,
  kOffline,
  kVendoredOffline,
  kOfflineSegmented,
  kVendoredSegmented,
};

uint32_t diskann_effective_offline_max_degree(size_t row_count,
                                              uint32_t max_degree) {
  /*
    DiskANN C++ offline build expects a non-trivial graph. Debug builds assert
    while pruning graphs with only one or two points. Keep tiny inputs out of
    offline build, and clamp the requested graph degree for small safe inputs
    so segmented tail chunks do not request an impossible neighborhood.
  */
  if (max_degree == 0) return 0;
  if (row_count == 0) return max_degree;
  if (row_count < kDiskAnnMinimumOfflineBuildRows) return 0;
  if (row_count > std::numeric_limits<uint32_t>::max()) return max_degree;
  return std::min(max_degree, static_cast<uint32_t>(row_count - 1));
}

bool diskann_offline_build_too_small(size_t row_count, uint32_t max_degree) {
  return row_count != 0 &&
         diskann_effective_offline_max_degree(row_count, max_degree) == 0;
}

const char *diskann_runtime_variant_name(diskann_runtime_variant variant) {
  switch (variant) {
    case diskann_runtime_variant::kSerial:
      return "diskann_serial";
    case diskann_runtime_variant::kOffline:
      return "diskann_offline";
    case diskann_runtime_variant::kVendoredOffline:
      return "diskann_vendored_offline";
    case diskann_runtime_variant::kOfflineSegmented:
      return "diskann_offline_segmented";
    case diskann_runtime_variant::kVendoredSegmented:
      return "diskann_vendored_segmented";
    case diskann_runtime_variant::kNone:
      break;
  }
  return "diskann_unloaded";
}

uint32_t diskann_offline_load_use_bfs_cache(uint32_t cache_nodes,
                                            bool force_bfs_cache) {
  return (force_bfs_cache || cache_nodes != 0) ? 1U : 0U;
}

bool diskann_force_offline_adapter() {
  bool force_offline_adapter = false;
  DBUG_EXECUTE_IF("vector_backend_diskann_force_offline_adapter",
                  force_offline_adapter = true;);
  return force_offline_adapter;
}

uint32_t diskann_cache_nodes_for_build(size_t row_count, size_t dimension,
                                       uint32_t explicit_cache_nodes,
                                       uint64_t search_cache_size,
                                       double search_cache_ratio,
                                       uint32_t max_degree = kDiskAnnMaxDegree,
                                       uint32_t disk_pq_dims = 0) {
  if (row_count == 0 || dimension == 0) return 0;
  if (explicit_cache_nodes != 0) {
    return static_cast<uint32_t>(
        std::min<uint64_t>(explicit_cache_nodes, row_count));
  }

  const size_t row_bytes = saturated_mul_size(dimension, sizeof(float));
  if (row_bytes == 0 || row_bytes == std::numeric_limits<size_t>::max()) {
    return 0;
  }

  if (dimension > std::numeric_limits<uint32_t>::max()) return 0;
  const size_t payload_size = saturated_mul_size(row_count, row_bytes);
  if (payload_size == std::numeric_limits<size_t>::max()) return 0;

  vector_index::diskann_segment_budget_input input;
  input.dimension = static_cast<uint32_t>(
      std::min<size_t>(dimension, std::numeric_limits<uint32_t>::max()));
  input.row_count = row_count;
  input.payload_size = payload_size;
  input.disk_pq_dims = disk_pq_dims;
  input.max_degree = max_degree;
  input.search_cache_size = search_cache_size;
  input.search_cache_ratio = search_cache_ratio;
  vector_index::diskann_segment_budget budget;
  if (!vector_index::make_diskann_segment_budget(input, &budget)) return 0;
  uint64_t nodes = budget.cache_nodes;
  if (nodes == 0) return 0;
  nodes = std::min<uint64_t>(nodes, row_count);
  nodes = std::min<uint64_t>(nodes, std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(nodes);
}

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
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
using diskann_vendored_build_from_manifest_fn =
    bool (*)(const mysql_vector_diskann_build_config *, char *, size_t);
using diskann_vendored_build_from_native_pq_fn =
    bool (*)(const mysql_vector_diskann_build_config *, const char *,
             const char *, const char *, const char *, char *, size_t);
using diskann_vendored_load_fn =
    mysql_vector_diskann_runtime_handle *(*)(const mysql_vector_diskann_build_config *,
                                             const mysql_vector_diskann_search_config *,
                                             char *, size_t);
using diskann_vendored_search_fn =
    int32_t (*)(const mysql_vector_diskann_runtime_handle *, const float *,
                size_t, const mysql_vector_diskann_search_config *, uint64_t *,
                float *);
using diskann_vendored_search_batch_fn =
    int32_t (*)(const mysql_vector_diskann_runtime_handle *, const float *,
                size_t, size_t, const mysql_vector_diskann_search_config *,
                uint64_t *, float *, uint32_t *);
using diskann_vendored_card_fn =
    uint64_t (*)(const mysql_vector_diskann_runtime_handle *);
using diskann_vendored_drop_fn =
    void (*)(const mysql_vector_diskann_runtime_handle *);
#endif

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
                    { card = nullptr; };);
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
                      { card = nullptr; };);
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
    return handle != nullptr && build != nullptr && build_from_manifest != nullptr &&
           load_index != nullptr && search != nullptr && card != nullptr &&
           drop_index != nullptr;
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

struct diskann_vendored_runtime_api {
  bool linked{false};
  uint32_t abi_version{0};
  uint64_t capabilities{0};
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
  diskann_vendored_build_from_manifest_fn build_from_manifest{nullptr};
  diskann_vendored_build_from_native_pq_fn build_from_native_pq{nullptr};
  diskann_vendored_load_fn load_index{nullptr};
  diskann_vendored_search_fn search{nullptr};
  diskann_vendored_search_batch_fn search_batch{nullptr};
  diskann_vendored_card_fn card{nullptr};
  diskann_vendored_drop_fn drop_index{nullptr};
#endif

  bool load() {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
    if (linked) return available();
    abi_version = mysql_vector_diskann_runtime_abi_version();
    capabilities = mysql_vector_diskann_runtime_capabilities();
    build_from_manifest = mysql_vector_diskann_build_from_manifest;
    build_from_native_pq = mysql_vector_diskann_build_from_native_pq;
    load_index = mysql_vector_diskann_runtime_load;
    search = mysql_vector_diskann_runtime_search;
    search_batch = mysql_vector_diskann_runtime_search_batch;
    card = mysql_vector_diskann_runtime_card;
    drop_index = mysql_vector_diskann_runtime_drop;
    linked = true;
    return available();
#else
    return false;
#endif
  }

  bool available() const {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
    return linked && abi_version == MYSQL_VECTOR_DISKANN_RUNTIME_ABI_VERSION &&
           ((capabilities & MYSQL_VECTOR_DISKANN_RUNTIME_CAPABILITY_CONFIG) != 0);
#else
    return false;
#endif
  }

  bool structured_build_available() const {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
    return available() &&
           ((capabilities &
             MYSQL_VECTOR_DISKANN_RUNTIME_CAPABILITY_STRUCTURED_BUILD) != 0) &&
           build_from_manifest != nullptr;
#else
    return false;
#endif
  }

  bool native_pq_bridge_available() const {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
    return available() &&
           ((capabilities &
             MYSQL_VECTOR_DISKANN_RUNTIME_CAPABILITY_NATIVE_PQ_BRIDGE) != 0) &&
           build_from_native_pq != nullptr;
#else
    return false;
#endif
  }

  bool runtime_handle_available() const {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
    return structured_build_available() && load_index != nullptr &&
           search != nullptr && card != nullptr && drop_index != nullptr;
#else
    return false;
#endif
  }

#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
  bool build_manifest(const mysql_vector_diskann_build_config &config,
                      std::string *error) const {
    if (!structured_build_available()) return false;
    char error_buffer[256]{};
    const bool ok =
        build_from_manifest(&config, error_buffer, sizeof(error_buffer));
    if (error != nullptr) *error = error_buffer;
    return ok;
  }

  bool build_native_pq(const mysql_vector_diskann_build_config &config,
                       const char *pq_pivots_path,
                       const char *pq_compressed_path,
                       const char *disk_pq_pivots_path,
                       const char *disk_pq_compressed_path,
                       char *error_buffer,
                       size_t error_buffer_size) const {
    if (!native_pq_bridge_available()) return false;
    return build_from_native_pq(
        &config, pq_pivots_path, pq_compressed_path, disk_pq_pivots_path,
        disk_pq_compressed_path, error_buffer, error_buffer_size);
  }

  const void *load(const mysql_vector_diskann_build_config &build_config,
                   const mysql_vector_diskann_search_config &search_config,
                   std::string *error) const {
    if (!runtime_handle_available()) return nullptr;
    char error_buffer[256]{};
    const void *handle =
        load_index(&build_config, &search_config, error_buffer,
                   sizeof(error_buffer));
    if (error != nullptr) *error = error_buffer;
    return handle;
  }

  int32_t search_one(const void *handle, const float *query, size_t dimension,
                     const mysql_vector_diskann_search_config &search_config,
                     uint64_t *doc_ids, float *distances) const {
    if (!runtime_handle_available()) return -1;
    return search(
        static_cast<const mysql_vector_diskann_runtime_handle *>(handle),
        query, dimension, &search_config, doc_ids, distances);
  }

  int32_t search_many(
      const void *handle, const float *queries, size_t query_count,
      size_t dimension, const mysql_vector_diskann_search_config &search_config,
      uint64_t *doc_ids, float *distances, uint32_t *result_counts) const {
    if (!batch_search_available()) return -1;
    return search_batch(
        static_cast<const mysql_vector_diskann_runtime_handle *>(handle),
        queries, query_count, dimension, &search_config, doc_ids, distances,
        result_counts);
  }

  void drop_handle(const void *handle) const {
    if (drop_index != nullptr) {
      drop_index(static_cast<const mysql_vector_diskann_runtime_handle *>(
          handle));
    }
  }
#endif

  bool batch_search_available() const {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
    return available() &&
           ((capabilities & MYSQL_VECTOR_DISKANN_RUNTIME_CAPABILITY_BATCH_SEARCH) !=
            0) &&
           search_batch != nullptr;
#else
    return false;
#endif
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

diskann_vendored_runtime_api &get_diskann_vendored_runtime_api() {
  static diskann_vendored_runtime_api api;
  if (!api.available()) (void)api.load();
  return api;
}

#if defined(MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED) || \
    defined(EXTRA_CODE_FOR_UNIT_TESTING)
double diskann_build_memory_size_gb(uint64_t build_memory_size) {
  return static_cast<double>(build_memory_size) / kBytesPerGiB;
}
#endif

uint64_t available_diskann_build_memory_size() {
#if defined(__linux__) && defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
  const long pages = sysconf(_SC_AVPHYS_PAGES);
  const long page_size = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page_size <= 0) return 0;

  const uint64_t page_count = static_cast<uint64_t>(pages);
  const uint64_t bytes_per_page = static_cast<uint64_t>(page_size);
  if (page_count >
      std::numeric_limits<uint64_t>::max() / bytes_per_page) {
    return std::numeric_limits<uint64_t>::max();
  }
  return page_count * bytes_per_page;
#elif defined(__APPLE__)
  uint64_t memory_size = 0;
  size_t length = sizeof(memory_size);
  if (sysctlbyname("hw.memsize", &memory_size, &length, nullptr, 0) != 0) {
    return 0;
  }
  return memory_size;
#else
  return 0;
#endif
}

template <typename T>
bool write_binary_value(std::ofstream &file, T value) {
  file.write(reinterpret_cast<const char *>(&value), sizeof(value));
  return file.good();
}

template <typename T>
bool read_binary_value(std::ifstream &file, T *value) {
  if (value == nullptr) return false;
  file.read(reinterpret_cast<char *>(value), sizeof(*value));
  return file.good();
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

namespace {

constexpr char kDiskAnnUnsupportedBuildFlags[] =
    "DiskANN offline adapter does not support accelerated or shuffled build";

}  // namespace

class diskann_native_state {
 public:
  diskann_native_state(
      size_t dimension, metric_type metric, const std::string &index_name,
      uint32_t build_complexity, uint32_t max_degree,
      uint32_t build_threads, uint32_t build_blas_threads,
      uint32_t offline_search_threads =
          vector_index::k_default_diskann_offline_search_threads,
      uint32_t search_io_limit = 0, uint32_t cache_nodes = 0,
      uint32_t search_beamwidth =
          vector_index::k_default_diskann_search_beamwidth,
      uint64_t pq_code_budget_size = 0, uint64_t search_cache_size = 0,
      double pq_code_budget_ratio =
          vector_index::k_default_diskann_pq_code_budget_ratio,
      double search_cache_ratio = 0.0, uint32_t disk_pq_dims = 0,
      bool accelerate_build = false, bool shuffle_build = false,
      bool use_bfs_cache = false)
      : m_dimension(dimension),
        m_metric(metric),
        m_index_name(index_name),
        m_store_directory(detail::diskann_store_directory(index_name)),
        m_api(&get_diskann_api()),
        m_offline_api(&get_diskann_offline_api()),
        m_vendored_api(&get_diskann_vendored_runtime_api()),
        m_build_complexity(build_complexity),
        m_max_degree(max_degree),
        m_build_threads(build_threads),
        m_build_blas_threads(build_blas_threads),
        m_offline_search_threads(offline_search_threads),
        m_search_io_limit(search_io_limit),
        m_cache_nodes(cache_nodes),
        m_search_beamwidth(search_beamwidth),
        m_pq_code_budget_size(pq_code_budget_size),
        m_pq_code_budget_ratio(pq_code_budget_ratio),
        m_disk_pq_dims(disk_pq_dims),
        m_accelerate_build(accelerate_build),
        m_shuffle_build(shuffle_build),
        m_use_bfs_cache(use_bfs_cache),
        m_search_cache_size(search_cache_size),
        m_search_cache_ratio(search_cache_ratio) {}

  ~diskann_native_state() {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    close_locked();
  }

  bool supported() const {
    DBUG_EXECUTE_IF("vector_backend_diskann_native_unavailable",
                    return false;);
    return m_api != nullptr && m_api->available();
  }

  std::string backend_variant() const {
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    return diskann_runtime_variant_name(m_runtime_variant);
  }

  backend_build_diagnostics build_diagnostics() const {
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    return m_last_build_diagnostics;
  }

  using ordered_entry = std::pair<uint64_t, const vector_data *>;
  using ordered_entries = std::vector<ordered_entry>;
  struct offline_loaded_handle {
    const void *handle{nullptr};
    diskann_runtime_variant variant{diskann_runtime_variant::kNone};
  };

  bool rebuild_from_reader_offline(const committed_entry_reader &reader,
                                   diskann_build_mode build_mode,
                                   size_t *entry_count,
                                   bool *fallback_allowed) {
    if (entry_count == nullptr || fallback_allowed == nullptr) return false;
    *entry_count = 0;
    *fallback_allowed = false;
    if (!reader) return false;
    if (build_mode == diskann_build_mode::kSerial) {
      std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
      begin_build_diagnostics_locked("serial_reader", kDiskAnnRuntimeSerial);
      const bool ok = rebuild_serial_reader_locked(reader, entry_count);
      set_build_input_stats_locked(*entry_count, 0);
      return ok;
    }

    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    const auto rebuild_serial_fallback = [&](const char *fallback_reason) {
      begin_build_diagnostics_locked("serial_reader",
                                     kDiskAnnRuntimeSerialFallback);
      set_build_fallback_reason_locked(fallback_reason);
      if (supported()) {
        const bool ok = rebuild_serial_reader_locked(reader, entry_count);
        set_build_input_stats_locked(*entry_count, 0);
        return ok;
      }
      *fallback_allowed = true;
      return false;
    };

    if (build_mode == diskann_build_mode::kAuto &&
        !offline_build_available_locked()) {
      if (!rebuild_serial_fallback("offline_unavailable")) {
        set_build_fallback_reason_locked("offline_and_serial_unavailable");
        return false;
      }
      return true;
    }

    const bool rebuild_ok = rebuild_offline_reader_locked(reader, entry_count);
    if (!rebuild_ok) {
      close_locked();
      if (build_mode == diskann_build_mode::kAuto &&
          diskann_offline_build_too_small(*entry_count, m_max_degree)) {
        return rebuild_serial_fallback(kDiskAnnFallbackOfflineMinRows);
      }
    }
    return rebuild_ok;
  }

  bool rebuild_from_raw_segments(const raw_vector_segment_reader &reader,
                                 diskann_build_mode build_mode,
                                 size_t *entry_count,
                                 bool *fallback_allowed) {
    if (entry_count == nullptr || fallback_allowed == nullptr) return false;
    *entry_count = 0;
    *fallback_allowed = false;
    if (!reader) return false;
    if (build_mode == diskann_build_mode::kSerial) {
      std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
      begin_build_diagnostics_locked("serial_raw_segments",
                                     kDiskAnnRuntimeSerial);
      const bool ok = rebuild_serial_raw_segments_locked(reader, entry_count);
      set_build_input_stats_locked(*entry_count, 0);
      return ok;
    }

    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    const auto rebuild_serial_fallback = [&](const char *fallback_reason) {
      begin_build_diagnostics_locked("serial_raw_segments",
                                     kDiskAnnRuntimeSerialFallback);
      set_build_fallback_reason_locked(fallback_reason);
      if (supported()) {
        const bool ok = rebuild_serial_raw_segments_locked(reader, entry_count);
        set_build_input_stats_locked(*entry_count, 0);
        return ok;
      }
      *fallback_allowed = true;
      return false;
    };

    if (build_mode == diskann_build_mode::kAuto &&
        !offline_build_available_locked()) {
      if (!rebuild_serial_fallback("offline_unavailable")) {
        set_build_fallback_reason_locked("offline_and_serial_unavailable");
        return false;
      }
      return true;
    }

    const bool rebuild_ok = rebuild_offline_raw_segments_locked(reader,
                                                                entry_count);
    if (!rebuild_ok) {
      close_locked();
      if (build_mode == diskann_build_mode::kAuto &&
          diskann_offline_build_too_small(*entry_count, m_max_degree)) {
        return rebuild_serial_fallback(kDiskAnnFallbackOfflineMinRows);
      }
    }
    return rebuild_ok;
  }

  bool rebuild_from_entries(
      const std::unordered_map<uint64_t, vector_data> &entries,
      diskann_build_mode build_mode) {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    ordered_entries ordered;
    ordered.reserve(entries.size());
    for (const auto &entry : entries) {
      if (entry.second.size() != m_dimension) {
        return false;
      }
      ordered.push_back({entry.first, &entry.second});
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto &lhs, const auto &rhs) {
                return lhs.first < rhs.first;
              });

    if (build_mode == diskann_build_mode::kOffline &&
        diskann_offline_build_too_small(ordered.size(), m_max_degree)) {
      begin_build_diagnostics_locked("entries", kDiskAnnRuntimeOfficialCppMain);
      set_build_input_stats_locked(ordered.size(), 0);
      set_build_fallback_reason_locked(kDiskAnnFallbackOfflineMinRows);
      return false;
    }

    if (build_mode == diskann_build_mode::kOffline ||
        (build_mode == diskann_build_mode::kAuto &&
         offline_build_available_locked() &&
         !diskann_offline_build_too_small(ordered.size(), m_max_degree))) {
      const bool rebuild_ok = rebuild_offline_entries_locked(ordered);
      if (!rebuild_ok) close_locked();
      return rebuild_ok;
    }

    begin_build_diagnostics_locked(
        "serial_entries", build_mode == diskann_build_mode::kAuto
                              ? kDiskAnnRuntimeSerialFallback
                              : kDiskAnnRuntimeSerial);
    if (build_mode == diskann_build_mode::kAuto) {
      set_build_fallback_reason_locked(
          diskann_offline_build_too_small(ordered.size(), m_max_degree)
              ? kDiskAnnFallbackOfflineMinRows
              : "offline_unavailable");
    }
    begin_build_memory_store_locked();
    if (!reopen_locked(false)) {
      set_build_input_stats_locked(ordered.size(), 0);
      discard_build_memory_store_locked();
      return false;
    }

    bool rebuild_ok = rebuild_ordered_entries_locked(ordered, build_mode);
    set_build_input_stats_locked(ordered.size(), 0);
    if (rebuild_ok) rebuild_ok = flush_build_memory_store_locked();
    if (!rebuild_ok) close_locked();
    discard_build_memory_store_locked();
    if (rebuild_ok) {
      m_last_build_diagnostics.build_invocations = ordered.empty() ? 0 : 1;
    }
    return rebuild_ok;
  }

  bool insert(uint64_t doc_id, const vector_data &vector) {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    if (m_offline_handle != nullptr) return false;
    return insert_locked(doc_id, vector);
  }

  bool remove(uint64_t doc_id) {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    if (m_offline_handle != nullptr) return false;
    if (!ensure_open_locked(false)) return false;
    const std::string doc_id_bytes = diskann_doc_id_bytes(doc_id);
    return m_api->remove(callback_context(), m_index_handle,
                         reinterpret_cast<const uint8_t *>(doc_id_bytes.data()),
                         doc_id_bytes.size());
  }

  bool supports_mutations() const {
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    return m_offline_handle == nullptr && m_index_handle != nullptr;
  }

  bool search(const vector_data &query, size_t top_k,
              std::vector<search_result> *results) const {
    if (results == nullptr || top_k > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    results->clear();
    if (m_offline_handle != nullptr) {
      std::vector<uint64_t> doc_ids(top_k, 0);
      std::vector<float> distances(top_k, 0.0F);
      const int32_t count = search_loaded_handle_locked(
          {m_offline_handle, m_offline_handle_variant}, query, top_k,
          doc_ids.data(), distances.data());
      if (count < 0) return false;
      for (int32_t i = 0; i < count; ++i) {
        results->push_back({doc_ids[static_cast<size_t>(i)],
                            distances[static_cast<size_t>(i)]});
      }
      return true;
    }
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

  bool collect_doc_ids(std::vector<uint64_t> *doc_ids) const {
    if (doc_ids == nullptr) return false;
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    doc_ids->clear();

    std::ifstream file(doc_ids_file_path_locked(),
                       std::ios::in | std::ios::binary);
    if (!file.is_open()) return false;

    uint64_t count = 0;
    if (!read_binary_value(file, &count) ||
        count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return false;
    }
    doc_ids->assign(static_cast<size_t>(count), 0);
    if (count == 0) return true;
    if (count > static_cast<uint64_t>(
                    std::numeric_limits<std::streamsize>::max() /
                    sizeof(uint64_t))) {
      return false;
    }
    file.read(reinterpret_cast<char *>(doc_ids->data()),
              static_cast<std::streamsize>(count * sizeof(uint64_t)));
    return file.good();
  }

  bool search_batch(const std::vector<vector_data> &queries, size_t begin,
                    size_t end, size_t top_k, uint32_t search_threads,
                    std::vector<std::vector<search_result>> *results) const {
    if (results == nullptr || begin > end || end > queries.size())
      return false;
    const size_t query_count = end - begin;
    results->clear();
    results->resize(query_count);
    if (query_count == 0 || top_k == 0) return true;

    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    if (m_offline_handle == nullptr ||
        top_k > std::numeric_limits<uint32_t>::max() ||
        query_count > std::numeric_limits<size_t>::max() / m_dimension ||
        query_count > std::numeric_limits<size_t>::max() / top_k) {
      return false;
    }
    if (m_offline_handle_variant == diskann_runtime_variant::kVendoredOffline) {
      if (m_vendored_api == nullptr || !m_vendored_api->batch_search_available())
        return false;
    } else if (m_offline_api == nullptr || m_offline_api->search_batch == nullptr) {
      return false;
    }

    static thread_local diskann_batch_search_scratch scratch;
    scratch.query_data.clear();
    scratch.query_data.reserve(query_count * m_dimension);
    for (size_t i = begin; i < end; ++i) {
      const vector_data &query = queries[i];
      if (query.size() != m_dimension) return false;
      scratch.query_data.insert(scratch.query_data.end(), query.begin(),
                                query.end());
    }

    const size_t result_slots = query_count * top_k;
    scratch.doc_ids.assign(result_slots, 0);
    scratch.distances.assign(result_slots, 0.0F);
    scratch.result_counts.assign(query_count, 0);
    int32_t searched = search_loaded_handle_batch_locked(
        {m_offline_handle, m_offline_handle_variant}, scratch.query_data.data(),
        query_count, top_k, search_threads, scratch.doc_ids.data(),
        scratch.distances.data(), scratch.result_counts.data());
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_offline_batch_search",
                    searched = -1;);
    DBUG_EXECUTE_IF(
        "vector_backend_diskann_offline_batch_search_oversized_count", {
          if (!scratch.result_counts.empty() &&
              top_k < std::numeric_limits<uint32_t>::max()) {
            scratch.result_counts.back() = static_cast<uint32_t>(top_k) + 1;
          }
        });
    if (searched < 0 || static_cast<size_t>(searched) != query_count) {
      results->clear();
      return false;
    }

    for (uint32_t result_count : scratch.result_counts) {
      if (result_count > top_k) {
        results->clear();
        return false;
      }
    }
    for (size_t query_idx = 0; query_idx < query_count; ++query_idx) {
      const size_t offset = query_idx * top_k;
      const uint32_t result_count = scratch.result_counts[query_idx];
      (*results)[query_idx].reserve(result_count);
      for (uint32_t i = 0; i < result_count; ++i) {
        (*results)[query_idx].push_back(
            {scratch.doc_ids[offset + i], scratch.distances[offset + i]});
      }
    }
    return true;
  }

  bool set_search_complexity(uint32_t search_complexity) {
    if (search_complexity == 0) return false;
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    m_search_complexity = search_complexity;
    return true;
  }

  bool set_search_beamwidth(uint32_t search_beamwidth) {
    if (search_beamwidth == 0) return false;
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    m_search_beamwidth = search_beamwidth;
    return true;
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

  bool loaded_handle_batch_rejects_null_for_testing() const {
    const std::vector<float> query(m_dimension, 0.0F);
    uint64_t doc_id = 0;
    float distance = 0.0F;
    uint32_t result_count = 0;
    const bool null_handle_rejected =
        search_loaded_handle_batch_locked({}, query.data(), 1, 1, 1, &doc_id,
                                          &distance, &result_count) < 0;
    const auto *fake_handle = reinterpret_cast<const void *>(0x1);
    const bool oversized_top_k_rejected =
        search_loaded_handle_batch_locked(
            {fake_handle, diskann_runtime_variant::kVendoredOffline},
            query.data(), 1,
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1, 1,
            &doc_id, &distance, &result_count) < 0;
    return null_handle_rejected && oversized_top_k_rejected;
  }

#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
  bool vendored_load_config_budget_for_testing(
      size_t row_count, double *build_memory_gb, uint32_t *pq_chunks,
      uint32_t *cache_nodes) const {
    if (build_memory_gb == nullptr || pq_chunks == nullptr ||
        cache_nodes == nullptr) {
      return false;
    }
    const mysql_vector_diskann_build_config config =
        make_vendored_load_config_locked(
            "idx_diskann_vendored_load_config_budget", row_count);
    *build_memory_gb = config.build_memory_gb;
    *pq_chunks = config.pq_chunks;
    *cache_nodes = config.cache_nodes;
    return true;
  }
#endif

  uint32_t effective_loaded_cache_nodes_for_testing(size_t row_count) const {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    m_loaded_cache_nodes = effective_cache_nodes_for_row_count(row_count);
    return effective_loaded_cache_nodes_locked();
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
  using build_clock = std::chrono::steady_clock;

  static uint64_t elapsed_build_ms(build_clock::time_point start) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            build_clock::now() - start)
            .count());
  }

  void begin_build_diagnostics_locked(const char *input_source,
                                      const char *runtime) {
    m_last_build_diagnostics = {};
    m_last_build_diagnostics.diskann_pq_runtime =
        vector_index::diskann_pq_runtime_mode_name(
            vector_index::global_diskann_pq_runtime_mode());
    if (runtime != nullptr) {
      m_last_build_diagnostics.runtime = runtime;
    }
    if (input_source != nullptr) {
      m_last_build_diagnostics.input_source = input_source;
    }
    const uint32_t build_threads = std::max<uint32_t>(1, m_build_threads);
    const uint32_t blas_threads = std::max<uint32_t>(1, m_build_blas_threads);
    m_last_build_diagnostics.concurrent_build_tasks = 1;
    m_last_build_diagnostics.scheduler_cpu_budget = build_threads;
    m_last_build_diagnostics.effective_build_threads = build_threads;
    m_last_build_diagnostics.effective_blas_threads =
        std::min(blas_threads, build_threads);
    m_last_build_diagnostics.pq_train_threads = build_threads;
    m_last_build_diagnostics.pq_compress_threads = build_threads;
  }

  void set_build_fallback_reason_locked(const char *fallback_reason) {
    if (fallback_reason != nullptr) {
      m_last_build_diagnostics.fallback_reason = fallback_reason;
    }
  }

  void set_build_runtime_locked(const char *runtime) {
    if (runtime != nullptr) m_last_build_diagnostics.runtime = runtime;
  }

  void set_build_input_stats_locked(uint64_t row_count,
                                    uint64_t segment_count) {
    m_last_build_diagnostics.row_count = row_count;
    m_last_build_diagnostics.segment_count = segment_count;
  }

  void add_build_load_ms_locked(build_clock::time_point start) {
    m_last_build_diagnostics.load_ms += elapsed_build_ms(start);
  }

  struct manifest_sample {
    uint64_t doc_id{0};
    uint64_t ordinal{0};
    vector_data vector;
  };

  static bool parse_manifest_uint64(const std::string &text,
                                    uint64_t *value) {
    if (value == nullptr || text.empty()) return false;
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0') return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
  }

  static bool split_manifest_line(const std::string &line,
                                  std::vector<std::string> *fields) {
    if (fields == nullptr) return false;
    fields->clear();

    size_t begin = 0;
    while (begin <= line.size()) {
      const size_t end = line.find('\t', begin);
      fields->push_back(line.substr(begin, end == std::string::npos
                                               ? std::string::npos
                                               : end - begin));
      if (end == std::string::npos) break;
      begin = end + 1;
    }
    return !fields->empty();
  }

  bool append_manifest_segment_samples_locked(
      const std::string &docids_path, const std::string &raw_path,
      uint64_t row_count, uint64_t ordinal_base, uint32_t dimension,
      std::vector<manifest_sample> *samples) const {
    if (samples == nullptr || row_count == 0 || dimension != m_dimension) {
      return false;
    }

    const size_t row_bytes = saturated_mul_size(m_dimension, sizeof(float));
    if (row_bytes == 0 || row_bytes == std::numeric_limits<size_t>::max() ||
        row_bytes >
            static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      return false;
    }

    std::ifstream docids_file(docids_path, std::ios::in | std::ios::binary);
    std::ifstream raw_file(raw_path, std::ios::in | std::ios::binary);
    if (!docids_file.is_open() || !raw_file.is_open()) return false;

    uint64_t docids_rows = 0;
    uint32_t raw_rows = 0;
    uint32_t raw_dimension = 0;
    if (!read_binary_value(docids_file, &docids_rows) ||
        !read_binary_value(raw_file, &raw_rows) ||
        !read_binary_value(raw_file, &raw_dimension) ||
        docids_rows != row_count || raw_rows != row_count ||
        raw_dimension != dimension) {
      return false;
    }

    std::array<uint64_t, 3> ordinals{0, row_count / 2, row_count - 1};
    for (const uint64_t ordinal : ordinals) {
      if (ordinal > (std::numeric_limits<uint64_t>::max() -
                     2 * sizeof(uint32_t)) /
                        row_bytes ||
          ordinal > (std::numeric_limits<uint64_t>::max() -
                     sizeof(uint64_t)) /
                        sizeof(uint64_t)) {
        return false;
      }

      const uint64_t raw_offset = 2 * sizeof(uint32_t) + ordinal * row_bytes;
      const uint64_t docid_offset = sizeof(uint64_t) +
                                    ordinal * sizeof(uint64_t);
      if (raw_offset >
              static_cast<uint64_t>(
                  std::numeric_limits<std::streamoff>::max()) ||
          docid_offset >
              static_cast<uint64_t>(
                  std::numeric_limits<std::streamoff>::max())) {
        return false;
      }

      uint64_t doc_id = 0;
      docids_file.seekg(static_cast<std::streamoff>(docid_offset),
                        std::ios::beg);
      if (!read_binary_value(docids_file, &doc_id)) return false;

      manifest_sample sample;
      sample.doc_id = doc_id;
      if (ordinal > std::numeric_limits<uint64_t>::max() - ordinal_base) {
        return false;
      }
      sample.ordinal = ordinal_base + ordinal;
      sample.vector.resize(m_dimension);
      raw_file.seekg(static_cast<std::streamoff>(raw_offset), std::ios::beg);
      raw_file.read(reinterpret_cast<char *>(sample.vector.data()),
                    static_cast<std::streamsize>(row_bytes));
      if (!raw_file.good()) return false;
      samples->push_back(std::move(sample));
    }
    return true;
  }

  bool read_manifest_self_hit_samples_locked(
      const std::string &manifest_path,
      std::vector<manifest_sample> *samples) const {
    if (samples == nullptr) return false;
    samples->clear();

    std::ifstream manifest_file(manifest_path);
    if (!manifest_file.is_open()) return false;

    std::string line;
    if (!std::getline(manifest_file, line) ||
        line != kDiskAnnRawManifestHeader) {
      return false;
    }

    uint32_t dimension = 0;
    uint64_t total_count = 0;
    uint64_t ordinal_base = 0;
    std::vector<std::string> fields;
    while (std::getline(manifest_file, line)) {
      if (!split_manifest_line(line, &fields)) return false;
      if (fields.empty()) continue;
      if (fields[0] == "dimension") {
        if (fields.size() != 2) return false;
        uint64_t parsed_dimension = 0;
        if (!parse_manifest_uint64(fields[1], &parsed_dimension) ||
            parsed_dimension > std::numeric_limits<uint32_t>::max()) {
          return false;
        }
        dimension = static_cast<uint32_t>(parsed_dimension);
        continue;
      }
      if (fields[0] == "count") {
        if (fields.size() != 2 ||
            !parse_manifest_uint64(fields[1], &total_count)) {
          return false;
        }
        continue;
      }
      if (fields[0] != "segment") continue;
      if (fields.size() != 4 || dimension == 0) return false;
      uint64_t row_count = 0;
      if (!parse_manifest_uint64(fields[3], &row_count) || row_count == 0) {
        return false;
      }
      if (!append_manifest_segment_samples_locked(fields[1], fields[2],
                                                  row_count, ordinal_base,
                                                  dimension, samples)) {
        return false;
      }
      if (row_count > std::numeric_limits<uint64_t>::max() - ordinal_base) {
        return false;
      }
      ordinal_base += row_count;
    }
    return total_count == 0 || !samples->empty();
  }

  bool read_segment_vector_by_doc_id_locked(const std::string &docids_path,
                                            const std::string &raw_path,
                                            uint64_t row_count,
                                            uint32_t dimension,
                                            uint64_t target_doc_id,
                                            vector_data *vector) const {
    if (vector == nullptr || dimension != m_dimension || row_count == 0) {
      return false;
    }

    std::ifstream docids_file(docids_path, std::ios::in | std::ios::binary);
    if (!docids_file.is_open()) return false;
    uint64_t docids_rows = 0;
    if (!read_binary_value(docids_file, &docids_rows) ||
        docids_rows != row_count) {
      return false;
    }

    constexpr size_t kDocIdReadBlock = 4096;
    std::vector<uint64_t> doc_ids(kDocIdReadBlock, 0);
    uint64_t ordinal = 0;
    bool found = false;
    while (ordinal < row_count) {
      const size_t rows_to_read = static_cast<size_t>(
          std::min<uint64_t>(kDocIdReadBlock, row_count - ordinal));
      docids_file.read(
          reinterpret_cast<char *>(doc_ids.data()),
          static_cast<std::streamsize>(rows_to_read * sizeof(uint64_t)));
      if (!docids_file.good()) return false;
      const auto match = std::find(
          doc_ids.begin(), doc_ids.begin() + rows_to_read, target_doc_id);
      if (match != doc_ids.begin() + rows_to_read) {
        ordinal += static_cast<uint64_t>(match - doc_ids.begin());
        found = true;
        break;
      }
      ordinal += rows_to_read;
    }
    if (!found) return false;

    const size_t row_bytes = saturated_mul_size(m_dimension, sizeof(float));
    if (row_bytes == 0 || row_bytes == std::numeric_limits<size_t>::max() ||
        row_bytes >
            static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) ||
        ordinal >
            (std::numeric_limits<uint64_t>::max() - 2 * sizeof(uint32_t)) /
                row_bytes) {
      return false;
    }
    const uint64_t raw_offset = 2 * sizeof(uint32_t) + ordinal * row_bytes;
    if (raw_offset >
        static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
      return false;
    }

    std::ifstream raw_file(raw_path, std::ios::in | std::ios::binary);
    if (!raw_file.is_open()) return false;
    uint32_t raw_rows = 0;
    uint32_t raw_dimension = 0;
    if (!read_binary_value(raw_file, &raw_rows) ||
        !read_binary_value(raw_file, &raw_dimension) || raw_rows != row_count ||
        raw_dimension != dimension) {
      return false;
    }
    vector->resize(m_dimension);
    raw_file.seekg(static_cast<std::streamoff>(raw_offset), std::ios::beg);
    raw_file.read(reinterpret_cast<char *>(vector->data()),
                  static_cast<std::streamsize>(row_bytes));
    if (!raw_file.good()) {
      vector->clear();
      return false;
    }
    return true;
  }

  bool read_manifest_vector_by_doc_id_locked(const std::string &manifest_path,
                                             uint64_t doc_id,
                                             vector_data *vector) const {
    if (vector == nullptr) return false;
    vector->clear();

    std::ifstream manifest_file(manifest_path);
    if (!manifest_file.is_open()) return false;
    std::string line;
    if (!std::getline(manifest_file, line) ||
        line != kDiskAnnRawManifestHeader) {
      return false;
    }

    uint32_t dimension = 0;
    std::vector<std::string> fields;
    while (std::getline(manifest_file, line)) {
      if (!split_manifest_line(line, &fields)) return false;
      if (fields.empty()) continue;
      if (fields[0] == "dimension") {
        if (fields.size() != 2) return false;
        uint64_t parsed_dimension = 0;
        if (!parse_manifest_uint64(fields[1], &parsed_dimension) ||
            parsed_dimension > std::numeric_limits<uint32_t>::max()) {
          return false;
        }
        dimension = static_cast<uint32_t>(parsed_dimension);
        continue;
      }
      if (fields[0] != "segment") continue;
      if (fields.size() != 4 || dimension == 0) return false;
      uint64_t row_count = 0;
      if (!parse_manifest_uint64(fields[3], &row_count) || row_count == 0) {
        return false;
      }
      if (read_segment_vector_by_doc_id_locked(fields[1], fields[2], row_count,
                                               dimension, doc_id, vector)) {
        return true;
      }
    }
    return false;
  }

  static std::string format_diagnostic_number(double value) {
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return output.str();
  }

  static bool validate_native_ordinal_samples_locked(
      const std::string &path, uint64_t row_count,
      const std::vector<manifest_sample> &samples) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    uint32_t stored_rows = 0;
    uint32_t stored_columns = 0;
    if (row_count > std::numeric_limits<uint32_t>::max() || !file.is_open() ||
        !read_binary_value(file, &stored_rows) ||
        !read_binary_value(file, &stored_columns) ||
        stored_rows != row_count || stored_columns != 1) {
      return false;
    }

    for (const manifest_sample &sample : samples) {
      if (sample.ordinal >= row_count ||
          sample.ordinal >
              (std::numeric_limits<uint64_t>::max() -
               2 * sizeof(uint32_t)) /
                  sizeof(uint64_t)) {
        return false;
      }
      const uint64_t offset =
          2 * sizeof(uint32_t) + sample.ordinal * sizeof(uint64_t);
      if (offset > static_cast<uint64_t>(
                       std::numeric_limits<std::streamoff>::max())) {
        return false;
      }
      file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
      uint64_t stored_ordinal = 0;
      if (!read_binary_value(file, &stored_ordinal) ||
          stored_ordinal != sample.ordinal) {
        return false;
      }
    }
    return true;
  }

  static bool validate_runtime_doc_id_samples_locked(
      const std::string &path, uint64_t row_count,
      const std::vector<manifest_sample> &samples) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    uint64_t stored_rows = 0;
    if (!file.is_open() || !read_binary_value(file, &stored_rows) ||
        stored_rows != row_count) {
      return false;
    }

    for (const manifest_sample &sample : samples) {
      if (sample.ordinal >= row_count ||
          sample.ordinal >
              (std::numeric_limits<uint64_t>::max() - sizeof(uint64_t)) /
                  sizeof(uint64_t)) {
        return false;
      }
      const uint64_t offset =
          sizeof(uint64_t) + sample.ordinal * sizeof(uint64_t);
      if (offset > static_cast<uint64_t>(
                       std::numeric_limits<std::streamoff>::max())) {
        return false;
      }
      file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
      uint64_t stored_doc_id = 0;
      if (!read_binary_value(file, &stored_doc_id) ||
          stored_doc_id != sample.doc_id) {
        return false;
      }
    }
    return true;
  }

  bool validate_native_pq_doc_ids_locked(
      const std::string &index_prefix, uint64_t row_count,
      const std::vector<manifest_sample> &samples, std::string *error) const {
    vector_index::diskann_pq_artifact_paths artifacts;
    if (!vector_index::make_diskann_pq_artifact_paths(
            native_pq_artifact_prefix(index_prefix), &artifacts) ||
        !validate_native_ordinal_samples_locked(artifacts.docid_ordinal_path,
                                                row_count, samples)) {
      if (error != nullptr) {
        *error = "native_pq_source_docid_ordinal_mismatch";
      }
      return false;
    }
    const bool valid_runtime = validate_runtime_doc_id_samples_locked(
        index_prefix + "_mysql_vector_docids.bin", row_count, samples);
    bool force_mismatch = false;
    DBUG_EXECUTE_IF(
        "vector_backend_diskann_native_pq_force_docid_ordinal_mismatch",
        force_mismatch = true;);
    if (valid_runtime && !force_mismatch) return true;
    if (error != nullptr) *error = "native_pq_runtime_docid_ordinal_mismatch";
    return false;
  }

  void record_native_pq_validation_failure_locked(
      const std::string &index_prefix, const std::string &manifest_path,
      const manifest_sample &sample, int32_t count,
      const std::vector<uint64_t> &doc_ids,
      const std::vector<float> &distances) {
    auto &diagnostics = m_last_build_diagnostics;
    diagnostics.native_pq_runtime_validation_failed = true;
    diagnostics.native_pq_runtime_validation_failed_doc_id =
        std::to_string(sample.doc_id);
    diagnostics.native_pq_runtime_validation_result_count =
        count > 0 ? static_cast<uint64_t>(count) : 0;

    if (count > 0) {
      diagnostics.native_pq_runtime_validation_best_doc_id =
          std::to_string(doc_ids.front());
      diagnostics.native_pq_runtime_validation_best_search_distance =
          format_diagnostic_number(distances.front());
      vector_data best_vector;
      bool found_best_vector = false;
      if (doc_ids.front() == sample.doc_id) {
        best_vector = sample.vector;
        found_best_vector = true;
      } else {
        found_best_vector = read_manifest_vector_by_doc_id_locked(
            manifest_path, doc_ids.front(), &best_vector);
      }
      double exact_distance = 0.0;
      if (found_best_vector && compute_distance(m_metric, sample.vector,
                                                best_vector, &exact_distance)) {
        diagnostics.native_pq_runtime_validation_best_exact_distance =
            format_diagnostic_number(exact_distance);
      } else {
        diagnostics.native_pq_runtime_validation_best_exact_distance =
            "unavailable";
      }
    } else {
      diagnostics.native_pq_runtime_validation_best_doc_id = "none";
      diagnostics.native_pq_runtime_validation_best_search_distance = "none";
      diagnostics.native_pq_runtime_validation_best_exact_distance = "none";
    }

    vector_index::diskann_pq_artifact_paths artifacts;
    vector_index::diskann_pq_bridge_manifest bridge_manifest;
    std::string artifact_error;
    if (!make_native_pq_bridge_artifact_paths_locked(index_prefix, &artifacts) ||
        !vector_index::read_diskann_pq_bridge_manifest(
            artifacts, &bridge_manifest, &artifact_error)) {
      diagnostics.native_pq_runtime_validation_self_pq_distance = "unavailable";
      diagnostics.native_pq_runtime_validation_pivots_checksum = "unavailable";
      diagnostics.native_pq_runtime_validation_compressed_checksum =
          "unavailable";
      return;
    }

    vector_index::diskann_pq_artifact_metadata metadata;
    metadata.row_count = bridge_manifest.row_count;
    metadata.dimension = bridge_manifest.dimension;
    metadata.pq_chunks = bridge_manifest.pq_chunks;
    metadata.centroid_count = bridge_manifest.centroid_count;
    metadata.zero_mean = bridge_manifest.zero_mean;

    vector_index::diskann_pq_bridge_manifest current_manifest;
    if (vector_index::make_diskann_pq_bridge_manifest(
            artifacts, metadata, bridge_manifest.max_degree,
            bridge_manifest.build_complexity, bridge_manifest.cache_nodes,
            bridge_manifest.artifacts_consumed,
            bridge_manifest.official_pq_used, &current_manifest,
            &artifact_error)) {
      diagnostics.native_pq_runtime_validation_pivots_checksum =
          std::to_string(current_manifest.pivots_checksum);
      diagnostics.native_pq_runtime_validation_compressed_checksum =
          std::to_string(current_manifest.compressed_checksum);
    } else {
      diagnostics.native_pq_runtime_validation_pivots_checksum = "unavailable";
      diagnostics.native_pq_runtime_validation_compressed_checksum =
          "unavailable";
    }

    vector_data reconstructed;
    double reconstruction_distance = 0.0;
    if (vector_index::reconstruct_diskann_pq_vector(
            artifacts, metadata, sample.ordinal, &reconstructed,
            &artifact_error) &&
        compute_distance(metric_type::kEuclidean, sample.vector, reconstructed,
                         &reconstruction_distance)) {
      diagnostics.native_pq_runtime_validation_self_pq_distance =
          format_diagnostic_number(reconstruction_distance);
    } else {
      diagnostics.native_pq_runtime_validation_self_pq_distance = "unavailable";
    }
  }

  bool validate_native_pq_serving_locked(const std::string &index_prefix,
                                         const std::string &manifest_path,
                                         uint64_t row_count,
                                         std::string *error) {
    if (row_count == 0) return true;

    std::vector<manifest_sample> samples;
    if (!read_manifest_self_hit_samples_locked(manifest_path, &samples)) {
      if (error != nullptr) *error = "native_pq_validation_sample_read_failed";
      return false;
    }
    if (!validate_native_pq_doc_ids_locked(index_prefix, row_count, samples,
                                           error)) {
      return false;
    }

    const offline_loaded_handle handle =
        load_offline_index_locked(index_prefix, row_count);
    if (handle.handle == nullptr) {
      if (error != nullptr) *error = "native_pq_validation_load_failed";
      return false;
    }

    const size_t validation_top_k =
        std::min<size_t>(10, static_cast<size_t>(row_count));
    std::vector<uint64_t> doc_ids(validation_top_k, 0);
    std::vector<float> distances(validation_top_k, 0.0F);
    bool ok = true;
    bool self_hit_miss_recorded = false;
    for (const manifest_sample &sample : samples) {
      std::fill(doc_ids.begin(), doc_ids.end(), 0);
      std::fill(distances.begin(), distances.end(), 0.0F);
      const int32_t count = search_loaded_handle_locked(
          handle, sample.vector, validation_top_k, doc_ids.data(),
          distances.data());
      bool force_failure = false;
      DBUG_EXECUTE_IF("vector_backend_diskann_native_pq_force_self_hit_failure",
                      force_failure = true;);
      const bool valid_count =
          count > 0 && static_cast<size_t>(count) <= validation_top_k;
      const bool self_hit =
          valid_count && std::find(doc_ids.begin(), doc_ids.begin() + count,
                                   sample.doc_id) != doc_ids.begin() + count;
      if (!valid_count) {
        ok = false;
        record_native_pq_validation_failure_locked(
            index_prefix, manifest_path, sample, 0, doc_ids, distances);
        if (error != nullptr) *error = "native_pq_validation_search_failed";
        break;
      }
      if ((!self_hit || force_failure) && !self_hit_miss_recorded) {
        record_native_pq_validation_failure_locked(
            index_prefix, manifest_path, sample, count, doc_ids, distances);
        self_hit_miss_recorded = true;
      }
    }

    drop_offline_handle_locked(handle);
    return ok;
  }

  bool build_native_pq_runtime_artifacts_locked(
      const std::string &input_file, const std::string &output_prefix,
      uint32_t pq_chunks, uint32_t disk_pq_chunks, uint32_t build_threads,
      uint64_t row_count, uint64_t memory_budget_size,
      vector_index::diskann_pq_artifact_paths *artifacts,
      vector_index::diskann_pq_artifact_paths *disk_artifacts,
      uint32_t *centroid_count, bool *memory_budget_exceeded) {
    if (memory_budget_exceeded != nullptr) *memory_budget_exceeded = false;
    if (!vector_index::diskann_pq_runtime_uses_native(
            vector_index::global_diskann_pq_runtime_mode())) {
      return false;
    }
    if (artifacts == nullptr || centroid_count == nullptr ||
        (disk_pq_chunks != 0 && disk_artifacts == nullptr) ||
        !vector_index::make_diskann_pq_artifact_paths(output_prefix,
                                                      artifacts)) {
      m_last_build_diagnostics.native_pq_runtime_selected_path =
          "artifact_path_unavailable";
      return false;
    }

    vector_index::diskann_pq_runtime_config config;
    config.dimension = static_cast<uint32_t>(m_dimension);
    config.pq_chunks = pq_chunks;
    config.disk_pq_chunks = disk_pq_chunks;
    config.threads = build_threads;
    config.memory_budget_size = memory_budget_size;
    config.prefer_avx512 = true;
    config.normalize_input = m_metric == metric_type::kCosine;

    vector_index::diskann_pq_runtime_result result;
    std::string error;
    if (!vector_index::build_diskann_pq_runtime(
            config, input_file.c_str(), output_prefix.c_str(), &result,
            &error)) {
      m_last_build_diagnostics.native_pq_runtime_selected_path =
          error.empty() ? "diagnostic_unavailable" : error;
      if (memory_budget_exceeded != nullptr &&
          error == "native_pq_memory_budget_exceeded") {
        *memory_budget_exceeded = true;
      }
      return false;
    }

    m_last_build_diagnostics.native_pq_runtime_selected_path =
        result.selected_path;
    m_last_build_diagnostics.native_pq_runtime_elapsed_ms = result.elapsed_ms;
    m_last_build_diagnostics.native_pq_runtime_raw_reader_ms =
        result.raw_reader_ms;
    m_last_build_diagnostics.native_pq_runtime_train_ms = result.train_ms;
    m_last_build_diagnostics.native_pq_runtime_encode_ms = result.encode_ms;
    m_last_build_diagnostics.native_pq_runtime_artifact_validation_ms =
        result.artifact_validation_ms;
    m_last_build_diagnostics.native_pq_runtime_centroid_scan_kernel =
        result.centroid_scan_kernel;
    m_last_build_diagnostics.native_pq_runtime_train_rows = result.train_rows;
    m_last_build_diagnostics.native_pq_runtime_compressed_rows =
        result.compressed_rows;
    m_last_build_diagnostics.native_pq_runtime_encode_block_rows =
        result.encode_block_rows;
    m_last_build_diagnostics.native_pq_runtime_memory_estimate =
        result.memory_estimate_bytes;
    m_last_build_diagnostics.native_pq_runtime_memory_budget =
        result.memory_budget_size;
    m_last_build_diagnostics.native_pq_runtime_effective_threads =
        result.effective_threads;
    m_last_build_diagnostics.native_pq_runtime_memory_adjustment =
        result.memory_budget_adjustment;
    m_last_build_diagnostics.native_pq_runtime_artifacts_written =
        result.artifacts_written;
    *artifacts = result.artifacts;
    if (disk_artifacts != nullptr) {
      *disk_artifacts = result.disk_artifacts;
    }
    *centroid_count = result.centroid_count;
    if (row_count != 0 &&
        row_count <= std::numeric_limits<uint64_t>::max() / pq_chunks) {
      m_last_build_diagnostics.native_pq_runtime_distance_calls =
          row_count * pq_chunks;
    } else {
      m_last_build_diagnostics.native_pq_runtime_distance_calls =
          result.distance_calls;
    }
    return result.artifacts_written &&
           (disk_pq_chunks == 0 || result.disk_artifacts_written);
  }

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

  bool serial_insert_entries_locked(const ordered_entries &entries) {
    for (const auto &entry : entries) {
      if (!insert_locked(entry.first, *entry.second)) return false;
    }
    return true;
  }

  bool serial_insert_reader_locked(const committed_entry_reader &reader,
                                   size_t *entry_count,
                                   std::ofstream *doc_ids_file) {
    if (!reader || entry_count == nullptr) return false;
    *entry_count = 0;
    return reader([&](uint64_t doc_id, const vector_data &vector) {
      if (vector.size() != m_dimension ||
          *entry_count == std::numeric_limits<size_t>::max()) {
        return false;
      }
      if (!insert_locked(doc_id, vector)) return false;
      if (doc_ids_file != nullptr &&
          !write_binary_value(*doc_ids_file, doc_id)) {
        return false;
      }
      ++(*entry_count);
      return true;
    });
  }

  std::filesystem::path doc_ids_file_path_for_store_locked(
      const std::string &store_directory) const {
    return offline_index_prefix_for_store(store_directory) +
           "_mysql_vector_docids.bin";
  }

  std::filesystem::path doc_ids_file_path_locked() const {
    return doc_ids_file_path_for_store_locked(m_store_directory);
  }

  std::filesystem::path doc_ids_staging_file_path_locked() const {
    return m_store_directory + ".docids-tmp-" +
           std::to_string(reinterpret_cast<std::uintptr_t>(this));
  }

  bool begin_doc_ids_file_locked(std::ofstream *file) const {
    if (file == nullptr) return false;
    const std::filesystem::path path = doc_ids_staging_file_path_locked();
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return false;
    std::filesystem::remove(path, ec);
    if (ec) return false;
    file->open(path, std::ios::out | std::ios::binary | std::ios::trunc);
    return file->is_open() && write_binary_value(*file, uint64_t{0});
  }

  bool finish_doc_ids_file_locked(std::ofstream *file, uint64_t entry_count) const {
    if (file == nullptr || !file->is_open()) return false;
    file->seekp(0);
    if (!write_binary_value(*file, entry_count)) return false;
    file->close();
    return file->good();
  }

  bool move_doc_ids_file_to_store_locked(
      const std::filesystem::path &store_directory) const {
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_doc_ids_stage", return false;);
    const std::filesystem::path source = doc_ids_staging_file_path_locked();
    const std::filesystem::path target =
        doc_ids_file_path_for_store_locked(store_directory.string());
    std::error_code ec;
    std::filesystem::create_directories(target.parent_path(), ec);
    if (ec) return false;
    std::filesystem::rename(source, target, ec);
    return !ec;
  }

  void discard_doc_ids_file_locked(std::ofstream *file) const {
    if (file != nullptr && file->is_open()) file->close();
    std::error_code ec;
    std::filesystem::remove(doc_ids_staging_file_path_locked(), ec);
  }

  bool rebuild_serial_reader_locked(const committed_entry_reader &reader,
                                    size_t *entry_count) {
    begin_build_memory_store_locked();
    if (!reopen_locked(false)) {
      discard_build_memory_store_locked();
      return false;
    }

    std::ofstream doc_ids_file;
    bool rebuild_ok = begin_doc_ids_file_locked(&doc_ids_file);
    if (rebuild_ok) {
      rebuild_ok = serial_insert_reader_locked(reader, entry_count,
                                               &doc_ids_file);
    }
    if (rebuild_ok) {
      rebuild_ok = finish_doc_ids_file_locked(
          &doc_ids_file, static_cast<uint64_t>(*entry_count));
    }
    if (rebuild_ok) {
      rebuild_ok = flush_build_memory_store_locked(true);
    }
    if (!rebuild_ok) close_locked();
    if (!rebuild_ok) discard_doc_ids_file_locked(&doc_ids_file);
    discard_build_memory_store_locked();
    if (rebuild_ok) {
      m_last_build_diagnostics.build_invocations = *entry_count == 0 ? 0 : 1;
      m_runtime_variant = diskann_runtime_variant::kSerial;
    }
    return rebuild_ok;
  }

  bool rebuild_serial_raw_segments_locked(
      const raw_vector_segment_reader &reader, size_t *entry_count) {
    if (!reader) return false;
    const committed_entry_reader entry_reader =
        [this, &reader](const committed_entry_visitor &visitor) {
          return reader([this, &visitor](const raw_vector_segment &segment) {
            if (!validate_raw_segment_locked(segment)) return false;
            vector_load_file_info info;
            std::string error;
            return read_fbin_vectors(
                       segment.vector_path, segment.docid_path, m_dimension,
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
    return rebuild_serial_reader_locked(entry_reader, entry_count);
  }

  bool offline_build_manifest_entries_locked(const ordered_entries &entries,
                                             const std::string &index_prefix) {
    if (!manifest_build_available_locked()) return false;
    const build_clock::time_point manifest_start = build_clock::now();

    const size_t row_bytes = saturated_mul_size(m_dimension, sizeof(float));
    if (row_bytes == 0 ||
        row_bytes == std::numeric_limits<size_t>::max() ||
        m_dimension > std::numeric_limits<uint32_t>::max()) {
      return false;
    }

    const std::filesystem::path prefix(index_prefix);
    std::filesystem::path segment_dir(prefix);
    segment_dir += "_raw_segments";

    std::error_code ec;
    std::filesystem::remove_all(segment_dir, ec);
    if (ec) return false;
    std::filesystem::create_directories(segment_dir, ec);
    if (ec) return false;

    const std::filesystem::path manifest_path =
        segment_dir / "mysql-vector-diskann-raw.manifest";
    std::ofstream manifest_file(manifest_path,
                                std::ios::out | std::ios::trunc);
    if (!manifest_file.is_open()) {
      std::filesystem::remove_all(segment_dir, ec);
      return false;
    }
    manifest_file << kDiskAnnRawManifestHeader << '\n'
                  << "dimension\t" << m_dimension << '\n'
                  << "count\t" << entries.size() << '\n';

    const size_t configured_segment_size =
        static_cast<size_t>(opt_vector_diskann_raw_segment_size);
    const size_t segment_size = std::max(configured_segment_size, row_bytes);
    size_t segment_id = 0;
    size_t segment_count = 0;
    size_t segment_payload_bytes = 0;
    std::filesystem::path raw_path;
    std::filesystem::path doc_ids_path;
    std::ofstream raw_file;
    std::ofstream doc_ids_file;

    auto close_segment = [&]() -> bool {
      if (segment_count == 0) return true;
      if (segment_count > std::numeric_limits<uint32_t>::max()) return false;
      raw_file.seekp(0);
      if (!write_binary_value(raw_file, static_cast<uint32_t>(segment_count)) ||
          !write_binary_value(raw_file, static_cast<uint32_t>(m_dimension))) {
        return false;
      }
      raw_file.close();

      doc_ids_file.seekp(0);
      if (!write_binary_value(doc_ids_file,
                              static_cast<uint64_t>(segment_count))) {
        return false;
      }
      doc_ids_file.close();

      manifest_file << "segment\t" << doc_ids_path.string() << '\t'
                    << raw_path.string() << '\t' << segment_count << '\n';
      segment_count = 0;
      segment_payload_bytes = 0;
      return manifest_file.good();
    };

    auto open_segment = [&]() -> bool {
      raw_path =
          segment_dir / ("segment-" + std::to_string(segment_id) + ".fbin");
      doc_ids_path =
          segment_dir / ("segment-" + std::to_string(segment_id) + ".docids");
      ++segment_id;
      raw_file.open(raw_path, std::ios::out | std::ios::binary |
                                  std::ios::trunc);
      doc_ids_file.open(doc_ids_path, std::ios::out | std::ios::binary |
                                          std::ios::trunc);
      if (!raw_file.is_open() || !doc_ids_file.is_open()) return false;

      return write_binary_value(raw_file, static_cast<uint32_t>(0)) &&
             write_binary_value(raw_file, static_cast<uint32_t>(m_dimension)) &&
             write_binary_value(doc_ids_file, static_cast<uint64_t>(0));
    };

    bool ok = true;
    for (const auto &entry : entries) {
      if (entry.second == nullptr || entry.second->size() != m_dimension) {
        ok = false;
        break;
      }
      if (segment_count != 0 &&
          segment_payload_bytes > segment_size - row_bytes) {
        if (!close_segment()) {
          ok = false;
          break;
        }
      }
      if (segment_count == 0 && !open_segment()) {
        ok = false;
        break;
      }

      const uint64_t doc_id = entry.first;
      if (!write_binary_value(doc_ids_file, doc_id)) {
        ok = false;
        break;
      }
      if (row_bytes >
          static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        ok = false;
        break;
      }
      raw_file.write(reinterpret_cast<const char *>(entry.second->data()),
                     static_cast<std::streamsize>(row_bytes));
      if (!raw_file.good()) {
        ok = false;
        break;
      }
      ++segment_count;
      segment_payload_bytes += row_bytes;
    }
    if (ok) ok = close_segment();
    manifest_file.close();

    if (!ok) {
      if (raw_file.is_open()) raw_file.close();
      if (doc_ids_file.is_open()) doc_ids_file.close();
      std::filesystem::remove_all(segment_dir, ec);
      return false;
    }

    const uint32_t build_threads = m_build_threads;
    vector_index::diskann_segment_budget budget;
    if (!make_segment_budget_locked(entries.size(), &budget)) return false;
    set_build_input_stats_locked(entries.size(), segment_id);
    m_last_build_diagnostics.manifest_ms =
        elapsed_build_ms(manifest_start);
    const std::string manifest_path_text = manifest_path.string();
    record_segment_budget_locked(budget);
    const uint32_t max_degree =
        diskann_effective_offline_max_degree(entries.size(), m_max_degree);
    if (max_degree == 0) return false;
    const build_clock::time_point build_start = build_clock::now();
    ok = build_offline_manifest_locked(
        index_prefix, manifest_path_text, static_cast<uint32_t>(m_dimension),
        diskann_metric_code(m_metric), max_degree, m_build_complexity,
        build_threads, budget.build_memory_gb, budget.pq_chunks,
        budget.cache_nodes, budget.build_memory_size, m_build_blas_threads,
        budget.disk_pq_dims, m_accelerate_build, m_shuffle_build,
        entries.size());
    m_last_build_diagnostics.offline_build_ms =
        elapsed_build_ms(build_start);
    std::filesystem::remove_all(segment_dir, ec);
    return ok;
  }

  bool offline_build_manifest_reader_locked(const committed_entry_reader &reader,
                                            const std::string &index_prefix,
                                            size_t *entry_count) {
    if (!reader || entry_count == nullptr ||
        !manifest_build_available_locked()) {
      return false;
    }
    *entry_count = 0;
    const build_clock::time_point manifest_start = build_clock::now();

    const size_t row_bytes = saturated_mul_size(m_dimension, sizeof(float));
    if (row_bytes == 0 ||
        row_bytes == std::numeric_limits<size_t>::max() ||
        m_dimension > std::numeric_limits<uint32_t>::max() ||
        row_bytes >
            static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      return false;
    }

    struct manifest_segment {
      std::filesystem::path doc_ids_path;
      std::filesystem::path raw_path;
      size_t row_count{0};
    };

    const std::filesystem::path prefix(index_prefix);
    std::filesystem::path segment_dir(prefix);
    segment_dir += "_raw_segments";

    std::error_code ec;
    std::filesystem::remove_all(segment_dir, ec);
    if (ec) return false;
    std::filesystem::create_directories(segment_dir, ec);
    if (ec) return false;

    const size_t configured_segment_size =
        static_cast<size_t>(opt_vector_diskann_raw_segment_size);
    const size_t segment_size = std::max(configured_segment_size, row_bytes);
    size_t segment_id = 0;
    size_t segment_count = 0;
    size_t segment_payload_bytes = 0;
    std::filesystem::path raw_path;
    std::filesystem::path doc_ids_path;
    std::ofstream raw_file;
    std::ofstream doc_ids_file;
    std::vector<manifest_segment> segments;

    auto close_open_files = [&]() {
      if (raw_file.is_open()) raw_file.close();
      if (doc_ids_file.is_open()) doc_ids_file.close();
    };

    auto close_segment = [&]() -> bool {
      if (segment_count == 0) return true;
      if (segment_count > std::numeric_limits<uint32_t>::max()) return false;
      raw_file.seekp(0);
      if (!write_binary_value(raw_file, static_cast<uint32_t>(segment_count)) ||
          !write_binary_value(raw_file, static_cast<uint32_t>(m_dimension))) {
        return false;
      }
      raw_file.close();

      doc_ids_file.seekp(0);
      if (!write_binary_value(doc_ids_file,
                              static_cast<uint64_t>(segment_count))) {
        return false;
      }
      doc_ids_file.close();

      segments.push_back({doc_ids_path, raw_path, segment_count});
      segment_count = 0;
      segment_payload_bytes = 0;
      return true;
    };

    auto open_segment = [&]() -> bool {
      raw_path =
          segment_dir / ("segment-" + std::to_string(segment_id) + ".fbin");
      doc_ids_path =
          segment_dir / ("segment-" + std::to_string(segment_id) + ".docids");
      ++segment_id;
      raw_file.open(raw_path, std::ios::out | std::ios::binary |
                                  std::ios::trunc);
      doc_ids_file.open(doc_ids_path, std::ios::out | std::ios::binary |
                                          std::ios::trunc);
      if (!raw_file.is_open() || !doc_ids_file.is_open()) return false;

      return write_binary_value(raw_file, static_cast<uint32_t>(0)) &&
             write_binary_value(raw_file, static_cast<uint32_t>(m_dimension)) &&
             write_binary_value(doc_ids_file, static_cast<uint64_t>(0));
    };

    bool ok = reader([&](uint64_t doc_id, const vector_data &vector) {
      if (vector.size() != m_dimension ||
          *entry_count == std::numeric_limits<size_t>::max()) {
        return false;
      }
      if (segment_count != 0 &&
          segment_payload_bytes > segment_size - row_bytes) {
        if (!close_segment()) return false;
      }
      if (segment_count == 0 && !open_segment()) return false;

      if (!write_binary_value(doc_ids_file, doc_id)) return false;
      raw_file.write(reinterpret_cast<const char *>(vector.data()),
                     static_cast<std::streamsize>(row_bytes));
      if (!raw_file.good()) return false;

      ++segment_count;
      ++(*entry_count);
      segment_payload_bytes += row_bytes;
      return true;
    });
    if (ok) ok = close_segment();

    if (!ok) {
      close_open_files();
      std::filesystem::remove_all(segment_dir, ec);
      return false;
    }
    close_open_files();

    if (*entry_count == 0) {
      std::filesystem::remove_all(segment_dir, ec);
      return true;
    }
    if (diskann_offline_build_too_small(*entry_count, m_max_degree)) {
      set_build_input_stats_locked(*entry_count, segments.size());
      set_build_fallback_reason_locked(kDiskAnnFallbackOfflineMinRows);
      std::filesystem::remove_all(segment_dir, ec);
      return false;
    }

    const std::filesystem::path manifest_path =
        segment_dir / "mysql-vector-diskann-raw.manifest";
    std::ofstream manifest_file(manifest_path, std::ios::out | std::ios::trunc);
    if (!manifest_file.is_open()) {
      std::filesystem::remove_all(segment_dir, ec);
      return false;
    }
    manifest_file << kDiskAnnRawManifestHeader << '\n'
                  << "dimension\t" << m_dimension << '\n'
                  << "count\t" << *entry_count << '\n';
    for (const manifest_segment &segment : segments) {
      manifest_file << "segment\t" << segment.doc_ids_path.string() << '\t'
                    << segment.raw_path.string() << '\t' << segment.row_count
                    << '\n';
    }
    manifest_file.close();
    if (!manifest_file.good()) {
      std::filesystem::remove_all(segment_dir, ec);
      return false;
    }

    vector_index::diskann_segment_budget budget;
    if (!make_segment_budget_locked(*entry_count, &budget)) {
      std::filesystem::remove_all(segment_dir, ec);
      return false;
    }

    set_build_input_stats_locked(*entry_count, segments.size());
    m_last_build_diagnostics.manifest_ms =
        elapsed_build_ms(manifest_start);
    const std::string manifest_path_text = manifest_path.string();
    record_segment_budget_locked(budget);
    const uint32_t max_degree =
        diskann_effective_offline_max_degree(*entry_count, m_max_degree);
    if (max_degree == 0) {
      std::filesystem::remove_all(segment_dir, ec);
      return false;
    }
    const build_clock::time_point build_start = build_clock::now();
    ok = build_offline_manifest_locked(
        index_prefix, manifest_path_text, static_cast<uint32_t>(m_dimension),
        diskann_metric_code(m_metric), max_degree, m_build_complexity,
        m_build_threads, budget.build_memory_gb, budget.pq_chunks,
        budget.cache_nodes, budget.build_memory_size, m_build_blas_threads,
        budget.disk_pq_dims, m_accelerate_build, m_shuffle_build,
        *entry_count);
    m_last_build_diagnostics.offline_build_ms =
        elapsed_build_ms(build_start);
    std::filesystem::remove_all(segment_dir, ec);
    return ok;
  }

  bool validate_raw_segment_locked(const raw_vector_segment &segment) const {
    if (segment.dimension != m_dimension || segment.row_count == 0 ||
        segment.row_count > std::numeric_limits<uint32_t>::max()) {
      return false;
    }

    const size_t row_bytes = saturated_mul_size(m_dimension, sizeof(float));
    if (row_bytes == 0 ||
        row_bytes == std::numeric_limits<size_t>::max()) {
      return false;
    }
    if (segment.row_count >
        (std::numeric_limits<size_t>::max() - 2 * sizeof(uint32_t)) /
            row_bytes) {
      return false;
    }
    if (segment.row_count >
        (std::numeric_limits<size_t>::max() - sizeof(uint64_t)) /
            sizeof(uint64_t)) {
      return false;
    }

    const size_t expected_vector_bytes =
        2 * sizeof(uint32_t) + segment.row_count * row_bytes;
    const size_t expected_docid_bytes =
        sizeof(uint64_t) + segment.row_count * sizeof(uint64_t);
    if (segment.bytes != 0 &&
        segment.bytes != expected_vector_bytes + expected_docid_bytes) {
      return false;
    }

    std::error_code ec;
    const uintmax_t vector_bytes =
        std::filesystem::file_size(segment.vector_path, ec);
    if (ec || vector_bytes != expected_vector_bytes) return false;
    ec.clear();
    const uintmax_t docid_bytes = std::filesystem::file_size(segment.docid_path,
                                                            ec);
    if (ec || docid_bytes != expected_docid_bytes) return false;

    std::ifstream vector_file(segment.vector_path, std::ios::in |
                                                       std::ios::binary);
    std::ifstream docid_file(segment.docid_path, std::ios::in |
                                                     std::ios::binary);
    if (!vector_file.is_open() || !docid_file.is_open()) return false;

    uint32_t fbin_rows = 0;
    uint32_t fbin_dimension = 0;
    uint64_t docid_rows = 0;
    return read_binary_value(vector_file, &fbin_rows) &&
           read_binary_value(vector_file, &fbin_dimension) &&
           read_binary_value(docid_file, &docid_rows) &&
           fbin_rows == segment.row_count &&
           docid_rows == segment.row_count &&
           fbin_dimension == m_dimension;
  }

  bool offline_build_raw_segments_locked(
      const raw_vector_segment_reader &reader, const std::string &index_prefix,
      size_t *entry_count) {
    if (!reader || entry_count == nullptr || m_offline_api == nullptr ||
        !manifest_build_available_locked()) {
      return false;
    }

    const build_clock::time_point manifest_start = build_clock::now();
    std::vector<raw_vector_segment> segments;
    size_t total_rows = 0;
    const bool read_ok = reader([&](const raw_vector_segment &segment) {
      if (!validate_raw_segment_locked(segment) ||
          segment.row_count > std::numeric_limits<size_t>::max() - total_rows) {
        return false;
      }
      total_rows += segment.row_count;
      segments.push_back(segment);
      return true;
    });
    if (!read_ok) return false;
    *entry_count = total_rows;
    if (segments.empty()) return total_rows == 0;
    if (diskann_offline_build_too_small(total_rows, m_max_degree)) {
      set_build_input_stats_locked(total_rows, segments.size());
      set_build_fallback_reason_locked(kDiskAnnFallbackOfflineMinRows);
      return false;
    }

    const std::filesystem::path prefix(index_prefix);
    std::filesystem::path manifest_dir(prefix);
    manifest_dir += "_raw_segments";

    std::error_code ec;
    std::filesystem::remove_all(manifest_dir, ec);
    if (ec) return false;
    std::filesystem::create_directories(manifest_dir, ec);
    if (ec) return false;

    const std::filesystem::path manifest_path =
        manifest_dir / "mysql-vector-diskann-raw.manifest";
    std::ofstream manifest_file(manifest_path, std::ios::out | std::ios::trunc);
    if (!manifest_file.is_open()) {
      std::filesystem::remove_all(manifest_dir, ec);
      return false;
    }
    manifest_file << kDiskAnnRawManifestHeader << '\n'
                  << "dimension\t" << m_dimension << '\n'
                  << "count\t" << total_rows << '\n';
    for (const raw_vector_segment &segment : segments) {
      manifest_file << "segment\t" << segment.docid_path << '\t'
                    << segment.vector_path << '\t' << segment.row_count << '\n';
    }
    manifest_file.close();
    if (!manifest_file.good()) {
      std::filesystem::remove_all(manifest_dir, ec);
      return false;
    }

    vector_index::diskann_segment_budget budget;
    if (!make_segment_budget_locked(total_rows, &budget)) {
      std::filesystem::remove_all(manifest_dir, ec);
      return false;
    }

    const std::string manifest_path_text = manifest_path.string();
    record_segment_budget_locked(budget);
    const uint32_t max_degree =
        diskann_effective_offline_max_degree(total_rows, m_max_degree);
    if (max_degree == 0) {
      std::filesystem::remove_all(manifest_dir, ec);
      return false;
    }
    set_build_input_stats_locked(total_rows, segments.size());
    m_last_build_diagnostics.single_index_build = true;
    m_last_build_diagnostics.raw_reader_threads =
        static_cast<uint32_t>(std::max<size_t>(
            1, std::min<size_t>(segments.size(),
                                std::max<uint32_t>(1, m_build_threads))));
    m_last_build_diagnostics.manifest_ms =
        elapsed_build_ms(manifest_start);
    const build_clock::time_point build_start = build_clock::now();
    const bool ok = build_offline_manifest_locked(
        index_prefix, manifest_path_text, static_cast<uint32_t>(m_dimension),
        diskann_metric_code(m_metric), max_degree, m_build_complexity,
        m_build_threads, budget.build_memory_gb, budget.pq_chunks,
        budget.cache_nodes, budget.build_memory_size, m_build_blas_threads,
        budget.disk_pq_dims, m_accelerate_build, m_shuffle_build, total_rows);
    m_last_build_diagnostics.offline_build_ms =
        elapsed_build_ms(build_start);
    std::filesystem::remove_all(manifest_dir, ec);
    return ok;
  }

  bool offline_build_entries_locked(const ordered_entries &entries,
                                    const std::string &index_prefix) {
    if (entries.empty()) return true;
    if (!manifest_build_available_locked()) return false;
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_offline_build",
                    return false;);
    return offline_build_manifest_entries_locked(entries, index_prefix);
  }

  bool offline_build_reader_locked(const committed_entry_reader &reader,
                                   const std::string &index_prefix,
                                   size_t *entry_count) {
    if (!manifest_build_available_locked()) return false;
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_offline_build",
                    return false;);
    return offline_build_manifest_reader_locked(reader, index_prefix,
                                                entry_count);
  }

  bool rebuild_offline_entries_locked(const ordered_entries &entries) {
    begin_build_diagnostics_locked("entries",
                                   kDiskAnnRuntimeOfficialCppMain);
    set_build_input_stats_locked(entries.size(), 0);
    if (m_store_directory.empty()) return false;
    if (entries.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(m_store_directory, ec);
      if (ec) return false;
      close_locked();
      m_runtime_variant = offline_runtime_variant_for_last_build_locked();
      return true;
    }
    if (m_offline_api == nullptr || !m_offline_api->available()) return false;

    const std::string suffix =
        ".offline-tmp-" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
    const std::filesystem::path staging(m_store_directory + suffix);
    std::error_code ec;
    std::filesystem::remove_all(staging, ec);
    if (ec) return false;
    std::filesystem::create_directories(staging, ec);
    if (ec) return false;

    const std::string staging_prefix =
        offline_index_prefix_for_store(staging.string());
    if (!offline_build_entries_locked(entries, staging_prefix)) {
      std::filesystem::remove_all(staging, ec);
      return false;
    }

    const build_clock::time_point staging_load_start = build_clock::now();
    const offline_loaded_handle staging_handle =
        load_offline_index_locked(staging_prefix, entries.size());
    add_build_load_ms_locked(staging_load_start);
    if (staging_handle.handle == nullptr) {
      std::filesystem::remove_all(staging, ec);
      return false;
    }
    drop_offline_handle_locked(staging_handle);

    offline_loaded_handle handle;
    const build_clock::time_point promote_start = build_clock::now();
    if (!promote_offline_store_locked(staging, &handle, entries.size())) {
      std::filesystem::remove_all(staging, ec);
      return false;
    }
    add_build_load_ms_locked(promote_start);
    close_locked();
    m_offline_handle = handle.handle;
    m_offline_handle_variant = handle.variant;
    m_runtime_variant = handle.variant;
    return true;
  }

  bool rebuild_offline_reader_locked(const committed_entry_reader &reader,
                                     size_t *entry_count) {
    begin_build_diagnostics_locked("reader",
                                   kDiskAnnRuntimeOfficialCppMain);
    if (m_store_directory.empty() || entry_count == nullptr) return false;
    *entry_count = 0;

    const std::string suffix =
        ".offline-tmp-" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
    const std::filesystem::path staging(m_store_directory + suffix);
    std::error_code ec;
    std::filesystem::remove_all(staging, ec);
    if (ec) return false;
    std::filesystem::create_directories(staging, ec);
    if (ec) return false;

    const std::string staging_prefix =
        offline_index_prefix_for_store(staging.string());
    if (!offline_build_reader_locked(reader, staging_prefix, entry_count)) {
      std::filesystem::remove_all(staging, ec);
      return false;
    }

    if (*entry_count == 0) {
      std::filesystem::remove_all(staging, ec);
      if (ec) return false;
      std::filesystem::remove_all(m_store_directory, ec);
      if (ec) return false;
      close_locked();
      m_runtime_variant = offline_runtime_variant_for_last_build_locked();
      return true;
    }

    const build_clock::time_point staging_load_start = build_clock::now();
    const offline_loaded_handle staging_handle =
        load_offline_index_locked(staging_prefix, *entry_count);
    add_build_load_ms_locked(staging_load_start);
    if (staging_handle.handle == nullptr) {
      std::filesystem::remove_all(staging, ec);
      return false;
    }
    drop_offline_handle_locked(staging_handle);

    offline_loaded_handle handle;
    const build_clock::time_point promote_start = build_clock::now();
    if (!promote_offline_store_locked(staging, &handle, *entry_count)) {
      std::filesystem::remove_all(staging, ec);
      return false;
    }
    add_build_load_ms_locked(promote_start);
    close_locked();
    m_offline_handle = handle.handle;
    m_offline_handle_variant = handle.variant;
    m_runtime_variant = handle.variant;
    return true;
  }

  bool rebuild_offline_raw_segments_single_locked(
      const raw_vector_segment_reader &reader, size_t *entry_count) {
    if (m_store_directory.empty() || entry_count == nullptr || !reader)
      return false;
    *entry_count = 0;

    const std::string suffix =
        ".offline-tmp-" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
    const std::filesystem::path staging(m_store_directory + suffix);
    std::error_code ec;
    std::filesystem::remove_all(staging, ec);
    if (ec) {
      set_build_fallback_reason_locked("remove_staging_failed");
      return false;
    }
    std::filesystem::create_directories(staging, ec);
    if (ec) {
      set_build_fallback_reason_locked("create_staging_failed");
      return false;
    }

    const std::string staging_prefix =
        offline_index_prefix_for_store(staging.string());
    if (!offline_build_raw_segments_locked(reader, staging_prefix,
                                           entry_count)) {
      if (m_last_build_diagnostics.fallback_reason.empty()) {
        set_build_fallback_reason_locked("offline_raw_segment_build_failed");
      }
      std::filesystem::remove_all(staging, ec);
      return false;
    }

    if (*entry_count == 0) {
      std::filesystem::remove_all(staging, ec);
      if (ec) return false;
      std::filesystem::remove_all(m_store_directory, ec);
      if (ec) return false;
      close_locked();
      m_runtime_variant = offline_runtime_variant_for_last_build_locked();
      return true;
    }

    const build_clock::time_point staging_load_start = build_clock::now();
    const offline_loaded_handle staging_handle =
        load_offline_index_locked(staging_prefix, *entry_count);
    add_build_load_ms_locked(staging_load_start);
    if (staging_handle.handle == nullptr) {
      set_build_fallback_reason_locked("staging_load_failed");
      std::filesystem::remove_all(staging, ec);
      return false;
    }
    drop_offline_handle_locked(staging_handle);

    offline_loaded_handle handle;
    const build_clock::time_point promote_start = build_clock::now();
    if (!promote_offline_store_locked(staging, &handle, *entry_count)) {
      set_build_fallback_reason_locked("promote_offline_store_failed");
      std::filesystem::remove_all(staging, ec);
      return false;
    }
    add_build_load_ms_locked(promote_start);
    close_locked();
    m_offline_handle = handle.handle;
    m_offline_handle_variant = handle.variant;
    m_runtime_variant = handle.variant;
    return true;
  }

  bool rebuild_offline_raw_segments_locked(
      const raw_vector_segment_reader &reader, size_t *entry_count) {
    begin_build_diagnostics_locked("raw_segments",
                                   kDiskAnnRuntimeOfficialCppMain);
    if (m_store_directory.empty() || entry_count == nullptr || !reader)
      return false;
    *entry_count = 0;

    std::vector<raw_vector_segment> segments;
    size_t total_rows = 0;
    const bool read_ok = reader([&](const raw_vector_segment &segment) {
      if (!validate_raw_segment_locked(segment) ||
          segment.row_count > std::numeric_limits<size_t>::max() - total_rows) {
        return false;
      }
      total_rows += segment.row_count;
      segments.push_back(segment);
      return true;
    });
    if (!read_ok) return false;
    *entry_count = total_rows;

    const auto collected_reader =
        [&segments](const raw_vector_segment_visitor &visitor) {
          if (!visitor) return false;
          for (const raw_vector_segment &segment : segments) {
            if (!visitor(segment)) return false;
          }
          return true;
        };
    const bool rebuild_ok =
        rebuild_offline_raw_segments_single_locked(collected_reader, entry_count);
    if (!rebuild_ok) close_locked();
    return rebuild_ok;
  }

  bool manifest_build_available_locked() const {
    DBUG_EXECUTE_IF("vector_backend_diskann_offline_unavailable",
                    return false;);
    return m_offline_api != nullptr && m_offline_api->available() &&
           ((m_vendored_api != nullptr &&
             m_vendored_api->structured_build_available()) ||
            m_offline_api->manifest_build_available());
  }

  bool build_from_manifest_locked(const std::string &index_prefix,
                                  const std::string &manifest_path,
                                  uint32_t dimension, int32_t metric_type,
                                  uint32_t max_degree,
                                  uint32_t build_complexity,
                                  uint32_t build_threads,
                                  double build_memory_gb, uint32_t pq_chunks,
                                  uint32_t cache_nodes,
                                  uint32_t build_blas_threads,
                                  uint32_t disk_pq_dims,
                                  bool accelerate_build,
                                  bool shuffle_build) {
    DBUG_EXECUTE_IF("vector_backend_diskann_fail_official_pq_build",
                    return false;);

    if (!diskann_force_offline_adapter() &&
        m_vendored_api != nullptr &&
        m_vendored_api->structured_build_available()) {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
      mysql_vector_diskann_build_config config{};
      config.data_path = manifest_path.c_str();
      config.index_prefix = index_prefix.c_str();
      config.dimension = dimension;
      config.metric_type = metric_type;
      config.max_degree = max_degree;
      config.build_complexity = build_complexity;
      config.build_threads = build_threads;
      config.build_blas_threads = build_blas_threads;
      config.build_memory_gb = build_memory_gb;
      config.pq_chunks = pq_chunks;
      config.cache_nodes = cache_nodes;
      config.search_cache_size = m_search_cache_size;
      config.search_cache_ratio = m_search_cache_ratio;
      config.disk_pq_dims = disk_pq_dims;
      config.use_bfs_cache = m_use_bfs_cache ? 1U : 0U;
      config.accelerate_build = accelerate_build ? 1U : 0U;
      config.shuffle_build = shuffle_build ? 1U : 0U;
      set_build_runtime_locked(kDiskAnnRuntimeVendored);
      m_last_build_diagnostics.native_pq_runtime_official_pq_used = true;
      std::string error;
      ++m_last_build_diagnostics.build_invocations;
      if (m_vendored_api->build_manifest(config, &error)) return true;
      set_build_fallback_reason_locked(
          error.empty() ? "vendored_build_failed" : error.c_str());
#endif
    }

    if (m_offline_api == nullptr ||
        !m_offline_api->manifest_build_available()) {
      return false;
    }
    if (accelerate_build || shuffle_build) {
      set_build_fallback_reason_locked(kDiskAnnUnsupportedBuildFlags);
      return false;
    }
    set_build_runtime_locked(kDiskAnnRuntimeOfficialCppMain);
    m_last_build_diagnostics.native_pq_runtime_official_pq_used = true;
    ++m_last_build_diagnostics.build_invocations;
    return m_offline_api->build_from_manifest(
        index_prefix.c_str(), manifest_path.c_str(), dimension, metric_type,
        max_degree, build_complexity, build_threads, build_memory_gb, pq_chunks,
        cache_nodes, build_blas_threads, disk_pq_dims,
        accelerate_build ? 1U : 0U, shuffle_build ? 1U : 0U);
  }

  bool remove_diskann_index_prefix_artifacts_locked(
      const std::string &index_prefix) const {
    if (index_prefix.empty()) return true;

    const std::filesystem::path prefix(index_prefix);
    const std::filesystem::path parent = prefix.parent_path();
    const std::string basename = prefix.filename().string();
    if (parent.empty() || basename.empty()) return false;

    std::error_code ec;
    if (!std::filesystem::exists(parent, ec)) return !ec;
    if (ec || !std::filesystem::is_directory(parent, ec)) return false;
    if (ec) return false;

    const std::string generated_prefix = basename + "_";
    const std::string raw_segments_dir = basename + "_raw_segments";
    for (const auto &entry : std::filesystem::directory_iterator(
             parent, std::filesystem::directory_options::skip_permission_denied,
             ec)) {
      if (ec) return false;
      const std::string filename = entry.path().filename().string();
      if (filename == raw_segments_dir ||
          !vector_index::detail::starts_with(filename, generated_prefix)) {
        continue;
      }
      std::filesystem::remove_all(entry.path(), ec);
      if (ec) return false;
    }
    return true;
  }

  static bool native_pq_bridge_build_callback(
      const char *index_prefix, const char *manifest_path,
      const char *pq_pivots_path, const char *pq_compressed_path,
      const char *disk_pq_pivots_path,
      const char *disk_pq_compressed_path,
      uint32_t dimension, int32_t metric_type, uint32_t max_degree,
      uint32_t build_complexity, uint32_t build_threads, double build_memory_gb,
      uint32_t pq_chunks, uint32_t cache_nodes, uint32_t blas_threads,
      uint32_t disk_pq_dims, bool accelerate_build, bool shuffle_build,
      char *error_buffer, size_t error_buffer_size, void *context) {
    auto *state = static_cast<diskann_native_state *>(context);
    if (state == nullptr) return false;
    return state->build_native_pq_bridge_locked(
        index_prefix, manifest_path, pq_pivots_path, pq_compressed_path,
        disk_pq_pivots_path, disk_pq_compressed_path, dimension, metric_type,
        max_degree, build_complexity, build_threads, build_memory_gb,
        pq_chunks, cache_nodes, blas_threads, disk_pq_dims, accelerate_build,
        shuffle_build, error_buffer, error_buffer_size);
  }

  bool build_native_pq_bridge_locked(
      const char *index_prefix, const char *manifest_path,
      const char *pq_pivots_path, const char *pq_compressed_path,
      const char *disk_pq_pivots_path,
      const char *disk_pq_compressed_path,
      uint32_t dimension, int32_t metric_type, uint32_t max_degree,
      uint32_t build_complexity, uint32_t build_threads, double build_memory_gb,
      uint32_t pq_chunks, uint32_t cache_nodes, uint32_t blas_threads,
      uint32_t disk_pq_dims, bool accelerate_build, bool shuffle_build,
      char *error_buffer, size_t error_buffer_size) {
    DBUG_EXECUTE_IF("vector_backend_diskann_native_pq_bridge_unavailable", {
      if (error_buffer != nullptr && error_buffer_size > 0) {
        std::strncpy(error_buffer, "native PQ DiskANN bridge is unavailable",
                     error_buffer_size - 1);
        error_buffer[error_buffer_size - 1] = '\0';
      }
      return false;
    });

    if (!diskann_force_offline_adapter() &&
        m_vendored_api != nullptr &&
        m_vendored_api->native_pq_bridge_available()) {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
      mysql_vector_diskann_build_config config{};
      config.data_path = manifest_path;
      config.index_prefix = index_prefix;
      config.dimension = dimension;
      config.metric_type = metric_type;
      config.max_degree = max_degree;
      config.build_complexity = build_complexity;
      config.build_threads = build_threads;
      config.build_blas_threads = blas_threads;
      config.build_memory_gb = build_memory_gb;
      config.pq_chunks = pq_chunks;
      config.cache_nodes = cache_nodes;
      config.search_cache_size = m_search_cache_size;
      config.search_cache_ratio = m_search_cache_ratio;
      config.disk_pq_dims = disk_pq_dims;
      config.use_bfs_cache = m_use_bfs_cache ? 1U : 0U;
      config.accelerate_build = accelerate_build ? 1U : 0U;
      config.shuffle_build = shuffle_build ? 1U : 0U;
      set_build_runtime_locked(kDiskAnnRuntimeVendored);
      ++m_last_build_diagnostics.build_invocations;
      return m_vendored_api->build_native_pq(
          config, pq_pivots_path, pq_compressed_path, disk_pq_pivots_path,
          disk_pq_compressed_path, error_buffer, error_buffer_size);
#endif
    }

    if (m_offline_api == nullptr ||
        !m_offline_api->native_pq_build_available()) {
      if (error_buffer != nullptr && error_buffer_size > 0) {
        std::strncpy(error_buffer, "native PQ DiskANN bridge is unavailable",
                     error_buffer_size - 1);
        error_buffer[error_buffer_size - 1] = '\0';
      }
      return false;
    }
    set_build_runtime_locked(kDiskAnnRuntimeOfficialCppMain);
    ++m_last_build_diagnostics.build_invocations;
    const bool ok = m_offline_api->build_from_native_pq(
        index_prefix, manifest_path, pq_pivots_path, pq_compressed_path,
        disk_pq_pivots_path, disk_pq_compressed_path, dimension, metric_type,
        max_degree, build_complexity, build_threads, build_memory_gb,
        pq_chunks, cache_nodes, blas_threads, disk_pq_dims,
        accelerate_build ? 1U : 0U, shuffle_build ? 1U : 0U);
    if (!ok && error_buffer != nullptr && error_buffer_size > 0) {
      const char *message = disk_pq_dims == 0
                                ? "offline adapter native PQ build failed"
                                : "native_pq_bridge_disk_pq_layout_failed";
      std::strncpy(error_buffer, message,
                   error_buffer_size - 1);
      error_buffer[error_buffer_size - 1] = '\0';
    }
    return ok;
  }

  bool build_with_native_pq_bridge_locked(
      const std::string &index_prefix, const std::string &manifest_path,
      uint32_t dimension, int32_t metric_type, uint32_t max_degree,
      uint32_t build_complexity, uint32_t build_threads,
      double build_memory_gb, uint32_t pq_chunks, uint32_t cache_nodes,
      uint64_t build_memory_size,
      uint32_t build_blas_threads, uint32_t disk_pq_dims,
      bool accelerate_build, bool shuffle_build, uint64_t row_count,
      bool *attempted) {
    if (attempted != nullptr) *attempted = false;
    if (!vector_index::diskann_pq_runtime_uses_native(
            vector_index::global_diskann_pq_runtime_mode())) {
      return false;
    }
    if (attempted != nullptr) *attempted = true;

    vector_index::diskann_pq_artifact_paths artifacts;
    vector_index::diskann_pq_artifact_paths disk_artifacts;
    uint32_t centroid_count = 0;
    bool memory_budget_exceeded = false;
    if (!build_native_pq_runtime_artifacts_locked(
            manifest_path, native_pq_artifact_prefix(index_prefix), pq_chunks,
            disk_pq_dims, build_threads, row_count, build_memory_size,
            &artifacts, &disk_artifacts, &centroid_count,
            &memory_budget_exceeded)) {
      if (memory_budget_exceeded) {
        set_build_fallback_reason_locked("native_pq_memory_budget_exceeded");
        return false;
      }
      set_build_fallback_reason_locked("native_pq_artifact_build_failed");
      return false;
    }

    if (!make_native_pq_bridge_artifact_paths_locked(index_prefix,
                                                     &artifacts)) {
      set_build_fallback_reason_locked("native_pq_runtime_path_unavailable");
      return false;
    }

    vector_index::diskann_graph_cache_bridge_config config;
    config.input_manifest_path = manifest_path;
    config.index_prefix = index_prefix;
    config.artifacts = artifacts;
    config.disk_artifacts = disk_artifacts;
    config.metric = m_metric;
    config.diskann_metric_type = metric_type;
    config.row_count = row_count;
    config.dimension = dimension;
    config.pq_chunks = pq_chunks;
    config.centroid_count = centroid_count;
    config.max_degree = max_degree;
    config.build_complexity = build_complexity;
    config.build_threads = build_threads;
    config.blas_threads = build_blas_threads;
    config.build_memory_gb = build_memory_gb;
    config.cache_nodes = cache_nodes;
    config.disk_pq_dims = disk_pq_dims;
    config.use_bfs_cache = m_use_bfs_cache;
    config.accelerate_build = accelerate_build;
    config.shuffle_build = shuffle_build;
    config.build_fn = diskann_native_state::native_pq_bridge_build_callback;
    config.build_context = this;

    vector_index::diskann_graph_cache_bridge_result bridge_result;
    std::string error;
    const bool ok = vector_index::build_diskann_graph_cache_from_native_pq(
        config, &bridge_result, &error);
    m_last_build_diagnostics.native_pq_runtime_artifacts_consumed =
        bridge_result.artifacts_consumed;
    m_last_build_diagnostics.native_pq_runtime_official_pq_used =
        bridge_result.official_pq_used;
    m_last_build_diagnostics.native_pq_runtime_bridge =
        bridge_result.bridge_name;
    m_last_build_diagnostics.native_pq_runtime_bridge_ms =
        bridge_result.elapsed_ms;
    m_last_build_diagnostics.native_pq_runtime_graph_ms =
        bridge_result.graph_ms;
    m_last_build_diagnostics.native_pq_runtime_cache_ms =
        bridge_result.cache_ms;
    m_last_build_diagnostics.native_pq_runtime_artifact_validation =
        ok ? "ok" : error;
    if (!ok) {
      set_build_fallback_reason_locked(
          error.empty() ? "native_pq_bridge_failed" : error.c_str());
      return false;
    }

    std::string validation_error;
    if (!validate_native_pq_serving_locked(index_prefix, manifest_path,
                                           row_count, &validation_error)) {
      m_last_build_diagnostics.native_pq_runtime_artifact_validation =
          validation_error.empty() ? "native_pq_serving_validation_failed"
                                   : validation_error;
      set_build_fallback_reason_locked(
          m_last_build_diagnostics.native_pq_runtime_artifact_validation
              .c_str());
      return false;
    }
    return true;
  }

  bool build_offline_manifest_locked(
      const std::string &index_prefix, const std::string &manifest_path,
      uint32_t dimension, int32_t metric_type, uint32_t max_degree,
      uint32_t build_complexity, uint32_t build_threads,
      double build_memory_gb, uint32_t pq_chunks, uint32_t cache_nodes,
      uint64_t build_memory_size,
      uint32_t build_blas_threads, uint32_t disk_pq_dims,
      bool accelerate_build, bool shuffle_build, uint64_t row_count) {
    if (accelerate_build || shuffle_build) {
      set_build_fallback_reason_locked(kDiskAnnUnsupportedBuildFlags);
      accelerate_build = false;
      shuffle_build = false;
    }

    bool native_pq_attempted = false;
    if (build_with_native_pq_bridge_locked(
            index_prefix, manifest_path, dimension, metric_type, max_degree,
            build_complexity, build_threads, build_memory_gb, pq_chunks,
            cache_nodes, build_memory_size, build_blas_threads, disk_pq_dims,
            accelerate_build, shuffle_build, row_count,
            &native_pq_attempted)) {
      return true;
    }
    const vector_index::diskann_pq_runtime_mode pq_runtime_mode =
        vector_index::global_diskann_pq_runtime_mode();
    if (native_pq_attempted) {
      if (!remove_diskann_index_prefix_artifacts_locked(index_prefix)) {
        set_build_fallback_reason_locked(
            "native_pq_artifact_cleanup_failed");
        return false;
      }
      if (!vector_index::diskann_pq_runtime_allows_official_fallback(
              pq_runtime_mode)) {
        return false;
      }
      m_last_build_diagnostics.native_pq_runtime_official_pq_used = true;
    }
    return build_from_manifest_locked(
        index_prefix, manifest_path, dimension, metric_type, max_degree,
        build_complexity, build_threads, build_memory_gb, pq_chunks,
        cache_nodes, build_blas_threads, disk_pq_dims, accelerate_build,
        shuffle_build);
  }

  diskann_runtime_variant offline_runtime_variant_for_last_build_locked() const {
    return m_last_build_diagnostics.runtime == kDiskAnnRuntimeVendored
               ? diskann_runtime_variant::kVendoredOffline
               : diskann_runtime_variant::kOffline;
  }

  bool offline_build_available_locked() const {
    DBUG_EXECUTE_IF("vector_backend_diskann_offline_unavailable",
                    return false;);
    return manifest_build_available_locked();
  }

  bool rebuild_ordered_entries_locked(const ordered_entries &entries,
                                      diskann_build_mode build_mode) {
    switch (build_mode) {
      case diskann_build_mode::kAuto:
        if (!serial_insert_entries_locked(entries)) return false;
        m_runtime_variant = diskann_runtime_variant::kSerial;
        return true;
      case diskann_build_mode::kSerial:
        if (!serial_insert_entries_locked(entries)) return false;
        m_runtime_variant = diskann_runtime_variant::kSerial;
        return true;
      default:
        return false;
    }
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
    if (m_offline_handle != nullptr) {
      drop_offline_handle_locked({m_offline_handle, m_offline_handle_variant});
      m_offline_handle = nullptr;
      m_offline_handle_variant = diskann_runtime_variant::kNone;
      m_loaded_cache_nodes = 0;
    }
    if (m_index_handle != nullptr && supported()) {
      m_api->drop_index(callback_context(), m_index_handle);
      m_index_handle = nullptr;
    }
    m_runtime_variant = diskann_runtime_variant::kNone;
  }

  std::string offline_index_prefix() const {
    return (std::filesystem::path(m_store_directory) / "offline" / "diskann")
        .string();
  }

  std::string native_pq_artifact_prefix(
      const std::string &index_prefix) const {
    return index_prefix + "_native_pq_artifacts";
  }

  bool make_native_pq_bridge_artifact_paths_locked(
      const std::string &index_prefix,
      vector_index::diskann_pq_artifact_paths *artifacts) const {
    vector_index::diskann_pq_artifact_paths runtime_artifacts;
    if (artifacts == nullptr ||
        !vector_index::make_diskann_pq_artifact_paths(
            native_pq_artifact_prefix(index_prefix), artifacts) ||
        !vector_index::make_diskann_pq_artifact_paths(index_prefix,
                                                      &runtime_artifacts)) {
      return false;
    }
    artifacts->disk_index_path = runtime_artifacts.disk_index_path;
    artifacts->disk_index_medoids_path =
        runtime_artifacts.disk_index_medoids_path;
    artifacts->disk_index_centroids_path =
        runtime_artifacts.disk_index_centroids_path;
    artifacts->disk_index_pq_pivots_path =
        runtime_artifacts.disk_index_pq_pivots_path;
    artifacts->sample_data_path = runtime_artifacts.sample_data_path;
    artifacts->cached_nodes_path = runtime_artifacts.cached_nodes_path;
    return true;
  }

  std::string offline_index_prefix_for_store(
      const std::string &store_directory) const {
    return (std::filesystem::path(store_directory) / "offline" / "diskann")
        .string();
  }

  uint32_t effective_cache_nodes_for_row_count(size_t row_count) const {
    return diskann_cache_nodes_for_build(row_count, m_dimension, m_cache_nodes,
                                         m_search_cache_size,
                                         m_search_cache_ratio, m_max_degree,
                                         m_disk_pq_dims);
  }

  uint32_t effective_loaded_cache_nodes_locked() const {
    return m_loaded_cache_nodes != 0 ? m_loaded_cache_nodes : m_cache_nodes;
  }

  bool make_segment_budget_locked(
      size_t row_count, vector_index::diskann_segment_budget *budget) const {
    if (budget == nullptr || row_count == 0 ||
        m_dimension > std::numeric_limits<uint32_t>::max()) {
      return false;
    }

    const size_t row_bytes = saturated_mul_size(m_dimension, sizeof(float));
    if (row_bytes == 0 || row_bytes == std::numeric_limits<size_t>::max() ||
        row_count > std::numeric_limits<size_t>::max() / row_bytes) {
      return false;
    }

    vector_index::diskann_segment_budget_input input;
    input.dimension = static_cast<uint32_t>(m_dimension);
    input.row_count = row_count;
    input.payload_size = row_count * row_bytes;
    input.pq_code_budget_size = m_pq_code_budget_size;
    input.pq_code_budget_ratio = m_pq_code_budget_ratio;
    input.disk_pq_dims = m_disk_pq_dims;
    input.max_degree = m_max_degree;
    input.search_cache_size = m_search_cache_size;
    input.search_cache_ratio = m_search_cache_ratio;
    input.build_memory_size = opt_vector_diskann_build_memory_size;
    input.available_build_memory_size = available_diskann_build_memory_size();
    return vector_index::make_diskann_segment_budget(input, budget);
  }

  void record_segment_budget_locked(
      const vector_index::diskann_segment_budget &budget) {
    m_last_build_diagnostics.pq_chunks = budget.pq_chunks;
    m_last_build_diagnostics.cache_nodes = budget.cache_nodes;
    m_last_build_diagnostics.requested_disk_pq_dims = m_disk_pq_dims;
    m_last_build_diagnostics.effective_disk_pq_dims = budget.disk_pq_dims;
  }

#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
  mysql_vector_diskann_search_config make_vendored_search_config_locked(
      uint32_t top_k, uint32_t search_threads, uint32_t cache_nodes) const {
    mysql_vector_diskann_search_config config{};
    config.top_k = top_k;
    config.search_complexity = std::max(m_search_complexity, top_k);
    config.beamwidth = m_search_beamwidth;
    config.batch_search_threads = search_threads;
    config.search_io_limit = m_search_io_limit;
    config.cache_nodes = cache_nodes;
    return config;
  }

  mysql_vector_diskann_build_config make_vendored_load_config_locked(
      const std::string &index_prefix, size_t row_count) const {
    vector_index::diskann_segment_budget budget;
    const bool has_budget = make_segment_budget_locked(row_count, &budget);
    mysql_vector_diskann_build_config config{};
    config.index_prefix = index_prefix.c_str();
    config.dimension = static_cast<uint32_t>(m_dimension);
    config.metric_type = diskann_metric_code(m_metric);
    config.max_degree = m_max_degree;
    config.build_complexity = m_build_complexity;
    config.build_threads = m_build_threads;
    config.build_blas_threads = m_build_blas_threads;
    config.build_memory_gb =
        has_budget ? budget.build_memory_gb
                   : diskann_build_memory_size_gb(
                         opt_vector_diskann_build_memory_size);
    config.pq_chunks = has_budget ? budget.pq_chunks : 1;
    config.cache_nodes = has_budget ? budget.cache_nodes : m_cache_nodes;
    config.search_cache_size = m_search_cache_size;
    config.search_cache_ratio = m_search_cache_ratio;
    config.disk_pq_dims = has_budget ? budget.disk_pq_dims : m_disk_pq_dims;
    config.use_bfs_cache = m_use_bfs_cache ? 1U : 0U;
    config.accelerate_build = m_accelerate_build ? 1U : 0U;
    config.shuffle_build = m_shuffle_build ? 1U : 0U;
    return config;
  }
#endif

  offline_loaded_handle load_offline_index_locked(
      const std::string &index_prefix, size_t row_count) const {
    /*
      DiskANN sample-query cache generation is asynchronous in the native
      library. Use the synchronous BFS cache path whenever the user explicitly
      enables node caching so the cache is populated before serving queries.
    */
    const uint32_t cache_nodes = effective_cache_nodes_for_row_count(row_count);
    if (!diskann_force_offline_adapter() &&
        m_vendored_api != nullptr &&
        m_vendored_api->runtime_handle_available()) {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
      mysql_vector_diskann_build_config build_config =
          make_vendored_load_config_locked(index_prefix, row_count);
      mysql_vector_diskann_search_config search_config =
          make_vendored_search_config_locked(1, m_offline_search_threads,
                                             build_config.cache_nodes);
      std::string error;
      const void *handle =
          m_vendored_api->load(build_config, search_config, &error);
      if (handle != nullptr) {
        m_loaded_cache_nodes = build_config.cache_nodes;
        return {handle, diskann_runtime_variant::kVendoredOffline};
      }
#endif
    }

    const void *handle = m_offline_api->load_index(
        index_prefix.c_str(), diskann_metric_code(m_metric),
        m_offline_search_threads, m_search_io_limit, cache_nodes,
        diskann_offline_load_use_bfs_cache(cache_nodes, m_use_bfs_cache));
    if (handle != nullptr) m_loaded_cache_nodes = cache_nodes;
    return {handle, diskann_runtime_variant::kOffline};
  }

  bool promote_offline_store_locked(const std::filesystem::path &staging,
                                    offline_loaded_handle *loaded_handle,
                                    size_t row_count) const {
    if (loaded_handle == nullptr) return false;
    *loaded_handle = {};

    std::error_code ec;
    const std::filesystem::path store(m_store_directory);
    const std::string suffix =
        ".offline-old-" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
    const std::filesystem::path backup(m_store_directory + suffix);

    std::filesystem::remove_all(backup, ec);
    if (ec) return false;

    const bool has_existing_store = std::filesystem::exists(store, ec);
    if (ec) return false;
    if (has_existing_store) {
      std::filesystem::rename(store, backup, ec);
      if (ec) return false;
    }

    std::filesystem::rename(staging, store, ec);
    if (ec) {
      std::error_code restore_ec;
      if (has_existing_store) std::filesystem::rename(backup, store, restore_ec);
      return false;
    }

    const offline_loaded_handle handle =
        load_offline_index_locked(offline_index_prefix(), row_count);
    if (handle.handle == nullptr) {
      const std::filesystem::path failed(
          m_store_directory + ".offline-failed-" +
          std::to_string(reinterpret_cast<std::uintptr_t>(this)));
      std::error_code restore_ec;
      std::filesystem::remove_all(failed, restore_ec);
      restore_ec.clear();
      std::filesystem::rename(store, failed, restore_ec);
      if (has_existing_store) {
        restore_ec.clear();
        std::filesystem::rename(backup, store, restore_ec);
      }
      restore_ec.clear();
      std::filesystem::remove_all(failed, restore_ec);
      return false;
    }

    std::filesystem::remove_all(backup, ec);
    *loaded_handle = handle;
    return true;
  }

  void drop_offline_handle_locked(offline_loaded_handle handle) const {
    if (handle.handle == nullptr) return;
    if (handle.variant == diskann_runtime_variant::kVendoredOffline) {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
      if (m_vendored_api != nullptr) {
        m_vendored_api->drop_handle(handle.handle);
        return;
      }
#endif
    }
    if (m_offline_api != nullptr && m_offline_api->available()) {
      m_offline_api->drop_index(handle.handle);
    }
  }

  int32_t search_loaded_handle_locked(offline_loaded_handle handle,
                                      const vector_data &query, size_t top_k,
                                      uint64_t *doc_ids,
                                      float *distances) const {
    if (handle.handle == nullptr ||
        top_k > std::numeric_limits<uint32_t>::max()) {
      return -1;
    }
    const uint32_t top_k_uint32 = static_cast<uint32_t>(top_k);
    const uint32_t effective_search_complexity =
        std::max(m_search_complexity, top_k_uint32);
    if (handle.variant == diskann_runtime_variant::kVendoredOffline) {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
      const mysql_vector_diskann_search_config search_config =
          make_vendored_search_config_locked(top_k_uint32,
                                             m_offline_search_threads,
                                             effective_loaded_cache_nodes_locked());
      return m_vendored_api->search_one(handle.handle, query.data(),
                                        query.size(), search_config, doc_ids,
                                        distances);
#else
      return -1;
#endif
    }
    return m_offline_api->search(
        handle.handle, query.data(), query.size(), top_k_uint32,
        effective_search_complexity, m_search_beamwidth, doc_ids, distances);
  }

  int32_t search_loaded_handle_batch_locked(
      offline_loaded_handle handle,
      const float *query_data, size_t query_count, size_t top_k,
      uint32_t search_threads, uint64_t *doc_ids, float *distances,
      uint32_t *result_counts) const {
    if (handle.handle == nullptr ||
        top_k > std::numeric_limits<uint32_t>::max()) {
      return -1;
    }
    const uint32_t top_k_uint32 = static_cast<uint32_t>(top_k);
    const uint32_t effective_search_complexity =
        std::max(m_search_complexity, top_k_uint32);
    if (handle.variant == diskann_runtime_variant::kVendoredOffline) {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
      const mysql_vector_diskann_search_config search_config =
          make_vendored_search_config_locked(top_k_uint32,
                                             search_threads,
                                             effective_loaded_cache_nodes_locked());
      return m_vendored_api->search_many(
          handle.handle, query_data, query_count, m_dimension, search_config,
          doc_ids, distances, result_counts);
#else
      return -1;
#endif
    }
    return m_offline_api->search_batch(
        handle.handle, query_data, query_count, m_dimension, top_k_uint32,
        effective_search_complexity, m_search_beamwidth, search_threads,
        doc_ids, distances, result_counts);
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

  bool flush_build_memory_store_locked(bool include_doc_ids_file = false) {
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
    if (include_doc_ids_file && !move_doc_ids_file_to_store_locked(staging)) {
      std::filesystem::remove_all(staging, ec);
      return false;
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
  const diskann_offline_api *m_offline_api{nullptr};
  const diskann_vendored_runtime_api *m_vendored_api{nullptr};
  const void *m_index_handle{nullptr};
  const void *m_offline_handle{nullptr};
  diskann_runtime_variant m_offline_handle_variant{diskann_runtime_variant::kNone};
  uint32_t m_build_complexity{kDiskAnnBuildComplexity};
  uint32_t m_max_degree{kDiskAnnMaxDegree};
  uint32_t m_build_threads{0};
  uint32_t m_build_blas_threads{1};
  uint32_t m_search_complexity{kDiskAnnBuildComplexity};
  uint32_t m_offline_search_threads{
      vector_index::k_default_diskann_offline_search_threads};
  uint32_t m_search_io_limit{0};
  uint32_t m_cache_nodes{0};
  uint32_t m_search_beamwidth{
      vector_index::k_default_diskann_search_beamwidth};
  uint64_t m_pq_code_budget_size{0};
  double m_pq_code_budget_ratio{
      vector_index::k_default_diskann_pq_code_budget_ratio};
  uint32_t m_disk_pq_dims{0};
  bool m_accelerate_build{false};
  bool m_shuffle_build{false};
  bool m_use_bfs_cache{false};
  uint64_t m_search_cache_size{0};
  double m_search_cache_ratio{0.0};
  mutable uint32_t m_loaded_cache_nodes{0};
  diskann_runtime_variant m_runtime_variant{diskann_runtime_variant::kNone};
  backend_build_diagnostics m_last_build_diagnostics;
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
                         external_sidecar_profile::kDiskAnn),
      m_diskann_max_degree(
          static_cast<uint32_t>(opt_vector_diskann_max_degree)),
      m_diskann_build_complexity(
          static_cast<uint32_t>(opt_vector_diskann_build_complexity)),
      m_diskann_build_blas_threads(
          static_cast<uint32_t>(opt_vector_diskann_build_blas_threads)),
      m_diskann_search_complexity(
          static_cast<uint32_t>(opt_vector_diskann_search_complexity)),
      m_diskann_search_beamwidth(
          static_cast<uint32_t>(opt_vector_diskann_search_beamwidth)),
      m_diskann_pq_code_budget_size(opt_vector_diskann_pq_code_budget_size),
      m_diskann_pq_code_budget_ratio(opt_vector_diskann_pq_code_budget_ratio),
      m_diskann_disk_pq_dims(
          static_cast<uint32_t>(opt_vector_diskann_disk_pq_dims)),
      m_diskann_accelerate_build(opt_vector_diskann_accelerate_build),
      m_diskann_shuffle_build(opt_vector_diskann_shuffle_build),
      m_diskann_use_bfs_cache(opt_vector_diskann_use_bfs_cache),
      m_diskann_offline_search_threads(
          static_cast<uint32_t>(opt_vector_diskann_offline_search_threads)),
      m_diskann_search_io_limit(
          static_cast<uint32_t>(opt_vector_diskann_search_io_limit)),
      m_diskann_cache_nodes(
          static_cast<uint32_t>(opt_vector_diskann_cache_nodes)),
      m_diskann_search_cache_size(opt_vector_diskann_search_cache_size),
      m_diskann_search_cache_ratio(opt_vector_diskann_search_cache_ratio) {}

diskann_backend::~diskann_backend() = default;

void diskann_backend::capture_native_build_diagnostics() {
  m_last_build_diagnostics =
      m_native_state != nullptr ? m_native_state->build_diagnostics()
                                : backend_build_diagnostics{};
}

void diskann_backend::clear_build_diagnostics() {
  m_last_build_diagnostics = {};
}

std::unique_ptr<diskann_native_state> diskann_backend::create_native_state()
    const {
  return std::make_unique<diskann_native_state>(
      m_dimension, m_metric, m_index_name, m_diskann_build_complexity,
      m_diskann_max_degree, m_diskann_build_threads,
      m_diskann_build_blas_threads, m_diskann_offline_search_threads,
      m_diskann_search_io_limit, m_diskann_cache_nodes,
      m_diskann_search_beamwidth, m_diskann_pq_code_budget_size,
      m_diskann_search_cache_size, m_diskann_pq_code_budget_ratio,
      m_diskann_search_cache_ratio, m_diskann_disk_pq_dims,
      m_diskann_accelerate_build, m_diskann_shuffle_build,
      m_diskann_use_bfs_cache);
}

bool diskann_backend::configure_native_state_for_serving(
    diskann_native_state *state) {
  if (state == nullptr) return false;

  const auto fail_with_diagnostics = [&](const char *reason) {
    m_last_build_diagnostics = state->build_diagnostics();
    m_last_build_diagnostics.fallback_reason = reason;
    return false;
  };
  bool search_complexity_ok =
      state->set_search_complexity(m_diskann_search_complexity);
  DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_set_search_complexity",
                  search_complexity_ok = false;);
  if (!search_complexity_ok) {
    return fail_with_diagnostics("set_search_complexity_failed");
  }
  bool search_beamwidth_ok =
      state->set_search_beamwidth(m_diskann_search_beamwidth);
  DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_set_search_beamwidth",
                  search_beamwidth_ok = false;);
  if (!search_beamwidth_ok) {
    return fail_with_diagnostics("set_search_beamwidth_failed");
  }
  if (!remove_diskann_external_artifacts(m_index_name, false)) {
    return fail_with_diagnostics("remove_diskann_artifacts_failed");
  }
  return true;
}

void diskann_backend::install_native_state(
    std::unique_ptr<diskann_native_state> state, size_t entry_count,
    bool entries_complete) {
  m_entry_count = entry_count;
  m_entries_complete = entries_complete;
  reset_streaming_doc_id_tracking();
  m_external_adapter_active = false;
  m_native_state = std::move(state);
  capture_native_build_diagnostics();
  m_native_runtime_enabled = true;
}

bool diskann_backend::activate_empty_mutable_sidecar() {
  if (!remove_diskann_external_artifacts(m_index_name, true)) return false;
  return load_committed_entries({});
}

void diskann_backend::reset_streaming_doc_id_tracking() {
  m_streaming_doc_ids_loaded = false;
  m_streaming_base_doc_ids.clear();
  m_streaming_added_doc_ids.clear();
  m_streaming_removed_doc_ids.clear();
}

bool diskann_backend::prepare_streaming_doc_id_tracking() {
  if (m_entries_complete || m_streaming_doc_ids_loaded) return true;
  if (!m_native_runtime_enabled || m_native_state == nullptr) return false;

  std::vector<uint64_t> doc_ids;
  if (!m_native_state->collect_doc_ids(&doc_ids)) return false;
  std::sort(doc_ids.begin(), doc_ids.end());
  if (doc_ids.size() != m_entry_count ||
      std::adjacent_find(doc_ids.begin(), doc_ids.end()) != doc_ids.end()) {
    return false;
  }
  m_streaming_base_doc_ids = std::move(doc_ids);
  m_streaming_doc_ids_loaded = true;
  return true;
}

bool diskann_backend::streaming_doc_id_exists(uint64_t doc_id) const {
  if (m_streaming_removed_doc_ids.find(doc_id) !=
      m_streaming_removed_doc_ids.end()) {
    return false;
  }
  if (m_streaming_added_doc_ids.find(doc_id) !=
      m_streaming_added_doc_ids.end()) {
    return true;
  }
  return std::binary_search(m_streaming_base_doc_ids.begin(),
                            m_streaming_base_doc_ids.end(), doc_id);
}

void diskann_backend::record_streaming_upsert(uint64_t doc_id, bool existed) {
  const bool existed_in_base = std::binary_search(
      m_streaming_base_doc_ids.begin(), m_streaming_base_doc_ids.end(), doc_id);
  m_streaming_removed_doc_ids.erase(doc_id);
  if (!existed_in_base) m_streaming_added_doc_ids.insert(doc_id);
  if (!existed) ++m_entry_count;
}

void diskann_backend::record_streaming_erase(uint64_t doc_id) {
  const bool existed_in_base = std::binary_search(
      m_streaming_base_doc_ids.begin(), m_streaming_base_doc_ids.end(), doc_id);
  if (m_streaming_added_doc_ids.erase(doc_id) == 0 && existed_in_base) {
    m_streaming_removed_doc_ids.insert(doc_id);
  }
  --m_entry_count;
}

bool diskann_backend::upsert(uint64_t doc_id, const vector_data &vector) {
  if (m_mode != backend_mode::kExternal) return false;
  if (vector.size() != m_dimension) return false;
  bool existed = false;
  if (m_entries_complete) {
    existed = m_entries.find(doc_id) != m_entries.end();
  } else {
    if (!prepare_streaming_doc_id_tracking()) return false;
    existed = streaming_doc_id_exists(doc_id);
    if (!existed && m_entry_count == std::numeric_limits<size_t>::max()) {
      return false;
    }
  }
  if (m_external_adapter_active && !diskann_sidecar_supports_doc_id(doc_id)) {
    if (!remove_diskann_external_artifacts(m_index_name, false)) return false;
    m_external_adapter_active = false;
  }
  if (m_external_adapter_active &&
      !m_external_adapter.upsert(doc_id, vector))
    return false;
  if (m_native_runtime_enabled && m_native_state != nullptr &&
      !m_native_state->supports_mutations()) {
    return false;
  }
  bool native_insert_ok = true;
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    native_insert_ok = m_native_state->insert(doc_id, vector);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_insert",
                    native_insert_ok = false;);
  }
  if (m_native_runtime_enabled && m_native_state != nullptr && !native_insert_ok) {
    if (!m_entries_complete) {
      m_native_runtime_enabled = false;
      m_native_state.reset();
      return false;
    }
    m_native_runtime_enabled = false;
    m_native_state.reset();
  }
  if (m_entries_complete) {
    m_entries[doc_id] = vector;
    m_entry_count = m_entries.size();
  } else {
    record_streaming_upsert(doc_id, existed);
  }
  return true;
}

bool diskann_backend::erase(uint64_t doc_id) {
  if (m_mode != backend_mode::kExternal) return false;
  bool existed = false;
  if (m_entries_complete) {
    existed = m_entries.find(doc_id) != m_entries.end();
  } else {
    if (!prepare_streaming_doc_id_tracking()) return false;
    existed = streaming_doc_id_exists(doc_id);
  }
  if (!existed) return true;
  if (m_external_adapter_active && !m_external_adapter.erase(doc_id))
    return false;
  if (m_native_runtime_enabled && m_native_state != nullptr &&
      !m_native_state->supports_mutations()) {
    return false;
  }
  bool native_remove_ok = true;
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    native_remove_ok = m_native_state->remove(doc_id);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_remove",
                    native_remove_ok = false;);
  }
  if (m_native_runtime_enabled && m_native_state != nullptr && !native_remove_ok) {
    if (!m_entries_complete) {
      m_native_runtime_enabled = false;
      m_native_state.reset();
      return false;
    }
    m_native_runtime_enabled = false;
    m_native_state.reset();
  }
  if (m_entries_complete) {
    m_entries.erase(doc_id);
    m_entry_count = m_entries.size();
  } else {
    record_streaming_erase(doc_id);
  }
  return true;
}

bool diskann_backend::search(const vector_data &query, size_t top_k,
                            std::vector<search_result> *results) const {
  if (results == nullptr || query.size() != m_dimension) return false;
  results->clear();
  if (top_k == 0) return true;
  if (m_entry_count == 0) return true;
  top_k = std::min(top_k, m_entry_count);
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    bool native_search_ok = m_native_state->search(query, top_k, results);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_search",
                    native_search_ok = false; results->clear(););
    if (native_search_ok) {
      if (!m_entries_complete) return true;
      std::vector<search_result> filtered;
      filtered.reserve(results->size());
      for (const search_result &result : *results) {
        if (m_entries.find(result.doc_id) == m_entries.end()) continue;
        const bool duplicate =
            std::find_if(filtered.begin(), filtered.end(),
                         [&](const search_result &kept) {
                           return kept.doc_id == result.doc_id;
                         }) != filtered.end();
        if (!duplicate) filtered.push_back(result);
      }
      results->swap(filtered);
      const size_t expected_results = std::min(top_k, m_entries.size());
      if (results->size() >= expected_results) {
        return true;
      }

      // Native DiskANN reopen/rebuild can transiently yield incomplete hits.
      // The committed-entry snapshot below keeps SQL-visible search stable.
    } else {
      results->clear();
    }
  }
  results->clear();
  if (m_entries_complete && detail::search_exact_entries(
                                m_entries, m_metric, query, top_k, results)) {
    return true;
  }
  return m_external_adapter_active &&
         m_external_adapter.search(query, top_k, results);
}

bool diskann_backend::search_batch(
    const std::vector<vector_data> &queries, size_t top_k,
    std::vector<std::vector<search_result>> *results) const {
  if (results == nullptr) return false;
  results->clear();
  results->resize(queries.size());

  for (const vector_data &query : queries) {
    if (query.size() != m_dimension) return false;
  }
  if (queries.empty() || top_k == 0) return true;
  if (m_entry_count == 0) return true;
  top_k = std::min(top_k, m_entry_count);

  const size_t thread_count =
      vector_index::effective_diskann_search_threads(queries.size());
  if (m_native_runtime_enabled && m_native_state != nullptr &&
      !m_entries_complete) {
    bool native_batch_search_ok =
        m_native_state->search_batch(queries, 0, queries.size(), top_k,
                                     static_cast<uint32_t>(thread_count),
                                     results);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_batch_search",
                    native_batch_search_ok = false;);
    if (native_batch_search_ok) return true;
    results->clear();
    results->resize(queries.size());
  }

  if (thread_count <= 1 || queries.size() <= 1) {
    for (size_t i = 0; i < queries.size(); ++i) {
      if (!search(queries[i], top_k, &(*results)[i])) return false;
    }
    return true;
  }

  return vector_index::parallel_for_queries(
      queries.size(), thread_count, [&](size_t begin, size_t end, size_t) {
        for (size_t i = begin; i < end; ++i) {
          if (!search(queries[i], top_k, &(*results)[i])) {
            return false;
          }
        }
        return true;
      });
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
  m_entry_count = entries.size();
  m_entries_complete = true;
  reset_streaming_doc_id_tracking();
  m_external_adapter_active = sidecar_compatible;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  clear_build_diagnostics();
  return true;
}

bool diskann_backend::rebuild_from_committed_entries_from_reader(
    const committed_entry_reader &reader) {
  if (m_mode != backend_mode::kExternal) return false;
  auto rebuilt = create_native_state();

  size_t rebuilt_entry_count = 0;
  bool materialized_fallback_allowed = false;
  bool rebuild_ok = rebuilt->rebuild_from_reader_offline(
      reader, m_diskann_build_mode, &rebuilt_entry_count,
      &materialized_fallback_allowed);
  DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_rebuild",
                  rebuild_ok = false; materialized_fallback_allowed = false;);
  if (!rebuild_ok) {
    if (!materialized_fallback_allowed) {
      m_last_build_diagnostics = rebuilt->build_diagnostics();
      return false;
    }
    return backend::rebuild_from_committed_entries_from_reader(reader);
  }

  if (rebuilt_entry_count == 0) {
    rebuilt.reset();
    return activate_empty_mutable_sidecar();
  }

  if (!configure_native_state_for_serving(rebuilt.get())) return false;
  m_entries.clear();
  install_native_state(std::move(rebuilt), rebuilt_entry_count, false);
  return true;
}

bool diskann_backend::rebuild_from_raw_segments(
    const raw_vector_segment_reader &reader) {
  if (m_mode != backend_mode::kExternal) return false;
  auto rebuilt = create_native_state();

  size_t rebuilt_entry_count = 0;
  bool materialized_fallback_allowed = false;
  bool rebuild_ok = rebuilt->rebuild_from_raw_segments(
      reader, m_diskann_build_mode, &rebuilt_entry_count,
      &materialized_fallback_allowed);
  DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_rebuild",
                  rebuild_ok = false; materialized_fallback_allowed = false;);
  if (!rebuild_ok) {
    if (!materialized_fallback_allowed) {
      m_last_build_diagnostics = rebuilt->build_diagnostics();
      return false;
    }
    return backend::rebuild_from_raw_segments(reader);
  }

  if (rebuilt_entry_count == 0) {
    rebuilt.reset();
    return activate_empty_mutable_sidecar();
  }

  if (!configure_native_state_for_serving(rebuilt.get())) return false;
  m_entries.clear();
  install_native_state(std::move(rebuilt), rebuilt_entry_count, false);
  return true;
}

bool diskann_backend::rebuild_from_committed_entries(
    const std::unordered_map<uint64_t, vector_data> &entries) {
  if (m_mode != backend_mode::kExternal) return false;
  if (entries.empty()) return activate_empty_mutable_sidecar();
  auto rebuilt = create_native_state();
  bool rebuild_ok = rebuilt->rebuild_from_entries(entries, m_diskann_build_mode);
  DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_rebuild",
                  rebuild_ok = false;);
  if (!rebuild_ok) {
    if (m_diskann_build_mode != diskann_build_mode::kAuto) {
      m_last_build_diagnostics = rebuilt->build_diagnostics();
      return false;
    }
    if (rebuilt->supported()) {
      m_last_build_diagnostics = rebuilt->build_diagnostics();
      return false;
    }
    const bool sidecar_compatible = diskann_sidecar_supports_entries(entries);
    if (sidecar_compatible) {
      if (!m_external_adapter.load_committed_entries(entries)) return false;
    } else if (!diskann_entries_match_dimension(entries, m_dimension)) {
      return false;
    }
    (void)rebuilt->set_search_complexity(m_diskann_search_complexity);
    m_entries = entries;
    m_entry_count = entries.size();
    m_entries_complete = true;
    reset_streaming_doc_id_tracking();
    m_external_adapter_active = sidecar_compatible;
    m_native_state = std::move(rebuilt);
    capture_native_build_diagnostics();
    m_native_runtime_enabled = false;
    return true;
  }
  if (!configure_native_state_for_serving(rebuilt.get())) return false;
  m_entries = entries;
  install_native_state(std::move(rebuilt), entries.size(), true);
  return true;
}

bool diskann_backend::recover() {
  if (m_mode != backend_mode::kExternal) return false;
  if (!m_external_adapter.recover()) return false;
  m_entries = m_external_adapter.external_snapshot_entries();
  m_entry_count = m_entries.size();
  m_entries_complete = true;
  reset_streaming_doc_id_tracking();
  m_external_adapter_active = true;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  clear_build_diagnostics();
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
  return m_entry_count;
}

bool diskann_backend::collect_doc_ids(std::vector<uint64_t> *doc_ids) const {
  if (doc_ids == nullptr) return false;
  doc_ids->clear();
  if (m_entries_complete) {
    doc_ids->reserve(m_entries.size());
    for (const auto &entry : m_entries) {
      doc_ids->push_back(entry.first);
    }
    return true;
  }
  if (!m_native_runtime_enabled || m_native_state == nullptr ||
      !m_native_state->collect_doc_ids(doc_ids)) {
    return false;
  }
  doc_ids->erase(
      std::remove_if(doc_ids->begin(), doc_ids->end(),
                     [&](uint64_t doc_id) {
                       return m_streaming_removed_doc_ids.find(doc_id) !=
                              m_streaming_removed_doc_ids.end();
                     }),
      doc_ids->end());
  doc_ids->insert(doc_ids->end(), m_streaming_added_doc_ids.begin(),
                  m_streaming_added_doc_ids.end());
  return doc_ids->size() == m_entry_count;
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
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    return m_native_state->backend_variant();
  }
  if (m_external_adapter_active) return "diskann_fallback";
  return "diskann_unloaded";
}

backend_build_diagnostics diskann_backend::build_diagnostics() const {
  return m_last_build_diagnostics;
}

bool diskann_backend::set_diskann_build_params(uint32_t diskann_max_degree,
                                               uint32_t diskann_build_complexity,
                                               uint32_t diskann_build_threads) {
  if (m_mode != backend_mode::kExternal || diskann_max_degree == 0 ||
      diskann_build_complexity == 0 ||
      diskann_build_threads > vector_index::k_max_build_threads) {
    return false;
  }
  if (m_diskann_max_degree == diskann_max_degree &&
      m_diskann_build_complexity == diskann_build_complexity &&
      m_diskann_build_threads == diskann_build_threads) {
    return true;
  }
  m_diskann_max_degree = diskann_max_degree;
  m_diskann_build_complexity = diskann_build_complexity;
  m_diskann_build_threads = diskann_build_threads;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  clear_build_diagnostics();
  return true;
}

bool diskann_backend::set_diskann_build_threads(
    uint32_t diskann_build_threads) {
  if (m_mode != backend_mode::kExternal ||
      diskann_build_threads > vector_index::k_max_build_threads) {
    return false;
  }
  if (m_diskann_build_threads == diskann_build_threads) return true;
  m_diskann_build_threads = diskann_build_threads;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  clear_build_diagnostics();
  return true;
}

bool diskann_backend::set_diskann_build_blas_threads(
    uint32_t diskann_build_blas_threads) {
  if (m_mode != backend_mode::kExternal ||
      diskann_build_blas_threads > vector_index::k_max_build_threads) {
    return false;
  }
  if (m_diskann_build_blas_threads == diskann_build_blas_threads) {
    return true;
  }
  m_diskann_build_blas_threads = diskann_build_blas_threads;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  clear_build_diagnostics();
  return true;
}

bool diskann_backend::set_diskann_build_mode(
    diskann_build_mode diskann_build_mode_value) {
  if (m_mode != backend_mode::kExternal) return false;
  if (m_diskann_build_mode == diskann_build_mode_value) return true;
  m_diskann_build_mode = diskann_build_mode_value;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  clear_build_diagnostics();
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

uint32_t diskann_backend::diskann_build_blas_threads() const {
  return m_mode == backend_mode::kExternal ? m_diskann_build_blas_threads : 0;
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
  bool native_search_beamwidth_ok = true;
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    native_search_beamwidth_ok =
        m_native_state->set_search_beamwidth(diskann_search_beamwidth);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_live_search_beamwidth",
                    native_search_beamwidth_ok = false;);
  }
  if (m_native_runtime_enabled && m_native_state != nullptr &&
      !native_search_beamwidth_ok) {
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
  if (m_diskann_pq_code_budget_size == diskann_pq_code_budget_size)
    return true;
  m_diskann_pq_code_budget_size = diskann_pq_code_budget_size;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  clear_build_diagnostics();
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
  if (m_diskann_disk_pq_dims == diskann_disk_pq_dims) return true;
  m_diskann_disk_pq_dims = diskann_disk_pq_dims;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  clear_build_diagnostics();
  return true;
}

uint32_t diskann_backend::diskann_disk_pq_dims() const {
  return m_mode == backend_mode::kExternal ? m_diskann_disk_pq_dims : 0;
}

bool diskann_backend::set_diskann_accelerate_build(
    bool diskann_accelerate_build) {
  if (m_mode != backend_mode::kExternal) return false;
  if (m_diskann_accelerate_build == diskann_accelerate_build) return true;
  m_diskann_accelerate_build = diskann_accelerate_build;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  clear_build_diagnostics();
  return true;
}

bool diskann_backend::diskann_accelerate_build() const {
  return m_mode == backend_mode::kExternal && m_diskann_accelerate_build;
}

bool diskann_backend::set_diskann_shuffle_build(bool diskann_shuffle_build) {
  if (m_mode != backend_mode::kExternal) return false;
  if (m_diskann_shuffle_build == diskann_shuffle_build) return true;
  m_diskann_shuffle_build = diskann_shuffle_build;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  clear_build_diagnostics();
  return true;
}

bool diskann_backend::diskann_shuffle_build() const {
  return m_mode == backend_mode::kExternal && m_diskann_shuffle_build;
}

bool diskann_backend::set_diskann_use_bfs_cache(bool diskann_use_bfs_cache) {
  if (m_mode != backend_mode::kExternal) return false;
  if (m_diskann_use_bfs_cache == diskann_use_bfs_cache) return true;
  m_diskann_use_bfs_cache = diskann_use_bfs_cache;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  clear_build_diagnostics();
  return true;
}

bool diskann_backend::diskann_use_bfs_cache() const {
  return m_mode == backend_mode::kExternal && m_diskann_use_bfs_cache;
}

uint32_t diskann_backend::diskann_offline_search_threads() const {
  return m_mode == backend_mode::kExternal ? m_diskann_offline_search_threads
                                           : 0;
}

uint32_t diskann_backend::diskann_search_io_limit() const {
  return m_mode == backend_mode::kExternal ? m_diskann_search_io_limit : 0;
}

uint32_t diskann_backend::diskann_cache_nodes() const {
  return m_mode == backend_mode::kExternal ? m_diskann_cache_nodes : 0;
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
size_t estimate_diskann_flatten_memory_bytes(size_t entry_count,
                                             size_t dimension) {
  const size_t doc_id_bytes = saturated_mul_size(entry_count, sizeof(uint64_t));
  const size_t vector_bytes =
      saturated_mul_size(saturated_mul_size(entry_count, dimension),
                         sizeof(float));
  return saturated_add_size(doc_id_bytes, vector_bytes);
}

bool diskann_flatten_memory_budget_allows(size_t entry_count, size_t dimension,
                                          uint64_t budget_size) {
  if (budget_size == 0) return true;
  return estimate_diskann_flatten_memory_bytes(entry_count, dimension) <=
         budget_size;
}

void diskann_backend::clear_committed_snapshot_for_testing() {
  m_entries.clear();
  m_entries_complete = false;
  reset_streaming_doc_id_tracking();
}

bool diskann_backend::native_search_batch_for_testing(
    const std::vector<vector_data> &queries, size_t begin, size_t end,
    size_t top_k, std::vector<std::vector<search_result>> *results) const {
  if (m_native_state == nullptr || begin > end) return false;
  const uint32_t search_threads = static_cast<uint32_t>(
      vector_index::effective_diskann_search_threads(end - begin));
  return m_native_state->search_batch(queries, begin, end, top_k,
                                      search_threads, results);
}

std::string diskann_term_directory_for_testing(const std::string &index_name,
                                               uint64_t ctx) {
  constexpr uint32_t offline_search_threads =
      vector_index::k_default_diskann_offline_search_threads;
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                             kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0, 1,
                             offline_search_threads, 0, 0);
  return state.term_directory_for_testing(ctx);
}

std::string diskann_key_path_for_testing(const std::string &index_name,
                                         uint64_t ctx,
                                         const std::string &key_bytes) {
  constexpr uint32_t offline_search_threads =
      vector_index::k_default_diskann_offline_search_threads;
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                             kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0, 1,
                             offline_search_threads, 0, 0);
  if (key_bytes.empty()) return state.key_path_for_testing(ctx, nullptr, 0);
  return state.key_path_for_testing(
      ctx, reinterpret_cast<const uint8_t *>(key_bytes.data()),
      key_bytes.size());
}

bool diskann_load_value_for_testing(const std::string &index_name, uint64_t ctx,
                                    const std::string &key_bytes,
                                    std::string *value) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                             kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0, 1);
  if (key_bytes.empty())
    return state.load_value_for_testing(ctx, nullptr, 0, value);
  return state.load_value_for_testing(
      ctx, reinterpret_cast<const uint8_t *>(key_bytes.data()),
      key_bytes.size(), value);
}

bool diskann_save_value_for_testing(const std::string &index_name, uint64_t ctx,
                                    const std::string &key_bytes,
                                    const std::string &value) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                             kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0, 1);
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
                             kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0, 1);
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
                             kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0, 1);
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
                             kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0, 1);
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
                             kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0, 1);
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

bool diskann_offline_api_native_pq_build_available_for_testing(
    bool api_available, bool has_native_pq_build) {
  diskann_offline_api api;
  api.handle = api_available ? reinterpret_cast<void *>(1) : nullptr;
  api.build =
      api_available ? reinterpret_cast<diskann_offline_build_fn>(1) : nullptr;
  api.build_from_manifest =
      api_available
          ? reinterpret_cast<diskann_offline_build_from_manifest_fn>(1)
          : nullptr;
  api.build_from_native_pq =
      has_native_pq_build
          ? reinterpret_cast<diskann_offline_build_from_native_pq_fn>(1)
          : nullptr;
  api.load_index =
      api_available ? reinterpret_cast<diskann_offline_load_fn>(1) : nullptr;
  api.search =
      api_available ? reinterpret_cast<diskann_offline_search_fn>(1) : nullptr;
  api.search_batch = api_available
                         ? reinterpret_cast<diskann_offline_search_batch_fn>(1)
                         : nullptr;
  api.card =
      api_available ? reinterpret_cast<diskann_offline_card_fn>(1) : nullptr;
  api.drop_index =
      api_available ? reinterpret_cast<diskann_offline_drop_fn>(1) : nullptr;
  return api.native_pq_build_available();
}

bool diskann_vendored_runtime_api_load_for_testing() {
  diskann_vendored_runtime_api api;
  return api.load();
}

uint64_t diskann_vendored_runtime_capabilities_for_testing() {
  diskann_vendored_runtime_api api;
  if (!api.load()) return 0;
  return api.capabilities;
}

bool diskann_vendored_build_config_valid_for_testing(
    const std::string &data_path, const std::string &index_prefix,
    uint32_t dimension, uint32_t max_degree, uint32_t build_complexity,
    double build_memory_gb, double search_cache_ratio, std::string *error) {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
  mysql_vector_diskann_build_config config{};
  config.data_path = data_path.c_str();
  config.index_prefix = index_prefix.c_str();
  config.dimension = dimension;
  config.metric_type = diskann_metric_code(metric_type::kEuclidean);
  config.max_degree = max_degree;
  config.build_complexity = build_complexity;
  config.build_memory_gb = build_memory_gb;
  config.pq_chunks = 1;
  config.search_cache_ratio = search_cache_ratio;
  char error_buffer[128]{};
  const bool ok = mysql_vector_diskann_validate_build_config(
      &config, error_buffer, sizeof(error_buffer));
  if (error != nullptr) *error = error_buffer;
  return ok;
#else
  (void)data_path;
  (void)index_prefix;
  (void)dimension;
  (void)max_degree;
  (void)build_complexity;
  (void)build_memory_gb;
  (void)search_cache_ratio;
  if (error != nullptr) *error = "vendored runtime is not linked";
  return false;
#endif
}

bool diskann_vendored_search_config_valid_for_testing(
    uint32_t top_k, uint32_t search_complexity, uint32_t beamwidth,
    std::string *error) {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
  mysql_vector_diskann_search_config config{};
  config.top_k = top_k;
  config.search_complexity = search_complexity;
  config.beamwidth = beamwidth;
  char error_buffer[128]{};
  const bool ok = mysql_vector_diskann_validate_search_config(
      &config, error_buffer, sizeof(error_buffer));
  if (error != nullptr) *error = error_buffer;
  return ok;
#else
  (void)top_k;
  (void)search_complexity;
  (void)beamwidth;
  if (error != nullptr) *error = "vendored runtime is not linked";
  return false;
#endif
}

bool diskann_vendored_batch_search_available_for_testing() {
  return get_diskann_vendored_runtime_api().batch_search_available();
}

bool diskann_vendored_allocation_failure_drops_handle_for_testing() {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
  return mysql_vector_diskann_runtime_allocation_failure_drops_handle_for_testing();
#else
  return false;
#endif
}

bool diskann_vendored_load_config_budget_for_testing(size_t row_count,
                                                     double *build_memory_gb,
                                                     uint32_t *pq_chunks,
                                                     uint32_t *cache_nodes) {
#ifdef MYSQL_VECTOR_DISKANN_VENDORED_STATIC_LINKED
  constexpr uint32_t offline_search_threads =
      vector_index::k_default_diskann_offline_search_threads;
  diskann_native_state state(
      128, metric_type::kEuclidean, "idx_diskann_vendored_load_config_budget",
      kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 4, 1,
      offline_search_threads, 0, 0,
      vector_index::k_default_diskann_search_beamwidth, 0, 0,
      vector_index::k_default_diskann_pq_code_budget_ratio, 0.1);
  return state.vendored_load_config_budget_for_testing(
      row_count, build_memory_gb, pq_chunks, cache_nodes);
#else
  (void)row_count;
  (void)build_memory_gb;
  (void)pq_chunks;
  (void)cache_nodes;
  return false;
#endif
}

bool diskann_loaded_handle_batch_rejects_null_for_testing() {
  constexpr uint32_t offline_search_threads =
      vector_index::k_default_diskann_offline_search_threads;
  diskann_native_state state(2, metric_type::kEuclidean,
                             "idx_diskann_loaded_handle_batch_guard",
                             kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0, 1,
                             offline_search_threads, 0, 0);
  return state.loaded_handle_batch_rejects_null_for_testing();
}

double diskann_build_memory_size_gb_for_testing(uint64_t build_memory_size) {
  return diskann_build_memory_size_gb(build_memory_size);
}

uint64_t diskann_available_build_memory_size_for_testing() {
  return available_diskann_build_memory_size();
}

bool diskann_flatten_memory_budget_allows_for_testing(size_t entry_count,
                                                      size_t dimension,
                                                      uint64_t budget_size) {
  return diskann_flatten_memory_budget_allows(entry_count, dimension,
                                             budget_size);
}

uint32_t diskann_cache_nodes_for_build_for_testing(
    size_t row_count, size_t dimension, uint32_t explicit_cache_nodes,
    uint64_t search_cache_size, double search_cache_ratio, uint32_t max_degree,
    uint32_t disk_pq_dims) {
  return diskann_cache_nodes_for_build(row_count, dimension,
                                       explicit_cache_nodes, search_cache_size,
                                       search_cache_ratio, max_degree,
                                       disk_pq_dims);
}

uint32_t diskann_effective_offline_max_degree_for_testing(size_t row_count,
                                                          uint32_t max_degree) {
  return diskann_effective_offline_max_degree(row_count, max_degree);
}

uint32_t diskann_effective_loaded_cache_nodes_for_testing(
    size_t row_count, uint64_t search_cache_size, double search_cache_ratio) {
  constexpr uint32_t offline_search_threads =
      vector_index::k_default_diskann_offline_search_threads;
  diskann_native_state state(
      128, metric_type::kEuclidean, "idx_diskann_effective_loaded_cache_nodes",
      kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 4, 1,
      offline_search_threads, 0, 0,
      vector_index::k_default_diskann_search_beamwidth, 0, search_cache_size,
      vector_index::k_default_diskann_pq_code_budget_ratio,
      search_cache_ratio);
  return state.effective_loaded_cache_nodes_for_testing(row_count);
}

const char *diskann_runtime_variant_name_for_testing(int variant) {
  switch (variant) {
    case 0:
      return diskann_runtime_variant_name(diskann_runtime_variant::kNone);
    case 1:
      return diskann_runtime_variant_name(diskann_runtime_variant::kSerial);
    case 2:
      return diskann_runtime_variant_name(diskann_runtime_variant::kOffline);
    case 3:
      return diskann_runtime_variant_name(
          diskann_runtime_variant::kVendoredOffline);
    case 4:
      return diskann_runtime_variant_name(
          diskann_runtime_variant::kOfflineSegmented);
    case 5:
      return diskann_runtime_variant_name(
          diskann_runtime_variant::kVendoredSegmented);
    default:
      return diskann_runtime_variant_name(
          static_cast<diskann_runtime_variant>(variant));
  }
}

bool diskann_write_binary_values_for_testing(const std::string &path,
                                             uint32_t uint32_value,
                                             uint64_t uint64_value) {
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.is_open()) return false;
  return write_binary_value(file, uint32_value) &&
         write_binary_value(file, uint64_value);
}

uint32_t diskann_offline_load_use_bfs_cache_for_testing(uint32_t cache_nodes,
                                                        bool force_bfs_cache) {
  return diskann_offline_load_use_bfs_cache(cache_nodes, force_bfs_cache);
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
