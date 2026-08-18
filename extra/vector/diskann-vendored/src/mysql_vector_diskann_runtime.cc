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

#include "mysql_vector_diskann_runtime.h"

#include <algorithm>
#include <cstring>
#include <new>

struct mysql_vector_diskann_runtime_handle {
  const void *offline_handle{nullptr};
};

namespace {

constexpr uint32_t k_max_search_beamwidth = 128;

#ifdef MYSQL_VECTOR_DISKANN_RUNTIME_WITH_OFFLINE_ADAPTER
extern "C" bool mysql_vector_diskann_offline_build_from_manifest(
    const char *index_prefix, const char *manifest_path, uint32_t dimension,
    int32_t metric_type, uint32_t max_degree, uint32_t search_list_size,
    uint32_t build_threads, double index_mem_gb, uint32_t pq_chunks,
    uint32_t num_nodes_to_cache, uint32_t build_blas_threads,
    uint32_t disk_pq_dims, uint32_t accelerate_build,
    uint32_t shuffle_build);
extern "C" bool mysql_vector_diskann_offline_build_from_native_pq(
    const char *index_prefix, const char *manifest_path,
    const char *pq_pivots_path, const char *pq_compressed_path,
    const char *disk_pq_pivots_path, const char *disk_pq_compressed_path,
    uint32_t dimension, int32_t metric_type, uint32_t max_degree,
    uint32_t search_list_size, uint32_t build_threads, double index_mem_gb,
    uint32_t pq_chunks, uint32_t num_nodes_to_cache,
    uint32_t build_blas_threads, uint32_t disk_pq_dims,
    uint32_t accelerate_build, uint32_t shuffle_build);
extern "C" const void *mysql_vector_diskann_offline_load(
    const char *index_prefix, int32_t metric_type, uint32_t search_threads,
    uint32_t search_io_limit, uint32_t cache_nodes, uint32_t use_bfs_cache);
extern "C" int32_t mysql_vector_diskann_offline_search(
    const void *index_ptr, const float *query, size_t dimension,
    uint32_t top_k, uint32_t search_list_size, uint32_t beam_width,
    uint64_t *doc_ids, float *distances);
extern "C" int32_t mysql_vector_diskann_offline_search_batch(
    const void *index_ptr, const float *queries, size_t query_count,
    size_t dimension, uint32_t top_k, uint32_t search_list_size,
    uint32_t beam_width, uint32_t search_threads, uint64_t *doc_ids,
    float *distances, uint32_t *result_counts);
extern "C" uint64_t mysql_vector_diskann_offline_card(const void *index_ptr);
extern "C" void mysql_vector_diskann_offline_drop(const void *index_ptr);
#endif

void copy_error(const char *message, char *error_buffer, size_t error_buffer_size) {
  if (error_buffer == nullptr || error_buffer_size == 0) return;
  const size_t message_length = std::strlen(message);
  const size_t copy_length = std::min(message_length, error_buffer_size - 1);
  std::memcpy(error_buffer, message, copy_length);
  error_buffer[copy_length] = '\0';
}

bool validate_path(const char *path, const char *field_name, char *error_buffer,
                   size_t error_buffer_size) {
  if (path != nullptr && path[0] != '\0') return true;
  copy_error(field_name, error_buffer, error_buffer_size);
  return false;
}

using offline_drop_function = void (*)(const void *);

mysql_vector_diskann_runtime_handle *wrap_offline_handle(
    const void *offline_handle, offline_drop_function drop_handle,
    char *error_buffer, size_t error_buffer_size,
    bool allocate_handle = true) noexcept {
  auto *handle = allocate_handle
                     ? new (std::nothrow) mysql_vector_diskann_runtime_handle()
                     : nullptr;
  if (handle == nullptr) {
    if (drop_handle != nullptr) drop_handle(offline_handle);
    copy_error("runtime handle allocation failed", error_buffer,
               error_buffer_size);
    return nullptr;
  }
  handle->offline_handle = offline_handle;
  if (error_buffer != nullptr && error_buffer_size > 0) error_buffer[0] = '\0';
  return handle;
}

}  // namespace

uint32_t mysql_vector_diskann_runtime_abi_version(void) {
  return MYSQL_VECTOR_DISKANN_RUNTIME_ABI_VERSION;
}

uint64_t mysql_vector_diskann_runtime_capabilities(void) {
  uint64_t capabilities = MYSQL_VECTOR_DISKANN_RUNTIME_CAPABILITY_CONFIG;
#ifdef MYSQL_VECTOR_DISKANN_RUNTIME_WITH_OFFLINE_ADAPTER
  capabilities |= MYSQL_VECTOR_DISKANN_RUNTIME_CAPABILITY_STRUCTURED_BUILD;
  capabilities |= MYSQL_VECTOR_DISKANN_RUNTIME_CAPABILITY_BATCH_SEARCH;
  capabilities |= MYSQL_VECTOR_DISKANN_RUNTIME_CAPABILITY_NATIVE_PQ_BRIDGE;
#endif
  return capabilities;
}

bool mysql_vector_diskann_validate_build_config(
    const mysql_vector_diskann_build_config *config, char *error_buffer,
    size_t error_buffer_size) {
  if (config == nullptr) {
    copy_error("build config is null", error_buffer, error_buffer_size);
    return false;
  }
  if (!validate_path(config->data_path, "data_path is empty", error_buffer,
                     error_buffer_size)) {
    return false;
  }
  if (!validate_path(config->index_prefix, "index_prefix is empty", error_buffer,
                     error_buffer_size)) {
    return false;
  }
  if (config->dimension == 0) {
    copy_error("dimension is zero", error_buffer, error_buffer_size);
    return false;
  }
  if (config->max_degree == 0) {
    copy_error("max_degree is zero", error_buffer, error_buffer_size);
    return false;
  }
  if (config->build_complexity == 0) {
    copy_error("build_complexity is zero", error_buffer, error_buffer_size);
    return false;
  }
  if (config->build_memory_gb < 0.0) {
    copy_error("build_memory_gb is negative", error_buffer, error_buffer_size);
    return false;
  }
  if (config->pq_chunks == 0) {
    copy_error("pq_chunks is zero", error_buffer, error_buffer_size);
    return false;
  }
  if (config->disk_pq_dims > config->dimension) {
    copy_error("disk_pq_dims is out of range", error_buffer,
               error_buffer_size);
    return false;
  }
  if (config->search_cache_ratio < 0.0 || config->search_cache_ratio > 1.0) {
    copy_error("search_cache_ratio is out of range", error_buffer,
               error_buffer_size);
    return false;
  }
  if (error_buffer != nullptr && error_buffer_size > 0) error_buffer[0] = '\0';
  return true;
}

bool mysql_vector_diskann_build_from_manifest(
    const mysql_vector_diskann_build_config *config, char *error_buffer,
    size_t error_buffer_size) {
  if (!mysql_vector_diskann_validate_build_config(config, error_buffer,
                                                  error_buffer_size)) {
    return false;
  }
  if (config->accelerate_build != 0 || config->shuffle_build != 0) {
    copy_error("DiskANN offline adapter does not support accelerated or shuffled "
               "build",
               error_buffer, error_buffer_size);
    return false;
  }
#ifdef MYSQL_VECTOR_DISKANN_RUNTIME_WITH_OFFLINE_ADAPTER
  const bool ok = mysql_vector_diskann_offline_build_from_manifest(
      config->index_prefix, config->data_path, config->dimension,
      config->metric_type, config->max_degree, config->build_complexity,
      config->build_threads, config->build_memory_gb, config->pq_chunks,
      config->cache_nodes, config->build_blas_threads, config->disk_pq_dims,
      config->accelerate_build, config->shuffle_build);
  if (!ok) {
    copy_error("offline adapter build failed", error_buffer, error_buffer_size);
    return false;
  }
  if (error_buffer != nullptr && error_buffer_size > 0) error_buffer[0] = '\0';
  return true;
#else
  copy_error("structured build is unavailable", error_buffer,
             error_buffer_size);
  return false;
#endif
}

bool mysql_vector_diskann_build_from_native_pq(
    const mysql_vector_diskann_build_config *config,
    const char *pq_pivots_path, const char *pq_compressed_path,
    const char *disk_pq_pivots_path, const char *disk_pq_compressed_path,
    char *error_buffer, size_t error_buffer_size) {
  if (!mysql_vector_diskann_validate_build_config(config, error_buffer,
                                                  error_buffer_size)) {
    return false;
  }
  if (!validate_path(pq_pivots_path, "pq_pivots_path is empty", error_buffer,
                     error_buffer_size) ||
      !validate_path(pq_compressed_path, "pq_compressed_path is empty",
                     error_buffer, error_buffer_size)) {
    return false;
  }
  if (config->disk_pq_dims != 0 &&
      (!validate_path(disk_pq_pivots_path,
                      "disk_pq_pivots_path is empty", error_buffer,
                      error_buffer_size) ||
       !validate_path(disk_pq_compressed_path,
                      "disk_pq_compressed_path is empty", error_buffer,
                      error_buffer_size))) {
    return false;
  }
  if (config->accelerate_build != 0 || config->shuffle_build != 0) {
    copy_error("DiskANN native PQ bridge does not support accelerated or "
               "shuffled build",
               error_buffer, error_buffer_size);
    return false;
  }
#ifdef MYSQL_VECTOR_DISKANN_RUNTIME_WITH_OFFLINE_ADAPTER
  const bool ok = mysql_vector_diskann_offline_build_from_native_pq(
      config->index_prefix, config->data_path, pq_pivots_path,
      pq_compressed_path, disk_pq_pivots_path, disk_pq_compressed_path,
      config->dimension, config->metric_type, config->max_degree,
      config->build_complexity, config->build_threads, config->build_memory_gb,
      config->pq_chunks, config->cache_nodes, config->build_blas_threads,
      config->disk_pq_dims, config->accelerate_build, config->shuffle_build);
  if (!ok) {
    copy_error("offline adapter native PQ build failed", error_buffer,
               error_buffer_size);
    return false;
  }
  if (error_buffer != nullptr && error_buffer_size > 0) error_buffer[0] = '\0';
  return true;
#else
  copy_error("native PQ bridge is unavailable", error_buffer,
             error_buffer_size);
  return false;
#endif
}

mysql_vector_diskann_runtime_handle *mysql_vector_diskann_runtime_load(
    const mysql_vector_diskann_build_config *build_config,
    const mysql_vector_diskann_search_config *search_config, char *error_buffer,
    size_t error_buffer_size) {
  if (build_config == nullptr) {
    copy_error("build config is null", error_buffer, error_buffer_size);
    return nullptr;
  }
  if (!validate_path(build_config->index_prefix, "index_prefix is empty",
                     error_buffer, error_buffer_size) ||
      build_config->dimension == 0) {
    if (build_config->dimension == 0) {
      copy_error("dimension is zero", error_buffer, error_buffer_size);
    }
    return nullptr;
  }
  if (!mysql_vector_diskann_validate_search_config(search_config, error_buffer,
                                                   error_buffer_size)) {
    return nullptr;
  }
#ifdef MYSQL_VECTOR_DISKANN_RUNTIME_WITH_OFFLINE_ADAPTER
  const void *offline_handle = mysql_vector_diskann_offline_load(
      build_config->index_prefix, build_config->metric_type,
      search_config->batch_search_threads, search_config->search_io_limit,
      search_config->cache_nodes, build_config->use_bfs_cache);
  if (offline_handle == nullptr) {
    copy_error("offline adapter load failed", error_buffer, error_buffer_size);
    return nullptr;
  }
  return wrap_offline_handle(offline_handle, mysql_vector_diskann_offline_drop,
                             error_buffer, error_buffer_size);
#else
  copy_error("runtime load is unavailable", error_buffer, error_buffer_size);
  return nullptr;
#endif
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool mysql_vector_diskann_runtime_allocation_failure_drops_handle_for_testing(
    void) {
  uint8_t dropped = 0;
  const auto drop_handle = [](const void *handle) {
    *static_cast<uint8_t *>(const_cast<void *>(handle)) = 1;
  };
  char error_buffer[64]{};
  mysql_vector_diskann_runtime_handle *handle = wrap_offline_handle(
      &dropped, drop_handle, error_buffer, sizeof(error_buffer), false);
  return handle == nullptr && dropped == 1 &&
         std::strcmp(error_buffer, "runtime handle allocation failed") == 0;
}
#endif

int32_t mysql_vector_diskann_runtime_search(
    const mysql_vector_diskann_runtime_handle *handle, const float *query,
    size_t dimension, const mysql_vector_diskann_search_config *search_config,
    uint64_t *doc_ids, float *distances) {
  if (handle == nullptr || handle->offline_handle == nullptr ||
      !mysql_vector_diskann_validate_search_config(search_config, nullptr, 0)) {
    return -1;
  }
#ifdef MYSQL_VECTOR_DISKANN_RUNTIME_WITH_OFFLINE_ADAPTER
  return mysql_vector_diskann_offline_search(
      handle->offline_handle, query, dimension, search_config->top_k,
      search_config->search_complexity, search_config->beamwidth, doc_ids,
      distances);
#else
  (void)query;
  (void)dimension;
  (void)doc_ids;
  (void)distances;
  return -1;
#endif
}

int32_t mysql_vector_diskann_runtime_search_batch(
    const mysql_vector_diskann_runtime_handle *handle, const float *queries,
    size_t query_count, size_t dimension,
    const mysql_vector_diskann_search_config *search_config, uint64_t *doc_ids,
    float *distances, uint32_t *result_counts) {
  if (handle == nullptr || handle->offline_handle == nullptr ||
      !mysql_vector_diskann_validate_search_config(search_config, nullptr, 0)) {
    return -1;
  }
#ifdef MYSQL_VECTOR_DISKANN_RUNTIME_WITH_OFFLINE_ADAPTER
  return mysql_vector_diskann_offline_search_batch(
      handle->offline_handle, queries, query_count, dimension,
      search_config->top_k, search_config->search_complexity,
      search_config->beamwidth, search_config->batch_search_threads, doc_ids,
      distances, result_counts);
#else
  (void)queries;
  (void)query_count;
  (void)dimension;
  (void)doc_ids;
  (void)distances;
  (void)result_counts;
  return -1;
#endif
}

uint64_t mysql_vector_diskann_runtime_card(
    const mysql_vector_diskann_runtime_handle *handle) {
#ifdef MYSQL_VECTOR_DISKANN_RUNTIME_WITH_OFFLINE_ADAPTER
  if (handle == nullptr || handle->offline_handle == nullptr) return 0;
  return mysql_vector_diskann_offline_card(handle->offline_handle);
#else
  (void)handle;
  return 0;
#endif
}

void mysql_vector_diskann_runtime_drop(
    const mysql_vector_diskann_runtime_handle *handle) {
#ifdef MYSQL_VECTOR_DISKANN_RUNTIME_WITH_OFFLINE_ADAPTER
  if (handle != nullptr && handle->offline_handle != nullptr) {
    mysql_vector_diskann_offline_drop(handle->offline_handle);
  }
#endif
  delete handle;
}

bool mysql_vector_diskann_validate_search_config(
    const mysql_vector_diskann_search_config *config, char *error_buffer,
    size_t error_buffer_size) {
  if (config == nullptr) {
    copy_error("search config is null", error_buffer, error_buffer_size);
    return false;
  }
  if (config->top_k == 0) {
    copy_error("top_k is zero", error_buffer, error_buffer_size);
    return false;
  }
  if (config->search_complexity == 0) {
    copy_error("search_complexity is zero", error_buffer, error_buffer_size);
    return false;
  }
  if (config->top_k > config->search_complexity) {
    copy_error("top_k exceeds search_complexity", error_buffer,
               error_buffer_size);
    return false;
  }
  if (config->beamwidth == 0) {
    copy_error("beamwidth is zero", error_buffer, error_buffer_size);
    return false;
  }
  if (config->beamwidth > k_max_search_beamwidth) {
    copy_error("beamwidth is too large", error_buffer, error_buffer_size);
    return false;
  }
  if (error_buffer != nullptr && error_buffer_size > 0) error_buffer[0] = '\0';
  return true;
}
