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

#ifndef MYSQL_VECTOR_DISKANN_RUNTIME_INCLUDED
#define MYSQL_VECTOR_DISKANN_RUNTIME_INCLUDED

#include <stddef.h>
#include <stdint.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MYSQL_VECTOR_DISKANN_RUNTIME_ABI_VERSION 2U

enum mysql_vector_diskann_runtime_capability {
  MYSQL_VECTOR_DISKANN_RUNTIME_CAPABILITY_CONFIG = 1ULL << 0,
  MYSQL_VECTOR_DISKANN_RUNTIME_CAPABILITY_STRUCTURED_BUILD = 1ULL << 1,
  MYSQL_VECTOR_DISKANN_RUNTIME_CAPABILITY_BATCH_SEARCH = 1ULL << 2,
  MYSQL_VECTOR_DISKANN_RUNTIME_CAPABILITY_NATIVE_PQ_BRIDGE = 1ULL << 3
};

struct mysql_vector_diskann_runtime_handle;

struct mysql_vector_diskann_build_config {
  const char *data_path;
  const char *index_prefix;
  uint32_t dimension;
  int32_t metric_type;
  uint32_t max_degree;
  uint32_t build_complexity;
  uint32_t build_threads;
  uint32_t build_blas_threads;
  double build_memory_gb;
  uint32_t pq_chunks;
  uint32_t cache_nodes;
  uint64_t search_cache_size;
  double search_cache_ratio;
  uint32_t disk_pq_dims;
  uint32_t use_bfs_cache;
  uint32_t accelerate_build;
  uint32_t shuffle_build;
};

struct mysql_vector_diskann_search_config {
  uint32_t top_k;
  uint32_t search_complexity;
  uint32_t beamwidth;
  uint32_t batch_search_threads;
  uint32_t search_io_limit;
  uint32_t cache_nodes;
};

uint32_t mysql_vector_diskann_runtime_abi_version(void);
uint64_t mysql_vector_diskann_runtime_capabilities(void);

bool mysql_vector_diskann_validate_build_config(
    const struct mysql_vector_diskann_build_config *config, char *error_buffer,
    size_t error_buffer_size);

bool mysql_vector_diskann_validate_search_config(
    const struct mysql_vector_diskann_search_config *config, char *error_buffer,
    size_t error_buffer_size);

bool mysql_vector_diskann_build_from_manifest(
    const struct mysql_vector_diskann_build_config *config, char *error_buffer,
    size_t error_buffer_size);

bool mysql_vector_diskann_build_from_native_pq(
    const struct mysql_vector_diskann_build_config *config,
    const char *pq_pivots_path, const char *pq_compressed_path,
    const char *disk_pq_pivots_path, const char *disk_pq_compressed_path,
    char *error_buffer, size_t error_buffer_size);

struct mysql_vector_diskann_runtime_handle *mysql_vector_diskann_runtime_load(
    const struct mysql_vector_diskann_build_config *build_config,
    const struct mysql_vector_diskann_search_config *search_config,
    char *error_buffer, size_t error_buffer_size);

int32_t mysql_vector_diskann_runtime_search(
    const struct mysql_vector_diskann_runtime_handle *handle,
    const float *query, size_t dimension,
    const struct mysql_vector_diskann_search_config *search_config,
    uint64_t *doc_ids, float *distances);

int32_t mysql_vector_diskann_runtime_search_batch(
    const struct mysql_vector_diskann_runtime_handle *handle,
    const float *queries, size_t query_count, size_t dimension,
    const struct mysql_vector_diskann_search_config *search_config,
    uint64_t *doc_ids, float *distances, uint32_t *result_counts);

uint64_t mysql_vector_diskann_runtime_card(
    const struct mysql_vector_diskann_runtime_handle *handle);

void mysql_vector_diskann_runtime_drop(
    const struct mysql_vector_diskann_runtime_handle *handle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MYSQL_VECTOR_DISKANN_RUNTIME_INCLUDED
