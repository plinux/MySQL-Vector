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

#ifndef SQL_VECTOR_DISKANN_GRAPH_CACHE_BRIDGE_INCLUDED
#define SQL_VECTOR_DISKANN_GRAPH_CACHE_BRIDGE_INCLUDED

#include <cstddef>
#include <cstdint>
#include <string>

#include "sql/vector/vector_diskann_artifact_layout.h"
#include "sql/vector/vector_index_backend.h"

namespace vector_index {

/**
  Low-level builder used by the bridge after MySQL native PQ artifacts pass
  validation. The callback must build graph/cache artifacts without retraining
  DiskANN PQ data.
*/
using diskann_graph_cache_bridge_build_fn = bool (*)(
    const char *index_prefix, const char *manifest_path,
    const char *pq_pivots_path, const char *pq_compressed_path,
    const char *disk_pq_pivots_path, const char *disk_pq_compressed_path,
    uint32_t dimension, int32_t metric_type, uint32_t max_degree,
    uint32_t build_complexity, uint32_t build_threads, double build_memory_gb,
    uint32_t pq_chunks, uint32_t cache_nodes, uint32_t blas_threads,
    uint32_t disk_pq_dims, bool accelerate_build, bool shuffle_build,
    char *error_buffer, size_t error_buffer_size, void *context);

/** Inputs for building DiskANN graph/cache artifacts from native PQ outputs. */
struct diskann_graph_cache_bridge_config {
  std::string input_manifest_path;
  std::string index_prefix;
  diskann_pq_artifact_paths artifacts;
  diskann_pq_artifact_paths disk_artifacts;
  metric_type metric{metric_type::kEuclidean};
  int32_t diskann_metric_type{0};
  uint64_t row_count{0};
  uint32_t dimension{0};
  uint32_t pq_chunks{0};
  uint32_t centroid_count{256};
  uint32_t max_degree{0};
  uint32_t build_complexity{0};
  uint32_t build_threads{0};
  uint32_t blas_threads{0};
  double build_memory_gb{0.0};
  uint32_t cache_nodes{0};
  uint32_t disk_pq_dims{0};
  bool use_bfs_cache{false};
  bool accelerate_build{false};
  bool shuffle_build{false};
  diskann_graph_cache_bridge_build_fn build_fn{nullptr};
  void *build_context{nullptr};
};

/** Result and timing information from the native PQ graph/cache bridge. */
struct diskann_graph_cache_bridge_result {
  bool artifacts_consumed{false};
  bool official_pq_used{false};
  std::string bridge_name;
  std::string fallback_reason;
  uint64_t elapsed_ms{0};
  uint64_t graph_ms{0};
  uint64_t cache_ms{0};
  uint64_t artifact_validation_ms{0};
};

/**
  Build DiskANN graph/cache artifacts from MySQL-Vector native PQ artifacts.

  This entry must only report success when the bridge consumes the native PQ
  artifacts and skips DiskANN official PQ training/compression.

  @retval true Graph/cache artifacts were built from native PQ artifacts.
  @retval false Validation failed, unsupported configuration, or bridge missing.
*/
bool build_diskann_graph_cache_from_native_pq(
    const diskann_graph_cache_bridge_config &config,
    diskann_graph_cache_bridge_result *result, std::string *error);

}  // namespace vector_index

#endif  // SQL_VECTOR_DISKANN_GRAPH_CACHE_BRIDGE_INCLUDED
