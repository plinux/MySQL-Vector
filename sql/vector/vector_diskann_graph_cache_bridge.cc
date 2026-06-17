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

#include "sql/vector/vector_diskann_graph_cache_bridge.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <string>

namespace vector_index {
namespace {

using bridge_clock = std::chrono::steady_clock;

uint64_t elapsed_ms(bridge_clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          bridge_clock::now() - start)
          .count());
}

void set_error(std::string *error, const char *message) {
  if (error != nullptr) *error = message;
}

bool validate_bridge_config(const diskann_graph_cache_bridge_config &config,
                            std::string *error) {
  if (config.input_manifest_path.empty()) {
    set_error(error, "input manifest path is empty");
    return false;
  }
  if (config.index_prefix.empty()) {
    set_error(error, "index prefix is empty");
    return false;
  }
  if (config.row_count == 0) {
    set_error(error, "row_count is zero");
    return false;
  }
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
  if (config.disk_pq_dims > config.dimension) {
    set_error(error, "disk_pq_dims exceeds dimension");
    return false;
  }
  if (config.centroid_count == 0) {
    set_error(error, "centroid_count is zero");
    return false;
  }
  if (config.max_degree == 0) {
    set_error(error, "max_degree is zero");
    return false;
  }
  if (config.build_complexity == 0) {
    set_error(error, "build_complexity is zero");
    return false;
  }
  return true;
}

void set_callback_error(std::string *error, const char *message,
                        const std::array<char, 1024> &buffer) {
  if (error == nullptr) return;
  if (buffer[0] != '\0') {
    *error = buffer.data();
    return;
  }
  *error = message;
}

}  // namespace

bool build_diskann_graph_cache_from_native_pq(
    const diskann_graph_cache_bridge_config &config,
    diskann_graph_cache_bridge_result *result, std::string *error) {
  if (result == nullptr) {
    set_error(error, "result is null");
    return false;
  }
  *result = {};
  const bridge_clock::time_point start = bridge_clock::now();

  if (!validate_bridge_config(config, error)) return false;

  const bridge_clock::time_point validation_start = bridge_clock::now();
  diskann_pq_artifact_metadata memory_metadata;
  memory_metadata.row_count = config.row_count;
  memory_metadata.dimension = config.dimension;
  memory_metadata.pq_chunks = config.pq_chunks;
  memory_metadata.centroid_count = config.centroid_count;
  memory_metadata.zero_mean = config.metric == metric_type::kEuclidean ||
                              config.metric == metric_type::kCosine;
  if (!validate_diskann_pq_artifacts(config.artifacts, memory_metadata,
                                     error)) {
    result->artifact_validation_ms = elapsed_ms(validation_start);
    result->elapsed_ms = elapsed_ms(start);
    return false;
  }
  if (config.disk_pq_dims != 0) {
    diskann_pq_artifact_metadata disk_metadata = memory_metadata;
    disk_metadata.pq_chunks = config.disk_pq_dims;
    disk_metadata.zero_mean = false;
    if (!validate_diskann_pq_artifacts(config.disk_artifacts, disk_metadata,
                                       error)) {
      result->artifact_validation_ms = elapsed_ms(validation_start);
      result->elapsed_ms = elapsed_ms(start);
      return false;
    }
    std::vector<float> disk_centroid;
    if (!read_diskann_pq_float_artifact(
            config.disk_artifacts.centroid_path, config.dimension, 1,
            &disk_centroid, error)) {
      result->artifact_validation_ms = elapsed_ms(validation_start);
      result->elapsed_ms = elapsed_ms(start);
      return false;
    }
    if (std::any_of(disk_centroid.begin(), disk_centroid.end(),
                    [](float value) { return value != 0.0F; })) {
      set_error(error, "disk PQ centroid is not zero");
      result->artifact_validation_ms = elapsed_ms(validation_start);
      result->elapsed_ms = elapsed_ms(start);
      return false;
    }
  }
  result->artifact_validation_ms = elapsed_ms(validation_start);

  if (config.build_fn == nullptr) {
    set_error(error, "native PQ graph/cache bridge is unavailable");
    result->elapsed_ms = elapsed_ms(start);
    return false;
  }
  result->bridge_name = "native_pq_graph_cache";
  result->official_pq_used = false;

  std::array<char, 1024> error_buffer{};
  const bridge_clock::time_point graph_start = bridge_clock::now();
  if (!config.build_fn(config.index_prefix.c_str(),
                       config.input_manifest_path.c_str(),
                       config.artifacts.pivot_path.c_str(),
                       config.artifacts.compressed_path.c_str(),
                       config.disk_pq_dims == 0
                           ? nullptr
                           : config.disk_artifacts.pivot_path.c_str(),
                       config.disk_pq_dims == 0
                           ? nullptr
                           : config.disk_artifacts.compressed_path.c_str(),
                       config.dimension, config.diskann_metric_type,
                       config.max_degree, config.build_complexity,
                       config.build_threads, config.build_memory_gb,
                       config.pq_chunks, config.cache_nodes,
                       config.blas_threads, config.disk_pq_dims,
                       config.accelerate_build, config.shuffle_build,
                       error_buffer.data(), error_buffer.size(),
                       config.build_context)) {
    result->graph_ms = elapsed_ms(graph_start);
    result->elapsed_ms = elapsed_ms(start);
    set_callback_error(error, "native PQ graph/cache bridge build failed",
                       error_buffer);
    return false;
  }
  result->graph_ms = elapsed_ms(graph_start);

  diskann_pq_bridge_manifest manifest;
  if (!make_diskann_pq_bridge_manifest(
          config.artifacts, memory_metadata, config.max_degree,
          config.build_complexity, config.cache_nodes, true, false, &manifest,
          error)) {
    result->elapsed_ms = elapsed_ms(start);
    return false;
  }
  if (!write_diskann_pq_bridge_manifest(config.artifacts, manifest, error)) {
    result->elapsed_ms = elapsed_ms(start);
    return false;
  }

  diskann_pq_bridge_manifest reread_manifest;
  if (!read_diskann_pq_bridge_manifest(config.artifacts, &reread_manifest,
                                       error)) {
    result->elapsed_ms = elapsed_ms(start);
    return false;
  }
  if (!validate_diskann_pq_bridge_manifest(config.artifacts, memory_metadata,
                                           reread_manifest, error)) {
    result->elapsed_ms = elapsed_ms(start);
    return false;
  }

  result->artifacts_consumed = true;
  result->cache_ms = 0;
  result->elapsed_ms = elapsed_ms(start);
  if (error != nullptr) error->clear();
  return true;
}

}  // namespace vector_index
