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

#ifndef SQL_VECTOR_INDEX_RUNTIME_CONFIG_INCLUDED
#define SQL_VECTOR_INDEX_RUNTIME_CONFIG_INCLUDED

#include <cstddef>
#include <cstdint>

#include "sql/vector/vector_build_pipeline_policy.h"
#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_index_limits.h"

namespace vector_index {

enum class faiss_runtime_index_kind { kHnsw, kIvfFlat, kIvfPq };

struct runtime_common_config {
  size_t dimension{0};
  metric_type metric{metric_type::kEuclidean};
  backend_provider provider{backend_provider::kDiskAnn};
  backend_mode mode{backend_mode::kExternal};
  index_consistency_mode consistency_mode{
      index_consistency_mode::kTransactional};
};

struct diskann_runtime_config {
  diskann_build_mode build_mode{diskann_build_mode::kAuto};
  uint32_t max_degree{k_default_diskann_max_degree};
  uint32_t build_complexity{k_default_diskann_build_complexity};
  uint32_t search_complexity{k_default_diskann_search_complexity};
  uint32_t search_beamwidth{k_default_diskann_search_beamwidth};
  uint64_t pq_code_budget_size{0};
  uint64_t build_memory_size{0};
  uint64_t search_cache_size{0};
  uint32_t build_threads{0};
  uint32_t build_blas_threads{1};
  uint32_t search_threads{0};
};

struct faiss_runtime_config {
  faiss_runtime_index_kind index_kind{faiss_runtime_index_kind::kHnsw};
  uint32_t nlist{0};
  uint32_t nprobe{0};
  uint32_t pq_m{0};
  uint32_t pq_bits{0};
  uint64_t train_size{0};
  uint32_t build_threads{0};
  uint32_t search_threads{0};
};

struct hnsw_runtime_config {
  uint32_t hnsw_m{k_default_hnsw_m};
  uint32_t ef_construction{k_default_hnsw_ef_construction};
  uint32_t ef_search{k_default_hnsw_search_ef};
  uint64_t index_memory_size{0};
  uint32_t build_threads{0};
  uint32_t search_threads{0};
};

struct build_pipeline_runtime_config {
  build_pipeline_thresholds thresholds;
  uint32_t progress_interval{k_default_build_pipeline_progress_interval};
};

/**
  Validate whether a provider can be used with a backend mode.
*/
bool runtime_provider_accepts_mode(backend_provider provider,
                                   backend_mode mode);

/**
  Validate the common runtime configuration.
*/
bool runtime_common_config_valid(const runtime_common_config &config);

/**
  Infer the FAISS runtime variant from IVF/PQ parameters.
*/
faiss_runtime_index_kind faiss_runtime_kind_from_params(uint32_t nlist,
                                                        uint32_t pq_m,
                                                        uint32_t pq_bits);

/**
  Validate FAISS IVF-PQ parameters against the configured vector dimension.
*/
bool valid_faiss_ivf_pq_config(size_t dimension, uint32_t nlist,
                               uint32_t nprobe, uint32_t pq_m,
                               uint32_t pq_bits);

/**
  Resolve a statement-level thread value against the global value.

  Statement-level 0 means inherit the global value. Global 0 means automatic.
*/
uint32_t resolve_runtime_threads(uint32_t statement_threads,
                                 uint32_t global_threads);

/**
  Return a deterministic uniform sample position.

  @retval true The requested sample position was produced.
  @retval false Invalid arguments.
*/
bool runtime_uniform_sample_position(size_t sample_index, size_t total_count,
                                     size_t sample_count, size_t *position);

/**
  Return the current global build-pipeline runtime configuration.
*/
build_pipeline_runtime_config global_build_pipeline_runtime_config();

}  // namespace vector_index

#endif  // SQL_VECTOR_INDEX_RUNTIME_CONFIG_INCLUDED
