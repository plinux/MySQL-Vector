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

struct build_pipeline_runtime_config {
  build_pipeline_thresholds thresholds;
  diskann_segment_profile diskann_profile{diskann_segment_profile::kManual};
  uint32_t progress_interval{k_default_build_pipeline_progress_interval};
};

/**
  Validate whether a provider can be used with a backend mode.
*/
bool runtime_provider_accepts_mode(backend_provider provider,
                                   backend_mode mode);

/**
  Validate FAISS IVF-PQ parameters against the configured vector dimension.
*/
bool valid_faiss_ivf_pq_config(size_t dimension, uint32_t nlist,
                               uint32_t nprobe, uint32_t pq_m,
                               uint32_t pq_bits);

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
