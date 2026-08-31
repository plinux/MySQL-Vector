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

#ifndef SQL_VECTOR_INDEX_SERVICE_INTERNAL_INCLUDED
#define SQL_VECTOR_INDEX_SERVICE_INTERNAL_INCLUDED

#include <memory>
#include <string>

#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_index_service.h"

namespace vector_index::detail {

size_t compute_diskann_exact_rerank_candidate_top_k(
    size_t top_k, size_t query_count, size_t authoritative_count,
    size_t segment_count, uint32_t search_complexity,
    diskann_search_profile search_profile, size_t result_budget,
    size_t candidate_target);

std::unique_ptr<backend> build_backend_from_config(
    const std::string &index_name,
    const vector_index::index_service::index_config &config);

/** Compare every persisted and runtime-relevant index configuration field. */
bool index_configs_equal(
    const vector_index::index_service::index_config &lhs,
    const vector_index::index_service::index_config &rhs);

/** Merge one segment's build diagnostics into the index-wide aggregate. */
void merge_segment_build_diagnostics(
    uint64_t segment_row_count, const backend_build_diagnostics &source,
    backend_build_diagnostics *aggregate);

/** Mark a lifecycle transition as failed with the supplied error code. */
void mark_lifecycle_failure(
    vector_index::index_service::lifecycle_info *lifecycle,
    uint32_t error_code);

template <typename... Bools>
inline bool all_true(Bools... values) {
  return (... && static_cast<bool>(values));
}

inline bool can_rebuild_after_search_failure(
    const vector_index::index_service::index_config &config) {
  return config.mode == backend_mode::kMemory &&
         config.provider == backend_provider::kHnswlib;
}

}  // namespace vector_index::detail

#endif  // SQL_VECTOR_INDEX_SERVICE_INTERNAL_INCLUDED
