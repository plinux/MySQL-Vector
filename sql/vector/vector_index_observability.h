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

#ifndef SQL_VECTOR_INDEX_OBSERVABILITY_INCLUDED
#define SQL_VECTOR_INDEX_OBSERVABILITY_INCLUDED

#include <cstdint>
#include <string>

#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_index_registry.h"

namespace vector_index_observability {

struct vector_search_advice {
  std::string advice{"not_applicable"};
  std::string reason{"not_segmented_diskann_search"};
  uint64_t suggested_complexity{0};
  uint64_t suggested_segment_target_size{0};
};

uint64_t pending_apply_count(const vector_index_registry::index_info &info);
uint64_t rebuild_progress(const vector_index_registry::index_info &info);
uint64_t recover_progress(const vector_index_registry::index_info &info);
bool is_loaded(const vector_index_registry::index_info &info);
bool is_writable(const vector_index_registry::index_info &info);
vector_search_advice derive_diskann_search_advice(
    const vector_index::backend_build_diagnostics &diagnostics,
    uint32_t requested_top_k, uint64_t result_budget);

}  // namespace vector_index_observability

#endif  // SQL_VECTOR_INDEX_OBSERVABILITY_INCLUDED
