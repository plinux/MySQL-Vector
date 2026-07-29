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

#ifndef SQL_VECTOR_DISKANN_SCHEDULER_INCLUDED
#define SQL_VECTOR_DISKANN_SCHEDULER_INCLUDED

#include <cstdint>

#include "sql/vector/vector_index_limits.h"

namespace vector_index {

struct diskann_segment_budget_input {
  uint32_t dimension{0};
  uint64_t row_count{0};
  uint64_t payload_size{0};
  uint64_t pq_code_budget_size{0};
  double pq_code_budget_ratio{k_default_diskann_pq_code_budget_ratio};
  uint32_t disk_pq_dims{0};
  uint32_t max_degree{k_default_diskann_max_degree};
  uint64_t search_cache_size{0};
  double search_cache_ratio{0.1};
  uint64_t build_memory_size{0};
  uint64_t available_build_memory_size{0};
};

struct diskann_segment_budget {
  uint32_t pq_chunks{0};
  uint32_t disk_pq_dims{0};
  uint64_t pq_code_budget_size{0};
  double pq_code_budget_gb{0.0};
  uint32_t cache_nodes{0};
  uint64_t build_memory_size{0};
  double build_memory_gb{0.0};
};

/**
  Derive per-segment DiskANN PQ/cache/build budget.

  @param input Segment dimensions, payload size and user budget settings.
  @param budget Output DiskANN budget.

  @retval true A valid budget was produced.
  @retval false Input is invalid.
*/
bool make_diskann_segment_budget(const diskann_segment_budget_input &input,
                                 diskann_segment_budget *budget);

}  // namespace vector_index

#endif  // SQL_VECTOR_DISKANN_SCHEDULER_INCLUDED
