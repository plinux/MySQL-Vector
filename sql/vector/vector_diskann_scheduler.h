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

enum class diskann_search_profile : uint32_t {
  kManual = 0,
  kFast = 1,
  kBalanced = 2,
  kHighRecall = 3
};

enum class diskann_search_profile_reason : uint32_t {
  kManual = 0,
  kFast = 1,
  kBalancedManySegments = 2,
  kBalancedFewSegments = 3,
  kHighRecallManySegments = 4,
  kHighRecallFewSegments = 5,
  kManualHigher = 6
};

struct diskann_search_budget {
  uint32_t search_complexity{0};
  uint32_t per_segment_top_k{0};
  uint32_t candidate_count{0};
  diskann_search_profile profile{diskann_search_profile::kManual};
  diskann_search_profile_reason reason{diskann_search_profile_reason::kManual};
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

/**
  Derive the effective segmented DiskANN search budget.

  @param requested_top_k SQL topK requested by the caller.
  @param fanout_segments Number of segments searched by the fan-out runtime.
  @param segment_min_entries Minimum entry count across segments.
  @param segment_max_entries Maximum entry count across segments.
  @param manual_search_complexity Search list requested by index/global config.
  @param profile Search profile selected by global policy.
  @param budget Output effective search budget.

  @retval true A valid budget was produced.
  @retval false Input is invalid.
*/
bool choose_diskann_search_budget(uint32_t requested_top_k,
                                  uint32_t fanout_segments,
                                  uint64_t segment_min_entries,
                                  uint64_t segment_max_entries,
                                  uint32_t manual_search_complexity,
                                  diskann_search_profile profile,
                                  diskann_search_budget *budget);

/** Return the SQL name for a DiskANN search profile. */
const char *diskann_search_profile_name(diskann_search_profile profile);

/** Return the diagnostic reason name for a DiskANN search profile choice. */
const char *diskann_search_profile_reason_name(
    diskann_search_profile_reason reason);

}  // namespace vector_index

#endif  // SQL_VECTOR_DISKANN_SCHEDULER_INCLUDED
