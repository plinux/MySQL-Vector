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

#include "sql/vector/vector_diskann_scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "sql/vector/vector_segmented_search.h"

namespace vector_index {

namespace {

constexpr double k_bytes_per_gib = 1024.0 * 1024.0 * 1024.0;
constexpr uint64_t k_diskann_sector_size = 4096;
constexpr long double k_diskann_cache_expansion_rate = 1.2L;
constexpr uint64_t k_diskann_coordinate_alignment = 8;

bool valid_ratio(double value) { return value >= 0.0 && value <= 1.0; }

uint32_t clamp_u64_to_u32(uint64_t value) {
  return value > std::numeric_limits<uint32_t>::max()
             ? std::numeric_limits<uint32_t>::max()
             : static_cast<uint32_t>(value);
}

uint32_t clamp_pq_chunks(uint64_t chunks, uint32_t dimension) {
  const uint64_t adjusted = std::max<uint64_t>(1, chunks);
  return clamp_u64_to_u32(
      std::min<uint64_t>(
          {adjusted, dimension, k_max_diskann_disk_pq_dims}));
}

uint64_t ceil_to_u64(long double value) {
  if (value <= 0.0L) return 0;
  if (value >= static_cast<long double>(std::numeric_limits<uint64_t>::max()))
    return std::numeric_limits<uint64_t>::max();
  return static_cast<uint64_t>(std::ceil(value));
}

uint64_t floor_to_u64(long double value) {
  if (value <= 0.0L) return 0;
  if (value >= static_cast<long double>(std::numeric_limits<uint64_t>::max()))
    return std::numeric_limits<uint64_t>::max();
  return static_cast<uint64_t>(std::floor(value));
}

uint64_t diskann_neighbor_bytes(uint32_t max_degree) {
  return (static_cast<uint64_t>(max_degree) + 1) * sizeof(uint32_t);
}

uint64_t diskann_serialized_node_size(uint32_t dimension,
                                      uint32_t max_degree,
                                      uint32_t disk_pq_dims) {
  const uint64_t vector_bytes =
      disk_pq_dims == 0
          ? static_cast<uint64_t>(dimension) * sizeof(float)
          : std::min<uint64_t>(disk_pq_dims, dimension) * sizeof(uint8_t);
  return diskann_neighbor_bytes(max_degree) + vector_bytes;
}

uint64_t diskann_runtime_cache_node_size(uint32_t dimension,
                                         uint32_t max_degree) {
  const uint64_t aligned_dimension =
      (static_cast<uint64_t>(dimension) + k_diskann_coordinate_alignment - 1) /
      k_diskann_coordinate_alignment * k_diskann_coordinate_alignment;
  return diskann_neighbor_bytes(max_degree) + aligned_dimension * sizeof(float);
}

uint32_t diskann_max_disk_pq_dims_for_sector(uint32_t max_degree) {
  const uint64_t metadata_bytes =
      (static_cast<uint64_t>(max_degree) + 1) * sizeof(uint32_t);
  if (metadata_bytes >= k_diskann_sector_size) return 0;
  return clamp_u64_to_u32(k_diskann_sector_size - metadata_bytes);
}

bool diskann_node_fits_sector(uint32_t dimension, uint32_t max_degree,
                              uint32_t disk_pq_dims) {
  return diskann_serialized_node_size(dimension, max_degree, disk_pq_dims) <=
         k_diskann_sector_size;
}

uint32_t effective_disk_pq_dims(uint32_t dimension, uint32_t max_degree,
                                uint32_t disk_pq_dims) {
  if (dimension == 0) return 0;
  if (disk_pq_dims != 0) {
    return std::min<uint32_t>(
        {disk_pq_dims, dimension, k_max_diskann_disk_pq_dims});
  }
  if (diskann_node_fits_sector(dimension, max_degree, 0)) return 0;

  const uint32_t max_safe_dims =
      diskann_max_disk_pq_dims_for_sector(max_degree);
  if (max_safe_dims == 0) return 0;
  return std::min<uint32_t>(
      {dimension, max_safe_dims, k_max_diskann_disk_pq_dims});
}

uint64_t cache_nodes_from_budget(uint64_t cache_budget_size,
                                 uint64_t cached_node_size) {
  if (cache_budget_size == 0) return 0;
  return floor_to_u64(static_cast<long double>(cache_budget_size) /
                      (static_cast<long double>(cached_node_size) *
                       k_diskann_cache_expansion_rate));
}

uint32_t normalize_search_complexity(uint32_t manual_search_complexity) {
  return manual_search_complexity == 0 ? k_default_diskann_search_complexity
                                       : manual_search_complexity;
}

uint32_t profile_search_complexity(uint32_t fanout_segments,
                                   diskann_search_profile profile,
                                   diskann_search_profile_reason &reason) {
  const bool many_segments = fanout_segments >= 32;
  switch (profile) {
    case diskann_search_profile::kManual:
      reason = diskann_search_profile_reason::kManual;
      return 0;
    case diskann_search_profile::kFast:
      reason = diskann_search_profile_reason::kFast;
      return 800;
    case diskann_search_profile::kBalanced:
      if (many_segments) {
        reason = diskann_search_profile_reason::kBalancedManySegments;
        return 1200;
      }
      reason = diskann_search_profile_reason::kBalancedFewSegments;
      return 2400;
    case diskann_search_profile::kHighRecall:
      if (many_segments) {
        reason = diskann_search_profile_reason::kHighRecallManySegments;
        return 1600;
      }
      reason = diskann_search_profile_reason::kHighRecallFewSegments;
      return 3200;
  }
  reason = diskann_search_profile_reason::kManual;
  return 0;
}

}  // namespace

bool make_diskann_segment_budget(const diskann_segment_budget_input &input,
                                 diskann_segment_budget *budget) {
  if (budget == nullptr || input.dimension == 0 || input.row_count == 0 ||
      input.payload_size == 0 || !valid_ratio(input.pq_code_budget_ratio) ||
      !valid_ratio(input.search_cache_ratio)) {
    return false;
  }

  uint64_t pq_code_budget_size = input.pq_code_budget_size;
  if (pq_code_budget_size == 0) {
    pq_code_budget_size =
        ceil_to_u64(static_cast<long double>(input.payload_size) *
                    static_cast<long double>(input.pq_code_budget_ratio));
  }

  uint64_t chunks = input.disk_pq_dims;
  if (chunks == 0) {
    chunks = pq_code_budget_size / input.row_count;
  }

  const uint32_t disk_pq_dims = effective_disk_pq_dims(
      input.dimension, input.max_degree, input.disk_pq_dims);
  if (!diskann_node_fits_sector(input.dimension, input.max_degree,
                                disk_pq_dims)) {
    return false;
  }
  const uint32_t pq_chunks =
      disk_pq_dims == 0 ? clamp_pq_chunks(chunks, input.dimension)
                        : disk_pq_dims;

  uint64_t cache_nodes = 0;
  const uint64_t cached_node_size =
      diskann_runtime_cache_node_size(input.dimension, input.max_degree);
  if (input.search_cache_size != 0) {
    cache_nodes =
        cache_nodes_from_budget(input.search_cache_size, cached_node_size);
  } else {
    const uint64_t derived_cache_size =
        ceil_to_u64(static_cast<long double>(input.payload_size) *
                    static_cast<long double>(input.search_cache_ratio));
    cache_nodes = cache_nodes_from_budget(derived_cache_size, cached_node_size);
  }
  cache_nodes = std::min<uint64_t>(cache_nodes, input.row_count);

  budget->pq_chunks = pq_chunks;
  budget->disk_pq_dims = disk_pq_dims;
  budget->pq_code_budget_size = pq_code_budget_size;
  budget->pq_code_budget_gb =
      static_cast<double>(pq_code_budget_size) / k_bytes_per_gib;
  budget->cache_nodes = clamp_u64_to_u32(cache_nodes);
  const uint64_t build_memory_size =
      input.build_memory_size != 0
          ? input.build_memory_size
          : (input.available_build_memory_size != 0
                 ? input.available_build_memory_size
                 : input.payload_size);
  budget->build_memory_size = build_memory_size;
  budget->build_memory_gb =
      static_cast<double>(build_memory_size) / k_bytes_per_gib;
  return budget->pq_chunks != 0;
}

bool choose_diskann_search_budget(uint32_t requested_top_k,
                                  uint32_t fanout_segments,
                                  uint64_t segment_min_entries [[maybe_unused]],
                                  uint64_t segment_max_entries,
                                  uint32_t manual_search_complexity,
                                  diskann_search_profile profile,
                                  diskann_search_budget *budget) {
  if (budget == nullptr || requested_top_k == 0 || fanout_segments == 0 ||
      segment_max_entries == 0) {
    return false;
  }

  diskann_search_profile_reason reason = diskann_search_profile_reason::kManual;
  const uint32_t manual_complexity =
      normalize_search_complexity(manual_search_complexity);
  uint32_t effective_complexity = manual_complexity;
  const uint32_t profile_complexity =
      profile_search_complexity(fanout_segments, profile, reason);
  if (profile != diskann_search_profile::kManual &&
      profile_complexity > effective_complexity) {
    effective_complexity = profile_complexity;
  } else if (profile != diskann_search_profile::kManual &&
             manual_complexity >= profile_complexity) {
    reason = diskann_search_profile_reason::kManualHigher;
  }
  const uint32_t min_complexity =
      requested_top_k == std::numeric_limits<uint32_t>::max()
          ? requested_top_k
          : requested_top_k + 1;
  effective_complexity = std::max<uint32_t>(effective_complexity,
                                            min_complexity);

  const uint32_t per_segment_top_k =
      diskann_search_list_slack_top_k(requested_top_k, effective_complexity,
                                      fanout_segments);
  const uint64_t candidate_count =
      static_cast<uint64_t>(per_segment_top_k) * fanout_segments;
  budget->search_complexity = effective_complexity;
  budget->per_segment_top_k = per_segment_top_k;
  budget->candidate_count = clamp_u64_to_u32(candidate_count);
  budget->profile = profile;
  budget->reason = reason;
  return true;
}

const char *diskann_search_profile_name(diskann_search_profile profile) {
  switch (profile) {
    case diskann_search_profile::kManual:
      return "manual";
    case diskann_search_profile::kFast:
      return "fast";
    case diskann_search_profile::kBalanced:
      return "balanced";
    case diskann_search_profile::kHighRecall:
      return "high_recall";
  }
  return "manual";
}

const char *diskann_search_profile_reason_name(
    diskann_search_profile_reason reason) {
  switch (reason) {
    case diskann_search_profile_reason::kManual:
      return "manual";
    case diskann_search_profile_reason::kFast:
      return "fast_800";
    case diskann_search_profile_reason::kBalancedManySegments:
      return "balanced_many_segments_1200";
    case diskann_search_profile_reason::kBalancedFewSegments:
      return "balanced_few_segments_2400";
    case diskann_search_profile_reason::kHighRecallManySegments:
      return "high_recall_many_segments_1600";
    case diskann_search_profile_reason::kHighRecallFewSegments:
      return "high_recall_few_segments_3200";
    case diskann_search_profile_reason::kManualHigher:
      return "manual_higher";
  }
  return "manual";
}

}  // namespace vector_index
