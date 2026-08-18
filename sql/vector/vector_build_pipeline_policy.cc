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

#include "sql/vector/vector_build_pipeline_policy.h"

#include <algorithm>
#include <limits>

namespace vector_index {

namespace {

inline constexpr uint64_t k_diskann_high_recall_segment_target_size =
    256ULL * 1024ULL * 1024ULL;
inline constexpr uint64_t k_diskann_balanced_segment_target_size =
    512ULL * 1024ULL * 1024ULL;
inline constexpr uint64_t k_diskann_throughput_segment_target_size =
    k_default_build_segment_target_size;

uint64_t saturating_vector_row_size(uint64_t dimension) {
  if (dimension >
      (std::numeric_limits<uint64_t>::max() - sizeof(uint64_t)) /
          sizeof(float)) {
    return std::numeric_limits<uint64_t>::max();
  }
  return dimension * sizeof(float) + sizeof(uint64_t);
}

}  // namespace

uint64_t build_segment_row_limit(uint64_t dimension,
                                 const build_pipeline_thresholds &thresholds) {
  uint64_t row_limit = thresholds.segment_max_rows == 0
                           ? std::numeric_limits<uint64_t>::max()
                           : thresholds.segment_max_rows;
  const uint64_t row_size = saturating_vector_row_size(dimension);
  if (thresholds.segment_target_size != 0) {
    row_limit = std::min(
        row_limit, std::max<uint64_t>(1, thresholds.segment_target_size /
                                             row_size));
  }
  return std::max<uint64_t>(1, row_limit);
}

diskann_segment_profile_result apply_diskann_segment_profile(
    uint64_t dimension, uint64_t, uint64_t, diskann_segment_profile profile,
    const build_pipeline_thresholds &manual, bool eligible) {
  diskann_segment_profile_result result;
  result.thresholds = manual;

  if (!eligible) {
    result.reason = "not_applicable";
    return result;
  }

  if (manual.mode == build_pipeline_mode::kDirect) {
    result.reason = "forced_direct";
    return result;
  }

  switch (profile) {
    case diskann_segment_profile::kManual:
      result.reason = "manual";
      return result;
    case diskann_segment_profile::kThroughput:
      result.thresholds.segment_target_size = std::max<uint64_t>(
          manual.segment_target_size, k_diskann_throughput_segment_target_size);
      result.reason = "throughput_large_segment";
      break;
    case diskann_segment_profile::kBalanced:
      result.thresholds.segment_target_size = std::clamp<uint64_t>(
          k_diskann_balanced_segment_target_size,
          k_min_build_segment_target_size,
          k_default_build_segment_target_size);
      result.reason = "balanced_512m";
      break;
    case diskann_segment_profile::kHighRecall:
      result.thresholds.segment_target_size = std::clamp<uint64_t>(
          dimension >= 768 ? k_diskann_high_recall_segment_target_size
                           : k_diskann_balanced_segment_target_size,
          k_min_build_segment_target_size,
          k_default_build_segment_target_size);
      result.reason =
          dimension >= 768 ? "high_recall_256m" : "high_recall_512m";
      break;
  }

  result.applied = result.thresholds.segment_target_size !=
                   manual.segment_target_size;
  return result;
}

build_pipeline_decision select_build_pipeline(
    const build_input_stats &input,
    const build_pipeline_thresholds &thresholds) {
  switch (thresholds.mode) {
    case build_pipeline_mode::kDirect:
      return {build_pipeline_path::kDirect,
              build_pipeline_trigger::kForcedDirect};
    case build_pipeline_mode::kSegmented:
      return {build_pipeline_path::kSegmented,
              build_pipeline_trigger::kForcedSegmented};
    case build_pipeline_mode::kAuto:
      break;
  }

  if (input.has_raw_segments && input.raw_segment_count > 1) {
    return {build_pipeline_path::kSegmented,
            build_pipeline_trigger::kRawSegmentCount};
  }

  if (thresholds.min_rows != 0 && input.row_count >= thresholds.min_rows) {
    return {build_pipeline_path::kSegmented,
            build_pipeline_trigger::kRowCount};
  }

  if (thresholds.min_size != 0 && input.payload_size >= thresholds.min_size) {
    return {build_pipeline_path::kSegmented,
            build_pipeline_trigger::kPayloadSize};
  }

  return {build_pipeline_path::kDirect,
          build_pipeline_trigger::kBelowThreshold};
}

const char *build_pipeline_mode_name(build_pipeline_mode mode) {
  switch (mode) {
    case build_pipeline_mode::kAuto:
      return "auto";
    case build_pipeline_mode::kDirect:
      return "direct";
    case build_pipeline_mode::kSegmented:
      return "segmented";
  }
  return "unknown";
}

const char *build_pipeline_path_name(build_pipeline_path path) {
  switch (path) {
    case build_pipeline_path::kDirect:
      return "direct";
    case build_pipeline_path::kSegmented:
      return "segmented";
  }
  return "unknown";
}

const char *build_pipeline_trigger_name(build_pipeline_trigger trigger) {
  switch (trigger) {
    case build_pipeline_trigger::kForcedDirect:
      return "forced_direct";
    case build_pipeline_trigger::kForcedSegmented:
      return "forced_segmented";
    case build_pipeline_trigger::kRawSegmentCount:
      return "raw_segment_count";
    case build_pipeline_trigger::kRowCount:
      return "row_count";
    case build_pipeline_trigger::kPayloadSize:
      return "payload_size";
    case build_pipeline_trigger::kBelowThreshold:
      return "below_threshold";
  }
  return "unknown";
}

const char *diskann_segment_profile_name(diskann_segment_profile profile) {
  switch (profile) {
    case diskann_segment_profile::kManual:
      return "manual";
    case diskann_segment_profile::kThroughput:
      return "throughput";
    case diskann_segment_profile::kBalanced:
      return "balanced";
    case diskann_segment_profile::kHighRecall:
      return "high_recall";
  }
  return "unknown";
}

}  // namespace vector_index
