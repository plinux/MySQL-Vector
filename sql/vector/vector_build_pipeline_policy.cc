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

}  // namespace vector_index
