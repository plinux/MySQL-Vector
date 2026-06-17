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

#ifndef SQL_VECTOR_BUILD_PIPELINE_POLICY_INCLUDED
#define SQL_VECTOR_BUILD_PIPELINE_POLICY_INCLUDED

#include <cstdint>

namespace vector_index {

inline constexpr uint64_t k_default_build_pipeline_min_rows = 1048576ULL;
inline constexpr uint64_t k_default_build_pipeline_min_size =
    4ULL * 1024ULL * 1024ULL * 1024ULL;
inline constexpr uint64_t k_default_build_segment_max_rows = 1048576ULL;
inline constexpr uint64_t k_min_build_segment_target_size =
    1ULL * 1024ULL * 1024ULL;
inline constexpr uint64_t k_default_build_segment_target_size =
    8ULL * 1024ULL * 1024ULL * 1024ULL;
inline constexpr uint32_t k_default_build_pipeline_max_tasks = 1;
inline constexpr uint32_t k_max_build_pipeline_max_tasks = 65535;
inline constexpr uint32_t k_default_build_pipeline_progress_interval = 10;
inline constexpr uint32_t k_max_build_pipeline_progress_interval = 3600;

enum class build_pipeline_mode {
  kAuto,
  kDirect,
  kSegmented,
};

enum class build_pipeline_path {
  kDirect,
  kSegmented,
};

enum class build_pipeline_trigger {
  kForcedDirect,
  kForcedSegmented,
  kRawSegmentCount,
  kRowCount,
  kPayloadSize,
  kBelowThreshold,
};

struct build_input_stats {
  uint64_t row_count{0};
  uint64_t payload_size{0};
  uint64_t raw_segment_count{0};
  bool has_raw_segments{false};
};

struct build_pipeline_thresholds {
  build_pipeline_mode mode{build_pipeline_mode::kAuto};
  uint64_t min_rows{k_default_build_pipeline_min_rows};
  uint64_t min_size{k_default_build_pipeline_min_size};
  uint64_t segment_max_rows{k_default_build_segment_max_rows};
  uint64_t segment_target_size{k_default_build_segment_target_size};
  uint32_t max_tasks{k_default_build_pipeline_max_tasks};
};

struct build_pipeline_decision {
  build_pipeline_path path{build_pipeline_path::kDirect};
  build_pipeline_trigger trigger{build_pipeline_trigger::kBelowThreshold};
};

/**
  Compute the maximum rows allowed in one raw build segment.

  The row limit is constrained by both row count and byte target. The byte
  target accounts for vector payload plus the persisted document id stream.
*/
uint64_t build_segment_row_limit(uint64_t dimension,
                                 const build_pipeline_thresholds &thresholds);

/**
  Select the build pipeline from input statistics and thresholds.

  This function is intentionally backend-independent. Provider-specific code
  must stay behind HAVE_FAISS, HAVE_HNSWLIB or HAVE_DISKANN guards.
*/
build_pipeline_decision select_build_pipeline(
    const build_input_stats &input,
    const build_pipeline_thresholds &thresholds);

/** Return a stable SQL/status name for the configured mode. */
const char *build_pipeline_mode_name(build_pipeline_mode mode);

/** Return a stable SQL/status name for the selected path. */
const char *build_pipeline_path_name(build_pipeline_path path);

/** Return a stable SQL/status name for the decision trigger. */
const char *build_pipeline_trigger_name(build_pipeline_trigger trigger);

}  // namespace vector_index

#endif  // SQL_VECTOR_BUILD_PIPELINE_POLICY_INCLUDED
