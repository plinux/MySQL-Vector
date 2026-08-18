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

#include "sql/vector/vector_index_observability.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <string>

namespace vector_index_observability {
namespace {

bool ascii_equal_ignore_case(const std::string &lhs, const char *rhs) {
  const size_t rhs_len = std::strlen(rhs);
  if (lhs.size() != rhs_len) return false;
  for (size_t i = 0; i < rhs_len; ++i) {
    const auto left = static_cast<unsigned char>(lhs[i]);
    const auto right = static_cast<unsigned char>(rhs[i]);
    if (std::tolower(left) != std::tolower(right)) return false;
  }
  return true;
}

vector_search_advice make_search_advice(const char *advice,
                                        const char *reason,
                                        uint64_t suggested_complexity,
                                        uint64_t suggested_segment_target_size) {
  vector_search_advice result;
  result.advice = advice;
  result.reason = reason;
  result.suggested_complexity = suggested_complexity;
  result.suggested_segment_target_size = suggested_segment_target_size;
  return result;
}

}  // namespace

uint64_t pending_apply_count(const vector_index_registry::index_info &info) {
  const auto entry_count = static_cast<uint64_t>(info.entry_count);
  const auto committed_entry_count =
      static_cast<uint64_t>(info.committed_entry_count);
  return entry_count > committed_entry_count
             ? entry_count - committed_entry_count
             : committed_entry_count - entry_count;
}

uint64_t rebuild_progress(const vector_index_registry::index_info &info) {
  if (ascii_equal_ignore_case(info.lifecycle_state, "ready")) return 100;
  if (ascii_equal_ignore_case(info.lifecycle_state, "rebuilding")) return 50;
  return 0;
}

uint64_t recover_progress(const vector_index_registry::index_info &info) {
  if (ascii_equal_ignore_case(info.lifecycle_state, "ready")) return 100;
  if (ascii_equal_ignore_case(info.lifecycle_state, "recovering")) return 50;
  return 0;
}

bool is_loaded(const vector_index_registry::index_info &info) {
  return !ascii_equal_ignore_case(info.lifecycle_state, "failed");
}

bool is_writable(const vector_index_registry::index_info &info) {
  return is_loaded(info) && info.supports_mutations;
}

vector_search_advice derive_diskann_search_advice(
    const vector_index::backend_build_diagnostics &diagnostics,
    uint32_t requested_top_k, uint64_t result_budget) {
  if (diagnostics.search_fanout_segments <= 1 ||
      diagnostics.search_query_count == 0 ||
      diagnostics.search_candidate_count == 0 ||
      diagnostics.search_diskann_search_list == 0) {
    return make_search_advice("not_applicable", "not_segmented_diskann_search",
                              0, 0);
  }

  const uint64_t required_result_budget =
      diagnostics.search_total_candidate_rows != 0
          ? diagnostics.search_total_candidate_rows
          : diagnostics.search_query_count * diagnostics.search_candidate_count;
  const uint64_t effective_result_budget =
      result_budget != 0 ? result_budget : diagnostics.search_result_budget;
  if (effective_result_budget != 0 &&
      effective_result_budget < required_result_budget) {
    return make_search_advice("increase_vector_search_batch_result_count",
                              "candidate_budget_truncates_batch",
                              diagnostics.search_effective_complexity, 0);
  }

  const uint64_t top_k =
      requested_top_k != 0 ? requested_top_k : diagnostics.search_global_top_k;
  const uint64_t low_candidate_window =
      std::max<uint64_t>(top_k * 2, 32);
  if (diagnostics.search_per_segment_top_k <= low_candidate_window ||
      diagnostics.search_candidate_count <=
          diagnostics.search_fanout_segments * low_candidate_window) {
    const uint64_t current_complexity =
        diagnostics.search_effective_complexity != 0
            ? diagnostics.search_effective_complexity
            : diagnostics.search_diskann_search_list;
    const uint64_t suggested_complexity =
        std::max<uint64_t>(current_complexity * 2, 800);
    return make_search_advice("increase_search_complexity",
                              "fanout_candidate_window_too_small",
                              suggested_complexity, 0);
  }

  if (!ascii_equal_ignore_case(diagnostics.search_profile, "high_recall") &&
      diagnostics.search_segment_max_entries >= 100000) {
    return make_search_advice(
        "use_smaller_segments_or_high_recall_profile",
        "large_segments_without_high_recall_profile",
        diagnostics.search_effective_complexity, 268435456);
  }

  return make_search_advice("no_action", "search_diagnostics_within_policy",
                            diagnostics.search_effective_complexity, 0);
}

}  // namespace vector_index_observability
