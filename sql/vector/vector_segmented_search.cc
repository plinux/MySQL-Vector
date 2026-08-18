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

#include "sql/vector/vector_segmented_search.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "sql/vector/vector_index_limits.h"

namespace vector_index {

namespace {

bool search_result_less(const search_result &lhs, const search_result &rhs) {
  if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
  return lhs.doc_id < rhs.doc_id;
}

}  // namespace

size_t segmented_search_candidate_top_k(size_t top_k, size_t segment_count,
                                        size_t segment_entry_count,
                                        size_t query_count,
                                        size_t result_budget) {
  if (top_k == 0 || segment_count == 0 || segment_entry_count == 0) {
    return 0;
  }

  const size_t requested =
      std::max(top_k, saturated_mul_size(top_k, segment_count));
  size_t candidate_top_k =
      std::min({requested, segment_entry_count,
                static_cast<size_t>(k_max_search_top_k)});

  const size_t safe_query_count = std::max<size_t>(query_count, 1);
  const size_t budget_denominator =
      saturated_mul_size(safe_query_count, segment_count);
  if (result_budget > 0 && budget_denominator > 0) {
    const size_t budget_per_segment = result_budget / budget_denominator;
    if (budget_per_segment < top_k) {
      candidate_top_k = top_k;
    } else {
      candidate_top_k = std::min(candidate_top_k, budget_per_segment);
    }
  }

  if (segment_entry_count <= top_k) return segment_entry_count;
  return std::min(std::max(top_k, candidate_top_k), segment_entry_count);
}

size_t diskann_search_list_slack_top_k(size_t top_k, size_t search_complexity,
                                       size_t fanout_count) {
  if (top_k == 0 || search_complexity == 0) return top_k;
  if (search_complexity <= top_k) return top_k;

  const size_t safe_fanout_count = std::max<size_t>(fanout_count, 1);
  const size_t max_returned_candidates = search_complexity - 1;
  size_t slack_top_k = max_returned_candidates / safe_fanout_count;
  if (slack_top_k == 0) slack_top_k = top_k;
  return std::max(top_k, slack_top_k);
}

bool merge_segment_topk(
    const std::vector<std::vector<search_result>> &segment_results,
    size_t top_k, std::vector<search_result> *results) {
  if (results == nullptr) return false;
  results->clear();
  if (top_k == 0) return true;

  std::unordered_map<uint64_t, search_result> best_by_doc_id;
  for (const auto &segment_result : segment_results) {
    for (const search_result &candidate : segment_result) {
      const auto [it, inserted] =
          best_by_doc_id.emplace(candidate.doc_id, candidate);
      if (!inserted && search_result_less(candidate, it->second)) {
        it->second = candidate;
      }
    }
  }

  if (best_by_doc_id.size() <= top_k) {
    results->reserve(best_by_doc_id.size());
    for (const auto &entry : best_by_doc_id) {
      results->push_back(entry.second);
    }
    std::sort(results->begin(), results->end(), search_result_less);
    return true;
  }

  results->reserve(top_k);
  for (const auto &entry : best_by_doc_id) {
    const search_result &candidate = entry.second;
    if (results->size() < top_k) {
      results->push_back(candidate);
      std::push_heap(results->begin(), results->end(), search_result_less);
      continue;
    }

    if (!search_result_less(candidate, results->front())) continue;

    std::pop_heap(results->begin(), results->end(), search_result_less);
    results->back() = candidate;
    std::push_heap(results->begin(), results->end(), search_result_less);
  }
  std::sort_heap(results->begin(), results->end(), search_result_less);
  return true;
}

bool merge_segment_batch_topk(
    const std::vector<std::vector<std::vector<search_result>>>
        &segment_batch_results,
    size_t top_k, std::vector<std::vector<search_result>> *results) {
  if (results == nullptr) return false;
  results->clear();

  size_t query_count = 0;
  if (!segment_batch_results.empty()) {
    query_count = segment_batch_results.front().size();
  }
  for (const auto &segment_results : segment_batch_results) {
    if (segment_results.size() != query_count) return false;
  }

  results->resize(query_count);
  if (top_k == 0) return true;

  std::vector<std::vector<search_result>> per_query_segments;
  per_query_segments.reserve(segment_batch_results.size());
  for (size_t query_index = 0; query_index < query_count; ++query_index) {
    per_query_segments.clear();
    for (const auto &segment_results : segment_batch_results) {
      per_query_segments.push_back(segment_results[query_index]);
    }
    merge_segment_topk(per_query_segments, top_k, &(*results)[query_index]);
  }
  return true;
}

}  // namespace vector_index
