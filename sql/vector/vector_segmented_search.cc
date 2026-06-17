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

namespace vector_index {

namespace {

bool search_result_less(const search_result &lhs, const search_result &rhs) {
  if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
  return lhs.doc_id < rhs.doc_id;
}

}  // namespace

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

  results->reserve(best_by_doc_id.size());
  for (const auto &entry : best_by_doc_id) {
    results->push_back(entry.second);
  }
  std::sort(results->begin(), results->end(), search_result_less);
  if (results->size() > top_k) results->resize(top_k);
  return true;
}

}  // namespace vector_index
