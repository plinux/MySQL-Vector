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

#ifndef SQL_VECTOR_SEGMENTED_SEARCH_INCLUDED
#define SQL_VECTOR_SEGMENTED_SEARCH_INCLUDED

#include <cstddef>
#include <vector>

#include "sql/vector/vector_index_backend.h"

namespace vector_index {

/**
  Calculate the internal per-segment topK used by segmented search.

  The returned value is an internal ANN candidate window. It is allowed to be
  larger than the SQL-visible final topK, but it is capped by the segment size,
  backend topK limit and batch result budget to avoid unbounded memory growth.
*/
size_t segmented_search_candidate_top_k(size_t top_k, size_t segment_count,
                                        size_t segment_entry_count,
                                        size_t query_count,
                                        size_t result_budget);

/**
  Merge per-segment topK results into one globally ordered result set.

  Distances use the same ascending semantics as backend::search(). Duplicate
  document ids can appear when a future segmented executor replays replacement
  segments; the merge keeps the best distance for each document id and applies
  the standard doc-id tie breaker.
*/
bool merge_segment_topk(
    const std::vector<std::vector<search_result>> &segment_results,
    size_t top_k, std::vector<search_result> *results);

/**
  Merge per-segment batch topK results.

  The first dimension is segment, the second dimension is query, and the third
  dimension is the per-query search result list returned by that segment.
*/
bool merge_segment_batch_topk(
    const std::vector<std::vector<std::vector<search_result>>>
        &segment_batch_results,
    size_t top_k, std::vector<std::vector<search_result>> *results);

}  // namespace vector_index

#endif  // SQL_VECTOR_SEGMENTED_SEARCH_INCLUDED
