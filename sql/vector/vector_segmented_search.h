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
  Merge per-segment topK results into one globally ordered result set.

  Distances use the same ascending semantics as backend::search(). Duplicate
  document ids can appear when a future segmented executor replays replacement
  segments; the merge keeps the best distance for each document id and applies
  the standard doc-id tie breaker.
*/
bool merge_segment_topk(
    const std::vector<std::vector<search_result>> &segment_results,
    size_t top_k, std::vector<search_result> *results);

}  // namespace vector_index

#endif  // SQL_VECTOR_SEGMENTED_SEARCH_INCLUDED
