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

#ifndef SQL_VECTOR_MAPPED_SEARCH_INCLUDED
#define SQL_VECTOR_MAPPED_SEARCH_INCLUDED

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sql/vector/vector_index_backend.h"

class THD;
class String;

namespace vector_mapped_search {

struct search_spec {
  std::string schema_name;
  std::string table_name;
  std::string column_name;
  std::string doc_id_column_name;
  vector_index::metric_type metric{vector_index::metric_type::kEuclidean};
  size_t top_k{0};
};

struct visible_candidate {
  uint64_t doc_id{0};
  vector_index::vector_data vector;
};

/**
  Rank visible row versions by exact distance and keep the top-k results.

  This helper is pure and is used both by mapped-search runtime code and
  unit tests.
*/
bool rank_visible_candidates(const vector_index::vector_data &query,
                           vector_index::metric_type metric,
                           const std::vector<visible_candidate> &visible_rows,
                           size_t top_k,
                           std::vector<vector_index::search_result> *results);

/**
  Rank each query's visible ANN candidates without mixing candidate sets.

  The visible rows are shared across the batch so callers can perform one
  MVCC lookup for the union of candidate document identifiers.
*/
bool rank_visible_candidate_batches(
    const std::vector<vector_index::vector_data> &queries,
    vector_index::metric_type metric,
    const std::vector<std::vector<vector_index::search_result>> &candidates,
    const std::vector<visible_candidate> &visible_rows, size_t top_k,
    std::vector<std::vector<vector_index::search_result>> *results);

/**
  Filter ANN candidates through the current THD read view and recompute
  exact distance from the visible row version.
*/
bool filter_visible_results(THD *thd, const search_spec &spec,
                          const vector_index::vector_data &query,
                          std::vector<vector_index::search_result> *results);

/**
  Filter a batch of ANN candidate sets through one current-THD MVCC lookup.
*/
bool filter_visible_results_batch(
    THD *thd, const search_spec &spec,
    const std::vector<vector_index::vector_data> &queries,
    const std::vector<std::vector<vector_index::search_result>> &candidates,
    std::vector<std::vector<vector_index::search_result>> *results);

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool decode_binary_vector_for_testing(const String *value,
                                      vector_index::vector_data *vector);
bool collect_visible_rows_for_testing(
    THD *thd, const search_spec &spec,
    const std::vector<vector_index::search_result> &candidates,
    std::vector<visible_candidate> *rows);
bool activate_scoped_context_for_testing(THD *thd);
bool parse_null_doc_id_column_for_testing(uint64_t *value);
bool parse_doc_id_column_for_testing(const char *data, size_t length,
                                     uint64_t *value);
std::string quote_identifier_for_testing(const std::string &identifier);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_mapped_search

#endif  // SQL_VECTOR_MAPPED_SEARCH_INCLUDED
