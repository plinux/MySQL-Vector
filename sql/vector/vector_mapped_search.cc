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

#include "sql/vector/vector_mapped_search.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lex_string.h"
#include "m_ctype.h"
#include "my_byteorder.h"
#include "sql/sql_class.h"
#include "sql/sql_lex.h"
#include "sql/log.h"
#include "sql/sql_prepare.h"
#include "sql/vector/vector_utils.h"
#include "sql_string.h"

namespace {

double compute_distance(const vector_index::vector_data &lhs,
                        const vector_index::vector_data &rhs,
                        vector_index::metric_type metric) {
  if (metric == vector_index::metric_type::kEuclidean) {
    double sum = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
      const double diff =
          static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]);
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }

  double dot = 0.0;
  if (metric == vector_index::metric_type::kInnerProduct) {
    for (size_t i = 0; i < lhs.size(); ++i) {
      dot += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
    }
    return -dot;
  }

  double lhs_norm = 0.0;
  double rhs_norm = 0.0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const double l = static_cast<double>(lhs[i]);
    const double r = static_cast<double>(rhs[i]);
    dot += l * r;
    lhs_norm += l * l;
    rhs_norm += r * r;
  }
  if (lhs_norm <= 0.0 || rhs_norm <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return 1.0 - (dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm)));
}

bool decode_binary_vector(const String *value, vector_index::vector_data *vector) {
  if (value == nullptr || vector == nullptr) return false;

  size_t dim = 0;
  if (!vector_utils::parse_binary_vector(value, &dim)) {
    std::vector<float> parsed;
    if (!vector_utils::parse_text_vector(value, &parsed)) return false;
    vector->assign(parsed.begin(), parsed.end());
    return true;
  }

  vector->clear();
  vector->reserve(dim);
  const uchar *ptr = reinterpret_cast<const uchar *>(value->ptr());
  for (size_t i = 0; i < dim; ++i) {
    vector->push_back(float4get(ptr + i * vector_utils::kVectorElemSize));
  }
  return true;
}

bool parse_doc_id_column(const Ed_column *column, uint64_t *value) {
  if (column == nullptr || value == nullptr || column->str == nullptr) return false;

  // Protocol_local stores integer result columns as host-order binary values.
  const uchar *ptr = reinterpret_cast<const uchar *>(column->str);
  switch (column->length) {
    case 1:
      *value = *ptr;
      return true;
    case 2: {
      uint16_t decoded = 0;
      std::memcpy(&decoded, ptr, sizeof(decoded));
      *value = decoded;
      return true;
    }
    case 3:
      *value = uint3korr(ptr);
      return true;
    case 4: {
      uint32_t decoded = 0;
      std::memcpy(&decoded, ptr, sizeof(decoded));
      *value = decoded;
      return true;
    }
    case 8: {
      uint64_t decoded = 0;
      std::memcpy(&decoded, ptr, sizeof(decoded));
      *value = decoded;
      return true;
    }
    default:
      return false;
  }
}

std::string quote_identifier(const std::string &identifier) {
  std::string quoted("`");
  for (char ch : identifier) {
    if (ch == '`') quoted.push_back('`');
    quoted.push_back(ch);
  }
  quoted.push_back('`');
  return quoted;
}

class scoped_mapped_search_context final {
 public:
  explicit scoped_mapped_search_context(THD *thd) : m_thd(thd) {}

  ~scoped_mapped_search_context() {
    if (m_opened) {
      m_thd->restore_backup_open_tables_state(&m_open_tables_backup);
      m_thd->lex->restore_backup_query_tables_list(&m_query_tables_backup);
    }
  }

  bool activate() {
    if (m_thd == nullptr || m_thd->lex == nullptr) return false;

    m_thd->lex->reset_n_backup_query_tables_list(&m_query_tables_backup);
    m_thd->reset_n_backup_open_tables_state(&m_open_tables_backup, 0);
    m_opened = true;
    return true;
  }

 private:
  THD *m_thd{nullptr};
  Query_tables_list m_query_tables_backup;
  Open_tables_backup m_open_tables_backup;
  bool m_opened{false};
};

bool collect_visible_rows(THD *thd, const vector_mapped_search::search_spec &spec,
                          const std::vector<vector_index::search_result> &candidates,
                          std::vector<vector_mapped_search::visible_candidate> *rows) {
  if (thd == nullptr || rows == nullptr) return false;
  rows->clear();
  if (candidates.empty()) return true;

  std::unordered_set<uint64_t> seen_doc_ids;
  std::string in_list;
  for (const auto &candidate : candidates) {
    if (!seen_doc_ids.insert(candidate.doc_id).second) continue;
    if (!in_list.empty()) in_list.append(",");
    in_list.append(std::to_string(candidate.doc_id));
  }
  if (in_list.empty()) return true;

  scoped_mapped_search_context scoped_context(thd);
  if (!scoped_context.activate()) return false;

  const std::string sql = "SELECT " + quote_identifier(spec.doc_id_column_name) + ", " +
                          quote_identifier(spec.column_name) + " FROM " +
                          quote_identifier(spec.schema_name) + "." +
                          quote_identifier(spec.table_name) + " WHERE " +
                          quote_identifier(spec.doc_id_column_name) + " IN (" +
                          in_list +
                          ")";

  Ed_connection connection(thd);
  LEX_STRING query{const_cast<char *>(sql.c_str()), sql.length()};
  if (connection.execute_direct(query)) {
    sql_print_error(
        "vector_mapped_search execute_direct failed errno=%u msg=%s sql=%s",
        connection.get_last_errno(), connection.get_last_error(), sql.c_str());
    return false;
  }

  Ed_result_set *result_set = connection.get_result_sets();
  if (result_set == nullptr || result_set->size() == 0 ||
      result_set->get_field_count() < 2) {
    sql_print_error("vector_mapped_search empty result sql=%s", sql.c_str());
    return true;
  }

  rows->reserve(result_set->size());
  List_iterator<Ed_row> row_it(static_cast<List<Ed_row> &>(*result_set));
  while (Ed_row *row = row_it++) {
    const Ed_column *doc_id_column = row->get_column(0);
    const Ed_column *vector_column = row->get_column(1);
    uint64_t doc_id = 0;
    if (!parse_doc_id_column(doc_id_column, &doc_id) || vector_column == nullptr ||
        vector_column->str == nullptr) {
      sql_print_error(
          "vector_mapped_search row decode failed sql=%s doc_id_ptr=%p doc_id_len=%lu vector_ptr=%p vector_len=%lu",
          sql.c_str(), doc_id_column != nullptr ? doc_id_column->str : nullptr,
          static_cast<unsigned long>(doc_id_column != nullptr ? doc_id_column->length : 0),
          vector_column != nullptr ? vector_column->str : nullptr,
          static_cast<unsigned long>(vector_column != nullptr ? vector_column->length : 0));
      return false;
    }

    String value;
    value.set(vector_column->str, vector_column->length, &my_charset_bin);
    vector_index::vector_data visible_vector;
    if (!decode_binary_vector(&value, &visible_vector)) {
      sql_print_error("vector_mapped_search vector decode failed sql=%s",
                      sql.c_str());
      return false;
    }
    rows->push_back(
        vector_mapped_search::visible_candidate{doc_id, std::move(visible_vector)});
  }
  return true;
}

}  // namespace

namespace vector_mapped_search {

bool rank_visible_candidates(const vector_index::vector_data &query,
                           vector_index::metric_type metric,
                           const std::vector<visible_candidate> &visible_rows,
                           size_t top_k,
                           std::vector<vector_index::search_result> *results) {
  if (results == nullptr) return false;
  results->clear();

  std::vector<vector_index::search_result> reranked;
  reranked.reserve(visible_rows.size());
  for (const auto &row : visible_rows) {
    if (row.vector.size() != query.size()) return false;
    reranked.push_back(vector_index::search_result{
        row.doc_id, compute_distance(row.vector, query, metric)});
  }

  std::sort(reranked.begin(), reranked.end(),
            [](const vector_index::search_result &lhs,
               const vector_index::search_result &rhs) {
              if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
              return lhs.doc_id < rhs.doc_id;
            });
  if (reranked.size() > top_k) reranked.resize(top_k);
  *results = std::move(reranked);
  return true;
}

bool filter_visible_results(THD *thd, const search_spec &spec,
                          const vector_index::vector_data &query,
                          std::vector<vector_index::search_result> *results) {
  if (results == nullptr) return false;
  if (results->empty()) return true;

  std::vector<visible_candidate> visible_rows;
  if (!collect_visible_rows(thd, spec, *results, &visible_rows)) return false;
  return rank_visible_candidates(query, spec.metric, visible_rows, spec.top_k,
                               results);
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool decode_binary_vector_for_testing(const String *value,
                                      vector_index::vector_data *vector) {
  return decode_binary_vector(value, vector);
}

bool collect_visible_rows_for_testing(
    THD *thd, const search_spec &spec,
    const std::vector<vector_index::search_result> &candidates,
    std::vector<visible_candidate> *rows) {
  return collect_visible_rows(thd, spec, candidates, rows);
}

bool activate_scoped_context_for_testing(THD *thd) {
  scoped_mapped_search_context scoped_context(thd);
  return scoped_context.activate();
}

bool parse_null_doc_id_column_for_testing(uint64_t *value) {
  return parse_doc_id_column(nullptr, value);
}

bool parse_doc_id_column_for_testing(const char *data, size_t length,
                                     uint64_t *value) {
  Ed_column column;
  column.str = const_cast<char *>(data);
  column.length = length;
  return parse_doc_id_column(&column, value);
}

std::string quote_identifier_for_testing(const std::string &identifier) {
  return quote_identifier(identifier);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_mapped_search
