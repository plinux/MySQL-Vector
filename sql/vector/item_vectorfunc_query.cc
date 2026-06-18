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

#include "sql/vector/item_vectorfunc.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "my_byteorder.h"
#include "mysqld_error.h"
#include "sql/sql_class.h"
#include "sql/vector/item_vectorfunc_internal.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_utils.h"

using namespace vector_itemfunc_internal;

namespace {

bool decode_vector_batch_arg(Item *arg, String *buf, size_t query_count,
                             std::vector<vector_index::vector_data> *queries) {
  if (queries == nullptr || query_count == 0) return false;

  const String *value = nullptr;
  if (!eval_vector_arg(arg, buf, &value)) return false;

  const size_t byte_length = value->length();
  if (byte_length == 0 || byte_length % query_count != 0) return false;

  const size_t bytes_per_query = byte_length / query_count;
  if (bytes_per_query == 0 ||
      bytes_per_query % vector_utils::kVectorElemSize != 0) {
    return false;
  }

  const size_t dim = bytes_per_query / vector_utils::kVectorElemSize;
  const uchar *ptr = reinterpret_cast<const uchar *>(value->ptr());
  queries->clear();
  queries->reserve(query_count);
  for (size_t query_idx = 0; query_idx < query_count; ++query_idx) {
    vector_index::vector_data query;
    query.reserve(dim);
    const uchar *query_ptr = ptr + query_idx * bytes_per_query;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      query.push_back(
          float4get(query_ptr + dim_idx * vector_utils::kVectorElemSize));
    }
    queries->push_back(std::move(query));
  }
  return true;
}

bool parse_search_top_k(ulonglong top_k_value, size_t *top_k) {
  if (top_k == nullptr || top_k_value > vector_index::k_max_search_top_k) {
    return false;
  }

  *top_k = static_cast<size_t>(top_k_value);
  return true;
}

bool parse_search_batch_bounds(ulonglong query_count_value,
                               ulonglong top_k_value,
                               size_t *query_count, size_t *top_k) {
  if (query_count == nullptr || query_count_value == 0 ||
      query_count_value > opt_vector_search_batch_count) {
    return false;
  }
  if (!parse_search_top_k(top_k_value, top_k)) return false;

  const size_t parsed_query_count = static_cast<size_t>(query_count_value);
  if (*top_k != 0 && parsed_query_count >
                         opt_vector_search_batch_result_count / *top_k) {
    return false;
  }

  *query_count = parsed_query_count;
  return true;
}

}  // namespace

bool Item_func_vec_index_info::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  const uint32 max_chars =
      static_cast<uint32>(MAX_BLOB_WIDTH / default_charset()->mbmaxlen);
  set_data_type_string(max_chars, default_charset());
  set_nullable(true);
  return false;
}

String *Item_func_vec_index_info::val_str(String *str [[maybe_unused]]) {
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  vector_index_registry::index_info info;
  if (!vector_index_registry::get_index_info(index_name, &info)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }
  if (check_vector_index_access(current_thd, info, SELECT_ACL)) {
    return error_str();
  }

  m_value.set_charset(default_charset());
  if (!format_vector_index_info_json(info, &m_value, &m_number_buf)) {
    return error_str();
  }

  null_value = false;
  return &m_value;
}

bool Item_func_vec_index_list::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 0)) return true;
  const uint32 max_chars =
      static_cast<uint32>(MAX_BLOB_WIDTH / default_charset()->mbmaxlen);
  set_data_type_string(max_chars, default_charset());
  set_nullable(false);
  return false;
}

String *Item_func_vec_index_list::val_str(String *str [[maybe_unused]]) {
  assert_fixed_arg_count(fixed, arg_count, 0);
  null_value = true;

  std::vector<std::string> index_names;
  if (!vector_index_registry::list_indexes(&index_names)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  std::vector<std::string> visible_index_names;
  visible_index_names.reserve(index_names.size());
  for (const std::string &index_name : index_names) {
    vector_index_registry::index_info info;
    if (vector_index_registry::get_index_info(index_name, &info) &&
        has_vector_index_access(current_thd, info, SELECT_ACL)) {
      visible_index_names.push_back(index_name);
    }
  }

  m_value.set_charset(default_charset());
  if (!format_vector_index_list_json(visible_index_names, &m_value)) {
    return error_str();
  }
  null_value = false;
  return &m_value;
}

bool Item_func_vec_index_search::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 3)) return true;
  const uint32 max_chars =
      static_cast<uint32>(MAX_BLOB_WIDTH / default_charset()->mbmaxlen);
  set_data_type_string(max_chars, default_charset());
  set_nullable(true);
  return false;
}

String *Item_func_vec_index_search::val_str(String *str [[maybe_unused]]) {
  assert_fixed_arg_count(fixed, arg_count, 3);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  ulonglong top_k_value = 0;
  size_t top_k = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint_arg(args[2], top_k_value) ||
      !parse_search_top_k(top_k_value, &top_k)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  vector_index::vector_data query;
  String query_buf;
  if (!decode_vector_arg(args[1], &query_buf, &query)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, SELECT_ACL, func_name(), false)) {
    return error_str();
  }

  std::vector<vector_index::search_result> results;
  if (!vector_index_registry::search_for_thd_txn(
          current_thd, static_cast<uint64_t>(current_thd->thread_id()),
          index_name, query, top_k, &results)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  m_value.set_charset(default_charset());
  if (!format_vector_search_result_doc_ids(results, &m_value, &m_number_buf)) {
    return error_str();
  }
  null_value = false;
  return &m_value;
}

bool Item_func_vec_index_search_batch::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 4)) return true;
  const uint32 max_chars =
      static_cast<uint32>(MAX_BLOB_WIDTH / default_charset()->mbmaxlen);
  set_data_type_string(max_chars, default_charset());
  set_nullable(true);
  return false;
}

String *Item_func_vec_index_search_batch::val_str(
    String *str [[maybe_unused]]) {
  assert_fixed_arg_count(fixed, arg_count, 4);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  ulonglong query_count_value = 0;
  ulonglong top_k_value = 0;
  size_t query_count = 0;
  size_t top_k = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint_arg(args[2], query_count_value) ||
      !eval_uint_arg(args[3], top_k_value) ||
      !parse_search_batch_bounds(query_count_value, top_k_value, &query_count,
                                 &top_k)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  std::vector<vector_index::vector_data> queries;
  String query_buf;
  if (!decode_vector_batch_arg(args[1], &query_buf, query_count, &queries)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, SELECT_ACL, func_name(), false)) {
    return error_str();
  }

  std::vector<std::vector<vector_index::search_result>> batches;
  if (!vector_index_registry::search_batch_for_thd_txn(
          current_thd, static_cast<uint64_t>(current_thd->thread_id()),
          index_name, queries, top_k, &batches)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  m_value.set_charset(default_charset());
  if (!format_vector_search_result_batches(batches, &m_value, &m_number_buf)) {
    return error_str();
  }
  null_value = false;
  return &m_value;
}

bool Item_func_vec_index_search_with_distance::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 3)) return true;
  const uint32 max_chars =
      static_cast<uint32>(MAX_BLOB_WIDTH / default_charset()->mbmaxlen);
  set_data_type_string(max_chars, default_charset());
  set_nullable(true);
  return false;
}

String *Item_func_vec_index_search_with_distance::val_str(
    String *str [[maybe_unused]]) {
  assert_fixed_arg_count(fixed, arg_count, 3);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  ulonglong top_k_value = 0;
  size_t top_k = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint_arg(args[2], top_k_value) ||
      !parse_search_top_k(top_k_value, &top_k)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  vector_index::vector_data query;
  String query_buf;
  if (!decode_vector_arg(args[1], &query_buf, &query)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, SELECT_ACL, func_name(), false)) {
    return error_str();
  }

  std::vector<vector_index::search_result> results;
  if (!vector_index_registry::search_for_thd_txn(
          current_thd, static_cast<uint64_t>(current_thd->thread_id()),
          index_name, query, top_k, &results)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  m_value.set_charset(default_charset());
  if (!format_vector_search_results_with_distance(results, &m_value,
                                                       &m_number_buf)) {
    return error_str();
  }
  null_value = false;
  return &m_value;
}
