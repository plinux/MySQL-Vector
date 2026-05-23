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

#include <string>
#include <vector>

#include "my_dbug.h"
#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_status_fields.h"

namespace {

bool append_json_unsigned_number(String *number_buf, ulonglong value,
                                 std::string *out) {
  assert(number_buf != nullptr);
  assert(out != nullptr);
  DBUG_EXECUTE_IF("vector_item_fail_append_json_number", return false;);
  if (number_buf->set_int(static_cast<longlong>(value), true, &my_charset_bin))
    return false;
  out->append(number_buf->ptr(), number_buf->length());
  return true;
}

bool append_json_real(String *number_buf, double value, std::string *out) {
  assert(number_buf != nullptr);
  assert(out != nullptr);
  DBUG_EXECUTE_IF("vector_item_fail_append_json_real", return false;);
  if (number_buf->set_real(value, DECIMAL_NOT_SPECIFIED, &my_charset_bin))
    return false;
  out->append(number_buf->ptr(), number_buf->length());
  return true;
}

void append_json_quoted_string(std::string *out, const std::string &value) {
  assert(out != nullptr);
  out->push_back('"');
  out->append(value);
  out->push_back('"');
}

bool append_json_real_field(std::string *out, String *number_buf, const char *key,
                            double value) {
  assert(key != nullptr);
  assert(number_buf != nullptr);
  out->append(key);
  return append_json_real(number_buf, value, out);
}

bool append_json_search_result_object(std::string *out, String *number_buf,
                                      const vector_index::search_result &result) {
  assert(out != nullptr);
  out->append("{\"doc_id\":");
  if (!append_json_unsigned_number(number_buf, result.doc_id, out))
    return false;
  if (!append_json_real_field(out, number_buf, ",\"distance\":",
                              result.distance)) {
    return false;
  }
  out->push_back('}');
  return true;
}

bool append_json_doc_id_array(
    std::string *out, String *number_buf,
    const std::vector<vector_index::search_result> &results) {
  assert(out != nullptr);
  assert(number_buf != nullptr);
  out->push_back('[');
  for (size_t i = 0; i < results.size(); ++i) {
    if (i > 0) out->push_back(',');
    if (!append_json_unsigned_number(number_buf, results[i].doc_id, out)) {
      return false;
    }
  }
  out->push_back(']');
  return true;
}

bool append_json_status_field(
    std::string *out, String *number_buf,
    const vector_index_status_fields::field_value &field) {
  assert(out != nullptr);
  assert(number_buf != nullptr);
  assert(field.name != nullptr);

  out->push_back('"');
  out->append(field.name);
  out->append("\":");

  switch (field.kind) {
    case vector_index_status_fields::field_kind::k_string:
      append_json_quoted_string(out, field.string_value);
      return true;
    case vector_index_status_fields::field_kind::k_nullable_string:
      if (field.string_value.empty()) {
        out->append("null");
      } else {
        append_json_quoted_string(out, field.string_value);
      }
      return true;
    case vector_index_status_fields::field_kind::k_uint:
      return append_json_unsigned_number(number_buf, field.uint_value, out);
    case vector_index_status_fields::field_kind::k_bool:
      out->append(field.bool_value ? "1" : "0");
      return true;
  }

  return false;
}

}  // namespace

bool format_vector_index_info_json(
    const vector_index_registry::index_info &info, String *out,
    String *number_buf) {
  if (out == nullptr || number_buf == nullptr) return false;

  vector_index_status_fields::field_values fields;
  vector_index_status_fields::collect_info_fields(info, &fields);

  std::string json;
  json.reserve(1024);
  json.push_back('{');
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i > 0) json.push_back(',');
    if (!append_json_status_field(&json, number_buf, fields[i])) return false;
  }
  json.push_back('}');
  out->length(0);
  return !out->append(json.c_str(), json.length());
}

bool format_vector_index_list_json(
    const std::vector<std::string> &index_names, String *out) {
  if (out == nullptr) return false;
  std::string json;
  json.push_back('[');

  for (size_t i = 0; i < index_names.size(); ++i) {
    if (i > 0) json.push_back(',');
    append_json_quoted_string(&json, index_names[i]);
  }

  json.push_back(']');
  out->length(0);
  return !out->append(json.c_str(), json.length());
}

bool format_vector_search_result_doc_ids(
    const std::vector<vector_index::search_result> &results, String *out,
    String *number_buf) {
  if (out == nullptr || number_buf == nullptr) return false;
  std::string json;
  if (!append_json_doc_id_array(&json, number_buf, results)) return false;
  out->length(0);
  return !out->append(json.c_str(), json.length());
}

bool format_vector_search_result_batches(
    const std::vector<std::vector<vector_index::search_result>> &batches,
    String *out, String *number_buf) {
  if (out == nullptr || number_buf == nullptr) return false;
  std::string json;
  json.push_back('[');

  for (size_t i = 0; i < batches.size(); ++i) {
    if (i > 0) json.push_back(',');
    if (!append_json_doc_id_array(&json, number_buf, batches[i])) return false;
  }

  json.push_back(']');
  out->length(0);
  return !out->append(json.c_str(), json.length());
}

bool format_vector_search_results_with_distance(
    const std::vector<vector_index::search_result> &results, String *out,
    String *number_buf) {
  if (out == nullptr || number_buf == nullptr) return false;
  std::string json;
  json.push_back('[');

  for (size_t i = 0; i < results.size(); ++i) {
    if (i > 0) json.push_back(',');
    if (!append_json_search_result_object(&json, number_buf, results[i]))
      return false;
  }

  json.push_back(']');
  out->length(0);
  return !out->append(json.c_str(), json.length());
}
