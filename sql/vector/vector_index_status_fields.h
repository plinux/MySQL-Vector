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

#ifndef SQL_VECTOR_INDEX_STATUS_FIELDS_INCLUDED
#define SQL_VECTOR_INDEX_STATUS_FIELDS_INCLUDED

#include <cstdint>
#include <string>
#include <vector>

#include "sql/vector/vector_index_registry.h"

namespace vector_index_status_fields {

/** Output type used by JSON and SHOW VECTOR STATUS renderers. */
enum class field_kind {
  k_string,
  k_nullable_string,
  k_uint,
  k_bool,
};

/** One index status field with its stable public name and typed value. */
struct field_value {
  const char *name{nullptr};
  field_kind kind{field_kind::k_string};
  std::string string_value;
  uint64_t uint_value{0};
  bool bool_value{false};
};

using field_values = std::vector<field_value>;

/** Return the non-negative difference between serving and committed rows. */
uint64_t pending_apply_count(const vector_index_registry::index_info &info);

/** Return the user-visible rebuild progress percentage. */
uint64_t rebuild_progress(const vector_index_registry::index_info &info);

/** Return the user-visible recovery progress percentage. */
uint64_t recover_progress(const vector_index_registry::index_info &info);

/** Return whether the backend is considered loaded for status reporting. */
bool is_loaded(const vector_index_registry::index_info &info);

/** Return whether the backend currently accepts online mutations. */
bool is_writable(const vector_index_registry::index_info &info);

/** Convert a typed field to the string value used by SHOW VECTOR STATUS. */
std::string status_value(const field_value &field);

/** Fill the complete VEC_INDEX_INFO() JSON field set in stable order. */
void collect_info_fields(const vector_index_registry::index_info &info,
                         field_values *fields);

/** Fill the SHOW VECTOR STATUS INDEX_STATE field set. */
void collect_index_state_fields(const vector_index_registry::index_info &info,
                                field_values *fields);

/** Fill the SHOW VECTOR STATUS BACKEND_HEALTH field set. */
void collect_backend_health_fields(
    const vector_index_registry::index_info &info, field_values *fields);

/** Fill the SHOW VECTOR STATUS SYNC_PIPELINE field set. */
void collect_sync_pipeline_fields(const vector_index_registry::index_info &info,
                                  field_values *fields);

}  // namespace vector_index_status_fields

#endif  // SQL_VECTOR_INDEX_STATUS_FIELDS_INCLUDED
