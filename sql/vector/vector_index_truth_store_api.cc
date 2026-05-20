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

#include "sql/vector/vector_index_truth_store.h"

#include <string>
#include <utility>
#include <vector>

#include "sql/vector/vector_index_truth_store_internal.h"

namespace vector_index_truth_store {

truth_store *get() { return detail::selected_backend(); }

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
void set_for_testing(truth_store *store) {
  detail::set_override_store_for_testing(store);
}

void reset_for_testing() { detail::reset_override_store_for_testing(); }
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

bool initialize_selected_backend() { return true; }

bool bootstrap_initialize_selected_backend(THD *thd) {
  return detail::bootstrap_initialize_mysql_store(thd);
}

void shutdown_selected_backend() { detail::shutdown_mysql_store(); }

bool internal_sql_active() { return detail::internal_sql_active(); }

bool internal_truth_store_access_allowed(const THD *thd) {
  return detail::internal_truth_store_access_allowed(thd);
}

const char *active_backend_name() {
  const char *name = get()->backend_name();
  return name != nullptr ? name : "unknown";
}

bool active_backend_transactional() { return get()->is_transactional(); }

bool is_truth_store_table(const char *schema_name, const char *table_name) {
  return detail::is_truth_store_table_name_impl(schema_name, table_name);
}

bool debug_get_artifact_payload(const std::string &artifact_name,
                             std::string *payload) {
  return detail::mysql_debug_get_artifact(artifact_name, payload);
}

bool debug_set_artifact_payload(const std::string &artifact_name,
                             const std::string &payload) {
  return detail::mysql_debug_set_artifact(artifact_name, payload);
}

bool debug_delete_artifact(const std::string &artifact_name) {
  return detail::mysql_debug_delete_artifact(artifact_name);
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
std::string sql_string_literal_for_testing(const char *text) {
  return detail::sql_string_literal_impl(text);
}

bool decode_hex_bytes_for_testing(const std::string &encoded,
                              std::string *decoded) {
  return detail::decode_hex_bytes_impl(encoded, decoded);
}

bool internal_execute_for_testing(const std::string &sql,
                               unsigned int *last_errno,
                               std::string *last_error) {
  return detail::internal_execute_impl(sql, last_errno, last_error);
}

bool internal_query_scalar_string_for_testing(const std::string &sql,
                                         std::string *value, bool *found) {
  return detail::internal_query_scalar_string_impl(sql, value, found);
}

bool deserialize_quarantine_entries_for_testing(
    const std::string &payload,
    std::vector<std::pair<std::string, std::string>> *entries) {
  return detail::deserialize_quarantine_entries_impl(payload, entries);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index_truth_store
