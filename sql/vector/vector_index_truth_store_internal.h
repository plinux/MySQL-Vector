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

#ifndef SQL_VECTOR_INDEX_TRUTH_STORE_INTERNAL_INCLUDED
#define SQL_VECTOR_INDEX_TRUTH_STORE_INTERNAL_INCLUDED

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "sql/vector/vector_index_truth_store.h"
#include "storage/innobase/include/dict0vectruth.h"

class THD;

namespace vector_index_truth_store::detail {

truth_store *selected_backend();
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
void set_override_store_for_testing(truth_store *store);
void reset_override_store_for_testing();
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
bool bootstrap_initialize_mysql_store(THD *thd);
void shutdown_mysql_store();
bool is_truth_store_table_name_impl(const char *schema_name,
                                    const char *table_name);
bool mysql_debug_get_artifact(const std::string &artifact_name,
                              std::string *payload);
bool mysql_debug_set_artifact(const std::string &artifact_name,
                              const std::string &payload);
bool mysql_debug_delete_artifact(const std::string &artifact_name);
std::string sql_string_literal_impl(const char *text);
bool decode_hex_bytes_impl(const std::string &encoded, std::string *decoded);
bool deserialize_quarantine_entries_impl(
    const std::string &payload, std::vector<quarantine_record> *entries);
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool split_tab_fields_impl(const std::string &line,
                           std::vector<std::string> *fields);
const char *quarantine_state_name_impl(quarantine_state state);
bool parse_quarantine_state_impl(const std::string &value,
                                 quarantine_state *state);
bool parse_uint64_impl(const std::string &value, uint64_t *result);
uint64_t quarantine_payload_checksum_impl(const std::string &payload);
bool serialize_quarantine_entries_impl(
    const std::vector<quarantine_record> &entries, std::string *payload);
bool append_quarantine_entry_impl(std::vector<quarantine_record> *entries,
                                  const std::string &artifact_name,
                                  const std::string &reason,
                                  uint64_t generation,
                                  const std::string &payload,
                                  std::string *identity);
bool update_quarantine_entry_state_impl(std::vector<quarantine_record> *entries,
                                        const std::string &identity,
                                        quarantine_state state);
void record_artifact_persist_event_impl(const char *backend_name,
                                        const char *artifact_name,
                                        size_t row_count, size_t payload_bytes,
                                        uint64_t serialize_ms, uint64_t save_ms,
                                        bool ok);
bool use_file_truth_store_backend_impl();
bool vector_to_truth_payload_impl(const vector_index::vector_data &vector,
                                  uint32_t *dimension, std::string *payload);
bool truth_payload_to_vector_impl(uint32_t dimension,
                                  const std::string &payload,
                                  vector_index::vector_data *vector);
bool encode_truth_op_impl(vector_index_metadata_store::change_op op,
                          uint8_t *value);
bool decode_truth_op_impl(uint8_t value,
                          vector_index_metadata_store::change_op *op);
bool is_row_artifact_name_impl(const char *artifact_name);
bool is_valid_artifact_name_impl(const char *artifact_name);
innodb_vector_truth_store::committed_row make_debug_committed_row_impl(
    const std::string &payload);
innodb_vector_truth_store::change_log_row make_debug_change_log_row_impl(
    const std::string &payload);
innodb_vector_truth_store::prepared_change_row make_debug_prepared_row_impl(
    const std::string &payload);
bool extract_debug_payload_impl(
    const std::vector<innodb_vector_truth_store::committed_row> &rows,
    std::string *payload);
bool extract_debug_payload_impl(
    const std::vector<innodb_vector_truth_store::change_log_row> &rows,
    std::string *payload);
bool extract_debug_payload_impl(
    const std::vector<innodb_vector_truth_store::prepared_change_row> &rows,
    std::string *payload);
bool to_innodb_committed_rows_impl(
    const std::vector<vector_index_metadata_store::committed_row> &rows,
    std::vector<innodb_vector_truth_store::committed_row> *out);
bool from_innodb_committed_rows_impl(
    const std::vector<innodb_vector_truth_store::committed_row> &rows,
    std::vector<vector_index_metadata_store::committed_row> *out);
bool to_innodb_change_log_rows_impl(
    const std::vector<vector_index_metadata_store::change_log_row> &rows,
    std::vector<innodb_vector_truth_store::change_log_row> *out);
bool from_innodb_change_log_rows_impl(
    const std::vector<innodb_vector_truth_store::change_log_row> &rows,
    std::vector<vector_index_metadata_store::change_log_row> *out);
bool to_innodb_prepared_rows_impl(
    const std::vector<vector_index_metadata_store::prepared_change_row> &rows,
    std::vector<innodb_vector_truth_store::prepared_change_row> *out);
bool from_innodb_prepared_rows_impl(
    const std::vector<innodb_vector_truth_store::prepared_change_row> &rows,
    std::vector<vector_index_metadata_store::prepared_change_row> *out);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index_truth_store::detail

#endif  // SQL_VECTOR_INDEX_TRUTH_STORE_INTERNAL_INCLUDED
