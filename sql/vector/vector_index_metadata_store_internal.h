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

#ifndef SQL_VECTOR_INDEX_METADATA_STORE_INTERNAL_INCLUDED
#define SQL_VECTOR_INDEX_METADATA_STORE_INTERNAL_INCLUDED

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "sql/vector/vector_index_metadata_store.h"

namespace vector_index_metadata_store::detail {

extern const char *const kCommittedHeaderV1;
extern const char *const kPreparedHeaderV1;
extern const char *const kManifestHeaderV1;
extern const char *const kChangeLogHeaderV1;
extern const char *const kSegmentTaskHeaderV1;

bool split_tab_fields(const std::string &line, std::vector<std::string> *fields);
std::string encode_hex(const std::string &input);
bool decode_hex(const std::string &encoded, std::string *decoded);
bool parse_uint64(const std::string &text, uint64_t *value);
std::string metadata_path();
std::string committed_path();
std::string manifest_path();
std::string prepared_path();
std::string change_log_path();
std::string segment_task_path();
bool ensure_parent_directory(const std::string &path);
bool open_read_primary(const std::string &path, std::ifstream *file);
bool remove_if_exists(const std::string &path);
bool quarantine_file_if_exists(const std::string &path);
bool raw_path_for_artifact(const std::string &artifact_name, std::string *path);
const char *change_op_to_string(change_op op);
bool parse_change_op(const std::string &text, change_op *op);

bool deserialize_metadata_rows_impl(const std::string &payload,
                                    std::vector<metadata_row> *rows);
bool serialize_metadata_rows_impl(const std::vector<metadata_row> &rows,
                                  std::string *payload);
bool deserialize_committed_rows_impl(const std::string &payload,
                                     std::vector<committed_row> *rows);
bool serialize_committed_rows_impl(const std::vector<committed_row> &rows,
                                   std::string *payload);
bool deserialize_manifest_row_impl(const std::string &payload, manifest_row *row);
bool serialize_manifest_row_impl(const manifest_row &row, std::string *payload);
bool deserialize_change_log_rows_impl(const std::string &payload,
                                      std::vector<change_log_row> *rows);
bool serialize_change_log_rows_impl(const std::vector<change_log_row> &rows,
                                    std::string *payload);
bool deserialize_prepared_rows_impl(const std::string &payload,
                                    std::vector<prepared_change_row> *rows);
bool serialize_prepared_rows_impl(const std::vector<prepared_change_row> &rows,
                                  std::string *payload);
bool deserialize_segment_task_rows_impl(
    const std::string &payload, std::vector<segment_task_row> *rows);
bool serialize_segment_task_rows_impl(
    const std::vector<segment_task_row> &rows, std::string *payload);

}  // namespace vector_index_metadata_store::detail

#endif  // SQL_VECTOR_INDEX_METADATA_STORE_INTERNAL_INCLUDED
