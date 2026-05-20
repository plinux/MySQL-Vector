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

#ifndef SQL_VECTOR_DML_SYNC_INCLUDED
#define SQL_VECTOR_DML_SYNC_INCLUDED

#include <cstdint>
#include <string>
#include <vector>

struct TABLE;
class THD;
class Field;
using uchar = unsigned char;

namespace vector_dml_sync {

struct prepared_change {
  std::string index_name;
  uint64_t doc_id{0};
  bool erase{false};
  std::vector<float> vector;
};

using prepared_changes = std::vector<prepared_change>;

bool has_vector_columns(const TABLE *table);
bool supports_vector_doc_id(TABLE *table);
bool prepare_insert_row(TABLE *table, const uchar *record,
                        prepared_changes *changes);
bool prepare_delete_row(TABLE *table, const uchar *record,
                        prepared_changes *changes);
bool prepare_update_row(TABLE *table, const uchar *old_record,
                        const uchar *new_record, prepared_changes *changes);
bool prepare_insert_row_for_index(TABLE *table, const std::string &index_name,
                                  const std::string &column_name,
                                  const uchar *record,
                                  prepared_changes *changes);
bool stage_prepared_changes(THD *thd, const prepared_changes &changes);

bool stage_insert_row(THD *thd, TABLE *table, const uchar *record);
bool stage_delete_row(THD *thd, TABLE *table, const uchar *record);
bool stage_update_row(THD *thd, TABLE *table, const uchar *old_record,
                      const uchar *new_record);
/**
  Apply the current table record update and stage vector changes on success.

  This keeps non-vector UPDATE callers on the native handler path while sharing
  the vector-specific prepare -> handler update -> stage sequence.
*/
int update_row_and_stage_changes(THD *thd, TABLE *table);

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool get_doc_id_field_for_testing(TABLE *table, Field **pk_field);
bool get_doc_id_for_record_for_testing(Field *pk_field, const uchar *record,
                                 uint64_t *doc_id);
bool decode_vector_field_for_testing(Field *field, const uchar *record,
                                 std::vector<float> *vector);
bool collect_upsert_for_field_for_testing(TABLE *table, Field *field, uint64_t doc_id,
                                     const uchar *record,
                                     prepared_changes *changes);
bool collect_erase_for_field_for_testing(TABLE *table, Field *field, uint64_t doc_id,
                                    prepared_changes *changes);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_dml_sync

#endif  // SQL_VECTOR_DML_SYNC_INCLUDED
