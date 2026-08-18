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

#include "sql/vector/vector_dml_sync.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "my_byteorder.h"
#include "mysql_com.h"
#include "mysqld_error.h"
#include "sql/field.h"
#include "sql/key.h"
#include "sql/sql_class.h"
#include "sql/table.h"
#include "sql/vector/vector_index_identity.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_trx_participant.h"
#include "sql/vector/vector_utils.h"

namespace {

bool get_doc_id_field(TABLE *table, Field **pk_field) {
  if (pk_field == nullptr || table == nullptr) return false;
  if (table->s == nullptr || table->s->primary_key == MAX_KEY) return false;

  const KEY &primary_key = table->key_info[table->s->primary_key];
  if (primary_key.user_defined_key_parts != 1) return false;

  Field *field = primary_key.key_part[0].field;
  if (field == nullptr || field->is_nullable()) return false;
  if (field->result_type() != INT_RESULT) return false;
  *pk_field = field;
  return true;
}

bool get_doc_id_for_record(Field *pk_field, const uchar *record, uint64_t *doc_id) {
  if (pk_field == nullptr || record == nullptr || doc_id == nullptr) return false;
  if (pk_field->is_null_in_record(record)) return false;
  if (pk_field->table == nullptr || pk_field->table->record[0] == nullptr)
    return false;

  my_bitmap_map *old_read_map = nullptr;
  if (pk_field->table != nullptr && pk_field->table->read_set != nullptr) {
    old_read_map = tmp_use_all_columns(pk_field->table, pk_field->table->read_set);
  }
  const ptrdiff_t row_offset = record - pk_field->table->record[0];
  const longlong value = pk_field->val_int_offset(row_offset);
  if (pk_field->table != nullptr && pk_field->table->read_set != nullptr &&
      old_read_map != nullptr) {
    tmp_restore_column_map(pk_field->table->read_set, old_read_map);
  }
  if (pk_field->is_unsigned()) {
    *doc_id = static_cast<uint64_t>(static_cast<ulonglong>(value));
    return true;
  }
  if (value < 0) return false;
  *doc_id = static_cast<uint64_t>(value);
  return true;
}

bool decode_vector_field(Field *field, const uchar *record,
                         std::vector<float> *vector) {
  if (field == nullptr || record == nullptr || vector == nullptr) return false;
  vector->clear();
  if (field->is_null_in_record(record)) return false;

  if (field->type() != MYSQL_TYPE_BLOB || field->table == nullptr ||
      field->table->record[0] == nullptr) {
    return false;
  }

  auto *blob_field = static_cast<Field_blob *>(field);
  const ptrdiff_t row_offset = record - field->table->record[0];
  const uint32 payload_len = blob_field->get_length(row_offset);
  if (payload_len == 0) return false;

  const uchar *payload = blob_field->get_blob_data(row_offset);
  if (payload == nullptr) return false;

  String binary(reinterpret_cast<const char *>(payload), payload_len,
                &my_charset_bin);
  size_t dim = 0;
  if (!vector_utils::parse_binary_vector(&binary, &dim) || dim == 0)
    return false;

  vector->resize(dim);
  for (size_t i = 0; i < dim; ++i) {
    (*vector)[i] = float4get(payload + (i * vector_utils::kVectorElemSize));
  }
  return true;
}

std::string make_index_name(TABLE *table, Field *field) {
  return vector_index_identity::make_index_name(
      table->s->db.str, table->s->table_name.str, field->field_name);
}

bool collect_upsert_for_field(TABLE *table, Field *field, uint64_t doc_id,
                              const uchar *record,
                              vector_dml_sync::prepared_changes *changes) {
  std::string index_name = make_index_name(table, field);
  vector_index_registry::index_info info;
  if (!vector_index_registry::get_index_info(index_name, &info)) return false;

  std::vector<float> vector;
  if (!decode_vector_field(field, record, &vector)) {
    my_error(ER_INTERNAL_ERROR, MYF(0), "Failed to decode vector payload");
    return true;
  }

  if (changes == nullptr) return true;
  vector_dml_sync::prepared_change change;
  change.index_name = std::move(index_name);
  change.doc_id = doc_id;
  change.erase = false;
  change.vector = std::move(vector);
  changes->push_back(std::move(change));
  return false;
}

bool collect_erase_for_field(TABLE *table, Field *field, uint64_t doc_id,
                             vector_dml_sync::prepared_changes *changes) {
  std::string index_name = make_index_name(table, field);
  vector_index_registry::index_info info;
  if (!vector_index_registry::get_index_info(index_name, &info)) return false;

  if (changes == nullptr) return true;
  vector_dml_sync::prepared_change change;
  change.index_name = std::move(index_name);
  change.doc_id = doc_id;
  change.erase = true;
  changes->push_back(std::move(change));
  return false;
}

bool for_each_vector_field(TABLE *table, bool (*callback)(Field *, void *),
                           void *arg) {
  for (Field **field = table->field; *field != nullptr; ++field) {
    if (!(*field)->is_flag_set(FIELD_IS_VECTOR)) continue;
    if (callback(*field, arg)) return true;
  }
  return false;
}

}  // namespace

namespace vector_dml_sync {

bool has_vector_columns(const TABLE *table) {
  if (table == nullptr) return false;
  for (Field *const *field = table->field; *field != nullptr; ++field) {
    if ((*field)->is_flag_set(FIELD_IS_VECTOR)) return true;
  }
  return false;
}

bool supports_vector_doc_id(TABLE *table) {
  Field *pk_field = nullptr;
  return get_doc_id_field(table, &pk_field);
}

bool prepare_insert_row(TABLE *table, const uchar *record,
                      prepared_changes *changes) {
  if (table == nullptr || record == nullptr || changes == nullptr) return true;
  changes->clear();
  if (!has_vector_columns(table)) return false;

  Field *pk_field = nullptr;
  if (!get_doc_id_field(table, &pk_field)) return false;

  uint64_t doc_id = 0;
  if (!get_doc_id_for_record(pk_field, record, &doc_id)) return false;

  struct callback_arg {
    TABLE *table;
    uint64_t doc_id;
    const uchar *record;
    prepared_changes *changes;
  } arg{table, doc_id, record, changes};

  auto callback = [](Field *field, void *raw_arg) -> bool {
    callback_arg *cb = static_cast<callback_arg *>(raw_arg);
    if (field->is_null_in_record(cb->record)) {
      return collect_erase_for_field(cb->table, field, cb->doc_id, cb->changes);
    }
    return collect_upsert_for_field(cb->table, field, cb->doc_id, cb->record,
                                    cb->changes);
  };

  return for_each_vector_field(table, callback, &arg);
}

bool prepare_insert_row_for_index(TABLE *table, const std::string &index_name,
                                const std::string &column_name,
                                const uchar *record, prepared_changes *changes) {
  if (table == nullptr || record == nullptr || changes == nullptr) return true;
  changes->clear();

  Field *pk_field = nullptr;
  if (!get_doc_id_field(table, &pk_field)) return false;

  uint64_t doc_id = 0;
  if (!get_doc_id_for_record(pk_field, record, &doc_id)) return false;

  Field *target_field = nullptr;
  for (Field **field = table->field; *field != nullptr; ++field) {
    if (!(*field)->is_flag_set(FIELD_IS_VECTOR)) continue;
    if (column_name == (*field)->field_name) {
      target_field = *field;
      break;
    }
  }
  if (target_field == nullptr) return false;

  if (target_field->is_null_in_record(record)) return false;

  std::vector<float> vector;
  if (!decode_vector_field(target_field, record, &vector)) {
    my_error(ER_INTERNAL_ERROR, MYF(0), "Failed to decode vector payload");
    return true;
  }

  prepared_change change;
  change.index_name = index_name;
  change.doc_id = doc_id;
  change.erase = false;
  change.vector = std::move(vector);
  changes->push_back(std::move(change));
  return false;
}

bool prepare_delete_row(TABLE *table, const uchar *record,
                      prepared_changes *changes) {
  if (table == nullptr || record == nullptr || changes == nullptr) return true;
  changes->clear();
  if (!has_vector_columns(table)) return false;

  Field *pk_field = nullptr;
  if (!get_doc_id_field(table, &pk_field)) return false;

  uint64_t doc_id = 0;
  if (!get_doc_id_for_record(pk_field, record, &doc_id)) return false;

  struct callback_arg {
    TABLE *table;
    uint64_t doc_id;
    prepared_changes *changes;
  } arg{table, doc_id, changes};

  auto callback = [](Field *field, void *raw_arg) -> bool {
    callback_arg *cb = static_cast<callback_arg *>(raw_arg);
    return collect_erase_for_field(cb->table, field, cb->doc_id, cb->changes);
  };

  return for_each_vector_field(table, callback, &arg);
}

bool prepare_update_row(TABLE *table, const uchar *old_record,
                      const uchar *new_record, prepared_changes *changes) {
  if (table == nullptr || old_record == nullptr || new_record == nullptr ||
      changes == nullptr) {
    return true;
  }
  changes->clear();
  if (!has_vector_columns(table)) return false;

  Field *pk_field = nullptr;
  if (!get_doc_id_field(table, &pk_field)) return false;

  uint64_t old_doc_id = 0;
  uint64_t new_doc_id = 0;
  if (!get_doc_id_for_record(pk_field, old_record, &old_doc_id) ||
      !get_doc_id_for_record(pk_field, new_record, &new_doc_id)) {
    return false;
  }

  struct callback_arg {
    TABLE *table;
    uint64_t old_doc_id;
    uint64_t new_doc_id;
    const uchar *old_record;
    const uchar *new_record;
    prepared_changes *changes;
  } arg{table, old_doc_id, new_doc_id, old_record, new_record, changes};

  auto callback = [](Field *field, void *raw_arg) -> bool {
    callback_arg *cb = static_cast<callback_arg *>(raw_arg);
    if (cb->old_doc_id != cb->new_doc_id) {
      if (collect_erase_for_field(cb->table, field, cb->old_doc_id,
                                  cb->changes)) {
        return true;
      }
    }

    if (field->is_null_in_record(cb->new_record)) {
      return collect_erase_for_field(cb->table, field, cb->new_doc_id,
                                     cb->changes);
    }
    return collect_upsert_for_field(cb->table, field, cb->new_doc_id,
                                    cb->new_record, cb->changes);
  };

  return for_each_vector_field(table, callback, &arg);
}

bool stage_prepared_changes(THD *thd, const prepared_changes &changes) {
  if (thd == nullptr) return true;
  if (changes.empty()) return false;

  if (!vector_trx_participant::register_participant(thd)) {
    my_error(ER_INTERNAL_ERROR, MYF(0),
             "Failed to register vector transaction participant");
    return true;
  }

  const uint64_t statement_id = static_cast<uint64_t>(thd->query_id);
  std::vector<vector_index::index_service::pending_change_snapshot>
      staged_changes;
  staged_changes.reserve(changes.size());
  for (const prepared_change &change : changes) {
    staged_changes.push_back({change.index_name, change.erase, change.doc_id,
                              change.vector});
  }
  if (!vector_index_registry::stage_changes_for_thd_txn(
          thd, statement_id, staged_changes)) {
    my_error(ER_INTERNAL_ERROR, MYF(0),
             "Failed to stage vector truth changes");
    return true;
  }
  return false;
}

bool stage_insert_row(THD *thd, TABLE *table, const uchar *record) {
  prepared_changes changes;
  if (prepare_insert_row(table, record, &changes)) return true;
  return stage_prepared_changes(thd, changes);
}

bool stage_delete_row(THD *thd, TABLE *table, const uchar *record) {
  prepared_changes changes;
  if (prepare_delete_row(table, record, &changes)) return true;
  return stage_prepared_changes(thd, changes);
}

bool stage_update_row(THD *thd, TABLE *table, const uchar *old_record,
                      const uchar *new_record) {
  prepared_changes changes;
  if (prepare_update_row(table, old_record, new_record, &changes)) return true;
  return stage_prepared_changes(thd, changes);
}

int update_row_and_stage_changes(THD *thd, TABLE *table) {
  prepared_changes changes;
  if (prepare_update_row(table, table->record[1], table->record[0],
                         &changes)) {
    return 1;
  }

  const int error = table->file->ha_update_row(table->record[1],
                                               table->record[0]);
  if (error == 0 && stage_prepared_changes(thd, changes)) return 1;
  return error;
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool get_doc_id_field_for_testing(TABLE *table, Field **pk_field) {
  return get_doc_id_field(table, pk_field);
}

bool get_doc_id_for_record_for_testing(Field *pk_field, const uchar *record,
                                 uint64_t *doc_id) {
  return get_doc_id_for_record(pk_field, record, doc_id);
}

bool decode_vector_field_for_testing(Field *field, const uchar *record,
                                 std::vector<float> *vector) {
  return decode_vector_field(field, record, vector);
}

bool collect_upsert_for_field_for_testing(TABLE *table, Field *field, uint64_t doc_id,
                                     const uchar *record,
                                     prepared_changes *changes) {
  return collect_upsert_for_field(table, field, doc_id, record, changes);
}

bool collect_erase_for_field_for_testing(TABLE *table, Field *field, uint64_t doc_id,
                                    prepared_changes *changes) {
  return collect_erase_for_field(table, field, doc_id, changes);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_dml_sync
