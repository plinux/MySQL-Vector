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

#include "mysqld_error.h"
#include "sql/vector/item_vectorfunc_internal.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql_string.h"

bool Item_func_vec_index_txn_begin::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 0)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_txn_begin::val_int() {
  assert(fixed && arg_count == 0);
  null_value = false;
  return static_cast<longlong>(vector_index_registry::begin_txn());
}

bool Item_func_vec_index_txn_pending::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_txn_pending::val_int() {
  assert(fixed && arg_count == 1);
  null_value = true;

  const longlong txn_id_ll = args[0]->val_int();
  if (args[0]->null_value || txn_id_ll < 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  null_value = false;
  return static_cast<longlong>(
      vector_index_registry::pending_txn_changes(
          static_cast<uint64_t>(txn_id_ll)));
}

bool Item_func_vec_index_stage_upsert::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 4)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_stage_upsert::val_int() {
  assert(fixed && arg_count == 4);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  const longlong txn_id_ll = args[1]->val_int();
  const longlong doc_id_ll = args[2]->val_int();
  if (name == nullptr || args[0]->null_value || args[1]->null_value ||
      args[2]->null_value || txn_id_ll < 0 || doc_id_ll < 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  vector_index::vector_data vector;
  String vector_buf;
  if (!decode_vector_arg(args[3], &vector_buf, &vector)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (!vector_index_registry::stage_upsert(static_cast<uint64_t>(txn_id_ll),
                                           index_name,
                                           static_cast<uint64_t>(doc_id_ll),
                                           vector)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  null_value = false;
  return 1;
}

bool Item_func_vec_index_stage_erase::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 3)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_stage_erase::val_int() {
  assert(fixed && arg_count == 3);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  const longlong txn_id_ll = args[1]->val_int();
  const longlong doc_id_ll = args[2]->val_int();
  if (name == nullptr || args[0]->null_value || args[1]->null_value ||
      args[2]->null_value || txn_id_ll < 0 || doc_id_ll < 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (!vector_index_registry::stage_erase(static_cast<uint64_t>(txn_id_ll),
                                          index_name,
                                          static_cast<uint64_t>(doc_id_ll))) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  null_value = false;
  return 1;
}

bool Item_func_vec_index_txn_commit::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_txn_commit::val_int() {
  assert(fixed && arg_count == 1);
  null_value = true;

  const longlong txn_id_ll = args[0]->val_int();
  if (args[0]->null_value || txn_id_ll < 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (!vector_index_registry::commit_txn(static_cast<uint64_t>(txn_id_ll))) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  null_value = false;
  return 1;
}

bool Item_func_vec_index_txn_rollback::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_txn_rollback::val_int() {
  assert(fixed && arg_count == 1);
  null_value = true;

  const longlong txn_id_ll = args[0]->val_int();
  if (args[0]->null_value || txn_id_ll < 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (!vector_index_registry::rollback_txn(static_cast<uint64_t>(txn_id_ll))) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  null_value = false;
  return 1;
}

bool Item_func_vec_index_txn_savepoint::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_txn_savepoint::val_int() {
  assert(fixed && arg_count == 2);
  null_value = true;

  String name_buf;
  uint64_t txn_id = 0;
  std::string savepoint_name;
  if (!decode_txn_and_name(args[0], args[1], &name_buf, &txn_id,
                           &savepoint_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (!vector_index_registry::savepoint_txn(txn_id, savepoint_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  null_value = false;
  return 1;
}

bool Item_func_vec_index_txn_rollback_to::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_txn_rollback_to::val_int() {
  assert(fixed && arg_count == 2);
  null_value = true;

  String name_buf;
  uint64_t txn_id = 0;
  std::string savepoint_name;
  if (!decode_txn_and_name(args[0], args[1], &name_buf, &txn_id,
                           &savepoint_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (!vector_index_registry::rollback_to_savepoint_txn(txn_id, savepoint_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  null_value = false;
  return 1;
}

bool Item_func_vec_index_txn_release_savepoint::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_txn_release_savepoint::val_int() {
  assert(fixed && arg_count == 2);
  null_value = true;

  String name_buf;
  uint64_t txn_id = 0;
  std::string savepoint_name;
  if (!decode_txn_and_name(args[0], args[1], &name_buf, &txn_id,
                           &savepoint_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (!vector_index_registry::release_savepoint_txn(txn_id, savepoint_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  null_value = false;
  return 1;
}

bool Item_func_vec_debug_truth_store_get_hex::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  const uint32 max_chars =
      static_cast<uint32>(MAX_BLOB_WIDTH / default_charset()->mbmaxlen);
  set_data_type_string(max_chars, default_charset());
  set_nullable(true);
  return false;
}

String *Item_func_vec_debug_truth_store_get_hex::val_str(
    String *str [[maybe_unused]]) {
  assert(fixed && arg_count == 1);
  null_value = true;

  if (!require_process_access(current_thd)) {
    my_error(ER_SPECIFIC_ACCESS_DENIED_ERROR, MYF(0), "PROCESS");
    return error_str();
  }

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value || name->length() == 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  std::string artifact_name;
  to_std_string(name, &artifact_name);
  std::string payload;
  if (!vector_index_truth_store::debug_get_artifact_payload(artifact_name, &payload)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  const std::string hex_payload = encode_hex_bytes(payload);
  m_value.length(0);
  m_value.set_charset(default_charset());
  if (!hex_payload.empty() &&
      m_value.append(hex_payload.c_str(), hex_payload.length())) {
    return error_str();
  }

  null_value = false;
  return &m_value;
}

bool Item_func_vec_debug_truth_store_set_hex::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_debug_truth_store_set_hex::val_int() {
  assert(fixed && arg_count == 2);
  null_value = true;

  if (!require_process_access(current_thd)) {
    my_error(ER_SPECIFIC_ACCESS_DENIED_ERROR, MYF(0), "PROCESS");
    return error_int();
  }

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value || name->length() == 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string artifact_name;
  to_std_string(name, &artifact_name);

  bool ok = false;
  if (args[1]->null_value) {
    ok = vector_index_truth_store::debug_delete_artifact(artifact_name);
  } else {
    String payload_buf;
    const String *payload_hex = args[1]->val_str(&payload_buf);
    if (payload_hex == nullptr) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
    std::string payload;
    if (!decode_hex_bytes(payload_hex, &payload)) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
    ok = vector_index_truth_store::debug_set_artifact_payload(artifact_name, payload);
  }

  if (!ok) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  null_value = false;
  return 1;
}
