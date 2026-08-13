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
#include "sql/sql_class.h"
#include "sql/vector/item_vectorfunc_internal.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql_string.h"

using namespace vector_itemfunc_internal;

bool Item_func_vec_index_txn_begin::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 0)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_txn_begin::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 0);
  null_value = false;
  const uint64_t txn_id = vector_index_registry::begin_txn(current_thd);
  if (txn_id == 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  return static_cast<longlong>(txn_id);
}

bool Item_func_vec_index_txn_pending::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_txn_pending::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  ulonglong txn_id = 0;
  if (!eval_uint_arg(args[0], txn_id)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  size_t pending_count = 0;
  if (!vector_index_registry::pending_txn_changes(
          current_thd, static_cast<uint64_t>(txn_id), &pending_count)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  null_value = false;
  return static_cast<longlong>(pending_count);
}

bool Item_func_vec_index_stage_upsert::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 4)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_stage_upsert::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 4);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  ulonglong txn_id = 0;
  ulonglong doc_id = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint_arg(args[1], txn_id) ||
      !eval_uint_arg(args[2], doc_id)) {
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
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::stage_upsert(
          current_thd, static_cast<uint64_t>(txn_id), index_name,
          static_cast<uint64_t>(doc_id), vector)) {
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
  assert_fixed_arg_count(fixed, arg_count, 3);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  ulonglong txn_id = 0;
  ulonglong doc_id = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint_arg(args[1], txn_id) ||
      !eval_uint_arg(args[2], doc_id)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::stage_erase(
          current_thd, static_cast<uint64_t>(txn_id), index_name,
          static_cast<uint64_t>(doc_id))) {
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
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  ulonglong txn_id = 0;
  if (!eval_uint_arg(args[0], txn_id)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (!vector_index_registry::commit_txn(current_thd,
                                         static_cast<uint64_t>(txn_id))) {
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
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  ulonglong txn_id = 0;
  if (!eval_uint_arg(args[0], txn_id)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (!vector_index_registry::rollback_txn(
          current_thd, static_cast<uint64_t>(txn_id))) {
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
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  uint64_t txn_id = 0;
  std::string savepoint_name;
  if (!decode_txn_and_name(args[0], args[1], &name_buf, &txn_id,
                           &savepoint_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (!vector_index_registry::savepoint_txn(current_thd, txn_id,
                                             savepoint_name)) {
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
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  uint64_t txn_id = 0;
  std::string savepoint_name;
  if (!decode_txn_and_name(args[0], args[1], &name_buf, &txn_id,
                           &savepoint_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (!vector_index_registry::rollback_to_savepoint_txn(
          current_thd, txn_id, savepoint_name)) {
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
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  uint64_t txn_id = 0;
  std::string savepoint_name;
  if (!decode_txn_and_name(args[0], args[1], &name_buf, &txn_id,
                           &savepoint_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (!vector_index_registry::release_savepoint_txn(
          current_thd, txn_id, savepoint_name)) {
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
  assert_fixed_arg_count(fixed, arg_count, 1);
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
  assert_fixed_arg_count(fixed, arg_count, 2);
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

  String payload_buf;
  const String *payload_hex = args[1]->val_str(&payload_buf);
  bool ok = false;
  if (args[1]->null_value) {
    ok = vector_index_truth_store::debug_delete_artifact(artifact_name);
  } else {
    if (payload_hex == nullptr) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
    std::string payload;
    if (!decode_hex_bytes(payload_hex, payload)) {
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
