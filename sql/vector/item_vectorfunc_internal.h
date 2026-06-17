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

#ifndef SQL_ITEM_VECTORFUNC_INTERNAL_INCLUDED
#define SQL_ITEM_VECTORFUNC_INTERNAL_INCLUDED

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sql/auth/auth_acls.h"
#include "sql/item.h"
#include "sql/vector/vector_index_registry.h"
#include "sql_string.h"

namespace vector_itemfunc_internal {

bool eval_vector_arg(Item *arg, String *buf, const String **value);
bool to_std_string(const String *value, std::string *out);
bool eval_uint_arg(Item *arg, ulonglong &value);
bool eval_uint32_arg(Item *arg, uint32_t min_value, uint32_t max_value,
                     uint32_t &value);
bool require_process_access(THD *thd);
bool check_vector_current_db_ddl_access(THD *thd, Access_bitmask privilege);
bool check_vector_table_ddl_access(
    THD *thd, const vector_index_registry::index_info &info,
    Access_bitmask privilege);
/** Return whether an index privilege is granted without raising ACL errors. */
bool has_vector_index_access(THD *thd,
                             const vector_index_registry::index_info &info,
                             Access_bitmask privilege);
bool check_vector_existing_index_ddl_access(
    THD *thd, const std::string &index_name, Access_bitmask privilege,
    const char *func_name, bool missing_index_uses_current_db);
bool check_vector_all_indexes_ddl_access(THD *thd, Access_bitmask privilege,
                                         const char *func_name);
bool maybe_binlog_vector_write_query(THD *thd);
bool decode_vector_arg(Item *arg, String *buf, std::vector<float> *out);
bool decode_txn_and_name(Item *txn_arg, Item *name_arg, String *name_buf,
                         uint64_t *txn_id, std::string *name);
std::string encode_hex_bytes(const std::string &input);
bool decode_hex_bytes(const String *input, std::string &output);

}  // namespace vector_itemfunc_internal

#endif  // SQL_ITEM_VECTORFUNC_INTERNAL_INCLUDED
