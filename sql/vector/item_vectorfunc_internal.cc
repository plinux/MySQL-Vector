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

#include "sql/vector/item_vectorfunc_internal.h"

#include <cstddef>

#include "my_byteorder.h"
#include "my_dbug.h"
#include "mysqld_error.h"
#include "sql/auth/auth_common.h"
#include "sql/binlog.h"
#include "sql/sql_class.h"
#include "sql/table.h"
#include "sql/vector/vector_utils.h"

namespace vector_itemfunc_internal {

bool eval_vector_arg(Item *arg, String *buf, const String **value) {
  *value = arg->val_str(buf);
  if (*value == nullptr || arg->null_value) return false;
  return true;
}

bool to_std_string(const String *value, std::string *out) {
  out->assign(value->ptr(), value->length());
  return true;
}

bool eval_uint_arg(Item *arg, ulonglong &value) {
  const longlong signed_value = arg->val_int();
  if (arg->null_value || (!arg->unsigned_flag && signed_value < 0)) {
    return false;
  }

  value = arg->val_uint();
  return true;
}

bool eval_uint32_arg(Item *arg, uint32_t min_value, uint32_t max_value,
                     uint32_t &value) {
  ulonglong parsed_value = 0;
  if (!eval_uint_arg(arg, parsed_value) || parsed_value < min_value ||
      parsed_value > max_value) {
    return false;
  }

  value = static_cast<uint32_t>(parsed_value);
  return true;
}

bool require_process_access(THD *thd) {
  return thd != nullptr && thd->security_context()->check_access(PROCESS_ACL);
}

bool check_vector_current_db_ddl_access(THD *thd, Access_bitmask privilege) {
  return thd == nullptr ||
         check_access(thd, privilege, nullptr, nullptr, nullptr, false, false);
}

static bool check_vector_table_ddl_access_impl(
    THD *thd, const vector_index_registry::index_info &info,
    Access_bitmask privilege, bool no_errors) {
  if (thd == nullptr) return true;
  if (info.schema_name.empty() || info.table_name.empty()) {
    Access_bitmask effective_access = 0;
    return check_access(thd, privilege, nullptr, &effective_access, nullptr,
                        false, no_errors);
  }

  Table_ref table_ref(info.schema_name.c_str(), info.schema_name.length(),
                      info.table_name.c_str(), info.table_name.length(),
                      info.table_name.c_str(), TL_IGNORE);
  if (check_access(thd, privilege, table_ref.db, &table_ref.grant.privilege,
                   &table_ref.grant.m_internal, false, no_errors)) {
    return true;
  }
  return check_grant(thd, privilege, &table_ref, false, 1, no_errors);
}

bool check_vector_table_ddl_access(
    THD *thd, const vector_index_registry::index_info &info,
    Access_bitmask privilege) {
  return check_vector_table_ddl_access_impl(thd, info, privilege, false);
}

bool has_vector_index_access(THD *thd,
                             const vector_index_registry::index_info &info,
                             Access_bitmask privilege) {
  return !check_vector_table_ddl_access_impl(thd, info, privilege, true);
}

bool check_vector_existing_index_ddl_access(
    THD *thd, const std::string &index_name, Access_bitmask privilege,
    const char *func_name, bool missing_index_uses_current_db) {
  vector_index_registry::index_info info;
  if (!vector_index_registry::get_index_info(index_name, &info)) {
    if (missing_index_uses_current_db) {
      return check_vector_current_db_ddl_access(thd, privilege);
    }
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name);
    return true;
  }
  return check_vector_table_ddl_access(thd, info, privilege);
}

bool check_vector_all_indexes_ddl_access(THD *thd, Access_bitmask privilege,
                                         const char *func_name) {
  std::vector<std::string> index_names;
  if (!vector_index_registry::list_indexes(&index_names)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name);
    return true;
  }
  for (const std::string &index_name : index_names) {
    if (check_vector_existing_index_ddl_access(
            thd, index_name, privilege, func_name, false)) {
      return true;
    }
  }
  return false;
}

bool maybe_binlog_vector_write_query(THD *thd) {
  DBUG_EXECUTE_IF("vector_item_fail_binlog_write", return false;);
  if (thd == nullptr) return false;
  if (!mysql_bin_log.is_open()) return true;
  if ((thd->variables.option_bits & OPTION_BIN_LOG) == 0) return true;
  if (thd->slave_thread || thd->in_sub_stmt) return true;
  return !mysql_bin_log.write_stmt_directly(thd, thd->query().str,
                                            thd->query().length,
                                            SQLCOM_SELECT);
}

bool decode_vector_arg(Item *arg, String *buf, std::vector<float> *out) {
  const String *value = nullptr;
  if (!eval_vector_arg(arg, buf, &value)) return false;

  size_t dim = 0;
  if (!vector_utils::parse_binary_vector(value, &dim)) {
    std::vector<float> parsed;
    if (!vector_utils::parse_text_vector(value, &parsed)) return false;
    out->assign(parsed.begin(), parsed.end());
    return true;
  }

  out->clear();
  out->reserve(dim);
  const uchar *ptr = reinterpret_cast<const uchar *>(value->ptr());
  for (size_t i = 0; i < dim; ++i) {
    out->push_back(float4get(ptr + i * vector_utils::kVectorElemSize));
  }
  return true;
}

bool decode_txn_and_name(Item *txn_arg, Item *name_arg, String *name_buf,
                         uint64_t *txn_id, std::string *name) {
  ulonglong parsed_txn_id = 0;
  if (!eval_uint_arg(txn_arg, parsed_txn_id)) return false;

  const String *name_value = name_arg->val_str(name_buf);
  if (name_value == nullptr || name_arg->null_value ||
      name_value->length() == 0) {
    return false;
  }

  *txn_id = static_cast<uint64_t>(parsed_txn_id);
  to_std_string(name_value, name);
  return true;
}

namespace {

char hex_digit(unsigned value) {
  return static_cast<char>((value < 10U) ? ('0' + value)
                                         : ('a' + (value - 10U)));
}

}  // namespace

std::string encode_hex_bytes(const std::string &input) {
  std::string out;
  out.resize(input.size() * 2);
  for (size_t i = 0; i < input.size(); ++i) {
    const unsigned value =
        static_cast<unsigned>(static_cast<unsigned char>(input[i]));
    out[i * 2] = hex_digit((value >> 4) & 0x0FU);
    out[i * 2 + 1] = hex_digit(value & 0x0FU);
  }
  return out;
}

bool decode_hex_bytes(const String *input, std::string &output) {
  if (input == nullptr) return false;
  if ((input->length() % 2) != 0) return false;
  output.clear();
  output.reserve(input->length() / 2);
  const char *ptr = input->ptr();
  for (size_t i = 0; i < input->length(); i += 2) {
    auto decode_nibble = [](char ch, unsigned *value) {
      if (ch >= '0' && ch <= '9') {
        *value = static_cast<unsigned>(ch - '0');
        return true;
      }
      if (ch >= 'a' && ch <= 'f') {
        *value = static_cast<unsigned>(10 + ch - 'a');
        return true;
      }
      if (ch >= 'A' && ch <= 'F') {
        *value = static_cast<unsigned>(10 + ch - 'A');
        return true;
      }
      return false;
    };
    unsigned high = 0;
    unsigned low = 0;
    if (!decode_nibble(ptr[i], &high) || !decode_nibble(ptr[i + 1], &low)) {
      return false;
    }
    output.push_back(static_cast<char>((high << 4) | low));
  }
  return true;
}

}  // namespace vector_itemfunc_internal
