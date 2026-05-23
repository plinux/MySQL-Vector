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

#include "my_byteorder.h"
#include "my_dbug.h"
#include "sql/auth/auth_acls.h"
#include "sql/binlog.h"
#include "sql/item.h"
#include "sql/sql_class.h"
#include "sql/vector/vector_utils.h"
#include "sql_string.h"

namespace {

[[maybe_unused]] bool eval_vector_arg(Item *arg, String *buf,
                                      const String **value) {
  *value = arg->val_str(buf);
  if (*value == nullptr || arg->null_value) return false;
  return true;
}

[[maybe_unused]] bool to_std_string(const String *value, std::string *out) {
  out->assign(value->ptr(), value->length());
  return true;
}

[[maybe_unused]] bool require_process_access(THD *thd) {
  return thd != nullptr && thd->security_context()->check_access(PROCESS_ACL);
}

[[maybe_unused]] bool maybe_binlog_vector_write_query(THD *thd) {
  DBUG_EXECUTE_IF("vector_item_fail_binlog_write", return false;);
  if (thd == nullptr) return false;
  if (!mysql_bin_log.is_open()) return true;
  if ((thd->variables.option_bits & OPTION_BIN_LOG) == 0) return true;
  if (thd->slave_thread || thd->in_sub_stmt) return true;
  return !mysql_bin_log.write_stmt_directly(thd, thd->query().str,
                                            thd->query().length,
                                            SQLCOM_SELECT);
}

[[maybe_unused]] bool decode_vector_arg(Item *arg, String *buf,
                                        std::vector<float> *out) {
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

[[maybe_unused]] bool decode_txn_and_name(Item *txn_arg, Item *name_arg,
                                          String *name_buf, uint64_t *txn_id,
                                          std::string *name) {
  const longlong txn_id_ll = txn_arg->val_int();
  if (txn_arg->null_value || txn_id_ll < 0) return false;

  const String *name_value = name_arg->val_str(name_buf);
  if (name_value == nullptr || name_arg->null_value || name_value->length() == 0) {
    return false;
  }

  *txn_id = static_cast<uint64_t>(txn_id_ll);
  to_std_string(name_value, name);
  return true;
}

[[maybe_unused]] char hex_digit(unsigned value) {
  return static_cast<char>((value < 10U) ? ('0' + value) : ('a' + (value - 10U)));
}

[[maybe_unused]] std::string encode_hex_bytes(const std::string &input) {
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

[[maybe_unused]] bool decode_hex_bytes(const String *input,
                                       std::string *output) {
  if (input == nullptr || output == nullptr) return false;
  if ((input->length() % 2) != 0) return false;
  output->clear();
  output->reserve(input->length() / 2);
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
    output->push_back(static_cast<char>((high << 4) | low));
  }
  return true;
}

}  // namespace

#endif  // SQL_ITEM_VECTORFUNC_INTERNAL_INCLUDED
