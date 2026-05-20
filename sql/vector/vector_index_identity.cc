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

#include "sql/vector/vector_index_identity.h"

#include <cstddef>

namespace vector_index_identity {

std::string make_index_name(const std::string &schema_name,
                            const std::string &table_name,
                            const std::string &column_name) {
  return make_table_prefix(schema_name, table_name) + column_name;
}

std::string make_schema_prefix(const std::string &schema_name) {
  return schema_name + ".";
}

std::string make_table_prefix(const std::string &schema_name,
                              const std::string &table_name) {
  return make_schema_prefix(schema_name) + table_name + ".";
}

bool parse_index_name(const std::string &index_name,
                      mapped_index_identity *identity) {
  const size_t first_dot = index_name.find('.');
  if (first_dot == std::string::npos || first_dot == 0) return false;

  const size_t second_dot = index_name.find('.', first_dot + 1);
  if (second_dot == std::string::npos || second_dot == first_dot + 1 ||
      second_dot == index_name.size() - 1) {
    return false;
  }

  if (index_name.find('.', second_dot + 1) != std::string::npos) {
    return false;
  }

  if (identity != nullptr) {
    identity->schema_name = index_name.substr(0, first_dot);
    identity->table_name =
        index_name.substr(first_dot + 1, second_dot - first_dot - 1);
    identity->column_name = index_name.substr(second_dot + 1);
  }
  return true;
}

bool matches_schema(const std::string &index_name,
                    const std::string &schema_name) {
  const std::string prefix = make_schema_prefix(schema_name);
  return index_name.rfind(prefix, 0) == 0;
}

bool matches_table(const std::string &index_name,
                   const std::string &schema_name,
                   const std::string &table_name) {
  const std::string prefix = make_table_prefix(schema_name, table_name);
  return index_name.rfind(prefix, 0) == 0;
}

bool replace_table_name(const std::string &index_name,
                        const std::string &old_schema_name,
                        const std::string &old_table_name,
                        const std::string &new_schema_name,
                        const std::string &new_table_name,
                        std::string *renamed_index_name) {
  if (renamed_index_name == nullptr) return false;
  *renamed_index_name = index_name;

  const std::string old_prefix =
      make_table_prefix(old_schema_name, old_table_name);
  if (index_name.rfind(old_prefix, 0) != 0) return true;

  *renamed_index_name = make_table_prefix(new_schema_name, new_table_name);
  renamed_index_name->append(index_name.substr(old_prefix.size()));
  return true;
}

}  // namespace vector_index_identity
