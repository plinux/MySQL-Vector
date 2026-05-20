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

#ifndef SQL_VECTOR_INDEX_IDENTITY_INCLUDED
#define SQL_VECTOR_INDEX_IDENTITY_INCLUDED

#include <string>

namespace vector_index_identity {

/** Parsed identity for mapped vector index names. */
struct mapped_index_identity {
  std::string schema_name;
  std::string table_name;
  std::string column_name;
};

/** Build the current mapped index name format: schema.table.column. */
std::string make_index_name(const std::string &schema_name,
                            const std::string &table_name,
                            const std::string &column_name);

/** Build the schema prefix used by mapped index scans. */
std::string make_schema_prefix(const std::string &schema_name);

/** Build the schema.table prefix used by mapped index scans. */
std::string make_table_prefix(const std::string &schema_name,
                              const std::string &table_name);

/** Parse the current mapped index name format. */
bool parse_index_name(const std::string &index_name,
                      mapped_index_identity *identity);

/** Return whether the mapped index name belongs to the given schema. */
bool matches_schema(const std::string &index_name,
                    const std::string &schema_name);

/** Return whether the mapped index name belongs to the given table. */
bool matches_table(const std::string &index_name,
                   const std::string &schema_name,
                   const std::string &table_name);

/** Replace the schema/table prefix while preserving the column suffix. */
bool replace_table_name(const std::string &index_name,
                        const std::string &old_schema_name,
                        const std::string &old_table_name,
                        const std::string &new_schema_name,
                        const std::string &new_table_name,
                        std::string *renamed_index_name);

}  // namespace vector_index_identity

#endif  // SQL_VECTOR_INDEX_IDENTITY_INCLUDED
