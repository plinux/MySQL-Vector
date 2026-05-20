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

#ifndef SQL_VECTOR_INDEX_DDL_FORMATTER_INCLUDED
#define SQL_VECTOR_INDEX_DDL_FORMATTER_INCLUDED

#include <cstddef>
#include <cstdint>
#include <string>

#include "sql/vector/vector_index_registry.h"

class String;
class THD;

namespace vector_index_ddl_formatter {

enum class tuning_emit_policy { k_skip_defaults, k_emit_nonzero };

uint32_t provider_build_threads(
    const vector_index_registry::index_info &info);
uint32_t first_nonzero_build_threads(
    const vector_index_registry::index_info &info);

void append_create_statement(
    THD *thd, String *query, const char *db_name, size_t db_name_length,
    const char *table_name, size_t table_name_length, bool qualify_table,
    const std::string &column_name,
    const vector_index_registry::index_info &info, uint32_t build_threads);

void append_tuning_statements(
    THD *thd, String *query, const std::string &index_name,
    const vector_index_registry::index_info &info,
    tuning_emit_policy emit_policy);

}  // namespace vector_index_ddl_formatter

#endif  // SQL_VECTOR_INDEX_DDL_FORMATTER_INCLUDED
