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

#ifndef SQL_VECTOR_LOAD_INCLUDED
#define SQL_VECTOR_LOAD_INCLUDED

#include <string>

#include "lex_string.h"
#include "my_sqlcommand.h"
#include "sql/sql_cmd.h"

class THD;

/**
  Command skeleton for LOAD VECTOR DATA.

  The parser owns the SQL syntax even when vector indexing is compiled out, so
  this command is always available and reports a clear runtime error when the
  vector implementation is not present.
*/
class Sql_cmd_load_vector_index final : public Sql_cmd {
 public:
  Sql_cmd_load_vector_index(bool is_local_file,
                            const LEX_STRING &vector_filename,
                            const LEX_STRING &docid_filename,
                            const LEX_STRING &index_name,
                            const LEX_STRING &format,
                            bool replace_duplicates, bool rebuild_after_load);

  static constexpr enum_sql_command command_code() {
    return SQLCOM_LOAD_VECTOR;
  }

  enum_sql_command sql_command_code() const override { return command_code(); }

  bool execute(THD *thd) override;

 private:
  const bool m_is_local_file;
  const std::string m_vector_filename;
  const std::string m_docid_filename;
  const std::string m_index_name;
  const std::string m_format;
  const bool m_replace_duplicates;
  const bool m_rebuild_after_load;
};

#endif /* SQL_VECTOR_LOAD_INCLUDED */
