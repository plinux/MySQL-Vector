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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "my_dbug.h"
#include "mysqld_error.h"
#include "sql/item.h"
#include "sql/sql_lex.h"
#include "sql/vector/vector_index_registry.h"
#include "sql_string.h"

bool Item_func_vector_mutator::itemize(Parse_context *pc, Item **res) {
  if (skip_itemize(res)) return false;
  if (Item_int_func::itemize(pc, res)) return true;
  pc->thd->lex->set_uncacheable(pc->select, UNCACHEABLE_SIDEEFFECT);
  return false;
}

bool Item_func_vector_mutator::check_function_as_value_generator(
    uchar *checker_args) {
  auto *parameters =
      pointer_cast<Check_function_as_value_generator_parameters *>(
          checker_args);
  parameters->banned_function_name = func_name();
  return true;
}

bool Item_func_vec_index_create::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, arg_count)) return true;
  set_nullable(false);
  return false;
}
