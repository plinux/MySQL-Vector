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

#ifndef SQL_VECTOR_VECTOR_ASCII_H_
#define SQL_VECTOR_VECTOR_ASCII_H_

#include <cstddef>
#include <string_view>

namespace vector_ascii {

/**
  Compare ASCII tokens without regard to letter case.

  Bytes outside the ASCII uppercase range are compared unchanged so the result
  is deterministic and independent of the process locale.
*/
inline bool equal_ignore_case(std::string_view lhs, std::string_view rhs) {
  if (lhs.size() != rhs.size()) return false;

  for (std::size_t index = 0; index < lhs.size(); ++index) {
    unsigned char lhs_value = static_cast<unsigned char>(lhs[index]);
    unsigned char rhs_value = static_cast<unsigned char>(rhs[index]);
    if (lhs_value >= 'A' && lhs_value <= 'Z') lhs_value += 'a' - 'A';
    if (rhs_value >= 'A' && rhs_value <= 'Z') rhs_value += 'a' - 'A';
    if (lhs_value != rhs_value) return false;
  }
  return true;
}

}  // namespace vector_ascii

#endif  // SQL_VECTOR_VECTOR_ASCII_H_
