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

#ifndef SQL_VECTOR_INDEX_DIAGNOSTICS_INCLUDED
#define SQL_VECTOR_INDEX_DIAGNOSTICS_INCLUDED

#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

namespace vector_index_diagnostics {

using string_field = std::pair<const char *, std::string>;
using number_field = std::pair<const char *, uint64_t>;
using time_point = std::chrono::steady_clock::time_point;

time_point now();
uint64_t elapsed_ms(time_point start);
bool enabled();

inline time_point now_if(bool is_enabled) {
  return is_enabled ? now() : time_point{};
}

inline uint64_t elapsed_ms_if(bool is_enabled, time_point start) {
  return is_enabled ? elapsed_ms(start) : 0;
}

void record_event(const char *event,
                  std::initializer_list<string_field> string_fields,
                  std::initializer_list<number_field> number_fields);

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
std::string format_event_for_testing(
    const char *event, std::initializer_list<string_field> string_fields,
    std::initializer_list<number_field> number_fields);
bool append_event_for_testing(
    const char *path, const char *event,
    std::initializer_list<string_field> string_fields,
    std::initializer_list<number_field> number_fields);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index_diagnostics

#endif  // SQL_VECTOR_INDEX_DIAGNOSTICS_INCLUDED
