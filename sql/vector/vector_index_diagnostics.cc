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

#include "sql/vector/vector_index_diagnostics.h"

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>

namespace {

constexpr const char *kDiagnosticsPathEnv = "MYSQL_VECTOR_DIAG_FILE";

std::mutex g_diagnostics_mutex;

const char *diagnostics_path() {
  const char *path = std::getenv(kDiagnosticsPathEnv);
  return (path != nullptr && path[0] != '\0') ? path : nullptr;
}

uint64_t system_time_ms() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

std::string escape_json_string(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char ch : value) {
    switch (ch) {
      case '\\':
        escaped.append("\\\\");
        break;
      case '"':
        escaped.append("\\\"");
        break;
      case '\n':
        escaped.append("\\n");
        break;
      case '\r':
        escaped.append("\\r");
        break;
      case '\t':
        escaped.append("\\t");
        break;
      default:
        escaped.push_back(ch);
        break;
    }
  }
  return escaped;
}

std::string format_event(
    const char *event,
    std::initializer_list<vector_index_diagnostics::string_field> string_fields,
    std::initializer_list<vector_index_diagnostics::number_field> number_fields) {
  std::ostringstream out;
  out << "{\"ts_ms\":" << system_time_ms() << ",\"event\":\""
      << escape_json_string(event == nullptr ? "" : event) << "\"";
  for (const auto &field : string_fields) {
    out << ",\"" << escape_json_string(field.first == nullptr ? "" : field.first)
        << "\":\"" << escape_json_string(field.second) << "\"";
  }
  for (const auto &field : number_fields) {
    out << ",\"" << escape_json_string(field.first == nullptr ? "" : field.first)
        << "\":" << field.second;
  }
  out << "}";
  return out.str();
}

bool append_event(
    const char *path, const char *event,
    std::initializer_list<vector_index_diagnostics::string_field> string_fields,
    std::initializer_list<vector_index_diagnostics::number_field> number_fields) {
  if (path == nullptr || path[0] == '\0') return false;
  std::lock_guard<std::mutex> guard(g_diagnostics_mutex);
  std::ofstream file(path, std::ios::out | std::ios::app);
  if (!file.good()) return false;
  file << format_event(event, string_fields, number_fields) << '\n';
  return file.good();
}

}  // namespace

namespace vector_index_diagnostics {

time_point now() { return std::chrono::steady_clock::now(); }

uint64_t elapsed_ms(time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(now() - start)
          .count());
}

bool enabled() { return diagnostics_path() != nullptr; }

void record_event(const char *event,
                  std::initializer_list<string_field> string_fields,
                  std::initializer_list<number_field> number_fields) {
  const char *path = diagnostics_path();
  if (path == nullptr) return;
  (void)append_event(path, event, string_fields, number_fields);
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
std::string format_event_for_testing(
    const char *event, std::initializer_list<string_field> string_fields,
    std::initializer_list<number_field> number_fields) {
  return format_event(event, string_fields, number_fields);
}

bool append_event_for_testing(const char *path, const char *event,
                              std::initializer_list<string_field> string_fields,
                              std::initializer_list<number_field> number_fields) {
  return append_event(path, event, string_fields, number_fields);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index_diagnostics
