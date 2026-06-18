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

#include "sql/vector/vector_index_observability.h"

#include <cctype>
#include <cstddef>
#include <cstring>
#include <string>

namespace vector_index_observability {
namespace {

bool ascii_equal_ignore_case(const std::string &lhs, const char *rhs) {
  const size_t rhs_len = std::strlen(rhs);
  if (lhs.size() != rhs_len) return false;
  for (size_t i = 0; i < rhs_len; ++i) {
    const auto left = static_cast<unsigned char>(lhs[i]);
    const auto right = static_cast<unsigned char>(rhs[i]);
    if (std::tolower(left) != std::tolower(right)) return false;
  }
  return true;
}

}  // namespace

uint64_t pending_apply_count(const vector_index_registry::index_info &info) {
  const auto entry_count = static_cast<uint64_t>(info.entry_count);
  const auto committed_entry_count =
      static_cast<uint64_t>(info.committed_entry_count);
  return entry_count > committed_entry_count
             ? entry_count - committed_entry_count
             : committed_entry_count - entry_count;
}

uint64_t rebuild_progress(const vector_index_registry::index_info &info) {
  if (ascii_equal_ignore_case(info.lifecycle_state, "ready")) return 100;
  if (ascii_equal_ignore_case(info.lifecycle_state, "rebuilding")) return 50;
  return 0;
}

uint64_t recover_progress(const vector_index_registry::index_info &info) {
  if (ascii_equal_ignore_case(info.lifecycle_state, "ready")) return 100;
  if (ascii_equal_ignore_case(info.lifecycle_state, "recovering")) return 50;
  return 0;
}

bool is_loaded(const vector_index_registry::index_info &info) {
  return !ascii_equal_ignore_case(info.lifecycle_state, "failed");
}

bool is_writable(const vector_index_registry::index_info &info) {
  return is_loaded(info) && info.supports_mutations;
}

}  // namespace vector_index_observability
