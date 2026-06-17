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

#include "sql/vector/vector_segment_task.h"

namespace vector_index_metadata_store {

const char *segment_task_state_to_string(segment_task_state state) {
  switch (state) {
    case segment_task_state::kPending:
      return "pending";
    case segment_task_state::kBuilding:
      return "building";
    case segment_task_state::kReady:
      return "ready";
    case segment_task_state::kFailed:
      return "failed";
    case segment_task_state::kAbandoned:
      return "abandoned";
  }
  return "pending";
}

bool parse_segment_task_state(const std::string &text,
                              segment_task_state *state) {
  if (state == nullptr) return false;
  if (text == "pending") {
    *state = segment_task_state::kPending;
    return true;
  }
  if (text == "building") {
    *state = segment_task_state::kBuilding;
    return true;
  }
  if (text == "ready") {
    *state = segment_task_state::kReady;
    return true;
  }
  if (text == "failed") {
    *state = segment_task_state::kFailed;
    return true;
  }
  if (text == "abandoned") {
    *state = segment_task_state::kAbandoned;
    return true;
  }
  return false;
}

}  // namespace vector_index_metadata_store
