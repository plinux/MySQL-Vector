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

#ifndef SQL_VECTOR_SEGMENT_TASK_INCLUDED
#define SQL_VECTOR_SEGMENT_TASK_INCLUDED

#include <cstdint>
#include <string>

namespace vector_index_metadata_store {

/**
  State of one persisted segmented-build task.
*/
enum class segment_task_state {
  kPending,
  kBuilding,
  kReady,
  kFailed,
  kAbandoned
};

/**
  Persisted metadata for one segment build task.

  The task row is the durable recovery boundary between raw segment input and
  published serving artifacts. The executor may rebuild pending/building tasks
  after restart, while ready tasks can be loaded when their artifacts are still
  complete.
*/
struct segment_task_row {
  std::string index_name;
  uint64_t generation{0};
  uint64_t segment_id{0};
  segment_task_state state{segment_task_state::kPending};
  uint64_t row_count{0};
  uint64_t payload_size{0};
  std::string vector_path;
  std::string docid_path;
  std::string artifact_prefix;
  uint32_t attempt{0};
  uint32_t last_error_code{0};
  uint64_t updated_ts{0};
};

const char *segment_task_state_to_string(segment_task_state state);
bool parse_segment_task_state(const std::string &text,
                              segment_task_state *state);

}  // namespace vector_index_metadata_store

#endif  // SQL_VECTOR_SEGMENT_TASK_INCLUDED
