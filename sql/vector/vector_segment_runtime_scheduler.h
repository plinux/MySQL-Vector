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

#ifndef SQL_VECTOR_SEGMENT_RUNTIME_SCHEDULER_INCLUDED
#define SQL_VECTOR_SEGMENT_RUNTIME_SCHEDULER_INCLUDED

#include <cstdint>

namespace vector_index {

struct segment_scheduler_input {
  uint32_t segment_count{0};
  uint32_t requested_task_count{0};
  uint32_t cpu_budget{0};
  uint32_t requested_build_threads{0};
  uint32_t requested_blas_threads{1};
  uint32_t top_k{0};
  uint32_t search_candidate_multiplier{1};
};

struct segment_scheduler_plan {
  uint32_t task_count{0};
  uint32_t effective_build_threads{0};
  uint32_t effective_blas_threads{1};
  uint32_t candidates_per_segment{0};
};

/**
  Build a backend-independent segment scheduler plan.

  @param input Segment count, thread requests and search candidate policy.
  @param plan Output scheduler plan.

  @retval true The plan is valid.
  @retval false The input cannot produce a safe runtime plan.
*/
bool make_segment_scheduler_plan(const segment_scheduler_input &input,
                                 segment_scheduler_plan *plan);

}  // namespace vector_index

#endif  // SQL_VECTOR_SEGMENT_RUNTIME_SCHEDULER_INCLUDED
