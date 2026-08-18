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

#ifndef SQL_VECTOR_TRX_PARTICIPANT_INCLUDED
#define SQL_VECTOR_TRX_PARTICIPANT_INCLUDED

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
#include <cstdint>
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

class THD;
struct handlerton;

namespace vector_trx_participant {

int init_plugin(void *p);
int deinit_plugin(void *p);
/**
  Ensure the vector transaction lifecycle observer is registered.

  @return true when the observer is available, false otherwise.
*/
bool ensure_observer_registered();
/**
  Register the vector participant and its transaction lifecycle observer.

  @param thd Thread context.
  @return true when both registrations are available, false otherwise.
*/
bool register_participant(THD *thd);
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
uint64_t thd_id_for_testing(const THD *thd);
uint64_t stmt_id_for_testing(const THD *thd);
bool in_multi_stmt_for_testing(THD *thd);
bool is_real_scope_for_testing(THD *thd, bool all);
bool is_xa_commit_publication_fallback_for_testing(THD *thd,
                                                   uint64_t thread_id);
void set_registration_bypass_for_testing(bool bypass);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_trx_participant

#endif  // SQL_VECTOR_TRX_PARTICIPANT_INCLUDED
