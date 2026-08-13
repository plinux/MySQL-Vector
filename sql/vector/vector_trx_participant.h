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

#include <vector>

#include "sql/vector/vector_index_truth_store.h"

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
#include <cstdint>
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

class THD;
struct handlerton;
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
struct Trans_param;
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

namespace vector_trx_participant {

int init_plugin(void *p);
int deinit_plugin(void *p);
/**
  Ensure the vector transaction lifecycle observer is registered.

  @return true when the observer is available, false otherwise.
*/
bool ensure_observer_registered();
/**
  Return whether a user statement is inside a multi-statement transaction.

  Replication appliers own transaction state while replaying source events, but
  that state must not trigger restrictions intended for interactive users.

  @param thd Thread context.
  @return true only for a user-controlled transaction mode or active
  multi-statement transaction.
*/
bool in_user_multi_statement_transaction(THD *thd);
/**
  Register the vector participant and its transaction lifecycle observer.

  @param thd Thread context.
  @return true when both registrations are available, false otherwise.
*/
bool register_participant(THD *thd);
/**
  Stage one statement-scoped publication for after-commit installation.

  @param thd current user thread
  @param intents per-index durable operations published after commit
  @param catalog_exclusive true for CREATE/DROP or stable all-index sets
  @return true when the participant owns the staged operations
*/
bool stage_statement_publication(
    THD *thd,
    std::vector<vector_index_truth_store::publication_intent> intents,
    bool catalog_exclusive, bool require_stable_catalog_set = false);
/** Mark a connection as owning one or more explicit vector transactions. */
bool register_explicit_txn_owner(THD *thd);
/** Clear explicit ownership after the connection's last transaction ends. */
void release_explicit_txn_owner(THD *thd);
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
uint64_t thd_id_for_testing(const THD *thd);
uint64_t stmt_id_for_testing(const THD *thd);
bool in_multi_stmt_for_testing(THD *thd);
bool is_real_scope_for_testing(THD *thd, bool all);
bool is_xa_commit_publication_fallback_for_testing(THD *thd,
                                                   uint64_t thread_id);
void set_registration_bypass_for_testing(bool bypass);
void set_context_for_testing(THD *thd, bool active);
bool has_context_for_testing(THD *thd);
bool has_explicit_txn_owner_for_testing(THD *thd);
int prepare_for_testing(THD *thd, bool all);
int commit_for_testing(THD *thd, bool all);
int rollback_for_testing(THD *thd, bool all);
int close_connection_for_testing(THD *thd);
void after_commit_for_testing(Trans_param *param);
void before_rollback_for_testing(Trans_param *param);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_trx_participant

#endif  // SQL_VECTOR_TRX_PARTICIPANT_INCLUDED
