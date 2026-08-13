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

#include <gtest/gtest.h>

#include "mysql/plugin.h"
#include "sql/handler.h"
#include "sql/query_options.h"
#include "sql/replication.h"
#include "sql/sql_class.h"
#include "sql/sql_lex.h"
#include "sql/vector/vector_trx_participant.h"
#include "unittest/gunit/test_utils.h"

namespace vector_trx_participant_unittest {

TEST(VectorTrxParticipantTest,
     InitPluginInstallsLifecycleCallbacksWithoutDurableXaRecovery) {
  handlerton hton{};
  hton.slot = HA_SLOT_UNDEF;

  ASSERT_EQ(0, vector_trx_participant::init_plugin(&hton));
  EXPECT_EQ(SHOW_OPTION_YES, hton.state);
  EXPECT_EQ(DB_TYPE_UNKNOWN, hton.db_type);
  EXPECT_EQ(0U, hton.savepoint_offset);
  EXPECT_NE(nullptr, hton.close_connection);
  EXPECT_NE(nullptr, hton.savepoint_set);
  EXPECT_NE(nullptr, hton.savepoint_rollback);
  EXPECT_NE(nullptr, hton.savepoint_rollback_can_release_mdl);
  EXPECT_NE(nullptr, hton.savepoint_release);
  EXPECT_NE(nullptr, hton.commit);
  EXPECT_NE(nullptr, hton.rollback);
  EXPECT_NE(nullptr, hton.prepare);
  EXPECT_EQ(nullptr, hton.se_after_commit);
  EXPECT_EQ(nullptr, hton.se_before_rollback);
  EXPECT_EQ(nullptr, hton.recover);
  EXPECT_EQ(nullptr, hton.recover_prepared_in_tc);
  EXPECT_EQ(nullptr, hton.commit_by_xid);
  EXPECT_EQ(nullptr, hton.rollback_by_xid);
  EXPECT_NE(nullptr, hton.set_prepared_in_tc);
  EXPECT_EQ(nullptr, hton.set_prepared_in_tc_by_xid);
  EXPECT_NE(0U, hton.flags & HTON_NOT_USER_SELECTABLE);
  EXPECT_NE(0U, hton.flags & HTON_HIDDEN);
  EXPECT_NE(0U, hton.flags & HTON_NO_GLOBAL_2PC);

  EXPECT_EQ(0, hton.savepoint_set(&hton, nullptr, nullptr));
  EXPECT_EQ(0, hton.savepoint_rollback(&hton, nullptr, nullptr));
  EXPECT_TRUE(hton.savepoint_rollback_can_release_mdl(&hton, nullptr));
  EXPECT_EQ(0, hton.savepoint_release(&hton, nullptr, nullptr));
  EXPECT_EQ(0, hton.close_connection(&hton, nullptr));
  EXPECT_EQ(0, hton.prepare(&hton, nullptr, true));
  EXPECT_EQ(0, hton.set_prepared_in_tc(&hton, nullptr));
  EXPECT_EQ(0, hton.commit(&hton, nullptr, true));
  EXPECT_EQ(0, hton.rollback(&hton, nullptr, true));

  EXPECT_FALSE(vector_trx_participant::ensure_observer_registered());
  EXPECT_FALSE(vector_trx_participant::register_participant(nullptr));
  EXPECT_EQ(0, vector_trx_participant::deinit_plugin(&hton));
}

TEST(VectorTrxParticipantTest, TestingWrappersCoverTransactionScopeHelpers) {
  EXPECT_EQ(0U, vector_trx_participant::thd_id_for_testing(nullptr));
  EXPECT_EQ(0U, vector_trx_participant::stmt_id_for_testing(nullptr));
  EXPECT_FALSE(vector_trx_participant::in_multi_stmt_for_testing(nullptr));
  EXPECT_TRUE(
      vector_trx_participant::is_real_scope_for_testing(nullptr, false));
  EXPECT_TRUE(vector_trx_participant::is_real_scope_for_testing(nullptr, true));

  my_testing::Server_initializer initializer;
  initializer.SetUp();
  THD *thd = initializer.thd();
  EXPECT_NE(0U, vector_trx_participant::thd_id_for_testing(thd));
  EXPECT_FALSE(vector_trx_participant::in_multi_stmt_for_testing(thd));
  EXPECT_TRUE(vector_trx_participant::is_real_scope_for_testing(thd, false));
  const ulonglong original_option_bits = thd->variables.option_bits;
  thd->variables.option_bits |= OPTION_BEGIN;
  EXPECT_TRUE(vector_trx_participant::in_multi_stmt_for_testing(thd));
  EXPECT_FALSE(vector_trx_participant::is_real_scope_for_testing(thd, false));
  EXPECT_TRUE(vector_trx_participant::is_real_scope_for_testing(thd, true));
  thd->variables.option_bits = original_option_bits;

  LEX *const original_lex = thd->lex;
  thd->lex = nullptr;
  EXPECT_FALSE(
      vector_trx_participant::is_xa_commit_publication_fallback_for_testing(
          thd, thd->thread_id()));
  thd->lex = original_lex;
  const enum_sql_command original_command = thd->lex->sql_command;
  EXPECT_FALSE(
      vector_trx_participant::is_xa_commit_publication_fallback_for_testing(
          nullptr, thd->thread_id()));
  EXPECT_FALSE(
      vector_trx_participant::is_xa_commit_publication_fallback_for_testing(
          thd, thd->thread_id() + 1));
  thd->lex->sql_command = SQLCOM_SELECT;
  EXPECT_FALSE(
      vector_trx_participant::is_xa_commit_publication_fallback_for_testing(
          thd, thd->thread_id()));
  thd->lex->sql_command = SQLCOM_XA_COMMIT;
  EXPECT_TRUE(
      vector_trx_participant::is_xa_commit_publication_fallback_for_testing(
          thd, thd->thread_id()));
  thd->lex->sql_command = original_command;
  initializer.TearDown();
}

TEST(VectorTrxParticipantTest,
     LifecycleCallbacksCoverStatementTransactionAndObserverBoundaries) {
  handlerton hton{};
  hton.slot = HA_SLOT_UNDEF;
  ASSERT_EQ(0, vector_trx_participant::init_plugin(&hton));

  my_testing::Server_initializer initializer;
  initializer.SetUp();
  THD *thd = initializer.thd();

  EXPECT_FALSE(vector_trx_participant::has_context_for_testing(nullptr));
  EXPECT_EQ(0, vector_trx_participant::prepare_for_testing(nullptr, true));
  EXPECT_EQ(0, vector_trx_participant::commit_for_testing(nullptr, true));
  EXPECT_EQ(0, vector_trx_participant::rollback_for_testing(nullptr, true));

  vector_trx_participant::set_context_for_testing(thd, true);
  ASSERT_TRUE(vector_trx_participant::has_context_for_testing(thd));
  const ulonglong original_option_bits = thd->variables.option_bits;
  thd->variables.option_bits |= OPTION_BEGIN;
  EXPECT_EQ(0, vector_trx_participant::prepare_for_testing(thd, false));
  EXPECT_EQ(0, vector_trx_participant::commit_for_testing(thd, false));
  EXPECT_EQ(0, vector_trx_participant::rollback_for_testing(thd, false));
  EXPECT_TRUE(vector_trx_participant::has_context_for_testing(thd));

  EXPECT_EQ(0, vector_trx_participant::prepare_for_testing(thd, true));
  EXPECT_EQ(0, vector_trx_participant::commit_for_testing(thd, true));
  EXPECT_FALSE(vector_trx_participant::has_context_for_testing(thd));
  EXPECT_TRUE(thd->get_transaction()->m_flags.run_hooks);
  thd->variables.option_bits = original_option_bits;

  vector_trx_participant::after_commit_for_testing(nullptr);
  vector_trx_participant::before_rollback_for_testing(nullptr);

  Trans_param statement_param{};
  statement_param.thread_id = thd->thread_id();
  vector_trx_participant::after_commit_for_testing(&statement_param);
  vector_trx_participant::before_rollback_for_testing(&statement_param);

  Trans_param transaction_param{};
  transaction_param.thread_id = thd->thread_id();
  transaction_param.flags = TRANS_IS_REAL_TRANS;
  vector_trx_participant::after_commit_for_testing(&transaction_param);
  vector_trx_participant::before_rollback_for_testing(&transaction_param);

  vector_trx_participant::set_context_for_testing(thd, true);
  ASSERT_TRUE(vector_trx_participant::has_context_for_testing(thd));
  EXPECT_EQ(0, vector_trx_participant::close_connection_for_testing(thd));
  EXPECT_FALSE(vector_trx_participant::has_context_for_testing(thd));

  vector_trx_participant::set_registration_bypass_for_testing(true);
  EXPECT_FALSE(vector_trx_participant::register_participant(nullptr));
  EXPECT_TRUE(vector_trx_participant::register_participant(thd));
  vector_trx_participant::set_registration_bypass_for_testing(false);

  initializer.TearDown();
  EXPECT_EQ(0, vector_trx_participant::deinit_plugin(&hton));
}

TEST(VectorTrxParticipantTest,
     ExplicitOwnerRoleSurvivesSqlTransactionCompletionUntilConnectionClose) {
  handlerton hton{};
  hton.slot = HA_SLOT_UNDEF;
  ASSERT_EQ(0, vector_trx_participant::init_plugin(&hton));

  my_testing::Server_initializer initializer;
  initializer.SetUp();
  THD *thd = initializer.thd();

  vector_trx_participant::set_registration_bypass_for_testing(true);
  vector_trx_participant::set_context_for_testing(thd, true);
  ASSERT_TRUE(vector_trx_participant::register_explicit_txn_owner(thd));
  ASSERT_TRUE(vector_trx_participant::has_context_for_testing(thd));
  ASSERT_TRUE(
      vector_trx_participant::has_explicit_txn_owner_for_testing(thd));

  ASSERT_EQ(0, vector_trx_participant::commit_for_testing(thd, true));
  EXPECT_FALSE(vector_trx_participant::has_context_for_testing(thd));
  EXPECT_TRUE(
      vector_trx_participant::has_explicit_txn_owner_for_testing(thd));

  ASSERT_EQ(0, vector_trx_participant::close_connection_for_testing(thd));
  EXPECT_FALSE(vector_trx_participant::has_context_for_testing(thd));
  EXPECT_FALSE(
      vector_trx_participant::has_explicit_txn_owner_for_testing(thd));

  vector_trx_participant::set_registration_bypass_for_testing(false);
  initializer.TearDown();
  EXPECT_EQ(0, vector_trx_participant::deinit_plugin(&hton));
}

TEST(VectorTrxParticipantTest,
     ParticipantAndExplicitOwnerRolesCanBeReleasedIndependently) {
  handlerton hton{};
  hton.slot = HA_SLOT_UNDEF;
  ASSERT_EQ(0, vector_trx_participant::init_plugin(&hton));

  my_testing::Server_initializer initializer;
  initializer.SetUp();
  THD *thd = initializer.thd();

  vector_trx_participant::set_context_for_testing(thd, false);
  vector_trx_participant::set_registration_bypass_for_testing(true);
  EXPECT_FALSE(vector_trx_participant::register_explicit_txn_owner(nullptr));
  vector_trx_participant::release_explicit_txn_owner(nullptr);

  ASSERT_TRUE(vector_trx_participant::register_participant(thd));
  EXPECT_TRUE(vector_trx_participant::has_context_for_testing(thd));
  EXPECT_FALSE(vector_trx_participant::has_explicit_txn_owner_for_testing(thd));

  ASSERT_TRUE(vector_trx_participant::register_explicit_txn_owner(thd));
  EXPECT_TRUE(vector_trx_participant::has_context_for_testing(thd));
  EXPECT_TRUE(vector_trx_participant::has_explicit_txn_owner_for_testing(thd));

  vector_trx_participant::release_explicit_txn_owner(thd);
  EXPECT_TRUE(vector_trx_participant::has_context_for_testing(thd));
  EXPECT_FALSE(vector_trx_participant::has_explicit_txn_owner_for_testing(thd));

  ASSERT_EQ(0, vector_trx_participant::close_connection_for_testing(thd));
  EXPECT_FALSE(vector_trx_participant::has_context_for_testing(thd));
  vector_trx_participant::set_context_for_testing(nullptr, false);
  vector_trx_participant::set_context_for_testing(thd, false);
  EXPECT_FALSE(vector_trx_participant::has_context_for_testing(thd));
  vector_trx_participant::set_registration_bypass_for_testing(false);

  initializer.TearDown();
  EXPECT_EQ(0, vector_trx_participant::deinit_plugin(&hton));
}

}  // namespace vector_trx_participant_unittest
