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

#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "sql/handler.h"
#include "sql/mysqld.h"
#include "sql/query_options.h"
#include "sql/sql_class.h"
#include "sql/sql_lex.h"
#include "sql/vector/vector_index_metadata_store.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_trx_participant.h"
#include "sql/xa.h"
#include "unittest/gunit/test_utils.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_trx_participant_unittest {

namespace {

class ServerStartedGuard {
 public:
  explicit ServerStartedGuard(bool started)
      : m_original(mysqld_server_started) {
    mysqld_server_started = started;
  }

  ~ServerStartedGuard() { mysqld_server_started = m_original; }

 private:
  bool m_original;
};

class OptInitializeGuard {
 public:
  explicit OptInitializeGuard(bool enabled) : m_original(opt_initialize) {
    opt_initialize = enabled;
  }

  ~OptInitializeGuard() { opt_initialize = m_original; }

 private:
  bool m_original;
};

class PreparedRowsTruthStore final : public vector_index_truth_store::truth_store {
 public:
  const char *backend_name() const override { return "prepared_rows"; }
  bool is_transactional() const override { return false; }

  bool load_metadata(
      std::vector<vector_index_metadata_store::metadata_row> *out) override {
    if (out != nullptr) out->clear();
    return true;
  }
  bool save_metadata(
      const std::vector<vector_index_metadata_store::metadata_row> &) override {
    return true;
  }
  bool quarantine_metadata() override { return true; }

  bool load_committed(
      std::vector<vector_index_metadata_store::committed_row> *out) override {
    if (out != nullptr) out->clear();
    return true;
  }
  bool save_committed(
      const std::vector<vector_index_metadata_store::committed_row> &) override {
    return true;
  }
  bool quarantine_committed() override { return true; }

  bool load_manifest(vector_index_metadata_store::manifest_row *row) override {
    if (row != nullptr) *row = vector_index_metadata_store::manifest_row();
    return true;
  }
  bool save_manifest(
      const vector_index_metadata_store::manifest_row &) override {
    return true;
  }
  bool quarantine_manifest() override { return true; }

  bool load_change_log(
      std::vector<vector_index_metadata_store::change_log_row> *out) override {
    if (out != nullptr) out->clear();
    return true;
  }
  bool save_change_log(
      const std::vector<vector_index_metadata_store::change_log_row> &) override {
    return true;
  }
  bool quarantine_change_log() override { return true; }

  bool load_prepared(
      std::vector<vector_index_metadata_store::prepared_change_row> *out)
      override {
    ++load_prepared_calls;
    if (fail_load_prepared) return false;
    if (out != nullptr) *out = rows;
    return true;
  }
  bool save_prepared(
      const std::vector<vector_index_metadata_store::prepared_change_row> &)
      override {
    return true;
  }
  bool quarantine_prepared() override {
    quarantine_prepared_called = true;
    return allow_quarantine_prepared;
  }

  std::vector<vector_index_metadata_store::prepared_change_row> rows;
  int load_prepared_calls{0};
  bool fail_load_prepared{false};
  bool allow_quarantine_prepared{true};
  bool quarantine_prepared_called{false};
};

class TruthStoreGuard {
 public:
  explicit TruthStoreGuard(vector_index_truth_store::truth_store *store) {
    vector_index_truth_store::set_for_testing(store);
  }

  ~TruthStoreGuard() {
    vector_index_truth_store::reset_for_testing();
    vector_index_registry::reset_for_testing();
  }

  TruthStoreGuard(const TruthStoreGuard &) = delete;
  TruthStoreGuard &operator=(const TruthStoreGuard &) = delete;
};

vector_index_metadata_store::prepared_change_row make_prepared_row(
    int64_t format_id, const std::string &gtrid, const std::string &bqual,
    bool prepared_in_tc = false) {
  vector_index_metadata_store::prepared_change_row row;
  row.format_id = format_id;
  row.gtrid_length = static_cast<int64_t>(gtrid.size());
  row.bqual_length = static_cast<int64_t>(bqual.size());
  row.xid_data = gtrid + bqual;
  row.prepared_in_tc = prepared_in_tc;
  row.txn_id = static_cast<uint64_t>(format_id);
  row.index_name = "idx_prepared";
  row.doc_id = static_cast<uint64_t>(format_id);
  row.vector = {1.0F, 2.0F};
  return row;
}

XID make_xid(int64_t format_id, const std::string &gtrid,
             const std::string &bqual) {
  XID xid;
  xid.set(static_cast<long>(format_id), gtrid.data(),
          static_cast<long>(gtrid.size()), bqual.data(),
          static_cast<long>(bqual.size()));
  return xid;
}

}  // namespace

TEST(VectorTrxParticipantTest,
     InitPluginInstallsHiddenParticipantCallbacksAndNoopSavepointHooks) {
  handlerton hton{};

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
  EXPECT_NE(nullptr, hton.recover);
  EXPECT_NE(nullptr, hton.recover_prepared_in_tc);
  EXPECT_NE(nullptr, hton.commit_by_xid);
  EXPECT_NE(nullptr, hton.rollback_by_xid);
  EXPECT_NE(nullptr, hton.set_prepared_in_tc);
  EXPECT_NE(nullptr, hton.set_prepared_in_tc_by_xid);
  EXPECT_NE(0U, hton.flags & HTON_NOT_USER_SELECTABLE);
  EXPECT_NE(0U, hton.flags & HTON_HIDDEN);

  EXPECT_EQ(0, hton.savepoint_set(&hton, nullptr, nullptr));
  EXPECT_EQ(0, hton.savepoint_rollback(&hton, nullptr, nullptr));
  EXPECT_TRUE(hton.savepoint_rollback_can_release_mdl(&hton, nullptr));
  EXPECT_EQ(0, hton.savepoint_release(&hton, nullptr, nullptr));
  EXPECT_EQ(0, hton.close_connection(&hton, nullptr));
  EXPECT_EQ(0, hton.prepare(&hton, nullptr, true));
  EXPECT_EQ(HA_ERR_INTERNAL_ERROR, hton.set_prepared_in_tc(&hton, nullptr));
  EXPECT_EQ(XAER_INVAL, hton.commit_by_xid(&hton, nullptr));
  EXPECT_EQ(XAER_INVAL, hton.rollback_by_xid(&hton, nullptr));
  EXPECT_EQ(XAER_INVAL, hton.set_prepared_in_tc_by_xid(&hton, nullptr));

  XA_recover_txn txn_list[1]{};
  EXPECT_EQ(0, hton.recover(&hton, nullptr, 1, nullptr));
  EXPECT_EQ(0, hton.recover(&hton, txn_list, 0, nullptr));

  XID xid;
  xid.set(123, "g", 1, "b", 1);
  EXPECT_TRUE(xid.eq(&xid));

  {
    ServerStartedGuard server_not_started(false);
    EXPECT_EQ(XA_OK, hton.commit_by_xid(&hton, &xid));
    EXPECT_EQ(XA_OK, hton.rollback_by_xid(&hton, &xid));
    EXPECT_EQ(XA_OK, hton.set_prepared_in_tc_by_xid(&hton, &xid));
  }

  vector_trx_participant::register_participant(nullptr);
  vector_index_registry::reset_for_testing();
  EXPECT_EQ(0, vector_trx_participant::deinit_plugin(&hton));
}

TEST(VectorTrxParticipantTest, TestingWrappersCoverHelperEdges) {
  EXPECT_EQ(0U, vector_trx_participant::thd_id_for_testing(nullptr));
  EXPECT_EQ(0U, vector_trx_participant::stmt_id_for_testing(nullptr));
  EXPECT_FALSE(vector_trx_participant::in_multi_stmt_for_testing(nullptr));
  EXPECT_TRUE(vector_trx_participant::is_real_scope_for_testing(nullptr, false));
  EXPECT_TRUE(vector_trx_participant::is_real_scope_for_testing(nullptr, true));
  EXPECT_FALSE(
      vector_trx_participant::skip_recovery_for_bootstrap_for_testing(nullptr));
  EXPECT_FALSE(vector_trx_participant::load_prepared_rows_for_testing(nullptr));

  {
    OptInitializeGuard bootstrap(true);
    EXPECT_TRUE(
        vector_trx_participant::skip_recovery_for_bootstrap_for_testing(nullptr));
  }

  my_testing::Server_initializer initializer;
  initializer.SetUp();
  THD *thd = initializer.thd();
  int savepoint_marker = 0;
  void *savepoint = &savepoint_marker;
  const std::string fallback_token =
      "ha:" + std::to_string(reinterpret_cast<uintptr_t>(savepoint));
  EXPECT_EQ(fallback_token,
            vector_trx_participant::savepoint_token_name_for_testing(
                nullptr, savepoint));

  char savepoint_name[] = "sp1";
  LEX_STRING original_ident = thd->lex->ident;
  thd->lex->ident = {savepoint_name, strlen(savepoint_name)};
  EXPECT_EQ("sp1", vector_trx_participant::savepoint_token_name_for_testing(
                       thd, savepoint));
  thd->lex->ident = {nullptr, strlen(savepoint_name)};
  EXPECT_EQ(fallback_token,
            vector_trx_participant::savepoint_token_name_for_testing(
                thd, savepoint));
  thd->lex->ident = {savepoint_name, 0};
  EXPECT_EQ(fallback_token,
            vector_trx_participant::savepoint_token_name_for_testing(
                thd, savepoint));
  thd->lex->ident = original_ident;

  EXPECT_NE(0U, vector_trx_participant::thd_id_for_testing(thd));
  EXPECT_FALSE(vector_trx_participant::in_multi_stmt_for_testing(thd));
  EXPECT_TRUE(vector_trx_participant::is_real_scope_for_testing(thd, false));
  const ulonglong original_option_bits = thd->variables.option_bits;
  thd->variables.option_bits |= OPTION_BEGIN;
  EXPECT_TRUE(vector_trx_participant::in_multi_stmt_for_testing(thd));
  EXPECT_FALSE(vector_trx_participant::is_real_scope_for_testing(thd, false));
  EXPECT_TRUE(vector_trx_participant::is_real_scope_for_testing(thd, true));
  thd->variables.option_bits = original_option_bits;

  const enum_thread_type original_system_thread = thd->system_thread;
  thd->system_thread = SYSTEM_THREAD_SERVER_INITIALIZE;
  EXPECT_TRUE(vector_trx_participant::skip_recovery_for_bootstrap_for_testing(thd));
  thd->system_thread = original_system_thread;
  initializer.TearDown();
}

TEST(VectorTrxParticipantTest, RecoverLoadsPreparedRowsAndDeduplicatesXids) {
  handlerton hton{};
  PreparedRowsTruthStore store;
  store.rows = {
      make_prepared_row(123, "g1", "b1"),
      make_prepared_row(123, "g1", "b1"),
      make_prepared_row(124, "g2", "b2", true),
  };
  TruthStoreGuard truth_store_guard(&store);

  ASSERT_EQ(0, vector_trx_participant::init_plugin(&hton));

  XA_recover_txn one_txn[1]{};
  EXPECT_EQ(1, hton.recover(&hton, one_txn, 1, nullptr));
  XID xid_1 = make_xid(123, "g1", "b1");
  EXPECT_TRUE(one_txn[0].id.eq(&xid_1));

  XA_recover_txn txn_list[4]{};
  EXPECT_EQ(2, hton.recover(&hton, txn_list, 4, nullptr));
  XID xid_2 = make_xid(124, "g2", "b2");
  EXPECT_TRUE(txn_list[0].id.eq(&xid_1));
  EXPECT_TRUE(txn_list[1].id.eq(&xid_2));

  auto xa_state_tuple = vector_gunit::make_xa_state_list_for_testing();
  auto *xa_state_list = xa_state_tuple.get();
  ASSERT_NE(nullptr, xa_state_list);
  EXPECT_EQ(0, hton.recover_prepared_in_tc(&hton, *xa_state_list));
  EXPECT_EQ(enum_ha_recover_xa_state::NOT_FOUND, xa_state_list->find(xid_1));
  EXPECT_EQ(enum_ha_recover_xa_state::PREPARED_IN_TC,
            xa_state_list->find(xid_2));

  EXPECT_EQ(0, vector_trx_participant::deinit_plugin(&hton));
}

TEST(VectorTrxParticipantTest,
     RecoverPreparedLoadFailureQuarantinesAndContinues) {
  handlerton hton{};
  PreparedRowsTruthStore store;
  store.fail_load_prepared = true;
  store.allow_quarantine_prepared = true;
  TruthStoreGuard truth_store_guard(&store);
  XA_recover_txn txn_list[2]{};

  ASSERT_EQ(0, vector_trx_participant::init_plugin(&hton));
  EXPECT_EQ(0, hton.recover(&hton, txn_list, 2, nullptr));
  auto xa_state_tuple = vector_gunit::make_xa_state_list_for_testing();
  auto *xa_state_list = xa_state_tuple.get();
  ASSERT_NE(nullptr, xa_state_list);
  EXPECT_EQ(0, hton.recover_prepared_in_tc(&hton, *xa_state_list));
  EXPECT_TRUE(store.quarantine_prepared_called);
  EXPECT_EQ(0, vector_trx_participant::deinit_plugin(&hton));
}

TEST(VectorTrxParticipantTest, RecoverHandlesBootstrapAndQuarantineFailure) {
  handlerton hton{};
  PreparedRowsTruthStore store;
  TruthStoreGuard truth_store_guard(&store);
  XA_recover_txn txn_list[2]{};

  ASSERT_EQ(0, vector_trx_participant::init_plugin(&hton));

  store.rows = {make_prepared_row(321, "g3", "b3", true)};
  {
    OptInitializeGuard bootstrap(true);
    EXPECT_EQ(0, hton.recover(&hton, txn_list, 2, nullptr));
    auto xa_state_tuple = vector_gunit::make_xa_state_list_for_testing();
    auto *xa_state_list = xa_state_tuple.get();
    ASSERT_NE(nullptr, xa_state_list);
    EXPECT_EQ(0, hton.recover_prepared_in_tc(&hton, *xa_state_list));
  }

  store.rows.clear();
  EXPECT_EQ(0, hton.recover(&hton, nullptr, 2, nullptr));
  EXPECT_EQ(0, hton.recover(&hton, txn_list, 0, nullptr));

  store.rows.clear();
  store.fail_load_prepared = true;
  store.allow_quarantine_prepared = false;
  EXPECT_EQ(0, hton.recover(&hton, txn_list, 2, nullptr));
  EXPECT_FALSE(vector_trx_participant::load_prepared_rows_for_testing(
      &store.rows));
  auto xa_state_tuple = vector_gunit::make_xa_state_list_for_testing();
  auto *xa_state_list = xa_state_tuple.get();
  ASSERT_NE(nullptr, xa_state_list);
  EXPECT_EQ(1, hton.recover_prepared_in_tc(&hton, *xa_state_list));

  EXPECT_EQ(0, vector_trx_participant::deinit_plugin(&hton));
}

TEST(VectorTrxParticipantTest, XidCallbacksDispatchAfterServerStart) {
  handlerton hton{};
  PreparedRowsTruthStore store;
  TruthStoreGuard truth_store_guard(&store);
  XID xid = make_xid(777, "g", "b");

  ASSERT_EQ(0, vector_trx_participant::init_plugin(&hton));
  {
    my_testing::Server_initializer initializer;
    initializer.SetUp();
    EXPECT_EQ(0, hton.close_connection(&hton, initializer.thd()));
    EXPECT_EQ(0, hton.set_prepared_in_tc(&hton, initializer.thd()));
    initializer.TearDown();
  }
  {
    ServerStartedGuard server_started(true);
    EXPECT_NE(XA_OK, hton.commit_by_xid(&hton, &xid));
    EXPECT_NE(XA_OK, hton.rollback_by_xid(&hton, &xid));
    EXPECT_EQ(XA_OK, hton.set_prepared_in_tc_by_xid(&hton, &xid));
  }
  EXPECT_EQ(0, vector_trx_participant::deinit_plugin(&hton));
}

TEST(VectorTrxParticipantTest, XidCallbacksProbeArtifactForUncachedNonVectorXids) {
  handlerton hton{};
  PreparedRowsTruthStore store;
  TruthStoreGuard truth_store_guard(&store);
  XID xid = make_xid(888, "g", "b");

  ASSERT_EQ(0, vector_trx_participant::init_plugin(&hton));
  {
    ServerStartedGuard server_started(true);
    EXPECT_EQ(XAER_NOTA, hton.commit_by_xid(&hton, &xid));
    EXPECT_EQ(XAER_NOTA, hton.rollback_by_xid(&hton, &xid));
  }
  EXPECT_EQ(2, store.load_prepared_calls);
  EXPECT_EQ(0, vector_trx_participant::deinit_plugin(&hton));
}

TEST(VectorTrxParticipantTest, XidCallbacksReportPreparedArtifactLoadFailure) {
  handlerton hton{};
  PreparedRowsTruthStore store;
  store.fail_load_prepared = true;
  TruthStoreGuard truth_store_guard(&store);
  XID xid = make_xid(889, "g", "b");

  ASSERT_EQ(0, vector_trx_participant::init_plugin(&hton));
  {
    ServerStartedGuard server_started(true);
    EXPECT_EQ(XAER_RMERR, hton.commit_by_xid(&hton, &xid));
    EXPECT_EQ(XAER_RMERR, hton.rollback_by_xid(&hton, &xid));
  }
  EXPECT_EQ(2, store.load_prepared_calls);
  EXPECT_EQ(0, vector_trx_participant::deinit_plugin(&hton));
}

}  // namespace vector_trx_participant_unittest
