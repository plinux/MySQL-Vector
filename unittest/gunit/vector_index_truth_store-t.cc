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

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "mysqld_error.h"
#include "sql/dd/impl/tables/vector_index_truth_tables.h"
#include "sql/vector/vector_index_metadata_store.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_index_truth_store_internal.h"
#include "storage/innobase/include/dict0vectruth.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_index_truth_store_unittest {

namespace {

class EnvVarGuard {
 public:
  explicit EnvVarGuard(const char *name) : m_name(name) {
    const char *value = std::getenv(name);
    if (value != nullptr) {
      m_had_value = true;
      m_value = value;
    }
  }

  ~EnvVarGuard() {
    if (m_had_value) {
      setenv(m_name.c_str(), m_value.c_str(), 1);
    } else {
      unsetenv(m_name.c_str());
    }
  }

 private:
  std::string m_name;
  bool m_had_value{false};
  std::string m_value;
};

std::string fp32_payload(size_t dimension) {
  return std::string(dimension * sizeof(float), '\0');
}

}  // namespace

class dummy_truth_store : public vector_index_truth_store::truth_store {
 public:
  const char *backend_name() const override { return "dummy"; }

  bool is_transactional() const override { return true; }

  bool load_metadata(
      std::vector<vector_index_metadata_store::metadata_row> *rows) override {
    if (rows != nullptr) rows->clear();
    return true;
  }
  bool save_metadata(
      const std::vector<vector_index_metadata_store::metadata_row> &rows
      [[maybe_unused]]) override {
    return true;
  }
  bool load_committed(
      std::vector<vector_index_metadata_store::committed_row> *rows) override {
    if (rows != nullptr) rows->clear();
    return true;
  }
  bool save_committed(
      const std::vector<vector_index_metadata_store::committed_row> &rows
      [[maybe_unused]]) override {
    return true;
  }
  bool load_manifest(vector_index_metadata_store::manifest_row *row) override {
    if (row != nullptr) *row = vector_index_metadata_store::manifest_row();
    return true;
  }
  bool save_manifest(const vector_index_metadata_store::manifest_row &row
                     [[maybe_unused]]) override {
    return true;
  }
  bool load_change_log(
      std::vector<vector_index_metadata_store::change_log_row> *rows) override {
    if (rows != nullptr) rows->clear();
    return true;
  }
  bool save_change_log(
      const std::vector<vector_index_metadata_store::change_log_row> &rows
      [[maybe_unused]]) override {
    return true;
  }
  bool load_prepared(
      std::vector<vector_index_metadata_store::prepared_change_row> *rows)
      override {
    if (rows != nullptr) rows->clear();
    return true;
  }
  bool save_prepared(
      const std::vector<vector_index_metadata_store::prepared_change_row> &rows
      [[maybe_unused]]) override {
    return true;
  }
  bool load_segment_tasks(
      std::vector<vector_index_metadata_store::segment_task_row> *rows)
      override {
    if (rows != nullptr) rows->clear();
    return true;
  }
  bool save_segment_tasks(
      const std::vector<vector_index_metadata_store::segment_task_row> &rows
      [[maybe_unused]]) override {
    return true;
  }
  bool quarantine_segment_tasks() override { return true; }
};

class NullNameTruthStore final : public dummy_truth_store {
 public:
  const char *backend_name() const override { return nullptr; }
};

TEST(VectorIndexTruthStoreTest, TruthStoreDefaultsCoverNoopAndDeltaFallbacks) {
  dummy_truth_store store;
  EXPECT_FALSE(store.supports_delta_persist());
  EXPECT_FALSE(store.supports_attached_dml());
  EXPECT_FALSE(store.supports_publication_intents());
  EXPECT_FALSE(store.apply_attached_committed(nullptr, {}));
  EXPECT_FALSE(store.append_attached_change_log(nullptr, {}));
  EXPECT_FALSE(store.insert_attached_publication_intent(nullptr, {}));
  EXPECT_FALSE(store.load_publication_intents(nullptr));
  EXPECT_FALSE(store.delete_publication_intent("idx", 1));
  EXPECT_TRUE(store.begin_persist());
  EXPECT_TRUE(store.commit_persist());
  store.rollback_persist();
  EXPECT_FALSE(store.apply_committed_delta({}));
  EXPECT_FALSE(store.append_change_log_delta({}));
  std::string identity;
  EXPECT_FALSE(store.stage_quarantine("metadata", "load_failed", 1, &identity));
  EXPECT_FALSE(store.update_quarantine_state(
      identity, vector_index_truth_store::quarantine_state::kComplete));
}

class CommittedRowsTruthStore final : public dummy_truth_store {
 public:
  explicit CommittedRowsTruthStore(
      std::vector<vector_index_metadata_store::committed_row> rows)
      : m_rows(std::move(rows)) {}

  bool load_committed(
      std::vector<vector_index_metadata_store::committed_row> *rows) override {
    if (rows == nullptr) return false;
    *rows = m_rows;
    return true;
  }

 private:
  std::vector<vector_index_metadata_store::committed_row> m_rows;
};

class FailingCommittedRowsTruthStore final : public dummy_truth_store {
 public:
  bool load_committed(
      std::vector<vector_index_metadata_store::committed_row> *) override {
    return false;
  }
};

TEST(VectorIndexTruthStoreTest, DefaultStoreIsAvailable) {
  EXPECT_NE(nullptr, vector_index_truth_store::get());
  EXPECT_STREQ("mysql", vector_index_truth_store::active_backend_name());
  EXPECT_TRUE(vector_index_truth_store::active_backend_transactional());
}

TEST(VectorIndexTruthStoreTest, OverrideStoreForTesting) {
  dummy_truth_store dummy_store;
  vector_index_truth_store::set_for_testing(&dummy_store);
  EXPECT_EQ(&dummy_store, vector_index_truth_store::get());
  EXPECT_STREQ("dummy", vector_index_truth_store::active_backend_name());
  EXPECT_TRUE(vector_index_truth_store::active_backend_transactional());
  vector_index_truth_store::reset_for_testing();
  EXPECT_NE(&dummy_store, vector_index_truth_store::get());
  EXPECT_STREQ("mysql", vector_index_truth_store::active_backend_name());
  EXPECT_TRUE(vector_index_truth_store::active_backend_transactional());
}

TEST(VectorIndexTruthStoreTest, ActiveBackendNameFallsBackToUnknown) {
  NullNameTruthStore null_name_store;
  vector_index_truth_store::set_for_testing(&null_name_store);
  EXPECT_STREQ("unknown", vector_index_truth_store::active_backend_name());
  vector_index_truth_store::reset_for_testing();
}

TEST(VectorIndexTruthStoreTest, DefaultCommittedIteratorVisitsLoadedRows) {
  CommittedRowsTruthStore store({
      {"idx", 2, {2.0F, 3.0F}},
      {"idx", 1, {1.0F, 0.0F}},
  });

  std::vector<uint64_t> visited;
  ASSERT_TRUE(store.for_each_committed(
      [&](const vector_index_metadata_store::committed_row &row) {
        visited.push_back(row.doc_id);
        return true;
      }));

  EXPECT_EQ((std::vector<uint64_t>{2, 1}), visited);
}

TEST(VectorIndexTruthStoreTest, DefaultCommittedIteratorStopsOnVisitorFailure) {
  CommittedRowsTruthStore store({
      {"idx", 2, {2.0F, 3.0F}},
      {"idx", 1, {1.0F, 0.0F}},
  });

  std::vector<uint64_t> visited;
  EXPECT_FALSE(store.for_each_committed(
      [&](const vector_index_metadata_store::committed_row &row) {
        visited.push_back(row.doc_id);
        return false;
      }));

  EXPECT_EQ((std::vector<uint64_t>{2}), visited);
}

TEST(VectorIndexTruthStoreTest, DefaultFindCommittedStopsWhenRowIsFound) {
  CommittedRowsTruthStore store({
      {"idx", 2, {2.0F, 3.0F}},
      {"idx", 1, {1.0F, 0.0F}},
      {"other", 1, {9.0F, 9.0F}},
  });

  vector_index_metadata_store::committed_row row;
  bool found = false;
  ASSERT_TRUE(store.find_committed("idx", 1, &row, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(1U, row.doc_id);
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), row.vector);

  ASSERT_TRUE(store.find_committed("idx", 9, &row, &found));
  EXPECT_FALSE(found);
  EXPECT_TRUE(row.index_name.empty());

  EXPECT_FALSE(store.find_committed("", 1, &row, &found));
  EXPECT_FALSE(store.find_committed("idx", 1, nullptr, &found));
  EXPECT_FALSE(store.find_committed("idx", 1, &row, nullptr));
}

TEST(VectorIndexTruthStoreTest, DefaultCommittedIteratorRejectsEmptyVisitor) {
  CommittedRowsTruthStore store({{"idx", 1, {1.0F, 0.0F}}});

  EXPECT_FALSE(store.for_each_committed(nullptr));
}

TEST(VectorIndexTruthStoreTest, MysqlBackendCanBeSelectedFromEnvironment) {
  EnvVarGuard guard("MYSQL_VECTOR_TRUTH_STORE");
  setenv("MYSQL_VECTOR_TRUTH_STORE", "mysql", 1);
  vector_index_truth_store::reset_for_testing();
  EXPECT_STREQ("mysql", vector_index_truth_store::active_backend_name());
  EXPECT_TRUE(vector_index_truth_store::active_backend_transactional());
}

TEST(VectorIndexTruthStoreTest, UnknownBackendUsesMysqlStore) {
  EnvVarGuard guard("MYSQL_VECTOR_TRUTH_STORE");
  setenv("MYSQL_VECTOR_TRUTH_STORE", "unknown-backend", 1);
  vector_index_truth_store::reset_for_testing();
  EXPECT_STREQ("mysql", vector_index_truth_store::active_backend_name());
  EXPECT_TRUE(vector_index_truth_store::active_backend_transactional());
}

TEST(VectorIndexTruthStoreTest, FileBackendBootstrapAndShutdownAreNoops) {
  EnvVarGuard guard("MYSQL_VECTOR_TRUTH_STORE");
  setenv("MYSQL_VECTOR_TRUTH_STORE", "file", 1);
  vector_index_truth_store::reset_for_testing();
  EXPECT_TRUE(
      vector_index_truth_store::bootstrap_initialize_selected_backend(nullptr));
  vector_index_truth_store::shutdown_selected_backend();
}

TEST(VectorIndexTruthStoreTest, OverrideStoreShortCircuitsInitHooks) {
  dummy_truth_store dummy_store;
  vector_index_truth_store::set_for_testing(&dummy_store);
  EXPECT_TRUE(
      vector_index_truth_store::bootstrap_initialize_selected_backend(nullptr));
  vector_index_truth_store::shutdown_selected_backend();
  vector_index_truth_store::reset_for_testing();
}

TEST(VectorIndexTruthStoreTest, MysqlBootstrapRejectsNullThread) {
  EnvVarGuard guard("MYSQL_VECTOR_TRUTH_STORE");
  setenv("MYSQL_VECTOR_TRUTH_STORE", "mysql", 1);
  vector_index_truth_store::reset_for_testing();
  EXPECT_FALSE(
      vector_index_truth_store::bootstrap_initialize_selected_backend(nullptr));
  vector_index_truth_store::shutdown_selected_backend();
}

TEST(VectorIndexTruthStoreTest, IsTruthStoreTableRecognizesKnownTables) {
  EXPECT_TRUE(vector_index_truth_store::is_truth_store_table(
      "mysql", "vector_index_truth_metadata"));
  EXPECT_TRUE(vector_index_truth_store::is_truth_store_table(
      "mysql", "vector_index_truth_committed"));
  EXPECT_TRUE(vector_index_truth_store::is_truth_store_table(
      "mysql", "vector_index_truth_manifest"));
  EXPECT_TRUE(vector_index_truth_store::is_truth_store_table(
      "mysql", "vector_index_truth_changelog"));
  EXPECT_TRUE(vector_index_truth_store::is_truth_store_table(
      "mysql", "vector_index_truth_prepared"));
  EXPECT_TRUE(vector_index_truth_store::is_truth_store_table(
      "mysql", "vector_index_truth_segment_tasks"));
  EXPECT_TRUE(vector_index_truth_store::is_truth_store_table(
      "mysql", "vector_index_truth_store_quarantine"));
  EXPECT_FALSE(vector_index_truth_store::is_truth_store_table(
      "mysql", "vector_index_truth_store"));
  EXPECT_FALSE(vector_index_truth_store::is_truth_store_table(
      "test", "vector_index_truth_metadata"));
  EXPECT_FALSE(vector_index_truth_store::is_truth_store_table(
      "mysql", "vector_index_truth_other"));
}

TEST(VectorIndexTruthStoreTest, IsTruthStoreTableRejectsNullNames) {
  EXPECT_FALSE(vector_index_truth_store::is_truth_store_table(
      nullptr, "vector_index_truth_metadata"));
  EXPECT_FALSE(
      vector_index_truth_store::is_truth_store_table("mysql", nullptr));
}

TEST(VectorIndexTruthStoreTest, RowTruthTablesUseRelationalDefinitions) {
  const auto committed_ddl =
      dd::tables::Vector_index_truth_committed::instance()
          .target_table_definition()
          ->get_ddl();
  EXPECT_NE(std::string::npos,
            committed_ddl.find("index_name VARBINARY(255) NOT NULL"));
  EXPECT_NE(std::string::npos,
            committed_ddl.find("doc_id BIGINT UNSIGNED NOT NULL"));
  EXPECT_NE(std::string::npos,
            committed_ddl.find("dimension INT UNSIGNED NOT NULL"));
  EXPECT_NE(std::string::npos,
            committed_ddl.find("vector_payload LONGBLOB NOT NULL"));
  EXPECT_NE(std::string::npos,
            committed_ddl.find("PRIMARY KEY (index_name, doc_id)"));
  EXPECT_EQ(std::string::npos, committed_ddl.find("singleton_id"));

  const auto changelog_ddl =
      dd::tables::Vector_index_truth_changelog::instance()
          .target_table_definition()
          ->get_ddl();
  EXPECT_NE(std::string::npos,
            changelog_ddl.find("sequence BIGINT UNSIGNED NOT NULL"));
  EXPECT_NE(std::string::npos,
            changelog_ddl.find("txn_id BIGINT UNSIGNED NOT NULL"));
  EXPECT_NE(std::string::npos,
            changelog_ddl.find("op TINYINT UNSIGNED NOT NULL"));
  EXPECT_NE(std::string::npos, changelog_ddl.find("PRIMARY KEY (sequence)"));
  EXPECT_EQ(std::string::npos, changelog_ddl.find("singleton_id"));

  const auto prepared_ddl = dd::tables::Vector_index_truth_prepared::instance()
                                .target_table_definition()
                                ->get_ddl();
  EXPECT_NE(std::string::npos,
            prepared_ddl.find("txn_id BIGINT UNSIGNED NOT NULL"));
  EXPECT_NE(std::string::npos,
            prepared_ddl.find("row_no BIGINT UNSIGNED NOT NULL"));
  EXPECT_NE(std::string::npos,
            prepared_ddl.find("xid_data VARBINARY(128) NOT NULL"));
  EXPECT_NE(std::string::npos,
            prepared_ddl.find("prepared_in_tc TINYINT UNSIGNED NOT NULL"));
  EXPECT_NE(std::string::npos,
            prepared_ddl.find("PRIMARY KEY (txn_id, row_no)"));
  EXPECT_EQ(std::string::npos, prepared_ddl.find("singleton_id"));
}

TEST(VectorIndexTruthStoreTest, SingletonTruthTablesCarryLayoutVersion) {
  const auto metadata_ddl = dd::tables::Vector_index_truth_metadata::instance()
                                .target_table_definition()
                                ->get_ddl();
  EXPECT_NE(std::string::npos,
            metadata_ddl.find("layout_version INT UNSIGNED NOT NULL"));
  EXPECT_NE(std::string::npos, metadata_ddl.find("PRIMARY KEY (singleton_id)"));

  const auto manifest_ddl = dd::tables::Vector_index_truth_manifest::instance()
                                .target_table_definition()
                                ->get_ddl();
  EXPECT_NE(std::string::npos,
            manifest_ddl.find("layout_version INT UNSIGNED NOT NULL"));
  EXPECT_NE(std::string::npos, manifest_ddl.find("PRIMARY KEY (singleton_id)"));

  const auto quarantine_ddl =
      dd::tables::Vector_index_truth_store_quarantine::instance()
          .target_table_definition()
          ->get_ddl();
  EXPECT_NE(std::string::npos,
            quarantine_ddl.find("layout_version INT UNSIGNED NOT NULL"));
  EXPECT_NE(std::string::npos,
            quarantine_ddl.find("PRIMARY KEY (singleton_id)"));

  const auto segment_tasks_ddl =
      dd::tables::Vector_index_truth_segment_tasks::instance()
          .target_table_definition()
          ->get_ddl();
  EXPECT_NE(std::string::npos,
            segment_tasks_ddl.find("layout_version INT UNSIGNED NOT NULL"));
  EXPECT_NE(std::string::npos,
            segment_tasks_ddl.find("PRIMARY KEY (singleton_id)"));
}

TEST(VectorIndexTruthStoreTest, SqlStringLiteralEscapesQuotesAndBackslashes) {
  EXPECT_EQ("''",
            vector_index_truth_store::sql_string_literal_for_testing(nullptr));
  EXPECT_EQ("'plain'",
            vector_index_truth_store::sql_string_literal_for_testing("plain"));
  EXPECT_EQ("'a\\'b\\\\c'",
            vector_index_truth_store::sql_string_literal_for_testing("a'b\\c"));
}

TEST(VectorIndexTruthStoreTest, DecodeHexBytesRejectsInvalidInputs) {
  std::string decoded;
  EXPECT_FALSE(
      vector_index_truth_store::decode_hex_bytes_for_testing("0", &decoded));
  EXPECT_FALSE(
      vector_index_truth_store::decode_hex_bytes_for_testing("0g", &decoded));
  EXPECT_FALSE(
      vector_index_truth_store::decode_hex_bytes_for_testing("g0", &decoded));
  EXPECT_FALSE(
      vector_index_truth_store::decode_hex_bytes_for_testing("00", nullptr));
}

TEST(VectorIndexTruthStoreTest, DecodeHexBytesHandlesUppercaseAndEmptyInput) {
  std::string decoded = "sentinel";
  EXPECT_TRUE(
      vector_index_truth_store::decode_hex_bytes_for_testing("", &decoded));
  EXPECT_TRUE(decoded.empty());

  EXPECT_TRUE(vector_index_truth_store::decode_hex_bytes_for_testing("414243",
                                                                     &decoded));
  EXPECT_EQ("ABC", decoded);
}

TEST(VectorIndexTruthStoreTest, RelationalRowCodecsCoverValidationBranches) {
  namespace detail = vector_index_truth_store::detail;

  uint32_t dimension = 0;
  std::string payload;
  vector_index::vector_data vector;
  EXPECT_FALSE(detail::vector_to_truth_payload_impl({1.0F}, nullptr, &payload));
  EXPECT_FALSE(
      detail::vector_to_truth_payload_impl({1.0F}, &dimension, nullptr));
  EXPECT_TRUE(detail::vector_to_truth_payload_impl({}, &dimension, &payload));
  EXPECT_EQ(0U, dimension);
  EXPECT_TRUE(payload.empty());
  EXPECT_FALSE(
      detail::truth_payload_to_vector_impl(1, fp32_payload(2), &vector));
  EXPECT_FALSE(
      detail::truth_payload_to_vector_impl(1, fp32_payload(1), nullptr));
  EXPECT_TRUE(detail::truth_payload_to_vector_impl(0, "", &vector));
  EXPECT_TRUE(vector.empty());

  uint8_t encoded_op = 0;
  vector_index_metadata_store::change_op decoded_op;
  EXPECT_FALSE(detail::encode_truth_op_impl(
      vector_index_metadata_store::change_op::kUpsert, nullptr));
  EXPECT_FALSE(detail::encode_truth_op_impl(
      static_cast<vector_index_metadata_store::change_op>(99), &encoded_op));
  EXPECT_TRUE(detail::encode_truth_op_impl(
      vector_index_metadata_store::change_op::kErase, &encoded_op));
  EXPECT_EQ(2U, encoded_op);
  EXPECT_FALSE(detail::decode_truth_op_impl(1, nullptr));
  EXPECT_TRUE(detail::decode_truth_op_impl(2, &decoded_op));
  EXPECT_EQ(vector_index_metadata_store::change_op::kErase, decoded_op);

  std::vector<innodb_vector_truth_store::committed_row> stored_committed;
  EXPECT_FALSE(detail::to_innodb_committed_rows_impl({}, nullptr));
  EXPECT_TRUE(detail::to_innodb_committed_rows_impl({{"idx", 1, {1.0F, 2.0F}}},
                                                    &stored_committed));
  ASSERT_EQ(1U, stored_committed.size());
  EXPECT_EQ(2U, stored_committed[0].dimension);
  const auto valid_stored_committed = stored_committed;
  EXPECT_FALSE(detail::to_innodb_committed_rows_impl({{"", 1, {1.0F}}},
                                                     &stored_committed));

  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  EXPECT_FALSE(detail::from_innodb_committed_rows_impl({}, nullptr));
  EXPECT_TRUE(detail::from_innodb_committed_rows_impl(valid_stored_committed,
                                                      &committed_rows));
  ASSERT_EQ(1U, committed_rows.size());
  EXPECT_EQ(1U, committed_rows[0].doc_id);
  auto multiple_stored_committed = valid_stored_committed;
  multiple_stored_committed.push_back(
      {"idx", 2, 4096, fp32_payload(4096)});
  EXPECT_TRUE(detail::from_innodb_committed_rows_impl(
      multiple_stored_committed, &committed_rows));
  ASSERT_EQ(2U, committed_rows.size());
  EXPECT_EQ(2U, committed_rows[1].doc_id);
  EXPECT_EQ(4096U, committed_rows[1].vector.size());
  EXPECT_FALSE(detail::from_innodb_committed_rows_impl(
      {{"", 1, 1, fp32_payload(1)}}, &committed_rows));
  EXPECT_FALSE(detail::from_innodb_committed_rows_impl(
      {{"idx", 1, 2, fp32_payload(1)}}, &committed_rows));

  std::vector<innodb_vector_truth_store::change_log_row> stored_change_log;
  EXPECT_FALSE(detail::to_innodb_change_log_rows_impl({}, nullptr));
  EXPECT_TRUE(detail::to_innodb_change_log_rows_impl(
      {{1,
        9,
        vector_index_metadata_store::change_op::kUpsert,
        "idx",
        1,
        {1.0F},
        42,
        1,
        1},
       {2, 9, vector_index_metadata_store::change_op::kErase, "idx", 1, {},
        42, 1, 1}},
      &stored_change_log));
  ASSERT_EQ(2U, stored_change_log.size());
  EXPECT_EQ(1U, stored_change_log[0].op);
  EXPECT_EQ(2U, stored_change_log[1].op);
  const auto valid_stored_change_log = stored_change_log;
  EXPECT_FALSE(detail::to_innodb_change_log_rows_impl(
      {{0,
        9,
        vector_index_metadata_store::change_op::kUpsert,
        "idx",
        1,
        {1.0F},
        42,
        1,
        1}},
      &stored_change_log));
  EXPECT_FALSE(detail::to_innodb_change_log_rows_impl(
      {{1, 9, vector_index_metadata_store::change_op::kUpsert, "", 1,
        {1.0F}, 42, 1, 1}},
      &stored_change_log));
  EXPECT_FALSE(detail::to_innodb_change_log_rows_impl(
      {{1,
        9,
        static_cast<vector_index_metadata_store::change_op>(99),
        "idx",
        1,
        {1.0F},
        42,
        1,
        1}},
      &stored_change_log));
  EXPECT_FALSE(detail::to_innodb_change_log_rows_impl(
      {{1,
        9,
        vector_index_metadata_store::change_op::kErase,
        "idx",
        1,
        {1.0F},
        42,
        1,
        1}},
      &stored_change_log));
  EXPECT_FALSE(detail::to_innodb_change_log_rows_impl(
      {{1, 9, vector_index_metadata_store::change_op::kUpsert, "idx", 1,
        {1.0F}, 0, 1, 1}},
      &stored_change_log));

  const std::vector<vector_index_metadata_store::change_log_row>
      unbound_truth_delta{{1,
                           9,
                           vector_index_metadata_store::change_op::kUpsert,
                           "idx",
                           1,
                           {1.0F},
                           42,
                           0,
                           0}};
  EXPECT_TRUE(detail::to_innodb_truth_delta_rows_impl(unbound_truth_delta,
                                                       &stored_change_log));
  EXPECT_FALSE(detail::to_innodb_change_log_rows_impl(unbound_truth_delta,
                                                       &stored_change_log));

  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  EXPECT_FALSE(detail::from_innodb_change_log_rows_impl({}, nullptr));
  EXPECT_TRUE(detail::from_innodb_change_log_rows_impl(valid_stored_change_log,
                                                       &change_log_rows));
  ASSERT_EQ(2U, change_log_rows.size());
  EXPECT_TRUE(change_log_rows[1].vector.empty());
  EXPECT_FALSE(detail::from_innodb_change_log_rows_impl(
      {{0, 9, 1, "idx", 1, 1, fp32_payload(1), 42, 1, 1}},
      &change_log_rows));
  EXPECT_FALSE(detail::from_innodb_change_log_rows_impl(
      {{1, 9, 1, "", 1, 1, fp32_payload(1), 42, 1, 1}},
      &change_log_rows));
  EXPECT_FALSE(detail::from_innodb_change_log_rows_impl(
      {{1, 9, 99, "idx", 1, 1, fp32_payload(1), 42, 1, 1}},
      &change_log_rows));
  EXPECT_FALSE(detail::from_innodb_change_log_rows_impl(
      {{1, 9, 1, "idx", 1, 2, fp32_payload(1), 42, 1, 1}},
      &change_log_rows));
  EXPECT_FALSE(detail::from_innodb_change_log_rows_impl(
      {{1, 9, 2, "idx", 1, 1, fp32_payload(1), 42, 1, 1}},
      &change_log_rows));
  EXPECT_FALSE(detail::from_innodb_change_log_rows_impl(
      {{1, 9, 1, "idx", 1, 1, fp32_payload(1), 0, 1, 1}},
      &change_log_rows));
  EXPECT_FALSE(detail::from_innodb_change_log_rows_impl(
      {{1, 9, 1, "idx", 1, 1, fp32_payload(1), 42, 0, 1}},
      &change_log_rows));
  EXPECT_FALSE(detail::from_innodb_change_log_rows_impl(
      {{1, 9, 1, "idx", 1, 1, fp32_payload(1), 42, 1, 0}},
      &change_log_rows));

  std::vector<innodb_vector_truth_store::prepared_change_row> stored_prepared;
  EXPECT_FALSE(detail::to_innodb_prepared_rows_impl({}, nullptr));
  EXPECT_TRUE(detail::to_innodb_prepared_rows_impl(
      {{7,
        1,
        1,
        "gb",
        true,
        9,
        vector_index_metadata_store::change_op::kUpsert,
        "idx",
        1,
        {1.0F}},
       {7,
        1,
        1,
        "gb",
        false,
        9,
        vector_index_metadata_store::change_op::kErase,
        "idx",
        1,
        {}}},
      &stored_prepared));
  ASSERT_EQ(2U, stored_prepared.size());
  EXPECT_EQ(1U, stored_prepared[0].row_no);
  EXPECT_EQ(2U, stored_prepared[1].row_no);
  const auto valid_stored_prepared = stored_prepared;
  EXPECT_FALSE(detail::to_innodb_prepared_rows_impl(
      {{-1,
        1,
        1,
        "gb",
        false,
        9,
        vector_index_metadata_store::change_op::kUpsert,
        "idx",
        1,
        {1.0F}}},
      &stored_prepared));
  EXPECT_FALSE(detail::to_innodb_prepared_rows_impl(
      {{7,
        -1,
        1,
        "gb",
        false,
        9,
        vector_index_metadata_store::change_op::kUpsert,
        "idx",
        1,
        {1.0F}}},
      &stored_prepared));
  EXPECT_FALSE(detail::to_innodb_prepared_rows_impl(
      {{7,
        1,
        -1,
        "gb",
        false,
        9,
        vector_index_metadata_store::change_op::kUpsert,
        "idx",
        1,
        {1.0F}}},
      &stored_prepared));
  EXPECT_FALSE(detail::to_innodb_prepared_rows_impl(
      {{7,
        65,
        0,
        std::string(65, 'g'),
        false,
        9,
        vector_index_metadata_store::change_op::kUpsert,
        "idx",
        1,
        {1.0F}}},
      &stored_prepared));
  EXPECT_FALSE(detail::to_innodb_prepared_rows_impl(
      {{7,
        0,
        65,
        std::string(65, 'b'),
        false,
        9,
        vector_index_metadata_store::change_op::kUpsert,
        "idx",
        1,
        {1.0F}}},
      &stored_prepared));
  EXPECT_FALSE(detail::to_innodb_prepared_rows_impl(
      {{7,
        1,
        0,
        "g",
        false,
        0,
        vector_index_metadata_store::change_op::kUpsert,
        "idx",
        1,
        {1.0F}}},
      &stored_prepared));
  EXPECT_FALSE(detail::to_innodb_prepared_rows_impl(
      {{7,
        1,
        1,
        "g",
        false,
        9,
        vector_index_metadata_store::change_op::kUpsert,
        "idx",
        1,
        {1.0F}}},
      &stored_prepared));
  EXPECT_FALSE(detail::to_innodb_prepared_rows_impl(
      {{7,
        1,
        1,
        "gb",
        false,
        9,
        vector_index_metadata_store::change_op::kUpsert,
        "",
        1,
        {1.0F}}},
      &stored_prepared));
  EXPECT_FALSE(detail::to_innodb_prepared_rows_impl(
      {{7,
        1,
        1,
        "gb",
        false,
        9,
        static_cast<vector_index_metadata_store::change_op>(99),
        "idx",
        1,
        {1.0F}}},
      &stored_prepared));
  EXPECT_FALSE(detail::to_innodb_prepared_rows_impl(
      {{7,
        1,
        1,
        "gb",
        false,
        9,
        vector_index_metadata_store::change_op::kErase,
        "idx",
        1,
        {1.0F}}},
      &stored_prepared));

  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  EXPECT_FALSE(detail::from_innodb_prepared_rows_impl({}, nullptr));
  EXPECT_TRUE(detail::from_innodb_prepared_rows_impl(valid_stored_prepared,
                                                     &prepared_rows));
  ASSERT_EQ(2U, prepared_rows.size());
  EXPECT_TRUE(prepared_rows[0].prepared_in_tc);
  EXPECT_TRUE(prepared_rows[1].vector.empty());

  const uint64_t overflow =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1U;
  EXPECT_FALSE(detail::from_innodb_prepared_rows_impl(
      {{9, 1, overflow, 1, 1, "gb", 0, 1, "idx", 1, 1, fp32_payload(1)}},
      &prepared_rows));
  EXPECT_FALSE(detail::from_innodb_prepared_rows_impl(
      {{9, 1, 7, overflow, 1, "gb", 0, 1, "idx", 1, 1, fp32_payload(1)}},
      &prepared_rows));
  EXPECT_FALSE(detail::from_innodb_prepared_rows_impl(
      {{9, 1, 7, 1, overflow, "gb", 0, 1, "idx", 1, 1, fp32_payload(1)}},
      &prepared_rows));
  EXPECT_FALSE(detail::from_innodb_prepared_rows_impl(
      {{9, 1, 7, 65, 0, std::string(65, 'g'), 0, 1, "idx", 1, 1,
        fp32_payload(1)}},
      &prepared_rows));
  EXPECT_FALSE(detail::from_innodb_prepared_rows_impl(
      {{9, 1, 7, 0, 65, std::string(65, 'b'), 0, 1, "idx", 1, 1,
        fp32_payload(1)}},
      &prepared_rows));
  EXPECT_FALSE(detail::from_innodb_prepared_rows_impl(
      {{0, 1, 7, 1, 0, "g", 0, 1, "idx", 1, 1, fp32_payload(1)}},
      &prepared_rows));
  EXPECT_FALSE(detail::from_innodb_prepared_rows_impl(
      {{9, 1, 7, 1, 1, "g", 0, 1, "idx", 1, 1, fp32_payload(1)}},
      &prepared_rows));
  EXPECT_FALSE(detail::from_innodb_prepared_rows_impl(
      {{9, 1, 7, 1, 1, "gb", 0, 1, "", 1, 1, fp32_payload(1)}},
      &prepared_rows));
  EXPECT_FALSE(detail::from_innodb_prepared_rows_impl(
      {{9, 1, 7, 1, 1, "gb", 0, 99, "idx", 1, 1, fp32_payload(1)}},
      &prepared_rows));
  EXPECT_FALSE(detail::from_innodb_prepared_rows_impl(
      {{9, 1, 7, 1, 1, "gb", 0, 1, "idx", 1, 2, fp32_payload(1)}},
      &prepared_rows));
  EXPECT_FALSE(detail::from_innodb_prepared_rows_impl(
      {{9, 1, 7, 1, 1, "gb", 0, 2, "idx", 1, 1, fp32_payload(1)}},
      &prepared_rows));
}

TEST(VectorIndexTruthStoreTest, DebugPayloadHelpersCoverArtifactBranches) {
  namespace detail = vector_index_truth_store::detail;

  EXPECT_FALSE(detail::is_row_artifact_name_impl(nullptr));
  EXPECT_TRUE(detail::is_row_artifact_name_impl("committed"));
  EXPECT_TRUE(detail::is_row_artifact_name_impl("changelog"));
  EXPECT_TRUE(detail::is_row_artifact_name_impl("prepared"));
  EXPECT_FALSE(detail::is_row_artifact_name_impl("metadata"));

  EXPECT_FALSE(detail::is_valid_artifact_name_impl(nullptr));
  EXPECT_TRUE(detail::is_valid_artifact_name_impl("metadata"));
  EXPECT_TRUE(detail::is_valid_artifact_name_impl("committed"));
  EXPECT_TRUE(detail::is_valid_artifact_name_impl("manifest"));
  EXPECT_TRUE(detail::is_valid_artifact_name_impl("changelog"));
  EXPECT_TRUE(detail::is_valid_artifact_name_impl("prepared"));
  EXPECT_FALSE(detail::is_valid_artifact_name_impl("quarantine_store"));

  std::string payload;
  auto committed = detail::make_debug_committed_row_impl("committed-payload");
  EXPECT_EQ(0U, committed.doc_id);
  EXPECT_TRUE(detail::extract_debug_payload_impl({committed}, &payload));
  EXPECT_EQ("committed-payload", payload);
  EXPECT_FALSE(detail::extract_debug_payload_impl(
      std::vector<innodb_vector_truth_store::committed_row>{}, &payload));
  EXPECT_FALSE(detail::extract_debug_payload_impl({committed}, nullptr));
  committed.index_name = "idx";
  EXPECT_FALSE(detail::extract_debug_payload_impl({committed}, &payload));
  committed = detail::make_debug_committed_row_impl("committed-payload");
  committed.doc_id = 1;
  EXPECT_FALSE(detail::extract_debug_payload_impl({committed}, &payload));
  committed = detail::make_debug_committed_row_impl("committed-payload");
  committed.dimension = 1;
  EXPECT_FALSE(detail::extract_debug_payload_impl({committed}, &payload));

  auto change_log = detail::make_debug_change_log_row_impl("change-payload");
  EXPECT_EQ(1U, change_log.sequence);
  EXPECT_TRUE(detail::extract_debug_payload_impl({change_log}, &payload));
  EXPECT_EQ("change-payload", payload);
  EXPECT_FALSE(detail::extract_debug_payload_impl(
      std::vector<innodb_vector_truth_store::change_log_row>{}, &payload));
  EXPECT_FALSE(detail::extract_debug_payload_impl({change_log}, nullptr));
  change_log.sequence = 2;
  EXPECT_FALSE(detail::extract_debug_payload_impl({change_log}, &payload));
  change_log = detail::make_debug_change_log_row_impl("change-payload");
  change_log.txn_id = 1;
  EXPECT_FALSE(detail::extract_debug_payload_impl({change_log}, &payload));
  change_log = detail::make_debug_change_log_row_impl("change-payload");
  change_log.op = 2;
  EXPECT_FALSE(detail::extract_debug_payload_impl({change_log}, &payload));
  change_log = detail::make_debug_change_log_row_impl("change-payload");
  change_log.index_name = "idx";
  EXPECT_FALSE(detail::extract_debug_payload_impl({change_log}, &payload));
  change_log = detail::make_debug_change_log_row_impl("change-payload");
  change_log.doc_id = 1;
  EXPECT_FALSE(detail::extract_debug_payload_impl({change_log}, &payload));
  change_log = detail::make_debug_change_log_row_impl("change-payload");
  change_log.dimension = 1;
  EXPECT_FALSE(detail::extract_debug_payload_impl({change_log}, &payload));

  auto prepared = detail::make_debug_prepared_row_impl("prepared-payload");
  EXPECT_EQ(1U, prepared.row_no);
  EXPECT_TRUE(detail::extract_debug_payload_impl({prepared}, &payload));
  EXPECT_EQ("prepared-payload", payload);
  EXPECT_FALSE(detail::extract_debug_payload_impl(
      std::vector<innodb_vector_truth_store::prepared_change_row>{}, &payload));
  EXPECT_FALSE(detail::extract_debug_payload_impl({prepared}, nullptr));
  prepared.txn_id = 1;
  EXPECT_FALSE(detail::extract_debug_payload_impl({prepared}, &payload));
  prepared = detail::make_debug_prepared_row_impl("prepared-payload");
  prepared.row_no = 2;
  EXPECT_FALSE(detail::extract_debug_payload_impl({prepared}, &payload));
  prepared = detail::make_debug_prepared_row_impl("prepared-payload");
  prepared.format_id = 1;
  EXPECT_FALSE(detail::extract_debug_payload_impl({prepared}, &payload));
  prepared = detail::make_debug_prepared_row_impl("prepared-payload");
  prepared.gtrid_length = 1;
  EXPECT_FALSE(detail::extract_debug_payload_impl({prepared}, &payload));
  prepared = detail::make_debug_prepared_row_impl("prepared-payload");
  prepared.bqual_length = 1;
  EXPECT_FALSE(detail::extract_debug_payload_impl({prepared}, &payload));
  prepared = detail::make_debug_prepared_row_impl("prepared-payload");
  prepared.xid_data = "x";
  EXPECT_FALSE(detail::extract_debug_payload_impl({prepared}, &payload));
  prepared = detail::make_debug_prepared_row_impl("prepared-payload");
  prepared.prepared_in_tc = 1;
  EXPECT_FALSE(detail::extract_debug_payload_impl({prepared}, &payload));
  prepared = detail::make_debug_prepared_row_impl("prepared-payload");
  prepared.op = 2;
  EXPECT_FALSE(detail::extract_debug_payload_impl({prepared}, &payload));
  prepared = detail::make_debug_prepared_row_impl("prepared-payload");
  prepared.index_name = "idx";
  EXPECT_FALSE(detail::extract_debug_payload_impl({prepared}, &payload));
  prepared = detail::make_debug_prepared_row_impl("prepared-payload");
  prepared.doc_id = 1;
  EXPECT_FALSE(detail::extract_debug_payload_impl({prepared}, &payload));
  prepared = detail::make_debug_prepared_row_impl("prepared-payload");
  prepared.dimension = 1;
  EXPECT_FALSE(detail::extract_debug_payload_impl({prepared}, &payload));
}

TEST(VectorIndexTruthStoreTest,
     DeserializeQuarantineEntriesRejectsMalformedPayloads) {
  std::vector<vector_index_truth_store::quarantine_record> entries;

  EXPECT_FALSE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          "mysql-vector-wrong-header\n", &entries));
  EXPECT_FALSE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          "mysql-vector-quarantine-v1\n6964\t6d65746164617461\n", &entries));
  EXPECT_FALSE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          "mysql-vector-quarantine-v1\nzz\t6d65746164617461\tcopied\t"
          "726561736f6e\t0\t1\t2\t\n",
          &entries));
  EXPECT_FALSE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          "mysql-vector-quarantine-v1\n6964\t6d65746164617461\tbad\t"
          "726561736f6e\t0\t1\t2\t\n",
          &entries));
  EXPECT_FALSE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          "mysql-vector-quarantine-v1\n6964\t6d65746164617461\tcopied\t"
          "zz\t0\t1\t2\t\n",
          &entries));
  EXPECT_FALSE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          "mysql-vector-quarantine-v1\n6964\t6d65746164617461\tcopied\t"
          "726561736f6e\tnan\t1\t2\t\n",
          &entries));
  EXPECT_FALSE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          "mysql-vector-quarantine-v1\n6964\t6d65746164617461\tcopied\t"
          "726561736f6e\t0\t1\t2\t00\n",
          &entries));
  EXPECT_FALSE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          "mysql-vector-quarantine-v1\n6964\t6d65746164617461\tcopied\t"
          "726561736f6e\t0\t1\t2\tzz\n",
          &entries));
  EXPECT_FALSE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          "mysql-vector-quarantine-v1\n", nullptr));
}

TEST(VectorIndexTruthStoreTest,
     DeserializeQuarantineEntriesRejectsEmptyAndHandlesValidPayloads) {
  namespace detail = vector_index_truth_store::detail;
  std::vector<vector_index_truth_store::quarantine_record> entries(1);
  EXPECT_FALSE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          "", &entries));
  EXPECT_TRUE(entries.empty());
  EXPECT_TRUE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          "mysql-vector-quarantine-v1\n", &entries));
  EXPECT_TRUE(entries.empty());

  std::string first_identity;
  std::string second_identity;
  ASSERT_TRUE(detail::append_quarantine_entry_impl(
      &entries, "metadata", "decode_failure", 7, "payload", &first_identity));
  ASSERT_TRUE(detail::append_quarantine_entry_impl(
      &entries, "prepared", "invalid_xid", 8, "ABC", &second_identity));
  ASSERT_TRUE(detail::update_quarantine_entry_state_impl(
      &entries, first_identity,
      vector_index_truth_store::quarantine_state::kComplete));
  std::string payload;
  ASSERT_TRUE(detail::serialize_quarantine_entries_impl(entries, &payload));
  entries.clear();
  EXPECT_TRUE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          payload, &entries));
  ASSERT_EQ(2U, entries.size());
  EXPECT_EQ(first_identity, entries[0].identity);
  EXPECT_EQ("metadata", entries[0].artifact_name);
  EXPECT_EQ(vector_index_truth_store::quarantine_state::kComplete,
            entries[0].state);
  EXPECT_EQ("decode_failure", entries[0].reason);
  EXPECT_EQ(7U, entries[0].generation);
  EXPECT_EQ("payload", entries[0].payload);
  EXPECT_EQ(second_identity, entries[1].identity);
  EXPECT_EQ("prepared", entries[1].artifact_name);
  EXPECT_EQ("ABC", entries[1].payload);
}

TEST(VectorIndexTruthStoreTest,
     QuarantineHelpersCoverSerializationAndReplacementBranches) {
  namespace detail = vector_index_truth_store::detail;

  std::vector<std::string> fields;
  EXPECT_FALSE(detail::split_tab_fields_impl("a\tb", nullptr));
  ASSERT_TRUE(detail::split_tab_fields_impl("", &fields));
  ASSERT_EQ(1U, fields.size());
  EXPECT_EQ("", fields[0]);
  ASSERT_TRUE(detail::split_tab_fields_impl("a\tb\t", &fields));
  ASSERT_EQ(3U, fields.size());
  EXPECT_EQ("a", fields[0]);
  EXPECT_EQ("b", fields[1]);
  EXPECT_EQ("", fields[2]);

  std::vector<vector_index_truth_store::quarantine_record> entries;
  std::string first_identity;
  ASSERT_TRUE(detail::append_quarantine_entry_impl(
      &entries, "metadata", "load_failed", 1, "one", &first_identity));
  std::string resumed_identity;
  ASSERT_TRUE(detail::append_quarantine_entry_impl(
      &entries, "metadata", "load_failed", 1, "one", &resumed_identity));
  EXPECT_EQ(first_identity, resumed_identity);
  ASSERT_TRUE(detail::update_quarantine_entry_state_impl(
      &entries, first_identity,
      vector_index_truth_store::quarantine_state::kComplete));
  std::string second_identity;
  ASSERT_TRUE(detail::append_quarantine_entry_impl(
      &entries, "metadata", "load_failed_again", 2, "one", &second_identity));
  ASSERT_EQ(2U, entries.size());
  EXPECT_NE(first_identity, second_identity);
  EXPECT_FALSE(detail::update_quarantine_entry_state_impl(
      &entries, "missing",
      vector_index_truth_store::quarantine_state::kComplete));
  EXPECT_FALSE(detail::update_quarantine_entry_state_impl(
      &entries, first_identity,
      vector_index_truth_store::quarantine_state::kSourceUpdateFailed));

  std::string payload;
  EXPECT_FALSE(detail::serialize_quarantine_entries_impl(entries, nullptr));
  ASSERT_TRUE(detail::serialize_quarantine_entries_impl(entries, &payload));
  std::vector<vector_index_truth_store::quarantine_record> decoded;
  ASSERT_TRUE(detail::deserialize_quarantine_entries_impl(payload, &decoded));
  ASSERT_EQ(entries.size(), decoded.size());
  EXPECT_EQ(entries[0].identity, decoded[0].identity);
  EXPECT_EQ(entries[1].identity, decoded[1].identity);
}

TEST(VectorIndexTruthStoreTest,
     QuarantineCodecRejectsEveryInvalidFieldAndStateTransition) {
  namespace detail = vector_index_truth_store::detail;
  using vector_index_truth_store::quarantine_record;
  using vector_index_truth_store::quarantine_state;

  EXPECT_EQ(nullptr, detail::quarantine_state_name_impl(
                         static_cast<quarantine_state>(99)));
  quarantine_state state = quarantine_state::kCopied;
  EXPECT_FALSE(detail::parse_quarantine_state_impl("copied", nullptr));
  EXPECT_TRUE(detail::parse_quarantine_state_impl("copied", &state));
  EXPECT_EQ(quarantine_state::kCopied, state);
  EXPECT_TRUE(
      detail::parse_quarantine_state_impl("source_update_failed", &state));
  EXPECT_EQ(quarantine_state::kSourceUpdateFailed, state);
  EXPECT_TRUE(detail::parse_quarantine_state_impl("complete", &state));
  EXPECT_EQ(quarantine_state::kComplete, state);
  EXPECT_FALSE(detail::parse_quarantine_state_impl("unknown", &state));

  uint64_t value = 0;
  EXPECT_FALSE(detail::parse_uint64_impl("1", nullptr));
  EXPECT_FALSE(detail::parse_uint64_impl("", &value));
  EXPECT_FALSE(detail::parse_uint64_impl("-1", &value));
  EXPECT_FALSE(detail::parse_uint64_impl("1x", &value));
  EXPECT_FALSE(detail::parse_uint64_impl("18446744073709551616", &value));
  EXPECT_TRUE(detail::parse_uint64_impl("18446744073709551615", &value));
  EXPECT_EQ(std::numeric_limits<uint64_t>::max(), value);

  std::vector<quarantine_record> entries;
  std::string identity;
  EXPECT_FALSE(detail::append_quarantine_entry_impl(
      nullptr, "metadata", "decode_failed", 1, "payload", &identity));
  EXPECT_FALSE(detail::append_quarantine_entry_impl(
      &entries, "metadata", "decode_failed", 1, "payload", nullptr));
  EXPECT_FALSE(detail::append_quarantine_entry_impl(
      &entries, "", "decode_failed", 1, "payload", &identity));
  EXPECT_FALSE(detail::append_quarantine_entry_impl(&entries, "metadata", "", 1,
                                                    "payload", &identity));
  ASSERT_TRUE(detail::append_quarantine_entry_impl(
      &entries, "metadata", "decode_failed", 1, "payload", &identity));

  EXPECT_FALSE(detail::update_quarantine_entry_state_impl(
      nullptr, identity, quarantine_state::kComplete));
  EXPECT_FALSE(detail::update_quarantine_entry_state_impl(
      &entries, "", quarantine_state::kComplete));

  const quarantine_record valid = entries.front();
  std::string serialized;
  const auto expect_serialize_failure = [&](const auto &mutate) {
    entries.assign(1, valid);
    mutate(&entries.front());
    EXPECT_FALSE(
        detail::serialize_quarantine_entries_impl(entries, &serialized));
  };
  expect_serialize_failure([](auto *entry) { entry->identity.clear(); });
  expect_serialize_failure([](auto *entry) { entry->artifact_name.clear(); });
  expect_serialize_failure([](auto *entry) { entry->reason.clear(); });
  expect_serialize_failure(
      [](auto *entry) { entry->state = static_cast<quarantine_state>(99); });
  expect_serialize_failure([](auto *entry) { ++entry->checksum; });

  const std::string checksum =
      std::to_string(detail::quarantine_payload_checksum_impl("payload"));
  const std::string header = "mysql-vector-quarantine-v1\n";
  const std::string valid_line =
      "6964\t6d65746164617461\tcopied\t726561736f6e\t" + checksum +
      "\t1\t2\t7061796c6f6164\n";
  const auto expect_decode_failure = [&](const std::string &line) {
    EXPECT_FALSE(
        detail::deserialize_quarantine_entries_impl(header + line, &entries));
  };
  expect_decode_failure("\t6d65746164617461\tcopied\t726561736f6e\t" +
                        checksum + "\t1\t2\t7061796c6f6164\n");
  expect_decode_failure("6964\t\tcopied\t726561736f6e\t" + checksum +
                        "\t1\t2\t7061796c6f6164\n");
  expect_decode_failure("6964\t6d65746164617461\tcopied\t\t" + checksum +
                        "\t1\t2\t7061796c6f6164\n");
  expect_decode_failure(
      "6964\t6d65746164617461\tcopied\t726561736f6e\t\t1\t2\t"
      "7061796c6f6164\n");
  expect_decode_failure("6964\t6d65746164617461\tcopied\t726561736f6e\t" +
                        checksum + "\t\t2\t7061796c6f6164\n");
  expect_decode_failure("6964\t6d65746164617461\tcopied\t726561736f6e\t" +
                        checksum + "\t1\t\t7061796c6f6164\n");
  expect_decode_failure(
      "6964\t6d65746164617461\tcopied\t726561736f6e\t" +
      std::to_string(detail::quarantine_payload_checksum_impl("other")) +
      "\t1\t2\t7061796c6f6164\n");

  ASSERT_TRUE(detail::deserialize_quarantine_entries_impl(
      header + "\n" + valid_line + "\n", &entries));
  ASSERT_EQ(1U, entries.size());
}

TEST(VectorIndexTruthStoreTest,
     DiagnosticsHelperRecordsNullSafeArtifactEvents) {
  namespace detail = vector_index_truth_store::detail;
  EnvVarGuard diag_guard("MYSQL_VECTOR_DIAG_FILE");
  const std::string path =
      std::string(testing::TempDir()) + "/vector_truth_store_diag_event.jsonl";
  std::remove(path.c_str());

  unsetenv("MYSQL_VECTOR_DIAG_FILE");
  detail::record_artifact_persist_event_impl("file", "metadata", 1, 2, 3, 4,
                                             true);
  EXPECT_FALSE(std::ifstream(path).good());

  setenv("MYSQL_VECTOR_DIAG_FILE", path.c_str(), 1);
  detail::record_artifact_persist_event_impl(nullptr, nullptr, 7, 11, 13, 17,
                                             false);

  std::ifstream input(path);
  ASSERT_TRUE(input.good());
  std::string line;
  std::getline(input, line);
  EXPECT_NE(std::string::npos,
            line.find("\"event\":\"vector_truth_store_artifact\""));
  EXPECT_NE(std::string::npos, line.find("\"backend\":\"\""));
  EXPECT_NE(std::string::npos, line.find("\"artifact\":\"\""));
  EXPECT_NE(std::string::npos, line.find("\"rows\":7"));
  EXPECT_NE(std::string::npos, line.find("\"ok\":0"));

  std::remove(path.c_str());
}

TEST(VectorIndexTruthStoreTest, TruthStoreBackendEnvParserCoversSwitches) {
  namespace detail = vector_index_truth_store::detail;
  EnvVarGuard guard("MYSQL_VECTOR_TRUTH_STORE");

  unsetenv("MYSQL_VECTOR_TRUTH_STORE");
  EXPECT_FALSE(detail::use_file_truth_store_backend_impl());
  setenv("MYSQL_VECTOR_TRUTH_STORE", "mysql", 1);
  EXPECT_FALSE(detail::use_file_truth_store_backend_impl());
  setenv("MYSQL_VECTOR_TRUTH_STORE", "file", 1);
  EXPECT_TRUE(detail::use_file_truth_store_backend_impl());
  setenv("MYSQL_VECTOR_TRUTH_STORE", "", 1);
  EXPECT_FALSE(detail::use_file_truth_store_backend_impl());
  setenv("MYSQL_VECTOR_TRUTH_STORE", "unknown-backend", 1);
  EXPECT_FALSE(detail::use_file_truth_store_backend_impl());
}

TEST(VectorIndexTruthStoreTest, DebugWrappersRejectInvalidArtifacts) {
  std::string payload;
  EXPECT_FALSE(vector_index_truth_store::debug_get_artifact_payload(
      "not_an_artifact", &payload));
  EXPECT_FALSE(vector_index_truth_store::debug_set_artifact_payload(
      "not_an_artifact", "abcd"));
  EXPECT_FALSE(
      vector_index_truth_store::debug_delete_artifact("not_an_artifact"));
}

TEST(VectorIndexTruthStoreTest, DebugWrappersRejectFileBackend) {
  EnvVarGuard guard("MYSQL_VECTOR_TRUTH_STORE");
  setenv("MYSQL_VECTOR_TRUTH_STORE", "file", 1);
  vector_index_truth_store::reset_for_testing();
  std::string payload;
  EXPECT_FALSE(vector_index_truth_store::debug_get_artifact_payload("metadata",
                                                                    &payload));
  EXPECT_FALSE(
      vector_index_truth_store::debug_set_artifact_payload("metadata", "abcd"));
  EXPECT_FALSE(vector_index_truth_store::debug_delete_artifact("metadata"));
}

TEST(VectorIndexTruthStoreTest, DebugGetRejectsNullPayload) {
  EXPECT_FALSE(vector_index_truth_store::debug_get_artifact_payload("metadata",
                                                                    nullptr));
}

TEST(VectorIndexTruthStoreTest,
     DebugWrappersHandleRowArtifactsWithoutHiddenTables) {
  std::string payload = "unchanged";

  EXPECT_FALSE(vector_index_truth_store::debug_get_artifact_payload("committed",
                                                                    &payload));
  EXPECT_FALSE(
      vector_index_truth_store::debug_set_artifact_payload("committed", "raw"));
  EXPECT_FALSE(vector_index_truth_store::debug_delete_artifact("committed"));

  EXPECT_FALSE(vector_index_truth_store::debug_get_artifact_payload("changelog",
                                                                    &payload));
  EXPECT_FALSE(
      vector_index_truth_store::debug_set_artifact_payload("changelog", "raw"));
  EXPECT_FALSE(vector_index_truth_store::debug_delete_artifact("changelog"));

  payload = "prepared";
  EXPECT_TRUE(vector_index_truth_store::debug_get_artifact_payload("prepared",
                                                                   &payload));
  EXPECT_TRUE(payload.empty());
  EXPECT_FALSE(
      vector_index_truth_store::debug_set_artifact_payload("prepared", "raw"));
  EXPECT_FALSE(vector_index_truth_store::debug_delete_artifact("prepared"));
}

TEST(VectorIndexTruthStoreTest,
     InnodbPreparedRowsMissingTableReturnsEmptyRows) {
  std::vector<innodb_vector_truth_store::prepared_change_row> rows{
      innodb_vector_truth_store::prepared_change_row{}};
  bool found = true;

  EXPECT_TRUE(innodb_vector_truth_store::load_prepared_rows(&rows, &found));
  EXPECT_FALSE(found);
  EXPECT_TRUE(rows.empty());
}

TEST(VectorIndexTruthStoreTest, ForEachCommittedCoversIteratorEdges) {
  vector_index_metadata_store::committed_row row;
  row.index_name = "idx_iter";
  row.doc_id = 10;
  row.vector = {1.0F, 2.0F};
  vector_index_metadata_store::committed_row other_row;
  other_row.index_name = "idx_other";
  other_row.doc_id = 20;
  other_row.vector = {2.0F, 3.0F};

  CommittedRowsTruthStore rows_store({row, other_row});
  EXPECT_FALSE(rows_store.for_each_committed(nullptr));

  std::vector<uint64_t> visited;
  EXPECT_TRUE(rows_store.for_each_committed(
      [&visited](const vector_index_metadata_store::committed_row &current) {
        visited.push_back(current.doc_id);
        return true;
      }));
  EXPECT_EQ(std::vector<uint64_t>({10, 20}), visited);

  visited.clear();
  EXPECT_FALSE(rows_store.for_each_committed(
      "",
      [](const vector_index_metadata_store::committed_row &) { return true; }));
  EXPECT_FALSE(rows_store.for_each_committed(
      "idx_iter", std::function<bool(
                      const vector_index_metadata_store::committed_row &)>()));
  EXPECT_TRUE(rows_store.for_each_committed(
      "idx_iter",
      [&visited](const vector_index_metadata_store::committed_row &current) {
        visited.push_back(current.doc_id);
        return true;
      }));
  EXPECT_EQ(std::vector<uint64_t>({10}), visited);

  EXPECT_FALSE(rows_store.for_each_committed(
      [](const vector_index_metadata_store::committed_row &) {
        return false;
      }));

  FailingCommittedRowsTruthStore failing_store;
  EXPECT_FALSE(failing_store.for_each_committed(
      [](const vector_index_metadata_store::committed_row &) { return true; }));
}

TEST(VectorIndexTruthStoreTest, InnodbFacadeRejectsInvalidArguments) {
  std::string payload;
  bool found = false;
  std::vector<innodb_vector_truth_store::committed_row> committed_rows;
  std::vector<innodb_vector_truth_store::change_log_row> change_log_rows;
  std::vector<innodb_vector_truth_store::prepared_change_row> prepared_rows;

  EXPECT_FALSE(
      innodb_vector_truth_store::load_artifact(nullptr, &payload, &found));
  EXPECT_FALSE(
      innodb_vector_truth_store::load_artifact("metadata", nullptr, &found));
  EXPECT_FALSE(
      innodb_vector_truth_store::load_artifact("metadata", &payload, nullptr));
  EXPECT_FALSE(
      innodb_vector_truth_store::load_artifact("metadata", &payload, &found));
  EXPECT_FALSE(innodb_vector_truth_store::load_artifact("not_an_artifact",
                                                        &payload, &found));
  EXPECT_FALSE(innodb_vector_truth_store::save_artifact(nullptr, payload));
  EXPECT_FALSE(innodb_vector_truth_store::save_artifact("metadata", payload));
  EXPECT_FALSE(
      innodb_vector_truth_store::save_artifact("not_an_artifact", payload));
  EXPECT_FALSE(innodb_vector_truth_store::delete_artifact(nullptr));
  EXPECT_FALSE(innodb_vector_truth_store::delete_artifact("metadata"));
  EXPECT_FALSE(innodb_vector_truth_store::delete_artifact("not_an_artifact"));

  EXPECT_FALSE(innodb_vector_truth_store::load_committed_rows(nullptr, &found));
  EXPECT_FALSE(
      innodb_vector_truth_store::load_committed_rows(&committed_rows, nullptr));
  EXPECT_FALSE(
      innodb_vector_truth_store::load_committed_rows(&committed_rows, &found));
  EXPECT_FALSE(innodb_vector_truth_store::save_committed_rows(committed_rows));
  EXPECT_FALSE(
      innodb_vector_truth_store::apply_committed_delta(change_log_rows));

  EXPECT_FALSE(
      innodb_vector_truth_store::load_change_log_rows(nullptr, &found));
  EXPECT_FALSE(innodb_vector_truth_store::load_change_log_rows(&change_log_rows,
                                                               nullptr));
  EXPECT_FALSE(innodb_vector_truth_store::load_change_log_rows(&change_log_rows,
                                                               &found));
  EXPECT_FALSE(
      innodb_vector_truth_store::save_change_log_rows(change_log_rows));
  EXPECT_FALSE(
      innodb_vector_truth_store::append_change_log_delta(change_log_rows));

  EXPECT_FALSE(innodb_vector_truth_store::load_prepared_rows(nullptr, &found));
  EXPECT_FALSE(
      innodb_vector_truth_store::load_prepared_rows(&prepared_rows, nullptr));
  EXPECT_FALSE(innodb_vector_truth_store::save_prepared_rows(prepared_rows));
}

TEST(VectorIndexTruthStoreTest, InternalFlagsDefaultToFalseWithoutSystemThd) {
  EXPECT_FALSE(vector_index_truth_store::internal_sql_active());
  EXPECT_FALSE(
      vector_index_truth_store::internal_truth_store_access_allowed(nullptr));
}

TEST(VectorIndexTruthStoreTest,
     MysqlTruthStoreQuarantineSegmentTasksRejectsMissingTables) {
  EnvVarGuard guard("MYSQL_VECTOR_TRUTH_STORE");
  unsetenv("MYSQL_VECTOR_TRUTH_STORE");
  vector_index_truth_store::reset_for_testing();

  vector_index_truth_store::truth_store *store =
      vector_index_truth_store::get();
  ASSERT_NE(nullptr, store);
  EXPECT_STREQ("mysql", store->backend_name());

  EXPECT_FALSE(store->quarantine_segment_tasks());
}

TEST(VectorIndexTruthStoreTest, FileStoreRoundTripMethods) {
  EnvVarGuard backend_guard("MYSQL_VECTOR_TRUTH_STORE");
  EnvVarGuard diag_guard("MYSQL_VECTOR_DIAG_FILE");
  setenv("MYSQL_VECTOR_TRUTH_STORE", "file", 1);
  const std::string path =
      std::string(DATA_DIR) + "/vector_truth_store_roundtrip_t.dat";
  const std::string diag_path =
      std::string(DATA_DIR) + "/vector_truth_store_roundtrip_diag.jsonl";
  vector_index_metadata_store::set_path_for_testing(path);
  std::remove(path.c_str());
  std::remove(diag_path.c_str());
  std::remove((path + ".committed").c_str());
  std::remove((path + ".manifest").c_str());
  std::remove((path + ".changelog").c_str());
  std::remove((path + ".prepared").c_str());
  std::remove((path + ".quarantine").c_str());
  std::remove((path + ".corrupt").c_str());
  std::remove((path + ".committed.corrupt").c_str());
  std::remove((path + ".manifest.corrupt").c_str());
  std::remove((path + ".changelog.corrupt").c_str());
  std::remove((path + ".prepared.corrupt").c_str());
  setenv("MYSQL_VECTOR_DIAG_FILE", diag_path.c_str(), 1);

  vector_index_truth_store::reset_for_testing();
  vector_index_truth_store::truth_store *store =
      vector_index_truth_store::get();
  ASSERT_NE(nullptr, store);

  vector_index_metadata_store::metadata_row metadata_row;
  metadata_row.index_name = "idx_truth";
  metadata_row.dimension = 2;
  metadata_row.metric = vector_index::metric_type::kEuclidean;
  metadata_row.mode = vector_index::backend_mode::kMemory;
  metadata_row.provider = vector_index::backend_provider::kNative;
  metadata_row.owner_schema = "test";

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows{
      metadata_row};
  ASSERT_TRUE(store->save_metadata(metadata_rows));
  std::vector<vector_index_metadata_store::metadata_row> loaded_metadata;
  ASSERT_TRUE(store->load_metadata(&loaded_metadata));
  ASSERT_EQ(1U, loaded_metadata.size());
  EXPECT_EQ("idx_truth", loaded_metadata[0].index_name);

  std::vector<vector_index_metadata_store::committed_row> committed_rows{
      {"idx_truth", 11, {1.0F, 2.0F}}};
  ASSERT_TRUE(store->save_committed(committed_rows));
  std::vector<vector_index_metadata_store::committed_row> loaded_committed;
  ASSERT_TRUE(store->load_committed(&loaded_committed));
  ASSERT_EQ(1U, loaded_committed.size());
  EXPECT_EQ(11U, loaded_committed[0].doc_id);

  vector_index_metadata_store::manifest_row manifest_row;
  manifest_row.state = "ready";
  manifest_row.version = 3;
  manifest_row.metadata_checkpoint = 1;
  manifest_row.committed_checkpoint = 1;
  manifest_row.change_log_checkpoint = 0;
  ASSERT_TRUE(store->save_manifest(manifest_row));
  vector_index_metadata_store::manifest_row loaded_manifest;
  ASSERT_TRUE(store->load_manifest(&loaded_manifest));
  EXPECT_EQ(3U, loaded_manifest.version);

  std::vector<vector_index_metadata_store::change_log_row> change_log_rows{
      {1,
       99,
       vector_index_metadata_store::change_op::kUpsert,
       "idx_truth",
       11,
       {1.0F, 2.0F},
       42,
       1,
       1}};
  ASSERT_TRUE(store->save_change_log(change_log_rows));
  std::vector<vector_index_metadata_store::change_log_row> loaded_change_log;
  ASSERT_TRUE(store->load_change_log(&loaded_change_log));
  ASSERT_EQ(1U, loaded_change_log.size());
  EXPECT_EQ(99U, loaded_change_log[0].txn_id);

  {
    std::ifstream diag_file(diag_path);
    ASSERT_TRUE(diag_file.good());
    const std::string diag_payload((std::istreambuf_iterator<char>(diag_file)),
                                   std::istreambuf_iterator<char>());
    EXPECT_NE(std::string::npos,
              diag_payload.find("\"artifact\":\"committed\""));
    EXPECT_NE(std::string::npos,
              diag_payload.find("\"artifact\":\"changelog\""));
  }

  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows{
      {1,
       16,
       0,
       std::string("0123456789abcdef", 16),
       false,
       99,
       vector_index_metadata_store::change_op::kUpsert,
       "idx_truth",
       11,
       {1.0F, 2.0F}}};
  ASSERT_TRUE(store->save_prepared(prepared_rows));
  std::vector<vector_index_metadata_store::prepared_change_row> loaded_prepared;
  ASSERT_TRUE(store->load_prepared(&loaded_prepared));
  ASSERT_EQ(1U, loaded_prepared.size());
  EXPECT_EQ(99U, loaded_prepared[0].txn_id);
  EXPECT_EQ("idx_truth", loaded_prepared[0].index_name);

  std::vector<vector_index_metadata_store::segment_task_row> segment_rows(1);
  segment_rows[0].index_name = "idx_truth";
  segment_rows[0].generation = 3;
  segment_rows[0].segment_id = 7;
  segment_rows[0].state =
      vector_index_metadata_store::segment_task_state::kReady;
  segment_rows[0].row_count = 1;
  segment_rows[0].payload_size = 8;
  segment_rows[0].vector_path = "/tmp/vector.fbin";
  segment_rows[0].docid_path = "/tmp/vector.u64";
  segment_rows[0].artifact_prefix = "/tmp/segment-7";
  ASSERT_TRUE(store->save_segment_tasks(segment_rows));
  std::vector<vector_index_metadata_store::segment_task_row>
      loaded_segment_rows;
  ASSERT_TRUE(store->load_segment_tasks(&loaded_segment_rows));
  ASSERT_EQ(1U, loaded_segment_rows.size());
  EXPECT_EQ(7U, loaded_segment_rows[0].segment_id);

  std::string quarantine_identity;
  ASSERT_TRUE(store->stage_quarantine("committed", "decode_failure", 3,
                                      &quarantine_identity));
  EXPECT_FALSE(quarantine_identity.empty());
  ASSERT_TRUE(store->update_quarantine_state(
      quarantine_identity,
      vector_index_truth_store::quarantine_state::kSourceUpdateFailed));
  ASSERT_TRUE(store->update_quarantine_state(
      quarantine_identity,
      vector_index_truth_store::quarantine_state::kComplete));
  std::string quarantine_payload;
  bool quarantine_found = false;
  ASSERT_TRUE(vector_index_metadata_store::load_raw_artifact(
      "quarantine_store", &quarantine_payload, &quarantine_found));
  ASSERT_TRUE(quarantine_found);
  std::vector<vector_index_truth_store::quarantine_record> quarantine_records;
  ASSERT_TRUE(
      vector_index_truth_store::deserialize_quarantine_entries_for_testing(
          quarantine_payload, &quarantine_records));
  ASSERT_EQ(1U, quarantine_records.size());
  EXPECT_EQ(quarantine_identity, quarantine_records[0].identity);
  EXPECT_EQ("committed", quarantine_records[0].artifact_name);
  EXPECT_EQ("decode_failure", quarantine_records[0].reason);
  EXPECT_EQ(3U, quarantine_records[0].generation);
  EXPECT_EQ(vector_index_truth_store::quarantine_state::kComplete,
            quarantine_records[0].state);

  ASSERT_TRUE(store->quarantine_segment_tasks());

  vector_index_metadata_store::reset_path_for_testing();
  std::remove(path.c_str());
  std::remove(diag_path.c_str());
  std::remove((path + ".committed").c_str());
  std::remove((path + ".manifest").c_str());
  std::remove((path + ".changelog").c_str());
  std::remove((path + ".prepared").c_str());
  std::remove((path + ".quarantine").c_str());
  std::remove((path + ".segment_tasks").c_str());
  std::remove((path + ".corrupt").c_str());
  std::remove((path + ".committed.corrupt").c_str());
  std::remove((path + ".manifest.corrupt").c_str());
  std::remove((path + ".changelog.corrupt").c_str());
  std::remove((path + ".prepared.corrupt").c_str());
  std::remove((path + ".segment_tasks.corrupt").c_str());
}

}  // namespace vector_index_truth_store_unittest
