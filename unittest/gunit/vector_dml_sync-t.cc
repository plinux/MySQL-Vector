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

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "m_ctype.h"
#include "my_byteorder.h"
#include "mysqld_error.h"
#include "sql/vector/vector_dml_sync.h"
#include "sql/vector/vector_index_metadata_store.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_trx_participant.h"
#include "unittest/gunit/base_mock_field.h"
#include "unittest/gunit/fake_table.h"
#include "unittest/gunit/mock_field_long.h"
#include "unittest/gunit/test_utils.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_dml_sync_unittest {

namespace {

using my_testing::Server_initializer;

class TestBlobField : public Field_blob {
 public:
  explicit TestBlobField(const char *field_name, bool nullable)
      : Field_blob(MAX_BLOB_WIDTH, nullable, field_name, &my_charset_bin, true) {}
};

class in_memory_truth_store final : public vector_index_truth_store::truth_store {
 public:
  const char *backend_name() const override { return "memory"; }
  bool is_transactional() const override { return true; }
  bool supports_attached_dml() const override { return true; }

  bool apply_attached_dml(
      THD *,
      const std::vector<vector_index_metadata_store::change_log_row> &rows)
      override {
    ++apply_attached_dml_calls;
    if (fail_apply_attached_dml) return false;
    attached_rows.insert(attached_rows.end(), rows.begin(), rows.end());
    return true;
  }

  void commit_attached_dml() {
    for (const auto &row : attached_rows) {
      auto it = std::find_if(committed_rows.begin(), committed_rows.end(),
                             [&](const auto &entry) {
                               return entry.index_name == row.index_name &&
                                      entry.doc_id == row.doc_id;
                             });
      if (row.op == vector_index_metadata_store::change_op::kErase) {
        if (it != committed_rows.end()) committed_rows.erase(it);
      } else if (it == committed_rows.end()) {
        committed_rows.push_back({row.index_name, row.doc_id, row.vector});
      } else {
        it->vector = row.vector;
      }
    }
    change_log_rows.insert(change_log_rows.end(), attached_rows.begin(),
                           attached_rows.end());
    attached_rows.clear();
  }

  void rollback_attached_dml() { attached_rows.clear(); }

  bool load_metadata(
      std::vector<vector_index_metadata_store::metadata_row> *rows) override {
    if (rows == nullptr) return false;
    *rows = metadata_rows;
    return true;
  }
  bool save_metadata(
      const std::vector<vector_index_metadata_store::metadata_row> &rows) override {
    metadata_rows = rows;
    return true;
  }
  bool quarantine_metadata() override {
    metadata_rows.clear();
    return true;
  }

  bool load_committed(
      std::vector<vector_index_metadata_store::committed_row> *rows) override {
    if (rows == nullptr) return false;
    *rows = committed_rows;
    return true;
  }
  bool save_committed(
      const std::vector<vector_index_metadata_store::committed_row> &rows) override {
    committed_rows = rows;
    return true;
  }
  bool quarantine_committed() override {
    committed_rows.clear();
    return true;
  }

  bool load_manifest(vector_index_metadata_store::manifest_row *row) override {
    if (row == nullptr) return false;
    *row = manifest_row;
    return true;
  }
  bool save_manifest(
      const vector_index_metadata_store::manifest_row &row) override {
    manifest_row = row;
    return true;
  }
  bool quarantine_manifest() override {
    manifest_row = vector_index_metadata_store::manifest_row();
    return true;
  }

  bool load_change_log(
      std::vector<vector_index_metadata_store::change_log_row> *rows) override {
    if (rows == nullptr) return false;
    *rows = change_log_rows;
    return true;
  }
  bool save_change_log(
      const std::vector<vector_index_metadata_store::change_log_row> &rows) override {
    change_log_rows = rows;
    return true;
  }
  bool quarantine_change_log() override {
    change_log_rows.clear();
    return true;
  }

  bool load_prepared(
      std::vector<vector_index_metadata_store::prepared_change_row> *rows) override {
    if (rows == nullptr) return false;
    *rows = prepared_rows;
    return true;
  }
  bool save_prepared(
      const std::vector<vector_index_metadata_store::prepared_change_row> &rows) override {
    prepared_rows = rows;
    return true;
  }
  bool quarantine_prepared() override {
    prepared_rows.clear();
    return true;
  }

  bool load_segment_tasks(
      std::vector<vector_index_metadata_store::segment_task_row> *rows)
      override {
    if (rows == nullptr) return false;
    rows->clear();
    return true;
  }
  bool save_segment_tasks(
      const std::vector<vector_index_metadata_store::segment_task_row> &) override {
    return true;
  }
  bool quarantine_segment_tasks() override { return true; }

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  std::vector<vector_index_metadata_store::change_log_row> attached_rows;
  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  vector_index_metadata_store::manifest_row manifest_row;
  bool fail_apply_attached_dml{false};
  uint64_t apply_attached_dml_calls{0};
};

std::string binary_vector_payload(std::initializer_list<float> values) {
  std::string payload(values.size() * sizeof(float), '\0');
  size_t idx = 0;
  for (float value : values) {
    float4store(reinterpret_cast<uchar *>(payload.data() + idx * sizeof(float)),
                value);
    ++idx;
  }
  return payload;
}

void assign_row(Fake_TABLE *table, Mock_field_long *id_field,
                TestBlobField *vector_field, longlong doc_id,
                const std::string *payload, uchar *target_record) {
  ASSERT_NE(nullptr, table);
  ASSERT_NE(nullptr, id_field);
  ASSERT_NE(nullptr, vector_field);
  ASSERT_NE(nullptr, target_record);

  bitmap_set_bit(table->write_set, id_field->field_index());
  bitmap_set_bit(table->write_set, vector_field->field_index());
  ASSERT_EQ(TYPE_OK, id_field->store(doc_id, false));
  if (payload == nullptr) {
    vector_field->set_null();
  } else {
    vector_field->set_notnull();
    vector_field->set_ptr(payload->size(),
                          pointer_cast<const uchar *>(payload->data()));
  }
  std::memcpy(target_record, table->record[0], MAX_FIELD_WIDTH * MAX_TABLE_COLUMNS);
}

class VectorDmlSyncFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    initializer_.SetUp();
    vector_trx_participant::set_registration_bypass_for_testing(true);
    vector_index_truth_store::set_for_testing(&store_);
    vector_index_registry::reset_for_testing();
  }

  void TearDown() override {
    vector_index_registry::reset_for_testing();
    vector_index_truth_store::reset_for_testing();
    vector_trx_participant::set_registration_bypass_for_testing(false);
    initializer_.TearDown();
  }

  THD *thd() { return initializer_.thd(); }

  std::unique_ptr<Fake_TABLE> MakeVectorTable(const char *db_name = "db_sync",
                                              const char *table_name = "t_sync",
                                              bool nullable_doc_id = false,
                                              bool nullable_vector = true) {
    auto *id_field = new (*THR_MALLOC)
        Mock_field_long("id", nullable_doc_id, false);
    auto *vector_field = new (*THR_MALLOC)
        TestBlobField("vec_col", nullable_vector);
    vector_field->set_flag(FIELD_IS_VECTOR);

    auto table = std::make_unique<Fake_TABLE>(id_field, vector_field);
    table->get_share()->db = {const_cast<char *>(db_name), std::strlen(db_name)};
    table->get_share()->table_name = {const_cast<char *>(table_name),
                                      std::strlen(table_name)};
    const int index_id = table->create_index(id_field, nullptr, true);
    table->get_share()->primary_key = static_cast<uint>(index_id);
    return table;
  }

  Mock_field_long *doc_id_field(Fake_TABLE *table) {
    return down_cast<Mock_field_long *>(table->field[0]);
  }

  TestBlobField *vector_field(Fake_TABLE *table) {
    return down_cast<TestBlobField *>(table->field[1]);
  }

  std::string mapped_index_name(Fake_TABLE *table) const {
    return std::string(table->s->db.str) + "." + table->s->table_name.str +
           "." + table->field[1]->field_name;
  }

  void CreateMappedIndexForTable(Fake_TABLE *table) {
    ASSERT_TRUE(vector_index_registry::create_mapped_index(
        mapped_index_name(table), 2, "euclidean", "memory", "native",
        table->s->db.str, table->s->table_name.str, table->field[1]->field_name,
        table->field[0]->field_name));
  }

  vector_gunit::NativeProviderGuard native_provider_guard_;
  Server_initializer initializer_;
  in_memory_truth_store store_;
};

}  // namespace

TEST(VectorDmlSyncTest, HasVectorColumnsRejectsNullTable) {
  EXPECT_FALSE(vector_dml_sync::has_vector_columns(nullptr));
}

TEST_F(VectorDmlSyncFixture, SupportsVectorDocIdCoversPrimaryKeyShapes) {
  auto table = MakeVectorTable();
  EXPECT_TRUE(vector_dml_sync::supports_vector_doc_id(table.get()));

  auto no_primary_key = MakeVectorTable();
  no_primary_key->get_share()->primary_key = MAX_KEY;
  EXPECT_FALSE(vector_dml_sync::supports_vector_doc_id(no_primary_key.get()));

  auto nullable_doc_id = MakeVectorTable("db_sync", "t_null_pk", true);
  EXPECT_FALSE(vector_dml_sync::supports_vector_doc_id(nullable_doc_id.get()));

  auto two_part_pk = MakeVectorTable("db_sync", "t_two_pk");
  auto *extra = new (*THR_MALLOC) Mock_field_long("other_col", false, false);
  two_part_pk->field[1]->clear_flag(FIELD_IS_VECTOR);
  auto composite = std::make_unique<Fake_TABLE>(doc_id_field(two_part_pk.get()), extra,
                                                vector_field(two_part_pk.get()));
  composite->get_share()->db = {const_cast<char *>("db_sync"), 7};
  composite->get_share()->table_name = {const_cast<char *>("t_two_pk"), 8};
  composite->field[2]->set_flag(FIELD_IS_VECTOR);
  const int key_id =
      composite->create_index(doc_id_field(composite.get()), extra, true);
  composite->get_share()->primary_key = static_cast<uint>(key_id);
  EXPECT_FALSE(vector_dml_sync::supports_vector_doc_id(composite.get()));
}

TEST(VectorDmlSyncTest, PrepareHelpersRejectNullInputs) {
  vector_dml_sync::prepared_changes changes;
  EXPECT_TRUE(vector_dml_sync::prepare_insert_row(nullptr, nullptr, &changes));
  EXPECT_TRUE(vector_dml_sync::prepare_delete_row(nullptr, nullptr, &changes));
  EXPECT_TRUE(
      vector_dml_sync::prepare_update_row(nullptr, nullptr, nullptr, &changes));
}

TEST_F(VectorDmlSyncFixture, PrepareRowWrappersCoverNullArgumentCombinations) {
  auto table = MakeVectorTable("db_null_args", "t_null_args");
  CreateMappedIndexForTable(table.get());

  const std::string payload = binary_vector_payload({1.0F, 2.0F});
  std::vector<uchar> old_record(MAX_FIELD_WIDTH * MAX_TABLE_COLUMNS);
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 71,
             &payload, old_record.data());
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 72,
             &payload, table->record[0]);

  vector_dml_sync::prepared_changes changes;
  EXPECT_TRUE(
      vector_dml_sync::prepare_insert_row(table.get(), nullptr, &changes));
  EXPECT_TRUE(
      vector_dml_sync::prepare_insert_row(table.get(), table->record[0], nullptr));
  EXPECT_TRUE(
      vector_dml_sync::prepare_delete_row(table.get(), nullptr, &changes));
  EXPECT_TRUE(
      vector_dml_sync::prepare_delete_row(table.get(), table->record[0], nullptr));
  EXPECT_TRUE(vector_dml_sync::prepare_update_row(table.get(), nullptr,
                                                  table->record[0], &changes));
  EXPECT_TRUE(vector_dml_sync::prepare_update_row(
      table.get(), old_record.data(), nullptr, &changes));
  EXPECT_TRUE(vector_dml_sync::prepare_update_row(
      table.get(), old_record.data(), table->record[0], nullptr));
}

TEST_F(VectorDmlSyncFixture, PrepareInsertDeleteAndUpdateCoverVectorBranches) {
  auto table = MakeVectorTable();
  CreateMappedIndexForTable(table.get());

  const std::string payload = binary_vector_payload({1.0F, 2.0F});
  vector_dml_sync::prepared_changes changes;

  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 7,
             &payload, table->record[0]);
  EXPECT_FALSE(
      vector_dml_sync::prepare_insert_row(table.get(), table->record[0], &changes));
  ASSERT_EQ(1U, changes.size());
  EXPECT_EQ(mapped_index_name(table.get()), changes[0].index_name);
  EXPECT_EQ(7U, changes[0].doc_id);
  EXPECT_FALSE(changes[0].erase);
  ASSERT_EQ(2U, changes[0].vector.size());
  EXPECT_FLOAT_EQ(1.0F, changes[0].vector[0]);
  EXPECT_FLOAT_EQ(2.0F, changes[0].vector[1]);

  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 8,
             nullptr, table->record[0]);
  EXPECT_FALSE(
      vector_dml_sync::prepare_insert_row(table.get(), table->record[0], &changes));
  ASSERT_EQ(1U, changes.size());
  EXPECT_TRUE(changes[0].erase);
  EXPECT_EQ(8U, changes[0].doc_id);

  const std::string bad_payload = "bad";
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 9,
             &bad_payload, table->record[0]);
  Server_initializer::set_expected_error(ER_INTERNAL_ERROR);
  EXPECT_TRUE(
      vector_dml_sync::prepare_insert_row(table.get(), table->record[0], &changes));
  thd()->clear_error();
  Server_initializer::set_expected_error(0);

  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 10,
             &payload, table->record[0]);
  EXPECT_FALSE(
      vector_dml_sync::prepare_delete_row(table.get(), table->record[0], &changes));
  ASSERT_EQ(1U, changes.size());
  EXPECT_TRUE(changes[0].erase);
  EXPECT_EQ(10U, changes[0].doc_id);

  std::vector<uchar> old_record(MAX_FIELD_WIDTH * MAX_TABLE_COLUMNS);
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 11,
             &payload, old_record.data());
  table->record[1] = old_record.data();

  const std::string new_payload = binary_vector_payload({3.0F, 4.0F});
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 12,
             &new_payload, table->record[0]);
  EXPECT_FALSE(vector_dml_sync::prepare_update_row(table.get(), table->record[1],
                                                 table->record[0], &changes));
  ASSERT_EQ(2U, changes.size());
  EXPECT_TRUE(changes[0].erase);
  EXPECT_EQ(11U, changes[0].doc_id);
  EXPECT_FALSE(changes[1].erase);
  EXPECT_EQ(12U, changes[1].doc_id);
  ASSERT_EQ(2U, changes[1].vector.size());
  EXPECT_FLOAT_EQ(3.0F, changes[1].vector[0]);
  EXPECT_FLOAT_EQ(4.0F, changes[1].vector[1]);
}

TEST_F(VectorDmlSyncFixture, PrepareHelpersRejectMissingVectorColumnsAndDocId) {
  auto no_vector = MakeVectorTable("db_novec", "t_novec");
  no_vector->field[1]->clear_flag(FIELD_IS_VECTOR);
  vector_dml_sync::prepared_changes changes;
  const std::string payload = binary_vector_payload({1.0F, 2.0F});
  assign_row(no_vector.get(), doc_id_field(no_vector.get()),
             vector_field(no_vector.get()), 1, &payload, no_vector->record[0]);
  EXPECT_FALSE(vector_dml_sync::prepare_insert_row(no_vector.get(),
                                                 no_vector->record[0], &changes));
  EXPECT_TRUE(changes.empty());
  EXPECT_FALSE(vector_dml_sync::prepare_delete_row(no_vector.get(),
                                                 no_vector->record[0], &changes));
  EXPECT_TRUE(changes.empty());
  EXPECT_FALSE(vector_dml_sync::prepare_update_row(
      no_vector.get(), no_vector->record[0], no_vector->record[0], &changes));
  EXPECT_TRUE(changes.empty());

  auto bad_pk = MakeVectorTable("db_badpk", "t_badpk");
  bad_pk->get_share()->primary_key = MAX_KEY;
  assign_row(bad_pk.get(), doc_id_field(bad_pk.get()), vector_field(bad_pk.get()),
             2, &payload, bad_pk->record[0]);
  EXPECT_FALSE(vector_dml_sync::prepare_insert_row(bad_pk.get(), bad_pk->record[0],
                                                 &changes));
  EXPECT_FALSE(vector_dml_sync::prepare_delete_row(bad_pk.get(), bad_pk->record[0],
                                                 &changes));
  EXPECT_FALSE(vector_dml_sync::prepare_update_row(
      bad_pk.get(), bad_pk->record[0], bad_pk->record[0], &changes));
}

TEST_F(VectorDmlSyncFixture,
       PrepareInsertRowForIndexCoversSelectionAndDecodeBranches) {
  auto table = MakeVectorTable("db_explicit", "t_explicit");
  const std::string payload = binary_vector_payload({4.0F, 5.0F});
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 80,
             &payload, table->record[0]);

  vector_dml_sync::prepared_changes changes;
  EXPECT_FALSE(vector_dml_sync::prepare_insert_row_for_index(
      table.get(), "explicit_index", "vec_col", table->record[0], &changes));
  ASSERT_EQ(1U, changes.size());
  EXPECT_EQ("explicit_index", changes[0].index_name);
  EXPECT_EQ(80U, changes[0].doc_id);
  ASSERT_EQ(2U, changes[0].vector.size());
  EXPECT_FLOAT_EQ(4.0F, changes[0].vector[0]);
  EXPECT_FLOAT_EQ(5.0F, changes[0].vector[1]);

  EXPECT_TRUE(vector_dml_sync::prepare_insert_row_for_index(
      nullptr, "explicit_index", "vec_col", table->record[0], &changes));
  EXPECT_TRUE(vector_dml_sync::prepare_insert_row_for_index(
      table.get(), "explicit_index", "vec_col", nullptr, &changes));
  EXPECT_TRUE(vector_dml_sync::prepare_insert_row_for_index(
      table.get(), "explicit_index", "vec_col", table->record[0], nullptr));

  auto bad_pk = MakeVectorTable("db_explicit", "t_bad_pk");
  bad_pk->get_share()->primary_key = MAX_KEY;
  assign_row(bad_pk.get(), doc_id_field(bad_pk.get()), vector_field(bad_pk.get()),
             81, &payload, bad_pk->record[0]);
  EXPECT_FALSE(vector_dml_sync::prepare_insert_row_for_index(
      bad_pk.get(), "explicit_index", "vec_col", bad_pk->record[0], &changes));

  EXPECT_FALSE(vector_dml_sync::prepare_insert_row_for_index(
      table.get(), "explicit_index", "missing_vec_col", table->record[0],
      &changes));

  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 82,
             nullptr, table->record[0]);
  EXPECT_FALSE(vector_dml_sync::prepare_insert_row_for_index(
      table.get(), "explicit_index", "vec_col", table->record[0], &changes));

  const std::string bad_payload = "bad";
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 83,
             &bad_payload, table->record[0]);
  Server_initializer::set_expected_error(ER_INTERNAL_ERROR);
  EXPECT_TRUE(vector_dml_sync::prepare_insert_row_for_index(
      table.get(), "explicit_index", "vec_col", table->record[0], &changes));
  thd()->clear_error();
  Server_initializer::set_expected_error(0);
}

TEST_F(VectorDmlSyncFixture, PrepareRowsRejectNegativeDocIds) {
  auto table = MakeVectorTable("db_negative", "t_negative");
  CreateMappedIndexForTable(table.get());

  const std::string payload = binary_vector_payload({1.0F, 2.0F});
  vector_dml_sync::prepared_changes changes;

  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), -1,
             &payload, table->record[0]);
  EXPECT_FALSE(
      vector_dml_sync::prepare_delete_row(table.get(), table->record[0], &changes));

  std::vector<uchar> old_record(MAX_FIELD_WIDTH * MAX_TABLE_COLUMNS);
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), -2,
             &payload, old_record.data());
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 84,
             &payload, table->record[0]);
  EXPECT_FALSE(vector_dml_sync::prepare_update_row(
      table.get(), old_record.data(), table->record[0], &changes));

  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 85,
             &payload, old_record.data());
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), -3,
             &payload, table->record[0]);
  EXPECT_FALSE(vector_dml_sync::prepare_update_row(
      table.get(), old_record.data(), table->record[0], &changes));
}

TEST_F(VectorDmlSyncFixture, HelperWrappersCoverDocIdAndVectorDecodingBranches) {
  auto table = MakeVectorTable("db_helper", "t_helper");
  CreateMappedIndexForTable(table.get());

  Field *pk_field = nullptr;
  ASSERT_TRUE(vector_dml_sync::get_doc_id_field_for_testing(table.get(), &pk_field));
  EXPECT_EQ(doc_id_field(table.get()), pk_field);
  EXPECT_FALSE(vector_dml_sync::get_doc_id_field_for_testing(nullptr, &pk_field));
  EXPECT_FALSE(vector_dml_sync::get_doc_id_field_for_testing(table.get(), nullptr));

  const std::string payload = binary_vector_payload({9.0F, 10.0F});
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 15,
             &payload, table->record[0]);

  uint64_t doc_id = 0;
  EXPECT_FALSE(vector_dml_sync::get_doc_id_for_record_for_testing(nullptr, table->record[0],
                                                            &doc_id));
  EXPECT_FALSE(vector_dml_sync::get_doc_id_for_record_for_testing(pk_field, nullptr,
                                                            &doc_id));
  EXPECT_FALSE(vector_dml_sync::get_doc_id_for_record_for_testing(pk_field, table->record[0],
                                                            nullptr));
  EXPECT_TRUE(vector_dml_sync::get_doc_id_for_record_for_testing(pk_field, table->record[0],
                                                           &doc_id));
  EXPECT_EQ(15U, doc_id);

  auto nullable_doc_id = MakeVectorTable("db_nullid", "t_nullid", true);
  bitmap_set_bit(nullable_doc_id->write_set,
                 doc_id_field(nullable_doc_id.get())->field_index());
  doc_id_field(nullable_doc_id.get())->set_null();
  EXPECT_FALSE(vector_dml_sync::get_doc_id_for_record_for_testing(
      doc_id_field(nullable_doc_id.get()), nullable_doc_id->record[0], &doc_id));

  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), -1,
             &payload, table->record[0]);
  EXPECT_FALSE(vector_dml_sync::get_doc_id_for_record_for_testing(pk_field, table->record[0],
                                                            &doc_id));

  std::vector<float> vector;
  EXPECT_FALSE(vector_dml_sync::decode_vector_field_for_testing(nullptr, table->record[0],
                                                            &vector));
  EXPECT_FALSE(vector_dml_sync::decode_vector_field_for_testing(doc_id_field(table.get()),
                                                            table->record[0], &vector));
  EXPECT_FALSE(vector_dml_sync::decode_vector_field_for_testing(vector_field(table.get()),
                                                            nullptr, &vector));
  EXPECT_FALSE(vector_dml_sync::decode_vector_field_for_testing(vector_field(table.get()),
                                                            table->record[0], nullptr));

  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 16,
             nullptr, table->record[0]);
  EXPECT_FALSE(vector_dml_sync::decode_vector_field_for_testing(vector_field(table.get()),
                                                            table->record[0], &vector));

  vector_field(table.get())->set_notnull();
  vector_field(table.get())->set_ptr(static_cast<uint32>(0),
                                     pointer_cast<const uchar *>(payload.data()));
  EXPECT_FALSE(vector_dml_sync::decode_vector_field_for_testing(vector_field(table.get()),
                                                            table->record[0], &vector));

  vector_field(table.get())->set_notnull();
  vector_field(table.get())->set_ptr(static_cast<uint32>(payload.size()),
                                     nullptr);
  EXPECT_FALSE(vector_dml_sync::decode_vector_field_for_testing(vector_field(table.get()),
                                                            table->record[0], &vector));

  const std::string bad_payload = "bad";
  vector_field(table.get())->set_notnull();
  vector_field(table.get())->set_ptr(bad_payload.size(),
                                     pointer_cast<const uchar *>(bad_payload.data()));
  EXPECT_FALSE(vector_dml_sync::decode_vector_field_for_testing(vector_field(table.get()),
                                                            table->record[0], &vector));

  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 17,
             &payload, table->record[0]);
  ASSERT_TRUE(vector_dml_sync::decode_vector_field_for_testing(vector_field(table.get()),
                                                           table->record[0], &vector));
  ASSERT_EQ(2U, vector.size());
  EXPECT_FLOAT_EQ(9.0F, vector[0]);
  EXPECT_FLOAT_EQ(10.0F, vector[1]);
}

TEST_F(VectorDmlSyncFixture, HelperWrappersCoverCollectBranches) {
  auto table = MakeVectorTable("db_collect", "t_collect");
  CreateMappedIndexForTable(table.get());

  const std::string payload = binary_vector_payload({6.0F, 7.0F});
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 31,
             &payload, table->record[0]);

  vector_dml_sync::prepared_changes changes;
  EXPECT_FALSE(vector_dml_sync::collect_upsert_for_field_for_testing(
      table.get(), vector_field(table.get()), 31, table->record[0], &changes));
  ASSERT_EQ(1U, changes.size());
  EXPECT_FALSE(changes[0].erase);

  changes.clear();
  EXPECT_TRUE(vector_dml_sync::collect_upsert_for_field_for_testing(
      table.get(), vector_field(table.get()), 31, table->record[0], nullptr));

  const std::string bad_payload = "bad";
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 31,
             &bad_payload, table->record[0]);
  Server_initializer::set_expected_error(ER_INTERNAL_ERROR);
  EXPECT_TRUE(vector_dml_sync::collect_upsert_for_field_for_testing(
      table.get(), vector_field(table.get()), 31, table->record[0], &changes));
  thd()->clear_error();
  Server_initializer::set_expected_error(0);

  changes.clear();
  EXPECT_FALSE(vector_dml_sync::collect_erase_for_field_for_testing(
      table.get(), vector_field(table.get()), 31, &changes));
  ASSERT_EQ(1U, changes.size());
  EXPECT_TRUE(changes[0].erase);

  EXPECT_TRUE(vector_dml_sync::collect_erase_for_field_for_testing(
      table.get(), vector_field(table.get()), 31, nullptr));

  auto missing = MakeVectorTable("db_missing", "t_missing");
  const std::string missing_payload = binary_vector_payload({1.0F, 2.0F});
  assign_row(missing.get(), doc_id_field(missing.get()), vector_field(missing.get()),
             32, &missing_payload, missing->record[0]);
  EXPECT_FALSE(vector_dml_sync::collect_upsert_for_field_for_testing(
      missing.get(), vector_field(missing.get()), 32, missing->record[0], &changes));
  EXPECT_FALSE(vector_dml_sync::collect_erase_for_field_for_testing(
      missing.get(), vector_field(missing.get()), 32, &changes));
}

TEST_F(VectorDmlSyncFixture, StagePreparedChangesCoversSuccessAndFailure) {
  auto table = MakeVectorTable("db_stage", "t_stage");
  CreateMappedIndexForTable(table.get());

  vector_dml_sync::prepared_changes changes;
  changes.push_back({mapped_index_name(table.get()), 21, false, {2.0F, 2.0F}});
  changes.push_back({mapped_index_name(table.get()), 22, true, {}});

  thd()->set_query_id(101);
  EXPECT_FALSE(vector_dml_sync::stage_prepared_changes(thd(), changes));
  EXPECT_EQ(1U, store_.apply_attached_dml_calls);
  ASSERT_EQ(2U, store_.attached_rows.size());
  EXPECT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(
      static_cast<uint64_t>(thd()->thread_id()),
      static_cast<uint64_t>(thd()->query_id)));
  store_.commit_attached_dml();
  EXPECT_TRUE(vector_index_registry::publish_thd_txn(
      static_cast<uint64_t>(thd()->thread_id())));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(mapped_index_name(table.get()), {2.0F, 2.0F},
                                    1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(21U, result[0].doc_id);

  thd()->set_query_id(102);
  vector_dml_sync::prepared_changes bad_changes;
  bad_changes.push_back({"missing.index", 1, false, {1.0F, 1.0F}});
  Server_initializer::set_expected_error(ER_INTERNAL_ERROR);
  EXPECT_TRUE(vector_dml_sync::stage_prepared_changes(thd(), bad_changes));
  thd()->clear_error();
  Server_initializer::set_expected_error(0);

  store_.fail_apply_attached_dml = true;
  thd()->set_query_id(103);
  vector_dml_sync::prepared_changes persist_failure;
  persist_failure.push_back(
      {mapped_index_name(table.get()), 23, false, {3.0F, 3.0F}});
  Server_initializer::set_expected_error(ER_INTERNAL_ERROR);
  EXPECT_TRUE(vector_dml_sync::stage_prepared_changes(thd(), persist_failure));
  thd()->clear_error();
  Server_initializer::set_expected_error(0);
  EXPECT_EQ(0U, vector_index_registry::total_pending_vector_memory_bytes());
  EXPECT_TRUE(store_.attached_rows.empty());
}

TEST_F(VectorDmlSyncFixture, StageUpdateRowCoversSuccessPath) {
  auto table = MakeVectorTable("db_stage_update", "t_stage_update");
  CreateMappedIndexForTable(table.get());

  const std::string old_payload = binary_vector_payload({1.0F, 2.0F});
  std::vector<uchar> old_record(MAX_FIELD_WIDTH * MAX_TABLE_COLUMNS);
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 41,
             &old_payload, old_record.data());
  table->record[1] = old_record.data();

  const std::string new_payload = binary_vector_payload({8.0F, 9.0F});
  assign_row(table.get(), doc_id_field(table.get()), vector_field(table.get()), 41,
             &new_payload, table->record[0]);

  thd()->set_query_id(201);
  EXPECT_FALSE(vector_dml_sync::stage_update_row(thd(), table.get(), table->record[1],
                                               table->record[0]));
  EXPECT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(
      static_cast<uint64_t>(thd()->thread_id()),
      static_cast<uint64_t>(thd()->query_id)));
  store_.commit_attached_dml();
  EXPECT_TRUE(vector_index_registry::publish_thd_txn(
      static_cast<uint64_t>(thd()->thread_id())));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search(mapped_index_name(table.get()),
                                            {8.0F, 9.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(41U, result[0].doc_id);
}

TEST(VectorDmlSyncTest, StagePreparedChangesHandlesNullThd) {
  vector_dml_sync::prepared_changes changes;
  EXPECT_TRUE(vector_dml_sync::stage_prepared_changes(nullptr, changes));

  vector_dml_sync::prepared_change staged;
  staged.index_name = "idx_cov";
  staged.doc_id = 1;
  staged.erase = false;
  staged.vector = {1.0F, 2.0F};
  changes.push_back(staged);
  EXPECT_TRUE(vector_dml_sync::stage_prepared_changes(nullptr, changes));
}

TEST(VectorDmlSyncTest, StageWrappersPropagateInvalidInputErrors) {
  EXPECT_TRUE(vector_dml_sync::stage_insert_row(nullptr, nullptr, nullptr));
  EXPECT_TRUE(vector_dml_sync::stage_delete_row(nullptr, nullptr, nullptr));
  EXPECT_TRUE(
      vector_dml_sync::stage_update_row(nullptr, nullptr, nullptr, nullptr));
}

}  // namespace vector_dml_sync_unittest
