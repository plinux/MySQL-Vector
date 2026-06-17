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
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "my_io.h"
#include "sql/mysqld.h"
#include "sql/vector/vector_index_metadata_store.h"
#include "sql/vector/vector_index_metadata_store_internal.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_index_metadata_store_unittest {

namespace {

class DataHomeGuard {
 public:
  DataHomeGuard() : m_original(mysql_real_data_home) {}

  ~DataHomeGuard() {
    std::snprintf(mysql_real_data_home, FN_REFLEN, "%s", m_original.c_str());
  }

  void Set(const std::string &value) {
    std::snprintf(mysql_real_data_home, FN_REFLEN, "%s", value.c_str());
  }

 private:
  std::string m_original;
};

std::string encode_hex_for_test(const std::string &input) {
  static const char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(input.size() * 2);
  for (unsigned char ch : input) {
    out.push_back(kHex[(ch >> 4) & 0x0F]);
    out.push_back(kHex[ch & 0x0F]);
  }
  return out;
}

std::vector<std::string> current_metadata_fields_for_test() {
  return {
      encode_hex_for_test("idx_v1"),
      "2",
      "cosine",
      "external",
      "faiss",
      "transactional",
      "recovering",
      "9",
      "7",
      "8",
      "5",
      "6",
      encode_hex_for_test("test"),
      encode_hex_for_test("t_vec"),
      encode_hex_for_test("v"),
      encode_hex_for_test("id"),
      "32",
      "16",
      "200",
      "3",
      "64",
      "8",
      "8",
      "6",
      "24",
      "120",
      "240",
      "16",
      "1048576",
      "5",
      "9",
      "offline",
      "1",
      encode_hex_for_test("owner_db")};
}

std::string join_metadata_fields_for_test(
    const std::vector<std::string> &fields) {
  std::string row;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i > 0) row.push_back('\t');
    row.append(fields[i]);
  }
  return row;
}

std::string current_metadata_payload_for_test(
    const std::vector<std::string> &fields) {
  return std::string("VECTOR_INDEX_METADATA_V1\n") +
         join_metadata_fields_for_test(fields) + "\n";
}

std::string encoded_vector_for_test(const vector_index::vector_data &vector) {
  const char *data = reinterpret_cast<const char *>(vector.data());
  return encode_hex_for_test(
      vector.empty() ? std::string()
                     : std::string(data, vector.size() * sizeof(float)));
}

bool deserialize_current_metadata_fields_for_test(
    const std::vector<std::string> &fields,
    std::vector<vector_index_metadata_store::metadata_row> *loaded) {
  return vector_index_metadata_store::deserialize_metadata_rows(
      current_metadata_payload_for_test(fields), loaded);
}

void assign_metadata_owner_for_test(
    std::vector<vector_index_metadata_store::metadata_row> *rows,
    const std::string &owner_schema = "test") {
  ASSERT_NE(nullptr, rows);
  for (auto &row : *rows) row.owner_schema = owner_schema;
}

}  // namespace

class MetadataStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    m_path = std::string(DATA_DIR) + "/vector_metadata_store_test.dat";
    m_committed_path = m_path + ".committed";
    m_manifest_path = m_path + ".manifest";
    m_change_log_path = m_path + ".changelog";
    m_prepared_path = m_path + ".prepared";
    m_segment_task_path = m_path + ".segment_tasks";
    vector_index_metadata_store::set_path_for_testing(m_path);
    std::remove(m_path.c_str());
    std::remove((m_path + ".tmp").c_str());
    std::remove(m_committed_path.c_str());
    std::remove((m_committed_path + ".tmp").c_str());
    std::remove(m_manifest_path.c_str());
    std::remove((m_manifest_path + ".tmp").c_str());
    std::remove(m_change_log_path.c_str());
    std::remove((m_change_log_path + ".tmp").c_str());
    std::remove(m_prepared_path.c_str());
    std::remove((m_prepared_path + ".tmp").c_str());
    std::remove(m_segment_task_path.c_str());
    std::remove((m_segment_task_path + ".tmp").c_str());
    std::filesystem::remove_all(m_path + ".corrupt");
    std::filesystem::remove_all(m_committed_path + ".corrupt");
    std::filesystem::remove_all(m_manifest_path + ".corrupt");
    std::filesystem::remove_all(m_change_log_path + ".corrupt");
    std::filesystem::remove_all(m_prepared_path + ".corrupt");
    std::filesystem::remove_all(m_segment_task_path + ".corrupt");
  }

  void TearDown() override {
    vector_index_metadata_store::reset_path_for_testing();
    std::remove(m_path.c_str());
    std::remove((m_path + ".tmp").c_str());
    std::remove(m_committed_path.c_str());
    std::remove((m_committed_path + ".tmp").c_str());
    std::remove(m_manifest_path.c_str());
    std::remove((m_manifest_path + ".tmp").c_str());
    std::remove(m_change_log_path.c_str());
    std::remove((m_change_log_path + ".tmp").c_str());
    std::remove(m_prepared_path.c_str());
    std::remove((m_prepared_path + ".tmp").c_str());
    std::remove(m_segment_task_path.c_str());
    std::remove((m_segment_task_path + ".tmp").c_str());
    std::filesystem::remove_all(m_path + ".corrupt");
    std::filesystem::remove_all(m_committed_path + ".corrupt");
    std::filesystem::remove_all(m_manifest_path + ".corrupt");
    std::filesystem::remove_all(m_change_log_path + ".corrupt");
    std::filesystem::remove_all(m_prepared_path + ".corrupt");
    std::filesystem::remove_all(m_segment_task_path + ".corrupt");
  }

  std::string m_path;
  std::string m_committed_path;
  std::string m_manifest_path;
  std::string m_change_log_path;
  std::string m_prepared_path;
  std::string m_segment_task_path;
};

TEST_F(MetadataStoreTest, DeserializeApisRejectNullOutputPointers) {
  const std::string metadata_payload = "VECTOR_INDEX_METADATA_V1\n";
  const std::string committed_payload = "mysql-vector-committed-v1\n";
  const std::string manifest_payload = "mysql-vector-manifest-v1\n";
  const std::string changelog_payload = "mysql-vector-changelog-v1\n";
  const std::string prepared_payload = "mysql-vector-prepared-v1\n";
  const std::string segment_task_payload = "mysql-vector-segment-task-v1\n";

  EXPECT_FALSE(
      vector_index_metadata_store::deserialize_metadata_rows(metadata_payload, nullptr));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_committed_rows(
      committed_payload, nullptr));
  EXPECT_FALSE(
      vector_index_metadata_store::deserialize_manifest_row(manifest_payload, nullptr));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_change_log_rows(
      changelog_payload, nullptr));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_prepared_rows(
      prepared_payload, nullptr));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_segment_task_rows(
      segment_task_payload, nullptr));
}

TEST_F(MetadataStoreTest, SerializeApisRejectNullOutputPointers) {
  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  std::vector<vector_index_metadata_store::change_log_row> changelog_rows;
  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  std::vector<vector_index_metadata_store::segment_task_row> segment_task_rows;
  vector_index_metadata_store::manifest_row manifest_row;
  manifest_row.state = "ready";
  manifest_row.version = 1;

  EXPECT_FALSE(
      vector_index_metadata_store::serialize_metadata_rows(metadata_rows, nullptr));
  EXPECT_FALSE(vector_index_metadata_store::serialize_committed_rows(
      committed_rows, nullptr));
  EXPECT_FALSE(
      vector_index_metadata_store::serialize_manifest_row(manifest_row, nullptr));
  EXPECT_FALSE(vector_index_metadata_store::serialize_change_log_rows(
      changelog_rows, nullptr));
  EXPECT_FALSE(vector_index_metadata_store::serialize_prepared_rows(
      prepared_rows, nullptr));
  EXPECT_FALSE(vector_index_metadata_store::serialize_segment_task_rows(
      segment_task_rows, nullptr));
}

TEST_F(MetadataStoreTest, DeserializersSkipBlankLinesAroundRows) {
  const std::string index_hex = encode_hex_for_test("idx_blank");
  const std::string vector_hex =
      encoded_vector_for_test({1.0F, 2.0F});

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  ASSERT_TRUE(vector_index_metadata_store::deserialize_metadata_rows(
      std::string("\nVECTOR_INDEX_METADATA_V1\n\n") +
          join_metadata_fields_for_test(current_metadata_fields_for_test()) +
          "\n\n",
      &metadata_rows));
  ASSERT_EQ(1U, metadata_rows.size());
  EXPECT_EQ("idx_v1", metadata_rows[0].index_name);
  EXPECT_EQ("owner_db", metadata_rows[0].owner_schema);

  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  ASSERT_TRUE(vector_index_metadata_store::deserialize_committed_rows(
      "mysql-vector-committed-v1\n\n" + index_hex + "\t7\t2\t" + vector_hex +
          "\n\n",
      &committed_rows));
  ASSERT_EQ(1U, committed_rows.size());
  EXPECT_EQ(7U, committed_rows[0].doc_id);

  vector_index_metadata_store::manifest_row manifest_row;
  ASSERT_TRUE(vector_index_metadata_store::deserialize_manifest_row(
      "mysql-vector-manifest-v1\n\nready\t2\t3\t4\t5\n\n", &manifest_row));
  EXPECT_EQ(3U, manifest_row.metadata_checkpoint);

  std::vector<vector_index_metadata_store::change_log_row> changelog_rows;
  ASSERT_TRUE(vector_index_metadata_store::deserialize_change_log_rows(
      "mysql-vector-changelog-v1\n\n1\t99\tupsert\t" + index_hex +
          "\t7\t" + vector_hex + "\n\n",
      &changelog_rows));
  ASSERT_EQ(1U, changelog_rows.size());
  EXPECT_EQ(99U, changelog_rows[0].txn_id);

  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  ASSERT_TRUE(vector_index_metadata_store::deserialize_prepared_rows(
      "mysql-vector-prepared-v1\n\n1\t3\t0\t616263\t1\t99\tupsert\t" +
          index_hex + "\t7\t" + vector_hex + "\n\n",
      &prepared_rows));
  ASSERT_EQ(1U, prepared_rows.size());
  EXPECT_TRUE(prepared_rows[0].prepared_in_tc);

  std::vector<vector_index_metadata_store::segment_task_row> segment_task_rows;
  ASSERT_TRUE(vector_index_metadata_store::deserialize_segment_task_rows(
      "mysql-vector-segment-task-v1\n\n" + index_hex +
          "\t1\t2\tpending\t3\t4\t2f766563732e6662696e\t2f646f63732e753634\t"
          "2f6172746966616374\t5\t6\t7\n\n",
      &segment_task_rows));
  ASSERT_EQ(1U, segment_task_rows.size());
  EXPECT_EQ(2U, segment_task_rows[0].segment_id);
}

TEST_F(MetadataStoreTest, LoadApisRejectNullOutputPointers) {
  EXPECT_FALSE(vector_index_metadata_store::load_all(nullptr));
  EXPECT_FALSE(vector_index_metadata_store::load_committed_all(nullptr));
  EXPECT_FALSE(vector_index_metadata_store::load_manifest(nullptr));
  EXPECT_FALSE(vector_index_metadata_store::load_change_log(nullptr));
  EXPECT_FALSE(vector_index_metadata_store::load_prepared(nullptr));
  EXPECT_FALSE(vector_index_metadata_store::load_segment_tasks(nullptr));
}

TEST_F(MetadataStoreTest, DetailHelpersCoverParserAndArtifactPathEdges) {
  namespace detail = vector_index_metadata_store::detail;

  std::vector<std::string> fields;
  ASSERT_TRUE(detail::split_tab_fields("", &fields));
  ASSERT_EQ(1U, fields.size());
  EXPECT_EQ("", fields[0]);
  ASSERT_TRUE(detail::split_tab_fields("a\t\tb\t", &fields));
  ASSERT_EQ(4U, fields.size());
  EXPECT_EQ("a", fields[0]);
  EXPECT_EQ("", fields[1]);
  EXPECT_EQ("b", fields[2]);
  EXPECT_EQ("", fields[3]);

  EXPECT_EQ("", detail::encode_hex(""));
  EXPECT_EQ("00417f", detail::encode_hex(std::string("\0A\177", 3)));

  std::string decoded;
  EXPECT_TRUE(detail::decode_hex("", &decoded));
  EXPECT_TRUE(decoded.empty());
  EXPECT_FALSE(detail::decode_hex("0", &decoded));
  EXPECT_FALSE(detail::decode_hex("gg", &decoded));
  EXPECT_FALSE(detail::decode_hex("0g", &decoded));

  uint64_t value = 0;
  EXPECT_FALSE(detail::parse_uint64("", &value));
  EXPECT_FALSE(detail::parse_uint64("x", &value));
  EXPECT_FALSE(detail::parse_uint64("12x", &value));
  EXPECT_FALSE(detail::parse_uint64("18446744073709551616", &value));
  EXPECT_TRUE(detail::parse_uint64("18446744073709551615", &value));
  EXPECT_EQ(UINT64_MAX, value);

  std::string path;
  EXPECT_FALSE(detail::raw_path_for_artifact("metadata", nullptr));
  EXPECT_FALSE(detail::raw_path_for_artifact("unknown", &path));
  for (const char *artifact_name : {"metadata", "committed", "manifest",
                                    "changelog", "prepared",
                                    "segment_tasks"}) {
    EXPECT_TRUE(detail::raw_path_for_artifact(artifact_name, &path));
    EXPECT_FALSE(path.empty());
  }

  EXPECT_STREQ("pending",
               vector_index_metadata_store::segment_task_state_to_string(
                   vector_index_metadata_store::segment_task_state::kPending));
  EXPECT_STREQ("pending",
               vector_index_metadata_store::segment_task_state_to_string(
                   static_cast<vector_index_metadata_store::segment_task_state>(
                       99)));
  vector_index_metadata_store::segment_task_state task_state =
      vector_index_metadata_store::segment_task_state::kFailed;
  EXPECT_FALSE(
      vector_index_metadata_store::parse_segment_task_state("pending", nullptr));
  EXPECT_TRUE(vector_index_metadata_store::parse_segment_task_state(
      "pending", &task_state));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kPending,
            task_state);
  EXPECT_TRUE(vector_index_metadata_store::parse_segment_task_state(
      "building", &task_state));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kBuilding,
            task_state);
  EXPECT_TRUE(vector_index_metadata_store::parse_segment_task_state(
      "ready", &task_state));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kReady,
            task_state);
  EXPECT_TRUE(vector_index_metadata_store::parse_segment_task_state(
      "failed", &task_state));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kFailed,
            task_state);
  EXPECT_TRUE(vector_index_metadata_store::parse_segment_task_state(
      "abandoned", &task_state));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kAbandoned,
            task_state);
  EXPECT_FALSE(vector_index_metadata_store::parse_segment_task_state(
      "done", &task_state));

  EXPECT_STREQ("upsert",
               detail::change_op_to_string(
                   vector_index_metadata_store::change_op::kUpsert));
  EXPECT_STREQ("erase",
               detail::change_op_to_string(
                   vector_index_metadata_store::change_op::kErase));
  EXPECT_STREQ(
      "upsert",
      detail::change_op_to_string(
          static_cast<vector_index_metadata_store::change_op>(99)));

  vector_index_metadata_store::change_op op =
      vector_index_metadata_store::change_op::kErase;
  EXPECT_FALSE(detail::parse_change_op("upsert", nullptr));
  EXPECT_TRUE(detail::parse_change_op("upsert", &op));
  EXPECT_EQ(vector_index_metadata_store::change_op::kUpsert, op);
  EXPECT_TRUE(detail::parse_change_op("erase", &op));
  EXPECT_EQ(vector_index_metadata_store::change_op::kErase, op);
  EXPECT_FALSE(detail::parse_change_op("delete", &op));

  std::ifstream file;
  EXPECT_FALSE(detail::open_read_primary(m_path, nullptr));
  EXPECT_FALSE(detail::open_read_primary(m_path, &file));

  {
    std::ofstream primary_file(
        m_committed_path, std::ios::out | std::ios::binary | std::ios::trunc);
    primary_file << "primary payload";
  }
  ASSERT_TRUE(detail::open_read_primary(m_committed_path, &file));
  std::string payload((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  EXPECT_EQ("primary payload", payload);
  file.close();

  EXPECT_TRUE(detail::remove_if_exists(m_path + ".missing"));
}

TEST_F(MetadataStoreTest, SegmentTaskRowsRoundTripThroughCodecAndStore) {
  std::vector<vector_index_metadata_store::segment_task_row> rows;
  vector_index_metadata_store::segment_task_row first;
  first.index_name = "idx_segment";
  first.generation = 3;
  first.segment_id = 7;
  first.state = vector_index_metadata_store::segment_task_state::kBuilding;
  first.row_count = 100;
  first.payload_size = 800;
  first.vector_path = "/tmp/vector.fbin";
  first.docid_path = "/tmp/vector.u64";
  first.artifact_prefix = "/tmp/artifact/seg_7";
  first.attempt = 2;
  first.last_error_code = 9;
  first.updated_ts = 12345;
  rows.push_back(first);

  vector_index_metadata_store::segment_task_row second;
  second.index_name = "idx_segment";
  second.generation = 3;
  second.segment_id = 8;
  second.state = vector_index_metadata_store::segment_task_state::kReady;
  second.row_count = 200;
  second.payload_size = 1600;
  second.vector_path = "/tmp/vector2.fbin";
  second.docid_path = "/tmp/vector2.u64";
  second.artifact_prefix = "/tmp/artifact/seg_8";
  second.attempt = 1;
  second.updated_ts = 12346;
  rows.push_back(second);

  std::string payload;
  ASSERT_TRUE(vector_index_metadata_store::serialize_segment_task_rows(
      rows, &payload));
  std::vector<vector_index_metadata_store::segment_task_row> decoded;
  ASSERT_TRUE(vector_index_metadata_store::deserialize_segment_task_rows(
      payload, &decoded));
  ASSERT_EQ(2U, decoded.size());
  EXPECT_EQ("idx_segment", decoded[0].index_name);
  EXPECT_EQ(3U, decoded[0].generation);
  EXPECT_EQ(7U, decoded[0].segment_id);
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kBuilding,
            decoded[0].state);
  EXPECT_EQ(100U, decoded[0].row_count);
  EXPECT_EQ(800U, decoded[0].payload_size);
  EXPECT_EQ("/tmp/vector.fbin", decoded[0].vector_path);
  EXPECT_EQ("/tmp/vector.u64", decoded[0].docid_path);
  EXPECT_EQ("/tmp/artifact/seg_7", decoded[0].artifact_prefix);
  EXPECT_EQ(2U, decoded[0].attempt);
  EXPECT_EQ(9U, decoded[0].last_error_code);
  EXPECT_EQ(12345U, decoded[0].updated_ts);

  ASSERT_TRUE(vector_index_metadata_store::save_segment_tasks(rows));
  decoded.clear();
  ASSERT_TRUE(vector_index_metadata_store::load_segment_tasks(&decoded));
  ASSERT_EQ(2U, decoded.size());
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kReady,
            decoded[1].state);
  EXPECT_EQ(8U, decoded[1].segment_id);

  ASSERT_TRUE(vector_index_metadata_store::save_segment_tasks({}));
  ASSERT_TRUE(vector_index_metadata_store::load_segment_tasks(&decoded));
  EXPECT_TRUE(decoded.empty());
}

TEST_F(MetadataStoreTest, SegmentTaskRowsRejectMalformedPayloads) {
  std::vector<vector_index_metadata_store::segment_task_row> rows;
  EXPECT_TRUE(vector_index_metadata_store::deserialize_segment_task_rows(
      "", &rows));
  EXPECT_TRUE(rows.empty());
  EXPECT_FALSE(vector_index_metadata_store::deserialize_segment_task_rows(
      "bad-header\n", &rows));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_segment_task_rows(
      "mysql-vector-segment-task-v1\n" + encode_hex_for_test("idx") +
          "\t1\t2\tunknown\t3\t4\t\t\t\t0\t0\t0\n",
      &rows));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_segment_task_rows(
      "mysql-vector-segment-task-v1\n" + encode_hex_for_test("idx") +
          "\t0\t2\tpending\t3\t4\t\t\t\t0\t0\t0\n",
      &rows));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_segment_task_rows(
      "mysql-vector-segment-task-v1\n\t1\t2\tpending\t3\t4\t\t\t\t0\t0\t0\n",
      &rows));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_segment_task_rows(
      "mysql-vector-segment-task-v1\n" + encode_hex_for_test("idx") +
          "\t1\t2\tpending\t3\t4\t\t\t\t4294967296\t0\t0\n",
      &rows));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_segment_task_rows(
      "mysql-vector-segment-task-v1\n" + encode_hex_for_test("idx") +
          "\t1\t2\tpending\t3\t4\t\t\t\t0\t4294967296\t0\n",
      &rows));

  std::vector<vector_index_metadata_store::segment_task_row> invalid_rows(1);
  invalid_rows[0].index_name = "idx_invalid";
  invalid_rows[0].generation = 0;
  std::string payload;
  EXPECT_FALSE(vector_index_metadata_store::serialize_segment_task_rows(
      invalid_rows, &payload));
  invalid_rows[0].generation = 1;
  invalid_rows[0].index_name.clear();
  EXPECT_FALSE(vector_index_metadata_store::serialize_segment_task_rows(
      invalid_rows, &payload));
}

TEST_F(MetadataStoreTest, SegmentTaskRawArtifactAndQuarantine) {
  const std::string payload = "mysql-vector-segment-task-v1\n";
  bool found = true;
  std::string loaded;

  ASSERT_TRUE(vector_index_metadata_store::load_raw_artifact(
      "segment_tasks", &loaded, &found));
  EXPECT_FALSE(found);
  ASSERT_TRUE(vector_index_metadata_store::save_raw_artifact(
      "segment_tasks", payload));
  ASSERT_TRUE(vector_index_metadata_store::load_raw_artifact(
      "segment_tasks", &loaded, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(payload, loaded);

  ASSERT_TRUE(vector_index_metadata_store::quarantine_segment_task_store());
  EXPECT_TRUE(std::filesystem::exists(m_segment_task_path + ".corrupt"));
  found = true;
  ASSERT_TRUE(vector_index_metadata_store::load_raw_artifact(
      "segment_tasks", &loaded, &found));
  EXPECT_FALSE(found);
}

TEST_F(MetadataStoreTest, DefaultDataHomeStoreRoundTripWithoutOverride) {
  DataHomeGuard data_home_guard;
  std::error_code ec;
  const std::string data_home =
      std::string(testing::TempDir()) + "/vector_metadata_default_home_t";
  std::filesystem::remove_all(data_home, ec);
  std::filesystem::create_directories(data_home, ec);
  ASSERT_FALSE(ec);

  vector_index_metadata_store::reset_path_for_testing();
  data_home_guard.Set(data_home);

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows{
      {"idx_default_home",
       2,
       vector_index::metric_type::kEuclidean,
       vector_index::backend_mode::kMemory,
       vector_index::backend_provider::kNative,
       vector_index::index_consistency_mode::kTransactional,
       "",
       "",
       "",
       "ready",
       1,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       vector_index::diskann_build_mode::kAuto,
       ""}};
  assign_metadata_owner_for_test(&metadata_rows);
  ASSERT_TRUE(vector_index_metadata_store::save_all(metadata_rows));
  std::vector<vector_index_metadata_store::metadata_row> loaded_metadata;
  ASSERT_TRUE(vector_index_metadata_store::load_all(&loaded_metadata));
  ASSERT_EQ(1U, loaded_metadata.size());

  std::vector<vector_index_metadata_store::committed_row> committed_rows{
      {"idx_default_home", 5, {5.0F, 6.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_committed_all(committed_rows));
  std::vector<vector_index_metadata_store::committed_row> loaded_committed;
  ASSERT_TRUE(vector_index_metadata_store::load_committed_all(&loaded_committed));
  ASSERT_EQ(1U, loaded_committed.size());

  vector_index_metadata_store::manifest_row manifest_row;
  manifest_row.state = "ready";
  manifest_row.version = 9;
  manifest_row.metadata_checkpoint = 3;
  manifest_row.committed_checkpoint = 4;
  manifest_row.change_log_checkpoint = 5;
  ASSERT_TRUE(vector_index_metadata_store::save_manifest(manifest_row));
  vector_index_metadata_store::manifest_row loaded_manifest;
  ASSERT_TRUE(vector_index_metadata_store::load_manifest(&loaded_manifest));
  EXPECT_EQ(9U, loaded_manifest.version);

  std::vector<vector_index_metadata_store::change_log_row> change_log_rows{
      {11,
       2,
       vector_index_metadata_store::change_op::kUpsert,
       "idx_default_home",
       7,
       {7.0F, 8.0F}},
      {12, 2, vector_index_metadata_store::change_op::kErase,
       "idx_default_home", 7, {}}};
  ASSERT_TRUE(vector_index_metadata_store::save_change_log(change_log_rows));
  std::vector<vector_index_metadata_store::change_log_row> loaded_change_log;
  ASSERT_TRUE(vector_index_metadata_store::load_change_log(&loaded_change_log));
  ASSERT_EQ(2U, loaded_change_log.size());

  std::filesystem::remove_all(data_home, ec);
  vector_index_metadata_store::set_path_for_testing(m_path);
}

TEST_F(MetadataStoreTest, DefaultDataHomePathSupportsTrailingSlash) {
  DataHomeGuard data_home_guard;
  std::error_code ec;
  const std::string data_home =
      std::string(testing::TempDir()) + "/vector_metadata_default_home_slash_t";
  std::filesystem::remove_all(data_home, ec);
  std::filesystem::create_directories(data_home, ec);
  ASSERT_FALSE(ec);

  vector_index_metadata_store::reset_path_for_testing();
  data_home_guard.Set(data_home + "/");

  std::vector<vector_index_metadata_store::metadata_row> loaded_metadata;
  ASSERT_TRUE(vector_index_metadata_store::load_all(&loaded_metadata));
  EXPECT_TRUE(loaded_metadata.empty());

  std::vector<vector_index_metadata_store::committed_row> loaded_committed;
  ASSERT_TRUE(vector_index_metadata_store::load_committed_all(&loaded_committed));
  EXPECT_TRUE(loaded_committed.empty());

  vector_index_metadata_store::manifest_row loaded_manifest;
  ASSERT_TRUE(vector_index_metadata_store::load_manifest(&loaded_manifest));
  EXPECT_EQ(1U, loaded_manifest.version);

  std::vector<vector_index_metadata_store::change_log_row> loaded_change_log;
  ASSERT_TRUE(vector_index_metadata_store::load_change_log(&loaded_change_log));
  EXPECT_TRUE(loaded_change_log.empty());

  std::filesystem::remove_all(data_home, ec);
  vector_index_metadata_store::set_path_for_testing(m_path);
}

TEST_F(MetadataStoreTest, DefaultDataHomeEmptyUsesRelativeStoreDirectory) {
  DataHomeGuard data_home_guard;
  data_home_guard.Set("");
  vector_index_metadata_store::reset_path_for_testing();
  std::error_code ec;
  std::filesystem::remove_all("mysql_vector_index", ec);
  std::remove("mysql_vector_index_metadata.v1");
  std::remove("mysql_vector_index_committed.v1");
  std::remove("mysql_vector_index_manifest.v1");
  std::remove("mysql_vector_index_changelog.v1");
  std::remove("mysql_vector_index_prepared.v1");

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows{
      {"idx_empty_home",
       2,
       vector_index::metric_type::kEuclidean,
       vector_index::backend_mode::kMemory,
       vector_index::backend_provider::kNative,
       vector_index::index_consistency_mode::kTransactional,
       "",
       "",
       "",
       "ready",
       1,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       vector_index::diskann_build_mode::kAuto,
       ""}};
  assign_metadata_owner_for_test(&metadata_rows);
  ASSERT_TRUE(vector_index_metadata_store::save_all(metadata_rows));

  std::vector<vector_index_metadata_store::committed_row> committed_rows{
      {"idx_empty_home", 1, {1.0F, 2.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_committed_all(committed_rows));

  vector_index_metadata_store::manifest_row manifest_row;
  manifest_row.state = "ready";
  manifest_row.version = 3;
  manifest_row.metadata_checkpoint = 1;
  manifest_row.committed_checkpoint = 2;
  manifest_row.change_log_checkpoint = 0;
  ASSERT_TRUE(vector_index_metadata_store::save_manifest(manifest_row));

  std::vector<vector_index_metadata_store::change_log_row> changelog_rows{
      {1, 1, vector_index_metadata_store::change_op::kUpsert,
       "idx_empty_home", 1, {1.0F, 2.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_change_log(changelog_rows));

  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows{
      {1, 3, 0, "abc", false, 1,
       vector_index_metadata_store::change_op::kUpsert, "idx_empty_home", 1,
       {1.0F, 2.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_prepared(prepared_rows));

  std::ifstream metadata_file("mysql_vector_index/metadata.v1");
  std::ifstream committed_file("mysql_vector_index/committed.v1");
  std::ifstream manifest_file("mysql_vector_index/manifest.v1");
  std::ifstream changelog_file("mysql_vector_index/changelog.v1");
  std::ifstream prepared_file("mysql_vector_index/prepared.v1");
  ASSERT_TRUE(metadata_file.good());
  ASSERT_TRUE(committed_file.good());
  ASSERT_TRUE(manifest_file.good());
  ASSERT_TRUE(changelog_file.good());
  ASSERT_TRUE(prepared_file.good());

  std::filesystem::remove_all("mysql_vector_index", ec);
  vector_index_metadata_store::set_path_for_testing(m_path);
}

TEST_F(MetadataStoreTest, RelativePathRoundTripUsesEmptyParentBranch) {
  const std::string relative_path = "vector_metadata_relative_roundtrip_t.dat";
  vector_index_metadata_store::set_path_for_testing(relative_path);
  std::remove(relative_path.c_str());
  std::remove((relative_path + ".tmp").c_str());
  std::remove((relative_path + ".committed").c_str());
  std::remove((relative_path + ".committed.tmp").c_str());
  std::remove((relative_path + ".manifest").c_str());
  std::remove((relative_path + ".manifest.tmp").c_str());
  std::remove((relative_path + ".changelog").c_str());
  std::remove((relative_path + ".changelog.tmp").c_str());
  std::remove((relative_path + ".prepared").c_str());
  std::remove((relative_path + ".prepared.tmp").c_str());

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows{
      {"idx_rel",
       2,
       vector_index::metric_type::kEuclidean,
       vector_index::backend_mode::kMemory,
       vector_index::backend_provider::kNative,
       vector_index::index_consistency_mode::kTransactional,
       "",
       "",
       "",
       "ready",
       1,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       vector_index::diskann_build_mode::kAuto,
       ""}};
  assign_metadata_owner_for_test(&metadata_rows);
  ASSERT_TRUE(vector_index_metadata_store::save_all(metadata_rows));
  std::vector<vector_index_metadata_store::metadata_row> loaded_metadata;
  ASSERT_TRUE(vector_index_metadata_store::load_all(&loaded_metadata));
  ASSERT_EQ(1U, loaded_metadata.size());

  std::vector<vector_index_metadata_store::committed_row> committed_rows{
      {"idx_rel", 7, {1.0F, 2.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_committed_all(committed_rows));
  std::vector<vector_index_metadata_store::committed_row> loaded_committed;
  ASSERT_TRUE(vector_index_metadata_store::load_committed_all(&loaded_committed));
  ASSERT_EQ(1U, loaded_committed.size());

  vector_index_metadata_store::manifest_row manifest_row;
  manifest_row.state = "ready";
  manifest_row.version = 9;
  manifest_row.metadata_checkpoint = 1;
  manifest_row.committed_checkpoint = 2;
  manifest_row.change_log_checkpoint = 3;
  ASSERT_TRUE(vector_index_metadata_store::save_manifest(manifest_row));
  vector_index_metadata_store::manifest_row loaded_manifest;
  ASSERT_TRUE(vector_index_metadata_store::load_manifest(&loaded_manifest));
  EXPECT_EQ(9U, loaded_manifest.version);

  std::vector<vector_index_metadata_store::change_log_row> changelog_rows{
      {1, 11, vector_index_metadata_store::change_op::kUpsert, "idx_rel", 7,
       {1.0F, 2.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_change_log(changelog_rows));
  std::vector<vector_index_metadata_store::change_log_row> loaded_changelog;
  ASSERT_TRUE(vector_index_metadata_store::load_change_log(&loaded_changelog));
  ASSERT_EQ(1U, loaded_changelog.size());

  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows{
      {1, 3, 0, "abc", false, 99,
       vector_index_metadata_store::change_op::kUpsert, "idx_rel", 7,
       {1.0F, 2.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_prepared(prepared_rows));
  std::vector<vector_index_metadata_store::prepared_change_row> loaded_prepared;
  ASSERT_TRUE(vector_index_metadata_store::load_prepared(&loaded_prepared));
  ASSERT_EQ(1U, loaded_prepared.size());

  ASSERT_TRUE(vector_index_metadata_store::save_raw_artifact("manifest", "payload"));
  std::string raw_payload;
  bool found = false;
  ASSERT_TRUE(
      vector_index_metadata_store::load_raw_artifact("manifest", &raw_payload, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ("payload", raw_payload);

  vector_index_metadata_store::set_path_for_testing(m_path);
  std::remove(relative_path.c_str());
  std::remove((relative_path + ".tmp").c_str());
  std::remove((relative_path + ".committed").c_str());
  std::remove((relative_path + ".committed.tmp").c_str());
  std::remove((relative_path + ".manifest").c_str());
  std::remove((relative_path + ".manifest.tmp").c_str());
  std::remove((relative_path + ".changelog").c_str());
  std::remove((relative_path + ".changelog.tmp").c_str());
  std::remove((relative_path + ".prepared").c_str());
  std::remove((relative_path + ".prepared.tmp").c_str());
}

TEST_F(MetadataStoreTest, SaveThenLoadRoundTrip) {
  std::vector<vector_index_metadata_store::metadata_row> rows{
      {"idx_mem",
       3,
       vector_index::metric_type::kEuclidean,
       vector_index::backend_mode::kMemory,
       vector_index::backend_provider::kNative,
       vector_index::index_consistency_mode::kTransactional,
       "",
       "",
       "",
       "ready",
       1,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       vector_index::diskann_build_mode::kAuto,
       ""},
      {"idx_ext",
       4,
       vector_index::metric_type::kCosine,
       vector_index::backend_mode::kExternal,
       vector_index::backend_provider::kFaiss,
       vector_index::index_consistency_mode::kTransactional,
       "test",
       "t_ext",
       "v",
       "recovering",
       7,
       1001,
       123456789ULL,
       5,
       987654321ULL,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       vector_index::diskann_build_mode::kAuto,
       ""}};

  assign_metadata_owner_for_test(&rows);
  ASSERT_TRUE(vector_index_metadata_store::save_all(rows));

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  ASSERT_TRUE(vector_index_metadata_store::load_all(&loaded));
  ASSERT_EQ(2U, loaded.size());

  EXPECT_EQ("idx_mem", loaded[0].index_name);
  EXPECT_EQ(3U, loaded[0].dimension);
  EXPECT_EQ(vector_index::metric_type::kEuclidean, loaded[0].metric);
  EXPECT_EQ(vector_index::backend_mode::kMemory, loaded[0].mode);
  EXPECT_EQ(vector_index::backend_provider::kNative, loaded[0].provider);
  EXPECT_EQ("ready", loaded[0].lifecycle_state);
  EXPECT_EQ(1U, loaded[0].lifecycle_version);
  EXPECT_EQ(0U, loaded[0].last_error_code);
  EXPECT_EQ(0U, loaded[0].last_error_ts);
  EXPECT_EQ(0U, loaded[0].recover_fallback_count);
  EXPECT_EQ(0U, loaded[0].last_recover_fallback_ts);
  EXPECT_TRUE(loaded[0].schema_name.empty());
  EXPECT_TRUE(loaded[0].table_name.empty());
  EXPECT_TRUE(loaded[0].column_name.empty());

  EXPECT_EQ("idx_ext", loaded[1].index_name);
  EXPECT_EQ(4U, loaded[1].dimension);
  EXPECT_EQ(vector_index::metric_type::kCosine, loaded[1].metric);
  EXPECT_EQ(vector_index::backend_mode::kExternal, loaded[1].mode);
  EXPECT_EQ(vector_index::backend_provider::kFaiss, loaded[1].provider);
  EXPECT_EQ("recovering", loaded[1].lifecycle_state);
  EXPECT_EQ(7U, loaded[1].lifecycle_version);
  EXPECT_EQ(1001U, loaded[1].last_error_code);
  EXPECT_EQ(123456789ULL, loaded[1].last_error_ts);
  EXPECT_EQ(5U, loaded[1].recover_fallback_count);
  EXPECT_EQ(987654321ULL, loaded[1].last_recover_fallback_ts);
  EXPECT_EQ("test", loaded[1].schema_name);
  EXPECT_EQ("t_ext", loaded[1].table_name);
  EXPECT_EQ("v", loaded[1].column_name);
}

TEST_F(MetadataStoreTest, MetadataSerializationRoundTrip) {
  std::vector<vector_index_metadata_store::metadata_row> rows{
      {"idx_mem",
       3,
       vector_index::metric_type::kEuclidean,
       vector_index::backend_mode::kMemory,
       vector_index::backend_provider::kNative,
       vector_index::index_consistency_mode::kTransactional,
       "",
       "",
       "",
       "ready",
       1,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       vector_index::diskann_build_mode::kAuto,
       ""},
      {"test.t_vec.v",
       4,
       vector_index::metric_type::kCosine,
       vector_index::backend_mode::kExternal,
       vector_index::backend_provider::kFaiss,
       vector_index::index_consistency_mode::kTransactional,
       "test",
       "t_vec",
       "v",
       "recovering",
       7,
       1001,
       123456789ULL,
       5,
       987654321ULL,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       vector_index::diskann_build_mode::kAuto,
       ""}};

  assign_metadata_owner_for_test(&rows);
  rows[1].hnsw_build_threads = 3;
  rows[1].faiss_build_threads = 4;
  rows[1].diskann_build_threads = 2;
  rows[1].diskann_build_mode_value =
      vector_index::diskann_build_mode::kOffline;
  rows[1].diskann_build_mode_specified = true;
  rows[1].diskann_pq_code_budget_size = 1048576;
  std::string payload;
  ASSERT_TRUE(
      vector_index_metadata_store::serialize_metadata_rows(rows, &payload));

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  ASSERT_TRUE(
      vector_index_metadata_store::deserialize_metadata_rows(payload, &loaded));
  ASSERT_EQ(rows.size(), loaded.size());
  EXPECT_EQ(rows[0].index_name, loaded[0].index_name);
  EXPECT_EQ(rows[1].schema_name, loaded[1].schema_name);
  EXPECT_EQ(rows[1].table_name, loaded[1].table_name);
  EXPECT_EQ(rows[1].column_name, loaded[1].column_name);
  EXPECT_EQ(3U, loaded[1].hnsw_build_threads);
  EXPECT_EQ(4U, loaded[1].faiss_build_threads);
  EXPECT_EQ(2U, loaded[1].diskann_build_threads);
  EXPECT_EQ(vector_index::diskann_build_mode::kOffline,
            loaded[1].diskann_build_mode_value);
  EXPECT_TRUE(loaded[1].diskann_build_mode_specified);
  EXPECT_EQ(1048576U, loaded[1].diskann_pq_code_budget_size);
}

TEST_F(MetadataStoreTest, ChangeLogSerializationRoundTrip) {
  std::vector<vector_index_metadata_store::change_log_row> rows{
      {1,
       99,
       vector_index_metadata_store::change_op::kUpsert,
       "idx_truth",
       11,
       {1.0F, 2.0F}},
      {2, 99, vector_index_metadata_store::change_op::kErase, "idx_truth", 11,
       {}}};

  std::string payload;
  ASSERT_TRUE(
      vector_index_metadata_store::serialize_change_log_rows(rows, &payload));

  std::vector<vector_index_metadata_store::change_log_row> loaded;
  ASSERT_TRUE(
      vector_index_metadata_store::deserialize_change_log_rows(payload, &loaded));
  ASSERT_EQ(rows.size(), loaded.size());
  EXPECT_EQ(rows[0].sequence, loaded[0].sequence);
  EXPECT_EQ(rows[0].vector, loaded[0].vector);
  EXPECT_EQ(rows[1].op, loaded[1].op);
  EXPECT_TRUE(loaded[1].vector.empty());
}

TEST_F(MetadataStoreTest, LoadMissingFileReturnsEmptyRows) {
  std::vector<vector_index_metadata_store::metadata_row> loaded;
  ASSERT_TRUE(vector_index_metadata_store::load_all(&loaded));
  EXPECT_TRUE(loaded.empty());
}

TEST_F(MetadataStoreTest, LoadRejectsCorruptedRow) {
  std::ofstream file(m_path, std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "VECTOR_INDEX_METADATA_V1\n";
  file << "bad\trow\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_all(&loaded));
}

TEST_F(MetadataStoreTest, LoadCurrentMetadataRowReadsAllFields) {
  std::vector<vector_index_metadata_store::metadata_row> loaded;
  ASSERT_TRUE(deserialize_current_metadata_fields_for_test(
      current_metadata_fields_for_test(), &loaded));
  ASSERT_EQ(1U, loaded.size());
  EXPECT_EQ("idx_v1", loaded[0].index_name);
  EXPECT_EQ(2U, loaded[0].dimension);
  EXPECT_EQ(vector_index::metric_type::kCosine, loaded[0].metric);
  EXPECT_EQ(vector_index::backend_mode::kExternal, loaded[0].mode);
  EXPECT_EQ(vector_index::backend_provider::kFaiss, loaded[0].provider);
  EXPECT_EQ("recovering", loaded[0].lifecycle_state);
  EXPECT_EQ(9U, loaded[0].lifecycle_version);
  EXPECT_EQ(7U, loaded[0].last_error_code);
  EXPECT_EQ(8U, loaded[0].last_error_ts);
  EXPECT_EQ(5U, loaded[0].recover_fallback_count);
  EXPECT_EQ(6U, loaded[0].last_recover_fallback_ts);
  EXPECT_EQ("test", loaded[0].schema_name);
  EXPECT_EQ("t_vec", loaded[0].table_name);
  EXPECT_EQ("v", loaded[0].column_name);
  EXPECT_EQ("id", loaded[0].doc_id_column_name);
  EXPECT_EQ(32U, loaded[0].search_ef);
  EXPECT_EQ(16U, loaded[0].hnsw_m);
  EXPECT_EQ(200U, loaded[0].hnsw_ef_construction);
  EXPECT_EQ(3U, loaded[0].hnsw_build_threads);
  EXPECT_EQ(64U, loaded[0].faiss_nlist);
  EXPECT_EQ(8U, loaded[0].faiss_nprobe);
  EXPECT_EQ(8U, loaded[0].faiss_pq_m);
  EXPECT_EQ(6U, loaded[0].faiss_pq_bits);
  EXPECT_EQ(24U, loaded[0].diskann_max_degree);
  EXPECT_EQ(120U, loaded[0].diskann_build_complexity);
  EXPECT_EQ(240U, loaded[0].diskann_search_complexity);
  EXPECT_EQ(16U, loaded[0].diskann_search_beamwidth);
  EXPECT_EQ(5U, loaded[0].diskann_build_threads);
  EXPECT_EQ(9U, loaded[0].faiss_build_threads);
  EXPECT_EQ(vector_index::diskann_build_mode::kOffline,
            loaded[0].diskann_build_mode_value);
  EXPECT_TRUE(loaded[0].diskann_build_mode_specified);
}

TEST_F(MetadataStoreTest, LoadRejectsStaleDevelopmentMetadataHeaders) {
  const char *headers[] = {"mysql-vector-metadata-v1",
                           "mysql-vector-metadata-v2",
                           "mysql-vector-metadata-v3",
                           "mysql-vector-metadata-v4",
                           "VECTOR_INDEX_METADATA_V2",
                           "VECTOR_INDEX_METADATA_V3",
                           "VECTOR_INDEX_METADATA_V4",
                           "VECTOR_INDEX_METADATA_V5",
                           "VECTOR_INDEX_METADATA_V6",
                           "VECTOR_INDEX_METADATA_V7",
                           "VECTOR_INDEX_METADATA_V8",
                           "VECTOR_INDEX_METADATA_V9",
                           "VECTOR_INDEX_METADATA_V10",
                           "VECTOR_INDEX_METADATA_V11",
                           "VECTOR_INDEX_METADATA_V12",
                           "VECTOR_INDEX_METADATA_V13"};

  const std::string row =
      join_metadata_fields_for_test(current_metadata_fields_for_test()) + "\n";
  for (const char *header : headers) {
    std::vector<vector_index_metadata_store::metadata_row> loaded;
    EXPECT_FALSE(vector_index_metadata_store::deserialize_metadata_rows(
        std::string(header) + "\n" + row, &loaded))
        << header;
  }
}

TEST_F(MetadataStoreTest, SerializeCurrentMetadataKeepsInitialHeader) {
  std::vector<vector_index_metadata_store::metadata_row> rows{
      {"idx",
       3,
       vector_index::metric_type::kEuclidean,
       vector_index::backend_mode::kMemory,
       vector_index::backend_provider::kNative,
       vector_index::index_consistency_mode::kTransactional,
       "test",
       "t_vec",
       "v",
       "ready",
       1,
       0,
       0,
       0,
       0,
       32,
       16,
       200,
       64,
       8,
       8,
       6,
       24,
       120,
       240,
       16,
       vector_index::diskann_build_mode::kAuto,
       "id",
       3,
       5,
       9}};

  assign_metadata_owner_for_test(&rows);
  std::string payload;
  ASSERT_TRUE(
      vector_index_metadata_store::serialize_metadata_rows(rows, &payload));
  EXPECT_EQ(0U, payload.find("VECTOR_INDEX_METADATA_V1\n"));
}

TEST_F(MetadataStoreTest, LoadRejectsCurrentMetadataWrongFieldCount) {
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields.pop_back();

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
}

TEST_F(MetadataStoreTest, MetadataRejectsEmptyOwnerSchema) {
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields.back().clear();

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));

  vector_index_metadata_store::metadata_row row;
  row.index_name = "idx_missing_owner";
  row.dimension = 2;
  std::string payload;
  EXPECT_FALSE(
      vector_index_metadata_store::serialize_metadata_rows({row}, &payload));
}

TEST_F(MetadataStoreTest,
       MetadataOwnerSchemaRoundTripsIndependentlyOfBinding) {
  vector_index_metadata_store::metadata_row row;
  row.index_name = "idx_owner";
  row.dimension = 2;
  row.owner_schema = "owner_db";

  std::string payload;
  ASSERT_TRUE(
      vector_index_metadata_store::serialize_metadata_rows({row}, &payload));

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  ASSERT_TRUE(
      vector_index_metadata_store::deserialize_metadata_rows(payload, &loaded));
  ASSERT_EQ(1U, loaded.size());
  EXPECT_EQ("owner_db", loaded[0].owner_schema);
  EXPECT_TRUE(loaded[0].schema_name.empty());
  EXPECT_TRUE(loaded[0].table_name.empty());
  EXPECT_TRUE(loaded[0].column_name.empty());
}

TEST_F(MetadataStoreTest, LoadRejectsInvalidBackendModeInCurrentRow) {
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields[3] = "disk";

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
}

TEST_F(MetadataStoreTest, LoadRejectsInvalidProviderInCurrentRow) {
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields[4] = "nmslib";

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
}

TEST_F(MetadataStoreTest, LoadRejectsInvalidDiskAnnBuildModeInCurrentRow) {
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields[31] = "background";

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
}

TEST_F(MetadataStoreTest, LoadRejectsInvalidDiskAnnBuildModeSpecifiedFlag) {
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields[32] = "true";

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
}

TEST_F(MetadataStoreTest, LoadRejectsInvalidIndexHexInCurrentRow) {
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields[0] = "zz";

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
}

TEST_F(MetadataStoreTest, LoadRejectsEmptyIndexNameInCurrentRow) {
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields[0].clear();

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
}

TEST_F(MetadataStoreTest, LoadRejectsInvalidDimensionAndMetricInCurrentRow) {
  std::vector<vector_index_metadata_store::metadata_row> loaded;
  {
    std::vector<std::string> fields = current_metadata_fields_for_test();
    fields[1] = "0";
    EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
  }
  {
    std::vector<std::string> fields = current_metadata_fields_for_test();
    fields[1] = "not_a_number";
    EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
  }
  {
    std::vector<std::string> fields = current_metadata_fields_for_test();
    fields[2] = "manhattan";
    EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
  }
}

TEST_F(MetadataStoreTest, LoadRejectsEmptyLifecycleState) {
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields[5].clear();

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
}

TEST_F(MetadataStoreTest, LoadRejectsZeroLifecycleVersion) {
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields[7] = "0";

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
}

TEST_F(MetadataStoreTest, LoadRejectsInvalidSchemaHexInCurrentRow) {
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields[12] = "zz";

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
}

TEST_F(MetadataStoreTest, LoadRejectsInvalidTableAndColumnHexInCurrentRow) {
  std::vector<vector_index_metadata_store::metadata_row> loaded;
  {
    std::vector<std::string> fields = current_metadata_fields_for_test();
    fields[13] = "zz";
    EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
  }
  {
    std::vector<std::string> fields = current_metadata_fields_for_test();
    fields[14] = "zz";
    EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
  }
}

TEST_F(MetadataStoreTest, LoadRejectsInvalidDocIdColumnHexInCurrentRow) {
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields[15] = "zz";

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded));
}

TEST_F(MetadataStoreTest, LoadRejectsOverflowCurrentMetadataFields) {
  const size_t uint32_overflow_fields[] = {8,  16, 17, 18, 19, 20, 21,
                                           22, 23, 24, 25, 26, 27, 29,
                                           30};
  for (size_t field_index : uint32_overflow_fields) {
    std::vector<std::string> fields = current_metadata_fields_for_test();
    fields[field_index] = "4294967296";

    std::vector<vector_index_metadata_store::metadata_row> loaded;
    EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded))
        << field_index;
  }

  const size_t uint64_overflow_fields[] = {7, 9, 10, 11, 28};
  for (size_t field_index : uint64_overflow_fields) {
    std::vector<std::string> fields = current_metadata_fields_for_test();
    fields[field_index] = "18446744073709551616";

    std::vector<vector_index_metadata_store::metadata_row> loaded;
    EXPECT_FALSE(deserialize_current_metadata_fields_for_test(fields, &loaded))
        << field_index;
  }
}

TEST_F(MetadataStoreTest, SaveEmptyRowsRemovesStoreFile) {
  std::vector<vector_index_metadata_store::metadata_row> rows{
      {"idx_mem",
       3,
       vector_index::metric_type::kEuclidean,
       vector_index::backend_mode::kMemory,
       vector_index::backend_provider::kNative,
       vector_index::index_consistency_mode::kTransactional,
       "",
       "",
       "",
       "ready",
       1,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       vector_index::diskann_build_mode::kAuto,
       ""}};
  assign_metadata_owner_for_test(&rows);
  ASSERT_TRUE(vector_index_metadata_store::save_all(rows));

  std::ifstream exists_before(m_path);
  ASSERT_TRUE(exists_before.good());
  exists_before.close();

  rows.clear();
  ASSERT_TRUE(vector_index_metadata_store::save_all(rows));

  std::ifstream exists_after(m_path);
  EXPECT_FALSE(exists_after.good());
}

TEST_F(MetadataStoreTest, SaveEmptyRowsFailsWhenRemovalIsInjected) {
  std::vector<vector_index_metadata_store::metadata_row> rows{
      {"idx_meta",
       2,
       vector_index::metric_type::kEuclidean,
       vector_index::backend_mode::kMemory,
       vector_index::backend_provider::kNative,
       vector_index::index_consistency_mode::kTransactional,
       "",
       "",
       "",
       "ready",
       1,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       vector_index::diskann_build_mode::kAuto,
       ""}};
  assign_metadata_owner_for_test(&rows);
  ASSERT_TRUE(vector_index_metadata_store::save_all(rows));
  rows.clear();
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_metadata_store_fail_remove_if_exists");
    EXPECT_FALSE(vector_index_metadata_store::save_all(rows));
  }
}

TEST_F(MetadataStoreTest, SaveAllFailsWhenParentDirectoryCannotBeCreated) {
  const std::string blocked_parent = m_path + ".blocked_parent";
  std::ofstream blocker(blocked_parent,
                        std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(blocker.good());
  blocker << "block";
  blocker.close();
  ASSERT_TRUE(blocker);

  const std::string blocked_store_path = blocked_parent + "/metadata.dat";
  vector_index_metadata_store::set_path_for_testing(blocked_store_path);

  std::vector<vector_index_metadata_store::metadata_row> rows{
      {"idx_blocked",
       2,
       vector_index::metric_type::kEuclidean,
       vector_index::backend_mode::kMemory,
       vector_index::backend_provider::kNative,
       vector_index::index_consistency_mode::kTransactional,
       "",
       "",
       "",
       "ready",
       1,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       vector_index::diskann_build_mode::kAuto,
       ""}};
  EXPECT_FALSE(vector_index_metadata_store::save_all(rows));

  vector_index_metadata_store::set_path_for_testing(m_path);
  std::remove(blocked_parent.c_str());
}

TEST_F(MetadataStoreTest, RawPreparedArtifactRoundTrip) {
  const std::string payload = "prepared-payload";
  bool found = false;
  std::string loaded;

  ASSERT_TRUE(vector_index_metadata_store::save_raw_artifact("prepared", payload));
  ASSERT_TRUE(
      vector_index_metadata_store::load_raw_artifact("prepared", &loaded, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(payload, loaded);

  ASSERT_TRUE(vector_index_metadata_store::delete_raw_artifact("prepared"));
  ASSERT_TRUE(
      vector_index_metadata_store::load_raw_artifact("prepared", &loaded, &found));
  EXPECT_FALSE(found);
  EXPECT_TRUE(loaded.empty());
}

TEST_F(MetadataStoreTest, SaveRawArtifactSupportsAllKnownArtifacts) {
  const std::vector<std::string> artifact_names{"metadata", "committed",
                                                "manifest", "changelog",
                                                "prepared"};
  for (const auto &artifact_name : artifact_names) {
    const std::string expected_payload = artifact_name + "-payload";
    bool found = false;
    std::string loaded;

    ASSERT_TRUE(vector_index_metadata_store::save_raw_artifact(artifact_name,
                                                             expected_payload));
    ASSERT_TRUE(vector_index_metadata_store::load_raw_artifact(artifact_name,
                                                             &loaded, &found));
    EXPECT_TRUE(found);
    EXPECT_EQ(expected_payload, loaded);
    ASSERT_TRUE(vector_index_metadata_store::delete_raw_artifact(artifact_name));
  }
}

TEST_F(MetadataStoreTest, SaveRawArtifactFailsWhenParentDirectoryCannotBeCreated) {
  const std::string blocked_parent = m_path + ".raw_blocked_parent";
  std::ofstream blocker(blocked_parent,
                        std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(blocker.good());
  blocker << "block";
  blocker.close();
  ASSERT_TRUE(blocker);

  const std::string blocked_store_path = blocked_parent + "/raw.dat";
  vector_index_metadata_store::set_path_for_testing(blocked_store_path);

  EXPECT_FALSE(vector_index_metadata_store::save_raw_artifact("prepared",
                                                            "payload"));

  vector_index_metadata_store::set_path_for_testing(m_path);
  std::remove(blocked_parent.c_str());
}

TEST_F(MetadataStoreTest, SaveRawArtifactEmptyPayloadDeletesArtifact) {
  ASSERT_TRUE(
      vector_index_metadata_store::save_raw_artifact("manifest", "payload"));

  bool found = true;
  std::string loaded;
  ASSERT_TRUE(vector_index_metadata_store::save_raw_artifact("manifest", ""));
  ASSERT_TRUE(
      vector_index_metadata_store::load_raw_artifact("manifest", &loaded, &found));
  EXPECT_FALSE(found);
  EXPECT_TRUE(loaded.empty());
}

TEST_F(MetadataStoreTest, RawArtifactRejectsUnknownArtifactName) {
  std::string payload;
  bool found = false;
  EXPECT_FALSE(
      vector_index_metadata_store::save_raw_artifact("unknown", "payload"));
  EXPECT_FALSE(vector_index_metadata_store::load_raw_artifact("unknown", &payload,
                                                            &found));
  EXPECT_FALSE(vector_index_metadata_store::delete_raw_artifact("unknown"));
}

TEST_F(MetadataStoreTest, QuarantinePreparedStoreRenamesPreparedFile) {
  ASSERT_TRUE(
      vector_index_metadata_store::save_raw_artifact("prepared", "bad\nrow\n"));

  ASSERT_TRUE(vector_index_metadata_store::quarantine_prepared_store());

  std::ifstream current(m_prepared_path);
  EXPECT_FALSE(current.good());
  std::ifstream quarantined(m_prepared_path + ".corrupt");
  EXPECT_TRUE(quarantined.good());
}

TEST_F(MetadataStoreTest, SavePreparedFailsWhenParentDirectoryCannotBeCreated) {
  const std::string blocked_parent = m_path + ".prepared_blocked_parent";
  std::ofstream blocker(blocked_parent,
                        std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(blocker.good());
  blocker << "block";
  blocker.close();
  ASSERT_TRUE(blocker);

  const std::string blocked_store_path = blocked_parent + "/prepared.dat";
  vector_index_metadata_store::set_path_for_testing(blocked_store_path);

  std::vector<vector_index_metadata_store::prepared_change_row> rows{
      {1, 3, 0, "abc", false, 99,
       vector_index_metadata_store::change_op::kUpsert, "idx_prepared", 7,
       {1.0F, 2.0F}}};
  EXPECT_FALSE(vector_index_metadata_store::save_prepared(rows));

  vector_index_metadata_store::set_path_for_testing(m_path);
  std::remove(blocked_parent.c_str());
}

TEST_F(MetadataStoreTest, SaveEmptyPreparedRowsRemovesStoreFile) {
  std::vector<vector_index_metadata_store::prepared_change_row> rows{
      {1, 3, 0, "abc", false, 99,
       vector_index_metadata_store::change_op::kUpsert, "idx_prepared", 7,
       {1.0F, 2.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_prepared(rows));

  std::ifstream exists_before(m_prepared_path);
  ASSERT_TRUE(exists_before.good());
  exists_before.close();

  rows.clear();
  ASSERT_TRUE(vector_index_metadata_store::save_prepared(rows));

  std::ifstream exists_after(m_prepared_path);
  EXPECT_FALSE(exists_after.good());
}

TEST_F(MetadataStoreTest, SaveEmptyPreparedRowsFailsWhenRemovalIsInjected) {
  std::vector<vector_index_metadata_store::prepared_change_row> rows{
      {1, 3, 0, "abc", false, 99,
       vector_index_metadata_store::change_op::kUpsert, "idx_prepared", 7,
       {1.0F, 2.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_prepared(rows));
  rows.clear();
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_metadata_store_fail_remove_if_exists");
    EXPECT_FALSE(vector_index_metadata_store::save_prepared(rows));
  }
}

TEST_F(MetadataStoreTest, SaveThenLoadPreparedRoundTrip) {
  std::vector<vector_index_metadata_store::prepared_change_row> rows{
      {1, 3, 0, "abc", false, 99,
       vector_index_metadata_store::change_op::kUpsert, "idx_prepared", 7,
       {1.0F, 2.0F}},
      {2, 4, 0, "wxyz", true, 100,
       vector_index_metadata_store::change_op::kErase, "idx_prepared", 8, {}}};

  ASSERT_TRUE(vector_index_metadata_store::save_prepared(rows));

  std::vector<vector_index_metadata_store::prepared_change_row> loaded;
  ASSERT_TRUE(vector_index_metadata_store::load_prepared(&loaded));
  ASSERT_EQ(2U, loaded.size());
  EXPECT_EQ(1, loaded[0].format_id);
  EXPECT_EQ("abc", loaded[0].xid_data);
  EXPECT_FALSE(loaded[0].prepared_in_tc);
  EXPECT_EQ(99U, loaded[0].txn_id);
  EXPECT_EQ(vector_index_metadata_store::change_op::kUpsert, loaded[0].op);
  EXPECT_EQ("idx_prepared", loaded[0].index_name);
  EXPECT_EQ(7U, loaded[0].doc_id);
  ASSERT_EQ(2U, loaded[0].vector.size());
  EXPECT_FLOAT_EQ(1.0F, loaded[0].vector[0]);
  EXPECT_FLOAT_EQ(2.0F, loaded[0].vector[1]);

  EXPECT_EQ(2, loaded[1].format_id);
  EXPECT_EQ("wxyz", loaded[1].xid_data);
  EXPECT_TRUE(loaded[1].prepared_in_tc);
  EXPECT_EQ(vector_index_metadata_store::change_op::kErase, loaded[1].op);
  EXPECT_TRUE(loaded[1].vector.empty());
}

TEST_F(MetadataStoreTest, LoadPreparedRejectsCorruptedRow) {
  std::ofstream file(m_prepared_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-prepared-v1\n";
  file << "bad\trow\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::prepared_change_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_prepared(&loaded));
}

TEST_F(MetadataStoreTest, LoadPreparedRejectsInvalidXidHexPayload) {
  std::ofstream file(m_prepared_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-prepared-v1\n";
  file << "1\t3\t0\tzz\t0\t99\tupsert\t696478\t7\t0000803f00000040\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::prepared_change_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_prepared(&loaded));
}

TEST_F(MetadataStoreTest, LoadPreparedRejectsXidLengthMismatch) {
  std::ofstream file(m_prepared_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-prepared-v1\n";
  file << "1\t5\t0\t616263\t0\t99\tupsert\t696478\t7\t0000803f00000040\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::prepared_change_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_prepared(&loaded));
}

TEST_F(MetadataStoreTest, LoadPreparedRejectsEraseWithVectorPayload) {
  std::ofstream file(m_prepared_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-prepared-v1\n";
  file << "1\t3\t0\t616263\t0\t99\terase\t696478\t7\t0000803f\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::prepared_change_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_prepared(&loaded));
}

TEST_F(MetadataStoreTest, LoadPreparedRejectsInvalidPreparedInTcToken) {
  std::ofstream file(m_prepared_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-prepared-v1\n";
  file << "1\t3\t0\t616263\t2\t99\tupsert\t696478\t7\t0000803f00000040\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::prepared_change_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_prepared(&loaded));
}

TEST_F(MetadataStoreTest, LoadPreparedRejectsUnknownOperationToken) {
  std::ofstream file(m_prepared_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-prepared-v1\n";
  file << "1\t3\t0\t616263\t0\t99\tunknown\t696478\t7\t0000803f00000040\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::prepared_change_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_prepared(&loaded));
}

TEST_F(MetadataStoreTest, LoadPreparedRejectsInvalidIndexHex) {
  std::ofstream file(m_prepared_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-prepared-v1\n";
  file << "1\t3\t0\t616263\t0\t99\tupsert\tzz\t7\t0000803f00000040\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::prepared_change_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_prepared(&loaded));
}

TEST_F(MetadataStoreTest, DeserializePreparedRejectsInvalidFields) {
  const std::string index_hex = encode_hex_for_test("idx_prepared");
  const std::string header = "mysql-vector-prepared-v1\n";
  std::vector<vector_index_metadata_store::prepared_change_row> loaded;

  EXPECT_FALSE(vector_index_metadata_store::deserialize_prepared_rows(
      header + "9223372036854775808\t3\t0\t616263\t0\t99\tupsert\t" +
          index_hex + "\t7\t0000803f\n",
      &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_prepared_rows(
      header + "1\t9223372036854775808\t0\t616263\t0\t99\tupsert\t" +
          index_hex + "\t7\t0000803f\n",
      &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_prepared_rows(
      header + "1\t3\t9223372036854775808\t616263\t0\t99\tupsert\t" +
          index_hex + "\t7\t0000803f\n",
      &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_prepared_rows(
      header + "1\t3\t0\t616263\t0\tnot_a_txn\tupsert\t" + index_hex +
          "\t7\t0000803f\n",
      &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_prepared_rows(
      header + "1\t3\t0\t616263\t0\t99\tupsert\t" + index_hex +
          "\tnot_a_doc\t0000803f\n",
      &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_prepared_rows(
      header + "1\t3\t0\t616263\t0\t99\tupsert\t" + index_hex + "\t7\t0\n",
      &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_prepared_rows(
      header + "1\t3\t0\t616263\t0\t99\tupsert\t" + index_hex + "\t7\t00\n",
      &loaded));
}

TEST_F(MetadataStoreTest, SavePreparedRejectsInvalidRows) {
  std::vector<vector_index_metadata_store::prepared_change_row> rows{
      {-1, 3, 0, "abc", false, 99,
       vector_index_metadata_store::change_op::kUpsert, "idx_prepared", 7,
       {1.0F, 2.0F}}};
  EXPECT_FALSE(vector_index_metadata_store::save_prepared(rows));

  rows[0].format_id = 1;
  rows[0].index_name.clear();
  EXPECT_FALSE(vector_index_metadata_store::save_prepared(rows));

  rows[0].index_name = "idx_prepared";
  rows[0].op = vector_index_metadata_store::change_op::kErase;
  rows[0].vector = {1.0F};
  EXPECT_FALSE(vector_index_metadata_store::save_prepared(rows));

  rows[0].vector.clear();
  rows[0].op = vector_index_metadata_store::change_op::kUpsert;
  rows[0].gtrid_length = 5;
  EXPECT_FALSE(vector_index_metadata_store::save_prepared(rows));

  rows[0].gtrid_length = -1;
  EXPECT_FALSE(vector_index_metadata_store::save_prepared(rows));

  rows[0].gtrid_length = 3;
  rows[0].bqual_length = -1;
  EXPECT_FALSE(vector_index_metadata_store::save_prepared(rows));
}

TEST_F(MetadataStoreTest, SaveThenLoadCommittedRoundTrip) {
  std::vector<vector_index_metadata_store::committed_row> rows{
      {"idx_mem", 11, {1.0F, 2.0F, 3.0F}},
      {"idx_mem", 22, {4.0F, 5.0F, 6.0F}},
      {"idx_ext", 7, {0.25F, 0.5F}}};

  ASSERT_TRUE(vector_index_metadata_store::save_committed_all(rows));

  std::vector<vector_index_metadata_store::committed_row> loaded;
  ASSERT_TRUE(vector_index_metadata_store::load_committed_all(&loaded));
  ASSERT_EQ(3U, loaded.size());
  EXPECT_EQ("idx_mem", loaded[0].index_name);
  EXPECT_EQ(11U, loaded[0].doc_id);
  ASSERT_EQ(3U, loaded[0].vector.size());
  EXPECT_FLOAT_EQ(1.0F, loaded[0].vector[0]);
  EXPECT_FLOAT_EQ(2.0F, loaded[0].vector[1]);
  EXPECT_FLOAT_EQ(3.0F, loaded[0].vector[2]);
  EXPECT_EQ("idx_mem", loaded[1].index_name);
  EXPECT_EQ(22U, loaded[1].doc_id);
  ASSERT_EQ(3U, loaded[1].vector.size());
  EXPECT_FLOAT_EQ(4.0F, loaded[1].vector[0]);
  EXPECT_FLOAT_EQ(5.0F, loaded[1].vector[1]);
  EXPECT_FLOAT_EQ(6.0F, loaded[1].vector[2]);
  EXPECT_EQ("idx_ext", loaded[2].index_name);
  EXPECT_EQ(7U, loaded[2].doc_id);
  ASSERT_EQ(2U, loaded[2].vector.size());
  EXPECT_FLOAT_EQ(0.25F, loaded[2].vector[0]);
  EXPECT_FLOAT_EQ(0.5F, loaded[2].vector[1]);
}

TEST_F(MetadataStoreTest, DeserializeCommittedRejectsInvalidFields) {
  const std::string index_hex = encode_hex_for_test("idx_bad");
  const std::string header = "mysql-vector-committed-v1\n";
  std::vector<vector_index_metadata_store::committed_row> loaded;

  EXPECT_FALSE(vector_index_metadata_store::deserialize_committed_rows(
      header + index_hex + "\tnot_a_doc\t2\t0000803f00000040\n", &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_committed_rows(
      header + index_hex + "\t7\tnot_a_dim\t0000803f00000040\n", &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_committed_rows(
      header + index_hex + "\t7\t2\t0\n", &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_committed_rows(
      header + index_hex + "\t7\t2\t0000803f\n", &loaded));
}

TEST_F(MetadataStoreTest,
       SaveCommittedAllFailsWhenParentDirectoryCannotBeCreated) {
  const std::string blocked_parent = m_path + ".committed_blocked_parent";
  std::ofstream blocker(blocked_parent,
                        std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(blocker.good());
  blocker << "block";
  blocker.close();
  ASSERT_TRUE(blocker);

  const std::string blocked_store_path = blocked_parent + "/committed.dat";
  vector_index_metadata_store::set_path_for_testing(blocked_store_path);

  std::vector<vector_index_metadata_store::committed_row> rows{
      {"idx_blocked", 1, {1.0F, 2.0F}}};
  EXPECT_FALSE(vector_index_metadata_store::save_committed_all(rows));

  vector_index_metadata_store::set_path_for_testing(m_path);
  std::remove(blocked_parent.c_str());
}

TEST_F(MetadataStoreTest, SaveEmptyCommittedRowsRemovesStoreFile) {
  std::vector<vector_index_metadata_store::committed_row> rows{
      {"idx_committed", 1, {1.0F, 2.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_committed_all(rows));

  std::ifstream exists_before(m_committed_path);
  ASSERT_TRUE(exists_before.good());
  exists_before.close();

  rows.clear();
  ASSERT_TRUE(vector_index_metadata_store::save_committed_all(rows));

  std::ifstream exists_after(m_committed_path);
  EXPECT_FALSE(exists_after.good());
}

TEST_F(MetadataStoreTest, SaveEmptyCommittedRowsFailsWhenRemovalIsInjected) {
  std::vector<vector_index_metadata_store::committed_row> rows{
      {"idx_committed", 1, {1.0F, 2.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_committed_all(rows));
  rows.clear();
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_metadata_store_fail_remove_if_exists");
    EXPECT_FALSE(vector_index_metadata_store::save_committed_all(rows));
  }
}

TEST_F(MetadataStoreTest, LoadCommittedRejectsCorruptedRow) {
  std::ofstream file(m_committed_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-committed-v1\n";
  file << "bad\trow\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::committed_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_committed_all(&loaded));
}

TEST_F(MetadataStoreTest, LoadCommittedRejectsInvalidHexVectorPayload) {
  std::ofstream file(m_committed_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-committed-v1\n";
  file << "6964785f6d656d\t1\t2\tzz\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::committed_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_committed_all(&loaded));
}

TEST_F(MetadataStoreTest, LoadCommittedRejectsEmptyIndexName) {
  std::ofstream file(m_committed_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-committed-v1\n";
  file << "\t1\t2\t0000803f00000040\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::committed_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_committed_all(&loaded));
}

TEST_F(MetadataStoreTest, LoadCommittedRejectsMismatchedVectorSize) {
  std::ofstream file(m_committed_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-committed-v1\n";
  file << "6964785f636f6d6d6974746564\t1\t3\t0000803f00000040\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::committed_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_committed_all(&loaded));
}

TEST_F(MetadataStoreTest, QuarantineRenamesCommittedStore) {
  std::ofstream file(m_committed_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-committed-v1\n";
  file << "bad\trow\n";
  file.close();
  ASSERT_TRUE(file);

  ASSERT_TRUE(vector_index_metadata_store::quarantine_committed_store());

  std::ifstream current(m_committed_path);
  EXPECT_FALSE(current.good());
  std::ifstream quarantined(m_committed_path + ".corrupt");
  EXPECT_TRUE(quarantined.good());
}

TEST_F(MetadataStoreTest, QuarantineRenamesCurrentStore) {
  std::ofstream file(m_path, std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "VECTOR_INDEX_METADATA_V1\n";
  file << "bad\trow\n";
  file.close();
  ASSERT_TRUE(file);

  ASSERT_TRUE(vector_index_metadata_store::quarantine_current_store());

  std::ifstream current(m_path);
  EXPECT_FALSE(current.good());
  std::ifstream quarantined(m_path + ".corrupt");
  EXPECT_TRUE(quarantined.good());
}

TEST_F(MetadataStoreTest, SaveThenLoadManifestRoundTrip) {
  vector_index_metadata_store::manifest_row row;
  row.state = "ready";
  row.version = 42;
  row.metadata_checkpoint = 9;
  row.committed_checkpoint = 18;
  row.change_log_checkpoint = 27;

  ASSERT_TRUE(vector_index_metadata_store::save_manifest(row));

  vector_index_metadata_store::manifest_row loaded;
  ASSERT_TRUE(vector_index_metadata_store::load_manifest(&loaded));
  EXPECT_EQ("ready", loaded.state);
  EXPECT_EQ(42U, loaded.version);
  EXPECT_EQ(9U, loaded.metadata_checkpoint);
  EXPECT_EQ(18U, loaded.committed_checkpoint);
  EXPECT_EQ(27U, loaded.change_log_checkpoint);
}

TEST_F(MetadataStoreTest, LoadMissingManifestReturnsDefaultValues) {
  vector_index_metadata_store::manifest_row loaded;
  ASSERT_TRUE(vector_index_metadata_store::load_manifest(&loaded));
  EXPECT_EQ("ready", loaded.state);
  EXPECT_EQ(1U, loaded.version);
  EXPECT_EQ(0U, loaded.metadata_checkpoint);
  EXPECT_EQ(0U, loaded.committed_checkpoint);
  EXPECT_EQ(0U, loaded.change_log_checkpoint);
}

TEST_F(MetadataStoreTest, LoadManifestRejectsCorruptedRow) {
  std::ofstream file(m_manifest_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-manifest-v1\n";
  file << "bad\trow\n";
  file.close();
  ASSERT_TRUE(file);

  vector_index_metadata_store::manifest_row loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_manifest(&loaded));
}

TEST_F(MetadataStoreTest, LoadManifestRejectsOverflowVersion) {
  std::ofstream file(m_manifest_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-manifest-v1\n";
  file << "ready\t18446744073709551616\t0\t0\t0\n";
  file.close();
  ASSERT_TRUE(file);

  vector_index_metadata_store::manifest_row loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_manifest(&loaded));
}

TEST_F(MetadataStoreTest, LoadManifestRejectsEmptyState) {
  std::ofstream file(m_manifest_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-manifest-v1\n";
  file << "\t9\t1\t2\t3\n";
  file.close();
  ASSERT_TRUE(file);

  vector_index_metadata_store::manifest_row loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_manifest(&loaded));
}

TEST_F(MetadataStoreTest, LoadManifestRejectsWrongFieldCount) {
  std::ofstream file(m_manifest_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-manifest-v1\n";
  file << "ready\t9\t1\t2\n";
  file.close();
  ASSERT_TRUE(file);

  vector_index_metadata_store::manifest_row loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_manifest(&loaded));
}

TEST_F(MetadataStoreTest, DeserializeManifestRejectsInvalidCheckpoints) {
  vector_index_metadata_store::manifest_row loaded;

  EXPECT_FALSE(vector_index_metadata_store::deserialize_manifest_row(
      "mysql-vector-manifest-v1\nready\t1\tbad\t2\t3\n", &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_manifest_row(
      "mysql-vector-manifest-v1\nready\t1\t2\tbad\t3\n", &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_manifest_row(
      "mysql-vector-manifest-v1\nready\t1\t2\t3\tbad\n", &loaded));
}

TEST_F(MetadataStoreTest, SaveManifestRejectsInvalidInput) {
  vector_index_metadata_store::manifest_row invalid_empty_state;
  invalid_empty_state.state = "";
  invalid_empty_state.version = 1;
  EXPECT_FALSE(vector_index_metadata_store::save_manifest(invalid_empty_state));

  vector_index_metadata_store::manifest_row invalid_zero_version;
  invalid_zero_version.state = "ready";
  invalid_zero_version.version = 0;
  EXPECT_FALSE(vector_index_metadata_store::save_manifest(invalid_zero_version));
}

TEST_F(MetadataStoreTest,
       SaveManifestFailsWhenParentDirectoryCannotBeCreated) {
  const std::string blocked_parent = m_path + ".manifest_blocked_parent";
  std::ofstream blocker(blocked_parent,
                        std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(blocker.good());
  blocker << "block";
  blocker.close();
  ASSERT_TRUE(blocker);

  const std::string blocked_store_path = blocked_parent + "/manifest.dat";
  vector_index_metadata_store::set_path_for_testing(blocked_store_path);

  vector_index_metadata_store::manifest_row row;
  row.state = "ready";
  row.version = 7;
  row.metadata_checkpoint = 1;
  row.committed_checkpoint = 2;
  row.change_log_checkpoint = 3;
  EXPECT_FALSE(vector_index_metadata_store::save_manifest(row));

  vector_index_metadata_store::set_path_for_testing(m_path);
  std::remove(blocked_parent.c_str());
}

TEST_F(MetadataStoreTest, SaveThenLoadChangeLogRoundTrip) {
  std::vector<vector_index_metadata_store::change_log_row> rows{
      {1,
       88,
       vector_index_metadata_store::change_op::kUpsert,
       "idx_mem",
       101,
       {1.0F, 2.0F}},
      {2,
       88,
       vector_index_metadata_store::change_op::kErase,
       "idx_mem",
       202,
       {}}};

  ASSERT_TRUE(vector_index_metadata_store::save_change_log(rows));

  std::vector<vector_index_metadata_store::change_log_row> loaded;
  ASSERT_TRUE(vector_index_metadata_store::load_change_log(&loaded));
  ASSERT_EQ(2U, loaded.size());
  EXPECT_EQ(1U, loaded[0].sequence);
  EXPECT_EQ(88U, loaded[0].txn_id);
  EXPECT_EQ(vector_index_metadata_store::change_op::kUpsert, loaded[0].op);
  EXPECT_EQ("idx_mem", loaded[0].index_name);
  EXPECT_EQ(101U, loaded[0].doc_id);
  ASSERT_EQ(2U, loaded[0].vector.size());
  EXPECT_FLOAT_EQ(1.0F, loaded[0].vector[0]);
  EXPECT_FLOAT_EQ(2.0F, loaded[0].vector[1]);

  EXPECT_EQ(2U, loaded[1].sequence);
  EXPECT_EQ(88U, loaded[1].txn_id);
  EXPECT_EQ(vector_index_metadata_store::change_op::kErase, loaded[1].op);
  EXPECT_EQ("idx_mem", loaded[1].index_name);
  EXPECT_EQ(202U, loaded[1].doc_id);
  EXPECT_TRUE(loaded[1].vector.empty());
}

TEST_F(MetadataStoreTest,
       SaveChangeLogFailsWhenParentDirectoryCannotBeCreated) {
  const std::string blocked_parent = m_path + ".changelog_blocked_parent";
  std::ofstream blocker(blocked_parent,
                        std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(blocker.good());
  blocker << "block";
  blocker.close();
  ASSERT_TRUE(blocker);

  const std::string blocked_store_path = blocked_parent + "/changelog.dat";
  vector_index_metadata_store::set_path_for_testing(blocked_store_path);

  std::vector<vector_index_metadata_store::change_log_row> rows{
      {1, 1, vector_index_metadata_store::change_op::kUpsert,
       "idx_blocked", 1, {1.0F, 2.0F}}};
  EXPECT_FALSE(vector_index_metadata_store::save_change_log(rows));

  vector_index_metadata_store::set_path_for_testing(m_path);
  std::remove(blocked_parent.c_str());
}

TEST_F(MetadataStoreTest, SaveEmptyChangeLogRemovesStoreFile) {
  std::vector<vector_index_metadata_store::change_log_row> rows{
      {1, 1, vector_index_metadata_store::change_op::kUpsert, "idx_changelog", 1,
       {1.0F, 2.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_change_log(rows));

  std::ifstream exists_before(m_change_log_path);
  ASSERT_TRUE(exists_before.good());
  exists_before.close();

  rows.clear();
  ASSERT_TRUE(vector_index_metadata_store::save_change_log(rows));

  std::ifstream exists_after(m_change_log_path);
  EXPECT_FALSE(exists_after.good());
}

TEST_F(MetadataStoreTest, SaveEmptyChangeLogFailsWhenRemovalIsInjected) {
  std::vector<vector_index_metadata_store::change_log_row> rows{
      {1, 1, vector_index_metadata_store::change_op::kUpsert, "idx_changelog", 1,
       {1.0F, 2.0F}}};
  ASSERT_TRUE(vector_index_metadata_store::save_change_log(rows));
  rows.clear();
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_metadata_store_fail_remove_if_exists");
    EXPECT_FALSE(vector_index_metadata_store::save_change_log(rows));
  }
}

TEST_F(MetadataStoreTest, LoadChangeLogRejectsCorruptedRow) {
  std::ofstream file(m_change_log_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-changelog-v1\n";
  file << "bad\trow\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::change_log_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_change_log(&loaded));
}

TEST_F(MetadataStoreTest, LoadChangeLogRejectsUnknownOperationToken) {
  std::ofstream file(m_change_log_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-changelog-v1\n";
  file << "1\t88\tnoop\t6964785f626164\t101\t0000803f\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::change_log_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_change_log(&loaded));
}

TEST_F(MetadataStoreTest, LoadChangeLogRejectsZeroSequence) {
  std::ofstream file(m_change_log_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-changelog-v1\n";
  file << "0\t88\tupsert\t6964785f626164\t101\t0000803f\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::change_log_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_change_log(&loaded));
}

TEST_F(MetadataStoreTest, LoadChangeLogRejectsInvalidIndexHex) {
  std::ofstream file(m_change_log_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-changelog-v1\n";
  file << "1\t88\tupsert\tzz\t101\t0000803f\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::change_log_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_change_log(&loaded));
}

TEST_F(MetadataStoreTest, LoadChangeLogRejectsEraseWithVectorPayload) {
  std::ofstream file(m_change_log_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-changelog-v1\n";
  file << "1\t88\terase\t6964785f626164\t101\t0000803f\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::change_log_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_change_log(&loaded));
}

TEST_F(MetadataStoreTest, LoadChangeLogRejectsOverflowSequence) {
  std::ofstream file(m_change_log_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-changelog-v1\n";
  file << "18446744073709551616\t88\tupsert\t6964785f626164\t101\t0000803f\n";
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::change_log_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_change_log(&loaded));
}

TEST_F(MetadataStoreTest, DeserializeChangeLogRejectsInvalidFields) {
  const std::string index_hex = encode_hex_for_test("idx_bad");
  const std::string header = "mysql-vector-changelog-v1\n";
  std::vector<vector_index_metadata_store::change_log_row> loaded;

  EXPECT_FALSE(vector_index_metadata_store::deserialize_change_log_rows(
      header + "1\tnot_a_txn\tupsert\t" + index_hex +
          "\t101\t0000803f\n",
      &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_change_log_rows(
      header + "1\t88\tupsert\t" + index_hex + "\tnot_a_doc\t0000803f\n",
      &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_change_log_rows(
      header + "1\t88\tupsert\t" + index_hex + "\t101\t0\n", &loaded));
  EXPECT_FALSE(vector_index_metadata_store::deserialize_change_log_rows(
      header + "1\t88\tupsert\t" + index_hex + "\t101\t00\n", &loaded));
}

TEST_F(MetadataStoreTest, SaveChangeLogRejectsInvalidRows) {
  std::vector<vector_index_metadata_store::change_log_row> bad_rows{
      {0,
       1,
       vector_index_metadata_store::change_op::kUpsert,
       "idx_bad",
       1,
       {1.0F, 2.0F}}};
  EXPECT_FALSE(vector_index_metadata_store::save_change_log(bad_rows));

  bad_rows[0].sequence = 1;
  bad_rows[0].index_name.clear();
  EXPECT_FALSE(vector_index_metadata_store::save_change_log(bad_rows));

  bad_rows[0].index_name = "idx_bad";
  bad_rows[0].op = vector_index_metadata_store::change_op::kErase;
  bad_rows[0].vector = {1.0F};
  EXPECT_FALSE(vector_index_metadata_store::save_change_log(bad_rows));
}

TEST_F(MetadataStoreTest, LoadAllRejectsUnknownMetricToken) {
  std::ofstream file(m_path, std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  std::vector<std::string> fields = current_metadata_fields_for_test();
  fields[2] = "manhattan";
  file << current_metadata_payload_for_test(fields);
  file.close();
  ASSERT_TRUE(file);

  std::vector<vector_index_metadata_store::metadata_row> loaded;
  EXPECT_FALSE(vector_index_metadata_store::load_all(&loaded));
}

TEST_F(MetadataStoreTest, QuarantineRenamesManifestStore) {
  std::ofstream file(m_manifest_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-manifest-v1\n";
  file << "bad\trow\n";
  file.close();
  ASSERT_TRUE(file);

  ASSERT_TRUE(vector_index_metadata_store::quarantine_manifest_store());
  std::ifstream current(m_manifest_path);
  EXPECT_FALSE(current.good());
  std::ifstream quarantined(m_manifest_path + ".corrupt");
  EXPECT_TRUE(quarantined.good());
}

TEST_F(MetadataStoreTest, QuarantineRenamesChangeLogStore) {
  std::ofstream file(m_change_log_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "mysql-vector-changelog-v1\n";
  file << "bad\trow\n";
  file.close();
  ASSERT_TRUE(file);

  ASSERT_TRUE(vector_index_metadata_store::quarantine_change_log_store());
  std::ifstream current(m_change_log_path);
  EXPECT_FALSE(current.good());
  std::ifstream quarantined(m_change_log_path + ".corrupt");
  EXPECT_TRUE(quarantined.good());
}

TEST_F(MetadataStoreTest, QuarantineMissingStoresReturnsTrue) {
  EXPECT_TRUE(vector_index_metadata_store::quarantine_current_store());
  EXPECT_TRUE(vector_index_metadata_store::quarantine_committed_store());
  EXPECT_TRUE(vector_index_metadata_store::quarantine_manifest_store());
  EXPECT_TRUE(vector_index_metadata_store::quarantine_change_log_store());
}

TEST_F(MetadataStoreTest, QuarantineFailsWhenCorruptTargetIsDirectory) {
  std::ofstream file(m_path, std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << "VECTOR_INDEX_METADATA_V1\n";
  file << "bad\trow\n";
  file.close();
  ASSERT_TRUE(file);

  std::error_code ec;
  std::filesystem::create_directories(m_path + ".corrupt", ec);
  ASSERT_FALSE(ec);
  std::ofstream blocker(m_path + ".corrupt/blocker.txt",
                        std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(blocker.good());
  blocker << "block";
  blocker.close();
  ASSERT_TRUE(blocker);

  EXPECT_FALSE(vector_index_metadata_store::quarantine_current_store());
}

TEST(VectorIndexMetadataCodecTest, EmptyTransactionalStoresDecodeCleanly) {
  std::vector<vector_index_metadata_store::committed_row> committed_rows(1);
  EXPECT_TRUE(
      vector_index_metadata_store::detail::deserialize_committed_rows_impl(
          "", &committed_rows));
  EXPECT_TRUE(committed_rows.empty());

  std::vector<vector_index_metadata_store::change_log_row> change_log_rows(1);
  EXPECT_TRUE(
      vector_index_metadata_store::detail::deserialize_change_log_rows_impl(
          "", &change_log_rows));
  EXPECT_TRUE(change_log_rows.empty());

  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows(1);
  EXPECT_TRUE(
      vector_index_metadata_store::detail::deserialize_prepared_rows_impl(
          "", &prepared_rows));
  EXPECT_TRUE(prepared_rows.empty());
}

}  // namespace vector_index_metadata_store_unittest
