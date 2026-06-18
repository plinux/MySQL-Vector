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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "m_ctype.h"
#include "my_byteorder.h"
#include "mysqld_error.h"
#include "sql/auth/auth_acls.h"
#include "sql/item.h"
#include "sql/vector/item_vectorfunc.h"
#include "sql/vector/item_vectorfunc_internal.h"
#include "sql/parse_tree_helpers.h"
#include "sql/sql_class.h"
#include "sql/vector/sql_vector_load.h"
#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_status.h"
#include "sql/vector/vector_utils.h"
#include "unittest/gunit/test_utils.h"
#include "unittest/gunit/vector_test_utils.h"

namespace item_vectorfunc_unittest {

namespace {

using my_testing::Server_initializer;

Item_param *make_param_item(MEM_ROOT *root, uint pos_in_query = 1);

class ItemVectorFuncFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    initializer.SetUp();
    ASSERT_FALSE(thd()->set_db({STRING_WITH_LEN("test")}));
    thd()->security_context()->set_master_access(CREATE_ACL | ALTER_ACL |
                                                 DROP_ACL | SELECT_ACL);
    vector_index_registry::reset_for_testing();
  }
  void TearDown() override {
    vector_index_registry::reset_for_testing();
    initializer.TearDown();
  }
  THD *thd() { return initializer.thd(); }

  PT_item_list *make_item_list(std::initializer_list<Item *> items) {
    auto *list = new (thd()->mem_root) PT_item_list;
    for (Item *item : items) {
      EXPECT_FALSE(list->push_back(item));
    }
    return list;
  }

  void expect_valid_param_types(std::initializer_list<Item_param *> params) {
    for (Item_param *param : params) {
      EXPECT_NE(MYSQL_TYPE_INVALID, param->data_type());
    }
  }

  template <typename ItemType>
  void expect_unary_propagate_failure(const char *label, Item *arg) {
    SCOPED_TRACE(label);
    EXPECT_TRUE(ItemType(POS(), arg).resolve_type(thd()));
  }

  template <typename ItemType>
  void expect_list_unary_propagate_failure(const char *label, Item *arg) {
    SCOPED_TRACE(label);
    EXPECT_TRUE(ItemType(POS(), make_item_list({arg})).resolve_type(thd()));
  }

  template <typename ItemType>
  void expect_binary_propagate_failure(const char *label, Item *lhs,
                                       Item *rhs) {
    SCOPED_TRACE(label);
    EXPECT_TRUE(ItemType(POS(), lhs, rhs).resolve_type(thd()));
  }

  template <typename ItemType>
  void expect_list_binary_propagate_failure(const char *label, Item *lhs,
                                            Item *rhs) {
    SCOPED_TRACE(label);
    EXPECT_TRUE(ItemType(POS(), make_item_list({lhs, rhs})).resolve_type(thd()));
  }

  template <typename ItemType>
  void expect_list_ternary_propagate_failure(const char *label, Item *arg0,
                                             Item *arg1, Item *arg2) {
    SCOPED_TRACE(label);
    EXPECT_TRUE(
        ItemType(POS(), make_item_list({arg0, arg1, arg2})).resolve_type(thd()));
  }

  template <typename ItemType>
  void expect_list_quaternary_propagate_failure(const char *label, Item *arg0,
                                                Item *arg1, Item *arg2,
                                                Item *arg3) {
    SCOPED_TRACE(label);
    EXPECT_TRUE(ItemType(POS(), make_item_list({arg0, arg1, arg2, arg3}))
                    .resolve_type(thd()));
  }

  template <typename ItemType>
  void expect_list_quinary_propagate_failure(const char *label, Item *arg0,
                                             Item *arg1, Item *arg2, Item *arg3,
                                             Item *arg4) {
    SCOPED_TRACE(label);
    EXPECT_TRUE(ItemType(POS(), make_item_list({arg0, arg1, arg2, arg3, arg4}))
                    .resolve_type(thd()));
  }

  template <typename ItemType>
  void expect_unary_dynamic_params(const char *label) {
    SCOPED_TRACE(label);
    auto *arg = make_param_item(thd()->mem_root);
    ItemType item(POS(), arg);
    EXPECT_FALSE(item.resolve_type(thd()));
    expect_valid_param_types({arg});
  }

  template <typename ItemType>
  void expect_list_unary_dynamic_params(const char *label) {
    SCOPED_TRACE(label);
    auto *arg = make_param_item(thd()->mem_root);
    ItemType item(POS(), make_item_list({arg}));
    EXPECT_FALSE(item.resolve_type(thd()));
    expect_valid_param_types({arg});
  }

  template <typename ItemType>
  void expect_binary_dynamic_params(const char *label) {
    SCOPED_TRACE(label);
    auto *lhs = make_param_item(thd()->mem_root);
    auto *rhs = make_param_item(thd()->mem_root, 2);
    ItemType item(POS(), lhs, rhs);
    EXPECT_FALSE(item.resolve_type(thd()));
    expect_valid_param_types({lhs, rhs});
  }

  template <typename ItemType>
  void expect_list_binary_dynamic_params(const char *label) {
    SCOPED_TRACE(label);
    auto *lhs = make_param_item(thd()->mem_root);
    auto *rhs = make_param_item(thd()->mem_root, 2);
    ItemType item(POS(), make_item_list({lhs, rhs}));
    EXPECT_FALSE(item.resolve_type(thd()));
    expect_valid_param_types({lhs, rhs});
  }

  template <typename ItemType>
  void expect_list_ternary_dynamic_params(const char *label) {
    SCOPED_TRACE(label);
    auto *a = make_param_item(thd()->mem_root);
    auto *b = make_param_item(thd()->mem_root, 2);
    auto *c = make_param_item(thd()->mem_root, 3);
    ItemType item(POS(), make_item_list({a, b, c}));
    EXPECT_FALSE(item.resolve_type(thd()));
    expect_valid_param_types({a, b, c});
  }

  template <typename ItemType>
  void expect_list_quaternary_dynamic_params(const char *label) {
    SCOPED_TRACE(label);
    auto *a = make_param_item(thd()->mem_root);
    auto *b = make_param_item(thd()->mem_root, 2);
    auto *c = make_param_item(thd()->mem_root, 3);
    auto *d = make_param_item(thd()->mem_root, 4);
    ItemType item(POS(), make_item_list({a, b, c, d}));
    EXPECT_FALSE(item.resolve_type(thd()));
    expect_valid_param_types({a, b, c, d});
  }

  template <typename ItemType>
  void expect_list_quinary_dynamic_params(const char *label) {
    SCOPED_TRACE(label);
    auto *a = make_param_item(thd()->mem_root);
    auto *b = make_param_item(thd()->mem_root, 2);
    auto *c = make_param_item(thd()->mem_root, 3);
    auto *d = make_param_item(thd()->mem_root, 4);
    auto *e = make_param_item(thd()->mem_root, 5);
    ItemType item(POS(), make_item_list({a, b, c, d, e}));
    EXPECT_FALSE(item.resolve_type(thd()));
    expect_valid_param_types({a, b, c, d, e});
  }

  Server_initializer initializer;
};

class FailingPropagateItem : public Parse_tree_item {
 public:
  FailingPropagateItem() : Parse_tree_item(POS()) {}
  bool propagate_type(THD *, const Type_properties &) override { return true; }
};

class NullStringValueItem final : public Item_string {
 public:
  NullStringValueItem() : Item_string("", 0, &my_charset_bin) {}
  String *val_str(String *) override { return nullptr; }
};

class NullMarkedStringValueItem final : public Item_string {
 public:
  NullMarkedStringValueItem(const char *text, size_t length,
                            const CHARSET_INFO *charset)
      : Item_string(text, length, charset) {}

  String *val_str(String *str) override {
    String *value = Item_string::val_str(str);
    null_value = true;
    return value;
  }
};

class NullMarkedIntValueItem final : public Item_int {
 public:
  explicit NullMarkedIntValueItem(longlong input_value)
      : Item_int(input_value) {}

  longlong val_int() override {
    const longlong value = Item_int::val_int();
    null_value = true;
    return value;
  }
};

class ProcessAccessGuard {
 public:
  explicit ProcessAccessGuard(THD *thd)
      : m_thd(thd),
        m_original(thd->security_context()->master_access()) {
    m_thd->security_context()->set_master_access(m_original | PROCESS_ACL);
  }

  ~ProcessAccessGuard() {
    m_thd->security_context()->set_master_access(m_original);
  }

  ProcessAccessGuard(const ProcessAccessGuard &) = delete;
  ProcessAccessGuard &operator=(const ProcessAccessGuard &) = delete;

 private:
  THD *m_thd;
  Access_bitmask m_original;
};

class FileAccessGuard {
 public:
  explicit FileAccessGuard(THD *thd)
      : m_thd(thd),
        m_original(thd->security_context()->master_access()) {
    m_thd->security_context()->set_master_access(m_original | FILE_ACL |
                                                 SELECT_ACL);
  }

  ~FileAccessGuard() { m_thd->security_context()->set_master_access(m_original); }

  FileAccessGuard(const FileAccessGuard &) = delete;
  FileAccessGuard &operator=(const FileAccessGuard &) = delete;

 private:
  THD *m_thd;
  Access_bitmask m_original;
};

class SecureFilePrivGuard {
 public:
  explicit SecureFilePrivGuard(const char *value)
      : m_original(opt_secure_file_priv) {
    opt_secure_file_priv = value;
  }

  ~SecureFilePrivGuard() { opt_secure_file_priv = m_original; }

  SecureFilePrivGuard(const SecureFilePrivGuard &) = delete;
  SecureFilePrivGuard &operator=(const SecureFilePrivGuard &) = delete;

 private:
  const char *m_original;
};

template <typename T>
void write_binary_value(std::ofstream *file, T value) {
  file->write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_raw_fbin_file(const std::string &path, uint32_t rows,
                         uint32_t dimension,
                         const std::vector<float> &values) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  write_binary_value(&file, rows);
  write_binary_value(&file, dimension);
  file.write(reinterpret_cast<const char *>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  ASSERT_TRUE(file.good());
}

void write_raw_docid_file(const std::string &path,
                          const std::vector<uint64_t> &doc_ids) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  write_binary_value<uint64_t>(&file, doc_ids.size());
  file.write(reinterpret_cast<const char *>(doc_ids.data()),
             static_cast<std::streamsize>(doc_ids.size() *
                                          sizeof(uint64_t)));
  ASSERT_TRUE(file.good());
}

bool hnswlib_tuning_supported() {
  std::unique_ptr<vector_index::backend> backend =
      vector_index::create_backend(2, vector_index::metric_type::kEuclidean,
                                   vector_index::backend_mode::kMemory,
                                   vector_index::backend_provider::kHnswlib,
                                   "");
  return backend != nullptr && backend->set_search_ef(64) &&
         backend->set_hnsw_build_params(16, 200);
}

class controlled_truth_store : public vector_index_truth_store::truth_store {
 public:
  const char *backend_name() const override { return "controlled"; }
  bool is_transactional() const override { return false; }

  bool load_metadata(
      std::vector<vector_index_metadata_store::metadata_row> *rows) override {
    if (fail_load_metadata) return false;
    if (rows != nullptr) rows->clear();
    return true;
  }
  bool save_metadata(
      const std::vector<vector_index_metadata_store::metadata_row> &) override {
    return true;
  }
  bool quarantine_metadata() override { return !fail_quarantine_metadata; }

  bool load_committed(
      std::vector<vector_index_metadata_store::committed_row> *rows) override {
    if (rows != nullptr) rows->clear();
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
      std::vector<vector_index_metadata_store::change_log_row> *rows) override {
    if (rows != nullptr) rows->clear();
    return true;
  }
  bool save_change_log(
      const std::vector<vector_index_metadata_store::change_log_row> &) override {
    return true;
  }
  bool quarantine_change_log() override { return true; }

  bool load_prepared(
      std::vector<vector_index_metadata_store::prepared_change_row> *rows)
      override {
    if (rows != nullptr) rows->clear();
    return true;
  }
  bool save_prepared(
      const std::vector<vector_index_metadata_store::prepared_change_row> &) override {
    return true;
  }
  bool quarantine_prepared() override { return true; }

  bool load_segment_tasks(
      std::vector<vector_index_metadata_store::segment_task_row> *rows)
      override {
    if (rows != nullptr) rows->clear();
    return true;
  }
  bool save_segment_tasks(
      const std::vector<vector_index_metadata_store::segment_task_row> &)
      override {
    return true;
  }
  bool quarantine_segment_tasks() override { return true; }

  bool fail_load_metadata{false};
  bool fail_quarantine_metadata{false};
};

class TruthStoreOverrideGuard {
 public:
  explicit TruthStoreOverrideGuard(vector_index_truth_store::truth_store *store) {
    vector_index_truth_store::set_for_testing(store);
    vector_index_registry::reset_for_testing();
  }

  ~TruthStoreOverrideGuard() {
    vector_index_truth_store::reset_for_testing();
    vector_index_registry::reset_for_testing();
  }

  TruthStoreOverrideGuard(const TruthStoreOverrideGuard &) = delete;
  TruthStoreOverrideGuard &operator=(const TruthStoreOverrideGuard &) = delete;
};

std::string as_std_string(const String &value) {
  return std::string(value.ptr(), value.length());
}

LEX_STRING make_lex_string(const char *value) {
  return {const_cast<char *>(value), static_cast<size_t>(std::strlen(value))};
}

vector_index_registry::index_info make_base_info() {
  vector_index_registry::index_info info;
  info.dimension = 2;
  info.metric = "euclidean";
  info.mode = "memory";
  info.provider = "native";
  info.lifecycle_state = "ready";
  info.lifecycle_version = 1;
  return info;
}

Item_string *make_string_item(const char *text,
                              const CHARSET_INFO *charset = &my_charset_bin) {
  return new Item_string(text, std::strlen(text), charset);
}

std::string binary_vector_payload(std::initializer_list<float> values) {
  std::string payload(values.size() * vector_utils::kVectorElemSize, '\0');
  size_t idx = 0;
  for (float value : values) {
    float4store(reinterpret_cast<uchar *>(payload.data() +
                                          idx * vector_utils::kVectorElemSize),
                value);
    ++idx;
  }
  return payload;
}

Item_string *make_binary_vector_item(std::initializer_list<float> values) {
  const std::string payload = binary_vector_payload(values);
  char *stable_payload = new char[payload.size()];
  std::memcpy(stable_payload, payload.data(), payload.size());
  return new Item_string(stable_payload, payload.size(), &my_charset_bin);
}

std::string binary_docid_payload(std::initializer_list<uint64_t> values) {
  std::string payload(values.size() * sizeof(uint64_t), '\0');
  size_t idx = 0;
  for (uint64_t value : values) {
    int8store(reinterpret_cast<uchar *>(payload.data() +
                                        idx * sizeof(uint64_t)),
              value);
    ++idx;
  }
  return payload;
}

Item_string *make_binary_blob_item(const std::string &payload) {
  char *stable_payload = new char[payload.size()];
  if (!payload.empty()) std::memcpy(stable_payload, payload.data(), payload.size());
  return new Item_string(stable_payload, payload.size(), &my_charset_bin);
}

Item_string *make_null_marked_string_item(const char *text) {
  return new NullMarkedStringValueItem(text, std::strlen(text),
                                       &my_charset_bin);
}

Item *make_null_string_value_item() { return new NullStringValueItem(); }

Item_string *make_null_marked_binary_vector_item(
    std::initializer_list<float> values) {
  const std::string payload = binary_vector_payload(values);
  char *stable_payload = new char[payload.size()];
  std::memcpy(stable_payload, payload.data(), payload.size());
  return new NullMarkedStringValueItem(stable_payload, payload.size(),
                                       &my_charset_bin);
}

Item_int *make_null_marked_int(longlong value) {
  return new NullMarkedIntValueItem(value);
}

Item *fix_item(THD *thd, Item *item) {
  Item *ref = item;
  EXPECT_FALSE(item->fix_fields(thd, &ref));
  EXPECT_EQ(item, ref);
  return item;
}

Item_param *make_param_item(MEM_ROOT *root, uint pos_in_query) {
  return new Item_param(POS(), root, pos_in_query);
}

}  // namespace

TEST(ItemVectorFuncTest, FormatVectorIndexInfoRejectsNullBuffers) {
  vector_index_registry::index_info info = make_base_info();
  String out;
  String number_buf;
  EXPECT_FALSE(format_vector_index_info_json(info, nullptr, &number_buf));
  EXPECT_FALSE(format_vector_index_info_json(info, &out, nullptr));
}

TEST(ItemVectorFuncTest,
     FormatVectorIndexInfoFormatsNullOptionalsAndFailedState) {
  vector_index_registry::index_info info = make_base_info();
  info.lifecycle_state = "failed";
  info.lifecycle_version = 9;
  info.last_error_code = 77;
  info.last_error_ts = 88;
  info.last_apply_latency_ms = 99;
  info.recover_fallback_count = 5;
  info.last_recover_fallback_ts = 1234;
  info.external_manifest_generation = 0;
  info.supports_mutations = false;
  info.entry_count = 3;
  info.committed_entry_count = 7;

  String out;
  String number_buf;
  out.set_charset(&my_charset_bin);
  ASSERT_TRUE(format_vector_index_info_json(info, &out, &number_buf));

  const std::string json = as_std_string(out);
  EXPECT_NE(std::string::npos, json.find("\"backend_variant\":null"));
  EXPECT_NE(std::string::npos, json.find("\"schema_name\":null"));
  EXPECT_NE(std::string::npos, json.find("\"table_name\":null"));
  EXPECT_NE(std::string::npos, json.find("\"column_name\":null"));
  EXPECT_NE(std::string::npos, json.find("\"supports_mutations\":0"));
  EXPECT_NE(std::string::npos, json.find("\"lifecycle_state\":\"failed\""));
  EXPECT_NE(std::string::npos, json.find("\"rebuild_progress\":0"));
  EXPECT_NE(std::string::npos, json.find("\"recover_progress\":0"));
  EXPECT_NE(std::string::npos, json.find("\"pending_apply_count\":4"));
}

TEST(ItemVectorFuncTest,
     FormatVectorIndexInfoFormatsReadyAndProgressVariants) {
  String out;
  String number_buf;
  out.set_charset(&my_charset_bin);

  vector_index_registry::index_info ready = make_base_info();
  ready.backend_variant = "hnsw";
  ready.schema_name = "db1";
  ready.table_name = "t1";
  ready.column_name = "v1";
  ready.search_ef = 96;
  ready.hnsw_m = 24;
  ready.hnsw_ef_construction = 320;
  ready.hnsw_build_threads = 3;
  ready.faiss_nlist = 8;
  ready.faiss_nprobe = 4;
  ready.faiss_pq_m = 2;
  ready.faiss_pq_bits = 8;
  ready.diskann_max_degree = 64;
  ready.diskann_build_complexity = 128;
  ready.diskann_search_complexity = 32;
  ready.supports_mutations = true;
  ready.external_manifest_present = true;
  ready.external_manifest_generation = 7;
  ready.entry_count = 9;
  ready.committed_entry_count = 4;

  ASSERT_TRUE(format_vector_index_info_json(ready, &out, &number_buf));
  std::string json = as_std_string(out);
  EXPECT_NE(std::string::npos, json.find("\"backend_variant\":\"hnsw\""));
  EXPECT_NE(std::string::npos, json.find("\"schema_name\":\"db1\""));
  EXPECT_NE(std::string::npos, json.find("\"table_name\":\"t1\""));
  EXPECT_NE(std::string::npos, json.find("\"column_name\":\"v1\""));
  EXPECT_NE(std::string::npos, json.find("\"supports_mutations\":1"));
  EXPECT_NE(std::string::npos, json.find("\"external_manifest_present\":1"));
  EXPECT_NE(std::string::npos,
            json.find("\"external_manifest_generation\":7"));
  EXPECT_NE(std::string::npos, json.find("\"pending_apply_count\":5"));
  EXPECT_NE(std::string::npos, json.find("\"hnsw_build_threads\":3"));
  EXPECT_NE(std::string::npos, json.find("\"rebuild_progress\":100"));
  EXPECT_NE(std::string::npos, json.find("\"recover_progress\":100"));

  vector_index_registry::index_info rebuilding = make_base_info();
  rebuilding.lifecycle_state = "rebuilding";
  ASSERT_TRUE(
      format_vector_index_info_json(rebuilding, &out, &number_buf));
  json = as_std_string(out);
  EXPECT_NE(std::string::npos, json.find("\"rebuild_progress\":50"));
  EXPECT_NE(std::string::npos, json.find("\"recover_progress\":0"));

  vector_index_registry::index_info recovering = make_base_info();
  recovering.lifecycle_state = "recovering";
  ASSERT_TRUE(
      format_vector_index_info_json(recovering, &out, &number_buf));
  json = as_std_string(out);
  EXPECT_NE(std::string::npos, json.find("\"rebuild_progress\":0"));
  EXPECT_NE(std::string::npos, json.find("\"recover_progress\":50"));
}

TEST(ItemVectorFuncTest, FormatVectorIndexListRejectsNullOutput) {
  EXPECT_FALSE(format_vector_index_list_json({}, nullptr));
}

TEST(ItemVectorFuncTest, FormatVectorIndexListFormatsEmptyAndMultipleNames) {
  String out;
  out.set_charset(&my_charset_bin);

  ASSERT_TRUE(format_vector_index_list_json({}, &out));
  EXPECT_EQ("[]", as_std_string(out));

  ASSERT_TRUE(format_vector_index_list_json(
      {"idx_a", "db.t.c", "idx_z"}, &out));
  EXPECT_EQ("[\"idx_a\",\"db.t.c\",\"idx_z\"]", as_std_string(out));
}

TEST(ItemVectorFuncTest, FormatVectorsearch_resultDocIdsRejectsNullBuffers) {
  String out;
  String number_buf;
  EXPECT_FALSE(
      format_vector_search_result_doc_ids({}, nullptr, &number_buf));
  EXPECT_FALSE(format_vector_search_result_doc_ids({}, &out, nullptr));
}

TEST(ItemVectorFuncTest,
     FormatVectorsearch_resultDocIdsFormatsEmptyAndMultipleResults) {
  String out;
  String number_buf;
  out.set_charset(&my_charset_bin);

  ASSERT_TRUE(format_vector_search_result_doc_ids({}, &out, &number_buf));
  EXPECT_EQ("[]", as_std_string(out));

  const std::vector<vector_index::search_result> results = {
      {7, 0.0}, {42, 1.25}, {100, 9.5}};
  ASSERT_TRUE(
      format_vector_search_result_doc_ids(results, &out, &number_buf));
  EXPECT_EQ("[7,42,100]", as_std_string(out));
}

TEST(ItemVectorFuncTest,
     FormatVectorsearch_resultsWithDistanceRejectsNullBuffers) {
  String out;
  String number_buf;
  EXPECT_FALSE(format_vector_search_results_with_distance({}, nullptr,
                                                               &number_buf));
  EXPECT_FALSE(
      format_vector_search_results_with_distance({}, &out, nullptr));
}

TEST(ItemVectorFuncTest,
     FormatVectorsearch_resultsWithDistanceFormatsEmptyAndMultipleResults) {
  String out;
  String number_buf;
  out.set_charset(&my_charset_bin);

  ASSERT_TRUE(
      format_vector_search_results_with_distance({}, &out, &number_buf));
  EXPECT_EQ("[]", as_std_string(out));

  const std::vector<vector_index::search_result> results = {
      {7, 0.0}, {42, 1.25}, {100, -2.5}};
  ASSERT_TRUE(format_vector_search_results_with_distance(results, &out,
                                                              &number_buf));
  const std::string json = as_std_string(out);
  EXPECT_NE(std::string::npos, json.find("{\"doc_id\":7,\"distance\":0"));
  EXPECT_NE(std::string::npos, json.find("{\"doc_id\":42,\"distance\":1.25"));
  EXPECT_NE(std::string::npos, json.find("{\"doc_id\":100,\"distance\":-2.5"));
}

TEST(ItemVectorFuncTest, FormattersPropagateInjectedAppendFailures) {
  vector_index_registry::index_info info = make_base_info();
  const std::vector<vector_index::search_result> results = {{7, 1.25}};
  const std::vector<std::vector<vector_index::search_result>> batches = {
      results};
  String out;
  String number_buf;
  out.set_charset(&my_charset_bin);

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug_flag,
                             "+d,vector_item_fail_append_json_number");
    EXPECT_FALSE(format_vector_index_info_json(info, &out, &number_buf));
    EXPECT_FALSE(
        format_vector_search_result_doc_ids(results, &out, &number_buf));
    EXPECT_FALSE(
        format_vector_search_result_batches(batches, &out, &number_buf));
  }

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug_flag,
                             "+d,vector_item_fail_append_json_real");
    EXPECT_FALSE(
        format_vector_search_results_with_distance(results, &out, &number_buf));
  }
}

TEST(ItemVectorFuncTest, FormatSearchResultBatchesCoversGuardsAndFailures) {
  const std::vector<vector_index::search_result> results = {{7, 0.0},
                                                            {42, 1.25}};
  const std::vector<vector_index::search_result> empty_results;
  const std::vector<std::vector<vector_index::search_result>> batches = {
      results, empty_results};
  String out;
  String number_buf;
  out.set_charset(&my_charset_bin);

  EXPECT_FALSE(format_vector_search_result_batches(batches, nullptr,
                                                   &number_buf));
  EXPECT_FALSE(format_vector_search_result_batches(batches, &out, nullptr));

  ASSERT_TRUE(format_vector_search_result_batches(batches, &out, &number_buf));
  EXPECT_EQ("[[7,42],[]]", as_std_string(out));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug_flag,
                             "+d,vector_item_fail_append_json_number");
    EXPECT_FALSE(
        format_vector_search_result_batches(batches, &out, &number_buf));
  }
}

TEST_F(ItemVectorFuncFixture,
       ResolveTypeRejectsPropagateFailuresAcrossRegisteredFuncs) {
  FailingPropagateItem *arg = new FailingPropagateItem;
  Item_int *ok_int = new Item_int(7);
  FailingPropagateItem *lhs = new FailingPropagateItem;
  FailingPropagateItem *rhs = new FailingPropagateItem;

  expect_unary_propagate_failure<Item_func_vec_fromtext>("vec_fromtext", arg);
  expect_unary_propagate_failure<Item_func_vec_totext>("vec_totext", arg);
  expect_unary_propagate_failure<Item_func_vec_normalize>("vec_normalize", arg);
  expect_unary_propagate_failure<Item_func_vector_dim>("vector_dim", arg);
  expect_list_unary_propagate_failure<Item_func_vec_index_drop>(
      "vec_index_drop", arg);
  expect_list_unary_propagate_failure<Item_func_vec_index_rebuild>(
      "vec_index_rebuild", arg);
  expect_list_unary_propagate_failure<Item_func_vec_index_recover>(
      "vec_index_recover", arg);
  expect_list_unary_propagate_failure<Item_func_vec_index_bulk_load_begin>(
      "vec_index_bulk_load_begin", arg);
  expect_list_unary_propagate_failure<Item_func_vec_index_bulk_build>(
      "vec_index_bulk_build", arg);
  expect_list_unary_propagate_failure<Item_func_vec_index_info>(
      "vec_index_info", arg);
  expect_list_unary_propagate_failure<Item_func_vec_index_txn_pending>(
      "vec_index_txn_pending", arg);
  expect_list_unary_propagate_failure<Item_func_vec_index_txn_commit>(
      "vec_index_txn_commit", arg);
  expect_list_unary_propagate_failure<Item_func_vec_index_txn_rollback>(
      "vec_index_txn_rollback", arg);
  expect_list_unary_propagate_failure<Item_func_vec_debug_truth_store_get_hex>(
      "vec_debug_truth_store_get_hex", arg);

  expect_binary_propagate_failure<Item_func_vec_dot_product>(
      "vec_dot_product", lhs, rhs);
  expect_binary_propagate_failure<Item_func_vec_inner_product>(
      "vec_inner_product", lhs, rhs);
  expect_binary_propagate_failure<Item_func_vec_distance_euclidean>(
      "vec_distance_euclidean", lhs, rhs);
  expect_binary_propagate_failure<Item_func_vec_distance_cosine>(
      "vec_distance_cosine", lhs, rhs);
  expect_binary_propagate_failure<Item_func_vec_index_set_search_ef>(
      "vec_index_set_search_ef", lhs, rhs);
  expect_binary_propagate_failure<Item_func_vec_index_set_diskann_build_mode>(
      "vec_index_set_diskann_build_mode", lhs, rhs);
  expect_list_binary_propagate_failure<Item_func_vec_index_txn_savepoint>(
      "vec_index_txn_savepoint", lhs, rhs);
  expect_list_binary_propagate_failure<Item_func_vec_index_txn_rollback_to>(
      "vec_index_txn_rollback_to", lhs, rhs);
  expect_list_binary_propagate_failure<Item_func_vec_index_txn_release_savepoint>(
      "vec_index_txn_release_savepoint", lhs, rhs);
  expect_list_binary_propagate_failure<Item_func_vec_debug_truth_store_set_hex>(
      "vec_debug_truth_store_set_hex", lhs, rhs);

  expect_list_ternary_propagate_failure<Item_func_vec_distance>(
      "vec_distance", arg, ok_int, ok_int);
  expect_list_quinary_propagate_failure<Item_func_vec_index_create>(
      "vec_index_create", arg, ok_int, ok_int, ok_int, ok_int);
  expect_list_ternary_propagate_failure<Item_func_vec_index_set_hnsw_build_params>(
      "vec_index_set_hnsw_build_params", arg, ok_int, ok_int);
  expect_list_ternary_propagate_failure<Item_func_vec_index_set_faiss_ivf_params>(
      "vec_index_set_faiss_ivf_params", arg, ok_int, ok_int);
  expect_list_quinary_propagate_failure<
      Item_func_vec_index_set_faiss_ivfpq_params>(
      "vec_index_set_faiss_ivfpq_params", arg, ok_int, ok_int, ok_int, ok_int);
  expect_list_ternary_propagate_failure<
      Item_func_vec_index_set_diskann_build_params>(
      "vec_index_set_diskann_build_params", arg, ok_int, ok_int);
  expect_binary_propagate_failure<Item_func_vec_index_set_diskann_search_complexity>(
      "vec_index_set_diskann_search_complexity", arg, ok_int);
  expect_binary_propagate_failure<
      Item_func_vec_index_set_diskann_pq_code_budget_size>(
      "vec_index_set_diskann_pq_code_budget_size", arg, ok_int);
  expect_list_ternary_propagate_failure<Item_func_vec_index_upsert>(
      "vec_index_upsert", arg, ok_int, ok_int);
  expect_list_binary_propagate_failure<Item_func_vec_index_erase>(
      "vec_index_erase", arg, ok_int);
  expect_list_ternary_propagate_failure<Item_func_vec_index_search>(
      "vec_index_search", arg, ok_int, ok_int);
  expect_list_ternary_propagate_failure<
      Item_func_vec_index_search_with_distance>(
      "vec_index_search_with_distance", arg, ok_int, ok_int);
  expect_list_quaternary_propagate_failure<Item_func_vec_index_stage_upsert>(
      "vec_index_stage_upsert", arg, ok_int, ok_int, ok_int);
  expect_list_ternary_propagate_failure<Item_func_vec_index_stage_erase>(
      "vec_index_stage_erase", arg, ok_int, ok_int);
}

TEST_F(ItemVectorFuncFixture,
       ResolveTypeAcceptsDynamicParametersAcrossRegisteredFuncs) {
  expect_unary_dynamic_params<Item_func_vec_fromtext>("vec_fromtext");
  expect_unary_dynamic_params<Item_func_vec_totext>("vec_totext");
  expect_unary_dynamic_params<Item_func_vec_normalize>("vec_normalize");
  expect_unary_dynamic_params<Item_func_vector_dim>("vector_dim");
  expect_list_unary_dynamic_params<Item_func_vec_index_drop>("vec_index_drop");
  expect_list_unary_dynamic_params<Item_func_vec_index_rebuild>(
      "vec_index_rebuild");
  expect_list_unary_dynamic_params<Item_func_vec_index_recover>(
      "vec_index_recover");
  expect_list_unary_dynamic_params<Item_func_vec_index_bulk_load_begin>(
      "vec_index_bulk_load_begin");
  expect_list_unary_dynamic_params<Item_func_vec_index_bulk_build>(
      "vec_index_bulk_build");
  expect_list_unary_dynamic_params<Item_func_vec_index_info>("vec_index_info");
  expect_list_unary_dynamic_params<Item_func_vec_index_txn_pending>(
      "vec_index_txn_pending");
  expect_list_unary_dynamic_params<Item_func_vec_index_txn_commit>(
      "vec_index_txn_commit");
  expect_list_unary_dynamic_params<Item_func_vec_index_txn_rollback>(
      "vec_index_txn_rollback");
  expect_list_unary_dynamic_params<Item_func_vec_debug_truth_store_get_hex>(
      "vec_debug_truth_store_get_hex");

  expect_binary_dynamic_params<Item_func_vec_dot_product>("vec_dot_product");
  expect_binary_dynamic_params<Item_func_vec_inner_product>(
      "vec_inner_product");
  expect_binary_dynamic_params<Item_func_vec_distance_euclidean>(
      "vec_distance_euclidean");
  expect_binary_dynamic_params<Item_func_vec_distance_cosine>(
      "vec_distance_cosine");
  expect_binary_dynamic_params<Item_func_vec_index_set_search_ef>(
      "vec_index_set_search_ef");
  expect_binary_dynamic_params<Item_func_vec_index_set_diskann_build_mode>(
      "vec_index_set_diskann_build_mode");
  expect_list_binary_dynamic_params<Item_func_vec_index_txn_savepoint>(
      "vec_index_txn_savepoint");
  expect_list_binary_dynamic_params<Item_func_vec_index_txn_rollback_to>(
      "vec_index_txn_rollback_to");
  expect_list_binary_dynamic_params<Item_func_vec_index_txn_release_savepoint>(
      "vec_index_txn_release_savepoint");
  expect_list_binary_dynamic_params<Item_func_vec_debug_truth_store_set_hex>(
      "vec_debug_truth_store_set_hex");

  expect_list_ternary_dynamic_params<Item_func_vec_distance>("vec_distance");
  expect_list_quinary_dynamic_params<Item_func_vec_index_create>(
      "vec_index_create");
  expect_list_ternary_dynamic_params<Item_func_vec_index_set_hnsw_build_params>(
      "vec_index_set_hnsw_build_params");
  expect_list_ternary_dynamic_params<Item_func_vec_index_set_faiss_ivf_params>(
      "vec_index_set_faiss_ivf_params");
  expect_list_quinary_dynamic_params<
      Item_func_vec_index_set_faiss_ivfpq_params>(
      "vec_index_set_faiss_ivfpq_params");
  expect_list_ternary_dynamic_params<
      Item_func_vec_index_set_diskann_build_params>(
      "vec_index_set_diskann_build_params");
  expect_binary_dynamic_params<Item_func_vec_index_set_diskann_search_complexity>(
      "vec_index_set_diskann_search_complexity");
  expect_binary_dynamic_params<
      Item_func_vec_index_set_diskann_pq_code_budget_size>(
      "vec_index_set_diskann_pq_code_budget_size");
  expect_list_ternary_dynamic_params<Item_func_vec_index_upsert>(
      "vec_index_upsert");
  expect_list_binary_dynamic_params<Item_func_vec_index_erase>(
      "vec_index_erase");
  expect_list_ternary_dynamic_params<Item_func_vec_index_search>(
      "vec_index_search");
  expect_list_ternary_dynamic_params<
      Item_func_vec_index_search_with_distance>(
      "vec_index_search_with_distance");
  expect_list_quaternary_dynamic_params<Item_func_vec_index_stage_upsert>(
      "vec_index_stage_upsert");
  expect_list_ternary_dynamic_params<Item_func_vec_index_stage_erase>(
      "vec_index_stage_erase");
}

TEST_F(ItemVectorFuncFixture, TxnItemHelpersRejectInvalidArguments) {
  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_pending(
        POS(), make_item_list({new Item_int(-1)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_rollback(
        POS(), make_item_list({new Item_int(-1)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_savepoint(
        POS(), make_item_list({new Item_int(1), make_string_item("")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_rollback_to(
        POS(), make_item_list({new Item_int(1), make_string_item("")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_release_savepoint(
        POS(), make_item_list({new Item_int(1), make_string_item("")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }
}

TEST_F(ItemVectorFuncFixture, InternalItemHelpersCoverNoThreadFallbacks) {
  using namespace vector_itemfunc_internal;

  EXPECT_FALSE(require_process_access(nullptr));
  EXPECT_TRUE(check_vector_current_db_ddl_access(nullptr, CREATE_ACL));

  vector_index_registry::index_info info;
  info.owner_schema = "test";
  EXPECT_FALSE(check_vector_index_access(thd(), info, CREATE_ACL));
  EXPECT_TRUE(check_vector_index_access(nullptr, info, CREATE_ACL));

  EXPECT_FALSE(maybe_binlog_vector_write_query(nullptr));

  std::string decoded = "sentinel";
  EXPECT_FALSE(decode_hex_bytes(nullptr, decoded));
  String invalid_high("zz", &my_charset_latin1);
  EXPECT_FALSE(decode_hex_bytes(&invalid_high, decoded));
}

TEST_F(ItemVectorFuncFixture, TxnItemHelpersCoverSuccessPaths) {
  auto *begin_item = new Item_func_vec_index_txn_begin(POS());
  fix_item(thd(), begin_item);
  EXPECT_STREQ("vec_index_txn_begin", begin_item->func_name());
  const longlong txn_id_ll = begin_item->val_int();
  ASSERT_FALSE(begin_item->null_value);
  ASSERT_GT(txn_id_ll, 0);
  const uint64_t txn_id = static_cast<uint64_t>(txn_id_ll);

  auto *pending_item = new Item_func_vec_index_txn_pending(
      POS(), make_item_list({new Item_int(static_cast<longlong>(txn_id))}));
  fix_item(thd(), pending_item);
  EXPECT_EQ(0, pending_item->val_int());
  EXPECT_FALSE(pending_item->null_value);

  auto *savepoint_item = new Item_func_vec_index_txn_savepoint(
      POS(), make_item_list({new Item_int(static_cast<longlong>(txn_id)),
                             make_string_item("sp1")}));
  fix_item(thd(), savepoint_item);
  EXPECT_EQ(1, savepoint_item->val_int());
  EXPECT_FALSE(savepoint_item->null_value);

  auto *rollback_to_item = new Item_func_vec_index_txn_rollback_to(
      POS(), make_item_list({new Item_int(static_cast<longlong>(txn_id)),
                             make_string_item("sp1")}));
  fix_item(thd(), rollback_to_item);
  EXPECT_EQ(1, rollback_to_item->val_int());
  EXPECT_FALSE(rollback_to_item->null_value);

  auto *release_item = new Item_func_vec_index_txn_release_savepoint(
      POS(), make_item_list({new Item_int(static_cast<longlong>(txn_id)),
                             make_string_item("sp1")}));
  fix_item(thd(), release_item);
  EXPECT_EQ(1, release_item->val_int());
  EXPECT_FALSE(release_item->null_value);

  auto *rollback_item = new Item_func_vec_index_txn_rollback(
      POS(), make_item_list({new Item_int(static_cast<longlong>(txn_id))}));
  fix_item(thd(), rollback_item);
  EXPECT_EQ(1, rollback_item->val_int());
  EXPECT_FALSE(rollback_item->null_value);
}

TEST_F(ItemVectorFuncFixture, TxnItemHelpersCoverMissingMarkerBranches) {
  auto *begin_item = new Item_func_vec_index_txn_begin(POS());
  fix_item(thd(), begin_item);
  const longlong txn_id_ll = begin_item->val_int();
  ASSERT_FALSE(begin_item->null_value);
  ASSERT_GT(txn_id_ll, 0);

  auto *rollback_missing = new Item_func_vec_index_txn_rollback_to(
      POS(), make_item_list({new Item_int(txn_id_ll), make_string_item("missing")}));
  fix_item(thd(), rollback_missing);
  Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
  EXPECT_EQ(0, rollback_missing->val_int());
  EXPECT_FALSE(rollback_missing->null_value);
  thd()->clear_error();
  Server_initializer::set_expected_error(0);

  auto *release_missing = new Item_func_vec_index_txn_release_savepoint(
      POS(), make_item_list({new Item_int(txn_id_ll), make_string_item("missing")}));
  fix_item(thd(), release_missing);
  Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
  EXPECT_EQ(0, release_missing->val_int());
  EXPECT_FALSE(release_missing->null_value);
  thd()->clear_error();
  Server_initializer::set_expected_error(0);

  auto *savepoint_item = new Item_func_vec_index_txn_savepoint(
      POS(), make_item_list({new Item_int(txn_id_ll), make_string_item("spx")}));
  fix_item(thd(), savepoint_item);
  EXPECT_EQ(1, savepoint_item->val_int());
  EXPECT_FALSE(savepoint_item->null_value);

  auto *release_item = new Item_func_vec_index_txn_release_savepoint(
      POS(), make_item_list({new Item_int(txn_id_ll), make_string_item("spx")}));
  fix_item(thd(), release_item);
  EXPECT_EQ(1, release_item->val_int());
  EXPECT_FALSE(release_item->null_value);

  Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
  EXPECT_EQ(0, release_item->val_int());
  EXPECT_FALSE(release_item->null_value);
  thd()->clear_error();
  Server_initializer::set_expected_error(0);

  auto *rollback_item = new Item_func_vec_index_txn_rollback(
      POS(), make_item_list({new Item_int(txn_id_ll)}));
  fix_item(thd(), rollback_item);
  EXPECT_EQ(1, rollback_item->val_int());
  EXPECT_FALSE(rollback_item->null_value);
}

TEST_F(ItemVectorFuncFixture, TxnItemHelpersTreatUnknownTxnAsNoopOrEmpty) {
  auto *pending_item =
      new Item_func_vec_index_txn_pending(POS(), make_item_list({new Item_int(999)}));
  fix_item(thd(), pending_item);
  EXPECT_EQ(0, pending_item->val_int());
  EXPECT_FALSE(pending_item->null_value);

  auto *commit_item =
      new Item_func_vec_index_txn_commit(POS(), make_item_list({new Item_int(999)}));
  fix_item(thd(), commit_item);
  EXPECT_EQ(1, commit_item->val_int());
  EXPECT_FALSE(commit_item->null_value);

  auto *rollback_item = new Item_func_vec_index_txn_rollback(
      POS(), make_item_list({new Item_int(999)}));
  fix_item(thd(), rollback_item);
  EXPECT_EQ(1, rollback_item->val_int());
  EXPECT_FALSE(rollback_item->null_value);

  auto *savepoint_item = new Item_func_vec_index_txn_savepoint(
      POS(), make_item_list({new Item_int(999), make_string_item("spx")}));
  fix_item(thd(), savepoint_item);
  EXPECT_EQ(1, savepoint_item->val_int());
  EXPECT_FALSE(savepoint_item->null_value);
}

TEST_F(ItemVectorFuncFixture, RebuildAllAndRecoverAllItemsCoverErrorAndSuccess) {
  controlled_truth_store store;
  TruthStoreOverrideGuard truth_store_override(&store);
  const uint64_t rebuild_all_before = vector_status::rebuild_all_requests();
  const uint64_t recover_all_before = vector_status::recover_all_requests();

  auto *rebuild_all = new Item_func_vec_index_rebuild_all(POS());
  fix_item(thd(), rebuild_all);
  EXPECT_GE(rebuild_all->val_int(), 0);
  EXPECT_FALSE(rebuild_all->null_value);
  EXPECT_EQ(rebuild_all_before + 1, vector_status::rebuild_all_requests());

  auto *recover_all = new Item_func_vec_index_recover_all(POS());
  fix_item(thd(), recover_all);
  EXPECT_GE(recover_all->val_int(), 0);
  EXPECT_FALSE(recover_all->null_value);
  EXPECT_EQ(recover_all_before + 1, vector_status::recover_all_requests());

  const std::string index_name = "idx_item_all_" +
                                 std::to_string(reinterpret_cast<uintptr_t>(this));
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                 "memory", "native"));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    EXPECT_EQ(0, rebuild_all->val_int());
    EXPECT_FALSE(rebuild_all->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    EXPECT_EQ(0, recover_all->val_int());
    EXPECT_FALSE(recover_all->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }
}

TEST_F(ItemVectorFuncFixture, IndexAdminItemsCoverSuccessAndErrorPaths) {
  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item("idx_bad_create"), new Item_int(0)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  const std::string index_name =
      "idx_item_admin_" + std::to_string(reinterpret_cast<uintptr_t>(this));
  auto *create_item = new Item_func_vec_index_create(
      POS(), make_item_list({make_string_item(index_name.c_str()), new Item_int(2),
                             make_string_item("euclidean"),
                             make_string_item("memory"),
                             make_string_item("native")}));
  fix_item(thd(), create_item);
  EXPECT_EQ(1, create_item->val_int());
  EXPECT_FALSE(create_item->null_value);

  {
    auto *drop_missing = new Item_func_vec_index_drop(
        POS(), make_item_list({make_string_item("idx_missing_drop")}));
    fix_item(thd(), drop_missing);
    EXPECT_EQ(0, drop_missing->val_int());
    EXPECT_FALSE(drop_missing->null_value);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *rebuild_missing = new Item_func_vec_index_rebuild(
        POS(), make_item_list({make_string_item("idx_missing_rebuild")}));
    fix_item(thd(), rebuild_missing);
    EXPECT_EQ(0, rebuild_missing->val_int());
    EXPECT_FALSE(rebuild_missing->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *recover_missing = new Item_func_vec_index_recover(
        POS(), make_item_list({make_string_item("idx_missing_recover")}));
    fix_item(thd(), recover_missing);
    EXPECT_EQ(0, recover_missing->val_int());
    EXPECT_FALSE(recover_missing->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *bulk_begin_missing = new Item_func_vec_index_bulk_load_begin(
        POS(), make_item_list({make_string_item("idx_missing_bulk_begin")}));
    fix_item(thd(), bulk_begin_missing);
    EXPECT_EQ(0, bulk_begin_missing->val_int());
    EXPECT_FALSE(bulk_begin_missing->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *bulk_build_missing = new Item_func_vec_index_bulk_build(
        POS(), make_item_list({make_string_item("idx_missing_bulk_build")}));
    fix_item(thd(), bulk_build_missing);
    EXPECT_EQ(0, bulk_build_missing->val_int());
    EXPECT_FALSE(bulk_build_missing->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  auto *rebuild_item = new Item_func_vec_index_rebuild(
      POS(), make_item_list({make_string_item(index_name.c_str())}));
  fix_item(thd(), rebuild_item);
  EXPECT_EQ(1, rebuild_item->val_int());
  EXPECT_FALSE(rebuild_item->null_value);

  auto *recover_item = new Item_func_vec_index_recover(
      POS(), make_item_list({make_string_item(index_name.c_str())}));
  fix_item(thd(), recover_item);
  EXPECT_EQ(1, recover_item->val_int());
  EXPECT_FALSE(recover_item->null_value);

  auto *bulk_begin_item = new Item_func_vec_index_bulk_load_begin(
      POS(), make_item_list({make_string_item(index_name.c_str())}));
  fix_item(thd(), bulk_begin_item);
  EXPECT_EQ(1, bulk_begin_item->val_int());
  EXPECT_FALSE(bulk_begin_item->null_value);

  auto *bulk_build_item = new Item_func_vec_index_bulk_build(
      POS(), make_item_list({make_string_item(index_name.c_str())}));
  fix_item(thd(), bulk_build_item);
  EXPECT_EQ(1, bulk_build_item->val_int());
  EXPECT_FALSE(bulk_build_item->null_value);

  auto *drop_item = new Item_func_vec_index_drop(
      POS(), make_item_list({make_string_item(index_name.c_str())}));
  fix_item(thd(), drop_item);
  EXPECT_EQ(1, drop_item->val_int());
  EXPECT_FALSE(drop_item->null_value);
}

TEST_F(ItemVectorFuncFixture,
       StandaloneIndexAdminItemsRequireCurrentDatabaseForDdlPrivilege) {
  ASSERT_FALSE(thd()->set_db({nullptr, 0}));

  auto expect_no_database = [this](Item *item) {
    Server_initializer::set_expected_error(ER_NO_DB_ERROR);
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  };

  expect_no_database(new Item_func_vec_index_create(
      POS(), make_item_list({make_string_item("idx_no_db_create"), new Item_int(2),
                             make_string_item("euclidean"),
                             make_string_item("memory"),
                             make_string_item("native")})));
  expect_no_database(new Item_func_vec_index_drop(
      POS(), make_item_list({make_string_item("idx_no_db_drop")})));

  ASSERT_FALSE(thd()->set_db({STRING_WITH_LEN("test")}));
}

TEST_F(ItemVectorFuncFixture,
       ItemAdminAndTuningApisPropagateInjectedBinlogFailureAfterSuccess) {
  const std::string native_index =
      "idx_item_binlog_mem_" + std::to_string(reinterpret_cast<uintptr_t>(this));
  const std::string hnsw_index = native_index + "_hnsw";
  const std::string faiss_index = native_index + "_faiss";
  const std::string faiss_pq_index = native_index + "_faiss_pq";
  const std::string diskann_index = native_index + "_diskann";
  const bool has_hnswlib_tuning = hnswlib_tuning_supported();

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug_flag, "+d,vector_item_fail_binlog_write");

    auto *create_mem = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item(native_index.c_str()),
                               new Item_int(2), make_string_item("euclidean"),
                               make_string_item("memory"),
                               make_string_item("native")}));
    fix_item(thd(), create_mem);
    EXPECT_EQ(0, create_mem->val_int());
    EXPECT_FALSE(create_mem->null_value);

    if (has_hnswlib_tuning) {
      auto *create_hnsw = new Item_func_vec_index_create(
          POS(), make_item_list({make_string_item(hnsw_index.c_str()),
                                 new Item_int(2),
                                 make_string_item("euclidean"),
                                 make_string_item("memory"),
                                 make_string_item("hnswlib")}));
      fix_item(thd(), create_hnsw);
      EXPECT_EQ(0, create_hnsw->val_int());
      EXPECT_FALSE(create_hnsw->null_value);
    }

    auto *create_faiss = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item(faiss_index.c_str()),
                               new Item_int(2), make_string_item("euclidean"),
                               make_string_item("external"),
                               make_string_item("faiss")}));
    fix_item(thd(), create_faiss);
    EXPECT_EQ(0, create_faiss->val_int());
    EXPECT_FALSE(create_faiss->null_value);

    auto *create_faiss_pq = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item(faiss_pq_index.c_str()),
                               new Item_int(4), make_string_item("euclidean"),
                               make_string_item("external"),
                               make_string_item("faiss")}));
    fix_item(thd(), create_faiss_pq);
    EXPECT_EQ(0, create_faiss_pq->val_int());
    EXPECT_FALSE(create_faiss_pq->null_value);

    auto *create_diskann = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item(diskann_index.c_str()),
                               new Item_int(2), make_string_item("euclidean"),
                               make_string_item("external"),
                               make_string_item("diskann")}));
    fix_item(thd(), create_diskann);
    EXPECT_EQ(0, create_diskann->val_int());
    EXPECT_FALSE(create_diskann->null_value);

    if (has_hnswlib_tuning) {
      auto *set_search_ef = new Item_func_vec_index_set_search_ef(
          POS(), make_string_item(hnsw_index.c_str()), new Item_int(64));
      fix_item(thd(), set_search_ef);
      EXPECT_EQ(0, set_search_ef->val_int());
      EXPECT_FALSE(set_search_ef->null_value);

      auto *set_hnsw = new Item_func_vec_index_set_hnsw_build_params(
          POS(), make_item_list({make_string_item(hnsw_index.c_str()),
                                 new Item_int(16), new Item_int(200)}));
      fix_item(thd(), set_hnsw);
      EXPECT_EQ(0, set_hnsw->val_int());
      EXPECT_FALSE(set_hnsw->null_value);
    }

    auto *set_faiss_ivf = new Item_func_vec_index_set_faiss_ivf_params(
        POS(), make_item_list({make_string_item(faiss_index.c_str()),
                               new Item_int(2), new Item_int(1)}));
    fix_item(thd(), set_faiss_ivf);
    EXPECT_EQ(0, set_faiss_ivf->val_int());
    EXPECT_FALSE(set_faiss_ivf->null_value);

    auto *set_faiss_ivfpq = new Item_func_vec_index_set_faiss_ivfpq_params(
        POS(), make_item_list({make_string_item(faiss_pq_index.c_str()),
                               new Item_int(2), new Item_int(1),
                               new Item_int(1), new Item_int(1)}));
    fix_item(thd(), set_faiss_ivfpq);
    EXPECT_EQ(0, set_faiss_ivfpq->val_int());
    EXPECT_FALSE(set_faiss_ivfpq->null_value);

    auto *set_diskann_build = new Item_func_vec_index_set_diskann_build_params(
        POS(), make_item_list({make_string_item(diskann_index.c_str()),
                               new Item_int(48), new Item_int(96)}));
    fix_item(thd(), set_diskann_build);
    EXPECT_EQ(0, set_diskann_build->val_int());
    EXPECT_FALSE(set_diskann_build->null_value);

    auto *set_diskann_search =
        new Item_func_vec_index_set_diskann_search_complexity(
            POS(), make_string_item(diskann_index.c_str()), new Item_int(80));
    fix_item(thd(), set_diskann_search);
    EXPECT_EQ(0, set_diskann_search->val_int());
    EXPECT_FALSE(set_diskann_search->null_value);

    auto *set_diskann_pq = new Item_func_vec_index_set_diskann_pq_code_budget_size(
        POS(), make_string_item(diskann_index.c_str()), new Item_int(1048576));
    fix_item(thd(), set_diskann_pq);
    EXPECT_EQ(0, set_diskann_pq->val_int());
    EXPECT_FALSE(set_diskann_pq->null_value);

    auto *rebuild_item = new Item_func_vec_index_rebuild(
        POS(), make_item_list({make_string_item(native_index.c_str())}));
    fix_item(thd(), rebuild_item);
    EXPECT_EQ(0, rebuild_item->val_int());
    EXPECT_FALSE(rebuild_item->null_value);

    auto *recover_item = new Item_func_vec_index_recover(
        POS(), make_item_list({make_string_item(native_index.c_str())}));
    fix_item(thd(), recover_item);
    EXPECT_EQ(0, recover_item->val_int());
    EXPECT_FALSE(recover_item->null_value);

    auto *bulk_begin_item = new Item_func_vec_index_bulk_load_begin(
        POS(), make_item_list({make_string_item(native_index.c_str())}));
    fix_item(thd(), bulk_begin_item);
    EXPECT_EQ(0, bulk_begin_item->val_int());
    EXPECT_FALSE(bulk_begin_item->null_value);

    auto *bulk_build_item = new Item_func_vec_index_bulk_build(
        POS(), make_item_list({make_string_item(native_index.c_str())}));
    fix_item(thd(), bulk_build_item);
    EXPECT_EQ(0, bulk_build_item->val_int());
    EXPECT_FALSE(bulk_build_item->null_value);

    auto *drop_mem = new Item_func_vec_index_drop(
        POS(), make_item_list({make_string_item(native_index.c_str())}));
    fix_item(thd(), drop_mem);
    EXPECT_EQ(0, drop_mem->val_int());
    EXPECT_FALSE(drop_mem->null_value);

    if (has_hnswlib_tuning) {
      auto *drop_hnsw = new Item_func_vec_index_drop(
          POS(), make_item_list({make_string_item(hnsw_index.c_str())}));
      fix_item(thd(), drop_hnsw);
      EXPECT_EQ(0, drop_hnsw->val_int());
      EXPECT_FALSE(drop_hnsw->null_value);
    }

    auto *drop_faiss = new Item_func_vec_index_drop(
        POS(), make_item_list({make_string_item(faiss_index.c_str())}));
    fix_item(thd(), drop_faiss);
    EXPECT_EQ(0, drop_faiss->val_int());
    EXPECT_FALSE(drop_faiss->null_value);

    auto *drop_faiss_pq = new Item_func_vec_index_drop(
        POS(), make_item_list({make_string_item(faiss_pq_index.c_str())}));
    fix_item(thd(), drop_faiss_pq);
    EXPECT_EQ(0, drop_faiss_pq->val_int());
    EXPECT_FALSE(drop_faiss_pq->null_value);

    auto *drop_diskann = new Item_func_vec_index_drop(
        POS(), make_item_list({make_string_item(diskann_index.c_str())}));
    fix_item(thd(), drop_diskann);
    EXPECT_EQ(0, drop_diskann->val_int());
    EXPECT_FALSE(drop_diskann->null_value);
  }
}

TEST_F(ItemVectorFuncFixture,
       ItemMutationApisPropagateInjectedBinlogFailureAfterSuccess) {
  const std::string index_name =
      "idx_item_mut_binlog_" + std::to_string(reinterpret_cast<uintptr_t>(this));
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                 "memory", "native"));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug_flag, "+d,vector_item_fail_binlog_write");

    auto *upsert_item = new Item_func_vec_index_upsert(
        POS(), make_item_list({make_string_item(index_name.c_str()),
                               new Item_int(17),
                               make_binary_vector_item({1.0F, 7.0F})}));
    fix_item(thd(), upsert_item);
    EXPECT_EQ(0, upsert_item->val_int());
    EXPECT_FALSE(upsert_item->null_value);
  }

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 7.0F}, 1, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(17U, results[0].doc_id);

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug_flag, "+d,vector_item_fail_binlog_write");

    auto *erase_item = new Item_func_vec_index_erase(
        POS(), make_item_list({make_string_item(index_name.c_str()),
                               new Item_int(17)}));
    fix_item(thd(), erase_item);
    EXPECT_EQ(0, erase_item->val_int());
    EXPECT_FALSE(erase_item->null_value);
  }

  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 7.0F}, 1, &results));
  EXPECT_TRUE(results.empty());

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(ItemVectorFuncFixture,
       ItemAdminAndMutationItemsRejectNullOrNegativeArguments) {
  const longlong uint32_overflow =
      static_cast<longlong>(std::numeric_limits<uint32_t>::max()) + 1LL;
  const longlong dimension_overflow =
      static_cast<longlong>(vector_index::k_max_vector_dimension) + 1LL;
  const longlong build_threads_overflow =
      static_cast<longlong>(vector_index::k_max_build_threads) + 1LL;
  auto expect_wrong_arguments_int = [this](Item *item) {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  };

  {
    auto *item = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item("idx_create_null_dimension"),
                               make_null_marked_int(2)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item("idx_create_dimension_overflow"),
                               new Item_int(dimension_overflow)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item("idx_create_null_threads"),
                               new Item_int(2), make_string_item("euclidean"),
                               make_string_item("memory"),
                               make_string_item("native"),
                               make_null_marked_int(0)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item("idx_create_neg_threads"),
                               new Item_int(2), make_string_item("euclidean"),
                               make_string_item("memory"),
                               make_string_item("native"),
                               new Item_int(-1)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item("idx_create_threads_overflow"),
                               new Item_int(2), make_string_item("euclidean"),
                               make_string_item("memory"),
                               make_string_item("native"),
                               new Item_int(build_threads_overflow)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item("idx_create_null_metric"),
                               new Item_int(2), new Item_null()}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item("idx_create_null_mode"),
                               new Item_int(2), make_string_item("euclidean"),
                               new Item_null()}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_create(
        POS(),
        make_item_list({make_string_item("idx_create_null_provider"),
                        new Item_int(2), make_string_item("euclidean"),
                        make_string_item("memory"), new Item_null()}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_create(
        POS(), make_item_list({make_string_item("idx_create_bad_metric"),
                               new Item_int(2), make_string_item("bad_metric")}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_search_ef(
        POS(), make_null_marked_string_item("idx_any"), new Item_int(16));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_search_ef(
        POS(), make_string_item("idx_any"), make_null_marked_int(16));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_search_ef(
        POS(), make_string_item("idx_any"), new Item_int(0));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_search_ef(
        POS(), make_string_item("idx_any"), new Item_int(uint32_overflow));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_hnsw_build_params(
        POS(), make_item_list({make_null_marked_string_item("idx_any"),
                               new Item_int(16), new Item_int(200)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_hnsw_build_params(
        POS(), make_item_list({make_string_item("idx_any"),
                               make_null_marked_int(16), new Item_int(200)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_hnsw_build_params(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(16),
                               make_null_marked_int(200)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_hnsw_build_params(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(0),
                               new Item_int(200)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_hnsw_build_params(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(16),
                               new Item_int(uint32_overflow)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_faiss_ivf_params(
        POS(), make_item_list({make_null_marked_string_item("idx_any"),
                               new Item_int(2), new Item_int(1)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_faiss_ivf_params(
        POS(), make_item_list({make_string_item("idx_any"),
                               make_null_marked_int(2), new Item_int(1)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_faiss_ivf_params(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(2),
                               make_null_marked_int(1)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_faiss_ivf_params(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(-1),
                               new Item_int(1)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_faiss_ivf_params(
        POS(), make_item_list({make_string_item("idx_any"),
                               new Item_int(uint32_overflow), new Item_int(1)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_faiss_ivfpq_params(
        POS(), make_item_list({make_null_marked_string_item("idx_any"),
                               new Item_int(2), new Item_int(1), new Item_int(2),
                               new Item_int(8)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_faiss_ivfpq_params(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(2),
                               make_null_marked_int(1), new Item_int(2),
                               new Item_int(8)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_faiss_ivfpq_params(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(2),
                               new Item_int(1), make_null_marked_int(2),
                               new Item_int(8)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_faiss_ivfpq_params(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(2),
                               new Item_int(1), new Item_int(2),
                               make_null_marked_int(8)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_faiss_ivfpq_params(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(0),
                               new Item_int(1), new Item_int(2),
                               new Item_int(8)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_faiss_ivfpq_params(
        POS(), make_item_list({make_string_item("idx_any"),
                               new Item_int(uint32_overflow), new Item_int(1),
                               new Item_int(2), new Item_int(8)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_diskann_build_params(
        POS(), make_item_list({make_null_marked_string_item("idx_any"),
                               new Item_int(64), new Item_int(100)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_diskann_build_params(
        POS(), make_item_list({make_string_item("idx_any"),
                               make_null_marked_int(64), new Item_int(100)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_diskann_build_params(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(64),
                               make_null_marked_int(100)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_diskann_build_params(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(0),
                               new Item_int(100)}));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_diskann_search_complexity(
        POS(), make_null_marked_string_item("idx_any"), new Item_int(100));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_diskann_search_complexity(
        POS(), make_string_item("idx_any"), make_null_marked_int(100));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_diskann_search_complexity(
        POS(), make_string_item("idx_any"), new Item_int(0));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_diskann_search_complexity(
        POS(), make_string_item("idx_any"), new Item_int(uint32_overflow));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_diskann_pq_code_budget_size(
        POS(), make_null_marked_string_item("idx_any"), new Item_int(1048576));
    expect_wrong_arguments_int(item);
  }

  {
    auto *item = new Item_func_vec_index_set_diskann_pq_code_budget_size(
        POS(), make_string_item("idx_any"), make_null_marked_int(1048576));
    expect_wrong_arguments_int(item);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_upsert(
        POS(), make_item_list({new Item_null(), new Item_int(1),
                               make_binary_vector_item({1.0F, 1.0F})}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_erase(
        POS(), make_item_list({make_string_item("idx_erase_neg_doc"),
                               new Item_int(-1)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  String out;
  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_search(
        POS(), make_item_list({make_string_item("idx_search_neg_topk"),
                               make_binary_vector_item({1.0F, 1.0F}),
                               new Item_int(-1)}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_search_with_distance(
        POS(), make_item_list({new Item_null(),
                               make_binary_vector_item({1.0F, 1.0F}),
                               new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_drop(POS(), make_item_list({new Item_null()}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item =
        new Item_func_vec_index_rebuild(POS(), make_item_list({new Item_null()}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item =
        new Item_func_vec_index_recover(POS(), make_item_list({new Item_null()}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }
}

TEST_F(ItemVectorFuncFixture, ItemTuningItemsRejectMissingIndexes) {
  auto expect_wrong_arguments_int = [this](Item *item) {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  };

  expect_wrong_arguments_int(new Item_func_vec_index_set_search_ef(
      POS(), make_string_item("idx_missing_search_ef"), new Item_int(16)));
  expect_wrong_arguments_int(new Item_func_vec_index_set_hnsw_build_params(
      POS(), make_item_list({make_string_item("idx_missing_hnsw_params"),
                             new Item_int(16), new Item_int(200)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_faiss_ivf_params(
      POS(), make_item_list({make_string_item("idx_missing_faiss_ivf"),
                             new Item_int(2), new Item_int(1)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_faiss_ivfpq_params(
      POS(), make_item_list({make_string_item("idx_missing_faiss_ivfpq"),
                             new Item_int(2), new Item_int(1), new Item_int(2),
                             new Item_int(8)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_diskann_build_params(
      POS(), make_item_list({make_string_item("idx_missing_diskann_build"),
                             new Item_int(64), new Item_int(100)})));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_search_complexity(
          POS(), make_string_item("idx_missing_diskann_search"),
          new Item_int(100)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_pq_code_budget_size(
          POS(), make_string_item("idx_missing_diskann_pq"), new Item_int(0)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_disk_pq_dims(
          POS(), make_string_item("idx_missing_diskann_disk_pq"),
          new Item_int(12)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_accelerate_build(
          POS(), make_string_item("idx_missing_diskann_accelerate"),
          new Item_int(1)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_shuffle_build(
          POS(), make_string_item("idx_missing_diskann_shuffle"),
          new Item_int(1)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_use_bfs_cache(
          POS(), make_string_item("idx_missing_diskann_bfs"),
          new Item_int(1)));
}

TEST_F(ItemVectorFuncFixture, ItemAdminItemsRejectNullStringValues) {
  auto expect_wrong_arguments_int = [this](Item *item) {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  };
  expect_wrong_arguments_int(new Item_func_vec_index_create(
      POS(), make_item_list({make_null_string_value_item(), new Item_int(2)})));
  expect_wrong_arguments_int(new Item_func_vec_index_create(
      POS(), make_item_list({make_string_item("idx_null_metric"),
                             new Item_int(2), make_null_string_value_item()})));
  expect_wrong_arguments_int(new Item_func_vec_index_create(
      POS(), make_item_list({make_string_item("idx_null_mode"),
                             new Item_int(2), make_string_item("euclidean"),
                             make_null_string_value_item()})));
  expect_wrong_arguments_int(new Item_func_vec_index_create(
      POS(), make_item_list({make_string_item("idx_null_provider"),
                             new Item_int(2), make_string_item("euclidean"),
                             make_string_item("memory"),
                             make_null_string_value_item()})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_search_ef(
      POS(), make_null_string_value_item(), new Item_int(16)));
  expect_wrong_arguments_int(new Item_func_vec_index_set_hnsw_build_params(
      POS(), make_item_list({make_null_string_value_item(), new Item_int(16),
                             new Item_int(200)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_faiss_ivf_params(
      POS(), make_item_list({make_null_string_value_item(), new Item_int(2),
                             new Item_int(1)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_faiss_ivfpq_params(
      POS(), make_item_list({make_null_string_value_item(), new Item_int(2),
                             new Item_int(1), new Item_int(2),
                             new Item_int(8)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_diskann_build_params(
      POS(), make_item_list({make_null_string_value_item(), new Item_int(64),
                             new Item_int(100)})));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_search_complexity(
          POS(), make_null_string_value_item(), new Item_int(100)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_pq_code_budget_size(
          POS(), make_null_string_value_item(), new Item_int(0)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_disk_pq_dims(
          POS(), make_null_string_value_item(), new Item_int(12)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_accelerate_build(
          POS(), make_null_string_value_item(), new Item_int(1)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_shuffle_build(
          POS(), make_null_string_value_item(), new Item_int(1)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_use_bfs_cache(
          POS(), make_null_string_value_item(), new Item_int(1)));

  expect_wrong_arguments_int(
      new Item_func_vec_index_drop(POS(),
                                   make_item_list({make_null_string_value_item()})));
  expect_wrong_arguments_int(new Item_func_vec_index_rebuild(
      POS(), make_item_list({make_null_string_value_item()})));
  expect_wrong_arguments_int(new Item_func_vec_index_bulk_load_begin(
      POS(), make_item_list({make_null_string_value_item()})));
  expect_wrong_arguments_int(new Item_func_vec_index_bulk_build(
      POS(), make_item_list({make_null_string_value_item()})));
  expect_wrong_arguments_int(new Item_func_vec_index_recover(
      POS(), make_item_list({make_null_string_value_item()})));
  expect_wrong_arguments_int(new Item_func_vec_index_upsert(
      POS(),
      make_item_list({make_string_item("idx_bad_vector"), new Item_int(1),
                      make_null_string_value_item()})));
  expect_wrong_arguments_int(new Item_func_vec_index_erase(
      POS(),
      make_item_list({make_null_string_value_item(), new Item_int(1)})));
}

TEST_F(ItemVectorFuncFixture, ItemAdminItemsCoverAdditionalInvalidArguments) {
  const longlong uint32_overflow =
      static_cast<longlong>(std::numeric_limits<uint32_t>::max()) + 1LL;
  const longlong build_threads_overflow =
      static_cast<longlong>(vector_index::k_max_build_threads) + 1LL;
  const longlong disk_pq_dims_overflow =
      static_cast<longlong>(vector_index::k_max_diskann_disk_pq_dims) + 1LL;
  auto expect_wrong_arguments_int = [this](Item *item) {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  };

  expect_wrong_arguments_int(new Item_func_vec_index_create(
      POS(), make_item_list({make_string_item("idx_create_null_metric_mark"),
                             new Item_int(2),
                             make_null_marked_string_item("euclidean")})));
  expect_wrong_arguments_int(new Item_func_vec_index_create(
      POS(), make_item_list({make_string_item("idx_create_null_mode_mark"),
                             new Item_int(2), make_string_item("euclidean"),
                             make_null_marked_string_item("memory")})));
  expect_wrong_arguments_int(new Item_func_vec_index_create(
      POS(), make_item_list({make_string_item("idx_create_null_provider_mark"),
                             new Item_int(2), make_string_item("euclidean"),
                             make_string_item("memory"),
                             make_null_marked_string_item("native")})));

  expect_wrong_arguments_int(new Item_func_vec_index_set_hnsw_build_params(
      POS(), make_item_list({make_string_item("idx_any"), new Item_int(16),
                             new Item_int(0)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_hnsw_build_params(
      POS(), make_item_list({make_string_item("idx_any"),
                             new Item_int(uint32_overflow),
                             new Item_int(200)})));

  expect_wrong_arguments_int(new Item_func_vec_index_set_faiss_ivf_params(
      POS(), make_item_list({make_string_item("idx_any"), new Item_int(2),
                             new Item_int(-1)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_faiss_ivf_params(
      POS(), make_item_list({make_string_item("idx_any"), new Item_int(2),
                             new Item_int(uint32_overflow)})));

  expect_wrong_arguments_int(new Item_func_vec_index_set_faiss_ivfpq_params(
      POS(), make_item_list({make_string_item("idx_any"), new Item_int(2),
                             new Item_int(0), new Item_int(2),
                             new Item_int(8)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_faiss_ivfpq_params(
      POS(), make_item_list({make_string_item("idx_any"), new Item_int(2),
                             new Item_int(1), new Item_int(0),
                             new Item_int(8)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_faiss_ivfpq_params(
      POS(), make_item_list({make_string_item("idx_any"), new Item_int(2),
                             new Item_int(1), new Item_int(2),
                             new Item_int(0)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_faiss_ivfpq_params(
      POS(), make_item_list({make_string_item("idx_any"), new Item_int(2),
                             new Item_int(1), new Item_int(2),
                             new Item_int(static_cast<longlong>(
                                 vector_index::k_max_faiss_pq_bits + 1))})));

  expect_wrong_arguments_int(new Item_func_vec_index_set_diskann_build_params(
      POS(), make_item_list({make_string_item("idx_any"), new Item_int(64),
                             new Item_int(0)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_diskann_build_params(
      POS(), make_item_list({make_string_item("idx_any"),
                             new Item_int(uint32_overflow),
                             new Item_int(100)})));

  expect_wrong_arguments_int(new Item_func_vec_index_set_diskann_build_params(
      POS(), make_item_list({make_string_item("idx_any"), new Item_int(64),
                             new Item_int(100), make_null_marked_int(1)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_diskann_build_params(
      POS(), make_item_list({make_string_item("idx_any"), new Item_int(64),
                             new Item_int(100), new Item_int(-1)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_diskann_build_params(
      POS(), make_item_list({make_string_item("idx_any"), new Item_int(64),
                             new Item_int(100),
                             new Item_int(build_threads_overflow)})));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_disk_pq_dims(
          POS(), make_string_item("idx_any"),
          new Item_int(disk_pq_dims_overflow)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_accelerate_build(
          POS(), make_string_item("idx_any"), new Item_int(2)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_shuffle_build(
          POS(), make_string_item("idx_any"), new Item_int(-1)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_use_bfs_cache(
          POS(), make_string_item("idx_any"), new Item_int(2)));

  expect_wrong_arguments_int(new Item_func_vec_index_set_diskann_build_mode(
      POS(), make_null_marked_string_item("idx_any"),
      make_string_item("auto")));
  expect_wrong_arguments_int(new Item_func_vec_index_set_diskann_build_mode(
      POS(), make_string_item("idx_any"), make_null_marked_string_item("auto")));
  expect_wrong_arguments_int(new Item_func_vec_index_set_diskann_build_mode(
      POS(), make_string_item("idx_any"), make_string_item("bad_mode")));
}

TEST_F(ItemVectorFuncFixture, LoadVectorCommandRejectsInvalidUserInputs) {
  const std::string unique_suffix =
      "_" + std::to_string(reinterpret_cast<uintptr_t>(this));
  const std::string transactional_index =
      "idx_load_transactional_cmd" + unique_suffix;
  const std::string standalone_index = "idx_load_standalone_cmd" + unique_suffix;
  auto expect_load_vector_error = [this](bool is_local_file,
                                         const char *vector_file,
                                         const char *docid_file,
                                         const char *index_name,
                                         const char *format,
                                         uint expected_error) {
    auto *command = new (thd()->mem_root) Sql_cmd_load_vector_index(
        is_local_file, make_lex_string(vector_file), make_lex_string(docid_file),
        make_lex_string(index_name), make_lex_string(format), false, false);
    EXPECT_EQ(SQLCOM_LOAD_VECTOR, command->sql_command_code());
    Server_initializer::set_expected_error(expected_error);
    EXPECT_TRUE(command->execute(thd()));
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  };

  expect_load_vector_error(false, "", "", "idx_load_cmd", "FBIN",
                           ER_WRONG_ARGUMENTS);
  expect_load_vector_error(false, "vectors.fbin", "", "", "FBIN",
                           ER_WRONG_ARGUMENTS);
  expect_load_vector_error(false, "vectors.fbin", "", "idx_load_cmd", "",
                           ER_WRONG_ARGUMENTS);
  expect_load_vector_error(false, "vectors.csv", "", "idx_load_cmd", "CSV",
                           ER_WRONG_ARGUMENTS);
  expect_load_vector_error(false, "vectors.fbin", "", "idx_load_missing",
                           "FBIN", ER_WRONG_ARGUMENTS);

  ASSERT_TRUE(vector_index_registry::create_index(
      transactional_index, 2, "euclidean", "memory", "native"));
  {
    FileAccessGuard file_access(thd());
    SecureFilePrivGuard secure_file_priv("");
    expect_load_vector_error(false, "vectors.fbin", "",
                             transactional_index.c_str(), "FBIN",
                             ER_WRONG_ARGUMENTS);
  }

  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(vector_index_registry::create_index(standalone_index, 2,
                                                  "euclidean", "memory",
                                                  "native", options));
  const bool saved_local_infile = opt_local_infile;
  opt_local_infile = false;
  expect_load_vector_error(true, "vectors.fbin", "", standalone_index.c_str(),
                           "FBIN", ER_CLIENT_LOCAL_FILES_DISABLED);
  opt_local_infile = saved_local_infile;
}

TEST_F(ItemVectorFuncFixture, LoadVectorCommandImportsRawFiles) {
  const std::string unique_suffix =
      "_" + std::to_string(reinterpret_cast<uintptr_t>(this));
  const std::string index_name = "idx_load_success_cmd" + unique_suffix;
  const std::string root =
      std::string(testing::TempDir()) + "/load_vector_command_success" +
      unique_suffix;
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);

  const std::string vector_path = root + "/vectors.fbin";
  const std::string docid_path = root + "/vectors.u64";
  write_raw_fbin_file(vector_path, 2, 2, {1.0F, 0.0F, 0.0F, 1.0F});
  write_raw_docid_file(docid_path, {101, 202});

  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(vector_index_registry::create_index(
      index_name, 2, "euclidean", "memory", "native", options));

  FileAccessGuard file_access(thd());
  SecureFilePrivGuard secure_file_priv("");
  auto *command = new (thd()->mem_root) Sql_cmd_load_vector_index(
      false, make_lex_string(vector_path.c_str()),
      make_lex_string(docid_path.c_str()), make_lex_string(index_name.c_str()),
      make_lex_string("FBIN"), false, true);
  EXPECT_EQ(SQLCOM_LOAD_VECTOR, command->sql_command_code());
  EXPECT_FALSE(command->execute(thd()));
  EXPECT_FALSE(thd()->is_error());

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(vector_index_registry::search(index_name, {1.0F, 0.0F}, 2,
                                           &results));
  ASSERT_EQ(2U, results.size());
  EXPECT_EQ(101U, results[0].doc_id);

  std::filesystem::remove_all(root, ec);
}

TEST_F(ItemVectorFuncFixture, UpsertBatchCoversStandaloneReaderEdges) {
  const std::string index_name =
      "idx_upsert_batch_item_" +
      std::to_string(reinterpret_cast<uintptr_t>(this));
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(vector_index_registry::create_index(
      index_name, 2, "euclidean", "memory", "native", options));

  const std::string docids = binary_docid_payload({11, 22});
  const std::string vectors = binary_vector_payload({1.0F, 0.0F, 0.0F, 1.0F});
  auto *success_item = new Item_func_vec_index_upsert_batch(
      POS(), make_item_list({make_string_item(index_name.c_str()),
                             make_binary_blob_item(docids),
                             make_binary_blob_item(vectors), new Item_uint(2)}));
  fix_item(thd(), success_item);
  EXPECT_EQ(2, success_item->val_int());
  EXPECT_FALSE(success_item->null_value);

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));
  ASSERT_TRUE(vector_index_registry::search(index_name, {1.0F, 0.0F}, 2,
                                           &results));
  ASSERT_EQ(2U, results.size());
  EXPECT_EQ(11U, results[0].doc_id);

  auto expect_wrong_arguments = [this](Item *item) {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  };

  expect_wrong_arguments(new Item_func_vec_index_upsert_batch(
      POS(), make_item_list({make_string_item(index_name.c_str()),
                             make_binary_blob_item(docids),
                             make_binary_blob_item(vectors), new Item_uint(0)})));
  expect_wrong_arguments(new Item_func_vec_index_upsert_batch(
      POS(), make_item_list({make_string_item(index_name.c_str()),
                             make_binary_blob_item(docids),
                             make_binary_blob_item(vectors), new Item_uint(3)})));
  expect_wrong_arguments(new Item_func_vec_index_upsert_batch(
      POS(), make_item_list({make_string_item(index_name.c_str()),
                             make_null_marked_string_item("docids"),
                             make_binary_blob_item(vectors), new Item_uint(2)})));

  const std::string non_finite_vectors = binary_vector_payload(
      {std::numeric_limits<float>::quiet_NaN(), 0.0F});
  expect_wrong_arguments(new Item_func_vec_index_upsert_batch(
      POS(), make_item_list({make_string_item(index_name.c_str()),
                             make_binary_blob_item(binary_docid_payload({33})),
                             make_binary_blob_item(non_finite_vectors),
                             new Item_uint(1)})));

  thd()->killed = THD::KILL_QUERY;
  expect_wrong_arguments(new Item_func_vec_index_upsert_batch(
      POS(), make_item_list({make_string_item(index_name.c_str()),
                             make_binary_blob_item(binary_docid_payload({44})),
                             make_binary_blob_item(binary_vector_payload(
                                 {4.0F, 4.0F})),
                             new Item_uint(1)})));
  thd()->killed = THD::NOT_KILLED;
}

TEST_F(ItemVectorFuncFixture, ItemTuningItemsRejectUnsupportedBackends) {
  const std::string native_index =
      "idx_item_tuning_native_" +
      std::to_string(reinterpret_cast<uintptr_t>(this));
  const std::string diskann_index = native_index + "_diskann";
  ASSERT_TRUE(vector_index_registry::create_index(native_index, 2, "euclidean",
                                                 "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(diskann_index, 2, "euclidean",
                                                 "external", "diskann"));

  auto expect_wrong_arguments_int = [this](Item *item) {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  };

  expect_wrong_arguments_int(new Item_func_vec_index_set_search_ef(
      POS(), make_string_item(native_index.c_str()), new Item_int(16)));
  expect_wrong_arguments_int(new Item_func_vec_index_set_hnsw_build_params(
      POS(), make_item_list({make_string_item(native_index.c_str()),
                             new Item_int(16), new Item_int(200)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_faiss_ivf_params(
      POS(), make_item_list({make_string_item(native_index.c_str()),
                             new Item_int(2), new Item_int(1)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_faiss_ivfpq_params(
      POS(), make_item_list({make_string_item(native_index.c_str()),
                             new Item_int(2), new Item_int(1),
                             new Item_int(2), new Item_int(8)})));
  expect_wrong_arguments_int(new Item_func_vec_index_set_diskann_build_params(
      POS(), make_item_list({make_string_item(native_index.c_str()),
                             new Item_int(64), new Item_int(100)})));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_search_complexity(
          POS(), make_string_item(native_index.c_str()), new Item_int(100)));
  expect_wrong_arguments_int(
      new Item_func_vec_index_set_diskann_pq_code_budget_size(
          POS(), make_string_item(native_index.c_str()), new Item_int(1048576)));

  auto *native_auto_mode = new Item_func_vec_index_set_diskann_build_mode(
      POS(), make_string_item(native_index.c_str()), make_string_item("auto"));
  fix_item(thd(), native_auto_mode);
  EXPECT_EQ(1, native_auto_mode->val_int());
  EXPECT_FALSE(native_auto_mode->null_value);

  auto *native_serial_mode = new Item_func_vec_index_set_diskann_build_mode(
      POS(), make_string_item(native_index.c_str()), make_string_item("serial"));
  fix_item(thd(), native_serial_mode);
  EXPECT_EQ(0, native_serial_mode->val_int());
  EXPECT_FALSE(native_serial_mode->null_value);

  auto *diskann_serial_mode = new Item_func_vec_index_set_diskann_build_mode(
      POS(), make_string_item(diskann_index.c_str()),
      make_string_item("serial"));
  fix_item(thd(), diskann_serial_mode);
  EXPECT_EQ(1, diskann_serial_mode->val_int());
  EXPECT_FALSE(diskann_serial_mode->null_value);
}

TEST_F(ItemVectorFuncFixture, VectorBinaryItemsCoverSuccessAndErrorPaths) {
  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_totext(POS(), make_string_item("bad"));
    fix_item(thd(), item);
    String out;
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    auto *item = new Item_func_vec_totext(POS(), make_binary_vector_item({1.0F, 2.0F}));
    fix_item(thd(), item);
    String out;
    String *result = item->val_str(&out);
    ASSERT_NE(nullptr, result);
    EXPECT_EQ("[1,2]", as_std_string(*result));
    EXPECT_FALSE(item->null_value);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_normalize(POS(), make_binary_vector_item({0.0F, 0.0F}));
    fix_item(thd(), item);
    String out;
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    auto *item =
        new Item_func_vec_normalize(POS(), make_binary_vector_item({3.0F, 4.0F}));
    fix_item(thd(), item);
    String out;
    String *result = item->val_str(&out);
    ASSERT_NE(nullptr, result);
    ASSERT_EQ(2U * vector_utils::kVectorElemSize, result->length());
    const uchar *ptr = reinterpret_cast<const uchar *>(result->ptr());
    EXPECT_NEAR(0.6, float4get(ptr), 1e-5);
    EXPECT_NEAR(0.8, float4get(ptr + vector_utils::kVectorElemSize), 1e-5);
    EXPECT_FALSE(item->null_value);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vector_dim(POS(), make_string_item("bad"));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    auto *item =
        new Item_func_vector_dim(POS(), make_binary_vector_item({1.0F, 2.0F, 3.0F}));
    fix_item(thd(), item);
    EXPECT_EQ(3, item->val_int());
    EXPECT_FALSE(item->null_value);
  }
}

TEST_F(ItemVectorFuncFixture, VectorFromTextCoversSuccessNullAndInjectedFailure) {
  {
    auto *item = new Item_func_vec_fromtext(POS(), new Item_null());
    fix_item(thd(), item);
    String out;
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_fromtext(POS(), make_string_item("bad"));
    fix_item(thd(), item);
    String out;
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    auto *item = new Item_func_vec_fromtext(POS(), make_string_item("[]"));
    fix_item(thd(), item);
    String out;
    String *result = item->val_str(&out);
    ASSERT_NE(nullptr, result);
    EXPECT_EQ(0U, result->length());
    EXPECT_FALSE(item->null_value);
  }

  {
    auto *item = new Item_func_vec_fromtext(POS(), make_string_item("[1,2.5]"));
    fix_item(thd(), item);
    String out;
    String *result = item->val_str(&out);
    ASSERT_NE(nullptr, result);
    ASSERT_EQ(2U * vector_utils::kVectorElemSize, result->length());
    const uchar *ptr = reinterpret_cast<const uchar *>(result->ptr());
    EXPECT_NEAR(1.0, float4get(ptr), 1e-6);
    EXPECT_NEAR(2.5, float4get(ptr + vector_utils::kVectorElemSize), 1e-6);
    EXPECT_FALSE(item->null_value);
  }

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug_flag, "+d,vector_item_fail_fromtext_alloc");
    auto *item =
        new Item_func_vec_fromtext(POS(), make_string_item("[3,4,5]"));
    fix_item(thd(), item);
    String out;
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
  }
}

TEST_F(ItemVectorFuncFixture, IndexInfoAndListItemsCoverSuccessAndErrorPaths) {
  auto *list_item = new Item_func_vec_index_list(POS());
  fix_item(thd(), list_item);
  String out;
  String *result = list_item->val_str(&out);
  ASSERT_NE(nullptr, result);
  EXPECT_FALSE(as_std_string(*result).empty());
  EXPECT_FALSE(list_item->null_value);

  const std::string index_name_a =
      "idx_item_info_" + std::to_string(reinterpret_cast<uintptr_t>(this));
  const std::string index_name_b = index_name_a + "_b";
  ASSERT_TRUE(vector_index_registry::create_index(index_name_a, 2, "euclidean",
                                                 "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(index_name_b, 2, "euclidean",
                                                 "memory", "native"));

  result = list_item->val_str(&out);
  ASSERT_NE(nullptr, result);
  const std::string list_json = as_std_string(*result);
  EXPECT_NE(std::string::npos, list_json.find(index_name_a));
  EXPECT_NE(std::string::npos, list_json.find(index_name_b));

  auto *info_item =
      new Item_func_vec_index_info(POS(),
                                   make_item_list({make_string_item(index_name_a.c_str())}));
  fix_item(thd(), info_item);
  result = info_item->val_str(&out);
  ASSERT_NE(nullptr, result);
  const std::string info_json = as_std_string(*result);
  EXPECT_NE(std::string::npos, info_json.find("\"dimension\":2"));
  EXPECT_NE(std::string::npos, info_json.find("\"metric\":\"euclidean\""));

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *missing_item = new Item_func_vec_index_info(
        POS(), make_item_list({make_string_item("idx_missing_info")}));
    fix_item(thd(), missing_item);
    EXPECT_EQ(nullptr, missing_item->val_str(&out));
    EXPECT_TRUE(missing_item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }
}

TEST_F(ItemVectorFuncFixture, SearchItemsCoverErrorAndSuccessPaths) {
  const std::string index_name =
      "idx_item_search_" + std::to_string(reinterpret_cast<uintptr_t>(this));
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                 "memory", "native"));
  ASSERT_TRUE(
      vector_index_registry::upsert(index_name, 11, {1.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::upsert(index_name, 22, {2.0F, 2.0F}));

  String out;

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_search(
        POS(), make_item_list({make_string_item(index_name.c_str()),
                               make_string_item("bad"), new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_search(
        POS(), make_item_list({make_string_item("idx_missing_search"),
                               make_binary_vector_item({1.0F, 1.0F}),
                               new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    auto *item = new Item_func_vec_index_search(
        POS(), make_item_list({make_string_item(index_name.c_str()),
                               make_binary_vector_item({1.0F, 1.0F}),
                               new Item_int(0)}));
    fix_item(thd(), item);
    String *result = item->val_str(&out);
    ASSERT_NE(nullptr, result);
    EXPECT_EQ("[]", as_std_string(*result));
    EXPECT_FALSE(item->null_value);
  }

  {
    auto *item = new Item_func_vec_index_search(
        POS(), make_item_list({make_string_item(index_name.c_str()),
                               make_binary_vector_item({1.0F, 1.0F}),
                               new Item_int(2)}));
    fix_item(thd(), item);
    String *result = item->val_str(&out);
    ASSERT_NE(nullptr, result);
    const std::string json = as_std_string(*result);
    EXPECT_NE(std::string::npos, json.find("11"));
    EXPECT_FALSE(item->null_value);
  }

  {
    auto *item = new Item_func_vec_index_search_with_distance(
        POS(), make_item_list({make_string_item(index_name.c_str()),
                               make_binary_vector_item({1.0F, 1.0F}),
                               new Item_int(2)}));
    fix_item(thd(), item);
    String *result = item->val_str(&out);
    ASSERT_NE(nullptr, result);
    const std::string json = as_std_string(*result);
    EXPECT_NE(std::string::npos, json.find("\"doc_id\":11"));
    EXPECT_NE(std::string::npos, json.find("\"distance\":"));
    EXPECT_FALSE(item->null_value);
  }
}

TEST_F(ItemVectorFuncFixture, SearchBatchItemCoversValidationAndSuccessPaths) {
  const std::string index_name =
      "idx_item_search_batch_" +
      std::to_string(reinterpret_cast<uintptr_t>(this));
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                 "memory", "native"));
  ASSERT_TRUE(
      vector_index_registry::upsert(index_name, 11, {1.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::upsert(index_name, 22, {2.0F, 2.0F}));

  auto expect_wrong_arguments_str = [this](Item *item) {
    String out;
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  };

  expect_wrong_arguments_str(new Item_func_vec_index_search_batch(
      POS(),
      make_item_list({new Item_null(), make_binary_vector_item({1.0F, 1.0F}),
                      new Item_int(1), new Item_int(1)})));
  expect_wrong_arguments_str(new Item_func_vec_index_search_batch(
      POS(),
      make_item_list({make_string_item(index_name.c_str()),
                      make_binary_vector_item({1.0F, 1.0F}), new Item_int(0),
                      new Item_int(1)})));
  expect_wrong_arguments_str(new Item_func_vec_index_search_batch(
      POS(),
      make_item_list({make_string_item(index_name.c_str()),
                      make_binary_vector_item({1.0F, 1.0F}), new Item_int(1),
                      new Item_int(-1)})));
  expect_wrong_arguments_str(new Item_func_vec_index_search_batch(
      POS(),
      make_item_list({make_string_item(index_name.c_str()), new Item_null(),
                      new Item_int(1), new Item_int(1)})));
  expect_wrong_arguments_str(new Item_func_vec_index_search_batch(
      POS(),
      make_item_list({make_string_item(index_name.c_str()),
                      make_string_item("", &my_charset_bin), new Item_int(1),
                      new Item_int(1)})));
  expect_wrong_arguments_str(new Item_func_vec_index_search_batch(
      POS(),
      make_item_list({make_string_item(index_name.c_str()),
                      make_string_item("abc", &my_charset_bin),
                      new Item_int(2), new Item_int(1)})));
  expect_wrong_arguments_str(new Item_func_vec_index_search_batch(
      POS(),
      make_item_list({make_string_item("idx_missing_search_batch"),
                      make_binary_vector_item({1.0F, 1.0F}), new Item_int(1),
                      new Item_int(1)})));
  expect_wrong_arguments_str(new Item_func_vec_index_search_batch(
      POS(),
      make_item_list({make_null_string_value_item(),
                      make_binary_vector_item({1.0F, 1.0F}), new Item_int(1),
                      new Item_int(1)})));
  expect_wrong_arguments_str(new Item_func_vec_index_search_batch(
      POS(),
      make_item_list({make_null_marked_string_item(index_name.c_str()),
                      make_binary_vector_item({1.0F, 1.0F}), new Item_int(1),
                      new Item_int(1)})));
  expect_wrong_arguments_str(new Item_func_vec_index_search_batch(
      POS(),
      make_item_list({make_string_item(index_name.c_str()),
                      make_binary_vector_item({1.0F, 1.0F}),
                      make_null_marked_int(1), new Item_int(1)})));
  expect_wrong_arguments_str(new Item_func_vec_index_search_batch(
      POS(),
      make_item_list({make_string_item(index_name.c_str()),
                      make_binary_vector_item({1.0F, 1.0F}), new Item_int(1),
                      make_null_marked_int(1)})));
  expect_wrong_arguments_str(new Item_func_vec_index_search_batch(
      POS(),
      make_item_list({make_string_item(index_name.c_str()),
                      make_binary_vector_item({1.0F, 1.0F}), new Item_int(1),
                      new Item_int(-1)})));

  {
    auto *item = new Item_func_vec_index_search_batch(
        POS(),
        make_item_list({make_string_item(index_name.c_str()),
                        make_binary_vector_item({1.0F, 1.0F, 2.0F, 2.0F}),
                        new Item_int(2), new Item_int(0)}));
    fix_item(thd(), item);
    String out;
    String *result = item->val_str(&out);
    ASSERT_NE(nullptr, result);
    EXPECT_EQ("[[],[]]", as_std_string(*result));
    EXPECT_FALSE(item->null_value);
  }

  {
    auto *item = new Item_func_vec_index_search_batch(
        POS(),
        make_item_list({make_string_item(index_name.c_str()),
                        make_binary_vector_item({1.0F, 1.0F, 2.0F, 2.0F}),
                        new Item_int(2), new Item_int(1)}));
    fix_item(thd(), item);
    String out;
    String *result = item->val_str(&out);
    ASSERT_NE(nullptr, result);
    const std::string json = as_std_string(*result);
    EXPECT_NE(std::string::npos, json.find("[[11]"));
    EXPECT_NE(std::string::npos, json.find("[22]"));
    EXPECT_FALSE(item->null_value);
  }
}

TEST_F(ItemVectorFuncFixture, VectorTextAndDimensionItemsHandleNullAndErrors) {
  {
    auto *item = new Item_func_vec_totext(POS(), new Item_null());
    fix_item(thd(), item);
    String out;
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_totext(POS(), make_string_item("bad"));
    fix_item(thd(), item);
    String out;
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    auto *item = new Item_func_vec_normalize(POS(), new Item_null());
    fix_item(thd(), item);
    String out;
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_normalize(POS(), make_string_item("bad"));
    fix_item(thd(), item);
    String out;
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    auto *item = new Item_func_vector_dim(POS(), new Item_null());
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_TRUE(item->null_value);
  }

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug_flag, "+d,vector_item_fail_totext_number_format");
    auto *item =
        new Item_func_vec_totext(POS(), make_binary_vector_item({1.0F, 2.0F}));
    fix_item(thd(), item);
    String out;
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
  }

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug_flag, "+d,vector_item_fail_normalize_alloc");
    auto *item =
        new Item_func_vec_normalize(POS(), make_binary_vector_item({3.0F, 4.0F}));
    fix_item(thd(), item);
    String out;
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
  }
}

TEST_F(ItemVectorFuncFixture, VectorDistanceItemsCoverSuccessAndErrorPaths) {
  Item *lhs = make_binary_vector_item({1.0F, 2.0F});
  Item *rhs = make_binary_vector_item({4.0F, 6.0F});
  Item *bad = make_string_item("bad");
  Item *short_vec = make_binary_vector_item({1.0F});

  {
    auto *item = new Item_func_vec_dot_product(POS(), lhs, rhs);
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(16.0, item->val_real());
    EXPECT_FALSE(item->null_value);
  }

  {
    auto *item = new Item_func_vec_inner_product(POS(), lhs, rhs);
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(16.0, item->val_real());
    EXPECT_FALSE(item->null_value);
  }

  {
    auto *item = new Item_func_vec_distance_euclidean(POS(), lhs, rhs);
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(5.0, item->val_real());
    EXPECT_FALSE(item->null_value);
  }

  {
    auto *item = new Item_func_vec_distance_cosine(POS(), lhs, lhs);
    fix_item(thd(), item);
    EXPECT_NEAR(0.0, item->val_real(), 1e-6);
    EXPECT_FALSE(item->null_value);
  }

  {
    auto *item = new Item_func_vec_distance(
        POS(), make_item_list({lhs, rhs, make_string_item("euclidean")}));
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(5.0, item->val_real());
    EXPECT_FALSE(item->null_value);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_dot_product(POS(), lhs, short_vec);
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(0.0, item->val_real());
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_inner_product(POS(), lhs, bad);
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(0.0, item->val_real());
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_distance_cosine(POS(), lhs,
                                                   make_binary_vector_item({0.0F, 0.0F}));
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(0.0, item->val_real());
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_distance(
        POS(), make_item_list({lhs, rhs, make_string_item("not_a_metric")}));
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(0.0, item->val_real());
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }
}

TEST_F(ItemVectorFuncFixture, VectorDistanceItemsReturnNullForNullOperands) {
  {
    auto *item = new Item_func_vec_dot_product(POS(), new Item_null(),
                                               make_binary_vector_item({1.0F, 2.0F}));
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(0.0, item->val_real());
    EXPECT_TRUE(item->null_value);
  }
  {
    auto *item = new Item_func_vec_inner_product(
        POS(), make_binary_vector_item({1.0F, 2.0F}), new Item_null());
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(0.0, item->val_real());
    EXPECT_TRUE(item->null_value);
  }
  {
    auto *item = new Item_func_vec_distance_euclidean(
        POS(), new Item_null(), make_binary_vector_item({1.0F, 2.0F}));
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(0.0, item->val_real());
    EXPECT_TRUE(item->null_value);
  }
  {
    auto *item = new Item_func_vec_distance_cosine(
        POS(), make_binary_vector_item({1.0F, 2.0F}), new Item_null());
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(0.0, item->val_real());
    EXPECT_TRUE(item->null_value);
  }
  {
    auto *item = new Item_func_vec_distance(
        POS(), make_item_list({new Item_null(),
                               make_binary_vector_item({1.0F, 2.0F}),
                               make_string_item("euclidean")}));
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(0.0, item->val_real());
    EXPECT_TRUE(item->null_value);
  }
}

TEST_F(ItemVectorFuncFixture, VectorItemsHonorNullMarkedArguments) {
  String out;

  {
    auto *item = new Item_func_vec_fromtext(
        POS(), make_null_marked_string_item("[1,2]"));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
  }

  {
    auto *item = new Item_func_vec_totext(
        POS(), make_null_marked_binary_vector_item({1.0F, 2.0F}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
  }

  {
    auto *item = new Item_func_vec_normalize(
        POS(), make_null_marked_binary_vector_item({3.0F, 4.0F}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
  }

  {
    auto *item = new Item_func_vector_dim(
        POS(), make_null_marked_binary_vector_item({1.0F, 2.0F}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_TRUE(item->null_value);
  }

  {
    auto *item = new Item_func_vec_dot_product(
        POS(), make_binary_vector_item({1.0F, 2.0F}),
        make_null_marked_binary_vector_item({1.0F, 2.0F}));
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(0.0, item->val_real());
    EXPECT_TRUE(item->null_value);
  }

  {
    auto *item = new Item_func_vec_distance(
        POS(),
        make_item_list({make_binary_vector_item({1.0F, 2.0F}),
                        make_binary_vector_item({2.0F, 3.0F}),
                        make_null_marked_string_item("euclidean")}));
    fix_item(thd(), item);
    EXPECT_DOUBLE_EQ(0.0, item->val_real());
    EXPECT_TRUE(item->null_value);
  }
}

TEST_F(ItemVectorFuncFixture, IndexItemsHonorNullMarkedArguments) {
  String out;

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_info(
        POS(), make_item_list({make_null_marked_string_item("idx_any")}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_search(
        POS(), make_item_list({make_null_marked_string_item("idx_any"),
                               make_binary_vector_item({1.0F, 1.0F}),
                               new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_search(
        POS(), make_item_list({make_string_item("idx_any"),
                               make_null_marked_binary_vector_item({1.0F, 1.0F}),
                               new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_search_with_distance(
        POS(), make_item_list({make_string_item("idx_any"),
                               make_binary_vector_item({1.0F, 1.0F}),
                               make_null_marked_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_search_with_distance(
        POS(), make_item_list({make_string_item("idx_any"),
                               make_null_marked_binary_vector_item({1.0F, 1.0F}),
                               new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_stage_upsert(
        POS(), make_item_list({make_null_marked_string_item("idx_any"),
                               new Item_int(1), new Item_int(1),
                               make_binary_vector_item({1.0F, 1.0F})}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_stage_upsert(
        POS(), make_item_list({make_string_item("idx_any"),
                               make_null_marked_int(1), new Item_int(1),
                               make_binary_vector_item({1.0F, 1.0F})}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_stage_upsert(
        POS(), make_item_list({make_string_item("idx_any"),
                               new Item_int(1), make_null_marked_int(1),
                               make_binary_vector_item({1.0F, 1.0F})}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_stage_upsert(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(1),
                               new Item_int(1),
                               make_null_marked_binary_vector_item({1.0F, 1.0F})}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_stage_erase(
        POS(), make_item_list({make_null_marked_string_item("idx_any"),
                               new Item_int(1), new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_stage_erase(
        POS(), make_item_list({make_string_item("idx_any"),
                               make_null_marked_int(1), new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_stage_erase(
        POS(), make_item_list({make_string_item("idx_any"), new Item_int(1),
                               make_null_marked_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }
}

TEST_F(ItemVectorFuncFixture, TxnItemsHonorNullMarkedArguments) {
  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_pending(
        POS(), make_item_list({make_null_marked_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_commit(
        POS(), make_item_list({make_null_marked_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_rollback(
        POS(), make_item_list({make_null_marked_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_savepoint(
        POS(), make_item_list({make_null_marked_int(1), make_string_item("sp")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_savepoint(
        POS(), make_item_list({new Item_int(1),
                               make_null_marked_string_item("sp")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_rollback_to(
        POS(), make_item_list({new Item_int(1),
                               make_null_marked_string_item("sp")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_release_savepoint(
        POS(), make_item_list({new Item_int(1),
                               make_null_marked_string_item("sp")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }
}

TEST_F(ItemVectorFuncFixture, DebugTruthStoreItemsValidateArgumentsWithPrivilege) {
  ProcessAccessGuard process_access(thd());
  String out;

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_debug_truth_store_get_hex(
        POS(), make_item_list({make_string_item("")}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_debug_truth_store_set_hex(
        POS(), make_item_list({make_string_item(""), make_string_item("00")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  for (const char *hex : {"0", "0g", "ag", "Ag"}) {
    SCOPED_TRACE(hex);
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_debug_truth_store_set_hex(
        POS(), make_item_list({make_string_item("vector_debug_payload"),
                               make_string_item(hex)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }
}

TEST_F(ItemVectorFuncFixture, DebugTruthStoreItemsRejectAccessAndNullArguments) {
  String out;

  {
    Server_initializer::set_expected_error(ER_SPECIFIC_ACCESS_DENIED_ERROR);
    auto *item = new Item_func_vec_debug_truth_store_get_hex(
        POS(), make_item_list({make_string_item("vector_debug_payload")}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_SPECIFIC_ACCESS_DENIED_ERROR);
    auto *item = new Item_func_vec_debug_truth_store_set_hex(
        POS(), make_item_list({make_string_item("vector_debug_payload"),
                               make_string_item("00")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  ProcessAccessGuard process_access(thd());

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_debug_truth_store_get_hex(
        POS(), make_item_list({make_null_string_value_item()}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_debug_truth_store_get_hex(
        POS(), make_item_list({make_null_marked_string_item("artifact")}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_debug_truth_store_set_hex(
        POS(), make_item_list({make_null_string_value_item(),
                               make_string_item("00")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_debug_truth_store_set_hex(
        POS(), make_item_list({make_null_marked_string_item("artifact"),
                               make_string_item("00")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_debug_truth_store_set_hex(
        POS(), make_item_list({make_string_item("artifact"),
                               make_null_string_value_item()}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }
}

TEST_F(ItemVectorFuncFixture, RegistryBackedItemsPropagateMetadataLoadFailures) {
  controlled_truth_store store;
  store.fail_load_metadata = true;
  store.fail_quarantine_metadata = true;
  vector_index_truth_store::set_for_testing(&store);
  vector_index_registry::reset_for_testing();

  String out;

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_list(POS());
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_info(
        POS(), make_item_list({make_string_item("idx_meta_fail")}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_search(
        POS(), make_item_list({make_string_item("idx_meta_fail"),
                               make_binary_vector_item({1.0F, 1.0F}),
                               new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_search_with_distance(
        POS(), make_item_list({make_string_item("idx_meta_fail"),
                               make_binary_vector_item({1.0F, 1.0F}),
                               new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(nullptr, item->val_str(&out));
    EXPECT_TRUE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_rebuild_all(POS());
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_recover_all(POS());
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_pending(
        POS(), make_item_list({new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_commit(
        POS(), make_item_list({new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_rollback(
        POS(), make_item_list({new Item_int(1)}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_savepoint(
        POS(), make_item_list({new Item_int(1), make_string_item("sp1")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_rollback_to(
        POS(), make_item_list({new Item_int(1), make_string_item("sp1")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  {
    Server_initializer::set_expected_error(ER_WRONG_ARGUMENTS);
    auto *item = new Item_func_vec_index_txn_release_savepoint(
        POS(), make_item_list({new Item_int(1), make_string_item("sp1")}));
    fix_item(thd(), item);
    EXPECT_EQ(0, item->val_int());
    EXPECT_FALSE(item->null_value);
    thd()->clear_error();
    Server_initializer::set_expected_error(0);
  }

  vector_index_truth_store::reset_for_testing();
}

TEST(ItemVectorFuncTest, FormatterHelpersPropagateInjectedNumberFailures) {
  vector_index_registry::index_info info = make_base_info();
  String out;
  String number_buf;
  out.set_charset(&my_charset_bin);

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug_flag, "+d,vector_item_fail_append_json_number");
    EXPECT_FALSE(format_vector_index_info_json(info, &out, &number_buf));
    EXPECT_FALSE(
        format_vector_search_result_doc_ids({{7, 0.0}}, &out, &number_buf));
  }

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug_flag, "+d,vector_item_fail_append_json_real");
    EXPECT_FALSE(format_vector_search_results_with_distance(
        {{7, 1.5}}, &out, &number_buf));
  }
}

}  // namespace item_vectorfunc_unittest
