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
#include <atomic>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "my_alloc.h"
#include "sql/mysqld.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_ddl_formatter.h"
#include "sql/vector/vector_index_identity.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_metadata_store.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_registry_internal.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_status.h"
#include "sql_string.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_index_registry_unittest {

namespace {

vector_index::index_service::index_config make_backend_config(
    vector_index::backend_mode mode, vector_index::backend_provider provider) {
  vector_index::index_service::index_config config;
  config.dimension = 2;
  config.metric = vector_index::metric_type::kEuclidean;
  config.mode = mode;
  config.provider = provider;
  return config;
}

bool hnswlib_tuning_supported() {
  vector_index::hnswlib_backend backend(2,
                                        vector_index::metric_type::kEuclidean,
                                        vector_index::backend_mode::kMemory);
  return backend.set_search_ef(64) && backend.set_hnsw_build_params(16, 200);
}

vector_index_metadata_store::metadata_row make_metadata_row_for_testing(
    const std::string &index_name, const std::string &lifecycle_state) {
  vector_index_metadata_store::metadata_row row;
  row.index_name = index_name;
  row.dimension = 2;
  row.metric = vector_index::metric_type::kEuclidean;
  row.mode = vector_index::backend_mode::kMemory;
  row.provider = vector_index::backend_provider::kNative;
  row.lifecycle_state = lifecycle_state;
  row.lifecycle_version = lifecycle_state == "ready" ? 1 : 2;
  return row;
}

vector_index_registry::index_info make_formatter_info_for_testing(
    const std::string &provider) {
  vector_index_registry::index_info info;
  info.dimension = 2;
  info.metric = "euclidean";
  info.mode = "memory";
  info.provider = provider;
  info.search_ef = 64;
  info.hnsw_m = 16;
  info.hnsw_ef_construction = 200;
  info.diskann_max_degree = 32;
  info.diskann_build_complexity = 64;
  info.diskann_search_complexity = 64;
  return info;
}

std::string string_value_for_testing(const String &value) {
  return value.length() == 0 ? std::string()
                             : std::string(value.ptr(), value.length());
}

}  // namespace

TEST(VectorIndexDdlFormatterTest, BuildThreadSelectionFollowsProvider) {
  auto info = make_formatter_info_for_testing("hnswlib");
  info.hnsw_build_threads = 3;
  info.faiss_build_threads = 5;
  info.diskann_build_threads = 7;
  EXPECT_EQ(3U, vector_index_ddl_formatter::provider_build_threads(info));

  info.provider = "FAISS";
  EXPECT_EQ(5U, vector_index_ddl_formatter::provider_build_threads(info));

  info.provider = "diskann";
  EXPECT_EQ(7U, vector_index_ddl_formatter::provider_build_threads(info));

  info.provider = "native";
  EXPECT_EQ(0U, vector_index_ddl_formatter::provider_build_threads(info));

  EXPECT_EQ(3U, vector_index_ddl_formatter::first_nonzero_build_threads(info));
  info.hnsw_build_threads = 0;
  EXPECT_EQ(5U, vector_index_ddl_formatter::first_nonzero_build_threads(info));
  info.faiss_build_threads = 0;
  EXPECT_EQ(7U, vector_index_ddl_formatter::first_nonzero_build_threads(info));
  info.diskann_build_threads = 0;
  EXPECT_EQ(0U, vector_index_ddl_formatter::first_nonzero_build_threads(info));
}

TEST(VectorIndexDdlFormatterTest, CreateStatementHonorsQualificationAndThreads) {
  auto info = make_formatter_info_for_testing("faiss");
  String query;

  vector_index_ddl_formatter::append_create_statement(
      nullptr, &query, "db", 2, "t", 1, true, "v", info, 8);
  EXPECT_EQ(";\nCREATE VECTOR INDEX ON `db`.`t`(`v`) WITH "
            "(2, 'euclidean', 'memory', 'faiss', 8)",
            string_value_for_testing(query));

  query.length(0);
  vector_index_ddl_formatter::append_create_statement(
      nullptr, &query, "db", 2, "table`name", 10, false, "vec`col", info, 0);
  EXPECT_EQ(";\nCREATE VECTOR INDEX ON `table``name`(`vec``col`) WITH "
            "(2, 'euclidean', 'memory', 'faiss')",
            string_value_for_testing(query));
}

TEST(VectorIndexDdlFormatterTest, TuningStatementsSkipProviderDefaults) {
  String query;

  auto hnsw = make_formatter_info_for_testing("hnswlib");
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_hnsw", hnsw,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_TRUE(string_value_for_testing(query).empty());

  hnsw.search_ef = 128;
  hnsw.hnsw_m = 32;
  hnsw.hnsw_ef_construction = 400;
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_hnsw", hnsw,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_SEARCH_EF('idx_hnsw', 128)"
            ";\nSELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS('idx_hnsw', 32, 400)",
            string_value_for_testing(query));

  auto faiss = make_formatter_info_for_testing("faiss");
  faiss.faiss_nlist = 64;
  faiss.faiss_nprobe = 8;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss", faiss,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_FAISS_IVF_PARAMS('idx_faiss', 64, 8)",
            string_value_for_testing(query));

  faiss.faiss_pq_m = 16;
  faiss.faiss_pq_bits = 8;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss", faiss,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_FAISS_IVFPQ_PARAMS"
            "('idx_faiss', 64, 8, 16, 8)",
            string_value_for_testing(query));

  auto diskann = make_formatter_info_for_testing("diskann");
  diskann.diskann_max_degree = 48;
  diskann.diskann_build_complexity = 96;
  diskann.diskann_build_threads = 4;
  diskann.diskann_search_complexity = 96;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_diskann", diskann,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_DISKANN_BUILD_PARAMS"
            "('idx_diskann', 48, 96, 4)"
            ";\nSELECT VEC_INDEX_SET_DISKANN_SEARCH_COMPLEXITY"
            "('idx_diskann', 96)",
            string_value_for_testing(query));
}

TEST(VectorIndexDdlFormatterTest, TuningStatementsCanEmitAllNonZeroValues) {
  String query;
  vector_index_registry::index_info info;
  info.provider = "native";

  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, nullptr, "idx", info,
      vector_index_ddl_formatter::tuning_emit_policy::k_emit_nonzero);

  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx", info,
      vector_index_ddl_formatter::tuning_emit_policy::k_emit_nonzero);
  EXPECT_TRUE(string_value_for_testing(query).empty());

  info.search_ef = 11;
  info.hnsw_m = 12;
  info.hnsw_ef_construction = 13;
  info.faiss_nlist = 14;
  info.faiss_nprobe = 15;
  info.faiss_pq_m = 16;
  info.faiss_pq_bits = 17;
  info.diskann_max_degree = 18;
  info.diskann_build_complexity = 19;
  info.diskann_build_threads = 21;
  info.diskann_search_complexity = 20;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx", info,
      vector_index_ddl_formatter::tuning_emit_policy::k_emit_nonzero);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_SEARCH_EF('idx', 11)"
            ";\nSELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS('idx', 12, 13)"
            ";\nSELECT VEC_INDEX_SET_FAISS_IVFPQ_PARAMS"
            "('idx', 14, 15, 16, 17)"
            ";\nSELECT VEC_INDEX_SET_DISKANN_BUILD_PARAMS('idx', 18, 19, 21)"
            ";\nSELECT VEC_INDEX_SET_DISKANN_SEARCH_COMPLEXITY('idx', 20)",
            string_value_for_testing(query));
}

TEST(VectorIndexDdlFormatterTest, TuningStatementsCoverTrailingOptionBranches) {
  String query;

  auto hnsw = make_formatter_info_for_testing("hnswlib");
  hnsw.hnsw_m = 32;
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_hnsw_m", hnsw,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS"
            "('idx_hnsw_m', 32, 200)",
            string_value_for_testing(query));

  query.length(0);
  hnsw = make_formatter_info_for_testing("hnswlib");
  hnsw.hnsw_ef_construction = 400;
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_hnsw_ef", hnsw,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS"
            "('idx_hnsw_ef', 16, 400)",
            string_value_for_testing(query));

  auto faiss = make_formatter_info_for_testing("faiss");
  faiss.faiss_pq_bits = 8;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss_pq_bits", faiss,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_FAISS_IVFPQ_PARAMS"
            "('idx_faiss_pq_bits', 0, 0, 0, 8)",
            string_value_for_testing(query));

  faiss = make_formatter_info_for_testing("faiss");
  faiss.faiss_nprobe = 8;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss_nprobe", faiss,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_FAISS_IVF_PARAMS"
            "('idx_faiss_nprobe', 0, 8)",
            string_value_for_testing(query));

  faiss = make_formatter_info_for_testing("faiss");
  faiss.faiss_nlist = 64;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss_nlist", faiss,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_FAISS_IVF_PARAMS"
            "('idx_faiss_nlist', 64, 0)",
            string_value_for_testing(query));

  faiss = make_formatter_info_for_testing("faiss");
  faiss.faiss_pq_m = 8;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss_pq_m", faiss,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_FAISS_IVFPQ_PARAMS"
            "('idx_faiss_pq_m', 0, 0, 8, 0)",
            string_value_for_testing(query));

  auto diskann = make_formatter_info_for_testing("diskann");
  diskann.diskann_build_complexity = 96;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_diskann_complexity", diskann,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_DISKANN_BUILD_PARAMS"
            "('idx_diskann_complexity', 32, 96)",
            string_value_for_testing(query));

  diskann = make_formatter_info_for_testing("diskann");
  diskann.diskann_build_threads = 4;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_diskann_threads", diskann,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_DISKANN_BUILD_PARAMS"
            "('idx_diskann_threads', 32, 64, 4)",
            string_value_for_testing(query));
}

class in_memory_truth_store final
    : public vector_index_truth_store::truth_store {
 public:
  const char *backend_name() const override { return "memory"; }
  bool is_transactional() const override { return transactional; }
  bool supports_delta_persist() const override {
    return transactional && delta_persist_supported;
  }

  bool begin_persist() override {
    ++begin_persist_calls;
    return !fail_begin_persist;
  }

  bool commit_persist() override {
    ++commit_persist_calls;
    return !fail_commit_persist;
  }

  void rollback_persist() override { ++rollback_persist_calls; }

  bool load_metadata(
      std::vector<vector_index_metadata_store::metadata_row> *rows) override {
    if (fail_load_metadata) return false;
    if (rows == nullptr) return false;
    *rows = metadata_rows;
    return true;
  }

  bool save_metadata(
      const std::vector<vector_index_metadata_store::metadata_row> &rows)
      override {
    ++save_metadata_calls;
    if (fail_save_metadata) return false;
    metadata_rows = rows;
    return true;
  }

  bool quarantine_metadata() override {
    if (fail_quarantine_metadata) return false;
    metadata_rows.clear();
    return true;
  }

  bool load_committed(
      std::vector<vector_index_metadata_store::committed_row> *rows) override {
    if (fail_load_committed) return false;
    if (rows == nullptr) return false;
    *rows = committed_rows;
    return true;
  }

  bool save_committed(
      const std::vector<vector_index_metadata_store::committed_row> &rows)
      override {
    ++save_committed_calls;
    if (fail_save_committed) return false;
    committed_rows = rows;
    return true;
  }

  bool apply_committed_delta(
      const std::vector<vector_index_metadata_store::change_log_row> &rows)
      override {
    ++apply_committed_delta_calls;
    if (fail_apply_committed_delta) return false;

    for (const auto &row : rows) {
      auto it = std::find_if(committed_rows.begin(), committed_rows.end(),
                             [&](const auto &entry) {
                               return entry.index_name == row.index_name &&
                                      entry.doc_id == row.doc_id;
                             });
      if (row.op == vector_index_metadata_store::change_op::kErase) {
        if (it != committed_rows.end()) committed_rows.erase(it);
        continue;
      }
      if (it == committed_rows.end()) {
        committed_rows.push_back(vector_index_metadata_store::committed_row{
            row.index_name, row.doc_id, row.vector});
      } else {
        it->vector = row.vector;
      }
    }
    return true;
  }

  bool quarantine_committed() override {
    if (fail_quarantine_committed) return false;
    committed_rows.clear();
    return true;
  }

  bool load_manifest(vector_index_metadata_store::manifest_row *row) override {
    if (fail_load_manifest) return false;
    if (row == nullptr) return false;
    *row = manifest_row;
    return true;
  }

  bool save_manifest(
      const vector_index_metadata_store::manifest_row &row) override {
    ++save_manifest_calls;
    if (fail_save_manifest_once) {
      fail_save_manifest_once = false;
      return false;
    }
    if (fail_save_manifest) return false;
    manifest_row = row;
    return true;
  }

  bool quarantine_manifest() override {
    if (fail_quarantine_manifest) return false;
    manifest_row = vector_index_metadata_store::manifest_row();
    return true;
  }

  bool load_change_log(
      std::vector<vector_index_metadata_store::change_log_row> *rows) override {
    if (fail_load_change_log) return false;
    if (rows == nullptr) return false;
    *rows = change_log_rows;
    return true;
  }

  bool save_change_log(
      const std::vector<vector_index_metadata_store::change_log_row> &rows)
      override {
    ++save_change_log_calls;
    if (fail_save_change_log) return false;
    change_log_rows = rows;
    return true;
  }

  bool append_change_log_delta(
      const std::vector<vector_index_metadata_store::change_log_row> &rows)
      override {
    ++append_change_log_delta_calls;
    if (fail_append_change_log_delta) return false;
    change_log_rows.insert(change_log_rows.end(), rows.begin(), rows.end());
    return true;
  }

  bool quarantine_change_log() override {
    if (fail_quarantine_change_log) return false;
    change_log_rows.clear();
    return true;
  }

  bool load_prepared(
      std::vector<vector_index_metadata_store::prepared_change_row> *rows)
      override {
    if (fail_load_prepared) return false;
    if (rows == nullptr) return false;
    *rows = prepared_rows;
    return true;
  }

  bool save_prepared(
      const std::vector<vector_index_metadata_store::prepared_change_row> &rows)
      override {
    ++save_prepared_calls;
    if (fail_save_prepared) return false;
    prepared_rows = rows;
    return true;
  }

  bool quarantine_prepared() override {
    if (fail_quarantine_prepared) return false;
    prepared_rows.clear();
    return true;
  }

  bool fail_load_metadata{false};
  bool fail_save_metadata{false};
  bool fail_quarantine_metadata{false};
  bool fail_load_committed{false};
  bool fail_save_committed{false};
  bool fail_apply_committed_delta{false};
  bool fail_quarantine_committed{false};
  bool fail_load_manifest{false};
  bool fail_save_manifest{false};
  bool fail_save_manifest_once{false};
  bool fail_quarantine_manifest{false};
  bool fail_load_change_log{false};
  bool fail_save_change_log{false};
  bool fail_append_change_log_delta{false};
  bool fail_quarantine_change_log{false};
  bool fail_load_prepared{false};
  bool fail_save_prepared{false};
  bool fail_quarantine_prepared{false};
  bool fail_begin_persist{false};
  bool fail_commit_persist{false};
  bool transactional{true};
  bool delta_persist_supported{true};
  uint64_t begin_persist_calls{0};
  uint64_t commit_persist_calls{0};
  uint64_t rollback_persist_calls{0};
  uint64_t save_metadata_calls{0};
  uint64_t save_committed_calls{0};
  uint64_t apply_committed_delta_calls{0};
  uint64_t save_change_log_calls{0};
  uint64_t append_change_log_delta_calls{0};
  uint64_t save_prepared_calls{0};
  uint64_t save_manifest_calls{0};

  void ResetPersistCountersForTesting() {
    save_metadata_calls = 0;
    save_committed_calls = 0;
    apply_committed_delta_calls = 0;
    save_change_log_calls = 0;
    append_change_log_delta_calls = 0;
    save_prepared_calls = 0;
    save_manifest_calls = 0;
  }

  void ClearCommittedRowsForTesting() { committed_rows.clear(); }

  void SetMetadataRowsForTesting(
      const std::vector<vector_index_metadata_store::metadata_row> &rows) {
    metadata_rows = rows;
  }

  void SetCommittedRowsForTesting(
      const std::vector<vector_index_metadata_store::committed_row> &rows) {
    committed_rows = rows;
  }

  void SetChangeLogRowsForTesting(
      const std::vector<vector_index_metadata_store::change_log_row> &rows) {
    change_log_rows = rows;
  }

  void SetManifestCheckpointsForTesting(uint64_t metadata_checkpoint,
                                        uint64_t committed_checkpoint,
                                        uint64_t change_log_checkpoint) {
    manifest_row.metadata_checkpoint = metadata_checkpoint;
    manifest_row.committed_checkpoint = committed_checkpoint;
    manifest_row.change_log_checkpoint = change_log_checkpoint;
  }

  void SetPreparedRowsForTesting(
      const std::vector<vector_index_metadata_store::prepared_change_row>
          &rows) {
    prepared_rows = rows;
  }

 private:
  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  vector_index_metadata_store::manifest_row manifest_row;
  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
};

TEST(VectorIndexRegistryInternalTest,
     IndexConfigMatchesComparesEveryTuningField) {
  namespace detail = vector_index_registry::detail;

  vector_index::index_service::index_config base =
      make_backend_config(vector_index::backend_mode::kExternal,
                          vector_index::backend_provider::kDiskAnn);
  base.backend_variant = "diskann-native";
  base.search_ef = 32;
  base.hnsw_m = 16;
  base.hnsw_ef_construction = 200;
  base.hnsw_build_threads = 2;
  base.faiss_nlist = 64;
  base.faiss_nprobe = 8;
  base.faiss_pq_m = 4;
  base.faiss_pq_bits = 8;
  base.faiss_build_threads = 2;
  base.diskann_max_degree = 32;
  base.diskann_build_complexity = 96;
  base.diskann_build_threads = 3;
  base.diskann_search_complexity = 80;

  EXPECT_TRUE(detail::index_config_matches_for_testing(base, base));

  auto expect_mismatch = [&](auto mutate) {
    vector_index::index_service::index_config changed = base;
    mutate(&changed);
    EXPECT_FALSE(detail::index_config_matches_for_testing(base, changed));
  };

  expect_mismatch([](auto *config) { config->dimension = 3; });
  expect_mismatch(
      [](auto *config) { config->metric = vector_index::metric_type::kCosine; });
  expect_mismatch(
      [](auto *config) { config->mode = vector_index::backend_mode::kMemory; });
  expect_mismatch([](auto *config) {
    config->provider = vector_index::backend_provider::kFaiss;
  });
  expect_mismatch([](auto *config) { config->search_ef = 33; });
  expect_mismatch([](auto *config) { config->hnsw_m = 17; });
  expect_mismatch([](auto *config) { config->hnsw_ef_construction = 201; });
  expect_mismatch([](auto *config) { config->hnsw_build_threads = 4; });
  expect_mismatch([](auto *config) { config->faiss_nlist = 65; });
  expect_mismatch([](auto *config) { config->faiss_nprobe = 9; });
  expect_mismatch([](auto *config) { config->faiss_pq_m = 5; });
  expect_mismatch([](auto *config) { config->faiss_pq_bits = 9; });
  expect_mismatch([](auto *config) { config->faiss_build_threads = 3; });
  expect_mismatch([](auto *config) { config->diskann_max_degree = 33; });
  expect_mismatch([](auto *config) { config->diskann_build_complexity = 97; });
  expect_mismatch([](auto *config) { config->diskann_build_threads = 4; });
  expect_mismatch([](auto *config) { config->diskann_search_complexity = 81; });
}

class VectorIndexRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    m_prepared_path = std::string(DATA_DIR) + "/vector_registry_prepared_t.dat";
    vector_index_metadata_store::set_path_for_testing(m_prepared_path);
    std::remove((m_prepared_path + ".prepared").c_str());
    std::remove((m_prepared_path + ".prepared.corrupt").c_str());
    vector_index_truth_store::set_for_testing(&store_);
    vector_index_registry::reset_for_testing();
  }

  void TearDown() override {
    vector_index_registry::reset_for_testing();
    vector_index_truth_store::reset_for_testing();
    vector_index_metadata_store::reset_path_for_testing();
    std::remove((m_prepared_path + ".prepared").c_str());
    std::remove((m_prepared_path + ".prepared.corrupt").c_str());
  }

  in_memory_truth_store store_;
  std::string m_prepared_path;
};

namespace {

void expect_metadata_only_persist(const in_memory_truth_store &store) {
  EXPECT_EQ(1U, store.save_metadata_calls);
  EXPECT_EQ(1U, store.save_manifest_calls);
  EXPECT_EQ(0U, store.save_committed_calls);
  EXPECT_EQ(0U, store.save_change_log_calls);
  EXPECT_EQ(0U, store.save_prepared_calls);
}

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

class UlongGuard {
 public:
  UlongGuard(ulong *value, ulong replacement)
      : m_value(value), m_original(*value) {
    *m_value = replacement;
  }

  ~UlongGuard() { *m_value = m_original; }

 private:
  ulong *m_value;
  ulong m_original;
};

class UlonglongGuard {
 public:
  UlonglongGuard(ulonglong *value, ulonglong replacement)
      : m_value(value), m_original(*value) {
    *m_value = replacement;
  }

  ~UlonglongGuard() { *m_value = m_original; }

 private:
  ulonglong *m_value;
  ulonglong m_original;
};

XID make_test_xid(long format_id, const char *data) {
  XID xid;
  xid.set_format_id(format_id);
  xid.set_gtrid_length(static_cast<long>(std::strlen(data)));
  xid.set_bqual_length(0);
  xid.set_data(data, static_cast<long>(std::strlen(data)));
  return xid;
}

}  // namespace

TEST_F(VectorIndexRegistryTest,
       CommitRollbackRestoresServingStateWhenPersistFails) {
  const std::string index_name = "idx_registry_persist";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  store_.fail_apply_committed_delta = true;
  EXPECT_FALSE(vector_index_registry::upsert(index_name, 2, {2.0F, 2.0F}));
  store_.fail_apply_committed_delta = false;

  ASSERT_TRUE(
      vector_index_registry::search(index_name, {2.0F, 2.0F}, 5, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(1U, info.entry_count);
  EXPECT_EQ(1U, info.committed_entry_count);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnUsesTransactionalTruthStoreDeltaPersist) {
  const std::string index_name = "idx_registry_delta_persist";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  store_.ResetPersistCountersForTesting();
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::commit_txn(txn_id));

  EXPECT_EQ(0U, store_.save_committed_calls);
  EXPECT_EQ(1U, store_.apply_committed_delta_calls);
  EXPECT_EQ(0U, store_.save_change_log_calls);
  EXPECT_EQ(1U, store_.append_change_log_delta_calls);

  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  ASSERT_TRUE(store_.load_committed(&committed_rows));
  EXPECT_EQ(2U, committed_rows.size());

  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  ASSERT_TRUE(store_.load_change_log(&change_log_rows));
  EXPECT_EQ(2U, change_log_rows.size());

  vector_index_metadata_store::manifest_row manifest;
  ASSERT_TRUE(store_.load_manifest(&manifest));
  EXPECT_EQ(2U, manifest.committed_checkpoint);
  EXPECT_EQ(2U, manifest.change_log_checkpoint);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnFallsBackWhenTransactionalTruthStoreLacksDeltaPersist) {
  store_.delta_persist_supported = false;
  const std::string index_name = "idx_registry_delta_fallback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  store_.ResetPersistCountersForTesting();
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::commit_txn(txn_id));

  EXPECT_EQ(1U, store_.save_committed_calls);
  EXPECT_EQ(0U, store_.apply_committed_delta_calls);
  EXPECT_EQ(1U, store_.save_change_log_calls);
  EXPECT_EQ(0U, store_.append_change_log_delta_calls);

  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  ASSERT_TRUE(store_.load_committed(&committed_rows));
  EXPECT_EQ(2U, committed_rows.size());

  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  ASSERT_TRUE(store_.load_change_log(&change_log_rows));
  EXPECT_EQ(2U, change_log_rows.size());

  vector_index_metadata_store::manifest_row manifest;
  ASSERT_TRUE(store_.load_manifest(&manifest));
  EXPECT_EQ(2U, manifest.committed_checkpoint);
  EXPECT_EQ(2U, manifest.change_log_checkpoint);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, SetSearchEfPersistsOnlyMetadata) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  const std::string index_name = "idx_registry_search_ef_metadata_only";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "hnswlib"));

  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 0.0F}));
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 2, {0.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::commit_txn(txn_id));

  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(vector_index_registry::set_search_ef(index_name, 96));

  expect_metadata_only_persist(store_);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(96U, info.search_ef);
  EXPECT_EQ(2U, info.committed_entry_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       SetDiskAnnSearchComplexityPersistsOnlyMetadata) {
  const std::string index_name = "idx_registry_diskann_search_metadata_only";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "external", "diskann"));

  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(
      vector_index_registry::set_diskann_search_complexity(index_name, 144));

  expect_metadata_only_persist(store_);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(144U, info.diskann_search_complexity);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, SetDiskAnnBuildThreadsPersistsOnlyMetadata) {
  const std::string index_name = "idx_registry_diskann_threads_metadata_only";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "external", "diskann"));

  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(vector_index_registry::set_diskann_build_threads(index_name, 3));

  expect_metadata_only_persist(store_);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(3U, info.diskann_build_threads);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, CommitTxnCompactsChangeLogAfterDebugThreshold) {
  vector_index_registry::set_change_log_compact_threshold_for_testing(3);
  const std::string index_name = "idx_registry_changelog_compact";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  store_.ResetPersistCountersForTesting();
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 3, {3.0F, 3.0F}));
  ASSERT_TRUE(vector_index_registry::commit_txn(txn_id));

  EXPECT_EQ(1U, store_.apply_committed_delta_calls);
  EXPECT_EQ(1U, store_.append_change_log_delta_calls);
  EXPECT_EQ(1U, store_.save_change_log_calls);

  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  ASSERT_TRUE(store_.load_committed(&committed_rows));
  EXPECT_EQ(3U, committed_rows.size());

  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  ASSERT_TRUE(store_.load_change_log(&change_log_rows));
  EXPECT_TRUE(change_log_rows.empty());

  vector_index_metadata_store::manifest_row manifest;
  ASSERT_TRUE(store_.load_manifest(&manifest));
  EXPECT_EQ(3U, manifest.committed_checkpoint);
  EXPECT_EQ(0U, manifest.change_log_checkpoint);

  vector_index_registry::reset_for_testing();
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {3.0F, 3.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(3U, result[0].doc_id);
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnRestoresChangeLogRowsWhenCompactionThenManifestFails) {
  vector_index_registry::set_change_log_compact_threshold_for_testing(3);
  const std::string index_name = "idx_registry_compact_manifest_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  const uint64_t baseline_txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(vector_index_registry::stage_upsert(
      baseline_txn_id, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::commit_txn(baseline_txn_id));

  std::vector<vector_index_metadata_store::metadata_row> metadata_before;
  vector_index::index_service::committed_state committed_before;
  std::vector<vector_index_metadata_store::change_log_row> change_log_before;
  std::vector<std::string> lagging_before;
  ASSERT_TRUE(vector_index_registry::snapshot_runtime_state_for_testing(
      &metadata_before, &committed_before, &change_log_before,
      &lagging_before));
  ASSERT_EQ(1U, change_log_before.size());
  EXPECT_EQ(index_name, change_log_before[0].index_name);
  EXPECT_EQ(1U, change_log_before[0].doc_id);

  const uint64_t failing_txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(vector_index_registry::stage_upsert(
      failing_txn_id, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::stage_upsert(
      failing_txn_id, index_name, 3, {3.0F, 3.0F}));

  store_.fail_save_manifest_once = true;
  EXPECT_FALSE(vector_index_registry::commit_txn(failing_txn_id));

  std::vector<vector_index_metadata_store::metadata_row> metadata_after;
  vector_index::index_service::committed_state committed_after;
  std::vector<vector_index_metadata_store::change_log_row> change_log_after;
  std::vector<std::string> lagging_after;
  ASSERT_TRUE(vector_index_registry::snapshot_runtime_state_for_testing(
      &metadata_after, &committed_after, &change_log_after, &lagging_after));
  ASSERT_EQ(change_log_before.size(), change_log_after.size());
  EXPECT_EQ(change_log_before[0].sequence, change_log_after[0].sequence);
  EXPECT_EQ(change_log_before[0].txn_id, change_log_after[0].txn_id);
  EXPECT_EQ(change_log_before[0].index_name, change_log_after[0].index_name);
  EXPECT_EQ(change_log_before[0].doc_id, change_log_after[0].doc_id);
  EXPECT_EQ(change_log_before[0].vector, change_log_after[0].vector);

  ASSERT_TRUE(vector_index_registry::rollback_txn(failing_txn_id));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, LifecycleHelpersNoopBeforeMetadataLoad) {
  EXPECT_FALSE(vector_index_registry::metadata_loaded());
  EXPECT_TRUE(
      vector_index_registry::drop_indexes_for_table("db_unloaded", "t1"));
  EXPECT_TRUE(vector_index_registry::drop_indexes_for_database("db_unloaded"));
  EXPECT_TRUE(vector_index_registry::reset_mapped_indexes_for_table(
      "db_unloaded", "t1"));
  EXPECT_TRUE(vector_index_registry::rename_indexes_for_table(
      "db_unloaded", "t1", "db_loaded", "t1"));
  EXPECT_TRUE(
      vector_index_registry::drop_index_for_column("db_unloaded", "t1", "v"));
  EXPECT_TRUE(vector_index_registry::rename_index_for_column("db_unloaded",
                                                             "t1", "v", "v2"));
  EXPECT_FALSE(vector_index_registry::metadata_loaded());
}

TEST_F(VectorIndexRegistryTest, MetadataLoadedTracksLazyLoad) {
  EXPECT_FALSE(vector_index_registry::metadata_loaded());

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));

  EXPECT_TRUE(vector_index_registry::metadata_loaded());
  EXPECT_TRUE(index_names.empty());
}

TEST_F(VectorIndexRegistryTest,
       LazyLoadRemovesIncompleteCreateLifecycleRows) {
  const std::string ready_index = "test.t_ready.v";
  const std::string creating_index = "test.t_creating.v";
  const std::string backfilling_index = "test.t_backfilling.v";

  store_.SetMetadataRowsForTesting({
      make_metadata_row_for_testing(ready_index, "ready"),
      make_metadata_row_for_testing(creating_index, "creating"),
      make_metadata_row_for_testing(backfilling_index, "backfilling"),
  });
  store_.SetCommittedRowsForTesting({
      vector_index_metadata_store::committed_row{ready_index, 1, {1.0F, 1.0F}},
      vector_index_metadata_store::committed_row{creating_index, 2,
                                                 {2.0F, 2.0F}},
      vector_index_metadata_store::committed_row{backfilling_index, 3,
                                                 {3.0F, 3.0F}},
  });
  store_.SetChangeLogRowsForTesting({
      vector_index_metadata_store::change_log_row{
          1, 100, vector_index_metadata_store::change_op::kUpsert,
          ready_index, 1, {1.5F, 1.5F}},
      vector_index_metadata_store::change_log_row{
          2, 101, vector_index_metadata_store::change_op::kUpsert,
          creating_index, 4, {4.0F, 4.0F}},
      vector_index_metadata_store::change_log_row{
          3, 102, vector_index_metadata_store::change_op::kUpsert,
          backfilling_index, 5, {5.0F, 5.0F}},
  });
  store_.SetPreparedRowsForTesting({
      vector_index_metadata_store::prepared_change_row{
          42, 3, 0, "abc", false, 200,
          vector_index_metadata_store::change_op::kUpsert, ready_index, 6,
          {6.0F, 6.0F}},
      vector_index_metadata_store::prepared_change_row{
          43, 3, 0, "def", false, 201,
          vector_index_metadata_store::change_op::kUpsert, creating_index, 7,
          {7.0F, 7.0F}},
      vector_index_metadata_store::prepared_change_row{
          44, 3, 0, "ghi", false, 202,
          vector_index_metadata_store::change_op::kUpsert, backfilling_index, 8,
          {8.0F, 8.0F}},
  });
  store_.SetManifestCheckpointsForTesting(3, 3, 3);

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));

  ASSERT_EQ(1U, index_names.size());
  EXPECT_EQ(ready_index, index_names[0]);

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  ASSERT_TRUE(store_.load_metadata(&metadata_rows));
  ASSERT_EQ(1U, metadata_rows.size());
  EXPECT_EQ(ready_index, metadata_rows[0].index_name);
  EXPECT_EQ("ready", metadata_rows[0].lifecycle_state);

  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  ASSERT_TRUE(store_.load_committed(&committed_rows));
  ASSERT_EQ(1U, committed_rows.size());
  EXPECT_EQ(ready_index, committed_rows[0].index_name);

  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  ASSERT_TRUE(store_.load_change_log(&change_log_rows));
  ASSERT_EQ(1U, change_log_rows.size());
  EXPECT_EQ(ready_index, change_log_rows[0].index_name);

  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  ASSERT_TRUE(store_.load_prepared(&prepared_rows));
  ASSERT_EQ(1U, prepared_rows.size());
  EXPECT_EQ(ready_index, prepared_rows[0].index_name);

  vector_index_metadata_store::manifest_row manifest;
  ASSERT_TRUE(store_.load_manifest(&manifest));
  EXPECT_EQ(1U, manifest.metadata_checkpoint);
  EXPECT_EQ(1U, manifest.committed_checkpoint);
  EXPECT_EQ(1U, manifest.change_log_checkpoint);
}

TEST_F(VectorIndexRegistryTest,
       LazyLoadKeepsMetadataUnloadedWhenIncompleteCleanupPersistFails) {
  store_.SetMetadataRowsForTesting({
      make_metadata_row_for_testing("test.t_creating.v", "creating"),
  });
  store_.fail_save_metadata = true;

  std::vector<std::string> index_names;
  EXPECT_FALSE(vector_index_registry::list_indexes(&index_names));
  EXPECT_FALSE(vector_index_registry::metadata_loaded());
}

TEST_F(VectorIndexRegistryTest, GetIndexInfoParsesMappedIndexName) {
  const std::string mapped_index_name = "test.t_vec_status_meta.v";
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      mapped_index_name, 2, "euclidean", "memory", "native", "test",
      "t_vec_status_meta", "v", "id"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(mapped_index_name, &info));
  EXPECT_EQ("test", info.schema_name);
  EXPECT_EQ("t_vec_status_meta", info.table_name);
  EXPECT_EQ("v", info.column_name);

  EXPECT_FALSE(vector_index_registry::get_index_info("idx_standalone", &info));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_standalone", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::get_index_info("idx_standalone", &info));
  EXPECT_TRUE(info.schema_name.empty());
  EXPECT_TRUE(info.table_name.empty());
  EXPECT_TRUE(info.column_name.empty());

  ASSERT_TRUE(vector_index_registry::create_index(
      "standalone.with.dots", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(
      vector_index_registry::get_index_info("standalone.with.dots", &info));
  EXPECT_TRUE(info.schema_name.empty());
  EXPECT_TRUE(info.table_name.empty());
  EXPECT_TRUE(info.column_name.empty());
}

TEST_F(VectorIndexRegistryTest, MappedIndexLifecycleHelpersMutateLoadedMetadata) {
  const std::string index_v1 = "db_lifecycle.t_lifecycle.v1";
  const std::string index_v2 = "db_lifecycle.t_lifecycle.v2";
  const std::string other_index = "db_other.t_lifecycle.v3";
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      index_v1, 2, "euclidean", "memory", "native", "db_lifecycle",
      "t_lifecycle", "v1", "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      index_v2, 2, "euclidean", "memory", "native", "db_lifecycle",
      "t_lifecycle", "v2", "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      other_index, 2, "euclidean", "memory", "native", "db_other",
      "t_lifecycle", "v3", "id"));
  ASSERT_TRUE(vector_index_registry::upsert(index_v1, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::upsert(index_v2, 2, {2.0F, 2.0F}));

  ASSERT_TRUE(vector_index_registry::reset_mapped_indexes_for_table(
      "db_lifecycle", "t_lifecycle"));
  ASSERT_TRUE(vector_index_registry::rename_indexes_for_table(
      "db_lifecycle", "t_lifecycle", "db_lifecycle_new",
      "t_lifecycle_new"));

  vector_index_registry::index_info info;
  EXPECT_FALSE(vector_index_registry::get_index_info(index_v1, &info));
  const std::string renamed_v1 = "db_lifecycle_new.t_lifecycle_new.v1";
  ASSERT_TRUE(vector_index_registry::get_index_info(renamed_v1, &info));
  EXPECT_EQ("db_lifecycle_new", info.schema_name);
  EXPECT_EQ("t_lifecycle_new", info.table_name);
  EXPECT_EQ("v1", info.column_name);

  ASSERT_TRUE(vector_index_registry::rename_index_for_column(
      "db_lifecycle_new", "t_lifecycle_new", "v1", "v1_new"));
  const std::string renamed_column =
      "db_lifecycle_new.t_lifecycle_new.v1_new";
  EXPECT_FALSE(vector_index_registry::get_index_info(renamed_v1, &info));
  ASSERT_TRUE(vector_index_registry::get_index_info(renamed_column, &info));
  EXPECT_EQ("v1_new", info.column_name);

  ASSERT_TRUE(vector_index_registry::drop_index_for_column(
      "db_lifecycle_new", "t_lifecycle_new", "v2"));
  EXPECT_FALSE(vector_index_registry::get_index_info(
      "db_lifecycle_new.t_lifecycle_new.v2", &info));

  ASSERT_TRUE(vector_index_registry::drop_indexes_for_table(
      "db_lifecycle_new", "t_lifecycle_new"));
  EXPECT_FALSE(vector_index_registry::get_index_info(renamed_column, &info));
  ASSERT_TRUE(vector_index_registry::get_index_info(other_index, &info));

  ASSERT_TRUE(vector_index_registry::drop_indexes_for_database("db_other"));
  EXPECT_FALSE(vector_index_registry::get_index_info(other_index, &info));
}

TEST_F(VectorIndexRegistryTest, MappedIndexHelpersNoopWhenMetadataIsUnloaded) {
  vector_index_registry::detail::g_metadata_loaded = false;

  EXPECT_TRUE(vector_index_registry::drop_indexes_for_table("db_none",
                                                            "t_none"));
  EXPECT_TRUE(vector_index_registry::drop_indexes_for_database("db_none"));
  EXPECT_TRUE(vector_index_registry::reset_mapped_indexes_for_table("db_none",
                                                                    "t_none"));
  EXPECT_TRUE(vector_index_registry::rename_indexes_for_table(
      "db_none", "t_none", "db_new", "t_new"));
  EXPECT_TRUE(vector_index_registry::drop_index_for_column("db_none", "t_none",
                                                           "v"));
  EXPECT_TRUE(vector_index_registry::rename_index_for_column("db_none", "t_none",
                                                             "v", "v"));
  EXPECT_TRUE(vector_index_registry::rename_index_for_column(
      "db_none", "t_none", "v_old", "v_new"));
}

TEST_F(VectorIndexRegistryTest,
       InternalBindingAndTuningHelpersCoverFallbackBranches) {
  {
    std::lock_guard<std::shared_mutex> guard(
        vector_index_registry::detail::g_registry_mutex);
    vector_index_registry::detail::set_index_binding_locked(
        "plain_binding", "db_original", "t_original", "v_original", "id");
    vector_index_registry::detail::rename_index_binding_locked(
        "plain_binding", "plain_binding_renamed");
    const vector_index_registry::detail::index_binding binding =
        vector_index_registry::detail::binding_for_index_locked(
            "plain_binding_renamed");
    EXPECT_EQ("db_original", binding.schema_name);
    EXPECT_EQ("t_original", binding.table_name);
    EXPECT_EQ("v_original", binding.column_name);

    const std::string mapped_index_name =
        vector_index_registry::detail::make_mapped_index_name("db_new",
                                                              "t_new",
                                                              "v_new");
    vector_index_registry::detail::set_index_binding_locked(
        "plain_binding_with_mapped_target", "db_original", "t_original",
        "v_original", "id");
    vector_index_registry::detail::rename_index_binding_locked(
        "plain_binding_with_mapped_target", mapped_index_name);
    const vector_index_registry::detail::index_binding mapped_binding =
        vector_index_registry::detail::binding_for_index_locked(
            mapped_index_name);
    EXPECT_EQ("db_new", mapped_binding.schema_name);
    EXPECT_EQ("t_new", mapped_binding.table_name);
    EXPECT_EQ("v_new", mapped_binding.column_name);

    std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
    EXPECT_FALSE(vector_index_registry::detail::snapshot_metadata_locked(
        nullptr));
    EXPECT_TRUE(vector_index_registry::detail::snapshot_metadata_locked(
        &metadata_rows));
  }

  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_apply_tuning_faiss", 2, "euclidean", "external", "faiss"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_apply_tuning_diskann", 2, "euclidean", "external", "diskann"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_apply_tuning_native", 2, "euclidean", "memory", "native"));

  {
    std::lock_guard<std::shared_mutex> guard(
        vector_index_registry::detail::g_registry_mutex);
    vector_index_registry::index_info info;
    info.search_ef = 64;
    EXPECT_TRUE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_faiss", info));

    info = vector_index_registry::index_info{};
    info.hnsw_m = 16;
    info.hnsw_ef_construction = 200;
    EXPECT_TRUE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_faiss", info));

    info = vector_index_registry::index_info{};
    info.search_ef = 64;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_native", info));

    info = vector_index_registry::index_info{};
    info.hnsw_ef_construction = 200;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_faiss", info));

    info = vector_index_registry::index_info{};
    info.hnsw_build_threads = 65536;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_faiss", info));

    info = vector_index_registry::index_info{};
    info.faiss_build_threads = 65536;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_faiss", info));

    info = vector_index_registry::index_info{};
    info.faiss_nlist = 2;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_faiss", info));

    info = vector_index_registry::index_info{};
    info.faiss_nprobe = 1;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_faiss", info));

    info = vector_index_registry::index_info{};
    info.faiss_nlist = 2;
    info.faiss_nprobe = 1;
    info.faiss_pq_m = 2;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_faiss", info));

    info = vector_index_registry::index_info{};
    info.faiss_pq_bits = 8;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_faiss", info));

    info = vector_index_registry::index_info{};
    info.diskann_max_degree = 32;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_diskann", info));

    info = vector_index_registry::index_info{};
    info.diskann_build_complexity = 96;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_diskann", info));

    info = vector_index_registry::index_info{};
    info.diskann_build_threads = 65536;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_diskann", info));

    info = vector_index_registry::index_info{};
    info.diskann_search_complexity = 80;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_native", info));

    info = vector_index_registry::index_info{};
    info.diskann_search_complexity = 80;
    EXPECT_TRUE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_diskann", info));
  }

  ASSERT_TRUE(vector_index_registry::drop_index("idx_apply_tuning_faiss"));
  ASSERT_TRUE(vector_index_registry::drop_index("idx_apply_tuning_diskann"));
  ASSERT_TRUE(vector_index_registry::drop_index("idx_apply_tuning_native"));
}

TEST_F(VectorIndexRegistryTest, CreateIndexFailsWhenMetadataPersistFails) {
  store_.fail_save_metadata = true;
  EXPECT_FALSE(vector_index_registry::create_index(
      "idx_registry_meta_fail", 2, "euclidean", "memory", "native"));
  store_.fail_save_metadata = false;

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  EXPECT_TRUE(index_names.empty());
}

TEST_F(VectorIndexRegistryTest,
       CreateIndexUsesTransactionalTruthStorePersistHooks) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_tx_hooks", 2, "euclidean", "memory", "native"));
  EXPECT_EQ(1U, store_.begin_persist_calls);
  EXPECT_EQ(1U, store_.commit_persist_calls);
  EXPECT_EQ(0U, store_.rollback_persist_calls);
}

TEST_F(VectorIndexRegistryTest,
       CreateIndexRollsBackTransactionalTruthStoreOnPersistFailure) {
  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::create_index(
      "idx_registry_tx_fail", 2, "euclidean", "memory", "native"));
  EXPECT_EQ(1U, store_.begin_persist_calls);
  EXPECT_EQ(0U, store_.commit_persist_calls);
  EXPECT_EQ(1U, store_.rollback_persist_calls);
}

TEST_F(VectorIndexRegistryTest, DropIndexFailsWhenManifestPersistFails) {
  const std::string index_name = "idx_registry_manifest_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::drop_index(index_name));
  store_.fail_save_manifest = false;

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  ASSERT_EQ(1U, index_names.size());
  EXPECT_EQ(index_name, index_names[0]);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, CommitTxnFailsWhenChangeLogPersistFails) {
  const std::string index_name = "idx_registry_commit_changelog_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));

  store_.fail_append_change_log_delta = true;
  EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  store_.fail_append_change_log_delta = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, CommitTxnFailsWhenManifestPersistFails) {
  const std::string index_name = "idx_registry_commit_manifest_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));

  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  store_.fail_save_manifest = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnFailsWhenCommittedSnapshotLoadForRollbackFails) {
  store_.transactional = false;
  const std::string index_name = "idx_registry_commit_load_committed_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 1.0F}));

  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(vector_index_registry::stage_upsert(txn_id, index_name, 20,
                                                  {2.0F, 2.0F}));

  store_.fail_load_committed = true;
  EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  store_.fail_load_committed = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 2, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(1U, info.entry_count);
  EXPECT_EQ(1U, info.committed_entry_count);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnFailsWhenChangeLogSnapshotLoadForRollbackFails) {
  store_.transactional = false;
  const std::string index_name = "idx_registry_commit_load_changelog_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 1.0F}));

  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(vector_index_registry::stage_upsert(txn_id, index_name, 20,
                                                  {2.0F, 2.0F}));

  store_.fail_load_change_log = true;
  EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  store_.fail_load_change_log = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 2, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(1U, info.entry_count);
  EXPECT_EQ(1U, info.committed_entry_count);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnFailsWhenPreparedSnapshotLoadForRollbackFails) {
  store_.transactional = false;
  const std::string index_name = "idx_registry_commit_load_prepared_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 1.0F}));

  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(vector_index_registry::stage_upsert(txn_id, index_name, 20,
                                                  {2.0F, 2.0F}));

  store_.fail_load_prepared = true;
  EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  store_.fail_load_prepared = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 2, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(1U, info.entry_count);
  EXPECT_EQ(1U, info.committed_entry_count);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnFailsWhenManifestSnapshotLoadForRollbackFails) {
  store_.transactional = false;
  const std::string index_name = "idx_registry_commit_load_manifest_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 1.0F}));

  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(vector_index_registry::stage_upsert(txn_id, index_name, 20,
                                                  {2.0F, 2.0F}));

  store_.fail_load_manifest = true;
  EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  store_.fail_load_manifest = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 2, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(1U, info.entry_count);
  EXPECT_EQ(1U, info.committed_entry_count);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnPersistFailureRollsBackCommittedSnapshotForRestart) {
  store_.transactional = false;
  const std::string index_name = "idx_registry_commit_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 2, {2.0F, 2.0F}));

  // First manifest write fails, rollback persistence should restore old
  // committed snapshot so restart does not observe the aborted change.
  store_.fail_save_manifest_once = true;
  EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));

  vector_index_registry::reset_for_testing();

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {2.0F, 2.0F}, 5, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, ThdTxnStatementRollbackAndCommitFlow) {
  const std::string index_name = "idx_registry_stmt";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      100, 10, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      100, 10, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::rollback_stmt_for_thd_txn(100, 10));
  ASSERT_TRUE(vector_index_registry::commit_thd_txn(100));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 5, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      100, 11, index_name, 3, {3.0F, 3.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(100, 11));
  ASSERT_TRUE(vector_index_registry::commit_thd_txn(100));
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {3.0F, 3.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(3U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       ThdTxnStatementIdSwitchReplacesActiveStmtSavepoint) {
  const std::string index_name = "idx_registry_stmt_switch";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      410, 10, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      410, 11, index_name, 2, {2.0F, 2.0F}));

  EXPECT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(410, 10));
  ASSERT_TRUE(vector_index_registry::rollback_stmt_for_thd_txn(410, 11));
  ASSERT_TRUE(vector_index_registry::commit_thd_txn(410));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 2, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, ThdTxnUserSavepointRollbackFlow) {
  const std::string index_name = "idx_registry_savepoint";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      200, 1, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(200, 1));
  ASSERT_TRUE(vector_index_registry::savepoint_thd_txn(200, "sp1"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      200, 2, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(200, 2));
  ASSERT_TRUE(vector_index_registry::rollback_to_savepoint_thd_txn(200, "sp1"));
  ASSERT_TRUE(vector_index_registry::commit_thd_txn(200));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {0.0F, 0.0F}, 10, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, ThdTxnReleaseSavepointFlow) {
  const std::string index_name = "idx_registry_release_sp";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      201, 1, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(201, 1));
  ASSERT_TRUE(vector_index_registry::savepoint_thd_txn(201, "sp_release"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      201, 2, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(201, 2));

  ASSERT_TRUE(
      vector_index_registry::release_savepoint_thd_txn(201, "sp_release"));
  EXPECT_FALSE(
      vector_index_registry::rollback_to_savepoint_thd_txn(201, "sp_release"));
  EXPECT_FALSE(
      vector_index_registry::release_savepoint_thd_txn(201, "sp_release"));
  EXPECT_TRUE(
      vector_index_registry::release_savepoint_thd_txn(999, "sp_release"));

  ASSERT_TRUE(vector_index_registry::commit_thd_txn(201));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {2.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       DiscardEmptyThdTxnOnlyRemovesSavepointOnlyContexts) {
  vector_index_registry::discard_empty_thd_txn(999);

  ASSERT_TRUE(vector_index_registry::savepoint_thd_txn(202, "sp_empty"));
  vector_index_registry::discard_empty_thd_txn(202);
  EXPECT_TRUE(vector_index_registry::commit_thd_txn(202));

  const std::string index_name = "idx_registry_discard_nonempty";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      203, 1, index_name, 7, {7.0F, 7.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(203, 1));

  vector_index_registry::discard_empty_thd_txn(203);
  ASSERT_TRUE(vector_index_registry::commit_thd_txn(203));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {7.0F, 7.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitStmtForThdTxnFailsWhenStmtSavepointWasReleasedOutOfBand) {
  const std::string index_name = "idx_registry_stmt_commit_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      400, 10, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::release_savepoint_txn(1, "__stmt_10"));
  EXPECT_FALSE(vector_index_registry::commit_stmt_for_thd_txn(400, 10));

  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(400));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       RollbackStmtForThdTxnFailsWhenStmtSavepointWasReleasedOutOfBand) {
  const std::string index_name = "idx_registry_stmt_rollback_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      401, 11, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::release_savepoint_txn(1, "__stmt_11"));
  EXPECT_FALSE(vector_index_registry::rollback_stmt_for_thd_txn(401, 11));

  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(401));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {2.0F, 2.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, PrepareAndCommitPreparedXidAppliesDurableRows) {
  const std::string index_name = "idx_registry_prepared_commit";
  const uint64_t pending_before = vector_status::pending_txn_changes();
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid;
  xid.set_format_id(42);
  xid.set_gtrid_length(3);
  xid.set_bqual_length(0);
  xid.set_data("abc", 3);

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      300, 1, index_name, 7, {7.0F, 7.0F}));
  EXPECT_EQ(pending_before + 1, vector_status::pending_txn_changes());
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(300, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(300, xid));
  ASSERT_TRUE(vector_index_registry::has_prepared_xid(xid));

  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  ASSERT_TRUE(vector_index_truth_store::get()->load_prepared(&prepared_rows));
  ASSERT_EQ(1U, prepared_rows.size());
  EXPECT_EQ(7U, prepared_rows[0].doc_id);

  ASSERT_EQ(XA_OK,
            vector_index_registry::commit_prepared_xid_for_thd(300, xid));
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));
  EXPECT_EQ(pending_before, vector_status::pending_txn_changes());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {7.0F, 7.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);

  ASSERT_TRUE(vector_index_truth_store::get()->load_prepared(&prepared_rows));
  EXPECT_TRUE(prepared_rows.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, PreparedXidMatchingCoversMismatchBranches) {
  XID xid;
  xid.set_format_id(42);
  xid.set_gtrid_length(3);
  xid.set_bqual_length(1);
  xid.set_data("abcx", 4);

  vector_index_metadata_store::prepared_change_row row;
  row.format_id = 42;
  row.gtrid_length = 3;
  row.bqual_length = 1;
  row.xid_data = "abcx";
  EXPECT_TRUE(vector_index_registry::detail::xid_matches_row(xid, row));

  row.format_id = 43;
  EXPECT_FALSE(vector_index_registry::detail::xid_matches_row(xid, row));
  row.format_id = 42;

  row.gtrid_length = 2;
  EXPECT_FALSE(vector_index_registry::detail::xid_matches_row(xid, row));
  row.gtrid_length = 3;

  row.bqual_length = 0;
  EXPECT_FALSE(vector_index_registry::detail::xid_matches_row(xid, row));
  row.bqual_length = 1;

  row.xid_data = "abc";
  EXPECT_FALSE(vector_index_registry::detail::xid_matches_row(xid, row));

  row.xid_data = "abcy";
  EXPECT_FALSE(vector_index_registry::detail::xid_matches_row(xid, row));
}

TEST_F(VectorIndexRegistryTest, PrepareThdTxnWithNoPendingRowsIsNoop) {
  const std::string index_name = "idx_registry_prepare_empty";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      304, 1, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::rollback_stmt_for_thd_txn(304, 1));

  XID xid = make_test_xid(45, "prepare-empty");
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(304, xid));
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));

  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  ASSERT_TRUE(vector_index_truth_store::get()->load_prepared(&prepared_rows));
  EXPECT_TRUE(prepared_rows.empty());

  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(304));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       PrepareAndCommitPreparedXidWithoutThdAppliesDurableRows) {
  const std::string index_name = "idx_registry_prepared_commit_no_thd";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid = make_test_xid(43, "commit-no-thd");
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      301, 1, index_name, 8, {8.0F, 8.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(301, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(301, xid));
  ASSERT_TRUE(vector_index_registry::has_prepared_xid(xid));

  ASSERT_EQ(XA_OK, vector_index_registry::commit_prepared_xid(xid));
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {8.0F, 8.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(8U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, CommitPreparedXidIfLoadedAppliesDurableRows) {
  const std::string index_name = "idx_registry_prepared_commit_if_loaded";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid = make_test_xid(46, "commit-if-loaded");
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      305, 1, index_name, 9, {9.0F, 9.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(305, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(305, xid));
  ASSERT_TRUE(vector_index_registry::has_prepared_xid(xid));

  ASSERT_EQ(XA_OK, vector_index_registry::commit_prepared_xid_if_loaded(xid));
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {9.0F, 9.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(9U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, DropIndexesForTableRemovesMatchingPrefix) {
  ASSERT_TRUE(vector_index_registry::create_index("db1.t1.c1", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index("db1.t1.c2", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index("db1.t2.c1", 2, "euclidean",
                                                  "memory", "native"));

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  ASSERT_EQ(3U, index_names.size());

  ASSERT_TRUE(vector_index_registry::drop_indexes_for_table("db1", "t1"));
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  ASSERT_EQ(1U, index_names.size());
  EXPECT_EQ("db1.t2.c1", index_names[0]);

  ASSERT_TRUE(vector_index_registry::drop_index("db1.t2.c1"));
}

TEST_F(VectorIndexRegistryTest,
       DropIndexesForTableRollbackOnPersistFailureKeepsRuntimeState) {
  ASSERT_TRUE(vector_index_registry::create_index("db2.t1.c1", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index("db2.t1.c2", 2, "euclidean",
                                                  "memory", "native"));

  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::drop_indexes_for_table("db2", "t1"));
  store_.fail_save_manifest = false;

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  EXPECT_EQ(2U, index_names.size());
  EXPECT_EQ("db2.t1.c1", index_names[0]);
  EXPECT_EQ("db2.t1.c2", index_names[1]);

  ASSERT_TRUE(vector_index_registry::drop_index("db2.t1.c1"));
  ASSERT_TRUE(vector_index_registry::drop_index("db2.t1.c2"));
}

TEST_F(VectorIndexRegistryTest, DropIndexesForTableNoMatchPreservesState) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbkeep_table.t1.v", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(
      vector_index_registry::upsert("dbkeep_table.t1.v", 31, {3.0F, 1.0F}));

  ASSERT_TRUE(vector_index_registry::drop_indexes_for_table("dbabsent", "t0"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(
      vector_index_registry::get_index_info("dbkeep_table.t1.v", &info));
  EXPECT_EQ(1U, info.committed_entry_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("dbkeep_table.t1.v", {3.0F, 1.0F},
                                            1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(31U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("dbkeep_table.t1.v"));
}

TEST_F(VectorIndexRegistryTest, DropIndexesForDatabaseRemovesMatchingPrefix) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbdrop.t1.c1", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbdrop.t2.c1", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbkeep.t1.c1", 2, "euclidean", "memory", "native"));

  ASSERT_TRUE(vector_index_registry::drop_indexes_for_database("dbdrop"));

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  ASSERT_EQ(1U, index_names.size());
  EXPECT_EQ("dbkeep.t1.c1", index_names[0]);

  ASSERT_TRUE(vector_index_registry::drop_index("dbkeep.t1.c1"));
}

TEST_F(VectorIndexRegistryTest,
       DropIndexesForDatabaseRollbackOnPersistFailureKeepsRuntimeState) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbdropfail.t1.c1", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbdropfail.t2.c1", 2, "euclidean", "memory", "native"));

  store_.fail_save_metadata = true;
  EXPECT_FALSE(vector_index_registry::drop_indexes_for_database("dbdropfail"));
  store_.fail_save_metadata = false;

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  ASSERT_EQ(2U, index_names.size());
  EXPECT_EQ("dbdropfail.t1.c1", index_names[0]);
  EXPECT_EQ("dbdropfail.t2.c1", index_names[1]);

  ASSERT_TRUE(vector_index_registry::drop_index("dbdropfail.t1.c1"));
  ASSERT_TRUE(vector_index_registry::drop_index("dbdropfail.t2.c1"));
}

TEST_F(VectorIndexRegistryTest, ResetMappedIndexesForTableRecreatesEmptyIndex) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  const std::string index_name = "dbreset.t1.v";
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      index_name, 2, "cosine", "memory", "hnswlib", "dbreset", "t1", "v",
      "id"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::set_search_ef(index_name, 88));

  ASSERT_TRUE(
      vector_index_registry::reset_mapped_indexes_for_table("dbreset", "t1"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("dbreset", info.schema_name);
  EXPECT_EQ("t1", info.table_name);
  EXPECT_EQ("v", info.column_name);
  EXPECT_EQ(88U, info.search_ef);
  EXPECT_EQ(0U, info.entry_count);
  EXPECT_EQ(0U, info.committed_entry_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       ResetMappedIndexesForTableRollbackOnPersistFailureKeepsRuntimeState) {
  const std::string index_name = "dbresetfail.t1.v";
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      index_name, 2, "euclidean", "memory", "native", "dbresetfail", "t1", "v",
      "id"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 11, {1.0F, 1.0F}));

  store_.fail_save_metadata = true;
  EXPECT_TRUE(vector_index_registry::reset_mapped_indexes_for_table(
      "dbresetfail", "t1"));
  store_.fail_save_metadata = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       ResetMappedIndexesForTableIgnoresStandaloneIndexes) {
  const std::string mapped_index_name = "dbresetmix.t1.v";
  const std::string standalone_index_name = "idx_resetmix_standalone";
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      mapped_index_name, 2, "euclidean", "memory", "native", "dbresetmix", "t1",
      "v", "id"));
  ASSERT_TRUE(vector_index_registry::create_index(
      standalone_index_name, 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(
      vector_index_registry::upsert(mapped_index_name, 41, {4.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::upsert(standalone_index_name, 82, {8.0F, 2.0F}));

  ASSERT_TRUE(vector_index_registry::reset_mapped_indexes_for_table(
      "dbresetmix", "t1"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(mapped_index_name, &info));
  EXPECT_EQ(0U, info.entry_count);
  EXPECT_EQ(0U, info.committed_entry_count);
  ASSERT_TRUE(
      vector_index_registry::get_index_info(standalone_index_name, &info));
  EXPECT_EQ(1U, info.entry_count);
  EXPECT_EQ(1U, info.committed_entry_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search(mapped_index_name, {4.0F, 1.0F}, 1,
                                            &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::search(standalone_index_name, {8.0F, 2.0F},
                                            1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(82U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(mapped_index_name));
  ASSERT_TRUE(vector_index_registry::drop_index(standalone_index_name));
}

TEST_F(VectorIndexRegistryTest, ResetMappedIndexesForTableNoMatchIsNoop) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbreset_nomatch.t1.v", 2, "euclidean", "memory", "native",
      "dbreset_nomatch", "t1", "v", "id"));
  ASSERT_TRUE(
      vector_index_registry::upsert("dbreset_nomatch.t1.v", 14, {1.0F, 4.0F}));

  ASSERT_TRUE(vector_index_registry::reset_mapped_indexes_for_table(
      "dbreset_nomatch", "t9"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(
      vector_index_registry::get_index_info("dbreset_nomatch.t1.v", &info));
  EXPECT_EQ(1U, info.committed_entry_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("dbreset_nomatch.t1.v",
                                            {1.0F, 4.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(14U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("dbreset_nomatch.t1.v"));
}

TEST_F(VectorIndexRegistryTest,
       RenameIndexesForTableMovesMetadataAndCommittedEntries) {
  ASSERT_TRUE(vector_index_registry::create_index("db3.t1.v", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert("db3.t1.v", 11, {1.0F, 1.0F}));

  ASSERT_TRUE(vector_index_registry::rename_indexes_for_table(
      "db3", "t1", "db3", "t1_renamed"));

  vector_index_registry::index_info info;
  ASSERT_FALSE(vector_index_registry::get_index_info("db3.t1.v", &info));
  ASSERT_TRUE(vector_index_registry::get_index_info("db3.t1_renamed.v", &info));
  EXPECT_EQ(1U, info.committed_entry_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("db3.t1_renamed.v", {1.0F, 1.0F}, 1,
                                            &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("db3.t1_renamed.v"));
}

TEST_F(VectorIndexRegistryTest,
       RenameIndexesForTableRollbackOnPersistFailureKeepsRuntimeState) {
  ASSERT_TRUE(vector_index_registry::create_index("db4.t1.v", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert("db4.t1.v", 22, {2.0F, 2.0F}));

  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::rename_indexes_for_table(
      "db4", "t1", "db4", "t1_renamed"));
  store_.fail_save_manifest = false;

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info("db4.t1.v", &info));
  ASSERT_FALSE(
      vector_index_registry::get_index_info("db4.t1_renamed.v", &info));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search("db4.t1.v", {2.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(22U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("db4.t1.v"));
}

TEST_F(VectorIndexRegistryTest,
       RenameIndexesForTableMovesMappedBindingsAcrossDatabase) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbmove.t1.v", 2, "euclidean", "memory", "native", "dbmove", "t1", "v",
      "id"));
  ASSERT_TRUE(vector_index_registry::upsert("dbmove.t1.v", 73, {7.0F, 3.0F}));

  ASSERT_TRUE(vector_index_registry::rename_indexes_for_table("dbmove", "t1",
                                                              "dbmove2", "t2"));

  vector_index_registry::index_info info;
  ASSERT_FALSE(vector_index_registry::get_index_info("dbmove.t1.v", &info));
  ASSERT_TRUE(vector_index_registry::get_index_info("dbmove2.t2.v", &info));
  EXPECT_EQ("dbmove2", info.schema_name);
  EXPECT_EQ("t2", info.table_name);
  EXPECT_EQ("v", info.column_name);
  EXPECT_EQ(1U, info.committed_entry_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search("dbmove2.t2.v", {7.0F, 3.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(73U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("dbmove2.t2.v"));
}

TEST_F(VectorIndexRegistryTest,
       RenameIndexesForTableKeepsOtherPendingTxnChanges) {
  ASSERT_TRUE(vector_index_registry::create_index("db5.t1.v", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index("db5.other.v", 2, "euclidean",
                                                  "memory", "native"));

  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(vector_index_registry::stage_upsert(txn_id, "db5.other.v", 33,
                                                  {3.0F, 3.0F}));

  ASSERT_TRUE(vector_index_registry::rename_indexes_for_table(
      "db5", "t1", "db5", "t1_renamed"));
  ASSERT_TRUE(vector_index_registry::commit_txn(txn_id));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search("db5.other.v", {3.0F, 3.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(33U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("db5.t1_renamed.v"));
  ASSERT_TRUE(vector_index_registry::drop_index("db5.other.v"));
}

TEST_F(VectorIndexRegistryTest, RenameIndexesForTableNoMatchIsNoop) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbnomatch.t1.v", 2, "euclidean", "memory", "native", "dbnomatch", "t1",
      "v", "id"));
  ASSERT_TRUE(vector_index_registry::upsert("dbnomatch.t1.v", 6, {6.0F, 6.0F}));

  ASSERT_TRUE(vector_index_registry::rename_indexes_for_table("dbabsent", "t0",
                                                              "dbnew", "t9"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info("dbnomatch.t1.v", &info));
  EXPECT_EQ("dbnomatch", info.schema_name);
  EXPECT_EQ("t1", info.table_name);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("dbnomatch.t1.v", {6.0F, 6.0F}, 1,
                                            &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(6U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("dbnomatch.t1.v"));
}

TEST_F(VectorIndexRegistryTest, RenameIndexesForTableRejectsTargetCollision) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbcollision.t1.v", 2, "euclidean", "memory", "native", "dbcollision",
      "t1", "v", "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbcollision2.t2.v", 2, "euclidean", "memory", "native", "dbcollision2",
      "t2", "v", "id"));

  EXPECT_FALSE(vector_index_registry::rename_indexes_for_table(
      "dbcollision", "t1", "dbcollision2", "t2"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info("dbcollision.t1.v", &info));
  EXPECT_EQ("dbcollision", info.schema_name);
  EXPECT_EQ("t1", info.table_name);
  ASSERT_TRUE(
      vector_index_registry::get_index_info("dbcollision2.t2.v", &info));
  EXPECT_EQ("dbcollision2", info.schema_name);
  EXPECT_EQ("t2", info.table_name);

  ASSERT_TRUE(vector_index_registry::drop_index("dbcollision.t1.v"));
  ASSERT_TRUE(vector_index_registry::drop_index("dbcollision2.t2.v"));
}

TEST_F(VectorIndexRegistryTest,
       RenameIndexForColumnMovesMetadataAndCommittedEntries) {
  ASSERT_TRUE(vector_index_registry::create_index("db6.t1.v", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert("db6.t1.v", 44, {4.0F, 4.0F}));

  ASSERT_TRUE(vector_index_registry::rename_index_for_column("db6", "t1", "v",
                                                             "vec_col"));

  vector_index_registry::index_info info;
  ASSERT_FALSE(vector_index_registry::get_index_info("db6.t1.v", &info));
  ASSERT_TRUE(vector_index_registry::get_index_info("db6.t1.vec_col", &info));
  EXPECT_EQ(1U, info.committed_entry_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("db6.t1.vec_col", {4.0F, 4.0F}, 1,
                                            &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(44U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("db6.t1.vec_col"));
}

TEST_F(VectorIndexRegistryTest,
       RenameIndexForColumnRollbackOnPersistFailureKeepsRuntimeState) {
  ASSERT_TRUE(vector_index_registry::create_index("db7.t1.v", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert("db7.t1.v", 55, {5.0F, 5.0F}));

  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::rename_index_for_column("db7", "t1", "v",
                                                              "vec_col"));
  store_.fail_save_manifest = false;

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info("db7.t1.v", &info));
  ASSERT_FALSE(vector_index_registry::get_index_info("db7.t1.vec_col", &info));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search("db7.t1.v", {5.0F, 5.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(55U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("db7.t1.v"));
}

TEST_F(VectorIndexRegistryTest, RenameIndexForColumnSameNameIsNoop) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbsame.t1.v", 2, "euclidean", "memory", "native", "dbsame", "t1", "v",
      "id"));
  ASSERT_TRUE(vector_index_registry::upsert("dbsame.t1.v", 21, {2.0F, 1.0F}));

  ASSERT_TRUE(
      vector_index_registry::rename_index_for_column("dbsame", "t1", "v", "v"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info("dbsame.t1.v", &info));
  EXPECT_EQ("dbsame", info.schema_name);
  EXPECT_EQ("t1", info.table_name);
  EXPECT_EQ("v", info.column_name);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search("dbsame.t1.v", {2.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(21U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("dbsame.t1.v"));
}

TEST_F(VectorIndexRegistryTest, RenameIndexForColumnRejectsExistingTarget) {
  ASSERT_TRUE(vector_index_registry::create_index("db7b.t1.v", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "db7b.t1.vec_col", 2, "euclidean", "memory", "native"));

  EXPECT_FALSE(vector_index_registry::rename_index_for_column("db7b", "t1", "v",
                                                              "vec_col"));

  ASSERT_TRUE(vector_index_registry::drop_index("db7b.t1.v"));
  ASSERT_TRUE(vector_index_registry::drop_index("db7b.t1.vec_col"));
}

TEST_F(VectorIndexRegistryTest, RenameIndexForColumnMissingSourceIsNoop) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbmissing.t1.other", 2, "euclidean", "memory", "native", "dbmissing",
      "t1", "other", "id"));

  ASSERT_TRUE(vector_index_registry::rename_index_for_column("dbmissing", "t1",
                                                             "v", "vec_col"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(
      vector_index_registry::get_index_info("dbmissing.t1.other", &info));
  EXPECT_EQ("other", info.column_name);
  ASSERT_FALSE(
      vector_index_registry::get_index_info("dbmissing.t1.vec_col", &info));

  ASSERT_TRUE(vector_index_registry::drop_index("dbmissing.t1.other"));
}

TEST_F(VectorIndexRegistryTest, DropIndexForColumnRemovesMappedIndexOnly) {
  ASSERT_TRUE(vector_index_registry::create_index("db8.t1.v", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "db8.t1.other", 2, "euclidean", "memory", "native"));

  ASSERT_TRUE(vector_index_registry::drop_index_for_column("db8", "t1", "v"));

  vector_index_registry::index_info info;
  ASSERT_FALSE(vector_index_registry::get_index_info("db8.t1.v", &info));
  ASSERT_TRUE(vector_index_registry::get_index_info("db8.t1.other", &info));
  ASSERT_TRUE(vector_index_registry::drop_index_for_column("db8", "t1", "v"));

  ASSERT_TRUE(vector_index_registry::drop_index("db8.t1.other"));
}

TEST_F(VectorIndexRegistryTest,
       DropIndexForColumnRollbackOnPersistFailureKeepsRuntimeState) {
  ASSERT_TRUE(vector_index_registry::create_index("db8b.t1.v", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert("db8b.t1.v", 12, {2.0F, 2.0F}));

  store_.fail_save_metadata = true;
  EXPECT_FALSE(vector_index_registry::drop_index_for_column("db8b", "t1", "v"));
  store_.fail_save_metadata = false;

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info("db8b.t1.v", &info));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search("db8b.t1.v", {2.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(12U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("db8b.t1.v"));
}

TEST_F(VectorIndexRegistryTest, DropIndexForColumnMissingIsNoop) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbdropmissing.t1.v", 2, "euclidean", "memory", "native", "dbdropmissing",
      "t1", "v", "id"));
  ASSERT_TRUE(
      vector_index_registry::upsert("dbdropmissing.t1.v", 15, {1.0F, 5.0F}));

  ASSERT_TRUE(vector_index_registry::drop_index_for_column("dbdropmissing",
                                                           "t1", "other"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(
      vector_index_registry::get_index_info("dbdropmissing.t1.v", &info));
  EXPECT_EQ(1U, info.committed_entry_count);

  ASSERT_TRUE(vector_index_registry::drop_index("dbdropmissing.t1.v"));
}

TEST_F(VectorIndexRegistryTest, DropIndexesForDatabaseNoMatchPreservesState) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbkeep_only.t1.v", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(
      vector_index_registry::upsert("dbkeep_only.t1.v", 19, {1.0F, 9.0F}));

  ASSERT_TRUE(vector_index_registry::drop_indexes_for_database("dbabsent"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info("dbkeep_only.t1.v", &info));
  EXPECT_EQ(1U, info.committed_entry_count);
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("dbkeep_only.t1.v", {1.0F, 9.0F}, 1,
                                            &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(19U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("dbkeep_only.t1.v"));
}

TEST_F(VectorIndexRegistryTest, RebuildAndRecoverAllRoundTrip) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_a", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_b", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(
      vector_index_registry::upsert("idx_registry_a", 10, {1.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::upsert("idx_registry_b", 20, {2.0F, 2.0F}));

  size_t rebuilt_count = 0;
  ASSERT_TRUE(vector_index_registry::rebuild_all_indexes(&rebuilt_count));
  EXPECT_EQ(2U, rebuilt_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("idx_registry_a", {1.0F, 1.0F}, 1,
                                            &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  size_t recovered_count = 0;
  ASSERT_TRUE(vector_index_registry::recover_all_indexes(&recovered_count));
  EXPECT_EQ(2U, recovered_count);
  ASSERT_TRUE(vector_index_registry::search("idx_registry_b", {2.0F, 2.0F}, 1,
                                            &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(20U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("idx_registry_a"));
  ASSERT_TRUE(vector_index_registry::drop_index("idx_registry_b"));
}

TEST_F(VectorIndexRegistryTest, SearchAndDifferentIndexRebuildRunConcurrently) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_concurrent_search", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_concurrent_rebuild", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert("idx_registry_concurrent_search", 1,
                                            {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::upsert("idx_registry_concurrent_rebuild",
                                            2, {2.0F, 2.0F}));

  std::atomic<bool> failed{false};
  std::thread searcher([&failed]() {
    for (int i = 0; i < 200 && !failed.load(); ++i) {
      std::vector<vector_index::search_result> result;
      if (!vector_index_registry::search("idx_registry_concurrent_search",
                                         {1.0F, 1.0F}, 1, &result) ||
          result.size() != 1 || result[0].doc_id != 1) {
        failed.store(true);
      }
    }
  });
  std::thread rebuilder([&failed]() {
    for (int i = 0; i < 20 && !failed.load(); ++i) {
      if (!vector_index_registry::rebuild_index(
              "idx_registry_concurrent_rebuild")) {
        failed.store(true);
      }
    }
  });
  searcher.join();
  rebuilder.join();

  EXPECT_FALSE(failed.load());
  ASSERT_TRUE(vector_index_registry::drop_index("idx_registry_concurrent_search"));
  ASSERT_TRUE(vector_index_registry::drop_index("idx_registry_concurrent_rebuild"));
}

TEST_F(VectorIndexRegistryTest,
       RebuildAllRollbackOnPersistFailureKeepsServingState) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_rall_a", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_rall_b", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(
      vector_index_registry::upsert("idx_registry_rall_a", 101, {1.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::upsert("idx_registry_rall_b", 202, {2.0F, 2.0F}));

  size_t rebuilt_count = 999;
  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::rebuild_all_indexes(&rebuilt_count));
  EXPECT_EQ(0U, rebuilt_count);
  store_.fail_save_manifest = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("idx_registry_rall_a", {1.0F, 1.0F},
                                            1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(101U, result[0].doc_id);
  ASSERT_TRUE(vector_index_registry::search("idx_registry_rall_b", {2.0F, 2.0F},
                                            1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(202U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("idx_registry_rall_a"));
  ASSERT_TRUE(vector_index_registry::drop_index("idx_registry_rall_b"));
}

TEST_F(VectorIndexRegistryTest, SetSearchEfRollbackOnPersistFailure) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  const std::string index_name = "idx_registry_search_ef_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "cosine",
                                                  "memory", "hnswlib"));

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &before));

  store_.fail_save_metadata = true;
  EXPECT_FALSE(vector_index_registry::set_search_ef(index_name, 123));
  store_.fail_save_metadata = false;

  vector_index_registry::index_info after;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &after));
  EXPECT_EQ(before.search_ef, after.search_ef);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       SetSearchEfRestoresPersistedCandidateOnApplyFailure) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  const std::string index_name = "idx_registry_search_ef_apply_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "cosine",
                                                  "memory", "hnswlib"));
  ASSERT_TRUE(vector_index_registry::set_search_ef(index_name, 48));

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &before));

  store_.ResetPersistCountersForTesting();
  EXPECT_FALSE(vector_index_registry::set_search_ef(index_name, 0));

  EXPECT_EQ(2U, store_.save_metadata_calls);
  vector_index_registry::index_info after;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &after));
  EXPECT_EQ(before.search_ef, after.search_ef);
  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  ASSERT_TRUE(store_.load_metadata(&metadata_rows));
  ASSERT_EQ(1U, metadata_rows.size());
  EXPECT_EQ(before.search_ef, metadata_rows[0].search_ef);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, SetHnswBuildParamsRollbackOnPersistFailure) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  const std::string index_name = "idx_registry_hnsw_params_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "cosine",
                                                  "memory", "hnswlib"));

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &before));

  store_.fail_save_metadata = true;
  EXPECT_FALSE(
      vector_index_registry::set_hnsw_build_params(index_name, 32, 512));
  store_.fail_save_metadata = false;

  vector_index_registry::index_info after;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &after));
  EXPECT_EQ(before.hnsw_m, after.hnsw_m);
  EXPECT_EQ(before.hnsw_ef_construction, after.hnsw_ef_construction);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, SetFaissIvfParamsRollbackOnPersistFailure) {
  const std::string index_name = "idx_registry_faiss_ivf_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "cosine",
                                                  "external", "faiss"));

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &before));

  store_.fail_save_metadata = true;
  EXPECT_FALSE(vector_index_registry::set_faiss_ivf_params(index_name, 64, 8));
  store_.fail_save_metadata = false;

  vector_index_registry::index_info after;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &after));
  EXPECT_EQ(before.faiss_nlist, after.faiss_nlist);
  EXPECT_EQ(before.faiss_nprobe, after.faiss_nprobe);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, SetFaissIvfPqParamsRollbackOnPersistFailure) {
  const std::string index_name = "idx_registry_faiss_ivfpq_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "cosine",
                                                  "external", "faiss"));

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &before));

  store_.fail_save_metadata = true;
  EXPECT_FALSE(
      vector_index_registry::set_faiss_ivf_pq_params(index_name, 64, 8, 8, 8));
  store_.fail_save_metadata = false;

  vector_index_registry::index_info after;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &after));
  EXPECT_EQ(before.faiss_nlist, after.faiss_nlist);
  EXPECT_EQ(before.faiss_nprobe, after.faiss_nprobe);
  EXPECT_EQ(before.faiss_pq_m, after.faiss_pq_m);
  EXPECT_EQ(before.faiss_pq_bits, after.faiss_pq_bits);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       TuningApisRollbackWhenRuntimeRejectsPersistedCandidate) {
  const std::string hnsw_name = "idx_registry_hnsw_runtime_reject";
  ASSERT_TRUE(vector_index_registry::create_index(hnsw_name, 2, "cosine",
                                                  "memory", "hnswlib"));

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(hnsw_name, &before));
  EXPECT_FALSE(vector_index_registry::set_hnsw_build_params(hnsw_name, 0, 200));

  vector_index_registry::index_info after;
  ASSERT_TRUE(vector_index_registry::get_index_info(hnsw_name, &after));
  EXPECT_EQ(before.hnsw_m, after.hnsw_m);
  EXPECT_EQ(before.hnsw_ef_construction, after.hnsw_ef_construction);
  ASSERT_TRUE(vector_index_registry::drop_index(hnsw_name));

  const std::string faiss_name = "idx_registry_faiss_runtime_reject";
  ASSERT_TRUE(vector_index_registry::create_index(faiss_name, 2, "cosine",
                                                  "external", "faiss"));

  ASSERT_TRUE(vector_index_registry::get_index_info(faiss_name, &before));
  EXPECT_FALSE(vector_index_registry::set_faiss_ivf_params(faiss_name, 64, 0));

  ASSERT_TRUE(vector_index_registry::get_index_info(faiss_name, &after));
  EXPECT_EQ(before.faiss_nlist, after.faiss_nlist);
  EXPECT_EQ(before.faiss_nprobe, after.faiss_nprobe);
  ASSERT_TRUE(vector_index_registry::drop_index(faiss_name));

  const std::string diskann_name = "idx_registry_diskann_runtime_reject";
  ASSERT_TRUE(vector_index_registry::create_index(diskann_name, 2, "cosine",
                                                  "external", "diskann"));

  ASSERT_TRUE(vector_index_registry::get_index_info(diskann_name, &before));
  EXPECT_FALSE(
      vector_index_registry::set_diskann_build_params(diskann_name, 0, 96, 4));
  ASSERT_TRUE(vector_index_registry::get_index_info(diskann_name, &after));
  EXPECT_EQ(before.diskann_max_degree, after.diskann_max_degree);
  EXPECT_EQ(before.diskann_build_complexity, after.diskann_build_complexity);
  EXPECT_EQ(before.diskann_build_threads, after.diskann_build_threads);
  EXPECT_FALSE(
      vector_index_registry::set_diskann_build_threads(diskann_name, 65536));
  ASSERT_TRUE(vector_index_registry::get_index_info(diskann_name, &after));
  EXPECT_EQ(before.diskann_build_threads, after.diskann_build_threads);
  EXPECT_FALSE(
      vector_index_registry::set_diskann_search_complexity(diskann_name, 0));
  ASSERT_TRUE(vector_index_registry::get_index_info(diskann_name, &after));
  EXPECT_EQ(before.diskann_search_complexity,
            after.diskann_search_complexity);
  ASSERT_TRUE(vector_index_registry::drop_index(diskann_name));
}

TEST_F(VectorIndexRegistryTest,
       CleanupDroppedIndexArtifactsIgnoresInvalidDescriptor) {
  vector_index_registry::cleanup_dropped_index_artifacts(
      vector_index_registry::dropped_index_artifacts{});
}

TEST_F(VectorIndexRegistryTest, TuningApisRejectMissingIndexBeforePersistence) {
  const std::string missing = "idx_registry_missing_tuning";

  EXPECT_FALSE(vector_index_registry::set_search_ef(missing, 128));
  EXPECT_FALSE(vector_index_registry::set_hnsw_build_params(missing, 32, 400));
  EXPECT_FALSE(vector_index_registry::set_faiss_ivf_params(missing, 64, 8));
  EXPECT_FALSE(
      vector_index_registry::set_faiss_ivf_pq_params(missing, 64, 8, 16, 8));
  EXPECT_FALSE(vector_index_registry::set_diskann_build_params(missing, 48, 96,
                                                               4));
  EXPECT_FALSE(vector_index_registry::set_diskann_build_threads(missing, 4));
  EXPECT_FALSE(
      vector_index_registry::set_diskann_search_complexity(missing, 96));
}

TEST_F(VectorIndexRegistryTest, SetDiskAnnBuildParamsRollbackOnPersistFailure) {
  const std::string index_name = "idx_registry_diskann_build_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "cosine",
                                                  "external", "diskann"));

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &before));

  store_.fail_save_metadata = true;
  EXPECT_FALSE(
      vector_index_registry::set_diskann_build_params(index_name, 24, 120, 2));
  store_.fail_save_metadata = false;

  vector_index_registry::index_info after;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &after));
  EXPECT_EQ(before.diskann_max_degree, after.diskann_max_degree);
  EXPECT_EQ(before.diskann_build_complexity, after.diskann_build_complexity);
  EXPECT_EQ(before.diskann_build_threads, after.diskann_build_threads);

  store_.fail_save_metadata = true;
  EXPECT_FALSE(vector_index_registry::set_diskann_build_threads(index_name, 4));
  store_.fail_save_metadata = false;

  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &after));
  EXPECT_EQ(before.diskann_build_threads, after.diskann_build_threads);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       SetDiskAnnSearchComplexityRollbackOnPersistFailure) {
  const std::string index_name = "idx_registry_diskann_search_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "cosine",
                                                  "external", "diskann"));

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &before));

  store_.fail_save_metadata = true;
  EXPECT_FALSE(
      vector_index_registry::set_diskann_search_complexity(index_name, 180));
  store_.fail_save_metadata = false;

  vector_index_registry::index_info after;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &after));
  EXPECT_EQ(before.diskann_search_complexity, after.diskann_search_complexity);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, TuningApisPropagateMetadataLoadFailure) {
  const std::string index_name = "idx_registry_unloaded_tuning";
  store_.fail_load_metadata = true;
  store_.fail_quarantine_metadata = true;

  EXPECT_FALSE(vector_index_registry::set_search_ef(index_name, 32));
  EXPECT_FALSE(
      vector_index_registry::set_hnsw_build_params(index_name, 16, 128));
  EXPECT_FALSE(vector_index_registry::set_faiss_ivf_params(index_name, 32, 4));
  EXPECT_FALSE(
      vector_index_registry::set_faiss_ivf_pq_params(index_name, 32, 4, 8, 8));
  EXPECT_FALSE(
      vector_index_registry::set_diskann_build_params(index_name, 32, 100, 2));
  EXPECT_FALSE(vector_index_registry::set_diskann_build_threads(index_name, 2));
  EXPECT_FALSE(
      vector_index_registry::set_diskann_search_complexity(index_name, 100));
}

TEST_F(VectorIndexRegistryTest,
       RebuildIndexRollbackOnPersistFailureKeepsServingState) {
  const std::string index_name = "idx_registry_rebuild_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 1.0F}));

  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::rebuild_index(index_name));
  store_.fail_save_manifest = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, RebuildIndexRoundTripPreservesEntries) {
  const std::string index_name = "idx_registry_rebuild_success";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 31, {3.0F, 1.0F}));

  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {3.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(31U, result[0].doc_id);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(1U, info.entry_count);
  EXPECT_EQ(1U, info.committed_entry_count);
  EXPECT_EQ("ready", info.lifecycle_state);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       LifecycleRebuildAndReplaceRespectCommittedCacheBudget) {
  UlonglongGuard guard(&opt_vector_entry_cache_size, 0);
  const std::string rebuild_index_name = "idx_registry_rebuild_budget";
  ASSERT_TRUE(vector_index_registry::create_index(
      rebuild_index_name, 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(
      vector_index_registry::upsert(rebuild_index_name, 10, {1.0F, 1.0F}));
  EXPECT_EQ(0U, vector_index_registry::committed_vector_memory_bytes());

  ASSERT_TRUE(vector_index_registry::rebuild_index(rebuild_index_name));
  EXPECT_EQ(0U, vector_index_registry::committed_vector_memory_bytes());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search(rebuild_index_name, {1.0F, 1.0F},
                                            1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  const std::string replace_index_name = "idx_registry_replace_budget";
  ASSERT_TRUE(vector_index_registry::create_index(
      replace_index_name, 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::replace_committed_entries(
      replace_index_name, {{20, {2.0F, 2.0F}}}));
  EXPECT_EQ(0U, vector_index_registry::committed_vector_memory_bytes());

  result.clear();
  ASSERT_TRUE(vector_index_registry::search(replace_index_name, {2.0F, 2.0F},
                                            1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(20U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(rebuild_index_name));
  ASSERT_TRUE(vector_index_registry::drop_index(replace_index_name));
}

TEST_F(VectorIndexRegistryTest, RebuildIndexRejectsMissingIndex) {
  EXPECT_FALSE(vector_index_registry::rebuild_index("idx_registry_missing"));
}

TEST_F(VectorIndexRegistryTest,
       RebuildMissingIndexRollbackOnPersistFailureKeepsServingState) {
  const std::string index_name = "idx_registry_rebuild_missing_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 61, {6.0F, 1.0F}));

  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::rebuild_index("idx_registry_missing"));
  store_.fail_save_manifest = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {6.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(61U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, RecoverIndexRoundTripPreservesEntries) {
  const std::string index_name = "idx_registry_recover_success";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 41, {4.0F, 1.0F}));

  ASSERT_TRUE(vector_index_registry::recover_index(index_name));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {4.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(41U, result[0].doc_id);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(1U, info.entry_count);
  EXPECT_EQ(1U, info.committed_entry_count);
  EXPECT_EQ("ready", info.lifecycle_state);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       RecoverIndexRollbackOnPersistFailureKeepsServingState) {
  const std::string index_name = "idx_registry_recover_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 51, {5.0F, 1.0F}));

  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::recover_index(index_name));
  store_.fail_save_manifest = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {5.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(51U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, RecoverIndexRejectsMissingIndex) {
  EXPECT_FALSE(vector_index_registry::recover_index("idx_registry_missing"));
}

TEST_F(VectorIndexRegistryTest,
       RecoverMissingIndexRollbackOnPersistFailureKeepsServingState) {
  const std::string index_name = "idx_registry_recover_missing_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 71, {7.0F, 1.0F}));

  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::recover_index("idx_registry_missing"));
  store_.fail_save_manifest = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {7.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(71U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       ReplaceCommittedEntriesSwapsServingStateAndHandlesMissingIndex) {
  const std::string index_name = "idx_registry_replace_committed";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 1.0F}));

  vector_index::index_service::committed_entries replacement{
      {20, {2.0F, 2.0F}},
      {30, {3.0F, 3.0F}},
  };
  ASSERT_TRUE(vector_index_registry::replace_committed_entries(index_name,
                                                               replacement));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {2.0F, 2.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(20U, result[0].doc_id);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(2U, info.entry_count);
  EXPECT_EQ(2U, info.committed_entry_count);

  EXPECT_FALSE(vector_index_registry::replace_committed_entries(
      "idx_registry_replace_missing", replacement));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       ReplaceCommittedEntriesPreservesLifecycleAndRejectsBadRows) {
  const std::string index_name = "idx_registry_replace_preserve_lifecycle";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::begin_bulk_load(index_name));

  vector_index::index_service::committed_entries replacement{
      {20, {2.0F, 2.0F}},
      {30, {3.0F, 3.0F}},
  };
  ASSERT_TRUE(vector_index_registry::replace_committed_entries_preserve_lifecycle(
      index_name, replacement));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("bulk_loading", info.lifecycle_state);
  EXPECT_EQ(2U, info.entry_count);
  EXPECT_EQ(2U, info.committed_entry_count);

  EXPECT_FALSE(vector_index_registry::replace_committed_entries_preserve_lifecycle(
      index_name, {{40, {4.0F, 4.0F, 4.0F}}}));
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("bulk_loading", info.lifecycle_state);
  EXPECT_EQ(2U, info.committed_entry_count);

  ASSERT_TRUE(vector_index_registry::bulk_build_index(index_name));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {2.0F, 2.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(20U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       ReplaceCommittedEntriesRollbackRestoresPreviousSnapshot) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  const std::string index_name = "idx_registry_replace_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "cosine",
                                                  "memory", "hnswlib"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 0.0F}));
  ASSERT_TRUE(vector_index_registry::set_search_ef(index_name, 48));
  ASSERT_TRUE(vector_index_registry::set_hnsw_build_params(index_name, 12, 96));

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &before));

  store_.fail_save_committed = true;
  EXPECT_FALSE(vector_index_registry::replace_committed_entries(
      index_name, {{20, {0.0F, 1.0F}}}));
  store_.fail_save_committed = false;

  vector_index_registry::index_info after;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &after));
  EXPECT_EQ(before.entry_count, after.entry_count);
  EXPECT_EQ(before.committed_entry_count, after.committed_entry_count);
  EXPECT_EQ(before.search_ef, after.search_ef);
  EXPECT_EQ(before.hnsw_m, after.hnsw_m);
  EXPECT_EQ(before.hnsw_ef_construction, after.hnsw_ef_construction);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, BeginBulkLoadRejectsPendingChanges) {
  const std::string index_name = "idx_registry_bulk_pending";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));

  EXPECT_FALSE(vector_index_registry::begin_bulk_load(index_name));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("ready", info.lifecycle_state);
  EXPECT_EQ(0U, info.last_error_code);

  ASSERT_TRUE(vector_index_registry::rollback_txn(txn_id));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, BeginBulkLoadThenBulkBuildRestoresServingState) {
  const std::string index_name = "idx_registry_bulk_roundtrip";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 1.0F}));

  ASSERT_TRUE(vector_index_registry::begin_bulk_load(index_name));
  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("bulk_loading", info.lifecycle_state);

  ASSERT_TRUE(vector_index_registry::bulk_build_index(index_name));
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("ready", info.lifecycle_state);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       BuildBackendFromConfigTestingWrapperCoversTunings) {
  using vector_index::backend_mode;
  using vector_index::backend_provider;

  vector_index::index_service::index_config config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  EXPECT_TRUE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kDiskAnn);
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_unsupported_provider_mode", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.hnsw_m = 16;
  config.hnsw_ef_construction = 200;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_hnsw_params", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.hnsw_m = 16;
  EXPECT_TRUE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_partial_hnsw_params", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.hnsw_build_threads = 1;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_hnsw_threads", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kHnswlib);
  config.hnsw_m = 12;
  config.hnsw_ef_construction = 96;
  config.hnsw_build_threads = 2;
  config.search_ef = 48;
  EXPECT_EQ(hnswlib_tuning_supported(),
            vector_index_registry::build_backend_from_config_for_testing(
                "idx_config_hnsw", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.faiss_nlist = 2;
  config.faiss_nprobe = 1;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_faiss_ivf", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.faiss_nprobe = 1;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_faiss_nprobe", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.faiss_pq_m = 1;
  config.faiss_pq_bits = 8;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_faiss_pq", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.faiss_pq_bits = 8;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_faiss_pq_bits", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.faiss_build_threads = 1;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_faiss_threads", config));

  config =
      make_backend_config(backend_mode::kExternal, backend_provider::kFaiss);
  config.faiss_nlist = 2;
  config.faiss_nprobe = 1;
  config.faiss_pq_m = 1;
  config.faiss_pq_bits = 8;
  config.faiss_build_threads = 2;
  EXPECT_TRUE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_faiss_pq", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.diskann_max_degree = 32;
  config.diskann_build_complexity = 64;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_diskann_build", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.diskann_max_degree = 32;
  EXPECT_TRUE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_partial_diskann_build", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.diskann_search_complexity = 32;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_diskann_search", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.diskann_build_threads = 1;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_diskann_threads", config));

  config =
      make_backend_config(backend_mode::kExternal, backend_provider::kDiskAnn);
  config.diskann_max_degree = 32;
  config.diskann_build_complexity = 64;
  config.diskann_build_threads = 2;
  config.diskann_search_complexity = 48;
  EXPECT_TRUE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_diskann", config));

  config =
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.search_ef = 64;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_search_ef", config));
}

TEST_F(VectorIndexRegistryTest,
       RecoverAllRollbackOnPersistFailureKeepsRebuiltServingState) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_ra", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_rb", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(
      vector_index_registry::upsert("idx_registry_ra", 10, {1.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::upsert("idx_registry_rb", 20, {2.0F, 2.0F}));

  size_t rebuilt_count = 0;
  ASSERT_TRUE(vector_index_registry::rebuild_all_indexes(&rebuilt_count));
  ASSERT_EQ(2U, rebuilt_count);

  store_.fail_save_manifest = true;
  size_t recovered_count = 0;
  EXPECT_FALSE(vector_index_registry::recover_all_indexes(&recovered_count));
  EXPECT_EQ(0U, recovered_count);
  store_.fail_save_manifest = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("idx_registry_ra", {1.0F, 1.0F}, 1,
                                            &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);
  ASSERT_TRUE(vector_index_registry::search("idx_registry_rb", {2.0F, 2.0F}, 1,
                                            &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(20U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("idx_registry_ra"));
  ASSERT_TRUE(vector_index_registry::drop_index("idx_registry_rb"));
}

TEST_F(VectorIndexRegistryTest, RebuildAllWithNoIndexesReportsZero) {
  size_t rebuilt_count = 999;
  ASSERT_TRUE(vector_index_registry::rebuild_all_indexes(&rebuilt_count));
  EXPECT_EQ(0U, rebuilt_count);
}

TEST_F(VectorIndexRegistryTest, RebuildAllRejectsPendingTxnChanges) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_pending", 2, "euclidean", "memory", "native"));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(vector_index_registry::stage_upsert(
      txn_id, "idx_registry_pending", 1, {1.0F, 1.0F}));
  size_t rebuilt_count = 0;
  EXPECT_FALSE(vector_index_registry::rebuild_all_indexes(&rebuilt_count));
  EXPECT_EQ(0U, rebuilt_count);
  ASSERT_TRUE(vector_index_registry::rollback_txn(txn_id));
  ASSERT_TRUE(vector_index_registry::drop_index("idx_registry_pending"));
}

TEST_F(VectorIndexRegistryTest, RebuildAllRejectsNullCount) {
  EXPECT_FALSE(vector_index_registry::rebuild_all_indexes(nullptr));
}

TEST_F(VectorIndexRegistryTest, RecoverAllWithNoIndexesReportsZero) {
  size_t recovered_count = 999;
  ASSERT_TRUE(vector_index_registry::recover_all_indexes(&recovered_count));
  EXPECT_EQ(0U, recovered_count);
}

TEST_F(VectorIndexRegistryTest, RecoverAllRejectsNullCount) {
  EXPECT_FALSE(vector_index_registry::recover_all_indexes(nullptr));
}

TEST_F(VectorIndexRegistryTest, RecoverAllRejectsPendingTxnChanges) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_pending_recover", 2, "euclidean", "memory", "native"));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(vector_index_registry::stage_upsert(
      txn_id, "idx_registry_pending_recover", 1, {1.0F, 1.0F}));

  size_t recovered_count = 999;
  EXPECT_FALSE(vector_index_registry::recover_all_indexes(&recovered_count));
  EXPECT_EQ(0U, recovered_count);

  ASSERT_TRUE(vector_index_registry::rollback_txn(txn_id));
  ASSERT_TRUE(
      vector_index_registry::drop_index("idx_registry_pending_recover"));
}

TEST_F(VectorIndexRegistryTest, GetIndexInfoRejectsNullOutput) {
  EXPECT_FALSE(
      vector_index_registry::get_index_info("idx_registry_missing", nullptr));
}

TEST_F(VectorIndexRegistryTest, ListIndexesRejectsNullOutput) {
  EXPECT_FALSE(vector_index_registry::list_indexes(nullptr));
}

TEST_F(VectorIndexRegistryTest,
       RecoverPreparedApisAndRollbackPreparedXidRoundTrip) {
  const std::string index_name = "idx_registry_prepared_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      300, 10, index_name, 5, {5.0F, 5.0F}));

  XID xid = make_test_xid(42, "rollback");
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(300, xid));
  ASSERT_TRUE(vector_index_registry::has_prepared_xid(xid));

  XA_recover_txn recovered[2]{};
  ASSERT_EQ(
      1, vector_index_registry::recover_prepared_xids(recovered, 2, nullptr));
  EXPECT_TRUE(recovered[0].id.eq(&xid));

  MEM_ROOT mem_root{PSI_NOT_INSTRUMENTED, 1024};
  XA_recover_txn recovered_with_mod_tables[2]{};
  ASSERT_EQ(1, vector_index_registry::recover_prepared_xids(
                   recovered_with_mod_tables, 2, &mem_root));
  EXPECT_TRUE(recovered_with_mod_tables[0].id.eq(&xid));
  EXPECT_NE(nullptr, recovered_with_mod_tables[0].mod_tables);
  mem_root.Clear();

  ASSERT_TRUE(vector_index_registry::set_prepared_in_tc(xid));
  auto xa_state_tuple = vector_gunit::make_xa_state_list_for_testing();
  auto *xa_state_list = xa_state_tuple.get();
  ASSERT_NE(nullptr, xa_state_list);
  ASSERT_EQ(0, vector_index_registry::recover_prepared_in_tc(*xa_state_list));
  EXPECT_EQ(enum_ha_recover_xa_state::PREPARED_IN_TC, xa_state_list->find(xid));

  ASSERT_EQ(XA_OK,
            vector_index_registry::rollback_prepared_xid_for_thd(300, xid));
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {5.0F, 5.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       PrepareAndRollbackPreparedXidWithoutThdDiscardsRows) {
  const std::string index_name = "idx_registry_prepared_rollback_no_thd";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid = make_test_xid(44, "rollback-no-thd");
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      302, 1, index_name, 6, {6.0F, 6.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(302, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(302, xid));
  ASSERT_TRUE(vector_index_registry::has_prepared_xid(xid));

  ASSERT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid_if_loaded(xid));
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {6.0F, 6.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       PrepareAndCommitPreparedEraseWithoutThdRemovesRow) {
  const std::string index_name = "idx_registry_prepared_erase_commit";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 6, {6.0F, 6.0F}));

  XID xid = make_test_xid(151, "erase-commit-no-thd");
  ASSERT_TRUE(
      vector_index_registry::stage_erase_for_thd_txn(317, 1, index_name, 6));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(317, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(317, xid));
  ASSERT_TRUE(vector_index_registry::has_prepared_xid(xid));

  ASSERT_EQ(XA_OK, vector_index_registry::commit_prepared_xid(xid));
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {6.0F, 6.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidReturnsRmErrWhenBeginPersistFails) {
  const std::string index_name = "idx_registry_prepared_begin_persist_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid = make_test_xid(144, "commit-begin-fail");
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      312, 1, index_name, 14, {14.0F, 14.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(312, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(312, xid));
  ASSERT_TRUE(vector_index_registry::has_prepared_xid(xid));

  const uint64_t begin_before = store_.begin_persist_calls;
  const uint64_t commit_before = store_.commit_persist_calls;
  const uint64_t rollback_before = store_.rollback_persist_calls;
  store_.fail_begin_persist = true;
  EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  store_.fail_begin_persist = false;
  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));
  EXPECT_EQ(begin_before + 1, store_.begin_persist_calls);
  EXPECT_EQ(commit_before, store_.commit_persist_calls);
  EXPECT_EQ(rollback_before, store_.rollback_persist_calls);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {14.0F, 14.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidReturnsRmErrWhenCommitPersistFails) {
  const std::string index_name = "idx_registry_prepared_commit_persist_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid = make_test_xid(145, "commit-commit-fail");
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      313, 1, index_name, 15, {15.0F, 15.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(313, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(313, xid));
  ASSERT_TRUE(vector_index_registry::has_prepared_xid(xid));

  const uint64_t begin_before = store_.begin_persist_calls;
  const uint64_t commit_before = store_.commit_persist_calls;
  const uint64_t rollback_before = store_.rollback_persist_calls;
  store_.fail_commit_persist = true;
  EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  store_.fail_commit_persist = false;
  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));
  EXPECT_GE(store_.begin_persist_calls, begin_before + 1);
  EXPECT_GE(store_.commit_persist_calls, commit_before + 1);
  EXPECT_GE(store_.rollback_persist_calls, rollback_before + 1);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {15.0F, 15.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       RollbackPreparedXidReturnsRmErrWhenPreparedPersistFails) {
  const std::string index_name = "idx_registry_prepared_rollback_persist_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid = make_test_xid(146, "rollback-persist-fail");
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      314, 1, index_name, 16, {16.0F, 16.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(314, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(314, xid));
  ASSERT_TRUE(vector_index_registry::has_prepared_xid(xid));

  store_.fail_save_prepared = true;
  EXPECT_EQ(XAER_RMERR, vector_index_registry::rollback_prepared_xid(xid));
  store_.fail_save_prepared = false;
  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {16.0F, 16.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidReturnsRmErrWhenCommittedPersistFails) {
  const std::string index_name = "idx_registry_prepared_committed_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid = make_test_xid(149, "commit-committed-fail");
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      315, 1, index_name, 17, {17.0F, 17.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(315, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(315, xid));
  ASSERT_TRUE(vector_index_registry::has_prepared_xid(xid));

  store_.fail_apply_committed_delta = true;
  EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  store_.fail_apply_committed_delta = false;
  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {17.0F, 17.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidReturnsRmErrWhenManifestPersistFails) {
  const std::string index_name = "idx_registry_prepared_manifest_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid = make_test_xid(150, "commit-manifest-fail");
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      316, 1, index_name, 18, {18.0F, 18.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(316, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(316, xid));
  ASSERT_TRUE(vector_index_registry::has_prepared_xid(xid));

  store_.fail_save_manifest = true;
  EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  store_.fail_save_manifest = false;
  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {18.0F, 18.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidReturnsRmErrForMissingPreparedIndex) {
  XID xid = make_test_xid(147, "prepared-missing-index");
  store_.SetPreparedRowsForTesting(
      {{xid.get_format_id(),
        xid.get_gtrid_length(),
        xid.get_bqual_length(),
        std::string(xid.get_data(),
                    static_cast<size_t>(xid.get_gtrid_length() +
                                        xid.get_bqual_length())),
        false,
        901,
        vector_index_metadata_store::change_op::kUpsert,
        "idx_missing_prepared",
        1,
        {1.0F, 1.0F}}});

  EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));
  EXPECT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid));
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));
}

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidReturnsRmErrForPreparedDimensionMismatch) {
  const std::string index_name = "idx_registry_prepared_dim_mismatch";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid = make_test_xid(148, "prepared-dim-mismatch");
  store_.SetPreparedRowsForTesting(
      {{xid.get_format_id(),
        xid.get_gtrid_length(),
        xid.get_bqual_length(),
        std::string(xid.get_data(),
                    static_cast<size_t>(xid.get_gtrid_length() +
                                        xid.get_bqual_length())),
        false,
        902,
        vector_index_metadata_store::change_op::kUpsert,
        index_name,
        1,
        {1.0F}}});
  vector_index_registry::reset_for_testing();

  EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));
  EXPECT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid));
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       PreparedRecoveryHelpersHandleEmptyInputsAndMissingXids) {
  XA_recover_txn recovered[1]{};
  EXPECT_EQ(0,
            vector_index_registry::recover_prepared_xids(nullptr, 1, nullptr));
  EXPECT_EQ(
      0, vector_index_registry::recover_prepared_xids(recovered, 0, nullptr));

  XID xid = make_test_xid(99, "missing-prepared");
  EXPECT_EQ(XAER_NOTA, vector_index_registry::commit_prepared_xid(xid));
  EXPECT_EQ(XAER_NOTA, vector_index_registry::rollback_prepared_xid(xid));
  EXPECT_TRUE(vector_index_registry::set_prepared_in_tc(xid));
}

TEST_F(VectorIndexRegistryTest, SetPreparedInTcRollbackOnPersistFailure) {
  const std::string index_name = "idx_registry_prepared_set_tc_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  XID xid = make_test_xid(46, "prepared-tc-fail");

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      305, 1, index_name, 3, {3.0F, 3.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(305, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(305, xid));
  store_.fail_save_prepared = true;
  EXPECT_FALSE(vector_index_registry::set_prepared_in_tc(xid));
  store_.fail_save_prepared = false;

  auto xa_state_tuple = vector_gunit::make_xa_state_list_for_testing();
  auto *xa_state_list = xa_state_tuple.get();
  ASSERT_NE(nullptr, xa_state_list);
  ASSERT_EQ(0, vector_index_registry::recover_prepared_in_tc(*xa_state_list));
  EXPECT_EQ(enum_ha_recover_xa_state::NOT_FOUND, xa_state_list->find(xid));

  ASSERT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, RecoverPreparedXidsDeduplicatesAndHonorsLimit) {
  const std::string index_name = "idx_registry_prepared_recover_limit";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid_a = make_test_xid(47, "recover-a");
  XID xid_b = make_test_xid(48, "recover-b");

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      306, 1, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(306, 1));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      306, 2, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(306, 2));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(306, xid_a));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      307, 1, index_name, 3, {3.0F, 3.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(307, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(307, xid_b));

  XA_recover_txn recovered_one[1]{};
  ASSERT_EQ(1, vector_index_registry::recover_prepared_xids(recovered_one, 1,
                                                            nullptr));

  XA_recover_txn recovered_all[4]{};
  ASSERT_EQ(2, vector_index_registry::recover_prepared_xids(recovered_all, 4,
                                                            nullptr));
  EXPECT_TRUE(recovered_all[0].id.eq(&xid_a) || recovered_all[0].id.eq(&xid_b));
  EXPECT_TRUE(recovered_all[1].id.eq(&xid_a) || recovered_all[1].id.eq(&xid_b));
  EXPECT_FALSE(recovered_all[0].id.eq(&recovered_all[1].id));

  ASSERT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid_a));
  ASSERT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid_b));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, QueueRecoveryCommitAppliesPreparedRowsOnReset) {
  const std::string index_name = "idx_registry_recovery_commit";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      301, 10, index_name, 7, {7.0F, 7.0F}));

  XID xid = make_test_xid(77, "recover-commit");
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(301, xid));
  vector_index_registry::queue_recovery_commit_xid(xid);

  std::vector<vector_index::search_result> result;
  {
    ServerStartedGuard server_started(true);
    vector_index_registry::reset_for_testing();
    ASSERT_TRUE(
        vector_index_registry::search(index_name, {7.0F, 7.0F}, 1, &result));
    EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));
  }
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       QueueRecoveryRollbackOverridesQueuedCommitOnReset) {
  const std::string index_name = "idx_registry_recovery_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      302, 10, index_name, 8, {8.0F, 8.0F}));

  XID xid = make_test_xid(88, "recover-rollback");
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(302, xid));
  vector_index_registry::queue_recovery_commit_xid(xid);
  vector_index_registry::queue_recovery_rollback_xid(xid);

  std::vector<vector_index::search_result> result;
  {
    ServerStartedGuard server_started(true);
    vector_index_registry::reset_for_testing();
    ASSERT_TRUE(
        vector_index_registry::search(index_name, {8.0F, 8.0F}, 1, &result));
    EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));
  }
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       QueueRecoveryCommitIgnoresConditionalNonVectorXidOnReset) {
  const std::string index_name = "idx_registry_recovery_non_vector";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  XID xid = make_test_xid(89, "recover-non-vector");
  vector_index_registry::queue_recovery_commit_xid(xid);

  std::vector<vector_index::search_result> result;
  {
    ServerStartedGuard server_started(true);
    vector_index_registry::reset_for_testing();
    ASSERT_TRUE(
        vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  }
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       QueueRecoveryPreparedInTcMarksRecoveredPreparedState) {
  const std::string index_name = "idx_registry_recovery_prepared_in_tc";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      303, 10, index_name, 9, {9.0F, 9.0F}));

  XID xid = make_test_xid(99, "recover-prepared-in-tc");
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(303, xid));
  vector_index_registry::queue_recovery_set_prepared_in_tc(xid);

  auto xa_state_tuple = vector_gunit::make_xa_state_list_for_testing();
  auto *xa_state_list = xa_state_tuple.get();
  ASSERT_NE(nullptr, xa_state_list);
  {
    ServerStartedGuard server_started(true);
    vector_index_registry::reset_for_testing();
    ASSERT_EQ(0, vector_index_registry::recover_prepared_in_tc(*xa_state_list));
  }
  EXPECT_EQ(enum_ha_recover_xa_state::PREPARED_IN_TC, xa_state_list->find(xid));

  ASSERT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       ChangeLogReplayRecoversCorruptCommittedSnapshot) {
  const std::string index_name = "idx_registry_changelog_replay";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 7, {7.0F, 7.0F}));
  store_.ClearCommittedRowsForTesting();

  vector_index_registry::reset_for_testing();

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {7.0F, 7.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(1U, info.entry_count);
  EXPECT_EQ(1U, info.committed_entry_count);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       InvalidChangeLogReplayFallsBackToCommittedSnapshot) {
  const std::string index_name = "idx_registry_changelog_invalid";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  vector_index_metadata_store::change_log_row invalid_row;
  invalid_row.sequence = 42;
  invalid_row.txn_id = 7;
  invalid_row.op = vector_index_metadata_store::change_op::kUpsert;
  invalid_row.index_name = "missing.index";
  invalid_row.doc_id = 99;
  invalid_row.vector = {9.0F, 9.0F};
  store_.SetChangeLogRowsForTesting({invalid_row});

  const uint64_t failures_before = vector_status::change_log_load_failures();
  const uint64_t replay_failures_before =
      vector_status::change_log_replay_failures();
  vector_index_registry::reset_for_testing();

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_EQ(failures_before + 1, vector_status::change_log_load_failures());
  EXPECT_EQ(replay_failures_before + 1,
            vector_status::change_log_replay_failures());

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       ChangeLogReplayOrdersRowsBySequenceIndexDocAndOp) {
  const std::string index_a = "idx_registry_changelog_order_a";
  const std::string index_b = "idx_registry_changelog_order_b";
  ASSERT_TRUE(vector_index_registry::create_index(index_a, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(index_b, 2, "euclidean",
                                                  "memory", "native"));

  store_.ClearCommittedRowsForTesting();

  std::vector<vector_index_metadata_store::change_log_row> rows;
  rows.push_back({2,
                  11,
                  vector_index_metadata_store::change_op::kUpsert,
                  index_b,
                  2,
                  {2.0F, 2.0F}});
  rows.push_back({1,
                  11,
                  vector_index_metadata_store::change_op::kUpsert,
                  index_b,
                  1,
                  {1.0F, 1.0F}});
  rows.push_back({1,
                  11,
                  vector_index_metadata_store::change_op::kUpsert,
                  index_a,
                  2,
                  {2.0F, 0.0F}});
  rows.push_back(
      {1, 11, vector_index_metadata_store::change_op::kErase, index_a, 1, {}});
  rows.push_back({1,
                  11,
                  vector_index_metadata_store::change_op::kUpsert,
                  index_a,
                  1,
                  {1.0F, 0.0F}});
  store_.SetChangeLogRowsForTesting(rows);

  vector_index_registry::reset_for_testing();

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search(index_a, {1.0F, 0.0F}, 2, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::search(index_b, {1.0F, 1.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_EQ(2U, result[1].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_a));
  ASSERT_TRUE(vector_index_registry::drop_index(index_b));
}

TEST_F(VectorIndexRegistryTest,
       InvalidChangeLogOpFallsBackToCommittedSnapshot) {
  const std::string index_name = "idx_registry_changelog_invalid_op";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  vector_index_metadata_store::change_log_row invalid_row;
  invalid_row.sequence = 42;
  invalid_row.txn_id = 7;
  invalid_row.op = static_cast<vector_index_metadata_store::change_op>(99);
  invalid_row.index_name = index_name;
  invalid_row.doc_id = 2;
  invalid_row.vector = {2.0F, 2.0F};
  store_.SetChangeLogRowsForTesting({invalid_row});

  const uint64_t failures_before = vector_status::change_log_load_failures();
  const uint64_t replay_failures_before =
      vector_status::change_log_replay_failures();
  vector_index_registry::reset_for_testing();

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 2, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_EQ(failures_before + 1, vector_status::change_log_load_failures());
  EXPECT_EQ(replay_failures_before + 1,
            vector_status::change_log_replay_failures());

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       MetadataLoadFailureFallsBackWhenQuarantineSucceeds) {
  store_.fail_load_metadata = true;
  const uint64_t metadata_load_failures_before =
      vector_status::metadata_load_failures();

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  EXPECT_TRUE(index_names.empty());
  EXPECT_EQ(metadata_load_failures_before + 1,
            vector_status::metadata_load_failures());
}

TEST_F(VectorIndexRegistryTest,
       ManifestCheckpointMismatchIsRecordedAndLoadStillSucceeds) {
  const std::string index_name = "idx_registry_manifest_ckpt";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  const uint64_t metadata_load_failures_before =
      vector_status::metadata_load_failures();
  const uint64_t committed_load_failures_before =
      vector_status::committed_load_failures();
  const uint64_t manifest_load_failures_before =
      vector_status::manifest_load_failures();
  const uint64_t change_log_load_failures_before =
      vector_status::change_log_load_failures();

  store_.SetManifestCheckpointsForTesting(/*metadata_checkpoint=*/9999,
                                          /*committed_checkpoint=*/9999,
                                          /*change_log_checkpoint=*/9999);
  vector_index_registry::reset_for_testing();

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_EQ(metadata_load_failures_before + 1,
            vector_status::metadata_load_failures());
  EXPECT_EQ(committed_load_failures_before + 1,
            vector_status::committed_load_failures());
  EXPECT_EQ(manifest_load_failures_before + 1,
            vector_status::manifest_load_failures());
  EXPECT_EQ(change_log_load_failures_before + 1,
            vector_status::change_log_load_failures());

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       MetadataLoadFailureReturnsFalseWhenQuarantineFails) {
  store_.fail_load_metadata = true;
  store_.fail_quarantine_metadata = true;

  std::vector<std::string> index_names;
  EXPECT_FALSE(vector_index_registry::list_indexes(&index_names));
}

TEST_F(VectorIndexRegistryTest,
       MetadataLoadFailurePropagatesAcrossRegistryWrappers) {
  store_.fail_load_metadata = true;
  store_.fail_quarantine_metadata = true;

  std::vector<std::string> index_names;
  vector_index_registry::index_info info;
  std::vector<vector_index::search_result> result;
  size_t rebuilt_count = 99;
  size_t recovered_count = 99;
  const uint64_t txn_id = vector_index_registry::begin_txn();

  EXPECT_FALSE(vector_index_registry::create_index(
      "idx_load_fail", 2, "euclidean", "memory", "native"));
  EXPECT_FALSE(vector_index_registry::create_mapped_index(
      "db_load_fail.t_load_fail.v", 2, "euclidean", "memory", "native",
      "db_load_fail", "t_load_fail", "v", "id"));
  EXPECT_FALSE(vector_index_registry::drop_index("missing"));
  EXPECT_FALSE(vector_index_registry::begin_bulk_load("missing"));
  EXPECT_FALSE(vector_index_registry::bulk_build_index("missing"));
  EXPECT_FALSE(vector_index_registry::rebuild_index("missing"));
  EXPECT_FALSE(vector_index_registry::recover_index("missing"));
  EXPECT_FALSE(vector_index_registry::list_indexes(&index_names));
  EXPECT_FALSE(vector_index_registry::get_index_info("missing", &info));
  EXPECT_FALSE(
      vector_index_registry::search("missing", {1.0F, 1.0F}, 1, &result));
  EXPECT_FALSE(vector_index_registry::rebuild_all_indexes(&rebuilt_count));
  EXPECT_EQ(99U, rebuilt_count);
  EXPECT_FALSE(vector_index_registry::recover_all_indexes(&recovered_count));
  EXPECT_EQ(99U, recovered_count);
  EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  EXPECT_FALSE(vector_index_registry::rollback_txn(txn_id));
}

TEST_F(VectorIndexRegistryTest,
       CommittedLoadFailurePropagatesAcrossRegistryWrappers) {
  store_.fail_load_committed = true;
  store_.fail_quarantine_committed = true;

  std::vector<std::string> index_names;
  vector_index_registry::index_info info;
  std::vector<vector_index::search_result> result;
  size_t rebuilt_count = 99;
  size_t recovered_count = 99;
  const uint64_t txn_id = vector_index_registry::begin_txn();

  EXPECT_FALSE(vector_index_registry::list_indexes(&index_names));
  EXPECT_FALSE(vector_index_registry::get_index_info("missing", &info));
  EXPECT_FALSE(
      vector_index_registry::search("missing", {1.0F, 1.0F}, 1, &result));
  EXPECT_FALSE(vector_index_registry::rebuild_all_indexes(&rebuilt_count));
  EXPECT_EQ(99U, rebuilt_count);
  EXPECT_FALSE(vector_index_registry::recover_all_indexes(&recovered_count));
  EXPECT_EQ(99U, recovered_count);
  EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  EXPECT_FALSE(vector_index_registry::rollback_txn(txn_id));
}

TEST_F(VectorIndexRegistryTest,
       CommittedLoadFailureFallsBackWhenQuarantineSucceeds) {
  const std::string index_name = "idx_registry_committed_quarantine";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  const uint64_t committed_failures_before =
      vector_status::committed_load_failures();
  store_.fail_load_committed = true;
  vector_index_registry::reset_for_testing();

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_EQ(committed_failures_before + 2,
            vector_status::committed_load_failures());

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       ManifestLoadFailureReturnsFalseWhenQuarantineFails) {
  store_.fail_load_manifest = true;
  store_.fail_quarantine_manifest = true;

  std::vector<std::string> index_names;
  EXPECT_FALSE(vector_index_registry::list_indexes(&index_names));
}

TEST_F(VectorIndexRegistryTest,
       ManifestLoadFailurePropagatesAcrossRegistryWrappers) {
  store_.fail_load_manifest = true;
  store_.fail_quarantine_manifest = true;

  std::vector<std::string> index_names;
  vector_index_registry::index_info info;
  std::vector<vector_index::search_result> result;
  size_t rebuilt_count = 99;
  size_t recovered_count = 99;
  const uint64_t txn_id = vector_index_registry::begin_txn();

  EXPECT_FALSE(vector_index_registry::list_indexes(&index_names));
  EXPECT_FALSE(vector_index_registry::get_index_info("missing", &info));
  EXPECT_FALSE(
      vector_index_registry::search("missing", {1.0F, 1.0F}, 1, &result));
  EXPECT_FALSE(vector_index_registry::rebuild_all_indexes(&rebuilt_count));
  EXPECT_EQ(99U, rebuilt_count);
  EXPECT_FALSE(vector_index_registry::recover_all_indexes(&recovered_count));
  EXPECT_EQ(99U, recovered_count);
  EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  EXPECT_FALSE(vector_index_registry::rollback_txn(txn_id));
}

TEST_F(VectorIndexRegistryTest,
       ManifestLoadFailureFallsBackWhenQuarantineSucceeds) {
  const std::string index_name = "idx_registry_manifest_quarantine";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  const uint64_t metadata_failures_before =
      vector_status::metadata_load_failures();
  const uint64_t manifest_failures_before =
      vector_status::manifest_load_failures();
  store_.fail_load_manifest = true;
  vector_index_registry::reset_for_testing();

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_EQ(metadata_failures_before + 1,
            vector_status::metadata_load_failures());
  EXPECT_EQ(manifest_failures_before + 1,
            vector_status::manifest_load_failures());

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       ChangeLogLoadFailureReturnsFalseWhenQuarantineFails) {
  store_.fail_load_change_log = true;
  store_.fail_quarantine_change_log = true;

  std::vector<std::string> index_names;
  EXPECT_FALSE(vector_index_registry::list_indexes(&index_names));
}

TEST_F(VectorIndexRegistryTest,
       ChangeLogLoadFailurePropagatesAcrossRegistryWrappers) {
  store_.fail_load_change_log = true;
  store_.fail_quarantine_change_log = true;

  std::vector<std::string> index_names;
  vector_index_registry::index_info info;
  std::vector<vector_index::search_result> result;
  size_t rebuilt_count = 99;
  size_t recovered_count = 99;
  const uint64_t txn_id = vector_index_registry::begin_txn();

  EXPECT_FALSE(vector_index_registry::list_indexes(&index_names));
  EXPECT_FALSE(vector_index_registry::get_index_info("missing", &info));
  EXPECT_FALSE(
      vector_index_registry::search("missing", {1.0F, 1.0F}, 1, &result));
  EXPECT_FALSE(vector_index_registry::rebuild_all_indexes(&rebuilt_count));
  EXPECT_EQ(99U, rebuilt_count);
  EXPECT_FALSE(vector_index_registry::recover_all_indexes(&recovered_count));
  EXPECT_EQ(99U, recovered_count);
  EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  EXPECT_FALSE(vector_index_registry::rollback_txn(txn_id));
}

TEST_F(VectorIndexRegistryTest,
       ChangeLogLoadFailureFallsBackWhenQuarantineSucceeds) {
  const std::string index_name = "idx_registry_changelog_quarantine";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  const uint64_t metadata_failures_before =
      vector_status::metadata_load_failures();
  const uint64_t changelog_failures_before =
      vector_status::change_log_load_failures();
  store_.fail_load_change_log = true;
  vector_index_registry::reset_for_testing();

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_EQ(metadata_failures_before + 2,
            vector_status::metadata_load_failures());
  EXPECT_EQ(changelog_failures_before + 2,
            vector_status::change_log_load_failures());

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       PreparedLoadFailurePropagatesAcrossPreparedWrappers) {
  store_.fail_load_prepared = true;
  store_.fail_quarantine_prepared = true;

  XA_recover_txn recovered[2]{};
  XID xid = make_test_xid(123, "prepared-load-fail");
  auto xa_state_tuple = vector_gunit::make_xa_state_list_for_testing();
  auto *xa_state_list = xa_state_tuple.get();
  ASSERT_NE(nullptr, xa_state_list);

  EXPECT_EQ(
      0, vector_index_registry::recover_prepared_xids(recovered, 2, nullptr));
  EXPECT_EQ(1, vector_index_registry::recover_prepared_in_tc(*xa_state_list));
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));
  EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  EXPECT_EQ(XAER_RMERR, vector_index_registry::rollback_prepared_xid(xid));
  EXPECT_EQ(XAER_RMERR,
            vector_index_registry::commit_prepared_xid_for_thd(77, xid));
  EXPECT_EQ(XAER_RMERR,
            vector_index_registry::rollback_prepared_xid_for_thd(77, xid));
}

TEST_F(VectorIndexRegistryTest,
       PreparedLoadFailureFallsBackWhenQuarantineSucceeds) {
  store_.fail_load_prepared = true;

  XA_recover_txn recovered[2]{};
  XID xid = make_test_xid(124, "prepared-load-quarantine");
  auto xa_state_tuple = vector_gunit::make_xa_state_list_for_testing();
  auto *xa_state_list = xa_state_tuple.get();
  ASSERT_NE(nullptr, xa_state_list);

  EXPECT_EQ(
      0, vector_index_registry::recover_prepared_xids(recovered, 2, nullptr));
  EXPECT_EQ(0, vector_index_registry::recover_prepared_in_tc(*xa_state_list));
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));
  EXPECT_EQ(XAER_NOTA, vector_index_registry::commit_prepared_xid(xid));
  EXPECT_EQ(XAER_NOTA, vector_index_registry::rollback_prepared_xid(xid));
  EXPECT_EQ(XAER_NOTA,
            vector_index_registry::commit_prepared_xid_for_thd(77, xid));
  EXPECT_EQ(XAER_NOTA,
            vector_index_registry::rollback_prepared_xid_for_thd(77, xid));
}

TEST_F(VectorIndexRegistryTest,
       ChangeLogReplayFailureReturnsFalseWhenQuarantineFails) {
  vector_index_metadata_store::metadata_row row;
  row.index_name = "idx_registry_replay_quarantine_fail";
  row.dimension = 2;
  row.metric = vector_index::metric_type::kEuclidean;
  row.mode = vector_index::backend_mode::kMemory;
  row.provider = vector_index::backend_provider::kNative;
  row.lifecycle_state = "ready";
  row.lifecycle_version = 1;
  row.last_error_code = 0;
  row.last_error_ts = 0;
  row.recover_fallback_count = 0;
  row.last_recover_fallback_ts = 0;
  store_.save_metadata({row});

  vector_index_metadata_store::change_log_row invalid_row;
  invalid_row.sequence = 1;
  invalid_row.txn_id = 1;
  invalid_row.op = vector_index_metadata_store::change_op::kUpsert;
  invalid_row.index_name = "missing.index";
  invalid_row.doc_id = 11;
  invalid_row.vector = {1.0F, 1.0F};
  store_.SetChangeLogRowsForTesting({invalid_row});
  store_.fail_quarantine_change_log = true;

  std::vector<std::string> index_names;
  EXPECT_FALSE(vector_index_registry::list_indexes(&index_names));
}

TEST_F(VectorIndexRegistryTest,
       ThdTxnStatementHelpersCoverMismatchAndEmptyNameBranches) {
  const std::string index_name = "idx_registry_stmt_mismatch";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      300, 10, index_name, 5, {5.0F, 5.0F}));

  EXPECT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(300, 99));
  EXPECT_TRUE(vector_index_registry::rollback_stmt_for_thd_txn(300, 99));
  EXPECT_FALSE(vector_index_registry::savepoint_thd_txn(300, ""));
  EXPECT_FALSE(vector_index_registry::rollback_to_savepoint_thd_txn(300, ""));
  EXPECT_FALSE(vector_index_registry::release_savepoint_thd_txn(300, ""));

  ASSERT_TRUE(vector_index_registry::commit_thd_txn(300));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {5.0F, 5.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(5U, result[0].doc_id);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, ThdTxnHelpersTreatMissingContextAsNoop) {
  XID xid = make_test_xid(100, "missing-thd");

  EXPECT_TRUE(vector_index_registry::commit_thd_txn(999));
  EXPECT_TRUE(vector_index_registry::rollback_thd_txn(999));
  EXPECT_TRUE(vector_index_registry::prepare_thd_txn(999, xid));
  EXPECT_TRUE(vector_index_registry::savepoint_thd_txn(999, "sp_missing"));
  EXPECT_TRUE(
      vector_index_registry::rollback_to_savepoint_thd_txn(999, "sp_missing"));
  EXPECT_TRUE(
      vector_index_registry::release_savepoint_thd_txn(999, "sp_missing"));
}

TEST_F(VectorIndexRegistryTest,
       ParseMappedIndexNameForTestingRejectsMalformedInputs) {
  std::string schema_name;
  std::string table_name;
  std::string column_name;

  EXPECT_FALSE(vector_index_registry::parse_mapped_index_name_for_testing(
      "", &schema_name, &table_name, &column_name));
  EXPECT_FALSE(vector_index_registry::parse_mapped_index_name_for_testing(
      ".t.c", &schema_name, &table_name, &column_name));
  EXPECT_FALSE(vector_index_registry::parse_mapped_index_name_for_testing(
      "db..c", &schema_name, &table_name, &column_name));
  EXPECT_FALSE(vector_index_registry::parse_mapped_index_name_for_testing(
      "db.t.", &schema_name, &table_name, &column_name));
  EXPECT_FALSE(vector_index_registry::parse_mapped_index_name_for_testing(
      "db.t.c.extra", &schema_name, &table_name, &column_name));

  EXPECT_TRUE(vector_index_registry::parse_mapped_index_name_for_testing(
      "db_ok.tbl_ok.col_ok", &schema_name, &table_name, &column_name));
  EXPECT_EQ("db_ok", schema_name);
  EXPECT_EQ("tbl_ok", table_name);
  EXPECT_EQ("col_ok", column_name);

  EXPECT_TRUE(vector_index_registry::parse_mapped_index_name_for_testing(
      "db_ok.tbl_ok.col_ok", nullptr, nullptr, nullptr));
}

TEST_F(VectorIndexRegistryTest, MappedIndexIdentityHelpersPreserveFormat) {
  vector_index_identity::mapped_index_identity identity;

  EXPECT_EQ("db1.t1.v1",
            vector_index_identity::make_index_name("db1", "t1", "v1"));
  EXPECT_EQ("db1.", vector_index_identity::make_schema_prefix("db1"));
  EXPECT_EQ("db1.t1.", vector_index_identity::make_table_prefix("db1", "t1"));
  ASSERT_TRUE(vector_index_identity::parse_index_name("db1.t1.v1", &identity));
  EXPECT_EQ("db1", identity.schema_name);
  EXPECT_EQ("t1", identity.table_name);
  EXPECT_EQ("v1", identity.column_name);
  EXPECT_TRUE(vector_index_identity::matches_schema("db1.t1.v1", "db1"));
  EXPECT_TRUE(vector_index_identity::matches_table("db1.t1.v1", "db1", "t1"));
  EXPECT_FALSE(vector_index_identity::matches_table("db1.t2.v1", "db1", "t1"));

  std::string renamed_index_name;
  EXPECT_TRUE(vector_index_identity::replace_table_name(
      "db1.t1.v1", "db1", "t1", "db2", "t2", &renamed_index_name));
  EXPECT_EQ("db2.t2.v1", renamed_index_name);
  EXPECT_TRUE(vector_index_identity::replace_table_name(
      "db1.t3.v1", "db1", "t1", "db2", "t2", &renamed_index_name));
  EXPECT_EQ("db1.t3.v1", renamed_index_name);
  EXPECT_FALSE(vector_index_identity::replace_table_name(
      "db1.t1.v1", "db1", "t1", "db2", "t2", nullptr));
}

TEST_F(VectorIndexRegistryTest, CreateIndexRejectsInvalidDefinitions) {
  EXPECT_FALSE(vector_index_registry::create_index(
      "idx_bad_dim", 0, "euclidean", "memory", "native"));
  EXPECT_FALSE(vector_index_registry::create_index(
      "idx_bad_metric", 2, "not_a_metric", "memory", "native"));
  EXPECT_FALSE(vector_index_registry::create_index(
      "idx_bad_mode", 2, "euclidean", "not_a_mode", "native"));
  EXPECT_FALSE(vector_index_registry::create_index(
      "idx_bad_provider", 2, "euclidean", "memory", "not_a_provider"));
}

TEST_F(VectorIndexRegistryTest, CreateMappedIndexRejectsInvalidDefinitions) {
  EXPECT_FALSE(vector_index_registry::create_mapped_index(
      "db_bad.t_bad.v", 0, "euclidean", "memory", "native", "db_bad", "t_bad",
      "v", "id"));
  EXPECT_FALSE(vector_index_registry::create_mapped_index(
      "db_bad.t_bad.v", 2, "not_a_metric", "memory", "native", "db_bad",
      "t_bad", "v", "id"));
  EXPECT_FALSE(vector_index_registry::create_mapped_index(
      "db_bad.t_bad.v", 2, "euclidean", "not_a_mode", "native", "db_bad",
      "t_bad", "v", "id"));
  EXPECT_FALSE(vector_index_registry::create_mapped_index(
      "db_bad.t_bad.v", 2, "euclidean", "memory", "not_a_provider", "db_bad",
      "t_bad", "v", "id"));
}

TEST_F(VectorIndexRegistryTest,
       TxnWrapperHelpersCoverPendingAndSavepointEdges) {
  const std::string index_name = "idx_registry_pending_edges";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  const uint64_t txn_id = vector_index_registry::begin_txn();
  EXPECT_EQ(0U, vector_index_registry::pending_txn_changes(txn_id));
  EXPECT_TRUE(vector_index_registry::savepoint_txn(9999, "sp_missing"));
  EXPECT_TRUE(vector_index_registry::release_savepoint_txn(9999, "sp_missing"));

  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));
  EXPECT_EQ(1U, vector_index_registry::pending_txn_changes(txn_id));
  EXPECT_TRUE(vector_index_registry::savepoint_txn(txn_id, "sp1"));
  EXPECT_TRUE(vector_index_registry::release_savepoint_txn(txn_id, "sp1"));

  ASSERT_TRUE(vector_index_registry::rollback_txn(txn_id));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, StageEraseForThdTxnCoversSuccessAndFailure) {
  const std::string index_name = "idx_registry_stage_erase";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 7, {7.0F, 7.0F}));

  EXPECT_FALSE(vector_index_registry::stage_erase_for_thd_txn(
      500, 1, "missing.index", 7));

  ASSERT_TRUE(
      vector_index_registry::stage_erase_for_thd_txn(500, 1, index_name, 7));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(500, 1));
  ASSERT_TRUE(vector_index_registry::commit_thd_txn(500));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {7.0F, 7.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
}

TEST_F(VectorIndexRegistryTest,
       ThdStmtHelpersTreatMissingContextAndMissingIndexAsNoopOrFailure) {
  EXPECT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(9000, 1));
  EXPECT_TRUE(vector_index_registry::rollback_stmt_for_thd_txn(9000, 1));
  EXPECT_FALSE(vector_index_registry::stage_upsert_for_thd_txn(
      9000, 1, "missing.index", 5, {1.0F, 1.0F}));
}

TEST_F(VectorIndexRegistryTest,
       RenameIndexBindingForTestingPreservesOrReplacesBindingShape) {
  const std::string mapped_index_name = "dbbind.t1.v";
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      mapped_index_name, 2, "euclidean", "memory", "native", "dbbind", "t1",
      "v", "id"));

  std::string schema_name;
  std::string table_name;
  std::string column_name;
  std::string doc_id_column_name;
  ASSERT_TRUE(vector_index_registry::get_index_binding_for_testing(
      mapped_index_name, &schema_name, &table_name, &column_name,
      &doc_id_column_name));
  EXPECT_EQ("dbbind", schema_name);
  EXPECT_EQ("t1", table_name);
  EXPECT_EQ("v", column_name);
  EXPECT_EQ("id", doc_id_column_name);

  vector_index_registry::rename_index_binding_for_testing(mapped_index_name,
                                                          "standalone_target");
  ASSERT_TRUE(vector_index_registry::get_index_binding_for_testing(
      "standalone_target", &schema_name, &table_name, &column_name,
      &doc_id_column_name));
  EXPECT_EQ("dbbind", schema_name);
  EXPECT_EQ("t1", table_name);
  EXPECT_EQ("v", column_name);
  EXPECT_EQ("id", doc_id_column_name);

  vector_index_registry::rename_index_binding_for_testing("standalone_target",
                                                          "dbnew.t2.vec2");
  ASSERT_TRUE(vector_index_registry::get_index_binding_for_testing(
      "dbnew.t2.vec2", &schema_name, &table_name, &column_name,
      &doc_id_column_name));
  EXPECT_EQ("dbnew", schema_name);
  EXPECT_EQ("t2", table_name);
  EXPECT_EQ("vec2", column_name);
  EXPECT_TRUE(doc_id_column_name.empty());

  vector_index_registry::rename_index_binding_for_testing("missing.index",
                                                          "other.index");
  EXPECT_FALSE(vector_index_registry::get_index_binding_for_testing(
      "other.index", &schema_name, &table_name, &column_name,
      &doc_id_column_name));
}

TEST_F(VectorIndexRegistryTest,
       SnapshotRuntimeStateForTestingRejectsNullsAndCapturesCommittedRows) {
  const std::string index_name = "idx_registry_snapshot_wrapper";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 17, {1.0F, 7.0F}));

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  vector_index::index_service::committed_state committed_state;
  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  std::vector<std::string> lagging_index_names;

  EXPECT_FALSE(vector_index_registry::snapshot_runtime_state_for_testing(
      nullptr, &committed_state, &change_log_rows, &lagging_index_names));
  EXPECT_FALSE(vector_index_registry::snapshot_runtime_state_for_testing(
      &metadata_rows, nullptr, &change_log_rows, &lagging_index_names));
  EXPECT_FALSE(vector_index_registry::snapshot_runtime_state_for_testing(
      &metadata_rows, &committed_state, nullptr, &lagging_index_names));
  EXPECT_FALSE(vector_index_registry::snapshot_runtime_state_for_testing(
      &metadata_rows, &committed_state, &change_log_rows, nullptr));

  ASSERT_TRUE(vector_index_registry::snapshot_runtime_state_for_testing(
      &metadata_rows, &committed_state, &change_log_rows,
      &lagging_index_names));
  ASSERT_EQ(1U, metadata_rows.size());
  ASSERT_EQ(1U, committed_state.size());
  ASSERT_EQ(1U, committed_state[index_name].size());
  EXPECT_FALSE(change_log_rows.empty());
  EXPECT_TRUE(lagging_index_names.empty());
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnDoesNotPersistNoopPendingChangesToChangeLog) {
  const std::string index_name = "idx_registry_changelog_noop_delta";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::stage_erase(txn_id, index_name, 2));
  ASSERT_TRUE(vector_index_registry::stage_erase(txn_id, index_name, 3));
  ASSERT_TRUE(vector_index_registry::commit_txn(txn_id));

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  vector_index::index_service::committed_state committed_state;
  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  std::vector<std::string> lagging_index_names;
  ASSERT_TRUE(vector_index_registry::snapshot_runtime_state_for_testing(
      &metadata_rows, &committed_state, &change_log_rows,
      &lagging_index_names));

  ASSERT_EQ(1U, change_log_rows.size());
  EXPECT_EQ(index_name, change_log_rows[0].index_name);
  EXPECT_EQ(1U, change_log_rows[0].doc_id);
  EXPECT_EQ(vector_index_metadata_store::change_op::kUpsert,
            change_log_rows[0].op);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       RestoreRuntimeStateForTestingRejectsBadMetadataAndMissingLaggingIndex) {
  {
    vector_index_metadata_store::metadata_row invalid_row;
    invalid_row.index_name = "idx_invalid_meta";
    invalid_row.dimension = 2;
    invalid_row.metric = vector_index::metric_type::kEuclidean;
    invalid_row.mode = vector_index::backend_mode::kMemory;
    invalid_row.provider = vector_index::backend_provider::kNative;
    invalid_row.lifecycle_state = "";
    invalid_row.lifecycle_version = 1;
    EXPECT_FALSE(vector_index_registry::restore_runtime_state_for_testing(
        {invalid_row}, {}, {}, {}));
  }

  {
    vector_index_metadata_store::metadata_row row;
    row.index_name = "idx_missing_lagging";
    row.dimension = 2;
    row.metric = vector_index::metric_type::kEuclidean;
    row.mode = vector_index::backend_mode::kMemory;
    row.provider = vector_index::backend_provider::kNative;
    row.lifecycle_state = "ready";
    row.lifecycle_version = 1;
    EXPECT_FALSE(vector_index_registry::restore_runtime_state_for_testing(
        {row}, {}, {}, {"missing.index"}));
  }

  {
    vector_index_metadata_store::metadata_row row;
    row.index_name = "idx_bad_tuning";
    row.dimension = 2;
    row.metric = vector_index::metric_type::kEuclidean;
    row.mode = vector_index::backend_mode::kMemory;
    row.provider = vector_index::backend_provider::kNative;
    row.lifecycle_state = "ready";
    row.lifecycle_version = 1;
    row.hnsw_m = 16;
    row.hnsw_ef_construction = 32;
    EXPECT_FALSE(vector_index_registry::restore_runtime_state_for_testing(
        {row}, {}, {}, {}));
  }
}

TEST_F(VectorIndexRegistryTest,
       RestoreRuntimeStateForTestingRestoresServingDataAndBindings) {
  vector_index_metadata_store::metadata_row row;
  row.index_name = "dbrestore.t1.v";
  row.dimension = 2;
  row.metric = vector_index::metric_type::kEuclidean;
  row.mode = vector_index::backend_mode::kMemory;
  row.provider = vector_index::backend_provider::kNative;
  row.schema_name = "dbrestore";
  row.table_name = "t1";
  row.column_name = "v";
  row.doc_id_column_name = "id";
  row.lifecycle_state = "ready";
  row.lifecycle_version = 1;

  vector_index::index_service::committed_state committed_state;
  committed_state[row.index_name][55] = {5.0F, 5.0F};

  vector_index_metadata_store::change_log_row row_change;
  row_change.sequence = 9;
  row_change.txn_id = 3;
  row_change.op = vector_index_metadata_store::change_op::kUpsert;
  row_change.index_name = row.index_name;
  row_change.doc_id = 55;
  row_change.vector = {5.0F, 5.0F};

  ASSERT_TRUE(vector_index_registry::restore_runtime_state_for_testing(
      {row}, committed_state, {row_change}, {}));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(row.index_name, &info));
  EXPECT_EQ("dbrestore", info.schema_name);
  EXPECT_EQ("t1", info.table_name);
  EXPECT_EQ("v", info.column_name);
  EXPECT_EQ(1U, info.entry_count);
  EXPECT_EQ(1U, info.committed_entry_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(row.index_name, {5.0F, 5.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(55U, result[0].doc_id);
}

TEST_F(VectorIndexRegistryTest,
       CommitThdTxnReturnsFalseWhenPendingIndexNamesCollectionIsInjected) {
  const std::string index_name = "idx_registry_commit_names_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      610, 1, index_name, 1, {1.0F, 1.0F}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_registry_fail_pending_change_index_names");
    EXPECT_FALSE(vector_index_registry::commit_thd_txn(610));
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(610));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitThdTxnReturnsFalseWhenBeforeSnapshotIsInjected) {
  const std::string index_name = "idx_registry_commit_snapshot_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      611, 1, index_name, 2, {2.0F, 2.0F}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_registry_fail_snapshot_committed_before_commit");
    EXPECT_FALSE(vector_index_registry::commit_thd_txn(611));
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {2.0F, 2.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(611));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       PrepareThdTxnReturnsFalseWhenPreparedSnapshotIsInjected) {
  const std::string index_name = "idx_registry_prepare_snapshot_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      612, 1, index_name, 3, {3.0F, 3.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(612, 1));

  XID xid = make_test_xid(612, "prepare-snapshot-fail");
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_registry_fail_snapshot_prepared_rows_for_thd_txn");
    EXPECT_FALSE(vector_index_registry::prepare_thd_txn(612, xid));
  }

  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));
  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(612));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {3.0F, 3.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       RollbackStmtForThdTxnReturnsFalseAfterInjectedReleaseFailure) {
  const std::string index_name = "idx_registry_stmt_release_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      613, 11, index_name, 4, {4.0F, 4.0F}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_registry_fail_release_stmt_savepoint_after_rollback");
    EXPECT_FALSE(vector_index_registry::rollback_stmt_for_thd_txn(613, 11));
  }

  ASSERT_TRUE(vector_index_registry::commit_thd_txn(613));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {4.0F, 4.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidReturnsRmErrWhenReplayRestoreIsInjected) {
  const std::string index_name = "idx_registry_prepared_restore_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      614, 1, index_name, 5, {5.0F, 5.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(614, 1));

  XID xid = make_test_xid(614, "prepared-restore");
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(614, xid));
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_registry_fail_restore_pending_for_prepared_replay");
    EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  }

  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {5.0F, 5.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
}

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidReturnsRmErrWhenReplaySnapshotIsInjected) {
  const std::string index_name = "idx_registry_prepared_snapshot_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      615, 1, index_name, 6, {6.0F, 6.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(615, 1));

  XID xid = make_test_xid(615, "prepared-snapshot");
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(615, xid));
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_registry_fail_snapshot_before_prepared_commit");
    EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  }

  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {6.0F, 6.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
}

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidReturnsRmErrWhenReplayCommitIsInjected) {
  const std::string index_name = "idx_registry_prepared_commit_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      616, 1, index_name, 7, {7.0F, 7.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(616, 1));

  XID xid = make_test_xid(616, "prepared-commit");
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(616, xid));
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
                             "+d,vector_registry_fail_commit_prepared_replay");
    EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  }

  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {7.0F, 7.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
}

TEST_F(VectorIndexRegistryTest, CommitThdTxnReturnsFalseWhenBeginPersistFails) {
  const std::string index_name = "idx_registry_commit_begin_persist_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      617, 1, index_name, 8, {8.0F, 8.0F}));

  store_.fail_begin_persist = true;
  EXPECT_FALSE(vector_index_registry::commit_thd_txn(617));
  store_.fail_begin_persist = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {8.0F, 8.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(617));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitThdTxnReturnsFalseWhenManifestPersistFails) {
  const std::string index_name = "idx_registry_commit_manifest_persist_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      618, 1, index_name, 9, {9.0F, 9.0F}));

  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::commit_thd_txn(618));
  store_.fail_save_manifest = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {9.0F, 9.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(618));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitThdTxnReturnsFalseWhenChangeLogPersistFails) {
  const std::string index_name = "idx_registry_commit_changelog_persist_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      619, 1, index_name, 10, {10.0F, 10.0F}));

  store_.fail_append_change_log_delta = true;
  EXPECT_FALSE(vector_index_registry::commit_thd_txn(619));
  store_.fail_append_change_log_delta = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {10.0F, 10.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(619));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitThdTxnReturnsFalseWhenCommitPersistFails) {
  const std::string index_name = "idx_registry_commit_commit_persist_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      620, 1, index_name, 11, {11.0F, 11.0F}));

  store_.fail_commit_persist = true;
  EXPECT_FALSE(vector_index_registry::commit_thd_txn(620));
  store_.fail_commit_persist = false;

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {11.0F, 11.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(620));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       RenameIndexesForTableMovesMultipleMappedIndexesAcrossDatabase) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbbulk.t1.v1", 2, "euclidean", "memory", "native", "dbbulk", "t1", "v1",
      "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbbulk.t1.v2", 2, "euclidean", "memory", "native", "dbbulk", "t1", "v2",
      "id"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_bulk_standalone", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert("dbbulk.t1.v1", 21, {2.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::upsert("dbbulk.t1.v2", 22, {2.0F, 2.0F}));
  ASSERT_TRUE(
      vector_index_registry::upsert("idx_bulk_standalone", 23, {9.0F, 9.0F}));

  ASSERT_TRUE(vector_index_registry::rename_indexes_for_table("dbbulk", "t1",
                                                              "dbbulk2", "t2"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info("dbbulk2.t2.v1", &info));
  EXPECT_EQ("dbbulk2", info.schema_name);
  EXPECT_EQ("t2", info.table_name);
  EXPECT_EQ("v1", info.column_name);
  ASSERT_TRUE(vector_index_registry::get_index_info("dbbulk2.t2.v2", &info));
  EXPECT_EQ("v2", info.column_name);
  ASSERT_FALSE(vector_index_registry::get_index_info("dbbulk.t1.v1", &info));
  ASSERT_FALSE(vector_index_registry::get_index_info("dbbulk.t1.v2", &info));
  ASSERT_TRUE(
      vector_index_registry::get_index_info("idx_bulk_standalone", &info));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search("dbbulk2.t2.v1", {2.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(21U, result[0].doc_id);
  ASSERT_TRUE(
      vector_index_registry::search("dbbulk2.t2.v2", {2.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(22U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("dbbulk2.t2.v1"));
  ASSERT_TRUE(vector_index_registry::drop_index("dbbulk2.t2.v2"));
  ASSERT_TRUE(vector_index_registry::drop_index("idx_bulk_standalone"));
}

TEST_F(VectorIndexRegistryTest,
       DropIndexesForDatabaseRemovesMultipleMatchingIndexesOnly) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbpurge.t1.v1", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbpurge.t2.v2", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index("dbkeep.t1.v", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert("dbpurge.t1.v1", 31, {3.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::upsert("dbpurge.t2.v2", 32, {3.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::upsert("dbkeep.t1.v", 33, {3.0F, 3.0F}));

  ASSERT_TRUE(vector_index_registry::drop_indexes_for_database("dbpurge"));

  vector_index_registry::index_info info;
  ASSERT_FALSE(vector_index_registry::get_index_info("dbpurge.t1.v1", &info));
  ASSERT_FALSE(vector_index_registry::get_index_info("dbpurge.t2.v2", &info));
  ASSERT_TRUE(vector_index_registry::get_index_info("dbkeep.t1.v", &info));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search("dbkeep.t1.v", {3.0F, 3.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(33U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("dbkeep.t1.v"));
}

TEST_F(
    VectorIndexRegistryTest,
    ResetMappedIndexesForTableResetsMultipleMappedIndexesAndPreservesOthers) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbresetmulti.t1.v1", 2, "euclidean", "memory", "native", "dbresetmulti",
      "t1", "v1", "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbresetmulti.t1.v2", 2, "euclidean", "memory", "native", "dbresetmulti",
      "t1", "v2", "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbresetmulti.t2.v3", 2, "euclidean", "memory", "native", "dbresetmulti",
      "t2", "v3", "id"));
  ASSERT_TRUE(
      vector_index_registry::upsert("dbresetmulti.t1.v1", 41, {4.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::upsert("dbresetmulti.t1.v2", 42, {4.0F, 2.0F}));
  ASSERT_TRUE(
      vector_index_registry::upsert("dbresetmulti.t2.v3", 43, {4.0F, 3.0F}));

  ASSERT_TRUE(vector_index_registry::reset_mapped_indexes_for_table(
      "dbresetmulti", "t1"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(
      vector_index_registry::get_index_info("dbresetmulti.t1.v1", &info));
  EXPECT_EQ(0U, info.entry_count);
  EXPECT_EQ(0U, info.committed_entry_count);
  ASSERT_TRUE(
      vector_index_registry::get_index_info("dbresetmulti.t1.v2", &info));
  EXPECT_EQ(0U, info.entry_count);
  EXPECT_EQ(0U, info.committed_entry_count);
  ASSERT_TRUE(
      vector_index_registry::get_index_info("dbresetmulti.t2.v3", &info));
  EXPECT_EQ(1U, info.entry_count);
  EXPECT_EQ(1U, info.committed_entry_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("dbresetmulti.t1.v1", {4.0F, 1.0F},
                                            1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::search("dbresetmulti.t1.v2", {4.0F, 2.0F},
                                            1, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(vector_index_registry::search("dbresetmulti.t2.v3", {4.0F, 3.0F},
                                            1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(43U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("dbresetmulti.t1.v1"));
  ASSERT_TRUE(vector_index_registry::drop_index("dbresetmulti.t1.v2"));
  ASSERT_TRUE(vector_index_registry::drop_index("dbresetmulti.t2.v3"));
}

TEST_F(VectorIndexRegistryTest,
       SearchForThdTxnUsesPendingUpsertsOnlyForMatchingThdContext) {
  const std::string index_name = "idx_registry_search_pending_upsert";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 51, {10.0F, 10.0F}));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      701, 1, index_name, 52, {1.0F, 1.0F}));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search_for_thd_txn(
      nullptr, 701, index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(52U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::search_for_thd_txn(
      nullptr, 999, index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(51U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(701));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       SearchForThdTxnUsesPendingEraseOnlyForMatchingThdContext) {
  const std::string index_name = "idx_registry_search_pending_erase";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 61, {1.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::stage_erase_for_thd_txn(702, 1, index_name, 61));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search_for_thd_txn(
      nullptr, 702, index_name, {1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_TRUE(vector_index_registry::search_for_thd_txn(
      nullptr, 999, index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(61U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(702));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       SearchForThdTxnSkipsMappedVisibilityFilterWithoutThd) {
  const std::string index_name = "dbsearch.t1.v";
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      index_name, 2, "euclidean", "memory", "native", "dbsearch", "t1", "v",
      "id"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 71, {7.0F, 1.0F}));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search_for_thd_txn(
      nullptr, 0, index_name, {7.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(71U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, MappedSearchCandidatePlannerUsesBoundedChunks) {
  const size_t max_size = std::numeric_limits<size_t>::max();

  EXPECT_EQ(max_size, vector_index::saturated_add_size(max_size, 1));
  EXPECT_TRUE(vector_index_registry::detail::mapped_search_candidate_limit_is_hard(
      vector_index::k_max_search_top_k, 1));
  EXPECT_FALSE(vector_index_registry::detail::mapped_search_candidate_limit_is_hard(
      10, 5));
  EXPECT_EQ(0U, vector_index_registry::detail::mapped_search_candidate_limit(
                    0, 100, 20));
  EXPECT_EQ(15U, vector_index_registry::detail::mapped_search_candidate_limit(
                     10, 12, 3));
  EXPECT_EQ(vector_index::k_max_search_top_k,
            vector_index_registry::detail::mapped_search_candidate_limit(
                10, vector_index::k_max_search_top_k + 1, 100));

  EXPECT_EQ(15U,
            vector_index_registry::detail::mapped_search_initial_candidate_top_k(
                10, 12, 3));
  EXPECT_EQ(32U,
            vector_index_registry::detail::mapped_search_initial_candidate_top_k(
                1, 100, 0));
  EXPECT_EQ(64U,
            vector_index_registry::detail::mapped_search_next_candidate_top_k(
                32, 1, 100));
  EXPECT_EQ(100U,
            vector_index_registry::detail::mapped_search_next_candidate_top_k(
                96, 10, 100));
  EXPECT_EQ(max_size,
            vector_index_registry::detail::mapped_search_next_candidate_top_k(
                max_size - 1, 1, max_size));
  EXPECT_EQ(max_size,
            vector_index_registry::detail::mapped_search_next_candidate_top_k(
                10, max_size, max_size));
}

TEST_F(VectorIndexRegistryTest,
       SearchBatchForThdTxnCoversEmptyMissingAndPendingFallback) {
  std::vector<vector_index::search_result> flat_results;
  std::vector<std::vector<vector_index::search_result>> batches = {
      {{99, 99.0}}};

  EXPECT_FALSE(vector_index_registry::search("missing_batch", {1.0F, 1.0F}, 1,
                                             nullptr));
  EXPECT_FALSE(vector_index_registry::search_for_thd_txn(
      nullptr, 0, "missing_batch", {1.0F, 1.0F}, 1, nullptr));
  EXPECT_FALSE(vector_index_registry::search_for_thd_txn(
      nullptr, 0, "missing_batch", {1.0F, 1.0F}, 1, &flat_results));
  EXPECT_FALSE(vector_index_registry::search_batch_for_thd_txn(
      nullptr, 0, "missing_batch", {{1.0F, 1.0F}}, 1, nullptr));
  ASSERT_TRUE(vector_index_registry::search_batch_for_thd_txn(
      nullptr, 0, "missing_batch", {}, 1, &batches));
  EXPECT_TRUE(batches.empty());
  EXPECT_FALSE(vector_index_registry::search_batch_for_thd_txn(
      nullptr, 0, "missing_batch", {{1.0F, 1.0F}, {2.0F, 2.0F}}, 1,
      &batches));

  const std::string index_name = "idx_registry_search_batch";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 51, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 52, {5.0F, 5.0F}));
  EXPECT_FALSE(vector_index_registry::search_for_thd_txn(
      nullptr, 0, index_name, {1.0F}, 1, &flat_results));
  EXPECT_FALSE(vector_index_registry::search_batch_for_thd_txn(
      nullptr, 0, index_name, {{1.0F}}, 1, &batches));

  ASSERT_TRUE(vector_index_registry::search_batch_for_thd_txn(
      nullptr, 0, index_name, {{1.0F, 1.0F}, {5.0F, 5.0F}}, 1, &batches));
  ASSERT_EQ(2U, batches.size());
  ASSERT_EQ(1U, batches[0].size());
  EXPECT_EQ(51U, batches[0][0].doc_id);
  ASSERT_EQ(1U, batches[1].size());
  EXPECT_EQ(52U, batches[1][0].doc_id);

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      703, 1, index_name, 53, {0.0F, 0.0F}));
  ASSERT_TRUE(vector_index_registry::search_batch_for_thd_txn(
      nullptr, 703, index_name, {{0.0F, 0.0F}, {1.0F, 1.0F}}, 1, &batches));
  ASSERT_EQ(2U, batches.size());
  ASSERT_EQ(1U, batches[0].size());
  EXPECT_EQ(53U, batches[0][0].doc_id);
  ASSERT_EQ(1U, batches[1].size());
  EXPECT_EQ(51U, batches[1][0].doc_id);

  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(703));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       SearchBatchForThdTxnRebuildsHnswRuntimeAfterSearchFailure) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native recovery requires HAVE_HNSWLIB";
  }

  const std::string index_name = "idx_registry_hnsw_batch_recover";
  {
    std::lock_guard<std::shared_mutex> guard(
        vector_index_registry::detail::g_registry_mutex);
    vector_index_registry::detail::g_metadata_loaded = true;
    ASSERT_TRUE(vector_index_registry::detail::g_index_service.register_index(
        index_name,
        std::make_unique<vector_gunit::FailingSearchHnswBackend>(2)));
  }

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      704, 1, index_name, 61, {1.0F, 0.0F}));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      704, 1, index_name, 62, {0.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::commit_thd_txn(704));

  std::vector<std::vector<vector_index::search_result>> batches;
  ASSERT_TRUE(vector_index_registry::search_batch_for_thd_txn(
      nullptr, 0, index_name, {{1.0F, 0.0F}, {0.0F, 1.0F}}, 1, &batches));
  ASSERT_EQ(2U, batches.size());
  ASSERT_EQ(1U, batches[0].size());
  EXPECT_EQ(61U, batches[0][0].doc_id);
  ASSERT_EQ(1U, batches[1].size());
  EXPECT_EQ(62U, batches[1][0].doc_id);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(1U, info.recover_fallback_count);
  EXPECT_GT(info.last_recover_fallback_ts, 0U);
}

TEST_F(VectorIndexRegistryTest,
       GetIndexBindingForTestingHandlesStandaloneAndNullOutputs) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_binding_standalone", 2, "euclidean", "memory", "native"));
  std::string schema_name;
  std::string table_name;
  std::string column_name;
  std::string doc_id_column_name;

  EXPECT_FALSE(vector_index_registry::get_index_binding_for_testing(
      "idx_binding_standalone", &schema_name, &table_name, &column_name,
      &doc_id_column_name));
  EXPECT_TRUE(schema_name.empty());
  EXPECT_TRUE(table_name.empty());
  EXPECT_TRUE(column_name.empty());
  EXPECT_TRUE(doc_id_column_name.empty());

  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbbind2.t1.v", 2, "euclidean", "memory", "native", "dbbind2", "t1", "v",
      "id"));
  EXPECT_TRUE(vector_index_registry::get_index_binding_for_testing(
      "dbbind2.t1.v", nullptr, nullptr, &column_name, nullptr));
  EXPECT_EQ("v", column_name);

  ASSERT_TRUE(vector_index_registry::drop_index("idx_binding_standalone"));
  ASSERT_TRUE(vector_index_registry::drop_index("dbbind2.t1.v"));
}

TEST_F(VectorIndexRegistryTest,
       RenameIndexForColumnMovesOnlySelectedColumnAmongMultipleMappedIndexes) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbrename_multi.t1.v1", 2, "euclidean", "memory", "native",
      "dbrename_multi", "t1", "v1", "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbrename_multi.t1.v2", 2, "euclidean", "memory", "native",
      "dbrename_multi", "t1", "v2", "id"));
  ASSERT_TRUE(
      vector_index_registry::upsert("dbrename_multi.t1.v1", 81, {8.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::upsert("dbrename_multi.t1.v2", 82, {8.0F, 2.0F}));

  ASSERT_TRUE(vector_index_registry::rename_index_for_column("dbrename_multi",
                                                             "t1", "v1", "v3"));

  vector_index_registry::index_info info;
  ASSERT_FALSE(
      vector_index_registry::get_index_info("dbrename_multi.t1.v1", &info));
  ASSERT_TRUE(
      vector_index_registry::get_index_info("dbrename_multi.t1.v2", &info));
  EXPECT_EQ("v2", info.column_name);
  ASSERT_TRUE(
      vector_index_registry::get_index_info("dbrename_multi.t1.v3", &info));
  EXPECT_EQ("v3", info.column_name);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("dbrename_multi.t1.v3",
                                            {8.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(81U, result[0].doc_id);
  ASSERT_TRUE(vector_index_registry::search("dbrename_multi.t1.v2",
                                            {8.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(82U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("dbrename_multi.t1.v2"));
  ASSERT_TRUE(vector_index_registry::drop_index("dbrename_multi.t1.v3"));
}

TEST_F(VectorIndexRegistryTest,
       DropIndexForColumnRemovesOnlySelectedColumnAmongMultipleMappedIndexes) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbdrop_multi.t1.v1", 2, "euclidean", "memory", "native", "dbdrop_multi",
      "t1", "v1", "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbdrop_multi.t1.v2", 2, "euclidean", "memory", "native", "dbdrop_multi",
      "t1", "v2", "id"));
  ASSERT_TRUE(
      vector_index_registry::upsert("dbdrop_multi.t1.v1", 91, {9.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::upsert("dbdrop_multi.t1.v2", 92, {9.0F, 2.0F}));

  ASSERT_TRUE(
      vector_index_registry::drop_index_for_column("dbdrop_multi", "t1", "v1"));

  vector_index_registry::index_info info;
  ASSERT_FALSE(
      vector_index_registry::get_index_info("dbdrop_multi.t1.v1", &info));
  ASSERT_TRUE(
      vector_index_registry::get_index_info("dbdrop_multi.t1.v2", &info));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("dbdrop_multi.t1.v2", {9.0F, 2.0F},
                                            1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(92U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("dbdrop_multi.t1.v2"));
}

}  // namespace vector_index_registry_unittest
