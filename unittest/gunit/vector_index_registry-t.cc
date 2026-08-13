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
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "my_alloc.h"
#include "my_byteorder.h"
#include "sql/mysqld.h"
#include "sql/vector/vector_build_pipeline_policy.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_ddl_formatter.h"
#include "sql/vector/vector_index_identity.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_metadata_store.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_registry_internal.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_load_staging.h"
#include "sql/vector/vector_statement_publication.h"
#include "sql/vector/vector_status.h"
#include "sql_string.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_index_registry_unittest {

namespace {

using vector_gunit::BoolGuard;
using vector_gunit::UlongGuard;
using vector_gunit::UlonglongGuard;

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

class two_thread_start_barrier {
 public:
  bool arrive_and_wait() {
    std::unique_lock<std::mutex> guard(m_mutex);
    ++m_arrived;
    m_condition.notify_all();
    return m_condition.wait_for(guard, std::chrono::seconds(5),
                                [&] { return m_arrived == 2; });
  }

 private:
  std::mutex m_mutex;
  std::condition_variable m_condition;
  size_t m_arrived{0};
};

vector_index_metadata_store::metadata_row make_metadata_row_for_testing(
    const std::string &index_name, const std::string &lifecycle_state) {
  static uint64_t next_index_identity = 1;
  vector_index_metadata_store::metadata_row row;
  row.index_name = index_name;
  row.dimension = 2;
  row.metric = vector_index::metric_type::kEuclidean;
  row.mode = vector_index::backend_mode::kMemory;
  row.provider = vector_index::backend_provider::kNative;
  row.owner_schema = "test";
  row.lifecycle_state = lifecycle_state;
  row.lifecycle_version = lifecycle_state == "ready" ? 1 : 2;
  row.index_identity = next_index_identity++;
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

std::vector<std::string> regular_files_below(const std::string &root) {
  std::vector<std::string> files;
  std::error_code ec;
  if (!std::filesystem::exists(root, ec) || ec) return files;
  for (std::filesystem::recursive_directory_iterator it(root, ec), end;
       !ec && it != end; it.increment(ec)) {
    if (!it->is_regular_file(ec) || ec) continue;
    files.push_back(std::filesystem::relative(it->path(), root, ec).string());
    if (ec) return {};
  }
  std::sort(files.begin(), files.end());
  return files;
}

class external_snapshot_root_guard {
 public:
  explicit external_snapshot_root_guard(const std::string &root) {
    vector_index::set_faiss_external_snapshot_root_for_testing(root);
  }
  ~external_snapshot_root_guard() {
    vector_index::reset_faiss_external_snapshot_root_for_testing();
  }
};

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

TEST(VectorIndexDdlFormatterTest,
     CreateStatementHonorsQualificationAndThreads) {
  auto info = make_formatter_info_for_testing("faiss");
  String query;

  vector_index_ddl_formatter::append_create_statement(
      nullptr, &query, "db", 2, "t", 1, true, "v", info, 8);
  EXPECT_EQ(
      ";\nCREATE VECTOR INDEX ON `db`.`t`(`v`) WITH "
      "(2, 'euclidean', 'memory', 'faiss', 8)",
      string_value_for_testing(query));

  query.length(0);
  vector_index_ddl_formatter::append_create_statement(
      nullptr, &query, "db", 2, "table`name", 10, false, "vec`col", info, 0);
  EXPECT_EQ(
      ";\nCREATE VECTOR INDEX ON `table``name`(`vec``col`) WITH "
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
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_SEARCH_EF('idx_hnsw', 128)"
      ";\nSELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS('idx_hnsw', 32, 400)",
      string_value_for_testing(query));

  auto faiss = make_formatter_info_for_testing("faiss");
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss_default", faiss,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_TRUE(string_value_for_testing(query).empty());

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
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_FAISS_IVFPQ_PARAMS"
      "('idx_faiss', 64, 8, 16, 8)",
      string_value_for_testing(query));

  auto diskann = make_formatter_info_for_testing("diskann");
  diskann.diskann_max_degree = 48;
  diskann.diskann_build_complexity = 96;
  diskann.diskann_build_threads = 4;
  diskann.diskann_build_mode_value = vector_index::diskann_build_mode::kOffline;
  diskann.diskann_build_mode_specified = true;
  diskann.diskann_search_complexity = 96;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_diskann", diskann,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_DISKANN_BUILD_PARAMS"
      "('idx_diskann', 48, 96, 4)"
      ";\nSELECT VEC_INDEX_SET_DISKANN_BUILD_MODE"
      "('idx_diskann', 'offline')"
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
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_SEARCH_EF('idx', 11)"
      ";\nSELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS('idx', 12, 13)"
      ";\nSELECT VEC_INDEX_SET_FAISS_IVFPQ_PARAMS"
      "('idx', 14, 15, 16, 17)"
      ";\nSELECT VEC_INDEX_SET_DISKANN_BUILD_PARAMS('idx', 18, 19, 21)"
      ";\nSELECT VEC_INDEX_SET_DISKANN_SEARCH_COMPLEXITY('idx', 20)",
      string_value_for_testing(query));
}

TEST(VectorIndexDdlFormatterTest,
     NonzeroTuningPolicyCoversTrailingProviderOperands) {
  String query;
  vector_index_registry::index_info info;
  info.provider = "native";

  info.hnsw_ef_construction = 13;
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_hnsw_tail", info,
      vector_index_ddl_formatter::tuning_emit_policy::k_emit_nonzero);
  EXPECT_EQ(";\nSELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS('idx_hnsw_tail', 0, 13)",
            string_value_for_testing(query));

  info = vector_index_registry::index_info{};
  info.provider = "native";
  info.faiss_nprobe = 15;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss_ivf_tail", info,
      vector_index_ddl_formatter::tuning_emit_policy::k_emit_nonzero);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_FAISS_IVF_PARAMS"
      "('idx_faiss_ivf_tail', 0, 15)",
      string_value_for_testing(query));

  info = vector_index_registry::index_info{};
  info.provider = "native";
  info.faiss_pq_bits = 8;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss_pq_tail", info,
      vector_index_ddl_formatter::tuning_emit_policy::k_emit_nonzero);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_FAISS_IVFPQ_PARAMS"
      "('idx_faiss_pq_tail', 0, 0, 0, 8)",
      string_value_for_testing(query));

  info = vector_index_registry::index_info{};
  info.provider = "native";
  info.diskann_build_complexity = 19;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_diskann_complexity_tail", info,
      vector_index_ddl_formatter::tuning_emit_policy::k_emit_nonzero);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_DISKANN_BUILD_PARAMS"
      "('idx_diskann_complexity_tail', 0, 19)",
      string_value_for_testing(query));

  info = vector_index_registry::index_info{};
  info.provider = "native";
  info.diskann_build_threads = 4;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_diskann_threads_tail", info,
      vector_index_ddl_formatter::tuning_emit_policy::k_emit_nonzero);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_DISKANN_BUILD_PARAMS"
      "('idx_diskann_threads_tail', 0, 0, 4)",
      string_value_for_testing(query));
}

TEST(VectorIndexDdlFormatterTest, SkipDefaultsDoesNotLeakOtherProviderTuning) {
  String query;
  auto info = make_formatter_info_for_testing("native");
  info.search_ef = 128;
  info.hnsw_m = 32;
  info.hnsw_ef_construction = 400;
  info.faiss_nlist = 64;
  info.faiss_nprobe = 8;
  info.faiss_pq_m = 16;
  info.faiss_pq_bits = 8;
  info.diskann_max_degree = 48;
  info.diskann_build_complexity = 96;
  info.diskann_build_threads = 4;
  info.diskann_search_complexity = 96;
  info.diskann_search_beamwidth = 16;
  info.diskann_pq_code_budget_size = 1024;
  info.diskann_disk_pq_dims = 8;
  info.diskann_accelerate_build = true;

  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_native", info,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_TRUE(string_value_for_testing(query).empty());
}

TEST(VectorIndexDdlFormatterTest, TuningStatementsCoverTrailingOptionBranches) {
  String query;

  auto hnsw = make_formatter_info_for_testing("hnswlib");
  hnsw.hnsw_m = 32;
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_hnsw_m", hnsw,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS"
      "('idx_hnsw_m', 32, 200)",
      string_value_for_testing(query));

  query.length(0);
  hnsw = make_formatter_info_for_testing("hnswlib");
  hnsw.hnsw_ef_construction = 400;
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_hnsw_ef", hnsw,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS"
      "('idx_hnsw_ef', 16, 400)",
      string_value_for_testing(query));

  auto faiss = make_formatter_info_for_testing("faiss");
  faiss.faiss_pq_bits = 8;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss_pq_bits", faiss,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_FAISS_IVFPQ_PARAMS"
      "('idx_faiss_pq_bits', 0, 0, 0, 8)",
      string_value_for_testing(query));

  faiss = make_formatter_info_for_testing("faiss");
  faiss.faiss_nprobe = 8;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss_nprobe", faiss,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_FAISS_IVF_PARAMS"
      "('idx_faiss_nprobe', 0, 8)",
      string_value_for_testing(query));

  faiss = make_formatter_info_for_testing("faiss");
  faiss.faiss_nlist = 64;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss_nlist", faiss,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_FAISS_IVF_PARAMS"
      "('idx_faiss_nlist', 64, 0)",
      string_value_for_testing(query));

  faiss = make_formatter_info_for_testing("faiss");
  faiss.faiss_pq_m = 8;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_faiss_pq_m", faiss,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_FAISS_IVFPQ_PARAMS"
      "('idx_faiss_pq_m', 0, 0, 8, 0)",
      string_value_for_testing(query));

  auto diskann = make_formatter_info_for_testing("diskann");
  diskann.diskann_build_complexity = 96;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_diskann_complexity", diskann,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_DISKANN_BUILD_PARAMS"
      "('idx_diskann_complexity', 32, 96)",
      string_value_for_testing(query));

  diskann = make_formatter_info_for_testing("diskann");
  diskann.diskann_build_threads = 4;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_diskann_threads", diskann,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_DISKANN_BUILD_PARAMS"
      "('idx_diskann_threads', 32, 64, 4)",
      string_value_for_testing(query));

  diskann = make_formatter_info_for_testing("diskann");
  diskann.diskann_search_beamwidth = 24;
  diskann.diskann_pq_code_budget_size = 1024;
  diskann.diskann_disk_pq_dims = 8;
  diskann.diskann_accelerate_build = true;
  diskann.diskann_shuffle_build = true;
  diskann.diskann_use_bfs_cache = true;
  query.length(0);
  vector_index_ddl_formatter::append_tuning_statements(
      nullptr, &query, "idx_diskann_trailing", diskann,
      vector_index_ddl_formatter::tuning_emit_policy::k_skip_defaults);
  EXPECT_EQ(
      ";\nSELECT VEC_INDEX_SET_DISKANN_SEARCH_BEAMWIDTH"
      "('idx_diskann_trailing', 24)"
      ";\nSELECT VEC_INDEX_SET_DISKANN_PQ_CODE_BUDGET_SIZE"
      "('idx_diskann_trailing', 1024)"
      ";\nSELECT VEC_INDEX_SET_DISKANN_DISK_PQ_DIMS"
      "('idx_diskann_trailing', 8)"
      ";\nSELECT VEC_INDEX_SET_DISKANN_ACCELERATE_BUILD"
      "('idx_diskann_trailing', 1)"
      ";\nSELECT VEC_INDEX_SET_DISKANN_SHUFFLE_BUILD"
      "('idx_diskann_trailing', 1)"
      ";\nSELECT VEC_INDEX_SET_DISKANN_USE_BFS_CACHE"
      "('idx_diskann_trailing', 1)",
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
  bool supports_publication_intents() const override { return true; }

  bool load_publication_intents(
      std::vector<vector_index_truth_store::publication_intent> *intents)
      override {
    std::unique_lock<std::mutex> guard(publication_intent_load_mutex);
    ++publication_intent_load_calls;
    publication_intent_load_condition.notify_all();
    publication_intent_load_condition.wait(
        guard, [&] { return !block_publication_intent_load; });
    if (fail_load_publication_intents || intents == nullptr) return false;
    *intents = publication_intents;
    return true;
  }

  bool save_publication_intent(
      const vector_index_truth_store::publication_intent &intent) override {
    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    if (fail_save_publication_intent || intent.index_name.empty() ||
        intent.publication_id == 0) {
      return false;
    }
    const auto same_index = [&](const auto &entry) {
      return entry.index_name == intent.index_name;
    };
    if (std::any_of(publication_intents.begin(), publication_intents.end(),
                    same_index) ||
        std::any_of(staged_publication_intents.begin(),
                    staged_publication_intents.end(), same_index)) {
      return false;
    }
    if (persist_active) {
      staged_publication_intents.push_back(intent);
    } else {
      publication_intents.push_back(intent);
    }
    return true;
  }

  bool delete_publication_intent(const std::string &index_name,
                                 uint64_t publication_id) override {
    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    if (fail_delete_publication_intent) return false;
    const auto it = std::find_if(
        publication_intents.begin(), publication_intents.end(),
        [&](const auto &intent) {
          return intent.index_name == index_name &&
                 intent.publication_id == publication_id;
        });
    if (it == publication_intents.end()) return false;
    if (persist_active) {
      staged_deleted_publication_intents.emplace_back(index_name,
                                                      publication_id);
      return true;
    }
    publication_intents.erase(it);
    return true;
  }

  void block_publication_intent_loads() {
    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    block_publication_intent_load = true;
  }

  bool wait_for_publication_intent_loads(size_t expected) {
    std::unique_lock<std::mutex> guard(publication_intent_load_mutex);
    return publication_intent_load_condition.wait_for(
        guard, std::chrono::seconds(5),
        [&] { return publication_intent_load_calls >= expected; });
  }

  void release_publication_intent_loads() {
    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    block_publication_intent_load = false;
    publication_intent_load_condition.notify_all();
  }

  size_t publication_intent_load_count() {
    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    return publication_intent_load_calls;
  }

  void AddPublicationIntentForTesting(
      const vector_index_truth_store::publication_intent &intent) {
    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    publication_intents.push_back(intent);
  }

  bool begin_persist() override {
    ++begin_persist_calls;
    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    if (fail_begin_persist || persist_active) return false;
    persist_active = true;
    staged_committed_rows = committed_rows;
    staged_change_log_rows = change_log_rows;
    staged_publication_intents.clear();
    staged_deleted_publication_intents.clear();
    return true;
  }

  bool commit_persist() override {
    ++commit_persist_calls;
    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    if (!persist_active || fail_commit_persist) return false;
    publication_intents.insert(publication_intents.end(),
                               staged_publication_intents.begin(),
                               staged_publication_intents.end());
    for (const auto &deleted : staged_deleted_publication_intents) {
      publication_intents.erase(
          std::remove_if(publication_intents.begin(), publication_intents.end(),
                         [&](const auto &intent) {
                           return intent.index_name == deleted.first &&
                                  intent.publication_id == deleted.second;
                         }),
          publication_intents.end());
    }
    committed_rows = std::move(staged_committed_rows);
    change_log_rows = std::move(staged_change_log_rows);
    persist_active = false;
    staged_committed_rows.clear();
    staged_change_log_rows.clear();
    staged_publication_intents.clear();
    staged_deleted_publication_intents.clear();
    return true;
  }

  void rollback_persist() override {
    {
      std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
      persist_active = false;
      staged_committed_rows.clear();
      staged_change_log_rows.clear();
      staged_publication_intents.clear();
      staged_deleted_publication_intents.clear();
    }
    {
      std::lock_guard<std::mutex> guard(committed_delta_mutex);
      ++rollback_persist_calls;
    }
    committed_delta_condition.notify_all();
  }

  bool stage_quarantine(const std::string &artifact_name,
                        const std::string &reason, uint64_t generation,
                        std::string *identity) override {
    if (identity == nullptr || fail_stage_quarantine ||
        (artifact_name == "metadata" && fail_quarantine_metadata) ||
        (artifact_name == "committed" && fail_quarantine_committed) ||
        (artifact_name == "manifest" && fail_quarantine_manifest) ||
        (artifact_name == "changelog" && fail_quarantine_change_log) ||
        (artifact_name == "prepared" && fail_quarantine_prepared) ||
        (artifact_name == "segment_tasks" && fail_quarantine_segment_tasks)) {
      return false;
    }
    vector_index_truth_store::quarantine_record record;
    record.identity =
        artifact_name + "-" + std::to_string(++quarantine_identity_sequence);
    record.artifact_name = artifact_name;
    record.reason = reason;
    record.generation = generation;
    quarantine_records.push_back(record);
    *identity = record.identity;
    return true;
  }

  bool update_quarantine_state(
      const std::string &identity,
      vector_index_truth_store::quarantine_state state) override {
    if (fail_update_quarantine_state) return false;
    for (auto &record : quarantine_records) {
      if (record.identity != identity) continue;
      record.state = state;
      return true;
    }
    return false;
  }

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
    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    (persist_active ? staged_committed_rows : committed_rows) = rows;
    return true;
  }

  bool apply_committed_delta(
      const std::vector<vector_index_metadata_store::change_log_row> &rows)
      override {
    {
      std::unique_lock<std::mutex> guard(committed_delta_mutex);
      ++apply_committed_delta_calls;
      committed_delta_condition.notify_all();
      committed_delta_condition.wait(
          guard, [&] { return !block_committed_delta; });
    }
    if (fail_apply_committed_delta) return false;

    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    auto &target_rows =
        persist_active ? staged_committed_rows : committed_rows;
    for (const auto &row : rows) {
      auto it = std::find_if(target_rows.begin(), target_rows.end(),
                             [&](const auto &entry) {
                               return entry.index_name == row.index_name &&
                                      entry.doc_id == row.doc_id;
                             });
      if (row.op == vector_index_metadata_store::change_op::kErase) {
        if (it != target_rows.end()) target_rows.erase(it);
        continue;
      }
      if (it == target_rows.end()) {
        target_rows.push_back(vector_index_metadata_store::committed_row{
            row.index_name, row.doc_id, row.vector});
      } else {
        it->vector = row.vector;
      }
    }
    return true;
  }

  bool erase_committed_index_batch(const std::string &index_name,
                                   size_t max_rows, size_t *erased_rows,
                                   bool *done) override {
    ++erase_committed_index_batch_calls;
    if (fail_erase_committed_index_batch || index_name.empty() ||
        max_rows == 0 || erased_rows == nullptr || done == nullptr) {
      return false;
    }
    *erased_rows = 0;
    for (auto it = committed_rows.begin();
         it != committed_rows.end() && *erased_rows < max_rows;) {
      if (it->index_name != index_name) {
        ++it;
        continue;
      }
      it = committed_rows.erase(it);
      ++*erased_rows;
    }
    *done = std::none_of(committed_rows.begin(), committed_rows.end(),
                         [&](const auto &row) {
                           return row.index_name == index_name;
                         });
    return true;
  }

  void block_committed_delta_apply() {
    std::lock_guard<std::mutex> guard(committed_delta_mutex);
    block_committed_delta = true;
  }

  bool wait_for_committed_delta_apply(uint64_t expected_calls) {
    std::unique_lock<std::mutex> guard(committed_delta_mutex);
    return committed_delta_condition.wait_for(
        guard, std::chrono::seconds(5),
        [&] { return apply_committed_delta_calls >= expected_calls; });
  }

  bool wait_for_rollback_persist(uint64_t expected_calls) {
    std::unique_lock<std::mutex> guard(committed_delta_mutex);
    return committed_delta_condition.wait_for(
        guard, std::chrono::seconds(5),
        [&] { return rollback_persist_calls >= expected_calls; });
  }

  void release_committed_delta_apply() {
    std::lock_guard<std::mutex> guard(committed_delta_mutex);
    block_committed_delta = false;
    committed_delta_condition.notify_all();
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
    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    (persist_active ? staged_change_log_rows : change_log_rows) = rows;
    return true;
  }

  bool append_change_log_delta(
      const std::vector<vector_index_metadata_store::change_log_row> &rows)
      override {
    ++append_change_log_delta_calls;
    if (fail_append_change_log_delta) return false;
    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    auto &target_rows =
        persist_active ? staged_change_log_rows : change_log_rows;
    target_rows.insert(target_rows.end(), rows.begin(), rows.end());
    return true;
  }

  bool erase_change_log_sequences(
      const std::vector<uint64_t> &sequences) override {
    ++erase_change_log_sequences_calls;
    if (fail_erase_change_log_sequences ||
        std::any_of(sequences.begin(), sequences.end(),
                    [](uint64_t sequence) { return sequence == 0; })) {
      return false;
    }
    std::lock_guard<std::mutex> guard(publication_intent_load_mutex);
    auto &target_rows =
        persist_active ? staged_change_log_rows : change_log_rows;
    target_rows.erase(
        std::remove_if(target_rows.begin(), target_rows.end(),
                       [&](const auto &row) {
                         return std::binary_search(
                             sequences.begin(), sequences.end(), row.sequence);
                       }),
        target_rows.end());
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

  bool load_segment_tasks(
      std::vector<vector_index_metadata_store::segment_task_row> *rows)
      override {
    if (fail_load_segment_tasks) return false;
    if (rows == nullptr) return false;
    *rows = segment_task_rows;
    return true;
  }

  bool save_segment_tasks(
      const std::vector<vector_index_metadata_store::segment_task_row> &rows)
      override {
    ++save_segment_tasks_calls;
    if (fail_save_segment_tasks) return false;
    segment_task_rows = rows;
    return true;
  }

  bool quarantine_segment_tasks() override {
    if (fail_quarantine_segment_tasks) return false;
    segment_task_rows.clear();
    return true;
  }

  bool fail_load_metadata{false};
  bool fail_load_publication_intents{false};
  bool fail_save_publication_intent{false};
  bool fail_delete_publication_intent{false};
  bool persist_active{false};
  std::vector<vector_index_truth_store::publication_intent>
      publication_intents;
  std::vector<vector_index_truth_store::publication_intent>
      staged_publication_intents;
  std::vector<std::pair<std::string, uint64_t>>
      staged_deleted_publication_intents;
  std::vector<vector_index_metadata_store::committed_row>
      staged_committed_rows;
  std::vector<vector_index_metadata_store::change_log_row>
      staged_change_log_rows;
  bool fail_save_metadata{false};
  bool fail_quarantine_metadata{false};
  bool fail_load_committed{false};
  bool fail_save_committed{false};
  bool fail_apply_committed_delta{false};
  bool fail_erase_committed_index_batch{false};
  uint64_t erase_committed_index_batch_calls{0};
  bool fail_quarantine_committed{false};
  bool fail_load_manifest{false};
  bool fail_save_manifest{false};
  bool fail_save_manifest_once{false};
  bool fail_quarantine_manifest{false};
  bool fail_load_change_log{false};
  bool fail_save_change_log{false};
  bool fail_append_change_log_delta{false};
  bool fail_erase_change_log_sequences{false};
  bool fail_quarantine_change_log{false};
  bool fail_load_prepared{false};
  bool fail_save_prepared{false};
  bool fail_quarantine_prepared{false};
  bool fail_load_segment_tasks{false};
  bool fail_save_segment_tasks{false};
  bool fail_quarantine_segment_tasks{false};
  bool fail_stage_quarantine{false};
  bool fail_update_quarantine_state{false};
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
  uint64_t erase_change_log_sequences_calls{0};
  uint64_t save_prepared_calls{0};
  uint64_t save_segment_tasks_calls{0};
  uint64_t save_manifest_calls{0};
  uint64_t quarantine_identity_sequence{0};
  std::vector<vector_index_truth_store::quarantine_record> quarantine_records;

  void ResetPersistCountersForTesting() {
    save_metadata_calls = 0;
    save_committed_calls = 0;
    apply_committed_delta_calls = 0;
    save_change_log_calls = 0;
    append_change_log_delta_calls = 0;
    erase_change_log_sequences_calls = 0;
    save_prepared_calls = 0;
    save_segment_tasks_calls = 0;
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

  void SetSegmentTaskRowsForTesting(
      const std::vector<vector_index_metadata_store::segment_task_row> &rows) {
    segment_task_rows = rows;
  }

  size_t CommittedRowsForIndexForTesting(
      const std::string &index_name) const {
    return static_cast<size_t>(std::count_if(
        committed_rows.begin(), committed_rows.end(), [&](const auto &row) {
          return row.index_name == index_name;
        }));
  }

 private:
  std::mutex committed_delta_mutex;
  std::condition_variable committed_delta_condition;
  bool block_committed_delta{false};
  std::mutex publication_intent_load_mutex;
  std::condition_variable publication_intent_load_condition;
  bool block_publication_intent_load{false};
  size_t publication_intent_load_calls{0};
  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  vector_index_metadata_store::manifest_row manifest_row;
  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  std::vector<vector_index_metadata_store::segment_task_row> segment_task_rows;
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
  base.diskann_build_blas_threads = 2;
  base.diskann_build_mode_value = vector_index::diskann_build_mode::kOffline;
  base.diskann_build_mode_specified = true;
  base.diskann_search_complexity = 80;
  base.diskann_search_beamwidth = 16;
  base.diskann_pq_code_budget_size = 1048576;
  base.diskann_disk_pq_dims = 12;
  base.diskann_cache_nodes = 128;
  base.diskann_accelerate_build = true;
  base.diskann_shuffle_build = true;
  base.diskann_use_bfs_cache = true;
  base.diskann_segmented_serving = true;

  EXPECT_TRUE(detail::index_config_matches_for_testing(base, base));

  auto expect_mismatch = [&](auto mutate) {
    vector_index::index_service::index_config changed = base;
    mutate(&changed);
    EXPECT_FALSE(detail::index_config_matches_for_testing(base, changed));
  };

  expect_mismatch([](auto *config) { config->dimension = 3; });
  expect_mismatch([](auto *config) {
    config->metric = vector_index::metric_type::kCosine;
  });
  expect_mismatch(
      [](auto *config) { config->mode = vector_index::backend_mode::kMemory; });
  expect_mismatch([](auto *config) {
    config->provider = vector_index::backend_provider::kFaiss;
  });
  expect_mismatch([](auto *config) { config->backend_variant = "different"; });
  expect_mismatch([](auto *config) {
    config->consistency_mode =
        vector_index::index_consistency_mode::kStandalone;
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
  expect_mismatch([](auto *config) { config->diskann_build_blas_threads = 3; });
  expect_mismatch([](auto *config) {
    config->diskann_build_mode_value =
        vector_index::diskann_build_mode::kSerial;
  });
  expect_mismatch(
      [](auto *config) { config->diskann_build_mode_specified = false; });
  expect_mismatch([](auto *config) { config->diskann_search_complexity = 81; });
  expect_mismatch([](auto *config) { config->diskann_search_beamwidth = 17; });
  expect_mismatch(
      [](auto *config) { config->diskann_pq_code_budget_size = 2097152; });
  expect_mismatch([](auto *config) { config->diskann_disk_pq_dims = 13; });
  expect_mismatch([](auto *config) { config->diskann_cache_nodes = 129; });
  expect_mismatch(
      [](auto *config) { config->diskann_accelerate_build = false; });
  expect_mismatch([](auto *config) { config->diskann_shuffle_build = false; });
  expect_mismatch([](auto *config) { config->diskann_use_bfs_cache = false; });
  expect_mismatch(
      [](auto *config) { config->diskann_segmented_serving = false; });
}

TEST(VectorIndexRegistryInternalTest,
     CommitTruthGenerationBindingRejectsInvalidRowsAndMapsEachIndex) {
  namespace detail = vector_index_registry::detail;
  using change_log_row = vector_index_metadata_store::change_log_row;
  using commit_build_plan = vector_index::index_service::commit_build_plan;

  EXPECT_FALSE(
      detail::bind_commit_truth_generations_for_testing(nullptr, nullptr));

  commit_build_plan empty_plan;
  std::vector<change_log_row> rows;
  EXPECT_TRUE(
      detail::bind_commit_truth_generations_for_testing(&rows, &empty_plan));

  commit_build_plan plan;
  auto &index_a = plan.indexes.emplace_back();
  index_a.index_name = "idx_generation_a";
  index_a.publication_before.index_identity = 101;
  index_a.publication_before.truth_generation = 11;
  auto &index_b = plan.indexes.emplace_back();
  index_b.index_name = "idx_generation_b";
  index_b.publication_before.index_identity = 202;
  index_b.publication_before.truth_generation = 22;

  EXPECT_TRUE(
      detail::bind_commit_truth_generations_for_testing(&rows, &plan));
  EXPECT_EQ(11U, plan.indexes[0].target_truth_generation);
  EXPECT_EQ(22U, plan.indexes[1].target_truth_generation);

  change_log_row invalid;
  invalid.sequence = 1;
  rows = {invalid};
  EXPECT_FALSE(
      detail::bind_commit_truth_generations_for_testing(&rows, &plan));
  rows[0].index_name = "idx_generation_a";
  invalid.sequence = 0;
  rows[0].sequence = 0;
  EXPECT_FALSE(
      detail::bind_commit_truth_generations_for_testing(&rows, &plan));

  const auto row = [](uint64_t sequence, const std::string &index_name) {
    change_log_row value;
    value.sequence = sequence;
    value.index_name = index_name;
    return value;
  };
  rows = {row(5, "idx_generation_a"), row(9, "idx_generation_a"),
          row(7, "idx_generation_b")};
  EXPECT_TRUE(
      detail::bind_commit_truth_generations_for_testing(&rows, &plan));
  EXPECT_EQ(12U, plan.indexes[0].target_truth_generation);
  EXPECT_EQ(23U, plan.indexes[1].target_truth_generation);
  EXPECT_EQ(101U, rows[0].index_identity);
  EXPECT_EQ(12U, rows[0].publication_id);
  EXPECT_EQ(12U, rows[0].truth_generation);
  EXPECT_EQ(202U, rows[2].index_identity);
  EXPECT_EQ(23U, rows[2].truth_generation);

  rows = {row(12, "idx_generation_a"),
          row(8, "idx_generation_unknown")};
  EXPECT_FALSE(
      detail::bind_commit_truth_generations_for_testing(&rows, &plan));

  rows = {row(12, "idx_generation_a"), row(8, "idx_generation_b")};
  rows[0].index_identity = 999;
  EXPECT_FALSE(
      detail::bind_commit_truth_generations_for_testing(&rows, &plan));

  rows = {row(12, "idx_generation_a"), row(8, "idx_generation_b")};
  rows[0].index_identity = 101;
  rows[0].publication_id = 99;
  rows[0].truth_generation = 99;
  EXPECT_FALSE(
      detail::bind_commit_truth_generations_for_testing(&rows, &plan));
}

TEST(VectorIndexRegistryInternalTest,
     RecoveryBindingComparisonRejectsEveryTableIdentityDrift) {
  namespace detail = vector_index_registry::detail;
  detail::index_binding expected{"db", "table", "vector", "doc_id"};
  EXPECT_TRUE(detail::index_bindings_equal_for_testing(expected, expected));

  const auto expect_mismatch = [&](const auto &mutate) {
    auto actual = expected;
    mutate(&actual);
    EXPECT_FALSE(detail::index_bindings_equal_for_testing(expected, actual));
  };
  expect_mismatch([](auto *binding) { binding->schema_name = "other_db"; });
  expect_mismatch([](auto *binding) { binding->table_name = "other_table"; });
  expect_mismatch([](auto *binding) { binding->column_name = "other_vector"; });
  expect_mismatch(
      [](auto *binding) { binding->doc_id_column_name = "other_doc_id"; });
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

  vector_gunit::NativeProviderGuard native_provider_guard_;
  in_memory_truth_store store_;
  std::string m_prepared_path;
};

TEST_F(VectorIndexRegistryTest,
       StatementConfigIntentPublishesOnceAndRecognizesRetry) {
  if (!hnswlib_tuning_supported()) GTEST_SKIP();
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_statement_config", 2, "euclidean", "memory", "hnswlib"));

  vector_statement_publication::operation_payload payload;
  payload.config = vector_statement_publication::config_change::kSearchEf;
  payload.unsigned_values = {77};
  std::string encoded;
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));

  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kUpdateConfig,
      "idx_statement_config", encoded, &intent));
  EXPECT_TRUE(
      vector_index_registry::validate_statement_publication_intent(intent));
  ASSERT_TRUE(
      vector_index_registry::publish_statement_publication_intent(intent));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info("idx_statement_config",
                                                    &info));
  EXPECT_EQ(77U, info.search_ef);
  EXPECT_GT(info.config_generation, intent.expected.config_generation);
  EXPECT_TRUE(
      vector_index_registry::publish_statement_publication_intent(intent));
}

TEST_F(VectorIndexRegistryTest,
       StatementConfigIntentRejectsUnsupportedProviderBeforeCommit) {
  const std::string index_name = "idx_statement_config_native";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  const auto validate_config =
      [&](vector_statement_publication::config_change config,
          std::vector<uint64_t> values) {
        vector_statement_publication::operation_payload payload;
        payload.config = config;
        payload.unsigned_values = std::move(values);
        std::string encoded;
        if (!vector_statement_publication::encode_operation_payload(payload,
                                                                    &encoded)) {
          return false;
        }
        vector_index_truth_store::publication_intent intent;
        if (!vector_index_registry::make_statement_publication_intent(
                vector_index_truth_store::publication_operation::kUpdateConfig,
                index_name, encoded, &intent)) {
          return false;
        }
        return vector_index_registry::validate_statement_publication_intent(
            intent);
      };

  using vector_statement_publication::config_change;
  EXPECT_FALSE(validate_config(config_change::kSearchEf, {16}));
  EXPECT_FALSE(validate_config(config_change::kHnswBuildParams, {16, 200}));
  EXPECT_FALSE(validate_config(config_change::kFaissIvfParams, {2, 1}));
  EXPECT_FALSE(validate_config(config_change::kFaissIvfPqParams, {2, 1, 2, 8}));
  EXPECT_FALSE(
      validate_config(config_change::kDiskannBuildParams, {64, 100, 1}));
  EXPECT_FALSE(validate_config(config_change::kDiskannSearchComplexity, {100}));
  EXPECT_FALSE(validate_config(config_change::kDiskannSearchBeamwidth, {16}));
  EXPECT_FALSE(
      validate_config(config_change::kDiskannPqCodeBudgetSize, {1048576}));
  EXPECT_FALSE(validate_config(config_change::kDiskannDiskPqDims, {1}));
  EXPECT_TRUE(validate_config(config_change::kDiskannDiskPqDims, {0}));
  EXPECT_FALSE(validate_config(config_change::kDiskannAccelerateBuild, {1}));
  EXPECT_TRUE(validate_config(config_change::kDiskannAccelerateBuild, {0}));
  EXPECT_FALSE(validate_config(config_change::kDiskannShuffleBuild, {1}));
  EXPECT_TRUE(validate_config(config_change::kDiskannShuffleBuild, {0}));
  EXPECT_FALSE(validate_config(config_change::kDiskannUseBfsCache, {1}));
  EXPECT_TRUE(validate_config(config_change::kDiskannUseBfsCache, {0}));
  EXPECT_FALSE(validate_config(
      config_change::kDiskannBuildMode,
      {static_cast<uint64_t>(vector_index::diskann_build_mode::kSerial)}));
  EXPECT_TRUE(validate_config(
      config_change::kDiskannBuildMode,
      {static_cast<uint64_t>(vector_index::diskann_build_mode::kAuto)}));
}

TEST_F(VectorIndexRegistryTest,
       StatementConfigIntentsApplyAndRecognizeProviderSpecificRetries) {
  using vector_statement_publication::config_change;
  const auto publish_config = [&](const std::string &index_name,
                                  config_change config,
                                  std::vector<uint64_t> values) {
    vector_statement_publication::operation_payload payload;
    payload.config = config;
    payload.unsigned_values = std::move(values);
    std::string encoded;
    ASSERT_TRUE(vector_statement_publication::encode_operation_payload(
        payload, &encoded));
    vector_index_truth_store::publication_intent intent;
    ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
        vector_index_truth_store::publication_operation::kUpdateConfig,
        index_name, encoded, &intent));
    ASSERT_TRUE(
        vector_index_registry::validate_statement_publication_intent(intent));
    ASSERT_TRUE(
        vector_index_registry::publish_statement_publication_intent(intent));
    EXPECT_TRUE(
        vector_index_registry::publish_statement_publication_intent(intent));
  };

  if (vector_index::backend_provider_supported(
          vector_index::backend_provider::kHnswlib)) {
    const std::string index_name = "idx_statement_config_hnsw";
    ASSERT_TRUE(vector_index_registry::create_index(index_name, 4, "euclidean",
                                                    "memory", "hnswlib"));
    publish_config(index_name, config_change::kSearchEf, {64});
    publish_config(index_name, config_change::kHnswBuildParams, {16, 200});
    vector_index_registry::index_info info;
    ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
    EXPECT_EQ(64U, info.search_ef);
    EXPECT_EQ(16U, info.hnsw_m);
    EXPECT_EQ(200U, info.hnsw_ef_construction);
    ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  }

  if (vector_index::backend_provider_supported(
          vector_index::backend_provider::kFaiss)) {
    const std::string index_name = "idx_statement_config_faiss";
    ASSERT_TRUE(vector_index_registry::create_index(index_name, 4, "euclidean",
                                                    "external", "faiss"));
    publish_config(index_name, config_change::kSearchEf, {48});
    publish_config(index_name, config_change::kHnswBuildParams, {12, 160});
    publish_config(index_name, config_change::kFaissIvfParams, {2, 1});
    publish_config(index_name, config_change::kFaissIvfPqParams, {2, 1, 2, 8});
    vector_index_registry::index_info info;
    ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
    EXPECT_EQ(2U, info.faiss_nlist);
    EXPECT_EQ(1U, info.faiss_nprobe);
    EXPECT_EQ(2U, info.faiss_pq_m);
    EXPECT_EQ(8U, info.faiss_pq_bits);
    ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  }

  if (vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    const std::string index_name = "idx_statement_config_diskann";
    ASSERT_TRUE(vector_index_registry::create_index(index_name, 4, "euclidean",
                                                    "external", "diskann"));
    publish_config(index_name, config_change::kDiskannBuildParams, {48, 96, 2});
    publish_config(index_name, config_change::kDiskannSearchComplexity, {80});
    publish_config(index_name, config_change::kDiskannSearchBeamwidth, {16});
    publish_config(index_name, config_change::kDiskannPqCodeBudgetSize, {1024});
    publish_config(index_name, config_change::kDiskannDiskPqDims, {2});
    publish_config(index_name, config_change::kDiskannAccelerateBuild, {1});
    publish_config(index_name, config_change::kDiskannShuffleBuild, {1});
    publish_config(index_name, config_change::kDiskannUseBfsCache, {1});
    publish_config(
        index_name, config_change::kDiskannBuildMode,
        {static_cast<uint64_t>(vector_index::diskann_build_mode::kOffline)});
    vector_index_registry::index_info info;
    ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
    EXPECT_EQ(48U, info.diskann_max_degree);
    EXPECT_EQ(96U, info.diskann_build_complexity);
    EXPECT_EQ(2U, info.diskann_build_threads);
    EXPECT_EQ(80U, info.diskann_search_complexity);
    EXPECT_EQ(16U, info.diskann_search_beamwidth);
    EXPECT_EQ(1024U, info.diskann_pq_code_budget_size);
    EXPECT_EQ(2U, info.diskann_disk_pq_dims);
    EXPECT_TRUE(info.diskann_accelerate_build);
    EXPECT_TRUE(info.diskann_shuffle_build);
    EXPECT_TRUE(info.diskann_use_bfs_cache);
    EXPECT_EQ(vector_index::diskann_build_mode::kOffline,
              info.diskann_build_mode_value);
    ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  }
}

TEST_F(VectorIndexRegistryTest,
       StatementConfigIntentRejectsMalformedPayloadMatrix) {
  const std::string index_name = "idx_statement_config_malformed";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 4, "euclidean",
                                                  "memory", "native"));
  using vector_statement_publication::config_change;
  const auto validate_config = [&](config_change config,
                                   std::vector<uint64_t> values) {
    vector_statement_publication::operation_payload payload;
    payload.config = config;
    payload.unsigned_values = std::move(values);
    std::string encoded;
    if (!vector_statement_publication::encode_operation_payload(payload,
                                                                &encoded)) {
      return false;
    }
    vector_index_truth_store::publication_intent intent;
    if (!vector_index_registry::make_statement_publication_intent(
            vector_index_truth_store::publication_operation::kUpdateConfig,
            index_name, encoded, &intent)) {
      return false;
    }
    return vector_index_registry::validate_statement_publication_intent(intent);
  };

  const uint64_t uint32_overflow =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_FALSE(validate_config(config_change::kSearchEf, {}));
  EXPECT_FALSE(validate_config(config_change::kSearchEf, {0}));
  EXPECT_FALSE(validate_config(config_change::kSearchEf, {uint32_overflow}));
  EXPECT_FALSE(validate_config(config_change::kHnswBuildParams, {16}));
  EXPECT_FALSE(validate_config(config_change::kHnswBuildParams, {0, 200}));
  EXPECT_FALSE(validate_config(config_change::kHnswBuildParams, {16, 0}));
  EXPECT_FALSE(
      validate_config(config_change::kHnswBuildParams, {16, uint32_overflow}));
  EXPECT_FALSE(validate_config(config_change::kFaissIvfParams, {2}));
  EXPECT_FALSE(
      validate_config(config_change::kFaissIvfParams, {2, uint32_overflow}));
  EXPECT_FALSE(validate_config(config_change::kFaissIvfPqParams, {2, 1, 2}));
  EXPECT_FALSE(validate_config(config_change::kFaissIvfPqParams, {0, 1, 2, 8}));
  EXPECT_FALSE(validate_config(config_change::kFaissIvfPqParams, {2, 0, 2, 8}));
  EXPECT_FALSE(validate_config(config_change::kFaissIvfPqParams, {2, 1, 0, 8}));
  EXPECT_FALSE(validate_config(config_change::kFaissIvfPqParams, {2, 1, 2, 0}));
  EXPECT_FALSE(
      validate_config(config_change::kFaissIvfPqParams,
                      {2, 1, 2, vector_index::k_max_faiss_pq_bits + 1}));
  EXPECT_FALSE(validate_config(config_change::kDiskannBuildParams, {48, 96}));
  EXPECT_FALSE(validate_config(config_change::kDiskannBuildParams, {0, 96, 1}));
  EXPECT_FALSE(validate_config(config_change::kDiskannBuildParams, {48, 0, 1}));
  EXPECT_FALSE(validate_config(config_change::kDiskannBuildParams,
                               {48, 96, uint32_overflow}));
  EXPECT_FALSE(validate_config(config_change::kDiskannSearchComplexity, {0}));
  EXPECT_FALSE(validate_config(config_change::kDiskannSearchComplexity,
                               {uint32_overflow}));
  EXPECT_FALSE(validate_config(config_change::kDiskannSearchBeamwidth, {0}));
  EXPECT_FALSE(validate_config(config_change::kDiskannSearchBeamwidth,
                               {uint32_overflow}));
  EXPECT_FALSE(validate_config(config_change::kDiskannPqCodeBudgetSize, {}));
  EXPECT_FALSE(
      validate_config(config_change::kDiskannDiskPqDims, {uint32_overflow}));
  EXPECT_FALSE(validate_config(config_change::kDiskannAccelerateBuild, {2}));
  EXPECT_FALSE(validate_config(config_change::kDiskannShuffleBuild, {2}));
  EXPECT_FALSE(validate_config(config_change::kDiskannUseBfsCache, {2}));
  EXPECT_FALSE(validate_config(
      config_change::kDiskannBuildMode,
      {static_cast<uint64_t>(vector_index::diskann_build_mode::kOffline) + 1}));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       StatementBulkLoadPayloadValidatesShapeAndReaderFailures) {
  const std::string index_name = "idx_statement_batch_payload";
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native", options));

  const auto make_payload = [](uint64_t doc_id, float first, float second,
                               bool replace_duplicates) {
    vector_statement_publication::operation_payload payload;
    payload.unsigned_values = {1, 1, 2, replace_duplicates ? 1U : 0U};
    payload.string_values.emplace_back(sizeof(uint64_t), '\0');
    int8store(reinterpret_cast<uchar *>(payload.string_values[0].data()),
              doc_id);
    payload.binary_value.resize(2 * sizeof(float));
    auto *bytes = reinterpret_cast<uchar *>(payload.binary_value.data());
    float4store(bytes, first);
    float4store(bytes + sizeof(float), second);
    return payload;
  };
  const auto make_intent =
      [&](const vector_statement_publication::operation_payload &payload,
          vector_index_truth_store::publication_intent *intent) {
        std::string encoded;
        if (!vector_statement_publication::encode_operation_payload(payload,
                                                                    &encoded)) {
          return false;
        }
        return vector_index_registry::make_statement_publication_intent(
            vector_index_truth_store::publication_operation::kBulkLoad,
            index_name, encoded, intent);
      };
  const auto expect_invalid = [&](const auto &mutate) {
    auto payload = make_payload(11, 1.0F, 2.0F, true);
    mutate(&payload);
    vector_index_truth_store::publication_intent intent;
    ASSERT_TRUE(make_intent(payload, &intent));
    EXPECT_FALSE(
        vector_index_registry::validate_statement_publication_intent(intent));
  };

  expect_invalid([](auto *payload) { payload->unsigned_values.pop_back(); });
  expect_invalid([](auto *payload) { payload->unsigned_values[0] = 2; });
  expect_invalid([](auto *payload) { payload->unsigned_values[1] = 0; });
  expect_invalid([](auto *payload) { payload->unsigned_values[2] = 0; });
  expect_invalid([](auto *payload) { payload->string_values.clear(); });
  expect_invalid([](auto *payload) {
    payload->unsigned_values[1] =
        std::numeric_limits<uint64_t>::max() / sizeof(uint64_t) + 1;
  });
  expect_invalid([](auto *payload) {
    payload->unsigned_values[1] =
        std::numeric_limits<uint64_t>::max() / sizeof(uint64_t);
    payload->unsigned_values[2] = 3;
  });
  expect_invalid([](auto *payload) { payload->string_values[0].clear(); });
  expect_invalid([](auto *payload) { payload->binary_value.clear(); });

  vector_index_truth_store::publication_intent intent;
  auto payload = make_payload(11, 1.0F, 2.0F, true);
  ASSERT_TRUE(make_intent(payload, &intent));
  ASSERT_TRUE(
      vector_index_registry::validate_statement_publication_intent(intent));
  ASSERT_TRUE(
      vector_index_registry::publish_statement_publication_intent(intent));

  payload = make_payload(11, 2.0F, 3.0F, false);
  ASSERT_TRUE(make_intent(payload, &intent));
  ASSERT_TRUE(
      vector_index_registry::validate_statement_publication_intent(intent));
  std::string failure_stage;
  EXPECT_FALSE(vector_index_registry::publish_statement_publication_intent(
      intent, &failure_stage));
  EXPECT_FALSE(failure_stage.empty());

  payload =
      make_payload(12, std::numeric_limits<float>::quiet_NaN(), 3.0F, true);
  ASSERT_TRUE(make_intent(payload, &intent));
  ASSERT_TRUE(
      vector_index_registry::validate_statement_publication_intent(intent));
  failure_stage.clear();
  EXPECT_FALSE(vector_index_registry::publish_statement_publication_intent(
      intent, &failure_stage));
  EXPECT_FALSE(failure_stage.empty());

  payload = make_payload(13, 3.0F, 4.0F, true);
  ASSERT_TRUE(make_intent(payload, &intent));
  ASSERT_TRUE(
      vector_index_registry::publish_statement_publication_intent(intent));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(0U, info.entry_count);
  EXPECT_EQ(2U, info.committed_entry_count);
  EXPECT_EQ("bulk_loading", info.lifecycle_state);

  EXPECT_FALSE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kBulkLoad, "",
      std::string(), &intent));
  EXPECT_FALSE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kBulkLoad, index_name,
      std::string(), nullptr));
  EXPECT_FALSE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kTransactionalDml,
      index_name, std::string(), &intent));

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       StatementIntentValidationRejectsModePayloadAndPendingStateDrift) {
  const std::string transactional_index = "idx_statement_validate_txn";
  const std::string standalone_index = "idx_statement_validate_standalone";
  ASSERT_TRUE(vector_index_registry::create_index(
      transactional_index, 2, "euclidean", "memory", "native"));
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(vector_index_registry::create_index(
      standalone_index, 2, "euclidean", "memory", "native", options));

  const auto encode =
      [](const vector_statement_publication::operation_payload &payload) {
        std::string encoded;
        EXPECT_TRUE(vector_statement_publication::encode_operation_payload(
            payload, &encoded));
        return encoded;
      };
  const auto validate = [&](auto operation, const std::string &index_name,
                            const std::string &payload) {
    vector_index_truth_store::publication_intent intent;
    if (!vector_index_registry::make_statement_publication_intent(
            operation, index_name, payload, &intent)) {
      return false;
    }
    return vector_index_registry::validate_statement_publication_intent(intent);
  };

  vector_statement_publication::operation_payload upsert;
  upsert.unsigned_values = {7, 2};
  upsert.binary_value.resize(2 * sizeof(float));
  auto *upsert_bytes = reinterpret_cast<uchar *>(upsert.binary_value.data());
  float4store(upsert_bytes, 1.0F);
  float4store(upsert_bytes + sizeof(float), 2.0F);
  using vector_index_truth_store::publication_operation;
  EXPECT_FALSE(validate(publication_operation::kStandaloneUpsert,
                        transactional_index, encode(upsert)));
  EXPECT_TRUE(validate(publication_operation::kStandaloneUpsert,
                       standalone_index, encode(upsert)));
  EXPECT_FALSE(validate(publication_operation::kStandaloneUpsert,
                        standalone_index, "malformed"));

  auto invalid_upsert = upsert;
  invalid_upsert.unsigned_values.pop_back();
  EXPECT_FALSE(validate(publication_operation::kStandaloneUpsert,
                        standalone_index, encode(invalid_upsert)));
  invalid_upsert = upsert;
  invalid_upsert.unsigned_values[1] = 0;
  EXPECT_FALSE(validate(publication_operation::kStandaloneUpsert,
                        standalone_index, encode(invalid_upsert)));
  invalid_upsert = upsert;
  invalid_upsert.unsigned_values[1] = 3;
  EXPECT_FALSE(validate(publication_operation::kStandaloneUpsert,
                        standalone_index, encode(invalid_upsert)));
  invalid_upsert = upsert;
  invalid_upsert.binary_value.clear();
  EXPECT_FALSE(validate(publication_operation::kStandaloneUpsert,
                        standalone_index, encode(invalid_upsert)));
  invalid_upsert = upsert;
  float4store(reinterpret_cast<uchar *>(invalid_upsert.binary_value.data()),
              std::numeric_limits<float>::infinity());
  EXPECT_FALSE(validate(publication_operation::kStandaloneUpsert,
                        standalone_index, encode(invalid_upsert)));

  vector_statement_publication::operation_payload erase;
  erase.unsigned_values = {7};
  EXPECT_TRUE(validate(publication_operation::kStandaloneErase,
                       standalone_index, encode(erase)));
  EXPECT_FALSE(validate(publication_operation::kStandaloneErase,
                        transactional_index, encode(erase)));
  erase.unsigned_values.push_back(8);
  EXPECT_FALSE(validate(publication_operation::kStandaloneErase,
                        standalone_index, encode(erase)));

  vector_statement_publication::operation_payload batch;
  batch.unsigned_values = {1, 1, 2, 1};
  batch.string_values.emplace_back(sizeof(uint64_t), '\0');
  int8store(reinterpret_cast<uchar *>(batch.string_values[0].data()), 9);
  batch.binary_value = upsert.binary_value;
  EXPECT_TRUE(validate(publication_operation::kBulkLoad, standalone_index,
                       encode(batch)));
  EXPECT_FALSE(validate(publication_operation::kBulkLoad, transactional_index,
                        encode(batch)));
  auto wrong_dimension_batch = batch;
  wrong_dimension_batch.unsigned_values[2] = 1;
  wrong_dimension_batch.binary_value.resize(sizeof(float));
  EXPECT_FALSE(validate(publication_operation::kBulkLoad, standalone_index,
                        encode(wrong_dimension_batch)));

  constexpr uint64_t thd_id = 9901;
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      thd_id, 1, standalone_index, 10, {3.0F, 4.0F}));
  EXPECT_FALSE(validate(publication_operation::kStandaloneUpsert,
                        standalone_index, encode(upsert)));
  erase.unsigned_values = {7};
  EXPECT_FALSE(validate(publication_operation::kStandaloneErase,
                        standalone_index, encode(erase)));
  EXPECT_FALSE(validate(publication_operation::kBulkLoad, standalone_index,
                        encode(batch)));

  for (const auto operation : {
           publication_operation::kRebuildIndex,
           publication_operation::kRecoverIndex,
           publication_operation::kBeginBulkLoad,
           publication_operation::kBulkBuildIndex,
       }) {
    EXPECT_FALSE(validate(operation, standalone_index, std::string()));
  }
  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(thd_id));

  for (const auto operation : {
           publication_operation::kRebuildIndex,
           publication_operation::kRecoverIndex,
           publication_operation::kBeginBulkLoad,
           publication_operation::kBulkBuildIndex,
       }) {
    EXPECT_TRUE(validate(operation, standalone_index, std::string()));
    EXPECT_FALSE(validate(operation, standalone_index, "unexpected"));
  }
  EXPECT_TRUE(validate(publication_operation::kDropIndex, standalone_index,
                       std::string()));
  EXPECT_FALSE(validate(publication_operation::kDropIndex, standalone_index,
                        "unexpected"));

  vector_index_truth_store::publication_intent missing_expected;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      publication_operation::kRebuildIndex, standalone_index, std::string(),
      &missing_expected));
  missing_expected.expected.exists = false;
  EXPECT_FALSE(vector_index_registry::validate_statement_publication_intent(
      missing_expected));

  ASSERT_TRUE(vector_index_registry::drop_index(transactional_index));
  ASSERT_TRUE(vector_index_registry::drop_index(standalone_index));
}

TEST_F(VectorIndexRegistryTest,
       StatementConfigRetryRejectsEveryMismatchedAppliedValue) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kHnswlib)) {
    GTEST_SKIP() << "HNSWLIB provider is not compiled in";
  }
  const std::string index_name = "idx_statement_config_retry_mismatch";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 4, "euclidean",
                                                  "memory", "hnswlib"));

  using vector_statement_publication::config_change;
  struct config_case {
    config_change config;
    std::vector<uint64_t> values;
  };
  const std::vector<config_case> cases = {
      {config_change::kSearchEf, {999}},
      {config_change::kHnswBuildParams, {17, 201}},
      {config_change::kFaissIvfParams, {3, 2}},
      {config_change::kFaissIvfPqParams, {3, 2, 1, 8}},
      {config_change::kDiskannBuildParams, {48, 96, 2}},
      {config_change::kDiskannSearchComplexity, {80}},
      {config_change::kDiskannSearchBeamwidth, {16}},
      {config_change::kDiskannPqCodeBudgetSize, {1024}},
      {config_change::kDiskannDiskPqDims, {2}},
      {config_change::kDiskannAccelerateBuild, {1}},
      {config_change::kDiskannShuffleBuild, {1}},
      {config_change::kDiskannUseBfsCache, {1}},
      {config_change::kDiskannBuildMode,
       {static_cast<uint64_t>(vector_index::diskann_build_mode::kOffline)}},
  };

  uint32_t search_ef = 100;
  const auto expect_retry_mismatch = [&](const config_case &test_case) {
    vector_statement_publication::operation_payload payload;
    payload.config = test_case.config;
    payload.unsigned_values = test_case.values;
    std::string encoded;
    ASSERT_TRUE(vector_statement_publication::encode_operation_payload(
        payload, &encoded));
    vector_index_truth_store::publication_intent intent;
    ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
        vector_index_truth_store::publication_operation::kUpdateConfig,
        index_name, encoded, &intent));
    ASSERT_TRUE(vector_index_registry::set_search_ef(index_name, search_ef++));
    std::string failure_stage;
    EXPECT_FALSE(vector_index_registry::publish_statement_publication_intent(
        intent, &failure_stage));
    EXPECT_EQ("statement_publication_cas_mismatch", failure_stage);
  };

  for (const auto &test_case : cases) {
    expect_retry_mismatch(test_case);
    auto malformed_case = test_case;
    malformed_case.values.clear();
    expect_retry_mismatch(malformed_case);
  }
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       StatementCreateIntentValidatesDefinitionBeforeCommit) {
  std::string encoded;
  vector_index_registry::create_index_options options;
  ASSERT_TRUE(vector_index_registry::make_create_statement_payload(
      2, "euclidean", "memory", "native", "test", options, &encoded));
  vector_statement_publication::operation_payload payload;
  ASSERT_TRUE(vector_statement_publication::decode_operation_payload(encoded,
                                                                     &payload));
  payload.string_values[0] = "manhattan";
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));
  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kCreateIndex,
      "idx_statement_create", encoded, &intent));
  EXPECT_FALSE(
      vector_index_registry::validate_statement_publication_intent(intent));

  payload.string_values[0] = "euclidean";
  payload.string_values[2] = "missing";
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kCreateIndex,
      "idx_statement_create", encoded, &intent));
  EXPECT_FALSE(
      vector_index_registry::validate_statement_publication_intent(intent));

  payload.string_values[2] = "native";
  payload.unsigned_values[1] =
      static_cast<uint64_t>(vector_index::k_max_build_threads) + 1;
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kCreateIndex,
      "idx_statement_create", encoded, &intent));
  EXPECT_FALSE(
      vector_index_registry::validate_statement_publication_intent(intent));

  payload.unsigned_values[1] = 0;
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kCreateIndex,
      "idx_statement_create", encoded, &intent));
  EXPECT_TRUE(
      vector_index_registry::validate_statement_publication_intent(intent));
}

TEST_F(VectorIndexRegistryTest,
       StatementCreateIntentRejectsMalformedPayloadMatrix) {
  std::string encoded;
  vector_index_registry::create_index_options options;
  ASSERT_TRUE(vector_index_registry::make_create_statement_payload(
      4, "euclidean", "memory", "native", "test", options, &encoded));
  vector_statement_publication::operation_payload base_payload;
  ASSERT_TRUE(vector_statement_publication::decode_operation_payload(
      encoded, &base_payload));

  const auto expect_invalid = [&](const auto &mutate) {
    auto payload = base_payload;
    mutate(&payload);
    std::string invalid_encoded;
    ASSERT_TRUE(vector_statement_publication::encode_operation_payload(
        payload, &invalid_encoded));
    vector_index_truth_store::publication_intent intent;
    ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
        vector_index_truth_store::publication_operation::kCreateIndex,
        "idx_statement_create_invalid_matrix", invalid_encoded, &intent));
    EXPECT_FALSE(
        vector_index_registry::validate_statement_publication_intent(intent));
  };

  const uint64_t uint32_overflow =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  expect_invalid([](auto *payload) { payload->unsigned_values.pop_back(); });
  expect_invalid([](auto *payload) { payload->string_values.pop_back(); });
  expect_invalid([](auto *payload) { payload->unsigned_values[0] = 0; });
  expect_invalid([](auto *payload) {
    payload->unsigned_values[0] = vector_index::k_max_vector_dimension + 1;
  });
  expect_invalid([](auto *payload) {
    payload->unsigned_values[1] = vector_index::k_max_build_threads + 1;
  });
  expect_invalid(
      [&](auto *payload) { payload->unsigned_values[2] = uint32_overflow; });
  expect_invalid(
      [&](auto *payload) { payload->unsigned_values[3] = uint32_overflow; });
  expect_invalid(
      [&](auto *payload) { payload->unsigned_values[5] = uint32_overflow; });
  expect_invalid([](auto *payload) { payload->unsigned_values[6] = 2; });
  expect_invalid([](auto *payload) { payload->unsigned_values[7] = 2; });
  expect_invalid([](auto *payload) { payload->unsigned_values[8] = 2; });
  expect_invalid(
      [&](auto *payload) { payload->unsigned_values[9] = uint32_overflow; });
  expect_invalid(
      [&](auto *payload) { payload->unsigned_values[10] = uint32_overflow; });
  expect_invalid([](auto *payload) {
    payload->unsigned_values[11] =
        static_cast<uint64_t>(
            vector_index::index_consistency_mode::kStandalone) +
        1;
  });
  expect_invalid([](auto *payload) { payload->string_values[0] = "missing"; });
  expect_invalid([](auto *payload) { payload->string_values[1].clear(); });
  expect_invalid([](auto *payload) { payload->string_values[1] = "missing"; });
  expect_invalid([](auto *payload) { payload->string_values[2].clear(); });
  expect_invalid([](auto *payload) { payload->string_values[2] = "missing"; });
  expect_invalid([](auto *payload) { payload->string_values[3].clear(); });
  expect_invalid([](auto *payload) {
    payload->string_values[1] = "external";
    payload->string_values[2] = "hnswlib";
  });

  for (const size_t field : {2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U}) {
    expect_invalid(
        [field](auto *payload) { payload->unsigned_values[field] = 1; });
  }

  if (vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    vector_index_registry::create_index_options diskann_options;
    ASSERT_TRUE(vector_index_registry::make_create_statement_payload(
        4, "euclidean", "external", "diskann", "test", diskann_options,
        &encoded));
    ASSERT_TRUE(vector_statement_publication::decode_operation_payload(
        encoded, &base_payload));
    for (const size_t required_field : {2U, 3U, 9U, 10U}) {
      expect_invalid([required_field](auto *payload) {
        payload->unsigned_values[required_field] = 0;
      });
    }
  }
}

TEST_F(VectorIndexRegistryTest,
       StatementCreateIntentKeepsResolvedDefaultsDuringRecovery) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "DiskANN provider is not compiled in";
  }

  UlongGuard build_threads(&opt_vector_diskann_build_threads, 7);
  UlongGuard max_degree(&opt_vector_diskann_max_degree, 48);
  UlongGuard build_complexity(&opt_vector_diskann_build_complexity, 96);
  UlonglongGuard pq_budget(&opt_vector_diskann_pq_code_budget_size, 4096);
  UlongGuard disk_pq_dims(&opt_vector_diskann_disk_pq_dims, 12);
  BoolGuard accelerate(&opt_vector_diskann_accelerate_build, true);
  BoolGuard shuffle(&opt_vector_diskann_shuffle_build, true);
  BoolGuard bfs(&opt_vector_diskann_use_bfs_cache, true);
  UlongGuard search_complexity(&opt_vector_diskann_search_complexity, 144);
  UlongGuard search_beamwidth(&opt_vector_diskann_search_beamwidth, 28);
  UlongGuard consistency(
      &opt_vector_index_consistency_mode,
      static_cast<ulong>(vector_index::index_consistency_mode::kStandalone));

  std::string encoded;
  vector_index_registry::create_index_options options;
  ASSERT_TRUE(vector_index_registry::make_create_statement_payload(
      2, "euclidean", "external", "diskann", "test", options, &encoded));
  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kCreateIndex,
      "idx_statement_create_resolved", encoded, &intent));

  opt_vector_diskann_build_threads = 23;
  opt_vector_diskann_max_degree = 64;
  opt_vector_diskann_build_complexity = 128;
  opt_vector_diskann_pq_code_budget_size = 8192;
  opt_vector_diskann_disk_pq_dims = 24;
  opt_vector_diskann_accelerate_build = false;
  opt_vector_diskann_shuffle_build = false;
  opt_vector_diskann_use_bfs_cache = false;
  opt_vector_diskann_search_complexity = 288;
  opt_vector_diskann_search_beamwidth = 56;
  opt_vector_index_consistency_mode = static_cast<ulong>(
      vector_index::index_consistency_mode::kTransactional);

  ASSERT_TRUE(
      vector_index_registry::publish_statement_publication_intent(intent));
  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_statement_create_resolved", &info));
  EXPECT_EQ("standalone", info.consistency_mode);
  EXPECT_EQ(7U, info.diskann_build_threads);
  EXPECT_EQ(48U, info.diskann_max_degree);
  EXPECT_EQ(96U, info.diskann_build_complexity);
  EXPECT_EQ(4096U, info.diskann_pq_code_budget_size);
  EXPECT_EQ(12U, info.diskann_disk_pq_dims);
  EXPECT_TRUE(info.diskann_accelerate_build);
  EXPECT_TRUE(info.diskann_shuffle_build);
  EXPECT_TRUE(info.diskann_use_bfs_cache);
  EXPECT_EQ(144U, info.diskann_search_complexity);
  EXPECT_EQ(28U, info.diskann_search_beamwidth);
}

TEST_F(VectorIndexRegistryTest, StatementIntentRejectsStaleGeneration) {
  if (!hnswlib_tuning_supported()) GTEST_SKIP();
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_statement_stale", 2, "euclidean", "memory", "hnswlib"));

  vector_statement_publication::operation_payload payload;
  payload.config = vector_statement_publication::config_change::kSearchEf;
  payload.unsigned_values = {77};
  std::string encoded;
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));
  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kUpdateConfig,
      "idx_statement_stale", encoded, &intent));

  ASSERT_TRUE(vector_index_registry::set_search_ef("idx_statement_stale", 55));
  EXPECT_FALSE(
      vector_index_registry::validate_statement_publication_intent(intent));
  std::string failure_stage;
  EXPECT_FALSE(vector_index_registry::publish_statement_publication_intent(
      intent, &failure_stage));
  EXPECT_EQ("statement_publication_cas_mismatch", failure_stage);
}

TEST_F(VectorIndexRegistryTest,
       StatementLifecycleIntentsRejectPendingIndexChangesDuringPreflight) {
  const std::string index_name = "idx_statement_pending_changes";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_NE(0U, txn_id);
  ASSERT_TRUE(vector_index_registry::stage_upsert(
      txn_id, index_name, 1, {1.0F, 2.0F}));

  using vector_index_truth_store::publication_operation;
  for (const publication_operation operation : {
           publication_operation::kRebuildIndex,
           publication_operation::kRecoverIndex,
           publication_operation::kBeginBulkLoad,
           publication_operation::kBulkBuildIndex}) {
    vector_index_truth_store::publication_intent intent;
    ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
        operation, index_name, std::string(), &intent));
    EXPECT_FALSE(
        vector_index_registry::validate_statement_publication_intent(intent));
  }

  EXPECT_TRUE(vector_index_registry::rollback_txn(txn_id));
  EXPECT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       StandaloneStatementUpsertAdvancesSourceAndIsIdempotent) {
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode =
      vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_statement_upsert", 2, "euclidean", "memory", "native",
      options));

  vector_statement_publication::operation_payload payload;
  payload.unsigned_values = {9, 2};
  payload.binary_value.resize(2 * sizeof(float));
  auto *bytes = reinterpret_cast<uchar *>(payload.binary_value.data());
  float4store(bytes, 1.0F);
  float4store(bytes + sizeof(float), 2.0F);
  std::string encoded;
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));
  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kStandaloneUpsert,
      "idx_statement_upsert", encoded, &intent));
  ASSERT_TRUE(
      vector_index_registry::publish_statement_publication_intent(intent));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info("idx_statement_upsert",
                                                    &info));
  EXPECT_EQ("bulk_loading", info.lifecycle_state);
  EXPECT_GT(info.standalone_ingest_memory_bytes, 0U);
  EXPECT_TRUE(
      vector_index_registry::publish_statement_publication_intent(intent));
  ASSERT_TRUE(vector_index_registry::get_index_info("idx_statement_upsert",
                                                    &info));
  EXPECT_EQ("bulk_loading", info.lifecycle_state);
}

TEST_F(VectorIndexRegistryTest,
       StatementLifecycleOperationsPublishAndRecognizeRetries) {
  const std::string index_name = "idx_statement_lifecycle_retry";
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;
  std::string encoded;
  ASSERT_TRUE(vector_index_registry::make_create_statement_payload(
      2, "euclidean", "memory", "native", "test", options, &encoded));
  vector_index_truth_store::publication_intent create_intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kCreateIndex, index_name,
      encoded, &create_intent));
  ASSERT_TRUE(vector_index_registry::publish_statement_publication_intent(
      create_intent));
  EXPECT_TRUE(vector_index_registry::publish_statement_publication_intent(
      create_intent));

  vector_statement_publication::operation_payload upsert_payload;
  upsert_payload.unsigned_values = {9, 2};
  upsert_payload.binary_value.resize(2 * sizeof(float));
  auto *bytes = reinterpret_cast<uchar *>(upsert_payload.binary_value.data());
  float4store(bytes, 1.0F);
  float4store(bytes + sizeof(float), 2.0F);
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(
      upsert_payload, &encoded));
  vector_index_truth_store::publication_intent upsert_intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kStandaloneUpsert,
      index_name, encoded, &upsert_intent));
  ASSERT_TRUE(vector_index_registry::publish_statement_publication_intent(
      upsert_intent));
  EXPECT_TRUE(vector_index_registry::publish_statement_publication_intent(
      upsert_intent));

  vector_statement_publication::operation_payload erase_payload;
  erase_payload.unsigned_values = {9};
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(
      erase_payload, &encoded));
  vector_index_truth_store::publication_intent erase_intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kStandaloneErase,
      index_name, encoded, &erase_intent));
  ASSERT_TRUE(vector_index_registry::publish_statement_publication_intent(
      erase_intent));
  EXPECT_TRUE(vector_index_registry::publish_statement_publication_intent(
      erase_intent));

  const auto publish_lifecycle = [&](auto operation) {
    vector_index_truth_store::publication_intent intent;
    ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
        operation, index_name, std::string(), &intent));
    ASSERT_TRUE(
        vector_index_registry::validate_statement_publication_intent(intent));
    ASSERT_TRUE(
        vector_index_registry::publish_statement_publication_intent(intent));
    EXPECT_TRUE(
        vector_index_registry::publish_statement_publication_intent(intent));
  };
  using vector_index_truth_store::publication_operation;
  publish_lifecycle(publication_operation::kBeginBulkLoad);
  publish_lifecycle(publication_operation::kBulkBuildIndex);
  publish_lifecycle(publication_operation::kRebuildIndex);
  publish_lifecycle(publication_operation::kRecoverIndex);
  publish_lifecycle(publication_operation::kDropIndex);
}

TEST_F(VectorIndexRegistryTest,
       CommittedStatementIntentPublishesAndAcknowledgesDuringRecovery) {
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode =
      vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_statement_recover_commit", 2, "euclidean", "memory", "native",
      options));

  vector_statement_publication::operation_payload payload;
  payload.unsigned_values = {17, 2};
  payload.binary_value.resize(2 * sizeof(float));
  auto *bytes = reinterpret_cast<uchar *>(payload.binary_value.data());
  float4store(bytes, 1.0F);
  float4store(bytes + sizeof(float), 7.0F);
  std::string encoded;
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));

  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kStandaloneUpsert,
      "idx_statement_recover_commit", encoded, &intent));
  store_.publication_intents.push_back(intent);
  vector_index_registry::schedule_publication_intent_recovery();

  vector_index_truth_store::publication_intent next_intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kBeginBulkLoad,
      "idx_statement_recover_commit", std::string(), &next_intent));
  EXPECT_TRUE(store_.publication_intents.empty());

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_statement_recover_commit", &info));
  EXPECT_EQ("bulk_loading", info.lifecycle_state);
  EXPECT_GT(next_intent.expected.source_generation,
            intent.expected.source_generation);
}

TEST_F(VectorIndexRegistryTest,
       ManagedLoadRecoveryUsesReceiptAfterAcknowledgementFailure) {
  const std::string root =
      std::string(DATA_DIR) + "/vector_registry_managed_load";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  external_snapshot_root_guard root_guard(root + "/managed");

  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(vector_index_registry::create_index("idx_managed_load_recovery",
                                                  2, "euclidean", "memory",
                                                  "native", options));

  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  const std::string source = root + "/source.csv";
  {
    std::ofstream file(source, std::ios::trunc);
    ASSERT_TRUE(file.good());
    file << "doc_id,vector\n101,\"[1,0]\"\n202,\"[0,1]\"\n";
    ASSERT_TRUE(file.good());
  }

  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kBulkLoad,
      "idx_managed_load_recovery", std::string(), &intent));
  const vector_index::load_staging_identity identity{intent.index_name,
                                                     intent.publication_id};
  vector_index::load_staging_artifact artifact;
  std::string error;
  ASSERT_TRUE(vector_index::stage_load_artifact(identity, source, "", "CSV",
                                                &artifact, &error))
      << error;
  vector_statement_publication::operation_payload payload;
  ASSERT_TRUE(vector_index::make_load_staging_payload(
      identity, artifact, false, true, 2, "CSV", &payload));
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(
      payload, &intent.payload));
  store_.publication_intents.push_back(intent);
  ASSERT_TRUE(std::filesystem::remove(source));

  store_.fail_delete_publication_intent = true;
  vector_index_registry::schedule_publication_intent_recovery();
  vector_index_registry::index_info info;
  EXPECT_FALSE(vector_index_registry::get_index_info(
      "idx_managed_load_recovery", &info));
  ASSERT_EQ(1U, store_.publication_intents.size());

  store_.fail_delete_publication_intent = false;
  std::string retry_failure_stage;
  EXPECT_TRUE(vector_index_registry::publish_statement_publication_intent(
      intent, &retry_failure_stage))
      << retry_failure_stage;
  vector_index_registry::schedule_publication_intent_recovery();
  ASSERT_TRUE(vector_index_registry::get_index_info("idx_managed_load_recovery",
                                                    &info));
  EXPECT_TRUE(store_.publication_intents.empty());
  EXPECT_EQ(2U, info.entry_count);

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(vector_index_registry::search("idx_managed_load_recovery",
                                            {1.0F, 0.0F}, 2, &results));
  ASSERT_EQ(2U, results.size());
  EXPECT_EQ(101U, results[0].doc_id);

  vector_index::load_staging_paths paths;
  EXPECT_FALSE(vector_index::verify_load_artifact(identity, artifact, "CSV",
                                                  &paths, &error));
  std::filesystem::remove_all(root, ec);
}

TEST_F(VectorIndexRegistryTest,
       ReadSurfacesRecoverCommittedStatementIntentsBeforeReturningState) {
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode =
      vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_statement_read_recovery", 2, "euclidean", "memory", "native",
      options));

  vector_index_truth_store::publication_intent begin_intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kBeginBulkLoad,
      "idx_statement_read_recovery", std::string(), &begin_intent));
  store_.publication_intents.push_back(begin_intent);
  vector_index_registry::schedule_publication_intent_recovery();

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_statement_read_recovery", &info));
  EXPECT_EQ("bulk_loading", info.lifecycle_state);
  EXPECT_TRUE(store_.publication_intents.empty());

  vector_index_truth_store::publication_intent drop_intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kDropIndex,
      "idx_statement_read_recovery", std::string(), &drop_intent));
  store_.publication_intents.push_back(drop_intent);
  vector_index_registry::schedule_publication_intent_recovery();

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  EXPECT_EQ(index_names.end(),
            std::find(index_names.begin(), index_names.end(),
                      "idx_statement_read_recovery"));
  EXPECT_TRUE(store_.publication_intents.empty());
}

TEST_F(VectorIndexRegistryTest,
       PublicationReadGuardRecoversIntentBeforeEnteringReadScope) {
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode =
      vector_index::index_consistency_mode::kStandalone;
  const std::string index_name = "idx_publication_read_guard";
  ASSERT_TRUE(vector_index_registry::create_index(
      index_name, 2, "euclidean", "memory", "native", options));

  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kBeginBulkLoad,
      index_name, std::string(), &intent));
  store_.publication_intents.push_back(intent);
  vector_index_registry::schedule_publication_intent_recovery();

  vector_index_registry::publication_read_guard publication_guard;
  ASSERT_TRUE(publication_guard.lock_index(index_name));
  EXPECT_TRUE(publication_guard.owns_lock());
  EXPECT_TRUE(store_.publication_intents.empty());

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("bulk_loading", info.lifecycle_state);
}

TEST_F(VectorIndexRegistryTest,
       PublicationReadGuardRechecksRecoveryAfterWaitingForWriter) {
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode =
      vector_index::index_consistency_mode::kStandalone;
  const std::string index_name = "idx_publication_read_retry";
  ASSERT_TRUE(vector_index_registry::create_index(
      index_name, 2, "euclidean", "memory", "native", options));

  vector_statement_publication::publication_guard writer;
  ASSERT_TRUE(writer.lock_indexes({index_name}));

  const size_t load_count = store_.publication_intent_load_count();
  store_.block_publication_intent_loads();
  vector_index_registry::schedule_publication_intent_recovery();
  auto reader = std::async(std::launch::async, [&] {
    vector_index_registry::publication_read_guard publication_guard;
    if (!publication_guard.lock_index(index_name)) return false;
    vector_index_registry::index_info info;
    return vector_index_registry::get_index_info(index_name, &info) &&
           info.lifecycle_state == "bulk_loading";
  });
  if (!store_.wait_for_publication_intent_loads(load_count + 1)) {
    store_.release_publication_intent_loads();
    writer = vector_statement_publication::publication_guard();
    FAIL() << "publication read guard did not enter recovery preflight";
  }

  store_.release_publication_intent_loads();
  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kBeginBulkLoad,
      index_name, std::string(), &intent));
  store_.AddPublicationIntentForTesting(intent);
  vector_index_registry::schedule_publication_intent_recovery();
  writer = vector_statement_publication::publication_guard();

  ASSERT_EQ(std::future_status::ready,
            reader.wait_for(std::chrono::seconds(5)));
  EXPECT_TRUE(reader.get());
  EXPECT_TRUE(store_.publication_intents.empty());
}

TEST_F(VectorIndexRegistryTest,
       PublicationIntentRecoverySerializesConcurrentReaders) {
  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  const size_t load_count = store_.publication_intent_load_count();

  store_.block_publication_intent_loads();
  vector_index_registry::schedule_publication_intent_recovery();
  auto first = std::async(std::launch::async, [&] {
    std::vector<std::string> names;
    return vector_index_registry::list_indexes(&names);
  });
  if (!store_.wait_for_publication_intent_loads(load_count + 1)) {
    store_.release_publication_intent_loads();
    const auto first_status = first.wait_for(std::chrono::seconds(5));
    if (first_status == std::future_status::ready) {
      EXPECT_TRUE(first.get());
    }
    FAIL() << "first recovery did not enter publication-intent loading";
  }
  std::promise<void> second_started;
  auto second_started_future = second_started.get_future();
  auto second = std::async(std::launch::async, [&] {
    std::vector<std::string> names;
    second_started.set_value();
    return vector_index_registry::list_indexes(&names);
  });
  second_started_future.wait();
  const auto second_status = second.wait_for(std::chrono::milliseconds(100));

  store_.release_publication_intent_loads();
  const bool first_ok = first.get();
  const bool second_ok = second.get();

  EXPECT_EQ(std::future_status::timeout, second_status);
  EXPECT_TRUE(first_ok);
  EXPECT_TRUE(second_ok);
  EXPECT_EQ(load_count + 1, store_.publication_intent_load_count());
}

TEST_F(VectorIndexRegistryTest,
       PublicationIntentRecoveryPreservesRequestsRaisedDuringReplay) {
  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  const size_t load_count = store_.publication_intent_load_count();

  store_.block_publication_intent_loads();
  vector_index_registry::schedule_publication_intent_recovery();
  auto first = std::async(std::launch::async, [&] {
    std::vector<std::string> names;
    return vector_index_registry::list_indexes(&names);
  });
  if (!store_.wait_for_publication_intent_loads(load_count + 1)) {
    store_.release_publication_intent_loads();
    const auto first_status = first.wait_for(std::chrono::seconds(5));
    if (first_status == std::future_status::ready) {
      EXPECT_TRUE(first.get());
    }
    FAIL() << "first recovery did not enter publication-intent loading";
  }
  vector_index_registry::schedule_publication_intent_recovery();
  std::promise<void> second_started;
  auto second_started_future = second_started.get_future();
  auto second = std::async(std::launch::async, [&] {
    std::vector<std::string> names;
    second_started.set_value();
    return vector_index_registry::list_indexes(&names);
  });
  second_started_future.wait();
  const auto second_status = second.wait_for(std::chrono::milliseconds(100));

  store_.release_publication_intent_loads();
  const bool first_ok = first.get();
  const bool second_ok = second.get();

  EXPECT_EQ(std::future_status::timeout, second_status);
  EXPECT_TRUE(first_ok);
  EXPECT_TRUE(second_ok);
  EXPECT_EQ(load_count + 2, store_.publication_intent_load_count());
}

TEST_F(VectorIndexRegistryTest,
       ExplicitTransactionBeginsAfterCommittedIntentRecovery) {
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode =
      vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_explicit_txn_recovery", 2, "euclidean", "memory", "native",
      options));

  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kBeginBulkLoad,
      "idx_explicit_txn_recovery", std::string(), &intent));
  store_.publication_intents.push_back(intent);
  vector_index_registry::schedule_publication_intent_recovery();

  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_NE(0U, txn_id);
  EXPECT_TRUE(store_.publication_intents.empty());
  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_explicit_txn_recovery", &info));
  EXPECT_EQ("bulk_loading", info.lifecycle_state);
  EXPECT_TRUE(vector_index_registry::rollback_txn(txn_id));
}

TEST_F(VectorIndexRegistryTest,
       AppliedStatementIntentRecoveryOnlyAcknowledgesPublishedState) {
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode =
      vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_statement_recover_ack", 2, "euclidean", "memory", "native",
      options));

  vector_statement_publication::operation_payload payload;
  payload.unsigned_values = {21, 2};
  payload.binary_value.resize(2 * sizeof(float));
  auto *bytes = reinterpret_cast<uchar *>(payload.binary_value.data());
  float4store(bytes, 2.0F);
  float4store(bytes + sizeof(float), 1.0F);
  std::string encoded;
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));

  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kStandaloneUpsert,
      "idx_statement_recover_ack", encoded, &intent));
  ASSERT_TRUE(
      vector_index_registry::publish_statement_publication_intent(intent));
  vector_index_truth_store::publication_intent before_recovery;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kBeginBulkLoad,
      "idx_statement_recover_ack", std::string(), &before_recovery));

  store_.publication_intents.push_back(intent);
  vector_index_registry::schedule_publication_intent_recovery();
  vector_index_truth_store::publication_intent next_intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kBeginBulkLoad,
      "idx_statement_recover_ack", std::string(), &next_intent));
  EXPECT_TRUE(store_.publication_intents.empty());

  EXPECT_EQ(before_recovery.expected.source_generation,
            next_intent.expected.source_generation);
}

TEST_F(VectorIndexRegistryTest,
       StatementIntentRecoveryRejectsGenerationDriftWithoutAcknowledging) {
  if (!hnswlib_tuning_supported()) GTEST_SKIP();
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_statement_recover_stale", 2, "euclidean", "memory", "hnswlib"));

  vector_statement_publication::operation_payload payload;
  payload.config = vector_statement_publication::config_change::kSearchEf;
  payload.unsigned_values = {77};
  std::string encoded;
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));
  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kUpdateConfig,
      "idx_statement_recover_stale", encoded, &intent));
  ASSERT_TRUE(
      vector_index_registry::set_search_ef("idx_statement_recover_stale", 55));

  store_.publication_intents.push_back(intent);
  vector_index_registry::schedule_publication_intent_recovery();
  vector_index_truth_store::publication_intent next_intent;
  EXPECT_FALSE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kBeginBulkLoad,
      "idx_statement_recover_stale", std::string(), &next_intent));
  ASSERT_EQ(1U, store_.publication_intents.size());
  EXPECT_EQ(intent.publication_id,
            store_.publication_intents.front().publication_id);
}

TEST_F(VectorIndexRegistryTest, StatementIntentRejectsSameNameIdentityAba) {
  if (!hnswlib_tuning_supported()) GTEST_SKIP();
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_statement_aba", 2, "euclidean", "memory", "hnswlib"));

  vector_statement_publication::operation_payload payload;
  payload.config = vector_statement_publication::config_change::kSearchEf;
  payload.unsigned_values = {77};
  std::string encoded;
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));
  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kUpdateConfig,
      "idx_statement_aba", encoded, &intent));

  ASSERT_TRUE(vector_index_registry::drop_index("idx_statement_aba"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_statement_aba", 2, "euclidean", "memory", "hnswlib"));
  std::string failure_stage;
  EXPECT_FALSE(vector_index_registry::publish_statement_publication_intent(
      intent, &failure_stage));
  EXPECT_EQ("statement_publication_cas_mismatch", failure_stage);
}

namespace {

void expect_metadata_only_persist(const in_memory_truth_store &store) {
  EXPECT_EQ(1U, store.save_metadata_calls);
  EXPECT_EQ(1U, store.save_manifest_calls);
  EXPECT_EQ(0U, store.save_committed_calls);
  EXPECT_EQ(0U, store.save_change_log_calls);
  EXPECT_EQ(0U, store.save_prepared_calls);
  EXPECT_EQ(0U, store.save_segment_tasks_calls);
}

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
       AfterCommitPublicationRejectsDurableDriftAndPublishesExactGeneration) {
  namespace detail = vector_index_registry::detail;
  using change_log_row = vector_index_metadata_store::change_log_row;

  store_.fail_load_metadata = true;
  std::string failure_stage;
  EXPECT_FALSE(detail::publish_pending_runtime_for_testing(
      1, {}, false, &failure_stage));
  EXPECT_EQ("metadata_load_before_snapshot", failure_stage);
  store_.fail_load_metadata = false;
  vector_index_registry::reset_for_testing();

  const std::string index_name = "idx_after_commit_generation";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_NE(0U, txn_id);
  ASSERT_TRUE(vector_index_registry::stage_upsert(
      txn_id, index_name, 7, {1.0F, 2.0F}));

  EXPECT_FALSE(detail::publish_pending_runtime_for_testing(txn_id, {}, false,
                                                           nullptr));
  failure_stage.clear();
  EXPECT_FALSE(detail::publish_pending_runtime_for_testing(
      txn_id, {}, false, &failure_stage));
  EXPECT_EQ("pending_change_count_mismatch", failure_stage);

  change_log_row durable_row;
  durable_row.sequence = 1;
  durable_row.op = vector_index_metadata_store::change_op::kUpsert;
  durable_row.doc_id = 7;
  durable_row.vector = {1.0F, 2.0F};

  failure_stage.clear();
  EXPECT_FALSE(detail::publish_pending_runtime_for_testing(
      txn_id, {durable_row}, false, &failure_stage));
  EXPECT_EQ("bind_commit_truth_generations", failure_stage);

  durable_row.index_name = index_name;
  durable_row.sequence = 0;
  failure_stage.clear();
  EXPECT_FALSE(detail::publish_pending_runtime_for_testing(
      txn_id, {durable_row}, false, &failure_stage));
  EXPECT_EQ("bind_commit_truth_generations", failure_stage);

  durable_row.sequence = 1;
  failure_stage.clear();
  ASSERT_TRUE(detail::publish_pending_runtime_for_testing(
      txn_id, {durable_row}, false, &failure_stage));
  EXPECT_TRUE(failure_stage.empty());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result.front().doc_id);

  EXPECT_TRUE(detail::publish_pending_runtime_for_testing(txn_id, {}, false));
  EXPECT_FALSE(detail::publish_pending_runtime_for_testing(
      txn_id, {durable_row}, false));
}

TEST_F(VectorIndexRegistryTest,
       TransactionalIntentAcceptsCanonicalRecoveredRuntimeOnly) {
  vector_index_truth_store::publication_intent intent;
  intent.operation =
      vector_index_truth_store::publication_operation::kTransactionalDml;
  intent.target.exists = true;
  intent.target.index_identity = 7;
  intent.target.truth_generation = 3;
  intent.target.config_generation = 2;
  intent.target.artifact_generation = 1;
  intent.target.runtime_generation = 1;
  intent.target.lifecycle_version = 5;

  EXPECT_TRUE(
      vector_index_registry::detail::publication_intent_target_matches_for_testing(
          intent, intent.target));

  vector_index_truth_store::publication_token recovered = intent.target;
  recovered.runtime_generation = recovered.truth_generation;
  recovered.artifact_generation = recovered.truth_generation;
  EXPECT_TRUE(
      vector_index_registry::detail::publication_intent_target_matches_for_testing(
          intent, recovered));

  recovered.artifact_generation = 0;
  EXPECT_TRUE(
      vector_index_registry::detail::publication_intent_target_matches_for_testing(
          intent, recovered));

  recovered.runtime_generation = 2;
  EXPECT_FALSE(
      vector_index_registry::detail::publication_intent_target_matches_for_testing(
          intent, recovered));
  recovered.runtime_generation = recovered.truth_generation;
  recovered.artifact_generation = 2;
  EXPECT_FALSE(
      vector_index_registry::detail::publication_intent_target_matches_for_testing(
          intent, recovered));
  recovered.artifact_generation = recovered.truth_generation;
  ++recovered.lifecycle_version;
  EXPECT_FALSE(
      vector_index_registry::detail::publication_intent_target_matches_for_testing(
          intent, recovered));
  recovered.lifecycle_version = intent.target.lifecycle_version;
  ++recovered.config_generation;
  EXPECT_FALSE(
      vector_index_registry::detail::publication_intent_target_matches_for_testing(
          intent, recovered));

  intent.operation =
      vector_index_truth_store::publication_operation::kRebuildIndex;
  recovered = intent.target;
  recovered.runtime_generation = recovered.truth_generation;
  recovered.artifact_generation = recovered.truth_generation;
  EXPECT_FALSE(
      vector_index_registry::detail::publication_intent_target_matches_for_testing(
          intent, recovered));
}

TEST_F(VectorIndexRegistryTest,
       TransactionEntryPointsRejectInvalidIdentityAndUnavailableMetadata) {
  EXPECT_EQ(0U, vector_index_registry::pending_txn_changes(0));
  EXPECT_FALSE(vector_index_registry::savepoint_txn(0, "sp"));
  EXPECT_FALSE(vector_index_registry::rollback_to_savepoint_txn(0, "sp"));
  EXPECT_FALSE(vector_index_registry::release_savepoint_txn(0, "sp"));
  EXPECT_FALSE(vector_index_registry::stage_erase(0, "idx", 1));

  store_.fail_load_metadata = true;
  vector_index_registry::reset_for_testing();
  EXPECT_EQ(0U, vector_index_registry::begin_txn());
  EXPECT_FALSE(vector_index_registry::commit_txn(1));
  EXPECT_FALSE(vector_index_registry::rollback_txn(1));
  EXPECT_EQ(0U, vector_index_registry::pending_txn_changes(1));
  EXPECT_FALSE(vector_index_registry::savepoint_txn(1, "sp"));
  EXPECT_FALSE(vector_index_registry::rollback_to_savepoint_txn(1, "sp"));
  EXPECT_FALSE(vector_index_registry::release_savepoint_txn(1, "sp"));
  EXPECT_FALSE(vector_index_registry::stage_upsert(1, "idx", 1,
                                                   {1.0F, 2.0F}));
  EXPECT_FALSE(vector_index_registry::stage_erase(1, "idx", 1));
  EXPECT_FALSE(vector_index_registry::stage_upsert_for_thd_txn(
      11, 22, "idx", 1, {1.0F, 2.0F}));
  EXPECT_FALSE(
      vector_index_registry::stage_erase_for_thd_txn(11, 22, "idx", 1));

  store_.fail_load_metadata = false;
  vector_index_registry::reset_for_testing();
}

TEST_F(VectorIndexRegistryTest,
       ExplicitTransactionRejectsIdsThatWereNeverAllocated) {
  const std::string index_name = "idx_explicit_txn_identity";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_NE(0U, txn_id);
  ASSERT_NE(std::numeric_limits<uint64_t>::max(), txn_id);

  EXPECT_FALSE(vector_index_registry::stage_upsert(
      txn_id + 1, index_name, 1, {1.0F, 2.0F}));
  EXPECT_FALSE(vector_index_registry::stage_upsert(
      std::numeric_limits<uint64_t>::max(), index_name, 2, {2.0F, 3.0F}));
  EXPECT_EQ(0U, vector_index_registry::pending_txn_changes(txn_id + 1));

  EXPECT_TRUE(vector_index_registry::rollback_txn(txn_id));
  EXPECT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       AfterCommitPublicationHandlesDetachedAndAttachedThdLifecycles) {
  namespace detail = vector_index_registry::detail;

  EXPECT_TRUE(vector_index_registry::publish_thd_txn(700, false, nullptr));

  std::string failure_stage = "stale";
  EXPECT_TRUE(
      vector_index_registry::publish_thd_txn(700, true, &failure_stage));
  EXPECT_TRUE(failure_stage.empty());

  const std::string index_name = "idx_after_commit_thd";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      701, 31, index_name, 9, {9.0F, 1.0F}));
  {
    std::lock_guard<std::shared_mutex> guard(detail::g_registry_mutex);
    auto context = detail::g_thd_txn_contexts.find(701);
    ASSERT_NE(context, detail::g_thd_txn_contexts.end());
    context->second.attached_dml = true;
    vector_index_metadata_store::change_log_row row;
    row.sequence = detail::g_next_change_log_sequence;
    row.txn_id = context->second.txn_id;
    row.op = vector_index_metadata_store::change_op::kUpsert;
    row.index_name = index_name;
    row.doc_id = 9;
    row.vector = {9.0F, 1.0F};
    context->second.durable_change_log_rows.push_back(std::move(row));
  }

  failure_stage = "stale";
  ASSERT_TRUE(
      vector_index_registry::publish_thd_txn(701, false, &failure_stage));
  EXPECT_TRUE(failure_stage.empty());
  EXPECT_EQ(detail::g_thd_txn_contexts.end(),
            detail::g_thd_txn_contexts.find(701));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {9.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(9U, result.front().doc_id);

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      702, 32, index_name, 10, {10.0F, 1.0F}));
  uint64_t txn_702 = 0;
  {
    std::lock_guard<std::shared_mutex> guard(detail::g_registry_mutex);
    auto context = detail::g_thd_txn_contexts.find(702);
    ASSERT_NE(context, detail::g_thd_txn_contexts.end());
    txn_702 = context->second.txn_id;
    ASSERT_TRUE(detail::g_index_service.release_savepoint(
        context->second.txn_id, context->second.stmt_savepoint_name));
  }
  const size_t load_count_before_savepoint_failure =
      store_.publication_intent_load_count();
  failure_stage.clear();
  EXPECT_FALSE(
      vector_index_registry::publish_thd_txn(702, false, &failure_stage));
  EXPECT_EQ("release_statement_savepoint", failure_stage);
  {
    std::lock_guard<std::shared_mutex> guard(detail::g_registry_mutex);
    EXPECT_EQ(detail::g_thd_txn_contexts.end(),
              detail::g_thd_txn_contexts.find(702));
  }
  EXPECT_EQ(0U, vector_index_registry::pending_txn_changes(txn_702));
  std::vector<std::string> index_names;
  EXPECT_TRUE(vector_index_registry::list_indexes(&index_names));
  EXPECT_EQ(load_count_before_savepoint_failure + 1,
            store_.publication_intent_load_count());

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      703, 33, index_name, 11, {11.0F, 1.0F}));
  uint64_t txn_703 = 0;
  {
    std::lock_guard<std::shared_mutex> guard(detail::g_registry_mutex);
    auto context = detail::g_thd_txn_contexts.find(703);
    ASSERT_NE(context, detail::g_thd_txn_contexts.end());
    txn_703 = context->second.txn_id;
    context->second.attached_dml = true;
  }
  const size_t load_count_before_runtime_failure =
      store_.publication_intent_load_count();
  failure_stage.clear();
  EXPECT_FALSE(
      vector_index_registry::publish_thd_txn(703, false, &failure_stage));
  EXPECT_EQ("pending_change_count_mismatch", failure_stage);
  {
    std::lock_guard<std::shared_mutex> guard(detail::g_registry_mutex);
    EXPECT_EQ(detail::g_thd_txn_contexts.end(),
              detail::g_thd_txn_contexts.find(703));
  }
  EXPECT_EQ(0U, vector_index_registry::pending_txn_changes(txn_703));
  EXPECT_TRUE(vector_index_registry::list_indexes(&index_names));
  EXPECT_EQ(load_count_before_runtime_failure + 1,
            store_.publication_intent_load_count());
}

TEST_F(VectorIndexRegistryTest,
       TransactionalDiskAnnAcceptsFirstCommittedUpsert) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "DiskANN provider is not compiled in";
  }

  const std::string root =
      std::string(testing::TempDir()) + "/registry_diskann_first_upsert_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  external_snapshot_root_guard root_guard(root);

  const std::string index_name = "idx_registry_diskann_first_upsert";
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
                             "+d,vector_backend_fail_diskann_native_rebuild");
    ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                    "external", "diskann"));
  }
  ASSERT_TRUE(
      vector_index_registry::set_diskann_build_params(index_name, 48, 96, 0));
  ASSERT_TRUE(
      vector_index_registry::set_diskann_search_complexity(index_name, 80));
  ASSERT_TRUE(vector_index_registry::set_diskann_pq_code_budget_size(
      index_name, 1024 * 1024));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_NE(0U, txn_id);
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::commit_txn(txn_id));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  std::filesystem::remove_all(root, ec);
}

TEST_F(VectorIndexRegistryTest,
       TransactionalDiskAnnRestartFallsBackFromFailedNativeRebuild) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "DiskANN provider is not compiled in";
  }

  const std::string root =
      std::string(testing::TempDir()) + "/registry_diskann_restart_fallback_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  external_snapshot_root_guard root_guard(root);

  const std::string index_name = "idx_registry_diskann_restart_fallback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "external", "diskann"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 2.0F}));

  vector_index_registry::reset_for_testing();

  std::vector<vector_index::search_result> result;
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
                             "+d,vector_backend_fail_diskann_native_rebuild");
    ASSERT_TRUE(
        vector_index_registry::search(index_name, {1.0F, 2.0F}, 1, &result));
  }
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(info.truth_generation, info.artifact_generation);
  EXPECT_EQ(info.truth_generation, info.runtime_generation);
  EXPECT_EQ("native_rebuild_failed_sidecar_fallback",
            info.build_diagnostics.fallback_reason);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  std::filesystem::remove_all(root, ec);
}

TEST_F(VectorIndexRegistryTest,
       FaissSidecarRevisionsKeepLogicalArtifactGeneration) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kFaiss)) {
    GTEST_SKIP() << "Faiss provider is not compiled in";
  }

  const std::string root =
      std::string(testing::TempDir()) + "/registry_faiss_generation_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  external_snapshot_root_guard root_guard(root);

  const std::string index_name = "idx_registry_faiss_generation";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "external", "faiss"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 2.0F}));

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &before));
  ASSERT_GT(before.external_manifest_generation, 0U);
  EXPECT_EQ(before.truth_generation, before.artifact_generation);

  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));
  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));

  vector_index_registry::index_info after;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &after));
  EXPECT_EQ(before.truth_generation, after.truth_generation);
  EXPECT_EQ(after.truth_generation, after.artifact_generation);
  EXPECT_GT(after.external_manifest_generation,
            before.external_manifest_generation);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  std::filesystem::remove_all(root, ec);
}

TEST_F(VectorIndexRegistryTest,
       BackfillPublishesOneDurableGenerationTransaction) {
  const std::string index_name = "idx_registry_backfill_generation";
  vector_index_registry::create_index_options options;
  options.initial_lifecycle_state = "creating";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native", options));

  vector_index_registry::index_backfill_token token;
  ASSERT_TRUE(vector_index_registry::begin_backfill(index_name, &token));
  EXPECT_NE(0U, token.index_identity);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("backfilling", info.lifecycle_state);

  vector_index::index_service::committed_entries entries;
  for (uint64_t doc_id = 1; doc_id <= 300; ++doc_id) {
    entries.emplace(doc_id,
                    vector_index::vector_data{static_cast<float>(doc_id),
                                              static_cast<float>(doc_id)});
  }

  const uint64_t begin_before = store_.begin_persist_calls;
  const uint64_t commit_before = store_.commit_persist_calls;
  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(
      vector_index_registry::publish_backfill(index_name, entries, token));
  EXPECT_EQ(begin_before + 1, store_.begin_persist_calls);
  EXPECT_EQ(commit_before + 1, store_.commit_persist_calls);
  EXPECT_EQ(2U, store_.apply_committed_delta_calls);
  EXPECT_EQ(1U, store_.append_change_log_delta_calls);
  EXPECT_EQ(1U, store_.save_metadata_calls);
  EXPECT_EQ(1U, store_.save_manifest_calls);

  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("ready", info.lifecycle_state);
  EXPECT_EQ(300U, info.entry_count);
  EXPECT_EQ(300U, info.committed_entry_count);
  EXPECT_EQ(info.truth_generation, info.runtime_generation);
  EXPECT_EQ(1U, info.truth_generation);

  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  ASSERT_TRUE(store_.load_change_log(&change_log_rows));
  ASSERT_EQ(300U, change_log_rows.size());
  EXPECT_EQ(1U, change_log_rows.front().sequence);
  EXPECT_EQ(300U, change_log_rows.back().sequence);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {300.0F, 300.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(300U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, DiskAnnEmptyBackfillPublishesZeroGeneration) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "DiskANN provider is not compiled in";
  }

  const std::string root =
      std::string(testing::TempDir()) + "/registry_diskann_empty_backfill_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  external_snapshot_root_guard root_guard(root);

  const std::string index_name = "idx_registry_diskann_empty_backfill";
  vector_index_registry::create_index_options options;
  options.initial_lifecycle_state = "creating";
  ASSERT_TRUE(vector_index_registry::create_index(
      index_name, 2, "euclidean", "external", "diskann", options));

  vector_index_registry::index_backfill_token token;
  ASSERT_TRUE(vector_index_registry::begin_backfill(index_name, &token));
  store_.ResetPersistCountersForTesting();
  const bool published =
      vector_index_registry::publish_backfill(index_name, {}, token);
  ASSERT_TRUE(published) << "metadata=" << store_.save_metadata_calls
                         << " manifest=" << store_.save_manifest_calls
                         << " changelog="
                         << store_.append_change_log_delta_calls;
  EXPECT_EQ(1U, store_.save_metadata_calls);
  EXPECT_EQ(1U, store_.save_manifest_calls);
  EXPECT_EQ(0U, store_.append_change_log_delta_calls);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("ready", info.lifecycle_state);
  EXPECT_EQ(0U, info.truth_generation);
  EXPECT_EQ(0U, info.artifact_generation);
  EXPECT_EQ(0U, info.runtime_generation);
  EXPECT_EQ(0U, info.entry_count);
  EXPECT_EQ(0U, info.committed_entry_count);
  EXPECT_TRUE(info.external_manifest_present);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  std::filesystem::remove_all(root, ec);
}

TEST_F(VectorIndexRegistryTest,
       BackfillFailureKeepsUnpublishedGenerationAndAllowsRetry) {
  const std::string index_name = "idx_registry_backfill_failure";
  vector_index_registry::create_index_options options;
  options.initial_lifecycle_state = "creating";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native", options));

  vector_index_registry::index_backfill_token token;
  store_.fail_begin_persist = true;
  EXPECT_FALSE(vector_index_registry::begin_backfill(index_name, &token));
  store_.fail_begin_persist = false;

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("creating", info.lifecycle_state);
  ASSERT_TRUE(vector_index_registry::begin_backfill(index_name, &token));

  const vector_index::index_service::committed_entries entries{
      {7, {7.0F, 7.0F}}};
  store_.fail_save_manifest = true;
  EXPECT_FALSE(
      vector_index_registry::publish_backfill(index_name, entries, token));
  store_.fail_save_manifest = false;

  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("backfilling", info.lifecycle_state);
  EXPECT_EQ(0U, info.entry_count);
  EXPECT_EQ(token.truth_generation, info.truth_generation);
  EXPECT_EQ(token.runtime_generation, info.runtime_generation);

  ASSERT_TRUE(
      vector_index_registry::publish_backfill(index_name, entries, token));
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("ready", info.lifecycle_state);
  EXPECT_EQ(1U, info.entry_count);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, BackfillRestoreFailureFailStopsRegistry) {
  const std::string index_name = "idx_registry_backfill_restore_failure";
  vector_index_registry::create_index_options options;
  options.initial_lifecycle_state = "creating";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native", options));

  store_.fail_save_manifest = true;
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
                             "+d,vector_registry_fail_restore_runtime_state");
    vector_index_registry::index_backfill_token token;
    EXPECT_FALSE(vector_index_registry::begin_backfill(index_name, &token));
  }
  store_.fail_save_manifest = false;

  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("registry_state_restore_failed",
            vector_index_registry::registry_failure_reason());
}

TEST_F(VectorIndexRegistryTest, BackfillRejectsSameNameIdentityAba) {
  const std::string index_name = "idx_registry_backfill_aba";
  vector_index_registry::create_index_options options;
  options.initial_lifecycle_state = "creating";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native", options));

  vector_index_registry::index_backfill_token stale_token;
  ASSERT_TRUE(vector_index_registry::begin_backfill(index_name, &stale_token));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native", options));

  vector_index_registry::index_backfill_token current_token;
  ASSERT_TRUE(
      vector_index_registry::begin_backfill(index_name, &current_token));
  EXPECT_NE(stale_token.index_identity, current_token.index_identity);
  EXPECT_FALSE(vector_index_registry::publish_backfill(
      index_name, {{1, {1.0F, 1.0F}}}, stale_token));
  ASSERT_TRUE(vector_index_registry::publish_backfill(
      index_name, {{2, {2.0F, 2.0F}}}, current_token));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(current_token.index_identity, info.index_identity);
  EXPECT_EQ(1U, info.entry_count);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       BackfillRejectsEveryStaleTokenFieldAndInvalidOwnerState) {
  vector_index_registry::create_index_options options;
  options.initial_lifecycle_state = "creating";

  EXPECT_FALSE(vector_index_registry::begin_backfill("missing", nullptr));

  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_backfill_ready", 2, "euclidean", "memory", "native"));
  vector_index_registry::index_backfill_token token;
  EXPECT_FALSE(
      vector_index_registry::begin_backfill("idx_backfill_ready", &token));
  ASSERT_TRUE(vector_index_registry::drop_index("idx_backfill_ready"));

  vector_index_registry::create_index_options standalone_options = options;
  standalone_options.consistency_mode_specified = true;
  standalone_options.consistency_mode =
      vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_backfill_standalone", 2, "euclidean", "memory", "native",
      standalone_options));
  EXPECT_FALSE(
      vector_index_registry::begin_backfill("idx_backfill_standalone", &token));
  ASSERT_TRUE(vector_index_registry::drop_index("idx_backfill_standalone"));

  const std::string index_name = "idx_backfill_token_matrix";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native", options));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));
  EXPECT_FALSE(vector_index_registry::begin_backfill(index_name, &token));
  vector_index_registry::rollback_txn(txn_id);

  ASSERT_TRUE(vector_index_registry::begin_backfill(index_name, &token));
  const vector_index::index_service::committed_entries entries{
      {1, {1.0F, 1.0F}}};

  const auto expect_stale_token = [&](auto mutate) {
    vector_index_registry::index_backfill_token stale = token;
    mutate(&stale);
    EXPECT_FALSE(
        vector_index_registry::publish_backfill(index_name, entries, stale));
  };

  expect_stale_token([](auto *stale) { ++stale->index_identity; });
  expect_stale_token([](auto *stale) { ++stale->truth_generation; });
  expect_stale_token([](auto *stale) { ++stale->config_generation; });
  expect_stale_token([](auto *stale) { ++stale->artifact_generation; });
  expect_stale_token([](auto *stale) { ++stale->runtime_generation; });
  expect_stale_token([](auto *stale) { ++stale->lifecycle_version; });

  ASSERT_TRUE(
      vector_index_registry::publish_backfill(index_name, entries, token));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       BackfillPersistenceFailureMatrixRestoresUnpublishedGeneration) {
  using failure_flag = bool in_memory_truth_store::*;
  const std::vector<std::pair<std::string, failure_flag>> delta_failures{
      {"committed_delta", &in_memory_truth_store::fail_apply_committed_delta},
      {"change_log_delta",
       &in_memory_truth_store::fail_append_change_log_delta},
      {"metadata", &in_memory_truth_store::fail_save_metadata},
      {"manifest", &in_memory_truth_store::fail_save_manifest},
      {"commit", &in_memory_truth_store::fail_commit_persist}};

  vector_index_registry::create_index_options options;
  options.initial_lifecycle_state = "creating";
  const vector_index::index_service::committed_entries entries{
      {7, {7.0F, 7.0F}}};

  const auto verify_failure = [&](const std::string &suffix,
                                  failure_flag flag) {
    const std::string index_name = "idx_backfill_delta_" + suffix;
    ASSERT_TRUE(vector_index_registry::create_index(
        index_name, 2, "euclidean", "memory", "native", options));
    vector_index_registry::index_backfill_token token;
    ASSERT_TRUE(vector_index_registry::begin_backfill(index_name, &token));

    store_.*flag = true;
    EXPECT_FALSE(
        vector_index_registry::publish_backfill(index_name, entries, token));
    store_.*flag = false;

    vector_index_registry::index_info info;
    ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
    EXPECT_EQ("backfilling", info.lifecycle_state);
    EXPECT_EQ(0U, info.entry_count);
    EXPECT_EQ(token.truth_generation, info.truth_generation);
    EXPECT_EQ(token.runtime_generation, info.runtime_generation);
    ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  };

  for (const auto &[suffix, flag] : delta_failures) {
    verify_failure(suffix, flag);
  }

  store_.delta_persist_supported = false;
  const std::vector<std::pair<std::string, failure_flag>> snapshot_failures{
      {"committed", &in_memory_truth_store::fail_save_committed},
      {"change_log", &in_memory_truth_store::fail_save_change_log},
      {"metadata", &in_memory_truth_store::fail_save_metadata},
      {"manifest", &in_memory_truth_store::fail_save_manifest},
      {"commit", &in_memory_truth_store::fail_commit_persist}};
  for (const auto &[suffix, flag] : snapshot_failures) {
    verify_failure("snapshot_" + suffix, flag);
  }
  store_.delta_persist_supported = true;
}

TEST_F(VectorIndexRegistryTest, LifecycleStatePersistsAndRollsBackOnFailures) {
  const std::string index_name = "idx_registry_lifecycle_direct";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  ASSERT_TRUE(
      vector_index_registry::set_lifecycle_state(index_name, "building"));
  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("building", info.lifecycle_state);

  store_.fail_save_manifest = true;
  EXPECT_FALSE(
      vector_index_registry::set_lifecycle_state(index_name, "failed"));
  store_.fail_save_manifest = false;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("building", info.lifecycle_state);

  EXPECT_FALSE(
      vector_index_registry::set_lifecycle_state("missing_index", "ready"));
  vector_index_registry::index_backfill_token token;
  EXPECT_FALSE(vector_index_registry::begin_backfill("missing_index", &token));

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       RestoreRuntimeStateRestoresAndPersistsSnapshot) {
  const std::string index_name = "idx_registry_restore_public";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  vector_index_registry::registry_state_snapshot snapshot;
  ASSERT_TRUE(vector_index_registry::snapshot_runtime_state(&snapshot));

  ASSERT_TRUE(vector_index_registry::upsert(index_name, 2, {2.0F, 2.0F}));
  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(vector_index_registry::restore_runtime_state(snapshot, false));
  EXPECT_EQ(0U, store_.save_metadata_calls);
  EXPECT_EQ(0U, store_.save_committed_calls);
  EXPECT_EQ(0U, store_.save_change_log_calls);
  EXPECT_EQ(0U, store_.save_prepared_calls);
  EXPECT_EQ(0U, store_.save_segment_tasks_calls);
  EXPECT_EQ(0U, store_.save_manifest_calls);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {2.0F, 2.0F}, 5, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::upsert(index_name, 3, {3.0F, 3.0F}));
  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(vector_index_registry::restore_runtime_state(snapshot, true));
  EXPECT_EQ(1U, store_.save_metadata_calls);
  EXPECT_EQ(1U, store_.save_committed_calls);
  EXPECT_EQ(1U, store_.save_change_log_calls);
  EXPECT_EQ(1U, store_.save_prepared_calls);
  EXPECT_EQ(1U, store_.save_segment_tasks_calls);
  EXPECT_EQ(1U, store_.save_manifest_calls);

  result.clear();
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {3.0F, 3.0F}, 5, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       SearchBatchForThdTxnCoversBatchAndPendingFallbackPaths) {
  const std::string index_name = "idx_registry_batch_search";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 2, {2.0F, 2.0F}));

  std::vector<std::vector<vector_index::search_result>> batch_results;
  EXPECT_FALSE(vector_index_registry::search_batch_for_thd_txn(
      nullptr, 0, index_name, {{1.0F, 1.0F}}, 1, nullptr));

  ASSERT_TRUE(vector_index_registry::search_batch_for_thd_txn(
      nullptr, 0, index_name, {}, 1, &batch_results));
  EXPECT_TRUE(batch_results.empty());

  ASSERT_TRUE(vector_index_registry::search_batch_for_thd_txn(
      nullptr, 0, index_name, {{1.0F, 1.0F}, {2.0F, 2.0F}}, 1, &batch_results));
  ASSERT_EQ(2U, batch_results.size());
  ASSERT_EQ(1U, batch_results[0].size());
  ASSERT_EQ(1U, batch_results[1].size());
  EXPECT_EQ(1U, batch_results[0][0].doc_id);
  EXPECT_EQ(2U, batch_results[1][0].doc_id);

  EXPECT_FALSE(vector_index_registry::search_batch_for_thd_txn(
      nullptr, 0, index_name, {{1.0F}}, 1, &batch_results));

  constexpr uint64_t thd_id = 9001;
  constexpr uint64_t statement_id = 77;
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      thd_id, statement_id, index_name, 3, {3.0F, 3.0F}));
  ASSERT_TRUE(vector_index_registry::search_batch_for_thd_txn(
      nullptr, thd_id, index_name, {{3.0F, 3.0F}, {1.0F, 1.0F}}, 1,
      &batch_results));
  ASSERT_EQ(2U, batch_results.size());
  ASSERT_EQ(1U, batch_results[0].size());
  ASSERT_EQ(1U, batch_results[1].size());
  EXPECT_EQ(3U, batch_results[0][0].doc_id);
  EXPECT_EQ(1U, batch_results[1][0].doc_id);
  ASSERT_TRUE(vector_index_registry::rollback_thd_txn(thd_id));

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
       TransactionalDeltaRollsBackBeforeWaitingForIndexPublication) {
  const std::string index_name = "idx_registry_delta_lock_order";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  store_.ResetPersistCountersForTesting();
  const uint64_t rollback_before = store_.rollback_persist_calls;
  store_.block_committed_delta_apply();
  auto upsert = std::async(std::launch::async, [&] {
    return vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F});
  });
  if (!store_.wait_for_committed_delta_apply(1)) {
    store_.release_committed_delta_apply();
    const auto upsert_status = upsert.wait_for(std::chrono::seconds(5));
    if (upsert_status == std::future_status::ready) {
      EXPECT_TRUE(upsert.get());
    }
    FAIL() << "upsert did not reach committed-delta persistence";
  }

  std::promise<void> release_publication;
  auto release_publication_future = release_publication.get_future();
  std::promise<bool> publication_acquired;
  auto publication_acquired_future = publication_acquired.get_future();
  auto publication = std::async(
      std::launch::async,
      [&, release = std::move(release_publication_future)]() mutable {
        vector_statement_publication::publication_guard guard;
        const bool locked = guard.lock_indexes({index_name});
        publication_acquired.set_value(locked);
        if (!locked) return false;
        release.wait();
        return true;
      });

  if (publication_acquired_future.wait_for(std::chrono::seconds(5)) !=
      std::future_status::ready) {
    store_.release_committed_delta_apply();
    release_publication.set_value();
    (void)upsert.wait_for(std::chrono::seconds(5));
    (void)publication.wait_for(std::chrono::seconds(5));
    FAIL() << "publication reservation was held while truth rows were staged";
  }
  const bool locked = publication_acquired_future.get();
  if (!locked) {
    store_.release_committed_delta_apply();
    release_publication.set_value();
    (void)upsert.wait_for(std::chrono::seconds(5));
    (void)publication.wait_for(std::chrono::seconds(5));
    FAIL() << "publication reservation could not be acquired";
  }

  store_.release_committed_delta_apply();
  if (!store_.wait_for_rollback_persist(rollback_before + 1)) {
    release_publication.set_value();
    (void)publication.wait_for(std::chrono::seconds(5));
    (void)upsert.wait_for(std::chrono::seconds(5));
    FAIL() << "staged truth rows were not rolled back before waiting";
  }
  EXPECT_EQ(std::future_status::timeout,
            upsert.wait_for(std::chrono::milliseconds(100)));
  release_publication.set_value();
  ASSERT_TRUE(publication.get());
  ASSERT_TRUE(upsert.get());

  EXPECT_EQ(2U, store_.apply_committed_delta_calls);
  EXPECT_EQ(2U, store_.append_change_log_delta_calls);
  EXPECT_EQ(rollback_before + 1, store_.rollback_persist_calls);
  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  ASSERT_TRUE(store_.load_committed(&committed_rows));
  ASSERT_EQ(1U, committed_rows.size());
  EXPECT_EQ(index_name, committed_rows[0].index_name);
  EXPECT_EQ(1U, committed_rows[0].doc_id);
  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  ASSERT_TRUE(store_.load_change_log(&change_log_rows));
  ASSERT_EQ(1U, change_log_rows.size());
  EXPECT_EQ(index_name, change_log_rows[0].index_name);
  EXPECT_EQ(1U, change_log_rows[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       MultiIndexCommitRollsBackRowsBeforeWaitingForCatalogPublication) {
  const std::string first_index = "idx_registry_catalog_lock_a";
  const std::string second_index = "idx_registry_catalog_lock_b";
  ASSERT_TRUE(vector_index_registry::create_index(first_index, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(second_index, 2,
                                                  "euclidean", "memory",
                                                  "native"));

  store_.ResetPersistCountersForTesting();
  const uint64_t rollback_before = store_.rollback_persist_calls;
  store_.block_committed_delta_apply();
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(vector_index_registry::stage_upsert(txn_id, first_index, 1,
                                                  {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::stage_upsert(txn_id, second_index, 2,
                                                  {2.0F, 2.0F}));
  auto commit = std::async(
      std::launch::async,
      [&] { return vector_index_registry::commit_txn(txn_id); });
  if (!store_.wait_for_committed_delta_apply(1)) {
    store_.release_committed_delta_apply();
    (void)commit.wait_for(std::chrono::seconds(5));
    FAIL() << "explicit commit did not reach committed-delta persistence";
  }

  std::promise<void> release_catalog;
  auto release_catalog_future = release_catalog.get_future();
  std::promise<bool> catalog_acquired;
  auto catalog_acquired_future = catalog_acquired.get_future();
  auto catalog = std::async(
      std::launch::async,
      [&, release = std::move(release_catalog_future)]() mutable {
        vector_statement_publication::publication_guard guard;
        const bool locked = guard.lock_catalog();
        catalog_acquired.set_value(locked);
        if (!locked) return false;
        release.wait();
        return true;
      });

  if (catalog_acquired_future.wait_for(std::chrono::seconds(5)) !=
      std::future_status::ready) {
    store_.release_committed_delta_apply();
    release_catalog.set_value();
    (void)commit.wait_for(std::chrono::seconds(5));
    (void)catalog.wait_for(std::chrono::seconds(5));
    FAIL() << "catalog publication was held while truth rows were staged";
  }
  const bool locked = catalog_acquired_future.get();
  if (!locked) {
    store_.release_committed_delta_apply();
    release_catalog.set_value();
    (void)commit.wait_for(std::chrono::seconds(5));
    (void)catalog.wait_for(std::chrono::seconds(5));
    FAIL() << "catalog publication could not be acquired";
  }

  store_.release_committed_delta_apply();
  if (!store_.wait_for_rollback_persist(rollback_before + 1)) {
    release_catalog.set_value();
    (void)catalog.wait_for(std::chrono::seconds(5));
    (void)commit.wait_for(std::chrono::seconds(5));
    FAIL() << "multi-index rows were not rolled back before catalog wait";
  }
  EXPECT_EQ(std::future_status::timeout,
            commit.wait_for(std::chrono::milliseconds(100)));
  release_catalog.set_value();
  ASSERT_TRUE(catalog.get());
  ASSERT_TRUE(commit.get());

  EXPECT_EQ(2U, store_.apply_committed_delta_calls);
  EXPECT_EQ(2U, store_.append_change_log_delta_calls);
  EXPECT_EQ(rollback_before + 1, store_.rollback_persist_calls);
  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  ASSERT_TRUE(store_.load_committed(&committed_rows));
  ASSERT_EQ(2U, committed_rows.size());
  EXPECT_EQ(1U, store_.CommittedRowsForIndexForTesting(first_index));
  EXPECT_EQ(1U, store_.CommittedRowsForIndexForTesting(second_index));
  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  ASSERT_TRUE(store_.load_change_log(&change_log_rows));
  EXPECT_EQ(2U, change_log_rows.size());

  ASSERT_TRUE(vector_index_registry::drop_index(first_index));
  ASSERT_TRUE(vector_index_registry::drop_index(second_index));
}

TEST_F(VectorIndexRegistryTest,
       TransactionalDeltaFailsIfDdlRemovesPendingChangesDuringWait) {
  const std::string index_name = "idx_registry_delta_drop_wins";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  store_.ResetPersistCountersForTesting();
  const uint64_t rollback_before = store_.rollback_persist_calls;
  store_.block_committed_delta_apply();
  auto upsert = std::async(std::launch::async, [&] {
    return vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F});
  });
  if (!store_.wait_for_committed_delta_apply(1)) {
    store_.release_committed_delta_apply();
    (void)upsert.wait_for(std::chrono::seconds(5));
    FAIL() << "upsert did not reach committed-delta persistence";
  }

  vector_statement_publication::publication_guard ddl_publication;
  if (!ddl_publication.lock_indexes({index_name})) {
    store_.release_committed_delta_apply();
    (void)upsert.wait_for(std::chrono::seconds(5));
    FAIL() << "DDL publication could not be acquired";
  }

  store_.release_committed_delta_apply();
  if (!store_.wait_for_rollback_persist(rollback_before + 1)) {
    ddl_publication = vector_statement_publication::publication_guard();
    (void)upsert.wait_for(std::chrono::seconds(5));
    FAIL() << "staged rows were not rolled back before DDL publication";
  }
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  ddl_publication = vector_statement_publication::publication_guard();

  EXPECT_FALSE(upsert.get());
  vector_index_registry::index_info info;
  EXPECT_FALSE(vector_index_registry::get_index_info(index_name, &info));
  std::vector<vector_index_metadata_store::committed_row> committed_rows;
  ASSERT_TRUE(store_.load_committed(&committed_rows));
  EXPECT_TRUE(committed_rows.empty());
  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  ASSERT_TRUE(store_.load_change_log(&change_log_rows));
  EXPECT_TRUE(change_log_rows.empty());
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

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &before));
  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(vector_index_registry::set_search_ef(index_name, 96));

  expect_metadata_only_persist(store_);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(96U, info.search_ef);
  EXPECT_EQ(2U, info.committed_entry_count);
  EXPECT_EQ(before.index_identity, info.index_identity);
  EXPECT_EQ(before.truth_generation, info.truth_generation);
  EXPECT_EQ(before.config_generation + 1, info.config_generation);
  EXPECT_EQ(before.artifact_generation, info.artifact_generation);
  EXPECT_EQ(before.runtime_generation, info.runtime_generation);

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  ASSERT_TRUE(store_.load_metadata(&metadata_rows));
  ASSERT_EQ(1U, metadata_rows.size());
  EXPECT_EQ(info.config_generation, metadata_rows[0].config_generation);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       StartupRecoveryAppliesPersistedFaissTuningBeforeTruthReplay) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kFaiss)) {
    GTEST_SKIP() << "Faiss provider is not compiled in";
  }

  const std::string root =
      std::string(testing::TempDir()) + "/registry_faiss_tuning_restart_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  external_snapshot_root_guard root_guard(root);

  const std::string index_name = "idx_registry_faiss_tuning_restart";
  vector_index_registry::create_index_options options;
  options.build_threads_specified = true;
  options.build_threads = 2;
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "external", "faiss",
                                                  options));
  for (uint64_t doc_id = 1; doc_id <= 4; ++doc_id) {
    ASSERT_TRUE(vector_index_registry::upsert(
        index_name, doc_id,
        {static_cast<float>(doc_id), static_cast<float>(doc_id + 1)}));
  }

  vector_index_registry::reset_for_testing();

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(2U, info.faiss_build_threads);
  EXPECT_EQ(2U, info.build_diagnostics.effective_build_threads);
  EXPECT_EQ(2U, info.build_diagnostics.effective_blas_threads);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search(index_name, {1.0F, 2.0F}, 1,
                                            &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  std::filesystem::remove_all(root, ec);
}

TEST_F(VectorIndexRegistryTest,
       SetDiskAnnSearchComplexityPersistsOnlyMetadata) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "DiskANN provider is not compiled in";
  }
  const std::string index_name = "idx_registry_diskann_search_metadata_only";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "external", "diskann"));

  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(
      vector_index_registry::set_diskann_search_complexity(index_name, 144));

  expect_metadata_only_persist(store_);

  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(
      vector_index_registry::set_diskann_search_beamwidth(index_name, 16));

  expect_metadata_only_persist(store_);

  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(vector_index_registry::set_diskann_pq_code_budget_size(index_name,
                                                                     1048576));

  expect_metadata_only_persist(store_);

  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(vector_index_registry::set_diskann_disk_pq_dims(index_name, 12));

  expect_metadata_only_persist(store_);

  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(
      vector_index_registry::set_diskann_accelerate_build(index_name, true));

  expect_metadata_only_persist(store_);

  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(
      vector_index_registry::set_diskann_shuffle_build(index_name, true));

  expect_metadata_only_persist(store_);

  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(
      vector_index_registry::set_diskann_use_bfs_cache(index_name, true));

  expect_metadata_only_persist(store_);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(144U, info.diskann_search_complexity);
  EXPECT_EQ(16U, info.diskann_search_beamwidth);
  EXPECT_EQ(1048576U, info.diskann_pq_code_budget_size);
  EXPECT_EQ(12U, info.diskann_disk_pq_dims);
  EXPECT_TRUE(info.diskann_accelerate_build);
  EXPECT_TRUE(info.diskann_shuffle_build);
  EXPECT_TRUE(info.diskann_use_bfs_cache);

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  ASSERT_TRUE(store_.load_metadata(&metadata_rows));
  ASSERT_EQ(1U, metadata_rows.size());
  EXPECT_EQ(12U, metadata_rows[0].diskann_disk_pq_dims);
  EXPECT_TRUE(metadata_rows[0].diskann_accelerate_build);
  EXPECT_TRUE(metadata_rows[0].diskann_shuffle_build);
  EXPECT_TRUE(metadata_rows[0].diskann_use_bfs_cache);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, SetDiskAnnBuildThreadsPersistsOnlyMetadata) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "DiskANN provider is not compiled in";
  }
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

TEST_F(VectorIndexRegistryTest, SetDiskAnnBuildModePersistsOnlyMetadata) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "DiskANN provider is not compiled in";
  }
  const std::string index_name = "idx_registry_diskann_mode_metadata_only";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "external", "diskann"));

  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(vector_index_registry::set_diskann_build_mode(
      index_name, vector_index::diskann_build_mode::kOffline));

  expect_metadata_only_persist(store_);

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(vector_index::diskann_build_mode::kOffline,
            info.diskann_build_mode_value);
  EXPECT_TRUE(info.diskann_build_mode_specified);

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  ASSERT_TRUE(store_.load_metadata(&metadata_rows));
  ASSERT_EQ(1U, metadata_rows.size());
  EXPECT_EQ(vector_index::diskann_build_mode::kOffline,
            metadata_rows[0].diskann_build_mode_value);
  EXPECT_TRUE(metadata_rows[0].diskann_build_mode_specified);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, CreateIndexBuildThreadsPreferStatementOption) {
  if (!hnswlib_tuning_supported() ||
      !vector_index::backend_provider_supported(
          vector_index::backend_provider::kFaiss) ||
      !vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn) ||
      !vector_index::backend_provider_supported(
          vector_index::backend_provider::kNative)) {
    GTEST_SKIP() << "build-thread routing requires all vector providers";
  }

  vector_index_registry::create_index_options options;
  options.build_threads_specified = true;
  options.build_threads = 2;

  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_create_hnsw_threads", 2, "euclidean", "memory", "hnswlib",
      options));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_create_faiss_threads", 4, "euclidean", "external", "faiss",
      options));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_create_diskann_threads", 2, "euclidean", "external",
      "diskann", options));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_create_native_threads_ignored", 2, "euclidean", "memory",
      "native", options));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_registry_create_hnsw_threads", &info));
  EXPECT_EQ(2U, info.hnsw_build_threads);
  EXPECT_EQ(0U, info.faiss_build_threads);
  EXPECT_EQ(0U, info.diskann_build_threads);

  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_registry_create_faiss_threads", &info));
  EXPECT_EQ(0U, info.hnsw_build_threads);
  EXPECT_EQ(2U, info.faiss_build_threads);
  EXPECT_EQ(0U, info.diskann_build_threads);

  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_registry_create_diskann_threads", &info));
  EXPECT_EQ(0U, info.hnsw_build_threads);
  EXPECT_EQ(0U, info.faiss_build_threads);
  EXPECT_EQ(2U, info.diskann_build_threads);

  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_registry_create_native_threads_ignored", &info));
  EXPECT_EQ(0U, info.hnsw_build_threads);
  EXPECT_EQ(0U, info.faiss_build_threads);
  EXPECT_EQ(0U, info.diskann_build_threads);
}

TEST_F(VectorIndexRegistryTest,
       CreateIndexBuildThreadsUseGlobalWhenStatementIsZero) {
  if (!hnswlib_tuning_supported() ||
      !vector_index::backend_provider_supported(
          vector_index::backend_provider::kFaiss) ||
      !vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "global build-thread routing requires all vector libraries";
  }
  UlongGuard hnsw_threads(&opt_vector_hnsw_build_threads, 3);
  UlongGuard faiss_threads(&opt_vector_faiss_build_threads, 4);
  UlongGuard diskann_threads(&opt_vector_diskann_build_threads, 5);

  vector_index_registry::create_index_options inherit_options;
  inherit_options.build_threads_specified = true;
  inherit_options.build_threads = 0;

  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_create_hnsw_global_threads", 2, "euclidean", "memory",
      "hnswlib", inherit_options));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_create_faiss_global_threads", 4, "euclidean", "external",
      "faiss", inherit_options));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_registry_create_diskann_global_threads", 2, "euclidean", "external",
      "diskann", inherit_options));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_registry_create_hnsw_global_threads", &info));
  EXPECT_EQ(3U, info.hnsw_build_threads);

  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_registry_create_faiss_global_threads", &info));
  EXPECT_EQ(4U, info.faiss_build_threads);

  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_registry_create_diskann_global_threads", &info));
  EXPECT_EQ(5U, info.diskann_build_threads);
}

TEST_F(VectorIndexRegistryTest, CreateIndexAppliesDiskAnnSearchGlobals) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "DiskANN provider is not compiled in";
  }
  UlongGuard complexity_guard(&opt_vector_diskann_search_complexity, 200);
  UlongGuard beamwidth_guard(&opt_vector_diskann_search_beamwidth, 32);

  const std::string index_name = "idx_registry_diskann_search_globals";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "external", "diskann"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(200U, info.diskann_search_complexity);
  EXPECT_EQ(32U, info.diskann_search_beamwidth);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, CreateIndexAppliesDiskAnnStatementOptions) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "DiskANN provider is not compiled in";
  }
  vector_index_registry::create_index_options options;
  options.diskann_max_degree = 24;
  options.diskann_build_complexity = 80;
  options.diskann_disk_pq_dims = 12;
  options.diskann_accelerate_build_specified = true;
  options.diskann_accelerate_build = true;
  options.diskann_shuffle_build_specified = true;
  options.diskann_shuffle_build = true;
  options.diskann_use_bfs_cache_specified = true;
  options.diskann_use_bfs_cache = true;

  const std::string index_name = "idx_registry_diskann_statement_options";
  ASSERT_TRUE(vector_index_registry::create_index(
      index_name, 2, "euclidean", "external", "diskann", options));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(24U, info.diskann_max_degree);
  EXPECT_EQ(80U, info.diskann_build_complexity);
  EXPECT_EQ(12U, info.diskann_disk_pq_dims);
  EXPECT_TRUE(info.diskann_accelerate_build);
  EXPECT_TRUE(info.diskann_shuffle_build);
  EXPECT_TRUE(info.diskann_use_bfs_cache);

  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  ASSERT_TRUE(store_.load_metadata(&metadata_rows));
  ASSERT_EQ(1U, metadata_rows.size());
  EXPECT_EQ(24U, metadata_rows[0].diskann_max_degree);
  EXPECT_EQ(80U, metadata_rows[0].diskann_build_complexity);
  EXPECT_EQ(12U, metadata_rows[0].diskann_disk_pq_dims);
  EXPECT_TRUE(metadata_rows[0].diskann_accelerate_build);
  EXPECT_TRUE(metadata_rows[0].diskann_shuffle_build);
  EXPECT_TRUE(metadata_rows[0].diskann_use_bfs_cache);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       EffectiveOptionsRespectProviderGlobalsAndStatementOverrides) {
  UlongGuard hnsw_threads(&opt_vector_hnsw_build_threads, 3);
  UlongGuard faiss_threads(&opt_vector_faiss_build_threads, 4);
  UlongGuard diskann_threads(&opt_vector_diskann_build_threads, 5);
  UlongGuard max_degree(&opt_vector_diskann_max_degree, 48);
  UlongGuard build_complexity(&opt_vector_diskann_build_complexity, 96);
  UlonglongGuard pq_budget(&opt_vector_diskann_pq_code_budget_size, 4096);
  UlongGuard disk_pq_dims(&opt_vector_diskann_disk_pq_dims, 12);
  BoolGuard accelerate(&opt_vector_diskann_accelerate_build, true);
  BoolGuard shuffle(&opt_vector_diskann_shuffle_build, true);
  BoolGuard bfs(&opt_vector_diskann_use_bfs_cache, true);
  UlongGuard search_complexity(&opt_vector_diskann_search_complexity, 144);
  UlongGuard search_beamwidth(&opt_vector_diskann_search_beamwidth, 28);

  vector_index_registry::create_index_options options;
  auto hnsw =
      vector_index_registry::effective_options_for_testing("hnswlib", options);
  EXPECT_EQ(3U, hnsw.hnsw_build_threads);
  EXPECT_EQ(0U, hnsw.faiss_build_threads);
  EXPECT_EQ(0U, hnsw.diskann_build_threads);
  EXPECT_FALSE(hnsw.diskann_accelerate_build);

  auto faiss =
      vector_index_registry::effective_options_for_testing("faiss", options);
  EXPECT_EQ(0U, faiss.hnsw_build_threads);
  EXPECT_EQ(4U, faiss.faiss_build_threads);
  EXPECT_EQ(0U, faiss.diskann_build_threads);
  EXPECT_EQ(0U, faiss.diskann_search_complexity);

  auto diskann =
      vector_index_registry::effective_options_for_testing("diskann", options);
  EXPECT_EQ(0U, diskann.hnsw_build_threads);
  EXPECT_EQ(0U, diskann.faiss_build_threads);
  EXPECT_EQ(5U, diskann.diskann_build_threads);
  EXPECT_EQ(48U, diskann.diskann_max_degree);
  EXPECT_EQ(96U, diskann.diskann_build_complexity);
  EXPECT_EQ(4096U, diskann.diskann_pq_code_budget_size);
  EXPECT_EQ(12U, diskann.diskann_disk_pq_dims);
  EXPECT_TRUE(diskann.diskann_accelerate_build);
  EXPECT_TRUE(diskann.diskann_shuffle_build);
  EXPECT_TRUE(diskann.diskann_use_bfs_cache);
  EXPECT_EQ(144U, diskann.diskann_search_complexity);
  EXPECT_EQ(28U, diskann.diskann_search_beamwidth);

  options.build_threads_specified = true;
  options.build_threads = 9;
  options.diskann_max_degree = 64;
  options.diskann_build_complexity = 128;
  options.diskann_disk_pq_dims = 16;
  options.diskann_accelerate_build_specified = true;
  options.diskann_accelerate_build = false;
  options.diskann_shuffle_build_specified = true;
  options.diskann_shuffle_build = false;
  options.diskann_use_bfs_cache_specified = true;
  options.diskann_use_bfs_cache = false;

  hnsw =
      vector_index_registry::effective_options_for_testing("hnswlib", options);
  EXPECT_EQ(9U, hnsw.hnsw_build_threads);
  faiss =
      vector_index_registry::effective_options_for_testing("faiss", options);
  EXPECT_EQ(9U, faiss.faiss_build_threads);
  diskann =
      vector_index_registry::effective_options_for_testing("diskann", options);
  EXPECT_EQ(9U, diskann.diskann_build_threads);
  EXPECT_EQ(64U, diskann.diskann_max_degree);
  EXPECT_EQ(128U, diskann.diskann_build_complexity);
  EXPECT_EQ(16U, diskann.diskann_disk_pq_dims);
  EXPECT_FALSE(diskann.diskann_accelerate_build);
  EXPECT_FALSE(diskann.diskann_shuffle_build);
  EXPECT_FALSE(diskann.diskann_use_bfs_cache);

  options.build_threads = 0;
  hnsw =
      vector_index_registry::effective_options_for_testing("hnswlib", options);
  EXPECT_EQ(3U, hnsw.hnsw_build_threads);

  const auto native =
      vector_index_registry::effective_options_for_testing("native", options);
  EXPECT_EQ(0U, native.hnsw_build_threads);
  EXPECT_EQ(0U, native.faiss_build_threads);
  EXPECT_EQ(0U, native.diskann_build_threads);
  EXPECT_FALSE(native.diskann_use_bfs_cache);

  const auto invalid =
      vector_index_registry::effective_options_for_testing("missing", options);
  EXPECT_EQ(0U, invalid.hnsw_build_threads);
  EXPECT_EQ(0U, invalid.faiss_build_threads);
  EXPECT_EQ(0U, invalid.diskann_build_threads);
  EXPECT_EQ(0U, invalid.diskann_max_degree);
  EXPECT_FALSE(invalid.diskann_accelerate_build);
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
  EXPECT_EQ(0U, store_.append_change_log_delta_calls);
  EXPECT_EQ(1U, store_.erase_change_log_sequences_calls);
  EXPECT_EQ(0U, store_.save_change_log_calls);

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
       CommitTxnCompactsFallbackStoreWithoutDeltaPersistence) {
  store_.delta_persist_supported = false;
  vector_index_registry::set_change_log_compact_threshold_for_testing(3);
  const std::string index_name = "idx_registry_changelog_full_compact";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  store_.ResetPersistCountersForTesting();
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_NE(0U, txn_id);
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 3, {3.0F, 3.0F}));
  ASSERT_TRUE(vector_index_registry::commit_txn(txn_id));

  EXPECT_EQ(1U, store_.save_committed_calls);
  EXPECT_EQ(2U, store_.save_change_log_calls);
  std::vector<vector_index_metadata_store::change_log_row> rows;
  ASSERT_TRUE(store_.load_change_log(&rows));
  EXPECT_TRUE(rows.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest, DirectEraseUsesInternalTransaction) {
  const std::string index_name = "idx_registry_direct_erase";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 7, {1.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::erase(index_name, 7));

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 2.0F}, 1, &results));
  EXPECT_TRUE(results.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnRestoresRuntimeWhenEntryStoreApplyFails) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  const std::string index_name = "idx_registry_commit_apply_failure";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_NE(0U, txn_id);
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 2.0F}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
                             "+d,vector_service_fail_commit_entry_store_apply");
    EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  }
  EXPECT_EQ(1U, vector_index_registry::pending_txn_changes(txn_id));
  ASSERT_TRUE(vector_index_registry::commit_txn(txn_id));

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 2.0F}, 1, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(1U, results.front().doc_id);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       DurableTransactionalIntentReportsEveryExpectedTokenDrift) {
  const std::string index_name = "idx_registry_intent_token_drift";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));

  vector_index_truth_store::publication_token current;
  current.exists = true;
  current.index_identity = info.index_identity;
  current.truth_generation = info.truth_generation;
  current.config_generation = info.config_generation;
  current.artifact_generation = info.artifact_generation;
  current.runtime_generation = info.runtime_generation;
  current.lifecycle_version = info.lifecycle_version;

  using token_mutator =
      std::function<void(vector_index_truth_store::publication_token *)>;
  const std::vector<token_mutator> mutations = {
      [](auto *token) { token->exists = !token->exists; },
      [](auto *token) { ++token->index_identity; },
      [](auto *token) { ++token->truth_generation; },
      [](auto *token) { ++token->config_generation; },
      [](auto *token) { ++token->artifact_generation; },
      [](auto *token) { ++token->runtime_generation; },
      [](auto *token) { ++token->lifecycle_version; },
      [](auto *token) { ++token->source_generation; },
  };

  uint64_t publication_id = 1;
  for (const auto &mutate : mutations) {
    vector_index_truth_store::publication_intent intent;
    intent.index_name = index_name;
    intent.publication_id = publication_id++;
    intent.operation =
        vector_index_truth_store::publication_operation::kTransactionalDml;
    intent.expected = current;
    mutate(&intent.expected);
    intent.target = current;
    ++intent.target.truth_generation;

    store_.publication_intents = {intent};
    vector_index_registry::detail::
        reset_publication_intent_recovery_for_testing();
    vector_index_registry::schedule_publication_intent_recovery();
    EXPECT_FALSE(vector_index_registry::ensure_publication_intents_recovered());
    ASSERT_EQ(1U, store_.publication_intents.size());
    store_.publication_intents.clear();
  }

  vector_index_registry::detail::
      reset_publication_intent_recovery_for_testing();
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnKeepsPendingChangesWhenExactCompactionEraseFails) {
  vector_index_registry::set_change_log_compact_threshold_for_testing(3);
  const std::string index_name = "idx_registry_compact_erase_failure";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 3, {3.0F, 3.0F}));

  store_.fail_erase_change_log_sequences = true;
  EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  EXPECT_EQ(3U, vector_index_registry::pending_txn_changes(txn_id));
  EXPECT_EQ(1U, store_.rollback_persist_calls);

  store_.fail_erase_change_log_sequences = false;
  ASSERT_TRUE(vector_index_registry::commit_txn(txn_id));
  EXPECT_EQ(0U, vector_index_registry::pending_txn_changes(txn_id));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {3.0F, 3.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(3U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnRestoresChangeLogRowsWhenCompactionThenManifestFails) {
  vector_index_registry::set_change_log_compact_threshold_for_testing(3);
  const std::string index_name = "idx_registry_compact_manifest_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  const uint64_t baseline_txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(vector_index_registry::stage_upsert(baseline_txn_id, index_name,
                                                  1, {1.0F, 1.0F}));
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
  ASSERT_TRUE(vector_index_registry::stage_upsert(failing_txn_id, index_name, 2,
                                                  {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::stage_upsert(failing_txn_id, index_name, 3,
                                                  {3.0F, 3.0F}));

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

TEST_F(VectorIndexRegistryTest, LazyLoadAdvancesTransactionIdPastPreparedRows) {
  vector_index_metadata_store::prepared_change_row row;
  row.format_id = 1;
  row.gtrid_length = 1;
  row.xid_data = "g";
  row.txn_id = 500;
  row.op = vector_index_metadata_store::change_op::kErase;
  row.index_name = "idx_prepared_high_water";
  row.doc_id = 1;
  store_.SetPreparedRowsForTesting({row});

  EXPECT_EQ(501U, vector_index_registry::begin_txn());
}

TEST_F(VectorIndexRegistryTest,
       LazyLoadExhaustsTransactionIdsAtPersistedMaximum) {
  vector_index_metadata_store::prepared_change_row row;
  row.format_id = 1;
  row.gtrid_length = 1;
  row.xid_data = "g";
  row.txn_id = std::numeric_limits<uint64_t>::max();
  row.op = vector_index_metadata_store::change_op::kErase;
  row.index_name = "idx_prepared_exhausted";
  row.doc_id = 1;
  store_.SetPreparedRowsForTesting({row});

  EXPECT_EQ(0U, vector_index_registry::begin_txn());
}

TEST_F(VectorIndexRegistryTest,
       LazyLoadRejectsConflictingPreparedTransactionIdentities) {
  const auto expect_rejected =
      [&](const std::vector<vector_index_metadata_store::prepared_change_row>
              &rows) {
        store_.SetPreparedRowsForTesting(rows);
        vector_index_registry::reset_for_testing();
        const size_t quarantine_count_before = store_.quarantine_records.size();

        std::vector<std::string> index_names;
        EXPECT_FALSE(vector_index_registry::list_indexes(&index_names));
        EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
                  vector_index_registry::registry_health());
        EXPECT_EQ("prepared_validation_failed",
                  vector_index_registry::registry_failure_reason());
        ASSERT_EQ(quarantine_count_before + 1,
                  store_.quarantine_records.size());
        EXPECT_EQ("prepared", store_.quarantine_records.back().artifact_name);
        std::vector<vector_index_metadata_store::prepared_change_row>
            preserved_rows;
        ASSERT_TRUE(store_.load_prepared(&preserved_rows));
        EXPECT_EQ(rows.size(), preserved_rows.size());

        store_.SetPreparedRowsForTesting({});
        vector_index_registry::reset_for_testing();
      };

  const vector_index_metadata_store::prepared_change_row base{
      70,
      3,
      0,
      "xid",
      false,
      700,
      vector_index_metadata_store::change_op::kErase,
      "idx_prepared",
      1,
      {}};

  auto same_xid_different_txn = base;
  same_xid_different_txn.txn_id = 701;
  expect_rejected({base, same_xid_different_txn});

  auto same_txn_different_xid = base;
  same_txn_different_xid.xid_data = "alt";
  expect_rejected({base, same_txn_different_xid});

  auto same_xid_different_tc_state = base;
  same_xid_different_tc_state.prepared_in_tc = true;
  expect_rejected({base, same_xid_different_tc_state});
}

TEST_F(VectorIndexRegistryTest, TransactionIdAllocationDoesNotWrap) {
  ASSERT_NE(0U, vector_index_registry::begin_txn());
  vector_index_registry::detail::g_next_txn_id.store(
      std::numeric_limits<uint64_t>::max(), std::memory_order_relaxed);

  EXPECT_EQ(std::numeric_limits<uint64_t>::max(),
            vector_index_registry::begin_txn());
  EXPECT_EQ(0U, vector_index_registry::begin_txn());
  EXPECT_FALSE(vector_index_registry::stage_upsert(0, "idx_invalid_txn", 1,
                                                   {1.0F, 1.0F}));
  EXPECT_FALSE(vector_index_registry::commit_txn(0));
  EXPECT_FALSE(vector_index_registry::rollback_txn(0));
}

TEST_F(VectorIndexRegistryTest, LazyLoadRemovesIncompleteCreateLifecycleRows) {
  const std::string ready_index = "test.t_ready.v";
  const std::string creating_index = "test.t_creating.v";
  const std::string backfilling_index = "test.t_backfilling.v";

  const auto ready_metadata =
      make_metadata_row_for_testing(ready_index, "ready");
  const auto creating_metadata =
      make_metadata_row_for_testing(creating_index, "creating");
  const auto backfilling_metadata =
      make_metadata_row_for_testing(backfilling_index, "backfilling");
  store_.SetMetadataRowsForTesting(
      {ready_metadata, creating_metadata, backfilling_metadata});
  store_.SetCommittedRowsForTesting({
      vector_index_metadata_store::committed_row{ready_index, 1, {1.0F, 1.0F}},
      vector_index_metadata_store::committed_row{
          creating_index, 2, {2.0F, 2.0F}},
      vector_index_metadata_store::committed_row{
          backfilling_index, 3, {3.0F, 3.0F}},
  });
  store_.SetChangeLogRowsForTesting({
      vector_index_metadata_store::change_log_row{
          1,
          100,
          vector_index_metadata_store::change_op::kUpsert,
          ready_index,
          1,
          {1.5F, 1.5F},
          ready_metadata.index_identity,
          1,
          1},
      vector_index_metadata_store::change_log_row{
          2,
          101,
          vector_index_metadata_store::change_op::kUpsert,
          creating_index,
          4,
          {4.0F, 4.0F},
          creating_metadata.index_identity,
          1,
          1},
      vector_index_metadata_store::change_log_row{
          3,
          102,
          vector_index_metadata_store::change_op::kUpsert,
          backfilling_index,
          5,
          {5.0F, 5.0F},
          backfilling_metadata.index_identity,
          1,
          1},
  });
  store_.SetPreparedRowsForTesting({
      vector_index_metadata_store::prepared_change_row{
          42,
          3,
          0,
          "abc",
          false,
          200,
          vector_index_metadata_store::change_op::kUpsert,
          ready_index,
          6,
          {6.0F, 6.0F}},
      vector_index_metadata_store::prepared_change_row{
          43,
          3,
          0,
          "def",
          false,
          201,
          vector_index_metadata_store::change_op::kUpsert,
          creating_index,
          7,
          {7.0F, 7.0F}},
      vector_index_metadata_store::prepared_change_row{
          44,
          3,
          0,
          "ghi",
          false,
          202,
          vector_index_metadata_store::change_op::kUpsert,
          backfilling_index,
          8,
          {8.0F, 8.0F}},
  });
  vector_index_metadata_store::segment_task_row ready_task;
  ready_task.index_name = ready_index;
  ready_task.generation = 1;
  ready_task.segment_id = 1;
  ready_task.state = vector_index_metadata_store::segment_task_state::kReady;
  ready_task.vector_path = "/tmp/ready.fbin";
  vector_index_metadata_store::segment_task_row creating_task = ready_task;
  creating_task.index_name = creating_index;
  creating_task.segment_id = 2;
  vector_index_metadata_store::segment_task_row backfilling_task = ready_task;
  backfilling_task.index_name = backfilling_index;
  backfilling_task.segment_id = 3;
  store_.SetSegmentTaskRowsForTesting(
      {ready_task, creating_task, backfilling_task});
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

  std::vector<vector_index_metadata_store::segment_task_row> segment_task_rows;
  ASSERT_TRUE(store_.load_segment_tasks(&segment_task_rows));
  ASSERT_EQ(1U, segment_task_rows.size());
  EXPECT_EQ(ready_index, segment_task_rows[0].index_name);

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

TEST_F(VectorIndexRegistryTest, LazyLoadNormalizesSegmentTaskRecoveryState) {
  const std::filesystem::path root =
      std::filesystem::path(DATA_DIR) / "vector_segment_task_recovery";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  ASSERT_TRUE(std::filesystem::create_directories(root, ec));

  const auto make_file = [](const std::filesystem::path &path) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    file << "x";
    return file.good();
  };
  const auto path_string = [](const std::filesystem::path &path) {
    return path.string();
  };

  const std::string ready_index = "test.t_segment_ready.v";
  const std::string pending_index = "test.t_segment_pending.v";
  const std::string building_index = "test.t_segment_building.v";
  const std::string missing_index = "test.t_segment_missing.v";
  const std::string exact_file_index = "test.t_segment_exact_file.v";
  const std::string prefix_file_index = "test.t_segment_prefix_file.v";
  const std::string empty_prefix_index = "test.t_segment_empty_prefix.v";
  const std::string missing_parent_index = "test.t_segment_missing_parent.v";
  const std::string stale_index = "test.t_segment_stale.v";
  const std::string abandoned_index = "test.t_segment_abandoned.v";

  store_.SetMetadataRowsForTesting({
      make_metadata_row_for_testing(ready_index, "ready"),
      make_metadata_row_for_testing(pending_index, "ready"),
      make_metadata_row_for_testing(building_index, "ready"),
      make_metadata_row_for_testing(missing_index, "ready"),
      make_metadata_row_for_testing(exact_file_index, "ready"),
      make_metadata_row_for_testing(prefix_file_index, "ready"),
      make_metadata_row_for_testing(empty_prefix_index, "ready"),
      make_metadata_row_for_testing(missing_parent_index, "ready"),
  });

  ASSERT_TRUE(make_file(root / "ready.fbin"));
  ASSERT_TRUE(make_file(root / "ready.u64"));
  ASSERT_TRUE(std::filesystem::create_directory(root / "ready_artifact", ec));
  ASSERT_TRUE(make_file(root / "pending.fbin"));
  ASSERT_TRUE(make_file(root / "pending.u64"));
  ASSERT_TRUE(make_file(root / "pending_artifact.tmp"));
  ASSERT_TRUE(make_file(root / "building.fbin"));
  ASSERT_TRUE(make_file(root / "building.u64"));
  ASSERT_TRUE(make_file(root / "building_artifact.tmp"));
  ASSERT_TRUE(make_file(root / "missing.u64"));
  ASSERT_TRUE(make_file(root / "exact_file.fbin"));
  ASSERT_TRUE(make_file(root / "exact_file.u64"));
  ASSERT_TRUE(make_file(root / "exact_file_artifact"));
  ASSERT_TRUE(make_file(root / "prefix_file.fbin"));
  ASSERT_TRUE(make_file(root / "prefix_file.u64"));
  ASSERT_TRUE(make_file(root / "prefix_file_artifact.disk.index"));
  ASSERT_TRUE(make_file(root / "empty_prefix.fbin"));
  ASSERT_TRUE(make_file(root / "empty_prefix.u64"));
  ASSERT_TRUE(make_file(root / "missing_parent.fbin"));
  ASSERT_TRUE(make_file(root / "missing_parent.u64"));
  ASSERT_TRUE(make_file(root / "stale_artifact.tmp"));
  ASSERT_TRUE(make_file(root / "abandoned_artifact.tmp"));

  auto make_task = [&](const std::string &index_name,
                       vector_index_metadata_store::segment_task_state state,
                       const std::string &basename) {
    vector_index_metadata_store::segment_task_row row;
    row.index_name = index_name;
    row.generation = 1;
    row.segment_id = 1;
    row.state = state;
    row.row_count = 1;
    row.payload_size = 8;
    row.vector_path = path_string(root / (basename + ".fbin"));
    row.docid_path = path_string(root / (basename + ".u64"));
    row.artifact_prefix = path_string(root / (basename + "_artifact"));
    return row;
  };

  auto empty_prefix_task = make_task(
      empty_prefix_index,
      vector_index_metadata_store::segment_task_state::kReady, "empty_prefix");
  empty_prefix_task.artifact_prefix.clear();
  auto missing_parent_task =
      make_task(missing_parent_index,
                vector_index_metadata_store::segment_task_state::kReady,
                "missing_parent");
  missing_parent_task.artifact_prefix =
      path_string(root / "missing-parent" / "artifact");
  store_.SetSegmentTaskRowsForTesting({
      make_task(ready_index,
                vector_index_metadata_store::segment_task_state::kReady,
                "ready"),
      make_task(pending_index,
                vector_index_metadata_store::segment_task_state::kPending,
                "pending"),
      make_task(building_index,
                vector_index_metadata_store::segment_task_state::kBuilding,
                "building"),
      make_task(missing_index,
                vector_index_metadata_store::segment_task_state::kBuilding,
                "missing"),
      make_task(exact_file_index,
                vector_index_metadata_store::segment_task_state::kReady,
                "exact_file"),
      make_task(prefix_file_index,
                vector_index_metadata_store::segment_task_state::kReady,
                "prefix_file"),
      empty_prefix_task,
      missing_parent_task,
      make_task(stale_index,
                vector_index_metadata_store::segment_task_state::kPending,
                "stale"),
      make_task(abandoned_index,
                vector_index_metadata_store::segment_task_state::kAbandoned,
                "abandoned"),
  });

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));

  std::vector<vector_index_metadata_store::segment_task_row> rows;
  ASSERT_TRUE(store_.load_segment_tasks(&rows));
  ASSERT_EQ(8U, rows.size());

  const auto find_task = [&rows](const std::string &index_name) {
    return std::find_if(rows.begin(), rows.end(),
                        [&index_name](const auto &row) {
                          return row.index_name == index_name;
                        });
  };

  ASSERT_NE(rows.end(), find_task(ready_index));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kReady,
            find_task(ready_index)->state);
  ASSERT_NE(rows.end(), find_task(pending_index));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kPending,
            find_task(pending_index)->state);
  ASSERT_NE(rows.end(), find_task(building_index));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kPending,
            find_task(building_index)->state);
  ASSERT_NE(rows.end(), find_task(missing_index));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kFailed,
            find_task(missing_index)->state);
  EXPECT_EQ(1U, find_task(missing_index)->last_error_code);
  ASSERT_NE(rows.end(), find_task(exact_file_index));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kReady,
            find_task(exact_file_index)->state);
  ASSERT_NE(rows.end(), find_task(prefix_file_index));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kReady,
            find_task(prefix_file_index)->state);
  ASSERT_NE(rows.end(), find_task(empty_prefix_index));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kReady,
            find_task(empty_prefix_index)->state);
  ASSERT_NE(rows.end(), find_task(missing_parent_index));
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kFailed,
            find_task(missing_parent_index)->state);
  EXPECT_EQ(1U, find_task(missing_parent_index)->last_error_code);
  EXPECT_EQ(rows.end(), find_task(stale_index));
  EXPECT_EQ(rows.end(), find_task(abandoned_index));

  EXPECT_TRUE(std::filesystem::exists(root / "ready_artifact", ec));
  EXPECT_FALSE(std::filesystem::exists(root / "pending_artifact.tmp", ec));
  EXPECT_FALSE(std::filesystem::exists(root / "building_artifact.tmp", ec));
  EXPECT_FALSE(std::filesystem::exists(root / "stale_artifact.tmp", ec));
  EXPECT_FALSE(std::filesystem::exists(root / "abandoned_artifact.tmp", ec));
  EXPECT_EQ(1U, store_.save_segment_tasks_calls);

  std::filesystem::remove_all(root, ec);
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

TEST_F(VectorIndexRegistryTest,
       StandaloneOwnerSchemaPersistsWithoutCreatingTableBinding) {
  const std::string index_name = "idx_standalone_owner";
  ASSERT_TRUE(vector_index_registry::create_index(
      index_name, 2, "euclidean", "memory", "native", "owner_db"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("owner_db", info.owner_schema);
  EXPECT_TRUE(info.schema_name.empty());
  EXPECT_TRUE(info.table_name.empty());
  EXPECT_TRUE(info.column_name.empty());

  vector_index_registry::reset_for_testing();
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("owner_db", info.owner_schema);
  EXPECT_TRUE(info.schema_name.empty());
  EXPECT_TRUE(info.table_name.empty());
  EXPECT_TRUE(info.column_name.empty());
}

TEST_F(VectorIndexRegistryTest,
       DropIndexesForDatabaseUsesStandaloneOwnerSchema) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_owned_drop", 2, "euclidean", "memory", "native", "db_owned"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_owned_keep", 2, "euclidean", "memory", "native", "db_keep"));

  ASSERT_TRUE(vector_index_registry::drop_indexes_for_database("db_owned"));

  vector_index_registry::index_info info;
  EXPECT_FALSE(vector_index_registry::get_index_info("idx_owned_drop", &info));
  ASSERT_TRUE(vector_index_registry::get_index_info("idx_owned_keep", &info));
  EXPECT_EQ("db_keep", info.owner_schema);
}

TEST_F(VectorIndexRegistryTest,
       DropDatabaseIgnoresStandaloneIndexNameSchemaPrefix) {
  const std::string index_name = "db_name_only.t1.v";
  ASSERT_TRUE(vector_index_registry::create_index(
      index_name, 2, "euclidean", "memory", "native", "db_owner"));

  ASSERT_TRUE(vector_index_registry::drop_indexes_for_database("db_name_only"));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("db_owner", info.owner_schema);
}

TEST_F(VectorIndexRegistryTest,
       TableLifecycleIgnoresStandaloneNamesThatLookMapped) {
  const std::string reset_name = "db_name_only.t_reset.v";
  const std::string rename_name = "db_name_only.t_rename.v";
  const std::string drop_name = "db_name_only.t_drop.v";
  for (const std::string *index_name :
       {&reset_name, &rename_name, &drop_name}) {
    ASSERT_TRUE(vector_index_registry::create_index(
        *index_name, 2, "euclidean", "memory", "native", "db_owner"));
    ASSERT_TRUE(vector_index_registry::upsert(*index_name, 1, {1.0F, 1.0F}));
  }

  ASSERT_TRUE(vector_index_registry::reset_mapped_indexes_for_table(
      "db_name_only", "t_reset"));
  ASSERT_TRUE(vector_index_registry::rename_indexes_for_table(
      "db_name_only", "t_rename", "db_renamed", "t_renamed"));
  ASSERT_TRUE(
      vector_index_registry::drop_indexes_for_table("db_name_only", "t_drop"));

  vector_index_registry::index_info info;
  for (const std::string *index_name :
       {&reset_name, &rename_name, &drop_name}) {
    ASSERT_TRUE(vector_index_registry::get_index_info(*index_name, &info));
    EXPECT_EQ("db_owner", info.owner_schema);
    EXPECT_EQ(1U, info.committed_entry_count);
  }
  EXPECT_FALSE(
      vector_index_registry::get_index_info("db_renamed.t_renamed.v", &info));
}

TEST_F(VectorIndexRegistryTest,
       MappedIndexLifecycleHelpersMutateLoadedMetadata) {
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
      "db_lifecycle", "t_lifecycle", "db_lifecycle_new", "t_lifecycle_new"));

  vector_index_registry::index_info info;
  EXPECT_FALSE(vector_index_registry::get_index_info(index_v1, &info));
  const std::string renamed_v1 = "db_lifecycle_new.t_lifecycle_new.v1";
  ASSERT_TRUE(vector_index_registry::get_index_info(renamed_v1, &info));
  EXPECT_EQ("db_lifecycle_new", info.schema_name);
  EXPECT_EQ("t_lifecycle_new", info.table_name);
  EXPECT_EQ("v1", info.column_name);

  ASSERT_TRUE(vector_index_registry::rename_index_for_column(
      "db_lifecycle_new", "t_lifecycle_new", "v1", "v1_new"));
  const std::string renamed_column = "db_lifecycle_new.t_lifecycle_new.v1_new";
  EXPECT_FALSE(vector_index_registry::get_index_info(renamed_v1, &info));
  ASSERT_TRUE(vector_index_registry::get_index_info(renamed_column, &info));
  EXPECT_EQ("v1_new", info.column_name);

  ASSERT_TRUE(vector_index_registry::drop_index_for_column(
      "db_lifecycle_new", "t_lifecycle_new", "v2"));
  EXPECT_FALSE(vector_index_registry::get_index_info(
      "db_lifecycle_new.t_lifecycle_new.v2", &info));

  ASSERT_TRUE(vector_index_registry::drop_indexes_for_table("db_lifecycle_new",
                                                            "t_lifecycle_new"));
  EXPECT_FALSE(vector_index_registry::get_index_info(renamed_column, &info));
  ASSERT_TRUE(vector_index_registry::get_index_info(other_index, &info));

  ASSERT_TRUE(vector_index_registry::drop_indexes_for_database("db_other"));
  EXPECT_FALSE(vector_index_registry::get_index_info(other_index, &info));
}

TEST_F(VectorIndexRegistryTest, MappedIndexHelpersNoopWhenMetadataIsUnloaded) {
  vector_index_registry::detail::g_metadata_loaded = false;

  EXPECT_TRUE(
      vector_index_registry::drop_indexes_for_table("db_none", "t_none"));
  EXPECT_TRUE(vector_index_registry::drop_indexes_for_database("db_none"));
  EXPECT_TRUE(vector_index_registry::reset_mapped_indexes_for_table("db_none",
                                                                    "t_none"));
  EXPECT_TRUE(vector_index_registry::rename_indexes_for_table(
      "db_none", "t_none", "db_new", "t_new"));
  EXPECT_TRUE(
      vector_index_registry::drop_index_for_column("db_none", "t_none", "v"));
  EXPECT_TRUE(vector_index_registry::rename_index_for_column(
      "db_none", "t_none", "v", "v"));
  EXPECT_TRUE(vector_index_registry::rename_index_for_column(
      "db_none", "t_none", "v_old", "v_new"));
}

TEST_F(VectorIndexRegistryTest,
       InternalBindingAndTuningHelpersCoverFallbackBranches) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kFaiss) ||
      !vector_index::backend_provider_supported(
          vector_index::backend_provider::kHnswlib) ||
      !vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn) ||
      !vector_index::backend_provider_supported(
          vector_index::backend_provider::kNative)) {
    GTEST_SKIP() << "tuning fallback coverage requires HNSWLIB, FAISS, "
                    "DiskANN, and native providers";
  }
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
        vector_index_registry::detail::make_mapped_index_name("db_new", "t_new",
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
    EXPECT_EQ("id", mapped_binding.doc_id_column_name);

    std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
    EXPECT_FALSE(
        vector_index_registry::detail::snapshot_metadata_locked(nullptr));
    EXPECT_TRUE(vector_index_registry::detail::snapshot_metadata_locked(
        &metadata_rows));
  }

  ASSERT_TRUE(vector_index_registry::create_index(
      "idx_apply_tuning_hnsw", 2, "euclidean", "memory", "hnswlib"));
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
    info.search_ef = 96;
    info.hnsw_build_threads = 2;
    info.hnsw_m = 24;
    info.hnsw_ef_construction = 240;
    EXPECT_TRUE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_hnsw", info));

    info = vector_index_registry::index_info{};
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
    info.diskann_build_mode_value = vector_index::diskann_build_mode::kSerial;
    info.diskann_build_mode_specified = true;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_faiss", info));

    info = vector_index_registry::index_info{};
    info.diskann_search_complexity = 80;
    EXPECT_FALSE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_native", info));

    info = vector_index_registry::index_info{};
    info.diskann_search_complexity = 80;
    EXPECT_TRUE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_diskann", info));

    info = vector_index_registry::index_info{};
    info.faiss_build_threads = 2;
    info.faiss_nlist = 2;
    info.faiss_nprobe = 1;
    EXPECT_TRUE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_faiss", info));

    info = vector_index_registry::index_info{};
    info.faiss_nlist = 2;
    info.faiss_nprobe = 1;
    info.faiss_pq_m = 1;
    info.faiss_pq_bits = 8;
    EXPECT_TRUE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_faiss", info));

    info = vector_index_registry::index_info{};
    info.diskann_max_degree = 32;
    info.diskann_build_complexity = 96;
    info.diskann_build_threads = 2;
    info.diskann_build_mode_specified = true;
    info.diskann_build_mode_value = vector_index::diskann_build_mode::kOffline;
    info.diskann_search_complexity = 80;
    info.diskann_search_beamwidth = 16;
    info.diskann_pq_code_budget_size = 1024;
    info.diskann_disk_pq_dims = 1;
    info.diskann_accelerate_build = true;
    info.diskann_shuffle_build = true;
    info.diskann_use_bfs_cache = true;
    EXPECT_TRUE(vector_index_registry::detail::apply_index_tuning_locked(
        "idx_apply_tuning_diskann", info));
  }

  ASSERT_TRUE(vector_index_registry::drop_index("idx_apply_tuning_hnsw"));
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

TEST_F(VectorIndexRegistryTest,
       DropCleanupFailureReservesNameUntilRestartRecoveryCompletes) {
  const std::string index_name = "idx_registry_drop_cleanup_recovery";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 7, {7.0F, 1.0F}));

  store_.fail_erase_committed_index_batch = true;
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  ASSERT_EQ(1U, store_.publication_intents.size());
  EXPECT_EQ(index_name, store_.publication_intents.front().index_name);
  EXPECT_EQ(vector_index_truth_store::publication_operation::kDropIndex,
            store_.publication_intents.front().operation);
  EXPECT_EQ(1U, store_.CommittedRowsForIndexForTesting(index_name));
  EXPECT_FALSE(vector_index_registry::create_index(
      index_name, 2, "euclidean", "memory", "native"));

  store_.fail_erase_committed_index_batch = false;
  vector_index_registry::reset_for_testing();
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  EXPECT_TRUE(store_.publication_intents.empty());
  EXPECT_EQ(0U, store_.CommittedRowsForIndexForTesting(index_name));

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {7.0F, 1.0F}, 1, &results));
  EXPECT_TRUE(results.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       DropIntentDeletionFailureRetriesIdempotentPhysicalCleanup) {
  const std::string index_name = "idx_registry_drop_intent_delete_retry";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 8, {8.0F, 1.0F}));

  store_.fail_delete_publication_intent = true;
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  ASSERT_EQ(1U, store_.publication_intents.size());
  EXPECT_EQ(0U, store_.CommittedRowsForIndexForTesting(index_name));
  const uint64_t cleanup_calls = store_.erase_committed_index_batch_calls;
  EXPECT_FALSE(vector_index_registry::create_index(
      index_name, 2, "euclidean", "memory", "native"));

  store_.fail_delete_publication_intent = false;
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  EXPECT_TRUE(store_.publication_intents.empty());
  EXPECT_GT(store_.erase_committed_index_batch_calls, cleanup_calls);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       DropIntentPersistFailureRestoresRuntimeWithoutReservingName) {
  const std::string index_name = "idx_registry_drop_intent_persist_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 9, {9.0F, 1.0F}));

  store_.fail_save_publication_intent = true;
  EXPECT_FALSE(vector_index_registry::drop_index(index_name));
  store_.fail_save_publication_intent = false;
  EXPECT_TRUE(store_.publication_intents.empty());

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(1U, info.committed_entry_count);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       DeferredDropRollbackRestoresStateAndAtomicallyDeletesCleanupIntent) {
  const std::string index_name = "idx_registry_drop_statement_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {10.0F, 1.0F}));

  vector_index_registry::registry_state_snapshot before_drop;
  ASSERT_TRUE(vector_index_registry::snapshot_runtime_state(&before_drop));
  vector_index_registry::dropped_index_artifacts artifacts;
  ASSERT_TRUE(vector_index_registry::drop_index(index_name, &artifacts));
  ASSERT_TRUE(artifacts.valid);
  ASSERT_TRUE(artifacts.has_cleanup_intent);
  ASSERT_EQ(1U, store_.publication_intents.size());
  EXPECT_EQ(1U, store_.CommittedRowsForIndexForTesting(index_name));

  ASSERT_TRUE(vector_index_registry::restore_runtime_state_after_drop_rollback(
      before_drop, artifacts));
  EXPECT_TRUE(store_.publication_intents.empty());
  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {10.0F, 1.0F}, 1, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(10U, results.front().doc_id);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       DeferredDropRollbackRestoresNonTransactionalTruthStoreState) {
  const std::string index_name = "idx_registry_drop_file_rollback";
  store_.transactional = false;
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 11, {11.0F, 1.0F}));

  vector_index_registry::registry_state_snapshot before_drop;
  ASSERT_TRUE(vector_index_registry::snapshot_runtime_state(&before_drop));
  vector_index_registry::dropped_index_artifacts artifacts;
  ASSERT_TRUE(vector_index_registry::drop_index(index_name, &artifacts));
  ASSERT_TRUE(artifacts.valid);
  EXPECT_FALSE(artifacts.has_cleanup_intent);

  ASSERT_TRUE(vector_index_registry::restore_runtime_state_after_drop_rollback(
      before_drop, artifacts));
  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {11.0F, 1.0F}, 1, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(11U, results.front().doc_id);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       DeferredDropRollbackPersistenceFailureMatrixRemainsRetryable) {
  const std::string index_name = "idx_registry_drop_rollback_failures";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 12, {12.0F, 1.0F}));

  vector_index_registry::registry_state_snapshot current_snapshot;
  ASSERT_TRUE(vector_index_registry::snapshot_runtime_state(&current_snapshot));
  vector_index_registry::dropped_index_artifacts invalid_artifacts;
  EXPECT_FALSE(vector_index_registry::restore_runtime_state_after_drop_rollback(
      current_snapshot, invalid_artifacts));
  invalid_artifacts.valid = true;
  invalid_artifacts.has_cleanup_intent = true;
  invalid_artifacts.cleanup_intent.operation =
      vector_index_truth_store::publication_operation::kRebuildIndex;
  EXPECT_FALSE(vector_index_registry::restore_runtime_state_after_drop_rollback(
      current_snapshot, invalid_artifacts));

  struct failure_case {
    const char *name;
    bool in_memory_truth_store::*flag;
  };
  const std::vector<failure_case> failures = {
      {"begin", &in_memory_truth_store::fail_begin_persist},
      {"metadata", &in_memory_truth_store::fail_save_metadata},
      {"committed", &in_memory_truth_store::fail_save_committed},
      {"change_log", &in_memory_truth_store::fail_save_change_log},
      {"prepared", &in_memory_truth_store::fail_save_prepared},
      {"segment_tasks", &in_memory_truth_store::fail_save_segment_tasks},
      {"manifest", &in_memory_truth_store::fail_save_manifest},
      {"intent", &in_memory_truth_store::fail_delete_publication_intent},
      {"commit", &in_memory_truth_store::fail_commit_persist},
  };

  for (const auto &failure : failures) {
    SCOPED_TRACE(failure.name);
    vector_index_registry::registry_state_snapshot before_drop;
    ASSERT_TRUE(vector_index_registry::snapshot_runtime_state(&before_drop));
    vector_index_registry::dropped_index_artifacts artifacts;
    ASSERT_TRUE(vector_index_registry::drop_index(index_name, &artifacts));
    ASSERT_TRUE(artifacts.valid);
    ASSERT_TRUE(artifacts.has_cleanup_intent);

    store_.*(failure.flag) = true;
    EXPECT_FALSE(
        vector_index_registry::restore_runtime_state_after_drop_rollback(
            before_drop, artifacts));
    store_.*(failure.flag) = false;

    vector_index_registry::index_info info;
    EXPECT_FALSE(vector_index_registry::get_index_info(index_name, &info));
    ASSERT_EQ(1U, store_.publication_intents.size());

    ASSERT_TRUE(
        vector_index_registry::restore_runtime_state_after_drop_rollback(
            before_drop, artifacts));
    ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
    EXPECT_EQ(1U, info.committed_entry_count);
    EXPECT_TRUE(store_.publication_intents.empty());
  }

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       DropPersistFailureFailStopsWhenRuntimeCannotBeRestored) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  const std::string index_name = "idx_registry_drop_restore_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  store_.fail_save_manifest = true;
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
                             "+d,vector_registry_fail_restore_runtime_state");
    EXPECT_FALSE(vector_index_registry::drop_index(index_name));
  }
  store_.fail_save_manifest = false;
  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("drop_runtime_restore_failed",
            vector_index_registry::registry_failure_reason());
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

TEST_F(VectorIndexRegistryTest,
       CommitTxnRejectsExhaustedChangeLogSequenceWithoutPublishing) {
  const std::string index_name = "idx_registry_changelog_exhausted";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));
  {
    vector_gunit::ScopedValueGuard<uint64_t> sequence_guard(
        &vector_index_registry::detail::g_next_change_log_sequence,
        std::numeric_limits<uint64_t>::max());
    EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  }
  EXPECT_EQ(1U, vector_index_registry::pending_txn_changes(txn_id));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_TRUE(vector_index_registry::rollback_txn(txn_id));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       ManifestPersistenceRejectsExhaustedVersionWithoutDroppingIndex) {
  const std::string index_name = "idx_registry_manifest_exhausted";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  {
    vector_gunit::ScopedValueGuard<uint64_t> version_guard(
        &vector_index_registry::detail::g_manifest_version,
        std::numeric_limits<uint64_t>::max() - 1);
    EXPECT_FALSE(vector_index_registry::drop_index(index_name));
  }

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  EXPECT_NE(index_names.end(),
            std::find(index_names.begin(), index_names.end(), index_name));

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
       CommitTxnFailsStopWhenCommittedStateRollbackFails) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  const std::string index_name = "idx_registry_commit_runtime_restore_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));

  store_.fail_save_manifest = true;
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_service_fail_restore_committed_state");
    EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  }
  store_.fail_save_manifest = false;

  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("runtime_commit_state_restore_failed",
            vector_index_registry::registry_failure_reason());
  std::vector<vector_index::search_result> result;
  EXPECT_FALSE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
}

TEST_F(VectorIndexRegistryTest,
       CommitTxnFailsStopWhenPendingStateRollbackFails) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  const std::string index_name = "idx_registry_commit_pending_restore_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  const uint64_t txn_id = vector_index_registry::begin_txn();
  ASSERT_TRUE(
      vector_index_registry::stage_upsert(txn_id, index_name, 1, {1.0F, 1.0F}));

  store_.fail_save_manifest = true;
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_service_fail_restore_pending_state");
    EXPECT_FALSE(vector_index_registry::commit_txn(txn_id));
  }
  store_.fail_save_manifest = false;

  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("pending_state_restore_failed",
            vector_index_registry::registry_failure_reason());
  std::vector<vector_index::search_result> result;
  EXPECT_FALSE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
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

TEST_F(VectorIndexRegistryTest,
       ThdTxnSavepointPreflightDoesNotMutatePendingState) {
  const std::string index_name = "idx_registry_savepoint_preflight";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      204, 1, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(204, 1));
  ASSERT_TRUE(
      vector_index_registry::preflight_savepoint_thd_txn(204, "sp1"));
  EXPECT_FALSE(vector_index_registry::preflight_rollback_to_savepoint_thd_txn(
      204, "sp1"));
  ASSERT_TRUE(vector_index_registry::savepoint_thd_txn(204, "sp1"));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      204, 2, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(204, 2));
  ASSERT_TRUE(vector_index_registry::preflight_rollback_to_savepoint_thd_txn(
      204, "SP1"));
  ASSERT_TRUE(
      vector_index_registry::preflight_release_savepoint_thd_txn(204, "sp1"));

  // Both preflight calls are read-only, so the original marker can still
  // roll back the second row.
  ASSERT_TRUE(
      vector_index_registry::rollback_to_savepoint_thd_txn(204, "sp1"));
  ASSERT_TRUE(vector_index_registry::commit_thd_txn(204));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {0.0F, 0.0F}, 10, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  EXPECT_TRUE(vector_index_registry::preflight_rollback_to_savepoint_thd_txn(
      999, "missing"));
  EXPECT_TRUE(vector_index_registry::preflight_release_savepoint_thd_txn(
      999, "missing"));
  ASSERT_TRUE(
      vector_index_registry::preflight_savepoint_thd_txn(999, "missing"));
  EXPECT_EQ(vector_index_registry::detail::g_thd_txn_contexts.end(),
            vector_index_registry::detail::g_thd_txn_contexts.find(999));
  EXPECT_FALSE(
      vector_index_registry::preflight_savepoint_thd_txn(999, ""));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       ThdTxnSavepointNamesFollowMysqlCollationSemantics) {
  const std::string index_name = "idx_registry_savepoint_collation";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      203, 1, index_name, 1, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(203, 1));
  ASSERT_TRUE(vector_index_registry::savepoint_thd_txn(203, "VectorPoint"));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      203, 2, index_name, 2, {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(203, 2));
  ASSERT_TRUE(vector_index_registry::savepoint_thd_txn(203, "vectorpoint"));

  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      203, 3, index_name, 3, {3.0F, 3.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(203, 3));
  ASSERT_TRUE(
      vector_index_registry::rollback_to_savepoint_thd_txn(203, "VECTORPOINT"));
  ASSERT_TRUE(
      vector_index_registry::release_savepoint_thd_txn(203, "VectorPoint"));
  ASSERT_TRUE(vector_index_registry::commit_thd_txn(203));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {0.0F, 0.0F}, 10, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_EQ(2U, result[1].doc_id);

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
  {
    std::lock_guard<std::shared_mutex> guard(
        vector_index_registry::detail::g_registry_mutex);
    ASSERT_TRUE(vector_index_registry::detail::g_index_service
                    .release_savepoint(1, "__stmt_10"));
  }
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
  {
    std::lock_guard<std::shared_mutex> guard(
        vector_index_registry::detail::g_registry_mutex);
    ASSERT_TRUE(vector_index_registry::detail::g_index_service
                    .release_savepoint(1, "__stmt_11"));
  }
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

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidPreservesExhaustedChangeLogSequence) {
  const std::string index_name = "idx_registry_prepared_sequence_exhausted";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid = make_test_xid(704, "prepared-sequence-exhausted");
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      704, 1, index_name, 7, {7.0F, 7.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(704, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(704, xid));

  {
    vector_gunit::ScopedValueGuard<uint64_t> sequence_guard(
        &vector_index_registry::detail::g_next_change_log_sequence,
        std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
    EXPECT_EQ(std::numeric_limits<uint64_t>::max(),
              vector_index_registry::detail::g_next_change_log_sequence);
  }

  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {7.0F, 7.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid));
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

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidIfLoadedScansPersistedRowsBeforeMetadataLoad) {
  XID xid = make_test_xid(701, "prepared-persisted-scan");
  store_.SetPreparedRowsForTesting(
      {{xid.get_format_id(),
        xid.get_gtrid_length(),
        xid.get_bqual_length(),
        std::string(xid.get_data(),
                    static_cast<size_t>(xid.get_gtrid_length() +
                                        xid.get_bqual_length())),
        false,
        1701,
        vector_index_metadata_store::change_op::kUpsert,
        "idx_registry_missing_persisted_scan",
        1,
        {1.0F, 1.0F}}});

  EXPECT_EQ(XAER_RMERR,
            vector_index_registry::commit_prepared_xid_if_loaded(xid));
  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));
  EXPECT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid));
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));
}

TEST_F(VectorIndexRegistryTest, DropIndexesForTableRemovesMatchingBindings) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db1.t1.c1", 2, "euclidean", "memory", "native", "db1", "t1", "c1",
      "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db1.t1.c2", 2, "euclidean", "memory", "native", "db1", "t1", "c2",
      "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db1.t2.c1", 2, "euclidean", "memory", "native", "db1", "t2", "c1",
      "id"));

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
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db2.t1.c1", 2, "euclidean", "memory", "native", "db2", "t1", "c1",
      "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db2.t1.c2", 2, "euclidean", "memory", "native", "db2", "t1", "c2",
      "id"));

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

TEST_F(VectorIndexRegistryTest, DropIndexesForDatabaseRemovesMatchingOwners) {
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbdrop.t1.c1", 2, "euclidean", "memory", "native", "dbdrop"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbdrop.t2.c1", 2, "euclidean", "memory", "native", "dbdrop"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbkeep.t1.c1", 2, "euclidean", "memory", "native", "dbkeep"));

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
      "dbdropfail.t1.c1", 2, "euclidean", "memory", "native", "dbdropfail"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbdropfail.t2.c1", 2, "euclidean", "memory", "native", "dbdropfail"));

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
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db3.t1.v", 2, "euclidean", "memory", "native", "db3", "t1", "v", "id"));
  ASSERT_TRUE(vector_index_registry::upsert("db3.t1.v", 11, {1.0F, 1.0F}));

  ASSERT_TRUE(vector_index_registry::rename_indexes_for_table(
      "db3", "t1", "db3", "t1_renamed"));

  vector_index_registry::index_info info;
  ASSERT_FALSE(vector_index_registry::get_index_info("db3.t1.v", &info));
  ASSERT_TRUE(vector_index_registry::get_index_info("db3.t1_renamed.v", &info));
  EXPECT_EQ(1U, info.committed_entry_count);
  std::string doc_id_column_name;
  ASSERT_TRUE(vector_index_registry::get_index_binding_for_testing(
      "db3.t1_renamed.v", nullptr, nullptr, nullptr, &doc_id_column_name));
  EXPECT_EQ("id", doc_id_column_name);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("db3.t1_renamed.v", {1.0F, 1.0F}, 1,
                                            &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("db3.t1_renamed.v"));
}

TEST_F(VectorIndexRegistryTest,
       RenameIndexesForTableRollbackOnPersistFailureKeepsRuntimeState) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db4.t1.v", 2, "euclidean", "memory", "native", "db4", "t1", "v", "id"));
  ASSERT_TRUE(vector_index_registry::upsert("db4.t1.v", 22, {2.0F, 2.0F}));

  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::rename_indexes_for_table(
      "db4", "t1", "db4", "t1_renamed"));
  store_.fail_save_manifest = false;

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info("db4.t1.v", &info));
  EXPECT_EQ("db4", info.owner_schema);
  EXPECT_EQ("db4", info.schema_name);
  EXPECT_EQ("t1", info.table_name);
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
  EXPECT_EQ("dbmove2", info.owner_schema);
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
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db5.t1.v", 2, "euclidean", "memory", "native", "db5", "t1", "v", "id"));
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
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db6.t1.v", 2, "euclidean", "memory", "native", "db6", "t1", "v", "id"));
  ASSERT_TRUE(vector_index_registry::upsert("db6.t1.v", 44, {4.0F, 4.0F}));

  ASSERT_TRUE(vector_index_registry::rename_index_for_column("db6", "t1", "v",
                                                             "vec_col"));

  vector_index_registry::index_info info;
  ASSERT_FALSE(vector_index_registry::get_index_info("db6.t1.v", &info));
  ASSERT_TRUE(vector_index_registry::get_index_info("db6.t1.vec_col", &info));
  EXPECT_EQ(1U, info.committed_entry_count);
  std::string doc_id_column_name;
  ASSERT_TRUE(vector_index_registry::get_index_binding_for_testing(
      "db6.t1.vec_col", nullptr, nullptr, nullptr, &doc_id_column_name));
  EXPECT_EQ("id", doc_id_column_name);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("db6.t1.vec_col", {4.0F, 4.0F}, 1,
                                            &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(44U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("db6.t1.vec_col"));
}

TEST_F(VectorIndexRegistryTest,
       RenameIndexForColumnRollbackOnPersistFailureKeepsRuntimeState) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db7.t1.v", 2, "euclidean", "memory", "native", "db7", "t1", "v", "id"));
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

TEST_F(VectorIndexRegistryTest,
       RenameIndexForColumnFailsStopOnDurabilityUnknown) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  const std::string root =
      std::string(testing::TempDir()) + "/registry_rename_unknown_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  external_snapshot_root_guard root_guard(root);

  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbrenameunknown.t1.v", 2, "euclidean", "memory", "native",
      "dbrenameunknown", "t1", "v", "id"));
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_standalone_rename_manifest_durability_unknown");
    EXPECT_FALSE(vector_index_registry::rename_index_for_column(
        "dbrenameunknown", "t1", "v", "w"));
  }
  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("standalone_rename_durability_unknown",
            vector_index_registry::registry_failure_reason());

  std::filesystem::remove_all(root, ec);
}

TEST_F(VectorIndexRegistryTest,
       RenameIndexForColumnFailsStopWhenDirectoryRollbackFails) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  const std::string root =
      std::string(testing::TempDir()) + "/registry_rename_rollback_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  external_snapshot_root_guard root_guard(root);

  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbrenamefail.t1.v", 2, "euclidean", "memory", "native",
      "dbrenamefail", "t1", "v", "id"));
  ASSERT_TRUE(vector_index_registry::rename_index_for_column(
      "dbrenamefail", "t1", "v", "prepared"));
  ASSERT_TRUE(vector_index_registry::rename_index_for_column(
      "dbrenamefail", "t1", "prepared", "v"));
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_standalone_rename_manifest_not_published,"
               "vector_standalone_rename_rollback_failure");
    EXPECT_FALSE(vector_index_registry::rename_index_for_column(
        "dbrenamefail", "t1", "v", "w"));
  }
  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("standalone_rename_durability_unknown",
            vector_index_registry::registry_failure_reason());

  std::filesystem::remove_all(root, ec);
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
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db7b.t1.v", 2, "euclidean", "memory", "native", "db7b", "t1", "v",
      "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db7b.t1.vec_col", 2, "euclidean", "memory", "native", "db7b", "t1",
      "vec_col", "id"));

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
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db8.t1.v", 2, "euclidean", "memory", "native", "db8", "t1", "v", "id"));
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db8.t1.other", 2, "euclidean", "memory", "native", "db8", "t1", "other",
      "id"));

  ASSERT_TRUE(vector_index_registry::drop_index_for_column("db8", "t1", "v"));

  vector_index_registry::index_info info;
  ASSERT_FALSE(vector_index_registry::get_index_info("db8.t1.v", &info));
  ASSERT_TRUE(vector_index_registry::get_index_info("db8.t1.other", &info));
  ASSERT_TRUE(vector_index_registry::drop_index_for_column("db8", "t1", "v"));

  ASSERT_TRUE(vector_index_registry::drop_index("db8.t1.other"));
}

TEST_F(VectorIndexRegistryTest,
       DropIndexForColumnRollbackOnPersistFailureKeepsRuntimeState) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "db8b.t1.v", 2, "euclidean", "memory", "native", "db8b", "t1", "v",
      "id"));
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

TEST_F(VectorIndexRegistryTest,
       RebuildAndRecoverAllRoundTripStandaloneSegments) {
  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, 0);
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;

  ASSERT_TRUE(vector_index_registry::create_index("idx_registry_standalone_all",
                                                  2, "euclidean", "memory",
                                                  "native", options));
  ASSERT_TRUE(vector_index_registry::upsert("idx_registry_standalone_all", 10,
                                            {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::upsert("idx_registry_standalone_all", 20,
                                            {2.0F, 2.0F}));
  ASSERT_TRUE(vector_index_registry::erase("idx_registry_standalone_all", 20));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_registry_standalone_all", &info));
  EXPECT_EQ("standalone", info.consistency_mode);
  EXPECT_FALSE(info.truth_store_enabled);
  EXPECT_EQ("delta_replay", info.build_source);
  EXPECT_EQ(0U, info.entry_count);
  EXPECT_EQ(1U, info.committed_entry_count);
  EXPECT_GT(info.standalone_segment_count, 0U);
  EXPECT_GT(info.standalone_segment_bytes, 0U);

  size_t rebuilt_count = 0;
  ASSERT_TRUE(vector_index_registry::rebuild_all_indexes(&rebuilt_count));
  EXPECT_EQ(1U, rebuilt_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(vector_index_registry::search("idx_registry_standalone_all",
                                            {1.0F, 1.0F}, 2, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);
  ASSERT_TRUE(vector_index_registry::get_index_info(
      "idx_registry_standalone_all", &info));
  EXPECT_EQ("raw_segments_compacted", info.build_source);
  EXPECT_EQ(1U, info.entry_count);
  EXPECT_EQ(1U, info.standalone_raw_segment_count);

  size_t recovered_count = 0;
  ASSERT_TRUE(vector_index_registry::recover_all_indexes(&recovered_count));
  EXPECT_EQ(1U, recovered_count);
  ASSERT_TRUE(vector_index_registry::search("idx_registry_standalone_all",
                                            {1.0F, 1.0F}, 2, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index("idx_registry_standalone_all"));
}

TEST_F(VectorIndexRegistryTest, SegmentedRebuildPersistsReadyTaskRows) {
  UlongGuard pipeline_guard(
      &opt_vector_build_pipeline_mode,
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented));
  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, 0);
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;

  const std::string index_name = "idx_registry_segment_task_ready";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native", options));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 0.0F}));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 20, {0.0F, 1.0F}));

  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));

  std::vector<vector_index_metadata_store::segment_task_row> rows;
  ASSERT_TRUE(store_.load_segment_tasks(&rows));
  ASSERT_EQ(1U, rows.size());
  EXPECT_EQ(index_name, rows[0].index_name);
  EXPECT_NE(0U, rows[0].segment_id);
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kReady,
            rows[0].state);
  EXPECT_EQ(2U, rows[0].row_count);
  EXPECT_GE(rows[0].payload_size, 2U * 2U * sizeof(float));
  EXPECT_FALSE(rows[0].vector_path.empty());
  EXPECT_FALSE(rows[0].docid_path.empty());
  EXPECT_EQ(1U, rows[0].attempt);
  EXPECT_EQ(0U, rows[0].last_error_code);
  EXPECT_NE(0U, rows[0].updated_ts);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       SegmentedRebuildPersistsTaskRowsBeforePublishFailure) {
  UlongGuard pipeline_guard(
      &opt_vector_build_pipeline_mode,
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented));
  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, 0);
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;

  const std::string index_name = "idx_registry_segment_task_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native", options));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 30, {3.0F, 0.0F}));

  const uint64_t rollback_count_before =
      vector_status::runtime_state_rollbacks();
  const uint64_t rollback_failure_count_before =
      vector_status::runtime_state_rollback_failures();

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_segment_task_before_publish");
    EXPECT_FALSE(vector_index_registry::rebuild_index(index_name));
  }

  EXPECT_EQ(rollback_count_before, vector_status::runtime_state_rollbacks());
  EXPECT_EQ(rollback_failure_count_before,
            vector_status::runtime_state_rollback_failures());

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("bulk_loading", info.lifecycle_state);

  std::vector<vector_index::search_result> result;
  EXPECT_FALSE(
      vector_index_registry::search(index_name, {3.0F, 0.0F}, 1, &result));

  std::vector<vector_index_metadata_store::segment_task_row> rows;
  ASSERT_TRUE(store_.load_segment_tasks(&rows));
  ASSERT_EQ(1U, rows.size());
  EXPECT_EQ(index_name, rows[0].index_name);
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kReady,
            rows[0].state);
  EXPECT_EQ(1U, rows[0].row_count);
  EXPECT_EQ(1U, rows[0].attempt);
  EXPECT_EQ(0U, rows[0].last_error_code);

  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {3.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(30U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       DurableSegmentedBuildIntentRecoversAfterPrePublishFailure) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  UlongGuard pipeline_guard(
      &opt_vector_build_pipeline_mode,
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented));
  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, 1024 * 1024);
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;

  const std::string index_name = "idx_segmented_intent_recovery";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native", options));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 41, {4.0F, 1.0F}));

  vector_index_truth_store::publication_intent intent;
  ASSERT_TRUE(vector_index_registry::make_statement_publication_intent(
      vector_index_truth_store::publication_operation::kBulkBuildIndex,
      index_name, std::string(), &intent));
  store_.publication_intents.push_back(intent);

  std::string failure_stage;
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_segment_task_before_publish");
    EXPECT_FALSE(vector_index_registry::publish_statement_publication_intent(
        intent, &failure_stage));
  }
  EXPECT_EQ("LOAD VECTOR DATA could not publish rebuilt backend",
            failure_stage);
  ASSERT_EQ(1U, store_.publication_intents.size());

  vector_index_registry::schedule_publication_intent_recovery();
  ASSERT_TRUE(vector_index_registry::ensure_publication_intents_recovered());
  EXPECT_TRUE(store_.publication_intents.empty());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {4.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(41U, result[0].doc_id);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       StandaloneLifecycleRollbackOnPersistFailureKeepsServingState) {
  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, 0);
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;

  const std::string index_name = "idx_registry_standalone_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native", options));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));

  store_.fail_save_manifest_once = true;
  EXPECT_FALSE(vector_index_registry::rebuild_index(index_name));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  store_.fail_save_manifest_once = true;
  EXPECT_FALSE(vector_index_registry::recover_index(index_name));
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       StandaloneCompactionPersistFailureRestoresRawFiles) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib provider is not compiled in";
  }
  UlongGuard pipeline_guard(
      &opt_vector_build_pipeline_mode,
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented));
  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, 0);
  UlonglongGuard segment_rows_guard(&opt_vector_build_segment_max_rows, 1);
  const std::string root =
      std::string(testing::TempDir()) + "/registry_compaction_persist_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  external_snapshot_root_guard root_guard(root);

  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;
  const std::string index_name = "idx_registry_compaction_persist";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "hnsw", options));
  for (uint64_t doc_id = 1; doc_id <= 8; ++doc_id) {
    ASSERT_TRUE(vector_index_registry::upsert(
        index_name, doc_id, {static_cast<float>(doc_id), 0.0F}));
  }
  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));

  const std::vector<std::string> source_files = regular_files_below(root);
  ASSERT_FALSE(source_files.empty());
  opt_vector_build_segment_max_rows = 4;
  store_.fail_save_manifest_once = true;
  EXPECT_FALSE(vector_index_registry::rebuild_index(index_name));
  EXPECT_EQ(source_files, regular_files_below(root));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ(8U, info.standalone_raw_segment_count);
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {8.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(8U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
  std::filesystem::remove_all(root, ec);
}

TEST_F(VectorIndexRegistryTest,
       StandaloneCompactionRollbackFailureFailStopsRegistry) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib provider is not compiled in";
  }
  UlongGuard pipeline_guard(
      &opt_vector_build_pipeline_mode,
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented));
  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, 0);
  UlonglongGuard segment_rows_guard(&opt_vector_build_segment_max_rows, 1);
  const std::string root =
      std::string(testing::TempDir()) + "/registry_compaction_rollback_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  external_snapshot_root_guard root_guard(root);

  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;
  const std::string index_name = "idx_registry_compaction_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "hnsw", options));
  for (uint64_t doc_id = 1; doc_id <= 8; ++doc_id) {
    ASSERT_TRUE(vector_index_registry::upsert(
        index_name, doc_id, {static_cast<float>(doc_id), 0.0F}));
  }
  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));

  opt_vector_build_segment_max_rows = 4;
  store_.fail_save_manifest_once = true;
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_standalone_compaction_rollback_failure");
    EXPECT_FALSE(vector_index_registry::rebuild_index(index_name));
  }
  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("standalone_rebuild_rollback_failed",
            vector_index_registry::registry_failure_reason());

  std::vector<vector_index::search_result> result;
  EXPECT_FALSE(
      vector_index_registry::search(index_name, {8.0F, 0.0F}, 1, &result));

  std::filesystem::remove_all(root, ec);
}

TEST_F(VectorIndexRegistryTest,
       StandaloneCompactionDurabilityUnknownFailStopsRegistry) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib provider is not compiled in";
  }
  UlongGuard pipeline_guard(
      &opt_vector_build_pipeline_mode,
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented));
  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, 0);
  UlonglongGuard segment_rows_guard(&opt_vector_build_segment_max_rows, 1);
  const std::string root =
      std::string(testing::TempDir()) + "/registry_compaction_unknown_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  external_snapshot_root_guard root_guard(root);

  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;
  const std::string index_name = "idx_registry_compaction_unknown";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "hnsw", options));
  for (uint64_t doc_id = 1; doc_id <= 8; ++doc_id) {
    ASSERT_TRUE(vector_index_registry::upsert(
        index_name, doc_id, {static_cast<float>(doc_id), 0.0F}));
  }
  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));

  const std::vector<std::string> source_files = regular_files_below(root);
  ASSERT_FALSE(source_files.empty());
  opt_vector_build_segment_max_rows = 4;
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_standalone_manifest_after_rename_failure");
    EXPECT_FALSE(vector_index_registry::rebuild_index(index_name));
  }
  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("artifact_generation_discard_failed",
            vector_index_registry::registry_failure_reason());
  for (const std::string &path : source_files) {
    const std::filesystem::path source_path =
        std::filesystem::path(root) / path;
    SCOPED_TRACE(source_path.string());
    EXPECT_TRUE(std::filesystem::exists(source_path));
  }
  EXPECT_GT(regular_files_below(root).size(), source_files.size());

  std::vector<vector_index::search_result> result;
  EXPECT_FALSE(
      vector_index_registry::search(index_name, {8.0F, 0.0F}, 1, &result));

  std::filesystem::remove_all(root, ec);
}

TEST_F(VectorIndexRegistryTest,
       StandaloneAllLifecycleRollbackOnPersistFailureKeepsServingState) {
  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, 0);
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;

  const std::string index_name = "idx_registry_standalone_all_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native", options));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 10, {1.0F, 1.0F}));
  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));

  size_t rebuilt_count = 99;
  store_.fail_save_manifest_once = true;
  EXPECT_FALSE(vector_index_registry::rebuild_all_indexes(&rebuilt_count));
  EXPECT_EQ(0U, rebuilt_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  size_t recovered_count = 99;
  store_.fail_save_manifest_once = true;
  EXPECT_FALSE(vector_index_registry::recover_all_indexes(&recovered_count));
  EXPECT_EQ(0U, recovered_count);
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       StandaloneConsistencyModeRejectsMappedIndexesAndRollsBack) {
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      "dbstandalone.t1.v", 2, "euclidean", "memory", "native", "dbstandalone",
      "t1", "v", "id"));
  EXPECT_FALSE(vector_index_registry::set_index_consistency_mode(
      "dbstandalone.t1.v", vector_index::index_consistency_mode::kStandalone));

  vector_index_registry::create_index_options standalone_options;
  standalone_options.consistency_mode_specified = true;
  standalone_options.consistency_mode =
      vector_index::index_consistency_mode::kStandalone;
  EXPECT_FALSE(vector_index_registry::create_mapped_index(
      "dbstandalone.t1.v2", 2, "euclidean", "memory", "native", "dbstandalone",
      "t1", "v2", "id", standalone_options));

  const std::string index_name = "idx_registry_standalone_mode_rollback";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  store_.fail_save_metadata = true;
  EXPECT_FALSE(vector_index_registry::set_index_consistency_mode(
      index_name, vector_index::index_consistency_mode::kStandalone));
  store_.fail_save_metadata = false;

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("transactional", info.consistency_mode);
  EXPECT_TRUE(info.truth_store_enabled);

  ASSERT_TRUE(vector_index_registry::drop_index("dbstandalone.t1.v"));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
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
  two_thread_start_barrier start_barrier;
  std::thread searcher([&]() {
    if (!start_barrier.arrive_and_wait()) {
      failed.store(true);
      return;
    }
    for (int i = 0; i < 200 && !failed.load(); ++i) {
      std::vector<vector_index::search_result> result;
      if (!vector_index_registry::search("idx_registry_concurrent_search",
                                         {1.0F, 1.0F}, 1, &result) ||
          result.size() != 1 || result[0].doc_id != 1) {
        failed.store(true);
      }
    }
  });
  std::thread rebuilder([&]() {
    if (!start_barrier.arrive_and_wait()) {
      failed.store(true);
      return;
    }
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
  ASSERT_TRUE(
      vector_index_registry::drop_index("idx_registry_concurrent_search"));
  ASSERT_TRUE(
      vector_index_registry::drop_index("idx_registry_concurrent_rebuild"));
}

TEST_F(VectorIndexRegistryTest,
       SearchAndConfigurationPublicationUseOneRuntimeSnapshot) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib provider is not compiled in";
  }

  const std::string index_name = "idx_registry_concurrent_config";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "hnsw"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &before));

  std::atomic<bool> failed{false};
  two_thread_start_barrier start_barrier;
  std::thread searcher([&]() {
    if (!start_barrier.arrive_and_wait()) {
      failed.store(true);
      return;
    }
    for (int i = 0; i < 200 && !failed.load(); ++i) {
      std::vector<vector_index::search_result> result;
      std::vector<std::vector<vector_index::search_result>> batch_result;
      if (!vector_index_registry::search(index_name, {1.0F, 1.0F}, 1,
                                         &result) ||
          result.size() != 1 || result[0].doc_id != 1) {
        failed.store(true);
        continue;
      }
      if (!vector_index_registry::search_batch_for_thd_txn(
              nullptr, 0, index_name, {{1.0F, 1.0F}, {1.0F, 1.0F}}, 1,
              &batch_result) ||
          batch_result.size() != 2 || batch_result[0].size() != 1 ||
          batch_result[1].size() != 1 || batch_result[0][0].doc_id != 1 ||
          batch_result[1][0].doc_id != 1) {
        failed.store(true);
      }
    }
  });
  std::thread tuner([&]() {
    if (!start_barrier.arrive_and_wait()) {
      failed.store(true);
      return;
    }
    for (int i = 0; i < 20 && !failed.load(); ++i) {
      const uint32_t search_ef = (i % 2 == 0) ? 32 : 64;
      if (!vector_index_registry::set_search_ef(index_name, search_ef)) {
        failed.store(true);
      }
    }
  });
  searcher.join();
  tuner.join();

  EXPECT_FALSE(failed.load());
  vector_index_registry::index_info after;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &after));
  EXPECT_EQ(before.index_identity, after.index_identity);
  EXPECT_EQ(before.truth_generation, after.truth_generation);
  EXPECT_EQ(before.config_generation + 20, after.config_generation);
  EXPECT_EQ(after.truth_generation, after.runtime_generation);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       StandaloneCompactionKeepsSameIndexSearchAvailable) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib provider is not compiled in";
  }
  UlongGuard pipeline_guard(
      &opt_vector_build_pipeline_mode,
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented));
  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, 0);
  UlonglongGuard segment_rows_guard(&opt_vector_build_segment_max_rows, 1);
  vector_index_registry::create_index_options options;
  options.consistency_mode_specified = true;
  options.consistency_mode = vector_index::index_consistency_mode::kStandalone;

  const std::string index_name = "idx_registry_compaction_search";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "hnsw", options));
  for (uint64_t doc_id = 1; doc_id <= 8; ++doc_id) {
    ASSERT_TRUE(vector_index_registry::upsert(
        index_name, doc_id, {static_cast<float>(doc_id), 0.0F}));
  }
  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));

  opt_vector_build_segment_max_rows = 4;
  std::promise<void> build_entered_promise;
  std::promise<void> release_build_promise;
  std::shared_future<void> release_build =
      release_build_promise.get_future().share();
  vector_index_registry::detail::set_standalone_rebuild_build_hook_for_testing(
      [&build_entered_promise, release_build]() {
        build_entered_promise.set_value();
        release_build.wait();
      });

  bool rebuild_ok = false;
  std::thread rebuilder(
      [&]() { rebuild_ok = vector_index_registry::rebuild_index(index_name); });
  const bool build_entered =
      build_entered_promise.get_future().wait_for(std::chrono::seconds(5)) ==
      std::future_status::ready;

  constexpr size_t kSearchWorkerCount = 8;
  constexpr size_t kSearchIterations = 32;
  std::promise<void> start_search_promise;
  const std::shared_future<void> start_search =
      start_search_promise.get_future().share();
  std::vector<std::future<bool>> search_futures;
  search_futures.reserve(kSearchWorkerCount);
  for (size_t worker = 0; worker < kSearchWorkerCount; ++worker) {
    search_futures.emplace_back(
        std::async(std::launch::async, [&, start_search]() {
          start_search.wait();
          for (size_t iteration = 0; iteration < kSearchIterations;
               ++iteration) {
            const uint64_t expected_doc_id =
                ((worker + iteration) % 2 == 0) ? 1 : 8;
            std::vector<vector_index::search_result> result;
            if (!vector_index_registry::search(
                    index_name, {static_cast<float>(expected_doc_id), 0.0F}, 1,
                    &result) ||
                result.size() != 1 || result[0].doc_id != expected_doc_id) {
              return false;
            }
          }
          return true;
        }));
  }
  start_search_promise.set_value();

  const auto search_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  bool all_searches_ready = true;
  for (std::future<bool> &search_future : search_futures) {
    if (search_future.wait_until(search_deadline) !=
        std::future_status::ready) {
      all_searches_ready = false;
    }
  }

  release_build_promise.set_value();
  rebuilder.join();
  bool all_searches_ok = true;
  for (std::future<bool> &search_future : search_futures) {
    all_searches_ok = search_future.get() && all_searches_ok;
  }
  vector_index_registry::detail::
      reset_standalone_rebuild_build_hook_for_testing();

  EXPECT_TRUE(build_entered);
  EXPECT_TRUE(all_searches_ready);
  EXPECT_TRUE(all_searches_ok);
  EXPECT_TRUE(rebuild_ok);
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
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
  EXPECT_EQ(before.config_generation, after.config_generation);
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

  EXPECT_EQ(0U, store_.save_metadata_calls);
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
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kFaiss)) {
    GTEST_SKIP() << "Faiss provider is not compiled in";
  }
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
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kFaiss)) {
    GTEST_SKIP() << "Faiss provider is not compiled in";
  }
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
  if (!hnswlib_tuning_supported() ||
      !vector_index::backend_provider_supported(
          vector_index::backend_provider::kFaiss) ||
      !vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "cross-provider tuning rollback requires all vector "
                    "libraries";
  }
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
  EXPECT_FALSE(
      vector_index_registry::set_faiss_ivf_pq_params(faiss_name, 64, 8, 0, 8));
  ASSERT_TRUE(vector_index_registry::get_index_info(faiss_name, &after));
  EXPECT_EQ(before.faiss_nlist, after.faiss_nlist);
  EXPECT_EQ(before.faiss_nprobe, after.faiss_nprobe);
  EXPECT_EQ(before.faiss_pq_m, after.faiss_pq_m);
  EXPECT_EQ(before.faiss_pq_bits, after.faiss_pq_bits);
  EXPECT_TRUE(vector_index_registry::set_diskann_build_mode(
      faiss_name, vector_index::diskann_build_mode::kAuto));
  ASSERT_TRUE(vector_index_registry::get_index_info(faiss_name, &after));
  EXPECT_EQ(vector_index::diskann_build_mode::kAuto,
            after.diskann_build_mode_value);
  EXPECT_FALSE(after.diskann_build_mode_specified);
  EXPECT_FALSE(vector_index_registry::set_diskann_build_mode(
      faiss_name, vector_index::diskann_build_mode::kSerial));
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
  EXPECT_EQ(before.diskann_search_complexity, after.diskann_search_complexity);
  EXPECT_FALSE(
      vector_index_registry::set_diskann_search_beamwidth(diskann_name, 0));
  ASSERT_TRUE(vector_index_registry::get_index_info(diskann_name, &after));
  EXPECT_EQ(before.diskann_search_beamwidth, after.diskann_search_beamwidth);
  EXPECT_TRUE(
      vector_index_registry::set_diskann_pq_code_budget_size(diskann_name, 0));
  ASSERT_TRUE(vector_index_registry::get_index_info(diskann_name, &after));
  EXPECT_EQ(0U, after.diskann_pq_code_budget_size);
  ASSERT_TRUE(vector_index_registry::drop_index(diskann_name));
}

TEST_F(VectorIndexRegistryTest,
       ConfigChangeFailStopsWhenRollbackManifestCannotBeSaved) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib provider is not compiled in";
  }

  const std::string hnsw_name =
      "idx_registry_hnsw_runtime_rollback_persist_fail";
  ASSERT_TRUE(vector_index_registry::create_index(hnsw_name, 2, "cosine",
                                                  "memory", "hnswlib"));
  vector_index_registry::index_info after;

  store_.transactional = false;
  store_.ResetPersistCountersForTesting();
  store_.fail_save_manifest = true;
  EXPECT_FALSE(vector_index_registry::set_search_ef(hnsw_name, 123));
  store_.fail_save_manifest = false;
  EXPECT_EQ(2U, store_.save_metadata_calls);
  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("index_config_metadata_restore_failed",
            vector_index_registry::registry_failure_reason());
  EXPECT_FALSE(vector_index_registry::get_index_info(hnsw_name, &after));
}

TEST_F(VectorIndexRegistryTest,
       ConfigChangeFailStopsWhenRuntimeRollbackCannotRestoreConfig) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib provider is not compiled in";
  }

  const std::string hnsw_name =
      "idx_registry_hnsw_runtime_rollback_restore_fail";
  ASSERT_TRUE(vector_index_registry::create_index(hnsw_name, 2, "cosine",
                                                  "memory", "hnswlib"));

  store_.fail_save_metadata = true;
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
                             "+d,vector_service_fail_restore_index_config");
    EXPECT_FALSE(vector_index_registry::set_search_ef(hnsw_name, 123));
  }
  store_.fail_save_metadata = false;
  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("index_config_runtime_restore_failed",
            vector_index_registry::registry_failure_reason());
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
  EXPECT_FALSE(
      vector_index_registry::set_diskann_build_params(missing, 48, 96, 4));
  EXPECT_FALSE(vector_index_registry::set_diskann_build_threads(missing, 4));
  EXPECT_FALSE(
      vector_index_registry::set_diskann_search_complexity(missing, 96));
  EXPECT_FALSE(vector_index_registry::set_diskann_search_beamwidth(missing, 8));
  EXPECT_FALSE(vector_index_registry::set_diskann_build_mode(
      missing, vector_index::diskann_build_mode::kOffline));
  EXPECT_FALSE(
      vector_index_registry::set_diskann_pq_code_budget_size(missing, 4096));
  EXPECT_FALSE(vector_index_registry::set_index_consistency_mode(
      missing, vector_index::index_consistency_mode::kStandalone));
}

TEST_F(VectorIndexRegistryTest, SetDiskAnnBuildParamsRollbackOnPersistFailure) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "DiskANN provider is not compiled in";
  }
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
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "DiskANN provider is not compiled in";
  }
  const std::string index_name = "idx_registry_diskann_search_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "cosine",
                                                  "external", "diskann"));

  vector_index_registry::index_info before;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &before));

  store_.fail_save_metadata = true;
  EXPECT_FALSE(
      vector_index_registry::set_diskann_search_complexity(index_name, 180));
  EXPECT_FALSE(
      vector_index_registry::set_diskann_search_beamwidth(index_name, 20));
  EXPECT_FALSE(vector_index_registry::set_diskann_pq_code_budget_size(
      index_name, 1048576));
  store_.fail_save_metadata = false;

  vector_index_registry::index_info after;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &after));
  EXPECT_EQ(before.diskann_search_complexity, after.diskann_search_complexity);
  EXPECT_EQ(before.diskann_search_beamwidth, after.diskann_search_beamwidth);
  EXPECT_EQ(before.diskann_pq_code_budget_size,
            after.diskann_pq_code_budget_size);
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
  EXPECT_FALSE(
      vector_index_registry::set_diskann_search_beamwidth(index_name, 8));
  EXPECT_FALSE(vector_index_registry::set_diskann_build_mode(
      index_name, vector_index::diskann_build_mode::kOffline));
  EXPECT_FALSE(
      vector_index_registry::set_diskann_pq_code_budget_size(index_name, 4096));
  EXPECT_FALSE(vector_index_registry::set_index_consistency_mode(
      index_name, vector_index::index_consistency_mode::kStandalone));
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

  store_.ResetPersistCountersForTesting();
  ASSERT_TRUE(vector_index_registry::rebuild_index(index_name));
  expect_metadata_only_persist(store_);

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
  ASSERT_TRUE(vector_index_registry::search(rebuild_index_name, {1.0F, 1.0F}, 1,
                                            &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  const std::string replace_index_name = "idx_registry_replace_budget";
  ASSERT_TRUE(vector_index_registry::create_index(
      replace_index_name, 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(vector_index_registry::replace_committed_entries(
      replace_index_name, {{20, {2.0F, 2.0F}}}));
  EXPECT_EQ(0U, vector_index_registry::committed_vector_memory_bytes());

  result.clear();
  ASSERT_TRUE(vector_index_registry::search(replace_index_name, {2.0F, 2.0F}, 1,
                                            &result));
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
  ASSERT_TRUE(
      vector_index_registry::replace_committed_entries_preserve_lifecycle(
          index_name, replacement));

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("bulk_loading", info.lifecycle_state);
  EXPECT_EQ(2U, info.entry_count);
  EXPECT_EQ(2U, info.committed_entry_count);

  EXPECT_FALSE(
      vector_index_registry::replace_committed_entries_preserve_lifecycle(
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

TEST_F(VectorIndexRegistryTest,
       BeginBulkLoadThenBulkBuildRestoresServingState) {
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

  if (!hnswlib_tuning_supported() ||
      !vector_index::backend_provider_supported(backend_provider::kFaiss) ||
      !vector_index::backend_provider_supported(backend_provider::kDiskAnn) ||
      !vector_index::backend_provider_supported(backend_provider::kNative)) {
    GTEST_SKIP() << "backend tuning wrapper coverage requires all vector "
                    "providers";
  }

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

  config.dimension = 3;
  config.faiss_pq_m = 2;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_faiss_pq_dimension", config));

  config.dimension = 4;
  config.faiss_pq_m = 1;
  config.faiss_pq_bits = vector_index::k_max_faiss_pq_bits + 1;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_faiss_pq_bits", config));

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
      make_backend_config(backend_mode::kMemory, backend_provider::kNative);
  config.diskann_build_mode_specified = true;
  config.diskann_build_mode_value = vector_index::diskann_build_mode::kAuto;
  EXPECT_TRUE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_diskann_mode_auto", config));
  config.diskann_build_mode_value = vector_index::diskann_build_mode::kOffline;
  EXPECT_FALSE(vector_index_registry::build_backend_from_config_for_testing(
      "idx_config_native_diskann_mode_offline", config));

  config =
      make_backend_config(backend_mode::kExternal, backend_provider::kDiskAnn);
  config.diskann_max_degree = 32;
  config.diskann_build_complexity = 64;
  config.diskann_build_threads = 2;
  config.diskann_build_mode_specified = true;
  config.diskann_build_mode_value = vector_index::diskann_build_mode::kOffline;
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

  store_.fail_save_manifest_once = true;
  EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  EXPECT_TRUE(vector_index_registry::has_prepared_xid(xid));
  EXPECT_EQ(vector_index_registry::registry_health_state::kReady,
            vector_index_registry::registry_health());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {18.0F, 18.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_EQ(XA_OK, vector_index_registry::rollback_prepared_xid(xid));
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidFailsStopWhenPersistedRollbackFails) {
  store_.transactional = false;
  const std::string index_name = "idx_registry_prepared_persist_rollback_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));

  XID xid = make_test_xid(622, "persisted-rollback-fail");
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      622, 1, index_name, 19, {19.0F, 19.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(622, 1));
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(622, xid));

  store_.fail_save_manifest = true;
  EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  store_.fail_save_manifest = false;

  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("persisted_commit_artifacts_restore_failed",
            vector_index_registry::registry_failure_reason());
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));
  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  ASSERT_TRUE(store_.load_prepared(&prepared_rows));
  ASSERT_EQ(1U, prepared_rows.size());
  EXPECT_EQ(index_name, prepared_rows.front().index_name);
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
       ChangeLogReplayValidatesEveryIdentityFieldAndSkipsStaleRows) {
  const std::string index_name = "idx_registry_changelog_fields";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));

  vector_index_metadata_store::change_log_row base;
  base.sequence = 1;
  base.txn_id = 1;
  base.op = vector_index_metadata_store::change_op::kUpsert;
  base.index_name = index_name;
  base.doc_id = 7;
  base.vector = {7.0F, 1.0F};
  base.index_identity = info.index_identity;
  base.publication_id = 1;
  base.truth_generation = 1;

  const auto apply_rows = [&](const auto &rows) {
    std::lock_guard<std::shared_mutex> guard(
        vector_index_registry::detail::g_registry_mutex);
    return vector_index_registry::detail::apply_change_log_rows_locked(rows);
  };
  const auto apply_one = [&](const auto &row) {
    return apply_rows(
        std::vector<vector_index_metadata_store::change_log_row>{row});
  };
  const auto expect_invalid = [&](const auto &mutate) {
    auto row = base;
    mutate(&row);
    EXPECT_FALSE(apply_one(row));
  };
  expect_invalid([](auto *row) { row->sequence = 0; });
  expect_invalid([](auto *row) { row->index_name.clear(); });
  expect_invalid([](auto *row) { row->index_identity = 0; });
  expect_invalid([](auto *row) { row->publication_id = 0; });
  expect_invalid([](auto *row) { row->truth_generation = 0; });
  expect_invalid([](auto *row) {
    row->op = static_cast<vector_index_metadata_store::change_op>(99);
  });

  auto unknown_index = base;
  unknown_index.index_name = "idx_registry_changelog_unknown";
  EXPECT_TRUE(apply_one(unknown_index));
  auto stale_identity = base;
  ++stale_identity.index_identity;
  EXPECT_TRUE(apply_one(stale_identity));

  EXPECT_TRUE(apply_one(base));
  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {7.0F, 1.0F}, 1, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(7U, results.front().doc_id);

  auto erase = base;
  erase.sequence = 2;
  erase.truth_generation = 2;
  erase.op = vector_index_metadata_store::change_op::kErase;
  erase.vector.clear();
  EXPECT_TRUE(apply_one(erase));
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {7.0F, 1.0F}, 1, &results));
  EXPECT_TRUE(results.empty());
  ASSERT_TRUE(vector_index_registry::drop_index(index_name));
}

TEST_F(VectorIndexRegistryTest,
       InvalidChangeLogReplayFailStopsWithoutTableBinding) {
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
  EXPECT_FALSE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  EXPECT_EQ(failures_before + 1, vector_status::change_log_load_failures());
  EXPECT_EQ(replay_failures_before + 1,
            vector_status::change_log_replay_failures());
  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("changelog_replay_failed",
            vector_index_registry::registry_failure_reason());
  ASSERT_EQ(1U, store_.quarantine_records.size());
  EXPECT_EQ("changelog", store_.quarantine_records[0].artifact_name);
}

TEST_F(VectorIndexRegistryTest,
       ChangeLogReplayOrdersRowsBySequenceIndexDocAndOp) {
  const std::string index_a = "idx_registry_changelog_order_a";
  const std::string index_b = "idx_registry_changelog_order_b";
  ASSERT_TRUE(vector_index_registry::create_index(index_a, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::create_index(index_b, 2, "euclidean",
                                                  "memory", "native"));

  vector_index_registry::index_info info_a;
  vector_index_registry::index_info info_b;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_a, &info_a));
  ASSERT_TRUE(vector_index_registry::get_index_info(index_b, &info_b));

  store_.ClearCommittedRowsForTesting();

  std::vector<vector_index_metadata_store::change_log_row> rows;
  rows.push_back({2,
                  11,
                  vector_index_metadata_store::change_op::kUpsert,
                  index_b,
                  2,
                  {2.0F, 2.0F},
                  info_b.index_identity,
                  1,
                  1});
  rows.push_back({1,
                  11,
                  vector_index_metadata_store::change_op::kUpsert,
                  index_b,
                  1,
                  {1.0F, 1.0F},
                  info_b.index_identity,
                  1,
                  1});
  rows.push_back({1,
                  11,
                  vector_index_metadata_store::change_op::kUpsert,
                  index_a,
                  2,
                  {2.0F, 0.0F},
                  info_a.index_identity,
                  1,
                  1});
  rows.push_back(
      {1, 11, vector_index_metadata_store::change_op::kErase, index_a, 1, {},
       info_a.index_identity, 1, 1});
  rows.push_back({1,
                  11,
                  vector_index_metadata_store::change_op::kUpsert,
                  index_a,
                  1,
                  {1.0F, 0.0F},
                  info_a.index_identity,
                  1,
                  1});
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
       InvalidChangeLogOpFailStopsWithoutTableBinding) {
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
  EXPECT_FALSE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 2, &result));
  EXPECT_EQ(failures_before + 1, vector_status::change_log_load_failures());
  EXPECT_EQ(replay_failures_before + 1,
            vector_status::change_log_replay_failures());
  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("changelog_replay_failed",
            vector_index_registry::registry_failure_reason());
}

TEST_F(VectorIndexRegistryTest,
       MetadataLoadFailureFailStopsAfterQuarantineCopy) {
  store_.fail_load_metadata = true;
  const uint64_t metadata_load_failures_before =
      vector_status::metadata_load_failures();

  std::vector<std::string> index_names;
  EXPECT_FALSE(vector_index_registry::list_indexes(&index_names));
  EXPECT_TRUE(index_names.empty());
  EXPECT_EQ(metadata_load_failures_before + 1,
            vector_status::metadata_load_failures());
  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("metadata_load_failed",
            vector_index_registry::registry_failure_reason());
  ASSERT_EQ(1U, store_.quarantine_records.size());
  EXPECT_EQ("metadata", store_.quarantine_records[0].artifact_name);
  EXPECT_EQ(vector_index_truth_store::quarantine_state::kCopied,
            store_.quarantine_records[0].state);

  store_.fail_load_metadata = false;
  EXPECT_FALSE(vector_index_registry::list_indexes(&index_names));
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
       CommittedLoadFailureRequiresMappedTruthRecovery) {
  const std::string index_name = "db_recovery.t_recovery.v";
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      index_name, 2, "euclidean", "memory", "native", "db_recovery",
      "t_recovery", "v", "id"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  const uint64_t committed_failures_before =
      vector_status::committed_load_failures();
  store_.fail_load_committed = true;
  vector_index_registry::reset_for_testing();

  std::vector<std::string> index_names;
  ASSERT_TRUE(vector_index_registry::list_indexes(&index_names));
  ASSERT_EQ(1U, index_names.size());
  EXPECT_EQ(index_name, index_names[0]);
  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("recovery_required", info.lifecycle_state);
  EXPECT_EQ(vector_index_registry::registry_health_state::kRecoveryRequired,
            vector_index_registry::registry_health());
  EXPECT_EQ("truth_projection_recovery_required",
            vector_index_registry::registry_failure_reason());

  std::vector<vector_index::search_result> result;
  EXPECT_FALSE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  EXPECT_FALSE(vector_index_registry::upsert(index_name, 2, {2.0F, 2.0F}));
  EXPECT_EQ(committed_failures_before + 1,
            vector_status::committed_load_failures());
  ASSERT_EQ(1U, store_.quarantine_records.size());
  EXPECT_EQ("committed", store_.quarantine_records[0].artifact_name);
  EXPECT_EQ(vector_index_truth_store::quarantine_state::kCopied,
            store_.quarantine_records[0].state);

  vector_index::index_service::committed_state recovered_state;
  recovered_state[index_name][7] = {7.0F, 7.0F};
  ASSERT_TRUE(vector_index_registry::recover_truth_projection_for_testing(
      recovered_state));
  EXPECT_EQ(vector_index_registry::registry_health_state::kReady,
            vector_index_registry::registry_health());
  EXPECT_EQ(vector_index_truth_store::quarantine_state::kComplete,
            store_.quarantine_records[0].state);
  ASSERT_TRUE(
      vector_index_registry::search(index_name, {7.0F, 7.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);
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
       ManifestLoadFailureFailStopsAfterQuarantineCopy) {
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
  EXPECT_FALSE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  EXPECT_EQ(metadata_failures_before + 1,
            vector_status::metadata_load_failures());
  EXPECT_EQ(manifest_failures_before + 1,
            vector_status::manifest_load_failures());
  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("manifest_load_failed",
            vector_index_registry::registry_failure_reason());
  ASSERT_EQ(1U, store_.quarantine_records.size());
  EXPECT_EQ("manifest", store_.quarantine_records[0].artifact_name);
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
       ChangeLogLoadFailureRequiresMappedTruthRecovery) {
  const std::string index_name = "db_recovery.t_recovery.v";
  ASSERT_TRUE(vector_index_registry::create_mapped_index(
      index_name, 2, "euclidean", "memory", "native", "db_recovery",
      "t_recovery", "v", "id"));
  ASSERT_TRUE(vector_index_registry::upsert(index_name, 1, {1.0F, 1.0F}));

  const uint64_t metadata_failures_before =
      vector_status::metadata_load_failures();
  const uint64_t changelog_failures_before =
      vector_status::change_log_load_failures();
  store_.fail_load_change_log = true;
  vector_index_registry::reset_for_testing();

  vector_index_registry::index_info info;
  ASSERT_TRUE(vector_index_registry::get_index_info(index_name, &info));
  EXPECT_EQ("recovery_required", info.lifecycle_state);
  EXPECT_EQ(vector_index_registry::registry_health_state::kRecoveryRequired,
            vector_index_registry::registry_health());

  std::vector<vector_index::search_result> result;
  EXPECT_FALSE(
      vector_index_registry::search(index_name, {1.0F, 1.0F}, 1, &result));
  EXPECT_EQ(metadata_failures_before + 1,
            vector_status::metadata_load_failures());
  EXPECT_EQ(changelog_failures_before + 1,
            vector_status::change_log_load_failures());
  ASSERT_EQ(1U, store_.quarantine_records.size());
  EXPECT_EQ("changelog", store_.quarantine_records[0].artifact_name);

  vector_index::index_service::committed_state recovered_state;
  recovered_state[index_name][9] = {9.0F, 9.0F};
  store_.fail_save_committed = true;
  EXPECT_FALSE(vector_index_registry::recover_truth_projection_for_testing(
      recovered_state));
  EXPECT_EQ(vector_index_registry::registry_health_state::kRecoveryRequired,
            vector_index_registry::registry_health());
  EXPECT_EQ(vector_index_truth_store::quarantine_state::kSourceUpdateFailed,
            store_.quarantine_records[0].state);

  store_.fail_save_committed = false;
  ASSERT_TRUE(vector_index_registry::recover_truth_projection_for_testing(
      recovered_state));
  EXPECT_EQ(vector_index_registry::registry_health_state::kReady,
            vector_index_registry::registry_health());
  EXPECT_EQ(vector_index_truth_store::quarantine_state::kComplete,
            store_.quarantine_records[0].state);
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
       PreparedLoadFailureFailStopsAfterQuarantineCopy) {
  store_.fail_load_prepared = true;

  XA_recover_txn recovered[2]{};
  XID xid = make_test_xid(124, "prepared-load-quarantine");
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
  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("prepared_load_failed",
            vector_index_registry::registry_failure_reason());
  ASSERT_EQ(1U, store_.quarantine_records.size());
  EXPECT_EQ("prepared", store_.quarantine_records[0].artifact_name);
}

TEST_F(VectorIndexRegistryTest,
       ChangeLogReplayFailureReturnsFalseWhenQuarantineFails) {
  vector_index_metadata_store::metadata_row row;
  row.index_name = "idx_registry_replay_quarantine_fail";
  row.dimension = 2;
  row.metric = vector_index::metric_type::kEuclidean;
  row.mode = vector_index::backend_mode::kMemory;
  row.provider = vector_index::backend_provider::kNative;
  row.owner_schema = "test";
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
  EXPECT_TRUE(vector_index_identity::parse_index_name("db1.t1.v1", nullptr));
  EXPECT_FALSE(vector_index_identity::parse_index_name("db1", &identity));
  EXPECT_FALSE(vector_index_identity::parse_index_name("db1.t1", &identity));
  EXPECT_FALSE(vector_index_identity::parse_index_name(".t1.v1", &identity));
  EXPECT_FALSE(vector_index_identity::parse_index_name("db1..v1", &identity));
  EXPECT_FALSE(vector_index_identity::parse_index_name("db1.t1.", &identity));
  EXPECT_FALSE(
      vector_index_identity::parse_index_name("db1.t1.v1.extra", &identity));
  EXPECT_TRUE(vector_index_identity::matches_schema("db1.t1.v1", "db1"));
  EXPECT_FALSE(vector_index_identity::matches_schema("xdb1.t1.v1", "db1"));
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
  EXPECT_FALSE(vector_index_registry::savepoint_txn(9999, "sp_missing"));
  EXPECT_FALSE(
      vector_index_registry::release_savepoint_txn(9999, "sp_missing"));

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
  EXPECT_EQ("id", doc_id_column_name);

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
    row.owner_schema = "test";
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
    row.owner_schema = "test";
    row.lifecycle_state = "ready";
    row.lifecycle_version = 1;
    row.hnsw_m = 16;
    row.hnsw_ef_construction = 32;
    EXPECT_FALSE(vector_index_registry::restore_runtime_state_for_testing(
        {row}, {}, {}, {}));
  }

  {
    vector_index_metadata_store::metadata_row row;
    row.index_name = "idx_bad_faiss_pq_metadata";
    row.dimension = 6;
    row.metric = vector_index::metric_type::kEuclidean;
    row.mode = vector_index::backend_mode::kExternal;
    row.provider = vector_index::backend_provider::kFaiss;
    row.owner_schema = "test";
    row.lifecycle_state = "ready";
    row.lifecycle_version = 1;
    row.faiss_nlist = 8;
    row.faiss_nprobe = 4;
    row.faiss_pq_m = 4;
    row.faiss_pq_bits = 8;
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
  row.owner_schema = "dbrestore";
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
  EXPECT_EQ("dbrestore", info.owner_schema);
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

TEST_F(VectorIndexRegistryTest,
       CommitPreparedXidFailsStopWhenRuntimeRollbackFails) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  const std::string index_name = "idx_registry_prepared_rollback_fail";
  ASSERT_TRUE(vector_index_registry::create_index(index_name, 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(vector_index_registry::stage_upsert_for_thd_txn(
      621, 1, index_name, 12, {12.0F, 12.0F}));
  ASSERT_TRUE(vector_index_registry::commit_stmt_for_thd_txn(621, 1));

  XID xid = make_test_xid(621, "prepared-rollback");
  ASSERT_TRUE(vector_index_registry::prepare_thd_txn(621, xid));
  {
    vector_gunit::ScopedValueGuard<uint64_t> sequence_guard(
        &vector_index_registry::detail::g_next_change_log_sequence,
        std::numeric_limits<uint64_t>::max());
    VECTOR_SCOPED_DEBUG_FLAG(debug,
                             "+d,vector_registry_fail_restore_runtime_state");
    EXPECT_EQ(XAER_RMERR, vector_index_registry::commit_prepared_xid(xid));
  }

  EXPECT_EQ(vector_index_registry::registry_health_state::kFailed,
            vector_index_registry::registry_health());
  EXPECT_EQ("prepared_runtime_state_restore_failed",
            vector_index_registry::registry_failure_reason());
  EXPECT_FALSE(vector_index_registry::has_prepared_xid(xid));
  std::vector<vector_index_metadata_store::prepared_change_row> prepared_rows;
  ASSERT_TRUE(store_.load_prepared(&prepared_rows));
  ASSERT_EQ(1U, prepared_rows.size());
  EXPECT_EQ(index_name, prepared_rows.front().index_name);
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
      "dbpurge.t1.v1", 2, "euclidean", "memory", "native", "dbpurge"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbpurge.t2.v2", 2, "euclidean", "memory", "native", "dbpurge"));
  ASSERT_TRUE(vector_index_registry::create_index(
      "dbkeep.t1.v", 2, "euclidean", "memory", "native", "dbkeep"));
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
  EXPECT_TRUE(
      vector_index_registry::detail::mapped_search_candidate_limit_is_hard(
          vector_index::k_max_search_top_k, 1));
  EXPECT_FALSE(
      vector_index_registry::detail::mapped_search_candidate_limit_is_hard(10,
                                                                           5));
  EXPECT_EQ(0U, vector_index_registry::detail::mapped_search_candidate_limit(
                    0, 100, 20));
  EXPECT_EQ(15U, vector_index_registry::detail::mapped_search_candidate_limit(
                     10, 12, 3));
  EXPECT_EQ(vector_index::k_max_search_top_k,
            vector_index_registry::detail::mapped_search_candidate_limit(
                10, vector_index::k_max_search_top_k + 1, 100));

  EXPECT_EQ(
      15U, vector_index_registry::detail::mapped_search_initial_candidate_top_k(
               10, 12, 3));
  EXPECT_EQ(
      32U, vector_index_registry::detail::mapped_search_initial_candidate_top_k(
               1, 100, 0));
  EXPECT_EQ(64U,
            vector_index_registry::detail::mapped_search_next_candidate_top_k(
                32, 1, 100));
  EXPECT_EQ(100U,
            vector_index_registry::detail::mapped_search_next_candidate_top_k(
                96, 10, 100));
  EXPECT_EQ(100U,
            vector_index_registry::detail::mapped_search_next_candidate_top_k(
                100, 10, 100));
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

  EXPECT_FALSE(
      vector_index_registry::search("missing_batch", {1.0F, 1.0F}, 1, nullptr));
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
      nullptr, 0, "missing_batch", {{1.0F, 1.0F}, {2.0F, 2.0F}}, 1, &batches));

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
