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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_service.h"
#include "sql/vector/vector_index_truth_store.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_index_service_unittest {

namespace {

bool has_prefix(const std::string &text, const std::string &prefix) {
  return text.size() >= prefix.size() &&
         text.compare(0, prefix.size(), prefix) == 0;
}

bool has_suffix(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string find_faiss_snapshot_file(const std::string &root) {
  std::error_code ec;
  if (!std::filesystem::exists(root, ec) || ec) return "";

  for (const auto &entry : std::filesystem::recursive_directory_iterator(
           root, std::filesystem::directory_options::skip_permission_denied, ec)) {
    if (ec) return "";
    if (!entry.is_regular_file(ec)) {
      if (ec) return "";
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (!has_prefix(filename, "faiss_external.snapshot.") ||
        !has_suffix(filename, ".v1")) {
      continue;
    }
    return entry.path().string();
  }
  return "";
}

std::string find_manifest_file(const std::string &root,
                               const std::string &manifest_filename) {
  std::error_code ec;
  if (!std::filesystem::exists(root, ec) || ec) return "";

  for (const auto &entry : std::filesystem::recursive_directory_iterator(
           root, std::filesystem::directory_options::skip_permission_denied,
           ec)) {
    if (ec) return "";
    if (!entry.is_regular_file(ec)) {
      if (ec) return "";
      continue;
    }
    if (entry.path().filename().string() == manifest_filename) {
      return entry.path().string();
    }
  }
  return "";
}

std::string find_diskann_manifest_file(const std::string &root) {
  return find_manifest_file(root, "diskann_external.manifest.v1");
}

std::string diskann_store_path_for_manifest(const std::string &manifest_path) {
  if (manifest_path.empty()) return "";
  return std::filesystem::path(manifest_path).parent_path().string() +
         "/diskann_external.store";
}

bool hnswlib_tuning_supported() {
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  return backend.set_search_ef(64) && backend.set_hnsw_build_params(16, 200);
}

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

class committed_rows_truth_store final : public vector_index_truth_store::truth_store {
 public:
  explicit committed_rows_truth_store(
      std::vector<vector_index_metadata_store::committed_row> rows)
      : m_rows(std::move(rows)) {}

  const char *backend_name() const override { return "committed_rows"; }

  bool is_transactional() const override { return true; }

  bool load_metadata(
      std::vector<vector_index_metadata_store::metadata_row> *rows) override {
    if (rows == nullptr) return false;
    rows->clear();
    return true;
  }

  bool save_metadata(
      const std::vector<vector_index_metadata_store::metadata_row> &) override {
    return true;
  }

  bool quarantine_metadata() override { return true; }

  bool load_committed(
      std::vector<vector_index_metadata_store::committed_row> *rows) override {
    if (rows == nullptr) return false;
    ++m_load_committed_calls;
    *rows = m_rows;
    return true;
  }

  using vector_index_truth_store::truth_store::for_each_committed;

  bool for_each_committed(
      const std::string &index_name,
      const std::function<bool(
          const vector_index_metadata_store::committed_row &row)> &visitor)
      override {
    if (index_name.empty() || !visitor) return false;
    ++m_for_each_committed_calls;
    for (const auto &row : m_rows) {
      if (row.index_name != index_name) continue;
      ++m_for_each_committed_rows;
      if (!visitor(row)) return false;
    }
    return true;
  }

  bool find_committed(const std::string &index_name, uint64_t doc_id,
                      vector_index_metadata_store::committed_row *row,
                      bool *found) override {
    if (index_name.empty() || row == nullptr || found == nullptr) return false;
    ++m_find_committed_calls;
    *row = vector_index_metadata_store::committed_row();
    *found = false;
    for (const auto &candidate : m_rows) {
      if (candidate.index_name == index_name && candidate.doc_id == doc_id) {
        *row = candidate;
        *found = true;
        return true;
      }
    }
    return true;
  }

  size_t load_committed_calls() const { return m_load_committed_calls; }
  size_t for_each_committed_calls() const { return m_for_each_committed_calls; }
  size_t for_each_committed_rows() const { return m_for_each_committed_rows; }
  size_t find_committed_calls() const { return m_find_committed_calls; }

  bool save_committed(
      const std::vector<vector_index_metadata_store::committed_row> &) override {
    return true;
  }

  bool quarantine_committed() override { return true; }

  bool load_manifest(vector_index_metadata_store::manifest_row *row) override {
    if (row == nullptr) return false;
    *row = vector_index_metadata_store::manifest_row();
    return true;
  }

  bool save_manifest(
      const vector_index_metadata_store::manifest_row &) override {
    return true;
  }

  bool quarantine_manifest() override { return true; }

  bool load_change_log(
      std::vector<vector_index_metadata_store::change_log_row> *rows) override {
    if (rows == nullptr) return false;
    rows->clear();
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
    if (rows == nullptr) return false;
    rows->clear();
    return true;
  }

  bool save_prepared(
      const std::vector<vector_index_metadata_store::prepared_change_row> &)
      override {
    return true;
  }

  bool quarantine_prepared() override { return true; }

 private:
  std::vector<vector_index_metadata_store::committed_row> m_rows;
  size_t m_load_committed_calls{0};
  size_t m_for_each_committed_calls{0};
  size_t m_for_each_committed_rows{0};
  size_t m_find_committed_calls{0};
};

class truth_store_guard {
 public:
  explicit truth_store_guard(vector_index_truth_store::truth_store *store) {
    vector_index_truth_store::set_for_testing(store);
  }

  ~truth_store_guard() { vector_index_truth_store::reset_for_testing(); }
};

class faiss_snapshot_root_guard {
 public:
  explicit faiss_snapshot_root_guard(const std::string &root) {
    vector_index::set_faiss_external_snapshot_root_for_testing(root);
  }

  ~faiss_snapshot_root_guard() {
    vector_index::reset_faiss_external_snapshot_root_for_testing();
  }
};

}  // namespace

class NonApplyingBackend final : public vector_index::backend {
 public:
  bool upsert(uint64_t doc_id [[maybe_unused]],
              const vector_index::vector_data &vector [[maybe_unused]]) override {
    return false;
  }
  bool erase(uint64_t doc_id [[maybe_unused]]) override { return false; }
  bool search(const vector_index::vector_data &query [[maybe_unused]],
              size_t top_k [[maybe_unused]],
              std::vector<vector_index::search_result> *results) const override {
    results->clear();
    return true;
  }
  size_t dimension() const override { return 2; }
  vector_index::metric_type metric() const override {
    return vector_index::metric_type::kEuclidean;
  }
  vector_index::backend_mode mode() const override {
    return vector_index::backend_mode::kMemory;
  }
  vector_index::backend_provider provider() const override {
    return vector_index::backend_provider::kNative;
  }
  bool supports_mutations() const override { return true; }
};

class NonWritableBackend final : public vector_index::backend {
 public:
  bool upsert(uint64_t doc_id [[maybe_unused]],
              const vector_index::vector_data &vector [[maybe_unused]]) override {
    return true;
  }
  bool erase(uint64_t doc_id [[maybe_unused]]) override { return true; }
  bool search(const vector_index::vector_data &query [[maybe_unused]],
              size_t top_k [[maybe_unused]],
              std::vector<vector_index::search_result> *results) const override {
    results->clear();
    return true;
  }
  size_t dimension() const override { return 2; }
  vector_index::metric_type metric() const override {
    return vector_index::metric_type::kEuclidean;
  }
  vector_index::backend_mode mode() const override {
    return vector_index::backend_mode::kExternal;
  }
  vector_index::backend_provider provider() const override {
    return vector_index::backend_provider::kDiskAnn;
  }
  bool supports_mutations() const override { return false; }
};

class IncompatibleBackend final : public vector_index::backend {
 public:
  bool upsert(uint64_t doc_id [[maybe_unused]],
              const vector_index::vector_data &vector [[maybe_unused]]) override {
    return true;
  }
  bool erase(uint64_t doc_id [[maybe_unused]]) override { return true; }
  bool search(const vector_index::vector_data &query [[maybe_unused]],
              size_t top_k [[maybe_unused]],
              std::vector<vector_index::search_result> *results) const override {
    results->clear();
    return true;
  }
  size_t dimension() const override { return 2; }
  vector_index::metric_type metric() const override {
    return vector_index::metric_type::kEuclidean;
  }
  vector_index::backend_mode mode() const override {
    return vector_index::backend_mode::kMemory;
  }
  vector_index::backend_provider provider() const override {
    return vector_index::backend_provider::kDiskAnn;
  }
  bool supports_mutations() const override { return true; }
};

TEST(VectorIndexServiceTest, RegisterAndDropIndex) {
  vector_index::index_service service;
  auto backend = std::make_unique<vector_index::memory_backend>(
      2, vector_index::metric_type::kEuclidean);

  EXPECT_TRUE(service.register_index("idx_mem", std::move(backend)));
  EXPECT_FALSE(service.register_index("idx_mem", std::make_unique<vector_index::memory_backend>(
                                                    2, vector_index::metric_type::kEuclidean)));
  EXPECT_TRUE(service.drop_index("idx_mem"));
  EXPECT_FALSE(service.drop_index("idx_mem"));
}

TEST(VectorIndexServiceTest, RenameIndexSameNameIsNoop) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_same", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(91, "idx_same", 7, {7.0F, 7.0F}));
  ASSERT_TRUE(service.commit(91));

  ASSERT_TRUE(service.rename_index("idx_same", "idx_same"));
  ASSERT_TRUE(service.search("idx_same", {7.0F, 7.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, RenameIndexRejectsMissingOrExistingTarget) {
  vector_index::index_service service;

  EXPECT_FALSE(service.rename_index("missing", "idx_other"));
  ASSERT_TRUE(service.register_index_from_strings("idx_src", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.register_index_from_strings("idx_dst", 2, "euclidean",
                                               "memory", "native"));
  EXPECT_FALSE(service.rename_index("idx_src", "idx_dst"));
}

TEST(VectorIndexServiceTest, RenameIndexMovesPendingChangesToNewName) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_old", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(92, "idx_old", 8, {8.0F, 8.0F}));
  ASSERT_TRUE(service.rename_index("idx_old", "idx_new"));
  ASSERT_TRUE(service.commit(92));

  ASSERT_TRUE(service.search("idx_new", {8.0F, 8.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(8U, result[0].doc_id);
  EXPECT_FALSE(service.search("idx_old", {8.0F, 8.0F}, 1, &result));
}

TEST(VectorIndexServiceTest, ReportsEntryStorePayloadMemoryBytes) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index_from_strings("idx_mem", 2, "euclidean",
                                                  "memory", "native"));
  EXPECT_EQ(0U, service.committed_vector_memory_bytes());
  EXPECT_EQ(0U, service.pending_vector_memory_bytes(10));
  EXPECT_EQ(0U, service.total_pending_vector_memory_bytes());

  ASSERT_TRUE(service.stage_upsert(10, "idx_mem", 1, {1.0F, 1.0F}));
  EXPECT_EQ(0U, service.committed_vector_memory_bytes());
  EXPECT_EQ(2U * sizeof(float), service.pending_vector_memory_bytes(10));
  EXPECT_EQ(2U * sizeof(float), service.total_pending_vector_memory_bytes());

  ASSERT_TRUE(service.commit(10));
  EXPECT_EQ(2U * sizeof(float), service.committed_vector_memory_bytes());
  EXPECT_EQ(0U, service.pending_vector_memory_bytes(10));
  EXPECT_EQ(0U, service.total_pending_vector_memory_bytes());
}

TEST(VectorIndexServiceTest, EntryStorePayloadMemoryTracksReplaceAndErase) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index_from_strings("idx_mem", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(11, "idx_mem", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(11));

  ASSERT_TRUE(service.stage_upsert(12, "idx_mem", 1, {2.0F, 2.0F}));
  ASSERT_TRUE(service.stage_upsert(12, "idx_mem", 2, {3.0F, 3.0F}));
  EXPECT_EQ(4U * sizeof(float), service.total_pending_vector_memory_bytes());
  ASSERT_TRUE(service.commit(12));
  EXPECT_EQ(4U * sizeof(float), service.committed_vector_memory_bytes());

  ASSERT_TRUE(service.stage_erase(13, "idx_mem", 1));
  EXPECT_EQ(0U, service.pending_vector_memory_bytes(13));
  ASSERT_TRUE(service.commit(13));
  EXPECT_EQ(2U * sizeof(float), service.committed_vector_memory_bytes());
}

TEST(VectorEntryStoreTest, IteratesByDocIdAndTracksMemory) {
  vector_index::vector_entry_store store;
  std::vector<uint64_t> doc_ids;

  ASSERT_TRUE(store.register_index("idx_store"));
  ASSERT_TRUE(store.upsert("idx_store", 9, {9.0F, 9.0F}));
  ASSERT_TRUE(store.upsert("idx_store", 1, {1.0F, 1.0F}));

  EXPECT_EQ(2U, store.entry_count());
  EXPECT_EQ(4U * sizeof(float), store.memory_bytes());
  ASSERT_TRUE(store.for_each_committed_entry(
      "idx_store", [&](uint64_t doc_id, const vector_index::vector_data &vector) {
        doc_ids.push_back(doc_id);
        EXPECT_EQ(2U, vector.size());
        return true;
      }));

  EXPECT_EQ(std::vector<uint64_t>({1, 9}), doc_ids);
  ASSERT_TRUE(store.erase("idx_store", 9));
  EXPECT_EQ(1U, store.entry_count());
  EXPECT_EQ(2U * sizeof(float), store.memory_bytes());
}

TEST(VectorEntryStoreTest, EvictsCachedEntriesButKeepsCommittedCount) {
  vector_index::vector_entry_store store;

  ASSERT_TRUE(store.register_index("idx_store"));
  ASSERT_TRUE(store.upsert("idx_store", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(store.upsert("idx_store", 2, {2.0F, 2.0F}));

  EXPECT_EQ(2U, store.entry_count("idx_store"));
  EXPECT_EQ(4U * sizeof(float), store.memory_bytes("idx_store"));
  EXPECT_TRUE(store.evict_until_under_budget(0));
  EXPECT_EQ(2U, store.entry_count("idx_store"));
  EXPECT_EQ(0U, store.memory_bytes("idx_store"));
}

TEST(VectorEntryStoreTest, EvictsLargestIndexAndHonorsUnlimitedBudget) {
  vector_index::vector_entry_store store;

  ASSERT_TRUE(store.register_index("idx_small"));
  ASSERT_TRUE(store.register_index("idx_large"));
  ASSERT_TRUE(store.upsert("idx_small", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(store.upsert("idx_large", 10, {10.0F, 10.0F}));
  ASSERT_TRUE(store.upsert("idx_large", 11, {11.0F, 11.0F}));

  EXPECT_TRUE(
      store.evict_until_under_budget(std::numeric_limits<size_t>::max()));
  EXPECT_EQ(6U * sizeof(float), store.memory_bytes());

  EXPECT_TRUE(store.evict_until_under_budget(2U * sizeof(float)));
  EXPECT_EQ(2U, store.entry_count("idx_large"));
  EXPECT_EQ(0U, store.memory_bytes("idx_large"));
  EXPECT_EQ(1U, store.entry_count("idx_small"));
  EXPECT_EQ(2U * sizeof(float), store.memory_bytes("idx_small"));

  EXPECT_TRUE(store.evict_until_under_budget(0));
  EXPECT_EQ(3U, store.entry_count());
  EXPECT_EQ(0U, store.memory_bytes());
}

TEST(VectorEntryStoreTest, EvictedIndexRequiresTruthStoreForReadThrough) {
  vector_index::vector_entry_store store;
  vector_index::committed_state state;
  vector_index::committed_entries entries;

  ASSERT_TRUE(store.register_index("idx_store"));
  ASSERT_TRUE(store.upsert("idx_store", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(store.evict_until_under_budget(0));

  EXPECT_FALSE(store.snapshot(&state));
  EXPECT_FALSE(store.snapshot_index("idx_store", &entries));
  EXPECT_FALSE(store.for_each_committed_entry(
      "idx_store", [](uint64_t, const vector_index::vector_data &) {
        return true;
      }));
  EXPECT_FALSE(store.upsert("idx_store", 2, {2.0F, 2.0F}));
  EXPECT_FALSE(store.erase("idx_store", 1));

  ASSERT_TRUE(store.rename_index("idx_store", "idx_renamed"));
  EXPECT_TRUE(store.has_index("idx_renamed"));
  EXPECT_FALSE(store.has_index("idx_store"));
  EXPECT_EQ(1U, store.entry_count("idx_renamed"));
  EXPECT_FALSE(store.snapshot_index("idx_renamed", &entries));

  ASSERT_TRUE(store.clear_index("idx_renamed"));
  EXPECT_EQ(0U, store.entry_count("idx_renamed"));
  EXPECT_EQ(0U, store.memory_bytes("idx_renamed"));

  ASSERT_TRUE(store.replace_index("idx_renamed", {{9, {9.0F, 9.0F}}}));
  ASSERT_TRUE(store.snapshot_index("idx_renamed", &entries));
  ASSERT_EQ(1U, entries.size());
  EXPECT_EQ(9U, entries.begin()->first);
}

TEST(VectorEntryStoreTest, ReadsEvictedEntriesFromTruthStore) {
  committed_rows_truth_store truth_store({
      {"idx_other", 99, {99.0F, 99.0F}},
      {"idx_store", 3, {3.0F, 3.0F}},
      {"idx_store", 1, {1.0F, 1.0F}},
  });
  truth_store_guard guard(&truth_store);
  vector_index::vector_entry_store store;
  vector_index::committed_state state;
  vector_index::committed_entries entries;
  std::vector<uint64_t> visited_doc_ids;

  ASSERT_TRUE(store.register_index("idx_store"));
  ASSERT_TRUE(store.upsert("idx_store", 7, {7.0F, 7.0F}));
  ASSERT_TRUE(store.evict_until_under_budget(0));

  ASSERT_TRUE(store.snapshot_index("idx_store", &entries));
  ASSERT_EQ(2U, entries.size());
  EXPECT_EQ(std::vector<float>({1.0F, 1.0F}), entries[1]);
  EXPECT_EQ(std::vector<float>({3.0F, 3.0F}), entries[3]);

  ASSERT_TRUE(store.snapshot(&state));
  ASSERT_EQ(1U, state.size());
  ASSERT_EQ(2U, state["idx_store"].size());

  ASSERT_TRUE(store.for_each_committed_entry(
      "idx_store", [&visited_doc_ids](
                       uint64_t doc_id,
                       const vector_index::vector_data &vector) {
        visited_doc_ids.push_back(doc_id);
        EXPECT_EQ(2U, vector.size());
        return true;
      }));
  EXPECT_EQ(std::vector<uint64_t>({3, 1}), visited_doc_ids);

  EXPECT_FALSE(store.for_each_committed_entry(
      "idx_store", [](uint64_t doc_id, const vector_index::vector_data &) {
        return doc_id != 3;
      }));
}

TEST(VectorIndexServiceTest, EvictsCommittedCacheToConfiguredBudget) {
  UlonglongGuard guard(&opt_vector_entry_cache_size, 0);
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index_from_strings("idx_mem", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(15, "idx_mem", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(15));
  EXPECT_EQ(2U * sizeof(float), service.committed_vector_memory_bytes());

  EXPECT_TRUE(service.evict_committed_cache_to_budget());
  EXPECT_EQ(1U, service.committed_entry_count());
  EXPECT_EQ(0U, service.committed_vector_memory_bytes());
}

TEST(VectorIndexServiceTest, EvictsLargestCommittedCacheShardOnly) {
  UlonglongGuard guard(&opt_vector_entry_cache_size, 2U * sizeof(float));
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index_from_strings("idx_small", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(service.register_index_from_strings("idx_large", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(16, "idx_small", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.stage_upsert(16, "idx_large", 10, {10.0F, 10.0F}));
  ASSERT_TRUE(service.stage_upsert(16, "idx_large", 11, {11.0F, 11.0F}));
  ASSERT_TRUE(service.commit(16));
  EXPECT_EQ(6U * sizeof(float), service.committed_vector_memory_bytes());

  EXPECT_TRUE(service.evict_committed_cache_to_budget());
  EXPECT_EQ(3U, service.committed_entry_count());
  EXPECT_EQ(2U * sizeof(float), service.committed_vector_memory_bytes());
}

TEST(VectorEntryStoreTest, RejectsInvalidOperationsAndPreservesState) {
  vector_index::vector_entry_store store;
  vector_index::committed_state snapshot;
  vector_index::committed_entries entries;
  std::vector<uint64_t> visited_doc_ids;

  EXPECT_FALSE(store.register_index(""));
  ASSERT_TRUE(store.register_index("idx_a"));
  EXPECT_FALSE(store.register_index("idx_a"));
  EXPECT_FALSE(store.drop_index("missing"));
  EXPECT_FALSE(store.rename_index("", "idx_b"));
  EXPECT_FALSE(store.rename_index("idx_a", ""));
  EXPECT_FALSE(store.rename_index("missing", "idx_b"));
  ASSERT_TRUE(store.register_index("idx_b"));
  EXPECT_FALSE(store.rename_index("idx_a", "idx_b"));
  EXPECT_FALSE(store.clear_index("missing"));
  EXPECT_FALSE(store.replace_index("missing", {{1, {1.0F, 1.0F}}}));
  EXPECT_FALSE(store.upsert("missing", 1, {1.0F, 1.0F}));
  EXPECT_FALSE(store.erase("missing", 1));
  EXPECT_FALSE(store.snapshot(nullptr));
  EXPECT_FALSE(store.snapshot_index("idx_a", nullptr));
  EXPECT_FALSE(store.snapshot_index("missing", &entries));
  EXPECT_FALSE(store.for_each_committed_entry("idx_a", nullptr));
  EXPECT_FALSE(store.for_each_committed_entry(
      "missing", [](uint64_t, const vector_index::vector_data &) {
        return true;
      }));
  EXPECT_EQ(0U, store.entry_count("missing"));
  EXPECT_EQ(0U, store.memory_bytes("missing"));

  ASSERT_TRUE(store.upsert("idx_a", 2, {2.0F, 2.0F}));
  ASSERT_TRUE(store.upsert("idx_a", 1, {1.0F, 1.0F}));
  EXPECT_FALSE(store.for_each_committed_entry(
      "idx_a", [&visited_doc_ids](uint64_t doc_id,
                                   const vector_index::vector_data &) {
        visited_doc_ids.push_back(doc_id);
        return false;
      }));
  ASSERT_EQ(1U, visited_doc_ids.size());
  EXPECT_EQ(1U, visited_doc_ids[0]);

  ASSERT_TRUE(store.snapshot(&snapshot));
  ASSERT_EQ(2U, snapshot["idx_a"].size());
  ASSERT_TRUE(store.snapshot_index("idx_a", &entries));
  ASSERT_EQ(2U, entries.size());
  ASSERT_TRUE(store.clear_index("idx_a"));
  EXPECT_EQ(0U, store.entry_count("idx_a"));
  ASSERT_TRUE(store.replace_index("idx_a", entries));
  ASSERT_TRUE(store.rename_index("idx_a", "idx_c"));
  EXPECT_TRUE(store.has_index("idx_c"));
  EXPECT_FALSE(store.has_index("idx_a"));
  EXPECT_EQ(2U, store.entry_count("idx_c"));
}

TEST(VectorIndexServiceTest, StageUpsertSpillsPendingPayloadOverBudget) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kNative)) {
    GTEST_SKIP() << "pending spill coverage requires debug native provider";
  }
  UlonglongGuard guard(&opt_vector_pending_cache_size, 2U * sizeof(float));
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_mem", 2, "euclidean",
                                                  "memory", "native"));
  EXPECT_TRUE(service.stage_upsert(14, "idx_mem", 1, {1.0F, 1.0F}));
  EXPECT_TRUE(service.stage_upsert(14, "idx_mem", 2, {2.0F, 2.0F}));
  EXPECT_EQ(2U * sizeof(float), service.pending_vector_memory_bytes(14));
  ASSERT_TRUE(service.search_with_pending(14, "idx_mem", {2.0F, 2.0F}, 2,
                                          &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
  ASSERT_TRUE(service.commit(14));
  EXPECT_EQ(4U * sizeof(float), service.committed_vector_memory_bytes());
}

TEST(VectorIndexServiceTest, DropIndexCleansFaissExternalSnapshotArtifacts) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_service_drop_cleanup_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_drop_cleanup", 2,
                                               "euclidean", "external",
                                               "faiss"));
  ASSERT_TRUE(service.stage_upsert(11, "idx_drop_cleanup", 1, {1.0F, 2.0F}));
  ASSERT_TRUE(service.commit(11));

  const std::string manifest_path =
      root + "/6964785f64726f705f636c65616e7570/faiss_external.manifest.v1";
  const std::string snapshot_path = find_faiss_snapshot_file(root);
  ASSERT_TRUE(std::filesystem::exists(manifest_path));
  ASSERT_TRUE(std::filesystem::exists(snapshot_path));
  ASSERT_TRUE(service.drop_index("idx_drop_cleanup"));
  EXPECT_FALSE(std::filesystem::exists(manifest_path));
  EXPECT_FALSE(std::filesystem::exists(snapshot_path));
  EXPECT_TRUE(find_faiss_snapshot_file(root).empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, RegisterIndexRejectsInvalidRawArguments) {
  vector_index::index_service service;
  EXPECT_FALSE(service.register_index("idx_null", nullptr));
  EXPECT_FALSE(service.register_index(
      "", std::make_unique<vector_index::memory_backend>(
              2, vector_index::metric_type::kEuclidean)));
}

TEST(VectorIndexServiceTest, RegisterIndexRejectsDuplicateAndInvalidConfig) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index(
      "idx_duplicate",
      std::make_unique<vector_index::memory_backend>(
          2, vector_index::metric_type::kEuclidean)));
  EXPECT_FALSE(service.register_index(
      "idx_duplicate",
      std::make_unique<vector_index::memory_backend>(
          2, vector_index::metric_type::kEuclidean)));

  vector_index::index_service::index_config config;
  config.dimension = 0;
  config.metric = vector_index::metric_type::kEuclidean;
  config.mode = vector_index::backend_mode::kMemory;
  config.provider = vector_index::backend_provider::kNative;
  EXPECT_FALSE(service.register_index("idx_zero_dim", config));

  config.dimension = 2;
  ASSERT_TRUE(service.register_index("idx_config_duplicate", config));
  EXPECT_FALSE(service.register_index("idx_config_duplicate", config));

  config.provider = vector_index::backend_provider::kNative;
  config.hnsw_build_threads = 1;
  EXPECT_FALSE(service.register_index("idx_native_hnsw_threads", config));
  config.hnsw_build_threads = 0;

  config.faiss_build_threads = 1;
  EXPECT_FALSE(service.register_index("idx_native_faiss_threads", config));
  config.faiss_build_threads = 0;

  config.faiss_nlist = 4;
  EXPECT_FALSE(service.register_index("idx_native_faiss_ivf", config));
  config.faiss_nlist = 0;

}

TEST(VectorIndexServiceTest,
     UnsupportedRegisteredConfigFailsRebuildRecoverAndRestore) {
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index("idx_incompatible",
                                    std::make_unique<IncompatibleBackend>()));

  EXPECT_FALSE(service.rebuild_index("idx_incompatible"));
  EXPECT_FALSE(service.recover_index("idx_incompatible"));

  vector_index::index_service::committed_state snapshot;
  EXPECT_FALSE(service.restore_committed_state(snapshot));

  size_t rebuilt_count = 0;
  size_t recovered_count = 0;
  EXPECT_FALSE(service.rebuild_all_indexes(&rebuilt_count));
  EXPECT_FALSE(service.recover_all_indexes(&recovered_count));
}

TEST(VectorIndexServiceTest, RegisterIndexByConfigBuildsBackend) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  vector_index::index_service::index_config config;
  config.dimension = 2;
  config.metric = vector_index::metric_type::kCosine;
  config.mode = vector_index::backend_mode::kMemory;
  config.provider = vector_index::backend_provider::kHnswlib;

  ASSERT_TRUE(service.register_index("idx_cfg", config));
  ASSERT_TRUE(service.stage_upsert(100, "idx_cfg", 11, {1.0F, 0.0F}));
  ASSERT_TRUE(service.commit(100));
  ASSERT_TRUE(service.search("idx_cfg", {1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, RegisterIndexFromStringsParsesOptions) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_opt", 2, "L2", " memory ",
                                               "FAISS"));
  ASSERT_TRUE(service.stage_upsert(101, "idx_opt", 22, {2.0F, 2.0F}));
  ASSERT_TRUE(service.commit(101));
  ASSERT_TRUE(service.search("idx_opt", {2.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(22U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, SetFaissIvfParamsRebuildsCommittedEntries) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_faiss_ivf", 2,
                                               "euclidean", "external",
                                               "faiss"));
  ASSERT_TRUE(service.stage_upsert(700, "idx_faiss_ivf", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.stage_upsert(700, "idx_faiss_ivf", 9, {9.0F, 9.0F}));
  ASSERT_TRUE(service.commit(700));

  ASSERT_TRUE(service.set_faiss_ivf_params("idx_faiss_ivf", 2, 1));
  ASSERT_TRUE(service.search("idx_faiss_ivf", {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, SetHnswBuildParamsRebuildsCommittedEntries) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_hnsw_params", 2,
                                               "euclidean", "memory",
                                               "hnswlib"));
  ASSERT_TRUE(service.stage_upsert(702, "idx_hnsw_params", 1, {1.0F, 0.0F}));
  ASSERT_TRUE(service.stage_upsert(702, "idx_hnsw_params", 2, {0.0F, 1.0F}));
  ASSERT_TRUE(service.commit(702));
  ASSERT_TRUE(service.set_search_ef("idx_hnsw_params", 32));

  ASSERT_TRUE(service.set_hnsw_build_params("idx_hnsw_params", 12, 96));

  vector_index::index_service::index_config config;
  bool supports_mutations = false;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  ASSERT_TRUE(service.describe_index("idx_hnsw_params", &config,
                                    &supports_mutations, &entry_count,
                                    &committed_entry_count));
  EXPECT_EQ(12U, config.hnsw_m);
  EXPECT_EQ(96U, config.hnsw_ef_construction);
  EXPECT_EQ(32U, config.search_ef);
  EXPECT_EQ(2U, entry_count);
  EXPECT_EQ(2U, committed_entry_count);

  ASSERT_TRUE(service.search("idx_hnsw_params", {1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, HnswSearchFailureRebuildsRuntimeFromTruthStore) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native recovery requires HAVE_HNSWLIB";
  }
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index(
      "idx_hnsw_recover",
      std::make_unique<vector_gunit::FailingSearchHnswBackend>(2)));
  ASSERT_TRUE(service.stage_upsert(703, "idx_hnsw_recover", 7, {1.0F, 0.0F}));
  ASSERT_TRUE(service.commit(703));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_hnsw_recover", {1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);

  vector_index::index_service::index_config config;
  uint64_t recover_fallback_count = 0;
  uint64_t last_recover_fallback_ts = 0;
  ASSERT_TRUE(service.describe_index(
      "idx_hnsw_recover", &config, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, &recover_fallback_count,
      &last_recover_fallback_ts));
  EXPECT_EQ(1U, recover_fallback_count);
  EXPECT_GT(last_recover_fallback_ts, 0U);
}

TEST(VectorIndexServiceTest,
     HnswBatchSearchFailureRebuildsRuntimeFromTruthStore) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native recovery requires HAVE_HNSWLIB";
  }
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index(
      "idx_hnsw_batch_recover",
      std::make_unique<vector_gunit::FailingSearchHnswBackend>(2)));
  ASSERT_TRUE(
      service.stage_upsert(704, "idx_hnsw_batch_recover", 7, {1.0F, 0.0F}));
  ASSERT_TRUE(
      service.stage_upsert(704, "idx_hnsw_batch_recover", 8, {0.0F, 1.0F}));
  ASSERT_TRUE(service.commit(704));

  std::vector<std::vector<vector_index::search_result>> batches;
  ASSERT_TRUE(service.search_batch(
      "idx_hnsw_batch_recover", {{1.0F, 0.0F}, {0.0F, 1.0F}}, 1, &batches));
  ASSERT_EQ(2U, batches.size());
  ASSERT_EQ(1U, batches[0].size());
  EXPECT_EQ(7U, batches[0][0].doc_id);
  ASSERT_EQ(1U, batches[1].size());
  EXPECT_EQ(8U, batches[1][0].doc_id);

  vector_index::index_service::index_config config;
  uint64_t recover_fallback_count = 0;
  uint64_t last_recover_fallback_ts = 0;
  ASSERT_TRUE(service.describe_index(
      "idx_hnsw_batch_recover", &config, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, &recover_fallback_count,
      &last_recover_fallback_ts));
  EXPECT_EQ(1U, recover_fallback_count);
  EXPECT_GT(last_recover_fallback_ts, 0U);
}

TEST(VectorIndexServiceTest, SearchBackendSnapshotKeepsRuntimeAliveAfterSwap) {
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_snapshot_runtime", 2,
                                                 "euclidean", "memory",
                                                 "native"));
  ASSERT_TRUE(
      service.stage_upsert(710, "idx_snapshot_runtime", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(710));

  vector_index::index_service::backend_ptr runtime;
  vector_index::index_service::index_config config;
  ASSERT_TRUE(service.snapshot_search_backend_loaded(
      "idx_snapshot_runtime", {1.0F, 1.0F}, &runtime, &config));
  ASSERT_NE(nullptr, runtime);
  EXPECT_EQ(2U, config.dimension);

  vector_index::index_service::committed_entries replacement;
  replacement.emplace(2, vector_index::vector_data{2.0F, 2.0F});
  ASSERT_TRUE(
      service.replace_committed_entries("idx_snapshot_runtime", replacement));

  std::vector<vector_index::search_result> result;
  {
    std::shared_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
    ASSERT_TRUE(runtime->search({1.0F, 1.0F}, 1, &result));
  }
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  ASSERT_TRUE(service.search("idx_snapshot_runtime", {2.0F, 2.0F}, 1,
                             &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, SetHnswBuildParamsRejectsMissingOrPendingChanges) {
  vector_index::index_service service;

  EXPECT_FALSE(service.set_hnsw_build_params("missing", 24, 320));
  ASSERT_TRUE(service.register_index_from_strings("idx_hnsw_pending", 2, "cosine",
                                               "memory", "hnswlib"));
  ASSERT_TRUE(service.stage_upsert(401, "idx_hnsw_pending", 1, {1.0F, 0.0F}));
  EXPECT_FALSE(service.set_hnsw_build_params("idx_hnsw_pending", 24, 320));
}

TEST(VectorIndexServiceTest, SetHnswBuildParamsRejectsZeroArguments) {
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_hnsw_zero", 2, "cosine",
                                               "memory", "hnswlib"));
  vector_index::index_service::index_config before;
  bool supports_mutations = false;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  ASSERT_TRUE(service.describe_index("idx_hnsw_zero", &before,
                                    &supports_mutations, &entry_count,
                                    &committed_entry_count));

  EXPECT_FALSE(service.set_hnsw_build_params("idx_hnsw_zero", 0, 320));
  EXPECT_FALSE(service.set_hnsw_build_params("idx_hnsw_zero", 24, 0));

  vector_index::index_service::index_config after;
  ASSERT_TRUE(service.describe_index("idx_hnsw_zero", &after,
                                    &supports_mutations, &entry_count,
                                    &committed_entry_count));
  EXPECT_EQ(before.hnsw_m, after.hnsw_m);
  EXPECT_EQ(before.hnsw_ef_construction, after.hnsw_ef_construction);
}

TEST(VectorIndexServiceTest,
     ExternalReloadFailuresKeepServingStateAcrossSetterAndRecoveryPaths) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_service_reload_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_ext", 2, "euclidean",
                                               "external", "faiss"));
  ASSERT_TRUE(service.stage_upsert(701, "idx_ext", 11, {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(701));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_ext", {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_save_manifest_generation");
    EXPECT_FALSE(service.set_hnsw_build_params("idx_ext", 24, 320));
  }
  ASSERT_TRUE(service.search("idx_ext", {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_save_manifest_generation");
    EXPECT_FALSE(service.rebuild_index("idx_ext"));
  }
  ASSERT_TRUE(service.search("idx_ext", {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_save_manifest_generation");
    EXPECT_FALSE(service.recover_index("idx_ext"));
  }
  ASSERT_TRUE(service.search("idx_ext", {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  size_t rebuilt_count = 777;
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_save_manifest_generation");
    EXPECT_FALSE(service.rebuild_all_indexes(&rebuilt_count));
  }
  EXPECT_EQ(777U, rebuilt_count);
  ASSERT_TRUE(service.search("idx_ext", {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  size_t recovered_count = 888;
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_save_manifest_generation");
    EXPECT_FALSE(service.recover_all_indexes(&recovered_count));
  }
  EXPECT_EQ(888U, recovered_count);
  ASSERT_TRUE(service.search("idx_ext", {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, SetFaissIvfParamsRejectsMissingOrPendingChanges) {
  vector_index::index_service service;

  EXPECT_FALSE(service.set_faiss_ivf_params("missing", 16, 4));
  ASSERT_TRUE(service.register_index_from_strings("idx_faiss_pending", 2,
                                               "euclidean", "external",
                                               "faiss"));
  ASSERT_TRUE(service.stage_upsert(402, "idx_faiss_pending", 1, {1.0F, 1.0F}));
  EXPECT_FALSE(service.set_faiss_ivf_params("idx_faiss_pending", 16, 4));
}

TEST(VectorIndexServiceTest, SetFaissIvfParamsRejectsHalfZeroPair) {
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_faiss_zero", 2,
                                               "euclidean", "external",
                                               "faiss"));
  EXPECT_FALSE(service.set_faiss_ivf_params("idx_faiss_zero", 0, 4));
  EXPECT_FALSE(service.set_faiss_ivf_params("idx_faiss_zero", 16, 0));
}

TEST(VectorIndexServiceTest,
     SetFaissIvfPqParamsRejectsMissingOrPendingChanges) {
  vector_index::index_service service;

  EXPECT_FALSE(service.set_faiss_ivf_pq_params("missing", 16, 4, 8, 8));
  ASSERT_TRUE(service.register_index_from_strings("idx_faiss_pq_pending", 4,
                                               "euclidean", "external",
                                               "faiss"));
  ASSERT_TRUE(service.stage_upsert(403, "idx_faiss_pq_pending", 1,
                                  {1.0F, 0.0F, 0.0F, 0.0F}));
  EXPECT_FALSE(
      service.set_faiss_ivf_pq_params("idx_faiss_pq_pending", 16, 4, 8, 8));
}

TEST(VectorIndexServiceTest, SetFaissIvfPqParamsRejectsZeroArguments) {
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_faiss_pq_zero", 4,
                                               "euclidean", "external",
                                               "faiss"));
  EXPECT_FALSE(service.set_faiss_ivf_pq_params("idx_faiss_pq_zero", 0, 4, 8, 8));
  EXPECT_FALSE(service.set_faiss_ivf_pq_params("idx_faiss_pq_zero", 16, 0, 8, 8));
  EXPECT_FALSE(service.set_faiss_ivf_pq_params("idx_faiss_pq_zero", 16, 4, 0, 8));
  EXPECT_FALSE(service.set_faiss_ivf_pq_params("idx_faiss_pq_zero", 16, 4, 8, 0));
}

TEST(VectorIndexServiceTest,
     SetDiskAnnBuildParamsRejectsMissingOrPendingChanges) {
  vector_index::index_service service;

  EXPECT_FALSE(service.set_diskann_build_params("missing", 24, 96, 0));
  ASSERT_TRUE(service.register_index_from_strings("idx_diskann_pending", 2,
                                               "euclidean", "external",
                                               "diskann"));
  ASSERT_TRUE(service.stage_upsert(404, "idx_diskann_pending", 1, {1.0F, 1.0F}));
  EXPECT_FALSE(service.set_diskann_build_params("idx_diskann_pending", 24, 96,
                                                0));
}

TEST(VectorIndexServiceTest, SetDiskAnnBuildParamsRejectsZeroArguments) {
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_diskann_zero", 2,
                                               "euclidean", "external",
                                               "diskann"));
  EXPECT_FALSE(service.set_diskann_build_params("idx_diskann_zero", 0, 96, 0));
  EXPECT_FALSE(service.set_diskann_build_params("idx_diskann_zero", 24, 0, 0));
  EXPECT_FALSE(service.set_diskann_build_params("idx_diskann_zero", 24, 96,
                                                65536));
}

TEST(VectorIndexServiceTest,
     SetDiskAnnSearchComplexityRejectsMissingAndZero) {
  vector_index::index_service service;
  EXPECT_FALSE(service.set_diskann_search_complexity("missing", 80));
  ASSERT_TRUE(service.register_index_from_strings("idx_diskann_search_zero", 2,
                                               "euclidean", "external",
                                               "diskann"));
  EXPECT_FALSE(service.set_diskann_search_complexity("idx_diskann_search_zero", 0));
}

TEST(VectorIndexServiceTest,
     SetterFamiliesRejectIncompatibleOrUnsupportedBackends) {
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index("idx_incompatible",
                                    std::make_unique<IncompatibleBackend>()));
  ASSERT_TRUE(service.register_index("idx_nowrite",
                                    std::make_unique<NonWritableBackend>()));

  EXPECT_FALSE(service.set_search_ef("idx_incompatible", 64));
  EXPECT_FALSE(service.set_hnsw_build_params("idx_incompatible", 24, 320));
  EXPECT_FALSE(service.set_faiss_ivf_params("idx_incompatible", 16, 4));
  EXPECT_FALSE(service.set_faiss_ivf_pq_params("idx_incompatible", 16, 4, 8, 8));
  EXPECT_FALSE(service.set_diskann_build_params("idx_incompatible", 24, 96, 0));
  EXPECT_FALSE(service.set_diskann_search_complexity("idx_incompatible", 80));
  EXPECT_FALSE(service.set_search_ef("idx_nowrite", 64));
}

TEST(VectorIndexServiceTest, SearchWithPendingSeesOwnUpsertBeforeCommit) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_pending_search", 2,
                                               "euclidean", "memory",
                                               "native"));
  ASSERT_TRUE(service.stage_upsert(500, "idx_pending_search", 77,
                                  {7.0F, 7.0F}));

  ASSERT_TRUE(service.search("idx_pending_search", {7.0F, 7.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_TRUE(service.search_with_pending(500, "idx_pending_search",
                                        {7.0F, 7.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(77U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, SearchWithPendingRejectsMissingIndex) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  EXPECT_FALSE(service.search_with_pending(1, "missing", {1.0F, 1.0F}, 1,
                                         &result));
}

TEST(VectorIndexServiceTest, SearchWithPendingIgnoresChangesForOtherIndexes) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_target", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.register_index_from_strings("idx_other", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(333, "idx_target", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(333));
  ASSERT_TRUE(service.stage_upsert(444, "idx_other", 9, {9.0F, 9.0F}));

  ASSERT_TRUE(service.search_with_pending(444, "idx_target", {1.0F, 1.0F}, 2,
                                        &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, RebuildAllIndexesRejectsNullCount) {
  vector_index::index_service service;
  EXPECT_FALSE(service.rebuild_all_indexes(nullptr));
}

TEST(VectorIndexServiceTest, RecoverAllIndexesRejectsNullCount) {
  vector_index::index_service service;
  EXPECT_FALSE(service.recover_all_indexes(nullptr));
}

TEST(VectorIndexServiceTest, SearchWithPendingAppliesEraseOverlay) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_pending_erase", 2,
                                               "euclidean", "memory",
                                               "native"));
  ASSERT_TRUE(service.stage_upsert(501, "idx_pending_erase", 88,
                                  {8.0F, 8.0F}));
  ASSERT_TRUE(service.commit(501));

  ASSERT_TRUE(service.stage_erase(777, "idx_pending_erase", 88));
  ASSERT_TRUE(service.search_with_pending(777, "idx_pending_erase",
                                        {8.0F, 8.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
}

TEST(VectorIndexServiceTest, SearchWithPendingRespectsSavepointRollback) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_pending_sp", 2,
                                               "euclidean", "memory",
                                               "native"));
  ASSERT_TRUE(service.stage_upsert(900, "idx_pending_sp", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.savepoint(900, "sp1"));
  ASSERT_TRUE(service.stage_upsert(900, "idx_pending_sp", 2, {2.0F, 2.0F}));

  ASSERT_TRUE(
      service.search_with_pending(900, "idx_pending_sp", {2.0F, 2.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());

  ASSERT_TRUE(service.rollback_to_savepoint(900, "sp1"));
  ASSERT_TRUE(
      service.search_with_pending(900, "idx_pending_sp", {2.0F, 2.0F}, 2, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, SearchBatchAndLoadedSearchCoverLifecycleGuards) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;
  std::vector<std::vector<vector_index::search_result>> batches;

  ASSERT_TRUE(service.register_index_from_strings("idx_batch", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(1, "idx_batch", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.stage_upsert(1, "idx_batch", 2, {2.0F, 2.0F}));
  ASSERT_TRUE(service.commit(1));

  EXPECT_FALSE(service.search_batch("missing", {{1.0F, 1.0F}}, 1, &batches));
  ASSERT_TRUE(service.search_batch("idx_batch",
                                   {{1.0F, 1.0F}, {2.0F, 2.0F}}, 1,
                                   &batches));
  ASSERT_EQ(2U, batches.size());
  ASSERT_EQ(1U, batches[0].size());
  EXPECT_EQ(1U, batches[0][0].doc_id);
  ASSERT_EQ(1U, batches[1].size());
  EXPECT_EQ(2U, batches[1][0].doc_id);

  EXPECT_FALSE(service.search_loaded("missing", {1.0F, 1.0F}, 1, &result));
  EXPECT_FALSE(service.search_batch_loaded("missing", {{1.0F, 1.0F}}, 1,
                                           &batches));
  ASSERT_TRUE(service.search_with_pending_loaded(999, "idx_batch",
                                                 {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  ASSERT_TRUE(service.begin_bulk_load("idx_batch"));
  EXPECT_FALSE(service.search("idx_batch", {1.0F, 1.0F}, 1, &result));
  EXPECT_FALSE(service.search_loaded("idx_batch", {1.0F, 1.0F}, 1, &result));
  EXPECT_FALSE(service.search_batch("idx_batch", {{1.0F, 1.0F}}, 1,
                                    &batches));
  EXPECT_FALSE(service.search_batch_loaded("idx_batch", {{1.0F, 1.0F}}, 1,
                                           &batches));
  ASSERT_TRUE(service.stage_upsert(2, "idx_batch", 3, {3.0F, 3.0F}));
  EXPECT_FALSE(service.search_with_pending_loaded(2, "idx_batch",
                                                  {3.0F, 3.0F}, 1, &result));
}

TEST(VectorIndexServiceTest, DescribeIndexReturnsConfigAndMutationFlag) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index_from_strings("idx_ext", 8, "cosine", "external",
                                               "faiss"));
  vector_index::index_service::index_config config;
  bool supports_mutations = true;
  size_t entry_count = 99;
  size_t committed_entry_count = 99;
  ASSERT_TRUE(service.describe_index("idx_ext", &config, &supports_mutations,
                                    &entry_count, &committed_entry_count));

  EXPECT_EQ(8U, config.dimension);
  EXPECT_EQ(vector_index::metric_type::kCosine, config.metric);
  EXPECT_EQ(vector_index::backend_mode::kExternal, config.mode);
  EXPECT_EQ(vector_index::backend_provider::kFaiss, config.provider);
  EXPECT_TRUE(supports_mutations);
  EXPECT_EQ(0U, entry_count);
  EXPECT_EQ(0U, committed_entry_count);
}

TEST(VectorIndexServiceTest,
     SetLifecycleInfoRejectsInvalidArgumentsAndMissingIndex) {
  vector_index::index_service service;

  EXPECT_FALSE(service.set_lifecycle_info("missing", "ready", 1));
  ASSERT_TRUE(service.register_index_from_strings("idx_lifecycle", 2, "euclidean",
                                               "memory", "native"));
  EXPECT_FALSE(service.set_lifecycle_info("idx_lifecycle", "", 1));
  EXPECT_FALSE(service.set_lifecycle_info("idx_lifecycle", "ready", 0));
  EXPECT_TRUE(service.set_lifecycle_info("idx_lifecycle", "failed", 9, 1006, 77));
}

TEST(VectorIndexServiceTest, DescribeIndexReportsLastApplyLatencyMs) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index_from_strings("idx_latency", 4, "euclidean",
                                               "memory", "native"));

  vector_index::index_service::index_config config;
  bool supports_mutations = false;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  uint64_t last_apply_latency_ms = 99;
  ASSERT_TRUE(service.describe_index("idx_latency", &config, &supports_mutations,
                                    &entry_count, &committed_entry_count,
                                    nullptr, nullptr, nullptr, nullptr,
                                    &last_apply_latency_ms));
  EXPECT_EQ(0U, last_apply_latency_ms);

  EXPECT_TRUE(service.set_last_apply_latency_ms("idx_latency", 17));
  last_apply_latency_ms = 0;
  ASSERT_TRUE(service.describe_index("idx_latency", &config, &supports_mutations,
                                    &entry_count, &committed_entry_count,
                                    nullptr, nullptr, nullptr, nullptr,
                                    &last_apply_latency_ms));
  EXPECT_EQ(17U, last_apply_latency_ms);

  EXPECT_FALSE(service.set_last_apply_latency_ms("missing", 3));
}

TEST(VectorIndexServiceTest,
     DescribeIndexReportsLifecycleAndExternalManifestOutputs) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_service_describe_manifest_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_describe_manifest", 2,
                                               "euclidean", "external",
                                               "faiss"));
  ASSERT_TRUE(service.stage_upsert(741, "idx_describe_manifest", 17,
                                  {1.0F, 7.0F}));
  ASSERT_TRUE(service.commit(741));
  ASSERT_TRUE(service.set_lifecycle_info("idx_describe_manifest", "ready", 3,
                                       0, 0, 2, 19));

  vector_index::index_service::index_config config;
  bool supports_mutations = false;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  std::string lifecycle_state;
  uint64_t lifecycle_version = 0;
  uint32_t last_error_code = 1;
  uint64_t last_error_ts = 1;
  uint64_t last_apply_latency_ms = 1;
  uint64_t recover_fallback_count = 0;
  uint64_t last_recover_fallback_ts = 0;
  bool external_manifest_present = false;
  uint64_t external_manifest_generation = 0;
  ASSERT_TRUE(service.describe_index(
      "idx_describe_manifest", &config, &supports_mutations, &entry_count,
      &committed_entry_count, &lifecycle_state, &lifecycle_version,
      &last_error_code, &last_error_ts, &last_apply_latency_ms,
      &recover_fallback_count, &last_recover_fallback_ts,
      &external_manifest_present, &external_manifest_generation));

  EXPECT_EQ("ready", lifecycle_state);
  EXPECT_EQ(3U, lifecycle_version);
  EXPECT_EQ(0U, last_error_code);
  EXPECT_EQ(0U, last_error_ts);
  EXPECT_EQ(0U, last_apply_latency_ms);
  EXPECT_EQ(2U, recover_fallback_count);
  EXPECT_EQ(19U, last_recover_fallback_ts);
  EXPECT_TRUE(external_manifest_present);
  EXPECT_GT(external_manifest_generation, 0U);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, DescribeIndexAllowsNullOptionalOutputs) {
  vector_index::index_service service;
  vector_index::index_service::index_config config;
  bool supports_mutations = false;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;

  ASSERT_TRUE(service.register_index_from_strings("idx_describe_null", 2,
                                               "euclidean", "memory",
                                               "native"));
  ASSERT_TRUE(service.describe_index("idx_describe_null", &config,
                                    &supports_mutations, &entry_count,
                                    &committed_entry_count, nullptr, nullptr,
                                    nullptr, nullptr, nullptr, nullptr,
                                    nullptr, nullptr, nullptr));
  EXPECT_EQ(2U, config.dimension);
  EXPECT_TRUE(supports_mutations);
}

TEST(VectorIndexServiceTest, ListIndexesReturnsSortedNames) {
  vector_index::index_service service;
  std::vector<std::string> index_names;

  ASSERT_TRUE(service.list_indexes(&index_names));
  EXPECT_TRUE(index_names.empty());

  ASSERT_TRUE(service.register_index_from_strings("idx_b", 2, "euclidean", "memory",
                                               "native"));
  ASSERT_TRUE(service.register_index_from_strings("idx_a", 2, "euclidean", "memory",
                                               "native"));
  ASSERT_TRUE(service.register_index_from_strings("idx_c", 2, "euclidean", "external",
                                               "faiss"));
  ASSERT_TRUE(service.list_indexes(&index_names));

  ASSERT_EQ(3U, index_names.size());
  EXPECT_EQ("idx_a", index_names[0]);
  EXPECT_EQ("idx_b", index_names[1]);
  EXPECT_EQ("idx_c", index_names[2]);
}

TEST(VectorIndexServiceTest, ListIndexesRejectsNullOutput) {
  vector_index::index_service service;
  EXPECT_FALSE(service.list_indexes(nullptr));
}

TEST(VectorIndexServiceTest, RebuildIndexReloadsCommittedDataAndPreservesConfig) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_rebuild", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(201, "idx_rebuild", 7, {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(201));
  ASSERT_TRUE(service.search("idx_rebuild", {1.0F, 1.0F}, 10, &result));
  ASSERT_EQ(1U, result.size());

  ASSERT_TRUE(service.rebuild_index("idx_rebuild"));
  ASSERT_TRUE(service.search("idx_rebuild", {1.0F, 1.0F}, 10, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);

  vector_index::index_service::index_config config;
  bool supports_mutations = false;
  size_t entry_count = 99;
  size_t committed_entry_count = 99;
  ASSERT_TRUE(service.describe_index("idx_rebuild", &config, &supports_mutations,
                                    &entry_count, &committed_entry_count));
  EXPECT_EQ(2U, config.dimension);
  EXPECT_EQ(vector_index::metric_type::kEuclidean, config.metric);
  EXPECT_EQ(vector_index::backend_mode::kMemory, config.mode);
  EXPECT_EQ(vector_index::backend_provider::kNative, config.provider);
  EXPECT_TRUE(supports_mutations);
  EXPECT_EQ(1U, entry_count);
  EXPECT_EQ(1U, committed_entry_count);
}

TEST(VectorIndexServiceTest, RebuildIndexRejectsMissingOrPendingTxnChanges) {
  vector_index::index_service service;

  EXPECT_FALSE(service.rebuild_index("missing"));
  ASSERT_TRUE(service.register_index_from_strings("idx_pending", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(301, "idx_pending", 1, {1.0F, 1.0F}));
  EXPECT_FALSE(service.rebuild_index("idx_pending"));

  vector_index::index_service::index_config config;
  bool supports_mutations = false;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  std::string lifecycle_state;
  uint64_t lifecycle_version = 0;
  uint32_t last_error_code = 0;
  uint64_t last_error_ts = 0;
  ASSERT_TRUE(service.describe_index("idx_pending", &config, &supports_mutations,
                                    &entry_count, &committed_entry_count,
                                    &lifecycle_state, &lifecycle_version,
                                    &last_error_code, &last_error_ts));
  EXPECT_EQ("failed", lifecycle_state);
  EXPECT_EQ(1001U, last_error_code);
  EXPECT_GT(last_error_ts, 0U);

  service.rollback(301);
  EXPECT_TRUE(service.rebuild_index("idx_pending"));
  ASSERT_TRUE(service.describe_index("idx_pending", &config, &supports_mutations,
                                    &entry_count, &committed_entry_count,
                                    &lifecycle_state, &lifecycle_version,
                                    &last_error_code, &last_error_ts));
  EXPECT_EQ("ready", lifecycle_state);
  EXPECT_EQ(0U, last_error_code);
  EXPECT_EQ(0U, last_error_ts);
}

TEST(VectorIndexServiceTest, RecoverIndexReplaysCommittedState) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_recover_one", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(311, "idx_recover_one", 9, {1.0F, 3.0F}));
  ASSERT_TRUE(service.commit(311));
  ASSERT_TRUE(service.rebuild_index("idx_recover_one"));
  ASSERT_TRUE(service.search("idx_recover_one", {1.0F, 3.0F}, 10, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(9U, result[0].doc_id);

  ASSERT_TRUE(service.recover_index("idx_recover_one"));
  ASSERT_TRUE(service.search("idx_recover_one", {1.0F, 3.0F}, 10, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(9U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, RecoverIndexRecordsFallbackMetadataForFaissExternal) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_service_recover_fallback_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::index_service service;
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.register_index_from_strings("idx_recover_fallback", 2,
                                               "euclidean", "external",
                                               "faiss"));
  ASSERT_TRUE(service.stage_upsert(313, "idx_recover_fallback", 7, {7.0F, 7.0F}));
  ASSERT_TRUE(service.commit(313));

  const std::string snapshot_path = find_faiss_snapshot_file(root);
  ASSERT_FALSE(snapshot_path.empty());
  {
    std::ofstream broken(snapshot_path,
                         std::ios::out | std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(broken.good());
    broken << "broken-sidecar";
  }

  ASSERT_TRUE(service.recover_index("idx_recover_fallback"));
  ASSERT_TRUE(service.search("idx_recover_fallback", {7.0F, 7.0F}, 10, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);

  vector_index::index_service::index_config config;
  bool supports_mutations = false;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  std::string lifecycle_state;
  uint64_t lifecycle_version = 0;
  uint32_t last_error_code = 0;
  uint64_t last_error_ts = 0;
  uint64_t last_apply_latency_ms = 0;
  uint64_t recover_fallback_count = 0;
  uint64_t last_recover_fallback_ts = 0;
  ASSERT_TRUE(service.describe_index(
      "idx_recover_fallback", &config, &supports_mutations, &entry_count,
      &committed_entry_count, &lifecycle_state, &lifecycle_version,
      &last_error_code, &last_error_ts, &last_apply_latency_ms,
      &recover_fallback_count, &last_recover_fallback_ts));
  EXPECT_EQ("ready", lifecycle_state);
  EXPECT_EQ(1U, recover_fallback_count);
  EXPECT_GT(last_recover_fallback_ts, 0U);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, RecoverIndexRejectsMissingOrPendingTxnChanges) {
  vector_index::index_service service;

  EXPECT_FALSE(service.recover_index("missing"));
  ASSERT_TRUE(service.register_index_from_strings("idx_recover_pending", 2,
                                               "euclidean", "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(312, "idx_recover_pending", 1, {1.0F, 1.0F}));
  EXPECT_FALSE(service.recover_index("idx_recover_pending"));
  service.rollback(312);
  EXPECT_TRUE(service.recover_index("idx_recover_pending"));
}

TEST(VectorIndexServiceTest, RebuildAllIndexesReloadsCommittedData) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;
  size_t rebuilt_count = 0;

  ASSERT_TRUE(service.register_index_from_strings("idx_mem_a", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.register_index_from_strings("idx_mem_b", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(401, "idx_mem_a", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.stage_upsert(401, "idx_mem_b", 2, {2.0F, 2.0F}));
  ASSERT_TRUE(service.commit(401));

  ASSERT_TRUE(service.rebuild_all_indexes(&rebuilt_count));
  EXPECT_EQ(2U, rebuilt_count);
  ASSERT_TRUE(service.search("idx_mem_a", {1.0F, 1.0F}, 10, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  ASSERT_TRUE(service.search("idx_mem_b", {2.0F, 2.0F}, 10, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, RebuildAllIndexesRejectsPendingChanges) {
  vector_index::index_service service;
  size_t rebuilt_count = 0;

  ASSERT_TRUE(service.register_index_from_strings("idx_pending_all", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(501, "idx_pending_all", 1, {1.0F, 1.0F}));
  EXPECT_FALSE(service.rebuild_all_indexes(&rebuilt_count));
}

TEST(VectorIndexServiceTest, RecoverAllIndexesReturnsRecoveredCount) {
  vector_index::index_service service;
  size_t recovered_count = 0;

  ASSERT_TRUE(service.register_index_from_strings("idx_recover_a", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.register_index_from_strings("idx_recover_b", 2, "cosine",
                                               "external", "faiss"));
  ASSERT_TRUE(service.recover_all_indexes(&recovered_count));
  EXPECT_EQ(2U, recovered_count);
}

TEST(VectorIndexServiceTest,
     DiskAnnBuildParamsAreDeferredUntilExplicitRebuild) {
  if (!vector_index::diskann_api_load_for_testing()) {
    GTEST_SKIP() << "DiskANN API unavailable in current build/runtime";
  }
  const std::string root =
      std::string(testing::TempDir()) + "/vector_service_diskann_deferred_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::index_service service;
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.register_index_from_strings("idx_diskann_deferred", 2,
                                               "euclidean", "external",
                                               "diskann"));
  ASSERT_TRUE(service.stage_upsert(701, "idx_diskann_deferred", 11,
                                  {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(701));
  ASSERT_TRUE(
      service.search("idx_diskann_deferred", {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  const std::string manifest_path = find_diskann_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string store_path = diskann_store_path_for_manifest(manifest_path);
  EXPECT_FALSE(std::filesystem::exists(store_path));

  ASSERT_TRUE(service.set_diskann_build_params("idx_diskann_deferred", 48, 96,
                                               0));
  EXPECT_FALSE(std::filesystem::exists(store_path));

  ASSERT_TRUE(service.rebuild_index("idx_diskann_deferred"));
  EXPECT_TRUE(std::filesystem::is_directory(store_path));
  ASSERT_TRUE(
      service.search("idx_diskann_deferred", {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     RecoverIndexWithoutNativeStoreRebuildsNativeStoreAndKeepsSearchable) {
  if (!vector_index::diskann_api_load_for_testing()) {
    GTEST_SKIP() << "DiskANN API unavailable in current build/runtime";
  }
  const std::string root =
      std::string(testing::TempDir()) + "/vector_service_diskann_recover_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::index_service service;
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.register_index_from_strings("idx_diskann_recover", 2,
                                               "euclidean", "external",
                                               "diskann"));
  ASSERT_TRUE(service.stage_upsert(702, "idx_diskann_recover", 21,
                                  {2.0F, 1.0F}));
  ASSERT_TRUE(service.commit(702));

  const std::string manifest_path = find_diskann_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string store_path = diskann_store_path_for_manifest(manifest_path);
  std::filesystem::remove_all(store_path, ec);
  ASSERT_FALSE(std::filesystem::exists(store_path));

  ASSERT_TRUE(service.recover_index("idx_diskann_recover"));
  EXPECT_TRUE(std::filesystem::is_directory(store_path));
  ASSERT_TRUE(
      service.search("idx_diskann_recover", {2.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(21U, result[0].doc_id);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     ExternalRuntimeIgnoresRemovedLazyEnvironmentOption) {
  const EnvVarGuard guard("MYSQL_VECTOR_LAZY_EXTERNAL_RUNTIME");
  const std::string root =
      std::string(testing::TempDir()) + "/vector_service_lazy_env_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  const char *values[] = {"1",     "true",  "TRUE", "yes", "YES",
                          "on",    "ON",    "0",    "false", "FALSE",
                          "no",    "NO",    "off",  "OFF",   "invalid"};
  size_t suffix = 0;

  auto exercise_value = [&](const char *value) {
    setenv("MYSQL_VECTOR_LAZY_EXTERNAL_RUNTIME", value, 1);

    vector_index::index_service service;
    const std::string index_name =
        "idx_lazy_env_" + std::to_string(++suffix);
    ASSERT_TRUE(service.register_index_from_strings(index_name, 2, "euclidean",
                                                 "external", "diskann"));

    vector_index::index_service::committed_state snapshot;
    snapshot[index_name][suffix] = {static_cast<float>(suffix), 1.0F};
    ASSERT_TRUE(service.restore_committed_state(snapshot));

    vector_index::index_service::index_config config;
    bool supports_mutations = false;
    size_t entry_count = 99;
    size_t committed_entry_count = 99;
    ASSERT_TRUE(service.describe_index(index_name, &config, &supports_mutations,
                                      &entry_count, &committed_entry_count));
    EXPECT_EQ(1U, entry_count) << value;
    EXPECT_EQ(1U, committed_entry_count) << value;

    std::vector<vector_index::search_result> result;
    ASSERT_TRUE(service.search_loaded(index_name,
                                      {static_cast<float>(suffix), 1.0F}, 1,
                                      &result));
    ASSERT_EQ(1U, result.size()) << value;
    EXPECT_EQ(suffix, result[0].doc_id) << value;
  };

  for (const char *value : values) {
    exercise_value(value);
  }

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     ExternalRuntimeCoversEmptyAndAlreadyLoadedBranches) {
  const EnvVarGuard guard("MYSQL_VECTOR_LAZY_EXTERNAL_RUNTIME");
  setenv("MYSQL_VECTOR_LAZY_EXTERNAL_RUNTIME", "1", 1);

  {
    vector_index::index_service service;
    ASSERT_TRUE(service.register_index_from_strings("idx_empty_env", 2,
                                                 "euclidean", "external",
                                                 "diskann"));
    vector_index::index_service::committed_state snapshot;
    snapshot["idx_empty_env"][1] = {1.0F, 1.0F};
    ASSERT_TRUE(service.restore_committed_state(snapshot));

    vector_index::index_service::index_config config;
    bool supports_mutations = false;
    size_t entry_count = 0;
    size_t committed_entry_count = 0;
    ASSERT_TRUE(service.describe_index("idx_empty_env", &config,
                                      &supports_mutations, &entry_count,
                                      &committed_entry_count));
    EXPECT_EQ(1U, entry_count);
    EXPECT_EQ(1U, committed_entry_count);
  }

  {
    vector_index::index_service service;
    ASSERT_TRUE(service.register_index_from_strings("idx_lazy_memory", 2,
                                                 "euclidean", "memory",
                                                 "native"));
    ASSERT_TRUE(service.stage_upsert(711, "idx_lazy_memory", 7,
                                    {7.0F, 1.0F}));
    ASSERT_TRUE(service.commit(711));
    EXPECT_TRUE(service.rebuild_index("idx_lazy_memory"));
  }

  {
    vector_index::index_service service;
    ASSERT_TRUE(service.register_index_from_strings("idx_lazy_empty", 2,
                                                 "euclidean", "external",
                                                 "diskann"));
    vector_index::index_service::committed_state snapshot;
    snapshot["idx_lazy_empty"] = {};
    ASSERT_TRUE(service.restore_committed_state(snapshot));

    std::vector<vector_index::search_result> result;
    ASSERT_TRUE(service.search("idx_lazy_empty", {1.0F, 1.0F}, 1, &result));
    EXPECT_TRUE(result.empty());
  }

  {
    vector_index::index_service service;
    ASSERT_TRUE(service.register_index_from_strings("idx_lazy_loaded", 2,
                                                 "euclidean", "external",
                                                 "diskann"));
    ASSERT_TRUE(service.stage_upsert(712, "idx_lazy_loaded", 9,
                                    {9.0F, 1.0F}));
    ASSERT_TRUE(service.commit(712));

    std::vector<vector_index::search_result> result;
    ASSERT_TRUE(service.search("idx_lazy_loaded", {9.0F, 1.0F}, 1, &result));
    ASSERT_EQ(1U, result.size());
    EXPECT_EQ(9U, result[0].doc_id);
  }
}

TEST(VectorIndexServiceTest, RecoverAllIndexesRejectsPendingChanges) {
  vector_index::index_service service;
  size_t recovered_count = 0;

  ASSERT_TRUE(service.register_index_from_strings("idx_recover_pending", 2,
                                               "euclidean", "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(601, "idx_recover_pending", 1, {1.0F, 1.0F}));
  EXPECT_FALSE(service.recover_all_indexes(&recovered_count));
}

TEST(VectorIndexServiceTest, RecoverAllIndexesReplaysCommittedState) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;
  size_t recovered_count = 0;

  ASSERT_TRUE(service.register_index_from_strings("idx_recover_state", 2,
                                               "euclidean", "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(602, "idx_recover_state", 7, {1.0F, 2.0F}));
  ASSERT_TRUE(service.commit(602));
  ASSERT_TRUE(service.rebuild_index("idx_recover_state"));
  ASSERT_TRUE(service.search("idx_recover_state", {1.0F, 2.0F}, 10, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);

  ASSERT_TRUE(service.recover_all_indexes(&recovered_count));
  EXPECT_EQ(1U, recovered_count);
  ASSERT_TRUE(service.search("idx_recover_state", {1.0F, 2.0F}, 10, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, SnapshotAndRestoreCommittedStateRoundTrip) {
  vector_index::index_service source;
  ASSERT_TRUE(source.register_index_from_strings("idx_mem", 2, "euclidean", "memory",
                                              "native"));
  ASSERT_TRUE(source.register_index_from_strings("idx_ext", 2, "euclidean",
                                              "external", "faiss"));
  ASSERT_TRUE(source.stage_upsert(603, "idx_mem", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(source.stage_upsert(603, "idx_ext", 9, {3.0F, 3.0F}));
  ASSERT_TRUE(source.commit(603));

  vector_index::index_service::committed_state snapshot;
  ASSERT_TRUE(source.snapshot_committed_state(&snapshot));
  ASSERT_EQ(2U, snapshot.size());

  vector_index::index_service restored;
  ASSERT_TRUE(restored.register_index_from_strings("idx_mem", 2, "euclidean", "memory",
                                                "native"));
  ASSERT_TRUE(restored.register_index_from_strings("idx_ext", 2, "euclidean",
                                                "external", "faiss"));
  ASSERT_TRUE(restored.restore_committed_state(snapshot));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(restored.search("idx_mem", {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  ASSERT_TRUE(restored.search("idx_ext", {3.0F, 3.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(9U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, RestoreCommittedStateSupportsMutableDiskAnn) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_diskann", 2, "euclidean",
                                               "external", "diskann"));

  vector_index::index_service::committed_state snapshot;
  snapshot["idx_diskann"][77] = {7.0F, 7.0F};
  snapshot["idx_diskann"][11] = {1.0F, 1.0F};
  ASSERT_TRUE(service.restore_committed_state(snapshot));

  ASSERT_TRUE(service.search("idx_diskann", {1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  ASSERT_TRUE(service.stage_upsert(710, "idx_diskann", 88, {8.0F, 8.0F}));
  EXPECT_TRUE(service.commit(710));
  ASSERT_TRUE(service.search("idx_diskann", {8.0F, 8.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(88U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, DescribeIndexReportsEntryCountForMemoryMode) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index_from_strings("idx_count", 2, "euclidean",
                                               "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(701, "idx_count", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.stage_upsert(701, "idx_count", 2, {2.0F, 2.0F}));
  ASSERT_TRUE(service.commit(701));

  vector_index::index_service::index_config config;
  bool supports_mutations = false;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  ASSERT_TRUE(service.describe_index("idx_count", &config, &supports_mutations,
                                    &entry_count, &committed_entry_count));
  EXPECT_EQ(2U, entry_count);
  EXPECT_EQ(2U, committed_entry_count);
}

TEST(VectorIndexServiceTest, RegisterIndexConfigAppliesDiskAnnTunings) {
#ifndef HAVE_DISKANN
  GTEST_SKIP() << "DiskANN tuning requires HAVE_DISKANN";
#else
  vector_index::index_service service;
  vector_index::index_service::index_config config{
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kDiskAnn, ""};
  config.diskann_max_degree = 48;
  config.diskann_build_complexity = 96;
  config.diskann_search_complexity = 72;

  ASSERT_TRUE(service.register_index("idx_diskann_tuned", config));

  vector_index::index_service::index_config described;
  bool supports_mutations = true;
  size_t entry_count = 99;
  size_t committed_entry_count = 99;
  ASSERT_TRUE(service.describe_index("idx_diskann_tuned", &described,
                                     &supports_mutations, &entry_count,
                                     &committed_entry_count));
  EXPECT_EQ(vector_index::backend_provider::kDiskAnn, described.provider);
  EXPECT_EQ(48U, described.diskann_max_degree);
  EXPECT_EQ(96U, described.diskann_build_complexity);
  EXPECT_EQ(72U, described.diskann_search_complexity);
  EXPECT_TRUE(supports_mutations);
  EXPECT_EQ(0U, entry_count);
  EXPECT_EQ(0U, committed_entry_count);
#endif
}

TEST(VectorIndexServiceTest, RegisterIndexConfigAppliesHnswTunings) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib tuning requires HAVE_HNSWLIB";
  }

  vector_index::index_service service;

  vector_index::index_service::index_config hnsw_config{
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kHnswlib, ""};
  hnsw_config.search_ef = 40;
  hnsw_config.hnsw_m = 12;
  hnsw_config.hnsw_ef_construction = 96;
  hnsw_config.hnsw_build_threads = 2;
  ASSERT_TRUE(service.register_index("idx_hnsw_config", hnsw_config));

  vector_index::index_service::index_config described;
  bool supports_mutations = false;
  size_t entry_count = 99;
  size_t committed_entry_count = 99;
  ASSERT_TRUE(service.describe_index("idx_hnsw_config", &described,
                                     &supports_mutations, &entry_count,
                                     &committed_entry_count));
  EXPECT_EQ(vector_index::backend_provider::kHnswlib, described.provider);
  EXPECT_EQ(40U, described.search_ef);
  EXPECT_EQ(12U, described.hnsw_m);
  EXPECT_EQ(96U, described.hnsw_ef_construction);
  EXPECT_EQ(2U, described.hnsw_build_threads);
  EXPECT_TRUE(supports_mutations);
  EXPECT_EQ(0U, entry_count);
  EXPECT_EQ(0U, committed_entry_count);
}

TEST(VectorIndexServiceTest, RegisterIndexConfigAppliesFaissTunings) {
#ifndef HAVE_FAISS
  GTEST_SKIP() << "Faiss tuning requires HAVE_FAISS";
#else
  vector_index::index_service service;

  vector_index::index_service::index_config faiss_config{
      4, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kFaiss, ""};
  faiss_config.faiss_nlist = 2;
  faiss_config.faiss_nprobe = 1;
  faiss_config.faiss_pq_m = 2;
  faiss_config.faiss_pq_bits = 4;
  faiss_config.faiss_build_threads = 2;
  ASSERT_TRUE(service.register_index("idx_faiss_config", faiss_config));

  vector_index::index_service::index_config described;
  bool supports_mutations = false;
  size_t entry_count = 99;
  size_t committed_entry_count = 99;
  ASSERT_TRUE(service.describe_index("idx_faiss_config", &described,
                                     &supports_mutations, &entry_count,
                                     &committed_entry_count));
  EXPECT_EQ(vector_index::backend_provider::kFaiss, described.provider);
  EXPECT_EQ(2U, described.faiss_nlist);
  EXPECT_EQ(1U, described.faiss_nprobe);
  EXPECT_EQ(2U, described.faiss_pq_m);
  EXPECT_EQ(4U, described.faiss_pq_bits);
  EXPECT_EQ(2U, described.faiss_build_threads);
  EXPECT_TRUE(supports_mutations);
#endif
}

TEST(VectorIndexServiceTest, LoadedSearchApisRejectBulkLoadingAndMissingState) {
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_bulk_guard", 2,
                                               "euclidean", "memory",
                                               "native"));
  ASSERT_TRUE(service.stage_upsert(801, "idx_bulk_guard", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(801));

  std::vector<vector_index::search_result> results;
  std::vector<std::vector<vector_index::search_result>> batch_results;
  EXPECT_FALSE(service.search_loaded("idx_missing", {1.0F, 1.0F}, 1,
                                     &results));
  EXPECT_FALSE(service.search_batch_loaded(
      "idx_missing", {{1.0F, 1.0F}}, 1, &batch_results));
  EXPECT_FALSE(service.search_with_pending_loaded(999, "idx_missing",
                                                 {1.0F, 1.0F}, 1, &results));

  ASSERT_TRUE(
      service.set_lifecycle_info("idx_bulk_guard", "bulk_loading", 2));
  EXPECT_FALSE(service.search_loaded("idx_bulk_guard", {1.0F, 1.0F}, 1,
                                     &results));
  EXPECT_FALSE(service.search_batch_loaded(
      "idx_bulk_guard", {{1.0F, 1.0F}}, 1, &batch_results));

  ASSERT_TRUE(service.stage_upsert(802, "idx_bulk_guard", 2, {2.0F, 2.0F}));
  EXPECT_FALSE(service.search_with_pending_loaded(802, "idx_bulk_guard",
                                                 {2.0F, 2.0F}, 1, &results));
}

TEST(VectorIndexServiceTest, RegisterIndexConfigRejectsInvalidCombinations) {
  vector_index::index_service service;

  EXPECT_FALSE(service.register_index(
      "idx_zero", vector_index::index_service::index_config{
                      0, vector_index::metric_type::kEuclidean,
                      vector_index::backend_mode::kMemory,
                      vector_index::backend_provider::kNative, ""}));
  EXPECT_FALSE(service.register_index(
      "idx_bad_combo", vector_index::index_service::index_config{
                           2, vector_index::metric_type::kEuclidean,
                           vector_index::backend_mode::kMemory,
                           vector_index::backend_provider::kDiskAnn, ""}));
  EXPECT_FALSE(service.register_index_from_strings("idx_bad_opt", 2, "cosine",
                                                "external", "hnswlib"));
  EXPECT_FALSE(service.register_index_from_strings("idx_bad_metric", 2, "l1",
                                                "memory", "native"));
  EXPECT_FALSE(service.register_index_from_strings("idx_bad_mode", 2, "cosine",
                                                "disk", "native"));
  EXPECT_FALSE(service.register_index_from_strings("idx_bad_provider", 2, "cosine",
                                                "memory", "nmslib"));

  vector_index::index_service::index_config bad_search_ef{
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kNative, ""};
  bad_search_ef.search_ef = 32;
  EXPECT_FALSE(service.register_index("idx_bad_search_ef", bad_search_ef));

  vector_index::index_service::index_config bad_hnsw_native{
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kNative, ""};
  bad_hnsw_native.hnsw_m = 16;
  bad_hnsw_native.hnsw_ef_construction = 200;
  EXPECT_FALSE(service.register_index("idx_bad_hnsw_native", bad_hnsw_native));

  vector_index::index_service::index_config bad_faiss_on_hnsw{
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kHnswlib, ""};
  bad_faiss_on_hnsw.faiss_nlist = 8;
  bad_faiss_on_hnsw.faiss_nprobe = 4;
  EXPECT_FALSE(service.register_index("idx_bad_faiss_on_hnsw", bad_faiss_on_hnsw));

  vector_index::index_service::index_config bad_diskann_on_faiss{
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kFaiss, ""};
  bad_diskann_on_faiss.diskann_max_degree = 32;
  bad_diskann_on_faiss.diskann_build_complexity = 64;
  EXPECT_FALSE(service.register_index("idx_bad_diskann_on_faiss",
                                     bad_diskann_on_faiss));

  vector_index::index_service::index_config bad_diskann_search_on_native{
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kNative, ""};
  bad_diskann_search_on_native.diskann_search_complexity = 64;
  EXPECT_FALSE(service.register_index("idx_bad_diskann_search_on_native",
                                     bad_diskann_search_on_native));

  vector_index::index_service::index_config bad_faiss_on_diskann{
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kDiskAnn, ""};
  bad_faiss_on_diskann.faiss_nlist = 8;
  bad_faiss_on_diskann.faiss_nprobe = 4;
  EXPECT_FALSE(service.register_index("idx_bad_faiss_on_diskann",
                                     bad_faiss_on_diskann));

  vector_index::index_service::index_config bad_half_ivf{
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kFaiss, ""};
  bad_half_ivf.faiss_nlist = 8;
  EXPECT_FALSE(service.register_index("idx_bad_half_ivf", bad_half_ivf));

  vector_index::index_service::index_config bad_half_ivf_probe{
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kFaiss, ""};
  bad_half_ivf_probe.faiss_nprobe = 4;
  EXPECT_FALSE(
      service.register_index("idx_bad_half_ivf_probe", bad_half_ivf_probe));

  vector_index::index_service::index_config bad_half_pq_bits{
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kFaiss, ""};
  bad_half_pq_bits.faiss_pq_bits = 8;
  EXPECT_FALSE(
      service.register_index("idx_bad_half_pq_bits", bad_half_pq_bits));
}

TEST(VectorIndexServiceTest, ServiceApiGuardBranchesRejectInvalidState) {
  vector_index::index_service service;
  vector_index::index_service::index_config config{
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kNative, ""};
  bool supports_mutations = true;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;

  EXPECT_FALSE(service.describe_index("missing", nullptr, &supports_mutations,
                                      &entry_count, &committed_entry_count));
  EXPECT_FALSE(service.describe_index("missing", &config, &supports_mutations,
                                      &entry_count, &committed_entry_count));
  EXPECT_FALSE(service.rename_index("", "idx_new"));
  EXPECT_FALSE(service.rename_index("idx_old", ""));
  EXPECT_FALSE(service.begin_bulk_load("missing"));
  EXPECT_FALSE(service.restore_index_config("missing", config));

  vector_index::index_service::index_config zero_dimension = config;
  zero_dimension.dimension = 0;
  EXPECT_FALSE(service.restore_index_config("missing", zero_dimension));

  ASSERT_TRUE(service.register_index_from_strings("idx_guard", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(9801, "idx_guard", 1, {1.0F, 1.0F}));
  EXPECT_FALSE(service.begin_bulk_load("idx_guard"));
  ASSERT_TRUE(service.commit(9801));
  ASSERT_TRUE(service.register_index_from_strings("idx_other", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(9802, "idx_other", 2, {2.0F, 2.0F}));
  EXPECT_TRUE(service.begin_bulk_load("idx_guard"));
  service.rollback(9802);

  vector_index::index_service::index_config mismatched_dimension = config;
  mismatched_dimension.dimension = 3;
  EXPECT_FALSE(service.restore_index_config("idx_guard", mismatched_dimension));

  vector_index::index_service::index_config unsupported_tuning = config;
  unsupported_tuning.search_ef = 32;
  EXPECT_FALSE(service.restore_index_config("idx_guard", unsupported_tuning));

  vector_index::index_service::committed_state invalid_state;
  invalid_state["idx_guard"].emplace(1, vector_index::vector_data{1.0F});
  EXPECT_FALSE(service.restore_committed_state(invalid_state));
}

TEST(VectorIndexServiceTest, CommitAppliesStagedUpserts) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  ASSERT_TRUE(service.stage_upsert(1, "idx_mem", 101, {1.0F, 1.0F}));
  ASSERT_TRUE(service.stage_upsert(1, "idx_mem", 102, {2.0F, 2.0F}));
  EXPECT_EQ(2U, service.pending_change_count(1));

  ASSERT_TRUE(service.commit(1));
  EXPECT_EQ(0U, service.pending_change_count(1));
  ASSERT_TRUE(service.search("idx_mem", {1.1F, 1.1F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(101U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, PendingChangeIndexNamesReturnsUniqueSortedNames) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index_from_strings("idx_b", 2, "euclidean", "memory",
                                               "native"));
  ASSERT_TRUE(service.register_index_from_strings("idx_a", 2, "euclidean", "memory",
                                               "native"));
  ASSERT_TRUE(service.stage_upsert(19, "idx_b", 10, {1.0F, 1.0F}));
  ASSERT_TRUE(service.stage_erase(19, "idx_b", 11));
  ASSERT_TRUE(service.stage_upsert(19, "idx_a", 12, {2.0F, 2.0F}));

  std::vector<std::string> index_names;
  ASSERT_TRUE(service.pending_change_index_names(19, &index_names));
  ASSERT_EQ(2U, index_names.size());
  EXPECT_EQ("idx_a", index_names[0]);
  EXPECT_EQ("idx_b", index_names[1]);

  ASSERT_TRUE(service.pending_change_index_names(20, &index_names));
  EXPECT_TRUE(index_names.empty());
}

TEST(VectorIndexServiceTest, PendingChangeIndexNamesRejectsNullOutput) {
  vector_index::index_service service;
  EXPECT_FALSE(service.pending_change_index_names(1, nullptr));
}

TEST(VectorIndexServiceTest, SnapshotCommittedStateRejectsNullOutput) {
  vector_index::index_service service;
  EXPECT_FALSE(service.snapshot_committed_state(nullptr));
}

TEST(VectorIndexServiceTest, SnapshotPendingChangesRejectsNullOutput) {
  vector_index::index_service service;
  EXPECT_FALSE(service.snapshot_pending_changes(1, nullptr));
}

TEST(VectorIndexServiceTest, SnapshotPendingChangesMissingTxnReturnsEmpty) {
  vector_index::index_service service;
  std::vector<vector_index::index_service::pending_change_snapshot> changes;
  ASSERT_TRUE(service.snapshot_pending_changes(999, &changes));
  EXPECT_TRUE(changes.empty());
}

TEST(VectorIndexServiceTest,
     SnapshotPendingChangeDeltaReturnsNetTouchedRowsOnly) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  ASSERT_TRUE(service.stage_upsert(1, "idx_mem", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(1));

  ASSERT_TRUE(service.stage_upsert(2, "idx_mem", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.stage_upsert(2, "idx_mem", 2, {2.0F, 2.0F}));
  ASSERT_TRUE(service.stage_erase(2, "idx_mem", 2));
  ASSERT_TRUE(service.stage_erase(2, "idx_mem", 3));
  ASSERT_TRUE(service.stage_erase(2, "idx_mem", 1));
  ASSERT_TRUE(service.stage_upsert(2, "idx_mem", 1, {3.0F, 3.0F}));

  std::vector<vector_index::index_service::pending_change_snapshot> changes;
  ASSERT_TRUE(service.snapshot_pending_change_delta(2, &changes));
  ASSERT_EQ(1U, changes.size());
  EXPECT_EQ("idx_mem", changes[0].index_name);
  EXPECT_FALSE(changes[0].erase);
  EXPECT_EQ(1U, changes[0].doc_id);
  EXPECT_EQ((vector_index::vector_data{3.0F, 3.0F}), changes[0].vector);
}

TEST(VectorEntryStoreTest, FindCommittedEntryUsesCachedAndEvictedEntries) {
  std::vector<vector_index_metadata_store::committed_row> rows;
  rows.push_back({"idx_evicted", 1, {1.0F, 1.0F}});
  rows.push_back({"idx_evicted", 2, {2.0F, 2.0F}});
  rows.push_back({"idx_evicted", 7, {7.0F, 7.0F}});
  committed_rows_truth_store store(std::move(rows));
  truth_store_guard guard(&store);

  vector_index::vector_entry_store entry_store;
  vector_index::vector_data vector;
  bool found = true;

  ASSERT_TRUE(entry_store.register_index("idx_cached"));
  ASSERT_TRUE(entry_store.upsert("idx_cached", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(entry_store.find_committed_entry("idx_cached", 1, &vector, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ((vector_index::vector_data{1.0F, 1.0F}), vector);

  vector.clear();
  ASSERT_TRUE(entry_store.find_committed_entry("idx_cached", 2, &vector, &found));
  EXPECT_FALSE(found);
  EXPECT_TRUE(vector.empty());

  ASSERT_TRUE(entry_store.register_index("idx_evicted"));
  ASSERT_TRUE(entry_store.upsert("idx_evicted", 99, {99.0F, 99.0F}));
  ASSERT_TRUE(entry_store.evict_until_under_budget(0));
  ASSERT_TRUE(entry_store.find_committed_entry("idx_evicted", 7, &vector, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ((vector_index::vector_data{7.0F, 7.0F}), vector);
  EXPECT_EQ(1U, store.find_committed_calls());
  EXPECT_EQ(0U, store.for_each_committed_calls());
  EXPECT_EQ(0U, store.load_committed_calls());
}

TEST(VectorIndexServiceTest, PendingStateSnapshotsCoverSavepointRestoreGuards) {
  using pending_change_snapshot =
      vector_index::index_service::pending_change_snapshot;
  using pending_savepoint_snapshot =
      vector_index::index_service::pending_savepoint_snapshot;
  using pending_state_snapshot =
      vector_index::index_service::pending_state_snapshot;

  vector_index::index_service service;
  std::vector<pending_change_snapshot> changes;
  pending_state_snapshot state;

  ASSERT_TRUE(service.register_index_from_strings("idx_state", 2, "euclidean",
                                                  "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(1, "idx_state", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(1));

  EXPECT_FALSE(service.snapshot_pending_state(2, nullptr));
  EXPECT_FALSE(service.snapshot_pending_change_delta(2, nullptr));
  ASSERT_TRUE(service.snapshot_pending_change_delta(999, &changes));
  EXPECT_TRUE(changes.empty());

  ASSERT_TRUE(service.stage_upsert(2, "idx_state", 2, {2.0F, 2.0F}));
  ASSERT_TRUE(service.savepoint(2, "sp1"));
  ASSERT_TRUE(service.stage_upsert(2, "idx_state", 3, {3.0F, 3.0F}));
  ASSERT_TRUE(service.snapshot_pending_state(2, &state));
  ASSERT_EQ(2U, state.changes.size());
  ASSERT_EQ(1U, state.savepoints.size());
  EXPECT_EQ("sp1", state.savepoints[0].name);
  EXPECT_TRUE(service.has_pending_changes_for_index("idx_state"));
  EXPECT_FALSE(service.has_pending_changes_for_index("idx_other"));

  ASSERT_TRUE(service.restore_pending_state(99, state));
  ASSERT_TRUE(service.rollback_to_savepoint(99, "sp1"));
  ASSERT_TRUE(service.commit(99));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_state", {2.0F, 2.0F}, 3, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
  EXPECT_EQ(1U, result[1].doc_id);

  pending_state_snapshot invalid_name_state;
  invalid_name_state.changes = {
      pending_change_snapshot{"idx_state", false, 4, {4.0F, 4.0F}}};
  invalid_name_state.savepoints = {pending_savepoint_snapshot{"", 1}};
  EXPECT_FALSE(service.restore_pending_state(100, invalid_name_state));

  pending_state_snapshot invalid_count_state;
  invalid_count_state.changes = {
      pending_change_snapshot{"idx_state", false, 5, {5.0F, 5.0F}}};
  invalid_count_state.savepoints = {pending_savepoint_snapshot{"sp2", 2}};
  EXPECT_FALSE(service.restore_pending_state(101, invalid_count_state));
}

TEST(VectorIndexServiceTest, RestorePendingChangesClearsTxnAndSavepoints) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  ASSERT_TRUE(service.stage_upsert(55, "idx_mem", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.savepoint(55, "sp1"));
  ASSERT_TRUE(service.restore_pending_changes(55, {}));
  EXPECT_EQ(0U, service.pending_change_count(55));
  EXPECT_FALSE(service.rollback_to_savepoint(55, "sp1"));
}

TEST(VectorIndexServiceTest, RestoreCommittedStateUsesEmptyStateForMissingIndex) {
  vector_index::index_service source;
  vector_index::index_service restored;
  vector_index::index_service::committed_state snapshot;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(source.register_index_from_strings("idx_present", 2, "euclidean",
                                              "memory", "native"));
  ASSERT_TRUE(source.register_index_from_strings("idx_absent", 2, "euclidean",
                                              "memory", "native"));
  ASSERT_TRUE(source.stage_upsert(777, "idx_present", 5, {5.0F, 5.0F}));
  ASSERT_TRUE(source.commit(777));
  ASSERT_TRUE(source.snapshot_committed_state(&snapshot));
  snapshot.erase("idx_absent");

  ASSERT_TRUE(restored.register_index_from_strings("idx_present", 2, "euclidean",
                                                "memory", "native"));
  ASSERT_TRUE(restored.register_index_from_strings("idx_absent", 2, "euclidean",
                                                "memory", "native"));
  ASSERT_TRUE(restored.restore_committed_state(snapshot));

  ASSERT_TRUE(restored.search("idx_present", {5.0F, 5.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  ASSERT_TRUE(restored.search("idx_absent", {1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
}

TEST(VectorIndexServiceTest, RestorePendingChangesRejectsUnknownIndex) {
  vector_index::index_service service;
  std::vector<vector_index::index_service::pending_change_snapshot> changes{
      {"missing", false, 1, {1.0F, 1.0F}}};
  EXPECT_FALSE(service.restore_pending_changes(66, changes));
}

TEST(VectorIndexServiceTest, RestorePendingChangesRejectsDimensionMismatch) {
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  std::vector<vector_index::index_service::pending_change_snapshot> changes{
      {"idx_mem", false, 1, {1.0F, 2.0F, 3.0F}}};
  EXPECT_FALSE(service.restore_pending_changes(67, changes));
}

TEST(VectorIndexServiceTest, RollbackDiscardsStagedChanges) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  ASSERT_TRUE(service.stage_upsert(7, "idx_mem", 101, {1.0F, 1.0F}));
  EXPECT_EQ(1U, service.pending_change_count(7));

  service.rollback(7);
  EXPECT_EQ(0U, service.pending_change_count(7));
  ASSERT_TRUE(service.search("idx_mem", {1.0F, 1.0F}, 5, &result));
  EXPECT_TRUE(result.empty());
}

TEST(VectorIndexServiceTest, SavepointRollbackTruncatesPendingChanges) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  ASSERT_TRUE(service.stage_upsert(9, "idx_mem", 100, {10.0F, 10.0F}));
  ASSERT_TRUE(service.savepoint(9, "sp1"));
  ASSERT_TRUE(service.stage_upsert(9, "idx_mem", 101, {11.0F, 11.0F}));
  ASSERT_TRUE(service.stage_upsert(9, "idx_mem", 102, {12.0F, 12.0F}));
  EXPECT_EQ(3U, service.pending_change_count(9));

  ASSERT_TRUE(service.rollback_to_savepoint(9, "sp1"));
  EXPECT_EQ(1U, service.pending_change_count(9));
  ASSERT_TRUE(service.commit(9));

  ASSERT_TRUE(service.search("idx_mem", {10.0F, 10.0F}, 10, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(100U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, SavepointReleaseRemovesMarker) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  ASSERT_TRUE(service.savepoint(10, "sp1"));
  ASSERT_TRUE(service.release_savepoint(10, "sp1"));
  EXPECT_FALSE(service.rollback_to_savepoint(10, "sp1"));
}

TEST(VectorIndexServiceTest, SavepointRejectsEmptyAndMissingNames) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  EXPECT_FALSE(service.savepoint(12, ""));
  ASSERT_TRUE(service.savepoint(12, "sp1"));
  EXPECT_FALSE(service.rollback_to_savepoint(12, ""));
  EXPECT_FALSE(service.rollback_to_savepoint(12, "missing"));
  EXPECT_FALSE(service.release_savepoint(12, ""));
}

TEST(VectorIndexServiceTest, ReleaseSavepointRejectsMissingMarker) {
  vector_index::index_service service;

  EXPECT_FALSE(service.release_savepoint(10, "missing"));
  ASSERT_TRUE(service.savepoint(10, "sp1"));
  EXPECT_FALSE(service.release_savepoint(10, "sp2"));
}

TEST(VectorIndexServiceTest, SavepointReplaceNameTracksLatestPosition) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  ASSERT_TRUE(service.stage_upsert(11, "idx_mem", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.savepoint(11, "sp"));
  ASSERT_TRUE(service.stage_upsert(11, "idx_mem", 2, {2.0F, 2.0F}));
  ASSERT_TRUE(service.savepoint(11, "sp"));
  ASSERT_TRUE(service.stage_upsert(11, "idx_mem", 3, {3.0F, 3.0F}));
  ASSERT_TRUE(service.rollback_to_savepoint(11, "sp"));
  ASSERT_TRUE(service.commit(11));

  ASSERT_TRUE(service.search("idx_mem", {0.0F, 0.0F}, 10, &result));
  ASSERT_EQ(2U, result.size());
}

TEST(VectorIndexServiceTest, StageRejectsUnknownIndexAndWrongDimension) {
  vector_index::index_service service;

  EXPECT_FALSE(service.stage_upsert(1, "missing", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  EXPECT_FALSE(service.stage_upsert(1, "idx_mem", 1, {1.0F, 2.0F, 3.0F}));
}

TEST(VectorIndexServiceTest, CommitEraseRemovesEntry) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  ASSERT_TRUE(service.stage_upsert(1, "idx_mem", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(1));
  ASSERT_TRUE(service.stage_erase(2, "idx_mem", 1));
  ASSERT_TRUE(service.commit(2));
  ASSERT_TRUE(service.search("idx_mem", {1.0F, 1.0F}, 2, &result));
  EXPECT_TRUE(result.empty());
}

TEST(VectorIndexServiceTest, SearchFailsForUnknownIndex) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;
  EXPECT_FALSE(service.search("missing", {1.0F, 1.0F}, 1, &result));
}

TEST(VectorIndexServiceTest, StageEraseRejectsUnknownIndex) {
  vector_index::index_service service;
  EXPECT_FALSE(service.stage_erase(1, "missing", 1));
}

TEST(VectorIndexServiceTest, CommitWithoutPendingChangesReturnsTrue) {
  vector_index::index_service service;
  EXPECT_TRUE(service.commit(999));
}

TEST(VectorIndexServiceTest, CommitFailsWhenIndexDroppedAfterStaging) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  ASSERT_TRUE(service.stage_upsert(42, "idx_mem", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.drop_index("idx_mem"));
  EXPECT_FALSE(service.commit(42));
  EXPECT_EQ(1U, service.pending_change_count(42));
}

TEST(VectorIndexServiceTest, CommitFailsWhenBackendApplyReturnsFalse) {
  vector_index::index_service service;

  ASSERT_TRUE(service.register_index("idx_fail",
                                    std::make_unique<NonApplyingBackend>()));
  ASSERT_TRUE(service.stage_upsert(7, "idx_fail", 1, {1.0F, 1.0F}));
  EXPECT_FALSE(service.commit(7));
  EXPECT_EQ(1U, service.pending_change_count(7));
}

TEST(VectorIndexServiceTest, CommitPreflightRejectsNonWritableAndKeepsAtomicity) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index(
      "idx_mem", std::make_unique<vector_index::memory_backend>(
                     2, vector_index::metric_type::kEuclidean)));
  ASSERT_TRUE(
      service.register_index("idx_ext", std::make_unique<NonWritableBackend>()));

  ASSERT_TRUE(service.stage_upsert(10, "idx_mem", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(service.stage_upsert(10, "idx_ext", 2, {2.0F, 2.0F}));
  EXPECT_EQ(2U, service.pending_change_count(10));

  EXPECT_FALSE(service.commit(10));
  EXPECT_EQ(2U, service.pending_change_count(10));
  ASSERT_TRUE(service.search("idx_mem", {1.0F, 1.0F}, 10, &result));
  EXPECT_TRUE(result.empty());

  vector_index::index_service::index_config config;
  bool supports_mutations = true;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  std::string lifecycle_state;
  uint64_t lifecycle_version = 0;
  uint32_t last_error_code = 0;
  uint64_t last_error_ts = 0;
  ASSERT_TRUE(service.describe_index("idx_ext", &config, &supports_mutations,
                                    &entry_count, &committed_entry_count,
                                    &lifecycle_state, &lifecycle_version,
                                    &last_error_code, &last_error_ts));
  EXPECT_EQ("failed", lifecycle_state);
  EXPECT_EQ(1005U, last_error_code);
  EXPECT_GT(last_error_ts, 0U);

  service.rollback(10);
  EXPECT_EQ(0U, service.pending_change_count(10));
}

TEST(VectorIndexServiceTest, StagedCommitPlanCoversGuardAndFailureBranches) {
  vector_index::index_service service;
  vector_index::index_service::commit_build_plan plan;

  EXPECT_FALSE(service.snapshot_commit_build_plan(1, nullptr));
  EXPECT_TRUE(service.snapshot_commit_build_plan(999, &plan));
  EXPECT_TRUE(plan.changes.empty());
  EXPECT_TRUE(plan.rebuilds.empty());
  EXPECT_TRUE(service.build_commit_backends(&plan));
  EXPECT_TRUE(service.apply_commit_build_plan(999, &plan));
  EXPECT_FALSE(service.build_commit_backends(nullptr));
  EXPECT_FALSE(service.apply_commit_build_plan(999, nullptr));

  vector_index::index_service::backend_ptr runtime;
  EXPECT_FALSE(service.snapshot_search_backend_loaded("missing", {1.0F, 1.0F},
                                                      nullptr));
  EXPECT_FALSE(service.snapshot_search_backend_loaded("missing", {1.0F, 1.0F},
                                                      &runtime));

  ASSERT_TRUE(service.register_index_from_strings("idx_guard", 2, "euclidean",
                                                  "memory", "native"));
  EXPECT_FALSE(service.snapshot_search_backend_loaded("idx_guard", {1.0F},
                                                      &runtime));
  ASSERT_TRUE(service.snapshot_search_backend_loaded("idx_guard", {1.0F, 1.0F},
                                                     &runtime));
  EXPECT_NE(nullptr, runtime.get());
  std::vector<std::vector<vector_index::search_result>> batch_results;
  EXPECT_FALSE(service.search_batch_loaded("idx_guard", {{1.0F, 1.0F}}, 1,
                                           nullptr));
  EXPECT_FALSE(service.search_batch_loaded("idx_guard", {{1.0F}}, 1,
                                           &batch_results));
  ASSERT_TRUE(service.begin_bulk_load("idx_guard"));
  EXPECT_FALSE(service.snapshot_search_backend_loaded("idx_guard", {1.0F, 1.0F},
                                                      &runtime));
  std::vector<vector_index::search_result> result;
  EXPECT_FALSE(service.search_loaded("idx_guard", {1.0F, 1.0F}, 1, &result));
  EXPECT_FALSE(service.search_batch_loaded("idx_guard", {{1.0F, 1.0F}}, 1,
                                           &batch_results));

  vector_index::index_service::commit_build_plan invalid_backend_plan;
  auto &invalid_rebuild = invalid_backend_plan.rebuilds.emplace_back();
  invalid_rebuild.index_name = "idx_invalid_backend";
  invalid_rebuild.config = {2, vector_index::metric_type::kEuclidean,
                            vector_index::backend_mode::kMemory,
                            vector_index::backend_provider::kDiskAnn, ""};
  EXPECT_FALSE(service.build_commit_backends(&invalid_backend_plan));

  vector_index::index_service::commit_build_plan invalid_entries_plan;
  ASSERT_TRUE(service.register_index_from_strings("idx_invalid_entries", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  auto &invalid_entries_rebuild = invalid_entries_plan.rebuilds.emplace_back();
  invalid_entries_rebuild.index_name = "idx_invalid_entries";
  invalid_entries_rebuild.config = {2, vector_index::metric_type::kEuclidean,
                                    vector_index::backend_mode::kMemory,
                                    vector_index::backend_provider::kNative, ""};
  invalid_entries_rebuild.changes.push_back(
      {"idx_invalid_entries", false, 1, {1.0F}});
  EXPECT_FALSE(service.build_commit_backends(&invalid_entries_plan));
}

TEST(VectorIndexServiceTest,
     StagedCommitPlanRejectsStalePendingAndNonWritableBackend) {
  vector_index::index_service service;
  vector_index::index_service::commit_build_plan plan;

  ASSERT_TRUE(service.register_index_from_strings("idx_stale_pending", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.stage_upsert(70, "idx_stale_pending", 1,
                                   {1.0F, 1.0F}));
  ASSERT_TRUE(service.snapshot_commit_build_plan(70, &plan));
  ASSERT_TRUE(service.build_commit_backends(&plan));
  ASSERT_TRUE(service.stage_upsert(70, "idx_stale_pending", 2,
                                   {2.0F, 2.0F}));
  EXPECT_FALSE(service.apply_commit_build_plan(70, &plan));
  EXPECT_EQ(2U, service.pending_change_count(70));
  service.rollback(70);

  ASSERT_TRUE(service.register_index_from_strings("idx_staged_drop_before_plan", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.stage_upsert(72, "idx_staged_drop_before_plan", 1,
                                   {1.0F, 1.0F}));
  ASSERT_TRUE(service.drop_index("idx_staged_drop_before_plan"));
  EXPECT_FALSE(service.snapshot_commit_build_plan(72, &plan));
  service.rollback(72);

  ASSERT_TRUE(service.register_index_from_strings("idx_staged_drop_before_apply",
                                                  2, "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.stage_upsert(73, "idx_staged_drop_before_apply", 1,
                                   {1.0F, 1.0F}));
  ASSERT_TRUE(service.snapshot_commit_build_plan(73, &plan));
  ASSERT_TRUE(service.build_commit_backends(&plan));
  ASSERT_TRUE(service.drop_index("idx_staged_drop_before_apply"));
  EXPECT_FALSE(service.apply_commit_build_plan(73, &plan));
  service.rollback(73);

  ASSERT_TRUE(service.register_index("idx_staged_apply_fail",
                                     std::make_unique<NonApplyingBackend>()));
  ASSERT_TRUE(service.stage_upsert(74, "idx_staged_apply_fail", 1,
                                   {1.0F, 1.0F}));
  ASSERT_TRUE(service.snapshot_commit_build_plan(74, &plan));
  ASSERT_TRUE(service.build_commit_backends(&plan));
  EXPECT_FALSE(service.apply_commit_build_plan(74, &plan));
  EXPECT_EQ(1U, service.pending_change_count(74));
  service.rollback(74);

  ASSERT_TRUE(service.register_index(
      "idx_staged_non_writable", std::make_unique<NonWritableBackend>()));
  ASSERT_TRUE(service.stage_upsert(71, "idx_staged_non_writable", 1,
                                   {1.0F, 1.0F}));
  EXPECT_FALSE(service.snapshot_commit_build_plan(71, &plan));
  EXPECT_EQ(1U, service.pending_change_count(71));

  vector_index::index_service::index_config config;
  bool supports_mutations = true;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  std::string lifecycle_state;
  uint32_t last_error_code = 0;
  ASSERT_TRUE(service.describe_index(
      "idx_staged_non_writable", &config, &supports_mutations, &entry_count,
      &committed_entry_count, &lifecycle_state, nullptr, &last_error_code));
  EXPECT_FALSE(supports_mutations);
  EXPECT_EQ("failed", lifecycle_state);
  EXPECT_EQ(1005U, last_error_code);
}

TEST(VectorIndexServiceTest,
     StagedCommitPlanExternalFaissRebuildDetectsStaleCommittedState) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_service_staged_faiss_t";
  std::filesystem::remove_all(root);
  faiss_snapshot_root_guard faiss_root(root);

  vector_index::index_service service;
  if (!service.register_index_from_strings("idx_staged_faiss", 2, "euclidean",
                                           "external", "faiss")) {
    GTEST_SKIP() << "FAISS external backend unavailable in current build";
  }

  ASSERT_TRUE(service.stage_upsert(80, "idx_staged_faiss", 1,
                                   {1.0F, 1.0F}));
  ASSERT_TRUE(service.stage_upsert(80, "idx_staged_faiss", 2,
                                   {2.0F, 2.0F}));
  ASSERT_TRUE(service.commit(80));

  ASSERT_TRUE(service.stage_erase(81, "idx_staged_faiss", 1));
  ASSERT_TRUE(service.stage_upsert(81, "idx_staged_faiss", 3,
                                   {3.0F, 3.0F}));
  vector_index::index_service::commit_build_plan stale_plan;
  ASSERT_TRUE(service.snapshot_commit_build_plan(81, &stale_plan));
  ASSERT_EQ(1U, stale_plan.rebuilds.size());
  ASSERT_TRUE(service.build_commit_backends(&stale_plan));

  ASSERT_TRUE(service.stage_upsert(82, "idx_staged_faiss", 4,
                                   {4.0F, 4.0F}));
  ASSERT_TRUE(service.commit(82));
  EXPECT_FALSE(service.apply_commit_build_plan(81, &stale_plan));
  EXPECT_EQ(2U, service.pending_change_count(81));

  vector_index::index_service::commit_build_plan fresh_plan;
  ASSERT_TRUE(service.snapshot_commit_build_plan(81, &fresh_plan));
  ASSERT_EQ(1U, fresh_plan.rebuilds.size());
  ASSERT_TRUE(service.build_commit_backends(&fresh_plan));
  ASSERT_TRUE(service.apply_commit_build_plan(81, &fresh_plan));
  EXPECT_EQ(0U, service.pending_change_count(81));

  vector_index::committed_state state;
  ASSERT_TRUE(service.snapshot_committed_state(&state));
  const auto index_it = state.find("idx_staged_faiss");
  ASSERT_NE(state.end(), index_it);
  EXPECT_EQ(0U, index_it->second.count(1));
  EXPECT_EQ(1U, index_it->second.count(2));
  EXPECT_EQ(1U, index_it->second.count(3));
  EXPECT_EQ(1U, index_it->second.count(4));

  std::filesystem::remove_all(root);
}

}  // namespace vector_index_service_unittest
