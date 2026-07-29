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

#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_service.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_load_file.h"
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

std::string find_standalone_segment_file(const std::string &root) {
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
    if (entry.path().extension().string() == ".vseg") {
      return entry.path().string();
    }
  }
  return "";
}

std::string find_standalone_file_with_extension(const std::string &root,
                                                const std::string &extension) {
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
    if (entry.path().extension().string() == extension) {
      return entry.path().string();
    }
  }
  return "";
}

std::vector<std::string> split_tab_fields(const std::string &line) {
  std::vector<std::string> fields;
  size_t begin = 0;
  while (begin <= line.size()) {
    const size_t tab = line.find('\t', begin);
    if (tab == std::string::npos) {
      fields.push_back(line.substr(begin));
      break;
    }
    fields.push_back(line.substr(begin, tab - begin));
    begin = tab + 1;
  }
  return fields;
}

std::string find_raw_segment_file_from_manifest(const std::string &root,
                                                const std::string &extension) {
  const std::string manifest = find_manifest_file(root, "manifest.v1");
  if (manifest.empty()) return "";
  std::ifstream file(manifest);
  std::string line;
  while (std::getline(file, line)) {
    if (!has_prefix(line, "segment_raw_fbin\t")) continue;
    const std::vector<std::string> fields = split_tab_fields(line);
    if (fields.size() < 7) return "";
    const size_t field_index = extension == ".fbin" ? 1 : 2;
    return (std::filesystem::path(manifest).parent_path() /
            fields[field_index])
        .string();
  }
  return "";
}

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

void write_binary_bytes(const std::string &path, const std::string &payload) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  ASSERT_TRUE(file.good());
}

std::unordered_map<uint64_t, vector_index::vector_data> collect_entries(
    vector_index::standalone_entry_store *store,
    const std::string &index_name) {
  std::unordered_map<uint64_t, vector_index::vector_data> entries;
  EXPECT_TRUE(store->for_each_entry(
      index_name, [&entries](uint64_t doc_id,
                             const vector_index::vector_data &vector) {
        entries.emplace(doc_id, vector);
        return true;
      }));
  return entries;
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

class BoolGuard {
 public:
  BoolGuard(bool *value, bool replacement)
      : m_value(value), m_original(*value) {
    *m_value = replacement;
  }

  ~BoolGuard() { *m_value = m_original; }

 private:
  bool *m_value;
  bool m_original;
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

class RawSegmentCountingBackend final : public vector_index::backend {
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

  bool rebuild_from_committed_entries_from_reader(
      const vector_index::committed_entry_reader &reader) override {
    ++committed_reader_calls;
    return reader([this](uint64_t doc_id,
                         const vector_index::vector_data &vector) {
      streamed_entries.emplace(doc_id, vector);
      return true;
    });
  }

  bool rebuild_from_raw_segments(
      const vector_index::raw_vector_segment_reader &reader) override {
    ++raw_segment_reader_calls;
    return reader([this](const vector_index::raw_vector_segment &segment) {
      raw_segment_doc_counts.push_back(segment.row_count);
      return !reject_raw_segments;
    });
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
  bool supports_mutations() const override { return false; }

  size_t committed_reader_calls{0};
  size_t raw_segment_reader_calls{0};
  bool reject_raw_segments{false};
  std::vector<size_t> raw_segment_doc_counts;
  std::unordered_map<uint64_t, vector_index::vector_data> streamed_entries;
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

TEST(VectorEntryStoreTest, CoversGuardNoopAndMissingTruthStoreReadThrough) {
  vector_index::vector_entry_store store;
  vector_index::vector_data vector;
  bool found = true;

  ASSERT_TRUE(store.register_index("idx_empty"));
  ASSERT_TRUE(store.clear_index("idx_empty"));
  EXPECT_EQ(0U, store.entry_count("idx_empty"));
  EXPECT_EQ(0U, store.memory_bytes("idx_empty"));

  EXPECT_FALSE(store.find_committed_entry("idx_empty", 1, nullptr, &found));
  EXPECT_FALSE(store.find_committed_entry("idx_empty", 1, &vector, nullptr));
  EXPECT_FALSE(store.find_committed_entry("missing", 1, &vector, &found));
  ASSERT_TRUE(store.find_committed_entry("idx_empty", 1, &vector, &found));
  EXPECT_FALSE(found);
  EXPECT_TRUE(vector.empty());

  ASSERT_TRUE(store.upsert("idx_empty", 1, {1.0F, 1.0F}));
  ASSERT_TRUE(store.evict_until_under_budget(0));
  EXPECT_FALSE(store.find_committed_entry("idx_empty", 1, &vector, &found));
  EXPECT_FALSE(store.for_each_committed_entry(
      "idx_empty", [](uint64_t, const vector_index::vector_data &) {
        return true;
      }));
}

TEST(VectorIndexServiceTest, StageUpsertSpillsPendingPayloadOverBudget) {
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

  config.diskann_build_mode_specified = true;
  config.diskann_build_mode_value = vector_index::diskann_build_mode::kSerial;
  EXPECT_FALSE(service.register_index("idx_native_diskann_mode", config));
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

TEST(VectorIndexServiceTest, DirectMutationsRequireStandaloneBulkRebuild) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  EXPECT_FALSE(service.set_index_consistency_mode(
      "missing_direct", vector_index::index_consistency_mode::kStandalone));
  EXPECT_FALSE(service.direct_upsert("missing_direct", 11, {1.0F, 0.0F}));
  EXPECT_FALSE(service.direct_erase("missing_direct", 11));

  ASSERT_TRUE(service.register_index_from_strings("idx_standalone_direct", 2,
                                                 "euclidean", "memory",
                                                 "native"));
  EXPECT_FALSE(service.direct_upsert("idx_standalone_direct", 11,
                                     {1.0F, 0.0F}));
  EXPECT_FALSE(service.direct_erase("idx_standalone_direct", 11));

  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_standalone_direct",
      vector_index::index_consistency_mode::kStandalone));
  EXPECT_FALSE(service.direct_upsert("idx_standalone_direct", 12, {1.0F}));
  EXPECT_TRUE(service.direct_upsert("idx_standalone_direct", 11,
                                    {1.0F, 0.0F}));

  vector_index::index_service::index_config config;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  std::string lifecycle_state;
  ASSERT_TRUE(service.describe_index("idx_standalone_direct", &config, nullptr,
                                     &entry_count, &committed_entry_count,
                                     &lifecycle_state));
  EXPECT_EQ(vector_index::index_consistency_mode::kStandalone,
            config.consistency_mode);
  EXPECT_EQ(0U, entry_count);
  EXPECT_EQ(0U, committed_entry_count);
  EXPECT_EQ("bulk_loading", lifecycle_state);

  EXPECT_FALSE(service.search("idx_standalone_direct", {1.0F, 0.0F}, 1,
                              &result));
  ASSERT_TRUE(service.rebuild_index("idx_standalone_direct"));
  ASSERT_TRUE(service.search("idx_standalone_direct", {1.0F, 0.0F}, 1,
                             &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  EXPECT_TRUE(service.direct_erase("idx_standalone_direct", 11));
  EXPECT_FALSE(service.search("idx_standalone_direct", {1.0F, 0.0F}, 1,
                              &result));
  ASSERT_TRUE(service.rebuild_index("idx_standalone_direct"));
  ASSERT_TRUE(service.search("idx_standalone_direct", {1.0F, 0.0F}, 1,
                             &result));
  EXPECT_TRUE(result.empty());
}

TEST(VectorIndexServiceTest, BulkLoadReaderFailureDoesNotPublishRows) {
  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_bulk_load_fail", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_bulk_load_fail",
      vector_index::index_consistency_mode::kStandalone));

  vector_index::index_service::bulk_load_options options;
  options.rebuild_after_load = true;
  std::string error;
  EXPECT_FALSE(service.bulk_upsert_from_reader(
      "idx_bulk_load_fail",
      [](const vector_index::index_service::bulk_load_visitor &visitor,
         std::string *reader_error) {
        const float row[] = {1.0F, 0.0F};
        if (!visitor(11, row, 2)) return false;
        if (reader_error != nullptr) *reader_error = "synthetic read failure";
        return false;
      },
      options, &error));
  EXPECT_EQ("synthetic read failure", error);
  EXPECT_EQ(0U, service.standalone_segment_count("idx_bulk_load_fail"));
  EXPECT_EQ(0U,
            service.standalone_ingest_memory_bytes("idx_bulk_load_fail"));

  ASSERT_TRUE(service.rebuild_index("idx_bulk_load_fail"));
  ASSERT_TRUE(service.search("idx_bulk_load_fail", {1.0F, 0.0F}, 1,
                             &result));
  EXPECT_TRUE(result.empty());
}

TEST(VectorIndexServiceTest, BulkLoadDuplicatePolicyAndReplace) {
  const std::string root =
      std::string(testing::TempDir()) + "/bulk_load_replace_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::index_service service;
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(service.register_index_from_strings("idx_bulk_load_replace", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_bulk_load_replace",
      vector_index::index_consistency_mode::kStandalone));

  vector_index::index_service::bulk_load_options options;
  options.rebuild_after_load = true;
  std::string error;
  ASSERT_TRUE(service.bulk_upsert_from_reader(
      "idx_bulk_load_replace",
      [](const vector_index::index_service::bulk_load_visitor &visitor,
         std::string *) {
        const float row[] = {1.0F, 0.0F};
        return visitor(7, row, 2);
      },
      options, &error));

  EXPECT_FALSE(service.bulk_upsert_from_reader(
      "idx_bulk_load_replace",
      [](const vector_index::index_service::bulk_load_visitor &visitor,
         std::string *) {
        const float row[] = {0.0F, 1.0F};
        return visitor(7, row, 2);
      },
      options, &error));
  EXPECT_NE(std::string::npos, error.find("duplicate doc_id"));

  ASSERT_TRUE(service.search("idx_bulk_load_replace", {1.0F, 0.0F}, 1,
                             &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);

  options.replace_duplicates = true;
  ASSERT_TRUE(service.bulk_upsert_from_reader(
      "idx_bulk_load_replace",
      [](const vector_index::index_service::bulk_load_visitor &visitor,
         std::string *) {
        const float first[] = {0.0F, 1.0F};
        const float second[] = {0.0F, 2.0F};
        return visitor(7, first, 2) && visitor(7, second, 2);
      },
      options, &error));
  ASSERT_TRUE(service.search("idx_bulk_load_replace", {0.0F, 2.0F}, 1,
                             &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);
}

TEST(VectorIndexServiceTest, TransactionalRawFileBulkLoadPublishesRows) {
  const std::string root =
      std::string(testing::TempDir()) + "/transactional_raw_bulk_load_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {1.0F, 0.0F, 0.0F, 1.0F});
  write_raw_docid_file(docid_path, {31, 42});

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_bulk_raw_txn", 2,
                                                  "euclidean", "memory",
                                                  "native"));

  vector_index::index_service::bulk_load_options options;
  uint64_t loaded_rows = 0;
  std::string error;
  ASSERT_TRUE(service.bulk_upsert_from_raw_files(
      "idx_bulk_raw_txn", vector_path, docid_path, options, &loaded_rows,
      &error))
      << error;
  EXPECT_EQ(2U, loaded_rows);
  EXPECT_EQ(0U, service.standalone_raw_segment_count("idx_bulk_raw_txn"));

  vector_index::index_service::index_config config;
  bool supports_mutations = false;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  std::string lifecycle_state;
  ASSERT_TRUE(service.describe_index("idx_bulk_raw_txn", &config,
                                     &supports_mutations, &entry_count,
                                     &committed_entry_count, &lifecycle_state));
  EXPECT_TRUE(supports_mutations);
  EXPECT_EQ(2U, entry_count);
  EXPECT_EQ(2U, committed_entry_count);
  EXPECT_EQ("ready", lifecycle_state);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_bulk_raw_txn", {0.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(42U, result[0].doc_id);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, BulkLoadValidationCoversBusinessErrorPaths) {
  const std::string root =
      std::string(testing::TempDir()) + "/bulk_load_validation_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::index_service service;
  vector_index::index_service::bulk_load_options options;
  std::string error;
  uint64_t loaded_rows = 99;

  EXPECT_FALSE(service.bulk_upsert_from_reader(
      "missing_bulk_reader",
      [](const vector_index::index_service::bulk_load_visitor &, std::string *) {
        return true;
      },
      options, nullptr));
  EXPECT_FALSE(service.bulk_upsert_from_raw_files(
      "missing_bulk_raw", root + "/missing.fbin", "", options, &loaded_rows,
      nullptr));
  EXPECT_EQ(0U, loaded_rows);

  ASSERT_TRUE(service.register_index_from_strings("idx_bulk_transactional", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  EXPECT_TRUE(service.bulk_upsert_from_reader(
      "idx_bulk_transactional",
      [](const vector_index::index_service::bulk_load_visitor &, std::string *) {
        return true;
      },
      options, &error));
  EXPECT_TRUE(error.empty());
  vector_index::index_service::index_config config;
  size_t entry_count = 99;
  size_t committed_entry_count = 99;
  std::string lifecycle_state;
  ASSERT_TRUE(service.describe_index(
      "idx_bulk_transactional", &config, nullptr, &entry_count,
      &committed_entry_count, &lifecycle_state));
  EXPECT_EQ(0U, entry_count);
  EXPECT_EQ(0U, committed_entry_count);
  EXPECT_EQ("ready", lifecycle_state);

  ASSERT_TRUE(service.register_index_from_strings("idx_bulk_validate", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_bulk_validate", vector_index::index_consistency_mode::kStandalone));

  EXPECT_FALSE(service.bulk_upsert_from_reader("idx_bulk_validate", {}, options,
                                               &error));
  EXPECT_EQ("LOAD VECTOR DATA reader is empty", error);

  EXPECT_FALSE(service.bulk_upsert_from_raw_files("idx_bulk_validate", "", "",
                                                  options, &loaded_rows,
                                                  &error));
  EXPECT_EQ("LOAD VECTOR DATA vector file is empty", error);

  EXPECT_FALSE(service.bulk_upsert_from_reader(
      "idx_bulk_validate",
      [](const vector_index::index_service::bulk_load_visitor &visitor,
         std::string *) { return visitor(1, nullptr, 2); },
      options, &error));
  EXPECT_EQ("LOAD VECTOR DATA returned a null row", error);

  EXPECT_FALSE(service.bulk_upsert_from_reader(
      "idx_bulk_validate",
      [](const vector_index::index_service::bulk_load_visitor &, std::string *) {
        return false;
      },
      options, &error));
  EXPECT_EQ("LOAD VECTOR DATA reader failed", error);

  EXPECT_FALSE(service.bulk_upsert_from_reader(
      "idx_bulk_validate",
      [](const vector_index::index_service::bulk_load_visitor &visitor,
         std::string *) {
        const float row[] = {1.0F};
        return visitor(1, row, 1);
      },
      options, &error));
  EXPECT_EQ("LOAD VECTOR DATA dimension mismatch", error);

  EXPECT_FALSE(service.bulk_upsert_from_reader(
      "idx_bulk_validate",
      [](const vector_index::index_service::bulk_load_visitor &visitor,
         std::string *) {
        const float first[] = {1.0F, 0.0F};
        const float second[] = {0.0F, 1.0F};
        return visitor(2, first, 2) && visitor(2, second, 2);
      },
      options, &error));
  EXPECT_EQ("LOAD VECTOR DATA duplicate doc_id in file", error);

  options.replace_duplicates = true;
  ASSERT_TRUE(service.bulk_upsert_from_reader(
      "idx_bulk_validate",
      [](const vector_index::index_service::bulk_load_visitor &visitor,
         std::string *) {
        const float first[] = {1.0F, 0.0F};
        const float second[] = {0.0F, 1.0F};
        return visitor(2, first, 2) && visitor(2, second, 2);
      },
      options, &error))
      << error;

  options.replace_duplicates = false;
  EXPECT_FALSE(service.bulk_upsert_from_reader(
      "idx_bulk_validate",
      [](const vector_index::index_service::bulk_load_visitor &visitor,
         std::string *) {
        const float row[] = {1.0F, 0.0F};
        return visitor(2, row, 2);
      },
      options, &error));
  EXPECT_EQ("LOAD VECTOR DATA duplicate doc_id in index", error);

  const std::string duplicate_fbin = root + "/duplicate_docids.fbin";
  const std::string duplicate_docids = root + "/duplicate_docids.u64";
  write_raw_fbin_file(duplicate_fbin, 2, 2, {1.0F, 0.0F, 0.0F, 1.0F});
  write_raw_docid_file(duplicate_docids, {10, 10});
  EXPECT_FALSE(service.bulk_upsert_from_raw_files(
      "idx_bulk_validate", duplicate_fbin, duplicate_docids, options,
      &loaded_rows, &error));
  EXPECT_EQ("LOAD VECTOR DATA duplicate doc_id in file", error);

  EXPECT_FALSE(service.bulk_upsert_from_raw_files(
      "idx_bulk_validate", root + "/missing_raw_file.fbin", "", options,
      &loaded_rows, nullptr));

  const std::string existing_fbin = root + "/existing_docid.fbin";
  const std::string existing_docids = root + "/existing_docid.u64";
  write_raw_fbin_file(existing_fbin, 1, 2, {2.0F, 0.0F});
  write_raw_docid_file(existing_docids, {2});
  EXPECT_FALSE(service.bulk_upsert_from_raw_files(
      "idx_bulk_validate", existing_fbin, existing_docids, options,
      &loaded_rows, &error));
  EXPECT_EQ("LOAD VECTOR DATA duplicate doc_id in index", error);

  options.replace_duplicates = true;
  EXPECT_TRUE(service.bulk_upsert_from_raw_files(
      "idx_bulk_validate", existing_fbin, existing_docids, options, nullptr,
      &error))
      << error;
  options.replace_duplicates = false;

  ASSERT_TRUE(service.register_index_from_strings("idx_bulk_pending", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_bulk_pending", vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(service.stage_upsert(123, "idx_bulk_pending", 99, {1.0F, 0.0F}));
  EXPECT_FALSE(service.bulk_upsert_from_reader(
      "idx_bulk_pending",
      [](const vector_index::index_service::bulk_load_visitor &, std::string *) {
        return true;
      },
      options, &error));
  EXPECT_EQ("vector index has pending changes", error);
  EXPECT_FALSE(service.bulk_upsert_from_raw_files(
      "idx_bulk_pending", existing_fbin, existing_docids, options,
      &loaded_rows, &error));
  EXPECT_EQ("vector index has pending changes", error);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, BulkLoadRawSegmentManifestRoundTrip) {
  const std::string root =
      std::string(testing::TempDir()) + "/bulk_load_raw_segment_round_trip_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {1.0F, 0.0F, 0.0F, 1.0F});
  write_raw_docid_file(docid_path, {11, 22});

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_raw_round_trip", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_raw_round_trip",
      vector_index::index_consistency_mode::kStandalone));

  vector_index::index_service::bulk_load_options options;
  uint64_t loaded_rows = 0;
  std::string error;
  ASSERT_TRUE(service.bulk_upsert_from_raw_files(
      "idx_raw_round_trip", vector_path, docid_path, options, &loaded_rows,
      &error))
      << error;
  EXPECT_EQ(2U, loaded_rows);
  EXPECT_EQ(1U, service.standalone_segment_count("idx_raw_round_trip"));
  EXPECT_EQ(1U, service.standalone_raw_segment_count("idx_raw_round_trip"));
  EXPECT_GT(service.standalone_raw_segment_bytes("idx_raw_round_trip"), 0U);

  const std::string manifest = find_manifest_file(root, "manifest.v1");
  ASSERT_FALSE(manifest.empty());
  {
    std::ifstream file(manifest);
    ASSERT_TRUE(file.is_open());
    const std::string manifest_text((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());
    EXPECT_NE(std::string::npos, manifest_text.find("segment_raw_fbin"));
    EXPECT_NE(std::string::npos, manifest_text.find(".fbin"));
    EXPECT_NE(std::string::npos, manifest_text.find(".u64"));
  }
  EXPECT_FALSE(find_standalone_file_with_extension(root, ".fbin").empty());
  EXPECT_FALSE(find_standalone_file_with_extension(root, ".u64").empty());

  vector_index::index_service recovered;
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(recovered.register_index_from_strings("idx_raw_round_trip", 2,
                                                    "euclidean", "memory",
                                                    "native"));
  ASSERT_TRUE(recovered.set_index_consistency_mode(
      "idx_raw_round_trip",
      vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(recovered.rebuild_index("idx_raw_round_trip"));
  ASSERT_TRUE(recovered.search("idx_raw_round_trip", {1.0F, 0.0F}, 2,
                               &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);
}

TEST(VectorLoadFileTest, FbinReaderCoversGeneratedDocIdsAndBadInputs) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_load_file_errors_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);

  const std::string valid_fbin = root + "/valid.fbin";
  const std::string valid_docid = root + "/valid.u64";
  write_raw_fbin_file(valid_fbin, 2, 2, {1.0F, 0.0F, 0.0F, 1.0F});
  write_raw_docid_file(valid_docid, {11, 22});

  vector_index::vector_load_file_info info;
  std::string error;
  std::vector<uint64_t> doc_ids;
  ASSERT_TRUE(vector_index::read_fbin_vectors(
      valid_fbin, "", 0, &info, &error,
      [&doc_ids](uint64_t doc_id, const float *values, size_t dimension) {
        doc_ids.push_back(doc_id);
        EXPECT_EQ(2U, dimension);
        EXPECT_NE(nullptr, values);
        return true;
      }))
      << error;
  EXPECT_EQ(2U, info.row_count);
  EXPECT_EQ(2U, info.dimension);
  EXPECT_EQ((std::vector<uint64_t>{0, 1}), doc_ids);

  doc_ids.clear();
  ASSERT_TRUE(vector_index::read_fbin_vectors(
      valid_fbin, valid_docid, 2, nullptr, nullptr,
      [&doc_ids](uint64_t doc_id, const float *values, size_t dimension) {
        doc_ids.push_back(doc_id);
        EXPECT_EQ(2U, dimension);
        EXPECT_NE(nullptr, values);
        return true;
      }));
  EXPECT_EQ((std::vector<uint64_t>{11, 22}), doc_ids);

  const std::string empty_fbin = root + "/empty.fbin";
  write_raw_fbin_file(empty_fbin, 0, 2, {});
  bool empty_visitor_called = false;
  ASSERT_TRUE(vector_index::read_fbin_vectors(
      empty_fbin, "", 2, nullptr, nullptr,
      [&empty_visitor_called](uint64_t, const float *, size_t) {
        empty_visitor_called = true;
        return true;
      }));
  EXPECT_FALSE(empty_visitor_called);

  EXPECT_FALSE(vector_index::read_fbin_vectors(valid_fbin, "", 2, &info,
                                               &error, nullptr));
  EXPECT_EQ("visitor is required", error);

  EXPECT_FALSE(vector_index::read_fbin_vectors(root + "/missing_no_error.fbin",
                                               "", 2, &info, nullptr,
                                               [](uint64_t, const float *,
                                                  size_t) { return true; }));

  EXPECT_FALSE(vector_index::read_fbin_vectors(root + "/missing.fbin", "", 2,
                                               &info, &error,
                                               [](uint64_t, const float *,
                                                  size_t) { return true; }));
  EXPECT_NE(std::string::npos, error.find("failed to open fbin file"));

  const std::string truncated_header = root + "/truncated_header.fbin";
  write_binary_bytes(truncated_header, std::string("\x01\x00", 2));
  EXPECT_FALSE(vector_index::read_fbin_vectors(
      truncated_header, "", 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_EQ("truncated fbin header", error);

  const std::string zero_dimension = root + "/zero_dimension.fbin";
  write_raw_fbin_file(zero_dimension, 1, 0, {});
  EXPECT_FALSE(vector_index::read_fbin_vectors(
      zero_dimension, "", 0, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_EQ("dimension must be greater than zero", error);

  EXPECT_FALSE(vector_index::read_fbin_vectors(
      valid_fbin, "", 3, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_NE(std::string::npos, error.find("dimension mismatch"));

  const std::string non_finite = root + "/non_finite.fbin";
  write_raw_fbin_file(non_finite, 1, 2,
                      {std::numeric_limits<float>::infinity(), 1.0F});
  EXPECT_FALSE(vector_index::read_fbin_vectors(
      non_finite, "", 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_EQ("non-finite vector value is not allowed", error);

  const std::string truncated_payload = root + "/truncated_payload.fbin";
  {
    std::ofstream file(truncated_payload, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file.good());
    write_binary_value<uint32_t>(&file, 1);
    write_binary_value<uint32_t>(&file, 2);
    write_binary_value<float>(&file, 1.0F);
    ASSERT_TRUE(file.good());
  }
  EXPECT_FALSE(vector_index::read_fbin_vectors(
      truncated_payload, "", 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_EQ("truncated fbin payload", error);

  EXPECT_FALSE(vector_index::read_fbin_vectors(
      valid_fbin, root + "/missing.u64", 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_NE(std::string::npos, error.find("failed to open docid file"));

  const std::string truncated_docid_header = root + "/truncated_docid.u64";
  write_binary_bytes(truncated_docid_header, std::string("\x01\x00", 2));
  EXPECT_FALSE(vector_index::read_fbin_vectors(
      valid_fbin, truncated_docid_header, 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_EQ("truncated docid header", error);

  const std::string mismatched_docid_count = root + "/mismatched_docid.u64";
  write_raw_docid_file(mismatched_docid_count, {11});
  EXPECT_FALSE(vector_index::read_fbin_vectors(
      valid_fbin, mismatched_docid_count, 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_NE(std::string::npos, error.find("docid row count mismatch"));

  const std::string truncated_docid_payload =
      root + "/truncated_docid_payload.u64";
  {
    std::ofstream file(truncated_docid_payload,
                       std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file.good());
    write_binary_value<uint64_t>(&file, 2);
    write_binary_value<uint64_t>(&file, 11);
    ASSERT_TRUE(file.good());
  }
  EXPECT_FALSE(vector_index::read_fbin_vectors(
      valid_fbin, truncated_docid_payload, 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_EQ("truncated docid payload", error);

  const std::string extra_fbin = root + "/extra_payload.fbin";
  write_raw_fbin_file(extra_fbin, 1, 2, {1.0F, 0.0F, 9.0F});
  EXPECT_FALSE(vector_index::read_fbin_vectors(
      extra_fbin, "", 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_EQ("fbin file has extra payload", error);

  const std::string extra_docid = root + "/extra_docid.u64";
  {
    std::ofstream file(extra_docid, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file.good());
    write_binary_value<uint64_t>(&file, 2);
    write_binary_value<uint64_t>(&file, 11);
    write_binary_value<uint64_t>(&file, 22);
    write_binary_value<uint64_t>(&file, 33);
    ASSERT_TRUE(file.good());
  }
  EXPECT_FALSE(vector_index::read_fbin_vectors(
      valid_fbin, extra_docid, 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_EQ("docid file has extra payload", error);

  EXPECT_FALSE(vector_index::read_fbin_vectors(
      valid_fbin, valid_docid, 2, &info, &error,
      [](uint64_t, const float *, size_t) { return false; }));
  EXPECT_EQ("visitor stopped vector load", error);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest, RawSegmentWithoutDocidsGeneratesDocIds) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_generated_docids_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  write_raw_fbin_file(vector_path, 2, 2, {3.0F, 0.0F, 1.0F, 0.0F});

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_raw_generated_docids", 2));
  EXPECT_TRUE(store.bulk_upsert_raw_files("idx_raw_generated_docids",
                                          vector_path, "", 0, 2));
  EXPECT_FALSE(store.bulk_upsert_raw_files("missing", vector_path, "", 2, 2));
  EXPECT_FALSE(store.bulk_upsert_raw_files("idx_raw_generated_docids",
                                           vector_path, "", 2, 3));
  EXPECT_FALSE(store.bulk_upsert_raw_files("idx_raw_generated_docids",
                                           vector_path, "", 3, 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_generated_docids",
                                          vector_path, "", 2, 2));
  EXPECT_EQ(2U, store.entry_count("idx_raw_generated_docids"));
  EXPECT_EQ(1U, store.raw_segment_count("idx_raw_generated_docids"));

  const auto entries = collect_entries(&store, "idx_raw_generated_docids");
  ASSERT_EQ(2U, entries.size());
  EXPECT_EQ((vector_index::vector_data{3.0F, 0.0F}), entries.at(0));
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), entries.at(1));

  RawSegmentCountingBackend backend;
  ASSERT_TRUE(store.rebuild_backend_input("idx_raw_generated_docids", &backend));
  EXPECT_EQ(1U, backend.raw_segment_reader_calls);
  EXPECT_EQ((std::vector<size_t>{2}), backend.raw_segment_doc_counts);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest, BulkUpsertPersistsDeltaSegment) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_bulk_upsert_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::standalone_entry_store store;
  vector_index::vector_data vector;
  bool found = false;
  ASSERT_TRUE(store.register_index("idx_bulk_delta", 2));
  EXPECT_FALSE(store.bulk_upsert("missing", {{1, {1.0F, 1.0F}}}));
  EXPECT_TRUE(store.bulk_upsert("idx_bulk_delta", {}));
  EXPECT_FALSE(store.bulk_upsert("idx_bulk_delta", {{1, {1.0F}}}));

  ASSERT_TRUE(store.bulk_upsert("idx_bulk_delta",
                                {{20, {2.0F, 0.0F}}, {10, {1.0F, 0.0F}}}));
  EXPECT_EQ(2U, store.entry_count("idx_bulk_delta"));
  EXPECT_EQ(0U, store.memory_bytes("idx_bulk_delta"));
  EXPECT_EQ(1U, store.segment_count("idx_bulk_delta"));
  EXPECT_EQ("delta_replay", store.build_source("idx_bulk_delta"));
  ASSERT_TRUE(store.find_entry("idx_bulk_delta", 10, &vector, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), vector);

  std::vector<uint64_t> visited_doc_ids;
  ASSERT_TRUE(store.for_each_entry(
      "idx_bulk_delta",
      [&visited_doc_ids](uint64_t doc_id,
                         const vector_index::vector_data &entry_vector) {
        visited_doc_ids.push_back(doc_id);
        EXPECT_EQ(2U, entry_vector.size());
        return true;
      }));
  EXPECT_EQ((std::vector<uint64_t>{10, 20}), visited_doc_ids);

  vector_index::standalone_entry_store reloaded;
  ASSERT_TRUE(reloaded.register_index("idx_bulk_delta", 2));
  ASSERT_TRUE(reloaded.find_entry("idx_bulk_delta", 20, &vector, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ((vector_index::vector_data{2.0F, 0.0F}), vector);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest, RawAndDeltaSegmentsReplayInDocIdOrder) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_delta_replay_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {3.0F, 0.0F, 1.0F, 0.0F});
  write_raw_docid_file(docid_path, {30, 10});

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_raw_delta", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_delta", vector_path,
                                          docid_path, 2, 2));
  ASSERT_TRUE(store.upsert("idx_raw_delta", 20, {2.0F, 0.0F}, 0));
  ASSERT_TRUE(store.erase("idx_raw_delta", 30, 0));

  std::vector<uint64_t> visited_doc_ids;
  ASSERT_TRUE(store.for_each_entry(
      "idx_raw_delta",
      [&visited_doc_ids](uint64_t doc_id,
                         const vector_index::vector_data &) {
        visited_doc_ids.push_back(doc_id);
        return true;
      }));
  EXPECT_EQ((std::vector<uint64_t>{10, 20}), visited_doc_ids);
  EXPECT_EQ(1U, store.raw_segment_count("idx_raw_delta"));
  EXPECT_EQ(3U, store.segment_count("idx_raw_delta"));

  vector_index::standalone_entry_store recovered;
  ASSERT_TRUE(recovered.register_index("idx_raw_delta", 2));
  const auto entries = collect_entries(&recovered, "idx_raw_delta");
  ASSERT_EQ(2U, entries.size());
  ASSERT_NE(entries.end(), entries.find(10));
  ASSERT_NE(entries.end(), entries.find(20));
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), entries.at(10));
  EXPECT_EQ((vector_index::vector_data{2.0F, 0.0F}), entries.at(20));
}

TEST(VectorStandaloneEntryStoreTest, RawOnlyRebuildUsesRawSegmentReader) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_direct_rebuild_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {3.0F, 0.0F, 1.0F, 0.0F});
  write_raw_docid_file(docid_path, {30, 10});

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_raw_direct", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_direct", vector_path,
                                          docid_path, 2, 2));

  RawSegmentCountingBackend backend;
  ASSERT_TRUE(store.rebuild_backend_input("idx_raw_direct", &backend));
  EXPECT_EQ(1U, backend.raw_segment_reader_calls);
  EXPECT_EQ(0U, backend.committed_reader_calls);
  EXPECT_EQ((std::vector<size_t>{2}), backend.raw_segment_doc_counts);

  RawSegmentCountingBackend rejecting_backend;
  rejecting_backend.reject_raw_segments = true;
  EXPECT_FALSE(store.rebuild_backend_input("idx_raw_direct",
                                           &rejecting_backend));
  EXPECT_EQ(1U, rejecting_backend.raw_segment_reader_calls);
}

TEST(VectorStandaloneEntryStoreTest, MixedSegmentsCompactBeforeRawRebuild) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_compact_rebuild_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {3.0F, 0.0F, 1.0F, 0.0F});
  write_raw_docid_file(docid_path, {30, 10});

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_raw_compact", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_compact", vector_path,
                                          docid_path, 2, 2));
  ASSERT_TRUE(store.upsert("idx_raw_compact", 20, {2.0F, 0.0F}, 0));
  ASSERT_TRUE(store.erase("idx_raw_compact", 30, 0));

  RawSegmentCountingBackend backend;
  ASSERT_TRUE(store.rebuild_backend_input("idx_raw_compact", &backend));
  EXPECT_EQ(1U, backend.raw_segment_reader_calls);
  EXPECT_EQ(0U, backend.committed_reader_calls);
  EXPECT_EQ((std::vector<size_t>{2}), backend.raw_segment_doc_counts);
  EXPECT_EQ(1U, store.segment_count("idx_raw_compact"));
  EXPECT_EQ(1U, store.raw_segment_count("idx_raw_compact"));

  const auto entries = collect_entries(&store, "idx_raw_compact");
  ASSERT_EQ(2U, entries.size());
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), entries.at(10));
  EXPECT_EQ((vector_index::vector_data{2.0F, 0.0F}), entries.at(20));
}

TEST(VectorStandaloneEntryStoreTest,
     RawSegmentRenameReloadAndMissingFileRejectRebuild) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_rename_reload_t";
  const std::string input_dir = root + "/input";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(input_dir, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = input_dir + "/source.fbin";
  const std::string docid_path = input_dir + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {4.0F, 0.0F, 1.0F, 0.0F});
  write_raw_docid_file(docid_path, {40, 10});

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_raw_rename", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_rename", vector_path,
                                          docid_path, 2, 2));
  ASSERT_TRUE(store.rename_index("idx_raw_rename", "idx_raw_moved"));
  EXPECT_FALSE(store.has_index("idx_raw_rename"));
  EXPECT_TRUE(store.has_index("idx_raw_moved"));
  EXPECT_EQ(1U, store.raw_segment_count("idx_raw_moved"));

  RawSegmentCountingBackend backend;
  ASSERT_TRUE(store.rebuild_backend_input("idx_raw_moved", &backend));
  EXPECT_EQ(1U, backend.raw_segment_reader_calls);
  EXPECT_EQ((std::vector<size_t>{2}), backend.raw_segment_doc_counts);

  vector_index::standalone_entry_store reloaded;
  ASSERT_TRUE(reloaded.register_index("idx_raw_moved", 2));
  vector_index::vector_data vector;
  bool found = false;
  ASSERT_TRUE(reloaded.find_entry("idx_raw_moved", 10, &vector, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), vector);

  const std::string stored_vector_path =
      find_raw_segment_file_from_manifest(root, ".fbin");
  ASSERT_FALSE(stored_vector_path.empty());
  ASSERT_TRUE(std::filesystem::remove(stored_vector_path, ec));
  ASSERT_FALSE(ec);

  RawSegmentCountingBackend missing_file_backend;
  EXPECT_FALSE(reloaded.rebuild_backend_input("idx_raw_moved",
                                              &missing_file_backend));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     ManifestWriteFailurePreservesOriginalRenameState) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_rename_rollback_t";
  const std::string input_dir = root + "/input";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(input_dir, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = input_dir + "/source.fbin";
  const std::string docid_path = input_dir + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {4.0F, 0.0F, 1.0F, 0.0F});
  write_raw_docid_file(docid_path, {40, 10});

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_rename_rollback", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_rename_rollback", vector_path,
                                          docid_path, 2, 2));

  const std::string manifest = find_manifest_file(root, "manifest.v1");
  ASSERT_FALSE(manifest.empty());
  std::filesystem::create_directory(manifest + ".tmp", ec);
  ASSERT_FALSE(ec);

  EXPECT_FALSE(store.rename_index("idx_rename_rollback", "idx_rename_failed"));
  EXPECT_TRUE(store.has_index("idx_rename_rollback"));
  EXPECT_FALSE(store.has_index("idx_rename_failed"));
  std::filesystem::remove_all(manifest + ".tmp", ec);
  ASSERT_FALSE(ec);

  RawSegmentCountingBackend backend;
  ASSERT_TRUE(store.rebuild_backend_input("idx_rename_rollback", &backend));
  EXPECT_EQ(1U, backend.raw_segment_reader_calls);
  EXPECT_EQ((std::vector<size_t>{2}), backend.raw_segment_doc_counts);

  vector_index::vector_data vector;
  bool found = false;
  ASSERT_TRUE(store.find_entry("idx_rename_rollback", 10, &vector, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), vector);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, DiskAnnStandaloneMutationsWaitForRebuild) {
  const std::string root =
      std::string(testing::TempDir()) + "/diskann_standalone_deferred_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::index_service service;
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.register_index_from_strings("idx_diskann_standalone", 2,
                                                  "euclidean", "external",
                                                  "diskann"));
  const bool standalone_ready = service.set_index_consistency_mode(
      "idx_diskann_standalone",
      vector_index::index_consistency_mode::kStandalone);
  if (!standalone_ready) {
    vector_index::index_service::index_config config;
    size_t entry_count = 0;
    size_t committed_entry_count = 0;
    std::string lifecycle_state;
    ASSERT_TRUE(service.describe_index("idx_diskann_standalone", &config,
                                       nullptr, &entry_count,
                                       &committed_entry_count,
                                       &lifecycle_state));
    EXPECT_EQ(vector_index::index_consistency_mode::kTransactional,
              config.consistency_mode);
    EXPECT_EQ(vector_index::backend_provider::kDiskAnn, config.provider);
    EXPECT_EQ(0U, entry_count);
    EXPECT_EQ(0U, committed_entry_count);
    EXPECT_EQ("failed", lifecycle_state);
    EXPECT_FALSE(service.direct_upsert("idx_diskann_standalone", 11,
                                       {1.0F, 0.0F}));
    std::filesystem::remove_all(root, ec);
    return;
  }

  ASSERT_TRUE(service.direct_upsert("idx_diskann_standalone", 11,
                                    {1.0F, 0.0F}));

  vector_index::index_service::index_config config;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  std::string lifecycle_state;
  ASSERT_TRUE(service.describe_index("idx_diskann_standalone", &config, nullptr,
                                     &entry_count, &committed_entry_count,
                                     &lifecycle_state));
  EXPECT_EQ(vector_index::index_consistency_mode::kStandalone,
            config.consistency_mode);
  EXPECT_EQ(vector_index::backend_provider::kDiskAnn, config.provider);
  EXPECT_EQ(0U, entry_count);
  EXPECT_EQ(0U, committed_entry_count);
  EXPECT_EQ("bulk_loading", lifecycle_state);
  EXPECT_GT(service.standalone_ingest_memory_bytes("idx_diskann_standalone"),
            0U);
  EXPECT_FALSE(service.search("idx_diskann_standalone", {1.0F, 0.0F}, 1,
                              &result));

  ASSERT_TRUE(service.rebuild_index("idx_diskann_standalone"));
  ASSERT_TRUE(service.describe_index("idx_diskann_standalone", &config, nullptr,
                                     &entry_count, &committed_entry_count,
                                     &lifecycle_state));
  EXPECT_EQ(1U, entry_count);
  EXPECT_EQ(0U, committed_entry_count);
  EXPECT_EQ("ready", lifecycle_state);
  ASSERT_TRUE(service.search("idx_diskann_standalone", {1.0F, 0.0F}, 1,
                             &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest, MemoryCacheOperationsAndInvalidInputs) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_entry_memory_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);
  vector_index::standalone_entry_store store;
  vector_index::vector_data vector;
  bool found = false;
  std::vector<uint64_t> visited_doc_ids;

  EXPECT_FALSE(store.register_index("", 2));
  EXPECT_FALSE(store.register_index("idx_zero", 0));
  ASSERT_TRUE(store.register_index("idx_memory", 2));
  EXPECT_TRUE(store.has_index("idx_memory"));
  EXPECT_FALSE(store.has_index("missing"));
  EXPECT_FALSE(store.register_index("idx_memory", 2));
  EXPECT_FALSE(store.upsert("missing", 1, {1.0F, 1.0F},
                            std::numeric_limits<size_t>::max()));
  EXPECT_FALSE(store.upsert("idx_memory", 1, {1.0F},
                            std::numeric_limits<size_t>::max()));
  EXPECT_FALSE(store.erase("missing", 1, std::numeric_limits<size_t>::max()));
  EXPECT_FALSE(store.for_each_entry("idx_memory", nullptr));
  EXPECT_FALSE(store.for_each_entry(
      "missing", [](uint64_t, const vector_index::vector_data &) {
        return true;
      }));
  EXPECT_FALSE(store.find_entry("idx_memory", 1, nullptr, &found));
  EXPECT_FALSE(store.find_entry("idx_memory", 1, &vector, nullptr));
  EXPECT_FALSE(store.find_entry("missing", 1, &vector, &found));

  ASSERT_TRUE(store.register_index("idx_budget", 2));
  ASSERT_TRUE(store.upsert("idx_budget", 10, {10.0F, 10.0F}, 1024));
  EXPECT_EQ(2U * sizeof(float), store.memory_bytes("idx_budget"));
  EXPECT_EQ(0U, store.segment_count("idx_budget"));
  ASSERT_TRUE(store.erase("idx_budget", 10, 1024));
  EXPECT_EQ(0U, store.entry_count("idx_budget"));
  EXPECT_EQ(0U, store.segment_count("idx_budget"));
  EXPECT_TRUE(store.drop_index("idx_budget"));

  ASSERT_TRUE(store.upsert("idx_memory", 2, {2.0F, 2.0F},
                           std::numeric_limits<size_t>::max()));
  ASSERT_TRUE(store.upsert("idx_memory", 1, {1.0F, 1.0F},
                           std::numeric_limits<size_t>::max()));
  EXPECT_EQ(2U, store.entry_count("idx_memory"));
  EXPECT_EQ(4U * sizeof(float), store.memory_bytes("idx_memory"));
  EXPECT_EQ(0U, store.segment_count("idx_memory"));
  EXPECT_EQ(2U, store.generation("idx_memory"));

  ASSERT_TRUE(store.find_entry("idx_memory", 1, &vector, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(std::vector<float>({1.0F, 1.0F}), vector);
  ASSERT_TRUE(store.find_entry("idx_memory", 99, &vector, &found));
  EXPECT_FALSE(found);

  EXPECT_FALSE(store.for_each_entry(
      "idx_memory", [&visited_doc_ids](
                        uint64_t doc_id,
                        const vector_index::vector_data &entry_vector) {
        visited_doc_ids.push_back(doc_id);
        EXPECT_EQ(2U, entry_vector.size());
        return false;
      }));
  ASSERT_EQ(1U, visited_doc_ids.size());
  EXPECT_EQ(1U, visited_doc_ids[0]);

  EXPECT_TRUE(store.erase("idx_memory", 99, std::numeric_limits<size_t>::max()));
  EXPECT_EQ(2U, store.entry_count("idx_memory"));
  ASSERT_TRUE(store.erase("idx_memory", 1, std::numeric_limits<size_t>::max()));
  EXPECT_EQ(1U, store.entry_count("idx_memory"));
  ASSERT_TRUE(store.find_entry("idx_memory", 1, &vector, &found));
  EXPECT_FALSE(found);

  EXPECT_FALSE(store.drop_index("missing"));
  EXPECT_TRUE(store.drop_index("idx_memory"));
  EXPECT_FALSE(store.has_index("idx_memory"));
  EXPECT_EQ(0U, store.entry_count("idx_memory"));
  EXPECT_EQ(0U, store.memory_bytes("idx_memory"));
  EXPECT_EQ(0U, store.segment_count("idx_memory"));
  EXPECT_EQ(0U, store.segment_bytes("idx_memory"));
  EXPECT_EQ(0U, store.generation("idx_memory"));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest, SegmentManifestRenameAndReload) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_entry_segment_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);
  vector_index::vector_data vector;
  bool found = false;
  std::vector<uint64_t> visited_doc_ids;

  {
    vector_index::standalone_entry_store store;
    ASSERT_TRUE(store.register_index("idx_segment", 2));
    ASSERT_TRUE(store.upsert("idx_segment", 3, {3.0F, 3.0F}, 0));
    ASSERT_TRUE(store.upsert("idx_segment", 1, {1.0F, 1.0F}, 0));
    ASSERT_TRUE(store.erase("idx_segment", 3, 0));
    EXPECT_EQ(1U, store.entry_count("idx_segment"));
    EXPECT_EQ(0U, store.memory_bytes("idx_segment"));
    EXPECT_GE(store.segment_count("idx_segment"), 3U);
    EXPECT_GT(store.segment_bytes("idx_segment"), 0U);
    EXPECT_EQ(3U, store.generation("idx_segment"));

    ASSERT_TRUE(store.find_entry("idx_segment", 1, &vector, &found));
    EXPECT_TRUE(found);
    EXPECT_EQ(std::vector<float>({1.0F, 1.0F}), vector);
    ASSERT_TRUE(store.find_entry("idx_segment", 3, &vector, &found));
    EXPECT_FALSE(found);

    ASSERT_TRUE(store.for_each_entry(
        "idx_segment", [&visited_doc_ids](
                           uint64_t doc_id,
                           const vector_index::vector_data &entry_vector) {
          visited_doc_ids.push_back(doc_id);
          EXPECT_EQ(std::vector<float>({1.0F, 1.0F}), entry_vector);
          return true;
        }));
    EXPECT_EQ(std::vector<uint64_t>({1}), visited_doc_ids);

    ASSERT_TRUE(store.register_index("idx_conflict", 2));
    EXPECT_FALSE(store.rename_index("", "idx_new"));
    EXPECT_FALSE(store.rename_index("idx_segment", ""));
    EXPECT_FALSE(store.rename_index("missing", "idx_new"));
    EXPECT_FALSE(store.rename_index("idx_segment", "idx_conflict"));
    ASSERT_TRUE(store.rename_index("idx_segment", "idx_moved"));
    EXPECT_FALSE(store.has_index("idx_segment"));
    EXPECT_TRUE(store.has_index("idx_moved"));
    ASSERT_TRUE(store.find_entry("idx_moved", 1, &vector, &found));
    EXPECT_TRUE(found);
  }

  vector_index::standalone_entry_store wrong_dimension_store;
  EXPECT_FALSE(wrong_dimension_store.register_index("idx_moved", 3));

  vector_index::standalone_entry_store reloaded_store;
  ASSERT_TRUE(reloaded_store.register_index("idx_moved", 2));
  ASSERT_TRUE(reloaded_store.find_entry("idx_moved", 1, &vector, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(std::vector<float>({1.0F, 1.0F}), vector);
  ASSERT_TRUE(reloaded_store.for_each_entry(
      "idx_moved", [](uint64_t doc_id,
                       const vector_index::vector_data &entry_vector) {
        EXPECT_EQ(1U, doc_id);
        EXPECT_EQ(std::vector<float>({1.0F, 1.0F}), entry_vector);
        return true;
      }));

  const std::string manifest = find_manifest_file(root, "manifest.v1");
  ASSERT_FALSE(manifest.empty());
  {
    std::ofstream file(manifest, std::ios::out | std::ios::app);
    ASSERT_TRUE(file.good());
    file << "unknown\t1\n";
  }
  vector_index::standalone_entry_store corrupt_manifest_store;
  EXPECT_FALSE(corrupt_manifest_store.register_index("idx_moved", 2));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     SegmentCorruptionAndReaderFailuresRejectReload) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_entry_corrupt_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::standalone_entry_store store;
  vector_index::memory_backend backend(2, vector_index::metric_type::kEuclidean);
  ASSERT_TRUE(store.register_index("idx_corrupt", 2));
  EXPECT_FALSE(store.rebuild_backend_input("missing", &backend));
  EXPECT_FALSE(store.rebuild_backend_input("idx_corrupt", nullptr));

  ASSERT_TRUE(store.upsert("idx_corrupt", 9, {9.0F, 9.0F}, 0));
  ASSERT_TRUE(store.rebuild_backend_input("idx_corrupt", &backend));
  EXPECT_EQ(1U, backend.entry_count());

  EXPECT_FALSE(store.for_each_entry(
      "idx_corrupt", [](uint64_t, const vector_index::vector_data &) {
        return false;
      }));

  const std::string raw_segment =
      find_standalone_file_with_extension(root, ".fbin");
  ASSERT_FALSE(raw_segment.empty());
  {
    std::fstream file(raw_segment,
                      std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(file.is_open());
    const char bad_header_prefix = 'X';
    file.write(&bad_header_prefix, 1);
    ASSERT_TRUE(file.good());
  }

  vector_index::standalone_entry_store corrupt_segment_store;
  EXPECT_FALSE(corrupt_segment_store.register_index("idx_corrupt", 2));

  const std::string manifest = find_manifest_file(root, "manifest.v1");
  ASSERT_FALSE(manifest.empty());
  {
    std::ofstream file(manifest, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(file.is_open());
    file << "mysql-vector-standalone-manifest-v1\n"
         << "dimension\t2\n";
    ASSERT_TRUE(file.good());
  }

  vector_index::standalone_entry_store incomplete_manifest_store;
  EXPECT_FALSE(incomplete_manifest_store.register_index("idx_corrupt", 2));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest, ManifestAndSegmentValidationRejectBadInput) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_entry_validation_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const auto write_invalid_manifest =
      [](const std::string &manifest_path, const std::string &segment_line) {
        std::ofstream file(manifest_path, std::ios::out | std::ios::trunc);
        if (!file.is_open()) return false;
        file << "mysql-vector-standalone-manifest-v1\n"
             << "dimension\t2\n"
             << "entry_count\t1\n"
             << "generation\t1\n"
             << "next_segment_id\t2\n"
             << "segment_count\t1\n"
             << segment_line << '\n';
        return file.good();
      };

  std::string manifest;
  std::string segment;
  {
    vector_index::standalone_entry_store store;
    ASSERT_TRUE(store.register_index("idx_validate", 2));
    ASSERT_TRUE(store.upsert("idx_validate", 1, {1.0F, 1.0F}, 0));
    manifest = find_manifest_file(root, "manifest.v1");
    segment = find_standalone_segment_file(root);
    ASSERT_FALSE(manifest.empty());
    ASSERT_FALSE(segment.empty());
  }

  ASSERT_TRUE(write_invalid_manifest(manifest,
                                     "segment\t../bad.vseg\t1\t1"));
  vector_index::standalone_entry_store parent_path_store;
  EXPECT_FALSE(parent_path_store.register_index("idx_validate", 2));

  const uintmax_t segment_size = std::filesystem::file_size(segment, ec);
  ASSERT_FALSE(ec);
  ASSERT_TRUE(write_invalid_manifest(
      manifest, "segment\t" +
                    std::filesystem::path(segment).filename().string() +
                    "\t1\t" + std::to_string(segment_size)));
  {
    std::fstream file(segment, std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(file.is_open());
    std::string header;
    ASSERT_TRUE(static_cast<bool>(std::getline(file, header)));
    const std::streamoff op_offset =
        static_cast<std::streamoff>(file.tellg()) +
        static_cast<std::streamoff>(sizeof(uint64_t) * 2);
    file.seekp(op_offset);
    ASSERT_TRUE(file.good());
    const char unknown_op = '\x7f';
    file.write(&unknown_op, 1);
    ASSERT_TRUE(file.good());
  }
  vector_index::standalone_entry_store unknown_op_store;
  EXPECT_FALSE(unknown_op_store.register_index("idx_validate", 2));

  ASSERT_TRUE(write_invalid_manifest(
      manifest, "segment\t" +
                    std::filesystem::path(segment).filename().string() +
                    "\t1\t" + std::to_string(segment_size)));
  {
    std::ofstream file(manifest, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(file.is_open());
    file << "mysql-vector-standalone-manifest-v1\n"
         << "dimension\t2\n"
         << "entry_count\t1\n"
         << "generation\t1\n"
         << "next_segment_id\t2\n"
         << "segment_count\t2\n"
         << "segment\t" << std::filesystem::path(segment).filename().string()
         << "\t1\t" << segment_size << '\n';
    ASSERT_TRUE(file.good());
  }
  vector_index::standalone_entry_store mismatched_count_store;
  EXPECT_FALSE(mismatched_count_store.register_index("idx_validate", 2));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     RawManifestValidationRejectsMalformedFields) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_manifest_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {1.0F, 0.0F, 0.0F, 1.0F});
  write_raw_docid_file(docid_path, {10, 20});

  std::string manifest;
  std::string stored_fbin;
  std::string stored_docid;
  {
    vector_index::standalone_entry_store store;
    ASSERT_TRUE(store.register_index("idx_raw_manifest", 2));
    ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_manifest", vector_path,
                                            docid_path, 2, 2));
    manifest = find_manifest_file(root, "manifest.v1");
    stored_fbin = find_standalone_file_with_extension(root, ".fbin");
    stored_docid = find_standalone_file_with_extension(root, ".u64");
    ASSERT_FALSE(manifest.empty());
    ASSERT_FALSE(stored_fbin.empty());
    ASSERT_FALSE(stored_docid.empty());
  }

  size_t vector_bytes = std::filesystem::file_size(stored_fbin, ec);
  ASSERT_FALSE(ec);
  size_t docid_bytes = std::filesystem::file_size(stored_docid, ec);
  ASSERT_FALSE(ec);
  const size_t segment_bytes = vector_bytes + docid_bytes;
  const std::string fbin_name =
      std::filesystem::path(stored_fbin).filename().string();
  const std::string docid_name =
      std::filesystem::path(stored_docid).filename().string();

  const auto write_manifest = [&](const std::string &segment_line) {
    std::ofstream file(manifest, std::ios::out | std::ios::trunc);
    if (!file.is_open()) return false;
    file << "mysql-vector-standalone-manifest-v1\n"
         << "dimension\t2\n"
         << "entry_count\t2\n"
         << "generation\t1\n"
         << "next_segment_id\t2\n"
         << "segment_count\t1\n"
         << segment_line << '\n';
    return file.good();
  };

  const std::vector<std::string> bad_raw_segment_lines = {
      "segment_raw_fbin\t" + fbin_name + '\t' + docid_name + "\t2\t2\t" +
          std::to_string(segment_bytes),
      "segment_raw_fbin\t../bad.fbin\t" + docid_name + "\t2\t2\t" +
          std::to_string(segment_bytes) + "\t1",
      "segment_raw_fbin\t" + fbin_name + "\t../bad.u64\t2\t2\t" +
          std::to_string(segment_bytes) + "\t1",
      "segment_raw_fbin\t" + fbin_name + '\t' + docid_name + "\tbad\t2\t" +
          std::to_string(segment_bytes) + "\t1",
      "segment_raw_fbin\t" + fbin_name + '\t' + docid_name + "\t2\tbad\t" +
          std::to_string(segment_bytes) + "\t1",
      "segment_raw_fbin\t" + fbin_name + '\t' + docid_name + "\t2\t2\tbad\t1",
      "segment_raw_fbin\t" + fbin_name + '\t' + docid_name + "\t2\t2\t" +
          std::to_string(segment_bytes) + "\tbad",
      "segment_raw_fbin\t" + fbin_name + '\t' + docid_name + "\t2\t3\t" +
          std::to_string(segment_bytes) + "\t1",
      "segment_raw_fbin\t" + fbin_name + '\t' + docid_name + "\t2\t2\t" +
          std::to_string(segment_bytes + 1) + "\t1"};

  for (const std::string &bad_line : bad_raw_segment_lines) {
    ASSERT_TRUE(write_manifest(bad_line));
    vector_index::standalone_entry_store bad_store;
    EXPECT_FALSE(bad_store.register_index("idx_raw_manifest", 2))
        << bad_line;
  }

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     ManifestFieldValidationRejectsMalformedValues) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_entry_manifest_fields_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  std::string manifest;
  std::string segment;
  uintmax_t segment_size = 0;
  {
    vector_index::standalone_entry_store store;
    ASSERT_TRUE(store.register_index("idx_manifest_fields", 2));
    ASSERT_TRUE(store.upsert("idx_manifest_fields", 1, {1.0F, 1.0F}, 0));
    manifest = find_manifest_file(root, "manifest.v1");
    segment = find_standalone_segment_file(root);
    ASSERT_FALSE(manifest.empty());
    ASSERT_FALSE(segment.empty());
    segment_size = std::filesystem::file_size(segment, ec);
    ASSERT_FALSE(ec);
  }

  const std::string segment_name =
      std::filesystem::path(segment).filename().string();
  const auto write_manifest =
      [&](const std::vector<std::string> &body_lines) {
        std::ofstream file(manifest, std::ios::out | std::ios::trunc);
        if (!file.is_open()) return false;
        file << "mysql-vector-standalone-manifest-v1\n";
        for (const std::string &line : body_lines) file << line << '\n';
        return file.good();
      };
  const std::vector<std::vector<std::string>> bad_manifests = {
      {"dimension\tbad", "entry_count\t1", "generation\t1",
       "next_segment_id\t2", "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2\textra", "entry_count\t1", "generation\t1",
       "next_segment_id\t2", "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\tbad", "generation\t1",
       "next_segment_id\t2", "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1\textra", "generation\t1",
       "next_segment_id\t2", "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1", "generation\tbad",
       "next_segment_id\t2", "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1", "generation\t1\textra",
       "next_segment_id\t2", "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1", "generation\t1",
       "next_segment_id\t0", "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1", "generation\t1",
       "next_segment_id\tbad", "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1", "generation\t1",
       "next_segment_id\t2\textra", "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1", "generation\t1",
       "next_segment_id\t2", "segment_count\tbad",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1", "generation\t1",
       "next_segment_id\t2", "segment_count\t1\textra",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1", "generation\t1",
       "next_segment_id\t2", "segment_count\t1",
       "segment\t" + segment_name + "\tbad\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1", "generation\t1",
       "next_segment_id\t2", "segment_count\t1",
       "segment\t" + segment_name + "\t1\tbad"},
      {"dimension\t2", "entry_count\t1", "generation\t1",
       "next_segment_id\t2", "segment_count\t1",
       "segment\t" + segment_name},
      {"dimension\t2", "entry_count\t1", "generation\t1",
       "next_segment_id\t2", "segment_count\t1",
       "segment\tmissing.vseg\t1\t1"},
      {"dimension\t2", "entry_count\t1", "generation\t1",
       "next_segment_id\t2", "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" +
           std::to_string(segment_size + 1)},
      {"dimension\t2", "generation\t1", "next_segment_id\t2",
       "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1", "next_segment_id\t2",
       "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1", "generation\t1",
       "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t1", "generation\t1",
       "next_segment_id\t2",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"entry_count\t1", "generation\t1", "next_segment_id\t2",
       "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)},
      {"dimension\t2", "entry_count\t2", "generation\t1",
       "next_segment_id\t2", "segment_count\t1",
       "segment\t" + segment_name + "\t1\t" + std::to_string(segment_size)}};

  for (const auto &bad_manifest : bad_manifests) {
    ASSERT_TRUE(write_manifest(bad_manifest));
    vector_index::standalone_entry_store store;
    EXPECT_FALSE(store.register_index("idx_manifest_fields", 2));
  }

  {
    std::ofstream file(manifest, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(file.is_open());
    file << "bad-manifest-header\n";
    ASSERT_TRUE(file.good());
  }
  vector_index::standalone_entry_store bad_header_store;
  EXPECT_FALSE(bad_header_store.register_index("idx_manifest_fields", 2));

  std::filesystem::remove_all(manifest, ec);
  std::filesystem::create_directories(manifest, ec);
  ASSERT_FALSE(ec);
  vector_index::standalone_entry_store manifest_directory_store;
  EXPECT_FALSE(manifest_directory_store.register_index("idx_manifest_fields", 2));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     SegmentReplayValidationRejectsMalformedRecords) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_entry_segment_replay_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  std::string manifest;
  std::string segment;
  {
    vector_index::standalone_entry_store store;
    ASSERT_TRUE(store.register_index("idx_segment_replay", 2));
    ASSERT_TRUE(store.upsert("idx_segment_replay", 1, {1.0F, 1.0F}, 0));
    manifest = find_manifest_file(root, "manifest.v1");
    segment = find_standalone_segment_file(root);
    ASSERT_FALSE(manifest.empty());
    ASSERT_FALSE(segment.empty());
  }

  const auto segment_name = std::filesystem::path(segment).filename().string();
  const auto rewrite_manifest_size = [&]() {
    std::error_code size_ec;
    const uintmax_t segment_size = std::filesystem::file_size(segment, size_ec);
    if (size_ec) return false;
    std::ofstream file(manifest, std::ios::out | std::ios::trunc);
    if (!file.is_open()) return false;
    file << "mysql-vector-standalone-manifest-v1\n"
         << "dimension\t2\n"
         << "entry_count\t1\n"
         << "generation\t1\n"
         << "next_segment_id\t2\n"
         << "segment_count\t1\n"
         << "segment\t" << segment_name << "\t1\t" << segment_size << '\n';
    return file.good();
  };
  const auto rewrite_segment_header = [&](uint64_t dimension,
                                          uint64_t record_count) {
    std::ofstream file(segment, std::ios::out | std::ios::binary |
                                    std::ios::trunc);
    if (!file.is_open()) return false;
    const uint8_t op = 1;
    const uint64_t doc_id = 1;
    const float values[2] = {1.0F, 1.0F};
    file << "mysql-vector-standalone-segment-v1\n";
    file.write(reinterpret_cast<const char *>(&dimension), sizeof(dimension));
    file.write(reinterpret_cast<const char *>(&record_count),
               sizeof(record_count));
    file.write(reinterpret_cast<const char *>(&op), sizeof(op));
    file.write(reinterpret_cast<const char *>(&doc_id), sizeof(doc_id));
    file.write(reinterpret_cast<const char *>(values), sizeof(values));
    return file.good();
  };

  ASSERT_TRUE(rewrite_segment_header(3, 1));
  ASSERT_TRUE(rewrite_manifest_size());
  vector_index::standalone_entry_store bad_dimension_store;
  EXPECT_FALSE(bad_dimension_store.register_index("idx_segment_replay", 2));

  ASSERT_TRUE(rewrite_segment_header(2, 2));
  ASSERT_TRUE(rewrite_manifest_size());
  vector_index::standalone_entry_store bad_record_count_store;
  EXPECT_FALSE(bad_record_count_store.register_index("idx_segment_replay", 2));

  {
    std::ofstream file(segment, std::ios::out | std::ios::binary |
                                    std::ios::trunc);
    ASSERT_TRUE(file.is_open());
    file << "mysql-vector-standalone-segment-v1\n";
    ASSERT_TRUE(file.good());
  }
  ASSERT_TRUE(rewrite_manifest_size());
  vector_index::standalone_entry_store missing_dimension_store;
  EXPECT_FALSE(missing_dimension_store.register_index("idx_segment_replay", 2));

  {
    std::ofstream file(segment, std::ios::out | std::ios::binary |
                                    std::ios::trunc);
    ASSERT_TRUE(file.is_open());
    const uint64_t dimension = 2;
    file << "mysql-vector-standalone-segment-v1\n";
    file.write(reinterpret_cast<const char *>(&dimension), sizeof(dimension));
    ASSERT_TRUE(file.good());
  }
  ASSERT_TRUE(rewrite_manifest_size());
  vector_index::standalone_entry_store missing_record_count_store;
  EXPECT_FALSE(
      missing_record_count_store.register_index("idx_segment_replay", 2));

  {
    std::ofstream file(segment, std::ios::out | std::ios::binary |
                                    std::ios::trunc);
    ASSERT_TRUE(file.is_open());
    const uint64_t dimension = 2;
    const uint64_t record_count = 1;
    file << "mysql-vector-standalone-segment-v1\n";
    file.write(reinterpret_cast<const char *>(&dimension), sizeof(dimension));
    file.write(reinterpret_cast<const char *>(&record_count),
               sizeof(record_count));
    ASSERT_TRUE(file.good());
  }
  ASSERT_TRUE(rewrite_manifest_size());
  vector_index::standalone_entry_store truncated_record_store;
  EXPECT_FALSE(truncated_record_store.register_index("idx_segment_replay", 2));

  {
    std::ofstream file(segment, std::ios::out | std::ios::binary |
                                    std::ios::trunc);
    ASSERT_TRUE(file.is_open());
    const uint64_t dimension = 2;
    const uint64_t record_count = 1;
    const uint8_t op = 1;
    const uint64_t doc_id = 1;
    file << "mysql-vector-standalone-segment-v1\n";
    file.write(reinterpret_cast<const char *>(&dimension), sizeof(dimension));
    file.write(reinterpret_cast<const char *>(&record_count),
               sizeof(record_count));
    file.write(reinterpret_cast<const char *>(&op), sizeof(op));
    file.write(reinterpret_cast<const char *>(&doc_id), sizeof(doc_id));
    ASSERT_TRUE(file.good());
  }
  ASSERT_TRUE(rewrite_manifest_size());
  vector_index::standalone_entry_store missing_vector_payload_store;
  EXPECT_FALSE(
      missing_vector_payload_store.register_index("idx_segment_replay", 2));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     StandaloneMutationsSpillSegmentsAndRebuildFromStandaloneSource) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_service_standalone_segments_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);
  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, 0);

  vector_index::index_service service;
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.register_index_from_strings("idx_standalone_spill", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_standalone_spill",
      vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(service.direct_upsert("idx_standalone_spill", 2,
                                    {2.0F, 0.0F}));
  ASSERT_TRUE(service.direct_upsert("idx_standalone_spill", 1,
                                    {1.0F, 0.0F}));
  ASSERT_TRUE(service.direct_erase("idx_standalone_spill", 2));

  EXPECT_EQ(0U, service.standalone_ingest_memory_bytes(
                    "idx_standalone_spill"));
  EXPECT_GE(service.standalone_segment_count("idx_standalone_spill"), 3U);
  EXPECT_GT(service.standalone_segment_bytes("idx_standalone_spill"), 0U);
  EXPECT_FALSE(find_manifest_file(root, "manifest.v1").empty());

  ASSERT_TRUE(service.rebuild_index("idx_standalone_spill"));
  vector_index::index_service::index_config config;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  ASSERT_TRUE(service.describe_index("idx_standalone_spill", &config, nullptr,
                                     &entry_count, &committed_entry_count));
  EXPECT_EQ(1U, entry_count);
  EXPECT_EQ(0U, committed_entry_count);
  ASSERT_TRUE(service.search("idx_standalone_spill", {1.0F, 0.0F}, 2,
                             &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  vector_index::index_service::committed_state committed_state;
  ASSERT_TRUE(service.snapshot_committed_state(&committed_state));
  vector_index::index_service recovered;
  ASSERT_TRUE(recovered.register_index_from_strings("idx_standalone_spill", 2,
                                                    "euclidean", "memory",
                                                    "native"));
  ASSERT_TRUE(recovered.set_index_consistency_mode(
      "idx_standalone_spill",
      vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(recovered.restore_committed_state(committed_state));
  ASSERT_TRUE(recovered.search("idx_standalone_spill", {1.0F, 0.0F}, 2,
                               &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
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
  EXPECT_FALSE(service.set_diskann_search_beamwidth("missing", 16));
  EXPECT_FALSE(service.set_diskann_pq_code_budget_size("missing", 1048576));
  ASSERT_TRUE(service.register_index_from_strings("idx_diskann_search_zero", 2,
                                               "euclidean", "external",
                                               "diskann"));
  EXPECT_FALSE(service.set_diskann_search_complexity("idx_diskann_search_zero", 0));
  EXPECT_FALSE(service.set_diskann_search_beamwidth("idx_diskann_search_zero", 0));
  EXPECT_TRUE(service.set_diskann_pq_code_budget_size("idx_diskann_search_zero",
                                                      0));
  EXPECT_TRUE(service.set_diskann_pq_code_budget_size("idx_diskann_search_zero",
                                                      1048576));
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
  EXPECT_FALSE(
      service.set_diskann_pq_code_budget_size("idx_incompatible", 1048576));
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

TEST(VectorIndexServiceTest, LazyExternalRuntimeSysvarControlsReload) {
  BoolGuard guard(&opt_vector_lazy_external_runtime, false);
  const std::string root =
      std::string(testing::TempDir()) + "/vector_service_lazy_sysvar_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  size_t suffix = 0;

  auto exercise_value = [&](bool expect_lazy) {
    opt_vector_lazy_external_runtime = expect_lazy;
    vector_index::index_service service;
    const std::string index_name =
        "idx_lazy_sysvar_" + std::to_string(++suffix);
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
    EXPECT_EQ(expect_lazy ? 0U : 1U, entry_count);
    EXPECT_EQ(1U, committed_entry_count);

    std::vector<vector_index::search_result> result;
    if (expect_lazy) {
      ASSERT_TRUE(service.search_loaded(
          index_name, {static_cast<float>(suffix), 1.0F}, 1, &result));
      EXPECT_TRUE(result.empty());
      ASSERT_TRUE(service.describe_index(index_name, &config, &supports_mutations,
                                         &entry_count,
                                         &committed_entry_count));
      EXPECT_EQ(0U, entry_count);
      result.clear();
      ASSERT_TRUE(service.ensure_runtime_loaded_for_search(index_name));
    }
    ASSERT_TRUE(service.search_loaded(index_name,
                                      {static_cast<float>(suffix), 1.0F}, 1,
                                      &result));
    ASSERT_EQ(1U, result.size());
    EXPECT_EQ(suffix, result[0].doc_id);
  };

  exercise_value(true);
  exercise_value(false);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     LazyExternalRuntimeCoversDisabledAndAlreadyLoadedBranches) {
  BoolGuard guard(&opt_vector_lazy_external_runtime, false);

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

  opt_vector_lazy_external_runtime = true;

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
  vector_index::index_service service;
  vector_index::index_service::index_config config{
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kDiskAnn, ""};
  config.diskann_max_degree = 48;
  config.diskann_build_complexity = 96;
  config.diskann_search_complexity = 72;
  config.diskann_search_beamwidth = 16;
  config.diskann_pq_code_budget_size = 1048576;

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
  EXPECT_EQ(16U, described.diskann_search_beamwidth);
  EXPECT_EQ(1048576U, described.diskann_pq_code_budget_size);
  EXPECT_TRUE(supports_mutations);
  EXPECT_EQ(0U, entry_count);
  EXPECT_EQ(0U, committed_entry_count);
}

TEST(VectorIndexServiceTest, RegisterIndexConfigAppliesBackendSpecificTunings) {
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

  vector_index::index_service::index_config diskann_config{
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kDiskAnn, ""};
  diskann_config.diskann_build_threads = 2;
  diskann_config.diskann_build_mode_value =
      vector_index::diskann_build_mode::kOffline;
  diskann_config.diskann_build_mode_specified = true;
  ASSERT_TRUE(service.register_index("idx_diskann_config", diskann_config));
  ASSERT_TRUE(service.describe_index("idx_diskann_config", &described,
                                     &supports_mutations, &entry_count,
                                     &committed_entry_count));
  EXPECT_EQ(vector_index::backend_provider::kDiskAnn, described.provider);
  EXPECT_EQ(2U, described.diskann_build_threads);
  EXPECT_EQ(vector_index::diskann_build_mode::kOffline,
            described.diskann_build_mode_value);
  EXPECT_TRUE(described.diskann_build_mode_specified);
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

  vector_index::index_service::index_config bad_diskann_pq_on_native =
      bad_diskann_search_on_native;
  bad_diskann_pq_on_native.diskann_search_complexity = 0;
  bad_diskann_pq_on_native.diskann_pq_code_budget_size = 1048576;
  EXPECT_FALSE(service.register_index("idx_bad_diskann_pq_on_native",
                                     bad_diskann_pq_on_native));

  vector_index::index_service::index_config diskann_auto_mode_on_native{
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kNative, ""};
  diskann_auto_mode_on_native.diskann_build_mode_specified = true;
  ASSERT_TRUE(service.register_index("idx_diskann_auto_mode_on_native",
                                    diskann_auto_mode_on_native));
  vector_index::index_service::index_config described;
  bool supports_mutations = false;
  ASSERT_TRUE(service.describe_index("idx_diskann_auto_mode_on_native",
                                     &described, &supports_mutations, nullptr,
                                     nullptr));
  EXPECT_FALSE(described.diskann_build_mode_specified);

  vector_index::index_service::index_config bad_diskann_mode_on_native =
      diskann_auto_mode_on_native;
  bad_diskann_mode_on_native.diskann_build_mode_value =
      vector_index::diskann_build_mode::kOffline;
  EXPECT_FALSE(service.register_index("idx_bad_diskann_mode_on_native",
                                     bad_diskann_mode_on_native));

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

  ASSERT_TRUE(service.register_index_from_strings("idx_guard_standalone", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_guard_standalone",
      vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(service.direct_upsert("idx_guard_standalone", 7,
                                    {7.0F, 7.0F}));
  vector_index::index_service::index_config standalone_config = config;
  standalone_config.consistency_mode =
      vector_index::index_consistency_mode::kStandalone;
  ASSERT_TRUE(
      service.restore_index_config("idx_guard_standalone", standalone_config));
  ASSERT_TRUE(service.rebuild_index("idx_guard_standalone"));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_guard_standalone", {7.0F, 7.0F}, 1,
                             &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);
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
