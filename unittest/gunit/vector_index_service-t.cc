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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "sql/vector/vector_diskann_generation_store.h"
#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_runtime_thread_pool.h"
#include "sql/vector/vector_index_service.h"
#include "sql/vector/vector_index_service_internal.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_load_file.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_index_service_unittest {

namespace {

using vector_gunit::BoolGuard;
using vector_gunit::ScopedDebugFlag;
using vector_gunit::UlongGuard;
using vector_gunit::UlonglongGuard;

// Service unit tests use the exact native backend to isolate service behavior
// from optional third-party libraries in both Debug and Release builds.
const vector_gunit::NativeProviderGuard native_provider_guard;

bool has_prefix(const std::string &text, const std::string &prefix) {
  return text.size() >= prefix.size() &&
         text.compare(0, prefix.size(), prefix) == 0;
}

bool has_suffix(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

TEST(VectorIndexServiceHelperTest, MergesSegmentBuildDiagnostics) {
  vector_index::backend_build_diagnostics aggregate;
  vector_index::backend_build_diagnostics first;
  first.diskann_pq_runtime = "native_strict";
  first.native_pq_runtime_selected_path = "avx2";
  first.native_pq_runtime_centroid_scan_kernel = "avx2";
  first.native_pq_runtime_memory_adjustment = "none";
  first.native_pq_runtime_bridge = "cpp_main";
  first.native_pq_runtime_artifact_validation = "ok";
  first.native_pq_runtime_artifacts_written = true;
  first.native_pq_runtime_artifacts_consumed = true;
  first.native_pq_runtime_official_pq_used = false;
  first.effective_disk_pq_dims = 64;
  first.native_pq_runtime_effective_threads = 8;
  first.native_pq_runtime_train_ms = 11;
  vector_index::detail::merge_segment_build_diagnostics(100, first,
                                                         &aggregate);

  EXPECT_EQ(100U, aggregate.row_count);
  EXPECT_EQ(1U, aggregate.segment_count);
  EXPECT_EQ(1U, aggregate.build_invocations);
  EXPECT_EQ("native_strict", aggregate.diskann_pq_runtime);
  EXPECT_EQ("avx2", aggregate.native_pq_runtime_selected_path);
  EXPECT_EQ("avx2", aggregate.native_pq_runtime_centroid_scan_kernel);
  EXPECT_EQ("none", aggregate.native_pq_runtime_memory_adjustment);
  EXPECT_EQ("cpp_main", aggregate.native_pq_runtime_bridge);
  EXPECT_EQ("ok", aggregate.native_pq_runtime_artifact_validation);
  EXPECT_EQ(11U, aggregate.native_pq_runtime_train_ms);

  vector_index::backend_build_diagnostics second = first;
  second.row_count = 90;
  second.segment_count = 2;
  second.build_invocations = 3;
  second.diskann_pq_runtime = "official";
  second.native_pq_runtime_selected_path = "scalar_fallback";
  second.native_pq_runtime_centroid_scan_kernel = "scalar_fallback";
  second.native_pq_runtime_memory_adjustment = "reduced_threads";
  second.native_pq_runtime_bridge = "fallback";
  second.native_pq_runtime_artifact_validation = "failed";
  second.native_pq_runtime_official_pq_used = true;
  second.native_pq_runtime_validation_failed = true;
  second.native_pq_runtime_validation_failed_doc_id = "17";
  second.native_pq_runtime_validation_best_doc_id = "19";
  second.native_pq_runtime_validation_result_count = 8;
  second.native_pq_runtime_validation_best_search_distance = "0.25";
  second.native_pq_runtime_validation_best_exact_distance = "0.5";
  second.native_pq_runtime_validation_self_pq_distance = "0.75";
  second.native_pq_runtime_validation_pivots_checksum = "101";
  second.native_pq_runtime_validation_compressed_checksum = "202";
  second.effective_disk_pq_dims = 128;
  second.native_pq_runtime_effective_threads = 4;
  second.native_pq_runtime_train_ms = 13;
  second.fallback_reason = "bridge_failed";
  vector_index::detail::merge_segment_build_diagnostics(100, second,
                                                         &aggregate);

  EXPECT_EQ(190U, aggregate.row_count);
  EXPECT_EQ(3U, aggregate.segment_count);
  EXPECT_EQ(4U, aggregate.build_invocations);
  EXPECT_EQ("mixed", aggregate.diskann_pq_runtime);
  EXPECT_EQ("mixed", aggregate.native_pq_runtime_selected_path);
  EXPECT_EQ("mixed", aggregate.native_pq_runtime_centroid_scan_kernel);
  EXPECT_EQ("mixed", aggregate.native_pq_runtime_memory_adjustment);
  EXPECT_EQ("mixed", aggregate.native_pq_runtime_bridge);
  EXPECT_EQ("mixed", aggregate.native_pq_runtime_artifact_validation);
  EXPECT_EQ(24U, aggregate.native_pq_runtime_train_ms);
  EXPECT_EQ(128U, aggregate.effective_disk_pq_dims);
  EXPECT_EQ(8U, aggregate.native_pq_runtime_effective_threads);
  EXPECT_TRUE(aggregate.native_pq_runtime_artifacts_written);
  EXPECT_TRUE(aggregate.native_pq_runtime_artifacts_consumed);
  EXPECT_TRUE(aggregate.native_pq_runtime_official_pq_used);
  EXPECT_TRUE(aggregate.native_pq_runtime_validation_failed);
  EXPECT_EQ("17", aggregate.native_pq_runtime_validation_failed_doc_id);
  EXPECT_EQ("19", aggregate.native_pq_runtime_validation_best_doc_id);
  EXPECT_EQ(8U, aggregate.native_pq_runtime_validation_result_count);
  EXPECT_EQ("0.25",
            aggregate.native_pq_runtime_validation_best_search_distance);
  EXPECT_EQ("0.5", aggregate.native_pq_runtime_validation_best_exact_distance);
  EXPECT_EQ("0.75", aggregate.native_pq_runtime_validation_self_pq_distance);
  EXPECT_EQ("101", aggregate.native_pq_runtime_validation_pivots_checksum);
  EXPECT_EQ("202", aggregate.native_pq_runtime_validation_compressed_checksum);
  EXPECT_EQ("bridge_failed", aggregate.fallback_reason);

  vector_index::backend_build_diagnostics empty_strings;
  vector_index::detail::merge_segment_build_diagnostics(
      1, empty_strings, &aggregate);
  EXPECT_EQ("mixed", aggregate.diskann_pq_runtime);
  EXPECT_EQ("mixed", aggregate.native_pq_runtime_selected_path);
  EXPECT_EQ("mixed", aggregate.native_pq_runtime_centroid_scan_kernel);
  EXPECT_EQ("mixed", aggregate.native_pq_runtime_memory_adjustment);
  EXPECT_EQ("mixed", aggregate.native_pq_runtime_bridge);
  EXPECT_EQ("mixed", aggregate.native_pq_runtime_artifact_validation);
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

const char *segmented_memory_provider_for_testing() {
#ifdef HAVE_HNSWLIB
  return "hnsw";
#else
#ifndef NDEBUG
  return "native";
#else
  return nullptr;
#endif
#endif
}

void install_diskann_offline_test_adapter() {
#if defined(MYSQL_VECTOR_DISKANN_OFFLINE_TEST_LIB) && \
    !defined(MYSQL_VECTOR_DISKANN_OFFLINE_STATIC_LINKED)
  vector_index::diskann_reset_offline_adapter_path_for_testing();
  vector_index::diskann_set_offline_adapter_path_for_testing(
      MYSQL_VECTOR_DISKANN_OFFLINE_TEST_LIB);
#endif
}

struct diskann_serial_test_adapter_installer {
  diskann_serial_test_adapter_installer() {
#ifdef MYSQL_VECTOR_DISKANN_TEST_LIB
    vector_index::diskann_reset_adapter_path_for_testing();
    vector_index::diskann_set_adapter_path_for_testing(
        MYSQL_VECTOR_DISKANN_TEST_LIB);
#endif
  }
};

[[maybe_unused]] diskann_serial_test_adapter_installer
    install_diskann_serial_test_adapter;

class configurable_backend final : public vector_index::backend {
 public:
  explicit configurable_backend(vector_index::backend_provider provider)
      : m_provider(provider),
        m_storage(2, vector_index::metric_type::kEuclidean) {}

  bool upsert(uint64_t doc_id,
              const vector_index::vector_data &vector) override {
    return m_storage.upsert(doc_id, vector);
  }
  bool erase(uint64_t doc_id) override { return m_storage.erase(doc_id); }
  bool search(
      const vector_index::vector_data &query, size_t top_k,
      std::vector<vector_index::search_result> *results) const override {
    return m_storage.search(query, top_k, results);
  }
  size_t entry_count() const override { return m_storage.entry_count(); }
  size_t dimension() const override { return m_storage.dimension(); }
  vector_index::metric_type metric() const override {
    return m_storage.metric();
  }
  vector_index::backend_mode mode() const override {
    return vector_index::backend_mode::kMemory;
  }
  vector_index::backend_provider provider() const override {
    return m_provider;
  }
  bool supports_mutations() const override { return true; }

  bool set_search_ef(uint32_t value) override {
    if (!accept_setters) return false;
    search_ef_value = value;
    return true;
  }
  bool set_faiss_ivf_params(uint32_t nlist, uint32_t nprobe) override {
    if (!accept_setters) return false;
    faiss_nlist = nlist;
    faiss_nprobe = nprobe;
    return true;
  }
  bool set_faiss_ivf_pq_params(uint32_t nlist, uint32_t nprobe, uint32_t pq_m,
                               uint32_t pq_bits) override {
    if (!accept_setters) return false;
    faiss_nlist = nlist;
    faiss_nprobe = nprobe;
    faiss_pq_m = pq_m;
    faiss_pq_bits = pq_bits;
    return true;
  }
  bool set_diskann_build_params(uint32_t max_degree, uint32_t build_complexity,
                                uint32_t build_threads) override {
    if (!accept_setters) return false;
    diskann_max_degree = max_degree;
    diskann_build_complexity = build_complexity;
    diskann_build_threads = build_threads;
    return true;
  }
  bool set_diskann_build_threads(uint32_t build_threads) override {
    if (!accept_setters) return false;
    diskann_build_threads = build_threads;
    return true;
  }
  bool set_diskann_build_mode(
      vector_index::diskann_build_mode build_mode) override {
    if (!accept_setters) return false;
    diskann_build_mode_value = build_mode;
    return true;
  }
  bool set_diskann_search_complexity(uint32_t search_complexity) override {
    if (!accept_setters) return false;
    diskann_search_complexity = search_complexity;
    return true;
  }
  bool set_diskann_search_beamwidth(uint32_t search_beamwidth) override {
    if (!accept_setters) return false;
    diskann_search_beamwidth = search_beamwidth;
    return true;
  }
  bool set_diskann_pq_code_budget_size(uint64_t budget_size) override {
    if (!accept_setters) return false;
    diskann_pq_code_budget_size = budget_size;
    return true;
  }
  bool set_diskann_disk_pq_dims(uint32_t disk_pq_dims) override {
    if (!accept_setters) return false;
    diskann_disk_pq_dims = disk_pq_dims;
    return true;
  }
  bool set_diskann_accelerate_build(bool accelerate_build) override {
    if (!accept_setters) return false;
    diskann_accelerate_build = accelerate_build;
    return true;
  }
  bool set_diskann_shuffle_build(bool shuffle_build) override {
    if (!accept_setters) return false;
    diskann_shuffle_build = shuffle_build;
    return true;
  }
  bool set_diskann_use_bfs_cache(bool use_bfs_cache) override {
    if (!accept_setters) return false;
    diskann_use_bfs_cache = use_bfs_cache;
    return true;
  }

  bool accept_setters{true};
  uint32_t search_ef_value{0};
  uint32_t faiss_nlist{0};
  uint32_t faiss_nprobe{0};
  uint32_t faiss_pq_m{0};
  uint32_t faiss_pq_bits{0};
  uint32_t diskann_max_degree{0};
  uint32_t diskann_build_complexity{0};
  uint32_t diskann_build_threads{0};
  vector_index::diskann_build_mode diskann_build_mode_value{
      vector_index::diskann_build_mode::kAuto};
  uint32_t diskann_search_complexity{0};
  uint32_t diskann_search_beamwidth{0};
  uint64_t diskann_pq_code_budget_size{0};
  uint32_t diskann_disk_pq_dims{0};
  bool diskann_accelerate_build{false};
  bool diskann_shuffle_build{false};
  bool diskann_use_bfs_cache{false};

 private:
  vector_index::backend_provider m_provider;
  vector_index::memory_backend m_storage;
};

struct segmented_search_concurrency_probe {
  void enter() {
    const size_t active = active_searches.fetch_add(1) + 1;
    size_t peak = peak_searches.load();
    while (active > peak &&
           !peak_searches.compare_exchange_weak(peak, active)) {
    }

    std::unique_lock<std::mutex> lock(m_gate_mutex);
    m_gate_cv.notify_all();
    m_gate_cv.wait(lock, [this]() { return !m_block_searches; });
  }

  void leave() { active_searches.fetch_sub(1); }

  void block_searches() {
    std::lock_guard<std::mutex> lock(m_gate_mutex);
    m_block_searches = true;
  }

  bool wait_for_active_searches(size_t count,
                                std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(m_gate_mutex);
    return m_gate_cv.wait_for(lock, timeout, [this, count]() {
      return active_searches.load() >= count;
    });
  }

  void release_searches() {
    {
      std::lock_guard<std::mutex> lock(m_gate_mutex);
      m_block_searches = false;
    }
    m_gate_cv.notify_all();
  }

  std::atomic<size_t> active_searches{0};
  std::atomic<size_t> peak_searches{0};
  std::atomic<size_t> search_calls{0};
  std::atomic<size_t> search_batch_calls{0};

 private:
  std::mutex m_gate_mutex;
  std::condition_variable m_gate_cv;
  bool m_block_searches{false};
};

class segmented_search_probe_backend final : public vector_index::backend {
 public:
  segmented_search_probe_backend(
      uint64_t doc_id,
      std::shared_ptr<segmented_search_concurrency_probe> probe,
      vector_index::backend_provider provider =
          vector_index::backend_provider::kHnswlib)
      : m_doc_id(doc_id), m_probe(std::move(probe)), m_provider(provider) {}

  bool upsert(uint64_t doc_id [[maybe_unused]],
              const vector_index::vector_data &vector
                  [[maybe_unused]]) override {
    return false;
  }

  bool erase(uint64_t doc_id [[maybe_unused]]) override { return false; }

  bool search(
      const vector_index::vector_data &query, size_t top_k,
      std::vector<vector_index::search_result> *results) const override {
    if (results == nullptr || query.size() != dimension()) return false;
    m_probe->search_calls.fetch_add(1);
    m_probe->enter();
    m_probe->leave();
    results->clear();
    if (top_k != 0) results->push_back({m_doc_id, 0.0});
    return true;
  }

  bool search_batch(
      const std::vector<vector_index::vector_data> &queries, size_t top_k,
      std::vector<std::vector<vector_index::search_result>> *results)
      const override {
    m_probe->search_batch_calls.fetch_add(1);
    return backend::search_batch(queries, top_k, results);
  }

  size_t entry_count() const override { return 1; }
  size_t dimension() const override { return 2; }
  vector_index::metric_type metric() const override {
    return vector_index::metric_type::kEuclidean;
  }
  vector_index::backend_mode mode() const override {
    return vector_index::backend_mode::kMemory;
  }
  vector_index::backend_provider provider() const override {
    return m_provider;
  }
  bool supports_mutations() const override { return false; }

 private:
  uint64_t m_doc_id;
  std::shared_ptr<segmented_search_concurrency_probe> m_probe;
  vector_index::backend_provider m_provider;
};

class shared_search_worker_pool_guard {
 public:
  shared_search_worker_pool_guard() {
    vector_index::reset_shared_search_worker_pool_for_testing();
  }

  ~shared_search_worker_pool_guard() {
    vector_index::reset_shared_search_worker_pool_for_testing();
  }
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

  bool load_segment_tasks(
      std::vector<vector_index_metadata_store::segment_task_row> *rows)
      override {
    if (rows == nullptr) return false;
    *rows = m_segment_tasks;
    return true;
  }

  bool save_segment_tasks(
      const std::vector<vector_index_metadata_store::segment_task_row> &rows)
      override {
    m_segment_tasks = rows;
    return true;
  }

  bool quarantine_segment_tasks() override {
    m_segment_tasks.clear();
    return true;
  }

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

  bool load_manifest(vector_index_metadata_store::manifest_row *row) override {
    if (row == nullptr) return false;
    *row = vector_index_metadata_store::manifest_row();
    return true;
  }

  bool save_manifest(
      const vector_index_metadata_store::manifest_row &) override {
    return true;
  }

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

 private:
  std::vector<vector_index_metadata_store::committed_row> m_rows;
  std::vector<vector_index_metadata_store::segment_task_row> m_segment_tasks;
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

class build_pipeline_options_guard {
 public:
  build_pipeline_options_guard()
      : m_mode(opt_vector_build_pipeline_mode),
        m_min_rows(opt_vector_build_pipeline_min_rows),
        m_min_size(opt_vector_build_pipeline_min_size),
        m_segment_max_rows(opt_vector_build_segment_max_rows),
        m_segment_target_size(opt_vector_build_segment_target_size),
        m_max_tasks(opt_vector_build_pipeline_max_tasks),
        m_progress_interval(opt_vector_build_pipeline_progress_interval),
        m_diskann_segmented_serving(opt_vector_diskann_segmented_serving) {}

  ~build_pipeline_options_guard() {
    opt_vector_build_pipeline_mode = m_mode;
    opt_vector_build_pipeline_min_rows = m_min_rows;
    opt_vector_build_pipeline_min_size = m_min_size;
    opt_vector_build_segment_max_rows = m_segment_max_rows;
    opt_vector_build_segment_target_size = m_segment_target_size;
    opt_vector_build_pipeline_max_tasks = m_max_tasks;
    opt_vector_build_pipeline_progress_interval = m_progress_interval;
    opt_vector_diskann_segmented_serving = m_diskann_segmented_serving;
  }

 private:
  ulong m_mode;
  ulonglong m_min_rows;
  ulonglong m_min_size;
  ulonglong m_segment_max_rows;
  ulonglong m_segment_target_size;
  ulong m_max_tasks;
  ulong m_progress_interval;
  bool m_diskann_segmented_serving;
};

void expect_pipeline_snapshot(
    const vector_index::index_service::build_pipeline_snapshot &snapshot,
    const char *mode, const char *decision, const char *trigger,
    uint64_t row_count, uint64_t payload_size, uint64_t raw_segment_count) {
  EXPECT_EQ(mode, snapshot.mode);
  EXPECT_EQ(decision, snapshot.decision);
  EXPECT_EQ(trigger, snapshot.trigger);
  EXPECT_EQ(row_count, snapshot.row_count);
  EXPECT_EQ(payload_size, snapshot.payload_size);
  EXPECT_EQ(raw_segment_count, snapshot.raw_segment_count);
}

bool prepare_levelled_compaction_index(vector_index::index_service *service,
                                       const std::string &root,
                                       const std::string &index_name,
                                       const char *provider,
                                       std::string *failure) {
  if (failure != nullptr) failure->clear();
  if (service == nullptr || provider == nullptr) {
    if (failure != nullptr) *failure = "invalid setup arguments";
    return false;
  }

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 8, 2,
                      {1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F, 4.0F, 0.0F, 5.0F,
                       0.0F, 6.0F, 0.0F, 7.0F, 0.0F, 8.0F, 0.0F});
  write_raw_docid_file(docid_path, {1, 2, 3, 4, 5, 6, 7, 8});

  if (!service->register_index_from_strings(index_name, 2, "euclidean",
                                            "memory", provider)) {
    if (failure != nullptr) *failure = "index registration failed";
    return false;
  }
  if (!service->set_index_consistency_mode(
          index_name, vector_index::index_consistency_mode::kStandalone)) {
    if (failure != nullptr) *failure = "standalone mode setup failed";
    return false;
  }

  vector_index::index_service::bulk_load_options options;
  uint64_t loaded_rows = 0;
  std::string error;
  if (!service->bulk_upsert_from_raw_files(index_name, vector_path, docid_path,
                                           options, &loaded_rows, &error)) {
    if (failure != nullptr) {
      *failure = error.empty() ? "raw bulk load failed" : error;
    }
    return false;
  }
  if (loaded_rows != 8) {
    if (failure != nullptr) *failure = "unexpected loaded row count";
    return false;
  }
  return true;
}

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

class DeferredMutationBackend final : public vector_index::backend {
 public:
  bool upsert(uint64_t doc_id,
              const vector_index::vector_data &vector) override {
    ++upsert_calls;
    return m_storage.upsert(doc_id, vector);
  }
  bool erase(uint64_t doc_id) override {
    ++erase_calls;
    return m_storage.erase(doc_id);
  }
  bool search(const vector_index::vector_data &query, size_t top_k,
              std::vector<vector_index::search_result> *results) const override {
    return m_storage.search(query, top_k, results);
  }
  size_t entry_count() const override { return m_storage.entry_count(); }
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
  bool supports_mutations() const override { return true; }
  bool supports_in_place_mutations() const override { return false; }

  size_t upsert_calls{0};
  size_t erase_calls{0};

 private:
  vector_index::memory_backend m_storage{
      2, vector_index::metric_type::kEuclidean};
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

class ArtifactLifecycleProbeBackend final : public vector_index::backend {
 public:
  explicit ArtifactLifecycleProbeBackend(bool fail_publish = false,
                                         size_t entry_count = 0)
      : m_entry_count(entry_count), m_fail_publish(fail_publish) {}

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
    return vector_index::backend_mode::kExternal;
  }
  vector_index::backend_provider provider() const override {
    return vector_index::backend_provider::kDiskAnn;
  }
  bool supports_mutations() const override { return false; }
  size_t entry_count() const override { return m_entry_count; }

  bool defer_artifact_publication() override {
    ++defer_calls;
    return true;
  }
  bool has_pending_artifact_publication() const override { return m_pending; }
  bool prepare_artifact_publication(
      const vector_index::diskann_artifact_identity &identity) override {
    ++prepare_calls;
    prepared_identity = identity;
    return true;
  }
  bool publish_artifact() override {
    ++publish_calls;
    return !m_fail_publish;
  }
  bool rollback_artifact() override {
    ++rollback_calls;
    m_pending = false;
    return true;
  }
  bool finalize_artifact() override {
    ++finalize_calls;
    m_pending = false;
    return true;
  }

  size_t defer_calls{0};
  size_t prepare_calls{0};
  size_t publish_calls{0};
  size_t rollback_calls{0};
  size_t finalize_calls{0};
  vector_index::diskann_artifact_identity prepared_identity;

 private:
  size_t m_entry_count{0};
  bool m_fail_publish{false};
  bool m_pending{true};
};

class FailAfterMutationBackend final : public vector_index::backend {
 public:
  void fail_after_successful_mutations(size_t successful_mutations) {
    m_successful_mutations_before_failure = successful_mutations;
    m_mutation_count = 0;
  }

  bool upsert(uint64_t doc_id,
              const vector_index::vector_data &vector) override {
    if (should_fail_mutation()) return false;
    return m_backend.upsert(doc_id, vector);
  }

  bool erase(uint64_t doc_id) override {
    if (should_fail_mutation()) return false;
    return m_backend.erase(doc_id);
  }

  bool search(const vector_index::vector_data &query, size_t top_k,
              std::vector<vector_index::search_result> *results) const override {
    return m_backend.search(query, top_k, results);
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

 private:
  bool should_fail_mutation() {
    if (m_mutation_count == m_successful_mutations_before_failure) return true;
    ++m_mutation_count;
    return false;
  }

  vector_index::memory_backend m_backend{
      2, vector_index::metric_type::kEuclidean};
  size_t m_successful_mutations_before_failure{
      std::numeric_limits<size_t>::max()};
  size_t m_mutation_count{0};
};

class RerankProbeDiskAnnBackend final : public vector_index::backend {
 public:
  explicit RerankProbeDiskAnnBackend(uint32_t search_complexity = 4)
      : m_search_complexity(search_complexity) {}

  bool upsert(uint64_t doc_id,
              const vector_index::vector_data &vector) override {
    m_entries[doc_id] = vector;
    return true;
  }

  bool erase(uint64_t doc_id) override {
    m_entries.erase(doc_id);
    return true;
  }

  bool search(const vector_index::vector_data &query [[maybe_unused]],
              size_t top_k,
              std::vector<vector_index::search_result> *results) const override {
    if (results == nullptr) return false;
    requested_top_k_values.push_back(top_k);
    results->clear();
    results->push_back({1, 0.0});
    results->push_back({2, 1.0});
    if (results->size() > top_k) results->resize(top_k);
    return true;
  }

  bool search_for_rerank(
      const vector_index::vector_data &query [[maybe_unused]], size_t top_k,
      size_t candidate_top_k,
      std::vector<vector_index::search_result> *results) const override {
    rerank_top_k_values.push_back(top_k);
    rerank_candidate_top_k_values.push_back(candidate_top_k);
    return search(query, candidate_top_k, results);
  }

  bool search_batch(
      const std::vector<vector_index::vector_data> &queries, size_t top_k,
      std::vector<std::vector<vector_index::search_result>> *results)
      const override {
    if (results == nullptr) return false;
    requested_top_k_values.push_back(top_k);
    results->clear();
    results->reserve(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
      std::vector<vector_index::search_result> row{{1, 0.0}, {2, 1.0}};
      if (row.size() > top_k) row.resize(top_k);
      results->push_back(std::move(row));
    }
    return true;
  }

  bool search_batch_for_rerank(
      const std::vector<vector_index::vector_data> &queries, size_t top_k,
      size_t candidate_top_k,
      vector_index::batch_search_candidates *results) const override {
    if (results == nullptr) return false;
    rerank_top_k_values.push_back(top_k);
    rerank_candidate_top_k_values.push_back(candidate_top_k);
    std::vector<std::vector<vector_index::search_result>> per_query_results;
    if (!search_batch(queries, candidate_top_k, &per_query_results)) {
      return false;
    }
    results->set_per_query(std::move(per_query_results));
    return true;
  }

  size_t entry_count() const override { return m_entries.size(); }
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
  bool supports_mutations() const override { return true; }
  uint32_t diskann_search_complexity() const override {
    return m_search_complexity;
  }

  mutable std::vector<size_t> requested_top_k_values;
  mutable std::vector<size_t> rerank_top_k_values;
  mutable std::vector<size_t> rerank_candidate_top_k_values;

 private:
  uint32_t m_search_complexity{4};
  std::unordered_map<uint64_t, vector_index::vector_data> m_entries;
};

class SearchOptionsProbeDiskAnnBackend final : public vector_index::backend {
 public:
  bool upsert(uint64_t doc_id [[maybe_unused]],
              const vector_index::vector_data &vector
              [[maybe_unused]]) override {
    return false;
  }

  bool erase(uint64_t doc_id [[maybe_unused]]) override { return false; }

  bool search(
      const vector_index::vector_data &query [[maybe_unused]],
      size_t top_k [[maybe_unused]],
      std::vector<vector_index::search_result> *results) const override {
    return record_search(nullptr, results);
  }

  bool search_with_options(
      const vector_index::vector_data &query [[maybe_unused]],
      size_t top_k [[maybe_unused]],
      const vector_index::backend_search_options &options,
      std::vector<vector_index::search_result> *results) const override {
    return record_search(&options, results);
  }

  bool search_batch(
      const std::vector<vector_index::vector_data> &queries,
      size_t top_k [[maybe_unused]],
      std::vector<std::vector<vector_index::search_result>> *results)
      const override {
    return record_batch_search(queries.size(), nullptr, results);
  }

  bool search_batch_with_options(
      const std::vector<vector_index::vector_data> &queries,
      size_t top_k [[maybe_unused]],
      const vector_index::backend_search_options &options,
      std::vector<std::vector<vector_index::search_result>> *results)
      const override {
    return record_batch_search(queries.size(), &options, results);
  }

  size_t entry_count() const override { return 256; }
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

  bool set_diskann_search_complexity(uint32_t search_complexity) override {
    ++m_setter_calls;
    m_search_complexity.store(search_complexity, std::memory_order_relaxed);
    return true;
  }

  uint32_t diskann_search_complexity() const override {
    return m_search_complexity.load(std::memory_order_relaxed);
  }

  void reset() {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_arrivals = 0;
    m_observed_options.clear();
  }

  size_t setter_calls() const {
    return m_setter_calls.load(std::memory_order_relaxed);
  }

  std::vector<vector_index::backend_search_options> observed_options() const {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_observed_options;
  }

 private:
  bool wait_and_record(
      const vector_index::backend_search_options *options) const {
    std::unique_lock<std::mutex> guard(m_mutex);
    ++m_arrivals;
    m_cv.notify_all();
    if (!m_cv.wait_for(guard, std::chrono::seconds(5),
                       [&]() { return m_arrivals >= 2; })) {
      return false;
    }

    vector_index::backend_search_options observed;
    if (options == nullptr) {
      observed.diskann_search_complexity =
          m_search_complexity.load(std::memory_order_relaxed);
      observed.diskann_search_beamwidth = 16;
    } else {
      observed = *options;
    }
    m_observed_options.push_back(observed);
    return true;
  }

  bool record_search(
      const vector_index::backend_search_options *options,
      std::vector<vector_index::search_result> *results) const {
    if (results == nullptr || !wait_and_record(options)) return false;
    results->assign({{1, 0.0}});
    return true;
  }

  bool record_batch_search(
      size_t query_count, const vector_index::backend_search_options *options,
      std::vector<std::vector<vector_index::search_result>> *results) const {
    if (results == nullptr || !wait_and_record(options)) return false;
    results->assign(query_count, {{1, 0.0}});
    return true;
  }

  mutable std::mutex m_mutex;
  mutable std::condition_variable m_cv;
  mutable size_t m_arrivals{0};
  mutable std::vector<vector_index::backend_search_options> m_observed_options;
  std::atomic<uint32_t> m_search_complexity{4};
  std::atomic<size_t> m_setter_calls{0};
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

TEST(VectorIndexServiceTest,
     PublicationIdentityAndGenerationsRejectStaleIndexInstances) {
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings(
      "idx_generation", 2, "euclidean", "memory", "native"));

  vector_index::index_service::index_publication_state first;
  ASSERT_TRUE(service.describe_publication_state("idx_generation", &first));
  EXPECT_NE(0U, first.index_identity);
  EXPECT_EQ(0U, first.truth_generation);
  EXPECT_EQ(1U, first.config_generation);
  EXPECT_EQ(0U, first.runtime_generation);

  ASSERT_TRUE(service.stage_upsert(51, "idx_generation", 7, {7.0F, 7.0F}));
  ASSERT_TRUE(service.commit(51));
  vector_index::index_service::index_publication_state committed;
  ASSERT_TRUE(service.describe_publication_state("idx_generation", &committed));
  EXPECT_EQ(first.index_identity, committed.index_identity);
  EXPECT_EQ(1U, committed.truth_generation);
  EXPECT_EQ(committed.truth_generation, committed.runtime_generation);

  ASSERT_TRUE(service.drop_index("idx_generation"));
  ASSERT_TRUE(service.register_index_from_strings(
      "idx_generation", 2, "euclidean", "memory", "native"));
  vector_index::index_service::index_publication_state recreated;
  ASSERT_TRUE(service.describe_publication_state("idx_generation", &recreated));
  EXPECT_GT(recreated.index_identity, committed.index_identity);

  vector_index::index_service::index_publication_state invalid = recreated;
  invalid.runtime_generation = invalid.truth_generation + 1;
  EXPECT_FALSE(service.restore_publication_state("idx_generation", invalid));
  EXPECT_FALSE(service.restore_next_index_identity(recreated.index_identity));

  ASSERT_TRUE(service.register_index_from_strings(
      "idx_generation_other", 2, "euclidean", "memory", "native"));
  vector_index::index_service::index_publication_state duplicate = recreated;
  EXPECT_FALSE(
      service.restore_publication_state("idx_generation_other", duplicate));
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
  const uint64_t generation_after_insert = store.generation("idx_a");
  ASSERT_TRUE(store.upsert("idx_a", 1, {1.0F, 1.0F}));
  EXPECT_EQ(generation_after_insert, store.generation("idx_a"));
  ASSERT_TRUE(store.erase("idx_a", 99));
  EXPECT_EQ(generation_after_insert, store.generation("idx_a"));
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
  EXPECT_EQ(1U, committed_entry_count);
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

TEST(VectorIndexServiceTest, BuildPipelineRecordsDefaultDirectDecision) {
  build_pipeline_options_guard guard;
  const std::string root =
      std::string(testing::TempDir()) + "/pipeline_direct_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);
  UlonglongGuard entry_cache_guard(&opt_vector_entry_cache_size, 0);

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_pipeline_direct", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_pipeline_direct",
      vector_index::index_consistency_mode::kStandalone));
  EXPECT_TRUE(service.standalone_build_source("missing_index").empty());
  EXPECT_EQ("memory", service.standalone_build_source("idx_pipeline_direct"));
  ASSERT_TRUE(service.direct_upsert("idx_pipeline_direct", 11,
                                    {1.0F, 0.0F}));
  ASSERT_TRUE(service.rebuild_index("idx_pipeline_direct"));

  vector_index::index_service::build_pipeline_snapshot snapshot;
  ASSERT_TRUE(
      service.describe_build_pipeline("idx_pipeline_direct", &snapshot));
  expect_pipeline_snapshot(snapshot, "auto", "direct", "below_threshold", 1, 8,
                           0);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     SegmentedBackendDelegatesSearchAndProviderTuningsToSegments) {
  vector_index::index_service::index_config search_config;
  search_config.dimension = 2;
  search_config.metric = vector_index::metric_type::kEuclidean;
  search_config.mode = vector_index::backend_mode::kMemory;
  search_config.provider = vector_index::backend_provider::kNative;
  auto first_segment = std::make_shared<configurable_backend>(
      vector_index::backend_provider::kNative);
  auto second_segment = std::make_shared<configurable_backend>(
      vector_index::backend_provider::kNative);
  ASSERT_TRUE(first_segment->upsert(1, {1.0F, 1.0F}));
  ASSERT_TRUE(second_segment->upsert(9, {9.0F, 9.0F}));
  auto search_backend = vector_index::make_segmented_backend_for_testing(
      search_config, {first_segment, second_segment});
  EXPECT_FALSE(search_backend->supports_mutations());
  EXPECT_FALSE(search_backend->upsert(3, {3.0F, 3.0F}));
  EXPECT_FALSE(search_backend->erase(1));

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(search_backend->search({1.0F, 1.0F}, 2, &results));
  ASSERT_EQ(2U, results.size());
  EXPECT_EQ(1U, results[0].doc_id);
  EXPECT_EQ(9U, results[1].doc_id);

  std::vector<std::vector<vector_index::search_result>> batch_results;
  ASSERT_TRUE(search_backend->search_batch({{1.0F, 1.0F}, {9.0F, 9.0F}}, 1,
                                           &batch_results));
  ASSERT_EQ(2U, batch_results.size());
  ASSERT_EQ(1U, batch_results[0].size());
  ASSERT_EQ(1U, batch_results[1].size());
  EXPECT_EQ(1U, batch_results[0][0].doc_id);
  EXPECT_EQ(9U, batch_results[1][0].doc_id);

  vector_index::index_service::index_config hnsw_config = search_config;
  hnsw_config.provider = vector_index::backend_provider::kHnswlib;
  auto hnsw_segment = std::make_shared<configurable_backend>(
      vector_index::backend_provider::kHnswlib);
  auto hnsw_backend = vector_index::make_segmented_backend_for_testing(
      hnsw_config, {hnsw_segment});
  EXPECT_TRUE(hnsw_backend->set_search_ef(72));
  EXPECT_EQ(72U, hnsw_backend->search_ef());
  EXPECT_EQ(72U, hnsw_segment->search_ef_value);

  vector_index::index_service::index_config faiss_config = search_config;
  faiss_config.provider = vector_index::backend_provider::kFaiss;
  auto faiss_segment = std::make_shared<configurable_backend>(
      vector_index::backend_provider::kFaiss);
  auto faiss_backend = vector_index::make_segmented_backend_for_testing(
      faiss_config, {faiss_segment});
  EXPECT_TRUE(faiss_backend->set_faiss_ivf_params(16, 4));
  EXPECT_TRUE(faiss_backend->set_faiss_ivf_pq_params(32, 8, 4, 8));
  EXPECT_EQ(32U, faiss_segment->faiss_nlist);
  EXPECT_EQ(8U, faiss_segment->faiss_nprobe);
  EXPECT_EQ(4U, faiss_segment->faiss_pq_m);
  EXPECT_EQ(8U, faiss_segment->faiss_pq_bits);

  auto rejecting_faiss_segment = std::make_shared<configurable_backend>(
      vector_index::backend_provider::kFaiss);
  rejecting_faiss_segment->accept_setters = false;
  auto rejecting_faiss_backend =
      vector_index::make_segmented_backend_for_testing(
          faiss_config, {faiss_segment, rejecting_faiss_segment});
  EXPECT_FALSE(rejecting_faiss_backend->set_faiss_ivf_params(64, 16));

  vector_index::index_service::index_config diskann_config = search_config;
  diskann_config.mode = vector_index::backend_mode::kExternal;
  diskann_config.provider = vector_index::backend_provider::kDiskAnn;
  auto diskann_segment = std::make_shared<configurable_backend>(
      vector_index::backend_provider::kDiskAnn);
  auto diskann_backend = vector_index::make_segmented_backend_for_testing(
      diskann_config, {diskann_segment});
  EXPECT_TRUE(diskann_backend->set_diskann_build_params(56, 100, 8));
  EXPECT_TRUE(diskann_backend->set_diskann_build_threads(12));
  EXPECT_TRUE(diskann_backend->set_diskann_build_mode(
      vector_index::diskann_build_mode::kOffline));
  EXPECT_TRUE(diskann_backend->set_diskann_search_complexity(800));
  EXPECT_TRUE(diskann_backend->set_diskann_search_beamwidth(32));
  EXPECT_TRUE(diskann_backend->set_diskann_pq_code_budget_size(4096));
  EXPECT_TRUE(diskann_backend->set_diskann_disk_pq_dims(16));
  EXPECT_TRUE(diskann_backend->set_diskann_accelerate_build(true));
  EXPECT_TRUE(diskann_backend->set_diskann_shuffle_build(true));
  EXPECT_TRUE(diskann_backend->set_diskann_use_bfs_cache(true));
  EXPECT_EQ(56U, diskann_backend->diskann_max_degree());
  EXPECT_EQ(100U, diskann_backend->diskann_build_complexity());
  EXPECT_EQ(12U, diskann_backend->diskann_build_threads());
  EXPECT_EQ(vector_index::diskann_build_mode::kOffline,
            diskann_backend->diskann_build_mode_value());
  EXPECT_EQ(800U, diskann_backend->diskann_search_complexity());
  EXPECT_EQ(32U, diskann_backend->diskann_search_beamwidth());
  EXPECT_EQ(4096U, diskann_backend->diskann_pq_code_budget_size());
  EXPECT_EQ(16U, diskann_backend->diskann_disk_pq_dims());
  EXPECT_TRUE(diskann_backend->diskann_accelerate_build());
  EXPECT_TRUE(diskann_backend->diskann_shuffle_build());
  EXPECT_TRUE(diskann_backend->diskann_use_bfs_cache());
  EXPECT_EQ(56U, diskann_segment->diskann_max_degree);
  EXPECT_EQ(100U, diskann_segment->diskann_build_complexity);
  EXPECT_EQ(12U, diskann_segment->diskann_build_threads);
  EXPECT_EQ(vector_index::diskann_build_mode::kOffline,
            diskann_segment->diskann_build_mode_value);
  EXPECT_EQ(800U, diskann_segment->diskann_search_complexity);
  EXPECT_EQ(32U, diskann_segment->diskann_search_beamwidth);
  EXPECT_EQ(4096U, diskann_segment->diskann_pq_code_budget_size);
  EXPECT_EQ(16U, diskann_segment->diskann_disk_pq_dims);
  EXPECT_TRUE(diskann_segment->diskann_accelerate_build);
  EXPECT_TRUE(diskann_segment->diskann_shuffle_build);
  EXPECT_TRUE(diskann_segment->diskann_use_bfs_cache);
}

TEST(VectorIndexServiceTest,
     SegmentedHnswBatchUsesOneFlattenedSearchExecutionLayer) {
  shared_search_worker_pool_guard search_pool_guard;
  UlongGuard hnsw_search_threads_guard(&opt_vector_hnsw_search_threads, 4);
  auto probe = std::make_shared<segmented_search_concurrency_probe>();
  auto first_segment =
      std::make_shared<segmented_search_probe_backend>(1, probe);
  auto second_segment =
      std::make_shared<segmented_search_probe_backend>(2, probe);

  vector_index::index_service::index_config config;
  config.dimension = 2;
  config.metric = vector_index::metric_type::kEuclidean;
  config.mode = vector_index::backend_mode::kMemory;
  config.provider = vector_index::backend_provider::kHnswlib;
  auto segmented = vector_index::make_segmented_backend_for_testing(
      config, {first_segment, second_segment});

  std::vector<std::vector<vector_index::search_result>> results;
  ASSERT_TRUE(segmented->search_batch(
      {{1.0F, 1.0F}, {2.0F, 2.0F}, {3.0F, 3.0F}}, 1, &results));
  ASSERT_EQ(3U, results.size());
  EXPECT_EQ(0U, probe->search_batch_calls.load());
  EXPECT_EQ(6U, probe->search_calls.load());
  EXPECT_LE(probe->peak_searches.load(), 4U);
  const auto diagnostics = segmented->build_diagnostics();
  EXPECT_EQ(4U, diagnostics.search_worker_budget);
  EXPECT_EQ(1U, diagnostics.search_active_requests);
  EXPECT_EQ(6U, diagnostics.search_work_items);
  EXPECT_EQ(4U, diagnostics.search_fanout_threads);
}

TEST(VectorIndexServiceTest,
     SegmentedArtifactPublicationRollsBackEveryPendingSegment) {
  auto first_segment =
      std::make_shared<ArtifactLifecycleProbeBackend>(false, 2);
  auto failing_segment =
      std::make_shared<ArtifactLifecycleProbeBackend>(true, 3);

  vector_index::index_service::index_config config;
  config.dimension = 2;
  config.metric = vector_index::metric_type::kEuclidean;
  config.mode = vector_index::backend_mode::kExternal;
  config.provider = vector_index::backend_provider::kDiskAnn;
  auto segmented = vector_index::make_segmented_backend_for_testing(
      config, {first_segment, failing_segment});

  vector_index::diskann_artifact_identity identity;
  identity.index_identity = 1;
  identity.truth_generation = 2;
  identity.config_generation = 3;
  identity.doc_id_count = 5;
  identity.dimension = 2;
  identity.provider = "diskann";

  ASSERT_TRUE(segmented->defer_artifact_publication());
  ASSERT_TRUE(segmented->prepare_artifact_publication(identity));
  EXPECT_FALSE(segmented->publish_artifact());
  EXPECT_FALSE(segmented->has_pending_artifact_publication());

  EXPECT_EQ(1U, first_segment->defer_calls);
  EXPECT_EQ(1U, failing_segment->defer_calls);
  EXPECT_EQ(1U, first_segment->prepare_calls);
  EXPECT_EQ(1U, failing_segment->prepare_calls);
  EXPECT_EQ(2U, first_segment->prepared_identity.doc_id_count);
  EXPECT_EQ(3U, failing_segment->prepared_identity.doc_id_count);
  EXPECT_EQ(1U, first_segment->publish_calls);
  EXPECT_EQ(1U, failing_segment->publish_calls);
  EXPECT_EQ(1U, first_segment->rollback_calls);
  EXPECT_EQ(1U, failing_segment->rollback_calls);
}

TEST(VectorIndexServiceTest,
     ConcurrentSegmentedHnswSearchesShareGlobalWorkerBudget) {
  shared_search_worker_pool_guard search_pool_guard;
  constexpr size_t kRequestCount = 8;
  constexpr size_t kWorkerBudget = 4;
  UlongGuard hnsw_search_threads_guard(&opt_vector_hnsw_search_threads,
                                       kWorkerBudget);
  auto probe = std::make_shared<segmented_search_concurrency_probe>();
  probe->block_searches();
  auto first_segment =
      std::make_shared<segmented_search_probe_backend>(1, probe);
  auto second_segment =
      std::make_shared<segmented_search_probe_backend>(2, probe);

  vector_index::index_service::index_config config;
  config.dimension = 2;
  config.metric = vector_index::metric_type::kEuclidean;
  config.mode = vector_index::backend_mode::kMemory;
  config.provider = vector_index::backend_provider::kHnswlib;
  auto segmented = vector_index::make_segmented_backend_for_testing(
      config, {first_segment, second_segment});

  std::atomic<size_t> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::future<bool>> searches;
  searches.reserve(kRequestCount);
  vector_index::backend *const segmented_backend = segmented.get();
  for (size_t i = 0; i < kRequestCount; ++i) {
    searches.push_back(
        std::async(std::launch::async, [&, segmented_backend]() {
          ready.fetch_add(1);
          while (!start.load()) std::this_thread::yield();
          std::vector<vector_index::search_result> results;
          return segmented_backend->search({1.0F, 1.0F}, 1, &results) &&
                 results.size() == 1;
        }));
  }

  const auto ready_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (ready.load() != kRequestCount &&
         std::chrono::steady_clock::now() < ready_deadline) {
    std::this_thread::yield();
  }
  EXPECT_EQ(kRequestCount, ready.load());
  start.store(true);
  EXPECT_TRUE(probe->wait_for_active_searches(2, std::chrono::seconds(2)));
  probe->release_searches();
  for (auto &search : searches) EXPECT_TRUE(search.get());

  EXPECT_GE(probe->peak_searches.load(), 2U);
  EXPECT_LE(probe->peak_searches.load(), kWorkerBudget);
}

TEST(VectorIndexServiceTest,
     SegmentedFaissBatchUsesOneFlattenedSearchExecutionLayer) {
  shared_search_worker_pool_guard search_pool_guard;
  UlongGuard faiss_search_threads_guard(&opt_vector_faiss_search_threads, 4);
  auto probe = std::make_shared<segmented_search_concurrency_probe>();
  auto first_segment = std::make_shared<segmented_search_probe_backend>(
      1, probe, vector_index::backend_provider::kFaiss);
  auto second_segment = std::make_shared<segmented_search_probe_backend>(
      2, probe, vector_index::backend_provider::kFaiss);

  vector_index::index_service::index_config config;
  config.dimension = 2;
  config.metric = vector_index::metric_type::kEuclidean;
  config.mode = vector_index::backend_mode::kExternal;
  config.provider = vector_index::backend_provider::kFaiss;
  auto segmented = vector_index::make_segmented_backend_for_testing(
      config, {first_segment, second_segment});

  std::vector<std::vector<vector_index::search_result>> results;
  ASSERT_TRUE(segmented->search_batch(
      {{1.0F, 1.0F}, {2.0F, 2.0F}, {3.0F, 3.0F}}, 1, &results));
  ASSERT_EQ(3U, results.size());
  EXPECT_EQ(0U, probe->search_batch_calls.load());
  EXPECT_EQ(6U, probe->search_calls.load());
  EXPECT_LE(probe->peak_searches.load(), 4U);
  const auto diagnostics = segmented->build_diagnostics();
  EXPECT_EQ(4U, diagnostics.search_worker_budget);
  EXPECT_EQ(1U, diagnostics.search_active_requests);
  EXPECT_EQ(6U, diagnostics.search_work_items);
  EXPECT_EQ(4U, diagnostics.search_fanout_threads);
}

TEST(VectorIndexServiceTest,
     ConcurrentSegmentedFaissSearchesShareGlobalWorkerBudget) {
  shared_search_worker_pool_guard search_pool_guard;
  constexpr size_t kRequestCount = 8;
  constexpr size_t kWorkerBudget = 4;
  UlongGuard faiss_search_threads_guard(&opt_vector_faiss_search_threads,
                                        kWorkerBudget);
  auto probe = std::make_shared<segmented_search_concurrency_probe>();
  probe->block_searches();
  auto first_segment = std::make_shared<segmented_search_probe_backend>(
      1, probe, vector_index::backend_provider::kFaiss);
  auto second_segment = std::make_shared<segmented_search_probe_backend>(
      2, probe, vector_index::backend_provider::kFaiss);

  vector_index::index_service::index_config config;
  config.dimension = 2;
  config.metric = vector_index::metric_type::kEuclidean;
  config.mode = vector_index::backend_mode::kExternal;
  config.provider = vector_index::backend_provider::kFaiss;
  auto segmented = vector_index::make_segmented_backend_for_testing(
      config, {first_segment, second_segment});

  std::atomic<size_t> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::future<bool>> searches;
  searches.reserve(kRequestCount);
  vector_index::backend *const segmented_backend = segmented.get();
  for (size_t i = 0; i < kRequestCount; ++i) {
    searches.push_back(
        std::async(std::launch::async, [&, segmented_backend]() {
          ready.fetch_add(1);
          while (!start.load()) std::this_thread::yield();
          std::vector<vector_index::search_result> results;
          return segmented_backend->search({1.0F, 1.0F}, 1, &results) &&
                 results.size() == 1;
        }));
  }

  const auto ready_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (ready.load() != kRequestCount &&
         std::chrono::steady_clock::now() < ready_deadline) {
    std::this_thread::yield();
  }
  EXPECT_EQ(kRequestCount, ready.load());
  start.store(true);
  const bool reached_parallel_search = probe->wait_for_active_searches(
      2, std::chrono::seconds(2));
  const bool exceeded_budget = probe->wait_for_active_searches(
      kWorkerBudget + 1, std::chrono::milliseconds(100));
  probe->release_searches();
  for (auto &search : searches) EXPECT_TRUE(search.get());

  EXPECT_TRUE(reached_parallel_search);
  EXPECT_FALSE(exceeded_budget);
  EXPECT_GE(probe->peak_searches.load(), 2U);
  EXPECT_LE(probe->peak_searches.load(), kWorkerBudget);
}

TEST(VectorIndexServiceTest,
     DiskAnnStandaloneOfflineRawSegmentsUseSegmentedBackendByDefault) {
  vector_index::index_service::index_config config;
  config.provider = vector_index::backend_provider::kDiskAnn;
  config.mode = vector_index::backend_mode::kExternal;
  config.diskann_segmented_serving = opt_vector_diskann_segmented_serving;
  config.diskann_build_mode_value = vector_index::diskann_build_mode::kOffline;
  EXPECT_FALSE(
      vector_index::raw_segments_use_single_backend_for_testing(config));

  config.diskann_build_mode_value = vector_index::diskann_build_mode::kAuto;
  EXPECT_FALSE(
      vector_index::raw_segments_use_single_backend_for_testing(config));
}

TEST(VectorIndexServiceTest,
     DiskAnnStandaloneOfflineRawSegmentsCanUseSingleBackendForDiagnostics) {
  vector_index::index_service::index_config config;
  config.provider = vector_index::backend_provider::kDiskAnn;
  config.mode = vector_index::backend_mode::kExternal;
  config.diskann_segmented_serving = false;
  config.diskann_build_mode_value = vector_index::diskann_build_mode::kOffline;
  EXPECT_TRUE(
      vector_index::raw_segments_use_single_backend_for_testing(config));

  config.diskann_build_mode_value = vector_index::diskann_build_mode::kAuto;
  EXPECT_TRUE(
      vector_index::raw_segments_use_single_backend_for_testing(config));

  config.diskann_build_mode_value = vector_index::diskann_build_mode::kSerial;
  EXPECT_FALSE(
      vector_index::raw_segments_use_single_backend_for_testing(config));
}

TEST(VectorIndexServiceTest,
     DiskAnnSegmentedServingDisablesRawSegmentSingleBackend) {
  vector_index::index_service::index_config config;
  config.provider = vector_index::backend_provider::kDiskAnn;
  config.mode = vector_index::backend_mode::kExternal;
  config.diskann_segmented_serving = true;

  config.diskann_build_mode_value = vector_index::diskann_build_mode::kOffline;
  EXPECT_FALSE(
      vector_index::raw_segments_use_single_backend_for_testing(config));

  config.diskann_build_mode_value = vector_index::diskann_build_mode::kAuto;
  EXPECT_FALSE(
      vector_index::raw_segments_use_single_backend_for_testing(config));
}

TEST(VectorIndexServiceTest, NonDiskAnnRawSegmentsKeepSegmentedBackend) {
  vector_index::index_service::index_config config;
  config.mode = vector_index::backend_mode::kExternal;
  config.provider = vector_index::backend_provider::kNative;
  EXPECT_FALSE(
      vector_index::raw_segments_use_single_backend_for_testing(config));

  config.provider = vector_index::backend_provider::kFaiss;
  EXPECT_FALSE(
      vector_index::raw_segments_use_single_backend_for_testing(config));
}

TEST(VectorIndexServiceTest, RawSegmentFileHelpersRejectInvalidInputs) {
  size_t value = 123;
  EXPECT_FALSE(vector_index::parse_manifest_size_for_testing("7", nullptr));
  EXPECT_FALSE(vector_index::parse_manifest_size_for_testing("abc", &value));
  ASSERT_TRUE(vector_index::parse_manifest_size_for_testing("7", &value));
  EXPECT_EQ(7U, value);

  EXPECT_FALSE(vector_index::file_size_as_size_for_testing("missing.file",
                                                           nullptr));
  EXPECT_FALSE(vector_index::file_size_as_size_for_testing("missing.file",
                                                           &value));

  size_t bytes = 1;
  EXPECT_FALSE(vector_index::vector_payload_bytes_for_testing(1, 1, nullptr));
  EXPECT_FALSE(vector_index::vector_payload_bytes_for_testing(
      std::numeric_limits<size_t>::max(), 2, &bytes));
  ASSERT_TRUE(vector_index::vector_payload_bytes_for_testing(5, 0, &bytes));
  EXPECT_EQ(0U, bytes);
  ASSERT_TRUE(vector_index::vector_payload_bytes_for_testing(3, 2, &bytes));
  EXPECT_EQ(3U * 2U * sizeof(float), bytes);

  const std::string root =
      std::string(testing::TempDir()) + "/vector_service_file_helpers";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  ASSERT_TRUE(std::filesystem::create_directories(root, ec));

  const std::string source = root + "/source.bin";
  const std::string target = root + "/nested/target.bin";
  const std::string parent_file = root + "/parent_file";
  write_binary_bytes(source, "payload");
  write_binary_bytes(parent_file, "not a directory");
  EXPECT_FALSE(vector_index::copy_or_link_file_for_testing("", target));
  EXPECT_FALSE(vector_index::copy_or_link_file_for_testing(source, ""));
  EXPECT_FALSE(vector_index::copy_or_link_file_for_testing(
      source, parent_file + "/target.bin"));
  EXPECT_FALSE(vector_index::copy_or_link_file_for_testing(root + "/missing",
                                                           target));
  ASSERT_TRUE(vector_index::copy_or_link_file_for_testing(source, target));
  ASSERT_TRUE(vector_index::file_size_as_size_for_testing(target, &value));
  EXPECT_EQ(7U, value);

  EXPECT_FALSE(vector_index::write_generated_docid_file_for_testing("", 1));
  EXPECT_FALSE(vector_index::write_generated_docid_file_for_testing(root, 1));
  EXPECT_FALSE(vector_index::write_generated_docid_file_for_testing(
      parent_file + "/docids.u64", 1));
  const std::string docids = root + "/docids.u64";
  ASSERT_TRUE(vector_index::write_generated_docid_file_for_testing(docids, 3));
  ASSERT_TRUE(vector_index::file_size_as_size_for_testing(docids, &value));
  EXPECT_EQ(sizeof(uint64_t) * 4U, value);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, BuildPipelineHonorsForcedSegmentedMode) {
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  const std::string root =
      std::string(testing::TempDir()) + "/pipeline_forced_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);
  UlonglongGuard entry_cache_guard(&opt_vector_entry_cache_size, 0);
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_pipeline_forced", 2,
                                                  "euclidean", "memory",
                                                  provider));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_pipeline_forced",
      vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(service.direct_upsert("idx_pipeline_forced", 11,
                                    {1.0F, 0.0F}));
  ASSERT_TRUE(service.rebuild_index("idx_pipeline_forced"));

  vector_index::index_service::build_pipeline_snapshot snapshot;
  ASSERT_TRUE(
      service.describe_build_pipeline("idx_pipeline_forced", &snapshot));
  expect_pipeline_snapshot(snapshot, "segmented", "segmented",
                           "forced_segmented", 1, 8, 0);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_pipeline_forced", {1.0F, 0.0F}, 1,
                             &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);

  ASSERT_TRUE(service.search("idx_pipeline_forced", {1.0F, 0.0F}, 0,
                             &result));
  EXPECT_TRUE(result.empty());
  EXPECT_FALSE(service.search("idx_pipeline_forced", {1.0F}, 1, &result));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, SegmentedBatchSearchKeepsEmptyQuerySlots) {
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  const std::string root =
      std::string(testing::TempDir()) + "/pipeline_empty_batch_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);
  UlonglongGuard entry_cache_guard(&opt_vector_entry_cache_size, 0);
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_pipeline_empty_batch", 2,
                                                  "euclidean", "memory",
                                                  provider));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_pipeline_empty_batch",
      vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(service.rebuild_index("idx_pipeline_empty_batch"));

  std::vector<std::vector<vector_index::search_result>> batch_result;
  ASSERT_TRUE(service.search_batch(
      "idx_pipeline_empty_batch", {{1.0F, 0.0F}, {0.0F, 1.0F}}, 2,
      &batch_result));
  ASSERT_EQ(2U, batch_result.size());
  EXPECT_TRUE(batch_result[0].empty());
  EXPECT_TRUE(batch_result[1].empty());

  ASSERT_TRUE(service.search_batch(
      "idx_pipeline_empty_batch", {{1.0F, 0.0F}, {0.0F, 1.0F}}, 0,
      &batch_result));
  ASSERT_EQ(2U, batch_result.size());
  EXPECT_TRUE(batch_result[0].empty());
  EXPECT_TRUE(batch_result[1].empty());
  EXPECT_FALSE(service.search_batch(
      "idx_pipeline_empty_batch", {{1.0F}, {0.0F, 1.0F}}, 1,
      &batch_result));
}

TEST(VectorIndexServiceTest, DiskAnnSearchTuningsUpdateConfig) {
  build_pipeline_options_guard guard;
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_segmented_diskann", 2,
                                                  "euclidean", "external",
                                                  "diskann"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_segmented_diskann",
      vector_index::index_consistency_mode::kStandalone));

  ASSERT_TRUE(service.set_diskann_search_complexity("idx_segmented_diskann",
                                                    200));
  ASSERT_TRUE(service.set_diskann_search_beamwidth("idx_segmented_diskann",
                                                  32));
  ASSERT_TRUE(service.set_diskann_disk_pq_dims("idx_segmented_diskann", 8));
  ASSERT_TRUE(service.set_diskann_accelerate_build("idx_segmented_diskann",
                                                  true));
  ASSERT_TRUE(service.set_diskann_shuffle_build("idx_segmented_diskann", true));
  ASSERT_TRUE(service.set_diskann_use_bfs_cache("idx_segmented_diskann", true));

  vector_index::index_service::index_config described;
  ASSERT_TRUE(service.describe_index("idx_segmented_diskann", &described,
                                     nullptr, nullptr, nullptr));
  EXPECT_EQ(200U, described.diskann_search_complexity);
  EXPECT_EQ(32U, described.diskann_search_beamwidth);
  EXPECT_EQ(8U, described.diskann_disk_pq_dims);
  EXPECT_TRUE(described.diskann_accelerate_build);
  EXPECT_TRUE(described.diskann_shuffle_build);
  EXPECT_TRUE(described.diskann_use_bfs_cache);
}

TEST(VectorIndexServiceTest, SegmentedFaissTuningsUpdateConfig) {
  build_pipeline_options_guard guard;
  const std::string root =
      std::string(testing::TempDir()) + "/pipeline_segmented_faiss_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_segmented_faiss", 4,
                                                  "euclidean", "external",
                                                  "faiss"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_segmented_faiss",
      vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(service.rebuild_index("idx_segmented_faiss"));

  ASSERT_TRUE(service.set_faiss_ivf_params("idx_segmented_faiss", 4, 2));
  ASSERT_TRUE(
      service.set_faiss_ivf_pq_params("idx_segmented_faiss", 4, 2, 2, 4));
  EXPECT_FALSE(service.set_diskann_disk_pq_dims("idx_segmented_faiss", 4));
  EXPECT_FALSE(service.set_diskann_accelerate_build("idx_segmented_faiss",
                                                   true));
  EXPECT_FALSE(service.set_diskann_shuffle_build("idx_segmented_faiss", true));
  EXPECT_FALSE(service.set_diskann_use_bfs_cache("idx_segmented_faiss", true));

  vector_index::index_service::index_config described;
  ASSERT_TRUE(service.describe_index("idx_segmented_faiss", &described,
                                     nullptr, nullptr, nullptr));
  EXPECT_EQ(4U, described.faiss_nlist);
  EXPECT_EQ(2U, described.faiss_nprobe);
  EXPECT_EQ(2U, described.faiss_pq_m);
  EXPECT_EQ(4U, described.faiss_pq_bits);
}

TEST(VectorIndexServiceTest, BuildPipelineHonorsRowThreshold) {
  build_pipeline_options_guard guard;
  const std::string root = std::string(testing::TempDir()) + "/pipeline_rows_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);
  UlonglongGuard entry_cache_guard(&opt_vector_entry_cache_size, 0);
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kAuto);
  opt_vector_build_pipeline_min_rows = 1;
  opt_vector_build_pipeline_min_size = 0;

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_pipeline_rows", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_pipeline_rows", vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(service.direct_upsert("idx_pipeline_rows", 11, {1.0F, 0.0F}));
  ASSERT_TRUE(service.rebuild_index("idx_pipeline_rows"));

  vector_index::index_service::build_pipeline_snapshot snapshot;
  ASSERT_TRUE(service.describe_build_pipeline("idx_pipeline_rows", &snapshot));
  expect_pipeline_snapshot(snapshot, "auto", "segmented", "row_count", 1, 8,
                           0);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, BuildPipelineHonorsMultipleRawSegments) {
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  opt_vector_build_pipeline_max_tasks = 2;
  opt_vector_build_segment_max_rows = 1;
  vector_index::reset_runtime_worker_pool_for_testing();
  const std::string root =
      std::string(testing::TempDir()) + "/pipeline_raw_segments_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string first_vector_path = root + "/first.fbin";
  const std::string first_docid_path = root + "/first.u64";
  const std::string second_vector_path = root + "/second.fbin";
  const std::string second_docid_path = root + "/second.u64";
  write_raw_fbin_file(first_vector_path, 1, 2, {1.0F, 0.0F});
  write_raw_docid_file(first_docid_path, {11});
  write_raw_fbin_file(second_vector_path, 1, 2, {0.0F, 1.0F});
  write_raw_docid_file(second_docid_path, {22});

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_pipeline_raw", 2,
                                                  "euclidean", "memory",
                                                  provider));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_pipeline_raw", vector_index::index_consistency_mode::kStandalone));

  vector_index::index_service::bulk_load_options options;
  uint64_t loaded_rows = 0;
  std::string error;
  ASSERT_TRUE(service.bulk_upsert_from_raw_files(
      "idx_pipeline_raw", first_vector_path, first_docid_path, options,
      &loaded_rows, &error))
      << error;
  EXPECT_EQ(1U, loaded_rows);
  ASSERT_TRUE(service.bulk_upsert_from_raw_files(
      "idx_pipeline_raw", second_vector_path, second_docid_path, options,
      &loaded_rows, &error))
      << error;
  EXPECT_EQ(1U, loaded_rows);
  ASSERT_TRUE(service.rebuild_index("idx_pipeline_raw"));
  EXPECT_EQ(0U, vector_index::runtime_worker_pool_size_for_testing());

  vector_index::index_service::build_pipeline_snapshot snapshot;
  ASSERT_TRUE(service.describe_build_pipeline("idx_pipeline_raw", &snapshot));
  expect_pipeline_snapshot(snapshot, "auto", "segmented", "raw_segment_count",
                           2, 16, 2);

  vector_index::index_service::index_config described;
  vector_index::backend_build_diagnostics diagnostics;
  ASSERT_TRUE(service.describe_index(
      "idx_pipeline_raw", &described, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      &diagnostics));
  EXPECT_EQ("segmented", diagnostics.runtime);
  EXPECT_EQ("raw_segments", diagnostics.input_source);
  EXPECT_EQ(2U, diagnostics.row_count);
  EXPECT_EQ(2U, diagnostics.segment_count);
  EXPECT_EQ(2U, diagnostics.build_invocations);
  EXPECT_EQ(2U, diagnostics.concurrent_build_tasks);
  EXPECT_GE(diagnostics.effective_build_threads, 1U);
  const uint32_t cpu_budget =
      std::thread::hardware_concurrency() == 0
          ? 1U
          : std::thread::hardware_concurrency();
  EXPECT_LE(static_cast<uint64_t>(diagnostics.effective_build_threads) *
                diagnostics.concurrent_build_tasks,
            cpu_budget);
  EXPECT_GE(diagnostics.effective_blas_threads, 1U);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_pipeline_raw", {1.0F, 0.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(11U, result[0].doc_id);
  EXPECT_EQ(22U, result[1].doc_id);

  std::vector<std::vector<vector_index::search_result>> batch_result;
  ASSERT_TRUE(service.search_batch("idx_pipeline_raw",
                                   {{1.0F, 0.0F}, {0.0F, 1.0F}}, 1,
                                   &batch_result));
  ASSERT_EQ(2U, batch_result.size());
  ASSERT_EQ(1U, batch_result[0].size());
  ASSERT_EQ(1U, batch_result[1].size());
  EXPECT_EQ(11U, batch_result[0][0].doc_id);
  EXPECT_EQ(22U, batch_result[1][0].doc_id);

  vector_index::reset_runtime_worker_pool_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, BuildPipelineSchedulesFourRawSegmentTasks) {
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  opt_vector_build_pipeline_max_tasks = 4;
  opt_vector_build_segment_max_rows = 2;
  opt_vector_build_segment_target_size = 1024 * 1024;
  vector_index::reset_runtime_worker_pool_for_testing();

  const std::string root =
      std::string(testing::TempDir()) + "/pipeline_four_segments_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 8, 2,
                      {1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F, 4.0F, 0.0F,
                       5.0F, 0.0F, 6.0F, 0.0F, 7.0F, 0.0F, 8.0F, 0.0F});
  write_raw_docid_file(docid_path, {1, 2, 3, 4, 5, 6, 7, 8});

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_pipeline_four", 2,
                                                  "euclidean", "memory",
                                                  provider));
  if (std::string(provider) == "hnsw") {
    ASSERT_TRUE(service.set_hnsw_build_threads("idx_pipeline_four", 16));
  } else if (std::string(provider) == "faiss") {
    ASSERT_TRUE(service.set_faiss_build_threads("idx_pipeline_four", 16));
  } else if (std::string(provider) == "diskann") {
    ASSERT_TRUE(service.set_diskann_build_threads("idx_pipeline_four", 16));
  }
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_pipeline_four", vector_index::index_consistency_mode::kStandalone));

  vector_index::index_service::bulk_load_options options;
  uint64_t loaded_rows = 0;
  std::string error;
  ASSERT_TRUE(service.bulk_upsert_from_raw_files(
      "idx_pipeline_four", vector_path, docid_path, options, &loaded_rows,
      &error))
      << error;
  EXPECT_EQ(8U, loaded_rows);
  EXPECT_EQ(4U, service.standalone_raw_segment_count("idx_pipeline_four"));

  ASSERT_TRUE(service.rebuild_index("idx_pipeline_four"));

  vector_index::index_service::build_pipeline_snapshot snapshot;
  ASSERT_TRUE(service.describe_build_pipeline("idx_pipeline_four", &snapshot));
  expect_pipeline_snapshot(snapshot, "auto", "segmented", "raw_segment_count",
                           8, 64, 4);

  vector_index::index_service::index_config described;
  vector_index::backend_build_diagnostics diagnostics;
  ASSERT_TRUE(service.describe_index(
      "idx_pipeline_four", &described, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      &diagnostics));
  EXPECT_EQ("segmented", diagnostics.runtime);
  EXPECT_EQ("raw_segments", diagnostics.input_source);
  EXPECT_EQ(8U, diagnostics.row_count);
  EXPECT_EQ(4U, diagnostics.segment_count);
  EXPECT_EQ(4U, diagnostics.build_invocations);
  EXPECT_EQ(4U, diagnostics.concurrent_build_tasks);
  if (std::string(provider) != "native") {
    EXPECT_EQ(4U, diagnostics.effective_build_threads);
    EXPECT_EQ(1U, diagnostics.effective_blas_threads);
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_pipeline_four", {8.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(8U, result[0].doc_id);

  vector_index::reset_runtime_worker_pool_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     LevelledCompactionPublishesFewerSegmentsAndKeepsOldRuntimeAlive) {
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);
  opt_vector_build_segment_max_rows = 1;
  opt_vector_build_segment_target_size = 1024 * 1024;

  const std::string root =
      std::string(testing::TempDir()) + "/levelled_compaction_publish_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::index_service service;
  std::string setup_error;
  ASSERT_TRUE(prepare_levelled_compaction_index(
      &service, root, "idx_levelled_publish", provider, &setup_error))
      << setup_error;
  ASSERT_EQ(8U, service.standalone_raw_segment_count("idx_levelled_publish"));
  ASSERT_TRUE(service.rebuild_index("idx_levelled_publish"));

  vector_index::index_service::backend_ptr old_runtime;
  ASSERT_TRUE(service.snapshot_search_backend_loaded(
      "idx_levelled_publish", {8.0F, 0.0F}, &old_runtime));
  ASSERT_NE(nullptr, old_runtime);

  opt_vector_build_segment_max_rows = 4;
  ASSERT_TRUE(service.rebuild_index("idx_levelled_publish"));
  EXPECT_EQ(2U, service.standalone_raw_segment_count("idx_levelled_publish"));
  EXPECT_EQ("raw_segments_compacted",
            service.standalone_build_source("idx_levelled_publish"));

  std::vector<vector_index::search_result> old_result;
  {
    std::shared_lock<std::shared_mutex> runtime_guard(
        old_runtime->runtime_mutex());
    ASSERT_TRUE(old_runtime->search({8.0F, 0.0F}, 1, &old_result));
  }
  ASSERT_EQ(1U, old_result.size());
  EXPECT_EQ(8U, old_result[0].doc_id);

  std::vector<vector_index::search_result> current_result;
  ASSERT_TRUE(
      service.search("idx_levelled_publish", {8.0F, 0.0F}, 1, &current_result));
  ASSERT_EQ(1U, current_result.size());
  EXPECT_EQ(8U, current_result[0].doc_id);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     LevelledCompactionPublishFailureKeepsManifestAndRuntime) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);
  opt_vector_build_segment_max_rows = 1;
  opt_vector_build_segment_target_size = 1024 * 1024;

  const std::string root =
      std::string(testing::TempDir()) + "/levelled_compaction_rollback_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::index_service service;
  std::string setup_error;
  ASSERT_TRUE(prepare_levelled_compaction_index(
      &service, root, "idx_levelled_rollback", provider, &setup_error))
      << setup_error;
  ASSERT_TRUE(service.rebuild_index("idx_levelled_rollback"));

  vector_index::index_service::backend_ptr old_runtime;
  ASSERT_TRUE(service.snapshot_search_backend_loaded(
      "idx_levelled_rollback", {8.0F, 0.0F}, &old_runtime));
  ASSERT_NE(nullptr, old_runtime);

  opt_vector_build_segment_max_rows = 4;
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
                             "+d,vector_standalone_compaction_before_publish");
    EXPECT_FALSE(service.rebuild_index("idx_levelled_rollback"));
  }
  EXPECT_EQ(8U, service.standalone_raw_segment_count("idx_levelled_rollback"));
  EXPECT_EQ("raw_segments_direct",
            service.standalone_build_source("idx_levelled_rollback"));

  std::vector<vector_index::search_result> result;
  {
    std::shared_lock<std::shared_mutex> runtime_guard(
        old_runtime->runtime_mutex());
    ASSERT_TRUE(old_runtime->search({8.0F, 0.0F}, 1, &result));
  }
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(8U, result[0].doc_id);
  ASSERT_TRUE(
      service.search("idx_levelled_rollback", {8.0F, 0.0F}, 1, &result));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     LevelledCompactionBackendFailureKeepsManifestAndRuntime) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);
  opt_vector_build_segment_max_rows = 1;
  opt_vector_build_segment_target_size = 1024 * 1024;

  const std::string root =
      std::string(testing::TempDir()) + "/levelled_compaction_build_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::index_service service;
  std::string setup_error;
  ASSERT_TRUE(prepare_levelled_compaction_index(
      &service, root, "idx_levelled_build_fail", provider, &setup_error))
      << setup_error;
  ASSERT_TRUE(service.rebuild_index("idx_levelled_build_fail"));

  vector_index::index_service::backend_ptr old_runtime;
  ASSERT_TRUE(service.snapshot_search_backend_loaded(
      "idx_levelled_build_fail", {8.0F, 0.0F}, &old_runtime));
  ASSERT_NE(nullptr, old_runtime);

  opt_vector_build_segment_max_rows = 4;
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_segment_backend_build_failure");
    EXPECT_FALSE(service.rebuild_index("idx_levelled_build_fail"));
  }
  EXPECT_EQ(8U,
            service.standalone_raw_segment_count("idx_levelled_build_fail"));
  EXPECT_EQ("raw_segments_direct",
            service.standalone_build_source("idx_levelled_build_fail"));

  std::vector<vector_index::search_result> result;
  {
    std::shared_lock<std::shared_mutex> runtime_guard(
        old_runtime->runtime_mutex());
    ASSERT_TRUE(old_runtime->search({8.0F, 0.0F}, 1, &result));
  }
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(8U, result[0].doc_id);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     FailedStandaloneRebuildRetainsCandidateDiagnostics) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for rebuild tests";
  }
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kDirect);

  const std::string root =
      std::string(testing::TempDir()) + "/standalone_failed_diagnostics_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::index_service service;
  std::string error;
  ASSERT_TRUE(prepare_levelled_compaction_index(
      &service, root, "idx_failed_diagnostics", provider, &error))
      << error;

  vector_index::index_service::standalone_rebuild_plan plan;
  ASSERT_TRUE(service.prepare_standalone_rebuild(
      "idx_failed_diagnostics", &plan, &error))
      << error;
  bool build_ok = false;
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_standalone_rebuild_after_backend_build_failure");
    build_ok = service.build_standalone_rebuild(&plan, &error);
  }
  EXPECT_FALSE(build_ok);
  service.discard_standalone_rebuild(&plan);

  vector_index::index_service::index_config config;
  vector_index::backend_build_diagnostics diagnostics;
  ASSERT_TRUE(service.describe_index(
      "idx_failed_diagnostics", &config, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      &diagnostics));
  EXPECT_GT(diagnostics.build_invocations, 0U);
  EXPECT_EQ("debug_forced_post_build_failure", diagnostics.fallback_reason);

  ASSERT_TRUE(service.rename_index("idx_failed_diagnostics",
                                   "idx_renamed_diagnostics"));
  EXPECT_FALSE(service.describe_index(
      "idx_failed_diagnostics", &config, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      &diagnostics));
  ASSERT_TRUE(service.describe_index(
      "idx_renamed_diagnostics", &config, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      &diagnostics));
  EXPECT_EQ("debug_forced_post_build_failure", diagnostics.fallback_reason);

  ASSERT_TRUE(service.rebuild_index("idx_renamed_diagnostics", &error))
      << error;
  ASSERT_TRUE(service.describe_index(
      "idx_renamed_diagnostics", &config, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      &diagnostics));
  EXPECT_NE("debug_forced_post_build_failure", diagnostics.fallback_reason);
  ASSERT_TRUE(service.drop_index("idx_renamed_diagnostics"));
  EXPECT_FALSE(service.describe_index(
      "idx_renamed_diagnostics", &config, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      &diagnostics));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, LevelledCompactionManifestReloadsAfterRestart) {
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);
  opt_vector_build_segment_max_rows = 1;
  opt_vector_build_segment_target_size = 1024 * 1024;

  const std::string root =
      std::string(testing::TempDir()) + "/levelled_compaction_restart_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  {
    vector_index::index_service service;
    std::string setup_error;
    ASSERT_TRUE(prepare_levelled_compaction_index(
        &service, root, "idx_levelled_restart", provider, &setup_error))
        << setup_error;
    ASSERT_TRUE(service.rebuild_index("idx_levelled_restart"));
    opt_vector_build_segment_max_rows = 4;
    ASSERT_TRUE(service.rebuild_index("idx_levelled_restart"));
    ASSERT_EQ(2U, service.standalone_raw_segment_count("idx_levelled_restart"));
  }

  vector_index::index_service reloaded;
  ASSERT_TRUE(reloaded.register_index_from_strings(
      "idx_levelled_restart", 2, "euclidean", "memory", provider));
  ASSERT_TRUE(reloaded.set_index_consistency_mode(
      "idx_levelled_restart",
      vector_index::index_consistency_mode::kStandalone));
  EXPECT_EQ(2U, reloaded.standalone_raw_segment_count("idx_levelled_restart"));
  EXPECT_EQ("raw_segments_compacted",
            reloaded.standalone_build_source("idx_levelled_restart"));
  ASSERT_TRUE(reloaded.rebuild_index("idx_levelled_restart"));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      reloaded.search("idx_levelled_restart", {8.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(8U, result[0].doc_id);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     LevelledCompactionGenerationConflictRejectsStagedFiles) {
  const std::string root =
      std::string(testing::TempDir()) + "/levelled_compaction_conflict_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 8, 2,
                      {1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F, 4.0F, 0.0F, 5.0F,
                       0.0F, 6.0F, 0.0F, 7.0F, 0.0F, 8.0F, 0.0F});
  write_raw_docid_file(docid_path, {1, 2, 3, 4, 5, 6, 7, 8});

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_levelled_conflict", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_levelled_conflict", vector_path,
                                          docid_path, 8, 2, 1));

  vector_index::standalone_entry_store::raw_segment_compaction compaction;
  ASSERT_TRUE(
      store.stage_levelled_compaction("idx_levelled_conflict", 1, &compaction));
  EXPECT_FALSE(compaction.needed);
  EXPECT_EQ(8U, compaction.segments.size());
  EXPECT_TRUE(compaction.created_files.empty());
  ASSERT_TRUE(
      store.stage_levelled_compaction("idx_levelled_conflict", 4, &compaction));
  ASSERT_TRUE(compaction.needed);
  ASSERT_EQ(2U, compaction.segments.size());
  const std::vector<std::string> staged_files = compaction.created_files;

  ASSERT_TRUE(store.upsert("idx_levelled_conflict", 9, {9.0F, 0.0F},
                           std::numeric_limits<size_t>::max()));
  EXPECT_FALSE(
      store.publish_levelled_compaction("idx_levelled_conflict", &compaction));
  store.discard_levelled_compaction(&compaction);

  EXPECT_EQ(8U, store.raw_segment_count("idx_levelled_conflict"));
  EXPECT_EQ(9U, store.entry_count("idx_levelled_conflict"));
  for (const std::string &path : staged_files) {
    EXPECT_FALSE(std::filesystem::exists(path));
  }

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     LevelledCompactionDefersCleanupUntilFinalizeAndSupportsRollback) {
  const std::string root =
      std::string(testing::TempDir()) + "/levelled_compaction_finalize_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 8, 2,
                      {1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F, 4.0F, 0.0F,
                       5.0F, 0.0F, 6.0F, 0.0F, 7.0F, 0.0F, 8.0F, 0.0F});
  write_raw_docid_file(docid_path, {1, 2, 3, 4, 5, 6, 7, 8});

  vector_index::standalone_entry_store store;
  const std::string index_name = "idx_levelled_finalize";
  ASSERT_TRUE(store.register_index(index_name, 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files(index_name, vector_path, docid_path,
                                          8, 2, 1));

  vector_index::standalone_entry_store::raw_segment_compaction compaction;
  ASSERT_TRUE(store.stage_levelled_compaction(index_name, 4, &compaction));
  ASSERT_TRUE(compaction.needed);
  const std::vector<std::string> source_files = compaction.replaced_files;
  const std::vector<std::string> first_staged_files = compaction.created_files;
  ASSERT_TRUE(store.publish_levelled_compaction(index_name, &compaction));
  EXPECT_EQ(2U, store.raw_segment_count(index_name));
  for (const std::string &path : source_files) {
    EXPECT_TRUE(std::filesystem::exists(path));
  }

  ASSERT_TRUE(store.rollback_levelled_compaction(index_name, &compaction));
  EXPECT_EQ(8U, store.raw_segment_count(index_name));
  for (const std::string &path : source_files) {
    EXPECT_TRUE(std::filesystem::exists(path));
  }
  for (const std::string &path : first_staged_files) {
    EXPECT_FALSE(std::filesystem::exists(path));
  }

  ASSERT_TRUE(store.stage_levelled_compaction(index_name, 4, &compaction));
  ASSERT_TRUE(compaction.needed);
  const std::vector<std::string> final_source_files = compaction.replaced_files;
  const std::vector<std::string> final_staged_files = compaction.created_files;
  ASSERT_TRUE(store.publish_levelled_compaction(index_name, &compaction));
  store.finalize_levelled_compaction(&compaction);
  EXPECT_EQ(2U, store.raw_segment_count(index_name));
  for (const std::string &path : final_source_files) {
    EXPECT_FALSE(std::filesystem::exists(path));
  }
  for (const std::string &path : final_staged_files) {
    EXPECT_TRUE(std::filesystem::exists(path));
  }

  vector_index::vector_data vector;
  bool found = false;
  ASSERT_TRUE(store.find_entry(index_name, 8, &vector, &found));
  ASSERT_TRUE(found);
  EXPECT_EQ((vector_index::vector_data{8.0F, 0.0F}), vector);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, SegmentedSearchHandlesNestedParallelBackend) {
  shared_search_worker_pool_guard search_pool_guard;
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  UlongGuard hnsw_search_threads_guard(&opt_vector_hnsw_search_threads, 2);
  UlongGuard batch_search_threads_guard(&opt_vector_batch_search_threads, 2);
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);
  opt_vector_build_pipeline_max_tasks = 2;
  opt_vector_build_segment_max_rows = 1;
  vector_index::reset_runtime_worker_pool_for_testing();

  const std::string root =
      std::string(testing::TempDir()) + "/pipeline_raw_parallel_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string first_vector_path = root + "/first.fbin";
  const std::string first_docid_path = root + "/first.u64";
  const std::string second_vector_path = root + "/second.fbin";
  const std::string second_docid_path = root + "/second.u64";
  write_raw_fbin_file(first_vector_path, 1, 2, {1.0F, 0.0F});
  write_raw_docid_file(first_docid_path, {11});
  write_raw_fbin_file(second_vector_path, 1, 2, {0.0F, 1.0F});
  write_raw_docid_file(second_docid_path, {22});

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_pipeline_raw_parallel",
                                                  2, "euclidean", "memory",
                                                  provider));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_pipeline_raw_parallel",
      vector_index::index_consistency_mode::kStandalone));

  vector_index::index_service::bulk_load_options options;
  uint64_t loaded_rows = 0;
  std::string error;
  ASSERT_TRUE(service.bulk_upsert_from_raw_files(
      "idx_pipeline_raw_parallel", first_vector_path, first_docid_path,
      options, &loaded_rows, &error))
      << error;
  EXPECT_EQ(1U, loaded_rows);
  ASSERT_TRUE(service.bulk_upsert_from_raw_files(
      "idx_pipeline_raw_parallel", second_vector_path, second_docid_path,
      options, &loaded_rows, &error))
      << error;
  EXPECT_EQ(1U, loaded_rows);

  ASSERT_TRUE(service.rebuild_index("idx_pipeline_raw_parallel"));
  EXPECT_EQ(0U, vector_index::runtime_worker_pool_size_for_testing());

  std::vector<std::vector<vector_index::search_result>> batch_result;
  ASSERT_TRUE(service.search_batch("idx_pipeline_raw_parallel",
                                   {{1.0F, 0.0F}, {0.0F, 1.0F}}, 1,
                                   &batch_result));
  ASSERT_EQ(2U, batch_result.size());
  ASSERT_EQ(1U, batch_result[0].size());
  ASSERT_EQ(1U, batch_result[1].size());
  EXPECT_EQ(11U, batch_result[0][0].doc_id);
  EXPECT_EQ(22U, batch_result[1][0].doc_id);
  EXPECT_EQ(0U, vector_index::runtime_worker_pool_size_for_testing());
  if (std::string(provider) == "hnsw") {
    EXPECT_EQ(2U,
              vector_index::shared_search_worker_pool_size_for_testing());
  } else {
    EXPECT_EQ(0U,
              vector_index::shared_search_worker_pool_size_for_testing());
  }

  vector_index::reset_runtime_worker_pool_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, DiskAnnRawSegmentsBuildSingleBackend) {
  install_diskann_offline_test_adapter();
  if (!vector_index::diskann_offline_api_manifest_build_load_for_testing()) {
    GTEST_SKIP() << "DiskANN offline manifest adapter unavailable in current "
                    "build/runtime";
  }
  build_pipeline_options_guard guard;
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);
  opt_vector_build_pipeline_max_tasks = 2;
  opt_vector_build_segment_max_rows = 4;
  opt_vector_diskann_segmented_serving = false;
  vector_index::reset_runtime_worker_pool_for_testing();
  const std::string root =
      std::string(testing::TempDir()) + "/pipeline_diskann_manifest_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string first_vector_path = root + "/first.fbin";
  const std::string first_docid_path = root + "/first.u64";
  const std::string second_vector_path = root + "/second.fbin";
  const std::string second_docid_path = root + "/second.u64";
  write_raw_fbin_file(first_vector_path, 4, 2,
                      {1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F, 4.0F, 4.0F});
  write_raw_docid_file(first_docid_path, {10, 20, 30, 40});
  write_raw_fbin_file(second_vector_path, 4, 2,
                      {9.0F, 9.0F, 8.0F, 8.0F, 7.0F, 7.0F, 6.0F, 6.0F});
  write_raw_docid_file(second_docid_path, {90, 80, 70, 60});

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_diskann_manifest", 2,
                                                  "euclidean", "external",
                                                  "diskann"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_diskann_manifest", vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(service.set_diskann_build_mode(
      "idx_diskann_manifest", vector_index::diskann_build_mode::kOffline));
  ASSERT_TRUE(service.set_diskann_build_params("idx_diskann_manifest", 4, 16,
                                               1));

  vector_index::index_service::bulk_load_options options;
  uint64_t loaded_rows = 0;
  std::string error;
  ASSERT_TRUE(service.bulk_upsert_from_raw_files(
      "idx_diskann_manifest", first_vector_path, first_docid_path, options,
      &loaded_rows, &error))
      << error;
  EXPECT_EQ(4U, loaded_rows);
  ASSERT_TRUE(service.bulk_upsert_from_raw_files(
      "idx_diskann_manifest", second_vector_path, second_docid_path, options,
      &loaded_rows, &error))
      << error;
  EXPECT_EQ(4U, loaded_rows);

  ASSERT_TRUE(service.rebuild_index("idx_diskann_manifest"));

  vector_index::index_service::index_config described;
  vector_index::backend_build_diagnostics diagnostics;
  ASSERT_TRUE(service.describe_index(
      "idx_diskann_manifest", &described, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      &diagnostics));
  EXPECT_EQ(vector_index::backend_provider::kDiskAnn, described.provider);
  EXPECT_TRUE(described.backend_variant == "diskann_offline" ||
              described.backend_variant == "diskann_vendored_offline")
      << described.backend_variant;
  EXPECT_TRUE(diagnostics.runtime == "official_cpp_main" ||
              diagnostics.runtime == "vendored_runtime")
      << diagnostics.runtime;
  EXPECT_EQ("raw_segments", diagnostics.input_source);
  EXPECT_EQ(8U, diagnostics.row_count);
  EXPECT_EQ(2U, diagnostics.segment_count);
  EXPECT_EQ(1U, diagnostics.build_invocations);

  std::vector<vector_index_metadata_store::segment_task_row> rows;
  ASSERT_TRUE(service.snapshot_segment_tasks("idx_diskann_manifest", &rows));
  ASSERT_EQ(2U, rows.size());
  for (const auto &row : rows) {
    EXPECT_EQ(vector_index_metadata_store::segment_task_state::kReady,
              row.state);
    EXPECT_EQ(1U, row.attempt);
    EXPECT_EQ(0U, row.last_error_code);
    EXPECT_GT(row.row_count, 0U);
    EXPECT_GT(row.payload_size, 0U);
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_diskann_manifest", {1.0F, 1.0F}, 4,
                             &result));
  ASSERT_GE(result.size(), 2U);
  EXPECT_EQ(10U, result[0].doc_id);

  std::vector<std::vector<vector_index::search_result>> batch_result;
  ASSERT_TRUE(service.search_batch(
      "idx_diskann_manifest", {{1.0F, 1.0F}, {9.0F, 9.0F}}, 2,
      &batch_result));
  ASSERT_EQ(2U, batch_result.size());
  ASSERT_GE(batch_result[0].size(), 1U);
  ASSERT_GE(batch_result[1].size(), 1U);
  EXPECT_EQ(10U, batch_result[0][0].doc_id);
  EXPECT_EQ(90U, batch_result[1][0].doc_id);

  vector_index::reset_runtime_worker_pool_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     TransactionalDiskAnnOfflineRebuildUsesTruthStoreRawSegments) {
  install_diskann_offline_test_adapter();
  if (!vector_index::diskann_offline_api_manifest_build_load_for_testing()) {
    GTEST_SKIP() << "DiskANN offline manifest adapter unavailable in current "
                    "build/runtime";
  }
  build_pipeline_options_guard guard;
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);
  opt_vector_build_pipeline_max_tasks = 2;
  opt_vector_build_segment_max_rows = 4;
  opt_vector_build_segment_target_size = 1024 * 1024;

  const std::string root =
      std::string(testing::TempDir()) + "/pipeline_diskann_txn_manifest_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_diskann_txn_offline", 2,
                                                  "euclidean", "external",
                                                  "diskann"));
  for (uint64_t doc_id = 1; doc_id <= 8; ++doc_id) {
    ASSERT_TRUE(service.stage_upsert(8801, "idx_diskann_txn_offline", doc_id,
                                     {static_cast<float>(doc_id), 0.0F}));
  }
  ASSERT_TRUE(service.commit(8801));
  ASSERT_TRUE(service.set_diskann_build_mode(
      "idx_diskann_txn_offline", vector_index::diskann_build_mode::kOffline));
  ASSERT_TRUE(service.rebuild_index("idx_diskann_txn_offline"));

  vector_index::index_service::index_config config;
  vector_index::backend_build_diagnostics diagnostics;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  ASSERT_TRUE(service.describe_index(
      "idx_diskann_txn_offline", &config, nullptr, &entry_count,
      &committed_entry_count, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, &diagnostics));
  EXPECT_EQ(vector_index::index_consistency_mode::kTransactional,
            config.consistency_mode);
  EXPECT_EQ(vector_index::diskann_build_mode::kOffline,
            config.diskann_build_mode_value);
  EXPECT_EQ("truth_store_snapshot", diagnostics.input_source);
  EXPECT_EQ(2U, diagnostics.segment_count);
  EXPECT_EQ(2U, diagnostics.build_invocations);
  EXPECT_GE(diagnostics.concurrent_build_tasks, 1U);
  EXPECT_EQ(8U, entry_count);
  EXPECT_EQ(8U, committed_entry_count);

  std::vector<vector_index_metadata_store::segment_task_row> task_rows;
  ASSERT_TRUE(
      service.snapshot_segment_tasks("idx_diskann_txn_offline", &task_rows));
  ASSERT_FALSE(task_rows.empty());
  for (const auto &row : task_rows) {
    EXPECT_EQ(vector_index_metadata_store::segment_task_state::kReady,
              row.state);
    EXPECT_TRUE(row.vector_path.empty());
    EXPECT_TRUE(row.docid_path.empty());
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(
      service.search("idx_diskann_txn_offline", {1.0F, 0.0F}, 8, &result));
  ASSERT_EQ(8U, result.size());

  vector_index::reset_runtime_worker_pool_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, SegmentedRebuildRecordsReadySegmentTasks) {
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);

  const std::string root =
      std::string(testing::TempDir()) + "/segment_tasks_ready_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_segment_tasks", 2,
                                                  "euclidean", "memory",
                                                  provider));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_segment_tasks", vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(service.direct_upsert("idx_segment_tasks", 11, {1.0F, 0.0F}));
  ASSERT_TRUE(service.direct_upsert("idx_segment_tasks", 22, {0.0F, 1.0F}));

  ASSERT_TRUE(service.rebuild_index("idx_segment_tasks"));

  std::vector<vector_index_metadata_store::segment_task_row> rows;
  ASSERT_TRUE(service.snapshot_segment_tasks("idx_segment_tasks", &rows));
  ASSERT_EQ(1U, rows.size());
  EXPECT_EQ("idx_segment_tasks", rows[0].index_name);
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

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, SegmentedRebuildKeepsTasksOnPrePublishFailure) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug sync failure injection requires a debug build";
#endif
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);

  const std::string root =
      std::string(testing::TempDir()) + "/segment_tasks_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_segment_fail", 2,
                                                  "euclidean", "memory",
                                                  provider));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_segment_fail", vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(service.direct_upsert("idx_segment_fail", 33, {3.0F, 0.0F}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_segment_task_before_publish");
    EXPECT_FALSE(service.rebuild_index("idx_segment_fail"));
  }

  vector_index::index_service::index_config config;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  std::string lifecycle_state;
  ASSERT_TRUE(service.describe_index(
      "idx_segment_fail", &config, nullptr, &entry_count,
      &committed_entry_count, &lifecycle_state));
  EXPECT_EQ("bulk_loading", lifecycle_state);
  EXPECT_EQ(0U, entry_count);
  EXPECT_EQ(1U, committed_entry_count);

  std::vector<vector_index::search_result> result;
  EXPECT_FALSE(
      service.search("idx_segment_fail", {3.0F, 0.0F}, 1, &result));

  std::vector<vector_index_metadata_store::segment_task_row> rows;
  ASSERT_TRUE(service.snapshot_segment_tasks("idx_segment_fail", &rows));
  ASSERT_EQ(1U, rows.size());
  EXPECT_EQ(vector_index_metadata_store::segment_task_state::kReady,
            rows[0].state);
  EXPECT_EQ(1U, service.standalone_raw_segment_count("idx_segment_fail"));
  EXPECT_EQ(1U, rows[0].row_count);

  ASSERT_TRUE(service.rebuild_index("idx_segment_fail"));
  ASSERT_TRUE(service.search("idx_segment_fail", {3.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(33U, result[0].doc_id);

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_segment_task_before_publish");
    EXPECT_FALSE(service.rebuild_index("idx_segment_fail"));
  }
  ASSERT_TRUE(service.describe_index(
      "idx_segment_fail", &config, nullptr, &entry_count,
      &committed_entry_count, &lifecycle_state));
  EXPECT_EQ("ready", lifecycle_state);
  ASSERT_TRUE(service.search("idx_segment_fail", {3.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(33U, result[0].doc_id);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest, RebuildAllIndexesHonorsSegmentedStandaloneSource) {
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_all_first", 2,
                                                  "euclidean", "memory",
                                                  provider));
  ASSERT_TRUE(service.register_index_from_strings("idx_all_second", 2,
                                                  "euclidean", "memory",
                                                  provider));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_all_first", vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_all_second", vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(service.direct_upsert("idx_all_first", 101, {1.0F, 0.0F}));
  ASSERT_TRUE(service.direct_upsert("idx_all_second", 202, {0.0F, 1.0F}));

  size_t rebuilt_count = 0;
  ASSERT_TRUE(service.rebuild_all_indexes(&rebuilt_count));
  EXPECT_EQ(2U, rebuilt_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_all_first", {1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(101U, result[0].doc_id);
  ASSERT_TRUE(service.search("idx_all_second", {0.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(202U, result[0].doc_id);
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
  faiss_snapshot_root_guard root_guard(root);

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

TEST(VectorIndexServiceTest, StandaloneRawFileBulkLoadSplitsBySegmentRowLimit) {
  build_pipeline_options_guard guard;
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
  opt_vector_build_segment_max_rows = 1;
  opt_vector_build_segment_target_size = 1024 * 1024;

  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_split_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 3, 2,
                      {1.0F, 0.0F, 0.0F, 1.0F, 2.0F, 0.0F});
  write_raw_docid_file(docid_path, {11, 22, 33});

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings("idx_raw_split", 2,
                                                  "euclidean", "memory",
                                                  provider));
  ASSERT_TRUE(service.set_index_consistency_mode(
      "idx_raw_split", vector_index::index_consistency_mode::kStandalone));

  vector_index::index_service::bulk_load_options options;
  uint64_t loaded_rows = 0;
  std::string error;
  ASSERT_TRUE(service.bulk_upsert_from_raw_files(
      "idx_raw_split", vector_path, docid_path, options, &loaded_rows, &error))
      << error;
  EXPECT_EQ(3U, loaded_rows);
  EXPECT_EQ(3U, service.standalone_raw_segment_count("idx_raw_split"));

  ASSERT_TRUE(service.rebuild_index("idx_raw_split"));
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_raw_split", {2.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(33U, result[0].doc_id);

  vector_index::index_service::build_pipeline_snapshot snapshot;
  ASSERT_TRUE(service.describe_build_pipeline("idx_raw_split", &snapshot));
  expect_pipeline_snapshot(snapshot, "auto", "segmented", "raw_segment_count",
                           3, 24, 3);

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
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
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
                                                  provider));
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
                                                    provider));
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
                                          vector_path, "", 0, 2, 2));
  EXPECT_FALSE(store.bulk_upsert_raw_files("missing", vector_path, "", 2, 2,
                                           2));
  EXPECT_FALSE(store.bulk_upsert_raw_files("idx_raw_generated_docids",
                                           vector_path, "", 2, 3, 2));
  EXPECT_FALSE(store.bulk_upsert_raw_files("idx_raw_generated_docids",
                                           vector_path, "", 3, 2, 3));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_generated_docids",
                                          vector_path, "", 2, 2, 2));
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

TEST(VectorStandaloneEntryStoreTest, RawSegmentCanReuseValidatedDocIdSet) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_reuse_docids_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {3.0F, 0.0F, 1.0F, 0.0F});
  write_raw_docid_file(docid_path, {30, 10});

  std::unordered_set<uint64_t> validated_doc_ids{30, 10};
  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_raw_reuse", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_reuse", vector_path,
                                          docid_path, 2, 2, 2,
                                          &validated_doc_ids));
  EXPECT_TRUE(validated_doc_ids.empty());
  EXPECT_EQ(2U, store.entry_count("idx_raw_reuse"));

  RawSegmentCountingBackend backend;
  ASSERT_TRUE(store.rebuild_backend_input("idx_raw_reuse", &backend));
  EXPECT_EQ(1U, backend.raw_segment_reader_calls);
  EXPECT_EQ(0U, backend.committed_reader_calls);

  const auto entries = collect_entries(&store, "idx_raw_reuse");
  ASSERT_EQ(2U, entries.size());
  EXPECT_EQ((vector_index::vector_data{3.0F, 0.0F}), entries.at(30));
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), entries.at(10));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     RawBulkManifestFailureRestoresPublishedState) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_publish_rollback_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string first_vector_path = root + "/first.fbin";
  const std::string first_docid_path = root + "/first.u64";
  const std::string second_vector_path = root + "/second.fbin";
  const std::string second_docid_path = root + "/second.u64";
  write_raw_fbin_file(first_vector_path, 2, 2, {1.0F, 0.0F, 2.0F, 0.0F});
  write_raw_docid_file(first_docid_path, {10, 20});
  write_raw_fbin_file(second_vector_path, 2, 2, {20.0F, 0.0F, 3.0F, 0.0F});
  write_raw_docid_file(second_docid_path, {20, 30});

  const auto count_files_with_extension = [&root](
                                              const std::string &extension) {
    std::error_code iterator_error;
    size_t count = 0;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(
             root, std::filesystem::directory_options::skip_permission_denied,
             iterator_error)) {
      if (iterator_error) return size_t{0};
      if (entry.is_regular_file(iterator_error) && !iterator_error &&
          entry.path().extension().string() == extension) {
        ++count;
      }
    }
    return count;
  };

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_raw_publish_rollback", 2));
  std::unordered_set<uint64_t> first_doc_ids{10, 20};
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_publish_rollback",
                                          first_vector_path, first_docid_path,
                                          2, 2, 2, &first_doc_ids));
  ASSERT_TRUE(first_doc_ids.empty());

  const std::string manifest = find_manifest_file(root, "manifest.v1");
  ASSERT_FALSE(manifest.empty());
  std::ifstream manifest_before_file(manifest);
  ASSERT_TRUE(manifest_before_file.is_open());
  const std::string manifest_before(
      (std::istreambuf_iterator<char>(manifest_before_file)),
      std::istreambuf_iterator<char>());
  ASSERT_FALSE(manifest_before.empty());
  const size_t fbin_files_before = count_files_with_extension(".fbin");
  const size_t docid_files_before = count_files_with_extension(".u64");

  std::unordered_set<uint64_t> failed_doc_ids{20, 30};
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
                             "+d,vector_standalone_bulk_fail_manifest_save");
    EXPECT_FALSE(store.bulk_upsert_raw_files(
        "idx_raw_publish_rollback", second_vector_path, second_docid_path, 2, 2,
        2, &failed_doc_ids));
  }
  EXPECT_TRUE(failed_doc_ids.empty());
  EXPECT_EQ(2U, store.entry_count("idx_raw_publish_rollback"));
  EXPECT_EQ(1U, store.raw_segment_count("idx_raw_publish_rollback"));
  EXPECT_EQ(1U, store.raw_locator_run_count("idx_raw_publish_rollback"));
  EXPECT_EQ("raw_segments_direct",
            store.build_source("idx_raw_publish_rollback"));
  EXPECT_EQ(fbin_files_before, count_files_with_extension(".fbin"));
  EXPECT_EQ(docid_files_before, count_files_with_extension(".u64"));

  std::ifstream manifest_after_file(manifest);
  ASSERT_TRUE(manifest_after_file.is_open());
  const std::string manifest_after(
      (std::istreambuf_iterator<char>(manifest_after_file)),
      std::istreambuf_iterator<char>());
  EXPECT_EQ(manifest_before, manifest_after);

  const auto entries_after_failure =
      collect_entries(&store, "idx_raw_publish_rollback");
  ASSERT_EQ(2U, entries_after_failure.size());
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}),
            entries_after_failure.at(10));
  EXPECT_EQ((vector_index::vector_data{2.0F, 0.0F}),
            entries_after_failure.at(20));
  EXPECT_EQ(entries_after_failure.end(), entries_after_failure.find(30));

  std::unordered_set<uint64_t> retry_doc_ids{20, 30};
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_publish_rollback",
                                          second_vector_path, second_docid_path,
                                          2, 2, 2, &retry_doc_ids));
  EXPECT_TRUE(retry_doc_ids.empty());
  EXPECT_EQ(3U, store.entry_count("idx_raw_publish_rollback"));
  EXPECT_EQ(2U, store.raw_segment_count("idx_raw_publish_rollback"));
  EXPECT_EQ(2U, store.raw_locator_run_count("idx_raw_publish_rollback"));

  const auto entries_after_retry =
      collect_entries(&store, "idx_raw_publish_rollback");
  ASSERT_EQ(3U, entries_after_retry.size());
  EXPECT_EQ((vector_index::vector_data{20.0F, 0.0F}),
            entries_after_retry.at(20));
  EXPECT_EQ((vector_index::vector_data{3.0F, 0.0F}),
            entries_after_retry.at(30));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     RawSegmentSplitReadsDocIdsWhenNoValidatedSetIsProvided) {
  build_pipeline_options_guard guard;
  opt_vector_build_segment_max_rows = 1;
  opt_vector_build_segment_target_size = 1024 * 1024;

  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_split_docids_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 3, 2,
                      {3.0F, 0.0F, 1.0F, 0.0F, 2.0F, 0.0F});
  write_raw_docid_file(docid_path, {30, 10, 20});

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_raw_split_docids", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_split_docids", vector_path,
                                          docid_path, 3, 2, 1));
  EXPECT_EQ(3U, store.entry_count("idx_raw_split_docids"));
  EXPECT_EQ(3U, store.raw_segment_count("idx_raw_split_docids"));

  const auto entries = collect_entries(&store, "idx_raw_split_docids");
  ASSERT_EQ(3U, entries.size());
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), entries.at(10));
  EXPECT_EQ((vector_index::vector_data{2.0F, 0.0F}), entries.at(20));
  EXPECT_EQ((vector_index::vector_data{3.0F, 0.0F}), entries.at(30));

  RawSegmentCountingBackend backend;
  ASSERT_TRUE(store.rebuild_backend_input("idx_raw_split_docids", &backend));
  EXPECT_EQ(1U, backend.raw_segment_reader_calls);
  EXPECT_EQ((std::vector<size_t>{1, 1, 1}), backend.raw_segment_doc_counts);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     RawSegmentSplitRejectsDocIdMismatchAndRollsBack) {
  build_pipeline_options_guard guard;
  opt_vector_build_segment_max_rows = 1;
  opt_vector_build_segment_target_size = 1024 * 1024;

  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_split_rollback_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string short_docid_path = root + "/short.u64";
  const std::string valid_docid_path = root + "/valid.u64";
  write_raw_fbin_file(vector_path, 3, 2,
                      {3.0F, 0.0F, 1.0F, 0.0F, 2.0F, 0.0F});
  write_raw_docid_file(short_docid_path, {30, 10});
  write_raw_docid_file(valid_docid_path, {30, 10, 20});

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_raw_split_rollback", 2));
  EXPECT_FALSE(store.bulk_upsert_raw_files("idx_raw_split_rollback",
                                           vector_path, short_docid_path, 3, 2,
                                           1));
  EXPECT_EQ(0U, store.entry_count("idx_raw_split_rollback"));
  EXPECT_EQ(0U, store.raw_segment_count("idx_raw_split_rollback"));
  EXPECT_EQ(0U, store.raw_segment_bytes("idx_raw_split_rollback"));

  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_split_rollback", vector_path,
                                          valid_docid_path, 3, 2, 1));
  EXPECT_EQ(3U, store.entry_count("idx_raw_split_rollback"));
  EXPECT_EQ(3U, store.raw_segment_count("idx_raw_split_rollback"));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     RebuildRejectsMaterializedFallbackOverCacheBudget) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_rebuild_budget_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {3.0F, 0.0F, 1.0F, 0.0F});
  write_raw_docid_file(docid_path, {30, 10});

  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, sizeof(float));
  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_rebuild_budget", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_rebuild_budget", vector_path,
                                          docid_path, 2, 2, 2));
  ASSERT_TRUE(store.upsert("idx_rebuild_budget", 20, {2.0F, 0.0F},
                           static_cast<size_t>(opt_vector_entry_cache_size)));

  RawSegmentCountingBackend backend;
  EXPECT_FALSE(store.rebuild_backend_input("idx_rebuild_budget", &backend));
  EXPECT_EQ(0U, backend.raw_segment_reader_calls);
  EXPECT_EQ(0U, backend.committed_reader_calls);
  EXPECT_EQ("raw_segments_compacted",
            store.build_source("idx_rebuild_budget"));

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

TEST(VectorStandaloneEntryStoreTest, FindAndIterationValidatePublicApiEdges) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_find_edges_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_find_edges", 2));

  vector_index::vector_data vector;
  bool found = true;
  EXPECT_FALSE(store.find_entry("idx_find_edges", 1, nullptr, &found));
  EXPECT_FALSE(store.find_entry("idx_find_edges", 1, &vector, nullptr));
  EXPECT_FALSE(store.find_entry("missing", 1, &vector, &found));
  ASSERT_TRUE(store.find_entry("idx_find_edges", 1, &vector, &found));
  EXPECT_FALSE(found);
  EXPECT_TRUE(vector.empty());

  ASSERT_TRUE(store.upsert("idx_find_edges", 7, {7.0F, 1.0F},
                           std::numeric_limits<size_t>::max()));
  ASSERT_TRUE(store.find_entry("idx_find_edges", 7, &vector, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ((vector_index::vector_data{7.0F, 1.0F}), vector);

  EXPECT_FALSE(store.for_each_entry(
      "idx_find_edges", vector_index::standalone_entry_store::entry_visitor{}));
  EXPECT_FALSE(store.for_each_entry(
      "missing", [](uint64_t, const vector_index::vector_data &) {
        return true;
      }));

  std::vector<uint64_t> visited_doc_ids;
  ASSERT_TRUE(store.for_each_entry(
      "idx_find_edges",
      [&visited_doc_ids](uint64_t doc_id, const vector_index::vector_data &entry) {
        visited_doc_ids.push_back(doc_id);
        EXPECT_EQ((vector_index::vector_data{7.0F, 1.0F}), entry);
        return true;
      }));
  EXPECT_EQ((std::vector<uint64_t>{7}), visited_doc_ids);

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
                                          docid_path, 2, 2, 2));
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

TEST(VectorStandaloneEntryStoreTest, FindEntriesReadsRawAndDeltaCandidates) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_find_entries_t";
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
  ASSERT_TRUE(store.register_index("idx_find_entries", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_find_entries", vector_path,
                                          docid_path, 2, 2, 2));
  EXPECT_EQ(2U, store.raw_locator_entry_count("idx_find_entries"));
  EXPECT_EQ(2U * 16U, store.raw_locator_bytes("idx_find_entries"));
  ASSERT_TRUE(store.upsert("idx_find_entries", 20, {2.0F, 0.0F}, 0));
  ASSERT_TRUE(store.erase("idx_find_entries", 30, 0));

  vector_index::committed_entries vectors;
  ASSERT_TRUE(store.find_entries("idx_find_entries", {10, 20, 30, 99},
                                 &vectors));
  ASSERT_EQ(2U, vectors.size());
  ASSERT_NE(vectors.end(), vectors.find(10));
  ASSERT_NE(vectors.end(), vectors.find(20));
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), vectors.at(10));
  EXPECT_EQ((vector_index::vector_data{2.0F, 0.0F}), vectors.at(20));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     FindEntriesUsesLatestRawLocationAndRestoresLocator) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_locator_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string first_vector_path = root + "/first.fbin";
  const std::string first_docid_path = root + "/first.u64";
  write_raw_fbin_file(first_vector_path, 2, 2, {3.0F, 0.0F, 1.0F, 0.0F});
  write_raw_docid_file(first_docid_path, {30, 10});

  const std::string second_vector_path = root + "/second.fbin";
  const std::string second_docid_path = root + "/second.u64";
  write_raw_fbin_file(second_vector_path, 2, 2, {9.0F, 0.0F, 4.0F, 0.0F});
  write_raw_docid_file(second_docid_path, {10, 40});

  const std::string index_name = "idx_raw_locator";
  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index(index_name, 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files(index_name, first_vector_path,
                                          first_docid_path, 2, 2, 2));
  ASSERT_TRUE(store.upsert(index_name, 10, {5.0F, 0.0F}, 0));
  ASSERT_TRUE(store.bulk_upsert_raw_files(index_name, second_vector_path,
                                          second_docid_path, 2, 2, 2));
  EXPECT_EQ(4U, store.raw_locator_entry_count(index_name));

  vector_index::committed_entries vectors;
  ASSERT_TRUE(store.find_entries(index_name, {10, 30, 40}, &vectors));
  ASSERT_EQ(3U, vectors.size());
  EXPECT_EQ((vector_index::vector_data{9.0F, 0.0F}), vectors.at(10));
  EXPECT_EQ((vector_index::vector_data{3.0F, 0.0F}), vectors.at(30));
  EXPECT_EQ((vector_index::vector_data{4.0F, 0.0F}), vectors.at(40));

  ASSERT_TRUE(store.upsert(index_name, 10, {11.0F, 0.0F}, 0));
  ASSERT_TRUE(store.find_entries(index_name, {10}, &vectors));
  ASSERT_EQ(1U, vectors.size());
  EXPECT_EQ((vector_index::vector_data{11.0F, 0.0F}), vectors.at(10));

  vector_index::standalone_entry_store recovered;
  ASSERT_TRUE(recovered.register_index(index_name, 2));
  EXPECT_EQ(4U, recovered.raw_locator_entry_count(index_name));
  ASSERT_TRUE(recovered.find_entries(index_name, {10, 30, 40}, &vectors));
  ASSERT_EQ(3U, vectors.size());
  EXPECT_EQ((vector_index::vector_data{11.0F, 0.0F}), vectors.at(10));
  EXPECT_EQ((vector_index::vector_data{3.0F, 0.0F}), vectors.at(30));
  EXPECT_EQ((vector_index::vector_data{4.0F, 0.0F}), vectors.at(40));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     BulkRawLocatorRunsConsolidateBeforeRebuild) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_locator_runs_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string first_vector_path = root + "/first.fbin";
  const std::string first_docid_path = root + "/first.u64";
  write_raw_fbin_file(first_vector_path, 2, 2, {3.0F, 0.0F, 1.0F, 0.0F});
  write_raw_docid_file(first_docid_path, {30, 10});

  const std::string second_vector_path = root + "/second.fbin";
  const std::string second_docid_path = root + "/second.u64";
  write_raw_fbin_file(second_vector_path, 2, 2, {2.0F, 0.0F, 4.0F, 0.0F});
  write_raw_docid_file(second_docid_path, {20, 40});

  const std::string index_name = "idx_raw_locator_runs";
  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index(index_name, 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files(index_name, first_vector_path,
                                          first_docid_path, 2, 2, 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files(index_name, second_vector_path,
                                          second_docid_path, 2, 2, 2));
  EXPECT_EQ(2U, store.raw_locator_run_count(index_name));
  EXPECT_EQ(4U, store.raw_locator_entry_count(index_name));

  vector_index::committed_entries before_rebuild;
  ASSERT_TRUE(
      store.find_entries(index_name, {10, 20, 30, 40}, &before_rebuild));
  ASSERT_EQ(4U, before_rebuild.size());

  ASSERT_TRUE(store.prepare_raw_segments_for_rebuild(index_name));
  EXPECT_EQ(1U, store.raw_locator_run_count(index_name));
  EXPECT_EQ(4U, store.raw_locator_entry_count(index_name));

  vector_index::committed_entries after_rebuild;
  ASSERT_TRUE(store.find_entries(index_name, {10, 20, 30, 40}, &after_rebuild));
  EXPECT_EQ(before_rebuild, after_rebuild);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     FindEntriesOverlaysResidentUpdatesAndErases) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_find_resident_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {1.0F, 0.0F, 2.0F, 0.0F});
  write_raw_docid_file(docid_path, {10, 20});

  vector_index::standalone_entry_store store;
  const std::string index_name = "idx_find_resident";
  ASSERT_TRUE(store.register_index(index_name, 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files(index_name, vector_path, docid_path,
                                          2, 2, 2));
  ASSERT_TRUE(store.upsert(index_name, 10, {9.0F, 0.0F},
                           std::numeric_limits<size_t>::max()));
  ASSERT_TRUE(store.upsert(index_name, 30, {3.0F, 0.0F},
                           std::numeric_limits<size_t>::max()));
  ASSERT_TRUE(store.erase(index_name, 20, std::numeric_limits<size_t>::max()));
  EXPECT_EQ(1U, store.segment_count(index_name));

  vector_index::committed_entries vectors;
  ASSERT_TRUE(store.find_entries(index_name, {10, 20, 99}, &vectors));
  ASSERT_EQ(1U, vectors.size());
  EXPECT_EQ((vector_index::vector_data{9.0F, 0.0F}), vectors.at(10));

  ASSERT_TRUE(store.find_entries(index_name, {30}, &vectors));
  ASSERT_EQ(1U, vectors.size());
  EXPECT_EQ((vector_index::vector_data{3.0F, 0.0F}), vectors.at(30));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest, FindEntriesRejectsMissingRawArtifacts) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_find_missing_raw_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {1.0F, 0.0F, 2.0F, 0.0F});
  write_raw_docid_file(docid_path, {10, 20});

  vector_index::standalone_entry_store store;
  const std::string index_name = "idx_find_missing_raw";
  ASSERT_TRUE(store.register_index(index_name, 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files(index_name, vector_path, docid_path,
                                          2, 2, 2));

  const std::string manifest = find_manifest_file(root, "manifest.v1");
  ASSERT_FALSE(manifest.empty());
  const std::string segment_directory =
      std::filesystem::path(manifest).parent_path().string();
  const std::string stored_fbin =
      find_standalone_file_with_extension(segment_directory, ".fbin");
  const std::string stored_docid =
      find_standalone_file_with_extension(segment_directory, ".u64");
  ASSERT_FALSE(stored_fbin.empty());
  ASSERT_FALSE(stored_docid.empty());

  const std::string saved_fbin = stored_fbin + ".saved";
  std::filesystem::rename(stored_fbin, saved_fbin, ec);
  ASSERT_FALSE(ec);
  vector_index::committed_entries vectors;
  EXPECT_FALSE(store.find_entries(index_name, {10}, &vectors));
  std::filesystem::rename(saved_fbin, stored_fbin, ec);
  ASSERT_FALSE(ec);

  const std::string saved_docid = stored_docid + ".saved";
  std::filesystem::rename(stored_docid, saved_docid, ec);
  ASSERT_FALSE(ec);
  EXPECT_FALSE(store.find_entries(index_name, {10}, &vectors));
  std::filesystem::rename(saved_docid, stored_docid, ec);
  ASSERT_FALSE(ec);

  ASSERT_TRUE(store.find_entries(index_name, {20}, &vectors));
  ASSERT_EQ(1U, vectors.size());
  EXPECT_EQ((vector_index::vector_data{2.0F, 0.0F}), vectors.at(20));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest, FindEntriesUsesDenseRawDocidOffsets) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_find_dense_docids_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  write_raw_fbin_file(vector_path, 3, 2,
                      {3.0F, 0.0F, 1.0F, 0.0F, 2.0F, 0.0F});

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_dense_find_entries", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_dense_find_entries",
                                          vector_path, "", 3, 2, 3));
  EXPECT_EQ(0U, store.raw_locator_entry_count("idx_dense_find_entries"));

  const std::string stored_docid =
      find_standalone_file_with_extension(root, ".u64");
  ASSERT_FALSE(stored_docid.empty());
  std::filesystem::remove(stored_docid, ec);
  ASSERT_FALSE(ec);

  vector_index::committed_entries vectors;
  ASSERT_TRUE(store.find_entries("idx_dense_find_entries", {1, 2}, &vectors));
  ASSERT_EQ(2U, vectors.size());
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), vectors.at(1));
  EXPECT_EQ((vector_index::vector_data{2.0F, 0.0F}), vectors.at(2));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     FindEntriesValidatesGuardsAndRawSegmentContents) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_find_entries_edges_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {1.0F, 0.0F, 2.0F, 0.0F});
  write_raw_docid_file(docid_path, {10, 20});

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_find_edges", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_find_edges", vector_path,
                                          docid_path, 2, 2, 2));
  vector_index::committed_entries vectors;
  EXPECT_FALSE(store.find_entries("idx_find_edges", {10}, nullptr));
  EXPECT_TRUE(store.find_entries("idx_find_edges", {}, &vectors));
  EXPECT_TRUE(vectors.empty());
  EXPECT_FALSE(store.find_entries("missing", {10}, &vectors));
  ASSERT_TRUE(store.find_entries("idx_find_edges", {10}, &vectors));
  ASSERT_EQ(1U, vectors.size());
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), vectors.at(10));

  const std::string stored_fbin =
      find_standalone_file_with_extension(root, ".fbin");
  const std::string stored_docid =
      find_standalone_file_with_extension(root, ".u64");
  ASSERT_FALSE(stored_fbin.empty());
  ASSERT_FALSE(stored_docid.empty());

  write_raw_fbin_file(stored_fbin, 3, 2,
                      {1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F});
  EXPECT_FALSE(store.find_entries("idx_find_edges", {10}, &vectors));
  write_raw_fbin_file(stored_fbin, 2, 3,
                      {1.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F});
  EXPECT_FALSE(store.find_entries("idx_find_edges", {10}, &vectors));
  write_raw_fbin_file(stored_fbin, 2, 2, {1.0F, 0.0F});
  EXPECT_FALSE(store.find_entries("idx_find_edges", {20}, &vectors));
  write_raw_fbin_file(stored_fbin, 2, 2,
                      {1.0F, 0.0F, 2.0F, 0.0F});

  write_raw_docid_file(stored_docid, {10, 20, 30});
  EXPECT_FALSE(store.find_entries("idx_find_edges", {10}, &vectors));
  {
    std::ofstream file(stored_docid, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file.good());
    write_binary_value<uint64_t>(&file, 2);
    write_binary_value<uint64_t>(&file, 10);
  }
  EXPECT_FALSE(store.find_entries("idx_find_edges", {20}, &vectors));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     FindEntriesValidatesDeltaSegmentSizeAndHeader) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_find_delta_edges_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_find_delta", 2));
  ASSERT_TRUE(store.bulk_upsert(
      "idx_find_delta", {{10, {1.0F, 0.0F}}, {20, {2.0F, 0.0F}}}));
  vector_index::committed_entries vectors;
  ASSERT_TRUE(store.find_entries("idx_find_delta", {10}, &vectors));
  ASSERT_EQ(1U, vectors.size());

  const std::string segment = find_standalone_segment_file(root);
  ASSERT_FALSE(segment.empty());
  const uintmax_t original_size = std::filesystem::file_size(segment, ec);
  ASSERT_FALSE(ec);
  {
    std::ofstream file(segment, std::ios::binary | std::ios::app);
    ASSERT_TRUE(file.good());
    file.put('\0');
  }
  EXPECT_FALSE(store.find_entries("idx_find_delta", {10}, &vectors));
  std::filesystem::resize_file(segment, original_size, ec);
  ASSERT_FALSE(ec);
  {
    std::fstream file(segment,
                      std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(file.good());
    file.put('X');
  }
  EXPECT_FALSE(store.find_entries("idx_find_delta", {10}, &vectors));

  std::filesystem::remove_all(root, ec);
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
                                          docid_path, 2, 2, 2));

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

TEST(VectorStandaloneEntryStoreTest,
     RawSegmentReaderRejectsMissingIndexAndInvalidVisitor) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_reader_visitor_t";
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
                                          docid_path, 2, 2, 2));

  EXPECT_FALSE(store.read_rebuild_raw_segments(
      "missing", [](const vector_index::raw_vector_segment &) {
        return true;
      }));
  EXPECT_FALSE(store.read_rebuild_raw_segments(
      "idx_raw_direct", vector_index::raw_vector_segment_visitor{}));

  bool visitor_called = false;
  EXPECT_FALSE(store.read_rebuild_raw_segments(
      "idx_raw_direct",
      [&visitor_called](const vector_index::raw_vector_segment &segment) {
        EXPECT_EQ(2U, segment.row_count);
        visitor_called = true;
        return false;
      }));
  EXPECT_TRUE(visitor_called);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest,
     DeltaSegmentsCannotBypassRawSegmentPreparation) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_delta_raw_reader_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_delta", 2));
  ASSERT_TRUE(store.bulk_upsert("idx_delta", {{20, {2.0F, 0.0F}}}));
  EXPECT_EQ("delta_replay", store.build_source("idx_delta"));
  EXPECT_FALSE(store.read_rebuild_raw_segments(
      "idx_delta", [](const vector_index::raw_vector_segment &) {
        ADD_FAILURE() << "delta segments must be compacted before raw rebuild";
        return true;
      }));

  ASSERT_TRUE(store.prepare_raw_segments_for_rebuild("idx_delta"));
  size_t segment_count = 0;
  ASSERT_TRUE(store.read_rebuild_raw_segments(
      "idx_delta",
      [&segment_count](const vector_index::raw_vector_segment &segment) {
        EXPECT_EQ(1U, segment.row_count);
        ++segment_count;
        return true;
      }));
  EXPECT_EQ(1U, segment_count);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest, RawSegmentReaderRejectsSizeDrift) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_size_drift_t";
  const std::string input_dir = root + "/input";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(input_dir, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = input_dir + "/source.fbin";
  const std::string docid_path = input_dir + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {3.0F, 0.0F, 1.0F, 0.0F});
  write_raw_docid_file(docid_path, {30, 10});

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_raw_drift", 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files("idx_raw_drift", vector_path,
                                          docid_path, 2, 2, 2));

  const std::string stored_docid_path =
      find_raw_segment_file_from_manifest(root, ".u64");
  ASSERT_FALSE(stored_docid_path.empty());
  {
    std::ofstream file(stored_docid_path,
                       std::ios::out | std::ios::binary | std::ios::app);
    ASSERT_TRUE(file.is_open());
    const char extra_byte = '\0';
    file.write(&extra_byte, 1);
    ASSERT_TRUE(file.good());
  }

  EXPECT_FALSE(store.read_rebuild_raw_segments(
      "idx_raw_drift", [](const vector_index::raw_vector_segment &) {
        ADD_FAILURE() << "size drift should reject the raw segment";
        return true;
      }));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest, MixedSegmentsCompactBeforeRawRebuild) {
  build_pipeline_options_guard guard;
  opt_vector_build_segment_max_rows = 1;
  opt_vector_build_segment_target_size = 1024 * 1024;

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
                                          docid_path, 2, 2, 1));
  ASSERT_TRUE(store.upsert("idx_raw_compact", 20, {2.0F, 0.0F}, 0));
  ASSERT_TRUE(store.erase("idx_raw_compact", 30, 0));

  RawSegmentCountingBackend backend;
  ASSERT_TRUE(store.rebuild_backend_input("idx_raw_compact", &backend));
  EXPECT_EQ(1U, backend.raw_segment_reader_calls);
  EXPECT_EQ(0U, backend.committed_reader_calls);
  EXPECT_EQ((std::vector<size_t>{1, 1}), backend.raw_segment_doc_counts);
  EXPECT_EQ(2U, store.segment_count("idx_raw_compact"));
  EXPECT_EQ(2U, store.raw_segment_count("idx_raw_compact"));

  const auto entries = collect_entries(&store, "idx_raw_compact");
  ASSERT_EQ(2U, entries.size());
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), entries.at(10));
  EXPECT_EQ((vector_index::vector_data{2.0F, 0.0F}), entries.at(20));
}

TEST(VectorStandaloneEntryStoreTest, EmptyCompactionClearsDerivedRawLocator) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_empty_compact_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {1.0F, 0.0F, 3.0F, 0.0F});
  write_raw_docid_file(docid_path, {10, 30});

  const std::string index_name = "idx_empty_compact";
  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index(index_name, 2));
  ASSERT_TRUE(store.bulk_upsert_raw_files(index_name, vector_path, docid_path,
                                          2, 2, 2));
  ASSERT_EQ(2U, store.raw_locator_entry_count(index_name));
  ASSERT_TRUE(store.erase(index_name, 10, 0));
  ASSERT_TRUE(store.erase(index_name, 30, 0));
  ASSERT_TRUE(store.prepare_raw_segments_for_rebuild(index_name));
  EXPECT_EQ(0U, store.segment_count(index_name));
  EXPECT_EQ(0U, store.raw_locator_entry_count(index_name));

  vector_index::committed_entries vectors;
  EXPECT_TRUE(store.find_entries(index_name, {10, 30}, &vectors));
  EXPECT_TRUE(vectors.empty());

  std::filesystem::remove_all(root, ec);
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
                                          docid_path, 2, 2, 2));
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
                                          docid_path, 2, 2, 2));

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
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
    GTEST_SKIP() << "DiskANN provider is not compiled in";
  }

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
  EXPECT_EQ(1U, committed_entry_count);
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
  EXPECT_EQ(1U, committed_entry_count);
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
  ASSERT_TRUE(store.upsert("idx_memory", 1, {1.5F, 1.5F},
                           std::numeric_limits<size_t>::max()));
  EXPECT_EQ(2U, store.entry_count("idx_memory"));
  EXPECT_EQ(4U * sizeof(float), store.memory_bytes("idx_memory"));
  EXPECT_EQ(0U, store.segment_count("idx_memory"));
  EXPECT_EQ(3U, store.generation("idx_memory"));

  ASSERT_TRUE(store.find_entry("idx_memory", 1, &vector, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(std::vector<float>({1.5F, 1.5F}), vector);
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
  EXPECT_EQ(0U, store.raw_segment_count("idx_memory"));
  EXPECT_EQ(0U, store.raw_segment_bytes("idx_memory"));
  EXPECT_EQ(0U, store.generation("idx_memory"));
  EXPECT_EQ("", store.build_source("idx_memory"));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorStandaloneEntryStoreTest, RawSegmentRejectsInvalidSourceFiles) {
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_raw_invalid_source_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/valid.fbin";
  const std::string truncated_docid_path = root + "/truncated.u64";
  write_raw_fbin_file(vector_path, 2, 2, {1.0F, 0.0F, 0.0F, 1.0F});
  {
    std::ofstream file(truncated_docid_path,
                       std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file.good());
    write_binary_value<uint64_t>(&file, 2);
    write_binary_value<uint64_t>(&file, 10);
    ASSERT_TRUE(file.good());
  }

  vector_index::standalone_entry_store store;
  ASSERT_TRUE(store.register_index("idx_raw_invalid", 2));

  EXPECT_FALSE(store.bulk_upsert_raw_files("idx_raw_invalid",
                                           root + "/missing.fbin", "", 2, 2,
                                           2));
  EXPECT_FALSE(store.bulk_upsert_raw_files("idx_raw_invalid", vector_path,
                                           truncated_docid_path, 2, 2, 2));
  EXPECT_EQ(0U, store.entry_count("idx_raw_invalid"));
  EXPECT_EQ(0U, store.raw_segment_count("idx_raw_invalid"));
  EXPECT_EQ(0U, store.raw_segment_bytes("idx_raw_invalid"));

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
                                            docid_path, 2, 2, 2));
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
  const char *provider = segmented_memory_provider_for_testing();
  if (provider == nullptr) {
    GTEST_SKIP() << "No memory provider is available for segmented tests";
  }
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
                                                  provider));
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
  EXPECT_EQ(1U, committed_entry_count);
  ASSERT_TRUE(service.search("idx_standalone_spill", {1.0F, 0.0F}, 2,
                             &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  vector_index::index_service::committed_state committed_state;
  ASSERT_TRUE(service.snapshot_committed_state(&committed_state));
  vector_index::index_service recovered;
  ASSERT_TRUE(recovered.register_index_from_strings("idx_standalone_spill", 2,
                                                    "euclidean", "memory",
                                                    provider));
  ASSERT_TRUE(recovered.set_index_consistency_mode(
      "idx_standalone_spill",
      vector_index::index_consistency_mode::kStandalone));
  ASSERT_TRUE(recovered.restore_committed_state(committed_state));

  vector_index::backend_build_diagnostics build_diagnostics;
  ASSERT_TRUE(recovered.describe_index(
      "idx_standalone_spill", &config, nullptr, &entry_count,
      &committed_entry_count, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, &build_diagnostics));
  EXPECT_EQ(0U, entry_count);
  EXPECT_EQ(1U, committed_entry_count);
  EXPECT_EQ(0U, build_diagnostics.build_invocations);

  ASSERT_TRUE(recovered.search("idx_standalone_spill", {1.0F, 0.0F}, 2,
                               &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  ASSERT_TRUE(recovered.describe_index(
      "idx_standalone_spill", &config, nullptr, &entry_count,
      &committed_entry_count, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, &build_diagnostics));
  EXPECT_EQ(1U, entry_count);
  EXPECT_EQ(1U, committed_entry_count);
  EXPECT_EQ(1U, build_diagnostics.build_invocations);
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
  ASSERT_TRUE(service.rebuild_index("idx_faiss_ivf"));
  vector_index::index_service::index_config config;
  vector_index::backend_build_diagnostics diagnostics;
  ASSERT_TRUE(service.describe_index("idx_faiss_ivf", &config, nullptr, nullptr,
                                     nullptr, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, &diagnostics));
  EXPECT_EQ("reader", diagnostics.input_source);
  EXPECT_EQ(2U, diagnostics.reader_passes);
  EXPECT_EQ(2U, diagnostics.training_rows);
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

TEST(VectorIndexServiceTest, InstallAndReplaceCommittedEntriesCoverGuards) {
  vector_index::index_service service;
  vector_index::index_service::committed_entries entries;
  entries.emplace(7, vector_index::vector_data{7.0F, 7.0F});

  EXPECT_FALSE(service.replace_committed_entries("missing", entries));
  EXPECT_FALSE(
      service.replace_committed_entries_preserve_lifecycle("missing", entries));
  EXPECT_FALSE(service.install_rebuilt_index(
      "missing", entries,
      std::make_unique<vector_index::memory_backend>(
          2, vector_index::metric_type::kEuclidean)));
  EXPECT_FALSE(service.install_recovered_index(
      "missing", entries,
      std::make_unique<vector_index::memory_backend>(
          2, vector_index::metric_type::kEuclidean),
      true));

  ASSERT_TRUE(service.register_index_from_strings("idx_install_entries", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  EXPECT_FALSE(service.replace_committed_entries(
      "idx_install_entries", {{8, vector_index::vector_data{8.0F}}}));
  EXPECT_FALSE(service.replace_committed_entries_preserve_lifecycle(
      "idx_install_entries", {{8, vector_index::vector_data{8.0F}}}));
  EXPECT_FALSE(
      service.install_rebuilt_index("idx_install_entries", entries, nullptr));
  EXPECT_FALSE(service.install_recovered_index("idx_install_entries", entries,
                                               nullptr, true));

  auto rebuilt_backend = std::make_unique<vector_index::memory_backend>(
      2, vector_index::metric_type::kEuclidean);
  ASSERT_TRUE(rebuilt_backend->rebuild_from_committed_entries(entries));
  ASSERT_TRUE(service.install_rebuilt_index("idx_install_entries", entries,
                                            std::move(rebuilt_backend)));

  vector_index::index_service::committed_entries recovered_entries;
  recovered_entries.emplace(9, vector_index::vector_data{9.0F, 9.0F});
  auto recovered_backend = std::make_unique<vector_index::memory_backend>(
      2, vector_index::metric_type::kEuclidean);
  ASSERT_TRUE(
      recovered_backend->rebuild_from_committed_entries(recovered_entries));
  ASSERT_TRUE(service.install_recovered_index("idx_install_entries",
                                              recovered_entries,
                                              std::move(recovered_backend),
                                              true));

  vector_index::index_service::index_config config;
  uint64_t recover_fallback_count = 0;
  ASSERT_TRUE(service.describe_index(
      "idx_install_entries", &config, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, &recover_fallback_count));
  EXPECT_EQ(1U, recover_fallback_count);
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
  EXPECT_TRUE(
      service.set_diskann_search_beamwidth("idx_diskann_search_zero", 128));
  EXPECT_FALSE(
      service.set_diskann_search_beamwidth("idx_diskann_search_zero", 129));
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

TEST(VectorIndexServiceTest, DiskAnnSearchExactReranksCommittedCandidates) {
  vector_index::index_service service;
  auto backend = std::make_unique<RerankProbeDiskAnnBackend>();
  RerankProbeDiskAnnBackend *probe = backend.get();

  ASSERT_TRUE(service.register_index("idx_rerank", std::move(backend)));
  ASSERT_TRUE(service.stage_upsert(1, "idx_rerank", 1, {0.0F, 0.0F}));
  ASSERT_TRUE(service.stage_upsert(1, "idx_rerank", 2, {10.0F, 10.0F}));
  ASSERT_TRUE(service.commit(1));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_rerank", {10.0F, 10.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
  ASSERT_FALSE(probe->requested_top_k_values.empty());
  EXPECT_EQ(2U, probe->requested_top_k_values.back());
  ASSERT_FALSE(probe->rerank_top_k_values.empty());
  EXPECT_EQ(1U, probe->rerank_top_k_values.back());
  ASSERT_FALSE(probe->rerank_candidate_top_k_values.empty());
  EXPECT_EQ(2U, probe->rerank_candidate_top_k_values.back());
}

TEST(VectorIndexServiceTest, DiskAnnBatchSearchExactReranksCommittedCandidates) {
  vector_index::index_service service;
  auto backend = std::make_unique<RerankProbeDiskAnnBackend>();
  RerankProbeDiskAnnBackend *probe = backend.get();

  ASSERT_TRUE(service.register_index("idx_batch_rerank", std::move(backend)));
  ASSERT_TRUE(service.stage_upsert(1, "idx_batch_rerank", 1, {0.0F, 0.0F}));
  ASSERT_TRUE(service.stage_upsert(1, "idx_batch_rerank", 2, {10.0F, 10.0F}));
  ASSERT_TRUE(service.commit(1));

  std::vector<std::vector<vector_index::search_result>> results;
  ASSERT_TRUE(service.search_batch(
      "idx_batch_rerank", {{10.0F, 10.0F}, {0.0F, 0.0F}}, 1, &results));
  ASSERT_EQ(2U, results.size());
  ASSERT_EQ(1U, results[0].size());
  EXPECT_EQ(2U, results[0][0].doc_id);
  ASSERT_EQ(1U, results[1].size());
  EXPECT_EQ(1U, results[1][0].doc_id);
  ASSERT_FALSE(probe->requested_top_k_values.empty());
  EXPECT_EQ(2U, probe->requested_top_k_values.back());
  ASSERT_FALSE(probe->rerank_top_k_values.empty());
  EXPECT_EQ(1U, probe->rerank_top_k_values.back());
  ASSERT_FALSE(probe->rerank_candidate_top_k_values.empty());
  EXPECT_EQ(2U, probe->rerank_candidate_top_k_values.back());
}

TEST(VectorIndexServiceTest,
     DiskAnnExactRerankUsesAutomaticCandidateWindowForSmallIndex) {
  vector_index::index_service service;
  auto backend = std::make_unique<RerankProbeDiskAnnBackend>(64);
  RerankProbeDiskAnnBackend *probe = backend.get();

  ASSERT_TRUE(service.register_index("idx_rerank_slack", std::move(backend)));
  for (uint64_t doc_id = 1; doc_id <= 100; ++doc_id) {
    ASSERT_TRUE(service.stage_upsert(1, "idx_rerank_slack", doc_id,
                                     {static_cast<float>(doc_id), 0.0F}));
  }
  ASSERT_TRUE(service.commit(1));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search("idx_rerank_slack", {2.0F, 0.0F}, 10, &result));
  ASSERT_FALSE(probe->requested_top_k_values.empty());
  const size_t requested_top_k = probe->requested_top_k_values.back();
  EXPECT_EQ(63U, requested_top_k);
}

TEST(VectorIndexServiceTest,
     DiskAnnExactRerankCandidateTopKUsesGlobalFanoutWindow) {
  constexpr size_t top_k = 10;
  constexpr size_t query_count = 100;
  constexpr size_t row_count = 4194304;
  constexpr size_t segment_count = 65;
  constexpr size_t result_budget = 262144;

  EXPECT_EQ(780U, vector_index::diskann_exact_rerank_candidate_top_k_for_testing(
                       top_k, query_count, row_count, segment_count, 800,
                       vector_index::diskann_search_profile::kManual,
                       result_budget, 0));
  EXPECT_EQ(1560U,
            vector_index::diskann_exact_rerank_candidate_top_k_for_testing(
                top_k, query_count, row_count, segment_count, 800,
                vector_index::diskann_search_profile::kHighRecall,
                result_budget, 0));
  EXPECT_EQ(2621U,
            vector_index::diskann_exact_rerank_candidate_top_k_for_testing(
                top_k, query_count, row_count, segment_count, 3200,
                vector_index::diskann_search_profile::kHighRecall,
                result_budget, 0));
}

TEST(VectorIndexServiceTest, DiskAnnExactRerankCandidateTopKCapsToBudget) {
  constexpr size_t top_k = 10;
  constexpr size_t query_count = 100;
  constexpr size_t row_count = 4194304;
  constexpr size_t segment_count = 65;
  constexpr size_t result_budget = 51200;

  EXPECT_EQ(512U,
            vector_index::diskann_exact_rerank_candidate_top_k_for_testing(
                top_k, query_count, row_count, segment_count, 800,
                vector_index::diskann_search_profile::kManual,
                result_budget, 8192));
}

TEST(VectorIndexServiceTest,
     DiskAnnExactRerankCandidateTopKKeepsComputedWindowWhenBudgetAllows) {
  constexpr size_t top_k = 10;
  constexpr size_t query_count = 100;
  constexpr size_t row_count = 4194304;
  constexpr size_t segment_count = 65;
  constexpr size_t result_budget = 1048576;

  EXPECT_EQ(780U,
            vector_index::diskann_exact_rerank_candidate_top_k_for_testing(
                top_k, query_count, row_count, segment_count, 800,
                vector_index::diskann_search_profile::kManual,
                result_budget, 0));
}

TEST(VectorIndexServiceTest,
     DiskAnnExactRerankCandidateTopKUsesExplicitTargetWhenBudgetAllows) {
  constexpr size_t top_k = 10;
  constexpr size_t query_count = 100;
  constexpr size_t row_count = 4194304;
  constexpr size_t segment_count = 65;
  constexpr size_t result_budget = 1048576;

  EXPECT_EQ(8192U,
            vector_index::diskann_exact_rerank_candidate_top_k_for_testing(
                top_k, query_count, row_count, segment_count, 800,
                vector_index::diskann_search_profile::kManual,
                result_budget, 8192));
}

TEST(VectorIndexServiceTest,
     SegmentedDiskAnnKeepsConcurrentSearchOptionsRequestLocal) {
  vector_index::index_service::index_config config;
  config.dimension = 2;
  config.metric = vector_index::metric_type::kEuclidean;
  config.mode = vector_index::backend_mode::kExternal;
  config.provider = vector_index::backend_provider::kDiskAnn;
  config.diskann_search_complexity = 4;
  config.diskann_search_beamwidth = 16;

  auto segment = std::make_shared<SearchOptionsProbeDiskAnnBackend>();
  SearchOptionsProbeDiskAnnBackend *probe = segment.get();
  std::vector<std::shared_ptr<vector_index::backend>> segments;
  segments.push_back(std::move(segment));
  auto segmented = vector_index::make_segmented_backend_for_testing(
      config, std::move(segments));

  const auto run_search = [&](size_t top_k) {
    std::vector<vector_index::search_result> results;
    return segmented->search({0.0F, 0.0F}, top_k, &results);
  };
  auto small_search = std::async(std::launch::async, run_search, 10);
  auto large_search = std::async(std::launch::async, run_search, 100);
  EXPECT_TRUE(small_search.get());
  EXPECT_TRUE(large_search.get());

  auto observed = probe->observed_options();
  ASSERT_EQ(2U, observed.size());
  size_t small_option_count = 0;
  size_t large_option_count = 0;
  for (const auto &options : observed) {
    EXPECT_EQ(16U, options.diskann_search_beamwidth);
    if (options.diskann_search_complexity == 11) ++small_option_count;
    if (options.diskann_search_complexity == 101) ++large_option_count;
  }
  EXPECT_EQ(1U, small_option_count);
  EXPECT_EQ(1U, large_option_count);
  EXPECT_EQ(0U, probe->setter_calls());

  probe->reset();
  const std::vector<vector_index::vector_data> queries{
      {0.0F, 0.0F}, {1.0F, 1.0F}};
  const auto run_batch_search = [&](size_t top_k) {
    std::vector<std::vector<vector_index::search_result>> results;
    return segmented->search_batch(queries, top_k, &results);
  };
  auto small_batch = std::async(std::launch::async, run_batch_search, 20);
  auto large_batch = std::async(std::launch::async, run_batch_search, 120);
  EXPECT_TRUE(small_batch.get());
  EXPECT_TRUE(large_batch.get());

  observed = probe->observed_options();
  ASSERT_EQ(2U, observed.size());
  small_option_count = 0;
  large_option_count = 0;
  for (const auto &options : observed) {
    EXPECT_EQ(16U, options.diskann_search_beamwidth);
    if (options.diskann_search_complexity == 21) ++small_option_count;
    if (options.diskann_search_complexity == 121) ++large_option_count;
  }
  EXPECT_EQ(1U, small_option_count);
  EXPECT_EQ(1U, large_option_count);
  EXPECT_EQ(0U, probe->setter_calls());
}

TEST(VectorIndexServiceTest,
     SegmentedDiskAnnRerankOverfetchesBeforeGlobalCandidateMerge) {
  constexpr size_t k_top_k = 10;
  constexpr size_t k_segment_count = 5;
  constexpr size_t k_entries_per_segment = 1000;
  constexpr size_t k_query_count = 32;
  constexpr size_t k_result_budget = 262144;
  constexpr uint32_t k_search_complexity = 400;
  UlongGuard result_budget_guard(&opt_vector_search_batch_result_count,
                                 k_result_budget);
  UlongGuard candidate_target_guard(
      &opt_vector_diskann_exact_rerank_candidates, 0);

  vector_index::index_service::index_config config;
  config.dimension = 2;
  config.metric = vector_index::metric_type::kEuclidean;
  config.mode = vector_index::backend_mode::kExternal;
  config.provider = vector_index::backend_provider::kDiskAnn;
  config.diskann_search_complexity = k_search_complexity;

  std::vector<std::shared_ptr<vector_index::backend>> segments;
  std::vector<RerankProbeDiskAnnBackend *> probes;
  for (size_t segment_index = 0; segment_index < k_segment_count;
       ++segment_index) {
    auto segment =
        std::make_shared<RerankProbeDiskAnnBackend>(k_search_complexity);
    for (size_t row_index = 0; row_index < k_entries_per_segment;
         ++row_index) {
      const uint64_t doc_id = segment_index * k_entries_per_segment +
                              row_index + 1;
      ASSERT_TRUE(segment->upsert(doc_id, {static_cast<float>(doc_id), 0.0F}));
    }
    probes.push_back(segment.get());
    segments.push_back(std::move(segment));
  }

  auto segmented = vector_index::make_segmented_backend_for_testing(
      config, std::move(segments));
  const size_t global_candidate_top_k =
      vector_index::diskann_exact_rerank_candidate_top_k_for_testing(
          k_top_k, 1, k_segment_count * k_entries_per_segment,
          k_segment_count, k_search_complexity,
          vector_index::diskann_search_profile::kManual, k_result_budget, 0);
  ASSERT_EQ(395U, global_candidate_top_k);

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(segmented->search_for_rerank({0.0F, 0.0F}, k_top_k,
                                           global_candidate_top_k, &results));
  for (const auto *probe : probes) {
    ASSERT_FALSE(probe->requested_top_k_values.empty());
    EXPECT_EQ(158U, probe->requested_top_k_values.back());
  }
  auto diagnostics = segmented->build_diagnostics();
  EXPECT_EQ(158U, diagnostics.search_per_segment_top_k);
  EXPECT_EQ(790U, diagnostics.search_candidate_count);
  EXPECT_EQ(790U, diagnostics.search_total_candidate_rows);

  for (auto *probe : probes) probe->requested_top_k_values.clear();
  std::vector<vector_index::vector_data> queries(
      k_query_count, vector_index::vector_data{0.0F, 0.0F});
  vector_index::batch_search_candidates batch_results;
  ASSERT_TRUE(segmented->search_batch_for_rerank(
      queries, k_top_k, global_candidate_top_k, &batch_results));
  ASSERT_EQ(k_query_count, batch_results.query_count());
  for (const auto *probe : probes) {
    ASSERT_FALSE(probe->requested_top_k_values.empty());
    EXPECT_EQ(158U, probe->requested_top_k_values.back());
  }
  diagnostics = segmented->build_diagnostics();
  EXPECT_EQ(158U, diagnostics.search_per_segment_top_k);
  EXPECT_EQ(790U, diagnostics.search_candidate_count);
  EXPECT_EQ(k_query_count * 790U,
            diagnostics.search_total_candidate_rows);
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

TEST(VectorIndexServiceTest, TuningApiGuardsRejectInvalidTargetsAndStates) {
  vector_index::index_service service;

  EXPECT_FALSE(service.set_search_ef("missing", 16));
  EXPECT_FALSE(service.set_hnsw_build_params("missing", 16, 64));
  EXPECT_FALSE(service.set_hnsw_build_threads("missing", 2));
  EXPECT_FALSE(service.set_faiss_ivf_params("missing", 8, 2));
  EXPECT_FALSE(service.set_faiss_ivf_pq_params("missing", 8, 2, 2, 4));
  EXPECT_FALSE(service.set_faiss_build_threads("missing", 2));
  EXPECT_FALSE(service.set_diskann_build_params("missing", 32, 64, 2));
  EXPECT_FALSE(service.set_diskann_build_threads("missing", 2));
  EXPECT_FALSE(service.set_diskann_build_mode(
      "missing", vector_index::diskann_build_mode::kSerial));
  EXPECT_FALSE(service.set_diskann_search_complexity("missing", 64));
  EXPECT_FALSE(service.set_diskann_search_beamwidth("missing", 8));
  EXPECT_FALSE(service.set_diskann_pq_code_budget_size("missing", 1024));

  ASSERT_TRUE(service.register_index_from_strings("idx_tuning_guard", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  EXPECT_FALSE(service.set_hnsw_build_params("idx_tuning_guard", 0, 64));
  EXPECT_FALSE(service.set_hnsw_build_params("idx_tuning_guard", 16, 0));
  EXPECT_FALSE(service.set_faiss_ivf_params("idx_tuning_guard", 8, 2));
  EXPECT_FALSE(service.set_diskann_build_params("idx_tuning_guard", 0, 64, 2));
  EXPECT_FALSE(service.set_diskann_build_params("idx_tuning_guard", 32, 0, 2));
  EXPECT_TRUE(service.set_diskann_build_mode(
      "idx_tuning_guard", vector_index::diskann_build_mode::kAuto));
  EXPECT_FALSE(service.set_diskann_build_mode(
      "idx_tuning_guard", vector_index::diskann_build_mode::kSerial));

  ASSERT_TRUE(service.stage_upsert(9911, "idx_tuning_guard", 1,
                                   {1.0F, 1.0F}));
  EXPECT_FALSE(service.set_hnsw_build_params("idx_tuning_guard", 16, 64));
  EXPECT_FALSE(service.set_hnsw_build_threads("idx_tuning_guard", 2));
  EXPECT_FALSE(service.set_faiss_ivf_pq_params("idx_tuning_guard", 8, 2, 2, 4));
  EXPECT_FALSE(service.set_faiss_build_threads("idx_tuning_guard", 2));
  EXPECT_FALSE(service.set_diskann_build_params("idx_tuning_guard", 32, 64, 2));
  EXPECT_FALSE(service.set_diskann_build_threads("idx_tuning_guard", 2));
  EXPECT_FALSE(service.set_diskann_pq_code_budget_size("idx_tuning_guard",
                                                       1024));
  service.rollback(9911);
}

TEST(VectorIndexServiceTest, SetLifecycleStateRejectsInvalidAndClearsErrors) {
  vector_index::index_service service;

  EXPECT_FALSE(service.set_lifecycle_state("missing", "ready"));
  ASSERT_TRUE(service.register_index_from_strings("idx_lifecycle_state", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  EXPECT_FALSE(service.set_lifecycle_state("idx_lifecycle_state", ""));
  ASSERT_TRUE(
      service.set_lifecycle_info("idx_lifecycle_state", "failed", 9, 1006, 77));
  ASSERT_TRUE(service.set_lifecycle_state("idx_lifecycle_state", "ready"));

  vector_index::index_service::index_config config;
  std::string lifecycle_state;
  uint64_t lifecycle_version = 0;
  uint32_t last_error_code = 1006;
  uint64_t last_error_ts = 77;
  ASSERT_TRUE(service.describe_index(
      "idx_lifecycle_state", &config, nullptr, nullptr, nullptr,
      &lifecycle_state, &lifecycle_version, &last_error_code, &last_error_ts));
  EXPECT_EQ("ready", lifecycle_state);
  EXPECT_EQ(10U, lifecycle_version);
  EXPECT_EQ(0U, last_error_code);
  EXPECT_EQ(0U, last_error_ts);

  ASSERT_TRUE(service.set_lifecycle_state("idx_lifecycle_state", "failed"));
  ASSERT_TRUE(service.describe_index(
      "idx_lifecycle_state", &config, nullptr, nullptr, nullptr,
      &lifecycle_state, &lifecycle_version, &last_error_code, &last_error_ts));
  EXPECT_EQ("failed", lifecycle_state);
  EXPECT_EQ(11U, lifecycle_version);
  EXPECT_EQ(0U, last_error_code);
  EXPECT_EQ(0U, last_error_ts);
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
     LazyExternalRuntimeReloadsStandaloneFromStandaloneStore) {
  BoolGuard lazy_runtime_guard(&opt_vector_lazy_external_runtime, true);
  const std::string root =
      std::string(testing::TempDir()) + "/standalone_lazy_runtime_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {1.0F, 0.0F, 0.0F, 1.0F});
  write_raw_docid_file(docid_path, {31, 42});

  vector_index::index_service service;
  const std::string index_name = "idx_standalone_lazy_runtime";
  ASSERT_TRUE(service.register_index_from_strings(
      index_name, 2, "euclidean", "external", "diskann"));
  ASSERT_TRUE(service.set_index_consistency_mode(
      index_name, vector_index::index_consistency_mode::kStandalone));

  vector_index::index_service::bulk_load_options options;
  options.rebuild_after_load = true;
  uint64_t loaded_rows = 0;
  std::string error;
  ASSERT_TRUE(service.bulk_upsert_from_raw_files(
      index_name, vector_path, docid_path, options, &loaded_rows, &error))
      << error;
  ASSERT_EQ(2U, loaded_rows);

  vector_index::index_service::index_config config;
  size_t entry_count = 0;
  size_t committed_entry_count = 0;
  ASSERT_TRUE(service.describe_index(index_name, &config, nullptr, &entry_count,
                                     &committed_entry_count));
  EXPECT_EQ(0U, entry_count);
  EXPECT_EQ(2U, committed_entry_count);

  ASSERT_TRUE(service.ensure_runtime_loaded_for_search(index_name));
  ASSERT_TRUE(service.describe_index(index_name, &config, nullptr, &entry_count,
                                     &committed_entry_count));
  EXPECT_EQ(2U, entry_count);
  EXPECT_EQ(2U, committed_entry_count);

  ASSERT_TRUE(service.rebuild_runtime_from_store_for_search(index_name));
  ASSERT_TRUE(service.describe_index(index_name, &config, nullptr, &entry_count,
                                     &committed_entry_count));
  EXPECT_EQ(2U, entry_count);
  EXPECT_EQ(2U, committed_entry_count);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search_loaded(index_name, {0.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(42U, result[0].doc_id);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     LazyExternalRuntimeReappliesSegmentedBuildPolicy) {
#ifdef __APPLE__
  GTEST_SKIP() << "DiskANN native runtime is validated on Linux";
#endif
  if (!vector_index::diskann_api_load_for_testing() ||
      !vector_index::diskann_vendored_runtime_api_load_for_testing()) {
    GTEST_SKIP() << "DiskANN runtime unavailable in current build";
  }
  install_diskann_offline_test_adapter();
  if (!vector_index::diskann_offline_api_manifest_build_load_for_testing()) {
    GTEST_SKIP() << "DiskANN offline manifest adapter unavailable in current "
                    "build/runtime";
  }

  build_pipeline_options_guard pipeline_guard;
  BoolGuard lazy_runtime_guard(&opt_vector_lazy_external_runtime, true);
  opt_vector_build_pipeline_mode =
      static_cast<ulong>(vector_index::build_pipeline_mode::kSegmented);
  opt_vector_build_segment_max_rows = 4;
  opt_vector_build_pipeline_max_tasks = 2;
  opt_vector_diskann_segmented_serving = true;

  const std::string root =
      std::string(testing::TempDir()) + "/standalone_lazy_segmented_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  faiss_snapshot_root_guard root_guard(root);

  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 8, 2,
                      {1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F, 4.0F, 4.0F,
                       9.0F, 9.0F, 8.0F, 8.0F, 7.0F, 7.0F, 6.0F, 6.0F});
  write_raw_docid_file(docid_path, {10, 20, 30, 40, 90, 80, 70, 60});

  const std::string index_name = "idx_standalone_lazy_segmented";
  vector_index::index_service::index_config persisted_config;
  {
    vector_index::index_service service;
    vector_index::index_service::index_config config;
    config.dimension = 2;
    config.metric = vector_index::metric_type::kEuclidean;
    config.mode = vector_index::backend_mode::kExternal;
    config.provider = vector_index::backend_provider::kDiskAnn;
    config.consistency_mode =
        vector_index::index_consistency_mode::kStandalone;
    config.diskann_build_mode_value =
        vector_index::diskann_build_mode::kOffline;
    config.diskann_build_mode_specified = true;
    ASSERT_TRUE(service.register_index(index_name, config));

    vector_index::index_service::bulk_load_options options;
    options.rebuild_after_load = true;
    uint64_t loaded_rows = 0;
    std::string error;
    ASSERT_TRUE(service.bulk_upsert_from_raw_files(
        index_name, vector_path, docid_path, options, &loaded_rows, &error))
        << error;
    ASSERT_EQ(8U, loaded_rows);
    ASSERT_TRUE(service.describe_index(index_name, &persisted_config, nullptr,
                                       nullptr, nullptr));
    EXPECT_EQ("segmented", persisted_config.backend_variant);
  }

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index(index_name, persisted_config));

  auto expect_segmented_runtime = [&]() {
    vector_index::index_service::index_config config;
    size_t entry_count = 0;
    ASSERT_TRUE(service.describe_index(index_name, &config, nullptr,
                                       &entry_count, nullptr));
    EXPECT_EQ("segmented", config.backend_variant);
    EXPECT_EQ(8U, entry_count);

    vector_index::index_service::build_pipeline_snapshot pipeline;
    ASSERT_TRUE(service.describe_build_pipeline(index_name, &pipeline));
    EXPECT_EQ("segmented", pipeline.mode);
    EXPECT_EQ("segmented", pipeline.decision);

    std::vector<vector_index_metadata_store::segment_task_row> tasks;
    ASSERT_TRUE(service.snapshot_segment_tasks(index_name, &tasks));
    EXPECT_EQ(2U, tasks.size());
  };

  ASSERT_TRUE(service.ensure_runtime_loaded_for_search(index_name));
  expect_segmented_runtime();

  ASSERT_TRUE(service.rebuild_runtime_from_store_for_search(index_name));
  expect_segmented_runtime();

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(service.search_loaded(index_name, {8.0F, 8.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(80U, result[0].doc_id);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexServiceTest,
     LazyExternalRuntimeCoversDisabledAndAlreadyLoadedBranches) {
  BoolGuard guard(&opt_vector_lazy_external_runtime, false);

  {
    if (vector_index::backend_provider_supported(
            vector_index::backend_provider::kDiskAnn)) {
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

  if (vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
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

  if (vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
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
  config.diskann_disk_pq_dims = 12;
  config.diskann_accelerate_build = true;
  config.diskann_shuffle_build = true;
  config.diskann_use_bfs_cache = true;

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
  EXPECT_EQ(12U, described.diskann_disk_pq_dims);
  EXPECT_TRUE(described.diskann_accelerate_build);
  EXPECT_TRUE(described.diskann_shuffle_build);
  EXPECT_TRUE(described.diskann_use_bfs_cache);
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

  if (vector_index::backend_provider_supported(
          vector_index::backend_provider::kDiskAnn)) {
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

  vector_index::index_service::index_config bad_diskann_disk_pq_on_native =
      bad_diskann_search_on_native;
  bad_diskann_disk_pq_on_native.diskann_search_complexity = 0;
  bad_diskann_disk_pq_on_native.diskann_disk_pq_dims = 12;
  EXPECT_FALSE(service.register_index("idx_bad_diskann_disk_pq_on_native",
                                     bad_diskann_disk_pq_on_native));

  vector_index::index_service::index_config bad_diskann_flag_on_native =
      bad_diskann_search_on_native;
  bad_diskann_flag_on_native.diskann_search_complexity = 0;
  bad_diskann_flag_on_native.diskann_use_bfs_cache = true;
  EXPECT_FALSE(service.register_index("idx_bad_diskann_flag_on_native",
                                     bad_diskann_flag_on_native));

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

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
                             "+d,vector_service_fail_restore_index_config");
    EXPECT_FALSE(service.restore_index_config("idx_guard", config));
  }

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

TEST(VectorIndexServiceTest, InternalRebuildGuardsCoverNegativeModes) {
  using vector_index::backend_mode;
  using vector_index::backend_provider;
  using vector_index::detail::all_true;
  using vector_index::detail::can_rebuild_after_search_failure;

  EXPECT_TRUE(all_true(true));
  EXPECT_FALSE(all_true(false));
  EXPECT_TRUE(all_true(true, true));
  EXPECT_FALSE(all_true(false, true));
  EXPECT_FALSE(all_true(true, false));
  EXPECT_TRUE(all_true(true, 1, "non-null"));
  EXPECT_FALSE(all_true(false, true, true));
  EXPECT_FALSE(all_true(true, 0, true));
  EXPECT_FALSE(all_true(true, true, false));
  EXPECT_TRUE(all_true(true, true, true, true));
  EXPECT_FALSE(all_true(false, true, true, true));
  EXPECT_FALSE(all_true(true, false, true, true));
  EXPECT_FALSE(all_true(true, true, false, true));
  EXPECT_FALSE(all_true(true, true, true, false));

  vector_index::index_service::index_config config{
      2, vector_index::metric_type::kEuclidean, backend_mode::kMemory,
      backend_provider::kHnswlib, ""};
  EXPECT_TRUE(can_rebuild_after_search_failure(config));

  config.mode = backend_mode::kExternal;
  EXPECT_FALSE(can_rebuild_after_search_failure(config));
  config.mode = backend_mode::kMemory;
  config.provider = backend_provider::kFaiss;
  EXPECT_FALSE(can_rebuild_after_search_failure(config));
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
  invalid_entries_rebuild.candidate_entries.emplace(
      1, vector_index::vector_data{1.0F});
  EXPECT_FALSE(service.build_commit_backends(&invalid_entries_plan));
}

TEST(VectorIndexServiceTest,
     SearchRuntimeSnapshotRejectsConfigurationGenerationChanges) {
  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings(
      "idx_search_snapshot", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(
      service.stage_upsert(1, "idx_search_snapshot", 11, {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(1));

  vector_index::index_service::search_runtime_snapshot snapshot;
  ASSERT_TRUE(service.snapshot_search_runtime_loaded(
      "idx_search_snapshot", 2, 1, 1, &snapshot));
  ASSERT_NE(nullptr, snapshot.runtime);
  EXPECT_EQ(1U, snapshot.candidate_top_k);
  EXPECT_TRUE(
      service.search_runtime_snapshot_matches("idx_search_snapshot", snapshot));

  std::vector<vector_index::search_result> candidates;
  {
    std::shared_lock<std::shared_mutex> guard(
        snapshot.runtime->runtime_mutex());
    ASSERT_TRUE(snapshot.runtime->search_for_rerank(
        {1.0F, 1.0F}, 1, snapshot.candidate_top_k, &candidates));
  }

  vector_index::index_service::index_publication_state changed =
      snapshot.publication;
  ++changed.config_generation;
  ASSERT_TRUE(
      service.restore_publication_state("idx_search_snapshot", changed));
  EXPECT_FALSE(
      service.search_runtime_snapshot_matches("idx_search_snapshot", snapshot));

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(service.finish_search_with_exact_rerank(
      "idx_search_snapshot", snapshot.config, {1.0F, 1.0F}, 1,
      std::move(candidates), &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(11U, results[0].doc_id);
}

TEST(VectorIndexServiceTest,
     StagedCommitPlanRestoresBackendAfterPartialMutationFailure) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kNative)) {
    GTEST_SKIP() << "native rollback backend unavailable in current build";
  }

  vector_index::index_service service;
  auto backend = std::make_unique<FailAfterMutationBackend>();
  FailAfterMutationBackend *backend_ptr = backend.get();
  ASSERT_TRUE(service.register_index("idx_partial_apply", std::move(backend)));
  ASSERT_TRUE(service.stage_upsert(750, "idx_partial_apply", 1,
                                   {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(750));

  backend_ptr->fail_after_successful_mutations(1);
  ASSERT_TRUE(service.stage_upsert(75, "idx_partial_apply", 2,
                                   {2.0F, 2.0F}));
  ASSERT_TRUE(service.stage_upsert(75, "idx_partial_apply", 3,
                                   {3.0F, 3.0F}));
  EXPECT_FALSE(service.commit(75));
  EXPECT_EQ(2U, service.pending_change_count(75));

  vector_index::committed_state state;
  ASSERT_TRUE(service.snapshot_committed_state(&state));
  ASSERT_EQ(1U, state["idx_partial_apply"].size());
  EXPECT_EQ(vector_index::vector_data({1.0F, 1.0F}),
            state["idx_partial_apply"][1]);

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(service.search("idx_partial_apply", {1.0F, 1.0F}, 10,
                             &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(1U, results[0].doc_id);
}

TEST(VectorIndexServiceTest,
     StagedCommitPlanRestoresBackendWhenEntryStoreApplyFails) {
#ifdef NDEBUG
  GTEST_SKIP() << "Debug failure injection requires a debug build";
#endif
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kNative)) {
    GTEST_SKIP() << "native rollback backend unavailable in current build";
  }

  vector_index::index_service service;
  ASSERT_TRUE(service.register_index_from_strings(
      "idx_entry_store_apply", 2, "euclidean", "memory", "native"));
  ASSERT_TRUE(service.stage_upsert(760, "idx_entry_store_apply", 1,
                                   {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(760));
  ASSERT_TRUE(service.stage_upsert(76, "idx_entry_store_apply", 1,
                                   {9.0F, 9.0F}));
  ASSERT_TRUE(service.stage_upsert(76, "idx_entry_store_apply", 2,
                                   {2.0F, 2.0F}));

  vector_index::index_service::commit_build_plan plan;
  ASSERT_TRUE(service.snapshot_commit_build_plan(76, &plan));
  ASSERT_TRUE(service.build_commit_backends(&plan));
  {
    ScopedDebugFlag fail_entry_store(
        "+d,vector_service_fail_commit_entry_store_apply");
    EXPECT_FALSE(service.apply_commit_build_plan(76, &plan));
  }
  EXPECT_EQ(2U, service.pending_change_count(76));

  vector_index::committed_state state;
  ASSERT_TRUE(service.snapshot_committed_state(&state));
  ASSERT_EQ(1U, state["idx_entry_store_apply"].size());
  EXPECT_EQ(vector_index::vector_data({1.0F, 1.0F}),
            state["idx_entry_store_apply"][1]);

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(service.search("idx_entry_store_apply", {1.0F, 1.0F}, 10,
                             &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(1U, results[0].doc_id);
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

  ASSERT_TRUE(service.register_index_from_strings("idx_stale_committed", 2,
                                                  "euclidean", "memory",
                                                  "native"));
  ASSERT_TRUE(service.stage_upsert(77, "idx_stale_committed", 1,
                                   {1.0F, 1.0F}));
  ASSERT_TRUE(service.snapshot_commit_build_plan(77, &plan));
  ASSERT_TRUE(service.stage_upsert(78, "idx_stale_committed", 2,
                                   {2.0F, 2.0F}));
  ASSERT_TRUE(service.commit(78));
  ASSERT_TRUE(service.build_commit_backends(&plan));
  EXPECT_FALSE(service.apply_commit_build_plan(77, &plan));
  EXPECT_EQ(1U, service.pending_change_count(77));
  service.rollback(77);

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
     StagedCommitDefersRuntimeWithoutInPlaceMutationSupport) {
  vector_index::index_service service;
  auto backend = std::make_unique<DeferredMutationBackend>();
  DeferredMutationBackend *runtime = backend.get();
  ASSERT_TRUE(service.register_index("idx_deferred_runtime", std::move(backend)));
  ASSERT_TRUE(service.stage_upsert(79, "idx_deferred_runtime", 7,
                                   {7.0F, 7.0F}));

  vector_index::index_service::commit_build_plan plan;
  ASSERT_TRUE(service.snapshot_commit_build_plan(79, &plan));
  ASSERT_EQ(1U, plan.indexes.size());
  EXPECT_TRUE(plan.indexes[0].defer_runtime_rebuild);
  EXPECT_TRUE(plan.rebuilds.empty());
  ASSERT_TRUE(service.build_commit_backends(&plan));
  ASSERT_TRUE(service.apply_commit_build_plan(79, &plan));
  EXPECT_EQ(0U, runtime->upsert_calls);
  EXPECT_EQ(0U, runtime->erase_calls);

  vector_index::index_service::index_publication_state publication;
  ASSERT_TRUE(service.describe_publication_state("idx_deferred_runtime",
                                                 &publication));
  EXPECT_EQ(1U, publication.truth_generation);
  EXPECT_EQ(0U, publication.runtime_generation);

  ASSERT_TRUE(service.stage_upsert(80, "idx_deferred_runtime", 8,
                                   {8.0F, 8.0F}));
  ASSERT_TRUE(service.snapshot_commit_build_plan(80, &plan));
  ASSERT_EQ(1U, plan.indexes.size());
  EXPECT_TRUE(plan.indexes[0].defer_runtime_rebuild);
  ASSERT_TRUE(service.build_commit_backends(&plan));
  ASSERT_TRUE(service.apply_commit_build_plan(80, &plan));
  EXPECT_EQ(0U, runtime->upsert_calls);
  EXPECT_EQ(0U, runtime->erase_calls);
  ASSERT_TRUE(service.describe_publication_state("idx_deferred_runtime",
                                                 &publication));
  EXPECT_EQ(2U, publication.truth_generation);
  EXPECT_EQ(0U, publication.runtime_generation);

  vector_index::index_service::committed_state committed;
  ASSERT_TRUE(service.snapshot_committed_state(&committed));
  ASSERT_EQ(1U, committed["idx_deferred_runtime"].count(7));
  ASSERT_EQ(1U, committed["idx_deferred_runtime"].count(8));
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
  const auto &stale_entries = stale_plan.rebuilds[0].candidate_entries;
  EXPECT_EQ(0U, stale_entries.count(1));
  EXPECT_EQ(1U, stale_entries.count(2));
  EXPECT_EQ(1U, stale_entries.count(3));
  EXPECT_EQ(0U, stale_entries.count(4));

  ASSERT_TRUE(service.stage_upsert(82, "idx_staged_faiss", 4,
                                   {4.0F, 4.0F}));
  ASSERT_TRUE(service.commit(82));
  EXPECT_EQ(0U, stale_entries.count(4));
  ASSERT_TRUE(service.build_commit_backends(&stale_plan));
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

TEST(VectorIndexServiceTest,
     StagedCommitPlanExternalFaissRejectsConfigurationChange) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_service_staged_faiss_config_t";
  std::filesystem::remove_all(root);
  faiss_snapshot_root_guard faiss_root(root);

  vector_index::index_service service;
  if (!service.register_index_from_strings("idx_staged_faiss_config", 2,
                                           "euclidean", "external", "faiss")) {
    GTEST_SKIP() << "FAISS external backend unavailable in current build";
  }
  ASSERT_TRUE(service.stage_upsert(830, "idx_staged_faiss_config", 1,
                                   {1.0F, 1.0F}));
  ASSERT_TRUE(service.commit(830));
  ASSERT_TRUE(service.stage_upsert(83, "idx_staged_faiss_config", 2,
                                   {2.0F, 2.0F}));

  vector_index::index_service::commit_build_plan plan;
  ASSERT_TRUE(service.snapshot_commit_build_plan(83, &plan));
  ASSERT_EQ(1U, plan.rebuilds.size());
  ASSERT_TRUE(service.build_commit_backends(&plan));

  vector_index::index_service::index_config changed_config;
  ASSERT_TRUE(service.describe_index("idx_staged_faiss_config",
                                     &changed_config, nullptr, nullptr,
                                     nullptr));
  changed_config.faiss_build_threads = 2;
  ASSERT_TRUE(
      service.restore_index_config("idx_staged_faiss_config", changed_config));
  EXPECT_FALSE(service.apply_commit_build_plan(83, &plan));
  EXPECT_EQ(1U, service.pending_change_count(83));

  std::filesystem::remove_all(root);
}

}  // namespace vector_index_service_unittest
