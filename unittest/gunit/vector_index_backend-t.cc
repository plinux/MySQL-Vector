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
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef HAVE_FAISS
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#endif

#include "my_io.h"
#include "sql/mysqld.h"
#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_index_backend_internal.h"
#include "sql/vector/vector_status.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_index_backend_unittest {

namespace {

bool has_prefix(const std::string &text, const std::string &prefix) {
  return text.size() >= prefix.size() &&
         text.compare(0, prefix.size(), prefix) == 0;
}

bool has_suffix(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

[[maybe_unused]] std::string find_snapshot_file(const std::string &directory,
                                                const std::string &prefix) {
  std::error_code ec;
  const std::filesystem::path dir(directory);
  if (!std::filesystem::exists(dir, ec) || ec) return "";

  for (const auto &entry : std::filesystem::directory_iterator(
           dir, std::filesystem::directory_options::skip_permission_denied, ec)) {
    if (ec) return "";
    if (!entry.is_regular_file(ec)) {
      if (ec) return "";
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (!has_prefix(filename, prefix) || !has_suffix(filename, ".v1")) {
      continue;
    }
    return entry.path().string();
  }
  return "";
}

[[maybe_unused]] std::string find_manifest_file(
    const std::string &root, const std::string &manifest_filename) {
  std::error_code ec;
  const std::filesystem::path dir(root);
  if (!std::filesystem::exists(dir, ec) || ec) return "";

  for (const auto &entry : std::filesystem::recursive_directory_iterator(
           dir, std::filesystem::directory_options::skip_permission_denied,
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

[[maybe_unused]] std::string find_faiss_snapshot_file(
    const std::string &directory) {
  return find_snapshot_file(directory, "faiss_external.snapshot.");
}

[[maybe_unused]] std::string find_diskann_snapshot_file(
    const std::string &directory) {
  return find_snapshot_file(directory, "diskann_external.snapshot.");
}

[[maybe_unused]] std::string find_faiss_manifest_file(
    const std::string &root) {
  return find_manifest_file(root, "faiss_external.manifest.v1");
}

[[maybe_unused]] std::string find_diskann_manifest_file(
    const std::string &root) {
  return find_manifest_file(root, "diskann_external.manifest.v1");
}

[[maybe_unused]] char hex_digit(unsigned value) {
  return static_cast<char>((value < 10U) ? ('0' + value)
                                         : ('a' + (value - 10U)));
}

[[maybe_unused]] std::string hex_encode(const std::string &input) {
  std::string out;
  out.resize(input.size() * 2);
  for (size_t i = 0; i < input.size(); ++i) {
    const unsigned value =
        static_cast<unsigned>(static_cast<unsigned char>(input[i]));
    out[i * 2] = hex_digit((value >> 4) & 0x0FU);
    out[i * 2 + 1] = hex_digit(value & 0x0FU);
  }
  return out;
}

[[maybe_unused]] std::string hex_encode_vector(
    const vector_index::vector_data &vector) {
  if (vector.empty()) return "";
  return hex_encode(std::string(reinterpret_cast<const char *>(vector.data()),
                                vector.size() * sizeof(float)));
}

[[maybe_unused]] void write_binary_file(const std::string &path,
                                        const std::string &payload) {
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << payload;
  file.close();
  ASSERT_TRUE(file);
}

#ifdef HAVE_FAISS
[[maybe_unused]] void write_faiss_index_file(const std::string &path,
                                             std::unique_ptr<faiss::Index> index) {
  ASSERT_NE(nullptr, index);
  faiss::write_index(index.get(), path.c_str());
}
#endif

bool hnswlib_tuning_supported() {
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  return backend.set_search_ef(64) &&
         backend.set_hnsw_build_params(16, 200);
}

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

class StubBackend final : public vector_index::backend {
 public:
  bool upsert(uint64_t doc_id [[maybe_unused]],
              const vector_index::vector_data &vector) override {
    ++upsert_calls;
    if (!allow_upsert) return false;
    last_vector = vector;
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
    return vector_index::backend_provider::kNative;
  }
  bool supports_mutations() const override { return m_supports_mutations; }

  bool m_supports_mutations{true};
  bool allow_upsert{true};
  size_t upsert_calls{0};
  vector_index::vector_data last_vector;
};

}  // namespace

TEST(VectorIndexBackendTest, BaseLoadCommittedEntriesUsesMutationContract) {
  StubBackend backend;
  std::unordered_map<uint64_t, vector_index::vector_data> entries{
      {1, {1.0F, 1.0F}}, {2, {2.0F, 2.0F}}};

  EXPECT_TRUE(backend.load_committed_entries(entries));
  EXPECT_EQ(2U, backend.upsert_calls);

  backend.allow_upsert = false;
  EXPECT_FALSE(backend.load_committed_entries(entries));

  backend.m_supports_mutations = false;
  EXPECT_TRUE(backend.load_committed_entries({}));
  EXPECT_FALSE(backend.load_committed_entries(entries));
}

TEST(VectorIndexBackendTest, BaseSearchBatchUsesSingleSearchContract) {
  StubBackend backend;
  std::vector<std::vector<vector_index::search_result>> results;

  EXPECT_FALSE(backend.search_batch({{1.0F, 1.0F}}, 1, nullptr));
  EXPECT_TRUE(backend.search_batch({}, 1, &results));
  EXPECT_TRUE(results.empty());
  ASSERT_TRUE(backend.search_batch({{1.0F, 1.0F}, {2.0F, 2.0F}}, 1,
                                   &results));
  ASSERT_EQ(2U, results.size());
  EXPECT_TRUE(results[0].empty());
  EXPECT_TRUE(results[1].empty());
}

TEST(VectorIndexBackendTest, BaseBackendDefaultsRejectUnsupportedTunings) {
  StubBackend backend;
  std::unordered_map<uint64_t, vector_index::vector_data> entries{
      {1, {1.0F, 1.0F}}};

  EXPECT_TRUE(backend.rebuild_from_committed_entries(entries));
  EXPECT_TRUE(backend.recover());
  EXPECT_TRUE(backend.recover_committed_entries(entries));
  EXPECT_FALSE(backend.last_recover_used_fallback());
  EXPECT_EQ(0U, backend.entry_count());
  EXPECT_TRUE(backend.backend_variant().empty());
  EXPECT_FALSE(backend.set_search_ef(64));
  EXPECT_EQ(0U, backend.search_ef());
  EXPECT_FALSE(backend.set_hnsw_build_params(16, 200));
  EXPECT_EQ(0U, backend.hnsw_m());
  EXPECT_EQ(0U, backend.hnsw_ef_construction());
  EXPECT_TRUE(backend.set_hnsw_build_threads(0));
  EXPECT_FALSE(backend.set_hnsw_build_threads(1));
  EXPECT_EQ(0U, backend.hnsw_build_threads());
  EXPECT_FALSE(backend.set_faiss_ivf_params(64, 8));
  EXPECT_EQ(0U, backend.faiss_nlist());
  EXPECT_EQ(0U, backend.faiss_nprobe());
  EXPECT_TRUE(backend.set_faiss_build_threads(0));
  EXPECT_FALSE(backend.set_faiss_build_threads(1));
  EXPECT_EQ(0U, backend.faiss_build_threads());
  EXPECT_FALSE(backend.set_faiss_ivf_pq_params(64, 8, 16, 8));
  EXPECT_EQ(0U, backend.faiss_pq_m());
  EXPECT_EQ(0U, backend.faiss_pq_bits());
  EXPECT_FALSE(backend.set_diskann_build_params(32, 64, 0));
  EXPECT_TRUE(backend.set_diskann_build_threads(0));
  EXPECT_FALSE(backend.set_diskann_build_threads(1));
  EXPECT_EQ(0U, backend.diskann_max_degree());
  EXPECT_EQ(0U, backend.diskann_build_complexity());
  EXPECT_EQ(0U, backend.diskann_build_threads());
  EXPECT_FALSE(backend.set_diskann_search_complexity(64));
  EXPECT_EQ(0U, backend.diskann_search_complexity());
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_EQ(0U, backend.external_manifest_generation());
}

TEST(VectorIndexBackendTest, MemoryBackendEuclideanTopK) {
  vector_index::memory_backend backend(2, vector_index::metric_type::kEuclidean);
  std::vector<vector_index::search_result> result;

  EXPECT_TRUE(backend.upsert(11, {0.0F, 0.0F}));
  EXPECT_TRUE(backend.upsert(22, {3.0F, 4.0F}));
  EXPECT_TRUE(backend.upsert(33, {1.0F, 1.0F}));

  EXPECT_TRUE(backend.search({0.9F, 0.9F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(33U, result[0].doc_id);
  EXPECT_EQ(11U, result[1].doc_id);
}

TEST(VectorIndexBackendTest, MemoryBackendEuclideanTieBreaksByDocId) {
  vector_index::memory_backend backend(2, vector_index::metric_type::kEuclidean);
  std::vector<vector_index::search_result> result;

  EXPECT_TRUE(backend.upsert(20, {1.0F, 0.0F}));
  EXPECT_TRUE(backend.upsert(10, {0.0F, 1.0F}));
  EXPECT_TRUE(backend.search({0.0F, 0.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);
  EXPECT_EQ(20U, result[1].doc_id);
}

TEST(VectorIndexBackendTest, MemoryBackendRejectsDimensionMismatch) {
  vector_index::memory_backend backend(2, vector_index::metric_type::kEuclidean);
  std::vector<vector_index::search_result> result;

  EXPECT_FALSE(backend.upsert(1, {1.0F, 2.0F, 3.0F}));
  EXPECT_TRUE(backend.upsert(1, {1.0F, 2.0F}));
  EXPECT_FALSE(backend.search({1.0F, 2.0F, 3.0F}, 1, &result));
}

TEST(VectorIndexBackendTest, MemoryBackendEraseAndZeroTopK) {
  vector_index::memory_backend backend(2, vector_index::metric_type::kEuclidean);
  std::vector<vector_index::search_result> result;

  EXPECT_TRUE(backend.upsert(1, {1.0F, 2.0F}));
  EXPECT_TRUE(backend.upsert(2, {2.0F, 3.0F}));
  EXPECT_TRUE(backend.erase(1));

  EXPECT_TRUE(backend.search({2.0F, 3.0F}, 0, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_TRUE(backend.search({2.0F, 3.0F}, 5, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
}

TEST(VectorIndexBackendTest, MemoryBackendCosineRejectsZeroNorm) {
  vector_index::memory_backend backend(2, vector_index::metric_type::kCosine);
  std::vector<vector_index::search_result> result;

  EXPECT_TRUE(backend.upsert(1, {0.0F, 0.0F}));
  EXPECT_FALSE(backend.search({1.0F, 0.0F}, 1, &result));
}

TEST(VectorIndexBackendTest, MemoryBackendInnerProductDistanceOrdering) {
  vector_index::memory_backend backend(2,
                                       vector_index::metric_type::kInnerProduct);
  std::vector<vector_index::search_result> result;

  EXPECT_TRUE(backend.upsert(1, {1.0F, 0.0F}));
  EXPECT_TRUE(backend.upsert(2, {2.0F, 0.0F}));
  EXPECT_TRUE(backend.search({1.0F, 0.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
  EXPECT_LT(result[0].distance, result[1].distance);
}

TEST(VectorIndexBackendTest, ExternalBackendSupportsMutableSearch) {
  vector_index::external_backend backend(2, vector_index::metric_type::kEuclidean);
  std::vector<vector_index::search_result> result;

  EXPECT_TRUE(backend.supports_mutations());
  EXPECT_TRUE(backend.upsert(1, {1.0F, 2.0F}));
  EXPECT_TRUE(backend.upsert(2, {2.0F, 2.0F}));
  EXPECT_EQ(2U, backend.entry_count());
  EXPECT_TRUE(backend.erase(2));
  EXPECT_EQ(1U, backend.entry_count());
  EXPECT_TRUE(backend.search({1.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexBackendTest, ExternalBackendEuclideanTieBreaksByDocId) {
  vector_index::external_backend backend(2, vector_index::metric_type::kEuclidean);
  std::vector<vector_index::search_result> result;

  EXPECT_TRUE(backend.upsert(20, {1.0F, 0.0F}));
  EXPECT_TRUE(backend.upsert(10, {0.0F, 1.0F}));
  EXPECT_TRUE(backend.search({0.0F, 0.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);
  EXPECT_EQ(20U, result[1].doc_id);
}

TEST(VectorIndexBackendTest, ExternalBackendRejectsBadDimensionsAndZeroTopK) {
  vector_index::external_backend backend(2, vector_index::metric_type::kEuclidean);
  std::vector<vector_index::search_result> result;

  EXPECT_FALSE(backend.upsert(1, {1.0F, 2.0F, 3.0F}));
  EXPECT_TRUE(backend.upsert(2, {2.0F, 2.0F}));
  EXPECT_TRUE(backend.search({2.0F, 2.0F}, 0, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_FALSE(backend.search({2.0F, 2.0F, 2.0F}, 1, &result));
}

TEST(VectorIndexBackendTest, CreateBackendRoutesNativeModes) {
  std::unique_ptr<vector_index::backend> memory = vector_index::create_backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kNative);
  ASSERT_NE(nullptr, memory);
  EXPECT_EQ(vector_index::backend_mode::kMemory, memory->mode());

  std::unique_ptr<vector_index::backend> external = vector_index::create_backend(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal, vector_index::backend_provider::kNative);
  ASSERT_NE(nullptr, external);
  EXPECT_EQ(vector_index::backend_mode::kExternal, external->mode());
}

TEST(VectorIndexBackendTest, OptionParsingAndStringConversionUseStableTokens) {
  vector_index::metric_type metric = vector_index::metric_type::kEuclidean;
  EXPECT_TRUE(vector_index::parse_metric(" L2 ", &metric));
  EXPECT_EQ(vector_index::metric_type::kEuclidean, metric);
  EXPECT_TRUE(vector_index::parse_metric("cosine", &metric));
  EXPECT_EQ(vector_index::metric_type::kCosine, metric);
  EXPECT_TRUE(vector_index::parse_metric("IP", &metric));
  EXPECT_EQ(vector_index::metric_type::kInnerProduct, metric);
  EXPECT_FALSE(vector_index::parse_metric("bad", &metric));
  EXPECT_FALSE(vector_index::parse_metric("l2", nullptr));

  vector_index::backend_mode mode = vector_index::backend_mode::kMemory;
  EXPECT_TRUE(vector_index::parse_backend_mode(" external ", &mode));
  EXPECT_EQ(vector_index::backend_mode::kExternal, mode);
  EXPECT_FALSE(vector_index::parse_backend_mode("bad", &mode));

  vector_index::backend_provider provider = vector_index::backend_provider::kNative;
  EXPECT_TRUE(vector_index::parse_backend_provider("hnsw", &provider));
  EXPECT_EQ(vector_index::backend_provider::kHnswlib, provider);
  EXPECT_TRUE(vector_index::parse_backend_provider("diskann", &provider));
  EXPECT_EQ(vector_index::backend_provider::kDiskAnn, provider);
  EXPECT_FALSE(vector_index::parse_backend_provider("bad", &provider));

  EXPECT_STREQ("euclidean",
               vector_index::metric_to_string(vector_index::metric_type::kEuclidean));
  EXPECT_STREQ("memory",
               vector_index::backend_mode_to_string(vector_index::backend_mode::kMemory));
  EXPECT_STREQ("hnswlib", vector_index::backend_provider_to_string(
                             vector_index::backend_provider::kHnswlib));
}

TEST(VectorIndexBackendTest, HnswlibMemoryFallbackBehavesLikeMemoryBackend) {
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  std::vector<vector_index::search_result> result;

  EXPECT_EQ(vector_index::backend_provider::kHnswlib, backend.provider());
  EXPECT_TRUE(backend.supports_mutations());
  EXPECT_TRUE(backend.upsert(1, {0.0F, 0.0F}));
  EXPECT_TRUE(backend.upsert(2, {2.0F, 2.0F}));
  EXPECT_FALSE(backend.search({0.1F, 0.1F}, 1, nullptr));
  EXPECT_TRUE(backend.search({0.1F, 0.1F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_TRUE(backend.erase(2));
}

TEST(VectorIndexBackendTest, HnswlibMemoryUpsertReplacementRebuildsGraph) {
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(backend.upsert(1, {0.0F, 0.0F}));
  ASSERT_TRUE(backend.upsert(1, {5.0F, 5.0F}));
  ASSERT_TRUE(backend.search({5.0F, 5.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_EQ(1U, backend.entry_count());
}

TEST(VectorIndexBackendTest, HnswlibMemorySearchEfCanBeConfigured) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);

  EXPECT_EQ(64U, backend.search_ef());
  EXPECT_TRUE(backend.set_search_ef(128));
  EXPECT_EQ(128U, backend.search_ef());
  EXPECT_FALSE(backend.set_search_ef(0));
}

TEST(VectorIndexBackendTest, HnswlibMemoryBuildParamsCanBeConfigured) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);

  EXPECT_EQ(16U, backend.hnsw_m());
  EXPECT_EQ(200U, backend.hnsw_ef_construction());
  EXPECT_TRUE(backend.set_hnsw_build_params(24, 320));
  EXPECT_EQ(24U, backend.hnsw_m());
  EXPECT_EQ(320U, backend.hnsw_ef_construction());
  EXPECT_FALSE(backend.set_hnsw_build_params(0, 320));
  EXPECT_FALSE(backend.set_hnsw_build_params(24, 0));
}

TEST(VectorIndexBackendTest, HnswlibMemoryBuildParamRebuildKeepsEntries) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(backend.upsert(10, {1.0F, 1.0F}));
  ASSERT_TRUE(backend.set_search_ef(32));
  ASSERT_TRUE(backend.set_hnsw_build_params(12, 96));
  EXPECT_EQ(12U, backend.hnsw_m());
  EXPECT_EQ(96U, backend.hnsw_ef_construction());
  EXPECT_EQ(32U, backend.search_ef());
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);
}

TEST(VectorIndexBackendTest,
     HnswlibMemoryRejectsDimensionMismatchAndEraseMissing) {
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  std::vector<vector_index::search_result> result;

  EXPECT_FALSE(backend.upsert(1, {1.0F, 2.0F, 3.0F}));
  EXPECT_TRUE(backend.erase(999));
  EXPECT_FALSE(backend.search({1.0F, 2.0F, 3.0F}, 1, &result));
}

TEST(VectorIndexBackendTest, HnswlibMemoryInnerProductSearchUsesBranch) {
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kInnerProduct,
                                       vector_index::backend_mode::kMemory);
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));
  ASSERT_TRUE(backend.upsert(2, {0.0F, 1.0F}));
  ASSERT_TRUE(backend.search({1.0F, 0.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexBackendTest, HnswlibMemoryCosineHandlesZeroNormAndZeroTopK) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kCosine,
                                       vector_index::backend_mode::kMemory);
  std::vector<vector_index::search_result> result{{99, 99.0}};

  ASSERT_TRUE(backend.upsert(1, {0.0F, 0.0F}));
  ASSERT_TRUE(backend.upsert(2, {1.0F, 0.0F}));
  EXPECT_TRUE(backend.search({1.0F, 0.0F}, 0, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_TRUE(backend.search({0.0F, 0.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
}

TEST(VectorIndexBackendTest, HnswlibExternalModeIsNonWritableWhenConstructed) {
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kExternal);
  std::vector<vector_index::search_result> result;

  EXPECT_FALSE(backend.supports_mutations());
  EXPECT_FALSE(backend.upsert(1, {1.0F, 2.0F}));
  EXPECT_FALSE(backend.erase(1));
  EXPECT_FALSE(backend.search({1.0F, 2.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
}

TEST(VectorIndexBackendTest, HnswlibExternalRejectsMemoryOnlyTunings) {
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kExternal);

  EXPECT_EQ(0U, backend.entry_count());
  EXPECT_EQ(0U, backend.search_ef());
  EXPECT_EQ(0U, backend.hnsw_m());
  EXPECT_EQ(0U, backend.hnsw_ef_construction());
  EXPECT_FALSE(backend.set_search_ef(64));
  EXPECT_FALSE(backend.set_hnsw_build_params(16, 200));
}

TEST(VectorIndexBackendTest,
     HnswlibExternalRejectsThreadAndCommittedStateMutation) {
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kExternal);

  EXPECT_FALSE(backend.set_hnsw_build_threads(1));
  EXPECT_EQ(0U, backend.hnsw_build_threads());
  EXPECT_TRUE(backend.rebuild_from_committed_entries({}));
  EXPECT_FALSE(backend.rebuild_from_committed_entries({{1, {1.0F, 0.0F}}}));
}

TEST(VectorIndexBackendTest,
     HnswlibMemoryRebuildRejectsWrongDimensionCommittedEntries) {
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));

  EXPECT_FALSE(backend.rebuild_from_committed_entries(
      {{2, vector_index::vector_data{1.0F, 0.0F, 3.0F}}}));
  EXPECT_EQ(1U, backend.entry_count());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexBackendTest, HnswlibMemoryResizesPastInitialCapacity) {
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  std::vector<vector_index::search_result> result;

  for (uint64_t doc_id = 0; doc_id < 1100; ++doc_id) {
    ASSERT_TRUE(
        backend.upsert(doc_id, {static_cast<float>(doc_id), 0.0F}));
  }

  EXPECT_EQ(1100U, backend.entry_count());
  ASSERT_TRUE(backend.search({1099.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1099U, result[0].doc_id);
}

TEST(VectorIndexBackendTest, HnswlibSearchBatchCoversGuardAndThreadPaths) {
  EnvVarGuard guard("MYSQL_VECTOR_HNSWLIB_BATCH_SEARCH_THREADS");
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));
  ASSERT_TRUE(backend.upsert(2, {0.0F, 1.0F}));
  ASSERT_TRUE(backend.upsert(3, {2.0F, 0.0F}));

  std::vector<std::vector<vector_index::search_result>> batch_results;
  EXPECT_FALSE(backend.search_batch({{1.0F, 0.0F}}, 1, nullptr));
  EXPECT_TRUE(backend.search_batch({}, 1, &batch_results));
  EXPECT_TRUE(batch_results.empty());
  EXPECT_FALSE(backend.search_batch({{1.0F, 0.0F, 0.0F}}, 1, &batch_results));

  unsetenv("MYSQL_VECTOR_HNSWLIB_BATCH_SEARCH_THREADS");
  ASSERT_TRUE(backend.search_batch({{1.0F, 0.0F}, {0.0F, 1.0F}}, 1,
                                   &batch_results));
  ASSERT_EQ(2U, batch_results.size());

  setenv("MYSQL_VECTOR_HNSWLIB_BATCH_SEARCH_THREADS", "", 1);
  ASSERT_TRUE(backend.search_batch({{1.0F, 0.0F}, {0.0F, 1.0F}}, 1,
                                   &batch_results));
  ASSERT_EQ(2U, batch_results.size());

  setenv("MYSQL_VECTOR_HNSWLIB_BATCH_SEARCH_THREADS", "0", 1);
  ASSERT_TRUE(backend.search_batch({{1.0F, 0.0F}, {0.0F, 1.0F}}, 1,
                                   &batch_results));
  ASSERT_EQ(2U, batch_results.size());

  setenv("MYSQL_VECTOR_HNSWLIB_BATCH_SEARCH_THREADS", "1", 1);
  ASSERT_TRUE(backend.search_batch({{1.0F, 0.0F}}, 1, &batch_results));
  ASSERT_EQ(1U, batch_results.size());
  ASSERT_EQ(1U, batch_results[0].size());
  EXPECT_EQ(1U, batch_results[0][0].doc_id);

  setenv("MYSQL_VECTOR_HNSWLIB_BATCH_SEARCH_THREADS", "not-a-number", 1);
  ASSERT_TRUE(backend.search_batch({{1.0F, 0.0F}, {0.0F, 1.0F}}, 1,
                                   &batch_results));
  ASSERT_EQ(2U, batch_results.size());
  EXPECT_EQ(1U, batch_results[0][0].doc_id);
  EXPECT_EQ(2U, batch_results[1][0].doc_id);

  setenv("MYSQL_VECTOR_HNSWLIB_BATCH_SEARCH_THREADS", "64", 1);
  ASSERT_TRUE(backend.search_batch({{1.0F, 0.0F}, {0.0F, 1.0F},
                                    {2.0F, 0.0F}, {1.0F, 1.0F}},
                                   2, &batch_results));
  ASSERT_EQ(4U, batch_results.size());
}

TEST(VectorIndexBackendTest,
     HnswlibParallelRebuildUsesBuildThreadConfiguration) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.set_hnsw_build_threads(2));
  EXPECT_EQ(2U, backend.hnsw_build_threads());
  EXPECT_FALSE(backend.set_hnsw_build_threads(65536));
  std::unordered_map<uint64_t, vector_index::vector_data> entries;
  for (uint64_t doc_id = 1; doc_id <= 16; ++doc_id) {
    entries.emplace(doc_id, vector_index::vector_data{
                                static_cast<float>(doc_id), 0.0F});
  }

  ASSERT_TRUE(backend.set_search_ef(32));
  ASSERT_TRUE(backend.rebuild_from_committed_entries(entries));
  EXPECT_EQ(entries.size(), backend.entry_count());
  EXPECT_EQ(32U, backend.search_ef());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({16.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(16U, result[0].doc_id);

  ASSERT_TRUE(backend.rebuild_from_committed_entries(entries));
  ASSERT_TRUE(backend.set_hnsw_build_threads(0));
  ASSERT_TRUE(backend.rebuild_from_committed_entries(entries));
}

TEST(VectorIndexBackendTest,
     HnswlibParallelRebuildFailureKeepsExistingServingState) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.set_hnsw_build_threads(2));
  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));

  std::unordered_map<uint64_t, vector_index::vector_data> entries{
      {1, {1.0F, 0.0F}}, {2, {2.0F, 0.0F}}, {3, {3.0F, 0.0F}}};
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_backend_fail_hnswlib_parallel_rebuild_worker");
    EXPECT_FALSE(backend.rebuild_from_committed_entries(entries));
  }

  EXPECT_EQ(1U, backend.entry_count());
  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexBackendTest, HnswlibSearchBatchExternalModeClearsResults) {
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kExternal);
  std::vector<std::vector<vector_index::search_result>> batch_results{
      {{99, 99.0}}};

  EXPECT_FALSE(backend.search_batch({{1.0F, 0.0F}}, 1, &batch_results));
  EXPECT_TRUE(batch_results.empty());
}

}  // namespace vector_index_backend_unittest
