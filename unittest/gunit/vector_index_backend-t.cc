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
#include <iterator>
#include <limits>
#include <stdexcept>
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
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_backend_common.h"
#include "sql/vector/vector_index_backend_internal.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_runtime_config.h"
#include "sql/vector/vector_index_runtime_thread_pool.h"
#include "sql/vector/vector_index_status_fields.h"
#include "sql/vector/vector_status.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_index_backend_unittest {

namespace {

using vector_gunit::ScopedTempDirectory;

bool has_prefix(const std::string &text, const std::string &prefix) {
  return text.size() >= prefix.size() &&
         text.compare(0, prefix.size(), prefix) == 0;
}

bool has_suffix(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const vector_index_status_fields::field_value *find_status_field(
    const vector_index_status_fields::field_values &fields,
    const char *name) {
  for (const vector_index_status_fields::field_value &field : fields) {
    if (std::strcmp(field.name, name) == 0) return &field;
  }
  return nullptr;
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

template <typename T>
void write_binary_value(std::ofstream *file, T value) {
  file->write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_raw_fbin_file(const std::string &path, uint32_t rows,
                         uint32_t dimension,
                         const std::vector<float> &values) {
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  write_binary_value(&file, rows);
  write_binary_value(&file, dimension);
  file.write(reinterpret_cast<const char *>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  file.close();
  ASSERT_TRUE(file);
}

void write_raw_docid_file(const std::string &path,
                          const std::vector<uint64_t> &doc_ids) {
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  write_binary_value<uint64_t>(&file, doc_ids.size());
  file.write(reinterpret_cast<const char *>(doc_ids.data()),
             static_cast<std::streamsize>(doc_ids.size() * sizeof(uint64_t)));
  file.close();
  ASSERT_TRUE(file);
}

void install_diskann_offline_test_adapter() {
#ifdef MYSQL_VECTOR_DISKANN_OFFLINE_TEST_LIB
  vector_index::diskann_reset_offline_adapter_path_for_testing();
  vector_index::diskann_set_offline_adapter_path_for_testing(
      MYSQL_VECTOR_DISKANN_OFFLINE_TEST_LIB);
#endif
}

struct diskann_offline_test_adapter_installer {
  diskann_offline_test_adapter_installer() {
    install_diskann_offline_test_adapter();
  }
};

[[maybe_unused]] diskann_offline_test_adapter_installer
    install_diskann_offline_test_adapter_before_tests;

#ifdef HAVE_FAISS
void write_faiss_index_file(const std::string &path,
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

class StubBackend final : public vector_index::backend {
 public:
  bool upsert(uint64_t doc_id [[maybe_unused]],
              const vector_index::vector_data &vector) override {
    ++upsert_calls;
    if (!allow_upsert) return false;
    upserted_entries[doc_id] = vector;
    last_vector = vector;
    return true;
  }

  bool erase(uint64_t doc_id [[maybe_unused]]) override { return true; }

  bool search(const vector_index::vector_data &query [[maybe_unused]],
              size_t top_k,
              std::vector<vector_index::search_result> *results) const override {
    ++search_calls;
    if (fail_search_call != 0 && search_calls == fail_search_call) return false;
    results->clear();
    if (top_k != 0) {
      results->push_back({search_calls, query.empty() ? 0.0 : query.front()});
    }
    return true;
  }

  bool recover() override { return allow_recover; }

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
  bool allow_recover{true};
  size_t upsert_calls{0};
  mutable size_t search_calls{0};
  size_t fail_search_call{0};
  std::unordered_map<uint64_t, vector_index::vector_data> upserted_entries;
  vector_index::vector_data last_vector;
};

}  // namespace

TEST(VectorIndexRuntimeConfigTest, ValidatesFaissIvfPqLibraryBoundaries) {
  EXPECT_TRUE(vector_index::valid_faiss_ivf_pq_config(16, 8, 4, 4, 8));
  EXPECT_TRUE(vector_index::valid_faiss_ivf_pq_config(
      24, 8, 4, 6, vector_index::k_max_faiss_pq_bits));

  EXPECT_FALSE(vector_index::valid_faiss_ivf_pq_config(0, 8, 4, 4, 8));
  EXPECT_FALSE(vector_index::valid_faiss_ivf_pq_config(16, 0, 4, 4, 8));
  EXPECT_FALSE(vector_index::valid_faiss_ivf_pq_config(16, 8, 0, 4, 8));
  EXPECT_FALSE(vector_index::valid_faiss_ivf_pq_config(16, 8, 4, 0, 8));
  EXPECT_FALSE(vector_index::valid_faiss_ivf_pq_config(16, 8, 4, 4, 0));
  EXPECT_FALSE(vector_index::valid_faiss_ivf_pq_config(15, 8, 4, 4, 8));
  EXPECT_FALSE(vector_index::valid_faiss_ivf_pq_config(
      16, 8, 4, 4, vector_index::k_max_faiss_pq_bits + 1));
}

TEST(VectorIndexBackendTest, BackendHelperParseUint64RejectsMalformedInputs) {
  uint64_t value = 0;
  EXPECT_FALSE(vector_index::parse_uint64_for_testing("", &value));
  EXPECT_FALSE(vector_index::parse_uint64_for_testing("abc", &value));
  EXPECT_FALSE(vector_index::parse_uint64_for_testing("12x", &value));
  EXPECT_FALSE(vector_index::parse_uint64_for_testing("18446744073709551616",
                                                   &value));
  EXPECT_FALSE(vector_index::parse_uint64_for_testing("7", nullptr));
  EXPECT_TRUE(vector_index::parse_uint64_for_testing("42", &value));
  EXPECT_EQ(42U, value);
}

TEST(VectorIndexBackendTest, BackendDefaultMethodsCoverReaderAndTuningGuards) {
  StubBackend backend;
  std::vector<std::vector<vector_index::search_result>> batch_results;

  backend.m_supports_mutations = false;
  EXPECT_TRUE(backend.load_committed_entries({}));
  EXPECT_FALSE(backend.load_committed_entries({{1, {1.0F, 1.0F}}}));
  EXPECT_TRUE(backend.recover());
  EXPECT_FALSE(backend.recover_committed_entries({{1, {1.0F, 1.0F}}}));

  backend.m_supports_mutations = true;
  backend.allow_upsert = false;
  EXPECT_FALSE(backend.load_committed_entries({{1, {1.0F, 1.0F}}}));
  EXPECT_EQ(1U, backend.upsert_calls);

  EXPECT_FALSE(backend.load_committed_entries_from_reader(nullptr));
  EXPECT_FALSE(backend.rebuild_from_committed_entries_from_reader(nullptr));
  EXPECT_FALSE(backend.recover_committed_entries_from_reader(nullptr));
  const vector_index::committed_entry_reader failing_reader =
      [](const vector_index::committed_entry_visitor &) { return false; };
  EXPECT_FALSE(backend.load_committed_entries_from_reader(failing_reader));

  backend.allow_upsert = true;
  const vector_index::committed_entry_reader reader =
      [](const vector_index::committed_entry_visitor &visitor) {
        return visitor(2, {2.0F, 2.0F});
      };
  EXPECT_TRUE(backend.load_committed_entries_from_reader(reader));
  EXPECT_TRUE(backend.rebuild_from_committed_entries_from_reader(reader));
  EXPECT_TRUE(backend.recover_committed_entries_from_reader(reader));
  EXPECT_EQ(1U, backend.upserted_entries.count(2));

  backend.allow_recover = false;
  EXPECT_FALSE(backend.recover_committed_entries({}));
  backend.allow_recover = true;

  EXPECT_FALSE(backend.search_batch({{1.0F, 1.0F}}, 1, nullptr));
  EXPECT_TRUE(backend.search_batch({}, 1, &batch_results));
  EXPECT_TRUE(batch_results.empty());
  EXPECT_TRUE(backend.search_batch({{1.0F, 1.0F}, {2.0F, 2.0F}}, 1,
                                   &batch_results));
  EXPECT_EQ(2U, batch_results.size());
  ASSERT_EQ(1U, batch_results[0].size());
  EXPECT_EQ(1U, batch_results[0][0].doc_id);
  backend.search_calls = 0;
  backend.fail_search_call = 2;
  EXPECT_FALSE(backend.search_batch({{1.0F, 1.0F}, {2.0F, 2.0F}}, 1,
                                    &batch_results));
  EXPECT_EQ(2U, backend.search_calls);
  backend.fail_search_call = 0;

  const std::string root =
      std::string(testing::TempDir()) + "/backend_default_raw_segments_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  const std::string vector_path = root + "/source.fbin";
  const std::string docid_path = root + "/source.u64";
  write_raw_fbin_file(vector_path, 2, 2, {5.0F, 0.0F, 6.0F, 0.0F});
  write_raw_docid_file(docid_path, {5, 6});

  vector_index::raw_vector_segment segment;
  segment.vector_path = vector_path;
  segment.docid_path = docid_path;
  segment.row_count = 2;
  segment.dimension = 2;
  segment.bytes = std::filesystem::file_size(vector_path, ec) +
                  std::filesystem::file_size(docid_path, ec);
  ASSERT_FALSE(ec);
  EXPECT_FALSE(backend.rebuild_from_raw_segments(nullptr));
  EXPECT_FALSE(backend.rebuild_from_raw_segments(
      [](const vector_index::raw_vector_segment_visitor &) { return false; }));
  EXPECT_FALSE(backend.rebuild_from_raw_segments(
      [&segment](const vector_index::raw_vector_segment_visitor &visitor) {
        vector_index::raw_vector_segment bad_segment = segment;
        bad_segment.dimension = 3;
        return visitor(bad_segment);
      }));
  EXPECT_FALSE(backend.rebuild_from_raw_segments(
      [&segment](const vector_index::raw_vector_segment_visitor &visitor) {
        vector_index::raw_vector_segment bad_segment = segment;
        bad_segment.row_count = 3;
        return visitor(bad_segment);
      }));
  backend.upserted_entries.clear();
  EXPECT_TRUE(backend.rebuild_from_raw_segments(
      [&segment](const vector_index::raw_vector_segment_visitor &visitor) {
        return visitor(segment);
      }));
  ASSERT_EQ(2U, backend.upserted_entries.size());
  EXPECT_EQ((vector_index::vector_data{5.0F, 0.0F}),
            backend.upserted_entries.at(5));
  EXPECT_EQ((vector_index::vector_data{6.0F, 0.0F}),
            backend.upserted_entries.at(6));

  vector_index::segment_build_input segment_input;
  segment_input.segment = segment;
  segment_input.segment_id = 7;
  segment_input.generation = 9;
  segment_input.artifact_prefix = root + "/segment-7";
  vector_index::segment_build_result segment_result;
  EXPECT_FALSE(backend.build_segment_from_raw(segment_input, nullptr));
  vector_index::segment_build_input invalid_segment_input = segment_input;
  invalid_segment_input.segment_id = 0;
  EXPECT_FALSE(
      backend.build_segment_from_raw(invalid_segment_input, &segment_result));

  backend.upserted_entries.clear();
  EXPECT_TRUE(backend.build_segment_from_raw(segment_input, &segment_result));
  EXPECT_EQ(7U, segment_result.segment_id);
  EXPECT_EQ(9U, segment_result.generation);
  EXPECT_EQ(2U, segment_result.row_count);
  EXPECT_EQ(segment.bytes, segment_result.payload_size);
  EXPECT_EQ(vector_path, segment_result.vector_path);
  EXPECT_EQ(docid_path, segment_result.docid_path);
  EXPECT_EQ(root + "/segment-7", segment_result.artifact_prefix);
  EXPECT_TRUE(segment_result.ready);
  EXPECT_TRUE(backend.load_segment_handle(segment_result));
  segment_result.ready = false;
  EXPECT_FALSE(backend.load_segment_handle(segment_result));
  std::filesystem::remove_all(root, ec);

  EXPECT_FALSE(backend.last_recover_used_fallback());
  EXPECT_EQ(0U, backend.entry_count());
  EXPECT_TRUE(backend.backend_variant().empty());
  EXPECT_FALSE(backend.set_search_ef(1));
  EXPECT_EQ(0U, backend.search_ef());
  EXPECT_FALSE(backend.set_hnsw_build_params(16, 200));
  EXPECT_EQ(0U, backend.hnsw_m());
  EXPECT_EQ(0U, backend.hnsw_ef_construction());
  EXPECT_TRUE(backend.set_hnsw_build_threads(0));
  EXPECT_FALSE(backend.set_hnsw_build_threads(1));
  EXPECT_FALSE(backend.set_faiss_ivf_params(1, 1));
  EXPECT_TRUE(backend.set_faiss_build_threads(0));
  EXPECT_FALSE(backend.set_faiss_build_threads(1));
  EXPECT_FALSE(backend.set_faiss_ivf_pq_params(1, 1, 1, 1));
  EXPECT_FALSE(backend.set_diskann_build_params(32, 64, 0));
  EXPECT_TRUE(backend.set_diskann_build_threads(0));
  EXPECT_FALSE(backend.set_diskann_build_threads(1));
  EXPECT_TRUE(
      backend.set_diskann_build_mode(vector_index::diskann_build_mode::kAuto));
  EXPECT_FALSE(
      backend.set_diskann_build_mode(vector_index::diskann_build_mode::kSerial));
  EXPECT_FALSE(backend.set_diskann_search_complexity(64));
  EXPECT_FALSE(backend.set_diskann_search_beamwidth(16));
  EXPECT_EQ(0U, backend.diskann_offline_search_threads());
  EXPECT_EQ(0U, backend.diskann_search_io_limit());
  EXPECT_EQ(0U, backend.diskann_cache_nodes());
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_EQ(0U, backend.external_manifest_generation());
}

TEST(VectorIndexBackendTest,
     BackendBuildDiagnosticsAppearInInfoAndBackendHealthFields) {
  vector_index_registry::index_info info;
  info.provider = "diskann";
  info.mode = "external";
  info.consistency_mode = "standalone";
  info.lifecycle_state = "ready";
  info.supports_mutations = true;
  info.build_diagnostics.runtime = "official_cpp_main";
  info.build_diagnostics.input_source = "raw_segments";
  info.build_diagnostics.row_count = 42;
  info.build_diagnostics.segment_count = 3;
  info.build_diagnostics.manifest_ms = 11;
  info.build_diagnostics.offline_build_ms = 22;
  info.build_diagnostics.load_ms = 33;
  info.build_diagnostics.fallback_reason = "offline_unavailable";

  vector_index_status_fields::field_values fields;
  vector_index_status_fields::collect_info_fields(info, &fields);

  const auto *runtime = find_status_field(fields, "backend_build_runtime");
  ASSERT_NE(nullptr, runtime);
  EXPECT_EQ("official_cpp_main", runtime->string_value);
  const auto *source =
      find_status_field(fields, "backend_build_input_source");
  ASSERT_NE(nullptr, source);
  EXPECT_EQ("raw_segments", source->string_value);
  const auto *rows = find_status_field(fields, "backend_build_rows");
  ASSERT_NE(nullptr, rows);
  EXPECT_EQ(42U, rows->uint_value);
  const auto *segments =
      find_status_field(fields, "backend_build_segments");
  ASSERT_NE(nullptr, segments);
  EXPECT_EQ(3U, segments->uint_value);
  const auto *offline_ms =
      find_status_field(fields, "backend_build_offline_ms");
  ASSERT_NE(nullptr, offline_ms);
  EXPECT_EQ("22", vector_index_status_fields::status_value(*offline_ms));

  vector_index_status_fields::collect_backend_health_fields(info, &fields);
  const auto *fallback =
      find_status_field(fields, "backend_build_fallback_reason");
  ASSERT_NE(nullptr, fallback);
  EXPECT_EQ("offline_unavailable", fallback->string_value);
  const auto *load_ms = find_status_field(fields, "backend_build_load_ms");
  ASSERT_NE(nullptr, load_ms);
  EXPECT_EQ(33U, load_ms->uint_value);
}

TEST(VectorIndexBackendTest,
     BackendStatusFieldsCoverEmptyDiagnosticsAndLifecycleBranches) {
  vector_index_registry::index_info info;
  info.provider = "diskann";
  info.mode = "external";
  info.consistency_mode = "transactional";
  info.supports_mutations = true;
  info.entry_count = 12;
  info.committed_entry_count = 5;

  vector_index_status_fields::field_values fields;
  vector_index_status_fields::collect_info_fields(info, &fields);
  EXPECT_EQ(nullptr, find_status_field(fields, "backend_build_runtime"));
  const auto *pending = find_status_field(fields, "pending_apply_count");
  ASSERT_NE(nullptr, pending);
  EXPECT_EQ(7U, pending->uint_value);

  info.lifecycle_state = "failed";
  vector_index_status_fields::collect_backend_health_fields(info, &fields);
  const auto *loaded = find_status_field(fields, "loaded");
  ASSERT_NE(nullptr, loaded);
  EXPECT_FALSE(loaded->bool_value);
  const auto *writable = find_status_field(fields, "writable");
  ASSERT_NE(nullptr, writable);
  EXPECT_FALSE(writable->bool_value);

  info.lifecycle_state = "rebuilding";
  vector_index_status_fields::collect_sync_pipeline_fields(info, &fields);
  const auto *rebuild_progress = find_status_field(fields, "rebuild_progress");
  ASSERT_NE(nullptr, rebuild_progress);
  EXPECT_EQ(50U, rebuild_progress->uint_value);
  const auto *recover_progress = find_status_field(fields, "recover_progress");
  ASSERT_NE(nullptr, recover_progress);
  EXPECT_EQ(0U, recover_progress->uint_value);

  info.lifecycle_state = "recovering";
  vector_index_status_fields::collect_sync_pipeline_fields(info, &fields);
  recover_progress = find_status_field(fields, "recover_progress");
  ASSERT_NE(nullptr, recover_progress);
  EXPECT_EQ(50U, recover_progress->uint_value);

  vector_index_status_fields::collect_info_fields(info, nullptr);
  vector_index_status_fields::collect_index_state_fields(info, nullptr);
  vector_index_status_fields::collect_backend_health_fields(info, nullptr);
  vector_index_status_fields::collect_sync_pipeline_fields(info, nullptr);
}

TEST(VectorIndexBackendTest, BackendPublicHelpersCoverEntryCountAndParserEdges) {
  vector_index::memory_backend memory_backend(
      2, vector_index::metric_type::kEuclidean);
  std::unordered_map<uint64_t, vector_index::vector_data> entries;
  EXPECT_FALSE(memory_backend.snapshot_entries(nullptr));
  ASSERT_TRUE(memory_backend.upsert(1, {1.0F, 0.0F}));
  ASSERT_TRUE(memory_backend.snapshot_entries(&entries));
  EXPECT_EQ((vector_index::vector_data{1.0F, 0.0F}), entries.at(1));

  vector_index::external_backend external_backend(
      2, vector_index::metric_type::kEuclidean);
  ASSERT_TRUE(external_backend.upsert(1, {1.0F, 0.0F}));
  ASSERT_TRUE(external_backend.upsert(2, {0.0F, 1.0F}));
  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(external_backend.search({1.0F, 0.0F}, 2, &results));
  ASSERT_EQ(2U, results.size());
  EXPECT_EQ(1U, results[0].doc_id);

  vector_index::index_consistency_mode consistency_mode;
  ASSERT_TRUE(vector_index::parse_index_consistency_mode("non-transactional",
                                                         &consistency_mode));
  EXPECT_EQ(vector_index::index_consistency_mode::kStandalone,
            consistency_mode);
  EXPECT_FALSE(vector_index::parse_index_consistency_mode("standalone",
                                                          nullptr));

#ifdef HAVE_FAISS
  vector_index::faiss_backend faiss_memory(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kMemory);
  ASSERT_TRUE(faiss_memory.upsert(10, {1.0F, 0.0F}));
  EXPECT_EQ(1U, faiss_memory.entry_count());

  vector_index::faiss_backend faiss_non_faiss_sidecar(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal, "idx_non_faiss_sidecar",
      vector_index::external_sidecar_profile::kDiskAnn);
  EXPECT_EQ(0U, faiss_non_faiss_sidecar.entry_count());
#endif
}

TEST(VectorIndexBackendTest,
     BackendHelperLoadManifestGenerationHandlesMissingAndMalformedFiles) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_backend_manifest_helper_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);

  const std::string manifest_path = root + "/manifest.v1";
  uint64_t generation = 99;
  bool exists = true;

  EXPECT_TRUE(vector_index::load_external_manifest_generation_for_testing(
      manifest_path, "header", &generation, &exists));
  EXPECT_FALSE(exists);
  EXPECT_EQ(0U, generation);

  ASSERT_TRUE(std::filesystem::create_directory(manifest_path, ec));
  EXPECT_FALSE(vector_index::load_external_manifest_generation_for_testing(
      manifest_path, "header", &generation, &exists));
  ASSERT_TRUE(std::filesystem::remove(manifest_path, ec));

  write_binary_file(manifest_path, "wrong\n1\n");
  EXPECT_FALSE(vector_index::load_external_manifest_generation_for_testing(
      manifest_path, "header", &generation, &exists));

  write_binary_file(manifest_path, "header\n");
  EXPECT_FALSE(vector_index::load_external_manifest_generation_for_testing(
      manifest_path, "header", &generation, &exists));

  write_binary_file(manifest_path, "header\n0\n");
  EXPECT_FALSE(vector_index::load_external_manifest_generation_for_testing(
      manifest_path, "header", &generation, &exists));

  write_binary_file(manifest_path, "header\nabc\n");
  EXPECT_FALSE(vector_index::load_external_manifest_generation_for_testing(
      manifest_path, "header", &generation, &exists));

  write_binary_file(manifest_path, "header\n18446744073709551616\n");
  EXPECT_FALSE(vector_index::load_external_manifest_generation_for_testing(
      manifest_path, "header", &generation, &exists));

  write_binary_file(manifest_path, "header\n7\n");
  EXPECT_TRUE(vector_index::load_external_manifest_generation_for_testing(
      manifest_path, "header", &generation, &exists));
  EXPECT_TRUE(exists);
  EXPECT_EQ(7U, generation);

  EXPECT_FALSE(vector_index::load_external_manifest_generation_for_testing(
      manifest_path, nullptr, &generation, &exists));
  EXPECT_FALSE(vector_index::load_external_manifest_generation_for_testing(
      manifest_path, "header", nullptr, &exists));
  EXPECT_FALSE(vector_index::load_external_manifest_generation_for_testing(
      manifest_path, "header", &generation, nullptr));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     BackendHelperSaveManifestGenerationRejectsInvalidArgsAndRenameFailure) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_backend_save_manifest_helper_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);

  const std::string manifest_path = root + "/nested/manifest.v1";
  EXPECT_FALSE(vector_index::save_external_manifest_generation_for_testing(
      manifest_path, "header", 0));
  EXPECT_FALSE(vector_index::save_external_manifest_generation_for_testing(
      manifest_path, nullptr, 1));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
        "+d,vector_backend_fail_save_manifest_generation_rename");
    EXPECT_FALSE(vector_index::save_external_manifest_generation_for_testing(
        manifest_path, "header", 1));
  }

  EXPECT_TRUE(vector_index::save_external_manifest_generation_for_testing(
      manifest_path, "header", 2));
  uint64_t generation = 0;
  bool exists = false;
  EXPECT_TRUE(vector_index::load_external_manifest_generation_for_testing(
      manifest_path, "header", &generation, &exists));
  EXPECT_TRUE(exists);
  EXPECT_EQ(2U, generation);

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     BackendHelperQuarantineAndCleanupWrappersHandleFilesystemShapes) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_backend_cleanup_helper_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root + "/snapshots", ec);
  ASSERT_FALSE(ec);

  const std::string artifact_path = root + "/artifact.v1";
  write_binary_file(artifact_path, "payload");
  EXPECT_TRUE(vector_index::quarantine_file_if_exists_for_testing(artifact_path));
  EXPECT_FALSE(std::filesystem::exists(artifact_path));
  EXPECT_TRUE(std::filesystem::exists(artifact_path + ".corrupt"));
  EXPECT_TRUE(
      vector_index::quarantine_file_if_exists_for_testing(root + "/missing.v1"));

  const std::string snapshots_dir = root + "/snapshots";
  write_binary_file(snapshots_dir + "/faiss_external.snapshot.1.v1", "one");
  write_binary_file(snapshots_dir + "/faiss_external.snapshot.2.v1", "two");
  write_binary_file(snapshots_dir + "/faiss_external.snapshot.no_suffix",
                    "keep-prefix");
  write_binary_file(snapshots_dir + "/other_file.txt", "keep");
  EXPECT_TRUE(vector_index::remove_generated_snapshots_with_prefix_for_testing(
      snapshots_dir, "faiss_external.snapshot."));
  EXPECT_FALSE(std::filesystem::exists(
      snapshots_dir + "/faiss_external.snapshot.1.v1"));
  EXPECT_FALSE(std::filesystem::exists(
      snapshots_dir + "/faiss_external.snapshot.2.v1"));
  EXPECT_TRUE(std::filesystem::exists(
      snapshots_dir + "/faiss_external.snapshot.no_suffix"));
  EXPECT_TRUE(std::filesystem::exists(snapshots_dir + "/other_file.txt"));

  EXPECT_TRUE(vector_index::remove_dir_if_empty_for_testing(snapshots_dir));
  std::filesystem::remove(snapshots_dir + "/faiss_external.snapshot.no_suffix",
                          ec);
  ASSERT_FALSE(ec);
  std::filesystem::remove(snapshots_dir + "/other_file.txt", ec);
  ASSERT_FALSE(ec);
  EXPECT_TRUE(vector_index::remove_dir_if_empty_for_testing(snapshots_dir));
  EXPECT_TRUE(vector_index::remove_dir_if_empty_for_testing(root + "/missing_dir"));

  const std::string not_dir = root + "/plain_file";
  write_binary_file(not_dir, "x");
  EXPECT_TRUE(vector_index::remove_dir_if_empty_for_testing(not_dir));

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     BackendHelperExternalProfileAndCleanupEdges) {
  namespace detail = vector_index::detail;

  EXPECT_STREQ(detail::kDiskAnnExternalSnapshotHeader,
               detail::external_snapshot_header(
                   vector_index::external_sidecar_profile::kDiskAnn));
  EXPECT_STREQ(detail::kDiskAnnExternalSnapshotPrefix,
               detail::external_snapshot_prefix(
                   vector_index::external_sidecar_profile::kDiskAnn));

  DataHomeGuard data_home_guard;
  vector_index::reset_faiss_external_snapshot_root_for_testing();
  data_home_guard.Set("");
  EXPECT_EQ("", detail::external_snapshot_directory("idx_empty_root"));
  EXPECT_EQ("", detail::external_manifest_path(
                    "idx_empty_root",
                    vector_index::external_sidecar_profile::kDiskAnn));

  data_home_guard.Set("root\\");
  EXPECT_NE(std::string::npos,
            detail::external_snapshot_directory("idx").find("696478"));
  EXPECT_EQ("", detail::external_snapshot_directory(""));

  EXPECT_TRUE(detail::quarantine_file_if_exists(""));
  EXPECT_TRUE(detail::remove_external_sidecar_artifacts(
      "", vector_index::external_sidecar_profile::kDiskAnn, true));
  EXPECT_TRUE(detail::remove_diskann_external_artifacts("", false));

  const std::string root =
      std::string(testing::TempDir()) + "/vector_backend_cleanup_edges_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);

  const std::string plain_file = root + "/plain_file";
  write_binary_file(plain_file, "x");
  EXPECT_TRUE(detail::remove_generated_snapshots_with_prefix(plain_file,
                                                            "prefix."));

  vector_index::set_faiss_external_snapshot_root_for_testing(root);
  const std::string manifest_quarantine_path = detail::external_manifest_path(
      "idx_cleanup_quarantine",
      vector_index::external_sidecar_profile::kFaiss) + ".corrupt";
  std::filesystem::create_directories(manifest_quarantine_path + "/child", ec);
  ASSERT_FALSE(ec);
  EXPECT_FALSE(detail::remove_external_sidecar_artifacts(
      "idx_cleanup_quarantine",
      vector_index::external_sidecar_profile::kFaiss, false));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
        "+d,vector_backend_fail_remove_dir_if_empty");
    EXPECT_FALSE(detail::remove_external_sidecar_artifacts(
        "idx_cleanup_edges", vector_index::external_sidecar_profile::kFaiss,
        true));
  }
  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     BackendHelperDiskAnnMetricAndDocIdParsersCoverEdgeCases) {
  EXPECT_EQ(0, vector_index::diskann_metric_code_for_testing(
                   vector_index::metric_type::kCosine));
  EXPECT_EQ(1, vector_index::diskann_metric_code_for_testing(
                   vector_index::metric_type::kInnerProduct));
  EXPECT_EQ(2, vector_index::diskann_metric_code_for_testing(
                   vector_index::metric_type::kEuclidean));
  EXPECT_EQ(2, vector_index::diskann_metric_code_for_testing(
                   static_cast<vector_index::metric_type>(99)));

  const uint64_t expected_doc_id = 0x0102030405060708ULL;
  uint64_t parsed_doc_id = 0;
  EXPECT_FALSE(vector_index::parse_diskann_doc_id_for_testing(
      nullptr, sizeof(uint64_t), &parsed_doc_id));
  EXPECT_FALSE(vector_index::parse_diskann_doc_id_for_testing(
      reinterpret_cast<const uint8_t *>(&expected_doc_id), sizeof(uint32_t),
      &parsed_doc_id));
  EXPECT_FALSE(vector_index::parse_diskann_doc_id_for_testing(
      reinterpret_cast<const uint8_t *>(&expected_doc_id), sizeof(uint64_t),
      nullptr));
  EXPECT_TRUE(vector_index::parse_diskann_doc_id_for_testing(
      reinterpret_cast<const uint8_t *>(&expected_doc_id), sizeof(uint64_t),
      &parsed_doc_id));
  EXPECT_EQ(expected_doc_id, parsed_doc_id);
}

TEST(VectorIndexBackendTest,
     BackendHelperDirectoryAndDiskAnnApiWrappersCoverBranches) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_backend_misc_helper_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  ASSERT_FALSE(ec);

  EXPECT_TRUE(
      vector_index::ensure_parent_directory_for_testing(root + "/nested/child.txt"));
  EXPECT_TRUE(std::filesystem::exists(root + "/nested"));
  EXPECT_TRUE(vector_index::ensure_parent_directory_for_testing("manifest.v1"));

  write_binary_file(root + "/plain_file", "x");
  EXPECT_FALSE(vector_index::ensure_parent_directory_for_testing(
      root + "/plain_file/child.txt"));

  EXPECT_TRUE(
      vector_index::ends_with_for_testing("diskann_external.snapshot.v1", ".v1"));
  EXPECT_TRUE(vector_index::ends_with_for_testing("abc", ""));
  EXPECT_FALSE(vector_index::ends_with_for_testing("abc", "abcd"));
  EXPECT_FALSE(vector_index::ends_with_for_testing("abc", ".v1"));

  std::string decoded;
  EXPECT_FALSE(vector_index::decode_hex_bytes_for_testing("0", &decoded));
  EXPECT_FALSE(vector_index::decode_hex_bytes_for_testing("zz", &decoded));
  EXPECT_FALSE(vector_index::decode_hex_bytes_for_testing("4142", nullptr));
  EXPECT_TRUE(vector_index::decode_hex_bytes_for_testing("414243", &decoded));
  EXPECT_EQ("ABC", decoded);

  vector_index::set_faiss_external_snapshot_root_for_testing(root);
  EXPECT_NE(std::string::npos,
            vector_index::diskann_store_directory_for_testing("idx_helper")
                .find("diskann_external.store"));
  vector_index::reset_faiss_external_snapshot_root_for_testing();

  EXPECT_FALSE(vector_index::diskann_api_available_for_testing(
      false, true, true, true, true, true, true));
  EXPECT_FALSE(vector_index::diskann_api_available_for_testing(
      true, false, true, true, true, true, true));
  EXPECT_FALSE(vector_index::diskann_api_available_for_testing(
      true, true, false, true, true, true, true));
  EXPECT_FALSE(vector_index::diskann_api_available_for_testing(
      true, true, true, false, true, true, true));
  EXPECT_TRUE(vector_index::diskann_api_available_for_testing(
      true, true, true, true, true, true, true));
  EXPECT_FALSE(vector_index::diskann_api_available_for_testing(
      true, true, true, true, false, true, true));
  EXPECT_FALSE(vector_index::diskann_api_available_for_testing(
      true, true, true, true, true, false, true));
  EXPECT_FALSE(vector_index::diskann_api_available_for_testing(
      true, true, true, true, true, true, false));
  EXPECT_TRUE(vector_index::diskann_api_available_for_testing(
      true, true, true, true, true, true, true));
  EXPECT_FALSE(vector_index::diskann_api_parallel_bulk_build_available_for_testing(
      true, true, false, true, true, true, true, true, true));
  EXPECT_FALSE(vector_index::diskann_api_parallel_bulk_build_available_for_testing(
      false, true, true, true, true, true, true, true, true));
  EXPECT_FALSE(vector_index::diskann_api_parallel_bulk_build_available_for_testing(
      true, false, true, true, true, true, true, true, true));
  EXPECT_FALSE(vector_index::diskann_api_parallel_bulk_build_available_for_testing(
      true, true, true, true, true, false, true, true, true));
  EXPECT_TRUE(vector_index::diskann_api_parallel_bulk_build_available_for_testing(
      true, true, true, true, true, true, true, true, true));

  bool used_parallel_create = false;
  uint32_t observed_build_threads = 0;
  EXPECT_TRUE(vector_index::diskann_api_create_index_route_for_testing(
      false, false, 7, &used_parallel_create, &observed_build_threads));
  EXPECT_FALSE(used_parallel_create);
  EXPECT_EQ(0U, observed_build_threads);
  EXPECT_TRUE(vector_index::diskann_api_create_index_route_for_testing(
      true, false, 7, &used_parallel_create, &observed_build_threads));
  EXPECT_FALSE(used_parallel_create);
  EXPECT_EQ(0U, observed_build_threads);
  EXPECT_TRUE(vector_index::diskann_api_create_index_route_for_testing(
      false, true, 7, &used_parallel_create, &observed_build_threads));
  EXPECT_FALSE(used_parallel_create);
  EXPECT_EQ(0U, observed_build_threads);
  EXPECT_TRUE(vector_index::diskann_api_create_index_route_for_testing(
      true, true, 7, &used_parallel_create, &observed_build_threads));
  EXPECT_TRUE(used_parallel_create);
  EXPECT_EQ(7U, observed_build_threads);
  EXPECT_TRUE(vector_index::diskann_api_create_index_route_for_testing(
      true, true, 0, &used_parallel_create, &observed_build_threads));
  EXPECT_TRUE(used_parallel_create);
  EXPECT_EQ(0U, observed_build_threads);
  EXPECT_FALSE(vector_index::diskann_api_create_index_route_for_testing(
      true, true, 7, nullptr, &observed_build_threads));
  EXPECT_FALSE(vector_index::diskann_api_create_index_route_for_testing(
      true, true, 7, &used_parallel_create, nullptr));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_diskann_api_load");
    EXPECT_FALSE(vector_index::diskann_api_load_for_testing());
  }

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_diskann_api_symbol");
    EXPECT_FALSE(vector_index::diskann_api_load_for_testing());
  }

  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest, DiskAnnApiLoadUsesEnvironmentCandidates) {
  EnvVarGuard lib_guard("MYSQL_VECTOR_DISKANN_LIB");
  EnvVarGuard lib_dir_guard("MYSQL_VECTOR_DISKANN_LIB_DIR");
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_env_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);

  setenv("MYSQL_VECTOR_DISKANN_LIB", (root + "/missing.dylib").c_str(), 1);
  setenv("MYSQL_VECTOR_DISKANN_LIB_DIR", root.c_str(), 1);
  (void)vector_index::diskann_api_load_for_testing();
  SUCCEED();

  setenv("MYSQL_VECTOR_DISKANN_LIB", "", 1);
  setenv("MYSQL_VECTOR_DISKANN_LIB_DIR", "", 1);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest, DiskAnnMemoryModeRejectsExternalOnlyOperations) {
  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kMemory,
      "idx_diskann_memory_reject");
  const std::unordered_map<uint64_t, vector_index::vector_data> entries{
      {1, {1.0F, 2.0F}}};

  EXPECT_FALSE(backend.upsert(1, {1.0F, 2.0F}));
  EXPECT_FALSE(backend.erase(1));
  EXPECT_FALSE(backend.load_committed_entries(entries));
  EXPECT_FALSE(backend.rebuild_from_committed_entries(entries));
  EXPECT_FALSE(backend.recover());
  EXPECT_FALSE(backend.recover_committed_entries(entries));
  EXPECT_FALSE(backend.set_diskann_build_params(16, 64, 0));
  EXPECT_FALSE(backend.set_diskann_build_threads(1));
  EXPECT_FALSE(backend.set_diskann_search_complexity(64));
  EXPECT_EQ(0U, backend.diskann_max_degree());
  EXPECT_EQ(0U, backend.diskann_build_complexity());
  EXPECT_EQ(0U, backend.diskann_build_threads());
  EXPECT_EQ(0U, backend.diskann_search_complexity());
}

TEST(VectorIndexBackendTest, DiskAnnExternalRejectsWrongDimensionsBeforeNativePath) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_wrong_dimension_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_wrong_dimension");
  const std::unordered_map<uint64_t, vector_index::vector_data> bad_entries{
      {1, {1.0F}}};
  std::vector<vector_index::search_result> result;

  EXPECT_FALSE(backend.upsert(1, {1.0F}));
  EXPECT_FALSE(backend.load_committed_entries(bad_entries));
  EXPECT_FALSE(backend.rebuild_from_committed_entries(bad_entries));
  EXPECT_FALSE(backend.recover_committed_entries(bad_entries));
  EXPECT_FALSE(backend.search({1.0F}, 1, &result));
  result = {{99, 99.0}};
  EXPECT_TRUE(backend.search({1.0F, 1.0F}, 0, &result));
  EXPECT_TRUE(result.empty());

  std::vector<std::vector<vector_index::search_result>> batch_results;
  EXPECT_FALSE(backend.search_batch({{1.0F, 1.0F}}, 1, nullptr));
  EXPECT_TRUE(backend.search_batch({}, 1, &batch_results));
  EXPECT_TRUE(batch_results.empty());
  EXPECT_FALSE(backend.search_batch({{1.0F}}, 1, &batch_results));
  ASSERT_TRUE(backend.search_batch({{1.0F, 1.0F}, {2.0F, 2.0F}}, 0,
                                   &batch_results));
  ASSERT_EQ(2U, batch_results.size());
  EXPECT_TRUE(batch_results[0].empty());
  EXPECT_TRUE(batch_results[1].empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     BackendHelperDiskAnnStorageWrappersCoverEmptyAndRoundTripBranches) {
  DataHomeGuard data_home_guard;
  vector_index::reset_faiss_external_snapshot_root_for_testing();
  data_home_guard.Set("");

  EXPECT_EQ("", vector_index::diskann_term_directory_for_testing("idx_diskann_empty", 7));
  EXPECT_EQ("",
            vector_index::diskann_key_path_for_testing("idx_diskann_empty", 7, ""));
  std::string value;
  EXPECT_FALSE(vector_index::diskann_load_value_for_testing("idx_diskann_empty", 7,
                                                        "", &value));
  EXPECT_FALSE(vector_index::diskann_load_value_for_testing("idx_diskann_empty", 7,
                                                        "aa", nullptr));
  EXPECT_FALSE(vector_index::diskann_save_value_for_testing("idx_diskann_empty", 7,
                                                        "", "value"));
  EXPECT_FALSE(vector_index::diskann_delete_value_for_testing("idx_diskann_empty", 7,
                                                          ""));

  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_helper_store_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  const std::string key_bytes = "ab";
  const std::string payload = "payload";
  const std::string term_directory =
      vector_index::diskann_term_directory_for_testing("idx_diskann_helper", 5);
  ASSERT_FALSE(term_directory.empty());
  EXPECT_NE(std::string::npos,
            vector_index::diskann_term_directory_for_testing(
                "idx_diskann_helper", 0)
                .find("/vector"));
  EXPECT_NE(std::string::npos,
            vector_index::diskann_term_directory_for_testing(
                "idx_diskann_helper", 1)
                .find("/neighbors"));
  EXPECT_NE(std::string::npos,
            vector_index::diskann_term_directory_for_testing(
                "idx_diskann_helper", 2)
                .find("/quantized"));
  EXPECT_NE(std::string::npos,
            vector_index::diskann_term_directory_for_testing(
                "idx_diskann_helper", 3)
                .find("/attributes"));
  EXPECT_NE(std::string::npos,
            vector_index::diskann_term_directory_for_testing(
                "idx_diskann_helper", 4)
                .find("/metadata"));
  EXPECT_NE(std::string::npos,
            vector_index::diskann_term_directory_for_testing(
                "idx_diskann_helper", 6)
                .find("/extmap"));
  EXPECT_NE(std::string::npos,
            vector_index::diskann_term_directory_for_testing(
                "idx_diskann_helper", 7)
                .find("/unknown"));
  const std::string key_path = vector_index::diskann_key_path_for_testing(
      "idx_diskann_helper", 5, key_bytes);
  ASSERT_FALSE(key_path.empty());

  ASSERT_TRUE(vector_index::diskann_save_value_for_testing("idx_diskann_helper", 5,
                                                       key_bytes, payload));
  value.clear();
  ASSERT_TRUE(vector_index::diskann_load_value_for_testing("idx_diskann_helper", 5,
                                                       key_bytes, &value));
  EXPECT_EQ(payload, value);
  ASSERT_TRUE(vector_index::diskann_delete_value_for_testing("idx_diskann_helper", 5,
                                                         key_bytes));
  EXPECT_FALSE(vector_index::diskann_load_value_for_testing("idx_diskann_helper", 5,
                                                        key_bytes, &value));

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     BackendHelperDiskAnnBuildMemoryStoreFlushesOnlyAfterSuccess) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_memory_store_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  const std::string key_bytes = "memory-key";
  const std::string payload = "memory-payload";
  std::string loaded_before_flush;
  std::string loaded_after_flush;
  bool file_exists_before_flush = true;

  EXPECT_FALSE(vector_index::diskann_build_memory_store_round_trip_for_testing(
      "idx_diskann_memory_store_null_before", 5, key_bytes, payload, nullptr,
      &file_exists_before_flush, &loaded_after_flush));
  EXPECT_FALSE(vector_index::diskann_build_memory_store_round_trip_for_testing(
      "idx_diskann_memory_store_null_exists", 5, key_bytes, payload,
      &loaded_before_flush, nullptr, &loaded_after_flush));
  EXPECT_FALSE(vector_index::diskann_build_memory_store_round_trip_for_testing(
      "idx_diskann_memory_store_null_after", 5, key_bytes, payload,
      &loaded_before_flush, &file_exists_before_flush, nullptr));

  ASSERT_TRUE(vector_index::diskann_build_memory_store_round_trip_for_testing(
      "idx_diskann_memory_store", 5, key_bytes, payload, &loaded_before_flush,
      &file_exists_before_flush, &loaded_after_flush));
  EXPECT_EQ(payload, loaded_before_flush);
  EXPECT_FALSE(file_exists_before_flush);
  EXPECT_EQ(payload, loaded_after_flush);
  EXPECT_TRUE(std::filesystem::exists(vector_index::diskann_key_path_for_testing(
      "idx_diskann_memory_store", 5, key_bytes)));

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     BackendHelperDiskAnnResidentStoreSurvivesPersistentFileRemoval) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_resident_store_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  std::string loaded_after_removing_file;
  EXPECT_FALSE(vector_index::diskann_resident_store_round_trip_for_testing(
      "idx_diskann_resident_store_null", 5, "resident-key",
      "resident-payload", nullptr));
  ASSERT_TRUE(vector_index::diskann_resident_store_round_trip_for_testing(
      "idx_diskann_resident_store", 5, "resident-key", "resident-payload",
      &loaded_after_removing_file));
  EXPECT_EQ("resident-payload", loaded_after_removing_file);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     BackendHelperDiskAnnReadModifyWriteCoversStoreModes) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_rmw_store_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  std::string persistent_value;
  std::string build_memory_value;
  std::string resident_value;
  ASSERT_TRUE(vector_index::diskann_read_modify_write_round_trip_for_testing(
      "idx_diskann_rmw_store", 5, "rmw-key", "abcdefgh", "XYZ", 8,
      &persistent_value, &build_memory_value, &resident_value));
  EXPECT_EQ("XYZdefgh", persistent_value);
  EXPECT_EQ("", build_memory_value);
  EXPECT_EQ("XYZdefgh", resident_value);
  EXPECT_FALSE(vector_index::diskann_read_modify_write_round_trip_for_testing(
      "idx_diskann_rmw_store_null_output", 5, "rmw-key", "abcdefgh", "XYZ", 8,
      nullptr, &build_memory_value, &resident_value));
  EXPECT_FALSE(vector_index::diskann_read_modify_write_round_trip_for_testing(
      "idx_diskann_rmw_store_null_build_memory", 5, "rmw-key", "abcdefgh",
      "XYZ", 8, &persistent_value, nullptr, &resident_value));
  EXPECT_FALSE(vector_index::diskann_read_modify_write_round_trip_for_testing(
      "idx_diskann_rmw_store_null_resident", 5, "rmw-key", "abcdefgh", "XYZ",
      8, &persistent_value, &build_memory_value, nullptr));
  EXPECT_FALSE(vector_index::diskann_read_modify_write_round_trip_for_testing(
      "idx_diskann_rmw_store_empty_key", 5, "", "abcdefgh", "XYZ", 8,
      &persistent_value, &build_memory_value, &resident_value));
  EXPECT_FALSE(vector_index::diskann_read_modify_write_round_trip_for_testing(
      "idx_diskann_rmw_store_overflow", 5, "rmw-key", "abcdefgh", "XYZ",
      std::numeric_limits<size_t>::max(), &persistent_value,
      &build_memory_value, &resident_value));

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     BackendHelperDiskAnnParsePrefixedKeysCoversMalformedAndVisitorFailure) {
  std::vector<std::string> keys;

  EXPECT_TRUE(vector_index::diskann_parse_prefixed_keys_for_testing("", 0, &keys));
  EXPECT_TRUE(keys.empty());
  EXPECT_FALSE(vector_index::diskann_parse_prefixed_keys_for_testing("", 0,
                                                                     nullptr));
  EXPECT_FALSE(
      vector_index::diskann_parse_prefixed_keys_for_testing("", 1, &keys));

  std::string truncated;
  uint32_t len = 4;
  truncated.append(reinterpret_cast<const char *>(&len), sizeof(len));
  truncated.append("abc", 3);
  EXPECT_FALSE(vector_index::diskann_parse_prefixed_keys_for_testing(truncated,
                                                                     1, &keys));
  EXPECT_FALSE(vector_index::diskann_parse_prefixed_keys_for_testing(
      truncated, 1, &keys, -1, true));

  std::string payload;
  const std::string k1 = "aa";
  const std::string k2 = "bbb";
  uint32_t l1 = static_cast<uint32_t>(k1.size());
  uint32_t l2 = static_cast<uint32_t>(k2.size());
  payload.append(reinterpret_cast<const char *>(&l1), sizeof(l1));
  payload.append(k1);
  payload.append(reinterpret_cast<const char *>(&l2), sizeof(l2));
  payload.append(k2);

  ASSERT_TRUE(vector_index::diskann_parse_prefixed_keys_for_testing(payload, 2, &keys));
  ASSERT_EQ(2U, keys.size());
  EXPECT_EQ(k1, keys[0]);
  EXPECT_EQ(k2, keys[1]);

  EXPECT_FALSE(
      vector_index::diskann_parse_prefixed_keys_for_testing(payload, 2, &keys, 1));
}

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

TEST(VectorIndexBackendTest,
     BaseLoadCommittedEntriesFromReaderUsesMutationContract) {
  StubBackend backend;
  size_t reader_calls = 0;

  auto reader = [&](const vector_index::committed_entry_visitor &visitor) {
    ++reader_calls;
    return visitor(1, {1.0F, 1.0F}) && visitor(2, {2.0F, 2.0F});
  };

  EXPECT_TRUE(backend.load_committed_entries_from_reader(reader));
  EXPECT_EQ(1U, reader_calls);
  EXPECT_EQ(2U, backend.upsert_calls);
  EXPECT_EQ((vector_index::vector_data{1.0F, 1.0F}),
            backend.upserted_entries[1]);
  EXPECT_EQ((vector_index::vector_data{2.0F, 2.0F}),
            backend.upserted_entries[2]);

  EXPECT_FALSE(backend.load_committed_entries_from_reader(nullptr));

  backend.allow_upsert = false;
  EXPECT_FALSE(backend.load_committed_entries_from_reader(reader));

  StubBackend rebuild_backend;
  EXPECT_TRUE(rebuild_backend.rebuild_from_committed_entries_from_reader(reader));
  EXPECT_EQ(2U, rebuild_backend.upsert_calls);

  StubBackend recover_backend;
  EXPECT_TRUE(recover_backend.recover_committed_entries_from_reader(reader));
  EXPECT_EQ(2U, recover_backend.upsert_calls);
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

TEST(VectorIndexBuildOptionsTest,
     EffectiveBatchSearchThreadsUseOverrideGlobalAndCap) {
  UlongGuard global_guard(&opt_vector_batch_search_threads, 32);
  UlongGuard hnsw_guard(&opt_vector_hnsw_search_threads, 0);
  UlongGuard faiss_guard(&opt_vector_faiss_search_threads, 0);
  UlongGuard diskann_guard(&opt_vector_diskann_search_threads, 0);

  EXPECT_EQ(0U, vector_index::effective_batch_search_threads(0, 0));
  EXPECT_EQ(1U, vector_index::effective_batch_search_threads(1, 0));
  EXPECT_EQ(8U, vector_index::effective_batch_search_threads(8, 0));
  EXPECT_EQ(32U, vector_index::effective_batch_search_threads(64, 0));
  EXPECT_EQ(4U, vector_index::effective_batch_search_threads(64, 4));
  EXPECT_EQ(2U, vector_index::effective_batch_search_threads(2, 4));
  EXPECT_EQ(64U, vector_index::effective_batch_search_threads(64, 65535));

  opt_vector_hnsw_search_threads = 2;
  opt_vector_faiss_search_threads = 3;
  opt_vector_diskann_search_threads = 4;
  EXPECT_EQ(2U, vector_index::effective_hnsw_search_threads(64));
  EXPECT_EQ(3U, vector_index::effective_faiss_search_threads(64));
  EXPECT_EQ(4U, vector_index::effective_diskann_search_threads(64));

  opt_vector_batch_search_threads = 0;
  opt_vector_hnsw_search_threads = 0;
  EXPECT_GE(vector_index::effective_hnsw_search_threads(64), 1U);
  EXPECT_LE(vector_index::effective_hnsw_search_threads(64), 32U);
}

TEST(VectorIndexRuntimeThreadPoolTest,
     EffectiveRuntimeWorkerCountUsesConfiguredAutoAndCaps) {
  EXPECT_EQ(0U, vector_index::effective_runtime_worker_count(0, 4));
  EXPECT_EQ(1U, vector_index::effective_runtime_worker_count(1, 4));
  EXPECT_EQ(4U, vector_index::effective_runtime_worker_count(64, 4));
  EXPECT_EQ(vector_index::k_max_build_threads,
            vector_index::effective_runtime_worker_count(
                vector_index::k_max_build_threads + 8U,
                vector_index::k_max_build_threads + 8U));

  const size_t auto_threads =
      vector_index::effective_runtime_worker_count(64, 0);
  EXPECT_GE(auto_threads, 1U);
  EXPECT_LE(auto_threads, 64U);
}

TEST(VectorIndexRuntimeThreadPoolTest,
     ParallelForQueriesVisitsRangesAndPropagatesFailure) {
  vector_index::reset_runtime_worker_pool_for_testing();
  std::vector<int> seen(7, 0);
  EXPECT_TRUE(vector_index::parallel_for_queries(
      seen.size(), 3, [&](size_t begin, size_t end, size_t) {
        EXPECT_LT(begin, end);
        for (size_t i = begin; i < end; ++i) ++seen[i];
        return true;
      }));
  for (int value : seen) EXPECT_EQ(1, value);

  EXPECT_TRUE(vector_index::parallel_for_queries(
      0, 3, [](size_t, size_t, size_t) { return false; }));
  EXPECT_FALSE(vector_index::parallel_for_queries(
      4, 2, [](size_t begin, size_t, size_t) { return begin == 0; }));
  EXPECT_FALSE(vector_index::parallel_for_queries(
      1, 1, vector_index::query_range_visitor{}));
  EXPECT_TRUE(vector_index::parallel_for_queries(
      3, 1, [](size_t begin, size_t end, size_t worker_id) {
        EXPECT_EQ(0U, begin);
        EXPECT_EQ(3U, end);
        EXPECT_EQ(0U, worker_id);
        return true;
      }));

  for (size_t attempt = 0; attempt < 128; ++attempt) {
    EXPECT_FALSE(vector_index::parallel_for_queries(
        4, 2, [](size_t begin, size_t, size_t) { return begin == 0; }));
  }
  EXPECT_FALSE(vector_index::parallel_for_queries(
      4, 2, [](size_t begin, size_t, size_t) {
        if (begin != 0) throw std::runtime_error("visitor failure");
        return true;
      }));
  vector_index::reset_runtime_worker_pool_for_testing();
}

TEST(VectorIndexRuntimeThreadPoolTest, ParallelForQueriesReusesWorkerPool) {
  vector_index::reset_runtime_worker_pool_for_testing();
  EXPECT_EQ(0U, vector_index::runtime_worker_pool_size_for_testing());

  EXPECT_TRUE(vector_index::parallel_for_queries(
      8, 3, [](size_t, size_t, size_t) { return true; }));
  EXPECT_EQ(3U, vector_index::runtime_worker_pool_size_for_testing());

  EXPECT_TRUE(vector_index::parallel_for_queries(
      6, 3, [](size_t, size_t, size_t) { return true; }));
  EXPECT_EQ(3U, vector_index::runtime_worker_pool_size_for_testing());

  EXPECT_TRUE(vector_index::parallel_for_queries(
      8, 4, [](size_t, size_t, size_t) { return true; }));
  EXPECT_EQ(4U, vector_index::runtime_worker_pool_size_for_testing());

  EXPECT_TRUE(vector_index::parallel_for_queries(
      3, 1, [](size_t, size_t, size_t) { return true; }));
  EXPECT_EQ(4U, vector_index::runtime_worker_pool_size_for_testing());

  vector_index::reset_runtime_worker_pool_for_testing();
  EXPECT_EQ(0U, vector_index::runtime_worker_pool_size_for_testing());
}

TEST(VectorIndexRuntimeThreadPoolTest,
     ParallelForRangesSupportsNonQueryWorkItems) {
  std::vector<int> seen(5, 0);
  EXPECT_TRUE(vector_index::parallel_for_ranges(
      seen.size(), 2, [&](size_t begin, size_t end, size_t) {
        for (size_t i = begin; i < end; ++i) ++seen[i];
        return true;
      }));
  for (int value : seen) EXPECT_EQ(1, value);
}

TEST(VectorIndexRuntimeThreadPoolTest,
     ScopedParallelForRangesAllowsNestedSharedPoolWork) {
  vector_index::reset_runtime_worker_pool_for_testing();
  std::vector<int> seen(4, 0);
  EXPECT_TRUE(vector_index::parallel_for_ranges_scoped(
      seen.size(), 2, [&](size_t begin, size_t end, size_t) {
        return vector_index::parallel_for_ranges(
            end - begin, 2, [&](size_t nested_begin, size_t nested_end,
                                size_t) {
              for (size_t i = begin + nested_begin; i < begin + nested_end;
                   ++i) {
                ++seen[i];
              }
              return true;
            });
      }));
  for (int value : seen) EXPECT_EQ(1, value);
  EXPECT_EQ(2U, vector_index::runtime_worker_pool_size_for_testing());
  vector_index::reset_runtime_worker_pool_for_testing();
}

TEST(VectorIndexRuntimeConfigTest,
     ValidatesCommonProviderModeAndThreadSemantics) {
  vector_index::runtime_common_config config;
  config.dimension = 4;
  config.provider = vector_index::backend_provider::kDiskAnn;
  config.mode = vector_index::backend_mode::kExternal;
  EXPECT_TRUE(vector_index::runtime_common_config_valid(config));

  config.dimension = 0;
  EXPECT_FALSE(vector_index::runtime_common_config_valid(config));

  config.dimension = 4;
  config.mode = vector_index::backend_mode::kMemory;
  EXPECT_FALSE(vector_index::runtime_common_config_valid(config));

  EXPECT_TRUE(vector_index::runtime_provider_accepts_mode(
      vector_index::backend_provider::kFaiss,
      vector_index::backend_mode::kMemory));
  EXPECT_TRUE(vector_index::runtime_provider_accepts_mode(
      vector_index::backend_provider::kFaiss,
      vector_index::backend_mode::kExternal));
  EXPECT_TRUE(vector_index::runtime_provider_accepts_mode(
      vector_index::backend_provider::kNative,
      vector_index::backend_mode::kMemory));
  EXPECT_TRUE(vector_index::runtime_provider_accepts_mode(
      vector_index::backend_provider::kNative,
      vector_index::backend_mode::kExternal));
  EXPECT_TRUE(vector_index::runtime_provider_accepts_mode(
      vector_index::backend_provider::kHnswlib,
      vector_index::backend_mode::kMemory));
  EXPECT_FALSE(vector_index::runtime_provider_accepts_mode(
      vector_index::backend_provider::kHnswlib,
      vector_index::backend_mode::kExternal));
  EXPECT_FALSE(vector_index::runtime_provider_accepts_mode(
      vector_index::backend_provider::kFaiss,
      static_cast<vector_index::backend_mode>(999)));
  EXPECT_FALSE(vector_index::runtime_provider_accepts_mode(
      vector_index::backend_provider::kNative,
      static_cast<vector_index::backend_mode>(999)));
  EXPECT_FALSE(vector_index::backend_provider_supported(
      static_cast<vector_index::backend_provider>(999)));
  EXPECT_FALSE(vector_index::default_backend_provider(nullptr));
  vector_index::backend_mode default_mode =
      vector_index::backend_mode::kExternal;
  EXPECT_FALSE(vector_index::default_backend_mode_for_provider(
      vector_index::backend_provider::kFaiss, nullptr));
  EXPECT_FALSE(vector_index::default_backend_mode_for_provider(
      static_cast<vector_index::backend_provider>(999), &default_mode));

  EXPECT_EQ(0U, vector_index::resolve_runtime_threads(0, 0));
  EXPECT_EQ(8U, vector_index::resolve_runtime_threads(0, 8));
  EXPECT_EQ(4U, vector_index::resolve_runtime_threads(4, 8));
  EXPECT_EQ(vector_index::k_max_build_threads,
            vector_index::resolve_runtime_threads(
                vector_index::k_max_build_threads + 1U, 8));
}

TEST(VectorIndexRuntimeConfigTest, InfersFaissRuntimeKindFromParams) {
  EXPECT_EQ(vector_index::faiss_runtime_index_kind::kHnsw,
            vector_index::faiss_runtime_kind_from_params(0, 0, 0));
  EXPECT_EQ(vector_index::faiss_runtime_index_kind::kIvfFlat,
            vector_index::faiss_runtime_kind_from_params(16, 4, 0));
  EXPECT_EQ(vector_index::faiss_runtime_index_kind::kIvfFlat,
            vector_index::faiss_runtime_kind_from_params(16, 0, 8));
  EXPECT_EQ(vector_index::faiss_runtime_index_kind::kIvfFlat,
            vector_index::faiss_runtime_kind_from_params(16, 0, 0));
  EXPECT_EQ(vector_index::faiss_runtime_index_kind::kIvfPq,
            vector_index::faiss_runtime_kind_from_params(16, 4, 8));
}

TEST(VectorIndexBackendTest, CommonHelpersCoverDistanceHexAndSizeBounds) {
  double distance = 0.0;
  EXPECT_FALSE(vector_index::detail::compute_distance(
      vector_index::metric_type::kEuclidean, {1.0F}, {1.0F, 2.0F},
      &distance));
  ASSERT_TRUE(vector_index::detail::compute_distance(
      vector_index::metric_type::kEuclidean, {1.0F, 1.0F}, {4.0F, 5.0F},
      &distance));
  EXPECT_DOUBLE_EQ(5.0, distance);
  ASSERT_TRUE(vector_index::detail::compute_distance(
      vector_index::metric_type::kInnerProduct, {1.0F, 2.0F}, {3.0F, 4.0F},
      &distance));
  EXPECT_DOUBLE_EQ(-11.0, distance);
  EXPECT_FALSE(vector_index::detail::compute_distance(
      vector_index::metric_type::kCosine, {0.0F, 0.0F}, {1.0F, 0.0F},
      &distance));
  ASSERT_TRUE(vector_index::detail::compute_distance(
      vector_index::metric_type::kCosine, {1.0F, 0.0F}, {1.0F, 0.0F},
      &distance));
  EXPECT_DOUBLE_EQ(0.0, distance);

  EXPECT_EQ("mixed token",
            vector_index::detail::normalize_token("  MIXED Token  "));
  EXPECT_TRUE(vector_index::detail::starts_with("abcdef", "abc"));
  EXPECT_FALSE(vector_index::detail::starts_with("ab", "abc"));
  EXPECT_TRUE(vector_index::detail::ends_with("abcdef", "def"));
  EXPECT_FALSE(vector_index::detail::ends_with("ab", "abc"));

  unsigned value = 99;
  EXPECT_FALSE(vector_index::detail::hex_value('0', nullptr));
  EXPECT_FALSE(vector_index::detail::hex_value('/', &value));
  ASSERT_TRUE(vector_index::detail::hex_value('0', &value));
  EXPECT_EQ(0U, value);
  ASSERT_TRUE(vector_index::detail::hex_value('9', &value));
  EXPECT_EQ(9U, value);
  ASSERT_TRUE(vector_index::detail::hex_value('a', &value));
  EXPECT_EQ(10U, value);
  ASSERT_TRUE(vector_index::detail::hex_value('F', &value));
  EXPECT_EQ(15U, value);
  EXPECT_FALSE(vector_index::detail::hex_value('G', &value));
  EXPECT_FALSE(vector_index::detail::hex_value('g', &value));

  std::string decoded;
  EXPECT_TRUE(vector_index::detail::decode_hex_bytes("", &decoded));
  EXPECT_TRUE(decoded.empty());
  EXPECT_FALSE(vector_index::detail::decode_hex_bytes("0", &decoded));
  EXPECT_FALSE(vector_index::detail::decode_hex_bytes("0g", &decoded));
  EXPECT_FALSE(vector_index::detail::decode_hex_bytes("g0", &decoded));
  EXPECT_FALSE(vector_index::detail::decode_hex_bytes("00", nullptr));
  ASSERT_TRUE(vector_index::detail::decode_hex_bytes("41427a", &decoded));
  EXPECT_EQ("ABz", decoded);

  const size_t max_size = std::numeric_limits<size_t>::max();
  EXPECT_EQ(0U, vector_index::saturated_mul_size(0, max_size));
  EXPECT_EQ(0U, vector_index::saturated_mul_size(max_size, 0));
  EXPECT_EQ(42U, vector_index::saturated_mul_size(6, 7));
  EXPECT_EQ(max_size, vector_index::saturated_mul_size(max_size, 2));
  EXPECT_EQ(9U, vector_index::saturated_add_size(4, 5));
  EXPECT_EQ(max_size, vector_index::saturated_add_size(max_size, 1));
}

TEST(VectorIndexRuntimeConfigTest, SelectsUniformSamplePositions) {
  size_t position = 0;

  ASSERT_TRUE(
      vector_index::runtime_uniform_sample_position(0, 10, 4, &position));
  EXPECT_EQ(0U, position);
  ASSERT_TRUE(
      vector_index::runtime_uniform_sample_position(1, 10, 4, &position));
  EXPECT_EQ(2U, position);
  ASSERT_TRUE(
      vector_index::runtime_uniform_sample_position(2, 10, 4, &position));
  EXPECT_EQ(5U, position);
  ASSERT_TRUE(
      vector_index::runtime_uniform_sample_position(3, 10, 4, &position));
  EXPECT_EQ(7U, position);

  EXPECT_FALSE(
      vector_index::runtime_uniform_sample_position(0, 0, 4, &position));
  EXPECT_FALSE(
      vector_index::runtime_uniform_sample_position(0, 4, 0, &position));
  EXPECT_FALSE(
      vector_index::runtime_uniform_sample_position(4, 10, 4, &position));
  EXPECT_FALSE(
      vector_index::runtime_uniform_sample_position(0, 4, 5, &position));
  EXPECT_FALSE(vector_index::runtime_uniform_sample_position(0, 4, 1, nullptr));
}

TEST(VectorIndexBuildOptionsTest, GlobalEnumOptionsReturnKnownValuesAndFallbacks) {
  UlongGuard diskann_build_mode_guard(
      &opt_vector_diskann_build_mode,
      static_cast<ulong>(vector_index::diskann_build_mode::kAuto));
  UlongGuard default_library_guard(
      &opt_vector_default_library,
      static_cast<ulong>(vector_index::vector_default_library::kNone));
  UlongGuard consistency_mode_guard(
      &opt_vector_index_consistency_mode,
      static_cast<ulong>(vector_index::index_consistency_mode::kTransactional));

  opt_vector_diskann_build_mode =
      static_cast<ulong>(vector_index::diskann_build_mode::kSerial);
  EXPECT_EQ(vector_index::diskann_build_mode::kSerial,
            vector_index::global_diskann_build_mode());
  opt_vector_diskann_build_mode =
      static_cast<ulong>(vector_index::diskann_build_mode::kOffline);
  EXPECT_EQ(vector_index::diskann_build_mode::kOffline,
            vector_index::global_diskann_build_mode());
  opt_vector_diskann_build_mode = 999;
  EXPECT_EQ(vector_index::diskann_build_mode::kAuto,
            vector_index::global_diskann_build_mode());

  opt_vector_default_library =
      static_cast<ulong>(vector_index::vector_default_library::kDiskAnn);
  EXPECT_EQ(vector_index::vector_default_library::kDiskAnn,
            vector_index::global_vector_default_library());
  opt_vector_default_library =
      static_cast<ulong>(vector_index::vector_default_library::kHnsw);
  EXPECT_EQ(vector_index::vector_default_library::kHnsw,
            vector_index::global_vector_default_library());
  opt_vector_default_library =
      static_cast<ulong>(vector_index::vector_default_library::kFaiss);
  EXPECT_EQ(vector_index::vector_default_library::kFaiss,
            vector_index::global_vector_default_library());
  opt_vector_default_library = 999;
  EXPECT_EQ(vector_index::vector_default_library::kNone,
            vector_index::global_vector_default_library());

  opt_vector_index_consistency_mode =
      static_cast<ulong>(vector_index::index_consistency_mode::kStandalone);
  EXPECT_EQ(vector_index::index_consistency_mode::kStandalone,
            vector_index::global_index_consistency_mode());
  opt_vector_index_consistency_mode = 999;
  EXPECT_EQ(vector_index::index_consistency_mode::kTransactional,
            vector_index::global_index_consistency_mode());
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

TEST(VectorIndexBackendTest, ExactBackendsRejectNullSearchResults) {
  vector_index::memory_backend memory(
      2, vector_index::metric_type::kEuclidean);
  vector_index::external_backend external(
      2, vector_index::metric_type::kEuclidean);
  vector_index::faiss_backend faiss(2,
                                    vector_index::metric_type::kEuclidean,
                                    vector_index::backend_mode::kMemory);

  EXPECT_FALSE(memory.search({1.0F, 1.0F}, 1, nullptr));
  EXPECT_FALSE(external.search({1.0F, 1.0F}, 1, nullptr));
  EXPECT_FALSE(faiss.search({1.0F, 1.0F}, 1, nullptr));
}

TEST(VectorIndexBackendTest, MemoryBackendCosineRejectsZeroNorm) {
  vector_index::memory_backend backend(2, vector_index::metric_type::kCosine);
  std::vector<vector_index::search_result> result;

  EXPECT_TRUE(backend.upsert(1, {0.0F, 0.0F}));
  EXPECT_FALSE(backend.search({1.0F, 0.0F}, 1, &result));
}

TEST(VectorIndexBackendTest, MemoryBackendInnerProductDistanceOrdering) {
  vector_index::memory_backend backend(2, vector_index::metric_type::kInnerProduct);
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

TEST(VectorIndexBackendTest, FaissMemoryFallbackBehavesLikeMemoryBackend) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kMemory);
  std::vector<vector_index::search_result> result;

  EXPECT_EQ(vector_index::backend_provider::kFaiss, backend.provider());
  EXPECT_TRUE(backend.supports_mutations());
  EXPECT_TRUE(backend.upsert(1, {0.0F, 0.0F}));
  EXPECT_TRUE(backend.upsert(2, {1.0F, 1.0F}));
  EXPECT_TRUE(backend.search({0.1F, 0.1F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_TRUE(backend.erase(2));
}

TEST(VectorIndexBackendTest, FaissMemoryRejectsInvalidUpdatesBeforeMutation) {
#ifndef HAVE_FAISS
  GTEST_SKIP() << "Faiss native mutation validation requires HAVE_FAISS";
#else
  vector_index::faiss_backend backend(2, vector_index::metric_type::kCosine,
                                      vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));

  EXPECT_FALSE(backend.upsert(1, {1.0F}));
  EXPECT_FALSE(backend.upsert(1, {0.0F, 0.0F}));
  EXPECT_EQ(1U, backend.entry_count());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
#endif
}

TEST(VectorIndexBackendTest, FaissMemoryLoadCommittedAndRecoverStayInMemoryMode) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kMemory);
  std::unordered_map<uint64_t, vector_index::vector_data> committed_entries{
      {7, {7.0F, 7.0F}}, {9, {9.0F, 9.0F}}};
  std::vector<vector_index::search_result> result;

  EXPECT_TRUE(backend.load_committed_entries(committed_entries));
  EXPECT_TRUE(backend.recover());
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_EQ(0U, backend.external_manifest_generation());
  EXPECT_TRUE(backend.search({7.0F, 7.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
}

TEST(VectorIndexBackendTest, FaissExternalFallbackSupportsMutableSearch) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal);
  std::vector<vector_index::search_result> result;

  EXPECT_EQ(vector_index::backend_provider::kFaiss, backend.provider());
  EXPECT_TRUE(backend.supports_mutations());
  EXPECT_TRUE(backend.upsert(1, {1.0F, 2.0F}));
  EXPECT_TRUE(backend.upsert(2, {2.0F, 2.0F}));
  EXPECT_EQ(2U, backend.entry_count());
  EXPECT_TRUE(backend.search({1.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_TRUE(backend.erase(2));
}

TEST(VectorIndexBackendTest, FaissMemoryInnerProductUsesFlatIpIndex) {
  vector_index::faiss_backend backend(
      2, vector_index::metric_type::kInnerProduct,
      vector_index::backend_mode::kMemory);
  std::vector<vector_index::search_result> result;

  EXPECT_TRUE(backend.upsert(1, {1.0F, 0.0F}));
  EXPECT_TRUE(backend.upsert(2, {0.0F, 1.0F}));
  EXPECT_TRUE(backend.search({1.0F, 0.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexBackendTest, FaissExternalIgnoresRuntimeEnvironmentOptions) {
  unsetenv("MYSQL_VECTOR_FAISS_HNSW_M");
  unsetenv("MYSQL_VECTOR_FAISS_HNSW_EF_CONSTRUCTION");
  unsetenv("MYSQL_VECTOR_FAISS_SEARCH_EF");
  unsetenv("MYSQL_VECTOR_FAISS_KEEP_LOADED");
  vector_index::faiss_backend unset_defaults(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal);
  EXPECT_EQ(32U, unset_defaults.hnsw_m());
  EXPECT_EQ(40U, unset_defaults.hnsw_ef_construction());
  EXPECT_EQ(64U, unset_defaults.search_ef());

  setenv("MYSQL_VECTOR_FAISS_HNSW_M", "", 1);
  setenv("MYSQL_VECTOR_FAISS_HNSW_EF_CONSTRUCTION", "", 1);
  setenv("MYSQL_VECTOR_FAISS_SEARCH_EF", "", 1);
  setenv("MYSQL_VECTOR_FAISS_KEEP_LOADED", "", 1);
  vector_index::faiss_backend empty_defaults(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal);
  EXPECT_EQ(32U, empty_defaults.hnsw_m());
  EXPECT_EQ(40U, empty_defaults.hnsw_ef_construction());
  EXPECT_EQ(64U, empty_defaults.search_ef());

  setenv("MYSQL_VECTOR_FAISS_HNSW_M", "24", 1);
  setenv("MYSQL_VECTOR_FAISS_HNSW_EF_CONSTRUCTION", "120", 1);
  setenv("MYSQL_VECTOR_FAISS_SEARCH_EF", "96", 1);
  setenv("MYSQL_VECTOR_FAISS_KEEP_LOADED", "1", 1);
  vector_index::faiss_backend tuned(2, vector_index::metric_type::kEuclidean,
                                   vector_index::backend_mode::kExternal);
  EXPECT_EQ(32U, tuned.hnsw_m());
  EXPECT_EQ(40U, tuned.hnsw_ef_construction());
  EXPECT_EQ(64U, tuned.search_ef());

  setenv("MYSQL_VECTOR_FAISS_HNSW_M", "0", 1);
  setenv("MYSQL_VECTOR_FAISS_HNSW_EF_CONSTRUCTION", "bad", 1);
  setenv("MYSQL_VECTOR_FAISS_SEARCH_EF", "4294967296", 1);
  setenv("MYSQL_VECTOR_FAISS_KEEP_LOADED", "invalid", 1);
  vector_index::faiss_backend fallback(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal);
  EXPECT_EQ(32U, fallback.hnsw_m());
  EXPECT_EQ(40U, fallback.hnsw_ef_construction());
  EXPECT_EQ(64U, fallback.search_ef());

  setenv("MYSQL_VECTOR_FAISS_HNSW_M", "12x", 1);
  setenv("MYSQL_VECTOR_FAISS_HNSW_EF_CONSTRUCTION", "40", 1);
  setenv("MYSQL_VECTOR_FAISS_SEARCH_EF", "64", 1);
  vector_index::faiss_backend trailing_junk(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal);
  EXPECT_EQ(32U, trailing_junk.hnsw_m());
  EXPECT_EQ(40U, trailing_junk.hnsw_ef_construction());
  EXPECT_EQ(64U, trailing_junk.search_ef());

  const char *bool_values[] = {"true", "TRUE", "yes", "YES", "on", "ON",
                               "0",    "false", "FALSE", "no",
                               "NO",   "off",   "OFF"};
  for (const char *value : bool_values) {
    setenv("MYSQL_VECTOR_FAISS_KEEP_LOADED", value, 1);
    vector_index::faiss_backend parser(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal);
    EXPECT_EQ(64U, parser.search_ef());
  }

  setenv("MYSQL_VECTOR_FAISS_HNSW_M", "32", 1);
  setenv("MYSQL_VECTOR_FAISS_HNSW_EF_CONSTRUCTION", "40", 1);
  setenv("MYSQL_VECTOR_FAISS_SEARCH_EF", "64", 1);
  setenv("MYSQL_VECTOR_FAISS_KEEP_LOADED", "1", 1);
}

TEST(VectorIndexBackendTest, FaissSearchAcceptsZeroTopK) {
  std::vector<vector_index::search_result> result{{99, 1.0}};

  vector_index::faiss_backend memory_backend(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kMemory);
  ASSERT_TRUE(memory_backend.upsert(1, {1.0F, 1.0F}));
  EXPECT_TRUE(memory_backend.search({1.0F, 1.0F}, 0, &result));
  EXPECT_TRUE(result.empty());

  vector_index::faiss_backend external_backend(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal);
  ASSERT_TRUE(external_backend.load_committed_entries({{1, {1.0F, 1.0F}}}));
  result.push_back({99, 1.0});
  EXPECT_TRUE(external_backend.search({1.0F, 1.0F}, 0, &result));
  EXPECT_TRUE(result.empty());
}

TEST(VectorIndexBackendTest, FaissSearchBatchCoversGuardAndEdgePaths) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));
  ASSERT_TRUE(backend.upsert(2, {2.0F, 2.0F}));

  std::vector<std::vector<vector_index::search_result>> batch_results;
  EXPECT_FALSE(backend.search_batch({{1.0F, 1.0F}}, 1, nullptr));
  EXPECT_TRUE(backend.search_batch({}, 1, &batch_results));
  EXPECT_TRUE(batch_results.empty());
  EXPECT_FALSE(backend.search_batch({{1.0F, 1.0F, 1.0F}}, 1,
                                    &batch_results));

  ASSERT_TRUE(backend.search_batch({{1.0F, 1.0F}, {2.0F, 2.0F}}, 0,
                                   &batch_results));
  ASSERT_EQ(2U, batch_results.size());
  EXPECT_TRUE(batch_results[0].empty());
  EXPECT_TRUE(batch_results[1].empty());

  ASSERT_TRUE(backend.search_batch({{1.0F, 1.0F}, {2.0F, 2.0F}}, 1,
                                   &batch_results));
  ASSERT_EQ(2U, batch_results.size());
  ASSERT_EQ(1U, batch_results[0].size());
  ASSERT_EQ(1U, batch_results[1].size());
  EXPECT_EQ(1U, batch_results[0][0].doc_id);
  EXPECT_EQ(2U, batch_results[1][0].doc_id);
}

TEST(VectorIndexBackendTest,
     FaissExternalSearchesSnapshotEntriesAfterNativeIndexRelease) {
  BoolGuard keep_loaded_guard(&opt_vector_faiss_keep_loaded, false);
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_released_search_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend euclidean(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal, "idx_faiss_released_l2");
  ASSERT_TRUE(euclidean.upsert(2, {2.0F, 0.0F}));
  ASSERT_TRUE(euclidean.upsert(1, {1.0F, 0.0F}));

  std::vector<vector_index::search_result> result{{99, 99.0}};
  EXPECT_TRUE(euclidean.search({1.0F, 0.0F}, 0, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_FALSE(euclidean.search({1.0F}, 1, &result));
  ASSERT_TRUE(euclidean.search({1.0F, 0.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  UlongGuard faiss_threads_guard(&opt_vector_faiss_search_threads, 2);
  std::vector<std::vector<vector_index::search_result>> batch_results;
  ASSERT_TRUE(euclidean.search_batch({{1.0F, 0.0F}, {2.0F, 0.0F}}, 1,
                                     &batch_results));
  ASSERT_EQ(2U, batch_results.size());
  ASSERT_EQ(1U, batch_results[0].size());
  ASSERT_EQ(1U, batch_results[1].size());
  EXPECT_EQ(1U, batch_results[0][0].doc_id);
  EXPECT_EQ(2U, batch_results[1][0].doc_id);

  vector_index::faiss_backend inner_product(
      2, vector_index::metric_type::kInnerProduct,
      vector_index::backend_mode::kExternal, "idx_faiss_released_ip");
  ASSERT_TRUE(inner_product.upsert(1, {1.0F, 0.0F}));
  ASSERT_TRUE(inner_product.upsert(2, {0.0F, 1.0F}));
  ASSERT_TRUE(inner_product.search({1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  vector_index::faiss_backend cosine(
      2, vector_index::metric_type::kCosine,
      vector_index::backend_mode::kExternal, "idx_faiss_released_cos");
  ASSERT_TRUE(cosine.upsert(1, {1.0F, 0.0F}));
  ASSERT_TRUE(cosine.upsert(2, {0.0F, 1.0F}));
  ASSERT_TRUE(cosine.search({1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest, FaissExternalClearsSnapshotWhenNativeIndexStaysLoaded) {
#ifndef HAVE_FAISS
  GTEST_SKIP() << "Faiss native snapshot dedup requires HAVE_FAISS";
#else
  BoolGuard keep_loaded_guard(&opt_vector_faiss_keep_loaded, true);

  vector_index::faiss_backend backend(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal, "idx_faiss_snapshot_dedup");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));
  ASSERT_TRUE(backend.upsert(2, {2.0F, 0.0F}));

  EXPECT_TRUE(backend.external_snapshot_entries().empty());
  EXPECT_EQ(2U, backend.entry_count());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
#endif
}

TEST(VectorIndexBackendTest, FaissExternalKeepsSnapshotWhenNativeIndexIsReleased) {
#ifndef HAVE_FAISS
  GTEST_SKIP() << "Faiss native snapshot fallback requires HAVE_FAISS";
#else
  BoolGuard keep_loaded_guard(&opt_vector_faiss_keep_loaded, false);

  vector_index::faiss_backend backend(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal, "idx_faiss_snapshot_fallback");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));
  ASSERT_TRUE(backend.upsert(2, {2.0F, 0.0F}));

  EXPECT_EQ(2U, backend.external_snapshot_entries().size());
  EXPECT_EQ(2U, backend.entry_count());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
#endif
}

TEST(VectorIndexBackendTest, FaissExternalSearchEfCanBeConfigured) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal);

  EXPECT_EQ(64U, backend.search_ef());
  EXPECT_TRUE(backend.set_search_ef(96));
  EXPECT_EQ(96U, backend.search_ef());
  EXPECT_FALSE(backend.set_search_ef(0));
}

TEST(VectorIndexBackendTest, FaissExternalHnswBuildParamsCanBeConfigured) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal);

  EXPECT_EQ(32U, backend.hnsw_m());
  EXPECT_EQ(40U, backend.hnsw_ef_construction());
  EXPECT_TRUE(backend.set_hnsw_build_params(48, 96));
  EXPECT_EQ(48U, backend.hnsw_m());
  EXPECT_EQ(96U, backend.hnsw_ef_construction());
  EXPECT_FALSE(backend.set_hnsw_build_params(0, 96));
  EXPECT_FALSE(backend.set_hnsw_build_params(48, 0));
}

TEST(VectorIndexBackendTest, FaissExternalIvfParamsCanBeConfigured) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal);

  EXPECT_EQ(0U, backend.faiss_nlist());
  EXPECT_EQ(0U, backend.faiss_nprobe());
  EXPECT_TRUE(backend.set_faiss_ivf_params(8, 4));
  EXPECT_EQ(8U, backend.faiss_nlist());
  EXPECT_EQ(4U, backend.faiss_nprobe());
  EXPECT_EQ("ivf_flat", backend.backend_variant());
  EXPECT_TRUE(backend.set_faiss_ivf_params(0, 0));
  EXPECT_EQ(0U, backend.faiss_nlist());
  EXPECT_EQ(0U, backend.faiss_nprobe());
  EXPECT_EQ("hnsw", backend.backend_variant());
}

TEST(VectorIndexBackendTest, FaissExternalIvfParamsRejectPartialZeroConfigs) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal);
  EXPECT_FALSE(backend.set_faiss_ivf_params(0, 1));
  EXPECT_FALSE(backend.set_faiss_ivf_params(1, 0));
}

TEST(VectorIndexBackendTest,
     FaissLiveSearchTuningFailurePreservesPreviousParameters) {
  if (!vector_index::backend_provider_supported(
          vector_index::backend_provider::kFaiss)) {
    GTEST_SKIP() << "Faiss provider is not compiled in";
  }

  vector_index::faiss_backend hnsw_backend(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal, "idx_faiss_ef_atomicity");
  ASSERT_TRUE(hnsw_backend.upsert(1, {1.0F, 0.0F}));
  ASSERT_TRUE(hnsw_backend.set_search_ef(96));
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
                             "+d,vector_backend_fail_faiss_live_search_ef");
    EXPECT_FALSE(hnsw_backend.set_search_ef(128));
  }
  EXPECT_EQ(96U, hnsw_backend.search_ef());

  vector_index::faiss_backend ivf_backend(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal, "idx_faiss_nprobe_atomicity");
  ASSERT_TRUE(ivf_backend.upsert(1, {1.0F, 0.0F}));
  ASSERT_TRUE(ivf_backend.upsert(2, {0.0F, 1.0F}));
  ASSERT_TRUE(ivf_backend.set_faiss_ivf_params(2, 1));
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_faiss_live_nprobe");
    EXPECT_FALSE(ivf_backend.set_faiss_ivf_params(2, 2));
  }
  EXPECT_EQ(1U, ivf_backend.faiss_nprobe());
}

TEST(VectorIndexBackendTest, FaissExternalIvfParamsWithEntriesCanBeConfigured) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_ivf_with_entries");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));
  ASSERT_TRUE(backend.upsert(9, {9.0F, 9.0F}));

  EXPECT_TRUE(backend.set_faiss_ivf_params(2, 1));
  EXPECT_EQ("ivf_flat", backend.backend_variant());
  EXPECT_EQ(2U, backend.faiss_nlist());
  EXPECT_EQ(1U, backend.faiss_nprobe());
}

TEST(VectorIndexBackendTest, FaissExternalIvfTrainingHonorsTrainSizeBudget) {
#ifndef HAVE_FAISS
  GTEST_SKIP() << "Faiss native training budget requires HAVE_FAISS";
#else
  UlonglongGuard guard(&opt_vector_faiss_train_size, 2U * 2U * sizeof(float));
  vector_index::faiss_backend backend(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal, "idx_faiss_ivf_train_budget");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));
  ASSERT_TRUE(backend.upsert(2, {2.0F, 2.0F}));
  ASSERT_TRUE(backend.upsert(3, {3.0F, 3.0F}));
  ASSERT_TRUE(backend.upsert(4, {4.0F, 4.0F}));

  ASSERT_TRUE(backend.set_faiss_ivf_params(2, 1));
  EXPECT_EQ(2U, backend.faiss_last_training_count_for_testing());

  opt_vector_faiss_train_size = 0;
  ASSERT_TRUE(backend.set_faiss_ivf_params(3, 1));
  EXPECT_EQ(4U, backend.faiss_last_training_count_for_testing());
#endif
}

TEST(VectorIndexBackendTest,
     FaissExternalReaderRebuildNonIvfUsesSingleReaderPass) {
#ifndef HAVE_FAISS
  GTEST_SKIP() << "Faiss reader rebuild requires HAVE_FAISS";
#else
  vector_index::faiss_backend backend(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal, "idx_faiss_reader_single_pass");

  const std::vector<std::pair<uint64_t, vector_index::vector_data>> rows{
      {1, {1.0F, 1.0F}},
      {2, {2.0F, 2.0F}},
      {3, {3.0F, 3.0F}}};
  size_t reader_calls = 0;
  ASSERT_TRUE(backend.rebuild_from_committed_entries_from_reader(
      [&rows, &reader_calls](
          const vector_index::committed_entry_visitor &visitor) {
        ++reader_calls;
        for (const auto &row : rows) {
          if (!visitor(row.first, row.second)) return false;
        }
        return true;
      }));
  EXPECT_EQ(1U, reader_calls);
  EXPECT_EQ(0U, backend.faiss_last_training_count_for_testing());
  EXPECT_EQ(rows.size(), backend.entry_count());

  std::vector<std::vector<vector_index::search_result>> batch_results;
  ASSERT_TRUE(backend.search_batch({{1.0F, 1.0F}, {3.0F, 3.0F}}, 1,
                                   &batch_results));
  ASSERT_EQ(2U, batch_results.size());
  ASSERT_EQ(1U, batch_results[0].size());
  ASSERT_EQ(1U, batch_results[1].size());

  size_t bad_reader_calls = 0;
  EXPECT_FALSE(backend.rebuild_from_committed_entries_from_reader(
      [&bad_reader_calls](
          const vector_index::committed_entry_visitor &visitor) {
        ++bad_reader_calls;
        return visitor(9, {9.0F});
      }));
  EXPECT_EQ(1U, bad_reader_calls);
  EXPECT_EQ(rows.size(), backend.entry_count());
  ASSERT_TRUE(backend.search_batch({{1.0F, 1.0F}}, 1, &batch_results));
  ASSERT_EQ(1U, batch_results.size());
  ASSERT_EQ(1U, batch_results[0].size());
  EXPECT_EQ(1U, batch_results[0][0].doc_id);
#endif
}

TEST(VectorIndexBackendTest, FaissExternalIvfPqParamsCanBeConfigured) {
  vector_index::faiss_backend backend(4, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_ivfpq");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F, 0.0F, 0.0F}));
  ASSERT_TRUE(backend.upsert(2, {0.0F, 1.0F, 0.0F, 0.0F}));
  ASSERT_TRUE(backend.upsert(3, {0.0F, 0.0F, 1.0F, 0.0F}));
  ASSERT_TRUE(backend.upsert(4, {0.0F, 0.0F, 0.0F, 1.0F}));

  EXPECT_TRUE(backend.set_faiss_ivf_pq_params(2, 1, 1, 1));
  EXPECT_EQ("ivf_pq", backend.backend_variant());
}

TEST(VectorIndexBackendTest,
     FaissExternalIvfPqRejectsZeroComponentsAndSupportsNonTrainableConfig) {
  vector_index::faiss_backend backend(4, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_ivfpq_small");
  EXPECT_FALSE(backend.set_faiss_ivf_pq_params(0, 1, 1, 1));
  EXPECT_FALSE(backend.set_faiss_ivf_pq_params(2, 0, 1, 1));
  EXPECT_FALSE(backend.set_faiss_ivf_pq_params(2, 1, 0, 1));
  EXPECT_FALSE(backend.set_faiss_ivf_pq_params(2, 1, 1, 0));
  EXPECT_FALSE(backend.set_faiss_ivf_pq_params(2, 1, 3, 8));
  EXPECT_FALSE(backend.set_faiss_ivf_pq_params(
      2, 1, 1, vector_index::k_max_faiss_pq_bits + 1));
  EXPECT_TRUE(backend.set_faiss_ivf_pq_params(
      2, 1, 1, vector_index::k_max_faiss_pq_bits));
  EXPECT_EQ(vector_index::k_max_faiss_pq_bits, backend.faiss_pq_bits());

  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F, 0.0F, 0.0F}));
  EXPECT_TRUE(backend.set_faiss_ivf_pq_params(2, 1, 1, 8));
  EXPECT_EQ("ivf_pq", backend.backend_variant());
}

TEST(VectorIndexBackendTest, FaissExternalSearchRejectsDimensionMismatch) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal);
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));
  EXPECT_FALSE(backend.search({1.0F, 1.0F, 1.0F}, 1, &result));
}

TEST(VectorIndexBackendTest, FaissExternalCosineRejectsZeroNormQuery) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kCosine,
                                     vector_index::backend_mode::kExternal);
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));
  EXPECT_FALSE(backend.search({0.0F, 0.0F}, 1, &result));
}

TEST(VectorIndexBackendTest,
     FaissExternalCosineRejectsZeroNormVectorDuringUpsertAndReload) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kCosine,
                                     vector_index::backend_mode::kExternal);
  EXPECT_FALSE(backend.upsert(1, {0.0F, 0.0F}));
  EXPECT_FALSE(backend.rebuild_from_committed_entries({{1, {0.0F, 0.0F}}}));
}

TEST(VectorIndexBackendTest,
     FaissExternalRebuildRejectsWrongDimensionCommittedEntries) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal);
  EXPECT_FALSE(backend.rebuild_from_committed_entries({{1, {1.0F}},
                                                    {2, {2.0F, 2.0F}}}));
}

TEST(VectorIndexBackendTest,
     FaissExternalCosineRebuildRejectsMixedZeroNormCommittedEntries) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kCosine,
                                     vector_index::backend_mode::kExternal);
  EXPECT_FALSE(backend.rebuild_from_committed_entries({{1, {1.0F, 0.0F}},
                                                    {2, {0.0F, 0.0F}}}));
}

TEST(VectorIndexBackendTest,
     FaissExternalIvfRebuildRejectsWrongDimensionCommittedEntries) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal);
  ASSERT_TRUE(backend.set_faiss_ivf_params(2, 1));
  EXPECT_FALSE(backend.rebuild_from_committed_entries({{1, {1.0F}},
                                                    {2, {2.0F, 2.0F}}}));
}

TEST(VectorIndexBackendTest,
     FaissExternalCosineIvfRebuildRejectsZeroNormCommittedEntries) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kCosine,
                                     vector_index::backend_mode::kExternal);
  ASSERT_TRUE(backend.set_faiss_ivf_params(2, 1));
  EXPECT_FALSE(backend.rebuild_from_committed_entries({{1, {1.0F, 0.0F}},
                                                    {2, {0.0F, 0.0F}}}));
}

TEST(VectorIndexBackendTest, FaissExternalInnerProductSearchUsesDistanceBranch) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kInnerProduct,
                                     vector_index::backend_mode::kExternal);
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));
  ASSERT_TRUE(backend.upsert(2, {0.0F, 1.0F}));
  ASSERT_TRUE(backend.search({1.0F, 0.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexBackendTest,
     FaissExternalUsesDataHomeRootWhenOverrideIsUnset) {
  DataHomeGuard data_home_guard;
  vector_index::reset_faiss_external_snapshot_root_for_testing();

  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_default_root_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  data_home_guard.Set(root);

  {
    vector_index::faiss_backend writer(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_default_root");
    EXPECT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
    EXPECT_TRUE(writer.external_manifest_present());
    EXPECT_GT(writer.external_manifest_generation(), 0U);
  }

  {
    vector_index::faiss_backend reader(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_default_root");
    std::vector<vector_index::search_result> result;
    EXPECT_TRUE(reader.recover());
    EXPECT_TRUE(reader.search({1.0F, 1.0F}, 1, &result));
    ASSERT_EQ(1U, result.size());
    EXPECT_EQ(1U, result[0].doc_id);
  }

  EXPECT_TRUE(vector_index::remove_backend_artifacts(
      "idx_faiss_default_root", vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kFaiss));
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalWithEmptyDataHomeSkipsManifestPersistence) {
  DataHomeGuard data_home_guard;
  vector_index::reset_faiss_external_snapshot_root_for_testing();
  data_home_guard.Set("");

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_empty_home");
  std::vector<vector_index::search_result> result;
  EXPECT_TRUE(backend.upsert(3, {3.0F, 3.0F}));
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_EQ(0U, backend.external_manifest_generation());
  EXPECT_TRUE(backend.search({3.0F, 3.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(3U, result[0].doc_id);
  EXPECT_TRUE(backend.recover());
  EXPECT_TRUE(vector_index::remove_backend_artifacts(
      "idx_faiss_empty_home", vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kFaiss));
}

TEST(VectorIndexBackendTest, FaissExternalRecoverWithoutArtifactsStaysEmpty) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_recover_empty_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_faiss_recover_empty");
  std::vector<vector_index::search_result> result;

  EXPECT_TRUE(backend.recover());
  EXPECT_FALSE(backend.last_recover_used_fallback());
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_EQ(0U, backend.external_manifest_generation());
  EXPECT_TRUE(backend.search({1.0F, 1.0F}, 4, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest, FaissExternalSnapshotPersistsAndRecovers) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_faiss_external_snapshot_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  {
    vector_index::faiss_backend writer(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_external_snapshot");
    std::vector<vector_index::search_result> result;
    EXPECT_TRUE(writer.upsert(101, {1.0F, 1.0F}));
    EXPECT_TRUE(writer.upsert(202, {2.0F, 2.0F}));
    EXPECT_TRUE(writer.erase(202));
    EXPECT_TRUE(writer.external_manifest_present());
    EXPECT_GT(writer.external_manifest_generation(), 0U);
    EXPECT_TRUE(writer.search({1.0F, 1.0F}, 1, &result));
    ASSERT_EQ(1U, result.size());
    EXPECT_EQ(101U, result[0].doc_id);
  }

  {
    vector_index::faiss_backend reader(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_external_snapshot");
    std::vector<vector_index::search_result> result;
    EXPECT_TRUE(reader.recover());
    EXPECT_TRUE(reader.external_manifest_present());
    EXPECT_GT(reader.external_manifest_generation(), 0U);
    EXPECT_EQ(1U, reader.entry_count());
    EXPECT_TRUE(reader.search({1.0F, 1.0F}, 1, &result));
    ASSERT_EQ(1U, result.size());
    EXPECT_EQ(101U, result[0].doc_id);
  }

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest, FaissExternalEraseLastEntryRemovesArtifacts) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_erase_last_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_faiss_erase_last");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  const std::string manifest_path = find_faiss_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string snapshot_path = find_faiss_snapshot_file(
      std::filesystem::path(manifest_path).parent_path().string());
  ASSERT_FALSE(snapshot_path.empty());

  ASSERT_TRUE(backend.erase(1));
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_EQ(0U, backend.external_manifest_generation());
  EXPECT_TRUE(find_faiss_snapshot_file(
                  std::filesystem::path(manifest_path).parent_path().string())
                  .empty());
  EXPECT_FALSE(std::filesystem::exists(manifest_path));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalEraseLastEntryRollbackWhenQuarantineCleanupFails) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_erase_last_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_faiss_erase_last_fail");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  const std::string manifest_path = find_faiss_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  ASSERT_TRUE(std::filesystem::create_directories(manifest_path + ".corrupt", ec));
  ASSERT_FALSE(ec);
  write_binary_file(manifest_path + ".corrupt/blocker", "block");

  EXPECT_FALSE(backend.erase(1));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest, FaissExternalLargeCommittedSnapshotRecoversSearch) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_faiss_external_large_snapshot_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  std::unordered_map<uint64_t, vector_index::vector_data> committed_entries;
  for (uint64_t id = 1; id <= 512; ++id) {
    committed_entries.emplace(
        id, vector_index::vector_data{static_cast<float>(id), static_cast<float>(id)});
  }

  {
    vector_index::faiss_backend writer(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_external_large");
    ASSERT_TRUE(writer.load_committed_entries(committed_entries));
    EXPECT_TRUE(writer.external_manifest_present());
    EXPECT_GT(writer.external_manifest_generation(), 0U);
  }

#ifdef HAVE_FAISS
  {
    const std::string manifest_path = find_faiss_manifest_file(root);
    ASSERT_FALSE(manifest_path.empty());
    const std::string snapshot_path = find_faiss_snapshot_file(
        std::filesystem::path(manifest_path).parent_path().string());
    ASSERT_FALSE(snapshot_path.empty());
    std::unique_ptr<faiss::Index> index(faiss::read_index(snapshot_path.c_str()));
    auto *id_map = dynamic_cast<faiss::IndexIDMap2 *>(index.get());
    ASSERT_NE(nullptr, id_map);
    EXPECT_EQ(nullptr, dynamic_cast<faiss::IndexFlat *>(id_map->index));
    EXPECT_NE(nullptr, dynamic_cast<faiss::IndexHNSW *>(id_map->index));
  }
#endif

  {
    vector_index::faiss_backend reader(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_external_large");
    std::vector<vector_index::search_result> result;
    ASSERT_TRUE(reader.recover());
    ASSERT_TRUE(reader.search({256.0F, 256.0F}, 5, &result));
    ASSERT_FALSE(result.empty());
    EXPECT_EQ(256U, result.front().doc_id);
  }

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalRecoverFallsBackOnMalformedCurrentTextSnapshots) {
  struct MalformedCase {
    const char *suffix;
    const char *payload;
  };
  const MalformedCase cases[] = {
      {"bad_header",
       "mysql-vector-wrong-header\n1\t0000803f0000803f\n"},
      {"missing_tab", "mysql-vector-faiss-external-v1\n1\n"},
      {"bad_docid", "mysql-vector-faiss-external-v1\nbad\t0000803f0000803f\n"},
      {"bad_hex", "mysql-vector-faiss-external-v1\n1\tzz\n"},
      {"misaligned_bytes", "mysql-vector-faiss-external-v1\n1\t00\n"},
      {"wrong_dimension", "mysql-vector-faiss-external-v1\n1\t0000803f\n"},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.suffix);
    const std::string root = std::string(testing::TempDir()) +
                             "/vector_faiss_malformed_text_" + test_case.suffix;
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);
    ASSERT_FALSE(ec);
    vector_index::set_faiss_external_snapshot_root_for_testing(root);

    {
      vector_index::faiss_backend writer(
          2, vector_index::metric_type::kEuclidean,
          vector_index::backend_mode::kExternal, "idx_faiss_malformed_text");
      ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
    }

    const std::string manifest_path = find_faiss_manifest_file(root);
    ASSERT_FALSE(manifest_path.empty());
    const std::string snapshot_path = find_faiss_snapshot_file(
        std::filesystem::path(manifest_path).parent_path().string());
    ASSERT_FALSE(snapshot_path.empty());
    write_binary_file(snapshot_path, test_case.payload);

    vector_index::faiss_backend recovered(
        2, vector_index::metric_type::kEuclidean,
        vector_index::backend_mode::kExternal, "idx_faiss_malformed_text");
    ASSERT_TRUE(recovered.recover());

    std::vector<vector_index::search_result> result;
    ASSERT_TRUE(recovered.search({1.0F, 1.0F}, 4, &result));
    EXPECT_TRUE(result.empty());

    vector_index::reset_faiss_external_snapshot_root_for_testing();
    std::filesystem::remove_all(root, ec);
  }
}

TEST(VectorIndexBackendTest,
     FaissExternalRecoverFallsBackOnWrongDimensionNativeIndexFile) {
#ifndef HAVE_FAISS
  GTEST_SKIP() << "Faiss native recovery requires HAVE_FAISS";
#else
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_faiss_wrong_dim_native_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  {
    vector_index::faiss_backend writer(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_wrong_dim_native");
    ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
  }

  const std::string manifest_path = find_faiss_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string snapshot_path = find_faiss_snapshot_file(
      std::filesystem::path(manifest_path).parent_path().string());
  ASSERT_FALSE(snapshot_path.empty());

  auto *base = new faiss::IndexHNSWFlat(1, 8);
  auto native_index = std::make_unique<faiss::IndexIDMap2>(base);
  const float vector_data[1] = {1.0F};
  const faiss::idx_t doc_id = 1;
  native_index->add_with_ids(1, vector_data, &doc_id);
  write_faiss_index_file(snapshot_path, std::move(native_index));

  const uint64_t fallback_before = vector_status::backend_recover_fallbacks();
  vector_index::faiss_backend recovered(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_faiss_wrong_dim_native");
  ASSERT_TRUE(recovered.recover());
  EXPECT_TRUE(recovered.last_recover_used_fallback());
  EXPECT_EQ(fallback_before + 1, vector_status::backend_recover_fallbacks());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(recovered.search({1.0F, 1.0F}, 2, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
#endif
}

TEST(VectorIndexBackendTest, FaissExternalRecoverLoadsCurrentTextSnapshot) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_faiss_valid_text_snapshot_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  {
    vector_index::faiss_backend writer(
        2, vector_index::metric_type::kEuclidean,
        vector_index::backend_mode::kExternal, "idx_faiss_valid_text_snapshot");
    ASSERT_TRUE(writer.upsert(1, {1.0F, 2.0F}));
  }

  const std::string manifest_path = find_faiss_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string snapshot_path = find_faiss_snapshot_file(
      std::filesystem::path(manifest_path).parent_path().string());
  ASSERT_FALSE(snapshot_path.empty());

  const vector_index::vector_data vector{1.0F, 2.0F};
  write_binary_file(snapshot_path,
                    "mysql-vector-faiss-external-v1\n\n1\t" +
                        hex_encode_vector(vector) + "\n");

  vector_index::faiss_backend recovered(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal, "idx_faiss_valid_text_snapshot");
  ASSERT_TRUE(recovered.recover());
  EXPECT_FALSE(recovered.last_recover_used_fallback());
  ASSERT_EQ(1U, recovered.external_snapshot_entries().size());
  EXPECT_EQ(1U, recovered.external_snapshot_entries().count(1));
  ASSERT_EQ(2U, recovered.external_snapshot_entries().at(1).size());
  EXPECT_FLOAT_EQ(1.0F, recovered.external_snapshot_entries().at(1)[0]);
  EXPECT_FLOAT_EQ(2.0F, recovered.external_snapshot_entries().at(1)[1]);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalRecoverFallsBackOnNonIdMapNativeIndexFile) {
#ifndef HAVE_FAISS
  GTEST_SKIP() << "Faiss native recovery requires HAVE_FAISS";
#else
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_faiss_non_idmap_native_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  {
    vector_index::faiss_backend writer(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_non_idmap_native");
    ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
  }

  const std::string manifest_path = find_faiss_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string snapshot_path = find_faiss_snapshot_file(
      std::filesystem::path(manifest_path).parent_path().string());
  ASSERT_FALSE(snapshot_path.empty());

  auto native_index = std::make_unique<faiss::IndexHNSWFlat>(2, 8);
  const float vector_data[2] = {1.0F, 1.0F};
  native_index->add(1, vector_data);
  write_faiss_index_file(snapshot_path, std::move(native_index));

  vector_index::faiss_backend recovered(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_faiss_non_idmap_native");
  ASSERT_TRUE(recovered.recover());
  EXPECT_TRUE(recovered.last_recover_used_fallback());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(recovered.search({1.0F, 1.0F}, 2, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
#endif
}

TEST(VectorIndexBackendTest,
     FaissExternalRecoverFallsBackOnUnsupportedFlatNativeIndexFile) {
#ifndef HAVE_FAISS
  GTEST_SKIP() << "Faiss native recovery requires HAVE_FAISS";
#else
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_faiss_flat_native_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  {
    vector_index::faiss_backend writer(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_flat_native");
    ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
  }

  const std::string manifest_path = find_faiss_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string snapshot_path = find_faiss_snapshot_file(
      std::filesystem::path(manifest_path).parent_path().string());
  ASSERT_FALSE(snapshot_path.empty());

  auto native_index =
      std::make_unique<faiss::IndexIDMap2>(new faiss::IndexFlatL2(2));
  const float vector_data[2] = {1.0F, 1.0F};
  const faiss::idx_t doc_id = 1;
  native_index->add_with_ids(1, vector_data, &doc_id);
  write_faiss_index_file(snapshot_path, std::move(native_index));

  vector_index::faiss_backend recovered(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_faiss_flat_native");
  ASSERT_TRUE(recovered.recover());
  EXPECT_TRUE(recovered.last_recover_used_fallback());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(recovered.search({1.0F, 1.0F}, 2, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
#endif
}

TEST(VectorIndexBackendTest,
     FaissExternalRecoversIvfFlatNativeIndexAndUpdatesParams) {
#ifndef HAVE_FAISS
  GTEST_SKIP() << "Faiss native recovery requires HAVE_FAISS";
#else
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_ivf_flat_native_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  {
    vector_index::faiss_backend writer(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_ivf_flat_native");
    ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
  }

  const std::string manifest_path = find_faiss_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string snapshot_path = find_faiss_snapshot_file(
      std::filesystem::path(manifest_path).parent_path().string());
  ASSERT_FALSE(snapshot_path.empty());

  auto *quantizer = new faiss::IndexFlatL2(2);
  auto *ivf = new faiss::IndexIVFFlat(quantizer, 2, 2, faiss::METRIC_L2);
  ivf->own_fields = true;
  const float train[8] = {0.0F, 0.0F, 1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F};
  ivf->train(4, train);
  auto native_index = std::make_unique<faiss::IndexIDMap2>(ivf);
  const float vector_data[2] = {3.0F, 3.0F};
  const faiss::idx_t doc_id = 7;
  native_index->add_with_ids(1, vector_data, &doc_id);
  ivf->make_direct_map();
  write_faiss_index_file(snapshot_path, std::move(native_index));

  vector_index::faiss_backend recovered(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_faiss_ivf_flat_native");
  ASSERT_TRUE(recovered.recover());
  EXPECT_FALSE(recovered.last_recover_used_fallback());
  EXPECT_EQ(2U, recovered.faiss_nlist());
  EXPECT_EQ(0U, recovered.faiss_nprobe());
  EXPECT_EQ(0U, recovered.faiss_pq_m());
  EXPECT_EQ(0U, recovered.faiss_pq_bits());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(recovered.search({3.0F, 3.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(7U, result[0].doc_id);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
#endif
}

TEST(VectorIndexBackendTest,
     FaissExternalRecoversIvfPqNativeIndexAndUpdatesParams) {
#ifndef HAVE_FAISS
  GTEST_SKIP() << "Faiss native recovery requires HAVE_FAISS";
#else
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_ivfpq_native_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  {
    vector_index::faiss_backend writer(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_ivfpq_native");
    ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
  }

  const std::string manifest_path = find_faiss_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string snapshot_path = find_faiss_snapshot_file(
      std::filesystem::path(manifest_path).parent_path().string());
  ASSERT_FALSE(snapshot_path.empty());

  auto *quantizer = new faiss::IndexFlatL2(2);
  auto *ivfpq = new faiss::IndexIVFPQ(quantizer, 2, 2, 1, 1);
  ivfpq->own_fields = true;
  const float train[8] = {0.0F, 0.0F, 1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F};
  ivfpq->train(4, train);
  auto native_index = std::make_unique<faiss::IndexIDMap2>(ivfpq);
  const float vector_data[2] = {2.0F, 2.0F};
  const faiss::idx_t doc_id = 9;
  native_index->add_with_ids(1, vector_data, &doc_id);
  ivfpq->make_direct_map();
  write_faiss_index_file(snapshot_path, std::move(native_index));

  vector_index::faiss_backend recovered(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_faiss_ivfpq_native");
  ASSERT_TRUE(recovered.recover());
  EXPECT_FALSE(recovered.last_recover_used_fallback());
  EXPECT_EQ(2U, recovered.faiss_nlist());
  EXPECT_EQ(0U, recovered.faiss_nprobe());
  EXPECT_EQ(1U, recovered.faiss_pq_m());
  EXPECT_EQ(1U, recovered.faiss_pq_bits());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(recovered.search({2.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(9U, result[0].doc_id);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
#endif
}

TEST(VectorIndexBackendTest,
     FaissExternalRecoverFallsBackOnMalformedManifestVariants) {
  struct ManifestCase {
    const char *suffix;
    const char *payload;
  };
  const ManifestCase cases[] = {
      {"bad_header", "mysql-vector-wrong-manifest\n1\n"},
      {"missing_generation", "mysql-vector-faiss-external-manifest-v1\n"},
      {"zero_generation", "mysql-vector-faiss-external-manifest-v1\n0\n"},
      {"bad_generation", "mysql-vector-faiss-external-manifest-v1\nabc\n"},
      {"overflow_generation",
       "mysql-vector-faiss-external-manifest-v1\n18446744073709551616\n"},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.suffix);
    const std::string root =
        std::string(testing::TempDir()) + "/vector_faiss_bad_manifest_" +
        test_case.suffix;
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);
    ASSERT_FALSE(ec);
    vector_index::set_faiss_external_snapshot_root_for_testing(root);

    {
      vector_index::faiss_backend writer(
          2, vector_index::metric_type::kEuclidean,
          vector_index::backend_mode::kExternal, "idx_faiss_bad_manifest");
      ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
    }

    const std::string manifest_path = find_faiss_manifest_file(root);
    ASSERT_FALSE(manifest_path.empty());
    write_binary_file(manifest_path, test_case.payload);

    vector_index::faiss_backend recovered(
        2, vector_index::metric_type::kEuclidean,
        vector_index::backend_mode::kExternal, "idx_faiss_bad_manifest");
    ASSERT_TRUE(recovered.recover());

    std::vector<vector_index::search_result> result;
    ASSERT_TRUE(recovered.search({1.0F, 1.0F}, 4, &result));
    EXPECT_TRUE(result.empty());

    vector_index::reset_faiss_external_snapshot_root_for_testing();
    std::filesystem::remove_all(root, ec);
  }
}

TEST(VectorIndexBackendTest,
     FaissExternalRecoverFallsBackWhenManifestLoadIsInjected) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_manifest_load_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  {
    vector_index::faiss_backend writer(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_manifest_load_fail");
    ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
  }

  vector_index::faiss_backend recovered(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_faiss_manifest_load_fail");
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_load_manifest_generation");
    ASSERT_TRUE(recovered.recover());
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(recovered.search({1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalSecondPersistRecoversFromInjectedManifestLoadFailure) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_faiss_manifest_load_persist_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_manifest_load_persist");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_load_manifest_generation");
    ASSERT_TRUE(backend.upsert(2, {2.0F, 2.0F}));
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({2.0F, 2.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
  EXPECT_EQ(1U, result[1].doc_id);
  EXPECT_TRUE(backend.external_manifest_present());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalEraseLastEntryRemovesCurrentArtifacts) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_cleanup_stale_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  const std::string index_name = "idx_faiss_cleanup_stale";
  const std::string encoded = hex_encode(index_name);
  const std::string snapshot_dir = root + "/" + encoded;
  const std::string manifest_path = snapshot_dir + "/faiss_external.manifest.v1";
  const std::string manifest_quarantine = manifest_path + ".corrupt";

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     index_name);
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  std::filesystem::create_directories(snapshot_dir, ec);
  ASSERT_FALSE(ec);
  ec.clear();
  write_binary_file(manifest_quarantine, "stale-manifest-corrupt");

  ASSERT_TRUE(backend.erase(1));
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_FALSE(std::filesystem::exists(manifest_path));
  EXPECT_FALSE(std::filesystem::exists(manifest_quarantine));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 2, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalEraseLastEntryRemovesCurrentArtifacts) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_cleanup_stale_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  const std::string index_name = "idx_diskann_cleanup_stale";
  const std::string encoded = hex_encode(index_name);
  const std::string snapshot_dir = root + "/" + encoded;
  const std::string diskann_manifest = snapshot_dir + "/diskann_external.manifest.v1";
  const std::string diskann_manifest_quarantine = diskann_manifest + ".corrupt";

  vector_index::diskann_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kExternal,
                                       index_name);
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  std::filesystem::create_directories(snapshot_dir, ec);
  ASSERT_FALSE(ec);
  ec.clear();
  write_binary_file(diskann_manifest_quarantine, "stale-diskann-manifest-corrupt");

  ASSERT_TRUE(backend.erase(1));
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_FALSE(std::filesystem::exists(diskann_manifest));
  EXPECT_FALSE(std::filesystem::exists(diskann_manifest_quarantine));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 2, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest, RemoveBackendArtifactsClearsFaissExternalSnapshot) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_faiss_external_cleanup_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_cleanup");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  const std::string snapshot_path =
      root + "/6964785f66616973735f636c65616e7570/faiss_external.snapshot.1.v1";
  const std::string manifest_path =
      root + "/6964785f66616973735f636c65616e7570/faiss_external.manifest.v1";
  EXPECT_TRUE(std::filesystem::exists(snapshot_path));
  EXPECT_TRUE(std::filesystem::exists(manifest_path));
  EXPECT_TRUE(vector_index::remove_backend_artifacts(
      "idx_faiss_cleanup", vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kFaiss));
  EXPECT_FALSE(std::filesystem::exists(snapshot_path));
  EXPECT_FALSE(std::filesystem::exists(manifest_path));

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest, RemoveBackendArtifactsClearsDiskAnnExternalSnapshot) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_diskann_external_cleanup_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::diskann_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kExternal,
                                       "idx_diskann_cleanup");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  const std::string manifest_path = find_diskann_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string store_path =
      std::filesystem::path(manifest_path).parent_path().string() +
      "/diskann_external.store";
  const std::string snapshot_path = find_diskann_snapshot_file(
      std::filesystem::path(manifest_path).parent_path().string());
  ASSERT_FALSE(snapshot_path.empty());
  EXPECT_TRUE(std::filesystem::exists(snapshot_path));
  EXPECT_TRUE(std::filesystem::exists(manifest_path));
  EXPECT_TRUE(vector_index::remove_backend_artifacts(
      "idx_diskann_cleanup", vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kDiskAnn));
  EXPECT_FALSE(std::filesystem::exists(snapshot_path));
  EXPECT_FALSE(std::filesystem::exists(manifest_path));
  EXPECT_FALSE(std::filesystem::exists(store_path));

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     RemoveBackendArtifactsFailsWhenGeneratedSnapshotCleanupIsInjected) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_faiss_external_cleanup_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_cleanup_fail");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  const std::string snapshot_directory =
      root + "/6964785f66616973735f636c65616e75705f6661696c";
  ASSERT_FALSE(find_faiss_snapshot_file(snapshot_directory).empty());
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_remove_generated_snapshots");
    EXPECT_FALSE(vector_index::remove_backend_artifacts(
        "idx_faiss_cleanup_fail", vector_index::backend_mode::kExternal,
        vector_index::backend_provider::kFaiss));
  }

  EXPECT_FALSE(find_faiss_snapshot_file(snapshot_directory).empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     RemoveBackendArtifactsFailsWhenArtifactRemovalIsInjected) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_faiss_external_remove_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_remove_fail");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  const std::string snapshot_directory =
      root + "/6964785f66616973735f72656d6f76655f6661696c";
  ASSERT_FALSE(find_faiss_manifest_file(root).empty());
  ASSERT_FALSE(find_faiss_snapshot_file(snapshot_directory).empty());
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_remove_if_exists");
    EXPECT_FALSE(vector_index::remove_backend_artifacts(
        "idx_faiss_remove_fail", vector_index::backend_mode::kExternal,
        vector_index::backend_provider::kFaiss));
  }

  EXPECT_FALSE(find_faiss_manifest_file(root).empty());
  EXPECT_FALSE(find_faiss_snapshot_file(snapshot_directory).empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     RemoveDiskAnnBackendArtifactsFailsWhenGeneratedSnapshotCleanupIsInjected) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_diskann_external_cleanup_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::diskann_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kExternal,
                                       "idx_diskann_cleanup_fail");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  const std::string manifest_path = find_diskann_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string snapshot_directory =
      std::filesystem::path(manifest_path).parent_path().string();
  ASSERT_FALSE(find_diskann_snapshot_file(snapshot_directory).empty());
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_remove_generated_snapshots");
    EXPECT_FALSE(vector_index::remove_backend_artifacts(
        "idx_diskann_cleanup_fail", vector_index::backend_mode::kExternal,
        vector_index::backend_provider::kDiskAnn));
  }

  EXPECT_FALSE(find_diskann_snapshot_file(snapshot_directory).empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     RemoveDiskAnnBackendArtifactsFailsWhenDirectoryCleanupIsInjected) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_diskann_external_dir_cleanup_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::diskann_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kExternal,
                                       "idx_diskann_dir_cleanup_fail");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  const std::string manifest_path = find_diskann_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string snapshot_directory =
      std::filesystem::path(manifest_path).parent_path().string();
  ASSERT_FALSE(find_diskann_snapshot_file(snapshot_directory).empty());
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_remove_dir_if_empty");
    EXPECT_FALSE(vector_index::remove_backend_artifacts(
        "idx_diskann_dir_cleanup_fail", vector_index::backend_mode::kExternal,
        vector_index::backend_provider::kDiskAnn));
  }

  EXPECT_TRUE(std::filesystem::exists(snapshot_directory));

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest, CreateBackendFactoryReturnsExpectedImplementations) {
#ifndef NDEBUG
  auto native_mem = vector_index::create_backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kNative);
  ASSERT_NE(nullptr, native_mem);
  EXPECT_EQ(vector_index::backend_provider::kNative, native_mem->provider());
  EXPECT_TRUE(native_mem->supports_mutations());
#else
  vector_gunit::NativeProviderGuard native_provider_guard(false);
  EXPECT_EQ(nullptr, vector_index::create_backend(
                         2, vector_index::metric_type::kEuclidean,
                         vector_index::backend_mode::kMemory,
                         vector_index::backend_provider::kNative));
#endif

  auto faiss_ext = vector_index::create_backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kFaiss);
  ASSERT_NE(nullptr, faiss_ext);
  EXPECT_EQ(vector_index::backend_provider::kFaiss, faiss_ext->provider());
  EXPECT_TRUE(faiss_ext->supports_mutations());

  auto diskann_ext = vector_index::create_backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kDiskAnn);
  ASSERT_NE(nullptr, diskann_ext);
  EXPECT_EQ(vector_index::backend_provider::kDiskAnn, diskann_ext->provider());
  EXPECT_TRUE(diskann_ext->supports_mutations());

  auto hnsw_mem = vector_index::create_backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kHnswlib);
  ASSERT_NE(nullptr, hnsw_mem);
  EXPECT_EQ(vector_index::backend_provider::kHnswlib, hnsw_mem->provider());
  EXPECT_TRUE(hnsw_mem->supports_mutations());

  auto diskann_mem = vector_index::create_backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kDiskAnn);
  EXPECT_EQ(nullptr, diskann_mem);
}

TEST(VectorIndexBackendTest,
     CreateBackendFactoryRoutesStandaloneAnnProvidersToNativeRuntime) {
  const vector_index::backend_provider providers[] = {
      vector_index::backend_provider::kFaiss,
      vector_index::backend_provider::kDiskAnn,
      vector_index::backend_provider::kHnswlib};

  for (const auto provider : providers) {
    auto backend = vector_index::create_backend(
        2, vector_index::metric_type::kEuclidean,
        provider == vector_index::backend_provider::kDiskAnn
            ? vector_index::backend_mode::kExternal
            : vector_index::backend_mode::kMemory,
        provider, vector_index::index_consistency_mode::kStandalone,
        "standalone_native_runtime");
    if (!vector_index::backend_provider_supported(provider)) {
      EXPECT_EQ(nullptr, backend);
      continue;
    }
    ASSERT_NE(nullptr, backend);
    EXPECT_EQ(provider, backend->provider());
    EXPECT_TRUE(backend->supports_mutations());
    EXPECT_FALSE(backend->backend_variant().empty());
  }
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalLoadsCommittedSnapshotForMutableSearchAndPersistsSidecar) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_external_snapshot_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::diskann_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kExternal,
                                       "idx_diskann_ext");
  std::vector<vector_index::search_result> result;
  std::unordered_map<uint64_t, vector_index::vector_data> committed_entries;
  committed_entries.emplace(9, vector_index::vector_data{9.0F, 9.0F});
  committed_entries.emplace(1, vector_index::vector_data{1.0F, 1.0F});

  EXPECT_EQ(vector_index::backend_provider::kDiskAnn, backend.provider());
  EXPECT_TRUE(backend.supports_mutations());
  EXPECT_TRUE(backend.upsert(1, {1.0F, 2.0F}));
  EXPECT_EQ(1U, backend.entry_count());
  EXPECT_TRUE(backend.erase(1));
  EXPECT_EQ(0U, backend.entry_count());

  EXPECT_TRUE(backend.load_committed_entries(committed_entries));
  EXPECT_EQ(2U, backend.entry_count());
  EXPECT_TRUE(backend.external_manifest_present());
  EXPECT_GT(backend.external_manifest_generation(), 0U);
  EXPECT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  vector_index::diskann_backend recovered(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_ext");
  ASSERT_TRUE(recovered.recover());
  EXPECT_FALSE(recovered.last_recover_used_fallback());
  EXPECT_TRUE(recovered.external_manifest_present());
  EXPECT_GT(recovered.external_manifest_generation(), 0U);
  EXPECT_TRUE(recovered.search({1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  EXPECT_FALSE(backend.load_committed_entries(
      {{3, vector_index::vector_data{1.0F, 2.0F, 3.0F}}}));

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
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

TEST(VectorIndexBackendTest, HnswlibMemoryBuildParamRejectsNonEmptyIndex) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  std::vector<vector_index::search_result> result;

  ASSERT_TRUE(backend.upsert(10, {1.0F, 1.0F}));
  ASSERT_TRUE(backend.set_search_ef(32));
  EXPECT_FALSE(backend.set_hnsw_build_params(12, 96));
  EXPECT_EQ(16U, backend.hnsw_m());
  EXPECT_EQ(200U, backend.hnsw_ef_construction());
  EXPECT_EQ(32U, backend.search_ef());
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);
}

TEST(VectorIndexBackendTest, HnswlibMemoryRejectsIndexMemoryLimit) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native tuning requires HAVE_HNSWLIB";
  }
  UlonglongGuard guard(&opt_vector_hnsw_index_memory_size, 1);
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);

  EXPECT_FALSE(backend.upsert(1, {1.0F, 0.0F}));
  EXPECT_EQ(0U, backend.entry_count());
  EXPECT_FALSE(backend.rebuild_from_committed_entries(
      {{1, vector_index::vector_data{1.0F, 0.0F}}}));
}

TEST(VectorIndexBackendTest,
     HnswlibMemoryEmptySnapshotRebuildClearsServingState) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native rebuild requires HAVE_HNSWLIB";
  }
  UlonglongGuard guard(&opt_vector_hnsw_index_memory_size, 1024 * 1024);
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));

  ASSERT_TRUE(backend.rebuild_from_committed_entries({}));
  EXPECT_EQ(0U, backend.entry_count());

  const auto diagnostics = backend.build_diagnostics();
  EXPECT_EQ("entries", diagnostics.input_source);
  EXPECT_EQ(0U, diagnostics.row_count);

  std::vector<vector_index::search_result> results{{1, 0.0}};
  ASSERT_TRUE(backend.search({1.0F, 0.0F}, 1, &results));
  EXPECT_TRUE(results.empty());
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
  const vector_index::committed_entry_reader committed_reader =
      [](const vector_index::committed_entry_visitor &visitor) {
        return visitor(1, {1.0F, 0.0F});
      };
  const vector_index::raw_vector_segment_reader raw_reader =
      [](const vector_index::raw_vector_segment_visitor &) { return true; };

  EXPECT_FALSE(backend.set_hnsw_build_threads(1));
  EXPECT_EQ(0U, backend.hnsw_build_threads());
  EXPECT_TRUE(backend.rebuild_from_committed_entries({}));
  EXPECT_FALSE(backend.rebuild_from_committed_entries({{1, {1.0F, 0.0F}}}));
  EXPECT_FALSE(
      backend.rebuild_from_committed_entries_from_reader(committed_reader));
  EXPECT_FALSE(backend.rebuild_from_raw_segments(raw_reader));
}

TEST(VectorIndexBackendTest,
     HnswlibFailedLiveUpsertDoesNotPublishBookkeepingLabel) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native upsert requires HAVE_HNSWLIB";
  }
  vector_index::hnswlib_backend backend(2,
                                        vector_index::metric_type::kEuclidean,
                                        vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_backend_fail_hnswlib_upsert_after_label_insert");
    EXPECT_FALSE(backend.upsert(2, {2.0F, 0.0F}));
  }

  EXPECT_EQ(1U, backend.entry_count());
  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(backend.search({2.0F, 0.0F}, 2, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(1U, results.front().doc_id);

  ASSERT_TRUE(backend.upsert(2, {2.0F, 0.0F}));
  EXPECT_EQ(2U, backend.entry_count());
  ASSERT_TRUE(backend.search({2.0F, 0.0F}, 1, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(2U, results.front().doc_id);
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

TEST(VectorIndexBackendTest,
     HnswlibMemoryReaderRebuildStreamsCommittedEntries) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native reader rebuild requires HAVE_HNSWLIB";
  }
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  size_t visits = 0;
  const vector_index::committed_entry_reader reader =
      [&visits](const vector_index::committed_entry_visitor &visitor) {
        ++visits;
        return visitor(2, {2.0F, 0.0F}) && visitor(1, {1.0F, 0.0F});
      };

  ASSERT_TRUE(backend.rebuild_from_committed_entries_from_reader(reader));
  EXPECT_EQ(1U, visits);
  EXPECT_EQ(2U, backend.entry_count());
  const auto diagnostics = backend.build_diagnostics();
  EXPECT_EQ("hnsw_scheduler", diagnostics.runtime);
  EXPECT_EQ("reader", diagnostics.input_source);
  EXPECT_EQ(2U, diagnostics.row_count);
  EXPECT_EQ(1U, diagnostics.segment_count);
  EXPECT_EQ(1U, diagnostics.build_invocations);
  EXPECT_EQ(1U, diagnostics.concurrent_build_tasks);
  EXPECT_GE(diagnostics.effective_build_threads, 1U);
  EXPECT_EQ(1U, diagnostics.effective_blas_threads);
  EXPECT_EQ(0U, diagnostics.pq_chunks);
  EXPECT_EQ(0U, diagnostics.cache_nodes);

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexBackendTest,
     HnswlibMemoryRawBlockRebuildUsesAllSegmentsAndKeepsOldStateOnFailure) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native raw rebuild requires HAVE_HNSWLIB";
  }

  const ScopedTempDirectory root("vector_hnsw_raw_blocks");
  ASSERT_TRUE(root.valid()) << root.error();
  const std::string vector_path_1 = (root.path() / "vectors-1.fbin").string();
  const std::string docid_path_1 = (root.path() / "docids-1.u64").string();
  const std::string vector_path_2 = (root.path() / "vectors-2.fbin").string();
  const std::string docid_path_2 = (root.path() / "docids-2.u64").string();
  write_raw_fbin_file(vector_path_1, 3, 2,
                      {1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F});
  write_raw_docid_file(docid_path_1, {10, 20, 30});
  write_raw_fbin_file(vector_path_2, 2, 2, {4.0F, 0.0F, 5.0F, 0.0F});
  write_raw_docid_file(docid_path_2, {40, 50});

  const vector_index::raw_vector_segment segment_1{
      vector_path_1, docid_path_1, 3, 2, 0, 1};
  const vector_index::raw_vector_segment segment_2{
      vector_path_2, docid_path_2, 2, 2, 0, 2};
  const vector_index::raw_vector_segment_reader reader =
      [&segment_1,
       &segment_2](const vector_index::raw_vector_segment_visitor &visitor) {
        return visitor(segment_1) && visitor(segment_2);
      };

  vector_index::hnswlib_backend backend(2,
                                        vector_index::metric_type::kEuclidean,
                                        vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.set_hnsw_build_threads(2));
  EXPECT_FALSE(backend.rebuild_from_raw_segments(nullptr));
  EXPECT_FALSE(backend.rebuild_from_raw_segments(
      [](const vector_index::raw_vector_segment_visitor &) { return false; }));
  vector_index::raw_vector_segment wrong_dimension = segment_1;
  wrong_dimension.dimension = 3;
  EXPECT_FALSE(backend.rebuild_from_raw_segments(
      [&wrong_dimension](
          const vector_index::raw_vector_segment_visitor &visitor) {
        return visitor(wrong_dimension);
      }));
  ASSERT_TRUE(backend.rebuild_from_raw_segments(reader));
  EXPECT_EQ(5U, backend.entry_count());

  auto diagnostics = backend.build_diagnostics();
  EXPECT_EQ("hnsw_scheduler", diagnostics.runtime);
  EXPECT_EQ("raw_blocks", diagnostics.input_source);
  EXPECT_EQ(5U, diagnostics.row_count);
  EXPECT_EQ(2U, diagnostics.segment_count);
  EXPECT_EQ(1U, diagnostics.reader_passes);

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(backend.search({5.0F, 0.0F}, 1, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(50U, results[0].doc_id);

  vector_index::raw_vector_segment bad_segment = segment_2;
  bad_segment.row_count = 3;
  EXPECT_FALSE(backend.rebuild_from_raw_segments(
      [&bad_segment](const vector_index::raw_vector_segment_visitor &visitor) {
        return visitor(bad_segment);
      }));
  EXPECT_EQ(5U, backend.entry_count());
  ASSERT_TRUE(backend.search({5.0F, 0.0F}, 1, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(50U, results[0].doc_id);

  const std::string duplicate_vector_path =
      (root.path() / "vectors-duplicate.fbin").string();
  const std::string duplicate_docid_path =
      (root.path() / "docids-duplicate.u64").string();
  write_raw_fbin_file(duplicate_vector_path, 2, 2,
                      {6.0F, 0.0F, 7.0F, 0.0F});
  write_raw_docid_file(duplicate_docid_path, {60, 60});
  const vector_index::raw_vector_segment duplicate_segment{
      duplicate_vector_path, duplicate_docid_path, 2, 2, 0, 3};
  EXPECT_FALSE(backend.rebuild_from_raw_segments(
      [&duplicate_segment](
          const vector_index::raw_vector_segment_visitor &visitor) {
        return visitor(duplicate_segment);
      }));
  EXPECT_FALSE(backend.rebuild_from_raw_segments(
      [&segment_1](const vector_index::raw_vector_segment_visitor &visitor) {
        return visitor(segment_1) && visitor(segment_1);
      }));
  {
    UlonglongGuard memory_guard(&opt_vector_hnsw_index_memory_size, 1);
    EXPECT_FALSE(backend.rebuild_from_raw_segments(reader));
  }
  EXPECT_EQ(5U, backend.entry_count());
  ASSERT_TRUE(backend.search({5.0F, 0.0F}, 1, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(50U, results[0].doc_id);
}

TEST(VectorIndexBackendTest,
     HnswlibMemoryRawBlockRebuildNormalizesCosineWithTinyCache) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native raw rebuild requires HAVE_HNSWLIB";
  }

  UlonglongGuard cache_guard(&opt_vector_entry_cache_size, 1);
  const ScopedTempDirectory root("vector_hnsw_raw_cosine");
  ASSERT_TRUE(root.valid()) << root.error();
  const std::string vector_path = (root.path() / "vectors.fbin").string();
  const std::string docid_path = (root.path() / "docids.u64").string();
  write_raw_fbin_file(vector_path, 2, 2, {3.0F, 4.0F, 0.0F, 0.0F});
  write_raw_docid_file(docid_path, {70, 80});
  const vector_index::raw_vector_segment segment{vector_path, docid_path, 2,
                                                  2,           0,          1};

  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kCosine,
                                        vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.set_hnsw_build_threads(2));
  ASSERT_TRUE(backend.rebuild_from_raw_segments(
      [&segment](const vector_index::raw_vector_segment_visitor &visitor) {
        return visitor(segment);
      }));
  EXPECT_EQ(2U, backend.entry_count());

  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(backend.search({6.0F, 8.0F}, 1, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(70U, results[0].doc_id);
}

TEST(VectorIndexBackendTest,
     HnswlibMemoryRawBlockPreparationAndWorkerFailuresKeepServingState) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native raw rebuild requires HAVE_HNSWLIB";
  }

  const ScopedTempDirectory root("vector_hnsw_raw_worker_failure");
  ASSERT_TRUE(root.valid()) << root.error();
  const std::string vector_path = (root.path() / "vectors.fbin").string();
  const std::string docid_path = (root.path() / "docids.u64").string();
  write_raw_fbin_file(vector_path, 2, 2, {2.0F, 0.0F, 3.0F, 0.0F});
  write_raw_docid_file(docid_path, {2, 3});
  const vector_index::raw_vector_segment segment{vector_path, docid_path, 2,
                                                  2,           0,          1};

  vector_index::hnswlib_backend backend(2,
                                        vector_index::metric_type::kEuclidean,
                                        vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.set_hnsw_build_threads(2));
  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));
  const auto rebuild = [&]() {
    return backend.rebuild_from_raw_segments(
        [&segment](const vector_index::raw_vector_segment_visitor &visitor) {
          return visitor(segment);
        });
  };
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_backend_fail_hnswlib_label_preparation");
    EXPECT_FALSE(rebuild());
  }
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_backend_fail_hnswlib_parallel_rebuild_worker");
    EXPECT_FALSE(rebuild());
  }

  EXPECT_EQ(1U, backend.entry_count());
  std::vector<vector_index::search_result> results;
  ASSERT_TRUE(backend.search({1.0F, 0.0F}, 1, &results));
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(1U, results[0].doc_id);
}

TEST(VectorIndexBackendTest,
     HnswlibNativeReaderRebuildDoesNotKeepExactFallbackEntries) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native reader rebuild requires HAVE_HNSWLIB";
  }
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  const vector_index::committed_entry_reader reader =
      [](const vector_index::committed_entry_visitor &visitor) {
        return visitor(1, {1.0F, 0.0F}) && visitor(2, {0.0F, 1.0F});
      };

  ASSERT_TRUE(backend.rebuild_from_committed_entries_from_reader(reader));
  EXPECT_TRUE(backend.hnsw_native_available_for_testing());
  EXPECT_FALSE(backend.hnsw_exact_fallback_active_for_testing());
  EXPECT_EQ(0U, backend.hnsw_exact_fallback_entry_count_for_testing());
  EXPECT_EQ(2U, backend.entry_count());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexBackendTest,
     HnswlibMemoryReaderRebuildFailureKeepsExistingServingState) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native reader rebuild requires HAVE_HNSWLIB";
  }
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));
  const vector_index::committed_entry_reader bad_reader =
      [](const vector_index::committed_entry_visitor &visitor) {
        return visitor(2, {2.0F, 0.0F, 1.0F});
      };

  EXPECT_FALSE(backend.rebuild_from_committed_entries_from_reader(bad_reader));
  EXPECT_EQ(1U, backend.entry_count());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 0.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexBackendTest,
     HnswlibMemoryReaderRebuildRejectsNullAndMemoryBudgetOverflow) {
  if (!hnswlib_tuning_supported()) {
    GTEST_SKIP() << "hnswlib native reader rebuild requires HAVE_HNSWLIB";
  }
  vector_index::hnswlib_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory);
  ASSERT_TRUE(backend.upsert(1, {1.0F, 0.0F}));

  EXPECT_FALSE(backend.rebuild_from_committed_entries_from_reader(nullptr));

  const vector_index::committed_entry_reader reader =
      [](const vector_index::committed_entry_visitor &visitor) {
        return visitor(2, {2.0F, 0.0F});
      };
  {
    UlonglongGuard guard(&opt_vector_hnsw_index_memory_size, 1);
    EXPECT_FALSE(backend.rebuild_from_committed_entries_from_reader(reader));
  }

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

TEST(VectorIndexBackendTest, parse_metricAcceptsAliasesAndTrimsSpace) {
  vector_index::metric_type metric = vector_index::metric_type::kEuclidean;
  EXPECT_TRUE(vector_index::parse_metric("euclidean", &metric));
  EXPECT_EQ(vector_index::metric_type::kEuclidean, metric);
  EXPECT_TRUE(vector_index::parse_metric(" cosine ", &metric));
  EXPECT_EQ(vector_index::metric_type::kCosine, metric);
  EXPECT_TRUE(vector_index::parse_metric("IP", &metric));
  EXPECT_EQ(vector_index::metric_type::kInnerProduct, metric);
  EXPECT_TRUE(vector_index::parse_metric("inner_product", &metric));
  EXPECT_EQ(vector_index::metric_type::kInnerProduct, metric);
  EXPECT_TRUE(vector_index::parse_metric("l2", &metric));
  EXPECT_EQ(vector_index::metric_type::kEuclidean, metric);
  EXPECT_FALSE(vector_index::parse_metric("manhattan", &metric));
  EXPECT_FALSE(vector_index::parse_metric("cosine", nullptr));
}

TEST(VectorIndexBackendTest, parse_backend_modeAcceptsMemoryAndExternal) {
  vector_index::backend_mode mode = vector_index::backend_mode::kMemory;
  EXPECT_TRUE(vector_index::parse_backend_mode("EXTERNAL", &mode));
  EXPECT_EQ(vector_index::backend_mode::kExternal, mode);
  EXPECT_TRUE(vector_index::parse_backend_mode(" memory ", &mode));
  EXPECT_EQ(vector_index::backend_mode::kMemory, mode);
  EXPECT_FALSE(vector_index::parse_backend_mode("disk", &mode));
  EXPECT_FALSE(vector_index::parse_backend_mode("memory", nullptr));
}

TEST(VectorIndexBackendTest, parse_backend_providerAcceptsAliases) {
  vector_index::backend_provider provider = vector_index::backend_provider::kNative;
  EXPECT_TRUE(vector_index::parse_backend_provider("native", &provider));
  EXPECT_EQ(vector_index::backend_provider::kNative, provider);
  EXPECT_TRUE(vector_index::parse_backend_provider("faiss", &provider));
  EXPECT_EQ(vector_index::backend_provider::kFaiss, provider);
  EXPECT_TRUE(vector_index::parse_backend_provider("DISKANN", &provider));
  EXPECT_EQ(vector_index::backend_provider::kDiskAnn, provider);
  EXPECT_TRUE(vector_index::parse_backend_provider(" hnsw ", &provider));
  EXPECT_EQ(vector_index::backend_provider::kHnswlib, provider);
  EXPECT_FALSE(vector_index::parse_backend_provider("nmslib", &provider));
  EXPECT_FALSE(vector_index::parse_backend_provider("native", nullptr));
}

TEST(VectorIndexBackendTest, EnumToStringHelpersReturnCanonicalTokens) {
  EXPECT_STREQ("euclidean",
               vector_index::metric_to_string(vector_index::metric_type::kEuclidean));
  EXPECT_STREQ("cosine",
               vector_index::metric_to_string(vector_index::metric_type::kCosine));
  EXPECT_STREQ("inner_product",
               vector_index::metric_to_string(vector_index::metric_type::kInnerProduct));

  EXPECT_STREQ(
      "memory",
      vector_index::backend_mode_to_string(vector_index::backend_mode::kMemory));
  EXPECT_STREQ(
      "external",
      vector_index::backend_mode_to_string(vector_index::backend_mode::kExternal));

  EXPECT_STREQ("native", vector_index::backend_provider_to_string(
                             vector_index::backend_provider::kNative));
  EXPECT_STREQ("faiss", vector_index::backend_provider_to_string(
                            vector_index::backend_provider::kFaiss));
  EXPECT_STREQ("diskann", vector_index::backend_provider_to_string(
                              vector_index::backend_provider::kDiskAnn));
  EXPECT_STREQ("hnswlib", vector_index::backend_provider_to_string(
                              vector_index::backend_provider::kHnswlib));
}

TEST(VectorIndexBackendTest, EnumToStringHelpersHandleUnknownValues) {
  EXPECT_STREQ(
      "unknown",
      vector_index::metric_to_string(static_cast<vector_index::metric_type>(999)));
  EXPECT_STREQ("unknown", vector_index::backend_mode_to_string(
                              static_cast<vector_index::backend_mode>(999)));
  EXPECT_STREQ("unknown", vector_index::backend_provider_to_string(
                              static_cast<vector_index::backend_provider>(999)));
}

TEST(VectorIndexBackendTest, CreateBackendFactoryCreatesNativeExternal) {
#ifdef NDEBUG
  GTEST_SKIP() << "native provider is available only in debug builds";
#else
  auto native_external = vector_index::create_backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kNative);
  ASSERT_NE(nullptr, native_external);
  EXPECT_EQ(vector_index::backend_provider::kNative, native_external->provider());
  EXPECT_TRUE(native_external->supports_mutations());
#endif
}

TEST(VectorIndexBackendTest,
     FaissExternalPersistRollbackWhenSnapshotRootIsBlockedByFile) {
  const std::string blocked_root =
      std::string(testing::TempDir()) + "/vector_faiss_blocked_root_file";
  std::error_code ec;
  std::filesystem::remove(blocked_root, ec);
  std::ofstream blocker(blocked_root,
                        std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(blocker.good());
  blocker << "blocked";
  blocker.close();
  ASSERT_TRUE(blocker);

  vector_index::set_faiss_external_snapshot_root_for_testing(blocked_root);
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_blocked_root");

  EXPECT_FALSE(backend.upsert(1, {1.0F, 1.0F}));
  EXPECT_EQ(0U, backend.entry_count());
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_EQ(0U, backend.external_manifest_generation());

  std::vector<vector_index::search_result> result;
  EXPECT_TRUE(backend.search({1.0F, 1.0F}, 5, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove(blocked_root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalPersistRollbackWhenNativeSnapshotWriteIsInjected) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_native_write_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_native_write_fail");
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_persist_faiss_index_file");
    EXPECT_FALSE(backend.upsert(1, {1.0F, 1.0F}));
  }

  EXPECT_EQ(0U, backend.entry_count());
  EXPECT_TRUE(find_faiss_manifest_file(root).empty());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalPersistRollbackWhenNativeSnapshotRenameIsInjected) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_native_rename_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_native_rename_fail");
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_persist_faiss_index_rename");
    EXPECT_FALSE(backend.upsert(1, {1.0F, 1.0F}));
  }

  EXPECT_EQ(0U, backend.entry_count());
  EXPECT_TRUE(find_faiss_manifest_file(root).empty());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalLoadCommittedEntriesFailsWhenPersistUnavailable) {
  const std::string blocked_root =
      std::string(testing::TempDir()) + "/vector_faiss_blocked_committed_file";
  std::error_code ec;
  std::filesystem::remove(blocked_root, ec);
  std::ofstream blocker(blocked_root,
                        std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(blocker.good());
  blocker << "blocked";
  blocker.close();
  ASSERT_TRUE(blocker);

  vector_index::set_faiss_external_snapshot_root_for_testing(blocked_root);
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_blocked_committed");

  std::unordered_map<uint64_t, vector_index::vector_data> committed_entries;
  committed_entries.emplace(3, vector_index::vector_data{3.0F, 3.0F});
  EXPECT_FALSE(backend.load_committed_entries(committed_entries));

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove(blocked_root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalPersistRollbackWhenManifestSaveIsInjected) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_manifest_save_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_manifest_save_fail");
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_save_manifest_generation");
    EXPECT_FALSE(backend.upsert(1, {1.0F, 1.0F}));
  }

  EXPECT_EQ(0U, backend.entry_count());
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_EQ(0U, backend.external_manifest_generation());
  EXPECT_TRUE(find_faiss_manifest_file(root).empty());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalPersistRollbackWhenManifestRenameIsInjected) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_manifest_rename_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_manifest_rename_fail");
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
        "+d,vector_backend_fail_save_manifest_generation_rename");
    EXPECT_FALSE(backend.upsert(1, {1.0F, 1.0F}));
  }

  EXPECT_EQ(0U, backend.entry_count());
  EXPECT_TRUE(find_faiss_manifest_file(root).empty());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalEraseLastEntryKeepsEmptyGenerationWhenCleanupIsInjected) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_erase_last_cleanup_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_erase_last_cleanup_fail");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_remove_generated_snapshots");
    EXPECT_TRUE(backend.erase(1));
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(0U, backend.entry_count());
  EXPECT_TRUE(find_faiss_manifest_file(root).empty());

  vector_index::faiss_backend recovered(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal,
      "idx_faiss_erase_last_cleanup_fail");
  ASSERT_TRUE(recovered.recover());
  EXPECT_EQ(0U, recovered.entry_count());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalPersistRollbackWhenManifestQuarantineIsInjected) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_manifest_quarantine_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  {
    vector_index::faiss_backend writer(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_manifest_quarantine_fail");
    ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
  }
  const std::string manifest_path = find_faiss_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  write_binary_file(manifest_path, "broken-manifest");

  vector_index::faiss_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_faiss_manifest_quarantine_fail");
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_quarantine_file_if_exists");
    EXPECT_FALSE(backend.upsert(2, {2.0F, 2.0F}));
  }

  EXPECT_EQ(0U, backend.entry_count());
  EXPECT_TRUE(std::filesystem::exists(manifest_path));
  EXPECT_FALSE(std::filesystem::exists(manifest_path + ".corrupt"));

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalSecondPersistKeepsPublishedGenerationWhenCleanupIsInjected) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_second_persist_remove_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_second_persist_remove");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_remove_if_exists");
    EXPECT_TRUE(backend.upsert(2, {2.0F, 2.0F}));
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({2.0F, 2.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
  EXPECT_EQ(1U, result[1].doc_id);
  EXPECT_EQ(2U, backend.entry_count());

  vector_index::faiss_backend recovered(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal,
      "idx_faiss_second_persist_remove");
  ASSERT_TRUE(recovered.recover());
  ASSERT_TRUE(recovered.search({2.0F, 2.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
  EXPECT_EQ(1U, result[1].doc_id);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalSecondPersistKeepsPublishedGenerationWhenCleanupIsInjected) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_diskann_second_persist_remove_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::diskann_backend backend(2,
                                        vector_index::metric_type::kEuclidean,
                                        vector_index::backend_mode::kExternal,
                                        "idx_diskann_second_persist_remove");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_remove_if_exists");
    EXPECT_TRUE(backend.upsert(2, {2.0F, 2.0F}));
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({2.0F, 2.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
  EXPECT_EQ(1U, result[1].doc_id);
  EXPECT_EQ(2U, backend.entry_count());

  vector_index::diskann_backend recovered(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal,
      "idx_diskann_second_persist_remove");
  ASSERT_TRUE(recovered.recover());
  ASSERT_TRUE(recovered.search({2.0F, 2.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
  EXPECT_EQ(1U, result[1].doc_id);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalEraseLastEntryRollbackWhenArtifactRemovalIsInjected) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_erase_last_remove_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_erase_last_remove_fail");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_remove_if_exists");
    EXPECT_FALSE(backend.erase(1));
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
  EXPECT_EQ(1U, backend.entry_count());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalEraseLastEntryKeepsEmptyGenerationWhenDirCleanupIsInjected) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_faiss_erase_last_dir_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kExternal,
                                     "idx_faiss_erase_last_dir_fail");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_remove_dir_if_empty");
    EXPECT_TRUE(backend.erase(1));
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(0U, backend.entry_count());
  EXPECT_TRUE(find_faiss_manifest_file(root).empty());

  vector_index::faiss_backend recovered(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal, "idx_faiss_erase_last_dir_fail");
  ASSERT_TRUE(recovered.recover());
  EXPECT_EQ(0U, recovered.entry_count());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     FaissExternalRecoverFallsBackWhenManifestGenerationSnapshotMissing) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_faiss_manifest_missing_generation_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  const std::string manifest_path =
      root + "/6964785f66616973735f6d616e69666573745f6d697373/"
             "faiss_external.manifest.v1";
  {
    vector_index::faiss_backend writer(2, vector_index::metric_type::kEuclidean,
                                      vector_index::backend_mode::kExternal,
                                      "idx_faiss_manifest_miss");
    ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
  }

  {
    std::ofstream manifest(manifest_path,
                           std::ios::out | std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(manifest.good());
    manifest << "mysql-vector-faiss-external-manifest-v1\n";
    manifest << "99\n";
    manifest.close();
    ASSERT_TRUE(manifest);
  }

  const uint64_t fallback_before = vector_status::backend_recover_fallbacks();
  vector_index::faiss_backend recovered(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_faiss_manifest_miss");
  ASSERT_TRUE(recovered.recover());
  EXPECT_TRUE(recovered.last_recover_used_fallback());
  EXPECT_EQ(fallback_before + 1, vector_status::backend_recover_fallbacks());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalRecoverFallsBackWhenManifestGenerationSnapshotMissing) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_manifest_missing_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  {
    vector_index::diskann_backend writer(
        2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
        "idx_diskann_manifest_miss");
    ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
  }

  const std::string manifest_path = find_diskann_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());

  {
    std::ofstream manifest(manifest_path,
                           std::ios::out | std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(manifest.good());
    manifest << "mysql-vector-diskann-external-manifest-v1\n";
    manifest << "99\n";
    manifest.close();
    ASSERT_TRUE(manifest);
  }

  vector_index::diskann_backend recovered(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_manifest_miss");
  ASSERT_TRUE(recovered.recover());
  EXPECT_TRUE(recovered.last_recover_used_fallback());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalRecoverFallsBackWhenManifestLoadIsInjected) {
  if (!vector_index::diskann_api_load_for_testing()) {
    GTEST_SKIP() << "DiskANN API unavailable in current build/runtime";
  }
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_manifest_load_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  {
    vector_index::diskann_backend writer(
        2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
        "idx_diskann_manifest_load_fail");
    ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));
  }

  vector_index::diskann_backend recovered(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_manifest_load_fail");
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_load_manifest_generation");
    ASSERT_TRUE(recovered.recover());
  }
  EXPECT_TRUE(recovered.last_recover_used_fallback());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(recovered.search({1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalSecondPersistRecoversFromInjectedManifestLoadFailure) {
  const std::string root = std::string(testing::TempDir()) +
                           "/vector_diskann_manifest_load_persist_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::diskann_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kExternal,
                                       "idx_diskann_manifest_load_persist");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_load_manifest_generation");
    ASSERT_TRUE(backend.upsert(2, {2.0F, 2.0F}));
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({2.0F, 2.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
  EXPECT_EQ(1U, result[1].doc_id);
  EXPECT_TRUE(backend.external_manifest_present());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest, DiskAnnExternalEraseLastEntryRemovesArtifacts) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_erase_last_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_erase_last");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  const std::string manifest_path = find_diskann_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  const std::string snapshot_path = find_diskann_snapshot_file(
      std::filesystem::path(manifest_path).parent_path().string());
  ASSERT_FALSE(snapshot_path.empty());
  ASSERT_TRUE(backend.erase(1));
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_EQ(0U, backend.external_manifest_generation());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 2, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_FALSE(std::filesystem::exists(manifest_path));
  EXPECT_TRUE(find_diskann_snapshot_file(
                  std::filesystem::path(manifest_path).parent_path().string())
                  .empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalEraseLastEntryRollbackWhenQuarantineCleanupFails) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_erase_last_fail_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_erase_last_fail");
  ASSERT_TRUE(backend.upsert(1, {1.0F, 1.0F}));

  const std::string manifest_path = find_diskann_manifest_file(root);
  ASSERT_FALSE(manifest_path.empty());
  ASSERT_TRUE(std::filesystem::create_directories(manifest_path + ".corrupt", ec));
  ASSERT_FALSE(ec);
  write_binary_file(manifest_path + ".corrupt/blocker", "block");

  EXPECT_FALSE(backend.erase(1));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest, DiskAnnExternalRecoverWithoutArtifactsStaysEmpty) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_recover_empty_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_recover_empty");
  std::vector<vector_index::search_result> result;

  EXPECT_TRUE(backend.recover());
  EXPECT_FALSE(backend.last_recover_used_fallback());
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_EQ(0U, backend.external_manifest_generation());
  EXPECT_TRUE(backend.search({1.0F, 1.0F}, 4, &result));
  EXPECT_TRUE(result.empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest, DiskAnnExternalBuildParamsCanBeConfigured) {
  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_params");

  EXPECT_EQ(32U, backend.diskann_max_degree());
  EXPECT_EQ(64U, backend.diskann_build_complexity());
  EXPECT_TRUE(backend.set_diskann_build_params(48, 96, 4));
  EXPECT_EQ(48U, backend.diskann_max_degree());
  EXPECT_EQ(96U, backend.diskann_build_complexity());
  EXPECT_EQ(4U, backend.diskann_build_threads());
  EXPECT_FALSE(backend.set_diskann_build_params(0, 96, 4));
  EXPECT_FALSE(backend.set_diskann_build_params(48, 0, 4));
  EXPECT_FALSE(backend.set_diskann_build_params(48, 96, 65536));
  EXPECT_TRUE(backend.set_diskann_build_threads(0));
  EXPECT_EQ(0U, backend.diskann_build_threads());
  EXPECT_FALSE(backend.set_diskann_build_threads(65536));
  EXPECT_EQ(vector_index::diskann_build_mode::kAuto,
            backend.diskann_build_mode_value());
  EXPECT_TRUE(
      backend.set_diskann_build_mode(vector_index::diskann_build_mode::kOffline));
  EXPECT_EQ(vector_index::diskann_build_mode::kOffline,
            backend.diskann_build_mode_value());
}

TEST(VectorIndexBackendTest, DiskAnnExternalSearchComplexityCanBeConfigured) {
  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_search_complexity");

  EXPECT_EQ(64U, backend.diskann_search_complexity());
  EXPECT_TRUE(backend.set_diskann_search_complexity(80));
  EXPECT_EQ(80U, backend.diskann_search_complexity());
  EXPECT_FALSE(backend.set_diskann_search_complexity(0));
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalRecoverCommittedEntriesUsesCommittedTruth) {
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_recover_entries_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::diskann_backend writer(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_recover_entries");
  ASSERT_TRUE(writer.upsert(1, {1.0F, 1.0F}));

  std::unordered_map<uint64_t, vector_index::vector_data> committed_entries{
      {9, {9.0F, 9.0F}}};
  vector_index::diskann_backend recovered(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_recover_entries");
  ASSERT_TRUE(recovered.recover_committed_entries(committed_entries));

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(recovered.search({9.0F, 9.0F}, 2, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(9U, result[0].doc_id);

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalNativeRebuildDoesNotCreateSidecar) {
  if (!vector_index::diskann_api_load_for_testing()) {
    GTEST_SKIP() << "DiskANN API unavailable in current build/runtime";
  }
  const std::string root =
      std::string(testing::TempDir()) + "/vector_diskann_native_no_sidecar_t";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  ASSERT_FALSE(ec);
  vector_index::set_faiss_external_snapshot_root_for_testing(root);

  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean,
      vector_index::backend_mode::kExternal, "idx_diskann_native_no_sidecar");
  const std::unordered_map<uint64_t, vector_index::vector_data> entries{
      {30, {3.0F, 3.0F}}, {20, {2.0F, 2.0F}}, {10, {1.0F, 1.0F}}};
  ASSERT_TRUE(backend.rebuild_from_committed_entries(entries));
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_EQ(0U, backend.external_manifest_generation());
  EXPECT_TRUE(find_diskann_manifest_file(root).empty());
  EXPECT_TRUE(find_faiss_manifest_file(root).empty());

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 2, &result));
  ASSERT_EQ(2U, result.size());
  EXPECT_EQ(10U, result[0].doc_id);

  ASSERT_TRUE(backend.upsert(30, {0.0F, 0.0F}));
  EXPECT_FALSE(backend.external_manifest_present());
  EXPECT_TRUE(find_diskann_manifest_file(root).empty());
  EXPECT_TRUE(find_faiss_manifest_file(root).empty());

  vector_index::reset_faiss_external_snapshot_root_for_testing();
  std::filesystem::remove_all(root, ec);
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalUpsertDisablesNativeRuntimeOnInjectedInsertFailure) {
  if (!vector_index::diskann_api_load_for_testing()) {
    GTEST_SKIP() << "DiskANN API unavailable in current build/runtime";
  }
  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_native_insert_fail");
  ASSERT_TRUE(backend.rebuild_from_committed_entries({{1, {1.0F, 1.0F}}}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_diskann_native_insert");
    ASSERT_TRUE(backend.upsert(2, {2.0F, 2.0F}));
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({2.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);
  EXPECT_TRUE(backend.set_diskann_search_complexity(80));
  EXPECT_EQ(80U, backend.diskann_search_complexity());
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalEraseDisablesNativeRuntimeOnInjectedRemoveFailure) {
  if (!vector_index::diskann_api_load_for_testing()) {
    GTEST_SKIP() << "DiskANN API unavailable in current build/runtime";
  }
  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_native_remove_fail");
  ASSERT_TRUE(backend.rebuild_from_committed_entries({{1, {1.0F, 1.0F}}}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_diskann_native_remove");
    ASSERT_TRUE(backend.erase(1));
  }

  std::vector<vector_index::search_result> result;
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 2, &result));
  EXPECT_TRUE(result.empty());
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalSearchFallsBackOnInjectedNativeSearchFailure) {
  if (!vector_index::diskann_api_load_for_testing()) {
    GTEST_SKIP() << "DiskANN API unavailable in current build/runtime";
  }
  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_native_search_fail");
  ASSERT_TRUE(backend.rebuild_from_committed_entries({{1, {1.0F, 1.0F}}}));

  std::vector<vector_index::search_result> result;
  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_diskann_native_search");
    ASSERT_TRUE(backend.search({1.0F, 1.0F}, 2, &result));
  }
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(1U, result[0].doc_id);
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalRebuildFromCommittedEntriesRejectsInjectedNativeRebuildFailure) {
  if (!vector_index::diskann_api_load_for_testing()) {
    GTEST_SKIP() << "DiskANN API unavailable in current build/runtime";
  }
  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_native_rebuild_fail");

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug, "+d,vector_backend_fail_diskann_native_rebuild");
    EXPECT_FALSE(backend.rebuild_from_committed_entries({{1, {1.0F, 1.0F}}}));
  }
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalSetSearchComplexityRejectsInjectedLiveNativeFailure) {
  if (!vector_index::diskann_api_load_for_testing()) {
    GTEST_SKIP() << "DiskANN API unavailable in current build/runtime";
  }
  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_native_search_complexity_fail");
  ASSERT_TRUE(backend.rebuild_from_committed_entries({{1, {1.0F, 1.0F}}}));

  {
    VECTOR_SCOPED_DEBUG_FLAG(debug,
        "+d,vector_backend_fail_diskann_native_live_search_complexity");
    EXPECT_FALSE(backend.set_diskann_search_complexity(80));
  }
  EXPECT_EQ(64U, backend.diskann_search_complexity());
}

TEST(VectorIndexBackendTest,
     DiskAnnExternalLiveNativeSearchComplexityAndZeroTopK) {
  if (!vector_index::diskann_api_load_for_testing()) {
    GTEST_SKIP() << "DiskANN API unavailable in current build/runtime";
  }
  vector_index::diskann_backend backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_native_live_zero_topk");
  ASSERT_TRUE(backend.rebuild_from_committed_entries({{1, {1.0F, 1.0F}}}));

  ASSERT_TRUE(backend.set_diskann_search_complexity(80));
  EXPECT_EQ(80U, backend.diskann_search_complexity());

  std::vector<vector_index::search_result> result{{99, 99.0}};
  ASSERT_TRUE(backend.search({1.0F, 1.0F}, 0, &result));
  EXPECT_TRUE(result.empty());

  ASSERT_TRUE(backend.upsert(2, {2.0F, 2.0F}));
  ASSERT_TRUE(backend.erase(1));
  ASSERT_TRUE(backend.search({2.0F, 2.0F}, 1, &result));
  ASSERT_EQ(1U, result.size());
  EXPECT_EQ(2U, result[0].doc_id);

  vector_index::diskann_backend empty_backend(
      2, vector_index::metric_type::kEuclidean, vector_index::backend_mode::kExternal,
      "idx_diskann_native_live_empty");
  ASSERT_TRUE(empty_backend.rebuild_from_committed_entries({}));
  result = {{99, 99.0}};
  ASSERT_TRUE(empty_backend.search({1.0F, 1.0F}, 1, &result));
  EXPECT_TRUE(result.empty());

}

TEST(VectorIndexBackendTest, FaissMemoryRejectsExternalOnlyTunings) {
  vector_index::faiss_backend backend(2, vector_index::metric_type::kEuclidean,
                                     vector_index::backend_mode::kMemory);

  EXPECT_EQ("flat", backend.backend_variant());
  EXPECT_EQ(0U, backend.search_ef());
  EXPECT_EQ(0U, backend.hnsw_m());
  EXPECT_EQ(0U, backend.hnsw_ef_construction());
  EXPECT_EQ(0U, backend.faiss_nlist());
  EXPECT_EQ(0U, backend.faiss_nprobe());
  EXPECT_EQ(0U, backend.faiss_pq_m());
  EXPECT_EQ(0U, backend.faiss_pq_bits());
  EXPECT_FALSE(backend.set_search_ef(32));
  EXPECT_FALSE(backend.set_hnsw_build_params(16, 32));
  EXPECT_FALSE(backend.set_faiss_ivf_params(4, 2));
  EXPECT_FALSE(backend.set_faiss_ivf_pq_params(4, 2, 2, 8));
}

TEST(VectorIndexBackendTest, DiskAnnMemoryRejectsExternalOnlyOperations) {
  vector_index::diskann_backend backend(2, vector_index::metric_type::kEuclidean,
                                       vector_index::backend_mode::kMemory,
                                       "idx_diskann_mem_mode");
  std::unordered_map<uint64_t, vector_index::vector_data> entries{
      {1, {1.0F, 1.0F}}};

  EXPECT_FALSE(backend.upsert(1, {1.0F, 1.0F}));
  EXPECT_FALSE(backend.erase(1));
  EXPECT_FALSE(backend.recover());
  EXPECT_FALSE(backend.recover_committed_entries(entries));
  EXPECT_FALSE(backend.set_diskann_build_params(16, 32, 0));
  EXPECT_FALSE(backend.set_diskann_search_complexity(48));
  EXPECT_EQ(0U, backend.diskann_max_degree());
  EXPECT_EQ(0U, backend.diskann_build_complexity());
  EXPECT_EQ(0U, backend.diskann_search_complexity());
}

TEST(VectorIndexBackendTest, RemoveBackendArtifactsIsNoopForNonExternalModes) {
  EXPECT_TRUE(vector_index::remove_backend_artifacts(
      "idx_native_mem", vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kNative));
  EXPECT_TRUE(vector_index::remove_backend_artifacts(
      "idx_diskann_mem", vector_index::backend_mode::kMemory,
      vector_index::backend_provider::kDiskAnn));
}

TEST(VectorIndexBackendTest,
     RemoveBackendArtifactsIsNoopForEmptyNameAndNonSidecarProviders) {
  EXPECT_TRUE(vector_index::remove_backend_artifacts(
      "", vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kFaiss));
  EXPECT_TRUE(vector_index::remove_backend_artifacts(
      "idx_native_ext", vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kNative));
  EXPECT_TRUE(vector_index::remove_backend_artifacts(
      "idx_hnswlib_ext", vector_index::backend_mode::kExternal,
      vector_index::backend_provider::kHnswlib));
}

}  // namespace vector_index_backend_unittest
