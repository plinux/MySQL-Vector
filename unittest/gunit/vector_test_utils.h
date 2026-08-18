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

#ifndef UNITTEST_GUNIT_VECTOR_TEST_UTILS_H
#define UNITTEST_GUNIT_VECTOR_TEST_UTILS_H

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "my_dbug.h"
#include "my_sys.h"
#include "sql/handler.h"
#include "sql/mysqld.h"
#include "sql/vector/vector_index_backend.h"
#include "sql/xa.h"

namespace vector_gunit {

template <typename T>
class ScopedValueGuard {
 public:
  ScopedValueGuard(T *value, T replacement)
      : m_value(value), m_original(*value) {
    *m_value = replacement;
  }

  ~ScopedValueGuard() { *m_value = m_original; }

  ScopedValueGuard(const ScopedValueGuard &) = delete;
  ScopedValueGuard &operator=(const ScopedValueGuard &) = delete;

 private:
  T *m_value;
  T m_original;
};

using BoolGuard = ScopedValueGuard<bool>;
using UlongGuard = ScopedValueGuard<ulong>;
using UlonglongGuard = ScopedValueGuard<ulonglong>;

class ScopedTempDirectory {
 public:
  explicit ScopedTempDirectory(const char *prefix) {
    static std::atomic<uint64_t> sequence{0};
    const auto timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const uint64_t id = sequence.fetch_add(1, std::memory_order_relaxed);
    m_path = std::filesystem::path(testing::TempDir()) /
             (std::string(prefix) + "-" + std::to_string(timestamp) + "-" +
              std::to_string(id));

    std::error_code error;
    if (!std::filesystem::create_directories(m_path, error) || error) {
      m_error = error ? error.message() : "temporary directory already exists";
      m_path.clear();
    }
  }

  ~ScopedTempDirectory() {
    if (m_path.empty()) return;
    std::error_code error;
    std::filesystem::remove_all(m_path, error);
  }

  ScopedTempDirectory(const ScopedTempDirectory &) = delete;
  ScopedTempDirectory &operator=(const ScopedTempDirectory &) = delete;

  bool valid() const { return !m_path.empty(); }
  const std::filesystem::path &path() const { return m_path; }
  const std::string &error() const { return m_error; }

 private:
  std::filesystem::path m_path;
  std::string m_error;
};

class ScopedDebugFlag {
 public:
  explicit ScopedDebugFlag(const char *flags [[maybe_unused]]) {
    DBUG_SET(flags);
  }
  ~ScopedDebugFlag() { DBUG_SET(""); }

  ScopedDebugFlag(const ScopedDebugFlag &) = delete;
  ScopedDebugFlag &operator=(const ScopedDebugFlag &) = delete;
};

class NativeProviderGuard {
 public:
  explicit NativeProviderGuard(bool supported = true)
      : m_original(vector_index::native_provider_supported_for_testing()) {
    vector_index::set_native_provider_supported_for_testing(supported);
  }
  ~NativeProviderGuard() {
    vector_index::set_native_provider_supported_for_testing(m_original);
  }

  NativeProviderGuard(const NativeProviderGuard &) = delete;
  NativeProviderGuard &operator=(const NativeProviderGuard &) = delete;

 private:
  bool m_original;
};

inline Xa_state_list::instantiation_tuple make_xa_state_list_for_testing() {
  const ulong original_tc_log_page_size = tc_log_page_size;
  if (tc_log_page_size == 0) tc_log_page_size = my_getpagesize();
  auto xa_state_list = Xa_state_list::new_instance();
  tc_log_page_size = original_tc_log_page_size;
  return xa_state_list;
}

class FailingSearchHnswBackend final : public vector_index::backend {
 public:
  explicit FailingSearchHnswBackend(size_t dimension)
      : m_entries(dimension, vector_index::metric_type::kEuclidean),
        m_dimension(dimension) {}

  bool upsert(uint64_t doc_id,
              const vector_index::vector_data &vector) override {
    return m_entries.upsert(doc_id, vector);
  }
  bool erase(uint64_t doc_id) override { return m_entries.erase(doc_id); }
  bool search(const vector_index::vector_data &query [[maybe_unused]],
              size_t top_k [[maybe_unused]],
              std::vector<vector_index::search_result> *results) const override {
    if (results != nullptr) results->clear();
    return false;
  }
  size_t entry_count() const override { return m_entries.entry_count(); }
  size_t dimension() const override { return m_dimension; }
  vector_index::metric_type metric() const override {
    return vector_index::metric_type::kEuclidean;
  }
  vector_index::backend_mode mode() const override {
    return vector_index::backend_mode::kMemory;
  }
  vector_index::backend_provider provider() const override {
    return vector_index::backend_provider::kHnswlib;
  }
  std::string backend_variant() const override { return "hnsw"; }
  uint32_t search_ef() const override { return 64; }
  uint32_t hnsw_m() const override { return 16; }
  uint32_t hnsw_ef_construction() const override { return 200; }
  bool supports_mutations() const override { return true; }

 private:
  vector_index::memory_backend m_entries;
  size_t m_dimension{0};
};

}  // namespace vector_gunit

#if defined(NDEBUG)
#define VECTOR_SCOPED_DEBUG_FLAG(name, flags)                           \
  GTEST_SKIP() << "Requires DBUG fault injection, which is compiled out " \
                  "in Release builds."
#else
#define VECTOR_SCOPED_DEBUG_FLAG(name, flags) \
  vector_gunit::ScopedDebugFlag name(flags)
#endif

#endif  // UNITTEST_GUNIT_VECTOR_TEST_UTILS_H
