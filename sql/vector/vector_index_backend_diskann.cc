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

#include "sql/vector/vector_index_backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "my_dbug.h"
#include "sql/vector/vector_index_backend_common.h"
#include "sql/vector/vector_index_backend_internal.h"
#include "sql/vector/vector_index_limits.h"

namespace {

using vector_index::detail::encode_hex_bytes;
using vector_index::detail::ensure_parent_directory;
using vector_index::detail::compute_distance;
using vector_index::detail::external_snapshot_directory;
using vector_index::detail::remove_dir_if_empty;
using vector_index::detail::remove_diskann_external_artifacts;
using vector_index::detail::remove_if_exists;

constexpr uint64_t kDiskAnnTermBitmask = 0x7ULL;
constexpr uint32_t kDiskAnnNoQuant = 1;
constexpr uint32_t kDiskAnnVectorValueFp32 = 1;
constexpr uint32_t kDiskAnnBuildComplexity = 64;
constexpr uint32_t kDiskAnnMaxDegree = 32;

using diskann_read_data_callback =
    void (*)(uint32_t, void *, const uint8_t *, size_t);
using diskann_rmw_data_callback = void (*)(void *, uint8_t *, size_t);
using diskann_read_callback =
    void (*)(uint64_t, uint32_t, const uint8_t *, size_t, diskann_read_data_callback,
             void *);
using diskann_write_callback =
    bool (*)(uint64_t, const uint8_t *, size_t, const uint8_t *, size_t);
using diskann_delete_callback = bool (*)(uint64_t, const uint8_t *, size_t);
using diskann_read_modify_write_callback =
    bool (*)(uint64_t, const uint8_t *, size_t, size_t, diskann_rmw_data_callback,
             void *);
using diskann_create_index_fn =
    const void *(*)(uint64_t, uint32_t, uint32_t, uint32_t, int32_t, uint32_t,
                    uint32_t, diskann_read_callback, diskann_write_callback,
                    diskann_delete_callback,
                    diskann_read_modify_write_callback);
using diskann_drop_index_fn = void (*)(uint64_t, const void *);
using diskann_insert_fn = bool (*)(uint64_t, const void *, const uint8_t *, size_t,
                                 uint32_t, const uint8_t *, size_t,
                                 const uint8_t *, size_t);
using diskann_search_vector_fn =
    int32_t (*)(uint64_t, const void *, uint32_t, const uint8_t *, size_t, float,
                uint32_t, const uint8_t *, size_t, size_t, uint8_t *, size_t,
                float *, size_t, void *);
using diskann_remove_fn = bool (*)(uint64_t, const void *, const uint8_t *, size_t);
using diskann_card_fn = uint64_t (*)(uint64_t, const void *);

struct diskann_api {
  void *handle{nullptr};
  diskann_create_index_fn create_index{nullptr};
  diskann_drop_index_fn drop_index{nullptr};
  diskann_insert_fn insert{nullptr};
  diskann_search_vector_fn search_vector{nullptr};
  diskann_remove_fn remove{nullptr};
  diskann_card_fn card{nullptr};

  bool load() {
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_api_load", return false;);
    if (handle != nullptr) return available();

    std::vector<std::string> candidates;
    if (const char *env = std::getenv("MYSQL_VECTOR_DISKANN_LIB");
        env != nullptr && env[0] != '\0') {
      candidates.emplace_back(env);
    }
    if (const char *env = std::getenv("MYSQL_VECTOR_DISKANN_LIB_DIR");
        env != nullptr && env[0] != '\0') {
      const std::filesystem::path lib_dir(env);
      candidates.emplace_back(
          (lib_dir / "libdiskann_garnet.dylib").string());
      candidates.emplace_back((lib_dir / "libdiskann_garnet.so").string());
    }
#ifdef MYSQL_VECTOR_DISKANN_DEFAULT_LIB
    candidates.emplace_back(MYSQL_VECTOR_DISKANN_DEFAULT_LIB);
#endif
#ifdef MYSQL_VECTOR_DISKANN_DEFAULT_LIB_DIR
    {
      const std::filesystem::path lib_dir(MYSQL_VECTOR_DISKANN_DEFAULT_LIB_DIR);
      candidates.emplace_back(
          (lib_dir / "libdiskann_garnet.dylib").string());
      candidates.emplace_back((lib_dir / "libdiskann_garnet.so").string());
    }
#endif
    candidates.emplace_back("libdiskann_garnet.dylib");
    candidates.emplace_back("libdiskann_garnet.so");

    for (const std::string &candidate : candidates) {
      if (candidate.empty()) continue;
      handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (handle == nullptr) continue;
      create_index = reinterpret_cast<diskann_create_index_fn>(
          dlsym(handle, "create_index"));
      drop_index =
          reinterpret_cast<diskann_drop_index_fn>(dlsym(handle, "drop_index"));
      insert = reinterpret_cast<diskann_insert_fn>(dlsym(handle, "insert"));
      search_vector = reinterpret_cast<diskann_search_vector_fn>(
          dlsym(handle, "search_vector"));
      remove = reinterpret_cast<diskann_remove_fn>(dlsym(handle, "remove"));
      card = reinterpret_cast<diskann_card_fn>(dlsym(handle, "card"));
      DBUG_EXECUTE_IF("vector_backend_fail_diskann_api_symbol", { card = nullptr; };);
      if (available()) return true;
      dlclose(handle);
      handle = nullptr;
      create_index = nullptr;
      drop_index = nullptr;
      insert = nullptr;
      search_vector = nullptr;
      remove = nullptr;
      card = nullptr;
    }

    return false;
  }

  bool available() const {
    return handle != nullptr && create_index != nullptr &&
           drop_index != nullptr && insert != nullptr &&
           search_vector != nullptr && remove != nullptr && card != nullptr;
  }
};

diskann_api &get_diskann_api() {
  static diskann_api api;
  if (!api.available()) (void)api.load();
  return api;
}

const char *diskann_term_name(uint64_t ctx) {
  switch (ctx & kDiskAnnTermBitmask) {
    case 0:
      return "vector";
    case 1:
      return "neighbors";
    case 2:
      return "quantized";
    case 3:
      return "attributes";
    case 4:
      return "metadata";
    case 5:
      return "intmap";
    case 6:
      return "extmap";
    default:
      return "unknown";
  }
}

int32_t diskann_metric_code(vector_index::metric_type metric) {
  switch (metric) {
    case vector_index::metric_type::kCosine:
      return 0;
    case vector_index::metric_type::kInnerProduct:
      return 1;
    case vector_index::metric_type::kEuclidean:
      return 2;
  }
  return 2;
}

std::string diskann_doc_id_bytes(uint64_t doc_id) {
  std::string bytes(sizeof(doc_id), '\0');
  std::memcpy(bytes.data(), &doc_id, sizeof(doc_id));
  return bytes;
}

bool parse_diskann_doc_id(const uint8_t *data, size_t length, uint64_t *doc_id) {
  if (doc_id == nullptr || data == nullptr || length != sizeof(uint64_t)) {
    return false;
  }
  std::memcpy(doc_id, data, sizeof(uint64_t));
  return true;
}

}  // namespace

namespace vector_index {

bool diskann_backend::search_entries_exact(
    const vector_data &query, size_t top_k,
    std::vector<search_result> *results) const {
  if (results == nullptr || query.size() != m_dimension) return false;
  results->clear();
  if (top_k == 0) return true;
  for (const auto &entry : m_entries) {
    double distance = 0.0;
    if (!compute_distance(m_metric, query, entry.second, &distance))
      return false;
    results->push_back(search_result{entry.first, distance});
  }
  const size_t count = std::min(top_k, results->size());
  std::partial_sort(results->begin(), results->begin() + count, results->end(),
                    [](const search_result &lhs, const search_result &rhs) {
                      if (lhs.distance != rhs.distance)
                        return lhs.distance < rhs.distance;
                      return lhs.doc_id < rhs.doc_id;
                    });
  results->resize(count);
  return true;
}

class diskann_native_state {
 public:
  diskann_native_state(size_t dimension, metric_type metric, const std::string &index_name,
                     uint32_t build_complexity, uint32_t max_degree,
                     uint32_t build_threads)
      : m_dimension(dimension),
        m_metric(metric),
        m_index_name(index_name),
        m_store_directory(detail::diskann_store_directory(index_name)),
        m_api(&get_diskann_api()),
        m_build_complexity(build_complexity),
        m_max_degree(max_degree),
        m_build_threads(build_threads) {}

  ~diskann_native_state() { close(); }

  bool supported() const { return m_api != nullptr && m_api->available(); }
  bool active() const {
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    return m_index_handle != nullptr;
  }
  const std::string &store_directory() const { return m_store_directory; }

  bool reopen(bool reset_store) {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    return reopen_locked(reset_store);
  }

  bool rebuild_from_entries(const std::unordered_map<uint64_t, vector_data> &entries) {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    std::vector<std::pair<uint64_t, const vector_data *>> ordered_entries;
    ordered_entries.reserve(entries.size());
    for (const auto &entry : entries) {
      if (entry.second.size() != m_dimension) {
        return false;
      }
      ordered_entries.push_back({entry.first, &entry.second});
    }
    std::sort(ordered_entries.begin(), ordered_entries.end(),
              [](const auto &lhs, const auto &rhs) {
                return lhs.first < rhs.first;
              });

    begin_build_memory_store_locked();
    if (!reopen_locked(false)) {
      discard_build_memory_store_locked();
      return false;
    }

    bool rebuild_ok = true;
    for (const auto &entry : ordered_entries) {
      if (!insert_locked(entry.first, *entry.second)) {
        rebuild_ok = false;
        break;
      }
    }
    if (rebuild_ok) rebuild_ok = flush_build_memory_store_locked();
    if (!rebuild_ok) close_locked();
    discard_build_memory_store_locked();
    return rebuild_ok;
  }

  bool insert(uint64_t doc_id, const vector_data &vector) {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    return insert_locked(doc_id, vector);
  }

  bool remove(uint64_t doc_id) {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    if (!ensure_open_locked(false)) return false;
    const std::string doc_id_bytes = diskann_doc_id_bytes(doc_id);
    return m_api->remove(callback_context(), m_index_handle,
                         reinterpret_cast<const uint8_t *>(doc_id_bytes.data()),
                         doc_id_bytes.size());
  }

  bool search(const vector_data &query, size_t top_k,
              std::vector<search_result> *results) const {
    if (results == nullptr) return false;
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    results->clear();
    if (m_index_handle == nullptr) return false;
    std::string ids_buffer(top_k * (sizeof(uint32_t) + sizeof(uint64_t)), '\0');
    std::vector<float> distances(top_k, 0.0F);
    const int32_t count = m_api->search_vector(
        callback_context(), m_index_handle, kDiskAnnVectorValueFp32,
        reinterpret_cast<const uint8_t *>(query.data()), query.size(), 0.0F,
        m_search_complexity, nullptr, 0, 0,
        reinterpret_cast<uint8_t *>(ids_buffer.data()), ids_buffer.size(),
        distances.data(), distances.size(), nullptr);
    if (count < 0) return false;

    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(ids_buffer.data());
    size_t remaining = ids_buffer.size();
    for (int32_t i = 0; i < count; ++i) {
      if (remaining < sizeof(uint32_t)) return false;
      uint32_t length = 0;
      std::memcpy(&length, ptr, sizeof(length));
      ptr += sizeof(length);
      remaining -= sizeof(length);
      if (remaining < length) return false;
      uint64_t doc_id = 0;
      if (!parse_diskann_doc_id(ptr, length, &doc_id)) return false;
      results->push_back({doc_id, distances[static_cast<size_t>(i)]});
      ptr += length;
      remaining -= length;
    }
    return true;
  }

  size_t entry_count() const {
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    if (m_index_handle == nullptr) return 0;
    return static_cast<size_t>(m_api->card(callback_context(), m_index_handle));
  }

  bool set_search_complexity(uint32_t search_complexity) {
    if (search_complexity == 0) return false;
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    m_search_complexity = search_complexity;
    return true;
  }

  uint32_t search_complexity() const {
    std::shared_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    return m_search_complexity;
  }

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  std::string term_directory_for_testing(uint64_t ctx) const {
    return term_directory(ctx);
  }

  std::string key_path_for_testing(uint64_t ctx, const uint8_t *key,
                                size_t length) const {
    return key_path(ctx, key, length);
  }

  bool load_value_for_testing(uint64_t ctx, const uint8_t *key, size_t key_length,
                           std::string *value) const {
    return load_value(ctx, key, key_length, value);
  }

  bool save_value_for_testing(uint64_t ctx, const uint8_t *key, size_t key_length,
                           const uint8_t *value, size_t value_length) {
    return save_value(ctx, key, key_length, value, value_length);
  }

  bool delete_value_for_testing(uint64_t ctx, const uint8_t *key,
                             size_t key_length) {
    return delete_value(ctx, key, key_length);
  }

  bool build_memory_store_round_trip_for_testing(
      uint64_t ctx, const uint8_t *key, size_t key_length,
      const std::string &value, std::string *loaded_before_flush,
      bool *file_exists_before_flush, std::string *loaded_after_flush) {
    if (loaded_before_flush == nullptr || file_exists_before_flush == nullptr ||
        loaded_after_flush == nullptr) {
      return false;
    }
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    const std::string path = key_path(ctx, key, key_length);
    if (path.empty()) return false;
    begin_build_memory_store_locked();
    const bool saved =
        save_value(ctx, key, key_length,
                   reinterpret_cast<const uint8_t *>(value.data()),
                   value.size());
    const bool loaded =
        load_value(ctx, key, key_length, loaded_before_flush);
    *file_exists_before_flush = std::filesystem::exists(path);
    const bool flushed = flush_build_memory_store_locked();
    discard_build_memory_store_locked();
    loaded_after_flush->clear();
    return saved && loaded && flushed &&
           load_value(ctx, key, key_length, loaded_after_flush);
  }

  bool resident_store_round_trip_for_testing(
      uint64_t ctx, const uint8_t *key, size_t key_length,
      const std::string &value, std::string *loaded_after_removing_file) {
    if (loaded_after_removing_file == nullptr) return false;
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    const std::string path = key_path(ctx, key, key_length);
    if (path.empty()) return false;
    begin_build_memory_store_locked();
    const bool saved =
        save_value(ctx, key, key_length,
                   reinterpret_cast<const uint8_t *>(value.data()),
                   value.size());
    const bool flushed = flush_build_memory_store_locked();
    discard_build_memory_store_locked();
    loaded_after_removing_file->clear();
    return saved && flushed && remove_if_exists(path) &&
           load_value(ctx, key, key_length, loaded_after_removing_file);
  }

  static bool parse_prefixed_keys_for_testing(
      const uint8_t *key_data, size_t key_length, uint32_t key_count,
      const std::function<bool(uint32_t, const uint8_t *, size_t)> &visitor) {
    return parse_prefixed_keys(key_data, key_length, key_count, visitor);
  }

  struct test_rmw_payload {
    const char *data;
    size_t length;
  };

  static void copy_rmw_payload_for_testing(void *user_data, uint8_t *data,
                                           size_t length) {
    if (user_data == nullptr || data == nullptr || length == 0) return;
    const auto *payload = static_cast<const test_rmw_payload *>(user_data);
    const size_t copy_length = std::min(length, payload->length);
    if (copy_length != 0) std::memcpy(data, payload->data, copy_length);
  }

  bool read_modify_write_round_trip_for_testing(
      uint64_t ctx, const uint8_t *key, size_t key_length,
      const std::string &initial_value, const std::string &patch_value,
      size_t write_length, std::string *persistent_value,
      std::string *build_memory_value, std::string *resident_value) {
    if (persistent_value == nullptr || build_memory_value == nullptr ||
        resident_value == nullptr) {
      return false;
    }

    test_rmw_payload payload{patch_value.data(), patch_value.size()};
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);

    bool ok = save_value(ctx, key, key_length,
                         reinterpret_cast<const uint8_t *>(initial_value.data()),
                         initial_value.size());
    ok = read_modify_write_value(ctx, key, key_length, write_length,
                                 &diskann_native_state::copy_rmw_payload_for_testing,
                                 &payload) &&
         ok;
    persistent_value->clear();
    ok = load_value(ctx, key, key_length, persistent_value) && ok;

    begin_build_memory_store_locked();
    ok = read_modify_write_value(ctx, key, key_length, 0,
                                 &diskann_native_state::copy_rmw_payload_for_testing,
                                 &payload) &&
         ok;
    build_memory_value->clear();
    ok = load_value(ctx, key, key_length, build_memory_value) && ok;
    std::string after_build_delete;
    ok = delete_value(ctx, key, key_length) && ok;
    ok = !load_value(ctx, key, key_length, &after_build_delete) && ok;
    discard_build_memory_store_locked();

    begin_build_memory_store_locked();
    ok = save_value(ctx, key, key_length,
                    reinterpret_cast<const uint8_t *>(initial_value.data()),
                    initial_value.size()) &&
         ok;
    ok = flush_build_memory_store_locked() && ok;
    discard_build_memory_store_locked();
    (void)remove_if_exists(key_path(ctx, key, key_length));
    ok = read_modify_write_value(ctx, key, key_length, write_length,
                                 &diskann_native_state::copy_rmw_payload_for_testing,
                                 &payload) &&
         ok;
    resident_value->clear();
    ok = load_value(ctx, key, key_length, resident_value) && ok;
    std::string after_resident_delete;
    ok = delete_value(ctx, key, key_length) && ok;
    ok = !load_value(ctx, key, key_length, &after_resident_delete) && ok;
    return ok;
  }
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

 private:
  bool reopen_locked(bool reset_store) {
    if (!supported()) return false;

    close_locked();
    if (m_store_directory.empty()) return false;

    std::error_code ec;
    if (!m_build_memory_store_active.load(std::memory_order_acquire)) {
      if (reset_store) {
        std::filesystem::remove_all(m_store_directory, ec);
        clear_store(&m_resident_store);
        m_resident_store_active.store(false, std::memory_order_release);
      }
      ec.clear();
      std::filesystem::create_directories(m_store_directory, ec);
      if (ec) return false;
    }

    const void *index = m_api->create_index(
        callback_context(), static_cast<uint32_t>(m_dimension), 0,
        kDiskAnnNoQuant, diskann_metric_code(m_metric), m_build_complexity,
        m_max_degree, &diskann_native_state::read_callback,
        &diskann_native_state::write_callback, &diskann_native_state::delete_callback,
        &diskann_native_state::read_modify_write_callback);
    if (index == nullptr) return false;
    m_index_handle = index;
    return true;
  }

  bool insert_locked(uint64_t doc_id, const vector_data &vector) {
    if (!ensure_open_locked(false)) return false;
    const std::string doc_id_bytes = diskann_doc_id_bytes(doc_id);
    return m_api->insert(callback_context(), m_index_handle,
                         reinterpret_cast<const uint8_t *>(doc_id_bytes.data()),
                         doc_id_bytes.size(), kDiskAnnVectorValueFp32,
                         reinterpret_cast<const uint8_t *>(vector.data()),
                         vector.size(), nullptr, 0);
  }

  static diskann_native_state *from_context(uint64_t ctx) {
    return reinterpret_cast<diskann_native_state *>(ctx & ~kDiskAnnTermBitmask);
  }

  uint64_t callback_context() const {
    return reinterpret_cast<uint64_t>(const_cast<diskann_native_state *>(this));
  }

  bool ensure_open_locked(bool reset_store) const {
    if (m_index_handle != nullptr) return true;
    return const_cast<diskann_native_state *>(this)->reopen_locked(reset_store);
  }

  void close() {
    std::unique_lock<std::shared_mutex> guard(m_lifecycle_mutex);
    close_locked();
  }

  void close_locked() {
    if (m_index_handle != nullptr && supported()) {
      m_api->drop_index(callback_context(), m_index_handle);
      m_index_handle = nullptr;
    }
  }

  std::string term_directory(uint64_t ctx) const {
    if (m_store_directory.empty()) return "";
    return m_store_directory + "/" + diskann_term_name(ctx);
  }

  std::string store_key(uint64_t ctx, const uint8_t *key, size_t length) const {
    if (key == nullptr || length == 0) return "";
    std::string relative_path = diskann_term_name(ctx);
    relative_path.push_back('/');
    relative_path.append(
        encode_hex_bytes(reinterpret_cast<const char *>(key), length));
    return relative_path;
  }

  std::string key_path_from_store_key(const std::string &root,
                                   const std::string &relative_path) const {
    if (root.empty() || relative_path.empty()) return "";
    return (std::filesystem::path(root) / relative_path).string();
  }

  std::string key_path(uint64_t ctx, const uint8_t *key, size_t length) const {
    return key_path_from_store_key(m_store_directory, store_key(ctx, key, length));
  }

  static constexpr size_t kDiskAnnStoreShardCount = 64;

  struct native_store_shard {
    mutable std::mutex mutex;
    std::unordered_map<std::string, std::string> entries;
  };

  using native_store_shards =
      std::array<native_store_shard, kDiskAnnStoreShardCount>;

  static size_t store_shard_index(const std::string &relative_path) {
    return std::hash<std::string>{}(relative_path) % kDiskAnnStoreShardCount;
  }

  static native_store_shard &store_shard(native_store_shards *store,
                                         const std::string &relative_path) {
    return (*store)[store_shard_index(relative_path)];
  }

  static const native_store_shard &store_shard(
      const native_store_shards &store, const std::string &relative_path) {
    return store[store_shard_index(relative_path)];
  }

  static void clear_store(native_store_shards *store) {
    for (native_store_shard &shard : *store) {
      std::lock_guard<std::mutex> guard(shard.mutex);
      shard.entries.clear();
    }
  }

  static bool load_from_store(const native_store_shards &store,
                              const std::string &relative_path,
                              std::string *value) {
    if (value == nullptr) return false;
    const native_store_shard &shard = store_shard(store, relative_path);
    std::lock_guard<std::mutex> guard(shard.mutex);
    const auto iter = shard.entries.find(relative_path);
    if (iter == shard.entries.end()) return false;
    *value = iter->second;
    return true;
  }

  static bool save_to_store(native_store_shards *store,
                            const std::string &relative_path,
                            std::string value) {
    native_store_shard &shard = store_shard(store, relative_path);
    std::lock_guard<std::mutex> guard(shard.mutex);
    shard.entries[relative_path] = std::move(value);
    return true;
  }

  static void delete_from_store(native_store_shards *store,
                                const std::string &relative_path) {
    native_store_shard &shard = store_shard(store, relative_path);
    std::lock_guard<std::mutex> guard(shard.mutex);
    shard.entries.erase(relative_path);
  }

  void move_build_store_to_resident_locked() {
    for (size_t i = 0; i < kDiskAnnStoreShardCount; ++i) {
      std::scoped_lock guard(m_build_memory_store[i].mutex,
                             m_resident_store[i].mutex);
      m_resident_store[i].entries =
          std::move(m_build_memory_store[i].entries);
      m_build_memory_store[i].entries.clear();
    }
    m_resident_store_active.store(true, std::memory_order_release);
    m_build_memory_store_active.store(false, std::memory_order_release);
  }

  void begin_build_memory_store_locked() {
    clear_store(&m_build_memory_store);
    m_build_memory_store_active.store(true, std::memory_order_release);
  }

  void discard_build_memory_store_locked() {
    clear_store(&m_build_memory_store);
    m_build_memory_store_active.store(false, std::memory_order_release);
  }

  bool save_persistent_value(const std::string &path,
                            const std::string &value) const {
    if (path.empty()) return false;
    if (!ensure_parent_directory(path)) return false;
    const std::string temp_path = path + ".tmp";
    std::ofstream file(temp_path,
                       std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file.good()) return false;
    file.write(value.data(), value.size());
    file.close();
    if (!file) {
      std::remove(temp_path.c_str());
      return false;
    }
    if (std::rename(temp_path.c_str(), path.c_str()) != 0) {
      std::remove(temp_path.c_str());
      return false;
    }
    return true;
  }

  bool swap_persistent_store_directory(const std::string &staging_directory) const {
    std::error_code ec;
    const std::filesystem::path store(m_store_directory);
    const std::filesystem::path staging(staging_directory);
    const std::string suffix =
        ".bulk-old-" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
    const std::filesystem::path backup(m_store_directory + suffix);

    std::filesystem::remove_all(backup, ec);
    if (ec) return false;

    bool has_existing_store = std::filesystem::exists(store, ec);
    if (ec) return false;
    if (has_existing_store) {
      std::filesystem::rename(store, backup, ec);
      if (ec) return false;
    }

    std::filesystem::rename(staging, store, ec);
    if (ec) {
      std::error_code restore_ec;
      if (has_existing_store) std::filesystem::rename(backup, store, restore_ec);
      std::filesystem::remove_all(staging, restore_ec);
      return false;
    }

    if (has_existing_store) {
      std::filesystem::remove_all(backup, ec);
      if (ec) return false;
    }
    return true;
  }

  bool flush_build_memory_store_locked() {
    if (!m_build_memory_store_active.load(std::memory_order_acquire) ||
        m_store_directory.empty()) {
      return false;
    }

    const std::string suffix =
        ".bulk-tmp-" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
    const std::filesystem::path staging(m_store_directory + suffix);
    std::error_code ec;
    std::filesystem::remove_all(staging, ec);
    if (ec) return false;
    std::filesystem::create_directories(staging, ec);
    if (ec) return false;

    for (const native_store_shard &shard : m_build_memory_store) {
      std::lock_guard<std::mutex> guard(shard.mutex);
      for (const auto &entry : shard.entries) {
        const std::string path =
            key_path_from_store_key(staging.string(), entry.first);
        if (!save_persistent_value(path, entry.second)) {
          std::filesystem::remove_all(staging, ec);
          return false;
        }
      }
    }
    if (!swap_persistent_store_directory(staging.string())) return false;
    move_build_store_to_resident_locked();
    return true;
  }

  bool load_value(uint64_t ctx, const uint8_t *key, size_t key_length,
                 std::string *value) const {
    if (value == nullptr) return false;
    const std::string relative_path = store_key(ctx, key, key_length);
    if (relative_path.empty()) return false;
    if (m_build_memory_store_active.load(std::memory_order_acquire)) {
      return load_from_store(m_build_memory_store, relative_path, value);
    }
    if (m_resident_store_active.load(std::memory_order_acquire) &&
        load_from_store(m_resident_store, relative_path, value)) {
      return true;
    }
    return load_persistent_value(relative_path, value);
  }

  bool load_persistent_value(const std::string &relative_path,
                             std::string *value) const {
    if (value == nullptr) return false;
    const std::string path = key_path_from_store_key(m_store_directory, relative_path);
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.good()) return false;
    *value = std::string(std::istreambuf_iterator<char>(file),
                         std::istreambuf_iterator<char>());
    return !file.bad();
  }

  bool save_value(uint64_t ctx, const uint8_t *key, size_t key_length,
                 const uint8_t *value, size_t value_length) {
    if (value == nullptr && value_length != 0) return false;
    const std::string relative_path = store_key(ctx, key, key_length);
    if (relative_path.empty()) return false;
    std::string payload;
    if (value_length != 0) {
      payload.assign(reinterpret_cast<const char *>(value), value_length);
    }
    if (m_build_memory_store_active.load(std::memory_order_acquire)) {
      return save_to_store(&m_build_memory_store, relative_path,
                           std::move(payload));
    }
    if (!save_persistent_value(
            key_path_from_store_key(m_store_directory, relative_path), payload)) {
      return false;
    }
    if (m_resident_store_active.load(std::memory_order_acquire)) {
      return save_to_store(&m_resident_store, relative_path,
                           std::move(payload));
    }
    return true;
  }

  bool delete_value(uint64_t ctx, const uint8_t *key, size_t key_length) {
    const std::string relative_path = store_key(ctx, key, key_length);
    if (relative_path.empty()) return false;
    if (m_build_memory_store_active.load(std::memory_order_acquire)) {
      delete_from_store(&m_build_memory_store, relative_path);
      return true;
    }
    const std::string path = key_path_from_store_key(m_store_directory, relative_path);
    if (!remove_if_exists(path)) return false;
    if (m_resident_store_active.load(std::memory_order_acquire)) {
      delete_from_store(&m_resident_store, relative_path);
    }
    return remove_dir_if_empty(std::filesystem::path(path).parent_path().string());
  }

  static std::string apply_read_modify_write(
      const std::string &value, size_t write_length,
      diskann_rmw_data_callback callback, void *user_data) {
    std::vector<uint64_t> aligned((write_length + sizeof(uint64_t) - 1) /
                                  sizeof(uint64_t));
    if (!aligned.empty()) {
      std::memset(aligned.data(), 0, aligned.size() * sizeof(uint64_t));
      if (!value.empty()) {
        std::memcpy(aligned.data(), value.data(),
                    std::min(value.size(), write_length));
      }
    }
    callback(user_data, reinterpret_cast<uint8_t *>(aligned.data()), write_length);
    if (write_length == 0) return "";
    return std::string(reinterpret_cast<const char *>(aligned.data()),
                       write_length);
  }

  static bool read_modify_write_store_value(
      native_store_shards *store, const std::string &relative_path,
      size_t write_length, diskann_rmw_data_callback callback, void *user_data) {
    native_store_shard &shard = store_shard(store, relative_path);
    std::lock_guard<std::mutex> guard(shard.mutex);
    std::string value;
    const auto iter = shard.entries.find(relative_path);
    if (iter != shard.entries.end()) value = iter->second;
    shard.entries[relative_path] =
        apply_read_modify_write(value, write_length, callback, user_data);
    return true;
  }

  bool read_modify_write_resident_value(
      const std::string &relative_path, size_t write_length,
      diskann_rmw_data_callback callback, void *user_data) {
    native_store_shard &shard = store_shard(&m_resident_store, relative_path);
    std::lock_guard<std::mutex> guard(shard.mutex);
    std::string value;
    const auto iter = shard.entries.find(relative_path);
    if (iter != shard.entries.end()) {
      value = iter->second;
    } else {
      (void)load_persistent_value(relative_path, &value);
    }
    std::string updated =
        apply_read_modify_write(value, write_length, callback, user_data);
    if (!save_persistent_value(
            key_path_from_store_key(m_store_directory, relative_path), updated)) {
      return false;
    }
    shard.entries[relative_path] = std::move(updated);
    return true;
  }

  bool read_modify_write_value(uint64_t ctx, const uint8_t *key,
                               size_t key_length, size_t write_length,
                               diskann_rmw_data_callback callback,
                               void *user_data) {
    const std::string relative_path = store_key(ctx, key, key_length);
    if (relative_path.empty()) return false;
    if (m_build_memory_store_active.load(std::memory_order_acquire)) {
      return read_modify_write_store_value(&m_build_memory_store, relative_path,
                                           write_length, callback, user_data);
    }
    if (m_resident_store_active.load(std::memory_order_acquire)) {
      return read_modify_write_resident_value(relative_path, write_length,
                                              callback, user_data);
    }
    std::string value;
    (void)load_persistent_value(relative_path, &value);
    const std::string updated =
        apply_read_modify_write(value, write_length, callback, user_data);
    return save_persistent_value(
        key_path_from_store_key(m_store_directory, relative_path), updated);
  }

  static bool parse_prefixed_keys(const uint8_t *key_data, size_t key_length,
                                uint32_t key_count,
                                const std::function<bool(uint32_t, const uint8_t *,
                                                         size_t)> &visitor) {
    if (key_count == 0) return true;
    const uint8_t *ptr = key_data;
    size_t remaining = key_length;
    for (uint32_t i = 0; i < key_count; ++i) {
      if (remaining < sizeof(uint32_t)) return false;
      uint32_t length = 0;
      std::memcpy(&length, ptr, sizeof(length));
      ptr += sizeof(length);
      remaining -= sizeof(length);
      if (remaining < length) return false;
      if (!visitor(i, ptr, length)) return false;
      ptr += length;
      remaining -= length;
    }
    return true;
  }

  static void read_callback(uint64_t ctx, uint32_t key_count,
                           const uint8_t *key_data, size_t key_length,
                           diskann_read_data_callback callback, void *user_data) {
    diskann_native_state *state = from_context(ctx);
    if (state == nullptr || callback == nullptr) return;
    (void)parse_prefixed_keys(
        key_data, key_length, key_count,
        [state, ctx, callback, user_data](uint32_t index, const uint8_t *key,
                                          size_t length) {
          std::string value;
          if (!state->load_value(ctx, key, length, &value)) return true;
          std::vector<uint64_t> aligned((value.size() + sizeof(uint64_t) - 1) /
                                        sizeof(uint64_t));
          if (!value.empty()) {
            std::memcpy(aligned.data(), value.data(), value.size());
          }
          callback(index, user_data,
                   reinterpret_cast<const uint8_t *>(aligned.data()),
                   value.size());
          return true;
        });
  }

  static bool write_callback(uint64_t ctx, const uint8_t *key, size_t key_length,
                            const uint8_t *value, size_t value_length) {
    diskann_native_state *state = from_context(ctx);
    if (state == nullptr || value == nullptr) return false;
    return state->save_value(ctx, key, key_length, value, value_length);
  }

  static bool delete_callback(uint64_t ctx, const uint8_t *key, size_t key_length) {
    diskann_native_state *state = from_context(ctx);
    if (state == nullptr) return false;
    return state->delete_value(ctx, key, key_length);
  }

  static bool read_modify_write_callback(uint64_t ctx, const uint8_t *key,
                                      size_t key_length, size_t write_length,
                                      diskann_rmw_data_callback callback,
                                      void *user_data) {
    diskann_native_state *state = from_context(ctx);
    if (state == nullptr || callback == nullptr) return false;
    return state->read_modify_write_value(ctx, key, key_length, write_length,
                                          callback, user_data);
  }

  size_t m_dimension{0};
  metric_type m_metric{metric_type::kEuclidean};
  std::string m_index_name;
  std::string m_store_directory;
  const diskann_api *m_api{nullptr};
  const void *m_index_handle{nullptr};
  uint32_t m_build_complexity{kDiskAnnBuildComplexity};
  uint32_t m_max_degree{kDiskAnnMaxDegree};
  uint32_t m_build_threads{0};
  uint32_t m_search_complexity{kDiskAnnBuildComplexity};
  mutable std::shared_mutex m_lifecycle_mutex;
  std::atomic_bool m_build_memory_store_active{false};
  native_store_shards m_build_memory_store;
  std::atomic_bool m_resident_store_active{false};
  native_store_shards m_resident_store;
};

diskann_backend::diskann_backend(size_t dimension, metric_type metric, backend_mode mode,
                               const std::string &index_name)
    : m_dimension(dimension),
      m_metric(metric),
      m_mode(mode),
      m_index_name(index_name),
      m_external_adapter(dimension, metric, mode, index_name,
                         external_sidecar_profile::kDiskAnn) {}

diskann_backend::~diskann_backend() = default;

bool diskann_backend::upsert(uint64_t doc_id, const vector_data &vector) {
  if (m_mode != backend_mode::kExternal) return false;
  if (vector.size() != m_dimension) return false;
  if (m_external_adapter_active &&
      !m_external_adapter.upsert(doc_id, vector))
    return false;
  m_entries[doc_id] = vector;
  bool native_insert_ok = true;
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    native_insert_ok = m_native_state->insert(doc_id, vector);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_insert",
                    native_insert_ok = false;);
  }
  if (m_native_runtime_enabled && m_native_state != nullptr && !native_insert_ok) {
    m_native_runtime_enabled = false;
    m_native_state.reset();
  }
  return true;
}

bool diskann_backend::erase(uint64_t doc_id) {
  if (m_mode != backend_mode::kExternal) return false;
  if (m_external_adapter_active && !m_external_adapter.erase(doc_id))
    return false;
  m_entries.erase(doc_id);
  bool native_remove_ok = true;
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    native_remove_ok = m_native_state->remove(doc_id);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_remove",
                    native_remove_ok = false;);
  }
  if (m_native_runtime_enabled && m_native_state != nullptr && !native_remove_ok) {
    m_native_runtime_enabled = false;
    m_native_state.reset();
  }
  return true;
}

bool diskann_backend::search(const vector_data &query, size_t top_k,
                            std::vector<search_result> *results) const {
  if (results == nullptr || query.size() != m_dimension) return false;
  results->clear();
  if (top_k == 0) return true;
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    bool native_search_ok = m_native_state->search(query, top_k, results);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_search",
                    native_search_ok = false; results->clear(););
    if (native_search_ok) {
      if (top_k == 0 || !results->empty() || m_entries.empty()) {
        return true;
      }

      // Native DiskANN reopen/rebuild can transiently yield an empty hit set
      // even though committed entries exist. The in-memory committed-entry
      // snapshot keeps SQL-visible search stable without requiring a sidecar.
    } else {
      results->clear();
    }
  }
  return search_entries_exact(query, top_k, results) ||
         m_external_adapter.search(query, top_k, results);
}

bool diskann_backend::load_committed_entries(
    const std::unordered_map<uint64_t, vector_data> &entries) {
  if (m_mode != backend_mode::kExternal) return false;
  if (!m_external_adapter.load_committed_entries(entries)) return false;
  m_entries = entries;
  m_external_adapter_active = true;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

bool diskann_backend::rebuild_from_committed_entries(
    const std::unordered_map<uint64_t, vector_data> &entries) {
  if (m_mode != backend_mode::kExternal) return false;
  auto rebuilt = std::make_unique<diskann_native_state>(
      m_dimension, m_metric, m_index_name, m_diskann_build_complexity,
      m_diskann_max_degree, m_diskann_build_threads);
  bool rebuild_ok = rebuilt->rebuild_from_entries(entries);
  DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_rebuild",
                  rebuild_ok = false;);
  if (!rebuild_ok) {
    if (rebuilt->supported()) return false;
    if (!m_external_adapter.load_committed_entries(entries)) return false;
    (void)rebuilt->set_search_complexity(m_diskann_search_complexity);
    m_entries = entries;
    m_external_adapter_active = true;
    m_native_state = std::move(rebuilt);
    m_native_runtime_enabled = false;
    return true;
  }
  bool search_complexity_ok =
      rebuilt->set_search_complexity(m_diskann_search_complexity);
  DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_set_search_complexity",
                  search_complexity_ok = false;);
  if (!search_complexity_ok) return false;
  if (!remove_diskann_external_artifacts(m_index_name, false))
    return false;
  m_entries = entries;
  m_external_adapter_active = false;
  m_native_state = std::move(rebuilt);
  m_native_runtime_enabled = true;
  return true;
}

bool diskann_backend::recover() {
  if (m_mode != backend_mode::kExternal) return false;
  if (!m_external_adapter.recover()) return false;
  m_entries = m_external_adapter.external_snapshot_entries();
  m_external_adapter_active = true;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

bool diskann_backend::recover_committed_entries(
    const std::unordered_map<uint64_t, vector_data> &entries) {
  if (m_mode != backend_mode::kExternal) return false;
  /*
    The sidecar is a rebuildable serving artifact. Recovery must reconcile it
    with the committed truth rows handed in by the registry instead of trusting
    sidecar contents as authoritative.
  */
  (void)recover();
  return rebuild_from_committed_entries(entries);
}

bool diskann_backend::last_recover_used_fallback() const {
  return m_external_adapter.last_recover_used_fallback();
}

size_t diskann_backend::entry_count() const {
  return m_entries.size();
}

bool diskann_backend::external_manifest_present() const {
  return m_external_adapter_active &&
         m_external_adapter.external_manifest_present();
}

uint64_t diskann_backend::external_manifest_generation() const {
  return m_external_adapter_active
             ? m_external_adapter.external_manifest_generation()
             : 0;
}

std::string diskann_backend::backend_variant() const {
  return "diskann_garnet";
}

bool diskann_backend::set_diskann_build_params(uint32_t diskann_max_degree,
                                               uint32_t diskann_build_complexity,
                                               uint32_t diskann_build_threads) {
  if (m_mode != backend_mode::kExternal || diskann_max_degree == 0 ||
      diskann_build_complexity == 0 ||
      diskann_build_threads > vector_index::k_max_build_threads) {
    return false;
  }
  m_diskann_max_degree = diskann_max_degree;
  m_diskann_build_complexity = diskann_build_complexity;
  m_diskann_build_threads = diskann_build_threads;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

bool diskann_backend::set_diskann_build_threads(
    uint32_t diskann_build_threads) {
  if (m_mode != backend_mode::kExternal ||
      diskann_build_threads > vector_index::k_max_build_threads) {
    return false;
  }
  m_diskann_build_threads = diskann_build_threads;
  m_native_runtime_enabled = false;
  m_native_state.reset();
  return true;
}

uint32_t diskann_backend::diskann_max_degree() const {
  return m_mode == backend_mode::kExternal ? m_diskann_max_degree : 0;
}

uint32_t diskann_backend::diskann_build_complexity() const {
  return m_mode == backend_mode::kExternal ? m_diskann_build_complexity : 0;
}

uint32_t diskann_backend::diskann_build_threads() const {
  return m_mode == backend_mode::kExternal ? m_diskann_build_threads : 0;
}

bool diskann_backend::set_diskann_search_complexity(
    uint32_t diskann_search_complexity) {
  if (m_mode != backend_mode::kExternal || diskann_search_complexity == 0) {
    return false;
  }
  bool native_search_complexity_ok = true;
  if (m_native_runtime_enabled && m_native_state != nullptr) {
    native_search_complexity_ok =
        m_native_state->set_search_complexity(diskann_search_complexity);
    DBUG_EXECUTE_IF("vector_backend_fail_diskann_native_live_search_complexity",
                    native_search_complexity_ok = false;);
  }
  if (m_native_runtime_enabled && m_native_state != nullptr &&
      !native_search_complexity_ok) {
    return false;
  }
  m_diskann_search_complexity = diskann_search_complexity;
  return true;
}

uint32_t diskann_backend::diskann_search_complexity() const {
  return m_mode == backend_mode::kExternal ? m_diskann_search_complexity : 0;
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
std::string diskann_term_directory_for_testing(const std::string &index_name,
                                           uint64_t ctx) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0);
  return state.term_directory_for_testing(ctx);
}

std::string diskann_key_path_for_testing(const std::string &index_name, uint64_t ctx,
                                     const std::string &key_bytes) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0);
  if (key_bytes.empty()) return state.key_path_for_testing(ctx, nullptr, 0);
  return state.key_path_for_testing(
      ctx, reinterpret_cast<const uint8_t *>(key_bytes.data()),
      key_bytes.size());
}

bool diskann_load_value_for_testing(const std::string &index_name, uint64_t ctx,
                                const std::string &key_bytes,
                                std::string *value) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0);
  if (key_bytes.empty()) return state.load_value_for_testing(ctx, nullptr, 0, value);
  return state.load_value_for_testing(
      ctx, reinterpret_cast<const uint8_t *>(key_bytes.data()),
      key_bytes.size(), value);
}

bool diskann_save_value_for_testing(const std::string &index_name, uint64_t ctx,
                                const std::string &key_bytes,
                                const std::string &value) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0);
  const uint8_t *key_ptr = key_bytes.empty()
                               ? nullptr
                               : reinterpret_cast<const uint8_t *>(key_bytes.data());
  return state.save_value_for_testing(ctx, key_ptr, key_bytes.size(),
                                   reinterpret_cast<const uint8_t *>(value.data()),
                                   value.size());
}

bool diskann_delete_value_for_testing(const std::string &index_name, uint64_t ctx,
                                  const std::string &key_bytes) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0);
  const uint8_t *key_ptr = key_bytes.empty()
                               ? nullptr
                               : reinterpret_cast<const uint8_t *>(key_bytes.data());
  return state.delete_value_for_testing(ctx, key_ptr, key_bytes.size());
}

bool diskann_build_memory_store_round_trip_for_testing(
    const std::string &index_name, uint64_t ctx, const std::string &key_bytes,
    const std::string &value, std::string *loaded_before_flush,
    bool *file_exists_before_flush, std::string *loaded_after_flush) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0);
  const uint8_t *key_ptr = key_bytes.empty()
                               ? nullptr
                               : reinterpret_cast<const uint8_t *>(key_bytes.data());
  return state.build_memory_store_round_trip_for_testing(
      ctx, key_ptr, key_bytes.size(), value, loaded_before_flush,
      file_exists_before_flush, loaded_after_flush);
}

bool diskann_resident_store_round_trip_for_testing(
    const std::string &index_name, uint64_t ctx, const std::string &key_bytes,
    const std::string &value, std::string *loaded_after_removing_file) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0);
  const uint8_t *key_ptr = key_bytes.empty()
                               ? nullptr
                               : reinterpret_cast<const uint8_t *>(key_bytes.data());
  return state.resident_store_round_trip_for_testing(
      ctx, key_ptr, key_bytes.size(), value, loaded_after_removing_file);
}

bool diskann_read_modify_write_round_trip_for_testing(
    const std::string &index_name, uint64_t ctx, const std::string &key_bytes,
    const std::string &initial_value, const std::string &patch_value,
    size_t write_length, std::string *persistent_value,
    std::string *build_memory_value, std::string *resident_value) {
  diskann_native_state state(2, metric_type::kEuclidean, index_name,
                           kDiskAnnBuildComplexity, kDiskAnnMaxDegree, 0);
  const uint8_t *key_ptr = key_bytes.empty()
                               ? nullptr
                               : reinterpret_cast<const uint8_t *>(key_bytes.data());
  return state.read_modify_write_round_trip_for_testing(
      ctx, key_ptr, key_bytes.size(), initial_value, patch_value, write_length,
      persistent_value, build_memory_value, resident_value);
}

bool diskann_parse_prefixed_keys_for_testing(const std::string &payload,
                                        uint32_t key_count,
                                        std::vector<std::string> *keys,
                                        int fail_index) {
  if (keys == nullptr) return false;
  keys->clear();
  return diskann_native_state::parse_prefixed_keys_for_testing(
      reinterpret_cast<const uint8_t *>(payload.data()), payload.size(),
      key_count,
      [keys, fail_index](uint32_t index, const uint8_t *key, size_t length) {
        if (static_cast<int>(index) == fail_index) return false;
        keys->emplace_back(reinterpret_cast<const char *>(key), length);
        return true;
      });
}

bool diskann_api_load_for_testing() {
  diskann_api api;
  return api.load();
}

bool diskann_api_available_for_testing(bool has_handle, bool has_create_index,
                                   bool has_drop_index, bool has_insert,
                                   bool has_bulk_insert [[maybe_unused]],
                                   bool has_search_vector, bool has_remove,
                                   bool has_card) {
  diskann_api api;
  api.handle = has_handle ? reinterpret_cast<void *>(1) : nullptr;
  api.create_index = has_create_index
                         ? reinterpret_cast<diskann_create_index_fn>(1)
                         : nullptr;
  api.drop_index =
      has_drop_index ? reinterpret_cast<diskann_drop_index_fn>(1) : nullptr;
  api.insert = has_insert ? reinterpret_cast<diskann_insert_fn>(1) : nullptr;
  api.search_vector = has_search_vector
                          ? reinterpret_cast<diskann_search_vector_fn>(1)
                          : nullptr;
  api.remove = has_remove ? reinterpret_cast<diskann_remove_fn>(1) : nullptr;
  api.card = has_card ? reinterpret_cast<diskann_card_fn>(1) : nullptr;
  return api.available();
}

int32_t diskann_metric_code_for_testing(metric_type metric) {
  return diskann_metric_code(metric);
}

bool parse_diskann_doc_id_for_testing(const uint8_t *data, size_t length,
                                 uint64_t *doc_id) {
  return parse_diskann_doc_id(data, length, doc_id);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index
