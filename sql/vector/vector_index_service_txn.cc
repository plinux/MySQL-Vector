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

#include "sql/vector/vector_index_service.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "my_sys.h"
#include "sql/mysqld.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_service_internal.h"
#include "sql/vector/vector_mapped_search.h"

#ifndef O_BINARY
#define O_BINARY 0
#endif

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

namespace vector_index {

namespace {

using detail::build_backend_from_config;

constexpr const char *LIFECYCLE_BULK_LOADING = "bulk_loading";
constexpr uint32_t ERROR_BACKEND_APPLY_FAILED = 1006;
constexpr uint32_t ERROR_MUTATION_NOT_SUPPORTED = 1005;

uint64_t now_unix_epoch_seconds() {
  return static_cast<uint64_t>(std::time(nullptr));
}

void mark_lifecycle_state(vector_index::index_service::lifecycle_info *lifecycle,
                          const char *state) {
  lifecycle->state = state;
  ++lifecycle->version;
}

void mark_lifecycle_failure(vector_index::index_service::lifecycle_info *lifecycle,
                            uint32_t error_code) {
  mark_lifecycle_state(lifecycle, "failed");
  lifecycle->last_error_code = error_code;
  lifecycle->last_error_ts = now_unix_epoch_seconds();
}

template <typename... Bools>
bool all_true(Bools... values) {
  bool result = true;
  ((result &= static_cast<bool>(values)), ...);
  return result;
}

bool should_batch_rebuild_on_commit(const backend *index_backend) {
  if (index_backend == nullptr) return false;
  if (index_backend->mode() != backend_mode::kExternal) return false;
  return index_backend->provider() == backend_provider::kFaiss;
}

bool pending_change_snapshots_equal(
    const std::vector<index_service::pending_change_snapshot> &lhs,
    const std::vector<index_service::pending_change_snapshot> &rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].index_name != rhs[i].index_name || lhs[i].erase != rhs[i].erase ||
        lhs[i].doc_id != rhs[i].doc_id || lhs[i].vector != rhs[i].vector) {
      return false;
    }
  }
  return true;
}

bool lifecycle_is_bulk_loading(const index_service::lifecycle_info &lifecycle) {
  return lifecycle.state == LIFECYCLE_BULK_LOADING;
}

bool pending_budget_allows(size_t current_bytes, size_t additional_bytes) {
  if (opt_vector_pending_cache_size == 0) return true;
  if (current_bytes > opt_vector_pending_cache_size) return false;
  return additional_bytes <= opt_vector_pending_cache_size - current_bytes;
}

class pending_spill_guard {
 public:
  explicit pending_spill_guard(std::string path) : m_path(std::move(path)) {}

  pending_spill_guard(const pending_spill_guard &) = delete;
  pending_spill_guard &operator=(const pending_spill_guard &) = delete;

  ~pending_spill_guard() {
    if (m_path.empty()) return;
    std::error_code ignored;
    std::filesystem::remove(m_path, ignored);
  }

  void release() { m_path.clear(); }

 private:
  std::string m_path;
};

committed_entry_reader make_commit_rebuild_reader(
    const vector_entry_store &entry_store, const std::string &index_name,
    const std::vector<index_service::pending_change_snapshot> &changes) {
  return [&entry_store, &index_name,
          &changes](const committed_entry_visitor &visitor) {
    if (!visitor) return false;

    std::map<uint64_t, const index_service::pending_change_snapshot *> overlay;
    for (const index_service::pending_change_snapshot &change : changes) {
      overlay[change.doc_id] = &change;
    }

    if (!entry_store.for_each_committed_entry(
            index_name,
            [&](uint64_t doc_id, const vector_data &committed_vector) {
              const auto overlay_it = overlay.find(doc_id);
              if (overlay_it == overlay.end())
                return visitor(doc_id, committed_vector);
              const index_service::pending_change_snapshot *change =
                  overlay_it->second;
              return change->erase ? true : visitor(doc_id, change->vector);
            })) {
      return false;
    }

    for (const auto &overlay_entry : overlay) {
      const index_service::pending_change_snapshot *change =
          overlay_entry.second;
      if (change->erase) continue;
      vector_data committed_vector;
      bool committed_found = false;
      if (!entry_store.find_committed_entry(index_name, overlay_entry.first,
                                            &committed_vector,
                                            &committed_found)) {
        return false;
      }
      if (committed_found) continue;
      if (!visitor(change->doc_id, change->vector)) return false;
    }
    return true;
  };
}

bool apply_entry_store_change(vector_entry_store *entry_store,
                              const index_service::pending_change_snapshot &change) {
  if (entry_store == nullptr) return false;
  if (!change.erase) {
    return entry_store->upsert(change.index_name, change.doc_id, change.vector);
  }
  return entry_store->erase(change.index_name, change.doc_id);
}

}  // namespace

size_t index_service::pending_change_memory_bytes(
    const pending_change &change) {
  if (change.type != change_type::kUpsert || !change.vector_spill_path.empty()) {
    return 0;
  }
  return change.vector.size() * sizeof(float);
}

bool index_service::read_pending_change_vector(const pending_change &change,
                                               vector_data *vector) {
  if (vector == nullptr || change.type != change_type::kUpsert) return false;
  if (change.vector_spill_path.empty()) {
    *vector = change.vector;
    return true;
  }

  std::error_code error;
  const uintmax_t file_bytes =
      std::filesystem::file_size(change.vector_spill_path, error);
  if (error || file_bytes < sizeof(uint64_t)) return false;

  std::ifstream input(change.vector_spill_path, std::ios::binary);
  uint64_t element_count = 0;
  if (!input.read(reinterpret_cast<char *>(&element_count),
                  sizeof(element_count))) {
    return false;
  }
  if (element_count >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max() /
                            sizeof(float))) {
    return false;
  }

  const uintmax_t payload_bytes =
      static_cast<uintmax_t>(element_count) * sizeof(float);
  if (file_bytes != sizeof(element_count) + payload_bytes) return false;

  vector->resize(static_cast<size_t>(element_count));
  if (payload_bytes == 0) return true;
  return static_cast<bool>(
      input.read(reinterpret_cast<char *>(vector->data()), payload_bytes));
}

bool index_service::write_pending_change_spill(
    uint64_t txn_id, const std::string &index_name, uint64_t doc_id,
    const vector_data &vector, std::string *path) {
  if (path == nullptr) return false;
  static std::atomic<uint64_t> spill_sequence{0};

  std::error_code error;
  std::filesystem::path directory =
      mysql_tmpdir != nullptr ? std::filesystem::path(mysql_tmpdir)
                              : std::filesystem::temp_directory_path(error);
  if (error) return false;
  directory /= "mysql-vector-pending-spill";
  std::filesystem::create_directories(directory, error);
  if (error) return false;

  const uint64_t element_count = static_cast<uint64_t>(vector.size());
  const auto *element_count_bytes =
      reinterpret_cast<const uchar *>(&element_count);
  const auto *vector_bytes = reinterpret_cast<const uchar *>(vector.data());
  const size_t vector_byte_count = vector.size() * sizeof(float);

  for (uint32_t attempt = 0; attempt < 32; ++attempt) {
    const uint64_t sequence =
        spill_sequence.fetch_add(1, std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const size_t index_hash = std::hash<std::string>{}(index_name);
    std::ostringstream file_name;
    file_name << "mysql-vector-pending-spill-" << now << "-" << sequence << "-"
              << attempt << "-" << txn_id << "-" << index_hash << "-" << doc_id
              << ".bin";

    const std::filesystem::path spill_path = directory / file_name.str();
    const std::string spill_path_text = spill_path.string();
    File fd = my_create(spill_path_text.c_str(), 0600,
                        O_WRONLY | O_EXCL | O_BINARY | O_NOFOLLOW, MYF(0));
    if (fd < 0) continue;

    bool ok = my_write(fd, element_count_bytes, sizeof(element_count),
                       MYF(0)) == sizeof(element_count);
    if (ok && vector_byte_count != 0) {
      ok = my_write(fd, vector_bytes, vector_byte_count, MYF(0)) ==
           vector_byte_count;
    }
    if (my_close(fd, MYF(0)) != 0) ok = false;

    if (ok) {
      *path = spill_path_text;
      return true;
    }

    (void)my_delete(spill_path_text.c_str(), MYF(0));
  }

  return false;
}

void index_service::remove_pending_change_spill(const pending_change &change) {
  if (change.vector_spill_path.empty()) return;
  std::error_code ignored;
  std::filesystem::remove(change.vector_spill_path, ignored);
}

void index_service::remove_pending_change_spills(
    const std::vector<pending_change> &changes, size_t first_change) {
  if (first_change >= changes.size()) return;
  for (auto change_it = changes.begin() + first_change;
       change_it != changes.end(); ++change_it) {
    remove_pending_change_spill(*change_it);
  }
}

bool index_service::stage_upsert(uint64_t txn_id, const std::string &index_name,
                               uint64_t doc_id, const vector_data &vector) {
  auto index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end()) return false;
  if (index_it->second->dimension() != vector.size()) return false;

  const size_t vector_bytes = vector.size() * sizeof(float);
  pending_change change{index_name, change_type::kUpsert, doc_id, vector, {}};
  if (!pending_budget_allows(pending_vector_memory_bytes(txn_id),
                             vector_bytes)) {
    std::string spill_path;
    if (!write_pending_change_spill(txn_id, index_name, doc_id, vector,
                                    &spill_path)) {
      return false;
    }
    change.vector.clear();
    change.vector.shrink_to_fit();
    pending_spill_guard spill_guard(spill_path);
    change.vector_spill_path = std::move(spill_path);
    m_pending_changes[txn_id].push_back(std::move(change));
    spill_guard.release();
    return true;
  }

  m_pending_changes[txn_id].push_back(std::move(change));
  return true;
}

bool index_service::stage_erase(uint64_t txn_id, const std::string &index_name,
                              uint64_t doc_id) {
  if (m_indexes.find(index_name) == m_indexes.end()) return false;

  m_pending_changes[txn_id].push_back(
      pending_change{index_name, change_type::kErase, doc_id, {}, {}});
  return true;
}

bool index_service::commit(uint64_t txn_id) {
  commit_build_plan plan;
  return snapshot_commit_build_plan(txn_id, &plan) &&
         build_commit_backends(&plan) &&
         apply_commit_build_plan(txn_id, &plan);
}

bool index_service::snapshot_commit_build_plan(uint64_t txn_id,
                                               commit_build_plan *plan) {
  if (plan == nullptr) return false;
  plan->changes.clear();
  plan->rebuilds.clear();

  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it == m_pending_changes.end()) return true;

  auto mark_failure_for_index = [&](const std::string &index_name,
                                    uint32_t error_code) {
    auto lifecycle_it = m_lifecycle_infos.find(index_name);
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_lifecycle_failure(&lifecycle_it->second, error_code);
    }
  };

  if (!snapshot_pending_changes(txn_id, &plan->changes)) return false;

  for (const pending_change &change : pending_it->second) {
    auto index_it = m_indexes.find(change.index_name);
    auto lifecycle_it = m_lifecycle_infos.find(change.index_name);
    if (index_it == m_indexes.end()) return false;
    if (change.type == change_type::kUpsert) {
      vector_data vector;
      if (!read_pending_change_vector(change, &vector) ||
          index_it->second->dimension() != vector.size()) {
        mark_failure_for_index(change.index_name, ERROR_BACKEND_APPLY_FAILED);
        return false;
      }
    }
    const bool bulk_loading =
        lifecycle_it != m_lifecycle_infos.end() &&
        lifecycle_is_bulk_loading(lifecycle_it->second);
    if (!bulk_loading) {
      if (!ensure_runtime_loaded(change.index_name)) {
        mark_failure_for_index(change.index_name, ERROR_BACKEND_APPLY_FAILED);
        return false;
      }
      index_it = m_indexes.find(change.index_name);
      if (index_it == m_indexes.end()) return false;
      if (!index_it->second->supports_mutations()) {
        mark_failure_for_index(change.index_name, ERROR_MUTATION_NOT_SUPPORTED);
        return false;
      }
    }
  }

  std::unordered_map<std::string, std::vector<const pending_change_snapshot *>>
      grouped_changes;
  grouped_changes.reserve(plan->changes.size());
  for (const pending_change_snapshot &change : plan->changes) {
    grouped_changes[change.index_name].push_back(&change);
  }

  for (const auto &entry : grouped_changes) {
    auto index_it = m_indexes.find(entry.first);
    auto config_it = m_index_configs.find(entry.first);
    if (!all_true(index_it != m_indexes.end(),
                  config_it != m_index_configs.end(),
                  m_entry_store.has_index(entry.first))) {
      return false;
    }

    auto lifecycle_it = m_lifecycle_infos.find(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end() &&
        lifecycle_is_bulk_loading(lifecycle_it->second)) {
      continue;
    }
    if (!should_batch_rebuild_on_commit(index_it->second.get())) continue;

    commit_rebuild_plan rebuild_plan;
    rebuild_plan.index_name = entry.first;
    rebuild_plan.config = config_it->second;
    rebuild_plan.before_generation = m_entry_store.generation(entry.first);
    rebuild_plan.changes.reserve(entry.second.size());
    for (const pending_change_snapshot *change : entry.second) {
      rebuild_plan.changes.push_back(*change);
    }
    plan->rebuilds.push_back(std::move(rebuild_plan));
  }
  return true;
}

bool index_service::build_commit_backends(commit_build_plan *plan) const {
  if (plan == nullptr) return false;
  for (commit_rebuild_plan &rebuild_plan : plan->rebuilds) {
    std::unique_ptr<backend> rebuilt =
        build_backend_from_config(rebuild_plan.index_name, rebuild_plan.config);
    const committed_entry_reader reader = make_commit_rebuild_reader(
        m_entry_store, rebuild_plan.index_name, rebuild_plan.changes);
    if (rebuilt == nullptr ||
        !rebuilt->rebuild_from_committed_entries_from_reader(reader)) {
      return false;
    }
    rebuild_plan.rebuilt_backend = std::move(rebuilt);
  }
  return true;
}

bool index_service::apply_commit_build_plan(uint64_t txn_id,
                                            commit_build_plan *plan) {
  if (plan == nullptr) return false;

  std::vector<pending_change_snapshot> current_changes;
  if (!snapshot_pending_changes(txn_id, &current_changes) ||
      !pending_change_snapshots_equal(plan->changes, current_changes)) {
    return false;
  }

  auto mark_failure_for_index = [&](const std::string &index_name,
                                    uint32_t error_code) {
    auto lifecycle_it = m_lifecycle_infos.find(index_name);
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_lifecycle_failure(&lifecycle_it->second, error_code);
    }
  };

  std::unordered_set<std::string> rebuild_indexes;
  rebuild_indexes.reserve(plan->rebuilds.size());
  for (const commit_rebuild_plan &rebuild_plan : plan->rebuilds) {
    rebuild_indexes.insert(rebuild_plan.index_name);
  }

  for (const commit_rebuild_plan &rebuild_plan : plan->rebuilds) {
    if (rebuild_plan.rebuilt_backend == nullptr) return false;
    auto index_it = m_indexes.find(rebuild_plan.index_name);
    auto config_it = m_index_configs.find(rebuild_plan.index_name);
    if (!all_true(index_it != m_indexes.end(), config_it != m_index_configs.end(),
                  m_entry_store.has_index(rebuild_plan.index_name),
                  m_entry_store.generation(rebuild_plan.index_name) ==
                      rebuild_plan.before_generation,
                  config_it->second.dimension == rebuild_plan.config.dimension,
                  config_it->second.metric == rebuild_plan.config.metric,
                  config_it->second.mode == rebuild_plan.config.mode,
                  config_it->second.provider == rebuild_plan.config.provider)) {
      return false;
    }
  }

  for (const pending_change_snapshot &change : plan->changes) {
    auto index_it = m_indexes.find(change.index_name);
    if (index_it == m_indexes.end()) return false;
    auto lifecycle_it = m_lifecycle_infos.find(change.index_name);
    if (lifecycle_it != m_lifecycle_infos.end() &&
        lifecycle_is_bulk_loading(lifecycle_it->second)) {
      continue;
    }
    if (rebuild_indexes.find(change.index_name) != rebuild_indexes.end()) {
      continue;
    }

    bool ok = false;
    {
      std::unique_lock<std::shared_mutex> runtime_guard(
          index_it->second->runtime_mutex());
      ok = !change.erase ? index_it->second->upsert(change.doc_id, change.vector)
                         : index_it->second->erase(change.doc_id);
    }
    if (!ok) {
      mark_failure_for_index(change.index_name, ERROR_BACKEND_APPLY_FAILED);
      return false;
    }
  }

  for (commit_rebuild_plan &rebuild_plan : plan->rebuilds) {
    auto index_it = m_indexes.find(rebuild_plan.index_name);
    if (!all_true(index_it != m_indexes.end(),
                  m_entry_store.has_index(rebuild_plan.index_name))) {
      return false;
    }
    for (const pending_change_snapshot &change : rebuild_plan.changes) {
      if (!apply_entry_store_change(&m_entry_store, change)) {
        return false;
      }
    }
    index_it->second = std::move(rebuild_plan.rebuilt_backend);
    maybe_unload_runtime(rebuild_plan.index_name);
  }

  for (const pending_change_snapshot &change : plan->changes) {
    if (rebuild_indexes.find(change.index_name) != rebuild_indexes.end()) {
      continue;
    }
    if (!m_entry_store.has_index(change.index_name)) return false;
    if (!apply_entry_store_change(&m_entry_store, change)) return false;
  }

  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it != m_pending_changes.end()) {
    remove_pending_change_spills(pending_it->second, 0);
    m_pending_changes.erase(pending_it);
  }
  m_savepoints.erase(txn_id);
  return true;
}

void index_service::rollback(uint64_t txn_id) {
  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it != m_pending_changes.end()) {
    remove_pending_change_spills(pending_it->second, 0);
    m_pending_changes.erase(pending_it);
  }
  m_savepoints.erase(txn_id);
}

bool index_service::savepoint(uint64_t txn_id, const std::string &name) {
  if (name.empty()) return false;

  const size_t pending_count = pending_change_count(txn_id);
  std::vector<savepoint_marker> &savepoints = m_savepoints[txn_id];
  savepoints.erase(std::remove_if(savepoints.begin(), savepoints.end(),
                                  [&](const savepoint_marker &marker) {
                                    return marker.name == name;
                                  }),
                   savepoints.end());
  savepoints.push_back(savepoint_marker{name, pending_count});
  return true;
}

bool index_service::rollback_to_savepoint(uint64_t txn_id, const std::string &name) {
  auto savepoint_it = m_savepoints.find(txn_id);
  if (savepoint_it == m_savepoints.end() || name.empty()) return false;

  std::vector<savepoint_marker> &savepoints = savepoint_it->second;
  auto marker_it = std::find_if(savepoints.begin(), savepoints.end(),
                                [&](const savepoint_marker &marker) {
                                  return marker.name == name;
                                });
  if (marker_it == savepoints.end()) return false;

  std::vector<pending_change> &changes = m_pending_changes[txn_id];
  if (changes.size() < marker_it->change_count) return false;
  remove_pending_change_spills(changes, marker_it->change_count);
  changes.resize(marker_it->change_count);

  savepoints.erase(marker_it + 1, savepoints.end());
  return true;
}

bool index_service::release_savepoint(uint64_t txn_id, const std::string &name) {
  auto savepoint_it = m_savepoints.find(txn_id);
  if (savepoint_it == m_savepoints.end() || name.empty()) return false;

  std::vector<savepoint_marker> &savepoints = savepoint_it->second;
  auto marker_it = std::find_if(savepoints.begin(), savepoints.end(),
                                [&](const savepoint_marker &marker) {
                                  return marker.name == name;
                                });
  if (marker_it == savepoints.end()) return false;

  savepoints.erase(marker_it);
  if (savepoints.empty()) m_savepoints.erase(savepoint_it);
  return true;
}

bool index_service::search(const std::string &index_name, const vector_data &query,
                           size_t top_k,
                           std::vector<search_result> *results) const {
  if (!const_cast<index_service *>(this)->ensure_runtime_loaded_for_search(
          index_name))
    return false;
  return search_loaded(index_name, query, top_k, results);
}

bool index_service::search_loaded(const std::string &index_name,
                                  const vector_data &query, size_t top_k,
                                  std::vector<search_result> *results) const {
  backend_ptr runtime;
  index_config config;
  if (!snapshot_search_backend_loaded(index_name, query, &runtime, &config))
    return false;
  {
    std::shared_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
    if (runtime->search(query, top_k, results)) return true;
  }
  if (!detail::can_rebuild_after_search_failure(config)) return false;

  index_service *self = const_cast<index_service *>(this);
  if (!self->rebuild_runtime_from_store_for_search(index_name)) return false;
  if (!snapshot_search_backend_loaded(index_name, query, &runtime, nullptr))
    return false;
  std::shared_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
  return runtime->search(query, top_k, results);
}

bool index_service::snapshot_search_backend_loaded(
    const std::string &index_name, const vector_data &query,
    backend_ptr *runtime, index_config *config) const {
  if (runtime == nullptr) return false;
  runtime->reset();
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (lifecycle_it != m_lifecycle_infos.end() &&
      lifecycle_is_bulk_loading(lifecycle_it->second)) {
    return false;
  }
  auto config_it = m_index_configs.find(index_name);
  if (config_it == m_index_configs.end() ||
      query.size() != config_it->second.dimension) {
    return false;
  }
  auto index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end() || index_it->second == nullptr)
    return false;
  *runtime = index_it->second;
  if (config != nullptr) *config = config_it->second;
  return true;
}

bool index_service::search_batch(
    const std::string &index_name, const std::vector<vector_data> &queries,
    size_t top_k, std::vector<std::vector<search_result>> *results) const {
  if (!const_cast<index_service *>(this)->ensure_runtime_loaded_for_search(
          index_name))
    return false;
  return search_batch_loaded(index_name, queries, top_k, results);
}

bool index_service::search_batch_loaded(
    const std::string &index_name, const std::vector<vector_data> &queries,
    size_t top_k, std::vector<std::vector<search_result>> *results) const {
  if (results == nullptr) return false;
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (lifecycle_it != m_lifecycle_infos.end() &&
      lifecycle_is_bulk_loading(lifecycle_it->second)) {
    return false;
  }
  auto config_it = m_index_configs.find(index_name);
  if (config_it != m_index_configs.end()) {
    for (const vector_data &query : queries) {
      if (query.size() != config_it->second.dimension) return false;
    }
  }
  auto index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end()) return false;
  backend_ptr runtime = index_it->second;
  {
    std::shared_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
    if (runtime->search_batch(queries, top_k, results)) return true;
  }
  if (config_it == m_index_configs.end() ||
      !detail::can_rebuild_after_search_failure(config_it->second)) {
    return false;
  }
  index_service *self = const_cast<index_service *>(this);
  if (!self->rebuild_runtime_from_store_for_search(index_name)) return false;
  index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end() || index_it->second == nullptr)
    return false;
  runtime = index_it->second;
  std::shared_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
  return runtime->search_batch(queries, top_k, results);
}

bool index_service::search_with_pending(uint64_t txn_id,
                                     const std::string &index_name,
                                     const vector_data &query, size_t top_k,
                                     std::vector<search_result> *results) const {
  if (!const_cast<index_service *>(this)->ensure_runtime_loaded_for_search(
          index_name))
    return false;
  return search_with_pending_loaded(txn_id, index_name, query, top_k, results);
}

bool index_service::search_with_pending_loaded(
    uint64_t txn_id, const std::string &index_name, const vector_data &query,
    size_t top_k, std::vector<search_result> *results) const {
  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it == m_pending_changes.end() || pending_it->second.empty()) {
    return search_loaded(index_name, query, top_k, results);
  }

  auto config_it = m_index_configs.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  committed_entries committed_entries;
  if (config_it == m_index_configs.end() ||
      !m_entry_store.snapshot_index(index_name, &committed_entries) ||
      (lifecycle_it != m_lifecycle_infos.end() &&
       lifecycle_is_bulk_loading(lifecycle_it->second))) {
    return false;
  }

  std::unique_ptr<backend> backend =
      build_backend_from_config(index_name, config_it->second);
  if (backend == nullptr) return false;
  if (!backend->load_committed_entries(committed_entries)) return false;

  for (const pending_change &change : pending_it->second) {
    if (change.index_name != index_name) continue;
    bool ok = false;
    if (change.type == change_type::kUpsert) {
      vector_data vector;
      ok = read_pending_change_vector(change, &vector) &&
           backend->upsert(change.doc_id, vector);
    } else {
      ok = backend->erase(change.doc_id);
    }
    if (!ok) return false;
  }

  return backend->search(query, top_k, results);
}

size_t index_service::pending_change_count(uint64_t txn_id) const {
  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it == m_pending_changes.end()) return 0;
  return pending_it->second.size();
}

bool index_service::has_pending_changes_for_index(
    const std::string &index_name) const {
  for (const auto &txn_changes : m_pending_changes) {
    for (const pending_change &change : txn_changes.second) {
      if (change.index_name == index_name) return true;
    }
  }
  return false;
}

bool index_service::pending_change_index_names(
    uint64_t txn_id, std::vector<std::string> *index_names) const {
  if (index_names == nullptr) return false;
  index_names->clear();

  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it == m_pending_changes.end()) return true;

  std::unordered_set<std::string> dedupe;
  dedupe.reserve(pending_it->second.size());
  for (const pending_change &change : pending_it->second) {
    dedupe.insert(change.index_name);
  }

  index_names->reserve(dedupe.size());
  for (const std::string &name : dedupe) {
    index_names->push_back(name);
  }
  std::sort(index_names->begin(), index_names->end());
  return true;
}

bool index_service::snapshot_pending_changes(
    uint64_t txn_id, std::vector<pending_change_snapshot> *changes) const {
  if (changes == nullptr) return false;
  changes->clear();

  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it == m_pending_changes.end()) return true;

  changes->reserve(pending_it->second.size());
  for (const pending_change &change : pending_it->second) {
    vector_data vector;
    if (change.type == change_type::kUpsert &&
        !read_pending_change_vector(change, &vector)) {
      changes->clear();
      return false;
    }
    changes->push_back(pending_change_snapshot{
        change.index_name, change.type == change_type::kErase, change.doc_id,
        std::move(vector)});
  }
  return true;
}

bool index_service::snapshot_pending_change_delta(
    uint64_t txn_id, std::vector<pending_change_snapshot> *changes) const {
  if (changes == nullptr) return false;
  changes->clear();

  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it == m_pending_changes.end()) return true;

  using delta_key = std::pair<std::string, uint64_t>;
  std::map<delta_key, pending_change_snapshot> deltas;
  std::map<delta_key, std::pair<bool, vector_data>> committed_cache;
  for (const pending_change &change : pending_it->second) {
    const delta_key key{change.index_name, change.doc_id};
    auto committed_it = committed_cache.find(key);
    if (committed_it == committed_cache.end()) {
      vector_data committed_vector;
      bool found = false;
      if (!m_entry_store.find_committed_entry(change.index_name, change.doc_id,
                                             &committed_vector, &found)) {
        return false;
      }
      committed_it =
          committed_cache.emplace(key, std::make_pair(found, committed_vector))
              .first;
    }
    const bool existed_before = committed_it->second.first;

    if (change.type == change_type::kUpsert) {
      vector_data vector;
      if (!read_pending_change_vector(change, &vector)) return false;
      if (existed_before && committed_it->second.second == vector) {
        deltas.erase(key);
      } else {
        deltas[key] = pending_change_snapshot{
            change.index_name, false, change.doc_id, std::move(vector)};
      }
      continue;
    }

    if (existed_before) {
      deltas[key] =
          pending_change_snapshot{change.index_name, true, change.doc_id, {}};
    } else {
      deltas.erase(key);
    }
  }

  changes->reserve(deltas.size());
  for (const auto &entry : deltas) {
    changes->push_back(entry.second);
  }
  return true;
}

bool index_service::restore_pending_changes(
    uint64_t txn_id, const std::vector<pending_change_snapshot> &changes) {
  if (changes.empty()) {
    auto pending_it = m_pending_changes.find(txn_id);
    if (pending_it != m_pending_changes.end()) {
      remove_pending_change_spills(pending_it->second, 0);
      m_pending_changes.erase(pending_it);
    }
    m_savepoints.erase(txn_id);
    return true;
  }

  std::vector<pending_change> restored;
  restored.reserve(changes.size());
  size_t restored_memory_bytes = 0;
  for (const pending_change_snapshot &change : changes) {
    auto index_it = m_indexes.find(change.index_name);
    if (index_it == m_indexes.end()) {
      remove_pending_change_spills(restored, 0);
      return false;
    }
    if (!change.erase && index_it->second->dimension() != change.vector.size()) {
      remove_pending_change_spills(restored, 0);
      return false;
    }
    pending_change restored_change{change.index_name,
                                   change.erase ? change_type::kErase
                                                : change_type::kUpsert,
                                   change.doc_id, change.vector, {}};
    if (!change.erase) {
      const size_t vector_bytes = change.vector.size() * sizeof(float);
      if (pending_budget_allows(restored_memory_bytes, vector_bytes)) {
        restored_memory_bytes += vector_bytes;
      } else {
        std::string spill_path;
        if (!write_pending_change_spill(txn_id, change.index_name,
                                        change.doc_id, change.vector,
                                        &spill_path)) {
          remove_pending_change_spills(restored, 0);
          return false;
        }
        restored_change.vector.clear();
        restored_change.vector.shrink_to_fit();
        restored_change.vector_spill_path = std::move(spill_path);
      }
    }
    restored.push_back(std::move(restored_change));
  }

  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it != m_pending_changes.end()) {
    remove_pending_change_spills(pending_it->second, 0);
  }
  m_pending_changes[txn_id] = std::move(restored);
  m_savepoints.erase(txn_id);
  return true;
}

bool index_service::snapshot_pending_state(uint64_t txn_id,
                                           pending_state_snapshot *state) const {
  if (state == nullptr) return false;
  state->changes.clear();
  state->savepoints.clear();
  if (!snapshot_pending_changes(txn_id, &state->changes)) return false;

  auto savepoint_it = m_savepoints.find(txn_id);
  if (savepoint_it == m_savepoints.end()) return true;
  state->savepoints.reserve(savepoint_it->second.size());
  for (const savepoint_marker &marker : savepoint_it->second) {
    state->savepoints.push_back(
        pending_savepoint_snapshot{marker.name, marker.change_count});
  }
  return true;
}

bool index_service::restore_pending_state(uint64_t txn_id,
                                          const pending_state_snapshot &state) {
  if (!restore_pending_changes(txn_id, state.changes)) return false;
  if (state.savepoints.empty()) {
    m_savepoints.erase(txn_id);
    return true;
  }

  std::vector<savepoint_marker> restored_savepoints;
  restored_savepoints.reserve(state.savepoints.size());
  const size_t pending_count = pending_change_count(txn_id);
  for (const pending_savepoint_snapshot &marker : state.savepoints) {
    if (marker.name.empty() || marker.change_count > pending_count) return false;
    restored_savepoints.push_back(
        savepoint_marker{marker.name, marker.change_count});
  }
  m_savepoints[txn_id] = std::move(restored_savepoints);
  return true;
}

}  // namespace vector_index
