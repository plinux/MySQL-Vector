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
#include <chrono>
#include <ctime>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sql/vector/vector_index_service_internal.h"
#include "sql/vector/vector_mapped_search.h"

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

bool lifecycle_is_bulk_loading(const index_service::lifecycle_info &lifecycle) {
  return lifecycle.state == LIFECYCLE_BULK_LOADING;
}

}  // namespace

bool index_service::stage_upsert(uint64_t txn_id, const std::string &index_name,
                               uint64_t doc_id, const vector_data &vector) {
  auto index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end()) return false;
  if (index_it->second->dimension() != vector.size()) return false;

  m_pending_changes[txn_id].push_back(
      pending_change{index_name, change_type::kUpsert, doc_id, vector});
  return true;
}

bool index_service::stage_erase(uint64_t txn_id, const std::string &index_name,
                              uint64_t doc_id) {
  if (m_indexes.find(index_name) == m_indexes.end()) return false;

  m_pending_changes[txn_id].push_back(
      pending_change{index_name, change_type::kErase, doc_id, {}});
  return true;
}

bool index_service::commit(uint64_t txn_id) {
  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it == m_pending_changes.end()) return true;

  auto mark_failure_for_index = [&](const std::string &index_name,
                                    uint32_t error_code) {
    auto lifecycle_it = m_lifecycle_infos.find(index_name);
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_lifecycle_failure(&lifecycle_it->second, error_code);
    }
  };

  for (const pending_change &change : pending_it->second) {
    auto index_it = m_indexes.find(change.index_name);
    auto lifecycle_it = m_lifecycle_infos.find(change.index_name);
    if (index_it == m_indexes.end()) return false;
    if (change.type == change_type::kUpsert &&
        index_it->second->dimension() != change.vector.size()) {
      mark_failure_for_index(change.index_name, ERROR_BACKEND_APPLY_FAILED);
      return false;
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

  std::unordered_map<std::string, std::vector<const pending_change *>>
      grouped_changes;
  grouped_changes.reserve(pending_it->second.size());
  for (const pending_change &change : pending_it->second) {
    grouped_changes[change.index_name].push_back(&change);
  }

  std::unordered_map<std::string, committed_entries> rebuilt_states;
  std::unordered_map<std::string, std::unique_ptr<backend>> rebuilt_backends;

  for (const auto &entry : grouped_changes) {
    auto index_it = m_indexes.find(entry.first);
    auto config_it = m_index_configs.find(entry.first);
    auto state_it = m_committed_entries.find(entry.first);
    if (!all_true(index_it != m_indexes.end(), config_it != m_index_configs.end(),
                  state_it != m_committed_entries.end())) {
      return false;
    }

    auto lifecycle_it = m_lifecycle_infos.find(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end() &&
        lifecycle_is_bulk_loading(lifecycle_it->second)) {
      continue;
    }
    if (!should_batch_rebuild_on_commit(index_it->second.get())) continue;

    committed_entries next_entries = state_it->second;
    for (const pending_change *change : entry.second) {
      if (change->type == change_type::kUpsert) {
        next_entries[change->doc_id] = change->vector;
      } else {
        next_entries.erase(change->doc_id);
      }
    }

    std::unique_ptr<backend> rebuilt =
        build_backend_from_config(entry.first, config_it->second);
    if (rebuilt == nullptr ||
        !rebuilt->rebuild_from_committed_entries(next_entries)) {
      mark_failure_for_index(entry.first, ERROR_BACKEND_APPLY_FAILED);
      return false;
    }

    rebuilt_states.emplace(entry.first, std::move(next_entries));
    rebuilt_backends.emplace(entry.first, std::move(rebuilt));
  }

  for (const pending_change &change : pending_it->second) {
    auto index_it = m_indexes.find(change.index_name);
    if (index_it == m_indexes.end()) return false;
    auto lifecycle_it = m_lifecycle_infos.find(change.index_name);
    if (lifecycle_it != m_lifecycle_infos.end() &&
        lifecycle_is_bulk_loading(lifecycle_it->second)) {
      continue;
    }
    if (rebuilt_backends.find(change.index_name) != rebuilt_backends.end()) {
      continue;
    }

    bool ok = false;
    if (change.type == change_type::kUpsert) {
      ok = index_it->second->upsert(change.doc_id, change.vector);
    } else {
      ok = index_it->second->erase(change.doc_id);
    }
    if (!ok) {
      mark_failure_for_index(change.index_name, ERROR_BACKEND_APPLY_FAILED);
      return false;
    }
  }

  for (auto &entry : rebuilt_backends) {
    auto index_it = m_indexes.find(entry.first);
    auto state_it = m_committed_entries.find(entry.first);
    if (!all_true(index_it != m_indexes.end(), state_it != m_committed_entries.end())) {
      return false;
    }
    index_it->second = std::move(entry.second);
    state_it->second = std::move(rebuilt_states[entry.first]);
    maybe_unload_runtime(entry.first);
  }

  for (const pending_change &change : pending_it->second) {
    if (rebuilt_backends.find(change.index_name) != rebuilt_backends.end()) {
      continue;
    }
    auto state_it = m_committed_entries.find(change.index_name);
    if (state_it == m_committed_entries.end()) return false;
    if (change.type == change_type::kUpsert) {
      state_it->second[change.doc_id] = change.vector;
    } else {
      state_it->second.erase(change.doc_id);
    }
  }

  m_pending_changes.erase(pending_it);
  m_savepoints.erase(txn_id);
  return true;
}

void index_service::rollback(uint64_t txn_id) {
  m_pending_changes.erase(txn_id);
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
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (lifecycle_it != m_lifecycle_infos.end() &&
      lifecycle_is_bulk_loading(lifecycle_it->second)) {
    return false;
  }
  auto index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end()) return false;
  return index_it->second->search(query, top_k, results);
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
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (lifecycle_it != m_lifecycle_infos.end() &&
      lifecycle_is_bulk_loading(lifecycle_it->second)) {
    return false;
  }
  auto index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end()) return false;
  return index_it->second->search_batch(queries, top_k, results);
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
  auto committed_it = m_committed_entries.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (config_it == m_index_configs.end() ||
      committed_it == m_committed_entries.end() ||
      (lifecycle_it != m_lifecycle_infos.end() &&
       lifecycle_is_bulk_loading(lifecycle_it->second))) {
    return false;
  }

  std::unique_ptr<backend> backend =
      build_backend_from_config(index_name, config_it->second);
  if (backend == nullptr) return false;
  if (!backend->load_committed_entries(committed_it->second)) return false;

  for (const pending_change &change : pending_it->second) {
    if (change.index_name != index_name) continue;
    bool ok = false;
    if (change.type == change_type::kUpsert) {
      ok = backend->upsert(change.doc_id, change.vector);
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
    changes->push_back(pending_change_snapshot{
        change.index_name, change.type == change_type::kErase, change.doc_id,
        change.vector});
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
  for (const pending_change &change : pending_it->second) {
    auto committed_it = m_committed_entries.find(change.index_name);
    if (committed_it == m_committed_entries.end()) return false;

    const delta_key key{change.index_name, change.doc_id};
    const auto doc_before_it = committed_it->second.find(change.doc_id);
    const bool existed_before = doc_before_it != committed_it->second.end();

    if (change.type == change_type::kUpsert) {
      if (existed_before && doc_before_it->second == change.vector) {
        deltas.erase(key);
      } else {
        deltas[key] = pending_change_snapshot{
            change.index_name, false, change.doc_id, change.vector};
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
    m_pending_changes.erase(txn_id);
    m_savepoints.erase(txn_id);
    return true;
  }

  std::vector<pending_change> restored;
  restored.reserve(changes.size());
  for (const pending_change_snapshot &change : changes) {
    auto index_it = m_indexes.find(change.index_name);
    if (index_it == m_indexes.end()) return false;
    if (!change.erase && index_it->second->dimension() != change.vector.size()) {
      return false;
    }
    restored.push_back(pending_change{change.index_name,
                                     change.erase ? change_type::kErase
                                                  : change_type::kUpsert,
                                     change.doc_id, change.vector});
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
