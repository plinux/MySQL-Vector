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
#include <ctime>
#include <limits>
#include <unordered_set>
#include <utility>

#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_service_internal.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_status.h"

namespace vector_index {

bool vector_entry_store::register_index(const std::string &index_name) {
  if (index_name.empty() ||
      m_entry_counts.find(index_name) != m_entry_counts.end()) {
    return false;
  }
  m_entries.emplace(index_name, committed_entries());
  m_entry_counts.emplace(index_name, 0);
  m_index_generations.emplace(index_name, 0);
  return true;
}

bool vector_entry_store::drop_index(const std::string &index_name) {
  const bool existed = m_entry_counts.erase(index_name) > 0;
  m_entries.erase(index_name);
  m_index_generations.erase(index_name);
  m_evicted_indexes.erase(index_name);
  return existed;
}

bool vector_entry_store::rename_index(const std::string &old_index_name,
                                      const std::string &new_index_name) {
  if (old_index_name.empty() || new_index_name.empty() ||
      m_entry_counts.find(new_index_name) != m_entry_counts.end()) {
    return false;
  }
  auto count_it = m_entry_counts.find(old_index_name);
  if (count_it == m_entry_counts.end()) return false;

  auto entries_it = m_entries.find(old_index_name);
  committed_entries entries;
  if (entries_it != m_entries.end()) {
    entries = std::move(entries_it->second);
    m_entries.erase(entries_it);
  }
  m_entries.emplace(new_index_name, std::move(entries));
  const size_t entry_count = count_it->second;
  m_entry_counts.erase(count_it);
  m_entry_counts.emplace(new_index_name, entry_count);
  auto generation_it = m_index_generations.find(old_index_name);
  const uint64_t generation =
      generation_it == m_index_generations.end() ? 0 : generation_it->second;
  if (generation_it != m_index_generations.end())
    m_index_generations.erase(generation_it);
  m_index_generations.emplace(new_index_name, generation);
  if (m_evicted_indexes.erase(old_index_name) > 0)
    m_evicted_indexes.insert(new_index_name);
  return true;
}

bool vector_entry_store::has_index(const std::string &index_name) const {
  return m_entry_counts.find(index_name) != m_entry_counts.end();
}

bool vector_entry_store::clear_index(const std::string &index_name) {
  auto count_it = m_entry_counts.find(index_name);
  if (count_it == m_entry_counts.end()) return false;
  auto entries_it = m_entries.find(index_name);
  if (entries_it == m_entries.end())
    entries_it = m_entries.emplace(index_name, committed_entries()).first;
  const bool changed = count_it->second != 0 || !entries_it->second.empty();
  entries_it->second.clear();
  count_it->second = 0;
  m_evicted_indexes.erase(index_name);
  if (changed) bump_generation(index_name);
  return true;
}

bool vector_entry_store::replace_index(const std::string &index_name,
                                       const committed_entries &entries) {
  auto count_it = m_entry_counts.find(index_name);
  if (count_it == m_entry_counts.end()) return false;
  m_entries[index_name] = entries;
  count_it->second = entries.size();
  m_evicted_indexes.erase(index_name);
  bump_generation(index_name);
  return true;
}

bool vector_entry_store::upsert(const std::string &index_name, uint64_t doc_id,
                                const vector_data &vector) {
  if (!ensure_index_cached(index_name)) return false;
  auto entries_it = m_entries.find(index_name);
  auto count_it = m_entry_counts.find(index_name);
  if (entries_it == m_entries.end() || count_it == m_entry_counts.end())
    return false;
  const auto doc_it = entries_it->second.find(doc_id);
  const bool inserted = doc_it == entries_it->second.end();
  const bool changed = inserted || doc_it->second != vector;
  if (inserted)
    ++count_it->second;
  if (changed) {
    entries_it->second[doc_id] = vector;
    bump_generation(index_name);
  }
  return true;
}

bool vector_entry_store::erase(const std::string &index_name, uint64_t doc_id) {
  if (!ensure_index_cached(index_name)) return false;
  auto entries_it = m_entries.find(index_name);
  auto count_it = m_entry_counts.find(index_name);
  if (entries_it == m_entries.end() || count_it == m_entry_counts.end())
    return false;
  if (entries_it->second.erase(doc_id) > 0) {
    if (count_it->second > 0) --count_it->second;
    bump_generation(index_name);
  }
  return true;
}

bool vector_entry_store::snapshot(committed_state *state) const {
  if (state == nullptr) return false;
  state->clear();
  for (const auto &index_entry : m_entry_counts) {
    committed_entries entries;
    if (!snapshot_index(index_entry.first, &entries)) return false;
    state->emplace(index_entry.first, std::move(entries));
  }
  return true;
}

bool vector_entry_store::snapshot_index(const std::string &index_name,
                                        committed_entries *entries) const {
  if (entries == nullptr) return false;
  if (m_evicted_indexes.find(index_name) != m_evicted_indexes.end()) {
    return load_index_from_truth_store(index_name, entries);
  }
  auto entries_it = m_entries.find(index_name);
  if (entries_it == m_entries.end()) return false;
  *entries = entries_it->second;
  return true;
}

bool vector_entry_store::find_committed_entry(const std::string &index_name,
                                              uint64_t doc_id,
                                              vector_data *vector,
                                              bool *found) const {
  if (vector == nullptr || found == nullptr) return false;
  vector->clear();
  *found = false;

  if (m_entry_counts.find(index_name) == m_entry_counts.end()) return false;

  if (m_evicted_indexes.find(index_name) != m_evicted_indexes.end()) {
    vector_index_truth_store::truth_store *truth_store =
        vector_index_truth_store::get();
    if (truth_store == nullptr) return false;
    vector_index_metadata_store::committed_row row;
    if (!truth_store->find_committed(index_name, doc_id, &row, found)) {
      return false;
    }
    if (*found) *vector = std::move(row.vector);
    return true;
  }

  auto entries_it = m_entries.find(index_name);
  if (entries_it == m_entries.end()) return false;
  const auto doc_it = entries_it->second.find(doc_id);
  if (doc_it == entries_it->second.end()) return true;

  *vector = doc_it->second;
  *found = true;
  return true;
}

bool vector_entry_store::for_each_committed_entry(
    const std::string &index_name, const entry_visitor &visitor) const {
  if (!visitor) return false;
  if (m_evicted_indexes.find(index_name) != m_evicted_indexes.end()) {
    vector_index_truth_store::truth_store *truth_store =
        vector_index_truth_store::get();
    if (truth_store == nullptr) return false;
    return truth_store->for_each_committed(
        index_name,
        [&](const vector_index_metadata_store::committed_row &row) {
          return visitor(row.doc_id, row.vector);
        });
  }
  auto entries_it = m_entries.find(index_name);
  if (entries_it == m_entries.end()) return false;

  std::vector<uint64_t> doc_ids;
  doc_ids.reserve(entries_it->second.size());
  for (const auto &entry : entries_it->second) {
    doc_ids.push_back(entry.first);
  }
  std::sort(doc_ids.begin(), doc_ids.end());

  for (const uint64_t doc_id : doc_ids) {
    const auto doc_it = entries_it->second.find(doc_id);
    if (doc_it == entries_it->second.end()) return false;
    if (!visitor(doc_id, doc_it->second)) return false;
  }
  return true;
}

bool vector_entry_store::evict_until_under_budget(size_t budget) {
  if (budget == std::numeric_limits<size_t>::max()) return true;

  while (memory_bytes() > budget) {
    auto victim_it = m_entries.end();
    size_t victim_bytes = 0;
    for (auto entries_it = m_entries.begin(); entries_it != m_entries.end();
         ++entries_it) {
      if (m_evicted_indexes.find(entries_it->first) != m_evicted_indexes.end())
        continue;
      const size_t bytes = memory_bytes(entries_it->first);
      if (bytes > victim_bytes) {
        victim_it = entries_it;
        victim_bytes = bytes;
      }
    }
    if (victim_it == m_entries.end() || victim_bytes == 0) return true;
    victim_it->second.clear();
    m_evicted_indexes.insert(victim_it->first);
  }
  return true;
}

size_t vector_entry_store::entry_count() const {
  size_t count = 0;
  for (const auto &index_entry : m_entry_counts) {
    count += index_entry.second;
  }
  return count;
}

size_t vector_entry_store::entry_count(const std::string &index_name) const {
  auto count_it = m_entry_counts.find(index_name);
  if (count_it == m_entry_counts.end()) return 0;
  return count_it->second;
}

size_t vector_entry_store::memory_bytes() const {
  size_t bytes = 0;
  for (const auto &index_entry : m_entries) {
    bytes += memory_bytes(index_entry.first);
  }
  return bytes;
}

uint64_t vector_entry_store::generation(const std::string &index_name) const {
  auto generation_it = m_index_generations.find(index_name);
  if (generation_it == m_index_generations.end()) return 0;
  return generation_it->second;
}

bool vector_entry_store::load_index_from_truth_store(
    const std::string &index_name, committed_entries *entries) const {
  if (entries == nullptr) return false;
  entries->clear();
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (truth_store == nullptr) return false;
  return truth_store->for_each_committed(
      index_name,
      [&](const vector_index_metadata_store::committed_row &row) {
        (*entries)[row.doc_id] = row.vector;
        return true;
      });
}

void vector_entry_store::bump_generation(const std::string &index_name) {
  ++m_index_generations[index_name];
}

bool vector_entry_store::ensure_index_cached(const std::string &index_name) {
  if (m_entry_counts.find(index_name) == m_entry_counts.end()) return false;
  if (m_evicted_indexes.find(index_name) == m_evicted_indexes.end()) {
    if (m_entries.find(index_name) == m_entries.end())
      m_entries.emplace(index_name, committed_entries());
    return true;
  }

  committed_entries entries;
  if (!load_index_from_truth_store(index_name, &entries)) return false;
  m_entries[index_name] = std::move(entries);
  m_entry_counts[index_name] = m_entries[index_name].size();
  m_evicted_indexes.erase(index_name);
  return true;
}

size_t vector_entry_store::memory_bytes(const std::string &index_name) const {
  auto entries_it = m_entries.find(index_name);
  if (entries_it == m_entries.end()) return 0;

  size_t bytes = 0;
  for (const auto &doc_entry : entries_it->second) {
    bytes += doc_entry.second.size() * sizeof(float);
  }
  return bytes;
}

namespace detail {

std::unique_ptr<backend> build_backend_from_config(
    const std::string &index_name,
    const vector_index::index_service::index_config &config) {
  std::unique_ptr<backend> backend =
      create_backend(config.dimension, config.metric, config.mode,
                     config.provider, index_name);
  if (backend == nullptr) return nullptr;
  if (config.hnsw_m != 0 && config.hnsw_ef_construction != 0 &&
      !backend->set_hnsw_build_params(config.hnsw_m,
                                      config.hnsw_ef_construction)) {
    return nullptr;
  }
  if (config.hnsw_build_threads != 0 &&
      !backend->set_hnsw_build_threads(config.hnsw_build_threads)) {
    return nullptr;
  }
  if (config.faiss_build_threads != 0 &&
      !backend->set_faiss_build_threads(config.faiss_build_threads)) {
    return nullptr;
  }
  const bool faiss_ivf_params_set =
      backend->set_faiss_ivf_params(config.faiss_nlist, config.faiss_nprobe);
  if (!faiss_ivf_params_set &&
      (config.faiss_nlist != 0 || config.faiss_nprobe != 0)) {
    return nullptr;
  }
  if (config.faiss_pq_m != 0 || config.faiss_pq_bits != 0) {
    if (!backend->set_faiss_ivf_pq_params(
            config.faiss_nlist, config.faiss_nprobe, config.faiss_pq_m,
            config.faiss_pq_bits)) {
      return nullptr;
    }
  }
  if (config.diskann_max_degree != 0 && config.diskann_build_complexity != 0 &&
      !backend->set_diskann_build_params(config.diskann_max_degree,
                                         config.diskann_build_complexity,
                                         config.diskann_build_threads)) {
    return nullptr;
  }
  if (config.diskann_build_threads != 0 &&
      !backend->set_diskann_build_threads(config.diskann_build_threads)) {
    return nullptr;
  }
  if (config.diskann_search_complexity != 0 &&
      !backend->set_diskann_search_complexity(
          config.diskann_search_complexity)) {
    return nullptr;
  }
  if (config.search_ef != 0 && !backend->set_search_ef(config.search_ef))
    return nullptr;
  return backend;
}

}  // namespace detail

namespace {

using detail::build_backend_from_config;

constexpr const char *LIFECYCLE_READY = "ready";
constexpr const char *LIFECYCLE_BULK_LOADING = "bulk_loading";
constexpr const char *LIFECYCLE_REBUILDING = "rebuilding";
constexpr const char *LIFECYCLE_RECOVERING = "recovering";
constexpr const char *LIFECYCLE_FAILED = "failed";
constexpr uint32_t ERROR_NONE = 0;
constexpr uint32_t ERROR_PENDING_CHANGES = 1001;
constexpr uint32_t ERROR_BACKEND_CREATE_FAILED = 1002;
constexpr uint32_t ERROR_BACKEND_RECOVER_FAILED = 1003;
constexpr uint32_t ERROR_REPLAY_STATE_FAILED = 1004;

uint64_t now_unix_epoch_seconds() {
  return static_cast<uint64_t>(std::time(nullptr));
}

void mark_lifecycle_state(
    vector_index::index_service::lifecycle_info *lifecycle, const char *state) {
  lifecycle->state = state;
  ++lifecycle->version;
}

void mark_lifecycle_failure(
    vector_index::index_service::lifecycle_info *lifecycle,
    uint32_t error_code) {
  mark_lifecycle_state(lifecycle, LIFECYCLE_FAILED);
  lifecycle->last_error_code = error_code;
  lifecycle->last_error_ts = now_unix_epoch_seconds();
}

void mark_lifecycle_ready(
    vector_index::index_service::lifecycle_info *lifecycle) {
  mark_lifecycle_state(lifecycle, LIFECYCLE_READY);
  lifecycle->last_error_code = ERROR_NONE;
  lifecycle->last_error_ts = 0;
}

void sync_search_config_from_backend(
    const backend &index_backend,
    vector_index::index_service::index_config *config) {
  config->backend_variant = index_backend.backend_variant();
  config->search_ef = index_backend.search_ef();
}

void sync_hnsw_config_from_backend(
    const backend &index_backend,
    vector_index::index_service::index_config *config) {
  config->hnsw_m = index_backend.hnsw_m();
  config->hnsw_ef_construction = index_backend.hnsw_ef_construction();
  config->hnsw_build_threads = index_backend.hnsw_build_threads();
}

void sync_faiss_config_from_backend(
    const backend &index_backend,
    vector_index::index_service::index_config *config) {
  config->faiss_nlist = index_backend.faiss_nlist();
  config->faiss_nprobe = index_backend.faiss_nprobe();
  config->faiss_pq_m = index_backend.faiss_pq_m();
  config->faiss_pq_bits = index_backend.faiss_pq_bits();
  config->faiss_build_threads = index_backend.faiss_build_threads();
}

void sync_diskann_build_config_from_backend(
    const backend &index_backend,
    vector_index::index_service::index_config *config) {
  config->diskann_max_degree = index_backend.diskann_max_degree();
  config->diskann_build_complexity = index_backend.diskann_build_complexity();
  config->diskann_build_threads = index_backend.diskann_build_threads();
}

template <typename pending_changes_map>
bool pending_changes_contain_index(const pending_changes_map &pending_changes,
                                   const std::string &index_name) {
  for (const auto &txn_changes : pending_changes) {
    for (const auto &change : txn_changes.second) {
      if (change.index_name == index_name) return true;
    }
  }
  return false;
}

template <typename pending_changes_map>
bool has_any_pending_changes(const pending_changes_map &pending_changes) {
  for (const auto &txn_changes : pending_changes) {
    if (!txn_changes.second.empty()) return true;
  }
  return false;
}

template <typename... Bools>
bool all_true(Bools... values) {
  bool result = true;
  ((result &= static_cast<bool>(values)), ...);
  return result;
}

bool lazy_external_runtime_enabled() {
  return false;
}

committed_entry_reader make_committed_entry_reader(
    const vector_entry_store &entry_store, const std::string &index_name) {
  return [&entry_store, index_name](const committed_entry_visitor &visitor) {
    return entry_store.for_each_committed_entry(index_name, visitor);
  };
}

bool load_backend_from_store(const vector_entry_store &entry_store,
                             const std::string &index_name, backend *target) {
  return target != nullptr &&
         target->load_committed_entries_from_reader(
             make_committed_entry_reader(entry_store, index_name));
}

bool rebuild_backend_from_store(const vector_entry_store &entry_store,
                                const std::string &index_name,
                                backend *target) {
  return target != nullptr &&
         target->rebuild_from_committed_entries_from_reader(
             make_committed_entry_reader(entry_store, index_name));
}

bool recover_backend_from_store(const vector_entry_store &entry_store,
                                const std::string &index_name,
                                backend *target) {
  return target != nullptr &&
         target->recover_committed_entries_from_reader(
             make_committed_entry_reader(entry_store, index_name));
}

bool entry_store_vectors_match_dimension(const vector_entry_store &entry_store,
                                         const std::string &index_name,
                                         size_t dimension) {
  return entry_store.for_each_committed_entry(
      index_name, [dimension](uint64_t doc_id [[maybe_unused]],
                              const vector_data &vector) {
        return vector.size() == dimension;
      });
}
}  // namespace

bool index_service::ensure_runtime_loaded(const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name))) {
    return false;
  }
  if (!lazy_external_runtime_enabled() ||
      config_it->second.mode != backend_mode::kExternal) {
    return true;
  }
  const size_t committed_count = m_entry_store.entry_count(index_name);
  if (committed_count == 0 ||
      index_it->second->entry_count() == committed_count)
    return true;

  std::unique_ptr<backend> loaded =
      build_backend_from_config(index_name, config_it->second);
  if (loaded == nullptr) return false;
  if (!load_backend_from_store(m_entry_store, index_name, loaded.get()))
    return false;
  index_it->second = std::move(loaded);
  return true;
}

bool index_service::ensure_runtime_loaded_for_search(
    const std::string &index_name) {
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (lifecycle_it != m_lifecycle_infos.end() &&
      lifecycle_it->second.state == LIFECYCLE_BULK_LOADING) {
    return false;
  }
  return ensure_runtime_loaded(index_name);
}

bool index_service::rebuild_runtime_from_store_for_search(
    const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                lifecycle_it != m_lifecycle_infos.end(),
                m_entry_store.has_index(index_name))) {
    return false;
  }
  if (lifecycle_it->second.state == LIFECYCLE_BULK_LOADING ||
      pending_changes_contain_index(m_pending_changes, index_name)) {
    return false;
  }

  std::unique_ptr<backend> rebuilt =
      build_backend_from_config(index_name, config_it->second);
  if (rebuilt == nullptr ||
      !rebuild_backend_from_store(m_entry_store, index_name, rebuilt.get())) {
    return false;
  }

  index_it->second = std::move(rebuilt);
  ++lifecycle_it->second.recover_fallback_count;
  lifecycle_it->second.last_recover_fallback_ts = now_unix_epoch_seconds();
  vector_status::record_backend_recover_fallback();
  return true;
}

void index_service::maybe_unload_runtime(const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return;
  }
  if (!lazy_external_runtime_enabled() ||
      config_it->second.mode != backend_mode::kExternal) {
    return;
  }
  std::unique_ptr<backend> unloaded =
      build_backend_from_config(index_name, config_it->second);
  if (unloaded == nullptr) return;
  index_it->second = std::move(unloaded);
}

bool index_service::register_index(const std::string &index_name,
                                   std::unique_ptr<backend> backend) {
  if (backend == nullptr || index_name.empty()) return false;

  index_config config{backend->dimension(),
                      backend->metric(),
                      backend->mode(),
                      backend->provider(),
                      backend->backend_variant(),
                      backend->search_ef(),
                      backend->hnsw_m(),
                      backend->hnsw_ef_construction(),
                      backend->hnsw_build_threads(),
                      backend->faiss_nlist(),
                      backend->faiss_nprobe(),
                      backend->faiss_pq_m(),
                      backend->faiss_pq_bits(),
                      backend->faiss_build_threads(),
                      backend->diskann_max_degree(),
                      backend->diskann_build_complexity(),
                      backend->diskann_build_threads(),
                      backend->diskann_search_complexity()};
  return register_index_impl(index_name, std::move(config), std::move(backend));
}

bool index_service::register_index_impl(const std::string &index_name,
                                        index_config config,
                                        std::unique_ptr<backend> backend) {
  auto [it, inserted] = m_indexes.emplace(index_name, std::move(backend));
  if (!inserted || it->second == nullptr) return false;
  if (!m_entry_store.register_index(index_name)) {
    m_indexes.erase(it);
    return false;
  }

  config.backend_variant = it->second->backend_variant();
  m_index_configs[index_name] = std::move(config);
  m_lifecycle_infos[index_name] =
      lifecycle_info{LIFECYCLE_READY, 1, ERROR_NONE, 0, 0, 0, 0};
  return true;
}

bool index_service::register_index(const std::string &index_name,
                                   const index_config &config) {
  if (config.dimension == 0) return false;

  std::unique_ptr<backend> backend =
      build_backend_from_config(index_name, config);
  if (backend == nullptr) return false;

  return register_index_impl(index_name, config, std::move(backend));
}

bool index_service::register_index_from_strings(const std::string &index_name,
                                                size_t dimension,
                                                const std::string &metric,
                                                const std::string &mode,
                                                const std::string &provider) {
  metric_type metric_value = metric_type::kEuclidean;
  backend_mode mode_value = backend_mode::kMemory;
  backend_provider provider_value = backend_provider::kNative;

  if (!parse_metric(metric, &metric_value)) return false;
  if (provider.empty()) {
    if (!default_backend_provider(&provider_value)) return false;
  } else if (!parse_backend_provider(provider, &provider_value)) {
    return false;
  }
  if (mode.empty()) {
    if (!default_backend_mode_for_provider(provider_value, &mode_value))
      return false;
  } else if (!parse_backend_mode(mode, &mode_value)) {
    return false;
  }

  index_config config;
  config.dimension = dimension;
  config.metric = metric_value;
  config.mode = mode_value;
  config.provider = provider_value;
  return register_index(index_name, config);
}

bool index_service::drop_index(const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  if (config_it == m_index_configs.end()) return false;
  if (!remove_backend_artifacts(index_name, config_it->second.mode,
                                config_it->second.provider)) {
    return false;
  }
  return unregister_index(index_name);
}

bool index_service::unregister_index(const std::string &index_name) {
  auto index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end()) return false;

  m_indexes.erase(index_it);
  m_index_configs.erase(index_name);
  m_entry_store.drop_index(index_name);
  m_lifecycle_infos.erase(index_name);
  return true;
}

bool index_service::rename_index(const std::string &old_index_name,
                                 const std::string &new_index_name) {
  if (old_index_name.empty() || new_index_name.empty()) return false;
  if (old_index_name == new_index_name) return true;

  auto old_index_it = m_indexes.find(old_index_name);
  auto old_config_it = m_index_configs.find(old_index_name);
  auto old_lifecycle_it = m_lifecycle_infos.find(old_index_name);
  if (old_index_it == m_indexes.end() ||
      old_config_it == m_index_configs.end() ||
      !m_entry_store.has_index(old_index_name) ||
      old_lifecycle_it == m_lifecycle_infos.end()) {
    return false;
  }
  if (m_indexes.find(new_index_name) != m_indexes.end()) return false;

  const index_config config = old_config_it->second;
  const lifecycle_info lifecycle = old_lifecycle_it->second;

  std::unique_ptr<backend> renamed_backend =
      build_backend_from_config(new_index_name, config);
  if (renamed_backend == nullptr) return false;
  if (!load_backend_from_store(m_entry_store, old_index_name,
                               renamed_backend.get()))
    return false;
  if (!m_entry_store.rename_index(old_index_name, new_index_name)) return false;

  m_indexes.erase(old_index_it);
  m_index_configs.erase(old_config_it);
  m_lifecycle_infos.erase(old_lifecycle_it);

  m_indexes.emplace(new_index_name, std::move(renamed_backend));
  m_index_configs.emplace(new_index_name, config);
  m_lifecycle_infos.emplace(new_index_name, lifecycle);
  maybe_unload_runtime(new_index_name);

  for (auto &txn_changes : m_pending_changes) {
    for (pending_change &change : txn_changes.second) {
      if (change.index_name == old_index_name)
        change.index_name = new_index_name;
    }
  }
  return true;
}

bool index_service::begin_bulk_load(const std::string &index_name) {
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (index_it == m_indexes.end() || lifecycle_it == m_lifecycle_infos.end())
    return false;

  if (pending_changes_contain_index(m_pending_changes, index_name)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_PENDING_CHANGES);
    return false;
  }

  mark_lifecycle_state(&lifecycle_it->second, LIFECYCLE_BULK_LOADING);
  lifecycle_it->second.last_error_code = ERROR_NONE;
  lifecycle_it->second.last_error_ts = 0;
  return true;
}

bool index_service::rebuild_index(const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end())) {
    return false;
  }
  mark_lifecycle_state(&lifecycle_it->second, LIFECYCLE_REBUILDING);

  // Rebuild currently requires no staged writes against this index.
  if (pending_changes_contain_index(m_pending_changes, index_name)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_PENDING_CHANGES);
    return false;
  }

  std::unique_ptr<backend> rebuilt =
      build_backend_from_config(index_name, config_it->second);
  if (rebuilt == nullptr) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_BACKEND_CREATE_FAILED);
    return false;
  }
  if (!rebuild_backend_from_store(m_entry_store, index_name, rebuilt.get())) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    return false;
  }

  index_it->second = std::move(rebuilt);
  mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

bool index_service::recover_index(const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end())) {
    return false;
  }
  mark_lifecycle_state(&lifecycle_it->second, LIFECYCLE_RECOVERING);

  // Recovery for one index requires no staged writes against this index.
  if (pending_changes_contain_index(m_pending_changes, index_name)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_PENDING_CHANGES);
    return false;
  }

  std::unique_ptr<backend> recovered =
      build_backend_from_config(index_name, config_it->second);
  if (recovered == nullptr ||
      !recover_backend_from_store(m_entry_store, index_name, recovered.get())) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_BACKEND_RECOVER_FAILED);
    return false;
  }
  const bool used_recover_fallback = recovered->last_recover_used_fallback();
  if (used_recover_fallback) {
    ++lifecycle_it->second.recover_fallback_count;
    lifecycle_it->second.last_recover_fallback_ts = now_unix_epoch_seconds();
  }

  index_it->second = std::move(recovered);
  mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

bool index_service::set_search_ef(const std::string &index_name,
                                  uint32_t search_ef) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (!index_it->second->set_search_ef(search_ef)) return false;
  sync_search_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_hnsw_build_params(const std::string &index_name,
                                          uint32_t hnsw_m,
                                          uint32_t hnsw_ef_construction) {
  if (hnsw_m == 0 || hnsw_ef_construction == 0) return false;

  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name))) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  index_config next_config = config_it->second;
  next_config.hnsw_m = hnsw_m;
  next_config.hnsw_ef_construction = hnsw_ef_construction;
  std::unique_ptr<backend> rebuilt =
      build_backend_from_config(index_name, next_config);
  if (rebuilt == nullptr) return false;
  if (!load_backend_from_store(m_entry_store, index_name, rebuilt.get()))
    return false;

  index_it->second = std::move(rebuilt);
  config_it->second = next_config;
  sync_search_config_from_backend(*index_it->second, &config_it->second);
  sync_hnsw_config_from_backend(*index_it->second, &config_it->second);
  sync_faiss_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_hnsw_build_threads(const std::string &index_name,
                                           uint32_t hnsw_build_threads) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;
  if (!index_it->second->set_hnsw_build_threads(hnsw_build_threads))
    return false;
  sync_hnsw_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_faiss_ivf_params(const std::string &index_name,
                                         uint32_t faiss_nlist,
                                         uint32_t faiss_nprobe) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  if (!index_it->second->set_faiss_ivf_params(faiss_nlist, faiss_nprobe))
    return false;
  sync_search_config_from_backend(*index_it->second, &config_it->second);
  sync_hnsw_config_from_backend(*index_it->second, &config_it->second);
  sync_faiss_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_faiss_ivf_pq_params(const std::string &index_name,
                                            uint32_t faiss_nlist,
                                            uint32_t faiss_nprobe,
                                            uint32_t faiss_pq_m,
                                            uint32_t faiss_pq_bits) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  if (!index_it->second->set_faiss_ivf_pq_params(faiss_nlist, faiss_nprobe,
                                                 faiss_pq_m, faiss_pq_bits)) {
    return false;
  }
  sync_search_config_from_backend(*index_it->second, &config_it->second);
  sync_hnsw_config_from_backend(*index_it->second, &config_it->second);
  sync_faiss_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_faiss_build_threads(
    const std::string &index_name, uint32_t faiss_build_threads) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  if (!index_it->second->set_faiss_build_threads(faiss_build_threads)) {
    return false;
  }
  sync_faiss_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_diskann_build_params(
    const std::string &index_name, uint32_t diskann_max_degree,
    uint32_t diskann_build_complexity, uint32_t diskann_build_threads) {
  if (diskann_max_degree == 0 || diskann_build_complexity == 0) return false;

  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  if (!index_it->second->set_diskann_build_params(
          diskann_max_degree, diskann_build_complexity,
          diskann_build_threads)) {
    return false;
  }
  sync_diskann_build_config_from_backend(*index_it->second,
                                         &config_it->second);
  sync_search_config_from_backend(*index_it->second, &config_it->second);
  sync_hnsw_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_diskann_build_threads(
    const std::string &index_name, uint32_t diskann_build_threads) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  if (!index_it->second->set_diskann_build_threads(diskann_build_threads)) {
    return false;
  }
  sync_diskann_build_config_from_backend(*index_it->second,
                                         &config_it->second);
  return true;
}

bool index_service::set_diskann_search_complexity(
    const std::string &index_name, uint32_t diskann_search_complexity) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (!index_it->second->set_diskann_search_complexity(
          diskann_search_complexity))
    return false;
  config_it->second.diskann_search_complexity =
      index_it->second->diskann_search_complexity();
  return true;
}

bool index_service::rebuild_all_indexes(size_t *rebuilt_count) {
  if (rebuilt_count == nullptr) return false;

  if (has_any_pending_changes(m_pending_changes)) return false;

  std::unordered_map<std::string, std::unique_ptr<backend>> rebuilt_backends;
  rebuilt_backends.reserve(m_index_configs.size());
  for (const auto &entry : m_index_configs) {
    const std::string &index_name = entry.first;
    const index_config &config = entry.second;
    if (m_indexes.find(index_name) == m_indexes.end()) return false;
    if (!m_entry_store.has_index(index_name)) return false;
    if (m_lifecycle_infos.find(index_name) == m_lifecycle_infos.end())
      return false;

    std::unique_ptr<backend> rebuilt =
        build_backend_from_config(index_name, config);
    if (rebuilt == nullptr) return false;
    if (!rebuild_backend_from_store(m_entry_store, index_name, rebuilt.get()))
      return false;
    rebuilt_backends.emplace(index_name, std::move(rebuilt));
  }

  for (auto &entry : rebuilt_backends) {
    auto lifecycle_it = m_lifecycle_infos.find(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_lifecycle_state(&lifecycle_it->second, LIFECYCLE_REBUILDING);
    }
    m_indexes[entry.first] = std::move(entry.second);
    maybe_unload_runtime(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_lifecycle_ready(&lifecycle_it->second);
    }
  }
  *rebuilt_count = rebuilt_backends.size();
  return true;
}

bool index_service::recover_all_indexes(size_t *recovered_count) {
  if (recovered_count == nullptr) return false;

  // Recovery cannot proceed with in-flight staged changes.
  if (has_any_pending_changes(m_pending_changes)) return false;

  std::unordered_map<std::string, std::unique_ptr<backend>> recovered_backends;
  std::unordered_set<std::string> recover_fallback_indexes;
  recovered_backends.reserve(m_index_configs.size());
  for (const auto &entry : m_index_configs) {
    const std::string &index_name = entry.first;
    const index_config &config = entry.second;
    if (!m_entry_store.has_index(index_name)) return false;

    std::unique_ptr<backend> recovered =
        build_backend_from_config(index_name, config);
    if (recovered == nullptr ||
        !recover_backend_from_store(m_entry_store, index_name,
                                    recovered.get())) {
      return false;
    }
    if (recovered->last_recover_used_fallback()) {
      recover_fallback_indexes.insert(index_name);
    }
    if (m_lifecycle_infos.find(index_name) == m_lifecycle_infos.end())
      return false;

    recovered_backends.emplace(index_name, std::move(recovered));
  }

  for (auto &entry : recovered_backends) {
    auto lifecycle_it = m_lifecycle_infos.find(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_lifecycle_state(&lifecycle_it->second, LIFECYCLE_RECOVERING);
    }
    m_indexes[entry.first] = std::move(entry.second);
    maybe_unload_runtime(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end() &&
        recover_fallback_indexes.find(entry.first) !=
            recover_fallback_indexes.end()) {
      ++lifecycle_it->second.recover_fallback_count;
      lifecycle_it->second.last_recover_fallback_ts = now_unix_epoch_seconds();
    }
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_lifecycle_ready(&lifecycle_it->second);
    }
  }

  *recovered_count = recovered_backends.size();
  return true;
}

bool index_service::describe_index(
    const std::string &index_name, index_config *config,
    bool *supports_mutations, size_t *entry_count,
    size_t *committed_entry_count, std::string *lifecycle_state,
    uint64_t *lifecycle_version, uint32_t *last_error_code,
    uint64_t *last_error_ts, uint64_t *last_apply_latency_ms,
    uint64_t *recover_fallback_count, uint64_t *last_recover_fallback_ts,
    bool *external_manifest_present,
    uint64_t *external_manifest_generation) const {
  if (config == nullptr) {
    return false;
  }

  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  auto config_it = m_index_configs.find(index_name);
  const bool has_committed_entries = m_entry_store.has_index(index_name);
  if (config_it == m_index_configs.end() || index_it == m_indexes.end() ||
      !has_committed_entries) {
    return false;
  }
  if (lifecycle_it == m_lifecycle_infos.end()) return false;

  const backend *backend = index_it->second.get();
  config->dimension = backend->dimension();
  config->metric = backend->metric();
  config->mode = backend->mode();
  config->provider = backend->provider();
  config->backend_variant = backend->backend_variant();
  config->search_ef = backend->search_ef();
  config->hnsw_m = backend->hnsw_m();
  config->hnsw_ef_construction = backend->hnsw_ef_construction();
  config->hnsw_build_threads = backend->hnsw_build_threads();
  config->faiss_nlist = backend->faiss_nlist();
  config->faiss_nprobe = backend->faiss_nprobe();
  config->faiss_pq_m = backend->faiss_pq_m();
  config->faiss_pq_bits = backend->faiss_pq_bits();
  config->faiss_build_threads = backend->faiss_build_threads();
  config->diskann_max_degree = backend->diskann_max_degree();
  config->diskann_build_complexity = backend->diskann_build_complexity();
  config->diskann_build_threads = backend->diskann_build_threads();
  config->diskann_search_complexity = backend->diskann_search_complexity();
  if (supports_mutations != nullptr) {
    *supports_mutations = backend->supports_mutations();
  }
  if (entry_count != nullptr) {
    *entry_count = backend->entry_count();
  }
  if (committed_entry_count != nullptr) {
    *committed_entry_count = m_entry_store.entry_count(index_name);
  }
  if (lifecycle_state != nullptr) *lifecycle_state = lifecycle_it->second.state;
  if (lifecycle_version != nullptr)
    *lifecycle_version = lifecycle_it->second.version;
  if (last_error_code != nullptr)
    *last_error_code = lifecycle_it->second.last_error_code;
  if (last_error_ts != nullptr)
    *last_error_ts = lifecycle_it->second.last_error_ts;
  if (last_apply_latency_ms != nullptr) {
    *last_apply_latency_ms = lifecycle_it->second.last_apply_latency_ms;
  }
  if (recover_fallback_count != nullptr) {
    *recover_fallback_count = lifecycle_it->second.recover_fallback_count;
  }
  if (last_recover_fallback_ts != nullptr) {
    *last_recover_fallback_ts = lifecycle_it->second.last_recover_fallback_ts;
  }
  if (external_manifest_present != nullptr) {
    *external_manifest_present = backend->external_manifest_present();
  }
  if (external_manifest_generation != nullptr) {
    *external_manifest_generation = backend->external_manifest_generation();
  }
  return true;
}

bool index_service::set_last_apply_latency_ms(const std::string &index_name,
                                              uint64_t latency_ms) {
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (lifecycle_it == m_lifecycle_infos.end()) return false;
  lifecycle_it->second.last_apply_latency_ms = latency_ms;
  return true;
}

bool index_service::set_lifecycle_state(
    const std::string &index_name, const std::string &lifecycle_state) {
  if (lifecycle_state.empty()) return false;

  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (lifecycle_it == m_lifecycle_infos.end()) return false;

  mark_lifecycle_state(&lifecycle_it->second, lifecycle_state.c_str());
  if (lifecycle_state != LIFECYCLE_FAILED) {
    lifecycle_it->second.last_error_code = ERROR_NONE;
    lifecycle_it->second.last_error_ts = 0;
  }
  return true;
}

bool index_service::set_lifecycle_info(const std::string &index_name,
                                       const std::string &lifecycle_state,
                                       uint64_t lifecycle_version,
                                       uint32_t last_error_code,
                                       uint64_t last_error_ts,
                                       uint64_t recover_fallback_count,
                                       uint64_t last_recover_fallback_ts) {
  if (lifecycle_state.empty() || lifecycle_version == 0) return false;

  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (lifecycle_it == m_lifecycle_infos.end()) return false;

  lifecycle_it->second.state = lifecycle_state;
  lifecycle_it->second.version = lifecycle_version;
  lifecycle_it->second.last_error_code = last_error_code;
  lifecycle_it->second.last_error_ts = last_error_ts;
  lifecycle_it->second.recover_fallback_count = recover_fallback_count;
  lifecycle_it->second.last_recover_fallback_ts = last_recover_fallback_ts;
  return true;
}

bool index_service::list_indexes(std::vector<std::string> *index_names) const {
  if (index_names == nullptr) return false;

  index_names->clear();
  index_names->reserve(m_indexes.size());
  for (const auto &entry : m_indexes) {
    index_names->push_back(entry.first);
  }
  std::sort(index_names->begin(), index_names->end());
  return true;
}

bool index_service::snapshot_committed_state(committed_state *state) const {
  return m_entry_store.snapshot(state);
}

size_t index_service::committed_entry_count() const {
  return m_entry_store.entry_count();
}

size_t index_service::committed_vector_memory_bytes() const {
  return m_entry_store.memory_bytes();
}

bool index_service::evict_committed_cache_to_budget() {
  return m_entry_store.evict_until_under_budget(
      static_cast<size_t>(opt_vector_entry_cache_size));
}

size_t index_service::pending_vector_memory_bytes(uint64_t txn_id) const {
  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it == m_pending_changes.end()) return 0;

  size_t bytes = 0;
  for (const pending_change &change : pending_it->second) {
    bytes += pending_change_memory_bytes(change);
  }
  return bytes;
}

size_t index_service::total_pending_vector_memory_bytes() const {
  size_t bytes = 0;
  for (const auto &txn_entry : m_pending_changes) {
    for (const pending_change &change : txn_entry.second) {
      bytes += pending_change_memory_bytes(change);
    }
  }
  return bytes;
}

bool index_service::restore_committed_state(const committed_state &state) {
  std::unordered_map<std::string, backend_ptr> restored_backends;
  restored_backends.reserve(m_index_configs.size());
  vector_entry_store restored_entry_store;

  for (const auto &config_entry : m_index_configs) {
    const std::string &index_name = config_entry.first;
    const index_config &config = config_entry.second;

    std::unique_ptr<backend> backend =
        build_backend_from_config(index_name, config);
    if (backend == nullptr) return false;

    auto state_it = state.find(index_name);
    if (state_it != state.end()) {
      for (const auto &doc_entry : state_it->second) {
        if (doc_entry.second.size() != backend->dimension()) return false;
      }
      if (!backend->load_committed_entries(state_it->second)) return false;
      if (!restored_entry_store.register_index(index_name) ||
          !restored_entry_store.replace_index(index_name, state_it->second)) {
        return false;
      }
    } else if (!restored_entry_store.register_index(index_name)) {
      return false;
    }

    restored_backends.emplace(index_name, std::move(backend));
  }

  m_indexes = std::move(restored_backends);
  m_entry_store = std::move(restored_entry_store);
  for (const auto &entry : m_index_configs) {
    maybe_unload_runtime(entry.first);
  }
  return true;
}

bool index_service::restore_index_config(const std::string &index_name,
                                         const index_config &config) {
  if (config.dimension == 0) return false;

  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end())) {
    return false;
  }

  if (!entry_store_vectors_match_dimension(m_entry_store, index_name,
                                           config.dimension))
    return false;

  std::unique_ptr<backend> restored =
      build_backend_from_config(index_name, config);
  if (restored == nullptr) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_BACKEND_CREATE_FAILED);
    return false;
  }
  if (!load_backend_from_store(m_entry_store, index_name, restored.get())) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    return false;
  }

  index_it->second = std::move(restored);
  config_it->second = config;
  config_it->second.backend_variant = index_it->second->backend_variant();
  config_it->second.search_ef = index_it->second->search_ef();
  config_it->second.hnsw_m = index_it->second->hnsw_m();
  config_it->second.hnsw_ef_construction =
      index_it->second->hnsw_ef_construction();
  config_it->second.hnsw_build_threads =
      index_it->second->hnsw_build_threads();
  config_it->second.faiss_nlist = index_it->second->faiss_nlist();
  config_it->second.faiss_nprobe = index_it->second->faiss_nprobe();
  config_it->second.faiss_pq_m = index_it->second->faiss_pq_m();
  config_it->second.faiss_pq_bits = index_it->second->faiss_pq_bits();
  config_it->second.diskann_max_degree = index_it->second->diskann_max_degree();
  config_it->second.diskann_build_complexity =
      index_it->second->diskann_build_complexity();
  config_it->second.diskann_build_threads =
      index_it->second->diskann_build_threads();
  config_it->second.diskann_search_complexity =
      index_it->second->diskann_search_complexity();
  mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

bool index_service::replace_committed_entries(
    const std::string &index_name, const committed_entries &entries) {
  return replace_committed_entries_impl(index_name, entries, false);
}

bool index_service::replace_committed_entries_preserve_lifecycle(
    const std::string &index_name, const committed_entries &entries) {
  return replace_committed_entries_impl(index_name, entries, true);
}

bool index_service::replace_committed_entries_impl(
    const std::string &index_name, const committed_entries &entries,
    bool preserve_lifecycle) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end())) {
    return false;
  }

  for (const auto &doc_entry : entries) {
    if (doc_entry.second.size() != config_it->second.dimension) return false;
  }

  std::unique_ptr<backend> rebuilt =
      build_backend_from_config(index_name, config_it->second);
  if (rebuilt == nullptr) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_BACKEND_CREATE_FAILED);
    return false;
  }
  if (!rebuilt->rebuild_from_committed_entries(entries)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    return false;
  }

  index_it->second = std::move(rebuilt);
  if (!m_entry_store.replace_index(index_name, entries)) return false;
  if (!preserve_lifecycle) mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

bool index_service::install_rebuilt_index(
    const std::string &index_name, const committed_entries &entries,
    std::unique_ptr<backend> rebuilt_backend) {
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end(),
                rebuilt_backend != nullptr)) {
    return false;
  }

  index_it->second = std::move(rebuilt_backend);
  if (!m_entry_store.replace_index(index_name, entries)) return false;
  mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

bool index_service::install_recovered_index(
    const std::string &index_name, const committed_entries &entries,
    std::unique_ptr<backend> recovered_backend, bool used_recover_fallback) {
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end(),
                recovered_backend != nullptr)) {
    return false;
  }

  index_it->second = std::move(recovered_backend);
  if (!m_entry_store.replace_index(index_name, entries)) return false;
  if (used_recover_fallback) {
    ++lifecycle_it->second.recover_fallback_count;
    lifecycle_it->second.last_recover_fallback_ts = now_unix_epoch_seconds();
  }
  mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

}  // namespace vector_index
