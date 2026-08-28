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

#include <fcntl.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
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

#include "my_dbug.h"
#include "my_sys.h"
#include "sql/mysqld.h"
#include "sql/vector/vector_index_backend_internal.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_service_internal.h"
#include "sql/vector/vector_mapped_search.h"
#include "sql/vector/vector_segmented_search.h"

#ifndef O_BINARY
#define O_BINARY 0
#endif

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

namespace vector_index {

namespace {

using detail::all_true;
using detail::build_backend_from_config;
using detail::index_configs_equal;

constexpr const char *LIFECYCLE_BULK_LOADING = "bulk_loading";
constexpr const char *k_pending_spill_directory = "pending_spill";
constexpr uint32_t ERROR_BACKEND_APPLY_FAILED = 1006;
constexpr uint32_t ERROR_MUTATION_NOT_SUPPORTED = 1005;

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
  mark_lifecycle_state(lifecycle, "failed");
  lifecycle->last_error_code = error_code;
  lifecycle->last_error_ts = now_unix_epoch_seconds();
}

std::filesystem::path pending_spill_directory_path() {
  const std::string root = detail::vector_index_root_path();
  if (root.empty()) return {};
  return std::filesystem::path(root) / k_pending_spill_directory;
}

bool should_batch_rebuild_on_commit(const backend *index_backend) {
  if (index_backend == nullptr) return false;
  if (index_backend->mode() != backend_mode::kExternal) return false;
  return index_backend->provider() == backend_provider::kFaiss;
}

bool should_defer_runtime_rebuild_on_commit(const backend *index_backend) {
  return index_backend != nullptr && index_backend->supports_mutations() &&
         !index_backend->supports_in_place_mutations();
}

bool is_external_diskann(const backend *index_backend) {
  return index_backend != nullptr &&
         index_backend->provider() == backend_provider::kDiskAnn &&
         index_backend->mode() == backend_mode::kExternal;
}

bool savepoint_names_equal(const std::string &lhs, const std::string &rhs) {
  return my_strnncoll(system_charset_info,
                      reinterpret_cast<const uchar *>(lhs.data()), lhs.size(),
                      reinterpret_cast<const uchar *>(rhs.data()),
                      rhs.size()) == 0;
}

bool pending_change_snapshots_equal(
    const std::vector<index_service::pending_change_snapshot> &lhs,
    const std::vector<index_service::pending_change_snapshot> &rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].index_name != rhs[i].index_name ||
        lhs[i].erase != rhs[i].erase || lhs[i].doc_id != rhs[i].doc_id ||
        lhs[i].vector != rhs[i].vector) {
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

bool rerank_candidates_with_vectors(
    const vector_data &query, metric_type metric,
    const std::vector<search_result> &candidates,
    const committed_entries &candidate_vectors, size_t top_k,
    std::vector<search_result> *results, bool *reranked) {
  if (results == nullptr || reranked == nullptr) return false;
  *reranked = false;

  std::vector<vector_mapped_search::visible_candidate> visible_rows;
  visible_rows.reserve(candidates.size());
  std::unordered_set<uint64_t> seen_doc_ids;
  for (const search_result &candidate : candidates) {
    if (!seen_doc_ids.insert(candidate.doc_id).second) continue;
    const auto vector_it = candidate_vectors.find(candidate.doc_id);
    if (vector_it == candidate_vectors.end()) return true;
    visible_rows.push_back(vector_mapped_search::visible_candidate{
        candidate.doc_id, vector_it->second});
  }

  if (!vector_mapped_search::rank_visible_candidates(
          query, metric, visible_rows, top_k, results)) {
    return false;
  }
  *reranked = true;
  return true;
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
    const committed_entries &candidate_entries) {
  return [&candidate_entries](const committed_entry_visitor &visitor) {
    if (!visitor) return false;

    std::vector<uint64_t> doc_ids;
    doc_ids.reserve(candidate_entries.size());
    for (const auto &entry : candidate_entries) {
      doc_ids.push_back(entry.first);
    }
    std::sort(doc_ids.begin(), doc_ids.end());

    for (const uint64_t doc_id : doc_ids) {
      const auto entry_it = candidate_entries.find(doc_id);
      if (!visitor(entry_it->first, entry_it->second)) {
        return false;
      }
    }
    return true;
  };
}

bool apply_entry_store_change(
    vector_entry_store *entry_store,
    const index_service::pending_change_snapshot &change) {
  if (entry_store == nullptr) return false;
  DBUG_EXECUTE_IF("vector_service_fail_commit_entry_store_apply",
                  return false;);
  if (!change.erase) {
    return entry_store->upsert(change.index_name, change.doc_id, change.vector);
  }
  return entry_store->erase(change.index_name, change.doc_id);
}

}  // namespace

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool pending_change_snapshots_equal_for_testing(
    const std::vector<index_service::pending_change_snapshot> &lhs,
    const std::vector<index_service::pending_change_snapshot> &rhs) {
  return pending_change_snapshots_equal(lhs, rhs);
}

bool pending_budget_allows_for_testing(size_t current_bytes,
                                       size_t additional_bytes) {
  return pending_budget_allows(current_bytes, additional_bytes);
}

bool rerank_candidates_with_vectors_for_testing(
    const vector_data &query, metric_type metric,
    const std::vector<search_result> &candidates,
    const committed_entries &candidate_vectors, size_t top_k,
    std::vector<search_result> *results, bool *reranked) {
  return rerank_candidates_with_vectors(
      query, metric, candidates, candidate_vectors, top_k, results, reranked);
}

committed_entry_reader make_commit_rebuild_reader_for_testing(
    const committed_entries &candidate_entries) {
  return make_commit_rebuild_reader(candidate_entries);
}

bool apply_entry_store_change_for_testing(
    vector_entry_store *entry_store,
    const index_service::pending_change_snapshot &change) {
  return apply_entry_store_change(entry_store, change);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

size_t index_service::pending_change_memory_bytes(
    const pending_change &change) {
  if (change.type != change_type::kUpsert ||
      !change.vector_spill_path.empty()) {
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
  if (element_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max() /
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

bool index_service::write_pending_change_spill(uint64_t txn_id,
                                               const std::string &index_name,
                                               uint64_t doc_id,
                                               const vector_data &vector,
                                               std::string *path) {
  if (path == nullptr) return false;
  static std::atomic<uint64_t> spill_sequence{0};

  std::error_code error;
  const std::filesystem::path directory = pending_spill_directory_path();
  if (directory.empty()) return false;
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
    const auto now =
        std::chrono::steady_clock::now().time_since_epoch().count();
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

bool index_service::cleanup_orphaned_pending_spills() {
  const std::filesystem::path directory = pending_spill_directory_path();
  if (directory.empty()) return true;

  std::error_code error;
  if (!std::filesystem::exists(directory, error)) return !error;
  if (error) return false;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(directory, error);
  if (error || status.type() != std::filesystem::file_type::directory) {
    return false;
  }
  std::filesystem::remove_all(directory, error);
  return !error;
}

void index_service::discard_all_pending_changes() {
  for (const auto &txn_changes : m_pending_changes) {
    remove_pending_change_spills(txn_changes.second, 0);
  }
  m_pending_changes.clear();
  m_savepoints.clear();
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

void index_service::clear_pending_state(uint64_t txn_id) {
  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it != m_pending_changes.end()) {
    remove_pending_change_spills(pending_it->second, 0);
    m_pending_changes.erase(pending_it);
  }
  m_savepoints.erase(txn_id);
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
         build_commit_backends(&plan) && apply_commit_build_plan(txn_id, &plan);
}

bool index_service::snapshot_commit_build_plan(uint64_t txn_id,
                                               commit_build_plan *plan) {
  if (plan == nullptr) return false;
  plan->changes.clear();
  plan->indexes.clear();
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
    const bool bulk_loading = lifecycle_it != m_lifecycle_infos.end() &&
                              lifecycle_is_bulk_loading(lifecycle_it->second);
    if (!bulk_loading) {
      const bool defer_diskann_runtime_load =
          is_external_diskann(index_it->second.get());
      if (!defer_diskann_runtime_load &&
          !ensure_runtime_loaded(change.index_name)) {
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

    commit_index_plan index_plan;
    index_plan.index_name = entry.first;
    index_plan.config = config_it->second;
    index_plan.before_generation = m_entry_store.generation(entry.first);
    if (!describe_publication_state(entry.first,
                                    &index_plan.publication_before)) {
      return false;
    }
    if (index_plan.publication_before.truth_generation ==
        std::numeric_limits<uint64_t>::max()) {
      return false;
    }
    index_plan.target_truth_generation =
        index_plan.publication_before.truth_generation + 1;
    std::unordered_set<uint64_t> captured_doc_ids;
    captured_doc_ids.reserve(entry.second.size());
    for (const pending_change_snapshot *change : entry.second) {
      if (!captured_doc_ids.insert(change->doc_id).second) continue;
      commit_entry_before_image before_image;
      before_image.doc_id = change->doc_id;
      if (!m_entry_store.find_committed_entry(entry.first, change->doc_id,
                                              &before_image.vector,
                                              &before_image.found)) {
        return false;
      }
      index_plan.before_images.push_back(std::move(before_image));
    }

    auto lifecycle_it = m_lifecycle_infos.find(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end() &&
        lifecycle_is_bulk_loading(lifecycle_it->second)) {
      plan->indexes.push_back(std::move(index_plan));
      continue;
    }
    if (is_external_diskann(index_it->second.get()) &&
        (!runtime_loaded_for_search(entry.first) ||
         should_defer_runtime_rebuild_on_commit(index_it->second.get()))) {
      index_plan.defer_runtime_rebuild = true;
      plan->indexes.push_back(std::move(index_plan));
      continue;
    }
    if (!should_batch_rebuild_on_commit(index_it->second.get())) {
      plan->indexes.push_back(std::move(index_plan));
      continue;
    }

    commit_rebuild_plan rebuild_plan;
    rebuild_plan.index_name = entry.first;
    rebuild_plan.config = config_it->second;
    if (!m_entry_store.snapshot_index(entry.first,
                                      &rebuild_plan.candidate_entries)) {
      return false;
    }
    rebuild_plan.changes.reserve(entry.second.size());
    for (const pending_change_snapshot *change : entry.second) {
      rebuild_plan.changes.push_back(*change);
      if (change->erase) {
        rebuild_plan.candidate_entries.erase(change->doc_id);
      } else {
        rebuild_plan.candidate_entries[change->doc_id] = change->vector;
      }
    }
    plan->indexes.push_back(std::move(index_plan));
    plan->rebuilds.push_back(std::move(rebuild_plan));
  }
  return true;
}

bool index_service::build_commit_backends(commit_build_plan *plan) const {
  if (plan == nullptr) return false;
  for (commit_rebuild_plan &rebuild_plan : plan->rebuilds) {
    std::unique_ptr<backend> rebuilt =
        build_backend_from_config(rebuild_plan.index_name, rebuild_plan.config);
    const committed_entry_reader reader =
        make_commit_rebuild_reader(rebuild_plan.candidate_entries);
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
  plan->failure_stage.clear();
  auto fail = [&](const char *stage) {
    plan->failure_stage = stage;
    return false;
  };

  std::vector<pending_change_snapshot> current_changes;
  if (!snapshot_pending_changes(txn_id, &current_changes) ||
      !pending_change_snapshots_equal(plan->changes, current_changes)) {
    return fail("pending_changes_changed");
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
    if (!rebuild_indexes.insert(rebuild_plan.index_name).second) {
      return fail("duplicate_rebuild_index");
    }
  }

  std::unordered_set<std::string> deferred_runtime_indexes;
  deferred_runtime_indexes.reserve(plan->indexes.size());
  for (const commit_index_plan &index_plan : plan->indexes) {
    if (index_plan.defer_runtime_rebuild) {
      deferred_runtime_indexes.insert(index_plan.index_name);
    }
  }

  std::unordered_map<std::string, const commit_index_plan *> index_plans;
  index_plans.reserve(plan->indexes.size());
  for (const commit_index_plan &index_plan : plan->indexes) {
    if (!index_plans.emplace(index_plan.index_name, &index_plan).second) {
      return fail("duplicate_index_plan");
    }
    auto index_it = m_indexes.find(index_plan.index_name);
    auto config_it = m_index_configs.find(index_plan.index_name);
    auto publication_it = m_publication_states.find(index_plan.index_name);
    if (index_it == m_indexes.end() || config_it == m_index_configs.end() ||
        publication_it == m_publication_states.end() ||
        !m_entry_store.has_index(index_plan.index_name)) {
      return fail("index_state_missing");
    }
    const index_publication_state &current_publication = publication_it->second;
    if (m_entry_store.generation(index_plan.index_name) !=
        index_plan.before_generation) {
      return fail("entry_store_generation_changed");
    }
    if (!index_configs_equal(config_it->second, index_plan.config)) {
      return fail("index_config_changed");
    }
    if (current_publication.index_identity !=
        index_plan.publication_before.index_identity) {
      return fail("index_identity_changed");
    }
    if (current_publication.truth_generation !=
        index_plan.publication_before.truth_generation) {
      return fail("truth_generation_changed");
    }
    if (current_publication.config_generation !=
        index_plan.publication_before.config_generation) {
      return fail("config_generation_changed");
    }
    if (current_publication.artifact_generation !=
        index_plan.publication_before.artifact_generation) {
      return fail("artifact_generation_changed");
    }
    if (current_publication.runtime_generation !=
        index_plan.publication_before.runtime_generation) {
      return fail("runtime_generation_changed");
    }
    if (index_plan.target_truth_generation == 0) {
      return fail("target_truth_generation_zero");
    }
    if (index_plan.target_truth_generation <
        current_publication.truth_generation) {
      return fail("target_truth_generation_stale");
    }
  }

  for (const pending_change_snapshot &change : plan->changes) {
    if (index_plans.find(change.index_name) == index_plans.end()) {
      return fail("pending_change_without_index_plan");
    }
  }

  std::vector<std::pair<backend_ptr *, commit_rebuild_plan *>> rebuild_targets;
  rebuild_targets.reserve(plan->rebuilds.size());
  for (commit_rebuild_plan &rebuild_plan : plan->rebuilds) {
    const auto plan_it = index_plans.find(rebuild_plan.index_name);
    auto index_it = m_indexes.find(rebuild_plan.index_name);
    if (rebuild_plan.rebuilt_backend == nullptr ||
        plan_it == index_plans.end() || index_it == m_indexes.end() ||
        !index_configs_equal(rebuild_plan.config, plan_it->second->config)) {
      return fail("rebuilt_backend_invalid");
    }
    rebuild_targets.emplace_back(&index_it->second, &rebuild_plan);
  }

  for (const commit_index_plan &index_plan : plan->indexes) {
    if (!m_entry_store.prepare_index_for_mutation(index_plan.index_name)) {
      return fail("prepare_entry_store_mutation");
    }
  }

  std::unordered_set<std::string> touched_backends;
  auto restore_entry_store = [&]() {
    DBUG_EXECUTE_IF("vector_service_fail_commit_entry_store_restore",
                    return false;);
    bool restored = true;
    for (const commit_index_plan &index_plan : plan->indexes) {
      for (const commit_entry_before_image &before_image :
           index_plan.before_images) {
        const bool ok =
            before_image.found
                ? m_entry_store.upsert(index_plan.index_name,
                                       before_image.doc_id, before_image.vector)
                : m_entry_store.erase(index_plan.index_name,
                                      before_image.doc_id);
        restored = ok && restored;
      }
    }
    return restored;
  };
  auto restore_touched_backends = [&]() {
    DBUG_EXECUTE_IF("vector_service_fail_commit_backend_restore",
                    return false;);
    bool restored = true;
    for (const std::string &index_name : touched_backends) {
      const auto plan_it = index_plans.find(index_name);
      if (plan_it == index_plans.end()) {
        restored = false;
        continue;
      }
      committed_entries entries;
      std::unique_ptr<backend> rebuilt =
          build_backend_from_config(index_name, plan_it->second->config);
      if (rebuilt == nullptr ||
          !m_entry_store.snapshot_index(index_name, &entries) ||
          !rebuilt->rebuild_from_committed_entries(entries)) {
        restored = false;
        continue;
      }
      auto index_it = m_indexes.find(index_name);
      if (index_it == m_indexes.end()) {
        restored = false;
        continue;
      }
      index_it->second = std::move(rebuilt);
    }
    return restored;
  };
  auto fail_and_restore = [&](const std::string &index_name,
                              const char *stage) {
    plan->failure_stage = stage;
    mark_failure_for_index(index_name, ERROR_BACKEND_APPLY_FAILED);
    const bool entry_store_restored = restore_entry_store();
    const bool backends_restored = restore_touched_backends();
    if (!entry_store_restored || !backends_restored) {
      plan->failure_stage += ":rollback_incomplete";
    }
    return false;
  };

  for (const pending_change_snapshot &change : plan->changes) {
    auto index_it = m_indexes.find(change.index_name);
    if (index_it == m_indexes.end()) return fail("mutation_index_missing");
    auto lifecycle_it = m_lifecycle_infos.find(change.index_name);
    if (lifecycle_it != m_lifecycle_infos.end() &&
        lifecycle_is_bulk_loading(lifecycle_it->second)) {
      continue;
    }
    if (rebuild_indexes.find(change.index_name) != rebuild_indexes.end()) {
      continue;
    }
    if (deferred_runtime_indexes.find(change.index_name) !=
        deferred_runtime_indexes.end()) {
      continue;
    }

    touched_backends.insert(change.index_name);
    bool ok = false;
    {
      std::unique_lock<std::shared_mutex> runtime_guard(
          index_it->second->runtime_mutex());
      ok = !change.erase
               ? index_it->second->upsert(change.doc_id, change.vector)
               : index_it->second->erase(change.doc_id);
    }
    if (!ok) {
      return fail_and_restore(change.index_name, "backend_mutation");
    }
  }

  for (const pending_change_snapshot &change : plan->changes) {
    if (!apply_entry_store_change(&m_entry_store, change)) {
      return fail_and_restore(change.index_name, "entry_store_mutation");
    }
  }

  for (auto &target : rebuild_targets) {
    *target.first = std::move(target.second->rebuilt_backend);
    maybe_unload_runtime(target.second->index_name);
  }

  for (const commit_index_plan &index_plan : plan->indexes) {
    auto publication_it = m_publication_states.find(index_plan.index_name);
    if (publication_it == m_publication_states.end()) {
      return fail("publication_state_missing_after_apply");
    }
    publication_it->second.truth_generation =
        index_plan.target_truth_generation;
    auto lifecycle_it = m_lifecycle_infos.find(index_plan.index_name);
    if (lifecycle_it == m_lifecycle_infos.end() ||
        !lifecycle_is_bulk_loading(lifecycle_it->second)) {
      if (index_plan.defer_runtime_rebuild) continue;
      if (!synchronize_runtime_publication(index_plan.index_name)) {
        return fail("synchronize_runtime_publication");
      }
    }
  }

  clear_pending_state(txn_id);
  return true;
}

void index_service::rollback(uint64_t txn_id) { clear_pending_state(txn_id); }

bool index_service::savepoint(uint64_t txn_id, const std::string &name) {
  if (name.empty()) return false;

  const size_t pending_count = pending_change_count(txn_id);
  std::vector<savepoint_marker> &savepoints = m_savepoints[txn_id];
  savepoints.erase(std::remove_if(savepoints.begin(), savepoints.end(),
                                  [&](const savepoint_marker &marker) {
                                    return savepoint_names_equal(marker.name,
                                                                 name);
                                  }),
                   savepoints.end());
  savepoints.push_back(savepoint_marker{name, pending_count});
  return true;
}

bool index_service::rollback_to_savepoint(uint64_t txn_id,
                                          const std::string &name) {
  auto savepoint_it = m_savepoints.find(txn_id);
  if (savepoint_it == m_savepoints.end() || name.empty()) return false;

  std::vector<savepoint_marker> &savepoints = savepoint_it->second;
  auto marker_it =
      std::find_if(savepoints.begin(), savepoints.end(),
                   [&](const savepoint_marker &marker) {
                     return savepoint_names_equal(marker.name, name);
                   });
  if (marker_it == savepoints.end()) return false;

  std::vector<pending_change> &changes = m_pending_changes[txn_id];
  if (changes.size() < marker_it->change_count) return false;
  remove_pending_change_spills(changes, marker_it->change_count);
  changes.resize(marker_it->change_count);

  savepoints.erase(marker_it + 1, savepoints.end());
  return true;
}

bool index_service::release_savepoint(uint64_t txn_id,
                                      const std::string &name) {
  auto savepoint_it = m_savepoints.find(txn_id);
  if (savepoint_it == m_savepoints.end() || name.empty()) return false;

  std::vector<savepoint_marker> &savepoints = savepoint_it->second;
  auto marker_it =
      std::find_if(savepoints.begin(), savepoints.end(),
                   [&](const savepoint_marker &marker) {
                     return savepoint_names_equal(marker.name, name);
                   });
  if (marker_it == savepoints.end()) return false;

  savepoints.erase(marker_it);
  if (savepoints.empty()) m_savepoints.erase(savepoint_it);
  return true;
}

size_t index_service::diskann_exact_rerank_segment_count(
    const std::string &index_name, const index_config &config) const {
  size_t segment_count = 0;
  if (config.consistency_mode == index_consistency_mode::kStandalone) {
    segment_count = m_standalone_store.raw_segment_count(index_name);
  }
  if (segment_count == 0) {
    const auto segment_task_it = m_segment_task_rows.find(index_name);
    if (segment_task_it != m_segment_task_rows.end()) {
      segment_count = segment_task_it->second.size();
    }
  }
  return std::max<size_t>(segment_count, 1);
}

size_t index_service::diskann_exact_rerank_candidate_top_k(
    const std::string &index_name, const index_config &config, size_t top_k,
    size_t query_count) const {
  if (config.provider != backend_provider::kDiskAnn ||
      config.mode != backend_mode::kExternal || top_k == 0 ||
      query_count == 0) {
    return top_k;
  }

  const size_t authoritative_count =
      config.consistency_mode == index_consistency_mode::kStandalone
          ? m_standalone_store.entry_count(index_name)
          : m_entry_store.entry_count(index_name);
  if (authoritative_count <= top_k) return top_k;

  const size_t segment_count =
      diskann_exact_rerank_segment_count(index_name, config);
  return detail::compute_diskann_exact_rerank_candidate_top_k(
      top_k, query_count, authoritative_count, segment_count,
      config.diskann_search_complexity,
      static_cast<diskann_search_profile>(opt_vector_diskann_search_profile),
      static_cast<size_t>(opt_vector_search_batch_result_count),
      static_cast<size_t>(opt_vector_diskann_exact_rerank_candidates));
}

bool index_service::exact_rerank_search_results(
    const std::string &index_name, const index_config &config,
    const vector_data &query, const std::vector<search_result> &candidates,
    size_t top_k, std::vector<search_result> *results, bool *reranked) const {
  if (results == nullptr || reranked == nullptr) return false;
  *reranked = false;
  if (candidates.empty()) {
    results->clear();
    *reranked = true;
    return true;
  }

  std::unordered_set<uint64_t> candidate_doc_ids;
  for (const search_result &candidate : candidates) {
    candidate_doc_ids.insert(candidate.doc_id);
  }

  committed_entries candidate_vectors;
  bool complete = false;
  if (!load_exact_rerank_vectors(index_name, config, candidate_doc_ids,
                                 &candidate_vectors, &complete)) {
    return false;
  }
  if (!complete) return true;
  return rerank_candidates_with_vectors(query, config.metric, candidates,
                                        candidate_vectors, top_k, results,
                                        reranked);
}

bool index_service::load_exact_rerank_vectors(
    const std::string &index_name, const index_config &config,
    const std::unordered_set<uint64_t> &doc_ids, committed_entries *vectors,
    bool *complete) const {
  if (vectors == nullptr || complete == nullptr) return false;
  vectors->clear();
  *complete = false;
  if (doc_ids.empty()) {
    *complete = true;
    return true;
  }

  if (config.consistency_mode == index_consistency_mode::kStandalone) {
    if (!m_standalone_store.find_entries(index_name, doc_ids, vectors))
      return false;
  } else {
    for (const uint64_t doc_id : doc_ids) {
      vector_data vector;
      bool found = false;
      if (!m_entry_store.find_committed_entry(index_name, doc_id, &vector,
                                              &found)) {
        return false;
      }
      if (found) (*vectors)[doc_id] = std::move(vector);
    }
  }

  *complete = vectors->size() == doc_ids.size();
  return true;
}

bool index_service::search_runtime_with_exact_rerank(
    const std::string &index_name, const index_config &config,
    const backend *runtime, const vector_data &query, size_t top_k,
    std::vector<search_result> *results) const {
  if (runtime == nullptr || results == nullptr) return false;

  const size_t candidate_top_k =
      diskann_exact_rerank_candidate_top_k(index_name, config, top_k, 1);
  std::vector<search_result> candidates;
  if (!runtime->search_for_rerank(query, top_k, candidate_top_k, &candidates))
    return false;

  return finish_search_with_exact_rerank(index_name, config, query, top_k,
                                         std::move(candidates), results);
}

bool index_service::finish_search_with_exact_rerank(
    const std::string &index_name, const index_config &config,
    const vector_data &query, size_t top_k,
    std::vector<search_result> candidates,
    std::vector<search_result> *results) const {
  if (results == nullptr) return false;

  const bool can_exact_rerank = config.provider == backend_provider::kDiskAnn &&
                                config.mode == backend_mode::kExternal;
  if (can_exact_rerank) {
    bool reranked = false;
    if (!exact_rerank_search_results(index_name, config, query, candidates,
                                     top_k, results, &reranked)) {
      return false;
    }
    if (reranked) return true;
  }

  if (candidates.size() > top_k) candidates.resize(top_k);
  *results = std::move(candidates);
  return true;
}

bool index_service::search_batch_runtime_with_exact_rerank(
    const std::string &index_name, const index_config &config,
    const backend *runtime, const std::vector<vector_data> &queries,
    size_t top_k, std::vector<std::vector<search_result>> *results) const {
  if (runtime == nullptr || results == nullptr) return false;

  const size_t candidate_top_k = diskann_exact_rerank_candidate_top_k(
      index_name, config, top_k, queries.size());
  batch_search_candidates candidates;
  if (!runtime->search_batch_for_rerank(queries, top_k, candidate_top_k,
                                        &candidates))
    return false;

  return finish_search_batch_with_exact_rerank(
      index_name, config, queries, top_k, std::move(candidates), results);
}

bool index_service::finish_search_batch_with_exact_rerank(
    const std::string &index_name, const index_config &config,
    const std::vector<vector_data> &queries, size_t top_k,
    batch_search_candidates candidates,
    std::vector<std::vector<search_result>> *results) const {
  if (results == nullptr || candidates.query_count() != queries.size()) {
    return false;
  }

  const bool can_exact_rerank = config.provider == backend_provider::kDiskAnn &&
                                config.mode == backend_mode::kExternal;
  if (can_exact_rerank) {
    std::unordered_set<uint64_t> candidate_doc_ids;
    for (size_t i = 0; i < candidates.query_count(); ++i) {
      const std::vector<search_result> *query_candidates =
          candidates.for_query(i);
      if (query_candidates == nullptr) return false;
      for (const search_result &candidate : *query_candidates) {
        candidate_doc_ids.insert(candidate.doc_id);
      }
    }
    committed_entries candidate_vectors;
    bool complete = false;
    if (!load_exact_rerank_vectors(index_name, config, candidate_doc_ids,
                                   &candidate_vectors, &complete)) {
      return false;
    }

    results->clear();
    results->resize(candidates.query_count());
    for (size_t i = 0; i < candidates.query_count(); ++i) {
      const std::vector<search_result> *query_candidates =
          candidates.for_query(i);
      if (query_candidates == nullptr) return false;
      bool reranked = false;
      if (complete) {
        if (!rerank_candidates_with_vectors(queries[i], config.metric,
                                            *query_candidates,
                                            candidate_vectors, top_k,
                                            &(*results)[i], &reranked)) {
          return false;
        }
      }
      if (!reranked) {
        const size_t result_count = std::min(top_k, query_candidates->size());
        (*results)[i].assign(query_candidates->begin(),
                             query_candidates->begin() + result_count);
      }
    }
    return true;
  }

  return std::move(candidates).materialize(results);
}

bool index_service::search(const std::string &index_name,
                           const vector_data &query, size_t top_k,
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
    if (search_runtime_with_exact_rerank(index_name, config, runtime.get(),
                                         query, top_k, results)) {
      return true;
    }
  }
  if (!detail::can_rebuild_after_search_failure(config)) return false;

  index_service *self = const_cast<index_service *>(this);
  if (!self->rebuild_runtime_from_store_for_search(index_name)) return false;
  if (!snapshot_search_backend_loaded(index_name, query, &runtime, &config))
    return false;
  std::shared_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
  return search_runtime_with_exact_rerank(index_name, config, runtime.get(),
                                          query, top_k, results);
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
  const auto publication_it = m_publication_states.find(index_name);
  if (publication_it == m_publication_states.end() ||
      publication_it->second.runtime_generation !=
          publication_it->second.truth_generation) {
    return false;
  }
  auto index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end() || index_it->second == nullptr) return false;
  *runtime = index_it->second;
  if (config != nullptr) *config = config_it->second;
  return true;
}

bool index_service::snapshot_search_runtime_loaded(
    const std::string &index_name, size_t query_dimension, size_t top_k,
    size_t query_count, search_runtime_snapshot *snapshot) const {
  if (snapshot == nullptr || query_count == 0) return false;
  *snapshot = search_runtime_snapshot{};

  const auto lifecycle_it = m_lifecycle_infos.find(index_name);
  const auto config_it = m_index_configs.find(index_name);
  const auto publication_it = m_publication_states.find(index_name);
  const auto index_it = m_indexes.find(index_name);
  if (lifecycle_it == m_lifecycle_infos.end() ||
      config_it == m_index_configs.end() ||
      publication_it == m_publication_states.end() ||
      index_it == m_indexes.end() || index_it->second == nullptr ||
      lifecycle_is_bulk_loading(lifecycle_it->second) ||
      query_dimension != config_it->second.dimension ||
      publication_it->second.runtime_generation !=
          publication_it->second.truth_generation) {
    return false;
  }

  snapshot->runtime = index_it->second;
  snapshot->config = config_it->second;
  snapshot->publication = publication_it->second;
  snapshot->candidate_top_k = diskann_exact_rerank_candidate_top_k(
      index_name, snapshot->config, top_k, query_count);
  return true;
}

bool index_service::search_runtime_snapshot_matches(
    const std::string &index_name,
    const search_runtime_snapshot &snapshot) const {
  const auto lifecycle_it = m_lifecycle_infos.find(index_name);
  const auto config_it = m_index_configs.find(index_name);
  const auto publication_it = m_publication_states.find(index_name);
  const auto index_it = m_indexes.find(index_name);
  if (lifecycle_it == m_lifecycle_infos.end() ||
      config_it == m_index_configs.end() ||
      publication_it == m_publication_states.end() ||
      index_it == m_indexes.end() || index_it->second == nullptr ||
      snapshot.runtime == nullptr ||
      lifecycle_is_bulk_loading(lifecycle_it->second) ||
      index_it->second != snapshot.runtime ||
      !index_configs_equal(config_it->second, snapshot.config)) {
    return false;
  }

  const index_publication_state &current = publication_it->second;
  const index_publication_state &expected = snapshot.publication;
  return current.index_identity == expected.index_identity &&
         current.truth_generation == expected.truth_generation &&
         current.config_generation == expected.config_generation &&
         current.artifact_generation == expected.artifact_generation &&
         current.runtime_generation == expected.runtime_generation &&
         current.runtime_generation == current.truth_generation;
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
  if (config_it == m_index_configs.end()) return false;
  const auto publication_it = m_publication_states.find(index_name);
  if (publication_it == m_publication_states.end() ||
      publication_it->second.runtime_generation !=
          publication_it->second.truth_generation) {
    return false;
  }
  for (const vector_data &query : queries) {
    if (query.size() != config_it->second.dimension) return false;
  }
  auto index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end() || index_it->second == nullptr) return false;
  backend_ptr runtime = index_it->second;
  {
    std::shared_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
    if (search_batch_runtime_with_exact_rerank(index_name, config_it->second,
                                               runtime.get(), queries, top_k,
                                               results)) {
      return true;
    }
  }
  if (!detail::can_rebuild_after_search_failure(config_it->second)) {
    return false;
  }
  index_service *self = const_cast<index_service *>(this);
  if (!self->rebuild_runtime_from_store_for_search(index_name)) return false;
  index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end() || index_it->second == nullptr) return false;
  runtime = index_it->second;
  std::shared_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
  return search_batch_runtime_with_exact_rerank(
      index_name, config_it->second, runtime.get(), queries, top_k, results);
}

bool index_service::search_with_pending(
    uint64_t txn_id, const std::string &index_name, const vector_data &query,
    size_t top_k, std::vector<search_result> *results) const {
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
        deltas[key] = pending_change_snapshot{change.index_name, false,
                                              change.doc_id, std::move(vector)};
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
    clear_pending_state(txn_id);
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
    if (!change.erase &&
        index_it->second->dimension() != change.vector.size()) {
      remove_pending_change_spills(restored, 0);
      return false;
    }
    pending_change restored_change{
        change.index_name,
        change.erase ? change_type::kErase : change_type::kUpsert,
        change.doc_id,
        change.vector,
        {}};
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

bool index_service::snapshot_pending_state(
    uint64_t txn_id, pending_state_snapshot *state) const {
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
  DBUG_EXECUTE_IF("vector_service_fail_restore_pending_state", return false;);
  if (!restore_pending_changes(txn_id, state.changes)) return false;
  if (state.savepoints.empty()) {
    m_savepoints.erase(txn_id);
    return true;
  }

  std::vector<savepoint_marker> restored_savepoints;
  restored_savepoints.reserve(state.savepoints.size());
  const size_t pending_count = pending_change_count(txn_id);
  for (const pending_savepoint_snapshot &marker : state.savepoints) {
    if (marker.name.empty() || marker.change_count > pending_count)
      return false;
    restored_savepoints.push_back(
        savepoint_marker{marker.name, marker.change_count});
  }
  m_savepoints[txn_id] = std::move(restored_savepoints);
  return true;
}

}  // namespace vector_index
