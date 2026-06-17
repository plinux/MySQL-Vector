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

#ifndef SQL_VECTOR_INDEX_SERVICE_INCLUDED
#define SQL_VECTOR_INDEX_SERVICE_INCLUDED

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sql/vector/vector_index_backend.h"

namespace vector_index {

using committed_entries = std::unordered_map<uint64_t, vector_data>;
using committed_state = std::unordered_map<std::string, committed_entries>;

/**
  Bounded committed vector entry cache.

  The truth-store is the authoritative source after commit persistence. This
  cache keeps hot committed entries in memory and can evict whole indexes once
  the configured memory budget is exceeded.
*/
class vector_entry_store {
 public:
  using entry_visitor = std::function<bool(uint64_t, const vector_data &)>;

  bool register_index(const std::string &index_name);
  bool drop_index(const std::string &index_name);
  bool rename_index(const std::string &old_index_name,
                    const std::string &new_index_name);
  bool has_index(const std::string &index_name) const;
  bool clear_index(const std::string &index_name);
  bool replace_index(const std::string &index_name,
                     const committed_entries &entries);
  bool upsert(const std::string &index_name, uint64_t doc_id,
              const vector_data &vector);
  bool erase(const std::string &index_name, uint64_t doc_id);
  bool snapshot(committed_state *state) const;
  bool snapshot_index(const std::string &index_name,
                      committed_entries *entries) const;
  bool find_committed_entry(const std::string &index_name, uint64_t doc_id,
                            vector_data *vector, bool *found) const;
  bool for_each_committed_entry(const std::string &index_name,
                                const entry_visitor &visitor) const;
  bool evict_until_under_budget(size_t budget);
  size_t entry_count() const;
  size_t entry_count(const std::string &index_name) const;
  size_t memory_bytes() const;
  size_t memory_bytes(const std::string &index_name) const;
  uint64_t generation(const std::string &index_name) const;

 private:
  bool load_index_from_truth_store(const std::string &index_name,
                                   committed_entries *entries) const;
  bool ensure_index_cached(const std::string &index_name);
  void bump_generation(const std::string &index_name);

  committed_state m_entries;
  std::unordered_map<std::string, size_t> m_entry_counts;
  std::unordered_map<std::string, uint64_t> m_index_generations;
  std::unordered_set<std::string> m_evicted_indexes;
};

/**
  Transaction-buffered index service on top of backend implementations.

  This service is a lightweight integration layer used by SQL/InnoDB glue code.
  It stages updates per transaction id and applies them atomically on commit.
*/
class index_service {
 public:
  using committed_entries = vector_index::committed_entries;
  using committed_state = vector_index::committed_state;
  using backend_ptr = std::shared_ptr<backend>;

  struct lifecycle_info {
    std::string state{"ready"};
    uint64_t version{1};
    uint32_t last_error_code{0};
    uint64_t last_error_ts{0};
    uint64_t last_apply_latency_ms{0};
    uint64_t recover_fallback_count{0};
    uint64_t last_recover_fallback_ts{0};
  };

  struct index_config {
    size_t dimension{0};
    metric_type metric{metric_type::kEuclidean};
    backend_mode mode{backend_mode::kMemory};
    backend_provider provider{backend_provider::kNative};
    std::string backend_variant;
    uint32_t search_ef{0};
    uint32_t hnsw_m{0};
    uint32_t hnsw_ef_construction{0};
    uint32_t hnsw_build_threads{0};
    uint32_t faiss_nlist{0};
    uint32_t faiss_nprobe{0};
    uint32_t faiss_pq_m{0};
    uint32_t faiss_pq_bits{0};
    uint32_t faiss_build_threads{0};
    uint32_t diskann_max_degree{0};
    uint32_t diskann_build_complexity{0};
    uint32_t diskann_build_threads{0};
    uint32_t diskann_search_complexity{0};
  };

  struct pending_change_snapshot {
    std::string index_name;
    bool erase{false};
    uint64_t doc_id{0};
    vector_data vector;
  };

  struct pending_savepoint_snapshot {
    std::string name;
    size_t change_count{0};
  };

  struct pending_state_snapshot {
    std::vector<pending_change_snapshot> changes;
    std::vector<pending_savepoint_snapshot> savepoints;
  };

  struct commit_rebuild_plan {
    std::string index_name;
    index_config config;
    uint64_t before_generation{0};
    std::vector<pending_change_snapshot> changes;
    std::unique_ptr<backend> rebuilt_backend;
  };

  struct commit_build_plan {
    std::vector<pending_change_snapshot> changes;
    std::vector<commit_rebuild_plan> rebuilds;
  };

  bool register_index(const std::string &index_name,
                      std::unique_ptr<backend> backend);
  bool register_index(const std::string &index_name,
                      const index_config &config);
  bool register_index_from_strings(const std::string &index_name,
                                   size_t dimension, const std::string &metric,
                                   const std::string &mode,
                                   const std::string &provider);
  bool drop_index(const std::string &index_name);
  bool unregister_index(const std::string &index_name);
  bool rename_index(const std::string &old_index_name,
                    const std::string &new_index_name);
  bool begin_bulk_load(const std::string &index_name);
  bool rebuild_index(const std::string &index_name);
  bool recover_index(const std::string &index_name);
  bool rebuild_all_indexes(size_t *rebuilt_count);
  bool recover_all_indexes(size_t *recovered_count);
  bool set_search_ef(const std::string &index_name, uint32_t search_ef);
  bool set_hnsw_build_params(const std::string &index_name, uint32_t hnsw_m,
                             uint32_t hnsw_ef_construction);
  bool set_hnsw_build_threads(const std::string &index_name,
                              uint32_t hnsw_build_threads);
  bool set_faiss_ivf_params(const std::string &index_name, uint32_t faiss_nlist,
                            uint32_t faiss_nprobe);
  bool set_faiss_ivf_pq_params(const std::string &index_name,
                               uint32_t faiss_nlist, uint32_t faiss_nprobe,
                               uint32_t faiss_pq_m, uint32_t faiss_pq_bits);
  bool set_faiss_build_threads(const std::string &index_name,
                               uint32_t faiss_build_threads);
  bool set_diskann_build_params(const std::string &index_name,
                                uint32_t diskann_max_degree,
                                uint32_t diskann_build_complexity,
                                uint32_t diskann_build_threads);
  bool set_diskann_build_threads(const std::string &index_name,
                                 uint32_t diskann_build_threads);
  bool set_diskann_search_complexity(const std::string &index_name,
                                     uint32_t diskann_search_complexity);
  bool restore_index_config(const std::string &index_name,
                            const index_config &config);

  bool stage_upsert(uint64_t txn_id, const std::string &index_name,
                    uint64_t doc_id, const vector_data &vector);
  bool stage_erase(uint64_t txn_id, const std::string &index_name,
                   uint64_t doc_id);

  bool commit(uint64_t txn_id);
  bool snapshot_commit_build_plan(uint64_t txn_id, commit_build_plan *plan);
  bool build_commit_backends(commit_build_plan *plan) const;
  bool apply_commit_build_plan(uint64_t txn_id, commit_build_plan *plan);
  void rollback(uint64_t txn_id);
  bool savepoint(uint64_t txn_id, const std::string &name);
  bool rollback_to_savepoint(uint64_t txn_id, const std::string &name);
  bool release_savepoint(uint64_t txn_id, const std::string &name);

  bool search(const std::string &index_name, const vector_data &query,
              size_t top_k, std::vector<search_result> *results) const;
  bool search_batch(const std::string &index_name,
                    const std::vector<vector_data> &queries, size_t top_k,
                    std::vector<std::vector<search_result>> *results) const;
  bool search_loaded(const std::string &index_name, const vector_data &query,
                     size_t top_k,
                     std::vector<search_result> *results) const;
  bool snapshot_search_backend_loaded(const std::string &index_name,
                                      const vector_data &query,
                                      backend_ptr *runtime,
                                      index_config *config = nullptr) const;
  bool search_batch_loaded(
      const std::string &index_name, const std::vector<vector_data> &queries,
      size_t top_k, std::vector<std::vector<search_result>> *results) const;
  bool search_with_pending(uint64_t txn_id, const std::string &index_name,
                           const vector_data &query, size_t top_k,
                           std::vector<search_result> *results) const;
  bool search_with_pending_loaded(
      uint64_t txn_id, const std::string &index_name, const vector_data &query,
      size_t top_k, std::vector<search_result> *results) const;
  bool ensure_runtime_loaded_for_search(const std::string &index_name);
  bool rebuild_runtime_from_store_for_search(const std::string &index_name);
  bool describe_index(const std::string &index_name, index_config *config,
                      bool *supports_mutations, size_t *entry_count,
                      size_t *committed_entry_count,
                      std::string *lifecycle_state = nullptr,
                      uint64_t *lifecycle_version = nullptr,
                      uint32_t *last_error_code = nullptr,
                      uint64_t *last_error_ts = nullptr,
                      uint64_t *last_apply_latency_ms = nullptr,
                      uint64_t *recover_fallback_count = nullptr,
                      uint64_t *last_recover_fallback_ts = nullptr,
                      bool *external_manifest_present = nullptr,
                      uint64_t *external_manifest_generation = nullptr) const;
  bool set_last_apply_latency_ms(const std::string &index_name,
                                 uint64_t latency_ms);
  bool set_lifecycle_state(const std::string &index_name,
                           const std::string &lifecycle_state);
  bool set_lifecycle_info(const std::string &index_name,
                          const std::string &lifecycle_state,
                          uint64_t lifecycle_version,
                          uint32_t last_error_code = 0,
                          uint64_t last_error_ts = 0,
                          uint64_t recover_fallback_count = 0,
                          uint64_t last_recover_fallback_ts = 0);
  bool list_indexes(std::vector<std::string> *index_names) const;
  bool snapshot_committed_state(committed_state *state) const;
  size_t committed_entry_count() const;
  size_t committed_vector_memory_bytes() const;
  bool evict_committed_cache_to_budget();
  size_t pending_vector_memory_bytes(uint64_t txn_id) const;
  size_t total_pending_vector_memory_bytes() const;
  bool restore_committed_state(const committed_state &state);
  bool replace_committed_entries(const std::string &index_name,
                                 const committed_entries &entries);
  bool replace_committed_entries_preserve_lifecycle(
      const std::string &index_name, const committed_entries &entries);
  bool install_rebuilt_index(const std::string &index_name,
                             const committed_entries &entries,
                             std::unique_ptr<backend> rebuilt_backend);
  bool install_recovered_index(const std::string &index_name,
                               const committed_entries &entries,
                               std::unique_ptr<backend> recovered_backend,
                               bool used_recover_fallback);

  size_t pending_change_count(uint64_t txn_id) const;
  bool has_pending_changes_for_index(const std::string &index_name) const;
  bool pending_change_index_names(uint64_t txn_id,
                                  std::vector<std::string> *index_names) const;
  bool snapshot_pending_changes(
      uint64_t txn_id, std::vector<pending_change_snapshot> *changes) const;
  bool snapshot_pending_change_delta(
      uint64_t txn_id, std::vector<pending_change_snapshot> *changes) const;
  bool restore_pending_changes(
      uint64_t txn_id, const std::vector<pending_change_snapshot> &changes);
  bool snapshot_pending_state(uint64_t txn_id,
                              pending_state_snapshot *state) const;
  bool restore_pending_state(uint64_t txn_id,
                             const pending_state_snapshot &state);

 private:
  enum class change_type { kUpsert, kErase };

  struct pending_change {
    std::string index_name;
    change_type type{change_type::kUpsert};
    uint64_t doc_id{0};
    vector_data vector;
    std::string vector_spill_path;
  };

  struct savepoint_marker {
    std::string name;
    size_t change_count{0};
  };

  std::unordered_map<std::string, backend_ptr> m_indexes;
  std::unordered_map<std::string, index_config> m_index_configs;
  vector_entry_store m_entry_store;
  std::unordered_map<std::string, lifecycle_info> m_lifecycle_infos;
  std::unordered_map<uint64_t, std::vector<pending_change>> m_pending_changes;
  std::unordered_map<uint64_t, std::vector<savepoint_marker>> m_savepoints;

  static size_t pending_change_memory_bytes(const pending_change &change);
  static bool read_pending_change_vector(const pending_change &change,
                                         vector_data *vector);
  static bool write_pending_change_spill(uint64_t txn_id,
                                         const std::string &index_name,
                                         uint64_t doc_id,
                                         const vector_data &vector,
                                         std::string *path);
  static void remove_pending_change_spill(const pending_change &change);
  static void remove_pending_change_spills(
      const std::vector<pending_change> &changes, size_t first_change);

  bool register_index_impl(const std::string &index_name, index_config config,
                           std::unique_ptr<backend> backend);
  bool replace_committed_entries_impl(const std::string &index_name,
                                      const committed_entries &entries,
                                      bool preserve_lifecycle);
  bool ensure_runtime_loaded(const std::string &index_name);
  void maybe_unload_runtime(const std::string &index_name);
};

}  // namespace vector_index

#endif  // SQL_VECTOR_INDEX_SERVICE_INCLUDED
