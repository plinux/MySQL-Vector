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

#include "sql/vector/vector_build_pipeline_policy.h"
#include "sql/vector/vector_diskann_scheduler.h"
#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_segment_task.h"

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
  /** Materialize an evicted index before a multi-step mutation sequence. */
  bool prepare_index_for_mutation(const std::string &index_name);
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
  Non-transactional named-index input store.

  Standalone indexes do not use the InnoDB truth-store as their authoritative
  input. This store keeps recent ingest deltas in memory and spills them to
  standalone segment files when the configured vector entry cache budget is
  exceeded.
*/
class standalone_entry_store {
 public:
  using entry_visitor = std::function<bool(uint64_t, const vector_data &)>;

  /**
    Staged replacement for a raw standalone segment set.

    Compaction writes the replacement files before changing the persisted
    manifest. The caller can therefore build a shadow runtime from `segments`
    and publish both only after the build succeeds.
  */
  struct raw_segment_compaction {
    bool needed{false};
    bool published{false};
    uint64_t source_generation{0};
    uint64_t source_next_segment_id{0};
    std::string source_build_source;
    uint64_t next_generation{0};
    uint64_t next_segment_id{0};
    std::vector<raw_vector_segment> source_segments;
    std::vector<raw_vector_segment> segments;
    std::vector<std::string> created_files;
    std::vector<std::string> replaced_files;
  };

  bool register_index(const std::string &index_name, size_t dimension);
  bool drop_index(const std::string &index_name);
  bool rename_index(const std::string &old_index_name,
                    const std::string &new_index_name);
  bool has_index(const std::string &index_name) const;
  bool upsert(const std::string &index_name, uint64_t doc_id,
              const vector_data &vector, size_t cache_budget);
  bool bulk_upsert(const std::string &index_name,
                   const committed_entries &entries);
  bool bulk_upsert_raw_files(const std::string &index_name,
                             const std::string &vector_filename,
                             const std::string &docid_filename,
                             uint64_t row_count, size_t dimension,
                             uint64_t row_limit,
                             std::unordered_set<uint64_t> *loaded_doc_ids =
                                 nullptr);
  bool erase(const std::string &index_name, uint64_t doc_id,
             size_t cache_budget);
  bool prepare_raw_segments_for_rebuild(const std::string &index_name);
  bool stage_levelled_compaction(const std::string &index_name,
                                 uint64_t row_limit,
                                 raw_segment_compaction *compaction);
  bool publish_levelled_compaction(const std::string &index_name,
                                   raw_segment_compaction *compaction);
  bool rollback_levelled_compaction(const std::string &index_name,
                                    raw_segment_compaction *compaction);
  void finalize_levelled_compaction(
      raw_segment_compaction *compaction) const;
  void discard_levelled_compaction(raw_segment_compaction *compaction) const;
  bool read_rebuild_raw_segments(
      const std::string &index_name,
      const raw_vector_segment_visitor &visitor) const;
  bool rebuild_backend_input(const std::string &index_name, backend *target);
  bool for_each_entry(const std::string &index_name,
                      const entry_visitor &visitor) const;
  bool find_entry(const std::string &index_name, uint64_t doc_id,
                  vector_data *vector, bool *found) const;
  bool find_entries(const std::string &index_name,
                    const std::unordered_set<uint64_t> &doc_ids,
                    committed_entries *vectors) const;
  size_t entry_count(const std::string &index_name) const;
  size_t memory_bytes(const std::string &index_name) const;
  size_t segment_count(const std::string &index_name) const;
  size_t segment_bytes(const std::string &index_name) const;
  size_t raw_segment_count(const std::string &index_name) const;
  size_t raw_segment_bytes(const std::string &index_name) const;
  /** Number of immutable runs in the derived non-dense raw locator. */
  size_t raw_locator_run_count(const std::string &index_name) const;
  /** Number of non-dense raw rows covered by the derived locator. */
  size_t raw_locator_entry_count(const std::string &index_name) const;
  /** Bytes used by the derived raw locator, excluding vector payloads. */
  size_t raw_locator_bytes(const std::string &index_name) const;
  std::string build_source(const std::string &index_name) const;
  uint64_t generation(const std::string &index_name) const;

 private:
  enum class standalone_segment_kind { kDelta, kRawFbin };

  struct standalone_segment {
    standalone_segment_kind kind{standalone_segment_kind::kDelta};
    std::string path;
    std::string vector_path;
    std::string docid_path;
    size_t record_count{0};
    size_t dimension{0};
    size_t bytes{0};
    uint64_t generation{0};
    bool dense_doc_ids{false};
    uint64_t first_doc_id{0};
  };

  struct raw_entry_location {
    uint64_t doc_id{0};
    uint32_t segment_index{0};
    uint32_t row{0};
  };

  static_assert(sizeof(raw_entry_location) == 16);
  using raw_entry_locator = std::vector<raw_entry_location>;

  struct index_state {
    size_t dimension{0};
    std::unordered_set<uint64_t> live_doc_ids;
    committed_entries memory_entries;
    std::unordered_set<uint64_t> memory_erases;
    std::vector<standalone_segment> segments;
    // Derived from non-dense raw doc-id files; vectors remain file-backed.
    std::vector<std::shared_ptr<const raw_entry_locator>> raw_locator_runs;
    size_t entry_count{0};
    size_t memory_bytes{0};
    uint64_t generation{0};
    uint64_t next_segment_id{1};
    std::string build_source{"memory"};
  };

  bool flush_index(const std::string &index_name, index_state *state);
  bool flush_if_needed(const std::string &index_name, index_state *state,
                       size_t cache_budget);
  bool compact_to_raw_segment(const std::string &index_name,
                              index_state *state);
  bool write_compacted_raw_segment(
      const std::string &index_name,
      const std::vector<raw_vector_segment> &source_segments,
      uint64_t segment_id, uint64_t generation,
      raw_vector_segment *compacted_segment) const;
  bool assign_raw_segments(
      index_state *state,
      const std::vector<raw_vector_segment> &raw_segments) const;
  bool build_raw_locator_run(
      const std::vector<standalone_segment> &segments,
      size_t first_segment_index,
      std::shared_ptr<const raw_entry_locator> *locator) const;
  bool consolidate_raw_locator_runs(index_state *state) const;
  bool materialized_rebuild_fits_budget(const index_state &state) const;
  bool read_raw_segments(const index_state &state,
                         const raw_vector_segment_visitor &visitor) const;
  bool can_rebuild_direct_from_raw_segments(const index_state &state) const;
  std::string build_source_for_state(const index_state &state) const;
  bool load_manifest(const std::string &index_name, index_state *state) const;
  bool save_manifest(const std::string &index_name,
                     const index_state &state) const;
  bool replay_segments(const index_state &state,
                       committed_entries *entries) const;
  bool load_entries(const index_state &state,
                    committed_entries *entries) const;
  std::string segment_directory(const std::string &index_name) const;
  std::string manifest_path(const std::string &index_name) const;
  std::string segment_path(const std::string &index_name,
                           uint64_t segment_id) const;
  std::string raw_vector_path(const std::string &index_name,
                              uint64_t segment_id) const;
  std::string raw_docid_path(const std::string &index_name,
                             uint64_t segment_id) const;

  std::unordered_map<std::string, index_state> m_indexes;
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

  /** Durable identity and generation tuple for one published index runtime. */
  struct index_publication_state {
    uint64_t index_identity{0};
    uint64_t truth_generation{0};
    uint64_t config_generation{1};
    uint64_t artifact_generation{0};
    uint64_t runtime_generation{0};
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
    uint32_t diskann_build_blas_threads{0};
    diskann_build_mode diskann_build_mode_value{diskann_build_mode::kAuto};
    uint32_t diskann_search_complexity{0};
    uint32_t diskann_search_beamwidth{0};
    uint64_t diskann_pq_code_budget_size{0};
    uint32_t diskann_disk_pq_dims{0};
    uint32_t diskann_cache_nodes{0};
    bool diskann_accelerate_build{false};
    bool diskann_shuffle_build{false};
    bool diskann_use_bfs_cache{false};
    bool diskann_build_mode_specified{false};
    bool diskann_segmented_serving{false};
    index_consistency_mode consistency_mode{
        index_consistency_mode::kTransactional};
  };

  /** Immutable backend, configuration, and generation tuple for one search. */
  struct search_runtime_snapshot {
    backend_ptr runtime;
    index_config config;
    index_publication_state publication;
    size_t candidate_top_k{0};
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

  struct commit_entry_before_image {
    uint64_t doc_id{0};
    bool found{false};
    vector_data vector;
  };

  /** Immutable validation and rollback state for one affected index. */
  struct commit_index_plan {
    std::string index_name;
    index_config config;
    uint64_t before_generation{0};
    index_publication_state publication_before;
    uint64_t target_truth_generation{0};
    bool defer_runtime_rebuild{false};
    std::vector<commit_entry_before_image> before_images;
  };

  /** Candidate committed state and shadow backend for a rebuild index. */
  struct commit_rebuild_plan {
    std::string index_name;
    index_config config;
    std::vector<pending_change_snapshot> changes;
    committed_entries candidate_entries;
    std::unique_ptr<backend> rebuilt_backend;
  };

  struct commit_build_plan {
    std::vector<pending_change_snapshot> changes;
    std::vector<commit_index_plan> indexes;
    std::vector<commit_rebuild_plan> rebuilds;
    std::string failure_stage;
  };

  using bulk_load_visitor =
      std::function<bool(uint64_t doc_id, const float *values,
                         size_t dimension)>;
  using bulk_load_reader =
      std::function<bool(const bulk_load_visitor &visitor,
                         std::string *error)>;

  struct bulk_load_options {
    bool replace_duplicates{false};
    bool rebuild_after_load{false};
    std::string source_format;
  };

  struct build_pipeline_snapshot {
    std::string mode{"auto"};
    std::string diskann_segment_profile{"manual"};
    std::string diskann_segment_profile_reason{"manual"};
    std::string decision{"direct"};
    std::string trigger{"below_threshold"};
    uint64_t row_count{0};
    uint64_t payload_size{0};
    uint64_t raw_segment_count{0};
    uint64_t effective_segment_target_size{0};
    uint64_t effective_segment_row_limit{0};
  };

  struct standalone_rebuild_plan {
    bool prepared{false};
    bool built{false};
    bool published{false};
    bool segmented{false};
    bool had_previous_pipeline{false};
    bool had_previous_segment_tasks{false};
    bool had_previous_failed_build_diagnostics{false};
    bool has_failed_build_diagnostics{false};
    std::string index_name;
    index_config source_config;
    index_config build_config;
    size_t source_entry_count{0};
    uint64_t source_lifecycle_version{0};
    uint64_t source_generation{0};
    backend_ptr previous_runtime;
    lifecycle_info previous_lifecycle;
    index_publication_state previous_publication;
    index_publication_state target_publication;
    build_pipeline_snapshot previous_pipeline;
    build_pipeline_snapshot pipeline;
    std::vector<vector_index_metadata_store::segment_task_row>
        previous_segment_tasks;
    std::vector<vector_index_metadata_store::segment_task_row> segment_tasks;
    backend_build_diagnostics previous_failed_build_diagnostics;
    backend_build_diagnostics failed_build_diagnostics;
    standalone_entry_store::raw_segment_compaction compaction;
    std::vector<raw_vector_segment> raw_segments;
    backend_ptr rebuilt_backend;
    bool artifact_prepared{false};
    bool artifact_published{false};
  };

  struct index_observability_state {
    bool truth_store_enabled{true};
    size_t authoritative_entry_count{0};
    size_t standalone_ingest_memory_bytes{0};
    size_t standalone_segment_count{0};
    size_t standalone_segment_bytes{0};
    size_t standalone_raw_segment_count{0};
    size_t standalone_raw_segment_bytes{0};
    std::string build_source{"truth_store"};
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
  bool rebuild_index(const std::string &index_name,
                     std::string *error = nullptr);
  bool prepare_standalone_rebuild(const std::string &index_name,
                                  standalone_rebuild_plan *plan,
                                  std::string *error = nullptr);
  bool build_standalone_rebuild(standalone_rebuild_plan *plan,
                                std::string *error = nullptr);
  bool prepare_standalone_rebuild_artifact(
      standalone_rebuild_plan *plan,
      const diskann_artifact_identity &identity,
      std::string *error = nullptr);
  bool publish_standalone_rebuild(standalone_rebuild_plan *plan,
                                  std::string *error = nullptr);
  bool rollback_standalone_rebuild(standalone_rebuild_plan *plan);
  bool discard_standalone_rebuild(standalone_rebuild_plan *plan);
  bool finalize_standalone_rebuild(standalone_rebuild_plan *plan);
  void record_standalone_rebuild_tasks(
      const standalone_rebuild_plan &plan);
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
  bool set_diskann_build_mode(const std::string &index_name,
                              diskann_build_mode diskann_build_mode_value);
  bool set_diskann_search_complexity(const std::string &index_name,
                                     uint32_t diskann_search_complexity);
  bool set_diskann_search_beamwidth(const std::string &index_name,
                                    uint32_t diskann_search_beamwidth);
  bool set_diskann_pq_code_budget_size(
      const std::string &index_name, uint64_t diskann_pq_code_budget_size);
  bool set_diskann_disk_pq_dims(const std::string &index_name,
                                uint32_t diskann_disk_pq_dims);
  bool set_diskann_accelerate_build(const std::string &index_name,
                                    bool diskann_accelerate_build);
  bool set_diskann_shuffle_build(const std::string &index_name,
                                 bool diskann_shuffle_build);
  bool set_diskann_use_bfs_cache(const std::string &index_name,
                                 bool diskann_use_bfs_cache);
  bool restore_index_config(const std::string &index_name,
                            const index_config &config);
  bool set_index_consistency_mode(
      const std::string &index_name,
      index_consistency_mode consistency_mode);

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
  bool snapshot_search_runtime_loaded(const std::string &index_name,
                                      size_t query_dimension, size_t top_k,
                                      size_t query_count,
                                      search_runtime_snapshot *snapshot) const;
  bool search_runtime_snapshot_matches(
      const std::string &index_name,
      const search_runtime_snapshot &snapshot) const;
  bool finish_search_with_exact_rerank(
      const std::string &index_name, const index_config &config,
      const vector_data &query, size_t top_k,
      std::vector<search_result> candidates,
      std::vector<search_result> *results) const;
  bool finish_search_batch_with_exact_rerank(
      const std::string &index_name, const index_config &config,
      const std::vector<vector_data> &queries, size_t top_k,
      batch_search_candidates candidates,
      std::vector<std::vector<search_result>> *results) const;
  bool search_batch_loaded(
      const std::string &index_name, const std::vector<vector_data> &queries,
      size_t top_k, std::vector<std::vector<search_result>> *results) const;
  bool search_runtime_with_exact_rerank(
      const std::string &index_name, const index_config &config,
      const backend *runtime, const vector_data &query, size_t top_k,
      std::vector<search_result> *results) const;
  bool search_batch_runtime_with_exact_rerank(
      const std::string &index_name, const index_config &config,
      const backend *runtime, const std::vector<vector_data> &queries,
      size_t top_k, std::vector<std::vector<search_result>> *results) const;
  bool search_with_pending(uint64_t txn_id, const std::string &index_name,
                           const vector_data &query, size_t top_k,
                           std::vector<search_result> *results) const;
  bool search_with_pending_loaded(
      uint64_t txn_id, const std::string &index_name, const vector_data &query,
      size_t top_k, std::vector<search_result> *results) const;
  bool ensure_runtime_loaded_for_search(const std::string &index_name);
  bool runtime_loaded_for_search(const std::string &index_name) const;
  bool rebuild_runtime_from_store_for_search(const std::string &index_name);
  bool describe_index(
      const std::string &index_name, index_config *config,
      bool *supports_mutations, size_t *entry_count,
      size_t *committed_entry_count, std::string *lifecycle_state = nullptr,
      uint64_t *lifecycle_version = nullptr,
      uint32_t *last_error_code = nullptr, uint64_t *last_error_ts = nullptr,
      uint64_t *last_apply_latency_ms = nullptr,
      uint64_t *recover_fallback_count = nullptr,
      uint64_t *last_recover_fallback_ts = nullptr,
      bool *external_manifest_present = nullptr,
      uint64_t *external_manifest_generation = nullptr,
      backend_build_diagnostics *build_diagnostics = nullptr,
      index_publication_state *publication_state = nullptr) const;
  bool describe_publication_state(
      const std::string &index_name,
      index_publication_state *publication_state) const;
  bool restore_publication_state(
      const std::string &index_name,
      const index_publication_state &publication_state);
  bool snapshot_publication_states(
      std::unordered_map<std::string, index_publication_state> *states) const;
  bool restore_publication_states(
      const std::unordered_map<std::string, index_publication_state> &states);
  bool rollback_artifact_publication(const std::string &index_name);
  bool finalize_artifact_publication(const std::string &index_name);
  bool recover_artifact_publication(
      const std::string &index_name,
      const diskann_artifact_identity &identity);
  bool artifact_publication_matches(
      const std::string &index_name,
      const diskann_artifact_identity &identity) const;
  uint64_t next_index_identity() const;
  bool restore_next_index_identity(uint64_t next_index_identity);
  bool describe_build_pipeline(const std::string &index_name,
                               build_pipeline_snapshot *snapshot) const;
  bool describe_index_observability(const std::string &index_name,
                                    index_observability_state *state) const;
  bool snapshot_segment_tasks(
      const std::string &index_name,
      std::vector<vector_index_metadata_store::segment_task_row> *rows) const;
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
  size_t standalone_ingest_memory_bytes(const std::string &index_name) const;
  size_t standalone_segment_count(const std::string &index_name) const;
  size_t standalone_segment_bytes(const std::string &index_name) const;
  size_t standalone_raw_segment_count(const std::string &index_name) const;
  size_t standalone_raw_segment_bytes(const std::string &index_name) const;
  std::string standalone_build_source(const std::string &index_name) const;
  bool restore_committed_state(const committed_state &state);
  bool restore_committed_state_for_startup(const committed_state &state);
  bool direct_upsert(const std::string &index_name, uint64_t doc_id,
                     const vector_data &vector);
  bool bulk_upsert_from_reader(const std::string &index_name,
                               const bulk_load_reader &reader,
                               const bulk_load_options &options,
                               std::string *error);
  bool bulk_upsert_from_raw_files(const std::string &index_name,
                                  const std::string &vector_filename,
                                  const std::string &docid_filename,
                                  const bulk_load_options &options,
                                  uint64_t *loaded_rows, std::string *error);
  bool direct_erase(const std::string &index_name, uint64_t doc_id);
  bool replace_committed_entries(const std::string &index_name,
                                 const committed_entries &entries);
  bool replace_committed_entries_preserve_lifecycle(
      const std::string &index_name, const committed_entries &entries);
  bool install_rebuilt_index(const std::string &index_name,
                             const committed_entries &entries,
                             std::unique_ptr<backend> rebuilt_backend,
                             const diskann_artifact_identity *artifact_identity =
                                 nullptr);
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
  standalone_entry_store m_standalone_store;
  std::unordered_map<std::string, lifecycle_info> m_lifecycle_infos;
  std::unordered_map<std::string, index_publication_state> m_publication_states;
  uint64_t m_next_index_identity{1};
  std::unordered_map<std::string, build_pipeline_snapshot>
      m_build_pipeline_snapshots;
  std::unordered_map<
      std::string, std::vector<vector_index_metadata_store::segment_task_row>>
      m_segment_task_rows;
  std::unordered_map<std::string, backend_build_diagnostics>
      m_last_failed_build_diagnostics;
  std::unordered_set<std::string> m_standalone_rebuilds;
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
  bool synchronize_runtime_publication(const std::string &index_name);
  void mark_index_ready(const std::string &index_name,
                        lifecycle_info *lifecycle);
  bool build_runtime_from_current_policy(
      const std::string &index_name, const index_config &persisted_config,
      std::unique_ptr<backend> *runtime);
  bool restore_committed_state_impl(const committed_state &state,
                                    bool defer_diskann_artifacts);
  bool ensure_runtime_loaded(const std::string &index_name);
  void maybe_unload_runtime(const std::string &index_name);
  size_t diskann_exact_rerank_segment_count(
      const std::string &index_name, const index_config &config) const;
  size_t diskann_exact_rerank_candidate_top_k(
      const std::string &index_name, const index_config &config, size_t top_k,
      size_t query_count) const;
  bool exact_rerank_search_results(
      const std::string &index_name, const index_config &config,
      const vector_data &query, const std::vector<search_result> &candidates,
      size_t top_k, std::vector<search_result> *results,
      bool *reranked) const;
  bool load_exact_rerank_vectors(
      const std::string &index_name, const index_config &config,
      const std::unordered_set<uint64_t> &doc_ids, committed_entries *vectors,
      bool *complete) const;
  build_input_stats collect_build_input_stats(
      const std::string &index_name, const index_config &config) const;
  build_pipeline_decision record_build_pipeline_decision(
      const std::string &index_name, const index_config &config);
  build_pipeline_decision make_build_pipeline_decision(
      const std::string &index_name, const index_config &config,
      build_pipeline_snapshot *snapshot) const;
};

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool parse_manifest_size_for_testing(const std::string &text, size_t *value);
bool file_size_as_size_for_testing(const std::string &path, size_t *bytes);
bool copy_or_link_file_for_testing(const std::string &source,
                                   const std::string &target);
bool write_generated_docid_file_for_testing(const std::string &path,
                                            uint64_t row_count);
bool vector_payload_bytes_for_testing(size_t entry_count, size_t dimension,
                                      size_t *bytes);
std::unique_ptr<backend> make_segmented_backend_for_testing(
    index_service::index_config config,
    std::vector<std::shared_ptr<backend>> segments,
    backend_build_diagnostics diagnostics = {});
bool raw_segments_use_single_backend_for_testing(
    const index_service::index_config &config);
size_t diskann_exact_rerank_candidate_top_k_for_testing(
    size_t top_k, size_t query_count, size_t authoritative_count,
    size_t segment_count, uint32_t search_complexity,
    diskann_search_profile search_profile, size_t result_budget,
    size_t candidate_target);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index

#endif  // SQL_VECTOR_INDEX_SERVICE_INCLUDED
