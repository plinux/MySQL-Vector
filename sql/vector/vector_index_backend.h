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

#ifndef SQL_VECTOR_INDEX_BACKEND_INCLUDED
#define SQL_VECTOR_INDEX_BACKEND_INCLUDED

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_resource_budget.h"

#ifdef HAVE_FAISS
namespace faiss {
struct Index;
struct IndexIVF;
struct IndexIVFPQ;
}
#endif

namespace vector_index {

class diskann_native_state;
class hnswlib_native_state;
struct diskann_artifact_identity;

using vector_data = std::vector<float>;
using committed_entry_visitor =
    std::function<bool(uint64_t doc_id, const vector_data &vector)>;
using committed_entry_reader =
    std::function<bool(const committed_entry_visitor &visitor)>;

/** Streaming committed entries plus an optional exact cardinality hint. */
struct committed_entry_source {
  committed_entry_reader reader;
  size_t exact_row_count{0};
  bool has_exact_row_count{false};
};

struct raw_vector_segment {
  std::string vector_path;
  std::string docid_path;
  size_t row_count{0};
  size_t dimension{0};
  size_t bytes{0};
  uint64_t generation{0};
};

using raw_vector_segment_visitor =
    std::function<bool(const raw_vector_segment &segment)>;
using raw_vector_segment_reader =
    std::function<bool(const raw_vector_segment_visitor &visitor)>;

enum class metric_type { kEuclidean, kCosine, kInnerProduct };
enum class backend_mode { kMemory, kExternal };
enum class backend_provider { kNative, kFaiss, kDiskAnn, kHnswlib };
enum class external_sidecar_profile { kFaiss, kDiskAnn };
enum class diskann_build_mode { kAuto, kSerial, kOffline };
enum class index_consistency_mode { kTransactional, kStandalone };

struct vector_library_status {
  const char *library{nullptr};
  bool supported{false};
  bool offline_build_applicable{false};
  bool offline_build_supported{false};
  const char *comment{nullptr};
};

struct search_result {
  uint64_t doc_id{0};
  double distance{0.0};
};

/** Batch rerank candidates stored per query or as one shared immutable list. */
class batch_search_candidates {
 public:
  void clear();
  void set_shared(size_t query_count,
                  std::vector<search_result> candidates);
  void set_per_query(
      std::vector<std::vector<search_result>> candidates);
  void prepare_per_query(size_t query_count);

  size_t query_count() const { return m_query_count; }
  bool uses_shared_candidates() const {
    return m_query_count != 0 && m_per_query_candidates.empty();
  }
  const std::vector<search_result> *for_query(size_t query_index) const;
  std::vector<search_result> *mutable_for_query(size_t query_index);
  bool materialize(
      std::vector<std::vector<search_result>> *results) &&;

 private:
  size_t m_query_count{0};
  std::vector<search_result> m_shared_candidates;
  std::vector<std::vector<search_result>> m_per_query_candidates;
};

/** Immutable backend search options owned by one request. */
struct backend_search_options {
  uint32_t diskann_search_complexity{0};
  uint32_t diskann_search_beamwidth{0};
};

/** Last backend build/rebuild phase timings exposed for observability. */
struct backend_build_diagnostics {
  std::string runtime;
  std::string input_source;
  uint64_t row_count{0};
  uint64_t segment_count{0};
  uint64_t build_invocations{0};
  uint32_t concurrent_build_tasks{0};
  uint32_t scheduler_cpu_budget{0};
  std::string resource_probe_source;
  uint64_t resource_configured_memory_budget{0};
  uint64_t resource_memory_reserve{0};
  uint64_t resource_memory_limit{0};
  uint64_t resource_memory_current{0};
  uint64_t resource_process_rss{0};
  uint64_t resource_memory_headroom{0};
  uint64_t resource_reserved_memory{0};
  uint64_t resource_effective_memory{0};
  uint32_t resource_effective_cpu_slots{0};
  uint32_t resource_reserved_cpu_slots{0};
  uint32_t resource_active_builds{0};
  uint32_t resource_waiting_builds{0};
  uint32_t effective_build_threads{0};
  uint32_t effective_blas_threads{0};
  uint32_t raw_reader_threads{0};
  uint32_t pq_train_threads{0};
  uint32_t pq_compress_threads{0};
  uint32_t candidates_per_segment{0};
  uint32_t effective_segment_tasks{0};
  uint64_t segment_memory_estimate{0};
  uint64_t segment_memory_budget{0};
  std::string segment_parallel_reason;
  uint64_t search_fanout_segments{0};
  uint64_t search_fanout_threads{0};
  uint64_t search_worker_budget{0};
  uint64_t search_active_requests{0};
  uint64_t search_work_items{0};
  uint64_t search_global_top_k{0};
  uint64_t search_per_segment_top_k{0};
  uint64_t search_result_budget{0};
  uint64_t search_candidate_count{0};
  uint64_t search_query_count{0};
  uint64_t search_segment_min_entries{0};
  uint64_t search_segment_max_entries{0};
  uint64_t search_segment_total_entries{0};
  uint64_t search_diskann_search_list{0};
  uint64_t search_diskann_beamwidth{0};
  uint64_t search_effective_complexity{0};
  uint64_t search_total_candidate_rows{0};
  std::string search_profile;
  std::string search_profile_reason;
  bool single_index_build{false};
  uint64_t pq_chunks{0};
  uint64_t cache_nodes{0};
  uint64_t requested_disk_pq_dims{0};
  uint64_t effective_disk_pq_dims{0};
  uint64_t build_wall_ms{0};
  uint64_t manifest_ms{0};
  uint64_t offline_build_ms{0};
  uint64_t load_ms{0};
  uint64_t reader_count_ms{0};
  uint64_t training_copy_ms{0};
  uint64_t train_ms{0};
  uint64_t add_ms{0};
  uint64_t persist_ms{0};
  uint64_t training_rows{0};
  uint64_t reader_passes{0};
  std::string diskann_pq_runtime;
  std::string native_pq_runtime_selected_path;
  uint64_t native_pq_runtime_elapsed_ms{0};
  uint64_t native_pq_runtime_raw_reader_ms{0};
  uint64_t native_pq_runtime_train_ms{0};
  uint64_t native_pq_runtime_encode_ms{0};
  uint64_t native_pq_runtime_artifact_validation_ms{0};
  uint64_t native_pq_runtime_distance_calls{0};
  uint64_t native_pq_runtime_train_rows{0};
  uint64_t native_pq_runtime_compressed_rows{0};
  uint64_t native_pq_runtime_encode_block_rows{0};
  uint64_t native_pq_runtime_memory_estimate{0};
  uint64_t native_pq_runtime_memory_budget{0};
  uint32_t native_pq_runtime_effective_threads{0};
  bool native_pq_runtime_artifacts_written{false};
  bool native_pq_runtime_artifacts_consumed{false};
  bool native_pq_runtime_official_pq_used{false};
  std::string native_pq_runtime_bridge;
  std::string native_pq_runtime_centroid_scan_kernel;
  std::string native_pq_runtime_memory_adjustment;
  uint64_t native_pq_runtime_bridge_ms{0};
  uint64_t native_pq_runtime_graph_ms{0};
  uint64_t native_pq_runtime_cache_ms{0};
  std::string native_pq_runtime_artifact_validation;
  bool native_pq_runtime_validation_failed{false};
  std::string native_pq_runtime_validation_failed_doc_id;
  std::string native_pq_runtime_validation_best_doc_id;
  uint64_t native_pq_runtime_validation_result_count{0};
  std::string native_pq_runtime_validation_best_search_distance;
  std::string native_pq_runtime_validation_best_exact_distance;
  std::string native_pq_runtime_validation_self_pq_distance;
  std::string native_pq_runtime_validation_pivots_checksum;
  std::string native_pq_runtime_validation_compressed_checksum;
  std::string fallback_reason;
};

struct segment_build_input {
  raw_vector_segment segment;
  uint64_t segment_id{0};
  uint64_t generation{0};
  std::string artifact_prefix;
};

struct segment_build_result {
  uint64_t segment_id{0};
  uint64_t generation{0};
  size_t row_count{0};
  size_t payload_size{0};
  std::string vector_path;
  std::string docid_path;
  std::string artifact_prefix;
  bool ready{false};
  backend_build_diagnostics diagnostics;
};

/**
  Abstract backend for ANN/exact vector search implementations.
*/
class backend {
 public:
  virtual ~backend() = default;

  /**
    Per-runtime guard used when registry-level code snapshots a backend handle
    and executes search outside the global registry mutex.
  */
  std::shared_mutex &runtime_mutex() const { return m_runtime_mutex; }

  /**
    insert or replace a vector by document id.

    @retval true Succeeded.
    @retval false Invalid input or backend error.
  */
  virtual bool upsert(uint64_t doc_id, const vector_data &vector) = 0;

  /**
    remove a vector by document id.

    @retval true Succeeded.
    @retval false backend error.
  */
  virtual bool erase(uint64_t doc_id) = 0;

  /**
    search nearest neighbors for a query vector.

    Returned distances follow ascending order semantics.

    @retval true Succeeded.
    @retval false Invalid input or backend error.
  */
  virtual bool search(const vector_data &query, size_t top_k,
                      std::vector<search_result> *results) const = 0;

  /** Search with immutable options derived for this request. */
  virtual bool search_with_options(
      const vector_data &query, size_t top_k,
      const backend_search_options &options,
      std::vector<search_result> *results) const;

  /**
    Search a batch of independent query vectors.

    The default implementation preserves existing single-query semantics and
    lets backends override only when their read path is safe to batch or
    parallelize internally.
  */
  virtual bool search_batch(
      const std::vector<vector_data> &queries, size_t top_k,
      std::vector<std::vector<search_result>> *results) const;

  /** Batch search with immutable options derived for this request. */
  virtual bool search_batch_with_options(
      const std::vector<vector_data> &queries, size_t top_k,
      const backend_search_options &options,
      std::vector<std::vector<search_result>> *results) const;

  /**
    Search candidates for a following exact rerank step.

    @param query Query vector.
    @param top_k User requested final topK. Segmented backends use this value to
      size per-segment ANN work.
    @param candidate_top_k Maximum candidates returned to the exact reranker.
    @param results Search results before exact rerank.

    @retval true Succeeded.
    @retval false Invalid input or backend error.
  */
  virtual bool search_for_rerank(
      const vector_data &query, size_t top_k, size_t candidate_top_k,
      std::vector<search_result> *results) const;

  /**
    Batch form of search_for_rerank().
  */
  virtual bool search_batch_for_rerank(
      const std::vector<vector_data> &queries, size_t top_k,
      size_t candidate_top_k,
      batch_search_candidates *results) const;

  /**
    Collect all document ids currently addressable by this backend.

    This is an optional support hook for exact-rerank callers that need an
    authoritative candidate list when ANN search is asked to cover an entire
    segment. Backends that cannot enumerate serving doc ids return false.
  */
  virtual bool collect_doc_ids(std::vector<uint64_t> *doc_ids) const;

  /**
    load serving state from committed entries snapshot.

    Default behavior replays entries through mutation APIs. Read-only backends
    should override this to provide provider-specific bulk-load behavior.

    @retval true Succeeded.
    @retval false Invalid input or backend error.
  */
  virtual bool load_committed_entries(
      const std::unordered_map<uint64_t, vector_data> &entries);

  /**
    Load serving state from a committed-entry reader.

    The default implementation materializes a snapshot and then delegates to
    load_committed_entries(). Streaming backends can override this method to
    consume truth-store rows without building a full map in SQL memory.
  */
  virtual bool load_committed_entries_from_reader(
      const committed_entry_reader &reader);

  /**
    Load serving state from committed entries with optional exact cardinality.

    Backends may use the exact row count to admit build resources before the
    reader performs any I/O. The default implementation delegates to the
    reader API.
  */
  virtual bool load_committed_entries_from_source(
      const committed_entry_source &source);

  /**
    Rebuild serving state from committed entries snapshot.

    Default behavior reuses load_committed_entries(). Backends that distinguish
    lightweight committed-state load from explicit serving-structure rebuild
    should override this.

    @retval true Succeeded.
    @retval false Invalid input or backend error.
  */
  virtual bool rebuild_from_committed_entries(
      const std::unordered_map<uint64_t, vector_data> &entries) {
    return load_committed_entries(entries);
  }

  /**
    Rebuild serving state from a committed-entry reader.
  */
  virtual bool rebuild_from_committed_entries_from_reader(
      const committed_entry_reader &reader);

  /**
    Rebuild serving state from a committed-entry source.

    Backends may use an exact row count to avoid a separate count pass. The
    default implementation ignores the hint and delegates to the reader API.
  */
  virtual bool rebuild_from_committed_entry_source(
      const committed_entry_source &source);

  /**
    Rebuild serving state from standalone raw vector segments.

    The default implementation streams rows from each segment and delegates to
    rebuild_from_committed_entries_from_reader(). Backends with native file or
    manifest build APIs should override this to avoid row materialization.
  */
  virtual bool rebuild_from_raw_segments(const raw_vector_segment_reader &reader);

  /**
    Build one segmented-pipeline serving handle from a raw segment.

    The default implementation is intentionally conservative: it rebuilds this
    backend from the single segment through rebuild_from_raw_segments() and
    reports a ready result. Segmented executors should call this on a fresh
    backend instance per segment until a provider overrides it with a true
    side-effect-free artifact build.
  */
  virtual bool build_segment_from_raw(const segment_build_input &input,
                                      segment_build_result *result);

  /**
    Load a previously built segment handle.

    Providers with durable segment artifacts should override this. The base
    implementation accepts results produced by the default
    build_segment_from_raw() path.
  */
  virtual bool load_segment_handle(const segment_build_result &result);

  /**
    Recover backend serving state from persisted metadata or snapshots.

    Backends without provider-specific persisted state default to a no-op
    success path and may rebuild from committed entries through
    recover_committed_entries().
  */
  virtual bool recover() { return true; }

  /**
    recover serving state while reconciling committed entries.

    Default behavior recovers provider state and then reloads committed entries.
    Backends with a distinct persisted serving-state format may override this to
    avoid expensive replay paths during recovery.

    @retval true Succeeded.
    @retval false Invalid input or backend error.
  */
  virtual bool recover_committed_entries(
      const std::unordered_map<uint64_t, vector_data> &entries) {
    if (!recover()) return false;
    return load_committed_entries(entries);
  }

  /**
    Recover serving state while consuming committed entries through a reader.
  */
  virtual bool recover_committed_entries_from_reader(
      const committed_entry_reader &reader);

  /**
    Recover serving state from committed entries with optional cardinality.

    The default implementation preserves recover-then-load ordering while
    allowing load_committed_entries_from_source() overrides to perform early
    resource admission.
  */
  virtual bool recover_committed_entries_from_source(
      const committed_entry_source &source);

  /**
    Whether the last recover() call had to use a fallback path.

    Default is false for backends without explicit fallback semantics.
  */
  virtual bool last_recover_used_fallback() const { return false; }

  /**
    Return currently visible entry count from the serving backend.

    Backends that cannot expose an accurate count may return 0.
  */
  virtual size_t entry_count() const { return 0; }

  virtual size_t dimension() const = 0;
  virtual metric_type metric() const = 0;
  virtual backend_mode mode() const = 0;
  virtual backend_provider provider() const = 0;
  virtual std::string backend_variant() const { return ""; }
  virtual backend_build_diagnostics build_diagnostics() const { return {}; }
  virtual bool set_search_ef(uint32_t search_ef [[maybe_unused]]) { return false; }
  virtual uint32_t search_ef() const { return 0; }
  virtual bool set_hnsw_build_params(uint32_t hnsw_m [[maybe_unused]],
                                     uint32_t hnsw_ef_construction
                                         [[maybe_unused]]) {
    return false;
  }
  virtual uint32_t hnsw_m() const { return 0; }
  virtual uint32_t hnsw_ef_construction() const { return 0; }
  virtual bool set_hnsw_build_threads(
      uint32_t hnsw_build_threads [[maybe_unused]]) {
    return hnsw_build_threads == 0;
  }
  virtual uint32_t hnsw_build_threads() const { return 0; }
  virtual bool set_faiss_ivf_params(uint32_t faiss_nlist [[maybe_unused]],
                                    uint32_t faiss_nprobe [[maybe_unused]]) {
    return false;
  }
  virtual uint32_t faiss_nlist() const { return 0; }
  virtual uint32_t faiss_nprobe() const { return 0; }
  virtual bool set_faiss_build_threads(
      uint32_t faiss_build_threads [[maybe_unused]]) {
    return faiss_build_threads == 0;
  }
  virtual uint32_t faiss_build_threads() const { return 0; }
  virtual bool set_faiss_ivf_pq_params(uint32_t faiss_nlist [[maybe_unused]],
                                   uint32_t faiss_nprobe [[maybe_unused]],
                                   uint32_t faiss_pq_m [[maybe_unused]],
                                   uint32_t faiss_pq_bits [[maybe_unused]]) {
    return false;
  }
  virtual uint32_t faiss_pq_m() const { return 0; }
  virtual uint32_t faiss_pq_bits() const { return 0; }
  virtual bool set_diskann_build_params(uint32_t diskann_max_degree
                                         [[maybe_unused]],
                                     uint32_t diskann_build_complexity
                                         [[maybe_unused]],
                                     uint32_t diskann_build_threads
                                         [[maybe_unused]]) {
    return false;
  }
  virtual bool set_diskann_build_threads(
      uint32_t diskann_build_threads [[maybe_unused]]) {
    return diskann_build_threads == 0;
  }
  virtual bool set_diskann_build_blas_threads(
      uint32_t diskann_build_blas_threads [[maybe_unused]]) {
    return diskann_build_blas_threads == 0;
  }
  virtual uint32_t diskann_max_degree() const { return 0; }
  virtual uint32_t diskann_build_complexity() const { return 0; }
  virtual uint32_t diskann_build_threads() const { return 0; }
  virtual uint32_t diskann_build_blas_threads() const { return 0; }
  virtual bool set_diskann_build_mode(
      diskann_build_mode diskann_build_mode_value) {
    return diskann_build_mode_value == diskann_build_mode::kAuto;
  }
  virtual diskann_build_mode diskann_build_mode_value() const {
    return diskann_build_mode::kAuto;
  }
  virtual bool set_diskann_search_complexity(
      uint32_t diskann_search_complexity [[maybe_unused]]) {
    return false;
  }
  virtual uint32_t diskann_search_complexity() const { return 0; }
  virtual bool set_diskann_search_beamwidth(
      uint32_t diskann_search_beamwidth [[maybe_unused]]) {
    return false;
  }
  virtual uint32_t diskann_search_beamwidth() const { return 0; }
  virtual bool set_diskann_pq_code_budget_size(
      uint64_t diskann_pq_code_budget_size [[maybe_unused]]) {
    return false;
  }
  virtual uint64_t diskann_pq_code_budget_size() const { return 0; }
  virtual bool set_diskann_disk_pq_dims(
      uint32_t diskann_disk_pq_dims [[maybe_unused]]) {
    return diskann_disk_pq_dims == 0;
  }
  virtual uint32_t diskann_disk_pq_dims() const { return 0; }
  virtual bool set_diskann_accelerate_build(
      bool diskann_accelerate_build [[maybe_unused]]) {
    return !diskann_accelerate_build;
  }
  virtual bool diskann_accelerate_build() const { return false; }
  virtual bool set_diskann_shuffle_build(
      bool diskann_shuffle_build [[maybe_unused]]) {
    return !diskann_shuffle_build;
  }
  virtual bool diskann_shuffle_build() const { return false; }
  virtual bool set_diskann_use_bfs_cache(
      bool diskann_use_bfs_cache [[maybe_unused]]) {
    return !diskann_use_bfs_cache;
  }
  virtual bool diskann_use_bfs_cache() const { return false; }
  virtual uint32_t diskann_offline_search_threads() const { return 0; }
  virtual uint32_t diskann_search_io_limit() const { return 0; }
  virtual uint32_t diskann_cache_nodes() const { return 0; }

  /**
    Whether transactional stage/commit can apply mutations to this backend.

    Writability is provider- and mode-specific. Some EXTERNAL providers are
    rebuilt from committed truth rows instead of accepting point mutations.
  */
  virtual bool supports_mutations() const = 0;

  /**
    Whether the current serving handle can apply point mutations in place.

    This is narrower than supports_mutations(): an index may support
    transactional DML while an immutable serving generation requires the
    service to advance truth and rebuild the runtime before the next search.
  */
  virtual bool supports_in_place_mutations() const {
    return supports_mutations();
  }

  /**
    EXTERNAL-sidecar manifest presence for backend observability.

    Non-sidecar backends return false by default.
  */
  virtual bool external_manifest_present() const { return false; }

  /**
    EXTERNAL-sidecar manifest generation for backend observability.

    Non-sidecar backends return 0 by default.
  */
  virtual uint64_t external_manifest_generation() const { return 0; }

  /** Keep the next rebuild artifact private until registry publication. */
  virtual bool defer_artifact_publication() { return true; }

  /** Whether this backend owns a rebuilt artifact awaiting publication. */
  virtual bool has_pending_artifact_publication() const { return false; }

  /** Bind a rebuilt artifact to the durable registry publication tuple. */
  virtual bool prepare_artifact_publication(
      const diskann_artifact_identity &identity [[maybe_unused]]) {
    return true;
  }

  /** Install a prepared artifact while retaining rollback state. */
  virtual bool publish_artifact() { return true; }

  /** Restore the generation that preceded the current publication attempt. */
  virtual bool rollback_artifact() { return true; }

  /** Discard rollback state after registry metadata becomes durable. */
  virtual bool finalize_artifact() { return true; }

  /** Recover and validate the artifact selected by durable registry metadata. */
  virtual bool recover_artifact_publication(
      const diskann_artifact_identity &identity [[maybe_unused]]) {
    return true;
  }

  /** Verify that the serving artifact matches one publication tuple. */
  virtual bool artifact_publication_matches(
      const diskann_artifact_identity &identity [[maybe_unused]]) const {
    return true;
  }

 protected:
  /** Reserve process-wide resources for one backend build phase. */
  bool acquire_build_resources(size_t row_count, uint32_t memory_multiplier,
                               uint64_t fixed_memory,
                               uint64_t per_build_memory_limit,
                               uint32_t requested_threads,
                               build_resource_lease *lease) const;

  /** Add the active lease and process envelope to build diagnostics. */
  void record_build_resource_diagnostics(
      const build_resource_lease &lease,
      backend_build_diagnostics *diagnostics) const;

 private:
  mutable std::shared_mutex m_runtime_mutex;
};

/**
  Exact in-memory backend used by MEMORY mode and debug/native fallback paths.
*/
class memory_backend final : public backend {
 public:
  memory_backend(size_t dimension, metric_type metric);

  bool upsert(uint64_t doc_id, const vector_data &vector) override;
  bool erase(uint64_t doc_id) override;
  bool search(const vector_data &query, size_t top_k,
              std::vector<search_result> *results) const override;
  void reset();
  bool contains(uint64_t doc_id) const;
  bool snapshot_entries(
      std::unordered_map<uint64_t, vector_data> *entries) const;

  size_t entry_count() const override { return m_entries.size(); }
  size_t dimension() const override { return m_dimension; }
  metric_type metric() const override { return m_metric; }
  backend_mode mode() const override { return backend_mode::kMemory; }
  backend_provider provider() const override { return backend_provider::kNative; }
  bool supports_mutations() const override { return true; }

 private:
  size_t m_dimension{0};
  metric_type m_metric{metric_type::kEuclidean};
  std::unordered_map<uint64_t, vector_data> m_entries;
};

/**
  Native EXTERNAL backend used as an exact in-process fallback.

  This backend keeps vectors in-process while exposing EXTERNAL mode semantics
  to the SQL layer when a provider-specific external runtime is unavailable or
  intentionally disabled for tests.
*/
class external_backend final : public backend {
 public:
  external_backend(size_t dimension, metric_type metric)
      : m_dimension(dimension), m_metric(metric) {}

  bool upsert(uint64_t doc_id, const vector_data &vector) override;
  bool erase(uint64_t doc_id) override;
  bool search(const vector_data &query, size_t top_k,
              std::vector<search_result> *results) const override;
  void reset();

  size_t entry_count() const override { return m_entries.size(); }
  size_t dimension() const override { return m_dimension; }
  metric_type metric() const override { return m_metric; }
  backend_mode mode() const override { return backend_mode::kExternal; }
  backend_provider provider() const override { return backend_provider::kNative; }
  bool supports_mutations() const override { return true; }

 private:
  size_t m_dimension{0};
  metric_type m_metric{metric_type::kEuclidean};
  std::unordered_map<uint64_t, vector_data> m_entries;
};

/**
  Faiss backend integration point.

  MEMORY mode uses the in-process exact fallback. EXTERNAL mode can persist
  serving snapshots through a sidecar manifest + snapshot layout.
*/
class faiss_backend final : public backend {
 public:
  faiss_backend(size_t dimension, metric_type metric, backend_mode mode,
               const std::string &index_name = "",
               external_sidecar_profile sidecar_profile =
                   external_sidecar_profile::kFaiss);
  ~faiss_backend() override;

  bool upsert(uint64_t doc_id, const vector_data &vector) override;
  bool erase(uint64_t doc_id) override;
  bool search(const vector_data &query, size_t top_k,
              std::vector<search_result> *results) const override;
  bool search_batch(const std::vector<vector_data> &queries, size_t top_k,
                    std::vector<std::vector<search_result>> *results)
      const override;
  bool load_committed_entries(
      const std::unordered_map<uint64_t, vector_data> &entries) override;
  bool load_committed_entries_from_source(
      const committed_entry_source &source) override;
  bool recover_committed_entries_from_source(
      const committed_entry_source &source) override;
  bool rebuild_from_committed_entries_from_reader(
      const committed_entry_reader &reader) override;
  bool rebuild_from_committed_entry_source(
      const committed_entry_source &source) override;
  bool rebuild_from_raw_segments(
      const raw_vector_segment_reader &reader) override;
  bool recover() override;
  bool last_recover_used_fallback() const override {
    return m_last_recover_used_fallback != 0;
  }

  size_t entry_count() const override {
    if (m_mode == backend_mode::kExternal) {
#ifdef HAVE_FAISS
      if (m_sidecar_profile == external_sidecar_profile::kFaiss &&
          (m_faiss_entry_count != 0 || m_external_snapshot_entries.empty()))
        return m_faiss_entry_count;
#endif
      return m_external_snapshot_entries.size();
    }
    if (m_faiss_entry_count != 0) return m_faiss_entry_count;
    if (m_mode == backend_mode::kMemory) return m_memory_fallback.entry_count();
    return m_external_fallback.entry_count();
  }
  size_t dimension() const override { return m_dimension; }
  metric_type metric() const override { return m_metric; }
  backend_mode mode() const override { return m_mode; }
  backend_provider provider() const override { return backend_provider::kFaiss; }
  std::string backend_variant() const override;
  bool set_search_ef(uint32_t search_ef) override;
  uint32_t search_ef() const override;
  bool set_hnsw_build_params(uint32_t hnsw_m,
                          uint32_t hnsw_ef_construction) override;
  uint32_t hnsw_m() const override;
  uint32_t hnsw_ef_construction() const override;
  bool set_faiss_ivf_params(uint32_t faiss_nlist,
                         uint32_t faiss_nprobe) override;
  uint32_t faiss_nlist() const override;
  uint32_t faiss_nprobe() const override;
  bool set_faiss_build_threads(uint32_t faiss_build_threads) override;
  uint32_t faiss_build_threads() const override;
  bool set_faiss_ivf_pq_params(uint32_t faiss_nlist, uint32_t faiss_nprobe,
                           uint32_t faiss_pq_m,
                           uint32_t faiss_pq_bits) override;
  uint32_t faiss_pq_m() const override;
  uint32_t faiss_pq_bits() const override;
  backend_build_diagnostics build_diagnostics() const override;
  bool supports_mutations() const override { return true; }
  bool external_manifest_present() const override;
  uint64_t external_manifest_generation() const override;
  const std::unordered_map<uint64_t, vector_data> &external_snapshot_entries() const {
    return m_external_snapshot_entries;
  }
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  size_t faiss_last_training_count_for_testing() const {
    return m_faiss_last_training_count;
  }
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

 private:
  size_t m_dimension{0};
  metric_type m_metric{metric_type::kEuclidean};
  backend_mode m_mode{backend_mode::kMemory};
  external_sidecar_profile m_sidecar_profile{external_sidecar_profile::kFaiss};
  std::string m_index_name;
  std::string m_external_snapshot_directory;
  std::string m_external_manifest_path;
  memory_backend m_memory_fallback;
  external_backend m_external_fallback;
  std::unordered_map<uint64_t, vector_data> m_external_snapshot_entries;
#ifdef HAVE_FAISS
  std::unique_ptr<faiss::Index> m_faiss_index;
#endif
  size_t m_faiss_entry_count{0};
  uint32_t m_last_recover_used_fallback{0};
  bool m_external_manifest_present{false};
  uint64_t m_external_manifest_generation{0};
  uint32_t m_search_ef{64};
  uint32_t m_hnsw_m{32};
  uint32_t m_hnsw_ef_construction{40};
  uint32_t m_faiss_nlist{0};
  uint32_t m_faiss_nprobe{0};
  uint32_t m_faiss_pq_m{0};
  uint32_t m_faiss_pq_bits{0};
  uint32_t m_faiss_build_threads{0};
  size_t m_faiss_last_training_count{0};
  bool m_keep_loaded_external_index{true};
  backend_build_diagnostics m_last_build_diagnostics;

  bool initialize_faiss_index(bool use_ivfpq = true,
                              uint32_t effective_nlist = 0);
  bool train_faiss_ivf_index(const std::vector<float> &training_data,
                             size_t training_rows, uint64_t *train_ms);
  bool prepare_faiss_index_for_recovery();
  void record_build_diagnostics(const char *input_source, size_t row_count,
                                size_t effective_threads,
                                size_t segment_count = 1);
  void record_build_phase_diagnostics(
      const backend_build_diagnostics &diagnostics);
  bool rebuild_external_faiss_index(
      const std::unordered_map<uint64_t, vector_data> &entries);
  bool rebuild_external_faiss_index_from_reader(
      const committed_entry_source &source);
  bool rebuild_external_faiss_index_from_raw_segments(
      const std::vector<raw_vector_segment> &segments, size_t total_rows);
  void configure_rebuild_candidate(faiss_backend *candidate) const;
  bool persist_and_install_rebuild_candidate(faiss_backend *candidate);
  bool apply_faiss_mutation(uint64_t doc_id, const vector_data *vector);
  bool search_with_faiss(const vector_data &query, size_t top_k,
                       std::vector<search_result> *results) const;
  bool search_batch_with_faiss(
      const std::vector<vector_data> &queries, size_t top_k,
      std::vector<std::vector<search_result>> *results) const;
  bool apply_faiss_ivf_nprobe(uint32_t faiss_nprobe);
  bool search_external_snapshot_entries(
      const vector_data &query, size_t top_k,
      std::vector<search_result> *results) const;
  bool persist_faiss_index_file(const std::string &path);
  bool recover_faiss_index_file(const std::string &path);
  vector_data normalize_for_faiss(const vector_data &vector) const;
  bool persist_external_snapshot();
  bool recover_external_snapshot();
  bool recover_external_snapshot_file(const std::string &path);
  bool materialize_external_entries(
      std::unordered_map<uint64_t, vector_data> *entries) const;
  bool materialize_external_entries_from_faiss(
      std::unordered_map<uint64_t, vector_data> *entries) const;
  bool load_external_manifest_generation(uint64_t *generation, bool *exists) const;
  bool save_external_manifest_generation(uint64_t generation);
  std::string external_snapshot_path_for_generation(uint64_t generation) const;
  bool remove_external_generated_snapshots() const;
  void maybe_clear_external_snapshot_entries();
  void maybe_release_external_serving_index();
};

/**
  DiskANN backend integration point.

  Native DiskANN serving state is rebuildable from committed truth rows. The
  sidecar adapter remains only as a fallback when the native library is absent
  or unusable.
*/
class diskann_backend final : public backend {
 public:
  diskann_backend(size_t dimension, metric_type metric, backend_mode mode,
                 const std::string &index_name = "");
  ~diskann_backend() override;

  bool upsert(uint64_t doc_id, const vector_data &vector) override;
  bool erase(uint64_t doc_id) override;
  bool search(const vector_data &query, size_t top_k,
              std::vector<search_result> *results) const override;
  bool search_with_options(const vector_data &query, size_t top_k,
                           const backend_search_options &options,
                           std::vector<search_result> *results) const override;
  bool search_batch(const std::vector<vector_data> &queries, size_t top_k,
                    std::vector<std::vector<search_result>> *results) const
      override;
  bool search_batch_with_options(
      const std::vector<vector_data> &queries, size_t top_k,
      const backend_search_options &options,
      std::vector<std::vector<search_result>> *results) const override;
  bool collect_doc_ids(std::vector<uint64_t> *doc_ids) const override;
  bool load_committed_entries(
      const std::unordered_map<uint64_t, vector_data> &entries) override;
  bool rebuild_from_committed_entries_from_reader(
      const committed_entry_reader &reader) override;
  bool rebuild_from_raw_segments(
      const raw_vector_segment_reader &reader) override;
  bool rebuild_from_committed_entries(
      const std::unordered_map<uint64_t, vector_data> &entries) override;
  bool recover() override;
  bool recover_committed_entries(
      const std::unordered_map<uint64_t, vector_data> &entries) override;
  bool last_recover_used_fallback() const override;

  size_t entry_count() const override;
  size_t dimension() const override { return m_dimension; }
  metric_type metric() const override { return m_metric; }
  backend_mode mode() const override { return m_mode; }
  backend_provider provider() const override { return backend_provider::kDiskAnn; }
  std::string backend_variant() const override;
  backend_build_diagnostics build_diagnostics() const override;
  bool set_diskann_build_params(uint32_t diskann_max_degree,
                             uint32_t diskann_build_complexity,
                             uint32_t diskann_build_threads) override;
  bool set_diskann_build_threads(uint32_t diskann_build_threads) override;
  bool set_diskann_build_blas_threads(
      uint32_t diskann_build_blas_threads) override;
  bool set_diskann_build_mode(
      diskann_build_mode diskann_build_mode) override;
  uint32_t diskann_max_degree() const override;
  uint32_t diskann_build_complexity() const override;
  uint32_t diskann_build_threads() const override;
  uint32_t diskann_build_blas_threads() const override;
  diskann_build_mode diskann_build_mode_value() const override;
  bool set_diskann_search_complexity(uint32_t diskann_search_complexity) override;
  uint32_t diskann_search_complexity() const override;
  bool set_diskann_search_beamwidth(uint32_t diskann_search_beamwidth) override;
  uint32_t diskann_search_beamwidth() const override;
  bool set_diskann_pq_code_budget_size(
      uint64_t diskann_pq_code_budget_size) override;
  uint64_t diskann_pq_code_budget_size() const override;
  bool set_diskann_disk_pq_dims(uint32_t diskann_disk_pq_dims) override;
  uint32_t diskann_disk_pq_dims() const override;
  bool set_diskann_accelerate_build(bool diskann_accelerate_build) override;
  bool diskann_accelerate_build() const override;
  bool set_diskann_shuffle_build(bool diskann_shuffle_build) override;
  bool diskann_shuffle_build() const override;
  bool set_diskann_use_bfs_cache(bool diskann_use_bfs_cache) override;
  bool diskann_use_bfs_cache() const override;
  uint32_t diskann_offline_search_threads() const override;
  uint32_t diskann_search_io_limit() const override;
  uint32_t diskann_cache_nodes() const override;
  bool supports_mutations() const override { return true; }
  bool supports_in_place_mutations() const override;
  bool external_manifest_present() const override;
  uint64_t external_manifest_generation() const override;
  bool defer_artifact_publication() override;
  bool has_pending_artifact_publication() const override;
  bool prepare_artifact_publication(
      const diskann_artifact_identity &identity) override;
  bool publish_artifact() override;
  bool rollback_artifact() override;
  bool finalize_artifact() override;
  bool recover_artifact_publication(
      const diskann_artifact_identity &identity) override;
  bool artifact_publication_matches(
      const diskann_artifact_identity &identity) const override;

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  void clear_committed_snapshot_for_testing();
  bool native_search_batch_for_testing(
      const std::vector<vector_data> &queries, size_t begin, size_t end,
      size_t top_k, std::vector<std::vector<search_result>> *results) const;
#endif

 private:
  size_t m_dimension{0};
  metric_type m_metric{metric_type::kEuclidean};
  backend_mode m_mode{backend_mode::kExternal};
  std::string m_index_name;
  std::unordered_map<uint64_t, vector_data> m_entries;
  std::unique_ptr<diskann_native_state> m_native_state;
  faiss_backend m_external_adapter;
  uint32_t m_diskann_max_degree{
      vector_index::k_default_diskann_max_degree};
  uint32_t m_diskann_build_complexity{
      vector_index::k_default_diskann_build_complexity};
  uint32_t m_diskann_build_threads{0};
  uint32_t m_diskann_build_blas_threads{1};
  diskann_build_mode m_diskann_build_mode{diskann_build_mode::kAuto};
  uint32_t m_diskann_search_complexity{
      vector_index::k_default_diskann_search_complexity};
  uint32_t m_diskann_search_beamwidth{
      vector_index::k_default_diskann_search_beamwidth};
  uint64_t m_diskann_pq_code_budget_size{0};
  double m_diskann_pq_code_budget_ratio{
      vector_index::k_default_diskann_pq_code_budget_ratio};
  uint32_t m_diskann_disk_pq_dims{0};
  bool m_diskann_accelerate_build{false};
  bool m_diskann_shuffle_build{false};
  bool m_diskann_use_bfs_cache{false};
  uint32_t m_diskann_offline_search_threads{1};
  uint32_t m_diskann_search_io_limit{0};
  uint32_t m_diskann_cache_nodes{0};
  uint64_t m_diskann_search_cache_size{0};
  double m_diskann_search_cache_ratio{0.0};
  bool m_native_runtime_enabled{false};
  bool m_external_adapter_active{true};
  bool m_defer_artifact_publication{false};
  size_t m_entry_count{0};
  bool m_entries_complete{true};
  bool m_streaming_doc_ids_loaded{false};
  std::vector<uint64_t> m_streaming_base_doc_ids;
  std::unordered_set<uint64_t> m_streaming_added_doc_ids;
  std::unordered_set<uint64_t> m_streaming_removed_doc_ids;
  backend_build_diagnostics m_last_build_diagnostics;

  std::unique_ptr<diskann_native_state> create_native_state(
      uint64_t build_memory_size = 0) const;
  bool configure_native_state_for_serving(diskann_native_state *state);
  void install_native_state(std::unique_ptr<diskann_native_state> state,
                            size_t entry_count, bool entries_complete);
  bool activate_empty_mutable_sidecar();
  void reset_streaming_doc_id_tracking();
  bool prepare_streaming_doc_id_tracking();
  bool streaming_doc_id_exists(uint64_t doc_id) const;
  void record_streaming_upsert(uint64_t doc_id, bool existed);
  void record_streaming_erase(uint64_t doc_id);
  void capture_native_build_diagnostics();
  void clear_build_diagnostics();
};

/**
  hnswlib backend integration point.

  MEMORY mode keeps hnswlib as the resident serving index. The exact-memory
  fallback is allocated only when native hnswlib is unavailable.
*/
class hnswlib_backend final : public backend {
 public:
  hnswlib_backend(size_t dimension, metric_type metric, backend_mode mode);
  ~hnswlib_backend() override;

  bool upsert(uint64_t doc_id, const vector_data &vector) override;
  bool erase(uint64_t doc_id) override;
  bool search(const vector_data &query, size_t top_k,
              std::vector<search_result> *results) const override;
  bool search_batch(const std::vector<vector_data> &queries, size_t top_k,
                    std::vector<std::vector<search_result>> *results)
      const override;

  size_t entry_count() const override;
  size_t dimension() const override { return m_dimension; }
  metric_type metric() const override { return m_metric; }
  backend_mode mode() const override { return m_mode; }
  backend_provider provider() const override { return backend_provider::kHnswlib; }
  std::string backend_variant() const override;
  bool set_search_ef(uint32_t search_ef) override;
  uint32_t search_ef() const override;
  bool set_hnsw_build_params(uint32_t hnsw_m,
                             uint32_t hnsw_ef_construction) override;
  uint32_t hnsw_m() const override;
  uint32_t hnsw_ef_construction() const override;
  bool set_hnsw_build_threads(uint32_t hnsw_build_threads) override;
  uint32_t hnsw_build_threads() const override;
  bool rebuild_from_committed_entries(
      const std::unordered_map<uint64_t, vector_data> &entries) override;
  bool rebuild_from_committed_entries_from_reader(
      const committed_entry_reader &reader) override;
  bool rebuild_from_committed_entry_source(
      const committed_entry_source &source) override;
  bool load_committed_entries_from_source(
      const committed_entry_source &source) override;
  bool recover_committed_entries_from_source(
      const committed_entry_source &source) override;
  bool rebuild_from_raw_segments(
      const raw_vector_segment_reader &reader) override;
  backend_build_diagnostics build_diagnostics() const override;
  bool supports_mutations() const override {
    return m_mode == backend_mode::kMemory;
  }
  bool hnsw_native_available_for_testing() const;
  bool hnsw_exact_fallback_active_for_testing() const;
  size_t hnsw_exact_fallback_entry_count_for_testing() const;

 private:
  bool native_available() const;
  memory_backend *ensure_exact_fallback();
  bool rebuild_native_from_entries(
      const std::unordered_map<uint64_t, vector_data> &entries);
  bool build_memory_fallback_from_entries(
      const std::unordered_map<uint64_t, vector_data> &entries,
      memory_backend *fallback) const;
  bool build_state_from_reader(const committed_entry_reader &reader,
                               std::unique_ptr<memory_backend> *fallback,
                               std::unique_ptr<hnswlib_native_state> *native,
                               size_t *entry_count) const;
  bool build_state_from_raw_segments(
      const std::vector<raw_vector_segment> &segments, size_t total_rows,
      std::unique_ptr<hnswlib_native_state> *native, size_t *entry_count) const;
  bool rebuild_memory_fallback_from_entries(
      const std::unordered_map<uint64_t, vector_data> &entries);
  void record_build_diagnostics(const char *input_source, size_t row_count,
                                size_t effective_threads,
                                size_t segment_count = 1,
                                size_t reader_passes = 0);
  void clear_build_diagnostics();

  size_t m_dimension{0};
  metric_type m_metric{metric_type::kEuclidean};
  backend_mode m_mode{backend_mode::kMemory};
  std::unique_ptr<hnswlib_native_state> m_native_state;
  std::unique_ptr<memory_backend> m_exact_fallback;
  uint32_t m_hnsw_m{16};
  uint32_t m_hnsw_ef_construction{200};
  uint32_t m_hnsw_build_threads{0};
  backend_build_diagnostics m_last_build_diagnostics;
};

/**
  Create a backend instance from mode/provider configuration.

  @retval nullptr Combination is unsupported.
*/
std::unique_ptr<backend> create_backend(size_t dimension, metric_type metric,
                                       backend_mode mode,
                                       backend_provider provider,
                                       const std::string &index_name = "");

std::unique_ptr<backend> create_backend(size_t dimension, metric_type metric,
                                       backend_mode mode,
                                       backend_provider provider,
                                       index_consistency_mode consistency_mode,
                                       const std::string &index_name = "");

/**
  Return compiled vector-library support rows for INFORMATION_SCHEMA.
*/
const vector_library_status *vector_library_statuses(size_t *count);

/**
  Check whether a provider can be selected by CREATE VECTOR INDEX.

  The native provider is an internal exact/debug implementation. It is only
  selectable in debug builds where NDEBUG is not defined and at least one real
  vector-search library is compiled, so it cannot mask all-libraries-disabled
  binaries.
*/
bool backend_provider_supported(backend_provider provider);

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
/** Return whether unit-test processes may construct the exact native provider. */
bool native_provider_supported_for_testing();

/** Enable or disable the exact native provider for unit-test processes. */
void set_native_provider_supported_for_testing(bool supported);
#endif

/**
  Return the default provider for an omitted provider option.

  The default is controlled by the global vector_default_library sysvar. Empty
  or unsupported values make provider-omitted index creation fail.
*/
bool default_backend_provider(backend_provider *provider);

/**
  Return the default mode for an omitted mode option after provider resolution.
*/
bool default_backend_mode_for_provider(backend_provider provider,
                                       backend_mode *mode);

/**
  Parse metric option text into enum value.

  Parsing is case-insensitive and ignores leading/trailing spaces.
*/
bool parse_metric(const std::string &value, metric_type *metric);

/**
  Parse backend mode option text into enum value.

  Parsing is case-insensitive and ignores leading/trailing spaces.
*/
bool parse_backend_mode(const std::string &value, backend_mode *mode);

/**
  Parse backend provider option text into enum value.

  Parsing is case-insensitive and ignores leading/trailing spaces.
*/
bool parse_backend_provider(const std::string &value, backend_provider *provider);

/**
  Parse DiskANN build mode option text into enum value.

  AUTO prefers the offline adapter when it is available and otherwise uses the
  official Garnet row-by-row insert API. SERIAL forces row-by-row Garnet insert.
  OFFLINE uses the MySQL DiskANN offline adapter for official disk-index builds.
*/
bool parse_diskann_build_mode(const std::string &value,
                              diskann_build_mode *build_mode);

/**
  Parse vector-index consistency text into enum value.

  TRANSACTIONAL is the default MySQL truth-store path. STANDALONE creates an
  independent non-transactional index that does not persist vector payload rows
  in the truth-store.
*/
bool parse_index_consistency_mode(const std::string &value,
                                  index_consistency_mode *consistency_mode);

/**
  Convert enum values to canonical lowercase option text.
*/
const char *metric_to_string(metric_type metric);
const char *backend_mode_to_string(backend_mode mode);
const char *backend_provider_to_string(backend_provider provider);
const char *diskann_build_mode_to_string(diskann_build_mode build_mode);
const char *index_consistency_mode_to_string(
    index_consistency_mode consistency_mode);

/**
  remove provider-specific persistent artifacts for a vector index.

  This is used by lifecycle DROP cleanup paths.
*/
bool remove_backend_artifacts(const std::string &index_name, backend_mode mode,
                            backend_provider provider);

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
/**
  Override FAISS EXTERNAL snapshot root directory for tests.
*/
void set_faiss_external_snapshot_root_for_testing(const std::string &root_path);
void reset_faiss_external_snapshot_root_for_testing();

/**
  Test-only wrappers for internal backend helper routines.

  These helpers expose parser and filesystem utility branches that are hard to
  exercise through backend public APIs alone.
*/
bool parse_uint64_for_testing(const std::string &text, uint64_t *value);
bool load_external_manifest_generation_for_testing(const std::string &path,
                                              const char *expected_header,
                                              uint64_t *generation,
                                              bool *exists);
bool save_external_manifest_generation_for_testing(const std::string &path,
                                              const char *header,
                                              uint64_t generation);
bool quarantine_file_if_exists_for_testing(const std::string &path);
bool remove_generated_snapshots_with_prefix_for_testing(const std::string &directory,
                                                  const std::string &prefix);
bool remove_dir_if_empty_for_testing(const std::string &path);
bool ensure_parent_directory_for_testing(const std::string &path);
bool ends_with_for_testing(const std::string &text, const std::string &suffix);
bool decode_hex_bytes_for_testing(const std::string &input, std::string *output);
std::string diskann_store_directory_for_testing(const std::string &index_name);
std::string diskann_term_directory_for_testing(const std::string &index_name,
                                           uint64_t ctx);
std::string diskann_key_path_for_testing(const std::string &index_name, uint64_t ctx,
                                     const std::string &key_bytes);
bool diskann_load_value_for_testing(const std::string &index_name, uint64_t ctx,
                                const std::string &key_bytes,
                                std::string *value);
bool diskann_save_value_for_testing(const std::string &index_name, uint64_t ctx,
                                const std::string &key_bytes,
                                const std::string &value);
bool diskann_delete_value_for_testing(const std::string &index_name, uint64_t ctx,
                                  const std::string &key_bytes);
bool diskann_build_memory_store_round_trip_for_testing(
    const std::string &index_name, uint64_t ctx, const std::string &key_bytes,
    const std::string &value, std::string *loaded_before_flush,
    bool *file_exists_before_flush, std::string *loaded_after_flush);
bool diskann_resident_store_round_trip_for_testing(
    const std::string &index_name, uint64_t ctx, const std::string &key_bytes,
    const std::string &value, std::string *loaded_after_removing_file);
bool diskann_read_modify_write_round_trip_for_testing(
    const std::string &index_name, uint64_t ctx, const std::string &key_bytes,
    const std::string &initial_value, const std::string &patch_value,
    size_t write_length, std::string *persistent_value,
    std::string *build_memory_value, std::string *resident_value);
bool diskann_parse_prefixed_keys_for_testing(const std::string &payload,
                                             uint32_t key_count,
                                             std::vector<std::string> *keys,
                                             int fail_index = -1,
                                             bool null_key_data = false);
bool diskann_api_load_for_testing();
void diskann_set_adapter_path_for_testing(const std::string &path);
void diskann_reset_adapter_path_for_testing();
bool diskann_api_available_for_testing(bool has_handle, bool has_create_index,
                                       bool has_drop_index, bool has_insert,
                                       bool has_search_vector, bool has_remove,
                                       bool has_card,
                                       bool has_build_quant_table,
                                       bool has_backfill_quant_vectors,
                                       bool has_random_members,
                                       bool has_search_neighbors);
bool diskann_offline_api_load_for_testing();
void diskann_set_offline_adapter_path_for_testing(const std::string &path);
void diskann_reset_offline_adapter_path_for_testing();
std::vector<std::string> diskann_offline_api_symbol_names_for_testing();
bool diskann_offline_api_available_for_testing(bool has_handle, bool has_build,
                                               bool has_build_from_manifest,
                                               bool has_load_index,
                                               bool has_search,
                                               bool has_search_batch,
                                               bool has_card,
                                               bool has_drop_index);
bool diskann_offline_api_manifest_build_load_for_testing();
bool diskann_offline_api_manifest_build_available_for_testing(
    bool has_handle, bool has_build, bool has_build_from_manifest,
    bool has_load_index, bool has_search, bool has_search_batch,
    bool has_card, bool has_drop_index);
bool diskann_offline_api_native_pq_build_available_for_testing(
    bool api_available, bool has_native_pq_build);
bool diskann_vendored_runtime_api_load_for_testing();
uint64_t diskann_vendored_runtime_capabilities_for_testing();
bool diskann_vendored_build_config_valid_for_testing(
    const std::string &data_path, const std::string &index_prefix,
    uint32_t dimension, uint32_t max_degree, uint32_t build_complexity,
    double build_memory_gb, double search_cache_ratio, std::string *error);
bool diskann_vendored_search_config_valid_for_testing(
    uint32_t top_k, uint32_t search_complexity, uint32_t beamwidth,
    std::string *error);
bool diskann_vendored_batch_search_available_for_testing();
bool diskann_vendored_allocation_failure_drops_handle_for_testing();
bool diskann_vendored_load_config_budget_for_testing(
    size_t row_count, uint64_t build_memory_size, double *build_memory_gb,
    uint32_t *pq_chunks, uint32_t *cache_nodes);
bool diskann_loaded_handle_batch_rejects_null_for_testing();
double diskann_build_memory_size_gb_for_testing(uint64_t build_memory_size);
bool diskann_flatten_memory_budget_allows_for_testing(size_t entry_count,
                                                      size_t dimension,
                                                      uint64_t budget_size);
uint32_t diskann_cache_nodes_for_build_for_testing(
    size_t row_count, size_t dimension, uint32_t explicit_cache_nodes,
    uint64_t search_cache_size, double search_cache_ratio,
    uint32_t max_degree = 32, uint32_t disk_pq_dims = 0);
uint32_t diskann_effective_offline_max_degree_for_testing(size_t row_count,
                                                          uint32_t max_degree);
uint32_t diskann_effective_loaded_cache_nodes_for_testing(
    size_t row_count, uint64_t search_cache_size, double search_cache_ratio);
const char *diskann_runtime_variant_name_for_testing(int variant);
bool diskann_write_binary_values_for_testing(const std::string &path,
                                             uint32_t uint32_value,
                                             uint64_t uint64_value);
uint32_t diskann_offline_load_use_bfs_cache_for_testing(uint32_t cache_nodes,
                                                        bool force_bfs_cache);
int32_t diskann_metric_code_for_testing(metric_type metric);
bool parse_diskann_doc_id_for_testing(const uint8_t *data, size_t length,
                                      uint64_t *doc_id);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index

#endif  // SQL_VECTOR_INDEX_BACKEND_INCLUDED
