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

#ifndef SQL_VECTOR_INDEX_REGISTRY_INCLUDED
#define SQL_VECTOR_INDEX_REGISTRY_INCLUDED

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sql/handler.h"
#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_index_metadata_store.h"
#include "sql/vector/vector_index_service.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_statement_publication.h"
#include "sql/xa.h"

class THD;

namespace vector_index_registry {

/**
  Recovery-aware publication ownership for vector search and observability.

  The guard completes durable-intent recovery before taking the underlying
  read reservation, then verifies that no recovery request appeared while it
  waited. Registry calls made during its lifetime reuse that verified recovery
  boundary instead of recursively upgrading the same reservation to a writer.
*/
class publication_read_guard {
 public:
  publication_read_guard() = default;
  publication_read_guard(const publication_read_guard &) = delete;
  publication_read_guard &operator=(const publication_read_guard &) = delete;
  publication_read_guard(publication_read_guard &&) = delete;
  publication_read_guard &operator=(publication_read_guard &&) = delete;
  ~publication_read_guard();

  /** Acquire one index reader after completing pending intent recovery. */
  bool lock_index(const std::string &index_name);

  /** Acquire the catalog reader after completing pending intent recovery. */
  bool lock_catalog();

  /** True when this guard owns a verified publication read boundary. */
  bool owns_lock() const;

 private:
  vector_statement_publication::read_guard m_guard;
  bool m_read_scope_active{false};
};

/** Global serving health derived from durable truth-store recovery. */
enum class registry_health_state { kReady, kRecoveryRequired, kFailed };

struct index_info {
  size_t dimension{0};
  std::string metric;
  std::string mode;
  std::string provider;
  std::string consistency_mode;
  bool truth_store_enabled{true};
  std::string build_source;
  std::string build_pipeline_mode;
  std::string build_segment_profile;
  std::string build_segment_profile_reason;
  std::string build_pipeline_decision;
  std::string build_pipeline_trigger;
  uint64_t build_pipeline_rows{0};
  uint64_t build_pipeline_payload_size{0};
  uint64_t build_pipeline_raw_segments{0};
  uint64_t build_segment_effective_row_limit{0};
  uint64_t build_segment_effective_target_size{0};
  uint64_t build_segment_target_size{0};
  uint64_t build_segment_max_rows{0};
  std::string build_segment_policy;
  size_t standalone_ingest_memory_bytes{0};
  size_t standalone_segment_count{0};
  size_t standalone_segment_bytes{0};
  size_t standalone_raw_segment_count{0};
  size_t standalone_raw_segment_bytes{0};
  std::string backend_variant;
  vector_index::backend_build_diagnostics build_diagnostics;
  std::string schema_name;
  std::string table_name;
  std::string column_name;
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
  vector_index::diskann_build_mode diskann_build_mode_value{
      vector_index::diskann_build_mode::kAuto};
  uint32_t diskann_search_complexity{0};
  uint32_t diskann_search_beamwidth{0};
  uint64_t diskann_pq_code_budget_size{0};
  uint32_t diskann_disk_pq_dims{0};
  uint32_t diskann_cache_nodes{0};
  bool diskann_accelerate_build{false};
  bool diskann_shuffle_build{false};
  bool diskann_use_bfs_cache{false};
  bool diskann_build_mode_specified{false};
  std::string lifecycle_state;
  uint64_t lifecycle_version{0};
  uint32_t last_error_code{0};
  uint64_t last_error_ts{0};
  uint64_t last_apply_latency_ms{0};
  uint64_t recover_fallback_count{0};
  uint64_t last_recover_fallback_ts{0};
  bool external_manifest_present{false};
  uint64_t external_manifest_generation{0};
  uint64_t index_identity{0};
  uint64_t truth_generation{0};
  uint64_t config_generation{0};
  uint64_t artifact_generation{0};
  uint64_t runtime_generation{0};
  bool supports_mutations{false};
  size_t entry_count{0};
  size_t committed_entry_count{0};
  std::string owner_schema;
};

struct global_status_summary {
  uint64_t backend_loaded_indexes{0};
  uint64_t backend_writable_indexes{0};
  uint64_t backend_readonly_indexes{0};
  uint64_t backend_error_indexes{0};
  uint64_t backend_manifest_present_indexes{0};
  uint64_t backend_manifest_generation_max{0};
  uint64_t backend_mode_memory_indexes{0};
  uint64_t backend_mode_external_indexes{0};
  uint64_t backend_provider_native_indexes{0};
  uint64_t backend_provider_faiss_indexes{0};
  uint64_t backend_provider_diskann_indexes{0};
  uint64_t backend_provider_hnswlib_indexes{0};
  uint64_t backend_lifecycle_ready_indexes{0};
  uint64_t backend_lifecycle_rebuilding_indexes{0};
  uint64_t backend_lifecycle_recovering_indexes{0};
  uint64_t backend_lifecycle_failed_indexes{0};
  uint64_t rebuild_progress{0};
  uint64_t recover_progress{0};
  uint64_t pending_apply_count{0};
  uint64_t backlog_indexes{0};
};

struct create_index_options {
  /** True when all inherited creation defaults have been captured. */
  bool defaults_resolved{false};
  bool build_threads_specified{false};
  uint32_t build_threads{0};
  uint32_t diskann_max_degree{0};
  uint32_t diskann_build_complexity{0};
  uint64_t diskann_pq_code_budget_size{0};
  uint32_t diskann_disk_pq_dims{0};
  uint32_t diskann_search_complexity{0};
  uint32_t diskann_search_beamwidth{0};
  bool diskann_accelerate_build_specified{false};
  bool diskann_accelerate_build{false};
  bool diskann_shuffle_build_specified{false};
  bool diskann_shuffle_build{false};
  bool diskann_use_bfs_cache_specified{false};
  bool diskann_use_bfs_cache{false};
  bool consistency_mode_specified{false};
  vector_index::index_consistency_mode consistency_mode{
      vector_index::index_consistency_mode::kTransactional};
  std::string initial_lifecycle_state;
};

/** Ownership token for publishing one mapped-index backfill generation. */
struct index_backfill_token {
  uint64_t index_identity{0};
  uint64_t truth_generation{0};
  uint64_t config_generation{0};
  uint64_t artifact_generation{0};
  uint64_t runtime_generation{0};
  uint64_t lifecycle_version{0};
};

struct registry_state_snapshot {
  std::vector<vector_index_metadata_store::metadata_row> metadata_rows;
  vector_index::index_service::committed_state committed_state;
  std::vector<vector_index_metadata_store::change_log_row> change_log_rows;
  std::vector<std::string> lagging_index_names;
};

struct dropped_index_artifacts {
  std::string index_name;
  vector_index::backend_mode mode{vector_index::backend_mode::kMemory};
  vector_index::backend_provider provider{
      vector_index::backend_provider::kNative};
  vector_index_truth_store::publication_intent cleanup_intent;
  bool has_cleanup_intent{false};
  bool valid{false};
};

bool create_index(const std::string &index_name, size_t dimension,
                  const std::string &metric, const std::string &mode,
                  const std::string &provider, const std::string &owner_schema);
bool create_index(const std::string &index_name, size_t dimension,
                  const std::string &metric, const std::string &mode,
                  const std::string &provider, const std::string &owner_schema,
                  const create_index_options &options);
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool create_index(const std::string &index_name, size_t dimension,
                  const std::string &metric, const std::string &mode,
                  const std::string &provider);
bool create_index(const std::string &index_name, size_t dimension,
                  const std::string &metric, const std::string &mode,
                  const std::string &provider,
                  const create_index_options &options);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
bool create_mapped_index(const std::string &index_name, size_t dimension,
                         const std::string &metric, const std::string &mode,
                         const std::string &provider,
                       const std::string &schema_name,
                       const std::string &table_name,
                         const std::string &column_name,
                         const std::string &doc_id_column_name);
bool create_mapped_index(const std::string &index_name, size_t dimension,
                         const std::string &metric, const std::string &mode,
                         const std::string &provider,
                         const std::string &schema_name,
                         const std::string &table_name,
                         const std::string &column_name,
                         const std::string &doc_id_column_name,
                         const create_index_options &options);
bool snapshot_runtime_state(registry_state_snapshot *snapshot);
bool restore_runtime_state(const registry_state_snapshot &snapshot, bool persist);
bool restore_runtime_state_after_drop_rollback(
    const registry_state_snapshot &snapshot,
    const dropped_index_artifacts &artifacts);
bool drop_index(const std::string &index_name);
bool drop_index(const std::string &index_name,
                dropped_index_artifacts *artifacts);
void cleanup_dropped_index_artifacts(
    const dropped_index_artifacts &artifacts);
bool cleanup_dropped_index_physical_state(
    const vector_index_truth_store::publication_intent &intent,
    std::string *failure_stage);
bool drop_indexes_for_table(const std::string &db_name, const std::string &table_name);
bool drop_indexes_for_database(const std::string &db_name);
bool reset_mapped_indexes_for_table(const std::string &db_name,
                                const std::string &table_name);
bool rename_indexes_for_table(const std::string &old_db_name,
                           const std::string &old_table_name,
                           const std::string &new_db_name,
                           const std::string &new_table_name);
bool drop_index_for_column(const std::string &db_name, const std::string &table_name,
                        const std::string &column_name);
bool rename_index_for_column(const std::string &db_name,
                          const std::string &table_name,
                          const std::string &old_column_name,
                          const std::string &new_column_name);
bool begin_bulk_load(const std::string &index_name);
bool bulk_upsert_from_reader(
    const std::string &index_name,
    const vector_index::index_service::bulk_load_reader &reader,
    const vector_index::index_service::bulk_load_options &options,
    std::string *error,
    const vector_index::standalone_load_receipt *receipt = nullptr);
bool bulk_upsert_from_raw_files(
    const std::string &index_name, const std::string &vector_filename,
    const std::string &docid_filename,
    const vector_index::index_service::bulk_load_options &options,
    uint64_t *loaded_rows, std::string *error,
    const vector_index::standalone_load_receipt *receipt = nullptr);
bool entry_exists(const std::string &index_name, uint64_t doc_id, bool *found);
bool bulk_build_index(const std::string &index_name,
                      std::string *error = nullptr);
bool rebuild_index(const std::string &index_name, std::string *error = nullptr);
bool recover_index(const std::string &index_name);
bool recover_index(THD *thd, const std::string &index_name);
bool replace_committed_entries(
    const std::string &index_name,
    const vector_index::index_service::committed_entries &entries);
bool replace_committed_entries_preserve_lifecycle(
    const std::string &index_name,
    const vector_index::index_service::committed_entries &entries);
bool begin_backfill(const std::string &index_name, index_backfill_token *token);
bool publish_backfill(
    const std::string &index_name,
    const vector_index::index_service::committed_entries &entries,
    const index_backfill_token &token);
bool set_lifecycle_state(const std::string &index_name,
                         const std::string &lifecycle_state);
bool set_search_ef(const std::string &index_name, uint32_t search_ef);
bool set_hnsw_build_params(const std::string &index_name, uint32_t hnsw_m,
                        uint32_t hnsw_ef_construction);
bool set_faiss_ivf_params(const std::string &index_name, uint32_t faiss_nlist,
                       uint32_t faiss_nprobe);
bool set_faiss_ivf_pq_params(const std::string &index_name, uint32_t faiss_nlist,
                         uint32_t faiss_nprobe, uint32_t faiss_pq_m,
                         uint32_t faiss_pq_bits);
bool set_diskann_build_params(const std::string &index_name,
                           uint32_t diskann_max_degree,
                           uint32_t diskann_build_complexity,
                           uint32_t diskann_build_threads);
bool set_diskann_build_threads(const std::string &index_name,
                            uint32_t diskann_build_threads);
bool set_diskann_build_mode(
    const std::string &index_name,
    vector_index::diskann_build_mode diskann_build_mode_value);
bool set_index_consistency_mode(
    const std::string &index_name,
    vector_index::index_consistency_mode consistency_mode);
bool set_diskann_search_complexity(const std::string &index_name,
                                uint32_t diskann_search_complexity);
bool set_diskann_search_beamwidth(const std::string &index_name,
                                  uint32_t diskann_search_beamwidth);
bool set_diskann_pq_code_budget_size(const std::string &index_name,
                                     uint64_t diskann_pq_code_budget_size);
bool set_diskann_disk_pq_dims(const std::string &index_name,
                              uint32_t diskann_disk_pq_dims);
bool set_diskann_accelerate_build(const std::string &index_name,
                                  bool diskann_accelerate_build);
bool set_diskann_shuffle_build(const std::string &index_name,
                               bool diskann_shuffle_build);
bool set_diskann_use_bfs_cache(const std::string &index_name,
                               bool diskann_use_bfs_cache);
bool rebuild_all_indexes(size_t *rebuilt_count);
bool recover_all_indexes(size_t *recovered_count);
bool recover_all_indexes(THD *thd, size_t *recovered_count);
bool get_index_info(const std::string &index_name, index_info *info);
bool get_global_status_summary(global_status_summary *summary);
bool list_indexes(std::vector<std::string> *index_names);

/**
  List current indexes while the caller owns the publication catalog.

  This internal staging helper deliberately skips publication-intent recovery,
  because recovery would recursively acquire the catalog reservation.
*/
bool list_indexes_for_publication_catalog_guard(
    std::vector<std::string> *index_names);
bool metadata_loaded();
registry_health_state registry_health();
std::string registry_failure_reason();
size_t committed_vector_memory_bytes();
size_t total_pending_vector_memory_bytes();

/** Build a durable, per-index statement publication intent. */
bool make_statement_publication_intent(
    vector_index_truth_store::publication_operation operation,
    const std::string &index_name, const std::string &payload,
    vector_index_truth_store::publication_intent *intent);

/**
  Resolve inherited CREATE defaults and encode a deterministic intent payload.

  Recovery must not re-read mutable global defaults after the statement has
  committed. This helper therefore captures provider, mode, consistency and
  provider-specific creation parameters before the intent is persisted.
*/
bool make_create_statement_payload(
    size_t dimension, const std::string &metric, const std::string &mode,
    const std::string &provider, const std::string &owner_schema,
    const create_index_options &options, std::string *payload);

/** Recheck the expected CAS token while publication ownership is held. */
bool validate_statement_publication_intent(
    const vector_index_truth_store::publication_intent &intent);

/** Publish or idempotently recognize one committed statement intent. */
bool publish_statement_publication_intent(
    const vector_index_truth_store::publication_intent &intent,
    std::string *failure_stage = nullptr);

/** Delete a successfully published durable statement intent. */
bool acknowledge_statement_publication_intent(
    const vector_index_truth_store::publication_intent &intent,
    std::string *failure_stage = nullptr);

/** Request a retry of committed publication intents on the next safe entry. */
void schedule_publication_intent_recovery();

/** Publish and acknowledge committed intents before exposing registry state. */
bool ensure_publication_intents_recovered();

uint64_t begin_txn(THD *thd);
bool commit_txn(THD *thd, uint64_t txn_id);
bool rollback_txn(THD *thd, uint64_t txn_id);
bool pending_txn_changes(THD *thd, uint64_t txn_id, size_t *pending_count);
bool savepoint_txn(THD *thd, uint64_t txn_id, const std::string &name);
bool rollback_to_savepoint_txn(THD *thd, uint64_t txn_id,
                               const std::string &name);
bool release_savepoint_txn(THD *thd, uint64_t txn_id,
                           const std::string &name);

bool stage_upsert(THD *thd, uint64_t txn_id, const std::string &index_name,
                  uint64_t doc_id,
                  const vector_index::vector_data &vector);
bool stage_erase(THD *thd, uint64_t txn_id, const std::string &index_name,
                 uint64_t doc_id);
/** Roll back every explicit vector transaction owned by this connection. */
size_t rollback_explicit_txns_for_thd(THD *thd);
bool stage_upsert_for_thd_txn(uint64_t thd_id, uint64_t statement_id,
                          const std::string &index_name, uint64_t doc_id,
                          const vector_index::vector_data &vector);
bool stage_erase_for_thd_txn(uint64_t thd_id, uint64_t statement_id,
                         const std::string &index_name, uint64_t doc_id);
/**
  Stage mapped DML and its durable truth rows in the caller's transaction.

  @param thd current user thread
  @param statement_id SQL statement identifier used by statement savepoints
  @param changes vector changes produced after the base-table row mutation
  @return true when runtime pending state and durable truth writes are staged
*/
bool stage_changes_for_thd_txn(
    THD *thd, uint64_t statement_id,
    const std::vector<vector_index::index_service::pending_change_snapshot>
        &changes);

/** Return the ordered set of indexes touched by a THD transaction. */
bool pending_index_names_for_thd_txn(uint64_t thd_id,
                                     std::vector<std::string> *index_names);

/**
  Bind publication generations and append changelog rows before InnoDB commit.

  The caller must hold exclusive statement-publication ownership for every
  index returned by pending_index_names_for_thd_txn() until after commit or
  rollback completes.
*/
bool prepare_thd_txn_publication(THD *thd, uint64_t thd_id);

bool commit_stmt_for_thd_txn(uint64_t thd_id, uint64_t statement_id);
bool rollback_stmt_for_thd_txn(uint64_t thd_id, uint64_t statement_id);
/**
  Publish a committed THD transaction into the derived ANN runtime.

  @param thd_id committed server thread identifier
  @param recover_detached_xa replay durable rows when XA prepare already
    discarded the volatile THD context
  @param failure_stage optional diagnostic label identifying the publication
    stage that failed

  @return true when the transaction is published or has no work to publish
*/
bool publish_thd_txn(uint64_t thd_id, bool recover_detached_xa = false,
                     std::string *failure_stage = nullptr);
/** Drop volatile pending state after InnoDB has accepted XA prepare. */
bool detach_thd_txn_for_prepare(uint64_t thd_id);
bool commit_thd_txn(uint64_t thd_id);
bool rollback_thd_txn(uint64_t thd_id);
/**
  Discard a THD transaction context that was created only for savepoint
  tracking and has no staged vector changes.
*/
void discard_empty_thd_txn(uint64_t thd_id);
bool prepare_thd_txn(uint64_t thd_id, const XID &xid);
bool set_prepared_in_tc(const XID &xid);
bool has_prepared_xid(const XID &xid);
xa_status_code commit_prepared_xid(const XID &xid);
xa_status_code commit_prepared_xid_if_loaded(const XID &xid);
xa_status_code commit_prepared_xid_for_thd(uint64_t thd_id, const XID &xid);
xa_status_code rollback_prepared_xid(const XID &xid);
xa_status_code rollback_prepared_xid_if_loaded(const XID &xid);
xa_status_code rollback_prepared_xid_for_thd(uint64_t thd_id, const XID &xid);
int recover_prepared_xids(XA_recover_txn *txn_list, uint len, MEM_ROOT *mem_root);
int recover_prepared_in_tc(Xa_state_list &xa_list);
/** Validate a SQL savepoint operation without changing vector state. */
bool preflight_savepoint_thd_txn(uint64_t thd_id, const std::string &name);
/** Validate SQL rollback-to-savepoint without changing vector state. */
bool preflight_rollback_to_savepoint_thd_txn(uint64_t thd_id,
                                             const std::string &name);
/** Validate SQL savepoint release without changing vector state. */
bool preflight_release_savepoint_thd_txn(uint64_t thd_id,
                                         const std::string &name);
bool savepoint_thd_txn(uint64_t thd_id, const std::string &name);
bool rollback_to_savepoint_thd_txn(uint64_t thd_id, const std::string &name);
bool release_savepoint_thd_txn(uint64_t thd_id, const std::string &name);

bool upsert(const std::string &index_name, uint64_t doc_id,
            const vector_index::vector_data &vector);
bool erase(const std::string &index_name, uint64_t doc_id);

bool search(const std::string &index_name, const vector_index::vector_data &query,
            size_t top_k, std::vector<vector_index::search_result> *results);
bool search_for_thd_txn(THD *thd, uint64_t thd_id, const std::string &index_name,
                     const vector_index::vector_data &query, size_t top_k,
                     std::vector<vector_index::search_result> *results);
bool search_batch_for_thd_txn(
    THD *thd, uint64_t thd_id, const std::string &index_name,
    const std::vector<vector_index::vector_data> &queries, size_t top_k,
    std::vector<std::vector<vector_index::search_result>> *results);

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
/**
  Test-only wrappers for internal registry helper logic.
*/
uint64_t begin_txn();
bool commit_txn(uint64_t txn_id);
bool rollback_txn(uint64_t txn_id);
size_t pending_txn_changes(uint64_t txn_id);
bool savepoint_txn(uint64_t txn_id, const std::string &name);
bool rollback_to_savepoint_txn(uint64_t txn_id, const std::string &name);
bool release_savepoint_txn(uint64_t txn_id, const std::string &name);
bool stage_upsert(uint64_t txn_id, const std::string &index_name,
                  uint64_t doc_id,
                  const vector_index::vector_data &vector);
bool stage_erase(uint64_t txn_id, const std::string &index_name,
                 uint64_t doc_id);
bool parse_mapped_index_name_for_testing(const std::string &index_name,
                                    std::string *schema_name,
                                    std::string *table_name,
                                    std::string *column_name);
void rename_index_binding_for_testing(const std::string &old_index_name,
                                  const std::string &new_index_name);
bool get_index_binding_for_testing(const std::string &index_name,
                               std::string *schema_name,
                               std::string *table_name,
                               std::string *column_name,
                               std::string *doc_id_column_name);
bool snapshot_runtime_state_for_testing(
    std::vector<vector_index_metadata_store::metadata_row> *metadata_rows,
    vector_index::index_service::committed_state *committed_state,
    std::vector<vector_index_metadata_store::change_log_row> *change_log_rows,
    std::vector<std::string> *lagging_index_names);
bool restore_runtime_state_for_testing(
    const std::vector<vector_index_metadata_store::metadata_row> &metadata_rows,
    const vector_index::index_service::committed_state &committed_state,
    const std::vector<vector_index_metadata_store::change_log_row> &change_log_rows,
    const std::vector<std::string> &lagging_index_names);
bool build_backend_from_config_for_testing(
    const std::string &index_name,
    const vector_index::index_service::index_config &config);
bool recover_truth_projection_for_testing(
    const vector_index::index_service::committed_state &state);

struct effective_index_options_for_testing {
  uint32_t hnsw_build_threads{0};
  uint32_t faiss_build_threads{0};
  uint32_t diskann_build_threads{0};
  uint32_t diskann_max_degree{0};
  uint32_t diskann_build_complexity{0};
  uint64_t diskann_pq_code_budget_size{0};
  uint32_t diskann_disk_pq_dims{0};
  bool diskann_accelerate_build{false};
  bool diskann_shuffle_build{false};
  bool diskann_use_bfs_cache{false};
  uint32_t diskann_search_complexity{0};
  uint32_t diskann_search_beamwidth{0};
};

effective_index_options_for_testing effective_options_for_testing(
    const std::string &provider, const create_index_options &options);

void set_change_log_compact_threshold_for_testing(size_t threshold);

/**
  reset registry global state for unit tests.

  This function is test-only and must not be used by production SQL paths.
*/
void reset_for_testing();
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index_registry

#endif  // SQL_VECTOR_INDEX_REGISTRY_INCLUDED
