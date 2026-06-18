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
#include "sql/xa.h"

namespace vector_index_registry {

struct index_info {
  size_t dimension{0};
  std::string metric;
  std::string mode;
  std::string provider;
  std::string consistency_mode;
  bool truth_store_enabled{true};
  std::string build_source;
  std::string build_pipeline_mode;
  std::string build_pipeline_decision;
  std::string build_pipeline_trigger;
  uint64_t build_pipeline_rows{0};
  uint64_t build_pipeline_payload_size{0};
  uint64_t build_pipeline_raw_segments{0};
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
  bool supports_mutations{false};
  size_t entry_count{0};
  size_t committed_entry_count{0};
  std::string owner_schema;
};

struct create_index_options {
  bool build_threads_specified{false};
  uint32_t build_threads{0};
  uint32_t diskann_max_degree{0};
  uint32_t diskann_build_complexity{0};
  uint32_t diskann_disk_pq_dims{0};
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
bool drop_index(const std::string &index_name);
bool drop_index(const std::string &index_name,
                dropped_index_artifacts *artifacts);
void cleanup_dropped_index_artifacts(
    const dropped_index_artifacts &artifacts);
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
    std::string *error);
bool bulk_upsert_from_raw_files(
    const std::string &index_name, const std::string &vector_filename,
    const std::string &docid_filename,
    const vector_index::index_service::bulk_load_options &options,
    uint64_t *loaded_rows, std::string *error);
bool bulk_build_index(const std::string &index_name);
bool rebuild_index(const std::string &index_name);
bool recover_index(const std::string &index_name);
bool replace_committed_entries(
    const std::string &index_name,
    const vector_index::index_service::committed_entries &entries);
bool replace_committed_entries_preserve_lifecycle(
    const std::string &index_name,
    const vector_index::index_service::committed_entries &entries);
bool replace_committed_entries_for_backfill(
    const std::string &index_name,
    const vector_index::index_service::committed_entries &entries);
bool finish_backfill(const std::string &index_name,
                     const std::string &lifecycle_state);
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
bool get_index_info(const std::string &index_name, index_info *info);
bool list_indexes(std::vector<std::string> *index_names);
bool metadata_loaded();
size_t committed_vector_memory_bytes();
size_t total_pending_vector_memory_bytes();

uint64_t begin_txn();
bool commit_txn(uint64_t txn_id);
bool rollback_txn(uint64_t txn_id);
size_t pending_txn_changes(uint64_t txn_id);
bool savepoint_txn(uint64_t txn_id, const std::string &name);
bool rollback_to_savepoint_txn(uint64_t txn_id, const std::string &name);
bool release_savepoint_txn(uint64_t txn_id, const std::string &name);

bool stage_upsert(uint64_t txn_id, const std::string &index_name, uint64_t doc_id,
                 const vector_index::vector_data &vector);
bool stage_erase(uint64_t txn_id, const std::string &index_name, uint64_t doc_id);
bool stage_upsert_for_thd_txn(uint64_t thd_id, uint64_t statement_id,
                          const std::string &index_name, uint64_t doc_id,
                          const vector_index::vector_data &vector);
bool stage_erase_for_thd_txn(uint64_t thd_id, uint64_t statement_id,
                         const std::string &index_name, uint64_t doc_id);

bool commit_stmt_for_thd_txn(uint64_t thd_id, uint64_t statement_id);
bool rollback_stmt_for_thd_txn(uint64_t thd_id, uint64_t statement_id);
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
void queue_recovery_commit_xid(const XID &xid);
void queue_recovery_rollback_xid(const XID &xid);
void queue_recovery_set_prepared_in_tc(const XID &xid);
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
