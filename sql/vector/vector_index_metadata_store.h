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

#ifndef SQL_VECTOR_INDEX_METADATA_STORE_INCLUDED
#define SQL_VECTOR_INDEX_METADATA_STORE_INCLUDED

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_segment_task.h"

namespace vector_index_metadata_store {

inline constexpr const char *kMetadataHeaderV1 = "VECTOR_INDEX_METADATA_V1";

/**
  Persisted vector-index metadata row.

  This row stores index definition and lifecycle state.
*/
struct metadata_row {
  std::string index_name;
  size_t dimension{0};
  vector_index::metric_type metric{vector_index::metric_type::kEuclidean};
  vector_index::backend_mode mode{vector_index::backend_mode::kMemory};
  vector_index::backend_provider provider{vector_index::backend_provider::kNative};
  vector_index::index_consistency_mode consistency_mode{
      vector_index::index_consistency_mode::kTransactional};
  std::string schema_name;
  std::string table_name;
  std::string column_name;
  std::string lifecycle_state{"ready"};
  uint64_t lifecycle_version{1};
  uint32_t last_error_code{0};
  uint64_t last_error_ts{0};
  uint64_t recover_fallback_count{0};
  uint64_t last_recover_fallback_ts{0};
  uint32_t search_ef{0};
  uint32_t hnsw_m{0};
  uint32_t hnsw_ef_construction{0};
  uint32_t faiss_nlist{0};
  uint32_t faiss_nprobe{0};
  uint32_t faiss_pq_m{0};
  uint32_t faiss_pq_bits{0};
  uint32_t diskann_max_degree{0};
  uint32_t diskann_build_complexity{0};
  uint32_t diskann_search_complexity{0};
  uint32_t diskann_search_beamwidth{0};
  vector_index::diskann_build_mode diskann_build_mode_value{
      vector_index::diskann_build_mode::kAuto};
  std::string doc_id_column_name;
  uint32_t hnsw_build_threads{0};
  uint32_t diskann_build_threads{0};
  uint32_t faiss_build_threads{0};
  bool diskann_build_mode_specified{false};
  uint64_t diskann_pq_code_budget_size{0};
  uint32_t diskann_disk_pq_dims{0};
  bool diskann_accelerate_build{false};
  bool diskann_shuffle_build{false};
  bool diskann_use_bfs_cache{false};
  std::string owner_schema;
};

/**
  Persisted committed vector entry row.

  This row stores one committed doc_id -> vector mapping for an index. It is the
  SQL-layer recovery source for rebuildable vector serving state.
*/
struct committed_row {
  std::string index_name;
  uint64_t doc_id{0};
  vector_index::vector_data vector;
};

/**
  Persisted manifest row for vector-index truth checkpoints.

  The manifest tracks coarse checkpoints for metadata, committed rows and
  change-log objects so recovery can detect the latest durable registry state.
*/
struct manifest_row {
  std::string state{"ready"};
  uint64_t version{1};
  uint64_t metadata_checkpoint{0};
  uint64_t committed_checkpoint{0};
  uint64_t change_log_checkpoint{0};
};

/**
  Change-log operation kind.
*/
enum class change_op { kUpsert, kErase };

/**
  Persisted vector change-log row.

  Upsert rows include vector payload; erase rows keep vector empty.
*/
struct change_log_row {
  uint64_t sequence{0};
  uint64_t txn_id{0};
  change_op op{change_op::kUpsert};
  std::string index_name;
  uint64_t doc_id{0};
  vector_index::vector_data vector;
};

/**
  Persisted prepared vector change row.

  Rows are keyed by the transaction XID and retain the staged vector delta until
  crash recovery decides whether to commit or roll it back.
*/
struct prepared_change_row {
  int64_t format_id{-1};
  int64_t gtrid_length{0};
  int64_t bqual_length{0};
  std::string xid_data;
  bool prepared_in_tc{false};
  uint64_t txn_id{0};
  change_op op{change_op::kUpsert};
  std::string index_name;
  uint64_t doc_id{0};
  vector_index::vector_data vector;
};

/**
  Load all metadata rows from persistent storage.

  @retval true Metadata loaded successfully.
  @retval false Storage is unreadable or format is invalid.
*/
bool load_all(std::vector<metadata_row> *rows);
bool deserialize_metadata_rows(const std::string &payload,
                             std::vector<metadata_row> *rows);

/**
  Persist all metadata rows atomically.

  @retval true Metadata saved successfully.
  @retval false Write or rename failed.
*/
bool save_all(const std::vector<metadata_row> &rows);
bool serialize_metadata_rows(const std::vector<metadata_row> &rows,
                           std::string *payload);

/**
  Load all committed vector-entry rows from persistent storage.

  @retval true Rows loaded successfully.
  @retval false Storage is unreadable or format is invalid.
*/
bool load_committed_all(std::vector<committed_row> *rows);
bool deserialize_committed_rows(const std::string &payload,
                              std::vector<committed_row> *rows);

/**
  Persist all committed vector-entry rows atomically.

  @retval true Rows saved successfully.
  @retval false Write or rename failed.
*/
bool save_committed_all(const std::vector<committed_row> &rows);
bool serialize_committed_rows(const std::vector<committed_row> &rows,
                            std::string *payload);

/**
  Load persisted vector manifest row.

  Missing manifest is treated as an empty/default state.

  @retval true Manifest loaded successfully.
  @retval false Storage is unreadable or format is invalid.
*/
bool load_manifest(manifest_row *row);
bool deserialize_manifest_row(const std::string &payload, manifest_row *row);

/**
  Persist vector manifest row atomically.

  @retval true Manifest saved successfully.
  @retval false Write or rename failed.
*/
bool save_manifest(const manifest_row &row);
bool serialize_manifest_row(const manifest_row &row, std::string *payload);

/**
  Load all persisted vector change-log rows.

  @retval true Change log loaded successfully.
  @retval false Storage is unreadable or format is invalid.
*/
bool load_change_log(std::vector<change_log_row> *rows);
bool deserialize_change_log_rows(const std::string &payload,
                              std::vector<change_log_row> *rows);

/**
  Persist all vector change-log rows atomically.

  @retval true Change log saved successfully.
  @retval false Write or rename failed.
*/
bool save_change_log(const std::vector<change_log_row> &rows);
bool serialize_change_log_rows(const std::vector<change_log_row> &rows,
                            std::string *payload);

/**
  Load all persisted prepared vector change rows.

  @retval true Prepared rows loaded successfully.
  @retval false Storage is unreadable or format is invalid.
*/
bool load_prepared(std::vector<prepared_change_row> *rows);
bool deserialize_prepared_rows(const std::string &payload,
                             std::vector<prepared_change_row> *rows);

/**
  Persist all prepared vector change rows atomically.

  @retval true Prepared rows saved successfully.
  @retval false Write or rename failed.
*/
bool save_prepared(const std::vector<prepared_change_row> &rows);
bool serialize_prepared_rows(const std::vector<prepared_change_row> &rows,
                           std::string *payload);

/**
  Load all persisted segmented-build task rows.

  @retval true Segment task rows loaded successfully.
  @retval false Storage is unreadable or format is invalid.
*/
bool load_segment_tasks(std::vector<segment_task_row> *rows);
bool deserialize_segment_task_rows(const std::string &payload,
                                   std::vector<segment_task_row> *rows);

/**
  Persist all segmented-build task rows atomically.

  @retval true Segment task rows saved successfully.
  @retval false Write or rename failed.
*/
bool save_segment_tasks(const std::vector<segment_task_row> &rows);
bool serialize_segment_task_rows(const std::vector<segment_task_row> &rows,
                                 std::string *payload);

/**
  Load raw persisted bytes for a file-backed artifact.

  This is used by debug injection helpers that need byte-exact overwrite
  semantics for corruption and recovery tests.
*/
bool load_raw_artifact(const std::string &artifact_name, std::string *payload,
                     bool *found);

/**
  Persist raw bytes for a file-backed artifact atomically.
*/
bool save_raw_artifact(const std::string &artifact_name,
                     const std::string &payload);

/**
  Remove a file-backed artifact if it exists.
*/
bool delete_raw_artifact(const std::string &artifact_name);

/**
  Move the current store file aside after detecting corruption.

  @retval true Store is absent or moved successfully.
  @retval false Rename failed.
*/
bool quarantine_current_store();

/**
  Move the current committed-entry store file aside after detecting corruption.

  @retval true Store is absent or moved successfully.
  @retval false Rename failed.
*/
bool quarantine_committed_store();

/**
  Move the current manifest store file aside after detecting corruption.

  @retval true Store is absent or moved successfully.
  @retval false Rename failed.
*/
bool quarantine_manifest_store();

/**
  Move the current change-log store file aside after detecting corruption.

  @retval true Store is absent or moved successfully.
  @retval false Rename failed.
*/
bool quarantine_change_log_store();

/**
  Move the current prepared-change store file aside after detecting
  corruption.

  @retval true Store is absent or moved successfully.
  @retval false Rename failed.
*/
bool quarantine_prepared_store();

/**
  Move the current segmented-build task store file aside after detecting
  corruption.

  @retval true Store is absent or moved successfully.
  @retval false Rename failed.
*/
bool quarantine_segment_task_store();

/**
  Testing-only path override.

  Unit tests may override the metadata file location to avoid touching
  server datadir state.
*/
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
void set_path_for_testing(const std::string &path);
void reset_path_for_testing();
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index_metadata_store

#endif  // SQL_VECTOR_INDEX_METADATA_STORE_INCLUDED
