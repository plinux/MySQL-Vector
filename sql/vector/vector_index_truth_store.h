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

#ifndef SQL_VECTOR_INDEX_TRUTH_STORE_INCLUDED
#define SQL_VECTOR_INDEX_TRUTH_STORE_INCLUDED

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "sql/vector/vector_index_metadata_store.h"

class THD;

namespace vector_index_truth_store {

/** Durable progress state for one quarantined truth-store artifact. */
enum class quarantine_state { kCopied, kSourceUpdateFailed, kComplete };

/** Byte-exact evidence retained for one truth-store corruption event. */
struct quarantine_record {
  std::string identity;
  std::string artifact_name;
  quarantine_state state{quarantine_state::kCopied};
  std::string reason;
  uint64_t checksum{0};
  uint64_t generation{0};
  uint64_t timestamp{0};
  std::string payload;
};

/**
  Truth-store abstraction for persisted vector index state.

  The registry uses this interface to isolate persistence backend details.
  The current default production backend is the mysql truth-store carried by
  hidden InnoDB tables. File truth-store remains available for bootstrap,
  testing, and explicit backend override flows.
*/
class truth_store {
 public:
  virtual ~truth_store() = default;

  /**
    Human-readable backend kind used by observability surfaces.
  */
  virtual const char *backend_name() const = 0;

  /**
    Whether this truth store provides transactional durability semantics.
  */
  virtual bool is_transactional() const = 0;

  /**
    Whether this truth store can persist committed rows and change-log rows
    through row-level delta operations. Transactional stores that keep the
    default false value use the full snapshot path.
  */
  virtual bool supports_delta_persist() const { return false; }

  /**
    Whether mapped DML can write truth rows through the caller's InnoDB trx.
  */
  virtual bool supports_attached_dml() const { return false; }

  /**
    Apply committed projection and changelog rows in the caller's transaction.

    Backends that do not share the user's InnoDB transaction must keep the
    default failure result. The SQL DML path must not fall back to a separate
    persistence transaction.

    @param thd current user thread
    @param rows exact durable rows allocated for this statement
    @return true when both projection and changelog writes were staged
  */
  virtual bool apply_attached_dml(
      THD *,
      const std::vector<vector_index_metadata_store::change_log_row> &) {
    return false;
  }

  /**
    Begin a grouped persist operation across truth-store objects.

    File-based stores may keep the default no-op behavior. Transactional
    backends can override this to start an internal transaction spanning
    metadata, committed snapshot, manifest and change-log updates.
  */
  virtual bool begin_persist() { return true; }

  /**
    Commit a grouped persist operation started by begin_persist().
  */
  virtual bool commit_persist() { return true; }

  /**
    Roll back a grouped persist operation started by begin_persist().
  */
  virtual void rollback_persist() {}

  /**
    Copy an artifact into durable quarantine history without changing it.

    @param artifact_name Logical truth-store artifact name.
    @param reason Stable diagnostic reason for the quarantine event.
    @param generation Generation observed by the recovery caller.
    @param identity Receives the new or resumed quarantine identity.

    @retval true Byte-exact evidence is durable.
    @retval false The source or quarantine history could not be read or saved.
  */
  virtual bool stage_quarantine(const std::string &artifact_name [[maybe_unused]],
                                const std::string &reason [[maybe_unused]],
                                uint64_t generation [[maybe_unused]],
                                std::string *identity [[maybe_unused]]) {
    return false;
  }

  /**
    Advance an existing quarantine record after source recovery progresses.

    @param identity Quarantine identity returned by stage_quarantine().
    @param state New durable recovery state.

    @retval true The state transition is durable.
    @retval false The record is absent, the transition regresses a completed
      event, or persistence failed.
  */
  virtual bool update_quarantine_state(
      const std::string &identity [[maybe_unused]],
      quarantine_state state [[maybe_unused]]) {
    return false;
  }

  virtual bool load_metadata(
      std::vector<vector_index_metadata_store::metadata_row> *rows) = 0;
  virtual bool save_metadata(
      const std::vector<vector_index_metadata_store::metadata_row> &rows) = 0;

  virtual bool load_committed(
      std::vector<vector_index_metadata_store::committed_row> *rows) = 0;
  /**
    Visit committed rows without exposing the caller to the storage shape.

    The default implementation is load-backed so existing truth-store backends
    keep their behavior. Row-store backends can override this method later to
    stream rows directly from the durable store.
  */
  virtual bool for_each_committed(
      const std::function<bool(
          const vector_index_metadata_store::committed_row &row)> &visitor);
  virtual bool for_each_committed(
      const std::string &index_name,
      const std::function<bool(
          const vector_index_metadata_store::committed_row &row)> &visitor);
  /**
    Find a single committed row by its stable vector index key.

    The default implementation scans only until the key is found. Row-store
    backends should override this with an indexed lookup to avoid materializing
    full committed snapshots when the committed entry cache has been evicted.
  */
  virtual bool find_committed(
      const std::string &index_name, uint64_t doc_id,
      vector_index_metadata_store::committed_row *row, bool *found);
  virtual bool save_committed(
      const std::vector<vector_index_metadata_store::committed_row> &rows) = 0;
  /**
    Apply committed-state row changes without rewriting the full snapshot.

    The input uses committed change-log semantics: upsert rows replace the
    (index_name, doc_id) vector payload; erase rows delete that key. Backends
    that cannot provide atomic row-level updates keep
    supports_delta_persist() false and the registry uses full snapshot
    persistence.
  */
  virtual bool apply_committed_delta(
      const std::vector<vector_index_metadata_store::change_log_row> &) {
    return false;
  }
  virtual bool load_manifest(vector_index_metadata_store::manifest_row *row) = 0;
  virtual bool save_manifest(
      const vector_index_metadata_store::manifest_row &row) = 0;

  virtual bool load_change_log(
      std::vector<vector_index_metadata_store::change_log_row> *rows) = 0;
  virtual bool save_change_log(
      const std::vector<vector_index_metadata_store::change_log_row> &rows) = 0;
  /**
    Append newly committed change-log rows without rewriting older rows.
  */
  virtual bool append_change_log_delta(
      const std::vector<vector_index_metadata_store::change_log_row> &) {
    return false;
  }
  virtual bool load_prepared(
      std::vector<vector_index_metadata_store::prepared_change_row> *rows) = 0;
  virtual bool save_prepared(
      const std::vector<vector_index_metadata_store::prepared_change_row> &rows) = 0;

  virtual bool load_segment_tasks(
      std::vector<vector_index_metadata_store::segment_task_row> *rows) = 0;
  virtual bool save_segment_tasks(
      const std::vector<vector_index_metadata_store::segment_task_row> &rows) = 0;
  virtual bool quarantine_segment_tasks() = 0;
};

truth_store *get();
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
void set_for_testing(truth_store *store);
void reset_for_testing();
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
bool bootstrap_initialize_selected_backend(THD *thd);
void shutdown_selected_backend();
const char *active_backend_name();
bool active_backend_transactional();
bool is_truth_store_table(const char *schema_name, const char *table_name);
bool debug_get_artifact_payload(const std::string &artifact_name,
                             std::string *payload);
bool debug_set_artifact_payload(const std::string &artifact_name,
                             const std::string &payload);
bool debug_delete_artifact(const std::string &artifact_name);

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
/**
  Test-only wrappers for internal truth-store helper logic.

  These helpers are intentionally exposed so gunit can exercise parser and
  escaping branches that are difficult to reach through production SQL paths.
*/
std::string sql_string_literal_for_testing(const char *text);
bool decode_hex_bytes_for_testing(const std::string &encoded, std::string *decoded);
bool deserialize_quarantine_entries_for_testing(
    const std::string &payload,
    std::vector<quarantine_record> *entries);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index_truth_store

#endif  // SQL_VECTOR_INDEX_TRUTH_STORE_INCLUDED
