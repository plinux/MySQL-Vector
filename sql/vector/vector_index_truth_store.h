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
#include <string>
#include <vector>

#include "sql/vector/vector_index_metadata_store.h"

class THD;

namespace vector_index_truth_store {

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

  virtual bool load_metadata(
      std::vector<vector_index_metadata_store::metadata_row> *rows) = 0;
  virtual bool save_metadata(
      const std::vector<vector_index_metadata_store::metadata_row> &rows) = 0;
  virtual bool quarantine_metadata() = 0;

  virtual bool load_committed(
      std::vector<vector_index_metadata_store::committed_row> *rows) = 0;
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
  virtual bool quarantine_committed() = 0;

  virtual bool load_manifest(vector_index_metadata_store::manifest_row *row) = 0;
  virtual bool save_manifest(
      const vector_index_metadata_store::manifest_row &row) = 0;
  virtual bool quarantine_manifest() = 0;

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
  virtual bool quarantine_change_log() = 0;

  virtual bool load_prepared(
      std::vector<vector_index_metadata_store::prepared_change_row> *rows) = 0;
  virtual bool save_prepared(
      const std::vector<vector_index_metadata_store::prepared_change_row> &rows) = 0;
  virtual bool quarantine_prepared() = 0;
};

truth_store *get();
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
void set_for_testing(truth_store *store);
void reset_for_testing();
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
bool initialize_selected_backend();
bool bootstrap_initialize_selected_backend(THD *thd);
void shutdown_selected_backend();
bool internal_sql_active();
bool internal_truth_store_access_allowed(const THD *thd);
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
bool internal_execute_for_testing(const std::string &sql,
                               unsigned int *last_errno = nullptr,
                               std::string *last_error = nullptr);
bool internal_query_scalar_string_for_testing(const std::string &sql,
                                         std::string *value,
                                         bool *found = nullptr);
bool deserialize_quarantine_entries_for_testing(
    const std::string &payload,
    std::vector<std::pair<std::string, std::string>> *entries);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index_truth_store

#endif  // SQL_VECTOR_INDEX_TRUTH_STORE_INCLUDED
