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

/** Durable operation kind carried by one publication intent. */
enum class publication_operation : uint8_t {
  kTransactionalDml = 1,
  kStandaloneUpsert = 2,
  kStandaloneErase = 3,
  kBulkLoad = 4,
  kCreateIndex = 5,
  kDropIndex = 6,
  kUpdateConfig = 7,
  kRebuildIndex = 8,
  kRecoverIndex = 9,
  kBeginBulkLoad = 10,
  kBulkBuildIndex = 11,
};

/** Persisted state of a publication intent visible after transaction commit. */
enum class publication_intent_state : uint8_t {
  /** The target token is exact and can be acknowledged directly. */
  kReadyToPublish = 1,
  /** The target is derived by publishing the operation payload. */
  kTargetToBeObserved = 2,
};

/** Complete identity and generation tuple used by publication CAS. */
struct publication_token {
  bool exists{false};
  uint64_t index_identity{0};
  uint64_t truth_generation{0};
  uint64_t config_generation{0};
  uint64_t artifact_generation{0};
  uint64_t runtime_generation{0};
  uint64_t lifecycle_version{0};
  uint64_t source_generation{0};
};

/** Per-index durable intent committed before a private candidate is published. */
struct publication_intent {
  std::string index_name;
  uint64_t publication_id{0};
  uint64_t txn_id{0};
  publication_operation operation{publication_operation::kTransactionalDml};
  publication_token expected;
  publication_token target;
  std::string payload;
  publication_intent_state state{
      publication_intent_state::kReadyToPublish};
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

  /** Whether this backend durably coordinates per-index publication intents. */
  virtual bool supports_publication_intents() const { return false; }

  /** Ensure the caller's MySQL transaction owns the backing store engine. */
  virtual bool ensure_attached_transaction(THD *) { return false; }

  /** Apply committed projection rows in the caller's InnoDB transaction. */
  virtual bool apply_attached_committed(
      THD *,
      const std::vector<vector_index_metadata_store::change_log_row> &) {
    return false;
  }

  /** Append publication-bound changelog rows in the caller's transaction. */
  virtual bool append_attached_change_log(
      THD *,
      const std::vector<vector_index_metadata_store::change_log_row> &) {
    return false;
  }

  /** Insert an intent in the caller-owned MySQL/InnoDB transaction. */
  virtual bool insert_attached_publication_intent(
      THD *, const publication_intent &) {
    return false;
  }

  /** Persist an intent in the current grouped or an internal transaction. */
  virtual bool save_publication_intent(const publication_intent &) {
    return false;
  }

  /**
    Append changelog rows and intents while the caller's transaction prepares.

    Production backends must not register a new MySQL statement participant
    from this callback. The default implementation preserves focused test
    stores that model the two operations independently.
  */
  virtual bool prepare_attached_publication(
      THD *thd,
      const std::vector<vector_index_metadata_store::change_log_row> &rows,
      const std::vector<publication_intent> &intents) {
    if (!append_attached_change_log(thd, rows)) return false;
    for (const auto &intent : intents) {
      if (!insert_attached_publication_intent(thd, intent)) return false;
    }
    return true;
  }

  /** Load committed intents that require publication or acknowledgement. */
  virtual bool load_publication_intents(
      std::vector<publication_intent> *) {
    return false;
  }

  /** Acknowledge one exact intent after its target token is visible. */
  virtual bool delete_publication_intent(const std::string &, uint64_t) {
    return false;
  }

  /**
    Acknowledge one committed intent in an independent transaction.

    The default preserves test and nontransactional stores whose delete path
    does not join an owner-thread grouped persist operation.
  */
  virtual bool delete_committed_publication_intent(
      const std::string &index_name, uint64_t publication_id) {
    return delete_publication_intent(index_name, publication_id);
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
  /** Delete at most one bounded batch of one dropped index's truth rows. */
  virtual bool erase_committed_index_batch(const std::string &, size_t,
                                           size_t *, bool *) {
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
  /** Delete only changelog sequences already folded into committed truth. */
  virtual bool erase_change_log_sequences(const std::vector<uint64_t> &) {
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
