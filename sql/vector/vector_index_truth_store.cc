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

#include "sql/vector/vector_index_truth_store.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lex_string.h"
#include "m_ctype.h"
#include "my_checksum.h"
#include "my_dbug.h"
#include "sql/dd/impl/bootstrap/bootstrap_ctx.h"
#include "sql/mysqld.h"
#include "sql/sql_class.h"
#include "sql/vector/vector_index_diagnostics.h"
#include "sql/vector/vector_index_truth_store_internal.h"
#include "storage/innobase/include/dict0vectruth.h"

namespace {

constexpr const char *kTruthStoreBackendEnv = "MYSQL_VECTOR_TRUTH_STORE";
constexpr const char *kMetadataArtifactName = "metadata";
constexpr const char *kCommittedArtifactName = "committed";
constexpr const char *kManifestArtifactName = "manifest";
constexpr const char *kChangeLogArtifactName = "changelog";
constexpr const char *kPreparedArtifactName = "prepared";
constexpr const char *kSegmentTasksArtifactName = "segment_tasks";
constexpr const char *kQuarantineStoreArtifactName = "quarantine_store";
constexpr const char *kQuarantinePayloadHeaderV1 = "mysql-vector-quarantine-v1";
std::atomic<uint64_t> g_quarantine_identity_sequence{1};

enum class row_artifact_kind { kNone, kCommitted, kChangeLog, kPrepared };

char hex_digit(unsigned value) {
  return static_cast<char>((value < 10U) ? ('0' + value)
                                         : ('a' + (value - 10U)));
}

std::string encode_hex_bytes(const std::string &input) {
  std::string out;
  out.resize(input.size() * 2);
  for (size_t i = 0; i < input.size(); ++i) {
    const unsigned value =
        static_cast<unsigned>(static_cast<unsigned char>(input[i]));
    out[i * 2] = hex_digit((value >> 4) & 0x0FU);
    out[i * 2 + 1] = hex_digit(value & 0x0FU);
  }
  return out;
}

std::string sql_string_literal(const char *text) {
  std::string escaped("'");
  if (text != nullptr) {
    for (const char *ptr = text; *ptr != '\0'; ++ptr) {
      if (*ptr == '\'' || *ptr == '\\') escaped.push_back('\\');
      escaped.push_back(*ptr);
    }
  }
  escaped.push_back('\'');
  return escaped;
}

bool hex_value(char ch, unsigned *value) {
  if (ch >= '0' && ch <= '9') {
    *value = static_cast<unsigned>(ch - '0');
    return true;
  }
  const char lower =
      static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (lower >= 'a' && lower <= 'f') {
    *value = static_cast<unsigned>(lower - 'a' + 10);
    return true;
  }
  return false;
}

bool decode_hex_bytes(const std::string &encoded, std::string *decoded) {
  if (decoded == nullptr) return false;
  decoded->clear();
  if ((encoded.size() % 2) != 0) return false;
  decoded->reserve(encoded.size() / 2);
  for (size_t i = 0; i < encoded.size(); i += 2) {
    unsigned high = 0;
    unsigned low = 0;
    if (!hex_value(encoded[i], &high) || !hex_value(encoded[i + 1], &low))
      return false;
    decoded->push_back(static_cast<char>((high << 4) | low));
  }
  return true;
}

bool split_tab_fields(const std::string &line,
                      std::vector<std::string> *fields) {
  if (fields == nullptr) return false;
  fields->clear();
  size_t pos = 0;
  while (pos <= line.size()) {
    const size_t tab = line.find('\t', pos);
    if (tab == std::string::npos) {
      fields->push_back(line.substr(pos));
      return true;
    }
    fields->push_back(line.substr(pos, tab - pos));
    pos = tab + 1;
  }
  return true;
}

using quarantine_entries =
    std::vector<vector_index_truth_store::quarantine_record>;

const char *quarantine_state_name(
    vector_index_truth_store::quarantine_state state) {
  using vector_index_truth_store::quarantine_state;
  switch (state) {
    case quarantine_state::kCopied:
      return "copied";
    case quarantine_state::kSourceUpdateFailed:
      return "source_update_failed";
    case quarantine_state::kComplete:
      return "complete";
  }
  return nullptr;
}

bool parse_quarantine_state(const std::string &value,
                            vector_index_truth_store::quarantine_state *state) {
  using vector_index_truth_store::quarantine_state;
  if (state == nullptr) return false;
  if (value == "copied") {
    *state = quarantine_state::kCopied;
    return true;
  }
  if (value == "source_update_failed") {
    *state = quarantine_state::kSourceUpdateFailed;
    return true;
  }
  if (value == "complete") {
    *state = quarantine_state::kComplete;
    return true;
  }
  return false;
}

bool parse_uint64(const std::string &value, uint64_t *result) {
  if (result == nullptr || value.empty()) return false;
  uint64_t parsed = 0;
  const char *begin = value.data();
  const char *end = begin + value.size();
  const auto conversion = std::from_chars(begin, end, parsed);
  if (conversion.ec != std::errc() || conversion.ptr != end) return false;
  *result = parsed;
  return true;
}

uint64_t quarantine_payload_checksum(const std::string &payload) {
  return static_cast<uint64_t>(
      my_checksum(0, reinterpret_cast<const unsigned char *>(payload.data()),
                  payload.size()));
}

uint64_t quarantine_timestamp() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

void record_artifact_persist_event(const char *backend_name,
                                   const char *artifact_name, size_t row_count,
                                   size_t payload_bytes, uint64_t serialize_ms,
                                   uint64_t save_ms, bool ok) {
  if (!vector_index_diagnostics::enabled()) return;
  vector_index_diagnostics::record_event(
      "vector_truth_store_artifact",
      {{"backend", backend_name == nullptr ? "" : backend_name},
       {"artifact", artifact_name == nullptr ? "" : artifact_name}},
      {{"rows", static_cast<uint64_t>(row_count)},
       {"payload_bytes", static_cast<uint64_t>(payload_bytes)},
       {"serialize_ms", serialize_ms},
       {"save_ms", save_ms},
       {"ok", ok ? 1U : 0U}});
}

constexpr uint8_t kTruthRowOpUpsert = 1;
constexpr uint8_t kTruthRowOpErase = 2;

bool vector_to_truth_payload(const vector_index::vector_data &vector,
                             uint32_t *dimension, std::string *payload) {
  if (dimension == nullptr || payload == nullptr ||
      vector.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *dimension = static_cast<uint32_t>(vector.size());
  const size_t bytes = vector.size() * sizeof(float);
  payload->clear();
  if (bytes > 0) {
    payload->assign(reinterpret_cast<const char *>(vector.data()), bytes);
  }
  return true;
}

bool truth_payload_to_vector(uint32_t dimension, const std::string &payload,
                             vector_index::vector_data *vector) {
  if (vector == nullptr ||
      payload.size() != static_cast<size_t>(dimension) * sizeof(float)) {
    return false;
  }
  vector->resize(dimension);
  if (!payload.empty()) {
    std::memcpy(vector->data(), payload.data(), payload.size());
  }
  return true;
}

bool encode_truth_op(vector_index_metadata_store::change_op op,
                     uint8_t *value) {
  if (value == nullptr) return false;
  switch (op) {
    case vector_index_metadata_store::change_op::kUpsert:
      *value = kTruthRowOpUpsert;
      return true;
    case vector_index_metadata_store::change_op::kErase:
      *value = kTruthRowOpErase;
      return true;
  }
  return false;
}

bool decode_truth_op(uint8_t value,
                     vector_index_metadata_store::change_op *op) {
  if (op == nullptr) return false;
  if (value == kTruthRowOpUpsert) {
    *op = vector_index_metadata_store::change_op::kUpsert;
    return true;
  }
  if (value == kTruthRowOpErase) {
    *op = vector_index_metadata_store::change_op::kErase;
    return true;
  }
  return false;
}

bool to_innodb_committed_rows(
    const std::vector<vector_index_metadata_store::committed_row> &rows,
    std::vector<innodb_vector_truth_store::committed_row> *out) {
  if (out == nullptr) return false;
  out->clear();
  out->reserve(rows.size());
  for (const auto &row : rows) {
    innodb_vector_truth_store::committed_row stored;
    stored.index_name = row.index_name;
    stored.doc_id = row.doc_id;
    if (stored.index_name.empty() ||
        !vector_to_truth_payload(row.vector, &stored.dimension,
                                 &stored.vector_payload)) {
      return false;
    }
    out->push_back(std::move(stored));
  }
  return true;
}

bool from_innodb_committed_row(
    const innodb_vector_truth_store::committed_row &row,
    vector_index_metadata_store::committed_row *out) {
  if (out == nullptr) return false;
  vector_index_metadata_store::committed_row loaded;
  loaded.index_name = row.index_name;
  loaded.doc_id = row.doc_id;
  if (loaded.index_name.empty() ||
      !truth_payload_to_vector(row.dimension, row.vector_payload,
                               &loaded.vector)) {
    return false;
  }
  *out = std::move(loaded);
  return true;
}

bool from_innodb_committed_rows(
    const std::vector<innodb_vector_truth_store::committed_row> &rows,
    std::vector<vector_index_metadata_store::committed_row> *out) {
  if (out == nullptr) return false;
  out->clear();
  out->reserve(rows.size());
  for (const auto &row : rows) {
    vector_index_metadata_store::committed_row loaded;
    if (!from_innodb_committed_row(row, &loaded)) return false;
    out->push_back(std::move(loaded));
  }
  return true;
}

bool to_innodb_change_rows(
    const std::vector<vector_index_metadata_store::change_log_row> &rows,
    bool require_publication_binding,
    std::vector<innodb_vector_truth_store::change_log_row> *out) {
  if (out == nullptr) return false;
  out->clear();
  out->reserve(rows.size());
  for (const auto &row : rows) {
    innodb_vector_truth_store::change_log_row stored;
    stored.sequence = row.sequence;
    stored.txn_id = row.txn_id;
    stored.index_identity = row.index_identity;
    stored.publication_id = row.publication_id;
    stored.truth_generation = row.truth_generation;
    stored.index_name = row.index_name;
    stored.doc_id = row.doc_id;
    if (stored.sequence == 0 || stored.index_identity == 0 ||
        (require_publication_binding &&
         (stored.publication_id == 0 || stored.truth_generation == 0)) ||
        stored.sequence == std::numeric_limits<uint64_t>::max() ||
        stored.index_identity == std::numeric_limits<uint64_t>::max() ||
        stored.publication_id == std::numeric_limits<uint64_t>::max() ||
        stored.truth_generation == std::numeric_limits<uint64_t>::max() ||
        stored.index_name.empty() ||
        !encode_truth_op(row.op, &stored.op) ||
        !vector_to_truth_payload(row.vector, &stored.dimension,
                                 &stored.vector_payload)) {
      return false;
    }
    if (row.op == vector_index_metadata_store::change_op::kErase &&
        stored.dimension != 0) {
      return false;
    }
    out->push_back(std::move(stored));
  }
  return true;
}

bool to_innodb_truth_delta_rows(
    const std::vector<vector_index_metadata_store::change_log_row> &rows,
    std::vector<innodb_vector_truth_store::change_log_row> *out) {
  return to_innodb_change_rows(rows, false, out);
}

bool to_innodb_change_log_rows(
    const std::vector<vector_index_metadata_store::change_log_row> &rows,
    std::vector<innodb_vector_truth_store::change_log_row> *out) {
  return to_innodb_change_rows(rows, true, out);
}

bool valid_persisted_change_log_rows(
    const std::vector<vector_index_metadata_store::change_log_row> &rows) {
  return std::all_of(
      rows.begin(), rows.end(), [](const auto &row) {
        return row.sequence != 0 && row.index_identity != 0 &&
               row.publication_id != 0 && row.truth_generation != 0 &&
               row.sequence != std::numeric_limits<uint64_t>::max() &&
               row.index_identity != std::numeric_limits<uint64_t>::max() &&
               row.publication_id != std::numeric_limits<uint64_t>::max() &&
               row.truth_generation != std::numeric_limits<uint64_t>::max() &&
               !row.index_name.empty();
      });
}

bool from_innodb_change_log_rows(
    const std::vector<innodb_vector_truth_store::change_log_row> &rows,
    std::vector<vector_index_metadata_store::change_log_row> *out) {
  if (out == nullptr) return false;
  out->clear();
  out->reserve(rows.size());
  for (const auto &row : rows) {
    vector_index_metadata_store::change_log_row loaded;
    loaded.sequence = row.sequence;
    loaded.txn_id = row.txn_id;
    loaded.index_identity = row.index_identity;
    loaded.publication_id = row.publication_id;
    loaded.truth_generation = row.truth_generation;
    loaded.index_name = row.index_name;
    loaded.doc_id = row.doc_id;
    if (loaded.sequence == 0 || loaded.index_identity == 0 ||
        loaded.publication_id == 0 || loaded.truth_generation == 0 ||
        loaded.sequence == std::numeric_limits<uint64_t>::max() ||
        loaded.index_identity == std::numeric_limits<uint64_t>::max() ||
        loaded.publication_id == std::numeric_limits<uint64_t>::max() ||
        loaded.truth_generation == std::numeric_limits<uint64_t>::max() ||
        loaded.index_name.empty() ||
        !decode_truth_op(row.op, &loaded.op) ||
        !truth_payload_to_vector(row.dimension, row.vector_payload,
                                 &loaded.vector)) {
      return false;
    }
    if (loaded.op == vector_index_metadata_store::change_op::kErase &&
        !loaded.vector.empty()) {
      return false;
    }
    out->push_back(std::move(loaded));
  }
  return true;
}

bool to_innodb_prepared_rows(
    const std::vector<vector_index_metadata_store::prepared_change_row> &rows,
    std::vector<innodb_vector_truth_store::prepared_change_row> *out) {
  if (out == nullptr) return false;
  out->clear();
  out->reserve(rows.size());
  uint64_t row_no = 1;
  for (const auto &row : rows) {
    if (row.txn_id == 0 ||
        !vector_index_metadata_store::valid_prepared_xid(row)) {
      return false;
    }
    innodb_vector_truth_store::prepared_change_row stored;
    stored.txn_id = row.txn_id;
    stored.row_no = row_no++;
    stored.format_id = static_cast<uint64_t>(row.format_id);
    stored.gtrid_length = static_cast<uint64_t>(row.gtrid_length);
    stored.bqual_length = static_cast<uint64_t>(row.bqual_length);
    stored.xid_data = row.xid_data;
    stored.prepared_in_tc = row.prepared_in_tc ? 1 : 0;
    stored.index_name = row.index_name;
    stored.doc_id = row.doc_id;
    if (stored.index_name.empty() || !encode_truth_op(row.op, &stored.op) ||
        !vector_to_truth_payload(row.vector, &stored.dimension,
                                 &stored.vector_payload)) {
      return false;
    }
    if (row.op == vector_index_metadata_store::change_op::kErase &&
        stored.dimension != 0) {
      return false;
    }
    out->push_back(std::move(stored));
  }
  return true;
}

bool from_innodb_prepared_rows(
    const std::vector<innodb_vector_truth_store::prepared_change_row> &rows,
    std::vector<vector_index_metadata_store::prepared_change_row> *out) {
  if (out == nullptr) return false;
  out->clear();
  out->reserve(rows.size());
  for (const auto &row : rows) {
    if (row.format_id >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        row.gtrid_length >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        row.bqual_length >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    vector_index_metadata_store::prepared_change_row loaded;
    loaded.txn_id = row.txn_id;
    loaded.format_id = static_cast<int64_t>(row.format_id);
    loaded.gtrid_length = static_cast<int64_t>(row.gtrid_length);
    loaded.bqual_length = static_cast<int64_t>(row.bqual_length);
    loaded.xid_data = row.xid_data;
    loaded.prepared_in_tc = row.prepared_in_tc != 0;
    loaded.index_name = row.index_name;
    loaded.doc_id = row.doc_id;
    if (loaded.txn_id == 0 ||
        !vector_index_metadata_store::valid_prepared_xid(loaded) ||
        loaded.index_name.empty() || !decode_truth_op(row.op, &loaded.op) ||
        !truth_payload_to_vector(row.dimension, row.vector_payload,
                                 &loaded.vector)) {
      return false;
    }
    if (loaded.op == vector_index_metadata_store::change_op::kErase &&
        !loaded.vector.empty()) {
      return false;
    }
    out->push_back(std::move(loaded));
  }
  return true;
}

bool valid_publication_operation(
    vector_index_truth_store::publication_operation operation) {
  using vector_index_truth_store::publication_operation;
  switch (operation) {
    case publication_operation::kTransactionalDml:
    case publication_operation::kStandaloneUpsert:
    case publication_operation::kStandaloneErase:
    case publication_operation::kBulkLoad:
    case publication_operation::kCreateIndex:
    case publication_operation::kDropIndex:
    case publication_operation::kUpdateConfig:
    case publication_operation::kRebuildIndex:
    case publication_operation::kRecoverIndex:
    case publication_operation::kBeginBulkLoad:
    case publication_operation::kBulkBuildIndex:
      return true;
  }
  return false;
}

bool valid_publication_token(
    const vector_index_truth_store::publication_token &token) {
  const uint64_t max_value = std::numeric_limits<uint64_t>::max();
  if (!token.exists) {
    return token.index_identity == 0 && token.truth_generation == 0 &&
           token.config_generation == 0 && token.artifact_generation == 0 &&
           token.runtime_generation == 0 && token.lifecycle_version == 0 &&
           token.source_generation == 0;
  }
  return token.index_identity != 0 && token.index_identity != max_value &&
         token.truth_generation != max_value &&
         token.config_generation != 0 &&
         token.config_generation != max_value &&
         token.artifact_generation != max_value &&
         token.runtime_generation != max_value &&
         token.lifecycle_version != 0 &&
         token.lifecycle_version != max_value &&
         token.source_generation != max_value;
}

bool valid_publication_intent(
    const vector_index_truth_store::publication_intent &intent) {
  const uint64_t max_value = std::numeric_limits<uint64_t>::max();
  return !intent.index_name.empty() && intent.publication_id != 0 &&
         intent.publication_id != max_value &&
         valid_publication_operation(intent.operation) &&
         (intent.state == vector_index_truth_store::publication_intent_state::
                              kReadyToPublish ||
          intent.state == vector_index_truth_store::publication_intent_state::
                              kTargetToBeObserved) &&
         valid_publication_token(intent.expected) &&
         valid_publication_token(intent.target) &&
         (intent.state == vector_index_truth_store::publication_intent_state::
                  kTargetToBeObserved ||
          intent.expected.exists || intent.target.exists) &&
         (!intent.expected.exists || !intent.target.exists ||
          intent.expected.index_identity == intent.target.index_identity);
}

void to_innodb_publication_token(
    const vector_index_truth_store::publication_token &token,
    uint8_t *exists, uint64_t *index_identity, uint64_t *truth_generation,
    uint64_t *config_generation, uint64_t *artifact_generation,
    uint64_t *runtime_generation, uint64_t *lifecycle_version,
    uint64_t *source_generation) {
  *exists = token.exists ? 1 : 0;
  *index_identity = token.index_identity;
  *truth_generation = token.truth_generation;
  *config_generation = token.config_generation;
  *artifact_generation = token.artifact_generation;
  *runtime_generation = token.runtime_generation;
  *lifecycle_version = token.lifecycle_version;
  *source_generation = token.source_generation;
}

bool to_innodb_publication_intent(
    const vector_index_truth_store::publication_intent &intent,
    innodb_vector_truth_store::publication_intent_row *stored) {
  if (stored == nullptr || !valid_publication_intent(intent)) return false;
  stored->index_name = intent.index_name;
  stored->publication_id = intent.publication_id;
  stored->txn_id = intent.txn_id;
  stored->operation = static_cast<uint8_t>(intent.operation);
  to_innodb_publication_token(
      intent.expected, &stored->expected_exists,
      &stored->expected_index_identity, &stored->expected_truth_generation,
      &stored->expected_config_generation,
      &stored->expected_artifact_generation,
      &stored->expected_runtime_generation,
      &stored->expected_lifecycle_version,
      &stored->expected_source_generation);
  to_innodb_publication_token(
      intent.target, &stored->target_exists, &stored->target_index_identity,
      &stored->target_truth_generation, &stored->target_config_generation,
      &stored->target_artifact_generation,
      &stored->target_runtime_generation,
      &stored->target_lifecycle_version,
      &stored->target_source_generation);
  stored->payload = intent.payload;
  stored->state = static_cast<uint8_t>(intent.state);
  return true;
}

void from_innodb_publication_token(
    uint8_t exists, uint64_t index_identity, uint64_t truth_generation,
    uint64_t config_generation, uint64_t artifact_generation,
    uint64_t runtime_generation, uint64_t lifecycle_version,
    uint64_t source_generation,
    vector_index_truth_store::publication_token *token) {
  token->exists = exists != 0;
  token->index_identity = index_identity;
  token->truth_generation = truth_generation;
  token->config_generation = config_generation;
  token->artifact_generation = artifact_generation;
  token->runtime_generation = runtime_generation;
  token->lifecycle_version = lifecycle_version;
  token->source_generation = source_generation;
}

bool from_innodb_publication_intent(
    const innodb_vector_truth_store::publication_intent_row &stored,
    vector_index_truth_store::publication_intent *intent) {
  if (intent == nullptr || stored.expected_exists > 1 ||
      stored.target_exists > 1) {
    return false;
  }
  intent->index_name = stored.index_name;
  intent->publication_id = stored.publication_id;
  intent->txn_id = stored.txn_id;
  intent->operation = static_cast<vector_index_truth_store::publication_operation>(
      stored.operation);
  from_innodb_publication_token(
      stored.expected_exists, stored.expected_index_identity,
      stored.expected_truth_generation, stored.expected_config_generation,
      stored.expected_artifact_generation, stored.expected_runtime_generation,
      stored.expected_lifecycle_version, stored.expected_source_generation,
      &intent->expected);
  from_innodb_publication_token(
      stored.target_exists, stored.target_index_identity,
      stored.target_truth_generation, stored.target_config_generation,
      stored.target_artifact_generation, stored.target_runtime_generation,
      stored.target_lifecycle_version, stored.target_source_generation,
      &intent->target);
  intent->payload = stored.payload;
  intent->state =
      static_cast<vector_index_truth_store::publication_intent_state>(
          stored.state);
  return valid_publication_intent(*intent);
}

bool from_innodb_publication_intents(
    const std::vector<innodb_vector_truth_store::publication_intent_row> &rows,
    std::vector<vector_index_truth_store::publication_intent> *intents) {
  if (intents == nullptr) return false;
  intents->clear();
  intents->reserve(rows.size());
  for (const auto &row : rows) {
    vector_index_truth_store::publication_intent intent;
    if (!from_innodb_publication_intent(row, &intent)) return false;
    intents->push_back(std::move(intent));
  }
  return true;
}

row_artifact_kind row_artifact_kind_from_name(const char *artifact_name) {
  if (artifact_name == nullptr) return row_artifact_kind::kNone;
  if (std::strcmp(artifact_name, kCommittedArtifactName) == 0)
    return row_artifact_kind::kCommitted;
  if (std::strcmp(artifact_name, kChangeLogArtifactName) == 0)
    return row_artifact_kind::kChangeLog;
  if (std::strcmp(artifact_name, kPreparedArtifactName) == 0)
    return row_artifact_kind::kPrepared;
  return row_artifact_kind::kNone;
}

bool is_row_artifact_name(const char *artifact_name) {
  return row_artifact_kind_from_name(artifact_name) != row_artifact_kind::kNone;
}

bool extract_debug_payload(
    const std::vector<innodb_vector_truth_store::committed_row> &rows,
    std::string *payload) {
  if (rows.size() != 1 || payload == nullptr) return false;
  const auto &row = rows.front();
  if (!row.index_name.empty() || row.doc_id != 0 || row.dimension != 0)
    return false;
  *payload = row.vector_payload;
  return true;
}

bool extract_debug_payload(
    const std::vector<innodb_vector_truth_store::change_log_row> &rows,
    std::string *payload) {
  if (rows.size() != 1 || payload == nullptr) return false;
  const auto &row = rows.front();
  if (row.sequence != 1 || row.txn_id != 0 || row.op != kTruthRowOpUpsert ||
      !row.index_name.empty() || row.doc_id != 0 || row.dimension != 0) {
    return false;
  }
  *payload = row.vector_payload;
  return true;
}

bool extract_debug_payload(
    const std::vector<innodb_vector_truth_store::prepared_change_row> &rows,
    std::string *payload) {
  if (rows.size() != 1 || payload == nullptr) return false;
  const auto &row = rows.front();
  if (row.txn_id != 0 || row.row_no != 1 || row.format_id != 0 ||
      row.gtrid_length != 0 || row.bqual_length != 0 || !row.xid_data.empty() ||
      row.prepared_in_tc != 0 || row.op != kTruthRowOpUpsert ||
      !row.index_name.empty() || row.doc_id != 0 || row.dimension != 0) {
    return false;
  }
  *payload = row.vector_payload;
  return true;
}

innodb_vector_truth_store::committed_row make_debug_committed_row(
    const std::string &payload) {
  innodb_vector_truth_store::committed_row row;
  row.doc_id = 0;
  row.dimension = 0;
  row.vector_payload = payload;
  return row;
}

innodb_vector_truth_store::change_log_row make_debug_change_log_row(
    const std::string &payload) {
  innodb_vector_truth_store::change_log_row row;
  row.sequence = 1;
  row.txn_id = 0;
  row.op = kTruthRowOpUpsert;
  row.doc_id = 0;
  row.dimension = 0;
  row.vector_payload = payload;
  return row;
}

innodb_vector_truth_store::prepared_change_row make_debug_prepared_row(
    const std::string &payload) {
  innodb_vector_truth_store::prepared_change_row row;
  row.txn_id = 0;
  row.row_no = 1;
  row.format_id = 0;
  row.gtrid_length = 0;
  row.bqual_length = 0;
  row.prepared_in_tc = 0;
  row.op = kTruthRowOpUpsert;
  row.doc_id = 0;
  row.dimension = 0;
  row.vector_payload = payload;
  return row;
}

bool deserialize_quarantine_entries(const std::string &payload,
                                    quarantine_entries *entries) {
  if (entries == nullptr) return false;
  entries->clear();
  if (payload.empty()) return false;

  std::istringstream stream(payload);
  std::string line;
  if (!std::getline(stream, line) || line != kQuarantinePayloadHeaderV1)
    return false;

  while (std::getline(stream, line)) {
    if (line.empty()) continue;
    std::vector<std::string> fields;
    if (!split_tab_fields(line, &fields) || fields.size() != 8) return false;
    vector_index_truth_store::quarantine_record record;
    if (!decode_hex_bytes(fields[0], &record.identity) ||
        !decode_hex_bytes(fields[1], &record.artifact_name) ||
        !parse_quarantine_state(fields[2], &record.state) ||
        !decode_hex_bytes(fields[3], &record.reason) ||
        !parse_uint64(fields[4], &record.checksum) ||
        !parse_uint64(fields[5], &record.generation) ||
        !parse_uint64(fields[6], &record.timestamp) ||
        !decode_hex_bytes(fields[7], &record.payload) ||
        record.identity.empty() || record.artifact_name.empty() ||
        record.reason.empty() ||
        record.checksum != quarantine_payload_checksum(record.payload)) {
      return false;
    }
    entries->push_back(std::move(record));
  }
  return true;
}

bool serialize_quarantine_entries(const quarantine_entries &entries,
                                  std::string *payload) {
  if (payload == nullptr) return false;
  std::ostringstream stream;
  stream << kQuarantinePayloadHeaderV1 << "\n";
  for (const auto &entry : entries) {
    const char *state = quarantine_state_name(entry.state);
    if (entry.identity.empty() || entry.artifact_name.empty() ||
        entry.reason.empty() || state == nullptr ||
        entry.checksum != quarantine_payload_checksum(entry.payload)) {
      return false;
    }
    stream << encode_hex_bytes(entry.identity) << "\t"
           << encode_hex_bytes(entry.artifact_name) << "\t" << state << "\t"
           << encode_hex_bytes(entry.reason) << "\t" << entry.checksum << "\t"
           << entry.generation << "\t" << entry.timestamp << "\t"
           << encode_hex_bytes(entry.payload) << "\n";
  }
  *payload = stream.str();
  return true;
}

bool append_quarantine_entry(quarantine_entries *entries,
                             const std::string &artifact_name,
                             const std::string &reason, uint64_t generation,
                             const std::string &payload,
                             std::string *identity) {
  using vector_index_truth_store::quarantine_record;
  using vector_index_truth_store::quarantine_state;
  if (entries == nullptr || identity == nullptr || artifact_name.empty() ||
      reason.empty()) {
    return false;
  }

  const uint64_t checksum = quarantine_payload_checksum(payload);
  for (auto it = entries->rbegin(); it != entries->rend(); ++it) {
    if (it->artifact_name == artifact_name && it->checksum == checksum &&
        it->state != quarantine_state::kComplete) {
      *identity = it->identity;
      return true;
    }
  }

  quarantine_record record;
  record.artifact_name = artifact_name;
  record.state = quarantine_state::kCopied;
  record.reason = reason;
  record.checksum = checksum;
  record.generation = generation;
  record.timestamp = quarantine_timestamp();
  record.payload = payload;
  const uint64_t sequence =
      g_quarantine_identity_sequence.fetch_add(1, std::memory_order_relaxed);
  record.identity = artifact_name + "-" + std::to_string(record.timestamp) +
                    "-" + std::to_string(sequence) + "-" +
                    std::to_string(checksum);
  *identity = record.identity;
  entries->push_back(std::move(record));
  return true;
}

bool update_quarantine_entry_state(
    quarantine_entries *entries, const std::string &identity,
    vector_index_truth_store::quarantine_state state) {
  using vector_index_truth_store::quarantine_state;
  if (entries == nullptr || identity.empty()) return false;
  for (auto &entry : *entries) {
    if (entry.identity != identity) continue;
    if (entry.state == quarantine_state::kComplete &&
        state != quarantine_state::kComplete) {
      return false;
    }
    entry.state = state;
    return true;
  }
  return false;
}

bool is_valid_artifact_name(const char *artifact_name) {
  return artifact_name != nullptr &&
         (std::strcmp(artifact_name, kMetadataArtifactName) == 0 ||
          std::strcmp(artifact_name, kCommittedArtifactName) == 0 ||
          std::strcmp(artifact_name, kManifestArtifactName) == 0 ||
          std::strcmp(artifact_name, kChangeLogArtifactName) == 0 ||
          std::strcmp(artifact_name, kPreparedArtifactName) == 0 ||
          std::strcmp(artifact_name, kSegmentTasksArtifactName) == 0);
}

bool use_file_truth_store_backend() {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  const char *backend = std::getenv(kTruthStoreBackendEnv);
  if (backend == nullptr) return false;
  return std::strcmp(backend, "file") == 0;
#else
  return false;
#endif
}

bool is_truth_store_table_name(const char *schema_name,
                               const char *table_name) {
  if (schema_name == nullptr || table_name == nullptr) return false;
  if (my_strcasecmp(system_charset_info, schema_name, "mysql") != 0)
    return false;
  return my_strcasecmp(system_charset_info, table_name,
                       "vector_index_truth_metadata") == 0 ||
         my_strcasecmp(system_charset_info, table_name,
                       "vector_index_truth_committed") == 0 ||
         my_strcasecmp(system_charset_info, table_name,
                       "vector_index_truth_manifest") == 0 ||
         my_strcasecmp(system_charset_info, table_name,
                       "vector_index_truth_changelog") == 0 ||
         my_strcasecmp(system_charset_info, table_name,
                       "vector_index_truth_prepared") == 0 ||
         my_strcasecmp(system_charset_info, table_name,
                       "vector_index_publication_intents") == 0 ||
         my_strcasecmp(system_charset_info, table_name,
                       "vector_index_truth_segment_tasks") == 0 ||
         my_strcasecmp(system_charset_info, table_name,
                       "vector_index_truth_store_quarantine") == 0;
}

class mysql_truth_store final : public vector_index_truth_store::truth_store {
 private:
  enum class change_log_persist_mode { kReplace, kAppend };

 public:
  using vector_index_truth_store::truth_store::for_each_committed;

  const char *backend_name() const override { return "mysql"; }

  bool is_transactional() const override { return true; }
  bool supports_delta_persist() const override { return true; }
  bool supports_attached_dml() const override { return true; }
  bool supports_publication_intents() const override { return true; }

  bool ensure_attached_transaction(THD *thd) override {
    innodb_vector_truth_store::Session *session =
        innodb_vector_truth_store::begin_attached_session(thd);
    if (session == nullptr) return false;
    innodb_vector_truth_store::close_session(session);
    return true;
  }

  bool apply_attached_committed(
      THD *thd,
      const std::vector<vector_index_metadata_store::change_log_row> &rows)
      override {
    if (thd == nullptr || rows.empty()) return false;

    std::vector<innodb_vector_truth_store::change_log_row> stored_rows;
    if (!to_innodb_truth_delta_rows(rows, &stored_rows)) return false;

    innodb_vector_truth_store::Session *session =
        innodb_vector_truth_store::begin_attached_session(thd);
    if (session == nullptr) return false;

    const bool committed_ok =
        innodb_vector_truth_store::apply_committed_delta(stored_rows, session);
    innodb_vector_truth_store::close_session(session);
    return committed_ok;
  }

  bool append_attached_change_log(
      THD *thd,
      const std::vector<vector_index_metadata_store::change_log_row> &rows)
      override {
    if (thd == nullptr || rows.empty() ||
        !valid_persisted_change_log_rows(rows)) {
      return false;
    }

    std::vector<innodb_vector_truth_store::change_log_row> stored_rows;
    if (!to_innodb_change_log_rows(rows, &stored_rows)) return false;

    innodb_vector_truth_store::Session *session =
        innodb_vector_truth_store::begin_attached_session(thd);
    if (session == nullptr) return false;

    DBUG_EXECUTE_IF("vector_truth_fail_attached_changelog",
                    innodb_vector_truth_store::close_session(session);
                    return false;);
    const bool changelog_ok =
        innodb_vector_truth_store::append_change_log_delta(stored_rows,
                                                           session);
    innodb_vector_truth_store::close_session(session);
    return changelog_ok;
  }

  bool insert_attached_publication_intent(
      THD *thd,
      const vector_index_truth_store::publication_intent &intent) override {
    if (thd == nullptr || !valid_publication_intent(intent)) return false;

    innodb_vector_truth_store::publication_intent_row stored;
    if (!to_innodb_publication_intent(intent, &stored)) return false;
    innodb_vector_truth_store::Session *session =
        innodb_vector_truth_store::begin_attached_session(thd);
    if (session == nullptr) return false;

    DBUG_EXECUTE_IF("vector_truth_fail_attached_publication_intent",
                    innodb_vector_truth_store::close_session(session);
                    return false;);
    const bool inserted =
        innodb_vector_truth_store::insert_publication_intent(stored, session);
    innodb_vector_truth_store::close_session(session);
    return inserted;
  }

  bool save_publication_intent(
      const vector_index_truth_store::publication_intent &intent) override {
    if (!valid_publication_intent(intent)) return false;
    innodb_vector_truth_store::publication_intent_row stored;
    if (!to_innodb_publication_intent(intent, &stored)) return false;
    return with_persist_session(
        [&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::save_publication_intent(stored,
                                                                    session);
        });
  }

  bool prepare_attached_publication(
      THD *thd,
      const std::vector<vector_index_metadata_store::change_log_row> &rows,
      const std::vector<vector_index_truth_store::publication_intent> &intents)
      override {
    if (thd == nullptr || rows.empty() || intents.empty() ||
        !valid_persisted_change_log_rows(rows)) {
      return false;
    }

    std::vector<innodb_vector_truth_store::change_log_row> stored_rows;
    if (!to_innodb_change_log_rows(rows, &stored_rows)) return false;
    std::vector<innodb_vector_truth_store::publication_intent_row>
        stored_intents;
    stored_intents.reserve(intents.size());
    for (const auto &intent : intents) {
      innodb_vector_truth_store::publication_intent_row stored;
      if (!to_innodb_publication_intent(intent, &stored)) return false;
      stored_intents.push_back(std::move(stored));
    }

    innodb_vector_truth_store::Session *session =
        innodb_vector_truth_store::begin_attached_session_for_prepare(thd);
    if (session == nullptr) return false;

    DBUG_EXECUTE_IF("vector_truth_fail_attached_changelog",
                    innodb_vector_truth_store::close_session(session);
                    return false;);
    bool prepared = innodb_vector_truth_store::append_change_log_delta(
        stored_rows, session);
    for (const auto &intent : stored_intents) {
      DBUG_EXECUTE_IF("vector_truth_fail_attached_publication_intent",
                      prepared = false;);
      if (!prepared || !innodb_vector_truth_store::insert_publication_intent(
                           intent, session)) {
        prepared = false;
        break;
      }
    }
    innodb_vector_truth_store::close_session(session);
    return prepared;
  }

  bool load_publication_intents(
      std::vector<vector_index_truth_store::publication_intent> *intents)
      override {
    if (intents == nullptr) return false;
    std::vector<innodb_vector_truth_store::publication_intent_row> stored_rows;
    bool found = false;
    if (!innodb_vector_truth_store::load_publication_intents(
            &stored_rows, &found, nullptr)) {
      return false;
    }
    if (!found) {
      intents->clear();
      return true;
    }
    return from_innodb_publication_intents(stored_rows, intents);
  }

  bool delete_publication_intent(const std::string &index_name,
                                 uint64_t publication_id) override {
    if (index_name.empty() || publication_id == 0) return false;
    DBUG_EXECUTE_IF("vector_truth_fail_delete_publication_intent",
                    return false;);
    return with_persist_session(
        [&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::delete_publication_intent(
              index_name, publication_id, session);
        });
  }

  bool delete_committed_publication_intent(
      const std::string &index_name, uint64_t publication_id) override {
    if (index_name.empty() || publication_id == 0) return false;
    DBUG_EXECUTE_IF("vector_truth_fail_delete_publication_intent",
                    return false;);
    return innodb_vector_truth_store::delete_publication_intent(
        index_name, publication_id, nullptr);
  }

  bool bootstrap_initialize(THD *thd) {
    return thd != nullptr;
  }

  bool begin_persist() override {
    std::lock_guard<std::mutex> guard(m_mutex);
    const std::thread::id owner = std::this_thread::get_id();
    if (m_persist_sessions.find(owner) != m_persist_sessions.end()) {
      return false;
    }
    innodb_vector_truth_store::Session *session =
        innodb_vector_truth_store::begin_session(true);
    if (session == nullptr) return false;
    m_persist_sessions.emplace(owner, session);
    return true;
  }

  bool commit_persist() override {
    std::lock_guard<std::mutex> guard(m_mutex);
    const auto it = m_persist_sessions.find(std::this_thread::get_id());
    if (it == m_persist_sessions.end()) return false;
    innodb_vector_truth_store::Session *session = it->second;
    const bool ok = innodb_vector_truth_store::commit_session(session);
    innodb_vector_truth_store::close_session(session);
    m_persist_sessions.erase(it);
    return ok;
  }

  void rollback_persist() override {
    std::lock_guard<std::mutex> guard(m_mutex);
    const auto it = m_persist_sessions.find(std::this_thread::get_id());
    if (it == m_persist_sessions.end()) return;
    innodb_vector_truth_store::rollback_session(it->second);
    innodb_vector_truth_store::close_session(it->second);
    m_persist_sessions.erase(it);
  }

  bool stage_quarantine(const std::string &artifact_name,
                        const std::string &reason, uint64_t generation,
                        std::string *identity) override {
    if (identity == nullptr || !is_valid_artifact_name(artifact_name.c_str())) {
      return false;
    }
    DBUG_EXECUTE_IF("vector_truth_store_fail_stage_quarantine", return false;);

    std::string artifact_payload;
    bool artifact_found = false;
    if (!load_source_artifact_payload(artifact_name.c_str(), &artifact_payload,
                                      &artifact_found)) {
      return false;
    }
    if (!artifact_found) artifact_payload.clear();

    quarantine_entries entries;
    if (!load_quarantine_entries(&entries) ||
        !append_quarantine_entry(&entries, artifact_name, reason, generation,
                                 artifact_payload, identity)) {
      return false;
    }
    return save_quarantine_entries(entries);
  }

  bool update_quarantine_state(
      const std::string &identity,
      vector_index_truth_store::quarantine_state state) override {
    DBUG_EXECUTE_IF("vector_truth_store_fail_quarantine_state", return false;);
    quarantine_entries entries;
    if (!load_quarantine_entries(&entries) ||
        !update_quarantine_entry_state(&entries, identity, state)) {
      return false;
    }
    return save_quarantine_entries(entries);
  }

  bool load_metadata(
      std::vector<vector_index_metadata_store::metadata_row> *rows) override {
    if (rows == nullptr) return false;
    std::string payload;
    bool found = false;
    if (!load_structured_artifact(kMetadataArtifactName, &payload, &found)) {
      return false;
    }
    if (!found) {
      rows->clear();
      return true;
    }
    if (payload.empty()) return false;
    return vector_index_metadata_store::deserialize_metadata_rows(payload,
                                                                  rows);
  }

  bool save_metadata(
      const std::vector<vector_index_metadata_store::metadata_row> &rows)
      override {
    std::string payload;
    if (!vector_index_metadata_store::serialize_metadata_rows(rows, &payload))
      return false;
    return save_artifact(kMetadataArtifactName, payload);
  }

  bool load_committed(
      std::vector<vector_index_metadata_store::committed_row> *rows) override {
    std::vector<innodb_vector_truth_store::committed_row> stored_rows;
    bool found = false;
    if (!with_persist_session([&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::load_committed_rows(
              &stored_rows, &found, session);
        })) {
      return false;
    }
    if (!found) {
      if (rows != nullptr) rows->clear();
      return true;
    }
    return from_innodb_committed_rows(stored_rows, rows);
  }

  bool for_each_committed(
      const std::string &index_name,
      const std::function<bool(const vector_index_metadata_store::committed_row
                                   &row)> &visitor) override {
    if (visitor == nullptr || index_name.empty()) return false;
    bool found = false;
    return with_persist_session(
        [&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::visit_committed_rows_for_index(
              index_name,
              [&](const innodb_vector_truth_store::committed_row &stored_row) {
                vector_index_metadata_store::committed_row row;
                return from_innodb_committed_row(stored_row, &row) &&
                       visitor(row);
              },
              &found, session);
        });
  }

  bool find_committed(const std::string &index_name, uint64_t doc_id,
                      vector_index_metadata_store::committed_row *row,
                      bool *found) override {
    if (index_name.empty() || row == nullptr || found == nullptr) return false;
    innodb_vector_truth_store::committed_row stored_row;
    if (!with_persist_session([&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::find_committed_row(
              index_name, doc_id, &stored_row, found, session);
        })) {
      return false;
    }
    if (!*found) {
      *row = vector_index_metadata_store::committed_row();
      return true;
    }
    return from_innodb_committed_row(stored_row, row);
  }

  bool save_committed(
      const std::vector<vector_index_metadata_store::committed_row> &rows)
      override {
    const bool diagnostics_enabled = vector_index_diagnostics::enabled();
    const auto serialize_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    std::vector<innodb_vector_truth_store::committed_row> stored_rows;
    if (!to_innodb_committed_rows(rows, &stored_rows)) return false;
    const uint64_t serialize_ms = vector_index_diagnostics::elapsed_ms_if(
        diagnostics_enabled, serialize_start);
    const auto save_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    const bool ok =
        with_persist_session([&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::save_committed_rows(stored_rows,
                                                                session);
        });
    if (diagnostics_enabled) {
      size_t payload_bytes = 0;
      for (const auto &row : stored_rows)
        payload_bytes += row.vector_payload.size();
      record_artifact_persist_event(backend_name(), kCommittedArtifactName,
                                    rows.size(), payload_bytes, serialize_ms,
                                    vector_index_diagnostics::elapsed_ms_if(
                                        diagnostics_enabled, save_start),
                                    ok);
    }
    return ok;
  }

  bool apply_committed_delta(
      const std::vector<vector_index_metadata_store::change_log_row> &rows)
      override {
    const bool diagnostics_enabled = vector_index_diagnostics::enabled();
    const auto serialize_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    std::vector<innodb_vector_truth_store::change_log_row> stored_rows;
    if (!to_innodb_truth_delta_rows(rows, &stored_rows)) return false;
    const uint64_t serialize_ms = vector_index_diagnostics::elapsed_ms_if(
        diagnostics_enabled, serialize_start);
    const auto save_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    const bool ok =
        with_persist_session([&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::apply_committed_delta(stored_rows,
                                                                  session);
        });
    if (diagnostics_enabled) {
      size_t payload_bytes = 0;
      for (const auto &row : stored_rows)
        payload_bytes += row.vector_payload.size();
      record_artifact_persist_event(backend_name(), "committed_delta",
                                    rows.size(), payload_bytes, serialize_ms,
                                    vector_index_diagnostics::elapsed_ms_if(
                                        diagnostics_enabled, save_start),
                                    ok);
    }
    return ok;
  }

  bool erase_committed_index_batch(const std::string &index_name,
                                   size_t max_rows, size_t *erased_rows,
                                   bool *done) override {
    if (index_name.empty() || max_rows == 0 || erased_rows == nullptr ||
        done == nullptr) {
      return false;
    }
    DBUG_EXECUTE_IF("vector_truth_fail_erase_committed_index_batch",
                    return false;);
    return with_persist_session(
        [&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::erase_committed_rows_for_index(
              index_name, max_rows, erased_rows, done, session);
        });
  }

  bool load_manifest(vector_index_metadata_store::manifest_row *row) override {
    if (row == nullptr) return false;
    std::string payload;
    bool found = false;
    if (!load_structured_artifact(kManifestArtifactName, &payload, &found)) {
      return false;
    }
    if (!found) {
      *row = vector_index_metadata_store::manifest_row();
      return true;
    }
    if (payload.empty()) return false;
    return vector_index_metadata_store::deserialize_manifest_row(payload, row);
  }

  bool save_manifest(
      const vector_index_metadata_store::manifest_row &row) override {
    std::string payload;
    if (!vector_index_metadata_store::serialize_manifest_row(row, &payload))
      return false;
    return save_artifact(kManifestArtifactName, payload);
  }

  bool load_change_log(
      std::vector<vector_index_metadata_store::change_log_row> *rows) override {
    std::vector<innodb_vector_truth_store::change_log_row> stored_rows;
    bool found = false;
    if (!with_persist_session([&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::load_change_log_rows(
              &stored_rows, &found, session);
        })) {
      return false;
    }
    if (!found) {
      if (rows != nullptr) rows->clear();
      return true;
    }
    return from_innodb_change_log_rows(stored_rows, rows);
  }

  bool save_change_log(
      const std::vector<vector_index_metadata_store::change_log_row> &rows)
      override {
    return persist_change_log(rows, change_log_persist_mode::kReplace);
  }

  bool append_change_log_delta(
      const std::vector<vector_index_metadata_store::change_log_row> &rows)
      override {
    return persist_change_log(rows, change_log_persist_mode::kAppend);
  }

  bool erase_change_log_sequences(
      const std::vector<uint64_t> &sequences) override {
    if (std::any_of(sequences.begin(), sequences.end(),
                    [](uint64_t sequence) { return sequence == 0; })) {
      return false;
    }
    return with_persist_session(
        [&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::erase_change_log_sequences(
              sequences, session);
        });
  }

  bool load_prepared(
      std::vector<vector_index_metadata_store::prepared_change_row> *rows)
      override {
    std::vector<innodb_vector_truth_store::prepared_change_row> stored_rows;
    bool structured_found = false;
    if (!with_persist_session([&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::load_prepared_rows(
              &stored_rows, &structured_found, session);
        })) {
      return false;
    }
    if (!structured_found) {
      if (rows != nullptr) rows->clear();
      return true;
    }
    return from_innodb_prepared_rows(stored_rows, rows);
  }

  bool save_prepared(
      const std::vector<vector_index_metadata_store::prepared_change_row> &rows)
      override {
    DBUG_EXECUTE_IF("vector_truth_store_fail_save", return false;);
    std::vector<innodb_vector_truth_store::prepared_change_row> stored_rows;
    if (!to_innodb_prepared_rows(rows, &stored_rows)) return false;
    return with_persist_session(
        [&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::save_prepared_rows(stored_rows,
                                                               session);
        });
  }

  bool load_segment_tasks(
      std::vector<vector_index_metadata_store::segment_task_row> *rows)
      override {
    if (rows == nullptr) return false;
    std::string payload;
    bool found = false;
    if (!load_structured_artifact(kSegmentTasksArtifactName, &payload,
                                  &found)) {
      return false;
    }
    if (!found) {
      rows->clear();
      return true;
    }
    if (payload.empty()) return false;
    return vector_index_metadata_store::deserialize_segment_task_rows(payload,
                                                                      rows);
  }

  bool save_segment_tasks(
      const std::vector<vector_index_metadata_store::segment_task_row> &rows)
      override {
    DBUG_EXECUTE_IF("vector_truth_store_fail_save", return false;);
    std::string payload;
    if (!vector_index_metadata_store::serialize_segment_task_rows(rows,
                                                                  &payload)) {
      return false;
    }
    return save_artifact(kSegmentTasksArtifactName, payload);
  }

  bool quarantine_segment_tasks() override {
    return quarantine_artifact(kSegmentTasksArtifactName);
  }

  void shutdown() {
    std::lock_guard<std::mutex> guard(m_mutex);
    for (const auto &entry : m_persist_sessions) {
      innodb_vector_truth_store::rollback_session(entry.second);
      innodb_vector_truth_store::close_session(entry.second);
    }
    m_persist_sessions.clear();
  }

  bool debug_get_artifact(const char *artifact_name, std::string *payload) {
    if (payload == nullptr || !is_valid_artifact_name(artifact_name))
      return false;
    if (is_row_artifact_name(artifact_name)) {
      bool found = false;
      return load_row_artifact_payload(artifact_name, payload, &found);
    }
    return load_artifact(artifact_name, payload);
  }

  bool debug_set_artifact(const char *artifact_name,
                          const std::string &payload) {
    if (!is_valid_artifact_name(artifact_name)) return false;
    if (is_row_artifact_name(artifact_name)) {
      return save_row_artifact_payload(artifact_name, payload);
    }
    return save_artifact(artifact_name, payload);
  }

  bool debug_delete_artifact(const char *artifact_name) {
    if (!is_valid_artifact_name(artifact_name)) return false;
    if (is_row_artifact_name(artifact_name) &&
        !clear_row_artifact(artifact_name)) {
      return false;
    }
    const bool structured_deleted = is_row_artifact_name(artifact_name) ||
                                    delete_structured_artifact(artifact_name);
    return structured_deleted;
  }

 private:
  // Grouped InnoDB sessions are thread-affine. Keep each owner call under the
  // mutex so commit, rollback, or shutdown cannot close it concurrently.
  template <typename Fn>
  bool with_persist_session(Fn &&operation) {
    std::unique_lock<std::mutex> guard(m_mutex);
    const auto it = m_persist_sessions.find(std::this_thread::get_id());
    if (it == m_persist_sessions.end()) {
      guard.unlock();
      return operation(nullptr);
    }
    return operation(it->second);
  }

  bool persist_change_log(
      const std::vector<vector_index_metadata_store::change_log_row> &rows,
      change_log_persist_mode mode) {
    const bool diagnostics_enabled = vector_index_diagnostics::enabled();
    const auto serialize_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    std::vector<innodb_vector_truth_store::change_log_row> stored_rows;
    if (!valid_persisted_change_log_rows(rows) ||
        !to_innodb_change_log_rows(rows, &stored_rows)) {
      return false;
    }
    const uint64_t serialize_ms = vector_index_diagnostics::elapsed_ms_if(
        diagnostics_enabled, serialize_start);
    const auto save_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    const bool ok =
        with_persist_session([&](innodb_vector_truth_store::Session *session) {
          if (mode == change_log_persist_mode::kReplace) {
            return innodb_vector_truth_store::save_change_log_rows(stored_rows,
                                                                   session);
          }
          return innodb_vector_truth_store::append_change_log_delta(stored_rows,
                                                                    session);
        });
    if (diagnostics_enabled) {
      size_t payload_bytes = 0;
      for (const auto &row : stored_rows)
        payload_bytes += row.vector_payload.size();
      const char *artifact_name = mode == change_log_persist_mode::kReplace
                                      ? kChangeLogArtifactName
                                      : "changelog_delta";
      record_artifact_persist_event(backend_name(), artifact_name, rows.size(),
                                    payload_bytes, serialize_ms,
                                    vector_index_diagnostics::elapsed_ms_if(
                                        diagnostics_enabled, save_start),
                                    ok);
    }
    return ok;
  }

  bool load_row_artifact_payload(const char *artifact_name,
                                 std::string *payload, bool *found) {
    if (payload == nullptr || found == nullptr ||
        !is_row_artifact_name(artifact_name)) {
      return false;
    }

    switch (row_artifact_kind_from_name(artifact_name)) {
      case row_artifact_kind::kCommitted: {
        std::vector<innodb_vector_truth_store::committed_row> stored_rows;
        if (!with_persist_session(
                [&](innodb_vector_truth_store::Session *session) {
                  return innodb_vector_truth_store::load_committed_rows(
                      &stored_rows, found, session);
                })) {
          return false;
        }
        if (!*found) {
          payload->clear();
          return true;
        }
        if (extract_debug_payload(stored_rows, payload)) return true;
        std::vector<vector_index_metadata_store::committed_row> rows;
        return from_innodb_committed_rows(stored_rows, &rows) &&
               vector_index_metadata_store::serialize_committed_rows(rows,
                                                                     payload);
      }
      case row_artifact_kind::kChangeLog: {
        std::vector<innodb_vector_truth_store::change_log_row> stored_rows;
        if (!with_persist_session(
                [&](innodb_vector_truth_store::Session *session) {
                  return innodb_vector_truth_store::load_change_log_rows(
                      &stored_rows, found, session);
                })) {
          return false;
        }
        if (!*found) {
          payload->clear();
          return true;
        }
        if (extract_debug_payload(stored_rows, payload)) return true;
        std::vector<vector_index_metadata_store::change_log_row> rows;
        return from_innodb_change_log_rows(stored_rows, &rows) &&
               vector_index_metadata_store::serialize_change_log_rows(rows,
                                                                      payload);
      }
      case row_artifact_kind::kPrepared:
        break;
      case row_artifact_kind::kNone:
        return false;
    }

    std::vector<innodb_vector_truth_store::prepared_change_row> stored_rows;
    if (!with_persist_session([&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::load_prepared_rows(
              &stored_rows, found, session);
        })) {
      return false;
    }
    if (!*found) {
      payload->clear();
      return true;
    }
    if (extract_debug_payload(stored_rows, payload)) return true;
    std::vector<vector_index_metadata_store::prepared_change_row> rows;
    return from_innodb_prepared_rows(stored_rows, &rows) &&
           vector_index_metadata_store::serialize_prepared_rows(rows, payload);
  }

  bool save_row_artifact_payload(const char *artifact_name,
                                 const std::string &payload) {
    DBUG_EXECUTE_IF("vector_truth_store_fail_save", return false;);
    if (!is_row_artifact_name(artifact_name)) return false;

    switch (row_artifact_kind_from_name(artifact_name)) {
      case row_artifact_kind::kCommitted: {
        std::vector<vector_index_metadata_store::committed_row> rows;
        std::vector<innodb_vector_truth_store::committed_row> stored_rows;
        if (payload.empty() ||
            (!vector_index_metadata_store::deserialize_committed_rows(payload,
                                                                      &rows) ||
             !to_innodb_committed_rows(rows, &stored_rows))) {
          stored_rows.clear();
          if (!payload.empty())
            stored_rows.push_back(make_debug_committed_row(payload));
        }
        return with_persist_session(
            [&](innodb_vector_truth_store::Session *session) {
              return innodb_vector_truth_store::save_committed_rows(stored_rows,
                                                                    session);
            });
      }
      case row_artifact_kind::kChangeLog: {
        std::vector<vector_index_metadata_store::change_log_row> rows;
        std::vector<innodb_vector_truth_store::change_log_row> stored_rows;
        if (payload.empty() ||
            (!vector_index_metadata_store::deserialize_change_log_rows(payload,
                                                                       &rows) ||
             !to_innodb_change_log_rows(rows, &stored_rows))) {
          stored_rows.clear();
          if (!payload.empty())
            stored_rows.push_back(make_debug_change_log_row(payload));
        }
        return with_persist_session(
            [&](innodb_vector_truth_store::Session *session) {
              return innodb_vector_truth_store::save_change_log_rows(stored_rows,
                                                                     session);
            });
      }
      case row_artifact_kind::kPrepared:
        break;
      case row_artifact_kind::kNone:
        return false;
    }

    std::vector<vector_index_metadata_store::prepared_change_row> rows;
    std::vector<innodb_vector_truth_store::prepared_change_row> stored_rows;
    if (payload.empty() ||
        (!vector_index_metadata_store::deserialize_prepared_rows(payload,
                                                                 &rows) ||
         !to_innodb_prepared_rows(rows, &stored_rows))) {
      stored_rows.clear();
      if (!payload.empty())
        stored_rows.push_back(make_debug_prepared_row(payload));
    }
    return with_persist_session(
        [&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::save_prepared_rows(stored_rows,
                                                               session);
        });
  }

  bool clear_row_artifact(const char *artifact_name) {
    return save_row_artifact_payload(artifact_name, std::string());
  }

  bool load_quarantine_entries(quarantine_entries *entries) {
    if (entries == nullptr) return false;
    std::string quarantine_payload;
    bool quarantine_found = false;
    if (!load_structured_artifact(kQuarantineStoreArtifactName,
                                  &quarantine_payload, &quarantine_found)) {
      return false;
    }
    if (!quarantine_found) {
      entries->clear();
      return true;
    }
    return deserialize_quarantine_entries(quarantine_payload, entries);
  }

  bool save_quarantine_entries(const quarantine_entries &entries) {
    std::string quarantine_payload;
    if (!serialize_quarantine_entries(entries, &quarantine_payload))
      return false;
    return save_structured_artifact(kQuarantineStoreArtifactName,
                                    quarantine_payload);
  }

  bool load_source_artifact_payload(const char *artifact_name,
                                    std::string *payload, bool *found) {
    if (is_row_artifact_name(artifact_name)) {
      return load_row_artifact_payload(artifact_name, payload, found);
    }
    return load_structured_artifact(artifact_name, payload, found);
  }

  bool load_structured_artifact(const char *artifact_name, std::string *payload,
                                bool *found) {
    DBUG_EXECUTE_IF("vector_truth_store_fail_load_structured_artifact",
                    return false;);
    DBUG_EXECUTE_IF("vector_truth_store_force_structured_artifact_missing", {
      if (payload != nullptr) payload->clear();
      if (found != nullptr) *found = false;
      return true;
    };);
    if (payload == nullptr || found == nullptr) return false;
    return with_persist_session(
        [&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::load_artifact(
              artifact_name, payload, found, session);
        });
  }

  bool save_structured_artifact(const char *artifact_name,
                                const std::string &payload) {
    return with_persist_session(
        [&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::save_artifact(artifact_name, payload,
                                                          session);
        });
  }

  bool delete_structured_artifact(const char *artifact_name) {
    DBUG_EXECUTE_IF("vector_truth_store_fail_delete_structured_artifact",
                    return false;);
    return with_persist_session(
        [&](innodb_vector_truth_store::Session *session) {
          return innodb_vector_truth_store::delete_artifact(artifact_name,
                                                            session);
        });
  }

  bool load_artifact(const char *artifact_name, std::string *payload) {
    if (payload == nullptr) return false;

    bool found = false;
    if (!load_structured_artifact(artifact_name, payload, &found)) return false;
    if (found) return true;
    payload->clear();
    return true;
  }

  bool save_artifact(const char *artifact_name, const std::string &payload) {
    DBUG_EXECUTE_IF("vector_truth_store_fail_save", return false;);
    return save_structured_artifact(artifact_name, payload);
  }

  bool quarantine_artifact(const char *artifact_name) {
    std::string identity;
    if (!stage_quarantine(artifact_name, "segment_tasks_load_failed", 0,
                          &identity))
      return false;
    if (!delete_structured_artifact(artifact_name)) {
      (void)update_quarantine_state(
          identity,
          vector_index_truth_store::quarantine_state::kSourceUpdateFailed);
      return false;
    }
    return update_quarantine_state(
        identity, vector_index_truth_store::quarantine_state::kComplete);
  }

  mutable std::mutex m_mutex;
  std::unordered_map<std::thread::id, innodb_vector_truth_store::Session *>
      m_persist_sessions;
};

class file_truth_store final : public vector_index_truth_store::truth_store {
 public:
  const char *backend_name() const override { return "file"; }

  bool is_transactional() const override { return false; }

  bool stage_quarantine(const std::string &artifact_name,
                        const std::string &reason, uint64_t generation,
                        std::string *identity) override {
    if (identity == nullptr || !is_valid_artifact_name(artifact_name.c_str())) {
      return false;
    }
    std::string artifact_payload;
    bool artifact_found = false;
    if (!vector_index_metadata_store::load_raw_artifact(
            artifact_name, &artifact_payload, &artifact_found)) {
      return false;
    }
    if (!artifact_found) artifact_payload.clear();

    std::string quarantine_payload;
    bool quarantine_found = false;
    quarantine_entries entries;
    if (!vector_index_metadata_store::load_raw_artifact(
            kQuarantineStoreArtifactName, &quarantine_payload,
            &quarantine_found) ||
        (quarantine_found &&
         !deserialize_quarantine_entries(quarantine_payload, &entries)) ||
        !append_quarantine_entry(&entries, artifact_name, reason, generation,
                                 artifact_payload, identity) ||
        !serialize_quarantine_entries(entries, &quarantine_payload)) {
      return false;
    }
    return vector_index_metadata_store::save_raw_artifact(
        kQuarantineStoreArtifactName, quarantine_payload);
  }

  bool update_quarantine_state(
      const std::string &identity,
      vector_index_truth_store::quarantine_state state) override {
    std::string quarantine_payload;
    bool quarantine_found = false;
    quarantine_entries entries;
    if (!vector_index_metadata_store::load_raw_artifact(
            kQuarantineStoreArtifactName, &quarantine_payload,
            &quarantine_found) ||
        !quarantine_found ||
        !deserialize_quarantine_entries(quarantine_payload, &entries) ||
        !update_quarantine_entry_state(&entries, identity, state) ||
        !serialize_quarantine_entries(entries, &quarantine_payload)) {
      return false;
    }
    return vector_index_metadata_store::save_raw_artifact(
        kQuarantineStoreArtifactName, quarantine_payload);
  }

  bool load_metadata(
      std::vector<vector_index_metadata_store::metadata_row> *rows) override {
    return vector_index_metadata_store::load_all(rows);
  }

  bool save_metadata(
      const std::vector<vector_index_metadata_store::metadata_row> &rows)
      override {
    DBUG_EXECUTE_IF("vector_truth_store_fail_save", return false;);
    return vector_index_metadata_store::save_all(rows);
  }

  bool load_committed(
      std::vector<vector_index_metadata_store::committed_row> *rows) override {
    return vector_index_metadata_store::load_committed_all(rows);
  }

  bool save_committed(
      const std::vector<vector_index_metadata_store::committed_row> &rows)
      override {
    DBUG_EXECUTE_IF("vector_truth_store_fail_save", return false;);
    const bool diagnostics_enabled = vector_index_diagnostics::enabled();
    std::string payload;
    uint64_t serialize_ms = 0;
    if (diagnostics_enabled) {
      const auto serialize_start =
          vector_index_diagnostics::now_if(diagnostics_enabled);
      if (!vector_index_metadata_store::serialize_committed_rows(rows,
                                                                 &payload))
        return false;
      serialize_ms = vector_index_diagnostics::elapsed_ms_if(
          diagnostics_enabled, serialize_start);
    }
    const auto save_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    const bool ok = vector_index_metadata_store::save_committed_all(rows);
    if (diagnostics_enabled) {
      record_artifact_persist_event(backend_name(), kCommittedArtifactName,
                                    rows.size(), payload.size(), serialize_ms,
                                    vector_index_diagnostics::elapsed_ms_if(
                                        diagnostics_enabled, save_start),
                                    ok);
    }
    return ok;
  }

  bool load_manifest(vector_index_metadata_store::manifest_row *row) override {
    return vector_index_metadata_store::load_manifest(row);
  }

  bool save_manifest(
      const vector_index_metadata_store::manifest_row &row) override {
    DBUG_EXECUTE_IF("vector_truth_store_fail_save", return false;);
    return vector_index_metadata_store::save_manifest(row);
  }

  bool load_change_log(
      std::vector<vector_index_metadata_store::change_log_row> *rows) override {
    return vector_index_metadata_store::load_change_log(rows);
  }

  bool save_change_log(
      const std::vector<vector_index_metadata_store::change_log_row> &rows)
      override {
    DBUG_EXECUTE_IF("vector_truth_store_fail_save", return false;);
    const bool diagnostics_enabled = vector_index_diagnostics::enabled();
    std::string payload;
    uint64_t serialize_ms = 0;
    if (diagnostics_enabled) {
      const auto serialize_start =
          vector_index_diagnostics::now_if(diagnostics_enabled);
      if (!vector_index_metadata_store::serialize_change_log_rows(rows,
                                                                  &payload))
        return false;
      serialize_ms = vector_index_diagnostics::elapsed_ms_if(
          diagnostics_enabled, serialize_start);
    }
    const auto save_start =
        vector_index_diagnostics::now_if(diagnostics_enabled);
    const bool ok = vector_index_metadata_store::save_change_log(rows);
    if (diagnostics_enabled) {
      record_artifact_persist_event(backend_name(), kChangeLogArtifactName,
                                    rows.size(), payload.size(), serialize_ms,
                                    vector_index_diagnostics::elapsed_ms_if(
                                        diagnostics_enabled, save_start),
                                    ok);
    }
    return ok;
  }

  bool load_prepared(
      std::vector<vector_index_metadata_store::prepared_change_row> *rows)
      override {
    return vector_index_metadata_store::load_prepared(rows);
  }

  bool save_prepared(
      const std::vector<vector_index_metadata_store::prepared_change_row> &rows)
      override {
    DBUG_EXECUTE_IF("vector_truth_store_fail_save", return false;);
    return vector_index_metadata_store::save_prepared(rows);
  }

  bool load_segment_tasks(
      std::vector<vector_index_metadata_store::segment_task_row> *rows)
      override {
    return vector_index_metadata_store::load_segment_tasks(rows);
  }

  bool save_segment_tasks(
      const std::vector<vector_index_metadata_store::segment_task_row> &rows)
      override {
    DBUG_EXECUTE_IF("vector_truth_store_fail_save", return false;);
    return vector_index_metadata_store::save_segment_tasks(rows);
  }

  bool quarantine_segment_tasks() override {
    return vector_index_metadata_store::quarantine_segment_task_store();
  }
};

file_truth_store g_file_store;
mysql_truth_store g_mysql_store;
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
vector_index_truth_store::truth_store *g_override_store = nullptr;
#endif

}  // namespace

namespace vector_index_truth_store::detail {

truth_store *selected_backend() {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  if (g_override_store != nullptr) return g_override_store;
#endif
  if (current_thd != nullptr &&
      current_thd->system_thread == SYSTEM_THREAD_SERVER_INITIALIZE) {
    return &g_file_store;
  }
  if (current_thd != nullptr &&
      dd::bootstrap::DD_bootstrap_ctx::instance().get_stage() <
          dd::bootstrap::Stage::FINISHED) {
    return &g_file_store;
  }
  if (use_file_truth_store_backend()) return &g_file_store;
  return &g_mysql_store;
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
void set_override_store_for_testing(truth_store *store) {
  g_override_store = store;
}

void reset_override_store_for_testing() { g_override_store = nullptr; }
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

bool bootstrap_initialize_mysql_store(THD *thd) {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  if (g_override_store != nullptr) return true;
#endif
  if (use_file_truth_store_backend()) return true;
  return g_mysql_store.bootstrap_initialize(thd);
}

void shutdown_mysql_store() {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  if (g_override_store != nullptr) return;
#endif
  if (use_file_truth_store_backend()) return;
  g_mysql_store.shutdown();
}

bool is_truth_store_table_name_impl(const char *schema_name,
                                    const char *table_name) {
  return is_truth_store_table_name(schema_name, table_name);
}

bool mysql_debug_get_artifact(const std::string &artifact_name,
                              std::string *payload) {
  if (payload == nullptr || !is_valid_artifact_name(artifact_name.c_str()))
    return false;
  if (use_file_truth_store_backend()) return false;
  return g_mysql_store.debug_get_artifact(artifact_name.c_str(), payload);
}

bool mysql_debug_set_artifact(const std::string &artifact_name,
                              const std::string &payload) {
  if (!is_valid_artifact_name(artifact_name.c_str())) return false;
  if (use_file_truth_store_backend()) return false;
  return g_mysql_store.debug_set_artifact(artifact_name.c_str(), payload);
}

bool mysql_debug_delete_artifact(const std::string &artifact_name) {
  if (!is_valid_artifact_name(artifact_name.c_str())) return false;
  if (use_file_truth_store_backend()) return false;
  return g_mysql_store.debug_delete_artifact(artifact_name.c_str());
}

std::string sql_string_literal_impl(const char *text) {
  return sql_string_literal(text);
}

bool decode_hex_bytes_impl(const std::string &encoded, std::string *decoded) {
  return decode_hex_bytes(encoded, decoded);
}

bool deserialize_quarantine_entries_impl(
    const std::string &payload, std::vector<quarantine_record> *entries) {
  return deserialize_quarantine_entries(payload, entries);
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool split_tab_fields_impl(const std::string &line,
                           std::vector<std::string> *fields) {
  return split_tab_fields(line, fields);
}

const char *quarantine_state_name_impl(quarantine_state state) {
  return quarantine_state_name(state);
}

bool parse_quarantine_state_impl(const std::string &value,
                                 quarantine_state *state) {
  return parse_quarantine_state(value, state);
}

bool parse_uint64_impl(const std::string &value, uint64_t *result) {
  return parse_uint64(value, result);
}

uint64_t quarantine_payload_checksum_impl(const std::string &payload) {
  return quarantine_payload_checksum(payload);
}

bool serialize_quarantine_entries_impl(
    const std::vector<quarantine_record> &entries, std::string *payload) {
  return serialize_quarantine_entries(entries, payload);
}

bool append_quarantine_entry_impl(std::vector<quarantine_record> *entries,
                                  const std::string &artifact_name,
                                  const std::string &reason,
                                  uint64_t generation,
                                  const std::string &payload,
                                  std::string *identity) {
  return append_quarantine_entry(entries, artifact_name, reason, generation,
                                 payload, identity);
}

bool update_quarantine_entry_state_impl(std::vector<quarantine_record> *entries,
                                        const std::string &identity,
                                        quarantine_state state) {
  return update_quarantine_entry_state(entries, identity, state);
}

void record_artifact_persist_event_impl(const char *backend_name,
                                        const char *artifact_name,
                                        size_t row_count, size_t payload_bytes,
                                        uint64_t serialize_ms, uint64_t save_ms,
                                        bool ok) {
  record_artifact_persist_event(backend_name, artifact_name, row_count,
                                payload_bytes, serialize_ms, save_ms, ok);
}

bool use_file_truth_store_backend_impl() {
  return use_file_truth_store_backend();
}

bool vector_to_truth_payload_impl(const vector_index::vector_data &vector,
                                  uint32_t *dimension, std::string *payload) {
  return vector_to_truth_payload(vector, dimension, payload);
}

bool truth_payload_to_vector_impl(uint32_t dimension,
                                  const std::string &payload,
                                  vector_index::vector_data *vector) {
  return truth_payload_to_vector(dimension, payload, vector);
}

bool encode_truth_op_impl(vector_index_metadata_store::change_op op,
                          uint8_t *value) {
  return encode_truth_op(op, value);
}

bool decode_truth_op_impl(uint8_t value,
                          vector_index_metadata_store::change_op *op) {
  return decode_truth_op(value, op);
}

bool is_row_artifact_name_impl(const char *artifact_name) {
  return is_row_artifact_name(artifact_name);
}

bool is_valid_artifact_name_impl(const char *artifact_name) {
  return is_valid_artifact_name(artifact_name);
}

innodb_vector_truth_store::committed_row make_debug_committed_row_impl(
    const std::string &payload) {
  return make_debug_committed_row(payload);
}

innodb_vector_truth_store::change_log_row make_debug_change_log_row_impl(
    const std::string &payload) {
  return make_debug_change_log_row(payload);
}

innodb_vector_truth_store::prepared_change_row make_debug_prepared_row_impl(
    const std::string &payload) {
  return make_debug_prepared_row(payload);
}

bool extract_debug_payload_impl(
    const std::vector<innodb_vector_truth_store::committed_row> &rows,
    std::string *payload) {
  return extract_debug_payload(rows, payload);
}

bool extract_debug_payload_impl(
    const std::vector<innodb_vector_truth_store::change_log_row> &rows,
    std::string *payload) {
  return extract_debug_payload(rows, payload);
}

bool extract_debug_payload_impl(
    const std::vector<innodb_vector_truth_store::prepared_change_row> &rows,
    std::string *payload) {
  return extract_debug_payload(rows, payload);
}

bool to_innodb_committed_rows_impl(
    const std::vector<vector_index_metadata_store::committed_row> &rows,
    std::vector<innodb_vector_truth_store::committed_row> *out) {
  return to_innodb_committed_rows(rows, out);
}

bool from_innodb_committed_rows_impl(
    const std::vector<innodb_vector_truth_store::committed_row> &rows,
    std::vector<vector_index_metadata_store::committed_row> *out) {
  return from_innodb_committed_rows(rows, out);
}

bool to_innodb_change_log_rows_impl(
    const std::vector<vector_index_metadata_store::change_log_row> &rows,
    std::vector<innodb_vector_truth_store::change_log_row> *out) {
  return to_innodb_change_log_rows(rows, out);
}

bool to_innodb_truth_delta_rows_impl(
    const std::vector<vector_index_metadata_store::change_log_row> &rows,
    std::vector<innodb_vector_truth_store::change_log_row> *out) {
  return to_innodb_truth_delta_rows(rows, out);
}

bool from_innodb_change_log_rows_impl(
    const std::vector<innodb_vector_truth_store::change_log_row> &rows,
    std::vector<vector_index_metadata_store::change_log_row> *out) {
  return from_innodb_change_log_rows(rows, out);
}

bool to_innodb_prepared_rows_impl(
    const std::vector<vector_index_metadata_store::prepared_change_row> &rows,
    std::vector<innodb_vector_truth_store::prepared_change_row> *out) {
  return to_innodb_prepared_rows(rows, out);
}

bool from_innodb_prepared_rows_impl(
    const std::vector<innodb_vector_truth_store::prepared_change_row> &rows,
    std::vector<vector_index_metadata_store::prepared_change_row> *out) {
  return from_innodb_prepared_rows(rows, out);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index_truth_store::detail

namespace vector_index_truth_store {

bool truth_store::for_each_committed(
    const std::function<
        bool(const vector_index_metadata_store::committed_row &row)> &visitor) {
  if (!visitor) return false;

  std::vector<vector_index_metadata_store::committed_row> rows;
  if (!load_committed(&rows)) return false;

  for (const auto &row : rows) {
    if (!visitor(row)) return false;
  }

  return true;
}

bool truth_store::for_each_committed(
    const std::string &index_name,
    const std::function<
        bool(const vector_index_metadata_store::committed_row &row)> &visitor) {
  if (index_name.empty() || !visitor) return false;
  return for_each_committed(
      [&](const vector_index_metadata_store::committed_row &row) {
        if (row.index_name != index_name) return true;
        return visitor(row);
      });
}

bool truth_store::find_committed(
    const std::string &index_name, uint64_t doc_id,
    vector_index_metadata_store::committed_row *row, bool *found) {
  if (index_name.empty() || row == nullptr || found == nullptr) return false;
  *row = vector_index_metadata_store::committed_row();
  *found = false;
  const bool scan_ok = for_each_committed(
      index_name, [&](const vector_index_metadata_store::committed_row &entry) {
        if (entry.doc_id != doc_id) return true;
        *row = entry;
        *found = true;
        return false;
      });
  return scan_ok || *found;
}

}  // namespace vector_index_truth_store
