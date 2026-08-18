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

#include "sql/vector/vector_index_metadata_store_internal.h"

#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace vector_index_metadata_store::detail {

bool deserialize_metadata_rows_impl(const std::string &payload,
                                    std::vector<metadata_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();
  if (payload.empty()) return true;

  std::string line;
  bool header_checked = false;
  std::istringstream stream(payload);
  while (std::getline(stream, line)) {
    if (line.empty()) continue;
    if (!header_checked) {
      if (line != kMetadataHeaderV1) return false;
      header_checked = true;
      continue;
    }

    std::vector<std::string> fields;
    split_tab_fields(line, &fields);
    if (fields.size() != 38) return false;

    auto parse_uint32_field = [](const std::string &field, uint32_t *value) {
      uint64_t parsed = 0;
      if (!parse_uint64(field, &parsed) ||
          parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
      }
      *value = static_cast<uint32_t>(parsed);
      return true;
    };

    metadata_row row;
    if (!decode_hex(fields[0], &row.index_name) || row.index_name.empty()) {
      return false;
    }

    uint64_t dimension = 0;
    if (!parse_uint64(fields[1], &dimension) || dimension == 0) return false;
    row.dimension = static_cast<size_t>(dimension);

    if (!vector_index::parse_metric(fields[2], &row.metric)) return false;
    if (!vector_index::parse_backend_mode(fields[3], &row.mode)) return false;
    if (!vector_index::parse_backend_provider(fields[4], &row.provider)) {
      return false;
    }
    if (!vector_index::parse_index_consistency_mode(fields[5],
                                                    &row.consistency_mode)) {
      return false;
    }
    row.lifecycle_state = fields[6];
    if (row.lifecycle_state.empty()) return false;
    if (!parse_uint64(fields[7], &row.lifecycle_version) ||
        row.lifecycle_version == 0) {
      return false;
    }
    if (!parse_uint32_field(fields[8], &row.last_error_code)) return false;
    if (!parse_uint64(fields[9], &row.last_error_ts)) return false;
    if (!parse_uint64(fields[10], &row.recover_fallback_count)) return false;
    if (!parse_uint64(fields[11], &row.last_recover_fallback_ts)) return false;
    if (!decode_hex(fields[12], &row.schema_name)) return false;
    if (!decode_hex(fields[13], &row.table_name)) return false;
    if (!decode_hex(fields[14], &row.column_name)) return false;
    if (!decode_hex(fields[15], &row.doc_id_column_name)) return false;
    if (!parse_uint32_field(fields[16], &row.search_ef)) return false;
    if (!parse_uint32_field(fields[17], &row.hnsw_m)) return false;
    if (!parse_uint32_field(fields[18], &row.hnsw_ef_construction)) {
      return false;
    }
    if (!parse_uint32_field(fields[19], &row.hnsw_build_threads)) return false;
    if (!parse_uint32_field(fields[20], &row.faiss_nlist)) return false;
    if (!parse_uint32_field(fields[21], &row.faiss_nprobe)) return false;
    if (!parse_uint32_field(fields[22], &row.faiss_pq_m)) return false;
    if (!parse_uint32_field(fields[23], &row.faiss_pq_bits)) return false;
    if (!parse_uint32_field(fields[24], &row.diskann_max_degree)) return false;
    if (!parse_uint32_field(fields[25], &row.diskann_build_complexity)) {
      return false;
    }
    if (!parse_uint32_field(fields[26], &row.diskann_search_complexity)) {
      return false;
    }
    if (!parse_uint32_field(fields[27], &row.diskann_search_beamwidth)) {
      return false;
    }
    if (!vector_index::valid_optional_diskann_search_beamwidth(
            row.diskann_search_beamwidth)) {
      return false;
    }
    if (!parse_uint64(fields[28], &row.diskann_pq_code_budget_size)) {
      return false;
    }
    if (!parse_uint32_field(fields[29], &row.diskann_disk_pq_dims)) {
      return false;
    }
    if (fields[30] == "0") {
      row.diskann_accelerate_build = false;
    } else if (fields[30] == "1") {
      row.diskann_accelerate_build = true;
    } else {
      return false;
    }
    if (fields[31] == "0") {
      row.diskann_shuffle_build = false;
    } else if (fields[31] == "1") {
      row.diskann_shuffle_build = true;
    } else {
      return false;
    }
    if (fields[32] == "0") {
      row.diskann_use_bfs_cache = false;
    } else if (fields[32] == "1") {
      row.diskann_use_bfs_cache = true;
    } else {
      return false;
    }
    if (!parse_uint32_field(fields[33], &row.diskann_build_threads)) {
      return false;
    }
    if (!parse_uint32_field(fields[34], &row.faiss_build_threads)) return false;
    if (!vector_index::parse_diskann_build_mode(
            fields[35], &row.diskann_build_mode_value)) {
      return false;
    }
    if (fields[36] == "0") {
      row.diskann_build_mode_specified = false;
    } else if (fields[36] == "1") {
      row.diskann_build_mode_specified = true;
    } else {
      return false;
    }
    if (!decode_hex(fields[37], &row.owner_schema) ||
        row.owner_schema.empty()) {
      return false;
    }

    rows->push_back(std::move(row));
  }

  return !stream.bad();
}

bool serialize_metadata_rows_impl(const std::vector<metadata_row> &rows,
                                  std::string *payload) {
  if (payload == nullptr) return false;
  payload->clear();
  std::ostringstream stream;
  stream << kMetadataHeaderV1 << "\n";
  for (const metadata_row &row : rows) {
    if (row.owner_schema.empty() ||
        !vector_index::valid_optional_diskann_search_beamwidth(
            row.diskann_search_beamwidth)) {
      return false;
    }
    stream << encode_hex(row.index_name) << "\t" << row.dimension << "\t"
           << vector_index::metric_to_string(row.metric) << "\t"
           << vector_index::backend_mode_to_string(row.mode) << "\t"
           << vector_index::backend_provider_to_string(row.provider) << "\t"
           << vector_index::index_consistency_mode_to_string(
                  row.consistency_mode)
           << "\t" << row.lifecycle_state << "\t" << row.lifecycle_version
           << "\t" << row.last_error_code << "\t" << row.last_error_ts << "\t"
           << row.recover_fallback_count << "\t" << row.last_recover_fallback_ts
           << "\t" << encode_hex(row.schema_name) << "\t"
           << encode_hex(row.table_name) << "\t" << encode_hex(row.column_name)
           << "\t" << encode_hex(row.doc_id_column_name) << "\t"
           << row.search_ef << "\t" << row.hnsw_m << "\t"
           << row.hnsw_ef_construction << "\t" << row.hnsw_build_threads << "\t"
           << row.faiss_nlist << "\t" << row.faiss_nprobe << "\t"
           << row.faiss_pq_m << "\t" << row.faiss_pq_bits << "\t"
           << row.diskann_max_degree << "\t" << row.diskann_build_complexity
           << "\t" << row.diskann_search_complexity << "\t"
           << row.diskann_search_beamwidth << "\t"
           << row.diskann_pq_code_budget_size << "\t"
           << row.diskann_disk_pq_dims << "\t"
           << (row.diskann_accelerate_build ? 1 : 0) << "\t"
           << (row.diskann_shuffle_build ? 1 : 0) << "\t"
           << (row.diskann_use_bfs_cache ? 1 : 0) << "\t"
           << row.diskann_build_threads << "\t" << row.faiss_build_threads
           << "\t"
           << vector_index::diskann_build_mode_to_string(
                  row.diskann_build_mode_value)
           << "\t" << (row.diskann_build_mode_specified ? 1 : 0) << "\t"
           << encode_hex(row.owner_schema)
           << "\n";
  }
  if (!stream) return false;
  *payload = stream.str();
  return true;
}

bool deserialize_committed_rows_impl(const std::string &payload,
                                     std::vector<committed_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();
  if (payload.empty()) return true;

  std::string line;
  bool header_checked = false;
  std::istringstream stream(payload);
  while (std::getline(stream, line)) {
    if (line.empty()) continue;
    if (!header_checked) {
      if (line != kCommittedHeaderV1) return false;
      header_checked = true;
      continue;
    }

    std::vector<std::string> fields;
    split_tab_fields(line, &fields);
    if (fields.size() != 4) return false;

    committed_row row;
    if (!decode_hex(fields[0], &row.index_name) || row.index_name.empty()) {
      return false;
    }
    if (!parse_uint64(fields[1], &row.doc_id)) return false;

    uint64_t dimension = 0;
    if (!parse_uint64(fields[2], &dimension)) return false;
    if (dimension > std::numeric_limits<size_t>::max() / sizeof(float)) {
      return false;
    }

    std::string vector_bytes;
    if (!decode_hex(fields[3], &vector_bytes)) return false;
    if (vector_bytes.size() != dimension * sizeof(float)) return false;

    row.vector.resize(static_cast<size_t>(dimension));
    if (!vector_bytes.empty()) {
      std::memcpy(row.vector.data(), vector_bytes.data(), vector_bytes.size());
    }
    rows->push_back(std::move(row));
  }

  return !stream.bad();
}

bool serialize_committed_rows_impl(const std::vector<committed_row> &rows,
                                   std::string *payload) {
  if (payload == nullptr) return false;
  payload->clear();
  std::ostringstream stream;
  stream << kCommittedHeaderV1 << "\n";
  for (const committed_row &row : rows) {
    const char *data = reinterpret_cast<const char *>(row.vector.data());
    const size_t bytes = row.vector.size() * sizeof(float);
    std::string vector_bytes;
    if (bytes > 0) vector_bytes.assign(data, bytes);

    stream << encode_hex(row.index_name) << "\t" << row.doc_id << "\t"
           << row.vector.size() << "\t" << encode_hex(vector_bytes) << "\n";
  }
  if (!stream) return false;
  *payload = stream.str();
  return true;
}

bool deserialize_manifest_row_impl(const std::string &payload, manifest_row *row) {
  if (row == nullptr) return false;

  *row = manifest_row();
  if (payload.empty()) return true;

  std::string line;
  bool header_checked = false;
  std::istringstream stream(payload);
  while (std::getline(stream, line)) {
    if (line.empty()) continue;
    if (!header_checked) {
      if (line != kManifestHeaderV1) return false;
      header_checked = true;
      continue;
    }

    std::vector<std::string> fields;
    split_tab_fields(line, &fields);
    if (fields.size() != 5) return false;

    row->state = fields[0];
    if (row->state.empty()) return false;
    if (!parse_uint64(fields[1], &row->version) || row->version == 0) {
      return false;
    }
    if (!parse_uint64(fields[2], &row->metadata_checkpoint)) return false;
    if (!parse_uint64(fields[3], &row->committed_checkpoint)) return false;
    if (!parse_uint64(fields[4], &row->change_log_checkpoint)) return false;
  }

  return !stream.bad();
}

bool serialize_manifest_row_impl(const manifest_row &row, std::string *payload) {
  if (payload == nullptr || row.state.empty() || row.version == 0) return false;
  std::ostringstream stream;
  stream << kManifestHeaderV1 << "\n";
  stream << row.state << "\t" << row.version << "\t"
         << row.metadata_checkpoint << "\t" << row.committed_checkpoint << "\t"
         << row.change_log_checkpoint << "\n";
  if (!stream) return false;
  *payload = stream.str();
  return true;
}

bool deserialize_change_log_rows_impl(const std::string &payload,
                                      std::vector<change_log_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();
  if (payload.empty()) return true;

  std::string line;
  bool header_checked = false;
  std::istringstream stream(payload);
  while (std::getline(stream, line)) {
    if (line.empty()) continue;
    if (!header_checked) {
      if (line != kChangeLogHeaderV1) return false;
      header_checked = true;
      continue;
    }

    std::vector<std::string> fields;
    split_tab_fields(line, &fields);
    if (fields.size() != 6) return false;

    change_log_row row;
    if (!parse_uint64(fields[0], &row.sequence) || row.sequence == 0) {
      return false;
    }
    if (!parse_uint64(fields[1], &row.txn_id)) return false;
    if (!parse_change_op(fields[2], &row.op)) return false;
    if (!decode_hex(fields[3], &row.index_name) || row.index_name.empty()) {
      return false;
    }
    if (!parse_uint64(fields[4], &row.doc_id)) return false;

    std::string vector_bytes;
    if (!decode_hex(fields[5], &vector_bytes)) return false;
    if ((vector_bytes.size() % sizeof(float)) != 0) return false;
    row.vector.resize(vector_bytes.size() / sizeof(float));
    if (!vector_bytes.empty()) {
      std::memcpy(row.vector.data(), vector_bytes.data(), vector_bytes.size());
    }
    if (row.op == change_op::kErase && !row.vector.empty()) return false;
    rows->push_back(std::move(row));
  }

  return !stream.bad();
}

bool serialize_change_log_rows_impl(const std::vector<change_log_row> &rows,
                                    std::string *payload) {
  if (payload == nullptr) return false;
  payload->clear();
  std::ostringstream stream;
  stream << kChangeLogHeaderV1 << "\n";
  for (const change_log_row &row : rows) {
    if (row.sequence == 0 || row.index_name.empty()) return false;
    if (row.op == change_op::kErase && !row.vector.empty()) return false;

    const char *data = reinterpret_cast<const char *>(row.vector.data());
    const size_t bytes = row.vector.size() * sizeof(float);
    std::string vector_bytes;
    if (bytes > 0) vector_bytes.assign(data, bytes);

    stream << row.sequence << "\t" << row.txn_id << "\t"
           << change_op_to_string(row.op) << "\t" << encode_hex(row.index_name)
           << "\t" << row.doc_id << "\t" << encode_hex(vector_bytes) << "\n";
  }
  if (!stream) return false;
  *payload = stream.str();
  return true;
}

bool deserialize_prepared_rows_impl(const std::string &payload,
                                    std::vector<prepared_change_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();
  if (payload.empty()) return true;

  std::string line;
  bool header_checked = false;
  std::istringstream stream(payload);
  while (std::getline(stream, line)) {
    if (line.empty()) continue;
    if (!header_checked) {
      if (line != kPreparedHeaderV1) return false;
      header_checked = true;
      continue;
    }

    std::vector<std::string> fields;
    split_tab_fields(line, &fields);
    if (fields.size() != 10) return false;

    prepared_change_row row;
    uint64_t u64 = 0;
    if (!parse_uint64(fields[0], &u64) ||
        u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    row.format_id = static_cast<int64_t>(u64);
    if (!parse_uint64(fields[1], &u64) ||
        u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    row.gtrid_length = static_cast<int64_t>(u64);
    if (!parse_uint64(fields[2], &u64) ||
        u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    row.bqual_length = static_cast<int64_t>(u64);
    if (!decode_hex(fields[3], &row.xid_data)) return false;
    if (static_cast<int64_t>(row.xid_data.size()) !=
        row.gtrid_length + row.bqual_length) {
      return false;
    }
    if (fields[4] == "0") {
      row.prepared_in_tc = false;
    } else if (fields[4] == "1") {
      row.prepared_in_tc = true;
    } else {
      return false;
    }
    if (!parse_uint64(fields[5], &row.txn_id)) return false;
    if (!parse_change_op(fields[6], &row.op)) return false;
    if (!decode_hex(fields[7], &row.index_name) || row.index_name.empty()) {
      return false;
    }
    if (!parse_uint64(fields[8], &row.doc_id)) return false;

    std::string vector_bytes;
    if (!decode_hex(fields[9], &vector_bytes)) return false;
    if ((vector_bytes.size() % sizeof(float)) != 0) return false;
    row.vector.resize(vector_bytes.size() / sizeof(float));
    if (!vector_bytes.empty()) {
      std::memcpy(row.vector.data(), vector_bytes.data(), vector_bytes.size());
    }
    if (row.op == change_op::kErase && !row.vector.empty()) return false;
    rows->push_back(std::move(row));
  }

  return !stream.bad();
}

bool serialize_prepared_rows_impl(const std::vector<prepared_change_row> &rows,
                                  std::string *payload) {
  if (payload == nullptr) return false;
  payload->clear();
  std::ostringstream stream;
  stream << kPreparedHeaderV1 << "\n";
  for (const prepared_change_row &row : rows) {
    if (row.format_id < 0 || row.gtrid_length < 0 || row.bqual_length < 0 ||
        row.index_name.empty()) {
      return false;
    }
    if (static_cast<int64_t>(row.xid_data.size()) !=
        row.gtrid_length + row.bqual_length) {
      return false;
    }
    if (row.op == change_op::kErase && !row.vector.empty()) return false;

    const char *data = reinterpret_cast<const char *>(row.vector.data());
    const size_t bytes = row.vector.size() * sizeof(float);
    std::string vector_bytes;
    if (bytes > 0) vector_bytes.assign(data, bytes);

    stream << static_cast<uint64_t>(row.format_id) << "\t"
           << static_cast<uint64_t>(row.gtrid_length) << "\t"
           << static_cast<uint64_t>(row.bqual_length) << "\t"
           << encode_hex(row.xid_data) << "\t"
           << (row.prepared_in_tc ? "1" : "0") << "\t" << row.txn_id << "\t"
           << change_op_to_string(row.op) << "\t" << encode_hex(row.index_name)
           << "\t" << row.doc_id << "\t" << encode_hex(vector_bytes) << "\n";
  }
  if (!stream) return false;
  *payload = stream.str();
  return true;
}

bool deserialize_segment_task_rows_impl(
    const std::string &payload, std::vector<segment_task_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();
  if (payload.empty()) return true;

  std::string line;
  bool header_checked = false;
  std::istringstream stream(payload);
  while (std::getline(stream, line)) {
    if (line.empty()) continue;
    if (!header_checked) {
      if (line != kSegmentTaskHeaderV1) return false;
      header_checked = true;
      continue;
    }

    std::vector<std::string> fields;
    split_tab_fields(line, &fields);
    if (fields.size() != 12) return false;

    segment_task_row row;
    if (!decode_hex(fields[0], &row.index_name) || row.index_name.empty()) {
      return false;
    }
    if (!parse_uint64(fields[1], &row.generation) || row.generation == 0) {
      return false;
    }
    if (!parse_uint64(fields[2], &row.segment_id)) return false;
    if (!parse_segment_task_state(fields[3], &row.state)) return false;
    if (!parse_uint64(fields[4], &row.row_count)) return false;
    if (!parse_uint64(fields[5], &row.payload_size)) return false;
    if (!decode_hex(fields[6], &row.vector_path)) return false;
    if (!decode_hex(fields[7], &row.docid_path)) return false;
    if (!decode_hex(fields[8], &row.artifact_prefix)) return false;

    uint64_t parsed = 0;
    if (!parse_uint64(fields[9], &parsed) ||
        parsed > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    row.attempt = static_cast<uint32_t>(parsed);
    if (!parse_uint64(fields[10], &parsed) ||
        parsed > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    row.last_error_code = static_cast<uint32_t>(parsed);
    if (!parse_uint64(fields[11], &row.updated_ts)) return false;

    rows->push_back(std::move(row));
  }

  return !stream.bad();
}

bool serialize_segment_task_rows_impl(
    const std::vector<segment_task_row> &rows, std::string *payload) {
  if (payload == nullptr) return false;
  payload->clear();
  std::ostringstream stream;
  stream << kSegmentTaskHeaderV1 << "\n";
  for (const segment_task_row &row : rows) {
    if (row.index_name.empty() || row.generation == 0) return false;

    stream << encode_hex(row.index_name) << "\t" << row.generation << "\t"
           << row.segment_id << "\t"
           << segment_task_state_to_string(row.state) << "\t"
           << row.row_count << "\t" << row.payload_size << "\t"
           << encode_hex(row.vector_path) << "\t"
           << encode_hex(row.docid_path) << "\t"
           << encode_hex(row.artifact_prefix) << "\t" << row.attempt << "\t"
           << row.last_error_code << "\t" << row.updated_ts << "\n";
  }
  if (!stream) return false;
  *payload = stream.str();
  return true;
}

}  // namespace vector_index_metadata_store::detail
