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

#include "sql/vector/vector_index_metadata_store.h"
#include "sql/vector/vector_index_metadata_store_internal.h"

#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "my_dbug.h"
#include "sql/handler.h"
#include "sql/mysqld.h"
#include "sql/xa.h"

namespace vector_index_metadata_store::detail {

constexpr const char *kStoreDirectory = "mysql_vector_index";
constexpr const char *kMetadataFilename = "metadata.v1";
constexpr const char *kCommittedHeaderV1 = "mysql-vector-committed-v1";
constexpr const char *kCommittedFilename = "committed.v1";
constexpr const char *kPreparedHeaderV1 = "mysql-vector-prepared-v1";
constexpr const char *kPreparedFilename = "prepared.v1";
constexpr const char *kManifestHeaderV1 = "mysql-vector-manifest-v1";
constexpr const char *kManifestFilename = "manifest.v1";
constexpr const char *kChangeLogHeaderV1 = "mysql-vector-changelog-v1";
constexpr const char *kChangeLogFilename = "changelog.v1";
constexpr const char *kSegmentTaskHeaderV1 = "mysql-vector-segment-task-v1";
constexpr const char *kSegmentTaskFilename = "segment_tasks.v1";

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
std::mutex g_path_mutex;
std::string g_path_override;
#endif

bool split_tab_fields(const std::string &line, std::vector<std::string> *fields) {
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

char hex_digit(unsigned value) {
  return static_cast<char>((value < 10U) ? ('0' + value) : ('a' + (value - 10U)));
}

bool hex_value(char ch, unsigned *value) {
  if (ch >= '0' && ch <= '9') {
    *value = static_cast<unsigned>(ch - '0');
    return true;
  }

  const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (lower >= 'a' && lower <= 'f') {
    *value = static_cast<unsigned>(lower - 'a' + 10);
    return true;
  }

  return false;
}

std::string encode_hex(const std::string &input) {
  std::string out;
  out.resize(input.size() * 2);
  for (size_t i = 0; i < input.size(); ++i) {
    const unsigned value = static_cast<unsigned>(
        static_cast<unsigned char>(input[i]));
    out[i * 2] = hex_digit((value >> 4) & 0x0FU);
    out[i * 2 + 1] = hex_digit(value & 0x0FU);
  }
  return out;
}

bool decode_hex(const std::string &encoded, std::string *decoded) {
  decoded->clear();
  if ((encoded.size() % 2) != 0) return false;
  decoded->reserve(encoded.size() / 2);
  for (size_t i = 0; i < encoded.size(); i += 2) {
    unsigned high = 0;
    unsigned low = 0;
    if (!hex_value(encoded[i], &high) || !hex_value(encoded[i + 1], &low))
      return false;
    const unsigned byte = (high << 4) | low;
    decoded->push_back(static_cast<char>(byte));
  }
  return true;
}

bool parse_uint64(const std::string &text, uint64_t *value) {
  if (value == nullptr || text.empty()) return false;
  for (const char character : text) {
    if (character < '0' || character > '9') return false;
  }
  errno = 0;
  const unsigned long long parsed = std::strtoull(text.c_str(), nullptr, 10);
  if (parsed == std::numeric_limits<unsigned long long>::max() &&
      errno == ERANGE) {
    return false;
  }
  *value = static_cast<uint64_t>(parsed);
  return true;
}

std::string store_path(const char *filename, const char *override_suffix) {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  std::lock_guard<std::mutex> guard(g_path_mutex);
  if (!g_path_override.empty()) return g_path_override + override_suffix;
#else
  (void)override_suffix;
#endif

  std::string path(mysql_real_data_home);
  if (!path.empty()) {
    const char tail = path.back();
    if (tail != '/' && tail != '\\') path.push_back('/');
  }
  path.append(kStoreDirectory);
  path.push_back('/');
  path.append(filename);
  return path;
}

std::string metadata_path() { return store_path(kMetadataFilename, ""); }

std::string committed_path() {
  return store_path(kCommittedFilename, ".committed");
}

std::string manifest_path() {
  return store_path(kManifestFilename, ".manifest");
}

std::string prepared_path() {
  return store_path(kPreparedFilename, ".prepared");
}

std::string change_log_path() {
  return store_path(kChangeLogFilename, ".changelog");
}

std::string segment_task_path() {
  return store_path(kSegmentTaskFilename, ".segment_tasks");
}

bool ensure_parent_directory(const std::string &path) {
  const std::filesystem::path file_path(path);
  const std::filesystem::path parent = file_path.parent_path();
  if (parent.empty()) return true;

  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  return !ec;
}

bool open_read_primary(const std::string &path, std::ifstream *file) {
  if (file == nullptr) return false;
  file->open(path, std::ios::in | std::ios::binary);
  return file->good();
}

bool remove_if_exists(const std::string &path) {
  DBUG_EXECUTE_IF("vector_metadata_store_fail_remove_if_exists", return false;);
  if (std::remove(path.c_str()) != 0 && errno != ENOENT) return false;
  return true;
}

bool write_payload_atomically(const std::string &path,
                              const std::string &payload) {
  if (!ensure_parent_directory(path)) return false;

  const std::string temp_path = path + ".tmp";
  std::ofstream file(temp_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.good()) return false;
  file << payload;
  file.close();
  if (!file) return false;

  if (std::rename(temp_path.c_str(), path.c_str()) != 0) {
    std::remove(temp_path.c_str());
    return false;
  }
  return true;
}

bool remove_payload_file(const std::string &path) {
  std::remove((path + ".tmp").c_str());
  return remove_if_exists(path);
}

bool quarantine_file_if_exists(const std::string &path) {
  const std::string quarantined = path + ".corrupt";
  std::remove(quarantined.c_str());
  if (std::rename(path.c_str(), quarantined.c_str()) == 0) return true;
  return errno == ENOENT;
}

bool raw_path_for_artifact(const std::string &artifact_name, std::string *path) {
  if (path == nullptr) return false;
  if (artifact_name == "metadata") {
    *path = metadata_path();
    return true;
  }
  if (artifact_name == "committed") {
    *path = committed_path();
    return true;
  }
  if (artifact_name == "manifest") {
    *path = manifest_path();
    return true;
  }
  if (artifact_name == "changelog") {
    *path = change_log_path();
    return true;
  }
  if (artifact_name == "prepared") {
    *path = prepared_path();
    return true;
  }
  if (artifact_name == "segment_tasks") {
    *path = segment_task_path();
    return true;
  }
  return false;
}

const char *change_op_to_string(vector_index_metadata_store::change_op op) {
  switch (op) {
    case vector_index_metadata_store::change_op::kUpsert:
      return "upsert";
    case vector_index_metadata_store::change_op::kErase:
      return "erase";
  }
  return "upsert";
}

bool parse_change_op(const std::string &text,
                     vector_index_metadata_store::change_op *op) {
  if (op == nullptr) return false;
  if (text == "upsert") {
    *op = vector_index_metadata_store::change_op::kUpsert;
    return true;
  }
  if (text == "erase") {
    *op = vector_index_metadata_store::change_op::kErase;
    return true;
  }
  return false;
}

}  // namespace vector_index_metadata_store::detail

namespace vector_index_metadata_store {

using namespace detail;

namespace {

bool xid_data_length(int64_t format_id, int64_t gtrid_length,
                     int64_t bqual_length, size_t *data_length) {
  if (data_length == nullptr) return false;
  if (format_id < 0 ||
      format_id > static_cast<int64_t>(std::numeric_limits<long>::max()) ||
      gtrid_length < 0 || gtrid_length > MAXGTRIDSIZE ||
      bqual_length < 0 || bqual_length > MAXBQUALSIZE) {
    return false;
  }

  *data_length = static_cast<size_t>(gtrid_length) +
                 static_cast<size_t>(bqual_length);
  return *data_length <= XIDDATASIZE;
}

bool valid_xid_fields(int64_t format_id, int64_t gtrid_length,
                      int64_t bqual_length, size_t stored_data_length) {
  size_t expected_data_length = 0;
  return xid_data_length(format_id, gtrid_length, bqual_length,
                         &expected_data_length) &&
         expected_data_length == stored_data_length;
}

bool valid_xid(const xid_t &xid, size_t *data_length) {
  return xid_data_length(xid.get_format_id(), xid.get_gtrid_length(),
                         xid.get_bqual_length(), data_length);
}

}  // namespace

bool valid_prepared_xid(const prepared_change_row &row) {
  return valid_xid_fields(row.format_id, row.gtrid_length, row.bqual_length,
                          row.xid_data.size());
}

bool set_prepared_xid(const xid_t &xid, prepared_change_row *row) {
  size_t data_length = 0;
  if (row == nullptr || !valid_xid(xid, &data_length)) return false;

  row->format_id = xid.get_format_id();
  row->gtrid_length = xid.get_gtrid_length();
  row->bqual_length = xid.get_bqual_length();
  row->xid_data.assign(xid.get_data(), data_length);
  return true;
}

bool get_prepared_xid(const prepared_change_row &row, xid_t *xid) {
  if (xid == nullptr || !valid_prepared_xid(row)) return false;

  xid->reset();
  xid->set_format_id(static_cast<long>(row.format_id));
  xid->set_gtrid_length(static_cast<long>(row.gtrid_length));
  xid->set_bqual_length(static_cast<long>(row.bqual_length));
  if (!row.xid_data.empty()) {
    xid->set_data(row.xid_data.data(), static_cast<long>(row.xid_data.size()));
  }
  return true;
}

bool prepared_xid_matches(const prepared_change_row &row, const xid_t &xid) {
  size_t xid_data_size = 0;
  if (!valid_prepared_xid(row) || !valid_xid(xid, &xid_data_size) ||
      row.xid_data.size() != xid_data_size ||
      row.format_id != xid.get_format_id() ||
      row.gtrid_length != xid.get_gtrid_length() ||
      row.bqual_length != xid.get_bqual_length()) {
    return false;
  }
  return row.xid_data.empty() ||
         std::memcmp(row.xid_data.data(), xid.get_data(), row.xid_data.size()) ==
             0;
}

bool deserialize_metadata_rows(const std::string &payload,
                             std::vector<metadata_row> *rows) {
  return detail::deserialize_metadata_rows_impl(payload, rows);
}

bool serialize_metadata_rows(const std::vector<metadata_row> &rows,
                           std::string *payload) {
  return detail::serialize_metadata_rows_impl(rows, payload);
}

bool load_all(std::vector<metadata_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();

  const std::string primary_path = metadata_path();
  std::ifstream file;
  if (!open_read_primary(primary_path, &file)) return true;
  std::string payload((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  if (file.bad()) return false;
  return deserialize_metadata_rows(payload, rows);
}

bool save_all(const std::vector<metadata_row> &rows) {
  const std::string path = metadata_path();
  if (rows.empty()) return remove_payload_file(path);

  std::string payload;
  if (!serialize_metadata_rows(rows, &payload)) return false;
  return write_payload_atomically(path, payload);
}

bool deserialize_committed_rows(const std::string &payload,
                              std::vector<committed_row> *rows) {
  return detail::deserialize_committed_rows_impl(payload, rows);
}

bool serialize_committed_rows(const std::vector<committed_row> &rows,
                            std::string *payload) {
  return detail::serialize_committed_rows_impl(rows, payload);
}

bool load_committed_all(std::vector<committed_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();

  const std::string primary_path = committed_path();
  std::ifstream file;
  if (!open_read_primary(primary_path, &file)) return true;
  std::string payload((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  if (file.bad()) return false;
  return deserialize_committed_rows(payload, rows);
}

bool save_committed_all(const std::vector<committed_row> &rows) {
  const std::string path = committed_path();
  if (rows.empty()) return remove_payload_file(path);

  std::string payload;
  if (!serialize_committed_rows(rows, &payload)) return false;
  return write_payload_atomically(path, payload);
}

bool deserialize_manifest_row(const std::string &payload, manifest_row *row) {
  return detail::deserialize_manifest_row_impl(payload, row);
}

bool serialize_manifest_row(const manifest_row &row, std::string *payload) {
  return detail::serialize_manifest_row_impl(row, payload);
}

bool load_manifest(manifest_row *row) {
  if (row == nullptr) return false;

  *row = manifest_row();
  const std::string primary_path = manifest_path();
  std::ifstream file;
  if (!open_read_primary(primary_path, &file)) return true;
  std::string payload((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  if (file.bad()) return false;
  return deserialize_manifest_row(payload, row);
}

bool save_manifest(const manifest_row &row) {
  if (row.state.empty() || row.version == 0) return false;

  const std::string path = manifest_path();
  std::string payload;
  if (!serialize_manifest_row(row, &payload)) return false;
  return write_payload_atomically(path, payload);
}

bool deserialize_change_log_rows(const std::string &payload,
                              std::vector<change_log_row> *rows) {
  return detail::deserialize_change_log_rows_impl(payload, rows);
}

bool serialize_change_log_rows(const std::vector<change_log_row> &rows,
                            std::string *payload) {
  return detail::serialize_change_log_rows_impl(rows, payload);
}

bool load_change_log(std::vector<change_log_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();

  const std::string primary_path = change_log_path();
  std::ifstream file;
  if (!open_read_primary(primary_path, &file)) return true;
  std::string payload((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  if (file.bad()) return false;
  return deserialize_change_log_rows(payload, rows);
}

bool save_change_log(const std::vector<change_log_row> &rows) {
  const std::string path = change_log_path();
  if (rows.empty()) return remove_payload_file(path);

  std::string payload;
  if (!serialize_change_log_rows(rows, &payload)) return false;
  return write_payload_atomically(path, payload);
}

bool deserialize_prepared_rows(const std::string &payload,
                             std::vector<prepared_change_row> *rows) {
  return detail::deserialize_prepared_rows_impl(payload, rows);
}

bool serialize_prepared_rows(const std::vector<prepared_change_row> &rows,
                           std::string *payload) {
  return detail::serialize_prepared_rows_impl(rows, payload);
}

bool load_prepared(std::vector<prepared_change_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();

  const std::string primary_path = prepared_path();
  std::ifstream file;
  if (!open_read_primary(primary_path, &file)) return true;
  std::string payload((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  if (file.bad()) return false;
  return deserialize_prepared_rows(payload, rows);
}

bool save_prepared(const std::vector<prepared_change_row> &rows) {
  const std::string path = prepared_path();
  if (rows.empty()) return remove_payload_file(path);

  std::string payload;
  if (!serialize_prepared_rows(rows, &payload)) return false;
  return write_payload_atomically(path, payload);
}

bool deserialize_segment_task_rows(const std::string &payload,
                                   std::vector<segment_task_row> *rows) {
  return detail::deserialize_segment_task_rows_impl(payload, rows);
}

bool serialize_segment_task_rows(const std::vector<segment_task_row> &rows,
                                 std::string *payload) {
  return detail::serialize_segment_task_rows_impl(rows, payload);
}

bool load_segment_tasks(std::vector<segment_task_row> *rows) {
  if (rows == nullptr) return false;
  rows->clear();

  const std::string primary_path = segment_task_path();
  std::ifstream file;
  if (!open_read_primary(primary_path, &file)) return true;
  std::string payload((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  if (file.bad()) return false;
  return deserialize_segment_task_rows(payload, rows);
}

bool save_segment_tasks(const std::vector<segment_task_row> &rows) {
  const std::string path = segment_task_path();
  if (rows.empty()) return remove_payload_file(path);

  std::string payload;
  if (!serialize_segment_task_rows(rows, &payload)) return false;
  return write_payload_atomically(path, payload);
}

bool load_raw_artifact(const std::string &artifact_name, std::string *payload,
                     bool *found) {
  if (payload == nullptr || found == nullptr) return false;
  payload->clear();
  *found = false;

  std::string path;
  if (!raw_path_for_artifact(artifact_name, &path)) return false;

  std::ifstream file;
  if (!open_read_primary(path, &file)) return true;
  *payload = std::string((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  if (file.bad()) return false;
  *found = true;
  return true;
}

bool save_raw_artifact(const std::string &artifact_name,
                     const std::string &payload) {
  std::string path;
  if (!raw_path_for_artifact(artifact_name, &path)) return false;

  if (payload.empty()) {
    return delete_raw_artifact(artifact_name);
  }

  return write_payload_atomically(path, payload);
}

bool delete_raw_artifact(const std::string &artifact_name) {
  std::string path;
  if (!raw_path_for_artifact(artifact_name, &path)) return false;
  if (!remove_if_exists(path)) return false;
  return true;
}

bool quarantine_current_store() {
  return quarantine_file_if_exists(metadata_path());
}

bool quarantine_committed_store() {
  return quarantine_file_if_exists(committed_path());
}

bool quarantine_manifest_store() {
  return quarantine_file_if_exists(manifest_path());
}

bool quarantine_change_log_store() {
  return quarantine_file_if_exists(change_log_path());
}

bool quarantine_prepared_store() {
  return quarantine_file_if_exists(prepared_path());
}

bool quarantine_segment_task_store() {
  return quarantine_file_if_exists(segment_task_path());
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
void set_path_for_testing(const std::string &path) {
  std::lock_guard<std::mutex> guard(g_path_mutex);
  g_path_override = path;
}

void reset_path_for_testing() {
  std::lock_guard<std::mutex> guard(g_path_mutex);
  g_path_override.clear();
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index_metadata_store
