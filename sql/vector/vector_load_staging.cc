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

#include "sql/vector/vector_load_staging.h"

#include <fcntl.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "my_sys.h"
#include "sql/vector/vector_index_backend_internal.h"

namespace vector_index {
namespace {

constexpr uint64_t k_load_staging_payload = 3;
constexpr uint64_t k_fnv_offset_basis = 1469598103934665603ULL;
constexpr uint64_t k_fnv_prime = 1099511628211ULL;
constexpr char k_load_staging_directory[] = "load_staging";
constexpr char k_csv_filename[] = "vectors.csv";
constexpr char k_fbin_filename[] = "vectors.fbin";
constexpr char k_docid_filename[] = "docids.u64";

enum load_staging_unsigned_field : size_t {
  k_payload_type = 0,
  k_replace_duplicates,
  k_rebuild_after_load,
  k_dimension,
  k_vector_size,
  k_vector_checksum,
  k_has_docids,
  k_docid_size,
  k_docid_checksum,
  k_unsigned_field_count,
};

enum load_staging_string_field : size_t {
  k_artifact_identity = 0,
  k_format,
  k_string_field_count,
};

std::mutex g_active_staging_mutex;
std::unordered_set<std::string> g_active_staging_directories;

void set_error(std::string *error, const std::string &message) {
  if (error != nullptr) *error = message;
}

bool valid_format(const std::string &format) {
  return format == "FBIN" || format == "CSV";
}

bool valid_identity(const load_staging_identity &identity) {
  return !identity.index_name.empty() && identity.publication_id != 0;
}

std::string staging_root_path() {
  const std::string root = detail::vector_index_root_path();
  if (root.empty()) return "";
  return (std::filesystem::path(root) / k_load_staging_directory).string();
}

std::string artifact_identity(const load_staging_identity &identity) {
  if (!valid_identity(identity)) return "";
  return detail::encoded_index_name(identity.index_name) + "-" +
         std::to_string(identity.publication_id);
}

std::string staging_directory(const load_staging_identity &identity) {
  const std::string root = staging_root_path();
  const std::string name = artifact_identity(identity);
  if (root.empty() || name.empty()) return "";
  return (std::filesystem::path(root) / name).string();
}

void fill_staging_paths(const std::string &directory, const std::string &format,
                        load_staging_paths *paths) {
  const std::filesystem::path root(directory);
  paths->vector_filename =
      (root / (format == "FBIN" ? k_fbin_filename : k_csv_filename)).string();
  paths->docid_filename =
      format == "FBIN" ? (root / k_docid_filename).string() : "";
}

bool resolve_paths(const load_staging_identity &identity,
                   const std::string &format, load_staging_paths *paths) {
  if (paths == nullptr || !valid_format(format)) return false;
  const std::string directory = staging_directory(identity);
  if (directory.empty()) return false;
  fill_staging_paths(directory, format, paths);
  return true;
}

bool resolve_paths_in_directory(const std::string &directory,
                                const std::string &format,
                                load_staging_paths *paths) {
  if (directory.empty() || paths == nullptr || !valid_format(format)) {
    return false;
  }
  fill_staging_paths(directory, format, paths);
  return true;
}

bool is_managed_staging_name(const std::string &name) {
  const bool temporary =
      name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0;
  const std::string_view base =
      temporary ? std::string_view(name).substr(0, name.size() - 4)
                : std::string_view(name);
  const size_t separator = base.rfind('-');
  if (separator == std::string_view::npos || separator == 0 ||
      separator + 1 == base.size()) {
    return false;
  }
  for (size_t i = 0; i < separator; ++i) {
    const char ch = base[i];
    if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) {
      return false;
    }
  }
  for (size_t i = separator + 1; i < base.size(); ++i) {
    if (base[i] < '0' || base[i] > '9') return false;
  }
  return true;
}

bool sync_path(const std::filesystem::path &path) {
#ifdef _WIN32
  (void)path;
  return true;
#else
  const File fd = my_open(path.string().c_str(), O_RDONLY, MYF(0));
  if (fd < 0) return false;
  const bool synced = my_sync(fd, MYF(0)) == 0;
  const bool closed = my_close(fd, MYF(0)) == 0;
  return synced && closed;
#endif
}

bool sync_parent_directory(const std::filesystem::path &path) {
#ifdef _WIN32
  (void)path;
  return true;
#else
  const std::filesystem::path parent = path.parent_path();
  return !parent.empty() && sync_path(parent);
#endif
}

bool durable_copy(const std::string &source,
                  const std::filesystem::path &target, std::string *error) {
  if (source.empty()) {
    set_error(error, "LOAD VECTOR DATA staging source is empty");
    return false;
  }
  std::error_code source_ec;
  if (!std::filesystem::is_regular_file(source, source_ec) || source_ec) {
    set_error(error, "LOAD VECTOR DATA staging source is not a regular file");
    return false;
  }
  const std::filesystem::path temporary = target.string() + ".tmp";
  std::error_code ec;
  std::filesystem::remove(temporary, ec);
  if (my_copy(source.c_str(), temporary.string().c_str(), MYF(MY_SYNC)) != 0) {
    set_error(error, "LOAD VECTOR DATA could not copy staging input");
    return false;
  }
  std::filesystem::rename(temporary, target, ec);
  if (ec || !sync_parent_directory(target)) {
    std::filesystem::remove(temporary, ec);
    set_error(error, "LOAD VECTOR DATA could not publish staging input");
    return false;
  }
  return true;
}

bool checksum_file(const std::string &path, uint64_t *size, uint64_t *checksum,
                   std::string *error) {
  if (size == nullptr || checksum == nullptr) return false;
  *size = 0;
  *checksum = k_fnv_offset_basis;
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    set_error(error, "LOAD VECTOR DATA staging artifact is missing");
    return false;
  }
  std::array<char, 64 * 1024> buffer{};
  while (file.good()) {
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = file.gcount();
    if (count < 0 || static_cast<uint64_t>(count) >
                         std::numeric_limits<uint64_t>::max() - *size) {
      set_error(error, "LOAD VECTOR DATA staging artifact is too large");
      return false;
    }
    *size += static_cast<uint64_t>(count);
    for (std::streamsize offset = 0; offset < count; ++offset) {
      *checksum ^= static_cast<unsigned char>(buffer[offset]);
      *checksum *= k_fnv_prime;
    }
  }
  if (!file.eof()) {
    set_error(error, "LOAD VECTOR DATA staging artifact could not be read");
    return false;
  }
  return true;
}

bool verify_file(const std::string &path, uint64_t expected_size,
                 uint64_t expected_checksum, std::string *error) {
  std::error_code ec;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, ec);
  if (ec || status.type() != std::filesystem::file_type::regular) {
    set_error(error, "LOAD VECTOR DATA staging artifact is not a regular file");
    return false;
  }
  uint64_t size = 0;
  uint64_t checksum = 0;
  if (!checksum_file(path, &size, &checksum, error)) return false;
  if (size != expected_size) {
    set_error(error, "LOAD VECTOR DATA staging artifact size mismatch");
    return false;
  }
  if (checksum != expected_checksum) {
    set_error(error, "LOAD VECTOR DATA staging artifact checksum mismatch");
    return false;
  }
  return true;
}

void release_active_directory(const std::string &directory) {
  std::lock_guard<std::mutex> guard(g_active_staging_mutex);
  g_active_staging_directories.erase(directory);
}

void release_active_artifact(const load_staging_identity &identity) {
  const std::string directory = staging_directory(identity);
  if (!directory.empty()) release_active_directory(directory);
}

bool remove_artifact_directory(const load_staging_identity &identity) {
  const std::string directory = staging_directory(identity);
  if (directory.empty()) return false;
  release_active_directory(directory);
  std::error_code ec;
  if (!std::filesystem::exists(directory, ec)) return !ec;
  if (ec) return false;
  std::filesystem::remove_all(directory, ec);
  return !ec && sync_parent_directory(directory);
}

bool is_bulk_load_intent(
    const vector_index_truth_store::publication_intent &intent) {
  return intent.operation ==
             vector_index_truth_store::publication_operation::kBulkLoad &&
         !intent.index_name.empty() && intent.publication_id != 0;
}

}  // namespace

bool stage_load_artifact(const load_staging_identity &identity,
                         const std::string &vector_filename,
                         const std::string &docid_filename,
                         const std::string &format,
                         load_staging_artifact *artifact, std::string *error) {
  if (error != nullptr) error->clear();
  if (artifact == nullptr || !valid_identity(identity) ||
      !valid_format(format) || vector_filename.empty() ||
      (format == "CSV" && !docid_filename.empty())) {
    set_error(error, "LOAD VECTOR DATA staging arguments are invalid");
    return false;
  }

  load_staging_paths paths;
  if (!resolve_paths(identity, format, &paths)) {
    set_error(error, "LOAD VECTOR DATA staging root is unavailable");
    return false;
  }
  const std::string directory = staging_directory(identity);
  const std::string temporary_directory = directory + ".tmp";
  {
    std::lock_guard<std::mutex> guard(g_active_staging_mutex);
    if (!g_active_staging_directories.insert(directory).second) {
      set_error(error, "LOAD VECTOR DATA staging identity is already active");
      return false;
    }
    if (!g_active_staging_directories.insert(temporary_directory).second) {
      g_active_staging_directories.erase(directory);
      set_error(error, "LOAD VECTOR DATA staging identity is already active");
      return false;
    }
  }

  const auto fail = [&]() {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    std::filesystem::remove_all(temporary_directory, ignored);
    release_active_artifact(identity);
    release_active_directory(temporary_directory);
    return false;
  };
  std::error_code ec;
  const std::string root = staging_root_path();
  std::filesystem::create_directories(root, ec);
  if (ec || !std::filesystem::is_directory(root, ec) || ec ||
      !sync_parent_directory(root) || std::filesystem::exists(directory, ec) ||
      ec || std::filesystem::exists(temporary_directory, ec) || ec ||
      !std::filesystem::create_directory(temporary_directory, ec) || ec ||
      !sync_parent_directory(temporary_directory)) {
    set_error(error, "LOAD VECTOR DATA staging directory could not be created");
    return fail();
  }

  load_staging_paths temporary_paths;
  if (!resolve_paths_in_directory(temporary_directory, format,
                                  &temporary_paths)) {
    set_error(error, "LOAD VECTOR DATA staging paths could not be resolved");
    return fail();
  }

  if (!durable_copy(vector_filename, temporary_paths.vector_filename, error)) {
    return fail();
  }
  if (format == "FBIN" && !docid_filename.empty() &&
      !durable_copy(docid_filename, temporary_paths.docid_filename, error)) {
    return fail();
  }

  load_staging_artifact staged;
  staged.identity = artifact_identity(identity);
  if (!checksum_file(temporary_paths.vector_filename, &staged.vector_size,
                     &staged.vector_checksum, error)) {
    return fail();
  }
  if (format == "FBIN" && !docid_filename.empty() &&
      !checksum_file(temporary_paths.docid_filename, &staged.docid_size,
                     &staged.docid_checksum, error)) {
    return fail();
  }
  staged.has_docids = format == "FBIN" && !docid_filename.empty();
  if (!sync_path(temporary_directory)) {
    set_error(error, "LOAD VECTOR DATA staging directory could not be synced");
    return fail();
  }
  std::filesystem::rename(temporary_directory, directory, ec);
  if (ec || !sync_path(root)) {
    set_error(error,
              "LOAD VECTOR DATA staging directory could not be published");
    return fail();
  }
  release_active_directory(temporary_directory);
  *artifact = std::move(staged);
  return true;
}

bool verify_load_artifact(const load_staging_identity &identity,
                          const load_staging_artifact &artifact,
                          const std::string &format, load_staging_paths *paths,
                          std::string *error) {
  if (error != nullptr) error->clear();
  if (paths == nullptr || artifact.identity != artifact_identity(identity) ||
      !resolve_paths(identity, format, paths)) {
    set_error(error, "LOAD VECTOR DATA staging metadata is invalid");
    return false;
  }
  std::error_code ec;
  const std::filesystem::file_status directory_status =
      std::filesystem::symlink_status(staging_directory(identity), ec);
  if (ec || directory_status.type() != std::filesystem::file_type::directory) {
    set_error(error, "LOAD VECTOR DATA staging directory is invalid");
    return false;
  }
  if (!verify_file(paths->vector_filename, artifact.vector_size,
                   artifact.vector_checksum, error)) {
    return false;
  }
  if (!artifact.has_docids) {
    paths->docid_filename.clear();
    return true;
  }
  if (format != "FBIN" ||
      !verify_file(paths->docid_filename, artifact.docid_size,
                   artifact.docid_checksum, error)) {
    return false;
  }
  return true;
}

bool make_load_staging_payload(
    const load_staging_identity &identity,
    const load_staging_artifact &artifact, bool replace_duplicates,
    bool rebuild_after_load, size_t dimension, const std::string &format,
    vector_statement_publication::operation_payload *payload) {
  if (payload == nullptr || dimension == 0 || !valid_format(format) ||
      artifact.identity != artifact_identity(identity) ||
      (format == "CSV" && (artifact.has_docids || artifact.docid_size != 0 ||
                           artifact.docid_checksum != 0)) ||
      (!artifact.has_docids &&
       (artifact.docid_size != 0 || artifact.docid_checksum != 0))) {
    return false;
  }
  payload->unsigned_values = {
      k_load_staging_payload,        replace_duplicates ? 1U : 0U,
      rebuild_after_load ? 1U : 0U,  static_cast<uint64_t>(dimension),
      artifact.vector_size,          artifact.vector_checksum,
      artifact.has_docids ? 1U : 0U, artifact.docid_size,
      artifact.docid_checksum};
  payload->string_values = {artifact.identity, format};
  payload->binary_value.clear();
  return true;
}

bool parse_load_staging_payload(
    const load_staging_identity &identity,
    const vector_statement_publication::operation_payload &payload,
    load_staging_artifact *artifact, bool *replace_duplicates,
    bool *rebuild_after_load, size_t *dimension, std::string *format) {
  if (payload.unsigned_values.size() != k_unsigned_field_count ||
      payload.string_values.size() != k_string_field_count ||
      !payload.binary_value.empty() ||
      payload.unsigned_values[k_payload_type] != k_load_staging_payload ||
      payload.unsigned_values[k_replace_duplicates] > 1 ||
      payload.unsigned_values[k_rebuild_after_load] > 1 ||
      payload.unsigned_values[k_dimension] == 0 ||
      payload.unsigned_values[k_dimension] >
          std::numeric_limits<size_t>::max() ||
      payload.unsigned_values[k_has_docids] > 1 ||
      payload.string_values[k_artifact_identity] !=
          artifact_identity(identity) ||
      !valid_format(payload.string_values[k_format]) ||
      (payload.string_values[k_format] == "CSV" &&
       (payload.unsigned_values[k_has_docids] != 0 ||
        payload.unsigned_values[k_docid_size] != 0 ||
        payload.unsigned_values[k_docid_checksum] != 0)) ||
      (payload.unsigned_values[k_has_docids] == 0 &&
       (payload.unsigned_values[k_docid_size] != 0 ||
        payload.unsigned_values[k_docid_checksum] != 0))) {
    return false;
  }
  if (artifact != nullptr) {
    artifact->identity = payload.string_values[k_artifact_identity];
    artifact->vector_size = payload.unsigned_values[k_vector_size];
    artifact->vector_checksum = payload.unsigned_values[k_vector_checksum];
    artifact->has_docids = payload.unsigned_values[k_has_docids] != 0;
    artifact->docid_size = payload.unsigned_values[k_docid_size];
    artifact->docid_checksum = payload.unsigned_values[k_docid_checksum];
  }
  if (replace_duplicates != nullptr) {
    *replace_duplicates = payload.unsigned_values[k_replace_duplicates] != 0;
  }
  if (rebuild_after_load != nullptr) {
    *rebuild_after_load = payload.unsigned_values[k_rebuild_after_load] != 0;
  }
  if (dimension != nullptr) {
    *dimension = static_cast<size_t>(payload.unsigned_values[k_dimension]);
  }
  if (format != nullptr) *format = payload.string_values[k_format];
  return true;
}

void discard_load_staging_intents(
    const std::vector<vector_index_truth_store::publication_intent> &intents) {
  for (const auto &intent : intents) {
    (void)remove_managed_load_staging_artifact(intent);
  }
}

bool remove_load_staging_artifact(const load_staging_identity &identity) {
  return valid_identity(identity) && remove_artifact_directory(identity);
}

bool remove_managed_load_staging_artifact(
    const vector_index_truth_store::publication_intent &intent) {
  if (!is_bulk_load_intent(intent)) return true;
  vector_statement_publication::operation_payload payload;
  const load_staging_identity identity{intent.index_name,
                                       intent.publication_id};
  if (!vector_statement_publication::decode_operation_payload(intent.payload,
                                                              &payload)) {
    return false;
  }
  if (payload.unsigned_values.empty() ||
      payload.unsigned_values[0] != k_load_staging_payload) {
    return true;
  }
  if (!parse_load_staging_payload(identity, payload, nullptr, nullptr, nullptr,
                                  nullptr, nullptr)) {
    return false;
  }
  return remove_artifact_directory(identity);
}

bool cleanup_orphaned_load_staging_artifacts(
    const std::vector<vector_index_truth_store::publication_intent> &intents) {
  const std::string root = staging_root_path();
  if (root.empty()) return true;

  std::unordered_set<std::string> referenced;
  for (const auto &intent : intents) {
    if (!is_bulk_load_intent(intent)) continue;
    referenced.insert(
        staging_directory({intent.index_name, intent.publication_id}));
  }

  std::lock_guard<std::mutex> guard(g_active_staging_mutex);
  std::error_code ec;
  if (!std::filesystem::exists(root, ec)) return !ec;
  if (ec || !std::filesystem::is_directory(root, ec) || ec) return false;
  for (std::filesystem::directory_iterator iter(root, ec), end;
       !ec && iter != end; iter.increment(ec)) {
    const std::string path = iter->path().string();
    const std::filesystem::file_status status = iter->symlink_status(ec);
    if (ec || status.type() != std::filesystem::file_type::directory ||
        !is_managed_staging_name(iter->path().filename().string())) {
      return false;
    }
    if (referenced.find(path) != referenced.end() ||
        g_active_staging_directories.find(path) !=
            g_active_staging_directories.end()) {
      continue;
    }
    std::error_code remove_ec;
    std::filesystem::remove_all(iter->path(), remove_ec);
    if (remove_ec) return false;
  }
  return !ec && sync_path(root);
}

}  // namespace vector_index
