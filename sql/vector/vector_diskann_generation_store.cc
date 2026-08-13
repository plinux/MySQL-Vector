/* Copyright (c) 2026, Oracle and/or its affiliates.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0,
   as published by the Free Software Foundation.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

#include "sql/vector/vector_diskann_generation_store.h"

#include <fcntl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

#include "my_dbug.h"
#include "my_sys.h"

namespace vector_index {
namespace {

constexpr char kArtifactManifestHeader[] =
    "mysql-vector-diskann-artifact-manifest-v1";
constexpr char kArtifactManifestFilename[] =
    "mysql-vector-artifact.manifest.v1";
constexpr char kDocIdMapFilenameSuffix[] = "mysql_vector_docids.bin";
constexpr char kSwapJournalHeader[] = "mysql-vector-diskann-swap-journal-v1";
constexpr char kSwapJournalSuffix[] = ".swap-journal.v1";
constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

struct artifact_file_record {
  std::string relative_path;
  uint64_t size{0};
  uint64_t checksum{0};
};

struct artifact_manifest {
  diskann_artifact_identity identity;
  std::vector<artifact_file_record> files;
};

void set_error(std::string *error, const std::string &message) {
  if (error != nullptr) *error = message;
}

bool parse_uint64(const std::string &text, uint64_t *value) {
  if (value == nullptr || text.empty()) return false;
  for (const char ch : text) {
    if (ch < '0' || ch > '9') return false;
  }
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0') return false;
  *value = static_cast<uint64_t>(parsed);
  return true;
}

bool parse_uint32(const std::string &text, uint32_t *value) {
  uint64_t parsed = 0;
  if (!parse_uint64(text, &parsed) ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *value = static_cast<uint32_t>(parsed);
  return true;
}

std::string encode_hex(const std::string &value) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string encoded;
  encoded.reserve(value.size() * 2);
  for (const unsigned char ch : value) {
    encoded.push_back(digits[ch >> 4]);
    encoded.push_back(digits[ch & 0x0f]);
  }
  return encoded;
}

bool decode_hex(const std::string &value, std::string *decoded) {
  if (decoded == nullptr || value.size() % 2 != 0) return false;
  auto nibble = [](char ch, unsigned char *out) {
    if (ch >= '0' && ch <= '9') {
      *out = static_cast<unsigned char>(ch - '0');
      return true;
    }
    if (ch >= 'a' && ch <= 'f') {
      *out = static_cast<unsigned char>(ch - 'a' + 10);
      return true;
    }
    return false;
  };

  decoded->clear();
  decoded->reserve(value.size() / 2);
  for (size_t offset = 0; offset < value.size(); offset += 2) {
    unsigned char high = 0;
    unsigned char low = 0;
    if (!nibble(value[offset], &high) || !nibble(value[offset + 1], &low)) {
      decoded->clear();
      return false;
    }
    decoded->push_back(static_cast<char>((high << 4) | low));
  }
  return true;
}

bool sync_path(const std::filesystem::path &path) {
  const File fd = my_open(path.string().c_str(), O_RDONLY, MYF(0));
  if (fd < 0) return false;
  const bool ok = my_sync(fd, MYF(0)) == 0;
  const bool close_ok = my_close(fd, MYF(0)) == 0;
  return ok && close_ok;
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

bool atomic_write_file(const std::filesystem::path &path,
                       const std::string &contents, std::string *error) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  if (ec) {
    set_error(error, "could not create artifact metadata directory");
    return false;
  }
  const std::filesystem::path temporary = path.string() + ".tmp";
  std::ofstream file(temporary,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.is_open()) {
    set_error(error, "could not open artifact metadata temporary file");
    return false;
  }
  file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  file.flush();
  file.close();
  if (!file || !sync_path(temporary)) {
    std::filesystem::remove(temporary, ec);
    set_error(error, "could not make artifact metadata durable");
    return false;
  }
  std::filesystem::rename(temporary, path, ec);
  if (ec || !sync_parent_directory(path)) {
    std::filesystem::remove(temporary, ec);
    set_error(error, "could not publish artifact metadata file");
    return false;
  }
  return true;
}

bool checksum_file(const std::filesystem::path &path, uint64_t *checksum,
                   uint64_t *size, std::string *error) {
  if (checksum == nullptr || size == nullptr) return false;
  *checksum = kFnvOffsetBasis;
  *size = 0;
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    set_error(error, "artifact file is not readable");
    return false;
  }
  std::array<char, 64 * 1024> buffer{};
  while (file.good()) {
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = file.gcount();
    if (count < 0 || static_cast<uint64_t>(count) >
                         std::numeric_limits<uint64_t>::max() - *size) {
      set_error(error, "artifact file size overflow");
      return false;
    }
    *size += static_cast<uint64_t>(count);
    for (std::streamsize i = 0; i < count; ++i) {
      *checksum ^= static_cast<unsigned char>(buffer[static_cast<size_t>(i)]);
      *checksum *= kFnvPrime;
    }
  }
  if (!file.eof()) {
    set_error(error, "artifact file read failed");
    return false;
  }
  return true;
}

bool safe_relative_path(const std::filesystem::path &path) {
  if (path.empty() || path.is_absolute()) return false;
  for (const auto &component : path) {
    if (component == ".." || component == ".") return false;
  }
  return true;
}

bool safe_directory_name(const std::string &name) {
  if (name.empty() || name == "." || name == "..") return false;
  return name.find('/') == std::string::npos &&
         name.find('\\') == std::string::npos;
}

std::filesystem::path artifact_manifest_path(const std::string &directory) {
  return std::filesystem::path(directory) / kArtifactManifestFilename;
}

bool collect_artifact_files(const std::string &directory,
                            std::vector<artifact_file_record> *files,
                            uint64_t *doc_id_checksum, std::string *error) {
  if (files == nullptr || doc_id_checksum == nullptr) return false;
  files->clear();
  *doc_id_checksum = 0;
  const std::filesystem::path root(directory);
  std::error_code ec;
  if (!std::filesystem::is_directory(root, ec) || ec) {
    set_error(error, "artifact generation directory is missing");
    return false;
  }

  size_t doc_id_file_count = 0;
  for (std::filesystem::recursive_directory_iterator iter(root, ec), end;
       !ec && iter != end; iter.increment(ec)) {
    const std::filesystem::directory_entry &entry = *iter;
    if (entry.is_symlink(ec) || ec) {
      set_error(error, "artifact generation contains a symbolic link");
      return false;
    }
    if (entry.is_directory(ec) && !ec) continue;
    if (!entry.is_regular_file(ec) || ec) {
      set_error(error, "artifact generation contains an unsupported entry");
      return false;
    }
    const std::filesystem::path relative =
        std::filesystem::relative(entry.path(), root, ec);
    if (ec || !safe_relative_path(relative)) {
      set_error(error, "artifact file path escapes generation directory");
      return false;
    }
    if (relative == kArtifactManifestFilename ||
        relative == std::string(kArtifactManifestFilename) + ".tmp") {
      continue;
    }

    artifact_file_record record;
    record.relative_path = relative.generic_string();
    if (!checksum_file(entry.path(), &record.checksum, &record.size, error)) {
      return false;
    }
    if (entry.path().filename().string().find(kDocIdMapFilenameSuffix) !=
        std::string::npos) {
      ++doc_id_file_count;
      *doc_id_checksum = record.checksum;
    }
    files->push_back(std::move(record));
  }
  if (ec) {
    set_error(error, "artifact directory traversal failed");
    return false;
  }
  if (files->empty()) {
    set_error(error, "artifact generation contains no files");
    return false;
  }
  if (doc_id_file_count != 1) {
    set_error(error, "artifact generation must contain one doc-id map");
    return false;
  }
  std::sort(files->begin(), files->end(), [](const auto &lhs, const auto &rhs) {
    return lhs.relative_path < rhs.relative_path;
  });
  return true;
}

bool validate_doc_id_map(const std::string &directory,
                         const artifact_manifest &manifest,
                         std::string *error) {
  const artifact_file_record *doc_id_record = nullptr;
  for (const auto &record : manifest.files) {
    if (std::filesystem::path(record.relative_path).filename().string().find(
            kDocIdMapFilenameSuffix) == std::string::npos) {
      continue;
    }
    if (doc_id_record != nullptr) {
      set_error(error, "artifact manifest contains multiple doc-id maps");
      return false;
    }
    doc_id_record = &record;
  }
  if (doc_id_record == nullptr ||
      doc_id_record->checksum != manifest.identity.doc_id_checksum) {
    set_error(error, "artifact manifest doc-id map is invalid");
    return false;
  }

  std::ifstream file(
      std::filesystem::path(directory) / doc_id_record->relative_path,
      std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    set_error(error, "artifact doc-id map is not readable");
    return false;
  }
  file.seekg(0, std::ios::end);
  const std::streamoff file_size = file.tellg();
  if (file_size < 0 || static_cast<uint64_t>(file_size) != doc_id_record->size) {
    set_error(error, "artifact doc-id map size does not match manifest");
    return false;
  }
  file.seekg(0, std::ios::beg);
  uint64_t count = 0;
  file.read(reinterpret_cast<char *>(&count), sizeof(count));
  if (!file.good() ||
      count > (std::numeric_limits<uint64_t>::max() - sizeof(count)) /
                  sizeof(uint64_t)) {
    set_error(error, "artifact doc-id map header is invalid");
    return false;
  }
  const uint64_t expected_size =
      sizeof(count) + count * sizeof(uint64_t);
  if (expected_size != doc_id_record->size ||
      count != manifest.identity.doc_id_count) {
    set_error(error, "artifact doc-id map count does not match");
    return false;
  }
  return true;
}

void write_identity(std::ostream &stream,
                    const diskann_artifact_identity &identity) {
  stream << "index_identity\t" << identity.index_identity << '\n'
         << "truth_generation\t" << identity.truth_generation << '\n'
         << "config_generation\t" << identity.config_generation << '\n'
         << "doc_id_count\t" << identity.doc_id_count << '\n'
         << "doc_id_checksum\t" << identity.doc_id_checksum << '\n'
         << "dimension\t" << identity.dimension << '\n'
         << "metric\t" << encode_hex(identity.metric) << '\n'
         << "mode\t" << encode_hex(identity.mode) << '\n'
         << "provider\t" << encode_hex(identity.provider) << '\n'
         << "consistency_mode\t" << encode_hex(identity.consistency_mode)
         << '\n'
         << "schema_name\t" << encode_hex(identity.schema_name) << '\n'
         << "table_name\t" << encode_hex(identity.table_name) << '\n'
         << "column_name\t" << encode_hex(identity.column_name) << '\n'
         << "doc_id_column_name\t" << encode_hex(identity.doc_id_column_name)
         << '\n';
}

bool read_key_values(std::istream &stream,
                     std::multimap<std::string, std::string> *values,
                     std::string *error) {
  if (values == nullptr) return false;
  values->clear();
  std::string line;
  while (std::getline(stream, line)) {
    const size_t separator = line.find('\t');
    if (separator == std::string::npos || separator == 0) {
      set_error(error, "artifact metadata line is malformed");
      return false;
    }
    values->emplace(line.substr(0, separator), line.substr(separator + 1));
  }
  if (!stream.eof()) {
    set_error(error, "artifact metadata read failed");
    return false;
  }
  return true;
}

bool single_value(const std::multimap<std::string, std::string> &values,
                  const char *key, std::string *value) {
  const auto range = values.equal_range(key);
  if (range.first == range.second || std::next(range.first) != range.second) {
    return false;
  }
  *value = range.first->second;
  return true;
}

bool read_identity(const std::multimap<std::string, std::string> &values,
                   diskann_artifact_identity *identity, std::string *error) {
  if (identity == nullptr) return false;
  *identity = {};
  std::string value;
  const bool ok = single_value(values, "index_identity", &value) &&
                  parse_uint64(value, &identity->index_identity) &&
                  single_value(values, "truth_generation", &value) &&
                  parse_uint64(value, &identity->truth_generation) &&
                  single_value(values, "config_generation", &value) &&
                  parse_uint64(value, &identity->config_generation) &&
                  single_value(values, "doc_id_count", &value) &&
                  parse_uint64(value, &identity->doc_id_count) &&
                  single_value(values, "doc_id_checksum", &value) &&
                  parse_uint64(value, &identity->doc_id_checksum) &&
                  single_value(values, "dimension", &value) &&
                  parse_uint32(value, &identity->dimension) &&
                  single_value(values, "metric", &value) &&
                  decode_hex(value, &identity->metric) &&
                  single_value(values, "mode", &value) &&
                  decode_hex(value, &identity->mode) &&
                  single_value(values, "provider", &value) &&
                  decode_hex(value, &identity->provider) &&
                  single_value(values, "consistency_mode", &value) &&
                  decode_hex(value, &identity->consistency_mode) &&
                  single_value(values, "schema_name", &value) &&
                  decode_hex(value, &identity->schema_name) &&
                  single_value(values, "table_name", &value) &&
                  decode_hex(value, &identity->table_name) &&
                  single_value(values, "column_name", &value) &&
                  decode_hex(value, &identity->column_name) &&
                  single_value(values, "doc_id_column_name", &value) &&
                  decode_hex(value, &identity->doc_id_column_name);
  if (!ok || identity->index_identity == 0 ||
      identity->config_generation == 0 || identity->dimension == 0) {
    set_error(error, "artifact identity is invalid");
    return false;
  }
  return true;
}

std::string serialize_manifest(const artifact_manifest &manifest) {
  std::ostringstream stream;
  stream << kArtifactManifestHeader << '\n';
  write_identity(stream, manifest.identity);
  stream << "file_count\t" << manifest.files.size() << '\n';
  for (const auto &file : manifest.files) {
    stream << "file\t" << encode_hex(file.relative_path) << ',' << file.size
           << ',' << file.checksum << '\n';
  }
  return stream.str();
}

bool parse_file_record(const std::string &value, artifact_file_record *record) {
  if (record == nullptr) return false;
  const size_t first = value.find(',');
  const size_t second = first == std::string::npos ? std::string::npos
                                                   : value.find(',', first + 1);
  if (first == std::string::npos || second == std::string::npos ||
      value.find(',', second + 1) != std::string::npos ||
      !decode_hex(value.substr(0, first), &record->relative_path) ||
      !safe_relative_path(record->relative_path) ||
      !parse_uint64(value.substr(first + 1, second - first - 1),
                    &record->size) ||
      !parse_uint64(value.substr(second + 1), &record->checksum)) {
    return false;
  }
  return true;
}

bool read_manifest(const std::string &directory, artifact_manifest *manifest,
                   std::string *error) {
  if (manifest == nullptr) return false;
  *manifest = {};
  std::ifstream file(artifact_manifest_path(directory),
                     std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    set_error(error, "artifact manifest is missing");
    return false;
  }
  std::string header;
  if (!std::getline(file, header) || header != kArtifactManifestHeader) {
    set_error(error, "artifact manifest header is invalid");
    return false;
  }
  std::multimap<std::string, std::string> values;
  if (!read_key_values(file, &values, error) ||
      !read_identity(values, &manifest->identity, error)) {
    return false;
  }
  std::string count_text;
  uint64_t file_count = 0;
  if (!single_value(values, "file_count", &count_text) ||
      !parse_uint64(count_text, &file_count) ||
      file_count > std::numeric_limits<size_t>::max()) {
    set_error(error, "artifact manifest file count is invalid");
    return false;
  }
  const auto range = values.equal_range("file");
  for (auto it = range.first; it != range.second; ++it) {
    artifact_file_record record;
    if (!parse_file_record(it->second, &record)) {
      set_error(error, "artifact manifest file record is invalid");
      return false;
    }
    manifest->files.push_back(std::move(record));
  }
  if (manifest->files.size() != file_count || manifest->files.empty()) {
    set_error(error, "artifact manifest file count does not match");
    return false;
  }
  std::sort(manifest->files.begin(), manifest->files.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.relative_path < rhs.relative_path;
            });
  if (std::adjacent_find(manifest->files.begin(), manifest->files.end(),
                         [](const auto &lhs, const auto &rhs) {
                           return lhs.relative_path == rhs.relative_path;
                         }) != manifest->files.end()) {
    set_error(error, "artifact manifest contains duplicate files");
    return false;
  }
  return true;
}

const char *swap_state_name(diskann_swap_state state) {
  switch (state) {
    case diskann_swap_state::kPrepared:
      return "prepared";
    case diskann_swap_state::kOldSaved:
      return "old_saved";
    case diskann_swap_state::kNewInstalled:
      return "new_installed";
    case diskann_swap_state::kVerified:
      return "verified";
    case diskann_swap_state::kComplete:
      return "complete";
  }
  return "invalid";
}

bool parse_swap_state(const std::string &value, diskann_swap_state *state) {
  if (state == nullptr) return false;
  if (value == "prepared") {
    *state = diskann_swap_state::kPrepared;
  } else if (value == "old_saved") {
    *state = diskann_swap_state::kOldSaved;
  } else if (value == "new_installed") {
    *state = diskann_swap_state::kNewInstalled;
  } else if (value == "verified") {
    *state = diskann_swap_state::kVerified;
  } else if (value == "complete") {
    *state = diskann_swap_state::kComplete;
  } else {
    return false;
  }
  return true;
}

std::string serialize_journal(const diskann_generation_swap &swap) {
  std::ostringstream stream;
  stream
      << kSwapJournalHeader << '\n'
      << "state\t" << swap_state_name(swap.state) << '\n'
      << "had_old_generation\t" << (swap.had_old_generation ? 1 : 0) << '\n'
      << "staging_name\t"
      << encode_hex(
             std::filesystem::path(swap.staging_directory).filename().string())
      << '\n'
      << "backup_name\t"
      << encode_hex(
             std::filesystem::path(swap.backup_directory).filename().string())
      << '\n';
  write_identity(stream, swap.target);
  return stream.str();
}

bool write_journal(const diskann_generation_swap &swap, std::string *error) {
  return atomic_write_file(swap.journal_path, serialize_journal(swap), error);
}

bool read_journal(const std::string &live_directory,
                  diskann_generation_swap *swap, bool *exists,
                  std::string *error) {
  if (swap == nullptr || exists == nullptr) return false;
  *swap = {};
  *exists = false;
  const std::filesystem::path live(live_directory);
  const std::filesystem::path journal =
      diskann_generation_journal_path(live_directory);
  std::ifstream file(journal, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    std::error_code ec;
    if (!std::filesystem::exists(journal, ec) && !ec) return true;
    set_error(error, "swap journal is not readable");
    return false;
  }
  std::string header;
  if (!std::getline(file, header) || header != kSwapJournalHeader) {
    set_error(error, "swap journal header is invalid");
    return false;
  }
  std::multimap<std::string, std::string> values;
  if (!read_key_values(file, &values, error) ||
      !read_identity(values, &swap->target, error)) {
    return false;
  }
  std::string state_text;
  std::string had_old_text;
  std::string staging_hex;
  std::string backup_hex;
  std::string staging_name;
  std::string backup_name;
  uint64_t had_old = 0;
  if (!single_value(values, "state", &state_text) ||
      !parse_swap_state(state_text, &swap->state) ||
      !single_value(values, "had_old_generation", &had_old_text) ||
      !parse_uint64(had_old_text, &had_old) || had_old > 1 ||
      !single_value(values, "staging_name", &staging_hex) ||
      !decode_hex(staging_hex, &staging_name) ||
      !safe_directory_name(staging_name) ||
      !single_value(values, "backup_name", &backup_hex) ||
      !decode_hex(backup_hex, &backup_name) ||
      !safe_directory_name(backup_name)) {
    set_error(error, "swap journal fields are invalid");
    return false;
  }
  swap->live_directory = live.string();
  swap->staging_directory = (live.parent_path() / staging_name).string();
  swap->backup_directory = (live.parent_path() / backup_name).string();
  swap->journal_path = journal.string();
  swap->had_old_generation = had_old != 0;
  *exists = true;
  return true;
}

bool remove_path(const std::filesystem::path &path, std::string *error) {
  std::error_code ec;
  std::filesystem::remove_all(path, ec);
  if (ec) {
    set_error(error, "could not remove generation path");
    return false;
  }
  return sync_parent_directory(path);
}

bool rename_path(const std::filesystem::path &from,
                 const std::filesystem::path &to, std::string *error) {
  std::error_code ec;
  std::filesystem::rename(from, to, ec);
  if (ec || !sync_parent_directory(to)) {
    set_error(error, "could not rename generation path");
    return false;
  }
  return true;
}

bool path_exists(const std::filesystem::path &path, bool *exists,
                 std::string *error) {
  std::error_code ec;
  *exists = std::filesystem::exists(path, ec);
  if (ec) set_error(error, "could not inspect generation path");
  return !ec;
}

bool advance_state(diskann_generation_swap *swap, diskann_swap_state state,
                   std::string *error) {
  if (swap == nullptr) return false;
  swap->state = state;
  return write_journal(*swap, error);
}

bool validate_manifest_files(const std::string &directory,
                             const artifact_manifest &manifest,
                             std::string *error) {
  for (const auto &record : manifest.files) {
    uint64_t checksum = 0;
    uint64_t size = 0;
    if (!checksum_file(std::filesystem::path(directory) / record.relative_path,
                       &checksum, &size, error) ||
        checksum != record.checksum || size != record.size) {
      set_error(error, "artifact file checksum or size does not match");
      return false;
    }
  }
  return true;
}

bool rollback_loaded_swap(diskann_generation_swap *swap, std::string *error) {
  if (swap == nullptr) return false;
  const std::filesystem::path live(swap->live_directory);
  const std::filesystem::path staging(swap->staging_directory);
  const std::filesystem::path backup(swap->backup_directory);
  bool live_exists = false;
  bool staging_exists = false;
  bool backup_exists = false;
  if (!path_exists(live, &live_exists, error) ||
      !path_exists(staging, &staging_exists, error) ||
      !path_exists(backup, &backup_exists, error)) {
    return false;
  }

  bool old_generation_is_live = false;
  if (swap->state == diskann_swap_state::kPrepared) {
    if (swap->had_old_generation) {
      if (live_exists && !backup_exists) {
        old_generation_is_live = true;
      } else if (live_exists == backup_exists) {
        set_error(error,
                  "prepared rollback has ambiguous old generation state");
        return false;
      }
    } else if (live_exists || backup_exists) {
      set_error(error, "prepared rollback contains an unexpected generation");
      return false;
    }
  } else if (swap->state == diskann_swap_state::kOldSaved) {
    if (live_exists == staging_exists) {
      set_error(error, "old-saved rollback has ambiguous candidate state");
      return false;
    }
    if (live_exists && !remove_path(live, error)) return false;
  } else if (live_exists && !remove_path(live, error)) {
    return false;
  }

  if (swap->had_old_generation && !old_generation_is_live) {
    if (!backup_exists || !rename_path(backup, live, error)) return false;
  } else if (backup_exists && !remove_path(backup, error)) {
    return false;
  }
  if (staging_exists && !remove_path(staging, error)) return false;
  if (!remove_path(swap->journal_path, error)) return false;
  *swap = {};
  return true;
}

bool finish_loaded_swap(diskann_generation_swap *swap, std::string *error) {
  if (swap == nullptr) return false;
  const std::filesystem::path live(swap->live_directory);
  const std::filesystem::path staging(swap->staging_directory);
  const std::filesystem::path backup(swap->backup_directory);
  bool live_exists = false;
  bool staging_exists = false;
  if (!path_exists(live, &live_exists, error) ||
      !path_exists(staging, &staging_exists, error)) {
    return false;
  }
  if (swap->state == diskann_swap_state::kPrepared) {
    bool backup_exists = false;
    if (!path_exists(backup, &backup_exists, error)) return false;
    if (swap->had_old_generation) {
      if (live_exists == backup_exists) {
        set_error(error, "prepared swap has ambiguous old generation state");
        return false;
      }
      if (live_exists && !rename_path(live, backup, error)) return false;
    } else if (live_exists || backup_exists) {
      set_error(error, "prepared swap unexpectedly contains an old generation");
      return false;
    }
    if (!advance_state(swap, diskann_swap_state::kOldSaved, error)) {
      return false;
    }
    live_exists = false;
  }
  if (swap->state == diskann_swap_state::kOldSaved) {
    if (live_exists == staging_exists) {
      set_error(error, "old-saved swap has ambiguous candidate state");
      return false;
    }
    if (staging_exists && !rename_path(staging, live, error)) return false;
    if (!advance_state(swap, diskann_swap_state::kNewInstalled, error)) {
      return false;
    }
  }
  diskann_artifact_identity actual;
  if (!validate_diskann_generation_store(live.string(), swap->target, &actual,
                                         error)) {
    return false;
  }
  if (swap->state == diskann_swap_state::kNewInstalled &&
      !advance_state(swap, diskann_swap_state::kVerified, error)) {
    return false;
  }
  if (swap->state == diskann_swap_state::kVerified &&
      !advance_state(swap, diskann_swap_state::kComplete, error)) {
    return false;
  }
  bool backup_exists = false;
  if (!path_exists(backup, &backup_exists, error)) return false;
  if (backup_exists && !remove_path(backup, error)) return false;
  if (!remove_path(swap->journal_path, error)) return false;
  *swap = {};
  return true;
}

}  // namespace

bool diskann_artifact_publication_matches(
    const diskann_artifact_identity &expected,
    const diskann_artifact_identity &actual) {
  return expected.index_identity == actual.index_identity &&
         expected.truth_generation == actual.truth_generation &&
         expected.config_generation == actual.config_generation &&
         expected.doc_id_count == actual.doc_id_count &&
         expected.dimension == actual.dimension &&
         expected.metric == actual.metric && expected.mode == actual.mode &&
         expected.provider == actual.provider &&
         expected.consistency_mode == actual.consistency_mode &&
         expected.schema_name == actual.schema_name &&
         expected.table_name == actual.table_name &&
         expected.column_name == actual.column_name &&
         expected.doc_id_column_name == actual.doc_id_column_name;
}

std::string diskann_generation_journal_path(const std::string &live_directory) {
  if (live_directory.empty()) return "";
  return live_directory + kSwapJournalSuffix;
}

bool prepare_diskann_generation_swap(const std::string &live_directory,
                                     const std::string &build_staging_directory,
                                     const diskann_artifact_identity &target,
                                     diskann_generation_swap *swap,
                                     std::string *error) {
  if (swap == nullptr || live_directory.empty() ||
      build_staging_directory.empty() || target.index_identity == 0 ||
      target.config_generation == 0 || target.dimension == 0 ||
      target.provider != "diskann") {
    set_error(error, "DiskANN generation publication input is invalid");
    return false;
  }
  *swap = {};
  const std::filesystem::path live(live_directory);
  const std::filesystem::path build_staging(build_staging_directory);
  if (live.parent_path() != build_staging.parent_path() ||
      live == build_staging) {
    set_error(error, "DiskANN staging directory is outside the index root");
    return false;
  }
  const std::string suffix = std::to_string(target.index_identity) + "." +
                             std::to_string(target.truth_generation) + "." +
                             std::to_string(target.config_generation);
  const std::filesystem::path staging = live.string() + ".stage." + suffix;
  const std::filesystem::path backup = live.string() + ".backup." + suffix;
  const std::filesystem::path journal =
      diskann_generation_journal_path(live_directory);
  bool journal_exists = false;
  if (!path_exists(journal, &journal_exists, error) || journal_exists) {
    if (journal_exists) set_error(error, "DiskANN swap journal already exists");
    return false;
  }
  bool build_staging_exists = false;
  if (!path_exists(build_staging, &build_staging_exists, error) ||
      !build_staging_exists) {
    set_error(error, "DiskANN build staging directory is missing");
    return false;
  }
  bool live_exists = false;
  DBUG_EXECUTE_IF("vector_diskann_generation_fail_live_preflight", {
    set_error(error, "could not inspect generation path");
    return false;
  });
  if (!path_exists(live, &live_exists, error)) return false;
  if (!remove_path(staging, error) || !remove_path(backup, error) ||
      !rename_path(build_staging, staging, error)) {
    return false;
  }

  artifact_manifest manifest;
  manifest.identity = target;
  if (!collect_artifact_files(staging.string(), &manifest.files,
                              &manifest.identity.doc_id_checksum, error) ||
      !validate_doc_id_map(staging.string(), manifest, error) ||
      !atomic_write_file(artifact_manifest_path(staging.string()),
                         serialize_manifest(manifest), error)) {
    (void)remove_path(staging, nullptr);
    return false;
  }

  swap->live_directory = live.string();
  swap->staging_directory = staging.string();
  swap->backup_directory = backup.string();
  swap->journal_path = journal.string();
  swap->target = manifest.identity;
  swap->state = diskann_swap_state::kPrepared;
  swap->had_old_generation = live_exists;
  if (!write_journal(*swap, error)) {
    (void)remove_path(staging, nullptr);
    *swap = {};
    return false;
  }
  return true;
}

bool publish_diskann_generation_swap(diskann_generation_swap *swap,
                                     std::string *error) {
  if (swap == nullptr || swap->state != diskann_swap_state::kPrepared) {
    set_error(error, "DiskANN generation is not prepared");
    return false;
  }
  const std::filesystem::path live(swap->live_directory);
  const std::filesystem::path staging(swap->staging_directory);
  const std::filesystem::path backup(swap->backup_directory);
  if (swap->had_old_generation && !rename_path(live, backup, error)) {
    return false;
  }
  if (!advance_state(swap, diskann_swap_state::kOldSaved, error)) {
    return false;
  }
  if (!rename_path(staging, live, error) ||
      !advance_state(swap, diskann_swap_state::kNewInstalled, error)) {
    return false;
  }
  diskann_artifact_identity actual;
  return validate_diskann_generation_store(live.string(), swap->target, &actual,
                                           error);
}

bool verify_diskann_generation_swap(diskann_generation_swap *swap,
                                    std::string *error) {
  if (swap == nullptr || swap->state != diskann_swap_state::kNewInstalled) {
    set_error(error, "DiskANN generation is not installed");
    return false;
  }
  return advance_state(swap, diskann_swap_state::kVerified, error);
}

bool rollback_diskann_generation_swap(diskann_generation_swap *swap,
                                      std::string *error) {
  if (swap == nullptr || swap->journal_path.empty()) return true;
  return rollback_loaded_swap(swap, error);
}

bool finalize_diskann_generation_swap(diskann_generation_swap *swap,
                                      std::string *error) {
  if (swap == nullptr || swap->state != diskann_swap_state::kVerified) {
    set_error(error, "DiskANN generation is not verified");
    return false;
  }
  if (!advance_state(swap, diskann_swap_state::kComplete, error)) return false;
  const std::filesystem::path backup(swap->backup_directory);
  bool backup_exists = false;
  if (!path_exists(backup, &backup_exists, error)) return false;
  if (backup_exists && !remove_path(backup, error)) return false;
  if (!remove_path(swap->journal_path, error)) return false;
  *swap = {};
  return true;
}

bool recover_diskann_generation_swap(
    const std::string &live_directory,
    const diskann_artifact_identity &expected_publication, bool *journal_found,
    std::string *error) {
  if (journal_found == nullptr) return false;
  *journal_found = false;
  diskann_generation_swap swap;
  bool exists = false;
  if (!read_journal(live_directory, &swap, &exists, error)) return false;
  if (!exists) return true;
  *journal_found = true;
  return diskann_artifact_publication_matches(expected_publication, swap.target)
             ? finish_loaded_swap(&swap, error)
             : rollback_loaded_swap(&swap, error);
}

bool validate_diskann_generation_store(
    const std::string &live_directory,
    const diskann_artifact_identity &expected_publication,
    diskann_artifact_identity *actual_identity, std::string *error) {
  artifact_manifest manifest;
  if (!read_manifest(live_directory, &manifest, error) ||
      !diskann_artifact_publication_matches(expected_publication,
                                            manifest.identity) ||
      !validate_manifest_files(live_directory, manifest, error) ||
      !validate_doc_id_map(live_directory, manifest, error)) {
    if (error != nullptr && error->empty()) {
      *error = "DiskANN artifact publication identity does not match";
    }
    return false;
  }
  if (actual_identity != nullptr) *actual_identity = manifest.identity;
  return true;
}

}  // namespace vector_index
