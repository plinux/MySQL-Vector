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

#include "sql/vector/vector_index_service.h"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <utility>

#include "my_dbug.h"
#include "sql/vector/vector_index_backend_common.h"
#include "sql/vector/vector_index_backend_internal.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_runtime_config.h"
#include "sql/vector/vector_index_service_internal.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_segmented_search.h"
#include "sql/vector/vector_load_file.h"
#include "sql/vector/vector_status.h"

namespace vector_index {

namespace {

constexpr const char *kStandaloneSegmentHeader =
    "mysql-vector-standalone-segment-v1";
constexpr const char *kStandaloneManifestHeader =
    "mysql-vector-standalone-manifest-v1";
constexpr uint8_t kStandaloneSegmentUpsert = 1;
constexpr uint8_t kStandaloneSegmentErase = 2;

template <typename T>
bool write_plain_value(std::ofstream &file, T value) {
  file.write(reinterpret_cast<const char *>(&value), sizeof(value));
  return file.good();
}

template <typename T>
bool read_plain_value(std::ifstream &file, T &value) {
  file.read(reinterpret_cast<char *>(&value), sizeof(value));
  return file.good();
}

std::vector<std::string> split_tab_line(const std::string &line) {
  std::vector<std::string> fields;
  size_t begin = 0;
  while (begin <= line.size()) {
    const size_t tab = line.find('\t', begin);
    if (tab == std::string::npos) {
      fields.push_back(line.substr(begin));
      break;
    }
    fields.push_back(line.substr(begin, tab - begin));
    begin = tab + 1;
  }
  return fields;
}

bool parse_manifest_size(const std::string &text, size_t *value) {
  if (value == nullptr) return false;
  uint64_t parsed = 0;
  if (!detail::parse_uint64(text, &parsed) ||
      parsed > std::numeric_limits<size_t>::max()) {
    return false;
  }
  *value = static_cast<size_t>(parsed);
  return true;
}

void remove_file_if_exists(const std::string &path) {
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}

bool file_size_as_size(const std::string &path, size_t *bytes) {
  if (bytes == nullptr) return false;
  std::error_code ec;
  const uintmax_t file_bytes = std::filesystem::file_size(path, ec);
  if (ec || file_bytes > std::numeric_limits<size_t>::max()) return false;
  *bytes = static_cast<size_t>(file_bytes);
  return true;
}

bool copy_or_link_file(const std::string &source, const std::string &target) {
  if (source.empty() || target.empty() || !detail::ensure_parent_directory(target))
    return false;

  remove_file_if_exists(target);
  std::error_code ec;
  std::filesystem::create_hard_link(source, target, ec);
  if (ec) {
    ec.clear();
    std::filesystem::copy_file(source, target,
                               std::filesystem::copy_options::overwrite_existing,
                               ec);
  }
  if (ec) {
    remove_file_if_exists(target);
    return false;
  }

  size_t source_size = 0;
  size_t target_size = 0;
  if (!file_size_as_size(source, &source_size) ||
      !file_size_as_size(target, &target_size) || source_size != target_size) {
    remove_file_if_exists(target);
    return false;
  }
  return true;
}

bool write_generated_docid_file(const std::string &path, uint64_t row_count) {
  if (path.empty() || !detail::ensure_parent_directory(path)) return false;
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.is_open()) return false;
  if (!write_plain_value<uint64_t>(file, row_count)) {
    remove_file_if_exists(path);
    return false;
  }
  for (uint64_t doc_id = 0; doc_id < row_count; ++doc_id) {
    if (!write_plain_value<uint64_t>(file, doc_id)) {
      remove_file_if_exists(path);
      return false;
    }
  }
  file.close();
  if (!file) {
    remove_file_if_exists(path);
    return false;
  }
  return true;
}

bool vector_payload_bytes(size_t entry_count, size_t dimension, size_t *bytes) {
  if (bytes == nullptr) return false;
  if (dimension != 0 &&
      entry_count > (std::numeric_limits<size_t>::max() / dimension) /
                        sizeof(float)) {
    return false;
  }
  *bytes = entry_count * dimension * sizeof(float);
  return true;
}

}  // namespace

bool vector_entry_store::register_index(const std::string &index_name) {
  if (index_name.empty() ||
      m_entry_counts.find(index_name) != m_entry_counts.end()) {
    return false;
  }
  m_entries.emplace(index_name, committed_entries());
  m_entry_counts.emplace(index_name, 0);
  m_index_generations.emplace(index_name, 0);
  return true;
}

bool vector_entry_store::drop_index(const std::string &index_name) {
  const bool existed = m_entry_counts.erase(index_name) > 0;
  m_entries.erase(index_name);
  m_index_generations.erase(index_name);
  m_evicted_indexes.erase(index_name);
  return existed;
}

bool vector_entry_store::rename_index(const std::string &old_index_name,
                                      const std::string &new_index_name) {
  if (old_index_name.empty() || new_index_name.empty() ||
      m_entry_counts.find(new_index_name) != m_entry_counts.end()) {
    return false;
  }
  auto count_it = m_entry_counts.find(old_index_name);
  if (count_it == m_entry_counts.end()) return false;

  auto entries_it = m_entries.find(old_index_name);
  committed_entries entries;
  if (entries_it != m_entries.end()) {
    entries = std::move(entries_it->second);
    m_entries.erase(entries_it);
  }
  m_entries.emplace(new_index_name, std::move(entries));
  const size_t entry_count = count_it->second;
  m_entry_counts.erase(count_it);
  m_entry_counts.emplace(new_index_name, entry_count);
  auto generation_it = m_index_generations.find(old_index_name);
  const uint64_t generation =
      generation_it == m_index_generations.end() ? 0 : generation_it->second;
  if (generation_it != m_index_generations.end())
    m_index_generations.erase(generation_it);
  m_index_generations.emplace(new_index_name, generation);
  if (m_evicted_indexes.erase(old_index_name) > 0)
    m_evicted_indexes.insert(new_index_name);
  return true;
}

bool vector_entry_store::has_index(const std::string &index_name) const {
  return m_entry_counts.find(index_name) != m_entry_counts.end();
}

bool vector_entry_store::clear_index(const std::string &index_name) {
  auto count_it = m_entry_counts.find(index_name);
  if (count_it == m_entry_counts.end()) return false;
  auto entries_it = m_entries.find(index_name);
  if (entries_it == m_entries.end())
    entries_it = m_entries.emplace(index_name, committed_entries()).first;
  const bool changed = count_it->second != 0 || !entries_it->second.empty();
  entries_it->second.clear();
  count_it->second = 0;
  m_evicted_indexes.erase(index_name);
  if (changed) bump_generation(index_name);
  return true;
}

bool vector_entry_store::replace_index(const std::string &index_name,
                                       const committed_entries &entries) {
  auto count_it = m_entry_counts.find(index_name);
  if (count_it == m_entry_counts.end()) return false;
  m_entries[index_name] = entries;
  count_it->second = entries.size();
  m_evicted_indexes.erase(index_name);
  bump_generation(index_name);
  return true;
}

bool vector_entry_store::upsert(const std::string &index_name, uint64_t doc_id,
                                const vector_data &vector) {
  if (!ensure_index_cached(index_name)) return false;
  auto entries_it = m_entries.find(index_name);
  auto count_it = m_entry_counts.find(index_name);
  if (entries_it == m_entries.end() || count_it == m_entry_counts.end())
    return false;
  const auto doc_it = entries_it->second.find(doc_id);
  const bool inserted = doc_it == entries_it->second.end();
  const bool changed = inserted || doc_it->second != vector;
  if (inserted)
    ++count_it->second;
  if (changed) {
    entries_it->second[doc_id] = vector;
    bump_generation(index_name);
  }
  return true;
}

bool vector_entry_store::erase(const std::string &index_name, uint64_t doc_id) {
  if (!ensure_index_cached(index_name)) return false;
  auto entries_it = m_entries.find(index_name);
  auto count_it = m_entry_counts.find(index_name);
  if (entries_it == m_entries.end() || count_it == m_entry_counts.end())
    return false;
  if (entries_it->second.erase(doc_id) > 0) {
    if (count_it->second > 0) --count_it->second;
    bump_generation(index_name);
  }
  return true;
}

bool vector_entry_store::snapshot(committed_state *state) const {
  if (state == nullptr) return false;
  state->clear();
  for (const auto &index_entry : m_entry_counts) {
    committed_entries entries;
    if (!snapshot_index(index_entry.first, &entries)) return false;
    state->emplace(index_entry.first, std::move(entries));
  }
  return true;
}

bool vector_entry_store::snapshot_index(const std::string &index_name,
                                        committed_entries *entries) const {
  if (entries == nullptr) return false;
  if (m_evicted_indexes.find(index_name) != m_evicted_indexes.end()) {
    return load_index_from_truth_store(index_name, entries);
  }
  auto entries_it = m_entries.find(index_name);
  if (entries_it == m_entries.end()) return false;
  *entries = entries_it->second;
  return true;
}

bool vector_entry_store::find_committed_entry(const std::string &index_name,
                                              uint64_t doc_id,
                                              vector_data *vector,
                                              bool *found) const {
  if (vector == nullptr || found == nullptr) return false;
  vector->clear();
  *found = false;

  if (m_entry_counts.find(index_name) == m_entry_counts.end()) return false;

  if (m_evicted_indexes.find(index_name) != m_evicted_indexes.end()) {
    vector_index_truth_store::truth_store *truth_store =
        vector_index_truth_store::get();
    if (truth_store == nullptr) return false;
    vector_index_metadata_store::committed_row row;
    if (!truth_store->find_committed(index_name, doc_id, &row, found)) {
      return false;
    }
    if (*found) *vector = std::move(row.vector);
    return true;
  }

  auto entries_it = m_entries.find(index_name);
  if (entries_it == m_entries.end()) return false;
  const auto doc_it = entries_it->second.find(doc_id);
  if (doc_it == entries_it->second.end()) return true;

  *vector = doc_it->second;
  *found = true;
  return true;
}

bool vector_entry_store::for_each_committed_entry(
    const std::string &index_name, const entry_visitor &visitor) const {
  if (!visitor) return false;
  if (m_evicted_indexes.find(index_name) != m_evicted_indexes.end()) {
    vector_index_truth_store::truth_store *truth_store =
        vector_index_truth_store::get();
    if (truth_store == nullptr) return false;
    return truth_store->for_each_committed(
        index_name,
        [&](const vector_index_metadata_store::committed_row &row) {
          return visitor(row.doc_id, row.vector);
        });
  }
  auto entries_it = m_entries.find(index_name);
  if (entries_it == m_entries.end()) return false;

  std::vector<uint64_t> doc_ids;
  doc_ids.reserve(entries_it->second.size());
  for (const auto &entry : entries_it->second) {
    doc_ids.push_back(entry.first);
  }
  std::sort(doc_ids.begin(), doc_ids.end());

  for (const uint64_t doc_id : doc_ids) {
    const auto doc_it = entries_it->second.find(doc_id);
    if (doc_it == entries_it->second.end()) return false;
    if (!visitor(doc_id, doc_it->second)) return false;
  }
  return true;
}

bool vector_entry_store::evict_until_under_budget(size_t budget) {
  if (budget == std::numeric_limits<size_t>::max()) return true;

  while (memory_bytes() > budget) {
    auto victim_it = m_entries.end();
    size_t victim_bytes = 0;
    for (auto entries_it = m_entries.begin(); entries_it != m_entries.end();
         ++entries_it) {
      if (m_evicted_indexes.find(entries_it->first) != m_evicted_indexes.end())
        continue;
      const size_t bytes = memory_bytes(entries_it->first);
      if (bytes > victim_bytes) {
        victim_it = entries_it;
        victim_bytes = bytes;
      }
    }
    if (victim_it == m_entries.end() || victim_bytes == 0) return true;
    victim_it->second.clear();
    m_evicted_indexes.insert(victim_it->first);
  }
  return true;
}

size_t vector_entry_store::entry_count() const {
  size_t count = 0;
  for (const auto &index_entry : m_entry_counts) {
    count += index_entry.second;
  }
  return count;
}

size_t vector_entry_store::entry_count(const std::string &index_name) const {
  auto count_it = m_entry_counts.find(index_name);
  if (count_it == m_entry_counts.end()) return 0;
  return count_it->second;
}

size_t vector_entry_store::memory_bytes() const {
  size_t bytes = 0;
  for (const auto &index_entry : m_entries) {
    bytes += memory_bytes(index_entry.first);
  }
  return bytes;
}

uint64_t vector_entry_store::generation(const std::string &index_name) const {
  auto generation_it = m_index_generations.find(index_name);
  if (generation_it == m_index_generations.end()) return 0;
  return generation_it->second;
}

bool vector_entry_store::load_index_from_truth_store(
    const std::string &index_name, committed_entries *entries) const {
  if (entries == nullptr) return false;
  entries->clear();
  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (truth_store == nullptr) return false;
  return truth_store->for_each_committed(
      index_name,
      [&](const vector_index_metadata_store::committed_row &row) {
        (*entries)[row.doc_id] = row.vector;
        return true;
      });
}

void vector_entry_store::bump_generation(const std::string &index_name) {
  ++m_index_generations[index_name];
}

bool vector_entry_store::ensure_index_cached(const std::string &index_name) {
  if (m_entry_counts.find(index_name) == m_entry_counts.end()) return false;
  if (m_evicted_indexes.find(index_name) == m_evicted_indexes.end()) {
    if (m_entries.find(index_name) == m_entries.end())
      m_entries.emplace(index_name, committed_entries());
    return true;
  }

  committed_entries entries;
  if (!load_index_from_truth_store(index_name, &entries)) return false;
  m_entries[index_name] = std::move(entries);
  m_entry_counts[index_name] = m_entries[index_name].size();
  m_evicted_indexes.erase(index_name);
  return true;
}

size_t vector_entry_store::memory_bytes(const std::string &index_name) const {
  auto entries_it = m_entries.find(index_name);
  if (entries_it == m_entries.end()) return 0;

  size_t bytes = 0;
  for (const auto &doc_entry : entries_it->second) {
    bytes += doc_entry.second.size() * sizeof(float);
  }
  return bytes;
}

bool standalone_entry_store::register_index(const std::string &index_name,
                                            size_t dimension) {
  if (index_name.empty() || dimension == 0 ||
      m_indexes.find(index_name) != m_indexes.end()) {
    return false;
  }
  index_state state;
  state.dimension = dimension;
  if (!load_manifest(index_name, &state)) return false;
  m_indexes.emplace(index_name, std::move(state));
  return true;
}

bool standalone_entry_store::drop_index(const std::string &index_name) {
  const bool existed = m_indexes.erase(index_name) > 0;
  std::error_code ignored;
  std::filesystem::remove_all(segment_directory(index_name), ignored);
  return existed;
}

bool standalone_entry_store::rename_index(const std::string &old_index_name,
                                          const std::string &new_index_name) {
  if (old_index_name.empty() || new_index_name.empty() ||
      m_indexes.find(new_index_name) != m_indexes.end()) {
    return false;
  }
  auto state_it = m_indexes.find(old_index_name);
  if (state_it == m_indexes.end()) return false;

  index_state renamed_state = state_it->second;
  const std::string old_directory = segment_directory(old_index_name);
  const std::string new_directory = segment_directory(new_index_name);
  if (!old_directory.empty() && !new_directory.empty()) {
    std::error_code ec;
    const bool old_directory_exists =
        std::filesystem::exists(old_directory, ec);
    if (ec) return false;
    if (old_directory_exists) {
      std::filesystem::create_directories(
          std::filesystem::path(new_directory).parent_path(), ec);
      if (ec) return false;
      std::filesystem::rename(old_directory, new_directory, ec);
      if (ec) return false;
      for (standalone_segment &segment : renamed_state.segments) {
        if (!segment.path.empty()) {
          segment.path = (std::filesystem::path(new_directory) /
                          std::filesystem::path(segment.path).filename())
                             .string();
        }
        if (!segment.vector_path.empty()) {
          segment.vector_path =
              (std::filesystem::path(new_directory) /
               std::filesystem::path(segment.vector_path).filename())
                  .string();
        }
        if (!segment.docid_path.empty()) {
          segment.docid_path =
              (std::filesystem::path(new_directory) /
               std::filesystem::path(segment.docid_path).filename())
                  .string();
        }
      }
    }
  }

  if (!save_manifest(new_index_name, renamed_state)) {
    if (!old_directory.empty() && !new_directory.empty()) {
      std::error_code rollback_ec;
      const bool new_directory_exists =
          std::filesystem::exists(new_directory, rollback_ec);
      if (!rollback_ec && new_directory_exists) {
        std::filesystem::rename(new_directory, old_directory, rollback_ec);
      }
    }
    return false;
  }
  m_indexes.erase(state_it);
  m_indexes.emplace(new_index_name, std::move(renamed_state));
  return true;
}

bool standalone_entry_store::has_index(const std::string &index_name) const {
  return m_indexes.find(index_name) != m_indexes.end();
}

bool standalone_entry_store::upsert(const std::string &index_name,
                                    uint64_t doc_id,
                                    const vector_data &vector,
                                    size_t cache_budget) {
  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end() ||
      vector.size() != state_it->second.dimension) {
    return false;
  }

  const bool inserted = state_it->second.live_doc_ids.insert(doc_id).second;
  if (inserted) ++state_it->second.entry_count;
  state_it->second.memory_erases.erase(doc_id);
  auto memory_it = state_it->second.memory_entries.find(doc_id);
  if (memory_it != state_it->second.memory_entries.end()) {
    state_it->second.memory_bytes -= memory_it->second.size() * sizeof(float);
    memory_it->second = vector;
  } else {
    state_it->second.memory_entries.emplace(doc_id, vector);
  }
  state_it->second.memory_bytes += vector.size() * sizeof(float);
  ++state_it->second.generation;
  state_it->second.build_source = build_source_for_state(state_it->second);
  return flush_if_needed(index_name, &state_it->second, cache_budget);
}

bool standalone_entry_store::bulk_upsert(
    const std::string &index_name, const committed_entries &entries) {
  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return false;
  if (entries.empty()) return true;

  for (const auto &entry : entries) {
    if (entry.second.size() != state_it->second.dimension) return false;
  }

  if (!flush_index(index_name, &state_it->second)) return false;
  const index_state before_bulk = state_it->second;

  const std::string path =
      segment_path(index_name, state_it->second.next_segment_id);
  if (path.empty() || !detail::ensure_parent_directory(path)) return false;
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.is_open()) return false;

  const uint64_t dimension = state_it->second.dimension;
  const uint64_t records = entries.size();
  file << kStandaloneSegmentHeader << '\n';
  if (!write_plain_value(file, dimension) ||
      !write_plain_value(file, records)) {
    remove_file_if_exists(path);
    return false;
  }

  std::vector<uint64_t> ordered_doc_ids;
  ordered_doc_ids.reserve(entries.size());
  for (const auto &entry : entries) {
    ordered_doc_ids.push_back(entry.first);
  }
  std::sort(ordered_doc_ids.begin(), ordered_doc_ids.end());

  for (const uint64_t doc_id : ordered_doc_ids) {
    const auto entry_it = entries.find(doc_id);
    if (entry_it == entries.end()) {
      remove_file_if_exists(path);
      return false;
    }
    const uint8_t op = kStandaloneSegmentUpsert;
    if (!write_plain_value(file, op) || !write_plain_value(file, doc_id)) {
      remove_file_if_exists(path);
      return false;
    }
    file.write(reinterpret_cast<const char *>(entry_it->second.data()),
               static_cast<std::streamsize>(entry_it->second.size() *
                                            sizeof(float)));
    if (!file.good()) {
      remove_file_if_exists(path);
      return false;
    }
  }

  file.close();
  if (!file) {
    remove_file_if_exists(path);
    return false;
  }

  std::error_code ec;
  const uintmax_t file_bytes = std::filesystem::file_size(path, ec);
  if (ec) {
    remove_file_if_exists(path);
    return false;
  }

  standalone_segment segment;
  segment.kind = standalone_segment_kind::kDelta;
  segment.path = path;
  segment.record_count = entries.size();
  segment.dimension = state_it->second.dimension;
  segment.bytes = static_cast<size_t>(file_bytes);
  segment.generation = state_it->second.generation + 1;
  state_it->second.segments.push_back(std::move(segment));
  ++state_it->second.next_segment_id;
  for (const auto &entry : entries) {
    if (state_it->second.live_doc_ids.insert(entry.first).second) {
      ++state_it->second.entry_count;
    }
  }
  state_it->second.generation = state_it->second.segments.back().generation;
  state_it->second.build_source = build_source_for_state(state_it->second);

  if (!save_manifest(index_name, state_it->second)) {
    state_it->second = before_bulk;
    remove_file_if_exists(path);
    return false;
  }
  return true;
}

bool standalone_entry_store::bulk_upsert_raw_files(
    const std::string &index_name, const std::string &vector_filename,
    const std::string &docid_filename, uint64_t row_count, size_t dimension,
    std::unordered_set<uint64_t> *loaded_doc_ids) {
  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end() || dimension != state_it->second.dimension)
    return false;
  if (row_count == 0) return true;
  if (row_count > std::numeric_limits<size_t>::max()) return false;

  vector_load_file_info file_info;
  std::string load_error;
  if (!read_fbin_file_info(vector_filename, state_it->second.dimension,
                           &file_info, &load_error) ||
      file_info.row_count != row_count ||
      file_info.dimension != state_it->second.dimension) {
    return false;
  }

  if (!flush_index(index_name, &state_it->second)) return false;
  const index_state before_bulk = state_it->second;

  const uint64_t segment_id = state_it->second.next_segment_id;
  const std::string target_vector_path = raw_vector_path(index_name, segment_id);
  const std::string target_docid_path = raw_docid_path(index_name, segment_id);
  if (target_vector_path.empty() || target_docid_path.empty()) return false;

  if (!copy_or_link_file(vector_filename, target_vector_path)) return false;
  bool docid_ready = false;
  if (docid_filename.empty()) {
    docid_ready = write_generated_docid_file(target_docid_path, row_count);
  } else {
    docid_ready = copy_or_link_file(docid_filename, target_docid_path);
  }
  if (!docid_ready) {
    remove_file_if_exists(target_vector_path);
    return false;
  }

  size_t vector_bytes = 0;
  size_t docid_bytes = 0;
  if (!file_size_as_size(target_vector_path, &vector_bytes) ||
      !file_size_as_size(target_docid_path, &docid_bytes) ||
      vector_bytes > std::numeric_limits<size_t>::max() - docid_bytes) {
    state_it->second = before_bulk;
    remove_file_if_exists(target_vector_path);
    remove_file_if_exists(target_docid_path);
    return false;
  }

  standalone_segment segment;
  segment.kind = standalone_segment_kind::kRawFbin;
  segment.vector_path = target_vector_path;
  segment.docid_path = target_docid_path;
  segment.record_count = static_cast<size_t>(row_count);
  segment.dimension = state_it->second.dimension;
  segment.bytes = vector_bytes + docid_bytes;
  segment.generation = state_it->second.generation + 1;
  state_it->second.segments.push_back(std::move(segment));
  ++state_it->second.next_segment_id;

  if (loaded_doc_ids != nullptr) {
    state_it->second.live_doc_ids.merge(*loaded_doc_ids);
    loaded_doc_ids->clear();
  } else {
    if (!read_docid_values(
            target_docid_path, row_count, &load_error,
            [&state_it](uint64_t doc_id) {
              state_it->second.live_doc_ids.insert(doc_id);
              return true;
            })) {
      state_it->second = before_bulk;
      remove_file_if_exists(target_vector_path);
      remove_file_if_exists(target_docid_path);
      return false;
    }
  }
  state_it->second.entry_count = state_it->second.live_doc_ids.size();
  state_it->second.generation = state_it->second.segments.back().generation;
  state_it->second.build_source = build_source_for_state(state_it->second);

  if (!save_manifest(index_name, state_it->second)) {
    state_it->second = before_bulk;
    remove_file_if_exists(target_vector_path);
    remove_file_if_exists(target_docid_path);
    return false;
  }
  return true;
}

bool standalone_entry_store::erase(const std::string &index_name,
                                   uint64_t doc_id, size_t cache_budget) {
  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return false;

  auto memory_it = state_it->second.memory_entries.find(doc_id);
  if (memory_it != state_it->second.memory_entries.end()) {
    state_it->second.memory_bytes -= memory_it->second.size() * sizeof(float);
    state_it->second.memory_entries.erase(memory_it);
  }
  if (state_it->second.live_doc_ids.erase(doc_id) > 0) {
    if (state_it->second.entry_count > 0) --state_it->second.entry_count;
    state_it->second.memory_erases.insert(doc_id);
    ++state_it->second.generation;
    state_it->second.build_source = build_source_for_state(state_it->second);
  }
  return flush_if_needed(index_name, &state_it->second, cache_budget);
}

bool standalone_entry_store::can_rebuild_direct_from_raw_segments(
    const index_state &state) const {
  if (!state.memory_entries.empty() || !state.memory_erases.empty()) return false;
  if (state.segments.empty()) return state.entry_count == 0;

  size_t raw_rows = 0;
  for (const standalone_segment &segment : state.segments) {
    if (segment.kind != standalone_segment_kind::kRawFbin ||
        segment.dimension != state.dimension ||
        segment.record_count > std::numeric_limits<size_t>::max() - raw_rows) {
      return false;
    }
    raw_rows += segment.record_count;
  }
  return raw_rows == state.entry_count;
}

bool standalone_entry_store::read_raw_segments(
    const index_state &state, const raw_vector_segment_visitor &visitor) const {
  if (!visitor) return false;
  for (const standalone_segment &segment : state.segments) {
    if (segment.kind != standalone_segment_kind::kRawFbin ||
        segment.dimension != state.dimension) {
      return false;
    }

    size_t vector_bytes = 0;
    size_t docid_bytes = 0;
    if (!file_size_as_size(segment.vector_path, &vector_bytes) ||
        !file_size_as_size(segment.docid_path, &docid_bytes) ||
        vector_bytes > std::numeric_limits<size_t>::max() - docid_bytes ||
        vector_bytes + docid_bytes != segment.bytes) {
      return false;
    }

    raw_vector_segment raw_segment;
    raw_segment.vector_path = segment.vector_path;
    raw_segment.docid_path = segment.docid_path;
    raw_segment.row_count = segment.record_count;
    raw_segment.dimension = segment.dimension;
    raw_segment.bytes = segment.bytes;
    raw_segment.generation = segment.generation;
    if (!visitor(raw_segment)) return false;
  }
  return true;
}

std::string standalone_entry_store::build_source_for_state(
    const index_state &state) const {
  if (state.entry_count == 0 && state.segments.empty() &&
      state.memory_entries.empty() && state.memory_erases.empty()) {
    return "memory";
  }
  if (can_rebuild_direct_from_raw_segments(state)) return "raw_segments_direct";
  bool has_raw_segment = false;
  for (const standalone_segment &segment : state.segments) {
    if (segment.kind == standalone_segment_kind::kRawFbin) {
      has_raw_segment = true;
      break;
    }
  }
  return has_raw_segment ? "raw_segments_compacted" : "delta_replay";
}

bool standalone_entry_store::compact_to_raw_segment(const std::string &index_name,
                                                    index_state *state) {
  if (state == nullptr) return false;

  committed_entries entries;
  if (!load_entries(*state, &entries)) return false;
  if (entries.size() > std::numeric_limits<uint32_t>::max()) return false;

  const index_state before_compact = *state;
  if (entries.empty()) {
    state->segments.clear();
    state->memory_entries.clear();
    state->memory_erases.clear();
    state->memory_bytes = 0;
    state->live_doc_ids.clear();
    state->entry_count = 0;
    state->generation = before_compact.generation + 1;
    state->build_source = "memory";
    if (!save_manifest(index_name, *state)) {
      *state = before_compact;
      return false;
    }
    return true;
  }

  const uint64_t segment_id = state->next_segment_id;
  const std::string vector_path = raw_vector_path(index_name, segment_id);
  const std::string docid_path = raw_docid_path(index_name, segment_id);
  if (vector_path.empty() || docid_path.empty() ||
      !detail::ensure_parent_directory(vector_path) ||
      !detail::ensure_parent_directory(docid_path)) {
    return false;
  }

  std::vector<uint64_t> doc_ids;
  doc_ids.reserve(entries.size());
  for (const auto &entry : entries) doc_ids.push_back(entry.first);
  std::sort(doc_ids.begin(), doc_ids.end());

  std::ofstream vector_file(vector_path,
                            std::ios::out | std::ios::binary | std::ios::trunc);
  std::ofstream docid_file(docid_path,
                           std::ios::out | std::ios::binary | std::ios::trunc);
  if (!vector_file.is_open() || !docid_file.is_open()) {
    remove_file_if_exists(vector_path);
    remove_file_if_exists(docid_path);
    return false;
  }

  const auto row_count = static_cast<uint32_t>(doc_ids.size());
  if (!write_plain_value(vector_file, row_count) ||
      !write_plain_value(vector_file, static_cast<uint32_t>(state->dimension)) ||
      !write_plain_value(docid_file, static_cast<uint64_t>(doc_ids.size()))) {
    remove_file_if_exists(vector_path);
    remove_file_if_exists(docid_path);
    return false;
  }

  for (const uint64_t doc_id : doc_ids) {
    const auto entry_it = entries.find(doc_id);
    if (entry_it == entries.end() || entry_it->second.size() != state->dimension ||
        !write_plain_value(docid_file, doc_id)) {
      remove_file_if_exists(vector_path);
      remove_file_if_exists(docid_path);
      return false;
    }
    const size_t vector_bytes = state->dimension * sizeof(float);
    if (vector_bytes >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      remove_file_if_exists(vector_path);
      remove_file_if_exists(docid_path);
      return false;
    }
    vector_file.write(reinterpret_cast<const char *>(entry_it->second.data()),
                      static_cast<std::streamsize>(vector_bytes));
    if (!vector_file.good()) {
      remove_file_if_exists(vector_path);
      remove_file_if_exists(docid_path);
      return false;
    }
  }

  vector_file.close();
  docid_file.close();
  if (!vector_file || !docid_file) {
    remove_file_if_exists(vector_path);
    remove_file_if_exists(docid_path);
    return false;
  }

  size_t vector_bytes = 0;
  size_t docid_bytes = 0;
  if (!file_size_as_size(vector_path, &vector_bytes) ||
      !file_size_as_size(docid_path, &docid_bytes) ||
      vector_bytes > std::numeric_limits<size_t>::max() - docid_bytes) {
    remove_file_if_exists(vector_path);
    remove_file_if_exists(docid_path);
    return false;
  }

  standalone_segment segment;
  segment.kind = standalone_segment_kind::kRawFbin;
  segment.vector_path = vector_path;
  segment.docid_path = docid_path;
  segment.record_count = entries.size();
  segment.dimension = state->dimension;
  segment.bytes = vector_bytes + docid_bytes;
  segment.generation = state->generation + 1;

  state->segments.clear();
  if (!entries.empty()) state->segments.push_back(std::move(segment));
  state->memory_entries.clear();
  state->memory_erases.clear();
  state->memory_bytes = 0;
  state->live_doc_ids.clear();
  for (uint64_t doc_id : doc_ids) state->live_doc_ids.insert(doc_id);
  state->entry_count = entries.size();
  state->generation = before_compact.generation + 1;
  ++state->next_segment_id;
  state->build_source = "raw_segments_compacted";

  if (!save_manifest(index_name, *state)) {
    *state = before_compact;
    remove_file_if_exists(vector_path);
    remove_file_if_exists(docid_path);
    return false;
  }
  return true;
}

bool standalone_entry_store::materialized_rebuild_fits_budget(
    const index_state &state) const {
  const uint64_t budget = opt_vector_entry_cache_size;
  if (budget == 0) return true;

  size_t estimated_bytes = 0;
  if (!vector_payload_bytes(state.entry_count, state.dimension,
                            &estimated_bytes)) {
    return false;
  }
  return static_cast<uint64_t>(estimated_bytes) <= budget;
}

bool standalone_entry_store::prepare_raw_segments_for_rebuild(
    const std::string &index_name) {
  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return false;
  if (!flush_index(index_name, &state_it->second)) return false;
  if (!can_rebuild_direct_from_raw_segments(state_it->second)) {
    if (!materialized_rebuild_fits_budget(state_it->second) ||
        !compact_to_raw_segment(index_name, &state_it->second)) {
      return false;
    }
  }

  const bool compacted =
      state_it->second.build_source == "raw_segments_compacted";
  if (!compacted) {
    state_it->second.build_source =
        build_source_for_state(state_it->second);
    if (!save_manifest(index_name, state_it->second)) return false;
  }
  return true;
}

bool standalone_entry_store::read_rebuild_raw_segments(
    const std::string &index_name,
    const raw_vector_segment_visitor &visitor) const {
  const auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return false;
  return read_raw_segments(state_it->second, visitor);
}

bool standalone_entry_store::rebuild_backend_input(
    const std::string &index_name, backend *target) {
  if (target == nullptr) return false;
  if (!prepare_raw_segments_for_rebuild(index_name)) return false;

  const bool rebuilt = target->rebuild_from_raw_segments(
      [this, &index_name](const raw_vector_segment_visitor &visitor) {
        return read_rebuild_raw_segments(index_name, visitor);
      });
  if (rebuilt) return true;

  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return false;
  committed_entries entries;
  return load_entries(state_it->second, &entries) &&
         target->rebuild_from_committed_entries(entries);
}

bool standalone_entry_store::for_each_entry(
    const std::string &index_name, const entry_visitor &visitor) const {
  if (!visitor) return false;
  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return false;

  committed_entries entries;
  if (!load_entries(state_it->second, &entries)) return false;

  std::vector<uint64_t> doc_ids;
  doc_ids.reserve(entries.size());
  for (const auto &entry : entries) {
    doc_ids.push_back(entry.first);
  }
  std::sort(doc_ids.begin(), doc_ids.end());
  for (const uint64_t doc_id : doc_ids) {
    const auto doc_it = entries.find(doc_id);
    if (doc_it == entries.end()) return false;
    if (!visitor(doc_id, doc_it->second)) return false;
  }
  return true;
}

bool standalone_entry_store::find_entry(const std::string &index_name,
                                        uint64_t doc_id, vector_data *vector,
                                        bool *found) const {
  if (vector == nullptr || found == nullptr) return false;
  vector->clear();
  *found = false;

  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return false;
  if (state_it->second.live_doc_ids.find(doc_id) ==
      state_it->second.live_doc_ids.end()) {
    return true;
  }
  auto memory_it = state_it->second.memory_entries.find(doc_id);
  if (memory_it != state_it->second.memory_entries.end()) {
    *vector = memory_it->second;
    *found = true;
    return true;
  }
  if (state_it->second.memory_erases.find(doc_id) !=
      state_it->second.memory_erases.end()) {
    return true;
  }

  committed_entries entries;
  if (!load_entries(state_it->second, &entries)) return false;
  const auto entry_it = entries.find(doc_id);
  if (entry_it == entries.end()) return true;
  *vector = entry_it->second;
  *found = true;
  return true;
}

size_t standalone_entry_store::entry_count(
    const std::string &index_name) const {
  auto state_it = m_indexes.find(index_name);
  return state_it == m_indexes.end() ? 0 : state_it->second.entry_count;
}

size_t standalone_entry_store::memory_bytes(
    const std::string &index_name) const {
  auto state_it = m_indexes.find(index_name);
  return state_it == m_indexes.end() ? 0 : state_it->second.memory_bytes;
}

size_t standalone_entry_store::segment_count(
    const std::string &index_name) const {
  auto state_it = m_indexes.find(index_name);
  return state_it == m_indexes.end() ? 0 : state_it->second.segments.size();
}

size_t standalone_entry_store::segment_bytes(
    const std::string &index_name) const {
  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return 0;
  size_t bytes = 0;
  for (const standalone_segment &segment : state_it->second.segments) {
    bytes += segment.bytes;
  }
  return bytes;
}

size_t standalone_entry_store::raw_segment_count(
    const std::string &index_name) const {
  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return 0;
  size_t count = 0;
  for (const standalone_segment &segment : state_it->second.segments) {
    if (segment.kind == standalone_segment_kind::kRawFbin) ++count;
  }
  return count;
}

size_t standalone_entry_store::raw_segment_bytes(
    const std::string &index_name) const {
  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return 0;
  size_t bytes = 0;
  for (const standalone_segment &segment : state_it->second.segments) {
    if (segment.kind == standalone_segment_kind::kRawFbin) {
      bytes += segment.bytes;
    }
  }
  return bytes;
}

std::string standalone_entry_store::build_source(
    const std::string &index_name) const {
  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return "";
  return state_it->second.build_source.empty()
             ? build_source_for_state(state_it->second)
             : state_it->second.build_source;
}

uint64_t standalone_entry_store::generation(
    const std::string &index_name) const {
  auto state_it = m_indexes.find(index_name);
  return state_it == m_indexes.end() ? 0 : state_it->second.generation;
}

bool standalone_entry_store::flush_index(const std::string &index_name,
                                         index_state *state) {
  if (state == nullptr) return false;
  const size_t record_count =
      state->memory_entries.size() + state->memory_erases.size();
  if (record_count == 0) return true;

  const std::string path = segment_path(index_name, state->next_segment_id);
  if (path.empty() || !detail::ensure_parent_directory(path)) return false;
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.is_open()) return false;

  const uint64_t dimension = state->dimension;
  const uint64_t records = record_count;
  file << kStandaloneSegmentHeader << '\n';
  if (!write_plain_value(file, dimension) ||
      !write_plain_value(file, records)) {
    remove_file_if_exists(path);
    return false;
  }

  std::map<uint64_t, const vector_data *> ordered_upserts;
  for (const auto &entry : state->memory_entries) {
    ordered_upserts.emplace(entry.first, &entry.second);
  }
  for (const auto &entry : ordered_upserts) {
    const uint8_t op = kStandaloneSegmentUpsert;
    const uint64_t doc_id = entry.first;
    if (!write_plain_value(file, op) || !write_plain_value(file, doc_id)) {
      remove_file_if_exists(path);
      return false;
    }
    file.write(reinterpret_cast<const char *>(entry.second->data()),
               static_cast<std::streamsize>(entry.second->size() *
                                            sizeof(float)));
    if (!file.good()) {
      remove_file_if_exists(path);
      return false;
    }
  }

  std::vector<uint64_t> ordered_erases(state->memory_erases.begin(),
                                       state->memory_erases.end());
  std::sort(ordered_erases.begin(), ordered_erases.end());
  for (const uint64_t doc_id : ordered_erases) {
    const uint8_t op = kStandaloneSegmentErase;
    if (!write_plain_value(file, op) || !write_plain_value(file, doc_id)) {
      remove_file_if_exists(path);
      return false;
    }
  }

  file.close();
  if (!file) {
    remove_file_if_exists(path);
    return false;
  }

  std::error_code ec;
  const uintmax_t file_bytes = std::filesystem::file_size(path, ec);
  if (ec) {
    remove_file_if_exists(path);
    return false;
  }
  standalone_segment segment;
  segment.kind = standalone_segment_kind::kDelta;
  segment.path = path;
  segment.record_count = record_count;
  segment.dimension = state->dimension;
  segment.bytes = static_cast<size_t>(file_bytes);
  segment.generation = state->generation;
  state->segments.push_back(std::move(segment));
  ++state->next_segment_id;
  state->memory_entries.clear();
  state->memory_erases.clear();
  state->memory_bytes = 0;
  state->build_source = build_source_for_state(*state);
  return save_manifest(index_name, *state);
}

bool standalone_entry_store::flush_if_needed(const std::string &index_name,
                                             index_state *state,
                                             size_t cache_budget) {
  if (state == nullptr) return false;
  if (cache_budget == std::numeric_limits<size_t>::max()) return true;
  if (memory_bytes(index_name) <= cache_budget &&
      (cache_budget != 0 || state->memory_erases.empty())) {
    return true;
  }
  return flush_index(index_name, state);
}

bool standalone_entry_store::load_manifest(const std::string &index_name,
                                           index_state *state) const {
  if (state == nullptr || state->dimension == 0) return false;
  const std::string path = manifest_path(index_name);
  if (path.empty()) return false;

  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) return !ec;

  std::ifstream file(path, std::ios::in);
  if (!file.is_open()) return false;

  std::string header;
  if (!std::getline(file, header) || header != kStandaloneManifestHeader) {
    return false;
  }

  index_state loaded;
  loaded.dimension = state->dimension;
  uint64_t manifest_dimension = 0;
  size_t expected_entry_count = 0;
  size_t expected_segment_count = 0;
  bool saw_dimension = false;
  bool saw_entry_count = false;
  bool saw_generation = false;
  bool saw_next_segment_id = false;
  bool saw_segment_count = false;

  std::string line;
  while (std::getline(file, line)) {
    const std::vector<std::string> fields = split_tab_line(line);
    if (fields.empty()) return false;

    if (fields[0] == "dimension") {
      if (fields.size() != 2 || !detail::parse_uint64(fields[1],
                                                       &manifest_dimension)) {
        return false;
      }
      saw_dimension = true;
      continue;
    }
    if (fields[0] == "entry_count") {
      if (fields.size() != 2 ||
          !parse_manifest_size(fields[1], &expected_entry_count)) {
        return false;
      }
      saw_entry_count = true;
      continue;
    }
    if (fields[0] == "generation") {
      if (fields.size() != 2 ||
          !detail::parse_uint64(fields[1], &loaded.generation)) {
        return false;
      }
      saw_generation = true;
      continue;
    }
    if (fields[0] == "next_segment_id") {
      if (fields.size() != 2 ||
          !detail::parse_uint64(fields[1], &loaded.next_segment_id) ||
          loaded.next_segment_id == 0) {
        return false;
      }
      saw_next_segment_id = true;
      continue;
    }
    if (fields[0] == "segment_count") {
      if (fields.size() != 2 ||
          !parse_manifest_size(fields[1], &expected_segment_count)) {
        return false;
      }
      saw_segment_count = true;
      continue;
    }
    if (fields[0] == "build_source") {
      if (fields.size() != 2) return false;
      loaded.build_source = fields[1];
      continue;
    }
    if (fields[0] == "segment" || fields[0] == "segment_delta") {
      if (fields[0] == "segment" && fields.size() != 4) return false;
      if (fields[0] == "segment_delta" && fields.size() != 5) return false;
      standalone_segment segment;
      segment.kind = standalone_segment_kind::kDelta;
      const std::filesystem::path filename(fields[1]);
      if (filename.has_parent_path() ||
          !parse_manifest_size(fields[2], &segment.record_count) ||
          !parse_manifest_size(fields[3], &segment.bytes)) {
        return false;
      }
      segment.dimension = loaded.dimension;
      if (fields[0] == "segment_delta") {
        if (!detail::parse_uint64(fields[4], &segment.generation))
          return false;
      }
      segment.path =
          (std::filesystem::path(segment_directory(index_name)) / filename)
              .string();
      const uintmax_t actual_size =
          std::filesystem::file_size(segment.path, ec);
      if (ec || actual_size != segment.bytes) return false;
      loaded.segments.push_back(std::move(segment));
      continue;
    }
    if (fields[0] == "segment_raw_fbin") {
      if (fields.size() != 7) return false;
      standalone_segment segment;
      segment.kind = standalone_segment_kind::kRawFbin;
      const std::filesystem::path vector_filename(fields[1]);
      const std::filesystem::path docid_filename(fields[2]);
      size_t segment_dimension = 0;
      if (vector_filename.has_parent_path() || docid_filename.has_parent_path() ||
          !parse_manifest_size(fields[3], &segment.record_count) ||
          !parse_manifest_size(fields[4], &segment_dimension) ||
          !parse_manifest_size(fields[5], &segment.bytes) ||
          !detail::parse_uint64(fields[6], &segment.generation) ||
          segment_dimension != loaded.dimension) {
        return false;
      }
      segment.dimension = segment_dimension;
      const std::filesystem::path directory(segment_directory(index_name));
      segment.vector_path = (directory / vector_filename).string();
      segment.docid_path = (directory / docid_filename).string();
      size_t vector_bytes = 0;
      size_t docid_bytes = 0;
      if (!file_size_as_size(segment.vector_path, &vector_bytes) ||
          !file_size_as_size(segment.docid_path, &docid_bytes) ||
          vector_bytes > std::numeric_limits<size_t>::max() - docid_bytes ||
          vector_bytes + docid_bytes != segment.bytes) {
        return false;
      }
      loaded.segments.push_back(std::move(segment));
      continue;
    }
    return false;
  }

  if (!saw_dimension || !saw_entry_count || !saw_generation ||
      !saw_next_segment_id || !saw_segment_count ||
      manifest_dimension != state->dimension ||
      loaded.segments.size() != expected_segment_count) {
    return false;
  }

  committed_entries entries;
  if (!replay_segments(loaded, &entries) ||
      entries.size() != expected_entry_count) {
    return false;
  }
  for (const auto &entry : entries) {
    loaded.live_doc_ids.insert(entry.first);
  }
  loaded.entry_count = loaded.live_doc_ids.size();
  if (loaded.build_source.empty()) {
    loaded.build_source = build_source_for_state(loaded);
  }
  *state = std::move(loaded);
  return true;
}

bool standalone_entry_store::save_manifest(const std::string &index_name,
                                           const index_state &state) const {
  const std::string path = manifest_path(index_name);
  if (path.empty() || !detail::ensure_parent_directory(path)) return false;

  const std::string tmp_path = path + ".tmp";
  std::ofstream file(tmp_path, std::ios::out | std::ios::trunc);
  if (!file.is_open()) return false;
  file << kStandaloneManifestHeader << '\n'
       << "dimension\t" << state.dimension << '\n'
       << "entry_count\t" << state.entry_count << '\n'
       << "generation\t" << state.generation << '\n'
       << "next_segment_id\t" << state.next_segment_id << '\n'
       << "segment_count\t" << state.segments.size() << '\n'
       << "build_source\t" << (state.build_source.empty()
                                   ? build_source_for_state(state)
                                   : state.build_source)
       << '\n';
  for (const standalone_segment &segment : state.segments) {
    if (segment.kind == standalone_segment_kind::kRawFbin) {
      file << "segment_raw_fbin\t"
           << std::filesystem::path(segment.vector_path).filename().string()
           << '\t'
           << std::filesystem::path(segment.docid_path).filename().string()
           << '\t' << segment.record_count << '\t' << segment.dimension
           << '\t' << segment.bytes << '\t' << segment.generation << '\n';
    } else {
      file << "segment_delta\t"
           << std::filesystem::path(segment.path).filename().string() << '\t'
           << segment.record_count << '\t' << segment.bytes << '\t'
           << segment.generation << '\n';
    }
  }
  file.close();
  if (!file) {
    remove_file_if_exists(tmp_path);
    return false;
  }

  std::error_code ec;
  std::filesystem::rename(tmp_path, path, ec);
  if (ec) {
    ec.clear();
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tmp_path, path, ec);
  }
  if (ec) {
    remove_file_if_exists(tmp_path);
    return false;
  }
  return true;
}

bool standalone_entry_store::replay_segments(const index_state &state,
                                             committed_entries *entries) const {
  if (entries == nullptr) return false;
  entries->clear();
  for (const standalone_segment &segment : state.segments) {
    if (segment.kind == standalone_segment_kind::kRawFbin) {
      size_t vector_bytes = 0;
      size_t docid_bytes = 0;
      if (!file_size_as_size(segment.vector_path, &vector_bytes) ||
          !file_size_as_size(segment.docid_path, &docid_bytes) ||
          vector_bytes > std::numeric_limits<size_t>::max() - docid_bytes ||
          vector_bytes + docid_bytes != segment.bytes ||
          segment.dimension != state.dimension) {
        return false;
      }

      vector_load_file_info info;
      std::string error;
      if (!read_fbin_vectors(
              segment.vector_path, segment.docid_path, state.dimension, &info,
              &error,
              [entries](uint64_t doc_id, const float *values,
                        size_t dimension) {
                (*entries)[doc_id] = vector_data(values, values + dimension);
                return true;
              }) ||
          info.row_count != segment.record_count ||
          info.dimension != state.dimension) {
        return false;
      }
      continue;
    }

    std::error_code ec;
    const uintmax_t actual_size = std::filesystem::file_size(segment.path, ec);
    if (ec || actual_size != segment.bytes) return false;

    std::ifstream file(segment.path, std::ios::in | std::ios::binary);
    if (!file.is_open()) return false;
    std::string header;
    if (!std::getline(file, header) || header != kStandaloneSegmentHeader) {
      return false;
    }
    uint64_t dimension = 0;
    uint64_t record_count = 0;
    if (!read_plain_value(file, dimension) ||
        !read_plain_value(file, record_count) ||
        dimension != state.dimension || record_count != segment.record_count) {
      return false;
    }
    for (uint64_t i = 0; i < record_count; ++i) {
      uint8_t op = 0;
      uint64_t doc_id = 0;
      if (!read_plain_value(file, op) || !read_plain_value(file, doc_id)) {
        return false;
      }
      if (op == kStandaloneSegmentUpsert) {
        vector_data vector(state.dimension);
        file.read(reinterpret_cast<char *>(vector.data()),
                  static_cast<std::streamsize>(state.dimension *
                                               sizeof(float)));
        if (!file.good()) return false;
        (*entries)[doc_id] = std::move(vector);
      } else if (op == kStandaloneSegmentErase) {
        entries->erase(doc_id);
      } else {
        return false;
      }
    }
  }
  return true;
}

bool standalone_entry_store::load_entries(const index_state &state,
                                          committed_entries *entries) const {
  if (!replay_segments(state, entries)) return false;
  for (const auto &entry : state.memory_entries) {
    (*entries)[entry.first] = entry.second;
  }
  for (const uint64_t doc_id : state.memory_erases) {
    entries->erase(doc_id);
  }
  return entries->size() == state.entry_count;
}

std::string standalone_entry_store::segment_directory(
    const std::string &index_name) const {
  const std::string root = detail::vector_index_root_path();
  if (root.empty() || index_name.empty()) return "";
  std::filesystem::path directory(root);
  directory /= detail::encoded_index_name(index_name);
  directory /= "standalone_segments";
  return directory.string();
}

std::string standalone_entry_store::manifest_path(
    const std::string &index_name) const {
  const std::string directory = segment_directory(index_name);
  if (directory.empty()) return "";
  std::filesystem::path path(directory);
  path /= "manifest.v1";
  return path.string();
}

std::string standalone_entry_store::segment_path(
    const std::string &index_name, uint64_t segment_id) const {
  const std::string directory = segment_directory(index_name);
  if (directory.empty() || segment_id == 0) return "";
  std::filesystem::path path(directory);
  path /= "segment-" + std::to_string(segment_id) + ".vseg";
  return path.string();
}

std::string standalone_entry_store::raw_vector_path(
    const std::string &index_name, uint64_t segment_id) const {
  const std::string directory = segment_directory(index_name);
  if (directory.empty() || segment_id == 0) return "";
  std::filesystem::path path(directory);
  path /= "raw-segment-" + std::to_string(segment_id) + ".fbin";
  return path.string();
}

std::string standalone_entry_store::raw_docid_path(
    const std::string &index_name, uint64_t segment_id) const {
  const std::string directory = segment_directory(index_name);
  if (directory.empty() || segment_id == 0) return "";
  std::filesystem::path path(directory);
  path /= "raw-segment-" + std::to_string(segment_id) + ".u64";
  return path.string();
}

namespace detail {

diskann_build_mode effective_diskann_build_mode(
    const vector_index::index_service::index_config &config) {
  if (config.provider != backend_provider::kDiskAnn ||
      config.mode != backend_mode::kExternal) {
    return diskann_build_mode::kAuto;
  }
  if (config.diskann_build_mode_specified) {
    return config.diskann_build_mode_value;
  }
  return global_diskann_build_mode();
}

bool config_accepts_diskann_build_mode(
    const vector_index::index_service::index_config &config) {
  return config.provider == backend_provider::kDiskAnn &&
         config.mode == backend_mode::kExternal;
}

bool config_rejects_diskann_build_mode(
    const vector_index::index_service::index_config &config) {
  return config.diskann_build_mode_specified &&
         !config_accepts_diskann_build_mode(config) &&
         config.diskann_build_mode_value != diskann_build_mode::kAuto;
}

void normalize_diskann_build_mode(
    vector_index::index_service::index_config *config) {
  if (config == nullptr || config_accepts_diskann_build_mode(*config)) {
    return;
  }
  config->diskann_build_mode_value = diskann_build_mode::kAuto;
  config->diskann_build_mode_specified = false;
}

std::unique_ptr<backend> build_backend_from_config(
    const std::string &index_name,
    const vector_index::index_service::index_config &config) {
  if (config_rejects_diskann_build_mode(config)) {
    return nullptr;
  }

  std::unique_ptr<backend> backend =
      create_backend(config.dimension, config.metric, config.mode,
                     config.provider, config.consistency_mode, index_name);
  if (backend == nullptr) return nullptr;
  if (config.hnsw_m != 0 && config.hnsw_ef_construction != 0 &&
      !backend->set_hnsw_build_params(config.hnsw_m,
                                      config.hnsw_ef_construction)) {
    return nullptr;
  }
  if (config.hnsw_build_threads != 0 &&
      !backend->set_hnsw_build_threads(config.hnsw_build_threads)) {
    return nullptr;
  }
  if (config.faiss_build_threads != 0 &&
      !backend->set_faiss_build_threads(config.faiss_build_threads)) {
    return nullptr;
  }
  const bool faiss_ivf_params_set =
      backend->set_faiss_ivf_params(config.faiss_nlist, config.faiss_nprobe);
  if (!faiss_ivf_params_set &&
      (config.faiss_nlist != 0 || config.faiss_nprobe != 0)) {
    return nullptr;
  }
  if (config.faiss_pq_m != 0 || config.faiss_pq_bits != 0) {
    if (!backend->set_faiss_ivf_pq_params(
            config.faiss_nlist, config.faiss_nprobe, config.faiss_pq_m,
            config.faiss_pq_bits)) {
      return nullptr;
    }
  }
  if (config.diskann_max_degree != 0 && config.diskann_build_complexity != 0 &&
      !backend->set_diskann_build_params(config.diskann_max_degree,
                                         config.diskann_build_complexity,
                                         config.diskann_build_threads)) {
    return nullptr;
  }
  if (config.diskann_build_threads != 0 &&
      !backend->set_diskann_build_threads(config.diskann_build_threads)) {
    return nullptr;
  }
  if (!backend->set_diskann_build_mode(effective_diskann_build_mode(config))) {
    return nullptr;
  }
  if ((config.provider == backend_provider::kDiskAnn ||
       config.diskann_pq_code_budget_size != 0) &&
      !backend->set_diskann_pq_code_budget_size(
          config.diskann_pq_code_budget_size)) {
    return nullptr;
  }
  if (config.diskann_search_complexity != 0 &&
      !backend->set_diskann_search_complexity(
          config.diskann_search_complexity)) {
    return nullptr;
  }
  if (config.diskann_search_beamwidth != 0 &&
      !backend->set_diskann_search_beamwidth(config.diskann_search_beamwidth)) {
    return nullptr;
  }
  if (config.search_ef != 0 && !backend->set_search_ef(config.search_ef))
    return nullptr;
  return backend;
}

}  // namespace detail

namespace {

using detail::build_backend_from_config;
using detail::all_true;

constexpr const char *LIFECYCLE_READY = "ready";
constexpr const char *LIFECYCLE_BULK_LOADING = "bulk_loading";
constexpr const char *LIFECYCLE_REBUILDING = "rebuilding";
constexpr const char *LIFECYCLE_RECOVERING = "recovering";
constexpr const char *LIFECYCLE_FAILED = "failed";
constexpr uint32_t ERROR_NONE = 0;
constexpr uint32_t ERROR_PENDING_CHANGES = 1001;
constexpr uint32_t ERROR_BACKEND_CREATE_FAILED = 1002;
constexpr uint32_t ERROR_BACKEND_RECOVER_FAILED = 1003;
constexpr uint32_t ERROR_REPLAY_STATE_FAILED = 1004;
constexpr const char *kSegmentedBackendVariant = "segmented";

uint64_t now_unix_epoch_seconds() {
  return static_cast<uint64_t>(std::time(nullptr));
}

void mark_lifecycle_state(
    vector_index::index_service::lifecycle_info *lifecycle, const char *state) {
  lifecycle->state = state;
  ++lifecycle->version;
}

void mark_lifecycle_failure(
    vector_index::index_service::lifecycle_info *lifecycle,
    uint32_t error_code) {
  mark_lifecycle_state(lifecycle, LIFECYCLE_FAILED);
  lifecycle->last_error_code = error_code;
  lifecycle->last_error_ts = now_unix_epoch_seconds();
}

void mark_lifecycle_ready(
    vector_index::index_service::lifecycle_info *lifecycle) {
  mark_lifecycle_state(lifecycle, LIFECYCLE_READY);
  lifecycle->last_error_code = ERROR_NONE;
  lifecycle->last_error_ts = 0;
}

void mark_lifecycle_bulk_loading(
    vector_index::index_service::lifecycle_info *lifecycle) {
  mark_lifecycle_state(lifecycle, LIFECYCLE_BULK_LOADING);
  lifecycle->last_error_code = ERROR_NONE;
  lifecycle->last_error_ts = 0;
}

void mark_recover_fallback(
    vector_index::index_service::lifecycle_info *lifecycle,
    bool record_global_status) {
  ++lifecycle->recover_fallback_count;
  lifecycle->last_recover_fallback_ts = now_unix_epoch_seconds();
  if (record_global_status) vector_status::record_backend_recover_fallback();
}

void set_bulk_load_error(std::string *error, const std::string &message) {
  if (error != nullptr) *error = message;
}

bool uses_deferred_standalone_mutations(
    const vector_index::index_service::index_config &config) {
  return config.consistency_mode == index_consistency_mode::kStandalone;
}

void sync_search_config_from_backend(
    const backend &index_backend,
    vector_index::index_service::index_config *config) {
  config->backend_variant = index_backend.backend_variant();
  config->search_ef = index_backend.search_ef();
}

void sync_hnsw_config_from_backend(
    const backend &index_backend,
    vector_index::index_service::index_config *config) {
  config->hnsw_m = index_backend.hnsw_m();
  config->hnsw_ef_construction = index_backend.hnsw_ef_construction();
  config->hnsw_build_threads = index_backend.hnsw_build_threads();
}

void sync_faiss_config_from_backend(
    const backend &index_backend,
    vector_index::index_service::index_config *config) {
  config->faiss_nlist = index_backend.faiss_nlist();
  config->faiss_nprobe = index_backend.faiss_nprobe();
  config->faiss_pq_m = index_backend.faiss_pq_m();
  config->faiss_pq_bits = index_backend.faiss_pq_bits();
  config->faiss_build_threads = index_backend.faiss_build_threads();
}

void sync_diskann_build_config_from_backend(
    const backend &index_backend,
    vector_index::index_service::index_config *config) {
  config->diskann_max_degree = index_backend.diskann_max_degree();
  config->diskann_build_complexity = index_backend.diskann_build_complexity();
  config->diskann_build_threads = index_backend.diskann_build_threads();
  config->diskann_build_mode_value =
      index_backend.diskann_build_mode_value();
  config->diskann_pq_code_budget_size =
      index_backend.diskann_pq_code_budget_size();
}

template <typename pending_changes_map>
bool pending_changes_contain_index(const pending_changes_map &pending_changes,
                                   const std::string &index_name) {
  for (const auto &txn_changes : pending_changes) {
    for (const auto &change : txn_changes.second) {
      if (change.index_name == index_name) return true;
    }
  }
  return false;
}

template <typename pending_changes_map>
bool has_any_pending_changes(const pending_changes_map &pending_changes) {
  for (const auto &txn_changes : pending_changes) {
    if (!txn_changes.second.empty()) return true;
  }
  return false;
}

bool lazy_external_runtime_enabled() {
  return opt_vector_lazy_external_runtime;
}

committed_entry_reader make_committed_entry_reader(
    const vector_entry_store &entry_store, const std::string &index_name) {
  return [&entry_store, index_name](const committed_entry_visitor &visitor) {
    return entry_store.for_each_committed_entry(index_name, visitor);
  };
}

bool load_backend_from_store(const vector_entry_store &entry_store,
                             const std::string &index_name, backend *target) {
  return target != nullptr &&
         target->load_committed_entries_from_reader(
             make_committed_entry_reader(entry_store, index_name));
}

bool rebuild_backend_from_store(const vector_entry_store &entry_store,
                                const std::string &index_name,
                                backend *target) {
  return target != nullptr &&
         target->rebuild_from_committed_entries_from_reader(
             make_committed_entry_reader(entry_store, index_name));
}

bool recover_backend_from_store(const vector_entry_store &entry_store,
                                const std::string &index_name,
                                backend *target) {
  return target != nullptr &&
         target->recover_committed_entries_from_reader(
             make_committed_entry_reader(entry_store, index_name));
}

bool rebuild_backend_from_source(const vector_entry_store &entry_store,
                                 standalone_entry_store &standalone_store,
                                 const std::string &index_name,
                                 const index_service::index_config &config,
                                 backend *target) {
  return config.consistency_mode == index_consistency_mode::kStandalone
             ? standalone_store.rebuild_backend_input(index_name, target)
             : rebuild_backend_from_store(entry_store, index_name, target);
}

bool recover_backend_from_source(const vector_entry_store &entry_store,
                                 standalone_entry_store &standalone_store,
                                 const std::string &index_name,
                                 const index_service::index_config &config,
                                 backend *target) {
  return config.consistency_mode == index_consistency_mode::kStandalone
             ? standalone_store.rebuild_backend_input(index_name, target)
             : recover_backend_from_store(entry_store, index_name, target);
}

class segmented_backend final : public backend {
 public:
  segmented_backend(index_service::index_config config,
                    std::vector<std::shared_ptr<backend>> segments,
                    backend_build_diagnostics diagnostics)
      : m_config(std::move(config)),
        m_segments(std::move(segments)),
        m_diagnostics(std::move(diagnostics)) {}

  bool upsert(uint64_t doc_id [[maybe_unused]],
              const vector_data &vector [[maybe_unused]]) override {
    return false;
  }

  bool erase(uint64_t doc_id [[maybe_unused]]) override { return false; }

  bool search(const vector_data &query, size_t top_k,
              std::vector<search_result> *results) const override {
    if (results == nullptr) return false;
    results->clear();
    if (query.size() != m_config.dimension) return false;
    if (top_k == 0) return true;

    std::vector<std::vector<search_result>> segment_results;
    segment_results.reserve(m_segments.size());
    for (const auto &segment : m_segments) {
      if (segment == nullptr) return false;
      std::vector<search_result> current;
      if (!segment->search(query, top_k, &current)) return false;
      segment_results.push_back(std::move(current));
    }
    return merge_segment_topk(segment_results, top_k, results);
  }

  bool search_batch(const std::vector<vector_data> &queries, size_t top_k,
                    std::vector<std::vector<search_result>> *results)
      const override {
    if (results == nullptr) return false;
    results->clear();
    results->resize(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
      if (!search(queries[i], top_k, &(*results)[i])) return false;
    }
    return true;
  }

  size_t entry_count() const override {
    size_t total = 0;
    for (const auto &segment : m_segments) {
      if (segment == nullptr) return 0;
      const size_t segment_count = segment->entry_count();
      if (segment_count > std::numeric_limits<size_t>::max() - total) {
        return std::numeric_limits<size_t>::max();
      }
      total += segment_count;
    }
    return total;
  }

  size_t dimension() const override { return m_config.dimension; }
  metric_type metric() const override { return m_config.metric; }
  backend_mode mode() const override { return m_config.mode; }
  backend_provider provider() const override { return m_config.provider; }
  std::string backend_variant() const override {
    return kSegmentedBackendVariant;
  }
  backend_build_diagnostics build_diagnostics() const override {
    return m_diagnostics;
  }
  bool set_search_ef(uint32_t search_ef) override {
    for (const auto &segment : m_segments) {
      if (segment != nullptr && !segment->set_search_ef(search_ef)) {
        return false;
      }
    }
    m_config.search_ef = search_ef;
    return true;
  }
  uint32_t search_ef() const override { return m_config.search_ef; }
  uint32_t hnsw_m() const override { return m_config.hnsw_m; }
  uint32_t hnsw_ef_construction() const override {
    return m_config.hnsw_ef_construction;
  }
  uint32_t hnsw_build_threads() const override {
    return m_config.hnsw_build_threads;
  }
  uint32_t faiss_nlist() const override { return m_config.faiss_nlist; }
  uint32_t faiss_nprobe() const override { return m_config.faiss_nprobe; }
  uint32_t faiss_pq_m() const override { return m_config.faiss_pq_m; }
  uint32_t faiss_pq_bits() const override { return m_config.faiss_pq_bits; }
  uint32_t faiss_build_threads() const override {
    return m_config.faiss_build_threads;
  }
  uint32_t diskann_max_degree() const override {
    return m_config.diskann_max_degree;
  }
  uint32_t diskann_build_complexity() const override {
    return m_config.diskann_build_complexity;
  }
  uint32_t diskann_build_threads() const override {
    return m_config.diskann_build_threads;
  }
  diskann_build_mode diskann_build_mode_value() const override {
    return m_config.diskann_build_mode_value;
  }
  uint32_t diskann_search_complexity() const override {
    return m_config.diskann_search_complexity;
  }
  uint32_t diskann_search_beamwidth() const override {
    return m_config.diskann_search_beamwidth;
  }
  uint64_t diskann_pq_code_budget_size() const override {
    return m_config.diskann_pq_code_budget_size;
  }
  bool supports_mutations() const override { return false; }

 private:
  index_service::index_config m_config;
  std::vector<std::shared_ptr<backend>> m_segments;
  backend_build_diagnostics m_diagnostics;
};

bool segmented_backend_contract_valid_impl() {
  index_service::index_config config;
  config.dimension = 2;
  config.metric = metric_type::kEuclidean;
  config.mode = backend_mode::kMemory;
  config.provider = backend_provider::kNative;
  config.diskann_build_mode_value = diskann_build_mode::kOffline;

  segmented_backend backend(config, {}, {});
  return !backend.supports_mutations() &&
         !backend.upsert(1, vector_data{1.0F, 0.0F}) &&
         !backend.erase(1) && backend.set_search_ef(32) &&
         backend.search_ef() == 32 &&
         backend.diskann_build_mode_value() == diskann_build_mode::kOffline;
}

std::string segment_backend_index_name(const std::string &index_name,
                                       uint64_t segment_id) {
  return index_name + "#segment-" + std::to_string(segment_id);
}

vector_index_metadata_store::segment_task_row make_segment_task_row(
    const std::string &index_name, const raw_vector_segment &segment,
    uint64_t segment_id, vector_index_metadata_store::segment_task_state state) {
  vector_index_metadata_store::segment_task_row row;
  row.index_name = index_name;
  row.generation = segment.generation;
  row.segment_id = segment_id;
  row.state = state;
  row.row_count = segment.row_count;
  row.payload_size = segment.bytes;
  row.vector_path = segment.vector_path;
  row.docid_path = segment.docid_path;
  row.updated_ts = now_unix_epoch_seconds();
  return row;
}

bool build_one_segment_backend(
    const std::string &index_name, const index_service::index_config &config,
    const raw_vector_segment &segment, uint64_t ordinal,
    std::shared_ptr<backend> *built_segment,
    backend_build_diagnostics *diagnostics,
    vector_index_metadata_store::segment_task_row *task_row) {
  if (built_segment == nullptr || diagnostics == nullptr ||
      task_row == nullptr) {
    return false;
  }
  const uint64_t segment_id = segment.generation == 0 ? ordinal : segment.generation;
  *task_row = make_segment_task_row(
      index_name, segment, segment_id,
      vector_index_metadata_store::segment_task_state::kBuilding);
  task_row->attempt = 1;

  std::unique_ptr<backend> segment_backend =
      build_backend_from_config(segment_backend_index_name(index_name, segment_id),
                                config);
  if (segment_backend == nullptr) {
    task_row->state = vector_index_metadata_store::segment_task_state::kFailed;
    task_row->last_error_code = ERROR_BACKEND_CREATE_FAILED;
    task_row->updated_ts = now_unix_epoch_seconds();
    return false;
  }

  segment_build_input input;
  input.segment = segment;
  input.segment_id = segment_id;
  input.generation = segment.generation;
  input.artifact_prefix = segment_backend_index_name(index_name, segment_id);

  segment_build_result result;
  if (!segment_backend->build_segment_from_raw(input, &result) ||
      !segment_backend->load_segment_handle(result)) {
    task_row->state = vector_index_metadata_store::segment_task_state::kFailed;
    task_row->last_error_code = ERROR_REPLAY_STATE_FAILED;
    task_row->updated_ts = now_unix_epoch_seconds();
    return false;
  }
  task_row->state = vector_index_metadata_store::segment_task_state::kReady;
  task_row->row_count = result.row_count;
  task_row->payload_size = result.payload_size;
  task_row->vector_path = result.vector_path;
  task_row->docid_path = result.docid_path;
  if (!result.artifact_prefix.empty()) {
    std::error_code ec;
    if (std::filesystem::exists(result.artifact_prefix, ec) && !ec) {
      task_row->artifact_prefix = result.artifact_prefix;
    }
  }
  task_row->last_error_code = ERROR_NONE;
  task_row->updated_ts = now_unix_epoch_seconds();
  *diagnostics = result.diagnostics;
  *built_segment = std::move(segment_backend);
  return true;
}

std::unique_ptr<backend> build_segmented_backend_from_standalone(
    const std::string &index_name, const index_service::index_config &config,
    standalone_entry_store *standalone_store,
    std::vector<vector_index_metadata_store::segment_task_row> *task_rows) {
  if (standalone_store == nullptr ||
      config.consistency_mode != index_consistency_mode::kStandalone) {
    return nullptr;
  }
  if (task_rows != nullptr) task_rows->clear();
  if (!standalone_store->prepare_raw_segments_for_rebuild(index_name)) {
    return nullptr;
  }

  std::vector<std::shared_ptr<backend>> segments;
  backend_build_diagnostics aggregate;
  aggregate.runtime = kSegmentedBackendVariant;
  aggregate.input_source = "raw_segments";

  uint64_t ordinal = 0;
  const bool read_ok = standalone_store->read_rebuild_raw_segments(
      index_name, [&](const raw_vector_segment &segment) {
        ++ordinal;
        std::shared_ptr<backend> segment_backend;
        backend_build_diagnostics diagnostics;
        vector_index_metadata_store::segment_task_row task_row =
            make_segment_task_row(
                index_name, segment,
                segment.generation == 0 ? ordinal : segment.generation,
                vector_index_metadata_store::segment_task_state::kPending);
        if (task_rows != nullptr) task_rows->push_back(task_row);
        if (!build_one_segment_backend(index_name, config, segment, ordinal,
                                       &segment_backend, &diagnostics,
                                       &task_row)) {
          if (task_rows != nullptr && !task_rows->empty()) {
            task_rows->back() = task_row;
          }
          return false;
        }
        if (task_rows != nullptr && !task_rows->empty()) {
          task_rows->back() = task_row;
        }
        aggregate.row_count += diagnostics.row_count == 0
                                   ? segment.row_count
                                   : diagnostics.row_count;
        aggregate.segment_count += diagnostics.segment_count == 0
                                       ? 1
                                       : diagnostics.segment_count;
        aggregate.manifest_ms += diagnostics.manifest_ms;
        aggregate.offline_build_ms += diagnostics.offline_build_ms;
        aggregate.load_ms += diagnostics.load_ms;
        if (!diagnostics.fallback_reason.empty()) {
          aggregate.fallback_reason = diagnostics.fallback_reason;
        }
        segments.push_back(std::move(segment_backend));
        return true;
      });
  if (!read_ok) return nullptr;

  return std::make_unique<segmented_backend>(config, std::move(segments),
                                             std::move(aggregate));
}

bool should_build_segmented(const index_service::index_config &config,
                            const build_pipeline_decision &decision) {
  return decision.path == build_pipeline_path::kSegmented &&
         config.consistency_mode == index_consistency_mode::kStandalone;
}

bool entry_store_vectors_match_dimension(const vector_entry_store &entry_store,
                                         const std::string &index_name,
                                         size_t dimension) {
  return entry_store.for_each_committed_entry(
      index_name, [dimension](uint64_t doc_id [[maybe_unused]],
                              const vector_data &vector) {
        return vector.size() == dimension;
      });
}
}  // namespace

bool segmented_backend_contract_valid_for_testing() {
  return segmented_backend_contract_valid_impl();
}

bool index_service::ensure_runtime_loaded(const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name))) {
    return false;
  }
  if (!lazy_external_runtime_enabled() ||
      config_it->second.mode != backend_mode::kExternal) {
    return true;
  }
  const size_t committed_count = m_entry_store.entry_count(index_name);
  if (committed_count == 0 ||
      index_it->second->entry_count() == committed_count)
    return true;

  std::unique_ptr<backend> loaded =
      build_backend_from_config(index_name, config_it->second);
  if (loaded == nullptr) return false;
  if (!rebuild_backend_from_source(m_entry_store, m_standalone_store,
                                   index_name, config_it->second,
                                   loaded.get()))
    return false;
  index_it->second = std::move(loaded);
  return true;
}

bool index_service::ensure_runtime_loaded_for_search(
    const std::string &index_name) {
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (lifecycle_it != m_lifecycle_infos.end() &&
      lifecycle_it->second.state == LIFECYCLE_BULK_LOADING) {
    return false;
  }
  return ensure_runtime_loaded(index_name);
}

bool index_service::rebuild_runtime_from_store_for_search(
    const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                lifecycle_it != m_lifecycle_infos.end(),
                m_entry_store.has_index(index_name))) {
    return false;
  }
  if (lifecycle_it->second.state == LIFECYCLE_BULK_LOADING ||
      pending_changes_contain_index(m_pending_changes, index_name)) {
    return false;
  }

  std::unique_ptr<backend> rebuilt =
      build_backend_from_config(index_name, config_it->second);
  if (rebuilt == nullptr ||
      !rebuild_backend_from_store(m_entry_store, index_name, rebuilt.get())) {
    return false;
  }

  index_it->second = std::move(rebuilt);
  mark_recover_fallback(&lifecycle_it->second, true);
  return true;
}

void index_service::maybe_unload_runtime(const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return;
  }
  if (!lazy_external_runtime_enabled() ||
      config_it->second.mode != backend_mode::kExternal) {
    return;
  }
  if (index_it->second->backend_variant() == kSegmentedBackendVariant) {
    return;
  }
  std::unique_ptr<backend> unloaded =
      build_backend_from_config(index_name, config_it->second);
  if (unloaded == nullptr) return;
  index_it->second = std::move(unloaded);
}

build_input_stats index_service::collect_build_input_stats(
    const std::string &index_name, const index_config &config) const {
  build_input_stats stats;
  const size_t rows =
      config.consistency_mode == index_consistency_mode::kStandalone
          ? m_standalone_store.entry_count(index_name)
          : m_entry_store.entry_count(index_name);
  stats.row_count = static_cast<uint64_t>(rows);

  const size_t payload_bytes =
      saturated_mul_size(saturated_mul_size(rows, config.dimension),
                         sizeof(float));
  stats.payload_size = payload_bytes == std::numeric_limits<size_t>::max()
                           ? std::numeric_limits<uint64_t>::max()
                           : static_cast<uint64_t>(payload_bytes);

  if (config.consistency_mode == index_consistency_mode::kStandalone) {
    const uint64_t raw_segment_count =
        static_cast<uint64_t>(m_standalone_store.raw_segment_count(index_name));
    stats.has_raw_segments = raw_segment_count > 0;
    stats.raw_segment_count =
        raw_segment_count == 0 && stats.row_count > 0 ? 1 : raw_segment_count;
  }
  return stats;
}

build_pipeline_decision index_service::record_build_pipeline_decision(
    const std::string &index_name, const index_config &config) {
  const build_pipeline_runtime_config runtime_config =
      global_build_pipeline_runtime_config();
  const build_input_stats stats = collect_build_input_stats(index_name, config);
  const build_pipeline_decision decision =
      select_build_pipeline(stats, runtime_config.thresholds);

  build_pipeline_snapshot snapshot;
  snapshot.mode = build_pipeline_mode_name(runtime_config.thresholds.mode);
  snapshot.decision = build_pipeline_path_name(decision.path);
  snapshot.trigger = build_pipeline_trigger_name(decision.trigger);
  snapshot.row_count = stats.row_count;
  snapshot.payload_size = stats.payload_size;
  snapshot.raw_segment_count = stats.raw_segment_count;
  m_build_pipeline_snapshots[index_name] = std::move(snapshot);
  return decision;
}

bool index_service::register_index(const std::string &index_name,
                                   std::unique_ptr<backend> backend) {
  if (backend == nullptr || index_name.empty()) return false;

  index_config config{backend->dimension(),
                      backend->metric(),
                      backend->mode(),
                      backend->provider(),
                      backend->backend_variant(),
                      backend->search_ef(),
                      backend->hnsw_m(),
                      backend->hnsw_ef_construction(),
                      backend->hnsw_build_threads(),
                      backend->faiss_nlist(),
                      backend->faiss_nprobe(),
                      backend->faiss_pq_m(),
                      backend->faiss_pq_bits(),
                      backend->faiss_build_threads(),
                      backend->diskann_max_degree(),
                      backend->diskann_build_complexity(),
                      backend->diskann_build_threads(),
                      backend->diskann_build_mode_value(),
                      backend->diskann_search_complexity(),
                      backend->diskann_search_beamwidth(),
                      backend->diskann_pq_code_budget_size(),
                      backend->provider() == backend_provider::kDiskAnn &&
                          backend->diskann_build_mode_value() !=
                              diskann_build_mode::kAuto,
                      index_consistency_mode::kTransactional};
  return register_index_impl(index_name, std::move(config), std::move(backend));
}

bool index_service::register_index_impl(const std::string &index_name,
                                        index_config config,
                                        std::unique_ptr<backend> backend) {
  auto [it, inserted] = m_indexes.emplace(index_name, std::move(backend));
  if (!inserted || it->second == nullptr) return false;
  if (!m_entry_store.register_index(index_name)) {
    m_indexes.erase(it);
    return false;
  }
  if (!m_standalone_store.register_index(index_name, config.dimension)) {
    m_entry_store.drop_index(index_name);
    m_indexes.erase(it);
    return false;
  }

  config.backend_variant = it->second->backend_variant();
  m_index_configs[index_name] = std::move(config);
  m_lifecycle_infos[index_name] =
      lifecycle_info{LIFECYCLE_READY, 1, ERROR_NONE, 0, 0, 0, 0};
  m_build_pipeline_snapshots[index_name] = build_pipeline_snapshot{};
  m_segment_task_rows[index_name] = {};
  return true;
}

bool index_service::register_index(const std::string &index_name,
                                   const index_config &config) {
  if (config.dimension == 0) return false;

  std::unique_ptr<backend> backend =
      build_backend_from_config(index_name, config);
  if (backend == nullptr) return false;
  index_config normalized_config = config;
  detail::normalize_diskann_build_mode(&normalized_config);

  return register_index_impl(index_name, std::move(normalized_config),
                             std::move(backend));
}

bool index_service::register_index_from_strings(const std::string &index_name,
                                                size_t dimension,
                                                const std::string &metric,
                                                const std::string &mode,
                                                const std::string &provider) {
  metric_type metric_value = metric_type::kEuclidean;
  backend_mode mode_value = backend_mode::kMemory;
  backend_provider provider_value = backend_provider::kNative;

  if (!parse_metric(metric, &metric_value)) return false;
  if (provider.empty()) {
    if (!default_backend_provider(&provider_value)) return false;
  } else if (!parse_backend_provider(provider, &provider_value)) {
    return false;
  }
  if (mode.empty()) {
    if (!default_backend_mode_for_provider(provider_value, &mode_value))
      return false;
  } else if (!parse_backend_mode(mode, &mode_value)) {
    return false;
  }

  index_config config;
  config.dimension = dimension;
  config.metric = metric_value;
  config.mode = mode_value;
  config.provider = provider_value;
  return register_index(index_name, config);
}

bool index_service::drop_index(const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  if (config_it == m_index_configs.end()) return false;
  if (!remove_backend_artifacts(index_name, config_it->second.mode,
                                config_it->second.provider)) {
    return false;
  }
  return unregister_index(index_name);
}

bool index_service::unregister_index(const std::string &index_name) {
  auto index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end()) return false;

  m_indexes.erase(index_it);
  m_index_configs.erase(index_name);
  m_entry_store.drop_index(index_name);
  m_standalone_store.drop_index(index_name);
  m_lifecycle_infos.erase(index_name);
  m_build_pipeline_snapshots.erase(index_name);
  m_segment_task_rows.erase(index_name);
  return true;
}

bool index_service::rename_index(const std::string &old_index_name,
                                 const std::string &new_index_name) {
  if (old_index_name.empty() || new_index_name.empty()) return false;
  if (old_index_name == new_index_name) return true;

  auto old_index_it = m_indexes.find(old_index_name);
  auto old_config_it = m_index_configs.find(old_index_name);
  auto old_lifecycle_it = m_lifecycle_infos.find(old_index_name);
  if (old_index_it == m_indexes.end() ||
      old_config_it == m_index_configs.end() ||
      !m_entry_store.has_index(old_index_name) ||
      old_lifecycle_it == m_lifecycle_infos.end()) {
    return false;
  }
  if (m_indexes.find(new_index_name) != m_indexes.end()) return false;

  const index_config config = old_config_it->second;
  const lifecycle_info lifecycle = old_lifecycle_it->second;
  build_pipeline_snapshot pipeline_snapshot;
  bool has_pipeline_snapshot = false;
  auto old_pipeline_it = m_build_pipeline_snapshots.find(old_index_name);
  if (old_pipeline_it != m_build_pipeline_snapshots.end()) {
    pipeline_snapshot = old_pipeline_it->second;
    has_pipeline_snapshot = true;
  }
  std::vector<vector_index_metadata_store::segment_task_row> segment_tasks;
  bool has_segment_tasks = false;
  auto old_segment_task_it = m_segment_task_rows.find(old_index_name);
  if (old_segment_task_it != m_segment_task_rows.end()) {
    segment_tasks = old_segment_task_it->second;
    for (auto &row : segment_tasks) row.index_name = new_index_name;
    has_segment_tasks = true;
  }

  std::unique_ptr<backend> renamed_backend =
      build_backend_from_config(new_index_name, config);
  if (renamed_backend == nullptr) return false;
  if (!rebuild_backend_from_source(m_entry_store, m_standalone_store,
                                   old_index_name, config,
                                   renamed_backend.get()))
    return false;
  if (!m_entry_store.rename_index(old_index_name, new_index_name)) return false;
  if (!m_standalone_store.rename_index(old_index_name, new_index_name)) {
    (void)m_entry_store.rename_index(new_index_name, old_index_name);
    return false;
  }

  m_indexes.erase(old_index_it);
  m_index_configs.erase(old_config_it);
  m_lifecycle_infos.erase(old_lifecycle_it);
  m_build_pipeline_snapshots.erase(old_index_name);
  m_segment_task_rows.erase(old_index_name);

  m_indexes.emplace(new_index_name, std::move(renamed_backend));
  m_index_configs.emplace(new_index_name, config);
  m_lifecycle_infos.emplace(new_index_name, lifecycle);
  if (has_pipeline_snapshot) {
    m_build_pipeline_snapshots.emplace(new_index_name,
                                       std::move(pipeline_snapshot));
  }
  if (has_segment_tasks) {
    m_segment_task_rows.emplace(new_index_name, std::move(segment_tasks));
  }
  maybe_unload_runtime(new_index_name);

  for (auto &txn_changes : m_pending_changes) {
    for (pending_change &change : txn_changes.second) {
      if (change.index_name == old_index_name)
        change.index_name = new_index_name;
    }
  }
  return true;
}

bool index_service::begin_bulk_load(const std::string &index_name) {
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (index_it == m_indexes.end() || lifecycle_it == m_lifecycle_infos.end())
    return false;

  if (pending_changes_contain_index(m_pending_changes, index_name)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_PENDING_CHANGES);
    return false;
  }

  mark_lifecycle_state(&lifecycle_it->second, LIFECYCLE_BULK_LOADING);
  lifecycle_it->second.last_error_code = ERROR_NONE;
  lifecycle_it->second.last_error_ts = 0;
  return true;
}

bool index_service::rebuild_index(const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end())) {
    return false;
  }
  mark_lifecycle_state(&lifecycle_it->second, LIFECYCLE_REBUILDING);

  // Rebuild currently requires no staged writes against this index.
  if (pending_changes_contain_index(m_pending_changes, index_name)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_PENDING_CHANGES);
    return false;
  }

  const build_pipeline_decision decision =
      record_build_pipeline_decision(index_name, config_it->second);
  std::vector<vector_index_metadata_store::segment_task_row> segment_tasks;
  std::unique_ptr<backend> rebuilt =
      should_build_segmented(config_it->second, decision)
          ? build_segmented_backend_from_standalone(
                index_name, config_it->second, &m_standalone_store,
                &segment_tasks)
          : build_backend_from_config(index_name, config_it->second);
  if (should_build_segmented(config_it->second, decision)) {
    m_segment_task_rows[index_name] = segment_tasks;
  } else {
    m_segment_task_rows[index_name].clear();
  }
  if (rebuilt == nullptr) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_BACKEND_CREATE_FAILED);
    return false;
  }
  if (!should_build_segmented(config_it->second, decision) &&
      !rebuild_backend_from_source(m_entry_store, m_standalone_store,
                                   index_name, config_it->second,
                                   rebuilt.get())) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    return false;
  }

  DBUG_EXECUTE_IF("vector_segment_task_before_publish", return false;);
  index_it->second = std::move(rebuilt);
  mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

bool index_service::recover_index(const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end())) {
    return false;
  }
  mark_lifecycle_state(&lifecycle_it->second, LIFECYCLE_RECOVERING);

  // Recovery for one index requires no staged writes against this index.
  if (pending_changes_contain_index(m_pending_changes, index_name)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_PENDING_CHANGES);
    return false;
  }

  std::unique_ptr<backend> recovered =
      build_backend_from_config(index_name, config_it->second);
  if (recovered == nullptr ||
      !recover_backend_from_source(m_entry_store, m_standalone_store,
                                   index_name, config_it->second,
                                   recovered.get())) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_BACKEND_RECOVER_FAILED);
    return false;
  }
  const bool used_recover_fallback = recovered->last_recover_used_fallback();
  if (used_recover_fallback) {
    mark_recover_fallback(&lifecycle_it->second, false);
  }

  index_it->second = std::move(recovered);
  mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

bool index_service::set_search_ef(const std::string &index_name,
                                  uint32_t search_ef) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (!index_it->second->set_search_ef(search_ef)) return false;
  sync_search_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_hnsw_build_params(const std::string &index_name,
                                          uint32_t hnsw_m,
                                          uint32_t hnsw_ef_construction) {
  if (hnsw_m == 0 || hnsw_ef_construction == 0) return false;

  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name))) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  index_config next_config = config_it->second;
  next_config.hnsw_m = hnsw_m;
  next_config.hnsw_ef_construction = hnsw_ef_construction;
  std::unique_ptr<backend> rebuilt =
      build_backend_from_config(index_name, next_config);
  if (rebuilt == nullptr) return false;
  if (!load_backend_from_store(m_entry_store, index_name, rebuilt.get()))
    return false;

  index_it->second = std::move(rebuilt);
  config_it->second = next_config;
  sync_search_config_from_backend(*index_it->second, &config_it->second);
  sync_hnsw_config_from_backend(*index_it->second, &config_it->second);
  sync_faiss_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_hnsw_build_threads(const std::string &index_name,
                                           uint32_t hnsw_build_threads) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;
  if (!index_it->second->set_hnsw_build_threads(hnsw_build_threads))
    return false;
  sync_hnsw_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_faiss_ivf_params(const std::string &index_name,
                                         uint32_t faiss_nlist,
                                         uint32_t faiss_nprobe) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (config_it->second.provider != backend_provider::kFaiss) return false;
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  if (!index_it->second->set_faiss_ivf_params(faiss_nlist, faiss_nprobe))
    return false;
  sync_search_config_from_backend(*index_it->second, &config_it->second);
  sync_hnsw_config_from_backend(*index_it->second, &config_it->second);
  sync_faiss_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_faiss_ivf_pq_params(const std::string &index_name,
                                            uint32_t faiss_nlist,
                                            uint32_t faiss_nprobe,
                                            uint32_t faiss_pq_m,
                                            uint32_t faiss_pq_bits) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  if (!index_it->second->set_faiss_ivf_pq_params(faiss_nlist, faiss_nprobe,
                                                 faiss_pq_m, faiss_pq_bits)) {
    return false;
  }
  sync_search_config_from_backend(*index_it->second, &config_it->second);
  sync_hnsw_config_from_backend(*index_it->second, &config_it->second);
  sync_faiss_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_faiss_build_threads(
    const std::string &index_name, uint32_t faiss_build_threads) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  if (!index_it->second->set_faiss_build_threads(faiss_build_threads)) {
    return false;
  }
  sync_faiss_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_diskann_build_params(
    const std::string &index_name, uint32_t diskann_max_degree,
    uint32_t diskann_build_complexity, uint32_t diskann_build_threads) {
  if (diskann_max_degree == 0 || diskann_build_complexity == 0) return false;

  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  if (!index_it->second->set_diskann_build_params(
          diskann_max_degree, diskann_build_complexity,
          diskann_build_threads)) {
    return false;
  }
  sync_diskann_build_config_from_backend(*index_it->second,
                                         &config_it->second);
  sync_search_config_from_backend(*index_it->second, &config_it->second);
  sync_hnsw_config_from_backend(*index_it->second, &config_it->second);
  return true;
}

bool index_service::set_diskann_build_threads(
    const std::string &index_name, uint32_t diskann_build_threads) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  if (!index_it->second->set_diskann_build_threads(diskann_build_threads)) {
    return false;
  }
  sync_diskann_build_config_from_backend(*index_it->second,
                                         &config_it->second);
  return true;
}

bool index_service::set_diskann_build_mode(
    const std::string &index_name,
    diskann_build_mode diskann_build_mode_value) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (!detail::config_accepts_diskann_build_mode(config_it->second)) {
    return diskann_build_mode_value == diskann_build_mode::kAuto;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;

  if (!index_it->second->set_diskann_build_mode(diskann_build_mode_value)) {
    return false;
  }
  sync_diskann_build_config_from_backend(*index_it->second,
                                         &config_it->second);
  config_it->second.diskann_build_mode_specified = true;
  return true;
}

bool index_service::set_diskann_search_complexity(
    const std::string &index_name, uint32_t diskann_search_complexity) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (!index_it->second->set_diskann_search_complexity(
          diskann_search_complexity))
    return false;
  config_it->second.diskann_search_complexity =
      index_it->second->diskann_search_complexity();
  return true;
}

bool index_service::set_diskann_search_beamwidth(
    const std::string &index_name, uint32_t diskann_search_beamwidth) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (!index_it->second->set_diskann_search_beamwidth(
          diskann_search_beamwidth))
    return false;
  config_it->second.diskann_search_beamwidth =
      index_it->second->diskann_search_beamwidth();
  return true;
}

bool index_service::set_diskann_pq_code_budget_size(
    const std::string &index_name, uint64_t diskann_pq_code_budget_size) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;
  if (!index_it->second->set_diskann_pq_code_budget_size(
          diskann_pq_code_budget_size)) {
    return false;
  }
  config_it->second.diskann_pq_code_budget_size =
      index_it->second->diskann_pq_code_budget_size();
  return true;
}

bool index_service::rebuild_all_indexes(size_t *rebuilt_count) {
  if (rebuilt_count == nullptr) return false;

  if (has_any_pending_changes(m_pending_changes)) return false;

  std::unordered_map<std::string, std::unique_ptr<backend>> rebuilt_backends;
  rebuilt_backends.reserve(m_index_configs.size());
  for (const auto &entry : m_index_configs) {
    const std::string &index_name = entry.first;
    const index_config &config = entry.second;
    if (m_indexes.find(index_name) == m_indexes.end()) return false;
    if (!m_entry_store.has_index(index_name)) return false;
    if (m_lifecycle_infos.find(index_name) == m_lifecycle_infos.end())
      return false;

    const build_pipeline_decision decision =
        record_build_pipeline_decision(index_name, config);
    std::vector<vector_index_metadata_store::segment_task_row> segment_tasks;
    std::unique_ptr<backend> rebuilt =
        should_build_segmented(config, decision)
            ? build_segmented_backend_from_standalone(
                  index_name, config, &m_standalone_store, &segment_tasks)
            : build_backend_from_config(index_name, config);
    if (should_build_segmented(config, decision)) {
      m_segment_task_rows[index_name] = segment_tasks;
    } else {
      m_segment_task_rows[index_name].clear();
    }
    if (rebuilt == nullptr) return false;
    if (!should_build_segmented(config, decision) &&
        !rebuild_backend_from_source(m_entry_store, m_standalone_store,
                                     index_name, config, rebuilt.get())) {
      return false;
    }
    rebuilt_backends.emplace(index_name, std::move(rebuilt));
  }

  for (auto &entry : rebuilt_backends) {
    auto lifecycle_it = m_lifecycle_infos.find(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_lifecycle_state(&lifecycle_it->second, LIFECYCLE_REBUILDING);
    }
    m_indexes[entry.first] = std::move(entry.second);
    maybe_unload_runtime(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_lifecycle_ready(&lifecycle_it->second);
    }
  }
  *rebuilt_count = rebuilt_backends.size();
  return true;
}

bool index_service::recover_all_indexes(size_t *recovered_count) {
  if (recovered_count == nullptr) return false;

  // Recovery cannot proceed with in-flight staged changes.
  if (has_any_pending_changes(m_pending_changes)) return false;

  std::unordered_map<std::string, std::unique_ptr<backend>> recovered_backends;
  std::unordered_set<std::string> recover_fallback_indexes;
  recovered_backends.reserve(m_index_configs.size());
  for (const auto &entry : m_index_configs) {
    const std::string &index_name = entry.first;
    const index_config &config = entry.second;
    if (!m_entry_store.has_index(index_name)) return false;

    std::unique_ptr<backend> recovered =
        build_backend_from_config(index_name, config);
    if (recovered == nullptr ||
        !recover_backend_from_source(m_entry_store, m_standalone_store,
                                     index_name, config, recovered.get())) {
      return false;
    }
    if (recovered->last_recover_used_fallback()) {
      recover_fallback_indexes.insert(index_name);
    }
    if (m_lifecycle_infos.find(index_name) == m_lifecycle_infos.end())
      return false;

    recovered_backends.emplace(index_name, std::move(recovered));
  }

  for (auto &entry : recovered_backends) {
    auto lifecycle_it = m_lifecycle_infos.find(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_lifecycle_state(&lifecycle_it->second, LIFECYCLE_RECOVERING);
    }
    m_indexes[entry.first] = std::move(entry.second);
    maybe_unload_runtime(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end() &&
        recover_fallback_indexes.find(entry.first) !=
            recover_fallback_indexes.end()) {
      mark_recover_fallback(&lifecycle_it->second, false);
    }
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_lifecycle_ready(&lifecycle_it->second);
    }
  }

  *recovered_count = recovered_backends.size();
  return true;
}

bool index_service::describe_index(
    const std::string &index_name, index_config *config,
    bool *supports_mutations, size_t *entry_count,
    size_t *committed_entry_count, std::string *lifecycle_state,
    uint64_t *lifecycle_version, uint32_t *last_error_code,
    uint64_t *last_error_ts, uint64_t *last_apply_latency_ms,
    uint64_t *recover_fallback_count, uint64_t *last_recover_fallback_ts,
    bool *external_manifest_present,
    uint64_t *external_manifest_generation,
    backend_build_diagnostics *build_diagnostics) const {
  if (config == nullptr) {
    return false;
  }

  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  const bool has_committed_entries = m_entry_store.has_index(index_name);
  if (config_it == m_index_configs.end() || index_it == m_indexes.end() ||
      !has_committed_entries) {
    return false;
  }
  if (lifecycle_it == m_lifecycle_infos.end()) return false;

  const backend *backend = index_it->second.get();
  config->dimension = backend->dimension();
  config->metric = backend->metric();
  config->mode = backend->mode();
  config->provider = backend->provider();
  config->consistency_mode = config_it->second.consistency_mode;
  config->backend_variant = backend->backend_variant();
  config->search_ef = backend->search_ef();
  config->hnsw_m = backend->hnsw_m();
  config->hnsw_ef_construction = backend->hnsw_ef_construction();
  config->hnsw_build_threads = backend->hnsw_build_threads();
  config->faiss_nlist = backend->faiss_nlist();
  config->faiss_nprobe = backend->faiss_nprobe();
  config->faiss_pq_m = backend->faiss_pq_m();
  config->faiss_pq_bits = backend->faiss_pq_bits();
  config->faiss_build_threads = backend->faiss_build_threads();
  config->diskann_max_degree = backend->diskann_max_degree();
  config->diskann_build_complexity = backend->diskann_build_complexity();
  config->diskann_build_threads = backend->diskann_build_threads();
  config->diskann_build_mode_value =
      detail::effective_diskann_build_mode(config_it->second);
  config->diskann_build_mode_specified =
      config_it->second.diskann_build_mode_specified;
  config->diskann_search_complexity = backend->diskann_search_complexity();
  config->diskann_search_beamwidth = backend->diskann_search_beamwidth();
  config->diskann_pq_code_budget_size =
      backend->diskann_pq_code_budget_size();
  if (supports_mutations != nullptr) {
    *supports_mutations = backend->supports_mutations();
  }
  if (entry_count != nullptr) {
    *entry_count = backend->entry_count();
  }
  if (committed_entry_count != nullptr) {
    *committed_entry_count = m_entry_store.entry_count(index_name);
  }
  if (lifecycle_state != nullptr) *lifecycle_state = lifecycle_it->second.state;
  if (lifecycle_version != nullptr)
    *lifecycle_version = lifecycle_it->second.version;
  if (last_error_code != nullptr)
    *last_error_code = lifecycle_it->second.last_error_code;
  if (last_error_ts != nullptr)
    *last_error_ts = lifecycle_it->second.last_error_ts;
  if (last_apply_latency_ms != nullptr) {
    *last_apply_latency_ms = lifecycle_it->second.last_apply_latency_ms;
  }
  if (recover_fallback_count != nullptr) {
    *recover_fallback_count = lifecycle_it->second.recover_fallback_count;
  }
  if (last_recover_fallback_ts != nullptr) {
    *last_recover_fallback_ts = lifecycle_it->second.last_recover_fallback_ts;
  }
  if (external_manifest_present != nullptr) {
    *external_manifest_present = backend->external_manifest_present();
  }
  if (external_manifest_generation != nullptr) {
    *external_manifest_generation = backend->external_manifest_generation();
  }
  if (build_diagnostics != nullptr) {
    *build_diagnostics = backend->build_diagnostics();
  }
  return true;
}

bool index_service::set_last_apply_latency_ms(const std::string &index_name,
                                              uint64_t latency_ms) {
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (lifecycle_it == m_lifecycle_infos.end()) return false;
  lifecycle_it->second.last_apply_latency_ms = latency_ms;
  return true;
}

bool index_service::set_lifecycle_state(
    const std::string &index_name, const std::string &lifecycle_state) {
  if (lifecycle_state.empty()) return false;

  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (lifecycle_it == m_lifecycle_infos.end()) return false;

  mark_lifecycle_state(&lifecycle_it->second, lifecycle_state.c_str());
  if (lifecycle_state != LIFECYCLE_FAILED) {
    lifecycle_it->second.last_error_code = ERROR_NONE;
    lifecycle_it->second.last_error_ts = 0;
  }
  return true;
}

bool index_service::set_lifecycle_info(const std::string &index_name,
                                       const std::string &lifecycle_state,
                                       uint64_t lifecycle_version,
                                       uint32_t last_error_code,
                                       uint64_t last_error_ts,
                                       uint64_t recover_fallback_count,
                                       uint64_t last_recover_fallback_ts) {
  if (lifecycle_state.empty() || lifecycle_version == 0) return false;

  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (lifecycle_it == m_lifecycle_infos.end()) return false;

  lifecycle_it->second.state = lifecycle_state;
  lifecycle_it->second.version = lifecycle_version;
  lifecycle_it->second.last_error_code = last_error_code;
  lifecycle_it->second.last_error_ts = last_error_ts;
  lifecycle_it->second.recover_fallback_count = recover_fallback_count;
  lifecycle_it->second.last_recover_fallback_ts = last_recover_fallback_ts;
  return true;
}

bool index_service::list_indexes(std::vector<std::string> *index_names) const {
  if (index_names == nullptr) return false;

  index_names->clear();
  index_names->reserve(m_indexes.size());
  for (const auto &entry : m_indexes) {
    index_names->push_back(entry.first);
  }
  std::sort(index_names->begin(), index_names->end());
  return true;
}

bool index_service::snapshot_committed_state(committed_state *state) const {
  return m_entry_store.snapshot(state);
}

size_t index_service::committed_entry_count() const {
  return m_entry_store.entry_count();
}

size_t index_service::committed_vector_memory_bytes() const {
  return m_entry_store.memory_bytes();
}

bool index_service::evict_committed_cache_to_budget() {
  return m_entry_store.evict_until_under_budget(
      static_cast<size_t>(opt_vector_entry_cache_size));
}

size_t index_service::pending_vector_memory_bytes(uint64_t txn_id) const {
  auto pending_it = m_pending_changes.find(txn_id);
  if (pending_it == m_pending_changes.end()) return 0;

  size_t bytes = 0;
  for (const pending_change &change : pending_it->second) {
    bytes += pending_change_memory_bytes(change);
  }
  return bytes;
}

size_t index_service::total_pending_vector_memory_bytes() const {
  size_t bytes = 0;
  for (const auto &txn_entry : m_pending_changes) {
    for (const pending_change &change : txn_entry.second) {
      bytes += pending_change_memory_bytes(change);
    }
  }
  return bytes;
}

size_t index_service::standalone_ingest_memory_bytes(
    const std::string &index_name) const {
  return m_standalone_store.memory_bytes(index_name);
}

size_t index_service::standalone_segment_count(
    const std::string &index_name) const {
  return m_standalone_store.segment_count(index_name);
}

size_t index_service::standalone_segment_bytes(
    const std::string &index_name) const {
  return m_standalone_store.segment_bytes(index_name);
}

size_t index_service::standalone_raw_segment_count(
    const std::string &index_name) const {
  return m_standalone_store.raw_segment_count(index_name);
}

size_t index_service::standalone_raw_segment_bytes(
    const std::string &index_name) const {
  return m_standalone_store.raw_segment_bytes(index_name);
}

std::string index_service::standalone_build_source(
    const std::string &index_name) const {
  return m_standalone_store.build_source(index_name);
}

bool index_service::describe_build_pipeline(
    const std::string &index_name, build_pipeline_snapshot *snapshot) const {
  if (snapshot == nullptr) return false;
  if (m_index_configs.find(index_name) == m_index_configs.end()) return false;

  auto snapshot_it = m_build_pipeline_snapshots.find(index_name);
  *snapshot = snapshot_it == m_build_pipeline_snapshots.end()
                  ? build_pipeline_snapshot{}
                  : snapshot_it->second;
  return true;
}

bool index_service::snapshot_segment_tasks(
    const std::string &index_name,
    std::vector<vector_index_metadata_store::segment_task_row> *rows) const {
  if (rows == nullptr) return false;
  if (m_index_configs.find(index_name) == m_index_configs.end()) return false;

  rows->clear();
  auto rows_it = m_segment_task_rows.find(index_name);
  if (rows_it != m_segment_task_rows.end()) *rows = rows_it->second;
  return true;
}

bool index_service::restore_committed_state(const committed_state &state) {
  std::unordered_map<std::string, backend_ptr> restored_backends;
  restored_backends.reserve(m_index_configs.size());
  vector_entry_store restored_entry_store;
  standalone_entry_store restored_standalone_store;

  for (const auto &config_entry : m_index_configs) {
    const std::string &index_name = config_entry.first;
    const index_config &config = config_entry.second;

    std::unique_ptr<backend> backend =
        build_backend_from_config(index_name, config);
    if (backend == nullptr) return false;

    auto state_it = state.find(index_name);
    if (!restored_standalone_store.register_index(index_name,
                                                  config.dimension)) {
      return false;
    }

    if (config.consistency_mode == index_consistency_mode::kStandalone) {
      if (!restored_entry_store.register_index(index_name)) return false;
      if (!restored_standalone_store.rebuild_backend_input(index_name,
                                                           backend.get())) {
        return false;
      }
    } else if (state_it != state.end()) {
      for (const auto &doc_entry : state_it->second) {
        if (doc_entry.second.size() != backend->dimension()) return false;
      }
      if (!backend->load_committed_entries(state_it->second)) return false;
      if (!restored_entry_store.register_index(index_name) ||
          !restored_entry_store.replace_index(index_name, state_it->second)) {
        return false;
      }
    } else if (!restored_entry_store.register_index(index_name)) {
      return false;
    }

    restored_backends.emplace(index_name, std::move(backend));
  }

  m_indexes = std::move(restored_backends);
  m_entry_store = std::move(restored_entry_store);
  m_standalone_store = std::move(restored_standalone_store);
  m_segment_task_rows.clear();
  for (const auto &entry : m_index_configs) {
    m_segment_task_rows[entry.first] = {};
    maybe_unload_runtime(entry.first);
  }
  return true;
}

bool index_service::restore_index_config(const std::string &index_name,
                                         const index_config &config) {
  DBUG_EXECUTE_IF("vector_service_fail_restore_index_config", return false;);
  if (config.dimension == 0) return false;

  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end())) {
    return false;
  }

  if (config.consistency_mode == index_consistency_mode::kStandalone) {
    if (!m_standalone_store.has_index(index_name)) return false;
  } else if (!entry_store_vectors_match_dimension(m_entry_store, index_name,
                                                  config.dimension)) {
    return false;
  }

  std::unique_ptr<backend> restored =
      build_backend_from_config(index_name, config);
  if (restored == nullptr) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_BACKEND_CREATE_FAILED);
    return false;
  }
  const bool source_loaded =
      config.consistency_mode == index_consistency_mode::kStandalone
          ? m_standalone_store.rebuild_backend_input(index_name,
                                                     restored.get())
          : load_backend_from_store(m_entry_store, index_name, restored.get());
  if (!source_loaded) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    return false;
  }

  index_it->second = std::move(restored);
  config_it->second = config;
  config_it->second.backend_variant = index_it->second->backend_variant();
  config_it->second.search_ef = index_it->second->search_ef();
  config_it->second.hnsw_m = index_it->second->hnsw_m();
  config_it->second.hnsw_ef_construction =
      index_it->second->hnsw_ef_construction();
  config_it->second.hnsw_build_threads =
      index_it->second->hnsw_build_threads();
  config_it->second.faiss_nlist = index_it->second->faiss_nlist();
  config_it->second.faiss_nprobe = index_it->second->faiss_nprobe();
  config_it->second.faiss_pq_m = index_it->second->faiss_pq_m();
  config_it->second.faiss_pq_bits = index_it->second->faiss_pq_bits();
  config_it->second.diskann_max_degree = index_it->second->diskann_max_degree();
  config_it->second.diskann_build_complexity =
      index_it->second->diskann_build_complexity();
  config_it->second.diskann_build_threads =
      index_it->second->diskann_build_threads();
  config_it->second.diskann_build_mode_value =
      index_it->second->diskann_build_mode_value();
  config_it->second.diskann_build_mode_specified =
      config.diskann_build_mode_specified;
  config_it->second.diskann_search_complexity =
      index_it->second->diskann_search_complexity();
  config_it->second.diskann_search_beamwidth =
      index_it->second->diskann_search_beamwidth();
  config_it->second.diskann_pq_code_budget_size =
      index_it->second->diskann_pq_code_budget_size();
  mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

bool index_service::set_index_consistency_mode(
    const std::string &index_name, index_consistency_mode consistency_mode) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                m_standalone_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_PENDING_CHANGES);
    return false;
  }

  index_config next_config = config_it->second;
  next_config.consistency_mode = consistency_mode;
  std::unique_ptr<backend> rebuilt =
      detail::build_backend_from_config(index_name, next_config);
  if (rebuilt == nullptr) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_BACKEND_CREATE_FAILED);
    return false;
  }
  if (!rebuild_backend_from_source(m_entry_store, m_standalone_store,
                                   index_name, next_config, rebuilt.get())) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    return false;
  }

  index_it->second = std::move(rebuilt);
  config_it->second = next_config;
  config_it->second.backend_variant = index_it->second->backend_variant();
  mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

bool index_service::direct_upsert(const std::string &index_name, uint64_t doc_id,
                                  const vector_data &vector) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                lifecycle_it != m_lifecycle_infos.end())) {
    return false;
  }
  if (config_it->second.consistency_mode !=
      index_consistency_mode::kStandalone) {
    return false;
  }
  if (config_it->second.dimension != vector.size()) return false;
  if (!m_standalone_store.upsert(
          index_name, doc_id, vector,
          static_cast<size_t>(opt_vector_entry_cache_size))) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    return false;
  }

  if (uses_deferred_standalone_mutations(config_it->second)) {
    mark_lifecycle_bulk_loading(&lifecycle_it->second);
    return true;
  }

  bool ok = false;
  {
    std::unique_lock<std::shared_mutex> runtime_guard(
        index_it->second->runtime_mutex());
    ok = index_it->second->upsert(doc_id, vector);
  }
  if (!ok) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    return false;
  }
  return true;
}

bool index_service::bulk_upsert_from_reader(
    const std::string &index_name, const bulk_load_reader &reader,
    const bulk_load_options &options, std::string *error) {
  if (error != nullptr) error->clear();
  auto config_it = m_index_configs.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                lifecycle_it != m_lifecycle_infos.end())) {
    set_bulk_load_error(error, "vector index not found");
    return false;
  }
  if (!reader) {
    set_bulk_load_error(error, "LOAD VECTOR DATA reader is empty");
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_PENDING_CHANGES);
    set_bulk_load_error(error, "vector index has pending changes");
    return false;
  }

  committed_entries staged_entries;
  std::unordered_set<uint64_t> seen_doc_ids;
  const size_t expected_dimension = config_it->second.dimension;
  const bulk_load_visitor visitor =
      [this, config_it, &index_name, &options, &staged_entries, &seen_doc_ids,
       expected_dimension, error](uint64_t doc_id, const float *values,
                                  size_t dimension) {
        if (values == nullptr) {
          set_bulk_load_error(error, "LOAD VECTOR DATA returned a null row");
          return false;
        }
        if (dimension != expected_dimension) {
          set_bulk_load_error(error, "LOAD VECTOR DATA dimension mismatch");
          return false;
        }

        if (!options.replace_duplicates) {
          if (!seen_doc_ids.insert(doc_id).second) {
            set_bulk_load_error(error,
                                "LOAD VECTOR DATA duplicate doc_id in file");
            return false;
          }
          vector_data existing;
          bool found = false;
          const bool lookup_ok =
              config_it->second.consistency_mode ==
                      index_consistency_mode::kStandalone
                  ? m_standalone_store.find_entry(index_name, doc_id,
                                                  &existing, &found)
                  : m_entry_store.find_committed_entry(index_name, doc_id,
                                                       &existing, &found);
          if (!lookup_ok) {
            set_bulk_load_error(error,
                                "LOAD VECTOR DATA could not read index state");
            return false;
          }
          if (found) {
            set_bulk_load_error(error,
                                "LOAD VECTOR DATA duplicate doc_id in index");
            return false;
          }
        } else {
          seen_doc_ids.insert(doc_id);
        }

        staged_entries[doc_id] =
            vector_data(values, values + expected_dimension);
        return true;
      };

  std::string reader_error;
  if (!reader(visitor, &reader_error)) {
    if (error != nullptr && error->empty()) {
      *error = reader_error.empty() ? "LOAD VECTOR DATA reader failed"
                                    : reader_error;
    }
    return false;
  }

  if (config_it->second.consistency_mode ==
      index_consistency_mode::kStandalone) {
    if (!m_standalone_store.bulk_upsert(index_name, staged_entries)) {
      mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
      set_bulk_load_error(error, "LOAD VECTOR DATA could not publish rows");
      return false;
    }

    if (options.rebuild_after_load) return rebuild_index(index_name);

    mark_lifecycle_bulk_loading(&lifecycle_it->second);
    return true;
  }

  for (const auto &entry : staged_entries) {
    if (!m_entry_store.upsert(index_name, entry.first, entry.second)) {
      mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
      set_bulk_load_error(error, "LOAD VECTOR DATA could not publish rows");
      return false;
    }
  }

  auto index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end()) {
    set_bulk_load_error(error, "vector index not found");
    return false;
  }
  std::unique_ptr<backend> rebuilt =
      build_backend_from_config(index_name, config_it->second);
  if (rebuilt == nullptr) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_BACKEND_CREATE_FAILED);
    set_bulk_load_error(error, "LOAD VECTOR DATA could not create backend");
    return false;
  }
  mark_lifecycle_state(&lifecycle_it->second, LIFECYCLE_REBUILDING);
  record_build_pipeline_decision(index_name, config_it->second);
  if (!rebuild_backend_from_store(m_entry_store, index_name, rebuilt.get())) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    set_bulk_load_error(error, "LOAD VECTOR DATA could not rebuild index");
    return false;
  }
  index_it->second = std::move(rebuilt);
  mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

bool index_service::bulk_upsert_from_raw_files(
    const std::string &index_name, const std::string &vector_filename,
    const std::string &docid_filename, const bulk_load_options &options,
    uint64_t *loaded_rows, std::string *error) {
  if (error != nullptr) error->clear();
  if (loaded_rows != nullptr) *loaded_rows = 0;
  auto config_it = m_index_configs.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                lifecycle_it != m_lifecycle_infos.end())) {
    set_bulk_load_error(error, "vector index not found");
    return false;
  }
  if (vector_filename.empty()) {
    set_bulk_load_error(error, "LOAD VECTOR DATA vector file is empty");
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_PENDING_CHANGES);
    set_bulk_load_error(error, "vector index has pending changes");
    return false;
  }

  if (config_it->second.consistency_mode !=
      index_consistency_mode::kStandalone) {
    uint64_t row_count = 0;
    const size_t expected_dimension = config_it->second.dimension;
    const bulk_load_reader reader =
        [vector_filename, docid_filename, expected_dimension,
         &row_count](const bulk_load_visitor &visitor,
                     std::string *reader_error) {
          return read_fbin_vectors(
              vector_filename, docid_filename, expected_dimension, nullptr,
              reader_error,
              [&row_count, &visitor](uint64_t doc_id, const float *values,
                                     size_t dimension) {
                if (!visitor(doc_id, values, dimension)) return false;
                ++row_count;
                return true;
              });
        };
    const bool ok = bulk_upsert_from_reader(index_name, reader, options, error);
    if (ok && loaded_rows != nullptr) *loaded_rows = row_count;
    return ok;
  }

  std::unordered_set<uint64_t> seen_doc_ids;
  const size_t expected_dimension = config_it->second.dimension;
  uint64_t row_count = 0;
  vector_load_file_info file_info;
  std::string reader_error;
  if (!read_fbin_vectors(
          vector_filename, docid_filename, expected_dimension, &file_info,
          &reader_error,
          [this, &index_name, &options, &seen_doc_ids, expected_dimension,
           &row_count, error](uint64_t doc_id, const float *values,
                              size_t dimension) {
            if (values == nullptr) {
              set_bulk_load_error(error, "LOAD VECTOR DATA returned a null row");
              return false;
            }
            if (dimension != expected_dimension) {
              set_bulk_load_error(error, "LOAD VECTOR DATA dimension mismatch");
              return false;
            }
            if (!options.replace_duplicates) {
              if (!seen_doc_ids.insert(doc_id).second) {
                set_bulk_load_error(error,
                                    "LOAD VECTOR DATA duplicate doc_id in file");
                return false;
              }
              vector_data existing;
              bool found = false;
              if (!m_standalone_store.find_entry(index_name, doc_id, &existing,
                                                 &found)) {
                set_bulk_load_error(
                    error, "LOAD VECTOR DATA could not read index state");
                return false;
              }
              if (found) {
                set_bulk_load_error(
                    error, "LOAD VECTOR DATA duplicate doc_id in index");
                return false;
              }
            } else {
              seen_doc_ids.insert(doc_id);
            }
            ++row_count;
            return true;
          })) {
    if (error != nullptr && error->empty()) {
      *error = reader_error.empty() ? "LOAD VECTOR DATA reader failed"
                                    : reader_error;
    }
    return false;
  }

  if (!m_standalone_store.bulk_upsert_raw_files(
          index_name, vector_filename, docid_filename, file_info.row_count,
          file_info.dimension, &seen_doc_ids)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    set_bulk_load_error(error, "LOAD VECTOR DATA could not publish raw files");
    return false;
  }
  if (loaded_rows != nullptr) *loaded_rows = row_count;

  if (options.rebuild_after_load) return rebuild_index(index_name);

  mark_lifecycle_bulk_loading(&lifecycle_it->second);
  return true;
}

bool index_service::direct_erase(const std::string &index_name,
                                 uint64_t doc_id) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                lifecycle_it != m_lifecycle_infos.end())) {
    return false;
  }
  if (config_it->second.consistency_mode !=
      index_consistency_mode::kStandalone) {
    return false;
  }
  if (!m_standalone_store.erase(
          index_name, doc_id,
          static_cast<size_t>(opt_vector_entry_cache_size))) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    return false;
  }

  if (uses_deferred_standalone_mutations(config_it->second)) {
    mark_lifecycle_bulk_loading(&lifecycle_it->second);
    return true;
  }

  bool ok = false;
  {
    std::unique_lock<std::shared_mutex> runtime_guard(
        index_it->second->runtime_mutex());
    ok = index_it->second->erase(doc_id);
  }
  if (!ok) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    return false;
  }
  return true;
}

bool index_service::replace_committed_entries(
    const std::string &index_name, const committed_entries &entries) {
  return replace_committed_entries_impl(index_name, entries, false);
}

bool index_service::replace_committed_entries_preserve_lifecycle(
    const std::string &index_name, const committed_entries &entries) {
  return replace_committed_entries_impl(index_name, entries, true);
}

bool index_service::replace_committed_entries_impl(
    const std::string &index_name, const committed_entries &entries,
    bool preserve_lifecycle) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end())) {
    return false;
  }

  for (const auto &doc_entry : entries) {
    if (doc_entry.second.size() != config_it->second.dimension) return false;
  }

  std::unique_ptr<backend> rebuilt =
      build_backend_from_config(index_name, config_it->second);
  if (rebuilt == nullptr) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_BACKEND_CREATE_FAILED);
    return false;
  }
  if (!rebuilt->rebuild_from_committed_entries(entries)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    return false;
  }

  index_it->second = std::move(rebuilt);
  if (!m_entry_store.replace_index(index_name, entries)) return false;
  if (!preserve_lifecycle) mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

bool index_service::install_rebuilt_index(
    const std::string &index_name, const committed_entries &entries,
    std::unique_ptr<backend> rebuilt_backend) {
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end(),
                rebuilt_backend != nullptr)) {
    return false;
  }

  index_it->second = std::move(rebuilt_backend);
  if (!m_entry_store.replace_index(index_name, entries)) return false;
  mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

bool index_service::install_recovered_index(
    const std::string &index_name, const committed_entries &entries,
    std::unique_ptr<backend> recovered_backend, bool used_recover_fallback) {
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end(),
                recovered_backend != nullptr)) {
    return false;
  }

  index_it->second = std::move(recovered_backend);
  if (!m_entry_store.replace_index(index_name, entries)) return false;
  if (used_recover_fallback) {
    mark_recover_fallback(&lifecycle_it->second, false);
  }
  mark_lifecycle_ready(&lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

}  // namespace vector_index
