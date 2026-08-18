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
#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <utility>

#include "my_dbug.h"
#include "sql/vector/vector_diskann_generation_store.h"
#include "sql/vector/vector_diskann_scheduler.h"
#include "sql/vector/vector_index_backend_common.h"
#include "sql/vector/vector_index_backend_internal.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_runtime_config.h"
#include "sql/vector/vector_index_runtime_thread_pool.h"
#include "sql/vector/vector_index_service_internal.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_segment_runtime_scheduler.h"
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
constexpr size_t k_diskann_segment_candidate_overfetch_factor = 2;

using build_clock = std::chrono::steady_clock;

uint64_t elapsed_build_ms(build_clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          build_clock::now() - start)
          .count());
}

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

uint64_t raw_segment_row_limit(
    size_t dimension, const build_pipeline_thresholds &thresholds) {
  return build_segment_row_limit(static_cast<uint64_t>(dimension), thresholds);
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

bool detect_dense_docid_file(const std::string &path, uint64_t row_count,
                             uint64_t *first_doc_id) {
  if (first_doc_id == nullptr) return false;
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) return false;

  uint64_t stored_rows = 0;
  if (!read_plain_value(file, stored_rows) || stored_rows != row_count) {
    return false;
  }
  if (row_count == 0) {
    *first_doc_id = 0;
    return true;
  }

  uint64_t first = 0;
  if (!read_plain_value(file, first)) return false;
  uint64_t expected = first;
  for (uint64_t row = 1; row < row_count; ++row) {
    if (expected == std::numeric_limits<uint64_t>::max()) return false;
    ++expected;
    uint64_t doc_id = 0;
    if (!read_plain_value(file, doc_id) || doc_id != expected) return false;
  }
  *first_doc_id = first;
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

using raw_segment_path_provider =
    std::function<bool(uint64_t, std::string *, std::string *)>;

bool write_entries_to_raw_segments(size_t dimension, uint64_t first_segment_id,
                                   uint64_t first_generation,
                                   const committed_entries &entries,
                                   raw_segment_path_provider path_provider,
                                   std::vector<raw_vector_segment> *segments);

}  // namespace

namespace detail {

namespace {

void merge_diagnostic_string(const std::string &source, std::string *target) {
  if (target->empty()) {
    *target = source;
  } else if (!source.empty() && *target != source) {
    *target = "mixed";
  }
}

}  // namespace

void merge_segment_build_diagnostics(
    uint64_t segment_row_count, const backend_build_diagnostics &source,
    backend_build_diagnostics *aggregate) {
  aggregate->row_count +=
      source.row_count == 0 ? segment_row_count : source.row_count;
  aggregate->segment_count +=
      source.segment_count == 0 ? 1 : source.segment_count;
  aggregate->build_invocations +=
      source.build_invocations == 0 ? 1 : source.build_invocations;
  aggregate->pq_chunks += source.pq_chunks;
  aggregate->cache_nodes += source.cache_nodes;
  aggregate->requested_disk_pq_dims = std::max(
      aggregate->requested_disk_pq_dims, source.requested_disk_pq_dims);
  aggregate->effective_disk_pq_dims = std::max(
      aggregate->effective_disk_pq_dims, source.effective_disk_pq_dims);
  aggregate->manifest_ms += source.manifest_ms;
  aggregate->offline_build_ms += source.offline_build_ms;
  aggregate->load_ms += source.load_ms;
  aggregate->reader_count_ms += source.reader_count_ms;
  aggregate->training_copy_ms += source.training_copy_ms;
  aggregate->train_ms += source.train_ms;
  aggregate->add_ms += source.add_ms;
  aggregate->persist_ms += source.persist_ms;
  aggregate->training_rows += source.training_rows;
  aggregate->reader_passes += source.reader_passes;
  merge_diagnostic_string(source.diskann_pq_runtime,
                          &aggregate->diskann_pq_runtime);
  merge_diagnostic_string(source.native_pq_runtime_selected_path,
                          &aggregate->native_pq_runtime_selected_path);
  aggregate->native_pq_runtime_elapsed_ms +=
      source.native_pq_runtime_elapsed_ms;
  aggregate->native_pq_runtime_raw_reader_ms +=
      source.native_pq_runtime_raw_reader_ms;
  aggregate->native_pq_runtime_train_ms += source.native_pq_runtime_train_ms;
  aggregate->native_pq_runtime_encode_ms += source.native_pq_runtime_encode_ms;
  aggregate->native_pq_runtime_artifact_validation_ms +=
      source.native_pq_runtime_artifact_validation_ms;
  merge_diagnostic_string(source.native_pq_runtime_centroid_scan_kernel,
                          &aggregate->native_pq_runtime_centroid_scan_kernel);
  aggregate->native_pq_runtime_distance_calls +=
      source.native_pq_runtime_distance_calls;
  aggregate->native_pq_runtime_train_rows += source.native_pq_runtime_train_rows;
  aggregate->native_pq_runtime_compressed_rows +=
      source.native_pq_runtime_compressed_rows;
  aggregate->native_pq_runtime_encode_block_rows =
      std::max(aggregate->native_pq_runtime_encode_block_rows,
               source.native_pq_runtime_encode_block_rows);
  aggregate->native_pq_runtime_memory_estimate +=
      source.native_pq_runtime_memory_estimate;
  aggregate->native_pq_runtime_memory_budget =
      std::max(aggregate->native_pq_runtime_memory_budget,
               source.native_pq_runtime_memory_budget);
  aggregate->native_pq_runtime_effective_threads =
      std::max(aggregate->native_pq_runtime_effective_threads,
               source.native_pq_runtime_effective_threads);
  merge_diagnostic_string(source.native_pq_runtime_memory_adjustment,
                          &aggregate->native_pq_runtime_memory_adjustment);
  aggregate->native_pq_runtime_artifacts_written =
      aggregate->native_pq_runtime_artifacts_written ||
      source.native_pq_runtime_artifacts_written;
  aggregate->native_pq_runtime_artifacts_consumed =
      aggregate->native_pq_runtime_artifacts_consumed ||
      source.native_pq_runtime_artifacts_consumed;
  aggregate->native_pq_runtime_official_pq_used =
      aggregate->native_pq_runtime_official_pq_used ||
      source.native_pq_runtime_official_pq_used;
  merge_diagnostic_string(source.native_pq_runtime_bridge,
                          &aggregate->native_pq_runtime_bridge);
  aggregate->native_pq_runtime_bridge_ms += source.native_pq_runtime_bridge_ms;
  aggregate->native_pq_runtime_graph_ms += source.native_pq_runtime_graph_ms;
  aggregate->native_pq_runtime_cache_ms += source.native_pq_runtime_cache_ms;
  merge_diagnostic_string(source.native_pq_runtime_artifact_validation,
                          &aggregate->native_pq_runtime_artifact_validation);
  aggregate->native_pq_runtime_validation_failed =
      aggregate->native_pq_runtime_validation_failed ||
      source.native_pq_runtime_validation_failed;
  merge_diagnostic_string(
      source.native_pq_runtime_validation_failed_doc_id,
      &aggregate->native_pq_runtime_validation_failed_doc_id);
  merge_diagnostic_string(source.native_pq_runtime_validation_best_doc_id,
                          &aggregate->native_pq_runtime_validation_best_doc_id);
  aggregate->native_pq_runtime_validation_result_count =
      std::max(aggregate->native_pq_runtime_validation_result_count,
               source.native_pq_runtime_validation_result_count);
  merge_diagnostic_string(
      source.native_pq_runtime_validation_best_search_distance,
      &aggregate->native_pq_runtime_validation_best_search_distance);
  merge_diagnostic_string(
      source.native_pq_runtime_validation_best_exact_distance,
      &aggregate->native_pq_runtime_validation_best_exact_distance);
  merge_diagnostic_string(
      source.native_pq_runtime_validation_self_pq_distance,
      &aggregate->native_pq_runtime_validation_self_pq_distance);
  merge_diagnostic_string(
      source.native_pq_runtime_validation_pivots_checksum,
      &aggregate->native_pq_runtime_validation_pivots_checksum);
  merge_diagnostic_string(
      source.native_pq_runtime_validation_compressed_checksum,
      &aggregate->native_pq_runtime_validation_compressed_checksum);
  if (!source.fallback_reason.empty()) {
    aggregate->fallback_reason = source.fallback_reason;
  }
}

size_t compute_diskann_exact_rerank_candidate_top_k(
    size_t top_k, size_t query_count, size_t authoritative_count,
    size_t segment_count, uint32_t search_complexity,
    diskann_search_profile search_profile, size_t result_budget,
    size_t candidate_target) {
  if (top_k == 0 || query_count == 0) return top_k;
  if (authoritative_count <= top_k) return top_k;

  const size_t safe_segment_count = std::max<size_t>(segment_count, 1);
  const uint32_t fanout_segments =
      safe_segment_count > std::numeric_limits<uint32_t>::max()
          ? std::numeric_limits<uint32_t>::max()
          : static_cast<uint32_t>(safe_segment_count);
  const uint64_t segment_max_entries =
      saturated_add_size(authoritative_count, safe_segment_count - 1) /
      safe_segment_count;

  diskann_search_budget budget;
  size_t candidate_top_k = top_k;
  if (choose_diskann_search_budget(
          static_cast<uint32_t>(top_k), fanout_segments, 1,
          std::max<uint64_t>(segment_max_entries, 1), search_complexity,
          search_profile, &budget)) {
    candidate_top_k = budget.candidate_count;
  } else {
    candidate_top_k = diskann_search_list_slack_top_k(
        top_k,
        search_complexity == 0 ? k_default_diskann_search_complexity
                               : search_complexity,
        safe_segment_count);
  }

  candidate_top_k = std::min(candidate_top_k, authoritative_count);
  if (candidate_target != 0) {
    candidate_top_k =
        std::max(candidate_top_k,
                 std::min(candidate_target, authoritative_count));
  }

  if (result_budget != 0) {
    const size_t per_query_limit = result_budget / query_count;
    if (per_query_limit < top_k) return top_k;
    candidate_top_k = std::min(candidate_top_k, per_query_limit);
  }
  candidate_top_k =
      std::min(candidate_top_k,
               static_cast<size_t>(k_max_search_batch_result_count));
  return std::max(candidate_top_k, top_k);
}

}  // namespace detail

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
size_t diskann_exact_rerank_candidate_top_k_for_testing(
    size_t top_k, size_t query_count, size_t authoritative_count,
    size_t segment_count, uint32_t search_complexity,
    diskann_search_profile search_profile, size_t result_budget,
    size_t candidate_target) {
  return detail::compute_diskann_exact_rerank_candidate_top_k(
      top_k, query_count, authoritative_count, segment_count, search_complexity,
      search_profile, result_budget, candidate_target);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

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

bool vector_entry_store::prepare_index_for_mutation(
    const std::string &index_name) {
  return ensure_index_cached(index_name);
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
    uint64_t row_limit,
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

  std::vector<std::string> created_files;
  std::vector<standalone_segment> new_segments;
  const auto cleanup_created_files = [&created_files]() {
    for (const std::string &path : created_files) remove_file_if_exists(path);
  };
  const auto mark_dense_docids = [](standalone_segment *segment) {
    if (segment == nullptr) return false;
    uint64_t first_doc_id = 0;
    if (!detect_dense_docid_file(segment->docid_path, segment->record_count,
                                 &first_doc_id)) {
      segment->dense_doc_ids = false;
      segment->first_doc_id = 0;
      return false;
    }
    segment->dense_doc_ids = true;
    segment->first_doc_id = first_doc_id;
    return true;
  };
  const auto append_segment =
      [this, &index_name, &state_it, &created_files, &new_segments](
          uint64_t segment_id, uint64_t generation, size_t rows) {
        const std::string target_vector_path =
            raw_vector_path(index_name, segment_id);
        const std::string target_docid_path =
            raw_docid_path(index_name, segment_id);
        if (target_vector_path.empty() || target_docid_path.empty())
          return false;
        if (!detail::ensure_parent_directory(target_vector_path) ||
            !detail::ensure_parent_directory(target_docid_path)) {
          return false;
        }

        created_files.push_back(target_vector_path);
        created_files.push_back(target_docid_path);
        standalone_segment segment;
        segment.kind = standalone_segment_kind::kRawFbin;
        segment.vector_path = target_vector_path;
        segment.docid_path = target_docid_path;
        segment.record_count = rows;
        segment.dimension = state_it->second.dimension;
        segment.generation = generation;
        new_segments.push_back(std::move(segment));
        return true;
      };

  if (row_limit == 0) return false;
  if (row_count <= row_limit) {
    const uint64_t segment_id = state_it->second.next_segment_id;
    if (!append_segment(segment_id, state_it->second.generation + 1,
                        static_cast<size_t>(row_count))) {
      return false;
    }

    standalone_segment &segment = new_segments.back();
    if (!copy_or_link_file(vector_filename, segment.vector_path)) {
      cleanup_created_files();
      return false;
    }
    bool docid_ready = false;
    if (docid_filename.empty()) {
      docid_ready = write_generated_docid_file(segment.docid_path, row_count);
      segment.dense_doc_ids = docid_ready;
      segment.first_doc_id = 0;
    } else {
      docid_ready = copy_or_link_file(docid_filename, segment.docid_path);
      if (docid_ready) mark_dense_docids(&segment);
    }
    if (!docid_ready) {
      cleanup_created_files();
      return false;
    }

    size_t vector_bytes = 0;
    size_t docid_bytes = 0;
    if (!file_size_as_size(segment.vector_path, &vector_bytes) ||
        !file_size_as_size(segment.docid_path, &docid_bytes) ||
        vector_bytes > std::numeric_limits<size_t>::max() - docid_bytes) {
      cleanup_created_files();
      return false;
    }
    segment.bytes = vector_bytes + docid_bytes;
  } else {
    std::ofstream vector_file;
    std::ofstream docid_file;
    uint64_t rows_seen = 0;
    uint64_t rows_in_segment = 0;
    uint64_t current_segment_rows = 0;

    const auto close_segment = [&]() {
      vector_file.close();
      docid_file.close();
      if (!vector_file || !docid_file) return false;

      standalone_segment &segment = new_segments.back();
      size_t vector_bytes = 0;
      size_t docid_bytes = 0;
      if (!file_size_as_size(segment.vector_path, &vector_bytes) ||
          !file_size_as_size(segment.docid_path, &docid_bytes) ||
          vector_bytes > std::numeric_limits<size_t>::max() - docid_bytes) {
        return false;
      }
      segment.bytes = vector_bytes + docid_bytes;
      mark_dense_docids(&segment);
      rows_in_segment = 0;
      current_segment_rows = 0;
      return true;
    };

    const auto open_segment = [&]() {
      const uint64_t remaining = row_count - rows_seen;
      current_segment_rows = std::min(row_limit, remaining);
      const uint64_t segment_id =
          state_it->second.next_segment_id + new_segments.size();
      const uint64_t generation =
          state_it->second.generation + new_segments.size() + 1;
      if (!append_segment(segment_id, generation,
                          static_cast<size_t>(current_segment_rows))) {
        return false;
      }
      vector_file.open(new_segments.back().vector_path,
                       std::ios::out | std::ios::binary | std::ios::trunc);
      docid_file.open(new_segments.back().docid_path,
                      std::ios::out | std::ios::binary | std::ios::trunc);
      if (!vector_file.is_open() || !docid_file.is_open()) return false;
      return write_plain_value(vector_file,
                               static_cast<uint32_t>(current_segment_rows)) &&
             write_plain_value(
                 vector_file,
                 static_cast<uint32_t>(state_it->second.dimension)) &&
             write_plain_value(docid_file, current_segment_rows);
    };

    const size_t vector_bytes_per_row =
        state_it->second.dimension * sizeof(float);
    if (vector_bytes_per_row >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      cleanup_created_files();
      return false;
    }

    const bool split_ok = read_fbin_vectors(
        vector_filename, docid_filename, state_it->second.dimension, nullptr,
        &load_error,
        [&](uint64_t doc_id, const float *values, size_t vector_dimension) {
          if (values == nullptr ||
              vector_dimension != state_it->second.dimension)
            return false;
          if (rows_in_segment == 0 && !open_segment()) return false;
          if (!write_plain_value(docid_file, doc_id)) return false;
          vector_file.write(reinterpret_cast<const char *>(values),
                            static_cast<std::streamsize>(
                                vector_bytes_per_row));
          if (!vector_file.good()) return false;
          ++rows_seen;
          ++rows_in_segment;
          if (rows_in_segment == current_segment_rows) {
            return close_segment();
          }
          return true;
        });
    if (!split_ok || rows_seen != row_count || rows_in_segment != 0) {
      cleanup_created_files();
      return false;
    }
  }

  std::shared_ptr<const raw_entry_locator> new_locator_run;
  if (!build_raw_locator_run(new_segments, state_it->second.segments.size(),
                             &new_locator_run)) {
    cleanup_created_files();
    return false;
  }

  struct raw_bulk_publish_journal {
    size_t segment_count;
    size_t locator_run_count;
    size_t entry_count;
    uint64_t generation;
    uint64_t next_segment_id;
    std::string build_source;
    std::vector<uint64_t> inserted_live_doc_ids;
  } journal{state_it->second.segments.size(),
            state_it->second.raw_locator_runs.size(),
            state_it->second.entry_count,
            state_it->second.generation,
            state_it->second.next_segment_id,
            state_it->second.build_source,
            {}};
  const auto rollback_publish = [&state_it, &journal]() {
    for (const uint64_t doc_id : journal.inserted_live_doc_ids) {
      state_it->second.live_doc_ids.erase(doc_id);
    }
    state_it->second.segments.resize(journal.segment_count);
    state_it->second.raw_locator_runs.resize(journal.locator_run_count);
    state_it->second.entry_count = journal.entry_count;
    state_it->second.generation = journal.generation;
    state_it->second.next_segment_id = journal.next_segment_id;
    state_it->second.build_source = journal.build_source;
  };

  state_it->second.segments.insert(state_it->second.segments.end(),
                                   new_segments.begin(), new_segments.end());
  state_it->second.next_segment_id += new_segments.size();
  if (new_locator_run != nullptr && !new_locator_run->empty()) {
    state_it->second.raw_locator_runs.push_back(std::move(new_locator_run));
  }

  if (loaded_doc_ids != nullptr) {
    journal.inserted_live_doc_ids.reserve(loaded_doc_ids->size());
    while (!loaded_doc_ids->empty()) {
      auto node = loaded_doc_ids->extract(loaded_doc_ids->begin());
      const uint64_t doc_id = node.value();
      auto result = state_it->second.live_doc_ids.insert(std::move(node));
      if (result.inserted) journal.inserted_live_doc_ids.push_back(doc_id);
    }
  } else {
    for (const standalone_segment &segment : new_segments) {
      if (!read_docid_values(
              segment.docid_path, segment.record_count, &load_error,
              [&state_it, &journal](uint64_t doc_id) {
                if (state_it->second.live_doc_ids.insert(doc_id).second) {
                  journal.inserted_live_doc_ids.push_back(doc_id);
                }
                return true;
              })) {
        rollback_publish();
        cleanup_created_files();
        return false;
      }
    }
  }
  state_it->second.entry_count = state_it->second.live_doc_ids.size();
  state_it->second.generation = state_it->second.segments.back().generation;
  state_it->second.build_source = build_source_for_state(state_it->second);

  bool fail_manifest_save = false;
  DBUG_EXECUTE_IF("vector_standalone_bulk_fail_manifest_save",
                  fail_manifest_save = true;);
  if (fail_manifest_save || !save_manifest(index_name, state_it->second)) {
    rollback_publish();
    cleanup_created_files();
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

bool standalone_entry_store::write_compacted_raw_segment(
    const std::string &index_name,
    const std::vector<raw_vector_segment> &source_segments, uint64_t segment_id,
    uint64_t generation, raw_vector_segment *compacted_segment) const {
  if (source_segments.empty() || segment_id == 0 || generation == 0 ||
      compacted_segment == nullptr) {
    return false;
  }

  const size_t dimension = source_segments.front().dimension;
  if (dimension == 0) return false;

  uint64_t row_count = 0;
  for (const raw_vector_segment &segment : source_segments) {
    if (segment.dimension != dimension ||
        segment.row_count > std::numeric_limits<uint64_t>::max() - row_count) {
      return false;
    }
    row_count += segment.row_count;
  }
  if (row_count == 0 || row_count > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const std::string vector_path = raw_vector_path(index_name, segment_id);
  const std::string docid_path = raw_docid_path(index_name, segment_id);
  if (vector_path.empty() || docid_path.empty() ||
      !detail::ensure_parent_directory(vector_path) ||
      !detail::ensure_parent_directory(docid_path)) {
    return false;
  }

  std::ofstream vector_file(vector_path,
                            std::ios::out | std::ios::binary | std::ios::trunc);
  std::ofstream docid_file(docid_path,
                           std::ios::out | std::ios::binary | std::ios::trunc);
  const auto cleanup_output = [&]() {
    if (vector_file.is_open()) vector_file.close();
    if (docid_file.is_open()) docid_file.close();
    remove_file_if_exists(vector_path);
    remove_file_if_exists(docid_path);
  };
  if (!vector_file.is_open() || !docid_file.is_open() ||
      !write_plain_value(vector_file, static_cast<uint32_t>(row_count)) ||
      !write_plain_value(vector_file, static_cast<uint32_t>(dimension)) ||
      !write_plain_value(docid_file, row_count)) {
    cleanup_output();
    return false;
  }

  const size_t vector_bytes = dimension * sizeof(float);
  if (vector_bytes >
      static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    cleanup_output();
    return false;
  }

  uint64_t rows_written = 0;
  std::string read_error;
  for (const raw_vector_segment &segment : source_segments) {
    vector_load_file_info file_info;
    const bool read_ok = read_fbin_vectors(
        segment.vector_path, segment.docid_path, dimension, &file_info,
        &read_error,
        [&](uint64_t doc_id, const float *values, size_t vector_dimension) {
          if (values == nullptr || vector_dimension != dimension ||
              !write_plain_value(docid_file, doc_id)) {
            return false;
          }
          vector_file.write(reinterpret_cast<const char *>(values),
                            static_cast<std::streamsize>(vector_bytes));
          if (!vector_file.good()) return false;
          ++rows_written;
          return true;
        });
    if (!read_ok || file_info.row_count != segment.row_count ||
        file_info.dimension != dimension) {
      cleanup_output();
      return false;
    }
  }

  vector_file.close();
  docid_file.close();
  if (!vector_file || !docid_file || rows_written != row_count) {
    cleanup_output();
    return false;
  }

  size_t persisted_vector_bytes = 0;
  size_t persisted_docid_bytes = 0;
  if (!file_size_as_size(vector_path, &persisted_vector_bytes) ||
      !file_size_as_size(docid_path, &persisted_docid_bytes) ||
      persisted_vector_bytes >
          std::numeric_limits<size_t>::max() - persisted_docid_bytes) {
    cleanup_output();
    return false;
  }

  compacted_segment->vector_path = vector_path;
  compacted_segment->docid_path = docid_path;
  compacted_segment->row_count = static_cast<size_t>(row_count);
  compacted_segment->dimension = dimension;
  compacted_segment->bytes = persisted_vector_bytes + persisted_docid_bytes;
  compacted_segment->generation = generation;
  return true;
}

void standalone_entry_store::discard_levelled_compaction(
    raw_segment_compaction *compaction) const {
  if (compaction == nullptr) return;
  if (compaction->published) return;
  for (const std::string &path : compaction->created_files) {
    remove_file_if_exists(path);
  }
  *compaction = raw_segment_compaction{};
}

bool standalone_entry_store::stage_levelled_compaction(
    const std::string &index_name, uint64_t row_limit,
    raw_segment_compaction *compaction) {
  if (compaction == nullptr || row_limit == 0) return false;
  discard_levelled_compaction(compaction);

  const auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end() ||
      !can_rebuild_direct_from_raw_segments(state_it->second)) {
    return false;
  }

  compaction->source_generation = state_it->second.generation;
  compaction->source_next_segment_id = state_it->second.next_segment_id;
  compaction->source_build_source = state_it->second.build_source;
  compaction->next_generation = state_it->second.generation;
  compaction->next_segment_id = state_it->second.next_segment_id;
  if (!read_raw_segments(state_it->second,
                         [compaction](const raw_vector_segment &segment) {
                           compaction->source_segments.push_back(segment);
                           return true;
                         })) {
    discard_levelled_compaction(compaction);
    return false;
  }

  if (compaction->source_segments.size() < 2) {
    compaction->segments = compaction->source_segments;
    return true;
  }

  size_t first = 0;
  while (first < compaction->source_segments.size()) {
    const raw_vector_segment &first_segment =
        compaction->source_segments[first];
    if (first_segment.row_count >= row_limit) {
      compaction->segments.push_back(first_segment);
      ++first;
      continue;
    }

    uint64_t grouped_rows = 0;
    size_t last = first;
    while (last < compaction->source_segments.size()) {
      const raw_vector_segment &candidate = compaction->source_segments[last];
      if (candidate.row_count >= row_limit ||
          candidate.row_count > row_limit - grouped_rows) {
        break;
      }
      grouped_rows += candidate.row_count;
      ++last;
    }

    if (last - first < 2) {
      compaction->segments.push_back(first_segment);
      ++first;
      continue;
    }
    if (compaction->next_segment_id == std::numeric_limits<uint64_t>::max() ||
        compaction->next_generation == std::numeric_limits<uint64_t>::max()) {
      discard_levelled_compaction(compaction);
      return false;
    }

    std::vector<raw_vector_segment> group(
        compaction->source_segments.begin() + first,
        compaction->source_segments.begin() + last);
    raw_vector_segment compacted_segment;
    if (!write_compacted_raw_segment(
            index_name, group, compaction->next_segment_id,
            compaction->next_generation + 1, &compacted_segment)) {
      discard_levelled_compaction(compaction);
      return false;
    }
    ++compaction->next_segment_id;
    ++compaction->next_generation;
    compaction->created_files.push_back(compacted_segment.vector_path);
    compaction->created_files.push_back(compacted_segment.docid_path);
    compaction->segments.push_back(std::move(compacted_segment));
    for (const raw_vector_segment &segment : group) {
      compaction->replaced_files.push_back(segment.vector_path);
      compaction->replaced_files.push_back(segment.docid_path);
    }
    first = last;
  }

  compaction->needed =
      compaction->segments.size() < compaction->source_segments.size();
  return true;
}

bool standalone_entry_store::publish_levelled_compaction(
    const std::string &index_name, raw_segment_compaction *compaction) {
  if (compaction == nullptr) return false;
  if (!compaction->needed) return true;

  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end() ||
      state_it->second.generation != compaction->source_generation ||
      state_it->second.next_segment_id != compaction->source_next_segment_id ||
      state_it->second.segments.size() != compaction->source_segments.size()) {
    return false;
  }

  for (size_t i = 0; i < compaction->source_segments.size(); ++i) {
    const standalone_segment &current = state_it->second.segments[i];
    const raw_vector_segment &source = compaction->source_segments[i];
    if (current.kind != standalone_segment_kind::kRawFbin ||
        current.vector_path != source.vector_path ||
        current.docid_path != source.docid_path ||
        current.record_count != source.row_count ||
        current.dimension != source.dimension ||
        current.bytes != source.bytes ||
        current.generation != source.generation) {
      return false;
    }
  }

  DBUG_EXECUTE_IF("vector_standalone_compaction_before_publish", return false;);

  index_state next_state = state_it->second;
  if (!assign_raw_segments(&next_state, compaction->segments)) return false;
  next_state.generation = compaction->next_generation;
  next_state.next_segment_id = compaction->next_segment_id;
  next_state.build_source = "raw_segments_compacted";
  if (!save_manifest(index_name, next_state)) return false;

  state_it->second = std::move(next_state);
  compaction->published = true;
  return true;
}

bool standalone_entry_store::rollback_levelled_compaction(
    const std::string &index_name, raw_segment_compaction *compaction) {
  if (compaction == nullptr) return false;
  if (!compaction->needed || !compaction->published) {
    discard_levelled_compaction(compaction);
    return true;
  }

  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return false;

  index_state restored_state = state_it->second;
  if (!assign_raw_segments(&restored_state, compaction->source_segments)) {
    return false;
  }
  restored_state.generation = compaction->source_generation;
  restored_state.next_segment_id = compaction->source_next_segment_id;
  restored_state.build_source = compaction->source_build_source;
  if (!save_manifest(index_name, restored_state)) return false;

  state_it->second = std::move(restored_state);
  compaction->published = false;
  discard_levelled_compaction(compaction);
  return true;
}

void standalone_entry_store::finalize_levelled_compaction(
    raw_segment_compaction *compaction) const {
  if (compaction == nullptr) return;
  if (!compaction->published) {
    discard_levelled_compaction(compaction);
    return;
  }
  for (const std::string &path : compaction->replaced_files) {
    remove_file_if_exists(path);
  }
  compaction->created_files.clear();
  *compaction = raw_segment_compaction{};
}

bool standalone_entry_store::assign_raw_segments(
    index_state *state,
    const std::vector<raw_vector_segment> &raw_segments) const {
  if (state == nullptr) return false;

  std::vector<standalone_segment> segments;
  segments.reserve(raw_segments.size());
  for (const raw_vector_segment &raw_segment : raw_segments) {
    standalone_segment segment;
    segment.kind = standalone_segment_kind::kRawFbin;
    segment.vector_path = raw_segment.vector_path;
    segment.docid_path = raw_segment.docid_path;
    segment.record_count = raw_segment.row_count;
    segment.dimension = raw_segment.dimension;
    segment.bytes = raw_segment.bytes;
    segment.generation = raw_segment.generation;
    uint64_t first_doc_id = 0;
    if (detect_dense_docid_file(segment.docid_path, segment.record_count,
                                &first_doc_id)) {
      segment.dense_doc_ids = true;
      segment.first_doc_id = first_doc_id;
    }
    segments.push_back(std::move(segment));
  }
  std::shared_ptr<const raw_entry_locator> locator_run;
  if (!build_raw_locator_run(segments, 0, &locator_run)) return false;
  state->segments = std::move(segments);
  state->raw_locator_runs.clear();
  if (locator_run != nullptr && !locator_run->empty()) {
    state->raw_locator_runs.push_back(std::move(locator_run));
  }
  return true;
}

bool standalone_entry_store::build_raw_locator_run(
    const std::vector<standalone_segment> &segments, size_t first_segment_index,
    std::shared_ptr<const raw_entry_locator> *locator) const {
  if (locator == nullptr ||
      first_segment_index > std::numeric_limits<uint32_t>::max() ||
      segments.size() >
          std::numeric_limits<uint32_t>::max() - first_segment_index) {
    return false;
  }

  size_t entry_count = 0;
  for (const standalone_segment &segment : segments) {
    if (segment.kind != standalone_segment_kind::kRawFbin ||
        segment.dense_doc_ids) {
      continue;
    }
    if (segment.record_count > std::numeric_limits<uint32_t>::max() ||
        segment.record_count >
            std::numeric_limits<size_t>::max() - entry_count) {
      return false;
    }
    entry_count += segment.record_count;
  }

  // Each bulk publish adds one immutable run so rollback snapshots only copy
  // shared handles instead of all previously ingested locator entries.
  auto next_locator = std::make_shared<raw_entry_locator>();
  next_locator->reserve(entry_count);
  std::string load_error;
  for (size_t segment_index = 0; segment_index < segments.size();
       ++segment_index) {
    const standalone_segment &segment = segments[segment_index];
    if (segment.kind != standalone_segment_kind::kRawFbin ||
        segment.dense_doc_ids) {
      continue;
    }

    uint32_t row = 0;
    if (!read_docid_values(
            segment.docid_path, segment.record_count, &load_error,
            [&next_locator, first_segment_index, segment_index,
             &row](uint64_t doc_id) {
              next_locator->push_back(raw_entry_location{
                  doc_id,
                  static_cast<uint32_t>(first_segment_index + segment_index),
                  row});
              ++row;
              return true;
            }) ||
        row != segment.record_count) {
      return false;
    }
  }

  std::sort(
      next_locator->begin(), next_locator->end(),
      [](const raw_entry_location &left, const raw_entry_location &right) {
        if (left.doc_id != right.doc_id) return left.doc_id < right.doc_id;
        if (left.segment_index != right.segment_index)
          return left.segment_index < right.segment_index;
        return left.row < right.row;
      });
  *locator = std::move(next_locator);
  return true;
}

bool standalone_entry_store::consolidate_raw_locator_runs(
    index_state *state) const {
  if (state == nullptr) return false;
  if (state->raw_locator_runs.size() <= 1) return true;

  size_t entry_count = 0;
  for (const std::shared_ptr<const raw_entry_locator> &run :
       state->raw_locator_runs) {
    if (run == nullptr ||
        run->size() > std::numeric_limits<size_t>::max() - entry_count) {
      return false;
    }
    entry_count += run->size();
  }

  auto consolidated = std::make_shared<raw_entry_locator>();
  consolidated->reserve(entry_count);
  for (const std::shared_ptr<const raw_entry_locator> &run :
       state->raw_locator_runs) {
    consolidated->insert(consolidated->end(), run->begin(), run->end());
  }
  std::sort(
      consolidated->begin(), consolidated->end(),
      [](const raw_entry_location &left, const raw_entry_location &right) {
        if (left.doc_id != right.doc_id) return left.doc_id < right.doc_id;
        if (left.segment_index != right.segment_index)
          return left.segment_index < right.segment_index;
        return left.row < right.row;
      });

  state->raw_locator_runs.clear();
  if (!consolidated->empty()) {
    state->raw_locator_runs.push_back(std::move(consolidated));
  }
  return true;
}

bool standalone_entry_store::compact_to_raw_segment(const std::string &index_name,
                                                    index_state *state) {
  if (state == nullptr) return false;

  committed_entries entries;
  if (!load_entries(*state, &entries)) return false;

  const index_state before_compact = *state;
  if (entries.empty()) {
    state->segments.clear();
    state->raw_locator_runs.clear();
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

  std::vector<std::string> created_files;
  std::vector<raw_vector_segment> raw_segments;
  if (!write_entries_to_raw_segments(
          state->dimension, state->next_segment_id, state->generation + 1,
          entries,
          [this, &index_name, &created_files](uint64_t segment_id,
                                              std::string *vector_path,
                                              std::string *docid_path) {
            if (vector_path == nullptr || docid_path == nullptr) return false;
            *vector_path = raw_vector_path(index_name, segment_id);
            *docid_path = raw_docid_path(index_name, segment_id);
            if (vector_path->empty() || docid_path->empty()) return false;
            created_files.push_back(*vector_path);
            created_files.push_back(*docid_path);
            return true;
          },
          &raw_segments)) {
    for (const std::string &path : created_files) remove_file_if_exists(path);
    return false;
  }
  if (!assign_raw_segments(state, raw_segments)) {
    *state = before_compact;
    for (const std::string &path : created_files) remove_file_if_exists(path);
    return false;
  }
  state->memory_entries.clear();
  state->memory_erases.clear();
  state->memory_bytes = 0;
  state->live_doc_ids.clear();
  for (const auto &entry : entries) state->live_doc_ids.insert(entry.first);
  state->entry_count = entries.size();
  state->generation = state->segments.back().generation;
  state->next_segment_id += state->segments.size();
  state->build_source = "raw_segments_compacted";

  if (!save_manifest(index_name, *state)) {
    *state = before_compact;
    for (const std::string &path : created_files) remove_file_if_exists(path);
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
  if (!consolidate_raw_locator_runs(&state_it->second)) return false;

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

  return target->rebuild_from_raw_segments(
      [this, &index_name](const raw_vector_segment_visitor &visitor) {
        return read_rebuild_raw_segments(index_name, visitor);
      });
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

bool standalone_entry_store::find_entries(
    const std::string &index_name, const std::unordered_set<uint64_t> &doc_ids,
    committed_entries *vectors) const {
  if (vectors == nullptr) return false;
  vectors->clear();
  if (doc_ids.empty()) return true;

  auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return false;
  const index_state &state = state_it->second;

  const size_t vector_bytes = state.dimension * sizeof(float);
  if (state.dimension == 0 ||
      vector_bytes >
          static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    return false;
  }

  using candidate_row = std::pair<uint64_t, uint32_t>;
  std::vector<std::vector<candidate_row>> raw_candidates(state.segments.size());
  for (const std::shared_ptr<const raw_entry_locator> &locator_run :
       state.raw_locator_runs) {
    if (locator_run == nullptr) return false;
    const raw_entry_locator &locator = *locator_run;
    for (const uint64_t doc_id : doc_ids) {
      auto location_it = std::lower_bound(
          locator.begin(), locator.end(), doc_id,
          [](const raw_entry_location &location, uint64_t target_doc_id) {
            return location.doc_id < target_doc_id;
          });
      while (location_it != locator.end() && location_it->doc_id == doc_id) {
        const size_t segment_index = location_it->segment_index;
        if (segment_index >= state.segments.size() ||
            state.segments[segment_index].kind !=
                standalone_segment_kind::kRawFbin ||
            state.segments[segment_index].dense_doc_ids ||
            location_it->row >= state.segments[segment_index].record_count) {
          return false;
        }
        raw_candidates[segment_index].emplace_back(doc_id, location_it->row);
        ++location_it;
      }
    }
  }
  for (std::vector<candidate_row> &candidates : raw_candidates) {
    std::sort(candidates.begin(), candidates.end(),
              [](const candidate_row &left, const candidate_row &right) {
                return left.second < right.second;
              });
  }

  for (size_t segment_index = 0; segment_index < state.segments.size();
       ++segment_index) {
    const standalone_segment &segment = state.segments[segment_index];
    if (segment.dimension != state.dimension) return false;

    if (segment.kind == standalone_segment_kind::kRawFbin) {
      constexpr uint64_t fbin_header_bytes =
          sizeof(uint32_t) + sizeof(uint32_t);
      if (segment.record_count >
          (std::numeric_limits<uint64_t>::max() - fbin_header_bytes) /
              vector_bytes) {
        return false;
      }
      const uint64_t expected_vector_bytes =
          fbin_header_bytes + segment.record_count * vector_bytes;
      size_t actual_vector_bytes = 0;
      if (expected_vector_bytes > std::numeric_limits<size_t>::max() ||
          !file_size_as_size(segment.vector_path, &actual_vector_bytes) ||
          actual_vector_bytes != expected_vector_bytes) {
        return false;
      }

      std::ifstream vector_file(segment.vector_path,
                                std::ios::in | std::ios::binary);
      if (!vector_file.is_open()) return false;

      uint32_t vector_rows = 0;
      uint32_t vector_dimension = 0;
      if (!read_plain_value(vector_file, vector_rows) ||
          !read_plain_value(vector_file, vector_dimension) ||
          vector_rows != segment.record_count ||
          vector_dimension != state.dimension) {
        return false;
      }

      if (segment.dense_doc_ids) {
        const uint64_t last_doc_id =
            segment.record_count == 0
                ? segment.first_doc_id
                : segment.first_doc_id + segment.record_count - 1;
        if (segment.record_count != 0 && last_doc_id < segment.first_doc_id) {
          return false;
        }
        for (const uint64_t doc_id : doc_ids) {
          if (segment.record_count == 0 || doc_id < segment.first_doc_id ||
              doc_id > last_doc_id) {
            continue;
          }
          const uint64_t row = doc_id - segment.first_doc_id;
          if (row > (std::numeric_limits<uint64_t>::max() -
                     fbin_header_bytes) /
                        vector_bytes) {
            return false;
          }
          const uint64_t offset = fbin_header_bytes + row * vector_bytes;
          if (offset >
              static_cast<uint64_t>(
                  std::numeric_limits<std::streamoff>::max())) {
            return false;
          }

          vector_data vector(state.dimension);
          vector_file.seekg(static_cast<std::streamoff>(offset),
                            std::ios::beg);
          vector_file.read(reinterpret_cast<char *>(vector.data()),
                           static_cast<std::streamsize>(vector_bytes));
          if (!vector_file.good()) return false;
          (*vectors)[doc_id] = std::move(vector);
        }
        continue;
      }

      constexpr uint64_t docid_header_bytes = sizeof(uint64_t);
      if (segment.record_count >
          (std::numeric_limits<uint64_t>::max() - docid_header_bytes) /
              sizeof(uint64_t)) {
        return false;
      }
      const uint64_t expected_docid_bytes =
          docid_header_bytes + segment.record_count * sizeof(uint64_t);
      size_t actual_docid_bytes = 0;
      if (expected_docid_bytes > std::numeric_limits<size_t>::max() ||
          !file_size_as_size(segment.docid_path, &actual_docid_bytes) ||
          actual_docid_bytes != expected_docid_bytes) {
        return false;
      }

      std::ifstream docid_file(segment.docid_path,
                               std::ios::in | std::ios::binary);
      if (!docid_file.is_open()) return false;

      uint64_t docid_rows = 0;
      if (!read_plain_value(docid_file, docid_rows) ||
          docid_rows != segment.record_count) {
        return false;
      }
      if (docid_rows != 0 && state.raw_locator_runs.empty()) {
        return false;
      }
      for (const candidate_row &candidate : raw_candidates[segment_index]) {
        const uint64_t doc_id = candidate.first;
        const uint64_t row = candidate.second;
        if (row > (std::numeric_limits<uint64_t>::max() -
                   fbin_header_bytes) /
                      vector_bytes) {
          return false;
        }
        const uint64_t offset = fbin_header_bytes + row * vector_bytes;
        if (offset >
            static_cast<uint64_t>(
                std::numeric_limits<std::streamoff>::max())) {
          return false;
        }

        vector_data vector(state.dimension);
        vector_file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        vector_file.read(reinterpret_cast<char *>(vector.data()),
                         static_cast<std::streamsize>(vector_bytes));
        if (!vector_file.good()) return false;
        (*vectors)[doc_id] = std::move(vector);
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
        !read_plain_value(file, record_count) || dimension != state.dimension ||
        record_count != segment.record_count) {
      return false;
    }
    for (uint64_t i = 0; i < record_count; ++i) {
      uint8_t op = 0;
      uint64_t doc_id = 0;
      if (!read_plain_value(file, op) || !read_plain_value(file, doc_id)) {
        return false;
      }

      if (op == kStandaloneSegmentUpsert) {
        if (doc_ids.find(doc_id) == doc_ids.end()) {
          file.seekg(static_cast<std::streamoff>(vector_bytes), std::ios::cur);
          if (!file.good()) return false;
          continue;
        }

        vector_data vector(state.dimension);
        file.read(reinterpret_cast<char *>(vector.data()),
                  static_cast<std::streamsize>(vector_bytes));
        if (!file.good()) return false;
        (*vectors)[doc_id] = std::move(vector);
      } else if (op == kStandaloneSegmentErase) {
        vectors->erase(doc_id);
      } else {
        return false;
      }
    }
  }

  for (const auto &entry : state.memory_entries) {
    if (doc_ids.find(entry.first) != doc_ids.end()) {
      (*vectors)[entry.first] = entry.second;
    }
  }
  for (const uint64_t doc_id : state.memory_erases) {
    vectors->erase(doc_id);
  }
  for (auto entry_it = vectors->begin(); entry_it != vectors->end();) {
    if (state.live_doc_ids.find(entry_it->first) == state.live_doc_ids.end()) {
      entry_it = vectors->erase(entry_it);
    } else {
      ++entry_it;
    }
  }
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

size_t standalone_entry_store::raw_locator_run_count(
    const std::string &index_name) const {
  const auto state_it = m_indexes.find(index_name);
  return state_it == m_indexes.end() ? 0
                                     : state_it->second.raw_locator_runs.size();
}

size_t standalone_entry_store::raw_locator_entry_count(
    const std::string &index_name) const {
  const auto state_it = m_indexes.find(index_name);
  if (state_it == m_indexes.end()) return 0;

  size_t entry_count = 0;
  for (const std::shared_ptr<const raw_entry_locator> &run :
       state_it->second.raw_locator_runs) {
    if (run == nullptr ||
        run->size() > std::numeric_limits<size_t>::max() - entry_count) {
      return std::numeric_limits<size_t>::max();
    }
    entry_count += run->size();
  }
  return entry_count;
}

size_t standalone_entry_store::raw_locator_bytes(
    const std::string &index_name) const {
  const size_t entry_count = raw_locator_entry_count(index_name);
  if (entry_count >
      std::numeric_limits<size_t>::max() / sizeof(raw_entry_location)) {
    return std::numeric_limits<size_t>::max();
  }
  return entry_count * sizeof(raw_entry_location);
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
      uint64_t first_doc_id = 0;
      if (detect_dense_docid_file(segment.docid_path, segment.record_count,
                                  &first_doc_id)) {
        segment.dense_doc_ids = true;
        segment.first_doc_id = first_doc_id;
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
  std::shared_ptr<const raw_entry_locator> locator_run;
  if (!build_raw_locator_run(loaded.segments, 0, &locator_run)) return false;
  if (locator_run != nullptr && !locator_run->empty()) {
    loaded.raw_locator_runs.push_back(std::move(locator_run));
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

bool index_configs_equal(const index_service::index_config &lhs,
                         const index_service::index_config &rhs) {
  return lhs.dimension == rhs.dimension && lhs.metric == rhs.metric &&
         lhs.mode == rhs.mode && lhs.provider == rhs.provider &&
         lhs.backend_variant == rhs.backend_variant &&
         lhs.search_ef == rhs.search_ef && lhs.hnsw_m == rhs.hnsw_m &&
         lhs.hnsw_ef_construction == rhs.hnsw_ef_construction &&
         lhs.hnsw_build_threads == rhs.hnsw_build_threads &&
         lhs.faiss_nlist == rhs.faiss_nlist &&
         lhs.faiss_nprobe == rhs.faiss_nprobe &&
         lhs.faiss_pq_m == rhs.faiss_pq_m &&
         lhs.faiss_pq_bits == rhs.faiss_pq_bits &&
         lhs.faiss_build_threads == rhs.faiss_build_threads &&
         lhs.diskann_max_degree == rhs.diskann_max_degree &&
         lhs.diskann_build_complexity == rhs.diskann_build_complexity &&
         lhs.diskann_build_threads == rhs.diskann_build_threads &&
         lhs.diskann_build_blas_threads == rhs.diskann_build_blas_threads &&
         lhs.diskann_build_mode_value == rhs.diskann_build_mode_value &&
         lhs.diskann_search_complexity == rhs.diskann_search_complexity &&
         lhs.diskann_search_beamwidth == rhs.diskann_search_beamwidth &&
         lhs.diskann_pq_code_budget_size ==
             rhs.diskann_pq_code_budget_size &&
         lhs.diskann_disk_pq_dims == rhs.diskann_disk_pq_dims &&
         lhs.diskann_cache_nodes == rhs.diskann_cache_nodes &&
         lhs.diskann_accelerate_build == rhs.diskann_accelerate_build &&
         lhs.diskann_shuffle_build == rhs.diskann_shuffle_build &&
         lhs.diskann_use_bfs_cache == rhs.diskann_use_bfs_cache &&
         lhs.diskann_build_mode_specified ==
             rhs.diskann_build_mode_specified &&
         lhs.diskann_segmented_serving == rhs.diskann_segmented_serving &&
         lhs.consistency_mode == rhs.consistency_mode;
}

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

vector_index::index_service::index_config with_dynamic_build_options(
    vector_index::index_service::index_config config) {
  config.diskann_segmented_serving = opt_vector_diskann_segmented_serving;
  return config;
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
  if (config.diskann_build_blas_threads != 0 &&
      !backend->set_diskann_build_blas_threads(
          config.diskann_build_blas_threads)) {
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
  if ((config.provider == backend_provider::kDiskAnn ||
       config.diskann_disk_pq_dims != 0) &&
      !backend->set_diskann_disk_pq_dims(config.diskann_disk_pq_dims)) {
    return nullptr;
  }
  if ((config.provider == backend_provider::kDiskAnn ||
       config.diskann_accelerate_build) &&
      !backend->set_diskann_accelerate_build(config.diskann_accelerate_build)) {
    return nullptr;
  }
  if ((config.provider == backend_provider::kDiskAnn ||
       config.diskann_shuffle_build) &&
      !backend->set_diskann_shuffle_build(config.diskann_shuffle_build)) {
    return nullptr;
  }
  if ((config.provider == backend_provider::kDiskAnn ||
       config.diskann_use_bfs_cache) &&
      !backend->set_diskann_use_bfs_cache(config.diskann_use_bfs_cache)) {
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
using detail::index_configs_equal;

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

std::string rebuild_error_from_diagnostics(
    const backend_build_diagnostics &diagnostics) {
  constexpr const char *prefix = "LOAD VECTOR DATA could not rebuild index";
  if (!diagnostics.native_pq_runtime_artifact_validation.empty() &&
      diagnostics.native_pq_runtime_artifact_validation != "ok") {
    return std::string(prefix) + ": " +
           diagnostics.native_pq_runtime_artifact_validation;
  }
  if (!diagnostics.fallback_reason.empty()) {
    return std::string(prefix) + ": " + diagnostics.fallback_reason;
  }
  return prefix;
}

void set_bulk_load_rebuild_error(std::string *error,
                                 const backend *rebuilt_backend) {
  if (rebuilt_backend == nullptr) {
    set_bulk_load_error(error, "LOAD VECTOR DATA could not rebuild index");
    return;
  }
  set_bulk_load_error(error,
                      rebuild_error_from_diagnostics(
                          rebuilt_backend->build_diagnostics()));
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
  config->diskann_build_blas_threads =
      index_backend.diskann_build_blas_threads();
  config->diskann_build_mode_value =
      index_backend.diskann_build_mode_value();
  config->diskann_pq_code_budget_size =
      index_backend.diskann_pq_code_budget_size();
  config->diskann_disk_pq_dims = index_backend.diskann_disk_pq_dims();
  config->diskann_cache_nodes = index_backend.diskann_cache_nodes();
  config->diskann_accelerate_build = index_backend.diskann_accelerate_build();
  config->diskann_shuffle_build = index_backend.diskann_shuffle_build();
  config->diskann_use_bfs_cache = index_backend.diskann_use_bfs_cache();
}

template <typename Apply>
bool apply_backend_config_exclusively(
    const vector_index::index_service::backend_ptr &runtime, Apply apply) {
  if (runtime == nullptr) return false;
  std::unique_lock<std::shared_mutex> runtime_guard(runtime->runtime_mutex());
  return apply(runtime.get());
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

committed_entry_source make_committed_entry_source(
    const vector_entry_store &entry_store, const std::string &index_name) {
  return {make_committed_entry_reader(entry_store, index_name),
          entry_store.entry_count(index_name), true};
}

bool load_backend_from_store(const vector_entry_store &entry_store,
                             const std::string &index_name, backend *target) {
  return target != nullptr &&
         target->load_committed_entries_from_source(
             make_committed_entry_source(entry_store, index_name));
}

bool rebuild_backend_from_store(const vector_entry_store &entry_store,
                                const std::string &index_name,
                                backend *target) {
  return target != nullptr &&
         target->rebuild_from_committed_entry_source(
             make_committed_entry_source(entry_store, index_name));
}

bool recover_backend_from_store(const vector_entry_store &entry_store,
                                const std::string &index_name,
                                backend *target) {
  return target != nullptr &&
         target->recover_committed_entries_from_source(
             make_committed_entry_source(entry_store, index_name));
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

size_t segmented_search_threads(const index_service::index_config &config,
                                size_t segment_count) {
  if (segment_count <= 1) return 1;
  switch (config.provider) {
    case backend_provider::kDiskAnn:
      return effective_diskann_search_threads(segment_count);
    case backend_provider::kFaiss:
      return effective_faiss_search_threads(segment_count);
    case backend_provider::kHnswlib:
      return effective_hnsw_search_threads(segment_count);
    case backend_provider::kNative:
      return effective_batch_search_threads(segment_count, 0);
  }
  return 1;
}

bool uses_shared_segmented_search_pool(backend_provider provider) {
  return provider == backend_provider::kHnswlib ||
         provider == backend_provider::kFaiss;
}

size_t shared_segmented_search_thread_budget(
    const index_service::index_config &config, size_t work_item_count) {
  switch (config.provider) {
    case backend_provider::kHnswlib:
      return effective_hnsw_search_thread_budget();
    case backend_provider::kFaiss:
      return effective_faiss_search_threads(work_item_count);
    case backend_provider::kDiskAnn:
    case backend_provider::kNative:
      return 1;
  }
  return 1;
}

class segmented_backend final : public backend {
 public:
  segmented_backend(index_service::index_config config,
                    std::vector<std::shared_ptr<backend>> segments,
                    backend_build_diagnostics diagnostics)
      : m_config(std::move(config)),
        m_segments(std::move(segments)),
        m_diagnostics(std::move(diagnostics)) {
    m_segment_entry_counts.reserve(m_segments.size());
    for (const auto &segment : m_segments) {
      m_segment_entry_counts.push_back(segment == nullptr
                                           ? 0
                                           : segment->entry_count());
    }
  }

  bool upsert(uint64_t doc_id [[maybe_unused]],
              const vector_data &vector [[maybe_unused]]) override {
    return false;
  }

  bool erase(uint64_t doc_id [[maybe_unused]]) override { return false; }

  bool search(const vector_data &query, size_t top_k,
              std::vector<search_result> *results) const override {
    return search_impl(query, top_k, top_k, false, results);
  }

  bool search_for_rerank(const vector_data &query, size_t top_k,
                         size_t candidate_top_k,
                         std::vector<search_result> *results) const override {
    if (results == nullptr) return false;
    results->clear();
    if (query.size() != m_config.dimension) return false;
    if (top_k == 0) return true;

    const size_t merge_top_k = std::max(top_k, candidate_top_k);
    if (collect_all_candidates_for_rerank(merge_top_k, top_k, 1, results)) {
      return true;
    }
    return search_impl(query, top_k, merge_top_k, true, results);
  }

  bool search_impl(const vector_data &query, size_t top_k, size_t merge_top_k,
                   bool exact_rerank_candidates,
                   std::vector<search_result> *results) const {
    if (results == nullptr) return false;
    results->clear();
    if (query.size() != m_config.dimension) return false;
    if (top_k == 0) return true;

    search_budget_summary budget;
    if (!compute_segment_search_budget(top_k, merge_top_k, 1, &budget)) {
      return false;
    }
    backend_search_options search_options;
    if (!make_backend_search_options(budget, &search_options)) return false;

    std::vector<std::vector<search_result>> segment_results(m_segments.size());
    const auto search_segment = [&](size_t segment_index) {
      const auto &segment = m_segments[segment_index];
      if (segment == nullptr) return false;
      if (exact_rerank_candidates &&
          collect_full_segment_candidates_for_rerank(
              *segment, budget.segment_top_k[segment_index],
              &segment_results[segment_index])) {
        return true;
      }
      return segment->search_with_options(
          query, budget.segment_top_k[segment_index], search_options,
          &segment_results[segment_index]);
    };

    size_t worker_count =
        segmented_search_threads(m_config, m_segments.size());
    shared_search_execution execution;
    bool search_ok = false;
    const bool use_shared_search_pool =
        uses_shared_segmented_search_pool(m_config.provider);
    if (use_shared_search_pool) {
      search_ok = parallel_for_shared_search_items(
          m_segments.size(),
          shared_segmented_search_thread_budget(m_config, m_segments.size()),
          [&](size_t item_index, size_t) {
            return search_segment(item_index);
          },
          &execution);
      worker_count = execution.effective_workers;
    } else {
      search_ok = parallel_for_ranges_scoped(
          m_segments.size(), worker_count,
          [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
              if (!search_segment(i)) return false;
            }
            return true;
          });
    }
    record_search_budget(
        worker_count, top_k, 1, budget,
        use_shared_search_pool ? &execution : nullptr);
    if (!search_ok) return false;
    return merge_segment_topk(segment_results, merge_top_k, results);
  }

  bool search_batch(const std::vector<vector_data> &queries, size_t top_k,
                    std::vector<std::vector<search_result>> *results)
      const override {
    return search_batch_impl(queries, top_k, top_k, false, results);
  }

  bool search_batch_for_rerank(
      const std::vector<vector_data> &queries, size_t top_k,
      size_t candidate_top_k, batch_search_candidates *results) const override {
    if (results == nullptr) return false;
    results->clear();
    for (const auto &query : queries) {
      if (query.size() != m_config.dimension) return false;
    }
    if (top_k == 0) {
      results->prepare_per_query(queries.size());
      return true;
    }

    const size_t merge_top_k = std::max(top_k, candidate_top_k);
    std::vector<search_result> candidates;
    if (collect_all_candidates_for_rerank(merge_top_k, top_k, queries.size(),
                                          &candidates)) {
      results->set_shared(queries.size(), std::move(candidates));
      return true;
    }
    std::vector<std::vector<search_result>> per_query_candidates;
    if (!search_batch_impl(queries, top_k, merge_top_k, true,
                           &per_query_candidates)) {
      return false;
    }
    results->set_per_query(std::move(per_query_candidates));
    return true;
  }

  bool search_batch_impl(const std::vector<vector_data> &queries, size_t top_k,
                         size_t merge_top_k, bool exact_rerank_candidates,
                         std::vector<std::vector<search_result>> *results)
      const {
    if (results == nullptr) return false;
    results->clear();
    for (const auto &query : queries) {
      if (query.size() != m_config.dimension) return false;
    }
    if (top_k == 0) {
      results->resize(queries.size());
      return true;
    }
    if (m_segments.empty()) {
      results->resize(queries.size());
      return true;
    }

    search_budget_summary budget;
    if (!compute_segment_search_budget(top_k, merge_top_k, queries.size(),
                                       &budget)) {
      return false;
    }
    backend_search_options search_options;
    if (!make_backend_search_options(budget, &search_options)) return false;

    std::vector<batch_search_candidates> segment_batch_results(
        m_segments.size());
    size_t worker_count =
        segmented_search_threads(m_config, m_segments.size());
    shared_search_execution execution;
    bool search_ok = false;
    const bool use_shared_search_pool =
        uses_shared_segmented_search_pool(m_config.provider);
    if (use_shared_search_pool) {
      if (!queries.empty() &&
          m_segments.size() >
              std::numeric_limits<size_t>::max() / queries.size()) {
        return false;
      }
      for (batch_search_candidates &segment_results : segment_batch_results) {
        segment_results.prepare_per_query(queries.size());
      }
      const size_t work_items = m_segments.size() * queries.size();
      search_ok = parallel_for_shared_search_items(
          work_items,
          shared_segmented_search_thread_budget(m_config, work_items),
          [&](size_t item_index, size_t) {
            const size_t segment_index = item_index / queries.size();
            const size_t query_index = item_index % queries.size();
            const auto &segment = m_segments[segment_index];
            if (segment == nullptr) return false;
            std::vector<search_result> *query_results =
                segment_batch_results[segment_index].mutable_for_query(
                    query_index);
            if (query_results == nullptr) return false;
            if (exact_rerank_candidates &&
                collect_full_segment_candidates_for_rerank(
                    *segment, budget.segment_top_k[segment_index],
                    query_results)) {
              return true;
            }
            return segment->search_with_options(
                queries[query_index], budget.segment_top_k[segment_index],
                search_options, query_results);
          },
          &execution);
      worker_count = execution.effective_workers;
    } else {
      search_ok = parallel_for_ranges_scoped(
          m_segments.size(), worker_count,
          [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
              const auto &segment = m_segments[i];
              if (segment == nullptr) return false;
              std::vector<search_result> segment_candidates;
              if (exact_rerank_candidates &&
                  collect_full_segment_candidates_for_rerank(
                      *segment, budget.segment_top_k[i],
                      &segment_candidates)) {
                segment_batch_results[i].set_shared(
                    queries.size(), std::move(segment_candidates));
                continue;
              }
              std::vector<std::vector<search_result>> per_query_candidates;
              if (!segment->search_batch_with_options(
                      queries, budget.segment_top_k[i], search_options,
                      &per_query_candidates)) {
                return false;
              }
              segment_batch_results[i].set_per_query(
                  std::move(per_query_candidates));
            }
            return true;
          });
    }
    record_search_budget(
        worker_count, top_k, queries.size(), budget,
        use_shared_search_pool ? &execution : nullptr);
    if (!search_ok) return false;
    return merge_segment_batch_candidates_topk(segment_batch_results,
                                               merge_top_k, results);
  }

  static bool collect_full_segment_candidates_for_rerank(
      const backend &segment, size_t requested_top_k,
      std::vector<search_result> *results) {
    if (results == nullptr) return false;
    const size_t segment_entry_count = segment.entry_count();
    if (segment_entry_count == 0 || requested_top_k < segment_entry_count) {
      return false;
    }

    std::vector<uint64_t> doc_ids;
    if (!segment.collect_doc_ids(&doc_ids) ||
        doc_ids.size() != segment_entry_count) {
      return false;
    }

    results->clear();
    results->reserve(doc_ids.size());
    for (const uint64_t doc_id : doc_ids) {
      results->push_back(search_result{doc_id, 0.0});
    }
    return true;
  }

  bool collect_doc_ids(std::vector<uint64_t> *doc_ids) const override {
    if (doc_ids == nullptr) return false;
    doc_ids->clear();

    const size_t total_entry_count = entry_count();
    doc_ids->reserve(total_entry_count);
    for (const auto &segment : m_segments) {
      if (segment == nullptr) return false;
      std::vector<uint64_t> segment_doc_ids;
      if (!segment->collect_doc_ids(&segment_doc_ids)) return false;
      doc_ids->insert(doc_ids->end(), segment_doc_ids.begin(),
                      segment_doc_ids.end());
    }
    return doc_ids->size() == total_entry_count;
  }

  bool collect_all_candidates_for_rerank(size_t requested_top_k, size_t top_k,
                                         size_t query_count,
                                         std::vector<search_result> *results)
      const {
    if (results == nullptr) return false;
    const size_t total_entry_count = entry_count();
    if (total_entry_count == 0) return false;

    search_budget_summary budget;
    if (!compute_segment_search_budget(top_k, requested_top_k, query_count,
                                       &budget)) {
      return false;
    }
    if (budget.segment_top_k.size() != m_segments.size()) return false;
    for (size_t i = 0; i < m_segments.size(); ++i) {
      const auto &segment = m_segments[i];
      if (segment == nullptr) return false;
      if (budget.segment_top_k[i] < segment->entry_count()) return false;
    }
    record_search_budget(segmented_search_threads(m_config, m_segments.size()),
                         top_k, query_count, budget);

    std::vector<uint64_t> doc_ids;
    if (!collect_doc_ids(&doc_ids)) return false;

    results->clear();
    results->reserve(doc_ids.size());
    for (const uint64_t doc_id : doc_ids) {
      results->push_back(search_result{doc_id, 0.0});
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
    backend_build_diagnostics diagnostics = m_diagnostics;
    diagnostics.search_fanout_segments =
        m_search_fanout_segments.load(std::memory_order_relaxed);
    diagnostics.search_fanout_threads =
        m_search_fanout_threads.load(std::memory_order_relaxed);
    diagnostics.search_worker_budget =
        m_search_worker_budget.load(std::memory_order_relaxed);
    diagnostics.search_active_requests =
        m_search_active_requests.load(std::memory_order_relaxed);
    diagnostics.search_work_items =
        m_search_work_items.load(std::memory_order_relaxed);
    diagnostics.search_global_top_k =
        m_search_global_top_k.load(std::memory_order_relaxed);
    diagnostics.search_per_segment_top_k =
        m_search_per_segment_top_k.load(std::memory_order_relaxed);
    diagnostics.search_result_budget =
        m_search_result_budget.load(std::memory_order_relaxed);
    diagnostics.search_candidate_count =
        m_search_candidate_count.load(std::memory_order_relaxed);
    diagnostics.search_query_count =
        m_search_query_count.load(std::memory_order_relaxed);
    diagnostics.search_segment_min_entries =
        m_search_segment_min_entries.load(std::memory_order_relaxed);
    diagnostics.search_segment_max_entries =
        m_search_segment_max_entries.load(std::memory_order_relaxed);
    diagnostics.search_segment_total_entries =
        m_search_segment_total_entries.load(std::memory_order_relaxed);
    diagnostics.search_diskann_search_list =
        m_search_diskann_search_list.load(std::memory_order_relaxed);
    diagnostics.search_diskann_beamwidth =
        m_search_diskann_beamwidth.load(std::memory_order_relaxed);
    diagnostics.search_effective_complexity =
        m_search_effective_complexity.load(std::memory_order_relaxed);
    diagnostics.search_total_candidate_rows =
        m_search_total_candidate_rows.load(std::memory_order_relaxed);
    if (diagnostics.search_effective_complexity != 0) {
      diagnostics.search_profile = diskann_search_profile_name(
          static_cast<diskann_search_profile>(
              m_search_profile.load(std::memory_order_relaxed)));
      diagnostics.search_profile_reason = diskann_search_profile_reason_name(
          static_cast<diskann_search_profile_reason>(
              m_search_profile_reason.load(std::memory_order_relaxed)));
    }
    return diagnostics;
  }
  bool set_search_ef(uint32_t search_ef) override {
    if (!apply_to_segments(
            [&](backend &segment) { return segment.set_search_ef(search_ef); })) {
      return false;
    }
    m_config.search_ef = search_ef;
    return true;
  }
  uint32_t search_ef() const override { return m_config.search_ef; }
  bool set_faiss_ivf_params(uint32_t faiss_nlist,
                            uint32_t faiss_nprobe) override {
    if (m_config.provider != backend_provider::kFaiss) return false;
    if (!apply_to_segments([&](backend &segment) {
          return segment.set_faiss_ivf_params(faiss_nlist, faiss_nprobe);
        })) {
      return false;
    }
    m_config.faiss_nlist = faiss_nlist;
    m_config.faiss_nprobe = faiss_nprobe;
    return true;
  }
  bool set_faiss_ivf_pq_params(uint32_t faiss_nlist, uint32_t faiss_nprobe,
                               uint32_t faiss_pq_m,
                               uint32_t faiss_pq_bits) override {
    if (m_config.provider != backend_provider::kFaiss) return false;
    if (!apply_to_segments([&](backend &segment) {
          return segment.set_faiss_ivf_pq_params(faiss_nlist, faiss_nprobe,
                                                 faiss_pq_m, faiss_pq_bits);
        })) {
      return false;
    }
    m_config.faiss_nlist = faiss_nlist;
    m_config.faiss_nprobe = faiss_nprobe;
    m_config.faiss_pq_m = faiss_pq_m;
    m_config.faiss_pq_bits = faiss_pq_bits;
    return true;
  }
  bool set_diskann_build_params(uint32_t diskann_max_degree,
                                uint32_t diskann_build_complexity,
                                uint32_t diskann_build_threads) override {
    if (m_config.provider != backend_provider::kDiskAnn ||
        diskann_max_degree == 0 || diskann_build_complexity == 0 ||
        diskann_build_threads > k_max_build_threads) {
      return false;
    }
    if (!apply_to_segments([&](backend &segment) {
          return segment.set_diskann_build_params(
              diskann_max_degree, diskann_build_complexity,
              diskann_build_threads);
        })) {
      return false;
    }
    m_config.diskann_max_degree = diskann_max_degree;
    m_config.diskann_build_complexity = diskann_build_complexity;
    m_config.diskann_build_threads = diskann_build_threads;
    return true;
  }
  bool set_diskann_build_threads(uint32_t diskann_build_threads) override {
    if (m_config.provider != backend_provider::kDiskAnn ||
        diskann_build_threads > k_max_build_threads) {
      return false;
    }
    if (!apply_to_segments([&](backend &segment) {
          return segment.set_diskann_build_threads(diskann_build_threads);
        })) {
      return false;
    }
    m_config.diskann_build_threads = diskann_build_threads;
    return true;
  }
  bool set_diskann_build_mode(
      diskann_build_mode diskann_build_mode_value) override {
    if (m_config.provider != backend_provider::kDiskAnn) return false;
    if (!apply_to_segments([&](backend &segment) {
          return segment.set_diskann_build_mode(diskann_build_mode_value);
        })) {
      return false;
    }
    m_config.diskann_build_mode_value = diskann_build_mode_value;
    return true;
  }
  bool set_diskann_search_complexity(
      uint32_t diskann_search_complexity) override {
    if (m_config.provider != backend_provider::kDiskAnn ||
        diskann_search_complexity == 0) {
      return false;
    }
    if (!apply_to_segments([&](backend &segment) {
          return segment.set_diskann_search_complexity(
              diskann_search_complexity);
        })) {
      return false;
    }
    m_config.diskann_search_complexity = diskann_search_complexity;
    return true;
  }
  bool set_diskann_search_beamwidth(
      uint32_t diskann_search_beamwidth) override {
    if (m_config.provider != backend_provider::kDiskAnn ||
        !valid_diskann_search_beamwidth(diskann_search_beamwidth)) {
      return false;
    }
    if (!apply_to_segments([&](backend &segment) {
          return segment.set_diskann_search_beamwidth(diskann_search_beamwidth);
        })) {
      return false;
    }
    m_config.diskann_search_beamwidth = diskann_search_beamwidth;
    return true;
  }
  bool set_diskann_pq_code_budget_size(
      uint64_t diskann_pq_code_budget_size) override {
    if (m_config.provider != backend_provider::kDiskAnn) return false;
    if (!apply_to_segments([&](backend &segment) {
          return segment.set_diskann_pq_code_budget_size(
              diskann_pq_code_budget_size);
        })) {
      return false;
    }
    m_config.diskann_pq_code_budget_size = diskann_pq_code_budget_size;
    return true;
  }
  bool set_diskann_disk_pq_dims(uint32_t diskann_disk_pq_dims) override {
    if (m_config.provider != backend_provider::kDiskAnn ||
        diskann_disk_pq_dims > k_max_diskann_disk_pq_dims) {
      return false;
    }
    if (!apply_to_segments([&](backend &segment) {
          return segment.set_diskann_disk_pq_dims(diskann_disk_pq_dims);
        })) {
      return false;
    }
    m_config.diskann_disk_pq_dims = diskann_disk_pq_dims;
    return true;
  }
  bool set_diskann_accelerate_build(
      bool diskann_accelerate_build) override {
    if (m_config.provider != backend_provider::kDiskAnn) return false;
    if (!apply_to_segments([&](backend &segment) {
          return segment.set_diskann_accelerate_build(diskann_accelerate_build);
        })) {
      return false;
    }
    m_config.diskann_accelerate_build = diskann_accelerate_build;
    return true;
  }
  bool set_diskann_shuffle_build(bool diskann_shuffle_build) override {
    if (m_config.provider != backend_provider::kDiskAnn) return false;
    if (!apply_to_segments([&](backend &segment) {
          return segment.set_diskann_shuffle_build(diskann_shuffle_build);
        })) {
      return false;
    }
    m_config.diskann_shuffle_build = diskann_shuffle_build;
    return true;
  }
  bool set_diskann_use_bfs_cache(bool diskann_use_bfs_cache) override {
    if (m_config.provider != backend_provider::kDiskAnn) return false;
    if (!apply_to_segments([&](backend &segment) {
          return segment.set_diskann_use_bfs_cache(diskann_use_bfs_cache);
        })) {
      return false;
    }
    m_config.diskann_use_bfs_cache = diskann_use_bfs_cache;
    return true;
  }
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
  uint32_t diskann_disk_pq_dims() const override {
    return m_config.diskann_disk_pq_dims;
  }
  bool diskann_accelerate_build() const override {
    return m_config.diskann_accelerate_build;
  }
  bool diskann_shuffle_build() const override {
    return m_config.diskann_shuffle_build;
  }
  bool diskann_use_bfs_cache() const override {
    return m_config.diskann_use_bfs_cache;
  }
  bool supports_mutations() const override { return false; }
  bool external_manifest_present() const override {
    if (m_segments.empty()) return false;
    for (const auto &segment : m_segments) {
      if (segment == nullptr || !segment->external_manifest_present()) {
        return false;
      }
    }
    return true;
  }
  uint64_t external_manifest_generation() const override {
    uint64_t generation = 0;
    for (const auto &segment : m_segments) {
      if (segment == nullptr) return 0;
      const uint64_t segment_generation =
          segment->external_manifest_generation();
      if (segment_generation == 0 ||
          (generation != 0 && segment_generation != generation)) {
        return 0;
      }
      generation = segment_generation;
    }
    return generation;
  }
  bool defer_artifact_publication() override {
    return apply_to_segments(
        [](backend &segment) { return segment.defer_artifact_publication(); });
  }
  bool has_pending_artifact_publication() const override {
    return std::any_of(
        m_segments.begin(), m_segments.end(), [](const auto &segment) {
          return segment != nullptr &&
                 segment->has_pending_artifact_publication();
        });
  }
  bool prepare_artifact_publication(
      const diskann_artifact_identity &identity) override {
    if (!artifact_identity_count_matches(identity)) return false;
    std::vector<backend *> prepared;
    prepared.reserve(m_segments.size());
    for (size_t i = 0; i < m_segments.size(); ++i) {
      const auto &segment = m_segments[i];
      const diskann_artifact_identity segment_identity =
          artifact_identity_for_segment(identity, i);
      if (segment == nullptr ||
          (segment->has_pending_artifact_publication() &&
           !segment->prepare_artifact_publication(segment_identity))) {
        for (auto it = prepared.rbegin(); it != prepared.rend(); ++it) {
          (void)(*it)->rollback_artifact();
        }
        return false;
      }
      if (segment->has_pending_artifact_publication()) {
        prepared.push_back(segment.get());
      }
    }
    return true;
  }
  bool publish_artifact() override {
    std::vector<backend *> pending;
    pending.reserve(m_segments.size());
    for (const auto &segment : m_segments) {
      if (segment == nullptr) return false;
      if (segment->has_pending_artifact_publication()) {
        pending.push_back(segment.get());
      }
    }
    for (backend *segment : pending) {
      if (segment->publish_artifact()) continue;
      for (auto it = pending.rbegin(); it != pending.rend(); ++it) {
        (void)(*it)->rollback_artifact();
      }
      return false;
    }
    return true;
  }
  bool rollback_artifact() override {
    bool ok = true;
    for (auto it = m_segments.rbegin(); it != m_segments.rend(); ++it) {
      if (*it != nullptr && (*it)->has_pending_artifact_publication() &&
          !(*it)->rollback_artifact()) {
        ok = false;
      }
    }
    return ok;
  }
  bool finalize_artifact() override {
    bool ok = true;
    for (const auto &segment : m_segments) {
      if (segment != nullptr &&
          segment->has_pending_artifact_publication() &&
          !segment->finalize_artifact()) {
        ok = false;
      }
    }
    return ok;
  }
  bool recover_artifact_publication(
      const diskann_artifact_identity &identity) override {
    if (!artifact_identity_count_matches(identity)) return false;
    for (size_t i = 0; i < m_segments.size(); ++i) {
      const auto &segment = m_segments[i];
      if (segment == nullptr ||
          !segment->recover_artifact_publication(
              artifact_identity_for_segment(identity, i))) {
        return false;
      }
    }
    return true;
  }
  bool artifact_publication_matches(
      const diskann_artifact_identity &identity) const override {
    if (m_segments.empty() || !artifact_identity_count_matches(identity)) {
      return false;
    }
    for (size_t i = 0; i < m_segments.size(); ++i) {
      const auto &segment = m_segments[i];
      if (segment == nullptr ||
          !segment->artifact_publication_matches(
              artifact_identity_for_segment(identity, i))) {
        return false;
      }
    }
    return true;
  }

 private:
  bool artifact_identity_count_matches(
      const diskann_artifact_identity &identity) const {
    uint64_t total = 0;
    for (const size_t segment_count : m_segment_entry_counts) {
      if (segment_count > std::numeric_limits<uint64_t>::max() - total) {
        return false;
      }
      total += static_cast<uint64_t>(segment_count);
    }
    return total == identity.doc_id_count;
  }

  diskann_artifact_identity artifact_identity_for_segment(
      const diskann_artifact_identity &identity, size_t segment_index) const {
    diskann_artifact_identity segment_identity = identity;
    segment_identity.doc_id_count =
        static_cast<uint64_t>(m_segment_entry_counts[segment_index]);
    return segment_identity;
  }

  struct search_budget_summary {
    std::vector<size_t> segment_top_k;
    size_t max_segment_top_k{0};
    size_t candidate_count{0};
    size_t min_segment_entry_count{0};
    size_t max_segment_entry_count{0};
    size_t total_segment_entry_count{0};
    size_t diskann_search_list{0};
    size_t diskann_beamwidth{0};
    size_t diskann_effective_complexity{0};
    diskann_search_profile diskann_profile{diskann_search_profile::kManual};
    diskann_search_profile_reason diskann_profile_reason{
        diskann_search_profile_reason::kManual};
  };

  bool compute_segment_search_budget(size_t top_k,
                                     size_t global_candidate_top_k,
                                     size_t query_count,
                                     search_budget_summary *budget) const {
    if (budget == nullptr) {
      return false;
    }
    budget->segment_top_k.assign(m_segments.size(), 0);
    budget->max_segment_top_k = 0;
    budget->candidate_count = 0;
    budget->min_segment_entry_count =
        m_segments.empty() ? 0 : std::numeric_limits<size_t>::max();
    budget->max_segment_entry_count = 0;
    budget->total_segment_entry_count = 0;
    budget->diskann_search_list = 0;
    budget->diskann_beamwidth = 0;
    budget->diskann_effective_complexity = 0;
    budget->diskann_profile = diskann_search_profile::kManual;
    budget->diskann_profile_reason = diskann_search_profile_reason::kManual;

    const size_t result_budget =
        static_cast<size_t>(opt_vector_search_batch_result_count);
    diskann_search_budget diskann_budget;
    for (size_t i = 0; i < m_segments.size(); ++i) {
      const auto &segment = m_segments[i];
      if (segment == nullptr) return false;
      const size_t segment_entry_count = segment->entry_count();
      budget->min_segment_entry_count =
          std::min(budget->min_segment_entry_count, segment_entry_count);
      budget->max_segment_entry_count =
          std::max(budget->max_segment_entry_count, segment_entry_count);
      budget->total_segment_entry_count =
          saturated_add_size(budget->total_segment_entry_count,
                             segment_entry_count);
    }
    if (m_segments.empty()) budget->min_segment_entry_count = 0;

    const bool use_diskann_budget =
        m_config.provider == backend_provider::kDiskAnn &&
        m_config.mode == backend_mode::kExternal;
    if (use_diskann_budget &&
        !choose_diskann_search_budget(
            static_cast<uint32_t>(
                std::min<size_t>(top_k, std::numeric_limits<uint32_t>::max())),
            static_cast<uint32_t>(std::min<size_t>(
                m_segments.size(), std::numeric_limits<uint32_t>::max())),
            static_cast<uint64_t>(budget->min_segment_entry_count),
            static_cast<uint64_t>(budget->max_segment_entry_count),
            m_config.diskann_search_complexity,
            static_cast<diskann_search_profile>(
                opt_vector_diskann_search_profile),
            &diskann_budget)) {
      return false;
    }

    for (size_t i = 0; i < m_segments.size(); ++i) {
      const auto &segment = m_segments[i];
      if (segment == nullptr) return false;
      const size_t segment_entry_count = segment->entry_count();
      size_t current_top_k = segmented_search_candidate_top_k(
          top_k, m_segments.size(), segment_entry_count, query_count,
          result_budget);
      if (use_diskann_budget) {
        budget->diskann_search_list = diskann_budget.search_complexity;
        budget->diskann_effective_complexity =
            diskann_budget.search_complexity;
        budget->diskann_profile = diskann_budget.profile;
        budget->diskann_profile_reason = diskann_budget.reason;
        budget->diskann_beamwidth =
            m_config.diskann_search_beamwidth == 0
                ? static_cast<size_t>(k_default_diskann_search_beamwidth)
                : static_cast<size_t>(m_config.diskann_search_beamwidth);
        current_top_k =
            std::min(current_top_k,
                     static_cast<size_t>(diskann_budget.per_segment_top_k));
      }
      if (global_candidate_top_k > top_k && !m_segments.empty()) {
        const size_t target_per_segment =
            saturated_add_size(global_candidate_top_k,
                               m_segments.size() - 1) /
            m_segments.size();
        const size_t approximate_candidate_limit =
            use_diskann_budget
                ? diskann_search_list_slack_top_k(
                      top_k, diskann_budget.search_complexity, 1)
                : segment_entry_count;
        const size_t overfetch_factor =
            use_diskann_budget
                ? k_diskann_segment_candidate_overfetch_factor
                : 1;
        const size_t overfetch_top_k = std::min(
            saturated_mul_size(target_per_segment, overfetch_factor),
            approximate_candidate_limit);
        current_top_k = std::max(current_top_k, overfetch_top_k);
        if (result_budget != 0 && query_count != 0) {
          const size_t budget_denominator =
              saturated_mul_size(query_count, m_segments.size());
          const size_t budget_per_segment =
              budget_denominator == 0 ? 0 : result_budget / budget_denominator;
          if (budget_per_segment < top_k) {
            current_top_k = top_k;
          } else {
            current_top_k = std::min(current_top_k, budget_per_segment);
          }
        }
      }
      current_top_k = std::min(current_top_k, segment_entry_count);
      budget->segment_top_k[i] = current_top_k;
      budget->max_segment_top_k =
          std::max(budget->max_segment_top_k, current_top_k);
      budget->candidate_count =
          saturated_add_size(budget->candidate_count, current_top_k);
    }
    return true;
  }

  bool make_backend_search_options(
      const search_budget_summary &budget,
      backend_search_options *options) const {
    if (options == nullptr) return false;
    *options = {};
    if (m_config.provider != backend_provider::kDiskAnn ||
        m_config.mode != backend_mode::kExternal) {
      return true;
    }
    if (budget.diskann_effective_complexity == 0 ||
        budget.diskann_effective_complexity >
            std::numeric_limits<uint32_t>::max() ||
        budget.diskann_beamwidth == 0 ||
        budget.diskann_beamwidth > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    options->diskann_search_complexity =
        static_cast<uint32_t>(budget.diskann_effective_complexity);
    options->diskann_search_beamwidth =
        static_cast<uint32_t>(budget.diskann_beamwidth);
    return true;
  }

  void record_search_budget(
      size_t worker_count, size_t top_k, size_t query_count,
      const search_budget_summary &budget,
      const shared_search_execution *execution = nullptr) const {
    m_search_fanout_segments.store(
        static_cast<uint64_t>(m_segments.size()), std::memory_order_relaxed);
    m_search_fanout_threads.store(static_cast<uint64_t>(worker_count),
                                  std::memory_order_relaxed);
    m_search_worker_budget.store(
        static_cast<uint64_t>(execution == nullptr ? worker_count
                                                   : execution->worker_budget),
        std::memory_order_relaxed);
    m_search_active_requests.store(
        static_cast<uint64_t>(execution == nullptr
                                  ? 1
                                  : execution->active_requests),
        std::memory_order_relaxed);
    m_search_work_items.store(
        static_cast<uint64_t>(execution == nullptr ? m_segments.size()
                                                   : execution->work_items),
        std::memory_order_relaxed);
    m_search_global_top_k.store(static_cast<uint64_t>(top_k),
                                std::memory_order_relaxed);
    m_search_per_segment_top_k.store(
        static_cast<uint64_t>(budget.max_segment_top_k),
        std::memory_order_relaxed);
    m_search_result_budget.store(
        static_cast<uint64_t>(opt_vector_search_batch_result_count),
        std::memory_order_relaxed);
    m_search_candidate_count.store(static_cast<uint64_t>(budget.candidate_count),
                                   std::memory_order_relaxed);
    m_search_query_count.store(static_cast<uint64_t>(query_count),
                               std::memory_order_relaxed);
    m_search_segment_min_entries.store(
        static_cast<uint64_t>(budget.min_segment_entry_count),
        std::memory_order_relaxed);
    m_search_segment_max_entries.store(
        static_cast<uint64_t>(budget.max_segment_entry_count),
        std::memory_order_relaxed);
    m_search_segment_total_entries.store(
        static_cast<uint64_t>(budget.total_segment_entry_count),
        std::memory_order_relaxed);
    m_search_diskann_search_list.store(
        static_cast<uint64_t>(budget.diskann_search_list),
        std::memory_order_relaxed);
    m_search_diskann_beamwidth.store(
        static_cast<uint64_t>(budget.diskann_beamwidth),
        std::memory_order_relaxed);
    m_search_effective_complexity.store(
        static_cast<uint64_t>(budget.diskann_effective_complexity),
        std::memory_order_relaxed);
    m_search_total_candidate_rows.store(
        static_cast<uint64_t>(
            saturated_mul_size(query_count, budget.candidate_count)),
        std::memory_order_relaxed);
    m_search_profile.store(static_cast<uint64_t>(budget.diskann_profile),
                           std::memory_order_relaxed);
    m_search_profile_reason.store(
        static_cast<uint64_t>(budget.diskann_profile_reason),
        std::memory_order_relaxed);
  }

  template <typename Apply>
  bool apply_to_segments(const Apply &apply) {
    for (const auto &segment : m_segments) {
      if (segment != nullptr && !apply(*segment)) return false;
    }
    return true;
  }

  index_service::index_config m_config;
  std::vector<std::shared_ptr<backend>> m_segments;
  std::vector<size_t> m_segment_entry_counts;
  backend_build_diagnostics m_diagnostics;
  mutable std::atomic<uint64_t> m_search_fanout_segments{0};
  mutable std::atomic<uint64_t> m_search_fanout_threads{0};
  mutable std::atomic<uint64_t> m_search_worker_budget{0};
  mutable std::atomic<uint64_t> m_search_active_requests{0};
  mutable std::atomic<uint64_t> m_search_work_items{0};
  mutable std::atomic<uint64_t> m_search_global_top_k{0};
  mutable std::atomic<uint64_t> m_search_per_segment_top_k{0};
  mutable std::atomic<uint64_t> m_search_result_budget{0};
  mutable std::atomic<uint64_t> m_search_candidate_count{0};
  mutable std::atomic<uint64_t> m_search_query_count{0};
  mutable std::atomic<uint64_t> m_search_segment_min_entries{0};
  mutable std::atomic<uint64_t> m_search_segment_max_entries{0};
  mutable std::atomic<uint64_t> m_search_segment_total_entries{0};
  mutable std::atomic<uint64_t> m_search_diskann_search_list{0};
  mutable std::atomic<uint64_t> m_search_diskann_beamwidth{0};
  mutable std::atomic<uint64_t> m_search_effective_complexity{0};
  mutable std::atomic<uint64_t> m_search_total_candidate_rows{0};
  mutable std::atomic<uint64_t> m_search_profile{
      static_cast<uint64_t>(diskann_search_profile::kManual)};
  mutable std::atomic<uint64_t> m_search_profile_reason{
      static_cast<uint64_t>(diskann_search_profile_reason::kManual)};
};

bool raw_segments_use_single_backend(
    const index_service::index_config &config) {
  if (config.provider != backend_provider::kDiskAnn) return false;
  if (config.mode != backend_mode::kExternal) return false;
  if (config.diskann_segmented_serving) return false;
  return config.diskann_build_mode_value == diskann_build_mode::kAuto ||
         config.diskann_build_mode_value == diskann_build_mode::kOffline;
}

std::string segment_backend_index_name(const std::string &index_name,
                                       uint64_t segment_id) {
  return index_name + "#segment-" + std::to_string(segment_id);
}

uint32_t configured_segment_build_threads(
    const index_service::index_config &config) {
  switch (config.provider) {
    case backend_provider::kDiskAnn:
      return config.diskann_build_threads;
    case backend_provider::kFaiss:
      return config.faiss_build_threads;
    case backend_provider::kHnswlib:
      return config.hnsw_build_threads;
    case backend_provider::kNative:
      return 0;
  }
  return 0;
}

uint32_t configured_segment_blas_threads(
    const index_service::index_config &config) {
  if (config.provider != backend_provider::kDiskAnn) return 1;
  return config.diskann_build_blas_threads != 0
             ? config.diskann_build_blas_threads
             : static_cast<uint32_t>(
                   std::min<ulong>(opt_vector_diskann_build_blas_threads,
                                   k_max_build_threads));
}

bool diskann_segment_profile_eligible(
    const index_service::index_config &config) {
  return config.provider == backend_provider::kDiskAnn &&
         config.mode == backend_mode::kExternal &&
         config.consistency_mode == index_consistency_mode::kStandalone &&
         detail::effective_diskann_build_mode(config) ==
             diskann_build_mode::kOffline;
}

diskann_segment_profile_result effective_diskann_segment_profile(
    const index_service::index_config &config,
    const build_input_stats &stats,
    const build_pipeline_runtime_config &pipeline_config) {
  return apply_diskann_segment_profile(
      static_cast<uint64_t>(config.dimension), stats.row_count,
      stats.payload_size, pipeline_config.diskann_profile,
      pipeline_config.thresholds, diskann_segment_profile_eligible(config));
}

segment_scheduler_input make_segment_scheduler_input(
    const index_service::index_config &config,
    const std::vector<raw_vector_segment> &raw_segments,
    const build_pipeline_runtime_config &pipeline_config) {
  segment_scheduler_input input;
  input.segment_count = static_cast<uint32_t>(std::min<size_t>(
      raw_segments.size(), std::numeric_limits<uint32_t>::max()));
  input.requested_task_count = pipeline_config.thresholds.max_tasks;
  input.requested_build_threads = configured_segment_build_threads(config);
  input.cpu_budget = input.requested_build_threads;
  input.requested_blas_threads = configured_segment_blas_threads(config);
  input.build_memory_budget =
      config.provider == backend_provider::kDiskAnn
          ? static_cast<uint64_t>(opt_vector_diskann_build_memory_size)
          : 0;
  if (input.build_memory_budget != 0) {
    size_t max_segment_payload = 0;
    for (const raw_vector_segment &segment : raw_segments) {
      max_segment_payload = std::max(max_segment_payload, segment.bytes);
    }
    input.per_segment_memory_estimate = static_cast<uint64_t>(
        saturated_add_size(saturated_mul_size(max_segment_payload, 4),
                           1024ULL * 1024ULL * 1024ULL));
  }
  input.single_index_build = raw_segments_use_single_backend(config);
  return input;
}

index_service::index_config apply_segment_scheduler_plan(
    index_service::index_config config, const segment_scheduler_plan &plan) {
  switch (config.provider) {
    case backend_provider::kDiskAnn:
      config.diskann_build_threads = plan.effective_build_threads;
      config.diskann_build_blas_threads = plan.effective_blas_threads;
      break;
    case backend_provider::kFaiss:
      config.faiss_build_threads = plan.effective_build_threads;
      break;
    case backend_provider::kHnswlib:
      config.hnsw_build_threads = plan.effective_build_threads;
      break;
    case backend_provider::kNative:
      break;
  }
  return config;
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

class raw_segment_snapshot_writer {
 public:
  raw_segment_snapshot_writer(std::string directory, size_t dimension)
      : raw_segment_snapshot_writer(
            dimension, 1, 1,
            [directory = std::move(directory)](uint64_t segment_id,
                                               std::string *vector_path,
                                               std::string *docid_path) {
              if (vector_path == nullptr || docid_path == nullptr) return false;
              std::filesystem::path vector_file_path(directory);
              vector_file_path /= "truth-store-segment-" +
                                  std::to_string(segment_id) + ".fbin";
              std::filesystem::path docid_file_path(directory);
              docid_file_path /= "truth-store-segment-" +
                                 std::to_string(segment_id) + ".u64";
              *vector_path = vector_file_path.string();
              *docid_path = docid_file_path.string();
              return true;
            }) {}

  raw_segment_snapshot_writer(size_t dimension, uint64_t first_segment_id,
                              uint64_t first_generation,
                              raw_segment_path_provider path_provider)
      : m_dimension(dimension),
        m_row_limit(std::min<uint64_t>(
            raw_segment_row_limit(
                dimension, global_build_pipeline_runtime_config().thresholds),
            std::numeric_limits<uint32_t>::max())),
        m_next_segment_id(first_segment_id),
        m_next_generation(first_generation),
        m_path_provider(std::move(path_provider)) {}

  bool append(uint64_t doc_id, const vector_data &vector) {
    if (vector.size() != m_dimension) return false;
    if (!m_open && !open_segment()) return false;
    if (m_current_rows >= m_row_limit) {
      if (!close_segment() || !open_segment()) return false;
    }

    if (!write_plain_value(m_docid_file, doc_id)) return false;
    const size_t vector_bytes = m_dimension * sizeof(float);
    if (vector_bytes >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      return false;
    }
    m_vector_file.write(reinterpret_cast<const char *>(vector.data()),
                        static_cast<std::streamsize>(vector_bytes));
    if (!m_vector_file.good()) return false;
    ++m_current_rows;
    return true;
  }

  bool finish(std::vector<raw_vector_segment> *segments) {
    if (segments == nullptr) return false;
    if (m_open && !close_segment()) return false;
    *segments = std::move(m_segments);
    return true;
  }

 private:
  bool open_segment() {
    if (m_open || m_dimension == 0 || m_row_limit == 0 || !m_path_provider)
      return false;
    const uint64_t segment_id = m_next_segment_id++;
    const uint64_t generation = m_next_generation++;
    std::string vector_path;
    std::string docid_path;
    if (!m_path_provider(segment_id, &vector_path, &docid_path) ||
        vector_path.empty() || docid_path.empty()) {
      return false;
    }

    if (!detail::ensure_parent_directory(vector_path) ||
        !detail::ensure_parent_directory(docid_path)) {
      return false;
    }

    m_current = raw_vector_segment{};
    m_current.vector_path = vector_path;
    m_current.docid_path = docid_path;
    m_current.dimension = m_dimension;
    m_current.generation = generation;
    m_current_rows = 0;

    m_vector_file.open(m_current.vector_path,
                       std::ios::out | std::ios::binary | std::ios::trunc);
    m_docid_file.open(m_current.docid_path,
                      std::ios::out | std::ios::binary | std::ios::trunc);
    if (!m_vector_file.is_open() || !m_docid_file.is_open()) {
      remove_file_if_exists(m_current.vector_path);
      remove_file_if_exists(m_current.docid_path);
      return false;
    }

    if (!write_plain_value(m_vector_file, static_cast<uint32_t>(0)) ||
        !write_plain_value(m_vector_file, static_cast<uint32_t>(m_dimension)) ||
        !write_plain_value(m_docid_file, static_cast<uint64_t>(0))) {
      remove_file_if_exists(m_current.vector_path);
      remove_file_if_exists(m_current.docid_path);
      return false;
    }
    m_open = true;
    return true;
  }

  bool close_segment() {
    if (!m_open) return true;
    if (m_current_rows > std::numeric_limits<uint32_t>::max()) return false;

    m_vector_file.seekp(0);
    m_docid_file.seekp(0);
    if (!write_plain_value(m_vector_file,
                           static_cast<uint32_t>(m_current_rows)) ||
        !write_plain_value(m_docid_file,
                           static_cast<uint64_t>(m_current_rows))) {
      return false;
    }
    m_vector_file.close();
    m_docid_file.close();
    if (!m_vector_file || !m_docid_file) return false;

    if (m_current_rows == 0) {
      remove_file_if_exists(m_current.vector_path);
      remove_file_if_exists(m_current.docid_path);
      m_open = false;
      return true;
    }

    size_t vector_bytes = 0;
    size_t docid_bytes = 0;
    if (!file_size_as_size(m_current.vector_path, &vector_bytes) ||
        !file_size_as_size(m_current.docid_path, &docid_bytes) ||
        vector_bytes > std::numeric_limits<size_t>::max() - docid_bytes) {
      return false;
    }
    m_current.row_count = static_cast<size_t>(m_current_rows);
    m_current.bytes = vector_bytes + docid_bytes;
    m_segments.push_back(m_current);
    m_open = false;
    return true;
  }

  size_t m_dimension{0};
  uint64_t m_row_limit{1};
  uint64_t m_next_segment_id{1};
  uint64_t m_next_generation{1};
  uint64_t m_current_rows{0};
  bool m_open{false};
  raw_segment_path_provider m_path_provider;
  raw_vector_segment m_current;
  std::ofstream m_vector_file;
  std::ofstream m_docid_file;
  std::vector<raw_vector_segment> m_segments;
};

bool write_entries_to_raw_segments(size_t dimension, uint64_t first_segment_id,
                                   uint64_t first_generation,
                                   const committed_entries &entries,
                                   raw_segment_path_provider path_provider,
                                   std::vector<raw_vector_segment> *segments) {
  if (dimension == 0 || first_segment_id == 0 || first_generation == 0 ||
      !path_provider || segments == nullptr) {
    return false;
  }
  segments->clear();
  if (entries.empty()) return true;

  raw_segment_snapshot_writer writer(dimension, first_segment_id,
                                     first_generation,
                                     std::move(path_provider));
  std::vector<uint64_t> doc_ids;
  doc_ids.reserve(entries.size());
  for (const auto &entry : entries) doc_ids.push_back(entry.first);
  std::sort(doc_ids.begin(), doc_ids.end());

  for (uint64_t doc_id : doc_ids) {
    const auto entry_it = entries.find(doc_id);
    if (entry_it == entries.end() || entry_it->second.size() != dimension ||
        !writer.append(doc_id, entry_it->second)) {
      return false;
    }
  }
  return writer.finish(segments) && !segments->empty();
}

struct raw_segment_snapshot {
  std::string directory;
  std::vector<raw_vector_segment> segments;

  void cleanup() const {
    if (directory.empty()) return;
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
  }
};

bool export_truth_store_snapshot_to_raw_segments(
    const vector_entry_store &entry_store, const std::string &index_name,
    const index_service::index_config &config, raw_segment_snapshot *snapshot) {
  if (snapshot == nullptr || config.dimension == 0) return false;
  snapshot->cleanup();
  *snapshot = raw_segment_snapshot{};

  std::filesystem::path directory(detail::external_snapshot_directory(index_name));
  if (directory.empty()) return false;
  directory /= "transactional-offline-raw-tmp";

  std::error_code ec;
  std::filesystem::remove_all(directory, ec);
  if (ec) return false;
  std::filesystem::create_directories(directory, ec);
  if (ec) return false;

  raw_segment_snapshot_writer writer(directory.string(), config.dimension);
  const bool read_ok = entry_store.for_each_committed_entry(
      index_name, [&writer](uint64_t doc_id, const vector_data &vector) {
        return writer.append(doc_id, vector);
      });
  if (!read_ok || !writer.finish(&snapshot->segments)) {
    std::filesystem::remove_all(directory, ec);
    return false;
  }
  snapshot->directory = directory.string();
  return true;
}

void clear_raw_input_paths(
    std::vector<vector_index_metadata_store::segment_task_row> *task_rows) {
  if (task_rows == nullptr) return;
  for (auto &row : *task_rows) {
    row.vector_path.clear();
    row.docid_path.clear();
  }
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
  if (segment_backend == nullptr ||
      !segment_backend->defer_artifact_publication()) {
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

std::unique_ptr<backend> build_segmented_backend_from_raw_segments(
    const std::string &index_name, const index_service::index_config &config,
    std::vector<raw_vector_segment> raw_segments, const char *input_source,
    bool keep_raw_input_paths,
    std::vector<vector_index_metadata_store::segment_task_row> *task_rows) {
  if (task_rows != nullptr) task_rows->clear();
  std::vector<std::shared_ptr<backend>> segments(raw_segments.size());
  std::vector<backend_build_diagnostics> segment_diagnostics(
      raw_segments.size());
  std::vector<vector_index_metadata_store::segment_task_row> segment_task_rows;
  segment_task_rows.reserve(raw_segments.size());
  for (size_t i = 0; i < raw_segments.size(); ++i) {
    const raw_vector_segment &segment = raw_segments[i];
    const uint64_t segment_id =
        segment.generation == 0 ? i + 1 : segment.generation;
    segment_task_rows.push_back(
        make_segment_task_row(index_name, segment, segment_id,
                              vector_index_metadata_store::
                                  segment_task_state::kPending));
  }
  DBUG_EXECUTE_IF("vector_segment_backend_build_failure", {
    if (!segment_task_rows.empty()) {
      segment_task_rows.front().state =
          vector_index_metadata_store::segment_task_state::kFailed;
      segment_task_rows.front().last_error_code = ERROR_REPLAY_STATE_FAILED;
      segment_task_rows.front().updated_ts = now_unix_epoch_seconds();
    }
    if (task_rows != nullptr) *task_rows = segment_task_rows;
    return nullptr;
  });

  if (raw_segments_use_single_backend(config)) {
    const build_pipeline_runtime_config pipeline_config =
        global_build_pipeline_runtime_config();
    segment_scheduler_plan scheduler_plan;
    if (!make_segment_scheduler_plan(
            make_segment_scheduler_input(config, raw_segments,
                                         pipeline_config),
            &scheduler_plan)) {
      if (!keep_raw_input_paths) clear_raw_input_paths(&segment_task_rows);
      if (task_rows != nullptr) *task_rows = segment_task_rows;
      return nullptr;
    }
    const index_service::index_config single_config =
        apply_segment_scheduler_plan(config, scheduler_plan);
    std::unique_ptr<backend> rebuilt =
        build_backend_from_config(index_name, single_config);
    const auto mark_tasks_done =
        [&segment_task_rows](
            vector_index_metadata_store::segment_task_state state,
            uint32_t error_code) {
          const uint64_t now = now_unix_epoch_seconds();
          for (auto &row : segment_task_rows) {
            row.state = state;
            row.attempt = 1;
            row.last_error_code = error_code;
            row.updated_ts = now;
          }
        };
    if (rebuilt == nullptr) {
      mark_tasks_done(vector_index_metadata_store::segment_task_state::kFailed,
                      ERROR_BACKEND_CREATE_FAILED);
      if (!keep_raw_input_paths) clear_raw_input_paths(&segment_task_rows);
      if (task_rows != nullptr) *task_rows = segment_task_rows;
      return nullptr;
    }
    if (raw_segments.empty()) {
      if (!keep_raw_input_paths) clear_raw_input_paths(&segment_task_rows);
      if (task_rows != nullptr) *task_rows = segment_task_rows;
      return rebuilt;
    }
    const raw_vector_segment_reader reader =
        [&raw_segments](const raw_vector_segment_visitor &visitor) {
          if (!visitor) return false;
          for (const raw_vector_segment &segment : raw_segments) {
            if (!visitor(segment)) return false;
          }
          return true;
        };
    if (!rebuilt->rebuild_from_raw_segments(reader)) {
      mark_tasks_done(vector_index_metadata_store::segment_task_state::kFailed,
                      ERROR_REPLAY_STATE_FAILED);
      if (!keep_raw_input_paths) clear_raw_input_paths(&segment_task_rows);
      if (task_rows != nullptr) *task_rows = segment_task_rows;
      return nullptr;
    }
    mark_tasks_done(vector_index_metadata_store::segment_task_state::kReady,
                    ERROR_NONE);
    if (!keep_raw_input_paths) clear_raw_input_paths(&segment_task_rows);
    if (task_rows != nullptr) *task_rows = segment_task_rows;
    return rebuilt;
  }

  backend_build_diagnostics aggregate;
  aggregate.runtime = kSegmentedBackendVariant;
  aggregate.input_source = input_source == nullptr ? "raw_segments" : input_source;

  if (raw_segments.empty()) {
    if (!keep_raw_input_paths) clear_raw_input_paths(&segment_task_rows);
    if (task_rows != nullptr) *task_rows = segment_task_rows;
    return std::make_unique<segmented_backend>(config, std::move(segments),
                                               std::move(aggregate));
  }

  const build_pipeline_runtime_config pipeline_config =
      global_build_pipeline_runtime_config();
  segment_scheduler_plan scheduler_plan;
  if (!make_segment_scheduler_plan(
          make_segment_scheduler_input(config, raw_segments, pipeline_config),
          &scheduler_plan)) {
    if (task_rows != nullptr) *task_rows = segment_task_rows;
    return nullptr;
  }
  const index_service::index_config segment_config =
      apply_segment_scheduler_plan(config, scheduler_plan);

  const build_clock::time_point aggregate_start = build_clock::now();
  const bool build_ok = parallel_for_ranges_scoped(
      raw_segments.size(), scheduler_plan.task_count,
      [&](size_t begin, size_t end, size_t) {
        for (size_t i = begin; i < end; ++i) {
          std::shared_ptr<backend> segment_backend;
          backend_build_diagnostics diagnostics;
          vector_index_metadata_store::segment_task_row task_row =
              segment_task_rows[i];
          if (!build_one_segment_backend(
                  index_name, segment_config, raw_segments[i], i + 1,
                  &segment_backend, &diagnostics, &task_row)) {
            segment_task_rows[i] = task_row;
            return false;
          }
          segment_task_rows[i] = task_row;
          segment_diagnostics[i] = diagnostics;
          segments[i] = std::move(segment_backend);
        }
        return true;
      });
  if (!keep_raw_input_paths) clear_raw_input_paths(&segment_task_rows);
  if (task_rows != nullptr) *task_rows = segment_task_rows;
  if (!build_ok) return nullptr;
  aggregate.build_wall_ms = elapsed_build_ms(aggregate_start);

  aggregate.concurrent_build_tasks = scheduler_plan.task_count;
  aggregate.scheduler_cpu_budget = scheduler_plan.cpu_budget;
  aggregate.effective_build_threads = scheduler_plan.effective_build_threads;
  aggregate.effective_blas_threads = scheduler_plan.effective_blas_threads;
  aggregate.raw_reader_threads = scheduler_plan.raw_reader_threads;
  aggregate.pq_train_threads = scheduler_plan.pq_train_threads;
  aggregate.pq_compress_threads = scheduler_plan.pq_compress_threads;
  aggregate.candidates_per_segment = scheduler_plan.candidates_per_segment;
  aggregate.effective_segment_tasks = scheduler_plan.task_count;
  aggregate.segment_memory_estimate = scheduler_plan.segment_memory_estimate;
  aggregate.segment_memory_budget = scheduler_plan.segment_memory_budget;
  aggregate.segment_parallel_reason = scheduler_plan.segment_parallel_reason;
  aggregate.single_index_build = scheduler_plan.single_index_build;

  for (size_t i = 0; i < raw_segments.size(); ++i) {
    if (segments[i] == nullptr) return nullptr;
    const raw_vector_segment &segment = raw_segments[i];
    const backend_build_diagnostics &diagnostics = segment_diagnostics[i];
    detail::merge_segment_build_diagnostics(segment.row_count, diagnostics,
                                            &aggregate);
  }
  return std::make_unique<segmented_backend>(config, std::move(segments),
                                             std::move(aggregate));
}

std::unique_ptr<backend> build_segmented_backend_from_standalone(
    const std::string &index_name, const index_service::index_config &config,
    standalone_entry_store *standalone_store, uint64_t segment_row_limit,
    standalone_entry_store::raw_segment_compaction *compaction,
    std::vector<vector_index_metadata_store::segment_task_row> *task_rows) {
  if (standalone_store == nullptr ||
      config.consistency_mode != index_consistency_mode::kStandalone) {
    return nullptr;
  }
  if (!standalone_store->prepare_raw_segments_for_rebuild(index_name)) {
    return nullptr;
  }

  std::vector<raw_vector_segment> raw_segments;
  bool read_ok = false;
  if (compaction != nullptr) {
    if (!standalone_store->stage_levelled_compaction(
            index_name, segment_row_limit, compaction)) {
      return nullptr;
    }
    if (compaction->needed) {
      raw_segments = compaction->segments;
      read_ok = true;
    }
  }
  if (!read_ok) {
    read_ok = standalone_store->read_rebuild_raw_segments(
        index_name, [&raw_segments](const raw_vector_segment &segment) {
          raw_segments.push_back(segment);
          return true;
        });
  }
  if (!read_ok) return nullptr;

  return build_segmented_backend_from_raw_segments(
      index_name, config, std::move(raw_segments),
      compaction != nullptr && compaction->needed ? "raw_segments_compacted"
                                                  : "raw_segments",
      true, task_rows);
}

std::unique_ptr<backend> build_segmented_backend_from_source(
    const std::string &index_name, const index_service::index_config &config,
    const vector_entry_store &entry_store,
    standalone_entry_store *standalone_store, uint64_t segment_row_limit,
    standalone_entry_store::raw_segment_compaction *compaction,
    std::vector<vector_index_metadata_store::segment_task_row> *task_rows) {
  if (config.consistency_mode == index_consistency_mode::kStandalone) {
    return build_segmented_backend_from_standalone(
        index_name, config, standalone_store, segment_row_limit, compaction,
        task_rows);
  }
  if (config.consistency_mode != index_consistency_mode::kTransactional ||
      config.provider != backend_provider::kDiskAnn ||
      config.mode != backend_mode::kExternal ||
      detail::effective_diskann_build_mode(config) !=
          diskann_build_mode::kOffline) {
    return nullptr;
  }

  raw_segment_snapshot snapshot;
  if (!export_truth_store_snapshot_to_raw_segments(entry_store, index_name,
                                                   config, &snapshot)) {
    return nullptr;
  }
  std::unique_ptr<backend> rebuilt = build_segmented_backend_from_raw_segments(
      index_name, config, std::move(snapshot.segments), "truth_store_snapshot",
      false, task_rows);
  snapshot.cleanup();
  return rebuilt;
}

bool should_build_segmented(const index_service::index_config &config,
                            const build_pipeline_decision &decision) {
  return decision.path == build_pipeline_path::kSegmented &&
         (config.consistency_mode == index_consistency_mode::kStandalone ||
          (config.consistency_mode == index_consistency_mode::kTransactional &&
           config.provider == backend_provider::kDiskAnn &&
           config.mode == backend_mode::kExternal &&
           detail::effective_diskann_build_mode(config) ==
               diskann_build_mode::kOffline));
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

bool parse_manifest_size_for_testing(const std::string &text, size_t *value) {
  return parse_manifest_size(text, value);
}

bool file_size_as_size_for_testing(const std::string &path, size_t *bytes) {
  return file_size_as_size(path, bytes);
}

bool copy_or_link_file_for_testing(const std::string &source,
                                   const std::string &target) {
  return copy_or_link_file(source, target);
}

bool write_generated_docid_file_for_testing(const std::string &path,
                                            uint64_t row_count) {
  return write_generated_docid_file(path, row_count);
}

bool vector_payload_bytes_for_testing(size_t entry_count, size_t dimension,
                                      size_t *bytes) {
  return vector_payload_bytes(entry_count, dimension, bytes);
}

std::unique_ptr<backend> make_segmented_backend_for_testing(
    index_service::index_config config,
    std::vector<std::shared_ptr<backend>> segments,
    backend_build_diagnostics diagnostics) {
  return std::make_unique<segmented_backend>(
      std::move(config), std::move(segments), std::move(diagnostics));
}

bool raw_segments_use_single_backend_for_testing(
    const index_service::index_config &config) {
  return raw_segments_use_single_backend(config);
}

void index_service::mark_index_ready(const std::string &index_name,
                                     lifecycle_info *lifecycle) {
  mark_lifecycle_ready(lifecycle);
  m_last_failed_build_diagnostics.erase(index_name);
}

bool index_service::build_runtime_from_current_policy(
    const std::string &index_name, const index_config &persisted_config,
    std::unique_ptr<backend> *runtime) {
  if (runtime == nullptr) return false;

  const build_pipeline_snapshot previous_pipeline =
      m_build_pipeline_snapshots[index_name];
  const auto previous_tasks = m_segment_task_rows[index_name];
  const index_config build_config =
      detail::with_dynamic_build_options(persisted_config);
  const build_pipeline_decision decision =
      record_build_pipeline_decision(index_name, build_config);
  const bool segmented = should_build_segmented(build_config, decision);
  const uint64_t segment_row_limit =
      m_build_pipeline_snapshots[index_name].effective_segment_row_limit;

  std::vector<vector_index_metadata_store::segment_task_row> segment_tasks;
  standalone_entry_store::raw_segment_compaction compaction;
  std::unique_ptr<backend> rebuilt =
      segmented
          ? build_segmented_backend_from_source(
                index_name, build_config, m_entry_store, &m_standalone_store,
                segment_row_limit, &compaction, &segment_tasks)
          : build_backend_from_config(index_name, build_config);
  if (rebuilt != nullptr && !segmented &&
      !rebuild_backend_from_source(m_entry_store, m_standalone_store,
                                   index_name, build_config, rebuilt.get())) {
    rebuilt.reset();
  }
  if (rebuilt == nullptr) {
    m_standalone_store.discard_levelled_compaction(&compaction);
    m_build_pipeline_snapshots[index_name] = previous_pipeline;
    m_segment_task_rows[index_name] = previous_tasks;
    return false;
  }
  if (segmented && !m_standalone_store.publish_levelled_compaction(
                       index_name, &compaction)) {
    m_standalone_store.discard_levelled_compaction(&compaction);
    m_build_pipeline_snapshots[index_name] = previous_pipeline;
    m_segment_task_rows[index_name] = previous_tasks;
    return false;
  }

  if (segmented) {
    m_segment_task_rows[index_name] = std::move(segment_tasks);
  } else {
    m_segment_task_rows[index_name].clear();
  }
  *runtime = std::move(rebuilt);
  m_standalone_store.finalize_levelled_compaction(&compaction);
  return true;
}

bool index_service::synchronize_runtime_publication(
    const std::string &index_name) {
  auto publication_it = m_publication_states.find(index_name);
  const auto index_it = m_indexes.find(index_name);
  if (publication_it == m_publication_states.end() ||
      index_it == m_indexes.end() || index_it->second == nullptr) {
    return false;
  }

  publication_it->second.runtime_generation =
      publication_it->second.truth_generation;
  // Backend-private file revisions are observed separately. The publication
  // token records which durable truth generation the artifact serves.
  publication_it->second.artifact_generation =
      index_it->second->external_manifest_present()
          ? publication_it->second.truth_generation
          : 0;
  return true;
}

bool index_service::ensure_runtime_loaded(const std::string &index_name) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto publication_it = m_publication_states.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                publication_it != m_publication_states.end(),
                m_entry_store.has_index(index_name))) {
    return false;
  }
  const bool generation_matches = publication_it->second.runtime_generation ==
                                  publication_it->second.truth_generation;
  const bool standalone =
      config_it->second.consistency_mode ==
      index_consistency_mode::kStandalone;
  const bool source_backed_runtime =
      standalone || (lazy_external_runtime_enabled() &&
                     config_it->second.mode == backend_mode::kExternal);
  if (!source_backed_runtime && generation_matches) return true;
  const size_t committed_count =
      standalone ? m_standalone_store.entry_count(index_name)
                 : m_entry_store.entry_count(index_name);
  if (generation_matches &&
      (committed_count == 0 ||
       index_it->second->entry_count() == committed_count)) {
    return true;
  }

  std::unique_ptr<backend> loaded;
  if (!build_runtime_from_current_policy(index_name, config_it->second,
                                         &loaded)) {
    return false;
  }
  index_it->second = std::move(loaded);
  m_last_failed_build_diagnostics.erase(index_name);
  return synchronize_runtime_publication(index_name);
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

bool index_service::runtime_loaded_for_search(
    const std::string &index_name) const {
  const auto config_it = m_index_configs.find(index_name);
  const auto index_it = m_indexes.find(index_name);
  const auto lifecycle_it = m_lifecycle_infos.find(index_name);
  const auto publication_it = m_publication_states.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                lifecycle_it != m_lifecycle_infos.end(),
                publication_it != m_publication_states.end(),
                m_entry_store.has_index(index_name))) {
    return false;
  }
  if (lifecycle_it->second.state == LIFECYCLE_BULK_LOADING) return false;
  if (publication_it->second.runtime_generation !=
      publication_it->second.truth_generation) {
    return false;
  }

  const bool standalone =
      config_it->second.consistency_mode ==
      index_consistency_mode::kStandalone;
  const bool source_backed_runtime =
      standalone || (lazy_external_runtime_enabled() &&
                     config_it->second.mode == backend_mode::kExternal);
  if (!source_backed_runtime) return true;

  const size_t source_count =
      standalone ? m_standalone_store.entry_count(index_name)
                 : m_entry_store.entry_count(index_name);
  const size_t runtime_count = index_it->second->entry_count();
  if (runtime_count == source_count) return true;
  return standalone && runtime_count > 0 &&
         m_standalone_rebuilds.find(index_name) !=
             m_standalone_rebuilds.end();
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

  std::unique_ptr<backend> rebuilt;
  if (!build_runtime_from_current_policy(index_name, config_it->second,
                                         &rebuilt)) {
    return false;
  }

  index_it->second = std::move(rebuilt);
  m_last_failed_build_diagnostics.erase(index_name);
  mark_recover_fallback(&lifecycle_it->second, true);
  return synchronize_runtime_publication(index_name);
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
    stats.raw_segment_count = raw_segment_count;
  }
  return stats;
}

build_pipeline_decision index_service::make_build_pipeline_decision(
    const std::string &index_name, const index_config &config,
    build_pipeline_snapshot *snapshot) const {
  const build_pipeline_runtime_config runtime_config =
      global_build_pipeline_runtime_config();
  const build_input_stats stats = collect_build_input_stats(index_name, config);
  const diskann_segment_profile_result segment_profile =
      effective_diskann_segment_profile(config, stats, runtime_config);
  const build_pipeline_decision decision =
      select_build_pipeline(stats, segment_profile.thresholds);

  build_pipeline_snapshot candidate;
  candidate.mode = build_pipeline_mode_name(runtime_config.thresholds.mode);
  candidate.diskann_segment_profile =
      diskann_segment_profile_name(runtime_config.diskann_profile);
  candidate.diskann_segment_profile_reason = segment_profile.reason;
  candidate.decision = build_pipeline_path_name(decision.path);
  candidate.trigger = build_pipeline_trigger_name(decision.trigger);
  candidate.row_count = stats.row_count;
  candidate.payload_size = stats.payload_size;
  candidate.raw_segment_count = stats.raw_segment_count;
  candidate.effective_segment_target_size =
      segment_profile.thresholds.segment_target_size;
  candidate.effective_segment_row_limit = build_segment_row_limit(
      static_cast<uint64_t>(config.dimension), segment_profile.thresholds);
  if (snapshot != nullptr) *snapshot = std::move(candidate);
  return decision;
}

build_pipeline_decision index_service::record_build_pipeline_decision(
    const std::string &index_name, const index_config &config) {
  build_pipeline_snapshot snapshot;
  const build_pipeline_decision decision =
      make_build_pipeline_decision(index_name, config, &snapshot);
  m_build_pipeline_snapshots[index_name] = std::move(snapshot);
  return decision;
}

bool index_service::register_index(const std::string &index_name,
                                   std::unique_ptr<backend> backend) {
  if (backend == nullptr || index_name.empty()) return false;

  index_config config;
  config.dimension = backend->dimension();
  config.metric = backend->metric();
  config.mode = backend->mode();
  config.provider = backend->provider();
  config.backend_variant = backend->backend_variant();
  config.search_ef = backend->search_ef();
  config.hnsw_m = backend->hnsw_m();
  config.hnsw_ef_construction = backend->hnsw_ef_construction();
  config.hnsw_build_threads = backend->hnsw_build_threads();
  config.faiss_nlist = backend->faiss_nlist();
  config.faiss_nprobe = backend->faiss_nprobe();
  config.faiss_pq_m = backend->faiss_pq_m();
  config.faiss_pq_bits = backend->faiss_pq_bits();
  config.faiss_build_threads = backend->faiss_build_threads();
  sync_diskann_build_config_from_backend(*backend, &config);
  config.diskann_search_complexity = backend->diskann_search_complexity();
  config.diskann_search_beamwidth = backend->diskann_search_beamwidth();
  config.diskann_build_mode_specified =
      backend->provider() == backend_provider::kDiskAnn &&
      backend->diskann_build_mode_value() != diskann_build_mode::kAuto;
  config.consistency_mode = index_consistency_mode::kTransactional;
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
  if (m_next_index_identity == 0 ||
      m_next_index_identity == std::numeric_limits<uint64_t>::max()) {
    (void)unregister_index(index_name);
    return false;
  }
  index_publication_state publication;
  publication.index_identity = m_next_index_identity++;
  m_publication_states.emplace(index_name, publication);
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
  m_publication_states.erase(index_name);
  m_build_pipeline_snapshots.erase(index_name);
  m_segment_task_rows.erase(index_name);
  m_last_failed_build_diagnostics.erase(index_name);
  m_standalone_rebuilds.erase(index_name);
  return true;
}

bool index_service::rename_index(const std::string &old_index_name,
                                 const std::string &new_index_name) {
  if (old_index_name.empty() || new_index_name.empty()) return false;
  if (old_index_name == new_index_name) return true;

  auto old_index_it = m_indexes.find(old_index_name);
  auto old_config_it = m_index_configs.find(old_index_name);
  auto old_lifecycle_it = m_lifecycle_infos.find(old_index_name);
  auto old_publication_it = m_publication_states.find(old_index_name);
  if (old_index_it == m_indexes.end() ||
      old_config_it == m_index_configs.end() ||
      !m_entry_store.has_index(old_index_name) ||
      old_lifecycle_it == m_lifecycle_infos.end() ||
      old_publication_it == m_publication_states.end()) {
    return false;
  }
  if (m_indexes.find(new_index_name) != m_indexes.end()) return false;

  const index_config config = old_config_it->second;
  const lifecycle_info lifecycle = old_lifecycle_it->second;
  const index_publication_state publication = old_publication_it->second;
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
  backend_build_diagnostics failed_build_diagnostics;
  bool has_failed_build_diagnostics = false;
  auto old_failed_diagnostics_it =
      m_last_failed_build_diagnostics.find(old_index_name);
  if (old_failed_diagnostics_it != m_last_failed_build_diagnostics.end()) {
    failed_build_diagnostics = old_failed_diagnostics_it->second;
    has_failed_build_diagnostics = true;
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
  m_publication_states.erase(old_publication_it);
  m_build_pipeline_snapshots.erase(old_index_name);
  m_segment_task_rows.erase(old_index_name);
  m_last_failed_build_diagnostics.erase(old_index_name);

  m_indexes.emplace(new_index_name, std::move(renamed_backend));
  m_index_configs.emplace(new_index_name, config);
  m_lifecycle_infos.emplace(new_index_name, lifecycle);
  m_publication_states.emplace(new_index_name, publication);
  if (has_pipeline_snapshot) {
    m_build_pipeline_snapshots.emplace(new_index_name,
                                       std::move(pipeline_snapshot));
  }
  if (has_segment_tasks) {
    m_segment_task_rows.emplace(new_index_name, std::move(segment_tasks));
  }
  if (has_failed_build_diagnostics) {
    m_last_failed_build_diagnostics.emplace(
        new_index_name, std::move(failed_build_diagnostics));
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

bool index_service::prepare_standalone_rebuild(
    const std::string &index_name, standalone_rebuild_plan *plan,
    std::string *error) {
  if (error != nullptr) error->clear();
  if (plan == nullptr) {
    set_bulk_load_error(error, "invalid standalone rebuild plan");
    return false;
  }
  *plan = standalone_rebuild_plan{};

  const auto config_it = m_index_configs.find(index_name);
  const auto index_it = m_indexes.find(index_name);
  const auto lifecycle_it = m_lifecycle_infos.find(index_name);
  const auto publication_it = m_publication_states.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end(),
                lifecycle_it != m_lifecycle_infos.end(),
                publication_it != m_publication_states.end(),
                m_entry_store.has_index(index_name),
                m_standalone_store.has_index(index_name))) {
    set_bulk_load_error(error, "vector index not found");
    return false;
  }
  if (config_it->second.consistency_mode !=
      index_consistency_mode::kStandalone) {
    set_bulk_load_error(error, "vector index is not standalone");
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name)) {
    set_bulk_load_error(error, "vector index has pending changes");
    return false;
  }
  if (m_standalone_rebuilds.find(index_name) !=
      m_standalone_rebuilds.end()) {
    set_bulk_load_error(error, "vector index rebuild is already active");
    return false;
  }
  if (!m_standalone_store.prepare_raw_segments_for_rebuild(index_name)) {
    set_bulk_load_error(error,
                        "standalone rebuild could not prepare raw input");
    return false;
  }

  plan->index_name = index_name;
  plan->source_config = config_it->second;
  plan->build_config = detail::with_dynamic_build_options(config_it->second);
  plan->source_entry_count = m_standalone_store.entry_count(index_name);
  plan->source_lifecycle_version = lifecycle_it->second.version;
  plan->source_generation = m_standalone_store.generation(index_name);
  plan->previous_runtime = index_it->second;
  plan->previous_lifecycle = lifecycle_it->second;
  plan->previous_publication = publication_it->second;
  plan->target_publication = publication_it->second;
  plan->target_publication.truth_generation = plan->source_generation;

  const auto pipeline_it = m_build_pipeline_snapshots.find(index_name);
  plan->had_previous_pipeline =
      pipeline_it != m_build_pipeline_snapshots.end();
  if (plan->had_previous_pipeline) {
    plan->previous_pipeline = pipeline_it->second;
  }
  const auto tasks_it = m_segment_task_rows.find(index_name);
  plan->had_previous_segment_tasks = tasks_it != m_segment_task_rows.end();
  if (plan->had_previous_segment_tasks) {
    plan->previous_segment_tasks = tasks_it->second;
  }
  const auto failed_diagnostics_it =
      m_last_failed_build_diagnostics.find(index_name);
  plan->had_previous_failed_build_diagnostics =
      failed_diagnostics_it != m_last_failed_build_diagnostics.end();
  if (plan->had_previous_failed_build_diagnostics) {
    plan->previous_failed_build_diagnostics = failed_diagnostics_it->second;
  }

  const build_pipeline_decision decision = make_build_pipeline_decision(
      index_name, plan->build_config, &plan->pipeline);
  plan->segmented = should_build_segmented(plan->build_config, decision);
  m_standalone_rebuilds.insert(index_name);
  plan->prepared = true;
  return true;
}

bool index_service::build_standalone_rebuild(standalone_rebuild_plan *plan,
                                             std::string *error) {
  if (error != nullptr) error->clear();
  if (plan == nullptr || !plan->prepared || plan->published ||
      plan->index_name.empty()) {
    set_bulk_load_error(error, "standalone rebuild plan is not prepared");
    return false;
  }

  plan->raw_segments.clear();
  plan->segment_tasks.clear();
  if (plan->segmented) {
    if (!m_standalone_store.stage_levelled_compaction(
            plan->index_name, plan->pipeline.effective_segment_row_limit,
            &plan->compaction)) {
      set_bulk_load_error(error,
                          "standalone rebuild could not stage compaction");
      return false;
    }
    if (plan->compaction.needed) {
      plan->raw_segments = plan->compaction.segments;
    }
  }
  if (plan->raw_segments.empty() &&
      !m_standalone_store.read_rebuild_raw_segments(
          plan->index_name,
          [plan](const raw_vector_segment &segment) {
            plan->raw_segments.push_back(segment);
            return true;
          })) {
    set_bulk_load_error(error,
                        "standalone rebuild could not read raw input");
    return false;
  }

  if (plan->segmented) {
    plan->rebuilt_backend = build_segmented_backend_from_raw_segments(
        plan->index_name, plan->build_config, plan->raw_segments,
        plan->compaction.needed ? "raw_segments_compacted" : "raw_segments",
        true, &plan->segment_tasks);
  } else {
    plan->rebuilt_backend =
        build_backend_from_config(plan->index_name, plan->build_config);
    if (plan->rebuilt_backend != nullptr &&
        plan->rebuilt_backend->defer_artifact_publication()) {
      const raw_vector_segment_reader reader =
          [plan](const raw_vector_segment_visitor &visitor) {
            if (!visitor) return false;
            for (const raw_vector_segment &segment : plan->raw_segments) {
              if (!visitor(segment)) return false;
            }
            return true;
          };
      bool rebuild_ok =
          plan->rebuilt_backend->rebuild_from_raw_segments(reader);
      DBUG_EXECUTE_IF(
          "vector_standalone_rebuild_after_backend_build_failure",
          rebuild_ok = false;);
      if (!rebuild_ok) {
        plan->failed_build_diagnostics =
            plan->rebuilt_backend->build_diagnostics();
        plan->has_failed_build_diagnostics = true;
        DBUG_EXECUTE_IF(
            "vector_standalone_rebuild_after_backend_build_failure", {
              plan->failed_build_diagnostics.fallback_reason =
                  "debug_forced_post_build_failure";
            });
        set_bulk_load_error(
            error,
            rebuild_error_from_diagnostics(plan->failed_build_diagnostics));
        plan->rebuilt_backend.reset();
      }
    } else {
      plan->rebuilt_backend.reset();
    }
  }
  if (plan->rebuilt_backend == nullptr) {
    if (error != nullptr && error->empty()) {
      set_bulk_load_error(error,
                          "standalone rebuild could not create backend");
    }
    return false;
  }
  plan->built = true;
  return true;
}

bool index_service::prepare_standalone_rebuild_artifact(
    standalone_rebuild_plan *plan,
    const diskann_artifact_identity &identity, std::string *error) {
  if (error != nullptr) error->clear();
  if (plan == nullptr || !plan->prepared || !plan->built || plan->published ||
      plan->rebuilt_backend == nullptr || plan->artifact_prepared) {
    set_bulk_load_error(error,
                        "standalone rebuild artifact is not ready to prepare");
    return false;
  }
  if (!plan->rebuilt_backend->prepare_artifact_publication(identity)) {
    (void)plan->rebuilt_backend->rollback_artifact();
    set_bulk_load_error(error,
                        "standalone rebuild artifact preparation failed");
    return false;
  }
  plan->artifact_prepared = true;
  return true;
}

bool index_service::publish_standalone_rebuild(
    standalone_rebuild_plan *plan, std::string *error) {
  if (error != nullptr) error->clear();
  if (plan == nullptr || !plan->prepared || !plan->built ||
      plan->published || plan->rebuilt_backend == nullptr) {
    set_bulk_load_error(error, "standalone rebuild plan is not built");
    return false;
  }

  const auto config_it = m_index_configs.find(plan->index_name);
  const auto index_it = m_indexes.find(plan->index_name);
  const auto lifecycle_it = m_lifecycle_infos.find(plan->index_name);
  const auto publication_it = m_publication_states.find(plan->index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end(),
                lifecycle_it != m_lifecycle_infos.end(),
                publication_it != m_publication_states.end(),
                m_standalone_rebuilds.find(plan->index_name) !=
                    m_standalone_rebuilds.end(),
                index_configs_equal(config_it->second, plan->source_config),
                lifecycle_it->second.version ==
                    plan->source_lifecycle_version,
                index_it->second == plan->previous_runtime,
                publication_it->second.index_identity ==
                    plan->previous_publication.index_identity,
                publication_it->second.truth_generation ==
                    plan->previous_publication.truth_generation,
                publication_it->second.config_generation ==
                    plan->previous_publication.config_generation,
                m_standalone_store.generation(plan->index_name) ==
                    plan->source_generation,
                !pending_changes_contain_index(m_pending_changes,
                                               plan->index_name))) {
    set_bulk_load_error(error, "vector index changed during rebuild");
    return false;
  }

  DBUG_EXECUTE_IF("vector_segment_task_before_publish", {
    set_bulk_load_error(error,
                        "LOAD VECTOR DATA could not publish rebuilt backend");
      return false;
  });
  if (plan->rebuilt_backend->has_pending_artifact_publication() &&
      !plan->artifact_prepared) {
    set_bulk_load_error(error,
                        "standalone rebuild artifact is not prepared");
    return false;
  }
  if (plan->segmented &&
      !m_standalone_store.publish_levelled_compaction(plan->index_name,
                                                      &plan->compaction)) {
    set_bulk_load_error(error,
                        "LOAD VECTOR DATA could not publish compacted input");
    return false;
  }
  if (!plan->rebuilt_backend->publish_artifact()) {
    (void)plan->rebuilt_backend->rollback_artifact();
    if (plan->segmented) {
      (void)m_standalone_store.rollback_levelled_compaction(
          plan->index_name, &plan->compaction);
    }
    set_bulk_load_error(error,
                        "standalone rebuild artifact publication failed");
    return false;
  }
  plan->artifact_published =
      plan->rebuilt_backend->has_pending_artifact_publication();

  m_build_pipeline_snapshots[plan->index_name] = plan->pipeline;
  m_segment_task_rows[plan->index_name] = plan->segment_tasks;
  index_it->second = plan->rebuilt_backend;
  publication_it->second = plan->target_publication;
  if (!synchronize_runtime_publication(plan->index_name)) {
    index_it->second = plan->previous_runtime;
    publication_it->second = plan->previous_publication;
    if (plan->artifact_published) {
      (void)plan->rebuilt_backend->rollback_artifact();
      plan->artifact_published = false;
    }
    if (plan->segmented) {
      (void)m_standalone_store.rollback_levelled_compaction(
          plan->index_name, &plan->compaction);
    }
    set_bulk_load_error(error,
                        "standalone rebuild publication state is unavailable");
    return false;
  }
  mark_index_ready(plan->index_name, &lifecycle_it->second);
  maybe_unload_runtime(plan->index_name);
  plan->published = true;
  return true;
}

void index_service::record_standalone_rebuild_tasks(
    const standalone_rebuild_plan &plan) {
  if (!plan.index_name.empty()) {
    m_segment_task_rows[plan.index_name] = plan.segment_tasks;
  }
}

bool index_service::rollback_standalone_rebuild(
    standalone_rebuild_plan *plan) {
  if (plan == nullptr || !plan->prepared || !plan->published) return false;
  if (plan->artifact_published &&
      !plan->rebuilt_backend->rollback_artifact()) {
    return false;
  }
  if (!m_standalone_store.rollback_levelled_compaction(plan->index_name,
                                                       &plan->compaction)) {
    return false;
  }

  m_indexes[plan->index_name] = plan->previous_runtime;
  m_lifecycle_infos[plan->index_name] = plan->previous_lifecycle;
  m_publication_states[plan->index_name] = plan->previous_publication;
  if (plan->had_previous_pipeline) {
    m_build_pipeline_snapshots[plan->index_name] = plan->previous_pipeline;
  } else {
    m_build_pipeline_snapshots.erase(plan->index_name);
  }
  if (plan->had_previous_segment_tasks) {
    m_segment_task_rows[plan->index_name] = plan->previous_segment_tasks;
  } else {
    m_segment_task_rows.erase(plan->index_name);
  }
  if (plan->had_previous_failed_build_diagnostics) {
    m_last_failed_build_diagnostics[plan->index_name] =
        plan->previous_failed_build_diagnostics;
  } else {
    m_last_failed_build_diagnostics.erase(plan->index_name);
  }
  m_standalone_rebuilds.erase(plan->index_name);
  plan->published = false;
  plan->prepared = false;
  plan->artifact_published = false;
  return true;
}

bool index_service::discard_standalone_rebuild(
    standalone_rebuild_plan *plan) {
  if (plan == nullptr || plan->published) return false;
  if (plan->rebuilt_backend != nullptr &&
      plan->rebuilt_backend->has_pending_artifact_publication() &&
      !plan->rebuilt_backend->rollback_artifact()) {
    return false;
  }
  if (plan->has_failed_build_diagnostics && !plan->index_name.empty()) {
    m_last_failed_build_diagnostics[plan->index_name] =
        plan->failed_build_diagnostics;
  }
  m_standalone_store.discard_levelled_compaction(&plan->compaction);
  m_standalone_rebuilds.erase(plan->index_name);
  *plan = standalone_rebuild_plan{};
  return true;
}

bool index_service::finalize_standalone_rebuild(
    standalone_rebuild_plan *plan) {
  if (plan == nullptr || !plan->published) return false;
  if (plan->artifact_published &&
      !plan->rebuilt_backend->finalize_artifact()) {
    return false;
  }
  m_standalone_store.finalize_levelled_compaction(&plan->compaction);
  m_standalone_rebuilds.erase(plan->index_name);
  *plan = standalone_rebuild_plan{};
  return true;
}

bool index_service::rebuild_index(const std::string &index_name,
                                  std::string *error) {
  if (error != nullptr) error->clear();
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(config_it != m_index_configs.end(), index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end())) {
    set_bulk_load_error(error, "vector index not found");
    return false;
  }
  const lifecycle_info lifecycle_before_rebuild = lifecycle_it->second;
  mark_lifecycle_state(&lifecycle_it->second, LIFECYCLE_REBUILDING);

  // Rebuild currently requires no staged writes against this index.
  if (pending_changes_contain_index(m_pending_changes, index_name)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_PENDING_CHANGES);
    set_bulk_load_error(error, "vector index has pending changes");
    return false;
  }

  const index_config build_config =
      detail::with_dynamic_build_options(config_it->second);
  const build_pipeline_decision decision =
      record_build_pipeline_decision(index_name, build_config);
  const bool segmented = should_build_segmented(build_config, decision);
  const uint64_t segment_row_limit =
      m_build_pipeline_snapshots[index_name].effective_segment_row_limit;
  std::vector<vector_index_metadata_store::segment_task_row> segment_tasks;
  standalone_entry_store::raw_segment_compaction compaction;
  std::unique_ptr<backend> rebuilt =
      segmented
          ? build_segmented_backend_from_source(
                index_name, build_config, m_entry_store, &m_standalone_store,
                segment_row_limit, &compaction, &segment_tasks)
          : build_backend_from_config(index_name, build_config);
  if (segmented) {
    m_segment_task_rows[index_name] = segment_tasks;
  } else {
    m_segment_task_rows[index_name].clear();
  }
  if (rebuilt == nullptr) {
    m_standalone_store.discard_levelled_compaction(&compaction);
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_BACKEND_CREATE_FAILED);
    set_bulk_load_error(error, "LOAD VECTOR DATA could not create backend");
    return false;
  }
  if (!segmented &&
      !rebuild_backend_from_source(m_entry_store, m_standalone_store,
                                   index_name, build_config, rebuilt.get())) {
    m_standalone_store.discard_levelled_compaction(&compaction);
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    m_last_failed_build_diagnostics[index_name] =
        rebuilt->build_diagnostics();
    set_bulk_load_rebuild_error(error, rebuilt.get());
    return false;
  }

  DBUG_EXECUTE_IF("vector_segment_task_before_publish", {
    m_standalone_store.discard_levelled_compaction(&compaction);
    lifecycle_it->second = lifecycle_before_rebuild;
    set_bulk_load_error(error,
                        "LOAD VECTOR DATA could not publish rebuilt backend");
    return false;
  });
  if (segmented && !m_standalone_store.publish_levelled_compaction(
                       index_name, &compaction)) {
    m_standalone_store.discard_levelled_compaction(&compaction);
    lifecycle_it->second = lifecycle_before_rebuild;
    set_bulk_load_error(error,
                        "LOAD VECTOR DATA could not publish compacted input");
    return false;
  }
  index_it->second = std::move(rebuilt);
  if (!synchronize_runtime_publication(index_name)) {
    lifecycle_it->second = lifecycle_before_rebuild;
    set_bulk_load_error(error,
                        "vector rebuild publication state is unavailable");
    return false;
  }
  mark_index_ready(index_name, &lifecycle_it->second);
  maybe_unload_runtime(index_name);
  m_standalone_store.finalize_levelled_compaction(&compaction);
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
  if (!synchronize_runtime_publication(index_name)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    return false;
  }
  mark_index_ready(index_name, &lifecycle_it->second);
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
  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_search_ef(search_ef)) return false;
        sync_search_config_from_backend(*runtime, &config_it->second);
        return true;
      });
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
  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_hnsw_build_threads(hnsw_build_threads)) return false;
        sync_hnsw_config_from_backend(*runtime, &config_it->second);
        return true;
      });
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

  if (config_it->second.faiss_nlist != faiss_nlist ||
      config_it->second.faiss_pq_m != 0 ||
      config_it->second.faiss_pq_bits != 0) {
    index_config next_config = config_it->second;
    next_config.faiss_nlist = faiss_nlist;
    next_config.faiss_nprobe = faiss_nprobe;
    next_config.faiss_pq_m = 0;
    next_config.faiss_pq_bits = 0;
    std::unique_ptr<backend> rebuilt =
        detail::build_backend_from_config(index_name, next_config);
    if (rebuilt == nullptr ||
        !rebuild_backend_from_source(m_entry_store, m_standalone_store,
                                     index_name, next_config, rebuilt.get())) {
      return false;
    }
    index_it->second = std::move(rebuilt);
    config_it->second = next_config;
    sync_search_config_from_backend(*index_it->second, &config_it->second);
    sync_hnsw_config_from_backend(*index_it->second, &config_it->second);
    sync_faiss_config_from_backend(*index_it->second, &config_it->second);
    return true;
  }

  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_faiss_ivf_params(faiss_nlist, faiss_nprobe)) {
          return false;
        }
        sync_search_config_from_backend(*runtime, &config_it->second);
        sync_hnsw_config_from_backend(*runtime, &config_it->second);
        sync_faiss_config_from_backend(*runtime, &config_it->second);
        return true;
      });
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

  if (config_it->second.provider != backend_provider::kFaiss) return false;
  if (config_it->second.faiss_nlist != faiss_nlist ||
      config_it->second.faiss_pq_m != faiss_pq_m ||
      config_it->second.faiss_pq_bits != faiss_pq_bits) {
    index_config next_config = config_it->second;
    next_config.faiss_nlist = faiss_nlist;
    next_config.faiss_nprobe = faiss_nprobe;
    next_config.faiss_pq_m = faiss_pq_m;
    next_config.faiss_pq_bits = faiss_pq_bits;
    std::unique_ptr<backend> rebuilt =
        detail::build_backend_from_config(index_name, next_config);
    if (rebuilt == nullptr ||
        !rebuild_backend_from_source(m_entry_store, m_standalone_store,
                                     index_name, next_config, rebuilt.get())) {
      return false;
    }
    index_it->second = std::move(rebuilt);
    config_it->second = next_config;
    sync_search_config_from_backend(*index_it->second, &config_it->second);
    sync_hnsw_config_from_backend(*index_it->second, &config_it->second);
    sync_faiss_config_from_backend(*index_it->second, &config_it->second);
    return true;
  }

  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_faiss_ivf_pq_params(faiss_nlist, faiss_nprobe,
                                              faiss_pq_m, faiss_pq_bits)) {
          return false;
        }
        sync_search_config_from_backend(*runtime, &config_it->second);
        sync_hnsw_config_from_backend(*runtime, &config_it->second);
        sync_faiss_config_from_backend(*runtime, &config_it->second);
        return true;
      });
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

  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_faiss_build_threads(faiss_build_threads))
          return false;
        sync_faiss_config_from_backend(*runtime, &config_it->second);
        return true;
      });
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

  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_diskann_build_params(diskann_max_degree,
                                               diskann_build_complexity,
                                               diskann_build_threads)) {
          return false;
        }
        sync_diskann_build_config_from_backend(*runtime, &config_it->second);
        sync_search_config_from_backend(*runtime, &config_it->second);
        sync_hnsw_config_from_backend(*runtime, &config_it->second);
        return true;
      });
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

  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_diskann_build_threads(diskann_build_threads)) {
          return false;
        }
        sync_diskann_build_config_from_backend(*runtime, &config_it->second);
        return true;
      });
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

  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_diskann_build_mode(diskann_build_mode_value)) {
          return false;
        }
        sync_diskann_build_config_from_backend(*runtime, &config_it->second);
        config_it->second.diskann_build_mode_specified = true;
        return true;
      });
}

bool index_service::set_diskann_search_complexity(
    const std::string &index_name, uint32_t diskann_search_complexity) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_diskann_search_complexity(
                diskann_search_complexity)) {
          return false;
        }
        config_it->second.diskann_search_complexity =
            runtime->diskann_search_complexity();
        return true;
      });
}

bool index_service::set_diskann_search_beamwidth(
    const std::string &index_name, uint32_t diskann_search_beamwidth) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_diskann_search_beamwidth(diskann_search_beamwidth)) {
          return false;
        }
        config_it->second.diskann_search_beamwidth =
            runtime->diskann_search_beamwidth();
        return true;
      });
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
  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_diskann_pq_code_budget_size(
                diskann_pq_code_budget_size)) {
          return false;
        }
        config_it->second.diskann_pq_code_budget_size =
            runtime->diskann_pq_code_budget_size();
        return true;
      });
}

bool index_service::set_diskann_disk_pq_dims(
    const std::string &index_name, uint32_t diskann_disk_pq_dims) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;
  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_diskann_disk_pq_dims(diskann_disk_pq_dims)) {
          return false;
        }
        config_it->second.diskann_disk_pq_dims =
            runtime->diskann_disk_pq_dims();
        return true;
      });
}

bool index_service::set_diskann_accelerate_build(
    const std::string &index_name, bool diskann_accelerate_build) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;
  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_diskann_accelerate_build(diskann_accelerate_build)) {
          return false;
        }
        config_it->second.diskann_accelerate_build =
            runtime->diskann_accelerate_build();
        return true;
      });
}

bool index_service::set_diskann_shuffle_build(
    const std::string &index_name, bool diskann_shuffle_build) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;
  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_diskann_shuffle_build(diskann_shuffle_build)) {
          return false;
        }
        config_it->second.diskann_shuffle_build =
            runtime->diskann_shuffle_build();
        return true;
      });
}

bool index_service::set_diskann_use_bfs_cache(
    const std::string &index_name, bool diskann_use_bfs_cache) {
  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (!all_true(config_it != m_index_configs.end(),
                index_it != m_indexes.end())) {
    return false;
  }
  if (pending_changes_contain_index(m_pending_changes, index_name))
    return false;
  return apply_backend_config_exclusively(
      index_it->second, [&](backend *runtime) {
        if (!runtime->set_diskann_use_bfs_cache(diskann_use_bfs_cache)) {
          return false;
        }
        config_it->second.diskann_use_bfs_cache =
            runtime->diskann_use_bfs_cache();
        return true;
      });
}

bool index_service::rebuild_all_indexes(size_t *rebuilt_count) {
  if (rebuilt_count == nullptr) return false;

  if (has_any_pending_changes(m_pending_changes)) return false;

  std::unordered_map<std::string, std::unique_ptr<backend>> rebuilt_backends;
  rebuilt_backends.reserve(m_index_configs.size());
  for (const auto &entry : m_index_configs) {
    const std::string &index_name = entry.first;
    const index_config build_config =
        detail::with_dynamic_build_options(entry.second);
    if (m_indexes.find(index_name) == m_indexes.end()) return false;
    if (!m_entry_store.has_index(index_name)) return false;
    if (m_lifecycle_infos.find(index_name) == m_lifecycle_infos.end())
      return false;

    const build_pipeline_decision decision =
        record_build_pipeline_decision(index_name, build_config);
    const uint64_t segment_row_limit =
        m_build_pipeline_snapshots[index_name].effective_segment_row_limit;
    std::vector<vector_index_metadata_store::segment_task_row> segment_tasks;
    std::unique_ptr<backend> rebuilt =
        should_build_segmented(build_config, decision)
            ? build_segmented_backend_from_source(
                  index_name, build_config, m_entry_store, &m_standalone_store,
                  segment_row_limit, nullptr, &segment_tasks)
            : build_backend_from_config(index_name, build_config);
    if (should_build_segmented(build_config, decision)) {
      m_segment_task_rows[index_name] = segment_tasks;
    } else {
      m_segment_task_rows[index_name].clear();
    }
    if (rebuilt == nullptr) return false;
    if (!should_build_segmented(build_config, decision) &&
        !rebuild_backend_from_source(m_entry_store, m_standalone_store,
                                     index_name, build_config, rebuilt.get())) {
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
    if (!synchronize_runtime_publication(entry.first)) return false;
    maybe_unload_runtime(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_index_ready(entry.first, &lifecycle_it->second);
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
    if (!synchronize_runtime_publication(entry.first)) return false;
    maybe_unload_runtime(entry.first);
    if (lifecycle_it != m_lifecycle_infos.end() &&
        recover_fallback_indexes.find(entry.first) !=
            recover_fallback_indexes.end()) {
      mark_recover_fallback(&lifecycle_it->second, false);
    }
    if (lifecycle_it != m_lifecycle_infos.end()) {
      mark_index_ready(entry.first, &lifecycle_it->second);
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
    bool *external_manifest_present, uint64_t *external_manifest_generation,
    backend_build_diagnostics *build_diagnostics,
    index_publication_state *publication_state) const {
  if (config == nullptr) {
    return false;
  }

  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  auto publication_it = m_publication_states.find(index_name);
  index_observability_state observability;
  if (!describe_index_observability(index_name, &observability)) {
    return false;
  }
  const bool has_serving_entries = m_entry_store.has_index(index_name);
  if (config_it == m_index_configs.end() || index_it == m_indexes.end() ||
      !has_serving_entries) {
    return false;
  }
  if (lifecycle_it == m_lifecycle_infos.end() ||
      publication_it == m_publication_states.end()) {
    return false;
  }

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
  config->diskann_build_blas_threads = backend->diskann_build_blas_threads();
  config->diskann_build_mode_value =
      detail::effective_diskann_build_mode(config_it->second);
  config->diskann_build_mode_specified =
      config_it->second.diskann_build_mode_specified;
  config->diskann_search_complexity = backend->diskann_search_complexity();
  config->diskann_search_beamwidth = backend->diskann_search_beamwidth();
  config->diskann_pq_code_budget_size =
      backend->diskann_pq_code_budget_size();
  config->diskann_disk_pq_dims = backend->diskann_disk_pq_dims();
  config->diskann_cache_nodes = backend->diskann_cache_nodes();
  config->diskann_accelerate_build = backend->diskann_accelerate_build();
  config->diskann_shuffle_build = backend->diskann_shuffle_build();
  config->diskann_use_bfs_cache = backend->diskann_use_bfs_cache();
  if (supports_mutations != nullptr) {
    *supports_mutations = backend->supports_mutations();
  }
  if (entry_count != nullptr) {
    *entry_count = backend->entry_count();
  }
  if (committed_entry_count != nullptr) {
    *committed_entry_count = observability.authoritative_entry_count;
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
    const auto failed_diagnostics_it =
        m_last_failed_build_diagnostics.find(index_name);
    *build_diagnostics =
        failed_diagnostics_it == m_last_failed_build_diagnostics.end()
            ? backend->build_diagnostics()
            : failed_diagnostics_it->second;
  }
  if (publication_state != nullptr) {
    *publication_state = publication_it->second;
  }
  return true;
}

bool index_service::describe_publication_state(
    const std::string &index_name,
    index_publication_state *publication_state) const {
  if (publication_state == nullptr) return false;
  const auto publication_it = m_publication_states.find(index_name);
  if (publication_it == m_publication_states.end()) return false;
  *publication_state = publication_it->second;
  return true;
}

bool index_service::restore_publication_state(
    const std::string &index_name,
    const index_publication_state &publication_state) {
  if (publication_state.index_identity == 0 ||
      publication_state.index_identity ==
          std::numeric_limits<uint64_t>::max() ||
      publication_state.config_generation == 0 ||
      publication_state.runtime_generation >
          publication_state.truth_generation ||
      publication_state.artifact_generation >
          publication_state.truth_generation) {
    return false;
  }
  auto publication_it = m_publication_states.find(index_name);
  if (publication_it == m_publication_states.end()) return false;
  for (const auto &entry : m_publication_states) {
    if (entry.first != index_name &&
        entry.second.index_identity == publication_state.index_identity) {
      return false;
    }
  }
  publication_it->second = publication_state;
  m_next_index_identity =
      std::max(m_next_index_identity, publication_state.index_identity + 1);
  return true;
}

bool index_service::snapshot_publication_states(
    std::unordered_map<std::string, index_publication_state> *states) const {
  if (states == nullptr) return false;
  *states = m_publication_states;
  return states->size() == m_indexes.size();
}

bool index_service::restore_publication_states(
    const std::unordered_map<std::string, index_publication_state> &states) {
  if (states.size() != m_indexes.size()) return false;
  std::unordered_set<uint64_t> identities;
  uint64_t next_identity = 1;
  for (const auto &entry : states) {
    const index_publication_state &publication = entry.second;
    if (m_indexes.find(entry.first) == m_indexes.end() ||
        publication.index_identity == 0 ||
        publication.index_identity == std::numeric_limits<uint64_t>::max() ||
        publication.config_generation == 0 ||
        publication.runtime_generation > publication.truth_generation ||
        publication.artifact_generation > publication.truth_generation ||
        !identities.insert(publication.index_identity).second) {
      return false;
    }
    next_identity = std::max(next_identity, publication.index_identity + 1);
  }
  m_publication_states = states;
  m_next_index_identity = std::max(m_next_index_identity, next_identity);
  return true;
}

bool index_service::rollback_artifact_publication(
    const std::string &index_name) {
  const auto index_it = m_indexes.find(index_name);
  return index_it != m_indexes.end() && index_it->second != nullptr &&
         index_it->second->rollback_artifact();
}

bool index_service::finalize_artifact_publication(
    const std::string &index_name) {
  const auto index_it = m_indexes.find(index_name);
  return index_it != m_indexes.end() && index_it->second != nullptr &&
         index_it->second->finalize_artifact();
}

bool index_service::recover_artifact_publication(
    const std::string &index_name,
    const diskann_artifact_identity &identity) {
  const auto index_it = m_indexes.find(index_name);
  if (index_it == m_indexes.end() || index_it->second == nullptr ||
      !index_it->second->recover_artifact_publication(identity)) {
    return false;
  }
  return synchronize_runtime_publication(index_name);
}

bool index_service::artifact_publication_matches(
    const std::string &index_name,
    const diskann_artifact_identity &identity) const {
  const auto index_it = m_indexes.find(index_name);
  return index_it != m_indexes.end() && index_it->second != nullptr &&
         index_it->second->artifact_publication_matches(identity);
}

uint64_t index_service::next_index_identity() const {
  return m_next_index_identity;
}

bool index_service::restore_next_index_identity(uint64_t next_index_identity) {
  if (next_index_identity == 0) return false;
  for (const auto &entry : m_publication_states) {
    if (entry.second.index_identity >= next_index_identity) return false;
  }
  m_next_index_identity = next_index_identity;
  return true;
}

bool index_service::describe_index_observability(
    const std::string &index_name, index_observability_state *state) const {
  if (state == nullptr) return false;

  auto config_it = m_index_configs.find(index_name);
  auto index_it = m_indexes.find(index_name);
  if (config_it == m_index_configs.end() || index_it == m_indexes.end()) {
    return false;
  }
  if (!m_entry_store.has_index(index_name) ||
      !m_standalone_store.has_index(index_name)) {
    return false;
  }

  const bool truth_store_enabled =
      config_it->second.consistency_mode == index_consistency_mode::kTransactional;
  state->truth_store_enabled = truth_store_enabled;
  state->authoritative_entry_count =
      truth_store_enabled ? m_entry_store.entry_count(index_name)
                          : m_standalone_store.entry_count(index_name);
  state->standalone_ingest_memory_bytes =
      truth_store_enabled ? 0 : m_standalone_store.memory_bytes(index_name);
  state->standalone_segment_count =
      truth_store_enabled ? 0 : m_standalone_store.segment_count(index_name);
  state->standalone_segment_bytes =
      truth_store_enabled ? 0 : m_standalone_store.segment_bytes(index_name);
  state->standalone_raw_segment_count =
      truth_store_enabled ? 0 : m_standalone_store.raw_segment_count(index_name);
  state->standalone_raw_segment_bytes =
      truth_store_enabled ? 0 : m_standalone_store.raw_segment_bytes(index_name);
  state->build_source =
      truth_store_enabled ? "truth_store"
                          : m_standalone_store.build_source(index_name);
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

bool index_service::restore_committed_state_impl(
    const committed_state &state, bool defer_diskann_artifacts) {
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
    } else if (state_it != state.end()) {
      for (const auto &doc_entry : state_it->second) {
        if (doc_entry.second.size() != backend->dimension()) return false;
      }
      const bool defer_artifact =
          defer_diskann_artifacts &&
          config.provider == backend_provider::kDiskAnn &&
          config.mode == backend_mode::kExternal;
      if (!defer_artifact &&
          !backend->load_committed_entries(state_it->second)) {
        return false;
      }
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
  m_standalone_rebuilds.clear();
  m_segment_task_rows.clear();
  m_last_failed_build_diagnostics.clear();
  for (const auto &entry : m_index_configs) {
    m_segment_task_rows[entry.first] = {};
    maybe_unload_runtime(entry.first);
  }
  return true;
}

bool index_service::restore_committed_state(const committed_state &state) {
  return restore_committed_state_impl(state, false);
}

bool index_service::restore_committed_state_for_startup(
    const committed_state &state) {
  return restore_committed_state_impl(state, true);
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
  config_it->second.diskann_build_blas_threads =
      index_it->second->diskann_build_blas_threads();
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
  config_it->second.diskann_disk_pq_dims =
      index_it->second->diskann_disk_pq_dims();
  config_it->second.diskann_accelerate_build =
      index_it->second->diskann_accelerate_build();
  config_it->second.diskann_shuffle_build =
      index_it->second->diskann_shuffle_build();
  config_it->second.diskann_use_bfs_cache =
      index_it->second->diskann_use_bfs_cache();
  mark_index_ready(index_name, &lifecycle_it->second);
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
  mark_index_ready(index_name, &lifecycle_it->second);
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

    if (options.rebuild_after_load) return rebuild_index(index_name, error);

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
  mark_index_ready(index_name, &lifecycle_it->second);
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

  const build_input_stats raw_stats{
      file_info.row_count,
      file_info.row_count *
          static_cast<uint64_t>(file_info.dimension) * sizeof(float),
      0,
      false};
  const build_pipeline_runtime_config pipeline_config =
      global_build_pipeline_runtime_config();
  const diskann_segment_profile_result segment_profile =
      effective_diskann_segment_profile(config_it->second, raw_stats,
                                        pipeline_config);
  const uint64_t row_limit = build_segment_row_limit(
      static_cast<uint64_t>(file_info.dimension), segment_profile.thresholds);

  if (!m_standalone_store.bulk_upsert_raw_files(
          index_name, vector_filename, docid_filename, file_info.row_count,
          file_info.dimension, row_limit, &seen_doc_ids)) {
    mark_lifecycle_failure(&lifecycle_it->second, ERROR_REPLAY_STATE_FAILED);
    set_bulk_load_error(error, "LOAD VECTOR DATA could not publish raw files");
    return false;
  }
  if (loaded_rows != nullptr) *loaded_rows = row_count;

  if (options.rebuild_after_load) return rebuild_index(index_name, error);

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
    std::unique_ptr<backend> rebuilt_backend,
    const diskann_artifact_identity *artifact_identity) {
  auto index_it = m_indexes.find(index_name);
  auto lifecycle_it = m_lifecycle_infos.find(index_name);
  if (!all_true(index_it != m_indexes.end(),
                m_entry_store.has_index(index_name),
                lifecycle_it != m_lifecycle_infos.end(),
                rebuilt_backend != nullptr)) {
    return false;
  }

  if (rebuilt_backend->has_pending_artifact_publication()) {
    if (artifact_identity == nullptr ||
        !rebuilt_backend->prepare_artifact_publication(*artifact_identity) ||
        !rebuilt_backend->publish_artifact()) {
      (void)rebuilt_backend->rollback_artifact();
      return false;
    }
  }

  index_it->second = std::move(rebuilt_backend);
  if (!m_entry_store.replace_index(index_name, entries) ||
      !synchronize_runtime_publication(index_name)) {
    (void)index_it->second->rollback_artifact();
    return false;
  }
  mark_index_ready(index_name, &lifecycle_it->second);
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
  if (!synchronize_runtime_publication(index_name)) return false;
  if (used_recover_fallback) {
    mark_recover_fallback(&lifecycle_it->second, false);
  }
  mark_index_ready(index_name, &lifecycle_it->second);
  maybe_unload_runtime(index_name);
  return true;
}

}  // namespace vector_index
