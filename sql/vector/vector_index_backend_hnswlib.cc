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

#include "sql/vector/vector_index_backend.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef HAVE_HNSWLIB
#include <hnswlib/hnswalg.h>
#include <hnswlib/space_ip.h>
#include <hnswlib/space_l2.h>
#endif

#include "my_dbug.h"
#include "sql/vector/vector_index_backend_common.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_runtime_thread_pool.h"
#include "sql/vector/vector_load_file.h"

namespace {
using vector_index::detail::check_dimension;
using vector_index::saturated_add_size;
using vector_index::saturated_mul_size;

constexpr size_t k_max_hnsw_build_threads = vector_index::k_max_build_threads;
constexpr size_t k_hnsw_build_chunk_rows = 4096;
constexpr uint32_t k_hnsw_build_memory_multiplier = 4;
constexpr uint64_t k_hnsw_build_fixed_memory = 64ULL * 1024ULL * 1024ULL;

class scoped_thread_budget {
 public:
  scoped_thread_budget(uint32_t *target, uint32_t value)
      : m_target(target), m_previous(target == nullptr ? 0 : *target) {
    if (m_target != nullptr) *m_target = value;
  }

  ~scoped_thread_budget() {
    if (m_target != nullptr) *m_target = m_previous;
  }

 private:
  uint32_t *m_target;
  uint32_t m_previous;
};

size_t read_build_threads(size_t entry_count, uint32_t configured_threads) {
  return vector_index::effective_build_scheduler_threads(entry_count,
                                                         configured_threads);
}

size_t hnsw_raw_block_rows(size_t dimension, size_t total_rows) {
  if (dimension == 0 || total_rows == 0) return 1;

  const size_t row_bytes = saturated_add_size(
      saturated_mul_size(dimension, sizeof(float)), sizeof(uint64_t));
  if (row_bytes == 0 || row_bytes == std::numeric_limits<size_t>::max())
    return 1;

  size_t block_rows = std::min(total_rows, k_hnsw_build_chunk_rows);
  if (opt_vector_entry_cache_size != 0) {
    const size_t cache_bytes = static_cast<size_t>(std::min<ulonglong>(
        opt_vector_entry_cache_size,
        static_cast<ulonglong>(std::numeric_limits<size_t>::max())));
    block_rows =
        std::min(block_rows, std::max<size_t>(1, cache_bytes / row_bytes));
  }
  return std::max<size_t>(1, block_rows);
}

size_t estimate_hnsw_index_memory_bytes(size_t entry_count, size_t dimension,
                                        uint32_t hnsw_m) {
  if (entry_count == 0) return 0;

  const size_t vector_bytes =
      saturated_mul_size(saturated_mul_size(entry_count, dimension),
                         sizeof(float));
  const size_t edge_bytes =
      saturated_mul_size(saturated_mul_size(entry_count, hnsw_m),
                         sizeof(uint32_t) * 2);
  const size_t label_and_overhead_bytes =
      saturated_mul_size(entry_count, sizeof(uint64_t) + 64);
  return saturated_add_size(saturated_add_size(vector_bytes, edge_bytes),
                            label_and_overhead_bytes);
}

bool hnsw_index_memory_budget_allows(size_t entry_count, size_t dimension,
                                     uint32_t hnsw_m) {
  if (opt_vector_hnsw_index_memory_size == 0) return true;
  return estimate_hnsw_index_memory_bytes(entry_count, dimension, hnsw_m) <=
         opt_vector_hnsw_index_memory_size;
}

}  // namespace

namespace vector_index {

class hnswlib_native_state {
 public:
  hnswlib_native_state(size_t dimension, metric_type metric, uint32_t hnsw_m,
                       uint32_t hnsw_ef_construction,
                       size_t initial_capacity = 1024)
      : m_dimension(dimension),
        m_metric(metric),
        m_hnsw_m(hnsw_m),
        m_hnsw_ef_construction(hnsw_ef_construction) {
#ifdef HAVE_HNSWLIB
    initialize_index(std::max<size_t>(1, initial_capacity));
#else
    (void)initial_capacity;
#endif
  }

  bool upsert(uint64_t doc_id, const vector_data &vector) {
#ifdef HAVE_HNSWLIB
    if (!check_dimension(vector, m_dimension)) return false;
    const bool is_new_label = m_labels.find(doc_id) == m_labels.end();
    const size_t target_count = m_labels.size() + (is_new_label ? 1 : 0);
    if (!ensure_capacity(target_count)) return false;
    vector_data stored = normalize(vector);
    std::unordered_set<uint64_t>::iterator inserted_label;
    if (is_new_label) {
      try {
        const auto result = m_labels.insert(doc_id);
        if (!result.second) return false;
        inserted_label = result.first;
      } catch (...) {
        return false;
      }
    }
    DBUG_EXECUTE_IF("vector_backend_fail_hnswlib_upsert_after_label_insert", {
      if (is_new_label) m_labels.erase(inserted_label);
      return false;
    });
    try {
      m_index->addPoint(stored.data(), static_cast<hnswlib::labeltype>(doc_id),
                        is_new_label);
    } catch (...) {
      if (is_new_label) {
        try {
          m_index->markDelete(static_cast<hnswlib::labeltype>(doc_id));
        } catch (...) {
          // addPoint() may have failed before publishing the label.
        }
        m_labels.erase(inserted_label);
      }
      return false;
    }
    return true;
#else
    (void)doc_id;
    (void)vector;
    return false;
#endif
  }

  bool erase(uint64_t doc_id) {
#ifdef HAVE_HNSWLIB
    auto label_it = m_labels.find(doc_id);
    if (label_it == m_labels.end()) return true;
    try {
      m_index->markDelete(static_cast<hnswlib::labeltype>(doc_id));
    } catch (...) {
      return false;
    }
    m_labels.erase(label_it);
    return true;
#else
    (void)doc_id;
    return false;
#endif
  }

  bool search(const vector_data &query, size_t top_k,
              std::vector<search_result> *results) const {
    if (results == nullptr) return false;
    results->clear();
#ifdef HAVE_HNSWLIB
    if (!check_dimension(query, m_dimension) || m_index == nullptr)
      return false;
    return search_with_current_ef(query, top_k, results);
#else
    (void)query;
    (void)top_k;
    return false;
#endif
  }

  bool search_batch(const std::vector<vector_data> &queries, size_t top_k,
                    std::vector<std::vector<search_result>> *results) const {
    if (results == nullptr) return false;
    results->clear();
    results->resize(queries.size());
#ifdef HAVE_HNSWLIB
    if (m_index == nullptr) return false;
    for (const vector_data &query : queries) {
      if (!check_dimension(query, m_dimension)) return false;
    }
    const size_t thread_count =
        vector_index::effective_hnsw_search_threads(queries.size());
    if (thread_count <= 1 || queries.size() <= 1) {
      for (size_t i = 0; i < queries.size(); ++i) {
        if (!search_with_current_ef(queries[i], top_k, &(*results)[i]))
          return false;
      }
      return true;
    }

    return vector_index::parallel_for_queries(
        queries.size(), thread_count,
        [&](size_t begin, size_t end, size_t) {
          for (size_t i = begin; i < end; ++i) {
            if (!search_with_current_ef(queries[i], top_k, &(*results)[i])) {
              return false;
            }
          }
          return true;
        });
#else
    (void)queries;
    (void)top_k;
    return false;
#endif
  }

  size_t entry_count() const { return m_labels.size(); }

  bool contains(uint64_t doc_id) const {
    return m_labels.find(doc_id) != m_labels.end();
  }

  bool build_from_entries(
      const std::unordered_map<uint64_t, vector_data> &entries,
      size_t thread_count) {
#ifdef HAVE_HNSWLIB
    if (m_index == nullptr) return false;
    std::vector<std::pair<uint64_t, vector_data>> prepared;
    try {
      prepared.reserve(entries.size());
      for (const auto &entry : entries) {
        if (!check_dimension(entry.second, m_dimension)) return false;
        prepared.emplace_back(entry.first, normalize(entry.second));
      }
    } catch (...) {
      return false;
    }

    if (!ensure_capacity(prepared.size())) return false;
    if (prepared.empty()) {
      m_labels.clear();
      return true;
    }

    m_labels.clear();
    std::vector<uint64_t> prepared_labels;
    if (!prepare_label_block(
            prepared.size(),
            [&prepared](size_t index) { return prepared[index].first; },
            &prepared_labels)) {
      return false;
    }
    if (!add_prepared_entries(prepared, thread_count)) {
      rollback_label_block(prepared_labels);
      return false;
    }
    return true;
#else
    (void)entries;
    (void)thread_count;
    return false;
#endif
  }

  bool build_from_reader(const committed_entry_reader &reader,
                         uint32_t configured_threads, size_t *entry_count) {
#ifdef HAVE_HNSWLIB
    if (!reader || entry_count == nullptr || m_index == nullptr) return false;

    std::vector<std::pair<uint64_t, vector_data>> prepared;
    size_t count = 0;
    m_labels.clear();
    try {
      prepared.reserve(k_hnsw_build_chunk_rows);
    } catch (...) {
      return false;
    }

    auto flush_prepared = [&]() {
      if (prepared.empty()) return true;
      if (prepared.size() > std::numeric_limits<size_t>::max() - count)
        return false;
      const size_t target_count = count + prepared.size();
      if (!hnsw_index_memory_budget_allows(target_count, m_dimension, m_hnsw_m))
        return false;
      if (!ensure_capacity(target_count)) return false;
      std::vector<uint64_t> prepared_labels;
      if (!prepare_label_block(
              prepared.size(),
              [&prepared](size_t index) { return prepared[index].first; },
              &prepared_labels)) {
        return false;
      }
      const size_t effective_threads =
          read_build_threads(prepared.size(), configured_threads);
      if (!add_prepared_entries(prepared, effective_threads)) {
        rollback_label_block(prepared_labels);
        return false;
      }
      count = target_count;
      prepared.clear();
      return true;
    };

    const bool read_ok = reader([&](uint64_t doc_id, const vector_data &vector) {
      if (!check_dimension(vector, m_dimension)) return false;
      try {
        prepared.emplace_back(doc_id, normalize(vector));
      } catch (...) {
        return false;
      }
      if (prepared.size() < k_hnsw_build_chunk_rows) return true;
      return flush_prepared();
    });
    if (!read_ok || !flush_prepared()) return false;

    *entry_count = count;
    return true;
#else
    (void)reader;
    (void)configured_threads;
    (void)entry_count;
    return false;
#endif
  }

  bool build_from_raw_segments(const std::vector<raw_vector_segment> &segments,
                               size_t block_rows, uint32_t configured_threads,
                               size_t *entry_count) {
#ifdef HAVE_HNSWLIB
    if (entry_count == nullptr || m_index == nullptr || block_rows == 0)
      return false;

    size_t count = 0;
    m_labels.clear();
    size_t expected_rows = 0;
    for (const raw_vector_segment &segment : segments) {
      if (segment.row_count >
          std::numeric_limits<size_t>::max() - expected_rows) {
        return false;
      }
      expected_rows += segment.row_count;
    }
    try {
      m_labels.reserve(expected_rows);
    } catch (...) {
      return false;
    }
    for (const raw_vector_segment &segment : segments) {
      vector_load_file_info info;
      std::string error;
      const bool read_ok = read_fbin_vector_blocks(
          segment.vector_path, segment.docid_path, m_dimension, block_rows,
          &info, &error,
          [&](const uint64_t *doc_ids, const float *values, size_t row_count,
              size_t dimension) {
            if (dimension != m_dimension ||
                row_count > std::numeric_limits<size_t>::max() - count) {
              return false;
            }
            const size_t target_count = count + row_count;
            if (!hnsw_index_memory_budget_allows(target_count, m_dimension,
                                                 m_hnsw_m) ||
                !ensure_capacity(target_count) ||
                !add_raw_entries(
                    doc_ids, values, row_count,
                    read_build_threads(row_count, configured_threads))) {
              return false;
            }
            count = target_count;
            return true;
          });
      if (!read_ok || info.row_count != segment.row_count ||
          info.dimension != segment.dimension) {
        return false;
      }
    }

    *entry_count = count;
    return true;
#else
    (void)segments;
    (void)block_rows;
    (void)configured_threads;
    (void)entry_count;
    return false;
#endif
  }

  bool available() const {
#ifdef HAVE_HNSWLIB
    return m_index != nullptr;
#else
    return false;
#endif
  }

  bool set_search_ef(uint32_t search_ef) {
#ifdef HAVE_HNSWLIB
    if (search_ef == 0 || m_index == nullptr) return false;
    m_search_ef = search_ef;
    m_index->setEf(search_ef);
    return true;
#else
    (void)search_ef;
    return false;
#endif
  }

  uint32_t search_ef() const { return m_search_ef; }

 private:
#ifdef HAVE_HNSWLIB
  bool add_prepared_entries(
      const std::vector<std::pair<uint64_t, vector_data>> &prepared,
      size_t thread_count) {
    if (prepared.empty()) return true;
    thread_count = std::max<size_t>(1, std::min(thread_count, prepared.size()));
    DBUG_EXECUTE_IF("vector_backend_fail_hnswlib_parallel_rebuild_worker",
                    return false;);
    if (thread_count <= 1 || prepared.size() <= 1) {
      for (const auto &entry : prepared) {
        try {
          m_index->addPoint(entry.second.data(),
                            static_cast<hnswlib::labeltype>(entry.first));
        } catch (...) {
          return false;
        }
      }
      return true;
    }

    return vector_index::parallel_for_ranges(
        prepared.size(), thread_count,
        [&](size_t begin, size_t end, size_t) {
        for (size_t i = begin; i < end; ++i) {
          DBUG_EXECUTE_IF("vector_backend_fail_hnswlib_parallel_rebuild_worker", {
            return false;
          });
          try {
            m_index->addPoint(
                prepared[i].second.data(),
                static_cast<hnswlib::labeltype>(prepared[i].first));
          } catch (...) {
            return false;
          }
        }
        return true;
      });
  }

  bool add_raw_entries(const uint64_t *doc_ids, const float *values,
                       size_t row_count, size_t thread_count) {
    if (doc_ids == nullptr || values == nullptr || row_count == 0) return false;

    std::vector<float> normalized_values;
    const float *build_values = values;
    if (m_metric == metric_type::kCosine) {
      try {
        normalized_values.assign(values, values + row_count * m_dimension);
      } catch (...) {
        return false;
      }
      for (size_t row = 0; row < row_count; ++row) {
        float *row_values = normalized_values.data() + row * m_dimension;
        double norm2 = 0.0;
        for (size_t column = 0; column < m_dimension; ++column) {
          norm2 += static_cast<double>(row_values[column]) * row_values[column];
        }
        if (norm2 <= 0.0) continue;
        const double inv_norm = 1.0 / std::sqrt(norm2);
        for (size_t column = 0; column < m_dimension; ++column) {
          row_values[column] =
              static_cast<float>(row_values[column] * inv_norm);
        }
      }
      build_values = normalized_values.data();
    }

    std::vector<uint64_t> prepared_labels;
    if (!prepare_label_block(
            row_count, [doc_ids](size_t row) { return doc_ids[row]; },
            &prepared_labels)) {
      return false;
    }

    thread_count = std::max<size_t>(1, std::min(thread_count, row_count));
    DBUG_EXECUTE_IF("vector_backend_fail_hnswlib_parallel_rebuild_worker", {
      rollback_label_block(prepared_labels);
      return false;
    });
    const auto add_range = [&](size_t begin, size_t end, size_t) {
      for (size_t row = begin; row < end; ++row) {
        DBUG_EXECUTE_IF("vector_backend_fail_hnswlib_parallel_rebuild_worker",
                        { return false; });
        try {
          m_index->addPoint(build_values + row * m_dimension,
                            static_cast<hnswlib::labeltype>(doc_ids[row]));
        } catch (...) {
          return false;
        }
      }
      return true;
    };
    if (thread_count <= 1 || row_count <= 1) {
      if (!add_range(0, row_count, 0)) {
        rollback_label_block(prepared_labels);
        return false;
      }
    } else if (!vector_index::parallel_for_ranges(row_count, thread_count,
                                                  add_range)) {
      rollback_label_block(prepared_labels);
      return false;
    }
    return true;
  }

  template <typename doc_id_reader>
  bool prepare_label_block(size_t row_count, doc_id_reader read_doc_id,
                           std::vector<uint64_t> *prepared_labels) {
    if (prepared_labels == nullptr ||
        row_count > std::numeric_limits<size_t>::max() - m_labels.size()) {
      return false;
    }
    DBUG_EXECUTE_IF("vector_backend_fail_hnswlib_label_preparation",
                    return false;);
    prepared_labels->clear();
    try {
      prepared_labels->reserve(row_count);
      m_labels.reserve(m_labels.size() + row_count);
      for (size_t row = 0; row < row_count; ++row) {
        const uint64_t doc_id = read_doc_id(row);
        if (!m_labels.insert(doc_id).second) {
          rollback_label_block(*prepared_labels);
          return false;
        }
        prepared_labels->push_back(doc_id);
      }
    } catch (...) {
      rollback_label_block(*prepared_labels);
      return false;
    }
    return true;
  }

  void rollback_label_block(const std::vector<uint64_t> &prepared_labels) {
    for (uint64_t doc_id : prepared_labels) m_labels.erase(doc_id);
  }

  bool search_with_current_ef(const vector_data &query, size_t top_k,
                              std::vector<search_result> *results) const {
    results->clear();
    if (top_k == 0) return true;
    vector_data normalized;
    const float *query_data = query.data();
    if (m_metric == metric_type::kCosine) {
      normalized = normalize(query);
      query_data = normalized.data();
    }
    std::priority_queue<std::pair<float, hnswlib::labeltype>> raw;
    try {
      raw = m_index->searchKnn(query_data, top_k);
    } catch (...) {
      return false;
    }
    const size_t result_count = raw.size();
    results->resize(result_count);
    /* searchKnn() returns the farthest result first; fill backwards to keep the
       public result order identical to searchKnnCloserFirst(). */
    for (size_t slot = result_count; slot > 0; --slot) {
      const auto entry = raw.top();
      raw.pop();
      double distance = static_cast<double>(entry.first);
      if (m_metric == metric_type::kEuclidean) {
        distance = std::sqrt(std::max(0.0, distance));
      } else if (m_metric == metric_type::kInnerProduct) {
        distance = distance - 1.0;
      }
      (*results)[slot - 1] = {static_cast<uint64_t>(entry.second), distance};
    }
    return true;
  }
#endif
#ifdef HAVE_HNSWLIB
  bool initialize_index(size_t capacity) {
    DBUG_EXECUTE_IF("vector_backend_force_hnswlib_native_unavailable",
                    return false;);
    try {
      if (m_metric == metric_type::kEuclidean) {
        m_l2_space =
            std::make_unique<hnswlib::L2Space>(static_cast<size_t>(m_dimension));
        m_ip_space.reset();
        m_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            m_l2_space.get(), capacity, m_hnsw_m, m_hnsw_ef_construction, 100,
            true);
      } else {
        m_ip_space = std::make_unique<hnswlib::InnerProductSpace>(
            static_cast<size_t>(m_dimension));
        m_l2_space.reset();
        m_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            m_ip_space.get(), capacity, m_hnsw_m, m_hnsw_ef_construction, 100,
            true);
      }
      m_index->setEf(m_search_ef);
      m_capacity = capacity;
      return true;
    } catch (...) {
      m_index.reset();
      m_l2_space.reset();
      m_ip_space.reset();
      return false;
    }
  }

  bool ensure_capacity(size_t target_count) {
    if (m_index == nullptr) return false;
    if (target_count <= m_capacity) return true;
    size_t next_capacity = std::max(target_count, m_capacity * 2);
    try {
      m_index->resizeIndex(next_capacity);
      m_capacity = next_capacity;
      return true;
    } catch (...) {
      return false;
    }
  }
#endif

  vector_data normalize(const vector_data &vector) const {
    if (m_metric != metric_type::kCosine) return vector;
    double norm2 = 0.0;
    for (float value : vector) norm2 += value * value;
    if (norm2 <= 0.0) return vector;
    const double inv_norm = 1.0 / std::sqrt(norm2);
    vector_data normalized(vector);
    for (float &value : normalized) {
      value = static_cast<float>(static_cast<double>(value) * inv_norm);
    }
    return normalized;
  }

  size_t m_dimension{0};
  metric_type m_metric{metric_type::kEuclidean};
  uint32_t m_search_ef{vector_index::k_default_hnsw_search_ef};
  uint32_t m_hnsw_m{vector_index::k_default_hnsw_m};
  uint32_t m_hnsw_ef_construction{
      vector_index::k_default_hnsw_ef_construction};
  std::unordered_set<uint64_t> m_labels;
#ifdef HAVE_HNSWLIB
  std::unique_ptr<hnswlib::SpaceInterface<float>> m_l2_space;
  std::unique_ptr<hnswlib::SpaceInterface<float>> m_ip_space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> m_index;
  size_t m_capacity{0};
#endif
};

hnswlib_backend::hnswlib_backend(size_t dimension, metric_type metric, backend_mode mode)
    : m_dimension(dimension),
      m_metric(metric),
      m_mode(mode),
      m_native_state(mode == backend_mode::kMemory
                         ? std::make_unique<hnswlib_native_state>(
                               dimension, metric,
                               vector_index::k_default_hnsw_m,
                               vector_index::k_default_hnsw_ef_construction)
                         : nullptr) {}

hnswlib_backend::~hnswlib_backend() = default;

bool hnswlib_backend::native_available() const {
  return m_native_state != nullptr && m_native_state->available();
}

memory_backend *hnswlib_backend::ensure_exact_fallback() {
  if (m_exact_fallback == nullptr) {
    m_exact_fallback = std::make_unique<memory_backend>(m_dimension, m_metric);
  }
  return m_exact_fallback.get();
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool hnswlib_backend::hnsw_native_available_for_testing() const {
  return native_available();
}

bool hnswlib_backend::hnsw_exact_fallback_active_for_testing() const {
  return m_exact_fallback != nullptr;
}

size_t hnswlib_backend::hnsw_exact_fallback_entry_count_for_testing() const {
  return m_exact_fallback == nullptr ? 0 : m_exact_fallback->entry_count();
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

void hnswlib_backend::record_build_diagnostics(const char *input_source,
                                               size_t row_count,
                                               size_t effective_threads,
                                               size_t segment_count,
                                               size_t reader_passes) {
  m_last_build_diagnostics = {};
  m_last_build_diagnostics.runtime =
      native_available() ? "hnsw_scheduler" : "hnsw_exact_fallback";
  m_last_build_diagnostics.input_source =
      input_source == nullptr ? "" : input_source;
  m_last_build_diagnostics.row_count = row_count;
  m_last_build_diagnostics.segment_count = segment_count;
  m_last_build_diagnostics.build_invocations = 1;
  m_last_build_diagnostics.concurrent_build_tasks = 1;
  m_last_build_diagnostics.scheduler_cpu_budget =
      static_cast<uint32_t>(std::max<size_t>(1, effective_threads));
  m_last_build_diagnostics.effective_build_threads =
      static_cast<uint32_t>(std::max<size_t>(1, effective_threads));
  m_last_build_diagnostics.effective_blas_threads = 1;
  m_last_build_diagnostics.reader_passes = reader_passes;
}

void hnswlib_backend::clear_build_diagnostics() {
  m_last_build_diagnostics = {};
}

bool hnswlib_backend::upsert(uint64_t doc_id, const vector_data &vector) {
  if (m_mode != backend_mode::kMemory) return false;
  if (!check_dimension(vector, m_dimension)) return false;

  if (!native_available()) {
    memory_backend *fallback = ensure_exact_fallback();
    const bool replacing_existing = fallback->contains(doc_id);
    const size_t target_count =
        replacing_existing ? fallback->entry_count() : fallback->entry_count() + 1;
    if (!hnsw_index_memory_budget_allows(target_count, m_dimension, m_hnsw_m))
      return false;
    return fallback->upsert(doc_id, vector);
  }

  const bool replacing_existing = m_native_state->contains(doc_id);
  const size_t target_count = replacing_existing ? m_native_state->entry_count()
                                                : m_native_state->entry_count() + 1;
  if (!hnsw_index_memory_budget_allows(target_count, m_dimension, m_hnsw_m))
    return false;
  return m_native_state->upsert(doc_id, vector);
}

bool hnswlib_backend::rebuild_native_from_entries(
    const std::unordered_map<uint64_t, vector_data> &entries) {
  if (m_mode != backend_mode::kMemory) return false;
  if (!hnsw_index_memory_budget_allows(entries.size(), m_dimension, m_hnsw_m))
    return false;
  auto rebuilt = std::make_unique<hnswlib_native_state>(
      m_dimension, m_metric, m_hnsw_m, m_hnsw_ef_construction,
      std::max<size_t>(1, entries.size()));
  if (!rebuilt->available()) return false;
  if (!rebuilt->build_from_entries(
          entries, read_build_threads(entries.size(), m_hnsw_build_threads)))
    return false;
  const uint32_t current_search_ef = search_ef();
  if (current_search_ef != 0 && !rebuilt->set_search_ef(current_search_ef))
    return false;
  m_native_state = std::move(rebuilt);
  m_exact_fallback.reset();
  return true;
}

bool hnswlib_backend::build_memory_fallback_from_entries(
    const std::unordered_map<uint64_t, vector_data> &entries,
    memory_backend *fallback) const {
  if (fallback == nullptr) return false;
  for (const auto &entry : entries) {
    if (!check_dimension(entry.second, m_dimension)) return false;
  }
  fallback->reset();
  for (const auto &entry : entries) {
    if (!fallback->upsert(entry.first, entry.second)) return false;
  }
  return true;
}

bool hnswlib_backend::rebuild_memory_fallback_from_entries(
    const std::unordered_map<uint64_t, vector_data> &entries) {
  auto rebuilt_fallback = std::make_unique<memory_backend>(m_dimension, m_metric);
  if (!build_memory_fallback_from_entries(entries, rebuilt_fallback.get()))
    return false;
  m_exact_fallback = std::move(rebuilt_fallback);
  return true;
}

bool hnswlib_backend::build_state_from_reader(
    const committed_entry_reader &reader,
    std::unique_ptr<memory_backend> *fallback,
    std::unique_ptr<hnswlib_native_state> *native, size_t *entry_count) const {
  if (!reader || fallback == nullptr || native == nullptr ||
      entry_count == nullptr) {
    return false;
  }

  std::unique_ptr<memory_backend> rebuilt_fallback;
  std::unique_ptr<hnswlib_native_state> rebuilt_native;
  if (native_available()) {
    rebuilt_native = std::make_unique<hnswlib_native_state>(
        m_dimension, m_metric, m_hnsw_m, m_hnsw_ef_construction);
    if (!rebuilt_native->available()) return false;
  } else {
    rebuilt_fallback = std::make_unique<memory_backend>(m_dimension, m_metric);
  }

  size_t count = 0;
  if (rebuilt_native != nullptr) {
    if (!rebuilt_native->build_from_reader(reader, m_hnsw_build_threads, &count))
      return false;
  } else {
    const bool read_ok = reader([&](uint64_t doc_id, const vector_data &vector) {
      if (!check_dimension(vector, m_dimension)) return false;
      if (!hnsw_index_memory_budget_allows(count + 1, m_dimension, m_hnsw_m))
        return false;
      if (rebuilt_fallback != nullptr &&
          !rebuilt_fallback->upsert(doc_id, vector))
        return false;
      ++count;
      return true;
    });
    if (!read_ok) return false;
  }

  if (rebuilt_native != nullptr) {
    const uint32_t current_search_ef = search_ef();
    if (current_search_ef != 0 &&
        !rebuilt_native->set_search_ef(current_search_ef))
      return false;
  }

  *fallback = std::move(rebuilt_fallback);
  *native = std::move(rebuilt_native);
  *entry_count = count;
  return true;
}

bool hnswlib_backend::build_state_from_raw_segments(
    const std::vector<raw_vector_segment> &segments, size_t total_rows,
    std::unique_ptr<hnswlib_native_state> *native, size_t *entry_count) const {
  if (native == nullptr || entry_count == nullptr || !native_available() ||
      !hnsw_index_memory_budget_allows(total_rows, m_dimension, m_hnsw_m)) {
    return false;
  }

  auto rebuilt_native = std::make_unique<hnswlib_native_state>(
      m_dimension, m_metric, m_hnsw_m, m_hnsw_ef_construction,
      std::max<size_t>(1, total_rows));
  if (!rebuilt_native->available()) return false;

  size_t count = 0;
  if (!rebuilt_native->build_from_raw_segments(
          segments, hnsw_raw_block_rows(m_dimension, total_rows),
          m_hnsw_build_threads, &count) ||
      count != total_rows) {
    return false;
  }
  const uint32_t current_search_ef = search_ef();
  if (current_search_ef != 0 &&
      !rebuilt_native->set_search_ef(current_search_ef)) {
    return false;
  }

  *native = std::move(rebuilt_native);
  *entry_count = count;
  return true;
}

bool hnswlib_backend::erase(uint64_t doc_id) {
  if (m_mode != backend_mode::kMemory) return false;
  if (!native_available()) {
    memory_backend *fallback = ensure_exact_fallback();
    return fallback->erase(doc_id);
  }

  return m_native_state->erase(doc_id);
}

bool hnswlib_backend::search(const vector_data &query, size_t top_k,
                            std::vector<search_result> *results) const {
  if (results == nullptr) return false;
  if (m_mode != backend_mode::kMemory) {
    results->clear();
    return false;
  }
  if (native_available()) return m_native_state->search(query, top_k, results);
  if (m_exact_fallback != nullptr)
    return m_exact_fallback->search(query, top_k, results);
  results->clear();
  return false;
}

bool hnswlib_backend::search_batch(
    const std::vector<vector_data> &queries, size_t top_k,
    std::vector<std::vector<search_result>> *results) const {
  if (results == nullptr) return false;
  if (m_mode != backend_mode::kMemory) {
    results->clear();
    return false;
  }
  if (native_available())
    return m_native_state->search_batch(queries, top_k, results);
  if (m_exact_fallback != nullptr)
    return backend::search_batch(queries, top_k, results);
  results->clear();
  return false;
}

size_t hnswlib_backend::entry_count() const {
  if (m_mode != backend_mode::kMemory) return 0;
  if (native_available()) return m_native_state->entry_count();
  return m_exact_fallback == nullptr ? 0 : m_exact_fallback->entry_count();
}

std::string hnswlib_backend::backend_variant() const { return "hnsw"; }

bool hnswlib_backend::set_search_ef(uint32_t search_ef) {
  if (m_mode != backend_mode::kMemory || m_native_state == nullptr ||
      !m_native_state->available())
    return false;
  return m_native_state->set_search_ef(search_ef);
}

uint32_t hnswlib_backend::search_ef() const {
  if (m_mode != backend_mode::kMemory || m_native_state == nullptr ||
      !m_native_state->available())
    return 0;
  return m_native_state->search_ef();
}

bool hnswlib_backend::set_hnsw_build_params(uint32_t hnsw_m,
                                        uint32_t hnsw_ef_construction) {
  if (m_mode != backend_mode::kMemory || hnsw_m == 0 ||
      hnsw_ef_construction == 0) {
    return false;
  }
  const size_t current_count = entry_count();
  if (!hnsw_index_memory_budget_allows(current_count, m_dimension, hnsw_m))
    return false;
  if (!native_available()) return false;
  if (current_count != 0) return false;

  auto rebuilt = std::make_unique<hnswlib_native_state>(
      m_dimension, m_metric, hnsw_m, hnsw_ef_construction,
      std::max<size_t>(1, current_count));
  if (!rebuilt->available()) return false;
  if (search_ef() != 0 && !rebuilt->set_search_ef(search_ef())) return false;
  m_hnsw_m = hnsw_m;
  m_hnsw_ef_construction = hnsw_ef_construction;
  m_native_state = std::move(rebuilt);
  m_exact_fallback.reset();
  return true;
}

uint32_t hnswlib_backend::hnsw_m() const {
  return m_mode == backend_mode::kMemory ? m_hnsw_m : 0;
}

uint32_t hnswlib_backend::hnsw_ef_construction() const {
  return m_mode == backend_mode::kMemory ? m_hnsw_ef_construction : 0;
}

bool hnswlib_backend::set_hnsw_build_threads(uint32_t hnsw_build_threads) {
  if (m_mode != backend_mode::kMemory ||
      hnsw_build_threads > k_max_hnsw_build_threads)
    return false;
  m_hnsw_build_threads = hnsw_build_threads;
  return true;
}

uint32_t hnswlib_backend::hnsw_build_threads() const {
  return m_mode == backend_mode::kMemory ? m_hnsw_build_threads : 0;
}

bool hnswlib_backend::rebuild_from_committed_entries(
    const std::unordered_map<uint64_t, vector_data> &entries) {
  if (m_mode != backend_mode::kMemory) {
    clear_build_diagnostics();
    return entries.empty();
  }
  const uint32_t requested_threads = static_cast<uint32_t>(
      read_build_threads(std::max<size_t>(1, entries.size()),
                         m_hnsw_build_threads));
  build_resource_lease resource_lease;
  if (!acquire_build_resources(entries.size(), k_hnsw_build_memory_multiplier,
                               k_hnsw_build_fixed_memory, 0,
                               requested_threads, &resource_lease)) {
    clear_build_diagnostics();
    return false;
  }
  scoped_thread_budget thread_budget(&m_hnsw_build_threads,
                                     resource_lease.cpu_slots());
  if (!native_available()) {
    if (!rebuild_memory_fallback_from_entries(entries)) return false;
    record_build_diagnostics("entries", entries.size(), 1);
    backend::record_build_resource_diagnostics(resource_lease,
                                               &m_last_build_diagnostics);
    return true;
  }
  if (!rebuild_native_from_entries(entries)) return false;
  record_build_diagnostics("entries", entries.size(),
                           read_build_threads(entries.size(),
                                              m_hnsw_build_threads));
  backend::record_build_resource_diagnostics(resource_lease,
                                             &m_last_build_diagnostics);
  return true;
}

bool hnswlib_backend::rebuild_from_committed_entries_from_reader(
    const committed_entry_reader &reader) {
  return rebuild_from_committed_entry_source({reader, 0, false});
}

bool hnswlib_backend::load_committed_entries_from_source(
    const committed_entry_source &source) {
  return rebuild_from_committed_entry_source(source);
}

bool hnswlib_backend::recover_committed_entries_from_source(
    const committed_entry_source &source) {
  if (!recover()) return false;
  return rebuild_from_committed_entry_source(source);
}

bool hnswlib_backend::rebuild_from_committed_entry_source(
    const committed_entry_source &source) {
  if (m_mode != backend_mode::kMemory || !source.reader) {
    clear_build_diagnostics();
    return false;
  }
  const uint32_t requested_threads = static_cast<uint32_t>(
      source.has_exact_row_count
          ? read_build_threads(std::max<size_t>(1, source.exact_row_count),
                               m_hnsw_build_threads)
          : effective_build_scheduler_thread_budget(m_hnsw_build_threads));
  build_resource_lease resource_lease;
  if (!acquire_build_resources(
          source.has_exact_row_count ? source.exact_row_count : 0,
          k_hnsw_build_memory_multiplier, k_hnsw_build_fixed_memory, 0,
          requested_threads, &resource_lease)) {
    clear_build_diagnostics();
    return false;
  }
  scoped_thread_budget thread_budget(&m_hnsw_build_threads,
                                     resource_lease.cpu_slots());
  std::unique_ptr<memory_backend> rebuilt_fallback;
  std::unique_ptr<hnswlib_native_state> rebuilt_native;
  size_t entry_count = 0;
  if (!build_state_from_reader(source.reader, &rebuilt_fallback,
                               &rebuilt_native, &entry_count) ||
      (source.has_exact_row_count &&
       entry_count != source.exact_row_count)) {
    return false;
  }
  const bool used_native = rebuilt_native != nullptr;
  m_exact_fallback = std::move(rebuilt_fallback);
  if (rebuilt_native != nullptr) m_native_state = std::move(rebuilt_native);
  record_build_diagnostics(
      "reader", entry_count,
      used_native ? read_build_threads(entry_count, m_hnsw_build_threads) : 1);
  backend::record_build_resource_diagnostics(resource_lease,
                                             &m_last_build_diagnostics);
  return true;
}

bool hnswlib_backend::rebuild_from_raw_segments(
    const raw_vector_segment_reader &reader) {
  if (m_mode != backend_mode::kMemory || !reader) {
    clear_build_diagnostics();
    return false;
  }
  if (!native_available()) return backend::rebuild_from_raw_segments(reader);

  std::vector<raw_vector_segment> segments;
  size_t total_rows = 0;
  if (!reader([this, &segments,
               &total_rows](const raw_vector_segment &segment) {
        if (segment.dimension != m_dimension ||
            segment.row_count > std::numeric_limits<size_t>::max() - total_rows)
          return false;
        vector_load_file_info info;
        std::string error;
        if (!read_fbin_file_info(segment.vector_path, m_dimension, &info,
                                 &error) ||
            info.row_count != segment.row_count ||
            info.dimension != segment.dimension) {
          return false;
        }
        total_rows += segment.row_count;
        segments.push_back(segment);
        return true;
      })) {
    return false;
  }

  const uint32_t requested_threads = static_cast<uint32_t>(
      read_build_threads(std::max<size_t>(1, total_rows),
                         m_hnsw_build_threads));
  build_resource_lease resource_lease;
  if (!acquire_build_resources(total_rows, k_hnsw_build_memory_multiplier,
                               k_hnsw_build_fixed_memory, 0,
                               requested_threads, &resource_lease)) {
    clear_build_diagnostics();
    return false;
  }
  scoped_thread_budget thread_budget(&m_hnsw_build_threads,
                                     resource_lease.cpu_slots());

  std::unique_ptr<hnswlib_native_state> rebuilt_native;
  size_t entry_count = 0;
  if (!build_state_from_raw_segments(segments, total_rows, &rebuilt_native,
                                     &entry_count)) {
    return false;
  }

  m_native_state = std::move(rebuilt_native);
  m_exact_fallback.reset();
  record_build_diagnostics(
      "raw_blocks", entry_count,
      read_build_threads(entry_count, m_hnsw_build_threads), segments.size(),
      segments.empty() ? 0 : 1);
  backend::record_build_resource_diagnostics(resource_lease,
                                             &m_last_build_diagnostics);
  return true;
}

backend_build_diagnostics hnswlib_backend::build_diagnostics() const {
  return m_last_build_diagnostics;
}

}  // namespace vector_index
