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
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef HAVE_HNSWLIB
#include <hnswlib/hnswalg.h>
#include <hnswlib/space_ip.h>
#include <hnswlib/space_l2.h>
#endif

#include "my_dbug.h"
#include "sql/vector/vector_env.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_backend_common.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_index_runtime_thread_pool.h"

namespace {
using vector_index::detail::check_dimension;

constexpr size_t k_default_batch_search_threads = 4;
constexpr size_t k_max_batch_search_threads = 32;
constexpr size_t k_default_build_threads = 4;
constexpr size_t k_max_hnsw_build_threads = vector_index::k_max_build_threads;

#ifdef HAVE_HNSWLIB
size_t read_batch_search_threads(size_t query_count) {
  return vector_env::limit_thread_count(k_default_batch_search_threads,
                                        k_max_batch_search_threads,
                                        query_count);
}
#endif

size_t read_build_threads(size_t entry_count, uint32_t configured_threads) {
  if (entry_count <= 1) return entry_count;
  if (configured_threads != 0) {
    return vector_env::limit_thread_count(
        configured_threads, k_max_hnsw_build_threads, entry_count);
  }
  const unsigned int hardware_threads = std::thread::hardware_concurrency();
  const size_t configured =
      hardware_threads > 0 ? hardware_threads : k_default_build_threads;
  return vector_env::limit_thread_count(configured, k_max_hnsw_build_threads,
                                        entry_count);
}

size_t saturated_mul(size_t left, size_t right) {
  if (left == 0 || right == 0) return 0;
  if (left > std::numeric_limits<size_t>::max() / right)
    return std::numeric_limits<size_t>::max();
  return left * right;
}

size_t saturated_add(size_t left, size_t right) {
  if (left > std::numeric_limits<size_t>::max() - right)
    return std::numeric_limits<size_t>::max();
  return left + right;
}

size_t estimate_hnsw_index_memory_bytes(size_t entry_count, size_t dimension,
                                        uint32_t hnsw_m) {
  if (entry_count == 0) return 0;

  const size_t vector_bytes =
      saturated_mul(saturated_mul(entry_count, dimension), sizeof(float));
  const size_t edge_bytes =
      saturated_mul(saturated_mul(entry_count, hnsw_m), sizeof(uint32_t) * 2);
  const size_t label_and_overhead_bytes =
      saturated_mul(entry_count, sizeof(uint64_t) + 64);
  return saturated_add(saturated_add(vector_bytes, edge_bytes),
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
    if (!ensure_capacity(m_entries.size() + 1)) return false;
    vector_data stored = normalize(vector);
    try {
      m_index->addPoint(stored.data(), static_cast<hnswlib::labeltype>(doc_id));
    } catch (...) {
      return false;
    }
    m_entries[doc_id] = std::move(stored);
    return true;
#else
    (void)doc_id;
    (void)vector;
    return false;
#endif
  }

  bool erase(uint64_t doc_id) {
#ifdef HAVE_HNSWLIB
    if (m_entries.erase(doc_id) == 0) return true;
    try {
      m_index->markDelete(static_cast<hnswlib::labeltype>(doc_id));
    } catch (...) {
      return false;
    }
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
    const size_t thread_count = read_batch_search_threads(queries.size());
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

  size_t entry_count() const { return m_entries.size(); }

  bool build_from_entries(
      const std::unordered_map<uint64_t, vector_data> &entries,
      size_t thread_count) {
#ifdef HAVE_HNSWLIB
    if (m_index == nullptr) return false;
    std::vector<std::pair<uint64_t, vector_data>> prepared;
    prepared.reserve(entries.size());
    for (const auto &entry : entries) {
      if (!check_dimension(entry.second, m_dimension)) return false;
      prepared.emplace_back(entry.first, normalize(entry.second));
    }

    if (!ensure_capacity(prepared.size())) return false;
    if (prepared.empty()) {
      m_entries.clear();
      return true;
    }

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
    } else {
      std::vector<std::thread> workers;
      std::vector<unsigned char> worker_ok(thread_count, 1);
      const size_t chunk_size =
          (prepared.size() + thread_count - 1) / thread_count;
      workers.reserve(thread_count);
      for (size_t worker_id = 0; worker_id < thread_count; ++worker_id) {
        const size_t begin = worker_id * chunk_size;
        const size_t end = std::min(prepared.size(), begin + chunk_size);
        if (begin >= end) break;
        workers.emplace_back([&, begin, end, worker_id]() {
          for (size_t i = begin; i < end; ++i) {
            DBUG_EXECUTE_IF(
                "vector_backend_fail_hnswlib_parallel_rebuild_worker", {
                  worker_ok[worker_id] = 0;
                  return;
                });
            try {
              m_index->addPoint(
                  prepared[i].second.data(),
                  static_cast<hnswlib::labeltype>(prepared[i].first));
            } catch (...) {
              worker_ok[worker_id] = 0;
              return;
            }
          }
        });
      }
      for (std::thread &worker : workers) worker.join();
      if (!std::all_of(worker_ok.begin(), worker_ok.end(),
                       [](unsigned char ok) { return ok != 0; }))
        return false;
    }

    m_entries.clear();
    m_entries.reserve(prepared.size());
    for (auto &entry : prepared) {
      m_entries.emplace(entry.first, std::move(entry.second));
    }
    return true;
#else
    (void)entries;
    (void)thread_count;
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
  uint32_t m_search_ef{64};
  uint32_t m_hnsw_m{16};
  uint32_t m_hnsw_ef_construction{200};
  std::unordered_map<uint64_t, vector_data> m_entries;
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
                         ? std::make_unique<hnswlib_native_state>(dimension, metric,
                                                                16, 200)
                         : nullptr),
      m_memory_fallback(dimension, metric) {}

hnswlib_backend::~hnswlib_backend() = default;

bool hnswlib_backend::upsert(uint64_t doc_id, const vector_data &vector) {
  if (m_mode != backend_mode::kMemory) return false;
  if (!check_dimension(vector, m_dimension)) return false;
  const bool replacing_existing = m_entries.find(doc_id) != m_entries.end();
  const size_t target_count =
      replacing_existing ? m_entries.size() : m_entries.size() + 1;
  if (!hnsw_index_memory_budget_allows(target_count, m_dimension, m_hnsw_m))
    return false;
  if (m_native_state == nullptr || !m_native_state->available()) {
    if (!m_memory_fallback.upsert(doc_id, vector)) return false;
    m_entries[doc_id] = vector;
    return true;
  }

  if (!replacing_existing && m_native_state->upsert(doc_id, vector)) {
    if (!m_memory_fallback.upsert(doc_id, vector)) return false;
    m_entries[doc_id] = vector;
    return true;
  }

  auto rebuilt_entries = m_entries;
  rebuilt_entries[doc_id] = vector;
  if (!rebuild_native_from_entries(rebuilt_entries) ||
      !rebuild_memory_fallback_from_entries(rebuilt_entries)) {
    return false;
  }
  m_entries = std::move(rebuilt_entries);
  return true;
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
  memory_backend rebuilt_fallback(m_dimension, m_metric);
  if (!build_memory_fallback_from_entries(entries, &rebuilt_fallback))
    return false;
  m_memory_fallback = std::move(rebuilt_fallback);
  return true;
}

bool hnswlib_backend::erase(uint64_t doc_id) {
  if (m_mode != backend_mode::kMemory) return false;
  if (m_native_state == nullptr || !m_native_state->available()) {
    if (!m_memory_fallback.erase(doc_id)) return false;
    m_entries.erase(doc_id);
    return true;
  }

  auto rebuilt_entries = m_entries;
  rebuilt_entries.erase(doc_id);
  if (!m_native_state->erase(doc_id) &&
      !rebuild_native_from_entries(rebuilt_entries)) {
    return false;
  }
  if (!rebuild_memory_fallback_from_entries(rebuilt_entries)) return false;
  m_entries = std::move(rebuilt_entries);
  return true;
}

/*
  Keep the exact fallback and native graph in lock-step. If hnswlib rejects a
  replacement label, rebuild the graph from the authoritative entry map instead
  of accepting the write only in the fallback path.
*/
bool hnswlib_backend::search(const vector_data &query, size_t top_k,
                            std::vector<search_result> *results) const {
  if (results == nullptr) return false;
  if (m_mode != backend_mode::kMemory) {
    results->clear();
    return false;
  }
  if (m_native_state != nullptr && m_native_state->available() &&
      m_native_state->search(query, top_k, results))
    return true;
  return m_memory_fallback.search(query, top_k, results);
}

bool hnswlib_backend::search_batch(
    const std::vector<vector_data> &queries, size_t top_k,
    std::vector<std::vector<search_result>> *results) const {
  if (results == nullptr) return false;
  if (m_mode != backend_mode::kMemory) {
    results->clear();
    return false;
  }
  if (m_native_state != nullptr && m_native_state->available() &&
      m_native_state->search_batch(queries, top_k, results))
    return true;
  return backend::search_batch(queries, top_k, results);
}

size_t hnswlib_backend::entry_count() const {
  if (m_mode != backend_mode::kMemory) return 0;
  return m_entries.size();
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
  if (!hnsw_index_memory_budget_allows(m_entries.size(), m_dimension, hnsw_m))
    return false;
  auto rebuilt = std::make_unique<hnswlib_native_state>(
      m_dimension, m_metric, hnsw_m, hnsw_ef_construction,
      std::max<size_t>(1, m_entries.size()));
  if (!rebuilt->available()) return false;
  if (!rebuilt->build_from_entries(
          m_entries, read_build_threads(m_entries.size(), m_hnsw_build_threads)))
    return false;
  if (search_ef() != 0 && !rebuilt->set_search_ef(search_ef())) return false;
  m_hnsw_m = hnsw_m;
  m_hnsw_ef_construction = hnsw_ef_construction;
  m_native_state = std::move(rebuilt);
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
  if (m_mode != backend_mode::kMemory) return entries.empty();
  memory_backend rebuilt_fallback(m_dimension, m_metric);
  if (!build_memory_fallback_from_entries(entries, &rebuilt_fallback))
    return false;
  if (m_native_state == nullptr || !m_native_state->available()) {
    m_memory_fallback = std::move(rebuilt_fallback);
    m_entries = entries;
    return true;
  }
  if (!rebuild_native_from_entries(entries)) return false;
  m_memory_fallback = std::move(rebuilt_fallback);
  m_entries = entries;
  return true;
}

}  // namespace vector_index
