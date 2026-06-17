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
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef HAVE_FAISS
#ifdef _OPENMP
#include <omp.h>
#endif
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIDMap.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>
#endif

#include "my_dbug.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_backend_common.h"
#include "sql/vector/vector_index_backend_internal.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_status.h"

namespace {

using vector_index::detail::check_dimension;
using vector_index::detail::compute_distance;
using vector_index::detail::decode_hex_bytes;
using vector_index::detail::encode_hex_bytes;
using vector_index::detail::ensure_parent_directory;
using vector_index::detail::external_manifest_header;
using vector_index::detail::external_manifest_path;
using vector_index::detail::external_snapshot_header;
using vector_index::detail::external_snapshot_prefix;
using vector_index::detail::faiss_external_snapshot_directory;
using vector_index::detail::kFaissExternalSnapshotSuffix;
using vector_index::detail::load_external_manifest_generation_from_file;
using vector_index::detail::parse_uint64;
using vector_index::detail::quarantine_file_if_exists;
using vector_index::detail::quarantine_path_for;
using vector_index::detail::remove_dir_if_empty;
using vector_index::detail::remove_generated_snapshots_with_prefix;
using vector_index::detail::remove_if_exists;
using vector_index::detail::save_external_manifest_generation_to_file;

constexpr size_t k_faiss_rebuild_add_batch_size = 16384;

size_t faiss_training_sample_count(size_t total_rows, size_t dimension,
                                   ulonglong train_size,
                                   size_t min_required_rows) {
  if (train_size == 0 || dimension == 0 || total_rows <= min_required_rows) {
    return total_rows;
  }

  const ulonglong row_bytes =
      static_cast<ulonglong>(dimension) * sizeof(float);
  if (row_bytes == 0) return total_rows;

  size_t sample_rows = static_cast<size_t>(train_size / row_bytes);
  sample_rows = std::max(sample_rows, min_required_rows);
  return std::min(sample_rows, total_rows);
}

#ifdef HAVE_FAISS
class faiss_build_thread_scope {
 public:
  explicit faiss_build_thread_scope(uint32_t build_threads) {
#ifdef _OPENMP
    if (build_threads == 0) return;
    m_previous_threads = omp_get_max_threads();
    omp_set_num_threads(static_cast<int>(build_threads));
    m_active = true;
#else
    (void)build_threads;
#endif
  }

  ~faiss_build_thread_scope() {
#ifdef _OPENMP
    if (m_active) omp_set_num_threads(m_previous_threads);
#endif
  }

  faiss_build_thread_scope(const faiss_build_thread_scope &) = delete;
  faiss_build_thread_scope &operator=(const faiss_build_thread_scope &) =
      delete;

 private:
#ifdef _OPENMP
  int m_previous_threads{0};
  bool m_active{false};
#endif
};
#endif

#ifdef HAVE_FAISS
double faiss_distance_to_public(vector_index::metric_type metric,
                                float distance) {
  if (metric == vector_index::metric_type::kEuclidean) {
    return std::sqrt(static_cast<double>(distance));
  }
  if (metric == vector_index::metric_type::kInnerProduct) {
    return -static_cast<double>(distance);
  }
  return 1.0 - static_cast<double>(distance);
}
#endif

}  // namespace

namespace vector_index {

faiss_backend::faiss_backend(size_t dimension, metric_type metric, backend_mode mode,
                           const std::string &index_name,
                           external_sidecar_profile sidecar_profile)
    : m_dimension(dimension),
      m_metric(metric),
      m_mode(mode),
      m_sidecar_profile(sidecar_profile),
      m_index_name(index_name),
      m_external_snapshot_directory(
          faiss_external_snapshot_directory(index_name)),
      m_external_manifest_path(
          external_manifest_path(index_name, sidecar_profile)),
      m_memory_fallback(dimension, metric),
      m_external_fallback(dimension, metric) {
#ifdef HAVE_FAISS
  (void)initialize_faiss_index();
#endif
}

faiss_backend::~faiss_backend() = default;

vector_data faiss_backend::normalize_for_faiss(const vector_data &vector) const {
  if (m_metric != metric_type::kCosine) return vector;

  double norm = 0.0;
  for (float value : vector) norm += static_cast<double>(value) * value;
  if (norm <= 0.0) return vector_data();

  const float inv_norm = static_cast<float>(1.0 / std::sqrt(norm));
  vector_data normalized(vector);
  for (float &value : normalized) value *= inv_norm;
  return normalized;
}

bool faiss_backend::initialize_faiss_index(bool use_ivfpq) {
#ifdef HAVE_FAISS
  if (m_faiss_index) return true;
  std::unique_ptr<faiss::Index> base_index;
  if (m_mode == backend_mode::kExternal) {
    const faiss::MetricType metric_type =
        m_metric == metric_type::kEuclidean ? faiss::METRIC_L2
                                       : faiss::METRIC_INNER_PRODUCT;
    if (m_faiss_nlist != 0) {
      const uint32_t effective_nlist = std::max<uint32_t>(1, m_faiss_nlist);
      std::unique_ptr<faiss::Index> quantizer;
      if (m_metric == metric_type::kEuclidean) {
        quantizer = std::make_unique<faiss::IndexFlatL2>(
            static_cast<faiss::idx_t>(m_dimension));
      } else {
        quantizer = std::make_unique<faiss::IndexFlatIP>(
            static_cast<faiss::idx_t>(m_dimension));
      }
      if (use_ivfpq && m_faiss_pq_m != 0 && m_faiss_pq_bits != 0) {
        auto ivfpq_index = std::make_unique<faiss::IndexIVFPQ>(
            quantizer.get(), static_cast<faiss::idx_t>(m_dimension),
            static_cast<faiss::idx_t>(effective_nlist),
            static_cast<size_t>(m_faiss_pq_m),
            static_cast<size_t>(m_faiss_pq_bits), metric_type);
        ivfpq_index->own_fields = true;
        (void)quantizer.release();
        ivfpq_index->nprobe = std::max<faiss::idx_t>(
            1, static_cast<faiss::idx_t>(m_faiss_nprobe));
        base_index = std::move(ivfpq_index);
      } else {
        auto ivf_index = std::make_unique<faiss::IndexIVFFlat>(
            quantizer.get(), static_cast<faiss::idx_t>(m_dimension),
            static_cast<faiss::idx_t>(effective_nlist), metric_type);
        ivf_index->own_fields = true;
        (void)quantizer.release();
        ivf_index->nprobe = std::max<faiss::idx_t>(
            1, static_cast<faiss::idx_t>(m_faiss_nprobe));
        base_index = std::move(ivf_index);
      }
    } else {
      auto hnsw_index = std::make_unique<faiss::IndexHNSWFlat>(
          static_cast<int>(m_dimension), static_cast<int>(m_hnsw_m),
          metric_type);
      hnsw_index->hnsw.efSearch = m_search_ef;
      hnsw_index->hnsw.efConstruction =
          static_cast<int>(m_hnsw_ef_construction);
      base_index = std::move(hnsw_index);
    }
  } else if (m_metric == metric_type::kEuclidean) {
    base_index = std::make_unique<faiss::IndexFlatL2>(
        static_cast<faiss::idx_t>(m_dimension));
  } else {
    base_index = std::make_unique<faiss::IndexFlatIP>(
        static_cast<faiss::idx_t>(m_dimension));
  }
  auto id_map = std::make_unique<faiss::IndexIDMap2>(base_index.get());
  id_map->own_fields = true;
  (void)base_index.release();
  m_faiss_index = std::move(id_map);
  m_faiss_entry_count = 0;
  return true;
#else
  (void)use_ivfpq;
  return false;
#endif
}

bool faiss_backend::rebuild_external_faiss_index(
    const std::unordered_map<uint64_t, vector_data> &entries) {
#ifdef HAVE_FAISS
  DBUG_EXECUTE_IF("vector_backend_fail_faiss_external_rebuild", return false;);
  faiss_build_thread_scope thread_scope(m_faiss_build_threads);
  m_faiss_index.reset();
  const bool pq_requested = m_faiss_pq_m != 0 && m_faiss_pq_bits != 0;
  const bool pq_trainable =
      !pq_requested ||
      entries.size() >=
          (static_cast<size_t>(1)
           << std::min<uint32_t>(m_faiss_pq_bits, 20U));
  if (!initialize_faiss_index(pq_trainable)) return false;
  m_faiss_last_training_count = 0;
  if (m_faiss_nlist != 0 && !entries.empty()) {
    const size_t total_rows = entries.size();
    const size_t min_training_rows =
        std::max<size_t>(1, std::min<size_t>(m_faiss_nlist, total_rows));
    const size_t sample_rows =
        faiss_training_sample_count(total_rows, m_dimension,
                                    opt_vector_faiss_train_size,
                                    min_training_rows);
    const faiss::idx_t training_count =
        static_cast<faiss::idx_t>(sample_rows);
    const faiss::idx_t effective_nlist =
        std::max<faiss::idx_t>(1, std::min<faiss::idx_t>(
                                      static_cast<faiss::idx_t>(m_faiss_nlist),
                                      training_count));
    std::vector<float> training_data;
    training_data.reserve(sample_rows * m_dimension);
    std::vector<uint64_t> training_doc_ids;
    training_doc_ids.reserve(total_rows);
    for (const auto &entry : entries) {
      training_doc_ids.push_back(entry.first);
    }
    std::sort(training_doc_ids.begin(), training_doc_ids.end());
    for (size_t sample_index = 0; sample_index < sample_rows; ++sample_index) {
      const size_t position = (sample_index * total_rows) / sample_rows;
      const auto entry = entries.find(training_doc_ids[position]);
      if (entry == entries.end()) return false;
      vector_data prepared = normalize_for_faiss(entry->second);
      if (!check_dimension(prepared, m_dimension) || prepared.empty())
        return false;
      training_data.insert(training_data.end(), prepared.begin(),
                           prepared.end());
    }
    m_faiss_last_training_count = sample_rows;
    auto *id_map = dynamic_cast<faiss::IndexIDMap2 *>(m_faiss_index.get());
    if (id_map == nullptr) return false;
    auto *ivf = dynamic_cast<faiss::IndexIVF *>(id_map->index);
    if (ivf == nullptr) return false;
    try {
      ivf->nlist = effective_nlist;
      ivf->quantizer->reset();
      ivf->train(training_count, training_data.data());
      ivf->nprobe =
          std::max<faiss::idx_t>(1, static_cast<faiss::idx_t>(m_faiss_nprobe));
    } catch (...) {
      if (pq_requested && pq_trainable) return false;
      return false;
    }
  }
  if (entries.empty()) {
    m_faiss_entry_count = 0;
    return true;
  }
  std::vector<faiss::idx_t> batch_ids;
  std::vector<float> batch_data;
  batch_ids.reserve(std::min(entries.size(), k_faiss_rebuild_add_batch_size));
  batch_data.reserve(std::min(entries.size(), k_faiss_rebuild_add_batch_size) *
                     m_dimension);

  const auto flush_batch = [this, &batch_ids, &batch_data]() -> bool {
    if (batch_ids.empty()) return true;
    try {
      m_faiss_index->add_with_ids(static_cast<faiss::idx_t>(batch_ids.size()),
                                  batch_data.data(), batch_ids.data());
    } catch (...) {
      return false;
    }
    batch_ids.clear();
    batch_data.clear();
    return true;
  };

  for (const auto &entry : entries) {
    const faiss::idx_t id = static_cast<faiss::idx_t>(entry.first);
    vector_data prepared = normalize_for_faiss(entry.second);
    if (!check_dimension(prepared, m_dimension) || prepared.empty())
      return false;
    batch_ids.push_back(id);
    batch_data.insert(batch_data.end(), prepared.begin(), prepared.end());
    if (batch_ids.size() == k_faiss_rebuild_add_batch_size && !flush_batch())
      return false;
  }
  if (!flush_batch()) return false;
  m_faiss_entry_count = static_cast<size_t>(m_faiss_index->ntotal);
  return true;
#else
  (void)entries;
  return false;
#endif
}

bool faiss_backend::apply_faiss_mutation(uint64_t doc_id, const vector_data *vector) {
#ifdef HAVE_FAISS
  if (!initialize_faiss_index()) return false;
  const faiss::idx_t id = static_cast<faiss::idx_t>(doc_id);
  faiss::IDSelectorBatch selector(1, &id);
  m_faiss_index->remove_ids(selector);
  m_faiss_entry_count = static_cast<size_t>(m_faiss_index->ntotal);
  if (vector == nullptr) return true;

  vector_data prepared = normalize_for_faiss(*vector);
  if (prepared.empty()) return false;
  m_faiss_index->add_with_ids(1, prepared.data(), &id);
  m_faiss_entry_count = static_cast<size_t>(m_faiss_index->ntotal);
  return true;
#else
  (void)doc_id;
  (void)vector;
  return false;
#endif
}

bool faiss_backend::search_with_faiss(const vector_data &query, size_t top_k,
                                   std::vector<search_result> *results) const {
#ifdef HAVE_FAISS
  if (!m_faiss_index) return false;
  results->clear();
  if (top_k == 0) return true;
  if (!check_dimension(query, m_dimension)) return false;

  vector_data prepared = normalize_for_faiss(query);
  if (prepared.empty()) return false;

  std::vector<faiss::idx_t> labels(top_k, -1);
  std::vector<float> distances(top_k, 0.0f);
  m_faiss_index->search(1, prepared.data(),
                        static_cast<faiss::idx_t>(top_k), distances.data(),
                        labels.data());
  for (size_t i = 0; i < top_k; ++i) {
    if (labels[i] < 0) continue;
    const double distance = faiss_distance_to_public(m_metric, distances[i]);
    results->push_back({static_cast<uint64_t>(labels[i]), distance});
  }
  std::sort(results->begin(), results->end(),
            [](const search_result &lhs, const search_result &rhs) {
              if (lhs.distance != rhs.distance)
                return lhs.distance < rhs.distance;
              return lhs.doc_id < rhs.doc_id;
            });
  return true;
#else
  (void)query;
  (void)top_k;
  (void)results;
  return false;
#endif
}

bool faiss_backend::search_batch_with_faiss(
    const std::vector<vector_data> &queries, size_t top_k,
    std::vector<std::vector<search_result>> *results) const {
#ifdef HAVE_FAISS
  if (results == nullptr) return false;
  results->clear();
  results->resize(queries.size());
  if (!m_faiss_index) return false;

  if (queries.empty()) return true;
  for (const vector_data &query : queries) {
    if (!check_dimension(query, m_dimension)) return false;
  }
  if (top_k == 0) return true;

  std::vector<float> prepared_queries;
  prepared_queries.reserve(queries.size() * m_dimension);
  for (const vector_data &query : queries) {
    vector_data prepared = normalize_for_faiss(query);
    if (prepared.empty()) return false;
    prepared_queries.insert(prepared_queries.end(), prepared.begin(),
                            prepared.end());
  }

  std::vector<faiss::idx_t> labels(queries.size() * top_k, -1);
  std::vector<float> distances(queries.size() * top_k, 0.0f);
  m_faiss_index->search(static_cast<faiss::idx_t>(queries.size()),
                        prepared_queries.data(),
                        static_cast<faiss::idx_t>(top_k), distances.data(),
                        labels.data());

  for (size_t row = 0; row < queries.size(); ++row) {
    std::vector<search_result> &row_results = (*results)[row];
    row_results.reserve(top_k);
    for (size_t i = 0; i < top_k; ++i) {
      const size_t offset = row * top_k + i;
      if (labels[offset] < 0) continue;
      const double distance =
          faiss_distance_to_public(m_metric, distances[offset]);
      row_results.push_back({static_cast<uint64_t>(labels[offset]), distance});
    }
  }

  return true;
#else
  (void)queries;
  (void)top_k;
  (void)results;
  return false;
#endif
}

bool faiss_backend::search_external_snapshot_entries(
    const vector_data &query, size_t top_k,
    std::vector<search_result> *results) const {
  results->clear();
  if (top_k == 0) return true;
  if (!check_dimension(query, m_dimension)) return false;

  for (const auto &entry : m_external_snapshot_entries) {
    double distance = 0.0;
    if (!compute_distance(m_metric, query, entry.second, &distance))
      return false;
    results->push_back(search_result{entry.first, distance});
  }

  const size_t count = std::min(top_k, results->size());
  std::partial_sort(
      results->begin(), results->begin() + count, results->end(),
      [](const search_result &lhs, const search_result &rhs) {
        if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
        return lhs.doc_id < rhs.doc_id;
      });
  results->resize(count);
  return true;
}

bool faiss_backend::external_manifest_present() const {
  uint64_t generation = 0;
  bool exists = false;
  if (!load_external_manifest_generation(&generation, &exists)) return false;
  return exists;
}

uint64_t faiss_backend::external_manifest_generation() const {
  uint64_t generation = 0;
  bool exists = false;
  if (!load_external_manifest_generation(&generation, &exists) || !exists)
    return 0;
  return generation;
}

std::string faiss_backend::backend_variant() const {
  if (m_mode != backend_mode::kExternal) return "flat";
  if (m_faiss_pq_m != 0 && m_faiss_pq_bits != 0) return "ivf_pq";
  return m_faiss_nlist != 0 ? "ivf_flat" : "hnsw";
}

bool faiss_backend::set_search_ef(uint32_t search_ef) {
#ifdef HAVE_FAISS
  if (m_mode != backend_mode::kExternal || search_ef == 0) return false;
  m_search_ef = search_ef;
  if (m_faiss_index == nullptr) return true;
  auto *id_map = dynamic_cast<faiss::IndexIDMap2 *>(m_faiss_index.get());
  if (id_map == nullptr) return false;
  if (auto *hnsw = dynamic_cast<faiss::IndexHNSW *>(id_map->index);
      hnsw != nullptr) {
    hnsw->hnsw.efSearch = search_ef;
  }
  return true;
#else
  (void)search_ef;
  return false;
#endif
}

uint32_t faiss_backend::search_ef() const {
  return m_mode == backend_mode::kExternal ? m_search_ef : 0;
}

bool faiss_backend::set_hnsw_build_params(uint32_t hnsw_m,
                                      uint32_t hnsw_ef_construction) {
#ifdef HAVE_FAISS
  if (m_mode != backend_mode::kExternal || hnsw_m == 0 ||
      hnsw_ef_construction == 0) {
    return false;
  }
  m_hnsw_m = hnsw_m;
  m_hnsw_ef_construction = hnsw_ef_construction;
  return rebuild_external_faiss_index(m_external_snapshot_entries);
#else
  (void)hnsw_m;
  (void)hnsw_ef_construction;
  return false;
#endif
}

uint32_t faiss_backend::hnsw_m() const {
  return m_mode == backend_mode::kExternal ? m_hnsw_m : 0;
}

uint32_t faiss_backend::hnsw_ef_construction() const {
  return m_mode == backend_mode::kExternal ? m_hnsw_ef_construction : 0;
}

bool faiss_backend::set_faiss_ivf_params(uint32_t faiss_nlist,
                                     uint32_t faiss_nprobe) {
#ifdef HAVE_FAISS
  if (m_mode != backend_mode::kExternal) return false;
  if ((faiss_nlist == 0) != (faiss_nprobe == 0)) return false;
  m_faiss_nlist = faiss_nlist;
  m_faiss_nprobe = faiss_nprobe;
  m_faiss_pq_m = 0;
  m_faiss_pq_bits = 0;
  return rebuild_external_faiss_index(m_external_snapshot_entries);
#else
  (void)faiss_nlist;
  (void)faiss_nprobe;
  return false;
#endif
}

uint32_t faiss_backend::faiss_nlist() const {
  return m_mode == backend_mode::kExternal ? m_faiss_nlist : 0;
}

uint32_t faiss_backend::faiss_nprobe() const {
  return m_mode == backend_mode::kExternal ? m_faiss_nprobe : 0;
}

bool faiss_backend::set_faiss_build_threads(uint32_t faiss_build_threads) {
  if (faiss_build_threads > k_max_build_threads) return false;
  m_faiss_build_threads = faiss_build_threads;
  return true;
}

uint32_t faiss_backend::faiss_build_threads() const {
  return m_faiss_build_threads;
}

bool faiss_backend::set_faiss_ivf_pq_params(uint32_t faiss_nlist,
                                       uint32_t faiss_nprobe,
                                       uint32_t faiss_pq_m,
                                       uint32_t faiss_pq_bits) {
#ifdef HAVE_FAISS
  if (m_mode != backend_mode::kExternal || faiss_nlist == 0 ||
      faiss_nprobe == 0 || faiss_pq_m == 0 || faiss_pq_bits == 0) {
    return false;
  }
  m_faiss_nlist = faiss_nlist;
  m_faiss_nprobe = faiss_nprobe;
  m_faiss_pq_m = faiss_pq_m;
  m_faiss_pq_bits = faiss_pq_bits;
  return rebuild_external_faiss_index(m_external_snapshot_entries);
#else
  (void)faiss_nlist;
  (void)faiss_nprobe;
  (void)faiss_pq_m;
  (void)faiss_pq_bits;
  return false;
#endif
}

uint32_t faiss_backend::faiss_pq_m() const {
  return m_mode == backend_mode::kExternal ? m_faiss_pq_m : 0;
}

uint32_t faiss_backend::faiss_pq_bits() const {
  return m_mode == backend_mode::kExternal ? m_faiss_pq_bits : 0;
}

bool faiss_backend::persist_faiss_index_file(const std::string &path) {
#ifdef HAVE_FAISS
  DBUG_EXECUTE_IF("vector_backend_fail_persist_faiss_index_file",
                  return false;);
  if (!m_faiss_index) return false;
  const std::string temp_path = path + ".tmp";
  try {
    faiss::write_index(m_faiss_index.get(), temp_path.c_str());
  } catch (...) {
    std::remove(temp_path.c_str());
    return false;
  }
  DBUG_EXECUTE_IF("vector_backend_fail_persist_faiss_index_rename", {
    std::remove(temp_path.c_str());
    return false;
  };);
  if (std::rename(temp_path.c_str(), path.c_str()) != 0) {
    std::remove(temp_path.c_str());
    return false;
  }
  return true;
#else
  (void)path;
  return false;
#endif
}

bool faiss_backend::recover_faiss_index_file(const std::string &path) {
#ifdef HAVE_FAISS
  std::unique_ptr<faiss::Index> index;
  try {
    index.reset(faiss::read_index(path.c_str()));
  } catch (...) {
    return false;
  }
  if (!index) return false;
  if (index->d != static_cast<faiss::idx_t>(m_dimension)) return false;
  auto *id_map = dynamic_cast<faiss::IndexIDMap2 *>(index.get());
  if (id_map == nullptr) return false;
  if (m_mode == backend_mode::kExternal) {
    if (auto *hnsw = dynamic_cast<faiss::IndexHNSW *>(id_map->index);
        hnsw != nullptr) {
      hnsw->hnsw.efSearch = m_search_ef;
      m_faiss_nlist = 0;
      m_faiss_nprobe = 0;
      m_faiss_pq_m = 0;
      m_faiss_pq_bits = 0;
    } else if (auto *ivf = dynamic_cast<faiss::IndexIVF *>(id_map->index);
               ivf != nullptr) {
      m_faiss_nlist = static_cast<uint32_t>(ivf->nlist);
      ivf->nprobe =
          std::max<faiss::idx_t>(1, static_cast<faiss::idx_t>(m_faiss_nprobe));
      if (auto *ivfpq = dynamic_cast<faiss::IndexIVFPQ *>(id_map->index);
          ivfpq != nullptr) {
        m_faiss_pq_m = static_cast<uint32_t>(ivfpq->pq.M);
        m_faiss_pq_bits = static_cast<uint32_t>(ivfpq->pq.nbits);
      } else {
        m_faiss_pq_m = 0;
        m_faiss_pq_bits = 0;
      }
    } else {
      return false;
    }
  }
  std::unordered_map<uint64_t, vector_data> recovered_entries;
  recovered_entries.reserve(id_map->id_map.size());
  for (faiss::idx_t mapped_id : id_map->id_map) {
    if (mapped_id < 0) continue;
    vector_data vector(m_dimension, 0.0F);
    try {
      id_map->reconstruct(mapped_id, vector.data());
    } catch (...) {
      return false;
    }
    recovered_entries.emplace(static_cast<uint64_t>(mapped_id),
                              std::move(vector));
  }
  m_faiss_entry_count = static_cast<size_t>(index->ntotal);
  m_faiss_index = std::move(index);
  m_external_snapshot_entries = std::move(recovered_entries);
  return true;
#else
  (void)path;
  return false;
#endif
}

bool faiss_backend::upsert(uint64_t doc_id, const vector_data &vector) {
  if (m_mode == backend_mode::kMemory) {
#ifdef HAVE_FAISS
    if (!check_dimension(vector, m_dimension)) return false;
    return apply_faiss_mutation(doc_id, &vector);
#else
    return m_memory_fallback.upsert(doc_id, vector);
#endif
  }
  const auto before_entries = m_external_snapshot_entries;
  m_external_snapshot_entries[doc_id] = vector;
#ifdef HAVE_FAISS
  if (!rebuild_external_faiss_index(m_external_snapshot_entries)) {
    m_external_snapshot_entries = before_entries;
    (void)rebuild_external_faiss_index(before_entries);
    return false;
  }
#endif
  if (persist_external_snapshot()) return true;

  m_external_snapshot_entries = before_entries;
  (void)rebuild_external_faiss_index(before_entries);
  return false;
}

bool faiss_backend::erase(uint64_t doc_id) {
  if (m_mode == backend_mode::kMemory) {
#ifdef HAVE_FAISS
    return apply_faiss_mutation(doc_id, nullptr);
#else
    return m_memory_fallback.erase(doc_id);
#endif
  }
  const auto before_entries = m_external_snapshot_entries;
  m_external_snapshot_entries.erase(doc_id);
#ifdef HAVE_FAISS
  if (!rebuild_external_faiss_index(m_external_snapshot_entries)) {
    m_external_snapshot_entries = before_entries;
    (void)rebuild_external_faiss_index(before_entries);
    return false;
  }
#endif
  if (persist_external_snapshot()) return true;

  m_external_snapshot_entries = before_entries;
  (void)rebuild_external_faiss_index(before_entries);
  return false;
}

bool faiss_backend::search(const vector_data &query, size_t top_k,
                          std::vector<search_result> *results) const {
  if (m_mode == backend_mode::kMemory) {
#ifdef HAVE_FAISS
    return search_with_faiss(query, top_k, results);
#else
    return m_memory_fallback.search(query, top_k, results);
#endif
  }
#ifdef HAVE_FAISS
  if (search_with_faiss(query, top_k, results)) return true;
#endif
  return search_external_snapshot_entries(query, top_k, results);
}

bool faiss_backend::search_batch(
    const std::vector<vector_data> &queries, size_t top_k,
    std::vector<std::vector<search_result>> *results) const {
  if (m_mode == backend_mode::kMemory) {
#ifdef HAVE_FAISS
    return search_batch_with_faiss(queries, top_k, results);
#else
    return m_memory_fallback.search_batch(queries, top_k, results);
#endif
  }
#ifdef HAVE_FAISS
  if (search_batch_with_faiss(queries, top_k, results)) return true;
#endif
  if (results == nullptr) return false;
  results->clear();
  results->resize(queries.size());
  for (size_t i = 0; i < queries.size(); ++i) {
    if (!search_external_snapshot_entries(queries[i], top_k, &(*results)[i]))
      return false;
  }
  return true;
}

bool faiss_backend::load_committed_entries(
    const std::unordered_map<uint64_t, vector_data> &entries) {
  if (m_mode == backend_mode::kMemory) {
#ifdef HAVE_FAISS
    faiss_build_thread_scope thread_scope(m_faiss_build_threads);
    m_faiss_index.reset();
    if (!initialize_faiss_index()) return false;
    for (const auto &entry : entries) {
      if (!apply_faiss_mutation(entry.first, &entry.second)) return false;
    }
    return true;
#else
    return backend::load_committed_entries(entries);
#endif
  }
#ifdef HAVE_FAISS
  if (!rebuild_external_faiss_index(entries)) return false;
#endif
  m_external_snapshot_entries = entries;
  return persist_external_snapshot();
}

bool faiss_backend::recover() {
  m_last_recover_used_fallback = 0;
  if (m_mode == backend_mode::kMemory) return true;
  if (recover_external_snapshot()) {
    m_last_recover_used_fallback = 0;
    return true;
  }

  m_external_snapshot_entries.clear();
  m_last_recover_used_fallback = 1;
  vector_status::record_backend_recover_fallback();
  return true;
}

bool faiss_backend::persist_external_snapshot() {
  if (m_mode != backend_mode::kExternal) return true;
  if (m_external_manifest_path.empty()) {
    m_external_manifest_present = false;
    m_external_manifest_generation = 0;
    return true;
  }
  if (m_external_snapshot_entries.empty()) {
    if (!remove_if_exists(m_external_manifest_path)) return false;
    if (!remove_if_exists(quarantine_path_for(m_external_manifest_path)))
      return false;
    if (!remove_external_generated_snapshots()) return false;
    if (!remove_dir_if_empty(m_external_snapshot_directory)) return false;
    m_external_manifest_present = false;
    m_external_manifest_generation = 0;
    return true;
  }
  if (!ensure_parent_directory(m_external_manifest_path)) return false;

  uint64_t current_generation = 0;
  bool has_manifest = false;
  if (!load_external_manifest_generation(&current_generation, &has_manifest)) {
    if (!quarantine_file_if_exists(m_external_manifest_path)) return false;
    if (!remove_external_generated_snapshots()) return false;
    has_manifest = false;
    current_generation = 0;
    m_external_manifest_present = false;
    m_external_manifest_generation = 0;
  }
  const uint64_t next_generation = has_manifest ? (current_generation + 1) : 1;
  const std::string next_snapshot_path =
      external_snapshot_path_for_generation(next_generation);
  if (next_snapshot_path.empty()) return false;

#ifdef HAVE_FAISS
  if (!persist_faiss_index_file(next_snapshot_path)) return false;
#else
  const std::string temp_path = next_snapshot_path + ".tmp";
  std::ofstream file(temp_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.good()) return false;

  file << external_snapshot_header(m_sidecar_profile) << "\n";
  std::vector<uint64_t> doc_ids;
  doc_ids.reserve(m_external_snapshot_entries.size());
  for (const auto &entry : m_external_snapshot_entries) {
    doc_ids.push_back(entry.first);
  }
  std::sort(doc_ids.begin(), doc_ids.end());

  for (uint64_t doc_id : doc_ids) {
    const auto it = m_external_snapshot_entries.find(doc_id);
    if (it == m_external_snapshot_entries.end()) continue;
    if (it->second.size() != m_dimension) {
      file.close();
      std::remove(temp_path.c_str());
      return false;
    }

    const char *vector_data = reinterpret_cast<const char *>(it->second.data());
    const size_t vector_bytes = it->second.size() * sizeof(float);
    file << doc_id << "\t" << encode_hex_bytes(vector_data, vector_bytes)
         << "\n";
  }
  file.close();
  if (!file) {
    std::remove(temp_path.c_str());
    return false;
  }

  if (std::rename(temp_path.c_str(), next_snapshot_path.c_str()) != 0) {
    std::remove(temp_path.c_str());
    return false;
  }
#endif
  if (!save_external_manifest_generation(next_generation)) {
    std::remove(next_snapshot_path.c_str());
    return false;
  }
  if (has_manifest && current_generation != next_generation) {
    if (!remove_if_exists(external_snapshot_path_for_generation(current_generation)))
      return false;
  }
  maybe_release_external_serving_index();
  return true;
}

bool faiss_backend::recover_external_snapshot() {
  if (m_mode != backend_mode::kExternal) return true;
  if (m_external_manifest_path.empty()) {
    m_external_manifest_present = false;
    m_external_manifest_generation = 0;
    return true;
  }
  m_external_manifest_present = false;
  m_external_manifest_generation = 0;

  uint64_t manifest_generation = 0;
  bool has_manifest = false;
  if (!load_external_manifest_generation(&manifest_generation, &has_manifest)) {
    return false;
  }
  if (has_manifest) {
    m_external_manifest_present = true;
    m_external_manifest_generation = manifest_generation;
    return recover_external_snapshot_file(
        external_snapshot_path_for_generation(manifest_generation));
  }

  return true;
}

bool faiss_backend::load_external_manifest_generation(uint64_t *generation,
                                                  bool *exists) const {
  return load_external_manifest_generation_from_file(
      m_external_manifest_path, external_manifest_header(m_sidecar_profile),
      generation, exists);
}

bool faiss_backend::save_external_manifest_generation(uint64_t generation) {
  if (!save_external_manifest_generation_to_file(
          m_external_manifest_path, external_manifest_header(m_sidecar_profile),
          generation)) {
    return false;
  }
  m_external_manifest_present = true;
  m_external_manifest_generation = generation;
  return true;
}

std::string faiss_backend::external_snapshot_path_for_generation(
    uint64_t generation) const {
  if (generation == 0 || m_external_snapshot_directory.empty()) return "";
  std::string path = m_external_snapshot_directory;
  path.push_back('/');
  path.append(external_snapshot_prefix(m_sidecar_profile));
  path.append(std::to_string(generation));
  path.append(kFaissExternalSnapshotSuffix);
  return path;
}

bool faiss_backend::remove_external_generated_snapshots() const {
  return remove_generated_snapshots_with_prefix(
      m_external_snapshot_directory, external_snapshot_prefix(m_sidecar_profile));
}

bool faiss_backend::recover_external_snapshot_file(const std::string &path) {
#ifdef HAVE_FAISS
  if (recover_faiss_index_file(path)) return true;
#endif
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.good()) return false;

  std::string line;
  if (!std::getline(file, line)) {
    return false;
  }
  const std::string expected_header = external_snapshot_header(m_sidecar_profile);
  if (line != expected_header) return false;

  std::unordered_map<uint64_t, vector_data> recovered_entries;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    const size_t tab = line.find('\t');
    if (tab == std::string::npos || tab == 0 || tab == line.size() - 1)
      return false;

    uint64_t doc_id = 0;
    if (!parse_uint64(line.substr(0, tab), &doc_id)) return false;

    std::string vector_bytes;
    if (!decode_hex_bytes(line.substr(tab + 1), &vector_bytes)) return false;
    if ((vector_bytes.size() % sizeof(float)) != 0) return false;

    vector_data vector;
    vector.resize(vector_bytes.size() / sizeof(float));
    if (!vector_bytes.empty()) {
      std::memcpy(vector.data(), vector_bytes.data(), vector_bytes.size());
    }
    if (vector.size() != m_dimension) return false;

    recovered_entries[doc_id] = std::move(vector);
  }
  if (file.bad()) return false;

  m_external_snapshot_entries = std::move(recovered_entries);
  maybe_release_external_serving_index();
  return true;
}

void faiss_backend::maybe_release_external_serving_index() {
#ifdef HAVE_FAISS
  m_external_fallback.reset();
  if (m_mode != backend_mode::kExternal || m_keep_loaded_external_index)
    return;
  m_faiss_index.reset();
#endif
}

}  // namespace vector_index
