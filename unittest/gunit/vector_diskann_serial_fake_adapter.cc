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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sql/vector/vector_diskann_garnet_abi.h"

namespace {

using read_data_callback =
    vector_index::diskann_garnet_abi::diskann_read_data_callback;
using read_callback = vector_index::diskann_garnet_abi::diskann_read_callback;
using write_callback = vector_index::diskann_garnet_abi::diskann_write_callback;
using delete_callback =
    vector_index::diskann_garnet_abi::diskann_delete_callback;
using rmw_data_callback =
    vector_index::diskann_garnet_abi::diskann_rmw_data_callback;
using read_modify_write_callback =
    vector_index::diskann_garnet_abi::diskann_read_modify_write_callback;
using filter_callback =
    vector_index::diskann_garnet_abi::diskann_filter_callback;
using log_callback = vector_index::diskann_garnet_abi::diskann_log_callback;

constexpr uint64_t k_vector_term = 0;
constexpr uint64_t k_metadata_term = 4;
constexpr uint32_t k_no_quant = 1;
constexpr uint8_t k_insert_failure = 0;
constexpr uint8_t k_insert_success = 1;

struct fake_serial_index {
  uint64_t context{0};
  size_t dimension{0};
  int32_t metric_type{2};
  read_callback read{nullptr};
  write_callback write{nullptr};
  delete_callback erase{nullptr};
  read_modify_write_callback read_modify_write{nullptr};
  std::unordered_map<uint64_t, std::vector<float>> vectors;
};

bool parse_doc_id(const uint8_t *data, size_t length, uint64_t *doc_id) {
  if (data == nullptr || doc_id == nullptr || length != sizeof(*doc_id)) {
    return false;
  }
  std::memcpy(doc_id, data, sizeof(*doc_id));
  return true;
}

double distance(int32_t metric_type, const float *lhs, const float *rhs,
                size_t dimension) {
  double dot = 0.0;
  double lhs_norm = 0.0;
  double rhs_norm = 0.0;
  double squared_distance = 0.0;
  for (size_t i = 0; i < dimension; ++i) {
    dot += static_cast<double>(lhs[i]) * rhs[i];
    lhs_norm += static_cast<double>(lhs[i]) * lhs[i];
    rhs_norm += static_cast<double>(rhs[i]) * rhs[i];
    const double delta = static_cast<double>(lhs[i]) - rhs[i];
    squared_distance += delta * delta;
  }
  if (metric_type == 1) return -dot;
  if (metric_type == 0) {
    if (lhs_norm == 0.0 || rhs_norm == 0.0) {
      return std::numeric_limits<double>::infinity();
    }
    return 1.0 - dot / std::sqrt(lhs_norm * rhs_norm);
  }
  return squared_distance;
}

struct read_capture {
  std::vector<uint8_t> value;
};

void capture_read(uint32_t, void *user_data, const uint8_t *data,
                  size_t length) {
  auto *capture = static_cast<read_capture *>(user_data);
  if (capture == nullptr || (data == nullptr && length != 0)) return;
  capture->value.assign(data, data + length);
}

void increment_counter(void *, uint8_t *data, size_t length) {
  if (data == nullptr || length < sizeof(uint64_t)) return;
  uint64_t value = 0;
  std::memcpy(&value, data, sizeof(value));
  ++value;
  std::memcpy(data, &value, sizeof(value));
}

bool persist_vector(fake_serial_index *index, const uint8_t *id_data,
                    size_t id_length, const uint8_t *vector_data,
                    size_t vector_bytes) {
  if (!index->write(index->context | k_vector_term, id_data, id_length,
                    vector_data, vector_bytes)) {
    return false;
  }
  if (!index->read_modify_write(index->context | k_metadata_term, id_data,
                                id_length, sizeof(uint64_t), increment_counter,
                                nullptr)) {
    return false;
  }

  std::vector<uint8_t> encoded_key(sizeof(uint32_t) + id_length);
  const uint32_t encoded_length = static_cast<uint32_t>(id_length);
  std::memcpy(encoded_key.data(), &encoded_length, sizeof(encoded_length));
  std::memcpy(encoded_key.data() + sizeof(encoded_length), id_data, id_length);
  read_capture capture;
  index->read(index->context | k_vector_term, 1,
              static_cast<uint32_t>(vector_bytes), encoded_key.data(),
              encoded_key.size(), capture_read, &capture);
  return capture.value.size() == vector_bytes &&
         std::memcmp(capture.value.data(), vector_data, vector_bytes) == 0;
}

}  // namespace

extern "C" {

const void *create_index(uint64_t context, uint32_t dimension,
                         uint32_t reduce_dimension, uint32_t quant_type,
                         int32_t metric_type, uint32_t, uint32_t,
                         read_callback read, write_callback write,
                         delete_callback erase,
                         read_modify_write_callback read_modify_write,
                         filter_callback filter, log_callback log,
                         bool *quantization_needed) {
  if (dimension == 0 || reduce_dimension != 0 || quant_type != k_no_quant ||
      read == nullptr || write == nullptr || erase == nullptr ||
      read_modify_write == nullptr || filter == nullptr || log == nullptr ||
      quantization_needed == nullptr || !filter(context, nullptr, 0)) {
    return nullptr;
  }
  static constexpr uint8_t k_created_message[] = "created";
  log(context, k_created_message, sizeof(k_created_message) - 1);
  *quantization_needed = false;
  auto *index = new fake_serial_index;
  index->context = context;
  index->dimension = dimension;
  index->metric_type = metric_type;
  index->read = read;
  index->write = write;
  index->erase = erase;
  index->read_modify_write = read_modify_write;
  return index;
}

void drop_index(uint64_t, const void *index) {
  delete static_cast<const fake_serial_index *>(index);
}

uint8_t insert(uint64_t, const void *index_ptr, const uint8_t *id_data,
               size_t id_length, const uint8_t *vector_data,
               size_t vector_length, const uint8_t *, size_t) {
  auto *index = const_cast<fake_serial_index *>(
      static_cast<const fake_serial_index *>(index_ptr));
  uint64_t doc_id = 0;
  if (index == nullptr || vector_data == nullptr ||
      vector_length != index->dimension ||
      !parse_doc_id(id_data, id_length, &doc_id)) {
    return k_insert_failure;
  }
  const size_t vector_bytes = vector_length * sizeof(float);
  if (!persist_vector(index, id_data, id_length, vector_data, vector_bytes)) {
    return k_insert_failure;
  }
  const auto *values = reinterpret_cast<const float *>(vector_data);
  index->vectors[doc_id] = std::vector<float>(values, values + vector_length);
  return k_insert_success;
}

bool build_quant_table(uint64_t, const void *index_ptr) {
  return index_ptr != nullptr;
}

bool backfill_quant_vectors(uint64_t, const void *index_ptr,
                            size_t task_index, size_t task_count) {
  return index_ptr != nullptr && task_count != 0 && task_index < task_count;
}

bool random_members(uint64_t, const void *, uint32_t, uint8_t *, size_t) {
  return false;
}

int32_t search_neighbors(uint64_t, const void *, const uint8_t *, size_t,
                         uint8_t *, size_t, float *, size_t, void *) {
  return -1;
}

int32_t search_vector(uint64_t, const void *index_ptr,
                      const uint8_t *vector_data, size_t vector_length, float,
                      uint32_t, const uint8_t *, size_t, size_t,
                      uint8_t *output_ids, size_t output_ids_length,
                      float *output_distances, size_t output_capacity,
                      uint32_t beam_width, void *) {
  const auto *index = static_cast<const fake_serial_index *>(index_ptr);
  if (index == nullptr || vector_data == nullptr ||
      vector_length != index->dimension ||
      beam_width == 0 ||
      (output_ids == nullptr && output_capacity != 0) ||
      (output_distances == nullptr && output_capacity != 0)) {
    return -1;
  }

  const auto *query = reinterpret_cast<const float *>(vector_data);
  std::vector<std::pair<uint64_t, double>> candidates;
  candidates.reserve(index->vectors.size());
  for (const auto &entry : index->vectors) {
    candidates.emplace_back(
        entry.first, distance(index->metric_type, query, entry.second.data(),
                              index->dimension));
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.second != rhs.second) return lhs.second < rhs.second;
              return lhs.first < rhs.first;
            });

  const size_t result_count = std::min(output_capacity, candidates.size());
  const size_t id_bytes = sizeof(uint32_t) + sizeof(uint64_t);
  if (result_count > output_ids_length / id_bytes ||
      result_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return -1;
  }
  uint8_t *id_output = output_ids;
  for (size_t i = 0; i < result_count; ++i) {
    const uint32_t length = sizeof(uint64_t);
    std::memcpy(id_output, &length, sizeof(length));
    id_output += sizeof(length);
    std::memcpy(id_output, &candidates[i].first, sizeof(candidates[i].first));
    id_output += sizeof(candidates[i].first);
    output_distances[i] = static_cast<float>(candidates[i].second);
  }
  return static_cast<int32_t>(result_count);
}

#ifdef __APPLE__
bool diskann_remove(uint64_t, const void *index_ptr, const uint8_t *id_data,
                    size_t id_length) __asm__("_remove");
#else
bool diskann_remove(uint64_t, const void *index_ptr, const uint8_t *id_data,
                    size_t id_length) __asm__("remove");
#endif

bool diskann_remove(uint64_t, const void *index_ptr, const uint8_t *id_data,
                    size_t id_length) {
  auto *index = const_cast<fake_serial_index *>(
      static_cast<const fake_serial_index *>(index_ptr));
  uint64_t doc_id = 0;
  if (index == nullptr || !parse_doc_id(id_data, id_length, &doc_id) ||
      index->vectors.erase(doc_id) == 0) {
    return false;
  }
  return index->erase(index->context | k_vector_term, id_data, id_length);
}

uint64_t card(uint64_t, const void *index) {
  if (index == nullptr) return 0;
  return static_cast<const fake_serial_index *>(index)->vectors.size();
}

}  // extern "C"

static_assert(std::is_same_v<decltype(&create_index),
                             vector_index::diskann_garnet_abi::
                                 diskann_create_index_fn>);
static_assert(std::is_same_v<decltype(&drop_index),
                             vector_index::diskann_garnet_abi::
                                 diskann_drop_index_fn>);
static_assert(std::is_same_v<decltype(&insert),
                             vector_index::diskann_garnet_abi::
                                 diskann_insert_fn>);
static_assert(std::is_same_v<decltype(&build_quant_table),
                             vector_index::diskann_garnet_abi::
                                 diskann_build_quant_table_fn>);
static_assert(std::is_same_v<decltype(&backfill_quant_vectors),
                             vector_index::diskann_garnet_abi::
                                 diskann_backfill_quant_vectors_fn>);
static_assert(std::is_same_v<decltype(&random_members),
                             vector_index::diskann_garnet_abi::
                                 diskann_random_members_fn>);
static_assert(std::is_same_v<decltype(&search_neighbors),
                             vector_index::diskann_garnet_abi::
                                 diskann_search_neighbors_fn>);
static_assert(std::is_same_v<decltype(&search_vector),
                             vector_index::diskann_garnet_abi::
                                 diskann_search_vector_fn>);
static_assert(std::is_same_v<decltype(&diskann_remove),
                             vector_index::diskann_garnet_abi::
                                 diskann_remove_fn>);
static_assert(std::is_same_v<decltype(&card),
                             vector_index::diskann_garnet_abi::diskann_card_fn>);
static_assert(
    vector_index::diskann_garnet_abi::k_diskann_garnet_abi_version == 2);
