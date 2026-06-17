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

#ifndef SQL_VECTOR_INDEX_LIMITS_INCLUDED
#define SQL_VECTOR_INDEX_LIMITS_INCLUDED

#include <cstddef>
#include <cstdint>
#include <limits>

namespace vector_index {

inline constexpr uint32_t k_max_build_threads = 65535;
inline constexpr uint32_t k_max_faiss_pq_bits = 24;
inline constexpr uint32_t k_max_diskann_search_beamwidth = 65535;
inline constexpr uint32_t k_max_vector_dimension =
    std::numeric_limits<uint32_t>::max() / sizeof(float);
inline constexpr uint32_t k_max_search_top_k = 10000;
inline constexpr uint32_t k_max_search_batch_count = 1024;
inline constexpr uint32_t k_max_search_batch_results = 65536;
inline constexpr uint32_t k_default_search_batch_count = 1024;
inline constexpr uint32_t k_default_search_batch_result_count = 65536;
inline constexpr uint32_t k_default_batch_search_threads = 32;
inline constexpr uint32_t k_default_diskann_max_degree = 32;
inline constexpr uint32_t k_default_diskann_build_complexity = 64;
inline constexpr uint32_t k_default_diskann_search_complexity = 64;
inline constexpr uint32_t k_default_diskann_search_beamwidth = 16;
inline constexpr uint32_t k_default_hnsw_search_ef = 64;
inline constexpr uint32_t k_default_hnsw_m = 16;
inline constexpr uint32_t k_default_hnsw_ef_construction = 200;

inline size_t saturated_add_size(size_t left, size_t right) {
  const size_t max_value = std::numeric_limits<size_t>::max();
  if (right > max_value - left) return max_value;
  return left + right;
}

inline size_t saturated_mul_size(size_t left, size_t right) {
  if (left == 0 || right == 0) return 0;
  const size_t max_value = std::numeric_limits<size_t>::max();
  if (left > max_value / right) return max_value;
  return left * right;
}

}  // namespace vector_index

#endif  // SQL_VECTOR_INDEX_LIMITS_INCLUDED
