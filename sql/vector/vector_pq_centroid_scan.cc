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
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA */

#include "sql/vector/vector_pq_centroid_scan.h"

#ifdef HAVE_VECTOR_INDEX

#include <algorithm>
#include <cmath>
#include <limits>

#if defined(MYSQL_VECTOR_ENABLE_NATIVE_SIMD) &&   \
    (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define MYSQL_VECTOR_PQ_HAS_X86_SIMD 1
#endif

namespace vector_index {
namespace {

constexpr uint32_t kAvx512CenterWidth = 16;
constexpr uint32_t kAvx2CenterWidth = 8;

void set_error(std::string *error, const char *message) {
  if (error != nullptr) *error = message == nullptr ? "" : message;
}

#if defined(MYSQL_VECTOR_PQ_HAS_X86_SIMD)
void init_x86_cpu_features() {
#if defined(__GNUC__) && !defined(__clang__)
  __builtin_cpu_init();
#endif
}

bool cpu_supports_avx512() {
  init_x86_cpu_features();
  return __builtin_cpu_supports("avx512f");
}

bool cpu_supports_avx2() {
  init_x86_cpu_features();
  return __builtin_cpu_supports("avx2");
}
#endif

pq_centroid_scan_kernel auto_kernel() {
#if defined(MYSQL_VECTOR_PQ_HAS_X86_SIMD)
  if (cpu_supports_avx512()) return pq_centroid_scan_kernel::kAvx512;
  if (cpu_supports_avx2()) return pq_centroid_scan_kernel::kAvx2;
#endif
  return pq_centroid_scan_kernel::kScalar;
}

uint32_t kernel_width(pq_centroid_scan_kernel kernel) {
  switch (kernel) {
    case pq_centroid_scan_kernel::kAvx512:
      return kAvx512CenterWidth;
    case pq_centroid_scan_kernel::kAvx2:
      return kAvx2CenterWidth;
    case pq_centroid_scan_kernel::kAuto:
    case pq_centroid_scan_kernel::kScalar:
      return 1;
  }
  return 1;
}

void record_distance(uint32_t center, float distance, float *squared_distances,
                     uint32_t *nearest_center, float *nearest_distance) {
  if (squared_distances != nullptr) squared_distances[center] = distance;
  if (distance < *nearest_distance) {
    *nearest_distance = distance;
    *nearest_center = center;
  }
}

void scan_scalar(const pq_centroid_model &model, const float *values,
                 float *squared_distances, uint32_t *nearest_center,
                 float *nearest_distance, bool euclidean) {
  for (uint32_t center = 0; center < model.center_count; ++center) {
    float distance = 0.0F;
    for (uint32_t dim = 0; dim < model.dimension; ++dim) {
      const float diff =
          values[dim] -
          model.packed_centers[static_cast<size_t>(dim) * model.center_stride +
                               center];
      distance += diff * diff;
    }
    if (euclidean) distance = std::sqrt(distance);
    record_distance(center, distance, squared_distances, nearest_center,
                    nearest_distance);
  }
}

#if defined(MYSQL_VECTOR_PQ_HAS_X86_SIMD)
__attribute__((target("avx512f"))) void scan_avx512(
    const pq_centroid_model &model, const float *values,
    float *squared_distances, uint32_t *nearest_center, float *nearest_distance,
    bool euclidean) {
  alignas(64) float lanes[kAvx512CenterWidth];
  for (uint32_t base = 0; base < model.center_stride;
       base += kAvx512CenterWidth) {
    __m512 distances = _mm512_setzero_ps();
    for (uint32_t dim = 0; dim < model.dimension; ++dim) {
      const __m512 centers = _mm512_loadu_ps(
          model.packed_centers.data() +
          static_cast<size_t>(dim) * model.center_stride + base);
      const __m512 query = _mm512_set1_ps(values[dim]);
      const __m512 diff = _mm512_sub_ps(query, centers);
      distances = _mm512_add_ps(distances, _mm512_mul_ps(diff, diff));
    }
    if (euclidean) distances = _mm512_sqrt_ps(distances);
    _mm512_storeu_ps(lanes, distances);
    const uint32_t valid_lanes =
        std::min(kAvx512CenterWidth, model.center_count - base);
    for (uint32_t lane = 0; lane < valid_lanes; ++lane) {
      record_distance(base + lane, lanes[lane], squared_distances,
                      nearest_center, nearest_distance);
    }
  }
}

__attribute__((target("avx2"))) void scan_avx2(const pq_centroid_model &model,
                                               const float *values,
                                               float *squared_distances,
                                               uint32_t *nearest_center,
                                               float *nearest_distance,
                                               bool euclidean) {
  alignas(32) float lanes[kAvx2CenterWidth];
  for (uint32_t base = 0; base < model.center_stride;
       base += kAvx2CenterWidth) {
    __m256 distances = _mm256_setzero_ps();
    for (uint32_t dim = 0; dim < model.dimension; ++dim) {
      const __m256 centers = _mm256_loadu_ps(
          model.packed_centers.data() +
          static_cast<size_t>(dim) * model.center_stride + base);
      const __m256 query = _mm256_set1_ps(values[dim]);
      const __m256 diff = _mm256_sub_ps(query, centers);
      distances = _mm256_add_ps(distances, _mm256_mul_ps(diff, diff));
    }
    if (euclidean) distances = _mm256_sqrt_ps(distances);
    _mm256_storeu_ps(lanes, distances);
    const uint32_t valid_lanes =
        std::min(kAvx2CenterWidth, model.center_count - base);
    for (uint32_t lane = 0; lane < valid_lanes; ++lane) {
      record_distance(base + lane, lanes[lane], squared_distances,
                      nearest_center, nearest_distance);
    }
  }
}
#endif

bool valid_model(const pq_centroid_model &model) {
  if (model.dimension == 0 || model.center_count == 0 ||
      model.center_stride < model.center_count ||
      model.kernel == pq_centroid_scan_kernel::kAuto ||
      static_cast<size_t>(model.dimension) >
          std::numeric_limits<size_t>::max() / model.center_stride) {
    return false;
  }
  return model.packed_centers.size() ==
         static_cast<size_t>(model.dimension) * model.center_stride;
}

bool scan_pq_centroids_impl(const pq_centroid_model &model, const float *values,
                            float *distances, uint32_t *nearest_center,
                            float *nearest_distance,
                            pq_centroid_scan_stats *stats, bool euclidean) {
  if (!valid_model(model) || values == nullptr ||
      (distances == nullptr && nearest_center == nullptr)) {
    return false;
  }

  uint32_t selected_center = 0;
  float selected_distance = std::numeric_limits<float>::infinity();
  switch (model.kernel) {
    case pq_centroid_scan_kernel::kScalar:
      scan_scalar(model, values, distances, &selected_center,
                  &selected_distance, euclidean);
      break;
    case pq_centroid_scan_kernel::kAvx2:
#if defined(MYSQL_VECTOR_PQ_HAS_X86_SIMD)
      scan_avx2(model, values, distances, &selected_center, &selected_distance,
                euclidean);
      break;
#else
      return false;
#endif
    case pq_centroid_scan_kernel::kAvx512:
#if defined(MYSQL_VECTOR_PQ_HAS_X86_SIMD)
      scan_avx512(model, values, distances, &selected_center,
                  &selected_distance, euclidean);
      break;
#else
      return false;
#endif
    case pq_centroid_scan_kernel::kAuto:
      return false;
  }

  if (nearest_center != nullptr) *nearest_center = selected_center;
  if (nearest_distance != nullptr) *nearest_distance = selected_distance;
  if (stats != nullptr) {
    stats->kernel = model.kernel;
    stats->distance_evaluations += model.center_count;
  }
  return true;
}

}  // namespace

bool pq_centroid_scan_kernel_supported(pq_centroid_scan_kernel kernel) {
  switch (kernel) {
    case pq_centroid_scan_kernel::kAuto:
    case pq_centroid_scan_kernel::kScalar:
      return true;
    case pq_centroid_scan_kernel::kAvx2:
#if defined(MYSQL_VECTOR_PQ_HAS_X86_SIMD)
      return cpu_supports_avx2();
#else
      return false;
#endif
    case pq_centroid_scan_kernel::kAvx512:
#if defined(MYSQL_VECTOR_PQ_HAS_X86_SIMD)
      return cpu_supports_avx512();
#else
      return false;
#endif
  }
  return false;
}

const char *pq_centroid_scan_kernel_name(pq_centroid_scan_kernel kernel) {
  switch (kernel) {
    case pq_centroid_scan_kernel::kAuto:
      return "auto";
    case pq_centroid_scan_kernel::kScalar:
      return "scalar_fallback";
    case pq_centroid_scan_kernel::kAvx2:
      return "avx2";
    case pq_centroid_scan_kernel::kAvx512:
      return "avx512";
  }
  return "unknown";
}

bool make_pq_centroid_model(const float *centers, uint32_t center_count,
                            uint32_t dimension,
                            pq_centroid_scan_kernel requested_kernel,
                            pq_centroid_model *model, std::string *error) {
  if (model == nullptr) {
    set_error(error, "centroid model is null");
    return false;
  }
  *model = {};
  if (centers == nullptr || center_count == 0 || dimension == 0) {
    set_error(error, "centroid matrix is invalid");
    return false;
  }
  if (static_cast<size_t>(center_count) >
      std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension)) {
    set_error(error, "centroid matrix size overflow");
    return false;
  }
  const size_t value_count =
      static_cast<size_t>(center_count) * static_cast<size_t>(dimension);
  if (std::any_of(centers, centers + value_count,
                  [](float value) { return !std::isfinite(value); })) {
    set_error(error, "centroid matrix contains non-finite value");
    return false;
  }

  const pq_centroid_scan_kernel selected_kernel =
      requested_kernel == pq_centroid_scan_kernel::kAuto ? auto_kernel()
                                                         : requested_kernel;
  if (!pq_centroid_scan_kernel_supported(selected_kernel)) {
    set_error(error, "requested centroid scan kernel is unavailable");
    return false;
  }
  const uint32_t width = kernel_width(selected_kernel);
  if (center_count > std::numeric_limits<uint32_t>::max() - (width - 1)) {
    set_error(error, "centroid stride overflow");
    return false;
  }
  const uint32_t stride = ((center_count + width - 1) / width) * width;
  if (static_cast<size_t>(dimension) >
      std::numeric_limits<size_t>::max() / stride) {
    set_error(error, "packed centroid size overflow");
    return false;
  }

  model->dimension = dimension;
  model->center_count = center_count;
  model->center_stride = stride;
  model->kernel = selected_kernel;
  model->packed_centers.assign(static_cast<size_t>(dimension) * stride, 0.0F);
  for (uint32_t dim = 0; dim < dimension; ++dim) {
    float *target =
        model->packed_centers.data() + static_cast<size_t>(dim) * stride;
    for (uint32_t center = 0; center < center_count; ++center) {
      target[center] = centers[static_cast<size_t>(center) * dimension + dim];
    }
  }
  if (error != nullptr) error->clear();
  return true;
}

bool scan_pq_centroids(const pq_centroid_model &model, const float *values,
                       float *squared_distances, uint32_t *nearest_center,
                       float *nearest_squared_distance,
                       pq_centroid_scan_stats *stats) {
  return scan_pq_centroids_impl(model, values, squared_distances,
                                nearest_center, nearest_squared_distance, stats,
                                false);
}

bool scan_pq_centroid_distances(const pq_centroid_model &model,
                                const float *values, float *distances,
                                uint32_t *nearest_center,
                                float *nearest_distance,
                                pq_centroid_scan_stats *stats) {
  return scan_pq_centroids_impl(model, values, distances, nearest_center,
                                nearest_distance, stats, true);
}

}  // namespace vector_index

#endif  // HAVE_VECTOR_INDEX
