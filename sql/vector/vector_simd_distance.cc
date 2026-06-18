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

#include "sql/vector/vector_simd_distance.h"

#if defined(MYSQL_VECTOR_ENABLE_NATIVE_SIMD) &&                         \
    (defined(__x86_64__) || defined(__i386__)) &&                       \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define MYSQL_VECTOR_HAS_X86_SIMD 1
#endif

namespace vector_index {
namespace {

void record_kernel(l2_distance_stats *stats, l2_distance_kernel kernel) {
  if (stats == nullptr) return;
  stats->kernel = kernel;
  ++stats->calls;
}

#if defined(MYSQL_VECTOR_HAS_X86_SIMD)
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

__attribute__((target("avx512f"))) float l2_distance_avx512(
    const float *lhs, const float *rhs, size_t dimension) {
  size_t i = 0;
  __m512 sum = _mm512_setzero_ps();
  for (; i + 16 <= dimension; i += 16) {
    const __m512 left = _mm512_loadu_ps(lhs + i);
    const __m512 right = _mm512_loadu_ps(rhs + i);
    const __m512 diff = _mm512_sub_ps(left, right);
    sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
  }

  alignas(64) float lanes[16];
  _mm512_storeu_ps(lanes, sum);
  float result = 0.0F;
  for (float value : lanes) result += value;
  for (; i < dimension; ++i) {
    const float diff = lhs[i] - rhs[i];
    result += diff * diff;
  }
  return result;
}

__attribute__((target("avx2"))) float l2_distance_avx2(const float *lhs,
                                                       const float *rhs,
                                                       size_t dimension) {
  size_t i = 0;
  __m256 sum = _mm256_setzero_ps();
  for (; i + 8 <= dimension; i += 8) {
    const __m256 left = _mm256_loadu_ps(lhs + i);
    const __m256 right = _mm256_loadu_ps(rhs + i);
    const __m256 diff = _mm256_sub_ps(left, right);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
  }

  alignas(32) float lanes[8];
  _mm256_storeu_ps(lanes, sum);
  float result = 0.0F;
  for (float value : lanes) result += value;
  for (; i < dimension; ++i) {
    const float diff = lhs[i] - rhs[i];
    result += diff * diff;
  }
  return result;
}
#endif

}  // namespace

float l2_distance_scalar(const float *lhs, const float *rhs,
                         size_t dimension) {
  float result = 0.0F;
  for (size_t i = 0; i < dimension; ++i) {
    const float diff = lhs[i] - rhs[i];
    result += diff * diff;
  }
  return result;
}

float l2_distance(const float *lhs, const float *rhs, size_t dimension,
                  l2_distance_stats *stats) {
#if defined(MYSQL_VECTOR_HAS_X86_SIMD)
  if (cpu_supports_avx512()) {
    record_kernel(stats, l2_distance_kernel::kAvx512);
    return l2_distance_avx512(lhs, rhs, dimension);
  }
  if (cpu_supports_avx2()) {
    record_kernel(stats, l2_distance_kernel::kAvx2);
    return l2_distance_avx2(lhs, rhs, dimension);
  }
#endif

  record_kernel(stats, l2_distance_kernel::kScalar);
  return l2_distance_scalar(lhs, rhs, dimension);
}

const char *l2_distance_kernel_name(l2_distance_kernel kernel) {
  switch (kernel) {
    case l2_distance_kernel::kScalar:
      return "scalar";
    case l2_distance_kernel::kAvx2:
      return "avx2";
    case l2_distance_kernel::kAvx512:
      return "avx512";
    case l2_distance_kernel::kNeon:
      return "neon";
  }
  return "unknown";
}

}  // namespace vector_index
