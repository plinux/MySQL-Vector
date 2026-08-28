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

#ifndef SQL_VECTOR_SIMD_DISTANCE_INCLUDED
#define SQL_VECTOR_SIMD_DISTANCE_INCLUDED

#include <cstddef>
#include <cstdint>

namespace vector_index {

/** L2 distance implementation selected for a single distance call. */
enum class l2_distance_kernel { kScalar, kAvx2, kAvx512, kNeon };

/** Lightweight per-call diagnostics for Native Runtime distance loops. */
struct l2_distance_stats {
  l2_distance_kernel kernel{l2_distance_kernel::kScalar};
  uint64_t calls{0};
};

using l2_distance_function = float (*)(const float *, const float *, size_t);

/** L2 kernel selected once for a fixed vector dimension. */
struct l2_distance_context {
  size_t dimension{0};
  l2_distance_kernel kernel{l2_distance_kernel::kScalar};
  l2_distance_function function{nullptr};
};

/**
  Compute squared L2 distance with the portable scalar implementation.

  @param lhs First vector.
  @param rhs Second vector.
  @param dimension Number of float elements.
  @return Sum of squared element-wise differences.
*/
float l2_distance_scalar(const float *lhs, const float *rhs, size_t dimension);

/**
  Compute squared L2 distance using the best runtime-selected kernel.

  The function always has a scalar fallback and accepts unaligned input.
  When @c stats is not null, it records the selected kernel and increments
  the number of distance calls.

  @param lhs First vector.
  @param rhs Second vector.
  @param dimension Number of float elements.
  @param stats Optional diagnostics sink.
  @return Sum of squared element-wise differences.
*/
float l2_distance(const float *lhs, const float *rhs, size_t dimension,
                  l2_distance_stats *stats);

/** Create a reusable L2 context without per-distance CPU feature checks. */
l2_distance_context make_l2_distance_context(size_t dimension);

/**
  Compute squared L2 distance with a previously selected kernel.

  @param context Reusable dimension and kernel selection.
  @param lhs First vector.
  @param rhs Second vector.
  @param stats Optional diagnostics sink.
  @return Sum of squared element-wise differences.
*/
float l2_distance_with_context(const l2_distance_context &context,
                               const float *lhs, const float *rhs,
                               l2_distance_stats *stats);

/** Return a stable, lowercase name for a distance kernel. */
const char *l2_distance_kernel_name(l2_distance_kernel kernel);

}  // namespace vector_index

#endif  // SQL_VECTOR_SIMD_DISTANCE_INCLUDED
