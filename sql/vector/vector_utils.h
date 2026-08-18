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

#ifndef SQL_VECTOR_UTILS_INCLUDED
#define SQL_VECTOR_UTILS_INCLUDED

#include <cstddef>
#include <vector>

class String;

namespace vector_utils {

constexpr size_t kVectorElemSize = sizeof(float);

enum class distance_metric { kEuclidean, kCosine, kInnerProduct };

/**
  Parse text form "[1,2,3]" into float elements.

  @param input Input string in session charset.
  @param out Output elements. Cleared on entry.
  @retval true Parsed successfully.
  @retval false Invalid syntax, non-finite value, or out-of-range value.
*/
bool parse_text_vector(const String *input, std::vector<float> *out);

/**
  Validate binary vector payload and return element count.

  @param input Binary payload where each element is float32.
  @param dim Output element count.
  @retval true Payload length is aligned and every float32 value is finite.
  @retval false Payload length is invalid or a value is NaN or infinity.
*/
bool parse_binary_vector(const String *input, size_t *dim);

/**
  Validate that two binary vectors have compatible dimensions.

  @param lhs Left-hand side binary vector.
  @param rhs Right-hand side binary vector.
  @param dim Output common dimension.
  @retval true Both vectors are valid and dimensions are equal.
  @retval false Invalid payload or mismatched dimensions.
*/
bool check_compatible_vector_inputs(const String *lhs, const String *rhs,
                                 size_t *dim);

/**
  Parse vector distance metric token.

  Accepted values are "L2", "EUCLIDEAN", "COSINE", "IP", and
  "INNER_PRODUCT" (case-insensitive).

  @param metric_arg metric_type token argument.
  @param metric Output parsed metric.
  @retval true metric_type token is recognized.
  @retval false Unsupported metric token.
*/
bool parse_distance_metric(const String *metric_arg, distance_metric *metric);

/**
  Compute distance between two binary vectors under selected metric.

  @param metric Distance metric.
  @param lhs Left-hand side binary vector.
  @param rhs Right-hand side binary vector.
  @param distance Output distance value.
  @retval true Computed successfully.
  @retval false Invalid payload, incompatible dimensions, or invalid norm.
*/
bool compute_distance(distance_metric metric, const String *lhs, const String *rhs,
                     double *distance);

/**
  Compute dot product of two binary vectors.

  @param lhs Left-hand side binary vector.
  @param rhs Right-hand side binary vector.
  @param dot_product Output dot product value.
  @retval true Computed successfully.
  @retval false Invalid payload or incompatible dimensions.
*/
bool compute_dot_product(const String *lhs, const String *rhs, double *dot_product);

}  // namespace vector_utils

#endif  // SQL_VECTOR_UTILS_INCLUDED
