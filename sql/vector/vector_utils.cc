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

#include "sql/vector/vector_utils.h"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>

#include "m_ctype.h"
#include "my_byteorder.h"
#include "my_inttypes.h"
#include "sql_string.h"

namespace {

bool is_ascii_space(char c) {
  return my_isspace(&my_charset_latin1, static_cast<unsigned char>(c)) != 0;
}

void skip_spaces(const char *ptr, size_t len, size_t *pos) {
  while (*pos < len && is_ascii_space(ptr[*pos])) (*pos)++;
}

bool equals_ascii_no_case(const char *ptr, size_t len, const char *literal) {
  size_t lit_len = 0;
  for (const char *p = literal; *p != '\0'; ++p) lit_len++;
  if (len != lit_len) return false;
  for (size_t i = 0; i < len; ++i) {
    const char lhs = static_cast<char>(
        std::toupper(static_cast<unsigned char>(ptr[i])));
    const char rhs = static_cast<char>(
        std::toupper(static_cast<unsigned char>(literal[i])));
    if (lhs != rhs) return false;
  }
  return true;
}

}  // namespace

namespace vector_utils {

bool parse_text_vector(const String *input, std::vector<float> *out) {
  out->clear();
  std::vector<float> parsed;
  const char *ptr = input->ptr();
  const size_t len = input->length();
  size_t pos = 0;

  skip_spaces(ptr, len, &pos);
  if (pos >= len || ptr[pos] != '[') return false;
  pos++;

  skip_spaces(ptr, len, &pos);
  if (pos < len && ptr[pos] == ']') {
    pos++;
    skip_spaces(ptr, len, &pos);
    return pos == len;
  }

  while (pos < len) {
    int err = 0;
    const char *end = nullptr;
    const char *start = ptr + pos;
    const double value = my_strntod(input->charset(), start,
                                    static_cast<uint>(len - pos), &end, &err);
    if (err != 0 || end == start || !std::isfinite(value)) return false;
    if (value < static_cast<double>(std::numeric_limits<float>::lowest()) ||
        value > static_cast<double>(std::numeric_limits<float>::max())) {
      return false;
    }
    parsed.push_back(static_cast<float>(value));
    pos = static_cast<size_t>(end - ptr);

    skip_spaces(ptr, len, &pos);
    if (pos >= len) return false;

    if (ptr[pos] == ',') {
      pos++;
      skip_spaces(ptr, len, &pos);
      continue;
    }
    if (ptr[pos] == ']') {
      pos++;
      skip_spaces(ptr, len, &pos);
      if (pos != len) return false;
      out->swap(parsed);
      return true;
    }
    return false;
  }

  return false;
}

bool parse_binary_vector(const String *input, size_t *dim) {
  if (input->length() % kVectorElemSize != 0) return false;
  *dim = input->length() / kVectorElemSize;
  return true;
}

bool check_compatible_vector_inputs(const String *lhs, const String *rhs,
                                    size_t *dim) {
  size_t lhs_dim = 0;
  size_t rhs_dim = 0;
  if (!parse_binary_vector(lhs, &lhs_dim) ||
      !parse_binary_vector(rhs, &rhs_dim) ||
      lhs_dim != rhs_dim) {
    return false;
  }
  *dim = lhs_dim;
  return true;
}

bool parse_distance_metric(const String *metric_arg, distance_metric *metric) {
  const char *ptr = metric_arg->ptr();
  size_t begin = 0;
  size_t end = metric_arg->length();
  while (begin < end && is_ascii_space(ptr[begin])) begin++;
  while (end > begin && is_ascii_space(ptr[end - 1])) end--;
  const size_t len = end - begin;
  const char *token = ptr + begin;

  if (equals_ascii_no_case(token, len, "L2") ||
      equals_ascii_no_case(token, len, "EUCLIDEAN")) {
    *metric = distance_metric::kEuclidean;
    return true;
  }
  if (equals_ascii_no_case(token, len, "COSINE")) {
    *metric = distance_metric::kCosine;
    return true;
  }
  if (equals_ascii_no_case(token, len, "IP") ||
      equals_ascii_no_case(token, len, "INNER_PRODUCT")) {
    *metric = distance_metric::kInnerProduct;
    return true;
  }
  return false;
}

bool compute_distance(distance_metric metric, const String *lhs,
                      const String *rhs, double *distance) {
  size_t dim = 0;
  if (!check_compatible_vector_inputs(lhs, rhs, &dim)) return false;

  const uchar *lhs_ptr = reinterpret_cast<const uchar *>(lhs->ptr());
  const uchar *rhs_ptr = reinterpret_cast<const uchar *>(rhs->ptr());
  if (metric == distance_metric::kEuclidean) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      const double l = static_cast<double>(float4get(lhs_ptr + i * kVectorElemSize));
      const double r = static_cast<double>(float4get(rhs_ptr + i * kVectorElemSize));
      const double diff = l - r;
      sum += diff * diff;
    }
    *distance = std::sqrt(sum);
    return true;
  }

  double dot = 0.0;
  if (metric == distance_metric::kInnerProduct) {
    for (size_t i = 0; i < dim; ++i) {
      const double l =
          static_cast<double>(float4get(lhs_ptr + i * kVectorElemSize));
      const double r =
          static_cast<double>(float4get(rhs_ptr + i * kVectorElemSize));
      dot += l * r;
    }
    // Unified distance API uses ascending order; larger inner product is better.
    *distance = -dot;
    return true;
  }

  double lhs_norm = 0.0;
  double rhs_norm = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    const double l = static_cast<double>(float4get(lhs_ptr + i * kVectorElemSize));
    const double r = static_cast<double>(float4get(rhs_ptr + i * kVectorElemSize));
    dot += l * r;
    lhs_norm += l * l;
    rhs_norm += r * r;
  }
  if (lhs_norm <= 0.0 || rhs_norm <= 0.0) return false;
  *distance = 1.0 - (dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm)));
  return true;
}

bool compute_dot_product(const String *lhs, const String *rhs,
                         double *dot_product) {
  size_t dim = 0;
  if (!check_compatible_vector_inputs(lhs, rhs, &dim)) return false;

  const uchar *lhs_ptr = reinterpret_cast<const uchar *>(lhs->ptr());
  const uchar *rhs_ptr = reinterpret_cast<const uchar *>(rhs->ptr());
  double dot = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    const double l = static_cast<double>(float4get(lhs_ptr + i * kVectorElemSize));
    const double r = static_cast<double>(float4get(rhs_ptr + i * kVectorElemSize));
    dot += l * r;
  }
  *dot_product = dot;
  return true;
}

}  // namespace vector_utils
