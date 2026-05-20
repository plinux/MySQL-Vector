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

#ifndef SQL_VECTOR_INDEX_BACKEND_COMMON_INCLUDED
#define SQL_VECTOR_INDEX_BACKEND_COMMON_INCLUDED

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "sql/vector/vector_index_backend.h"

namespace vector_index::detail {

inline bool check_dimension(const vector_data &vector, size_t expected_dim) {
  return vector.size() == expected_dim;
}

inline bool compute_distance(metric_type metric, const vector_data &lhs, const vector_data &rhs,
                             double *distance) {
  if (lhs.size() != rhs.size()) return false;

  if (metric == metric_type::kEuclidean) {
    double sum = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
      const double diff = static_cast<double>(lhs[i]) -
                          static_cast<double>(rhs[i]);
      sum += diff * diff;
    }
    *distance = std::sqrt(sum);
    return true;
  }

  double dot = 0.0;
  if (metric == metric_type::kInnerProduct) {
    for (size_t i = 0; i < lhs.size(); ++i) {
      dot += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
    }
    *distance = -dot;
    return true;
  }

  double lhs_norm = 0.0;
  double rhs_norm = 0.0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const double l = static_cast<double>(lhs[i]);
    const double r = static_cast<double>(rhs[i]);
    dot += l * r;
    lhs_norm += l * l;
    rhs_norm += r * r;
  }
  if (lhs_norm <= 0.0 || rhs_norm <= 0.0) return false;
  *distance = 1.0 - (dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm)));
  return true;
}

inline std::string normalize_token(const std::string &input) {
  size_t begin = 0;
  while (begin < input.size() &&
         std::isspace(static_cast<unsigned char>(input[begin]))) {
    ++begin;
  }

  size_t end = input.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(input[end - 1]))) {
    --end;
  }

  std::string token = input.substr(begin, end - begin);
  std::transform(token.begin(), token.end(), token.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return token;
}

inline char hex_digit(unsigned value) {
  return static_cast<char>((value < 10) ? ('0' + value)
                                        : ('a' + (value - 10)));
}

inline std::string encode_hex_bytes(const char *data, size_t size) {
  std::string hex;
  hex.reserve(size * 2);
  for (size_t i = 0; i < size; ++i) {
    const unsigned char byte = static_cast<unsigned char>(data[i]);
    hex.push_back(hex_digit(byte >> 4));
    hex.push_back(hex_digit(byte & 0x0f));
  }
  return hex;
}

inline bool hex_value(char ch, unsigned *value) {
  if (value == nullptr) return false;
  if (ch >= '0' && ch <= '9') {
    *value = static_cast<unsigned>(ch - '0');
    return true;
  }
  if (ch >= 'a' && ch <= 'f') {
    *value = static_cast<unsigned>(10 + ch - 'a');
    return true;
  }
  if (ch >= 'A' && ch <= 'F') {
    *value = static_cast<unsigned>(10 + ch - 'A');
    return true;
  }
  return false;
}

inline bool decode_hex_bytes(const std::string &input, std::string *output) {
  if (output == nullptr) return false;
  if ((input.size() % 2) != 0) return false;
  output->clear();
  output->reserve(input.size() / 2);
  for (size_t i = 0; i < input.size(); i += 2) {
    unsigned hi = 0;
    unsigned lo = 0;
    if (!hex_value(input[i], &hi) || !hex_value(input[i + 1], &lo))
      return false;
    output->push_back(static_cast<char>((hi << 4) | lo));
  }
  return true;
}

inline bool parse_uint64(const std::string &text, uint64_t *value) {
  if (value == nullptr || text.empty()) return false;
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0') return false;
  *value = static_cast<uint64_t>(parsed);
  return true;
}

inline bool starts_with(const std::string &text, const std::string &prefix) {
  return text.size() >= prefix.size() &&
         text.compare(0, prefix.size(), prefix) == 0;
}

inline bool ends_with(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace vector_index::detail

#endif  // SQL_VECTOR_INDEX_BACKEND_COMMON_INCLUDED
