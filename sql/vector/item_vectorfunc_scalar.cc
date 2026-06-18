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

#include "sql/vector/item_vectorfunc.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "my_dbug.h"
#include "my_byteorder.h"
#include "mysqld_error.h"
#include "sql/item.h"
#include "sql/vector/item_vectorfunc_internal.h"
#include "sql/vector/vector_utils.h"

using namespace vector_itemfunc_internal;

namespace {

bool eval_scalar_vector_arg(Item *arg, String *buf, const String **value) {
  *value = arg->val_str(buf);
  if (*value == nullptr || arg->null_value) return false;
  return true;
}

template <typename Operation>
bool eval_binary_vector_function(Item_real_func *item, Item *lhs_arg,
                                 Item *rhs_arg, Operation operation,
                                 double *result) {
  item->null_value = true;

  String lhs_buf;
  String rhs_buf;
  const String *lhs = nullptr;
  const String *rhs = nullptr;
  if (!eval_scalar_vector_arg(lhs_arg, &lhs_buf, &lhs) ||
      !eval_scalar_vector_arg(rhs_arg, &rhs_buf, &rhs)) {
    return false;
  }

  if (!operation(lhs, rhs, result)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), item->func_name());
    return false;
  }

  item->null_value = false;
  return true;
}

}  // namespace

bool Item_func_vec_fromtext::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_data_type_blob(static_cast<uint32>(MAX_BLOB_WIDTH));
  collation.set(&my_charset_bin, DERIVATION_COERCIBLE, MY_REPERTOIRE_ASCII);
  set_nullable(true);
  return false;
}

String *Item_func_vec_fromtext::val_str(String *str [[maybe_unused]]) {
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  String input_buf;
  const String *input = args[0]->val_str(&input_buf);
  if (input == nullptr || args[0]->null_value) return nullptr;

  std::vector<float> values;
  if (!vector_utils::parse_text_vector(input, &values)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  const size_t byte_length = values.size() * vector_utils::kVectorElemSize;
  DBUG_EXECUTE_IF("vector_item_fail_fromtext_alloc", return error_str(););
  if (byte_length > 0 && m_value.mem_realloc(byte_length)) return error_str();

  m_value.length(0);
  m_value.set_charset(&my_charset_bin);
  for (size_t i = 0; i < values.size(); ++i) {
    float4store(m_value.ptr() + i * vector_utils::kVectorElemSize, values[i]);
  }
  m_value.length(byte_length);
  null_value = false;
  return &m_value;
}

bool Item_func_vec_totext::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  const uint32 max_chars =
      static_cast<uint32>(MAX_BLOB_WIDTH / default_charset()->mbmaxlen);
  set_data_type_string(max_chars, default_charset());
  set_nullable(true);
  return false;
}

String *Item_func_vec_totext::val_str(String *str [[maybe_unused]]) {
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  String input_buf;
  const String *input = args[0]->val_str(&input_buf);
  if (input == nullptr || args[0]->null_value) return nullptr;

  size_t dim = 0;
  if (!vector_utils::parse_binary_vector(input, &dim)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  m_value.length(0);
  m_value.set_charset(default_charset());
  if (m_value.append('[')) return error_str();

  const uchar *ptr = reinterpret_cast<const uchar *>(input->ptr());
  for (size_t i = 0; i < dim; ++i) {
    if (i > 0 && m_value.append(',')) return error_str();
    const double value =
        static_cast<double>(float4get(ptr + i * vector_utils::kVectorElemSize));
    DBUG_EXECUTE_IF("vector_item_fail_totext_number_format", return error_str(););
    if (m_number_buf.set_real(value, DECIMAL_NOT_SPECIFIED, &my_charset_bin))
      return error_str();
    if (m_value.append(m_number_buf.ptr(), m_number_buf.length()))
      return error_str();
  }

  if (m_value.append(']')) return error_str();
  null_value = false;
  return &m_value;
}

bool Item_func_vec_normalize::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_data_type_blob(static_cast<uint32>(MAX_BLOB_WIDTH));
  collation.set(&my_charset_bin, DERIVATION_COERCIBLE, MY_REPERTOIRE_ASCII);
  set_nullable(true);
  return false;
}

String *Item_func_vec_normalize::val_str(String *str [[maybe_unused]]) {
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  String input_buf;
  const String *input = args[0]->val_str(&input_buf);
  if (input == nullptr || args[0]->null_value) return nullptr;

  size_t dim = 0;
  if (!vector_utils::parse_binary_vector(input, &dim)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  const uchar *ptr = reinterpret_cast<const uchar *>(input->ptr());
  double norm2 = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    const double value =
        static_cast<double>(float4get(ptr + i * vector_utils::kVectorElemSize));
    norm2 += value * value;
  }
  if (norm2 <= 0.0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_str();
  }

  const double inv_norm = 1.0 / std::sqrt(norm2);
  const size_t byte_length = dim * vector_utils::kVectorElemSize;
  DBUG_EXECUTE_IF("vector_item_fail_normalize_alloc", return error_str(););
  if (byte_length > 0 && m_value.mem_realloc(byte_length)) return error_str();

  m_value.length(0);
  m_value.set_charset(&my_charset_bin);
  for (size_t i = 0; i < dim; ++i) {
    const double value =
        static_cast<double>(
            float4get(ptr + i * vector_utils::kVectorElemSize)) *
        inv_norm;
    float4store(m_value.ptr() + i * vector_utils::kVectorElemSize,
                static_cast<float>(value));
  }
  m_value.length(byte_length);
  null_value = false;
  return &m_value;
}

bool Item_func_vector_dim::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_nullable(true);
  return false;
}

longlong Item_func_vector_dim::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  String input_buf;
  const String *input = args[0]->val_str(&input_buf);
  if (input == nullptr || args[0]->null_value) return 0;

  size_t dim = 0;
  if (!vector_utils::parse_binary_vector(input, &dim)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  null_value = false;
  return static_cast<longlong>(dim);
}

bool Item_func_vec_dot_product::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(true);
  return false;
}

double Item_func_vec_dot_product::val_real() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  double result = 0.0;
  if (!eval_binary_vector_function(this, args[0], args[1],
                                   vector_utils::compute_dot_product,
                                   &result)) {
    return error_real();
  }
  return result;
}

bool Item_func_vec_inner_product::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(true);
  return false;
}

double Item_func_vec_inner_product::val_real() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  double result = 0.0;
  if (!eval_binary_vector_function(this, args[0], args[1],
                                   vector_utils::compute_dot_product,
                                   &result)) {
    return error_real();
  }
  return result;
}

bool Item_func_vec_distance_euclidean::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(true);
  return false;
}

double Item_func_vec_distance_euclidean::val_real() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  double result = 0.0;
  if (!eval_binary_vector_function(
          this, args[0], args[1],
          [](const String *lhs, const String *rhs, double *distance) {
            return vector_utils::compute_distance(
                vector_utils::distance_metric::kEuclidean, lhs, rhs,
                distance);
          },
          &result)) {
    return error_real();
  }
  return result;
}

bool Item_func_vec_distance_cosine::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(true);
  return false;
}

double Item_func_vec_distance_cosine::val_real() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  double result = 0.0;
  if (!eval_binary_vector_function(
          this, args[0], args[1],
          [](const String *lhs, const String *rhs, double *distance) {
            return vector_utils::compute_distance(
                vector_utils::distance_metric::kCosine, lhs, rhs, distance);
          },
          &result)) {
    return error_real();
  }
  return result;
}

bool Item_func_vec_distance::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, arg_count)) return true;
  set_nullable(true);
  return false;
}

double Item_func_vec_distance::val_real() {
  assert_fixed_arg_count_is_one_of(fixed, arg_count, 2, 3);
  null_value = true;

  String lhs_buf;
  String rhs_buf;
  const String *lhs = nullptr;
  const String *rhs = nullptr;
  if (!eval_scalar_vector_arg(args[0], &lhs_buf, &lhs) ||
      !eval_scalar_vector_arg(args[1], &rhs_buf, &rhs)) {
    return 0.0;
  }

  vector_utils::distance_metric metric =
      vector_utils::distance_metric::kEuclidean;
  if (arg_count == 3) {
    String metric_buf;
    const String *metric_arg = args[2]->val_str(&metric_buf);
    if (metric_arg == nullptr || args[2]->null_value) return 0.0;
    if (!vector_utils::parse_distance_metric(metric_arg, &metric)) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_real();
    }
  }

  double distance = 0.0;
  if (!vector_utils::compute_distance(metric, lhs, rhs, &distance)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_real();
  }

  null_value = false;
  return distance;
}
