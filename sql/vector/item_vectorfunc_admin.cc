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
#include <cstdint>
#include <limits>
#include <string>

#include "my_byteorder.h"
#include "sql/derror.h"  // ER_THD
#include "sql/mysqld.h"
#include "sql/sql_error.h"
#include "sql/sql_class.h"
#include "mysqld_error.h"
#include "sql/vector/item_vectorfunc_internal.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_limits.h"
#include "sql/vector/vector_utils.h"

using namespace vector_itemfunc_internal;

static bool eval_bool01_arg(Item *arg, bool *value) {
  if (arg == nullptr || value == nullptr) return false;
  const longlong parsed = arg->val_int();
  if (arg->null_value || (parsed != 0 && parsed != 1)) return false;
  *value = parsed != 0;
  return true;
}

longlong Item_func_vec_index_create::val_int() {
  assert_fixed_arg_count_between(fixed, arg_count, 2, 7);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  ulonglong dim = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint_arg(args[1], dim) || dim == 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (dim > vector_index::k_max_vector_dimension) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  std::string owner_schema;
  if (resolve_vector_current_db(current_thd, &owner_schema) ||
      check_vector_schema_access(current_thd, owner_schema, CREATE_ACL)) {
    return error_int();
  }

  std::string metric = "euclidean";
  std::string mode;
  std::string provider;
  vector_index_registry::create_index_options options;

  String metric_buf;
  String mode_buf;
  String provider_buf;
  String consistency_buf;
  if (arg_count >= 3) {
    const String *metric_arg = args[2]->val_str(&metric_buf);
    if (metric_arg == nullptr || args[2]->null_value) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
    to_std_string(metric_arg, &metric);
  }
  if (arg_count >= 4) {
    const String *mode_arg = args[3]->val_str(&mode_buf);
    if (mode_arg == nullptr || args[3]->null_value) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
    to_std_string(mode_arg, &mode);
    if (mode.empty()) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
  }
  if (arg_count >= 5) {
    const String *provider_arg = args[4]->val_str(&provider_buf);
    if (provider_arg == nullptr || args[4]->null_value) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
    to_std_string(provider_arg, &provider);
    if (provider.empty()) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
  }
  if (arg_count >= 6) {
    uint32_t build_threads = 0;
    if (!eval_uint32_arg(args[5], 0, vector_index::k_max_build_threads,
                         build_threads)) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
    options.build_threads_specified = true;
    options.build_threads = build_threads;
  }
  if (arg_count >= 7) {
    const String *consistency_arg = args[6]->val_str(&consistency_buf);
    if (consistency_arg == nullptr || args[6]->null_value) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }

    std::string consistency_text;
    to_std_string(consistency_arg, &consistency_text);
    vector_index::index_consistency_mode consistency_mode;
    if (!vector_index::parse_index_consistency_mode(consistency_text,
                                                    &consistency_mode)) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
    options.consistency_mode_specified = true;
    options.consistency_mode = consistency_mode;
  }
  options.diskann_max_degree =
      static_cast<uint32_t>(opt_vector_diskann_max_degree);
  options.diskann_build_complexity =
      static_cast<uint32_t>(opt_vector_diskann_build_complexity);

  if (!vector_index_registry::create_index(index_name, static_cast<size_t>(dim),
                                           metric, mode, provider, owner_schema,
                                           options)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_search_ef::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_search_ef::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  uint32_t search_ef = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint32_arg(args[1], 1, std::numeric_limits<uint32_t>::max(),
                       search_ef)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::set_search_ef(index_name, search_ef)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_hnsw_build_params::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 3)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_hnsw_build_params::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 3);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  uint32_t hnsw_m = 0;
  uint32_t hnsw_ef_construction = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint32_arg(args[1], 1, std::numeric_limits<uint32_t>::max(),
                       hnsw_m) ||
      !eval_uint32_arg(args[2], 1, std::numeric_limits<uint32_t>::max(),
                       hnsw_ef_construction)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::set_hnsw_build_params(
          index_name, hnsw_m, hnsw_ef_construction)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_faiss_ivf_params::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 3)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_faiss_ivf_params::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 3);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  uint32_t faiss_nlist = 0;
  uint32_t faiss_nprobe = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint32_arg(args[1], 0, std::numeric_limits<uint32_t>::max(),
                       faiss_nlist) ||
      !eval_uint32_arg(args[2], 0, std::numeric_limits<uint32_t>::max(),
                       faiss_nprobe)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::set_faiss_ivf_params(
          index_name, faiss_nlist, faiss_nprobe)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_faiss_ivfpq_params::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 5)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_faiss_ivfpq_params::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 5);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  uint32_t faiss_nlist = 0;
  uint32_t faiss_nprobe = 0;
  uint32_t faiss_pq_m = 0;
  uint32_t faiss_pq_bits = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint32_arg(args[1], 1, std::numeric_limits<uint32_t>::max(),
                       faiss_nlist) ||
      !eval_uint32_arg(args[2], 1, std::numeric_limits<uint32_t>::max(),
                       faiss_nprobe) ||
      !eval_uint32_arg(args[3], 1, std::numeric_limits<uint32_t>::max(),
                       faiss_pq_m) ||
      !eval_uint32_arg(args[4], 1, vector_index::k_max_faiss_pq_bits,
                       faiss_pq_bits)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::set_faiss_ivf_pq_params(
          index_name, faiss_nlist, faiss_nprobe, faiss_pq_m,
          faiss_pq_bits)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_diskann_build_params::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, arg_count)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_diskann_build_params::val_int() {
  assert_fixed_arg_count_between(fixed, arg_count, 3, 4);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  uint32_t diskann_max_degree = 0;
  uint32_t diskann_build_complexity = 0;
  uint32_t diskann_build_threads_arg = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint32_arg(args[1], 1, std::numeric_limits<uint32_t>::max(),
                       diskann_max_degree) ||
      !eval_uint32_arg(args[2], 1, std::numeric_limits<uint32_t>::max(),
                       diskann_build_complexity) ||
      (arg_count >= 4 &&
       !eval_uint32_arg(args[3], 0, vector_index::k_max_build_threads,
                        diskann_build_threads_arg))) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  const uint32_t diskann_build_threads =
      arg_count >= 4 && diskann_build_threads_arg != 0
          ? diskann_build_threads_arg
          : static_cast<uint32_t>(opt_vector_diskann_build_threads);
  if (!vector_index_registry::set_diskann_build_params(
          index_name, diskann_max_degree, diskann_build_complexity,
          diskann_build_threads)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_diskann_search_complexity::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_diskann_search_complexity::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  uint32_t diskann_search_complexity = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint32_arg(args[1], 1, std::numeric_limits<uint32_t>::max(),
                       diskann_search_complexity)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::set_diskann_search_complexity(
          index_name, diskann_search_complexity)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_diskann_search_beamwidth::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_diskann_search_beamwidth::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  uint32_t diskann_search_beamwidth = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint32_arg(args[1], 1,
                       vector_index::k_max_diskann_search_beamwidth,
                       diskann_search_beamwidth)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::set_diskann_search_beamwidth(
          index_name, diskann_search_beamwidth)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_diskann_pq_code_budget_size::resolve_type(
    THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_diskann_pq_code_budget_size::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  ulonglong diskann_pq_code_budget_size = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint_arg(args[1], diskann_pq_code_budget_size)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::set_diskann_pq_code_budget_size(
          index_name, diskann_pq_code_budget_size)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_diskann_disk_pq_dims::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_diskann_disk_pq_dims::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  uint32_t diskann_disk_pq_dims = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint32_arg(args[1], 0,
                       vector_index::k_max_diskann_disk_pq_dims,
                       diskann_disk_pq_dims)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::set_diskann_disk_pq_dims(
          index_name, diskann_disk_pq_dims)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_diskann_accelerate_build::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_diskann_accelerate_build::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  bool accelerate_build = false;
  if (name == nullptr || args[0]->null_value ||
      !eval_bool01_arg(args[1], &accelerate_build)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::set_diskann_accelerate_build(
          index_name, accelerate_build)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_diskann_shuffle_build::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_diskann_shuffle_build::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  bool shuffle_build = false;
  if (name == nullptr || args[0]->null_value ||
      !eval_bool01_arg(args[1], &shuffle_build)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::set_diskann_shuffle_build(index_name,
                                                        shuffle_build)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_diskann_use_bfs_cache::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_diskann_use_bfs_cache::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  bool use_bfs_cache = false;
  if (name == nullptr || args[0]->null_value ||
      !eval_bool01_arg(args[1], &use_bfs_cache)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::set_diskann_use_bfs_cache(index_name,
                                                        use_bfs_cache)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_set_diskann_build_mode::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_set_diskann_build_mode::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  String mode_buf;
  const String *name = args[0]->val_str(&name_buf);
  const String *mode = args[1]->val_str(&mode_buf);
  if (name == nullptr || mode == nullptr || args[0]->null_value ||
      args[1]->null_value) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  std::string build_mode_text;
  to_std_string(name, &index_name);
  to_std_string(mode, &build_mode_text);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();

  vector_index::diskann_build_mode build_mode;
  if (!vector_index::parse_diskann_build_mode(build_mode_text, &build_mode)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  vector_index_registry::index_info info;
  if (!vector_index_registry::get_index_info(index_name, &info)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (info.provider != "diskann" || info.mode != "external") {
    null_value = false;
    if (build_mode == vector_index::diskann_build_mode::kAuto) return 1;
    push_warning_printf(current_thd, Sql_condition::SL_WARNING,
                        ER_WRONG_ARGUMENTS,
                        ER_THD(current_thd, ER_WRONG_ARGUMENTS), func_name());
    return 0;
  }

  if (!vector_index_registry::set_diskann_build_mode(index_name, build_mode)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_drop::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_drop::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, DROP_ACL, func_name(), true))
    return error_int();
  const bool dropped = vector_index_registry::drop_index(index_name);
  if (dropped && !maybe_binlog_vector_write_query(current_thd)) return error_int();
  null_value = false;
  return dropped ? 1 : 0;
}

bool Item_func_vec_index_rebuild::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_rebuild::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  std::string error;
  if (!vector_index_registry::rebuild_index(index_name, &error)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0),
             error.empty() ? func_name() : error.c_str());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();

  null_value = false;
  return 1;
}

bool Item_func_vec_index_bulk_load_begin::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_bulk_load_begin::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::begin_bulk_load(index_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();

  null_value = false;
  return 1;
}

bool Item_func_vec_index_bulk_build::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_bulk_build::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  std::string error;
  if (!vector_index_registry::bulk_build_index(index_name, &error)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0),
             error.empty() ? func_name() : error.c_str());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();

  null_value = false;
  return 1;
}

bool Item_func_vec_index_recover::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_recover::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::recover_index(index_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();

  null_value = false;
  return 1;
}

bool Item_func_vec_index_rebuild_all::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 0)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_rebuild_all::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 0);
  null_value = true;

  size_t rebuilt_count = 0;
  if (check_vector_all_indexes_access(current_thd, ALTER_ACL, func_name()))
    return error_int();
  if (!vector_index_registry::rebuild_all_indexes(&rebuilt_count)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();

  null_value = false;
  return static_cast<longlong>(rebuilt_count);
}

bool Item_func_vec_index_recover_all::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 0)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_recover_all::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 0);
  null_value = true;

  size_t recovered_count = 0;
  if (check_vector_all_indexes_access(current_thd, ALTER_ACL, func_name()))
    return error_int();
  if (!vector_index_registry::recover_all_indexes(&recovered_count)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();

  null_value = false;
  return static_cast<longlong>(recovered_count);
}

bool Item_func_vec_index_upsert::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 3)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_upsert::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 3);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  ulonglong doc_id = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint_arg(args[1], doc_id)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  vector_index::vector_data vector;
  String vector_buf;
  if (!decode_vector_arg(args[2], &vector_buf, &vector)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::upsert(index_name, static_cast<uint64_t>(doc_id),
                                     vector)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();

  null_value = false;
  return 1;
}

bool Item_func_vec_index_upsert_batch::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 4)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_upsert_batch::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 4);
  null_value = true;

  String name_buf;
  String docid_buf;
  String vector_buf;
  const String *name = args[0]->val_str(&name_buf);
  const String *docid_blob = args[1]->val_str(&docid_buf);
  const String *vector_blob = args[2]->val_str(&vector_buf);
  ulonglong row_count_value = 0;
  if (name == nullptr || args[0]->null_value || docid_blob == nullptr ||
      args[1]->null_value || vector_blob == nullptr || args[2]->null_value ||
      !eval_uint_arg(args[3], row_count_value) || row_count_value == 0 ||
      row_count_value > std::numeric_limits<size_t>::max()) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();

  vector_index_registry::index_info info;
  if (!vector_index_registry::get_index_info(index_name, &info) ||
      info.dimension == 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  const size_t row_count = static_cast<size_t>(row_count_value);
  if (row_count > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  const size_t expected_docid_bytes = row_count * sizeof(uint64_t);
  if (docid_blob->length() != expected_docid_bytes) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  const size_t dimension = info.dimension;
  if (row_count > std::numeric_limits<size_t>::max() / dimension ||
      row_count * dimension >
          std::numeric_limits<size_t>::max() /
              vector_utils::kVectorElemSize) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  const size_t expected_vector_bytes =
      row_count * dimension * vector_utils::kVectorElemSize;
  if (vector_blob->length() != expected_vector_bytes) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  const uchar *docid_ptr =
      reinterpret_cast<const uchar *>(docid_blob->ptr());
  const uchar *vector_ptr =
      reinterpret_cast<const uchar *>(vector_blob->ptr());
  vector_index::index_service::bulk_load_reader reader =
      [docid_ptr, vector_ptr, row_count,
       dimension](const vector_index::index_service::bulk_load_visitor
                      &visitor,
                  std::string *error) {
        vector_index::vector_data row;
        row.resize(dimension);
        for (size_t row_idx = 0; row_idx < row_count; ++row_idx) {
          if (current_thd != nullptr && current_thd->killed) {
            if (error != nullptr) *error = "VEC_INDEX_UPSERT_BATCH killed";
            return false;
          }
          const uint64_t doc_id =
              uint8korr(docid_ptr + row_idx * sizeof(uint64_t));
          const uchar *row_ptr =
              vector_ptr + row_idx * dimension * vector_utils::kVectorElemSize;
          for (size_t dim_idx = 0; dim_idx < dimension; ++dim_idx) {
            const float value =
                float4get(row_ptr + dim_idx * vector_utils::kVectorElemSize);
            if (!std::isfinite(value)) {
              if (error != nullptr) {
                *error = "VEC_INDEX_UPSERT_BATCH non-finite vector value";
              }
              return false;
            }
            row[dim_idx] = value;
          }
          if (!visitor(doc_id, row.data(), row.size())) return false;
        }
        return true;
      };

  vector_index::index_service::bulk_load_options options;
  options.replace_duplicates = true;
  options.rebuild_after_load = false;
  options.source_format = "BINARY_BLOB";
  std::string error;
  if (!vector_index_registry::bulk_upsert_from_reader(index_name, reader,
                                                      options, &error)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0),
             error.empty() ? func_name() : error.c_str());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();

  null_value = false;
  return static_cast<longlong>(row_count);
}

bool Item_func_vec_index_erase::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_erase::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 2);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  ulonglong doc_id = 0;
  if (name == nullptr || args[0]->null_value ||
      !eval_uint_arg(args[1], doc_id)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!vector_index_registry::erase(index_name,
                                    static_cast<uint64_t>(doc_id))) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();

  null_value = false;
  return 1;
}
