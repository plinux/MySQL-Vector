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

#include <cstdint>
#include <limits>
#include <string>

#include "sql/mysqld.h"
#include "mysqld_error.h"
#include "sql/vector/item_vectorfunc_internal.h"
#include "sql/vector/vector_index_build_options.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_limits.h"

longlong Item_func_vec_index_create::val_int() {
  assert(fixed && arg_count >= 2 && arg_count <= 6);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  const longlong dim_ll = args[1]->val_int();
  if (name == nullptr || args[0]->null_value || args[1]->null_value ||
      dim_ll <= 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  std::string metric = "euclidean";
  std::string mode = "memory";
  std::string provider = "native";
  vector_index_registry::create_index_options options;

  String metric_buf;
  String mode_buf;
  String provider_buf;
  if (arg_count >= 3) {
    const String *metric_arg = args[2]->val_str(&metric_buf);
    if (metric_arg == nullptr || args[2]->null_value) return error_int();
    to_std_string(metric_arg, &metric);
  }
  if (arg_count >= 4) {
    const String *mode_arg = args[3]->val_str(&mode_buf);
    if (mode_arg == nullptr || args[3]->null_value) return error_int();
    to_std_string(mode_arg, &mode);
  }
  if (arg_count >= 5) {
    const String *provider_arg = args[4]->val_str(&provider_buf);
    if (provider_arg == nullptr || args[4]->null_value) return error_int();
    to_std_string(provider_arg, &provider);
  }
  if (arg_count >= 6) {
    const longlong build_threads_ll = args[5]->val_int();
    if (args[5]->null_value || build_threads_ll < 0 ||
        static_cast<unsigned long long>(build_threads_ll) >
            vector_index::k_max_build_threads) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
    options.build_threads_specified = true;
    options.build_threads = static_cast<uint32_t>(build_threads_ll);
  }

  if (!vector_index_registry::create_index(index_name,
                                           static_cast<size_t>(dim_ll), metric,
                                           mode, provider, options)) {
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
  assert(fixed && arg_count == 2);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  const longlong search_ef_ll = args[1]->val_int();
  if (name == nullptr || args[0]->null_value || args[1]->null_value ||
      search_ef_ll <= 0 ||
      static_cast<unsigned long long>(search_ef_ll) >
          std::numeric_limits<uint32_t>::max()) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (!vector_index_registry::set_search_ef(
          index_name, static_cast<uint32_t>(search_ef_ll))) {
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
  assert(fixed && arg_count == 3);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  const longlong hnsw_m_ll = args[1]->val_int();
  const longlong hnsw_ef_construction_ll = args[2]->val_int();
  if (name == nullptr || args[0]->null_value || args[1]->null_value ||
      args[2]->null_value || hnsw_m_ll <= 0 || hnsw_ef_construction_ll <= 0 ||
      static_cast<unsigned long long>(hnsw_m_ll) >
          std::numeric_limits<uint32_t>::max() ||
      static_cast<unsigned long long>(hnsw_ef_construction_ll) >
          std::numeric_limits<uint32_t>::max()) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (!vector_index_registry::set_hnsw_build_params(
          index_name, static_cast<uint32_t>(hnsw_m_ll),
          static_cast<uint32_t>(hnsw_ef_construction_ll))) {
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
  assert(fixed && arg_count == 3);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  const longlong faiss_nlist_ll = args[1]->val_int();
  const longlong faiss_nprobe_ll = args[2]->val_int();
  if (name == nullptr || args[0]->null_value || args[1]->null_value ||
      args[2]->null_value || faiss_nlist_ll < 0 || faiss_nprobe_ll < 0 ||
      static_cast<unsigned long long>(faiss_nlist_ll) >
          std::numeric_limits<uint32_t>::max() ||
      static_cast<unsigned long long>(faiss_nprobe_ll) >
          std::numeric_limits<uint32_t>::max()) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (!vector_index_registry::set_faiss_ivf_params(
          index_name, static_cast<uint32_t>(faiss_nlist_ll),
          static_cast<uint32_t>(faiss_nprobe_ll))) {
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
  assert(fixed && arg_count == 5);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  const longlong faiss_nlist_ll = args[1]->val_int();
  const longlong faiss_nprobe_ll = args[2]->val_int();
  const longlong faiss_pq_m_ll = args[3]->val_int();
  const longlong faiss_pq_bits_ll = args[4]->val_int();
  if (name == nullptr || args[0]->null_value || args[1]->null_value ||
      args[2]->null_value || args[3]->null_value || args[4]->null_value ||
      faiss_nlist_ll <= 0 || faiss_nprobe_ll <= 0 || faiss_pq_m_ll <= 0 ||
      faiss_pq_bits_ll <= 0 ||
      static_cast<unsigned long long>(faiss_nlist_ll) >
          std::numeric_limits<uint32_t>::max() ||
      static_cast<unsigned long long>(faiss_nprobe_ll) >
          std::numeric_limits<uint32_t>::max() ||
      static_cast<unsigned long long>(faiss_pq_m_ll) >
          std::numeric_limits<uint32_t>::max() ||
      static_cast<unsigned long long>(faiss_pq_bits_ll) >
          std::numeric_limits<uint32_t>::max()) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (!vector_index_registry::set_faiss_ivf_pq_params(
          index_name, static_cast<uint32_t>(faiss_nlist_ll),
          static_cast<uint32_t>(faiss_nprobe_ll),
          static_cast<uint32_t>(faiss_pq_m_ll),
          static_cast<uint32_t>(faiss_pq_bits_ll))) {
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
  assert(fixed && arg_count >= 3 && arg_count <= 4);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  const longlong diskann_max_degree_ll = args[1]->val_int();
  const longlong diskann_build_complexity_ll = args[2]->val_int();
  const longlong diskann_build_threads_ll =
      arg_count >= 4 ? args[3]->val_int() : 0;
  if (name == nullptr || args[0]->null_value || args[1]->null_value ||
      args[2]->null_value || diskann_max_degree_ll <= 0 ||
      diskann_build_complexity_ll <= 0 ||
      (arg_count >= 4 &&
       (args[3]->null_value || diskann_build_threads_ll < 0 ||
        static_cast<unsigned long long>(diskann_build_threads_ll) >
            vector_index::k_max_build_threads)) ||
      static_cast<unsigned long long>(diskann_max_degree_ll) >
          std::numeric_limits<uint32_t>::max() ||
      static_cast<unsigned long long>(diskann_build_complexity_ll) >
          std::numeric_limits<uint32_t>::max()) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  const uint32_t diskann_build_threads =
      arg_count >= 4 && diskann_build_threads_ll != 0
          ? static_cast<uint32_t>(diskann_build_threads_ll)
          : static_cast<uint32_t>(opt_vector_diskann_build_threads);
  if (!vector_index_registry::set_diskann_build_params(
          index_name, static_cast<uint32_t>(diskann_max_degree_ll),
          static_cast<uint32_t>(diskann_build_complexity_ll),
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
  assert(fixed && arg_count == 2);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  const longlong diskann_search_complexity_ll = args[1]->val_int();
  if (name == nullptr || args[0]->null_value || args[1]->null_value ||
      diskann_search_complexity_ll <= 0 ||
      static_cast<unsigned long long>(diskann_search_complexity_ll) >
          std::numeric_limits<uint32_t>::max()) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (!vector_index_registry::set_diskann_search_complexity(
          index_name, static_cast<uint32_t>(diskann_search_complexity_ll))) {
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
  assert(fixed && arg_count == 1);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value) return error_int();

  std::string index_name;
  to_std_string(name, &index_name);
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
  assert(fixed && arg_count == 1);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (!vector_index_registry::rebuild_index(index_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
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
  assert(fixed && arg_count == 1);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
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
  assert(fixed && arg_count == 1);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (!vector_index_registry::bulk_build_index(index_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
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
  assert(fixed && arg_count == 1);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  if (name == nullptr || args[0]->null_value) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
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
  assert(fixed && arg_count == 0);
  null_value = true;

  size_t rebuilt_count = 0;
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
  assert(fixed && arg_count == 0);
  null_value = true;

  size_t recovered_count = 0;
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
  assert(fixed && arg_count == 3);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  const longlong doc_id_ll = args[1]->val_int();
  if (name == nullptr || args[0]->null_value || args[1]->null_value ||
      doc_id_ll < 0) {
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
  if (!vector_index_registry::upsert(index_name, static_cast<uint64_t>(doc_id_ll),
                                     vector)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();

  null_value = false;
  return 1;
}

bool Item_func_vec_index_erase::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 2)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_erase::val_int() {
  assert(fixed && arg_count == 2);
  null_value = true;

  String name_buf;
  const String *name = args[0]->val_str(&name_buf);
  const longlong doc_id_ll = args[1]->val_int();
  if (name == nullptr || args[0]->null_value || args[1]->null_value ||
      doc_id_ll < 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  std::string index_name;
  to_std_string(name, &index_name);
  if (!vector_index_registry::erase(index_name, static_cast<uint64_t>(doc_id_ll))) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!maybe_binlog_vector_write_query(current_thd)) return error_int();

  null_value = false;
  return 1;
}
