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
#include <utility>
#include <vector>

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
#include "sql/vector/vector_statement_publication.h"
#include "sql/vector/vector_status.h"
#include "sql/vector/vector_trx_participant.h"
#include "sql/vector/vector_utils.h"

using namespace vector_itemfunc_internal;

static bool eval_bool01_arg(Item *arg, bool *value) {
  if (arg == nullptr || value == nullptr) return false;
  const longlong parsed = arg->val_int();
  if (arg->null_value || (parsed != 0 && parsed != 1)) return false;
  *value = parsed != 0;
  return true;
}

static bool eval_string_arg(Item *arg, String *buffer, std::string *value) {
  if (arg == nullptr || buffer == nullptr || value == nullptr) return false;
  const String *parsed = arg->val_str(buffer);
  if (parsed == nullptr || arg->null_value) return false;
  to_std_string(parsed, value);
  return true;
}

static bool encode_statement_payload(
    const vector_statement_publication::operation_payload &payload,
    const char *function_name, std::string *encoded) {
  if (vector_statement_publication::encode_operation_payload(payload,
                                                             encoded)) {
    return true;
  }
  my_error(ER_INTERNAL_ERROR, MYF(0), function_name);
  return false;
}

static bool stage_and_binlog_statement(
    vector_index_truth_store::publication_operation operation,
    const std::string &index_name, const std::string &payload,
    bool catalog_exclusive, const char *function_name,
    const char *argument_error = nullptr) {
  if (current_thd == nullptr ||
      vector_trx_participant::in_user_multi_statement_transaction(
          current_thd)) {
    my_error(ER_LOCK_OR_ACTIVE_TRANSACTION, MYF(0));
    return false;
  }
  if (!stage_vector_statement_publication(current_thd, operation, index_name,
                                          payload, catalog_exclusive)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0),
             argument_error == nullptr ? function_name : argument_error);
    return false;
  }
  if (maybe_binlog_vector_write_query(current_thd)) return true;
  if (!current_thd->is_error())
    my_error(ER_INTERNAL_ERROR, MYF(0), function_name);
  return false;
}

static bool stage_config_statement(
    const std::string &index_name,
    vector_statement_publication::config_change config,
    std::vector<uint64_t> values, const char *function_name) {
  vector_statement_publication::operation_payload payload;
  payload.config = config;
  payload.unsigned_values = std::move(values);
  std::string encoded;
  return encode_statement_payload(payload, function_name, &encoded) &&
         stage_and_binlog_statement(
             vector_index_truth_store::publication_operation::kUpdateConfig,
             index_name, encoded, false, function_name);
}

static bool stage_diskann_boolean_config(
    Item *index_arg, Item *value_arg,
    vector_statement_publication::config_change config,
    const char *function_name) {
  String name_buffer;
  std::string index_name;
  bool value = false;
  if (!eval_string_arg(index_arg, &name_buffer, &index_name) ||
      !eval_bool01_arg(value_arg, &value)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), function_name);
    return false;
  }
  if (check_vector_existing_index_access(current_thd, index_name, ALTER_ACL,
                                         function_name, false)) {
    return false;
  }
  return stage_config_statement(index_name, config, {value ? 1U : 0U},
                                function_name);
}

static bool stage_index_statement(
    const std::string &index_name,
    vector_index_truth_store::publication_operation operation,
    bool catalog_exclusive, const char *function_name,
    const char *argument_error = nullptr) {
  return stage_and_binlog_statement(operation, index_name, std::string(),
                                    catalog_exclusive, function_name,
                                    argument_error);
}

static bool stage_unary_index_statement(
    Item *index_arg, vector_index_truth_store::publication_operation operation,
    const char *function_name, const char *argument_error = nullptr) {
  String name_buffer;
  std::string index_name;
  if (!eval_string_arg(index_arg, &name_buffer, &index_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), function_name);
    return false;
  }
  if (check_vector_existing_index_access(current_thd, index_name, ALTER_ACL,
                                         function_name, false)) {
    return false;
  }
  return stage_index_statement(index_name, operation, false, function_name,
                               argument_error);
}

static bool stage_all_index_statements(
    vector_index_truth_store::publication_operation operation,
    const char *function_name, size_t *index_count) {
  if (index_count == nullptr || current_thd == nullptr ||
      vector_trx_participant::in_user_multi_statement_transaction(
          current_thd)) {
    my_error(ER_LOCK_OR_ACTIVE_TRANSACTION, MYF(0));
    return false;
  }
  std::vector<std::string> index_names;
  if (!vector_index_registry::list_indexes(&index_names)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), function_name);
    return false;
  }
  *index_count = index_names.size();
  if (index_names.empty()) return true;

  std::vector<vector_index_truth_store::publication_intent> intents;
  intents.reserve(index_names.size());
  for (const std::string &index_name : index_names) {
    vector_index_truth_store::publication_intent intent;
    if (!vector_index_registry::make_statement_publication_intent(
            operation, index_name, std::string(), &intent)) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), function_name);
      return false;
    }
    intents.push_back(std::move(intent));
  }
  if (!vector_trx_participant::stage_statement_publication(
          current_thd, std::move(intents), true, true)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), function_name);
    return false;
  }
  if (maybe_binlog_vector_write_query(current_thd)) return true;
  if (!current_thd->is_error())
    my_error(ER_INTERNAL_ERROR, MYF(0), function_name);
  return false;
}

static bool stage_transactional_changes(
    std::vector<vector_index::index_service::pending_change_snapshot> changes,
    const char *function_name) {
  if (current_thd == nullptr || changes.empty() ||
      !vector_trx_participant::register_participant(current_thd) ||
      !vector_index_registry::stage_changes_for_thd_txn(
          current_thd, static_cast<uint64_t>(current_thd->query_id), changes)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), function_name);
    return false;
  }
  if (maybe_binlog_vector_transactional_write_query(current_thd)) return true;
  if (!current_thd->is_error())
    my_error(ER_INTERNAL_ERROR, MYF(0), function_name);
  return false;
}

static bool check_truth_recovery_statement_context(THD *thd,
                                                   bool recovery_required) {
  if (!recovery_required) return false;
  if (thd == nullptr || thd->locked_tables_mode != LTM_NONE ||
      vector_trx_participant::in_user_multi_statement_transaction(thd)) {
    my_error(ER_LOCK_OR_ACTIVE_TRANSACTION, MYF(0));
    return true;
  }
  return false;
}

longlong Item_func_vec_index_create::val_int() {
  assert_fixed_arg_count_between(fixed, arg_count, 2, 7);
  null_value = true;

  String name_buf;
  std::string index_name;
  ulonglong dim = 0;
  if (!eval_string_arg(args[0], &name_buf, &index_name) ||
      !eval_uint_arg(args[1], dim) || dim == 0) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

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
    if (!eval_string_arg(args[2], &metric_buf, &metric)) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
  }
  if (arg_count >= 4) {
    if (!eval_string_arg(args[3], &mode_buf, &mode) || mode.empty()) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
  }
  if (arg_count >= 5) {
    if (!eval_string_arg(args[4], &provider_buf, &provider) ||
        provider.empty()) {
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
    std::string consistency_text;
    if (!eval_string_arg(args[6], &consistency_buf, &consistency_text)) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }

    vector_index::index_consistency_mode consistency_mode;
    if (!vector_index::parse_index_consistency_mode(consistency_text,
                                                    &consistency_mode)) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
    options.consistency_mode_specified = true;
    options.consistency_mode = consistency_mode;
  }
  std::string encoded;
  if (!vector_index_registry::make_create_statement_payload(
          static_cast<size_t>(dim), metric, mode, provider, owner_schema,
          options, &encoded)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (!stage_and_binlog_statement(
          vector_index_truth_store::publication_operation::kCreateIndex,
          index_name, encoded, true, func_name())) {
    return error_int();
  }
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
  std::string index_name;
  uint32_t search_ef = 0;
  if (!eval_string_arg(args[0], &name_buf, &index_name) ||
      !eval_uint32_arg(args[1], 1, std::numeric_limits<uint32_t>::max(),
                       search_ef)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!stage_config_statement(
          index_name, vector_statement_publication::config_change::kSearchEf,
          {search_ef}, func_name()))
    return error_int();
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
  std::string index_name;
  uint32_t hnsw_m = 0;
  uint32_t hnsw_ef_construction = 0;
  if (!eval_string_arg(args[0], &name_buf, &index_name) ||
      !eval_uint32_arg(args[1], 1, std::numeric_limits<uint32_t>::max(),
                       hnsw_m) ||
      !eval_uint32_arg(args[2], 1, std::numeric_limits<uint32_t>::max(),
                       hnsw_ef_construction)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!stage_config_statement(
          index_name,
          vector_statement_publication::config_change::kHnswBuildParams,
          {hnsw_m, hnsw_ef_construction}, func_name()))
    return error_int();
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
  std::string index_name;
  uint32_t faiss_nlist = 0;
  uint32_t faiss_nprobe = 0;
  if (!eval_string_arg(args[0], &name_buf, &index_name) ||
      !eval_uint32_arg(args[1], 0, std::numeric_limits<uint32_t>::max(),
                       faiss_nlist) ||
      !eval_uint32_arg(args[2], 0, std::numeric_limits<uint32_t>::max(),
                       faiss_nprobe)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!stage_config_statement(
          index_name,
          vector_statement_publication::config_change::kFaissIvfParams,
          {faiss_nlist, faiss_nprobe}, func_name()))
    return error_int();
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
  std::string index_name;
  uint32_t faiss_nlist = 0;
  uint32_t faiss_nprobe = 0;
  uint32_t faiss_pq_m = 0;
  uint32_t faiss_pq_bits = 0;
  if (!eval_string_arg(args[0], &name_buf, &index_name) ||
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

  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!stage_config_statement(
          index_name,
          vector_statement_publication::config_change::kFaissIvfPqParams,
          {faiss_nlist, faiss_nprobe, faiss_pq_m, faiss_pq_bits},
          func_name()))
    return error_int();
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
  std::string index_name;
  uint32_t diskann_max_degree = 0;
  uint32_t diskann_build_complexity = 0;
  uint32_t diskann_build_threads_arg = 0;
  if (!eval_string_arg(args[0], &name_buf, &index_name) ||
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

  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  const uint32_t diskann_build_threads =
      arg_count >= 4 && diskann_build_threads_arg != 0
          ? diskann_build_threads_arg
          : static_cast<uint32_t>(opt_vector_diskann_build_threads);
  if (!stage_config_statement(
          index_name,
          vector_statement_publication::config_change::kDiskannBuildParams,
          {diskann_max_degree, diskann_build_complexity,
           diskann_build_threads},
          func_name()))
    return error_int();
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
  std::string index_name;
  uint32_t diskann_search_complexity = 0;
  if (!eval_string_arg(args[0], &name_buf, &index_name) ||
      !eval_uint32_arg(args[1], 1, std::numeric_limits<uint32_t>::max(),
                       diskann_search_complexity)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!stage_config_statement(
          index_name,
          vector_statement_publication::config_change::
              kDiskannSearchComplexity,
          {diskann_search_complexity}, func_name()))
    return error_int();
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
  std::string index_name;
  uint32_t diskann_search_beamwidth = 0;
  if (!eval_string_arg(args[0], &name_buf, &index_name) ||
      !eval_uint32_arg(args[1], 1, vector_index::k_max_diskann_search_beamwidth,
                       diskann_search_beamwidth)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!stage_config_statement(
          index_name,
          vector_statement_publication::config_change::kDiskannSearchBeamwidth,
          {diskann_search_beamwidth}, func_name()))
    return error_int();
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
  std::string index_name;
  ulonglong diskann_pq_code_budget_size = 0;
  if (!eval_string_arg(args[0], &name_buf, &index_name) ||
      !eval_uint_arg(args[1], diskann_pq_code_budget_size)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!stage_config_statement(
          index_name,
          vector_statement_publication::config_change::
              kDiskannPqCodeBudgetSize,
          {static_cast<uint64_t>(diskann_pq_code_budget_size)}, func_name()))
    return error_int();
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
  std::string index_name;
  uint32_t diskann_disk_pq_dims = 0;
  if (!eval_string_arg(args[0], &name_buf, &index_name) ||
      !eval_uint32_arg(args[1], 0, vector_index::k_max_diskann_disk_pq_dims,
                       diskann_disk_pq_dims)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  if (!stage_config_statement(
          index_name,
          vector_statement_publication::config_change::kDiskannDiskPqDims,
          {diskann_disk_pq_dims}, func_name()))
    return error_int();
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

  if (!stage_diskann_boolean_config(
          args[0], args[1],
          vector_statement_publication::config_change::
              kDiskannAccelerateBuild,
          func_name())) {
    return error_int();
  }
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

  if (!stage_diskann_boolean_config(
          args[0], args[1],
          vector_statement_publication::config_change::kDiskannShuffleBuild,
          func_name())) {
    return error_int();
  }
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

  if (!stage_diskann_boolean_config(
          args[0], args[1],
          vector_statement_publication::config_change::kDiskannUseBfsCache,
          func_name())) {
    return error_int();
  }
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
  std::string index_name;
  std::string build_mode_text;
  const bool name_valid = eval_string_arg(args[0], &name_buf, &index_name);
  const bool mode_valid = eval_string_arg(args[1], &mode_buf, &build_mode_text);
  if (!name_valid || !mode_valid) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

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

  if (!stage_config_statement(
          index_name,
          vector_statement_publication::config_change::kDiskannBuildMode,
          {static_cast<uint64_t>(build_mode)}, func_name()))
    return error_int();
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
  std::string index_name;
  if (!eval_string_arg(args[0], &name_buf, &index_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (check_vector_existing_index_access(
          current_thd, index_name, DROP_ACL, func_name(), true))
    return error_int();
  vector_index_registry::index_info info;
  if (!vector_index_registry::get_index_info(index_name, &info)) {
    null_value = false;
    return 0;
  }
  const bool dropped = stage_index_statement(
      index_name, vector_index_truth_store::publication_operation::kDropIndex,
      true, func_name());
  if (!dropped) return error_int();
  null_value = false;
  return 1;
}

bool Item_func_vec_index_rebuild::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1)) return true;
  set_nullable(false);
  return false;
}

longlong Item_func_vec_index_rebuild::val_int() {
  assert_fixed_arg_count(fixed, arg_count, 1);
  null_value = true;

  if (!stage_unary_index_statement(
          args[0],
          vector_index_truth_store::publication_operation::kRebuildIndex,
          func_name(), "vector index not found or not rebuildable"))
    return error_int();

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

  if (!stage_unary_index_statement(
          args[0],
          vector_index_truth_store::publication_operation::kBeginBulkLoad,
          func_name()))
    return error_int();

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

  if (!stage_unary_index_statement(
          args[0],
          vector_index_truth_store::publication_operation::kBulkBuildIndex,
          func_name()))
    return error_int();

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
  std::string index_name;
  if (!eval_string_arg(args[0], &name_buf, &index_name)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  const bool repairing_truth_projection =
      vector_index_registry::registry_health() ==
      vector_index_registry::registry_health_state::kRecoveryRequired;
  if (check_truth_recovery_statement_context(current_thd,
                                             repairing_truth_projection)) {
    return error_int();
  }
  if (repairing_truth_projection) {
    if (!vector_index_registry::recover_index(current_thd, index_name)) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
  } else if (!stage_index_statement(
                 index_name,
                 vector_index_truth_store::publication_operation::
                     kRecoverIndex,
                 false, func_name())) {
    return error_int();
  }
  // Corruption repair is local reconstruction from replicated base-table rows;
  // replaying it on replicas would couple independent derived-artifact health.
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
  vector_status::record_rebuild_all_request();
  if (!stage_all_index_statements(
          vector_index_truth_store::publication_operation::kRebuildIndex,
          func_name(), &rebuilt_count))
    return error_int();

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
  const bool repairing_truth_projection =
      vector_index_registry::registry_health() ==
      vector_index_registry::registry_health_state::kRecoveryRequired;
  if (check_truth_recovery_statement_context(current_thd,
                                             repairing_truth_projection)) {
    return error_int();
  }
  if (repairing_truth_projection) {
    if (!vector_index_registry::recover_all_indexes(current_thd,
                                                    &recovered_count)) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
      return error_int();
    }
  } else {
    vector_status::record_recover_all_request();
    if (!stage_all_index_statements(
            vector_index_truth_store::publication_operation::kRecoverIndex,
            func_name(), &recovered_count)) {
      return error_int();
    }
  }

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
  std::string index_name;
  ulonglong doc_id = 0;
  if (!eval_string_arg(args[0], &name_buf, &index_name) ||
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

  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  vector_index_registry::index_info info;
  if (!vector_index_registry::get_index_info(index_name, &info)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (info.consistency_mode == "transactional") {
    vector_index::index_service::pending_change_snapshot change;
    change.index_name = index_name;
    change.doc_id = static_cast<uint64_t>(doc_id);
    change.vector = std::move(vector);
    if (!stage_transactional_changes({std::move(change)}, func_name()))
      return error_int();
  } else {
    vector_statement_publication::operation_payload payload;
    payload.unsigned_values = {static_cast<uint64_t>(doc_id), vector.size()};
    payload.binary_value.resize(vector.size() * sizeof(float));
    auto *bytes = reinterpret_cast<uchar *>(payload.binary_value.data());
    for (size_t i = 0; i < vector.size(); ++i) {
      float4store(bytes + i * sizeof(float), vector[i]);
    }
    std::string encoded;
    if (!encode_statement_payload(payload, func_name(), &encoded) ||
        !stage_and_binlog_statement(
            vector_index_truth_store::publication_operation::
                kStandaloneUpsert,
            index_name, encoded, false, func_name())) {
      return error_int();
    }
  }

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
  std::vector<vector_index::index_service::pending_change_snapshot> changes;
  if (info.consistency_mode == "transactional") changes.reserve(row_count);
  for (size_t row_idx = 0; row_idx < row_count; ++row_idx) {
    if (current_thd != nullptr && current_thd->killed) {
      my_error(ER_QUERY_INTERRUPTED, MYF(0));
      return error_int();
    }
    vector_index::vector_data row(dimension);
    const uchar *row_ptr =
        vector_ptr + row_idx * dimension * vector_utils::kVectorElemSize;
    for (size_t dim_idx = 0; dim_idx < dimension; ++dim_idx) {
      const float value =
          float4get(row_ptr + dim_idx * vector_utils::kVectorElemSize);
      if (!std::isfinite(value)) {
        my_error(ER_WRONG_ARGUMENTS, MYF(0),
                 "VEC_INDEX_UPSERT_BATCH non-finite vector value");
        return error_int();
      }
      row[dim_idx] = value;
    }
    if (info.consistency_mode == "transactional") {
      vector_index::index_service::pending_change_snapshot change;
      change.index_name = index_name;
      change.doc_id = uint8korr(docid_ptr + row_idx * sizeof(uint64_t));
      change.vector = std::move(row);
      changes.push_back(std::move(change));
    }
  }

  if (info.consistency_mode == "transactional") {
    if (!stage_transactional_changes(std::move(changes), func_name()))
      return error_int();
  } else {
    vector_statement_publication::operation_payload payload;
    payload.unsigned_values = {1, row_count, dimension, 1};
    payload.string_values.emplace_back(docid_blob->ptr(), docid_blob->length());
    payload.binary_value.assign(vector_blob->ptr(), vector_blob->length());
    std::string encoded;
    if (!encode_statement_payload(payload, func_name(), &encoded) ||
        !stage_and_binlog_statement(
            vector_index_truth_store::publication_operation::kBulkLoad,
            index_name, encoded, false, func_name())) {
      return error_int();
    }
  }

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
  std::string index_name;
  ulonglong doc_id = 0;
  if (!eval_string_arg(args[0], &name_buf, &index_name) ||
      !eval_uint_arg(args[1], doc_id)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }

  if (check_vector_existing_index_access(
          current_thd, index_name, ALTER_ACL, func_name(), false))
    return error_int();
  vector_index_registry::index_info info;
  if (!vector_index_registry::get_index_info(index_name, &info)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return error_int();
  }
  if (info.consistency_mode == "transactional") {
    vector_index::index_service::pending_change_snapshot change;
    change.index_name = index_name;
    change.erase = true;
    change.doc_id = static_cast<uint64_t>(doc_id);
    if (!stage_transactional_changes({std::move(change)}, func_name()))
      return error_int();
  } else {
    vector_statement_publication::operation_payload payload;
    payload.unsigned_values = {static_cast<uint64_t>(doc_id)};
    std::string encoded;
    if (!encode_statement_payload(payload, func_name(), &encoded) ||
        !stage_and_binlog_statement(
            vector_index_truth_store::publication_operation::kStandaloneErase,
            index_name, encoded, false, func_name())) {
      return error_int();
    }
  }

  null_value = false;
  return 1;
}
