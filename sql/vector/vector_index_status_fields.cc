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

#include "sql/vector/vector_index_status_fields.h"

#include <cctype>
#include <cstddef>
#include <cstring>
#include <utility>

#include "sql/vector/vector_index_observability.h"

namespace vector_index_status_fields {
namespace {

bool ascii_equal_ignore_case(const std::string &lhs, const char *rhs) {
  const size_t rhs_len = std::strlen(rhs);
  if (lhs.size() != rhs_len) return false;
  for (size_t i = 0; i < rhs_len; ++i) {
    const auto left = static_cast<unsigned char>(lhs[i]);
    const auto right = static_cast<unsigned char>(rhs[i]);
    if (std::tolower(left) != std::tolower(right)) return false;
  }
  return true;
}

void append_string(field_values *fields, const char *name,
                   const std::string &value) {
  field_value field;
  field.name = name;
  field.kind = field_kind::k_string;
  field.string_value = value;
  fields->push_back(std::move(field));
}

void append_nullable_string(field_values *fields, const char *name,
                            const std::string &value) {
  field_value field;
  field.name = name;
  field.kind = field_kind::k_nullable_string;
  field.string_value = value;
  fields->push_back(std::move(field));
}

void append_uint(field_values *fields, const char *name, uint64_t value) {
  field_value field;
  field.name = name;
  field.kind = field_kind::k_uint;
  field.uint_value = value;
  fields->push_back(std::move(field));
}

void append_bool(field_values *fields, const char *name, bool value) {
  field_value field;
  field.name = name;
  field.kind = field_kind::k_bool;
  field.bool_value = value;
  fields->push_back(std::move(field));
}

bool has_build_diagnostics(
    const vector_index::backend_build_diagnostics &diagnostics) {
  bool present = false;
  present |= !diagnostics.runtime.empty();
  present |= !diagnostics.input_source.empty();
  present |= diagnostics.row_count != 0;
  present |= diagnostics.segment_count != 0;
  present |= diagnostics.build_invocations != 0;
  present |= diagnostics.concurrent_build_tasks != 0;
  present |= diagnostics.scheduler_cpu_budget != 0;
  present |= diagnostics.effective_build_threads != 0;
  present |= diagnostics.effective_blas_threads != 0;
  present |= diagnostics.raw_reader_threads != 0;
  present |= diagnostics.pq_train_threads != 0;
  present |= diagnostics.pq_compress_threads != 0;
  present |= diagnostics.candidates_per_segment != 0;
  present |= diagnostics.search_fanout_segments != 0;
  present |= diagnostics.search_fanout_threads != 0;
  present |= diagnostics.search_global_top_k != 0;
  present |= diagnostics.search_per_segment_top_k != 0;
  present |= diagnostics.search_result_budget != 0;
  present |= diagnostics.search_candidate_count != 0;
  present |= diagnostics.search_query_count != 0;
  present |= diagnostics.search_segment_min_entries != 0;
  present |= diagnostics.search_segment_max_entries != 0;
  present |= diagnostics.search_segment_total_entries != 0;
  present |= diagnostics.search_diskann_search_list != 0;
  present |= diagnostics.search_diskann_beamwidth != 0;
  present |= diagnostics.search_total_candidate_rows != 0;
  present |= diagnostics.single_index_build;
  present |= diagnostics.pq_chunks != 0;
  present |= diagnostics.cache_nodes != 0;
  present |= diagnostics.requested_disk_pq_dims != 0;
  present |= diagnostics.effective_disk_pq_dims != 0;
  present |= diagnostics.build_wall_ms != 0;
  present |= diagnostics.manifest_ms != 0;
  present |= diagnostics.offline_build_ms != 0;
  present |= diagnostics.load_ms != 0;
  present |= !diagnostics.native_pq_runtime_selected_path.empty();
  present |= diagnostics.native_pq_runtime_elapsed_ms != 0;
  present |= diagnostics.native_pq_runtime_raw_reader_ms != 0;
  present |= diagnostics.native_pq_runtime_distance_calls != 0;
  present |= diagnostics.native_pq_runtime_train_rows != 0;
  present |= diagnostics.native_pq_runtime_compressed_rows != 0;
  present |= diagnostics.native_pq_runtime_artifacts_written;
  present |= diagnostics.native_pq_runtime_artifacts_consumed;
  present |= diagnostics.native_pq_runtime_official_pq_used;
  present |= !diagnostics.native_pq_runtime_bridge.empty();
  present |= diagnostics.native_pq_runtime_bridge_ms != 0;
  present |= diagnostics.native_pq_runtime_graph_ms != 0;
  present |= diagnostics.native_pq_runtime_cache_ms != 0;
  present |= !diagnostics.native_pq_runtime_artifact_validation.empty();
  present |= !diagnostics.fallback_reason.empty();
  return present;
}

void append_build_diagnostics(
    field_values *fields, bool diskann_provider,
    const vector_index::backend_build_diagnostics &diagnostics) {
  if (!has_build_diagnostics(diagnostics)) return;
  append_nullable_string(fields, "backend_build_runtime", diagnostics.runtime);
  append_nullable_string(fields, "backend_build_input_source",
                         diagnostics.input_source);
  append_uint(fields, "backend_build_rows", diagnostics.row_count);
  append_uint(fields, "backend_build_segments", diagnostics.segment_count);
  append_uint(fields, "backend_build_invocations",
              diagnostics.build_invocations);
  append_uint(fields, "backend_build_concurrent_tasks",
              diagnostics.concurrent_build_tasks);
  append_uint(fields, "backend_build_effective_threads",
              diagnostics.effective_build_threads);
  append_uint(fields, "backend_build_effective_blas_threads",
              diagnostics.effective_blas_threads);
  append_uint(fields, "backend_build_raw_reader_threads",
              diagnostics.raw_reader_threads);
  append_uint(fields, "backend_build_pq_train_threads",
              diagnostics.pq_train_threads);
  append_uint(fields, "backend_build_pq_compress_threads",
              diagnostics.pq_compress_threads);
  append_bool(fields, "backend_build_single_index",
              diagnostics.single_index_build);
  append_uint(fields, "backend_build_pq_chunks", diagnostics.pq_chunks);
  append_uint(fields, "backend_build_cache_nodes", diagnostics.cache_nodes);
  if (diskann_provider && diagnostics.effective_disk_pq_dims != 0) {
    append_uint(fields, "diskann_effective_disk_pq_dims",
                diagnostics.effective_disk_pq_dims);
  }
  append_uint(fields, "backend_build_manifest_ms", diagnostics.manifest_ms);
  append_uint(fields, "backend_build_offline_ms",
              diagnostics.offline_build_ms);
  append_uint(fields, "backend_build_load_ms", diagnostics.load_ms);
  if (diskann_provider && !diagnostics.diskann_pq_runtime.empty()) {
    append_nullable_string(fields, "diskann_pq_runtime",
                           diagnostics.diskann_pq_runtime);
  }
  const bool has_native_pq_diagnostics =
      !diagnostics.native_pq_runtime_selected_path.empty() ||
      diagnostics.native_pq_runtime_elapsed_ms != 0 ||
      diagnostics.native_pq_runtime_raw_reader_ms != 0 ||
      diagnostics.native_pq_runtime_distance_calls != 0 ||
      diagnostics.native_pq_runtime_train_rows != 0 ||
      diagnostics.native_pq_runtime_compressed_rows != 0 ||
      diagnostics.native_pq_runtime_artifacts_written ||
      diagnostics.native_pq_runtime_artifacts_consumed ||
      diagnostics.native_pq_runtime_official_pq_used ||
      !diagnostics.native_pq_runtime_bridge.empty() ||
      diagnostics.native_pq_runtime_bridge_ms != 0 ||
      diagnostics.native_pq_runtime_graph_ms != 0 ||
      diagnostics.native_pq_runtime_cache_ms != 0 ||
      !diagnostics.native_pq_runtime_artifact_validation.empty();
  if (diskann_provider && has_native_pq_diagnostics) {
    append_nullable_string(fields, "native_pq_runtime_selected_path",
                           diagnostics.native_pq_runtime_selected_path);
    append_uint(fields, "native_pq_runtime_elapsed_ms",
                diagnostics.native_pq_runtime_elapsed_ms);
    append_uint(fields, "native_pq_runtime_raw_reader_ms",
                diagnostics.native_pq_runtime_raw_reader_ms);
    append_uint(fields, "native_pq_runtime_distance_calls",
                diagnostics.native_pq_runtime_distance_calls);
    append_uint(fields, "native_pq_runtime_train_rows",
                diagnostics.native_pq_runtime_train_rows);
    append_uint(fields, "native_pq_runtime_compressed_rows",
                diagnostics.native_pq_runtime_compressed_rows);
    append_bool(fields, "native_pq_runtime_artifacts_written",
                diagnostics.native_pq_runtime_artifacts_written);
    append_bool(fields, "native_pq_runtime_artifacts_consumed",
                diagnostics.native_pq_runtime_artifacts_consumed);
    append_bool(fields, "native_pq_runtime_official_pq_used",
                diagnostics.native_pq_runtime_official_pq_used);
    append_nullable_string(fields, "native_pq_runtime_bridge",
                           diagnostics.native_pq_runtime_bridge);
    append_uint(fields, "native_pq_runtime_bridge_ms",
                diagnostics.native_pq_runtime_bridge_ms);
    append_uint(fields, "native_pq_runtime_graph_ms",
                diagnostics.native_pq_runtime_graph_ms);
    append_uint(fields, "native_pq_runtime_cache_ms",
                diagnostics.native_pq_runtime_cache_ms);
    append_nullable_string(fields, "native_pq_runtime_artifact_validation",
                           diagnostics.native_pq_runtime_artifact_validation);
  }
  append_nullable_string(fields, "backend_build_fallback_reason",
                         diagnostics.fallback_reason);

  append_nullable_string(fields, "scheduler_path", diagnostics.runtime);
  append_uint(fields, "scheduler_segment_count", diagnostics.segment_count);
  append_uint(fields, "scheduler_task_count", diagnostics.build_invocations);
  append_uint(fields, "scheduler_concurrent_tasks",
              diagnostics.concurrent_build_tasks);
  append_uint(fields, "scheduler_cpu_budget", diagnostics.scheduler_cpu_budget);
  append_uint(fields, "scheduler_effective_build_threads",
              diagnostics.effective_build_threads);
  append_uint(fields, "scheduler_effective_blas_threads",
              diagnostics.effective_blas_threads);
  append_uint(fields, "scheduler_raw_reader_threads",
              diagnostics.raw_reader_threads);
  append_uint(fields, "scheduler_pq_train_threads",
              diagnostics.pq_train_threads);
  append_uint(fields, "scheduler_pq_compress_threads",
              diagnostics.pq_compress_threads);
  append_bool(fields, "scheduler_single_index_build",
              diagnostics.single_index_build);
  append_uint(fields, "scheduler_candidates_per_segment",
              diagnostics.candidates_per_segment);
  const bool has_search_diagnostics =
      diagnostics.search_fanout_segments != 0 ||
      diagnostics.search_fanout_threads != 0 ||
      diagnostics.search_global_top_k != 0 ||
      diagnostics.search_per_segment_top_k != 0 ||
      diagnostics.search_result_budget != 0 ||
      diagnostics.search_candidate_count != 0 ||
      diagnostics.search_query_count != 0 ||
      diagnostics.search_segment_min_entries != 0 ||
      diagnostics.search_segment_max_entries != 0 ||
      diagnostics.search_segment_total_entries != 0 ||
      diagnostics.search_diskann_search_list != 0 ||
      diagnostics.search_diskann_beamwidth != 0 ||
      diagnostics.search_total_candidate_rows != 0;
  if (has_search_diagnostics) {
    append_uint(fields, "scheduler_search_fanout_segments",
                diagnostics.search_fanout_segments);
    append_uint(fields, "scheduler_search_fanout_threads",
                diagnostics.search_fanout_threads);
    append_uint(fields, "scheduler_search_global_top_k",
                diagnostics.search_global_top_k);
    append_uint(fields, "scheduler_search_per_segment_top_k",
                diagnostics.search_per_segment_top_k);
    append_uint(fields, "scheduler_search_result_budget",
                diagnostics.search_result_budget);
    append_uint(fields, "scheduler_search_candidate_count",
                diagnostics.search_candidate_count);
    append_uint(fields, "scheduler_search_query_count",
                diagnostics.search_query_count);
    append_uint(fields, "scheduler_search_segment_min_entries",
                diagnostics.search_segment_min_entries);
    append_uint(fields, "scheduler_search_segment_max_entries",
                diagnostics.search_segment_max_entries);
    append_uint(fields, "scheduler_search_segment_total_entries",
                diagnostics.search_segment_total_entries);
    append_uint(fields, "scheduler_search_diskann_search_list",
                diagnostics.search_diskann_search_list);
    append_uint(fields, "scheduler_search_diskann_beamwidth",
                diagnostics.search_diskann_beamwidth);
    append_uint(fields, "scheduler_search_total_candidate_rows",
                diagnostics.search_total_candidate_rows);
  }
  append_uint(fields, "diskann_segment_pq_chunks",
              diskann_provider ? diagnostics.pq_chunks : 0);
  if (diskann_provider && diagnostics.effective_disk_pq_dims != 0) {
    append_uint(fields, "diskann_segment_effective_disk_pq_dims",
                diagnostics.effective_disk_pq_dims);
  }
  append_uint(fields, "diskann_segment_cache_nodes",
              diskann_provider ? diagnostics.cache_nodes : 0);
  const uint64_t build_wall_ms =
      diagnostics.build_wall_ms != 0 ? diagnostics.build_wall_ms
                                     : diagnostics.offline_build_ms;
  append_uint(fields, "diskann_segment_build_wall_ms",
              diskann_provider ? build_wall_ms : 0);
  append_uint(fields, "diskann_segment_build_sum_ms",
              diskann_provider ? diagnostics.offline_build_ms : 0);
  append_nullable_string(fields, "diskann_segment_fallback_reason",
                         diskann_provider ? diagnostics.fallback_reason
                                          : "not_applicable");
}

}  // namespace

uint64_t pending_apply_count(const vector_index_registry::index_info &info) {
  return vector_index_observability::pending_apply_count(info);
}

uint64_t rebuild_progress(const vector_index_registry::index_info &info) {
  return vector_index_observability::rebuild_progress(info);
}

uint64_t recover_progress(const vector_index_registry::index_info &info) {
  return vector_index_observability::recover_progress(info);
}

bool is_loaded(const vector_index_registry::index_info &info) {
  return vector_index_observability::is_loaded(info);
}

bool is_writable(const vector_index_registry::index_info &info) {
  return vector_index_observability::is_writable(info);
}

std::string status_value(const field_value &field) {
  switch (field.kind) {
    case field_kind::k_string:
    case field_kind::k_nullable_string:
      return field.string_value;
    case field_kind::k_uint:
      return std::to_string(field.uint_value);
    case field_kind::k_bool:
      return field.bool_value ? "1" : "0";
  }
  return {};
}

void collect_info_fields(const vector_index_registry::index_info &info,
                         field_values *fields) {
  if (fields == nullptr) return;
  fields->clear();
  fields->reserve(72);

  append_uint(fields, "dimension", info.dimension);
  append_string(fields, "metric", info.metric);
  append_string(fields, "mode", info.mode);
  append_string(fields, "provider", info.provider);
  append_string(fields, "consistency_mode", info.consistency_mode);
  append_bool(fields, "truth_store_enabled", info.truth_store_enabled);
  append_string(fields, "build_source", info.build_source);
  append_string(fields, "build_pipeline_mode", info.build_pipeline_mode);
  append_string(fields, "build_pipeline_decision",
                info.build_pipeline_decision);
  append_string(fields, "build_pipeline_trigger",
                info.build_pipeline_trigger);
  append_uint(fields, "build_pipeline_rows", info.build_pipeline_rows);
  append_uint(fields, "build_pipeline_payload_size",
              info.build_pipeline_payload_size);
  append_uint(fields, "build_pipeline_raw_segments",
              info.build_pipeline_raw_segments);
  append_uint(fields, "build_segment_effective_row_limit",
              info.build_segment_effective_row_limit);
  append_uint(fields, "build_segment_target_size",
              info.build_segment_target_size);
  append_uint(fields, "build_segment_max_rows", info.build_segment_max_rows);
  append_string(fields, "build_segment_policy", info.build_segment_policy);
  append_uint(fields, "standalone_ingest_memory_bytes",
              info.standalone_ingest_memory_bytes);
  append_uint(fields, "standalone_segment_count",
              info.standalone_segment_count);
  append_uint(fields, "standalone_segment_bytes",
              info.standalone_segment_bytes);
  append_uint(fields, "standalone_raw_segment_count",
              info.standalone_raw_segment_count);
  append_uint(fields, "standalone_raw_segment_bytes",
              info.standalone_raw_segment_bytes);
  append_nullable_string(fields, "backend_variant", info.backend_variant);
  append_build_diagnostics(fields, ascii_equal_ignore_case(info.provider,
                                                           "diskann"),
                           info.build_diagnostics);
  append_nullable_string(fields, "owner_schema", info.owner_schema);
  append_nullable_string(fields, "schema_name", info.schema_name);
  append_nullable_string(fields, "table_name", info.table_name);
  append_nullable_string(fields, "column_name", info.column_name);
  append_uint(fields, "search_ef", info.search_ef);
  append_uint(fields, "hnsw_m", info.hnsw_m);
  append_uint(fields, "hnsw_ef_construction", info.hnsw_ef_construction);
  append_uint(fields, "hnsw_build_threads", info.hnsw_build_threads);
  append_uint(fields, "faiss_nlist", info.faiss_nlist);
  append_uint(fields, "faiss_nprobe", info.faiss_nprobe);
  append_uint(fields, "faiss_pq_m", info.faiss_pq_m);
  append_uint(fields, "faiss_pq_bits", info.faiss_pq_bits);
  append_uint(fields, "faiss_build_threads", info.faiss_build_threads);
  append_uint(fields, "diskann_max_degree", info.diskann_max_degree);
  append_uint(fields, "diskann_build_complexity",
              info.diskann_build_complexity);
  append_uint(fields, "diskann_build_threads", info.diskann_build_threads);
  append_string(fields, "diskann_build_mode",
                vector_index::diskann_build_mode_to_string(
                    info.diskann_build_mode_value));
  append_uint(fields, "diskann_search_complexity",
              info.diskann_search_complexity);
  append_uint(fields, "diskann_search_beamwidth",
              info.diskann_search_beamwidth);
  append_uint(fields, "diskann_pq_code_budget_size",
              info.diskann_pq_code_budget_size);
  append_uint(fields, "diskann_disk_pq_dims", info.diskann_disk_pq_dims);
  append_uint(fields, "diskann_cache_nodes", info.diskann_cache_nodes);
  append_bool(fields, "diskann_accelerate_build",
              info.diskann_accelerate_build);
  append_bool(fields, "diskann_shuffle_build", info.diskann_shuffle_build);
  append_bool(fields, "diskann_use_bfs_cache", info.diskann_use_bfs_cache);
  append_bool(fields, "supports_mutations", info.supports_mutations);
  append_string(fields, "lifecycle_state", info.lifecycle_state);
  append_uint(fields, "lifecycle_version", info.lifecycle_version);
  append_uint(fields, "last_error_code", info.last_error_code);
  append_uint(fields, "last_error_ts", info.last_error_ts);
  append_uint(fields, "last_apply_latency_ms", info.last_apply_latency_ms);
  append_uint(fields, "recover_fallback_count", info.recover_fallback_count);
  append_uint(fields, "last_recover_fallback_ts",
              info.last_recover_fallback_ts);
  append_bool(fields, "external_manifest_present",
              info.external_manifest_present);
  append_uint(fields, "external_manifest_generation",
              info.external_manifest_generation);
  append_uint(fields, "pending_apply_count", pending_apply_count(info));
  append_uint(fields, "rebuild_progress", rebuild_progress(info));
  append_uint(fields, "recover_progress", recover_progress(info));
  append_uint(fields, "entry_count", info.entry_count);
  append_uint(fields, "committed_entry_count", info.committed_entry_count);
}

void collect_index_state_fields(const vector_index_registry::index_info &info,
                                field_values *fields) {
  if (fields == nullptr) return;
  fields->clear();
  fields->reserve(49);

  append_uint(fields, "dimension", info.dimension);
  append_string(fields, "metric", info.metric);
  append_string(fields, "mode", info.mode);
  append_string(fields, "provider", info.provider);
  append_string(fields, "consistency_mode", info.consistency_mode);
  append_bool(fields, "truth_store_enabled", info.truth_store_enabled);
  append_string(fields, "build_source", info.build_source);
  append_string(fields, "build_pipeline_mode", info.build_pipeline_mode);
  append_string(fields, "build_pipeline_decision",
                info.build_pipeline_decision);
  append_string(fields, "build_pipeline_trigger",
                info.build_pipeline_trigger);
  append_uint(fields, "build_pipeline_rows", info.build_pipeline_rows);
  append_uint(fields, "build_pipeline_payload_size",
              info.build_pipeline_payload_size);
  append_uint(fields, "build_pipeline_raw_segments",
              info.build_pipeline_raw_segments);
  append_uint(fields, "build_segment_effective_row_limit",
              info.build_segment_effective_row_limit);
  append_uint(fields, "build_segment_target_size",
              info.build_segment_target_size);
  append_uint(fields, "build_segment_max_rows", info.build_segment_max_rows);
  append_string(fields, "build_segment_policy", info.build_segment_policy);
  append_uint(fields, "standalone_ingest_memory_bytes",
              info.standalone_ingest_memory_bytes);
  append_uint(fields, "standalone_segment_count",
              info.standalone_segment_count);
  append_uint(fields, "standalone_segment_bytes",
              info.standalone_segment_bytes);
  append_uint(fields, "standalone_raw_segment_count",
              info.standalone_raw_segment_count);
  append_uint(fields, "standalone_raw_segment_bytes",
              info.standalone_raw_segment_bytes);
  append_nullable_string(fields, "backend_variant", info.backend_variant);
  append_nullable_string(fields, "schema_name", info.schema_name);
  append_nullable_string(fields, "table_name", info.table_name);
  append_nullable_string(fields, "column_name", info.column_name);
  append_uint(fields, "search_ef", info.search_ef);
  append_uint(fields, "hnsw_m", info.hnsw_m);
  append_uint(fields, "hnsw_ef_construction", info.hnsw_ef_construction);
  append_uint(fields, "hnsw_build_threads", info.hnsw_build_threads);
  append_uint(fields, "faiss_nlist", info.faiss_nlist);
  append_uint(fields, "faiss_nprobe", info.faiss_nprobe);
  append_uint(fields, "faiss_pq_m", info.faiss_pq_m);
  append_uint(fields, "faiss_pq_bits", info.faiss_pq_bits);
  append_uint(fields, "faiss_build_threads", info.faiss_build_threads);
  append_uint(fields, "diskann_max_degree", info.diskann_max_degree);
  append_uint(fields, "diskann_build_complexity",
              info.diskann_build_complexity);
  append_uint(fields, "diskann_build_threads", info.diskann_build_threads);
  append_string(fields, "diskann_build_mode",
                vector_index::diskann_build_mode_to_string(
                    info.diskann_build_mode_value));
  append_uint(fields, "diskann_search_complexity",
              info.diskann_search_complexity);
  append_uint(fields, "diskann_search_beamwidth",
              info.diskann_search_beamwidth);
  append_uint(fields, "diskann_pq_code_budget_size",
              info.diskann_pq_code_budget_size);
  append_uint(fields, "diskann_disk_pq_dims", info.diskann_disk_pq_dims);
  append_uint(fields, "diskann_cache_nodes", info.diskann_cache_nodes);
  append_bool(fields, "diskann_accelerate_build",
              info.diskann_accelerate_build);
  append_bool(fields, "diskann_shuffle_build", info.diskann_shuffle_build);
  append_bool(fields, "diskann_use_bfs_cache", info.diskann_use_bfs_cache);
  append_string(fields, "lifecycle_state", info.lifecycle_state);
  append_uint(fields, "lifecycle_version", info.lifecycle_version);
  append_bool(fields, "supports_mutations", info.supports_mutations);
  append_uint(fields, "entry_count", info.entry_count);
  append_uint(fields, "committed_entry_count", info.committed_entry_count);
}

void collect_backend_health_fields(
    const vector_index_registry::index_info &info, field_values *fields) {
  if (fields == nullptr) return;
  fields->clear();
  fields->reserve(21);

  append_string(fields, "backend_type", info.provider);
  append_nullable_string(fields, "backend_variant", info.backend_variant);
  append_build_diagnostics(fields, ascii_equal_ignore_case(info.provider,
                                                           "diskann"),
                           info.build_diagnostics);
  append_string(fields, "mode", info.mode);
  append_string(fields, "consistency_mode", info.consistency_mode);
  append_bool(fields, "truth_store_enabled", info.truth_store_enabled);
  append_string(fields, "build_source", info.build_source);
  append_bool(fields, "loaded", is_loaded(info));
  append_bool(fields, "writable", is_writable(info));
  append_uint(fields, "last_error_code", info.last_error_code);
  append_uint(fields, "last_error_ts", info.last_error_ts);
  append_uint(fields, "recover_fallback_count", info.recover_fallback_count);
  append_uint(fields, "last_recover_fallback_ts",
              info.last_recover_fallback_ts);
  append_bool(fields, "external_manifest_present",
              info.external_manifest_present);
  append_uint(fields, "external_manifest_generation",
              info.external_manifest_generation);
}

void collect_sync_pipeline_fields(const vector_index_registry::index_info &info,
                                  field_values *fields) {
  if (fields == nullptr) return;
  fields->clear();
  fields->reserve(4);

  append_uint(fields, "pending_apply_count", pending_apply_count(info));
  append_uint(fields, "apply_latency_ms", info.last_apply_latency_ms);
  append_uint(fields, "rebuild_progress", rebuild_progress(info));
  append_uint(fields, "recover_progress", recover_progress(info));
}

}  // namespace vector_index_status_fields
