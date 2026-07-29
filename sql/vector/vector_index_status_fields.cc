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

namespace vector_index_status_fields {
namespace {

bool ascii_equal_ignore_case(const std::string &lhs, const char *rhs) {
  if (rhs == nullptr) return false;
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

}  // namespace

uint64_t pending_apply_count(const vector_index_registry::index_info &info) {
  const auto entry_count = static_cast<uint64_t>(info.entry_count);
  const auto committed_entry_count =
      static_cast<uint64_t>(info.committed_entry_count);
  return entry_count > committed_entry_count
             ? entry_count - committed_entry_count
             : committed_entry_count - entry_count;
}

uint64_t rebuild_progress(const vector_index_registry::index_info &info) {
  if (ascii_equal_ignore_case(info.lifecycle_state, "ready")) return 100;
  if (ascii_equal_ignore_case(info.lifecycle_state, "rebuilding")) return 50;
  return 0;
}

uint64_t recover_progress(const vector_index_registry::index_info &info) {
  if (ascii_equal_ignore_case(info.lifecycle_state, "ready")) return 100;
  if (ascii_equal_ignore_case(info.lifecycle_state, "recovering")) return 50;
  return 0;
}

bool is_loaded(const vector_index_registry::index_info &info) {
  return !ascii_equal_ignore_case(info.lifecycle_state, "failed");
}

bool is_writable(const vector_index_registry::index_info &info) {
  return is_loaded(info) && info.supports_mutations;
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
  fields->reserve(45);

  append_uint(fields, "dimension", info.dimension);
  append_string(fields, "metric", info.metric);
  append_string(fields, "mode", info.mode);
  append_string(fields, "provider", info.provider);
  append_string(fields, "consistency_mode", info.consistency_mode);
  append_bool(fields, "truth_store_enabled", info.truth_store_enabled);
  append_string(fields, "build_source", info.build_source);
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
  fields->reserve(35);

  append_uint(fields, "dimension", info.dimension);
  append_string(fields, "metric", info.metric);
  append_string(fields, "mode", info.mode);
  append_string(fields, "provider", info.provider);
  append_string(fields, "consistency_mode", info.consistency_mode);
  append_bool(fields, "truth_store_enabled", info.truth_store_enabled);
  append_string(fields, "build_source", info.build_source);
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
  fields->reserve(14);

  append_string(fields, "backend_type", info.provider);
  append_nullable_string(fields, "backend_variant", info.backend_variant);
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
