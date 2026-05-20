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

#include "sql/vector/vector_index_ddl_formatter.h"

#include <cstring>

#include "m_ctype.h"
#include "sql/sql_show.h"
#include "sql/table.h"

namespace vector_index_ddl_formatter {
namespace {

bool provider_is(const vector_index_registry::index_info &info,
                 const char *provider) {
  return my_strcasecmp(system_charset_info, info.provider.c_str(), provider) ==
         0;
}

void append_set_call_prefix(THD *thd [[maybe_unused]], String *query,
                            const char *func_name,
                            const std::string &index_name) {
  query->append(STRING_WITH_LEN(";\nSELECT "));
  query->append(func_name, std::strlen(func_name));
  query->append(STRING_WITH_LEN("("));
  append_unescaped(query, index_name.c_str(), index_name.length());
}

bool emit_search_ef(const vector_index_registry::index_info &info,
                    tuning_emit_policy emit_policy) {
  if (emit_policy == tuning_emit_policy::k_emit_nonzero) {
    return info.search_ef != 0;
  }
  return (provider_is(info, "faiss") || provider_is(info, "hnswlib")) &&
         info.search_ef != 64;
}

bool emit_hnsw_build_params(const vector_index_registry::index_info &info,
                            tuning_emit_policy emit_policy) {
  if (emit_policy == tuning_emit_policy::k_emit_nonzero) {
    return info.hnsw_m != 0 || info.hnsw_ef_construction != 0;
  }
  return provider_is(info, "hnswlib") &&
         (info.hnsw_m != 16 || info.hnsw_ef_construction != 200);
}

bool emit_faiss_ivfpq_params(const vector_index_registry::index_info &info,
                             tuning_emit_policy emit_policy) {
  if (emit_policy == tuning_emit_policy::k_emit_nonzero) {
    return info.faiss_pq_m != 0 || info.faiss_pq_bits != 0;
  }
  return provider_is(info, "faiss") &&
         (info.faiss_pq_m != 0 || info.faiss_pq_bits != 0);
}

bool emit_faiss_ivf_params(const vector_index_registry::index_info &info,
                           tuning_emit_policy emit_policy) {
  if (emit_policy == tuning_emit_policy::k_emit_nonzero) {
    return info.faiss_nlist != 0 || info.faiss_nprobe != 0;
  }
  return provider_is(info, "faiss") &&
         (info.faiss_nlist != 0 || info.faiss_nprobe != 0);
}

bool emit_diskann_build_params(const vector_index_registry::index_info &info,
                               tuning_emit_policy emit_policy) {
  if (emit_policy == tuning_emit_policy::k_emit_nonzero) {
    return info.diskann_max_degree != 0 ||
           info.diskann_build_complexity != 0 ||
           info.diskann_build_threads != 0;
  }
  return provider_is(info, "diskann") &&
         (info.diskann_max_degree != 32 ||
          info.diskann_build_complexity != 64 ||
          info.diskann_build_threads != 0);
}

bool emit_diskann_search_complexity(
    const vector_index_registry::index_info &info,
    tuning_emit_policy emit_policy) {
  if (emit_policy == tuning_emit_policy::k_emit_nonzero) {
    return info.diskann_search_complexity != 0;
  }
  return provider_is(info, "diskann") && info.diskann_search_complexity != 64;
}

}  // namespace

uint32_t provider_build_threads(
    const vector_index_registry::index_info &info) {
  if (provider_is(info, "hnswlib")) return info.hnsw_build_threads;
  if (provider_is(info, "faiss")) return info.faiss_build_threads;
  if (provider_is(info, "diskann")) return info.diskann_build_threads;
  return 0;
}

uint32_t first_nonzero_build_threads(
    const vector_index_registry::index_info &info) {
  if (info.hnsw_build_threads != 0) return info.hnsw_build_threads;
  if (info.faiss_build_threads != 0) return info.faiss_build_threads;
  return info.diskann_build_threads;
}

void append_create_statement(
    THD *thd, String *query, const char *db_name, size_t db_name_length,
    const char *table_name, size_t table_name_length, bool qualify_table,
    const std::string &column_name,
    const vector_index_registry::index_info &info, uint32_t build_threads) {
  query->append(STRING_WITH_LEN(";\nCREATE VECTOR INDEX ON "));
  if (qualify_table) {
    append_identifier(thd, query, db_name, db_name_length);
    query->append(STRING_WITH_LEN("."));
  }
  append_identifier(thd, query, table_name, table_name_length);
  query->append(STRING_WITH_LEN("("));
  append_identifier(thd, query, column_name.c_str(), column_name.length());
  query->append(STRING_WITH_LEN(") WITH ("));
  query->append_ulonglong(static_cast<ulonglong>(info.dimension));
  query->append(STRING_WITH_LEN(", "));
  append_unescaped(query, info.metric.c_str(), info.metric.length());
  query->append(STRING_WITH_LEN(", "));
  append_unescaped(query, info.mode.c_str(), info.mode.length());
  query->append(STRING_WITH_LEN(", "));
  append_unescaped(query, info.provider.c_str(), info.provider.length());
  if (build_threads != 0) {
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(static_cast<ulonglong>(build_threads));
  }
  query->append(STRING_WITH_LEN(")"));
}

void append_tuning_statements(
    THD *thd, String *query, const std::string &index_name,
    const vector_index_registry::index_info &info,
    tuning_emit_policy emit_policy) {
  if (query == nullptr) return;

  if (emit_search_ef(info, emit_policy)) {
    append_set_call_prefix(thd, query, "VEC_INDEX_SET_SEARCH_EF", index_name);
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(static_cast<ulonglong>(info.search_ef));
    query->append(STRING_WITH_LEN(")"));
  }

  if (emit_hnsw_build_params(info, emit_policy)) {
    append_set_call_prefix(thd, query, "VEC_INDEX_SET_HNSW_BUILD_PARAMS",
                           index_name);
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(static_cast<ulonglong>(info.hnsw_m));
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(
        static_cast<ulonglong>(info.hnsw_ef_construction));
    query->append(STRING_WITH_LEN(")"));
  }

  if (emit_faiss_ivfpq_params(info, emit_policy)) {
    append_set_call_prefix(thd, query, "VEC_INDEX_SET_FAISS_IVFPQ_PARAMS",
                           index_name);
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(static_cast<ulonglong>(info.faiss_nlist));
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(static_cast<ulonglong>(info.faiss_nprobe));
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(static_cast<ulonglong>(info.faiss_pq_m));
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(static_cast<ulonglong>(info.faiss_pq_bits));
    query->append(STRING_WITH_LEN(")"));
  } else if (emit_faiss_ivf_params(info, emit_policy)) {
    append_set_call_prefix(thd, query, "VEC_INDEX_SET_FAISS_IVF_PARAMS",
                           index_name);
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(static_cast<ulonglong>(info.faiss_nlist));
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(static_cast<ulonglong>(info.faiss_nprobe));
    query->append(STRING_WITH_LEN(")"));
  }

  if (emit_diskann_build_params(info, emit_policy)) {
    append_set_call_prefix(thd, query, "VEC_INDEX_SET_DISKANN_BUILD_PARAMS",
                           index_name);
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(static_cast<ulonglong>(info.diskann_max_degree));
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(
        static_cast<ulonglong>(info.diskann_build_complexity));
    if (info.diskann_build_threads != 0) {
      query->append(STRING_WITH_LEN(", "));
      query->append_ulonglong(
          static_cast<ulonglong>(info.diskann_build_threads));
    }
    query->append(STRING_WITH_LEN(")"));
  }

  if (emit_diskann_search_complexity(info, emit_policy)) {
    append_set_call_prefix(thd, query,
                           "VEC_INDEX_SET_DISKANN_SEARCH_COMPLEXITY",
                           index_name);
    query->append(STRING_WITH_LEN(", "));
    query->append_ulonglong(
        static_cast<ulonglong>(info.diskann_search_complexity));
    query->append(STRING_WITH_LEN(")"));
  }
}

}  // namespace vector_index_ddl_formatter
