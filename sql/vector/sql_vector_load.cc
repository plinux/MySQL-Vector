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

#include "sql/vector/sql_vector_load.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "m_ctype.h"
#include "my_byteorder.h"
#include "mysql_com.h"
#include "mysqld_error.h"
#include "scope_guard.h"
#include "sql/auth/auth_acls.h"
#include "sql/auth/auth_common.h"
#include "sql/binlog.h"
#include "sql/debug_sync.h"
#include "sql/log_event.h"
#include "sql/mysqld.h"
#include "sql/protocol.h"
#include "sql/sql_class.h"

#ifdef HAVE_VECTOR_INDEX
#include "sql/vector/item_vectorfunc_internal.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_load_file.h"
#include "sql/vector/vector_load_staging.h"
#include "sql/vector/vector_statement_publication.h"
#include "sql/vector/vector_trx_participant.h"
#endif

namespace {

#ifdef HAVE_VECTOR_INDEX
constexpr const char *k_load_vector_data_command = "load vector data";
constexpr size_t k_load_vector_binlog_chunk_bytes = 256 * 1024;
using bulk_load_reader = vector_index::index_service::bulk_load_reader;
using bulk_load_visitor = vector_index::index_service::bulk_load_visitor;
#endif

std::string to_std_string(const LEX_STRING &value) {
  return value.str == nullptr ? std::string()
                              : std::string(value.str, value.length);
}

#ifdef HAVE_VECTOR_INDEX
bool equals_ci(const std::string &lhs, const char *rhs) {
  return my_strcasecmp(system_charset_info, lhs.c_str(), rhs) == 0;
}

bool check_load_vector_secure_file_path(const std::string &filename) {
  if (!filename.empty() && !is_secure_file_path(filename.c_str())) {
    my_error(ER_OPTION_PREVENTS_STATEMENT, MYF(0), "--secure-file-priv");
    return true;
  }
  return false;
}

bool check_load_vector_file_access(THD *thd, bool is_local_file,
                                   const std::string &vector_filename,
                                   const std::string &docid_filename) {
  if (is_local_file) {
    if (!thd->get_protocol()->has_client_capability(CLIENT_LOCAL_FILES) ||
        !opt_local_infile) {
      my_error(ER_CLIENT_LOCAL_FILES_DISABLED, MYF(0));
      return true;
    }
    return false;
  }

  if (check_access(thd, FILE_ACL, any_db, nullptr, nullptr, false, false))
    return true;

  return check_load_vector_secure_file_path(vector_filename) ||
         check_load_vector_secure_file_path(docid_filename);
}

bool should_binlog_load_vector(THD *thd) {
  return thd != nullptr && mysql_bin_log.is_open() &&
         (thd->variables.option_bits & OPTION_BIN_LOG) != 0 &&
         !thd->slave_thread && !thd->in_sub_stmt;
}

void append_doc_id_bytes(uint64_t doc_id, std::string *buffer) {
  const size_t offset = buffer->size();
  buffer->resize(offset + sizeof(uint64_t));
  int8store(reinterpret_cast<uchar *>(&(*buffer)[offset]), doc_id);
}

void append_vector_bytes(const float *values, size_t dimension,
                         std::string *buffer) {
  const size_t offset = buffer->size();
  buffer->resize(offset + dimension * sizeof(float));
  uchar *ptr = reinterpret_cast<uchar *>(&(*buffer)[offset]);
  for (size_t dim_idx = 0; dim_idx < dimension; ++dim_idx) {
    float4store(ptr + dim_idx * sizeof(float), values[dim_idx]);
  }
}

bool append_index_name_literal(THD *thd, const std::string &index_name,
                               String *statement) {
  String name(index_name.c_str(), index_name.length(), system_charset_info);
  return append_query_string(thd, system_charset_info, &name, statement) != 0;
}

bool begin_load_vector_stmt(THD *thd, const char *function_name,
                            const std::string &index_name, String *statement,
                            std::string *error) {
  String local_statement;
  local_statement.set_charset(system_charset_info);
  local_statement.append(STRING_WITH_LEN("SELECT "));
  local_statement.append(function_name);
  local_statement.append(STRING_WITH_LEN("("));
  if (append_index_name_literal(thd, index_name, &local_statement)) {
    if (error != nullptr) *error = "LOAD VECTOR DATA could not encode index name";
    return false;
  }
  *statement = std::move(local_statement);
  return true;
}

bool write_load_vector_batch_event(THD *thd, const std::string &index_name,
                                   const std::string &docid_blob,
                                   const std::string &vector_blob,
                                   size_t row_count, std::string *error) {
  String statement;
  if (!begin_load_vector_stmt(thd, "VEC_INDEX_UPSERT_BATCH", index_name,
                              &statement, error)) {
    return false;
  }
  const std::string docid_hex =
      vector_itemfunc_internal::encode_hex_bytes(docid_blob);
  const std::string vector_hex =
      vector_itemfunc_internal::encode_hex_bytes(vector_blob);
  statement.append(STRING_WITH_LEN(", X'"));
  statement.append(docid_hex.c_str(), docid_hex.length());
  statement.append(STRING_WITH_LEN("', X'"));
  statement.append(vector_hex.c_str(), vector_hex.length());
  statement.append(STRING_WITH_LEN("', "));
  statement.append_ulonglong(static_cast<ulonglong>(row_count));
  statement.append(STRING_WITH_LEN(")"));
  return thd->binlog_query(THD::STMT_QUERY_TYPE, statement.ptr(),
                           statement.length(), false, false, false, 0) == 0;
}

bool write_load_vector_rebuild_event(THD *thd, const std::string &index_name,
                                     std::string *error) {
  String statement;
  if (!begin_load_vector_stmt(thd, "VEC_INDEX_BULK_BUILD", index_name,
                              &statement, error)) {
    return false;
  }
  statement.append(STRING_WITH_LEN(")"));
  return thd->binlog_query(THD::STMT_QUERY_TYPE, statement.ptr(),
                           statement.length(), false, false, false, 0) == 0;
}

bulk_load_reader make_csv_load_reader(const std::string &vector_filename,
                                      uint64_t *loaded_rows) {
  return [vector_filename,
          loaded_rows](const bulk_load_visitor &visitor,
                       std::string *reader_error) {
    return vector_index::read_csv_vectors(
        vector_filename, 0, nullptr, reader_error,
        [loaded_rows, &visitor](uint64_t doc_id, const float *values,
                                size_t dimension) {
          if (!visitor(doc_id, values, dimension)) return false;
          if (loaded_rows != nullptr) ++(*loaded_rows);
          return true;
        });
  };
}

bulk_load_reader make_fbin_load_reader(const std::string &vector_filename,
                                       const std::string &docid_filename,
                                       size_t dimension) {
  return [dimension, docid_filename,
          vector_filename](const bulk_load_visitor &visitor,
                           std::string *reader_error) {
    vector_index::vector_load_file_info file_info;
    return vector_index::read_fbin_vectors(
        vector_filename, docid_filename, dimension, &file_info, reader_error,
        [&visitor](uint64_t doc_id, const float *values, size_t read_dimension) {
          return visitor(doc_id, values, read_dimension);
        });
  };
}

bool binlog_load_vector_rows(
    THD *thd, const std::string &index_name, size_t dimension,
    const bulk_load_reader &reader,
    bool rebuild_after_load, bool is_standalone, std::string *error) {
  if (!should_binlog_load_vector(thd)) return true;
  if (!reader) {
    if (error != nullptr) *error = "LOAD VECTOR DATA reader is empty";
    return false;
  }

  const size_t row_binary_bytes = sizeof(uint64_t) + dimension * sizeof(float);
  const size_t chunk_budget =
      std::max<size_t>(row_binary_bytes, std::min<size_t>(
                                            k_load_vector_binlog_chunk_bytes,
                                            thd->variables.max_allowed_packet /
                                                4));
  std::string docid_blob;
  std::string vector_blob;
  size_t row_count = 0;

  auto flush_chunk = [&]() -> bool {
    if (row_count == 0) return true;
    if (!write_load_vector_batch_event(thd, index_name, docid_blob, vector_blob,
                                       row_count, error)) {
      if (error != nullptr && error->empty()) {
        *error = "LOAD VECTOR DATA could not write binlog batch";
      }
      return false;
    }
    docid_blob.clear();
    vector_blob.clear();
    row_count = 0;
    return true;
  };

  std::string reader_error;
  const bool read_ok = reader(
      [&](uint64_t doc_id, const float *values, size_t read_dimension) {
        if (values == nullptr || read_dimension != dimension) return false;
        if (row_count > 0 &&
            docid_blob.size() + vector_blob.size() + row_binary_bytes >
                chunk_budget &&
            !flush_chunk()) {
          return false;
        }
        append_doc_id_bytes(doc_id, &docid_blob);
        append_vector_bytes(values, dimension, &vector_blob);
        ++row_count;
        return true;
      },
      &reader_error);
  if (!read_ok) {
    if (error != nullptr && error->empty()) {
      *error =
          reader_error.empty() ? "LOAD VECTOR DATA reader failed" : reader_error;
    }
    return false;
  }
  if (!flush_chunk()) return false;

  if (is_standalone && rebuild_after_load &&
      !write_load_vector_rebuild_event(thd, index_name, error)) {
    if (error != nullptr && error->empty()) {
      *error = "LOAD VECTOR DATA could not write binlog rebuild";
    }
    return false;
  }
  return true;
}

bool stage_transactional_load_rows(THD *thd, const std::string &index_name,
                                   const bulk_load_reader &reader,
                                   uint64_t *loaded_rows,
                                   std::string *error) {
  constexpr size_t k_stage_chunk_rows = 256;
  if (thd == nullptr || !reader || loaded_rows == nullptr ||
      !vector_trx_participant::register_participant(thd)) {
    return false;
  }

  std::vector<vector_index::index_service::pending_change_snapshot> changes;
  changes.reserve(k_stage_chunk_rows);
  const auto flush_changes = [&]() {
    if (changes.empty()) return true;
    const bool ok = vector_index_registry::stage_changes_for_thd_txn(
        thd, static_cast<uint64_t>(thd->query_id), changes);
    changes.clear();
    return ok;
  };

  std::string reader_error;
  const bool read_ok = reader(
      [&](uint64_t doc_id, const float *values, size_t dimension) {
        if (thd->killed || values == nullptr || dimension == 0) return false;
        vector_index::index_service::pending_change_snapshot change;
        change.index_name = index_name;
        change.doc_id = doc_id;
        change.vector.assign(values, values + dimension);
        changes.push_back(std::move(change));
        ++(*loaded_rows);
        return changes.size() < k_stage_chunk_rows || flush_changes();
      },
      &reader_error);
  if (!read_ok || !flush_changes()) {
    if (error != nullptr && error->empty()) {
      *error = reader_error.empty()
                   ? "LOAD VECTOR DATA could not stage transactional rows"
                   : reader_error;
    }
    return false;
  }
  return true;
}

bool validate_load_rows(const bulk_load_reader &reader,
                        const std::string &index_name, size_t dimension,
                        bool replace_duplicates, uint64_t *loaded_rows,
                        std::string *error) {
  if (!reader || index_name.empty() || loaded_rows == nullptr ||
      dimension == 0) {
    return false;
  }
  std::unordered_set<uint64_t> seen_doc_ids;
  std::string validation_error;
  std::string reader_error;
  const bool ok = reader(
      [&](uint64_t doc_id, const float *values, size_t read_dimension) {
        if (current_thd != nullptr && current_thd->killed) {
          validation_error = "LOAD VECTOR DATA interrupted";
          return false;
        }
        if (values == nullptr) {
          validation_error = "LOAD VECTOR DATA invalid vector row";
          return false;
        }
        if (read_dimension != dimension) {
          validation_error = "LOAD VECTOR DATA dimension mismatch";
          return false;
        }
        if (!replace_duplicates) {
          if (!seen_doc_ids.insert(doc_id).second) {
            validation_error = "LOAD VECTOR DATA duplicate doc_id in file";
            return false;
          }
          bool found = false;
          if (!vector_index_registry::entry_exists(index_name, doc_id,
                                                   &found)) {
            validation_error =
                "LOAD VECTOR DATA could not read index state";
            return false;
          }
          if (found) {
            validation_error = "LOAD VECTOR DATA duplicate doc_id in index";
            return false;
          }
        }
        ++(*loaded_rows);
        return true;
      },
      &reader_error);
  if (!ok && error != nullptr) {
    if (!validation_error.empty()) {
      *error = validation_error;
    } else if (!reader_error.empty()) {
      *error = reader_error;
    } else {
      *error = "LOAD VECTOR DATA input validation failed";
    }
  }
  return ok;
}
#endif

}  // namespace

Sql_cmd_load_vector_index::Sql_cmd_load_vector_index(
    bool is_local_file, const LEX_STRING &vector_filename,
    const LEX_STRING &docid_filename, const LEX_STRING &index_name,
    const LEX_STRING &format, bool replace_duplicates, bool rebuild_after_load)
    : m_is_local_file(is_local_file),
      m_vector_filename(to_std_string(vector_filename)),
      m_docid_filename(to_std_string(docid_filename)),
      m_index_name(to_std_string(index_name)),
      m_format(to_std_string(format)),
      m_replace_duplicates(replace_duplicates),
      m_rebuild_after_load(rebuild_after_load) {}

bool Sql_cmd_load_vector_index::execute(THD *thd) {
#ifndef HAVE_VECTOR_INDEX
  (void)thd;
  my_error(ER_NOT_SUPPORTED_YET, MYF(0), "LOAD VECTOR DATA");
  return true;
#else
  if (m_vector_filename.empty() || m_index_name.empty() || m_format.empty()) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), k_load_vector_data_command);
    return true;
  }

  const bool format_fbin = equals_ci(m_format, "FBIN");
  const bool format_csv = equals_ci(m_format, "CSV");
  if (!format_fbin && !format_csv) {
    const std::string unsupported_format =
        "LOAD VECTOR DATA FORMAT " + m_format;
    my_error(ER_NOT_SUPPORTED_YET, MYF(0), unsupported_format.c_str());
    return true;
  }

  vector_index_registry::index_info info;
  if (!vector_index_registry::get_index_info(m_index_name, &info)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), k_load_vector_data_command);
    return true;
  }

  if (vector_itemfunc_internal::check_vector_existing_index_access(
          thd, m_index_name, ALTER_ACL, k_load_vector_data_command, false)) {
    return true;
  }

  if (format_csv && !m_docid_filename.empty()) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), k_load_vector_data_command);
    return true;
  }

  if (check_load_vector_file_access(thd, m_is_local_file, m_vector_filename,
                                    m_docid_filename))
    return true;

  if (m_is_local_file) {
    my_error(ER_NOT_SUPPORTED_YET, MYF(0), "LOAD VECTOR DATA LOCAL");
    return true;
  }

  const bool is_standalone = info.consistency_mode == "standalone";
  uint64_t loaded_rows = 0;
  std::string error;
  bool staged = false;
  if (is_standalone) {
    if (vector_trx_participant::in_user_multi_statement_transaction(thd)) {
      my_error(ER_LOCK_OR_ACTIVE_TRANSACTION, MYF(0));
      return true;
    }
  }

  vector_index_truth_store::publication_intent intent;
  if (!vector_index_registry::make_statement_publication_intent(
          vector_index_truth_store::publication_operation::kBulkLoad,
          m_index_name, std::string(), &intent)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), k_load_vector_data_command);
    return true;
  }
  const vector_index::load_staging_identity staging_identity{
      intent.index_name, intent.publication_id};
  vector_index::load_staging_artifact staging_artifact;
  const std::string normalized_format = format_fbin ? "FBIN" : "CSV";
  if (!vector_index::stage_load_artifact(staging_identity, m_vector_filename,
                                         m_docid_filename, normalized_format,
                                         &staging_artifact, &error)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0),
             error.empty() ? k_load_vector_data_command : error.c_str());
    return true;
  }
  auto staging_cleanup = create_scope_guard([&]() {
    (void)vector_index::remove_load_staging_artifact(staging_identity);
  });

  vector_index::load_staging_paths staging_paths;
  if (!vector_index::verify_load_artifact(staging_identity, staging_artifact,
                                          normalized_format, &staging_paths,
                                          &error)) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0),
             error.empty() ? k_load_vector_data_command : error.c_str());
    return true;
  }
  DEBUG_SYNC(thd, "vector_load_after_staging");
  const bulk_load_reader load_reader =
      format_fbin
          ? make_fbin_load_reader(staging_paths.vector_filename,
                                  staging_paths.docid_filename, info.dimension)
          : make_csv_load_reader(staging_paths.vector_filename, nullptr);

  if (is_standalone) {
    if (!validate_load_rows(load_reader, m_index_name, info.dimension,
                            m_replace_duplicates, &loaded_rows, &error) ||
        !binlog_load_vector_rows(thd, m_index_name, info.dimension, load_reader,
                                 m_rebuild_after_load, true, &error)) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0),
               error.empty() ? k_load_vector_data_command : error.c_str());
      return true;
    }

    vector_statement_publication::operation_payload payload;
    std::vector<vector_index_truth_store::publication_intent> intents;
    if (vector_index::make_load_staging_payload(
            staging_identity, staging_artifact, m_replace_duplicates,
            m_rebuild_after_load, info.dimension, normalized_format,
            &payload) &&
        vector_statement_publication::encode_operation_payload(
            payload, &intent.payload)) {
      intents.push_back(std::move(intent));
      staged = vector_trx_participant::stage_statement_publication(
          thd, std::move(intents), false);
    }
    if (!staged) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0), k_load_vector_data_command);
      return true;
    }
    staging_cleanup.release();
  } else {
    if (!validate_load_rows(load_reader, m_index_name, info.dimension,
                            m_replace_duplicates, &loaded_rows, &error)) {
      my_error(ER_WRONG_ARGUMENTS, MYF(0),
               error.empty() ? k_load_vector_data_command : error.c_str());
      return true;
    }
    uint64_t staged_rows = 0;
    staged = stage_transactional_load_rows(thd, m_index_name, load_reader,
                                           &staged_rows, &error);
    if (staged && staged_rows != loaded_rows) {
      error = "LOAD VECTOR DATA staged row count mismatch";
      staged = false;
    }
    if (staged) {
      staged = binlog_load_vector_rows(thd, m_index_name, info.dimension,
                                       load_reader, m_rebuild_after_load, false,
                                       &error);
    }
  }
  if (!staged) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0),
             error.empty() ? k_load_vector_data_command : error.c_str());
    return true;
  }

  my_ok(thd, loaded_rows);
  return false;
#endif
}
