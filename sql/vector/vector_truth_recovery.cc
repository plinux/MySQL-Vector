/* Copyright (c) 2026, Oracle and/or its affiliates.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0,
   as published by the Free Software Foundation.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

#include "sql/vector/vector_truth_recovery.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "my_base.h"
#include "mysqld_error.h"
#include "sql/handler.h"
#include "sql/mdl.h"
#include "sql/sql_base.h"
#include "sql/sql_class.h"
#include "sql/sql_lex.h"
#include "sql/table.h"
#include "sql/vector/vector_dml_sync.h"

namespace vector_truth_recovery {
namespace {

class scoped_scan_context final {
 public:
  explicit scoped_scan_context(THD *thd) : m_thd(thd) {}

  ~scoped_scan_context() {
    if (!m_active) return;
    close_thread_tables(m_thd);
    m_thd->mdl_context.rollback_to_savepoint(m_mdl_savepoint);
    m_thd->restore_backup_open_tables_state(&m_open_tables_backup);
    m_thd->lex->restore_backup_query_tables_list(&m_query_tables_backup);
  }

  bool activate() {
    if (m_thd == nullptr || m_thd->lex == nullptr) return false;
    m_mdl_savepoint = m_thd->mdl_context.mdl_savepoint();
    m_thd->lex->reset_n_backup_query_tables_list(&m_query_tables_backup);
    m_thd->reset_n_backup_open_tables_state(&m_open_tables_backup, 0);
    m_active = true;
    return true;
  }

  void close_open_tables() { close_thread_tables(m_thd); }

 private:
  THD *m_thd{nullptr};
  Query_tables_list m_query_tables_backup;
  Open_tables_backup m_open_tables_backup;
  MDL_savepoint m_mdl_savepoint;
  bool m_active{false};
};

std::string table_identity(const index_scan_spec &spec) {
  std::string identity = spec.schema_name;
  identity.push_back('\0');
  identity.append(spec.table_name);
  return identity;
}

bool index_scan_spec_less(const index_scan_spec &lhs,
                          const index_scan_spec &rhs) {
  if (lhs.schema_name != rhs.schema_name)
    return lhs.schema_name < rhs.schema_name;
  if (lhs.table_name != rhs.table_name)
    return lhs.table_name < rhs.table_name;
  if (lhs.column_name != rhs.column_name)
    return lhs.column_name < rhs.column_name;
  if (lhs.doc_id_column_name != rhs.doc_id_column_name)
    return lhs.doc_id_column_name < rhs.doc_id_column_name;
  return lhs.index_name < rhs.index_name;
}

bool acquire_table_locks(THD *thd,
                         const std::vector<index_scan_spec> &specs) {
  if (thd == nullptr) return false;

  MDL_request_list requests;
  std::unordered_set<std::string> seen_tables;
  for (const auto &spec : specs) {
    if (!seen_tables.insert(table_identity(spec)).second) continue;
    MDL_request *request = new (thd->mem_root) MDL_request;
    if (request == nullptr) return false;
    MDL_REQUEST_INIT(request, MDL_key::TABLE, spec.schema_name.c_str(),
                     spec.table_name.c_str(), MDL_SHARED_NO_WRITE,
                     MDL_STATEMENT);
    requests.push_front(request);
  }
  return !thd->mdl_context.acquire_locks(&requests,
                                         thd->variables.lock_wait_timeout);
}

bool table_matches_binding(TABLE *table, const index_scan_spec &spec) {
  if (table == nullptr || table->file == nullptr || table->file->ht == nullptr ||
      table->file->ht->db_type != DB_TYPE_INNODB || table->s == nullptr ||
      table->s->primary_key == MAX_KEY) {
    return false;
  }

  const KEY &primary_key = table->key_info[table->s->primary_key];
  if (primary_key.user_defined_key_parts != 1 ||
      primary_key.key_part[0].field == nullptr) {
    return false;
  }
  const Field *doc_id_field = primary_key.key_part[0].field;
  return doc_id_field->result_type() == INT_RESULT &&
         !doc_id_field->is_nullable() &&
         spec.doc_id_column_name == doc_id_field->field_name;
}

bool scan_one_index(THD *thd, const index_scan_spec &spec,
                    index_entries *entries) {
  if (thd == nullptr || entries == nullptr) return false;
  entries->clear();

  Table_ref table_ref(spec.schema_name.c_str(), spec.schema_name.length(),
                      spec.table_name.c_str(), spec.table_name.length(),
                      spec.table_name.c_str(), TL_READ_NO_INSERT);
  table_ref.mdl_request.set_type(MDL_SHARED_NO_WRITE);
  TABLE *table =
      open_n_lock_single_table(thd, &table_ref, TL_READ_NO_INSERT, 0);
  if (table == nullptr || table->file == nullptr) {
    my_error(ER_INTERNAL_ERROR, MYF(0),
             "Failed to open table for vector truth recovery");
    return false;
  }
  if (!table_matches_binding(table, spec)) {
    my_error(ER_INTERNAL_ERROR, MYF(0),
             "Vector truth recovery table binding changed");
    return false;
  }

  table->use_all_columns();
  const int init_error = table->file->ha_rnd_init(true);
  if (init_error != 0) {
    table->file->print_error(init_error, MYF(0));
    return false;
  }

  bool ok = true;
  while (ok) {
    const int scan_error = table->file->ha_rnd_next(table->record[0]);
    if (scan_error == HA_ERR_END_OF_FILE) break;
    if (scan_error == HA_ERR_RECORD_DELETED) continue;
    if (scan_error != 0) {
      table->file->print_error(scan_error, MYF(0));
      ok = false;
      break;
    }

    vector_dml_sync::prepared_changes changes;
    if (vector_dml_sync::prepare_insert_row_for_index(
            table, spec.index_name, spec.column_name, table->record[0],
            &changes)) {
      ok = false;
      break;
    }
    if (changes.empty()) continue;
    if (changes.size() != 1 || changes[0].erase ||
        changes[0].index_name != spec.index_name) {
      my_error(ER_INTERNAL_ERROR, MYF(0),
               "Invalid row produced during vector truth recovery");
      ok = false;
      break;
    }
    (*entries)[changes[0].doc_id] = std::move(changes[0].vector);
  }

  const int end_error = table->file->ha_rnd_end();
  if (end_error != 0 && ok) {
    table->file->print_error(end_error, MYF(0));
    ok = false;
  }
  return ok;
}

}  // namespace

bool scan_and_publish(THD *thd, std::vector<index_scan_spec> specs,
                      const publish_callback &publish) {
  if (thd == nullptr || specs.empty() || !publish) return false;
  std::sort(specs.begin(), specs.end(), index_scan_spec_less);

  scoped_scan_context context(thd);
  if (!context.activate()) return false;

  if (!acquire_table_locks(thd, specs)) return false;

  recovered_state state;
  state.reserve(specs.size());
  for (const auto &spec : specs) {
    if (!scan_one_index(thd, spec, &state[spec.index_name])) return false;
    context.close_open_tables();
  }
  return publish(state);
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool index_scan_spec_less_for_testing(const index_scan_spec &lhs,
                                      const index_scan_spec &rhs) {
  return index_scan_spec_less(lhs, rhs);
}

bool table_matches_binding_for_testing(TABLE *table,
                                       const index_scan_spec &spec) {
  return table_matches_binding(table, spec);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_truth_recovery
