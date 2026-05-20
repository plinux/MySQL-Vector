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

#include "sql/vector/vector_trx_participant.h"

#include <cstdint>
#include <string>

#include "mysql/plugin.h"
#include "my_base.h"
#include "sql/handler.h"
#include "sql/sql_class.h"
#include "sql/log.h"
#include "sql/mysqld.h"
#include "sql/sql_lex.h"
#include "sql/vector/vector_index_metadata_store.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_truth_store.h"

namespace {

handlerton *vector_trx_hton = nullptr;
unsigned char vector_trx_token = 0;

uint64_t vector_thd_id(const THD *thd) {
  return thd == nullptr ? 0 : static_cast<uint64_t>(thd->thread_id());
}

uint64_t vector_stmt_id(const THD *thd) {
  return thd == nullptr ? 0 : static_cast<uint64_t>(thd->query_id);
}

bool in_multi_stmt_vector_trx(THD *thd) {
  return thd != nullptr && thd->in_multi_stmt_transaction_mode();
}

bool has_vector_trx_context(THD *thd) {
  return thd != nullptr && vector_trx_hton != nullptr &&
         thd_get_ha_data(thd, vector_trx_hton) != nullptr;
}

bool is_real_vector_scope(THD *thd, bool all) {
  return all || !in_multi_stmt_vector_trx(thd);
}

bool skip_recovery_for_bootstrap(THD *thd) {
  return opt_initialize ||
         (thd != nullptr &&
          thd->system_thread == SYSTEM_THREAD_SERVER_INITIALIZE);
}

XID current_xid(THD *thd) {
  XID xid;
  if (thd != nullptr) thd_get_xid(thd, reinterpret_cast<MYSQL_XID *>(&xid));
  return xid;
}

void xid_from_prepared_row(const vector_index_metadata_store::prepared_change_row &row,
                           XID *xid) {
  xid->reset();
  xid->set_format_id(static_cast<long>(row.format_id));
  xid->set_gtrid_length(static_cast<long>(row.gtrid_length));
  xid->set_bqual_length(static_cast<long>(row.bqual_length));
  if (!row.xid_data.empty()) {
    xid->set_data(row.xid_data.data(), static_cast<long>(row.xid_data.size()));
  }
}

bool load_prepared_rows(
    std::vector<vector_index_metadata_store::prepared_change_row> *rows) {
  if (rows == nullptr) return false;
  if (vector_index_truth_store::get()->load_prepared(rows)) return true;

  // Missing prepared rows are represented by a successful empty load. Corrupt
  // prepared rows can be treated as empty only after they are durably
  // quarantined; otherwise the TC could miss an RM that needs recovery.
  rows->clear();
  return vector_index_truth_store::get()->quarantine_prepared();
}

bool init_empty_mod_tables(XA_recover_txn *txn, MEM_ROOT *mem_root) {
  if (txn == nullptr) return false;
  txn->mod_tables = nullptr;
  if (mem_root == nullptr) return true;

  txn->mod_tables = new (mem_root) List<st_handler_tablename>();
  return txn->mod_tables != nullptr;
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
std::string savepoint_token_name(THD *thd, void *savepoint) {
  if (thd != nullptr && thd->lex != nullptr && thd->lex->ident.str != nullptr &&
      thd->lex->ident.length > 0) {
    return std::string(thd->lex->ident.str, thd->lex->ident.length);
  }
  return std::string("ha:") +
         std::to_string(reinterpret_cast<uintptr_t>(savepoint));
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

int vector_trx_savepoint_noop(handlerton *, THD *, void *) { return 0; }

bool vector_trx_savepoint_rollback_can_release_mdl(handlerton *, THD *) {
  return true;
}

int vector_trx_prepare(handlerton *, THD *thd, bool all) {
  if (!has_vector_trx_context(thd) || !is_real_vector_scope(thd, all) ||
      vector_index_truth_store::internal_sql_active()) {
    return 0;
  }
  XID xid = current_xid(thd);
  if (xid.is_null()) return 0;
  return vector_index_registry::prepare_thd_txn(vector_thd_id(thd), xid)
             ? 0
             : HA_ERR_INTERNAL_ERROR;
}

int vector_trx_commit(handlerton *, THD *thd, bool all) {
  if (!has_vector_trx_context(thd)) return 0;
  if (vector_index_truth_store::internal_sql_active()) return 0;
  bool ok = false;
  if (is_real_vector_scope(thd, all)) {
    XID xid = current_xid(thd);
    const xa_status_code xa_ret =
        xid.is_null() ? XAER_NOTA
                      : vector_index_registry::commit_prepared_xid_for_thd(
                            vector_thd_id(thd), xid);
    ok = xa_ret == XA_OK
             ? true
             : (xa_ret == XAER_NOTA
                    ? vector_index_registry::commit_thd_txn(vector_thd_id(thd))
                    : false);
  } else {
    ok = in_multi_stmt_vector_trx(thd)
             ? vector_index_registry::commit_stmt_for_thd_txn(vector_thd_id(thd),
                                                         vector_stmt_id(thd))
             : vector_index_registry::commit_thd_txn(vector_thd_id(thd));
  }
  if (ok && is_real_vector_scope(thd, all)) {
    thd_set_ha_data(thd, vector_trx_hton, nullptr);
  }
  return ok ? 0 : HA_ERR_INTERNAL_ERROR;
}

int vector_trx_rollback(handlerton *, THD *thd, bool all) {
  if (!has_vector_trx_context(thd)) return 0;
  if (vector_index_truth_store::internal_sql_active()) return 0;
  bool ok = false;
  if (is_real_vector_scope(thd, all)) {
    XID xid = current_xid(thd);
    const xa_status_code xa_ret =
        xid.is_null() ? XAER_NOTA
                      : vector_index_registry::rollback_prepared_xid_for_thd(
                            vector_thd_id(thd), xid);
    ok = xa_ret == XA_OK
             ? true
             : (xa_ret == XAER_NOTA
                    ? vector_index_registry::rollback_thd_txn(vector_thd_id(thd))
                    : false);
  } else {
    ok = in_multi_stmt_vector_trx(thd)
             ? vector_index_registry::rollback_stmt_for_thd_txn(vector_thd_id(thd),
                                                           vector_stmt_id(thd))
             : vector_index_registry::rollback_thd_txn(vector_thd_id(thd));
  }
  if (ok && is_real_vector_scope(thd, all)) {
    thd_set_ha_data(thd, vector_trx_hton, nullptr);
  }
  return ok ? 0 : HA_ERR_INTERNAL_ERROR;
}

int vector_trx_recover(handlerton *, XA_recover_txn *txn_list, uint len,
                       MEM_ROOT *mem_root) {
  if (skip_recovery_for_bootstrap(current_thd)) return 0;
  std::vector<vector_index_metadata_store::prepared_change_row> rows;
  if (!load_prepared_rows(&rows) || txn_list == nullptr || len == 0) return 0;

  uint count = 0;
  for (const auto &row : rows) {
    if (count >= len) break;

    XID xid;
    xid_from_prepared_row(row, &xid);
    bool seen = false;
    for (uint i = 0; i < count; ++i) {
      if (txn_list[i].id.eq(&xid)) {
        seen = true;
        break;
      }
    }
    if (seen) continue;

    txn_list[count].id = xid;
    if (!init_empty_mod_tables(&txn_list[count], mem_root)) break;
    ++count;
  }
  return static_cast<int>(count);
}

int vector_trx_recover_prepared_in_tc(handlerton *, Xa_state_list &xa_list) {
  if (skip_recovery_for_bootstrap(current_thd)) return 0;
  std::vector<vector_index_metadata_store::prepared_change_row> rows;
  /*
    The truth-store backend reports a missing prepared artifact as an empty set
    for pre-vector data directories. If corrupt rows can be quarantined, the
    isolated artifact is no longer a recoverable prepared set; if quarantine
    fails, abort recovery instead of pretending there are no prepared rows.
  */
  if (!load_prepared_rows(&rows)) return 1;

  for (const auto &row : rows) {
    if (!row.prepared_in_tc) continue;

    XID xid;
    xid_from_prepared_row(row, &xid);
    if (xid.get_my_xid() != 0) continue;
    (void)xa_list.add(xid, enum_ha_recover_xa_state::PREPARED_IN_TC);
  }
  return 0;
}

xa_status_code vector_trx_commit_by_xid(handlerton *, XID *xid) {
  if (xid == nullptr) return XAER_INVAL;
  if (!mysqld_server_started) {
    vector_index_registry::queue_recovery_commit_xid(*xid);
    return XA_OK;
  }
  return vector_index_registry::commit_prepared_xid_if_loaded(*xid);
}

xa_status_code vector_trx_rollback_by_xid(handlerton *, XID *xid) {
  if (xid == nullptr) return XAER_INVAL;
  if (!mysqld_server_started) {
    vector_index_registry::queue_recovery_rollback_xid(*xid);
    return XA_OK;
  }
  return vector_index_registry::rollback_prepared_xid_if_loaded(*xid);
}

int vector_trx_set_prepared_in_tc(handlerton *, THD *thd) {
  if (thd == nullptr) return HA_ERR_INTERNAL_ERROR;
  if (!has_vector_trx_context(thd)) return 0;
  XID xid = current_xid(thd);
  return xid.is_null() || vector_index_registry::set_prepared_in_tc(xid)
             ? 0
             : HA_ERR_INTERNAL_ERROR;
}

xa_status_code vector_trx_set_prepared_in_tc_by_xid(handlerton *, XID *xid) {
  if (xid == nullptr) return XAER_INVAL;
  if (!mysqld_server_started) {
    vector_index_registry::queue_recovery_set_prepared_in_tc(*xid);
    return XA_OK;
  }
  return vector_index_registry::set_prepared_in_tc(*xid) ? XA_OK : XAER_RMERR;
}

int vector_trx_close_connection(handlerton *, THD *thd) {
  if (thd != nullptr) {
    (void)vector_index_registry::rollback_thd_txn(vector_thd_id(thd));
    thd_set_ha_data(thd, vector_trx_hton, nullptr);
  }
  return 0;
}

int vector_trx_init(void *p) {
  auto *hton = static_cast<handlerton *>(p);
  vector_trx_hton = hton;
  hton->state = SHOW_OPTION_YES;
  hton->db_type = DB_TYPE_UNKNOWN;
  hton->savepoint_offset = 0;
  hton->close_connection = vector_trx_close_connection;
  hton->savepoint_set = vector_trx_savepoint_noop;
  hton->savepoint_rollback = vector_trx_savepoint_noop;
  hton->savepoint_rollback_can_release_mdl =
      vector_trx_savepoint_rollback_can_release_mdl;
  hton->savepoint_release = vector_trx_savepoint_noop;
  hton->commit = vector_trx_commit;
  hton->rollback = vector_trx_rollback;
  hton->prepare = vector_trx_prepare;
  hton->recover = vector_trx_recover;
  hton->recover_prepared_in_tc = vector_trx_recover_prepared_in_tc;
  hton->commit_by_xid = vector_trx_commit_by_xid;
  hton->rollback_by_xid = vector_trx_rollback_by_xid;
  hton->set_prepared_in_tc = vector_trx_set_prepared_in_tc;
  hton->set_prepared_in_tc_by_xid = vector_trx_set_prepared_in_tc_by_xid;
  hton->flags =
      HTON_NOT_USER_SELECTABLE | HTON_HIDDEN | HTON_NO_GLOBAL_2PC;
  return 0;
}

int vector_trx_deinit(void *) {
  vector_trx_hton = nullptr;
  return 0;
}

}  // namespace

namespace vector_trx_participant {

int init_plugin(void *p) { return vector_trx_init(p); }

int deinit_plugin(void *p) { return vector_trx_deinit(p); }

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
uint64_t thd_id_for_testing(const THD *thd) { return vector_thd_id(thd); }

uint64_t stmt_id_for_testing(const THD *thd) { return vector_stmt_id(thd); }

bool in_multi_stmt_for_testing(THD *thd) { return in_multi_stmt_vector_trx(thd); }

bool is_real_scope_for_testing(THD *thd, bool all) {
  return is_real_vector_scope(thd, all);
}

bool skip_recovery_for_bootstrap_for_testing(THD *thd) {
  return skip_recovery_for_bootstrap(thd);
}

std::string savepoint_token_name_for_testing(THD *thd, void *savepoint) {
  return savepoint_token_name(thd, savepoint);
}

bool load_prepared_rows_for_testing(
    std::vector<vector_index_metadata_store::prepared_change_row> *rows) {
  return load_prepared_rows(rows);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

void register_participant(THD *thd) {
  if (thd == nullptr || vector_trx_hton == nullptr) return;

  if (thd_get_ha_data(thd, vector_trx_hton) == nullptr) {
    thd_set_ha_data(thd, vector_trx_hton, &vector_trx_token);
  }

  trans_register_ha(thd, false, vector_trx_hton, nullptr);
  if (thd->in_multi_stmt_transaction_mode()) {
    trans_register_ha(thd, true, vector_trx_hton, nullptr);
  }

  thd->get_ha_data(vector_trx_hton->slot)->ha_info[0].set_trx_read_write();
  auto *all_info = &thd->get_ha_data(vector_trx_hton->slot)->ha_info[1];
  if (all_info->is_started()) all_info->set_trx_read_write();
}

}  // namespace vector_trx_participant

static st_mysql_storage_engine vector_trx_storage_engine{
    MYSQL_HANDLERTON_INTERFACE_VERSION};

mysql_declare_plugin(vector_trx){
    MYSQL_STORAGE_ENGINE_PLUGIN,
    &vector_trx_storage_engine,
    "VECTOR_TRX",
    PLUGIN_AUTHOR_ORACLE,
    "vector_data transaction participant",
    PLUGIN_LICENSE_GPL,
    vector_trx_participant::init_plugin,
    nullptr,
    vector_trx_participant::deinit_plugin,
    0x0100,
    nullptr,
    nullptr,
    nullptr,
    0,
} mysql_declare_plugin_end;
