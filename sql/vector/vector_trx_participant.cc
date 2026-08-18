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
#include <mutex>
#include <string>
#include <unordered_set>

#include "mysql/components/services/log_builtins.h"
#include "mysql/plugin.h"
#include "my_base.h"
#include "sql/handler.h"
#include "sql/replication.h"
#include "sql/sql_class.h"
#include "sql/sql_lex.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_truth_store.h"

namespace {

handlerton *vector_trx_hton = nullptr;
unsigned char vector_trx_token = 0;

std::mutex publication_mutex;
std::unordered_set<my_thread_id> publication_threads;
std::mutex observer_registration_mutex;
bool observer_registered = false;
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool registration_bypass_for_testing = false;
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

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

bool is_xa_commit_publication_fallback(THD *thd, my_thread_id thread_id) {
  return thd != nullptr && thd->thread_id() == thread_id && thd->lex != nullptr &&
         thd->lex->sql_command == SQLCOM_XA_COMMIT;
}

bool is_real_vector_scope(THD *thd, bool all) {
  return all || !in_multi_stmt_vector_trx(thd);
}

/*
  SQL transaction entry points own vector savepoint state. They also observe
  savepoints created before this hidden handlerton joins the transaction.
  These callbacks remain installed only so MySQL accepts the participant in a
  transaction that already has savepoints; mutating state here would process
  rollback/release twice.
*/
int vector_trx_savepoint_set(handlerton *, THD *, void *) { return 0; }

int vector_trx_savepoint_rollback(handlerton *, THD *, void *) { return 0; }

int vector_trx_savepoint_release(handlerton *, THD *, void *) { return 0; }

bool vector_trx_savepoint_rollback_can_release_mdl(handlerton *, THD *) {
  return true;
}

int vector_trx_prepare(handlerton *, THD *thd, bool all) {
  if (!has_vector_trx_context(thd) || !is_real_vector_scope(thd, all)) {
    return 0;
  }
  return vector_index_registry::detach_thd_txn_for_prepare(vector_thd_id(thd))
             ? 0
             : HA_ERR_INTERNAL_ERROR;
}

int vector_trx_set_prepared_in_tc(handlerton *, THD *) {
  // InnoDB owns the XA record; the vector participant has no durable TC state.
  return 0;
}

int vector_trx_commit(handlerton *, THD *thd, bool all) {
  if (!has_vector_trx_context(thd)) return 0;
  if (is_real_vector_scope(thd, all)) {
    {
      std::lock_guard<std::mutex> guard(publication_mutex);
      publication_threads.insert(thd->thread_id());
    }
    thd->get_transaction()->m_flags.run_hooks = true;
    thd_set_ha_data(thd, vector_trx_hton, nullptr);
    return 0;
  }

  const bool ok = vector_index_registry::commit_stmt_for_thd_txn(
      vector_thd_id(thd), vector_stmt_id(thd));
  return ok ? 0 : HA_ERR_INTERNAL_ERROR;
}

int vector_trx_rollback(handlerton *, THD *thd, bool all) {
  if (!has_vector_trx_context(thd)) return 0;
  bool ok = false;
  if (is_real_vector_scope(thd, all)) {
    ok = vector_index_registry::rollback_thd_txn(vector_thd_id(thd));
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

int vector_trx_close_connection(handlerton *, THD *thd) {
  if (thd != nullptr) {
    {
      std::lock_guard<std::mutex> guard(publication_mutex);
      publication_threads.erase(thd->thread_id());
    }
    (void)vector_index_registry::rollback_thd_txn(vector_thd_id(thd));
    thd_set_ha_data(thd, vector_trx_hton, nullptr);
  }
  return 0;
}

void vector_trx_after_commit(void *arg) {
  auto *param = static_cast<Trans_param *>(arg);
  if (param == nullptr || (param->flags & TRANS_IS_REAL_TRANS) == 0) return;

  bool publish = false;
  {
    std::lock_guard<std::mutex> guard(publication_mutex);
    publish = publication_threads.erase(param->thread_id) != 0;
  }
  const bool recover_detached_xa =
      is_xa_commit_publication_fallback(current_thd, param->thread_id);
  if (!publish && !recover_detached_xa) {
    return;
  }
  std::string failure_stage;
  if (!vector_index_registry::publish_thd_txn(
          static_cast<uint64_t>(param->thread_id), recover_detached_xa,
          &failure_stage)) {
    const std::string message =
        "Vector runtime publication failed after transaction commit at stage '" +
        (failure_stage.empty() ? "unknown" : failure_stage) + "'";
    LogErr(ERROR_LEVEL, ER_LOG_PRINTF_MSG, message.c_str());
  }
}

void vector_trx_before_rollback(void *arg) {
  auto *param = static_cast<Trans_param *>(arg);
  if (param == nullptr || (param->flags & TRANS_IS_REAL_TRANS) == 0) return;

  {
    std::lock_guard<std::mutex> guard(publication_mutex);
    publication_threads.erase(param->thread_id);
  }
  (void)vector_index_registry::rollback_thd_txn(
      static_cast<uint64_t>(param->thread_id));
}

int vector_trx_observe_before_rollback(Trans_param *param) {
  vector_trx_before_rollback(param);
  return 0;
}

int vector_trx_observe_after_commit(Trans_param *param) {
  vector_trx_after_commit(param);
  return 0;
}

Trans_observer vector_trx_observer{
    sizeof(Trans_observer),
    nullptr,
    nullptr,
    vector_trx_observe_before_rollback,
    vector_trx_observe_after_commit,
    nullptr,
    nullptr,
};

bool ensure_observer_registered_impl() {
  std::lock_guard<std::mutex> guard(observer_registration_mutex);
  if (observer_registered) return true;
  if (vector_trx_hton == nullptr || vector_trx_hton->slot == HA_SLOT_UNDEF) {
    return false;
  }

  st_plugin_int *plugin = hton2plugin(vector_trx_hton->slot);
  if (plugin == nullptr || register_trans_observer(&vector_trx_observer, plugin)) {
    return false;
  }
  observer_registered = true;
  return true;
}

int vector_trx_init(void *p) {
  auto *hton = static_cast<handlerton *>(p);
  {
    std::lock_guard<std::mutex> guard(publication_mutex);
    publication_threads.clear();
  }
  {
    std::lock_guard<std::mutex> guard(observer_registration_mutex);
    observer_registered = false;
  }
  vector_trx_hton = hton;
  hton->state = SHOW_OPTION_YES;
  hton->db_type = DB_TYPE_UNKNOWN;
  hton->savepoint_offset = 0;
  hton->close_connection = vector_trx_close_connection;
  hton->savepoint_set = vector_trx_savepoint_set;
  hton->savepoint_rollback = vector_trx_savepoint_rollback;
  hton->savepoint_rollback_can_release_mdl =
      vector_trx_savepoint_rollback_can_release_mdl;
  hton->savepoint_release = vector_trx_savepoint_release;
  hton->commit = vector_trx_commit;
  hton->rollback = vector_trx_rollback;
  hton->prepare = vector_trx_prepare;
  hton->set_prepared_in_tc = vector_trx_set_prepared_in_tc;
  hton->flags =
      HTON_NOT_USER_SELECTABLE | HTON_HIDDEN | HTON_NO_GLOBAL_2PC;
  return 0;
}

int vector_trx_deinit(void *) {
  {
    std::lock_guard<std::mutex> guard(observer_registration_mutex);
    if (observer_registered) {
      (void)unregister_trans_observer(&vector_trx_observer, nullptr);
      observer_registered = false;
    }
  }
  {
    std::lock_guard<std::mutex> guard(publication_mutex);
    publication_threads.clear();
  }
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

bool is_xa_commit_publication_fallback_for_testing(THD *thd,
                                                   uint64_t thread_id) {
  return is_xa_commit_publication_fallback(
      thd, static_cast<my_thread_id>(thread_id));
}

void set_registration_bypass_for_testing(bool bypass) {
  registration_bypass_for_testing = bypass;
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

bool ensure_observer_registered() {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  if (registration_bypass_for_testing) return true;
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
  return ensure_observer_registered_impl();
}

bool register_participant(THD *thd) {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  if (registration_bypass_for_testing) return thd != nullptr;
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
  if (thd == nullptr || vector_trx_hton == nullptr ||
      !vector_trx_participant::ensure_observer_registered()) {
    return false;
  }

  if (thd_get_ha_data(thd, vector_trx_hton) == nullptr) {
    thd_set_ha_data(thd, vector_trx_hton, &vector_trx_token);
  }

  trans_register_ha(thd, false, vector_trx_hton, nullptr);
  if (thd->in_multi_stmt_transaction_mode()) {
    trans_register_ha(thd, true, vector_trx_hton, nullptr);
  }
  return true;
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
