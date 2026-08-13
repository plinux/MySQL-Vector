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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "my_base.h"
#include "mysql/components/services/log_builtins.h"
#include "mysql/plugin.h"
#include "sql/handler.h"
#include "sql/replication.h"
#include "sql/sql_class.h"
#include "sql/sql_lex.h"
#include "sql/vector/vector_index_registry.h"
#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_statement_publication.h"
#include "sql/xa.h"

namespace {

handlerton *vector_trx_hton = nullptr;
unsigned char vector_trx_participant_token = 0;
unsigned char vector_explicit_txn_token = 0;
unsigned char vector_combined_txn_token = 0;

enum vector_context_role : unsigned int {
  kNoVectorContext = 0,
  kTransactionParticipant = 1U << 0,
  kExplicitTxnOwner = 1U << 1,
};

std::mutex publication_mutex;
std::unordered_set<my_thread_id> publication_threads;
std::unordered_map<my_thread_id,
                   std::unique_ptr<vector_statement_publication::publication_guard>>
    publication_guards;
struct statement_publication_context {
  uint64_t statement_id{0};
  bool catalog_exclusive{false};
  std::vector<vector_index_truth_store::publication_intent> intents;
};
std::unordered_map<my_thread_id, statement_publication_context>
    statement_publications;
std::mutex observer_registration_mutex;
bool observer_registered = false;
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool registration_bypass_for_testing = false;
bool context_override_enabled_for_testing = false;
THD *context_thd_for_testing = nullptr;
unsigned int context_roles_for_testing = kNoVectorContext;
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

unsigned int vector_context_roles(THD *thd) {
  if (thd == nullptr) return kNoVectorContext;
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  if (context_override_enabled_for_testing)
    return context_thd_for_testing == thd ? context_roles_for_testing
                                          : kNoVectorContext;
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
  if (vector_trx_hton == nullptr) return kNoVectorContext;

  const void *context = thd_get_ha_data(thd, vector_trx_hton);
  if (context == &vector_trx_participant_token) return kTransactionParticipant;
  if (context == &vector_explicit_txn_token) return kExplicitTxnOwner;
  if (context == &vector_combined_txn_token)
    return kTransactionParticipant | kExplicitTxnOwner;
  return kNoVectorContext;
}

void set_vector_context_roles(THD *thd, unsigned int roles) {
  if (thd == nullptr) return;
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  if (context_override_enabled_for_testing) {
    context_thd_for_testing = roles == kNoVectorContext ? nullptr : thd;
    context_roles_for_testing = roles;
    return;
  }
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
  if (vector_trx_hton == nullptr) return;
  void *context = nullptr;
  if (roles == kTransactionParticipant) {
    context = &vector_trx_participant_token;
  } else if (roles == kExplicitTxnOwner) {
    context = &vector_explicit_txn_token;
  } else if (roles == (kTransactionParticipant | kExplicitTxnOwner)) {
    context = &vector_combined_txn_token;
  }
  thd_set_ha_data(thd, vector_trx_hton, context);
}

bool has_vector_trx_context(THD *thd) {
  return (vector_context_roles(thd) & kTransactionParticipant) != 0;
}

void clear_vector_trx_context(THD *thd) {
  set_vector_context_roles(thd,
                           vector_context_roles(thd) & ~kTransactionParticipant);
}

void clear_all_vector_context(THD *thd) {
  set_vector_context_roles(thd, kNoVectorContext);
}

bool is_xa_commit_publication_fallback(THD *thd, my_thread_id thread_id) {
  return thd != nullptr && thd->thread_id() == thread_id &&
         thd->lex != nullptr && thd->lex->sql_command == SQLCOM_XA_COMMIT;
}

bool is_real_vector_scope(THD *thd, bool all) {
  return all || !in_multi_stmt_vector_trx(thd);
}

bool prepare_vector_publication(THD *thd) {
  if (thd == nullptr || !has_vector_trx_context(thd)) return true;

  bool has_publication_guard = false;
  {
    std::lock_guard<std::mutex> guard(publication_mutex);
    has_publication_guard = publication_guards.find(thd->thread_id()) !=
                            publication_guards.end();
  }

  std::vector<std::string> index_names;
  if (!vector_index_registry::pending_index_names_for_thd_txn(
          vector_thd_id(thd), &index_names)) {
    return false;
  }
  // The handlerton prepare callback and the binlog observer can both reach
  // this function for the same commit. Once the guard exists, publication
  // metadata and index ownership have already been prepared.
  if (has_publication_guard) return true;

  bool catalog_exclusive = false;
  {
    std::lock_guard<std::mutex> guard(publication_mutex);
    const auto statement_it = statement_publications.find(thd->thread_id());
    if (statement_it != statement_publications.end()) {
      if (statement_it->second.statement_id != vector_stmt_id(thd)) {
        return false;
      }
      catalog_exclusive = statement_it->second.catalog_exclusive;
      for (const auto &intent : statement_it->second.intents) {
        index_names.push_back(intent.index_name);
      }
    }
  }
  if (index_names.empty()) return true;

  auto publication_guard =
      std::make_unique<vector_statement_publication::publication_guard>();
  const bool locked = catalog_exclusive
                          ? publication_guard->lock_catalog()
                          : publication_guard->lock_indexes(index_names);
  if (!locked || !vector_index_registry::prepare_thd_txn_publication(
                     thd, vector_thd_id(thd))) {
    return false;
  }

  std::lock_guard<std::mutex> guard(publication_mutex);
  return publication_guards
      .emplace(thd->thread_id(), std::move(publication_guard))
      .second;
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
  if (!prepare_vector_publication(thd)) return HA_ERR_INTERNAL_ERROR;
  if (!is_xa_prepare(thd)) return 0;
  if (!vector_index_registry::detach_thd_txn_for_prepare(vector_thd_id(thd))) {
    return HA_ERR_INTERNAL_ERROR;
  }
  {
    std::lock_guard<std::mutex> guard(publication_mutex);
    publication_guards.erase(thd->thread_id());
  }
  return 0;
}

int vector_trx_set_prepared_in_tc(handlerton *, THD *) {
  // InnoDB owns the XA record; the vector participant has no durable TC state.
  return 0;
}

int vector_trx_commit(handlerton *, THD *thd, bool all) {
  if (!has_vector_trx_context(thd)) return 0;
  if (is_real_vector_scope(thd, all)) {
    if (!prepare_vector_publication(thd)) return HA_ERR_INTERNAL_ERROR;
    {
      std::lock_guard<std::mutex> guard(publication_mutex);
      publication_threads.insert(thd->thread_id());
    }
    thd->get_transaction()->m_flags.run_hooks = true;
    clear_vector_trx_context(thd);
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
             ? vector_index_registry::rollback_stmt_for_thd_txn(
                   vector_thd_id(thd), vector_stmt_id(thd))
             : vector_index_registry::rollback_thd_txn(vector_thd_id(thd));
  }
  if (ok && is_real_vector_scope(thd, all)) {
    {
      std::lock_guard<std::mutex> guard(publication_mutex);
      publication_threads.erase(thd->thread_id());
      publication_guards.erase(thd->thread_id());
      statement_publications.erase(thd->thread_id());
    }
    clear_vector_trx_context(thd);
  }
  return ok ? 0 : HA_ERR_INTERNAL_ERROR;
}

int vector_trx_close_connection(handlerton *, THD *thd) {
  if (thd != nullptr) {
    {
      std::lock_guard<std::mutex> guard(publication_mutex);
      publication_threads.erase(thd->thread_id());
      publication_guards.erase(thd->thread_id());
      statement_publications.erase(thd->thread_id());
    }
    (void)vector_index_registry::rollback_thd_txn(vector_thd_id(thd));
    (void)vector_index_registry::rollback_explicit_txns_for_thd(thd);
    clear_all_vector_context(thd);
  }
  return 0;
}

void vector_trx_after_commit(void *arg) {
  auto *param = static_cast<Trans_param *>(arg);
  if (param == nullptr || (param->flags & TRANS_IS_REAL_TRANS) == 0) return;

  bool publish = false;
  statement_publication_context statement_publication;
  bool publish_statement = false;
  std::unique_ptr<vector_statement_publication::publication_guard>
      publication_guard;
  {
    std::lock_guard<std::mutex> guard(publication_mutex);
    publish = publication_threads.erase(param->thread_id) != 0;
    auto guard_it = publication_guards.find(param->thread_id);
    if (guard_it != publication_guards.end()) {
      publication_guard = std::move(guard_it->second);
      publication_guards.erase(guard_it);
    }
    auto statement_it = statement_publications.find(param->thread_id);
    if (statement_it != statement_publications.end()) {
      statement_publication = std::move(statement_it->second);
      statement_publications.erase(statement_it);
      publish_statement = true;
    }
  }
  const bool recover_detached_xa =
      is_xa_commit_publication_fallback(current_thd, param->thread_id);
  if (!publish && !recover_detached_xa && !publish_statement) {
    return;
  }
  std::string failure_stage;
  bool publication_ok = true;
  if (publish || recover_detached_xa) {
    publication_ok = vector_index_registry::publish_thd_txn(
        static_cast<uint64_t>(param->thread_id), recover_detached_xa,
        &failure_stage);
  }
  if (publication_ok && publish_statement) {
    for (const auto &intent : statement_publication.intents) {
      if (!vector_index_registry::publish_statement_publication_intent(
              intent, &failure_stage) ||
          !vector_index_registry::acknowledge_statement_publication_intent(
              intent, &failure_stage)) {
        publication_ok = false;
        vector_index_registry::schedule_publication_intent_recovery();
        break;
      }
    }
  }
  if (!publication_ok) {
    const std::string message =
        "Vector runtime publication failed after transaction commit at stage "
        "'" +
        (failure_stage.empty() ? "unknown" : failure_stage) + "'";
    LogErr(ERROR_LEVEL, ER_LOG_PRINTF_MSG, message.c_str());
  }
}

int vector_trx_observe_before_commit(Trans_param *param) {
  if (param == nullptr || (param->flags & TRANS_IS_REAL_TRANS) == 0) return 0;

  THD *thd = current_thd;
  if (thd == nullptr || thd->thread_id() != param->thread_id ||
      !has_vector_trx_context(thd)) {
    return 0;
  }

  return prepare_vector_publication(thd) ? 0 : 1;
}

void vector_trx_before_rollback(void *arg) {
  auto *param = static_cast<Trans_param *>(arg);
  if (param == nullptr || (param->flags & TRANS_IS_REAL_TRANS) == 0) return;

  {
    std::lock_guard<std::mutex> guard(publication_mutex);
    publication_threads.erase(param->thread_id);
    publication_guards.erase(param->thread_id);
    statement_publications.erase(param->thread_id);
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
    vector_trx_observe_before_commit,
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
  if (plugin == nullptr ||
      register_trans_observer(&vector_trx_observer, plugin)) {
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
    publication_guards.clear();
    statement_publications.clear();
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
  hton->flags = HTON_NOT_USER_SELECTABLE | HTON_HIDDEN | HTON_NO_GLOBAL_2PC;
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
    publication_guards.clear();
    statement_publications.clear();
  }
  vector_trx_hton = nullptr;
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  context_thd_for_testing = nullptr;
  context_roles_for_testing = kNoVectorContext;
  context_override_enabled_for_testing = false;
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
  return 0;
}

}  // namespace

namespace vector_trx_participant {

int init_plugin(void *p) { return vector_trx_init(p); }

int deinit_plugin(void *p) { return vector_trx_deinit(p); }

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
uint64_t thd_id_for_testing(const THD *thd) { return vector_thd_id(thd); }

uint64_t stmt_id_for_testing(const THD *thd) { return vector_stmt_id(thd); }

bool in_multi_stmt_for_testing(THD *thd) {
  return in_multi_stmt_vector_trx(thd);
}

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

void set_context_for_testing(THD *thd, bool active) {
  if (thd == nullptr) return;
  context_override_enabled_for_testing = true;
  context_thd_for_testing = active ? thd : nullptr;
  context_roles_for_testing =
      active ? kTransactionParticipant : kNoVectorContext;
}

bool has_context_for_testing(THD *thd) { return has_vector_trx_context(thd); }

bool has_explicit_txn_owner_for_testing(THD *thd) {
  return (vector_context_roles(thd) & kExplicitTxnOwner) != 0;
}

int prepare_for_testing(THD *thd, bool all) {
  return vector_trx_prepare(nullptr, thd, all);
}

int commit_for_testing(THD *thd, bool all) {
  return vector_trx_commit(nullptr, thd, all);
}

int rollback_for_testing(THD *thd, bool all) {
  return vector_trx_rollback(nullptr, thd, all);
}

int close_connection_for_testing(THD *thd) {
  return vector_trx_close_connection(nullptr, thd);
}

void after_commit_for_testing(Trans_param *param) {
  vector_trx_after_commit(param);
}

void before_rollback_for_testing(Trans_param *param) {
  vector_trx_before_rollback(param);
}
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

bool ensure_observer_registered() {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  if (registration_bypass_for_testing) return true;
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
  return ensure_observer_registered_impl();
}

bool in_user_multi_statement_transaction(THD *thd) {
  return thd != nullptr && !thd->slave_thread && !thd->is_binlog_applier() &&
         (thd->in_multi_stmt_transaction_mode() ||
          thd->in_active_multi_stmt_transaction());
}

bool register_participant(THD *thd) {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  if (registration_bypass_for_testing) {
    if (thd == nullptr) return false;
    set_vector_context_roles(
        thd, vector_context_roles(thd) | kTransactionParticipant);
    return true;
  }
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
  if (thd == nullptr || vector_trx_hton == nullptr ||
      !vector_trx_participant::ensure_observer_registered()) {
    return false;
  }

  set_vector_context_roles(thd,
                           vector_context_roles(thd) | kTransactionParticipant);

  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (truth_store->supports_attached_dml() &&
      !truth_store->ensure_attached_transaction(thd)) {
    return false;
  }

  trans_register_ha(thd, false, vector_trx_hton, nullptr);
  // Publication metadata uses the attached InnoDB transaction. Mark this
  // participant read-write so its prepare callback runs before InnoDB enters
  // the prepared state.
  thd->get_ha_data(vector_trx_hton->slot)->ha_info[0].set_trx_read_write();
  if (thd->in_multi_stmt_transaction_mode()) {
    trans_register_ha(thd, true, vector_trx_hton, nullptr);
    thd->get_ha_data(vector_trx_hton->slot)->ha_info[1].set_trx_read_write();
  }
  return true;
}

bool stage_statement_publication(
    THD *thd,
    std::vector<vector_index_truth_store::publication_intent> intents,
    bool catalog_exclusive, bool require_stable_catalog_set) {
  if (thd == nullptr || intents.empty() ||
      in_user_multi_statement_transaction(thd)) {
    return false;
  }

  std::unordered_set<std::string> seen_index_names;
  seen_index_names.reserve(intents.size());
  std::vector<std::string> index_names;
  index_names.reserve(intents.size());
  for (const auto &intent : intents) {
    if (intent.index_name.empty() || intent.publication_id == 0 ||
        !seen_index_names.insert(intent.index_name).second) {
      return false;
    }
    index_names.push_back(intent.index_name);
  }
  std::sort(index_names.begin(), index_names.end());

  std::vector<std::string> pending_dml_indexes;
  if (!vector_index_registry::pending_index_names_for_thd_txn(
          vector_thd_id(thd), &pending_dml_indexes) ||
      !pending_dml_indexes.empty()) {
    return false;
  }

  {
    std::lock_guard<std::mutex> guard(publication_mutex);
    if (statement_publications.find(thd->thread_id()) !=
        statement_publications.end()) {
      return false;
    }
  }

  if (!register_participant(thd)) return false;

  auto publication_guard =
      std::make_unique<vector_statement_publication::publication_guard>();
  const bool locked = catalog_exclusive
                          ? publication_guard->lock_catalog()
                          : publication_guard->lock_indexes(index_names);
  if (!locked) return false;

  if (require_stable_catalog_set) {
    if (!catalog_exclusive) return false;
    std::vector<std::string> current_index_names;
    if (!vector_index_registry::list_indexes_for_publication_catalog_guard(
            &current_index_names) ||
        !vector_statement_publication::index_sets_equal(current_index_names,
                                                        index_names)) {
      return false;
    }
  }

  vector_index_truth_store::truth_store *truth_store =
      vector_index_truth_store::get();
  if (!truth_store->supports_publication_intents()) return false;
  for (const auto &intent : intents) {
    if (!vector_index_registry::validate_statement_publication_intent(intent) ||
        !truth_store->insert_attached_publication_intent(thd, intent)) {
      return false;
    }
  }

  statement_publication_context context;
  context.statement_id = vector_stmt_id(thd);
  context.catalog_exclusive = catalog_exclusive;
  context.intents = std::move(intents);
  {
    std::lock_guard<std::mutex> guard(publication_mutex);
    if (!statement_publications.emplace(thd->thread_id(), std::move(context))
             .second ||
        !publication_guards
             .emplace(thd->thread_id(), std::move(publication_guard))
             .second) {
      statement_publications.erase(thd->thread_id());
      return false;
    }
  }
  return true;
}

bool register_explicit_txn_owner(THD *thd) {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  if (registration_bypass_for_testing) {
    if (thd == nullptr) return false;
    set_vector_context_roles(thd,
                             vector_context_roles(thd) | kExplicitTxnOwner);
    return true;
  }
#endif  // EXTRA_CODE_FOR_UNIT_TESTING
  if (thd == nullptr || vector_trx_hton == nullptr) return false;
  set_vector_context_roles(thd,
                           vector_context_roles(thd) | kExplicitTxnOwner);
  return true;
}

void release_explicit_txn_owner(THD *thd) {
  if (thd == nullptr) return;
  set_vector_context_roles(thd,
                           vector_context_roles(thd) & ~kExplicitTxnOwner);
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
