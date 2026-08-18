/*****************************************************************************

Copyright (c) 2026, Oracle and/or its affiliates.

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License, version 2.0, as published by the
Free Software Foundation.

This program is designed to work with certain software (including
but not limited to OpenSSL) that is licensed under separate terms,
as designated in a particular file or component or in included license
documentation.  The authors of MySQL hereby grant you an additional
permission to link the program and your derivative works with the
separately licensed software that they have either included with
the program or referenced in the documentation.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License, version 2.0,
for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA

*****************************************************************************/

#include "dict0vectruth.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "btr0cur.h"
#include "data0type.h"
#include "dict0dict.h"
#include "ha_prototypes.h"
#include "ha_innodb.h"
#include "lock0lock.h"
#include "my_dbug.h"
#include "pars0pars.h"
#include "que0que.h"
#include "rem0cmp.h"
#include "row0ins.h"
#include "row0mysql.h"
#include "row0row.h"
#include "row0sel.h"
#include "row0upd.h"
#include "row0vers.h"
#include "sql/handler.h"
#include "trx0rec.h"
#include "trx0roll.h"
#include "trx0sys.h"
#include "trx0trx.h"
#include "univ.i"

namespace innodb_vector_truth_store {

struct Session {
  trx_t *trx{nullptr};
  mem_heap_t *heap{nullptr};
  que_thr_t *thr{nullptr};
  bool owns_trx{true};
  bool finished{false};
};

}  // namespace innodb_vector_truth_store

namespace {

constexpr const char *kMetadataArtifactName = "metadata";
constexpr const char *kCommittedArtifactName = "committed";
constexpr const char *kManifestArtifactName = "manifest";
constexpr const char *kChangeLogArtifactName = "changelog";
constexpr const char *kPreparedArtifactName = "prepared";
constexpr const char *kPublicationIntentsArtifactName = "publication_intents";
constexpr const char *kSegmentTasksArtifactName = "segment_tasks";
constexpr const char *kQuarantineStoreArtifactName = "quarantine_store";

constexpr const char *kMetadataTableName = "mysql/vector_index_truth_metadata";
constexpr const char *kCommittedTableName =
    "mysql/vector_index_truth_committed";
constexpr const char *kManifestTableName = "mysql/vector_index_truth_manifest";
constexpr const char *kChangeLogTableName =
    "mysql/vector_index_truth_changelog";
constexpr const char *kPreparedTableName =
    "mysql/vector_index_truth_prepared";
constexpr const char *kPublicationIntentsTableName =
    "mysql/vector_index_publication_intents";
constexpr const char *kSegmentTasksTableName =
    "mysql/vector_index_truth_segment_tasks";
constexpr const char *kQuarantineTableName =
    "mysql/vector_index_truth_store_quarantine";

constexpr uint32_t kStructuredTruthStoreSingletonId = 1;
constexpr uint32_t kStructuredTruthStoreLayoutVersion = 1;
constexpr unsigned kSingletonIdColNo = 0;
constexpr unsigned kLayoutVersionColNo = 1;
constexpr unsigned kPayloadColNo = 2;
constexpr unsigned kUserCols = kPayloadColNo + 1;
constexpr unsigned kCols = kUserCols + DATA_N_SYS_COLS;
constexpr unsigned kSingletonIdFieldNo = kSingletonIdColNo;
constexpr uint8_t kTruthRowOpUpsert = 1;
constexpr uint8_t kTruthRowOpErase = 2;

/** Resolves a clustered record to the version visible outside active writers. */
class Latest_committed_record_reader {
 public:
  Latest_committed_record_reader(
      dict_index_t *index, mtr_t *mtr,
      const innodb_vector_truth_store::Session *session)
      : m_index(index), m_mtr(mtr), m_session(session) {}

  ~Latest_committed_record_reader() {
    if (m_offset_heap != nullptr) mem_heap_free(m_offset_heap);
    if (m_old_version_heap != nullptr) mem_heap_free(m_old_version_heap);
  }

  /**
    Return the current transaction's own version, the latest committed version,
    or nullptr when the record is an uncommitted insert by another transaction.
  */
  const rec_t *resolve(const rec_t *rec) {
    ut_ad(rec != nullptr);
    ut_ad(m_index != nullptr);
    ut_ad(m_mtr != nullptr);

    if (m_offset_heap != nullptr) mem_heap_empty(m_offset_heap);
    if (m_old_version_heap != nullptr) mem_heap_empty(m_old_version_heap);

    ulint stack_offsets[REC_OFFS_NORMAL_SIZE];
    rec_offs_init(stack_offsets);
    ulint *offsets = rec_get_offsets(rec, m_index, stack_offsets,
                                     ULINT_UNDEFINED, UT_LOCATION_HERE,
                                     &m_offset_heap);
    const trx_id_t rec_trx_id = row_get_rec_trx_id(rec, m_index, offsets);
    if (m_session != nullptr && m_session->trx != nullptr &&
        rec_trx_id == m_session->trx->id) {
      return rec;
    }
    if (!trx_rw_is_active(rec_trx_id, false)) return rec;

    if (m_old_version_heap == nullptr) {
      m_old_version_heap =
          mem_heap_create(rec_offs_size(offsets), UT_LOCATION_HERE);
    }
    const rec_t *old_version = nullptr;
    row_vers_build_for_semi_consistent_read(
        rec, m_mtr, m_index, &offsets, &m_offset_heap, m_old_version_heap,
        &old_version, nullptr);
    return old_version;
  }

 private:
  dict_index_t *m_index;
  mtr_t *m_mtr;
  const innodb_vector_truth_store::Session *m_session;
  mem_heap_t *m_offset_heap{nullptr};
  mem_heap_t *m_old_version_heap{nullptr};
};

dict_table_t *hidden_table_for_artifact(const char *artifact_name) {
  if (artifact_name == nullptr || dict_sys == nullptr) return nullptr;
  if (strcmp(artifact_name, kMetadataArtifactName) == 0) {
    if (dict_sys->vector_truth_metadata == nullptr) {
      dict_sys->vector_truth_metadata = dict_table_open_on_name(
          kMetadataTableName, false, false, DICT_ERR_IGNORE_NONE);
    }
    return dict_sys->vector_truth_metadata;
  }
  if (strcmp(artifact_name, kCommittedArtifactName) == 0) {
    if (dict_sys->vector_truth_committed == nullptr) {
      dict_sys->vector_truth_committed = dict_table_open_on_name(
          kCommittedTableName, false, false, DICT_ERR_IGNORE_NONE);
    }
    return dict_sys->vector_truth_committed;
  }
  if (strcmp(artifact_name, kManifestArtifactName) == 0) {
    if (dict_sys->vector_truth_manifest == nullptr) {
      dict_sys->vector_truth_manifest = dict_table_open_on_name(
          kManifestTableName, false, false, DICT_ERR_IGNORE_NONE);
    }
    return dict_sys->vector_truth_manifest;
  }
  if (strcmp(artifact_name, kChangeLogArtifactName) == 0) {
    if (dict_sys->vector_truth_changelog == nullptr) {
      dict_sys->vector_truth_changelog = dict_table_open_on_name(
          kChangeLogTableName, false, false, DICT_ERR_IGNORE_NONE);
    }
    return dict_sys->vector_truth_changelog;
  }
  if (strcmp(artifact_name, kPreparedArtifactName) == 0) {
    if (dict_sys->vector_truth_prepared == nullptr) {
      dict_sys->vector_truth_prepared = dict_table_open_on_name(
          kPreparedTableName, false, false, DICT_ERR_IGNORE_NONE);
    }
    return dict_sys->vector_truth_prepared;
  }
  if (strcmp(artifact_name, kPublicationIntentsArtifactName) == 0) {
    if (dict_sys->vector_publication_intents == nullptr) {
      dict_sys->vector_publication_intents = dict_table_open_on_name(
          kPublicationIntentsTableName, false, false, DICT_ERR_IGNORE_NONE);
    }
    return dict_sys->vector_publication_intents;
  }
  if (strcmp(artifact_name, kSegmentTasksArtifactName) == 0) {
    if (dict_sys->vector_truth_segment_tasks == nullptr) {
      dict_sys->vector_truth_segment_tasks = dict_table_open_on_name(
          kSegmentTasksTableName, false, false, DICT_ERR_IGNORE_NONE);
    }
    return dict_sys->vector_truth_segment_tasks;
  }
  if (strcmp(artifact_name, kQuarantineStoreArtifactName) == 0) {
    if (dict_sys->vector_truth_store_quarantine == nullptr) {
      dict_sys->vector_truth_store_quarantine = dict_table_open_on_name(
          kQuarantineTableName, false, false, DICT_ERR_IGNORE_NONE);
    }
    return dict_sys->vector_truth_store_quarantine;
  }
  return nullptr;
}

class Singleton_payload_table_buffer {
 public:
  bool get(dict_table_t *table, std::string *payload, bool *found,
           innodb_vector_truth_store::Session *session) {
    if (!init(table) || payload == nullptr || found == nullptr) return false;
    std::lock_guard<std::mutex> guard(m_mutex);

    init_tuple_with_id(m_search_tuple, kStructuredTruthStoreSingletonId);
    mtr_t mtr;
    btr_pcur_t pcur;
    ulint len = 0;
    const byte *field = nullptr;

    mtr.start();
    pcur.open(m_index, 0, m_search_tuple, PAGE_CUR_LE, BTR_SEARCH_LEAF, &mtr,
              UT_LOCATION_HERE);

    Latest_committed_record_reader committed_reader(m_index, &mtr, session);
    *found = current_matches_search_tuple(&pcur);
    const rec_t *visible_rec = nullptr;
    while (*found && pcur.is_on_user_rec()) {
      visible_rec = committed_reader.resolve(pcur.get_rec());
      if (visible_rec != nullptr &&
          !rec_get_deleted_flag(visible_rec, true)) {
        break;
      }
      *found = pcur.move_to_next_user_rec(&mtr) == DB_SUCCESS &&
               current_matches_search_tuple(&pcur);
    }

    payload->clear();
    if (*found && visible_rec != nullptr) {
      ulint offsets[REC_OFFS_NORMAL_SIZE];
      mem_heap_t *heap = nullptr;
      rec_offs_init(offsets);
      rec_offs_set_n_fields(offsets, m_index->n_fields);
      const rec_t *rec = visible_rec;
      rec_init_offsets_comp_ordinary(rec, false, m_index, offsets);
      if (!layout_version_matches(rec, offsets)) {
        pcur.close();
        mtr.commit();
        return false;
      }
      field =
          rec_get_nth_field(nullptr, rec, offsets, m_payload_field_no, &len);
      ut_ad(len != UNIV_SQL_NULL);

      if (rec_offs_nth_extern(m_index, offsets, m_payload_field_no)) {
        heap = mem_heap_create(256, UT_LOCATION_HERE);
        field = lob::btr_rec_copy_externally_stored_field(
            nullptr, m_index, rec, offsets, dict_table_page_size(m_index->table),
            m_payload_field_no, &len, nullptr, dict_index_is_sdi(m_index),
            heap);
        if (field == nullptr) {
          pcur.close();
          mtr.commit();
          mem_heap_free(heap);
          return false;
        }
      }

      payload->assign(reinterpret_cast<const char *>(field), len);
      if (heap != nullptr) {
        mem_heap_free(heap);
      }
    }

    pcur.close();
    mtr.commit();
    return true;
  }

  bool replace(dict_table_t *table, const std::string &payload,
               innodb_vector_truth_store::Session *session) {
    if (!init(table)) return false;
    if (session == nullptr || session->thr == nullptr || session->trx == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> guard(m_mutex);

    dtuple_t *entry;
    init_tuple_with_id(m_replace_tuple, kStructuredTruthStoreSingletonId);
    set_layout_version_field(m_replace_tuple);
    init_tuple_system_fields(session);
    dfield_t *dfield = dtuple_get_nth_field(m_replace_tuple, kPayloadColNo);
    dfield_set_data(dfield, payload.data(), payload.size());

    entry =
        row_build_index_entry(m_replace_tuple, nullptr, m_index, m_replace_heap);
    return transactional_replace_locked(entry, session);
  }

  bool remove_all(dict_table_t *table,
                  innodb_vector_truth_store::Session *session) {
    if (!init(table)) return false;
    if (session == nullptr || session->thr == nullptr || session->trx == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> guard(m_mutex);
    return transactional_remove_id_locked(kStructuredTruthStoreSingletonId,
                                          session);
  }

 private:
  upd_t *update_set_payload(const dtuple_t *entry, const rec_t *rec) {
    ulint offsets[REC_OFFS_NORMAL_SIZE];
    upd_field_t *upd_field;
    const dfield_t *layout_dfield;
    const dfield_t *payload_dfield;
    upd_t *update;

    rec_offs_init(offsets);
    rec_offs_set_n_fields(offsets, m_index->n_fields);
    rec_init_offsets_comp_ordinary(rec, false, m_index, offsets);
    ut_ad(!rec_get_deleted_flag(rec, true));

    layout_dfield = dtuple_get_nth_field(entry, m_layout_version_field_no);
    payload_dfield = dtuple_get_nth_field(entry, m_payload_field_no);

    update = upd_create(2, m_replace_heap);
    update->table = m_index->table;
    upd_field = upd_get_nth_field(update, 0);
    dfield_copy(&upd_field->new_val, layout_dfield);
    upd_field_set_field_no(upd_field, m_layout_version_field_no, m_index);

    upd_field = upd_get_nth_field(update, 1);
    dfield_copy(&upd_field->new_val, payload_dfield);
    upd_field_set_field_no(upd_field, m_payload_field_no, m_index);
    ut_d(update->validate());
    return update;
  }

  void init_tuple_system_fields(innodb_vector_truth_store::Session *session) {
    ut_ad(session != nullptr);
    ut_ad(session->trx != nullptr);
    const dict_col_t *col = m_index->table->get_sys_col(DATA_TRX_ID);
    dfield_t *dfield = dtuple_get_nth_field(m_replace_tuple, dict_col_get_no(col));
    mach_write_to_6(static_cast<byte *>(dfield->data), session->trx->id);
    dfield_set_data(dfield, dfield->data, DATA_TRX_ID_LEN);
  }

  bool transactional_replace_locked(
      dtuple_t *entry, innodb_vector_truth_store::Session *session) {
    btr_pcur_t pcur;
    mtr_t mtr;
    dberr_t error = DB_SUCCESS;

    init_tuple_with_id(m_search_tuple, kStructuredTruthStoreSingletonId);
    mtr.start();
    pcur.open(m_index, 0, m_search_tuple, PAGE_CUR_LE, BTR_MODIFY_TREE, &mtr,
              UT_LOCATION_HERE);

    bool found = current_matches_search_tuple(&pcur);
    while (found && pcur.is_on_user_rec() &&
           rec_get_deleted_flag(pcur.get_rec(), true)) {
      found = pcur.move_to_next_user_rec(&mtr) == DB_SUCCESS &&
              current_matches_search_tuple(&pcur);
    }

    if (!found || !pcur.is_on_user_rec()) {
      pcur.close();
      mtr.commit();

      static const ulint flags = BTR_NO_LOCKING_FLAG;
      error = row_ins_clust_index_entry_low(flags, BTR_MODIFY_LEAF, m_index,
                                            m_index->n_uniq, entry,
                                            session->thr, false);
      if (error == DB_FAIL) {
        error = row_ins_clust_index_entry_low(flags, BTR_MODIFY_TREE, m_index,
                                              m_index->n_uniq, entry,
                                              session->thr, false);
      }
      if (error != DB_SUCCESS) {
        ib::error() << "vector truth store insert failed for table "
                    << m_index->table->name << " with error " << error;
      }
      mem_heap_empty(m_dynamic_heap);
      mem_heap_empty(m_replace_heap);
      return error == DB_SUCCESS;
    }

    upd_t *update = update_set_payload(entry, pcur.get_rec());
    if (update != nullptr) {
      ulint *cur_offsets = nullptr;
      big_rec_t *big_rec = nullptr;
      static const ulint flags = BTR_NO_LOCKING_FLAG | BTR_KEEP_POS_FLAG;
      error = btr_cur_pessimistic_update(
          flags, pcur.get_btr_cur(), &cur_offsets, &m_dynamic_heap,
          m_replace_heap, &big_rec, update, 0, session->thr, session->trx->id,
          0, &mtr);

      if (error == DB_SUCCESS && big_rec != nullptr) {
        error = lob::btr_store_big_rec_extern_fields(
            session->trx, &pcur, update, cur_offsets, big_rec, &mtr,
            lob::OPCODE_UPDATE);
      }

      if (error != DB_SUCCESS) {
        ib::error() << "vector truth store update failed for table "
                    << m_index->table->name << " with error " << error;
      }

      if (big_rec != nullptr) {
        dtuple_big_rec_free(big_rec);
      }
    }

    pcur.close();
    mtr.commit();
    mem_heap_empty(m_dynamic_heap);
    mem_heap_empty(m_replace_heap);
    return error == DB_SUCCESS;
  }

  bool transactional_remove_id_locked(
      uint32_t row_id, innodb_vector_truth_store::Session *session) {
    btr_pcur_t pcur;
    mtr_t mtr;
    dberr_t error = DB_SUCCESS;

    init_tuple_with_id(m_search_tuple, row_id);
    mtr.start();
    pcur.open(m_index, 0, m_search_tuple, PAGE_CUR_LE,
              BTR_MODIFY_TREE | BTR_LATCH_FOR_DELETE, &mtr, UT_LOCATION_HERE);

    bool found = current_matches_search_tuple(&pcur);
    while (found && pcur.is_on_user_rec() &&
           rec_get_deleted_flag(pcur.get_rec(), true)) {
      found = pcur.move_to_next_user_rec(&mtr) == DB_SUCCESS &&
              current_matches_search_tuple(&pcur);
    }

    if (found && pcur.is_on_user_rec()) {
      ulint *offsets = rec_get_offsets(pcur.get_rec(), m_index, nullptr,
                                       ULINT_UNDEFINED, UT_LOCATION_HERE,
                                       &m_dynamic_heap);
      error = btr_cur_del_mark_set_clust_rec(
          BTR_NO_LOCKING_FLAG, btr_cur_get_block(pcur.get_btr_cur()),
          btr_cur_get_rec(pcur.get_btr_cur()), m_index, offsets, session->thr,
          m_search_tuple, &mtr);
    }

    pcur.close();
    mtr.commit();
    mem_heap_empty(m_dynamic_heap);
    return error == DB_SUCCESS;
  }

  bool init(dict_table_t *table) {
    if (table == nullptr) return false;
    std::lock_guard<std::mutex> guard(m_mutex);
    if (m_index != nullptr && m_index->table == table) return true;

    ut_ad(dict_table_is_comp(table));
    m_index = table->first_index();
    ut_ad(m_index != nullptr);
    ut_ad(m_index->next() == nullptr);
    ut_ad(m_index->n_uniq == 1);
    ut_ad(m_index->table->n_cols == kCols);
    m_layout_version_field_no = m_index->get_col_pos(kLayoutVersionColNo);
    ut_ad(m_layout_version_field_no != ULINT_UNDEFINED);
    m_payload_field_no = m_index->get_col_pos(kPayloadColNo);
    ut_ad(m_payload_field_no != ULINT_UNDEFINED);

    if (m_heap == nullptr) {
      m_heap = mem_heap_create(256, UT_LOCATION_HERE);
      m_dynamic_heap = mem_heap_create(128, UT_LOCATION_HERE);
      m_replace_heap = mem_heap_create(128, UT_LOCATION_HERE);
      create_tuples();
    }

    m_index->disable_ahi = true;
    m_index->cached = true;
    return true;
  }

  void create_tuples() {
    const dict_col_t *col;
    dfield_t *dfield;
    byte *row_id_buf;
    byte *trx_id_buf;
    byte *roll_ptr_buf;
    byte *id_buf;

    id_buf = static_cast<byte *>(mem_heap_alloc(m_heap, 4));
    memset(id_buf, 0, 4);

    m_search_tuple = dtuple_create(m_heap, 1);
    dict_index_copy_types(m_search_tuple, m_index, 1);
    dfield = dtuple_get_nth_field(m_search_tuple, 0);
    dfield_set_data(dfield, id_buf, 4);

    id_buf = static_cast<byte *>(mem_heap_alloc(m_heap, 4));
    memset(id_buf, 0, 4);

    m_replace_tuple = dtuple_create(m_heap, kCols);
    dict_table_copy_types(m_replace_tuple, m_index->table);
    dfield = dtuple_get_nth_field(m_replace_tuple, kSingletonIdColNo);
    dfield_set_data(dfield, id_buf, 4);
    set_layout_version_field(m_replace_tuple);

    row_id_buf = static_cast<byte *>(mem_heap_alloc(m_heap, DATA_ROW_ID_LEN));
    memset(row_id_buf, 0xFF, DATA_ROW_ID_LEN);

    col = m_index->table->get_sys_col(DATA_ROW_ID);
    dfield = dtuple_get_nth_field(m_replace_tuple, dict_col_get_no(col));
    dfield_set_data(dfield, row_id_buf, DATA_ROW_ID_LEN);

    trx_id_buf = static_cast<byte *>(mem_heap_alloc(m_heap, DATA_TRX_ID_LEN));
    memset(trx_id_buf, 0xFF, DATA_TRX_ID_LEN);
    col = m_index->table->get_sys_col(DATA_TRX_ID);
    dfield = dtuple_get_nth_field(m_replace_tuple, dict_col_get_no(col));
    dfield_set_data(dfield, trx_id_buf, DATA_TRX_ID_LEN);

    roll_ptr_buf =
        static_cast<byte *>(mem_heap_alloc(m_heap, DATA_ROLL_PTR_LEN));
    memset(roll_ptr_buf, 0xFF, DATA_ROLL_PTR_LEN);
    col = m_index->table->get_sys_col(DATA_ROLL_PTR);
    dfield = dtuple_get_nth_field(m_replace_tuple, dict_col_get_no(col));
    dfield_set_data(dfield, roll_ptr_buf, DATA_ROLL_PTR_LEN);
  }

  void init_tuple_with_id(dtuple_t *tuple, uint32_t row_id) {
    dfield_t *dfield = dtuple_get_nth_field(tuple, kSingletonIdFieldNo);
    void *data = dfield->data;
    mach_write_to_4(static_cast<byte *>(data), row_id);
    dfield_set_data(dfield, data, 4);
  }

  void set_layout_version_field(dtuple_t *tuple) {
    dfield_t *dfield = dtuple_get_nth_field(tuple, kLayoutVersionColNo);
    void *data = dfield->data;
    if (data == nullptr) data = mem_heap_alloc(m_heap, 4);
    mach_write_to_4(static_cast<byte *>(data),
                    kStructuredTruthStoreLayoutVersion);
    dfield_set_data(dfield, data, 4);
  }

  bool layout_version_matches(const rec_t *rec, const ulint *offsets) const {
    ulint len = 0;
    const byte *field = rec_get_nth_field(
        nullptr, rec, offsets, m_layout_version_field_no, &len);
    return len == 4 &&
           mach_read_from_4(field) == kStructuredTruthStoreLayoutVersion;
  }

  bool current_matches_search_tuple(btr_pcur_t *pcur) const {
    if (pcur == nullptr || !pcur->is_on_user_rec() ||
        page_rec_is_infimum(pcur->get_rec())) {
      return false;
    }
    ulint offsets[REC_OFFS_NORMAL_SIZE];
    rec_offs_init(offsets);
    rec_offs_set_n_fields(offsets, m_index->n_fields);
    rec_init_offsets_comp_ordinary(pcur->get_rec(), false, m_index, offsets);
    return cmp_dtuple_rec(m_search_tuple, pcur->get_rec(), m_index, offsets) ==
           0;
  }

  dict_index_t *m_index{nullptr};
  mem_heap_t *m_heap{nullptr};
  mem_heap_t *m_dynamic_heap{nullptr};
  mem_heap_t *m_replace_heap{nullptr};
  dtuple_t *m_search_tuple{nullptr};
  dtuple_t *m_replace_tuple{nullptr};
  ulint m_layout_version_field_no{ULINT_UNDEFINED};
  ulint m_payload_field_no{ULINT_UNDEFINED};
  std::mutex m_mutex;
};

class Row_truth_table_buffer {
 public:
  ~Row_truth_table_buffer() {
    if (m_replace_heap != nullptr) mem_heap_free(m_replace_heap);
    if (m_dynamic_heap != nullptr) mem_heap_free(m_dynamic_heap);
    if (m_heap != nullptr) mem_heap_free(m_heap);
  }

  bool get_committed(
      dict_table_t *table,
      std::vector<innodb_vector_truth_store::committed_row> *rows,
      bool *found, innodb_vector_truth_store::Session *session) {
    if (!init(table, 4, 2) || rows == nullptr || found == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> guard(m_mutex);
    rows->clear();
    if (!scan_rows_locked([&](const rec_t *rec) {
          innodb_vector_truth_store::committed_row row;
          if (!read_committed_row_fields(rec, &row)) return false;
          rows->push_back(std::move(row));
          return true;
        },
                          session)) {
      return false;
    }
    *found = !rows->empty();
    return true;
  }

  bool get_committed_for_index(
      dict_table_t *table, const std::string &index_name,
      std::vector<innodb_vector_truth_store::committed_row> *rows,
      bool *found, innodb_vector_truth_store::Session *session) {
    if (!init(table, 4, 2) || index_name.empty() || rows == nullptr ||
        found == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> guard(m_mutex);
    rows->clear();
    if (!scan_committed_index_rows_locked(
            index_name, [&](const rec_t *rec) {
              innodb_vector_truth_store::committed_row row;
              if (!read_committed_row_fields(rec, &row)) return false;
              rows->push_back(std::move(row));
              return true;
            },
            session)) {
      return false;
    }
    *found = !rows->empty();
    return true;
  }

  bool visit_committed_for_index(
      dict_table_t *table, const std::string &index_name,
      const std::function<bool(
          const innodb_vector_truth_store::committed_row &row)> &visitor,
      bool *found, innodb_vector_truth_store::Session *session) {
    if (!init(table, 4, 2) || index_name.empty() || !visitor ||
        found == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> guard(m_mutex);
    *found = false;
    return scan_committed_index_rows_locked(
        index_name, [&](const rec_t *rec) {
          innodb_vector_truth_store::committed_row row;
          if (!read_committed_row_fields(rec, &row)) return false;
          *found = true;
          return visitor(row);
        },
        session);
  }

  bool find_committed(dict_table_t *table, const std::string &index_name,
                      uint64_t doc_id,
                      innodb_vector_truth_store::committed_row *row,
                      bool *found,
                      innodb_vector_truth_store::Session *session) {
    if (!init(table, 4, 2) || index_name.empty() || row == nullptr ||
        found == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> guard(m_mutex);
    *found = false;
    set_search_field(0, index_name);
    set_search_uint64(1, doc_id);

    btr_pcur_t pcur;
    mtr_t mtr;
    mtr.start();
    pcur.open(m_index, 0, m_search_tuple, PAGE_CUR_LE, BTR_SEARCH_LEAF, &mtr,
              UT_LOCATION_HERE);
    Latest_committed_record_reader committed_reader(m_index, &mtr, session);
    bool matched = pcur.is_on_user_rec() &&
                   current_matches_search_key_locked(pcur.get_rec());
    const rec_t *visible_rec = nullptr;
    while (matched && pcur.is_on_user_rec()) {
      visible_rec = committed_reader.resolve(pcur.get_rec());
      if (visible_rec != nullptr &&
          !rec_get_deleted_flag(visible_rec, true)) {
        break;
      }
      matched = pcur.move_to_next_user_rec(&mtr) == DB_SUCCESS &&
                current_matches_search_key_locked(pcur.get_rec());
    }
    if (matched && visible_rec != nullptr) {
      *found = true;
      if (!read_committed_row_fields(visible_rec, row)) {
        pcur.close();
        mtr.commit();
        return false;
      }
    }
    pcur.close();
    mtr.commit();
    return true;
  }

  bool replace_committed(
      dict_table_t *table,
      const std::vector<innodb_vector_truth_store::committed_row> &rows,
      innodb_vector_truth_store::Session *session) {
    if (!init(table, 4, 2)) return false;
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!lock_table_for_write_locked(session)) return false;
    if (!remove_all_locked(session)) return false;
    for (const auto &row : rows) {
      set_replace_string(0, row.index_name);
      set_replace_uint64(1, row.doc_id);
      set_replace_uint32(2, row.dimension);
      set_replace_string(3, row.vector_payload);
      init_tuple_system_fields(session);
      if (!insert_locking_tuple_locked(session)) return false;
    }
    return true;
  }

  bool apply_committed_delta(
      dict_table_t *table,
      const std::vector<innodb_vector_truth_store::change_log_row> &rows,
      innodb_vector_truth_store::Session *session) {
    if (!init(table, 4, 2)) return false;
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!lock_table_for_write_locked(session)) return false;
    DEBUG_SYNC_C("vector_truth_store_committed_writer_ready");
    for (const auto &row : rows) {
      if (!remove_committed_key_locked(row.index_name, row.doc_id, session)) {
        return false;
      }
      if (row.op == kTruthRowOpErase) continue;
      if (row.op != kTruthRowOpUpsert) return false;

      set_replace_string(0, row.index_name);
      set_replace_uint64(1, row.doc_id);
      set_replace_uint32(2, row.dimension);
      set_replace_string(3, row.vector_payload);
      init_tuple_system_fields(session);
      if (!insert_locking_tuple_locked(session)) return false;
    }
    return true;
  }

  bool get_change_log(
      dict_table_t *table,
      std::vector<innodb_vector_truth_store::change_log_row> *rows,
      bool *found, innodb_vector_truth_store::Session *session) {
    if (!init(table, 10, 1) || rows == nullptr || found == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> guard(m_mutex);
    rows->clear();
    if (!scan_rows_locked([&](const rec_t *rec) {
          innodb_vector_truth_store::change_log_row row;
          if (!read_uint64_field(rec, 0, &row.sequence) ||
              !read_uint64_field(rec, 1, &row.txn_id) ||
              !read_uint64_field(rec, 2, &row.index_identity) ||
              !read_uint64_field(rec, 3, &row.publication_id) ||
              !read_uint64_field(rec, 4, &row.truth_generation) ||
              !read_uint8_field(rec, 5, &row.op) ||
              !read_string_field(rec, 6, &row.index_name) ||
              !read_uint64_field(rec, 7, &row.doc_id) ||
              !read_uint32_field(rec, 8, &row.dimension) ||
              !read_blob_field(rec, 9, &row.vector_payload)) {
            return false;
          }
          rows->push_back(std::move(row));
          return true;
        },
                          session)) {
      return false;
    }
    *found = !rows->empty();
    return true;
  }

  bool replace_change_log(
      dict_table_t *table,
      const std::vector<innodb_vector_truth_store::change_log_row> &rows,
      innodb_vector_truth_store::Session *session) {
    if (!init(table, 10, 1)) return false;
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!lock_table_for_write_locked(session)) return false;
    if (!remove_all_locked(session)) return false;
    for (const auto &row : rows) {
      set_replace_uint64(0, row.sequence);
      set_replace_uint64(1, row.txn_id);
      set_replace_uint64(2, row.index_identity);
      set_replace_uint64(3, row.publication_id);
      set_replace_uint64(4, row.truth_generation);
      set_replace_uint8(5, row.op);
      set_replace_string(6, row.index_name);
      set_replace_uint64(7, row.doc_id);
      set_replace_uint32(8, row.dimension);
      set_replace_string(9, row.vector_payload);
      init_tuple_system_fields(session);
      if (!insert_locking_tuple_locked(session)) return false;
    }
    return true;
  }

  bool append_change_log(
      dict_table_t *table,
      const std::vector<innodb_vector_truth_store::change_log_row> &rows,
      innodb_vector_truth_store::Session *session) {
    if (!init(table, 10, 1)) return false;
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!lock_table_for_write_locked(session)) return false;
    for (const auto &row : rows) {
      set_replace_uint64(0, row.sequence);
      set_replace_uint64(1, row.txn_id);
      set_replace_uint64(2, row.index_identity);
      set_replace_uint64(3, row.publication_id);
      set_replace_uint64(4, row.truth_generation);
      set_replace_uint8(5, row.op);
      set_replace_string(6, row.index_name);
      set_replace_uint64(7, row.doc_id);
      set_replace_uint32(8, row.dimension);
      set_replace_string(9, row.vector_payload);
      init_tuple_system_fields(session);
      if (!insert_locking_tuple_locked(session)) return false;
    }
    return true;
  }

  bool erase_change_log_sequences(
      dict_table_t *table, const std::vector<uint64_t> &sequences,
      innodb_vector_truth_store::Session *session) {
    if (!init(table, 10, 1)) return false;
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!lock_table_for_write_locked(session)) return false;
    for (const uint64_t sequence : sequences) {
      if (sequence == 0) return false;
      set_search_uint64(0, sequence);
      if (!remove_current_search_key_locked(session)) return false;
    }
    return true;
  }

  bool get_prepared(
      dict_table_t *table,
      std::vector<innodb_vector_truth_store::prepared_change_row> *rows,
      bool *found, innodb_vector_truth_store::Session *session) {
    if (!init(table, 12, 2) || rows == nullptr || found == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> guard(m_mutex);
    rows->clear();
    if (!scan_rows_locked([&](const rec_t *rec) {
          innodb_vector_truth_store::prepared_change_row row;
          if (!read_uint64_field(rec, 0, &row.txn_id) ||
              !read_uint64_field(rec, 1, &row.row_no) ||
              !read_uint64_field(rec, 2, &row.format_id) ||
              !read_uint64_field(rec, 3, &row.gtrid_length) ||
              !read_uint64_field(rec, 4, &row.bqual_length) ||
              !read_string_field(rec, 5, &row.xid_data) ||
              !read_uint8_field(rec, 6, &row.prepared_in_tc) ||
              !read_uint8_field(rec, 7, &row.op) ||
              !read_string_field(rec, 8, &row.index_name) ||
              !read_uint64_field(rec, 9, &row.doc_id) ||
              !read_uint32_field(rec, 10, &row.dimension) ||
              !read_blob_field(rec, 11, &row.vector_payload)) {
            return false;
          }
          rows->push_back(std::move(row));
          return true;
        },
                          session)) {
      return false;
    }
    *found = !rows->empty();
    return true;
  }

  bool replace_prepared(
      dict_table_t *table,
      const std::vector<innodb_vector_truth_store::prepared_change_row> &rows,
      innodb_vector_truth_store::Session *session) {
    if (!init(table, 12, 2)) return false;
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!lock_table_for_write_locked(session)) return false;
    if (!remove_all_locked(session)) return false;
    for (const auto &row : rows) {
      set_replace_uint64(0, row.txn_id);
      set_replace_uint64(1, row.row_no);
      set_replace_uint64(2, row.format_id);
      set_replace_uint64(3, row.gtrid_length);
      set_replace_uint64(4, row.bqual_length);
      set_replace_string(5, row.xid_data);
      set_replace_uint8(6, row.prepared_in_tc);
      set_replace_uint8(7, row.op);
      set_replace_string(8, row.index_name);
      set_replace_uint64(9, row.doc_id);
      set_replace_uint32(10, row.dimension);
      set_replace_string(11, row.vector_payload);
      init_tuple_system_fields(session);
      if (!insert_locking_tuple_locked(session)) return false;
    }
    return true;
  }

  bool get_publication_intents(
      dict_table_t *table,
      std::vector<innodb_vector_truth_store::publication_intent_row> *rows,
      bool *found, innodb_vector_truth_store::Session *session) {
    if (!init(table, 22, 1) || rows == nullptr || found == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> guard(m_mutex);
    rows->clear();
    if (!scan_rows_locked(
            [&](const rec_t *rec) {
              innodb_vector_truth_store::publication_intent_row row;
              if (!read_string_field(rec, 0, &row.index_name) ||
                  !read_uint64_field(rec, 1, &row.publication_id) ||
                  !read_uint64_field(rec, 2, &row.txn_id) ||
                  !read_uint8_field(rec, 3, &row.operation) ||
                  !read_uint8_field(rec, 4, &row.expected_exists) ||
                  !read_uint64_field(rec, 5,
                                     &row.expected_index_identity) ||
                  !read_uint64_field(rec, 6,
                                     &row.expected_truth_generation) ||
                  !read_uint64_field(rec, 7,
                                     &row.expected_config_generation) ||
                  !read_uint64_field(rec, 8,
                                     &row.expected_artifact_generation) ||
                  !read_uint64_field(rec, 9,
                                     &row.expected_runtime_generation) ||
                  !read_uint64_field(rec, 10,
                                     &row.expected_lifecycle_version) ||
                  !read_uint64_field(rec, 11,
                                     &row.expected_source_generation) ||
                  !read_uint8_field(rec, 12, &row.target_exists) ||
                  !read_uint64_field(rec, 13, &row.target_index_identity) ||
                  !read_uint64_field(rec, 14,
                                     &row.target_truth_generation) ||
                  !read_uint64_field(rec, 15,
                                     &row.target_config_generation) ||
                  !read_uint64_field(rec, 16,
                                     &row.target_artifact_generation) ||
                  !read_uint64_field(rec, 17,
                                     &row.target_runtime_generation) ||
                  !read_uint64_field(rec, 18,
                                     &row.target_lifecycle_version) ||
                  !read_uint64_field(rec, 19,
                                     &row.target_source_generation) ||
                  !read_blob_field(rec, 20, &row.payload) ||
                  !read_uint8_field(rec, 21, &row.state)) {
                return false;
              }
              rows->push_back(std::move(row));
              return true;
            },
            session)) {
      return false;
    }
    *found = !rows->empty();
    return true;
  }

  bool insert_publication_intent(
      dict_table_t *table,
      const innodb_vector_truth_store::publication_intent_row &row,
      innodb_vector_truth_store::Session *session) {
    if (!init(table, 22, 1) || row.index_name.empty() ||
        row.publication_id == 0 || session == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> guard(m_mutex);
    set_replace_string(0, row.index_name);
    set_replace_uint64(1, row.publication_id);
    set_replace_uint64(2, row.txn_id);
    set_replace_uint8(3, row.operation);
    set_replace_uint8(4, row.expected_exists);
    set_replace_uint64(5, row.expected_index_identity);
    set_replace_uint64(6, row.expected_truth_generation);
    set_replace_uint64(7, row.expected_config_generation);
    set_replace_uint64(8, row.expected_artifact_generation);
    set_replace_uint64(9, row.expected_runtime_generation);
    set_replace_uint64(10, row.expected_lifecycle_version);
    set_replace_uint64(11, row.expected_source_generation);
    set_replace_uint8(12, row.target_exists);
    set_replace_uint64(13, row.target_index_identity);
    set_replace_uint64(14, row.target_truth_generation);
    set_replace_uint64(15, row.target_config_generation);
    set_replace_uint64(16, row.target_artifact_generation);
    set_replace_uint64(17, row.target_runtime_generation);
    set_replace_uint64(18, row.target_lifecycle_version);
    set_replace_uint64(19, row.target_source_generation);
    set_replace_string(20, row.payload);
    set_replace_uint8(21, row.state);
    init_tuple_system_fields(session);
    return insert_locking_tuple_locked(session);
  }

  bool delete_publication_intent(
      dict_table_t *table, const std::string &index_name,
      uint64_t publication_id,
      innodb_vector_truth_store::Session *session) {
    if (!init(table, 22, 1) || index_name.empty() || publication_id == 0 ||
        session == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> guard(m_mutex);
    set_search_field(0, index_name);
    return delete_publication_intent_locked(publication_id, session);
  }

 private:
  bool init(dict_table_t *table, ulint user_cols, ulint unique_cols) {
    if (table == nullptr) return false;
    if (m_index != nullptr && m_index->table == table) return true;

    ut_ad(dict_table_is_comp(table));
    m_index = table->first_index();
    ut_ad(m_index != nullptr);
    ut_ad(m_index->next() == nullptr);
    ut_ad(m_index->n_uniq == unique_cols);
    ut_ad(m_index->table->n_cols == user_cols + DATA_N_SYS_COLS);
    m_user_cols = user_cols;
    m_unique_cols = unique_cols;

    m_field_nos.assign(user_cols, ULINT_UNDEFINED);
    for (ulint col_no = 0; col_no < user_cols; ++col_no) {
      m_field_nos[col_no] = m_index->get_col_pos(col_no);
      ut_ad(m_field_nos[col_no] != ULINT_UNDEFINED);
    }

    if (m_heap == nullptr) {
      m_heap = mem_heap_create(512, UT_LOCATION_HERE);
      m_dynamic_heap = mem_heap_create(256, UT_LOCATION_HERE);
      m_replace_heap = mem_heap_create(256, UT_LOCATION_HERE);
      create_tuples();
    }

    m_index->disable_ahi = true;
    m_index->cached = true;
    return true;
  }

  void create_tuples() {
    m_search_tuple = dtuple_create(m_heap, m_unique_cols);
    dict_index_copy_types(m_search_tuple, m_index, m_unique_cols);
    m_replace_tuple = dtuple_create(m_heap, m_user_cols + DATA_N_SYS_COLS);
    dict_table_copy_types(m_replace_tuple, m_index->table);

    m_user_buffers.reserve(m_user_cols);
    for (ulint i = 0; i < m_user_cols; ++i) {
      m_user_buffers.push_back(static_cast<byte *>(
          mem_heap_alloc(m_heap, sizeof(uint64_t))));
    }
    m_search_buffers.reserve(m_unique_cols);
    for (ulint i = 0; i < m_unique_cols; ++i) {
      m_search_buffers.push_back(static_cast<byte *>(
          mem_heap_alloc(m_heap, sizeof(uint64_t))));
    }

    byte *row_id_buf = static_cast<byte *>(
        mem_heap_alloc(m_heap, DATA_ROW_ID_LEN));
    memset(row_id_buf, 0xFF, DATA_ROW_ID_LEN);
    const dict_col_t *col = m_index->table->get_sys_col(DATA_ROW_ID);
    dfield_t *dfield =
        dtuple_get_nth_field(m_replace_tuple, dict_col_get_no(col));
    dfield_set_data(dfield, row_id_buf, DATA_ROW_ID_LEN);

    byte *trx_id_buf = static_cast<byte *>(
        mem_heap_alloc(m_heap, DATA_TRX_ID_LEN));
    memset(trx_id_buf, 0xFF, DATA_TRX_ID_LEN);
    col = m_index->table->get_sys_col(DATA_TRX_ID);
    dfield = dtuple_get_nth_field(m_replace_tuple, dict_col_get_no(col));
    dfield_set_data(dfield, trx_id_buf, DATA_TRX_ID_LEN);

    byte *roll_ptr_buf = static_cast<byte *>(
        mem_heap_alloc(m_heap, DATA_ROLL_PTR_LEN));
    memset(roll_ptr_buf, 0xFF, DATA_ROLL_PTR_LEN);
    col = m_index->table->get_sys_col(DATA_ROLL_PTR);
    dfield = dtuple_get_nth_field(m_replace_tuple, dict_col_get_no(col));
    dfield_set_data(dfield, roll_ptr_buf, DATA_ROLL_PTR_LEN);
  }

  void set_replace_string(ulint col_no, const std::string &value) {
    dfield_t *dfield = dtuple_get_nth_field(m_replace_tuple, col_no);
    dfield_set_data(dfield, value.data(), value.size());
  }

  void set_replace_uint8(ulint col_no, uint8_t value) {
    byte *data = m_user_buffers[col_no];
    mach_write_to_1(data, value);
    dfield_t *dfield = dtuple_get_nth_field(m_replace_tuple, col_no);
    dfield_set_data(dfield, data, 1);
  }

  void set_replace_uint32(ulint col_no, uint32_t value) {
    byte *data = m_user_buffers[col_no];
    mach_write_to_4(data, value);
    dfield_t *dfield = dtuple_get_nth_field(m_replace_tuple, col_no);
    dfield_set_data(dfield, data, 4);
  }

  void set_replace_uint64(ulint col_no, uint64_t value) {
    byte *data = m_user_buffers[col_no];
    mach_write_to_8(data, value);
    dfield_t *dfield = dtuple_get_nth_field(m_replace_tuple, col_no);
    dfield_set_data(dfield, data, 8);
  }

  void init_tuple_system_fields(innodb_vector_truth_store::Session *session) {
    ut_ad(session != nullptr);
    ut_ad(session->trx != nullptr);
    const dict_col_t *col = m_index->table->get_sys_col(DATA_TRX_ID);
    dfield_t *dfield =
        dtuple_get_nth_field(m_replace_tuple, dict_col_get_no(col));
    mach_write_to_6(static_cast<byte *>(dfield->data), session->trx->id);
    dfield_set_data(dfield, dfield->data, DATA_TRX_ID_LEN);
  }

  dtuple_t *build_replace_entry() {
    return row_build_index_entry(m_replace_tuple, nullptr, m_index,
                                 m_replace_heap);
  }

  bool lock_table_for_write_locked(
      innodb_vector_truth_store::Session *session) {
    ut_ad(session != nullptr);
    ut_ad(session->thr != nullptr);
    ut_ad(session->trx != nullptr);

    que_thr_stop_for_mysql_no_error(session->thr, session->trx);
    const dberr_t error =
        lock_table_for_trx(m_index->table, session->trx, LOCK_IX);
    que_thr_move_to_run_state_for_mysql(session->thr, session->trx);
    if (error != DB_SUCCESS) {
      ib::error() << "vector truth store table lock failed for table "
                  << m_index->table->name << " with error " << error;
    }
    return error == DB_SUCCESS;
  }

  bool insert_locking_tuple_locked(
      innodb_vector_truth_store::Session *session) {
    ut_ad(session != nullptr);
    ut_ad(session->thr != nullptr);
    ut_ad(session->trx != nullptr);
    if (!lock_table_for_write_locked(session)) return false;

    dtuple_t *entry = build_replace_entry();
    trx_savept_t savepoint = trx_savept_take(session->trx);
    dberr_t error = DB_SUCCESS;
    bool retry = false;
    do {
      error = row_ins_clust_index_entry(m_index, entry, session->thr, false);
      if (error == DB_SUCCESS) break;

      if (error == DB_LOCK_WAIT) {
        DEBUG_SYNC_C("vector_truth_store_row_lock_wait");
      }
      session->trx->error_state = error;
      que_thr_stop_for_mysql(session->thr);
      session->thr->lock_state = QUE_THR_LOCK_ROW;
      retry = row_mysql_handle_errors(&error, session->trx, session->thr,
                                      &savepoint);
      session->thr->lock_state = QUE_THR_LOCK_NOLOCK;
    } while (retry);

    if (!session->thr->is_active) {
      que_thr_move_to_run_state_for_mysql(session->thr, session->trx);
    }
    if (error != DB_SUCCESS) {
      ib::error() << "vector truth store row insert failed for table "
                  << m_index->table->name << " with error " << error;
    }
    mem_heap_empty(m_dynamic_heap);
    mem_heap_empty(m_replace_heap);
    return error == DB_SUCCESS;
  }

  template <typename Fn>
  bool scan_rows_locked(Fn &&read_row,
                        innodb_vector_truth_store::Session *session) {
    btr_pcur_t pcur;
    mtr_t mtr;
    dberr_t error = DB_SUCCESS;

    mtr.start();
    pcur.open_at_side(true, m_index, BTR_SEARCH_LEAF, true, 0, &mtr);
    Latest_committed_record_reader committed_reader(m_index, &mtr, session);
    error = pcur.move_to_next_user_rec(&mtr);
    while (error == DB_SUCCESS && pcur.is_on_user_rec()) {
      const rec_t *visible_rec = committed_reader.resolve(pcur.get_rec());
      if (visible_rec != nullptr &&
          !rec_get_deleted_flag(visible_rec, true) && !read_row(visible_rec)) {
        pcur.close();
        mtr.commit();
        return false;
      }
      error = pcur.move_to_next_user_rec(&mtr);
    }

    pcur.close();
    mtr.commit();
    return error == DB_SUCCESS || error == DB_END_OF_INDEX;
  }

  template <typename Fn>
  bool scan_committed_index_rows_locked(const std::string &index_name,
                                        Fn &&read_row,
                                        innodb_vector_truth_store::Session *session) {
    set_search_field(0, index_name);
    set_search_uint64(1, 0);

    btr_pcur_t pcur;
    mtr_t mtr;
    dberr_t error = DB_SUCCESS;

    mtr.start();
    pcur.open(m_index, 0, m_search_tuple, PAGE_CUR_GE, BTR_SEARCH_LEAF, &mtr,
              UT_LOCATION_HERE);
    Latest_committed_record_reader committed_reader(m_index, &mtr, session);
    while (error == DB_SUCCESS && pcur.is_on_user_rec()) {
      std::string current_index_name;
      if (!read_string_field(pcur.get_rec(), 0, &current_index_name)) {
        pcur.close();
        mtr.commit();
        return false;
      }
      if (current_index_name != index_name) break;

      const rec_t *visible_rec = committed_reader.resolve(pcur.get_rec());
      if (visible_rec != nullptr &&
          !rec_get_deleted_flag(visible_rec, true) && !read_row(visible_rec)) {
        pcur.close();
        mtr.commit();
        return false;
      }
      error = pcur.move_to_next_user_rec(&mtr);
    }

    pcur.close();
    mtr.commit();
    return error == DB_SUCCESS || error == DB_END_OF_INDEX;
  }

  bool read_committed_row_fields(
      const rec_t *rec, innodb_vector_truth_store::committed_row *row) {
    return row != nullptr && read_string_field(rec, 0, &row->index_name) &&
           read_uint64_field(rec, 1, &row->doc_id) &&
           read_uint32_field(rec, 2, &row->dimension) &&
           read_blob_field(rec, 3, &row->vector_payload);
  }

  bool read_field(const rec_t *rec, ulint col_no, const byte **field,
                  ulint *len) {
    if (field == nullptr || len == nullptr || col_no >= m_field_nos.size()) {
      return false;
    }
    ulint offsets[REC_OFFS_NORMAL_SIZE];
    rec_offs_init(offsets);
    rec_offs_set_n_fields(offsets, m_index->n_fields);
    rec_init_offsets_comp_ordinary(rec, false, m_index, offsets);
    *field = rec_get_nth_field(nullptr, rec, offsets, m_field_nos[col_no], len);
    return *len != UNIV_SQL_NULL;
  }

  bool read_blob_field(const rec_t *rec, ulint col_no, std::string *value) {
    if (value == nullptr || col_no >= m_field_nos.size()) return false;
    ulint offsets[REC_OFFS_NORMAL_SIZE];
    mem_heap_t *heap = nullptr;
    ulint len = 0;
    const byte *field = nullptr;

    rec_offs_init(offsets);
    rec_offs_set_n_fields(offsets, m_index->n_fields);
    rec_init_offsets_comp_ordinary(rec, false, m_index, offsets);
    field = rec_get_nth_field(nullptr, rec, offsets, m_field_nos[col_no], &len);
    if (len == UNIV_SQL_NULL) return false;

    if (rec_offs_nth_extern(m_index, offsets, m_field_nos[col_no])) {
      heap = mem_heap_create(256, UT_LOCATION_HERE);
      field = lob::btr_rec_copy_externally_stored_field(
          nullptr, m_index, rec, offsets, dict_table_page_size(m_index->table),
          m_field_nos[col_no], &len, nullptr, dict_index_is_sdi(m_index), heap);
      if (field == nullptr) {
        mem_heap_free(heap);
        return false;
      }
    }

    value->assign(reinterpret_cast<const char *>(field), len);
    if (heap != nullptr) mem_heap_free(heap);
    return true;
  }

  bool read_string_field(const rec_t *rec, ulint col_no, std::string *value) {
    const byte *field = nullptr;
    ulint len = 0;
    if (value == nullptr || !read_field(rec, col_no, &field, &len)) {
      return false;
    }
    value->assign(reinterpret_cast<const char *>(field), len);
    return true;
  }

  bool read_uint8_field(const rec_t *rec, ulint col_no, uint8_t *value) {
    const byte *field = nullptr;
    ulint len = 0;
    if (value == nullptr || !read_field(rec, col_no, &field, &len) || len != 1) {
      return false;
    }
    *value = mach_read_from_1(field);
    return true;
  }

  bool read_uint32_field(const rec_t *rec, ulint col_no, uint32_t *value) {
    const byte *field = nullptr;
    ulint len = 0;
    if (value == nullptr || !read_field(rec, col_no, &field, &len) || len != 4) {
      return false;
    }
    *value = mach_read_from_4(field);
    return true;
  }

  bool read_uint64_field(const rec_t *rec, ulint col_no, uint64_t *value) {
    const byte *field = nullptr;
    ulint len = 0;
    if (value == nullptr || !read_field(rec, col_no, &field, &len) || len != 8) {
      return false;
    }
    *value = mach_read_from_8(field);
    return true;
  }

  bool load_keys_locked(std::vector<std::vector<std::string>> *keys,
                        innodb_vector_truth_store::Session *session) {
    if (keys == nullptr) return false;
    keys->clear();
    return scan_rows_locked([&](const rec_t *rec) {
      std::vector<std::string> key;
      key.reserve(m_unique_cols);
      for (ulint field_no = 0; field_no < m_unique_cols; ++field_no) {
        const byte *field = nullptr;
        ulint len = 0;
        if (!read_field(rec, field_no, &field, &len)) return false;
        key.emplace_back(reinterpret_cast<const char *>(field), len);
      }
      keys->push_back(std::move(key));
      return true;
    },
                            session);
  }

  void set_search_field(ulint field_no, const std::string &value) {
    dfield_t *dfield = dtuple_get_nth_field(m_search_tuple, field_no);
    dfield_set_data(dfield, value.data(), value.size());
  }

  void set_search_uint64(ulint field_no, uint64_t value) {
    byte *data = m_search_buffers[field_no];
    mach_write_to_8(data, value);
    dfield_t *dfield = dtuple_get_nth_field(m_search_tuple, field_no);
    dfield_set_data(dfield, data, 8);
  }

  bool remove_committed_key_locked(
      const std::string &index_name, uint64_t doc_id,
      innodb_vector_truth_store::Session *session) {
    set_search_field(0, index_name);
    set_search_uint64(1, doc_id);
    return remove_current_search_key_locked(session);
  }

  bool delete_publication_intent_locked(
      uint64_t publication_id,
      innodb_vector_truth_store::Session *session) {
    ut_ad(session != nullptr);
    ut_ad(session->thr != nullptr);
    ut_ad(session->trx != nullptr);
    if (!lock_table_for_write_locked(session)) return false;

    trx_savept_t savepoint = trx_savept_take(session->trx);
    for (;;) {
      btr_pcur_t pcur;
      mtr_t mtr;
      dberr_t error = DB_SUCCESS;
      mtr.start();
      pcur.open(m_index, 0, m_search_tuple, PAGE_CUR_LE,
                BTR_MODIFY_TREE | BTR_LATCH_FOR_DELETE, &mtr,
                UT_LOCATION_HERE);

      const bool found =
          pcur.is_on_user_rec() &&
          !page_rec_is_infimum(pcur.get_rec()) &&
          current_matches_search_key_locked(pcur.get_rec());
      if (!found) {
        pcur.close();
        mtr.commit();
        return true;
      }

      ulint *offsets =
          rec_get_offsets(pcur.get_rec(), m_index, nullptr, ULINT_UNDEFINED,
                          UT_LOCATION_HERE, &m_dynamic_heap);
      error = lock_clust_rec_modify_check_and_lock(
          0, pcur.get_block(), pcur.get_rec(), m_index, offsets,
          session->thr);
      bool publication_id_matches = true;
      if (error == DB_SUCCESS &&
          !rec_get_deleted_flag(pcur.get_rec(), true)) {
        uint64_t stored_publication_id = 0;
        publication_id_matches =
            read_uint64_field(pcur.get_rec(), 1, &stored_publication_id) &&
            stored_publication_id == publication_id;
      }
      if (error == DB_SUCCESS && publication_id_matches &&
          !rec_get_deleted_flag(pcur.get_rec(), true)) {
        error = btr_cur_del_mark_set_clust_rec(
            0, btr_cur_get_block(pcur.get_btr_cur()),
            btr_cur_get_rec(pcur.get_btr_cur()), m_index, offsets,
            session->thr, m_search_tuple, &mtr);
      }

      pcur.close();
      mtr.commit();
      mem_heap_empty(m_dynamic_heap);
      if (!publication_id_matches) return false;
      if (error == DB_SUCCESS) return true;

      session->trx->error_state = error;
      que_thr_stop_for_mysql(session->thr);
      session->thr->lock_state = QUE_THR_LOCK_ROW;
      const bool retry = row_mysql_handle_errors(
          &error, session->trx, session->thr, &savepoint);
      session->thr->lock_state = QUE_THR_LOCK_NOLOCK;
      if (retry) continue;

      if (!session->thr->is_active) {
        que_thr_move_to_run_state_for_mysql(session->thr, session->trx);
      }
      ib::error() << "vector publication intent delete failed for table "
                  << m_index->table->name << " with error " << error;
      return false;
    }
  }

  bool remove_key_locked(const std::vector<std::string> &key,
                         innodb_vector_truth_store::Session *session) {
    if (key.size() != m_unique_cols) return false;
    for (ulint field_no = 0; field_no < m_unique_cols; ++field_no) {
      set_search_field(field_no, key[field_no]);
    }
    return remove_current_search_key_locked(session);
  }

  bool remove_current_search_key_locked(
      innodb_vector_truth_store::Session *session) {
    ut_ad(session != nullptr);
    ut_ad(session->thr != nullptr);
    ut_ad(session->trx != nullptr);
    trx_savept_t savepoint = trx_savept_take(session->trx);
    for (;;) {
      btr_pcur_t pcur;
      mtr_t mtr;
      dberr_t error = DB_SUCCESS;
      mtr.start();
      pcur.open(m_index, 0, m_search_tuple, PAGE_CUR_LE,
                BTR_MODIFY_TREE | BTR_LATCH_FOR_DELETE, &mtr,
                UT_LOCATION_HERE);

      const bool found =
          pcur.is_on_user_rec() && !page_rec_is_infimum(pcur.get_rec()) &&
          current_matches_search_key_locked(pcur.get_rec());
      if (!found) {
        pcur.close();
        mtr.commit();
        return true;
      }

      ulint *offsets =
          rec_get_offsets(pcur.get_rec(), m_index, nullptr, ULINT_UNDEFINED,
                          UT_LOCATION_HERE, &m_dynamic_heap);
      error = lock_clust_rec_modify_check_and_lock(
          0, pcur.get_block(), pcur.get_rec(), m_index, offsets,
          session->thr);
      if (error == DB_SUCCESS &&
          !rec_get_deleted_flag(pcur.get_rec(), true)) {
        error = btr_cur_del_mark_set_clust_rec(
            0, btr_cur_get_block(pcur.get_btr_cur()),
            btr_cur_get_rec(pcur.get_btr_cur()), m_index, offsets,
            session->thr, m_search_tuple, &mtr);
      }

      pcur.close();
      mtr.commit();
      mem_heap_empty(m_dynamic_heap);
      if (error == DB_SUCCESS) return true;

      if (error == DB_LOCK_WAIT) {
        DEBUG_SYNC_C("vector_truth_store_row_lock_wait");
      }
      session->trx->error_state = error;
      que_thr_stop_for_mysql(session->thr);
      session->thr->lock_state = QUE_THR_LOCK_ROW;
      const bool retry = row_mysql_handle_errors(
          &error, session->trx, session->thr, &savepoint);
      session->thr->lock_state = QUE_THR_LOCK_NOLOCK;
      if (retry) continue;

      if (!session->thr->is_active) {
        que_thr_move_to_run_state_for_mysql(session->thr, session->trx);
      }
      ib::error() << "vector truth store row delete failed for table "
                  << m_index->table->name << " with error " << error;
      return false;
    }
  }

  bool current_matches_search_key_locked(const rec_t *rec) {
    if (rec == nullptr || page_rec_is_infimum(rec) || page_rec_is_supremum(rec))
      return false;
    ulint *offsets = rec_get_offsets(rec, m_index, nullptr, ULINT_UNDEFINED,
                                     UT_LOCATION_HERE, &m_dynamic_heap);
    const bool matches =
        cmp_dtuple_rec(m_search_tuple, rec, m_index, offsets) == 0;
    mem_heap_empty(m_dynamic_heap);
    return matches;
  }

  bool remove_all_locked(innodb_vector_truth_store::Session *session) {
    std::vector<std::vector<std::string>> keys;
    if (!load_keys_locked(&keys, session)) return false;
    for (const auto &key : keys) {
      if (!remove_key_locked(key, session)) return false;
    }
    return true;
  }

  dict_index_t *m_index{nullptr};
  mem_heap_t *m_heap{nullptr};
  mem_heap_t *m_dynamic_heap{nullptr};
  mem_heap_t *m_replace_heap{nullptr};
  dtuple_t *m_search_tuple{nullptr};
  dtuple_t *m_replace_tuple{nullptr};
  ulint m_user_cols{0};
  ulint m_unique_cols{0};
  std::vector<ulint> m_field_nos;
  std::vector<byte *> m_user_buffers;
  std::vector<byte *> m_search_buffers;
  std::mutex m_mutex;
};

Singleton_payload_table_buffer g_metadata_buffer;
Singleton_payload_table_buffer g_manifest_buffer;
Singleton_payload_table_buffer g_segment_tasks_buffer;
Singleton_payload_table_buffer g_quarantine_buffer;

Row_truth_table_buffer g_committed_row_buffer;
Row_truth_table_buffer g_changelog_row_buffer;
Row_truth_table_buffer g_prepared_row_buffer;

Singleton_payload_table_buffer *buffer_for_artifact(const char *artifact_name) {
  if (artifact_name == nullptr) return nullptr;
  if (strcmp(artifact_name, kMetadataArtifactName) == 0) {
    return &g_metadata_buffer;
  }
  if (strcmp(artifact_name, kManifestArtifactName) == 0) {
    return &g_manifest_buffer;
  }
  if (strcmp(artifact_name, kSegmentTasksArtifactName) == 0) {
    return &g_segment_tasks_buffer;
  }
  if (strcmp(artifact_name, kQuarantineStoreArtifactName) == 0) {
    return &g_quarantine_buffer;
  }
  return nullptr;
}

class Session_scope_guard {
 public:
  Session_scope_guard(innodb_vector_truth_store::Session *session, bool read_write)
      : m_session(session), m_owns(false) {
    if (m_session == nullptr) {
      m_session = innodb_vector_truth_store::begin_session(read_write);
      m_owns = true;
    }
  }

  ~Session_scope_guard() {
    if (!m_owns || m_session == nullptr) return;
    if (!m_session->finished) {
      innodb_vector_truth_store::rollback_session(m_session);
    }
    innodb_vector_truth_store::close_session(m_session);
  }

  innodb_vector_truth_store::Session *get() const { return m_session; }
  bool owns_session() const { return m_owns; }

 private:
  innodb_vector_truth_store::Session *m_session;
  bool m_owns;
};

bool commit_session_if_owned(
    const Session_scope_guard &session_guard,
    innodb_vector_truth_store::Session *active_session) {
  if (!session_guard.owns_session()) return true;
  return innodb_vector_truth_store::commit_session(active_session);
}

bool do_load_artifact(const char *artifact_name, std::string *payload, bool *found,
                      innodb_vector_truth_store::Session *session) {
  if (payload == nullptr || found == nullptr) return false;

  auto *buffer = buffer_for_artifact(artifact_name);
  if (buffer == nullptr) return false;
  dict_table_t *table = hidden_table_for_artifact(artifact_name);
  if (table == nullptr) return false;
  return buffer->get(table, payload, found, session);
}

bool do_save_artifact(const char *artifact_name, const std::string &payload,
                      innodb_vector_truth_store::Session *session) {
  auto *buffer = buffer_for_artifact(artifact_name);
  if (buffer == nullptr) return false;
  dict_table_t *table = hidden_table_for_artifact(artifact_name);
  if (table == nullptr) return false;

  Session_scope_guard session_guard(session, true);
  auto *active_session = session_guard.get();
  if (active_session == nullptr) return false;

  const bool ok = buffer->replace(table, payload, active_session);
  if (!ok) {
    ib::error() << "vector truth store save failed for artifact "
                << artifact_name << " with payload bytes " << payload.size();
    return false;
  }
  if (session_guard.owns_session()) {
    const bool committed =
        innodb_vector_truth_store::commit_session(active_session);
    if (!committed) {
      ib::error() << "vector truth store commit failed for artifact "
                  << artifact_name;
    }
    return committed;
  }
  return true;
}

bool do_load_committed_rows(
    std::vector<innodb_vector_truth_store::committed_row> *rows, bool *found,
    innodb_vector_truth_store::Session *session) {
  if (rows == nullptr || found == nullptr) return false;
  dict_table_t *table = hidden_table_for_artifact(kCommittedArtifactName);
  if (table == nullptr) return false;
  return g_committed_row_buffer.get_committed(table, rows, found, session);
}

bool do_load_committed_rows_for_index(
    const std::string &index_name,
    std::vector<innodb_vector_truth_store::committed_row> *rows, bool *found,
    innodb_vector_truth_store::Session *session) {
  if (index_name.empty() || rows == nullptr || found == nullptr) return false;
  dict_table_t *table = hidden_table_for_artifact(kCommittedArtifactName);
  if (table == nullptr) return false;
  return g_committed_row_buffer.get_committed_for_index(table, index_name, rows,
                                                       found, session);
}

bool do_visit_committed_rows_for_index(
    const std::string &index_name,
    const std::function<bool(
        const innodb_vector_truth_store::committed_row &row)> &visitor,
    bool *found, innodb_vector_truth_store::Session *session) {
  if (index_name.empty() || !visitor || found == nullptr) return false;
  dict_table_t *table = hidden_table_for_artifact(kCommittedArtifactName);
  if (table == nullptr) return false;
  return g_committed_row_buffer.visit_committed_for_index(table, index_name,
                                                         visitor, found, session);
}

bool do_find_committed_row(const std::string &index_name, uint64_t doc_id,
                           innodb_vector_truth_store::committed_row *row,
                           bool *found,
                           innodb_vector_truth_store::Session *session) {
  if (index_name.empty() || row == nullptr || found == nullptr) return false;
  dict_table_t *table = hidden_table_for_artifact(kCommittedArtifactName);
  if (table == nullptr) return false;
  return g_committed_row_buffer.find_committed(table, index_name, doc_id, row,
                                              found, session);
}

bool do_save_committed_rows(
    const std::vector<innodb_vector_truth_store::committed_row> &rows,
    innodb_vector_truth_store::Session *session) {
  dict_table_t *table = hidden_table_for_artifact(kCommittedArtifactName);
  if (table == nullptr) return false;

  Session_scope_guard session_guard(session, true);
  auto *active_session = session_guard.get();
  if (active_session == nullptr) return false;

  Row_truth_table_buffer buffer;
  const bool ok = buffer.replace_committed(table, rows, active_session);
  if (!ok) return false;
  return commit_session_if_owned(session_guard, active_session);
}

bool do_apply_committed_delta(
    const std::vector<innodb_vector_truth_store::change_log_row> &rows,
    innodb_vector_truth_store::Session *session) {
  dict_table_t *table = hidden_table_for_artifact(kCommittedArtifactName);
  if (table == nullptr) return false;

  Session_scope_guard session_guard(session, true);
  auto *active_session = session_guard.get();
  if (active_session == nullptr) return false;

  Row_truth_table_buffer buffer;
  const bool ok = buffer.apply_committed_delta(table, rows, active_session);
  if (!ok) return false;
  return commit_session_if_owned(session_guard, active_session);
}

bool do_load_change_log_rows(
    std::vector<innodb_vector_truth_store::change_log_row> *rows, bool *found,
    innodb_vector_truth_store::Session *session) {
  if (rows == nullptr || found == nullptr) return false;
  dict_table_t *table = hidden_table_for_artifact(kChangeLogArtifactName);
  if (table == nullptr) return false;
  return g_changelog_row_buffer.get_change_log(table, rows, found, session);
}

bool do_save_change_log_rows(
    const std::vector<innodb_vector_truth_store::change_log_row> &rows,
    innodb_vector_truth_store::Session *session) {
  dict_table_t *table = hidden_table_for_artifact(kChangeLogArtifactName);
  if (table == nullptr) return false;

  Session_scope_guard session_guard(session, true);
  auto *active_session = session_guard.get();
  if (active_session == nullptr) return false;

  Row_truth_table_buffer buffer;
  const bool ok = buffer.replace_change_log(table, rows, active_session);
  if (!ok) return false;
  return commit_session_if_owned(session_guard, active_session);
}

bool do_append_change_log_delta(
    const std::vector<innodb_vector_truth_store::change_log_row> &rows,
    innodb_vector_truth_store::Session *session) {
  dict_table_t *table = hidden_table_for_artifact(kChangeLogArtifactName);
  if (table == nullptr) return false;

  Session_scope_guard session_guard(session, true);
  auto *active_session = session_guard.get();
  if (active_session == nullptr) return false;

  Row_truth_table_buffer buffer;
  const bool ok = buffer.append_change_log(table, rows, active_session);
  if (!ok) return false;
  return commit_session_if_owned(session_guard, active_session);
}

bool do_erase_change_log_sequences(
    const std::vector<uint64_t> &sequences,
    innodb_vector_truth_store::Session *session) {
  dict_table_t *table = hidden_table_for_artifact(kChangeLogArtifactName);
  if (table == nullptr) return false;

  Session_scope_guard session_guard(session, true);
  auto *active_session = session_guard.get();
  if (active_session == nullptr) return false;

  Row_truth_table_buffer buffer;
  const bool ok =
      buffer.erase_change_log_sequences(table, sequences, active_session);
  if (!ok) return false;
  return commit_session_if_owned(session_guard, active_session);
}

bool do_load_prepared_rows(
    std::vector<innodb_vector_truth_store::prepared_change_row> *rows,
    bool *found, innodb_vector_truth_store::Session *session) {
  if (rows == nullptr || found == nullptr) return false;
  dict_table_t *table = hidden_table_for_artifact(kPreparedArtifactName);
  if (table == nullptr) {
    /*
      Pre-vector data directories do not have the hidden prepared truth table
      during early TC recovery. Absence of that table means there cannot be
      vector prepared rows to recover.
    */
    rows->clear();
    *found = false;
    return true;
  }
  return g_prepared_row_buffer.get_prepared(table, rows, found, session);
}

bool do_save_prepared_rows(
    const std::vector<innodb_vector_truth_store::prepared_change_row> &rows,
    innodb_vector_truth_store::Session *session) {
  dict_table_t *table = hidden_table_for_artifact(kPreparedArtifactName);
  if (table == nullptr) return false;

  Session_scope_guard session_guard(session, true);
  auto *active_session = session_guard.get();
  if (active_session == nullptr) return false;

  Row_truth_table_buffer buffer;
  const bool ok = buffer.replace_prepared(table, rows, active_session);
  if (!ok) return false;
  return commit_session_if_owned(session_guard, active_session);
}

bool do_load_publication_intents(
    std::vector<innodb_vector_truth_store::publication_intent_row> *rows,
    bool *found, innodb_vector_truth_store::Session *session) {
  if (rows == nullptr || found == nullptr) return false;
  dict_table_t *table =
      hidden_table_for_artifact(kPublicationIntentsArtifactName);
  if (table == nullptr) return false;

  Row_truth_table_buffer buffer;
  return buffer.get_publication_intents(table, rows, found, session);
}

bool do_insert_publication_intent(
    const innodb_vector_truth_store::publication_intent_row &row,
    innodb_vector_truth_store::Session *session) {
  if (session == nullptr) return false;
  dict_table_t *table =
      hidden_table_for_artifact(kPublicationIntentsArtifactName);
  if (table == nullptr) return false;

  Row_truth_table_buffer buffer;
  return buffer.insert_publication_intent(table, row, session);
}

bool do_delete_publication_intent(
    const std::string &index_name, uint64_t publication_id,
    innodb_vector_truth_store::Session *session) {
  dict_table_t *table =
      hidden_table_for_artifact(kPublicationIntentsArtifactName);
  if (table == nullptr) return false;

  Session_scope_guard session_guard(session, true);
  auto *active_session = session_guard.get();
  if (active_session == nullptr) return false;

  Row_truth_table_buffer buffer;
  if (!buffer.delete_publication_intent(table, index_name, publication_id,
                                        active_session)) {
    return false;
  }
  return commit_session_if_owned(session_guard, active_session);
}

bool do_delete_artifact(const char *artifact_name,
                        innodb_vector_truth_store::Session *session) {
  auto *buffer = buffer_for_artifact(artifact_name);
  if (buffer == nullptr) return false;
  dict_table_t *table = hidden_table_for_artifact(artifact_name);
  if (table == nullptr) return false;

  Session_scope_guard session_guard(session, true);
  auto *active_session = session_guard.get();
  if (active_session == nullptr) return false;

  const bool ok = buffer->remove_all(table, active_session);
  if (!ok) return false;
  if (session_guard.owns_session()) {
    return innodb_vector_truth_store::commit_session(active_session);
  }
  return true;
}

}  // namespace

namespace innodb_vector_truth_store {

namespace {

Session *create_session(trx_t *trx, bool owns_trx) {
  if (trx == nullptr) return nullptr;

  auto *session = ut::new_withkey<Session>(UT_NEW_THIS_FILE_PSI_KEY);
  if (session == nullptr) return nullptr;
  session->trx = trx;
  session->owns_trx = owns_trx;
  session->heap = mem_heap_create(128, UT_LOCATION_HERE);
  que_t *graph = static_cast<que_fork_t *>(que_node_get_parent(
      pars_complete_graph_for_exec(nullptr, session->trx, session->heap, nullptr)));
  session->thr = que_fork_start_command(graph);
  return session;
}

}  // namespace

Session *begin_session(bool read_write) {
  trx_t *trx = trx_allocate_for_background();
  if (trx == nullptr) return nullptr;

  if (read_write) {
    trx_start_internal(trx, UT_LOCATION_HERE);
  } else {
    trx_start_internal_read_only(trx, UT_LOCATION_HERE);
  }
  Session *session = create_session(trx, true);
  if (session == nullptr) trx_free_for_background(trx);
  return session;
}

Session *begin_attached_session(THD *thd) {
  if (thd == nullptr) return nullptr;
  trx_t *trx = check_trx_exists(thd);
  handlerton *hton = ha_resolve_by_legacy_type(thd, DB_TYPE_INNODB);
  if (trx == nullptr || hton == nullptr) return nullptr;
  if (!trx_is_started(trx)) {
    // Match InnoDB handler writes: mark the autocommit statement as locking
    // before starting its read-write transaction.
    ++trx->will_lock;
  }
  trx_start_if_not_started_xa(trx, true, UT_LOCATION_HERE);
  innobase_register_trx(hton, thd, trx);
  return create_session(trx, false);
}

Session *begin_attached_session_for_prepare(THD *thd) {
  if (thd == nullptr) return nullptr;
  trx_t *trx = check_trx_exists(thd);
  if (trx == nullptr || !trx_is_started(trx) ||
      !trx_is_registered_for_2pc(trx)) {
    return nullptr;
  }
  return create_session(trx, false);
}

void close_session(Session *session) {
  if (session == nullptr) return;
  if (session->thr != nullptr && session->trx != nullptr) {
    que_thr_stop_for_mysql_no_error(session->thr, session->trx);
    session->thr = nullptr;
  }
  if (session->owns_trx && !session->finished && session->trx != nullptr) {
    (void)trx_rollback_to_savepoint(session->trx, nullptr);
    session->finished = true;
  }
  if (session->owns_trx && session->trx != nullptr) {
    trx_free_for_background(session->trx);
  }
  session->trx = nullptr;
  if (session->heap != nullptr) {
    mem_heap_free(session->heap);
    session->heap = nullptr;
  }
  ut::delete_(session);
}

bool commit_session(Session *session) {
  if (session == nullptr || session->trx == nullptr || !session->owns_trx ||
      session->finished) {
    return false;
  }
  const dberr_t err = trx_commit_for_mysql(session->trx);
  if (err != DB_SUCCESS) {
    ib::error() << "vector truth store background transaction commit failed with error "
                << err;
    return false;
  }
  session->finished = true;
  return true;
}

void rollback_session(Session *session) {
  if (session == nullptr || session->trx == nullptr || !session->owns_trx ||
      session->finished) {
    return;
  }
  if (session->thr != nullptr) {
    que_thr_stop_for_mysql_no_error(session->thr, session->trx);
    session->thr = nullptr;
  }
  (void)trx_rollback_to_savepoint(session->trx, nullptr);
  session->finished = true;
}

bool load_artifact(const char *artifact_name, std::string *payload, bool *found,
                   Session *session) {
  return do_load_artifact(artifact_name, payload, found, session);
}

bool save_artifact(const char *artifact_name, const std::string &payload,
                   Session *session) {
  return do_save_artifact(artifact_name, payload, session);
}

bool load_committed_rows(std::vector<committed_row> *rows, bool *found,
                         Session *session) {
  return do_load_committed_rows(rows, found, session);
}

bool load_committed_rows_for_index(const std::string &index_name,
                                   std::vector<committed_row> *rows,
                                   bool *found, Session *session) {
  return do_load_committed_rows_for_index(index_name, rows, found, session);
}

bool visit_committed_rows_for_index(
    const std::string &index_name,
    const std::function<bool(const committed_row &row)> &visitor, bool *found,
    Session *session) {
  return do_visit_committed_rows_for_index(index_name, visitor, found, session);
}

bool find_committed_row(const std::string &index_name, uint64_t doc_id,
                        committed_row *row, bool *found, Session *session) {
  return do_find_committed_row(index_name, doc_id, row, found, session);
}

bool save_committed_rows(const std::vector<committed_row> &rows,
                         Session *session) {
  return do_save_committed_rows(rows, session);
}

bool apply_committed_delta(const std::vector<change_log_row> &rows,
                           Session *session) {
  return do_apply_committed_delta(rows, session);
}

bool load_change_log_rows(std::vector<change_log_row> *rows, bool *found,
                          Session *session) {
  return do_load_change_log_rows(rows, found, session);
}

bool save_change_log_rows(const std::vector<change_log_row> &rows,
                          Session *session) {
  return do_save_change_log_rows(rows, session);
}

bool append_change_log_delta(const std::vector<change_log_row> &rows,
                             Session *session) {
  return do_append_change_log_delta(rows, session);
}

bool erase_change_log_sequences(const std::vector<uint64_t> &sequences,
                                Session *session) {
  return do_erase_change_log_sequences(sequences, session);
}

bool load_prepared_rows(std::vector<prepared_change_row> *rows, bool *found,
                        Session *session) {
  return do_load_prepared_rows(rows, found, session);
}

bool save_prepared_rows(const std::vector<prepared_change_row> &rows,
                        Session *session) {
  return do_save_prepared_rows(rows, session);
}

bool load_publication_intents(std::vector<publication_intent_row> *rows,
                              bool *found, Session *session) {
  return do_load_publication_intents(rows, found, session);
}

bool insert_publication_intent(const publication_intent_row &row,
                               Session *session) {
  return do_insert_publication_intent(row, session);
}

bool delete_publication_intent(const std::string &index_name,
                               uint64_t publication_id, Session *session) {
  return do_delete_publication_intent(index_name, publication_id, session);
}

bool delete_artifact(const char *artifact_name, Session *session) {
  return do_delete_artifact(artifact_name, session);
}

}  // namespace innodb_vector_truth_store
