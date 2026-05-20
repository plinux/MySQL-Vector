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

#include "sql/dd/impl/tables/vector_index_truth_tables.h"

namespace dd {
namespace tables {

namespace {

void init_singleton_payload_table(Object_table_impl *table,
                                  const char *table_name) {
  table->target_table_definition()->set_table_name(table_name);
  table->target_table_definition()->add_field(
      0, "FIELD_SINGLETON_ID", "singleton_id INT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(
      1, "FIELD_LAYOUT_VERSION", "layout_version INT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(2, "FIELD_PAYLOAD",
                                              "payload LONGBLOB NOT NULL");
  table->target_table_definition()->add_index(
      0, "INDEX_PK_SINGLETON_ID", "PRIMARY KEY (singleton_id)");
}

void init_committed_rows_table(Object_table_impl *table) {
  table->target_table_definition()->set_table_name(
      "vector_index_truth_committed");
  table->target_table_definition()->add_field(
      0, "FIELD_INDEX_NAME", "index_name VARBINARY(255) NOT NULL");
  table->target_table_definition()->add_field(
      1, "FIELD_DOC_ID", "doc_id BIGINT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(
      2, "FIELD_DIMENSION", "dimension INT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(
      3, "FIELD_VECTOR_PAYLOAD", "vector_payload LONGBLOB NOT NULL");
  table->target_table_definition()->add_index(
      0, "INDEX_PK_INDEX_DOC", "PRIMARY KEY (index_name, doc_id)");
}

void init_changelog_rows_table(Object_table_impl *table) {
  table->target_table_definition()->set_table_name(
      "vector_index_truth_changelog");
  table->target_table_definition()->add_field(
      0, "FIELD_SEQUENCE", "sequence BIGINT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(
      1, "FIELD_TXN_ID", "txn_id BIGINT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(2, "FIELD_OP",
                                              "op TINYINT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(
      3, "FIELD_INDEX_NAME", "index_name VARBINARY(255) NOT NULL");
  table->target_table_definition()->add_field(
      4, "FIELD_DOC_ID", "doc_id BIGINT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(
      5, "FIELD_DIMENSION", "dimension INT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(
      6, "FIELD_VECTOR_PAYLOAD", "vector_payload LONGBLOB NOT NULL");
  table->target_table_definition()->add_index(
      0, "INDEX_PK_SEQUENCE", "PRIMARY KEY (sequence)");
}

void init_prepared_rows_table(Object_table_impl *table) {
  table->target_table_definition()->set_table_name(
      "vector_index_truth_prepared");
  table->target_table_definition()->add_field(
      0, "FIELD_TXN_ID", "txn_id BIGINT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(
      1, "FIELD_ROW_NO", "row_no BIGINT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(
      2, "FIELD_FORMAT_ID", "format_id BIGINT NOT NULL");
  table->target_table_definition()->add_field(
      3, "FIELD_GTRID_LENGTH", "gtrid_length BIGINT NOT NULL");
  table->target_table_definition()->add_field(
      4, "FIELD_BQUAL_LENGTH", "bqual_length BIGINT NOT NULL");
  table->target_table_definition()->add_field(
      5, "FIELD_XID_DATA", "xid_data VARBINARY(128) NOT NULL");
  table->target_table_definition()->add_field(
      6, "FIELD_PREPARED_IN_TC", "prepared_in_tc TINYINT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(7, "FIELD_OP",
                                              "op TINYINT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(
      8, "FIELD_INDEX_NAME", "index_name VARBINARY(255) NOT NULL");
  table->target_table_definition()->add_field(
      9, "FIELD_DOC_ID", "doc_id BIGINT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(
      10, "FIELD_DIMENSION", "dimension INT UNSIGNED NOT NULL");
  table->target_table_definition()->add_field(
      11, "FIELD_VECTOR_PAYLOAD", "vector_payload LONGBLOB NOT NULL");
  table->target_table_definition()->add_index(
      0, "INDEX_PK_TXN_ROW", "PRIMARY KEY (txn_id, row_no)");
}

}  // namespace

Vector_index_truth_metadata::Vector_index_truth_metadata() {
  init_singleton_payload_table(this, "vector_index_truth_metadata");
}

const Vector_index_truth_metadata &Vector_index_truth_metadata::instance() {
  static Vector_index_truth_metadata *s_instance =
      new Vector_index_truth_metadata();
  return *s_instance;
}

Vector_index_truth_committed::Vector_index_truth_committed() {
  init_committed_rows_table(this);
}

const Vector_index_truth_committed &Vector_index_truth_committed::instance() {
  static Vector_index_truth_committed *s_instance =
      new Vector_index_truth_committed();
  return *s_instance;
}

Vector_index_truth_manifest::Vector_index_truth_manifest() {
  init_singleton_payload_table(this, "vector_index_truth_manifest");
}

const Vector_index_truth_manifest &Vector_index_truth_manifest::instance() {
  static Vector_index_truth_manifest *s_instance =
      new Vector_index_truth_manifest();
  return *s_instance;
}

Vector_index_truth_changelog::Vector_index_truth_changelog() {
  init_changelog_rows_table(this);
}

const Vector_index_truth_changelog &Vector_index_truth_changelog::instance() {
  static Vector_index_truth_changelog *s_instance =
      new Vector_index_truth_changelog();
  return *s_instance;
}

Vector_index_truth_prepared::Vector_index_truth_prepared() {
  init_prepared_rows_table(this);
}

const Vector_index_truth_prepared &Vector_index_truth_prepared::instance() {
  static Vector_index_truth_prepared *s_instance =
      new Vector_index_truth_prepared();
  return *s_instance;
}

Vector_index_truth_store_quarantine::Vector_index_truth_store_quarantine() {
  init_singleton_payload_table(this, "vector_index_truth_store_quarantine");
}

const Vector_index_truth_store_quarantine &
Vector_index_truth_store_quarantine::instance() {
  static Vector_index_truth_store_quarantine *s_instance =
      new Vector_index_truth_store_quarantine();
  return *s_instance;
}

}  // namespace tables
}  // namespace dd
