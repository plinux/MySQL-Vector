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

#ifndef dict0vectruth_h
#define dict0vectruth_h

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

class THD;

namespace innodb_vector_truth_store {

struct Session;

/**
  Semantic committed vector row exchanged across the SQL/InnoDB boundary.

  This is intentionally not a DD physical row description; the DD layout and
  clustered-index accessors remain private to dict0vectruth.cc.
*/
struct committed_row {
  std::string index_name;
  uint64_t doc_id{0};
  uint32_t dimension{0};
  std::string vector_payload;
};

/** Semantic change-log row exchanged with the vector truth-store facade. */
struct change_log_row {
  uint64_t sequence{0};
  uint64_t txn_id{0};
  uint8_t op{0};
  std::string index_name;
  uint64_t doc_id{0};
  uint32_t dimension{0};
  std::string vector_payload;
};

/** Semantic prepared change row exchanged with the vector XA participant. */
struct prepared_change_row {
  uint64_t txn_id{0};
  uint64_t row_no{0};
  uint64_t format_id{0};
  uint64_t gtrid_length{0};
  uint64_t bqual_length{0};
  std::string xid_data;
  uint8_t prepared_in_tc{0};
  uint8_t op{0};
  std::string index_name;
  uint64_t doc_id{0};
  uint32_t dimension{0};
  std::string vector_payload;
};

Session *begin_session(bool read_write);
/**
  Open a truth-store session on the InnoDB transaction owned by a THD.

  The returned session owns only its query graph and heap. Closing it never
  commits, rolls back, or frees the borrowed user transaction.

  @param thd user thread whose active InnoDB transaction is borrowed
  @return attached session, or nullptr when no InnoDB transaction is available
*/
Session *begin_attached_session(THD *thd);
void close_session(Session *session);
bool commit_session(Session *session);
void rollback_session(Session *session);

bool load_artifact(const char *artifact_name, std::string *payload, bool *found,
                   Session *session = nullptr);
bool save_artifact(const char *artifact_name, const std::string &payload,
                   Session *session = nullptr);
bool load_committed_rows(std::vector<committed_row> *rows, bool *found,
                         Session *session = nullptr);
bool load_committed_rows_for_index(const std::string &index_name,
                                   std::vector<committed_row> *rows,
                                   bool *found,
                                   Session *session = nullptr);
bool visit_committed_rows_for_index(
    const std::string &index_name,
    const std::function<bool(const committed_row &row)> &visitor, bool *found,
    Session *session = nullptr);
bool find_committed_row(const std::string &index_name, uint64_t doc_id,
                        committed_row *row, bool *found,
                        Session *session = nullptr);
bool save_committed_rows(const std::vector<committed_row> &rows,
                         Session *session = nullptr);
bool apply_committed_delta(const std::vector<change_log_row> &rows,
                           Session *session = nullptr);
bool load_change_log_rows(std::vector<change_log_row> *rows, bool *found,
                          Session *session = nullptr);
bool save_change_log_rows(const std::vector<change_log_row> &rows,
                          Session *session = nullptr);
bool append_change_log_delta(const std::vector<change_log_row> &rows,
                             Session *session = nullptr);
bool load_prepared_rows(std::vector<prepared_change_row> *rows, bool *found,
                        Session *session = nullptr);
bool save_prepared_rows(const std::vector<prepared_change_row> &rows,
                        Session *session = nullptr);
bool delete_artifact(const char *artifact_name, Session *session = nullptr);

}  // namespace innodb_vector_truth_store

#endif /* dict0vectruth_h */
