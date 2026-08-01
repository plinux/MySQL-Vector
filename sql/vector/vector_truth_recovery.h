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

#ifndef SQL_VECTOR_TRUTH_RECOVERY_INCLUDED
#define SQL_VECTOR_TRUTH_RECOVERY_INCLUDED

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "sql/vector/vector_index_backend.h"

class THD;
struct TABLE;

namespace vector_truth_recovery {

/** One authoritative table binding that must be scanned for recovery. */
struct index_scan_spec {
  std::string index_name;
  std::string schema_name;
  std::string table_name;
  std::string column_name;
  std::string doc_id_column_name;
};

/** Recovered vectors keyed by document ID for one index. */
using index_entries =
    std::unordered_map<uint64_t, vector_index::vector_data>;
/** Complete recovered state keyed by index name. */
using recovered_state = std::unordered_map<std::string, index_entries>;
/** Callback that atomically publishes one complete recovered state. */
using publish_callback = std::function<bool(const recovered_state &)>;

/**
  Scan bound InnoDB tables while writes are blocked, then publish the result.

  All target table metadata locks are acquired before the first row is read and
  remain held until publish returns. The caller must not hold the vector registry
  mutex while invoking this function.

  @param thd Current SQL thread.
  @param specs Index-to-table bindings to recover.
  @param publish Callback that validates and publishes the complete snapshot.

  @retval true Every table was scanned and publish succeeded.
  @retval false A lock, open, scan, decode, or publish operation failed.
*/
bool scan_and_publish(THD *thd, std::vector<index_scan_spec> specs,
                      const publish_callback &publish);

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
/** Test-only access to the deterministic recovery scan ordering. */
bool index_scan_spec_less_for_testing(const index_scan_spec &lhs,
                                      const index_scan_spec &rhs);
/** Test-only access to authoritative table binding validation. */
bool table_matches_binding_for_testing(TABLE *table,
                                       const index_scan_spec &spec);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_truth_recovery

#endif  // SQL_VECTOR_TRUTH_RECOVERY_INCLUDED
