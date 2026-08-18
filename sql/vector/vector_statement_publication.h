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

#ifndef SQL_VECTOR_STATEMENT_PUBLICATION_INCLUDED
#define SQL_VECTOR_STATEMENT_PUBLICATION_INCLUDED

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sql/vector/vector_index_truth_store.h"

namespace vector_statement_publication {

class reservation_lease;

/** Optional cancellation predicate for one publication reservation wait. */
using cancellation_predicate = std::function<bool()>;

/** Configuration field changed by a staged administrative statement. */
enum class config_change : uint8_t {
  kSearchEf = 1,
  kHnswBuildParams = 2,
  kFaissIvfParams = 3,
  kFaissIvfPqParams = 4,
  kDiskannBuildParams = 5,
  kDiskannSearchComplexity = 6,
  kDiskannSearchBeamwidth = 7,
  kDiskannPqCodeBudgetSize = 8,
  kDiskannDiskPqDims = 9,
  kDiskannAccelerateBuild = 10,
  kDiskannShuffleBuild = 11,
  kDiskannUseBfsCache = 12,
  kDiskannBuildMode = 13,
};

/**
  Versioned operation payload persisted with one publication intent.

  The operation enum selects the interpretation of the fields. Keeping the
  payload independent of Item objects makes crash recovery use the same
  publication path as the original statement without reparsing SQL.
*/
struct operation_payload {
  config_change config{config_change::kSearchEf};
  std::vector<uint64_t> unsigned_values;
  std::vector<std::string> string_values;
  std::string binary_value;
};

/** Encode one operation payload into the durable intent representation. */
bool encode_operation_payload(const operation_payload &payload,
                              std::string *encoded);

/** Decode and structurally validate one durable operation payload. */
bool decode_operation_payload(const std::string &encoded,
                              operation_payload *payload);

/** Compare two index-name collections after dropping empties and duplicates. */
bool index_sets_equal(const std::vector<std::string> &lhs,
                      const std::vector<std::string> &rhs);

/**
  Ordered ownership for one or more vector-index publication targets.

  The catalog lock prevents an all-index operation from racing with CREATE or
  DROP. Per-index locks serialize writers for the same index while preserving
  concurrency for independent indexes. Read guards hide an unpublished
  statement candidate from search and observability paths.
*/
class publication_guard {
 public:
  publication_guard() = default;
  publication_guard(const publication_guard &) = delete;
  publication_guard &operator=(const publication_guard &) = delete;
  publication_guard(publication_guard &&) = default;
  publication_guard &operator=(publication_guard &&) = default;

  /** Acquire shared catalog ownership and exclusive target-index ownership. */
  bool lock_indexes(
      const std::vector<std::string> &index_names,
      const cancellation_predicate &cancelled = cancellation_predicate{});

  /** Try once without waiting for conflicting index/catalog ownership. */
  bool try_lock_indexes(
      const std::vector<std::string> &index_names,
      const cancellation_predicate &cancelled = cancellation_predicate{});

  /** Acquire exclusive catalog ownership for an all-index/catalog mutation. */
  bool lock_catalog(
      const cancellation_predicate &cancelled = cancellation_predicate{});

  /** True when this object currently owns a publication boundary. */
  bool owns_lock() const;

 private:
  std::shared_ptr<reservation_lease> m_lease;
};

/** Shared ownership used by search and per-index observability reads. */
class read_guard {
 public:
  read_guard() = default;
  read_guard(const read_guard &) = delete;
  read_guard &operator=(const read_guard &) = delete;
  read_guard(read_guard &&) = default;
  read_guard &operator=(read_guard &&) = default;

  bool lock_index(
      const std::string &index_name,
      const cancellation_predicate &cancelled = cancellation_predicate{});
  bool lock_catalog(
      const cancellation_predicate &cancelled = cancellation_predicate{});
  bool owns_lock() const;

 private:
  std::shared_ptr<reservation_lease> m_lease;
};

}  // namespace vector_statement_publication

#endif  // SQL_VECTOR_STATEMENT_PUBLICATION_INCLUDED
