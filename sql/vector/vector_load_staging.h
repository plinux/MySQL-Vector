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

#ifndef SQL_VECTOR_LOAD_STAGING_INCLUDED
#define SQL_VECTOR_LOAD_STAGING_INCLUDED

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sql/vector/vector_index_truth_store.h"
#include "sql/vector/vector_statement_publication.h"

namespace vector_index {

/** Durable identity of one managed LOAD VECTOR DATA input. */
struct load_staging_identity {
  std::string index_name;
  uint64_t publication_id{0};
};

/** Immutable file metadata persisted in the statement publication payload. */
struct load_staging_artifact {
  std::string identity;
  uint64_t vector_size{0};
  uint64_t vector_checksum{0};
  bool has_docids{false};
  uint64_t docid_size{0};
  uint64_t docid_checksum{0};
};

/** Managed paths returned only after size and checksum verification. */
struct load_staging_paths {
  std::string vector_filename;
  std::string docid_filename;
};

/**
  Copy user input into a MySQL-managed immutable staging artifact.

  The source is not trusted after this function returns. Callers must validate
  and binlog the returned managed files rather than reopening the source paths.
*/
bool stage_load_artifact(const load_staging_identity &identity,
                         const std::string &vector_filename,
                         const std::string &docid_filename,
                         const std::string &format,
                         load_staging_artifact *artifact, std::string *error);

/** Resolve and verify one managed artifact before publication or recovery. */
bool verify_load_artifact(const load_staging_identity &identity,
                          const load_staging_artifact &artifact,
                          const std::string &format, load_staging_paths *paths,
                          std::string *error);

/** Encode the managed artifact reference into a durable operation payload. */
bool make_load_staging_payload(
    const load_staging_identity &identity,
    const load_staging_artifact &artifact, bool replace_duplicates,
    bool rebuild_after_load, size_t dimension, const std::string &format,
    vector_statement_publication::operation_payload *payload);

/** Decode and validate the managed artifact reference in a durable intent. */
bool parse_load_staging_payload(
    const load_staging_identity &identity,
    const vector_statement_publication::operation_payload &payload,
    load_staging_artifact *artifact, bool *replace_duplicates,
    bool *rebuild_after_load, size_t *dimension, std::string *format);

/** Remove managed artifacts referenced by rolled back or acknowledged intents.
 */
void discard_load_staging_intents(
    const std::vector<vector_index_truth_store::publication_intent> &intents);

/** Remove one artifact after its durable intent is acknowledged. */
bool remove_load_staging_artifact(const load_staging_identity &identity);

/** Remove an artifact only when the intent contains a managed LOAD payload. */
bool remove_managed_load_staging_artifact(
    const vector_index_truth_store::publication_intent &intent);

/** Remove directories not referenced by durable intents or active statements.
 */
bool cleanup_orphaned_load_staging_artifacts(
    const std::vector<vector_index_truth_store::publication_intent> &intents);

}  // namespace vector_index

#endif  // SQL_VECTOR_LOAD_STAGING_INCLUDED
