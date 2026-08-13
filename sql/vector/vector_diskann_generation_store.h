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

#ifndef SQL_VECTOR_DISKANN_GENERATION_STORE_INCLUDED
#define SQL_VECTOR_DISKANN_GENERATION_STORE_INCLUDED

#include <cstdint>
#include <string>

namespace vector_index {

/** Durable identity attached to one DiskANN artifact generation. */
struct diskann_artifact_identity {
  uint64_t index_identity{0};
  uint64_t truth_generation{0};
  uint64_t config_generation{0};
  uint64_t doc_id_count{0};
  uint64_t doc_id_checksum{0};
  uint32_t dimension{0};
  std::string metric;
  std::string mode;
  std::string provider;
  std::string consistency_mode;
  std::string schema_name;
  std::string table_name;
  std::string column_name;
  std::string doc_id_column_name;
};

enum class diskann_swap_state {
  kPrepared,
  kOldSaved,
  kNewInstalled,
  kVerified,
  kComplete
};

/** In-memory handle for one durable DiskANN directory swap. */
struct diskann_generation_swap {
  std::string live_directory;
  std::string staging_directory;
  std::string backup_directory;
  std::string journal_path;
  diskann_artifact_identity target;
  diskann_swap_state state{diskann_swap_state::kPrepared};
  bool had_old_generation{false};
};

/** Compare the identity fields that bind an artifact to registry metadata. */
bool diskann_artifact_publication_matches(
    const diskann_artifact_identity &expected,
    const diskann_artifact_identity &actual);

/** Write the generation manifest and durable prepared journal. */
bool prepare_diskann_generation_swap(const std::string &live_directory,
                                     const std::string &build_staging_directory,
                                     const diskann_artifact_identity &target,
                                     diskann_generation_swap *swap,
                                     std::string *error);

/** Move the old and new generations, retaining rollback state. */
bool publish_diskann_generation_swap(diskann_generation_swap *swap,
                                     std::string *error);

/** Mark the installed generation as load-verified. */
bool verify_diskann_generation_swap(diskann_generation_swap *swap,
                                    std::string *error);

/** Restore the previous generation and remove the candidate generation. */
bool rollback_diskann_generation_swap(diskann_generation_swap *swap,
                                      std::string *error);

/** Delete the previous generation and complete the durable journal. */
bool finalize_diskann_generation_swap(diskann_generation_swap *swap,
                                      std::string *error);

/** Recover an interrupted swap according to durable registry publication. */
bool recover_diskann_generation_swap(
    const std::string &live_directory,
    const diskann_artifact_identity &expected_publication, bool *journal_found,
    std::string *error);

/** Validate the live artifact manifest, identity and file checksums. */
bool validate_diskann_generation_store(
    const std::string &live_directory,
    const diskann_artifact_identity &expected_publication,
    diskann_artifact_identity *actual_identity, std::string *error);

/** Return the stable journal path adjacent to a live store directory. */
std::string diskann_generation_journal_path(const std::string &live_directory);

}  // namespace vector_index

#endif  // SQL_VECTOR_DISKANN_GENERATION_STORE_INCLUDED
