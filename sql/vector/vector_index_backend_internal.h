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

#ifndef SQL_VECTOR_INDEX_BACKEND_INTERNAL_INCLUDED
#define SQL_VECTOR_INDEX_BACKEND_INTERNAL_INCLUDED

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sql/vector/vector_index_backend.h"

namespace vector_index::detail {

extern const char *kFaissExternalSnapshotHeader;
extern const char *kFaissExternalManifestHeader;
extern const char *kDiskAnnExternalSnapshotHeader;
extern const char *kDiskAnnExternalManifestHeader;
extern const char *kVectorIndexDirectory;
extern const char *kFaissExternalManifestFilename;
extern const char *kDiskAnnExternalManifestFilename;
extern const char *kFaissExternalSnapshotPrefix;
extern const char *kDiskAnnExternalSnapshotPrefix;
extern const char *kFaissExternalSnapshotSuffix;

/**
  Validate and collect raw FBIN segments for a backend rebuild.

  @param reader Supplies raw segments.
  @param dimension Expected vector dimension.
  @param segments Collected validated segments.
  @param total_rows Total row count across collected segments.

  @retval true All supplied segments are valid and collected.
  @retval false Arguments, row counts, or FBIN metadata are invalid.
*/
bool collect_validated_raw_segments(const raw_vector_segment_reader &reader,
                                    size_t dimension,
                                    std::vector<raw_vector_segment> *segments,
                                    size_t *total_rows);
bool ensure_parent_directory(const std::string &path);
bool load_external_manifest_generation_from_file(const std::string &path,
                                                 const char *expected_header,
                                                 uint64_t *generation,
                                                 bool *exists);
bool save_external_manifest_generation_to_file(const std::string &path,
                                               const char *header,
                                               uint64_t generation);
std::string vector_index_root_path();
const char *external_snapshot_header(external_sidecar_profile profile);
const char *external_manifest_header(external_sidecar_profile profile);
const char *external_manifest_filename(external_sidecar_profile profile);
const char *external_snapshot_prefix(external_sidecar_profile profile);
std::string encoded_index_name(const std::string &index_name);
std::string external_snapshot_directory(const std::string &index_name);
std::string faiss_external_snapshot_directory(const std::string &index_name);
std::string external_manifest_path(const std::string &index_name,
                                   external_sidecar_profile profile);
bool remove_if_exists(const std::string &path);
std::string quarantine_path_for(const std::string &path);
bool quarantine_file_if_exists(const std::string &path);
bool remove_generated_snapshots_with_prefix(const std::string &directory,
                                            const std::string &prefix);
bool remove_dir_if_empty(const std::string &path);
std::string diskann_store_directory(const std::string &index_name);
bool remove_external_sidecar_artifacts(const std::string &index_name,
                                       external_sidecar_profile profile,
                                       bool remove_snapshot_directory);
bool remove_diskann_external_artifacts(const std::string &index_name,
                                       bool remove_native_store);

}  // namespace vector_index::detail

#endif  // SQL_VECTOR_INDEX_BACKEND_INTERNAL_INCLUDED
