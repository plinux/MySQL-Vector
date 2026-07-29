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

#ifndef SQL_VECTOR_DISKANN_ARTIFACT_LAYOUT_INCLUDED
#define SQL_VECTOR_DISKANN_ARTIFACT_LAYOUT_INCLUDED

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace vector_index {

/** File paths produced or consumed by the native DiskANN PQ build runtime. */
struct diskann_pq_artifact_paths {
  std::string pivot_path;
  std::string compressed_path;
  std::string centroid_path;
  std::string chunk_offsets_path;
  std::string docid_ordinal_path;
  std::string disk_index_path;
  std::string disk_index_medoids_path;
  std::string disk_index_centroids_path;
  std::string disk_index_pq_pivots_path;
  std::string sample_data_path;
  std::string cached_nodes_path;
  std::string artifact_manifest_path;
};

/** Header metadata expected in the native DiskANN PQ build artifacts. */
struct diskann_pq_artifact_metadata {
  uint64_t row_count{0};
  uint32_t dimension{0};
  uint32_t pq_chunks{0};
  uint32_t centroid_count{256};
  bool zero_mean{true};
};

/** In-memory payload for the native DiskANN PQ build artifacts. */
struct diskann_pq_artifact_payload {
  std::vector<float> pivots;
  std::vector<uint8_t> compressed_codes;
  std::vector<float> centroid;
  std::vector<uint32_t> chunk_offsets;
  std::vector<uint64_t> docid_ordinals;
};

/** Manifest proving which native PQ artifacts a bridge consumed. */
struct diskann_pq_bridge_manifest {
  uint64_t row_count{0};
  uint32_t dimension{0};
  uint32_t pq_chunks{0};
  uint32_t centroid_count{256};
  uint32_t max_degree{0};
  uint32_t build_complexity{0};
  uint32_t cache_nodes{0};
  uint64_t pivots_checksum{0};
  uint64_t compressed_checksum{0};
  uint64_t medoids_checksum{0};
  bool zero_mean{true};
  bool artifacts_consumed{false};
  bool official_pq_used{false};
};

/** Streaming writer for the row-major uint8 PQ compressed artifact. */
class diskann_compressed_artifact_writer {
 public:
  diskann_compressed_artifact_writer() = default;
  diskann_compressed_artifact_writer(
      const diskann_compressed_artifact_writer &) = delete;
  diskann_compressed_artifact_writer &operator=(
      const diskann_compressed_artifact_writer &) = delete;

  bool open(const std::string &path, uint32_t rows, uint32_t columns,
            std::string *error);
  bool write_block(const uint8_t *codes, uint32_t block_rows,
                   std::string *error);
  bool close(std::string *error);

 private:
  std::ofstream m_file;
  uint32_t m_rows{0};
  uint32_t m_columns{0};
  uint64_t m_written_rows{0};
  bool m_open{false};
};

/**
  Build the DiskANN-compatible PQ artifact path set for an index prefix.

  @param[in] index_prefix DiskANN index prefix.
  @param[out] paths Output path set.

  @retval true Paths were generated.
  @retval false Invalid output pointer or empty prefix.
*/
bool make_diskann_pq_artifact_paths(const std::string &index_prefix,
                                    diskann_pq_artifact_paths *paths);

/**
  Validate DiskANN PQ artifact headers against expected metadata.

  @param[in] paths Artifact path set.
  @param[in] metadata Expected row count, dimension and PQ layout.
  @param[out] error Optional validation failure message.

  @retval true All required artifacts exist and match expected headers.
  @retval false Missing file, invalid metadata or header mismatch.
*/
bool validate_diskann_pq_artifacts(
    const diskann_pq_artifact_paths &paths,
    const diskann_pq_artifact_metadata &metadata, std::string *error);

/**
  Write a complete DiskANN PQ artifact set.

  @param[in] paths Artifact path set.
  @param[in] metadata Expected row count, dimension and PQ layout.
  @param[in] payload Artifact payload vectors.
  @param[out] error Optional write failure message.

  @retval true All artifacts were written.
  @retval false Invalid payload or IO failure.
*/
bool write_diskann_pq_artifacts(
    const diskann_pq_artifact_paths &paths,
    const diskann_pq_artifact_metadata &metadata,
    const diskann_pq_artifact_payload &payload, std::string *error);

/**
  Write all native PQ artifacts except the compressed-code matrix.

  This is used by streaming builders that write @c paths.compressed_path through
  @c diskann_compressed_artifact_writer while still sharing validation and
  layout logic for pivots, centroid, chunk offsets and doc-id ordinals.
*/
bool write_diskann_pq_static_artifacts(
    const diskann_pq_artifact_paths &paths,
    const diskann_pq_artifact_metadata &metadata,
    const diskann_pq_artifact_payload &payload, std::string *error);

/**
  Read a float artifact and validate its header.

  @retval true Artifact was read into @c values.
  @retval false Missing file, mismatched header or truncated payload.
*/
bool read_diskann_pq_float_artifact(const std::string &path,
                                    uint32_t expected_rows,
                                    uint32_t expected_columns,
                                    std::vector<float> *values,
                                    std::string *error);

/**
  Reconstruct one source vector from its native PQ code.

  This diagnostic helper reads only the requested compressed-code row. The
  pivots, centroid and chunk offsets are validated through the common artifact
  layout before reconstruction.

  @param[in] paths Artifact path set.
  @param[in] metadata Expected artifact metadata.
  @param[in] ordinal Zero-based source row ordinal.
  @param[out] vector Reconstructed vector.
  @param[out] error Optional failure message.

  @retval true The vector was reconstructed.
  @retval false The artifacts or ordinal are invalid.
*/
bool reconstruct_diskann_pq_vector(const diskann_pq_artifact_paths &paths,
                                   const diskann_pq_artifact_metadata &metadata,
                                   uint64_t ordinal, std::vector<float> *vector,
                                   std::string *error);

/**
  Build a bridge manifest by checksumming artifacts on disk.

  @retval true Manifest was populated.
  @retval false Required artifacts are missing or unreadable.
*/
bool make_diskann_pq_bridge_manifest(
    const diskann_pq_artifact_paths &paths,
    const diskann_pq_artifact_metadata &metadata, uint32_t max_degree,
    uint32_t build_complexity, uint32_t cache_nodes, bool artifacts_consumed,
    bool official_pq_used, diskann_pq_bridge_manifest *manifest,
    std::string *error);

/** Write the bridge manifest atomically to @c paths.artifact_manifest_path. */
bool write_diskann_pq_bridge_manifest(
    const diskann_pq_artifact_paths &paths,
    const diskann_pq_bridge_manifest &manifest, std::string *error);

/** Read the bridge manifest from @c paths.artifact_manifest_path. */
bool read_diskann_pq_bridge_manifest(const diskann_pq_artifact_paths &paths,
                                     diskann_pq_bridge_manifest *manifest,
                                     std::string *error);

/** Validate manifest metadata and artifact checksums against current files. */
bool validate_diskann_pq_bridge_manifest(
    const diskann_pq_artifact_paths &paths,
    const diskann_pq_artifact_metadata &metadata,
    const diskann_pq_bridge_manifest &manifest, std::string *error);

}  // namespace vector_index

#endif  // SQL_VECTOR_DISKANN_ARTIFACT_LAYOUT_INCLUDED
