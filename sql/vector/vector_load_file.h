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

#ifndef SQL_VECTOR_VECTOR_LOAD_FILE_H
#define SQL_VECTOR_VECTOR_LOAD_FILE_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace vector_index {

using vector_load_visitor =
    std::function<bool(uint64_t doc_id, const float *values, size_t dimension)>;
using vector_load_block_visitor =
    std::function<bool(const uint64_t *doc_ids, const float *values,
                       size_t row_count, size_t dimension)>;
using vector_docid_visitor = std::function<bool(uint64_t doc_id)>;

struct vector_load_file_info {
  uint64_t row_count{0};
  size_t dimension{0};
};

/**
  Stream vectors from a DiskANN-style FBIN file and optional docid file.

  @param vector_filename FBIN file path. The file layout is uint32 rows,
         uint32 dimension, followed by rows * dimension little-endian float32
         values.
  @param docid_filename Optional docid file path. When non-empty, the layout is
         uint64 rows followed by rows little-endian uint64 doc_id values. When
         empty, doc_id values are generated as 0..rows-1.
  @param expected_dimension Non-zero dimension expected by the target index.
         Pass 0 to accept the file dimension.
  @param info Output file metadata. May be null.
  @param error Output diagnostic for user-facing validation failures. May be
         null.
  @param visitor Called once per row. Returning false stops the load and reports
         a visitor failure.

  @retval true All rows were read and visited.
  @retval false The file is invalid, the visitor stopped, or IO failed.
*/
bool read_fbin_vectors(const std::string &vector_filename,
                       const std::string &docid_filename,
                       size_t expected_dimension,
                       vector_load_file_info *info, std::string *error,
                       const vector_load_visitor &visitor);

/**
  Stream vectors from an FBIN file in bounded contiguous blocks.

  The vector payload is row-major. The doc_ids and values pointers remain valid
  only for the duration of each visitor call.

  @param vector_filename FBIN file path.
  @param docid_filename Optional docid file path. Sequential doc_ids are
         generated when the path is empty.
  @param expected_dimension Non-zero dimension expected by the target index.
  @param block_rows Maximum rows passed to one visitor call. Must be non-zero.
  @param info Output file metadata. May be null.
  @param error Output diagnostic. May be null.
  @param visitor Called once per block.

  @retval true All blocks were read and visited.
  @retval false The arguments or file are invalid, the visitor stopped, or IO
          failed.
*/
bool read_fbin_vector_blocks(const std::string &vector_filename,
                             const std::string &docid_filename,
                             size_t expected_dimension, size_t block_rows,
                             vector_load_file_info *info, std::string *error,
                             const vector_load_block_visitor &visitor);

/**
  Read and validate only the FBIN metadata.

  This validates the header, dimension and file size without reading each vector
  payload value. It is used by raw file-native build paths that hand the FBIN
  directly to the ANN backend.

  @retval true The file header and size are valid.
  @retval false The file is invalid or cannot be read.
*/
bool read_fbin_file_info(const std::string &vector_filename,
                         size_t expected_dimension,
                         vector_load_file_info *info, std::string *error);

/**
  Stream doc_id values from a standalone docid file.

  @retval true All doc_ids were visited and the file has no extra payload.
  @retval false The file is invalid, visitor stopped, or IO failed.
*/
bool read_docid_values(const std::string &docid_filename,
                       uint64_t expected_rows, std::string *error,
                       const vector_docid_visitor &visitor);

/**
  Stream vectors from a CSV file.

  The supported public layout is two columns per row:
  doc_id,vector
  1,"[0.1,0.2,0.3]"

  The header row is optional. The vector field may be quoted using standard
  CSV double-quote escaping.

  @param filename CSV file path.
  @param expected_dimension Non-zero dimension expected by the target index.
         Pass 0 to accept the row dimension.
  @param info Output file metadata. May be null.
  @param error Output diagnostic for user-facing validation failures. May be
         null.
  @param visitor Called once per data row.

  @retval true All rows were read and visited.
  @retval false The file is invalid, the visitor stopped, or IO failed.
*/
bool read_csv_vectors(const std::string &filename, size_t expected_dimension,
                      vector_load_file_info *info, std::string *error,
                      const vector_load_visitor &visitor);

}  // namespace vector_index

#endif  // SQL_VECTOR_VECTOR_LOAD_FILE_H
