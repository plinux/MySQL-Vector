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

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "sql/vector/vector_load_file.h"

namespace vector_load_file_unittest {

namespace {

class temp_directory {
 public:
  temp_directory() {
    const auto root = std::filesystem::temp_directory_path();
    const auto name =
        "mysql-vector-load-file-" +
        std::to_string(reinterpret_cast<uintptr_t>(this));
    m_path = root / name;
    std::filesystem::create_directories(m_path);
  }

  ~temp_directory() {
    std::error_code ec;
    std::filesystem::remove_all(m_path, ec);
  }

  std::string path(const std::string &filename) const {
    return (m_path / filename).string();
  }

 private:
  std::filesystem::path m_path;
};

template <typename T>
void write_value(std::ofstream *file, T value) {
  file->write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_fbin_file(const std::string &path, uint32_t rows, uint32_t dimension,
                     const std::vector<float> &values) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  write_value(&file, rows);
  write_value(&file, dimension);
  file.write(reinterpret_cast<const char *>(values.data()),
             values.size() * sizeof(float));
  ASSERT_TRUE(file.good());
}

void write_docid_file(const std::string &path, const std::vector<uint64_t> &ids) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.good());
  write_value<uint64_t>(&file, ids.size());
  file.write(reinterpret_cast<const char *>(ids.data()),
             ids.size() * sizeof(uint64_t));
  ASSERT_TRUE(file.good());
}

void append_byte(const std::string &path) {
  std::ofstream file(path, std::ios::binary | std::ios::app);
  ASSERT_TRUE(file.good());
  const char extra = '\1';
  file.write(&extra, sizeof(extra));
  ASSERT_TRUE(file.good());
}

void write_text_file(const std::string &path, const std::string &content) {
  std::ofstream file(path, std::ios::trunc);
  ASSERT_TRUE(file.good());
  file << content;
  ASSERT_TRUE(file.good());
}

bool has_error_text(const std::string &error, const std::string &needle) {
  return error.find(needle) != std::string::npos;
}

struct loaded_csv_row {
  uint64_t doc_id;
  std::vector<float> values;
};

bool read_csv_rows(const std::string &path, size_t expected_dimension,
                   vector_index::vector_load_file_info *info,
                   std::string *error, std::vector<loaded_csv_row> *rows) {
  rows->clear();
  return vector_index::read_csv_vectors(
      path, expected_dimension, info, error,
      [rows](uint64_t doc_id, const float *values, size_t dimension) {
        rows->push_back(
            {doc_id, std::vector<float>(values, values + dimension)});
        return true;
      });
}

}  // namespace

TEST(VectorLoadFileTest, ReadsFbinRowsWithExplicitDocids) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  const std::string docid_path = tmp.path("docids.u64");
  write_fbin_file(fbin_path, 2, 3, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  write_docid_file(docid_path, {42, 99});

  vector_index::vector_load_file_info info;
  std::string error;
  std::vector<uint64_t> docids;
  std::vector<std::vector<float>> rows;

  const bool ok = vector_index::read_fbin_vectors(
      fbin_path, docid_path, 3, &info, &error,
      [&](uint64_t doc_id, const float *values, size_t dimension) {
        docids.push_back(doc_id);
        rows.emplace_back(values, values + dimension);
        return true;
      });

  EXPECT_TRUE(ok) << error;
  EXPECT_EQ(2U, info.row_count);
  EXPECT_EQ(3U, info.dimension);
  EXPECT_EQ((std::vector<uint64_t>{42, 99}), docids);
  ASSERT_EQ(2U, rows.size());
  EXPECT_EQ((std::vector<float>{1.0F, 2.0F, 3.0F}), rows[0]);
  EXPECT_EQ((std::vector<float>{4.0F, 5.0F, 6.0F}), rows[1]);
}

TEST(VectorLoadFileTest, ReadsFbinRowsInContiguousBlocks) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  const std::string docid_path = tmp.path("docids.u64");
  write_fbin_file(
      fbin_path, 5, 2,
      {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F});
  write_docid_file(docid_path, {10, 20, 30, 40, 50});

  vector_index::vector_load_file_info info;
  std::string error;
  std::vector<size_t> block_sizes;
  std::vector<uint64_t> docids;
  std::vector<float> values;

  const bool ok = vector_index::read_fbin_vector_blocks(
      fbin_path, docid_path, 2, 2, &info, &error,
      [&](const uint64_t *block_docids, const float *block_values,
          size_t row_count, size_t dimension) {
        block_sizes.push_back(row_count);
        docids.insert(docids.end(), block_docids, block_docids + row_count);
        values.insert(values.end(), block_values,
                      block_values + row_count * dimension);
        return true;
      });

  EXPECT_TRUE(ok) << error;
  EXPECT_EQ(5U, info.row_count);
  EXPECT_EQ(2U, info.dimension);
  EXPECT_EQ((std::vector<size_t>{2, 2, 1}), block_sizes);
  EXPECT_EQ((std::vector<uint64_t>{10, 20, 30, 40, 50}), docids);
  EXPECT_EQ((std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
                                9.0F, 10.0F}),
            values);
}

TEST(VectorLoadFileTest, BlockReaderGeneratesSequentialDocids) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  write_fbin_file(fbin_path, 3, 2, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

  std::string error;
  std::vector<uint64_t> docids;
  ASSERT_TRUE(vector_index::read_fbin_vector_blocks(
      fbin_path, "", 2, 2, nullptr, &error,
      [&](const uint64_t *block_docids,
          const float *block_values [[maybe_unused]], size_t row_count,
          size_t dimension) {
        EXPECT_EQ(2U, dimension);
        docids.insert(docids.end(), block_docids, block_docids + row_count);
        return true;
      }))
      << error;
  EXPECT_EQ((std::vector<uint64_t>{0, 1, 2}), docids);
}

TEST(VectorLoadFileTest, BlockReaderRejectsInvalidArgumentsAndVisitorStop) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  write_fbin_file(fbin_path, 2, 2, {1.0F, 2.0F, 3.0F, 4.0F});

  std::string error;
  EXPECT_FALSE(vector_index::read_fbin_vector_blocks(
      fbin_path, "", 2, 0, nullptr, &error,
      [](const uint64_t *, const float *, size_t, size_t) { return true; }));
  EXPECT_TRUE(has_error_text(error, "block rows")) << error;

  EXPECT_FALSE(vector_index::read_fbin_vector_blocks(fbin_path, "", 2, 1,
                                                     nullptr, &error, {}));
  EXPECT_TRUE(has_error_text(error, "visitor is required")) << error;

  size_t visits = 0;
  EXPECT_FALSE(vector_index::read_fbin_vector_blocks(
      fbin_path, "", 2, 1, nullptr, &error,
      [&](const uint64_t *, const float *, size_t, size_t) {
        ++visits;
        return false;
      }));
  EXPECT_EQ(1U, visits);
  EXPECT_TRUE(has_error_text(error, "visitor stopped")) << error;
}

TEST(VectorLoadFileTest, ReadsFbinFileInfoWithoutVisitingRows) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  write_fbin_file(fbin_path, 2, 3, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

  vector_index::vector_load_file_info info;
  std::string error;

  ASSERT_TRUE(vector_index::read_fbin_file_info(fbin_path, 3, &info, &error))
      << error;
  EXPECT_EQ(2U, info.row_count);
  EXPECT_EQ(3U, info.dimension);

  EXPECT_FALSE(vector_index::read_fbin_file_info(fbin_path, 4, &info, &error));
  EXPECT_TRUE(has_error_text(error, "dimension")) << error;
}

TEST(VectorLoadFileTest, ReadsFbinWithoutOptionalOutputs) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  write_fbin_file(fbin_path, 1, 2, {1.0F, 2.0F});

  size_t visits = 0;
  ASSERT_TRUE(vector_index::read_fbin_vectors(
      fbin_path, "", 0, nullptr, nullptr,
      [&visits](uint64_t doc_id, const float *values, size_t dimension) {
        EXPECT_EQ(0U, doc_id);
        EXPECT_EQ(2U, dimension);
        EXPECT_FLOAT_EQ(1.0F, values[0]);
        ++visits;
        return true;
      }));
  EXPECT_EQ(1U, visits);
}

TEST(VectorLoadFileTest, RejectsFbinFileInfoSizeMismatch) {
  temp_directory tmp;
  const std::string truncated_path = tmp.path("truncated.fbin");
  const std::string extra_path = tmp.path("extra.fbin");
  const std::string overflow_path = tmp.path("overflow.fbin");
  write_fbin_file(truncated_path, 2, 2, {1.0F, 2.0F, 3.0F});
  write_fbin_file(extra_path, 1, 2, {1.0F, 2.0F});
  append_byte(extra_path);
  {
    std::ofstream file(overflow_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file.good());
    write_value<uint32_t>(&file, std::numeric_limits<uint32_t>::max());
    write_value<uint32_t>(&file, std::numeric_limits<uint32_t>::max());
  }

  vector_index::vector_load_file_info info;
  std::string error;

  EXPECT_FALSE(
      vector_index::read_fbin_file_info(truncated_path, 2, &info, &error));
  EXPECT_TRUE(has_error_text(error, "size mismatch")) << error;

  EXPECT_FALSE(vector_index::read_fbin_file_info(extra_path, 2, &info, &error));
  EXPECT_TRUE(has_error_text(error, "size mismatch")) << error;

  EXPECT_FALSE(
      vector_index::read_fbin_file_info(overflow_path, 0, &info, &error));
  EXPECT_TRUE(has_error_text(error, "too large")) << error;
}

TEST(VectorLoadFileTest, RejectsFbinHeaderAndZeroDimension) {
  temp_directory tmp;
  const std::string truncated_header_path = tmp.path("truncated_header.fbin");
  const std::string truncated_dimension_path =
      tmp.path("truncated_dimension.fbin");
  const std::string zero_dimension_path = tmp.path("zero_dimension.fbin");
  write_text_file(truncated_header_path, "abc");
  {
    std::ofstream file(truncated_dimension_path,
                       std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file.good());
    write_value<uint32_t>(&file, 1);
  }
  write_fbin_file(zero_dimension_path, 1, 0, {});

  vector_index::vector_load_file_info info;
  std::string error;
  EXPECT_FALSE(vector_index::read_fbin_vectors(
      truncated_header_path, "", 0, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_TRUE(has_error_text(error, "truncated fbin header")) << error;

  EXPECT_FALSE(vector_index::read_fbin_vectors(
      truncated_dimension_path, "", 0, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_TRUE(has_error_text(error, "truncated fbin header")) << error;

  EXPECT_FALSE(vector_index::read_fbin_file_info(zero_dimension_path, 0, nullptr,
                                                 &error));
  EXPECT_TRUE(has_error_text(error, "dimension")) << error;
}

TEST(VectorLoadFileTest, RejectsFbinVisitorAndMissingFileWithNullError) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  write_fbin_file(fbin_path, 1, 2, {1.0F, 2.0F});

  std::string error;
  EXPECT_FALSE(
      vector_index::read_fbin_vectors(fbin_path, "", 2, nullptr, &error, {}));
  EXPECT_TRUE(has_error_text(error, "visitor is required")) << error;

  EXPECT_FALSE(vector_index::read_fbin_vectors(
      tmp.path("missing.fbin"), "", 2, nullptr, nullptr,
      [](uint64_t, const float *, size_t) { return true; }));
}

TEST(VectorLoadFileTest, StreamsDocidValues) {
  temp_directory tmp;
  const std::string docid_path = tmp.path("docids.u64");
  write_docid_file(docid_path, {42, 99});

  std::string error;
  std::vector<uint64_t> docids;
  ASSERT_TRUE(vector_index::read_docid_values(
      docid_path, 2, &error, [&docids](uint64_t doc_id) {
        docids.push_back(doc_id);
        return true;
      }))
      << error;
  EXPECT_EQ((std::vector<uint64_t>{42, 99}), docids);
}

TEST(VectorLoadFileTest, RejectsMissingDocidFileAndVisitor) {
  temp_directory tmp;
  const std::string missing_path = tmp.path("missing.u64");

  std::string error;
  EXPECT_FALSE(vector_index::read_docid_values(missing_path, 1, &error, {}));
  EXPECT_TRUE(has_error_text(error, "visitor is required")) << error;

  EXPECT_FALSE(vector_index::read_docid_values(
      missing_path, 1, &error, [](uint64_t) { return true; }));
  EXPECT_TRUE(has_error_text(error, "failed to open docid file")) << error;
}

TEST(VectorLoadFileTest, RejectsDocidStreamErrors) {
  temp_directory tmp;
  const std::string mismatch_path = tmp.path("mismatch.u64");
  const std::string extra_path = tmp.path("extra.u64");
  write_docid_file(mismatch_path, {7});
  write_docid_file(extra_path, {7});
  append_byte(extra_path);

  std::string error;
  EXPECT_FALSE(vector_index::read_docid_values(
      mismatch_path, 2, &error, [](uint64_t) { return true; }));
  EXPECT_TRUE(has_error_text(error, "row count mismatch")) << error;

  EXPECT_FALSE(vector_index::read_docid_values(
      extra_path, 1, &error, [](uint64_t) { return true; }));
  EXPECT_TRUE(has_error_text(error, "extra payload")) << error;

  EXPECT_FALSE(vector_index::read_docid_values(
      extra_path, 1, &error, [](uint64_t) { return false; }));
  EXPECT_TRUE(has_error_text(error, "visitor stopped docid load")) << error;
}

TEST(VectorLoadFileTest, GeneratesSequentialDocidsWhenDocidFileIsMissing) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  write_fbin_file(fbin_path, 3, 2, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

  vector_index::vector_load_file_info info;
  std::string error;
  std::vector<uint64_t> docids;

  const bool ok = vector_index::read_fbin_vectors(
      fbin_path, "", 2, &info, &error,
      [&](uint64_t doc_id, const float *values [[maybe_unused]],
          size_t dimension [[maybe_unused]]) {
        docids.push_back(doc_id);
        return true;
      });

  EXPECT_TRUE(ok) << error;
  EXPECT_EQ(3U, info.row_count);
  EXPECT_EQ((std::vector<uint64_t>{0, 1, 2}), docids);
}

TEST(VectorLoadFileTest, PreservesDuplicateDocidsForUpperLayerPolicy) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  const std::string docid_path = tmp.path("docids.u64");
  write_fbin_file(fbin_path, 2, 2, {1.0F, 2.0F, 3.0F, 4.0F});
  write_docid_file(docid_path, {7, 7});

  vector_index::vector_load_file_info info;
  std::string error;
  std::vector<uint64_t> docids;

  const bool ok = vector_index::read_fbin_vectors(
      fbin_path, docid_path, 2, &info, &error,
      [&](uint64_t doc_id, const float *values [[maybe_unused]],
          size_t dimension [[maybe_unused]]) {
        docids.push_back(doc_id);
        return true;
      });

  EXPECT_TRUE(ok) << error;
  EXPECT_EQ((std::vector<uint64_t>{7, 7}), docids);
}

TEST(VectorLoadFileTest, RejectsUnexpectedDimension) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  write_fbin_file(fbin_path, 1, 4, {1.0F, 2.0F, 3.0F, 4.0F});

  vector_index::vector_load_file_info info;
  std::string error;

  const bool ok = vector_index::read_fbin_vectors(
      fbin_path, "", 3, &info, &error,
      [](uint64_t, const float *, size_t) { return true; });

  EXPECT_FALSE(ok);
  EXPECT_TRUE(has_error_text(error, "dimension")) << error;
}

TEST(VectorLoadFileTest, RejectsTruncatedFbinPayload) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  write_fbin_file(fbin_path, 2, 2, {1.0F, 2.0F, 3.0F});

  vector_index::vector_load_file_info info;
  std::string error;

  const bool ok = vector_index::read_fbin_vectors(
      fbin_path, "", 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; });

  EXPECT_FALSE(ok);
  EXPECT_TRUE(has_error_text(error, "truncated")) << error;
}

TEST(VectorLoadFileTest, RejectsExtraFbinPayload) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  write_fbin_file(fbin_path, 1, 2, {1.0F, 2.0F});
  append_byte(fbin_path);

  vector_index::vector_load_file_info info;
  std::string error;

  const bool ok = vector_index::read_fbin_vectors(
      fbin_path, "", 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; });

  EXPECT_FALSE(ok);
  EXPECT_TRUE(has_error_text(error, "extra payload")) << error;
}

TEST(VectorLoadFileTest, RejectsDocidRowCountMismatch) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  const std::string docid_path = tmp.path("docids.u64");
  write_fbin_file(fbin_path, 2, 2, {1.0F, 2.0F, 3.0F, 4.0F});
  write_docid_file(docid_path, {7});

  vector_index::vector_load_file_info info;
  std::string error;

  const bool ok = vector_index::read_fbin_vectors(
      fbin_path, docid_path, 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; });

  EXPECT_FALSE(ok);
  EXPECT_TRUE(has_error_text(error, "docid")) << error;
}

TEST(VectorLoadFileTest, RejectsDocidPayloadErrorsDuringFbinLoad) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  const std::string truncated_docid_path = tmp.path("truncated_docids.u64");
  const std::string extra_docid_path = tmp.path("extra_docids.u64");
  write_fbin_file(fbin_path, 1, 2, {1.0F, 2.0F});
  {
    std::ofstream file(truncated_docid_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file.good());
    write_value<uint64_t>(&file, 1);
  }
  write_docid_file(extra_docid_path, {7});
  append_byte(extra_docid_path);

  vector_index::vector_load_file_info info;
  std::string error;
  EXPECT_FALSE(vector_index::read_fbin_vectors(
      fbin_path, truncated_docid_path, 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_TRUE(has_error_text(error, "truncated docid payload")) << error;

  EXPECT_FALSE(vector_index::read_fbin_vectors(
      fbin_path, extra_docid_path, 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_TRUE(has_error_text(error, "docid file has extra payload")) << error;
}

TEST(VectorLoadFileTest, RejectsNonFiniteVectorValues) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  write_fbin_file(fbin_path, 1, 2,
                  {1.0F, std::numeric_limits<float>::quiet_NaN()});

  vector_index::vector_load_file_info info;
  std::string error;

  const bool ok = vector_index::read_fbin_vectors(
      fbin_path, "", 2, &info, &error,
      [](uint64_t, const float *, size_t) { return true; });

  EXPECT_FALSE(ok);
  EXPECT_TRUE(has_error_text(error, "non-finite")) << error;
}

TEST(VectorLoadFileTest, StopsWhenVisitorReturnsFalse) {
  temp_directory tmp;
  const std::string fbin_path = tmp.path("vectors.fbin");
  write_fbin_file(fbin_path, 2, 2, {1.0F, 2.0F, 3.0F, 4.0F});

  vector_index::vector_load_file_info info;
  std::string error;
  size_t visits = 0;

  const bool ok = vector_index::read_fbin_vectors(
      fbin_path, "", 2, &info, &error,
      [&](uint64_t, const float *, size_t) {
        ++visits;
        return false;
      });

  EXPECT_FALSE(ok);
  EXPECT_EQ(1U, visits);
  EXPECT_TRUE(has_error_text(error, "visitor")) << error;
}

TEST(VectorLoadFileTest, ReadsCsvRowsWithHeaderQuotesWhitespaceAndCrLf) {
  temp_directory tmp;
  const std::string csv_path = tmp.path("vectors.csv");
  write_text_file(csv_path,
                  "doc_id,vector\r\n"
                  " 7 , \" [1.0, 2.5] \"\r\n"
                  "8,\"[3,4]\"\n");

  vector_index::vector_load_file_info info;
  std::string error;
  std::vector<loaded_csv_row> rows;

  ASSERT_TRUE(read_csv_rows(csv_path, 2, &info, &error, &rows)) << error;
  EXPECT_EQ(2U, info.row_count);
  EXPECT_EQ(2U, info.dimension);
  ASSERT_EQ(2U, rows.size());
  EXPECT_EQ(7U, rows[0].doc_id);
  EXPECT_EQ((std::vector<float>{1.0F, 2.5F}), rows[0].values);
  EXPECT_EQ(8U, rows[1].doc_id);
  EXPECT_EQ((std::vector<float>{3.0F, 4.0F}), rows[1].values);
}

TEST(VectorLoadFileTest, ReadsCsvWithoutOptionalOutputs) {
  temp_directory tmp;
  const std::string csv_path = tmp.path("vectors.csv");
  write_text_file(csv_path, "DOC_ID,VECTOR\n1,\"[1,2]\"\n2,\"[3, 4 ]\"\n");

  size_t visits = 0;
  ASSERT_TRUE(vector_index::read_csv_vectors(
      csv_path, 0, nullptr, nullptr,
      [&visits](uint64_t doc_id, const float *values, size_t dimension) {
        EXPECT_EQ(2U, dimension);
        if (doc_id == 1U) {
          EXPECT_FLOAT_EQ(2.0F, values[1]);
        } else {
          EXPECT_EQ(2U, doc_id);
          EXPECT_FLOAT_EQ(3.0F, values[0]);
          EXPECT_FLOAT_EQ(4.0F, values[1]);
        }
        ++visits;
        return true;
      }));
  EXPECT_EQ(2U, visits);
}

TEST(VectorLoadFileTest, RejectsCsvMalformedRowsAndVisitorErrors) {
  struct csv_error_case {
    const char *name;
    const char *content;
    size_t expected_dimension;
    const char *expected_error;
  };
  const csv_error_case cases[] = {
      {"missing_columns", "1\n", 0, "doc_id and vector"},
      {"same_length_bad_header", "doc_id,vectox\n", 0, "invalid CSV doc_id"},
      {"unterminated_quote", "1,\"[1,2]\n", 0, "unterminated"},
      {"negative_docid", "-1,\"[1,2]\"\n", 0, "invalid CSV doc_id"},
      {"empty_docid", " ,\"[1,2]\"\n", 0, "invalid CSV doc_id"},
      {"text_docid", "abc,\"[1,2]\"\n", 0, "invalid CSV doc_id"},
      {"trailing_docid", "12x,\"[1,2]\"\n", 0, "invalid CSV doc_id"},
      {"overflow_docid", "184467440737095516160,\"[1,2]\"\n", 0,
       "invalid CSV doc_id"},
      {"empty_vector_field", "1,\"   \"\n", 0, "invalid CSV vector"},
      {"missing_open_bracket", "1,\"1,2]\"\n", 0, "invalid CSV vector"},
      {"overflow_vector_value", "1,\"[1e999999]\"\n", 0,
       "invalid CSV vector value"},
      {"invalid_vector_value", "1,\"[1, nope]\"\n", 0,
       "invalid CSV vector value"},
      {"non_finite_vector_value", "1,\"[1, nan]\"\n", 0,
       "invalid CSV vector value"},
      {"trailing_after_close", "1,\"[1,2] trailing\"\n", 0,
       "invalid CSV vector"},
      {"missing_separator", "1,\"[1 2]\"\n", 0, "invalid CSV vector"},
      {"truncated_vector", "1,\"[1,2\"\n", 0, "invalid CSV vector"},
      {"empty_vector", "1,\"[]\"\n", 0, "dimension must be greater than zero"},
      {"expected_dimension_mismatch", "1,\"[1,2]\"\n", 3,
       "dimension mismatch"},
      {"row_dimension_mismatch", "1,\"[1,2]\"\n2,\"[1,2,3]\"\n", 0,
       "CSV vector dimension mismatch"},
      {"escaped_quote_in_vector", "1,\"[\"\"bad\"\"]\"\n", 0,
       "invalid CSV vector value"},
      {"header_only", "doc_id,vector\n\n", 0, "no vector rows"},
  };

  for (const csv_error_case &test_case : cases) {
    temp_directory tmp;
    const std::string csv_path = tmp.path(std::string(test_case.name) + ".csv");
    write_text_file(csv_path, test_case.content);

    vector_index::vector_load_file_info info;
    std::string error;
    std::vector<loaded_csv_row> rows;

    EXPECT_FALSE(read_csv_rows(csv_path, test_case.expected_dimension, &info,
                               &error, &rows))
        << test_case.name;
    EXPECT_TRUE(has_error_text(error, test_case.expected_error))
        << test_case.name << ": " << error;
  }

  temp_directory tmp;
  const std::string csv_path = tmp.path("visitor.csv");
  write_text_file(csv_path, "1,\"[1,2]\"\n");

  std::string error;
  EXPECT_FALSE(vector_index::read_csv_vectors(csv_path, 0, nullptr, &error, {}));
  EXPECT_TRUE(has_error_text(error, "visitor is required")) << error;

  EXPECT_FALSE(vector_index::read_csv_vectors(
      tmp.path("missing.csv"), 0, nullptr, &error,
      [](uint64_t, const float *, size_t) { return true; }));
  EXPECT_TRUE(has_error_text(error, "failed to open csv file")) << error;

  size_t visits = 0;
  EXPECT_FALSE(vector_index::read_csv_vectors(
      csv_path, 2, nullptr, &error,
      [&visits](uint64_t, const float *, size_t) {
        ++visits;
        return false;
      }));
  EXPECT_EQ(1U, visits);
  EXPECT_TRUE(has_error_text(error, "visitor stopped vector load")) << error;
}

}  // namespace vector_load_file_unittest
