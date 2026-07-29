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

#include "sql/vector/vector_load_file.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "my_byteorder.h"

namespace vector_index {

namespace {

constexpr size_t kDefaultFbinBlockRows = 4096;

void set_error(std::string *error, const std::string &message) {
  if (error != nullptr) *error = message;
}

template <typename UInt>
bool read_le_integer(std::ifstream *file, UInt *value) {
  static_assert(std::is_unsigned<UInt>::value,
                "read_le_integer expects an unsigned integer type");
  std::array<unsigned char, sizeof(UInt)> bytes{};
  file->read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  if (!*file) return false;

  UInt result = 0;
  for (size_t i = 0; i < bytes.size(); ++i) {
    result |= static_cast<UInt>(bytes[i]) << (i * 8);
  }
  *value = result;
  return true;
}

bool has_extra_payload(std::ifstream *file) {
  file->peek();
  return !file->eof();
}

bool read_fbin_header(std::ifstream *file, uint32_t *rows, uint32_t *dimension,
                      std::string *error) {
  if (!read_le_integer(file, rows) || !read_le_integer(file, dimension)) {
    set_error(error, "truncated fbin header");
    return false;
  }
  if (*dimension == 0) {
    set_error(error, "dimension must be greater than zero");
    return false;
  }
  return true;
}

bool validate_dimension(uint32_t file_dimension, size_t expected_dimension,
                        std::string *error) {
  if (expected_dimension != 0 && expected_dimension != file_dimension) {
    std::ostringstream message;
    message << "dimension mismatch: expected " << expected_dimension
            << ", file has " << file_dimension;
    set_error(error, message.str());
    return false;
  }
  if (file_dimension >
      static_cast<uint32_t>(std::vector<float>().max_size())) {
    set_error(error, "dimension is too large");
    return false;
  }
  return true;
}

bool open_binary_file(const std::string &filename, std::ifstream *file,
                      const char *description, std::string *error) {
  file->open(filename, std::ios::binary);
  if (file->good()) return true;

  std::ostringstream message;
  message << "failed to open " << description << " file: " << filename;
  set_error(error, message.str());
  return false;
}

bool read_docid_header(std::ifstream *file, uint64_t expected_rows,
                       std::string *error) {
  uint64_t docid_rows = 0;
  if (!read_le_integer(file, &docid_rows)) {
    set_error(error, "truncated docid header");
    return false;
  }
  if (docid_rows != expected_rows) {
    std::ostringstream message;
    message << "docid row count mismatch: expected " << expected_rows
            << ", file has " << docid_rows;
    set_error(error, message.str());
    return false;
  }
  return true;
}

bool expected_fbin_file_size(uint32_t rows, uint32_t dimension,
                             uintmax_t *bytes) {
  constexpr uintmax_t header_bytes = sizeof(uint32_t) + sizeof(uint32_t);
  if (rows > (std::numeric_limits<uintmax_t>::max() / dimension) /
                 sizeof(float)) {
    return false;
  }
  const uintmax_t payload_bytes =
      static_cast<uintmax_t>(rows) * dimension * sizeof(float);
  if (payload_bytes > std::numeric_limits<uintmax_t>::max() - header_bytes) {
    return false;
  }
  *bytes = header_bytes + payload_bytes;
  return true;
}

bool validate_file_size(const std::string &filename, uintmax_t expected_size,
                        const char *description, std::string *error) {
  std::error_code ec;
  const uintmax_t actual_size = std::filesystem::file_size(filename, ec);
  if (ec) {
    std::ostringstream message;
    message << "failed to stat " << description << " file: " << filename;
    set_error(error, message.str());
    return false;
  }
  if (actual_size != expected_size) {
    std::ostringstream message;
    message << description << " file size mismatch";
    set_error(error, message.str());
    return false;
  }
  return true;
}

bool read_row_docid(std::ifstream *file, uint64_t fallback_doc_id,
                    uint64_t *doc_id, std::string *error) {
  if (file == nullptr) {
    *doc_id = fallback_doc_id;
    return true;
  }
  if (!read_le_integer(file, doc_id)) {
    set_error(error, "truncated docid payload");
    return false;
  }
  return true;
}

bool read_block_values(std::ifstream *file, size_t value_count,
                       std::vector<float> *values, std::string *error) {
  if (value_count > values->max_size()) {
    set_error(error, "fbin block is too large");
    return false;
  }
  if (value_count >
      static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) /
          sizeof(float)) {
    set_error(error, "fbin block is too large");
    return false;
  }

  values->resize(value_count);
  const auto byte_count =
      static_cast<std::streamsize>(value_count * sizeof(float));
  file->read(reinterpret_cast<char *>(values->data()), byte_count);
  if (!*file) {
    set_error(error, "truncated fbin payload");
    return false;
  }

  for (float &value : *values) {
    value = float4get(reinterpret_cast<const unsigned char *>(&value));
    if (!std::isfinite(value)) {
      set_error(error, "non-finite vector value is not allowed");
      return false;
    }
  }
  return true;
}

bool read_block_docids(std::ifstream *file, uint64_t first_fallback_doc_id,
                       size_t row_count, std::vector<uint64_t> *doc_ids,
                       std::string *error) {
  doc_ids->resize(row_count);
  for (size_t row = 0; row < row_count; ++row) {
    if (!read_row_docid(file, first_fallback_doc_id + row, &(*doc_ids)[row],
                        error)) {
      return false;
    }
  }
  return true;
}

bool is_space(char value) {
  return std::isspace(static_cast<unsigned char>(value)) != 0;
}

std::string trim_copy(const std::string &value) {
  size_t begin = 0;
  while (begin < value.size() && is_space(value[begin])) ++begin;

  size_t end = value.size();
  while (end > begin && is_space(value[end - 1])) --end;
  return value.substr(begin, end - begin);
}

bool equals_ascii_no_case(const std::string &lhs, const char *rhs) {
  size_t rhs_len = 0;
  for (const char *ptr = rhs; *ptr != '\0'; ++ptr) ++rhs_len;
  if (lhs.size() != rhs_len) return false;
  for (size_t idx = 0; idx < lhs.size(); ++idx) {
    const char lhs_char = static_cast<char>(
        std::toupper(static_cast<unsigned char>(lhs[idx])));
    const char rhs_char = static_cast<char>(
        std::toupper(static_cast<unsigned char>(rhs[idx])));
    if (lhs_char != rhs_char) return false;
  }
  return true;
}

bool split_csv_record(const std::string &line, std::vector<std::string> *fields,
                      std::string *error) {
  fields->clear();
  std::string field;
  bool in_quotes = false;

  for (size_t idx = 0; idx < line.size(); ++idx) {
    const char value = line[idx];
    if (value == '"') {
      if (in_quotes && idx + 1 < line.size() && line[idx + 1] == '"') {
        field.push_back('"');
        ++idx;
        continue;
      }
      in_quotes = !in_quotes;
      continue;
    }
    if (value == ',' && !in_quotes) {
      fields->push_back(field);
      field.clear();
      continue;
    }
    field.push_back(value);
  }

  if (in_quotes) {
    set_error(error, "unterminated quoted CSV field");
    return false;
  }
  fields->push_back(field);
  return true;
}

bool parse_csv_doc_id(const std::string &field, uint64_t *doc_id,
                      std::string *error) {
  const std::string trimmed = trim_copy(field);
  if (trimmed.empty() || trimmed[0] == '-') {
    set_error(error, "invalid CSV doc_id");
    return false;
  }

  errno = 0;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(trimmed.c_str(), &end, 10);
  if (errno != 0 || end == trimmed.c_str() || *end != '\0') {
    set_error(error, "invalid CSV doc_id");
    return false;
  }

  *doc_id = static_cast<uint64_t>(value);
  return true;
}

bool parse_csv_vector_field(const std::string &field,
                            std::vector<float> *vector,
                            std::string *error) {
  vector->clear();
  const std::string text = trim_copy(field);
  const char *ptr = text.c_str();
  const char *end = ptr + text.size();

  if (ptr == end || *ptr != '[') {
    set_error(error, "invalid CSV vector");
    return false;
  }
  ++ptr;

  while (ptr < end) {
    while (ptr < end && is_space(*ptr)) ++ptr;
    if (ptr < end && *ptr == ']') {
      ++ptr;
      while (ptr < end && is_space(*ptr)) ++ptr;
      if (ptr != end) {
        set_error(error, "invalid CSV vector");
        return false;
      }
      return true;
    }

    errno = 0;
    char *value_end = nullptr;
    const float value = std::strtof(ptr, &value_end);
    if (errno != 0 || value_end == ptr || !std::isfinite(value)) {
      set_error(error, "invalid CSV vector value");
      return false;
    }
    vector->push_back(value);
    ptr = value_end;

    while (ptr < end && is_space(*ptr)) ++ptr;
    if (ptr == end) {
      set_error(error, "invalid CSV vector");
      return false;
    }
    if (*ptr == ',') {
      ++ptr;
      continue;
    }
    if (*ptr == ']') continue;

    set_error(error, "invalid CSV vector");
    return false;
  }

  set_error(error, "invalid CSV vector");
  return false;
}

}  // namespace

bool read_fbin_vectors(const std::string &vector_filename,
                       const std::string &docid_filename,
                       size_t expected_dimension, vector_load_file_info *info,
                       std::string *error,
                       const vector_load_visitor &visitor) {
  if (error != nullptr) error->clear();
  if (!visitor) {
    set_error(error, "visitor is required");
    return false;
  }

  return read_fbin_vector_blocks(
      vector_filename, docid_filename, expected_dimension,
      kDefaultFbinBlockRows, info, error,
      [&visitor](const uint64_t *doc_ids, const float *values, size_t row_count,
                 size_t dimension) {
        for (size_t row = 0; row < row_count; ++row) {
          if (!visitor(doc_ids[row], values + row * dimension, dimension))
            return false;
        }
        return true;
      });
}

bool read_fbin_vector_blocks(const std::string &vector_filename,
                             const std::string &docid_filename,
                             size_t expected_dimension, size_t block_rows,
                             vector_load_file_info *info, std::string *error,
                             const vector_load_block_visitor &visitor) {
  if (error != nullptr) error->clear();
  if (block_rows == 0) {
    set_error(error, "block rows must be greater than zero");
    return false;
  }
  if (!visitor) {
    set_error(error, "visitor is required");
    return false;
  }

  std::ifstream vector_file;
  if (!open_binary_file(vector_filename, &vector_file, "fbin", error))
    return false;

  uint32_t rows = 0;
  uint32_t dimension = 0;
  if (!read_fbin_header(&vector_file, &rows, &dimension, error) ||
      !validate_dimension(dimension, expected_dimension, error)) {
    return false;
  }

  std::ifstream docid_file;
  std::ifstream *docid_stream = nullptr;
  if (!docid_filename.empty()) {
    if (!open_binary_file(docid_filename, &docid_file, "docid", error) ||
        !read_docid_header(&docid_file, rows, error)) {
      return false;
    }
    docid_stream = &docid_file;
  }

  if (info != nullptr) {
    info->row_count = rows;
    info->dimension = dimension;
  }

  const size_t effective_block_rows =
      std::min(block_rows, static_cast<size_t>(rows));
  if (effective_block_rows != 0 &&
      effective_block_rows > std::vector<float>().max_size() / dimension) {
    set_error(error, "fbin block is too large");
    return false;
  }

  std::vector<float> block_values;
  std::vector<uint64_t> block_docids;
  uint64_t first_row = 0;
  while (first_row < rows) {
    const size_t current_rows = static_cast<size_t>(std::min<uint64_t>(
        effective_block_rows, static_cast<uint64_t>(rows) - first_row));
    if (!read_block_values(&vector_file, current_rows * dimension,
                           &block_values, error) ||
        !read_block_docids(docid_stream, first_row, current_rows, &block_docids,
                           error)) {
      return false;
    }

    if (!visitor(block_docids.data(), block_values.data(), current_rows,
                 dimension)) {
      set_error(error, "visitor stopped vector load");
      return false;
    }
    first_row += current_rows;
  }

  if (has_extra_payload(&vector_file)) {
    set_error(error, "fbin file has extra payload");
    return false;
  }
  if (docid_stream != nullptr && has_extra_payload(docid_stream)) {
    set_error(error, "docid file has extra payload");
    return false;
  }

  return true;
}

bool read_fbin_file_info(const std::string &vector_filename,
                         size_t expected_dimension,
                         vector_load_file_info *info, std::string *error) {
  if (error != nullptr) error->clear();

  std::ifstream vector_file;
  if (!open_binary_file(vector_filename, &vector_file, "fbin", error))
    return false;

  uint32_t rows = 0;
  uint32_t dimension = 0;
  if (!read_fbin_header(&vector_file, &rows, &dimension, error) ||
      !validate_dimension(dimension, expected_dimension, error)) {
    return false;
  }

  uintmax_t expected_size = 0;
  if (!expected_fbin_file_size(rows, dimension, &expected_size)) {
    set_error(error, "fbin file is too large");
    return false;
  }
  if (!validate_file_size(vector_filename, expected_size, "fbin", error)) {
    return false;
  }

  if (info != nullptr) {
    info->row_count = rows;
    info->dimension = dimension;
  }
  return true;
}

bool read_docid_values(const std::string &docid_filename,
                       uint64_t expected_rows, std::string *error,
                       const vector_docid_visitor &visitor) {
  if (error != nullptr) error->clear();
  if (!visitor) {
    set_error(error, "visitor is required");
    return false;
  }

  std::ifstream docid_file;
  if (!open_binary_file(docid_filename, &docid_file, "docid", error) ||
      !read_docid_header(&docid_file, expected_rows, error)) {
    return false;
  }

  for (uint64_t row = 0; row < expected_rows; ++row) {
    uint64_t doc_id = 0;
    if (!read_le_integer(&docid_file, &doc_id)) {
      set_error(error, "truncated docid payload");
      return false;
    }
    if (!visitor(doc_id)) {
      set_error(error, "visitor stopped docid load");
      return false;
    }
  }

  if (has_extra_payload(&docid_file)) {
    set_error(error, "docid file has extra payload");
    return false;
  }
  return true;
}

bool read_csv_vectors(const std::string &filename, size_t expected_dimension,
                      vector_load_file_info *info, std::string *error,
                      const vector_load_visitor &visitor) {
  if (error != nullptr) error->clear();
  if (!visitor) {
    set_error(error, "visitor is required");
    return false;
  }

  std::ifstream file;
  if (!open_binary_file(filename, &file, "csv", error)) return false;

  uint64_t row_count = 0;
  size_t file_dimension = 0;
  std::string line;
  std::vector<std::string> fields;
  std::vector<float> row;
  bool first_record = true;

  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (trim_copy(line).empty()) continue;

    if (!split_csv_record(line, &fields, error)) return false;
    if (fields.size() != 2) {
      set_error(error, "CSV row must contain doc_id and vector columns");
      return false;
    }

    if (first_record &&
        equals_ascii_no_case(trim_copy(fields[0]), "doc_id") &&
        equals_ascii_no_case(trim_copy(fields[1]), "vector")) {
      first_record = false;
      continue;
    }
    first_record = false;

    uint64_t doc_id = 0;
    if (!parse_csv_doc_id(fields[0], &doc_id, error) ||
        !parse_csv_vector_field(fields[1], &row, error)) {
      return false;
    }
    if (row.empty()) {
      set_error(error, "CSV vector dimension must be greater than zero");
      return false;
    }
    if (expected_dimension != 0 && row.size() != expected_dimension) {
      std::ostringstream message;
      message << "dimension mismatch: expected " << expected_dimension
              << ", file has " << row.size();
      set_error(error, message.str());
      return false;
    }
    if (file_dimension == 0) {
      file_dimension = row.size();
    } else if (file_dimension != row.size()) {
      set_error(error, "CSV vector dimension mismatch");
      return false;
    }

    if (!visitor(doc_id, row.data(), row.size())) {
      set_error(error, "visitor stopped vector load");
      return false;
    }
    ++row_count;
  }

  if (file.bad()) {
    set_error(error, "failed to read CSV file");
    return false;
  }
  if (row_count == 0) {
    set_error(error, "CSV file contains no vector rows");
    return false;
  }

  if (info != nullptr) {
    info->row_count = row_count;
    info->dimension = file_dimension;
  }
  return true;
}

}  // namespace vector_index
