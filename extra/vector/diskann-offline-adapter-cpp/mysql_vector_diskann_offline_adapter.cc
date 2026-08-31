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

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if __has_include(<dlfcn.h>)
#include <dlfcn.h>
#define MYSQL_VECTOR_HAVE_DLFCN 1
#endif

#include <fcntl.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#include <xmmintrin.h>
#endif

#include "aligned_file_reader.h"
#include "disk_utils.h"
#include "mysql_vector_diskann_offline_checked_io.h"
#include "pq_flash_index.h"
#include "utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if __has_include(<mkl.h>)
#include <mkl.h>
#define MYSQL_VECTOR_HAVE_MKL 1
#endif

namespace {

constexpr const char *kRawManifestHeader =
    "mysql-vector-diskann-raw-manifest-v1";
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kMaxPqTrainingSetSize = 256000.0;
constexpr uint64_t kMaxWarmupSampleRows = 100000;
#ifndef O_LARGEFILE
#define O_LARGEFILE 0
#endif

struct raw_segment {
  std::filesystem::path docid_path;
  std::filesystem::path vector_path;
  uint64_t row_count{0};
};

struct raw_manifest {
  uint32_t dimension{0};
  uint64_t count{0};
  std::vector<raw_segment> segments;
};

struct offline_index {
  std::shared_ptr<AlignedFileReader> reader;
  std::unique_ptr<diskann::PQFlashIndex<float>> index;
  std::vector<uint64_t> doc_ids;
  size_t public_dimension{0};
  uint32_t search_threads{1};
  uint32_t search_io_limit{0};
};

class scoped_directory_cleanup {
 public:
  explicit scoped_directory_cleanup(std::filesystem::path path)
      : m_path(std::move(path)) {}

  ~scoped_directory_cleanup() {
    std::error_code ec;
    std::filesystem::remove_all(m_path, ec);
  }

  scoped_directory_cleanup(const scoped_directory_cleanup &) = delete;
  scoped_directory_cleanup &operator=(const scoped_directory_cleanup &) =
      delete;

 private:
  std::filesystem::path m_path;
};

/*
  Keep MySQL process ownership over I/O errors. The official C++ Linux reader
  calls exit() on libaio failures, while this adapter must surface failures
  through the C ABI return code.
*/
class pread_aligned_file_reader final : public AlignedFileReader {
 public:
  pread_aligned_file_reader() = default;
  ~pread_aligned_file_reader() override { close(); }

  IOContext &get_ctx() override {
    std::lock_guard<std::mutex> guard(ctx_mut);
    auto iter = ctx_map.find(std::this_thread::get_id());
    if (iter == ctx_map.end()) return m_bad_ctx;
    return iter.value();
  }

  void register_thread() override {
    std::lock_guard<std::mutex> guard(ctx_mut);
    ctx_map.emplace(std::this_thread::get_id(), IOContext{});
  }

  void deregister_thread() override {
    std::lock_guard<std::mutex> guard(ctx_mut);
    ctx_map.erase(std::this_thread::get_id());
  }

  void deregister_all_threads() override {
    std::lock_guard<std::mutex> guard(ctx_mut);
    ctx_map.clear();
  }

  void open(const std::string &fname) override {
    close();
    m_file_desc = ::open(fname.c_str(), O_RDONLY | O_LARGEFILE);
    if (m_file_desc < 0) {
      throw std::runtime_error("failed to open DiskANN index file: " + fname);
    }
  }

  void close() override {
    if (m_file_desc >= 0) {
      ::close(m_file_desc);
      m_file_desc = -1;
    }
  }

  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
            bool async = false) override {
    (void)ctx;
    (void)async;
    if (m_file_desc < 0) {
      throw std::runtime_error("DiskANN index file is not open");
    }

    for (AlignedRead &request : read_reqs) {
      read_exact(request);
    }
  }

 private:
  void read_exact(const AlignedRead &request) const {
    auto *buffer = static_cast<char *>(request.buf);
    uint64_t remaining = request.len;
    uint64_t offset = request.offset;

    while (remaining > 0) {
      const size_t chunk = static_cast<size_t>(
          std::min<uint64_t>(remaining, static_cast<uint64_t>(SSIZE_MAX)));
      const ssize_t bytes_read = ::pread(m_file_desc, buffer, chunk,
                                         static_cast<off_t>(offset));
      if (bytes_read <= 0) {
        throw std::runtime_error("failed to read DiskANN index file");
      }
      buffer += bytes_read;
      offset += static_cast<uint64_t>(bytes_read);
      remaining -= static_cast<uint64_t>(bytes_read);
    }
  }

  int m_file_desc{-1};
  IOContext m_bad_ctx{reinterpret_cast<IOContext>(-1)};
};

class scoped_omp_threads {
 public:
  explicit scoped_omp_threads(uint32_t thread_count) {
#ifdef _OPENMP
    if (thread_count > 0) {
      m_active = true;
      m_previous = omp_get_max_threads();
      omp_set_num_threads(static_cast<int>(thread_count));
    }
#else
    (void)thread_count;
#endif
  }

  ~scoped_omp_threads() {
#ifdef _OPENMP
    if (m_active) omp_set_num_threads(m_previous);
#endif
  }

 private:
  bool m_active{false};
  int m_previous{0};
};

struct openblas_thread_api {
  using get_num_threads_func = int (*)();
  using set_num_threads_func = void (*)(int);

  get_num_threads_func get_num_threads{nullptr};
  set_num_threads_func set_num_threads{nullptr};
};

template <typename Func>
Func resolve_openblas_symbol(const char *name, const char *alternate_name,
                             const char *interface64_name) {
#ifdef MYSQL_VECTOR_HAVE_DLFCN
  if (void *symbol = dlsym(RTLD_DEFAULT, name); symbol != nullptr) {
    return reinterpret_cast<Func>(symbol);
  }
  if (alternate_name != nullptr) {
    if (void *symbol = dlsym(RTLD_DEFAULT, alternate_name); symbol != nullptr) {
      return reinterpret_cast<Func>(symbol);
    }
  }
  if (interface64_name != nullptr) {
    if (void *symbol = dlsym(RTLD_DEFAULT, interface64_name); symbol != nullptr) {
      return reinterpret_cast<Func>(symbol);
    }
  }
#else
  (void)name;
  (void)alternate_name;
  (void)interface64_name;
#endif
  return nullptr;
}

const openblas_thread_api &openblas_threads() {
  static const openblas_thread_api api{
      resolve_openblas_symbol<openblas_thread_api::get_num_threads_func>(
          "openblas_get_num_threads", "openblas_get_num_threads_",
          "openblas_get_num_threads64_"),
      resolve_openblas_symbol<openblas_thread_api::set_num_threads_func>(
          "openblas_set_num_threads", "openblas_set_num_threads_",
          "openblas_set_num_threads64_")};
  return api;
}

struct openblas_thread_control_state {
  uint32_t thread_count{0};
  int previous_thread_count{0};
  uint32_t active_users{0};
};

std::mutex &openblas_thread_control_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::condition_variable &openblas_thread_control_cv() {
  static std::condition_variable cv;
  return cv;
}

openblas_thread_control_state &openblas_threads_state() {
  static openblas_thread_control_state state;
  return state;
}

class scoped_openblas_threads {
 public:
  void activate(uint32_t thread_count) {
    const openblas_thread_api &openblas = openblas_threads();
    if (thread_count == 0 || openblas.get_num_threads == nullptr ||
        openblas.set_num_threads == nullptr) {
      return;
    }

    std::unique_lock<std::mutex> lock(openblas_thread_control_mutex());
    openblas_thread_control_state &state = openblas_threads_state();
    openblas_thread_control_cv().wait(lock, [&state, thread_count]() {
      return state.active_users == 0 || state.thread_count == thread_count;
    });

    if (state.active_users == 0) {
      state.thread_count = thread_count;
      state.previous_thread_count = openblas.get_num_threads();
      openblas.set_num_threads(static_cast<int>(thread_count));
    }
    ++state.active_users;
    m_active = true;
  }

  ~scoped_openblas_threads() {
    if (!m_active) return;

    const openblas_thread_api &openblas = openblas_threads();
    std::unique_lock<std::mutex> lock(openblas_thread_control_mutex());
    openblas_thread_control_state &state = openblas_threads_state();
    if (state.active_users > 0) --state.active_users;
    if (state.active_users == 0) {
      if (openblas.set_num_threads != nullptr) {
        openblas.set_num_threads(state.previous_thread_count);
      }
      state.thread_count = 0;
      state.previous_thread_count = 0;
      lock.unlock();
      openblas_thread_control_cv().notify_all();
    }
  }

 private:
  bool m_active{false};
};

class scoped_build_thread_controls {
 public:
  scoped_build_thread_controls(uint32_t build_threads,
                               uint32_t build_blas_threads) {
#ifdef _OPENMP
    m_omp_active = build_threads > 0;
    if (m_omp_active) {
      m_previous_omp_threads = omp_get_max_threads();
      omp_set_num_threads(static_cast<int>(build_threads));
    }
    m_previous_omp_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
#if _OPENMP >= 200805
    m_active_levels_supported = true;
    m_previous_active_levels = omp_get_max_active_levels();
    omp_set_max_active_levels(1);
#endif
    m_previous_nested = omp_get_nested();
    omp_set_nested(0);
#else
    (void)build_threads;
#endif
#ifdef MYSQL_VECTOR_HAVE_MKL
    if (build_blas_threads > 0) {
      m_mkl_active = true;
      m_previous_mkl_dynamic = mkl_get_dynamic();
      m_previous_mkl_threads =
          mkl_set_num_threads_local(static_cast<int>(build_blas_threads));
      mkl_set_dynamic(0);
    }
#endif
    m_openblas_threads.activate(build_blas_threads);
  }

  ~scoped_build_thread_controls() {
#ifdef MYSQL_VECTOR_HAVE_MKL
    if (m_mkl_active) {
      mkl_set_num_threads_local(m_previous_mkl_threads);
      mkl_set_dynamic(m_previous_mkl_dynamic);
    }
#endif
#ifdef _OPENMP
    omp_set_nested(m_previous_nested);
#if _OPENMP >= 200805
    if (m_active_levels_supported) {
      omp_set_max_active_levels(m_previous_active_levels);
    }
#endif
    omp_set_dynamic(m_previous_omp_dynamic);
    if (m_omp_active) omp_set_num_threads(m_previous_omp_threads);
#endif
  }

 private:
  bool m_omp_active{false};
  int m_previous_omp_threads{0};
  int m_previous_omp_dynamic{0};
  int m_previous_nested{0};
  bool m_active_levels_supported{false};
  int m_previous_active_levels{0};
  bool m_mkl_active{false};
  int m_previous_mkl_threads{0};
  int m_previous_mkl_dynamic{0};
  scoped_openblas_threads m_openblas_threads;
};

std::filesystem::path doc_ids_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() + "_mysql_vector_docids.bin");
}

std::filesystem::path work_dir_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() + "_mysql_vector_build");
}

uint32_t normalize_thread_count(uint32_t thread_count) {
  if (thread_count > 0) return thread_count;
  const unsigned int hardware_threads = std::thread::hardware_concurrency();
  return hardware_threads == 0 ? 1 : hardware_threads;
}

template <typename Func>
bool run_parallel_tasks(size_t task_count, uint32_t requested_threads,
                        Func &&func) {
  if (task_count == 0) return true;

  const size_t requested_worker_count =
      static_cast<size_t>(normalize_thread_count(requested_threads));
  const size_t worker_count = std::min(task_count, requested_worker_count);
  if (worker_count <= 1) {
    for (size_t task_idx = 0; task_idx < task_count; ++task_idx) func(task_idx);
    return true;
  }

  std::atomic<size_t> next_task{0};
  std::atomic_bool failed{false};
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  try {
    for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
      workers.emplace_back([&]() {
        scoped_omp_threads omp_scope(1);
        while (!failed.load(std::memory_order_relaxed)) {
          const size_t task_idx = next_task.fetch_add(1);
          if (task_idx >= task_count) break;
          try {
            func(task_idx);
          } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            break;
          }
        }
      });
    }
  } catch (...) {
    failed.store(true, std::memory_order_relaxed);
  }

  for (std::thread &worker : workers) {
    if (worker.joinable()) worker.join();
  }
  return !failed.load(std::memory_order_relaxed);
}

uint32_t copy_diskann_search_results(const offline_index &handle,
                                     const uint64_t *internal_ids,
                                     const float *internal_distances,
                                     uint32_t top_k, uint64_t *doc_ids,
                                     float *distances) {
  if (internal_ids == nullptr || internal_distances == nullptr ||
      doc_ids == nullptr || distances == nullptr) {
    return 0;
  }

  uint32_t result_count = 0;
  for (uint32_t i = 0; i < top_k; ++i) {
    const uint64_t offset = internal_ids[i];
    if (offset >= handle.doc_ids.size()) continue;
    doc_ids[result_count] = handle.doc_ids[static_cast<size_t>(offset)];
    distances[result_count] = internal_distances[i];
    ++result_count;
  }
  return result_count;
}

bool read_exact(std::ifstream *file, void *data, size_t length) {
  if (file == nullptr || data == nullptr) return false;
  if (length > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    return false;
  }
  file->read(reinterpret_cast<char *>(data),
             static_cast<std::streamsize>(length));
  return file->good();
}

bool write_exact(std::ofstream *file, const void *data, size_t length) {
  if (file == nullptr || data == nullptr) return false;
  if (length > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    return false;
  }
  file->write(reinterpret_cast<const char *>(data),
              static_cast<std::streamsize>(length));
  return file->good();
}

bool file_has_exact_size(const std::filesystem::path &path,
                         uintmax_t expected_size) {
  std::error_code error;
  return std::filesystem::file_size(path, error) == expected_size && !error;
}

template <typename T>
bool read_binary_value(std::ifstream *file, T *value) {
  return read_exact(file, value, sizeof(*value));
}

template <typename T>
bool write_binary_value(std::ofstream *file, T value) {
  return write_exact(file, &value, sizeof(value));
}

bool parse_u64(const std::string &text, uint64_t *value) {
  if (value == nullptr || text.empty()) return false;
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0') return false;
  *value = static_cast<uint64_t>(parsed);
  return true;
}

std::vector<std::string> split_tab_fields(const std::string &line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, '\t')) fields.push_back(field);
  if (!line.empty() && line.back() == '\t') fields.emplace_back();
  return fields;
}

bool parse_manifest(const char *manifest_path, raw_manifest *manifest) {
  if (manifest_path == nullptr || manifest == nullptr) return false;

  std::ifstream file(manifest_path);
  if (!file.is_open()) return false;

  std::string line;
  if (!std::getline(file, line) || line != kRawManifestHeader) return false;

  raw_manifest parsed;
  bool has_dimension = false;
  bool has_count = false;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> fields = split_tab_fields(line);
    if (fields.size() == 2 && fields[0] == "dimension") {
      uint64_t dimension = 0;
      if (!parse_u64(fields[1], &dimension) || dimension == 0 ||
          dimension > std::numeric_limits<uint32_t>::max()) {
        return false;
      }
      parsed.dimension = static_cast<uint32_t>(dimension);
      has_dimension = true;
      continue;
    }
    if (fields.size() == 2 && fields[0] == "count") {
      if (!parse_u64(fields[1], &parsed.count) || parsed.count == 0) {
        return false;
      }
      has_count = true;
      continue;
    }
    if (fields.size() == 4 && fields[0] == "segment") {
      raw_segment segment;
      segment.docid_path = fields[1];
      segment.vector_path = fields[2];
      if (segment.docid_path.empty() || segment.vector_path.empty() ||
          !parse_u64(fields[3], &segment.row_count) ||
          segment.row_count == 0) {
        return false;
      }
      parsed.segments.push_back(std::move(segment));
      continue;
    }
    return false;
  }

  uint64_t total_rows = 0;
  for (const raw_segment &segment : parsed.segments) {
    if (segment.row_count >
        std::numeric_limits<uint64_t>::max() - total_rows) {
      return false;
    }
    total_rows += segment.row_count;
  }
  if (!has_dimension || !has_count || parsed.segments.empty() ||
      total_rows != parsed.count) {
    return false;
  }

  *manifest = std::move(parsed);
  return true;
}

bool read_raw_header(std::ifstream *file, uint32_t *row_count,
                     uint32_t *dimension) {
  return read_binary_value(file, row_count) &&
         read_binary_value(file, dimension) && *row_count != 0 &&
         *dimension != 0;
}

bool copy_bytes(std::ifstream *source, std::ofstream *target, uint64_t bytes) {
  std::vector<char> buffer(4 * 1024 * 1024);
  while (bytes > 0) {
    const size_t chunk =
        static_cast<size_t>(std::min<uint64_t>(bytes, buffer.size()));
    if (!read_exact(source, buffer.data(), chunk)) return false;
    if (!write_exact(target, buffer.data(), chunk)) return false;
    bytes -= chunk;
  }
  return true;
}

bool merge_manifest_inputs(const raw_manifest &manifest,
                           const std::filesystem::path &merged_raw,
                           const std::filesystem::path &merged_doc_ids) {
  if (manifest.count > std::numeric_limits<uint32_t>::max()) return false;

  std::ofstream raw_file(merged_raw, std::ios::out | std::ios::binary |
                                         std::ios::trunc);
  std::ofstream docid_file(merged_doc_ids, std::ios::out | std::ios::binary |
                                               std::ios::trunc);
  if (!raw_file.is_open() || !docid_file.is_open()) return false;

  if (!write_binary_value(&raw_file, static_cast<uint32_t>(manifest.count)) ||
      !write_binary_value(&raw_file, manifest.dimension) ||
      !write_binary_value(&docid_file, manifest.count)) {
    return false;
  }

  const uint64_t row_bytes =
      static_cast<uint64_t>(manifest.dimension) * sizeof(float);
  for (const raw_segment &segment : manifest.segments) {
    if (segment.row_count > std::numeric_limits<uint32_t>::max()) return false;

    std::ifstream segment_raw(segment.vector_path,
                              std::ios::in | std::ios::binary);
    std::ifstream segment_doc_ids(segment.docid_path,
                                  std::ios::in | std::ios::binary);
    if (!segment_raw.is_open() || !segment_doc_ids.is_open()) return false;

    uint32_t segment_count = 0;
    uint32_t segment_dimension = 0;
    uint64_t docid_count = 0;
    if (!read_raw_header(&segment_raw, &segment_count, &segment_dimension) ||
        !read_binary_value(&segment_doc_ids, &docid_count) ||
        segment_count != segment.row_count ||
        docid_count != segment.row_count ||
        segment_dimension != manifest.dimension) {
      return false;
    }

    if (!copy_bytes(&segment_raw, &raw_file, segment.row_count * row_bytes) ||
        !copy_bytes(&segment_doc_ids, &docid_file,
                    segment.row_count * sizeof(uint64_t))) {
      return false;
    }
  }

  return raw_file.good() && docid_file.good();
}

bool normalize_fbin_for_cosine(const std::filesystem::path &input_path,
                               const std::filesystem::path &output_path,
                               uint64_t expected_rows,
                               uint32_t expected_dimension) {
  constexpr uint64_t kNormalizeBlockRows = 131072;
  if (expected_rows == 0 ||
      expected_rows > std::numeric_limits<uint32_t>::max() ||
      expected_dimension == 0) {
    return false;
  }

  std::ifstream input(input_path, std::ios::in | std::ios::binary);
  std::ofstream output(output_path,
                       std::ios::out | std::ios::binary | std::ios::trunc);
  uint32_t rows = 0;
  uint32_t dimension = 0;
  if (!input.is_open() || !output.is_open() ||
      !read_raw_header(&input, &rows, &dimension) || rows != expected_rows ||
      dimension != expected_dimension || !write_binary_value(&output, rows) ||
      !write_binary_value(&output, dimension)) {
    return false;
  }

  uint64_t remaining_rows = expected_rows;
  std::vector<float> block;
  while (remaining_rows != 0) {
    const uint64_t block_rows = std::min(remaining_rows, kNormalizeBlockRows);
    const uint64_t value_count = block_rows * expected_dimension;
    if (value_count > std::numeric_limits<size_t>::max() / sizeof(float) ||
        value_count >
            static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) /
                sizeof(float)) {
      return false;
    }
    block.resize(static_cast<size_t>(value_count));
    const size_t block_size = static_cast<size_t>(value_count) * sizeof(float);
    if (!read_exact(&input, block.data(), block_size) ||
        std::any_of(block.begin(), block.end(),
                    [](float value) { return !std::isfinite(value); })) {
      return false;
    }
    for (uint64_t row = 0; row < block_rows; ++row) {
      float *vector = block.data() + row * expected_dimension;
      float norm = std::numeric_limits<float>::epsilon();
      for (uint32_t dim = 0; dim < expected_dimension; ++dim) {
        norm += vector[dim] * vector[dim];
      }
      norm = std::sqrt(norm);
      for (uint32_t dim = 0; dim < expected_dimension; ++dim) {
        vector[dim] /= norm;
      }
    }
    if (!write_exact(&output, block.data(), block_size)) {
      return false;
    }
    remaining_rows -= block_rows;
  }

  char extra = 0;
  input.read(&extra, 1);
  return input.eof() && output.good();
}

bool validate_manifest_segment(const raw_manifest &manifest,
                               const raw_segment &segment) {
  if (segment.row_count > std::numeric_limits<uint32_t>::max()) return false;

  std::ifstream segment_raw(segment.vector_path,
                            std::ios::in | std::ios::binary);
  std::ifstream segment_doc_ids(segment.docid_path,
                                std::ios::in | std::ios::binary);
  if (!segment_raw.is_open() || !segment_doc_ids.is_open()) return false;

  uint32_t segment_count = 0;
  uint32_t segment_dimension = 0;
  uint64_t docid_count = 0;
  if (!read_raw_header(&segment_raw, &segment_count, &segment_dimension) ||
      !read_binary_value(&segment_doc_ids, &docid_count) ||
      segment_count != segment.row_count || docid_count != segment.row_count ||
      segment_dimension != manifest.dimension) {
    return false;
  }

  mysql_vector_diskann_offline::checked_payload_layout raw_layout;
  mysql_vector_diskann_offline::checked_payload_layout docid_layout;
  const uint64_t vector_values =
      static_cast<uint64_t>(segment_count) * segment_dimension;
  return mysql_vector_diskann_offline::make_checked_payload_layout(
             vector_values, sizeof(float), 2 * sizeof(uint32_t), &raw_layout) &&
         mysql_vector_diskann_offline::make_checked_payload_layout(
             docid_count, sizeof(uint64_t), sizeof(uint64_t), &docid_layout) &&
         file_has_exact_size(segment.vector_path, raw_layout.file_bytes) &&
         file_has_exact_size(segment.docid_path, docid_layout.file_bytes);
}

bool read_doc_ids(const std::filesystem::path &path, uint64_t expected_count,
                  std::vector<uint64_t> *doc_ids) {
  if (doc_ids == nullptr) return false;
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) return false;

  uint64_t count = 0;
  if (!read_binary_value(&file, &count) || count != expected_count ||
      count > static_cast<uint64_t>(doc_ids->max_size())) {
    return false;
  }

  mysql_vector_diskann_offline::checked_payload_layout layout;
  if (!mysql_vector_diskann_offline::make_checked_payload_layout(
          count, sizeof(uint64_t), sizeof(uint64_t), &layout) ||
      !file_has_exact_size(path, layout.file_bytes)) {
    return false;
  }

  doc_ids->assign(layout.element_count, 0);
  if (count == 0) return true;
  return read_exact(&file, doc_ids->data(), layout.payload_bytes);
}

diskann::Metric metric_from_code(int32_t metric_type) {
  switch (metric_type) {
    case 0:
      return diskann::COSINE;
    case 1:
      return diskann::INNER_PRODUCT;
    case 2:
      return diskann::L2;
    default:
      return diskann::L2;
  }
}

double pq_code_size_gb(uint32_t pq_chunks, uint64_t row_count) {
  if (pq_chunks == 0 || row_count == 0) return 0.0;
  const double budget_bytes =
      static_cast<double>(pq_chunks) * static_cast<double>(row_count);
  return std::nextafter(budget_bytes / kBytesPerGiB,
                        std::numeric_limits<double>::infinity());
}

double build_memory_size_gb(double requested_size_gb) {
  return std::isfinite(requested_size_gb) && requested_size_gb > 0.0
             ? requested_size_gb
             : 1.0;
}

std::string diskann_build_parameters(uint32_t max_degree,
                                     uint32_t search_list_size,
                                     uint32_t build_threads,
                                     double index_mem_gb,
                                     uint32_t pq_chunks,
                                     uint64_t row_count,
                                     uint32_t disk_pq_dims) {
  const uint32_t threads = normalize_thread_count(build_threads);
  const uint32_t append_reorder_data = 0;
  const uint32_t build_pq_bytes = 0;
  const double search_memory_size_gb = pq_code_size_gb(pq_chunks, row_count);
  std::ostringstream params;
  params << max_degree << ' ' << search_list_size << ' '
         << search_memory_size_gb << ' ' << build_memory_size_gb(index_mem_gb)
         << ' ' << threads << ' ' << disk_pq_dims << ' '
         << append_reorder_data << ' ' << build_pq_bytes << ' ' << pq_chunks;
  return params.str();
}

std::filesystem::path disk_index_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() + "_disk.index");
}

std::filesystem::path disk_index_pq_pivots_path(
    const std::filesystem::path &prefix) {
  return std::filesystem::path(disk_index_path(prefix).string() +
                               "_pq_pivots.bin");
}

std::filesystem::path disk_index_pq_compressed_path(
    const std::filesystem::path &prefix) {
  return std::filesystem::path(disk_index_path(prefix).string() +
                               "_pq_compressed.bin");
}

std::filesystem::path medoids_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() + "_disk.index_medoids.bin");
}

std::filesystem::path centroids_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() + "_disk.index_centroids.bin");
}

std::filesystem::path mem_index_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() + "_mem.index");
}

std::filesystem::path pq_pivots_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() + "_pq_pivots.bin");
}

std::filesystem::path pq_compressed_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() + "_pq_compressed.bin");
}

std::filesystem::path sample_data_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() + "_sample_data.bin");
}

std::filesystem::path build_options_path(const std::filesystem::path &prefix) {
  return std::filesystem::path(prefix.string() +
                               "_mysql_vector_build_options.txt");
}

bool ensure_parent_directory(const std::filesystem::path &path) {
  const std::filesystem::path parent = path.parent_path();
  if (parent.empty()) return true;
  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  return !ec;
}

bool copy_file_if_different(const std::filesystem::path &source,
                            const std::filesystem::path &target) {
  if (source.empty() || target.empty() || !std::filesystem::exists(source)) {
    return false;
  }
  if (source == target) return true;
  if (!ensure_parent_directory(target)) return false;
  std::error_code ec;
  std::filesystem::copy_file(source, target,
                             std::filesystem::copy_options::overwrite_existing,
                             ec);
  return !ec;
}

std::filesystem::path artifact_sidecar_path(const std::filesystem::path &path,
                                            const char *suffix) {
  return std::filesystem::path(path.string() + suffix);
}

bool validate_bin_shape(uint32_t rows, uint32_t columns,
                        uint32_t expected_rows,
                        uint32_t expected_columns) {
  return rows == expected_rows && columns == expected_columns;
}

template <typename T>
bool read_bin_artifact(const std::filesystem::path &path,
                       uint32_t expected_rows, uint32_t expected_columns,
                       std::vector<T> *values) {
  if (values == nullptr || path.empty()) return false;
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) return false;

  uint32_t rows = 0;
  uint32_t columns = 0;
  if (!read_binary_value(&file, &rows) ||
      !read_binary_value(&file, &columns) ||
      !validate_bin_shape(rows, columns, expected_rows, expected_columns)) {
    return false;
  }

  const uint64_t value_count =
      static_cast<uint64_t>(rows) * static_cast<uint64_t>(columns);
  mysql_vector_diskann_offline::checked_payload_layout layout;
  if (value_count > static_cast<uint64_t>(values->max_size()) ||
      !mysql_vector_diskann_offline::make_checked_payload_layout(
          value_count, sizeof(T), 2 * sizeof(uint32_t), &layout) ||
      !file_has_exact_size(path, layout.file_bytes)) {
    return false;
  }

  values->assign(layout.element_count, T{});
  if (value_count == 0) return true;
  return read_exact(&file, values->data(), layout.payload_bytes);
}

template <typename T>
bool write_bin_section(std::ofstream *file, uint64_t offset, uint32_t rows,
                       uint32_t columns, const std::vector<T> &values,
                       uint64_t *bytes_written) {
  if (file == nullptr || bytes_written == nullptr) return false;
  const uint64_t expected_values =
      static_cast<uint64_t>(rows) * static_cast<uint64_t>(columns);
  if (values.size() != expected_values ||
      offset > static_cast<uint64_t>(
                   std::numeric_limits<std::streamoff>::max())) {
    return false;
  }

  file->seekp(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!file->good() || !write_binary_value(file, rows) ||
      !write_binary_value(file, columns)) {
    return false;
  }
  if (!values.empty()) {
    if (values.size() >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) /
            sizeof(T)) {
      return false;
    }
    if (!write_exact(file, values.data(), values.size() * sizeof(T))) {
      return false;
    }
  }
  *bytes_written = sizeof(uint32_t) * 2 +
                   static_cast<uint64_t>(values.size()) * sizeof(T);
  return true;
}

bool compose_diskann_pq_pivots_file(
    const std::filesystem::path &source_pq_pivots_path,
    const std::filesystem::path &target_pq_pivots_path, uint32_t dimension,
    uint32_t pq_chunks) {
  if (dimension == 0 || pq_chunks == 0 ||
      !ensure_parent_directory(target_pq_pivots_path)) {
    return false;
  }

  std::vector<float> pivots;
  std::vector<float> centroid;
  std::vector<uint32_t> chunk_offsets;
  if (!read_bin_artifact(source_pq_pivots_path, 256, dimension, &pivots) ||
      !read_bin_artifact(
          artifact_sidecar_path(source_pq_pivots_path, "_centroid.bin"),
          dimension, 1, &centroid) ||
      !read_bin_artifact(
          artifact_sidecar_path(source_pq_pivots_path, "_chunk_offsets.bin"),
          pq_chunks + 1, 1, &chunk_offsets)) {
    return false;
  }

  std::ofstream output(target_pq_pivots_path,
                       std::ios::out | std::ios::binary | std::ios::trunc);
  if (!output.is_open()) return false;

  constexpr uint64_t metadata_size = 4096;
  std::vector<size_t> offsets(4, 0);
  offsets[0] = metadata_size;

  uint64_t bytes_written = 0;
  if (!write_bin_section(&output, offsets[0], 256, dimension, pivots,
                         &bytes_written)) {
    return false;
  }
  offsets[1] = offsets[0] + bytes_written;

  if (!write_bin_section(&output, offsets[1], dimension, 1, centroid,
                         &bytes_written)) {
    return false;
  }
  offsets[2] = offsets[1] + bytes_written;

  if (!write_bin_section(&output, offsets[2], pq_chunks + 1, 1, chunk_offsets,
                         &bytes_written)) {
    return false;
  }
  offsets[3] = offsets[2] + bytes_written;

  uint64_t metadata_bytes = 0;
  if (!write_bin_section(&output, 0, static_cast<uint32_t>(offsets.size()), 1,
                         offsets, &metadata_bytes)) {
    return false;
  }
  return output.good();
}

bool write_sample_data_file(const std::filesystem::path &data_path,
                            const std::filesystem::path &sample_path,
                            uint64_t row_count, uint32_t dimension) {
  if (row_count == 0 || dimension == 0) return false;
  std::ifstream input(data_path, std::ios::in | std::ios::binary);
  if (!input.is_open()) return false;

  uint32_t header_count = 0;
  uint32_t header_dimension = 0;
  if (!read_raw_header(&input, &header_count, &header_dimension) ||
      header_count != row_count || header_dimension != dimension) {
    return false;
  }

  const uint64_t ten_percent = (row_count + 9) / 10;
  const uint64_t sample_count =
      std::max<uint64_t>(1, std::min(kMaxWarmupSampleRows, ten_percent));
  const uint64_t step = std::max<uint64_t>(1, row_count / sample_count);
  if (!ensure_parent_directory(sample_path)) return false;

  std::ofstream output(sample_path, std::ios::out | std::ios::binary |
                                      std::ios::trunc);
  if (!output.is_open()) return false;

  const uint32_t sample_count_header = static_cast<uint32_t>(sample_count);
  if (!write_binary_value(&output, sample_count_header) ||
      !write_binary_value(&output, dimension)) {
    return false;
  }

  const size_t row_bytes = static_cast<size_t>(dimension) * sizeof(float);
  if (row_bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
    return false;
  std::vector<float> row(dimension);
  for (uint64_t sample = 0; sample < sample_count; ++sample) {
    const uint64_t row_id = std::min<uint64_t>(row_count - 1, sample * step);
    const std::streamoff offset =
        static_cast<std::streamoff>(sizeof(uint32_t) * 2 + row_id * row_bytes);
    input.seekg(offset, std::ios::beg);
    if (!input.good() || !read_exact(&input, row.data(), row_bytes)) {
      return false;
    }
    output.write(reinterpret_cast<const char *>(row.data()),
                 static_cast<std::streamsize>(row_bytes));
    if (!output.good()) return false;
  }
  return true;
}

bool read_disk_index_medoid(const std::filesystem::path &index_path,
                            uint32_t *medoid) {
  if (medoid == nullptr) return false;
  std::ifstream file(index_path, std::ios::in | std::ios::binary);
  if (!file.is_open()) return false;

  uint64_t expected_file_size = 0;
  uint32_t max_degree = 0;
  uint32_t medoid_on_file = 0;
  if (!read_binary_value(&file, &expected_file_size) ||
      !read_binary_value(&file, &max_degree) ||
      !read_binary_value(&file, &medoid_on_file)) {
    return false;
  }
  if (expected_file_size == 0 || max_degree == 0) return false;
  *medoid = medoid_on_file;
  return true;
}

bool write_single_medoid_file(const std::filesystem::path &path,
                              uint32_t medoid) {
  if (!ensure_parent_directory(path)) return false;
  std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.is_open()) return false;
  const uint32_t count = 1;
  const uint32_t dim = 1;
  return write_binary_value(&file, count) && write_binary_value(&file, dim) &&
         write_binary_value(&file, medoid);
}

bool ensure_medoids_artifact(const std::filesystem::path &index_prefix) {
  const std::filesystem::path medoids = medoids_path(index_prefix);
  std::error_code ec;
  if (std::filesystem::exists(medoids, ec) &&
      std::filesystem::file_size(medoids, ec) != 0 && !ec) {
    return true;
  }

  uint32_t medoid = 0;
  if (!read_disk_index_medoid(disk_index_path(index_prefix), &medoid)) {
    return false;
  }
  return write_single_medoid_file(medoids, medoid);
}

bool create_native_pq_disk_layout(
    const std::filesystem::path &index_prefix,
    const std::filesystem::path &data_path,
    const std::filesystem::path &mem_index,
    const std::filesystem::path &source_disk_pq_pivots_path,
    const std::filesystem::path &source_disk_pq_compressed_path,
    uint32_t dimension, uint32_t disk_pq_dims) {
  const std::filesystem::path disk_index = disk_index_path(index_prefix);
  if (disk_pq_dims == 0) {
    diskann::create_disk_layout<float>(data_path.string(), mem_index.string(),
                                       disk_index.string());
    return true;
  }

  if (source_disk_pq_pivots_path.empty() ||
      source_disk_pq_compressed_path.empty()) {
    return false;
  }

  const std::filesystem::path target_pivots =
      disk_index_pq_pivots_path(index_prefix);
  const std::filesystem::path target_compressed =
      disk_index_pq_compressed_path(index_prefix);
  if (!compose_diskann_pq_pivots_file(source_disk_pq_pivots_path,
                                      target_pivots, dimension,
                                      disk_pq_dims) ||
      !copy_file_if_different(source_disk_pq_compressed_path,
                              target_compressed)) {
    return false;
  }

  diskann::create_disk_layout<uint8_t>(target_compressed.string(),
                                       mem_index.string(),
                                       disk_index.string());
  std::error_code ec;
  std::filesystem::remove(target_compressed, ec);
  return true;
}

bool write_build_options(const std::filesystem::path &index_prefix,
                         uint32_t build_threads,
                         uint32_t build_blas_threads,
                         const char *build_source) {
  const std::filesystem::path path = build_options_path(index_prefix);
  if (build_source == nullptr || !ensure_parent_directory(path)) return false;
  std::ofstream file(path, std::ios::out | std::ios::trunc);
  if (!file.is_open()) return false;
  file << "build_threads=" << build_threads << '\n';
  file << "build_blas_threads=" << build_blas_threads << '\n';
  file << "build_source=" << build_source << '\n';
  return file.good();
}

void remove_mem_index_files(const std::filesystem::path &mem_index) {
  std::error_code ec;
  std::filesystem::remove(mem_index, ec);
  std::filesystem::remove(std::filesystem::path(mem_index.string() + ".data"),
                          ec);
  std::filesystem::remove(std::filesystem::path(mem_index.string() + ".tags"),
                          ec);
}

bool build_index_from_files(const std::filesystem::path &index_prefix,
                            const std::filesystem::path &data_path,
                            const std::filesystem::path &source_doc_ids_path,
                            uint64_t row_count, uint32_t dimension,
                            int32_t metric_type, uint32_t max_degree,
                            uint32_t search_list_size, uint32_t build_threads,
                            uint32_t build_blas_threads, double index_mem_gb,
                            uint32_t pq_chunks, uint32_t num_nodes_to_cache,
                            uint32_t disk_pq_dims, uint32_t accelerate_build,
                            uint32_t shuffle_build, const char *build_source) {
  if (row_count == 0 || dimension == 0 || max_degree == 0 ||
      search_list_size == 0 || pq_chunks == 0 ||
      !ensure_parent_directory(index_prefix)) {
    return false;
  }
  if (disk_pq_dims > dimension) return false;
  if (accelerate_build != 0 || shuffle_build != 0) return false;

  const std::filesystem::path final_doc_ids_path = doc_ids_path(index_prefix);
  if (!ensure_parent_directory(final_doc_ids_path)) return false;

  std::error_code ec;
  std::filesystem::remove(final_doc_ids_path, ec);
  std::filesystem::copy_file(source_doc_ids_path, final_doc_ids_path,
                             std::filesystem::copy_options::overwrite_existing,
                             ec);
  if (ec) return false;

  const std::string params = diskann_build_parameters(
      max_degree, search_list_size, build_threads, index_mem_gb, pq_chunks,
      row_count, disk_pq_dims);
  scoped_build_thread_controls thread_controls(build_threads,
                                               build_blas_threads);
  (void)dimension;
  (void)num_nodes_to_cache;
  if (diskann::build_disk_index<float>(
          data_path.string().c_str(), index_prefix.string().c_str(),
          params.c_str(), metric_from_code(metric_type)) != 0) {
    return false;
  }
  return write_build_options(index_prefix, build_threads, build_blas_threads,
                             build_source);
}

bool build_index_from_native_pq_files(
    const std::filesystem::path &index_prefix,
    const std::filesystem::path &data_path,
    const std::filesystem::path &source_doc_ids_path,
    const std::filesystem::path &source_pq_pivots_path,
    const std::filesystem::path &source_pq_compressed_path,
    const std::filesystem::path &source_disk_pq_pivots_path,
    const std::filesystem::path &source_disk_pq_compressed_path,
    uint64_t row_count,
    uint32_t dimension, int32_t metric_type, uint32_t max_degree,
    uint32_t search_list_size, uint32_t build_threads,
    uint32_t build_blas_threads, double index_mem_gb, uint32_t pq_chunks,
    uint32_t num_nodes_to_cache, uint32_t disk_pq_dims,
    uint32_t accelerate_build, uint32_t shuffle_build) {
  if (row_count == 0 || dimension == 0 || max_degree == 0 ||
      search_list_size == 0 || pq_chunks == 0 ||
      !ensure_parent_directory(index_prefix)) {
    return false;
  }
  const diskann::Metric metric = metric_from_code(metric_type);
  if ((metric != diskann::L2 && metric != diskann::COSINE) ||
      disk_pq_dims > dimension || accelerate_build != 0 || shuffle_build != 0) {
    return false;
  }

  const std::filesystem::path final_doc_ids_path = doc_ids_path(index_prefix);
  if (!copy_file_if_different(source_doc_ids_path, final_doc_ids_path) ||
      !compose_diskann_pq_pivots_file(source_pq_pivots_path,
                                      pq_pivots_path(index_prefix), dimension,
                                      pq_chunks) ||
      !copy_file_if_different(source_pq_compressed_path,
                              pq_compressed_path(index_prefix))) {
    return false;
  }

  const uint32_t threads = normalize_thread_count(build_threads);
  const double sample_rate =
      std::min(1.0, kMaxPqTrainingSetSize / static_cast<double>(row_count));
  const std::filesystem::path mem_index = mem_index_path(index_prefix);
  scoped_build_thread_controls thread_controls(build_threads,
                                               build_blas_threads);
  const int graph_result = diskann::build_merged_vamana_index<float, uint32_t>(
      data_path.string(), diskann::Metric::L2, search_list_size, max_degree,
      sample_rate, build_memory_size_gb(index_mem_gb), mem_index.string(),
      medoids_path(index_prefix).string(), centroids_path(index_prefix).string(),
      0, false, threads);
  if (graph_result != 0) return false;

  if (!create_native_pq_disk_layout(index_prefix, data_path, mem_index,
                                    source_disk_pq_pivots_path,
                                    source_disk_pq_compressed_path, dimension,
                                    disk_pq_dims)) {
    remove_mem_index_files(mem_index);
    return false;
  }
  remove_mem_index_files(mem_index);
  if (!ensure_medoids_artifact(index_prefix)) return false;

  if (num_nodes_to_cache != 0 &&
      !write_sample_data_file(data_path, sample_data_path(index_prefix),
                              row_count, dimension)) {
    return false;
  }
  return write_build_options(index_prefix, build_threads, build_blas_threads,
                             "native_pq_bridge");
}

bool build_from_manifest_impl(const char *index_prefix, const char *manifest_path,
                              uint32_t dimension, int32_t metric_type,
                              uint32_t max_degree,
                              uint32_t search_list_size,
                              uint32_t build_threads,
                              uint32_t build_blas_threads,
                              double index_mem_gb, uint32_t pq_chunks,
                              uint32_t num_nodes_to_cache,
                              uint32_t disk_pq_dims,
                              uint32_t accelerate_build,
                              uint32_t shuffle_build) {
  raw_manifest manifest;
  if (!parse_manifest(manifest_path, &manifest) ||
      manifest.dimension != dimension) {
    return false;
  }

  if (manifest.segments.size() == 1) {
    const raw_segment &segment = manifest.segments.front();
    if (!validate_manifest_segment(manifest, segment)) return false;
    return build_index_from_files(
        std::filesystem::path(index_prefix), segment.vector_path,
        segment.docid_path, manifest.count, dimension, metric_type, max_degree,
        search_list_size, build_threads, build_blas_threads, index_mem_gb,
        pq_chunks, num_nodes_to_cache, disk_pq_dims, accelerate_build,
        shuffle_build, "manifest_direct");
  }

  const std::filesystem::path prefix(index_prefix);
  const std::filesystem::path work_dir = work_dir_path(prefix);
  std::error_code ec;
  std::filesystem::remove_all(work_dir, ec);
  std::filesystem::create_directories(work_dir, ec);
  if (ec) return false;

  const std::filesystem::path merged_raw = work_dir / "base.fbin";
  const std::filesystem::path merged_doc_ids = work_dir / "docids.bin";
  bool ok = merge_manifest_inputs(manifest, merged_raw, merged_doc_ids);
  if (ok) {
    ok = build_index_from_files(
        prefix, merged_raw, merged_doc_ids, manifest.count, dimension,
        metric_type, max_degree, search_list_size, build_threads,
        build_blas_threads, index_mem_gb, pq_chunks, num_nodes_to_cache,
        disk_pq_dims, accelerate_build, shuffle_build, "manifest_merge");
  }

  std::filesystem::remove_all(work_dir, ec);
  return ok;
}

bool build_from_native_pq_manifest_impl(
    const char *index_prefix, const char *manifest_path,
    const char *pq_pivots_path_text, const char *pq_compressed_path_text,
    const char *disk_pq_pivots_path_text,
    const char *disk_pq_compressed_path_text,
    uint32_t dimension, int32_t metric_type, uint32_t max_degree,
    uint32_t search_list_size, uint32_t build_threads,
    uint32_t build_blas_threads, double index_mem_gb, uint32_t pq_chunks,
    uint32_t num_nodes_to_cache, uint32_t disk_pq_dims,
    uint32_t accelerate_build, uint32_t shuffle_build) {
  raw_manifest manifest;
  if (!parse_manifest(manifest_path, &manifest) ||
      manifest.dimension != dimension) {
    return false;
  }

  const std::filesystem::path source_pq_pivots(pq_pivots_path_text);
  const std::filesystem::path source_pq_compressed(pq_compressed_path_text);
  const std::filesystem::path source_disk_pq_pivots =
      disk_pq_pivots_path_text == nullptr
          ? std::filesystem::path{}
          : std::filesystem::path(disk_pq_pivots_path_text);
  const std::filesystem::path source_disk_pq_compressed =
      disk_pq_compressed_path_text == nullptr
          ? std::filesystem::path{}
          : std::filesystem::path(disk_pq_compressed_path_text);
  const std::filesystem::path prefix(index_prefix);
  const diskann::Metric metric = metric_from_code(metric_type);
  const bool normalize_input = metric == diskann::COSINE;
  if (metric != diskann::L2 && !normalize_input) return false;

  if (manifest.segments.size() == 1 && !normalize_input) {
    const raw_segment &segment = manifest.segments.front();
    if (!validate_manifest_segment(manifest, segment)) return false;
    return build_index_from_native_pq_files(
        prefix, segment.vector_path, segment.docid_path, source_pq_pivots,
        source_pq_compressed, source_disk_pq_pivots, source_disk_pq_compressed,
        manifest.count, dimension, metric_type, max_degree, search_list_size,
        build_threads, build_blas_threads, index_mem_gb, pq_chunks,
        num_nodes_to_cache, disk_pq_dims, accelerate_build, shuffle_build);
  }

  const std::filesystem::path work_dir = work_dir_path(prefix);
  const scoped_directory_cleanup work_dir_cleanup(work_dir);
  std::error_code ec;
  std::filesystem::remove_all(work_dir, ec);
  std::filesystem::create_directories(work_dir, ec);
  if (ec) return false;

  std::filesystem::path input_raw;
  std::filesystem::path input_doc_ids;
  bool ok = true;
  if (manifest.segments.size() == 1) {
    const raw_segment &segment = manifest.segments.front();
    ok = validate_manifest_segment(manifest, segment);
    input_raw = segment.vector_path;
    input_doc_ids = segment.docid_path;
  } else {
    input_raw = work_dir / "base.fbin";
    input_doc_ids = work_dir / "docids.bin";
    ok = merge_manifest_inputs(manifest, input_raw, input_doc_ids);
  }

  std::filesystem::path build_raw = input_raw;
  if (ok && normalize_input) {
    build_raw = work_dir / "base.cosine.fbin";
    ok = normalize_fbin_for_cosine(input_raw, build_raw, manifest.count,
                                   dimension);
  }
  if (ok) {
    ok = build_index_from_native_pq_files(
        prefix, build_raw, input_doc_ids, source_pq_pivots,
        source_pq_compressed, source_disk_pq_pivots, source_disk_pq_compressed,
        manifest.count, dimension, metric_type, max_degree, search_list_size,
        build_threads, build_blas_threads, index_mem_gb, pq_chunks,
        num_nodes_to_cache, disk_pq_dims, accelerate_build, shuffle_build);
  }

  return ok;
}

bool build_from_flat_arrays_impl(
    const char *index_prefix, const uint8_t *id_data, size_t id_length,
    size_t id_stride, const uint8_t *vector_data, size_t dimension,
    size_t vector_stride, size_t count, int32_t metric_type,
    uint32_t max_degree, uint32_t search_list_size, uint32_t build_threads,
    uint32_t build_blas_threads, double index_mem_gb, uint32_t pq_chunks,
    uint32_t num_nodes_to_cache, uint32_t disk_pq_dims,
    uint32_t accelerate_build, uint32_t shuffle_build) {
  if (index_prefix == nullptr || id_data == nullptr || vector_data == nullptr ||
      id_length < sizeof(uint64_t) || id_stride < id_length ||
      dimension == 0 || dimension > std::numeric_limits<uint32_t>::max() ||
      count == 0 || count > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const size_t row_bytes = dimension * sizeof(float);
  if (vector_stride < row_bytes) return false;

  const std::filesystem::path prefix(index_prefix);
  const std::filesystem::path work_dir = work_dir_path(prefix);
  std::error_code ec;
  std::filesystem::remove_all(work_dir, ec);
  std::filesystem::create_directories(work_dir, ec);
  if (ec) return false;

  const std::filesystem::path raw_path = work_dir / "base.fbin";
  const std::filesystem::path ids_path = work_dir / "docids.bin";
  std::ofstream raw_file(raw_path, std::ios::out | std::ios::binary |
                                       std::ios::trunc);
  std::ofstream ids_file(ids_path, std::ios::out | std::ios::binary |
                                     std::ios::trunc);
  bool ok = raw_file.is_open() && ids_file.is_open();
  if (ok) {
    ok = write_binary_value(&raw_file, static_cast<uint32_t>(count)) &&
         write_binary_value(&raw_file, static_cast<uint32_t>(dimension)) &&
         write_binary_value(&ids_file, static_cast<uint64_t>(count));
  }

  for (size_t i = 0; ok && i < count; ++i) {
    const uint8_t *id_ptr = id_data + i * id_stride;
    const uint8_t *vector_ptr = vector_data + i * vector_stride;
    ok = write_exact(&ids_file, id_ptr, sizeof(uint64_t)) &&
         write_exact(&raw_file, vector_ptr, row_bytes);
  }
  raw_file.close();
  ids_file.close();

  if (ok) {
    ok = build_index_from_files(
        prefix, raw_path, ids_path, count, static_cast<uint32_t>(dimension),
        metric_type, max_degree, search_list_size, build_threads,
        build_blas_threads, index_mem_gb, pq_chunks, num_nodes_to_cache,
        disk_pq_dims, accelerate_build, shuffle_build, "flat_arrays");
  }

  std::filesystem::remove_all(work_dir, ec);
  return ok;
}

std::vector<uint32_t> build_cache_nodes(
    const std::unique_ptr<diskann::PQFlashIndex<float>> &index,
    const std::string &index_prefix, uint32_t num_nodes_to_cache,
    uint32_t threads, uint32_t use_bfs_cache) {
  std::vector<uint32_t> nodes;
  if (index == nullptr || num_nodes_to_cache == 0) return nodes;

  const std::string sample_path = index_prefix + "_sample_data.bin";
  if (use_bfs_cache == 0 && std::filesystem::exists(sample_path)) {
    index->generate_cache_list_from_sample_queries(
        sample_path, 15, 6, num_nodes_to_cache, threads, nodes);
  }
  if (nodes.empty()) index->cache_bfs_levels(num_nodes_to_cache, nodes);
  return nodes;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) bool
mysql_vector_diskann_offline_build(
    const char *index_prefix, const uint8_t *id_data, size_t id_length,
    size_t id_stride, const uint8_t *vector_data, size_t dimension,
    size_t vector_stride, size_t count, int32_t metric_type,
    uint32_t max_degree, uint32_t search_list_size, uint32_t build_threads,
    double index_mem_gb, uint32_t pq_chunks, uint32_t num_nodes_to_cache,
    uint32_t build_blas_threads, uint32_t disk_pq_dims,
    uint32_t accelerate_build, uint32_t shuffle_build) {
  try {
    return build_from_flat_arrays_impl(
        index_prefix, id_data, id_length, id_stride, vector_data, dimension,
        vector_stride, count, metric_type, max_degree, search_list_size,
        build_threads, build_blas_threads, index_mem_gb, pq_chunks,
        num_nodes_to_cache, disk_pq_dims, accelerate_build, shuffle_build);
  } catch (...) {
    return false;
  }
}

extern "C" __attribute__((visibility("default"))) bool
mysql_vector_diskann_offline_build_from_manifest(
    const char *index_prefix, const char *manifest_path, uint32_t dimension,
    int32_t metric_type, uint32_t max_degree, uint32_t search_list_size,
    uint32_t build_threads, double index_mem_gb, uint32_t pq_chunks,
    uint32_t num_nodes_to_cache, uint32_t build_blas_threads,
    uint32_t disk_pq_dims, uint32_t accelerate_build,
    uint32_t shuffle_build) {
  try {
    if (index_prefix == nullptr || manifest_path == nullptr) return false;
    return build_from_manifest_impl(index_prefix, manifest_path, dimension,
                                    metric_type, max_degree, search_list_size,
                                    build_threads, build_blas_threads,
                                    index_mem_gb, pq_chunks, num_nodes_to_cache,
                                    disk_pq_dims, accelerate_build,
                                    shuffle_build);
  } catch (...) {
    return false;
  }
}

extern "C" __attribute__((visibility("default"))) bool
mysql_vector_diskann_offline_build_from_native_pq(
    const char *index_prefix, const char *manifest_path,
    const char *pq_pivots_path, const char *pq_compressed_path,
    const char *disk_pq_pivots_path, const char *disk_pq_compressed_path,
    uint32_t dimension, int32_t metric_type, uint32_t max_degree,
    uint32_t search_list_size, uint32_t build_threads, double index_mem_gb,
    uint32_t pq_chunks, uint32_t num_nodes_to_cache,
    uint32_t build_blas_threads, uint32_t disk_pq_dims,
    uint32_t accelerate_build, uint32_t shuffle_build) {
  try {
    if (index_prefix == nullptr || manifest_path == nullptr ||
        pq_pivots_path == nullptr || pq_compressed_path == nullptr ||
        (disk_pq_dims != 0 &&
         (disk_pq_pivots_path == nullptr ||
          disk_pq_compressed_path == nullptr))) {
      return false;
    }
    return build_from_native_pq_manifest_impl(
        index_prefix, manifest_path, pq_pivots_path, pq_compressed_path,
        disk_pq_pivots_path, disk_pq_compressed_path, dimension, metric_type,
        max_degree, search_list_size, build_threads, build_blas_threads,
        index_mem_gb, pq_chunks, num_nodes_to_cache, disk_pq_dims,
        accelerate_build, shuffle_build);
  } catch (...) {
    return false;
  }
}

extern "C" __attribute__((visibility("default"))) const void *
mysql_vector_diskann_offline_load(const char *index_prefix,
                                  int32_t metric_type,
                                  uint32_t search_threads,
                                  uint32_t search_io_limit,
                                  uint32_t num_nodes_to_cache,
                                  uint32_t use_bfs_cache) {
  try {
    if (index_prefix == nullptr) return nullptr;

    auto handle = std::make_unique<offline_index>();
    handle->reader = std::make_shared<pread_aligned_file_reader>();
    handle->index = std::make_unique<diskann::PQFlashIndex<float>>(
        handle->reader, metric_from_code(metric_type));

    const uint32_t threads = normalize_thread_count(search_threads);
    handle->search_threads = threads;
    handle->search_io_limit = search_io_limit;
    if (handle->index->load(threads, index_prefix) != 0) return nullptr;
    const size_t loaded_dimension = handle->index->get_data_dim();
    handle->public_dimension =
        metric_type == 1 && loaded_dimension > 0 ? loaded_dimension - 1
                                                 : loaded_dimension;

    if (!read_doc_ids(doc_ids_path(index_prefix), handle->index->get_num_points(),
                      &handle->doc_ids)) {
      return nullptr;
    }

    if (num_nodes_to_cache > 0) {
      std::vector<uint32_t> nodes =
          build_cache_nodes(handle->index, index_prefix, num_nodes_to_cache,
                            threads, use_bfs_cache);
      if (!nodes.empty()) handle->index->load_cache_list(nodes);
    }

    return handle.release();
  } catch (...) {
    return nullptr;
  }
}

extern "C" __attribute__((visibility("default"))) int32_t
mysql_vector_diskann_offline_search(const void *index_ptr,
                                    const float *query, size_t dimension,
                                    uint32_t top_k,
                                    uint32_t search_list_size,
                                    uint32_t beam_width,
                                    uint64_t *doc_ids, float *distances) {
  try {
    if (index_ptr == nullptr || query == nullptr || doc_ids == nullptr ||
        distances == nullptr || dimension == 0 || top_k == 0) {
      return -1;
    }

    const auto *handle = static_cast<const offline_index *>(index_ptr);
    if (handle->index == nullptr ||
        dimension != handle->public_dimension || search_list_size < top_k ||
        top_k > handle->doc_ids.size() ||
        top_k > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      return -1;
    }

    std::vector<uint64_t> internal_ids(
        top_k, std::numeric_limits<uint64_t>::max());
    std::vector<float> internal_distances(top_k, 0.0F);
    const uint32_t io_limit = handle->search_io_limit == 0
                                  ? std::numeric_limits<uint32_t>::max()
                                  : handle->search_io_limit;
    handle->index->cached_beam_search(
        query, top_k, search_list_size, internal_ids.data(),
        internal_distances.data(), beam_width, io_limit);

    return static_cast<int32_t>(copy_diskann_search_results(
        *handle, internal_ids.data(), internal_distances.data(), top_k, doc_ids,
        distances));
  } catch (...) {
    return -1;
  }
}

extern "C" __attribute__((visibility("default"))) int32_t
mysql_vector_diskann_offline_search_batch(
    const void *index_ptr, const float *queries, size_t query_count,
    size_t dimension, uint32_t top_k, uint32_t search_list_size,
    uint32_t beam_width, uint32_t search_threads, uint64_t *doc_ids,
    float *distances, uint32_t *result_counts) {
  try {
    if (index_ptr == nullptr || queries == nullptr || doc_ids == nullptr ||
        distances == nullptr || result_counts == nullptr || query_count == 0 ||
        dimension == 0 || top_k == 0) {
      return -1;
    }

    const auto *handle = static_cast<const offline_index *>(index_ptr);
    if (handle->index == nullptr || dimension != handle->public_dimension ||
        search_list_size < top_k || top_k > handle->doc_ids.size() ||
        top_k > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      return -1;
    }
    if (query_count > std::numeric_limits<size_t>::max() / dimension ||
        query_count > std::numeric_limits<size_t>::max() / top_k) {
      return -1;
    }

    const size_t result_slots = query_count * top_k;
    std::vector<uint64_t> internal_ids(
        result_slots, std::numeric_limits<uint64_t>::max());
    std::vector<float> internal_distances(result_slots, 0.0F);

    const auto search_one = [&](size_t query_idx) {
      const size_t output_offset = query_idx * top_k;
      const float *query = queries + query_idx * dimension;
      const uint32_t io_limit = handle->search_io_limit == 0
                                    ? std::numeric_limits<uint32_t>::max()
                                    : handle->search_io_limit;
      handle->index->cached_beam_search(
          query, top_k, search_list_size, internal_ids.data() + output_offset,
          internal_distances.data() + output_offset, beam_width, io_limit);
    };

    if (!run_parallel_tasks(query_count, search_threads, search_one))
      return -1;

    for (size_t query_idx = 0; query_idx < query_count; ++query_idx) {
      const size_t output_offset = query_idx * top_k;
      result_counts[query_idx] = copy_diskann_search_results(
          *handle, internal_ids.data() + output_offset,
          internal_distances.data() + output_offset, top_k,
          doc_ids + output_offset, distances + output_offset);
    }
    return static_cast<int32_t>(query_count);
  } catch (...) {
    return -1;
  }
}

extern "C" __attribute__((visibility("default"))) uint64_t
mysql_vector_diskann_offline_card(const void *index_ptr) {
  if (index_ptr == nullptr) return 0;
  const auto *handle = static_cast<const offline_index *>(index_ptr);
  return static_cast<uint64_t>(handle->doc_ids.size());
}

extern "C" __attribute__((visibility("default"))) void
mysql_vector_diskann_offline_drop(const void *index_ptr) {
  delete static_cast<const offline_index *>(index_ptr);
}
