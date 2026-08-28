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

#ifndef SQL_VECTOR_DISKANN_PQ_RUNTIME_INCLUDED
#define SQL_VECTOR_DISKANN_PQ_RUNTIME_INCLUDED

#include "sql/vector/vector_diskann_artifact_layout.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vector_index {

/** Inputs used to bound the native DiskANN PQ training working set. */
struct diskann_pq_memory_plan_input {
  uint64_t rows{0};
  uint64_t training_rows{0};
  uint64_t memory_budget_size{0};
  uint32_t dimension{0};
  uint32_t pq_chunks{0};
  uint32_t requested_threads{0};
};

/** Effective worker and block limits selected for native PQ training. */
struct diskann_pq_memory_plan {
  uint64_t encode_block_rows{0};
  uint64_t estimated_size{0};
  uint32_t worker_count{0};
  std::string adjustment;
};

/**
  Select a bounded native PQ training plan.

  The policy first reduces worker parallelism, then shrinks encode blocks. It
  fails when the minimum legal working set still exceeds the configured budget.

  @retval true A valid worker and block plan was selected.
  @retval false Inputs overflowed or the minimum working set exceeds the budget.
*/
bool make_diskann_pq_memory_plan(const diskann_pq_memory_plan_input &input,
                                 diskann_pq_memory_plan *plan,
                                 std::string *error);

/** Runtime knobs used by the native DiskANN PQ/kmeans training path. */
struct diskann_pq_runtime_config {
  uint32_t dimension{0};
  uint32_t pq_chunks{0};
  uint32_t disk_pq_chunks{0};
  uint32_t threads{0};
  uint64_t memory_budget_size{0};
  uint64_t training_row_limit{0};
  bool prefer_avx512{true};
  bool zero_mean{true};
  bool normalize_input{false};
};

/** Observable result from the native DiskANN PQ/kmeans training path. */
struct diskann_pq_runtime_result {
  diskann_pq_artifact_paths artifacts;
  diskann_pq_artifact_paths disk_artifacts;
  uint64_t row_count{0};
  uint64_t elapsed_ms{0};
  uint64_t raw_reader_ms{0};
  uint64_t distance_calls{0};
  uint64_t skipped_distance_calls{0};
  uint64_t wait_cycles_hint{0};
  uint64_t train_rows{0};
  uint64_t compressed_rows{0};
  uint64_t encode_block_rows{0};
  uint64_t memory_estimate_bytes{0};
  uint64_t memory_budget_size{0};
  uint32_t effective_threads{0};
  uint32_t pq_chunks{0};
  uint32_t centroid_count{0};
  bool artifacts_written{false};
  bool disk_artifacts_written{false};
  std::string selected_path;
  std::string memory_budget_adjustment;
};

/**
  Run the native DiskANN PQ/kmeans runtime training path.

  The current implementation validates input, trains PQ chunks with the native
  Elkan kmeans path, compresses each row into uint8 PQ codes and writes the
  native DiskANN PQ artifact set. A later build stage decides whether those
  artifacts can replace the official DiskANN build output.

  @retval true Training completed and runtime statistics were populated.
  @retval false Invalid input, inaccessible file or unsupported configuration.
*/
bool build_diskann_pq_runtime(const diskann_pq_runtime_config &config,
                              const char *input_file, const char *output_prefix,
                              diskann_pq_runtime_result *result,
                              std::string *error);

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool read_diskann_fbin_header_for_testing(const std::string &input_file,
                                          uint32_t expected_dimension,
                                          uint64_t *row_count,
                                          std::string *error);
bool for_each_diskann_fbin_block_for_testing(
    const std::string &input_file, uint32_t expected_dimension,
    uint64_t expected_row_count, uint64_t block_rows, bool visitor_result,
    uint64_t *visited_rows, std::string *error);
bool read_diskann_raw_training_sample_for_testing(const std::string &input_file,
                                                  uint32_t expected_dimension,
                                                  uint64_t row_count,
                                                  uint64_t sample_rows,
                                                  std::vector<float> *sample,
                                                  std::string *error);
bool inspect_diskann_raw_input_rows_for_testing(const std::string &input_file,
                                                uint32_t expected_dimension,
                                                uint64_t *row_count,
                                                std::string *error);
#endif  // EXTRA_CODE_FOR_UNIT_TESTING

}  // namespace vector_index

#endif  // SQL_VECTOR_DISKANN_PQ_RUNTIME_INCLUDED
