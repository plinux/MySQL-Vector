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

#ifndef SQL_VECTOR_ELKAN_KMEANS_INCLUDED
#define SQL_VECTOR_ELKAN_KMEANS_INCLUDED

#include <cstdint>
#include <string>
#include <vector>

namespace vector_index {

/** Configuration for deterministic Elkan kmeans training. */
struct elkan_kmeans_config {
  uint32_t dimension{0};
  uint32_t center_count{256};
  uint32_t max_iterations{20};
  uint32_t threads{1};
  uint64_t seed{0};
  bool zero_mean{true};
};

/** Observable output and pruning statistics from Elkan kmeans training. */
struct elkan_kmeans_result {
  std::vector<float> centers;
  std::vector<uint32_t> assignments;
  std::vector<float> centroid;
  uint64_t distance_calls{0};
  uint64_t packed_distance_calls{0};
  uint64_t skipped_distance_calls{0};
  std::string centroid_scan_kernel{"not_run"};
  uint32_t iterations{0};
  bool converged{false};
};

/**
  Run deterministic Elkan kmeans over a row-major float matrix.

  @param data Row-major input matrix.
  @param row_count Number of rows in @c data.
  @param config Kmeans configuration.
  @param result Output centers, assignments and diagnostics.
  @param error Optional error output.
  @retval true Training completed.
  @retval false Invalid input or unsupported configuration.
*/
bool run_elkan_kmeans(const float *data, uint64_t row_count,
                      const elkan_kmeans_config &config,
                      elkan_kmeans_result *result, std::string *error);

}  // namespace vector_index

#endif  // SQL_VECTOR_ELKAN_KMEANS_INCLUDED
