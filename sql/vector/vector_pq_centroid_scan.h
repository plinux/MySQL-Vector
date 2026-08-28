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
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA */

#ifndef SQL_VECTOR_PQ_CENTROID_SCAN_INCLUDED
#define SQL_VECTOR_PQ_CENTROID_SCAN_INCLUDED

#ifdef HAVE_VECTOR_INDEX

#include <cstdint>
#include <string>
#include <vector>

namespace vector_index {

/** Execution kernel selected for a packed PQ centroid model. */
enum class pq_centroid_scan_kernel { kAuto, kScalar, kAvx2, kAvx512 };

/** Dimension-major centroid layout used by the Native PQ scan loop. */
struct pq_centroid_model {
  uint32_t dimension{0};
  uint32_t center_count{0};
  uint32_t center_stride{0};
  pq_centroid_scan_kernel kernel{pq_centroid_scan_kernel::kScalar};
  std::vector<float> packed_centers;
};

/** Per-model centroid scan diagnostics. */
struct pq_centroid_scan_stats {
  pq_centroid_scan_kernel kernel{pq_centroid_scan_kernel::kScalar};
  uint64_t distance_evaluations{0};
};

/** Return whether the current CPU can execute a requested scan kernel. */
bool pq_centroid_scan_kernel_supported(pq_centroid_scan_kernel kernel);

/** Return the stable status name for a centroid scan kernel. */
const char *pq_centroid_scan_kernel_name(pq_centroid_scan_kernel kernel);

/**
  Pack row-major centroids and select the scan kernel once.

  @param centers Row-major center matrix.
  @param center_count Number of centers.
  @param dimension Number of float values in each center.
  @param requested_kernel Requested kernel or @c kAuto.
  @param model Output packed model.
  @param error Optional error text.
  @retval true Model was created.
  @retval false Input is invalid or the requested kernel is unavailable.
*/
bool make_pq_centroid_model(const float *centers, uint32_t center_count,
                            uint32_t dimension,
                            pq_centroid_scan_kernel requested_kernel,
                            pq_centroid_model *model, std::string *error);

/**
  Scan all packed centers and return the nearest center.

  Squared distances are optional. Equal distances preserve the lowest center
  id so scalar and SIMD kernels produce deterministic PQ codes.

  @param model Packed centroid model.
  @param values Query vector.
  @param squared_distances Optional center_count-sized output buffer.
  @param nearest_center Optional nearest center output.
  @param nearest_squared_distance Optional nearest squared distance output.
  @param stats Optional scan diagnostics.
  @retval true Scan completed.
  @retval false Model or input is invalid.
*/
bool scan_pq_centroids(const pq_centroid_model &model, const float *values,
                       float *squared_distances, uint32_t *nearest_center,
                       float *nearest_squared_distance,
                       pq_centroid_scan_stats *stats = nullptr);

/**
  Scan all packed centers and return Euclidean distances.

  This variant applies square root inside the selected SIMD kernel so Elkan
  lower bounds do not need a separate scalar conversion pass.

  @param model Packed centroid model.
  @param values Query vector.
  @param distances Optional center_count-sized output buffer.
  @param nearest_center Optional nearest center output.
  @param nearest_distance Optional nearest Euclidean distance output.
  @param stats Optional scan diagnostics.
  @retval true Scan completed.
  @retval false Model or input is invalid.
*/
bool scan_pq_centroid_distances(const pq_centroid_model &model,
                                const float *values, float *distances,
                                uint32_t *nearest_center,
                                float *nearest_distance,
                                pq_centroid_scan_stats *stats = nullptr);

}  // namespace vector_index

#endif  // HAVE_VECTOR_INDEX

#endif  // SQL_VECTOR_PQ_CENTROID_SCAN_INCLUDED
