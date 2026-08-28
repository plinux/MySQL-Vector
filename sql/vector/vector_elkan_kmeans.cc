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

#include "sql/vector/vector_elkan_kmeans.h"

#include "sql/vector/vector_pq_centroid_scan.h"
#include "sql/vector/vector_simd_distance.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace vector_index {
namespace {

using center_counts = std::vector<uint64_t>;

void set_error(std::string *error, const char *message) {
  if (error != nullptr) *error = message == nullptr ? "" : message;
}

bool multiply_size(uint64_t lhs, uint32_t rhs, size_t *out) {
  if (out == nullptr) return false;
  if (lhs > std::numeric_limits<size_t>::max() / rhs) return false;
  *out = static_cast<size_t>(lhs) * static_cast<size_t>(rhs);
  return true;
}

float euclidean_distance(const l2_distance_context &distance_context,
                         const float *lhs, const float *rhs,
                         l2_distance_stats *stats) {
  return std::sqrt(l2_distance_with_context(distance_context, lhs, rhs, stats));
}

const float *row_ptr(const std::vector<float> &points, uint64_t row,
                     uint32_t dimension) {
  return points.data() + static_cast<size_t>(row) * dimension;
}

float *center_ptr(std::vector<float> *centers, uint32_t center,
                  uint32_t dimension) {
  return centers->data() + static_cast<size_t>(center) * dimension;
}

const float *center_ptr(const std::vector<float> &centers, uint32_t center,
                        uint32_t dimension) {
  return centers.data() + static_cast<size_t>(center) * dimension;
}

void compute_centroid(const float *data, uint64_t row_count, uint32_t dimension,
                      std::vector<float> *centroid) {
  centroid->assign(dimension, 0.0F);
  if (row_count == 0) return;

  for (uint64_t row = 0; row < row_count; ++row) {
    const float *values = data + static_cast<size_t>(row) * dimension;
    for (uint32_t dim = 0; dim < dimension; ++dim) {
      (*centroid)[dim] += values[dim];
    }
  }
  const float scale = 1.0F / static_cast<float>(row_count);
  for (float &value : *centroid) value *= scale;
}

bool make_normalized_points(const float *data, uint64_t row_count,
                            const elkan_kmeans_config &config,
                            const std::vector<float> &centroid,
                            std::vector<float> *points,
                            std::string *error) {
  size_t value_count = 0;
  if (!multiply_size(row_count, config.dimension, &value_count)) {
    set_error(error, "input size overflow");
    return false;
  }

  points->assign(data, data + value_count);
  if (config.zero_mean) {
    for (uint64_t row = 0; row < row_count; ++row) {
      float *values = points->data() + static_cast<size_t>(row) * config.dimension;
      for (uint32_t dim = 0; dim < config.dimension; ++dim) {
        values[dim] -= centroid[dim];
      }
    }
  }
  return true;
}

void initialize_centers(const std::vector<float> &points, uint64_t row_count,
                        const elkan_kmeans_config &config,
                        std::vector<float> *centers) {
  centers->assign(static_cast<size_t>(config.center_count) * config.dimension,
                  0.0F);
  const uint64_t offset = row_count == 0 ? 0 : config.seed % row_count;
  for (uint32_t center = 0; center < config.center_count; ++center) {
    const uint64_t sample =
        (offset + (static_cast<uint64_t>(center) * row_count) /
                      config.center_count) %
        row_count;
    std::copy(row_ptr(points, sample, config.dimension),
              row_ptr(points, sample, config.dimension) + config.dimension,
              center_ptr(centers, center, config.dimension));
  }
}

bool assign_all_points(const std::vector<float> &points, uint64_t row_count,
                       const elkan_kmeans_config &config,
                       const std::vector<float> &centers,
                       std::vector<uint32_t> *assignments,
                       std::vector<float> *upper_bounds,
                       std::vector<float> *lower_bounds,
                       pq_centroid_scan_stats *scan_stats, std::string *error) {
  size_t bound_count = 0;
  if (!multiply_size(row_count, config.center_count, &bound_count)) {
    set_error(error, "lower bound size overflow");
    return false;
  }
  assignments->assign(static_cast<size_t>(row_count), 0);
  upper_bounds->assign(static_cast<size_t>(row_count), 0.0F);
  lower_bounds->assign(bound_count, 0.0F);

  pq_centroid_model centroid_model;
  if (!make_pq_centroid_model(centers.data(), config.center_count,
                              config.dimension, pq_centroid_scan_kernel::kAuto,
                              &centroid_model, error)) {
    return false;
  }

  for (uint64_t row = 0; row < row_count; ++row) {
    uint32_t best_center = 0;
    float best_distance = 0.0F;
    float *row_lower_bounds =
        lower_bounds->data() + static_cast<size_t>(row) * config.center_count;
    if (!scan_pq_centroid_distances(
            centroid_model, row_ptr(points, row, config.dimension),
            row_lower_bounds, &best_center, &best_distance, scan_stats)) {
      set_error(error, "initial centroid scan failed");
      return false;
    }
    (*assignments)[static_cast<size_t>(row)] = best_center;
    (*upper_bounds)[static_cast<size_t>(row)] = best_distance;
  }
  return true;
}

void recompute_sums(const std::vector<float> &points, uint64_t row_count,
                    const elkan_kmeans_config &config,
                    const std::vector<uint32_t> &assignments,
                    std::vector<float> *sums, center_counts *counts) {
  sums->assign(static_cast<size_t>(config.center_count) * config.dimension,
               0.0F);
  counts->assign(config.center_count, 0);
  for (uint64_t row = 0; row < row_count; ++row) {
    const uint32_t center = assignments[static_cast<size_t>(row)];
    ++(*counts)[center];
    float *sum = center_ptr(sums, center, config.dimension);
    const float *values = row_ptr(points, row, config.dimension);
    for (uint32_t dim = 0; dim < config.dimension; ++dim) sum[dim] += values[dim];
  }
}

uint64_t farthest_repair_row(const std::vector<uint32_t> &assignments,
                             const center_counts &counts,
                             const std::vector<float> &upper_bounds) {
  uint64_t selected_row = 0;
  float selected_distance = -1.0F;
  for (uint64_t row = 0; row < assignments.size(); ++row) {
    const uint32_t center = assignments[static_cast<size_t>(row)];
    if (counts[center] <= 1) continue;
    const float distance = upper_bounds[static_cast<size_t>(row)];
    if (distance > selected_distance) {
      selected_distance = distance;
      selected_row = row;
    }
  }
  return selected_row;
}

void repair_empty_clusters(const std::vector<float> &points, uint64_t row_count,
                           const elkan_kmeans_config &config,
                           std::vector<uint32_t> *assignments,
                           const std::vector<float> &upper_bounds) {
  std::vector<float> sums;
  center_counts counts;
  recompute_sums(points, row_count, config, *assignments, &sums, &counts);

  for (uint32_t center = 0; center < config.center_count; ++center) {
    if (counts[center] != 0) continue;
    const uint64_t repair_row = farthest_repair_row(*assignments, counts,
                                                    upper_bounds);
    const uint32_t old_center = (*assignments)[static_cast<size_t>(repair_row)];
    if (counts[old_center] > 1) {
      --counts[old_center];
      ++counts[center];
      (*assignments)[static_cast<size_t>(repair_row)] = center;
    }
  }
}

void update_centers(const std::vector<float> &points, uint64_t row_count,
                    const elkan_kmeans_config &config,
                    std::vector<uint32_t> *assignments,
                    const std::vector<float> &upper_bounds,
                    std::vector<float> *centers,
                    std::vector<float> *center_movement,
                    const l2_distance_context &distance_context,
                    l2_distance_stats *distance_stats) {
  repair_empty_clusters(points, row_count, config, assignments, upper_bounds);

  std::vector<float> sums;
  center_counts counts;
  recompute_sums(points, row_count, config, *assignments, &sums, &counts);

  center_movement->assign(config.center_count, 0.0F);
  std::vector<float> new_centers(centers->size(), 0.0F);
  for (uint32_t center = 0; center < config.center_count; ++center) {
    float *new_center = center_ptr(&new_centers, center, config.dimension);
    if (counts[center] == 0) {
      const uint64_t fallback_row = center % row_count;
      std::copy(row_ptr(points, fallback_row, config.dimension),
                row_ptr(points, fallback_row, config.dimension) +
                    config.dimension,
                new_center);
    } else {
      const float inv_count = 1.0F / static_cast<float>(counts[center]);
      const float *sum = center_ptr(sums, center, config.dimension);
      for (uint32_t dim = 0; dim < config.dimension; ++dim) {
        new_center[dim] = sum[dim] * inv_count;
      }
    }
    (*center_movement)[center] = euclidean_distance(
        distance_context, center_ptr(*centers, center, config.dimension),
        new_center, distance_stats);
  }
  *centers = std::move(new_centers);
}

void compute_center_half_distances(const std::vector<float> &centers,
                                   const elkan_kmeans_config &config,
                                   std::vector<float> *half_distances,
                                   std::vector<float> *min_half_distance,
                                   const l2_distance_context &distance_context,
                                   l2_distance_stats *distance_stats) {
  half_distances->assign(static_cast<size_t>(config.center_count) *
                             config.center_count,
                         0.0F);
  min_half_distance->assign(config.center_count,
                            std::numeric_limits<float>::infinity());

  for (uint32_t left = 0; left < config.center_count; ++left) {
    for (uint32_t right = left + 1; right < config.center_count; ++right) {
      const float half_distance =
          0.5F *
          euclidean_distance(
              distance_context, center_ptr(centers, left, config.dimension),
              center_ptr(centers, right, config.dimension), distance_stats);
      (*half_distances)[static_cast<size_t>(left) * config.center_count +
                        right] = half_distance;
      (*half_distances)[static_cast<size_t>(right) * config.center_count +
                        left] = half_distance;
      (*min_half_distance)[left] =
          std::min((*min_half_distance)[left], half_distance);
      (*min_half_distance)[right] =
          std::min((*min_half_distance)[right], half_distance);
    }
  }
}

void apply_center_movement_bounds(const std::vector<uint32_t> &assignments,
                                  const std::vector<float> &center_movement,
                                  uint32_t center_count,
                                  std::vector<float> *upper_bounds,
                                  std::vector<float> *lower_bounds) {
  for (uint64_t row = 0; row < assignments.size(); ++row) {
    (*upper_bounds)[static_cast<size_t>(row)] +=
        center_movement[assignments[static_cast<size_t>(row)]];
    for (uint32_t center = 0; center < center_count; ++center) {
      float &lower =
          (*lower_bounds)[static_cast<size_t>(row) * center_count + center];
      lower = std::max(0.0F, lower - center_movement[center]);
    }
  }
}

bool assign_with_elkan_bounds(
    const std::vector<float> &points, uint64_t row_count,
    const elkan_kmeans_config &config, const std::vector<float> &centers,
    const std::vector<float> &half_distances,
    const std::vector<float> &min_half_distance,
    std::vector<uint32_t> *assignments, std::vector<float> *upper_bounds,
    std::vector<float> *lower_bounds, uint64_t *skipped_distance_calls,
    const l2_distance_context &distance_context,
    l2_distance_stats *distance_stats) {
  bool changed = false;
  for (uint64_t row = 0; row < row_count; ++row) {
    const size_t row_index = static_cast<size_t>(row);
    uint32_t assigned = (*assignments)[row_index];
    float upper = (*upper_bounds)[row_index];
    if (upper <= min_half_distance[assigned]) {
      *skipped_distance_calls += config.center_count - 1;
      continue;
    }

    upper = euclidean_distance(
        distance_context, row_ptr(points, row, config.dimension),
        center_ptr(centers, assigned, config.dimension), distance_stats);
    (*upper_bounds)[row_index] = upper;

    for (uint32_t center = 0; center < config.center_count; ++center) {
      if (center == assigned) continue;
      float &lower =
          (*lower_bounds)[row_index * config.center_count + center];
      const float half_distance =
          half_distances[static_cast<size_t>(assigned) * config.center_count +
                         center];
      if (upper <= lower || upper <= half_distance) {
        ++(*skipped_distance_calls);
        continue;
      }

      const float distance = euclidean_distance(
          distance_context, row_ptr(points, row, config.dimension),
          center_ptr(centers, center, config.dimension), distance_stats);
      lower = distance;
      if (distance < upper) {
        assigned = center;
        upper = distance;
        changed = true;
        (*assignments)[row_index] = center;
        (*upper_bounds)[row_index] = distance;
      }
    }
  }
  return changed;
}

bool validate_config(const float *data, uint64_t row_count,
                     const elkan_kmeans_config &config,
                     const elkan_kmeans_result *result, std::string *error) {
  if (result == nullptr) {
    set_error(error, "result is null");
    return false;
  }
  if (config.dimension == 0) {
    set_error(error, "dimension is zero");
    return false;
  }
  if (config.center_count == 0) {
    set_error(error, "center_count is zero");
    return false;
  }
  if (config.max_iterations == 0) {
    set_error(error, "max_iterations is zero");
    return false;
  }
  if (data == nullptr && row_count != 0) {
    set_error(error, "data is null");
    return false;
  }
  if (row_count < config.center_count) {
    set_error(error, "row_count is smaller than center_count");
    return false;
  }
  return true;
}

}  // namespace

bool run_elkan_kmeans(const float *data, uint64_t row_count,
                      const elkan_kmeans_config &config,
                      elkan_kmeans_result *result, std::string *error) {
  if (!validate_config(data, row_count, config, result, error)) return false;
  *result = {};

  compute_centroid(data, row_count, config.dimension, &result->centroid);
  if (!config.zero_mean) {
    std::fill(result->centroid.begin(), result->centroid.end(), 0.0F);
  }

  std::vector<float> points;
  if (!make_normalized_points(data, row_count, config, result->centroid,
                              &points, error)) {
    return false;
  }

  const l2_distance_context distance_context =
      make_l2_distance_context(config.dimension);
  l2_distance_stats distance_stats;
  pq_centroid_scan_stats scan_stats;
  std::vector<float> upper_bounds;
  std::vector<float> lower_bounds;
  initialize_centers(points, row_count, config, &result->centers);
  if (!assign_all_points(points, row_count, config, result->centers,
                         &result->assignments, &upper_bounds, &lower_bounds,
                         &scan_stats, error)) {
    return false;
  }

  for (uint32_t iteration = 0; iteration < config.max_iterations; ++iteration) {
    std::vector<float> center_movement;
    update_centers(points, row_count, config, &result->assignments,
                   upper_bounds, &result->centers, &center_movement,
                   distance_context, &distance_stats);
    apply_center_movement_bounds(result->assignments, center_movement,
                                 config.center_count, &upper_bounds,
                                 &lower_bounds);

    std::vector<float> half_distances;
    std::vector<float> min_half_distance;
    compute_center_half_distances(result->centers, config, &half_distances,
                                  &min_half_distance, distance_context,
                                  &distance_stats);

    const bool changed = assign_with_elkan_bounds(
        points, row_count, config, result->centers, half_distances,
        min_half_distance, &result->assignments, &upper_bounds, &lower_bounds,
        &result->skipped_distance_calls, distance_context, &distance_stats);
    result->iterations = iteration + 1;
    if (!changed) {
      result->converged = true;
      break;
    }
  }

  std::vector<float> final_movement;
  update_centers(points, row_count, config, &result->assignments, upper_bounds,
                 &result->centers, &final_movement, distance_context,
                 &distance_stats);
  result->packed_distance_calls = scan_stats.distance_evaluations;
  result->centroid_scan_kernel =
      scan_stats.distance_evaluations == 0
          ? "not_run"
          : pq_centroid_scan_kernel_name(scan_stats.kernel);
  result->distance_calls = distance_stats.calls + result->packed_distance_calls;
  if (error != nullptr) error->clear();
  return true;
}

}  // namespace vector_index
