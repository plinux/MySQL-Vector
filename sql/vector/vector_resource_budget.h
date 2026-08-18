/* Copyright (c) 2026, Oracle and/or its affiliates.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0,
   as published by the Free Software Foundation.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

#ifndef SQL_VECTOR_VECTOR_RESOURCE_BUDGET_INCLUDED
#define SQL_VECTOR_VECTOR_RESOURCE_BUDGET_INCLUDED

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace vector_index {

/** Source that supplied the effective vector build resource envelope. */
enum class resource_probe_source { kCgroupV2, kCgroupV1, kHost };

/** Filesystem inputs used by the Linux resource probe. */
struct resource_probe_paths {
  std::string proc_meminfo;
  std::string proc_self_cgroup{"/proc/self/cgroup"};
  std::string cgroup_root{"/sys/fs/cgroup"};
};

/** One point-in-time view of process and container build resources. */
struct resource_probe_snapshot {
  resource_probe_source source{resource_probe_source::kHost};
  uint64_t memory_limit{0};
  uint64_t memory_current{0};
  uint64_t host_available_memory{0};
  uint64_t process_rss{0};
  uint64_t memory_headroom{0};
  uint32_t hardware_cpu_slots{1};
  uint32_t quota_cpu_slots{0};
  uint32_t cpuset_cpu_slots{0};
  uint32_t effective_cpu_slots{1};
};

/** Resource request made by one vector backend build. */
struct build_resource_request {
  uint64_t memory_bytes{0};
  uint32_t cpu_slots{0};
};

/** Cancellation and timeout policy for one build admission request. */
struct resource_wait_options {
  std::function<bool()> cancelled;
  std::chrono::milliseconds timeout{0};
};

/** Process-wide admission state exposed through vector observability. */
struct build_resource_snapshot {
  resource_probe_snapshot probe;
  uint64_t configured_memory_budget{0};
  uint64_t memory_reserve{0};
  uint64_t effective_memory_budget{0};
  uint64_t reserved_memory{0};
  uint32_t effective_cpu_slots{1};
  uint32_t reserved_cpu_slots{0};
  uint32_t active_builds{0};
  uint32_t waiting_builds{0};
  uint64_t last_effective_memory{0};
  uint32_t last_effective_cpu_slots{0};
};

class resource_budget_manager;

/** Move-only lease that releases a vector build reservation on destruction. */
class build_resource_lease {
 public:
  build_resource_lease() = default;
  ~build_resource_lease();

  build_resource_lease(build_resource_lease &&other) noexcept;
  build_resource_lease &operator=(build_resource_lease &&other) noexcept;

  build_resource_lease(const build_resource_lease &) = delete;
  build_resource_lease &operator=(const build_resource_lease &) = delete;

  uint64_t memory_bytes() const { return m_memory_bytes; }
  uint32_t cpu_slots() const { return m_cpu_slots; }
  explicit operator bool() const { return m_manager != nullptr; }

  void reset();

 private:
  friend class resource_budget_manager;

  resource_budget_manager *m_manager{nullptr};
  uint64_t m_memory_bytes{0};
  uint32_t m_cpu_slots{0};
};

/**
  Admit concurrent vector builds against one process-wide memory/CPU envelope.

  Nonzero memory requests require a complete grant and fail when they exceed
  the current envelope. CPU requests may be clipped to the effective slot
  count. Other builds wait until the aggregate reservation fits; the lease
  destructor releases the reservation and wakes waiters.
*/
class resource_budget_manager {
 public:
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  using probe_function = std::function<resource_probe_snapshot()>;
#endif

  bool acquire(const build_resource_request &request,
               build_resource_lease *lease,
               const resource_wait_options &wait_options = {});
  build_resource_snapshot snapshot() const;

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  void set_probe_for_testing(probe_function probe);
  void reset_for_testing();
#endif

 private:
  friend class build_resource_lease;

  build_resource_snapshot snapshot_locked(
      const resource_probe_snapshot &probe) const;
  resource_probe_snapshot probe_resources() const;
  void release(uint64_t memory_bytes, uint32_t cpu_slots);

  mutable std::mutex m_mutex;
  std::condition_variable m_condition;
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  probe_function m_probe;
#endif
  uint64_t m_reserved_memory{0};
  uint32_t m_reserved_cpu_slots{0};
  uint32_t m_active_builds{0};
  uint32_t m_waiting_builds{0};
  uint64_t m_last_effective_memory{0};
  uint32_t m_last_effective_cpu_slots{0};
};

/** Return the process-wide vector build admission manager. */
resource_budget_manager &global_resource_budget_manager();

/** Probe the current cgroup or host build resource envelope. */
resource_probe_snapshot probe_build_resources();

/** Return the stable lowercase name used by status output. */
const char *resource_probe_source_name(resource_probe_source source);

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool parse_resource_limit_for_testing(const std::string &text,
                                      uint64_t *value, bool *unlimited);
bool count_cpu_set_for_testing(const std::string &text, uint32_t *cpu_count);
bool probe_build_resources_for_testing(
    const resource_probe_paths &paths, uint64_t host_available_memory,
    uint32_t hardware_cpu_slots, uint64_t process_rss,
    resource_probe_snapshot *snapshot);
#endif

}  // namespace vector_index

#endif  // SQL_VECTOR_VECTOR_RESOURCE_BUDGET_INCLUDED
