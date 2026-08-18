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

#include "sql/vector/vector_resource_budget.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>

#include "sql/vector/vector_index_build_options.h"

#ifdef __APPLE__
#include <mach/mach.h>
#else
#include <unistd.h>
#endif

namespace vector_index {
namespace {

constexpr uint64_t kUnlimitedCgroupV1Threshold =
    std::numeric_limits<uint64_t>::max() / 2;

std::string trim_ascii_whitespace(const std::string &text) {
  const size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const size_t last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

bool parse_uint64(const std::string &text, uint64_t *value) {
  if (value == nullptr) return false;
  const std::string trimmed = trim_ascii_whitespace(text);
  if (trimmed.empty() || trimmed.front() == '-') return false;
  uint64_t parsed = 0;
  const char *const begin = trimmed.data();
  const char *const end = begin + trimmed.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc() || result.ptr != end) return false;
  *value = parsed;
  return true;
}

bool parse_resource_limit(const std::string &text, uint64_t *value,
                          bool *unlimited) {
  if (value == nullptr || unlimited == nullptr) return false;
  const std::string trimmed = trim_ascii_whitespace(text);
  if (trimmed == "max") {
    *value = 0;
    *unlimited = true;
    return true;
  }
  uint64_t parsed = 0;
  if (!parse_uint64(trimmed, &parsed)) return false;
  *value = parsed;
  *unlimited = false;
  return true;
}

bool read_text_file(const std::filesystem::path &path, std::string *text) {
  if (text == nullptr) return false;
  std::ifstream input(path);
  if (!input.is_open()) return false;
  std::ostringstream output;
  output << input.rdbuf();
  if (input.bad()) return false;
  *text = output.str();
  return true;
}

bool read_mem_available(const std::filesystem::path &path,
                        uint64_t *available_memory) {
  if (available_memory == nullptr || path.empty()) return false;

  std::string text;
  if (!read_text_file(path, &text)) return false;

  std::istringstream lines(text);
  std::string line;
  constexpr char kMemAvailablePrefix[] = "MemAvailable:";
  while (std::getline(lines, line)) {
    if (line.rfind(kMemAvailablePrefix, 0) != 0) continue;

    std::istringstream value_stream(
        line.substr(sizeof(kMemAvailablePrefix) - 1));
    std::string value_text;
    std::string unit;
    std::string trailing;
    if (!(value_stream >> value_text >> unit) || unit != "kB" ||
        (value_stream >> trailing)) {
      return false;
    }

    uint64_t value_kib = 0;
    if (!parse_uint64(value_text, &value_kib) ||
        value_kib > std::numeric_limits<uint64_t>::max() / 1024) {
      return false;
    }
    *available_memory = value_kib * 1024;
    return true;
  }
  return false;
}

std::filesystem::path cgroup_path(const resource_probe_paths &paths,
                                  const std::string &controller,
                                  const std::string &relative) {
  std::filesystem::path path(paths.cgroup_root);
  if (!controller.empty()) path /= controller;
  std::filesystem::path suffix(relative);
  if (suffix.is_absolute()) suffix = suffix.relative_path();
  return path / suffix;
}

bool count_cpu_set(const std::string &text, uint32_t *cpu_count) {
  if (cpu_count == nullptr) return false;
  const std::string trimmed = trim_ascii_whitespace(text);
  if (trimmed.empty()) return false;

  uint64_t total = 0;
  size_t offset = 0;
  while (offset < trimmed.size()) {
    const size_t comma = trimmed.find(',', offset);
    const std::string token = trim_ascii_whitespace(trimmed.substr(
        offset, comma == std::string::npos ? std::string::npos
                                           : comma - offset));
    if (token.empty()) return false;
    const size_t dash = token.find('-');
    uint64_t first = 0;
    uint64_t last = 0;
    if (dash == std::string::npos) {
      if (!parse_uint64(token, &first)) return false;
      last = first;
    } else {
      if (token.find('-', dash + 1) != std::string::npos ||
          !parse_uint64(token.substr(0, dash), &first) ||
          !parse_uint64(token.substr(dash + 1), &last) || last < first) {
        return false;
      }
    }
    const uint64_t range_count = last - first + 1;
    if (range_count > std::numeric_limits<uint32_t>::max() - total) {
      return false;
    }
    total += range_count;
    if (comma == std::string::npos) break;
    offset = comma + 1;
  }
  if (total == 0) return false;
  *cpu_count = static_cast<uint32_t>(total);
  return true;
}

uint32_t quota_cpu_slots(uint64_t quota, uint64_t period) {
  if (quota == 0 || period == 0) return 0;
  return static_cast<uint32_t>(std::min<uint64_t>(
      std::max<uint64_t>(1, quota / period),
      std::numeric_limits<uint32_t>::max()));
}

void apply_cpu_limit(uint32_t candidate, uint32_t *effective) {
  if (effective == nullptr || candidate == 0) return;
  *effective = std::min(*effective, candidate);
}

bool parse_cgroup_membership(
    const std::string &text, std::string *v2_path,
    std::unordered_map<std::string, std::string> *v1_paths) {
  if (v2_path == nullptr || v1_paths == nullptr) return false;
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    const size_t first = line.find(':');
    const size_t second =
        first == std::string::npos ? std::string::npos : line.find(':', first + 1);
    if (first == std::string::npos || second == std::string::npos) continue;
    const std::string controllers = line.substr(first + 1, second - first - 1);
    const std::string path = line.substr(second + 1);
    if (controllers.empty()) {
      *v2_path = path;
      continue;
    }
    size_t offset = 0;
    while (offset <= controllers.size()) {
      const size_t comma = controllers.find(',', offset);
      const std::string controller = controllers.substr(
          offset, comma == std::string::npos ? std::string::npos
                                             : comma - offset);
      if (!controller.empty()) (*v1_paths)[controller] = path;
      if (comma == std::string::npos) break;
      offset = comma + 1;
    }
  }
  return !v2_path->empty() || !v1_paths->empty();
}

bool read_uint64_file(const std::filesystem::path &path, uint64_t *value) {
  std::string text;
  return read_text_file(path, &text) && parse_uint64(text, value);
}

bool path_is_within(const std::filesystem::path &root,
                    const std::filesystem::path &path) {
  auto root_part = root.begin();
  auto path_part = path.begin();
  while (root_part != root.end()) {
    if (path_part == path.end() || *root_part != *path_part) return false;
    ++root_part;
    ++path_part;
  }
  return true;
}

bool read_cgroup_v2_memory(const std::filesystem::path &root,
                           uint64_t *limit, uint64_t *current,
                           bool *unlimited) {
  std::string limit_text;
  return read_text_file(root / "memory.max", &limit_text) &&
         parse_resource_limit(limit_text, limit, unlimited) &&
         read_uint64_file(root / "memory.current", current);
}

uint32_t read_cgroup_v2_cpu_slots(const std::filesystem::path &root) {
  std::string cpu_max;
  if (!read_text_file(root / "cpu.max", &cpu_max)) return 0;

  std::istringstream values(cpu_max);
  std::string quota_text;
  std::string period_text;
  values >> quota_text >> period_text;
  uint64_t quota = 0;
  uint64_t period = 0;
  if (quota_text == "max" || !parse_uint64(quota_text, &quota) ||
      !parse_uint64(period_text, &period)) {
    return 0;
  }
  return quota_cpu_slots(quota, period);
}

bool probe_cgroup_v2(const resource_probe_paths &paths,
                     const std::string &relative,
                     resource_probe_snapshot *snapshot) {
  if (snapshot == nullptr || relative.empty()) return false;
  const std::filesystem::path hierarchy_root =
      std::filesystem::path(paths.cgroup_root).lexically_normal();
  const std::filesystem::path leaf =
      cgroup_path(paths, "", relative).lexically_normal();
  if (!path_is_within(hierarchy_root, leaf)) return false;

  uint64_t current = 0;
  uint64_t limit = 0;
  bool unlimited = false;
  if (!read_cgroup_v2_memory(leaf, &limit, &current, &unlimited)) {
    return false;
  }

  snapshot->source = resource_probe_source::kCgroupV2;
  snapshot->memory_limit = unlimited ? 0 : limit;
  snapshot->memory_current = current;
  uint64_t effective_headroom = snapshot->host_available_memory;
  bool has_effective_headroom = effective_headroom != 0;
  uint64_t tightest_cgroup_headroom = 0;
  bool has_finite_cgroup_memory = false;

  for (std::filesystem::path level = leaf;; level = level.parent_path()) {
    uint64_t level_limit = 0;
    uint64_t level_current = 0;
    bool level_unlimited = false;
    if (level == leaf || read_cgroup_v2_memory(level, &level_limit,
                                               &level_current,
                                               &level_unlimited)) {
      if (level == leaf) {
        level_limit = limit;
        level_current = current;
        level_unlimited = unlimited;
      }
      if (!level_unlimited) {
        const uint64_t level_headroom =
            level_current >= level_limit ? 0 : level_limit - level_current;
        if (!has_effective_headroom || level_headroom < effective_headroom) {
          effective_headroom = level_headroom;
          has_effective_headroom = true;
        }
        if (!has_finite_cgroup_memory ||
            level_headroom < tightest_cgroup_headroom) {
          tightest_cgroup_headroom = level_headroom;
          snapshot->memory_limit = level_limit;
          snapshot->memory_current = level_current;
          has_finite_cgroup_memory = true;
        }
      }
    }

    const uint32_t level_cpu_slots = read_cgroup_v2_cpu_slots(level);
    if (level_cpu_slots != 0 &&
        (snapshot->quota_cpu_slots == 0 ||
         level_cpu_slots < snapshot->quota_cpu_slots)) {
      snapshot->quota_cpu_slots = level_cpu_slots;
    }

    if (level == hierarchy_root) break;
  }
  snapshot->memory_headroom =
      has_effective_headroom ? effective_headroom : 0;

  std::string cpuset;
  if (read_text_file(leaf / "cpuset.cpus.effective", &cpuset)) {
    count_cpu_set(cpuset, &snapshot->cpuset_cpu_slots);
  }
  return true;
}

bool probe_cgroup_v1(
    const resource_probe_paths &paths,
    const std::unordered_map<std::string, std::string> &memberships,
    resource_probe_snapshot *snapshot) {
  if (snapshot == nullptr) return false;
  const auto memory = memberships.find("memory");
  if (memory == memberships.end()) return false;
  const std::filesystem::path memory_root =
      cgroup_path(paths, "memory", memory->second);
  uint64_t limit = 0;
  uint64_t current = 0;
  if (!read_uint64_file(memory_root / "memory.limit_in_bytes", &limit) ||
      !read_uint64_file(memory_root / "memory.usage_in_bytes", &current)) {
    return false;
  }

  const bool unlimited = limit >= kUnlimitedCgroupV1Threshold;
  snapshot->source = resource_probe_source::kCgroupV1;
  snapshot->memory_limit = unlimited ? 0 : limit;
  snapshot->memory_current = current;
  uint64_t headroom = snapshot->host_available_memory;
  if (!unlimited) {
    const uint64_t cgroup_headroom = current >= limit ? 0 : limit - current;
    headroom = headroom == 0 ? cgroup_headroom
                             : std::min(headroom, cgroup_headroom);
  }
  snapshot->memory_headroom = headroom;

  const auto cpu = memberships.find("cpu");
  if (cpu != memberships.end()) {
    const std::filesystem::path cpu_root =
        cgroup_path(paths, "cpu", cpu->second);
    uint64_t quota = 0;
    uint64_t period = 0;
    if (read_uint64_file(cpu_root / "cpu.cfs_quota_us", &quota) &&
        read_uint64_file(cpu_root / "cpu.cfs_period_us", &period) &&
        quota <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      snapshot->quota_cpu_slots = quota_cpu_slots(quota, period);
    }
  }
  const auto cpuset = memberships.find("cpuset");
  if (cpuset != memberships.end()) {
    std::string cpus;
    if (read_text_file(cgroup_path(paths, "cpuset", cpuset->second) /
                           "cpuset.cpus",
                       &cpus)) {
      count_cpu_set(cpus, &snapshot->cpuset_cpu_slots);
    }
  }
  return true;
}

bool probe_with_paths(const resource_probe_paths &paths,
                      uint64_t host_available_memory,
                      uint32_t hardware_cpu_slots, uint64_t process_rss,
                      resource_probe_snapshot *snapshot) {
  if (snapshot == nullptr) return false;
  uint64_t mem_available = 0;
  if (read_mem_available(paths.proc_meminfo, &mem_available)) {
    host_available_memory = mem_available;
  }
  *snapshot = resource_probe_snapshot{};
  snapshot->host_available_memory = host_available_memory;
  snapshot->process_rss = process_rss;
  snapshot->memory_headroom = host_available_memory;
  snapshot->hardware_cpu_slots = std::max<uint32_t>(1, hardware_cpu_slots);
  snapshot->effective_cpu_slots = snapshot->hardware_cpu_slots;

  std::string membership_text;
  std::string v2_path;
  std::unordered_map<std::string, std::string> v1_paths;
  if (read_text_file(paths.proc_self_cgroup, &membership_text) &&
      parse_cgroup_membership(membership_text, &v2_path, &v1_paths)) {
    if (!probe_cgroup_v2(paths, v2_path, snapshot)) {
      probe_cgroup_v1(paths, v1_paths, snapshot);
    }
  }

  apply_cpu_limit(snapshot->quota_cpu_slots,
                  &snapshot->effective_cpu_slots);
  apply_cpu_limit(snapshot->cpuset_cpu_slots,
                  &snapshot->effective_cpu_slots);
  snapshot->effective_cpu_slots =
      std::max<uint32_t>(1, snapshot->effective_cpu_slots);
  return true;
}

uint64_t host_available_memory() {
#ifdef __APPLE__
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  vm_statistics64_data_t statistics{};
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&statistics),
                        &count) != KERN_SUCCESS) {
    return 0;
  }
  uint64_t page_size = 0;
  vm_size_t native_page_size = 0;
  if (host_page_size(mach_host_self(), &native_page_size) == KERN_SUCCESS) {
    page_size = native_page_size;
  }
  return page_size *
         (statistics.free_count + statistics.inactive_count);
#else
  const long pages = sysconf(_SC_AVPHYS_PAGES);
  const long page_size = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page_size <= 0) return 0;
  return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
#endif
}

uint64_t process_resident_memory() {
#ifdef __APPLE__
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info),
                &count) != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<uint64_t>(info.resident_size);
#else
  std::ifstream statm("/proc/self/statm");
  uint64_t pages = 0;
  uint64_t resident = 0;
  if (!(statm >> pages >> resident)) return 0;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) return 0;
  return resident * static_cast<uint64_t>(page_size);
#endif
}

uint64_t safe_subtract(uint64_t value, uint64_t amount) {
  return amount >= value ? 0 : value - amount;
}

}  // namespace

build_resource_lease::~build_resource_lease() { reset(); }

build_resource_lease::build_resource_lease(
    build_resource_lease &&other) noexcept
    : m_manager(other.m_manager),
      m_memory_bytes(other.m_memory_bytes),
      m_cpu_slots(other.m_cpu_slots) {
  other.m_manager = nullptr;
  other.m_memory_bytes = 0;
  other.m_cpu_slots = 0;
}

build_resource_lease &build_resource_lease::operator=(
    build_resource_lease &&other) noexcept {
  if (this == &other) return *this;
  reset();
  m_manager = other.m_manager;
  m_memory_bytes = other.m_memory_bytes;
  m_cpu_slots = other.m_cpu_slots;
  other.m_manager = nullptr;
  other.m_memory_bytes = 0;
  other.m_cpu_slots = 0;
  return *this;
}

void build_resource_lease::reset() {
  if (m_manager != nullptr) {
    m_manager->release(m_memory_bytes, m_cpu_slots);
  }
  m_manager = nullptr;
  m_memory_bytes = 0;
  m_cpu_slots = 0;
}

resource_probe_snapshot resource_budget_manager::probe_resources() const {
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
  probe_function probe;
  {
    std::lock_guard<std::mutex> guard(m_mutex);
    probe = m_probe;
  }
  return probe ? probe() : probe_build_resources();
#else
  return probe_build_resources();
#endif
}

build_resource_snapshot resource_budget_manager::snapshot_locked(
    const resource_probe_snapshot &probe) const {
  build_resource_snapshot state;
  state.probe = probe;
  state.configured_memory_budget = opt_vector_build_memory_size;
  state.memory_reserve = opt_vector_build_memory_reserve_size;
  const uint64_t safe_headroom =
      safe_subtract(probe.memory_headroom, state.memory_reserve);
  state.effective_memory_budget =
      state.configured_memory_budget == 0
          ? safe_headroom
          : std::min(state.configured_memory_budget, safe_headroom);
  state.reserved_memory = m_reserved_memory;
  state.effective_cpu_slots = std::max<uint32_t>(1, probe.effective_cpu_slots);
  state.reserved_cpu_slots = m_reserved_cpu_slots;
  state.active_builds = m_active_builds;
  state.waiting_builds = m_waiting_builds;
  state.last_effective_memory = m_last_effective_memory;
  state.last_effective_cpu_slots = m_last_effective_cpu_slots;
  return state;
}

bool resource_budget_manager::acquire(const build_resource_request &request,
                                      build_resource_lease *lease,
                                      const resource_wait_options &wait_options) {
  if (lease == nullptr) return false;
  lease->reset();

  using clock = std::chrono::steady_clock;
  constexpr auto k_cancel_poll_interval = std::chrono::milliseconds(100);
  const auto wait_started = clock::now();
  const bool has_timeout = wait_options.timeout.count() > 0;

  std::unique_lock<std::mutex> lock(m_mutex);
  bool waiting = false;
  const auto leave_waiting = [&] {
    if (!waiting) return;
    --m_waiting_builds;
    waiting = false;
  };
  const auto cancelled = [&] {
    return wait_options.cancelled && wait_options.cancelled();
  };
  const auto timeout_remaining = [&] {
    if (!has_timeout) return k_cancel_poll_interval;
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() -
                                                              wait_started);
    if (elapsed >= wait_options.timeout) return std::chrono::milliseconds(0);
    return std::min(k_cancel_poll_interval, wait_options.timeout - elapsed);
  };

  for (;;) {
    if (cancelled()) {
      leave_waiting();
      return false;
    }
    const auto remaining = timeout_remaining();
    if (has_timeout && remaining.count() == 0) {
      leave_waiting();
      return false;
    }
#ifdef EXTRA_CODE_FOR_UNIT_TESTING
    probe_function probe_function_copy = m_probe;
    lock.unlock();
    const resource_probe_snapshot probe =
        probe_function_copy ? probe_function_copy() : probe_build_resources();
#else
    lock.unlock();
    const resource_probe_snapshot probe = probe_build_resources();
#endif
    lock.lock();
    if (cancelled()) {
      leave_waiting();
      return false;
    }
    if (has_timeout && timeout_remaining().count() == 0) {
      leave_waiting();
      return false;
    }
    const build_resource_snapshot state = snapshot_locked(probe);
    if (state.effective_memory_budget == 0 ||
        state.effective_cpu_slots == 0) {
      leave_waiting();
      return false;
    }
    if (request.memory_bytes > state.effective_memory_budget) {
      leave_waiting();
      return false;
    }

    const uint64_t memory =
        request.memory_bytes == 0
            ? state.effective_memory_budget
            : request.memory_bytes;
    const uint32_t cpu =
        request.cpu_slots == 0
            ? state.effective_cpu_slots
            : std::min(request.cpu_slots, state.effective_cpu_slots);
    const bool memory_fits =
        m_reserved_memory <= state.effective_memory_budget - memory;
    const bool cpu_fits =
        m_reserved_cpu_slots <= state.effective_cpu_slots - cpu;
    if (memory_fits && cpu_fits) {
      leave_waiting();
      m_reserved_memory += memory;
      m_reserved_cpu_slots += cpu;
      ++m_active_builds;
      m_last_effective_memory = memory;
      m_last_effective_cpu_slots = cpu;
      lease->m_manager = this;
      lease->m_memory_bytes = memory;
      lease->m_cpu_slots = cpu;
      return true;
    }

    if (!waiting) {
      ++m_waiting_builds;
      waiting = true;
    }
    m_condition.wait_for(lock, timeout_remaining());
  }
}

build_resource_snapshot resource_budget_manager::snapshot() const {
  const resource_probe_snapshot probe = probe_resources();
  std::lock_guard<std::mutex> guard(m_mutex);
  return snapshot_locked(probe);
}

void resource_budget_manager::release(uint64_t memory_bytes,
                                      uint32_t cpu_slots) {
  {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_reserved_memory =
        memory_bytes >= m_reserved_memory ? 0 : m_reserved_memory - memory_bytes;
    m_reserved_cpu_slots =
        cpu_slots >= m_reserved_cpu_slots ? 0 : m_reserved_cpu_slots - cpu_slots;
    if (m_active_builds != 0) --m_active_builds;
  }
  m_condition.notify_all();
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
void resource_budget_manager::set_probe_for_testing(probe_function probe) {
  {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_probe = std::move(probe);
  }
  m_condition.notify_all();
}

void resource_budget_manager::reset_for_testing() {
  {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_probe = nullptr;
    m_reserved_memory = 0;
    m_reserved_cpu_slots = 0;
    m_active_builds = 0;
    m_waiting_builds = 0;
    m_last_effective_memory = 0;
    m_last_effective_cpu_slots = 0;
  }
  m_condition.notify_all();
}
#endif

resource_budget_manager &global_resource_budget_manager() {
  static resource_budget_manager manager;
  return manager;
}

resource_probe_snapshot probe_build_resources() {
  resource_probe_snapshot snapshot;
  resource_probe_paths paths;
#ifndef __APPLE__
  paths.proc_meminfo = "/proc/meminfo";
#endif
  probe_with_paths(paths, host_available_memory(),
                   std::max(1U, std::thread::hardware_concurrency()),
                   process_resident_memory(), &snapshot);
  return snapshot;
}

const char *resource_probe_source_name(resource_probe_source source) {
  switch (source) {
    case resource_probe_source::kCgroupV2:
      return "cgroup_v2";
    case resource_probe_source::kCgroupV1:
      return "cgroup_v1";
    case resource_probe_source::kHost:
      return "host";
  }
  return "host";
}

#ifdef EXTRA_CODE_FOR_UNIT_TESTING
bool parse_resource_limit_for_testing(const std::string &text,
                                      uint64_t *value, bool *unlimited) {
  return parse_resource_limit(text, value, unlimited);
}

bool count_cpu_set_for_testing(const std::string &text, uint32_t *cpu_count) {
  return count_cpu_set(text, cpu_count);
}

bool probe_build_resources_for_testing(
    const resource_probe_paths &paths, uint64_t host_available_memory,
    uint32_t hardware_cpu_slots, uint64_t process_rss,
    resource_probe_snapshot *snapshot) {
  return probe_with_paths(paths, host_available_memory, hardware_cpu_slots,
                          process_rss, snapshot);
}
#endif

}  // namespace vector_index
