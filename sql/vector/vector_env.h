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

#ifndef SQL_VECTOR_ENV_INCLUDED
#define SQL_VECTOR_ENV_INCLUDED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

namespace vector_env {

inline bool read_bool_or(const char *name, bool fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
      std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "yes") == 0 ||
      std::strcmp(value, "YES") == 0 || std::strcmp(value, "on") == 0 ||
      std::strcmp(value, "ON") == 0) {
    return true;
  }
  if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
      std::strcmp(value, "FALSE") == 0 || std::strcmp(value, "no") == 0 ||
      std::strcmp(value, "NO") == 0 || std::strcmp(value, "off") == 0 ||
      std::strcmp(value, "OFF") == 0) {
    return false;
  }
  return fallback;
}

inline uint32_t read_u32_or(const char *name, uint32_t fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;

  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    return fallback;
  }
  return static_cast<uint32_t>(parsed);
}

inline size_t limit_thread_count(size_t configured, size_t max_threads,
                                 size_t work_count) {
  const unsigned int hardware_threads = std::thread::hardware_concurrency();
  if (hardware_threads > 0)
    configured = std::min<size_t>(configured, hardware_threads);
  configured = std::min(configured, max_threads);
  return std::max<size_t>(1, std::min(work_count, configured));
}

inline size_t read_thread_count(const char *name, size_t default_threads,
                                size_t max_threads, size_t work_count,
                                bool default_to_hardware) {
  if (work_count <= 1) return work_count;
  const unsigned int hardware_threads = std::thread::hardware_concurrency();
  size_t configured =
      default_to_hardware && hardware_threads > 0 ? hardware_threads
                                                  : default_threads;
  const char *value = std::getenv(name);
  if (value != nullptr && *value != '\0') {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end != value && parsed > 0) {
      configured = std::min<size_t>(static_cast<size_t>(parsed), max_threads);
    }
  }
  return limit_thread_count(configured, max_threads, work_count);
}

}  // namespace vector_env

#endif  // SQL_VECTOR_ENV_INCLUDED
