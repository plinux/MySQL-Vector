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

#include "sql/vector/vector_statement_publication.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <new>
#include <unordered_map>
#include <utility>

#include "my_dbug.h"
#include "sql/mysqld.h"
#include "sql/sql_class.h"

namespace vector_statement_publication {
namespace {

enum class reservation_kind : uint8_t {
  kIndexRead,
  kIndexWrite,
  kCatalogRead,
  kCatalogWrite,
};

struct index_reservation_state {
  size_t readers{0};
  bool writer{false};
};

std::mutex g_reservation_mutex;
std::condition_variable g_reservation_changed;
std::unordered_map<std::string, index_reservation_state> g_index_reservations;
size_t g_catalog_readers{0};
size_t g_waiting_catalog_writers{0};
bool g_catalog_writer{false};

constexpr uint8_t k_operation_payload_version = 1;
constexpr auto k_reservation_wait_poll_interval = std::chrono::milliseconds(50);

bool reservation_wait_cancelled(
    const cancellation_predicate &cancelled) noexcept {
  try {
    if (cancelled) return cancelled();
  } catch (...) {
    return true;
  }
  return connection_events_loop_aborted() ||
         (current_thd != nullptr &&
          current_thd->killed != THD::NOT_KILLED);
}

void append_uint64(uint64_t value, std::string *encoded) {
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    encoded->push_back(static_cast<char>((value >> shift) & 0xffU));
  }
}

bool read_uint64(const std::string &encoded, size_t *offset, uint64_t *value) {
  if (offset == nullptr || value == nullptr ||
      *offset > encoded.size() || encoded.size() - *offset < sizeof(uint64_t)) {
    return false;
  }
  uint64_t decoded = 0;
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    decoded |= static_cast<uint64_t>(
                   static_cast<unsigned char>(encoded[*offset + shift / 8]))
               << shift;
  }
  *offset += sizeof(uint64_t);
  *value = decoded;
  return true;
}

bool append_sized_string(const std::string &value, std::string *encoded) {
  if (value.size() > std::numeric_limits<uint64_t>::max()) return false;
  append_uint64(static_cast<uint64_t>(value.size()), encoded);
  encoded->append(value);
  return true;
}

bool read_sized_string(const std::string &encoded, size_t *offset,
                       std::string *value) {
  uint64_t length = 0;
  if (!read_uint64(encoded, offset, &length) ||
      length > std::numeric_limits<size_t>::max() ||
      *offset > encoded.size() || length > encoded.size() - *offset) {
    return false;
  }
  value->assign(encoded.data() + *offset, static_cast<size_t>(length));
  *offset += static_cast<size_t>(length);
  return true;
}

bool count_fits_minimum_frame(const std::string &encoded, size_t offset,
                              uint64_t count, size_t element_bytes,
                              size_t trailing_bytes) {
  if (offset > encoded.size() || element_bytes == 0) return false;
  const size_t remaining_bytes = encoded.size() - offset;
  return remaining_bytes >= trailing_bytes &&
         count <= (remaining_bytes - trailing_bytes) / element_bytes;
}

bool valid_config_change(config_change change) {
  switch (change) {
    case config_change::kSearchEf:
    case config_change::kHnswBuildParams:
    case config_change::kFaissIvfParams:
    case config_change::kFaissIvfPqParams:
    case config_change::kDiskannBuildParams:
    case config_change::kDiskannSearchComplexity:
    case config_change::kDiskannSearchBeamwidth:
    case config_change::kDiskannPqCodeBudgetSize:
    case config_change::kDiskannDiskPqDims:
    case config_change::kDiskannAccelerateBuild:
    case config_change::kDiskannShuffleBuild:
    case config_change::kDiskannUseBfsCache:
    case config_change::kDiskannBuildMode:
      return true;
  }
  return false;
}

std::vector<std::string> normalized_index_names(
    const std::vector<std::string> &index_names) {
  std::vector<std::string> normalized;
  normalized.reserve(index_names.size());
  for (const std::string &index_name : index_names) {
    if (!index_name.empty()) normalized.push_back(index_name);
  }
  std::sort(normalized.begin(), normalized.end());
  normalized.erase(std::unique(normalized.begin(), normalized.end()),
                   normalized.end());
  return normalized;
}

bool index_reservation_available(const std::vector<std::string> &index_names,
                                 reservation_kind kind) {
  if (g_catalog_writer || g_waiting_catalog_writers != 0) return false;
  for (const std::string &index_name : index_names) {
    const auto it = g_index_reservations.find(index_name);
    if (it == g_index_reservations.end()) continue;
    if (it->second.writer ||
        (kind == reservation_kind::kIndexWrite && it->second.readers != 0)) {
      return false;
    }
  }
  return true;
}

void release_reservation(reservation_kind kind,
                         const std::vector<std::string> &index_names) {
  {
    std::lock_guard<std::mutex> guard(g_reservation_mutex);
    switch (kind) {
      case reservation_kind::kIndexRead:
      case reservation_kind::kIndexWrite:
        for (const std::string &index_name : index_names) {
          const auto it = g_index_reservations.find(index_name);
          if (it == g_index_reservations.end()) continue;
          if (kind == reservation_kind::kIndexRead) {
            if (it->second.readers != 0) --it->second.readers;
          } else {
            it->second.writer = false;
          }
          if (it->second.readers == 0 && !it->second.writer) {
            g_index_reservations.erase(it);
          }
        }
        if (g_catalog_readers != 0) --g_catalog_readers;
        break;
      case reservation_kind::kCatalogRead:
        if (g_catalog_readers != 0) --g_catalog_readers;
        break;
      case reservation_kind::kCatalogWrite:
        g_catalog_writer = false;
        break;
    }
  }
  g_reservation_changed.notify_all();
}

}  // namespace

/**
  Thread-independent ownership token for one publication boundary.

  MySQL group commit can execute a follower's after-commit callback on the
  leader thread. The token therefore records coordinator state instead of
  owning a C++ mutex lock that must be released by its acquiring thread.
*/
class reservation_lease {
 public:
  reservation_lease(reservation_kind kind, std::vector<std::string> index_names)
      : m_kind(kind), m_index_names(std::move(index_names)) {}

  reservation_lease(const reservation_lease &) = delete;
  reservation_lease &operator=(const reservation_lease &) = delete;

  ~reservation_lease() {
    if (m_active) release_reservation(m_kind, m_index_names);
  }

  void activate() { m_active = true; }

 private:
  reservation_kind m_kind;
  std::vector<std::string> m_index_names;
  bool m_active{false};
};

namespace {

void erase_unused_index_reservations_locked(
    const std::vector<std::string> &index_names) {
  for (const std::string &index_name : index_names) {
    const auto it = g_index_reservations.find(index_name);
    if (it != g_index_reservations.end() && it->second.readers == 0 &&
        !it->second.writer) {
      g_index_reservations.erase(it);
    }
  }
}

std::shared_ptr<reservation_lease> grant_index_reservation_locked(
    const std::vector<std::string> &index_names, reservation_kind kind) {
  std::shared_ptr<reservation_lease> lease;
  try {
    lease = std::make_shared<reservation_lease>(kind, index_names);
    for (const std::string &index_name : index_names) {
      g_index_reservations.try_emplace(index_name);
      DBUG_EXECUTE_IF("vector_publication_reservation_allocation_failure",
                      { throw std::bad_alloc(); });
    }
  } catch (...) {
    erase_unused_index_reservations_locked(index_names);
    return nullptr;
  }

  if (g_catalog_readers == std::numeric_limits<size_t>::max()) {
    erase_unused_index_reservations_locked(index_names);
    return nullptr;
  }
  if (kind == reservation_kind::kIndexRead) {
    for (const std::string &index_name : index_names) {
      if (g_index_reservations.find(index_name)->second.readers ==
          std::numeric_limits<size_t>::max()) {
        erase_unused_index_reservations_locked(index_names);
        return nullptr;
      }
    }
  }

  for (const std::string &index_name : index_names) {
    auto &state = g_index_reservations.find(index_name)->second;
    if (kind == reservation_kind::kIndexRead) {
      ++state.readers;
    } else {
      state.writer = true;
    }
  }
  ++g_catalog_readers;
  lease->activate();
  return lease;
}

std::shared_ptr<reservation_lease> acquire_index_reservation(
    std::vector<std::string> index_names, reservation_kind kind,
    const cancellation_predicate &cancelled) {
  if (index_names.empty()) return nullptr;

  std::unique_lock<std::mutex> guard(g_reservation_mutex);
  while (!index_reservation_available(index_names, kind)) {
    if (reservation_wait_cancelled(cancelled)) return nullptr;
    g_reservation_changed.wait_for(guard, k_reservation_wait_poll_interval);
  }
  if (reservation_wait_cancelled(cancelled)) return nullptr;
  return grant_index_reservation_locked(index_names, kind);
}

std::shared_ptr<reservation_lease> try_acquire_index_reservation(
    std::vector<std::string> index_names, reservation_kind kind,
    const cancellation_predicate &cancelled) {
  if (index_names.empty()) return nullptr;

  std::lock_guard<std::mutex> guard(g_reservation_mutex);
  if (reservation_wait_cancelled(cancelled) ||
      !index_reservation_available(index_names, kind)) {
    return nullptr;
  }
  return grant_index_reservation_locked(index_names, kind);
}

std::shared_ptr<reservation_lease> acquire_catalog_reservation(
    reservation_kind kind, const cancellation_predicate &cancelled) {
  auto lease =
      std::make_shared<reservation_lease>(kind, std::vector<std::string>{});
  std::unique_lock<std::mutex> guard(g_reservation_mutex);
  if (kind == reservation_kind::kCatalogWrite) {
    ++g_waiting_catalog_writers;
    while (g_catalog_writer || g_catalog_readers != 0) {
      if (reservation_wait_cancelled(cancelled)) {
        --g_waiting_catalog_writers;
        guard.unlock();
        g_reservation_changed.notify_all();
        return nullptr;
      }
      g_reservation_changed.wait_for(guard, k_reservation_wait_poll_interval);
    }
    if (reservation_wait_cancelled(cancelled)) {
      --g_waiting_catalog_writers;
      guard.unlock();
      g_reservation_changed.notify_all();
      return nullptr;
    }
    --g_waiting_catalog_writers;
    g_catalog_writer = true;
  } else {
    while (g_catalog_writer || g_waiting_catalog_writers != 0) {
      if (reservation_wait_cancelled(cancelled)) return nullptr;
      g_reservation_changed.wait_for(guard, k_reservation_wait_poll_interval);
    }
    if (reservation_wait_cancelled(cancelled)) return nullptr;
    ++g_catalog_readers;
  }
  lease->activate();
  return lease;
}

}  // namespace

bool index_sets_equal(const std::vector<std::string> &lhs,
                      const std::vector<std::string> &rhs) {
  return normalized_index_names(lhs) == normalized_index_names(rhs);
}

bool encode_operation_payload(const operation_payload &payload,
                              std::string *encoded) {
  if (encoded == nullptr || !valid_config_change(payload.config)) return false;
  if (payload.unsigned_values.size() > std::numeric_limits<uint64_t>::max() ||
      payload.string_values.size() > std::numeric_limits<uint64_t>::max()) {
    return false;
  }

  encoded->clear();
  encoded->push_back(static_cast<char>(k_operation_payload_version));
  encoded->push_back(static_cast<char>(payload.config));
  append_uint64(static_cast<uint64_t>(payload.unsigned_values.size()), encoded);
  for (uint64_t value : payload.unsigned_values) append_uint64(value, encoded);
  append_uint64(static_cast<uint64_t>(payload.string_values.size()), encoded);
  for (const std::string &value : payload.string_values) {
    if (!append_sized_string(value, encoded)) return false;
  }
  return append_sized_string(payload.binary_value, encoded);
}

bool decode_operation_payload(const std::string &encoded,
                              operation_payload *payload) {
  if (payload == nullptr || encoded.size() < 2 ||
      static_cast<uint8_t>(encoded[0]) != k_operation_payload_version) {
    return false;
  }

  operation_payload decoded;
  decoded.config = static_cast<config_change>(
      static_cast<unsigned char>(encoded[1]));
  if (!valid_config_change(decoded.config)) return false;

  size_t offset = 2;
  uint64_t unsigned_count = 0;
  if (!read_uint64(encoded, &offset, &unsigned_count) ||
      !count_fits_minimum_frame(encoded, offset, unsigned_count,
                                sizeof(uint64_t), 2 * sizeof(uint64_t))) {
    return false;
  }
  decoded.unsigned_values.reserve(static_cast<size_t>(unsigned_count));
  for (uint64_t i = 0; i < unsigned_count; ++i) {
    uint64_t value = 0;
    if (!read_uint64(encoded, &offset, &value)) return false;
    decoded.unsigned_values.push_back(value);
  }

  uint64_t string_count = 0;
  if (!read_uint64(encoded, &offset, &string_count) ||
      !count_fits_minimum_frame(encoded, offset, string_count, sizeof(uint64_t),
                                sizeof(uint64_t))) {
    return false;
  }
  decoded.string_values.reserve(static_cast<size_t>(string_count));
  for (uint64_t i = 0; i < string_count; ++i) {
    std::string value;
    if (!read_sized_string(encoded, &offset, &value)) return false;
    decoded.string_values.push_back(std::move(value));
  }
  if (!read_sized_string(encoded, &offset, &decoded.binary_value) ||
      offset != encoded.size()) {
    return false;
  }
  *payload = std::move(decoded);
  return true;
}

bool publication_guard::lock_indexes(
    const std::vector<std::string> &index_names,
    const cancellation_predicate &cancelled) {
  if (owns_lock()) return false;
  m_lease = acquire_index_reservation(normalized_index_names(index_names),
                                      reservation_kind::kIndexWrite,
                                      cancelled);
  return m_lease != nullptr;
}

bool publication_guard::try_lock_indexes(
    const std::vector<std::string> &index_names,
    const cancellation_predicate &cancelled) {
  if (owns_lock()) return false;
  m_lease = try_acquire_index_reservation(normalized_index_names(index_names),
                                          reservation_kind::kIndexWrite,
                                          cancelled);
  return m_lease != nullptr;
}

bool publication_guard::lock_catalog(
    const cancellation_predicate &cancelled) {
  if (owns_lock()) return false;
  m_lease =
      acquire_catalog_reservation(reservation_kind::kCatalogWrite, cancelled);
  return m_lease != nullptr;
}

bool publication_guard::owns_lock() const { return m_lease != nullptr; }

bool read_guard::lock_index(const std::string &index_name,
                            const cancellation_predicate &cancelled) {
  if (owns_lock() || index_name.empty()) return false;
  m_lease = acquire_index_reservation({index_name},
                                      reservation_kind::kIndexRead, cancelled);
  return m_lease != nullptr;
}

bool read_guard::lock_catalog(const cancellation_predicate &cancelled) {
  if (owns_lock()) return false;
  m_lease =
      acquire_catalog_reservation(reservation_kind::kCatalogRead, cancelled);
  return m_lease != nullptr;
}

bool read_guard::owns_lock() const { return m_lease != nullptr; }

}  // namespace vector_statement_publication
