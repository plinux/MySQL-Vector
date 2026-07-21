/* Copyright (c) 2026, Oracle and/or its affiliates.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0,
   as published by the Free Software Foundation.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA */

#ifndef MYSQL_VECTOR_DISKANN_OFFLINE_CHECKED_IO_H
#define MYSQL_VECTOR_DISKANN_OFFLINE_CHECKED_IO_H

#include <cstddef>
#include <cstdint>
#include <ios>
#include <limits>

namespace mysql_vector_diskann_offline {

struct checked_payload_layout {
  size_t element_count{0};
  size_t payload_bytes{0};
  uintmax_t file_bytes{0};
};

inline bool make_checked_payload_layout(uint64_t element_count,
                                        size_t element_size,
                                        size_t header_bytes,
                                        checked_payload_layout *layout) {
  if (layout == nullptr || element_size == 0 ||
      element_count > std::numeric_limits<size_t>::max() / element_size) {
    return false;
  }

  const size_t payload_bytes =
      static_cast<size_t>(element_count) * element_size;
  if (payload_bytes >
          static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) ||
      payload_bytes > std::numeric_limits<uintmax_t>::max() - header_bytes) {
    return false;
  }

  layout->element_count = static_cast<size_t>(element_count);
  layout->payload_bytes = payload_bytes;
  layout->file_bytes = static_cast<uintmax_t>(header_bytes) + payload_bytes;
  return true;
}

}  // namespace mysql_vector_diskann_offline

#endif  // MYSQL_VECTOR_DISKANN_OFFLINE_CHECKED_IO_H
