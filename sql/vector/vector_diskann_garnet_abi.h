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

#ifndef SQL_VECTOR_VECTOR_DISKANN_GARNET_ABI_H_
#define SQL_VECTOR_VECTOR_DISKANN_GARNET_ABI_H_

#include <cstddef>
#include <cstdint>

namespace vector_index::diskann_garnet_abi {

/** MySQL-Vector Garnet adapter ABI revision. */
inline constexpr uint32_t k_diskann_garnet_abi_version = 2;

/** Garnet ABI v2 types used by the serial fallback. */
using diskann_read_data_callback =
    void (*)(uint32_t, void *, const uint8_t *, std::size_t);
using diskann_rmw_data_callback = void (*)(void *, uint8_t *, std::size_t);
using diskann_read_callback =
    void (*)(uint64_t, uint32_t, uint32_t, const uint8_t *, std::size_t,
             diskann_read_data_callback, void *);
using diskann_write_callback = bool (*)(uint64_t, const uint8_t *, std::size_t,
                                        const uint8_t *, std::size_t);
using diskann_delete_callback = bool (*)(uint64_t, const uint8_t *, std::size_t);
using diskann_read_modify_write_callback =
    bool (*)(uint64_t, const uint8_t *, std::size_t, std::size_t,
             diskann_rmw_data_callback, void *);
using diskann_filter_callback =
    bool (*)(uint64_t, const uint8_t *, std::size_t);
using diskann_log_callback = void (*)(uint64_t, const uint8_t *, std::size_t);

using diskann_create_index_fn =
    const void *(*)(uint64_t, uint32_t, uint32_t, uint32_t, int32_t, uint32_t,
                    uint32_t, diskann_read_callback, diskann_write_callback,
                    diskann_delete_callback,
                    diskann_read_modify_write_callback,
                    diskann_filter_callback, diskann_log_callback, bool *);
using diskann_drop_index_fn = void (*)(uint64_t, const void *);
using diskann_insert_fn = uint8_t (*)(uint64_t, const void *, const uint8_t *,
                                     std::size_t, const uint8_t *, std::size_t,
                                     const uint8_t *, std::size_t);
using diskann_build_quant_table_fn = bool (*)(uint64_t, const void *);
using diskann_backfill_quant_vectors_fn =
    bool (*)(uint64_t, const void *, std::size_t, std::size_t);
using diskann_random_members_fn =
    bool (*)(uint64_t, const void *, uint32_t, uint8_t *, std::size_t);
using diskann_search_neighbors_fn = int32_t (*)(
    uint64_t, const void *, const uint8_t *, std::size_t, uint8_t *,
    std::size_t, float *, std::size_t, void *);
using diskann_search_vector_fn = int32_t (*)(
    uint64_t, const void *, const uint8_t *, std::size_t, float, uint32_t,
    const uint8_t *, std::size_t, std::size_t, uint8_t *, std::size_t, float *,
    std::size_t, uint32_t, void *);
using diskann_remove_fn =
    bool (*)(uint64_t, const void *, const uint8_t *, std::size_t);
using diskann_card_fn = uint64_t (*)(uint64_t, const void *);

}  // namespace vector_index::diskann_garnet_abi

#endif  // SQL_VECTOR_VECTOR_DISKANN_GARNET_ABI_H_
