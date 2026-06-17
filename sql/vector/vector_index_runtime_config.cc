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

#include "sql/vector/vector_index_runtime_config.h"

#include <algorithm>

namespace vector_index {

bool runtime_provider_accepts_mode(backend_provider provider,
                                   backend_mode mode) {
  switch (provider) {
    case backend_provider::kNative:
    case backend_provider::kFaiss:
      return mode == backend_mode::kMemory || mode == backend_mode::kExternal;
    case backend_provider::kDiskAnn:
      return mode == backend_mode::kExternal;
    case backend_provider::kHnswlib:
      return mode == backend_mode::kMemory;
  }
  return false;
}

bool runtime_common_config_valid(const runtime_common_config &config) {
  return config.dimension > 0 &&
         runtime_provider_accepts_mode(config.provider, config.mode);
}

faiss_runtime_index_kind faiss_runtime_kind_from_params(uint32_t nlist,
                                                        uint32_t pq_m,
                                                        uint32_t pq_bits) {
  if (nlist != 0 && pq_m != 0 && pq_bits != 0) {
    return faiss_runtime_index_kind::kIvfPq;
  }
  if (nlist != 0) return faiss_runtime_index_kind::kIvfFlat;
  return faiss_runtime_index_kind::kHnsw;
}

uint32_t resolve_runtime_threads(uint32_t statement_threads,
                                 uint32_t global_threads) {
  const uint32_t selected =
      statement_threads == 0 ? global_threads : statement_threads;
  return std::min<uint32_t>(selected, k_max_build_threads);
}

}  // namespace vector_index
