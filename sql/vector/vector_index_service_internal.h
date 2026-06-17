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

#ifndef SQL_VECTOR_INDEX_SERVICE_INTERNAL_INCLUDED
#define SQL_VECTOR_INDEX_SERVICE_INTERNAL_INCLUDED

#include <memory>
#include <string>

#include "sql/vector/vector_index_backend.h"
#include "sql/vector/vector_index_service.h"

namespace vector_index::detail {

std::unique_ptr<backend> build_backend_from_config(
    const std::string &index_name,
    const vector_index::index_service::index_config &config);

inline bool can_rebuild_after_search_failure(
    const vector_index::index_service::index_config &config) {
  return config.mode == backend_mode::kMemory &&
         config.provider == backend_provider::kHnswlib;
}

}  // namespace vector_index::detail

#endif  // SQL_VECTOR_INDEX_SERVICE_INTERNAL_INCLUDED
