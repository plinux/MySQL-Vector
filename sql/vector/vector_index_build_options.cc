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

#include "sql/vector/vector_index_build_options.h"

ulong opt_vector_hnsw_build_threads = 0;
ulong opt_vector_faiss_build_threads = 0;
ulong opt_vector_diskann_build_threads = 0;
ulong opt_vector_default_library =
    static_cast<ulong>(vector_index::vector_default_library::kNone);

namespace vector_index {

vector_default_library global_vector_default_library() {
  switch (static_cast<vector_default_library>(opt_vector_default_library)) {
    case vector_default_library::kNone:
      return vector_default_library::kNone;
    case vector_default_library::kDiskAnn:
      return vector_default_library::kDiskAnn;
    case vector_default_library::kHnsw:
      return vector_default_library::kHnsw;
    case vector_default_library::kFaiss:
      return vector_default_library::kFaiss;
  }
  return vector_default_library::kNone;
}

}  // namespace vector_index
