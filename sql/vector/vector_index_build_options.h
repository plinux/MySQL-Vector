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

#ifndef SQL_VECTOR_VECTOR_INDEX_BUILD_OPTIONS_H
#define SQL_VECTOR_VECTOR_INDEX_BUILD_OPTIONS_H

#include "my_inttypes.h"

/**
  Global vector index build-thread defaults.

  The vector module owns these option values so backend, registry and SQL
  integration commits can share one storage location while exposing each
  backend-specific sysvar at the first commit where the backend becomes usable.
*/
extern ulong opt_vector_hnsw_build_threads;
extern ulong opt_vector_faiss_build_threads;
extern ulong opt_vector_diskann_build_threads;
extern ulong opt_vector_default_library;
extern ulonglong opt_vector_entry_cache_size;
extern ulonglong opt_vector_pending_cache_size;
extern ulonglong opt_vector_build_memory_size;
extern ulonglong opt_vector_diskann_build_memory_size;
extern ulonglong opt_vector_diskann_raw_segment_size;
extern ulonglong opt_vector_faiss_train_size;
extern ulonglong opt_vector_hnsw_index_memory_size;

namespace vector_index {

enum class vector_default_library : ulong {
  kNone = 0,
  kDiskAnn = 1,
  kHnsw = 2,
  kFaiss = 3
};

/** Return the global default vector library. */
vector_default_library global_vector_default_library();

}  // namespace vector_index

#endif  // SQL_VECTOR_VECTOR_INDEX_BUILD_OPTIONS_H
