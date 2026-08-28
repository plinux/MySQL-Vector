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

#ifndef DD_TABLES__VECTOR_INDEX_TRUTH_TABLES_INCLUDED
#define DD_TABLES__VECTOR_INDEX_TRUTH_TABLES_INCLUDED

#include "sql/dd/impl/types/object_table_impl.h"

namespace dd {
namespace tables {

class Vector_index_truth_metadata : public Object_table_impl {
 public:
  Vector_index_truth_metadata();
  static const Vector_index_truth_metadata &instance();
};

class Vector_index_truth_committed : public Object_table_impl {
 public:
  Vector_index_truth_committed();
  static const Vector_index_truth_committed &instance();
};

class Vector_index_truth_manifest : public Object_table_impl {
 public:
  Vector_index_truth_manifest();
  static const Vector_index_truth_manifest &instance();
};

class Vector_index_truth_changelog : public Object_table_impl {
 public:
  Vector_index_truth_changelog();
  static const Vector_index_truth_changelog &instance();
};

class Vector_index_truth_prepared : public Object_table_impl {
 public:
  Vector_index_truth_prepared();
  static const Vector_index_truth_prepared &instance();
};

class Vector_index_publication_intents : public Object_table_impl {
 public:
  Vector_index_publication_intents();
  static const Vector_index_publication_intents &instance();
};

class Vector_index_truth_segment_tasks : public Object_table_impl {
 public:
  Vector_index_truth_segment_tasks();
  static const Vector_index_truth_segment_tasks &instance();
};

class Vector_index_truth_store_quarantine : public Object_table_impl {
 public:
  Vector_index_truth_store_quarantine();
  static const Vector_index_truth_store_quarantine &instance();
};

}  // namespace tables
}  // namespace dd

#endif  // DD_TABLES__VECTOR_INDEX_TRUTH_TABLES_INCLUDED
