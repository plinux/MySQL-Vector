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

#ifndef UNITTEST_GUNIT_VECTOR_TEST_UTILS_H
#define UNITTEST_GUNIT_VECTOR_TEST_UTILS_H

#include <gtest/gtest.h>

#include "my_dbug.h"
#include "my_sys.h"
#include "sql/handler.h"
#include "sql/mysqld.h"
#include "sql/xa.h"

namespace vector_gunit {

class ScopedDebugFlag {
 public:
  explicit ScopedDebugFlag(const char *flags [[maybe_unused]]) {
    DBUG_SET(flags);
  }
  ~ScopedDebugFlag() { DBUG_SET(""); }

  ScopedDebugFlag(const ScopedDebugFlag &) = delete;
  ScopedDebugFlag &operator=(const ScopedDebugFlag &) = delete;
};

class NativeProviderGuard {
 public:
  explicit NativeProviderGuard(bool supported = true)
      : m_original(vector_index::native_provider_supported_for_testing()) {
    vector_index::set_native_provider_supported_for_testing(supported);
  }
  ~NativeProviderGuard() {
    vector_index::set_native_provider_supported_for_testing(m_original);
  }

  NativeProviderGuard(const NativeProviderGuard &) = delete;
  NativeProviderGuard &operator=(const NativeProviderGuard &) = delete;

 private:
  bool m_original;
};

inline Xa_state_list::instantiation_tuple make_xa_state_list_for_testing() {
  const ulong original_tc_log_page_size = tc_log_page_size;
  if (tc_log_page_size == 0) tc_log_page_size = my_getpagesize();
  auto xa_state_list = Xa_state_list::new_instance();
  tc_log_page_size = original_tc_log_page_size;
  return xa_state_list;
}

}  // namespace vector_gunit

#if defined(NDEBUG)
#define VECTOR_SCOPED_DEBUG_FLAG(name, flags)                           \
  GTEST_SKIP() << "Requires DBUG fault injection, which is compiled out " \
                  "in Release builds."
#else
#define VECTOR_SCOPED_DEBUG_FLAG(name, flags) \
  vector_gunit::ScopedDebugFlag name(flags)
#endif

#endif  // UNITTEST_GUNIT_VECTOR_TEST_UTILS_H
