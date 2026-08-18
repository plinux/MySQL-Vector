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

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "sql/vector/vector_index_diagnostics.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_index_diagnostics_unittest {

constexpr const char *kDiagnosticsPathEnv = "MYSQL_VECTOR_DIAG_FILE";

TEST(VectorIndexDiagnosticsTest, FormatsJsonEventWithEscapedStrings) {
  const std::string event = vector_index_diagnostics::format_event_for_testing(
      "truth_store_save", {{"artifact", "committed\"v1"}},
      {{"rows", 42}, {"payload_bytes", 1024}});

  EXPECT_NE(std::string::npos, event.find("\"event\":\"truth_store_save\""));
  EXPECT_NE(std::string::npos, event.find("\"artifact\":\"committed\\\"v1\""));
  EXPECT_NE(std::string::npos, event.find("\"rows\":42"));
  EXPECT_NE(std::string::npos, event.find("\"payload_bytes\":1024"));
}

TEST(VectorIndexDiagnosticsTest, FormatsNullAndControlCharacterFields) {
  const std::string event = vector_index_diagnostics::format_event_for_testing(
      nullptr, {{nullptr, "\\\"\n\r\t"}}, {{nullptr, 9}});

  EXPECT_NE(std::string::npos, event.find("\"event\":\"\""));
  EXPECT_NE(std::string::npos, event.find("\"\":\"\\\\\\\"\\n\\r\\t\""));
  EXPECT_NE(std::string::npos, event.find("\"\":9"));
}

TEST(VectorIndexDiagnosticsTest, AppendsJsonEventWhenPathIsConfigured) {
  const std::string path =
      std::string(::testing::TempDir()) + "vector_index_diagnostics.jsonl";
  std::remove(path.c_str());

  ASSERT_TRUE(vector_index_diagnostics::append_event_for_testing(
      path.c_str(), "commit_summary", {{"scope", "explicit"}},
      {{"txn_id", 7}, {"pending_changes", 11}}));

  std::ifstream input(path);
  ASSERT_TRUE(input.good());
  std::string line;
  std::getline(input, line);
  EXPECT_NE(std::string::npos, line.find("\"event\":\"commit_summary\""));
  EXPECT_NE(std::string::npos, line.find("\"scope\":\"explicit\""));
  EXPECT_NE(std::string::npos, line.find("\"txn_id\":7"));
  EXPECT_NE(std::string::npos, line.find("\"pending_changes\":11"));

  std::remove(path.c_str());
}

TEST(VectorIndexDiagnosticsTest, AppendEventRejectsInvalidPaths) {
  EXPECT_FALSE(vector_index_diagnostics::append_event_for_testing(
      nullptr, "bad_path", {}, {}));
  EXPECT_FALSE(vector_index_diagnostics::append_event_for_testing(
      "", "bad_path", {}, {}));
  EXPECT_FALSE(vector_index_diagnostics::append_event_for_testing(
      "/no/such/vector/diagnostics/file.jsonl", "bad_path", {}, {}));
}

TEST(VectorIndexDiagnosticsTest, RecordEventHonorsEnvironmentSwitch) {
  vector_gunit::EnvVarGuard guard(kDiagnosticsPathEnv);
  unsetenv(kDiagnosticsPathEnv);

  const std::string path =
      std::string(::testing::TempDir()) + "vector_index_diagnostics_env.jsonl";
  std::remove(path.c_str());

  vector_index_diagnostics::record_event("disabled", {}, {{"value", 1}});
  EXPECT_FALSE(std::ifstream(path).good());

  setenv(kDiagnosticsPathEnv, "", 1);
  vector_index_diagnostics::record_event("empty_path", {}, {{"value", 1}});
  EXPECT_FALSE(std::ifstream(path).good());

  setenv(kDiagnosticsPathEnv, path.c_str(), 1);
  vector_index_diagnostics::record_event("enabled", {{"mode", "env"}},
                                         {{"value", 2}});

  std::ifstream input(path);
  ASSERT_TRUE(input.good());
  std::string line;
  std::getline(input, line);
  EXPECT_NE(std::string::npos, line.find("\"event\":\"enabled\""));
  EXPECT_NE(std::string::npos, line.find("\"mode\":\"env\""));
  EXPECT_NE(std::string::npos, line.find("\"value\":2"));

  std::remove(path.c_str());
}

}  // namespace vector_index_diagnostics_unittest
