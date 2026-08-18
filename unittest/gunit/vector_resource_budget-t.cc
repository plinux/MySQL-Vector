/* Copyright (c) 2026, Oracle and/or its affiliates.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0,
   as published by the Free Software Foundation.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

#include "sql/vector/vector_resource_budget.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <string>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "sql/vector/vector_index_build_options.h"

namespace vector_resource_budget_unittest {

namespace {

constexpr uint64_t kMiB = 1024ULL * 1024ULL;

class option_guard {
 public:
  option_guard(ulonglong *option, ulonglong value)
      : m_option(option), m_previous(*option) {
    *m_option = value;
  }

  ~option_guard() { *m_option = m_previous; }

 private:
  ulonglong *m_option;
  ulonglong m_previous;
};

void write_text_file(const std::filesystem::path &path,
                     const std::string &value) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  ASSERT_TRUE(output.is_open());
  output << value;
  ASSERT_TRUE(output.good());
}

vector_index::resource_probe_snapshot fixed_probe(uint64_t headroom,
                                                  uint32_t cpu_slots) {
  vector_index::resource_probe_snapshot probe;
  probe.source = vector_index::resource_probe_source::kHost;
  probe.host_available_memory = headroom;
  probe.memory_headroom = headroom;
  probe.hardware_cpu_slots = cpu_slots;
  probe.effective_cpu_slots = cpu_slots;
  return probe;
}

}  // namespace

TEST(VectorResourceBudgetTest, ParsesLimitsAndCpuSets) {
  uint64_t value = 0;
  bool unlimited = false;
  EXPECT_TRUE(vector_index::parse_resource_limit_for_testing(
      "1073741824\n", &value, &unlimited));
  EXPECT_EQ(1073741824U, value);
  EXPECT_FALSE(unlimited);

  EXPECT_TRUE(vector_index::parse_resource_limit_for_testing(
      "max\n", &value, &unlimited));
  EXPECT_TRUE(unlimited);
  EXPECT_FALSE(vector_index::parse_resource_limit_for_testing(
      "invalid", &value, &unlimited));

  uint32_t cpu_count = 0;
  EXPECT_TRUE(vector_index::count_cpu_set_for_testing(
      "0-3,8,10-11\n", &cpu_count));
  EXPECT_EQ(7U, cpu_count);
  EXPECT_FALSE(
      vector_index::count_cpu_set_for_testing("3-1", &cpu_count));
}

TEST(VectorResourceBudgetTest, RejectsMalformedLimitsAndCpuSets) {
  uint64_t value = 0;
  bool unlimited = false;
  EXPECT_FALSE(
      vector_index::parse_resource_limit_for_testing("1", nullptr, &unlimited));
  EXPECT_FALSE(
      vector_index::parse_resource_limit_for_testing("1", &value, nullptr));
  EXPECT_FALSE(
      vector_index::parse_resource_limit_for_testing("", &value, &unlimited));
  EXPECT_FALSE(
      vector_index::parse_resource_limit_for_testing("-1", &value, &unlimited));
  EXPECT_FALSE(
      vector_index::parse_resource_limit_for_testing("1K", &value, &unlimited));
  EXPECT_FALSE(vector_index::parse_resource_limit_for_testing(
      "18446744073709551616", &value, &unlimited));

  uint32_t cpu_count = 0;
  EXPECT_FALSE(vector_index::count_cpu_set_for_testing("0", nullptr));
  EXPECT_FALSE(vector_index::count_cpu_set_for_testing("", &cpu_count));
  EXPECT_FALSE(vector_index::count_cpu_set_for_testing(",", &cpu_count));
  EXPECT_FALSE(vector_index::count_cpu_set_for_testing("0,,1", &cpu_count));
  EXPECT_FALSE(vector_index::count_cpu_set_for_testing("0-1-2", &cpu_count));
  EXPECT_FALSE(
      vector_index::count_cpu_set_for_testing("0-4294967295", &cpu_count));
  EXPECT_TRUE(vector_index::count_cpu_set_for_testing("7", &cpu_count));
  EXPECT_EQ(1U, cpu_count);
}

TEST(VectorResourceBudgetTest, ProbesCgroupV2BeforeHostFallback) {
  const std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "vector_resource_v2";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup", "0::/mysql.slice/server\n");
  write_text_file(root / "cgroup/mysql.slice/server/memory.max",
                  std::to_string(1024ULL * kMiB));
  write_text_file(root / "cgroup/mysql.slice/server/memory.current",
                  std::to_string(256ULL * kMiB));
  write_text_file(root / "cgroup/mysql.slice/server/cpu.max", "200000 100000\n");
  write_text_file(root / "cgroup/mysql.slice/server/cpuset.cpus.effective",
                  "0-3\n");

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 2ULL * 1024ULL * kMiB, 8, 64ULL * kMiB, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kCgroupV2, probe.source);
  EXPECT_EQ(1024ULL * kMiB, probe.memory_limit);
  EXPECT_EQ(256ULL * kMiB, probe.memory_current);
  EXPECT_EQ(768ULL * kMiB, probe.memory_headroom);
  EXPECT_EQ(2U, probe.quota_cpu_slots);
  EXPECT_EQ(4U, probe.cpuset_cpu_slots);
  EXPECT_EQ(2U, probe.effective_cpu_slots);
  EXPECT_EQ(64ULL * kMiB, probe.process_rss);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, ProbesUnlimitedCgroupV2WithOptionalCpuFiles) {
  const std::filesystem::path root = std::filesystem::path(testing::TempDir()) /
                                     "vector_resource_v2_unlimited";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup", "0::/mysql.slice/server\n");
  write_text_file(root / "cgroup/mysql.slice/server/memory.max", "max\n");
  write_text_file(root / "cgroup/mysql.slice/server/memory.current",
                  std::to_string(256ULL * kMiB));
  write_text_file(root / "cgroup/mysql.slice/server/cpu.max", "max 100000\n");
  write_text_file(root / "cgroup/mysql.slice/server/cpuset.cpus.effective",
                  "invalid\n");

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 768ULL * kMiB, 0, 32ULL * kMiB, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kCgroupV2, probe.source);
  EXPECT_EQ(0U, probe.memory_limit);
  EXPECT_EQ(256ULL * kMiB, probe.memory_current);
  EXPECT_EQ(768ULL * kMiB, probe.memory_headroom);
  EXPECT_EQ(0U, probe.quota_cpu_slots);
  EXPECT_EQ(0U, probe.cpuset_cpu_slots);
  EXPECT_EQ(1U, probe.effective_cpu_slots);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, ProbesFiniteCgroupV2AncestorLimits) {
  const std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "vector_resource_v2_parent";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup", "0::/mysql.slice/server\n");
  write_text_file(root / "cgroup/mysql.slice/server/memory.max", "max\n");
  write_text_file(root / "cgroup/mysql.slice/server/memory.current",
                  std::to_string(256ULL * kMiB));
  write_text_file(root / "cgroup/mysql.slice/server/cpu.max", "max 100000\n");
  write_text_file(root / "cgroup/mysql.slice/server/cpuset.cpus.effective",
                  "0-7\n");
  write_text_file(root / "cgroup/mysql.slice/memory.max",
                  std::to_string(2ULL * 1024ULL * kMiB));
  write_text_file(root / "cgroup/mysql.slice/memory.current",
                  std::to_string(1536ULL * kMiB));
  write_text_file(root / "cgroup/mysql.slice/cpu.max", "400000 100000\n");

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 4ULL * 1024ULL * kMiB, 16, 64ULL * kMiB, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kCgroupV2, probe.source);
  EXPECT_EQ(2ULL * 1024ULL * kMiB, probe.memory_limit);
  EXPECT_EQ(1536ULL * kMiB, probe.memory_current);
  EXPECT_EQ(512ULL * kMiB, probe.memory_headroom);
  EXPECT_EQ(4U, probe.quota_cpu_slots);
  EXPECT_EQ(8U, probe.cpuset_cpu_slots);
  EXPECT_EQ(4U, probe.effective_cpu_slots);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, ChoosesTightestCgroupV2AncestorLimits) {
  const std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "vector_resource_v2_tightest";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup", "0::/group/parent/leaf\n");
  write_text_file(root / "cgroup/group/parent/leaf/memory.max",
                  std::to_string(1024ULL * kMiB));
  write_text_file(root / "cgroup/group/parent/leaf/memory.current",
                  std::to_string(128ULL * kMiB));
  write_text_file(root / "cgroup/group/parent/leaf/cpu.max",
                  "800000 100000\n");
  write_text_file(root / "cgroup/group/parent/leaf/cpuset.cpus.effective",
                  "0-7\n");
  write_text_file(root / "cgroup/group/parent/memory.max",
                  std::to_string(768ULL * kMiB));
  write_text_file(root / "cgroup/group/parent/memory.current",
                  std::to_string(512ULL * kMiB));
  write_text_file(root / "cgroup/group/parent/cpu.max", "300000 100000\n");
  write_text_file(root / "cgroup/group/memory.max",
                  std::to_string(512ULL * kMiB));
  write_text_file(root / "cgroup/group/memory.current",
                  std::to_string(128ULL * kMiB));
  write_text_file(root / "cgroup/group/cpu.max", "600000 100000\n");

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 2ULL * 1024ULL * kMiB, 16, 64ULL * kMiB, &probe));
  EXPECT_EQ(768ULL * kMiB, probe.memory_limit);
  EXPECT_EQ(512ULL * kMiB, probe.memory_current);
  EXPECT_EQ(256ULL * kMiB, probe.memory_headroom);
  EXPECT_EQ(3U, probe.quota_cpu_slots);
  EXPECT_EQ(8U, probe.cpuset_cpu_slots);
  EXPECT_EQ(3U, probe.effective_cpu_slots);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, IgnoresMalformedCgroupV2Ancestors) {
  const std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "vector_resource_v2_bad_parent";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup", "0::/group/leaf\n");
  write_text_file(root / "cgroup/group/leaf/memory.max",
                  std::to_string(512ULL * kMiB));
  write_text_file(root / "cgroup/group/leaf/memory.current",
                  std::to_string(128ULL * kMiB));
  write_text_file(root / "cgroup/group/leaf/cpu.max", "200000 100000\n");
  write_text_file(root / "cgroup/group/memory.max", "invalid\n");
  write_text_file(root / "cgroup/group/memory.current",
                  std::to_string(64ULL * kMiB));
  write_text_file(root / "cgroup/group/cpu.max", "invalid 100000\n");

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 1024ULL * kMiB, 8, 64ULL * kMiB, &probe));
  EXPECT_EQ(512ULL * kMiB, probe.memory_limit);
  EXPECT_EQ(128ULL * kMiB, probe.memory_current);
  EXPECT_EQ(384ULL * kMiB, probe.memory_headroom);
  EXPECT_EQ(2U, probe.quota_cpu_slots);
  EXPECT_EQ(2U, probe.effective_cpu_slots);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, ReportsExhaustedFiniteCgroupV2) {
  const std::filesystem::path root = std::filesystem::path(testing::TempDir()) /
                                     "vector_resource_v2_exhausted";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup", "0::/mysql\n");
  write_text_file(root / "cgroup/mysql/memory.max",
                  std::to_string(128ULL * kMiB));
  write_text_file(root / "cgroup/mysql/memory.current",
                  std::to_string(256ULL * kMiB));
  write_text_file(root / "cgroup/memory.max",
                  std::to_string(1024ULL * kMiB));
  write_text_file(root / "cgroup/memory.current",
                  std::to_string(128ULL * kMiB));

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(
      vector_index::probe_build_resources_for_testing(paths, 0, 2, 0, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kCgroupV2, probe.source);
  EXPECT_EQ(0U, probe.memory_headroom);
  EXPECT_EQ(2U, probe.effective_cpu_slots);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, HandlesRelativeCgroupV2AndInvalidCpuControls) {
  const std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "vector_resource_v2_controls";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup",
                  "5:memory\n4:cpu,:/unused\n0::mysql\n");
  write_text_file(root / "cgroup/mysql/memory.max",
                  std::to_string(512ULL * kMiB));
  write_text_file(root / "cgroup/mysql/memory.current",
                  std::to_string(128ULL * kMiB));

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;

  write_text_file(root / "cgroup/mysql/cpu.max", "invalid 100000\n");
  write_text_file(root / "cgroup/mysql/cpuset.cpus.effective", "a-1\n");
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 1024ULL * kMiB, 8, 16ULL * kMiB, &probe));
  EXPECT_EQ(0U, probe.quota_cpu_slots);
  EXPECT_EQ(0U, probe.cpuset_cpu_slots);

  write_text_file(root / "cgroup/mysql/cpu.max", "100000 invalid\n");
  write_text_file(root / "cgroup/mysql/cpuset.cpus.effective", "1-a\n");
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 1024ULL * kMiB, 8, 16ULL * kMiB, &probe));
  EXPECT_EQ(0U, probe.quota_cpu_slots);
  EXPECT_EQ(0U, probe.cpuset_cpu_slots);

  write_text_file(root / "cgroup/mysql/cpu.max", "0 100000\n");
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 1024ULL * kMiB, 8, 16ULL * kMiB, &probe));
  EXPECT_EQ(0U, probe.quota_cpu_slots);

  write_text_file(root / "cgroup/mysql/cpu.max", "100000 0\n");
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 1024ULL * kMiB, 8, 16ULL * kMiB, &probe));
  EXPECT_EQ(0U, probe.quota_cpu_slots);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, FallsBackForIncompleteCgroupV2MemoryFiles) {
  const std::filesystem::path root = std::filesystem::path(testing::TempDir()) /
                                     "vector_resource_v2_incomplete";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup", "0::/mysql\n");

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;

  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 768ULL * kMiB, 8, 16ULL * kMiB, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kHost, probe.source);

  write_text_file(root / "cgroup/mysql/memory.max",
                  std::to_string(512ULL * kMiB));
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 768ULL * kMiB, 8, 16ULL * kMiB, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kHost, probe.source);

  write_text_file(root / "cgroup/mysql/memory.current", "invalid\n");
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 768ULL * kMiB, 8, 16ULL * kMiB, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kHost, probe.source);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, ProbesCgroupV1BeforeHostFallback) {
  const std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "vector_resource_v1";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup",
                  "5:memory:/mysql\n4:cpu,cpuacct:/mysql\n3:cpuset:/mysql\n");
  write_text_file(root / "cgroup/memory/mysql/memory.limit_in_bytes",
                  std::to_string(512ULL * kMiB));
  write_text_file(root / "cgroup/memory/mysql/memory.usage_in_bytes",
                  std::to_string(128ULL * kMiB));
  write_text_file(root / "cgroup/cpu/mysql/cpu.cfs_quota_us", "300000\n");
  write_text_file(root / "cgroup/cpu/mysql/cpu.cfs_period_us", "100000\n");
  write_text_file(root / "cgroup/cpuset/mysql/cpuset.cpus", "2-5\n");

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 2ULL * 1024ULL * kMiB, 16, 32ULL * kMiB, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kCgroupV1, probe.source);
  EXPECT_EQ(384ULL * kMiB, probe.memory_headroom);
  EXPECT_EQ(3U, probe.quota_cpu_slots);
  EXPECT_EQ(4U, probe.cpuset_cpu_slots);
  EXPECT_EQ(3U, probe.effective_cpu_slots);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, ProbesUnlimitedCgroupV1WithoutCpuControllers) {
  const std::filesystem::path root = std::filesystem::path(testing::TempDir()) /
                                     "vector_resource_v1_unlimited";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup", "5:memory:/mysql\n");
  write_text_file(root / "cgroup/memory/mysql/memory.limit_in_bytes",
                  std::to_string(std::numeric_limits<uint64_t>::max()));
  write_text_file(root / "cgroup/memory/mysql/memory.usage_in_bytes",
                  std::to_string(128ULL * kMiB));

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 768ULL * kMiB, 16, 32ULL * kMiB, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kCgroupV1, probe.source);
  EXPECT_EQ(0U, probe.memory_limit);
  EXPECT_EQ(768ULL * kMiB, probe.memory_headroom);
  EXPECT_EQ(0U, probe.quota_cpu_slots);
  EXPECT_EQ(0U, probe.cpuset_cpu_slots);
  EXPECT_EQ(16U, probe.effective_cpu_slots);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, FallsBackWhenCgroupV1MemoryFilesAreMissing) {
  const std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "vector_resource_v1_missing";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup",
                  "5:memory:/mysql\n4:cpu:/mysql\n3:cpuset:/mysql\n");

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 512ULL * kMiB, 8, 16ULL * kMiB, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kHost, probe.source);
  EXPECT_EQ(512ULL * kMiB, probe.memory_headroom);
  EXPECT_EQ(8U, probe.effective_cpu_slots);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, HandlesCgroupV1PressureAndControllerFailures) {
  const std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "vector_resource_v1_controls";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup",
                  "5:memory,:/mysql\n4:cpu:/mysql\n3:cpuset:/mysql\n");
  write_text_file(root / "cgroup/memory/mysql/memory.limit_in_bytes",
                  std::to_string(128ULL * kMiB));
  write_text_file(root / "cgroup/memory/mysql/memory.usage_in_bytes",
                  std::to_string(256ULL * kMiB));
  write_text_file(root / "cgroup/cpu/mysql/cpu.cfs_quota_us", "0\n");
  write_text_file(root / "cgroup/cpu/mysql/cpu.cfs_period_us", "100000\n");

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(
      vector_index::probe_build_resources_for_testing(paths, 0, 8, 0, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kCgroupV1, probe.source);
  EXPECT_EQ(0U, probe.memory_headroom);
  EXPECT_EQ(0U, probe.quota_cpu_slots);
  EXPECT_EQ(0U, probe.cpuset_cpu_slots);

  write_text_file(root / "cgroup/cpu/mysql/cpu.cfs_quota_us", "100000\n");
  write_text_file(root / "cgroup/cpu/mysql/cpu.cfs_period_us", "0\n");
  ASSERT_TRUE(
      vector_index::probe_build_resources_for_testing(paths, 0, 8, 0, &probe));
  EXPECT_EQ(0U, probe.quota_cpu_slots);

  write_text_file(root / "cgroup/cpu/mysql/cpu.cfs_quota_us",
                  "9223372036854775808\n");
  write_text_file(root / "cgroup/cpu/mysql/cpu.cfs_period_us", "100000\n");
  ASSERT_TRUE(
      vector_index::probe_build_resources_for_testing(paths, 0, 8, 0, &probe));
  EXPECT_EQ(0U, probe.quota_cpu_slots);

  std::filesystem::remove(root / "cgroup/cpu/mysql/cpu.cfs_quota_us", ec);
  ASSERT_TRUE(
      vector_index::probe_build_resources_for_testing(paths, 0, 8, 0, &probe));
  EXPECT_EQ(0U, probe.quota_cpu_slots);

  write_text_file(root / "cgroup/cpu/mysql/cpu.cfs_quota_us", "100000\n");
  std::filesystem::remove(root / "cgroup/cpu/mysql/cpu.cfs_period_us", ec);
  ASSERT_TRUE(
      vector_index::probe_build_resources_for_testing(paths, 0, 8, 0, &probe));
  EXPECT_EQ(0U, probe.quota_cpu_slots);

  write_text_file(root / "cgroup/memory/mysql/memory.usage_in_bytes",
                  "invalid\n");
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 512ULL * kMiB, 8, 0, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kHost, probe.source);
  EXPECT_EQ(0U, probe.quota_cpu_slots);
  EXPECT_EQ(0U, probe.cpuset_cpu_slots);
  EXPECT_EQ(8U, probe.effective_cpu_slots);
  EXPECT_EQ(512ULL * kMiB, probe.memory_headroom);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, UsesHostWhenCgroupFilesAreUnavailable) {
  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = "/path/that/does/not/exist";
  paths.cgroup_root = "/path/that/does/not/exist";
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 768ULL * kMiB, 6, 24ULL * kMiB, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kHost, probe.source);
  EXPECT_EQ(768ULL * kMiB, probe.memory_headroom);
  EXPECT_EQ(6U, probe.effective_cpu_slots);
}

TEST(VectorResourceBudgetTest, UsesLinuxMemAvailableForHostHeadroom) {
  const std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "vector_resource_meminfo";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-meminfo",
                  "MemTotal:       1048576 kB\n"
                  "MemFree:          65536 kB\n"
                  "MemAvailable:    786432 kB\n");

  vector_index::resource_probe_paths paths;
  paths.proc_meminfo = (root / "proc-meminfo").string();
  paths.proc_self_cgroup = (root / "missing-cgroup").string();
  paths.cgroup_root = (root / "missing-root").string();
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 64ULL * kMiB, 8, 16ULL * kMiB, &probe));
  EXPECT_EQ(768ULL * kMiB, probe.host_available_memory);
  EXPECT_EQ(768ULL * kMiB, probe.memory_headroom);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, UsesHostForMalformedCgroupMembership) {
  const std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "vector_resource_membership";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup", "malformed\n::\n");

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;
  EXPECT_FALSE(vector_index::probe_build_resources_for_testing(
      paths, 512ULL * kMiB, 8, 16ULL * kMiB, nullptr));
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 512ULL * kMiB, 8, 16ULL * kMiB, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kHost, probe.source);
  EXPECT_EQ(512ULL * kMiB, probe.memory_headroom);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest, UsesHostWhenCgroupMemoryValuesAreMalformed) {
  const std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "vector_resource_invalid";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  write_text_file(root / "proc-self-cgroup", "0::/mysql.slice/server\n");
  write_text_file(root / "cgroup/mysql.slice/server/memory.max", "invalid\n");
  write_text_file(root / "cgroup/mysql.slice/server/memory.current",
                  std::to_string(256ULL * kMiB));
  write_text_file(root / "cgroup/mysql.slice/server/cpu.max", "200000 100000\n");

  vector_index::resource_probe_paths paths;
  paths.proc_self_cgroup = (root / "proc-self-cgroup").string();
  paths.cgroup_root = (root / "cgroup").string();
  vector_index::resource_probe_snapshot probe;
  ASSERT_TRUE(vector_index::probe_build_resources_for_testing(
      paths, 768ULL * kMiB, 6, 24ULL * kMiB, &probe));
  EXPECT_EQ(vector_index::resource_probe_source::kHost, probe.source);
  EXPECT_EQ(0U, probe.memory_limit);
  EXPECT_EQ(0U, probe.memory_current);
  EXPECT_EQ(768ULL * kMiB, probe.memory_headroom);
  EXPECT_EQ(0U, probe.quota_cpu_slots);
  EXPECT_EQ(6U, probe.effective_cpu_slots);
  std::filesystem::remove_all(root, ec);
}

TEST(VectorResourceBudgetTest,
     ExplicitBudgetRequiresFullMemoryGrantAndClipsCpuSlots) {
  option_guard build_memory(&opt_vector_build_memory_size, 600ULL * kMiB);
  option_guard reserve_memory(&opt_vector_build_memory_reserve_size,
                              100ULL * kMiB);
  vector_index::resource_budget_manager manager;
  manager.set_probe_for_testing(
      [] { return fixed_probe(640ULL * kMiB, 4); });

  vector_index::build_resource_lease lease;
  EXPECT_FALSE(manager.acquire({800ULL * kMiB, 8}, &lease));
  EXPECT_FALSE(lease);
  EXPECT_EQ(0U, manager.snapshot().active_builds);

  ASSERT_TRUE(manager.acquire({540ULL * kMiB, 8}, &lease));
  EXPECT_EQ(540ULL * kMiB, lease.memory_bytes());
  EXPECT_EQ(4U, lease.cpu_slots());

  const auto active = manager.snapshot();
  EXPECT_EQ(1U, active.active_builds);
  EXPECT_EQ(540ULL * kMiB, active.reserved_memory);
  EXPECT_EQ(4U, active.reserved_cpu_slots);
  EXPECT_EQ(540ULL * kMiB, active.effective_memory_budget);

  lease.reset();
  const auto released = manager.snapshot();
  EXPECT_EQ(0U, released.active_builds);
  EXPECT_EQ(0U, released.reserved_memory);
  EXPECT_EQ(0U, released.reserved_cpu_slots);
}

TEST(VectorResourceBudgetTest, AutomaticBudgetWaitsForLeaseRelease) {
  option_guard build_memory(&opt_vector_build_memory_size, 0);
  option_guard reserve_memory(&opt_vector_build_memory_reserve_size,
                              100ULL * kMiB);
  vector_index::resource_budget_manager manager;
  manager.set_probe_for_testing(
      [] { return fixed_probe(600ULL * kMiB, 2); });

  vector_index::build_resource_lease first;
  ASSERT_TRUE(manager.acquire({400ULL * kMiB, 2}, &first));

  std::promise<void> acquired;
  auto future = acquired.get_future();
  std::thread waiter([&] {
    vector_index::build_resource_lease second;
    if (manager.acquire({100ULL * kMiB, 1}, &second)) acquired.set_value();
  });

  EXPECT_EQ(std::future_status::timeout,
            future.wait_for(std::chrono::milliseconds(50)));
  EXPECT_EQ(1U, manager.snapshot().waiting_builds);
  first.reset();
  EXPECT_EQ(std::future_status::ready,
            future.wait_for(std::chrono::seconds(2)));
  waiter.join();
  EXPECT_EQ(0U, manager.snapshot().active_builds);
}

TEST(VectorResourceBudgetTest, WaitingBuildRechecksChangedEnvelope) {
  option_guard build_memory(&opt_vector_build_memory_size, 0);
  option_guard reserve_memory(&opt_vector_build_memory_reserve_size, 0);
  vector_index::resource_budget_manager manager;
  manager.set_probe_for_testing([] { return fixed_probe(512ULL * kMiB, 1); });

  vector_index::build_resource_lease first;
  ASSERT_TRUE(manager.acquire({400ULL * kMiB, 1}, &first));

  auto waiter = std::async(std::launch::async, [&manager] {
    vector_index::build_resource_lease lease;
    return manager.acquire({200ULL * kMiB, 1}, &lease);
  });
  EXPECT_EQ(std::future_status::timeout,
            waiter.wait_for(std::chrono::milliseconds(50)));
  EXPECT_EQ(1U, manager.snapshot().waiting_builds);

  manager.set_probe_for_testing([] { return fixed_probe(512ULL * kMiB, 1); });
  EXPECT_EQ(std::future_status::timeout,
            waiter.wait_for(std::chrono::milliseconds(50)));
  EXPECT_EQ(1U, manager.snapshot().waiting_builds);

  manager.set_probe_for_testing([] { return fixed_probe(0, 1); });
  EXPECT_FALSE(waiter.get());
  EXPECT_EQ(0U, manager.snapshot().waiting_builds);
  first.reset();
}

TEST(VectorResourceBudgetTest, WaitingBuildHonorsCancellation) {
  option_guard build_memory(&opt_vector_build_memory_size, 512ULL * kMiB);
  option_guard reserve_memory(&opt_vector_build_memory_reserve_size, 0);
  vector_index::resource_budget_manager manager;
  manager.set_probe_for_testing([] { return fixed_probe(512ULL * kMiB, 1); });

  vector_index::build_resource_lease first;
  ASSERT_TRUE(manager.acquire({512ULL * kMiB, 1}, &first));

  std::atomic<bool> cancelled{false};
  auto waiter = std::async(std::launch::async, [&] {
    vector_index::build_resource_lease lease;
    vector_index::resource_wait_options options;
    options.cancelled = [&] { return cancelled.load(); };
    return manager.acquire({1, 1}, &lease, options);
  });
  EXPECT_EQ(std::future_status::timeout,
            waiter.wait_for(std::chrono::milliseconds(50)));
  EXPECT_EQ(1U, manager.snapshot().waiting_builds);

  cancelled.store(true);
  EXPECT_EQ(std::future_status::ready,
            waiter.wait_for(std::chrono::seconds(2)));
  EXPECT_FALSE(waiter.get());
  EXPECT_EQ(0U, manager.snapshot().waiting_builds);
  first.reset();
}

TEST(VectorResourceBudgetTest, WaitingBuildHonorsFiniteTimeout) {
  option_guard build_memory(&opt_vector_build_memory_size, 512ULL * kMiB);
  option_guard reserve_memory(&opt_vector_build_memory_reserve_size, 0);
  vector_index::resource_budget_manager manager;
  manager.set_probe_for_testing([] { return fixed_probe(512ULL * kMiB, 1); });

  vector_index::build_resource_lease first;
  ASSERT_TRUE(manager.acquire({512ULL * kMiB, 1}, &first));

  vector_index::build_resource_lease second;
  vector_index::resource_wait_options options;
  options.timeout = std::chrono::milliseconds(50);
  EXPECT_FALSE(manager.acquire({1, 1}, &second, options));
  EXPECT_FALSE(second);
  EXPECT_EQ(0U, manager.snapshot().waiting_builds);
  EXPECT_EQ(1U, manager.snapshot().active_builds);
  first.reset();
}

TEST(VectorResourceBudgetTest, MoveOnlyLeaseTransfersOneReservation) {
  option_guard build_memory(&opt_vector_build_memory_size, 512ULL * kMiB);
  option_guard reserve_memory(&opt_vector_build_memory_reserve_size, 0);
  vector_index::resource_budget_manager manager;
  manager.set_probe_for_testing([] { return fixed_probe(512ULL * kMiB, 4); });

  vector_index::build_resource_lease first;
  ASSERT_TRUE(manager.acquire({128ULL * kMiB, 1}, &first));
  vector_index::build_resource_lease moved(std::move(first));
  EXPECT_FALSE(first);
  EXPECT_TRUE(moved);
  EXPECT_EQ(1U, manager.snapshot().active_builds);

  vector_index::build_resource_lease assigned;
  ASSERT_TRUE(manager.acquire({64ULL * kMiB, 1}, &assigned));
  EXPECT_EQ(2U, manager.snapshot().active_builds);
  assigned = std::move(moved);
  EXPECT_FALSE(moved);
  EXPECT_TRUE(assigned);
  EXPECT_EQ(1U, manager.snapshot().active_builds);

  auto *const same_lease = &assigned;
  *same_lease = std::move(*same_lease);
  EXPECT_TRUE(assigned);
  assigned.reset();
  EXPECT_EQ(0U, manager.snapshot().active_builds);
  manager.reset_for_testing();
}

TEST(VectorResourceBudgetTest, RejectsNullLeaseAndExhaustedEnvelope) {
  option_guard build_memory(&opt_vector_build_memory_size, 0);
  option_guard reserve_memory(&opt_vector_build_memory_reserve_size,
                              64ULL * kMiB);
  vector_index::resource_budget_manager manager;
  manager.set_probe_for_testing([] { return fixed_probe(64ULL * kMiB, 2); });

  EXPECT_FALSE(manager.acquire({1, 1}, nullptr));
  vector_index::build_resource_lease lease;
  EXPECT_FALSE(manager.acquire({1, 1}, &lease));
  EXPECT_FALSE(lease);
  EXPECT_EQ(0U, manager.snapshot().active_builds);
}

TEST(VectorResourceBudgetTest, AutomaticRequestUsesWholeSafeEnvelope) {
  option_guard build_memory(&opt_vector_build_memory_size, 0);
  option_guard reserve_memory(&opt_vector_build_memory_reserve_size, 0);
  vector_index::resource_budget_manager manager;
  manager.set_probe_for_testing([] { return fixed_probe(256ULL * kMiB, 3); });

  vector_index::build_resource_lease lease;
  ASSERT_TRUE(manager.acquire({0, 0}, &lease));
  EXPECT_EQ(256ULL * kMiB, lease.memory_bytes());
  EXPECT_EQ(3U, lease.cpu_slots());
  lease.reset();
  EXPECT_EQ(0U, manager.snapshot().active_builds);
  manager.reset_for_testing();
}

TEST(VectorResourceBudgetTest, NamesEveryProbeSource) {
  EXPECT_STREQ("cgroup_v2",
               vector_index::resource_probe_source_name(
                   vector_index::resource_probe_source::kCgroupV2));
  EXPECT_STREQ("cgroup_v1",
               vector_index::resource_probe_source_name(
                   vector_index::resource_probe_source::kCgroupV1));
  EXPECT_STREQ("host", vector_index::resource_probe_source_name(
                           vector_index::resource_probe_source::kHost));
  EXPECT_STREQ("host",
               vector_index::resource_probe_source_name(
                   static_cast<vector_index::resource_probe_source>(99)));
}

}  // namespace vector_resource_budget_unittest
