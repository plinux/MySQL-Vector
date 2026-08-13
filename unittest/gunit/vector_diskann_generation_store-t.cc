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

#include "sql/vector/vector_diskann_generation_store.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "unittest/gunit/vector_test_utils.h"

namespace vector_diskann_generation_store_unittest {
namespace {

class generation_store_test : public ::testing::Test {
 protected:
  static constexpr char kManifestFilename[] =
      "mysql-vector-artifact.manifest.v1";

  void SetUp() override {
    root = std::filesystem::temp_directory_path() /
           ("mysql-vector-generation-store-" +
            std::to_string(reinterpret_cast<uintptr_t>(this)));
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    live = root / "diskann_external.store";
    staging = root / "diskann_external.store.build";
  }

  void TearDown() override { std::filesystem::remove_all(root); }

  static void write_generation(const std::filesystem::path &directory,
                               const std::string &payload, uint64_t doc_id) {
    std::filesystem::create_directories(directory / "offline");
    std::ofstream data(directory / "offline" / "diskann_disk.index",
                       std::ios::out | std::ios::binary | std::ios::trunc);
    data << payload;
    data.close();
    std::ofstream doc_ids(
        directory / "offline" / "diskann_mysql_vector_docids.bin",
        std::ios::out | std::ios::binary | std::ios::trunc);
    const uint64_t count = 1;
    doc_ids.write(reinterpret_cast<const char *>(&count), sizeof(count));
    doc_ids.write(reinterpret_cast<const char *>(&doc_id), sizeof(doc_id));
  }

  static void write_doc_id_map(const std::filesystem::path &directory,
                               uint64_t count,
                               const std::vector<uint64_t> &doc_ids) {
    std::filesystem::create_directories(directory / "offline");
    std::ofstream file(
        directory / "offline" / "diskann_mysql_vector_docids.bin",
        std::ios::out | std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for (const uint64_t doc_id : doc_ids) {
      file.write(reinterpret_cast<const char *>(&doc_id), sizeof(doc_id));
    }
  }

  static vector_index::diskann_artifact_identity identity(
      uint64_t index_identity, uint64_t truth_generation,
      uint64_t config_generation) {
    vector_index::diskann_artifact_identity value;
    value.index_identity = index_identity;
    value.truth_generation = truth_generation;
    value.config_generation = config_generation;
    value.doc_id_count = 1;
    value.dimension = 3;
    value.metric = "l2";
    value.mode = "external";
    value.provider = "diskann";
    value.consistency_mode = "transactional";
    value.schema_name = "db";
    value.table_name = "t";
    value.column_name = "embedding";
    value.doc_id_column_name = "id";
    return value;
  }

  static void replace_journal_state(const std::filesystem::path &path,
                                    const std::string &from,
                                    const std::string &to) {
    std::ifstream input(path, std::ios::in | std::ios::binary);
    std::string contents((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
    const size_t offset = contents.find("state\t" + from);
    ASSERT_NE(offset, std::string::npos);
    contents.replace(offset, std::string("state\t" + from).size(),
                     "state\t" + to);
    std::ofstream output(path,
                         std::ios::out | std::ios::binary | std::ios::trunc);
    output << contents;
  }

  static std::string read_file(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::in | std::ios::binary);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
  }

  static void write_file(const std::filesystem::path &path,
                         const std::string &contents) {
    std::ofstream output(path,
                         std::ios::out | std::ios::binary | std::ios::trunc);
    output << contents;
  }

  static void replace_metadata_value(const std::filesystem::path &path,
                                     const std::string &key,
                                     const std::string &value) {
    std::string contents = read_file(path);
    const std::string prefix = key + "\t";
    const size_t offset = contents.find(prefix);
    ASSERT_NE(offset, std::string::npos);
    const size_t end = contents.find('\n', offset);
    ASSERT_NE(end, std::string::npos);
    contents.replace(offset + prefix.size(), end - offset - prefix.size(),
                     value);
    write_file(path, contents);
  }

  static void remove_metadata_value(const std::filesystem::path &path,
                                    const std::string &key) {
    std::string contents = read_file(path);
    const std::string prefix = key + "\t";
    const size_t offset = contents.find(prefix);
    ASSERT_NE(offset, std::string::npos);
    const size_t end = contents.find('\n', offset);
    ASSERT_NE(end, std::string::npos);
    contents.erase(offset, end - offset + 1);
    write_file(path, contents);
  }

  static void remove_metadata_values(const std::filesystem::path &path,
                                     const std::string &key) {
    std::string contents = read_file(path);
    const std::string prefix = key + "\t";
    size_t offset = 0;
    while ((offset = contents.find(prefix, offset)) != std::string::npos) {
      if (offset != 0 && contents[offset - 1] != '\n') {
        offset += prefix.size();
        continue;
      }
      const size_t end = contents.find('\n', offset);
      ASSERT_NE(end, std::string::npos);
      contents.erase(offset, end - offset + 1);
    }
    write_file(path, contents);
  }

  static void append_metadata_line(const std::filesystem::path &path,
                                   const std::string &line) {
    std::ofstream output(path,
                         std::ios::out | std::ios::binary | std::ios::app);
    output << line << '\n';
  }

  static std::filesystem::path manifest_path(
      const std::filesystem::path &directory) {
    return directory / kManifestFilename;
  }

  void prepare_and_publish(
      const vector_index::diskann_artifact_identity &target,
      vector_index::diskann_generation_swap *swap, std::string *error,
      bool with_live = false) {
    if (with_live) write_generation(live, "old", 1);
    write_generation(staging, "new", 2);
    ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
        live.string(), staging.string(), target, swap, error))
        << *error;
    ASSERT_TRUE(vector_index::publish_diskann_generation_swap(swap, error))
        << *error;
  }

  void reset_store() {
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
  }

  std::filesystem::path root;
  std::filesystem::path live;
  std::filesystem::path staging;
};

TEST_F(generation_store_test, publishes_and_finalizes_verified_generation) {
  write_generation(live, "old", 1);
  write_generation(staging, "new", 2);
  vector_index::diskann_generation_swap swap;
  std::string error;
  ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), identity(7, 11, 3), &swap, &error))
      << error;
  EXPECT_TRUE(std::filesystem::exists(swap.journal_path));
  ASSERT_TRUE(vector_index::publish_diskann_generation_swap(&swap, &error))
      << error;
  ASSERT_TRUE(vector_index::verify_diskann_generation_swap(&swap, &error))
      << error;
  ASSERT_TRUE(vector_index::finalize_diskann_generation_swap(&swap, &error))
      << error;
  EXPECT_TRUE(std::filesystem::exists(live));
  EXPECT_FALSE(std::filesystem::exists(
      vector_index::diskann_generation_journal_path(live.string())));
}

TEST_F(generation_store_test, rollback_restores_previous_generation) {
  write_generation(live, "old", 1);
  write_generation(staging, "new", 2);
  vector_index::diskann_generation_swap swap;
  std::string error;
  ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), identity(7, 12, 3), &swap, &error))
      << error;
  ASSERT_TRUE(vector_index::publish_diskann_generation_swap(&swap, &error))
      << error;
  ASSERT_TRUE(vector_index::rollback_diskann_generation_swap(&swap, &error))
      << error;
  std::ifstream old_data(live / "offline" / "diskann_disk.index",
                         std::ios::in | std::ios::binary);
  std::string payload;
  old_data >> payload;
  EXPECT_EQ(payload, "old");
}

TEST_F(generation_store_test, rollback_before_publish_keeps_live_generation) {
  write_generation(live, "old", 1);
  write_generation(staging, "new", 2);
  vector_index::diskann_generation_swap swap;
  std::string error;
  ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), identity(7, 13, 3), &swap, &error))
      << error;
  ASSERT_TRUE(vector_index::rollback_diskann_generation_swap(&swap, &error))
      << error;
  std::ifstream old_data(live / "offline" / "diskann_disk.index",
                         std::ios::in | std::ios::binary);
  std::string payload;
  old_data >> payload;
  EXPECT_EQ(payload, "old");
}

TEST_F(generation_store_test, recovery_finishes_metadata_selected_generation) {
  write_generation(live, "old", 1);
  write_generation(staging, "new", 2);
  vector_index::diskann_generation_swap swap;
  std::string error;
  const auto target = identity(8, 21, 4);
  ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error))
      << error;
  ASSERT_TRUE(vector_index::publish_diskann_generation_swap(&swap, &error))
      << error;

  bool journal_found = false;
  ASSERT_TRUE(vector_index::recover_diskann_generation_swap(
      live.string(), target, &journal_found, &error))
      << error;
  EXPECT_TRUE(journal_found);
  vector_index::diskann_artifact_identity actual;
  EXPECT_TRUE(vector_index::validate_diskann_generation_store(
      live.string(), target, &actual, &error))
      << error;
}

TEST_F(generation_store_test,
       recovery_finishes_install_when_journal_state_lags_rename) {
  write_generation(live, "old", 1);
  write_generation(staging, "new", 2);
  vector_index::diskann_generation_swap swap;
  std::string error;
  const auto target = identity(8, 22, 4);
  ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error))
      << error;
  ASSERT_TRUE(vector_index::publish_diskann_generation_swap(&swap, &error))
      << error;
  replace_journal_state(swap.journal_path, "new_installed", "old_saved");

  bool journal_found = false;
  ASSERT_TRUE(vector_index::recover_diskann_generation_swap(
      live.string(), target, &journal_found, &error))
      << error;
  EXPECT_TRUE(journal_found);
  vector_index::diskann_artifact_identity actual;
  EXPECT_TRUE(vector_index::validate_diskann_generation_store(
      live.string(), target, &actual, &error))
      << error;
}

TEST_F(generation_store_test,
       recovery_finishes_prepared_verified_and_complete_states) {
  const auto recover_state = [&](uint64_t truth_generation, const char *state) {
    reset_store();
    write_generation(live, "old", 1);
    write_generation(staging, "new", 2);
    const auto target = identity(18, truth_generation, 7);
    vector_index::diskann_generation_swap swap;
    std::string error;
    ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
        live.string(), staging.string(), target, &swap, &error))
        << error;
    if (std::string(state) != "prepared") {
      ASSERT_TRUE(vector_index::publish_diskann_generation_swap(&swap, &error))
          << error;
    }
    if (std::string(state) == "verified" || std::string(state) == "complete") {
      ASSERT_TRUE(vector_index::verify_diskann_generation_swap(&swap, &error))
          << error;
    }
    if (std::string(state) == "complete") {
      replace_journal_state(swap.journal_path, "verified", "complete");
    }

    bool journal_found = false;
    ASSERT_TRUE(vector_index::recover_diskann_generation_swap(
        live.string(), target, &journal_found, &error))
        << state << ": " << error;
    EXPECT_TRUE(journal_found);
    vector_index::diskann_artifact_identity actual;
    EXPECT_TRUE(vector_index::validate_diskann_generation_store(
        live.string(), target, &actual, &error))
        << state << ": " << error;
    EXPECT_FALSE(std::filesystem::exists(
        vector_index::diskann_generation_journal_path(live.string())));
  };

  recover_state(51, "prepared");
  recover_state(52, "verified");
  recover_state(53, "complete");
}

TEST_F(generation_store_test, recovery_rejects_invalid_or_unsafe_journal) {
  const auto corrupt_and_recover = [&](const std::string &key,
                                       const std::string &value) {
    reset_store();
    write_generation(live, "old", 1);
    write_generation(staging, "new", 2);
    const auto target = identity(19, 61, 8);
    vector_index::diskann_generation_swap swap;
    std::string error;
    ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
        live.string(), staging.string(), target, &swap, &error))
        << error;
    replace_metadata_value(swap.journal_path, key, value);

    bool journal_found = true;
    EXPECT_FALSE(vector_index::recover_diskann_generation_swap(
        live.string(), target, &journal_found, &error));
    EXPECT_FALSE(journal_found);
    EXPECT_FALSE(error.empty());
  };

  corrupt_and_recover("state", "unknown");
  corrupt_and_recover("had_old_generation", "2");
  corrupt_and_recover("had_old_generation", "");
  corrupt_and_recover("had_old_generation", "184467440737095516160");
  corrupt_and_recover("staging_name", "");
  corrupt_and_recover("staging_name", "2e");
  corrupt_and_recover("staging_name", "2e2e");
  corrupt_and_recover("staging_name", "2e2e2f657363617065");
  corrupt_and_recover("backup_name", "");
  corrupt_and_recover("backup_name", "2e");
  corrupt_and_recover("backup_name", "2e2e");
  corrupt_and_recover("backup_name", "5c657363617065");
}

TEST_F(generation_store_test,
       recovery_rejects_each_missing_or_malformed_journal_field) {
  const auto mutate_and_recover = [&](uint64_t generation, const auto &mutate) {
    reset_store();
    write_generation(live, "old", 1);
    write_generation(staging, "new", 2);
    const auto target = identity(40, generation, 30);
    vector_index::diskann_generation_swap swap;
    std::string error;
    ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
        live.string(), staging.string(), target, &swap, &error))
        << error;
    mutate(swap.journal_path);

    bool journal_found = true;
    EXPECT_FALSE(vector_index::recover_diskann_generation_swap(
        live.string(), target, &journal_found, &error));
    EXPECT_FALSE(journal_found);
    EXPECT_FALSE(error.empty());
  };

  uint64_t generation = 200;
  for (const char *field :
       {"state", "had_old_generation", "staging_name", "backup_name"}) {
    SCOPED_TRACE(field);
    mutate_and_recover(generation++, [&](const auto &path) {
      remove_metadata_value(path, field);
    });
  }
  mutate_and_recover(generation++, [&](const auto &path) {
    replace_metadata_value(path, "staging_name", "4G");
  });
  mutate_and_recover(generation++, [&](const auto &path) {
    replace_metadata_value(path, "backup_name", "4A");
  });
  mutate_and_recover(generation++, [&](const auto &path) {
    append_metadata_line(path, "backup_name\t6261636b7570");
  });
  mutate_and_recover(generation++, [&](const auto &path) {
    append_metadata_line(path, "\tmissing-key");
  });
}

TEST_F(generation_store_test, recovery_without_journal_is_a_clean_noop) {
  bool journal_found = true;
  std::string error = "stale";
  EXPECT_TRUE(vector_index::recover_diskann_generation_swap(
      live.string(), identity(20, 70, 9), &journal_found, &error));
  EXPECT_FALSE(journal_found);
}

TEST_F(generation_store_test,
       publication_identity_rejects_each_stale_binding_field) {
  const auto expected = identity(21, 81, 10);
  EXPECT_TRUE(
      vector_index::diskann_artifact_publication_matches(expected, expected));

  const auto expect_mismatch = [&](const auto &mutate) {
    auto actual = expected;
    mutate(&actual);
    EXPECT_FALSE(
        vector_index::diskann_artifact_publication_matches(expected, actual));
  };
  expect_mismatch([](auto *value) { ++value->index_identity; });
  expect_mismatch([](auto *value) { ++value->truth_generation; });
  expect_mismatch([](auto *value) { ++value->config_generation; });
  expect_mismatch([](auto *value) { ++value->doc_id_count; });
  expect_mismatch([](auto *value) { ++value->dimension; });
  expect_mismatch([](auto *value) { value->metric = "cosine"; });
  expect_mismatch([](auto *value) { value->mode = "memory"; });
  expect_mismatch([](auto *value) { value->provider = "faiss"; });
  expect_mismatch([](auto *value) { value->consistency_mode = "standalone"; });
  expect_mismatch([](auto *value) { value->schema_name = "other_db"; });
  expect_mismatch([](auto *value) { value->table_name = "other_table"; });
  expect_mismatch([](auto *value) { value->column_name = "other_column"; });
  expect_mismatch([](auto *value) { value->doc_id_column_name = "other_id"; });
}

TEST_F(generation_store_test, public_api_rejects_invalid_swap_states) {
  std::string error;
  vector_index::diskann_generation_swap swap;
  EXPECT_TRUE(vector_index::diskann_generation_journal_path("").empty());
  EXPECT_FALSE(vector_index::publish_diskann_generation_swap(nullptr, &error));
  EXPECT_FALSE(vector_index::verify_diskann_generation_swap(nullptr, &error));
  EXPECT_TRUE(vector_index::rollback_diskann_generation_swap(nullptr, &error));
  EXPECT_TRUE(vector_index::rollback_diskann_generation_swap(&swap, &error));
  EXPECT_FALSE(vector_index::finalize_diskann_generation_swap(nullptr, &error));
  EXPECT_FALSE(vector_index::recover_diskann_generation_swap(
      live.string(), identity(22, 82, 11), nullptr, &error));
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      "", staging.string(), identity(22, 82, 11), &swap, nullptr));
  EXPECT_FALSE(vector_index::validate_diskann_generation_store(
      live.string(), identity(22, 82, 11), nullptr, nullptr));

  swap.state = vector_index::diskann_swap_state::kOldSaved;
  EXPECT_FALSE(vector_index::publish_diskann_generation_swap(&swap, &error));
  EXPECT_FALSE(vector_index::verify_diskann_generation_swap(&swap, &error));
  EXPECT_FALSE(vector_index::finalize_diskann_generation_swap(&swap, &error));

  swap.state = vector_index::diskann_swap_state::kComplete;
  EXPECT_FALSE(vector_index::publish_diskann_generation_swap(&swap, &error));
  EXPECT_FALSE(vector_index::verify_diskann_generation_swap(&swap, &error));
  EXPECT_FALSE(vector_index::finalize_diskann_generation_swap(&swap, &error));
}

TEST_F(generation_store_test,
       preparation_rejects_missing_staging_and_existing_journal) {
  vector_index::diskann_generation_swap swap;
  std::string error;
  auto target = identity(23, 83, 12);
  target.config_generation = 0;
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error));
  target = identity(23, 83, 12);
  target.dimension = 0;
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error));

  target = identity(23, 83, 12);
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error));

  write_generation(staging, "new", 2);
  write_file(vector_index::diskann_generation_journal_path(live.string()),
             "unfinished publication");
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error));
}

TEST_F(generation_store_test,
       preparation_live_preflight_failure_has_no_side_effects) {
  write_generation(live, "old", 1);
  write_generation(staging, "new", 2);
  vector_index::diskann_generation_swap swap;
  std::string error;

  VECTOR_SCOPED_DEBUG_FLAG(
      fail_live_preflight,
      "+d,vector_diskann_generation_fail_live_preflight");
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), identity(23, 84, 12), &swap, &error));

  EXPECT_EQ(error, "could not inspect generation path");
  EXPECT_EQ(read_file(live / "offline" / "diskann_disk.index"), "old");
  EXPECT_EQ(read_file(staging / "offline" / "diskann_disk.index"), "new");
  EXPECT_FALSE(std::filesystem::exists(manifest_path(staging)));
  EXPECT_FALSE(std::filesystem::exists(
      vector_index::diskann_generation_journal_path(live.string())));
  size_t root_entries = 0;
  for ([[maybe_unused]] const auto &entry :
       std::filesystem::directory_iterator(root)) {
    ++root_entries;
  }
  EXPECT_EQ(root_entries, 2U);
}

TEST_F(generation_store_test,
       preparation_ignores_stale_manifest_temporary_files) {
  write_generation(staging, "new", 2);
  write_file(manifest_path(staging), "stale manifest");
  write_file(manifest_path(staging).string() + ".tmp", "stale temporary");
  vector_index::diskann_generation_swap swap;
  std::string error;
  const auto target = identity(24, 84, 13);
  ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error))
      << error;
  ASSERT_TRUE(vector_index::publish_diskann_generation_swap(&swap, &error))
      << error;
  vector_index::diskann_artifact_identity actual;
  EXPECT_TRUE(vector_index::validate_diskann_generation_store(
      live.string(), target, &actual, &error))
      << error;
}

TEST_F(generation_store_test,
       preparation_rejects_invalid_doc_id_map_count_and_size) {
  vector_index::diskann_generation_swap swap;
  std::string error;

  write_generation(staging, "new", 2);
  auto target = identity(24, 85, 13);
  target.doc_id_count = 2;
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error));
  EXPECT_FALSE(error.empty());

  reset_store();
  write_generation(staging, "new", 2);
  write_doc_id_map(staging, std::numeric_limits<uint64_t>::max(), {});
  target = identity(24, 86, 13);
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error));
  EXPECT_FALSE(error.empty());
}

TEST_F(generation_store_test, validation_rejects_each_invalid_identity_field) {
  struct invalid_field {
    const char *key;
    const char *value;
  };
  const std::vector<invalid_field> cases{
      {"index_identity", "x"},
      {"index_identity", "0"},
      {"truth_generation", "x"},
      {"truth_generation", ""},
      {"config_generation", "0"},
      {"config_generation", "184467440737095516160"},
      {"doc_id_count", "x"},
      {"doc_id_count", "2"},
      {"doc_id_checksum", "x"},
      {"dimension", "4294967296"},
      {"dimension", "0"},
      {"metric", "x"},
      {"metric", "6x"},
      {"mode", "x"},
      {"provider", "x"},
      {"consistency_mode", "x"},
      {"schema_name", "x"},
      {"table_name", "x"},
      {"column_name", "x"},
      {"doc_id_column_name", "x"},
  };

  uint64_t generation = 90;
  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.key);
    reset_store();
    const auto target = identity(25, generation++, 14);
    vector_index::diskann_generation_swap swap;
    std::string error;
    prepare_and_publish(target, &swap, &error);
    replace_metadata_value(manifest_path(live), test_case.key, test_case.value);
    EXPECT_FALSE(vector_index::validate_diskann_generation_store(
        live.string(), target, nullptr, &error));
    EXPECT_FALSE(error.empty());
  }
}

TEST_F(generation_store_test, validation_rejects_missing_identity_fields) {
  const std::vector<const char *> fields{
      "index_identity",
      "truth_generation",
      "config_generation",
      "doc_id_count",
      "doc_id_checksum",
      "dimension",
      "metric",
      "mode",
      "provider",
      "consistency_mode",
      "schema_name",
      "table_name",
      "column_name",
      "doc_id_column_name",
  };

  uint64_t generation = 110;
  for (const char *field : fields) {
    SCOPED_TRACE(field);
    reset_store();
    const auto target = identity(26, generation++, 15);
    vector_index::diskann_generation_swap swap;
    std::string error;
    prepare_and_publish(target, &swap, &error);
    remove_metadata_value(manifest_path(live), field);
    EXPECT_FALSE(vector_index::validate_diskann_generation_store(
        live.string(), target, nullptr, &error));
    EXPECT_FALSE(error.empty());
  }
}

TEST_F(generation_store_test, validation_rejects_malformed_manifest_records) {
  struct manifest_mutation {
    const char *key;
    const char *value;
  };
  const std::vector<manifest_mutation> cases{
      {"file", "payload"},      {"file", "61,1"},    {"file", "61,1,2,3"},
      {"file", "zz,1,2"},       {"file", ",1,2"},    {"file", "2e,1,2"},
      {"file", "2f746d70,1,2"}, {"file", "61,x,2"},  {"file", "61,1,x"},
      {"file_count", "x"},      {"file_count", "0"},
  };

  uint64_t generation = 130;
  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.value);
    reset_store();
    const auto target = identity(27, generation++, 16);
    vector_index::diskann_generation_swap swap;
    std::string error;
    prepare_and_publish(target, &swap, &error);
    replace_metadata_value(manifest_path(live), test_case.key, test_case.value);
    EXPECT_FALSE(vector_index::validate_diskann_generation_store(
        live.string(), target, nullptr, &error));
    EXPECT_FALSE(error.empty());
  }
}

TEST_F(generation_store_test,
       validation_rejects_bad_header_lines_duplicates_and_missing_files) {
  const auto expect_invalid = [&](uint64_t generation, const auto &mutate) {
    reset_store();
    const auto target = identity(28, generation, 17);
    vector_index::diskann_generation_swap swap;
    std::string error;
    prepare_and_publish(target, &swap, &error);
    mutate(manifest_path(live));
    EXPECT_FALSE(vector_index::validate_diskann_generation_store(
        live.string(), target, nullptr, &error));
    EXPECT_FALSE(error.empty());
  };

  expect_invalid(150, [&](const auto &path) {
    std::string contents = read_file(path);
    contents.replace(0, contents.find('\n'), "invalid-header");
    write_file(path, contents);
  });
  expect_invalid(151,
                 [&](const auto &path) { append_metadata_line(path, "bad"); });
  expect_invalid(152, [&](const auto &path) {
    append_metadata_line(path, "\tmissing-key");
  });
  expect_invalid(153, [&](const auto &path) {
    std::string contents = read_file(path);
    const size_t offset = contents.find("file\t");
    ASSERT_NE(offset, std::string::npos);
    const size_t end = contents.find('\n', offset);
    ASSERT_NE(end, std::string::npos);
    replace_metadata_value(path, "file_count", "3");
    append_metadata_line(path, contents.substr(offset, end - offset));
  });
  expect_invalid(154, [&](const auto &path) {
    replace_metadata_value(path, "file_count", "0");
    remove_metadata_values(path, "file");
  });
  expect_invalid(155, [&](const auto &) {
    std::filesystem::remove(live / "offline" / "diskann_disk.index");
  });
}

TEST_F(generation_store_test,
       recovery_rejects_unreadable_header_and_duplicate_journal_fields) {
  const auto target = identity(29, 160, 18);
  bool journal_found = true;
  std::string error;
  std::filesystem::create_directories(
      vector_index::diskann_generation_journal_path(live.string()));
  EXPECT_FALSE(vector_index::recover_diskann_generation_swap(
      live.string(), target, &journal_found, &error));
  EXPECT_FALSE(journal_found);

  reset_store();
  write_generation(staging, "new", 2);
  vector_index::diskann_generation_swap swap;
  ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error))
      << error;
  write_file(swap.journal_path, "invalid-header\n");
  EXPECT_FALSE(vector_index::recover_diskann_generation_swap(
      live.string(), target, &journal_found, &error));
  EXPECT_FALSE(journal_found);

  reset_store();
  write_generation(staging, "new", 2);
  ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error))
      << error;
  append_metadata_line(swap.journal_path, "state\tprepared");
  EXPECT_FALSE(vector_index::recover_diskann_generation_swap(
      live.string(), target, &journal_found, &error));
  EXPECT_FALSE(journal_found);
}

TEST_F(generation_store_test,
       recovery_rejects_ambiguous_prepared_and_old_saved_layouts) {
  const auto recover_layout = [&](uint64_t generation, bool expected_matches,
                                  const auto &mutate) {
    reset_store();
    write_generation(live, "old", 1);
    write_generation(staging, "new", 2);
    const auto target = identity(30, generation, 19);
    vector_index::diskann_generation_swap swap;
    std::string error;
    ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
        live.string(), staging.string(), target, &swap, &error))
        << error;
    mutate(&swap);
    bool journal_found = false;
    const auto expected =
        expected_matches ? target : identity(30, generation - 1, 19);
    EXPECT_FALSE(vector_index::recover_diskann_generation_swap(
        live.string(), expected, &journal_found, &error));
    EXPECT_TRUE(journal_found);
    EXPECT_FALSE(error.empty());
  };

  recover_layout(170, true, [&](const auto *swap) {
    std::filesystem::copy(live, swap->backup_directory,
                          std::filesystem::copy_options::recursive);
  });
  recover_layout(171, false, [&](const auto *swap) {
    std::filesystem::copy(live, swap->backup_directory,
                          std::filesystem::copy_options::recursive);
  });
  recover_layout(172, true, [&](const auto *swap) {
    replace_journal_state(swap->journal_path, "prepared", "old_saved");
  });
  recover_layout(173, false, [&](const auto *swap) {
    replace_journal_state(swap->journal_path, "prepared", "old_saved");
  });
}

TEST_F(generation_store_test,
       recovery_rejects_unexpected_new_generation_and_corrupt_candidate) {
  const auto recover_unexpected_live = [&](uint64_t generation,
                                           bool expected_matches) {
    reset_store();
    write_generation(staging, "new", 2);
    const auto target = identity(31, generation, 20);
    vector_index::diskann_generation_swap swap;
    std::string error;
    ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
        live.string(), staging.string(), target, &swap, &error))
        << error;
    write_generation(live, "unexpected", 3);
    bool journal_found = false;
    const auto expected =
        expected_matches ? target : identity(31, generation - 1, 20);
    EXPECT_FALSE(vector_index::recover_diskann_generation_swap(
        live.string(), expected, &journal_found, &error));
    EXPECT_TRUE(journal_found);
  };

  recover_unexpected_live(180, true);
  recover_unexpected_live(181, false);

  reset_store();
  const auto target = identity(31, 182, 20);
  vector_index::diskann_generation_swap swap;
  std::string error;
  prepare_and_publish(target, &swap, &error, true);
  std::filesystem::remove(live / "offline" / "diskann_disk.index");
  bool journal_found = false;
  EXPECT_FALSE(vector_index::recover_diskann_generation_swap(
      live.string(), target, &journal_found, &error));
  EXPECT_TRUE(journal_found);
}

TEST_F(generation_store_test,
       rollback_without_previous_generation_removes_unpublished_candidate) {
  write_generation(staging, "new", 2);
  vector_index::diskann_generation_swap swap;
  std::string error;
  const auto target = identity(32, 190, 21);
  ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error))
      << error;
  ASSERT_FALSE(swap.had_old_generation);
  ASSERT_TRUE(vector_index::publish_diskann_generation_swap(&swap, &error))
      << error;
  ASSERT_TRUE(vector_index::rollback_diskann_generation_swap(&swap, &error))
      << error;
  EXPECT_FALSE(std::filesystem::exists(live));
  EXPECT_FALSE(std::filesystem::exists(
      vector_index::diskann_generation_journal_path(live.string())));
}

TEST_F(generation_store_test,
       preparation_rejects_invalid_inputs_and_artifact_layouts) {
  vector_index::diskann_generation_swap swap;
  std::string error;
  const auto target = identity(20, 71, 9);
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      "", staging.string(), target, &swap, &error));
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), "", target, &swap, &error));

  auto invalid_identity = target;
  invalid_identity.index_identity = 0;
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), invalid_identity, &swap, &error));
  invalid_identity = target;
  invalid_identity.provider = "faiss";
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), invalid_identity, &swap, &error));

  std::filesystem::create_directories(staging);
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, nullptr, &error));
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), live.string(), target, &swap, &error));
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), (root.parent_path() / "outside.build").string(), target,
      &swap, &error));
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error));

  reset_store();
  std::filesystem::create_directories(staging / "offline");
  std::ofstream(staging / "offline" / "diskann_disk.index") << "payload";
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error));

  reset_store();
  write_generation(staging, "new", 2);
  std::ofstream(staging / "offline" / "copy_mysql_vector_docids.bin")
      << "duplicate";
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error));

#ifndef _WIN32
  reset_store();
  write_generation(staging, "new", 2);
  std::error_code ec;
  std::filesystem::create_symlink(staging / "offline" / "diskann_disk.index",
                                  staging / "offline" / "index.link", ec);
  ASSERT_FALSE(ec);
  EXPECT_FALSE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error));
#endif
}

TEST_F(generation_store_test,
       recovery_restores_old_generation_on_token_mismatch) {
  write_generation(live, "old", 1);
  write_generation(staging, "new", 2);
  vector_index::diskann_generation_swap swap;
  std::string error;
  ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), identity(9, 31, 5), &swap, &error))
      << error;
  ASSERT_TRUE(vector_index::publish_diskann_generation_swap(&swap, &error))
      << error;

  bool journal_found = false;
  ASSERT_TRUE(vector_index::recover_diskann_generation_swap(
      live.string(), identity(9, 30, 5), &journal_found, &error))
      << error;
  EXPECT_TRUE(journal_found);
  std::ifstream old_data(live / "offline" / "diskann_disk.index",
                         std::ios::in | std::ios::binary);
  std::string payload;
  old_data >> payload;
  EXPECT_EQ(payload, "old");
}

TEST_F(generation_store_test,
       recovery_rolls_back_install_when_journal_state_lags_rename) {
  write_generation(live, "old", 1);
  write_generation(staging, "new", 2);
  vector_index::diskann_generation_swap swap;
  std::string error;
  ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), identity(9, 32, 5), &swap, &error))
      << error;
  ASSERT_TRUE(vector_index::publish_diskann_generation_swap(&swap, &error))
      << error;
  replace_journal_state(swap.journal_path, "new_installed", "old_saved");

  bool journal_found = false;
  ASSERT_TRUE(vector_index::recover_diskann_generation_swap(
      live.string(), identity(9, 31, 5), &journal_found, &error))
      << error;
  EXPECT_TRUE(journal_found);
  std::ifstream old_data(live / "offline" / "diskann_disk.index",
                         std::ios::in | std::ios::binary);
  std::string payload;
  old_data >> payload;
  EXPECT_EQ(payload, "old");
}

TEST_F(generation_store_test, validation_rejects_modified_artifact) {
  write_generation(staging, "new", 2);
  vector_index::diskann_generation_swap swap;
  std::string error;
  const auto target = identity(10, 41, 6);
  ASSERT_TRUE(vector_index::prepare_diskann_generation_swap(
      live.string(), staging.string(), target, &swap, &error))
      << error;
  ASSERT_TRUE(vector_index::publish_diskann_generation_swap(&swap, &error))
      << error;
  std::ofstream changed(live / "offline" / "diskann_disk.index",
                        std::ios::out | std::ios::binary | std::ios::app);
  changed << "corrupt";
  changed.close();
  vector_index::diskann_artifact_identity actual;
  EXPECT_FALSE(vector_index::validate_diskann_generation_store(
      live.string(), target, &actual, &error));
  EXPECT_FALSE(error.empty());
}

}  // namespace
}  // namespace vector_diskann_generation_store_unittest
