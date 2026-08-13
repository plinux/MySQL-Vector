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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <stdexcept>
#include <thread>

#include "sql/vector/vector_statement_publication.h"
#include "unittest/gunit/vector_test_utils.h"

namespace vector_statement_publication_unittest {
namespace {

using namespace std::chrono_literals;

TEST(VectorStatementPublicationTest, SameIndexWritersAreSerialized) {
  std::future<bool> acquired;
  std::promise<void> started;
  auto started_future = started.get_future();
  {
    vector_statement_publication::publication_guard first;
    ASSERT_TRUE(first.lock_indexes({"db.same"}));

    acquired = std::async(std::launch::async, [&] {
      vector_statement_publication::publication_guard second;
      started.set_value();
      return second.lock_indexes({"db.same"});
    });
    started_future.wait();
    EXPECT_EQ(std::future_status::timeout, acquired.wait_for(50ms));
  }

  ASSERT_EQ(std::future_status::ready, acquired.wait_for(2s));
  EXPECT_TRUE(acquired.get());
}

TEST(VectorStatementPublicationTest, DifferentIndexWritersCanRunTogether) {
  vector_statement_publication::publication_guard first;
  ASSERT_TRUE(first.lock_indexes({"db.alpha"}));

  auto acquired = std::async(std::launch::async, [] {
    vector_statement_publication::publication_guard second;
    return second.lock_indexes({"db.beta"});
  });
  ASSERT_EQ(std::future_status::ready, acquired.wait_for(2s));
  EXPECT_TRUE(acquired.get());
}

TEST(VectorStatementPublicationTest, TryIndexWriterNeverWaitsForOwnership) {
  vector_statement_publication::publication_guard first;
  ASSERT_TRUE(first.lock_indexes({"db.try"}));

  vector_statement_publication::publication_guard second;
  EXPECT_FALSE(second.try_lock_indexes({"db.try"}));
  EXPECT_FALSE(second.owns_lock());

  first = vector_statement_publication::publication_guard();
  EXPECT_TRUE(second.try_lock_indexes({"db.try"}));
  EXPECT_TRUE(second.owns_lock());
}

TEST(VectorStatementPublicationTest, ReservationCanBeReleasedByAnotherThread) {
  vector_statement_publication::publication_guard first;
  ASSERT_TRUE(first.lock_indexes({"db.cross_thread"}));

  std::promise<void> started;
  auto started_future = started.get_future();
  auto waiting_writer = std::async(std::launch::async, [&] {
    vector_statement_publication::publication_guard writer;
    started.set_value();
    return writer.lock_indexes({"db.cross_thread"});
  });
  started_future.wait();
  EXPECT_EQ(std::future_status::timeout, waiting_writer.wait_for(50ms));

  std::thread releaser([guard = std::move(first)]() mutable {
    guard = vector_statement_publication::publication_guard();
  });
  releaser.join();

  ASSERT_EQ(std::future_status::ready, waiting_writer.wait_for(2s));
  EXPECT_TRUE(waiting_writer.get());
}

TEST(VectorStatementPublicationTest, CatalogWriterWaitsForIndexPublication) {
  std::future<bool> acquired;
  std::promise<void> started;
  auto started_future = started.get_future();
  {
    vector_statement_publication::publication_guard index_publication;
    ASSERT_TRUE(index_publication.lock_indexes({"db.index"}));

    acquired = std::async(std::launch::async, [&] {
      vector_statement_publication::publication_guard catalog_publication;
      started.set_value();
      return catalog_publication.lock_catalog();
    });
    started_future.wait();
    EXPECT_EQ(std::future_status::timeout, acquired.wait_for(50ms));
  }

  ASSERT_EQ(std::future_status::ready, acquired.wait_for(2s));
  EXPECT_TRUE(acquired.get());
}

TEST(VectorStatementPublicationTest, CatalogWriterBlocksIndexReadersAndWriters) {
  std::future<bool> writer_acquired;
  std::future<bool> reader_acquired;
  std::promise<void> writer_started;
  std::promise<void> reader_started;
  auto writer_started_future = writer_started.get_future();
  auto reader_started_future = reader_started.get_future();
  {
    vector_statement_publication::publication_guard catalog_publication;
    ASSERT_TRUE(catalog_publication.lock_catalog());

    writer_acquired = std::async(std::launch::async, [&] {
      vector_statement_publication::publication_guard writer;
      writer_started.set_value();
      return writer.lock_indexes({"db.index"});
    });
    reader_acquired = std::async(std::launch::async, [&] {
      vector_statement_publication::read_guard reader;
      reader_started.set_value();
      return reader.lock_index("db.index");
    });
    writer_started_future.wait();
    reader_started_future.wait();
    EXPECT_EQ(std::future_status::timeout, writer_acquired.wait_for(50ms));
    EXPECT_EQ(std::future_status::timeout, reader_acquired.wait_for(50ms));
  }

  ASSERT_EQ(std::future_status::ready, writer_acquired.wait_for(2s));
  ASSERT_EQ(std::future_status::ready, reader_acquired.wait_for(2s));
  EXPECT_TRUE(writer_acquired.get());
  EXPECT_TRUE(reader_acquired.get());
}

TEST(VectorStatementPublicationTest, IndexWriterWaitCanBeCancelled) {
  vector_statement_publication::publication_guard first;
  ASSERT_TRUE(first.lock_indexes({"db.cancel_index"}));

  std::atomic<bool> cancelled{false};
  std::promise<void> started;
  auto waiting_writer = std::async(std::launch::async, [&] {
    vector_statement_publication::publication_guard writer;
    started.set_value();
    return writer.lock_indexes({"db.cancel_index"},
                               [&] { return cancelled.load(); });
  });
  started.get_future().wait();
  EXPECT_EQ(std::future_status::timeout, waiting_writer.wait_for(50ms));

  cancelled.store(true);
  if (waiting_writer.wait_for(2s) != std::future_status::ready) {
    first = vector_statement_publication::publication_guard();
  }
  ASSERT_EQ(std::future_status::ready, waiting_writer.wait_for(2s));
  EXPECT_FALSE(waiting_writer.get());
  EXPECT_TRUE(first.owns_lock());
}

TEST(VectorStatementPublicationTest,
     CancelledCatalogWriterDoesNotBlockNewReaders) {
  vector_statement_publication::publication_guard index_publication;
  ASSERT_TRUE(index_publication.lock_indexes({"db.catalog_cancel"}));

  std::atomic<bool> cancelled{false};
  std::promise<void> started;
  auto waiting_catalog_writer = std::async(std::launch::async, [&] {
    vector_statement_publication::publication_guard writer;
    started.set_value();
    return writer.lock_catalog([&] { return cancelled.load(); });
  });
  started.get_future().wait();
  EXPECT_EQ(std::future_status::timeout,
            waiting_catalog_writer.wait_for(50ms));

  cancelled.store(true);
  if (waiting_catalog_writer.wait_for(2s) != std::future_status::ready) {
    index_publication = vector_statement_publication::publication_guard();
  }
  ASSERT_EQ(std::future_status::ready,
            waiting_catalog_writer.wait_for(2s));
  EXPECT_FALSE(waiting_catalog_writer.get());

  vector_statement_publication::read_guard reader;
  EXPECT_TRUE(reader.lock_index("db.after_catalog_cancel"));
}

TEST(VectorStatementPublicationTest, IndexReaderWaitCanBeCancelled) {
  vector_statement_publication::publication_guard catalog_publication;
  ASSERT_TRUE(catalog_publication.lock_catalog());

  std::atomic<bool> cancelled{false};
  std::promise<void> started;
  auto waiting_reader = std::async(std::launch::async, [&] {
    vector_statement_publication::read_guard reader;
    started.set_value();
    return reader.lock_index("db.cancel_reader",
                             [&] { return cancelled.load(); });
  });
  started.get_future().wait();
  EXPECT_EQ(std::future_status::timeout, waiting_reader.wait_for(50ms));

  cancelled.store(true);
  if (waiting_reader.wait_for(2s) != std::future_status::ready) {
    catalog_publication = vector_statement_publication::publication_guard();
  }
  ASSERT_EQ(std::future_status::ready, waiting_reader.wait_for(2s));
  EXPECT_FALSE(waiting_reader.get());
  EXPECT_TRUE(catalog_publication.owns_lock());
}

TEST(VectorStatementPublicationTest, StableIndexSetComparisonNormalizesInputs) {
  EXPECT_TRUE(vector_statement_publication::index_sets_equal(
      {"db.b", "db.a", "db.b", ""}, {"db.a", "db.b"}));
  EXPECT_FALSE(vector_statement_publication::index_sets_equal(
      {"db.a", "db.b"}, {"db.a", "db.c"}));
}

TEST(VectorStatementPublicationTest, InputsAreNormalizedBeforeLocking) {
  vector_statement_publication::publication_guard guard;
  EXPECT_FALSE(guard.lock_indexes({"", ""}));
  EXPECT_TRUE(guard.lock_indexes({"db.z", "db.a", "db.z", ""}));
  EXPECT_TRUE(guard.owns_lock());
  EXPECT_FALSE(guard.lock_catalog());
}

TEST(VectorStatementPublicationTest,
     GuardBoundaryAndCancellationFailuresDoNotLeakReservations) {
  const auto cancelled = [] { return true; };
  const auto throws = []() -> bool { throw std::runtime_error("cancel"); };

  vector_statement_publication::publication_guard writer;
  EXPECT_FALSE(writer.lock_indexes({}));
  EXPECT_FALSE(writer.try_lock_indexes({}));
  EXPECT_FALSE(writer.lock_indexes({"db.cancelled"}, cancelled));
  EXPECT_FALSE(writer.try_lock_indexes({"db.throw"}, throws));
  EXPECT_FALSE(writer.lock_catalog(throws));
  EXPECT_FALSE(writer.owns_lock());

  vector_statement_publication::read_guard reader;
  EXPECT_FALSE(reader.lock_index(""));
  EXPECT_FALSE(reader.lock_index("db.cancelled", cancelled));
  EXPECT_FALSE(reader.lock_catalog(throws));
  EXPECT_FALSE(reader.owns_lock());

  EXPECT_TRUE(writer.lock_indexes({"db.recovered"}));
  EXPECT_FALSE(writer.lock_indexes({"db.other"}));
  EXPECT_FALSE(writer.try_lock_indexes({"db.other"}));
  EXPECT_FALSE(writer.lock_catalog());
  EXPECT_TRUE(writer.owns_lock());
}

TEST(VectorStatementPublicationTest,
     MultipleReadersBlockWriterUntilEveryLeaseIsReleased) {
  vector_statement_publication::publication_guard writer;
  {
    vector_statement_publication::read_guard first;
    vector_statement_publication::read_guard second;
    ASSERT_TRUE(first.lock_index("db.shared_read"));
    ASSERT_TRUE(second.lock_index("db.shared_read"));
    EXPECT_FALSE(writer.try_lock_indexes({"db.shared_read"}));
    EXPECT_FALSE(first.lock_index("db.other"));
  }
  EXPECT_TRUE(writer.try_lock_indexes({"db.shared_read"}));
}

TEST(VectorStatementPublicationTest,
     CatalogReaderBlocksCatalogWriterAndReleasesCleanly) {
  std::future<bool> acquired;
  std::promise<void> started;
  {
    vector_statement_publication::read_guard catalog_reader;
    ASSERT_TRUE(catalog_reader.lock_catalog());
    EXPECT_FALSE(catalog_reader.lock_catalog());
    acquired = std::async(std::launch::async, [&] {
      vector_statement_publication::publication_guard catalog_writer;
      started.set_value();
      return catalog_writer.lock_catalog();
    });
    started.get_future().wait();
    EXPECT_EQ(std::future_status::timeout, acquired.wait_for(50ms));
  }
  ASSERT_EQ(std::future_status::ready, acquired.wait_for(2s));
  EXPECT_TRUE(acquired.get());
}

TEST(VectorStatementPublicationTest,
     AllocationFailureDoesNotLeakReservationState) {
  {
    VECTOR_SCOPED_DEBUG_FLAG(
        debug, "+d,vector_publication_reservation_allocation_failure");
    vector_statement_publication::publication_guard failed;
    EXPECT_FALSE(failed.lock_indexes({"db.allocation_a", "db.allocation_b"}));
    EXPECT_FALSE(failed.owns_lock());
  }

  vector_statement_publication::publication_guard retry;
  EXPECT_TRUE(retry.lock_indexes({"db.allocation_a", "db.allocation_b"}));
  EXPECT_TRUE(retry.owns_lock());
}

TEST(VectorStatementPublicationTest, OperationPayloadRoundTripsBinaryData) {
  vector_statement_publication::operation_payload input;
  input.config =
      vector_statement_publication::config_change::kDiskannBuildParams;
  input.unsigned_values = {32, 100, 8};
  input.string_values = {"diskann", std::string("a\0b", 3)};
  input.binary_value = std::string("\0\1\2\xff", 4);

  std::string encoded;
  ASSERT_TRUE(
      vector_statement_publication::encode_operation_payload(input, &encoded));
  vector_statement_publication::operation_payload output;
  ASSERT_TRUE(vector_statement_publication::decode_operation_payload(encoded,
                                                                      &output));
  EXPECT_EQ(input.config, output.config);
  EXPECT_EQ(input.unsigned_values, output.unsigned_values);
  EXPECT_EQ(input.string_values, output.string_values);
  EXPECT_EQ(input.binary_value, output.binary_value);
}

TEST(VectorStatementPublicationTest, OperationPayloadRejectsCorruption) {
  vector_statement_publication::operation_payload payload;
  payload.unsigned_values = {1};
  std::string encoded;
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));

  vector_statement_publication::operation_payload decoded;
  EXPECT_FALSE(vector_statement_publication::decode_operation_payload(
      std::string(), &decoded));
  encoded[0] = static_cast<char>(0xff);
  EXPECT_FALSE(vector_statement_publication::decode_operation_payload(encoded,
                                                                      &decoded));
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));
  encoded.pop_back();
  EXPECT_FALSE(vector_statement_publication::decode_operation_payload(encoded,
                                                                      &decoded));
}

TEST(VectorStatementPublicationTest,
     OperationPayloadRejectsEveryTruncationAndInvalidBoundary) {
  vector_statement_publication::operation_payload payload;
  payload.config = vector_statement_publication::config_change::kSearchEf;
  payload.unsigned_values = {7};
  payload.string_values = {"index"};
  payload.binary_value = "payload";

  std::string encoded;
  EXPECT_FALSE(
      vector_statement_publication::encode_operation_payload(payload, nullptr));
  auto invalid_payload = payload;
  invalid_payload.config =
      static_cast<vector_statement_publication::config_change>(255);
  EXPECT_FALSE(vector_statement_publication::encode_operation_payload(
      invalid_payload, &encoded));
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));

  EXPECT_FALSE(
      vector_statement_publication::decode_operation_payload(encoded, nullptr));
  vector_statement_publication::operation_payload decoded;
  for (size_t length = 0; length < encoded.size(); ++length) {
    EXPECT_FALSE(vector_statement_publication::decode_operation_payload(
        encoded.substr(0, length), &decoded))
        << "accepted truncated length " << length;
  }

  auto invalid_version = encoded;
  invalid_version[0] = 2;
  EXPECT_FALSE(vector_statement_publication::decode_operation_payload(
      invalid_version, &decoded));
  auto invalid_config = encoded;
  invalid_config[1] = static_cast<char>(255);
  EXPECT_FALSE(vector_statement_publication::decode_operation_payload(
      invalid_config, &decoded));
  auto trailing_data = encoded;
  trailing_data.push_back('\0');
  EXPECT_FALSE(vector_statement_publication::decode_operation_payload(
      trailing_data, &decoded));

  const auto overwrite_uint64 = [](std::string *target, size_t offset,
                                   uint64_t value) {
    ASSERT_NE(nullptr, target);
    ASSERT_LE(offset + sizeof(uint64_t), target->size());
    for (size_t byte = 0; byte < sizeof(uint64_t); ++byte) {
      (*target)[offset + byte] =
          static_cast<char>((value >> (byte * 8)) & 0xffU);
    }
  };
  constexpr size_t k_first_string_length_offset = 2 + 3 * sizeof(uint64_t);
  auto oversized_string = encoded;
  overwrite_uint64(&oversized_string, k_first_string_length_offset,
                   std::numeric_limits<uint64_t>::max());
  EXPECT_FALSE(vector_statement_publication::decode_operation_payload(
      oversized_string, &decoded));
}

TEST(VectorStatementPublicationTest,
     OperationPayloadRejectsCountsBeyondRemainingFrame) {
  vector_statement_publication::operation_payload payload;
  payload.unsigned_values = {1};
  std::string encoded;
  ASSERT_TRUE(vector_statement_publication::encode_operation_payload(payload,
                                                                     &encoded));

  auto overwrite_count = [](std::string *target, size_t offset,
                            uint64_t count) {
    ASSERT_NE(nullptr, target);
    ASSERT_LE(offset + sizeof(uint64_t), target->size());
    for (size_t byte = 0; byte < sizeof(uint64_t); ++byte) {
      (*target)[offset + byte] =
          static_cast<char>((count >> (byte * 8)) & 0xffU);
    }
  };

  constexpr size_t k_unsigned_count_offset = 2;
  std::string excessive_unsigned_count = encoded;
  overwrite_count(&excessive_unsigned_count, k_unsigned_count_offset, 4);
  vector_statement_publication::operation_payload decoded;
  EXPECT_FALSE(vector_statement_publication::decode_operation_payload(
      excessive_unsigned_count, &decoded));

  constexpr size_t k_string_count_offset =
      k_unsigned_count_offset + 2 * sizeof(uint64_t);
  std::string excessive_string_count = encoded;
  overwrite_count(&excessive_string_count, k_string_count_offset, 2);
  EXPECT_FALSE(vector_statement_publication::decode_operation_payload(
      excessive_string_count, &decoded));
}

}  // namespace
}  // namespace vector_statement_publication_unittest
