# Big vector integration tests run only when MTR explicitly enables big tests
# on Linux and the current host or cgroup has enough available memory for the
# caller's estimated peak plus 25 percent headroom.
--source include/big_test.inc
--source include/linux.inc

if (!$vector_big_test_estimated_peak_kib)
{
  --die vector_big_test_estimated_peak_kib must be set before this include
}

--source suite/vector/include/vector_big_test_memory_detect.inc

if (!$vector_big_test_memory_detected)
{
  --skip Cannot determine safe Linux memory availability for vector big test
}

if (!$vector_big_test_memory_sufficient)
{
  --skip Vector big test needs $vector_big_test_required_kib KiB available, found $vector_big_test_available_kib KiB
}
