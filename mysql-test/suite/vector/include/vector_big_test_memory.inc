# Big vector integration tests run only when MTR explicitly enables big tests
# on Linux and the current host has enough available memory for the caller's
# estimated peak plus 25 percent headroom.
--source include/big_test.inc
--source include/linux.inc

if (!$vector_big_test_estimated_peak_kib)
{
  --die vector_big_test_estimated_peak_kib must be set before this include
}

--let $vector_big_test_memory_result=$MYSQL_TMP_DIR/vector_big_test_memory.inc
let vector_big_test_estimated_peak_kib_for_perl = $vector_big_test_estimated_peak_kib;
let vector_big_test_memory_result_for_perl = $vector_big_test_memory_result;

--perl
use strict;
use warnings;

my $estimated_peak_kib =
  $ENV{'vector_big_test_estimated_peak_kib_for_perl'};
my $result_file = $ENV{'vector_big_test_memory_result_for_perl'};
die "vector_big_test_estimated_peak_kib_for_perl is not set\n"
  unless defined $estimated_peak_kib && $estimated_peak_kib =~ /^\d+$/ &&
         $estimated_peak_kib > 0;
die "vector_big_test_memory_result_for_perl is not set\n"
  unless defined $result_file && length $result_file;

my $available_kib = 0;
if (open(my $meminfo, '<', '/proc/meminfo')) {
  while (my $line = <$meminfo>) {
    if ($line =~ /^MemAvailable:\s+(\d+)\s+kB$/) {
      $available_kib = $1;
      last;
    }
  }
  close $meminfo;
}

my $required_kib = int(($estimated_peak_kib * 5 + 3) / 4);
my $detected = $available_kib > 0 ? 1 : 0;
my $sufficient = $detected && $available_kib >= $required_kib ? 1 : 0;

open(my $result, '>', $result_file)
  or die "cannot create $result_file: $!\n";
print {$result} "let \$vector_big_test_memory_detected = $detected;\n";
print {$result} "let \$vector_big_test_memory_sufficient = $sufficient;\n";
print {$result} "let \$vector_big_test_available_kib = $available_kib;\n";
print {$result} "let \$vector_big_test_required_kib = $required_kib;\n";
close $result;
EOF

--source $vector_big_test_memory_result
--remove_file $vector_big_test_memory_result

if (!$vector_big_test_memory_detected)
{
  --skip Cannot determine Linux MemAvailable for vector big test
}

if (!$vector_big_test_memory_sufficient)
{
  --skip Vector big test needs $vector_big_test_required_kib KiB available, found $vector_big_test_available_kib KiB
}
