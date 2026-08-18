# Detect the memory available to this process. Test-only callers may override
# the proc and cgroup roots with fixture directories.
if (!$vector_big_test_estimated_peak_kib)
{
  --die vector_big_test_estimated_peak_kib must be set before this include
}
if (!$vector_big_test_proc_root)
{
  --let $vector_big_test_proc_root=/proc
}
if (!$vector_big_test_cgroup_root)
{
  --let $vector_big_test_cgroup_root=/sys/fs/cgroup
}

--let $vector_big_test_memory_result=$MYSQL_TMP_DIR/vector_big_test_memory.inc
let vector_big_test_estimated_peak_kib_for_perl = $vector_big_test_estimated_peak_kib;
let vector_big_test_memory_result_for_perl = $vector_big_test_memory_result;
let vector_big_test_proc_root_for_perl = $vector_big_test_proc_root;
let vector_big_test_cgroup_root_for_perl = $vector_big_test_cgroup_root;

--perl
use strict;
use warnings;
use File::Spec;

sub read_first_line {
  my ($path) = @_;
  open(my $input, '<', $path) or return undef;
  my $value = <$input>;
  close $input;
  return undef unless defined $value;
  chomp $value;
  return $value;
}

sub read_unsigned {
  my ($path) = @_;
  my $value = read_first_line($path);
  return undef unless defined $value && $value =~ /^\d+$/;
  return 0 + $value;
}

sub read_meminfo {
  my ($path) = @_;
  open(my $meminfo, '<', $path) or return;
  my ($available_kib, $total_kib);
  while (my $line = <$meminfo>) {
    $available_kib = $1 if $line =~ /^MemAvailable:\s+(\d+)\s+kB$/;
    $total_kib = $1 if $line =~ /^MemTotal:\s+(\d+)\s+kB$/;
  }
  close $meminfo;
  return ($available_kib, $total_kib);
}

sub cgroup_path {
  my ($root, $group, $file) = @_;
  $group =~ s{^/}{};
  return File::Spec->catfile($root, $group, $file);
}

sub detect_cgroup {
  my ($proc_root, $cgroup_root, $host_total_kib) = @_;
  my $membership_path = File::Spec->catfile($proc_root, 'self', 'cgroup');
  open(my $membership, '<', $membership_path) or return (0, undef);

  my ($version, $group);
  while (my $line = <$membership>) {
    chomp $line;
    if ($line =~ /^0::(.*)$/) {
      ($version, $group) = (2, $1);
      last;
    }
    if ($line =~ /^\d+:([^:]*):(.+)$/) {
      my ($controller_list, $candidate_group) = ($1, $2);
      my %controllers = map { $_ => 1 } split(/,/, $controller_list);
      if ($controllers{'memory'}) {
        ($version, $group) = (1, $candidate_group);
        last;
      }
    }
  }
  close $membership;
  return (1, undef) unless defined $version;

  my ($limit_path, $current_path);
  if ($version == 2) {
    $limit_path = cgroup_path($cgroup_root, $group, 'memory.max');
    $current_path = cgroup_path($cgroup_root, $group, 'memory.current');
  } else {
    my $memory_root = File::Spec->catdir($cgroup_root, 'memory');
    $limit_path = cgroup_path($memory_root, $group,
                             'memory.limit_in_bytes');
    $current_path = cgroup_path($memory_root, $group,
                               'memory.usage_in_bytes');
  }

  my $limit_text = read_first_line($limit_path);
  my $current_bytes = read_unsigned($current_path);
  return (0, undef) unless defined $limit_text && defined $current_bytes;
  return (1, undef) if $limit_text eq 'max';
  return (0, undef) unless $limit_text =~ /^\d+$/;

  my $limit_bytes = 0 + $limit_text;
  if (defined $host_total_kib && $host_total_kib > 0 &&
      $limit_bytes > $host_total_kib * 1024) {
    return (1, undef);
  }
  my $remaining_bytes =
    $limit_bytes > $current_bytes ? $limit_bytes - $current_bytes : 0;
  return (1, int($remaining_bytes / 1024));
}

my $estimated_peak_kib =
  $ENV{'vector_big_test_estimated_peak_kib_for_perl'};
my $result_file = $ENV{'vector_big_test_memory_result_for_perl'};
my $proc_root = $ENV{'vector_big_test_proc_root_for_perl'};
my $cgroup_root = $ENV{'vector_big_test_cgroup_root_for_perl'};
die "vector_big_test_estimated_peak_kib_for_perl is not set\n"
  unless defined $estimated_peak_kib && $estimated_peak_kib =~ /^\d+$/ &&
         $estimated_peak_kib > 0;
die "vector_big_test_memory_result_for_perl is not set\n"
  unless defined $result_file && length $result_file;
die "vector_big_test_proc_root_for_perl is not set\n"
  unless defined $proc_root && length $proc_root;
die "vector_big_test_cgroup_root_for_perl is not set\n"
  unless defined $cgroup_root && length $cgroup_root;

my ($host_available_kib, $host_total_kib) =
  read_meminfo(File::Spec->catfile($proc_root, 'meminfo'));
my ($cgroup_detected, $cgroup_available_kib) =
  detect_cgroup($proc_root, $cgroup_root, $host_total_kib);
my $detected = defined $host_available_kib && $host_available_kib > 0 &&
               $cgroup_detected;
my $available_kib = $detected ? $host_available_kib : 0;
if ($detected && defined $cgroup_available_kib &&
    $cgroup_available_kib < $available_kib) {
  $available_kib = $cgroup_available_kib;
}

my $required_kib = int(($estimated_peak_kib * 5 + 3) / 4);
my $sufficient = $detected && $available_kib >= $required_kib ? 1 : 0;

open(my $result, '>', $result_file)
  or die "cannot create $result_file: $!\n";
print {$result} "let \$vector_big_test_memory_detected = " .
                ($detected ? 1 : 0) . ";\n";
print {$result} "let \$vector_big_test_memory_sufficient = $sufficient;\n";
print {$result} "let \$vector_big_test_available_kib = $available_kib;\n";
print {$result} "let \$vector_big_test_required_kib = $required_kib;\n";
close $result;
EOF

--source $vector_big_test_memory_result
--remove_file $vector_big_test_memory_result
