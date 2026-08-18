#!/usr/bin/env perl
use strict;
use warnings;
use Fcntl qw(SEEK_SET);

my ($fbin_path, $docid_path, $rows, $dimension) = @ARGV;
die "usage: $0 <fbin> <docids> <rows> <dimension>\n"
  unless defined $fbin_path && defined $docid_path &&
         defined $rows && defined $dimension;

open(my $fbin, ">:raw", $fbin_path) or die "open $fbin_path: $!\n";
open(my $docids, ">:raw", $docid_path) or die "open $docid_path: $!\n";

print {$fbin} pack("L<L<", $rows, $dimension);
print {$docids} pack("Q<", $rows);

my $zero_tail = pack("f<", 0.0) x ($dimension - 1);
for (my $doc_id = 0; $doc_id < $rows; ++$doc_id) {
  print {$fbin} pack("f<", $doc_id) . $zero_tail;
  print {$docids} pack("Q<", $doc_id);
}

close($fbin) or die "close $fbin_path: $!\n";
close($docids) or die "close $docid_path: $!\n";
