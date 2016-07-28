#!/usr/bin/perl

use strict;
use warnings;
use utf8;
use Getopt::Long;
use List::Util qw(sum min max shuffle);
binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my $LANG = "en";
GetOptions(
"lang=s" => \$LANG,
);

if(@ARGV != 0) {
  print STDERR "Usage: $0\n";
  exit 1;
}

my $THRESH = -20;
print "<s>\t<s>\t1.0\n";
print "<unk>\t<unk>\t1.0\n";
while(<STDIN>) {
  chomp;
  my @arr = split(/\t/);
  if($arr[2] > $THRESH) {
    $arr[2] = exp($arr[2]);
    print join("\t", @arr)."\n";
  }
}
