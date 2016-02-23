#!/usr/bin/perl

use strict;
use warnings;
use utf8;
use Getopt::Long;
use List::Util qw(sum min max shuffle);
binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my $SYM = "<unk>"; # The symbol to use for unknown words
GetOptions(
"sym=s" => \$SYM,
);

if(@ARGV != 1) {
    print STDERR "Usage: $0 VOCAB_FILE\n";
    exit 1;
}

my %vocab;
open FILE0, "<:utf8", $ARGV[0] or die "Couldn't open $ARGV[0]";
while(<FILE0>) {
  chomp;
  $vocab{$_}++;
}

while(<STDIN>) {
  chomp;
  my @arr = map { $vocab{$_} ? $_ : $SYM } split(/ /);
  print "@arr\n";
}
