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

my %inv;
open FILE0, "<:utf8", $ARGV[0] or die "Couldn't open $ARGV[0]\n";
while(<FILE0>) {
  chomp;
  for(split(/ /)) {
    $inv{$_}++;
  }
}

while(<STDIN>) {
  chomp;
  my @arr = map { $inv{$_} ? $_ : $SYM } split(/ /);
  print "@arr\n";
}
