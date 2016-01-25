#!/usr/bin/perl

use strict;
use warnings;
use utf8;
use Getopt::Long;
use List::Util qw(sum min max shuffle);
binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my $THRESH = 20000; # Unk words less than the most frequent
my $SYM = "<unk>"; # The symbol to use for unknown words
GetOptions(
"thresh=s" => \$THRESH,
"sym=s" => \$SYM,
);

if(@ARGV != 0) {
    print STDERR "Usage: $0\n";
    exit 1;
}

my @crp;
my %cnt;
while(<STDIN>) {
    chomp;
    push @crp, $_;
    for(split(/ /)) {
        $cnt{$_}++;
    }
}

my %invocab;
for(sort { $cnt{$b} <=> $cnt{$a} } keys %cnt) {
  if($THRESH-- == 0) { last; }
  $invocab{$_}++;
}

for(@crp) {
    my @arr = map { defined($invocab{$_}) ? $_ : $SYM } split(/ /);
    print "@arr\n";
}
