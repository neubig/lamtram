#!/usr/bin/perl

use strict;
use warnings;
use utf8;
use Getopt::Long;
use List::Util qw(sum min max shuffle);
binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my $THRESH = 1; # Unk words less than this frequency
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

for(@crp) {
    my @arr = map { ($cnt{$_} > $THRESH) ? $_ : $SYM } split(/ /);
    print "@arr\n";
}
