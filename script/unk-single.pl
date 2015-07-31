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
    my @arr = map { ($cnt{$_} > 1) ? $_ : "<unk>" } split(/ /);
    print "@arr\n";
}
