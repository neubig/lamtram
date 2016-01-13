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

if(@ARGV != 1) {
    print STDERR "Usage: $0 PREFIX\n";
    exit 1;
}

my @vals = qw(0 1 2 3 4 5 6 7 8 9 A B C D E F);
sub tohex {
    my $v = shift;
    $v = min(240, $v*500);
    my $str = $vals[int($v/16)].$vals[int($v)%16];
    return "$str$str$str";
}

sub create_context {
  my @ret;
  for(@_) {
    my @arr = split(/,/, $_);
    foreach my $i (0 .. $#arr) {
      $ret[$i] += $arr[$i];
    }
  }
  return map { $_/@_ } @ret;
}

my $id = 0;
my ($src, $trg, @context);
while(<STDIN>) {
    if(/^ll/) {
        open FILE0, ">:utf8", "$ARGV[0]$id.html" or die "Couldn't open $ARGV[0]$id.html\n";
        print FILE0 "<html><body><table>";
        my @sa = split(/ +/, $src); push @sa, "&lt;s&gt;";
        print FILE0 "<tr><td>&nbsp;</td>"; for(@sa) { print FILE0 "<td>$_</td>"; } print FILE0 "</tr>";
        my @ta = split(/ +/, $trg); push @ta, "&lt;s&gt;";
        if(@context % @ta != 0) { die "Context not divisible by ta: context=".scalar(@context).", ta=".scalar(@ta)."\n"; }
        my $num_ens = @context / @ta;
        foreach my $i (0 .. $#ta) {
            print FILE0 "<tr><td>$ta[$i]</td>";
            my @align = create_context($context[$i*$num_ens,($i+1)*$num_ens-1]);
            for(@align) { print FILE0 "<td bgcolor=\"".tohex($_)."\">&nbsp;</td>"; }
            print FILE0 "<tr>";
        }
        print FILE0 "</table></body></html>";
        close FILE0;
        @context = ();
        $id++;
    } elsif(/SentLL trg: (.*)/) {
        $trg = $1;
    } elsif(/SentLL src: (.*)/) {
        $src = $1;
    } elsif(/Alignments: \[(.*)\]/) {
        push @context, $1;
    }
}
