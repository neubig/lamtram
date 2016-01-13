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

if(@ARGV < 1) {
    print STDERR "Usage: $0 PREFIX [INTERLEAVE]\n";
    exit 1;
}
my $inter = $ARGV[1] ? $ARGV[1] : 1;

my @vals = qw(0 1 2 3 4 5 6 7 8 9 A B C D E F);
sub tohex {
    my $v = shift;
    $v = min(240, $v*500);
    my $str = $vals[int($v/16)].$vals[int($v)%16];
    return "$str$str$str";
}

my $id = 0;
my ($src, $trg, @context);
while(<STDIN>) {
    if(/^ll/) {
        foreach my $fid (0 .. $inter-1) {
            open FILE0, ">:utf8", "$ARGV[0]$id-$fid.html" or die "Couldn't open $ARGV[0]$id-$fid.html\n";
            print FILE0 "<html><body><table>";
            my @sa = split(/ +/, $src); push @sa, "&lt;s&gt;";
            print FILE0 "<tr><td>&nbsp;</td>"; for(@sa) { print FILE0 "<td>$_</td>"; } print FILE0 "</tr>";
            my @ta = split(/ +/, $trg); push @ta, "&lt;s&gt;";
            foreach my $i (0 .. $#ta) {
                print FILE0 "<tr><td>$ta[$i]</td>";
                for(split(/,/, $context[$i*$inter+$fid])) { print FILE0 "<td bgcolor=\"".tohex($_)."\">&nbsp;</td>"; }
                print FILE0 "<tr>";
            }
            print FILE0 "</table></body></html>";
            close FILE0;
        }
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
