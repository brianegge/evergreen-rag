#!/usr/bin/perl
# rag-reingest.pl -- CLI tool for manual RAG ingest of bibliographic records
#
# Usage:
#   rag-reingest.pl 1001 1002 1003       # re-ingest specific records
#   rag-reingest.pl --all                 # full re-ingest
#   rag-reingest.pl --health              # check RAG service health
#   rag-reingest.pl --url http://host:8000 1001  # custom RAG URL

use strict;
use warnings;

use FindBin;
use lib "$FindBin::Bin";
use lib "$FindBin::Bin/..";

use Getopt::Long;
use Pod::Usage;

# Inline a minimal HTTP client to keep this script self-contained.
# In production, OpenILS::RAG::IngestHook would be used instead.
use HTTP::Tiny;
use JSON::PP qw(encode_json decode_json);

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

my $rag_url  = $ENV{EVERGREEN_RAG_URL} // 'http://localhost:8000';
my $timeout  = 300;
my $do_all   = 0;
my $do_health = 0;
my $help     = 0;
my $verbose  = 0;

GetOptions(
    'url=s'    => \$rag_url,
    'timeout=i'=> \$timeout,
    'all'      => \$do_all,
    'health'   => \$do_health,
    'verbose'  => \$verbose,
    'help|h'   => \$help,
) or pod2usage(2);

pod2usage(1) if $help;

$rag_url =~ s{/+$}{};

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

if ($do_health) {
    exit cmd_health();
}

if ($do_all) {
    exit cmd_reingest_all();
}

my @record_ids = @ARGV;
if (!@record_ids) {
    print STDERR "Error: provide record IDs or use --all for full re-ingest\n";
    pod2usage(2);
}

# Validate record IDs are integers
for my $id (@record_ids) {
    unless ($id =~ /^\d+$/) {
        die "Error: '$id' is not a valid record ID (must be a positive integer)\n";
    }
}

exit cmd_reingest_records(@record_ids);

# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

sub cmd_health {
    my $http = HTTP::Tiny->new(timeout => 5);
    my $response = $http->get("$rag_url/health");

    unless ($response->{success}) {
        printf STDERR "RAG service unreachable: %s %s\n",
            $response->{status}, $response->{reason};
        return 1;
    }

    my $data = decode_json($response->{content});
    printf "RAG service status: %s\n", $data->{status};

    if ($verbose && $data->{checks}) {
        for my $check (sort keys %{ $data->{checks} }) {
            printf "  %-20s %s\n", $check,
                $data->{checks}{$check} ? 'OK' : 'FAIL';
        }
    }

    return $data->{status} eq 'ok' ? 0 : 1;
}

sub cmd_reingest_all {
    print "Starting full RAG re-ingest...\n";

    my $result = post_ingest({ all => \1 });
    return 1 unless $result;

    printf "Re-ingest complete: %d total, %d embedded, %d failed\n",
        $result->{total}, $result->{embedded}, $result->{failed};

    return $result->{failed} > 0 ? 1 : 0;
}

sub cmd_reingest_records {
    my @ids = @_;

    printf "Ingesting %d record(s): %s\n", scalar @ids, join(', ', @ids);

    my $result = post_ingest({ record_ids => \@ids });
    return 1 unless $result;

    printf "Ingest complete: %d total, %d embedded, %d failed\n",
        $result->{total}, $result->{embedded}, $result->{failed};

    return $result->{failed} > 0 ? 1 : 0;
}

sub post_ingest {
    my ($payload) = @_;

    my $http = HTTP::Tiny->new(timeout => $timeout);
    my $url  = "$rag_url/ingest";

    print "POST $url\n" if $verbose;

    my $response = eval {
        $http->post($url, {
            headers => { 'Content-Type' => 'application/json' },
            content => encode_json($payload),
        });
    };

    if ($@) {
        print STDERR "Request failed: $@\n";
        return undef;
    }

    unless ($response->{success}) {
        printf STDERR "HTTP error: %s %s\n",
            $response->{status}, $response->{reason};
        if ($verbose && $response->{content}) {
            print STDERR "Response: $response->{content}\n";
        }
        return undef;
    }

    return decode_json($response->{content});
}

__END__

=head1 NAME

rag-reingest.pl - CLI tool for manual RAG embedding ingest

=head1 SYNOPSIS

    # Re-ingest specific records
    rag-reingest.pl 1001 1002 1003

    # Full re-ingest of all records
    rag-reingest.pl --all

    # Check RAG service health
    rag-reingest.pl --health

    # Use a custom RAG service URL
    rag-reingest.pl --url http://rag-host:8000 1001 1002

    # Verbose output
    rag-reingest.pl --verbose --all

=head1 DESCRIPTION

A standalone CLI tool for triggering RAG embedding ingest outside of the normal
Evergreen ingest pipeline. Useful for:

=over 4

=item * Manual re-ingest of specific records after MARC edits

=item * Full re-ingest after model changes or schema updates

=item * Verifying RAG service connectivity

=item * Scripted batch operations via cron or system administration

=back

=head1 OPTIONS

=over 4

=item B<--url> URL

RAG sidecar base URL. Defaults to C<$EVERGREEN_RAG_URL> environment variable,
or C<http://localhost:8000>.

=item B<--timeout> SECONDS

HTTP request timeout in seconds. Default: 300 (5 minutes).

=item B<--all>

Trigger a full re-ingest of all bibliographic records.

=item B<--health>

Check RAG service health and exit. Exit code 0 = healthy, 1 = unhealthy.

=item B<--verbose>

Print additional details (HTTP requests, individual check results).

=item B<--help>, B<-h>

Show this help message.

=back

=head1 EXIT CODES

=over 4

=item B<0> - Success (all records ingested, or health check passed)

=item B<1> - Failure (network error, service error, or some records failed)

=item B<2> - Usage error (bad arguments)

=back

=head1 ENVIRONMENT

=over 4

=item C<EVERGREEN_RAG_URL>

Default RAG service URL if C<--url> is not specified.

=back

=head1 DEPENDENCIES

=over 4

=item L<HTTP::Tiny> (core since Perl 5.14)

=item L<JSON::PP> (core since Perl 5.14)

=back

=head1 EXAMPLES

    # Re-ingest after a batch MARC import
    rag-reingest.pl $(cat imported_ids.txt)

    # Cron job for nightly full re-ingest
    0 2 * * * /path/to/rag-reingest.pl --all >> /var/log/rag-reingest.log 2>&1

    # Quick health check in a monitoring script
    rag-reingest.pl --health --verbose || alert "RAG service down"

=head1 AUTHOR

Evergreen RAG Project

=cut
