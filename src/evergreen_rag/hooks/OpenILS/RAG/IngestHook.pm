package OpenILS::RAG::IngestHook;

use strict;
use warnings;

use HTTP::Tiny;
use JSON::PP qw(encode_json decode_json);

our $VERSION = '0.1.0';

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

my $DEFAULT_RAG_URL = 'http://localhost:8000';
my $DEFAULT_TIMEOUT = 30;

sub new {
    my ($class, %opts) = @_;

    my $rag_url = $opts{rag_url}
        // _read_opensrf_config()
        // $ENV{EVERGREEN_RAG_URL}
        // $DEFAULT_RAG_URL;

    # Strip trailing slash
    $rag_url =~ s{/+$}{};

    return bless {
        rag_url => $rag_url,
        timeout => $opts{timeout} // $DEFAULT_TIMEOUT,
        logger  => $opts{logger},
    }, $class;
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

sub on_record_create_or_update {
    my ($self, @record_ids) = @_;

    return unless @record_ids;

    $self->_log('info',
        sprintf('RAG ingest triggered for %d record(s): %s',
            scalar @record_ids, join(', ', @record_ids)));

    my $result = $self->_post_ingest({ record_ids => \@record_ids });

    if ($result) {
        $self->_log('info',
            sprintf('RAG ingest completed: %d/%d embedded, %d failed',
                $result->{embedded}, $result->{total}, $result->{failed}));
    }

    # Always return true -- RAG failure must not break Evergreen ingest
    return 1;
}

sub reingest_all {
    my ($self) = @_;

    $self->_log('info', 'RAG full re-ingest triggered');

    my $result = $self->_post_ingest({ all => \1 });

    if ($result) {
        $self->_log('info',
            sprintf('RAG full re-ingest completed: %d/%d embedded, %d failed',
                $result->{embedded}, $result->{total}, $result->{failed}));
    }

    return $result;
}

sub health_check {
    my ($self) = @_;

    my $http = HTTP::Tiny->new(timeout => 5);
    my $url  = $self->{rag_url} . '/health';

    my $response = eval { $http->get($url) };
    if ($@ || !$response->{success}) {
        $self->_log('warn', 'RAG health check failed: '
            . ($@ // "$response->{status} $response->{reason}"));
        return 0;
    }

    my $data = eval { decode_json($response->{content}) };
    return 0 if $@;

    return $data->{status} eq 'ok' ? 1 : 0;
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

sub _post_ingest {
    my ($self, $payload) = @_;

    my $http = HTTP::Tiny->new(timeout => $self->{timeout});
    my $url  = $self->{rag_url} . '/ingest';

    my $response = eval {
        $http->post($url, {
            headers => { 'Content-Type' => 'application/json' },
            content => encode_json($payload),
        });
    };

    if ($@) {
        $self->_log('error', "RAG ingest request failed: $@");
        return undef;
    }

    unless ($response->{success}) {
        $self->_log('error',
            sprintf('RAG ingest HTTP error: %s %s',
                $response->{status}, $response->{reason}));
        return undef;
    }

    my $data = eval { decode_json($response->{content}) };
    if ($@) {
        $self->_log('error', "RAG ingest response decode error: $@");
        return undef;
    }

    return $data;
}

sub _read_opensrf_config {
    # Attempt to read RAG URL from opensrf.xml via OpenSRF if available.
    # Returns undef if OpenSRF is not loaded (e.g. standalone CLI usage).
    my $url = eval {
        require OpenSRF::Utils::SettingsClient;
        OpenSRF::Utils::SettingsClient->new->config_value(
            'apps', 'open-ils.ingest', 'app_settings', 'rag_base_url'
        );
    };
    return $url;
}

sub _log {
    my ($self, $level, $msg) = @_;

    if ($self->{logger}) {
        # OpenSRF logger or compatible object
        if ($self->{logger}->can($level)) {
            $self->{logger}->$level($msg);
        }
    } else {
        # Fallback to stderr
        warn "[RAG IngestHook] [$level] $msg\n";
    }
}

1;

__END__

=head1 NAME

OpenILS::RAG::IngestHook - Trigger RAG embedding ingest from Evergreen

=head1 SYNOPSIS

    use OpenILS::RAG::IngestHook;

    # Create a hook instance (auto-detects config from opensrf.xml or env)
    my $hook = OpenILS::RAG::IngestHook->new();

    # Or with explicit URL
    my $hook = OpenILS::RAG::IngestHook->new(
        rag_url => 'http://rag-sidecar:8000',
        timeout => 60,
    );

    # Trigger ingest when records are created or updated
    $hook->on_record_create_or_update(1001, 1002, 1003);

    # Full re-ingest
    $hook->reingest_all();

    # Health check
    if ($hook->health_check()) {
        print "RAG service is healthy\n";
    }

=head1 DESCRIPTION

This module provides an integration hook for triggering RAG (Retrieval-Augmented
Generation) embedding ingest from Evergreen's ingest pipeline. When bibliographic
records are created or updated in Evergreen, this hook notifies the RAG sidecar
service to re-embed those records.

B<Important:> The RAG service being unavailable will I<never> cause Evergreen's
ingest pipeline to fail. All RAG communication errors are caught, logged, and
silently ignored to ensure the core cataloging workflow is unaffected.

=head1 CONFIGURATION

The RAG service URL is resolved in this order:

=over 4

=item 1. Explicit C<rag_url> parameter to C<new()>

=item 2. OpenSRF C<opensrf.xml> setting: C<apps.open-ils.ingest.app_settings.rag_base_url>

=item 3. Environment variable C<EVERGREEN_RAG_URL>

=item 4. Default: C<http://localhost:8000>

=back

To configure via C<opensrf.xml>:

    <open-ils.ingest>
      <app_settings>
        <rag_base_url>http://rag-sidecar:8000</rag_base_url>
      </app_settings>
    </open-ils.ingest>

=head1 METHODS

=head2 new(%opts)

Constructor. Options:

=over 4

=item rag_url - RAG service base URL (optional, auto-detected)

=item timeout - HTTP request timeout in seconds (default: 30)

=item logger  - Logger object with info/warn/error methods (optional)

=back

=head2 on_record_create_or_update(@record_ids)

Trigger RAG ingest for the given bibliographic record IDs. Always returns true,
even if the RAG service is unavailable.

=head2 reingest_all()

Trigger a full re-ingest of all bibliographic records. Returns the ingest
statistics hashref on success, or undef on failure.

=head2 health_check()

Check if the RAG service is healthy. Returns 1 if healthy, 0 otherwise.

=head1 INTEGRATION WITH EVERGREEN INGEST

To hook into Evergreen's existing ingest pipeline, add calls to
C<on_record_create_or_update> in the appropriate ingest action handlers.
For example, in the biblio record create/update path:

    use OpenILS::RAG::IngestHook;

    my $rag_hook = OpenILS::RAG::IngestHook->new();

    # After a successful record create/update in the existing pipeline:
    $rag_hook->on_record_create_or_update($record_id);

=head1 DEPENDENCIES

=over 4

=item L<HTTP::Tiny> (core since Perl 5.14)

=item L<JSON::PP> (core since Perl 5.14)

=back

No non-core Perl modules are required.

=head1 AUTHOR

Evergreen RAG Project

=cut
