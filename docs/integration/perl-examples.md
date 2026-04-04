# Evergreen RAG API -- Perl Integration Examples

These examples show how to call the RAG sidecar API from Perl, as would be done within Evergreen's codebase. Both lightweight (`HTTP::Tiny`, core since Perl 5.14) and full-featured (`LWP::UserAgent`) approaches are shown.

---

## Configuration

The RAG sidecar URL should be configurable. In Evergreen, this would typically come from `opensrf.xml` or an org unit setting:

```perl
# From opensrf.xml or environment variable
my $RAG_BASE_URL = $ENV{EVERGREEN_RAG_URL} // 'http://localhost:8000';
```

---

## Using HTTP::Tiny (Lightweight, Core Module)

`HTTP::Tiny` is included with Perl 5.14+ and has no external dependencies, making it the simplest option.

### Semantic Search

```perl
use HTTP::Tiny;
use JSON::XS qw(encode_json decode_json);

sub rag_search {
    my ($query_text, %opts) = @_;

    my $http = HTTP::Tiny->new(timeout => 10);

    my $payload = encode_json({
        query          => $query_text,
        limit          => $opts{limit}          // 10,
        org_unit       => $opts{org_unit}       // undef,
        format         => $opts{format}         // undef,
        min_similarity => $opts{min_similarity} // 0.0,
    });

    my $response = $http->post(
        "$RAG_BASE_URL/search",
        {
            headers => { 'Content-Type' => 'application/json' },
            content => $payload,
        }
    );

    unless ($response->{success}) {
        warn "RAG search failed: $response->{status} $response->{reason}";
        return undef;
    }

    return decode_json($response->{content});
}

# Usage:
my $results = rag_search(
    "books about coping with grief for teenagers",
    limit    => 5,
    org_unit => 1,
);

if ($results) {
    for my $hit (@{ $results->{results} }) {
        printf "Record %d (score: %.2f): %s\n",
            $hit->{record_id},
            $hit->{similarity},
            substr($hit->{chunk_text}, 0, 80);
    }
}
```

### Ingest Records

```perl
sub rag_ingest {
    my (@record_ids) = @_;

    my $http = HTTP::Tiny->new(timeout => 300);  # ingest can be slow

    my $payload = encode_json({
        record_ids => \@record_ids,
    });

    my $response = $http->post(
        "$RAG_BASE_URL/ingest",
        {
            headers => { 'Content-Type' => 'application/json' },
            content => $payload,
        }
    );

    unless ($response->{success}) {
        warn "RAG ingest failed: $response->{status} $response->{reason}";
        return undef;
    }

    return decode_json($response->{content});
}

# Usage:
my $stats = rag_ingest(100, 200, 300);
if ($stats) {
    print "Ingested: $stats->{embedded} / $stats->{total}\n";
}
```

### Health Check

```perl
sub rag_health_check {
    my $http = HTTP::Tiny->new(timeout => 5);
    my $response = $http->get("$RAG_BASE_URL/health");

    return 0 unless $response->{success};

    my $data = decode_json($response->{content});
    return $data->{status} eq 'ok' ? 1 : 0;
}
```

---

## Using LWP::UserAgent (Full-Featured)

`LWP::UserAgent` provides more control over retries, SSL, proxies, and cookie handling. Evergreen already depends on `libwww-perl`.

### Semantic Search

```perl
use LWP::UserAgent;
use HTTP::Request;
use JSON::XS qw(encode_json decode_json);

sub rag_search_lwp {
    my ($query_text, %opts) = @_;

    my $ua = LWP::UserAgent->new(
        timeout => 10,
        agent   => 'Evergreen-RAG-Client/1.0',
    );

    my $payload = encode_json({
        query          => $query_text,
        limit          => $opts{limit}          // 10,
        org_unit       => $opts{org_unit}       // undef,
        format         => $opts{format}         // undef,
        min_similarity => $opts{min_similarity} // 0.0,
    });

    my $req = HTTP::Request->new(
        POST => "$RAG_BASE_URL/search",
    );
    $req->header('Content-Type' => 'application/json');
    $req->content($payload);

    my $response = $ua->request($req);

    unless ($response->is_success) {
        warn sprintf "RAG search failed: %s %s",
            $response->code, $response->message;
        return undef;
    }

    return decode_json($response->decoded_content);
}
```

### Merged Search (Phase 2)

```perl
sub rag_merged_search_lwp {
    my ($query_text, $keyword_ids, %opts) = @_;

    my $ua = LWP::UserAgent->new(
        timeout => 10,
        agent   => 'Evergreen-RAG-Client/1.0',
    );

    my $payload = encode_json({
        query              => $query_text,
        keyword_result_ids => $keyword_ids,
        limit              => $opts{limit}           // 10,
        semantic_weight    => $opts{semantic_weight}  // 0.5,
    });

    my $req = HTTP::Request->new(
        POST => "$RAG_BASE_URL/search/merged",
    );
    $req->header('Content-Type' => 'application/json');
    $req->content($payload);

    my $response = $ua->request($req);

    unless ($response->is_success) {
        warn sprintf "RAG merged search failed: %s %s",
            $response->code, $response->message;
        return undef;
    }

    return decode_json($response->decoded_content);
}
```

---

## Calling from an OpenSRF Method

In Evergreen, API methods are registered as OpenSRF services. Here is how a RAG search could be exposed as an OpenSRF method within an existing Evergreen service:

```perl
package OpenILS::Application::Search::RAG;

use base qw(OpenILS::Application);
use strict;
use warnings;

use OpenSRF::Utils::Logger qw($logger);
use OpenILS::Utils::CStoreEditor qw(:funcs);
use HTTP::Tiny;
use JSON::XS qw(encode_json decode_json);

my $RAG_BASE_URL;

sub initialize {
    # Read RAG URL from opensrf.xml at startup
    $RAG_BASE_URL = OpenSRF::Utils::SettingsClient
        ->new
        ->config_value('apps', 'open-ils.search', 'app_settings', 'rag_base_url')
        // 'http://localhost:8000';

    $logger->info("RAG sidecar URL: $RAG_BASE_URL");
}

__PACKAGE__->register_method(
    method    => 'rag_search',
    api_name  => 'open-ils.search.rag',
    api_level => 1,
    argc      => 2,
    signature => {
        desc   => 'Perform a semantic search via the RAG sidecar',
        params => [
            { name => 'auth',  desc => 'Auth token', type => 'string' },
            { name => 'args',  desc => 'Search arguments hash', type => 'hash' },
        ],
        return => {
            desc => 'Search results with record IDs and similarity scores',
            type => 'hash',
        },
    },
);

sub rag_search {
    my ($self, $conn, $auth, $args) = @_;

    # Validate auth token
    my $e = new_editor(authtoken => $auth);
    return $e->event unless $e->checkauth;

    my $query_text = $args->{query}
        or return OpenILS::Event->new('BAD_PARAMS', note => 'query required');

    my $http = HTTP::Tiny->new(timeout => 10);

    my $payload = encode_json({
        query          => $query_text,
        limit          => $args->{limit}          // 10,
        org_unit       => $args->{org_unit}       // undef,
        format         => $args->{format}         // undef,
        min_similarity => $args->{min_similarity} // 0.0,
    });

    my $response = $http->post(
        "$RAG_BASE_URL/search",
        {
            headers => { 'Content-Type' => 'application/json' },
            content => $payload,
        }
    );

    unless ($response->{success}) {
        $logger->error("RAG search failed: $response->{status} $response->{reason}");
        return OpenILS::Event->new(
            'INTERNAL_SERVER_ERROR',
            note => "RAG service unavailable",
        );
    }

    return decode_json($response->{content});
}

1;
```

### Registering in opensrf.xml

Add the RAG URL to your `opensrf.xml` app settings for `open-ils.search`:

```xml
<open-ils.search>
  <app_settings>
    <!-- existing settings ... -->
    <rag_base_url>http://localhost:8000</rag_base_url>
  </app_settings>
</open-ils.search>
```

---

## Error Handling Patterns

All examples above include basic error handling. Here is a more robust pattern with retries and fallback:

```perl
use HTTP::Tiny;
use JSON::XS qw(encode_json decode_json);
use Time::HiRes qw(sleep);

sub rag_search_with_retry {
    my ($query_text, %opts) = @_;

    my $max_retries = $opts{retries} // 2;
    my $http = HTTP::Tiny->new(timeout => $opts{timeout} // 10);

    my $payload = encode_json({
        query          => $query_text,
        limit          => $opts{limit}          // 10,
        org_unit       => $opts{org_unit}       // undef,
        min_similarity => $opts{min_similarity} // 0.0,
    });

    for my $attempt (0 .. $max_retries) {
        my $response = $http->post(
            "$RAG_BASE_URL/search",
            {
                headers => { 'Content-Type' => 'application/json' },
                content => $payload,
            }
        );

        if ($response->{success}) {
            return decode_json($response->{content});
        }

        # Don't retry client errors (4xx)
        if ($response->{status} >= 400 && $response->{status} < 500) {
            warn "RAG client error: $response->{status} $response->{content}";
            return undef;
        }

        # Retry on server errors (5xx) and network failures
        if ($attempt < $max_retries) {
            my $delay = 0.5 * (2 ** $attempt);  # exponential backoff
            warn "RAG request failed (attempt $attempt), retrying in ${delay}s...";
            sleep($delay);
        }
    }

    warn "RAG search failed after $max_retries retries";
    return undef;
}
```
