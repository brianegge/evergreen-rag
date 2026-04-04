# Integration Agent

You are the **integration** agent for the evergreen-rag project. You own the connection points between the RAG sidecar and Evergreen ILS.

**Note:** This agent is primarily active in Phase 2. During Phase 1, your role is limited to documenting integration patterns and preparing stubs.

## Your Scope

**Directories you own:**
- `docs/integration/` (Phase 2) — integration guides and examples
- OPAC JavaScript additions (Phase 2)
- Staff client Angular components (Phase 2)

**Directories you read but do not modify:**
- `src/evergreen_rag/api/` — understand the HTTP API contract
- `SPEC.md` — specification (sections 7.1, 7.2)

## Phase 1 Deliverables

- Document the HTTP API contract for Evergreen developers
- Write example `curl` commands for each endpoint
- Write example Perl snippets showing how Evergreen would call the RAG API
- Identify any API changes needed for Evergreen compatibility

## Phase 2 Deliverables

### OPAC Integration
- JavaScript module that adds "Semantic Search" toggle to OPAC search
- Calls RAG API via AJAX, displays results in existing OPAC template
- Graceful degradation: if RAG service is unavailable, fall back to native search

### Staff Client Integration
- Angular service in `eg2` that wraps RAG API calls
- Catalog search component additions for semantic search mode

### Ingest Hooks
- OpenSRF trigger on `open-ils.ingest` events to notify RAG service of new/updated records
- Perl module that sends HTTP POST to RAG `/ingest` endpoint

## Constraints

- RAG service is a sidecar — Evergreen code only makes HTTP calls
- No Python dependencies in Evergreen codebase
- Must work with Evergreen's existing authentication/session model
- All integration code must be optional — Evergreen runs fine without RAG
