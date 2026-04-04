# OPAC Semantic Search Module

A standalone JavaScript module that adds a "Semantic Search" toggle to the Evergreen OPAC. No build step required.

## Files

- `semantic-search.js` -- Main JavaScript module
- `semantic-search.css` -- Minimal CSS styles (designed to blend with Evergreen's Bootstrap OPAC theme)

## Quick Setup

### 1. Serve the static files

The RAG sidecar can serve these files, or copy them to your Evergreen web server. If using Apache, add an alias:

```apache
Alias /rag-static /path/to/evergreen-rag/src/evergreen_rag/static
<Directory /path/to/evergreen-rag/src/evergreen_rag/static>
    Require all granted
</Directory>
```

Or if the RAG sidecar serves static files (e.g., via FastAPI `StaticFiles`):

```python
from fastapi.staticfiles import StaticFiles
app.mount("/rag-static", StaticFiles(directory="static"), name="static")
```

### 2. Include in OPAC templates

Add these lines to your Evergreen OPAC template (typically `opac/parts/base.tt2` or the search page template):

```html
<!-- In the <head> section -->
<link rel="stylesheet" href="/rag-static/opac/semantic-search.css" />

<!-- Before </body> -->
<script src="/rag-static/opac/semantic-search.js"></script>
<script>
  EvergreenRAG.init({
    ragUrl: 'http://localhost:8000',   // RAG sidecar URL
    limit: 10,                          // Max results
    searchFormSelector: '#search-wrapper' // CSS selector for the OPAC search form
  });
</script>
```

### 3. Adjust the search form selector

The default `searchFormSelector` is `#search-wrapper`. If your OPAC template uses a different ID or class for the search form, update it:

```javascript
EvergreenRAG.init({
  ragUrl: 'http://localhost:8000',
  searchFormSelector: '#search-box'  // your OPAC's search form selector
});
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ragUrl` | string | `'/rag'` | Base URL of the RAG sidecar API |
| `limit` | integer | `10` | Maximum number of results to return |
| `minSimilarity` | float | `0.0` | Minimum similarity threshold (0.0-1.0) |
| `mergeWithKeyword` | boolean | `false` | If true, merge semantic results with keyword results from the current page via `/search/merged` |
| `semanticWeight` | float | `0.5` | Weight for semantic results when merging (0.0-1.0) |
| `bibRecordUrlPattern` | string | `'/eg/opac/record/{record_id}'` | URL pattern for bib record links. `{record_id}` is replaced with the actual ID. |
| `searchFormSelector` | string | `'#search-wrapper'` | CSS selector for the OPAC search form element |
| `timeout` | integer | `10000` | Request timeout in milliseconds |

## How It Works

1. On page load, the module injects a "Semantic Search" checkbox toggle next to the search form.
2. It performs a health check against the RAG sidecar. If the service is unavailable, the toggle is disabled with a warning icon.
3. When the toggle is checked and the user submits a search, the module intercepts the form submission and sends the query to the RAG API instead.
4. Results are displayed below the toggle, with each result linking to the standard Evergreen bib record display page.
5. If the RAG service fails during a search, an error message is shown and the user can switch back to keyword search.

## Graceful Degradation

- If the RAG sidecar is unreachable at page load, the toggle is disabled and shows a warning. Keyword search works normally.
- If the RAG sidecar fails during a search request, an error message is shown. The user can uncheck the toggle to return to keyword mode.
- The module never interferes with normal OPAC search when the toggle is unchecked.

## Org Unit Detection

The module automatically detects the selected org unit (library branch) from:
1. The `locg` URL parameter
2. A hidden `locg` form field
3. The `eg_org_unit` global JavaScript variable

This is passed to the RAG API as the `org_unit` filter.

## Customization

Override the CSS classes (prefixed with `rag-`) in your local OPAC stylesheet to match your library's branding. The styles are intentionally minimal and use Evergreen's default Bootstrap color palette.
