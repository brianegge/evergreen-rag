# Evergreen RAG Staff Client Component

Angular component for integrating semantic search into Evergreen's eg2 staff client.

## Files

| File | Purpose |
|------|---------|
| `rag-search.service.ts` | Angular service wrapping RAG API calls |
| `rag-search.component.ts` | Search component with toggle, input, and results display |
| `rag-search.component.html` | Component template |
| `rag-search.component.css` | Component styles |
| `rag-search.module.ts` | Angular module exporting the component and service |

## Integration with eg2

### 1. Copy files into the eg2 source tree

Copy this directory into the eg2 staff client source:

```
cp -r src/evergreen_rag/static/staff/ \
  /path/to/Evergreen/Open-ILS/src/eg2/src/app/rag-search/
```

### 2. Import the module

In your target module (e.g., the catalog module), import `RagSearchModule`:

```typescript
import { RagSearchModule } from '../rag-search/rag-search.module';

@NgModule({
  imports: [
    // ... existing imports
    RagSearchModule.forRoot('http://localhost:8000'),
  ],
})
export class CatalogModule {}
```

Set the base URL to the address where the RAG sidecar is running. For production, this should match your deployment configuration (e.g., `https://your-evergreen-host:8000` or a reverse-proxied path like `/rag`).

### 3. Add the component to a template

Place the `<eg-rag-search>` selector in a catalog search template:

```html
<!-- In your catalog search template -->
<eg-rag-search></eg-rag-search>
```

### 4. Merged search mode

To use merged search (combining keyword + semantic results), pass keyword result IDs to the component programmatically:

```typescript
import { ViewChild } from '@angular/core';
import { RagSearchComponent } from '../rag-search/rag-search.component';

export class CatalogSearchComponent {
  @ViewChild(RagSearchComponent) ragSearch: RagSearchComponent;

  onKeywordSearchComplete(recordIds: number[]): void {
    // Feed keyword results into RAG component for RRF merging
    this.ragSearch.setKeywordResults(recordIds);
  }
}
```

Then enable "Merge with keyword results" in the UI, or set `mergedMode = true` programmatically.

### 5. Using the service directly

You can also inject `RagSearchService` directly for custom integrations:

```typescript
import { RagSearchService } from '../rag-search/rag-search.service';

export class MyComponent {
  constructor(private ragService: RagSearchService) {}

  doSearch(): void {
    this.ragService.search('books about grief', { limit: 5 }).subscribe(
      (response) => console.log(response.results)
    );
  }
}
```

## API Endpoints Used

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/search` | Semantic search |
| POST | `/search/merged` | Merged keyword + semantic search (RRF) |
| GET | `/health` | Service health check |

See `docs/integration/api-reference.md` for full API documentation.

## Requirements

- Angular 12+ (compatible with Evergreen's eg2 client)
- `@angular/common/http` for HTTP calls
- `@angular/forms` for ngModel binding
- RAG sidecar service running and accessible from the browser
