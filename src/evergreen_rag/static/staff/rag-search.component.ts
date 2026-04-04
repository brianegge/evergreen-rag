import { Component, OnInit, OnDestroy } from '@angular/core';
import { Subject } from 'rxjs';
import { takeUntil, finalize } from 'rxjs/operators';

import {
  RagSearchService,
  SearchResult,
  SearchResponse,
} from './rag-search.service';

@Component({
  selector: 'eg-rag-search',
  templateUrl: './rag-search.component.html',
  styleUrls: ['./rag-search.component.css'],
})
export class RagSearchComponent implements OnInit, OnDestroy {
  query = '';
  results: SearchResult[] = [];
  loading = false;
  error: string | null = null;
  semanticEnabled = true;
  mergedMode = false;
  healthy: boolean | null = null;
  totalResults = 0;
  modelName = '';

  /** Record IDs from keyword search, supplied externally for merged mode. */
  keywordResultIds: number[] = [];

  private destroy$ = new Subject<void>();

  constructor(private ragService: RagSearchService) {}

  ngOnInit(): void {
    this.checkHealth();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  checkHealth(): void {
    this.ragService
      .getHealth()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (resp) => {
          this.healthy = resp.status === 'ok';
        },
        error: () => {
          this.healthy = false;
        },
      });
  }

  search(): void {
    const trimmed = this.query.trim();
    if (!trimmed) {
      return;
    }

    this.loading = true;
    this.error = null;
    this.results = [];

    const search$ =
      this.mergedMode && this.keywordResultIds.length > 0
        ? this.ragService.mergedSearch(trimmed, this.keywordResultIds)
        : this.ragService.search(trimmed);

    search$
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.loading = false;
        })
      )
      .subscribe({
        next: (resp: SearchResponse) => {
          this.results = resp.results;
          this.totalResults = resp.total;
          this.modelName = resp.model;
        },
        error: (err) => {
          if (err.status === 503) {
            this.error =
              'The semantic search service is temporarily unavailable. Please try again later.';
          } else if (err.status === 422) {
            this.error = 'Invalid search query. Please check your input.';
          } else {
            this.error = 'An unexpected error occurred. Please try again.';
          }
        },
      });
  }

  toggleSemantic(): void {
    this.semanticEnabled = !this.semanticEnabled;
  }

  toggleMergedMode(): void {
    this.mergedMode = !this.mergedMode;
  }

  /** Build a staff catalog link for a given record ID. */
  catalogLink(recordId: number): string {
    return `/eg2/staff/catalog/record/${recordId}`;
  }

  /** Accept keyword results from external caller (e.g., catalog search). */
  setKeywordResults(ids: number[]): void {
    this.keywordResultIds = ids;
  }
}
