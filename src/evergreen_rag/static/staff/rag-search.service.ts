import { Injectable, Inject, InjectionToken } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, throwError, timer } from 'rxjs';
import { catchError, retryWhen, mergeMap } from 'rxjs/operators';

export const RAG_BASE_URL = new InjectionToken<string>('RAG_BASE_URL');

export interface SearchResult {
  record_id: number;
  similarity: number;
  chunk_text: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
  model: string;
}

export interface SearchOptions {
  limit?: number;
  org_unit?: number;
  format?: string;
  min_similarity?: number;
}

export interface MergedSearchOptions {
  limit?: number;
  weights?: { semantic?: number; keyword?: number };
}

export interface HealthResponse {
  status: string;
  checks: { [key: string]: boolean };
}

@Injectable()
export class RagSearchService {
  private readonly maxRetries = 2;
  private readonly retryDelayMs = 1000;

  constructor(
    private http: HttpClient,
    @Inject(RAG_BASE_URL) private baseUrl: string
  ) {}

  search(query: string, options: SearchOptions = {}): Observable<SearchResponse> {
    const body = { query, ...options };
    return this.http
      .post<SearchResponse>(`${this.baseUrl}/search`, body)
      .pipe(this.retryOnServerError());
  }

  mergedSearch(
    query: string,
    keywordResults: number[],
    options: MergedSearchOptions = {}
  ): Observable<SearchResponse> {
    const body = {
      query,
      keyword_results: keywordResults,
      ...options,
    };
    return this.http
      .post<SearchResponse>(`${this.baseUrl}/search/merged`, body)
      .pipe(this.retryOnServerError());
  }

  getHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`);
  }

  private retryOnServerError() {
    return retryWhen<any>((errors: Observable<any>) =>
      errors.pipe(
        mergeMap((error, attempt) => {
          if (attempt >= this.maxRetries || error.status < 500) {
            return throwError(() => error);
          }
          return timer(this.retryDelayMs * (attempt + 1));
        })
      )
    );
  }
}
