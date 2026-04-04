import { NgModule, ModuleWithProviders } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';

import { RagSearchComponent } from './rag-search.component';
import { RagSearchService, RAG_BASE_URL } from './rag-search.service';

@NgModule({
  declarations: [RagSearchComponent],
  imports: [CommonModule, FormsModule, HttpClientModule],
  exports: [RagSearchComponent],
})
export class RagSearchModule {
  /**
   * Configure the module with the RAG service base URL.
   *
   * Usage:
   *   RagSearchModule.forRoot('http://localhost:8000')
   */
  static forRoot(baseUrl: string): ModuleWithProviders<RagSearchModule> {
    return {
      ngModule: RagSearchModule,
      providers: [
        RagSearchService,
        { provide: RAG_BASE_URL, useValue: baseUrl },
      ],
    };
  }
}
