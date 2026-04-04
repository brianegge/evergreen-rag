/**
 * Evergreen RAG -- OPAC Semantic Search Module
 *
 * A standalone JavaScript module that adds semantic (natural language) search
 * to the Evergreen OPAC. No build step required -- include this file in your
 * OPAC template and call EvergreenRAG.init().
 *
 * Usage:
 *   <script src="/rag-static/opac/semantic-search.js"></script>
 *   <script>
 *     EvergreenRAG.init({ ragUrl: 'http://localhost:8000' });
 *   </script>
 *
 * See README.md in this directory for full integration instructions.
 */
(function (global) {
  'use strict';

  var DEFAULTS = {
    ragUrl: '/rag',
    limit: 10,
    minSimilarity: 0.0,
    mergeWithKeyword: false,
    semanticWeight: 0.5,
    bibRecordUrlPattern: '/eg/opac/record/{record_id}',
    searchFormSelector: '#search-wrapper',
    timeout: 10000
  };

  var config = {};
  var containerEl = null;

  // ------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------

  function init(userConfig) {
    config = assign({}, DEFAULTS, userConfig || {});

    // Normalize: strip trailing slash from ragUrl
    config.ragUrl = config.ragUrl.replace(/\/+$/, '');

    // Wait for DOM if needed
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', setup);
    } else {
      setup();
    }
  }

  function setup() {
    var searchForm = document.querySelector(config.searchFormSelector);
    if (!searchForm) {
      console.warn('[EvergreenRAG] Search form not found:', config.searchFormSelector);
      return;
    }

    injectToggle(searchForm);
    injectResultsContainer(searchForm);
    checkHealth();
  }

  // ------------------------------------------------------------------
  // UI Injection
  // ------------------------------------------------------------------

  function injectToggle(searchForm) {
    var toggle = document.createElement('div');
    toggle.className = 'rag-toggle';
    toggle.innerHTML =
      '<label class="rag-toggle-label">' +
        '<input type="checkbox" id="rag-semantic-toggle" class="rag-toggle-input" />' +
        '<span class="rag-toggle-text">Semantic Search</span>' +
        '<span class="rag-toggle-hint" title="Search using natural language instead of keywords">\u2139</span>' +
      '</label>';

    // Insert after the search form
    searchForm.parentNode.insertBefore(toggle, searchForm.nextSibling);

    var checkbox = document.getElementById('rag-semantic-toggle');
    checkbox.addEventListener('change', onToggleChange);

    // Intercept form submit when semantic mode is active
    searchForm.addEventListener('submit', onFormSubmit);
  }

  function injectResultsContainer(searchForm) {
    containerEl = document.createElement('div');
    containerEl.id = 'rag-results';
    containerEl.className = 'rag-results';
    containerEl.style.display = 'none';

    // Insert after the toggle
    var toggle = document.querySelector('.rag-toggle');
    if (toggle) {
      toggle.parentNode.insertBefore(containerEl, toggle.nextSibling);
    } else {
      searchForm.parentNode.appendChild(containerEl);
    }
  }

  // ------------------------------------------------------------------
  // Event Handlers
  // ------------------------------------------------------------------

  function onToggleChange() {
    if (!this.checked) {
      // Switched back to keyword mode -- hide semantic results
      if (containerEl) {
        containerEl.style.display = 'none';
      }
    }
  }

  function onFormSubmit(evt) {
    var checkbox = document.getElementById('rag-semantic-toggle');
    if (!checkbox || !checkbox.checked) {
      // Keyword mode -- let the form submit normally
      return;
    }

    evt.preventDefault();

    // Extract query from the search input
    var input = evt.target.querySelector('input[type="text"], input[name="query"]');
    if (!input) {
      input = evt.target.querySelector('input:not([type="hidden"]):not([type="submit"])');
    }
    if (!input || !input.value.trim()) {
      return;
    }

    performSearch(input.value.trim());
  }

  // ------------------------------------------------------------------
  // Search
  // ------------------------------------------------------------------

  function performSearch(queryText) {
    showLoading();

    var body = {
      query: queryText,
      limit: config.limit,
      min_similarity: config.minSimilarity
    };

    // Detect org_unit from OPAC context if available
    var orgUnit = detectOrgUnit();
    if (orgUnit !== null) {
      body.org_unit = orgUnit;
    }

    var endpoint = config.ragUrl + '/search';

    // If merge mode is on and we can get keyword results, use /search/merged
    if (config.mergeWithKeyword) {
      var keywordIds = getKeywordResultIds();
      if (keywordIds && keywordIds.length > 0) {
        endpoint = config.ragUrl + '/search/merged';
        body.keyword_result_ids = keywordIds;
        body.semantic_weight = config.semanticWeight;
      }
    }

    fetchJSON(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
      .then(function (data) {
        renderResults(data, queryText);
      })
      .catch(function (err) {
        renderError(err);
      });
  }

  // ------------------------------------------------------------------
  // Result Rendering
  // ------------------------------------------------------------------

  function showLoading() {
    if (!containerEl) return;
    containerEl.style.display = 'block';
    containerEl.innerHTML =
      '<div class="rag-loading">' +
        '<span class="rag-spinner"></span> Searching...' +
      '</div>';
  }

  function renderResults(data, queryText) {
    if (!containerEl) return;
    containerEl.style.display = 'block';

    if (!data.results || data.results.length === 0) {
      containerEl.innerHTML =
        '<div class="rag-no-results">' +
          '<p>No semantic results found for <strong>' + escapeHtml(queryText) + '</strong>.</p>' +
          '<p class="rag-hint">Try rephrasing your query in natural language, e.g., ' +
          '"books about learning to cook Italian food".</p>' +
        '</div>';
      return;
    }

    var html =
      '<div class="rag-results-header">' +
        '<h3>Semantic Search Results</h3>' +
        '<p class="rag-results-meta">' + data.total + ' result' +
          (data.total !== 1 ? 's' : '') + ' for <strong>' +
          escapeHtml(queryText) + '</strong>' +
          ' (model: ' + escapeHtml(data.model) + ')' +
        '</p>' +
      '</div>' +
      '<ul class="rag-results-list">';

    for (var i = 0; i < data.results.length; i++) {
      var r = data.results[i];
      var url = config.bibRecordUrlPattern.replace('{record_id}', r.record_id);
      var score = Math.round(r.similarity * 100);
      var chunkPreview = truncate(r.chunk_text, 200);
      var title = extractTitle(r.chunk_text);

      html +=
        '<li class="rag-result-item">' +
          '<div class="rag-result-score" title="Similarity: ' + r.similarity.toFixed(3) + '">' +
            score + '%' +
          '</div>' +
          '<div class="rag-result-body">' +
            '<a href="' + escapeAttr(url) + '" class="rag-result-title">' +
              escapeHtml(title) +
            '</a>' +
            '<p class="rag-result-chunk">' + escapeHtml(chunkPreview) + '</p>' +
          '</div>' +
        '</li>';
    }

    html += '</ul>';
    containerEl.innerHTML = html;
  }

  function renderError(err) {
    if (!containerEl) return;
    containerEl.style.display = 'block';
    containerEl.innerHTML =
      '<div class="rag-error">' +
        '<p><strong>Semantic search is currently unavailable.</strong></p>' +
        '<p>The RAG service could not be reached. Your keyword search results ' +
        'are still available above.</p>' +
        '<p class="rag-error-detail">' + escapeHtml(String(err)) + '</p>' +
      '</div>';
  }

  // ------------------------------------------------------------------
  // Health Check
  // ------------------------------------------------------------------

  function checkHealth() {
    fetchJSON(config.ragUrl + '/health', { method: 'GET' })
      .then(function (data) {
        if (data.status !== 'ok') {
          disableToggle('RAG service is degraded');
        }
      })
      .catch(function () {
        disableToggle('RAG service is unavailable');
      });
  }

  function disableToggle(reason) {
    var checkbox = document.getElementById('rag-semantic-toggle');
    if (checkbox) {
      checkbox.disabled = true;
      checkbox.title = reason;
    }
    var hint = document.querySelector('.rag-toggle-hint');
    if (hint) {
      hint.title = reason;
      hint.textContent = '\u26A0';
    }
  }

  // ------------------------------------------------------------------
  // Evergreen OPAC Helpers
  // ------------------------------------------------------------------

  function detectOrgUnit() {
    // Try common Evergreen OPAC patterns for the selected org unit
    // 1. URL parameter
    var match = window.location.search.match(/[?&]locg=(\d+)/);
    if (match) return parseInt(match[1], 10);

    // 2. Hidden form field
    var field = document.querySelector('input[name="locg"]');
    if (field && field.value) return parseInt(field.value, 10);

    // 3. Global JS variable (set by some OPAC templates)
    if (typeof global.eg_org_unit !== 'undefined') {
      return parseInt(global.eg_org_unit, 10);
    }

    return null;
  }

  function getKeywordResultIds() {
    // Extract record IDs from the current OPAC result page
    var ids = [];
    var links = document.querySelectorAll('a[href*="/record/"]');
    for (var i = 0; i < links.length; i++) {
      var href = links[i].getAttribute('href');
      var m = href.match(/\/record\/(\d+)/);
      if (m) {
        var id = parseInt(m[1], 10);
        if (ids.indexOf(id) === -1) {
          ids.push(id);
        }
      }
    }
    return ids;
  }

  // ------------------------------------------------------------------
  // Utilities
  // ------------------------------------------------------------------

  function fetchJSON(url, options) {
    return new Promise(function (resolve, reject) {
      var controller;
      var timeoutId;

      if (typeof AbortController !== 'undefined') {
        controller = new AbortController();
        options.signal = controller.signal;
        timeoutId = setTimeout(function () {
          controller.abort();
        }, config.timeout);
      }

      fetch(url, options)
        .then(function (response) {
          if (timeoutId) clearTimeout(timeoutId);
          if (!response.ok) {
            throw new Error('HTTP ' + response.status + ': ' + response.statusText);
          }
          return response.json();
        })
        .then(resolve)
        .catch(function (err) {
          if (timeoutId) clearTimeout(timeoutId);
          reject(err);
        });
    });
  }

  function extractTitle(chunkText) {
    // The first line of chunk_text is typically the title
    var firstLine = chunkText.split('\n')[0];
    return firstLine || 'Untitled';
  }

  function truncate(text, maxLen) {
    if (!text) return '';
    if (text.length <= maxLen) return text;
    return text.substring(0, maxLen).replace(/\s+\S*$/, '') + '...';
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  function escapeAttr(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function assign(target) {
    for (var i = 1; i < arguments.length; i++) {
      var source = arguments[i];
      if (source) {
        for (var key in source) {
          if (source.hasOwnProperty(key)) {
            target[key] = source[key];
          }
        }
      }
    }
    return target;
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  global.EvergreenRAG = {
    init: init,
    search: performSearch,
    checkHealth: checkHealth,
    version: '0.1.0'
  };

})(window);
