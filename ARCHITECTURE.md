# Architecture Report: Research Swarm — Production-Grade Overhaul

## 0. Vector Memory Layer: Qdrant + Ollama

### Overview

The system uses a local vector database (Qdrant) and local embedding inference (Ollama) to enable **semantic deduplication, cross-iteration memory, and smart context retrieval**. This eliminates redundant LLM calls and prevents research stagnation.

### Infrastructure

```yaml
# docker-compose.yml
qdrant:     image: qdrant/qdrant:latest      # port 6333
ollama:     image: ollama/ollama:latest      # port 11434
ollama-provisioner:                          # auto-downloads nomic-embed-text
```

### SwarmMemoryBank (`src/research_swarm/memory/vector_storage.py`)

| Method | Purpose |
|--------|---------|
| `upsert_facts(facts, iteration, task)` | Embed facts via Ollama → check for semantic duplicates (cosine ≥ 0.92) → store unique ones in Qdrant. Returns count of *newly* stored facts. |
| `retrieve_context(query, limit=25)` | Embed query → semantic search via `query_points()` → return top-N fact payloads. Used by Summarizer to build reports from clean, de-duplicated context. |
| `_is_semantic_duplicate(vector, threshold=0.92)` | Cosine similarity check against existing Qdrant collection. Used by Fact Checker to pre-filter evidence before LLM validation. |

**Embedding model**: `nomic-embed-text` (768-dim vectors, Cosine distance)

### 3-Layer Fact Checking Pipeline

```
Raw evidence batch
    ↓
Layer 1: Semantic pre-filter (Qdrant cosine ≥ 0.92 check)
    → duplicates skipped, no LLM cost incurred
    ↓
Layer 2: LLM validation (only unique items reach the LLM)
    → validated_facts + rejected_facts
    ↓
Layer 3: Commit to Qdrant (embed + upsert unique facts)
    → new_evidence_count = how many were actually new
    → new_evidence_found = new_evidence_count > 0
```

If Layer 1 filters ALL items as duplicates, the Fact Checker skips the LLM entirely and sets `new_evidence_found = False`. This feeds into the routing layer's semantic stagnation detection.

### Semantic Stagnation Stop Condition

When the Fact Checker reports `new_evidence_found = False` (meaning Qdrant found every incoming item duplicated), the router triggers a `no_new_evidence` stop condition. This prevents the system from burning tokens re-validating facts it already knows — a smarter termination than simple iteration counting.

### Summarizer Integration

The Summarizer no longer reads raw `state.validated_results` or `state.search_results`. Instead, it calls `memory.retrieve_context(query=state.query, limit=50)` to pull the top-50 semantically relevant facts from Qdrant. This means reports are built from:
- De-duplicated context (no repeated claims)
- Cross-iteration knowledge (facts from ALL iterations, not just the latest)
- Quality-filtered content (only facts that passed the 3-layer pipeline)

---

## 1. Root-Cause Analysis: Why the Workflow Collapses

### Observed Failure Mode

```
Planner → Searcher → FactChecker → Summarizer → Judge → Router → END
             ↓
   DuckDuckGo HTML scraping returns []
             ↓
      zero evidence → zero validated facts → score = 0 → terminate
```

### Root Causes (3 Interlocking Weaknesses)

| Weakness | Mechanism | Impact |
|----------|-----------|--------|
| **Single-provider dependency** | `mcp_client.py` was the only retrieval path — hardcoded DDG HTML scraping. No fallback. | Any DDG failure (HTML layout change, rate limit, CDN block) causes total evidence starvation. |
| **No failure differentiation in routing** | `no_new_evidence` treated "search produced zero results" identically to "search infrastructure failed." | The router couldn't tell whether retrying with a different provider would help. It just terminated. |
| **No diagnostics** | When DDG returned empty results, there was no snapshot of what the raw HTTP response looked like. | Operators couldn't debug HTML parsing failures without reproducing the exact query. |

### Why DDG HTML Scraping Is Inherently Fragile

The core issue is **structural coupling** between retrieval logic and HTML parsing. DDG's `html.duckduckgo.com` endpoint:

- Uses server-side rendered HTML with CSS class selectors (`div.result__body`, `a.result__a`, `a.result__snippet`)
- These selectors are **stable for years** but change without notice
- The endpoint may return `200 OK` with an empty body or a CAPTCHA page when rate-limited
- `BeautifulSoup` parses successfully but finds zero result containers → logs "No search result containers found" → returns `[]`

This would be acceptable **if** there were a fallback. But there wasn't.

---

## 2. Retrieval Layer Redesign

### New Architecture: Provider + Orchestrator + Health + Diagnostics

```
src/research_swarm/search/
├── __init__.py              # Public API: SearchOrchestrator, get_orchestrator(), etc.
├── base.py                  # BaseSearchProvider (ABC), SearchResultItem, SearchResponse
├── orchestrator.py          # Priority fallback chain
├── health.py                # SearchHealthMonitor — per-provider metrics
├── diagnostics.py           # save_failure_diagnostic() — JSON snapshots
└── providers/
    ├── __init__.py
    ├── tavily.py            # Tavily Search API (AI-optimized, needs TAVILY_API_KEY)
    ├── brave.py             # Brave Search API (needs BRAVE_API_KEY)
    ├── serpapi.py           # SerpAPI Google Search (needs SERPAPI_API_KEY)
    ├── searxng.py           # SearXNG self-hosted metasearch (needs SEARXNG_BASE_URL)
    └── duckduckgo.py        # DDG HTML scraping (free, always available)
```

### Provider Abstraction

```python
class BaseSearchProvider(ABC):
    @property
    def slug(self) -> str: ...         # "tavily", "brave", etc.
    @property
    def is_available(self) -> bool: ... # credentials present?
    @property
    def priority(self) -> int: ...     # 0 = highest, 4 = lowest (last resort)
    async def search(self, query, max_results) -> SearchResponse: ...
```

Every provider returns a `SearchResponse` containing `list[SearchResultItem]`, where each item has `{title, url, snippet, provider, confidence}`. This normalised shape means the caller (Searcher agent) never knows which provider produced the results.

### Fallback Strategy

```
Priority chain: Tavily → Brave → SerpAPI → SearXNG → DuckDuckGo

For each query:
  1. Try Tavily (highest priority, best results)
     → success? Stop. Return results.
     → failed? Continue to next.
  2. Try Brave
     → success? Stop.
     → failed? Continue to next.
  ...
  5. Try DuckDuckGo (always available, free)
     → success? Return results.
     → failed? ALL FAILED — report retrieval_failed=True.
```

Unavailable providers (missing API keys) are silently skipped.

---

## 3. Resilience Improvements

### 3a. Retrieval Health Monitoring (`SearchHealthMonitor`)

In-memory metrics per provider:

| Metric | Description |
|--------|-------------|
| `total_attempts` | How many times this provider was tried |
| `total_successes / total_failures` | Success/failure counts |
| `success_rate` | Percentage of attempts that produced results |
| `avg_results` | Average number of results per successful call |
| `avg_latency_s` | Average response time |
| `total_timeouts` | Number of timeout-based failures |
| `last_error` | Most recent error message |

Available via `orchestrator.health.snapshot()` and logged via `orchestrator.health.log_summary()`.

### 3b. Retrieval Failure Diagnostics

When ALL providers fail for a query, a JSON diagnostic file is saved to `logs/retrieval_failures/` containing:
- Timestamp
- Query string
- Per-provider attempt details (success/failure, error messages, HTTP status codes)
- Optional HTML snapshot (for HTML-scraping providers)

This enables post-mortem debugging without reproducing the exact query.

### 3c. Retrieval Failure Recovery Flow

```
Retrieval FAILED (all providers exhausted)
    ↓
state.retrieval_failed = True
state.retrieval_failure_reason = "All providers failed across N queries..."
    ↓
Judge receives retrieval_failed context
    ↓
Judge produces explanation: "evidence unavailable — infrastructure failure"
    ↓
Router sees retrieval_failed → stops with "retrieval_failed" reason
    ↓
Workflow terminates gracefully with full diagnostic context
```

### 3d. Evidence Quality Tracking

Each evidence item now carries structured metadata:

```json
{
    "content": "AI assistants boost developer productivity by 25%",
    "source": "https://example.com",
    "provider": "tavily",
    "search_mode": "full",
    "retrieved_at": 1718123456789,
    "confidence": 0.6
}
```

This feeds into the Judge's evidence quality scoring (0-20 component).

---

## 4. Searcher Agent Redesign

### Dual Search Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| **FULL** | `state.search_mode == "full"` (iteration 0) | Searches across ALL research questions from the plan. |
| **TARGETED** | `state.search_mode == "targeted"` (iteration ≥ 1) | Searches ONLY for Judge-identified missing topics. Generates focused queries like `"{query} {missing_topic}"`. |

The `search_mode` is set by the routing node on iteration transitions. This eliminates wasteful full re-searches and instead conducts surgical investigations of specific gaps.

---

## 5. Router Improvements: Differentiated Stop Conditions

### New Priority Order (6 conditions)

| Priority | Condition | Stop Reason | Meaning |
|----------|-----------|-------------|---------|
| A | `score >= 80` | `score_threshold_met` | Report is good enough |
| B | `iteration >= max` | `max_iterations_reached` | Safety limit reached |
| C | `delta < 5` | `insufficient_progress` | Iterations aren't helping |
| D | `missing_topics == []` | `no_missing_topics` | Nothing left to research |
| **E (NEW)** | `retrieval_failed` | **`retrieval_failed`** | Infrastructure failure — NOT "nothing found" |
| F | `!new_evidence_found` | `no_new_evidence` | Search succeeded but found nothing new |

### Critical Distinction: Condition E vs F

- **E (`retrieval_failed`)**: Search *infrastructure* failed. All providers timed out, returned errors, or were unavailable. This is an operational problem, not a content problem.
- **F (`no_new_evidence`)**: Search *succeeded* but found nothing new (deduplicated). This is a content exhaustion problem.

The old code conflated these into a single `no_new_evidence` condition, causing premature termination when DDG was down. The new code explicitly separates them, with `retrieval_failed` (E) taking priority over `no_new_evidence` (F).

---

## 6. Scalability Improvements

| Dimension | Before | After |
|-----------|--------|-------|
| **Providers** | 1 (DDG, hardcoded) | 5 (pluggable, priority-ordered) |
| **Adding a provider** | Rewrite mcp_client.py | Implement `BaseSearchProvider`, drop into `search/providers/` |
| **Fallback behavior** | None — single point of failure | Automatic cascading fallback |
| **Observability** | None | Per-provider health metrics + failure diagnostics |
| **Testing** | Mock hardcoded functions | Mock orchestrator or individual providers |
| **Evidence quality** | None | Structured metadata per item |
| **Vector memory** | None | Qdrant + Ollama — semantic dedup, cross-iteration memory, smart retrieval |
| **LLM token waste** | All evidence sent to LLM every time | Semantic pre-filter skips duplicates before LLM |
| **Report context** | Raw search results | Top-N semantically relevant de-duplicated facts from Qdrant |

---

## 7. Production-Readiness Assessment

### Ready

- ✅ **Multiple providers with automatic failover**
- ✅ **Health monitoring** — metrics for every provider call
- ✅ **Failure diagnostics** — JSON snapshots for debugging
- ✅ **Searcher dual-mode** — FULL/TARGETED research
- ✅ **Differentiated stop conditions** — `retrieval_failed` vs `no_new_evidence`
- ✅ **Evidence quality tracking** — structured metadata with timestamps
- ✅ **Vector memory** — Qdrant + Ollama semantic dedup, cross-iteration memory, smart retrieval
- ✅ **Token efficiency** — semantic pre-filter skips duplicate evidence before LLM validation
- ✅ **Comprehensive test suite** — 242 tests, 0 failures
- ✅ **Async-first, fully typed, no TODOs or placeholders**
- ✅ **Graceful degradation** — providers without API keys are silently skipped
- ✅ **Infrastructure as code** — `docker-compose.yml` for one-command local setup

### Recommendations for Further Hardening

1. **Circuit breaker**: If a provider fails N times consecutively, temporarily remove it from the chain for a cooldown period.
2. **Response caching**: Cache search results per query to avoid redundant API calls across iterations.
3. **Provider SLA tiers**: Track which providers produce the highest-quality evidence (by downstream Judge scores) and weight priority dynamically.
4. **Rate limit awareness**: Track `Retry-After` headers from providers that return 429 and respect backoff windows.
5. **Distributed health**: If running multiple instances, push health metrics to a shared store (Redis, statsd) for cluster-wide provider health visibility.
6. **Ollama GPU acceleration**: If running on a machine with a GPU, pass `--gpus all` to the Ollama container for faster embedding inference.
7. **Qdrant persistence**: The `qdrant_storage` volume ensures facts survive container restarts. Consider backups for long-running research projects.
8. **Multi-model embeddings**: Add support for alternative Ollama embedding models (e.g., `mxbai-embed-large` for 1024-dim vectors) via configuration.

---

## Summary of Changes

| File | Change |
|------|--------|
| `search/base.py` | **NEW** — `BaseSearchProvider`, `SearchResultItem`, `SearchResponse` |
| `search/orchestrator.py` | **NEW** — `SearchOrchestrator` with priority fallback chain |
| `search/health.py` | **NEW** — `SearchHealthMonitor` per-provider metrics |
| `search/diagnostics.py` | **NEW** — `save_failure_diagnostic()` JSON snapshots |
| `search/providers/*.py` | **NEW** — Tavily, Brave, SerpAPI, SearXNG, DuckDuckGo providers |
| `agents/searcher.py` | **REWRITTEN** — orchestrator-based, FULL/TARGETED modes, evidence quality tracking |
| `agents/judge.py` | **UPDATED** — receives retrieval context, removed redundant JSON field |
| `graph/routing.py` | **UPDATED** — added `retrieval_failed` stop condition (E), `search_mode` transitions |
| `graph/state.py` | **UPDATED** — 6 new fields: `retrieval_failed`, `search_providers_tried`, `search_provider_used`, `evidence_quality`, `search_mode`, `retrieval_failure_reason` |
| `config/settings.py` | **UPDATED** — search provider API key env vars |
| `main.py` | **UPDATED** — health summary logging, retrieval diagnostics in output |
| `mcp_client.py` | **DEPRECATED** — thin wrapper around `DuckDuckGoProvider` |
| `tests/test_agents.py` | **UPDATED** — searcher tests use orchestrator mocks |
| `tests/test_routing.py` | **UPDATED** — `retrieval_failed` condition + `search_mode` transition tests |
| `memory/vector_storage.py` | **NEW** — `SwarmMemoryBank` with Qdrant + Ollama for semantic dedup & retrieval |
| `docker-compose.yml` | **NEW** — Qdrant, Ollama, ollama-provisioner service definitions |
| `tests/test_search.py` | **NEW** — 10 integration tests for orchestrator, fallback, health monitor |

---

## 8. Quality Improvement Overhaul (experiment-80)

All six items from the TODO.md quality plan targeting 80+ judge scores.

### 8a. Judge — Deterministic Scoring

- **temperature=0** for zero-variance scores
- **Concrete scoring rubric** with per-category examples (e.g., coverage 25-30 means all research questions answered with specific data)
- **Bias fix**: `missing_topics` based strictly on `research_questions` from the plan, not hardcoded "frontier AI" framing
- **Hard guard rails** (post-parse, deterministic):
  - Fewer than 10 total evidence items → score capped at 70
  - Fewer than 5 unique sources → score capped at 75
  - Guard rail messages prefixed with `[SYSTEM]` and filtered by Searcher

### 8b. Searcher — Adaptive Fan-Out

- **Adaptive `max_results`**: 12-15 on iteration 0 (broad research), 7 on iterations ≥1 (targeted)
- **Topic slice** `[:3]` → `[:5]` for broader missing-topic coverage
- **Batch formatting**: raw search entries sent to LLM formatter in groups of 12 to avoid token blow-up
- **`[SYSTEM]` filtering**: operational guard-rail messages excluded from search topics

### 8c. Fact Checker — Gradual Threshold Decay

- Iteration 0 → threshold **0.94** (strict)
- Iteration 1 → threshold **0.90** (moderate)
- Iteration 2+ → threshold **0.82** (relaxed)
- Dynamic threshold passed through `memory.upsert_facts(..., threshold=...)`

### 8d. Summarizer — Larger Context + Citations

- Qdrant retrieval limit **25 → 50** facts
- **Structured citations** required in LLM output (`citations` array with fact-to-source mappings)
- **Fallback merge**: if parsed JSON is empty, secondary LLM call merges previous report with new facts at low temperature

### 8e. Incremental Rewrite Loop

- `ResearchState.previous_report` field preserves the last report before each Summarizer run
- Rewrite prompt: "Preserve strong sections from previous report. Address weaknesses and missing_topics using new facts. Do NOT restart from scratch."
- `merge_state()` extracted to `utils.py` and shared between `main.py` and `news_sender.py`

### 8f. Colored Logging

- ANSI-colored `_ColoredFormatter` in `logging_config.py`:
  - Green INFO with `key=value` highlighting
  - Yellow WARN, Red ERROR, Red-bg CRITICAL
  - Cyan logger names, grey timestamps
- `separator()` helper for styled section dividers
- Logger name auto-truncation (e.g., `research_swarm.agents.searcher` → `r.agents.searcher`)

---

## 9. News Sender — Multi-Channel Report Delivery

### Architecture

```
theme-of-the-news.txt  →  research-swarm workflow  →  NewsChannel ABC
                                                          ├── EmailChannel (Resend API)
                                                          ├── TelegramChannel (Bot API)
                                                          └── DiscordChannel (Webhook)
```

### Channel Abstraction

```python
class NewsChannel(ABC):
    @property
    def name(self) -> str: ...
    async def send(self, subject: str, report: str) -> bool: ...
```

Each channel is independently gated by a `NEWS_SEND_*` env var. All three can be enabled simultaneously.

### Email via Resend

- REST API: `POST https://api.resend.com/emails`
- Auth: `Bearer re_...`
- Env vars: `RESEND_API_KEY`, `RESEND_FROM`, `RESEND_TO`, `NEWS_SEND_EMAIL`

### Telegram

- Bot API: `POST https://api.telegram.org/bot{token}/sendMessage`
- Plain text mode (no parse_mode to avoid markdown rendering issues)
- Auto-truncates messages exceeding 4000 chars
- Env vars: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `NEWS_SEND_TELEGRAM`

### Discord

- Webhook: `POST {webhook_url}` with embed object
- Embed title = subject, description = report, color = blue
- Auto-truncates descriptions exceeding 4000 chars
- Env vars: `DISCORD_WEBHOOK_URL`, `NEWS_SEND_DISCORD`

### CLI

```bash
news-sender                          # uses theme-of-the-news.txt
news-sender path/to/custom-theme.txt # custom theme file
```

### Configuration vars

| File | Purpose |
|------|---------|
| `.env.example` | All 24 env vars documented across 7 categories |
| `pyproject.toml` | `news-sender` and `research-swarm` CLI entry points |
| `theme-of-the-news.txt` | Default news theme (edit this to change topic) |
