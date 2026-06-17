# Research Swarm

A multi-agent research system powered by LangGraph, MCP, Langfuse, OpenAI-compatible LLMs, Qdrant (vector DB), and Ollama (local embeddings).

Includes an **agentic news sender** that runs research on a theme and delivers reports via email (Resend), Telegram, and Discord.

## Architecture

```
                          ┌─────────────────────────────────┐
                          │  Qdrant + Ollama Vector Memory  │
                          │  (semantic dedup & retrieval)   │
                          └──────────┬──────────────────────┘
                                     │
  User Query -> Planner -> Searcher -> Fact Checker -> Summarizer -> Judge
                                                                  |
                                                    score < 80 ? -> Searcher (targeted)
                                                    score >= 80 ? -> END
```

Agents:
- **Planner**: produces 3-7 research questions. On iteration 0 creates a full plan; on subsequent iterations refines the plan based on missing topics from the Judge.
- **Searcher**: gathers evidence via 5 search providers with automatic fallback (Tavily → Brave → SerpAPI → SearXNG → DuckDuckGo). On iteration 0 performs a **full search** across all research questions. On subsequent iterations performs **targeted searches** focused exclusively on missing topics identified by the Judge.
- **Fact Checker**: validates evidence through a 3-layer pipeline: (1) semantic pre-filtering via Qdrant, (2) LLM validation, (3) commits unique facts to Qdrant. Skips LLM entirely for semantic duplicates — saving tokens and cost.
- **Summarizer**: pulls top-50 semantically relevant facts from Qdrant vector storage, compiles a cited markdown report with structured citations. Supports incremental rewrite across research iterations via `previous_report`.
- **Judge**: scores the report (0-100) with component breakdown (coverage /30, evidence /20, sources /20, depth /15, completeness /15). Uses temperature=0 for deterministic scoring with a concrete rubric. Post-parse hard guard rails cap scores when evidence or sources are insufficient. Identifies strengths, weaknesses, missing topics, and detailed reasoning. Decides if more research is needed.
- **Vector Memory Layer**: Qdrant + Ollama (`nomic-embed-text`) — handles semantic deduplication, long-term fact storage, and context retrieval across iterations.

## Infrastructure (Docker)

The project uses Docker Compose for local infrastructure:

```bash
docker compose up -d
```

Services:

| Service | Port | Purpose |
|---------|------|---------|
| **Qdrant** | `6333` (REST), `6334` (gRPC) | Vector database for semantic fact storage & dedup |
| **Ollama** | `11434` | Local LLM inference engine for embeddings |
| **ollama-provisioner** | — | Auto-downloads `nomic-embed-text` model on startup |

After `docker compose up -d`, Ollama provisions the embedding model. You can verify:

```bash
curl http://localhost:6333/healthz          # Qdrant health
ollama list                                  # should show nomic-embed-text
```

## Vector Memory Layer

Every validated fact is stored in Qdrant with a 768-dim embedding from Ollama's `nomic-embed-text` model. This enables three key capabilities:

### 1. Semantic Deduplication

Before the Fact Checker sends evidence to the LLM for validation, each item is checked against Qdrant using cosine similarity (threshold: 0.92). Already-seen facts are skipped — saving LLM tokens and cost.

### 2. Cross-Iteration Memory

Facts persist across research loop iterations. When the Searcher brings in new evidence, the Fact Checker only validates what's *actually new* — not what was already found in previous iterations.

### 3. Semantic Stagnation Detection

When Qdrant filters ALL incoming facts as duplicates, the router triggers a `no_new_evidence` stop condition — the research has exhausted all novel information and should terminate rather than burn tokens re-validating the same content.

### 4. Smart Summarization

The Summarizer doesn't see raw search results. Instead, it retrieves the top-50 most semantically relevant facts from Qdrant via `query_points()`. This means reports are built from de-duplicated, cross-iteration, quality-filtered context.

## News Sender

Send research digests to email, Telegram, or Discord — each independently gated by an env-var flag.

```bash
# Run with default theme file
news-sender

# Run with a custom theme file
news-sender path/to/my-theme.txt
```

The news sender reads a theme from `theme-of-the-news.txt`, runs the full research-swarm workflow, and delivers the final report to all enabled channels.

### Channel Configuration

| Channel | Env vars |
|---------|----------|
| **Email** (Resend) | `RESEND_API_KEY`, `RESEND_FROM`, `RESEND_TO`, `NEWS_SEND_EMAIL=true` |
| **Telegram** | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `NEWS_SEND_TELEGRAM=true` |
| **Discord** | `DISCORD_WEBHOOK_URL`, `NEWS_SEND_DISCORD=true` |

See `.env.example` for all available config vars.



## Iterative Research Loop

### Colored Logging

All CLI output uses ANSI-colored log lines for readability:
- 🟢 **INFO** in green with `key=value` highlighting
- 🟡 **WARN** in yellow
- 🔴 **ERROR** in red, **CRITICAL** on red background
- `separator()` helper for styled section dividers

### Quality Gate Design

The Judge evaluates each report iteration and either approves it (score ≥ 80) or identifies specific gaps. When gaps are found, the system conducts targeted follow-up research rather than re-running the entire search.

### Loop Prevention Strategy

Research stops when ANY of these conditions is met:

| Condition | Trigger | Stop Reason |
|-----------|---------|-------------|
| **A** | `score >= 80` | `score_threshold_met` |
| **B** | `iteration >= max_iterations` (default 3) | `max_iterations_reached` |
| **C** | Score improvement too small (`delta < 5`) | `insufficient_progress` |
| **D** | No additional missing topics reported by Judge | `no_missing_topics` |
| **E** | Search infrastructure failed (all providers) | `retrieval_failed` |
| **F** | No new evidence — Qdrant filtered everything as duplicate | `no_new_evidence` |

### Judge-Driven Targeted Research

When the Judge score is below threshold, instead of repeating the entire search:

1. The Judge identifies specific `missing_topics` (security risks, cost comparison, technical details, etc.)
2. The Planner refines the research focus based on these missing topics
3. The Searcher performs a targeted search ONLY for the missing information
4. New evidence is validated and incorporated into an updated report

### State Model

The `ResearchState` tracks the full lifecycle:

- `query` — the original research topic
- `plan` — current research plan (updated each iteration)
- `search_results` — all evidence gathered (accumulates across iterations)
- `validated_results` — validated facts
- `final_report` — the latest report
- `judge_score` — current quality score
- `iteration` — current iteration number
- `max_iterations` — safety limit (default 3)
- `previous_score` — score from previous iteration (for delta calculation)
- `score_delta` — score improvement since last iteration
- `missing_topics` — topics the Judge identified as gaps
- `no_progress` — whether research stalled
- `stop_reason` — why the loop ended
- `retrieval_failed` — whether all search providers failed
- `retrieval_failure_reason` — detailed failure explanation
- `search_providers_tried` — list of provider slugs that were attempted
- `search_provider_used` — which provider ultimately returned results
- `evidence_quality` — structured metadata per evidence item (content, source, provider, confidence, retrieved_at)
- `search_mode` — "full" (iteration 0) or "targeted" (subsequent iterations)
- `coverage_score`, `evidence_score`, `source_score`, `depth_score`, `completeness_score` — Judge component scores
- `new_evidence_count` — how many truly new facts were stored in Qdrant
- `new_evidence_found` — whether the latest search found anything new

## Setup

```bash
cp .env.example .env
uv sync
docker compose up -d      # Start Qdrant + Ollama
```

Ensure `.env` contains your `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`,
`TAVILY_API_KEY`, `BRAVE_API_KEY`, `SERPAPI_API_KEY`, `SEARXNG_BASE_URL`,
and any news sender vars you need (`RESEND_API_KEY`, `TELEGRAM_BOT_TOKEN`, `DISCORD_WEBHOOK_URL`).
See `.env.example` for the full list.

## Commands

Run the research workflow:

```bash
uv run research-swarm
```

Run the news sender:

```bash
# Edit theme-of-the-news.txt first, then:
uv run news-sender
```

Run tests:

```bash
uv run pytest
```

## MCP Tools

The MCP server exposes these tools over HTTP:

- `search_web(query: str)`
- `fetch_page(url: str)`
- `vector_search(query: str)`

By default, tools return mock data to keep the architecture runnable without external APIs.
