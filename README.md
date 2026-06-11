# Research Swarm

A multi-agent research system powered by LangGraph, MCP, Langfuse, and OpenAI-compatible LLMs.

## Architecture

```
User Query -> Planner -> Searcher -> Fact Checker -> Summarizer -> Judge
                                                            |
                                                    score < 80 ? -> Searcher (targeted)
                                                    score >= 80 ? -> END
```

Agents:
- **Planner**: produces 3-7 research questions. On iteration 0 creates a full plan; on subsequent iterations refines the plan based on missing topics from the Judge.
- **Searcher**: gathers evidence via MCP tools. On iteration 0 performs a full search across all research questions. On subsequent iterations performs **targeted searches** focused exclusively on missing topics identified by the Judge.
- **Fact Checker**: validates and filters evidence.
- **Summarizer**: writes a cited report from validated facts.
- **Judge**: scores the report (0-100), identifies strengths, weaknesses, missing topics, and provides detailed reasoning. Decides if more research is needed.

## Iterative Research Loop

The workflow supports a production-grade iterative research loop with five safety mechanisms:

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
| **E** | No new evidence found during latest search | `no_new_evidence` |

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
- `new_evidence_found` — whether the latest search found anything new

## Setup

```bash
cp .env.example .env
uv sync
```

Ensure `.env` contains your `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`,
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`,
`MCP_HOST`, and `MCP_PORT`.

## Commands

Run the workflow:

```bash
uv run python -m research_swarm.main
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
