# Research Swarm

A multi-agent research system powered by LangGraph, MCP, Langfuse, and OpenAI-compatible LLMs.

## Architecture

```text
User Query -> Planner -> Searcher -> Fact Checker -> Summarizer -> Judge
                                                            |
                                                    score < 80 ? -> Searcher
                                                    score >= 80 ? -> END
```

Agents:
- **Planner**: produces 3-7 research questions.
- **Searcher**: gathers evidence via MCP tools.
- **Fact Checker**: validates and filters evidence.
- **Summarizer**: writes a cited report from validated facts.
- **Judge**: scores the report (0-100) and decides if more research is needed.

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
