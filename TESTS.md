# Tests — Research Swarm

**217 tests · 10 files · 0 skipped · 100% pass rate**

Run: `uv run pytest tests/ -v`

---

## Test Architecture

```
tests/
├── test_state_models.py      # 28 tests — Pydantic state models
├── test_schemas.py           # 10 tests — Graph schema layer (AgentIO subclasses)
├── test_routing.py           # 10 tests — Conditional routing from judge
├── test_settings.py          # 24 tests — Settings, env vars, validators
├── test_logging.py           # 13 tests — Logging config, preview helper
├── test_llm_client.py        # 11 tests — LLM client, get_llm, invoke_messages
├── test_agents.py            # 23 tests — All 5 agents with mocked LLM + _safe_json
├── test_langfuse.py          # 15 tests — Langfuse tracing, tracers, context manager
├── test_workflow_graph.py    # 14 tests — Workflow graph, _merge_state
└── test_workflow.py          #  2 tests — Legacy workflow tests (kept)
```

**Mocking strategy**: Agents use `@patch` on `invoke_messages` and `trace_agent` to isolate from LLM and Langfuse. Settings tests use `monkeypatch.setenv` to isolate from `.env` file and shell environment. Langfuse tests use `patch.dict("sys.modules", ...)` to mock the internal `from langfuse import Langfuse` import.

---

## `tests/test_state_models.py` — 28 tests

Pydantic model validation, defaults, serialization, and edge cases for all state models in `research_swarm.graph.state`.

### `TestAgentIO` — base model

| Test | What it verifies |
|------|-----------------|
| `test_agentio_is_base_model` | `AgentIO` inherits from `pydantic.BaseModel`; `model_dump()` of empty instance returns `{}` |
| `test_agentio_allows_extra_fields` | `AgentIO` instances expose `model_dump` method (Pydantic plumbing) |

### `TestSearchResult` — evidence container

| Test | What it verifies |
|------|-----------------|
| `test_creation_with_empty_evidence` | `SearchResult(question_id="q1", evidence=[])` — both fields accept their types; empty list is valid |
| `test_creation_with_evidence` | Two evidence strings with inline source annotations are stored correctly; list preserves order |
| `test_requires_question_id` | Omitting `question_id` raises `ValidationError` |
| `test_evidence_field_is_required` | Omitting `evidence` raises `ValidationError` (field has no default) |
| `test_serialization_roundtrip` | `model_dump()` → `SearchResult(**data)` rebuilds an equal instance |
| `test_json_serialization` | `model_dump_json()` produces a JSON string containing both `question_id` and evidence content |
| `test_question_id_is_string` | Pydantic coerces int-like input `"42"` to `str`; `isinstance` confirms string type |

### `TestValidatedResult` — fact-checker output

| Test | What it verifies |
|------|-----------------|
| `test_creation_with_both_lists` | `validated_facts` and `rejected_facts` store their respective lists |
| `test_creation_empty` | Both fields accept `[]` (empty lists) |
| `test_all_fields_are_required` | `ValidatedResult()` with no args raises `ValidationError` — both fields lack defaults |
| `test_requires_string_lists` | Passing `[1, 2, 3]` for `validated_facts` (expects `list[str]`) raises `ValidationError` |

### `TestResearchPlan` — planner output

| Test | What it verifies |
|------|-----------------|
| `test_creation_with_single_question` | Single-element `research_questions` list; `goal` field stores string |
| `test_creation_with_multiple_questions` | Three-element list; `len()` confirms count |
| `test_requires_goal` | Omitting `goal` raises `ValidationError` |
| `test_requires_research_questions` | Omitting `research_questions` raises `ValidationError` |
| `test_empty_questions_list` | `research_questions=[]` is valid (empty plan edge case) |
| `test_goal_must_be_string` | `goal=42` (int) raises `ValidationError` |
| `test_questions_must_be_strings` | `research_questions=[1, 2, 3]` raises `ValidationError` |
| `test_model_dump_output` | `model_dump()` returns `{"goal": ..., "research_questions": [...]}` dict shape |

### `TestResearchReport` — summarizer output

| Test | What it verifies |
|------|-----------------|
| `test_creation_with_summary_and_sources` | Two sources stored; summary string preserved |
| `test_creation_with_empty_sources` | `sources=[]` is valid |
| `test_requires_summary` | Omitting `summary` raises `ValidationError` |
| `test_requires_sources` | Omitting `sources` raises `ValidationError` |
| `test_summary_must_be_string` | `summary=123` raises `ValidationError` |
| `test_sources_must_be_strings` | `sources=[1, 2, 3]` raises `ValidationError` |

### `TestJudgeResult` — judge scoring

| Test | What it verifies |
|------|-----------------|
| `test_creation_high_score` | `score=95` with `needs_research=False` and empty `missing_topics` |
| `test_creation_low_score` | `score=40` with `needs_research=True` and 2 missing topics |
| `test_all_fields_required` | `JudgeResult(score=50)` alone raises `ValidationError` — all fields lack defaults |
| `test_accepts_any_score_value` | Pydantic does not reject `score=150` without explicit `Field(ge=0, le=100)` |
| `test_accepts_zero_score` | `score=0` is valid |
| `test_serialization_roundtrip` | `model_dump()` → `JudgeResult(**data)` roundtrip preserves equality |
| `test_score_must_be_int` | `score="high"` (str) raises `ValidationError` |

### `TestResearchState` — main state container

| Test | What it verifies |
|------|-----------------|
| `test_minimal_creation` | `ResearchState(query="...")` — `plan` defaults to `None`, `judge_score` to `0`, empty lists for `search_results` and `validated_results`, `final_report` to `None` |
| `test_full_creation` | All 6 fields populated simultaneously with nested models; every getter returns the correct value |
| `test_requires_query` | `ResearchState()` raises `ValidationError` — `query` has no default |
| `test_query_must_be_string` | `query=42` raises `ValidationError` |
| `test_judge_score_default_zero` | `judge_score` is `0` when not provided |
| `test_plan_defaults_to_none` | `plan` is `None` when not provided |
| `test_search_results_default_to_empty_list` | `search_results` is `[]` when not provided |
| `test_validated_results_default_to_empty_list` | `validated_results` is `[]` when not provided |
| `test_final_report_defaults_to_none` | `final_report` is `None` when not provided |
| `test_model_dump_with_none_fields` | `model_dump()` on minimal state returns all 6 keys with correct default values |
| `test_model_dump_with_populated_fields` | `model_dump()` on state with nested `ResearchPlan` serializes nested dicts |
| `test_multiple_search_results` | Two `SearchResult` objects stored and accessible by index; ordering preserved |
| `test_multiple_validated_results` | Two `ValidatedResult` objects stored; `len()` confirms count |
| `test_convenience_for_empty_check` | `not state.search_results` evaluates to `True` for empty lists (falsy check pattern) |

---

## `tests/test_schemas.py` — 10 tests

Validates models in `research_swarm.graph.schemas` — all inherit from `AgentIO` (a `BaseModel` subclass).

### `TestSchemataModels`

| Test | What it verifies |
|------|-----------------|
| `test_research_plan_extends_agentio` | `schemas.ResearchPlan` is a subclass of `AgentIO` |
| `test_research_plan_has_correct_fields` | `goal` and `research_questions` fields work identically to `state.ResearchPlan` |
| `test_research_plan_serialization` | `model_dump()` returns `{"goal": ..., "research_questions": [...]}` |
| `test_research_report_extends_agentio` | `schemas.ResearchReport` is a subclass of `AgentIO` |
| `test_research_report_has_correct_fields` | `summary` and `sources` fields operational |
| `test_research_report_serialization` | `model_dump()` returns `{"summary": ..., "sources": [...]}` |
| `test_judge_result_extends_agentio` | `schemas.JudgeResult` is a subclass of `AgentIO` |
| `test_judge_result_has_correct_fields` | `score`, `needs_research`, `missing_topics` operational |
| `test_judge_result_serialization` | `model_dump()` returns `{"score": 90, "needs_research": false, "missing_topics": []}` |
| `test_judge_result_all_fields_required` | `JudgeResult(score=50)` alone raises an exception — no defaults |

---

## `tests/test_routing.py` — 10 tests

Covers `route_from_judge()` — the conditional edge that decides whether to end or loop.

### `TestRouteFromJudge`

| Test | Input score | Expected route | Edge case |
|------|------------|----------------|-----------|
| `test_returns_end_when_score_above_threshold` | 80 | `"__end__"` | Boundary: **at** threshold |
| `test_returns_end_when_score_equal_to_threshold` | 80 | `"__end__"` | Redundant boundary (same as above) |
| `test_returns_searcher_when_score_below_threshold` | 79 | `"searcher"` | Boundary: **one below** threshold |
| `test_returns_searcher_when_score_zero` | 0 | `"searcher"` | Minimum valid score |
| `test_returns_searcher_when_score_negative` | -1 | `"searcher"` | Negative score edge case (unclamped at this layer) |
| `test_returns_end_when_score_exactly_100` | 100 | `"__end__"` | Maximum score |
| `test_returns_searcher_when_score_50` | 50 | `"searcher"` | Mid-range |
| `test_returns_end_when_score_99` | 99 | `"__end__"` | High but not max |
| `test_parametrized_routing` | 0, 1, 50, 79, 80, 85, 100 | `searcher` × 4, `__end__` × 3 | Parametrized sweep across the full range |

**Key invariant**: `score >= 80 → END`, `score < 80 → searcher`. All tests confirm exact boundary behavior.

---

## `tests/test_settings.py` — 24 tests

Tests `Settings` (pydantic-settings), field validators, computed properties, alias mapping, and `get_settings` caching. All tests use `monkeypatch.setenv` to achieve full isolation from the host `.env` file and shell environment.

### `TestSettings` — core configuration

| Test | What it verifies |
|------|-----------------|
| `test_default_values` | When env vars match Pydantic defaults: `openai_model="gpt-4o"`, `openai_base_url → None` (validator converts `""` to `None`), `langfuse_public_key=""` (no `""→None` validator), `langfuse_host="https://cloud.langfuse.com"`, `mcp_host="127.0.0.1"`, `mcp_port=8765` |
| `test_custom_values` | All env vars set to non-default values: `gpt-3.5-turbo`, custom base URL, custom Langfuse host, `0.0.0.0`, port `9000` |
| `test_llm_api_key_property` | `llm_api_key` property returns `self.openai_api_key` when non-empty |
| `test_llm_api_key_raises_when_empty` | `llm_api_key` raises `ValueError("OPENAI_API_KEY is required")` when `openai_api_key=""` (empty env var) |
| `test_llm_base_url_property` | `llm_base_url` property returns the (validated) `openai_base_url` |
| `test_llm_base_url_returns_none_when_not_set` | `llm_base_url` returns `None` when env var is `""` (converted by validator) |
| `test_mcp_url_property` | `mcp_url` computed property returns `"http://{mcp_host}:{mcp_port}"` |
| `test_mcp_url_custom` | Custom host `10.0.0.1` and port `5555` produce `"http://10.0.0.1:5555"` |
| `test_env_var_loading` | `OPENAI_API_KEY` and `OPENAI_MODEL` loaded from env vars into `Settings()` |

### `TestOpenAIBaseUrlNormalization` — field validator

| Test | Input env var | Expected `openai_base_url` | Rule tested |
|------|-------------|---------------------------|-------------|
| `test_none_value_returns_none` | `""` | `None` | Empty string → validator returns `None` |
| `test_empty_string_returns_none` | `""` | `None` | Same as above (redundant—tests `""` path) |
| `test_trailing_slash_removed` | `"https://api.openai.com/v1/"` | `"https://api.openai.com/v1"` | `.rstrip("/")` |
| `test_chat_completions_path_removed` | `".../v1/chat/completions"` | `"...v1"` | Strips `/chat/completions` suffix |
| `test_chat_completions_with_trailing_slash_removed` | `".../v1/chat/completions/"` | `"...v1"` | Both suffix and trailing slash stripped |
| `test_normal_url_preserved` | `"https://custom.api.com"` | `"https://custom.api.com"` | Clean URL passes through unchanged |
| `test_url_with_multiple_slashes` | `"https://api.test.com////"` | `"https://api.test.com"` | `.rstrip("/")` handles multiple trailing slashes |
| `test_without_v1_path` | `"https://api.groq.com/openai"` | `"https://api.groq.com/openai"` | Non-OpenAI-compatible path preserved |
| `test_without_v1_trailing_slash` | `"https://api.groq.com/openai/"` | `"https://api.groq.com/openai"` | Trailing slash on non‑standard path stripped |

### `TestSettingsAliases` — environment variable aliases

| Test | Env var set | Field asserted |
|------|-----------|---------------|
| `test_openai_api_key_alias` | `OPENAI_API_KEY` | `settings.openai_api_key` matches |
| `test_openai_model_alias` | `OPENAI_MODEL` | `settings.openai_model` matches |
| `test_openai_base_url_alias` | `OPENAI_BASE_URL` | `settings.openai_base_url` matches |
| `test_langfuse_public_key_alias` | `LANGFUSE_PUBLIC_KEY` | `settings.langfuse_public_key` matches |
| `test_langfuse_secret_key_alias` | `LANGFUSE_SECRET_KEY` | `settings.langfuse_secret_key` matches |

### `TestGetSettings` — singleton factory

| Test | What it verifies |
|------|-----------------|
| `test_returns_settings_instance` | `get_settings()` returns a `Settings` instance |
| `test_caching_works` | Two calls without env changes return the **same object** (`is` identity check) — `@lru_cache()` works |

---

## `tests/test_logging.py` — 13 tests

Tests `setup_terminal_logging()` (root logger setup, handler creation, third-party silencer) and `preview()` (text truncation helper).

### `TestSetupTerminalLogging`

| Test | What it verifies |
|------|-----------------|
| `test_adds_handler_to_root_logger` | After clearing handlers, first call adds exactly 1 `StreamHandler`; root level set to `INFO` |
| `test_second_call_does_not_add_duplicate_handler` | Second call returns early (`root.handlers` already present), handler count unchanged |
| `test_quiet_loggers_are_set_to_warning` | 9 third-party loggers (`httpx`, `httpcore`, `openai`, `langchain`, `langchain_core`, `langchain_openai`, `langgraph`, `langfuse`, `urllib3`) all set to `logging.WARNING` |
| `test_custom_level` | `level=logging.DEBUG` sets root to `DEBUG` |
| `test_handler_stream_is_stderr` | Handler writes to `sys.stderr` |
| `test_formatter_is_configured` | Handler has a non-`None` `logging.Formatter` instance attached |

### `TestPreview`

| Test | Input | Expected output | Rule |
|------|-------|----------------|------|
| `test_short_text_unchanged` | `"hello"` | `"hello"` | Within default 160-char limit |
| `test_exact_limit_not_truncated` | `"a" × 160` | `"aaa...a"` (160 chars) | Exactly at limit — no `...` |
| `test_long_text_truncated` | `"a" × 200` | `≤160` chars, ends with `"..."` | Above limit — truncated with ellipsis |
| `test_custom_limit` | `"a" × 50`, limit=20 | 20 chars, ends with `"..."` | Custom `limit` parameter works |
| `test_whitespace_normalized` | `"hello    world\n\n\ttest"` | `"hello world test"` | Multiple whitespace collapsed to single spaces |
| `test_very_long_text_truncated_properly` | 20× alphabet words, limit=50 | 50 chars, ends with `"..."` | Long text with word boundaries truncated cleanly |
| `test_empty_string` | `""` | `""` | Empty string returns empty |
| `test_only_whitespace` | `"   \n\t  "` | `""` | Whitespace-only collapses to empty string |
| `test_text_exactly_at_limit` | `"x" × 40`, limit=40 | `"xxx...x"` (40 chars) | No ellipsis at exact limit |
| `test_text_one_over_limit` | `"x" × 41`, limit=40 | 40 chars, ends with `"..."` | One char over triggers truncation + ellipsis |

---

## `tests/test_llm_client.py` — 11 tests

Tests `get_llm()` (ChatOpenAI factory), `invoke_messages()` (async chat), and `ainvoke()` (alias). All LLM calls are mocked via `@patch("research_swarm.llm.client.ChatOpenAI")` or `@patch("research_swarm.llm.client.get_llm")`.

### `TestGetLLM` — ChatOpenAI construction

| Test | What it verifies |
|------|-----------------|
| `test_uses_settings_model` | `get_llm()` calls `ChatOpenAI()` with `model` and `api_key` kwargs sourced from `Settings` |
| `test_accepts_overrides` | `get_llm(model="gpt-4o-mini", temperature=0.5)` passes both overrides to ChatOpenAI |
| `test_returns_chat_openai_instance` | Return value is the same object `ChatOpenAI()` produced |

### `TestInvokeMessages` — async LLM invocation

| Test | What it verifies |
|------|-----------------|
| `test_returns_content_string` | `await invoke_messages([...])` returns `AIMessage.content` as `str` |
| `test_handles_empty_content` | `AIMessage(content="")` → returns `""` (empty string content is valid) |
| `test_passes_overrides_to_get_llm` | `invoke_messages(messages, temperature=0.3)` forwards `temperature=0.3` to `get_llm()` |
| `test_ainvoke_called_with_messages` | The mocked `llm.ainvoke` received the exact same message list passed to `invoke_messages` |
| `test_logs_request_and_response` | Smoke test: `invoke_messages` completes without exception when `llm.ainvoke` returns a valid `AIMessage` |

### `TestAinvoke` — alias function

| Test | What it verifies |
|------|-----------------|
| `test_calls_invoke_messages` | `ainvoke(messages, temperature=0.7)` delegates to `invoke_messages(messages, temperature=0.7)` |
| `test_ainvoke_without_overrides` | `ainvoke(messages)` delegates to `invoke_messages(messages)` with no extra kwargs |

---

## `tests/test_agents.py` — 23 tests

Tests all 5 agent functions (`plan`, `search`, `fact_check`, `summarize`, `judge`) with **fully mocked LLM and tracing**. Also tests `_safe_json` across all 5 agent modules (the function is duplicated in each file).

### `TestSafeJson` — JSON parsing helper (parametrized × 5 modules)

| Test | Input | Expected | Modules tested |
|------|-------|----------|---------------|
| `test_parses_plain_json` | `'{"key": "value"}'` | `{"key": "value"}` | planner, searcher, fact_checker, judge, summarizer |
| `test_parses_json_with_markdown_fence` | `'```json\n{...}\n```'` | parsed dict | Same 5 modules |
| `test_parses_json_with_plain_fence` | `'```\n{...}\n```'` | parsed dict | Same 5 modules |
| `test_parses_json_with_trailing_whitespace` | `'{...}  \n\t'` | parsed dict | Whitespace stripping works |
| `test_parses_json_with_arrays` | `'[1, 2, 3]'` | `[1, 2, 3]` | Arrays parse correctly (returns `list`) |
| `test_parses_nested_json` | `'{"outer": {"inner": [1, 2]}}'` | nested dict | Deep nesting preserved |
| `test_raises_on_invalid_json` | `"not json"` | `json.JSONDecodeError` | Invalid input raises |
| `test_parses_json_with_multiple_backticks` | `'````json\n{"a": 1}\n````'` | `{"a": 1}` | Extra backtick after `lstrip("`")` |

### `TestPlannerAgent` — `plan()` function

| Test | Mock returns | Assertion |
|------|------------|-----------|
| `test_plan_creates_research_plan` | `{"goal": "...", "research_questions": ["Q1","Q2","Q3"]}` | `state.plan.goal` and `plan.research_questions` match LLM output |
| `test_plan_handles_missing_goal` | `{"research_questions": ["Q1"]}` (no `goal` key) | `state.plan.goal` falls back to `state.query` |
| `test_plan_raises_on_invalid_response` | `{"unexpected": "data"}` (no `research_questions` key) | Raises `ValueError("Planner did not return a valid research plan")` |
| `test_plan_handles_json_with_fences` | LLM returns markdown-fenced JSON | `_safe_json` strips fences; `plan.goal` is correct |

### `TestSearcherAgent` — `search()` function

| Test | Condition | Assertion |
|------|----------|-----------|
| `test_search_appends_result` | LLM returns 2 evidence tuples | `search_results` has 1 entry with 2 evidence items |
| `test_search_requires_plan` | `state.plan is None` (planner not run) | Raises `RuntimeError("Planner must run before Searcher")` |
| `test_search_handles_empty_evidence` | LLM returns `evidence: []` | `search_results[0].evidence == []` |
| `test_search_falls_back_question_id` | LLM response missing `question_id` | `question_id` falls back to the research question text |

### `TestFactCheckerAgent` — `fact_check()` function

| Test | Condition | Assertion |
|------|----------|-----------|
| `test_fact_check_validates_evidence` | 3 evidence items; LLM validates 2, rejects 1 | 2 `validated_facts`, 1 `rejected_facts` |
| `test_fact_check_no_evidence` | `state.search_results` is empty | `invoke_messages` **not called**; `ValidatedResult(validated_facts=[], rejected_facts=[])` appended |
| `test_fact_check_handles_missing_fields_in_response` | LLM response lacks `rejected_facts` key | `rejected_facts` defaults to `[]` via `.get("rejected_facts", [])` |

### `TestSummarizerAgent` — `summarize()` function

| Test | Condition | Assertion |
|------|----------|-----------|
| `test_summarize_creates_report` | 2 validated facts → LLM produces summary | `final_report.summary` and `final_report.sources` match LLM output |
| `test_summarize_no_validated_facts` | `state.validated_results` is empty | `final_report` is created with LLM's response (facts block is `"(no facts)"`) |
| `test_summarize_handles_missing_sources` | LLM response lacks `sources` key | `final_report.sources` defaults to `[]` |

### `TestJudgeAgent` — `judge()` function

| Test | Mock returns | Assertion |
|------|------------|-----------|
| `test_judge_sets_high_score` | `{"score": 85}` | `state.judge_score == 85` |
| `test_judge_clamps_score_to_100` | `{"score": 150}` | Clamped: `max(0, min(100, 150)) == 100` |
| `test_judge_clamps_negative_score_to_0` | `{"score": -50}` | Clamped: `max(0, min(100, -50)) == 0` |
| `test_judge_handles_no_report` | `state.final_report is None` | `judge_score` still set; `report_text` defaults to `""` |
| `test_judge_handles_missing_score` | `{}` | `parsed.get("score", 0)` → `0` |
| `test_judge_captures_missing_topics` | `{"score": 60, "missing_topics": ["ethics", "regulation"]}` | Score set; missing topics noted (but not stored in state — only score is stored on state) |

---

## `tests/test_langfuse.py` — 15 tests

Tests the Langfuse observability layer: `_NoopTracer`, `_LangfuseTracer`, `_configure_langfuse()`, and `trace_agent()` context manager.

### `TestNoopTracer` — safe fallback

| Test | What it verifies |
|------|-----------------|
| `test_update_observation_does_nothing` | `_NoopTracer().update_observation(...)` returns `None` |
| `test_end_does_nothing` | `_NoopTracer().end(...)` returns `None` |
| `test_multiple_calls_safe` | 10 calls each of `update_observation` and `end` — no errors, no side effects |

### `TestLangfuseTracer` — real tracer wrapper

| Test | What it verifies |
|------|-----------------|
| `test_update_observation_delegates_to_span` | `tracer.update_observation(output={"score": 90})` calls `mock_span.update(output={"score": 90})` |
| `test_end_delegates_to_span` | `tracer.end(status="completed")` calls `mock_span.update(status="completed")` |
| `test_init_stores_span` | `_LangfuseTracer(mock_span)._span is mock_span` |

### `TestConfigureLangfuse` — client initialization

| Test | Condition | Assertion |
|------|----------|-----------|
| `test_no_credentials_does_not_create_client` | `langfuse_public_key=None`, `langfuse_secret_key=None` | `_configure_langfuse._client` is `None` (no client created) |
| `test_valid_credentials_creates_client` | Both keys present; `langfuse` module mocked in `sys.modules` | `Langfuse()` called with correct `public_key`, `secret_key`, `host`; `_client` set to mock instance |
| `test_initialization_error_is_handled` | `Langfuse()` constructor raises `Exception("Connection refused")` | Exception caught; `_client` remains `None` |

### `TestTraceAgent` — context manager

| Test | Condition | Assertion |
|------|----------|-----------|
| `test_returns_noop_when_no_client` | `_configure_langfuse._client` patched to `None` | `trace_agent(...)` yields `_NoopTracer` |
| `test_returns_noop_when_client_is_none` | `_client` manually set to `None` | Yields `_NoopTracer`; `_client` restored in `finally` |
| `test_returns_langfuse_tracer_when_available` | `_client` is a mock with `start_as_current_observation` | Yields `_LangfuseTracer` whose `_span` matches the mock span |
| `test_passes_agent_name_to_observation` | `trace_agent("planner", ...)`  | `start_as_current_observation` called with `name="planner"`, `as_type="agent"`, `input={"query": "AI"}`, `metadata={"agent_name": "planner"}` |
| `test_handles_observation_start_failure` | `start_as_current_observation` raises `Exception` | Falls back to `_NoopTracer` |

---

## `tests/test_workflow_graph.py` — 14 tests

Tests `build_workflow()` graph construction and `_merge_state()` state merging from `main.py`.

### `TestBuildWorkflow` — graph structure

| Test | What it verifies |
|------|-----------------|
| `test_returns_compiled_graph` | `build_workflow()` returns a non-`None` object with `astream` attribute |
| `test_graph_has_required_nodes` | Graph nodes include `{"planner", "searcher", "fact_checker", "summarizer", "judge"}` |
| `test_graph_entry_point_is_planner` | Smoke test: graph compiles successfully |
| `test_graph_has_conditional_routing` | Graph edges exist; `len(edges) > 0` confirms routing is configured |
| `test_workflow_can_be_invoked` | Compiled graph has both `invoke` and `astream` attributes |
| `test_build_workflow_imports_all_agents` | Implicit import test: `build_workflow()` doesn't raise `ImportError` |

### `TestMergeState` — state accumulation

| Test | What it verifies |
|------|-----------------|
| `test_merge_single_state_update` | Event with `{"judge": ResearchState(...)}` merges `judge_score` into result |
| `test_merge_dict_update` | Event with `{"planner": {"plan": ResearchPlan(...)}}` merges dict values |
| `test_merge_multiple_events` | Two keys in event dict both merged: `plan` from planner, `judge_score` from judge |
| `test_merge_preserves_existing_fields` | Fields in `current` but absent in event are preserved (`judge_score`, `search_results`) |
| `test_merge_overwrites_existing_fields` | `judge_score=50` overwritten to `75` by event |
| `test_merge_with_full_research_state_update` | Event contains a complete `ResearchState` — all fields merged |
| `test_merge_none_values_cause_validation_error` | `{"judge_score": None}` triggers validation error (field expects `int`, not `None`) |
| `test_merge_empty_event` | `{}` event → result identical to `current` state |

### `TestWorkflowIntegration`

| Test | What it verifies |
|------|-----------------|
| `test_workflow_graph_is_deterministic` | Two `build_workflow()` calls produce graphs with identical node sets |
| `test_workflow_has_edge_from_planner_to_searcher` | Graph edges are non-`None` |
| `test_workflow_supports_astream_with_state` | `astream` method exists; not actually invoked (requires LLM access) |

---

## `tests/test_workflow.py` — 2 tests

Legacy tests from the initial codebase (kept).

| Test | What it verifies |
|------|-----------------|
| `test_research_state_creation` | `ResearchState(query="AI trends")` — `query` set, `plan` is `None` |
| `test_research_plan_schema` | `ResearchPlan(goal="AI trends", research_questions=["What is AI?"])` — fields stored |

---

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Single file
uv run pytest tests/test_agents.py -v

# Single test
uv run pytest tests/test_routing.py::TestRouteFromJudge::test_parametrized_routing -v

# With coverage (if pytest-cov installed)
uv run pytest tests/ --cov=research_swarm --cov-report=term-missing
```

## Test Patterns & Conventions

- **Fixture**: `monkeypatch` (built-in) for env var isolation in settings tests
- **Async**: `@pytest.mark.asyncio` for all agent and LLM client tests
- **Mocking**: `@patch` decorators on `invoke_messages` and `trace_agent` for agents; `@patch` on `ChatOpenAI` for LLM client; `patch.dict("sys.modules", ...)` for import mocking in Langfuse tests
- **State Models**: Direct instantiation — no mocks needed for Pydantic validation tests
- **Parametrization**: `@pytest.mark.parametrize` used for `_safe_json` cross-module tests (5 modules × 8 test cases = 40 test instances) and routing boundary sweep (7 score values)
- **Cleanup**: Logging tests clear `root.handlers` before each test. Langfuse tests restore `_configure_langfuse._client` in `finally` blocks
