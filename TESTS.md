# Tests — Research Swarm

**242 tests · 10 files · 0 skipped · 100% pass rate**

Run: `uv run pytest tests/ -v`

---

## Test Architecture

```
tests/
├── test_state_models.py      # 49 tests — Pydantic state models (incl. new fields)
├── test_schemas.py           # 11 tests — Graph schema layer (AgentIO subclasses)
├── test_routing.py           # 34 tests — 5-stop-condition routing with parametrized sweep
├── test_settings.py          # 25 tests — Settings, env vars, validators
├── test_logging.py           # 16 tests — Logging config, preview helper
├── test_llm_client.py        # 10 tests — LLM client, get_llm, invoke_messages
├── test_agents.py            # 60 tests — All 5 agents (40 parametrized + 20 agent-specific) + _safe_json
├── test_langfuse.py          # 18 tests — Langfuse tracing, tracers, routing_decision span
├── test_workflow_graph.py    # 17 tests — Workflow graph, _merge_state, integration
└── test_workflow.py          #  2 tests — Legacy workflow tests (kept)
```

**Mocking strategy**: Agents use `@patch` on `invoke_messages` and `trace_agent` to isolate from LLM and Langfuse. Settings tests use `monkeypatch.setenv` to isolate from `.env` file and shell environment. Langfuse tests use `patch.dict("sys.modules", ...)` to mock the internal `from langfuse import Langfuse` import.

---

## `tests/test_state_models.py` — 49 tests

Pydantic model validation, defaults, serialization, and edge cases for all models in `research_swarm.graph.state`. Extended with tests for the iterative research loop state fields.

### `TestAgentIO` (2 tests)

| Test | What it verifies |
|------|-----------------|
| `test_agentio_is_base_model` | `AgentIO` inherits from `pydantic.BaseModel`; `model_dump()` returns `{}` |
| `test_agentio_allows_extra_fields` | `AgentIO` exposes `model_dump` method |

### `TestSearchResult` (7 tests)

| Test | What it verifies |
|------|-----------------|
| `test_creation_with_empty_evidence` | `SearchResult(question_id="q1", evidence=[])` — empty list valid |
| `test_creation_with_evidence` | Two evidence strings stored with source annotations; order preserved |
| `test_requires_question_id` | Omitting `question_id` raises `ValidationError` |
| `test_evidence_field_is_required` | Omitting `evidence` raises `ValidationError` (no default) |
| `test_serialization_roundtrip` | `model_dump()` → `SearchResult(**data)` equality |
| `test_json_serialization` | `model_dump_json()` contains both `question_id` and evidence |
| `test_question_id_is_string` | Pydantic coerces `"42"` to `str` |

### `TestValidatedResult` (4 tests)

| Test | What it verifies |
|------|-----------------|
| `test_creation_with_both_lists` | `validated_facts` and `rejected_facts` both stored |
| `test_creation_empty` | Both fields accept `[]` |
| `test_all_fields_are_required` | `ValidatedResult()` raises `ValidationError` |
| `test_requires_string_lists` | `[1, 2, 3]` rejected (expects `list[str]`) |

### `TestResearchPlan` (8 tests)

| Test | What it verifies |
|------|-----------------|
| `test_creation_with_single_question` | Single-element list and string goal |
| `test_creation_with_multiple_questions` | Three-element list with `len()` check |
| `test_requires_goal` | Omitting `goal` raises `ValidationError` |
| `test_requires_research_questions` | Omitting `research_questions` raises `ValidationError` |
| `test_empty_questions_list` | `research_questions=[]` is valid |
| `test_goal_must_be_string` | `goal=42` rejected |
| `test_questions_must_be_strings` | `research_questions=[1,2,3]` rejected |
| `test_model_dump_output` | Returns `{"goal": ..., "research_questions": [...]}` |

### `TestResearchReport` (6 tests)

| Test | What it verifies |
|------|-----------------|
| `test_creation_with_summary_and_sources` | Two sources and summary stored |
| `test_creation_with_empty_sources` | `sources=[]` valid |
| `test_requires_summary` | Omitting `summary` raises `ValidationError` |
| `test_requires_sources` | Omitting `sources` raises `ValidationError` |
| `test_summary_must_be_string` | `summary=123` rejected |
| `test_sources_must_be_strings` | `sources=[1,2,3]` rejected |

### `TestJudgeResult` (8 tests) — **new fields**: `strengths`, `weaknesses`, `reasoning`

| Test | What it verifies |
|------|-----------------|
| `test_creation_high_score` | `score=95`, `needs_research=False`, empty `missing_topics` |
| `test_creation_low_score` | `score=40`, `needs_research=True`, 2 missing topics |
| `test_does_not_validate_score_range` | `score=150` accepted (no Pydantic constraints) |
| `test_score_zero` | `score=0` valid |
| `test_all_fields_defaulted_except_score_needs_research_missing_topics` | `strengths`, `weaknesses`, `reasoning` default to `[]`/`""` |
| `test_full_judge_result` | All 6 fields populated: strengths, weaknesses, reasoning |
| `test_serialization_roundtrip` | `model_dump()` → `JudgeResult(**data)` equality |
| `test_score_must_be_int` | `score="high"` rejected |

### `TestResearchState` (14 tests) — **new fields**: `iteration`, `max_iterations`, `previous_score`, `score_delta`, `missing_topics`, `no_progress`, `stop_reason`, `new_evidence_found`

| Test | What it verifies |
|------|-----------------|
| `test_minimal_creation` | All 14 fields verified: `iteration=0`, `max_iterations=3`, `previous_score=None`, `score_delta=None`, `missing_topics=[]`, `no_progress=False`, `stop_reason=""`, `new_evidence_found=True` |
| `test_full_creation` | All 14 fields populated simultaneously with nested models and iteration metadata |
| `test_requires_query` | `ResearchState()` raises `ValidationError` |
| `test_query_must_be_string` | `query=42` rejected |
| `test_judge_score_default_zero` | `judge_score` is `0` by default |
| `test_plan_defaults_to_none` | `plan` is `None` by default |
| `test_search_results_default_to_empty_list` | `search_results` is `[]` by default |
| `test_validated_results_default_to_empty_list` | `validated_results` is `[]` by default |
| `test_final_report_defaults_to_none` | `final_report` is `None` by default |
| `test_model_dump_with_none_fields` | `model_dump()` returns all 14 keys with correct defaults |
| `test_model_dump_with_populated_fields` | `model_dump()` on state with nested `ResearchPlan` serializes dicts |
| `test_multiple_search_results` | Two `SearchResult` objects stored by index |
| `test_multiple_validated_results` | Two `ValidatedResult` objects stored |
| `test_convenience_for_empty_check` | `not state.search_results` evaluates `True` for empty lists |

---

## `tests/test_schemas.py` — 11 tests

Tests models in `research_swarm.graph.schemas` — all inherit from `AgentIO`. Updated for new `JudgeResult` fields.

### `TestSchemataModels` (11 tests)

| Test | What it verifies |
|------|-----------------|
| `test_research_plan_extends_agentio` | `schemas.ResearchPlan` is a subclass of `AgentIO` |
| `test_research_plan_has_correct_fields` | `goal` and `research_questions` operational |
| `test_research_plan_serialization` | `model_dump()` returns correct dict shape |
| `test_research_report_extends_agentio` | `schemas.ResearchReport` is a subclass of `AgentIO` |
| `test_research_report_has_correct_fields` | `summary` and `sources` operational |
| `test_research_report_serialization` | `model_dump()` returns correct dict shape |
| `test_judge_result_extends_agentio` | `schemas.JudgeResult` is a subclass of `AgentIO` |
| `test_judge_result_has_correct_fields` | `score`, `needs_research`, `missing_topics` operational |
| `test_judge_result_serialization` | `model_dump()` returns all 6 keys including new `strengths`, `weaknesses`, `reasoning` fields with default `[]`/`""` |
| `test_judge_result_all_fields_required` | `JudgeResult(score=50)` alone raises exception |
| `test_judge_result_optional_fields_default` | `strengths`, `weaknesses`, `reasoning` default to `[]`/`""` |

---

## `tests/test_routing.py` — 34 tests

Completely rewritten for the production-grade loop protection logic. Covers all 5 stop conditions, delta computation, priority ordering, and parametrized boundary sweep. Constants `MIN_DELTA=5` and `PASS_THRESHOLD=80` imported from source.

### `TestRouteFromJudgeStopConditions` (15 tests) — each stop condition + continue path

#### Condition A: score ≥ 80

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_stops_on_score_above_threshold` | `judge_score=80` | Route = `"__end__"`, stop_reason = `"score_threshold_met"` |
| `test_stops_on_score_equals_threshold` | `judge_score=PASS_THRESHOLD` (80) | Same as above |
| `test_score_above_threshold_takes_priority` | Score=85, iteration≥max | Score beats max_iterations; stop_reason = `"score_threshold_met"` |

#### Condition B: max iterations reached

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_stops_on_max_iterations_reached` | `iteration=3, max_iterations=3` | Route = `"__end__"`, reason = `"max_iterations_reached"` |
| `test_stops_on_iteration_exceeds_max` | `iteration=4, max_iterations=3` | Same (exceeds safety limit) |
| `test_iteration_below_max_continues` | `iteration=2, max_iterations=3` | Route = `"searcher"` |

#### Condition C: insufficient progress (delta < 5)

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_stops_on_insufficient_progress` | Score 62→64, delta=2 | Route = `"__end__"`, reason = `"insufficient_progress"`, `no_progress=True` |
| `test_small_positive_delta_stops` | Score 60→62, delta=2 | `score_delta=2` computed; stops |
| `test_exact_min_delta_stops` | Score 60→64, delta=4 (`< 5`) | Stops (delta is *strictly* less than `MIN_DELTA`) |
| `test_large_delta_continues` | Score 60→70, delta=10 | Route = `"searcher"`, `score_delta=10` |
| `test_delta_not_checked_on_first_iteration` | `iteration=0, previous_score=None` | `score_delta=None`; routing continues (delta check skipped) |

#### Condition D: no missing topics

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_stops_when_no_missing_topics` | `missing_topics=[]` | Route = `"__end__"`, reason = `"no_missing_topics"` |

#### Condition E: no new evidence

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_stops_when_no_new_evidence` | `new_evidence_found=False` | Route = `"__end__"`, reason = `"no_new_evidence"` |

#### Continue path

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_continues_when_all_checks_pass` | Score=70, iter=1, has topics, has evidence, delta=10 | Route = `"searcher"`, `iteration` incremented to 2 |
| `test_iteration_incremented_on_continue` | `iteration=0`, all checks pass | `state.iteration == 1` after routing |

### `TestScoreDeltaComputation` (4 tests)

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_delta_none_on_first_call` | `previous_score=None, judge_score=70` | `score_delta is None` |
| `test_delta_computed_on_second_call` | `previous_score=70, judge_score=75` | `score_delta == 5` |
| `test_delta_negative_on_score_regression` | `previous_score=70, judge_score=60` | `score_delta == -10` |
| `test_previous_score_updated_after_routing` | `previous_score=70, judge_score=75` | `previous_score` updated to `75` |

### `TestStopConditionPriority` (4 tests)

Verify that stop conditions are checked in the correct priority order: **A → B → C → D → E**.

| Test | Conflicting conditions | Winning reason |
|------|----------------------|---------------|
| `test_score_beats_max_iterations` | Score≥80 vs iter≥max | `"score_threshold_met"` |
| `test_max_iterations_beats_insufficient_progress` | iter≥max vs delta<5 | `"max_iterations_reached"` |
| `test_insufficient_progress_beats_no_missing_topics` | delta<5 vs topics=[] | `"insufficient_progress"` |
| `test_no_missing_topics_beats_no_new_evidence` | topics=[] vs no_evidence | `"no_missing_topics"` |

### `TestParametrizedRouting` (1 test, 11 cases)

Sweeps all 5 stop conditions and the continue path:

| Cases | Score | Iter | Max | Prev | Topics | NewEv | Route | Reason |
|-------|-------|------|-----|------|--------|-------|-------|--------|
| A×3 | 80,95,100 | 0 | 3 | None | ["t"] | True | `__end__` | `score_threshold_met` |
| B×2 | 70 | 3,5 | 3,5 | None | ["t"] | True | `__end__` | `max_iterations_reached` |
| C×2 | 64,42 | 1,2 | 3,5 | 62,40 | ["t"] | True | `__end__` | `insufficient_progress` |
| D×1 | 70 | 0 | 3 | None | [] | True | `__end__` | `no_missing_topics` |
| E×1 | 70 | 0 | 5 | None | ["t"] | False | `__end__` | `no_new_evidence` |
| Continue×2 | 70,79 | 0,1 | 3 | None,70 | ["t1","t2"],["t"] | True | `searcher` | — |

---

## `tests/test_settings.py` — 25 tests

Tests `Settings` (pydantic-settings), field validators, computed properties, alias mapping, and `get_settings` caching.

### `TestSettings` (9 tests)

| Test | What it verifies |
|------|-----------------|
| `test_default_values` | Defaults with env vars matching: model=`gpt-4o`, base_url→`None`, langfuse keys=`""`, mcp host/port |
| `test_custom_values` | All env vars set to non-default values (model, base_url, langfuse host, custom MCP) |
| `test_llm_api_key_property` | `llm_api_key` returns `openai_api_key` when non-empty |
| `test_llm_api_key_raises_when_empty` | `ValueError("OPENAI_API_KEY is required")` on empty key |
| `test_llm_base_url_property` | `llm_base_url` returns validated `openai_base_url` |
| `test_llm_base_url_returns_none_when_not_set` | `llm_base_url` → `None` when env `""` (validator converts) |
| `test_mcp_url_property` | `mcp_url` = `"http://{host}:{port}"` |
| `test_mcp_url_custom` | Custom host `10.0.0.1:5555` → `"http://10.0.0.1:5555"` |
| `test_env_var_loading` | `OPENAI_API_KEY` and `OPENAI_MODEL` loaded from env |

### `TestOpenAIBaseUrlNormalization` (9 tests) — field validator

| Test | Input env var | Expected | Rule |
|------|-------------|----------|------|
| `test_none_value_returns_none` | `""` | `None` | Empty → validator returns `None` |
| `test_empty_string_returns_none` | `""` | `None` | Redundant coverage of `""` path |
| `test_trailing_slash_removed` | `"...v1/"` | `"...v1"` | `.rstrip("/")` |
| `test_chat_completions_path_removed` | `".../chat/completions"` | `"...v1"` | Strips `/chat/completions` |
| `test_chat_completions_with_trailing_slash_removed` | `".../chat/completions/"` | `"...v1"` | Both stripped |
| `test_normal_url_preserved` | `"https://custom.api.com"` | `"https://custom.api.com"` | Clean pass-through |
| `test_url_with_multiple_slashes` | `"...////"` | `"...test.com"` | Multiple trailing slashes |
| `test_without_v1_path` | `"...groq.com/openai"` | `"...groq.com/openai"` | Non-standard preserved |
| `test_without_v1_trailing_slash` | `"...groq.com/openai/"` | `"...groq.com/openai"` | Trailing slash stripped |

### `TestSettingsAliases` (5 tests)

| Test | Env var | Field asserted |
|------|---------|---------------|
| `test_openai_api_key_alias` | `OPENAI_API_KEY` | `settings.openai_api_key` |
| `test_openai_model_alias` | `OPENAI_MODEL` | `settings.openai_model` |
| `test_openai_base_url_alias` | `OPENAI_BASE_URL` | `settings.openai_base_url` |
| `test_langfuse_public_key_alias` | `LANGFUSE_PUBLIC_KEY` | `settings.langfuse_public_key` |
| `test_langfuse_secret_key_alias` | `LANGFUSE_SECRET_KEY` | `settings.langfuse_secret_key` |

### `TestGetSettings` (2 tests)

| Test | What it verifies |
|------|-----------------|
| `test_returns_settings_instance` | `get_settings()` returns `Settings` |
| `test_caching_works` | Two calls return same object (`is`) — `@lru_cache()` |

---

## `tests/test_logging.py` — 16 tests

### `TestSetupTerminalLogging` (6 tests)

| Test | What it verifies |
|------|-----------------|
| `test_adds_handler_to_root_logger` | First call adds 1 `StreamHandler`; root at `INFO` |
| `test_second_call_does_not_add_duplicate_handler` | Second call returns early, no duplicate handler |
| `test_quiet_loggers_are_set_to_warning` | 9 third-party loggers at `WARNING` |
| `test_custom_level` | `level=logging.DEBUG` sets root to `DEBUG` |
| `test_handler_stream_is_stderr` | Handler writes to `sys.stderr` |
| `test_formatter_is_configured` | Handler has `logging.Formatter` instance |

### `TestPreview` (10 tests)

| Test | Input | Expected | Rule |
|------|-------|----------|------|
| `test_short_text_unchanged` | `"hello"` | `"hello"` | Within 160-char limit |
| `test_exact_limit_not_truncated` | `"a"×160` | `"a"×160` | At limit, no `...` |
| `test_long_text_truncated` | `"a"×200` | `≤160`, ends `"..."` | Above limit |
| `test_custom_limit` | `"a"×50`, limit=20 | 20 chars, ends `"..."` | Custom limit |
| `test_whitespace_normalized` | Multi-whitespace | `"hello world test"` | Collapsed |
| `test_very_long_text_truncated_properly` | 20× words, limit=50 | 50 chars, ends `"..."` | Word boundaries |
| `test_empty_string` | `""` | `""` | Empty → empty |
| `test_only_whitespace` | `"   \n\t  "` | `""` | Whitespace-only → empty |
| `test_text_exactly_at_limit` | `"x"×40`, limit=40 | `"x"×40` | No ellipsis |
| `test_text_one_over_limit` | `"x"×41`, limit=40 | 40 chars, ends `"..."` | One over triggers |

---

## `tests/test_llm_client.py` — 10 tests

### `TestGetLLM` (3 tests)

| Test | What it verifies |
|------|-----------------|
| `test_uses_settings_model` | `get_llm()` passes `model` and `api_key` to `ChatOpenAI()` |
| `test_accepts_overrides` | `get_llm(model="gpt-4o-mini", temperature=0.5)` forwards overrides |
| `test_returns_chat_openai_instance` | Return value is the mock instance |

### `TestInvokeMessages` (5 tests)

| Test | What it verifies |
|------|-----------------|
| `test_returns_content_string` | `await invoke_messages(...)` returns `str` content |
| `test_handles_empty_content` | `AIMessage(content="")` → returns `""` |
| `test_passes_overrides_to_get_llm` | `temperature=0.3` forwarded to `get_llm()` |
| `test_ainvoke_called_with_messages` | `llm.ainvoke` receives exact message list |
| `test_logs_request_and_response` | Smoke test: no exception with valid `AIMessage` |

### `TestAinvoke` (2 tests)

| Test | What it verifies |
|------|-----------------|
| `test_calls_invoke_messages` | `ainvoke(messages, temperature=0.7)` delegates correctly |
| `test_ainvoke_without_overrides` | `ainvoke(messages)` delegates with no extra kwargs |

---

## `tests/test_agents.py` — 60 tests

Tests all 5 agent functions with fully mocked LLM and tracing. The **60** count comes from 40 parametrized `_safe_json` instances (8 cases × 5 modules) + 20 agent-specific async tests.

### `TestSafeJson` (8 functions × 5 modules = 40 test instances)

Parametrized across `planner`, `searcher`, `fact_checker`, `judge`, `summarizer`:

| Test | Input | Expected |
|------|-------|----------|
| `test_parses_plain_json` | `'{"key": "value"}'` | `{"key": "value"}` |
| `test_parses_json_with_markdown_fence` | `'```json\n{...}\n```'` | parsed dict |
| `test_parses_json_with_plain_fence` | `'```\n{...}\n```'` | parsed dict |
| `test_parses_json_with_trailing_whitespace` | `'{...}  \n\t'` | parsed dict |
| `test_parses_json_with_arrays` | `'[1, 2, 3]'` | `[1, 2, 3]` |
| `test_parses_nested_json` | `'{"outer": {"inner": [1, 2]}}'` | nested dict |
| `test_raises_on_invalid_json` | `"not json"` | `json.JSONDecodeError` |
| `test_parses_json_with_multiple_backticks` | `'````json\n{"a":1}\n````'` | `{"a": 1}` |

### `TestPlannerAgent` (4 tests)

| Test | Mock returns | Assertion |
|------|------------|-----------|
| `test_plan_creates_research_plan` | `{"goal": "...", "research_questions": ["Q1","Q2","Q3"]}` | `plan.goal` and `plan.research_questions` match |
| `test_plan_handles_missing_goal` | No `goal` key | Falls back to `state.query` |
| `test_plan_raises_on_invalid_response` | No `research_questions` key | `ValueError("Planner did not return a valid research plan")` |
| `test_plan_handles_json_with_fences` | Markdown-fenced JSON | `_safe_json` strips fences; `plan.goal` correct |

### `TestSearcherAgent` (3 tests) — **updated for targeted search**

| Test | Condition | Assertion |
|------|----------|-----------|
| `test_full_search_appends_result` | `iteration=0`, has plan | Full search; 2 evidence items; `new_evidence_found=True` |
| `test_targeted_search_on_later_iteration` | `iteration=1`, `missing_topics=["security risks", "cost analysis"]` | Targeted search using `TARGETED_SEARCH_SYSTEM`; `new_evidence_found=True` |
| `test_search_no_evidence_on_iteration_zero` | `iteration=0`, empty evidence returned | `new_evidence_found=False` |

### `TestFactCheckerAgent` (3 tests)

| Test | Condition | Assertion |
|------|----------|-----------|
| `test_fact_check_validates_evidence` | 3 items; LLM validates 2, rejects 1 | 2 validated, 1 rejected |
| `test_fact_check_no_evidence` | Empty `search_results` | `invoke_messages` not called; empty `ValidatedResult` |
| `test_fact_check_handles_missing_fields_in_response` | No `rejected_facts` key | `rejected_facts=[]` via `.get()` |

### `TestSummarizerAgent` (3 tests)

| Test | Condition | Assertion |
|------|----------|-----------|
| `test_summarize_creates_report` | 2 validated facts | `final_report.summary` and `sources` match LLM |
| `test_summarize_no_validated_facts` | Empty `validated_results` | `final_report` created with `"(no facts)"` block |
| `test_summarize_handles_missing_sources` | No `sources` key | `final_report.sources=[]` |

### `TestJudgeAgent` (7 tests) — **enhanced with `missing_topics` tracking**

| Test | Mock returns | Assertion |
|------|------------|-----------|
| `test_judge_sets_high_score` | `score=85`, includes strengths/reasoning | `judge_score=85`, `missing_topics=[]` |
| `test_judge_sets_missing_topics` | `score=60`, `missing_topics=["security", "cost analysis"]` | `missing_topics` stored on state |
| `test_judge_clamps_score_to_100` | `score=150` | Clamped to `100` |
| `test_judge_clamps_negative_score_to_0` | `score=-50` | Clamped to `0` |
| `test_judge_handles_no_report` | `final_report is None` | Score still set; empty report handled |
| `test_judge_handles_missing_score` | `{}` | `.get("score", 0)` → `0` |
| `test_judge_captures_missing_topics` | `missing_topics=["ethics", "regulation"]` | Score 60; topics stored on state |

---

## `tests/test_langfuse.py` — 18 tests

Tests `_NoopTracer`, `_LangfuseTracer`, `_configure_langfuse()`, `trace_agent()`, and **new** `trace_routing_decision()` context manager.

### `TestNoopTracer` (3 tests)

| Test | What it verifies |
|------|-----------------|
| `test_update_observation_does_nothing` | Returns `None` |
| `test_end_does_nothing` | Returns `None` |
| `test_multiple_calls_safe` | 10× `update_observation` + 10× `end` — no errors |

### `TestLangfuseTracer` (3 tests)

| Test | What it verifies |
|------|-----------------|
| `test_update_observation_delegates_to_span` | Calls `mock_span.update(output=...)` |
| `test_end_delegates_to_span` | Calls `mock_span.update(status=...)` |
| `test_init_stores_span` | `tracer._span is mock_span` |

### `TestConfigureLangfuse` (3 tests)

| Test | Condition | Assertion |
|------|----------|-----------|
| `test_no_credentials_does_not_create_client` | Keys are `None` | `_client` is `None` |
| `test_valid_credentials_creates_client` | Both keys present; `langfuse` mocked in `sys.modules` | `Langfuse()` called; `_client` set |
| `test_initialization_error_is_handled` | `Langfuse()` raises `Exception` | Caught; `_client` remains `None` |

### `TestTraceAgent` (5 tests)

| Test | Condition | Assertion |
|------|----------|-----------|
| `test_returns_noop_when_no_client` | `_client` patched to `None` | Yields `_NoopTracer` |
| `test_returns_noop_when_client_is_none` | `_client` manually `None` | Yields `_NoopTracer`; restored in `finally` |
| `test_returns_langfuse_tracer_when_available` | `_client` with mock `start_as_current_observation` | Yields `_LangfuseTracer` |
| `test_passes_agent_name_to_observation` | `trace_agent("planner", ...)` | Called with `name="planner"`, `as_type="agent"`, input + metadata |
| `test_handles_observation_start_failure` | `start_as_current_observation` raises | Falls back to `_NoopTracer` |

### `TestTraceRoutingDecision` (4 tests) — **new**

| Test | Condition | Assertion |
|------|----------|-----------|
| `test_returns_noop_when_no_client` | `_client` patched to `None` | Yields `_NoopTracer` |
| `test_returns_langfuse_tracer_when_available` | `_client` with mock observation | Yields `_LangfuseTracer`; span accessible |
| `test_passes_routing_metadata_to_observation` | State with iteration=2, score=70, delta=5, missing_topics, max_iterations=3, new_evidence_found=True | `start_as_current_observation` called with `name="routing_decision"`, `as_type="chain"`, all routing input fields, and correct metadata |
| `test_handles_observation_start_failure` | `start_as_current_observation` raises | Falls back to `_NoopTracer` |

---

## `tests/test_workflow_graph.py` — 17 tests

### `TestBuildWorkflow` (6 tests)

| Test | What it verifies |
|------|-----------------|
| `test_returns_compiled_graph` | `build_workflow()` returns non-None with `astream` |
| `test_graph_has_required_nodes` | Nodes include all 5 agents |
| `test_graph_entry_point_is_planner` | Graph compiles successfully |
| `test_graph_has_conditional_routing` | Edges exist; `len(edges) > 0` |
| `test_workflow_can_be_invoked` | Has `invoke` and `astream` attributes |
| `test_build_workflow_imports_all_agents` | No `ImportError` on graph build |

### `TestMergeState` (8 tests) — **updated with new fields test**

| Test | What it verifies |
|------|-----------------|
| `test_merge_single_state_update` | Event `ResearchState` merges `judge_score` |
| `test_merge_dict_update` | Event dict merges `plan` |
| `test_merge_multiple_events` | Two keys (plan + judge_score) both merged |
| `test_merge_preserves_existing_fields` | Fields absent from event preserved |
| `test_merge_overwrites_existing_fields` | `judge_score` overwritten from 50→75 |
| `test_merge_with_full_research_state_update` | Complete `ResearchState` in event merged |
| `test_merge_updates_new_fields` | `iteration=2`, `missing_topics=["security"]`, `judge_score=70` all merged correctly |
| `test_merge_empty_event` | `{}` event → result matches current |

### `TestWorkflowIntegration` (3 tests)

| Test | What it verifies |
|------|-----------------|
| `test_workflow_graph_is_deterministic` | Two builds produce identical node sets |
| `test_workflow_has_edge_from_planner_to_searcher` | Graph edges non-None |
| `test_workflow_supports_astream_with_state` | `astream` method exists |

---

## `tests/test_workflow.py` — 2 tests

Legacy tests from the initial codebase (kept).

| Test | What it verifies |
|------|-----------------|
| `test_research_state_creation` | `ResearchState(query="AI trends")` — query set, plan is None |
| `test_research_plan_schema` | `ResearchPlan(goal="...", research_questions=["..."])` — fields stored |

---

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Single file
uv run pytest tests/test_routing.py -v

# Single test
uv run pytest tests/test_routing.py::TestParametrizedRouting::test_parametrized_routing -v

# With coverage (if pytest-cov installed)
uv run pytest tests/ --cov=research_swarm --cov-report=term-missing
```

## Test Patterns & Conventions

- **Fixture**: `monkeypatch` (built-in) for env var isolation in settings tests
- **Async**: `@pytest.mark.asyncio` for all agent and LLM client tests
- **Mocking**: `@patch` on `invoke_messages` and `trace_agent` for agents; `@patch` on `ChatOpenAI` for LLM client; `patch.dict("sys.modules", ...)` for import mocking in Langfuse tests
- **State Models**: Direct instantiation — no mocks needed for Pydantic validation tests
- **Parametrization**: `@pytest.mark.parametrize` for `_safe_json` cross-module tests (5 modules × 8 cases = 40 instances) and routing boundary sweep (11 score/delta/topic combinations)
- **Cleanup**: Logging tests clear `root.handlers` before each test. Langfuse tests restore `_configure_langfuse._client` in `finally` blocks
- **State factory**: `TestRouteFromJudgeStopConditions._state()` helper method provides sensible defaults for all 14 ResearchState fields, accepting kwargs overrides
