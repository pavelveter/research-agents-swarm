# Research Swarm — Quality Improvement Plan (Target: 80+ Judge Score)

## 1. Summarizer: 50 facts + strict citation format
- **Increase** Qdrant retrieval limit from 25 → 50 facts.
- **Add** explicit JSON schema in `summarizer_summarizer_system.jinja` requiring `citations` array alongside `summary`.
- **Add** fallback in `summarizer.py`: if parsed JSON is empty, merge previous report with new facts via a secondary LLM call (low temperature).
- **Risk**: 50 facts ≈ 30-40k tokens. Verify model context window supports this before merging.

## 2. Judge: explicit rubric + temperature=0
- **Set** `temperature=0` in Judge LLM invocation to eliminate score variance.
- **Add** concrete scoring rubric with per-category examples (not abstract guidelines).
- **Fix** system prompt bias: instruct Judge to base `missing_topics` ONLY on `research_questions` from the plan and actual report gaps. Strip hardcoded "frontier AI" framing unless the query is about AI.
- **Risk**: If current `invoke_messages` wrapper lacks `temperature` param, add it in `llm/client.py`.

## 3. Searcher: 12-15 results + adaptive fan-out
- **Increase** `max_results` to 12-15 on iteration 0 (full research), keep 7 for iterations >= 1 (targeted).
- **Fix** topic slice: use `state.missing_topics[:5]` instead of hardcoded `[:3]`.
- **Guard** against token blow-up: format raw search entries in batches of 10-15 before sending to LLM formatter, not one giant block.

## 4. Fact Checker: adaptive semantic threshold
- **Implement** gradual decay instead of one jump:
  - iteration 0 → threshold 0.94
  - iteration 1 → threshold 0.90
  - iteration 2+ → threshold 0.82
- **Pass** dynamic threshold into `memory.upsert_facts()` (currently hardcoded 0.92 in `vector_storage.py:80`).
- **Risk**: Lower thresholds may admit noise. Monitor `rejected_facts` ratio after deploy.

## 5. Explicit rewrite loop with previous_report
- **Add** field in `ResearchState`: `previous_report: ResearchReport | None = None`.
- **Copy** `state.final_report` into `state.previous_report` before each new Summarizer run (use non-deep copy for payloads to keep Qdrant dedup intact).
- **Rewrite prompt** instruction: "This is an incremental redesign. Preserve strong sections from previous report. Address Judge weaknesses and missing_topics using new facts. Do not restart from scratch."
- **Risk**: Without explicit "preserve + extend" instruction, Summarizer may ignore or shallow-copy the previous report.

## 6. Hard guard rails on Judge score (post-parse, not in prompt)
- **Add** deterministic score caps AFTER LLM response parsing in `judge.py`:
  - if `len(state.final_report.sources) < 5`: cap score at 75, append missing topic
  - if `total_facts_retrieved < 10`: cap score at 70, append missing topic
- **Why post-parse**: models reliably ignore conditional logic embedded in prose; code-level caps are deterministic.
