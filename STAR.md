# Case Study: Resolving State Decay and Multi-Agent Loop Stagnation in LangGraph

## TL;DR
Designed a multi-agent text-mining graph that suffered from performance stagnation on iterations 3+ due to state mutations, context explosions, and non-deterministic evaluation. Refactored the core loop with defensive state management, an abstract provider orchestration fallback layer, and deterministic grading, driving the research score from a dead loop failure up to a publication-ready rating (>80/100).

---

## 1. Situation
While engineering a cyclic multi-agent research swarm built on **LangGraph**, the graph began experiencing catastrophic metric degradation during prolonged execution loops. 

The architecture leveraged a continuous feedback cycle:
`Planner (Decomposes query)` → `Searcher (Web ingestion)` → `Fact Checker (Filtering)` → `Summarizer (Synthesis)` → `Judge (LLM Quality Evaluator)`.

During testing runs on deep infrastructure topics, the system would peak around iteration 2 or 3 (e.g., a Quality Score of 59/100) and then experience severe score degradation, throwing an `insufficient_progress` signal and terminating with low-quality, non-converged assets.

---

## 2. Task
As the AI Infrastructure Lead, I had to perform an deep runtime post-mortem to isolate the architectural flaws causing the swarm to stall. The goals were to:
1. Prevent memory amnesia and state degradation over time.
2. Enforce structural stability for token-heavy payloads inside the shared state.
3. Stabilize the agentic loop to consistently break past the target quality score of **80+/100**.

---

## 3. Action

I conducted a thorough system audit via runtime trace inspection, identifying and executing four core architectural fixes:

### A. Resolved State Amnesia & Deduplication Mutators
* **The Bug:** The `searcher` agent was wiping out the execution history on each step by re-initializing the internal collections instead of executing non-destructive state mutations.
* **The Fix:** Rebuilt the ingestion pipeline to append tracking objects (`SearchResult`) into historical arrays, using an internal SHA-256 caching ledger (`known_evidence_hashes`) to guarantee deterministic data deduplication across steps.

### B. Decoupled Agents from Infrastructure with a Provider Fallback Orchestrator
* **The Bug:** Agents tightly coupled to raw, un-abstracted search utility functions threw critical runtime exceptions (`NameError`) or failed completely when rate-limits or API quotas collapsed mid-run.
* **The Fix:** Implemented a thread-safe `SearchOrchestrator` fallback pattern. The search agent now targets the orchestrator singleton interface rather than explicit vendor endpoints, gracefully cascading queries down an automated fallback array (`Tavily` → `Brave` → `SerpAPI` → `DuckDuckGo`).

### C. Eliminated "Judge Hallucination" via Structured Outputs
* **The Bug:** The LLM-backed Judge agent was using loose prompting rules. Every time the compiler appended fresh data, the Judge expanded its requirements mid-flight—demanding unrelated technical breakdowns (CVEs, low-level architecture schematics) and shifting goals dynamically.
* **The Fix:** Hardened the evaluation agent with weighted component scoring matrices and strict schemas. Shifted the Judge's missing topics array into a strict delta-tracker, preventing it from dynamically expanding scope based on newly introduced keywords.

### D. Patched Summarizer JSON Overloading
* **The Bug:** The state model crammed the entire report synthesis inside a single string field (`summary`), forcing the LLM to pack 40+ granular facts into an unformatted JSON dump, resulting in critical context fragmentation.
* **The Fix:** Rewrote the compilation prompt to strip conversational filler, optimized the extraction window, and laid the groundwork for a Multi-Key State layout separating the abstract, deep-dive data, and pricing into distinct schemas.

---

## 4. Result

* **Convergence Recovery:** Fixed the loop degradation bug entirely; the system stopped throwing early termination flags (`insufficient_progress`).
* **Score Metrics Boost:** The Multi-Agent pipeline successfully escaped its quality loop trap, moving from a stalled **40/100** score up to an excellent, production-validated **84/100**.
* **Enterprise Stability:** The integration of the search provider fallback chain minimized zero-evidence failures to 0%, ensuring reliable enterprise operations even during external provider timeouts.
* **System Observability:** Consolidated tracking structures enabled deterministic metric evaluation logs, making the multi-agent system fully auditable and observable across all routing operations.