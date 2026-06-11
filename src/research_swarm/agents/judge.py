from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.llm.client import invoke_messages
from research_swarm.observability.langfuse import trace_agent
from research_swarm.graph.state import JudgeResult, ResearchState

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You are a research quality judge. Evaluate the report on completeness, "
    "source coverage, and factual consistency. Respond with ONLY valid JSON: "
    "{ \"score\": 0, \"needs_research\": false, \"missing_topics\": [] } "
    "Score must be an integer from 0 to 100."
)


async def judge(state: ResearchState) -> ResearchState:
    """Score the final report and decide if more research is needed."""
    with trace_agent(
        "judge", input_data={"has_report": state.final_report is not None}
    ) as tracer:
        report_text = state.final_report.summary if state.final_report else ""
        sources = (
            "\n".join(f"- {s}" for s in state.final_report.sources[:20])
            if state.final_report
            else "(none)"
        )
        messages = [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(
                content=(
                    "Report:\n"
                    f"{report_text}\n\n"
                    "Sources:\n"
                    f"{sources}\n\n"
                    "Research questions:\n"
                    + (
                        "\n".join(f"- {q}" for q in state.plan.research_questions)
                        if state.plan
                        else "(none)"
                    )
                )
            ),
        ]
        logger.info("Judging report")
        raw = await invoke_messages(messages)
        parsed = _safe_json(raw)
        score = int(parsed.get("score", 0))
        state.judge_score = max(0, min(100, score))
        needs_research = bool(parsed.get("needs_research", False))
        missing_topics = [str(t) for t in parsed.get("missing_topics", [])]
        logger.info(
            "Judge score=%s needs_research=%s missing_topics=%s",
            state.judge_score,
            needs_research,
            len(missing_topics),
        )
        if hasattr(tracer, "update_observation"):
            tracer.update_observation(
                output={
                    "score": state.judge_score,
                    "needs_research": needs_research,
                    "missing_topics": missing_topics,
                }
            )
    return state


def _safe_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip("`\n ")
    return json.loads(cleaned)
