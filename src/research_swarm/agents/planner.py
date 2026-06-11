"""Planner agent — production grade with non-destructive state mutations."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.graph.state import ResearchPlan, ResearchState
from research_swarm.llm.client import invoke_messages
from research_swarm.logging_config import preview
from research_swarm.observability.langfuse import trace_agent
from research_swarm.utils import safe_json

logger = logging.getLogger(__name__)

PLAN_SYSTEM = (
    "You are a research planner. Given a user research topic, decompose it into "
    "3-7 focused research questions. Return ONLY valid JSON with keys:\n"
    "{\n"
    '  "goal": "string describing main objective",\n'
    '  "research_questions": ["question 1", "question 2"]\n'
    "}"
)

REPLAN_SYSTEM = (
    "You are an AI Infrastructure Re-planner. Review existing questions and missing topics.\n"
    "Generate 1-3 highly technical, targeted questions to bridge the gaps. "
    "Do not use generic verbs like 'analyze' or 'study'. Use operational tasks like "
    "'Compare the AST parsing overhead in Cursor vs Windsurf', 'Identify the specific "
    "per-engineer license compliance risks of Copilot in enterprise VPC environments'.\n"
    "Return ONLY valid JSON with keys 'goal' and 'research_questions'."
)


async def plan(state: ResearchState) -> ResearchState:
    """Analyze the user query and produce or extend a research plan without data loss."""
    with trace_agent(
        "planner",
        input_data={"query": state.query, "iteration": state.iteration},
    ) as tracer:
        if state.iteration == 0:
            plan_data = await _initial_plan(state)

            # Bulletproof fallback if initial logic fails
            if not isinstance(plan_data, dict) or "research_questions" not in plan_data:
                logger.warning(
                    "Initial planner failed to return valid schema. Applying rescue plan."
                )
                plan_data = {
                    "goal": state.query,
                    "research_questions": [
                        f"Analyze primary aspects and current state of: {state.query}"
                    ],
                }

            state.plan = ResearchPlan(
                goal=str(plan_data.get("goal", state.query)),
                research_questions=[str(q) for q in plan_data["research_questions"]],
            )
        else:
            plan_data = await _refine_plan(state)

            # If refinement fails, just keep the current plan intact
            if isinstance(plan_data, dict) and "research_questions" in plan_data:
                new_questions = [str(q) for q in plan_data["research_questions"]]

                if state.plan:
                    # КРИТИЧЕСКИЙ ФИКС: Не затираем, а мержим списки вопросов
                    existing_set = set(state.plan.research_questions)
                    deduped_new = [q for q in new_questions if q not in existing_set]

                    state.plan.research_questions.extend(deduped_new)
                    if "goal" in plan_data:
                        state.plan.goal = str(plan_data["goal"])
                else:
                    state.plan = ResearchPlan(
                        goal=str(plan_data.get("goal", state.query)),
                        research_questions=new_questions,
                    )
            else:
                logger.warning(
                    "Refine plan failed or returned invalid format. Carrying over existing plan."
                )

        logger.info(
            "Plan updated | iteration=%s goal=%r total_questions=%s",
            state.iteration,
            preview(state.plan.goal, 80) if state.plan else "None",
            len(state.plan.research_questions) if state.plan else 0,
        )

        if state.plan and hasattr(tracer, "update_observation"):
            tracer.update_observation(output=state.plan.model_dump())

    return state


async def _initial_plan(state: ResearchState) -> dict[str, Any]:
    logger.info("Planning research | mode=initial query=%s", preview(state.query, 80))
    messages = [
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=f"Topic: {state.query}"),
    ]
    try:
        raw = await invoke_messages(messages)
        return safe_json(raw)
    except Exception as exc:
        logger.error("Initial plan LLM invocation failed: %s", exc, exc_info=True)
        return {}


async def _refine_plan(state: ResearchState) -> dict[str, Any]:
    topics = state.missing_topics if state.missing_topics else ["(no missing topics)"]
    existing = state.plan.research_questions if state.plan else []
    logger.info(
        "Planning research | mode=refine iteration=%s missing_topics=%s",
        state.iteration,
        topics,
    )
    messages = [
        SystemMessage(content=REPLAN_SYSTEM),
        HumanMessage(
            content=(
                "Existing questions:\n" + "\n".join(f"- {q}" for q in existing) + "\n\n"
                "Missing topics to cover:\n" + "\n".join(f"- {t}" for t in topics)
            )
        ),
    ]
    try:
        raw = await invoke_messages(messages)
        return safe_json(raw)
    except Exception as exc:
        logger.error("Refine plan LLM invocation failed: %s", exc, exc_info=True)
        return {}
