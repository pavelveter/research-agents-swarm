"""Planner agent — production grade with non-destructive state mutations."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents._planner_helpers import (
    _adversarial_challenge,
    _build_search_packet,
    _enforce_traceability,
)
from config.domain import get_domain
from graph.state import ResearchPlan, ResearchState
from llm.client import invoke_messages
from logging_config import preview
from observability.langfuse import trace_agent
from utils import safe_json, render_prompt

logger = logging.getLogger(__name__)

# Prompts moved to jinja


async def plan(state: ResearchState) -> ResearchState:
    """Analyze the user query and produce or extend a research plan without data loss."""
    with trace_agent(
        "planner",
        input_data={"query": state.query, "iteration": state.iteration},
    ) as tracer:
        if state.iteration == 0:
            plan_data = await _initial_plan(state)

            # T15: Adversarial challenge — review draft plan for blind spots
            if isinstance(plan_data, dict) and "research_questions" in plan_data:
                challenge = await _adversarial_challenge(plan_data, state.query)
                if not challenge["passed"]:
                    logger.info(
                        "Adversarial challenge flagged risks — refining initial plan"
                    )
                    refine_prompt = (
                        "Refine the research plan to address these risks:\n"
                        + "\n".join(f"- {r}" for r in challenge.get("risk_flags", [])[:5])
                        + "\n\nOriginal plan questions to improve:\n"
                        + "\n".join(f"- {q}" for q in plan_data["research_questions"])
                    )
                    refine_messages = [
                        SystemMessage(
                            content=(
                                "You are a research planner. Refine the following research questions "
                                "to address the identified risks. Keep questions specific and scoped. "
                                "Return ONLY valid JSON with keys 'goal' and 'research_questions'."
                            )
                        ),
                        HumanMessage(content=refine_prompt),
                    ]
                    try:
                        domain = get_domain(state.query)
                        temperature = max(0.0, 1.0 - domain.strictness)
                        refined_raw = await invoke_messages(refine_messages, temperature=temperature)
                        refined = safe_json(refined_raw)
                        if refined and "research_questions" in refined:
                            plan_data = refined
                            logger.info("Adversarial refinement produced updated plan")
                    except Exception as exc:
                        logger.warning("Adversarial refinement call failed: %s", exc)

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
                # T17: Enforce traceability — at least 2 questions must share keywords with topics
                topics = [t for t in state.missing_topics if not str(t).startswith("[SYSTEM]")]
                new_questions = _enforce_traceability(new_questions, topics)

                if state.plan:
                    # T6: cap at 3 narrow questions, enforce analytical quota
                    new_questions = new_questions[:3]
                    state.plan.research_questions = new_questions
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
        SystemMessage(content=render_prompt("planner_plan_system.jinja")),
        HumanMessage(content=f"Topic: {state.query}"),
    ]
    try:
        domain = get_domain(state.query)
        temperature = max(0.0, 1.0 - domain.strictness)
        logger.info("Planner temperature=%.2f (domain=%s strictness=%.2f)", temperature, domain.slug, domain.strictness)
        raw = await invoke_messages(messages, temperature=temperature)
        return safe_json(raw)
    except Exception as exc:
        logger.error("Initial plan LLM invocation failed: %s", exc, exc_info=True)
        return {}


async def _refine_plan(state: ResearchState) -> dict[str, Any]:
    # T6: deep-dive mode — build plan ONLY from missing_topics when iteration > 0
    topics = [t for t in state.missing_topics if not str(t).startswith("[SYSTEM]")]
    if not topics:
        logger.info("Re-plan: all missing_topics are [SYSTEM] flags — skipping refinement")
        return {}
    logger.info(
        "Planning research | mode=deep_dive iteration=%s missing_topics=%s",
        state.iteration,
        topics,
    )
    # T17: Build focused search packet from missing_topics
    packet = _build_search_packet(state.missing_topics)
    packet_lines = ""
    if packet:
        packet_lines = (
            f"Keywords: {', '.join(packet.get('keywords', []))}\n"
            f"Source hints: {', '.join(packet.get('source_hints', []))}\n"
        )
    # T6+T9: pass iteration to prompt for analytical depth quota
    human_content = (
        f"Iteration: {state.iteration}\n"
        + packet_lines
        + "MISSING TOPICS (derive questions from these — PRIMARY):\n"
        + "\n".join(f"- {t}" for t in topics)
    )
    messages = [
        SystemMessage(content=render_prompt("planner_replan_system.jinja")),
        HumanMessage(content=human_content),
    ]
    try:
        domain = get_domain(state.query)
        temperature = max(0.0, 1.0 - domain.strictness)
        logger.info("Re-planner temperature=%.2f (domain=%s strictness=%.2f)", temperature, domain.slug, domain.strictness)
        raw = await invoke_messages(messages, temperature=temperature)
        return safe_json(raw)
    except Exception as exc:
        logger.error("Refine plan LLM invocation failed: %s", exc, exc_info=True)
        return {}
