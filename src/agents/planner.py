"""Planner agent — production grade with non-destructive state mutations."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.domain import get_domain
from graph.state import ResearchPlan, ResearchState
from llm.client import invoke_messages
from logging_config import preview
from observability.langfuse import trace_agent
from utils import safe_json, render_prompt

logger = logging.getLogger(__name__)

# T17: Minimum keyword overlap required for a question to be traceable
_MIN_KEYWORD_OVERLAP = 2

# Prompts moved to jinja


def _build_search_packet(
    missing_topics: list[str],
) -> dict[str, Any]:
    """T17: Package judge's missing_topics into a focused search packet.

    Extracts keywords (>3 chars) and infers source-type hints (e.g., regulatory,
    academic, clinical) so the planner receives actionable input, not raw topics.
    """
    topics = [t for t in missing_topics if not str(t).startswith("[SYSTEM]")]
    if not topics:
        return {}
    keywords: list[str] = []
    for t in topics[:5]:
        words = [w.lower() for w in str(t).split() if len(w) > 3]
        keywords.extend(words)
    unique_keywords = list(dict.fromkeys(keywords))[:15]

    # Source-type hints — always include academic fallbacks for deep-dive queries
    text = " ".join(topics).lower()
    hints: list[str] = [
        "site:ssrn.com",
        "site:arxiv.org",
        "filetype:pdf",
        "technical whitepaper",
    ]
    if any(w in text for w in ("regulation", "act", "law", "compliance", "guideline")):
        hints.append("regulatory")
    if any(w in text for w in ("clinical", "trial", "drug", "patient", "disease")):
        hints.append("clinical")
    if any(w in text for w in ("algorithm", "model", "benchmark", "architecture")):
        hints.append("technical")
    if any(w in text for w in ("study", "research", "paper", "review", "meta")):
        hints.append("academic")

    return {
        "keywords": unique_keywords,
        "source_hints": hints,
        "topics": topics,
    }


async def _adversarial_challenge(
    plan_data: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    """T15: Devil's Advocate step — generate 3 challenge questions.

    Returns dict with keys: 'passed' (bool), 'challenges' (list[str]),
    'risk_flags' (list[str]). If passed=False, the plan should be refined.
    """
    questions = plan_data.get("research_questions", [])
    if not questions:
        return {"passed": True, "challenges": [], "risk_flags": []}

    challenge_prompt = (
        "You are a Devil's Advocate for research planning. Review the proposed "
        "research questions and identify weaknesses. Generate exactly 3 challenge "
        "questions covering: (1) data-bias risk — could the evidence be skewed by "
        "source selection or recency bias?, (2) alternative academic views — what "
        "contrary position or framework is NOT being investigated?, (3) untested "
        "assumptions — which core assumption in the questions might be false?. "
        "Return ONLY valid JSON with keys: 'passed' (true if plan is robust, false "
        "if it needs refinement), 'challenges' (list of 3 strings), 'risk_flags' "
        "(list of specific weaknesses to address)."
    )

    messages = [
        SystemMessage(content=challenge_prompt),
        HumanMessage(
            content=(
                f"Query: {query}\n"
                f"Research Questions:\n"
                + "\n".join(f"- {q}" for q in questions)
            )
        ),
    ]

    try:
        raw = await invoke_messages(messages, max_tokens=300, temperature=0.4)
        result = safe_json(raw)
        passed = bool(result.get("passed", True))
        challenges = [str(c) for c in result.get("challenges", [])]
        risk_flags = [str(r) for r in result.get("risk_flags", [])]
        if not passed:
            logger.info(
                "Adversarial challenge: plan FAILED — risks=%s",
                ", ".join(risk_flags[:3]) if risk_flags else "none",
            )
        else:
            logger.info("Adversarial challenge: plan passed")
        return {"passed": passed, "challenges": challenges, "risk_flags": risk_flags}
    except Exception as exc:
        logger.warning("Adversarial challenge LLM call failed: %s", exc)
        return {"passed": True, "challenges": [], "risk_flags": []}


def _enforce_traceability(
    questions: list[str],
    topics: list[str],
) -> list[str]:
    """T17: Verify at least 2 of 3 refined questions share keywords with topics."""
    topic_words = set()
    for t in topics:
        topic_words.update(w.lower() for w in str(t).split() if len(w) > 3)
    if not topic_words:
        return questions
    traceable: list[str] = []
    for q in questions:
        q_words = set(w.lower() for w in q.split() if len(w) > 3)
        if len(q_words & topic_words) >= _MIN_KEYWORD_OVERLAP:
            traceable.append(q)
    return traceable if len(traceable) >= 2 else questions


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
                        f"Refine the research plan to address these risks:\n"
                        + "\n".join(f"- {r}" for r in challenge.get("risk_flags", [])[:5])
                        + f"\n\nOriginal plan questions to improve:\n"
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
