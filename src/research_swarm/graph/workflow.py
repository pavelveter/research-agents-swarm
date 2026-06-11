from __future__ import annotations

from langgraph.graph import END, StateGraph

from research_swarm.agents.fact_checker import fact_check
from research_swarm.agents.judge import judge
from research_swarm.agents.planner import plan
from research_swarm.agents.searcher import search
from research_swarm.agents.summarizer import summarize
from research_swarm.graph.routing import route_from_judge
from research_swarm.graph.state import ResearchState


def build_workflow() -> StateGraph:
    builder = StateGraph(ResearchState)

    builder.add_node("planner", plan)
    builder.add_node("searcher", search)
    builder.add_node("fact_checker", fact_check)
    builder.add_node("summarizer", summarize)
    builder.add_node("judge", judge)

    builder.add_edge("planner", "searcher")
    builder.add_edge("searcher", "fact_checker")
    builder.add_edge("fact_checker", "summarizer")
    builder.add_edge("summarizer", "judge")

    builder.add_conditional_edges("judge", route_from_judge)

    builder.set_entry_point("planner")
    return builder.compile()
