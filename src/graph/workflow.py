from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.fact_checker import fact_check
from agents.judge import judge
from agents.planner import plan
from agents.searcher import search
from agents.summarizer import summarize
from graph.routing import route_from_judge, route_node
from graph.state import ResearchState


def build_workflow() -> CompiledStateGraph[ResearchState, None, ResearchState, ResearchState]:
    builder = StateGraph(ResearchState)

    builder.add_node("planner", plan)
    builder.add_node("searcher", search)
    builder.add_node("fact_checker", fact_check)
    builder.add_node("summarizer", summarize)
    builder.add_node("judge", judge)
    builder.add_node("routing", route_node)

    builder.add_edge("planner", "searcher")
    builder.add_edge("searcher", "fact_checker")
    builder.add_edge("fact_checker", "summarizer")
    builder.add_edge("summarizer", "judge")
    builder.add_edge("judge", "routing")

    builder.add_conditional_edges(
        "routing",
        route_from_judge,
        {"searcher": "searcher", "__end__": END},
    )

    builder.set_entry_point("planner")
    return builder.compile()
