from __future__ import annotations

from langgraph.graph import END
from research_swarm.graph.state import ResearchState


def route_from_judge(state: ResearchState) -> str:
    """Conditional routing: continue searching if score is below threshold."""
    return END if state.judge_score >= 80 else "searcher"
