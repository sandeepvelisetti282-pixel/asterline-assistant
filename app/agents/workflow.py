# app/agents/workflow.py
"""LangGraph workflow that orchestrates Planner, Research, Synthesis, and Verification agents.
The workflow operates on the shared :class:`AgentState` model defined in ``state.py``.
"""

from __future__ import annotations

from pathlib import Path

# Optional LangSmith tracing – if the SDK is unavailable we fall back to a no‑op wrapper
try:
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(x):
        return x

from langgraph.graph import StateGraph, END

from .state import AgentState
from .planner import planner_node
from .research import research_node
from .synthesis import synthesis_node
from .verification import verification_node


MAX_RETRIES = 2


def _route_after_verification(state: AgentState) -> str:
    """Conditional edge: retry research+synthesis if verification failed and
    retry budget remains; otherwise exit."""
    if state.verification in ("insufficient", "conflict") and state.retries < MAX_RETRIES:
        return "research"  # loop back
    return END


def build_workflow() -> StateGraph:
    """Create and compile the LangGraph StateGraph.
    Normal flow: planner -> research -> synthesis -> verification
    On failure with retries remaining: verification loops back to research.
    """
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("verification", verification_node)

    graph.add_edge("planner", "research")
    graph.add_edge("research", "synthesis")
    graph.add_edge("synthesis", "verification")
    graph.add_conditional_edges("verification", _route_after_verification)

    graph.set_entry_point("planner")
    return graph

# Compile once for reuse
_GRAPH = build_workflow().compile()


def run_workflow(question: str, persist_dir: Path | None = None) -> AgentState:
    """Run the multi‑agent workflow for a single question.

    Args:
        question: User query string.
        persist_dir: Optional path to the ChromaDB persistence directory.
            If ``None`` the default ``./chroma`` will be used by the research
            node.
    Returns:
        Final ``AgentState`` after verification.
    """
    init_state = AgentState(question=question)
    # LangGraph invoke() returns a plain dict; convert back to AgentState.
    raw: dict = _GRAPH.invoke(init_state.model_dump())
    return AgentState(**raw)
