# app/agents/planner.py
"""Planner Agent
Determines the nature of the user question and whether multiple searches are needed.
"""

from __future__ import annotations

from typing import Dict
from .state import AgentState


def planner_node(state: AgentState) -> AgentState:
    """Simple heuristic planner.
    - If the question contains conjunctions like "and", "or", or a list separated by commas,
      we assume multiple independent searches are required.
    - Otherwise a single search suffices.
    The planner does not modify the question; it only sets ``need_multiple``.
    """
    q = state.question.lower()
    if any(tok in q for tok in [" and ", " or ", ", ", " vs ", " versus "]):
        state.need_multiple = True
    else:
        state.need_multiple = False
    return state
