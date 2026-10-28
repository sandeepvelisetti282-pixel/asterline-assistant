# app/agents/synthesis.py
"""Synthesis Agent
Creates an answer from retrieved chunks and records citations.
"""

from __future__ import annotations

from typing import List

from .state import AgentState, RetrievalChunk


def synthesis_node(state: AgentState) -> AgentState:
    """Generate a human‑readable answer.
    * If no chunks were retrieved, respond with a fallback answer.
    * Otherwise concatenate the first few chunk texts and add citations.
    The citations are stored as a list of ``chunk_id`` strings in ``state.citations``.
    """
    if not state.retrieved:
        state.answer = "I could not find relevant information to answer the question."
        state.citations = []
        return state

    # Simple synthesis: take up to three top chunks and summarize.
    top_chunks: List[RetrievalChunk] = state.retrieved[:3]
    summary_parts: List[str] = []
    citations: List[str] = []
    for chunk in top_chunks:
        summary_parts.append(chunk.text.strip())
        citations.append(chunk.chunk_id)

    combined = " ".join(summary_parts)
    state.answer = f"Based on the retrieved documents, {combined}"
    state.citations = citations
    return state
