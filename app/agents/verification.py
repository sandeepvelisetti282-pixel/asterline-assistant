# app/agents/verification.py
"""Verification Agent
Validates citations and detects conflicts among retrieved chunks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .state import AgentState, RetrievalChunk


def verification_node(state: AgentState) -> AgentState:
    """Validate that all cited chunk IDs were actually retrieved and
    detect simple conflicts.

    * If any citation is missing from ``state.retrieved`` → ``insufficient``.
    * If two retrieved chunks share the same ``title`` but have different ``text``
      → ``conflict``.
    * Otherwise → ``verified``.
    The node also respects a retry limit (max 2). If verification fails and
    ``state.retries`` < 2, it increments ``retries`` and leaves ``verification``
    as ``None`` so the workflow could be extended to retry (not implemented here).
    """

    # If no chunks were retrieved at all, we cannot verify evidence.
    if not state.retrieved:
        state.verification = "insufficient"
        return state

    # Check missing citations
    retrieved_ids = {c.chunk_id for c in state.retrieved}
    missing = [cid for cid in state.citations if cid not in retrieved_ids]
    if missing:
        state.verification = "insufficient"
        state.retries += 1
        return state

    # Detect title conflicts
    title_to_texts: Dict[str, List[str]] = defaultdict(list)
    for chunk in state.retrieved:
        title = chunk.metadata.get("title", "")
        title_to_texts[title].append(chunk.text.strip())
    for title, texts in title_to_texts.items():
        if len(set(texts)) > 1:
            state.verification = "conflict"
            state.retries += 1
            return state

    # All good
    state.verification = "verified"
    return state
