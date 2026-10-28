# app/agents/research.py
"""Research Agent
Performs a ChromaDB query and returns a list of RetrievalChunk objects.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from .state import AgentState, RetrievalChunk

# Lazy import to avoid import errors if dependencies are missing
def _load_corpus_module():
    """Import the retrieval module safely.
    Returns the module object so that its functions (e.g., ``query_index``) can be accessed.
    """
    try:
        import importlib
        module = importlib.import_module('app.retrieval.load_corpus')
        return module
    except Exception as e:
        raise RuntimeError(f"Failed to import retrieval module: {e}")


def research_node(state: AgentState) -> AgentState:
    """Execute a search against the local ChromaDB store.
    Uses ``query_index`` from ``app.retrieval.load_corpus``.
    The ``persist`` directory defaults to ``./chroma`` relative to the project root.
    """
    query = state.question
    persist_dir = Path("./chroma")
    try:
        load_corpus_mod = _load_corpus_module()
        results = load_corpus_mod.query_index(persist_dir, query)
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}", file=sys.stderr)
        state.retrieved = []
        return state

    chunks: List[RetrievalChunk] = []
    for r in results:
        chunk = RetrievalChunk(
            chunk_id=r["chunk_id"],
            text=r["text"],
            metadata=r["metadata"],
            distance=r["distance"],
        )
        chunks.append(chunk)
    state.retrieved = chunks
    return state
