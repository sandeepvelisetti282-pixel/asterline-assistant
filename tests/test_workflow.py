# tests/test_workflow.py
"""Tests for the multi‑agent LangGraph workflow.
We monkey‑patch the research node so the tests run without needing a real
ChromaDB index.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.agents.state import RetrievalChunk, AgentState
from app.agents import workflow

# Helper to create a dummy RetrievalChunk
def make_chunk(chunk_id: str, text: str, title: str, distance: float = 0.0):
    return RetrievalChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"title": title},
        distance=distance,
    )

# ---------------------------------------------------------------------------
# Fixture that patches the research node for each test
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_research(monkeypatch):
    def fake_research(state: AgentState) -> AgentState:
        # By default return a single generic chunk; individual tests can
        # replace ``state.retrieved`` after calling this function.
        state.retrieved = []
        return state
    monkeypatch.setattr("app.agents.research.research_node", fake_research)
    yield

# ---------------------------------------------------------------------------
# 1️⃣ Direct question – single search
# ---------------------------------------------------------------------------
def test_direct_question(monkeypatch):
    # Prepare a stub that returns one chunk
    def stub(state):
        state.retrieved = [make_chunk("c1", "Pricing is tiered.", "Pricing")]
        return state
    monkeypatch.setattr("app.agents.research.research_node", stub)

    result = workflow.run_workflow("What is the pricing model?")
    assert result.answer is not None
    assert "Pricing is tiered" in result.answer
    assert result.citations == ["c1"]
    assert result.verification == "verified"

# ---------------------------------------------------------------------------
# 2️⃣ Multi‑document question – planner should set need_multiple True
# ---------------------------------------------------------------------------
def test_multi_document_question(monkeypatch):
    def stub(state):
        state.retrieved = [
            make_chunk("c1", "Pricing is tiered.", "Pricing"),
            make_chunk("c2", "SLA guarantees 99.9% uptime.", "SLA"),
        ]
        return state
    monkeypatch.setattr("app.agents.research.research_node", stub)

    result = workflow.run_workflow("What is the pricing model and SLA?")
    # Planner heuristic sets need_multiple True because of "and"
    assert result.need_multiple is True
    assert len(result.citations) == 2
    assert result.verification == "verified"

# ---------------------------------------------------------------------------
# 3️⃣ Conflicting documents – same title, different text
# ---------------------------------------------------------------------------
def test_conflict_detection(monkeypatch):
    def stub(state):
        state.retrieved = [
            make_chunk("c1", "Version A of the policy.", "Policy"),
            make_chunk("c2", "Version B of the policy.", "Policy"),
        ]
        return state
    monkeypatch.setattr("app.agents.research.research_node", stub)

    result = workflow.run_workflow("Explain the policy.")
    assert result.verification == "conflict"

# ---------------------------------------------------------------------------
# 4️⃣ No relevant documents – fallback answer
# ---------------------------------------------------------------------------
def test_no_documents(monkeypatch):
    def stub(state):
        state.retrieved = []
        return state
    monkeypatch.setattr("app.agents.research.research_node", stub)

    result = workflow.run_workflow("What is the unknown term?")
    assert "could not find" in result.answer.lower()
    # With no citations the verification will still be "verified" (no conflict)
    assert result.verification == "verified"
