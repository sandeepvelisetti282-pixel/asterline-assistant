# Asterline AI Research Assistant – Design Document

## 1. Project Objective and Scope
The Asterline AI Research Assistant is a multi‑agent research assistant built with **LangGraph**, **LangChain**, **ChromaDB**, **SQLite**, and a **Gradio** UI. Its purpose is to answer user questions by retrieving relevant company documents, synthesizing answers, and providing citations, while maintaining per‑user memory and observability via **LangSmith** tracing.

## 2. Architecture Overview
```mermaid
flowchart TD
    subgraph UI[Gradio UI]
        UI_Research[Research Tab]
        UI_Memory[Memory Tab]
    end
    subgraph Workflow[LangGraph Workflow]
        Planner[Planner]
        Research[Research]
        Synthesis[Synthesis]
        Verification[Verification]
    end
    subgraph Retrieval[Retrieval Layer]
        Chroma[ChromaDB (vector store)]
        Embeddings[Local Sentence‑Transformers embeddings]
    end
    subgraph Memory[SQLite Memory Store]
        SQLite[SQLite DB]
    end
    subgraph LLM[LLM Provider]
        OpenAI[OpenAI / other LLM]
    end
    subgraph Tracing[LangSmith Tracing]
        Tracer[LangChainTracer]
    end

    UI_Research -->|question| Planner
    Planner -->|need_multiple flag| Research
    Research -->|retrieved chunks| Retrieval
    Retrieval --> Chroma & Embeddings
    Research -->|chunks| Synthesis
    Synthesis -->|answer| Verification
    Verification -->|final state| UI_Research
    UI_Memory -->|CRUD ops| SQLite
    Planner & Research & Synthesis & Verification -->|LLM calls| OpenAI
    OpenAI -->|observability| Tracer
    Tracer -->|logs| Tracing
```

## 3. Agent Responsibilities & LangGraph Flow
| Agent | Responsibility |
|-------|----------------|
| **Planner** | Analyzes the user question to decide whether a single or multiple document search is required (`need_multiple`). |
| **Research** | Performs similarity search against **ChromaDB** using the configured embeddings, returning a list of `RetrievalChunk` objects. |
| **Synthesis** | Generates a natural‑language answer using the retrieved chunks and the LLM. |
| **Verification** | Checks the answer for **conflict** (different chunks with the same title but contradictory content) or **insufficient evidence** (no relevant chunks). Sets `verification` status accordingly. |

The workflow is defined in `app/agents/workflow.py` and orchestrates these agents via LangGraph state (`AgentState`).

## 4. Document Ingestion, Embeddings & Retrieval
* Documents are stored in `data/corpus.jsonl` (each line contains `chunk_id`, `text`, `metadata`).
* During the initial setup a **Sentence‑Transformers** model (e.g., `all-MiniLM-L6-v2`) creates local embeddings.
* Embeddings and documents are persisted in **ChromaDB** (`app/retrieval/chroma.py`).
* Retrieval (`Research` agent) performs a similarity search and returns `RetrievalChunk` objects defined in `app/agents/state.py`.

## 5. Grounded Answers & Citations
* The `Synthesis` agent receives the retrieved chunks and prompts the LLM to produce an answer that **must reference** the source material.
* Citations are constructed as `[{chunk_id}] title` strings and stored in `AgentState.citations`.
* The UI displays citations alongside the answer (see `app/ui.py`).

## 6. Conflict & Insufficient Evidence Handling
* **Conflict**: If multiple retrieved chunks share the same `title` but contain differing `text`, the `Verification` agent marks `verification = "conflict"`.
* **Insufficient Evidence**: If the retrieval returns an empty list, the answer falls back to a generic message and `verification = "insufficient"`.
* These statuses are surfaced in the UI and used by the test suite (`tests/test_workflow.py`).

## 7. SQLite Memory Schema & User Isolation
The memory store (`app/memory/memory.py`) uses a single SQLite file (`memory.db`) with the following schema:
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
```
* **User Isolation** – Every CRUD operation filters by `user_id`; a user can only read/write their own rows.
* The schema supports per‑user CRUD via the `MemoryStore` static methods.

## 8. Separation of Personal Memory & Authoritative Evidence
* **Personal Memory** – Stored in SQLite, managed via the Memory Management tab in the UI. Used for user‑specific notes.
* **Authoritative Evidence** – Stored in ChromaDB, derived from the immutable `corpus.jsonl`. Retrieval never accesses personal memory, ensuring evidence remains company‑controlled.

## 9. LangSmith Tracing & Privacy Considerations
* Tracing is initialized in `app/tracing.py`:
  ```python
  from langchain_core.tracers.langchain import LangChainTracer
  tracer = LangChainTracer() if os.getenv("LANGSMITH_API_KEY") else None
  ```
* All LangChain components automatically emit spans to LangSmith when the tracer is active.
* **Privacy** – Tracing is only enabled when `LANGSMITH_API_KEY` is present. No user‑generated personal memory or raw question text is logged unless explicitly passed to the LLM (the LLM call itself is traced, but the content is subject to the same privacy policy as any LangChain usage).

## 10. Evaluation Metrics, Results & Representative Failures
* **Metrics** – The test suite validates:
  * Correct answer generation (`answer` not empty).
  * Presence of expected citations.
  * Accurate `verification` status (`verified`, `conflict`, `insufficient`).
* **Actual Results** – All tests in `tests/` pass (`pytest -q` reports 0 failures). Sample failures captured in the test suite include:
  * Conflict detection when two chunks share a title but differ in content.
  * Proper fallback when no documents are retrieved.
* **Improvement Demonstrated** – Adding the `Verification` agent reduced false‑positive confidence in answers and provided explicit conflict flags, which were previously missing.

## 11. Setup & Execution Assumptions
* Python 3.11+, virtual environment (`.venv`).
* Install dependencies: `pip install -r requirements.txt` (includes `langchain`, `langgraph`, `chromadb`, `sentence-transformers`, `gradio`, `langsmith`).
* Populate `.env` with required keys (`OPENAI_API_KEY`, `LANGSMITH_API_KEY`, etc.).
* Run the UI: `python -m app.ui` (or `python app/ui.py`).
* Tests: `pytest` (ensure the virtual environment is activated).

## 12. Limitations & Technical Trade‑offs
* **No authentication** – Users are identified by synthetic IDs; real‑world deployments would need auth.
* **Single‑process SQLite** – Suitable for prototyping but not for high‑concurrency workloads.
* **Local embeddings** – Faster for small corpora but may not scale to millions of documents.
* **Tracing optional** – If `LANGSMITH_API_KEY` is missing, no observability is provided.
* **Gradio UI** – Simple demo UI; a production UI would require a more robust front‑end framework.

---
*Document generated on 2026‑10‑28, reflecting the current codebase.*
