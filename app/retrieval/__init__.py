import os
from pathlib import Path
from typing import List, Dict

from .load_corpus import build_index, query_index, load_corpus, validate_corpus

# Default directory for the Chroma vector store
DEFAULT_PERSIST_DIR = Path(__file__).parents[2] / "chroma"

def _ensure_index(persist_dir: Path = DEFAULT_PERSIST_DIR) -> None:
    """Ensure the vector index exists, building it if necessary.

    The function checks whether the Chroma collection directory exists. If not, it
    builds the index from the corpus file.
    """
    # Simple existence check: if the directory is empty, assume index not built
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        corpus_path = Path(__file__).parents[2] / "data" / "corpus.jsonl"
        print(f"Building vector index at {persist_dir} from {corpus_path}")
        build_index(corpus_path, persist_dir)
    else:
        # Optionally, could verify the collection exists, but Chroma will handle it.
        pass

def retrieve(query: str, top_k: int = 5, persist_dir: Path = DEFAULT_PERSIST_DIR) -> List[Dict]:
    """Retrieve the most relevant document chunks for a query.

    Parameters
    ----------
    query: str
        The user question or search string.
    top_k: int, default 5
        Number of chunks to return.
    persist_dir: Path, optional
        Directory where the Chroma DB is stored.

    Returns
    -------
    List[Dict]
        Each dict contains ``chunk_id``, ``text``, ``metadata`` and ``distance``.
    """
    if not query.strip():
        raise ValueError("Query string must not be empty.")
    _ensure_index(persist_dir)
    results = query_index(persist_dir, query, top_k=top_k)
    return results

def validate_and_load_corpus() -> List[Dict]:
    """Utility to load and validate the corpus, exposing it for other modules.
    """
    corpus_path = Path(__file__).parents[2] / "data" / "corpus.jsonl"
    records = load_corpus(corpus_path)
    valid, errors = validate_corpus(records)
    if not valid:
        raise ValueError("Corpus validation failed: " + "; ".join(errors))
    return records
