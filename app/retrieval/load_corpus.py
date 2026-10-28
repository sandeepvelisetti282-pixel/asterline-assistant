import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

REQUIRED_FIELDS = [
    "chunk_id",
    "doc_id",
    "title",
    "category",
    "owner",
    "source_url",
    "published",
    "text",
]

def load_corpus(file_path: Path) -> List[Dict]:
    """Load the JSONL corpus file, returning a list of records.
    Empty lines are ignored.
    """
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {line_num}: {e}")
    return records

def validate_record(record: Dict, line_num: int) -> List[str]:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"Missing required field '{field}' (line {line_num})")
    return errors

def validate_corpus(records: List[Dict]) -> Tuple[bool, List[str]]:
    """Validate the corpus.
    Returns (is_valid, list_of_error_messages).
    Checks record count (expected 75), required fields, and duplicate chunk_id.
    """
    errors = []
    expected_count = 75
    if len(records) != expected_count:
        errors.append(f"Expected {expected_count} records, found {len(records)}")
    # Validate each record
    for i, rec in enumerate(records, start=1):
        errors.extend(validate_record(rec, i))
    # Duplicate chunk_id check
    chunk_ids = [rec.get("chunk_id") for rec in records]
    duplicates = {cid for cid in chunk_ids if chunk_ids.count(cid) > 1}
    if duplicates:
        errors.append(f"Duplicate chunk_id values found: {', '.join(sorted(duplicates))}")
    return (len(errors) == 0, errors)

def build_index(corpus_path: Path, persist_dir: Path) -> None:
    """Build a ChromaDB vector store from the corpus.
    Uses SentenceTransformer embeddings and stores metadata for retrieval.
    """
    records = load_corpus(corpus_path)
    valid, validation_errors = validate_corpus(records)
    if not valid:
        for err in validation_errors:
            print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)
    # Initialize embedding function with fallback
    if SentenceTransformer and embedding_functions and chromadb:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=str(persist_dir))
        collection = client.get_or_create_collection(name="asterline_corpus", embedding_function=embed_fn)
    else:
        # Fallback: use a dummy embedding function that returns zero vectors
        class DummyEmbeddingFunction:
            def __call__(self, texts):
                # Return a list of zero vectors (length 384) for each text
                return [[0.0] * 384 for _ in texts]
        embed_fn = DummyEmbeddingFunction()
        # Use in‑memory Chroma client if persistent client unavailable
        if chromadb:
            client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            raise RuntimeError("ChromaDB is not installed; cannot build index.")
        collection = client.get_or_create_collection(name="asterline_corpus", embedding_function=embed_fn)
    # Insert documents
    ids = []
    embeddings = []
    metadatas = []
    documents = []
    for rec in records:
        ids.append(rec["chunk_id"])
        documents.append(rec["text"])
        # Preserve all metadata fields for citation
        meta = {k: rec.get(k) for k in REQUIRED_FIELDS if k != "text"}
        metadatas.append(meta)
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Index built with {len(ids)} chunks and persisted to {persist_dir}")

def query_index(persist_dir: Path, query: str, top_k: int = 5) -> List[Dict]:
    """Search the index and return matching chunks with metadata."""
    if not chromadb:
        raise RuntimeError("ChromaDB is not installed; cannot query index.")
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name="asterline_corpus")
    results = collection.query(query_texts=[query], n_results=top_k)
    # results dict contains 'ids', 'documents', 'metadatas', 'distances'
    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build or query the Asterline corpus index.")
    parser.add_argument("--build", action="store_true", help="Build the index from corpus.jsonl")
    parser.add_argument("--query", type=str, help="Query string to search the index")
    parser.add_argument("--persist", type=str, default="./chroma", help="Directory to store ChromaDB data")
    args = parser.parse_args()
    persist_path = Path(args.persist)
    if args.build:
        corpus_path = Path(__file__).parents[2] / "data" / "corpus.jsonl"
        build_index(corpus_path, persist_path)
    elif args.query:
        results = query_index(persist_path, args.query)
        for res in results:
            print(f"Chunk ID: {res['chunk_id']}")
            print(f"Title: {res['metadata'].get('title')}")
            print(f"Score (distance): {res['distance']:.4f}")
            print(f"Snippet: {res['text'][:200]}...\n")
    else:
        parser.print_help()
