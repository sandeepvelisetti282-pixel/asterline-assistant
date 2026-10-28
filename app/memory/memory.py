# app/memory/memory.py
"""Memory module for Asterline Assistant.

Database location is configured via the MEMORY_DB_PATH environment variable.
If unset, defaults to data/memory.db inside the project root.
All operations enforce strict user isolation.
"""

import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _resolve_db_path() -> Path:
    env_val = os.getenv("MEMORY_DB_PATH", "").strip()
    if env_val:
        p = Path(env_val)
    else:
        project_root = Path(__file__).resolve().parents[2]
        p = project_root / "data" / "memory.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


_DB_PATH: Path = _resolve_db_path()
_LOCK = threading.Lock()


def _init_db(db_path: Path = None) -> None:
    path = db_path or _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        conn = sqlite3.connect(str(path))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   TEXT    NOT NULL,
                    content   TEXT    NOT NULL,
                    timestamp TEXT    NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


_init_db(_DB_PATH)


class MemoryStore:
    """CRUD helper for per-user memories stored in SQLite."""

    @staticmethod
    def _connect(db_path: Path = None) -> sqlite3.Connection:
        path = db_path or _DB_PATH
        return sqlite3.connect(str(path), detect_types=sqlite3.PARSE_DECLTYPES)

    @staticmethod
    def add_memory(user_id: str, content: str, db_path: Path = None) -> None:
        if not content.strip():
            raise ValueError("Memory content cannot be empty.")
        lower = content.lower()
        if any(kw in lower for kw in ("api_key", "api key", "password", "secret", "token")):
            raise ValueError("Refusing to store a potential secret as a memory.")
        ts = datetime.utcnow().isoformat()
        with _LOCK:
            with MemoryStore._connect(db_path) as conn:
                conn.execute(
                    "INSERT INTO memories (user_id, content, timestamp) VALUES (?, ?, ?)",
                    (user_id, content, ts),
                )
                conn.commit()

    @staticmethod
    def list_memories(user_id: str, db_path: Path = None) -> List[Tuple[int, str, str]]:
        with _LOCK:
            with MemoryStore._connect(db_path) as conn:
                cur = conn.execute(
                    "SELECT id, content, timestamp FROM memories WHERE user_id = ? ORDER BY id",
                    (user_id,),
                )
                return [(row[0], row[1], row[2]) for row in cur.fetchall()]

    @staticmethod
    def update_memory(user_id: str, memory_id: int, new_content: str, db_path: Path = None) -> bool:
        if not new_content.strip():
            raise ValueError("New content cannot be empty.")
        ts = datetime.utcnow().isoformat()
        with _LOCK:
            with MemoryStore._connect(db_path) as conn:
                cur = conn.execute(
                    "UPDATE memories SET content = ?, timestamp = ? WHERE id = ? AND user_id = ?",
                    (new_content, ts, memory_id, user_id),
                )
                conn.commit()
                return cur.rowcount > 0

    @staticmethod
    def delete_memory(user_id: str, memory_id: int, db_path: Path = None) -> bool:
        with _LOCK:
            with MemoryStore._connect(db_path) as conn:
                cur = conn.execute(
                    "DELETE FROM memories WHERE id = ? AND user_id = ?",
                    (memory_id, user_id),
                )
                conn.commit()
                return cur.rowcount > 0

    @staticmethod
    def delete_all(user_id: str, db_path: Path = None) -> int:
        with _LOCK:
            with MemoryStore._connect(db_path) as conn:
                cur = conn.execute(
                    "DELETE FROM memories WHERE user_id = ?",
                    (user_id,),
                )
                conn.commit()
                return cur.rowcount
