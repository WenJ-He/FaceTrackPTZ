"""Lightweight vector database using SQLite + in-memory cosine similarity.

Per 14.7: MVP must use local lightweight vector storage.
Allowed: in-memory vectors with cosine similarity, or SQLite with vector field + in-memory compute.
Prohibited: Milvus, FAISS IVF/PQ, distributed vector DBs.
"""

from __future__ import annotations

import os
import sqlite3
from typing import List, Optional, Tuple

import numpy as np

from .config import Config
from . import logger as log


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class VectorDB:
    """SQLite-backed face vector store with in-memory cosine search."""

    def __init__(self, config: Config) -> None:
        self._db_path = config.get("vector_db.path", "./data/face_vectors.db")
        self._top_k = config.get("recognition.top_k", 5)
        self._cache: dict[str, np.ndarray] = {}
        self._conn: Optional[sqlite3.Connection] = None

    def open(self) -> bool:
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                vector BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()
        self._load_cache()
        log.log(
            f"VectorDB opened: {len(self._cache)} identities loaded",
            result="OK",
        )
        return True

    def _load_cache(self) -> None:
        self._cache.clear()
        if self._conn is None:
            return
        cursor = self._conn.execute("SELECT name, vector FROM faces")
        for name, blob in cursor.fetchall():
            vector = np.frombuffer(blob, dtype=np.float32)
            self._cache[name] = vector.copy()

    def add_face(self, name: str, vector: np.ndarray) -> None:
        blob = vector.astype(np.float32).tobytes()
        if self._conn is None:
            raise RuntimeError("VectorDB not opened")
        self._conn.execute(
            "INSERT OR REPLACE INTO faces (name, vector) VALUES (?, ?)",
            (name, blob),
        )
        self._conn.commit()
        self._cache[name] = vector.astype(np.float32).copy()
        log.log(f"Added face vector: {name}", result="OK")

    def remove_face(self, name: str) -> None:
        if self._conn is None:
            return
        self._conn.execute("DELETE FROM faces WHERE name = ?", (name,))
        self._conn.commit()
        self._cache.pop(name, None)

    def search(self, query_vector: np.ndarray, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Search for the most similar faces. Returns [(name, similarity), ...]."""
        if not self._cache:
            return []

        k = top_k or self._top_k
        query = query_vector.flatten().astype(np.float32)

        results: List[Tuple[str, float]] = []
        for name, vec in self._cache.items():
            sim = _cosine_similarity(query, vec)
            results.append((name, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def list_identities(self) -> List[str]:
        return list(self._cache.keys())

    def count(self) -> int:
        return len(self._cache)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._cache.clear()

    @property
    def is_open(self) -> bool:
        return self._conn is not None
