from __future__ import annotations

import json
import sqlite3
from typing import Iterable, List, Optional

import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


class VectorStore:
    def __init__(self, db_path: str, embedding_model) -> None:
        self.db_path = db_path
        self.embedding_model = embedding_model
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                title TEXT,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                UNIQUE(url, chunk_index)
            )
            """
        )
        conn.commit()
        conn.close()

    def add_pages(self, pages: Iterable[dict]) -> None:
        for page in pages:
            text = page.get("text") or ""
            url = page.get("url") or ""
            title = page.get("title") or ""
            if not text or not url:
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            embeddings = self.embedding_model.encode(chunks)
            conn = self._connect()
            cur = conn.cursor()
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                emb_json = json.dumps(emb.tolist())
                cur.execute(
                    """
                    INSERT OR REPLACE INTO vector_chunks
                    (url, title, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (url, title, idx, chunk, emb_json),
                )
            conn.commit()
            conn.close()

    def query(
        self,
        query: str,
        top_k: int = 6,
        urls: Optional[List[str]] = None,
    ) -> List[dict]:
        if not query.strip():
            return []

        q_emb = self.embedding_model.encode(query)

        conn = self._connect()
        cur = conn.cursor()
        if urls:
            placeholders = ",".join("?" for _ in urls)
            cur.execute(
                f"""
                SELECT url, title, content, embedding
                FROM vector_chunks
                WHERE url IN ({placeholders})
                """,
                urls,
            )
        else:
            cur.execute(
                "SELECT url, title, content, embedding FROM vector_chunks"
            )
        rows = cur.fetchall()
        conn.close()

        scored = []
        for url, title, content, emb_json in rows:
            emb = np.array(json.loads(emb_json), dtype=float)
            score = _cosine_sim(q_emb, emb)
            scored.append(
                {
                    "url": url,
                    "title": title,
                    "content": content,
                    "score": score,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
