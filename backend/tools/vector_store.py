"""Lightweight in-process cosine-similarity vector index.

Chosen over a full vector database (Chroma/FAISS/pgvector) to keep the
project dependency-light and trivially runnable on any machine, including
plain Windows without native build tools. Callers only depend on `build`,
`save`, `load`, and `search`, so this class can be swapped for a managed
vector DB in production without touching any agent or tool code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VectorRecord:
    id: str
    metadata: dict


class SchemeVectorStore:
    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._records: list[VectorRecord] = []

    def build(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict]) -> None:
        self._embeddings = embeddings.astype("float32")
        self._records = [VectorRecord(id=i, metadata=m) for i, m in zip(ids, metadatas)]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._embeddings is not None:
            np.save(directory / "embeddings.npy", self._embeddings)
        with open(directory / "records.json", "w", encoding="utf-8") as f:
            json.dump([{"id": r.id, "metadata": r.metadata} for r in self._records], f)

    def load(self, directory: Path) -> bool:
        emb_path = directory / "embeddings.npy"
        rec_path = directory / "records.json"
        if not emb_path.exists() or not rec_path.exists():
            return False
        self._embeddings = np.load(emb_path)
        with open(rec_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._records = [VectorRecord(id=r["id"], metadata=r["metadata"]) for r in raw]
        return True

    @property
    def size(self) -> int:
        return len(self._records)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[VectorRecord, float]]:
        if self._embeddings is None or len(self._records) == 0:
            return []
        scores = self._embeddings @ query_embedding
        k = min(top_k, len(self._records))
        top_indices = np.argsort(-scores)[:k]
        return [(self._records[i], float(scores[i])) for i in top_indices]
