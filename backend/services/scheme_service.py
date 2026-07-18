"""Loads the seed welfare-scheme dataset and manages the local vector index
that the Retrieval Agent's kb_search tool queries."""
from __future__ import annotations

import json
import logging
from functools import lru_cache

from backend.core.config import get_model_settings
from backend.tools.embeddings import embed_query, embed_texts
from backend.tools.vector_store import SchemeVectorStore

logger = logging.getLogger(__name__)


def scheme_to_corpus_text(scheme: dict) -> str:
    """Builds the text blob embedded for a scheme. Shared by the service
    (incremental load) and scripts/build_vector_index.py (explicit rebuild)."""
    elig = scheme.get("eligibility", {}) or {}
    return (
        f"{scheme['name']} ({scheme.get('short_name', '')}). "
        f"Category: {scheme.get('category', '')}. {scheme.get('description', '')} "
        f"Benefits: {scheme.get('benefits', '')} "
        f"Eligibility notes: {elig.get('additional_notes', '')}"
    )


def load_seed_schemes(seed_path) -> list[dict]:
    with open(seed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("schemes", [])


class SchemeService:
    """Singleton-ish facade over the scheme knowledge base: raw records +
    vector index. Instantiate via get_scheme_service()."""

    def __init__(self) -> None:
        self._settings = get_model_settings()
        self._schemes_by_id: dict[str, dict] = {}
        self._store = SchemeVectorStore()
        self._load()

    def _load(self) -> None:
        schemes = load_seed_schemes(self._settings.app.seed_data_file)
        self._schemes_by_id = {s["id"]: s for s in schemes}

        store_dir = self._settings.app.vector_store_dir
        if self._store.load(store_dir) and self._store.size == len(schemes):
            logger.info("Loaded existing vector index (%d schemes) from %s", self._store.size, store_dir)
            return

        logger.info("Building vector index for %d schemes with model %s", len(schemes), self._settings.huggingface.models.embeddings)
        ids = [s["id"] for s in schemes]
        texts = [scheme_to_corpus_text(s) for s in schemes]
        embeddings = embed_texts(texts)
        self._store.build(ids=ids, embeddings=embeddings, metadatas=schemes)
        self._store.save(store_dir)
        logger.info("Vector index built and saved to %s", store_dir)

    @property
    def is_loaded(self) -> bool:
        return self._store.size > 0

    @property
    def scheme_count(self) -> int:
        return self._store.size

    def get_scheme(self, scheme_id: str) -> dict | None:
        return self._schemes_by_id.get(scheme_id)

    def list_schemes(self) -> list[dict]:
        return list(self._schemes_by_id.values())

    def search(self, query: str, top_k: int = 8) -> list[tuple[dict, float]]:
        if not query.strip():
            return []
        query_embedding = embed_query(query)
        hits = self._store.search(query_embedding, top_k=top_k)
        return [(hit.metadata, score) for hit, score in hits]


@lru_cache(maxsize=1)
def get_scheme_service() -> SchemeService:
    return SchemeService()
