"""Local, no-API-key embedding generation using an open-source Hugging Face
sentence-transformers model (downloaded once from the HF Hub and cached
locally thereafter -- no HF token required for this particular model)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.core.config import get_model_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    settings = get_model_settings()
    model_name = settings.huggingface.models.embeddings
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings.astype("float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
