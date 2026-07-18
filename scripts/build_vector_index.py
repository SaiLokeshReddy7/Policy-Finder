"""Rebuilds the local scheme vector index from data/schemes/seed_schemes.json.

Run this after editing the seed dataset (e.g. after using the add-scheme
Claude Code skill), or any time you want to force a fresh index:

    python scripts/build_vector_index.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import get_model_settings  # noqa: E402
from backend.services.scheme_service import load_seed_schemes, scheme_to_corpus_text  # noqa: E402
from backend.tools.embeddings import embed_texts  # noqa: E402
from backend.tools.vector_store import SchemeVectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_vector_index")


def main() -> None:
    settings = get_model_settings()
    schemes = load_seed_schemes(settings.app.seed_data_file)
    logger.info("Embedding %d schemes with %s", len(schemes), settings.huggingface.models.embeddings)

    ids = [s["id"] for s in schemes]
    texts = [scheme_to_corpus_text(s) for s in schemes]
    embeddings = embed_texts(texts)

    store = SchemeVectorStore()
    store.build(ids=ids, embeddings=embeddings, metadatas=schemes)
    store.save(settings.app.vector_store_dir)
    logger.info("Vector index saved to %s", settings.app.vector_store_dir)


if __name__ == "__main__":
    main()
