"""Structured (JSON-line) logging setup for the backend process."""
from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Keep noisy third-party libraries at WARNING unless the whole app is in DEBUG.
    if level > logging.DEBUG:
        for noisy_logger in ("httpx", "httpcore", "sentence_transformers", "urllib3"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)
