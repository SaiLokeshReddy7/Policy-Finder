"""Shared slowapi Limiter instance (importable from both main.py and route
modules without circular imports)."""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
