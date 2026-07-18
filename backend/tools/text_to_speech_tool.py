"""Text-to-speech for the "listen to this answer" feature.

This was originally built against open-source Hugging Face TTS models
(Meta's MMS-TTS, one model per language). While building it, live testing
against the HF Inference Providers router showed that no current provider
(free or paid) serves MMS-TTS, or any other checked open Hindi/Telugu-capable
TTS model, for this account -- HF's free "hf-inference" tier now serves ASR
and embeddings, but not general TTS (see ARCHITECTURE.md §6 for the full
trail). Rather than ship a feature that silently never works, this uses
gTTS (free, no API key, no account needed, actively supports en/hi/te) as the
practical engine today. It's isolated behind this module's synthesize_speech
interface, so swapping in an HF-hosted model later (once one is available)
is a one-file change -- nothing upstream needs to know how the audio is made.
"""
from __future__ import annotations

import io
import logging

from gtts import gTTS

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"en", "hi", "te"}


def synthesize_speech(text: str, language: str) -> tuple[bytes, str] | None:
    """Synthesizes spoken audio for `text` in the given language code
    (en/hi/te). Returns (audio_bytes, content_type), or None if the language
    isn't supported or synthesis fails for any reason."""
    if not text.strip():
        return None
    lang = language if language in SUPPORTED_LANGUAGES else "en"
    try:
        buffer = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buffer)
        return buffer.getvalue(), "audio/mpeg"
    except Exception:
        logger.warning("Text-to-speech synthesis failed for language=%s", lang, exc_info=True)
        return None
