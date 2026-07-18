"""Voice input widget: records speech (English/Hindi/Telugu), transcribes it
via the backend's Whisper-based /voice/transcribe endpoint, and pre-fills the
profile form + free-text box from the result.

Kept outside st.form (audio_input needs to process immediately on capture,
before the user submits the form) -- see profile_form.py for the shared
FIELD_KEYS / session_state contract this writes into.
"""
from __future__ import annotations

import hashlib

import streamlit as st

from components.api_client import BackendError, transcribe_voice
from components.profile_form import CATEGORIES, FIELD_KEYS, INDIAN_STATES, ensure_defaults

VALID_GENDERS = {"male", "female", "other"}


def _apply_suggested_profile(suggested: dict) -> None:
    """Only ever writes a field if the extracted value is safe for that
    widget's fixed options (selectboxes crash if given an out-of-list
    value) -- occupation is deliberately never auto-mapped for that reason
    and stays available via the free-text transcript instead."""
    if isinstance(suggested.get("age"), int):
        st.session_state[FIELD_KEYS["age"]] = max(0, min(120, suggested["age"]))
    if suggested.get("gender") in VALID_GENDERS:
        st.session_state[FIELD_KEYS["gender"]] = suggested["gender"]
    if isinstance(suggested.get("annual_income"), (int, float)) and suggested["annual_income"] >= 0:
        st.session_state[FIELD_KEYS["annual_income"]] = int(suggested["annual_income"])
    if isinstance(suggested.get("family_size"), int) and suggested["family_size"] >= 1:
        st.session_state[FIELD_KEYS["family_size"]] = suggested["family_size"]
    if suggested.get("state") in INDIAN_STATES:
        st.session_state[FIELD_KEYS["state"]] = suggested["state"]
    if suggested.get("category") in CATEGORIES:
        st.session_state[FIELD_KEYS["category"]] = suggested["category"]
    for flag in ("is_student", "is_disabled", "is_farmer", "is_bpl"):
        if isinstance(suggested.get(flag), bool):
            st.session_state[FIELD_KEYS[flag]] = suggested[flag]


def render_voice_input() -> None:
    ensure_defaults()
    st.markdown("**🎙️ Or speak your situation**")
    audio = st.audio_input("Tap to record — English, Hindi, or Telugu all work")
    if audio is None:
        return

    audio_bytes = audio.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    if st.session_state.get("_last_voice_hash") == audio_hash:
        return  # already processed this exact recording, avoid re-transcribing on every rerun

    with st.spinner("Transcribing your recording..."):
        try:
            result = transcribe_voice(audio_bytes, content_type=getattr(audio, "type", "audio/wav"))
        except BackendError as exc:
            st.error(str(exc))
            return

    st.session_state["_last_voice_hash"] = audio_hash
    st.session_state[FIELD_KEYS["free_text_context"]] = result.get("transcript", "")
    _apply_suggested_profile(result.get("suggested_profile", {}))
    st.success(f"Heard you: “{result.get('transcript', '')}”")
    st.rerun()
