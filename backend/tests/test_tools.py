from __future__ import annotations

import sys
import types

import numpy as np

from backend.tools.hf_inference_tool import simplify_text
from backend.tools.speech_to_text_tool import transcribe_audio
from backend.tools.text_to_speech_tool import synthesize_speech
from backend.tools.vector_store import SchemeVectorStore
from backend.tools.web_search_tool import web_search


def test_vector_store_round_trip(tmp_path):
    store = SchemeVectorStore()
    ids = ["a", "b", "c"]
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]],
        dtype="float32",
    )
    metadatas = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
    store.build(ids, embeddings, metadatas)
    store.save(tmp_path)

    loaded = SchemeVectorStore()
    assert loaded.load(tmp_path) is True
    assert loaded.size == 3

    hits = loaded.search(np.array([1.0, 0.0], dtype="float32"), top_k=1)
    assert len(hits) == 1
    assert hits[0][0].metadata["name"] == "A"


def test_vector_store_load_missing_directory_returns_false(tmp_path):
    store = SchemeVectorStore()
    assert store.load(tmp_path / "does-not-exist") is False


def test_simplify_text_returns_none_without_hf_token():
    # conftest sets HF_API_TOKEN="" by default, so this must fail soft (no
    # network call, no exception) rather than raising.
    assert simplify_text("Explain this scheme simply.") is None


def test_transcribe_audio_returns_none_without_hf_token():
    # conftest sets HF_API_TOKEN="" by default, so this must fail soft (no
    # network call, no exception) rather than raising.
    assert transcribe_audio(b"fake-audio-bytes") is None


def test_transcribe_audio_returns_none_for_empty_bytes(monkeypatch):
    import types as _types

    import backend.tools.speech_to_text_tool as stt_module

    fake_settings = _types.SimpleNamespace(
        huggingface=_types.SimpleNamespace(
            api_key="fake-token",
            models=_types.SimpleNamespace(speech_to_text="fake/model"),
            inference_api_url="https://router.huggingface.co",
            timeout_seconds=10,
        )
    )
    monkeypatch.setattr(stt_module, "get_model_settings", lambda: fake_settings)
    assert transcribe_audio(b"") is None


def test_web_search_falls_back_to_empty_list_on_failure(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("network is unreachable in this sandbox")

    monkeypatch.setattr("backend.tools.web_search_tool._duckduckgo_search", _boom)
    results = web_search("pm kisan scheme")
    assert results == []


def test_web_search_parses_duckduckgo_results(monkeypatch):
    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, max_results):
            return [{"title": "PM-KISAN", "href": "https://pmkisan.gov.in", "body": "Farmer income support"}]

    fake_ddgs_module = types.ModuleType("ddgs")
    fake_ddgs_module.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_ddgs_module)

    results = web_search("pm kisan scheme", max_results=1)
    assert len(results) == 1
    assert results[0].url == "https://pmkisan.gov.in"


def test_synthesize_speech_returns_none_for_blank_text():
    assert synthesize_speech("   ", "en") is None


def test_synthesize_speech_falls_back_to_english_for_unsupported_language(monkeypatch):
    captured = {}

    class _FakeGTTS:
        def __init__(self, text, lang):
            captured["lang"] = lang

        def write_to_fp(self, buffer):
            buffer.write(b"fake-mp3-bytes")

    monkeypatch.setattr("backend.tools.text_to_speech_tool.gTTS", _FakeGTTS)
    result = synthesize_speech("Hello", "fr")
    assert result == (b"fake-mp3-bytes", "audio/mpeg")
    assert captured["lang"] == "en"


def test_synthesize_speech_returns_none_on_failure(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("network is unreachable in this sandbox")

    monkeypatch.setattr("backend.tools.text_to_speech_tool.gTTS", _boom)
    assert synthesize_speech("Hello", "en") is None
