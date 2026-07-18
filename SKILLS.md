# SKILLS.md

What each agent and tool in the navigator's multi-agent system does: its role,
model, inputs/outputs, and fallback behavior. Companion to
[ARCHITECTURE.md](ARCHITECTURE.md) (system design) and [CLAUDE.md](CLAUDE.md)
(dev conventions). For repo-maintenance skills invokable from Claude Code
itself (not the navigator's own agents), see `.claude/skills/`.

## Agents (LangGraph nodes, `backend/agents/`)

Executed in this order for every `/api/v1/navigate` request:
`intake → retrieval → eligibility → document_guidance → simplification → traceability`.

### 1. Intake Agent
**File:** `intake_agent.py` · **Model:** Claude (`anthropic.models.intake`, default
`claude-haiku-4-5-20251001`)

- **Input:** the structured `CitizenProfile` from the form + optional free-text context
  the citizen typed in their own words.
- **Output:** `profile_summary` — a 3-5 sentence natural-language summary used as context
  by every downstream agent's prompt.
- **Fallback:** if the Claude call fails, builds a templated summary directly from the
  structured fields (no free-text reconciliation, but never blocks the pipeline).

### 2. Retrieval Agent
**File:** `retrieval_agent.py` · **Model:** none (calls tools directly, no LLM reasoning)

- **Input:** `profile_summary` + profile flags (farmer/student/disabled/BPL).
- **Tools used:** `kb_search` (local knowledge base) always; `web_search` only if the KB
  returns fewer than 3 candidates.
- **Output:** `candidates: list[SchemeCandidate]`, each carrying a `source` (knowledge_base
  or web_search) with a real URL.
- **Fallback:** if web search fails, appends a warning and continues with whatever the KB
  found rather than failing the request.

### 3. Eligibility Reasoning Agent
**File:** `eligibility_agent.py` · **Model:** Claude (`anthropic.models.eligibility_reasoning`,
default `claude-sonnet-5`)

- **Input:** profile + profile_summary + all candidate schemes' eligibility rules.
- **Output:** `eligibility_results: list[EligibilityResult]` — one verdict per scheme:
  `Likely Eligible / Possibly Eligible / Not Eligible / Needs More Info`, each with a
  plain-language `reason` and a `confidence` (0-1).
- **Fallback:** if the Claude call fails or returns unparseable JSON, every candidate gets a
  `Needs More Info` / confidence 0.0 result with a message to verify manually.

### 4. Document Guidance Agent
**File:** `document_agent.py` · **Model:** Claude (`anthropic.models.document_guidance`,
default `claude-sonnet-5`)

- **Input:** candidates with a non-"Not Eligible" verdict.
- **Output:** `document_guidance: dict[scheme_id, DocumentGuidance]` — required documents,
  ordered application steps, common rejection/delay reasons.
- **Fallback:** falls back to the KB's known `required_documents` list (no generated steps
  or blockers) if the Claude call fails.

### 5. Language Simplification Agent
**File:** `simplification_agent.py` · **Primary model:** open-source Hugging Face
instruction-tuned model (`huggingface.models.simplification`, default
`Qwen/Qwen2.5-7B-Instruct`, via HF Inference API) · **Fallback model:** Claude
(`anthropic.models.simplification_fallback`)

- **Input:** each relevant scheme's benefits + eligibility reason + documents, and the
  citizen's selected language (English/Hindi/Telugu).
- **Output:** `simplified_explanations: dict[scheme_id, str]` — a 3-4 sentence, plain-language,
  localized explanation.
- **Fallback chain:** no `HF_API_TOKEN` configured, or the HF call fails/rate-limits →
  automatically retries with the Claude fallback model → if that also fails, falls back to
  the raw eligibility `reason` text. This agent is the one that satisfies the "use an
  open-source Hugging Face model" requirement for a reasoning agent (embeddings are the other).

### 6. Traceability Agent
**File:** `traceability_agent.py` · **Model:** none (deterministic)

- **Input:** all prior agent outputs.
- **Output:** `recommendations: list[SchemeRecommendation]` — the final API response payload,
  sorted by verdict priority then confidence, each one carrying its `source` (name, url,
  origin) so every claim is traceable back to where it came from.

## Tools (`backend/tools/`)

| Tool | File | Purpose | Provider / fallback |
|---|---|---|---|
| `kb_search` | `kb_search_tool.py` | Vector similarity search over the local scheme knowledge base | Local numpy cosine-similarity index (`vector_store.py`) over `sentence-transformers` embeddings — no API key |
| `web_search` / `web_search_tool` | `web_search_tool.py` | Free internet search for schemes not in the KB | DuckDuckGo (`ddgs`, zero API key) by default; switch to Tavily or SerpAPI via `config/models.yaml` `search.provider` + the matching key |
| `simplify_text` | `hf_inference_tool.py` | Calls the open-source HF model for plain-language rewriting | HF Inference API; returns `None` (triggering the Claude fallback) if no token or on error |
| `embed_texts` / `embed_query` | `embeddings.py` | Turns scheme text / search queries into vectors | `sentence-transformers/all-MiniLM-L6-v2`, runs locally, no API key |
| `SchemeVectorStore` | `vector_store.py` | Persists/queries the embedding index | In-process numpy index; swappable for a managed vector DB |
| `transcribe_audio` | `speech_to_text_tool.py` | Speech-to-text for the "speak your situation" voice input feature | HF Inference API, open-source Whisper (`huggingface.models.speech_to_text`); no fallback — Claude can't accept raw audio, so this returns `None` (→ 503) if HF isn't configured |
| `synthesize_speech` | `text_to_speech_tool.py` | Text-to-speech for the "listen to this answer" feature | gTTS (free, no API key). Originally built against open-source HF TTS models (Meta MMS-TTS); live testing showed no provider currently serves them, so this uses gTTS instead — see ARCHITECTURE.md §6. Returns `None` (→ 503) on failure or an unsupported language |

## Voice input/output (`backend/api/routes_voice.py`)

Two endpoints, both HF-only (no Claude fallback exists for audio):

- **`POST /api/v1/voice/transcribe`** — accepts a recorded audio file (multipart), transcribes it
  with `transcribe_audio`, then makes a best-effort attempt to lift structured `CitizenProfile`
  fields (age, gender, income, state, category, farmer/student/disabled/BPL flags, family size)
  out of the transcript using Claude (`backend/services/voice_intake_service.py`,
  `extract_profile_fields()` — reuses the `intake` model, returns `{}` on any failure rather than
  blocking the transcript). Returns `{transcript, suggested_profile}`.
- **`POST /api/v1/voice/speak`** — accepts `{text, language}`, returns raw audio bytes
  (`audio/flac`) via `synthesize_speech`.

The Streamlit UI wires these into `frontend/components/voice_input.py` (records via
`st.audio_input`, transcribes, then pre-fills the profile form's session-state-keyed widgets —
see `FIELD_KEYS` in `profile_form.py`; only values that are safe for a given widget's fixed
options are ever auto-applied, occupation is deliberately left to the transcript instead of being
force-mapped onto the dropdown) and `results_view.py` (a "🔊 Listen" button per recommendation
that calls `/voice/speak` on that scheme's `simplified_explanation`).

## LLM clients (`backend/llm/`)

| Client | Purpose |
|---|---|
| `anthropic_client.py` | The only module that calls the Anthropic SDK. `complete()` for free-text, `complete_json()` for structured JSON-mode style calls (used by eligibility + document guidance agents, and by `voice_intake_service.extract_profile_fields`). Retries with exponential backoff via `tenacity`. |
