# AI Citizen Scheme & Support Navigator

A multi-agent public-service navigator that turns fragmented Indian government
welfare-scheme information into traceable, personalized, and easy-to-act-on
guidance for citizens, NGO volunteers, and CSC/field-worker operators.

- **Backend:** FastAPI + LangGraph multi-agent pipeline
- **Frontend:** Streamlit (English UI) with a follow-up chat: after getting recommendations,
  ask questions like "Why am I not eligible?" and get answers grounded in your own results
- **Core reasoning agents:** Claude (Anthropic API)
- **Language simplification agent:** open-source Hugging Face model (with a Claude fallback)
- **Voice input:** speak your situation (English/Hindi/Telugu) via open-source Whisper
  (speech-to-text, no Claude fallback exists since Claude doesn't accept audio)
- **Voice output:** listen to any recommendation read back via gTTS text-to-speech (see
  ARCHITECTURE.md §6 for why this uses gTTS rather than an HF-hosted TTS model)
- **Retrieval:** local knowledge base (21 real Indian schemes) + free DuckDuckGo web search fallback
- **Every recommendation is traceable** back to its official `.gov.in` source or the web result it came from

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design, [SKILLS.md](SKILLS.md)
for what each agent/tool does, and [CLAUDE.md](CLAUDE.md) for developer/AI-assistant
conventions.

## 1. Prerequisites

- Python 3.11+ (3.11-3.13 recommended; project also runs on 3.14)
- An [Anthropic API key](https://console.anthropic.com/) (required — powers the core agents)
- A [Hugging Face access token](https://huggingface.co/settings/tokens) (optional for text —
  enables the open-source simplification model, falling back to Claude without it; **required**
  for voice *input* (speech-to-text), since Claude has no audio fallback. Voice *output*
  (listening to a recommendation) uses gTTS and needs no token at all.)

## 2. Setup

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env    # then edit .env and add your keys

# macOS/Linux/Git Bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # then edit .env and add your keys
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
HF_API_TOKEN=hf_...            # optional
```

Model names, prompts-routing, and provider selection live in
[config/models.yaml](config/models.yaml) — edit that file to change which Claude/HF
models each agent uses, or to switch the free web search provider. `api_key` fields
there accept either an `${ENV_VAR}` placeholder (recommended) or a literal key.

## 3. Build the vector index (first run only)

```bash
python scripts/build_vector_index.py
```

This downloads the open-source `sentence-transformers/all-MiniLM-L6-v2` embedding
model (once, cached locally by Hugging Face) and embeds the seed scheme dataset in
`data/schemes/seed_schemes.json` into `data/vectorstore/`. The backend also runs this
automatically on startup if no index is found yet.

## 4. Run locally

One terminal is all you need — the backend also serves the web UI:

```bash
uvicorn backend.main:app --reload --port 8000
```

Open **http://localhost:8000** (Tailwind single-page UI in `frontend/web/index.html`,
no build step). Backend health at http://localhost:8000/api/v1/health, interactive API
docs at http://localhost:8000/docs.

The older Streamlit UI still works if you prefer it:
`streamlit run frontend/streamlit_app.py` → http://localhost:8501.

## 5. Voice input/output

With `HF_API_TOKEN` set, the sidebar has a "🎙️ Or speak your situation" recorder above the
profile form: record in English, Hindi, or Telugu, and it transcribes your recording (open-source
Whisper) and pre-fills whatever profile fields it can confidently pick out (age, income, state,
category, farmer/student/disabled/BPL flags) — the transcript itself always lands in the
free-text box, and you can review/edit everything before submitting. Without `HF_API_TOKEN`,
voice input returns a clear "unavailable" message instead of failing silently — there's no
Claude fallback, since Claude doesn't accept audio.

Each recommendation also gets a "🔊 Listen" button that reads its plain-language explanation back
out loud via gTTS (free, no API key needed — see ARCHITECTURE.md §6 for why this isn't an
HF-hosted model), so this one works even without `HF_API_TOKEN` configured.

## 6. Run with Docker

```bash
docker compose up --build
```

Backend: http://localhost:8000 · Frontend: http://localhost:8501

## 7. Run tests

```bash
pytest -q
```

The full test suite (26+ tests) mocks every external call (Anthropic, HF Inference — including
Whisper — gTTS, DuckDuckGo, and the embedding model itself), so it runs offline and never
touches your API keys or the real `data/vectorstore/`.

## 8. Project structure

```
backend/
  agents/     LangGraph state + the six agents (intake, retrieval, eligibility,
              document guidance, simplification, traceability)
  tools/      kb_search, web_search, hf_inference, embeddings, vector_store,
              speech_to_text_tool (Whisper via HF), text_to_speech_tool (gTTS)
  llm/        Anthropic client wrapper
  api/        FastAPI routes (navigate, schemes, health, voice)
  core/       config loader, logging, exception handlers, rate limiter
  services/   scheme_service (knowledge base + vector index),
              voice_intake_service (Claude-based profile extraction from a transcript)
  models/     Pydantic request/response/domain schemas
  tests/      pytest suite (fully mocked, no network calls)
frontend/     Streamlit UI (profile form, voice input, results view with "Listen" buttons)
data/schemes/ seed_schemes.json — the local welfare-scheme knowledge base
config/       models.yaml — models, providers, and API keys
scripts/      build_vector_index.py
.claude/      CLAUDE.md-adjacent Claude Code skills for maintaining this repo
```

## 9. Adding or updating schemes

Edit `data/schemes/seed_schemes.json` and rerun `python scripts/build_vector_index.py`.
See [SKILLS.md](SKILLS.md) and the `.claude/skills/add-scheme` and
`.claude/skills/rebuild-index` Claude Code skills for a guided workflow.

## 10. Troubleshooting

- **"ANTHROPIC_API_KEY is not configured"** — set it in `.env`; `config/models.yaml`
  references it as `${ANTHROPIC_API_KEY}`.
- **Simplification always uses Claude, never the HF model** — set `HF_API_TOKEN` in
  `.env`; the app falls back silently by design when it's missing.
- **Voice recorder says unavailable (503)** — set `HF_API_TOKEN` in `.env`; speech-to-text has
  no Claude fallback. The "🔊 Listen" button uses gTTS and needs no token; if it fails, it's a
  transient gTTS/network issue — try again.
- **`/api/v1/navigate` returns 502** — check the backend logs; almost always a missing/
  invalid `ANTHROPIC_API_KEY` or an Anthropic rate limit.
- **Streamlit can't reach the backend** — set `BACKEND_BASE_URL` in `.env` (defaults to
  `http://localhost:8000`; use `http://backend:8000` under Docker Compose).
