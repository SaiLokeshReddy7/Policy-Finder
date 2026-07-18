"""Voice input/output endpoints: speech-to-text for the "speak your
situation" feature, and text-to-speech for the "listen to this answer"
feature. Both are HF-only (Anthropic doesn't accept audio), so both return a
clear 503 rather than crashing if HF_API_TOKEN isn't configured."""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from backend.core.rate_limit import limiter
from backend.models.schemas import VoiceSpeakRequest, VoiceTranscribeResponse
from backend.services.voice_intake_service import extract_profile_fields
from backend.tools.speech_to_text_tool import transcribe_audio
from backend.tools.text_to_speech_tool import synthesize_speech

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB is generous for a ~2 minute recording


@router.post("/voice/transcribe", response_model=VoiceTranscribeResponse)
@limiter.limit("15/minute")
async def voice_transcribe(request: Request, file: UploadFile = File(...)) -> VoiceTranscribeResponse:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data received.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio recording is too large (max 10 MB).")

    transcript = transcribe_audio(audio_bytes, content_type=file.content_type or "audio/wav")
    if transcript is None:
        raise HTTPException(
            status_code=503,
            detail="Speech-to-text is unavailable. Set HF_API_TOKEN in .env to enable voice input.",
        )

    suggested_profile = extract_profile_fields(transcript)
    return VoiceTranscribeResponse(transcript=transcript, suggested_profile=suggested_profile)


@router.post("/voice/speak")
@limiter.limit("20/minute")
def voice_speak(request: Request, payload: VoiceSpeakRequest) -> Response:
    result = synthesize_speech(payload.text, payload.language)
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Text-to-speech is temporarily unavailable. Please try again in a moment.",
        )
    audio_bytes, content_type = result
    return Response(content=audio_bytes, media_type=content_type)
