"""
E3N Voice Module — STT (Whisper) + TTS (Edge-TTS / espeak fallback)

Provides speech-to-text and text-to-speech capabilities for E3N.
STT: faster-whisper (CTranslate2 backend) — runs locally on GPU or CPU.
TTS: edge-tts (Microsoft Edge free TTS API) — high quality, no API key needed.

Endpoints are registered in main.py:
  POST /voice/stt     — Upload audio, get transcription
  POST /voice/tts     — Send text, get audio stream
  GET  /voice/status  — Voice subsystem health
  WS   /voice/stream  — Real-time bidirectional voice (future)
"""

import os
import io
import time
import wave
import json
import struct
import asyncio
import logging
import tempfile
import subprocess
from pathlib import Path

logger = logging.getLogger("e3n.voice")

# ── CONFIGURATION ──────────────────────────────────────────────────────

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large-v3
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto, cpu, cuda
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")  # float16, int8, float32
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-GuyNeural")  # Microsoft Edge TTS voice
TTS_RATE = os.getenv("TTS_RATE", "+0%")  # Speech rate adjustment
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").lower() in ("true", "1", "yes")

# ── STT (Speech-to-Text) via faster-whisper ────────────────────────────

_whisper_model = None
_stt_available = False


def _init_whisper():
    """Lazy-load the Whisper model on first use."""
    global _whisper_model, _stt_available
    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel

        device = WHISPER_DEVICE
        if device == "auto":
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_mb = (mem.total - mem.used) / 1e6
                # Only use GPU if enough free VRAM (base model ~150MB, small ~500MB)
                model_vram = {"tiny": 75, "base": 150, "small": 500, "medium": 1500, "large-v3": 3000}
                needed = model_vram.get(WHISPER_MODEL, 150)
                device = "cuda" if free_mb > needed + 500 else "cpu"  # 500MB buffer
            except Exception:
                device = "cpu"

        compute_type = WHISPER_COMPUTE
        if device == "cpu" and compute_type == "float16":
            compute_type = "float32"  # float16 not supported on CPU for some backends

        logger.info(f"Loading Whisper model '{WHISPER_MODEL}' on {device} ({compute_type})")
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
        )
        _stt_available = True
        logger.info(f"Whisper model loaded: {WHISPER_MODEL} ({device})")
        return _whisper_model
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")
        _stt_available = False
        return None


def transcribe_audio(audio_bytes: bytes, language: str = "en") -> dict:
    """
    Transcribe audio bytes to text using Whisper.

    Args:
        audio_bytes: Raw audio data (WAV, MP3, FLAC, etc.)
        language: Language code (default "en")

    Returns:
        {"text": str, "language": str, "duration": float, "segments": list}
    """
    model = _init_whisper()
    if model is None:
        return {"error": "Whisper model not available", "text": ""}

    # Write to temp file (faster-whisper needs a file path or file-like)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        start = time.time()
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,  # Voice activity detection — skip silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        # Collect all segments
        text_parts = []
        segment_list = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            segment_list.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "confidence": round(seg.avg_logprob, 3) if hasattr(seg, 'avg_logprob') else None,
            })

        full_text = " ".join(text_parts).strip()
        elapsed = time.time() - start

        return {
            "text": full_text,
            "language": info.language if info else language,
            "duration": round(info.duration, 2) if info else 0,
            "processing_time": round(elapsed, 2),
            "segments": segment_list,
        }
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"error": str(e), "text": ""}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── TTS (Text-to-Speech) via edge-tts ──────────────────────────────────

_tts_available = False


def _check_tts():
    """Check if edge-tts is available."""
    global _tts_available
    try:
        import edge_tts  # noqa: F401
        _tts_available = True
        return True
    except ImportError:
        # Fallback: check if espeak is available on Windows
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Add-Type -AssemblyName System.Speech; echo 'OK'"],
                capture_output=True, text=True, timeout=5
            )
            _tts_available = result.stdout.strip() == "OK"
            return _tts_available
        except Exception:
            _tts_available = False
            return False


async def synthesize_speech(text: str, voice: str = None, rate: str = None) -> dict:
    """
    Convert text to speech audio bytes.

    Args:
        text: Text to speak
        voice: Voice ID (default from config)
        rate: Speed adjustment (e.g., "+10%", "-20%")

    Returns:
        {"audio": bytes, "format": "mp3", "duration_estimate": float}
        or {"error": str} on failure
    """
    voice = voice or TTS_VOICE
    rate = rate or TTS_RATE

    if not text or not text.strip():
        return {"error": "Empty text"}

    # Clean text for TTS — remove markdown, code blocks, excessive punctuation
    clean = _clean_for_tts(text)
    if not clean:
        return {"error": "No speakable text after cleanup"}

    # Try edge-tts first
    try:
        import edge_tts
        communicate = edge_tts.Communicate(clean, voice=voice, rate=rate)

        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])

        if not audio_data:
            return {"error": "TTS produced no audio"}

        # Estimate duration (~150 words per minute)
        word_count = len(clean.split())
        duration_est = word_count / 2.5  # seconds

        return {
            "audio": bytes(audio_data),
            "format": "mp3",
            "size": len(audio_data),
            "duration_estimate": round(duration_est, 1),
        }
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"edge-tts failed: {e}")

    # Fallback: Windows SAPI (System.Speech)
    try:
        return await _windows_tts(clean)
    except Exception as e:
        logger.error(f"All TTS backends failed: {e}")
        return {"error": f"TTS unavailable: {e}"}


async def _windows_tts(text: str) -> dict:
    """Fallback TTS using Windows SAPI via PowerShell."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    # Escape for PowerShell
    escaped = text.replace("'", "''").replace('"', '`"')

    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('{tmp_path}')
$synth.Speak('{escaped}')
$synth.Dispose()
"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", ps_script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode != 0:
            return {"error": f"Windows TTS failed: {stderr.decode()[:200]}"}

        with open(tmp_path, "rb") as f:
            audio_data = f.read()

        return {
            "audio": audio_data,
            "format": "wav",
            "size": len(audio_data),
            "duration_estimate": round(len(text.split()) / 2.5, 1),
        }
    except asyncio.TimeoutError:
        return {"error": "TTS timed out"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _clean_for_tts(text: str) -> str:
    """Clean text for TTS — remove code blocks, markdown, URLs."""
    import re

    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', ' code block omitted ', text)
    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove markdown bold/italic
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove file paths (Windows-style)
    text = re.sub(r'[A-Z]:\\[\w\\]+\.\w+', 'file path', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Truncate very long text (TTS shouldn't read novels)
    if len(text) > 2000:
        text = text[:2000] + "... Message truncated for speech."

    return text


# ── VOICE STATUS ───────────────────────────────────────────────────────


def get_voice_status() -> dict:
    """Get voice subsystem status."""
    stt_status = "loaded" if _whisper_model is not None else ("available" if _stt_available else "not_loaded")
    tts_ok = _check_tts()

    return {
        "enabled": VOICE_ENABLED,
        "stt": {
            "status": stt_status,
            "model": WHISPER_MODEL,
            "device": WHISPER_DEVICE,
        },
        "tts": {
            "status": "available" if tts_ok else "unavailable",
            "voice": TTS_VOICE,
            "rate": TTS_RATE,
        },
    }


# ── AUDIO RECORDING HELPER ────────────────────────────────────────────

def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> bytes:
    """
    Record audio from the default microphone.
    Returns WAV bytes. For use in local/Electron context only.
    """
    try:
        import sounddevice as sd
        import numpy as np

        logger.info(f"Recording {duration}s of audio at {sample_rate}Hz...")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()

        # Convert to WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

        return buf.getvalue()
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        return b""
