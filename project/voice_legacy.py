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
import threading
import subprocess
from pathlib import Path

logger = logging.getLogger("e3n.voice")

# ── CONFIGURATION ──────────────────────────────────────────────────────

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large-v3
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto, cpu, cuda
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")  # float16, int8, float32
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AndrewNeural")  # E3N's voice — warm American, slightly deeper
TTS_RATE = os.getenv("TTS_RATE", "+2%")  # Slightly faster — confident, not rushed
TTS_PITCH = os.getenv("TTS_PITCH", "-3Hz")  # Slightly deeper than default — calm, not booming
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").lower() in ("true", "1", "yes")

# Vocabulary hint for Whisper — biases transcription toward expected racing/engineering terms
# This dramatically reduces misinterpretation of technical jargon
WHISPER_VOCAB_PROMPT = (
    "E3N, Ethan, thermodynamics, aerodynamics, understeer, oversteer, downforce, drag, "
    "Le Mans, Le Mans Ultimate, LMU, telemetry, chicane, apex, stint, degradation, "
    "compound, intermediate, slick, wet, hypercar, LMDh, LMH, GTE, GT3, GT4, ECU, "
    "kinematics, statics, dynamics, calculus, differential equations, integrals, "
    "torque, horsepower, RPM, camber, toe, caster, ride height, anti-roll bar, "
    "damper, spring rate, aero balance, center of pressure, center of gravity, "
    "Ollama, ChromaDB, VRAM, FastAPI, Qwen, LoRA, QLoRA, GGUF, GPU, CPU, "
    "PowerShell, uvicorn, Python, Electron, inference, embeddings, "
    "pit stop, pit window, fuel load, tire pressure, tire temperature, "
    "sector, split, delta, gap, position, DRS, ERS, MGU-K, MGU-H, "
    "mechanical engineering, fluid dynamics, heat transfer, convection, conduction, "
    "finite element analysis, stress analysis, material science, Young's modulus, "
    "Ferrari, Porsche, Toyota, BMW, Cadillac, Peugeot, Alpine, Lamborghini, "
    "Spa, Monza, Silverstone, Daytona, Sebring, Indianapolis, Nurburgring, "
    "strategy, undercut, overcut, safety car, VSC, yellow flag, red flag, "
    "Reynolds number, Bernoulli, Navier-Stokes, coefficient of friction, "
    "suspension geometry, Ackermann, bump steer, roll center, pitch sensitivity"
)

# ── STT (Speech-to-Text) via faster-whisper ────────────────────────────

_whisper_model = None
_stt_available = False
_whisper_lock = threading.Lock()


def _init_whisper():
    """Lazy-load the Whisper model on first use (thread-safe)."""
    global _whisper_model, _stt_available
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        # Double-check after acquiring lock
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
            initial_prompt=WHISPER_VOCAB_PROMPT,  # Bias toward racing/engineering vocabulary
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


async def synthesize_speech(text: str, voice: str = None, rate: str = None, pitch: str = None) -> dict:
    """
    Convert text to speech audio bytes.

    Args:
        text: Text to speak
        voice: Voice ID (default from config)
        rate: Speed adjustment (e.g., "+10%", "-20%")
        pitch: Pitch adjustment (e.g., "-5Hz", "+10Hz")

    Returns:
        {"audio": bytes, "format": "mp3", "duration_estimate": float}
        or {"error": str} on failure
    """
    voice = voice or TTS_VOICE
    rate = rate or TTS_RATE
    pitch = pitch or TTS_PITCH

    if not text or not text.strip():
        return {"error": "Empty text"}

    # Clean text for TTS — remove markdown, code blocks, excessive punctuation
    clean = _clean_for_tts(text)
    if not clean:
        return {"error": "No speakable text after cleanup"}

    # Try edge-tts first
    try:
        import edge_tts
        communicate = edge_tts.Communicate(clean, voice=voice, rate=rate, pitch=pitch)

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
    # Write text to a temp file to avoid PowerShell injection
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as tf:
        tf.write(text)
        text_path = tf.name

    # Escape single quotes in paths for safe PowerShell string embedding
    safe_tmp = tmp_path.replace("'", "''")
    safe_text = text_path.replace("'", "''")
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('{safe_tmp}')
$text = [System.IO.File]::ReadAllText('{safe_text}', [System.Text.Encoding]::UTF8)
$synth.Speak($text)
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
        for p in (tmp_path, text_path):
            try:
                os.unlink(p)
            except Exception:
                pass


def _clean_for_tts(text: str) -> str:
    """Clean text for TTS — remove code/markdown, convert to natural spoken form."""
    import re  # noqa: local import avoids top-level dep when voice disabled

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
    # Convert bullet points / dashes to natural speech pauses
    text = re.sub(r'^[\s]*[-•*]\s+', '', text, flags=re.MULTILINE)
    # Remove numbered list markers
    text = re.sub(r'^[\s]*\d+[\.\)]\s+', '', text, flags=re.MULTILINE)
    # Remove pipe tables
    text = re.sub(r'\|[^\n]+\|', '', text)
    # Remove horizontal rules
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    # Convert multiple newlines to period (natural pause)
    text = re.sub(r'\n{2,}', '. ', text)
    # Convert single newlines to space
    text = re.sub(r'\n', ' ', text)
    # Remove emoji
    text = re.sub(r'[\U0001F600-\U0001F9FF\U0001FA00-\U0001FAFF\U00002702-\U000027B0]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove trailing "code block omitted" if it's the last thing
    text = re.sub(r'\s*code block omitted\s*\.?\s*$', '', text).strip()
    # Truncate very long text (TTS shouldn't read novels)
    if len(text) > 1500:
        # Cut at sentence boundary
        cut = text[:1500].rfind('. ')
        if cut > 500:
            text = text[:cut + 1]
        else:
            text = text[:1500] + "."

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
