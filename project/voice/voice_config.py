"""
Voice pipeline configuration — every tunable parameter lives here.

Zero magic numbers in processing code. All effects read from VoiceConfig.
Presets stored as JSON in ~/deltai/data\\voice\\presets\\.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json
import logging

logger = logging.getLogger("deltai.voice.config")

PRESETS_DIR = Path(r"~/deltai/data\voice\presets")
VOICE_DATA_DIR = Path(r"~/deltai/data\voice")


@dataclass
class TTSSettings:
    """Piper TTS engine settings."""
    model_name: str = "en_US-ryan-medium"
    model_dir: str = str(VOICE_DATA_DIR / "piper_models")
    sample_rate: int = 22050
    speaker_id: Optional[int] = None


@dataclass
class RVCSettings:
    """RVC voice conversion settings."""
    enabled: bool = False
    model_dir: str = str(VOICE_DATA_DIR / "models")
    model_name: str = "deltai_voice"
    index_path: Optional[str] = None
    f0_method: str = "dio"          # dio (fast), harvest (quality), rmvpe (best, needs model)
    f0_up_key: int = 0              # Pitch shift in semitones (0 = no shift)
    protect: float = 0.33           # Protect voiceless consonants (0-0.5)
    device: str = "auto"            # auto, cuda, cpu
    # Training parameters
    training_audio_dir: str = str(VOICE_DATA_DIR / "training_audio")
    training_sample_rate: int = 40000   # RVC v2 standard
    training_epochs: int = 200
    training_batch_size: int = 8
    training_save_interval: int = 50
    training_lr: float = 1e-4
    min_segment_duration: float = 3.0   # seconds
    max_segment_duration: float = 15.0  # seconds


@dataclass
class PitchShiftSettings:
    """Pitch shift effect — lowers voice 1-2 semitones."""
    enabled: bool = True
    semitones: float = -1.5


@dataclass
class RingModSettings:
    """Ring modulation — adds metallic/electronic character."""
    enabled: bool = True
    carrier_freq_hz: float = 300.0
    wet_mix: float = 0.07  # 5-10% wet


@dataclass
class LowPassSettings:
    """Low-pass filter — tames harsh highs."""
    enabled: bool = True
    cutoff_hz: float = 9000.0
    order: int = 4


@dataclass
class ReverbSettings:
    """Short convolution reverb — metallic space."""
    enabled: bool = True
    ir_path: Optional[str] = str(VOICE_DATA_DIR / "impulse_responses" / "metallic_short.npy")
    ir_duration_ms: float = 40.0  # 30-50ms synthetic IR
    wet_mix: float = 0.15
    decay: float = 0.3


@dataclass
class ChorusSettings:
    """Light chorus/flanger — subtle movement."""
    enabled: bool = True
    rate_hz: float = 0.75  # 0.5-1Hz
    depth_ms: float = 2.0
    wet_mix: float = 0.05  # ~5% wet


@dataclass
class SubBassSettings:
    """Sub-bass hum layer — electronic presence."""
    enabled: bool = True
    freq_hz: float = 120.0
    level_db: float = -32.0  # -30 to -35dB


@dataclass
class CompressorSettings:
    """Dynamic compression — evens out levels."""
    enabled: bool = True
    ratio: float = 2.0
    threshold_db: float = -20.0
    attack_ms: float = 5.0
    release_ms: float = 50.0


@dataclass
class PlaybackSettings:
    """Audio playback settings."""
    device_index: Optional[int] = None
    blocksize: int = 1024
    latency: str = "low"


@dataclass
class VoiceConfig:
    """Master voice pipeline configuration.

    Every tunable parameter for the TTS -> RVC -> PostProcess -> Playback
    pipeline. No magic numbers anywhere else.
    """
    tts: TTSSettings = field(default_factory=TTSSettings)
    rvc: RVCSettings = field(default_factory=RVCSettings)
    pitch_shift: PitchShiftSettings = field(default_factory=PitchShiftSettings)
    ring_mod: RingModSettings = field(default_factory=RingModSettings)
    low_pass: LowPassSettings = field(default_factory=LowPassSettings)
    reverb: ReverbSettings = field(default_factory=ReverbSettings)
    chorus: ChorusSettings = field(default_factory=ChorusSettings)
    sub_bass: SubBassSettings = field(default_factory=SubBassSettings)
    compressor: CompressorSettings = field(default_factory=CompressorSettings)
    playback: PlaybackSettings = field(default_factory=PlaybackSettings)

    def to_dict(self) -> dict:
        """Serialize config to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceConfig":
        """Reconstruct config from a plain dict."""
        cfg = cls()
        for section_name, section_cls in [
            ("tts", TTSSettings),
            ("rvc", RVCSettings),
            ("pitch_shift", PitchShiftSettings),
            ("ring_mod", RingModSettings),
            ("low_pass", LowPassSettings),
            ("reverb", ReverbSettings),
            ("chorus", ChorusSettings),
            ("sub_bass", SubBassSettings),
            ("compressor", CompressorSettings),
            ("playback", PlaybackSettings),
        ]:
            if section_name in data:
                setattr(cfg, section_name, section_cls(**data[section_name]))
        return cfg

    def save_preset(self, name: str) -> Path:
        """Save current config as a named preset JSON file.

        Args:
            name: Preset name (used as filename, .json appended).

        Returns:
            Path to the saved preset file.
        """
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        path = PRESETS_DIR / f"{name}.json"
        try:
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
            logger.info("Saved voice preset: %s -> %s", name, path)
        except Exception as e:
            logger.error("Failed to save preset '%s': %s", name, e)
        return path

    @classmethod
    def load_preset(cls, name: str) -> "VoiceConfig":
        """Load a named preset from disk.

        Args:
            name: Preset name (without .json extension).

        Returns:
            VoiceConfig populated from the preset, or DEFAULT_CONFIG if not found.
        """
        path = PRESETS_DIR / f"{name}.json"
        if not path.exists():
            logger.warning("Preset '%s' not found at %s — using defaults", name, path)
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info("Loaded voice preset: %s", name)
            return cls.from_dict(data)
        except Exception as e:
            logger.error("Failed to load preset '%s': %s — using defaults", name, e)
            return cls()


# Singleton default config instance
DEFAULT_CONFIG = VoiceConfig()
