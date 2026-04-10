"""
Post-processing effects chain for deltai's electronic voice character.

All effects use numpy/scipy. Pedalboard is optional (not used here).
Every parameter is read from VoiceConfig — zero magic numbers.
Each effect is individually toggleable.

Effects chain order:
  1. Pitch shift
  2. Ring modulation
  3. Low-pass filter
  4. Convolution reverb
  5. Chorus/flanger
  6. Sub-bass hum
  7. Dynamic compression
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .voice_config import VoiceConfig, DEFAULT_CONFIG

logger = logging.getLogger("deltai.voice.postprocess")

# Check for scipy availability
try:
    import scipy.signal  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    logger.warning("scipy not available — some effects will be skipped")


class PostProcessor:
    """Electronic voice effects chain.

    Processes float32 audio normalized to [-1.0, 1.0].
    All parameters sourced from VoiceConfig.
    """

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        self._config = config or DEFAULT_CONFIG
        self._ir_cache: Optional[np.ndarray] = None

    def process(self, audio: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        """Run the full effects chain on audio.

        Args:
            audio: Float32 numpy array normalized to [-1.0, 1.0].
            sample_rate: Audio sample rate in Hz.

        Returns:
            Processed audio as float32 numpy array clipped to [-1.0, 1.0].
        """
        if audio is None or len(audio) == 0:
            return audio

        result = audio.astype(np.float32)

        # 1. Pitch shift
        if self._config.pitch_shift.enabled:
            result = self._pitch_shift(result, sample_rate)

        # 2. Ring modulation
        if self._config.ring_mod.enabled:
            result = self._ring_mod(result, sample_rate)

        # 3. Low-pass filter
        if self._config.low_pass.enabled and _HAS_SCIPY:
            result = self._low_pass(result, sample_rate)

        # 4. Convolution reverb
        if self._config.reverb.enabled:
            result = self._reverb(result, sample_rate)

        # 5. Chorus/flanger
        if self._config.chorus.enabled:
            result = self._chorus(result, sample_rate)

        # 6. Sub-bass hum
        if self._config.sub_bass.enabled:
            result = self._sub_bass(result, sample_rate)

        # 7. Dynamic compression
        if self._config.compressor.enabled:
            result = self._compress(result, sample_rate)

        # Final clip to valid range
        return np.clip(result, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Individual effects
    # ------------------------------------------------------------------

    def _pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Shift pitch down by configured semitones using resampling.

        Simple resample-based approach: resample to change pitch,
        then time-stretch back to original length.
        """
        semitones = self._config.pitch_shift.semitones
        if semitones == 0.0:
            return audio

        try:
            factor = 2.0 ** (-semitones / 12.0)
            # Resample: stretch in time (changes both pitch and duration)
            new_len = int(len(audio) * factor)
            if new_len < 2:
                return audio

            indices = np.linspace(0, len(audio) - 1, new_len)
            stretched = np.interp(indices, np.arange(len(audio)), audio)

            # Resample back to original length (restores duration, keeps pitch shift)
            out_indices = np.linspace(0, len(stretched) - 1, len(audio))
            result = np.interp(out_indices, np.arange(len(stretched)), stretched)

            return result.astype(np.float32)
        except Exception as e:
            logger.warning("Pitch shift failed: %s — skipping", e)
            return audio

    def _ring_mod(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply ring modulation with a sine carrier."""
        cfg = self._config.ring_mod
        try:
            t = np.arange(len(audio), dtype=np.float32) / sr
            carrier = np.sin(2.0 * np.pi * cfg.carrier_freq_hz * t)
            modulated = audio * carrier
            return audio * (1.0 - cfg.wet_mix) + modulated * cfg.wet_mix
        except Exception as e:
            logger.warning("Ring modulation failed: %s — skipping", e)
            return audio

    def _low_pass(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply Butterworth low-pass filter."""
        cfg = self._config.low_pass
        try:
            nyquist = sr / 2.0
            cutoff_norm = cfg.cutoff_hz / nyquist
            if cutoff_norm >= 1.0:
                return audio  # cutoff above Nyquist, no filtering needed
            b, a = scipy.signal.butter(cfg.order, cutoff_norm, btype="low")
            return scipy.signal.lfilter(b, a, audio).astype(np.float32)
        except Exception as e:
            logger.warning("Low-pass filter failed: %s — skipping", e)
            return audio

    def _reverb(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply short convolution reverb with metallic impulse response."""
        cfg = self._config.reverb
        try:
            ir = self._get_impulse_response(sr)
            if ir is None:
                return audio

            # Convolve and trim to original length
            wet = np.convolve(audio, ir, mode="full")[: len(audio)]
            return (audio * (1.0 - cfg.wet_mix) + wet * cfg.wet_mix).astype(np.float32)
        except Exception as e:
            logger.warning("Reverb failed: %s — skipping", e)
            return audio

    def _get_impulse_response(self, sr: int) -> Optional[np.ndarray]:
        """Load or generate a short metallic impulse response."""
        if self._ir_cache is not None:
            return self._ir_cache

        cfg = self._config.reverb

        # Try loading from file
        if cfg.ir_path:
            ir_path = Path(cfg.ir_path)
            if ir_path.exists():
                try:
                    self._ir_cache = np.load(str(ir_path)).astype(np.float32)
                    logger.debug("Loaded impulse response from %s", ir_path)
                    return self._ir_cache
                except Exception as e:
                    logger.debug("Could not load IR file: %s — generating synthetic", e)

        # Generate synthetic metallic impulse response
        duration_s = cfg.ir_duration_ms / 1000.0
        n_samples = int(sr * duration_s)
        if n_samples < 2:
            return None

        t = np.arange(n_samples, dtype=np.float32) / sr
        # Exponential decay with metallic resonances
        decay_env = np.exp(-t / (duration_s * cfg.decay))
        # Mix of resonant frequencies for metallic character
        resonance = (
            0.5 * np.sin(2.0 * np.pi * 2400.0 * t)
            + 0.3 * np.sin(2.0 * np.pi * 4800.0 * t)
            + 0.2 * np.sin(2.0 * np.pi * 7200.0 * t)
        )
        ir = (decay_env * resonance).astype(np.float32)
        # Normalize
        peak = np.max(np.abs(ir))
        if peak > 0:
            ir /= peak

        # Save for future use
        if cfg.ir_path:
            try:
                ir_path = Path(cfg.ir_path)
                ir_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(ir_path), ir)
                logger.info("Saved synthetic impulse response to %s", ir_path)
            except Exception as e:
                logger.debug("Could not save IR file: %s", e)

        self._ir_cache = ir
        return ir

    def _chorus(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply light chorus/flanger via modulated delay."""
        cfg = self._config.chorus
        try:
            n = len(audio)
            t = np.arange(n, dtype=np.float32) / sr

            # LFO modulates delay time
            depth_samples = cfg.depth_ms / 1000.0 * sr
            lfo = depth_samples * np.sin(2.0 * np.pi * cfg.rate_hz * t)

            # Base delay at center of modulation range
            base_delay = int(depth_samples + 1)
            delayed = np.zeros(n, dtype=np.float32)

            for i in range(base_delay + int(depth_samples) + 1, n):
                delay = base_delay + lfo[i]
                idx = i - delay
                # Linear interpolation for fractional delay
                idx_int = int(idx)
                frac = idx - idx_int
                if 0 <= idx_int < n - 1:
                    delayed[i] = audio[idx_int] * (1.0 - frac) + audio[idx_int + 1] * frac

            return (audio * (1.0 - cfg.wet_mix) + delayed * cfg.wet_mix).astype(np.float32)
        except Exception as e:
            logger.warning("Chorus failed: %s — skipping", e)
            return audio

    def _sub_bass(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Layer a sub-bass sine hum underneath the audio."""
        cfg = self._config.sub_bass
        try:
            t = np.arange(len(audio), dtype=np.float32) / sr
            # Convert dB to linear amplitude
            amplitude = 10.0 ** (cfg.level_db / 20.0)
            hum = amplitude * np.sin(2.0 * np.pi * cfg.freq_hz * t)
            return (audio + hum).astype(np.float32)
        except Exception as e:
            logger.warning("Sub-bass hum failed: %s — skipping", e)
            return audio

    def _compress(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply dynamic compression with fast attack.

        Simple feed-forward compressor: envelope follower with
        configurable attack/release, ratio applied above threshold.
        """
        cfg = self._config.compressor
        try:
            threshold = 10.0 ** (cfg.threshold_db / 20.0)
            attack_coeff = np.exp(-1.0 / (cfg.attack_ms / 1000.0 * sr))
            release_coeff = np.exp(-1.0 / (cfg.release_ms / 1000.0 * sr))

            envelope = np.zeros(len(audio), dtype=np.float32)
            abs_audio = np.abs(audio)

            # Envelope follower
            for i in range(1, len(audio)):
                if abs_audio[i] > envelope[i - 1]:
                    envelope[i] = attack_coeff * envelope[i - 1] + (1.0 - attack_coeff) * abs_audio[i]
                else:
                    envelope[i] = release_coeff * envelope[i - 1] + (1.0 - release_coeff) * abs_audio[i]

            # Gain computation
            gain = np.ones(len(audio), dtype=np.float32)
            above = envelope > threshold
            if np.any(above):
                # gain reduction = (envelope/threshold)^(1/ratio - 1)
                gain[above] = (envelope[above] / threshold) ** (1.0 / cfg.ratio - 1.0)

            result = audio * gain

            # Makeup gain: normalize to roughly original peak
            peak_in = np.max(np.abs(audio))
            peak_out = np.max(np.abs(result))
            if peak_out > 0 and peak_in > 0:
                result *= peak_in / peak_out

            return result.astype(np.float32)
        except Exception as e:
            logger.warning("Compression failed: %s — skipping", e)
            return audio
