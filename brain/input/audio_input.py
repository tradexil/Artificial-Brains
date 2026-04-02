"""Audio input pipeline: convert sound data into audio region activations.

Processes audio chunks into auditory features:
  - Frequency (30000–34999): pitch decomposition (log-scale Hz mapping)
  - Temporal (35000–39999): rhythm, onset detection, duration patterns
  - Complex (40000–44999): timbre, phoneme-like patterns
"""

from __future__ import annotations

import math

import brain_core

from brain.utils.config import REGIONS

_AUDIO_START = REGIONS["audio"][0]  # 30000

# Sub-region offsets (local to audio)
_FREQ_START = 0       # 30000
_FREQ_END = 5000      # 34999
_TEMPORAL_START = 5000   # 35000
_TEMPORAL_END = 10000    # 39999
_COMPLEX_START = 10000   # 40000
_COMPLEX_END = 15000     # 44999

# Frequency range (human hearing)
_MIN_FREQ = 20.0
_MAX_FREQ = 20000.0


class AudioInput:
    """Encodes audio data into audio region neuron activations.

    Usage:
        encoder = AudioInput()
        result = encoder.encode(samples, sample_rate=44100)
    """

    def __init__(self, boost: float = 0.7, spread: int = 20):
        self.boost = boost
        self.spread = spread
        self._prev_energy: float = 0.0  # for onset detection
        self._onset_history: list[float] = []

    def encode(self, samples: list[float], sample_rate: int = 44100) -> dict:
        """Encode an audio chunk into brain activations.

        Args:
            samples: list of audio sample values (mono, [-1.0, 1.0] or [0, 255])
            sample_rate: sampling rate in Hz

        Returns:
            dict with keys: neurons_activated, freq_count, temporal_count, complex_count
        """
        if not samples:
            return {
                "neurons_activated": 0,
                "freq_count": 0,
                "temporal_count": 0,
                "complex_count": 0,
            }

        # Normalize samples
        norm = self._normalize(samples)

        # Extract features
        freq_signals = self._extract_frequency(norm, sample_rate)
        temporal_signals = self._extract_temporal(norm, sample_rate)
        complex_signals = self._extract_complex(norm, sample_rate)

        all_signals = freq_signals + temporal_signals + complex_signals

        # Activate audio neurons
        neurons = [gid for gid, _ in all_signals]
        activated = 0
        if neurons:
            activated = brain_core.boost_audio(neurons, self.boost)

        return {
            "neurons_activated": activated,
            "freq_count": len(freq_signals),
            "temporal_count": len(temporal_signals),
            "complex_count": len(complex_signals),
            "total_signals": len(all_signals),
        }

    def _normalize(self, samples: list[float]) -> list[float]:
        """Normalize audio samples to [-1.0, 1.0]."""
        max_val = max(abs(s) for s in samples) if samples else 1.0
        if max_val > 1.0:
            return [s / max_val for s in samples]
        return list(samples)

    def _extract_frequency(
        self, samples: list[float], sample_rate: int
    ) -> list[tuple[int, float]]:
        """Simple DFT-based frequency analysis → frequency neurons."""
        signals = []
        n = len(samples)
        if n < 2:
            return signals

        # Compute power spectrum via simple DFT on a few key frequencies
        # Use log-spaced frequency bins for efficiency
        num_bins = min(64, n // 2)
        freq_bins = [
            _MIN_FREQ * (_MAX_FREQ / _MIN_FREQ) ** (i / max(num_bins - 1, 1))
            for i in range(num_bins)
        ]

        for freq in freq_bins:
            if freq > sample_rate / 2:
                break

            # Goertzel-like: compute power at this frequency
            omega = 2.0 * math.pi * freq / sample_rate
            real_sum = 0.0
            imag_sum = 0.0
            for i, s in enumerate(samples):
                real_sum += s * math.cos(omega * i)
                imag_sum += s * math.sin(omega * i)

            power = math.sqrt(real_sum ** 2 + imag_sum ** 2) / n
            if power > 0.01:
                # Map frequency to neuron ID using log scale
                neuron_signals = brain_core.frequency_to_neurons(freq, self.spread)
                for gid, act in neuron_signals:
                    signals.append((gid, act * min(1.0, power)))

        return signals

    def _extract_temporal(
        self, samples: list[float], sample_rate: int
    ) -> list[tuple[int, float]]:
        """Extract temporal patterns: energy, onsets → temporal neurons."""
        signals = []
        n = len(samples)
        if n < 2:
            return signals

        # Compute short-term energy
        energy = sum(s * s for s in samples) / n

        # Onset detection: sudden energy increase
        onset_strength = max(0.0, energy - self._prev_energy * 1.5)
        self._prev_energy = energy

        self._onset_history.append(onset_strength)
        if len(self._onset_history) > 100:
            self._onset_history = self._onset_history[-100:]

        # Energy level → temporal neurons (front half)
        energy_norm = min(1.0, energy * 5.0)
        if energy_norm > 0.01:
            center = int(energy_norm * 2499)
            spread = 30
            for offset in range(-spread * 3, spread * 3 + 1):
                gid = _AUDIO_START + _TEMPORAL_START + max(0, min(center + offset, 2499))
                dist = abs(offset)
                act = math.exp(-0.5 * dist * dist / (spread * spread))
                if act > 0.01:
                    signals.append((gid, act * energy_norm))

        # Onset → temporal neurons (back half, 2500–4999)
        if onset_strength > 0.01:
            onset_norm = min(1.0, onset_strength * 10.0)
            onset_center = 2500 + int(onset_norm * 2499)
            for offset in range(-20, 21):
                gid = _AUDIO_START + _TEMPORAL_START + max(2500, min(onset_center + offset, 4999))
                dist = abs(offset)
                act = math.exp(-0.5 * dist * dist / (15.0 * 15.0))
                if act > 0.01:
                    signals.append((gid, act * onset_norm))

        return signals

    def _extract_complex(
        self, samples: list[float], sample_rate: int
    ) -> list[tuple[int, float]]:
        """Extract complex features: zero-crossing rate (timbre) → complex neurons."""
        signals = []
        n = len(samples)
        if n < 2:
            return signals

        # Zero-crossing rate (related to timbre/noisiness)
        zc = 0
        for i in range(1, n):
            if (samples[i] >= 0) != (samples[i - 1] >= 0):
                zc += 1
        zcr = zc / (n - 1)

        # ZCR → complex sub-region neuron
        if zcr > 0.01:
            zcr_norm = min(1.0, zcr * 2.0)
            center = int(zcr_norm * 4999)
            gid = _AUDIO_START + _COMPLEX_START + max(0, min(center, 4999))
            signals.append((gid, zcr_norm))
            # Spread
            for offset in [-1, 1, -2, 2]:
                neighbor = center + offset
                if 0 <= neighbor < 5000:
                    signals.append((_AUDIO_START + _COMPLEX_START + neighbor, zcr_norm * 0.5))

        return signals

    def encode_frequency(self, freq_hz: float) -> list[tuple[int, float]]:
        """Encode a single frequency into neuron activations (without injecting)."""
        return brain_core.frequency_to_neurons(freq_hz, self.spread)

    def reset(self) -> None:
        """Reset internal state (onset history, energy tracking)."""
        self._prev_energy = 0.0
        self._onset_history.clear()
