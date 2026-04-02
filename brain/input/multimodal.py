"""Multimodal input synchronizer: merges multiple input streams with timing.

Handles different input rates and synchronizes visual, audio, sensory,
and text inputs into a single combined signal per tick.
"""

from __future__ import annotations

from typing import Any

import brain_core

from brain.input.visual_input import VisualInput
from brain.input.audio_input import AudioInput
from brain.input.sensory_input import SensoryInput
from brain.input.text_input import TextInput


class MultimodalInput:
    """Synchronizes and merges multiple input streams per tick.

    Usage:
        mm = MultimodalInput(text_encoder=text_input)
        result = mm.process({
            "text": "hello",
            "sensory": {"temperature": 0.5, "pain": 0.0},
            "visual": frame_array,
        }, tick=100)
    """

    def __init__(
        self,
        text_encoder: TextInput | None = None,
        visual_encoder: VisualInput | None = None,
        audio_encoder: AudioInput | None = None,
        sensory_encoder: SensoryInput | None = None,
    ):
        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder or VisualInput()
        self.audio_encoder = audio_encoder or AudioInput()
        self.sensory_encoder = sensory_encoder or SensoryInput()

        # Input timing: last tick each modality was active
        self._last_active: dict[str, int] = {}

    def process(self, inputs: dict[str, Any], tick: int = 0) -> dict:
        """Process all input modalities for this tick.

        Args:
            inputs: dict with optional keys:
                "text": str — text to encode
                "visual": array — image frame
                "audio": (samples, sample_rate) — audio chunk
                "sensory": dict with temperature/pressure/pain/texture
            tick: current tick number

        Returns:
            dict with per-modality results and combined stats.
        """
        results: dict[str, Any] = {}
        total_neurons = 0
        active_modalities: list[str] = []

        # Text input
        if "text" in inputs and self.text_encoder is not None:
            text_result = self.text_encoder.encode(inputs["text"])
            results["text"] = text_result
            total_neurons += text_result.get("neurons_activated", 0)
            active_modalities.append("text")
            self._last_active["text"] = tick

        # Visual input
        if "visual" in inputs:
            vis_result = self.visual_encoder.encode(inputs["visual"])
            results["visual"] = vis_result
            total_neurons += vis_result.get("neurons_activated", 0)
            active_modalities.append("visual")
            self._last_active["visual"] = tick

        # Audio input
        if "audio" in inputs:
            audio_data = inputs["audio"]
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                samples, sr = audio_data
            else:
                samples = audio_data
                sr = 44100
            audio_result = self.audio_encoder.encode(samples, sr)
            results["audio"] = audio_result
            total_neurons += audio_result.get("neurons_activated", 0)
            active_modalities.append("audio")
            self._last_active["audio"] = tick

        # Sensory input
        if "sensory" in inputs:
            sensory_data = inputs["sensory"]
            sensory_result = self.sensory_encoder.encode(**sensory_data)
            results["sensory"] = sensory_result
            total_neurons += sensory_result.get("neurons_activated", 0)
            active_modalities.append("sensory")
            self._last_active["sensory"] = tick

        # Boost integration if multiple modalities are active
        n_active = len(active_modalities)
        if n_active >= 2:
            strength = min(1.0, n_active / 4.0)
            brain_core.boost_integration(strength, min(100, 10000))

        results["_summary"] = {
            "total_neurons_activated": total_neurons,
            "active_modalities": active_modalities,
            "modality_count": n_active,
            "tick": tick,
        }

        return results

    def last_active_tick(self, modality: str) -> int | None:
        """Get the last tick a modality was active."""
        return self._last_active.get(modality)

    def reset(self) -> None:
        """Reset all encoders and timing state."""
        self._last_active.clear()
        self.audio_encoder.reset()
