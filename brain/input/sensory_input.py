"""Sensory input pipeline: convert sensor values into sensory region activations.

Uses population coding: each value activates a gaussian bump of neurons
in the corresponding sub-range.

Sub-ranges:
  0–2499:    temperature (cold → hot gradient)
  2500–4999: pressure (light → heavy)
  5000–7499: pain (none → severe)
  7500–9999: texture (smooth → rough)
"""

from __future__ import annotations

import brain_core


class SensoryInput:
    """Encodes sensory values into sensory region neuron activations.

    Usage:
        encoder = SensoryInput()
        result = encoder.encode(temperature=0.5, pressure=0.3, pain=0.0, texture=0.6)
    """

    def __init__(self, boost: float = 0.8, spread: int = 30):
        self.boost = boost
        self.spread = spread

    def encode(
        self,
        temperature: float = 0.0,
        pressure: float = 0.0,
        pain: float = 0.0,
        texture: float = 0.0,
    ) -> dict:
        """Encode sensory values into brain activations.

        All values should be in [0.0, 1.0]:
          temperature: 0.0 = cold, 1.0 = hot
          pressure: 0.0 = none, 1.0 = heavy
          pain: 0.0 = none, 1.0 = severe
          texture: 0.0 = smooth, 1.0 = rough

        Returns:
            dict with keys: neurons_activated, signals_generated
        """
        # Use Rust-side population coding
        signals = brain_core.encode_sensory(
            temperature, pressure, pain, texture, self.spread
        )

        # Inject as activations (with boost scaling)
        if signals:
            scaled = [(gid, act * self.boost) for gid, act in signals]
            brain_core.inject_activations(scaled)
            # Also boost for activation readout
            neurons = [gid for gid, _ in signals]
            activated = brain_core.boost_sensory(neurons, self.boost * 0.5)
        else:
            activated = 0

        return {
            "neurons_activated": activated,
            "signals_generated": len(signals),
            "temperature": temperature,
            "pressure": pressure,
            "pain": pain,
            "texture": texture,
        }

    def encode_raw(self, signals: list[tuple[int, float]]) -> int:
        """Inject pre-computed sensory neuron activations."""
        if not signals:
            return 0
        neurons = [gid for gid, _ in signals]
        return brain_core.boost_sensory(neurons, self.boost)
