"""Hardwire sensory→motor reflex pathways.

Reflexes are fast, un-gated connections: sensory input → motor output
in 2–3 ticks, bypassing pattern/integration/executive.

Key reflexes:
  - Pain → Withdraw: pain sensory neurons (5000–7499) → withdraw motor (135000–137999)
  - Hot → Withdraw: high temperature (near 2499) → withdraw motor
  - Pressure → Approach/Grasp: pressure neurons → approach motor (for grip reflex)

Reflex synapses have:
  - Low delay (2)
  - Low plasticity (0.05) — nearly fixed, but slight adaptation possible
  - Moderate weight (0.4) — strong enough to trigger motor without executive
"""

from __future__ import annotations

import random

from brain.utils.config import REGIONS


# Reflex pathway definitions
_PAIN_START = 5000
_PAIN_END = 7500
_TEMP_START = 0
_TEMP_END = 2500
_PRESSURE_START = 2500
_PRESSURE_END = 5000

_APPROACH_START = 130_000
_APPROACH_END = 135_000
_WITHDRAW_START = 135_000
_WITHDRAW_END = 138_000

# Reflex parameters
_REFLEX_DELAY = 2
_REFLEX_PLASTICITY = 0.05
_REFLEX_WEIGHT = 0.4


def wire_reflexes(
    rng: random.Random | None = None,
    pain_to_withdraw: int = 100,
    heat_to_withdraw: int = 50,
    pressure_to_approach: int = 30,
) -> list[tuple[int, int, float, int, float]]:
    """Create reflex synapses: sensory → motor.

    Args:
        rng: random generator
        pain_to_withdraw: number of pain→withdraw synapses
        heat_to_withdraw: number of hot→withdraw synapses
        pressure_to_approach: number of pressure→approach synapses

    Returns:
        list of (from, to, weight, delay, plasticity) tuples.
    """
    if rng is None:
        rng = random.Random(800)

    synapses: list[tuple[int, int, float, int, float]] = []

    # Pain → Withdraw
    pain_neurons = list(range(_PAIN_START, _PAIN_END))
    withdraw_neurons = list(range(_WITHDRAW_START, _WITHDRAW_END))
    for _ in range(pain_to_withdraw):
        src = rng.choice(pain_neurons)
        tgt = rng.choice(withdraw_neurons)
        weight = _REFLEX_WEIGHT + rng.uniform(-0.05, 0.05)
        synapses.append((src, tgt, weight, _REFLEX_DELAY, _REFLEX_PLASTICITY))

    # High temperature → Withdraw
    # High temp = neurons near the end of temp range (close to 2499)
    hot_neurons = list(range(_TEMP_END - 500, _TEMP_END))
    for _ in range(heat_to_withdraw):
        src = rng.choice(hot_neurons)
        tgt = rng.choice(withdraw_neurons)
        weight = _REFLEX_WEIGHT * 0.8 + rng.uniform(-0.05, 0.05)
        synapses.append((src, tgt, weight, _REFLEX_DELAY, _REFLEX_PLASTICITY))

    # Pressure → Approach (grip reflex)
    pressure_neurons = list(range(_PRESSURE_START, _PRESSURE_END))
    approach_neurons = list(range(_APPROACH_START, _APPROACH_END))
    for _ in range(pressure_to_approach):
        src = rng.choice(pressure_neurons)
        tgt = rng.choice(approach_neurons)
        weight = _REFLEX_WEIGHT * 0.5 + rng.uniform(-0.03, 0.03)
        synapses.append((src, tgt, weight, _REFLEX_DELAY + 1, _REFLEX_PLASTICITY))

    return synapses
