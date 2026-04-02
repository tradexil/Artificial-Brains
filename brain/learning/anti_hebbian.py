"""Anti-Hebbian learning: decorrelation / competitive weakening.

If neuron A fires and neuron B does NOT fire (within ±3 ticks),
AND a synapse exists A → B, then weaken that synapse slightly.

This prevents everything from connecting to everything.

delta_weight = -anti_rate × A.activation × (1 - B.activation) × plasticity
"""

from __future__ import annotations

import brain_core

from brain.structures.brain_state import ActivationHistory, NeuromodulatorState
from brain.utils.config import ANTI_HEBBIAN_RATE, BLOOM_END_TICK, CRITICAL_END_TICK


def compute_anti_hebbian_rate(tick: int) -> float:
    """Phase-dependent anti-Hebbian rate."""
    base = ANTI_HEBBIAN_RATE
    if tick < BLOOM_END_TICK:
        return base * 0.5  # Gentle during BLOOM (don't fight growth)
    elif tick < CRITICAL_END_TICK:
        return base * 2.0  # Aggressive during CRITICAL (help prune)
    else:
        return base  # Normal during MATURE


def anti_hebbian_update(
    history: ActivationHistory,
    tick: int,
) -> int:
    """Weaken synapses where source fires but target does not.

    Uses Rust batch_anti_hebbian for parallelized synapse traversal.

    Returns:
        Number of synapse updates queued.
    """
    current = history.current
    if current is None:
        return 0

    rate = compute_anti_hebbian_rate(tick)

    # Flatten all active neurons from current snapshot
    active_neurons: list[tuple[int, float]] = []
    for neurons in current.active_neurons.values():
        active_neurons.extend(neurons)

    if not active_neurons:
        return 0

    # All neurons active in the window
    window_active = history.neurons_active_in_window()

    # Delegate entire loop to Rust (parallel with rayon)
    return brain_core.batch_anti_hebbian(active_neurons, window_active, rate)
