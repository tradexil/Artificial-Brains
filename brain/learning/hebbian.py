"""Hebbian learning: "fire together, wire together."

Detects co-activation of pre/post-synaptic neurons within a ±3 tick
window and queues synapse weight increases.

delta_weight = learning_rate × A.activation × B.activation × plasticity

Learning rate is modulated by:
  - Brain phase (BLOOM vs MATURE)
  - Novelty (novel patterns learn faster)
  - Arousal (high arousal = stronger memory formation)
  - Attention gain (attended signals learn faster)
"""

from __future__ import annotations

import brain_core

from brain.structures.brain_state import ActivationHistory, NeuromodulatorState
from brain.utils.config import (
    BLOOM_END_TICK,
    CRITICAL_END_TICK,
    HEBBIAN_RATE,
    HEBBIAN_WINDOW,
    NOVELTY_LEARNING_BOOST,
    AROUSAL_LEARNING_BOOST,
)


def compute_effective_learning_rate(
    tick: int,
    neuromod: NeuromodulatorState,
    novelty_signal: float = 0.0,
) -> float:
    """Compute the effective Hebbian learning rate based on state."""
    base = HEBBIAN_RATE

    # Phase modulation
    if tick < BLOOM_END_TICK:
        phase_mult = 2.0  # Aggressive learning during BLOOM
    elif tick < CRITICAL_END_TICK:
        phase_mult = 1.0  # Normal during CRITICAL
    else:
        phase_mult = 0.5  # Conservative during MATURE

    # Novelty boost
    novelty_mult = 1.0 + novelty_signal * (NOVELTY_LEARNING_BOOST - 1.0)

    # Arousal boost
    arousal_mult = 1.0 + neuromod.arousal * (AROUSAL_LEARNING_BOOST - 1.0)

    return base * phase_mult * novelty_mult * arousal_mult


def hebbian_update(
    history: ActivationHistory,
    tick: int,
    neuromod: NeuromodulatorState,
    novelty_signal: float = 0.0,
    attention_gains: dict[str, float] | None = None,
    prediction_multiplier: float = 1.0,
) -> int:
    """Detect co-active neuron pairs and queue synapse strengthening.

    Uses Rust batch_hebbian for parallelized synapse traversal.
    The entire loop runs in Rust with rayon — no per-neuron FFI calls.

    Args:
        history: activation history with recent snapshots.
        tick: current tick number.
        neuromod: current neuromodulator state.
        novelty_signal: current novelty score (0.0-1.0).
        attention_gains: region→gain map (unused in batch path).
        prediction_multiplier: from prediction engine (surprise=2x, alarm=3x).

    Returns:
        Number of synapse updates queued.
    """
    current = history.current
    if current is None:
        return 0

    lr = compute_effective_learning_rate(tick, neuromod, novelty_signal)
    lr *= prediction_multiplier

    # Flatten all active neurons from current snapshot
    active_neurons: list[tuple[int, float]] = []
    for neurons in current.active_neurons.values():
        active_neurons.extend(neurons)

    if not active_neurons:
        return 0

    # Collect window-active neurons as dict {neuron_id: max_activation}
    window_active = history.neurons_active_in_window()

    # Delegate entire loop to Rust (parallel with rayon)
    return brain_core.batch_hebbian(active_neurons, window_active, lr)
