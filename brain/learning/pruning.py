"""Phase-aware synapse pruning.

Three lifecycle phases:
  BLOOM     (0 → 500k ticks):    No pruning, synapse count grows freely.
  CRITICAL  (500k → 2M ticks):   Aggressive pruning by weight & dormancy.
  MATURE    (2M+ ticks):         Gentle pruning, only dead synapses.

Pruning reads all synapses, identifies candidates, and queues batch prunes.
The actual CSR rebuild happens in the maintenance cycle.
"""

from __future__ import annotations

import brain_core

from brain.utils.config import (
    BLOOM_END_TICK,
    CRITICAL_END_TICK,
    CRITICAL_PRUNE_DORMANT_TICKS,
    CRITICAL_PRUNE_DORMANT_WEIGHT,
    CRITICAL_PRUNE_WEIGHT,
    MATURE_PRUNE_DORMANT_TICKS,
    MATURE_PRUNE_WEIGHT,
    REGIONS,
    TOTAL_NEURONS,
)


def get_phase(tick: int) -> str:
    """Return current lifecycle phase name."""
    if tick < BLOOM_END_TICK:
        return "bloom"
    elif tick < CRITICAL_END_TICK:
        return "critical"
    else:
        return "mature"


def pruning_pass(
    tick: int,
    last_fired: dict[tuple[int, int], int] | None = None,
) -> int:
    """Run a pruning pass based on current lifecycle phase.

    Args:
        tick: current tick number.
        last_fired: optional dict of (from, to) → last_tick_fired for
                    dormancy detection. If None, dormancy checks are skipped.

    Returns:
        Number of synapses queued for pruning.
    """
    phase = get_phase(tick)

    if phase == "bloom":
        return 0  # No pruning during BLOOM

    prune_pairs: list[tuple[int, int]] = []

    # Iterate all source neurons and check their synapses
    for src in range(TOTAL_NEURONS):
        outgoing = brain_core.get_outgoing_synapses(src)
        if not outgoing:
            continue

        for tgt, weight, delay, plasticity in outgoing:
            if plasticity < 0.01:
                continue  # Never prune hardwired synapses

            should_prune = False

            if phase == "critical":
                # Aggressive: prune low-weight or dormant
                if weight < CRITICAL_PRUNE_WEIGHT:
                    should_prune = True
                elif last_fired is not None:
                    last_fire = last_fired.get((src, tgt), 0)
                    dormant = tick - last_fire
                    if dormant > CRITICAL_PRUNE_DORMANT_TICKS and weight < CRITICAL_PRUNE_DORMANT_WEIGHT:
                        should_prune = True

            elif phase == "mature":
                # Gentle: only prune very weak + very dormant
                if weight < MATURE_PRUNE_WEIGHT:
                    if last_fired is not None:
                        last_fire = last_fired.get((src, tgt), 0)
                        dormant = tick - last_fire
                        if dormant > MATURE_PRUNE_DORMANT_TICKS:
                            should_prune = True
                    else:
                        # Without dormancy data, prune only the weakest
                        should_prune = True

            if should_prune:
                prune_pairs.append((src, tgt))

    if prune_pairs:
        brain_core.batch_prune_synapses(prune_pairs)

    return len(prune_pairs)


def pruning_pass_sampled(
    tick: int,
    sample_neurons: list[int],
    last_fired: dict[tuple[int, int], int] | None = None,
) -> int:
    """Run pruning on a subset of source neurons (for incremental pruning).

    Instead of iterating all 152k neurons (expensive), call this with a
    rotating sample each maintenance cycle.

    Returns:
        Number of synapses queued for pruning.
    """
    phase = get_phase(tick)
    if phase == "bloom":
        return 0

    prune_pairs: list[tuple[int, int]] = []

    for src in sample_neurons:
        outgoing = brain_core.get_outgoing_synapses(src)
        if not outgoing:
            continue

        for tgt, weight, delay, plasticity in outgoing:
            if plasticity < 0.01:
                continue

            should_prune = False

            if phase == "critical":
                if weight < CRITICAL_PRUNE_WEIGHT:
                    should_prune = True
                elif last_fired is not None:
                    last_fire = last_fired.get((src, tgt), 0)
                    if tick - last_fire > CRITICAL_PRUNE_DORMANT_TICKS and weight < CRITICAL_PRUNE_DORMANT_WEIGHT:
                        should_prune = True
            elif phase == "mature":
                if weight < MATURE_PRUNE_WEIGHT:
                    if last_fired is None or tick - last_fired.get((src, tgt), 0) > MATURE_PRUNE_DORMANT_TICKS:
                        should_prune = True

            if should_prune:
                prune_pairs.append((src, tgt))

    if prune_pairs:
        brain_core.batch_prune_synapses(prune_pairs)

    return len(prune_pairs)
