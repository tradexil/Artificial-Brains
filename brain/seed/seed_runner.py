"""Seed runner: orchestrate full brain initialization.

Steps:
  1. Spawn neurons (Rust-side — verify counts)
  2. Spawn within-region synapses
  3. Spawn traces (100k random)
  4. Spawn cross-region synapses (from trace structure)
  5. Initialize Rust brain with all synapses
  6. Save trace store to disk

Usage:
    from brain.seed.seed_runner import seed_brain
    brain_module, trace_store = seed_brain()
"""

import os
import random
import time

import brain_core  # the Rust PyO3 module

from brain.seed.spawn_neurons import report as neuron_report, verify_neurons
from brain.seed.spawn_synapses import (
    spawn_cross_region_synapses,
    spawn_within_region_synapses,
    report as synapse_report,
)
from brain.seed.spawn_traces import report as trace_report, spawn_traces
from brain.seed.physics_traces import spawn_physics_traces
from brain.seed.relational_traces import spawn_relational_traces
from brain.seed.numbers_wiring import wire_numbers, create_number_traces
from brain.seed.reflex_wiring import wire_reflexes
from brain.structures.trace_store import TraceStore
from brain.utils.config import SIGNAL_FLOW_CONNECTIONS


def seed_brain(
    seed: int = 42,
    save_dir: str | None = None,
    verbose: bool = True,
) -> tuple:
    """Run the full seed procedure.

    Args:
        seed: random seed for reproducibility.
        save_dir: directory to save trace store (optional).
        verbose: print progress reports.

    Returns:
        (brain_core, trace_store) — the Rust brain module and Python trace store.
    """
    rng_synapses = random.Random(seed)
    rng_traces = random.Random(seed + 1)

    t0 = time.time()

    # --- Step 1: Spawn within-region synapses ---
    if verbose:
        print("Step 1: Generating within-region synapses...")
    within_synapses = spawn_within_region_synapses(rng=rng_synapses)
    if verbose:
        print(f"  Within-region synapses: {len(within_synapses):,}")

    # --- Step 2: Spawn traces ---
    if verbose:
        print("Step 2: Generating traces...")
    trace_store = spawn_traces(rng=rng_traces)
    if verbose:
        trace_report(trace_store)

    # --- Step 2b: Add physics seed traces ---
    if verbose:
        print("Step 2b: Adding physics seed traces...")
    n_physics = spawn_physics_traces(trace_store, rng=random.Random(seed + 10))
    if verbose:
        print(f"  Physics traces: {n_physics}")

    # --- Step 2c: Add relational seed traces ---
    if verbose:
        print("Step 2c: Adding relational seed traces...")
    n_relational = spawn_relational_traces(trace_store, rng=random.Random(seed + 11))
    if verbose:
        print(f"  Relational traces: {n_relational}")

    # --- Step 2d: Add number traces ---
    if verbose:
        print("Step 2d: Adding number traces (0–99)...")
    n_numbers = create_number_traces(trace_store)
    if verbose:
        print(f"  Number traces: {n_numbers}")

    # --- Step 3: Spawn cross-region synapses from trace structure ---
    if verbose:
        print("Step 3: Generating cross-region synapses...")
    traces_neurons = [t.neurons for t in trace_store.traces.values()]
    cross_synapses = spawn_cross_region_synapses(
        traces_neurons=traces_neurons,
        flow_connections=SIGNAL_FLOW_CONNECTIONS,
        rng=rng_synapses,
    )
    if verbose:
        print(f"  Cross-region synapses: {len(cross_synapses):,}")

    # --- Step 4: Initialize Rust brain with all synapses ---
    all_synapses = within_synapses + cross_synapses

    # --- Step 4b: Wire hardwired numbers region ---
    if verbose:
        print("Step 4b: Wiring numbers region (0–99)...")
    number_synapses = wire_numbers(rng=random.Random(seed + 20))
    all_synapses.extend(number_synapses)
    if verbose:
        print(f"  Number synapses: {len(number_synapses):,}")

    # --- Step 4c: Wire reflex pathways ---
    if verbose:
        print("Step 4c: Wiring reflex pathways (sensory→motor)...")
    reflex_synapses = wire_reflexes(rng=random.Random(seed + 21))
    all_synapses.extend(reflex_synapses)
    if verbose:
        print(f"  Reflex synapses: {len(reflex_synapses):,}")

    if verbose:
        print(f"Step 4d: Initializing Rust brain with {len(all_synapses):,} total synapses...")
    brain_core.init_brain_with_synapses(all_synapses)

    # --- Step 5: Verify neuron counts ---
    if verbose:
        print("Step 5: Verifying neuron counts...")
    counts = verify_neurons(brain_core)
    if verbose:
        neuron_report(counts)
        synapse_report(len(all_synapses))

    # --- Step 6: Optionally save ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        trace_path = os.path.join(save_dir, "traces.json")
        if verbose:
            print(f"Step 6: Saving traces to {trace_path}...")
        trace_store.save(trace_path)

    elapsed = time.time() - t0
    if verbose:
        print(f"\nSeed complete in {elapsed:.1f}s")
        print(f"  Neurons:  {brain_core.get_neuron_count():>10,}")
        print(f"  Synapses: {brain_core.get_synapse_count():>10,}")
        print(f"  Traces:   {len(trace_store):>10,}")

    return brain_core, trace_store


def seed_brain_fast(
    n_traces: int = 5000,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """Lightweight seed for fast interactive learning (fewer traces, fewer synapses).

    Args:
        n_traces: number of random traces (default 5000 instead of 100k).
        seed: random seed.
        verbose: print progress.

    Returns:
        (brain_core, trace_store)
    """
    rng_synapses = random.Random(seed)
    rng_traces = random.Random(seed + 1)

    t0 = time.time()

    # Within-region synapses (same as full)
    if verbose:
        print("Fast seed: within-region synapses...")
    within_synapses = spawn_within_region_synapses(rng=rng_synapses)

    # Fewer traces
    if verbose:
        print(f"Fast seed: {n_traces} traces...")
    trace_store = spawn_traces(count=n_traces, rng=rng_traces)

    # Physics + relational + number traces (small, always include)
    spawn_physics_traces(trace_store, rng=random.Random(seed + 10))
    spawn_relational_traces(trace_store, rng=random.Random(seed + 11))
    create_number_traces(trace_store)

    # Cross-region synapses (much fewer with fewer traces)
    if verbose:
        print("Fast seed: cross-region synapses...")
    traces_neurons = [t.neurons for t in trace_store.traces.values()]
    cross_synapses = spawn_cross_region_synapses(
        traces_neurons=traces_neurons,
        flow_connections=SIGNAL_FLOW_CONNECTIONS,
        rng=rng_synapses,
    )

    all_synapses = within_synapses + cross_synapses
    number_synapses = wire_numbers(rng=random.Random(seed + 20))
    reflex_synapses = wire_reflexes(rng=random.Random(seed + 21))
    all_synapses.extend(number_synapses)
    all_synapses.extend(reflex_synapses)

    if verbose:
        print(f"Fast seed: init Rust with {len(all_synapses):,} synapses...")
    brain_core.init_brain_with_synapses(all_synapses)

    elapsed = time.time() - t0
    if verbose:
        print(f"Fast seed done in {elapsed:.1f}s — "
              f"{brain_core.get_neuron_count()} neurons, "
              f"{brain_core.get_synapse_count():,} synapses, "
              f"{len(trace_store)} traces")

    return brain_core, trace_store
