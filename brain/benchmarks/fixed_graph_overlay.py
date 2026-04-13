"""Fixed-graph overlay benchmark helpers.

Builds one structural graph plus a deterministic same-region overlay derived
from the alternate within-region seed policy. This keeps the cross-region
graph fixed so future schedule experiments can reuse a clean comparison.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import brain_core

from brain.seed.numbers_wiring import create_number_traces, wire_numbers
from brain.seed.physics_traces import spawn_physics_traces
from brain.seed.reflex_wiring import wire_reflexes
from brain.seed.relational_traces import spawn_relational_traces
from brain.seed.spawn_synapses import (
    SynapseTuple,
    spawn_cross_region_synapses,
    spawn_within_region_synapses,
)
from brain.seed.spawn_traces import spawn_traces
from brain.structures.neuron_map import region_for_neuron
from brain.structures.trace_store import Trace, TraceStore
from brain.utils.config import (
    INITIAL_TRACES,
    SEED_WITHIN_REGION_DISABLED_REGIONS,
    SIGNAL_FLOW_CONNECTIONS,
)


DEFAULT_FIXED_GRAPH_OVERLAY_REGIONS = ("emotion", "integration", "pattern")
DEFAULT_FIXED_GRAPH_OVERLAY_DELAYS = (1, 2)


@dataclass
class FixedGraphSpec:
    """Structural graph plus deterministic overlay for clean comparisons."""

    base_trace_store: TraceStore
    structural_synapses: list[SynapseTuple]
    overlay_synapses: list[SynapseTuple]
    seed: int
    chunk_count: int
    fast: bool
    n_traces: int
    overlay_regions: tuple[str, ...]
    overlay_delay_histogram: dict[int, int]
    overlay_weight_values: list[float]

    @property
    def structural_synapse_count(self) -> int:
        return len(self.structural_synapses)

    @property
    def overlay_synapse_count(self) -> int:
        return len(self.overlay_synapses)

    @property
    def overlay_graph_synapse_count(self) -> int:
        return self.structural_synapse_count + self.overlay_synapse_count

    def clone_trace_store(self) -> TraceStore:
        cloned = TraceStore()
        for trace in self.base_trace_store.traces.values():
            cloned.add(Trace.from_dict(trace.to_dict()))
        return cloned


def _build_trace_store(
    seed: int,
    fast: bool,
    n_traces: int,
    chunk_count: int,
) -> TraceStore:
    trace_store = spawn_traces(
        count=n_traces if fast else INITIAL_TRACES,
        rng=random.Random(seed + 1),
        chunk_count=chunk_count,
    )
    spawn_physics_traces(trace_store, rng=random.Random(seed + 10))
    spawn_relational_traces(trace_store, rng=random.Random(seed + 11))
    create_number_traces(trace_store)
    return trace_store


def build_fixed_graph_spec(
    seed: int = 42,
    fast: bool = True,
    n_traces: int = 5000,
    chunk_count: int | None = None,
    overlay_regions: tuple[str, ...] | list[str] | None = None,
    verbose: bool = True,
) -> FixedGraphSpec:
    """Build the structural graph and deterministic E/I/P overlay once.

    The overlay is generated from the alternate within-region seed policy using
    an isolated RNG stream so the cross-region graph remains identical to the
    structural baseline.
    """

    locality_chunks = max(1, chunk_count or brain_core.get_num_threads())
    normalized_overlay_regions = tuple(
        dict.fromkeys(overlay_regions or DEFAULT_FIXED_GRAPH_OVERLAY_REGIONS)
    )
    overlay_region_set = set(normalized_overlay_regions)

    if verbose:
        print(
            "Building fixed graph spec "
            f"(seed={seed}, fast={fast}, traces={n_traces if fast else INITIAL_TRACES}, "
            f"chunks={locality_chunks})..."
        )

    rng_synapses = random.Random(seed)
    within_synapses = spawn_within_region_synapses(
        rng=rng_synapses,
        chunk_count=locality_chunks,
    )

    trace_store = _build_trace_store(
        seed=seed,
        fast=fast,
        n_traces=n_traces,
        chunk_count=locality_chunks,
    )
    traces_neurons = [trace.neurons for trace in trace_store.traces.values()]
    cross_synapses = spawn_cross_region_synapses(
        traces_neurons=traces_neurons,
        flow_connections=SIGNAL_FLOW_CONNECTIONS,
        rng=rng_synapses,
        chunk_count=locality_chunks,
    )
    number_synapses = wire_numbers(rng=random.Random(seed + 20))
    reflex_synapses = wire_reflexes(rng=random.Random(seed + 21))
    structural_synapses = (
        within_synapses + cross_synapses + number_synapses + reflex_synapses
    )

    alternate_disabled_regions = tuple(
        region_name
        for region_name in SEED_WITHIN_REGION_DISABLED_REGIONS
        if region_name not in overlay_region_set
    )
    alternate_within_synapses = spawn_within_region_synapses(
        rng=random.Random(seed),
        chunk_count=locality_chunks,
        disabled_regions=alternate_disabled_regions,
    )
    overlay_synapses = [
        synapse
        for synapse in alternate_within_synapses
        if region_for_neuron(synapse[0]) in overlay_region_set
    ]

    overlay_delay_histogram = {
        int(delay): count
        for delay, count in sorted(Counter(delay for *_rest, delay, _plasticity in overlay_synapses).items())
    }
    overlay_weight_values = sorted({weight for *_prefix, weight, _delay, _plasticity in overlay_synapses})

    if verbose:
        print(f"  Structural synapses: {len(structural_synapses):,}")
        print(f"  Overlay synapses:    {len(overlay_synapses):,}")
        print(f"  Overlay weights:     {overlay_weight_values}")
        print(f"  Overlay delays:      {overlay_delay_histogram}")

    return FixedGraphSpec(
        base_trace_store=trace_store,
        structural_synapses=structural_synapses,
        overlay_synapses=overlay_synapses,
        seed=seed,
        chunk_count=locality_chunks,
        fast=fast,
        n_traces=n_traces,
        overlay_regions=normalized_overlay_regions,
        overlay_delay_histogram=overlay_delay_histogram,
        overlay_weight_values=overlay_weight_values,
    )