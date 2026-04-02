"""Spawn initial within-region synapses.

Creates ~20 random synapses per neuron within each region (no cross-region).
Returns a list of raw synapse tuples: (from, to, weight, delay, plasticity).

Cross-region synapses are created later by spawn_traces.py, because they
follow the trace structure (which neurons belong to which trace).
"""

import random

from brain.structures.neuron_map import (
    all_region_names,
    inhibitory_range,
    is_inhibitory,
    region_size,
)
from brain.utils.config import (
    INITIAL_WITHIN_REGION_SYNAPSES_PER_NEURON,
    REGIONS,
    SEED_MIN_LOCAL_WINDOW,
    SEED_WITHIN_REGION_LOCAL_WINDOW_FRACTION,
)

# Type alias for a raw synapse tuple
SynapseTuple = tuple[int, int, float, int, float]


def _bucket_index(local_idx: int, size: int, chunk_count: int) -> int:
    return min(chunk_count - 1, local_idx * chunk_count // size)


def _bucket_bounds(size: int, bucket_idx: int, bucket_count: int) -> tuple[int, int]:
    bucket_count = max(1, min(bucket_count, size))
    bucket_size = max(1, (size + bucket_count - 1) // bucket_count)
    bucket_start = min(size - 1, bucket_idx * bucket_size)
    bucket_end = min(size, bucket_start + bucket_size)
    return bucket_start, bucket_end


def spawn_within_region_synapses(
    rng: random.Random | None = None,
    chunk_count: int | None = None,
) -> list[SynapseTuple]:
    """Generate random within-region synapses for all regions.

    Returns list of (from_id, to_id, weight, delay, plasticity).
    """
    if rng is None:
        rng = random.Random(42)

    all_synapses: list[SynapseTuple] = []

    for name in all_region_names():
        start, end = REGIONS[name]
        size = region_size(name)

        # Numbers region is hardwired separately
        if name == "numbers":
            continue

        synapses_per = INITIAL_WITHIN_REGION_SYNAPSES_PER_NEURON
        for local_idx in range(size):
            global_id = start + local_idx

            # Pick random targets within the same region (no self-connections)
            targets = set()
            attempts = 0
            max_attempts = synapses_per * 3
            if chunk_count is not None and chunk_count > 1:
                source_bucket = _bucket_index(local_idx, size, chunk_count)
                bucket_local_start, bucket_local_end = _bucket_bounds(
                    size, source_bucket, chunk_count
                )
                local_start = start + bucket_local_start
                local_end = start + bucket_local_end - 1
            else:
                local_start = start
                local_end = end
            while len(targets) < synapses_per and attempts < max_attempts:
                t = rng.randint(local_start, local_end)
                if t != global_id:
                    targets.add(t)
                attempts += 1

            for t in targets:
                weight = rng.uniform(0.01, 0.15)
                delay = rng.randint(1, 2)  # within-region: low delay
                plasticity = 1.0
                all_synapses.append((global_id, t, weight, delay, plasticity))

    return all_synapses


def spawn_cross_region_synapses(
    traces_neurons: list[dict[str, list[int]]],
    flow_connections: list[tuple[str, str]],
    rng: random.Random | None = None,
    chunk_count: int | None = None,
) -> list[SynapseTuple]:
    """Generate cross-region synapses from trace structure.

    For each trace, create synapses between its neurons across regions
    following the signal flow connections.

    Args:
        traces_neurons: list of trace neuron dicts (region_name → [neuron_ids]).
        flow_connections: list of (src_region, dst_region) pairs.
        rng: random generator.

    Returns list of (from_id, to_id, weight, delay, plasticity).
    """
    if rng is None:
        rng = random.Random(42)

    all_synapses: list[SynapseTuple] = []

    for trace_neurons in traces_neurons:
        for src_region, dst_region in flow_connections:
            src_neurons = trace_neurons.get(src_region, [])
            dst_neurons = trace_neurons.get(dst_region, [])

            if not src_neurons or not dst_neurons:
                continue

            # Connect a subset: each source neuron connects to 1-2 targets
            src_start, _ = REGIONS[src_region]
            dst_start, dst_end = REGIONS[dst_region]
            dst_size = dst_end - dst_start + 1
            for src in src_neurons:
                candidate_targets = dst_neurons
                if chunk_count is not None and chunk_count > 1:
                    src_bucket = _bucket_index(src - src_start, region_size(src_region), chunk_count)
                    bucket_targets = [
                        target
                        for target in dst_neurons
                        if _bucket_index(target - dst_start, dst_size, chunk_count) == src_bucket
                    ]
                    if bucket_targets:
                        candidate_targets = bucket_targets

                n_targets = min(len(candidate_targets), rng.randint(1, 2))
                targets = rng.sample(candidate_targets, n_targets)
                for t in targets:
                    weight = rng.uniform(0.05, 0.2)
                    delay = rng.randint(3, 8)  # cross-region: higher delay
                    plasticity = 1.0
                    all_synapses.append((src, t, weight, delay, plasticity))

    return all_synapses


def report(synapse_count: int) -> None:
    print(f"Synapses spawned: {synapse_count:,}")
