"""Spawn initial within-region synapses.

Creates random within-region synapses with optional region-specific density
overrides. Returns a list of raw synapse tuples:
(from, to, weight, delay, plasticity).

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
    SEED_CROSS_REGION_DEFAULT_DELAY_RANGE,
    SEED_CROSS_REGION_DELAY_RANGE_BY_CONNECTION,
    SEED_CROSS_REGION_DEFAULT_WEIGHT_RANGE,
    SEED_CROSS_REGION_WEIGHT_RANGE_BY_CONNECTION,
    INITIAL_WITHIN_REGION_SYNAPSES_PER_NEURON,
    REGIONS,
    SEED_WITHIN_REGION_DEFAULT_DELAY_RANGE,
    SEED_WITHIN_REGION_DEFAULT_WEIGHT_RANGE,
    SEED_WITHIN_REGION_DELAY_RANGE_BY_REGION,
    SEED_WITHIN_REGION_DISABLED_REGIONS,
    SEED_WITHIN_REGION_SYNAPSES_PER_NEURON_BY_REGION,
    SEED_WITHIN_REGION_WEIGHT_RANGE_BY_REGION,
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


def _within_region_delay_range(
    region_name: str,
    disabled_regions: set[str],
    delay_range_by_region: dict[str, tuple[int, int]],
) -> tuple[int, int] | None:
    if region_name in disabled_regions:
        return None
    return delay_range_by_region.get(
        region_name,
        SEED_WITHIN_REGION_DEFAULT_DELAY_RANGE,
    )


def _within_region_weight_range(
    region_name: str,
    weight_range_by_region: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    return weight_range_by_region.get(
        region_name,
        SEED_WITHIN_REGION_DEFAULT_WEIGHT_RANGE,
    )


def _within_region_synapse_count(
    region_name: str,
    synapses_per_neuron_by_region: dict[str, int],
) -> int:
    return synapses_per_neuron_by_region.get(
        region_name,
        INITIAL_WITHIN_REGION_SYNAPSES_PER_NEURON,
    )


def _cross_region_delay_range(
    src_region: str,
    dst_region: str,
) -> tuple[int, int]:
    return SEED_CROSS_REGION_DELAY_RANGE_BY_CONNECTION.get(
        (src_region, dst_region),
        SEED_CROSS_REGION_DEFAULT_DELAY_RANGE,
    )


def _cross_region_weight_range(
    src_region: str,
    dst_region: str,
) -> tuple[float, float]:
    return SEED_CROSS_REGION_WEIGHT_RANGE_BY_CONNECTION.get(
        (src_region, dst_region),
        SEED_CROSS_REGION_DEFAULT_WEIGHT_RANGE,
    )


def spawn_within_region_synapses(
    rng: random.Random | None = None,
    chunk_count: int | None = None,
    disabled_regions: tuple[str, ...] | set[str] | None = None,
    delay_range_by_region: dict[str, tuple[int, int]] | None = None,
    weight_range_by_region: dict[str, tuple[float, float]] | None = None,
    synapses_per_neuron_by_region: dict[str, int] | None = None,
) -> list[SynapseTuple]:
    """Generate random within-region synapses for all regions.

    Optional overrides allow controlled benchmark generation without mutating
    the canonical seed configuration.

    Returns list of (from_id, to_id, weight, delay, plasticity).
    """
    if rng is None:
        rng = random.Random(42)

    disabled_region_names = set(
        SEED_WITHIN_REGION_DISABLED_REGIONS
        if disabled_regions is None
        else disabled_regions
    )
    configured_delay_ranges = (
        SEED_WITHIN_REGION_DELAY_RANGE_BY_REGION
        if delay_range_by_region is None
        else delay_range_by_region
    )
    configured_weight_ranges = (
        SEED_WITHIN_REGION_WEIGHT_RANGE_BY_REGION
        if weight_range_by_region is None
        else weight_range_by_region
    )
    configured_synapse_counts = (
        SEED_WITHIN_REGION_SYNAPSES_PER_NEURON_BY_REGION
        if synapses_per_neuron_by_region is None
        else synapses_per_neuron_by_region
    )

    all_synapses: list[SynapseTuple] = []

    for name in all_region_names():
        start, end = REGIONS[name]
        size = region_size(name)

        # Numbers region is hardwired separately
        if name == "numbers":
            continue

        delay_range = _within_region_delay_range(
            name,
            disabled_region_names,
            configured_delay_ranges,
        )
        if delay_range is None:
            continue
        min_delay, max_delay = delay_range
        min_weight, max_weight = _within_region_weight_range(
            name,
            configured_weight_ranges,
        )

        synapses_per = _within_region_synapse_count(
            name,
            configured_synapse_counts,
        )
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
                weight = rng.uniform(min_weight, max_weight)
                delay = rng.randint(min_delay, max_delay)
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
                delay_min, delay_max = _cross_region_delay_range(src_region, dst_region)
                weight_min, weight_max = _cross_region_weight_range(src_region, dst_region)
                for t in targets:
                    weight = rng.uniform(weight_min, weight_max)
                    delay = rng.randint(delay_min, delay_max)
                    plasticity = 1.0
                    all_synapses.append((src, t, weight, delay, plasticity))

    return all_synapses


def report(synapse_count: int) -> None:
    print(f"Synapses spawned: {synapse_count:,}")
