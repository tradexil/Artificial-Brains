"""Spawn 100k random traces, each selecting neurons per region.

Each trace gets NEURONS_PER_TRACE[region] random neurons per region.
Traces are initialized with random strength, novelty=1.0, etc.
"""

import random
import uuid

from brain.structures.trace_store import Trace, TraceStore
from brain.structures.neuron_map import region_size
from brain.utils.config import (
    INITIAL_TRACES,
    NEURONS_PER_TRACE,
    REGIONS,
    SEED_TRACE_BUCKET_JITTER_FRACTION,
)


def _bucket_window(size: int, bucket_idx: int, bucket_count: int, jitter_fraction: float) -> tuple[int, int]:
    bucket_count = max(1, min(bucket_count, size))
    bucket_size = max(1, (size + bucket_count - 1) // bucket_count)
    bucket_start = min(size - 1, bucket_idx * bucket_size)
    bucket_end = min(size - 1, bucket_start + bucket_size - 1)
    jitter = max(1, int(bucket_size * jitter_fraction))
    window_start = max(0, bucket_start - jitter)
    window_end = min(size - 1, bucket_end + jitter)
    return window_start, window_end


def spawn_traces(
    rng: random.Random | None = None,
    count: int | None = None,
    chunk_count: int | None = None,
) -> TraceStore:
    """Generate random traces and return a populated TraceStore.

    Args:
        rng: random generator (default: seeded at 123).
        count: number of traces to create (default: INITIAL_TRACES).

    Returns:
        TraceStore populated with all generated traces.
    """
    if rng is None:
        rng = random.Random(123)
    if count is None:
        count = INITIAL_TRACES

    store = TraceStore()

    for i in range(count):
        neurons: dict[str, list[int]] = {}
        anchor_bucket = None
        if chunk_count is not None and chunk_count > 1:
            anchor_bucket = rng.randrange(chunk_count)

        for region_name, per_trace_count in NEURONS_PER_TRACE.items():
            if per_trace_count == 0:
                continue

            start, end = REGIONS[region_name]
            size = region_size(region_name)

            # Sample random neurons from this region
            n = min(per_trace_count, size)
            if anchor_bucket is None:
                selected = rng.sample(range(start, end + 1), n)
            else:
                window_start, window_end = _bucket_window(
                    size,
                    anchor_bucket,
                    chunk_count,
                    SEED_TRACE_BUCKET_JITTER_FRACTION,
                )
                selected_local = rng.sample(range(window_start, window_end + 1), n)
                selected = [start + local_idx for local_idx in selected_local]
            neurons[region_name] = selected

        trace = Trace(
            id=f"trace_{i:06d}",
            neurons=neurons,
            strength=rng.uniform(0.1, 0.3),
            decay=1.0,
            polarity=0.0,
            abstraction=rng.uniform(0.0, 0.4),
            novelty=1.0,
            formation_tick=0,
        )

        store.add(trace)

    return store


def report(store: TraceStore) -> None:
    stats = store.stats()
    print(f"Traces spawned:   {stats['count']:,}")
    print(f"  avg strength:   {stats['avg_strength']:.3f}")
    print(f"  neurons indexed: {stats['total_neurons_indexed']:,}")
