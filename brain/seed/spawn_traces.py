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
)


def spawn_traces(
    rng: random.Random | None = None,
    count: int | None = None,
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

        for region_name, per_trace_count in NEURONS_PER_TRACE.items():
            if per_trace_count == 0:
                continue

            start, end = REGIONS[region_name]
            size = region_size(region_name)

            # Sample random neurons from this region
            n = min(per_trace_count, size)
            selected = rng.sample(range(start, end + 1), n)
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
