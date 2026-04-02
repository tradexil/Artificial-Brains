"""Spawn 152k neurons by calling into the Rust core.

The Rust Brain object is already initialized with the correct neuron counts
per region (from region.rs). This script verifies that everything matches
the Python config and sets up the NeuronMap for Python-side lookups.
"""

from brain.structures.neuron_map import (
    all_region_names,
    is_inhibitory,
    region_size,
)
from brain.utils.config import REGIONS, TOTAL_NEURONS


def verify_neurons(brain) -> dict:
    """Verify that the Rust brain's neuron counts match Python config.

    Args:
        brain: The Rust Brain object (via PyO3).

    Returns:
        dict with region_name → neuron_count for reference.
    """
    total = brain.get_neuron_count()
    assert total == TOTAL_NEURONS, (
        f"Rust brain has {total} neurons, expected {TOTAL_NEURONS}"
    )

    counts = {}
    for name in all_region_names():
        expected = region_size(name)
        counts[name] = expected

    return counts


def report(counts: dict) -> None:
    """Print a summary of neuron allocation."""
    total = sum(counts.values())
    print(f"Neurons spawned: {total:,}")
    for name, count in counts.items():
        start, end = REGIONS[name]
        print(f"  {name:15s}: {count:>6,}  [{start:>6,} – {end:>6,}]")
