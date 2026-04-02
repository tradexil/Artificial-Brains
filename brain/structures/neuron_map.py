"""Neuron map: region ranges, neuron type lookup, global↔local conversion.

This is the Python-side mirror of what Rust knows about regions.
Used by seed scripts and learning modules that need region info without
calling into Rust.
"""

from brain.utils.config import REGIONS, REGION_CONFIG


def region_for_neuron(global_id: int) -> str | None:
    """Return region name for a global neuron ID, or None if out of range."""
    for name, (start, end) in REGIONS.items():
        if start <= global_id <= end:
            return name
    return None


def local_to_global(region: str, local_idx: int) -> int:
    """Convert local index within a region to global neuron ID."""
    start, _ = REGIONS[region]
    return start + local_idx


def global_to_local(region: str, global_id: int) -> int:
    """Convert global neuron ID to local index within a region."""
    start, _ = REGIONS[region]
    return global_id - start


def region_size(region: str) -> int:
    """Number of neurons in a region."""
    start, end = REGIONS[region]
    return end - start + 1


def inhibitory_range(region: str) -> tuple[int, int]:
    """Return (start, end) global IDs of inhibitory neurons in a region.

    Inhibitory neurons are placed at the END of each region's range.
    """
    start, end = REGIONS[region]
    count = region_size(region)
    pct = REGION_CONFIG[region]["inhibitory_pct"]
    inhib_count = int(count * pct)
    inhib_start = end - inhib_count + 1
    return (inhib_start, end)


def is_inhibitory(global_id: int) -> bool:
    """Check if a neuron is inhibitory based on its position in the region."""
    region = region_for_neuron(global_id)
    if region is None:
        return False
    inhib_start, inhib_end = inhibitory_range(region)
    return inhib_start <= global_id <= inhib_end


def all_region_names() -> list[str]:
    """Return all region names in order."""
    return list(REGIONS.keys())
