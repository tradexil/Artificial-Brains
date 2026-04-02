"""Handcraft ~200 physics seed traces.

Physics traces encode basic spatial/physical concepts:
  gravity, collision, weight, temperature, distance, speed, etc.

Each trace activates neurons across relevant regions:
  sensory (physical properties), visual (spatial features),
  pattern (abstract patterns), language (symbol names),
  memory_long (persistent storage).
"""

from __future__ import annotations

import random

from brain.structures.trace_store import Trace, TraceStore
from brain.utils.config import REGIONS, SEED_PHYSICS_TRACES

# Physics concepts to encode
PHYSICS_CONCEPTS = [
    # Basic forces and motion
    "gravity", "fall", "drop", "heavy", "light", "weight",
    "push", "pull", "force", "momentum", "speed", "slow",
    "fast", "acceleration", "stop", "bounce", "roll", "slide",
    "spin", "orbit", "float", "sink",
    # Spatial relations
    "up", "down", "left", "right", "near", "far",
    "inside", "outside", "above", "below", "between",
    "center", "edge", "corner", "surface", "depth",
    # Temperature and state
    "hot", "cold", "warm", "cool", "freeze", "melt",
    "boil", "evaporate", "solid", "liquid", "gas",
    # Material properties
    "hard", "soft", "rigid", "flexible", "elastic",
    "brittle", "smooth", "rough", "sharp", "dull",
    "dense", "hollow", "thick", "thin",
    # Collisions and interactions
    "collide", "crash", "break", "crack", "shatter",
    "compress", "stretch", "bend", "twist", "squeeze",
    # Energy and waves
    "energy", "vibrate", "wave", "oscillate", "resonate",
    "reflect", "absorb", "transfer",
    # Fluid dynamics
    "flow", "pour", "drip", "splash", "pressure",
    "suction", "viscous", "turbulent",
    # Light and optics
    "bright", "dark", "shadow", "transparent", "opaque",
    "reflect_light", "refract",
    # Size and scale
    "big", "small", "tiny", "huge", "tall", "short_height",
    "wide", "narrow", "long_length", "flat",
    # Balance and stability
    "balance", "stable", "unstable", "tilt", "lean",
    "topple", "stack", "support",
    # Containment
    "contain", "hold", "release", "overflow", "empty",
    "full", "fill", "wrap",
]


def spawn_physics_traces(
    store: TraceStore,
    rng: random.Random | None = None,
    count: int | None = None,
) -> int:
    """Add physics seed traces to an existing TraceStore.

    Returns number of traces created.
    """
    if rng is None:
        rng = random.Random(500)
    if count is None:
        count = SEED_PHYSICS_TRACES

    concepts = PHYSICS_CONCEPTS[:count] if count <= len(PHYSICS_CONCEPTS) else PHYSICS_CONCEPTS
    # If we need more, cycle
    while len(concepts) < count:
        concepts.append(f"physics_{len(concepts):03d}")

    created = 0
    for i, concept in enumerate(concepts[:count]):
        neurons: dict[str, list[int]] = {}

        # Sensory neurons: physics concepts have strong sensory presence
        s_start, s_end = REGIONS["sensory"]
        neurons["sensory"] = rng.sample(range(s_start, s_end + 1), 5)

        # Visual neurons: spatial/visual representation
        v_start, v_end = REGIONS["visual"]
        neurons["visual"] = rng.sample(range(v_start, v_end + 1), 8)

        # Pattern neurons: abstract pattern
        p_start, p_end = REGIONS["pattern"]
        neurons["pattern"] = rng.sample(range(p_start, p_end + 1), 4)

        # Language neurons: symbol name
        l_start, l_end = REGIONS["language"]
        neurons["language"] = rng.sample(range(l_start, l_start + 9000), 3)  # token sub-region

        # Memory_long neurons: persistent storage
        ml_start, ml_end = REGIONS["memory_long"]
        neurons["memory_long"] = rng.sample(range(ml_start, ml_end + 1), 4)

        # Motor neurons: some physics concepts have motor implications
        if concept in ("push", "pull", "grab", "throw", "drop", "release",
                       "squeeze", "stretch", "bend", "twist", "pour", "break"):
            m_start, m_end = REGIONS["motor"]
            neurons["motor"] = rng.sample(range(m_start, m_start + 8000), 3)

        # Integration neurons: cross-modal binding
        int_start, int_end = REGIONS["integration"]
        neurons["integration"] = rng.sample(range(int_start, int_end + 1), 3)

        trace = Trace(
            id=f"physics_{i:04d}",
            neurons=neurons,
            strength=rng.uniform(0.3, 0.6),
            decay=1.0,
            polarity=0.0,
            abstraction=rng.uniform(0.2, 0.6),
            novelty=0.5,
            formation_tick=0,
            label=concept,
        )
        store.add(trace)
        created += 1

    return created
