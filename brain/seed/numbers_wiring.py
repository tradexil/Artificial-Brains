"""Hardwire the numbers region (150000–151999).

The numbers region is unique: it's hardwired, not learned.
Each number (0–99) gets a fixed cluster of 20 neurons.
Numbers are connected to each other with successor/predecessor synapses,
and have cross-region connections to language (word forms).

Layout:
  Number N → neurons [150000 + N*20, 150000 + N*20 + 19]
  100 numbers × 20 neurons = 2000 neurons (full region)
"""

from __future__ import annotations

import random

import brain_core

from brain.structures.trace_store import Trace, TraceStore
from brain.utils.config import REGIONS

_NUM_START = REGIONS["numbers"][0]  # 150000
_NEURONS_PER_NUMBER = 20
_MAX_NUMBER = 100  # 0–99


def number_neurons(n: int) -> list[int]:
    """Get the neuron IDs for a given number (0–99)."""
    if n < 0 or n >= _MAX_NUMBER:
        return []
    base = _NUM_START + n * _NEURONS_PER_NUMBER
    return list(range(base, base + _NEURONS_PER_NUMBER))


def wire_numbers(rng: random.Random | None = None) -> list[tuple[int, int, float, int, float]]:
    """Create hardwired synapses for the numbers region.

    Connections:
      - Within-number: neurons in same cluster fully connect (recurrent)
      - Successor: N → N+1 (weak forward connection)
      - Predecessor: N → N-1 (weak backward connection)

    Returns list of synapse tuples: (from, to, weight, delay, plasticity).
    Plasticity is 0.0 (hardwired, not learnable).
    """
    if rng is None:
        rng = random.Random(700)

    synapses: list[tuple[int, int, float, int, float]] = []

    for n in range(_MAX_NUMBER):
        cluster = number_neurons(n)

        # Within-cluster recurrent connections (not all-to-all, ~50%)
        for i, src in enumerate(cluster):
            for j, tgt in enumerate(cluster):
                if i != j and rng.random() < 0.5:
                    synapses.append((src, tgt, 0.3, 1, 0.0))  # plasticity=0

        # Successor: N → N+1
        if n + 1 < _MAX_NUMBER:
            next_cluster = number_neurons(n + 1)
            # Connect 5 random neurons from N to 5 in N+1
            srcs = rng.sample(cluster, min(5, len(cluster)))
            tgts = rng.sample(next_cluster, min(5, len(next_cluster)))
            for s, t in zip(srcs, tgts):
                synapses.append((s, t, 0.15, 1, 0.0))

        # Predecessor: N → N-1
        if n - 1 >= 0:
            prev_cluster = number_neurons(n - 1)
            srcs = rng.sample(cluster, min(3, len(cluster)))
            tgts = rng.sample(prev_cluster, min(3, len(prev_cluster)))
            for s, t in zip(srcs, tgts):
                synapses.append((s, t, 0.1, 1, 0.0))

    return synapses


def create_number_traces(store: TraceStore) -> int:
    """Create traces for numbers 0–99 in the trace store.

    Each number trace has neurons in the numbers region and
    language token neurons (for the word form).
    """
    lang_start = REGIONS["language"][0]
    rng = random.Random(701)

    created = 0
    for n in range(_MAX_NUMBER):
        neurons: dict[str, list[int]] = {
            "numbers": number_neurons(n),
        }

        # Language token neurons for the number word
        neurons["language"] = rng.sample(range(lang_start, lang_start + 9000), 3)

        # Memory_long for persistent number knowledge
        ml_start, ml_end = REGIONS["memory_long"]
        neurons["memory_long"] = rng.sample(range(ml_start, ml_end + 1), 2)

        # Number name
        if n < 20:
            names = [
                "zero", "one", "two", "three", "four", "five",
                "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                "sixteen", "seventeen", "eighteen", "nineteen",
            ]
            label = names[n]
        elif n < 100:
            tens = ["twenty", "thirty", "forty", "fifty", "sixty",
                    "seventy", "eighty", "ninety"]
            label = tens[n // 10 - 2]
            if n % 10 != 0:
                label += f"_{n % 10}"
        else:
            label = f"num_{n}"

        trace = Trace(
            id=f"number_{n:03d}",
            neurons=neurons,
            strength=0.8,  # numbers are strong (hardwired)
            decay=1.0,
            polarity=0.0,
            abstraction=0.3,
            novelty=0.0,  # not novel (innate)
            formation_tick=0,
            label=label,
        )
        store.add(trace)
        created += 1

    return created
