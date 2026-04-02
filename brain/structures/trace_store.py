"""Trace store: create, load, save, query traces + inverted index.

A Trace is a concept = set of neuron IDs across regions + metadata.
The inverted index maps neuron_id → list of trace_ids for fast matching.
"""

from __future__ import annotations

import itertools
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field

import brain_core


_STORE_ID_COUNTER = itertools.count(1)


@dataclass
class Trace:
    id: str
    label: str | None = None

    # Structure — which neurons define this concept per region
    neurons: dict[str, list[int]] = field(default_factory=dict)

    # Binding IDs that connect this trace's cross-region patterns
    binding_ids: list[int] = field(default_factory=list)

    # Dynamics
    strength: float = 0.1
    decay: float = 1.0
    polarity: float = 0.0
    abstraction: float = 0.0
    novelty: float = 1.0

    # Relationships
    co_traces: list[str] = field(default_factory=list)
    context_tags: list[str] = field(default_factory=list)

    # Bookkeeping
    fire_count: int = 0
    last_fired: int = 0
    formation_tick: int = 0

    def total_neurons(self) -> int:
        return sum(len(ns) for ns in self.neurons.values())

    def regions_present(self) -> list[str]:
        return [r for r, ns in self.neurons.items() if len(ns) > 0]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "neurons": self.neurons,
            "binding_ids": self.binding_ids,
            "strength": self.strength,
            "decay": self.decay,
            "polarity": self.polarity,
            "abstraction": self.abstraction,
            "novelty": self.novelty,
            "co_traces": self.co_traces,
            "context_tags": self.context_tags,
            "fire_count": self.fire_count,
            "last_fired": self.last_fired,
            "formation_tick": self.formation_tick,
        }

    @staticmethod
    def from_dict(d: dict) -> Trace:
        return Trace(
            id=d["id"],
            label=d.get("label"),
            neurons=d.get("neurons", {}),
            binding_ids=d.get("binding_ids", []),
            strength=d.get("strength", 0.1),
            decay=d.get("decay", 1.0),
            polarity=d.get("polarity", 0.0),
            abstraction=d.get("abstraction", 0.0),
            novelty=d.get("novelty", 1.0),
            co_traces=d.get("co_traces", []),
            context_tags=d.get("context_tags", []),
            fire_count=d.get("fire_count", 0),
            last_fired=d.get("last_fired", 0),
            formation_tick=d.get("formation_tick", 0),
        )


class TraceStore:
    """Manages all traces + inverted index for fast lookup."""

    def __init__(self):
        self.traces: dict[str, Trace] = {}
        # Inverted index: neuron_id → set of trace_ids
        self._neuron_to_traces: dict[int, set[str]] = defaultdict(set)
        self._store_id = next(_STORE_ID_COUNTER)
        brain_core.trace_index_create(self._store_id)

    def __del__(self):
        try:
            brain_core.trace_index_drop(self._store_id)
        except Exception:
            pass

    @staticmethod
    def _flatten_trace_neurons(trace: Trace) -> list[int]:
        neurons: list[int] = []
        for neuron_ids in trace.neurons.values():
            neurons.extend(neuron_ids)
        return neurons

    def sync_trace(self, trace_id: str) -> None:
        trace = self.traces.get(trace_id)
        if trace is None:
            return
        brain_core.trace_index_upsert_trace(
            self._store_id,
            trace.id,
            self._flatten_trace_neurons(trace),
        )

    def clear(self) -> None:
        self.traces.clear()
        self._neuron_to_traces.clear()
        brain_core.trace_index_clear(self._store_id)

    def add(self, trace: Trace) -> None:
        """Add a trace and update inverted index."""
        self.traces[trace.id] = trace
        for neurons in trace.neurons.values():
            for nid in neurons:
                self._neuron_to_traces[nid].add(trace.id)
        self.sync_trace(trace.id)

    def remove(self, trace_id: str) -> Trace | None:
        """Remove a trace and clean inverted index."""
        trace = self.traces.pop(trace_id, None)
        if trace is None:
            return None
        for neurons in trace.neurons.values():
            for nid in neurons:
                self._neuron_to_traces[nid].discard(trace_id)
        brain_core.trace_index_remove_trace(self._store_id, trace_id)
        return trace

    def get(self, trace_id: str) -> Trace | None:
        return self.traces.get(trace_id)

    def __len__(self) -> int:
        return len(self.traces)

    def __contains__(self, trace_id: str) -> bool:
        return trace_id in self.traces

    @property
    def store_id(self) -> int:
        return self._store_id

    def traces_for_neuron(self, neuron_id: int) -> set[str]:
        """Which traces contain this neuron?"""
        return self._neuron_to_traces.get(neuron_id, set())

    def candidate_traces(self, active_neurons: list[int]) -> dict[str, int]:
        """Given a list of active neuron IDs, find candidate traces
        and their overlap count (how many of their neurons are active).

        Returns: {trace_id: active_neuron_count}
        """
        counts: dict[str, int] = defaultdict(int)
        for nid in active_neurons:
            for tid in self._neuron_to_traces.get(nid, set()):
                counts[tid] += 1
        return dict(counts)

    def matching_traces(
        self, active_neurons: list[int], threshold: float = 0.6
    ) -> list[tuple[str, float]]:
        """Find traces whose activation ratio exceeds threshold.

        Returns: [(trace_id, match_score)] sorted by score descending.
        """
        if not active_neurons:
            return []
        return brain_core.trace_index_matching_traces(
            self._store_id,
            active_neurons,
            threshold,
        )

    def save(self, path: str) -> None:
        """Save all traces to a JSON file."""
        data = [t.to_dict() for t in self.traces.values()]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load traces from a JSON file, replacing current store."""
        with open(path) as f:
            data = json.load(f)
        self.clear()
        for d in data:
            self.add(Trace.from_dict(d))

    def stats(self) -> dict:
        """Summary statistics."""
        if not self.traces:
            return {"count": 0}
        strengths = [t.strength for t in self.traces.values()]
        return {
            "count": len(self.traces),
            "avg_strength": sum(strengths) / len(strengths),
            "min_strength": min(strengths),
            "max_strength": max(strengths),
            "total_neurons_indexed": len(self._neuron_to_traces),
        }
