"""Binding formation: detect co-active patterns across regions and create bindings.

Conditions for forming a binding:
  1. Two patterns in different regions co-activate within ±5 ticks
  2. This co-activation happens > 5 times
  3. No existing binding already covers this pair

Bindings link cross-region patterns into unified concepts.
E.g., "red" (visual) + "ball" (pattern) → one object.
"""

from __future__ import annotations

from collections import defaultdict

import brain_core

from brain.structures.brain_state import ActivationSnapshot, ActivationHistory
from brain.structures.trace_store import TraceStore
from brain.utils.config import (
    BINDING_DISSOLUTION_MIN_FIRES,
    BINDING_DISSOLUTION_WEIGHT,
    BINDING_FORMATION_COUNT,
    BINDING_TEMPORAL_WINDOW,
    TRACE_ACTIVATION_THRESHOLD,
)


class CoActivationTracker:
    """Tracks cross-region pattern co-activations within a temporal window."""

    def __init__(self):
        # (trace_id_a, region_a, trace_id_b, region_b) → list of tick deltas
        self._co_activations: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)

    def record(
        self,
        active_traces: list[tuple[str, float]],
        trace_store: TraceStore,
        tick: int,
        history: ActivationHistory,
    ) -> list[tuple[str, str, str, str, float]]:
        """Record co-activations and return pairs ready for binding formation.

        Returns list of (trace_id_a, region_a, trace_id_b, region_b, avg_time_delta)
        for pairs that have crossed the formation threshold.
        """
        if len(active_traces) < 2:
            return []

        # Get currently active traces and their primary regions
        trace_regions: list[tuple[str, str, list[int]]] = []
        for tid, score in active_traces:
            trace = trace_store.get(tid)
            if trace is None:
                continue
            # Find primary region (most neurons)
            best_region = max(
                trace.neurons.items(),
                key=lambda x: len(x[1]),
                default=(None, []),
            )
            if best_region[0] is not None and best_region[1]:
                trace_regions.append((tid, best_region[0], best_region[1]))

        # Record all cross-region pairs
        ready = []
        for i, (tid_a, region_a, _) in enumerate(trace_regions):
            for tid_b, region_b, _ in trace_regions[i + 1:]:
                if region_a == region_b:
                    continue  # Same region — not a cross-region binding

                # Canonical key (sorted to avoid duplicates)
                if (tid_a, region_a) > (tid_b, region_b):
                    key = (tid_b, region_b, tid_a, region_a)
                else:
                    key = (tid_a, region_a, tid_b, region_b)

                self._co_activations[key].append(tick)

                # Check if formation threshold reached
                activations = self._co_activations[key]
                if len(activations) >= BINDING_FORMATION_COUNT:
                    # Check temporal window: are they within ±WINDOW ticks of each other?
                    recent = activations[-BINDING_FORMATION_COUNT:]
                    if max(recent) - min(recent) <= BINDING_TEMPORAL_WINDOW * BINDING_FORMATION_COUNT:
                        avg_delta = 0.0  # simultaneous by default
                        ready.append((*key, avg_delta))
                        # Reset counter after formation
                        del self._co_activations[key]

        return ready

    def cleanup(self, current_tick: int, max_age: int = 10000) -> None:
        """Remove old co-activation records."""
        stale = []
        for key, ticks in self._co_activations.items():
            # Remove entries older than max_age
            self._co_activations[key] = [t for t in ticks if current_tick - t < max_age]
            if not self._co_activations[key]:
                stale.append(key)
        for key in stale:
            del self._co_activations[key]


class BindingFormationEngine:
    """Creates and maintains bindings between cross-region patterns."""

    def __init__(self, trace_store: TraceStore):
        self.trace_store = trace_store
        self.tracker = CoActivationTracker()
        # Track which trace pairs already have bindings
        self._bound_pairs: set[tuple[str, str]] = set()
        self._recently_formed: list[u32] = []

    @property
    def recently_formed(self) -> list:
        return self._recently_formed

    def step(
        self,
        active_traces: list[tuple[str, float]],
        tick: int,
        history: ActivationHistory,
    ) -> dict:
        """Process one tick of binding formation and maintenance.

        Returns stats dict.
        """
        self._recently_formed = []

        # Record co-activations and check for formation-ready pairs
        ready_pairs = self.tracker.record(
            active_traces, self.trace_store, tick, history,
        )

        formed = 0
        for tid_a, region_a, tid_b, region_b, avg_delta in ready_pairs:
            pair_key = tuple(sorted([tid_a, tid_b]))
            if pair_key in self._bound_pairs:
                continue  # Already bound

            trace_a = self.trace_store.get(tid_a)
            trace_b = self.trace_store.get(tid_b)
            if trace_a is None or trace_b is None:
                continue

            neurons_a = trace_a.neurons.get(region_a, [])
            neurons_b = trace_b.neurons.get(region_b, [])
            if not neurons_a or not neurons_b:
                continue

            # Create binding via Rust
            binding_id = brain_core.create_binding(
                region_a, neurons_a, TRACE_ACTIVATION_THRESHOLD,
                region_b, neurons_b, TRACE_ACTIVATION_THRESHOLD,
                avg_delta,
            )

            # Record in trace metadata
            trace_a.binding_ids.append(binding_id)
            trace_b.binding_ids.append(binding_id)

            self._bound_pairs.add(pair_key)
            self._recently_formed.append(binding_id)
            formed += 1

        # Evaluate existing bindings: strengthen active, weaken partial
        strengthened = 0
        missed = 0
        active_bindings = brain_core.evaluate_bindings(0.01)
        for binding_id, weight in active_bindings:
            brain_core.strengthen_binding(binding_id, tick)
            strengthened += 1

        partial_bindings = brain_core.find_partial_bindings(0.01)
        for binding_id in partial_bindings:
            brain_core.record_binding_miss(binding_id)
            missed += 1

        return {
            "formed": formed,
            "strengthened": strengthened,
            "missed": missed,
            "total_bindings": brain_core.get_binding_count(),
        }

    def periodic_prune(self) -> int:
        """Prune dissolved bindings. Call periodically."""
        pruned = brain_core.prune_bindings(
            BINDING_DISSOLUTION_WEIGHT,
            BINDING_DISSOLUTION_MIN_FIRES,
        )
        # Clean up bound_pairs for pruned bindings
        # (We can't easily track which pairs were pruned, so we just
        # accept that re-binding is possible after pruning)
        return pruned

    def periodic_cleanup(self, tick: int) -> None:
        """Clean up stale co-activation tracking data."""
        self.tracker.cleanup(tick)
