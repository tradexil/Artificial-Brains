"""Trace formation: create new traces from novel co-active patterns.

Conditions (ALL must be met):
  1. Pattern recognition reports NO MATCH above 0.6 threshold
  2. Novel pattern persists for > 20 consecutive ticks
  3. Pattern involves >= 2 regions (cross-modal = worth remembering)
  4. Novelty signal is high (prediction error significant)
  5. Working memory has capacity (< 7 active traces)

Also handles trace merging: if two traces have > 80% neuron overlap
and frequently co-activate, merge them.
"""

from __future__ import annotations

import itertools
import uuid

import brain_core

from brain.structures.brain_state import ActivationSnapshot
from brain.structures.trace_store import Trace, TraceStore
from brain.utils.config import (
    TRACE_FORMATION_MIN_REGIONS,
    TRACE_FORMATION_PERSISTENCE,
    TRACE_MERGE_OVERLAP,
    WORKING_MEMORY_CAPACITY,
)


_NOVEL_TRACKER_ID_COUNTER = itertools.count(1)


class NovelPatternTracker:
    """Tracks persistent novel activation patterns as candidates for trace formation."""

    def __init__(self):
        self._tracker_id = next(_NOVEL_TRACKER_ID_COUNTER)
        brain_core.novel_tracker_create(self._tracker_id)

    def __del__(self):
        try:
            brain_core.novel_tracker_drop(self._tracker_id)
        except Exception:
            pass

    def update(
        self,
        snapshot: ActivationSnapshot,
        active_traces: list[tuple[str, float]],
        novelty: float,
    ) -> list[dict[str, list[int]]]:
        """Track novel patterns and return any that meet formation criteria.

        Returns list of region→neuron dicts for patterns ready to become traces.
        """
        current_neurons = {
            region_name: [nid for nid, _ in neurons]
            for region_name, neurons in snapshot.active_neurons.items()
            if neurons
        }
        return brain_core.novel_tracker_update(
            self._tracker_id,
            current_neurons,
            novelty,
            TRACE_FORMATION_MIN_REGIONS,
            TRACE_FORMATION_PERSISTENCE,
        )


class TraceFormationEngine:
    """Creates new traces from persistent novel patterns and handles merging."""

    def __init__(self, trace_store: TraceStore):
        self.trace_store = trace_store
        self.tracker = NovelPatternTracker()
        self._recently_formed: list[str] = []

    @property
    def recently_formed(self) -> list[str]:
        """Trace IDs formed in the last step."""
        return self._recently_formed

    def step(
        self,
        snapshot: ActivationSnapshot,
        active_traces: list[tuple[str, float]],
        novelty: float,
        tick: int,
        working_memory_count: int,
        co_trace_ids: list[str] | None = None,
        context_tags: list[str] | None = None,
    ) -> int:
        """Check for and create new traces. Returns number created."""
        self._recently_formed = []

        # Don't form if working memory is full
        if working_memory_count >= WORKING_MEMORY_CAPACITY:
            return 0

        # Get patterns ready for formation
        ready_patterns = self.tracker.update(snapshot, active_traces, novelty)

        formed = 0
        for neurons_by_region in ready_patterns:
            if working_memory_count + formed >= WORKING_MEMORY_CAPACITY:
                break

            trace_id = f"trace_{uuid.uuid4().hex[:8]}"
            trace = Trace(
                id=trace_id,
                neurons=neurons_by_region,
                strength=0.1,
                novelty=1.0,
                decay=1.0,
                co_traces=list(co_trace_ids or []),
                context_tags=list(context_tags or []),
                formation_tick=tick,
            )
            self.trace_store.add(trace)
            self._recently_formed.append(trace_id)
            formed += 1

        return formed

    def merge_overlapping(self, min_co_fires: int = 10) -> int:
        """Merge traces with > 80% neuron overlap that frequently co-activate.

        Returns number of merges performed.
        """
        merges = 0
        trace_ids = list(self.trace_store.traces.keys())

        merged_away: set[str] = set()

        for i, tid_a in enumerate(trace_ids):
            if tid_a in merged_away:
                continue
            trace_a = self.trace_store.get(tid_a)
            if trace_a is None:
                continue

            neurons_a = set()
            for nids in trace_a.neurons.values():
                neurons_a.update(nids)

            if not neurons_a:
                continue

            for tid_b in trace_ids[i + 1:]:
                if tid_b in merged_away:
                    continue
                trace_b = self.trace_store.get(tid_b)
                if trace_b is None:
                    continue

                neurons_b = set()
                for nids in trace_b.neurons.values():
                    neurons_b.update(nids)

                if not neurons_b:
                    continue

                # Check overlap
                overlap = len(neurons_a & neurons_b)
                union = len(neurons_a | neurons_b)
                if union == 0:
                    continue
                ratio = overlap / union

                if ratio >= TRACE_MERGE_OVERLAP:
                    # Check co-activation count
                    both_fired = min(trace_a.fire_count, trace_b.fire_count)
                    if both_fired >= min_co_fires:
                        self._merge_traces(trace_a, trace_b)
                        merged_away.add(tid_b)
                        merges += 1

        # Remove merged-away traces
        for tid in merged_away:
            self.trace_store.remove(tid)

        return merges

    def _merge_traces(self, into: Trace, from_trace: Trace) -> None:
        """Merge from_trace into the 'into' trace."""
        # Merge neurons: union per region
        for region, nids in from_trace.neurons.items():
            if region not in into.neurons:
                into.neurons[region] = nids
            else:
                existing = set(into.neurons[region])
                existing.update(nids)
                into.neurons[region] = sorted(existing)

        # Merge metadata
        into.strength = max(into.strength, from_trace.strength)
        into.fire_count += from_trace.fire_count
        into.co_traces = list(set(into.co_traces) | set(from_trace.co_traces))
        into.context_tags = list(set(into.context_tags) | set(from_trace.context_tags))

        # Update inverted index for new neurons
        for nids in from_trace.neurons.values():
            for nid in nids:
                self.trace_store._neuron_to_traces[nid].add(into.id)

        self.trace_store.sync_trace(into.id)
