"""Consolidation: short-term → long-term memory transfer.

Triggered when energy < 0.2 OR every CONSOLIDATION_TRIGGER_TICKS.
Analogous to sleep: replay important traces, strengthen memory_long
neurons, strip context (episodic→semantic), reduce input gain.

Process per cycle:
  1. Sort candidates: emotional polarity > novelty > strength > fire_count
  2. For each trace: strengthen memory_long neurons + synapses
  3. Context stripping: drop some context_tags each consolidation
  4. Clear working memory after transfer
  5. During consolidation: reduce input gain (sleep-like)
"""

from __future__ import annotations

import brain_core

from brain.structures.brain_state import NeuromodulatorState
from brain.structures.trace_store import Trace, TraceStore
from brain.utils.config import (
    CONSOLIDATION_DURATION,
    CONSOLIDATION_INPUT_GAIN,
    CONSOLIDATION_TRIGGER_ENERGY,
    CONSOLIDATION_TRIGGER_TICKS,
)


class ConsolidationEngine:
    """Manages memory consolidation cycles."""

    def __init__(self):
        self._last_consolidation_tick: int = 0
        self._consolidating: bool = False
        self._consolidation_start: int = 0
        self._consolidation_queue: list[str] = []  # trace IDs to replay
        self._consolidation_idx: int = 0

    @property
    def is_consolidating(self) -> bool:
        return self._consolidating

    def should_consolidate(self, tick: int, neuromod: NeuromodulatorState) -> bool:
        """Check if consolidation should start."""
        if self._consolidating:
            return False
        energy_trigger = neuromod.energy < CONSOLIDATION_TRIGGER_ENERGY
        tick_trigger = (tick - self._last_consolidation_tick) >= CONSOLIDATION_TRIGGER_TICKS
        return energy_trigger or tick_trigger

    def start_consolidation(
        self,
        tick: int,
        trace_store: TraceStore,
        recent_trace_ids: list[str],
    ) -> int:
        """Begin a consolidation cycle. Returns number of traces queued.

        Traces are prioritized by:
          1. Absolute emotional polarity (important stuff first)
          2. Novelty (new stuff second)
          3. Strength (frequent stuff third)
        """
        trace_store.sync_runtime_state(recent_trace_ids)

        # Filter to traces that actually exist and were recently active
        candidates: list[Trace] = []
        for tid in recent_trace_ids:
            trace = trace_store.get(tid)
            if trace is not None:
                candidates.append(trace)

        # Sort by priority: polarity desc, novelty desc, strength desc
        candidates.sort(
            key=lambda t: (abs(t.polarity), t.novelty, t.strength),
            reverse=True,
        )

        self._consolidating = True
        self._consolidation_start = tick
        self._last_consolidation_tick = tick
        self._consolidation_queue = [t.id for t in candidates]
        self._consolidation_idx = 0

        # Reduce input gain during consolidation (sleep-like)
        for region in [
            "sensory", "visual", "audio",
        ]:
            brain_core.set_attention_gain(region, CONSOLIDATION_INPUT_GAIN)

        return len(self._consolidation_queue)

    def consolidation_step(self, tick: int, trace_store: TraceStore) -> dict:
        """Process one step of consolidation. Call each tick during consolidation.

        Returns stats dict.
        """
        if not self._consolidating:
            return {"active": False}

        elapsed = tick - self._consolidation_start
        if elapsed >= CONSOLIDATION_DURATION or self._consolidation_idx >= len(self._consolidation_queue):
            self._end_consolidation()
            return {"active": False, "finished": True}

        # Replay one trace per tick (spread work across the consolidation window)
        traces_per_tick = max(1, len(self._consolidation_queue) // CONSOLIDATION_DURATION + 1)
        strengthened = 0
        context_stripped = 0

        for _ in range(traces_per_tick):
            if self._consolidation_idx >= len(self._consolidation_queue):
                break

            tid = self._consolidation_queue[self._consolidation_idx]
            self._consolidation_idx += 1

            trace = trace_store.get(tid)
            if trace is None:
                continue

            # Strengthen memory_long neurons
            ml_neurons = trace.neurons.get("memory_long", [])
            if ml_neurons:
                brain_core.strengthen_memory_trace(ml_neurons, 0.3)
                strengthened += 1

            # Context stripping: remove ~20% of context_tags each consolidation
            # This is how episodic → semantic happens over time
            if trace.context_tags and len(trace.context_tags) > 1:
                n_strip = max(1, len(trace.context_tags) // 5)
                trace.context_tags = trace.context_tags[n_strip:]
                context_stripped += 1

            # Boost trace strength from consolidation
            trace.strength = min(1.0, trace.strength + 0.02)
            trace_store.sync_trace(trace.id)

        return {
            "active": True,
            "progress": self._consolidation_idx / max(1, len(self._consolidation_queue)),
            "strengthened": strengthened,
            "context_stripped": context_stripped,
        }

    def _end_consolidation(self) -> None:
        """End consolidation: restore input gains."""
        self._consolidating = False
        self._consolidation_queue = []
        self._consolidation_idx = 0

        # Restore input gains
        for region in ["sensory", "visual", "audio"]:
            brain_core.set_attention_gain(region, 1.0)
