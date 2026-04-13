"""Binding formation: detect co-active patterns across regions and create bindings.

Conditions for forming a binding:
    1. Two patterns in different regions co-activate within the configured
         effective horizon (`BINDING_TEMPORAL_WINDOW * BINDING_FORMATION_COUNT`)
    2. This co-activation happens at least `BINDING_FORMATION_COUNT` times
    3. No existing binding already covers this pair

Bindings link cross-region patterns into unified concepts.
E.g., "red" (visual) + "ball" (pattern) → one object.
"""

from __future__ import annotations

import itertools

import brain_core

from brain.structures.brain_state import ActivationSnapshot, ActivationHistory
from brain.structures.trace_store import TraceStore
from brain.utils.config import (
    BINDING_CANDIDATE_AUDIO_CROSS_MODAL_RESERVE,
    BINDING_CANDIDATE_BUDGET,
    BINDING_DISSOLUTION_MIN_FIRES,
    BINDING_DISSOLUTION_WEIGHT,
    BINDING_FORMATION_COUNT,
    BINDING_TEMPORAL_WINDOW,
    TRACE_ACTIVATION_THRESHOLD,
)


_BINDING_TRACKER_ID_COUNTER = itertools.count(1)
_BINDING_PRIORITY_MODALITIES = frozenset({"audio", "text", "visual"})


def _trace_modalities(trace) -> set[str]:
    regions = {region for region, neurons in trace.neurons.items() if neurons}
    modalities = set()
    if "audio" in regions:
        modalities.add("audio")
    if "visual" in regions:
        modalities.add("visual")
    if "language" in regions or "memory_long" in regions or "speech" in regions:
        modalities.add("text")
    if trace.id.startswith("number_") or "numbers" in regions:
        modalities.add("numbers")
    if not modalities:
        modalities.add("other")
    return modalities


def _candidate_priority_bucket(trace_a, trace_b) -> int:
    modalities_a = _trace_modalities(trace_a)
    modalities_b = _trace_modalities(trace_b)
    pair_modalities = (modalities_a | modalities_b) & _BINDING_PRIORITY_MODALITIES

    if "audio" in pair_modalities and len(pair_modalities) >= 2:
        return 0
    if len(pair_modalities) >= 2:
        return 1
    if modalities_a == {"text"} and modalities_b == {"text"}:
        return 3
    return 2


def _bound_pair_key(
    trace_id_a: str,
    region_a: str,
    trace_id_b: str,
    region_b: str,
) -> tuple[str, str, str, str]:
    left = (trace_id_a, region_a)
    right = (trace_id_b, region_b)
    if left <= right:
        return trace_id_a, region_a, trace_id_b, region_b
    return trace_id_b, region_b, trace_id_a, region_a


class CoActivationTracker:
    """Tracks cross-region pattern co-activations within a temporal window."""

    def __init__(self):
        self._tracker_id = next(_BINDING_TRACKER_ID_COUNTER)
        brain_core.binding_tracker_create(self._tracker_id)

    def __del__(self):
        try:
            brain_core.binding_tracker_drop(self._tracker_id)
        except Exception:
            pass

    def record(
        self,
        active_traces: list[tuple[str, float]],
        trace_store: TraceStore,
        tick: int,
        history: ActivationHistory,
    ) -> list[dict[str, object]]:
        """Record co-activations and return pairs ready for binding formation.

        Returns list of (trace_id_a, region_a, trace_id_b, region_b, avg_time_delta)
        for pairs that have crossed the formation threshold.
        """
        if len(active_traces) < 2:
            return []

        return [
            {
                "trace_id_a": tid_a,
                "region_a": region_a,
                "trace_id_b": tid_b,
                "region_b": region_b,
                "avg_delta": avg_delta,
                "support_count": support_count,
                "span_ticks": span_ticks,
                "first_tick": first_tick,
                "last_tick": last_tick,
            }
            for tid_a, region_a, tid_b, region_b, avg_delta, support_count, span_ticks, first_tick, last_tick in brain_core.binding_tracker_record_detailed_from_active_traces(
                self._tracker_id,
                trace_store.store_id,
                active_traces,
                tick,
                BINDING_FORMATION_COUNT,
                BINDING_TEMPORAL_WINDOW,
            )
        ]

    def consume(self, ready_pairs: list[dict[str, object]]) -> None:
        """Acknowledge ready pairs that were formed or intentionally cleared."""
        if not ready_pairs:
            return

        brain_core.binding_tracker_consume(
            self._tracker_id,
            [
                (
                    pair["trace_id_a"],
                    pair["region_a"],
                    pair["trace_id_b"],
                    pair["region_b"],
                )
                for pair in ready_pairs
            ],
        )

    def cleanup(self, current_tick: int, max_age: int = 10000) -> None:
        """Remove old co-activation records."""
        brain_core.binding_tracker_cleanup(self._tracker_id, current_tick, max_age)

    def clear(self) -> None:
        """Clear pending co-activation evidence without touching formed bindings."""
        brain_core.binding_tracker_clear(self._tracker_id)


class BindingFormationEngine:
    """Creates and maintains bindings between cross-region patterns."""

    def __init__(
        self,
        trace_store: TraceStore,
        candidate_budget: int = BINDING_CANDIDATE_BUDGET,
        audio_cross_modal_reserve: int = BINDING_CANDIDATE_AUDIO_CROSS_MODAL_RESERVE,
    ):
        self.trace_store = trace_store
        self.tracker = CoActivationTracker()
        self.candidate_budget = max(0, candidate_budget)
        self.audio_cross_modal_reserve = max(0, audio_cross_modal_reserve)
        # Track which trace-region endpoint pairs already have bindings.
        self._bound_pairs: set[tuple[str, str, str, str]] = set()
        self._recently_formed: list[int] = []
        self._recently_formed_details: list[dict[str, object]] = []
        self._binding_details: dict[int, dict[str, object]] = {}
        self.last_step_debug: dict[str, object] = {}

    @property
    def recently_formed(self) -> list:
        return self._recently_formed

    @property
    def recently_formed_details(self) -> list[dict[str, object]]:
        return self._recently_formed_details

    @property
    def binding_details(self) -> dict[int, dict[str, object]]:
        return dict(self._binding_details)

    def _select_candidate_batch(
        self,
        prepared_candidates: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if self.candidate_budget <= 0 or len(prepared_candidates) <= self.candidate_budget:
            return prepared_candidates

        audio_cross_candidates = [
            item for item in prepared_candidates if item["priority_bucket"] == 0
        ]
        reserve = min(
            self.audio_cross_modal_reserve,
            self.candidate_budget,
            len(audio_cross_candidates),
        )
        selected = audio_cross_candidates[:reserve]
        if len(selected) >= self.candidate_budget:
            return selected

        selected_ids = {id(item) for item in selected}
        for item in prepared_candidates:
            if len(selected) >= self.candidate_budget:
                break
            if id(item) in selected_ids:
                continue
            selected.append(item)
        return selected

    def step(
        self,
        active_traces: list[tuple[str, float]],
        tick: int,
        history: ActivationHistory,
        tick_span: int = 1,
    ) -> dict:
        """Process one tick of binding formation and maintenance.

        Returns stats dict.
        """
        self._recently_formed = []
        self._recently_formed_details = []

        # Record co-activations and check for formation-ready pairs
        ready_pairs_by_key: dict[tuple[str, str, str, str], dict[str, object]] = {}
        effective_tick_span = max(1, int(tick_span))
        tick_start = tick - effective_tick_span + 1
        for tick_offset in range(effective_tick_span):
            record_tick = tick_start + tick_offset
            for pair in self.tracker.record(
                active_traces,
                self.trace_store,
                record_tick,
                history,
            ):
                pair_key = (
                    str(pair["trace_id_a"]),
                    str(pair["region_a"]),
                    str(pair["trace_id_b"]),
                    str(pair["region_b"]),
                )
                ready_pairs_by_key[pair_key] = pair
        ready_pairs = list(ready_pairs_by_key.values())
        active_scores = {trace_id: float(score) for trace_id, score in active_traces}
        prepared_candidates: list[dict[str, object]] = []
        stale_pairs: list[dict[str, object]] = []
        ready_audio_cross_modal = 0
        ready_cross_modal = 0
        ready_text_text = 0

        for pair in ready_pairs:
            trace_a = self.trace_store.get(pair["trace_id_a"])
            trace_b = self.trace_store.get(pair["trace_id_b"])
            if trace_a is None or trace_b is None:
                stale_pairs.append(pair)
                continue

            priority_bucket = _candidate_priority_bucket(trace_a, trace_b)
            if priority_bucket == 0:
                ready_audio_cross_modal += 1
                ready_cross_modal += 1
            elif priority_bucket == 1:
                ready_cross_modal += 1
            elif priority_bucket == 3:
                ready_text_text += 1

            pair_score = (
                active_scores.get(pair["trace_id_a"], 0.0)
                * active_scores.get(pair["trace_id_b"], 0.0)
            )
            prepared_candidates.append(
                {
                    "pair": pair,
                    "trace_a": trace_a,
                    "trace_b": trace_b,
                    "priority_bucket": priority_bucket,
                    "pair_score": pair_score,
                }
            )

        prepared_candidates.sort(
            key=lambda item: (
                item["priority_bucket"],
                -item["pair_score"],
                item["pair"]["span_ticks"],
                -item["pair"]["support_count"],
                item["pair"]["trace_id_a"],
                item["pair"]["region_a"],
                item["pair"]["trace_id_b"],
                item["pair"]["region_b"],
            )
        )
        selected_candidates = self._select_candidate_batch(prepared_candidates)
        candidate_count = len(selected_candidates)
        deferred_count = max(len(prepared_candidates) - candidate_count, 0)
        consumed_pairs = list(stale_pairs)

        formed = 0
        for item in selected_candidates:
            pair = item["pair"]
            trace_a = item["trace_a"]
            trace_b = item["trace_b"]
            tid_a = pair["trace_id_a"]
            region_a = pair["region_a"]
            tid_b = pair["trace_id_b"]
            region_b = pair["region_b"]
            avg_delta = pair["avg_delta"]
            pair_key = _bound_pair_key(tid_a, region_a, tid_b, region_b)
            consumed_pairs.append(pair)
            if pair_key in self._bound_pairs:
                continue  # Already bound

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
            brain_core.annotate_binding_traces(binding_id, tid_a, tid_b)

            # Record in trace metadata
            trace_a.binding_ids.append(binding_id)
            trace_b.binding_ids.append(binding_id)

            self._bound_pairs.add(pair_key)
            self._recently_formed.append(binding_id)
            detail = {
                "binding_id": binding_id,
                "trace_id_a": tid_a,
                "region_a": region_a,
                "trace_id_b": tid_b,
                "region_b": region_b,
                "avg_delta": avg_delta,
            }
            self._recently_formed_details.append(detail)
            self._binding_details[binding_id] = detail
            formed += 1

        self.tracker.consume(consumed_pairs)

        selected_audio_cross_modal = sum(
            1 for item in selected_candidates if item["priority_bucket"] == 0
        )
        selected_cross_modal = sum(
            1 for item in selected_candidates if item["priority_bucket"] in (0, 1)
        )
        self.last_step_debug = {
            "ready_pairs_total": len(ready_pairs),
            "selected_pairs_total": candidate_count,
            "deferred_pairs_total": deferred_count,
            "ready_text_text": ready_text_text,
            "ready_cross_modal": ready_cross_modal,
            "ready_audio_cross_modal": ready_audio_cross_modal,
            "selected_cross_modal": selected_cross_modal,
            "selected_audio_cross_modal": selected_audio_cross_modal,
        }

        strengthened, missed, total_bindings = brain_core.process_bindings(0.01, tick)

        return {
            "candidates": candidate_count,
            "ready_total": len(ready_pairs),
            "deferred": deferred_count,
            "ready_text_text": ready_text_text,
            "ready_cross_modal": ready_cross_modal,
            "ready_audio_cross_modal": ready_audio_cross_modal,
            "selected_cross_modal": selected_cross_modal,
            "selected_audio_cross_modal": selected_audio_cross_modal,
            "formed": formed,
            "strengthened": strengthened,
            "missed": missed,
            "total_bindings": total_bindings,
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

    def reset_sample_boundary(self) -> None:
        """Drop cross-sample co-activation carryover while preserving learned bindings."""
        self.tracker.clear()
