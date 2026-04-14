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

import heapq
import itertools
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass

import brain_core

from brain.input.visual_input import visual_family_for_neuron
from brain.structures.brain_state import ActivationHistory, ActivationSnapshot
from brain.structures.neuron_map import global_to_local, region_for_neuron
from brain.structures.trace_store import Trace, TraceStore
from brain.utils.config import (
    NEURONS_PER_TRACE,
    REGION_CONFIG,
    REGIONS,
    TRACE_FORMATION_BASELINE_WINDOW,
    TRACE_FORMATION_EXCLUDED_REGIONS,
    TRACE_FORMATION_EXCLUDED_REGION_PREFIX_NEURONS,
    TRACE_FORMATION_JACCARD_THRESHOLD,
    TRACE_FORMATION_MAX_NEURONS_PER_REGION,
    TRACE_FORMATION_MAX_TOTAL_NEURONS,
    TRACE_FORMATION_MIN_REGIONS,
    TRACE_FORMATION_MIN_BASELINE_DELTA,
    TRACE_FORMATION_MIN_BASELINE_RATIO,
    TRACE_FORMATION_PERSISTENCE,
    TRACE_FORMATION_REGION_MAX_NEURONS,
    TRACE_FORMATION_VISUAL_FAMILY_MAX_NEURONS,
    TRACE_MERGE_OVERLAP,
    WORKING_MEMORY_CAPACITY,
)


_NOVEL_TRACKER_ID_COUNTER = itertools.count(1)
_TRACE_FORMATION_NOVELTY_THRESHOLD = 0.02
_VISUAL_CANDIDATE_LOCK_REGIONS = frozenset(
    {"pattern", "integration", "language", "memory_long", "emotion"}
)
_VISUAL_CANDIDATE_LOCK_VISUAL_REGION = "visual"
_AUDIO_CANDIDATE_AUDIO_REGION = "audio"
_VISUAL_CANDIDATE_LOCK_DELAY_TICKS = 2
_VISUAL_SCENE_CHANGE_JACCARD_THRESHOLD = 0.5
_AUDIO_FAMILY_MIN_NEURONS = {
    "frequency": 4,
    "temporal": 4,
    "complex": 4,
}
_AUDIO_SELECTION_DEBUG_TOP_N = 8
_TRACE_MODALITY_FAMILY_BY_REGION = {
    "audio": "audio",
    "language": "text",
    "motor": "motor",
    "numbers": "numbers",
    "sensory": "sensory",
    "speech": "text",
    "visual": "visual",
}
_ORDERED_REGION_BOUNDS: tuple[tuple[int, str], ...] = tuple(
    (end, region_name)
    for region_name, (_start, end) in sorted(
        REGIONS.items(),
        key=lambda item: item[1][0],
    )
)


def _audio_family_for_neuron(neuron_id: int) -> str | None:
    if region_for_neuron(neuron_id) != _AUDIO_CANDIDATE_AUDIO_REGION:
        return None

    local_id = global_to_local(_AUDIO_CANDIDATE_AUDIO_REGION, neuron_id)
    if 0 <= local_id < 5000:
        return "frequency"
    if 5000 <= local_id < 10000:
        return "temporal"
    if 10000 <= local_id < 15000:
        return "complex"
    return None


def _group_active_values_by_region(
    active_values: list[tuple[int, float]],
) -> dict[str, list[tuple[int, float]]]:
    region_values: dict[str, list[tuple[int, float]]] = defaultdict(list)
    if not active_values:
        return region_values

    previous_neuron_id = -1
    bound_index = 0
    upper_bound, region_name = _ORDERED_REGION_BOUNDS[bound_index]
    for neuron_id, activation in active_values:
        if neuron_id < previous_neuron_id:
            region_values.clear()
            for fallback_neuron_id, fallback_activation in active_values:
                fallback_region = region_for_neuron(fallback_neuron_id)
                if fallback_region is not None:
                    region_values[fallback_region].append(
                        (fallback_neuron_id, fallback_activation)
                    )
            return region_values

        previous_neuron_id = neuron_id
        while neuron_id > upper_bound and bound_index + 1 < len(_ORDERED_REGION_BOUNDS):
            bound_index += 1
            upper_bound, region_name = _ORDERED_REGION_BOUNDS[bound_index]

        if neuron_id <= upper_bound:
            region_values[region_name].append((neuron_id, activation))

    return region_values


def _family_reserve_limits(
    family_limits: dict[str, int],
    region_limit: int,
) -> dict[str, int]:
    reserve_limits = {family_name: 0 for family_name in family_limits}
    if region_limit <= 0:
        return reserve_limits

    total_reserve = sum(max(0, limit) for limit in family_limits.values())
    if total_reserve <= region_limit:
        return {
            family_name: max(0, limit)
            for family_name, limit in family_limits.items()
        }

    family_names = [
        family_name
        for family_name, limit in family_limits.items()
        if limit > 0
    ]
    family_index = 0
    slots_remaining = region_limit
    while family_names and slots_remaining > 0:
        family_name = family_names[family_index % len(family_names)]
        if reserve_limits[family_name] < family_limits[family_name]:
            reserve_limits[family_name] += 1
            slots_remaining -= 1
        family_index += 1
    return reserve_limits


@dataclass
class _VisualProvisionalCandidate:
    snapshot: ActivationSnapshot
    visual_ids: tuple[int, ...]
    eligible_ticks: int = 1


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

    def clear(self) -> None:
        """Drop any pending candidate state without destroying the tracker."""
        brain_core.novel_tracker_clear(self._tracker_id)

    _UPDATE_FROM_BRAIN_MIN_ACTIVATION = 0.01
    _LIGHTWEIGHT_MIN_ACTIVATION = 0.15

    def update_from_brain(self, novelty: float) -> list[dict[str, list[int]]]:
        """Track novel patterns and return any that meet formation criteria.

        Returns list of region→neuron dicts for patterns ready to become traces.
        """
        return brain_core.novel_tracker_update_from_brain(
            self._tracker_id,
            novelty,
            TRACE_FORMATION_MIN_REGIONS,
            TRACE_FORMATION_PERSISTENCE,
            self._UPDATE_FROM_BRAIN_MIN_ACTIVATION,
            TRACE_FORMATION_JACCARD_THRESHOLD,
        )

    def update_from_brain_lightweight(self, novelty: float) -> list[dict[str, list[int]]]:
        """Like update_from_brain but with a higher activation threshold.

        Used by the lightweight tracker path to get a smaller, more stable
        neuron fingerprint that persists better across consecutive ticks.
        """
        return brain_core.novel_tracker_update_from_brain(
            self._tracker_id,
            novelty,
            TRACE_FORMATION_MIN_REGIONS,
            TRACE_FORMATION_PERSISTENCE,
            self._LIGHTWEIGHT_MIN_ACTIVATION,
            TRACE_FORMATION_JACCARD_THRESHOLD,
        )

    def update_from_snapshot(
        self,
        snapshot: ActivationSnapshot,
        novelty: float,
    ) -> list[dict[str, list[int]]]:
        """Track novel patterns from an explicit snapshot.

        This keeps unit tests and any offline callers working even though the
        hot tick path now reads activations directly from Rust.
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
            TRACE_FORMATION_JACCARD_THRESHOLD,
        )


class TraceFormationEngine:
    """Creates new traces from persistent novel patterns and handles merging."""

    def __init__(self, trace_store: TraceStore):
        self.trace_store = trace_store
        self.tracker = NovelPatternTracker()
        self._recently_formed: list[str] = []
        self.last_step_debug: dict[str, object] = {}
        self.last_tracker_snapshot: ActivationSnapshot | None = None
        self._visual_candidate_lock: dict[str, list[tuple[int, float]]] | None = None
        self._visual_provisional_candidate: _VisualProvisionalCandidate | None = None
        self._visual_lock_scene_ids: tuple[int, ...] | None = None
        self._visual_quality_scores: dict[int, float] = {}
        self._visual_quality_families: set[str] = set()
        self._audio_quality_scores: dict[int, float] = {}
        self._audio_quality_families: set[str] = set()
        self._visual_candidate_lock_enabled = True
        self.last_audio_family_selection_debug: dict[str, object] = {}

    def lightweight_tracker_update(self, novelty: float) -> int:
        """Feed the Rust-side tracker from brain activations without the full step() path.

        Uses a higher activation threshold to get a smaller, more stable
        fingerprint (strongly-active neurons persist better across ticks).
        Returns the number of traces created.
        """
        if novelty < _TRACE_FORMATION_NOVELTY_THRESHOLD:
            return 0
        ready_patterns = self.tracker.update_from_brain_lightweight(novelty)
        if not ready_patterns:
            return 0
        formed = 0
        for neurons_by_region in ready_patterns:
            if formed >= 1:  # cap lightweight formation per tick
                break
            trace_id = f"trace_{uuid.uuid4().hex[:8]}"
            trace = Trace(
                id=trace_id,
                neurons=neurons_by_region,
                strength=0.1,
                novelty=1.0,
                decay=1.0,
            )
            self.trace_store.add(trace)
            self._recently_formed.append(trace_id)
            formed += 1
        if formed > 0:
            self.tracker.clear()
        return formed

    def lightweight_tracker_update_no_formation(self, novelty: float) -> None:
        """Feed the Rust-side tracker without forming any traces.

        Keeps the tracker accumulating persistence so patterns are
        ready when step() next runs (e.g. during rest ticks when WM dips).
        Uses the same update_from_brain path as step() to ensure
        fingerprint compatibility for Jaccard continuity.
        """
        if novelty < _TRACE_FORMATION_NOVELTY_THRESHOLD:
            return
        self.tracker.update_from_brain(novelty)

    def set_visual_candidate_lock_enabled(self, enabled: bool) -> None:
        self._visual_candidate_lock_enabled = bool(enabled)
        if not self._visual_candidate_lock_enabled:
            self._clear_visual_candidate_state(clear_tracker=False)

    def set_visual_quality_scores(
        self,
        quality_scores: dict[int, float],
        *,
        families: tuple[str, ...] | None = None,
    ) -> None:
        self._visual_quality_scores = {
            int(neuron_id): float(score)
            for neuron_id, score in quality_scores.items()
            if float(score) > 0.0
        }
        family_names = families or tuple(TRACE_FORMATION_VISUAL_FAMILY_MAX_NEURONS)
        self._visual_quality_families = {
            family_name
            for family_name in family_names
            if family_name in TRACE_FORMATION_VISUAL_FAMILY_MAX_NEURONS
        }

    def clear_visual_quality_scores(self) -> None:
        self._visual_quality_scores = {}
        self._visual_quality_families = set()

    def set_audio_quality_scores(
        self,
        quality_scores: dict[int, float],
        *,
        families: tuple[str, ...] | None = None,
    ) -> None:
        self._audio_quality_scores = {
            int(neuron_id): float(score)
            for neuron_id, score in quality_scores.items()
            if float(score) > 0.0
        }
        family_names = families or tuple(_AUDIO_FAMILY_MIN_NEURONS)
        self._audio_quality_families = {
            family_name
            for family_name in family_names
            if family_name in _AUDIO_FAMILY_MIN_NEURONS
        }

    def clear_audio_quality_scores(self) -> None:
        self._audio_quality_scores = {}
        self._audio_quality_families = set()

    def _preferred_visual_ids(
        self,
        region_candidates: list[tuple[int, float, float, int]],
    ) -> set[int]:
        if not region_candidates:
            return set()

        if self._visual_provisional_candidate is not None:
            return set(self._visual_provisional_candidate.visual_ids)

        if self._visual_lock_scene_ids is not None:
            return set(self._visual_lock_scene_ids)

        return set()

    def _select_visual_family_candidates(
        self,
        region_candidates: list[tuple[int, float, float, int]],
        region_limit: int,
    ) -> list[tuple[int, float, float, int]]:
        family_candidates: dict[str, list[tuple[int, float, float, int]]] = {
            family_name: []
            for family_name in TRACE_FORMATION_VISUAL_FAMILY_MAX_NEURONS
        }

        for candidate in region_candidates:
            family_name = visual_family_for_neuron(candidate[3])
            if family_name in family_candidates:
                family_candidates[family_name].append(candidate)

        if not any(family_candidates.values()):
            return region_candidates[:region_limit]

        preferred_ids = self._preferred_visual_ids(region_candidates)
        selected: list[tuple[int, float, float, int]] = []
        selected_ids: set[int] = set()
        for family_name, family_limit in TRACE_FORMATION_VISUAL_FAMILY_MAX_NEURONS.items():
            family_pool = family_candidates[family_name]
            family_reserve = max(1, family_limit // 2)
            quality_enabled = (
                family_name in self._visual_quality_families
                and bool(self._visual_quality_scores)
            )
            if quality_enabled:
                family_pool = sorted(
                    family_pool,
                    key=lambda candidate: (
                        self._visual_quality_scores.get(candidate[3], 0.0),
                        1 if candidate[3] in preferred_ids else 0,
                        candidate[1],
                        candidate[2],
                        -candidate[3],
                    ),
                    reverse=True,
                )
                if self._visual_quality_scores.get(family_pool[0][3], 0.0) > 0.0:
                    family_selected = family_pool[:family_reserve]
                else:
                    family_preferred = [
                        candidate for candidate in family_pool if candidate[3] in preferred_ids
                    ]
                    family_other = [
                        candidate for candidate in family_pool if candidate[3] not in preferred_ids
                    ]
                    family_selected = family_preferred[:family_reserve]
                    if len(family_selected) < family_reserve:
                        family_selected.extend(
                            family_other[: family_reserve - len(family_selected)]
                        )
            else:
                family_preferred = [
                    candidate for candidate in family_pool if candidate[3] in preferred_ids
                ]
                family_other = [
                    candidate for candidate in family_pool if candidate[3] not in preferred_ids
                ]
                family_selected = family_preferred[:family_reserve]
                if len(family_selected) < family_reserve:
                    family_selected.extend(
                        family_other[: family_reserve - len(family_selected)]
                    )
            selected.extend(family_selected)
            selected_ids.update(candidate[3] for candidate in family_selected)

        if len(selected) < region_limit:
            remaining_preferred = [
                candidate
                for candidate in region_candidates
                if candidate[3] not in selected_ids and candidate[3] in preferred_ids
            ]
            remaining_other = [
                candidate
                for candidate in region_candidates
                if candidate[3] not in selected_ids and candidate[3] not in preferred_ids
            ]
            for candidate in itertools.chain(remaining_preferred, remaining_other):
                if candidate[3] in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate[3])
                if len(selected) >= region_limit:
                    break
        return selected[:region_limit]

    def _select_audio_family_candidates(
        self,
        region_candidates: list[tuple[int, float, float, int]],
        region_limit: int,
    ) -> list[tuple[int, float, float, int]]:
        family_candidates: dict[str, list[tuple[int, float, float, int]]] = {
            family_name: []
            for family_name in _AUDIO_FAMILY_MIN_NEURONS
        }

        for candidate in region_candidates:
            family_name = _audio_family_for_neuron(candidate[3])
            if family_name in family_candidates:
                family_candidates[family_name].append(candidate)

        if not any(family_candidates.values()):
            selected_candidates = region_candidates[:region_limit]
            self.last_audio_family_selection_debug = {
                "region_limit": region_limit,
                "reserve_limits": {},
                "families": {},
                "selected_audio_ids": [candidate[3] for candidate in selected_candidates],
            }
            return selected_candidates

        selected: list[tuple[int, float, float, int]] = []
        selected_ids: set[int] = set()
        reserve_limits = _family_reserve_limits(
            _AUDIO_FAMILY_MIN_NEURONS,
            region_limit,
        )
        selection_debug = {
            "region_limit": region_limit,
            "reserve_limits": dict(reserve_limits),
            "families": {},
        }
        for family_name, family_limit in reserve_limits.items():
            family_pool = family_candidates[family_name]
            quality_enabled = (
                family_name in self._audio_quality_families
                and bool(self._audio_quality_scores)
            )
            if quality_enabled:
                quality_sorted = sorted(
                    family_pool,
                    key=lambda candidate: (
                        self._audio_quality_scores.get(candidate[3], 0.0),
                        candidate[0],
                        candidate[1],
                        candidate[2],
                        -candidate[3],
                    ),
                    reverse=True,
                )
                if quality_sorted and self._audio_quality_scores.get(quality_sorted[0][3], 0.0) > 0.0:
                    family_pool = quality_sorted
            family_selected = family_pool[:family_limit]
            selected.extend(family_selected)
            selected_ids.update(candidate[3] for candidate in family_selected)
            selection_debug["families"][family_name] = {
                "candidate_count": len(family_pool),
                "reserve_limit": family_limit,
                "top_candidates": [
                    {
                        "neuron_id": neuron_id,
                        "support_count": int(support_count),
                        "specificity_score": round(float(specificity), 6),
                        "activation": round(float(activation), 6),
                        **(
                            {"quality_score": round(self._audio_quality_scores.get(neuron_id, 0.0), 6)}
                            if self._audio_quality_scores.get(neuron_id, 0.0) > 0.0
                            else {}
                        ),
                    }
                    for support_count, specificity, activation, neuron_id in family_pool[
                        :_AUDIO_SELECTION_DEBUG_TOP_N
                    ]
                ],
            }

        if len(selected) < region_limit:
            for candidate in region_candidates:
                if candidate[3] in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate[3])
                if len(selected) >= region_limit:
                    break

        selected = selected[:region_limit]
        for family_name in _AUDIO_FAMILY_MIN_NEURONS:
            family_selected = [
                candidate
                for candidate in selected
                if _audio_family_for_neuron(candidate[3]) == family_name
            ]
            family_debug = selection_debug["families"].setdefault(
                family_name,
                {
                    "candidate_count": 0,
                    "reserve_limit": reserve_limits.get(family_name, 0),
                    "top_candidates": [],
                },
            )
            family_debug["selected_candidates"] = [
                {
                    "neuron_id": neuron_id,
                    "support_count": int(support_count),
                    "specificity_score": round(float(specificity), 6),
                    "activation": round(float(activation), 6),
                    **(
                        {"quality_score": round(self._audio_quality_scores.get(neuron_id, 0.0), 6)}
                        if self._audio_quality_scores.get(neuron_id, 0.0) > 0.0
                        else {}
                    ),
                }
                for support_count, specificity, activation, neuron_id in family_selected
            ]
            family_debug["selected_ids"] = [candidate[3] for candidate in family_selected]
        selection_debug["selected_audio_ids"] = [candidate[3] for candidate in selected]
        self.last_audio_family_selection_debug = selection_debug

        return selected

    @property
    def recently_formed(self) -> list[str]:
        """Trace IDs formed in the last step."""
        return self._recently_formed

    def reset_sample_boundary(self) -> None:
        """Clear sample-scoped candidate state so patterns do not bleed across inputs."""
        self._clear_visual_candidate_state(clear_tracker=True)
        self.last_tracker_snapshot = None

    @staticmethod
    def _snapshot_region_ids(
        snapshot: ActivationSnapshot,
        region_name: str,
    ) -> tuple[int, ...]:
        return tuple(
            sorted({neuron_id for neuron_id, _ in snapshot.active_neurons.get(region_name, [])})
        )

    @staticmethod
    def _id_jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        set_a = set(a)
        set_b = set(b)
        union = len(set_a | set_b)
        if union == 0:
            return 1.0
        return len(set_a & set_b) / union

    def _same_visual_scene(
        self,
        current_visual_ids: tuple[int, ...],
        reference_visual_ids: tuple[int, ...],
    ) -> bool:
        return (
            self._id_jaccard(current_visual_ids, reference_visual_ids)
            >= _VISUAL_SCENE_CHANGE_JACCARD_THRESHOLD
        )

    def _clear_visual_candidate_state(self, clear_tracker: bool = False) -> None:
        self._visual_candidate_lock = None
        self._visual_provisional_candidate = None
        self._visual_lock_scene_ids = None
        if clear_tracker:
            self.tracker.clear()

    @staticmethod
    def _snapshot_from_active_neurons(
        tick: int,
        active_neurons: dict[str, list[tuple[int, float]]],
    ) -> ActivationSnapshot:
        active_values: list[tuple[int, float]] = []
        active_ids: list[int] = []
        region_active_counts: dict[str, int] = {}
        for region_name, neurons in active_neurons.items():
            if not neurons:
                continue
            region_active_counts[region_name] = len(neurons)
            for neuron_id, activation in neurons:
                active_values.append((neuron_id, activation))
                active_ids.append(neuron_id)

        total_active = len(active_ids)
        return ActivationSnapshot(
            tick=tick,
            active_neurons=active_neurons,
            active_values=active_values,
            total_active=total_active,
            active_ids=active_ids,
            region_active_counts=region_active_counts,
        )

    def _should_start_visual_candidate_lock(
        self,
        formation_snapshot: ActivationSnapshot,
    ) -> bool:
        if self._visual_candidate_lock is not None:
            return False

        visual_neurons = formation_snapshot.active_neurons.get(
            _VISUAL_CANDIDATE_LOCK_VISUAL_REGION,
            [],
        )
        if len(visual_neurons) < TRACE_FORMATION_MAX_NEURONS_PER_REGION:
            return False

        region_counts = {
            region_name: len(neurons)
            for region_name, neurons in formation_snapshot.active_neurons.items()
            if neurons
        }
        if not region_counts:
            return False
        if region_counts.get(_VISUAL_CANDIDATE_LOCK_VISUAL_REGION, 0) < max(region_counts.values()):
            return False

        return any(
            region_name in _VISUAL_CANDIDATE_LOCK_REGIONS
            for region_name in region_counts
        )

    def _start_visual_candidate_lock(
        self,
        formation_snapshot: ActivationSnapshot,
    ) -> None:
        locked_regions: dict[str, list[tuple[int, float]]] = {}
        for region_name in _VISUAL_CANDIDATE_LOCK_REGIONS:
            neurons = formation_snapshot.active_neurons.get(region_name)
            if neurons:
                locked_regions[region_name] = list(neurons)
        self._visual_candidate_lock = locked_regions or None
        self._visual_lock_scene_ids = self._snapshot_region_ids(
            formation_snapshot,
            _VISUAL_CANDIDATE_LOCK_VISUAL_REGION,
        )

    def _apply_visual_candidate_lock(
        self,
        formation_snapshot: ActivationSnapshot,
    ) -> ActivationSnapshot:
        if self._visual_candidate_lock is None:
            return formation_snapshot

        active_neurons: dict[str, list[tuple[int, float]]] = {}
        visual_neurons = formation_snapshot.active_neurons.get(
            _VISUAL_CANDIDATE_LOCK_VISUAL_REGION,
            [],
        )
        if visual_neurons:
            active_neurons[_VISUAL_CANDIDATE_LOCK_VISUAL_REGION] = list(visual_neurons)

        if self.last_audio_family_selection_debug.get("selected_audio_ids"):
            audio_neurons = formation_snapshot.active_neurons.get(
                _AUDIO_CANDIDATE_AUDIO_REGION,
                [],
            )
            if audio_neurons:
                active_neurons[_AUDIO_CANDIDATE_AUDIO_REGION] = list(audio_neurons)

        for region_name, neurons in self._visual_candidate_lock.items():
            if neurons:
                active_neurons[region_name] = list(neurons)

        return self._snapshot_from_active_neurons(formation_snapshot.tick, active_neurons)

    def _trace_support_counts(
        self,
        active_traces: list[tuple[str, float]] | None = None,
        co_trace_ids: list[str] | None = None,
    ) -> dict[int, int]:
        """Count how many current content traces support each neuron.

        Working-memory traces are the strongest signal for "what this sample is
        about" right now. When those traces already cover multiple regions, new
        trace formation should stay grounded in that supported neuron set instead
        of drifting into generic broad-hot state neurons.
        """
        ordered_trace_ids: list[str] = []
        seen_trace_ids: set[str] = set()

        for trace_id in co_trace_ids or []:
            if trace_id in seen_trace_ids or self.trace_store.get(trace_id) is None:
                continue
            ordered_trace_ids.append(trace_id)
            seen_trace_ids.add(trace_id)

        if len(ordered_trace_ids) < WORKING_MEMORY_CAPACITY:
            for trace_id, _score in active_traces or []:
                if trace_id in seen_trace_ids or self.trace_store.get(trace_id) is None:
                    continue
                ordered_trace_ids.append(trace_id)
                seen_trace_ids.add(trace_id)
                if len(ordered_trace_ids) >= WORKING_MEMORY_CAPACITY:
                    break

        support_counts: dict[int, int] = defaultdict(int)
        for trace_id in ordered_trace_ids:
            trace = self.trace_store.get(trace_id)
            if trace is None:
                continue
            for neuron_ids in trace.neurons.values():
                for neuron_id in neuron_ids:
                    support_counts[neuron_id] += 1

        return dict(support_counts)

    def _matching_existing_traces(
        self,
        formation_snapshot: ActivationSnapshot,
        *,
        source_snapshot: ActivationSnapshot | None = None,
        duplicate_guard_family: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return active trace matches that already represent this candidate snapshot."""
        if not formation_snapshot.active_ids:
            return []
        if duplicate_guard_family is None:
            duplicate_guard_family = self._duplicate_suppression_modality_family(
                source_snapshot
            )
        if duplicate_guard_family not in {"audio", "visual"}:
            return []
        matches = self.trace_store.matching_traces(
            formation_snapshot.active_ids,
            threshold=TRACE_MERGE_OVERLAP,
        )
        if duplicate_guard_family == "audio":
            matches = [
                (trace_id, score)
                for trace_id, score in matches
                if self._trace_contains_modality_family(trace_id, duplicate_guard_family)
            ]
        return matches

    def _duplicate_suppression_modality_family(
        self,
        source_snapshot: ActivationSnapshot | None,
    ) -> str | None:
        selected_audio_ids = self.last_audio_family_selection_debug.get("selected_audio_ids", [])
        if selected_audio_ids:
            return "audio"
        return self._snapshot_primary_modality_family(source_snapshot)

    @staticmethod
    def _resolve_novelty_gate(
        novelty: float,
        novelty_by_family: dict[str, float] | None,
        modality_family: str | None,
    ) -> tuple[str | None, float]:
        if modality_family == "audio" and novelty_by_family is not None:
            family_novelty = novelty_by_family.get(modality_family)
            if family_novelty is not None:
                return modality_family, float(family_novelty)
        return None, float(novelty)

    def _snapshot_primary_modality_family(
        self,
        snapshot: ActivationSnapshot | None,
    ) -> str | None:
        if snapshot is None:
            return None

        family_counts: dict[str, int] = defaultdict(int)
        if snapshot.region_active_counts:
            for region_name, count in snapshot.region_active_counts.items():
                family_name = _TRACE_MODALITY_FAMILY_BY_REGION.get(region_name)
                if family_name is not None and count > 0:
                    family_counts[family_name] += int(count)
        elif snapshot.active_neurons:
            for region_name, neurons in snapshot.active_neurons.items():
                family_name = _TRACE_MODALITY_FAMILY_BY_REGION.get(region_name)
                if family_name is not None and neurons:
                    family_counts[family_name] += len(neurons)
        else:
            for neuron_id, _activation in snapshot.active_values:
                region_name = region_for_neuron(neuron_id)
                family_name = _TRACE_MODALITY_FAMILY_BY_REGION.get(region_name)
                if family_name is not None:
                    family_counts[family_name] += 1

        if not family_counts:
            return None

        return sorted(
            family_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    def _trace_contains_modality_family(
        self,
        trace_id: str,
        family_name: str,
    ) -> bool:
        trace = self.trace_store.get(trace_id)
        if trace is None:
            return False
        return any(
            _TRACE_MODALITY_FAMILY_BY_REGION.get(region_name) == family_name
            for region_name in trace.neurons
        )

    def _trace_ids_contain_modality_family(
        self,
        trace_ids: list[str] | None,
        family_name: str,
    ) -> bool:
        return any(
            self._trace_contains_modality_family(trace_id, family_name)
            for trace_id in (trace_ids or [])
        )

    def _resolve_working_memory_reserve(
        self,
        working_memory_count: int,
        co_trace_ids: list[str] | None,
        modality_family: str | None,
    ) -> tuple[str | None, bool]:
        if working_memory_count < WORKING_MEMORY_CAPACITY:
            return None, False
        if modality_family != "audio":
            return None, False
        if self._trace_ids_contain_modality_family(co_trace_ids, modality_family):
            return modality_family, False
        return modality_family, True

    def _handle_visual_scene_change(
        self,
        raw_snapshot: ActivationSnapshot,
    ) -> bool:
        current_visual_ids = self._snapshot_region_ids(
            raw_snapshot,
            _VISUAL_CANDIDATE_LOCK_VISUAL_REGION,
        )
        scene_reset = False

        if (
            self._visual_candidate_lock is not None
            and self._visual_lock_scene_ids is not None
            and not self._same_visual_scene(current_visual_ids, self._visual_lock_scene_ids)
        ):
            self._clear_visual_candidate_state(clear_tracker=True)
            scene_reset = True

        if (
            self._visual_provisional_candidate is not None
            and not self._same_visual_scene(
                current_visual_ids,
                self._visual_provisional_candidate.visual_ids,
            )
        ):
            self._visual_provisional_candidate = None
            scene_reset = True

        return scene_reset

    @staticmethod
    def _candidate_region_count_hint(snapshot: ActivationSnapshot | None) -> int | None:
        if snapshot is None or not snapshot.region_active_counts:
            return None

        candidate_regions = 0
        for region_name, count in snapshot.region_active_counts.items():
            if count <= 0 or region_name in TRACE_FORMATION_EXCLUDED_REGIONS:
                continue
            prefix_exclusion = TRACE_FORMATION_EXCLUDED_REGION_PREFIX_NEURONS.get(
                region_name,
                0,
            )
            if count <= prefix_exclusion:
                continue
            candidate_regions += 1
        return candidate_regions

    def fast_skip_reason(
        self,
        snapshot: ActivationSnapshot | None,
        novelty: float,
        working_memory_count: int,
    ) -> str | None:
        """Return an early no-op reason for the common text hot path.

        This mirrors the cheap prechecks at the top of `step` so callers that do
        not need detailed formation diagnostics can avoid entering the full
        trace-formation path on ticks that are guaranteed to short-circuit.
        """
        candidate_region_hint = self._candidate_region_count_hint(snapshot)
        if candidate_region_hint == 0:
            return "no_candidate_snapshot"

        if (
            candidate_region_hint is not None
            and candidate_region_hint < TRACE_FORMATION_MIN_REGIONS
        ):
            return "formation_min_regions"

        source_primary_modality_family = self._snapshot_primary_modality_family(snapshot)
        if source_primary_modality_family in {"audio", "visual"}:
            return None

        if novelty < _TRACE_FORMATION_NOVELTY_THRESHOLD:
            return "novelty_threshold"

        if working_memory_count >= WORKING_MEMORY_CAPACITY:
            return "working_memory_full"

        return None

    def _advance_visual_candidate_provisional(
        self,
        raw_snapshot: ActivationSnapshot,
    ) -> bool:
        current_visual_ids = self._snapshot_region_ids(
            raw_snapshot,
            _VISUAL_CANDIDATE_LOCK_VISUAL_REGION,
        )
        if self._visual_provisional_candidate is None:
            self._visual_provisional_candidate = _VisualProvisionalCandidate(
                snapshot=raw_snapshot,
                visual_ids=current_visual_ids,
                eligible_ticks=1,
            )
            return False

        self._visual_provisional_candidate = _VisualProvisionalCandidate(
            snapshot=raw_snapshot,
            visual_ids=current_visual_ids,
            eligible_ticks=self._visual_provisional_candidate.eligible_ticks + 1,
        )

        if self._visual_provisional_candidate.eligible_ticks < _VISUAL_CANDIDATE_LOCK_DELAY_TICKS:
            return False

        self._start_visual_candidate_lock(self._visual_provisional_candidate.snapshot)
        self._visual_provisional_candidate = None
        return True

    def prepare_snapshot_for_formation(
        self,
        snapshot: ActivationSnapshot,
        history: ActivationHistory | None = None,
        active_traces: list[tuple[str, float]] | None = None,
        co_trace_ids: list[str] | None = None,
    ) -> ActivationSnapshot:
        """Select input-specific neurons for trace formation.

        The hot tick path uses compact flat activations, so trace formation must
        reconstruct per-region candidates itself. We explicitly exclude the
        globally active attention region and keep only neurons that are either
        novel in the recent window or firing materially above their short-term
        rolling baseline. When current working-memory / active traces already
        span multiple regions, we further ground the selector in neurons that
        those traces actually contain.
        """
        self.last_audio_family_selection_debug = {}
        region_values: dict[str, list[tuple[int, float]]] = defaultdict(list)

        if snapshot.active_neurons:
            for region_name, neurons in snapshot.active_neurons.items():
                region_values[region_name].extend(neurons)
        elif snapshot.active_values:
            region_values = _group_active_values_by_region(snapshot.active_values)
        else:
            # Flat snapshot: active_values truncated at 500, but _flat_ids/_flat_vals
            # always have the full data.  Use top-K by activation for efficiency —
            # the strongest-firing neurons provide the most stable fingerprint.
            flat_ids = getattr(snapshot, "_flat_ids", None)
            flat_vals = getattr(snapshot, "_flat_vals", None)
            if flat_ids and flat_vals and len(flat_ids) == len(flat_vals):
                n = len(flat_ids)
                if n <= 500:
                    pairs = list(zip(flat_ids, flat_vals))
                else:
                    # Select top-500 by activation value (avoids full sort)
                    top_indices = heapq.nlargest(
                        500, range(n), key=lambda i: flat_vals[i]
                    )
                    top_indices.sort(key=lambda i: flat_ids[i])  # sort by neuron ID for region grouper
                    pairs = [(flat_ids[i], flat_vals[i]) for i in top_indices]
                region_values = _group_active_values_by_region(pairs)

        if not region_values:
            return ActivationSnapshot(tick=snapshot.tick)

        baseline_map: dict[int, float] = {}
        baseline_window = 0
        if history is not None:
            baseline_map, baseline_window = history.rolling_activation_baseline(
                TRACE_FORMATION_BASELINE_WINDOW,
            )
        use_baseline_gate = baseline_window >= TRACE_FORMATION_BASELINE_WINDOW

        trace_support_counts = self._trace_support_counts(active_traces, co_trace_ids)
        region_candidate_key = lambda item: (item[0], item[1], item[2], -item[3])
        scored_candidate_key = lambda item: (item[0], item[1], item[2], -item[4])

        scored_candidates: list[tuple[int, float, float, str, int]] = []
        supported_regions: set[str] = set()
        baseline_get = baseline_map.get
        trace_support_get = trace_support_counts.get if trace_support_counts else None
        for region_name, neurons in region_values.items():
            if region_name in TRACE_FORMATION_EXCLUDED_REGIONS:
                continue

            region_candidates: list[tuple[int, float, float, int]] = []
            prefix_exclusion = TRACE_FORMATION_EXCLUDED_REGION_PREFIX_NEURONS.get(
                region_name,
                0,
            )
            prefix_cutoff = None
            if prefix_exclusion > 0:
                prefix_cutoff = REGIONS[region_name][0] + prefix_exclusion
            for neuron_id, activation in neurons:
                if prefix_cutoff is not None and neuron_id < prefix_cutoff:
                    continue
                baseline = baseline_get(neuron_id, 0.0) if use_baseline_gate else 0.0
                specificity = activation - baseline
                if use_baseline_gate:
                    baseline_gate = baseline <= 0.0 or activation >= (
                        baseline * TRACE_FORMATION_MIN_BASELINE_RATIO
                    )
                    if specificity < TRACE_FORMATION_MIN_BASELINE_DELTA or not baseline_gate:
                        continue
                support_count = trace_support_get(neuron_id, 0) if trace_support_get else 0
                if support_count > 0:
                    supported_regions.add(region_name)
                region_candidates.append((support_count, specificity, activation, neuron_id))

            region_limit = TRACE_FORMATION_REGION_MAX_NEURONS.get(
                region_name,
                TRACE_FORMATION_MAX_NEURONS_PER_REGION,
            )
            if region_name == _VISUAL_CANDIDATE_LOCK_VISUAL_REGION:
                region_candidates.sort(key=region_candidate_key, reverse=True)
                selected_candidates = self._select_visual_family_candidates(
                    region_candidates,
                    region_limit,
                )
            elif region_name == _AUDIO_CANDIDATE_AUDIO_REGION:
                region_candidates.sort(key=region_candidate_key, reverse=True)
                selected_candidates = self._select_audio_family_candidates(
                    region_candidates,
                    region_limit,
                )
            elif len(region_candidates) > region_limit:
                selected_candidates = heapq.nlargest(
                    region_limit,
                    region_candidates,
                    key=region_candidate_key,
                )
                selected_candidates.sort(key=region_candidate_key, reverse=True)
            else:
                region_candidates.sort(key=region_candidate_key, reverse=True)
                selected_candidates = region_candidates
            for support_count, specificity, activation, neuron_id in selected_candidates:
                scored_candidates.append(
                    (support_count, specificity, activation, region_name, neuron_id)
                )

        if len(supported_regions) >= TRACE_FORMATION_MIN_REGIONS:
            supported_candidates = [item for item in scored_candidates if item[0] > 0]
            if supported_candidates:
                scored_candidates = supported_candidates

        if len(scored_candidates) > TRACE_FORMATION_MAX_TOTAL_NEURONS:
            selected = heapq.nlargest(
                TRACE_FORMATION_MAX_TOTAL_NEURONS,
                scored_candidates,
                key=scored_candidate_key,
            )
            selected.sort(key=scored_candidate_key, reverse=True)
        else:
            scored_candidates.sort(key=scored_candidate_key, reverse=True)
            selected = scored_candidates

        active_neurons: dict[str, list[tuple[int, float]]] = defaultdict(list)
        active_values: list[tuple[int, float]] = []
        active_ids: list[int] = []
        for _support_count, _specificity, activation, region_name, neuron_id in selected:
            active_neurons[region_name].append((neuron_id, activation))
            active_values.append((neuron_id, activation))
            active_ids.append(neuron_id)

        region_active_counts = {
            region_name: len(neurons)
            for region_name, neurons in active_neurons.items()
        }
        total_active = sum(region_active_counts.values())
        formation_snapshot = ActivationSnapshot(
            tick=snapshot.tick,
            active_neurons=dict(active_neurons),
            active_values=active_values,
            total_active=total_active,
            active_ids=active_ids,
            region_active_counts=region_active_counts,
        )
        return formation_snapshot

    def step(
        self,
        snapshot: ActivationSnapshot | None,
        active_traces: list[tuple[str, float]],
        novelty: float,
        tick: int,
        working_memory_count: int,
        co_trace_ids: list[str] | None = None,
        context_tags: list[str] | None = None,
        history: ActivationHistory | None = None,
        novelty_by_family: dict[str, float] | None = None,
        label: str | None = None,
    ) -> int:
        """Check for and create new traces. Returns number created."""
        self._recently_formed = []
        self.last_tracker_snapshot = None
        debug: dict[str, object] = {
            "snapshot_present": snapshot is not None,
            "working_memory_count": working_memory_count,
            "working_memory_effective_count": working_memory_count,
            "working_memory_full": working_memory_count >= WORKING_MEMORY_CAPACITY,
            "working_memory_reserve_modality_family": None,
            "working_memory_reserve_applied": False,
            "source_primary_modality_family": self._snapshot_primary_modality_family(snapshot),
            "duplicate_suppression_modality_family": None,
            "novelty": float(novelty),
            "effective_novelty": float(novelty),
            "novelty_gate_modality_family": None,
            "novelty_threshold": _TRACE_FORMATION_NOVELTY_THRESHOLD,
            "candidate_snapshot_total_neurons": 0,
            "candidate_snapshot_region_count": 0,
            "candidate_snapshot_region_counts": {},
            "passed_novelty_gate": False,
            "passed_region_gate": False,
            "attempted": False,
            "ready_pattern_count": 0,
            "formed_count": 0,
            "failure_stage": "none",
            "visual_candidate_lock_active": False,
            "visual_candidate_lock_started": False,
            "visual_candidate_lock_regions": [],
            "visual_candidate_provisional_active": False,
            "visual_candidate_provisional_ticks": 0,
            "visual_candidate_scene_reset": False,
            "representation_suppressed": False,
            "existing_trace_match_count": 0,
            "existing_trace_best_id": None,
            "existing_trace_best_score": 0.0,
        }
        source_primary_modality_family = debug["source_primary_modality_family"]

        # Get patterns ready for formation
        if snapshot is not None:
            candidate_region_hint = self._candidate_region_count_hint(snapshot)
            if candidate_region_hint == 0:
                debug["candidate_snapshot_total_neurons"] = int(snapshot.total_active)
                debug["candidate_snapshot_region_count"] = 0
                debug["candidate_snapshot_region_counts"] = dict(snapshot.region_active_counts)
                debug["failure_stage"] = "no_candidate_snapshot"
                self.last_step_debug = debug
                return 0

            if candidate_region_hint is not None and candidate_region_hint < TRACE_FORMATION_MIN_REGIONS:
                debug["candidate_snapshot_total_neurons"] = int(snapshot.total_active)
                debug["candidate_snapshot_region_count"] = int(candidate_region_hint)
                debug["candidate_snapshot_region_counts"] = dict(snapshot.region_active_counts)
                debug["failure_stage"] = "formation_min_regions"
                self.last_step_debug = debug
                return 0

            # For non-audio/non-visual workloads, low-novelty and full working-memory
            # ticks cannot advance trace formation, so skip the expensive snapshot
            # selection path entirely.
            if (
                source_primary_modality_family not in {"audio", "visual"}
                and novelty < _TRACE_FORMATION_NOVELTY_THRESHOLD
            ):
                debug["duplicate_suppression_modality_family"] = source_primary_modality_family
                debug["failure_stage"] = "novelty_threshold"
                self.last_step_debug = debug
                return 0

            # For non-visual/non-audio modalities, allow the tracker to
            # keep accumulating even when working memory is full.  The per-trace
            # cap in the formation loop below limits actual output.

            raw_snapshot = self.prepare_snapshot_for_formation(
                snapshot,
                history,
                active_traces=active_traces,
                co_trace_ids=co_trace_ids,
            )
            formation_snapshot = raw_snapshot
            duplicate_guard_family = self._duplicate_suppression_modality_family(snapshot)
            novelty_gate_modality_family, effective_novelty = self._resolve_novelty_gate(
                novelty,
                novelty_by_family,
                duplicate_guard_family,
            )
            reserve_modality_family, reserve_applied = self._resolve_working_memory_reserve(
                working_memory_count,
                co_trace_ids,
                duplicate_guard_family,
            )
            effective_working_memory_count = max(
                0,
                working_memory_count - (1 if reserve_applied else 0),
            )
            debug["duplicate_suppression_modality_family"] = duplicate_guard_family
            debug["novelty_gate_modality_family"] = novelty_gate_modality_family
            debug["effective_novelty"] = effective_novelty
            debug["working_memory_reserve_modality_family"] = reserve_modality_family
            debug["working_memory_reserve_applied"] = reserve_applied
            debug["working_memory_effective_count"] = effective_working_memory_count
            raw_candidate_region_counts = {
                region_name: len(neurons)
                for region_name, neurons in raw_snapshot.active_neurons.items()
                if neurons
            }
            raw_passed_novelty_gate = effective_novelty >= _TRACE_FORMATION_NOVELTY_THRESHOLD
            raw_passed_region_gate = (
                len(raw_candidate_region_counts) >= TRACE_FORMATION_MIN_REGIONS
            )
            tracker_update_allowed = True

            if self._visual_candidate_lock_enabled:
                debug["visual_candidate_scene_reset"] = self._handle_visual_scene_change(raw_snapshot)

                visual_candidate_eligible = (
                    raw_passed_novelty_gate
                    and raw_passed_region_gate
                    and self._should_start_visual_candidate_lock(raw_snapshot)
                )

                if self._visual_candidate_lock is None:
                    if visual_candidate_eligible:
                        if self._advance_visual_candidate_provisional(raw_snapshot):
                            debug["visual_candidate_lock_started"] = True
                        else:
                            tracker_update_allowed = False
                    else:
                        self._visual_provisional_candidate = None

                if self._visual_candidate_lock is not None:
                    formation_snapshot = self._apply_visual_candidate_lock(raw_snapshot)
                    debug["visual_candidate_lock_active"] = True
                    debug["visual_candidate_lock_regions"] = sorted(
                        self._visual_candidate_lock.keys()
                    )
            else:
                debug["visual_candidate_scene_reset"] = False
                self._visual_provisional_candidate = None
            debug["visual_candidate_provisional_active"] = self._visual_provisional_candidate is not None
            debug["visual_candidate_provisional_ticks"] = (
                self._visual_provisional_candidate.eligible_ticks
                if self._visual_provisional_candidate is not None
                else 0
            )
            candidate_region_counts = {
                region_name: len(neurons)
                for region_name, neurons in formation_snapshot.active_neurons.items()
                if neurons
            }
            debug["candidate_snapshot_total_neurons"] = formation_snapshot.total_active
            debug["candidate_snapshot_region_count"] = len(candidate_region_counts)
            debug["candidate_snapshot_region_counts"] = candidate_region_counts
            if self.last_audio_family_selection_debug:
                debug["audio_family_selection_debug"] = self.last_audio_family_selection_debug
            # For visual/audio modalities, still gate on strict WM capacity.
            # Text modalities are allowed through — the formation loop caps output.
            if (
                source_primary_modality_family in {"audio", "visual"}
                and working_memory_count >= WORKING_MEMORY_CAPACITY
                and not reserve_applied
            ):
                debug["failure_stage"] = "working_memory_full"
                self.last_step_debug = debug
                return 0
            if not formation_snapshot.active_neurons:
                debug["failure_stage"] = "no_candidate_snapshot"
                self.last_step_debug = debug
                return 0
            self.last_tracker_snapshot = formation_snapshot
            debug["passed_novelty_gate"] = (
                effective_novelty >= _TRACE_FORMATION_NOVELTY_THRESHOLD
            )
            debug["passed_region_gate"] = len(candidate_region_counts) >= TRACE_FORMATION_MIN_REGIONS
            debug["attempted"] = bool(
                debug["passed_novelty_gate"]
                and debug["passed_region_gate"]
                and tracker_update_allowed
            )

            if debug["attempted"]:
                existing_matches: list[tuple[str, float]] = []
                if duplicate_guard_family in {"audio", "visual"}:
                    existing_matches = self._matching_existing_traces(
                        formation_snapshot,
                        duplicate_guard_family=duplicate_guard_family,
                    )
                debug["existing_trace_match_count"] = len(existing_matches)
                if existing_matches:
                    best_trace_id, best_score = existing_matches[0]
                    debug["representation_suppressed"] = True
                    debug["existing_trace_best_id"] = best_trace_id
                    debug["existing_trace_best_score"] = float(best_score)
                    debug["failure_stage"] = "already_represented"
                    self._clear_visual_candidate_state(clear_tracker=True)
                    self.last_step_debug = debug
                    return 0

            if debug["attempted"]:
                ready_patterns = self.tracker.update_from_snapshot(
                    formation_snapshot,
                    effective_novelty,
                )
            else:
                ready_patterns = []
        else:
            debug["passed_novelty_gate"] = novelty >= _TRACE_FORMATION_NOVELTY_THRESHOLD
            debug["attempted"] = bool(debug["passed_novelty_gate"])
            ready_patterns = self.tracker.update_from_brain(novelty)

        debug["ready_pattern_count"] = len(ready_patterns)
        if debug["failure_stage"] == "none":
            if debug["visual_candidate_provisional_active"] and not debug["visual_candidate_lock_active"]:
                debug["failure_stage"] = "visual_lock_delay"
            elif not debug["passed_novelty_gate"]:
                debug["failure_stage"] = "novelty_threshold"
            elif snapshot is not None and not debug["passed_region_gate"]:
                debug["failure_stage"] = "formation_min_regions"
            elif ready_patterns:
                debug["failure_stage"] = "ready_pattern"
            elif debug["attempted"]:
                debug["failure_stage"] = "tracker_pending"
            else:
                debug["failure_stage"] = "not_attempted"

        formed = 0
        effective_working_memory_count = int(debug.get("working_memory_effective_count", working_memory_count))
        # Allow up to 3 formation slots beyond working-memory capacity so text
        # workloads (where seed traces fill WM quickly) can still learn.
        formation_cap = WORKING_MEMORY_CAPACITY + 3
        for neurons_by_region in ready_patterns:
            if effective_working_memory_count + formed >= formation_cap:
                break

            trace_id = f"trace_{uuid.uuid4().hex[:8]}"
            # Assign label-deterministic speech neurons so all traces with the
            # same label share a consistent speech representation.  Any speech
            # neurons inherited from the activation snapshot (boosted by prior
            # traces in working memory) are replaced.
            speech_start, speech_end = REGIONS["speech"]
            speech_count = speech_end - speech_start + 1
            inhib_pct = REGION_CONFIG["speech"]["inhibitory_pct"]
            exc_end = speech_start + int(speech_count * (1.0 - inhib_pct)) - 1
            n_speech = NEURONS_PER_TRACE.get("speech", 2)
            if n_speech > 0 and label:
                # Deterministic assignment: hash the label to pick neurons
                import hashlib
                h = int(hashlib.sha256(label.encode()).hexdigest(), 16)
                exc_range = exc_end - speech_start + 1
                assigned = []
                for j in range(n_speech):
                    offset = (h + j * 2654435761) % exc_range
                    assigned.append(speech_start + offset)
                neurons_by_region["speech"] = assigned
            elif n_speech > 0 and "language" in neurons_by_region and "speech" not in neurons_by_region:
                neurons_by_region["speech"] = random.sample(
                    range(speech_start, exc_end + 1), n_speech
                )
            trace = Trace(
                id=trace_id,
                neurons=neurons_by_region,
                strength=0.1,
                novelty=1.0,
                decay=1.0,
                co_traces=list(co_trace_ids or []),
                context_tags=list(context_tags or []),
                label=label,
                last_fired=tick,
                formation_tick=tick,
            )
            self.trace_store.add(trace)
            self._recently_formed.append(trace_id)
            # Directly boost the new trace's speech neurons so the decode
            # pipeline sees an immediate signal for the current label.
            if "speech" in neurons_by_region and neurons_by_region["speech"]:
                brain_core.boost_speech(neurons_by_region["speech"], 0.5)
            formed += 1

        debug["formed_count"] = formed
        if formed > 0:
            self.tracker.clear()
        if ready_patterns or formed > 0:
            self._clear_visual_candidate_state()
        self.last_step_debug = debug
        return formed

    def merge_overlapping(self, min_co_fires: int = 10) -> int:
        """Merge traces with > 80% neuron overlap that frequently co-activate.

        Returns number of merges performed.
        """
        self.trace_store.sync_runtime_state()
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
