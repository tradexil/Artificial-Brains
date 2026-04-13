"""Python-side tick loop orchestrator.

Wraps the Rust tick (Phase 1: propagate + Phase 2: update) with
Python-driven Phase 2 (evaluate) + Phase 3 (learn) + Phase 4 (maintain).

Tick phases:
  Phase 1 (Rust): propagate signals, deliver delayed, update neurons,
                   update attention gains (three-drive), compute prediction errors
  Phase 2 (Python): evaluate — trace matching, prediction, attention drives,
                     binding evaluation, integration boost
  Phase 3 (Python): learn — Hebbian, anti-Hebbian, trace formation, binding formation
  Phase 4 (Python): maintain — apply weight updates, rebuild CSR, prune,
                     consolidation
"""

from __future__ import annotations

import brain_core
import time

from brain.learning.hebbian import compute_effective_learning_rate
from brain.learning.anti_hebbian import compute_anti_hebbian_rate
from brain.learning.prediction import PredictionEngine
from brain.learning.pruning import get_phase, pruning_pass_sampled
from brain.learning.consolidation import ConsolidationEngine
from brain.learning.trace_formation import TraceFormationEngine
from brain.learning.binding_formation import BindingFormationEngine
from brain.learning.homeostasis import HomeostasisManager, SleepSessionStats
from brain.structures.brain_state import (
    ActivationHistory,
    ActivationSnapshot,
    NeuromodulatorState,
)
from brain.structures.trace_store import TraceStore
from brain.utils.config import (
    BINDING_MAINTENANCE_INTERVAL,
    BINDING_RECALL_BOOST_SCALE,
    BINDING_RECALL_MIN_RELATIVE_WEIGHT,
    BINDING_RECALL_PATTERN_COMPLETION_BOOST,
    BINDING_RECALL_PATTERN_COMPLETION_THRESHOLD,
    BINDING_RECALL_TRACE_MATCH_THRESHOLD,
    COACTIVE_TRACK_INTERVAL,
    HEBBIAN_WINDOW,
    PRUNE_INTERVAL,
    REBUILD_INTERVAL,
    SYNAPSE_UPDATE_MAX_BATCH_SYNAPSE_MULTIPLIER,
    SYNAPSE_UPDATE_RELEASE_INTERVAL,
    SYNAPSE_UPDATE_TARGET_DEFERRED_SYNAPSE_MULTIPLIER,
    TRACE_ACTIVE_NEURON_BUDGET,
    TRACE_AGE_DECAY_WINDOW,
    TRACE_AGE_FLOOR_CEILING,
    TRACE_ACTIVATION_THRESHOLD,
    TRACE_FRESHNESS_FLOOR,
    TRACE_FRESHNESS_MIN_SCORE,
    TRACE_FRESHNESS_RETENTION,
    TRACE_REFRESH_MAX_BOOST,
    WORKING_MEMORY_CAPACITY,
    WORKING_MEMORY_DECAY_RATE,
)
from brain.structures.neuron_map import all_region_names
from brain.structures.neuron_map import all_region_names


def _merge_trace_scores(*trace_groups: list[tuple[str, float]]) -> list[tuple[str, float]]:
    merged_scores: dict[str, float] = {}
    for trace_group in trace_groups:
        for trace_id, score in trace_group:
            merged_scores[trace_id] = max(merged_scores.get(trace_id, 0.0), float(score))
    return sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)


class WorkingMemory:
    """Capacity-limited buffer of active trace IDs."""

    def __init__(self, capacity: int = WORKING_MEMORY_CAPACITY):
        self.capacity = capacity
        # (trace_id, activation_strength) — most recent is strongest
        self.slots: list[tuple[str, float]] = []

    def update(self, active_traces: list[tuple[str, float]]) -> None:
        """Update working memory with currently active traces.

        Active traces refresh/enter slots. All slots decay slightly.
        If over capacity, weakest is evicted.
        """
        # Decay existing slots
        self.slots = [
            (tid, strength * (1.0 - WORKING_MEMORY_DECAY_RATE))
            for tid, strength in self.slots
        ]

        # Refresh or add active traces
        active_map = dict(active_traces)
        existing_ids = {tid for tid, _ in self.slots}

        for tid, score in active_traces:
            if tid in existing_ids:
                # Refresh — update strength
                self.slots = [
                    (t, max(s, score)) if t == tid else (t, s)
                    for t, s in self.slots
                ]
            else:
                self.slots.append((tid, score))

        # Sort by strength descending
        self.slots.sort(key=lambda x: x[1], reverse=True)

        # Enforce capacity
        if len(self.slots) > self.capacity:
            self.slots = self.slots[: self.capacity]

        # Remove dead entries
        self.slots = [(t, s) for t, s in self.slots if s > 0.01]

    def replace(self, slots: list[tuple[str, float]]) -> None:
        """Replace working memory slots from the Rust-side tracker."""
        self.slots = list(slots[: self.capacity])

    @property
    def trace_ids(self) -> list[str]:
        return [tid for tid, _ in self.slots]

    def __len__(self) -> int:
        return len(self.slots)


class TickLoop:
    """Orchestrates the full tick cycle: Rust tick + Python learning.

    Usage:
        loop = TickLoop(trace_store)
        for i in range(N):
            brain_core.inject_activations(signals)
            result = loop.step()
    """

    def __init__(
        self,
        trace_store: TraceStore,
        *,
        rust_tick_batch_size: int = 1,
        collect_full_metrics: bool = True,
        synapse_update_release_interval: int = SYNAPSE_UPDATE_RELEASE_INTERVAL,
        prune_interval: int = PRUNE_INTERVAL,
        prune_batch_size: int = 10_000,
        binding_maintenance_interval: int = BINDING_MAINTENANCE_INTERVAL,
        rebuild_interval: int = REBUILD_INTERVAL,
    ):
        self.trace_store = trace_store
        self.history = ActivationHistory(window=HEBBIAN_WINDOW)
        self.neuromod = NeuromodulatorState()
        self.prediction = PredictionEngine(trace_store)
        self.working_memory = WorkingMemory()

        # Phase 5 engines
        self.consolidation = ConsolidationEngine()
        self.trace_formation = TraceFormationEngine(trace_store)
        self.binding_formation = BindingFormationEngine(trace_store)

        # Phase 9: homeostasis manager
        self.homeostasis = HomeostasisManager(trace_store)

        # Track all traces that fired during this awake period (for consolidation)
        self._awake_trace_ids: list[str] = []

        # Stats per step
        self.last_tick_number: int = 0
        self.last_total_active: int = 0
        self.last_novelty: float = 0.0
        self.last_hebbian_updates: int = 0
        self.last_anti_hebbian_updates: int = 0
        self.last_active_traces: list[tuple[str, float]] = []
        self.last_binding_recall_candidates: list[tuple[int, float, float]] = []
        self.last_synapse_update_profile: dict[str, float] = {}
        self.working_memory_overlay_cap: int = 0
        self.rust_tick_batch_size: int = max(1, int(rust_tick_batch_size))
        self.collect_full_metrics: bool = bool(collect_full_metrics)

        # Synapse fire tracking for pruning (sparse — only updated during learning)
        self._synapse_last_fired: dict[tuple[int, int], int] = {}

        # Maintenance schedule
        self._synapse_update_release_interval: int = max(
            1,
            int(synapse_update_release_interval),
        )
        self._prune_interval: int = max(1, int(prune_interval))
        self._binding_maintenance_interval: int = max(
            1,
            int(binding_maintenance_interval),
        )
        self._rebuild_interval: int = max(1, int(rebuild_interval))

        # Pruning rotation counter
        self._prune_offset: int = 0
        self._prune_batch_size: int = max(1, int(prune_batch_size))

    def _effective_rust_tick_batch_size(self, max_rust_ticks: int | None = None) -> int:
        if max_rust_ticks is None:
            return self.rust_tick_batch_size
        return max(1, min(self.rust_tick_batch_size, int(max_rust_ticks)))

    def iter_steps(
        self,
        total_ticks: int,
        *,
        learn: bool = True,
        allow_trace_formation: bool = True,
        allow_binding_formation: bool = True,
        preserve_first_tick: bool = False,
    ):
        remaining_ticks = max(0, int(total_ticks))
        preserve_leading_tick = bool(preserve_first_tick and self.rust_tick_batch_size > 1)
        while remaining_ticks > 0:
            result = self.step(
                learn=learn,
                allow_trace_formation=allow_trace_formation,
                allow_binding_formation=allow_binding_formation,
                max_rust_ticks=1 if preserve_leading_tick else remaining_ticks,
            )
            executed_ticks = max(1, int(result.get("executed_ticks", 1) or 1))
            remaining_ticks = max(0, remaining_ticks - executed_ticks)
            preserve_leading_tick = False
            yield result

    def step(
        self,
        learn: bool = True,
        allow_trace_formation: bool = True,
        allow_binding_formation: bool = True,
        max_rust_ticks: int | None = None,
    ) -> dict:
        """Execute one full tick cycle.

        Returns dict with tick stats.
        """
        step_started = time.perf_counter()
        previous_active_traces = list(self.last_active_traces)

        # === Pre-tick: Predict what should happen ===
        self.prediction.predict(
            self.last_active_traces,
            self.working_memory.trace_ids,
        )

        # Sync Python neuromodulator state into Rust before the tick so energy,
        # focus, and prior-step arousal affect the current tick dynamics.
        brain_core.set_neuromodulator(
            self.neuromod.arousal,
            self.neuromod.valence,
            self.neuromod.focus,
            self.neuromod.energy,
        )

        # === Phase 1: Rust tick ===
        # (propagate + update + attention gain update + prediction error)
        executed_ticks = self._effective_rust_tick_batch_size(max_rust_ticks)
        binding_recall_bindings = 0.0
        binding_recall_neurons = 0.0
        binding_recall_max_relative_weight = 0.0
        binding_recall_max_boost = 0.0
        working_memory_boost_neurons = 0.0
        pattern_completion_neurons = 0.0
        speech_boost_neurons = 0.0

        if self.collect_full_metrics:
            rust_tick_started = time.perf_counter()
            (
                tick_num,
                _active_counts,
                total_active,
                tick_profile,
                executed_ticks,
            ) = brain_core.tick_batch(executed_ticks)
            rust_tick_ms = (time.perf_counter() - rust_tick_started) * 1000
            tick_prepare_ms = tick_profile.get("prepare_ms", 0.0)
            tick_delayed_delivery_ms = tick_profile.get("delayed_delivery_ms", 0.0)
            tick_propagate_ms = tick_profile.get("propagate_ms", 0.0)
            tick_update_ms = tick_profile.get("update_ms", 0.0)
            incoming_signal_count = tick_profile.get("incoming_signal_count", 0.0)
            incoming_signal_abs_sum = tick_profile.get("incoming_signal_abs_sum", 0.0)
            immediate_signal_count = tick_profile.get("immediate_signal_count", 0.0)
            immediate_signal_abs_sum = tick_profile.get("immediate_signal_abs_sum", 0.0)
            delayed_delivery_signal_count = tick_profile.get(
                "delayed_delivery_signal_count", 0.0
            )
            delayed_delivery_signal_abs_sum = tick_profile.get(
                "delayed_delivery_signal_abs_sum", 0.0
            )
            scheduled_delayed_signal_count = tick_profile.get(
                "scheduled_delayed_signal_count", 0.0
            )
            scheduled_delayed_signal_abs_sum = tick_profile.get(
                "scheduled_delayed_signal_abs_sum", 0.0
            )
            total_fired = tick_profile.get("total_fired", 0.0)
            refractory_ignored_abs_sum = tick_profile.get("refractory_ignored_abs_sum", 0.0)
            fire_interval_sum = tick_profile.get("fire_interval_sum", 0.0)
            fire_interval_count = tick_profile.get("fire_interval_count", 0.0)
            self.last_tick_number = tick_num
            self.last_total_active = total_active

            evaluation_started = time.perf_counter()
            (
                snapshot_values,
                active_traces,
                compact_state,
                snapshot_counts,
                snapshot_total_active,
                working_memory_slots,
                binding_recall_candidates,
                evaluation_profile,
            ) = brain_core.evaluate_tick_compact(
                self.trace_store.store_id,
                0.01,
                TRACE_ACTIVATION_THRESHOLD,
                tick_num,
                WORKING_MEMORY_DECAY_RATE,
                self.working_memory.capacity,
                self.working_memory_overlay_cap,
                TRACE_ACTIVE_NEURON_BUDGET,
                TRACE_FRESHNESS_RETENTION,
                TRACE_FRESHNESS_FLOOR,
                TRACE_FRESHNESS_MIN_SCORE,
                TRACE_REFRESH_MAX_BOOST,
                TRACE_AGE_DECAY_WINDOW,
                TRACE_AGE_FLOOR_CEILING,
                BINDING_RECALL_MIN_RELATIVE_WEIGHT,
                BINDING_RECALL_BOOST_SCALE,
            )
            snapshot = self.history.push_compact_snapshot(
                tick_num,
                snapshot_values,
                total_active=snapshot_total_active,
                region_active_counts=snapshot_counts,
            )
            # Also push into Rust-side cache for zero-FFI learning
            if snapshot_values:
                sv_ids = [nid for nid, _ in snapshot_values]
                sv_vals = [act for _, act in snapshot_values]
                brain_core.push_activation_snapshot(sv_ids, sv_vals)
            evaluation_ms = (time.perf_counter() - evaluation_started) * 1000
            snapshot_ms = evaluation_profile.get("snapshot_ms", 0.0)
            batch_state_ms = evaluation_profile.get("batch_state_ms", 0.0)
            trace_match_ms = evaluation_profile.get("trace_match_ms", 0.0)
            trace_candidates = evaluation_profile.get(
                "trace_candidates", float(len(active_traces))
            )
            trace_side_effects_ms = evaluation_profile.get("trace_side_effects_ms", 0.0)
            binding_recall_ms = evaluation_profile.get("binding_recall_ms", 0.0)
            binding_recall_bindings = evaluation_profile.get("binding_recall_bindings", 0.0)
            binding_recall_neurons = evaluation_profile.get("binding_recall_neurons", 0.0)
            binding_recall_max_relative_weight = evaluation_profile.get(
                "binding_recall_max_relative_weight", 0.0
            )
            binding_recall_max_boost = evaluation_profile.get(
                "binding_recall_max_boost", 0.0
            )
            boost_region_metrics: dict[str, float] = {}
            for prefix in (
                "working_memory_boost",
                "pattern_completion",
                "speech_boost",
                "binding_recall",
            ):
                for region_name in all_region_names():
                    key = f"{prefix}_region_{region_name}_neurons"
                    boost_region_metrics[key] = evaluation_profile.get(key, 0.0)
            region_active_metrics = {
                f"active_region_{region_name}_neurons": snapshot_counts.get(region_name, 0)
                for region_name in all_region_names()
            }
            propagation_region_metrics = {
                f"incoming_region_{region_name}_signals": tick_profile.get(
                    f"incoming_region_{region_name}_signals", 0.0
                )
                for region_name in all_region_names()
            }
            propagation_region_abs_metrics = {
                f"incoming_region_{region_name}_abs_sum": tick_profile.get(
                    f"incoming_region_{region_name}_abs_sum", 0.0
                )
                for region_name in all_region_names()
            }
            fired_region_metrics = {
                f"fired_region_{region_name}_neurons": tick_profile.get(
                    f"fired_region_{region_name}_neurons", 0.0
                )
                for region_name in all_region_names()
            }
            refractory_region_metrics = {
                f"refractory_ignored_region_{region_name}_abs_sum": tick_profile.get(
                    f"refractory_ignored_region_{region_name}_abs_sum", 0.0
                )
                for region_name in all_region_names()
            }
            refractory_region_metrics.update(
                {
                    f"refractory_ignored_region_{region_name}_immediate_same_abs_sum": tick_profile.get(
                        f"refractory_ignored_region_{region_name}_immediate_same_abs_sum", 0.0
                    )
                    for region_name in all_region_names()
                }
            )
            refractory_region_metrics.update(
                {
                    f"refractory_ignored_region_{region_name}_immediate_cross_abs_sum": tick_profile.get(
                        f"refractory_ignored_region_{region_name}_immediate_cross_abs_sum", 0.0
                    )
                    for region_name in all_region_names()
                }
            )
            refractory_region_metrics.update(
                {
                    f"refractory_ignored_region_{region_name}_delayed_same_abs_sum": tick_profile.get(
                        f"refractory_ignored_region_{region_name}_delayed_same_abs_sum", 0.0
                    )
                    for region_name in all_region_names()
                }
            )
            refractory_region_metrics.update(
                {
                    f"refractory_ignored_region_{region_name}_delayed_cross_abs_sum": tick_profile.get(
                        f"refractory_ignored_region_{region_name}_delayed_cross_abs_sum", 0.0
                    )
                    for region_name in all_region_names()
                }
            )
            fire_interval_region_metrics = {
                f"fire_interval_region_{region_name}_sum": tick_profile.get(
                    f"fire_interval_region_{region_name}_sum", 0.0
                )
                for region_name in all_region_names()
            }
            fire_interval_region_metrics.update(
                {
                    f"fire_interval_region_{region_name}_count": tick_profile.get(
                        f"fire_interval_region_{region_name}_count", 0.0
                    )
                    for region_name in all_region_names()
                }
            )
            potential_region_metrics = {
                f"potential_region_{region_name}_pre_leak_sum": tick_profile.get(
                    f"potential_region_{region_name}_pre_leak_sum", 0.0
                )
                for region_name in all_region_names()
            }
            potential_region_metrics.update(
                {
                    f"potential_region_{region_name}_leak_loss_sum": tick_profile.get(
                        f"potential_region_{region_name}_leak_loss_sum", 0.0
                    )
                    for region_name in all_region_names()
                }
            )
            potential_region_metrics.update(
                {
                    f"potential_region_{region_name}_reset_sum": tick_profile.get(
                        f"potential_region_{region_name}_reset_sum", 0.0
                    )
                    for region_name in all_region_names()
                }
            )
            potential_region_metrics.update(
                {
                    f"potential_region_{region_name}_carried_sum": tick_profile.get(
                        f"potential_region_{region_name}_carried_sum", 0.0
                    )
                    for region_name in all_region_names()
                }
            )
            delayed_flow_metrics = {
                f"delayed_flow_{source_region}_to_{target_region}_signals": tick_profile.get(
                    f"delayed_flow_{source_region}_to_{target_region}_signals", 0.0
                )
                for source_region in all_region_names()
                for target_region in all_region_names()
            }
            delayed_flow_metrics.update(
                {
                    f"delayed_flow_{source_region}_to_{target_region}_abs_sum": tick_profile.get(
                        f"delayed_flow_{source_region}_to_{target_region}_abs_sum", 0.0
                    )
                    for source_region in all_region_names()
                    for target_region in all_region_names()
                }
            )
            working_memory_boost_neurons = evaluation_profile.get(
                "working_memory_boost_neurons", 0.0
            )
            pattern_completion_neurons = evaluation_profile.get(
                "pattern_completion_neurons", 0.0
            )
            speech_boost_neurons = evaluation_profile.get("speech_boost_neurons", 0.0)
            evaluation_rust_ms = evaluation_profile.get(
                "evaluation_rust_ms",
                snapshot_ms
                + batch_state_ms
                + trace_match_ms
                + trace_side_effects_ms
                + binding_recall_ms,
            )
        else:
            tick_started = time.perf_counter()
            (
                tick_num,
                snapshot_counts,
                total_active,
                tick_profile,
                executed_ticks,
            ) = brain_core.tick_batch_compact(
                executed_ticks,
            )
            tick_ms = (time.perf_counter() - tick_started) * 1000
            tick_prepare_ms = tick_profile.get("prepare_ms", 0.0)
            tick_delayed_delivery_ms = tick_profile.get("delayed_delivery_ms", 0.0)
            tick_propagate_ms = tick_profile.get("propagate_ms", 0.0)
            tick_update_ms = tick_profile.get("update_ms", 0.0)
            rust_tick_ms = tick_ms
            incoming_signal_count = 0.0
            incoming_signal_abs_sum = 0.0
            immediate_signal_count = 0.0
            immediate_signal_abs_sum = 0.0
            delayed_delivery_signal_count = 0.0
            delayed_delivery_signal_abs_sum = 0.0
            scheduled_delayed_signal_count = 0.0
            scheduled_delayed_signal_abs_sum = 0.0
            total_fired = 0.0
            refractory_ignored_abs_sum = 0.0
            fire_interval_sum = 0.0
            fire_interval_count = 0.0
            self.last_tick_number = tick_num
            self.last_total_active = total_active

            evaluation_started = time.perf_counter()
            (
                snapshot_ids,
                snapshot_vals,
                active_traces,
                compact_state,
                snapshot_counts,
                snapshot_total_active,
                working_memory_slots,
                evaluation_profile,
            ) = brain_core.evaluate_tick_compact_minimal(
                self.trace_store.store_id,
                0.01,
                TRACE_ACTIVATION_THRESHOLD,
                tick_num,
                WORKING_MEMORY_DECAY_RATE,
                self.working_memory.capacity,
                self.working_memory_overlay_cap,
                TRACE_ACTIVE_NEURON_BUDGET,
                TRACE_FRESHNESS_RETENTION,
                TRACE_FRESHNESS_FLOOR,
                TRACE_FRESHNESS_MIN_SCORE,
                TRACE_REFRESH_MAX_BOOST,
                TRACE_AGE_DECAY_WINDOW,
                TRACE_AGE_FLOOR_CEILING,
                BINDING_RECALL_MIN_RELATIVE_WEIGHT,
                BINDING_RECALL_BOOST_SCALE,
            )
            evaluation_ms = (time.perf_counter() - evaluation_started) * 1000
            binding_recall_candidates = []

            snapshot_started = time.perf_counter()
            snapshot = self.history.push_flat_snapshot(
                tick_num,
                snapshot_ids,
                snapshot_vals,
                total_active=snapshot_total_active,
                region_active_counts=snapshot_counts,
            )
            # Rust-side snapshot cache is pushed inside evaluate_tick_compact_minimal
            snapshot_push_ms = (time.perf_counter() - snapshot_started) * 1000

            snapshot_ms = evaluation_profile.get("snapshot_ms", 0.0)
            batch_state_ms = evaluation_profile.get("batch_state_ms", 0.0)
            trace_match_ms = evaluation_profile.get("trace_match_ms", 0.0)
            trace_candidates = evaluation_profile.get("trace_candidates", 0.0)
            trace_side_effects_ms = evaluation_profile.get("trace_side_effects_ms", 0.0)
            binding_recall_ms = evaluation_profile.get("binding_recall_ms", 0.0)
            evaluation_rust_ms = evaluation_profile.get(
                "evaluation_rust_ms",
                snapshot_ms
                + batch_state_ms
                + trace_match_ms
                + trace_side_effects_ms
                + binding_recall_ms,
            )
            evaluation_ms += snapshot_push_ms
            boost_region_metrics = {}
            region_active_metrics = {}
            propagation_region_metrics = {}
            propagation_region_abs_metrics = {}
            fired_region_metrics = {}
            refractory_region_metrics = {}
            fire_interval_region_metrics = {}
            potential_region_metrics = {}
            delayed_flow_metrics = {}
        binding_recall_python_started = time.perf_counter()
        binding_recall_python_replace_working_memory_started = time.perf_counter()
        self.working_memory.replace(working_memory_slots)
        binding_recall_python_replace_working_memory_ms = (
            time.perf_counter() - binding_recall_python_replace_working_memory_started
        ) * 1000

        binding_recall_python_build_weights_ms = 0.0
        binding_recall_python_merge_active_bindings_ms = 0.0
        binding_recall_python_detail_check_ms = 0.0
        binding_recall_python_augment_ms = 0.0
        binding_recall_python_working_memory_update_ms = 0.0
        binding_recall_python_pattern_complete_ms = 0.0
        augment_binding_recall_profile: dict[str, float] = {}
        pattern_complete_binding_recall_profile: dict[str, float] = {}
        binding_recall_completion_rust_ms = 0.0
        binding_recall_completion_profile: dict[str, float] = {}
        recall_reactivated_traces: list[tuple[str, float]] = []
        binding_pattern_completion_traces = 0
        binding_pattern_completion_neurons = 0
        if snapshot is not None:
            binding_recall_completion_rust_started = time.perf_counter()
            (
                active_traces,
                recall_reactivated_traces,
                binding_pattern_completion_traces,
                binding_pattern_completion_neurons,
                binding_recall_completion_profile,
            ) = brain_core.complete_binding_recall(
                self.trace_store.store_id,
                list(snapshot.active_ids),
                active_traces,
                BINDING_RECALL_MIN_RELATIVE_WEIGHT,
                BINDING_RECALL_TRACE_MATCH_THRESHOLD,
                TRACE_ACTIVATION_THRESHOLD,
                BINDING_RECALL_PATTERN_COMPLETION_THRESHOLD,
                BINDING_RECALL_PATTERN_COMPLETION_BOOST,
            )
            binding_recall_completion_rust_ms = (
                time.perf_counter() - binding_recall_completion_rust_started
            ) * 1000
            if recall_reactivated_traces:
                binding_recall_python_working_memory_update_started = time.perf_counter()
                self.working_memory.update(recall_reactivated_traces)
                binding_recall_python_working_memory_update_ms = (
                    time.perf_counter()
                    - binding_recall_python_working_memory_update_started
                ) * 1000

        binding_recall_python_total_ms = (
            time.perf_counter() - binding_recall_python_started
        ) * 1000
        binding_recall_python_ms = max(
            0.0,
            binding_recall_python_total_ms - binding_recall_completion_rust_ms,
        )
        interval_active_traces = active_traces
        if executed_ticks > 1 and previous_active_traces:
            interval_active_traces = _merge_trace_scores(
                previous_active_traces,
                active_traces,
            )
        self.last_active_traces = interval_active_traces
        self.last_binding_recall_candidates = binding_recall_candidates

        # Prediction error (trace-based)
        errors = self.prediction.compute_errors(snapshot.region_active_counts)
        novelty = self.prediction.global_error(errors)
        novelty_by_family = self.prediction.modality_family_errors(errors)
        self.last_novelty = novelty
        self.prediction.apply_effects(novelty, self.neuromod)

        # Feed novelty drives to Rust attention system (batch — single FFI call)
        drives = {}
        for region in all_region_names():
            error = errors.get(region, 0.0)
            drives[region] = (error, 0.0, 0.0)
        brain_core.batch_set_attention_drives(drives)

        # === Phase 3: Learn (Hebbian + anti-Hebbian + coactive tracking — zero-copy Rust) ===
        current = self.history.current
        learn_step_ms = 0.0
        if learn and current is not None:
            # Compute effective learning rates
            hebbian_lr = compute_effective_learning_rate(
                tick_num, self.neuromod, novelty
            )
            hebbian_lr *= self.prediction.learning_rate_multiplier
            anti_hebbian_rate = compute_anti_hebbian_rate(tick_num)

            has_active = bool(getattr(current, '_flat_ids', None)) or bool(current.active_values)

            if has_active:
                track_coactive = (tick_num % COACTIVE_TRACK_INTERVAL) == 0

                # Learn from Rust-side snapshot cache — no activation data over FFI
                learn_step_started = time.perf_counter()
                hebb_count, anti_count, coactive_pairs = brain_core.learn_from_snapshot_cache(
                    hebbian_lr,
                    anti_hebbian_rate,
                    track_coactive,
                )
                learn_step_ms = (time.perf_counter() - learn_step_started) * 1000

                # Update synapse fire tracking for pruning
                if track_coactive:
                    for src_id, tgt_id in coactive_pairs:
                        self._synapse_last_fired[(src_id, tgt_id)] = tick_num
            else:
                hebb_count = 0
                anti_count = 0
        else:
            hebb_count = 0
            anti_count = 0

        self.last_hebbian_updates = hebb_count
        self.last_anti_hebbian_updates = anti_count

        # Read back Rust-computed neuromodulator state from this tick before any
        # Python-side post-processing that depends on it.
        python_arousal = self.neuromod.arousal
        rust_arousal, rust_valence, rust_focus, rust_energy = (
            brain_core.get_neuromodulator()
        )

        # === Phase 5: Integration & Memory ===

        if self.collect_full_metrics:
            n_input_regions = int(compact_state.get("integration_input_count", 0.0))
            emotion_polarity = compact_state.get("emotion_polarity", 0.0)
            emotion_arousal = max(compact_state.get("emotion_arousal", 0.0), rust_arousal)
            exec_engagement = compact_state.get("executive_engagement", 0.0)
            motor_conflict = compact_state.get("motor_conflict", 0.0)
            planning = compact_state.get("planning_signal", 0.0)
            language_activation = compact_state.get("language_activation", 0.0)
            inner_monologue = compact_state.get("inner_monologue", 0.0)
            speech_activity = compact_state.get("speech_activity", 0.0)
            sensory_activation = compact_state.get("sensory_activation", 0.0)
            visual_activation = compact_state.get("visual_activation", 0.0)
            audio_activation = compact_state.get("audio_activation", 0.0)
            motor_activation = compact_state.get("motor_activation", 0.0)
            motor_approach = compact_state.get("motor_approach", 0.0)
            motor_withdraw = compact_state.get("motor_withdraw", 0.0)
            pain_level = compact_state.get("pain_level", 0.0)
        else:
            # compact_state is a flat Vec<f64> from read_state_compact_from_brain
            # Order: sensory(0), visual(1), audio(2), motor(3), language(4),
            #   speech(5), emotion_polarity(6), emotion_arousal(7),
            #   executive(8), motor_conflict(9), planning(10),
            #   motor_approach(11), motor_withdraw(12), inner_monologue(13),
            #   pain(14), integration_input_count(15)
            sensory_activation = compact_state[0]
            visual_activation = compact_state[1]
            audio_activation = compact_state[2]
            motor_activation = compact_state[3]
            language_activation = compact_state[4]
            speech_activity = compact_state[5]
            emotion_polarity = compact_state[6]
            emotion_arousal = max(compact_state[7], rust_arousal)
            exec_engagement = compact_state[8]
            motor_conflict = compact_state[9]
            planning = compact_state[10]
            motor_approach = compact_state[11]
            motor_withdraw = compact_state[12]
            inner_monologue = compact_state[13]
            pain_level = compact_state[14]
            n_input_regions = int(compact_state[15])

        # Integration boost based on multi-modal convergence (from batch_read_state)
        if n_input_regions >= 2:
            from brain.utils.config import REGIONS
            integration_size = REGIONS["integration"][1] - REGIONS["integration"][0] + 1
            strength = min(1.0, n_input_regions / 6.0)
            brain_core.boost_integration(strength, min(100, integration_size))

        # Track awake trace IDs for consolidation
        for tid, _ in interval_active_traces:
            if tid not in self._awake_trace_ids:
                self._awake_trace_ids.append(tid)

        # Trace formation: detect persistent novel patterns
        formation_started = time.perf_counter()
        if learn and allow_trace_formation:
            if (
                not self.collect_full_metrics
                and self.trace_formation.fast_skip_reason(
                    snapshot,
                    novelty,
                    len(self.working_memory),
                )
                is not None
            ):
                traces_formed = 0
            else:
                traces_formed = self.trace_formation.step(
                    snapshot,
                    interval_active_traces,
                    novelty,
                    tick_num,
                    len(self.working_memory),
                    co_trace_ids=self.working_memory.trace_ids,
                    history=self.history,
                    novelty_by_family=novelty_by_family,
                )

            if allow_binding_formation:
                # Binding formation: detect co-active cross-region patterns
                binding_stats = self.binding_formation.step(
                    interval_active_traces,
                    tick_num,
                    self.history,
                    tick_span=executed_ticks,
                )
            else:
                binding_stats = {
                    "candidates": 0,
                    "formed": 0,
                    "total_bindings": brain_core.get_binding_count(),
                }
        elif learn and allow_binding_formation:
            traces_formed = 0
            binding_stats = self.binding_formation.step(
                interval_active_traces,
                tick_num,
                self.history,
                tick_span=executed_ticks,
            )
        else:
            traces_formed = 0
            binding_stats = {
                "candidates": 0,
                "formed": 0,
                "total_bindings": brain_core.get_binding_count(),
            }
        formation_ms = (time.perf_counter() - formation_started) * 1000

        # === Phase 6: Emotion & Executive ===

        # Combine: Python arousal (prediction-based) + Rust arousal (emotion-based)
        # Use max to preserve signal from either source
        self.neuromod.arousal = max(python_arousal, rust_arousal)
        self.neuromod.valence = rust_valence
        self.neuromod.focus = rust_focus
        self.neuromod.energy = rust_energy

        # Sync the merged neuromodulator state back to Rust for the next tick.
        brain_core.set_neuromodulator(
            self.neuromod.arousal,
            self.neuromod.valence,
            self.neuromod.focus,
            self.neuromod.energy,
        )

        # === Phase 7: Language & Speech ===

        # === Phase 8: Full I/O ===

        # Read motor action from batch state
        if motor_approach > 0 and motor_withdraw > 0:
            motor_action_str = "conflict"
        elif motor_approach > 0:
            motor_action_str = "approach"
        elif motor_withdraw > 0:
            motor_action_str = "withdraw"
        else:
            motor_action_str = "idle"

        # === Phase 9: Homeostasis & Sleep ===

        if learn:
            # Record active traces for dream replay content
            if not brain_core.is_asleep():
                self.homeostasis.record_active_traces(
                    [tid for tid, _ in active_traces]
                )

            # Homeostasis step (dream replay during REM, wake alarm, etc.)
            homeo_stats = self.homeostasis.step(tick_num, self.neuromod)

            # Consolidation: integrate with sleep cycle
            # During sleep: consolidation can be triggered by deep sleep phase
            # While awake: original energy/tick triggers still apply
            if self.consolidation.is_consolidating:
                consol_stats = self.consolidation.consolidation_step(tick_num, self.trace_store)
                # Recover energy during consolidation
                brain_core.recover_energy(0.001)
                self.neuromod.energy = min(1.0, self.neuromod.energy + 0.001)
                if consol_stats.get("finished"):
                    self._awake_trace_ids = []  # Reset awake tracking
                    self.homeostasis.mark_consolidation_done()
            elif brain_core.is_asleep() and self.homeostasis.should_consolidate_in_sleep():
                # Sleep-triggered consolidation (during deep sleep)
                self.consolidation.start_consolidation(
                    tick_num, self.trace_store,
                    self.homeostasis.get_dream_candidates(),
                )
            elif self.consolidation.should_consolidate(tick_num, self.neuromod):
                # Normal consolidation trigger (energy/tick based). Keep this active
                # even if the same low-energy tick has already pushed the brain into
                # a drowsy sleep state.
                self.consolidation.start_consolidation(
                    tick_num, self.trace_store, self._awake_trace_ids,
                )

            # === Phase 4: Maintain (periodic) ===
            maintain_started = time.perf_counter()
            synapse_update_profile = self._maintain(tick_num)
            maintain_ms = (time.perf_counter() - maintain_started) * 1000
        else:
            sleep_pressure, circadian_phase, _ticks_awake, _ticks_asleep = brain_core.get_homeostasis_summary()
            sleep_state, _ticks_in_state, _cycles_completed, _rem_episodes = brain_core.get_sleep_summary()
            homeo_stats = {
                "sleep_state": sleep_state,
                "sleep_pressure": sleep_pressure,
                "circadian_phase": circadian_phase,
                "is_asleep": brain_core.is_asleep(),
                "in_rem": brain_core.in_rem(),
                "dream_replayed": 0,
            }
            synapse_update_profile = {}
            maintain_ms = 0.0
        step_internal_ms = (time.perf_counter() - step_started) * 1000
        other_python_ms = max(
            0.0,
            step_internal_ms
            - rust_tick_ms
            - evaluation_ms
            - learn_step_ms
            - formation_ms
            - maintain_ms,
        )
        other_python_non_binding_recall_ms = max(
            0.0,
            other_python_ms
            - binding_recall_python_ms
            - binding_recall_completion_rust_ms,
        )

        result = {
            "tick": tick_num,
            "executed_ticks": int(executed_ticks),
            "total_active": total_active,
            "snapshot_total_active": snapshot_total_active,
            "active_traces": len(interval_active_traces),
            "trace_candidates": int(trace_candidates),
            "working_memory": len(self.working_memory),
            "novelty": novelty,
            "hebbian_updates": hebb_count,
            "anti_hebbian_updates": anti_count,
            "phase": get_phase(tick_num),
            "arousal": self.neuromod.arousal,
            "valence": self.neuromod.valence,
            "energy": self.neuromod.energy,
            "in_surprise": self.prediction.in_surprise,
            "in_alarm": self.prediction.in_alarm,
            "learning_multiplier": self.prediction.learning_rate_multiplier,
            "traces_formed": traces_formed,
            "binding_candidates": binding_stats.get("candidates", 0),
            "bindings_formed": binding_stats.get("formed", 0),
            "total_bindings": binding_stats.get("total_bindings", 0),
            "rust_tick_ms": rust_tick_ms,
            "tick_prepare_ms": tick_prepare_ms,
            "tick_delayed_delivery_ms": tick_delayed_delivery_ms,
            "tick_propagate_ms": tick_propagate_ms,
            "tick_update_ms": tick_update_ms,
            "evaluation_ms": evaluation_ms,
            "evaluation_rust_ms": evaluation_rust_ms,
            "snapshot_ms": snapshot_ms,
            "batch_state_ms": batch_state_ms,
            "trace_match_ms": trace_match_ms,
            "trace_side_effects_ms": trace_side_effects_ms,
            "binding_recall_ms": binding_recall_ms,
            "binding_recall_python_ms": binding_recall_python_ms,
            "learn_step_ms": learn_step_ms,
            "formation_ms": formation_ms,
            "maintain_ms": maintain_ms,
            "other_python_ms": other_python_ms,
            "other_python_non_binding_recall_ms": other_python_non_binding_recall_ms,
            "step_internal_ms": step_internal_ms,
        }

        if not self.collect_full_metrics:
            return result

        result.update(
            {
                "consolidating": self.consolidation.is_consolidating,
                "emotion_polarity": emotion_polarity,
                "emotion_arousal": emotion_arousal,
                "executive_engagement": exec_engagement,
                "motor_conflict": motor_conflict,
                "planning_signal": planning,
                "language_activation": language_activation,
                "inner_monologue": inner_monologue,
                "speech_activity": speech_activity,
                "sensory_activation": sensory_activation,
                "visual_activation": visual_activation,
                "audio_activation": audio_activation,
                "motor_activation": motor_activation,
                "motor_action": motor_action_str,
                "motor_approach": motor_approach,
                "motor_withdraw": motor_withdraw,
                "pain_level": pain_level,
                "sleep_state": homeo_stats.get("sleep_state", "awake"),
                "sleep_pressure": homeo_stats.get("sleep_pressure", 0.0),
                "circadian_phase": homeo_stats.get("circadian_phase", 0.0),
                "is_asleep": homeo_stats.get("is_asleep", False),
                "in_rem": homeo_stats.get("in_rem", False),
                "dream_replayed": homeo_stats.get("dream_replayed", 0),
                "incoming_signal_count": incoming_signal_count,
                "incoming_signal_abs_sum": incoming_signal_abs_sum,
                "immediate_signal_count": immediate_signal_count,
                "immediate_signal_abs_sum": immediate_signal_abs_sum,
                "delayed_delivery_signal_count": delayed_delivery_signal_count,
                "delayed_delivery_signal_abs_sum": delayed_delivery_signal_abs_sum,
                "scheduled_delayed_signal_count": scheduled_delayed_signal_count,
                "scheduled_delayed_signal_abs_sum": scheduled_delayed_signal_abs_sum,
                "total_fired": total_fired,
                "refractory_ignored_abs_sum": refractory_ignored_abs_sum,
                "fire_interval_sum": fire_interval_sum,
                "fire_interval_count": fire_interval_count,
                "working_memory_boost_neurons": working_memory_boost_neurons,
                "pattern_completion_neurons": pattern_completion_neurons,
                "speech_boost_neurons": speech_boost_neurons,
                "binding_recall_python_replace_working_memory_ms": binding_recall_python_replace_working_memory_ms,
                "binding_recall_python_build_weights_ms": binding_recall_python_build_weights_ms,
                "binding_recall_python_merge_active_bindings_ms": binding_recall_python_merge_active_bindings_ms,
                "binding_recall_python_detail_check_ms": binding_recall_python_detail_check_ms,
                "binding_recall_python_augment_ms": binding_recall_python_augment_ms,
                "binding_recall_python_working_memory_update_ms": binding_recall_python_working_memory_update_ms,
                "binding_recall_python_pattern_complete_ms": binding_recall_python_pattern_complete_ms,
                "binding_recall_completion_rust_ms": binding_recall_completion_rust_ms,
                "binding_recall_bindings": binding_recall_bindings,
                "binding_recall_neurons": binding_recall_neurons,
                "binding_recall_max_relative_weight": binding_recall_max_relative_weight,
                "binding_recall_max_boost": binding_recall_max_boost,
                "binding_recall_trace_reactivations": len(recall_reactivated_traces),
                "binding_pattern_completion_traces": binding_pattern_completion_traces,
                "binding_pattern_completion_neurons": binding_pattern_completion_neurons,
                "synapse_update_pending_count": synapse_update_profile.get("pending_update_count", 0.0),
                "synapse_update_deferred_count": synapse_update_profile.get("deferred_update_count", 0.0),
                "synapse_update_applied_count": synapse_update_profile.get("applied_update_count", 0.0),
                "synapse_update_unmatched_count": synapse_update_profile.get("unmatched_update_count", 0.0),
                "synapse_update_positive_count": synapse_update_profile.get("positive_update_count", 0.0),
                "synapse_update_negative_count": synapse_update_profile.get("negative_update_count", 0.0),
                "synapse_update_delta_sum": synapse_update_profile.get("delta_sum", 0.0),
                "synapse_update_delta_abs_sum": synapse_update_profile.get("delta_abs_sum", 0.0),
                "synapse_update_delta_min": synapse_update_profile.get("delta_min", 0.0),
                "synapse_update_delta_max": synapse_update_profile.get("delta_max", 0.0),
                "synapse_update_release_interval": synapse_update_profile.get("release_interval", 0.0),
                "synapse_update_release_max_batch": synapse_update_profile.get("release_max_batch", 0.0),
                "synapse_update_before_weight_avg": synapse_update_profile.get("before_weight_avg", 0.0),
                "synapse_update_after_weight_avg": synapse_update_profile.get("after_weight_avg", 0.0),
                "synapse_update_crossed_up_0p05_count": synapse_update_profile.get("crossed_up_0p05_count", 0.0),
                "synapse_update_crossed_up_0p10_count": synapse_update_profile.get("crossed_up_0p10_count", 0.0),
                "synapse_update_crossed_up_0p20_count": synapse_update_profile.get("crossed_up_0p20_count", 0.0),
                "synapse_update_crossed_down_0p05_count": synapse_update_profile.get("crossed_down_0p05_count", 0.0),
                "synapse_update_crossed_down_0p10_count": synapse_update_profile.get("crossed_down_0p10_count", 0.0),
                "synapse_update_crossed_down_0p20_count": synapse_update_profile.get("crossed_down_0p20_count", 0.0),
                "synapse_pruned_count": synapse_update_profile.get("pruned_synapse_count", 0.0),
                "binding_pruned_count": synapse_update_profile.get("binding_pruned_count", 0.0),
                "coactivation_cleanup_ran": synapse_update_profile.get("coactivation_cleanup_ran", 0.0),
                "synapse_rebuild_ran": synapse_update_profile.get("synapse_rebuild_ran", 0.0),
                "synapse_update_memory_long_same_count": synapse_update_profile.get(
                    "region_pair_memory_long_to_memory_long_count", 0.0
                ),
                "synapse_update_memory_long_same_delta_abs_sum": synapse_update_profile.get(
                    "region_pair_memory_long_to_memory_long_delta_abs_sum", 0.0
                ),
                "synapse_update_memory_long_same_before_weight_avg": synapse_update_profile.get(
                    "region_pair_memory_long_to_memory_long_before_weight_avg", 0.0
                ),
                "synapse_update_memory_long_same_after_weight_avg": synapse_update_profile.get(
                    "region_pair_memory_long_to_memory_long_after_weight_avg", 0.0
                ),
                "synapse_update_delay_9_count": synapse_update_profile.get("delay_9_count", 0.0),
                "synapse_update_delay_10_count": synapse_update_profile.get("delay_10_count", 0.0),
                **region_active_metrics,
                **propagation_region_metrics,
                **propagation_region_abs_metrics,
                **fired_region_metrics,
                **refractory_region_metrics,
                **fire_interval_region_metrics,
                **potential_region_metrics,
                **delayed_flow_metrics,
                **boost_region_metrics,
                **augment_binding_recall_profile,
                **pattern_complete_binding_recall_profile,
                **binding_recall_completion_profile,
            }
        )
        return result

    def _augment_binding_recall_active_traces(
        self,
        snapshot: ActivationSnapshot,
        active_traces: list[tuple[str, float]],
        binding_recall_weights: dict[int, float],
        return_profile: bool = False,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]] | tuple[
        list[tuple[str, float]],
        list[tuple[str, float]],
        dict[str, float],
    ]:
        """Reactivate binding-linked traces using overlap alone when recall is already engaged."""
        profile: dict[str, float] = {
            "binding_recall_python_augment_active_set_ms": 0.0,
            "binding_recall_python_augment_detail_lookup_ms": 0.0,
            "binding_recall_python_augment_overlap_ms": 0.0,
            "binding_recall_python_augment_sort_ms": 0.0,
            "binding_recall_python_augment_binding_count": float(
                len(binding_recall_weights)
            ),
            "binding_recall_python_augment_trace_checks": 0.0,
        }
        active_trace_scores = {trace_id: float(score) for trace_id, score in active_traces}
        reactivated: list[tuple[str, float]] = []

        active_set_started = time.perf_counter()
        active_set = snapshot.active_set()
        profile["binding_recall_python_augment_active_set_ms"] = (
            time.perf_counter() - active_set_started
        ) * 1000

        detail_lookup_started = time.perf_counter()
        detail_map = self.binding_formation.binding_details
        profile["binding_recall_python_augment_detail_lookup_ms"] = (
            time.perf_counter() - detail_lookup_started
        ) * 1000

        overlap_started = time.perf_counter()
        for binding_id in binding_recall_weights:
            detail = detail_map.get(int(binding_id))
            if detail is None:
                continue

            for trace_id in (str(detail["trace_id_a"]), str(detail["trace_id_b"])):
                profile["binding_recall_python_augment_trace_checks"] += 1
                if trace_id in active_trace_scores:
                    continue

                trace = self.trace_store.get(trace_id)
                if trace is None:
                    continue

                total_neurons = trace.total_neurons()
                if total_neurons <= 0:
                    continue

                active_neurons = sum(
                    1
                    for region_neurons in trace.neurons.values()
                    for neuron_id in region_neurons
                    if neuron_id in active_set
                )
                overlap_ratio = active_neurons / total_neurons
                if overlap_ratio < BINDING_RECALL_TRACE_MATCH_THRESHOLD:
                    continue

                active_trace_scores[trace_id] = overlap_ratio
                reactivated.append((trace_id, overlap_ratio))
        profile["binding_recall_python_augment_overlap_ms"] = (
            time.perf_counter() - overlap_started
        ) * 1000

        sort_started = time.perf_counter()
        augmented = sorted(
            active_trace_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        profile["binding_recall_python_augment_sort_ms"] = (
            time.perf_counter() - sort_started
        ) * 1000
        if return_profile:
            return augmented, reactivated, profile
        return augmented, reactivated

    def _pattern_complete_binding_recall_traces(
        self,
        binding_recall_weights: dict[int, float],
        active_traces: list[tuple[str, float]],
        return_profile: bool = False,
    ) -> tuple[int, int] | tuple[int, int, dict[str, float]]:
        """Reuse memory_long pattern completion for partially reactivated bound traces."""
        profile: dict[str, float] = {
            "binding_recall_python_pattern_complete_activation_fetch_ms": 0.0,
            "binding_recall_python_pattern_complete_activation_map_ms": 0.0,
            "binding_recall_python_pattern_complete_detail_lookup_ms": 0.0,
            "binding_recall_python_pattern_complete_apply_ms": 0.0,
            "binding_recall_python_pattern_complete_binding_count": float(
                len(binding_recall_weights)
            ),
            "binding_recall_python_pattern_complete_endpoint_checks": 0.0,
            "binding_recall_python_pattern_complete_calls": 0.0,
        }
        activation_fetch_started = time.perf_counter()
        activation_rows = brain_core.get_binding_activations(
            list(binding_recall_weights),
            0.01,
        )
        profile["binding_recall_python_pattern_complete_activation_fetch_ms"] = (
            time.perf_counter() - activation_fetch_started
        ) * 1000

        activation_map_started = time.perf_counter()
        activation_by_binding = {
            int(binding_id): (float(ratio_a), float(ratio_b))
            for binding_id, ratio_a, ratio_b in activation_rows
        }
        profile["binding_recall_python_pattern_complete_activation_map_ms"] = (
            time.perf_counter() - activation_map_started
        ) * 1000
        active_trace_ids = {trace_id for trace_id, _ in active_traces}
        completed_trace_ids: set[str] = set()
        completed_traces = 0
        completed_neurons = 0

        detail_lookup_started = time.perf_counter()
        detail_map = self.binding_formation.binding_details
        profile["binding_recall_python_pattern_complete_detail_lookup_ms"] = (
            time.perf_counter() - detail_lookup_started
        ) * 1000

        apply_started = time.perf_counter()
        for binding_id, relative_weight in binding_recall_weights.items():
            detail = detail_map.get(int(binding_id))
            if detail is None:
                continue

            ratios = activation_by_binding.get(int(binding_id))
            if ratios is None:
                continue

            ratio_a, ratio_b = ratios
            endpoints = (
                (str(detail["trace_id_a"]), ratio_a),
                (str(detail["trace_id_b"]), ratio_b),
            )

            for trace_id, endpoint_ratio in endpoints:
                profile["binding_recall_python_pattern_complete_endpoint_checks"] += 1
                if endpoint_ratio < TRACE_ACTIVATION_THRESHOLD:
                    continue
                if trace_id in active_trace_ids or trace_id in completed_trace_ids:
                    continue

                trace = self.trace_store.get(trace_id)
                if trace is None:
                    continue

                memory_long_neurons = list(trace.neurons.get("memory_long", []))
                if not memory_long_neurons:
                    continue

                boost = (
                    BINDING_RECALL_PATTERN_COMPLETION_BOOST
                    * max(0.0, float(relative_weight))
                    * max(0.0, float(endpoint_ratio))
                )
                if boost <= 0.0:
                    continue

                boosted = brain_core.pattern_complete(
                    memory_long_neurons,
                    BINDING_RECALL_PATTERN_COMPLETION_THRESHOLD,
                    boost,
                )
                if boosted <= 0:
                    continue

                profile["binding_recall_python_pattern_complete_calls"] += 1
                completed_trace_ids.add(trace_id)
                completed_traces += 1
                completed_neurons += boosted
        profile["binding_recall_python_pattern_complete_apply_ms"] = (
            time.perf_counter() - apply_started
        ) * 1000

        if return_profile:
            return completed_traces, completed_neurons, profile
        return completed_traces, completed_neurons

    def _track_synapse_fires(self, snapshot: ActivationSnapshot, tick: int) -> None:
        """Track last-fired tick for synapses between active neurons.

        Uses Rust batch_track_coactive for parallelized synapse traversal.
        """
        active_ids = snapshot.active_ids
        if not active_ids:
            return
        active_set = set(active_ids)
        pairs = brain_core.batch_track_coactive(active_ids, active_set)
        for src_id, tgt_id in pairs:
            self._synapse_last_fired[(src_id, tgt_id)] = tick

    def _get_working_memory_neurons(self) -> list[int]:
        """Get all neurons belonging to working-memory traces.

        Retained for compatibility with tests and cold-path inspection.
        """
        neurons = []
        for tid in self.working_memory.trace_ids:
            trace = self.trace_store.get(tid)
            if trace is not None:
                neurons.extend(trace.neurons.get("memory_short", []))
        return neurons

    def _maintain(self, tick: int) -> dict[str, float]:
        """Periodic maintenance: apply updates, prune, rebuild."""
        maintenance_profile: dict[str, float] = {
            "pruned_synapse_count": 0.0,
            "binding_pruned_count": 0.0,
            "coactivation_cleanup_ran": 0.0,
            "synapse_rebuild_ran": 0.0,
        }
        if tick <= 0:
            self.neuromod.energy = max(0.0, self.neuromod.energy - 0.0001)
            self.neuromod.clamp()
            self.last_synapse_update_profile = maintenance_profile
            return maintenance_profile

        # Release queued weight updates in bounded batches so longer runs do
        # not hit a single maintenance cliff.
        if tick % self._synapse_update_release_interval == 0:
            synapse_count = brain_core.get_synapse_count()
            pending_updates = brain_core.get_pending_synapse_update_count()
            base_batch = max(
                1,
                int(
                    synapse_count * SYNAPSE_UPDATE_MAX_BATCH_SYNAPSE_MULTIPLIER
                ),
            )
            target_deferred = max(
                1,
                int(
                    synapse_count
                    * SYNAPSE_UPDATE_TARGET_DEFERRED_SYNAPSE_MULTIPLIER
                ),
            )
            max_batch = min(
                pending_updates,
                max(base_batch, pending_updates - target_deferred),
            )
            maintenance_profile = brain_core.apply_synapse_updates_profiled_bounded(
                max_batch
            )
            maintenance_profile["pruned_synapse_count"] = 0.0
            maintenance_profile["binding_pruned_count"] = 0.0
            maintenance_profile["coactivation_cleanup_ran"] = 0.0
            maintenance_profile["synapse_rebuild_ran"] = 0.0
            maintenance_profile["release_interval"] = self._synapse_update_release_interval
            maintenance_profile["release_max_batch"] = max_batch

        # Periodic incremental pruning pass
        if tick % self._prune_interval == 0:
            from brain.utils.config import TOTAL_NEURONS
            start = self._prune_offset
            end = min(start + self._prune_batch_size, TOTAL_NEURONS)
            sample = list(range(start, end))
            maintenance_profile["pruned_synapse_count"] = float(
                pruning_pass_sampled(tick, sample, self._synapse_last_fired)
            )
            self._prune_offset = end if end < TOTAL_NEURONS else 0

        # Periodic binding prune + stale co-activation cleanup
        if tick % self._binding_maintenance_interval == 0:
            maintenance_profile["binding_pruned_count"] = float(
                self.binding_formation.periodic_prune()
            )
            self.binding_formation.periodic_cleanup(tick)
            maintenance_profile["coactivation_cleanup_ran"] = 1.0

        # Periodic full CSR rebuild + trace merge check
        if tick % self._rebuild_interval == 0:
            brain_core.rebuild_synapse_index()
            self.trace_formation.merge_overlapping()
            maintenance_profile["synapse_rebuild_ran"] = 1.0

        # Slow energy decay
        self.neuromod.energy = max(0.0, self.neuromod.energy - 0.0001)
        self.neuromod.clamp()
        self.last_synapse_update_profile = maintenance_profile
        return maintenance_profile

    def export_checkpoint_state(self) -> dict:
        return {
            "history": self.history.to_dict(),
            "neuromod": self.neuromod.to_dict(),
            "prediction": {
                "predicted": dict(self.prediction._predicted),
                "ema_rates": dict(self.prediction._ema_rates),
                "alpha": float(self.prediction._alpha),
                "surprise_remaining": int(self.prediction._surprise_remaining),
                "alarm_remaining": int(self.prediction._alarm_remaining),
            },
            "working_memory": {
                "capacity": int(self.working_memory.capacity),
                "slots": [(str(tid), float(score)) for tid, score in self.working_memory.slots],
            },
            "consolidation": {
                "last_consolidation_tick": int(self.consolidation._last_consolidation_tick),
                "consolidating": bool(self.consolidation._consolidating),
                "consolidation_start": int(self.consolidation._consolidation_start),
                "consolidation_queue": list(self.consolidation._consolidation_queue),
                "consolidation_idx": int(self.consolidation._consolidation_idx),
            },
            "binding_formation": {
                "bound_pairs": [tuple(pair) for pair in self.binding_formation._bound_pairs],
                "binding_details": dict(self.binding_formation._binding_details),
            },
            "homeostasis": {
                "dream_queue": list(self.homeostasis._dream_queue),
                "dream_idx": int(self.homeostasis._dream_idx),
                "dreams_per_tick": int(self.homeostasis._dreams_per_tick),
                "recent_trace_ids": list(self.homeostasis._recent_trace_ids),
                "recent_max": int(self.homeostasis._recent_max),
                "in_sleep_session": bool(self.homeostasis._in_sleep_session),
                "sleep_session": {
                    "total_ticks": int(self.homeostasis._sleep_session.total_ticks),
                    "rem_episodes": int(self.homeostasis._sleep_session.rem_episodes),
                    "traces_replayed": int(self.homeostasis._sleep_session.traces_replayed),
                    "cycles_completed": int(self.homeostasis._sleep_session.cycles_completed),
                    "energy_start": float(self.homeostasis._sleep_session.energy_start),
                    "energy_end": float(self.homeostasis._sleep_session.energy_end),
                },
                "consolidation_done_this_sleep": bool(self.homeostasis._consolidation_done_this_sleep),
                "wake_alarm_threshold": float(self.homeostasis._wake_alarm_threshold),
            },
            "awake_trace_ids": list(self._awake_trace_ids),
            "last_tick_number": int(self.last_tick_number),
            "last_total_active": int(self.last_total_active),
            "last_novelty": float(self.last_novelty),
            "last_hebbian_updates": int(self.last_hebbian_updates),
            "last_anti_hebbian_updates": int(self.last_anti_hebbian_updates),
            "last_active_traces": [
                (str(trace_id), float(score))
                for trace_id, score in self.last_active_traces
            ],
            "working_memory_overlay_cap": int(self.working_memory_overlay_cap),
            "synapse_last_fired": [
                (int(src_id), int(tgt_id), int(tick))
                for (src_id, tgt_id), tick in self._synapse_last_fired.items()
            ],
            "prune_offset": int(self._prune_offset),
            "maintenance": {
                "synapse_update_release_interval": int(self._synapse_update_release_interval),
                "prune_interval": int(self._prune_interval),
                "binding_maintenance_interval": int(self._binding_maintenance_interval),
                "rebuild_interval": int(self._rebuild_interval),
                "prune_batch_size": int(self._prune_batch_size),
            },
        }

    def export_async_sync_state(self) -> dict:
        return {
            "prediction": {
                "predicted": dict(self.prediction._predicted),
                "ema_rates": dict(self.prediction._ema_rates),
                "alpha": float(self.prediction._alpha),
                "surprise_remaining": int(self.prediction._surprise_remaining),
                "alarm_remaining": int(self.prediction._alarm_remaining),
            },
            "working_memory": {
                "capacity": int(self.working_memory.capacity),
                "slots": [
                    (str(trace_id), float(score))
                    for trace_id, score in self.working_memory.slots
                ],
            },
        }

    def apply_async_sync_state(self, state: dict) -> None:
        prediction_state = dict(state.get("prediction", {}))
        self.prediction = PredictionEngine(self.trace_store)
        self.prediction._predicted = {
            str(region_name): float(value)
            for region_name, value in dict(prediction_state.get("predicted", {})).items()
        }
        self.prediction._ema_rates = {
            str(region_name): float(value)
            for region_name, value in dict(prediction_state.get("ema_rates", {})).items()
        }
        self.prediction._alpha = float(prediction_state.get("alpha", self.prediction._alpha))
        self.prediction._surprise_remaining = int(
            prediction_state.get("surprise_remaining", 0)
        )
        self.prediction._alarm_remaining = int(
            prediction_state.get("alarm_remaining", 0)
        )

        working_memory_state = dict(state.get("working_memory", {}))
        self.working_memory = WorkingMemory(
            capacity=int(working_memory_state.get("capacity", WORKING_MEMORY_CAPACITY))
        )
        self.working_memory.slots = [
            (str(trace_id), float(score))
            for trace_id, score in list(working_memory_state.get("slots", []))
        ]

        self.last_active_traces = [
            (str(trace_id), float(score))
            for trace_id, score in list(state.get("last_active_traces", []))
        ]

        brain_core.trace_index_set_working_memory(
            self.trace_store.store_id,
            self.working_memory.slots,
        )

    def restore_checkpoint_state(self, state: dict) -> None:
        self.history = ActivationHistory.from_dict(dict(state.get("history", {})))
        self.neuromod = NeuromodulatorState.from_dict(dict(state.get("neuromod", {})))

        prediction_state = dict(state.get("prediction", {}))
        self.prediction = PredictionEngine(self.trace_store)
        self.prediction._predicted = {
            str(region_name): float(value)
            for region_name, value in dict(prediction_state.get("predicted", {})).items()
        }
        self.prediction._ema_rates = {
            str(region_name): float(value)
            for region_name, value in dict(prediction_state.get("ema_rates", {})).items()
        }
        self.prediction._alpha = float(prediction_state.get("alpha", self.prediction._alpha))
        self.prediction._surprise_remaining = int(
            prediction_state.get("surprise_remaining", 0)
        )
        self.prediction._alarm_remaining = int(prediction_state.get("alarm_remaining", 0))

        working_memory_state = dict(state.get("working_memory", {}))
        self.working_memory = WorkingMemory(
            capacity=int(working_memory_state.get("capacity", WORKING_MEMORY_CAPACITY))
        )
        self.working_memory.slots = [
            (str(trace_id), float(score))
            for trace_id, score in list(working_memory_state.get("slots", []))
        ]

        consolidation_state = dict(state.get("consolidation", {}))
        self.consolidation = ConsolidationEngine()
        self.consolidation._last_consolidation_tick = int(
            consolidation_state.get("last_consolidation_tick", 0)
        )
        self.consolidation._consolidating = bool(
            consolidation_state.get("consolidating", False)
        )
        self.consolidation._consolidation_start = int(
            consolidation_state.get("consolidation_start", 0)
        )
        self.consolidation._consolidation_queue = list(
            consolidation_state.get("consolidation_queue", [])
        )
        self.consolidation._consolidation_idx = int(
            consolidation_state.get("consolidation_idx", 0)
        )

        self.trace_formation = TraceFormationEngine(self.trace_store)

        binding_state = dict(state.get("binding_formation", {}))
        self.binding_formation = BindingFormationEngine(self.trace_store)
        self.binding_formation._bound_pairs = {
            tuple(pair)
            for pair in list(binding_state.get("bound_pairs", []))
        }
        self.binding_formation._binding_details = {
            int(binding_id): dict(detail)
            for binding_id, detail in dict(binding_state.get("binding_details", {})).items()
        }

        homeostasis_state = dict(state.get("homeostasis", {}))
        self.homeostasis = HomeostasisManager(self.trace_store)
        self.homeostasis._dream_queue = list(homeostasis_state.get("dream_queue", []))
        self.homeostasis._dream_idx = int(homeostasis_state.get("dream_idx", 0))
        self.homeostasis._dreams_per_tick = int(homeostasis_state.get("dreams_per_tick", 3))
        self.homeostasis._recent_trace_ids = list(homeostasis_state.get("recent_trace_ids", []))
        self.homeostasis._recent_max = int(homeostasis_state.get("recent_max", 500))
        self.homeostasis._in_sleep_session = bool(
            homeostasis_state.get("in_sleep_session", False)
        )
        sleep_session_state = dict(homeostasis_state.get("sleep_session", {}))
        self.homeostasis._sleep_session = SleepSessionStats(
            total_ticks=int(sleep_session_state.get("total_ticks", 0)),
            rem_episodes=int(sleep_session_state.get("rem_episodes", 0)),
            traces_replayed=int(sleep_session_state.get("traces_replayed", 0)),
            cycles_completed=int(sleep_session_state.get("cycles_completed", 0)),
            energy_start=float(sleep_session_state.get("energy_start", 0.0)),
            energy_end=float(sleep_session_state.get("energy_end", 0.0)),
        )
        self.homeostasis._consolidation_done_this_sleep = bool(
            homeostasis_state.get("consolidation_done_this_sleep", False)
        )
        self.homeostasis._wake_alarm_threshold = float(
            homeostasis_state.get("wake_alarm_threshold", 0.8)
        )

        self._awake_trace_ids = list(state.get("awake_trace_ids", []))
        self.last_tick_number = int(state.get("last_tick_number", 0))
        self.last_total_active = int(state.get("last_total_active", 0))
        self.last_novelty = float(state.get("last_novelty", 0.0))
        self.last_hebbian_updates = int(state.get("last_hebbian_updates", 0))
        self.last_anti_hebbian_updates = int(state.get("last_anti_hebbian_updates", 0))
        self.last_active_traces = [
            (str(trace_id), float(score))
            for trace_id, score in list(state.get("last_active_traces", []))
        ]
        self.last_binding_recall_candidates = []
        self.last_synapse_update_profile = {}
        self.working_memory_overlay_cap = int(state.get("working_memory_overlay_cap", 0))
        self._synapse_last_fired = {
            (int(src_id), int(tgt_id)): int(tick)
            for src_id, tgt_id, tick in list(state.get("synapse_last_fired", []))
        }
        self._prune_offset = int(state.get("prune_offset", 0))
        maintenance_state = dict(state.get("maintenance", {}))
        self._synapse_update_release_interval = max(
            1,
            int(
                maintenance_state.get(
                    "synapse_update_release_interval",
                    self._synapse_update_release_interval,
                )
            ),
        )
        self._prune_interval = max(
            1,
            int(maintenance_state.get("prune_interval", self._prune_interval)),
        )
        self._binding_maintenance_interval = max(
            1,
            int(
                maintenance_state.get(
                    "binding_maintenance_interval",
                    self._binding_maintenance_interval,
                )
            ),
        )
        self._rebuild_interval = max(
            1,
            int(maintenance_state.get("rebuild_interval", self._rebuild_interval)),
        )
        self._prune_batch_size = max(
            1,
            int(maintenance_state.get("prune_batch_size", self._prune_batch_size)),
        )
        brain_core.trace_index_set_working_memory(
            self.trace_store.store_id,
            self.working_memory.slots,
        )

    def reset_sample_boundary(self) -> None:
        """Clear sample-scoped learning state that should not bleed across inputs."""
        self.binding_formation.reset_sample_boundary()
        self.trace_formation.reset_sample_boundary()

    def reset_working_memory_boundary(self) -> None:
        """Clear working-memory slots while preserving other runtime state."""
        brain_core.trace_index_clear_working_memory(self.trace_store.store_id)
        self.working_memory = WorkingMemory()

    def reset_runtime_boundary(self, preserve_binding_state: bool = False) -> None:
        """Clear transient runtime state while preserving learned structure.

        When preserve_binding_state is true, keep binding formation bookkeeping
        and only clear its sample-scoped co-activation tracker.
        """
        brain_core.reset_runtime_state()
        self.trace_store.reset_runtime_index()
        self.history = ActivationHistory(window=HEBBIAN_WINDOW)
        self.neuromod = NeuromodulatorState()
        self.prediction = PredictionEngine(self.trace_store)
        self.working_memory = WorkingMemory()
        self.consolidation = ConsolidationEngine()
        self.trace_formation = TraceFormationEngine(self.trace_store)
        if preserve_binding_state:
            self.binding_formation.reset_sample_boundary()
        else:
            self.binding_formation = BindingFormationEngine(self.trace_store)
        self.homeostasis = HomeostasisManager(self.trace_store)
        self._awake_trace_ids = []
        self.last_tick_number = 0
        self.last_total_active = 0
        self.last_novelty = 0.0
        self.last_hebbian_updates = 0
        self.last_anti_hebbian_updates = 0
        self.last_active_traces = []
        self.last_binding_recall_candidates = []
        self.last_synapse_update_profile = {}
        self._synapse_last_fired = {}
        self._prune_offset = 0

    def reset_probe_boundary(self) -> None:
        """Clear transient runtime state while preserving learned structure."""
        self.reset_runtime_boundary(preserve_binding_state=True)
