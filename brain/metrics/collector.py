"""Rich JSON metrics collector for brain learning sessions.

Collects per-tick, per-sample, and global session metrics.
All data stored in a single JSON structure for analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brain.structures.neuron_map import all_region_names


BOOST_TYPES = (
    "working_memory_boost",
    "pattern_completion",
    "speech_boost",
    "binding_recall",
)

BOOST_NUMERIC_KEYS = [
    "working_memory_boost_neurons",
    "pattern_completion_neurons",
    "speech_boost_neurons",
]

BOOST_REGION_NUMERIC_KEYS = [
    f"{boost_type}_region_{region_name}_neurons"
    for boost_type in BOOST_TYPES
    for region_name in all_region_names()
]

REGION_ACTIVE_NUMERIC_KEYS = [
    f"active_region_{region_name}_neurons"
    for region_name in all_region_names()
]

PROPAGATION_NUMERIC_KEYS = [
    "incoming_signal_count",
    "incoming_signal_abs_sum",
    "immediate_signal_count",
    "immediate_signal_abs_sum",
    "delayed_delivery_signal_count",
    "delayed_delivery_signal_abs_sum",
    "scheduled_delayed_signal_count",
    "scheduled_delayed_signal_abs_sum",
    "total_fired",
    "refractory_ignored_abs_sum",
    "fire_interval_sum",
    "fire_interval_count",
]

PROPAGATION_REGION_NUMERIC_KEYS = [
    f"incoming_region_{region_name}_signals"
    for region_name in all_region_names()
] + [
    f"incoming_region_{region_name}_abs_sum"
    for region_name in all_region_names()
] + [
    f"fired_region_{region_name}_neurons"
    for region_name in all_region_names()
]

POTENTIAL_REGION_NUMERIC_KEYS = [
    f"potential_region_{region_name}_pre_leak_sum"
    for region_name in all_region_names()
] + [
    f"potential_region_{region_name}_leak_loss_sum"
    for region_name in all_region_names()
] + [
    f"potential_region_{region_name}_reset_sum"
    for region_name in all_region_names()
] + [
    f"potential_region_{region_name}_carried_sum"
    for region_name in all_region_names()
]

REFRACTORY_REGION_NUMERIC_KEYS = [
    f"refractory_ignored_region_{region_name}_abs_sum"
    for region_name in all_region_names()
] + [
    f"refractory_ignored_region_{region_name}_immediate_same_abs_sum"
    for region_name in all_region_names()
] + [
    f"refractory_ignored_region_{region_name}_immediate_cross_abs_sum"
    for region_name in all_region_names()
] + [
    f"refractory_ignored_region_{region_name}_delayed_same_abs_sum"
    for region_name in all_region_names()
] + [
    f"refractory_ignored_region_{region_name}_delayed_cross_abs_sum"
    for region_name in all_region_names()
]

FIRE_INTERVAL_REGION_NUMERIC_KEYS = [
    f"fire_interval_region_{region_name}_sum"
    for region_name in all_region_names()
] + [
    f"fire_interval_region_{region_name}_count"
    for region_name in all_region_names()
]

DELAYED_FLOW_NUMERIC_KEYS = [
    f"delayed_flow_{source_region}_to_{target_region}_signals"
    for source_region in all_region_names()
    for target_region in all_region_names()
] + [
    f"delayed_flow_{source_region}_to_{target_region}_abs_sum"
    for source_region in all_region_names()
    for target_region in all_region_names()
]

RECURRENT_DELAYED_CANDIDATE_MIN_GLOBAL_SHARE = 0.01
RECURRENT_DELAYED_CANDIDATE_MIN_TARGET_SHARE = 0.25

PER_TICK_BATCH_TOTAL_KEYS = {
    "hebbian_updates",
    "anti_hebbian_updates",
    "traces_formed",
    "bindings_formed",
    "rust_tick_ms",
    "tick_prepare_ms",
    "tick_delayed_delivery_ms",
    "tick_propagate_ms",
    "tick_update_ms",
    "evaluation_ms",
    "evaluation_rust_ms",
    "snapshot_ms",
    "batch_state_ms",
    "trace_match_ms",
    "trace_side_effects_ms",
    "binding_recall_ms",
    "binding_recall_python_ms",
    "binding_recall_bindings",
    "binding_recall_neurons",
    "synapse_update_pending_count",
    "synapse_update_deferred_count",
    "synapse_update_applied_count",
    "synapse_update_unmatched_count",
    "synapse_update_positive_count",
    "synapse_update_negative_count",
    "synapse_update_delta_sum",
    "synapse_update_delta_abs_sum",
    "synapse_update_delta_min",
    "synapse_update_delta_max",
    "synapse_update_release_interval",
    "synapse_update_release_max_batch",
    "synapse_update_before_weight_avg",
    "synapse_update_after_weight_avg",
    "synapse_update_crossed_up_0p05_count",
    "synapse_update_crossed_up_0p10_count",
    "synapse_update_crossed_up_0p20_count",
    "synapse_update_crossed_down_0p05_count",
    "synapse_update_crossed_down_0p10_count",
    "synapse_update_crossed_down_0p20_count",
    "synapse_update_memory_long_same_count",
    "synapse_update_memory_long_same_delta_abs_sum",
    "synapse_update_memory_long_same_before_weight_avg",
    "synapse_update_memory_long_same_after_weight_avg",
    "synapse_update_delay_9_count",
    "synapse_update_delay_10_count",
    "learn_step_ms",
    "formation_ms",
    "maintain_ms",
    "other_python_ms",
    "other_python_non_binding_recall_ms",
    "step_internal_ms",
    *PROPAGATION_NUMERIC_KEYS,
    *PROPAGATION_REGION_NUMERIC_KEYS,
    *POTENTIAL_REGION_NUMERIC_KEYS,
    *REFRACTORY_REGION_NUMERIC_KEYS,
    *FIRE_INTERVAL_REGION_NUMERIC_KEYS,
    *DELAYED_FLOW_NUMERIC_KEYS,
}


@dataclass
class SampleMetrics:
    """Metrics for a single input sample."""

    index: int
    label: str | None = None
    modality: str = ""
    ticks: list[dict[str, Any]] = field(default_factory=list)

    def add_tick(self, tick_result: dict[str, Any]) -> None:
        self.ticks.append(tick_result)

    def summary(self) -> dict[str, Any]:
        if not self.ticks:
            return {}
        weights = [max(1, int(t.get("executed_ticks", 1) or 1)) for t in self.ticks]
        n = sum(weights)
        # Aggregate numeric fields
        numeric_keys = [
            "total_active", "snapshot_total_active", "active_traces", "trace_candidates", "working_memory", "total_bindings", "novelty",
            "hebbian_updates", "anti_hebbian_updates", "learning_multiplier",
            "traces_formed", "binding_candidates", "bindings_formed",
            "arousal", "valence", "energy",
            "emotion_polarity", "emotion_arousal", "executive_engagement",
            "language_activation", "speech_activity",
            "sensory_activation", "visual_activation", "audio_activation",
            "motor_activation", "motor_approach", "motor_withdraw", "pain_level",
            "sleep_pressure",
            "matched_trace_score", "matched_trace_rank",
            "cue_trace_score", "cue_trace_rank",
            "partner_trace_score", "partner_trace_rank",
            "competitor_trace_score", "competitor_trace_rank",
            "cue_pattern_ratio", "partner_pattern_ratio",
            "strong_source_pattern_ratio", "weak_source_pattern_ratio",
            "competitor_pattern_ratio",
            "false_trace_score", "false_trace_rank",
            "binding_weight", "binding_confidence", "binding_fires",
            "competitor_binding_weight", "competitor_binding_confidence",
            "competitor_binding_fires",
            "strong_recall_candidate_relative_weight",
            "strong_recall_candidate_source_ratio",
            "weak_recall_candidate_relative_weight",
            "weak_recall_candidate_source_ratio",
            "partner_peak_region_ratio", "partner_peak_region_active_neurons",
            "partner_peak_region_total_neurons",
            "competitor_peak_region_ratio", "competitor_peak_region_active_neurons",
            "competitor_peak_region_total_neurons",
            "strong_binding_relative_weight", "weak_binding_relative_weight",
            "binding_weight_margin", "pattern_ratio_margin", "trace_score_margin",
            "collision_target_overlap_ratio",
            "rust_tick_ms", "tick_prepare_ms", "tick_delayed_delivery_ms",
            "tick_propagate_ms", "tick_update_ms", "evaluation_ms",
            "evaluation_rust_ms", "snapshot_ms", "batch_state_ms", "trace_match_ms",
            *PROPAGATION_NUMERIC_KEYS,
            "trace_side_effects_ms", "binding_recall_ms", "binding_recall_python_ms", "binding_recall_bindings",
            "binding_recall_neurons", "binding_recall_max_relative_weight",
            "binding_recall_max_boost",
            "synapse_update_pending_count", "synapse_update_deferred_count",
            "synapse_update_applied_count", "synapse_update_unmatched_count",
            "synapse_update_positive_count", "synapse_update_negative_count",
            "synapse_update_delta_sum", "synapse_update_delta_abs_sum",
            "synapse_update_delta_min", "synapse_update_delta_max",
            "synapse_update_release_interval", "synapse_update_release_max_batch",
            "synapse_update_before_weight_avg", "synapse_update_after_weight_avg",
            "synapse_update_crossed_up_0p05_count", "synapse_update_crossed_up_0p10_count",
            "synapse_update_crossed_up_0p20_count", "synapse_update_crossed_down_0p05_count",
            "synapse_update_crossed_down_0p10_count", "synapse_update_crossed_down_0p20_count",
            "synapse_update_memory_long_same_count",
            "synapse_update_memory_long_same_delta_abs_sum",
            "synapse_update_memory_long_same_before_weight_avg",
            "synapse_update_memory_long_same_after_weight_avg",
            "synapse_update_delay_9_count", "synapse_update_delay_10_count",
            "learn_step_ms", "formation_ms", "maintain_ms", "other_python_ms",
            "other_python_non_binding_recall_ms",
            "step_internal_ms",
        ] + REGION_ACTIVE_NUMERIC_KEYS + PROPAGATION_REGION_NUMERIC_KEYS + POTENTIAL_REGION_NUMERIC_KEYS + REFRACTORY_REGION_NUMERIC_KEYS + FIRE_INTERVAL_REGION_NUMERIC_KEYS + DELAYED_FLOW_NUMERIC_KEYS + BOOST_NUMERIC_KEYS + BOOST_REGION_NUMERIC_KEYS
        agg: dict[str, Any] = {}
        for key in numeric_keys:
            weighted_vals = [
                (float(t.get(key, 0)), weight)
                for t, weight in zip(self.ticks, weights)
                if isinstance(t.get(key), (int, float))
            ]
            vals = [value for value, _weight in weighted_vals]
            if vals:
                if key in PER_TICK_BATCH_TOTAL_KEYS:
                    agg[f"{key}_avg"] = sum(value for value, _weight in weighted_vals) / n
                else:
                    agg[f"{key}_avg"] = sum(
                        value * weight for value, weight in weighted_vals
                    ) / sum(weight for _value, weight in weighted_vals)
                agg[f"{key}_max"] = max(vals)
                agg[f"{key}_min"] = min(vals)
                if key in (
                    "hebbian_updates",
                    "anti_hebbian_updates",
                    "traces_formed",
                    "trace_candidates",
                    "binding_candidates",
                    "bindings_formed",
                ):
                    agg[f"{key}_total"] = sum(vals)

        # Count-based aggregations
        agg["surprise_ticks"] = sum(1 for t in self.ticks if t.get("in_surprise"))
        agg["alarm_ticks"] = sum(1 for t in self.ticks if t.get("in_alarm"))
        agg["asleep_ticks"] = sum(1 for t in self.ticks if t.get("is_asleep"))
        agg["rem_ticks"] = sum(1 for t in self.ticks if t.get("in_rem"))
        agg["consolidating_ticks"] = sum(1 for t in self.ticks if t.get("consolidating"))
        if any("matched_trace_hit" in t for t in self.ticks):
            matched_hit_ticks = sum(1 for t in self.ticks if t.get("matched_trace_hit"))
            matched_top1_ticks = sum(1 for t in self.ticks if t.get("matched_trace_rank") == 1)
            agg["matched_trace_hit_ticks"] = matched_hit_ticks
            agg["matched_trace_hit_rate"] = matched_hit_ticks / n
            agg["matched_trace_top1_ticks"] = matched_top1_ticks
            agg["matched_trace_top1_rate"] = matched_top1_ticks / n
            first_hit_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("matched_trace_hit")),
                None,
            )
            first_top1_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("matched_trace_rank") == 1),
                None,
            )
            if first_hit_tick is not None:
                agg["matched_trace_first_hit_tick"] = first_hit_tick
            if first_top1_tick is not None:
                agg["matched_trace_first_top1_tick"] = first_top1_tick

        if any("partner_trace_hit" in t for t in self.ticks):
            cue_hit_ticks = sum(1 for t in self.ticks if t.get("cue_trace_hit"))
            partner_hit_ticks = sum(1 for t in self.ticks if t.get("partner_trace_hit"))
            partner_top1_ticks = sum(1 for t in self.ticks if t.get("partner_trace_rank") == 1)
            binding_partial_ticks = sum(1 for t in self.ticks if t.get("binding_partial"))
            binding_active_ticks = sum(1 for t in self.ticks if t.get("binding_active"))

            agg["cue_trace_hit_ticks"] = cue_hit_ticks
            agg["cue_trace_hit_rate"] = cue_hit_ticks / n
            agg["partner_trace_hit_ticks"] = partner_hit_ticks
            agg["partner_trace_hit_rate"] = partner_hit_ticks / n
            agg["partner_trace_top1_ticks"] = partner_top1_ticks
            agg["partner_trace_top1_rate"] = partner_top1_ticks / n
            agg["binding_partial_ticks"] = binding_partial_ticks
            agg["binding_partial_rate"] = binding_partial_ticks / n
            agg["binding_active_ticks"] = binding_active_ticks
            agg["binding_active_rate"] = binding_active_ticks / n

            cue_pattern_active_ticks = sum(1 for t in self.ticks if t.get("cue_pattern_active"))
            partner_pattern_active_ticks = sum(
                1 for t in self.ticks if t.get("partner_pattern_active")
            )
            agg["cue_pattern_active_ticks"] = cue_pattern_active_ticks
            agg["cue_pattern_active_rate"] = cue_pattern_active_ticks / n
            agg["partner_pattern_active_ticks"] = partner_pattern_active_ticks
            agg["partner_pattern_active_rate"] = partner_pattern_active_ticks / n

            false_partner_activation_ticks = sum(
                1 for t in self.ticks if t.get("false_partner_activation")
            )
            agg["false_partner_activation_ticks"] = false_partner_activation_ticks
            agg["false_partner_activation_rate"] = false_partner_activation_ticks / n

            first_cue_hit_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("cue_trace_hit")),
                None,
            )
            first_partner_hit_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("partner_trace_hit")),
                None,
            )
            first_partner_top1_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("partner_trace_rank") == 1),
                None,
            )
            first_binding_partial_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("binding_partial")),
                None,
            )
            first_binding_active_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("binding_active")),
                None,
            )
            first_partner_pattern_active_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("partner_pattern_active")),
                None,
            )

            if first_cue_hit_tick is not None:
                agg["cue_trace_first_hit_tick"] = first_cue_hit_tick
            if first_partner_hit_tick is not None:
                agg["partner_trace_first_hit_tick"] = first_partner_hit_tick
            if first_partner_top1_tick is not None:
                agg["partner_trace_first_top1_tick"] = first_partner_top1_tick
            if first_binding_partial_tick is not None:
                agg["binding_partial_first_tick"] = first_binding_partial_tick
            if first_binding_active_tick is not None:
                agg["binding_active_first_tick"] = first_binding_active_tick
            if first_partner_pattern_active_tick is not None:
                agg["partner_pattern_first_active_tick"] = first_partner_pattern_active_tick
            if (
                first_binding_partial_tick is not None
                and first_partner_pattern_active_tick is not None
            ):
                agg["binding_partial_to_partner_pattern_gap"] = (
                    first_partner_pattern_active_tick - first_binding_partial_tick
                )
            if first_binding_partial_tick is not None and first_partner_hit_tick is not None:
                agg["binding_partial_to_partner_gap"] = (
                    first_partner_hit_tick - first_binding_partial_tick
                )

        if any("competitor_trace_hit" in t for t in self.ticks):
            strong_recall_candidate_ticks = sum(
                1 for t in self.ticks if t.get("strong_recall_candidate")
            )
            weak_recall_candidate_ticks = sum(
                1 for t in self.ticks if t.get("weak_recall_candidate")
            )
            competitor_hit_ticks = sum(1 for t in self.ticks if t.get("competitor_trace_hit"))
            competitor_pattern_active_ticks = sum(
                1 for t in self.ticks if t.get("competitor_pattern_active")
            )
            competitor_binding_partial_ticks = sum(
                1 for t in self.ticks if t.get("competitor_binding_partial")
            )
            competitor_binding_active_ticks = sum(
                1 for t in self.ticks if t.get("competitor_binding_active")
            )
            competitor_leak_ticks = sum(1 for t in self.ticks if t.get("competitor_leak"))
            selective_recall_legacy_ticks = sum(
                1 for t in self.ticks if t.get("selective_recall")
            )
            selective_recall_window_eligible_ticks = sum(
                1 for t in self.ticks if t.get("selective_recall_window_eligible")
            )
            selective_recall_ticks = sum(
                1 for t in self.ticks if t.get("selective_recall_scored")
            )
            competitor_outcompetes_partner_ticks = sum(
                1 for t in self.ticks if t.get("competitor_outcompetes_partner")
            )

            agg["strong_recall_candidate_ticks"] = strong_recall_candidate_ticks
            agg["strong_recall_candidate_rate"] = strong_recall_candidate_ticks / n
            agg["weak_recall_candidate_ticks"] = weak_recall_candidate_ticks
            agg["weak_recall_candidate_rate"] = weak_recall_candidate_ticks / n
            agg["competitor_trace_hit_ticks"] = competitor_hit_ticks
            agg["competitor_trace_hit_rate"] = competitor_hit_ticks / n
            agg["competitor_pattern_active_ticks"] = competitor_pattern_active_ticks
            agg["competitor_pattern_active_rate"] = competitor_pattern_active_ticks / n
            agg["competitor_binding_partial_ticks"] = competitor_binding_partial_ticks
            agg["competitor_binding_partial_rate"] = competitor_binding_partial_ticks / n
            agg["competitor_binding_active_ticks"] = competitor_binding_active_ticks
            agg["competitor_binding_active_rate"] = competitor_binding_active_ticks / n
            agg["competitor_leak_ticks"] = competitor_leak_ticks
            agg["competitor_leak_rate"] = competitor_leak_ticks / n
            agg["selective_recall_legacy_ticks"] = selective_recall_legacy_ticks
            agg["selective_recall_legacy_rate"] = selective_recall_legacy_ticks / n
            agg["selective_recall_window_eligible_ticks"] = (
                selective_recall_window_eligible_ticks
            )
            agg["selective_recall_window_eligible_rate"] = (
                selective_recall_window_eligible_ticks / n
            )
            agg["selective_recall_ticks"] = selective_recall_ticks
            agg["selective_recall_rate"] = (
                selective_recall_ticks / selective_recall_window_eligible_ticks
                if selective_recall_window_eligible_ticks
                else 0.0
            )
            agg["competitor_outcompetes_partner_ticks"] = (
                competitor_outcompetes_partner_ticks
            )
            agg["competitor_outcompetes_partner_rate"] = (
                competitor_outcompetes_partner_ticks / n
            )

            first_strong_recall_candidate_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("strong_recall_candidate")),
                None,
            )
            first_weak_recall_candidate_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("weak_recall_candidate")),
                None,
            )
            first_competitor_pattern_active_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("competitor_pattern_active")),
                None,
            )
            first_competitor_binding_partial_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("competitor_binding_partial")),
                None,
            )
            first_competitor_binding_active_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("competitor_binding_active")),
                None,
            )
            first_competitor_trace_hit_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("competitor_trace_hit")),
                None,
            )
            first_selective_recall_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("selective_recall_scored")),
                None,
            )
            first_selective_recall_legacy_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("selective_recall")),
                None,
            )
            first_selective_recall_window_start_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("selective_recall_window_eligible")),
                None,
            )
            if first_strong_recall_candidate_tick is not None:
                agg["strong_recall_candidate_first_tick"] = first_strong_recall_candidate_tick
            if first_weak_recall_candidate_tick is not None:
                agg["weak_recall_candidate_first_tick"] = first_weak_recall_candidate_tick
            if first_competitor_pattern_active_tick is not None:
                agg["competitor_pattern_first_active_tick"] = first_competitor_pattern_active_tick
            if first_competitor_binding_partial_tick is not None:
                agg["competitor_binding_partial_first_tick"] = first_competitor_binding_partial_tick
            if first_competitor_binding_active_tick is not None:
                agg["competitor_binding_active_first_tick"] = first_competitor_binding_active_tick
            if first_competitor_trace_hit_tick is not None:
                agg["competitor_trace_first_hit_tick"] = first_competitor_trace_hit_tick
            if first_selective_recall_tick is not None:
                agg["selective_recall_first_tick"] = first_selective_recall_tick
            if first_selective_recall_legacy_tick is not None:
                agg["selective_recall_legacy_first_tick"] = first_selective_recall_legacy_tick
            if first_selective_recall_window_start_tick is not None:
                agg["selective_recall_window_start_tick"] = (
                    first_selective_recall_window_start_tick
                )
            if (
                first_competitor_binding_partial_tick is not None
                and first_competitor_binding_active_tick is not None
            ):
                agg["competitor_binding_partial_to_active_gap"] = (
                    first_competitor_binding_active_tick - first_competitor_binding_partial_tick
                )

            first_competitor_leak_tick = next(
                (tick_idx for tick_idx, tick in enumerate(self.ticks, start=1)
                 if tick.get("competitor_leak")),
                None,
            )
            if first_competitor_leak_tick is not None:
                leak_tick = self.ticks[first_competitor_leak_tick - 1]
                agg["competitor_leak_origin_tick"] = first_competitor_leak_tick
                leak_origin_region = leak_tick.get("competitor_peak_region")
                if isinstance(leak_origin_region, str) and leak_origin_region:
                    agg["competitor_leak_origin_region"] = leak_origin_region
                leak_origin_ratio = leak_tick.get("competitor_peak_region_ratio")
                if isinstance(leak_origin_ratio, (int, float)):
                    agg["competitor_leak_origin_ratio"] = leak_origin_ratio

            if len(self.ticks) >= 3:
                tick3 = self.ticks[2]
                tick3_peak_region = tick3.get("competitor_peak_region")
                if isinstance(tick3_peak_region, str) and tick3_peak_region:
                    agg["competitor_tick3_peak_region"] = tick3_peak_region
                tick3_peak_ratio = tick3.get("competitor_peak_region_ratio")
                if isinstance(tick3_peak_ratio, (int, float)):
                    agg["competitor_tick3_peak_region_ratio"] = tick3_peak_ratio
                tick3_pattern_ratio = tick3.get("competitor_pattern_ratio")
                if isinstance(tick3_pattern_ratio, (int, float)):
                    agg["competitor_tick3_pattern_ratio"] = tick3_pattern_ratio

        # Motor action distribution
        actions = [t.get("motor_action", "idle") for t in self.ticks]
        agg["motor_actions"] = {a: actions.count(a) for a in set(actions)}

        trace_candidates_avg = agg.get("trace_candidates_avg")
        binding_candidates_avg = agg.get("binding_candidates_avg")
        active_traces_avg = agg.get("active_traces_avg")
        if (
            isinstance(binding_candidates_avg, (int, float))
            and isinstance(trace_candidates_avg, (int, float))
            and trace_candidates_avg > 0
        ):
            agg["binding_candidates_per_trace_candidate_avg"] = (
                binding_candidates_avg / trace_candidates_avg
            )
        if (
            isinstance(binding_candidates_avg, (int, float))
            and isinstance(active_traces_avg, (int, float))
            and active_traces_avg > 0
        ):
            agg["binding_candidates_per_active_trace_avg"] = (
                binding_candidates_avg / active_traces_avg
            )

        # Phase info from last tick
        agg["final_phase"] = self.ticks[-1].get("phase", "unknown")
        agg["tick_count"] = n

        # Speech output from last tick (if present)
        last_speech = self.ticks[-1].get("speech_output", "")
        if last_speech:
            agg["speech_output"] = last_speech
            agg["speech_tokens"] = self.ticks[-1].get("speech_tokens", [])

        return agg

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "modality": self.modality,
            "ticks": self.ticks,
            "summary": self.summary(),
        }


class MetricsCollector:
    """Collects and aggregates metrics across an entire learning session.

    Usage:
        mc = MetricsCollector(dataset="ag_news", mode="sequential")
        mc.start()
        sample = mc.begin_sample(0, label="World", modality="text")
        for tick_result in tick_results:
            sample.add_tick(tick_result)
        mc.end_sample()
        mc.finish()
        mc.save("metrics.json")
    """

    def __init__(
        self,
        dataset: str = "",
        mode: str = "sequential",
        threads: int = 0,
        ticks_per_sample: int = 10,
        extra_config: dict[str, Any] | None = None,
    ):
        self.session_config = {
            "dataset": dataset,
            "mode": mode,
            "threads": threads,
            "ticks_per_sample": ticks_per_sample,
        }
        if extra_config:
            self.session_config.update(extra_config)

        self.samples: list[SampleMetrics] = []
        self._current_sample: SampleMetrics | None = None
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._tick_times: list[float] = []
        self._sample_times: list[float] = []
        self._sample_start_time: float = 0.0

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def finish(self) -> None:
        self._end_time = time.perf_counter()

    def begin_sample(self, index: int, label: str | None = None, modality: str = "") -> SampleMetrics:
        self._current_sample = SampleMetrics(index=index, label=label, modality=modality)
        self._sample_start_time = time.perf_counter()
        return self._current_sample

    def end_sample(self) -> None:
        if self._current_sample is not None:
            self.samples.append(self._current_sample)
            elapsed = time.perf_counter() - self._sample_start_time
            self._sample_times.append(elapsed)
            self._current_sample = None

    def record_tick_time(self, elapsed: float) -> None:
        self._tick_times.append(elapsed)

    @property
    def duration(self) -> float:
        return self._end_time - self._start_time

    def global_summary(self) -> dict[str, Any]:
        """Compute global metrics across all samples."""
        total_ticks = sum(
            max(1, int(tick.get("executed_ticks", 1) or 1))
            for sample in self.samples
            for tick in sample.ticks
        )
        all_ticks = [t for s in self.samples for t in s.ticks]
        all_weights = [
            max(1, int(tick.get("executed_ticks", 1) or 1))
            for tick in all_ticks
        ]
        sample_summaries = [s.summary() for s in self.samples]

        summary: dict[str, Any] = {
            "total_samples": len(self.samples),
            "total_ticks": total_ticks,
            "duration_sec": round(self.duration, 3),
        }

        if total_ticks > 0:
            summary["ticks_per_sec"] = round(total_ticks / max(self.duration, 0.001), 1)

        # Aggregate key metrics across ALL ticks
        for key in ("hebbian_updates", "anti_hebbian_updates",
                     "traces_formed", "trace_candidates", "binding_candidates", "bindings_formed"):
            vals = [t.get(key, 0) for t in all_ticks if isinstance(t.get(key), (int, float))]
            if vals:
                summary[f"{key}_total"] = sum(vals)
                summary[f"{key}_avg"] = round(sum(vals) / len(vals), 3)

        for key in ("total_active", "snapshot_total_active", "active_traces", "working_memory", "total_bindings"):
            weighted_vals = [
                (float(t.get(key, 0)), weight)
                for t, weight in zip(all_ticks, all_weights)
                if isinstance(t.get(key), (int, float))
            ]
            vals = [value for value, _weight in weighted_vals]
            if vals:
                summary[f"{key}_avg"] = round(
                    sum(value * weight for value, weight in weighted_vals)
                    / sum(weight for _value, weight in weighted_vals),
                    3,
                )
                summary[f"{key}_max"] = round(max(vals), 3)

        trace_candidates_avg = summary.get("trace_candidates_avg")
        binding_candidates_avg = summary.get("binding_candidates_avg")
        active_traces_avg = summary.get("active_traces_avg")
        if (
            isinstance(binding_candidates_avg, (int, float))
            and isinstance(trace_candidates_avg, (int, float))
            and trace_candidates_avg > 0
        ):
            summary["binding_candidates_per_trace_candidate_avg"] = round(
                binding_candidates_avg / trace_candidates_avg,
                5,
            )
        if (
            isinstance(binding_candidates_avg, (int, float))
            and isinstance(active_traces_avg, (int, float))
            and active_traces_avg > 0
        ):
            summary["binding_candidates_per_active_trace_avg"] = round(
                binding_candidates_avg / active_traces_avg,
                5,
            )

        for key in ("novelty", "arousal", "valence", "energy",
                     "emotion_polarity", "emotion_arousal",
                     "language_activation", "speech_activity",
                     "sensory_activation", "visual_activation", "audio_activation",
                     "motor_activation", "learning_multiplier",
                     "matched_trace_score", "matched_trace_rank",
                     "cue_trace_score", "cue_trace_rank",
                     "partner_trace_score", "partner_trace_rank",
                     "competitor_trace_score", "competitor_trace_rank",
                     "cue_pattern_ratio", "partner_pattern_ratio",
                     "strong_source_pattern_ratio", "weak_source_pattern_ratio",
                     "competitor_pattern_ratio",
                     "false_trace_score", "false_trace_rank",
                     "binding_weight", "binding_confidence", "binding_fires",
                     "competitor_binding_weight", "competitor_binding_confidence",
                     "competitor_binding_fires",
                     "strong_recall_candidate_relative_weight",
                     "strong_recall_candidate_source_ratio",
                     "weak_recall_candidate_relative_weight",
                     "weak_recall_candidate_source_ratio",
                     "strong_binding_relative_weight",
                     "weak_binding_relative_weight", "binding_weight_margin",
                     "pattern_ratio_margin", "trace_score_margin",
                     "collision_target_overlap_ratio",
                     "rust_tick_ms", "tick_prepare_ms", "tick_delayed_delivery_ms",
                     "tick_propagate_ms", "tick_update_ms", "evaluation_ms",
                     "evaluation_rust_ms", "snapshot_ms", "batch_state_ms",
                     "trace_match_ms", *PROPAGATION_NUMERIC_KEYS,
                     "trace_side_effects_ms", "binding_recall_ms", "binding_recall_python_ms",
                     "binding_recall_bindings", "binding_recall_neurons",
                     "binding_recall_max_relative_weight", "binding_recall_max_boost",
                     "synapse_update_pending_count", "synapse_update_deferred_count",
                     "synapse_update_applied_count", "synapse_update_unmatched_count",
                     "synapse_update_positive_count", "synapse_update_negative_count",
                     "synapse_update_delta_sum", "synapse_update_delta_abs_sum",
                     "synapse_update_delta_min", "synapse_update_delta_max",
                     "synapse_update_release_interval", "synapse_update_release_max_batch",
                     "synapse_update_before_weight_avg", "synapse_update_after_weight_avg",
                     "synapse_update_crossed_up_0p05_count", "synapse_update_crossed_up_0p10_count",
                     "synapse_update_crossed_up_0p20_count", "synapse_update_crossed_down_0p05_count",
                     "synapse_update_crossed_down_0p10_count", "synapse_update_crossed_down_0p20_count",
                     "synapse_update_memory_long_same_count",
                     "synapse_update_memory_long_same_delta_abs_sum",
                     "synapse_update_memory_long_same_before_weight_avg",
                     "synapse_update_memory_long_same_after_weight_avg",
                     "synapse_update_delay_9_count", "synapse_update_delay_10_count",
                     "learn_step_ms", "formation_ms", "maintain_ms", "other_python_ms",
                     "other_python_non_binding_recall_ms",
                     "step_internal_ms", *REGION_ACTIVE_NUMERIC_KEYS,
                     *PROPAGATION_REGION_NUMERIC_KEYS,
                     *POTENTIAL_REGION_NUMERIC_KEYS,
                     *REFRACTORY_REGION_NUMERIC_KEYS,
                     *FIRE_INTERVAL_REGION_NUMERIC_KEYS,
                     *DELAYED_FLOW_NUMERIC_KEYS,
                     *BOOST_NUMERIC_KEYS, *BOOST_REGION_NUMERIC_KEYS):
            weighted_vals = [
                (float(t.get(key, 0)), weight)
                for t, weight in zip(all_ticks, all_weights)
                if isinstance(t.get(key), (int, float))
            ]
            vals = [value for value, _weight in weighted_vals]
            if vals:
                if key in PER_TICK_BATCH_TOTAL_KEYS:
                    avg_value = sum(value for value, _weight in weighted_vals) / total_ticks
                else:
                    avg_value = sum(value * weight for value, weight in weighted_vals) / sum(
                        weight for _value, weight in weighted_vals
                    )
                summary[f"{key}_avg"] = round(avg_value, 5)
                summary[f"{key}_max"] = round(max(vals), 5)

        for region_name in all_region_names():
            interval_sum_total = sum(
                t.get(f"fire_interval_region_{region_name}_sum", 0)
                for t in all_ticks
                if isinstance(t.get(f"fire_interval_region_{region_name}_sum"), (int, float))
            )
            interval_count_total = sum(
                t.get(f"fire_interval_region_{region_name}_count", 0)
                for t in all_ticks
                if isinstance(t.get(f"fire_interval_region_{region_name}_count"), (int, float))
            )
            if interval_count_total > 0:
                summary[f"fire_interval_region_{region_name}_avg_weighted"] = round(
                    interval_sum_total / interval_count_total,
                    5,
                )

        delayed_total = summary.get("scheduled_delayed_signal_count_avg", 0.0)
        if delayed_total > 0:
            recurrent_candidates = []
            for region_name in all_region_names():
                self_flow = summary.get(
                    f"delayed_flow_{region_name}_to_{region_name}_signals_avg", 0.0
                )
                target_total = sum(
                    summary.get(
                        f"delayed_flow_{source_region}_to_{region_name}_signals_avg", 0.0
                    )
                    for source_region in all_region_names()
                )
                global_share = self_flow / delayed_total if delayed_total else 0.0
                target_share = self_flow / target_total if target_total else 0.0
                if (
                    global_share >= RECURRENT_DELAYED_CANDIDATE_MIN_GLOBAL_SHARE
                    and target_share >= RECURRENT_DELAYED_CANDIDATE_MIN_TARGET_SHARE
                ):
                    recurrent_candidates.append(
                        {
                            "region": region_name,
                            "signals_avg": round(self_flow, 5),
                            "share_of_delayed": round(global_share, 5),
                            "share_of_target_delayed": round(target_share, 5),
                        }
                    )
            if recurrent_candidates:
                recurrent_candidates.sort(
                    key=lambda item: (-item["share_of_delayed"], item["region"])
                )
                summary["recurrent_delayed_candidates"] = recurrent_candidates
                summary["recurrent_delayed_candidate_min_global_share"] = (
                    RECURRENT_DELAYED_CANDIDATE_MIN_GLOBAL_SHARE
                )
                summary["recurrent_delayed_candidate_min_target_share"] = (
                    RECURRENT_DELAYED_CANDIDATE_MIN_TARGET_SHARE
                )

        if any("matched_trace_hit" in t for t in all_ticks):
            matched_hit_ticks = sum(1 for t in all_ticks if t.get("matched_trace_hit"))
            matched_top1_ticks = sum(1 for t in all_ticks if t.get("matched_trace_rank") == 1)
            summary["matched_trace_hit_ticks"] = matched_hit_ticks
            summary["matched_trace_hit_rate"] = round(matched_hit_ticks / max(total_ticks, 1), 5)
            summary["matched_trace_top1_ticks"] = matched_top1_ticks
            summary["matched_trace_top1_rate"] = round(matched_top1_ticks / max(total_ticks, 1), 5)

        if any("partner_trace_hit" in t for t in all_ticks):
            cue_hit_ticks = sum(1 for t in all_ticks if t.get("cue_trace_hit"))
            partner_hit_ticks = sum(1 for t in all_ticks if t.get("partner_trace_hit"))
            partner_top1_ticks = sum(1 for t in all_ticks if t.get("partner_trace_rank") == 1)
            binding_partial_ticks = sum(1 for t in all_ticks if t.get("binding_partial"))
            binding_active_ticks = sum(1 for t in all_ticks if t.get("binding_active"))
            summary["cue_trace_hit_ticks"] = cue_hit_ticks
            summary["cue_trace_hit_rate"] = round(cue_hit_ticks / max(total_ticks, 1), 5)
            summary["partner_trace_hit_ticks"] = partner_hit_ticks
            summary["partner_trace_hit_rate"] = round(partner_hit_ticks / max(total_ticks, 1), 5)
            summary["partner_trace_top1_ticks"] = partner_top1_ticks
            summary["partner_trace_top1_rate"] = round(partner_top1_ticks / max(total_ticks, 1), 5)
            summary["binding_partial_ticks"] = binding_partial_ticks
            summary["binding_partial_rate"] = round(binding_partial_ticks / max(total_ticks, 1), 5)
            summary["binding_active_ticks"] = binding_active_ticks
            summary["binding_active_rate"] = round(binding_active_ticks / max(total_ticks, 1), 5)
            cue_pattern_active_ticks = sum(1 for t in all_ticks if t.get("cue_pattern_active"))
            partner_pattern_active_ticks = sum(
                1 for t in all_ticks if t.get("partner_pattern_active")
            )
            summary["cue_pattern_active_ticks"] = cue_pattern_active_ticks
            summary["cue_pattern_active_rate"] = round(
                cue_pattern_active_ticks / max(total_ticks, 1),
                5,
            )
            summary["partner_pattern_active_ticks"] = partner_pattern_active_ticks
            summary["partner_pattern_active_rate"] = round(
                partner_pattern_active_ticks / max(total_ticks, 1),
                5,
            )
            false_partner_activation_ticks = sum(
                1 for t in all_ticks if t.get("false_partner_activation")
            )
            summary["false_partner_activation_ticks"] = false_partner_activation_ticks
            summary["false_partner_activation_rate"] = round(
                false_partner_activation_ticks / max(total_ticks, 1),
                5,
            )

        if any("competitor_trace_hit" in t for t in all_ticks):
            strong_recall_candidate_ticks = sum(
                1 for t in all_ticks if t.get("strong_recall_candidate")
            )
            weak_recall_candidate_ticks = sum(
                1 for t in all_ticks if t.get("weak_recall_candidate")
            )
            summary["strong_recall_candidate_ticks"] = strong_recall_candidate_ticks
            summary["strong_recall_candidate_rate"] = round(
                strong_recall_candidate_ticks / max(total_ticks, 1),
                5,
            )
            summary["weak_recall_candidate_ticks"] = weak_recall_candidate_ticks
            summary["weak_recall_candidate_rate"] = round(
                weak_recall_candidate_ticks / max(total_ticks, 1),
                5,
            )
            competitor_hit_ticks = sum(1 for t in all_ticks if t.get("competitor_trace_hit"))
            competitor_pattern_active_ticks = sum(
                1 for t in all_ticks if t.get("competitor_pattern_active")
            )
            competitor_leak_ticks = sum(1 for t in all_ticks if t.get("competitor_leak"))
            selective_recall_legacy_ticks = sum(
                1 for t in all_ticks if t.get("selective_recall")
            )
            selective_recall_window_eligible_ticks = sum(
                1 for t in all_ticks if t.get("selective_recall_window_eligible")
            )
            selective_recall_ticks = sum(
                1 for t in all_ticks if t.get("selective_recall_scored")
            )
            competitor_outcompetes_partner_ticks = sum(
                1 for t in all_ticks if t.get("competitor_outcompetes_partner")
            )

            summary["competitor_trace_hit_ticks"] = competitor_hit_ticks
            summary["competitor_trace_hit_rate"] = round(
                competitor_hit_ticks / max(total_ticks, 1),
                5,
            )
            summary["competitor_pattern_active_ticks"] = competitor_pattern_active_ticks
            summary["competitor_pattern_active_rate"] = round(
                competitor_pattern_active_ticks / max(total_ticks, 1),
                5,
            )
            competitor_binding_partial_ticks = sum(
                1 for t in all_ticks if t.get("competitor_binding_partial")
            )
            competitor_binding_active_ticks = sum(
                1 for t in all_ticks if t.get("competitor_binding_active")
            )
            summary["competitor_binding_partial_ticks"] = competitor_binding_partial_ticks
            summary["competitor_binding_partial_rate"] = round(
                competitor_binding_partial_ticks / max(total_ticks, 1),
                5,
            )
            summary["competitor_binding_active_ticks"] = competitor_binding_active_ticks
            summary["competitor_binding_active_rate"] = round(
                competitor_binding_active_ticks / max(total_ticks, 1),
                5,
            )
            summary["competitor_leak_ticks"] = competitor_leak_ticks
            summary["competitor_leak_rate"] = round(
                competitor_leak_ticks / max(total_ticks, 1),
                5,
            )
            summary["selective_recall_legacy_ticks"] = selective_recall_legacy_ticks
            summary["selective_recall_legacy_rate"] = round(
                selective_recall_legacy_ticks / max(total_ticks, 1),
                5,
            )
            summary["selective_recall_window_eligible_ticks"] = (
                selective_recall_window_eligible_ticks
            )
            summary["selective_recall_window_eligible_rate"] = round(
                selective_recall_window_eligible_ticks / max(total_ticks, 1),
                5,
            )
            summary["selective_recall_ticks"] = selective_recall_ticks
            summary["selective_recall_rate"] = round(
                selective_recall_ticks / max(selective_recall_window_eligible_ticks, 1),
                5,
            ) if selective_recall_window_eligible_ticks else 0.0
            summary["competitor_outcompetes_partner_ticks"] = (
                competitor_outcompetes_partner_ticks
            )
            summary["competitor_outcompetes_partner_rate"] = round(
                competitor_outcompetes_partner_ticks / max(total_ticks, 1),
                5,
            )

        for key in ("matched_trace_first_hit_tick", "matched_trace_first_top1_tick"):
            vals = [sample.get(key) for sample in sample_summaries if isinstance(sample.get(key), (int, float))]
            if vals:
                summary[f"{key}_avg"] = round(sum(vals) / len(vals), 5)
                summary[f"{key}_max"] = round(max(vals), 5)
                summary[f"{key}_min"] = round(min(vals), 5)

        for key in (
            "cue_trace_first_hit_tick",
            "partner_trace_first_hit_tick",
            "partner_trace_first_top1_tick",
            "binding_partial_first_tick",
            "binding_active_first_tick",
            "partner_pattern_first_active_tick",
            "strong_recall_candidate_first_tick",
            "weak_recall_candidate_first_tick",
            "competitor_binding_partial_first_tick",
            "competitor_binding_active_first_tick",
            "competitor_pattern_first_active_tick",
            "competitor_trace_first_hit_tick",
            "competitor_leak_origin_tick",
            "selective_recall_first_tick",
            "selective_recall_legacy_first_tick",
            "selective_recall_window_start_tick",
            "binding_partial_to_partner_pattern_gap",
            "binding_partial_to_partner_gap",
            "competitor_binding_partial_to_active_gap",
            "competitor_leak_origin_ratio",
            "competitor_tick3_peak_region_ratio",
            "competitor_tick3_pattern_ratio",
        ):
            vals = [sample.get(key) for sample in sample_summaries if isinstance(sample.get(key), (int, float))]
            if vals:
                summary[f"{key}_avg"] = round(sum(vals) / len(vals), 5)
                summary[f"{key}_max"] = round(max(vals), 5)
                summary[f"{key}_min"] = round(min(vals), 5)

        for key in ("competitor_leak_origin_region", "competitor_tick3_peak_region"):
            vals = [sample.get(key) for sample in sample_summaries if isinstance(sample.get(key), str)]
            if vals:
                counts: dict[str, int] = {}
                for value in vals:
                    counts[value] = counts.get(value, 0) + 1
                summary[key] = sorted(
                    counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[0][0]
                summary[f"{key}_counts"] = counts

        if any("matched_trace_hit_ticks" in sample for sample in sample_summaries):
            summary["matched_trace_unrecalled_samples"] = sum(
                1 for sample in sample_summaries if sample.get("matched_trace_hit_ticks", 0) == 0
            )
            summary["matched_trace_non_top1_samples"] = sum(
                1 for sample in sample_summaries if sample.get("matched_trace_top1_ticks", 0) == 0
            )

        if any("partner_trace_hit_ticks" in sample for sample in sample_summaries):
            summary["cue_trace_unseen_samples"] = sum(
                1 for sample in sample_summaries if sample.get("cue_trace_hit_ticks", 0) == 0
            )
            summary["partner_trace_unrecalled_samples"] = sum(
                1 for sample in sample_summaries if sample.get("partner_trace_hit_ticks", 0) == 0
            )
            summary["partner_trace_non_top1_samples"] = sum(
                1 for sample in sample_summaries if sample.get("partner_trace_top1_ticks", 0) == 0
            )
            summary["binding_partial_missing_samples"] = sum(
                1 for sample in sample_summaries if sample.get("binding_partial_ticks", 0) == 0
            )
            summary["binding_active_missing_samples"] = sum(
                1 for sample in sample_summaries if sample.get("binding_active_ticks", 0) == 0
            )
            summary["partner_pattern_inactive_samples"] = sum(
                1 for sample in sample_summaries if sample.get("partner_pattern_active_ticks", 0) == 0
            )

        if any("competitor_trace_hit_ticks" in sample for sample in sample_summaries):
            summary["competitor_trace_silent_samples"] = sum(
                1 for sample in sample_summaries if sample.get("competitor_trace_hit_ticks", 0) == 0
            )
            summary["competitor_pattern_inactive_samples"] = sum(
                1 for sample in sample_summaries if sample.get("competitor_pattern_active_ticks", 0) == 0
            )
            summary["perfect_selective_recall_samples"] = sum(
                1
                for sample in sample_summaries
                if sample.get("selective_recall_window_eligible_ticks", 0) > 0
                and sample.get("selective_recall_ticks", 0)
                == sample.get("selective_recall_window_eligible_ticks", 0)
            )

        # Timing stats
        if self._tick_times:
            summary["tick_time_avg_ms"] = round(1000 * sum(self._tick_times) / len(self._tick_times), 2)
            summary["tick_time_max_ms"] = round(1000 * max(self._tick_times), 2)
            summary["tick_time_min_ms"] = round(1000 * min(self._tick_times), 2)

        if self._sample_times:
            summary["sample_time_avg_ms"] = round(1000 * sum(self._sample_times) / len(self._sample_times), 2)
            summary["sample_time_max_ms"] = round(1000 * max(self._sample_times), 2)

        # State at end
        if all_ticks:
            last = all_ticks[-1]
            summary["final_energy"] = last.get("energy", 0)
            summary["final_arousal"] = last.get("arousal", 0)
            summary["final_valence"] = last.get("valence", 0)
            summary["final_sleep_state"] = last.get("sleep_state", "awake")
            summary["final_sleep_pressure"] = last.get("sleep_pressure", 0)

        return summary

    def to_dict(self) -> dict[str, Any]:
        """Full session data as nested dict."""
        return {
            "session": self.session_config,
            "global_summary": self.global_summary(),
            "samples": [s.to_dict() for s in self.samples],
        }

    def to_dict_compact(self) -> dict[str, Any]:
        """Compact version — sample summaries only, no per-tick data."""
        return {
            "session": self.session_config,
            "global_summary": self.global_summary(),
            "sample_summaries": [
                {"index": s.index, "label": s.label, "modality": s.modality, **s.summary()}
                for s in self.samples
            ],
        }

    def save(self, path: str | Path, compact: bool = False) -> None:
        """Save metrics to JSON file."""
        data = self.to_dict_compact() if compact else self.to_dict()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def print_summary(self) -> None:
        """Print a concise summary to stdout."""
        gs = self.global_summary()
        print(f"\n{'='*60}")
        print(f"SESSION SUMMARY: {self.session_config.get('dataset', '?')} "
              f"({self.session_config.get('mode', '?')})")
        print(f"{'='*60}")
        print(f"  Samples:          {gs.get('total_samples', 0)}")
        print(f"  Total ticks:      {gs.get('total_ticks', 0)}")
        print(f"  Duration:         {gs.get('duration_sec', 0):.1f}s")
        print(f"  Ticks/sec:        {gs.get('ticks_per_sec', 0):.0f}")
        print(f"  Tick avg:         {gs.get('tick_time_avg_ms', 0):.2f}ms")
        print(f"  Rust tick avg:    {gs.get('rust_tick_ms_avg', 0):.2f}ms")
        print(f"  Propagate avg:    {gs.get('tick_propagate_ms_avg', 0):.2f}ms")
        print(f"  Eval avg:         {gs.get('evaluation_ms_avg', 0):.2f}ms")
        print(f"  Trace match avg:  {gs.get('trace_match_ms_avg', 0):.2f}ms")
        print(f"  Other Python avg: {gs.get('other_python_ms_avg', 0):.2f}ms")
        if "binding_recall_python_ms_avg" in gs:
            print(f"  Recall Py avg:    {gs.get('binding_recall_python_ms_avg', 0):.2f}ms")
        if "other_python_non_binding_recall_ms_avg" in gs:
            print(
                f"  Other Py excl Rc: {gs.get('other_python_non_binding_recall_ms_avg', 0):.2f}ms"
            )
        print(f"  Hebbian total:    {gs.get('hebbian_updates_total', 0)}")
        print(f"  Traces formed:    {gs.get('traces_formed_total', 0)}")
        if "trace_candidates_avg" in gs:
            print(f"  Trace cands:      {gs.get('trace_candidates_avg', 0):.1f} avg")
        if "binding_candidates_avg" in gs:
            print(f"  Binding cands:    {gs.get('binding_candidates_avg', 0):.1f} avg")
        if "binding_candidates_per_trace_candidate_avg" in gs:
            print(
                f"  Bind/trace cand:  {gs.get('binding_candidates_per_trace_candidate_avg', 0):.2f}"
            )
        print(f"  Bindings formed:  {gs.get('bindings_formed_total', 0)}")
        if "active_traces_avg" in gs:
            print(f"  Active traces:    {gs.get('active_traces_avg', 0):.1f} avg")
        if "total_bindings_avg" in gs:
            print(f"  Total bindings:   {gs.get('total_bindings_avg', 0):.1f} avg")
        if "incoming_signal_count_avg" in gs:
            print(f"  Incoming sigs:    {gs.get('incoming_signal_count_avg', 0):.1f} avg")
        if "scheduled_delayed_signal_count_avg" in gs:
            print(f"  Delayed queued:   {gs.get('scheduled_delayed_signal_count_avg', 0):.1f} avg")
        if "total_fired_avg" in gs:
            print(f"  Fired neurons:    {gs.get('total_fired_avg', 0):.1f} avg")
        if "refractory_ignored_abs_sum_avg" in gs:
            print(f"  Refr ignored:     {gs.get('refractory_ignored_abs_sum_avg', 0):.1f} abs avg")
        if "fire_interval_region_memory_long_avg_weighted" in gs:
            print(
                f"  ML fire intvl:    {gs.get('fire_interval_region_memory_long_avg_weighted', 0):.2f} ticks"
            )
        recurrent_candidates = gs.get("recurrent_delayed_candidates")
        if isinstance(recurrent_candidates, list) and recurrent_candidates:
            labels = ", ".join(
                f"{item['region']} ({100 * item['share_of_delayed']:.1f}% delayed)"
                for item in recurrent_candidates[:3]
            )
            print(f"  Recurrent cand:   {labels}")
        if "working_memory_boost_neurons_avg" in gs:
            print(f"  WM boost:         {gs.get('working_memory_boost_neurons_avg', 0):.1f} neurons avg")
        if "pattern_completion_neurons_avg" in gs:
            print(f"  Pattern complete: {gs.get('pattern_completion_neurons_avg', 0):.1f} neurons avg")
        if "speech_boost_neurons_avg" in gs:
            print(f"  Speech boost:     {gs.get('speech_boost_neurons_avg', 0):.1f} neurons avg")
        if "binding_recall_neurons_avg" in gs:
            print(f"  Recall boost:     {gs.get('binding_recall_neurons_avg', 0):.1f} neurons avg")
        if "matched_trace_hit_rate" in gs:
            print(f"  Match hit rate:   {100 * gs.get('matched_trace_hit_rate', 0):.1f}%")
        if "cue_trace_hit_rate" in gs:
            print(f"  Cue hit rate:     {100 * gs.get('cue_trace_hit_rate', 0):.1f}%")
        if "binding_partial_rate" in gs:
            print(f"  Bind partial rt:  {100 * gs.get('binding_partial_rate', 0):.1f}%")
        if "partner_pattern_active_rate" in gs:
            print(f"  Partner patt rt:  {100 * gs.get('partner_pattern_active_rate', 0):.1f}%")
        if "partner_trace_hit_rate" in gs:
            print(f"  Partner hit rate: {100 * gs.get('partner_trace_hit_rate', 0):.1f}%")
        if "competitor_pattern_active_rate" in gs:
            print(f"  Comp patt rt:     {100 * gs.get('competitor_pattern_active_rate', 0):.1f}%")
        if "competitor_trace_hit_rate" in gs:
            print(f"  Comp hit rate:    {100 * gs.get('competitor_trace_hit_rate', 0):.1f}%")
        if "selective_recall_rate" in gs:
            print(f"  Selective rt:     {100 * gs.get('selective_recall_rate', 0):.1f}%")
        if "selective_recall_legacy_rate" in gs:
            print(f"  Selective raw:    {100 * gs.get('selective_recall_legacy_rate', 0):.1f}%")
        if "selective_recall_window_eligible_ticks" in gs:
            print(f"  Scored ticks:     {gs.get('selective_recall_window_eligible_ticks', 0)}")
        if "competitor_leak_rate" in gs:
            print(f"  Comp leak rate:   {100 * gs.get('competitor_leak_rate', 0):.1f}%")
        if "weak_recall_candidate_rate" in gs:
            print(f"  Weak recall rt:   {100 * gs.get('weak_recall_candidate_rate', 0):.1f}%")
        if "competitor_leak_origin_region" in gs:
            print(f"  Leak origin:      {gs.get('competitor_leak_origin_region')}"
                  f" ({gs.get('competitor_leak_origin_ratio_avg', 0):.2f})")
        if "competitor_tick3_peak_region" in gs:
            print(f"  Tick3 peak reg:   {gs.get('competitor_tick3_peak_region')}"
                  f" ({gs.get('competitor_tick3_peak_region_ratio_avg', 0):.2f})")
        if "false_partner_activation_rate" in gs:
            print(f"  False act rate:   {100 * gs.get('false_partner_activation_rate', 0):.1f}%")
        if "matched_trace_first_hit_tick_avg" in gs:
            print(f"  Recall latency:   {gs.get('matched_trace_first_hit_tick_avg', 0):.2f} ticks avg")
        if "matched_trace_first_top1_tick_avg" in gs:
            print(f"  Top-1 latency:    {gs.get('matched_trace_first_top1_tick_avg', 0):.2f} ticks avg")
        if "binding_partial_first_tick_avg" in gs:
            print(f"  Bind partial:     {gs.get('binding_partial_first_tick_avg', 0):.2f} ticks avg")
        if "partner_pattern_first_active_tick_avg" in gs:
            print(f"  Pattern latency:  {gs.get('partner_pattern_first_active_tick_avg', 0):.2f} ticks avg")
        if "partner_trace_first_hit_tick_avg" in gs:
            print(f"  Partner latency:  {gs.get('partner_trace_first_hit_tick_avg', 0):.2f} ticks avg")
        if "selective_recall_window_start_tick_avg" in gs:
            print(f"  Window start:     {gs.get('selective_recall_window_start_tick_avg', 0):.2f} ticks avg")
        if "selective_recall_first_tick_avg" in gs:
            print(f"  Selective tick:   {gs.get('selective_recall_first_tick_avg', 0):.2f} ticks avg")
        if "binding_partial_to_partner_pattern_gap_avg" in gs:
            print(f"  Pattern gap:      {gs.get('binding_partial_to_partner_pattern_gap_avg', 0):.2f} ticks avg")
        if "binding_partial_to_partner_gap_avg" in gs:
            print(f"  Recall gap:       {gs.get('binding_partial_to_partner_gap_avg', 0):.2f} ticks avg")
        print(f"  Activity avg:     {gs.get('total_active_avg', 0):.1f}")
        print(f"  Novelty avg:      {gs.get('novelty_avg', 0):.4f}")
        print(f"  Final energy:     {gs.get('final_energy', 0):.3f}")
        print(f"  Final arousal:    {gs.get('final_arousal', 0):.3f}")

        # Speech accuracy (if speech_output is present in sample summaries)
        speech_correct = 0
        speech_total = 0
        speech_nonempty = 0
        for sm in self.samples:
            s = sm.summary()
            label = sm.label or ""
            speech_out = s.get("speech_output", "")
            if not label:
                continue
            speech_total += 1
            if speech_out:
                speech_nonempty += 1
                top1 = speech_out.split()[0]
                if top1 == label:
                    speech_correct += 1
        if speech_total > 0 and speech_nonempty > 0:
            print(f"  Speech output:    {speech_nonempty}/{speech_total} samples produced text")
            print(f"  Speech accuracy:  {speech_correct}/{speech_total} = {100*speech_correct/speech_total:.1f}%")

        print(f"{'='*60}\n")
