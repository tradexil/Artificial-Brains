"""Real-content binding probe on fixed learned traces harvested from text."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import random
from pathlib import Path

import brain_core

from brain.benchmarks.text_learning_probe import _add_anchor_trace, _coverage_row, _prepare_probe_samples
from brain.input.text_input import TextInput, is_content_text_token
from brain.learning.tick_loop import TickLoop
from brain.seed.text_vocab_overlay import apply_text_vocab_overlay
from brain.seed.seed_runner import seed_brain_fast
from brain.structures.neuron_map import all_region_names
from brain.structures.trace_store import Trace
from brain.utils.config import (
    BINDING_RECALL_TRACE_MATCH_THRESHOLD,
    TOTAL_NEURONS,
    TRACE_ACTIVATION_THRESHOLD,
    WORKING_MEMORY_CAPACITY,
)


_BINDING_PROBE_SEED = 12345

_CANDIDATE_SOURCE_REGION_PRIORITY = {
    "pattern": 3,
    "memory_long": 2,
    "integration": 1,
    "language": 0,
}

_CANDIDATE_TARGET_REGION_PRIORITY = {
    "memory_long": 4,
    "integration": 3,
    "pattern": 2,
    "language": 1,
}


def _derive_overlay_working_memory_cap(capacity: int) -> int:
    if capacity <= 0:
        return 0
    return max(1, capacity // 2)


def _dominant_region(trace: Trace) -> str | None:
    candidates = [
        (region_name, len(neuron_ids))
        for region_name, neuron_ids in trace.neurons.items()
        if neuron_ids
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[1], item[0]))[0]


def _trace_match_metrics(active_traces: list[tuple[str, float]], trace_id: str) -> tuple[bool, int | None, float]:
    for rank, (active_trace_id, score) in enumerate(active_traces, start=1):
        if active_trace_id == trace_id:
            return True, rank, score
    return False, None, 0.0


def _top_non_excluded_trace(
    active_traces: list[tuple[str, float]],
    excluded_ids: set[str],
) -> tuple[str | None, int | None, float]:
    for rank, (trace_id, score) in enumerate(active_traces, start=1):
        if trace_id not in excluded_ids:
            return trace_id, rank, score
    return None, None, 0.0


def _candidate_probe_seed(sample_key: str, cue_trace_id: str) -> int:
    seed_material = f"{_BINDING_PROBE_SEED}:{sample_key}:{cue_trace_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")


def _overlay_trace_ordinal(trace_id: str) -> int:
    if not trace_id.startswith("overlay_"):
        return -1

    suffix = trace_id.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else -1


def _binding_selection_features(binding: dict[str, object]) -> tuple[float, int, int, int, int, int, int]:
    return (
        round(float(binding.get("weight", 0.0)), 12),
        int(binding.get("fires", 0)),
        int(binding.get("selection_partner_trace_distinct_regions", 0)),
        int(binding.get("selection_partner_trace_non_memory_long_regions", 0)),
        int(binding.get("selection_source_region_priority", 0)),
        int(binding.get("selection_target_region_priority", 0)),
        -int(binding.get("selection_partner_trace_overlay_ordinal", -1)),
    )


def _binding_row_sort_key(binding: dict[str, object]) -> tuple[bool, float, int, int, int, int, int, int, int]:
    return (
        binding.get("scope") == "same",
        *_binding_selection_features(binding),
        -int(binding.get("binding_id", 0)),
    )


def _annotate_binding_row_selection_fields(trace_store, binding_row: dict[str, object]) -> None:
    partner_trace = trace_store.get(str(binding_row["trace_id_b"]))
    if partner_trace is None:
        binding_row.update(
            {
                "selection_partner_trace_distinct_regions": 0,
                "selection_partner_trace_non_memory_long_regions": 0,
                "selection_partner_trace_total_neurons": 0,
                "selection_partner_trace_overlay_ordinal": -1,
                "selection_source_region_priority": 0,
                "selection_target_region_priority": 0,
            }
        )
        return

    partner_region_counts = {
        region_name: len(neuron_ids)
        for region_name, neuron_ids in partner_trace.neurons.items()
    }
    binding_row.update(
        {
            "selection_partner_trace_distinct_regions": sum(
                1 for count in partner_region_counts.values() if count > 0
            ),
            "selection_partner_trace_non_memory_long_regions": sum(
                1
                for region_name, count in partner_region_counts.items()
                if count > 0 and region_name != "memory_long"
            ),
            "selection_partner_trace_total_neurons": partner_trace.total_neurons(),
            "selection_partner_trace_overlay_ordinal": _overlay_trace_ordinal(partner_trace.id),
            "selection_source_region_priority": _CANDIDATE_SOURCE_REGION_PRIORITY.get(
                str(binding_row["region_a"]),
                0,
            ),
            "selection_target_region_priority": _CANDIDATE_TARGET_REGION_PRIORITY.get(
                str(binding_row["region_b"]),
                0,
            ),
        }
    )


def _ratio_distribution(values: list[float]) -> list[dict[str, object]]:
    rounded_values = [round(float(value), 4) for value in values]
    if not rounded_values:
        return []

    counts = Counter(rounded_values)
    total = len(rounded_values)
    return [
        {
            "ratio": ratio,
            "count": count,
            "fraction": round(count / total, 4),
        }
        for ratio, count in sorted(counts.items(), key=lambda item: (-item[0], -item[1]))
    ]


def _candidate_recall_sort_key(candidate: dict[str, object]) -> tuple[bool, float, float, float, float, int]:
    binding = candidate.get("binding", {})
    binding_id = int(binding.get("binding_id", 0))
    return (
        candidate.get("reason") is None,
        float(candidate.get("partner_trace_active_ratio_max", 0.0)),
        float(candidate.get("partner_trace_hit_rate", 0.0)),
        float(candidate.get("selective_recall_rate", 0.0)),
        float(candidate.get("partner_pattern_active_rate", 0.0)),
        -binding_id,
    )


def _candidate_selection_rank_key(candidate: dict[str, object]) -> tuple[float, int]:
    binding = candidate.get("binding", {})
    return _binding_selection_features(binding)


def _candidate_ratio_max(candidate: dict[str, object]) -> float:
    if candidate.get("reason") is not None:
        return 0.0
    return round(float(candidate.get("partner_trace_active_ratio_max", 0.0)), 4)


def _group_candidates_by_selection_rank(
    candidate_results: list[dict[str, object]],
) -> list[list[dict[str, object]]]:
    groups: list[list[dict[str, object]]] = []
    current_group: list[dict[str, object]] = []
    current_key: tuple[float, int] | None = None

    for candidate in candidate_results:
        group_key = _candidate_selection_rank_key(candidate)
        if current_key is None or group_key == current_key:
            current_group.append(candidate)
            current_key = group_key
            continue

        groups.append(current_group)
        current_group = [candidate]
        current_key = group_key

    if current_group:
        groups.append(current_group)
    return groups


def _top_k_best_ratio_distribution(
    candidate_results: list[dict[str, object]],
    k: int,
) -> dict[str, object]:
    if k <= 0 or not candidate_results:
        return {
            "distribution": [],
            "expected_best_ratio": 0.0,
            "requires_random_tie_break": False,
            "tie_group_size": 0,
            "selected_from_tie_group": 0,
        }

    effective_k = min(k, len(candidate_results))
    prefix: list[dict[str, object]] = []
    cutoff_group: list[dict[str, object]] = []
    selected_from_cutoff = 0
    for group in _group_candidates_by_selection_rank(candidate_results):
        if len(prefix) + len(group) < effective_k:
            prefix.extend(group)
            continue
        if len(prefix) + len(group) == effective_k:
            prefix.extend(group)
            break

        cutoff_group = group
        selected_from_cutoff = effective_k - len(prefix)
        break

    base_best_ratio = max((_candidate_ratio_max(candidate) for candidate in prefix), default=0.0)
    if not cutoff_group or selected_from_cutoff <= 0:
        distribution = _ratio_distribution([base_best_ratio])
        return {
            "distribution": distribution,
            "expected_best_ratio": base_best_ratio,
            "requires_random_tie_break": False,
            "tie_group_size": 0,
            "selected_from_tie_group": 0,
        }

    cutoff_ratios = [_candidate_ratio_max(candidate) for candidate in cutoff_group]
    total_combinations = math.comb(len(cutoff_group), selected_from_cutoff)
    if total_combinations <= 0:
        distribution = _ratio_distribution([base_best_ratio])
        return {
            "distribution": distribution,
            "expected_best_ratio": base_best_ratio,
            "requires_random_tie_break": False,
            "tie_group_size": len(cutoff_group),
            "selected_from_tie_group": selected_from_cutoff,
        }

    cutoff_counts = Counter(cutoff_ratios)
    higher_count = 0
    max_ratio_probabilities: dict[float, float] = {}
    for ratio, count in sorted(cutoff_counts.items(), key=lambda item: item[0], reverse=True):
        no_higher = math.comb(len(cutoff_group) - higher_count, selected_from_cutoff)
        no_higher_or_ratio = math.comb(
            len(cutoff_group) - higher_count - count,
            selected_from_cutoff,
        )
        probability = (no_higher - no_higher_or_ratio) / total_combinations
        final_ratio = max(base_best_ratio, ratio)
        max_ratio_probabilities[final_ratio] = max_ratio_probabilities.get(final_ratio, 0.0) + probability
        higher_count += count

    if not max_ratio_probabilities:
        max_ratio_probabilities[base_best_ratio] = 1.0

    distribution = [
        {
            "ratio": ratio,
            "count": count,
            "fraction": round(fraction, 4),
        }
        for ratio, fraction in sorted(
            max_ratio_probabilities.items(),
            key=lambda item: (-item[0], -item[1]),
        )
        if fraction > 0.0
        for count in [round(fraction * total_combinations)]
    ]
    expected_best_ratio = round(
        sum(ratio * fraction for ratio, fraction in max_ratio_probabilities.items()),
        4,
    )
    return {
        "distribution": distribution,
        "expected_best_ratio": expected_best_ratio,
        "requires_random_tie_break": True,
        "tie_group_size": len(cutoff_group),
        "selected_from_tie_group": selected_from_cutoff,
    }


def _summarize_top_k_reliability(
    candidate_results: list[dict[str, object]],
    top_ks: tuple[int, ...] = (1, 3),
    success_threshold: float = BINDING_RECALL_TRACE_MATCH_THRESHOLD,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "ranking_mode": "binding_weight_fires_partner_diversity_source_target_overlay_ordinal_then_binding_id",
        "tie_aware_grouping": "binding_weight_fires_partner_diversity_source_target_overlay_ordinal",
        "success_threshold": round(float(success_threshold), 4),
        "selection_order_binding_ids": [
            int(candidate.get("binding", {}).get("binding_id", 0))
            for candidate in candidate_results
        ],
        "top_k": {},
    }

    for k in sorted(set(value for value in top_ks if value > 0)):
        prefix = candidate_results[: min(k, len(candidate_results))]
        deterministic_best_ratio = round(
            max((_candidate_ratio_max(candidate) for candidate in prefix), default=0.0),
            4,
        )
        deterministic_evaluated = [
            candidate
            for candidate in prefix
            if candidate.get("reason") is None
        ]
        tie_aware = _top_k_best_ratio_distribution(candidate_results, k)
        tie_aware_distribution = tie_aware["distribution"]
        tie_aware_success_probability = round(
            sum(
                float(row["fraction"])
                for row in tie_aware_distribution
                if float(row["ratio"]) >= success_threshold
            ),
            4,
        )
        summary["top_k"][f"top_{k}"] = {
            "candidate_count": len(prefix),
            "binding_ids": [
                int(candidate.get("binding", {}).get("binding_id", 0))
                for candidate in prefix
            ],
            "evaluated_candidate_count": len(deterministic_evaluated),
            "skipped_candidate_count": len(prefix) - len(deterministic_evaluated),
            "best_ratio": deterministic_best_ratio,
            "meets_success_threshold": deterministic_best_ratio >= success_threshold,
            "tie_aware_best_ratio_distribution": tie_aware_distribution,
            "tie_aware_expected_best_ratio": tie_aware["expected_best_ratio"],
            "tie_aware_meets_success_threshold_probability": tie_aware_success_probability,
            "tie_aware_requires_random_tie_break": tie_aware["requires_random_tie_break"],
            "tie_group_size": tie_aware["tie_group_size"],
            "selected_from_tie_group": tie_aware["selected_from_tie_group"],
        }

    return summary


def _summarize_candidate_recall_results(candidate_results: list[dict[str, object]]) -> dict[str, object]:
    ranked_results = sorted(
        candidate_results,
        key=_candidate_recall_sort_key,
        reverse=True,
    )
    evaluated_results = [
        candidate
        for candidate in ranked_results
        if candidate.get("reason") is None
    ]
    ratio_max_values = [
        float(candidate["partner_trace_active_ratio_max"])
        for candidate in evaluated_results
    ]
    ratio_avg_values = [
        float(candidate["partner_trace_active_ratio_avg"])
        for candidate in evaluated_results
    ]
    top_k_reliability = _summarize_top_k_reliability(candidate_results)
    return {
        "candidate_results": ranked_results,
        "evaluated_candidate_count": len(evaluated_results),
        "skipped_candidate_count": len(ranked_results) - len(evaluated_results),
        "evaluated_candidate_binding_ids": [
            int(candidate["binding"]["binding_id"])
            for candidate in evaluated_results
        ],
        "partner_trace_active_ratio_max_distribution": _ratio_distribution(ratio_max_values),
        "partner_trace_active_ratio_avg_distribution": _ratio_distribution(ratio_avg_values),
        "best_candidate": evaluated_results[0] if evaluated_results else None,
        "selection_reliability": top_k_reliability,
    }


def _build_trace_cue(
    trace: Trace,
    cue_fraction: float,
    cue_noise_fraction: float,
    rng: random.Random,
    cue_mode: str = "dominant-region",
) -> tuple[list[tuple[int, float]], int, int]:
    cue_fraction = min(1.0, max(0.0, cue_fraction))
    cue_noise_fraction = max(0.0, cue_noise_fraction)
    cue_mode = cue_mode.lower()

    trace_neurons = sorted(
        {
            neuron_id
            for neuron_ids in trace.neurons.values()
            for neuron_id in neuron_ids
        }
    )
    if not trace_neurons:
        return [], 0, 0

    if cue_mode == "dominant-region":
        dominant_region = _dominant_region(trace)
        source = sorted(trace.neurons.get(dominant_region or "", [])) or trace_neurons
        keep_count = len(source)
        if cue_fraction < 1.0:
            keep_count = max(1, min(len(source), math.ceil(len(source) * cue_fraction)))
        signal_neurons = rng.sample(source, keep_count)
    elif cue_mode == "per-region":
        signal_neurons = []
        for neuron_ids in trace.neurons.values():
            if not neuron_ids:
                continue
            keep_count = len(neuron_ids)
            if cue_fraction < 1.0:
                keep_count = max(1, min(len(neuron_ids), math.ceil(len(neuron_ids) * cue_fraction)))
            signal_neurons.extend(rng.sample(list(neuron_ids), keep_count))
        signal_neurons = sorted(set(signal_neurons))
    else:
        keep_count = len(trace_neurons)
        if cue_fraction < 1.0:
            keep_count = max(1, min(len(trace_neurons), math.ceil(len(trace_neurons) * cue_fraction)))
        signal_neurons = rng.sample(trace_neurons, keep_count)

    noise_count = int(round(len(signal_neurons) * cue_noise_fraction))
    trace_neuron_set = set(trace_neurons)
    noise_neurons: list[int] = []
    seen_noise: set[int] = set()
    while len(noise_neurons) < noise_count:
        neuron_id = rng.randrange(TOTAL_NEURONS)
        if neuron_id in trace_neuron_set or neuron_id in seen_noise:
            continue
        noise_neurons.append(neuron_id)
        seen_noise.add(neuron_id)

    injected = [(neuron_id, 1.0) for neuron_id in signal_neurons]
    injected.extend((neuron_id, 1.0) for neuron_id in noise_neurons)
    return injected, len(signal_neurons), len(noise_neurons)


def _clone_sample_trace(
    trace: Trace,
    sample_key: str,
    ordinal: int,
    *,
    kind: str = "learned",
) -> Trace:
    clone_id = f"{kind}_{sample_key}_{ordinal:02d}"
    clone_label = trace.label or f"{sample_key}_{ordinal:02d}"
    return Trace(
        id=clone_id,
        label=clone_label,
        neurons={
            region_name: list(neuron_ids)
            for region_name, neuron_ids in sorted(trace.neurons.items())
        },
        binding_ids=[],
        strength=trace.strength,
        decay=trace.decay,
        polarity=trace.polarity,
        abstraction=trace.abstraction,
        novelty=trace.novelty,
        co_traces=sorted(set(trace.co_traces) | {sample_key}),
        context_tags=[f"sample:{sample_key}", "harvested", f"source:{kind}"],
        fire_count=0,
        last_fired=0,
        formation_tick=0,
    )


def _trace_overlap(trace_a: Trace, trace_b: Trace) -> dict[str, object]:
    by_region: dict[str, int] = {}
    for region_name in sorted(set(trace_a.neurons) | set(trace_b.neurons)):
        overlap = len(set(trace_a.neurons.get(region_name, [])) & set(trace_b.neurons.get(region_name, [])))
        if overlap > 0:
            by_region[region_name] = overlap
    return {
        "trace_id_a": trace_a.id,
        "trace_id_b": trace_b.id,
        "total_overlap": sum(by_region.values()),
        "by_region": by_region,
    }


def _sample_token_source_stats(
    encoder: TextInput,
    text: str,
    overlay_trace_ids: set[str],
) -> dict[str, object]:
    tokens = encoder.tokenize(text)
    content_token_total = sum(1 for token in tokens if is_content_text_token(token))
    overlay_content_occurrences = 0
    known_non_overlay_occurrences = 0
    unknown_content_occurrences = 0
    overlay_content_tokens: set[str] = set()
    known_non_overlay_tokens: set[str] = set()
    unknown_content_tokens: set[str] = set()
    rows: list[dict[str, object]] = []

    idx = 0
    while idx < len(tokens):
        span_len, trace_ids = encoder._find_span_match(tokens, idx)
        if span_len > 0:
            span_tokens = tokens[idx:idx + span_len]
            content_tokens = [token for token in span_tokens if is_content_text_token(token)]
            matched_overlay_trace_ids = sorted(
                trace_id for trace_id in trace_ids if trace_id in overlay_trace_ids
            )
            source = "overlay" if matched_overlay_trace_ids else "known"
            if content_tokens:
                if matched_overlay_trace_ids:
                    overlay_content_occurrences += len(content_tokens)
                    overlay_content_tokens.update(content_tokens)
                else:
                    known_non_overlay_occurrences += len(content_tokens)
                    known_non_overlay_tokens.update(content_tokens)
            rows.append(
                {
                    "span": " ".join(span_tokens),
                    "tokens": span_tokens,
                    "source": source,
                    "matched_trace_ids": trace_ids,
                    "matched_overlay_trace_ids": matched_overlay_trace_ids,
                }
            )
            idx += span_len
            continue

        token = tokens[idx]
        if is_content_text_token(token):
            unknown_content_occurrences += 1
            unknown_content_tokens.add(token)
        rows.append(
            {
                "span": token,
                "tokens": [token],
                "source": "unknown",
                "matched_trace_ids": [],
                "matched_overlay_trace_ids": [],
            }
        )
        idx += 1

    return {
        "content_token_total": content_token_total,
        "overlay_content_token_occurrences": overlay_content_occurrences,
        "overlay_content_token_share": round(
            overlay_content_occurrences / max(1, content_token_total),
            4,
        ),
        "overlay_content_tokens": sorted(overlay_content_tokens),
        "known_non_overlay_content_occurrences": known_non_overlay_occurrences,
        "known_non_overlay_content_tokens": sorted(known_non_overlay_tokens),
        "unknown_content_occurrences": unknown_content_occurrences,
        "unknown_content_tokens": sorted(unknown_content_tokens),
        "rows": rows,
    }


def _empty_formation_diagnostics() -> dict[str, object]:
    return {
        "training_ticks": 0,
        "formation_attempt_ticks": 0,
        "not_attempted_ticks": 0,
        "working_memory_full_ticks": 0,
        "no_candidate_snapshot_ticks": 0,
        "formation_threshold_block_ticks": 0,
        "novelty_threshold_block_ticks": 0,
        "min_region_block_ticks": 0,
        "tracker_pending_ticks": 0,
        "ready_pattern_ticks": 0,
        "formed_ticks": 0,
        "ready_pattern_count_total": 0,
        "traces_formed_total": 0,
        "novelty_total": 0.0,
        "trace_candidates_total": 0.0,
        "active_traces_total": 0.0,
        "candidate_snapshot_total_neurons_total": 0.0,
        "candidate_snapshot_region_count_total": 0.0,
        "active_region_sum": defaultdict(float),
        "active_region_max": defaultdict(int),
        "candidate_region_sum": defaultdict(float),
        "candidate_region_max": defaultdict(int),
        "tick_rows": [],
    }


def _summarize_region_totals(
    region_sum: dict[str, float],
    region_max: dict[str, int],
    tick_count: int,
) -> dict[str, object]:
    if tick_count <= 0:
        return {
            "avg_counts": {},
            "max_counts": {},
            "top_regions": [],
        }

    avg_counts = {
        region_name: round(total / tick_count, 3)
        for region_name, total in region_sum.items()
        if total > 0
    }
    max_counts = {
        region_name: int(count)
        for region_name, count in region_max.items()
        if count > 0
    }
    top_regions = [
        {
            "region": region_name,
            "avg": avg_counts[region_name],
            "max": max_counts.get(region_name, 0),
        }
        for region_name in sorted(
            avg_counts,
            key=lambda name: (avg_counts[name], name),
            reverse=True,
        )[:5]
    ]
    return {
        "avg_counts": avg_counts,
        "max_counts": max_counts,
        "top_regions": top_regions,
    }


def _finalize_formation_diagnostics(raw: dict[str, object]) -> dict[str, object]:
    training_ticks = int(raw["training_ticks"])
    formation_attempt_ticks = int(raw["formation_attempt_ticks"])
    not_attempted_ticks = int(raw["not_attempted_ticks"])
    return {
        "training_ticks": training_ticks,
        "formation_attempt_ticks": formation_attempt_ticks,
        "formation_attempt_rate": round(
            formation_attempt_ticks / max(1, training_ticks),
            4,
        ),
        "not_attempted_ticks": not_attempted_ticks,
        "not_attempted_rate": round(not_attempted_ticks / max(1, training_ticks), 4),
        "working_memory_full_ticks": int(raw["working_memory_full_ticks"]),
        "no_candidate_snapshot_ticks": int(raw["no_candidate_snapshot_ticks"]),
        "formation_threshold_block_ticks": int(raw["formation_threshold_block_ticks"]),
        "novelty_threshold_block_ticks": int(raw["novelty_threshold_block_ticks"]),
        "min_region_block_ticks": int(raw["min_region_block_ticks"]),
        "tracker_pending_ticks": int(raw["tracker_pending_ticks"]),
        "ready_pattern_ticks": int(raw["ready_pattern_ticks"]),
        "formed_ticks": int(raw["formed_ticks"]),
        "ready_pattern_count_total": int(raw["ready_pattern_count_total"]),
        "traces_formed_total": int(raw["traces_formed_total"]),
        "novelty_avg": round(float(raw["novelty_total"]) / max(1, training_ticks), 4),
        "trace_candidates_avg": round(float(raw["trace_candidates_total"]) / max(1, training_ticks), 4),
        "active_traces_avg": round(float(raw["active_traces_total"]) / max(1, training_ticks), 4),
        "candidate_snapshot_total_neurons_avg": round(
            float(raw["candidate_snapshot_total_neurons_total"]) / max(1, training_ticks),
            4,
        ),
        "candidate_snapshot_region_count_avg": round(
            float(raw["candidate_snapshot_region_count_total"]) / max(1, training_ticks),
            4,
        ),
        "active_region_summary": _summarize_region_totals(
            raw["active_region_sum"],
            raw["active_region_max"],
            training_ticks,
        ),
        "candidate_region_summary": _summarize_region_totals(
            raw["candidate_region_sum"],
            raw["candidate_region_max"],
            training_ticks,
        ),
        "tick_rows": raw["tick_rows"],
    }


def _update_formation_diagnostics(
    diagnostics: dict[str, object],
    result: dict[str, object],
    formation_debug: dict[str, object],
    repeat_index: int,
    tick_offset: int,
) -> None:
    diagnostics["training_ticks"] += 1
    diagnostics["novelty_total"] += float(formation_debug.get("novelty", result.get("novelty", 0.0)))
    diagnostics["trace_candidates_total"] += float(result.get("trace_candidates", 0.0))
    diagnostics["active_traces_total"] += float(result.get("active_traces", 0.0))
    diagnostics["candidate_snapshot_total_neurons_total"] += float(
        formation_debug.get("candidate_snapshot_total_neurons", 0.0)
    )
    diagnostics["candidate_snapshot_region_count_total"] += float(
        formation_debug.get("candidate_snapshot_region_count", 0.0)
    )
    diagnostics["ready_pattern_count_total"] += int(formation_debug.get("ready_pattern_count", 0))
    diagnostics["traces_formed_total"] += int(result.get("traces_formed", 0))

    if formation_debug.get("attempted"):
        diagnostics["formation_attempt_ticks"] += 1
    else:
        diagnostics["not_attempted_ticks"] += 1

    failure_stage = str(formation_debug.get("failure_stage", "none"))
    if failure_stage == "working_memory_full":
        diagnostics["working_memory_full_ticks"] += 1
    elif failure_stage == "no_candidate_snapshot":
        diagnostics["no_candidate_snapshot_ticks"] += 1
    elif failure_stage == "novelty_threshold":
        diagnostics["formation_threshold_block_ticks"] += 1
        diagnostics["novelty_threshold_block_ticks"] += 1
    elif failure_stage == "formation_min_regions":
        diagnostics["formation_threshold_block_ticks"] += 1
        diagnostics["min_region_block_ticks"] += 1
    elif failure_stage == "tracker_pending":
        diagnostics["tracker_pending_ticks"] += 1
    elif failure_stage == "ready_pattern":
        diagnostics["ready_pattern_ticks"] += 1

    if int(result.get("traces_formed", 0)) > 0:
        diagnostics["formed_ticks"] += 1

    for region_name in all_region_names():
        active_count = int(result.get(f"active_region_{region_name}_neurons", 0))
        diagnostics["active_region_sum"][region_name] += active_count
        if active_count > diagnostics["active_region_max"][region_name]:
            diagnostics["active_region_max"][region_name] = active_count

    for region_name, count in formation_debug.get("candidate_snapshot_region_counts", {}).items():
        region_count = int(count)
        diagnostics["candidate_region_sum"][region_name] += region_count
        if region_count > diagnostics["candidate_region_max"][region_name]:
            diagnostics["candidate_region_max"][region_name] = region_count

    diagnostics["tick_rows"].append(
        {
            "repeat_index": repeat_index,
            "tick_offset": tick_offset,
            "tick": int(result.get("tick", 0)),
            "novelty": round(float(formation_debug.get("novelty", result.get("novelty", 0.0))), 4),
            "active_traces": int(result.get("active_traces", 0)),
            "trace_candidates": int(result.get("trace_candidates", 0)),
            "traces_formed": int(result.get("traces_formed", 0)),
            "formation_attempted": bool(formation_debug.get("attempted", False)),
            "failure_stage": failure_stage,
            "candidate_snapshot_total_neurons": int(
                formation_debug.get("candidate_snapshot_total_neurons", 0)
            ),
            "candidate_snapshot_region_count": int(
                formation_debug.get("candidate_snapshot_region_count", 0)
            ),
            "candidate_snapshot_region_counts": {
                region_name: int(count)
                for region_name, count in formation_debug.get("candidate_snapshot_region_counts", {}).items()
                if int(count) > 0
            },
            "ready_pattern_count": int(formation_debug.get("ready_pattern_count", 0)),
        }
    )


def _harvest_learned_traces(
    probe_samples: list[dict[str, object]],
    ticks_per_sample: int,
    train_repeats: int,
    *,
    n_traces: int,
    seed_chunks: int | None,
    rest_ticks: int,
    overlay_terms: int,
    overlay_samples: int,
    overlay_working_memory_cap: int,
) -> tuple[
    dict[str, list[Trace]],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, object] | None,
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    harvested: dict[str, list[Trace]] = {}
    summaries: dict[str, dict[str, object]] = {}
    sample_text_coverage: dict[str, dict[str, object]] = {}
    overlay_summary: dict[str, object] | None = None
    sample_overlay_token_stats: dict[str, dict[str, object]] = {}
    sample_formation_diagnostics: dict[str, dict[str, object]] = {}

    for trained_sample in probe_samples:
        _, trace_store = seed_brain_fast(
            n_traces=n_traces,
            verbose=False,
            chunk_count=seed_chunks,
        )
        trace_store.clear()
        for sample in probe_samples:
            for ordinal, (label, primary_region) in enumerate(sample["anchors"], start=1):
                _add_anchor_trace(
                    trace_store,
                    str(sample["key"]),
                    str(label),
                    str(primary_region),
                    ordinal,
                )

        if overlay_terms > 0:
            current_overlay = apply_text_vocab_overlay(
                trace_store,
                "ag_news",
                max_terms=overlay_terms,
                max_samples=overlay_samples,
            )
            if overlay_summary is None:
                overlay_summary = current_overlay

        encoder = TextInput(trace_store)
        if not sample_text_coverage:
            overlay_trace_ids = {
                str(row["trace_id"])
                for row in (overlay_summary or {}).get("added_rows", [])
            }
            sample_text_coverage = {
                str(sample["key"]): _coverage_row(encoder, sample)
                for sample in probe_samples
            }
            sample_overlay_token_stats = {
                str(sample["key"]): _sample_token_source_stats(
                    encoder,
                    str(sample["text"]),
                    overlay_trace_ids,
                )
                for sample in probe_samples
            }
        tick_loop = TickLoop(trace_store)
        tick_loop.working_memory_overlay_cap = overlay_working_memory_cap
        learned_trace_ids: set[str] = set()
        matched_overlay_trace_ids: set[str] = set()
        sample_key = str(trained_sample["key"])
        formation_diagnostics = _empty_formation_diagnostics()

        for repeat_idx in range(train_repeats):
            tick_loop.reset_sample_boundary()
            encoded = encoder.encode(str(trained_sample["text"]))
            if overlay_terms > 0:
                matched_overlay_trace_ids.update(
                    trace_id
                    for trace_id in encoded.get("matched_traces", [])
                    if isinstance(trace_id, str) and trace_id.startswith("overlay_")
                )
            for tick_offset in range(ticks_per_sample):
                result = tick_loop.step()
                learned_trace_ids.update(tick_loop.trace_formation.recently_formed)
                _update_formation_diagnostics(
                    formation_diagnostics,
                    result,
                    tick_loop.trace_formation.last_step_debug,
                    repeat_idx,
                    tick_offset + 1,
                )
            for _ in range(rest_ticks):
                tick_loop.step()

        sample_formation_diagnostics[sample_key] = _finalize_formation_diagnostics(
            formation_diagnostics,
        )

        learned_cloned: list[Trace] = []
        for ordinal, trace_id in enumerate(sorted(learned_trace_ids), start=1):
            trace = trace_store.get(trace_id)
            if trace is None:
                continue
            learned_cloned.append(_clone_sample_trace(trace, sample_key, ordinal, kind="learned"))

        overlay_cloned: list[Trace] = []
        for ordinal, trace_id in enumerate(sorted(matched_overlay_trace_ids), start=1):
            trace = trace_store.get(trace_id)
            if trace is None:
                continue
            overlay_cloned.append(_clone_sample_trace(trace, sample_key, ordinal, kind="overlay"))

        cloned = learned_cloned + overlay_cloned

        harvested[sample_key] = cloned
        summaries[sample_key] = {
            "count": len(cloned),
            "learned_count": len(learned_cloned),
            "overlay_match_count": len(overlay_cloned),
            "size_avg": round(
                sum(trace.total_neurons() for trace in cloned) / len(cloned),
                3,
            ) if cloned else 0.0,
            "rows": [
                {
                    "trace_id": trace.id,
                    "source": "overlay" if trace.id.startswith("overlay_") else "learned",
                    "dominant_region": _dominant_region(trace),
                    "total_neurons": trace.total_neurons(),
                    "regions": {
                        region_name: len(neuron_ids)
                        for region_name, neuron_ids in trace.neurons.items()
                    },
                }
                for trace in cloned
            ],
        }

    return (
        harvested,
        summaries,
        sample_text_coverage,
        overlay_summary,
        sample_overlay_token_stats,
        sample_formation_diagnostics,
    )


def _sample_pair_overlap_rows(
    traces_a: list[Trace],
    traces_b: list[Trace],
) -> list[dict[str, object]]:
    rows = [_trace_overlap(trace_a, trace_b) for trace_a in traces_a for trace_b in traces_b]
    rows.sort(
        key=lambda row: (row["total_overlap"], row["trace_id_a"], row["trace_id_b"]),
        reverse=True,
    )
    return rows


def _inject_sample_traces(trace_store, trace_ids: list[str]) -> int:
    neuron_ids: set[int] = set()
    for trace_id in trace_ids:
        trace = trace_store.get(trace_id)
        if trace is None:
            continue
        for neuron_ids_region in trace.neurons.values():
            neuron_ids.update(neuron_ids_region)
    brain_core.inject_activations([(neuron_id, 1.0) for neuron_id in sorted(neuron_ids)])
    return len(neuron_ids)


def _binding_state(binding_id: int) -> dict[str, float | int | bool]:
    binding_info = brain_core.get_binding_info(binding_id)
    binding_activation = brain_core.get_binding_activation(binding_id, 0.01)

    weight = 0.0
    fires = 0
    confidence = 0.0
    last_fired = 0
    pattern_a_ratio = 0.0
    pattern_b_ratio = 0.0
    if binding_info is not None:
        weight, fires, confidence, last_fired = binding_info
    if binding_activation is not None:
        pattern_a_ratio, pattern_b_ratio = binding_activation

    return {
        "exists": binding_info is not None,
        "weight": weight,
        "fires": fires,
        "confidence": confidence,
        "last_fired": last_fired,
        "pattern_a_ratio": pattern_a_ratio,
        "pattern_a_active": pattern_a_ratio >= TRACE_ACTIVATION_THRESHOLD,
        "pattern_b_ratio": pattern_b_ratio,
        "pattern_b_active": pattern_b_ratio >= TRACE_ACTIVATION_THRESHOLD,
    }


def _evaluate_recall_candidate(
    tick_loop: TickLoop,
    trace_store,
    owner_by_trace_id: dict[str, str],
    cross_sample_binding_ids: set[int],
    sample_key: str,
    binding_row: dict[str, object],
    *,
    settle_ticks: int,
    probe_ticks: int,
    cue_fraction: float,
    cue_noise_fraction: float,
    cue_mode: str,
) -> dict[str, object]:
    cue_trace = trace_store.get(str(binding_row["trace_id_a"]))
    partner_trace = trace_store.get(str(binding_row["trace_id_b"]))
    if cue_trace is None or partner_trace is None:
        return {
            "binding": binding_row,
            "reason": "missing_trace",
        }

    tick_loop.reset_probe_boundary()
    for _ in range(settle_ticks):
        tick_loop.step(learn=False)

    baseline_hits = [
        trace_id
        for trace_id, _score in tick_loop.last_active_traces
        if owner_by_trace_id.get(trace_id) is not None
    ]

    cue_rng = random.Random(_candidate_probe_seed(sample_key, cue_trace.id))
    injected, cue_signal_neurons, cue_noise_neurons = _build_trace_cue(
        cue_trace,
        cue_fraction,
        cue_noise_fraction,
        cue_rng,
        cue_mode=cue_mode,
    )
    tick_rows: list[dict[str, object]] = []
    partner_trace_total_neurons = partner_trace.total_neurons()
    partner_binding_region_neurons = len(
        partner_trace.neurons.get(str(binding_row["region_b"]), [])
    )
    for tick_index in range(probe_ticks):
        if tick_index == 0:
            brain_core.inject_activations(injected)
        result = tick_loop.step(learn=False)
        cue_hit, cue_rank, cue_score = _trace_match_metrics(
            tick_loop.last_active_traces,
            cue_trace.id,
        )
        partner_hit, partner_rank, partner_score = _trace_match_metrics(
            tick_loop.last_active_traces,
            partner_trace.id,
        )
        false_trace_id, false_trace_rank, false_trace_score = _top_non_excluded_trace(
            tick_loop.last_active_traces,
            {cue_trace.id, partner_trace.id},
        )
        false_trace_sample = owner_by_trace_id.get(false_trace_id or "")
        cross_sample_hits = [
            trace_id
            for trace_id, _score in tick_loop.last_active_traces
            if owner_by_trace_id.get(trace_id) not in (None, sample_key)
        ]
        active_bindings = {
            active_binding_id
            for active_binding_id, _score in brain_core.evaluate_bindings(0.01)
        }
        partial_bindings = set(brain_core.find_partial_bindings(0.01))
        recall_candidates = {
            binding_id: (relative_weight, source_ratio)
            for binding_id, relative_weight, source_ratio in tick_loop.last_binding_recall_candidates
        }
        binding_state = _binding_state(int(binding_row["binding_id"]))
        cross_sample_binding_active = bool(active_bindings & cross_sample_binding_ids)
        active_set = tick_loop.history.current.active_set() if tick_loop.history.current is not None else set()
        partner_trace_active_neurons = sum(
            1
            for region_neurons in partner_trace.neurons.values()
            for neuron_id in region_neurons
            if neuron_id in active_set
        )
        partner_trace_active_ratio = (
            partner_trace_active_neurons / partner_trace_total_neurons
            if partner_trace_total_neurons > 0
            else 0.0
        )
        tick_rows.append(
            {
                "tick": result["tick"],
                "cue_trace_hit": cue_hit,
                "cue_trace_rank": cue_rank,
                "cue_trace_score": cue_score,
                "partner_trace_hit": partner_hit,
                "partner_trace_rank": partner_rank,
                "partner_trace_score": partner_score,
                "binding_active": int(binding_row["binding_id"]) in active_bindings,
                "binding_partial": int(binding_row["binding_id"]) in partial_bindings,
                "binding_recall_candidate": int(binding_row["binding_id"]) in recall_candidates,
                "binding_recall_relative_weight": recall_candidates.get(int(binding_row["binding_id"]), (0.0, 0.0))[0],
                "binding_recall_source_ratio": recall_candidates.get(int(binding_row["binding_id"]), (0.0, 0.0))[1],
                "pattern_a_ratio": binding_state["pattern_a_ratio"],
                "pattern_b_ratio": binding_state["pattern_b_ratio"],
                "pattern_b_active": binding_state["pattern_b_active"],
                "partner_trace_active_neurons": partner_trace_active_neurons,
                "partner_trace_active_ratio": partner_trace_active_ratio,
                "cross_sample_trace_hits": cross_sample_hits,
                "cross_sample_binding_active": cross_sample_binding_active,
                "false_trace_id": false_trace_id,
                "false_trace_rank": false_trace_rank,
                "false_trace_score": false_trace_score,
                "false_trace_sample": false_trace_sample,
                "selective_recall": (partner_hit or binding_state["pattern_b_active"]) and not cross_sample_hits and not cross_sample_binding_active,
            }
        )

    partner_ranks = [
        row["partner_trace_rank"]
        for row in tick_rows
        if row["partner_trace_rank"] is not None
    ]
    false_ranks = [
        row["false_trace_rank"]
        for row in tick_rows
        if row["false_trace_rank"] is not None and row["false_trace_sample"] not in (None, sample_key)
    ]
    partner_trace_active_ratios = [row["partner_trace_active_ratio"] for row in tick_rows]
    return {
        "binding": binding_row,
        "baseline_trace_hits": baseline_hits,
        "cue_trace_id": cue_trace.id,
        "partner_trace_id": partner_trace.id,
        "cue_signal_neurons": cue_signal_neurons,
        "cue_noise_neurons": cue_noise_neurons,
        "trace_activation_threshold": TRACE_ACTIVATION_THRESHOLD,
        "partner_total_neurons": partner_trace_total_neurons,
        "partner_binding_region_neurons": partner_binding_region_neurons,
        "partner_binding_region_fraction_of_trace": (
            partner_binding_region_neurons / partner_trace_total_neurons
            if partner_trace_total_neurons > 0
            else 0.0
        ),
        "partner_trace_hit_rate": round(
            sum(1 for row in tick_rows if row["partner_trace_hit"]) / len(tick_rows),
            4,
        ),
        "partner_pattern_active_rate": round(
            sum(1 for row in tick_rows if row["pattern_b_active"]) / len(tick_rows),
            4,
        ),
        "partner_trace_active_ratio_avg": round(
            sum(partner_trace_active_ratios) / len(partner_trace_active_ratios),
            4,
        ) if partner_trace_active_ratios else 0.0,
        "partner_trace_active_ratio_max": round(
            max(partner_trace_active_ratios),
            4,
        ) if partner_trace_active_ratios else 0.0,
        "cross_sample_trace_hit_rate": round(
            sum(1 for row in tick_rows if row["cross_sample_trace_hits"]) / len(tick_rows),
            4,
        ),
        "cross_sample_binding_active_rate": round(
            sum(1 for row in tick_rows if row["cross_sample_binding_active"]) / len(tick_rows),
            4,
        ),
        "selective_recall_rate": round(
            sum(1 for row in tick_rows if row["selective_recall"]) / len(tick_rows),
            4,
        ),
        "partner_rank_avg": round(sum(partner_ranks) / len(partner_ranks), 3) if partner_ranks else None,
        "cross_sample_false_rank_avg": round(sum(false_ranks) / len(false_ranks), 3) if false_ranks else None,
        "ticks": tick_rows,
    }


def run_text_binding_probe(
    max_samples: int,
    ticks_per_sample: int,
    train_repeats: int,
    threads: int,
    output_path: str,
    *,
    n_traces: int = 5500,
    seed_chunks: int | None = 1,
    rest_ticks: int = 1,
    settle_ticks: int = 3,
    probe_ticks: int = 4,
    cue_fraction: float = 0.75,
    cue_noise_fraction: float = 0.0,
    cue_mode: str = "dominant-region",
    overlay_terms: int = 0,
    overlay_samples: int = 500,
    overlay_working_memory_cap: int | None = None,
) -> dict[str, object]:
    if threads > 0:
        try:
            brain_core.set_num_threads(threads)
        except Exception:
            pass
    actual_threads = brain_core.get_num_threads()
    effective_overlay_working_memory_cap = 0
    if overlay_terms > 0:
        if overlay_working_memory_cap is None:
            effective_overlay_working_memory_cap = _derive_overlay_working_memory_cap(
                WORKING_MEMORY_CAPACITY,
            )
        else:
            effective_overlay_working_memory_cap = max(0, overlay_working_memory_cap)

    probe_samples = _prepare_probe_samples(max_samples)
    (
        harvested_by_sample,
        harvested_summary,
        sample_text_coverage,
        overlay_summary,
        sample_overlay_token_stats,
        sample_formation_diagnostics,
    ) = _harvest_learned_traces(
        probe_samples,
        ticks_per_sample,
        train_repeats,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        rest_ticks=rest_ticks,
        overlay_terms=overlay_terms,
        overlay_samples=overlay_samples,
        overlay_working_memory_cap=effective_overlay_working_memory_cap,
    )
    sample_known_token_avg = round(
        sum(int(row["known"]) for row in sample_text_coverage.values()) / len(sample_text_coverage),
        4,
    ) if sample_text_coverage else 0.0

    hp_hubble_overlap = _sample_pair_overlap_rows(
        harvested_by_sample.get("hp_profit", []),
        harvested_by_sample.get("hubble_space", []),
    )

    _, trace_store = seed_brain_fast(
        n_traces=n_traces,
        verbose=False,
        chunk_count=seed_chunks,
    )
    trace_store.clear()
    owner_by_trace_id: dict[str, str] = {}
    learned_trace_ids_by_sample: dict[str, list[str]] = {}
    for sample in probe_samples:
        sample_key = str(sample["key"])
        trace_ids: list[str] = []
        for trace in harvested_by_sample.get(sample_key, []):
            trace_store.add(trace)
            trace_ids.append(trace.id)
            owner_by_trace_id[trace.id] = sample_key
        learned_trace_ids_by_sample[sample_key] = trace_ids

    tick_loop = TickLoop(trace_store)
    training_results: list[dict[str, object]] = []
    formed_binding_ids: set[int] = set()
    training_activity: dict[str, list[dict[str, object]]] = {
        str(sample["key"]): []
        for sample in probe_samples
    }

    for repeat_idx in range(train_repeats):
        for sample in probe_samples:
            sample_key = str(sample["key"])
            sample_trace_ids = learned_trace_ids_by_sample.get(sample_key, [])
            if len(sample_trace_ids) < 2:
                continue
            tick_loop.reset_runtime_boundary(preserve_binding_state=True)
            injected_neurons = _inject_sample_traces(trace_store, sample_trace_ids)
            for tick_offset in range(ticks_per_sample):
                result = tick_loop.step(allow_trace_formation=False)
                result["sample_key"] = sample_key
                result["repeat_index"] = repeat_idx
                result["injected_neurons"] = injected_neurons if tick_offset == 0 else 0
                training_results.append(result)
                formed_binding_ids.update(tick_loop.binding_formation.recently_formed)
                active_same_sample_trace_ids = [
                    trace_id
                    for trace_id, _score in tick_loop.last_active_traces
                    if owner_by_trace_id.get(trace_id) == sample_key
                ]
                training_activity[sample_key].append(
                    {
                        "repeat_index": repeat_idx,
                        "tick_offset": tick_offset + 1,
                        "active_same_sample_trace_ids": active_same_sample_trace_ids,
                        "active_same_sample_count": len(active_same_sample_trace_ids),
                    }
                )
            for _ in range(rest_ticks):
                result = tick_loop.step(allow_trace_formation=False)
                result["sample_key"] = sample_key
                result["repeat_index"] = repeat_idx
                result["injected_neurons"] = 0
                training_results.append(result)

    trace_candidates_avg = (
        sum(float(row.get("trace_candidates", 0.0)) for row in training_results) / len(training_results)
        if training_results else 0.0
    )
    binding_candidates_avg = (
        sum(float(row.get("binding_candidates", 0.0)) for row in training_results) / len(training_results)
        if training_results else 0.0
    )
    binding_candidates_per_trace_candidate_avg = (
        binding_candidates_avg / trace_candidates_avg
        if trace_candidates_avg > 0
        else 0.0
    )

    binding_rows: list[dict[str, object]] = []
    same_sample_binding_ids: set[int] = set()
    cross_sample_binding_ids: set[int] = set()
    detail_map = tick_loop.binding_formation.binding_details
    for binding_id in sorted(formed_binding_ids):
        detail = detail_map.get(binding_id)
        if detail is None:
            continue
        trace_id_a = str(detail["trace_id_a"])
        trace_id_b = str(detail["trace_id_b"])
        sample_a = owner_by_trace_id.get(trace_id_a)
        sample_b = owner_by_trace_id.get(trace_id_b)
        state = _binding_state(binding_id)
        row = {
            "binding_id": binding_id,
            "trace_id_a": trace_id_a,
            "region_a": detail["region_a"],
            "trace_id_b": trace_id_b,
            "region_b": detail["region_b"],
            "sample_a": sample_a,
            "sample_b": sample_b,
            "scope": "same" if sample_a == sample_b else "cross",
            "weight": state["weight"],
            "fires": state["fires"],
            "confidence": state["confidence"],
            "last_fired": state["last_fired"],
        }
        _annotate_binding_row_selection_fields(trace_store, row)
        binding_rows.append(row)
        if row["scope"] == "same":
            same_sample_binding_ids.add(binding_id)
        else:
            cross_sample_binding_ids.add(binding_id)

    binding_rows.sort(key=_binding_row_sort_key, reverse=True)

    training_activity_summary = {
        sample_key: {
            "max_active_same_sample": max(
                (row["active_same_sample_count"] for row in rows),
                default=0,
            ),
            "ticks_with_ge2": sum(
                1 for row in rows if row["active_same_sample_count"] >= 2
            ),
            "ticks": rows,
        }
        for sample_key, rows in training_activity.items()
    }

    recall_results: dict[str, object] = {}
    for sample in probe_samples:
        sample_key = str(sample["key"])
        same_sample_rows = [
            row
            for row in binding_rows
            if row["sample_a"] == sample_key and row["sample_b"] == sample_key
        ]
        if not same_sample_rows:
            recall_results[sample_key] = {
                "selected_binding": None,
                "best_binding": None,
                "best_candidate": None,
                "selection_mode": "best_same_sample_candidate_by_partner_trace_active_ratio_max",
                "same_sample_candidate_count": 0,
                "evaluated_candidate_count": 0,
                "skipped_candidate_count": 0,
                "evaluated_candidate_binding_ids": [],
                "partner_trace_active_ratio_max_distribution": [],
                "partner_trace_active_ratio_avg_distribution": [],
                "candidate_results": [],
                "reason": "no_same_sample_binding",
            }
            continue

        candidate_results = [
            _evaluate_recall_candidate(
                tick_loop,
                trace_store,
                owner_by_trace_id,
                cross_sample_binding_ids,
                sample_key,
                binding_row,
                settle_ticks=settle_ticks,
                probe_ticks=probe_ticks,
                cue_fraction=cue_fraction,
                cue_noise_fraction=cue_noise_fraction,
                cue_mode=cue_mode,
            )
            for binding_row in same_sample_rows
        ]
        candidate_summary = _summarize_candidate_recall_results(candidate_results)
        best_candidate = candidate_summary["best_candidate"]
        recall_result = {
            "selected_binding": best_candidate["binding"] if best_candidate is not None else None,
            "best_binding": best_candidate["binding"] if best_candidate is not None else None,
            "best_candidate": best_candidate,
            "selection_mode": "best_same_sample_candidate_by_partner_trace_active_ratio_max",
            "same_sample_candidate_count": len(same_sample_rows),
            **candidate_summary,
        }
        if best_candidate is None:
            recall_result["reason"] = "no_evaluable_same_sample_candidate"
            recall_results[sample_key] = recall_result
            continue

        recall_result.update({
            key: value
            for key, value in best_candidate.items()
            if key != "binding"
        })
        recall_results[sample_key] = recall_result

    output = {
        "benchmark": "text_binding_probe",
        "dataset": "ag_news",
        "threads": actual_threads,
        "seed_traces": n_traces,
        "seed_chunks": seed_chunks,
        "ticks_per_sample": ticks_per_sample,
        "train_repeats": train_repeats,
        "rest_ticks": rest_ticks,
        "settle_ticks": settle_ticks,
        "probe_ticks": probe_ticks,
        "cue_fraction": cue_fraction,
        "cue_noise_fraction": cue_noise_fraction,
        "cue_mode": cue_mode,
        "overlay_terms": overlay_terms,
        "overlay_samples": overlay_samples,
        "overlay_working_memory_cap": effective_overlay_working_memory_cap,
        "overlay_summary": overlay_summary,
        "sample_text_coverage": sample_text_coverage,
        "sample_overlay_token_stats": sample_overlay_token_stats,
        "sample_trace_formation_diagnostics": sample_formation_diagnostics,
        "sample_known_token_avg": sample_known_token_avg,
        "harvested_trace_summary": harvested_summary,
        "sample_overlap": {
            "hp_profit__hubble_space": {
                "max_total_overlap": hp_hubble_overlap[0]["total_overlap"] if hp_hubble_overlap else 0,
                "rows": hp_hubble_overlap,
            }
        },
        "binding_training_summary": {
            "training_ticks": len(training_results),
            "trace_candidates_avg": round(trace_candidates_avg, 4),
            "binding_candidates_avg": round(binding_candidates_avg, 4),
            "binding_candidates_per_trace_candidate_avg": round(binding_candidates_per_trace_candidate_avg, 5),
            "formed_binding_count": len(binding_rows),
            "same_sample_binding_count": sum(1 for row in binding_rows if row["scope"] == "same"),
            "cross_sample_binding_count": sum(1 for row in binding_rows if row["scope"] == "cross"),
        },
        "training_activity": training_activity_summary,
        "formed_bindings": binding_rows,
        "recall_results": recall_results,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print("\nTEXT BINDING PROBE SUMMARY")
    binding_summary = output["binding_training_summary"]
    print(
        "  formed="
        f"{binding_summary['formed_binding_count']}, same={binding_summary['same_sample_binding_count']}, "
        f"cross={binding_summary['cross_sample_binding_count']}, "
        f"bind/trace={binding_summary['binding_candidates_per_trace_candidate_avg']:.5f}"
    )
    overlap_summary = output["sample_overlap"]["hp_profit__hubble_space"]
    print(f"  hp_profit vs hubble_space overlap: max_total={overlap_summary['max_total_overlap']}")
    if overlay_summary is not None:
        print(
            "  overlay="
            f"{overlay_summary['added_trace_count']} traces, "
            f"selected_content_share={overlay_summary['selected_term_content_coverage']:.2%}"
        )
        if effective_overlay_working_memory_cap > 0:
            print(f"  overlay working-memory cap={effective_overlay_working_memory_cap}")
    if sample_text_coverage:
        print(f"  probe sample known-token avg={sample_known_token_avg:.2f}")
    for sample_key, recall in recall_results.items():
        selected = recall.get("selected_binding")
        coverage = sample_text_coverage.get(sample_key)
        coverage_prefix = ""
        if coverage is not None:
            coverage_prefix = (
                f"known={coverage['known']}, unknown={coverage['unknown']}, "
                f"coverage={coverage['coverage']:.2%}, "
            )
        if selected is None:
            print(f"  {sample_key}: {coverage_prefix}no same-sample binding")
            continue
        distribution = recall.get("partner_trace_active_ratio_max_distribution", [])
        distribution_text = ", ".join(
            f"{row['ratio']:.4f}x{row['count']}"
            for row in distribution[:4]
        )
        top_k = recall.get("selection_reliability", {}).get("top_k", {})
        top_1 = top_k.get("top_1", {})
        top_3 = top_k.get("top_3", {})
        print(
            f"  {sample_key}: {coverage_prefix}candidates={recall['evaluated_candidate_count']}/{recall['same_sample_candidate_count']}, "
            f"best_ratio={recall['partner_trace_active_ratio_max']:.4f}, "
            f"top1={top_1.get('best_ratio', 0.0):.4f}, "
            f"top3={top_3.get('best_ratio', 0.0):.4f}, "
            f"tie_p(top3)={top_3.get('tie_aware_meets_success_threshold_probability', 0.0):.2f}, "
            f"ratio_dist=[{distribution_text}], "
            f"partner_hit={recall['partner_trace_hit_rate']:.2f}, "
            f"partner_pattern={recall['partner_pattern_active_rate']:.2f}, "
            f"cross_trace={recall['cross_sample_trace_hit_rate']:.2f}, "
            f"cross_binding={recall['cross_sample_binding_active_rate']:.2f}, "
            f"selective={recall['selective_recall_rate']:.2f}"
        )

    print(f"Binding probe metrics saved to: {path}")
    return output