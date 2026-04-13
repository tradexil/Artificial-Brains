"""Reproducible visual-learning probe for CIFAR trace selectivity and recall quality."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import brain_core
import numpy as np

from brain.datasets.downloader import load_image_dataset
from brain.input.visual_input import (
    VISUAL_TRACE_FEATURE_FAMILIES,
    VisualInput,
    visual_family_for_neuron,
)
from brain.learning.tick_loop import TickLoop
from brain.seed.seed_runner import seed_brain_fast
from brain.serialize.runtime_bundle import load_runtime_bundle
from brain.structures.brain_state import ActivationSnapshot
from brain.structures.neuron_map import region_for_neuron
from brain.structures.trace_store import Trace, TraceStore


DEFAULT_VISUAL_LEARNING_PROBE_LABELS = (
    "airplane",
    "automobile",
    "dog",
    "ship",
)

_VISUAL_PROBE_DATASET_SAMPLE_LIMIT = 200
_VISUAL_TRACE_TARGET_SIZE = 48
_VISUAL_QUALITY_MASK_TRIALS = 64
_VISUAL_QUALITY_TARGET_FAMILIES = ("low", "mid", "spatial")


def _visual_signal_map(image: Any) -> dict[int, float]:
    encoder = VisualInput()
    pixels = encoder._normalize(image)
    signals: list[tuple[int, float]] = []
    signals.extend(encoder._extract_low_level(pixels))
    signals.extend(encoder._extract_mid_level(pixels))
    signals.extend(encoder._extract_spatial(pixels))

    activation_by_id: dict[int, float] = {}
    for neuron_id, activation in signals:
        activation_by_id[neuron_id] = max(
            activation_by_id.get(neuron_id, 0.0),
            float(activation),
        )
    return activation_by_id


def _build_visual_quality_cache(
    probe_samples: list[dict[str, Any]],
    cue_fraction: float,
    *,
    mask_trials: int = _VISUAL_QUALITY_MASK_TRIALS,
    target_families: tuple[str, ...] = _VISUAL_QUALITY_TARGET_FAMILIES,
) -> dict[str, dict[str, object]]:
    raw_signal_maps = {
        str(sample["key"]): _visual_signal_map(sample["image"])
        for sample in probe_samples
    }

    quality_cache: dict[str, dict[str, object]] = {}
    for trained_sample in probe_samples:
        trained_key = str(trained_sample["key"])
        trained_signals = raw_signal_maps[trained_key]
        other_class_counts: Counter[int] = Counter()
        for sample_key, activation_map in raw_signal_maps.items():
            if sample_key == trained_key:
                continue
            for neuron_id in activation_map:
                other_class_counts[neuron_id] += 1

        survival_counts: Counter[int] = Counter()
        for trial_index in range(mask_trials):
            cue_image = _build_visual_partial_cue(
                trained_sample["image"],
                cue_fraction,
                salt=f"{trained_key}:quality:{trial_index}",
            )
            for neuron_id in _visual_signal_map(cue_image):
                survival_counts[neuron_id] += 1

        quality_scores: dict[int, float] = {}
        family_summary: dict[str, dict[str, object]] = {}
        for family_name in target_families:
            family_ids = {
                neuron_id
                for neuron_id in trained_signals
                if visual_family_for_neuron(neuron_id) == family_name
            }
            unique_ids = {
                neuron_id
                for neuron_id in family_ids
                if other_class_counts.get(neuron_id, 0) == 0
            }
            shared_ids = family_ids - unique_ids
            for neuron_id in unique_ids:
                survival = survival_counts.get(neuron_id, 0) / max(1, mask_trials)
                if survival > 0.0:
                    quality_scores[neuron_id] = survival

            ranked_unique = sorted(
                unique_ids,
                key=lambda neuron_id: (
                    quality_scores.get(neuron_id, 0.0),
                    trained_signals.get(neuron_id, 0.0),
                    -neuron_id,
                ),
                reverse=True,
            )
            family_summary[family_name] = {
                "raw_total": len(family_ids),
                "raw_unique": len(unique_ids),
                "raw_shared": len(shared_ids),
                "positive_quality_count": sum(
                    1 for neuron_id in unique_ids if quality_scores.get(neuron_id, 0.0) > 0.0
                ),
                "top_quality_ids": ranked_unique[:8],
            }

        quality_cache[trained_key] = {
            "quality_scores": quality_scores,
            "families": family_summary,
        }

    return quality_cache


def _activation_snapshot_from_signal_map(
    activation_map: dict[int, float],
    *,
    tick: int = 0,
) -> ActivationSnapshot:
    active_values = sorted(
        ((neuron_id, float(activation)) for neuron_id, activation in activation_map.items()),
        key=lambda item: item[0],
    )
    active_ids = [neuron_id for neuron_id, _activation in active_values]
    active_neurons = {"visual": active_values} if active_values else {}
    region_active_counts = {"visual": len(active_values)} if active_values else {}
    return ActivationSnapshot(
        tick=tick,
        active_neurons=active_neurons,
        active_values=active_values,
        total_active=len(active_values),
        active_ids=active_ids,
        region_active_counts=region_active_counts,
    )


def _visual_signal_family_counts(activation_map: dict[int, float]) -> dict[str, int]:
    counts = {
        family_name: 0
        for family_name in VISUAL_TRACE_FEATURE_FAMILIES
    }
    for neuron_id in activation_map:
        family_name = visual_family_for_neuron(neuron_id)
        if family_name in counts:
            counts[family_name] += 1
    counts["total"] = len(activation_map)
    return counts


def _formation_fingerprint(active_neurons: dict[str, list[tuple[int, float]]]) -> tuple[int, ...]:
    fingerprint: list[int] = []
    for neurons in active_neurons.values():
        fingerprint.extend(neuron_id for neuron_id, _activation in neurons)
    return tuple(sorted(set(fingerprint)))


def _jaccard_ratio(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a and not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def _fingerprint_id(fingerprint: tuple[int, ...]) -> str:
    joined = ",".join(str(neuron_id) for neuron_id in fingerprint)
    return hashlib.sha1(joined.encode("ascii")).hexdigest()[:12]


def _stats_or_none(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "avg": None, "max": None}
    return {
        "min": round(min(values), 4),
        "avg": round(sum(values) / len(values), 4),
        "max": round(max(values), 4),
    }


def _summarize_fingerprint_sequence(
    fingerprints: list[tuple[int, ...]],
    target_size: int = _VISUAL_TRACE_TARGET_SIZE,
) -> dict[str, object]:
    consecutive = [
        _jaccard_ratio(fingerprints[index - 1], fingerprints[index])
        for index in range(1, len(fingerprints))
    ]
    exact_prev_repeat_ticks = sum(
        1 for index in range(1, len(fingerprints)) if fingerprints[index] == fingerprints[index - 1]
    )
    tracker_match_prev_ticks = sum(1 for score in consecutive if score > 0.5)

    target_tick_indices = [
        index + 1
        for index, fingerprint in enumerate(fingerprints)
        if len(fingerprint) == target_size
    ]
    target_fingerprints = [
        fingerprint for fingerprint in fingerprints if len(fingerprint) == target_size
    ]
    target_consecutive = [
        _jaccard_ratio(target_fingerprints[index - 1], target_fingerprints[index])
        for index in range(1, len(target_fingerprints))
    ]

    return {
        "tick_count": len(fingerprints),
        "unique_exact_fingerprint_count": len(set(fingerprints)),
        "exact_prev_repeat_ticks": exact_prev_repeat_ticks,
        "tracker_match_prev_ticks": tracker_match_prev_ticks,
        "consecutive_jaccard": _stats_or_none(consecutive),
        "first_target_size_tick": target_tick_indices[0] if target_tick_indices else None,
        "target_size_tick_count": len(target_tick_indices),
        "target_size_consecutive_jaccard": _stats_or_none(target_consecutive),
    }


def _dominant_region(trace: Trace) -> str | None:
    if not trace.neurons:
        return None
    return max(trace.neurons.items(), key=lambda item: len(item[1]))[0]


def _summarize_learned_traces(
    trace_store: TraceStore,
    learned_trace_ids: set[str],
    formation_order: list[str] | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    region_totals: Counter[str] = Counter()
    ordered_trace_ids: list[str] = []
    seen_trace_ids: set[str] = set()
    for trace_id in formation_order or []:
        if trace_id in learned_trace_ids and trace_id not in seen_trace_ids:
            ordered_trace_ids.append(trace_id)
            seen_trace_ids.add(trace_id)
    for trace_id in sorted(learned_trace_ids):
        if trace_id not in seen_trace_ids:
            ordered_trace_ids.append(trace_id)
            seen_trace_ids.add(trace_id)

    for trace_id in ordered_trace_ids:
        trace = trace_store.get(trace_id)
        if trace is None:
            continue
        region_counts = {
            region_name: len(neurons)
            for region_name, neurons in sorted(trace.neurons.items())
        }
        for region_name, count in region_counts.items():
            region_totals[region_name] += count
        rows.append(
            {
                "trace_id": trace_id,
                "label": trace.label,
                "dominant_region": _dominant_region(trace),
                "total_neurons": trace.total_neurons(),
                "regions": region_counts,
                "co_traces": list(trace.co_traces),
            }
        )

    sizes = [int(row["total_neurons"]) for row in rows]
    dominant_regions: Counter[str] = Counter()
    for row in rows:
        dominant = row["dominant_region"]
        if dominant is not None:
            dominant_regions[str(dominant)] += 1

    return {
        "count": len(rows),
        "primary_trace_id": rows[0]["trace_id"] if rows else None,
        "size_min": min(sizes) if sizes else 0,
        "size_max": max(sizes) if sizes else 0,
        "size_avg": round(sum(sizes) / len(sizes), 3) if sizes else 0.0,
        "dominant_regions": dict(dominant_regions),
        "region_totals": dict(sorted(region_totals.items())),
        "rows": rows,
    }


def _copy_trace(trace: Trace | None) -> Trace | None:
    if trace is None:
        return None
    return Trace.from_dict(trace.to_dict())


def _collect_primary_visual_reference_traces(
    probe_samples: list[dict[str, Any]],
    *,
    ticks_per_sample: int,
    train_repeats: int,
    n_traces: int,
    seed_chunks: int | None,
    rest_ticks: int,
    visual_quality_cache: dict[str, dict[str, object]] | None = None,
    initial_state_dir: str | None = None,
) -> dict[str, Trace | None]:
    primary_traces: dict[str, Trace | None] = {}

    for trained_sample in probe_samples:
        if initial_state_dir is not None:
            trace_store, tick_loop, _python_state, _metadata = load_runtime_bundle(
                initial_state_dir
            )
            trace_store.clear()
            brain_core.replace_bindings([])
            tick_loop.reset_runtime_boundary()
        else:
            _, trace_store = seed_brain_fast(
                n_traces=n_traces,
                verbose=False,
                chunk_count=seed_chunks,
            )
            trace_store.clear()
            tick_loop = TickLoop(trace_store)
        encoder = VisualInput()
        learned_trace_order: list[str] = []
        learned_trace_ids: set[str] = set()
        quality_entry = (visual_quality_cache or {}).get(str(trained_sample["key"]), {})
        tick_loop.trace_formation.set_visual_quality_scores(
            dict(quality_entry.get("quality_scores", {})),
            families=_VISUAL_QUALITY_TARGET_FAMILIES,
        )

        for _ in range(train_repeats):
            tick_loop.reset_sample_boundary()
            for _ in range(ticks_per_sample):
                encoder.encode(trained_sample["image"])
                tick_loop.step(allow_binding_formation=False)
                for trace_id in tick_loop.trace_formation.recently_formed:
                    if trace_id not in learned_trace_ids:
                        learned_trace_order.append(trace_id)
                    learned_trace_ids.add(trace_id)
            for _ in range(rest_ticks):
                tick_loop.step(allow_binding_formation=False)

        primary_trace_id = learned_trace_order[0] if learned_trace_order else None
        primary_traces[str(trained_sample["key"])] = _copy_trace(
            trace_store.get(primary_trace_id) if primary_trace_id is not None else None
        )

    return primary_traces


def _build_probe_trace_catalog(
    reference_primary_traces: dict[str, Trace | None],
    *,
    trained_key: str,
    trained_trace: Trace | None,
) -> dict[str, Trace | None]:
    """Use the current run's trained trace for own-class diagnostics.

    The probe also keeps a cross-class reference catalog so wrong-class overlap
    remains comparable across trained runs.
    """
    catalog = dict(reference_primary_traces)
    if trained_trace is not None:
        catalog[trained_key] = _copy_trace(trained_trace)
    return catalog


def _snapshot_active_ids_by_region(snapshot: ActivationSnapshot) -> dict[str, set[int]]:
    active_by_region: dict[str, set[int]] = {}
    if snapshot.active_neurons:
        for region_name, neurons in snapshot.active_neurons.items():
            if neurons:
                active_by_region[region_name] = {neuron_id for neuron_id, _ in neurons}
        return active_by_region

    for neuron_id, _activation in snapshot.active_values:
        region_name = region_for_neuron(neuron_id)
        if region_name is None:
            continue
        active_by_region.setdefault(region_name, set()).add(neuron_id)
    return active_by_region


def _visual_family_overlap_breakdown(
    ids_a: set[int],
    ids_b: set[int],
) -> dict[str, dict[str, float | int]]:
    breakdown: dict[str, dict[str, float | int]] = {}
    for family_name in VISUAL_TRACE_FEATURE_FAMILIES:
        family_a = {neuron_id for neuron_id in ids_a if visual_family_for_neuron(neuron_id) == family_name}
        family_b = {neuron_id for neuron_id in ids_b if visual_family_for_neuron(neuron_id) == family_name}
        shared = len(family_a & family_b)
        smaller = min(len(family_a), len(family_b))
        union = len(family_a | family_b)
        breakdown[family_name] = {
            "count_a": len(family_a),
            "count_b": len(family_b),
            "shared_count": shared,
            "shared_ratio_of_smaller": round(shared / smaller, 4) if smaller > 0 else 0.0,
            "jaccard": round(shared / union, 4) if union > 0 else 0.0,
        }
    return breakdown


def _trace_overlap_breakdown_for_snapshot(
    snapshot: ActivationSnapshot,
    trace: Trace,
) -> dict[str, object]:
    active_by_region = _snapshot_active_ids_by_region(snapshot)
    region_breakdown: dict[str, dict[str, float | int]] = {}
    total_shared = 0

    region_names = sorted(set(active_by_region) | set(trace.neurons))
    for region_name in region_names:
        active_ids = active_by_region.get(region_name, set())
        trace_ids = set(trace.neurons.get(region_name, []))
        shared = len(active_ids & trace_ids)
        total_shared += shared
        active_count = len(active_ids)
        trace_count = len(trace_ids)
        region_breakdown[region_name] = {
            "active_count": active_count,
            "trace_count": trace_count,
            "shared_count": shared,
            "trace_overlap_ratio": round(shared / trace_count, 4) if trace_count > 0 else 0.0,
            "active_overlap_ratio": round(shared / active_count, 4) if active_count > 0 else 0.0,
        }
        if region_name == "visual":
            region_breakdown[region_name]["families"] = _visual_family_overlap_breakdown(
                active_ids,
                trace_ids,
            )

    total_trace_neurons = trace.total_neurons()
    total_active = snapshot.total_active if snapshot.total_active > 0 else len(snapshot.active_ids)
    return {
        "trace_id": trace.id,
        "shared_total": total_shared,
        "trace_overlap_ratio": round(total_shared / total_trace_neurons, 4)
        if total_trace_neurons > 0
        else 0.0,
        "active_overlap_ratio": round(total_shared / total_active, 4)
        if total_active > 0
        else 0.0,
        "regions": region_breakdown,
    }


def _top_overlap_region(region_breakdown: dict[str, dict[str, float | int]]) -> str | None:
    best_region = None
    best_key: tuple[int, float, float, str] | None = None
    for region_name, row in region_breakdown.items():
        candidate_key = (
            int(row.get("shared_count", 0)),
            float(row.get("trace_overlap_ratio", 0.0) or 0.0),
            float(row.get("active_overlap_ratio", 0.0) or 0.0),
            region_name,
        )
        if best_key is None or candidate_key > best_key:
            best_region = region_name
            best_key = candidate_key
    return best_region


def _cue_activation_breakdown(
    snapshot: ActivationSnapshot,
    class_primary_traces: dict[str, Trace | None],
    *,
    cue_key: str,
    trained_key: str,
    probe_tick: int,
    trace_hit_count: int,
    best_score: float,
) -> dict[str, object]:
    region_active_counts = {
        region_name: len(active_ids)
        for region_name, active_ids in sorted(_snapshot_active_ids_by_region(snapshot).items())
        if active_ids
    }

    class_trace_overlaps: dict[str, dict[str, object]] = {}
    for class_key, trace in sorted(class_primary_traces.items()):
        if trace is None:
            continue
        overlap = _trace_overlap_breakdown_for_snapshot(snapshot, trace)
        overlap["top_region"] = _top_overlap_region(overlap["regions"])
        class_trace_overlaps[class_key] = overlap

    overlap_rank = lambda item: (
        float(item[1].get("trace_overlap_ratio", 0.0) or 0.0),
        int(item[1].get("shared_total", 0) or 0),
        item[0],
    )

    best_class_key = None
    if class_trace_overlaps:
        best_class_key = max(class_trace_overlaps.items(), key=overlap_rank)[0]

    wrong_class_candidates = [
        (class_key, row)
        for class_key, row in class_trace_overlaps.items()
        if class_key != cue_key
    ]
    wrong_class_key = (
        max(wrong_class_candidates, key=overlap_rank)[0]
        if wrong_class_candidates
        else None
    )

    return {
        "probe_tick": probe_tick,
        "trace_hit_count": trace_hit_count,
        "best_score": round(best_score, 4),
        "region_active_counts": region_active_counts,
        "best_class_overlap_key": best_class_key,
        "wrong_class_overlap_key": wrong_class_key,
        "cue_class_overlap": class_trace_overlaps.get(cue_key),
        "trained_class_overlap": class_trace_overlaps.get(trained_key),
        "wrong_class_overlap": (
            class_trace_overlaps.get(wrong_class_key) if wrong_class_key is not None else None
        ),
        "wrong_class_top_region": (
            class_trace_overlaps[wrong_class_key].get("top_region")
            if wrong_class_key is not None and wrong_class_key in class_trace_overlaps
            else None
        ),
        "class_trace_overlaps": class_trace_overlaps,
    }


def _probe_class_overlap_summary(
    probe_breakdown: dict[str, object],
    class_key: str | None,
) -> dict[str, object] | None:
    if class_key is None:
        return None

    row = dict((probe_breakdown.get("class_trace_overlaps", {}) or {}).get(class_key, {}))
    if not row:
        return None

    visual_row = dict((row.get("regions", {}) or {}).get("visual", {}))
    return {
        "class_key": class_key,
        "trace_id": row.get("trace_id"),
        "shared_total": int(row.get("shared_total", 0) or 0),
        "trace_overlap_ratio": float(row.get("trace_overlap_ratio", 0.0) or 0.0),
        "visual_shared_count": int(visual_row.get("shared_count", 0) or 0),
        "visual_trace_overlap_ratio": float(visual_row.get("trace_overlap_ratio", 0.0) or 0.0),
        "visual_active_overlap_ratio": float(visual_row.get("active_overlap_ratio", 0.0) or 0.0),
        "visual_family_overlap": visual_row.get("families"),
    }


def _compare_cue_stage_probes(
    raw_probe: dict[str, object],
    post_tick_probe: dict[str, object],
    *,
    cue_key: str,
    trained_key: str,
) -> dict[str, object]:
    raw_wrong_key = raw_probe.get("wrong_class_overlap_key")
    post_wrong_key = post_tick_probe.get("wrong_class_overlap_key")
    return {
        "mask_scope": "full_image_before_visual_encoding",
        "raw_best_class_overlap_key": raw_probe.get("best_class_overlap_key"),
        "post_tick_best_class_overlap_key": post_tick_probe.get("best_class_overlap_key"),
        "raw_wrong_class_overlap_key": raw_wrong_key,
        "post_tick_wrong_class_overlap_key": post_wrong_key,
        "visual_active_counts": {
            "raw": int((raw_probe.get("region_active_counts", {}) or {}).get("visual", 0) or 0),
            "post_tick": int(
                (post_tick_probe.get("region_active_counts", {}) or {}).get("visual", 0) or 0
            ),
        },
        "cue_class": {
            "raw": _probe_class_overlap_summary(raw_probe, cue_key),
            "post_tick": _probe_class_overlap_summary(post_tick_probe, cue_key),
        },
        "trained_class": {
            "raw": _probe_class_overlap_summary(raw_probe, trained_key),
            "post_tick": _probe_class_overlap_summary(post_tick_probe, trained_key),
        },
        "competing_class": {
            "raw": _probe_class_overlap_summary(
                raw_probe,
                raw_wrong_key if isinstance(raw_wrong_key, str) else None,
            ),
            "post_tick": _probe_class_overlap_summary(
                post_tick_probe,
                post_wrong_key if isinstance(post_wrong_key, str) else None,
            ),
        },
    }


def _region_overlap_breakdown(trace_a: Trace, trace_b: Trace) -> dict[str, dict[str, float | int]]:
    breakdown: dict[str, dict[str, float | int]] = {}
    for region_name in ("visual", "pattern", "integration", "language"):
        neurons_a = set(trace_a.neurons.get(region_name, []))
        neurons_b = set(trace_b.neurons.get(region_name, []))
        shared = len(neurons_a & neurons_b)
        smaller = min(len(neurons_a), len(neurons_b))
        union = len(neurons_a | neurons_b)
        breakdown[region_name] = {
            "count_a": len(neurons_a),
            "count_b": len(neurons_b),
            "shared_count": shared,
            "shared_ratio_of_smaller": round(shared / smaller, 4) if smaller > 0 else 0.0,
            "jaccard": round(shared / union, 4) if union > 0 else 0.0,
        }
        if region_name == "visual":
            breakdown[region_name]["families"] = _visual_family_overlap_breakdown(
                neurons_a,
                neurons_b,
            )
    return breakdown


def _summarize_primary_trace_overlaps(
    primary_traces: dict[str, Trace | None],
) -> dict[str, object]:
    sample_keys = sorted(primary_traces)
    pair_rows: list[dict[str, object]] = []
    max_pattern_shared_ratio = 0.0
    max_visual_shared_ratio = 0.0
    max_visual_family_shared_ratio = {
        family_name: 0.0
        for family_name in VISUAL_TRACE_FEATURE_FAMILIES
    }

    for index, sample_a in enumerate(sample_keys):
        trace_a = primary_traces.get(sample_a)
        if trace_a is None:
            continue
        for sample_b in sample_keys[index + 1:]:
            trace_b = primary_traces.get(sample_b)
            if trace_b is None:
                continue
            regions = _region_overlap_breakdown(trace_a, trace_b)
            max_pattern_shared_ratio = max(
                max_pattern_shared_ratio,
                float(regions["pattern"]["shared_ratio_of_smaller"]),
            )
            max_visual_shared_ratio = max(
                max_visual_shared_ratio,
                float(regions["visual"]["shared_ratio_of_smaller"]),
            )
            for family_name in VISUAL_TRACE_FEATURE_FAMILIES:
                max_visual_family_shared_ratio[family_name] = max(
                    max_visual_family_shared_ratio[family_name],
                    float(regions["visual"]["families"][family_name]["shared_ratio_of_smaller"]),
                )
            pair_rows.append(
                {
                    "sample_a": sample_a,
                    "sample_b": sample_b,
                    "trace_id_a": trace_a.id,
                    "trace_id_b": trace_b.id,
                    "regions": regions,
                }
            )

    return {
        "primary_trace_ids": {
            sample_key: (trace.id if trace is not None else None)
            for sample_key, trace in sorted(primary_traces.items())
        },
        "max_pattern_shared_ratio_of_smaller": round(max_pattern_shared_ratio, 4),
        "max_visual_shared_ratio_of_smaller": round(max_visual_shared_ratio, 4),
        "max_visual_family_shared_ratio_of_smaller": {
            family_name: round(shared_ratio, 4)
            for family_name, shared_ratio in max_visual_family_shared_ratio.items()
        },
        "pairs": pair_rows,
    }


def _select_visual_probe_samples(
    samples: list[dict[str, Any]],
    labels: tuple[str, ...] = DEFAULT_VISUAL_LEARNING_PROBE_LABELS,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    target_labels = list(labels[: max_samples or len(labels)])
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, sample in enumerate(samples):
        label_name = str(sample.get("label_name", ""))
        if label_name not in target_labels or label_name in seen:
            continue
        selected.append(
            {
                "index": index,
                "key": label_name,
                "label_name": label_name,
                "label": sample.get("label"),
                "image": sample["image"],
            }
        )
        seen.add(label_name)
        if len(selected) >= len(target_labels):
            break

    missing = [label for label in target_labels if label not in seen]
    if missing:
        raise ValueError(
            "Missing required visual probe labels in dataset slice: "
            + ", ".join(missing)
        )
    return selected


def _prepare_visual_probe_samples(max_samples: int) -> list[dict[str, Any]]:
    dataset = load_image_dataset(
        "cifar10",
        max_samples=_VISUAL_PROBE_DATASET_SAMPLE_LIMIT,
    )
    return _select_visual_probe_samples(dataset, max_samples=max_samples)


def _cue_mask_seed(salt: str) -> int:
    seed = 0
    for ch in salt.encode("utf-8"):
        seed = ((seed * 131) + ch) & 0xFFFFFFFF
    return seed


def _neutral_fill_value(source: list[list[Any]]) -> Any:
    if not source or not source[0]:
        return 0.0

    example = source[0][0]
    if isinstance(example, list):
        count = 0
        totals = [0.0 for _ in example]
        for row in source:
            for value in row:
                count += 1
                for index, channel in enumerate(value):
                    totals[index] += float(channel)
        return [total / max(1, count) for total in totals]

    if isinstance(example, tuple):
        count = 0
        totals = [0.0 for _ in example]
        for row in source:
            for value in row:
                count += 1
                for index, channel in enumerate(value):
                    totals[index] += float(channel)
        return tuple(total / max(1, count) for total in totals)

    total = 0.0
    count = 0
    for row in source:
        for value in row:
            count += 1
            total += float(value)
    return total / max(1, count)


def _clone_fill_value(fill_value: Any) -> Any:
    if isinstance(fill_value, list):
        return list(fill_value)
    if isinstance(fill_value, tuple):
        return tuple(fill_value)
    return fill_value


def _select_occlusion_patch(
    height: int,
    width: int,
    keep_fraction: float,
    seed: int,
) -> tuple[int, int, int, int]:
    total_cells = height * width
    if total_cells <= 0:
        return 0, 0, 0, 0

    occluded_cells = min(
        total_cells,
        max(1, int(round(total_cells * (1.0 - keep_fraction)))),
    )
    target_aspect = width / max(1, height)
    best_key: tuple[float, ...] | None = None
    candidates: list[tuple[int, int]] = []

    for patch_height in range(1, height + 1):
        estimated_width = int(round(occluded_cells / patch_height))
        width_candidates = {
            max(1, min(width, estimated_width + delta))
            for delta in (-1, 0, 1)
        }
        for patch_width in sorted(width_candidates):
            area = patch_height * patch_width
            aspect = patch_width / max(1, patch_height)
            key = (
                abs(area - occluded_cells),
                0 if area >= occluded_cells else 1,
                abs(aspect - target_aspect),
                abs(patch_height - patch_width),
            )
            if best_key is None or key < best_key:
                best_key = key
                candidates = [(patch_height, patch_width)]
            elif key == best_key:
                candidates.append((patch_height, patch_width))

    if not candidates:
        candidates = [(height, width)]

    patch_height, patch_width = candidates[seed % len(candidates)]
    max_top = max(0, height - patch_height)
    max_left = max(0, width - patch_width)
    top = (seed ^ 0x9E3779B9) % (max_top + 1) if max_top > 0 else 0
    left_seed = ((seed * 1664525) + 1013904223) & 0xFFFFFFFF
    left = left_seed % (max_left + 1) if max_left > 0 else 0
    return top, left, patch_height, patch_width


def _build_visual_partial_cue(
    image: Any,
    keep_fraction: float,
    salt: str,
) -> Any:
    keep_fraction = min(1.0, max(0.0, keep_fraction))
    if keep_fraction >= 1.0:
        if hasattr(image, "shape"):
            return np.array(image, copy=True)
        if hasattr(image, "tolist"):
            return image.tolist()
        return [
            list(row) if isinstance(row, list) else row
            for row in image
        ]

    source = image.tolist() if hasattr(image, "tolist") else image
    if not source:
        return np.array(source, dtype=getattr(image, "dtype", None)) if hasattr(image, "shape") else source

    height = len(source)
    width = len(source[0]) if source and source[0] else 0
    if width == 0:
        return np.array(source, dtype=getattr(image, "dtype", None)) if hasattr(image, "shape") else source

    seed = _cue_mask_seed(salt)
    top, left, patch_height, patch_width = _select_occlusion_patch(
        height,
        width,
        keep_fraction,
        seed,
    )
    fill_value = _neutral_fill_value(source)

    cue: list[list[Any]] = []
    for row_index, row in enumerate(source):
        masked_row: list[Any] = []
        for col_index, value in enumerate(row):
            occluded = (
                top <= row_index < top + patch_height
                and left <= col_index < left + patch_width
            )
            if isinstance(value, list):
                masked_row.append(list(value) if not occluded else _clone_fill_value(fill_value))
            elif isinstance(value, tuple):
                masked_row.append(tuple(value) if not occluded else _clone_fill_value(fill_value))
            else:
                masked_row.append(value if not occluded else fill_value)
        cue.append(masked_row)

    if hasattr(image, "shape"):
        return np.array(cue, dtype=getattr(image, "dtype", None))
    return cue


def _cue_order_key(cue_result: dict[str, object]) -> tuple[int, float, float, float]:
    trace_hit_rate = float(cue_result.get("trace_hit_rate", 0.0) or 0.0)
    best_rank_avg = cue_result.get("best_rank_avg")
    best_score_avg = float(cue_result.get("best_score_avg", 0.0) or 0.0)
    return (
        1 if trace_hit_rate > 0.0 else 0,
        trace_hit_rate,
        -float(best_rank_avg) if best_rank_avg is not None else float("-inf"),
        best_score_avg,
    )


def _summarize_visual_probe_results(results: dict[str, dict[str, object]]) -> dict[str, object]:
    sample_summaries: dict[str, dict[str, object]] = {}
    passed_samples = 0

    for sample_key, sample_result in results.items():
        trace_summary = dict(sample_result.get("learned_trace_summary", {}))
        cue_results = dict(sample_result.get("cue_results", {}))

        best_cue_key = None
        if cue_results and any(
            float(cue_result.get("trace_hit_rate", 0.0) or 0.0) > 0.0
            for cue_result in cue_results.values()
        ):
            best_cue_key = max(cue_results.items(), key=lambda item: _cue_order_key(item[1]))[0]

        target_cue = cue_results.get(sample_key, {})
        off_target_rows = [
            cue_result
            for cue_key, cue_result in cue_results.items()
            if cue_key != sample_key
        ]
        region_totals = dict(trace_summary.get("region_totals", {}))
        visual_pattern_total = int(region_totals.get("visual", 0)) + int(region_totals.get("pattern", 0))
        attention_total = int(region_totals.get("attention", 0))
        memory_short_total = int(region_totals.get("memory_short", 0))

        compact_size_pass = (
            int(trace_summary.get("count", 0)) > 0
            and int(trace_summary.get("size_max", 0)) <= _VISUAL_TRACE_TARGET_SIZE
        )
        target_trace_hit = float(target_cue.get("trace_hit_rate", 0.0) or 0.0) > 0.0
        target_best = best_cue_key == sample_key and target_trace_hit
        off_target_dark = all(
            float(row.get("trace_hit_rate", 0.0) or 0.0) == 0.0
            for row in off_target_rows
        )
        baseline_clear = all(
            int(row.get("baseline_trace_hits", 0) or 0) == 0
            for row in cue_results.values()
        )
        attention_clean = attention_total == 0 and memory_short_total == 0
        visual_pattern_bias = visual_pattern_total > 0 and visual_pattern_total >= attention_total

        sample_pass = (
            compact_size_pass
            and target_best
            and off_target_dark
            and baseline_clear
            and attention_clean
            and visual_pattern_bias
        )
        if sample_pass:
            passed_samples += 1

        sample_summaries[sample_key] = {
            "best_cue_key": best_cue_key,
            "compact_size_pass": compact_size_pass,
            "target_trace_hit": target_trace_hit,
            "target_best": target_best,
            "off_target_dark": off_target_dark,
            "baseline_clear": baseline_clear,
            "attention_clean": attention_clean,
            "visual_pattern_bias": visual_pattern_bias,
            "passes_sample_gate": sample_pass,
        }

    return {
        "sample_summaries": sample_summaries,
        "passed_sample_count": passed_samples,
        "sample_count": len(results),
        "passes_3_of_4_gate": passed_samples >= min(3, len(results)),
    }


def run_visual_learning_probe(
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
) -> dict[str, object]:
    """Train isolated CIFAR images and probe whether their learned traces are selective."""
    if threads > 0:
        try:
            brain_core.set_num_threads(threads)
        except Exception:
            pass
    actual_threads = brain_core.get_num_threads()

    probe_samples = _prepare_visual_probe_samples(max_samples)
    visual_quality_cache = _build_visual_quality_cache(
        probe_samples,
        cue_fraction,
    )
    reference_primary_traces = _collect_primary_visual_reference_traces(
        probe_samples,
        ticks_per_sample=ticks_per_sample,
        train_repeats=train_repeats,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        rest_ticks=rest_ticks,
        visual_quality_cache=visual_quality_cache,
    )
    results: dict[str, dict[str, object]] = {}

    for trained_sample in probe_samples:
        _, trace_store = seed_brain_fast(
            n_traces=n_traces,
            verbose=False,
            chunk_count=seed_chunks,
        )
        trace_store.clear()
        tick_loop = TickLoop(trace_store)
        encoder = VisualInput()
        learned_trace_ids: set[str] = set()
        learned_trace_order: list[str] = []
        quality_entry = visual_quality_cache.get(str(trained_sample["key"]), {})
        tick_loop.trace_formation.set_visual_quality_scores(
            dict(quality_entry.get("quality_scores", {})),
            families=_VISUAL_QUALITY_TARGET_FAMILIES,
        )

        failure_stage_counts: Counter[str] = Counter()
        candidate_region_counts_total: Counter[str] = Counter()
        attempt_ticks = 0
        novelty_gate_ticks = 0
        region_gate_ticks = 0
        ready_pattern_ticks = 0
        candidate_snapshot_total = 0.0
        candidate_snapshot_region_total = 0.0
        training_ticks = 0
        last_training_debug: dict[str, object] = {}
        pattern_fingerprints: list[tuple[int, ...]] = []
        pattern_fingerprint_timeline: list[dict[str, object]] = []

        for _ in range(train_repeats):
            tick_loop.reset_sample_boundary()
            for repeat_tick in range(ticks_per_sample):
                encoder.encode(trained_sample["image"])
                tick_loop.step(allow_binding_formation=False)
                training_ticks += 1
                for trace_id in tick_loop.trace_formation.recently_formed:
                    if trace_id not in learned_trace_ids:
                        learned_trace_order.append(trace_id)
                    learned_trace_ids.add(trace_id)

                debug = tick_loop.trace_formation.last_step_debug
                last_training_debug = dict(debug)
                stage = str(debug.get("failure_stage", "none"))
                failure_stage_counts[stage] += 1
                if bool(debug.get("attempted", False)):
                    attempt_ticks += 1
                if bool(debug.get("passed_novelty_gate", False)):
                    novelty_gate_ticks += 1
                if bool(debug.get("passed_region_gate", False)):
                    region_gate_ticks += 1
                ready_pattern_ticks += int(debug.get("ready_pattern_count", 0) or 0) > 0
                candidate_snapshot_total += float(
                    debug.get("candidate_snapshot_total_neurons", 0.0) or 0.0
                )
                candidate_snapshot_region_total += float(
                    debug.get("candidate_snapshot_region_count", 0.0) or 0.0
                )
                for region_name, count in dict(
                    debug.get("candidate_snapshot_region_counts", {})
                ).items():
                    candidate_region_counts_total[str(region_name)] += int(count)

                formation_snapshot = (
                    tick_loop.trace_formation.last_tracker_snapshot
                    or tick_loop.trace_formation.prepare_snapshot_for_formation(
                        tick_loop.history.current,
                        tick_loop.history,
                        active_traces=tick_loop.last_active_traces,
                        co_trace_ids=tick_loop.working_memory.trace_ids,
                    )
                )
                fingerprint = _formation_fingerprint(formation_snapshot.active_neurons)
                previous = pattern_fingerprints[-1] if pattern_fingerprints else None
                consecutive_jaccard = (
                    _jaccard_ratio(previous, fingerprint) if previous is not None else None
                )
                pattern_fingerprints.append(fingerprint)
                pattern_fingerprint_timeline.append(
                    {
                        "tick": training_ticks,
                        "repeat_tick": repeat_tick + 1,
                        "fingerprint_id": _fingerprint_id(fingerprint),
                        "fingerprint_size": len(fingerprint),
                        "candidate_region_counts": {
                            region_name: len(neurons)
                            for region_name, neurons in sorted(
                                formation_snapshot.active_neurons.items()
                            )
                        },
                        "exact_match_prev": fingerprint == previous if previous is not None else None,
                        "tracker_match_prev": (
                            consecutive_jaccard > 0.5 if consecutive_jaccard is not None else None
                        ),
                        "consecutive_jaccard": (
                            round(consecutive_jaccard, 4)
                            if consecutive_jaccard is not None
                            else None
                        ),
                    }
                )

            for _ in range(rest_ticks):
                tick_loop.step(allow_binding_formation=False)

        for _ in range(settle_ticks):
            tick_loop.step(learn=False, allow_trace_formation=False, allow_binding_formation=False)

        primary_trace_id = learned_trace_order[0] if learned_trace_order else None
        primary_trace = _copy_trace(
            trace_store.get(primary_trace_id) if primary_trace_id is not None else None
        )
        probe_trace_catalog = _build_probe_trace_catalog(
            reference_primary_traces,
            trained_key=str(trained_sample["key"]),
            trained_trace=primary_trace,
        )

        cue_results: dict[str, dict[str, object]] = {}
        for cue_sample in probe_samples:
            tick_loop.reset_probe_boundary()
            for _ in range(settle_ticks):
                tick_loop.step(learn=False, allow_trace_formation=False, allow_binding_formation=False)

            baseline_hits = sum(
                1
                for trace_id, _score in tick_loop.last_active_traces
                if trace_id in learned_trace_ids
            )

            cue_image = _build_visual_partial_cue(
                cue_sample["image"],
                cue_fraction,
                salt=f"{cue_sample['key']}:{cue_fraction:.4f}",
            )
            raw_signal_map = _visual_signal_map(cue_image)
            raw_encoder_probe = _cue_activation_breakdown(
                _activation_snapshot_from_signal_map(raw_signal_map),
                probe_trace_catalog,
                cue_key=str(cue_sample["key"]),
                trained_key=str(trained_sample["key"]),
                probe_tick=0,
                trace_hit_count=0,
                best_score=0.0,
            )
            tick_rows: list[dict[str, object]] = []
            cue_activation_rows: list[dict[str, object]] = []
            cue_encode_counts: list[dict[str, int]] = []
            first_tick_activation_probe: dict[str, object] | None = None
            for probe_tick_index in range(probe_ticks):
                encoded = encoder.encode(cue_image)
                tick_loop.step(
                    learn=False,
                    allow_trace_formation=False,
                    allow_binding_formation=False,
                )
                ranks = {
                    trace_id: (rank, score)
                    for rank, (trace_id, score) in enumerate(tick_loop.last_active_traces, start=1)
                }
                hit_ranks = [
                    ranks[trace_id][0]
                    for trace_id in learned_trace_ids
                    if trace_id in ranks
                ]
                hit_scores = [
                    ranks[trace_id][1]
                    for trace_id in learned_trace_ids
                    if trace_id in ranks
                ]
                tick_rows.append(
                    {
                        "probe_tick": probe_tick_index + 1,
                        "trace_hit_count": len(hit_ranks),
                        "best_rank": min(hit_ranks) if hit_ranks else None,
                        "best_score": round(max(hit_scores), 4) if hit_scores else 0.0,
                        "active_trace_count": len(tick_loop.last_active_traces),
                    }
                )
                cue_activation_rows.append(
                    _cue_activation_breakdown(
                        tick_loop.history.current,
                        probe_trace_catalog,
                        cue_key=str(cue_sample["key"]),
                        trained_key=str(trained_sample["key"]),
                        probe_tick=probe_tick_index + 1,
                        trace_hit_count=len(hit_ranks),
                        best_score=max(hit_scores) if hit_scores else 0.0,
                    )
                )
                if probe_tick_index == 0:
                    first_tick_activation_probe = cue_activation_rows[-1]
                cue_encode_counts.append(
                    {
                        "neurons_activated": int(encoded.get("neurons_activated", 0) or 0),
                        "low_count": int(encoded.get("low_count", 0) or 0),
                        "mid_count": int(encoded.get("mid_count", 0) or 0),
                        "spatial_count": int(encoded.get("spatial_count", 0) or 0),
                    }
                )

            ranked_hits = [
                row["best_rank"]
                for row in tick_rows
                if row["best_rank"] is not None
            ]
            scored_hits = [
                float(row["best_score"])
                for row in tick_rows
                if float(row["best_score"]) > 0.0
            ]
            representative_index = max(
                range(len(tick_rows)),
                key=lambda index: (
                    float(tick_rows[index].get("best_score", 0.0) or 0.0),
                    int(tick_rows[index].get("trace_hit_count", 0) or 0),
                    -int(tick_rows[index].get("probe_tick", 0) or 0),
                ),
            )
            first_tick_activation_probe = first_tick_activation_probe or (
                cue_activation_rows[0] if cue_activation_rows else None
            )
            cue_results[str(cue_sample["key"])] = {
                "cue_label": cue_sample["label_name"],
                "baseline_trace_hits": baseline_hits,
                "trace_hit_rate": round(
                    sum(1 for row in tick_rows if int(row["trace_hit_count"]) > 0) / len(tick_rows),
                    4,
                ),
                "best_rank_avg": round(sum(ranked_hits) / len(ranked_hits), 3)
                if ranked_hits
                else None,
                "best_score_avg": round(sum(scored_hits) / len(scored_hits), 4)
                if scored_hits
                else 0.0,
                "cue_neurons_activated_avg": round(
                    sum(row["neurons_activated"] for row in cue_encode_counts) / len(cue_encode_counts),
                    3,
                ),
                "cue_stage_diagnostic": {
                    "mask_scope": "full_image_before_visual_encoding",
                    "raw_visual_signal_counts": _visual_signal_family_counts(raw_signal_map),
                    "raw_encoder_probe": raw_encoder_probe,
                    "first_tick_activation_probe": first_tick_activation_probe,
                    "comparison": (
                        _compare_cue_stage_probes(
                            raw_encoder_probe,
                            first_tick_activation_probe,
                            cue_key=str(cue_sample["key"]),
                            trained_key=str(trained_sample["key"]),
                        )
                        if first_tick_activation_probe is not None
                        else None
                    ),
                },
                "cue_activation_probe": cue_activation_rows[representative_index],
                "ticks": tick_rows,
            }

            for _ in range(settle_ticks):
                tick_loop.step(learn=False, allow_trace_formation=False, allow_binding_formation=False)

        results[str(trained_sample["key"])] = {
            "trained_label": trained_sample["label_name"],
            "reference_index": int(trained_sample["index"]),
            "learned_trace_summary": _summarize_learned_traces(
                trace_store,
                learned_trace_ids,
                formation_order=learned_trace_order,
            ),
            "training_diagnostics": {
                "attempt_ticks": attempt_ticks,
                "passed_novelty_gate_ticks": novelty_gate_ticks,
                "passed_region_gate_ticks": region_gate_ticks,
                "ready_pattern_ticks": ready_pattern_ticks,
                "visual_quality_summary": quality_entry.get("families", {}),
                "failure_stage_counts": dict(sorted(failure_stage_counts.items())),
                "candidate_snapshot_total_neurons_avg": round(
                    candidate_snapshot_total / max(1, training_ticks),
                    3,
                ),
                "candidate_snapshot_region_count_avg": round(
                    candidate_snapshot_region_total / max(1, training_ticks),
                    3,
                ),
                "candidate_snapshot_region_counts_avg": {
                    region_name: round(total / max(1, training_ticks), 3)
                    for region_name, total in sorted(candidate_region_counts_total.items())
                },
                "pattern_fingerprint_summary": _summarize_fingerprint_sequence(
                    pattern_fingerprints,
                ),
                "pattern_fingerprint_timeline": pattern_fingerprint_timeline,
                "last_step_debug": last_training_debug,
            },
            "cue_results": cue_results,
        }

    summary = _summarize_visual_probe_results(results)
    overlap_summary = _summarize_primary_trace_overlaps(reference_primary_traces)
    for sample_key, sample_summary in summary["sample_summaries"].items():
        results[sample_key]["sample_summary"] = sample_summary

    output = {
        "benchmark": "visual_learning_probe",
        "dataset": "cifar10",
        "threads": actual_threads,
        "seed_traces": n_traces,
        "seed_chunks": seed_chunks,
        "ticks_per_sample": ticks_per_sample,
        "train_repeats": train_repeats,
        "rest_ticks": rest_ticks,
        "settle_ticks": settle_ticks,
        "probe_ticks": probe_ticks,
        "cue_fraction": cue_fraction,
        "cue_mask_mode": "contiguous_mean_patch",
        "reference_labels": list(DEFAULT_VISUAL_LEARNING_PROBE_LABELS[:max_samples]),
        "probe_mode": "isolated_single_image",
        "trace_store_mode": "learned_only",
        "summary": summary,
        "primary_trace_overlap": overlap_summary,
        "results": results,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print("\nVISUAL LEARNING PROBE SUMMARY")
    for sample_key, sample_result in results.items():
        trace_summary = sample_result["learned_trace_summary"]
        target_cue = sample_result["cue_results"].get(sample_key, {})
        sample_summary = sample_result["sample_summary"]
        print(
            f"  {sample_key}: traces={trace_summary['count']}, "
            f"size_avg={trace_summary['size_avg']}, size_max={trace_summary['size_max']}, "
            f"target_hit={target_cue.get('trace_hit_rate', 0.0):.2f}, "
            f"target_rank={target_cue.get('best_rank_avg')}, "
            f"best_cue={sample_summary.get('best_cue_key')}, "
            f"pass={sample_summary.get('passes_sample_gate')}"
        )
    print(
        f"  3-of-4 gate: {summary['passed_sample_count']}/{summary['sample_count']} "
        f"-> {summary['passes_3_of_4_gate']}"
    )
    print(f"Probe metrics saved to: {path}")
    return output