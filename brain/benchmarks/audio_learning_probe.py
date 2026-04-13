"""Reproducible audio-learning probe for trace selectivity and recall quality."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import brain_core

from brain.datasets.downloader import load_audio_dataset
from brain.input.audio_input import AudioInput
from brain.learning.tick_loop import TickLoop
from brain.seed.seed_runner import seed_brain_fast
from brain.serialize.runtime_bundle import load_runtime_bundle
from brain.structures.brain_state import ActivationSnapshot
from brain.structures.neuron_map import region_for_neuron
from brain.structures.trace_store import Trace, TraceStore


DEFAULT_AUDIO_LEARNING_PROBE_LABELS = (
    "airplane",
    "crow",
    "door_wood_knock",
    "vacuum_cleaner",
)

_AUDIO_PROBE_DATASET_SAMPLE_LIMIT = 2000
_AUDIO_TRACE_TARGET_SIZE = 48
_AUDIO_STABILIZATION_TICKS = 3
_AUDIO_FROZEN_REPLAY_TICKS = 3
_AUDIO_FORMATION_NOVELTY_THRESHOLD = 0.1
_AUDIO_TRACE_AUDIO_NEURONS = 12
_AUDIO_INTERNAL_FRAME_FLOOR = 12
_AUDIO_FAMILY_RANGES = {
    "frequency": (30000, 34999),
    "temporal": (35000, 39999),
    "complex": (40000, 44999),
}
_AUDIO_START = _AUDIO_FAMILY_RANGES["frequency"][0]
_AUDIO_TEMPORAL_START = _AUDIO_FAMILY_RANGES["temporal"][0] - _AUDIO_START
_AUDIO_QUALITY_MASK_TRIALS = 64
_AUDIO_QUALITY_TARGET_FAMILIES = tuple(_AUDIO_FAMILY_RANGES)
_AUDIO_PREPARED_CUE_CANDIDATE_TRIALS = _AUDIO_QUALITY_MASK_TRIALS
_AUDIO_PROFILE_KEYS = (
    "rust_tick_ms",
    "evaluation_ms",
    "binding_recall_ms",
    "other_python_ms",
    "step_internal_ms",
    "total_active",
)


def _to_mono_float_list(audio: Any) -> list[float]:
    if hasattr(audio, "tolist"):
        audio = audio.tolist()
    if not audio:
        return []
    first = audio[0]
    if isinstance(first, (list, tuple)):
        return [
            float(sum(frame) / max(1, len(frame)))
            for frame in audio
        ]
    return [float(sample) for sample in audio]


def _split_audio_frames(audio: Any, frame_count: int) -> list[list[float]]:
    samples = _to_mono_float_list(audio)
    if not samples:
        return [[]]
    frame_count = max(1, frame_count)
    frame_len = max(1, len(samples) // frame_count)
    frames: list[list[float]] = []
    for frame_index in range(frame_count):
        start = frame_index * frame_len
        end = len(samples) if frame_index == frame_count - 1 else (frame_index + 1) * frame_len
        frames.append(samples[start:end])
    return frames


def _build_audio_partial_cue_frames(
    frames: list[list[float]],
    cue_fraction: float,
    *,
    salt: str = "probe",
) -> list[list[float]]:
    return _build_audio_quality_masked_frames(frames, cue_fraction, salt=salt)


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
    target_size: int = _AUDIO_TRACE_TARGET_SIZE,
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
        audio_ids = sorted(trace.neurons.get("audio", []))
        region_counts = {
            region_name: len(neurons)
            for region_name, neurons in sorted(trace.neurons.items())
        }
        for region_name, count in region_counts.items():
            region_totals[region_name] += count
        rows.append(
            {
                "trace_id": trace_id,
                "dominant_region": _dominant_region(trace),
                "total_neurons": trace.total_neurons(),
                "audio_ids": audio_ids,
                "audio_family_counts": _audio_family_counts_from_ids(audio_ids),
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
        "primary_audio_ids": list(rows[0]["audio_ids"]) if rows else [],
        "primary_audio_family_counts": dict(rows[0]["audio_family_counts"]) if rows else {},
        "primary_has_complex_family": bool(
            rows and rows[0]["audio_family_counts"].get("complex", 0) > 0
        ),
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
        if region_name == "audio":
            region_breakdown[region_name]["families"] = _audio_family_overlap_breakdown(
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


def _summarize_audio_probe_results(results: dict[str, dict[str, object]]) -> dict[str, object]:
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

        compact_size_pass = (
            int(trace_summary.get("count", 0)) > 0
            and int(trace_summary.get("size_max", 0)) <= _AUDIO_TRACE_TARGET_SIZE
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
        audio_present = int(region_totals.get("audio", 0)) > 0

        sample_pass = (
            compact_size_pass
            and target_best
            and off_target_dark
            and baseline_clear
            and audio_present
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
            "audio_present": audio_present,
            "passes_sample_gate": sample_pass,
        }

    return {
        "sample_summaries": sample_summaries,
        "passed_sample_count": passed_samples,
        "sample_count": len(results),
        "passes_3_of_4_gate": passed_samples >= min(3, len(results)),
    }


def _select_audio_probe_samples(
    samples: list[dict[str, Any]],
    labels: tuple[str, ...] = DEFAULT_AUDIO_LEARNING_PROBE_LABELS,
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
                "audio": sample["audio"],
                "sample_rate": int(sample.get("sample_rate", 16000) or 16000),
            }
        )
        seen.add(label_name)
        if len(selected) >= len(target_labels):
            break

    missing = [label for label in target_labels if label not in seen]
    if missing:
        raise ValueError(
            "Missing required audio probe labels in dataset slice: " + ", ".join(missing)
        )
    return selected


def _prepare_audio_probe_samples(max_samples: int) -> list[dict[str, Any]]:
    dataset = load_audio_dataset(
        "esc50",
        max_samples=_AUDIO_PROBE_DATASET_SAMPLE_LIMIT,
    )
    return _select_audio_probe_samples(dataset, max_samples=max_samples)


def _run_audio_sequence(
    tick_loop: TickLoop,
    encoder: AudioInput,
    frames: list[list[float]],
    sample_rate: int,
    *,
    learn: bool,
    allow_trace_formation: bool,
    allow_binding_formation: bool,
) -> list[dict[str, int]]:
    frame_counts: list[dict[str, int]] = []
    for frame in frames:
        encoded = encoder.encode(frame, sample_rate)
        tick_loop.step(
            learn=learn,
            allow_trace_formation=allow_trace_formation,
            allow_binding_formation=allow_binding_formation,
        )
        frame_counts.append(
            {
                "neurons_activated": int(encoded.get("neurons_activated", 0) or 0),
                "freq_count": int(encoded.get("freq_count", 0) or 0),
                "temporal_count": int(encoded.get("temporal_count", 0) or 0),
                "complex_count": int(encoded.get("complex_count", 0) or 0),
                "total_signals": int(encoded.get("total_signals", 0) or 0),
            }
        )
    return frame_counts


def _create_audio_probe_tick_loop(
    *,
    n_traces: int,
    seed_chunks: int | None,
    initial_state_dir: str | None = None,
) -> tuple[TickLoop, TraceStore]:
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
    tick_loop.trace_formation.set_visual_candidate_lock_enabled(False)
    return tick_loop, trace_store


def _formation_snapshot_region_counts(snapshot: ActivationSnapshot) -> dict[str, int]:
    return {
        region_name: len(neurons)
        for region_name, neurons in sorted(snapshot.active_neurons.items())
        if neurons
    }


def _audio_family_for_neuron(neuron_id: int) -> str | None:
    for family_name, (start, end) in _AUDIO_FAMILY_RANGES.items():
        if start <= neuron_id <= end:
            return family_name
    return None


def _audio_family_counts_from_ids(neuron_ids: set[int] | list[int]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for neuron_id in neuron_ids:
        family_name = _audio_family_for_neuron(neuron_id)
        if family_name is not None:
            counts[family_name] += 1
    return dict(sorted(counts.items()))


def _audio_internal_frame_count(ticks_per_sample: int) -> int:
    return max(int(ticks_per_sample), _AUDIO_INTERNAL_FRAME_FLOOR)


def _empty_audio_frame_encoding() -> dict[str, object]:
    return {
        "neurons": [],
        "signal_map": {},
        "freq_count": 0,
        "temporal_count": 0,
        "complex_count": 0,
        "total_signals": 0,
    }


def _audio_energy_signals(energy: float) -> list[tuple[int, float]]:
    signals: list[tuple[int, float]] = []
    energy_norm = min(1.0, energy * 5.0)
    if energy_norm <= 0.01:
        return signals

    center = int(energy_norm * 2499)
    spread = 30
    for offset in range(-spread * 3, spread * 3 + 1):
        gid = _AUDIO_START + _AUDIO_TEMPORAL_START + max(0, min(center + offset, 2499))
        dist = abs(offset)
        act = math.exp(-0.5 * dist * dist / (spread * spread))
        if act > 0.01:
            signals.append((gid, act * energy_norm))
    return signals


def _audio_onset_signals(onset_strength: float) -> list[tuple[int, float]]:
    signals: list[tuple[int, float]] = []
    if onset_strength <= 0.01:
        return signals

    onset_norm = min(1.0, onset_strength * 10.0)
    onset_center = 2500 + int(onset_norm * 2499)
    for offset in range(-20, 21):
        gid = _AUDIO_START + _AUDIO_TEMPORAL_START + max(2500, min(onset_center + offset, 4999))
        dist = abs(offset)
        act = math.exp(-0.5 * dist * dist / (15.0 * 15.0))
        if act > 0.01:
            signals.append((gid, act * onset_norm))
    return signals


def _merge_audio_signals(
    signal_groups: list[list[tuple[int, float]]],
) -> tuple[list[int], dict[int, float]]:
    neurons: list[int] = []
    signal_map: dict[int, float] = {}

    for signals in signal_groups:
        for neuron_id, activation in signals:
            neurons.append(neuron_id)
            previous = signal_map.get(neuron_id)
            if previous is None or activation > previous:
                signal_map[neuron_id] = float(activation)

    return neurons, signal_map


def _audio_frame_components(
    frame: list[float],
    sample_rate: int,
    *,
    encoder: AudioInput | None = None,
) -> dict[str, object]:
    if not frame:
        return {
            "energy": 0.0,
            "freq_signals": [],
            "complex_signals": [],
            "energy_signals": [],
        }

    encoder = encoder or AudioInput()
    norm = encoder._normalize(frame)
    energy = sum(sample * sample for sample in norm) / len(norm)
    return {
        "energy": energy,
        "freq_signals": encoder._extract_frequency(norm, sample_rate),
        "complex_signals": encoder._extract_complex(norm, sample_rate),
        "energy_signals": _audio_energy_signals(energy),
    }


def _build_audio_frame_encoding(
    frame_components: dict[str, object],
    *,
    prev_energy: float,
) -> dict[str, object]:
    freq_signals = list(frame_components.get("freq_signals", []))
    complex_signals = list(frame_components.get("complex_signals", []))
    energy_signals = list(frame_components.get("energy_signals", []))
    energy = float(frame_components.get("energy", 0.0) or 0.0)
    onset_strength = max(0.0, energy - prev_energy * 1.5)
    onset_signals = _audio_onset_signals(onset_strength)
    temporal_signals = energy_signals + onset_signals
    neurons, signal_map = _merge_audio_signals(
        [freq_signals, temporal_signals, complex_signals]
    )
    return {
        "neurons": neurons,
        "signal_map": signal_map,
        "freq_count": len(freq_signals),
        "temporal_count": len(temporal_signals),
        "complex_count": len(complex_signals),
        "total_signals": len(neurons),
    }


def _precompute_audio_frame_components(
    frames: list[list[float]],
    sample_rate: int,
) -> list[dict[str, object]]:
    encoder = AudioInput()
    return [
        _audio_frame_components(frame, sample_rate, encoder=encoder)
        for frame in frames
    ]


def _build_audio_frame_encodings(
    frame_components: list[dict[str, object]],
    *,
    keep_indices: set[int] | None = None,
) -> list[dict[str, object]]:
    encodings: list[dict[str, object]] = []
    prev_energy = 0.0

    for index, components in enumerate(frame_components):
        if keep_indices is not None and index not in keep_indices:
            prev_energy = 0.0
            encodings.append(_empty_audio_frame_encoding())
            continue

        encodings.append(
            _build_audio_frame_encoding(components, prev_energy=prev_energy)
        )
        prev_energy = float(components.get("energy", 0.0) or 0.0)

    return encodings


def _audio_sequence_signal_map_from_encodings(
    frame_encodings: list[dict[str, object]],
) -> dict[int, float]:
    signal_map: dict[int, float] = {}
    for frame_encoding in frame_encodings:
        for neuron_id, activation in dict(frame_encoding.get("signal_map", {})).items():
            previous = signal_map.get(neuron_id)
            if previous is None or activation > previous:
                signal_map[neuron_id] = float(activation)
    return signal_map


def _inject_audio_frame_encoding(
    frame_encoding: dict[str, object],
    *,
    boost: float = 0.7,
) -> dict[str, int]:
    neurons = list(frame_encoding.get("neurons", []))
    activated = brain_core.boost_audio(neurons, boost) if neurons else 0
    return {
        "neurons_activated": int(activated),
        "freq_count": int(frame_encoding.get("freq_count", 0) or 0),
        "temporal_count": int(frame_encoding.get("temporal_count", 0) or 0),
        "complex_count": int(frame_encoding.get("complex_count", 0) or 0),
        "total_signals": int(frame_encoding.get("total_signals", 0) or 0),
    }


def _merge_audio_frame_encodings(
    frame_encodings: list[dict[str, object]],
) -> dict[str, object]:
    if not frame_encodings:
        return _empty_audio_frame_encoding()

    neurons: list[int] = []
    signal_map: dict[int, float] = {}
    freq_count = 0
    temporal_count = 0
    complex_count = 0
    total_signals = 0

    for frame_encoding in frame_encodings:
        neurons.extend(list(frame_encoding.get("neurons", [])))
        freq_count += int(frame_encoding.get("freq_count", 0) or 0)
        temporal_count += int(frame_encoding.get("temporal_count", 0) or 0)
        complex_count += int(frame_encoding.get("complex_count", 0) or 0)
        total_signals += int(frame_encoding.get("total_signals", 0) or 0)
        for neuron_id, activation in dict(frame_encoding.get("signal_map", {})).items():
            previous = signal_map.get(neuron_id)
            if previous is None or activation > previous:
                signal_map[neuron_id] = float(activation)

    return {
        "neurons": neurons,
        "signal_map": signal_map,
        "freq_count": freq_count,
        "temporal_count": temporal_count,
        "complex_count": complex_count,
        "total_signals": total_signals,
    }


def _group_audio_frame_encodings(
    frame_encodings: list[dict[str, object]],
    target_count: int,
) -> list[dict[str, object]]:
    if target_count <= 0:
        return []
    if not frame_encodings:
        return []
    if len(frame_encodings) <= target_count:
        return list(frame_encodings)

    grouped: list[dict[str, object]] = []
    total = len(frame_encodings)
    for group_index in range(target_count):
        start = math.floor(group_index * total / target_count)
        end = math.floor((group_index + 1) * total / target_count)
        if end <= start:
            end = min(total, start + 1)
        grouped.append(_merge_audio_frame_encodings(frame_encodings[start:end]))
    return grouped


def _select_audio_quality_keep_indices(
    frame_count: int,
    cue_fraction: float,
    *,
    salt: str,
) -> set[int]:
    if frame_count <= 0:
        return set()

    cue_fraction = min(1.0, max(0.0, cue_fraction))
    keep = max(1, min(frame_count, int(math.ceil(frame_count * cue_fraction))))
    if keep >= frame_count:
        return set(range(frame_count))

    ranked_indices = sorted(
        range(frame_count),
        key=lambda index: hashlib.sha1(f"{salt}:{index}".encode("ascii")).hexdigest(),
    )
    return set(ranked_indices[:keep])


def _audio_signal_map(
    encoder: AudioInput,
    frame: list[float],
    sample_rate: int,
) -> dict[int, float]:
    frame_components = _audio_frame_components(
        frame,
        sample_rate,
        encoder=AudioInput(boost=encoder.boost, spread=encoder.spread),
    )
    frame_encoding = _build_audio_frame_encoding(
        frame_components,
        prev_energy=encoder._prev_energy,
    )
    return dict(frame_encoding["signal_map"])


def _audio_sequence_signal_map(
    frames: list[list[float]],
    sample_rate: int,
) -> dict[int, float]:
    frame_components = _precompute_audio_frame_components(frames, sample_rate)
    frame_encodings = _build_audio_frame_encodings(frame_components)
    return _audio_sequence_signal_map_from_encodings(frame_encodings)


def _build_audio_quality_masked_frames(
    frames: list[list[float]],
    cue_fraction: float,
    *,
    salt: str,
) -> list[list[float]]:
    if not frames:
        return []

    keep_indices = _select_audio_quality_keep_indices(
        len(frames),
        cue_fraction,
        salt=salt,
    )
    if len(keep_indices) >= len(frames):
        return [list(frame) for frame in frames]
    return [
        list(frame) if index in keep_indices else [0.0 for _ in frame]
        for index, frame in enumerate(frames)
    ]


def _build_audio_quality_cache(
    probe_samples: list[dict[str, Any]],
    ticks_per_sample: int,
    cue_fraction: float,
    *,
    mask_trials: int = _AUDIO_QUALITY_MASK_TRIALS,
    target_families: tuple[str, ...] = _AUDIO_QUALITY_TARGET_FAMILIES,
) -> dict[str, dict[str, object]]:
    brain_core.init_brain()
    internal_frame_count = _audio_internal_frame_count(ticks_per_sample)
    frame_sequences = {
        str(sample["key"]): _split_audio_frames(sample["audio"], internal_frame_count)
        for sample in probe_samples
    }
    frame_components = {
        str(sample["key"]): _precompute_audio_frame_components(
            frame_sequences[str(sample["key"])],
            int(sample["sample_rate"]),
        )
        for sample in probe_samples
    }
    frame_encodings = {
        sample_key: _build_audio_frame_encodings(sample_components)
        for sample_key, sample_components in frame_components.items()
    }
    raw_signal_maps = {
        str(sample["key"]): _audio_sequence_signal_map_from_encodings(
            frame_encodings[str(sample["key"])]
        )
        for sample in probe_samples
    }

    quality_cache: dict[str, dict[str, object]] = {}
    for trained_sample in probe_samples:
        trained_key = str(trained_sample["key"])
        trained_signals = raw_signal_maps[trained_key]
        trained_frame_components = frame_components[trained_key]
        other_class_counts: Counter[int] = Counter()
        for sample_key, activation_map in raw_signal_maps.items():
            if sample_key == trained_key:
                continue
            for neuron_id in activation_map:
                other_class_counts[neuron_id] += 1

        survival_counts: Counter[int] = Counter()
        for trial_index in range(mask_trials):
            keep_indices = _select_audio_quality_keep_indices(
                len(trained_frame_components),
                cue_fraction,
                salt=f"{trained_key}:quality:{trial_index}",
            )
            masked_frame_encodings = _build_audio_frame_encodings(
                trained_frame_components,
                keep_indices=keep_indices,
            )
            for neuron_id in _audio_sequence_signal_map_from_encodings(masked_frame_encodings):
                survival_counts[neuron_id] += 1

        quality_scores: dict[int, float] = {}
        family_summary: dict[str, dict[str, object]] = {}
        for family_name in target_families:
            family_ids = {
                neuron_id
                for neuron_id in trained_signals
                if _audio_family_for_neuron(neuron_id) == family_name
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
    signal_map: dict[int, float],
) -> ActivationSnapshot:
    active_neurons: dict[str, list[tuple[int, float]]] = {}
    for neuron_id, activation in sorted(signal_map.items()):
        region_name = region_for_neuron(neuron_id)
        if region_name is None:
            continue
        active_neurons.setdefault(region_name, []).append((neuron_id, activation))
    active_values = [
        (neuron_id, activation)
        for neurons in active_neurons.values()
        for neuron_id, activation in neurons
    ]
    active_ids = [neuron_id for neuron_id, _activation in active_values]
    return ActivationSnapshot(
        tick=0,
        active_neurons=active_neurons,
        active_values=active_values,
        total_active=len(active_ids),
        active_ids=active_ids,
        region_active_counts={
            region_name: len(neurons)
            for region_name, neurons in active_neurons.items()
        },
    )


def _audio_family_overlap_breakdown(
    ids_a: set[int],
    ids_b: set[int],
) -> dict[str, dict[str, float | int]]:
    breakdown: dict[str, dict[str, float | int]] = {}
    for family_name in _AUDIO_FAMILY_RANGES:
        family_a = {
            neuron_id
            for neuron_id in ids_a
            if _audio_family_for_neuron(neuron_id) == family_name
        }
        family_b = {
            neuron_id
            for neuron_id in ids_b
            if _audio_family_for_neuron(neuron_id) == family_name
        }
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


def _append_step_profile_row(
    step_profile_rows: list[dict[str, float]],
    step_result: dict[str, object],
) -> None:
    step_profile_rows.append(
        {
            key: float(step_result.get(key, 0.0) or 0.0)
            for key in _AUDIO_PROFILE_KEYS
        }
    )


def _summarize_step_profile_rows(
    step_profile_rows: list[dict[str, float]],
) -> dict[str, float | int]:
    if not step_profile_rows:
        return {
            "tick_count": 0,
            "rust_tick_ms_avg": 0.0,
            "evaluation_ms_avg": 0.0,
            "eval_ms_avg": 0.0,
            "binding_recall_ms_avg": 0.0,
            "other_python_ms_avg": 0.0,
            "step_internal_ms_avg": 0.0,
            "total_active_avg": 0.0,
        }
    tick_count = len(step_profile_rows)
    averages = {
        f"{key}_avg": round(
            sum(row.get(key, 0.0) for row in step_profile_rows) / tick_count,
            4,
        )
        for key in _AUDIO_PROFILE_KEYS
    }
    averages["tick_count"] = tick_count
    averages["eval_ms_avg"] = averages.get("evaluation_ms_avg", 0.0)
    return averages


def _candidate_snapshot_row(
    snapshot: ActivationSnapshot,
    *,
    phase: str,
) -> dict[str, object]:
    audio_ids = [
        neuron_id
        for neuron_id, _activation in snapshot.active_neurons.get("audio", [])
    ]
    return {
        "phase": phase,
        "total_neurons": int(snapshot.total_active),
        "region_count": len(snapshot.active_neurons),
        "audio_family_counts": _audio_family_counts_from_ids(audio_ids),
    }


def _summarize_candidate_snapshot_rows(
    candidate_snapshot_rows: list[dict[str, object]],
) -> dict[str, object]:
    if not candidate_snapshot_rows:
        return {
            "candidate_count": 0,
            "candidate_neurons_avg": 0.0,
            "candidate_neurons_max": 0,
            "candidate_region_count_avg": 0.0,
            "audio_family_counts_avg": {},
            "by_phase": {},
        }

    phase_groups: dict[str, list[dict[str, object]]] = {}
    family_totals: Counter[str] = Counter()
    for row in candidate_snapshot_rows:
        phase_groups.setdefault(str(row["phase"]), []).append(row)
        family_totals.update(dict(row.get("audio_family_counts", {})))

    def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "candidate_count": len(rows),
            "candidate_neurons_avg": round(
                sum(int(row.get("total_neurons", 0) or 0) for row in rows) / len(rows),
                4,
            ),
            "candidate_neurons_max": max(
                int(row.get("total_neurons", 0) or 0) for row in rows
            ),
            "candidate_region_count_avg": round(
                sum(int(row.get("region_count", 0) or 0) for row in rows) / len(rows),
                4,
            ),
        }

    summary = summarize_rows(candidate_snapshot_rows)
    summary["audio_family_counts_avg"] = {
        family_name: round(total / len(candidate_snapshot_rows), 4)
        for family_name, total in sorted(family_totals.items())
    }
    summary["by_phase"] = {
        phase_name: summarize_rows(rows)
        for phase_name, rows in sorted(phase_groups.items())
    }
    return summary


def _audio_probe_class_overlap_summary(
    probe_breakdown: dict[str, object],
    class_key: str | None,
) -> dict[str, object] | None:
    if class_key is None:
        return None

    row = dict((probe_breakdown.get("class_trace_overlaps", {}) or {}).get(class_key, {}))
    if not row:
        return None

    audio_row = dict((row.get("regions", {}) or {}).get("audio", {}))
    return {
        "class_key": class_key,
        "trace_id": row.get("trace_id"),
        "shared_total": int(row.get("shared_total", 0) or 0),
        "trace_overlap_ratio": float(row.get("trace_overlap_ratio", 0.0) or 0.0),
        "audio_shared_count": int(audio_row.get("shared_count", 0) or 0),
        "audio_trace_overlap_ratio": float(audio_row.get("trace_overlap_ratio", 0.0) or 0.0),
        "audio_active_overlap_ratio": float(audio_row.get("active_overlap_ratio", 0.0) or 0.0),
        "audio_family_overlap": audio_row.get("families"),
    }


def _compare_audio_cue_stage_probes(
    raw_probe: dict[str, object],
    post_tick_probe: dict[str, object],
    *,
    cue_key: str,
    trained_key: str,
) -> dict[str, object]:
    raw_wrong_key = raw_probe.get("wrong_class_overlap_key")
    post_wrong_key = post_tick_probe.get("wrong_class_overlap_key")
    return {
        "mask_scope": "raw_first_cue_frame_vs_first_probe_tick",
        "raw_best_class_overlap_key": raw_probe.get("best_class_overlap_key"),
        "post_tick_best_class_overlap_key": post_tick_probe.get("best_class_overlap_key"),
        "raw_wrong_class_overlap_key": raw_wrong_key,
        "post_tick_wrong_class_overlap_key": post_wrong_key,
        "audio_active_counts": {
            "raw": int((raw_probe.get("region_active_counts", {}) or {}).get("audio", 0) or 0),
            "post_tick": int(
                (post_tick_probe.get("region_active_counts", {}) or {}).get("audio", 0) or 0
            ),
        },
        "cue_class": {
            "raw": _audio_probe_class_overlap_summary(raw_probe, cue_key),
            "post_tick": _audio_probe_class_overlap_summary(post_tick_probe, cue_key),
        },
        "trained_class": {
            "raw": _audio_probe_class_overlap_summary(raw_probe, trained_key),
            "post_tick": _audio_probe_class_overlap_summary(post_tick_probe, trained_key),
        },
        "competing_class": {
            "raw": _audio_probe_class_overlap_summary(
                raw_probe,
                raw_wrong_key if isinstance(raw_wrong_key, str) else None,
            ),
            "post_tick": _audio_probe_class_overlap_summary(
                post_tick_probe,
                post_wrong_key if isinstance(post_wrong_key, str) else None,
            ),
        },
    }


def _capture_audio_formation_window(
    tick_loop: TickLoop,
    *,
    silent_ticks: int,
) -> list[dict[str, object]]:
    capture_rows: list[dict[str, object]] = []
    for silent_tick in range(1, silent_ticks + 1):
        result = tick_loop.step(
            learn=False,
            allow_trace_formation=False,
            allow_binding_formation=False,
        )
        snapshot = tick_loop.trace_formation.prepare_snapshot_for_formation(
            tick_loop.history.current,
            tick_loop.history,
            active_traces=tick_loop.last_active_traces,
            co_trace_ids=tick_loop.working_memory.trace_ids,
        )
        capture_rows.append(
            {
                "phase_step": silent_tick,
                "novelty": float(result.get("novelty", 0.0)),
                "step_result": result,
                "snapshot": snapshot,
                "region_counts": _formation_snapshot_region_counts(snapshot),
            }
        )
    return capture_rows


def _select_audio_formation_snapshot(
    capture_rows: list[dict[str, object]],
) -> tuple[ActivationSnapshot | None, float]:
    qualifying_rows = [
        row
        for row in capture_rows
        if float(row.get("novelty", 0.0) or 0.0) >= _AUDIO_FORMATION_NOVELTY_THRESHOLD
        and len(dict(row.get("region_counts", {}))) >= 2
        and isinstance(row.get("snapshot"), ActivationSnapshot)
        and bool(row["snapshot"].active_neurons)
    ]
    if qualifying_rows:
        chosen = qualifying_rows[-1]
        return chosen["snapshot"], float(chosen.get("novelty", 0.0) or 0.0)
    if not capture_rows:
        return None, 0.0
    chosen = max(
        capture_rows,
        key=lambda row: (
            float(row.get("novelty", 0.0) or 0.0),
            len(dict(row.get("region_counts", {}))),
            int(getattr(row.get("snapshot"), "total_active", 0) or 0),
            int(row.get("phase_step", 0) or 0),
        ),
    )
    snapshot = chosen.get("snapshot")
    if not isinstance(snapshot, ActivationSnapshot):
        return None, 0.0
    return snapshot, float(chosen.get("novelty", 0.0) or 0.0)


def _build_prepared_audio_cue_snapshot(
    sample: dict[str, Any],
    *,
    ticks_per_sample: int,
    n_traces: int,
    seed_chunks: int | None,
    audio_quality_entry: dict[str, object] | None = None,
    frame_encodings: list[dict[str, object]] | None = None,
) -> tuple[ActivationSnapshot | None, dict[str, object]]:
    tick_loop, _trace_store = _create_audio_probe_tick_loop(
        n_traces=n_traces,
        seed_chunks=seed_chunks,
    )
    tick_loop.trace_formation.set_audio_quality_scores(
        dict((audio_quality_entry or {}).get("quality_scores", {})),
        families=_AUDIO_QUALITY_TARGET_FAMILIES,
    )
    frame_encodings = frame_encodings or _build_audio_frame_encodings(
        _precompute_audio_frame_components(
            _split_audio_frames(sample["audio"], ticks_per_sample),
            int(sample["sample_rate"]),
        )
    )

    frame_snapshots = []
    for frame_encoding in frame_encodings:
        _inject_audio_frame_encoding(frame_encoding)
        tick_loop.step(
            allow_trace_formation=False,
            allow_binding_formation=False,
        )
        frame_snapshots.append(
            tick_loop.trace_formation.prepare_snapshot_for_formation(
                tick_loop.history.current,
                tick_loop.history,
                active_traces=tick_loop.last_active_traces,
                co_trace_ids=tick_loop.working_memory.trace_ids,
            )
        )

    capture_rows = _capture_audio_formation_window(
        tick_loop,
        silent_ticks=_AUDIO_STABILIZATION_TICKS,
    )
    frozen_snapshot, _novelty = _select_audio_formation_snapshot(capture_rows)
    composite_snapshot = _build_audio_composite_snapshot(
        frame_snapshots,
        frozen_snapshot,
    )
    if composite_snapshot is None:
        return None, {}

    prepared_snapshot = tick_loop.trace_formation.prepare_snapshot_for_formation(
        composite_snapshot,
        tick_loop.history,
        active_traces=tick_loop.last_active_traces,
        co_trace_ids=tick_loop.working_memory.trace_ids,
    )
    return prepared_snapshot, dict(tick_loop.trace_formation.last_audio_family_selection_debug)


def _prepared_audio_snapshot_summary(
    snapshot: ActivationSnapshot | None,
    *,
    selector_debug: dict[str, object] | None = None,
    selection_debug: dict[str, object] | None = None,
) -> dict[str, object]:
    audio_ids = sorted(
        neuron_id
        for neuron_id, _activation in (snapshot.active_neurons.get("audio", []) if snapshot else [])
    )
    pattern_ids = sorted(
        neuron_id
        for neuron_id, _activation in (snapshot.active_neurons.get("pattern", []) if snapshot else [])
    )
    return {
        "audio_ids": audio_ids,
        "audio_family_counts": _audio_family_counts_from_ids(audio_ids),
        "audio_neuron_count": len(audio_ids),
        "has_complex_family": any(
            _audio_family_for_neuron(neuron_id) == "complex"
            for neuron_id in audio_ids
        ),
        "pattern_ids": pattern_ids,
        "pattern_neuron_count": len(pattern_ids),
        "selector_debug": dict(selector_debug or {}),
        "selection_debug": dict(selection_debug or {}),
    }


def _score_prepared_audio_cue_snapshot(
    snapshot: ActivationSnapshot,
    *,
    cue_key: str,
    reference_primary_traces: dict[str, Trace | None],
) -> dict[str, object]:
    own_trace = reference_primary_traces.get(cue_key)
    own_overlap = (
        _trace_overlap_breakdown_for_snapshot(snapshot, own_trace)
        if own_trace is not None
        else None
    )
    off_target_overlaps: list[tuple[str, dict[str, object]]] = []
    for other_key, trace in sorted(reference_primary_traces.items()):
        if other_key == cue_key or trace is None:
            continue
        off_target_overlaps.append(
            (other_key, _trace_overlap_breakdown_for_snapshot(snapshot, trace))
        )

    max_off_target_key = None
    max_off_target_overlap = None
    if off_target_overlaps:
        max_off_target_key, max_off_target_overlap = max(
            off_target_overlaps,
            key=lambda item: (
                float(item[1].get("trace_overlap_ratio", 0.0) or 0.0),
                int(item[1].get("shared_total", 0) or 0),
                item[0],
            ),
        )

    own_ratio = float((own_overlap or {}).get("trace_overlap_ratio", 0.0) or 0.0)
    own_shared = int((own_overlap or {}).get("shared_total", 0) or 0)
    max_off_ratio = float((max_off_target_overlap or {}).get("trace_overlap_ratio", 0.0) or 0.0)
    max_off_shared = int((max_off_target_overlap or {}).get("shared_total", 0) or 0)
    avg_off_ratio = (
        sum(
            float(row.get("trace_overlap_ratio", 0.0) or 0.0)
            for _other_key, row in off_target_overlaps
        )
        / len(off_target_overlaps)
        if off_target_overlaps
        else 0.0
    )
    return {
        "own_trace_overlap_ratio": round(own_ratio, 4),
        "own_shared_total": own_shared,
        "max_off_target_overlap_key": max_off_target_key,
        "max_off_target_overlap_ratio": round(max_off_ratio, 4),
        "max_off_target_shared_total": max_off_shared,
        "avg_off_target_overlap_ratio": round(avg_off_ratio, 4),
        "selection_score": round(own_ratio - max_off_ratio, 4),
    }


def _prepared_audio_cue_selection_key(selection_debug: dict[str, object]) -> tuple[float, ...]:
    return (
        float(selection_debug.get("selection_score", 0.0) or 0.0),
        float(selection_debug.get("own_trace_overlap_ratio", 0.0) or 0.0),
        -float(selection_debug.get("max_off_target_overlap_ratio", 0.0) or 0.0),
        -float(selection_debug.get("avg_off_target_overlap_ratio", 0.0) or 0.0),
        float(selection_debug.get("own_shared_total", 0) or 0),
        -float(selection_debug.get("max_off_target_shared_total", 0) or 0),
    )


def _select_prepared_audio_cue_snapshot(
    sample: dict[str, Any],
    *,
    ticks_per_sample: int,
    cue_fraction: float,
    n_traces: int,
    seed_chunks: int | None,
    audio_quality_entry: dict[str, object] | None,
    frame_components: list[dict[str, object]],
    reference_primary_traces: dict[str, Trace | None],
    candidate_trials: int = _AUDIO_PREPARED_CUE_CANDIDATE_TRIALS,
) -> tuple[ActivationSnapshot | None, list[dict[str, object]], dict[str, object]]:
    sample_key = str(sample["key"])
    best_snapshot: ActivationSnapshot | None = None
    best_frame_encodings: list[dict[str, object]] = []
    best_selector_debug: dict[str, object] = {}
    best_selection_debug: dict[str, object] | None = None

    for trial_index in range(candidate_trials):
        frame_encodings = _build_audio_frame_encodings(
            frame_components,
            keep_indices=_select_audio_quality_keep_indices(
                len(frame_components),
                cue_fraction,
                salt=f"{sample_key}:probe:{trial_index}",
            ),
        )
        grouped_frame_encodings = _group_audio_frame_encodings(
            frame_encodings,
            ticks_per_sample,
        )
        snapshot, selector_debug = _build_prepared_audio_cue_snapshot(
            sample,
            ticks_per_sample=ticks_per_sample,
            n_traces=n_traces,
            seed_chunks=seed_chunks,
            audio_quality_entry=audio_quality_entry,
            frame_encodings=grouped_frame_encodings,
        )
        if snapshot is None:
            continue

        selection_debug = _score_prepared_audio_cue_snapshot(
            snapshot,
            cue_key=sample_key,
            reference_primary_traces=reference_primary_traces,
        )
        selection_debug["trial_index"] = trial_index
        if best_selection_debug is None or _prepared_audio_cue_selection_key(
            selection_debug
        ) > _prepared_audio_cue_selection_key(best_selection_debug):
            best_snapshot = snapshot
            best_frame_encodings = grouped_frame_encodings
            best_selector_debug = dict(selector_debug)
            best_selection_debug = dict(selection_debug)

    if best_selection_debug is None:
        best_selection_debug = {
            "trial_index": None,
            "own_trace_overlap_ratio": 0.0,
            "own_shared_total": 0,
            "max_off_target_overlap_key": None,
            "max_off_target_overlap_ratio": 0.0,
            "max_off_target_shared_total": 0,
            "avg_off_target_overlap_ratio": 0.0,
            "selection_score": 0.0,
        }

    return (
        best_snapshot,
        best_frame_encodings,
        _prepared_audio_snapshot_summary(
            best_snapshot,
            selector_debug=best_selector_debug,
            selection_debug=best_selection_debug,
        ),
    )


def _build_audio_composite_snapshot(
    frame_snapshots: list[ActivationSnapshot],
    fallback_snapshot: ActivationSnapshot | None,
) -> ActivationSnapshot | None:
    if not frame_snapshots and fallback_snapshot is None:
        return None

    family_pools: dict[str, dict[int, float]] = {
        family_name: {}
        for family_name in _AUDIO_FAMILY_RANGES
    }
    for snapshot in frame_snapshots:
        for neuron_id, activation in snapshot.active_neurons.get("audio", []):
            family_name = _audio_family_for_neuron(neuron_id)
            if family_name is None:
                continue
            existing = family_pools[family_name].get(neuron_id)
            if existing is None or activation > existing:
                family_pools[family_name][neuron_id] = activation

    selected_audio: list[tuple[int, float]] = []
    selected_audio_ids: set[int] = set()
    available_families = [
        family_name
        for family_name, pool in family_pools.items()
        if pool
    ]
    if available_families:
        family_reserve = max(1, _AUDIO_TRACE_AUDIO_NEURONS // len(available_families))
        for family_name in available_families:
            family_candidates = sorted(
                family_pools[family_name].items(),
                key=lambda item: (item[1], -item[0]),
                reverse=True,
            )
            for neuron_id, activation in family_candidates[:family_reserve]:
                if neuron_id in selected_audio_ids:
                    continue
                selected_audio.append((neuron_id, activation))
                selected_audio_ids.add(neuron_id)

        remaining_candidates: list[tuple[int, float]] = []
        for family_name in available_families:
            for neuron_id, activation in family_pools[family_name].items():
                if neuron_id not in selected_audio_ids:
                    remaining_candidates.append((neuron_id, activation))
        remaining_candidates.sort(
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )
        for neuron_id, activation in remaining_candidates:
            if len(selected_audio) >= _AUDIO_TRACE_AUDIO_NEURONS:
                break
            selected_audio.append((neuron_id, activation))
            selected_audio_ids.add(neuron_id)

    if not selected_audio and fallback_snapshot is not None:
        selected_audio = list(fallback_snapshot.active_neurons.get("audio", []))

    pattern_source = fallback_snapshot
    if pattern_source is None or not pattern_source.active_neurons.get("pattern"):
        for snapshot in reversed(frame_snapshots):
            if snapshot.active_neurons.get("pattern"):
                pattern_source = snapshot
                break

    active_neurons: dict[str, list[tuple[int, float]]] = {}
    if selected_audio:
        active_neurons["audio"] = selected_audio[:_AUDIO_TRACE_AUDIO_NEURONS]
    if pattern_source is not None and pattern_source.active_neurons.get("pattern"):
        active_neurons["pattern"] = list(pattern_source.active_neurons["pattern"])

    if not active_neurons:
        return fallback_snapshot

    active_values: list[tuple[int, float]] = []
    active_ids: list[int] = []
    for neurons in active_neurons.values():
        active_values.extend(neurons)
        active_ids.extend(neuron_id for neuron_id, _activation in neurons)
    tick = pattern_source.tick if pattern_source is not None else frame_snapshots[-1].tick
    return ActivationSnapshot(
        tick=tick,
        active_neurons=active_neurons,
        active_values=active_values,
        total_active=len(active_ids),
        active_ids=active_ids,
        region_active_counts={
            region_name: len(neurons)
            for region_name, neurons in active_neurons.items()
        },
    )


def _replay_audio_frozen_snapshot(
    tick_loop: TickLoop,
    snapshot: ActivationSnapshot | None,
    novelty: float,
    *,
    replay_ticks: int,
) -> list[dict[str, object]]:
    if snapshot is None or not snapshot.active_neurons:
        return []

    replay_rows: list[dict[str, object]] = []
    for replay_tick in range(1, replay_ticks + 1):
        replay_started = time.perf_counter()
        tick_loop.trace_formation.step(
            snapshot,
            tick_loop.last_active_traces,
            novelty,
            tick_loop.last_tick_number,
            len(tick_loop.working_memory.trace_ids),
            co_trace_ids=tick_loop.working_memory.trace_ids,
            history=tick_loop.history,
        )
        replay_rows.append(
            {
                "phase_step": replay_tick,
                "novelty": novelty,
                "snapshot": snapshot,
                "region_counts": _formation_snapshot_region_counts(snapshot),
                "debug": dict(tick_loop.trace_formation.last_step_debug),
                "replay_ms": (time.perf_counter() - replay_started) * 1000,
                "recently_formed": list(tick_loop.trace_formation.recently_formed),
            }
        )
        if tick_loop.trace_formation.recently_formed:
            break
    return replay_rows


def _append_audio_fingerprint_observation(
    fingerprints: list[tuple[int, ...]],
    timeline: list[dict[str, object]],
    snapshot: ActivationSnapshot,
    *,
    logical_tick: int,
    phase: str,
    phase_step: int,
    novelty: float | None = None,
) -> None:
    fingerprint = _formation_fingerprint(snapshot.active_neurons)
    previous = fingerprints[-1] if fingerprints else None
    consecutive_jaccard = (
        _jaccard_ratio(previous, fingerprint) if previous is not None else None
    )
    fingerprints.append(fingerprint)
    row: dict[str, object] = {
        "tick": logical_tick,
        "frame": phase_step,
        "phase": phase,
        "phase_step": phase_step,
        "fingerprint_size": len(fingerprint),
        "candidate_region_counts": _formation_snapshot_region_counts(snapshot),
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
    if novelty is not None:
        row["novelty"] = round(novelty, 4)
    timeline.append(row)


def _copy_traces_into_store(
    source_store: TraceStore,
    trace_ids: set[str],
    target_store: TraceStore,
    *,
    formation_order: list[str] | None = None,
) -> set[str]:
    copied_trace_ids: set[str] = set()
    ordered_trace_ids: list[str] = []
    seen_trace_ids: set[str] = set()
    for trace_id in formation_order or []:
        if trace_id in trace_ids and trace_id not in seen_trace_ids:
            ordered_trace_ids.append(trace_id)
            seen_trace_ids.add(trace_id)
    for trace_id in sorted(trace_ids):
        if trace_id not in seen_trace_ids:
            ordered_trace_ids.append(trace_id)
            seen_trace_ids.add(trace_id)

    for trace_id in ordered_trace_ids:
        trace = source_store.get(trace_id)
        trace_copy = _copy_trace(trace)
        if trace_copy is None:
            continue
        target_store.add(trace_copy)
        copied_trace_ids.add(trace_copy.id)
    return copied_trace_ids


def _train_audio_probe_sample(
    trained_sample: dict[str, Any],
    *,
    ticks_per_sample: int,
    train_repeats: int,
    n_traces: int,
    seed_chunks: int | None,
    rest_ticks: int,
    collect_diagnostics: bool,
    audio_quality_entry: dict[str, object] | None = None,
    frame_encodings: list[dict[str, object]] | None = None,
    initial_state_dir: str | None = None,
) -> dict[str, object]:
    tick_loop, trace_store = _create_audio_probe_tick_loop(
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        initial_state_dir=initial_state_dir,
    )
    tick_loop.trace_formation.set_audio_quality_scores(
        dict((audio_quality_entry or {}).get("quality_scores", {})),
        families=_AUDIO_QUALITY_TARGET_FAMILIES,
    )
    learned_trace_ids: set[str] = set()
    learned_trace_order: list[str] = []

    failure_stage_counts: Counter[str] = Counter()
    attempt_ticks = 0
    novelty_gate_ticks = 0
    region_gate_ticks = 0
    ready_pattern_ticks = 0
    training_ticks = 0
    last_training_debug: dict[str, object] = {}
    pattern_fingerprints: list[tuple[int, ...]] = []
    pattern_fingerprint_timeline: list[dict[str, object]] = []
    training_signal_rows: list[dict[str, int]] = []
    training_step_profile_rows: list[dict[str, float]] = []
    candidate_snapshot_rows: list[dict[str, object]] = []
    sequence_encode_ms_total = 0.0
    sequence_tick_loop_ms_total = 0.0
    sequence_processing_wall_ms_total = 0.0
    sequence_frame_count = 0
    capture_tick_ms_total = 0.0
    capture_tick_count = 0
    rest_tick_ms_total = 0.0
    rest_tick_count = 0
    frozen_replay_ms_total = 0.0
    frozen_replay_count = 0

    frame_encodings = frame_encodings or _build_audio_frame_encodings(
        _precompute_audio_frame_components(
            _split_audio_frames(trained_sample["audio"], ticks_per_sample),
            int(trained_sample["sample_rate"]),
        )
    )
    for _ in range(train_repeats):
        if learned_trace_ids:
            break

        tick_loop.reset_sample_boundary()
        sequence_started = time.perf_counter()
        frame_snapshots: list[ActivationSnapshot] = []
        for frame_index, frame_encoding in enumerate(frame_encodings, start=1):
            encode_started = time.perf_counter()
            encoded = _inject_audio_frame_encoding(frame_encoding)
            encode_ms = (time.perf_counter() - encode_started) * 1000
            step_result = tick_loop.step(
                allow_trace_formation=False,
                allow_binding_formation=False,
            )
            sequence_encode_ms_total += encode_ms
            sequence_tick_loop_ms_total += float(step_result.get("step_internal_ms", 0.0) or 0.0)
            sequence_frame_count += 1
            _append_step_profile_row(training_step_profile_rows, step_result)
            formation_snapshot = tick_loop.trace_formation.prepare_snapshot_for_formation(
                tick_loop.history.current,
                tick_loop.history,
                active_traces=tick_loop.last_active_traces,
                co_trace_ids=tick_loop.working_memory.trace_ids,
            )
            frame_snapshots.append(formation_snapshot)
            candidate_snapshot_rows.append(
                _candidate_snapshot_row(formation_snapshot, phase="frame")
            )
            if collect_diagnostics:
                training_ticks += 1
                training_signal_rows.append(
                    {
                        "frame": frame_index,
                        "freq_count": int(encoded.get("freq_count", 0) or 0),
                        "temporal_count": int(encoded.get("temporal_count", 0) or 0),
                        "complex_count": int(encoded.get("complex_count", 0) or 0),
                        "total_signals": int(encoded.get("total_signals", 0) or 0),
                    }
                )
                _append_audio_fingerprint_observation(
                    pattern_fingerprints,
                    pattern_fingerprint_timeline,
                    formation_snapshot,
                    logical_tick=training_ticks,
                    phase="frame",
                    phase_step=frame_index,
                )
        sequence_processing_wall_ms_total += (time.perf_counter() - sequence_started) * 1000

        capture_rows = _capture_audio_formation_window(
            tick_loop,
            silent_ticks=_AUDIO_STABILIZATION_TICKS,
        )
        for capture_row in capture_rows:
            step_result = dict(capture_row.get("step_result", {}))
            _append_step_profile_row(training_step_profile_rows, step_result)
            capture_tick_ms_total += float(step_result.get("step_internal_ms", 0.0) or 0.0)
            capture_tick_count += 1
            candidate_snapshot_rows.append(
                _candidate_snapshot_row(capture_row["snapshot"], phase="silent_capture")
            )
        if collect_diagnostics:
            for capture_row in capture_rows:
                training_ticks += 1
                _append_audio_fingerprint_observation(
                    pattern_fingerprints,
                    pattern_fingerprint_timeline,
                    capture_row["snapshot"],
                    logical_tick=training_ticks,
                    phase="silent_capture",
                    phase_step=int(capture_row["phase_step"]),
                    novelty=float(capture_row["novelty"]),
                )

        frozen_snapshot, frozen_novelty = _select_audio_formation_snapshot(capture_rows)
        composite_snapshot = _build_audio_composite_snapshot(
            frame_snapshots,
            frozen_snapshot,
        )
        if composite_snapshot is not None:
            candidate_snapshot_rows.append(
                _candidate_snapshot_row(composite_snapshot, phase="composite")
            )
        replay_rows = _replay_audio_frozen_snapshot(
            tick_loop,
            composite_snapshot,
            frozen_novelty,
            replay_ticks=_AUDIO_FROZEN_REPLAY_TICKS,
        )

        for replay_row in replay_rows:
            frozen_replay_ms_total += float(replay_row.get("replay_ms", 0.0) or 0.0)
            frozen_replay_count += 1
            candidate_snapshot_rows.append(
                _candidate_snapshot_row(replay_row["snapshot"], phase="frozen_replay")
            )
            for trace_id in replay_row["recently_formed"]:
                if trace_id not in learned_trace_ids:
                    learned_trace_order.append(trace_id)
                learned_trace_ids.add(trace_id)

            if collect_diagnostics:
                debug = dict(replay_row["debug"])
                last_training_debug = debug
                training_ticks += 1
                _append_audio_fingerprint_observation(
                    pattern_fingerprints,
                    pattern_fingerprint_timeline,
                    replay_row["snapshot"],
                    logical_tick=training_ticks,
                    phase="frozen_replay",
                    phase_step=int(replay_row["phase_step"]),
                    novelty=float(replay_row["novelty"]),
                )
                failure_stage_counts[str(debug.get("failure_stage", "none"))] += 1
                attempt_ticks += 1 if bool(debug.get("attempted", False)) else 0
                novelty_gate_ticks += 1 if bool(debug.get("passed_novelty_gate", False)) else 0
                region_gate_ticks += 1 if bool(debug.get("passed_region_gate", False)) else 0
                ready_pattern_ticks += (
                    1 if int(debug.get("ready_pattern_count", 0) or 0) > 0 else 0
                )

            if replay_row["recently_formed"]:
                break

        for _ in range(rest_ticks):
            rest_result = tick_loop.step(
                learn=False,
                allow_trace_formation=False,
                allow_binding_formation=False,
            )
            _append_step_profile_row(training_step_profile_rows, rest_result)
            rest_tick_ms_total += float(rest_result.get("step_internal_ms", 0.0) or 0.0)
            rest_tick_count += 1

    training_diagnostics = {
        "attempt_ticks": attempt_ticks,
        "passed_novelty_gate_ticks": novelty_gate_ticks,
        "passed_region_gate_ticks": region_gate_ticks,
        "ready_pattern_ticks": ready_pattern_ticks,
        "failure_stage_counts": dict(sorted(failure_stage_counts.items())),
        "frame_signal_avg": {
            "freq": round(
                sum(row["freq_count"] for row in training_signal_rows) / max(1, len(training_signal_rows)),
                3,
            ),
            "temporal": round(
                sum(row["temporal_count"] for row in training_signal_rows) / max(1, len(training_signal_rows)),
                3,
            ),
            "complex": round(
                sum(row["complex_count"] for row in training_signal_rows) / max(1, len(training_signal_rows)),
                3,
            ),
            "signals": round(
                sum(row["total_signals"] for row in training_signal_rows) / max(1, len(training_signal_rows)),
                3,
            ),
        },
        "pattern_fingerprint_summary": _summarize_fingerprint_sequence(
            pattern_fingerprints,
        ),
        "candidate_snapshot_summary": _summarize_candidate_snapshot_rows(candidate_snapshot_rows),
        "tick_profile": _summarize_step_profile_rows(training_step_profile_rows),
        "phase_timing_ms": {
            "sequence_encode_ms_total": round(sequence_encode_ms_total, 4),
            "sequence_encode_ms_avg_per_frame": round(
                sequence_encode_ms_total / max(1, sequence_frame_count),
                4,
            ),
            "sequence_tick_loop_ms_total": round(sequence_tick_loop_ms_total, 4),
            "sequence_tick_loop_ms_avg_per_frame": round(
                sequence_tick_loop_ms_total / max(1, sequence_frame_count),
                4,
            ),
            "sequence_processing_wall_ms_total": round(sequence_processing_wall_ms_total, 4),
            "sequence_processing_wall_ms_avg_per_repeat": round(
                sequence_processing_wall_ms_total / max(1, train_repeats),
                4,
            ),
            "silent_capture_tick_ms_total": round(capture_tick_ms_total, 4),
            "silent_capture_tick_ms_avg": round(
                capture_tick_ms_total / max(1, capture_tick_count),
                4,
            ),
            "frozen_replay_ms_total": round(frozen_replay_ms_total, 4),
            "frozen_replay_ms_avg": round(
                frozen_replay_ms_total / max(1, frozen_replay_count),
                4,
            ),
            "rest_tick_ms_total": round(rest_tick_ms_total, 4),
            "rest_tick_ms_avg": round(
                rest_tick_ms_total / max(1, rest_tick_count),
                4,
            ),
        },
        "pattern_fingerprint_timeline": pattern_fingerprint_timeline,
        "last_step_debug": last_training_debug,
    }

    return {
        "tick_loop": tick_loop,
        "trace_store": trace_store,
        "learned_trace_ids": learned_trace_ids,
        "learned_trace_order": learned_trace_order,
        "training_step_profile_rows": training_step_profile_rows,
        "candidate_snapshot_rows": candidate_snapshot_rows,
        "training_diagnostics": training_diagnostics,
    }


def _collect_audio_training_runs(
    probe_samples: list[dict[str, Any]],
    *,
    ticks_per_sample: int,
    train_repeats: int,
    n_traces: int,
    seed_chunks: int | None,
    rest_ticks: int,
    audio_quality_cache: dict[str, dict[str, object]] | None = None,
    frame_encodings_by_key: dict[str, list[dict[str, object]]] | None = None,
    initial_state_dir: str | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, Trace | None]]:
    training_runs: dict[str, dict[str, object]] = {}
    primary_traces: dict[str, Trace | None] = {}

    for trained_sample in probe_samples:
        trained_key = str(trained_sample["key"])
        quality_entry = (audio_quality_cache or {}).get(str(trained_sample["key"]), {})
        training = _train_audio_probe_sample(
            trained_sample,
            ticks_per_sample=ticks_per_sample,
            train_repeats=train_repeats,
            n_traces=n_traces,
            seed_chunks=seed_chunks,
            rest_ticks=rest_ticks,
            collect_diagnostics=True,
            audio_quality_entry=quality_entry,
            frame_encodings=(frame_encodings_by_key or {}).get(trained_key),
            initial_state_dir=initial_state_dir,
        )
        training_runs[trained_key] = training
        trace_store = training["trace_store"]
        learned_trace_order = training["learned_trace_order"]
        primary_trace_id = learned_trace_order[0] if learned_trace_order else None
        primary_traces[trained_key] = _copy_trace(
            trace_store.get(primary_trace_id) if primary_trace_id is not None else None
        )

    return training_runs, primary_traces


def run_audio_learning_probe(
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
    probe_samples: list[dict[str, Any]] | None = None,
    audio_quality_cache: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Train curated audio samples one at a time and probe recall from prefix cues."""
    if threads > 0:
        try:
            brain_core.set_num_threads(threads)
        except Exception:
            pass
    actual_threads = brain_core.get_num_threads()

    probe_samples = probe_samples or _prepare_audio_probe_samples(max_samples)
    audio_quality_cache = audio_quality_cache or _build_audio_quality_cache(
        probe_samples,
        ticks_per_sample,
        cue_fraction,
    )
    brain_core.init_brain()
    internal_frame_count = _audio_internal_frame_count(ticks_per_sample)
    sample_frame_components_by_key = {
        str(sample["key"]): _precompute_audio_frame_components(
            _split_audio_frames(sample["audio"], internal_frame_count),
            int(sample["sample_rate"]),
        )
        for sample in probe_samples
    }
    sample_frame_encodings_by_key = {
        sample_key: _group_audio_frame_encodings(
            _build_audio_frame_encodings(frame_components),
            ticks_per_sample,
        )
        for sample_key, frame_components in sample_frame_components_by_key.items()
    }
    training_runs, reference_primary_traces = _collect_audio_training_runs(
        probe_samples,
        ticks_per_sample=ticks_per_sample,
        train_repeats=train_repeats,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        rest_ticks=rest_ticks,
        audio_quality_cache=audio_quality_cache,
        frame_encodings_by_key=sample_frame_encodings_by_key,
    )
    cue_frame_encodings_by_key: dict[str, list[dict[str, object]]] = {}
    prepared_cue_snapshots_by_key: dict[str, ActivationSnapshot | None] = {}
    prepared_cue_selection_by_key: dict[str, dict[str, object]] = {}
    for sample in probe_samples:
        sample_key = str(sample["key"])
        prepared_snapshot, cue_frame_encodings, prepared_summary = _select_prepared_audio_cue_snapshot(
            sample,
            ticks_per_sample=ticks_per_sample,
            cue_fraction=cue_fraction,
            n_traces=n_traces,
            seed_chunks=seed_chunks,
            audio_quality_entry=audio_quality_cache.get(sample_key, {}),
            frame_components=sample_frame_components_by_key[sample_key],
            reference_primary_traces=reference_primary_traces,
        )
        cue_frame_encodings_by_key[sample_key] = cue_frame_encodings
        prepared_cue_snapshots_by_key[sample_key] = prepared_snapshot
        prepared_cue_selection_by_key[sample_key] = prepared_summary
    results: dict[str, dict[str, object]] = {}
    global_training_step_profile_rows: list[dict[str, float]] = []
    global_probe_step_profile_rows: list[dict[str, float]] = []
    global_candidate_snapshot_rows: list[dict[str, object]] = []
    global_phase_timing_ms: Counter[str] = Counter()

    for trained_sample in probe_samples:
        trained_key = str(trained_sample["key"])
        quality_entry = audio_quality_cache.get(trained_key, {})
        training = training_runs[trained_key]
        training["training_diagnostics"]["audio_quality_summary"] = quality_entry.get("families", {})
        global_training_step_profile_rows.extend(training["training_step_profile_rows"])
        global_candidate_snapshot_rows.extend(training["candidate_snapshot_rows"])
        for key, value in dict(training["training_diagnostics"].get("phase_timing_ms", {})).items():
            global_phase_timing_ms[f"training_{key}"] += float(value or 0.0)
        trace_store = training["trace_store"]
        learned_trace_ids = set(training["learned_trace_ids"])
        learned_trace_order = list(training["learned_trace_order"])
        probe_tick_loop, probe_trace_store = _create_audio_probe_tick_loop(
            n_traces=n_traces,
            seed_chunks=seed_chunks,
        )
        probe_learned_trace_ids = _copy_traces_into_store(
            trace_store,
            learned_trace_ids,
            probe_trace_store,
            formation_order=learned_trace_order,
        )
        probe_step_profile_rows: list[dict[str, float]] = []
        probe_encode_ms_total = 0.0
        probe_sequence_tick_ms_total = 0.0
        probe_sequence_frame_count = 0
        probe_settle_tick_ms_total = 0.0
        probe_settle_tick_count = 0

        for _ in range(settle_ticks):
            settle_result = probe_tick_loop.step(
                learn=False,
                allow_trace_formation=False,
                allow_binding_formation=False,
            )
            _append_step_profile_row(probe_step_profile_rows, settle_result)
            probe_settle_tick_ms_total += float(settle_result.get("step_internal_ms", 0.0) or 0.0)
            probe_settle_tick_count += 1

        cue_results: dict[str, dict[str, object]] = {}
        for cue_sample in probe_samples:
            cue_key = str(cue_sample["key"])
            probe_tick_loop.reset_probe_boundary()
            for _ in range(settle_ticks):
                settle_result = probe_tick_loop.step(
                    learn=False,
                    allow_trace_formation=False,
                    allow_binding_formation=False,
                )
                _append_step_profile_row(probe_step_profile_rows, settle_result)
                probe_settle_tick_ms_total += float(settle_result.get("step_internal_ms", 0.0) or 0.0)
                probe_settle_tick_count += 1

            baseline_hits = sum(
                1
                for trace_id, _score in probe_tick_loop.last_active_traces
                if trace_id in probe_learned_trace_ids
            )

            cue_frame_encodings = cue_frame_encodings_by_key[cue_key]
            prepared_cue_snapshot = prepared_cue_snapshots_by_key[cue_key]
            cue_frame_counts: list[dict[str, int]] = []
            raw_encoder_probe: dict[str, object] | None = None
            first_tick_activation_probe: dict[str, object] | None = None
            raw_audio_neuron_ids: list[int] = []
            first_tick_audio_neuron_ids: list[int] = []
            raw_audio_signal_counts: dict[str, int] = {}
            first_tick_audio_family_counts: dict[str, int] = {}
            for cue_frame_index, cue_frame_encoding in enumerate(cue_frame_encodings, start=1):
                raw_signal_map = dict(cue_frame_encoding.get("signal_map", {}))
                cue_frame_counts.append(
                    {
                        "neurons_activated": len(list(cue_frame_encoding.get("neurons", []))),
                        "freq_count": int(cue_frame_encoding.get("freq_count", 0) or 0),
                        "temporal_count": int(cue_frame_encoding.get("temporal_count", 0) or 0),
                        "complex_count": int(cue_frame_encoding.get("complex_count", 0) or 0),
                        "total_signals": int(cue_frame_encoding.get("total_signals", 0) or 0),
                    }
                )
                if cue_frame_index == 1:
                    raw_audio_neuron_ids = sorted(raw_signal_map)
                    raw_audio_signal_counts = _audio_family_counts_from_ids(set(raw_signal_map))
                    raw_encoder_probe = _cue_activation_breakdown(
                        _activation_snapshot_from_signal_map(raw_signal_map),
                        reference_primary_traces,
                        cue_key=cue_key,
                        trained_key=trained_key,
                        probe_tick=0,
                        trace_hit_count=0,
                        best_score=0.0,
                    )

            tick_rows: list[dict[str, object]] = []
            cue_activation_rows: list[dict[str, object]] = []
            for probe_tick_index in range(probe_ticks):
                if probe_tick_index == 0 and prepared_cue_snapshot is not None:
                    encode_started = time.perf_counter()
                    brain_core.inject_activations(prepared_cue_snapshot.active_values)
                    probe_encode_ms_total += (time.perf_counter() - encode_started) * 1000
                probe_step_result = probe_tick_loop.step(
                    learn=False,
                    allow_trace_formation=False,
                    allow_binding_formation=False,
                )
                _append_step_profile_row(probe_step_profile_rows, probe_step_result)
                probe_sequence_tick_ms_total += float(
                    probe_step_result.get("step_internal_ms", 0.0) or 0.0
                )
                if probe_tick_index == 0 and prepared_cue_snapshot is not None:
                    probe_sequence_frame_count += 1
                ranks = {
                    trace_id: (rank, score)
                    for rank, (trace_id, score) in enumerate(
                        probe_tick_loop.last_active_traces,
                        start=1,
                    )
                }
                hit_ranks = [
                    ranks[trace_id][0]
                    for trace_id in probe_learned_trace_ids
                    if trace_id in ranks
                ]
                hit_scores = [
                    ranks[trace_id][1]
                    for trace_id in probe_learned_trace_ids
                    if trace_id in ranks
                ]
                tick_rows.append(
                    {
                        "probe_tick": probe_tick_index + 1,
                        "trace_hit_count": len(hit_ranks),
                        "best_rank": min(hit_ranks) if hit_ranks else None,
                        "best_score": round(max(hit_scores), 4) if hit_scores else 0.0,
                        "active_trace_count": len(probe_tick_loop.last_active_traces),
                        "cue_recall_signals": 0,
                    }
                )
                cue_activation_rows.append(
                    _cue_activation_breakdown(
                        probe_tick_loop.history.current,
                        reference_primary_traces,
                        cue_key=cue_key,
                        trained_key=trained_key,
                        probe_tick=probe_tick_index + 1,
                        trace_hit_count=len(hit_ranks),
                        best_score=max(hit_scores) if hit_scores else 0.0,
                    )
                )
                if probe_tick_index == 0:
                    first_tick_audio_ids = _snapshot_active_ids_by_region(
                        probe_tick_loop.history.current,
                    ).get("audio", set())
                    first_tick_audio_neuron_ids = sorted(first_tick_audio_ids)
                    first_tick_audio_family_counts = _audio_family_counts_from_ids(
                        first_tick_audio_neuron_ids,
                    )
                    first_tick_activation_probe = cue_activation_rows[-1]

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

            cue_results[cue_key] = {
                "cue_label": cue_sample["label_name"],
                "cue_frame_count": len(cue_frame_encodings),
                "baseline_trace_hits": baseline_hits,
                "prepared_cue_selection": prepared_cue_selection_by_key.get(cue_key, {}),
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
                "cue_input_signals_avg": round(
                    sum(row["total_signals"] for row in cue_frame_counts) / max(1, len(cue_frame_counts)),
                    3,
                ),
                "cue_input_freq_avg": round(
                    sum(row["freq_count"] for row in cue_frame_counts) / max(1, len(cue_frame_counts)),
                    3,
                ),
                "cue_input_temporal_avg": round(
                    sum(row["temporal_count"] for row in cue_frame_counts) / max(1, len(cue_frame_counts)),
                    3,
                ),
                "cue_stage_diagnostic": {
                    "mask_scope": "raw_first_cue_frame_vs_first_probe_tick",
                    "raw_audio_neuron_ids": raw_audio_neuron_ids,
                    "raw_audio_signal_counts": raw_audio_signal_counts,
                    "first_tick_audio_neuron_ids": first_tick_audio_neuron_ids,
                    "first_tick_audio_family_counts": first_tick_audio_family_counts,
                    "raw_encoder_probe": raw_encoder_probe,
                    "first_tick_activation_probe": first_tick_activation_probe,
                    "comparison": (
                        _compare_audio_cue_stage_probes(
                            raw_encoder_probe,
                            first_tick_activation_probe,
                            cue_key=cue_key,
                            trained_key=trained_key,
                        )
                        if raw_encoder_probe is not None and first_tick_activation_probe is not None
                        else None
                    ),
                },
                "cue_activation_probe": cue_activation_rows[representative_index],
                "ticks": tick_rows,
            }

            for _ in range(settle_ticks):
                settle_result = probe_tick_loop.step(
                    learn=False,
                    allow_trace_formation=False,
                    allow_binding_formation=False,
                )
                _append_step_profile_row(probe_step_profile_rows, settle_result)
                probe_settle_tick_ms_total += float(settle_result.get("step_internal_ms", 0.0) or 0.0)
                probe_settle_tick_count += 1

        probe_diagnostics = {
            "tick_profile": _summarize_step_profile_rows(probe_step_profile_rows),
            "phase_timing_ms": {
                "cue_encode_ms_total": round(probe_encode_ms_total, 4),
                "cue_encode_ms_avg_per_frame": round(
                    probe_encode_ms_total / max(1, probe_sequence_frame_count),
                    4,
                ),
                "cue_tick_loop_ms_total": round(probe_sequence_tick_ms_total, 4),
                "cue_tick_loop_ms_avg_per_frame": round(
                    probe_sequence_tick_ms_total / max(1, probe_sequence_frame_count),
                    4,
                ),
                "settle_tick_ms_total": round(probe_settle_tick_ms_total, 4),
                "settle_tick_ms_avg": round(
                    probe_settle_tick_ms_total / max(1, probe_settle_tick_count),
                    4,
                ),
            },
        }
        global_probe_step_profile_rows.extend(probe_step_profile_rows)
        for key, value in dict(probe_diagnostics.get("phase_timing_ms", {})).items():
            global_phase_timing_ms[f"probe_{key}"] += float(value or 0.0)

        results[str(trained_sample["key"])] = {
            "trained_label": trained_sample["label_name"],
            "reference_index": int(trained_sample["index"]),
            "sample_rate": trained_sample["sample_rate"],
            "frame_count": len(sample_frame_encodings_by_key[trained_key]),
            "learned_trace_summary": _summarize_learned_traces(
                trace_store,
                learned_trace_ids,
                formation_order=learned_trace_order,
            ),
            "training_diagnostics": training["training_diagnostics"],
            "probe_diagnostics": probe_diagnostics,
            "cue_results": cue_results,
        }

    summary = _summarize_audio_probe_results(results)
    for sample_key, sample_summary in summary["sample_summaries"].items():
        results[sample_key]["sample_summary"] = sample_summary

    output = {
        "benchmark": "audio_learning_probe",
        "dataset": "esc50",
        "threads": actual_threads,
        "seed_traces": n_traces,
        "seed_chunks": seed_chunks,
        "ticks_per_sample": ticks_per_sample,
        "train_repeats": train_repeats,
        "rest_ticks": rest_ticks,
        "capture_ticks": _AUDIO_STABILIZATION_TICKS,
        "frozen_replay_ticks": _AUDIO_FROZEN_REPLAY_TICKS,
        "settle_ticks": settle_ticks,
        "probe_ticks": probe_ticks,
        "cue_fraction": cue_fraction,
        "reference_labels": [str(sample["key"]) for sample in probe_samples],
        "audio_internal_frame_count": internal_frame_count,
        "probe_mode": "isolated_audio_masked_cue_silent_recall",
        "trace_store_mode": "learned_only",
        "prepared_cue_selection": prepared_cue_selection_by_key,
        "performance_profile": {
            "training_tick_profile": _summarize_step_profile_rows(global_training_step_profile_rows),
            "probe_tick_profile": _summarize_step_profile_rows(global_probe_step_profile_rows),
            "overall_tick_profile": _summarize_step_profile_rows(
                global_training_step_profile_rows + global_probe_step_profile_rows
            ),
            "phase_timing_ms": {
                key: round(value, 4)
                for key, value in sorted(global_phase_timing_ms.items())
            },
            "candidate_snapshot_summary": _summarize_candidate_snapshot_rows(
                global_candidate_snapshot_rows,
            ),
            "candidate_neuron_baselines": {
                "text_learning_probe": 48,
                "visual_learning_probe": 32,
            },
        },
        "summary": summary,
        "results": results,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print("\nAUDIO LEARNING PROBE SUMMARY")
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