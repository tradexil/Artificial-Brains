"""Controlled multimodal binding probe on a small matched concept catalog."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import brain_core

from brain.benchmarks.audio_learning_probe import (
    _audio_internal_frame_count,
    _build_audio_frame_encodings,
    _build_audio_quality_cache,
    _collect_audio_training_runs,
    _group_audio_frame_encodings,
    _precompute_audio_frame_components,
    _select_audio_probe_samples,
    _split_audio_frames,
)
from brain.benchmarks.text_learning_probe import _add_anchor_trace, _coverage_row
from brain.benchmarks.visual_learning_probe import (
    _build_visual_quality_cache,
    _collect_primary_visual_reference_traces,
    _select_visual_probe_samples,
)
from brain.datasets.downloader import (
    load_audio_dataset,
    load_image_dataset,
    load_text_dataset,
)
from brain.input.text_input import TextInput
from brain.learning.tick_loop import TickLoop
from brain.seed.seed_runner import seed_brain_fast
from brain.serialize.runtime_bundle import load_runtime_bundle
from brain.structures.trace_store import Trace, TraceStore
from brain.utils.config import TOTAL_NEURONS


DEFAULT_MULTIMODAL_BINDING_PROBE_SPEC: dict[str, object] = {
    "concept_key": "airplane",
    "text": {
        "dataset": "ag_news",
        "index": 224,
        "key": "airplane_text",
        "anchors": (
            ("plane", "pattern"),
            ("solo trip", "integration"),
            ("world", "memory_long"),
            ("refuelling", "executive"),
        ),
    },
    "visual": {
        "dataset": "cifar10",
        "label_name": "airplane",
    },
    "audio": {
        "dataset": "esc50",
        "label_name": "airplane",
    },
}

DEFAULT_MULTIMODAL_BINDING_PROBE_CATALOG: tuple[dict[str, object], ...] = (
    DEFAULT_MULTIMODAL_BINDING_PROBE_SPEC,
    {
        "concept_key": "dog",
        "text": {
            "dataset": "ag_news",
            "index": 2063,
            "key": "dog_text",
            "anchors": (
                ("dog", "pattern"),
                ("leash", "integration"),
                ("hike", "memory_long"),
                ("companion", "executive"),
            ),
        },
        "visual": {
            "dataset": "cifar10",
            "label_name": "dog",
        },
        "audio": {
            "dataset": "esc50",
            "label_name": "dog",
        },
    },
    {
        "concept_key": "frog",
        "text": {
            "dataset": "ag_news",
            "index": 145,
            "key": "frog_text",
            "anchors": (
                ("chorus frog", "pattern"),
                ("croaking", "integration"),
                ("virginia", "memory_long"),
                ("range", "executive"),
            ),
        },
        "visual": {
            "dataset": "cifar10",
            "label_name": "frog",
        },
        "audio": {
            "dataset": "esc50",
            "label_name": "frog",
        },
    },
)

_MULTIMODAL_TEXT_DATASET_SAMPLE_LIMIT = 700
_MULTIMODAL_VISUAL_DATASET_SAMPLE_LIMIT = 200
_MULTIMODAL_AUDIO_DATASET_SAMPLE_LIMIT = 2000
_MULTIMODAL_HARVEST_QUALITY_CUE_FRACTION = 0.75
_MULTIMODAL_TRACE_ID_TEMPLATE = "multimodal_probe_{concept_key}_{modality}"


def _trace_match_metrics(
    active_traces: list[tuple[str, float]],
    trace_id: str,
) -> tuple[bool, int | None, float]:
    for rank, (active_trace_id, score) in enumerate(active_traces, start=1):
        if active_trace_id == trace_id:
            return True, rank, float(score)
    return False, None, 0.0


def _copy_trace(trace: Trace | None) -> Trace | None:
    if trace is None:
        return None
    return Trace.from_dict(trace.to_dict())


def _flatten_trace_neurons(trace: Trace) -> list[int]:
    neuron_ids: list[int] = []
    for region_ids in trace.neurons.values():
        neuron_ids.extend(int(neuron_id) for neuron_id in region_ids)
    return sorted(set(neuron_ids))


def _trace_summary(trace: Trace | None) -> dict[str, object]:
    if trace is None:
        return {
            "present": False,
            "trace_id": None,
            "label": None,
            "total_neurons": 0,
            "regions": {},
        }

    return {
        "present": True,
        "trace_id": trace.id,
        "label": trace.label,
        "total_neurons": trace.total_neurons(),
        "regions": {
            region_name: len(neuron_ids)
            for region_name, neuron_ids in sorted(trace.neurons.items())
            if neuron_ids
        },
    }


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {
        key: int(counter[key])
        for key in sorted(counter)
    }


def _canonical_pair_key(left: str, right: str) -> str:
    return "<->".join(sorted((left, right)))


def _deterministic_rank(salt: str, value: int) -> str:
    return hashlib.sha1(f"{salt}:{value}".encode("ascii")).hexdigest()


def _build_trace_cue_injections(
    trace: Trace,
    cue_fraction: float,
    *,
    salt: str,
) -> tuple[list[tuple[int, float]], dict[str, object]]:
    activation_by_id: dict[int, float] = {}
    region_counts: dict[str, int] = {}
    total_neurons = trace.total_neurons()
    cue_fraction = min(1.0, max(0.0, float(cue_fraction)))

    for region_name, neuron_ids in sorted(trace.neurons.items()):
        unique_ids = sorted(set(int(neuron_id) for neuron_id in neuron_ids))
        if not unique_ids:
            continue
        if cue_fraction >= 1.0:
            selected = unique_ids
        else:
            keep = max(1, min(len(unique_ids), int(math.ceil(len(unique_ids) * cue_fraction))))
            ranked = sorted(unique_ids, key=lambda neuron_id: _deterministic_rank(f"{salt}:{region_name}", neuron_id))
            selected = ranked[:keep]
        region_counts[region_name] = len(selected)
        for neuron_id in selected:
            activation_by_id[neuron_id] = 1.0

    injections = [
        (neuron_id, activation)
        for neuron_id, activation in sorted(activation_by_id.items())
    ]
    return injections, {
        "selected_neurons": len(injections),
        "total_neurons": total_neurons,
        "selected_fraction": round(len(injections) / max(1, total_neurons), 4),
        "region_counts": region_counts,
    }


def _inject_trace_cues(
    traces_by_modality: dict[str, Trace],
    cue_fraction: float,
    *,
    salt: str,
) -> dict[str, dict[str, object]]:
    combined: dict[int, float] = {}
    cue_summaries: dict[str, dict[str, object]] = {}

    for modality, trace in sorted(traces_by_modality.items()):
        injections, summary = _build_trace_cue_injections(
            trace,
            cue_fraction,
            salt=f"{salt}:{modality}",
        )
        cue_summaries[modality] = summary
        for neuron_id, activation in injections:
            previous = combined.get(neuron_id)
            if previous is None or activation > previous:
                combined[neuron_id] = activation

    if combined:
        brain_core.inject_activations(
            [
                (neuron_id, activation)
                for neuron_id, activation in sorted(combined.items())
            ]
        )
    return cue_summaries


def _trace_active_ratio(tick_loop: TickLoop, trace: Trace) -> float:
    snapshot = tick_loop.history.current
    if snapshot is None:
        return 0.0
    active_set = snapshot.active_set()
    trace_neurons = set(_flatten_trace_neurons(trace))
    if not trace_neurons:
        return 0.0
    return round(len(active_set & trace_neurons) / len(trace_neurons), 4)


def _select_text_sample(spec: dict[str, object]) -> dict[str, object]:
    sample_index = int(spec["index"])
    dataset = load_text_dataset(
        str(spec.get("dataset", "ag_news")),
        max_samples=max(_MULTIMODAL_TEXT_DATASET_SAMPLE_LIMIT, sample_index + 1),
    )
    row = dict(dataset[sample_index])
    row["index"] = sample_index
    row["key"] = str(spec["key"])
    row["anchors"] = list(spec["anchors"])
    return row


def _select_visual_sample(label_name: str) -> dict[str, object]:
    dataset = load_image_dataset(
        "cifar10",
        max_samples=_MULTIMODAL_VISUAL_DATASET_SAMPLE_LIMIT,
    )
    return _select_visual_probe_samples(
        dataset,
        labels=(label_name,),
        max_samples=1,
    )[0]


def _select_audio_sample(label_name: str) -> dict[str, object]:
    dataset = load_audio_dataset(
        "esc50",
        max_samples=_MULTIMODAL_AUDIO_DATASET_SAMPLE_LIMIT,
    )
    return _select_audio_probe_samples(
        dataset,
        labels=(label_name,),
        max_samples=1,
    )[0]


def _harvest_text_primary_trace(
    sample: dict[str, object],
    *,
    ticks_per_sample: int,
    train_repeats: int,
    n_traces: int,
    seed_chunks: int | None,
    rest_ticks: int,
    initial_state_dir: str | None = None,
) -> tuple[Trace | None, dict[str, object]]:
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

    for ordinal, (label, primary_region) in enumerate(sample["anchors"], start=1):
        _add_anchor_trace(
            trace_store,
            str(sample["key"]),
            str(label),
            str(primary_region),
            ordinal,
        )

    encoder = TextInput(trace_store)
    coverage = _coverage_row(encoder, sample)
    learned_trace_ids: set[str] = set()
    learned_trace_order: list[str] = []
    failure_stages: Counter[str] = Counter()
    training_started = time.perf_counter()

    for _ in range(train_repeats):
        tick_loop.reset_sample_boundary()
        encoder.encode(str(sample["text"]))
        for _tick in range(ticks_per_sample):
            tick_loop.step(allow_binding_formation=False)
            debug = dict(tick_loop.trace_formation.last_step_debug)
            failure_stage = debug.get("failure_stage")
            if isinstance(failure_stage, str) and failure_stage:
                failure_stages[failure_stage] += 1
            for trace_id in tick_loop.trace_formation.recently_formed:
                if trace_id not in learned_trace_ids:
                    learned_trace_order.append(trace_id)
                learned_trace_ids.add(trace_id)
        for _ in range(rest_ticks):
            tick_loop.step(learn=False, allow_trace_formation=False, allow_binding_formation=False)

    primary_trace_id = learned_trace_order[0] if learned_trace_order else None
    primary_trace = _copy_trace(
        trace_store.get(primary_trace_id) if primary_trace_id is not None else None
    )
    harvest_summary = {
        "sample_index": int(sample["index"]),
        "sample_key": sample["key"],
        "label_name": sample.get("label_name"),
        "coverage": coverage,
        "anchors": [
            {
                "label": str(label),
                "primary_region": str(primary_region),
            }
            for label, primary_region in sample["anchors"]
        ],
        "learned_trace_count": len(learned_trace_ids),
        "learned_trace_order": list(learned_trace_order),
        "failure_stages": _sorted_counter(failure_stages),
        "primary_trace": _trace_summary(primary_trace),
        "training_wall_ms": round((time.perf_counter() - training_started) * 1000, 4),
    }
    return primary_trace, harvest_summary


def _harvest_visual_primary_trace(
    sample: dict[str, object],
    *,
    ticks_per_sample: int,
    train_repeats: int,
    n_traces: int,
    seed_chunks: int | None,
    rest_ticks: int,
    initial_state_dir: str | None = None,
) -> tuple[Trace | None, dict[str, object]]:
    visual_quality_cache = _build_visual_quality_cache(
        [sample],
        _MULTIMODAL_HARVEST_QUALITY_CUE_FRACTION,
    )
    started = time.perf_counter()
    primary_trace = _collect_primary_visual_reference_traces(
        [sample],
        ticks_per_sample=ticks_per_sample,
        train_repeats=train_repeats,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        rest_ticks=rest_ticks,
        visual_quality_cache=visual_quality_cache,
        initial_state_dir=initial_state_dir,
    ).get(str(sample["key"]))
    return _copy_trace(primary_trace), {
        "sample_index": int(sample["index"]),
        "sample_key": sample["key"],
        "label_name": sample.get("label_name"),
        "quality_summary": dict(visual_quality_cache.get(str(sample["key"]), {}).get("families", {})),
        "primary_trace": _trace_summary(primary_trace),
        "training_wall_ms": round((time.perf_counter() - started) * 1000, 4),
    }


def _harvest_audio_primary_trace(
    sample: dict[str, object],
    *,
    ticks_per_sample: int,
    train_repeats: int,
    n_traces: int,
    seed_chunks: int | None,
    rest_ticks: int,
    initial_state_dir: str | None = None,
) -> tuple[Trace | None, dict[str, object]]:
    audio_quality_cache = _build_audio_quality_cache(
        [sample],
        ticks_per_sample,
        _MULTIMODAL_HARVEST_QUALITY_CUE_FRACTION,
    )
    internal_frame_count = _audio_internal_frame_count(ticks_per_sample)
    frame_components = _precompute_audio_frame_components(
        _split_audio_frames(sample["audio"], internal_frame_count),
        int(sample["sample_rate"]),
    )
    frame_encodings = _group_audio_frame_encodings(
        _build_audio_frame_encodings(frame_components),
        ticks_per_sample,
    )
    started = time.perf_counter()
    training_runs, primary_traces = _collect_audio_training_runs(
        [sample],
        ticks_per_sample=ticks_per_sample,
        train_repeats=train_repeats,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        rest_ticks=rest_ticks,
        audio_quality_cache=audio_quality_cache,
        frame_encodings_by_key={str(sample["key"]): frame_encodings},
        initial_state_dir=initial_state_dir,
    )
    training = training_runs[str(sample["key"])]
    primary_trace = primary_traces.get(str(sample["key"]))
    return _copy_trace(primary_trace), {
        "sample_index": int(sample["index"]),
        "sample_key": sample["key"],
        "label_name": sample.get("label_name"),
        "quality_summary": dict(audio_quality_cache.get(str(sample["key"]), {}).get("families", {})),
        "learned_trace_count": len(training.get("learned_trace_ids", [])),
        "primary_trace": _trace_summary(primary_trace),
        "training_wall_ms": round((time.perf_counter() - started) * 1000, 4),
    }


def _build_learned_only_trace_store(
    concept_key: str,
    harvested_traces: dict[str, Trace | None],
    *,
    n_traces: int,
    seed_chunks: int | None,
    initial_state_dir: str | None = None,
) -> tuple[TraceStore, dict[str, Trace], dict[str, str]]:
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
    traces_by_modality: dict[str, Trace] = {}
    modality_by_trace_id: dict[str, str] = {}

    for modality, trace in sorted(harvested_traces.items()):
        if trace is None:
            continue
        cloned = Trace.from_dict(trace.to_dict())
        cloned.id = _MULTIMODAL_TRACE_ID_TEMPLATE.format(
            concept_key=concept_key,
            modality=modality,
        )
        cloned.label = f"{concept_key}:{modality}"
        trace_store.add(cloned)
        traces_by_modality[modality] = cloned
        modality_by_trace_id[cloned.id] = modality

    return trace_store, traces_by_modality, modality_by_trace_id


def _active_primary_patterns(
    trace_store: TraceStore,
    active_traces: list[tuple[str, float]],
) -> list[tuple[str, str]]:
    return list(
        brain_core.trace_index_active_primary_patterns(
            trace_store.store_id,
            [(trace_id, float(score)) for trace_id, score in active_traces],
        )
    )


def _coactive_pair_diagnostics(
    active_patterns: list[tuple[str, str]],
    modality_by_trace_id: dict[str, str],
) -> dict[str, object]:
    same_trace_rejected = 0
    same_region_rejected = 0
    eligible_cross_modal_pairs = 0
    region_pair_counts: Counter[str] = Counter()
    modality_pair_counts: Counter[str] = Counter()
    active_trace_ids = {
        trace_id
        for trace_id, _region_name in active_patterns
        if trace_id in modality_by_trace_id
    }
    active_pattern_rows = [
        {
            "trace_id": trace_id,
            "region": region_name,
            "modality": modality_by_trace_id.get(trace_id, "unknown"),
        }
        for trace_id, region_name in active_patterns
    ]

    for index, (trace_id_a, region_a) in enumerate(active_patterns):
        for trace_id_b, region_b in active_patterns[index + 1 :]:
            if trace_id_a == trace_id_b:
                same_trace_rejected += 1
                continue
            if region_a == region_b:
                same_region_rejected += 1
                continue

            modality_a = modality_by_trace_id.get(trace_id_a, "unknown")
            modality_b = modality_by_trace_id.get(trace_id_b, "unknown")
            eligible_cross_modal_pairs += 1
            region_pair_counts[_canonical_pair_key(region_a, region_b)] += 1
            modality_pair_counts[_canonical_pair_key(modality_a, modality_b)] += 1

    if len(active_trace_ids) < 2:
        status = "never_coactive"
    elif eligible_cross_modal_pairs == 0:
        status = "coactive_but_rejected"
    else:
        status = "eligible_pairs_seen"

    return {
        "status": status,
        "active_trace_count": len(active_trace_ids),
        "active_pattern_count": len(active_patterns),
        "active_patterns": active_pattern_rows,
        "same_trace_rejected_pair_count": same_trace_rejected,
        "same_region_rejected_pair_count": same_region_rejected,
        "eligible_cross_modal_pair_count": eligible_cross_modal_pairs,
        "pair_region_distribution": _sorted_counter(region_pair_counts),
        "pair_modality_distribution": _sorted_counter(modality_pair_counts),
    }


def _binding_rows(
    tick_loop: TickLoop,
    modality_by_trace_id: dict[str, str],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    cross_modal_count = 0

    for binding_id, detail in sorted(tick_loop.binding_formation.binding_details.items()):
        trace_id_a = str(detail["trace_id_a"])
        trace_id_b = str(detail["trace_id_b"])
        modality_a = modality_by_trace_id.get(trace_id_a, "unknown")
        modality_b = modality_by_trace_id.get(trace_id_b, "unknown")
        binding_info = brain_core.get_binding_info(binding_id)
        weight = fires = confidence = last_fired = 0.0
        if binding_info is not None:
            weight, fires, confidence, last_fired = binding_info
        row = {
            "binding_id": int(binding_id),
            "trace_id_a": trace_id_a,
            "modality_a": modality_a,
            "region_a": str(detail["region_a"]),
            "trace_id_b": trace_id_b,
            "modality_b": modality_b,
            "region_b": str(detail["region_b"]),
            "weight": round(float(weight), 4),
            "fires": int(fires),
            "confidence": round(float(confidence), 4),
            "last_fired": int(last_fired),
            "cross_modal": modality_a != modality_b,
        }
        if row["cross_modal"]:
            cross_modal_count += 1
        rows.append(row)

    return {
        "count": len(rows),
        "cross_modal_count": cross_modal_count,
        "rows": rows,
    }


def _summarize_training_rows(training_rows: list[dict[str, object]]) -> dict[str, object]:
    if not training_rows:
        return {
            "tick_count": 0,
            "active_traces_avg": 0.0,
            "binding_candidates_avg": 0.0,
            "bindings_formed_total": 0,
            "total_bindings_max": 0,
            "sparsity_avg": 0.0,
            "sparsity_max": 0.0,
            "pair_region_distribution_total": {},
            "pair_modality_distribution_total": {},
            "diagnostic_status_counts": {},
            "never_coactive_ticks": 0,
            "coactive_but_rejected_ticks": 0,
            "eligible_but_not_ready_ticks": 0,
        }

    pair_region_total: Counter[str] = Counter()
    pair_modality_total: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    never_coactive_ticks = 0
    coactive_but_rejected_ticks = 0
    eligible_but_not_ready_ticks = 0

    for row in training_rows:
        diagnostics = dict(row["coactive_pair_diagnostics"])
        status = str(diagnostics.get("status", "unknown"))
        status_counts[status] += 1
        if status == "never_coactive":
            never_coactive_ticks += 1
        elif status == "coactive_but_rejected":
            coactive_but_rejected_ticks += 1
        if int(diagnostics.get("eligible_cross_modal_pair_count", 0) or 0) > 0 and int(row["binding_candidates"]) == 0:
            eligible_but_not_ready_ticks += 1

        pair_region_total.update(dict(diagnostics.get("pair_region_distribution", {})))
        pair_modality_total.update(dict(diagnostics.get("pair_modality_distribution", {})))

    tick_count = len(training_rows)
    active_traces_avg = round(
        sum(float(row["active_traces"]) for row in training_rows) / tick_count,
        4,
    )
    binding_candidates_avg = round(
        sum(float(row["binding_candidates"]) for row in training_rows) / tick_count,
        4,
    )
    sparsity_values = [float(row["sparsity_fraction"]) for row in training_rows]

    return {
        "tick_count": tick_count,
        "active_traces_avg": active_traces_avg,
        "binding_candidates_avg": binding_candidates_avg,
        "bindings_formed_total": sum(int(row["bindings_formed"]) for row in training_rows),
        "total_bindings_max": max(int(row["total_bindings"]) for row in training_rows),
        "sparsity_avg": round(sum(sparsity_values) / tick_count, 4),
        "sparsity_max": round(max(sparsity_values), 4),
        "pair_region_distribution_total": _sorted_counter(pair_region_total),
        "pair_modality_distribution_total": _sorted_counter(pair_modality_total),
        "diagnostic_status_counts": _sorted_counter(status_counts),
        "never_coactive_ticks": never_coactive_ticks,
        "coactive_but_rejected_ticks": coactive_but_rejected_ticks,
        "eligible_but_not_ready_ticks": eligible_but_not_ready_ticks,
    }


def _run_recall_probe(
    tick_loop: TickLoop,
    traces_by_modality: dict[str, Trace],
    source_modality: str,
    *,
    cue_fraction: float,
    settle_ticks: int,
    probe_ticks: int,
) -> dict[str, object]:
    cue_trace = traces_by_modality[source_modality]
    partner_modalities = [
        modality
        for modality in sorted(traces_by_modality)
        if modality != source_modality
    ]
    cue_injections, cue_summary = _build_trace_cue_injections(
        cue_trace,
        cue_fraction,
        salt=f"probe:{source_modality}",
    )

    tick_loop.reset_probe_boundary()
    for _ in range(settle_ticks):
        tick_loop.step(learn=False, allow_trace_formation=False, allow_binding_formation=False)

    tick_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for tick_index in range(probe_ticks):
        if tick_index == 0 and cue_injections:
            brain_core.inject_activations(cue_injections)
        result = tick_loop.step(learn=False, allow_trace_formation=False, allow_binding_formation=False)
        row: dict[str, object] = {
            "tick": int(result["tick"]),
            "active_traces": int(result["active_traces"]),
            "total_active": int(result["total_active"]),
            "sparsity_fraction": round(float(result["total_active"]) / TOTAL_NEURONS, 6),
            "total_bindings": int(result["total_bindings"]),
            "step_internal_ms": round(float(result.get("step_internal_ms", 0.0) or 0.0), 4),
            "partners": {},
        }
        for partner_modality in partner_modalities:
            partner_trace = traces_by_modality[partner_modality]
            hit, rank, score = _trace_match_metrics(
                tick_loop.last_active_traces,
                partner_trace.id,
            )
            row["partners"][partner_modality] = {
                "hit": hit,
                "rank": rank,
                "score": round(score, 4),
                "active_ratio": _trace_active_ratio(tick_loop, partner_trace),
            }
        tick_rows.append(row)

    partner_results: dict[str, dict[str, object]] = {}
    for partner_modality in partner_modalities:
        partner_rows = [
            dict(row["partners"][partner_modality])
            for row in tick_rows
        ]
        hit_rows = [row for row in partner_rows if row["hit"]]
        partner_results[partner_modality] = {
            "trace_hit_rate": round(len(hit_rows) / max(1, probe_ticks), 4),
            "best_rank_avg": round(
                sum(int(row["rank"]) for row in hit_rows) / len(hit_rows),
                4,
            ) if hit_rows else None,
            "max_score": round(max(float(row["score"]) for row in partner_rows), 4),
            "max_active_ratio": round(max(float(row["active_ratio"]) for row in partner_rows), 4),
        }

    return {
        "source_modality": source_modality,
        "cue_summary": cue_summary,
        "partner_results": partner_results,
        "tick_rows": tick_rows,
        "sparsity_max": round(max(float(row["sparsity_fraction"]) for row in tick_rows), 4) if tick_rows else 0.0,
        "step_internal_ms_total": round(
            sum(float(row["step_internal_ms"]) for row in tick_rows),
            4,
        ),
        "probe_wall_ms": round((time.perf_counter() - started) * 1000, 4),
    }


def _summarize_multimodal_binding_probe(
    binding_summary: dict[str, object],
    training_summary: dict[str, object],
    recall_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    successful_partner_recalls = 0
    for probe in recall_results.values():
        for partner_summary in probe.get("partner_results", {}).values():
            if float(partner_summary.get("trace_hit_rate", 0.0) or 0.0) > 0.0:
                successful_partner_recalls += 1

    probe_sparsity_max = max(
        [float(probe.get("sparsity_max", 0.0) or 0.0) for probe in recall_results.values()] or [0.0]
    )
    global_sparsity_max = max(
        float(training_summary.get("sparsity_max", 0.0) or 0.0),
        probe_sparsity_max,
    )

    validations = {
        "cross_modal_binding_formed": int(binding_summary.get("cross_modal_count", 0) or 0) >= 1,
        "partner_recall_observed": successful_partner_recalls >= 1,
        "sparsity_under_12_percent": global_sparsity_max < 0.12,
    }
    validations["passes_probe"] = all(validations.values())

    return {
        "active_traces_avg": float(training_summary.get("active_traces_avg", 0.0) or 0.0),
        "binding_candidates_avg": float(training_summary.get("binding_candidates_avg", 0.0) or 0.0),
        "bindings_formed_total": int(training_summary.get("bindings_formed_total", 0) or 0),
        "cross_modal_binding_count": int(binding_summary.get("cross_modal_count", 0) or 0),
        "successful_partner_recall_count": successful_partner_recalls,
        "training_sparsity_max": round(float(training_summary.get("sparsity_max", 0.0) or 0.0), 4),
        "probe_sparsity_max": round(probe_sparsity_max, 4),
        "global_sparsity_max": round(global_sparsity_max, 4),
        "validations": validations,
    }


def _summarize_multimodal_binding_probe_catalog(
    concept_results: list[dict[str, object]],
) -> dict[str, object]:
    concept_count = len(concept_results)
    required_pass_count = max(1, math.ceil((2 * concept_count) / 3)) if concept_count else 0
    rows: list[dict[str, object]] = []
    passed_concept_count = 0

    for result in concept_results:
        concept_key = str(result.get("concept_key", "unknown"))
        summary = dict(result.get("summary", {}))
        validations = dict(summary.get("validations", {}))
        passed = bool(validations.get("passes_probe", False))
        if passed:
            passed_concept_count += 1
        rows.append(
            {
                "concept_key": concept_key,
                "bindings_formed_total": int(summary.get("bindings_formed_total", 0) or 0),
                "cross_modal_binding_count": int(summary.get("cross_modal_binding_count", 0) or 0),
                "successful_partner_recall_count": int(summary.get("successful_partner_recall_count", 0) or 0),
                "global_sparsity_max": round(float(summary.get("global_sparsity_max", 0.0) or 0.0), 4),
                "passed": passed,
            }
        )

    pass_rate = round(passed_concept_count / concept_count, 4) if concept_count else 0.0
    return {
        "concept_count": concept_count,
        "passed_concept_count": passed_concept_count,
        "required_pass_count": required_pass_count,
        "pass_rate": pass_rate,
        "passes_catalog_gate": passed_concept_count >= required_pass_count if concept_count else False,
        "per_concept_results": rows,
    }


def _run_single_multimodal_binding_probe(
    ticks_per_sample: int,
    train_repeats: int,
    threads: int,
    *,
    n_traces: int = 5500,
    seed_chunks: int | None = 1,
    rest_ticks: int = 1,
    settle_ticks: int = 3,
    probe_ticks: int = 4,
    cue_fraction: float = 1.0,
    spec: dict[str, object] | None = None,
) -> dict[str, object]:
    """Harvest one trace per modality, train cross-modal bindings, and probe recall."""
    if threads > 0:
        try:
            brain_core.set_num_threads(threads)
        except Exception:
            pass
    actual_threads = brain_core.get_num_threads()

    spec = dict(spec or DEFAULT_MULTIMODAL_BINDING_PROBE_SPEC)
    concept_key = str(spec["concept_key"])
    started = time.perf_counter()

    text_sample = _select_text_sample(dict(spec["text"]))
    visual_sample = _select_visual_sample(str(dict(spec["visual"])["label_name"]))
    audio_sample = _select_audio_sample(str(dict(spec["audio"])["label_name"]))

    text_trace, text_harvest = _harvest_text_primary_trace(
        text_sample,
        ticks_per_sample=ticks_per_sample,
        train_repeats=train_repeats,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        rest_ticks=rest_ticks,
    )
    visual_trace, visual_harvest = _harvest_visual_primary_trace(
        visual_sample,
        ticks_per_sample=ticks_per_sample,
        train_repeats=train_repeats,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        rest_ticks=rest_ticks,
    )
    audio_trace, audio_harvest = _harvest_audio_primary_trace(
        audio_sample,
        ticks_per_sample=ticks_per_sample,
        train_repeats=train_repeats,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        rest_ticks=rest_ticks,
    )

    harvested_traces = {
        "text": text_trace,
        "visual": visual_trace,
        "audio": audio_trace,
    }
    trace_store, traces_by_modality, modality_by_trace_id = _build_learned_only_trace_store(
        concept_key,
        harvested_traces,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
    )
    tick_loop = TickLoop(trace_store)

    training_rows: list[dict[str, object]] = []
    training_started = time.perf_counter()
    if len(traces_by_modality) == 3:
        for repeat_index in range(train_repeats):
            for tick_index in range(ticks_per_sample):
                cue_summary = _inject_trace_cues(
                    traces_by_modality,
                    1.0,
                    salt=f"train:{repeat_index}:{tick_index}",
                )
                result = tick_loop.step(allow_trace_formation=False, allow_binding_formation=True)
                active_patterns = _active_primary_patterns(trace_store, tick_loop.last_active_traces)
                diagnostics = _coactive_pair_diagnostics(active_patterns, modality_by_trace_id)
                training_rows.append(
                    {
                        "tick": int(result["tick"]),
                        "repeat_index": repeat_index,
                        "repeat_tick_index": tick_index,
                        "active_traces": int(result["active_traces"]),
                        "binding_candidates": int(result["binding_candidates"]),
                        "bindings_formed": int(result["bindings_formed"]),
                        "total_bindings": int(result["total_bindings"]),
                        "total_active": int(result["total_active"]),
                        "sparsity_fraction": round(float(result["total_active"]) / TOTAL_NEURONS, 6),
                        "step_internal_ms": round(float(result.get("step_internal_ms", 0.0) or 0.0), 4),
                        "cue_summary": cue_summary,
                        "active_trace_ids": [trace_id for trace_id, _score in tick_loop.last_active_traces],
                        "coactive_pair_diagnostics": diagnostics,
                    }
                )
            for _ in range(rest_ticks):
                tick_loop.step(learn=False, allow_trace_formation=False, allow_binding_formation=False)
            if repeat_index + 1 < train_repeats:
                tick_loop.reset_runtime_boundary(preserve_binding_state=True)

    binding_summary = _binding_rows(tick_loop, modality_by_trace_id)
    training_summary = _summarize_training_rows(training_rows)
    training_summary["training_wall_ms"] = round((time.perf_counter() - training_started) * 1000, 4)
    training_summary["step_internal_ms_total"] = round(
        sum(float(row["step_internal_ms"]) for row in training_rows),
        4,
    )

    recall_results: dict[str, dict[str, object]] = {}
    if len(traces_by_modality) == 3:
        for source_modality in ("text", "visual"):
            recall_results[source_modality] = _run_recall_probe(
                tick_loop,
                traces_by_modality,
                source_modality,
                cue_fraction=cue_fraction,
                settle_ticks=settle_ticks,
                probe_ticks=probe_ticks,
            )

    summary = _summarize_multimodal_binding_probe(
        binding_summary,
        training_summary,
        recall_results,
    )
    performance = {
        "text_harvest_wall_ms": round(float(text_harvest.get("training_wall_ms", 0.0) or 0.0), 4),
        "visual_harvest_wall_ms": round(float(visual_harvest.get("training_wall_ms", 0.0) or 0.0), 4),
        "audio_harvest_wall_ms": round(float(audio_harvest.get("training_wall_ms", 0.0) or 0.0), 4),
        "binding_training_wall_ms": round(float(training_summary.get("training_wall_ms", 0.0) or 0.0), 4),
        "binding_training_step_internal_ms_total": round(float(training_summary.get("step_internal_ms_total", 0.0) or 0.0), 4),
        "recall_probe_wall_ms_total": round(
            sum(float(result.get("probe_wall_ms", 0.0) or 0.0) for result in recall_results.values()),
            4,
        ),
        "recall_probe_step_internal_ms_total": round(
            sum(float(result.get("step_internal_ms_total", 0.0) or 0.0) for result in recall_results.values()),
            4,
        ),
        "total_wall_ms": round((time.perf_counter() - started) * 1000, 4),
    }

    result = {
        "benchmark": "multimodal_binding_probe",
        "concept_key": concept_key,
        "threads": actual_threads,
        "config": {
            "ticks_per_sample": ticks_per_sample,
            "train_repeats": train_repeats,
            "rest_ticks": rest_ticks,
            "settle_ticks": settle_ticks,
            "probe_ticks": probe_ticks,
            "cue_fraction": cue_fraction,
            "n_traces": n_traces,
            "seed_chunks": seed_chunks,
        },
        "concept_sources": {
            "text": {
                "dataset": spec["text"].get("dataset", "ag_news"),
                "sample_index": int(text_sample["index"]),
                "label_name": text_sample.get("label_name"),
                "text": str(text_sample["text"]),
            },
            "visual": {
                "dataset": spec["visual"].get("dataset", "cifar10"),
                "sample_index": int(visual_sample["index"]),
                "label_name": visual_sample.get("label_name"),
            },
            "audio": {
                "dataset": spec["audio"].get("dataset", "esc50"),
                "sample_index": int(audio_sample["index"]),
                "label_name": audio_sample.get("label_name"),
                "sample_rate": int(audio_sample["sample_rate"]),
            },
        },
        "harvest": {
            "text": text_harvest,
            "visual": visual_harvest,
            "audio": audio_harvest,
        },
        "learned_triplet": {
            modality: _trace_summary(trace)
            for modality, trace in sorted(traces_by_modality.items())
        },
        "binding_training": {
            **training_summary,
            "tick_rows": training_rows,
        },
        "bindings": binding_summary,
        "recall_probes": recall_results,
        "performance": performance,
        "summary": summary,
    }

    return result


def run_multimodal_binding_probe(
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
    cue_fraction: float = 1.0,
    spec: dict[str, object] | None = None,
    catalog: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Run the multimodal binding probe across a small concept catalog."""
    concept_specs = list(catalog or ([] if spec is None else [spec]) or DEFAULT_MULTIMODAL_BINDING_PROBE_CATALOG)
    concept_results: list[dict[str, object]] = []
    started = time.perf_counter()

    for concept_spec in concept_specs:
        concept_results.append(
            _run_single_multimodal_binding_probe(
                ticks_per_sample=ticks_per_sample,
                train_repeats=train_repeats,
                threads=threads,
                n_traces=n_traces,
                seed_chunks=seed_chunks,
                rest_ticks=rest_ticks,
                settle_ticks=settle_ticks,
                probe_ticks=probe_ticks,
                cue_fraction=cue_fraction,
                spec=concept_spec,
            )
        )

    aggregate = _summarize_multimodal_binding_probe_catalog(concept_results)
    result = {
        "benchmark": "multimodal_binding_probe",
        "threads": brain_core.get_num_threads(),
        "config": {
            "ticks_per_sample": ticks_per_sample,
            "train_repeats": train_repeats,
            "rest_ticks": rest_ticks,
            "settle_ticks": settle_ticks,
            "probe_ticks": probe_ticks,
            "cue_fraction": cue_fraction,
            "n_traces": n_traces,
            "seed_chunks": seed_chunks,
            "catalog_concepts": [str(item.get("concept_key", "unknown")) for item in concept_specs],
        },
        "concept_results": {
            str(concept_result["concept_key"]): concept_result
            for concept_result in concept_results
        },
        "aggregate": aggregate,
        "performance": {
            "total_wall_ms": round((time.perf_counter() - started) * 1000, 4),
            "concept_wall_ms_total": round(
                sum(
                    float(dict(concept_result.get("performance", {})).get("total_wall_ms", 0.0) or 0.0)
                    for concept_result in concept_results
                ),
                4,
            ),
        },
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result