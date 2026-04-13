"""Stable-regime cross-modal cue-only recall probe for one concept triplet."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import brain_core

from brain.datasets.downloader import load_audio_dataset
from brain.benchmarks.multimodal_binding_probe import (
    DEFAULT_MULTIMODAL_BINDING_PROBE_CATALOG,
    _active_primary_patterns,
    _binding_rows,
    _build_learned_only_trace_store,
    _build_trace_cue_injections,
    _coactive_pair_diagnostics,
    _harvest_audio_primary_trace,
    _harvest_text_primary_trace,
    _harvest_visual_primary_trace,
    _inject_trace_cues,
    _select_text_sample,
    _select_visual_sample,
    _summarize_training_rows,
    _trace_active_ratio,
    _trace_match_metrics,
    _trace_summary,
)
from brain.learning.tick_loop import TickLoop
from brain.structures.trace_store import Trace
from brain.utils.config import TOTAL_NEURONS


DEFAULT_CROSSMODAL_RECALL_PROBE_TRAIN_SAMPLES = 6
MAX_CROSSMODAL_RECALL_PROBE_TRAIN_SAMPLES = 10
DEFAULT_CROSSMODAL_RECALL_PROBE_SPARSITY_LIMIT = 0.08
_CROSSMODAL_RECALL_AUDIO_DATASET_SAMPLE_LIMIT = 2000
_CROSSMODAL_RECALL_AUDIO_CANDIDATE_LIMIT = 8
DEFAULT_CROSSMODAL_RECALL_PROBE_DIRECTIONS: tuple[tuple[str, str, str], ...] = (
    ("audio_to_text", "audio", "text"),
    ("text_to_audio", "text", "audio"),
    ("visual_to_audio", "visual", "audio"),
)


def _default_crossmodal_recall_probe_spec() -> dict[str, Any]:
    for spec in DEFAULT_MULTIMODAL_BINDING_PROBE_CATALOG:
        if str(spec.get("concept_key", "")) == "dog":
            return copy.deepcopy(dict(spec))
    raise ValueError("DEFAULT_MULTIMODAL_BINDING_PROBE_CATALOG is missing the dog concept")


DEFAULT_CROSSMODAL_RECALL_PROBE_SPEC: dict[str, Any] = _default_crossmodal_recall_probe_spec()


def _harvest_first_trainable_audio_trace(
    label_name: str,
    *,
    ticks_per_sample: int,
    train_repeats: int,
    n_traces: int,
    seed_chunks: int | None,
    rest_ticks: int,
    initial_state_dir: str | None = None,
) -> tuple[dict[str, object], Trace | None, dict[str, object]]:
    dataset = load_audio_dataset(
        "esc50",
        max_samples=_CROSSMODAL_RECALL_AUDIO_DATASET_SAMPLE_LIMIT,
    )
    attempts: list[dict[str, object]] = []
    selected_sample: dict[str, object] | None = None
    selected_trace: Trace | None = None
    selected_harvest: dict[str, object] = {
        "sample_index": None,
        "sample_key": label_name,
        "label_name": label_name,
        "quality_summary": {},
        "learned_trace_count": 0,
        "primary_trace": {
            "present": False,
            "trace_id": None,
            "label": None,
            "total_neurons": 0,
            "regions": {},
        },
        "training_wall_ms": 0.0,
    }

    candidate_count = 0
    for index, sample in enumerate(dataset):
        if str(sample.get("label_name", "")) != label_name:
            continue

        candidate = {
            "index": index,
            "key": f"{label_name}_{index}",
            "label_name": label_name,
            "label": sample.get("label"),
            "audio": sample["audio"],
            "sample_rate": int(sample.get("sample_rate", 16000) or 16000),
        }
        candidate_trace, candidate_harvest = _harvest_audio_primary_trace(
            candidate,
            ticks_per_sample=ticks_per_sample,
            train_repeats=train_repeats,
            n_traces=n_traces,
            seed_chunks=seed_chunks,
            rest_ticks=rest_ticks,
            initial_state_dir=initial_state_dir,
        )
        attempts.append(
            {
                "sample_index": int(candidate["index"]),
                "sample_key": str(candidate["key"]),
                "learned_trace_count": int(candidate_harvest.get("learned_trace_count", 0) or 0),
                "primary_trace_present": bool(
                    dict(candidate_harvest.get("primary_trace", {})).get("present", False)
                ),
            }
        )
        selected_sample = candidate
        selected_trace = candidate_trace
        selected_harvest = dict(candidate_harvest)
        candidate_count += 1
        if candidate_trace is not None or candidate_count >= _CROSSMODAL_RECALL_AUDIO_CANDIDATE_LIMIT:
            break

    if selected_sample is None:
        raise ValueError(f"Missing required audio label in ESC-50 slice: {label_name}")

    selected_harvest["candidate_attempts"] = attempts
    selected_harvest["attempted_sample_count"] = len(attempts)
    return selected_sample, selected_trace, selected_harvest


def _empty_direction_result(
    direction_key: str,
    source_modality: str,
    target_modality: str,
    *,
    reason: str,
) -> dict[str, object]:
    return {
        "direction_key": direction_key,
        "source_modality": source_modality,
        "target_modality": target_modality,
        "executed": False,
        "skip_reason": reason,
        "cue_summary": {},
        "source_trace_hit_rate": 0.0,
        "partner_trace_hit_rate": 0.0,
        "partner_trace_rank_avg": None,
        "partner_trace_score_max": 0.0,
        "partner_trace_active_ratio_avg": 0.0,
        "partner_trace_active_ratio_max": 0.0,
        "sparsity_max": 0.0,
        "step_internal_ms_total": 0.0,
        "probe_wall_ms": 0.0,
        "passed": False,
        "tick_rows": [],
    }


def _run_direction_probe(
    tick_loop: TickLoop,
    traces_by_modality: dict[str, Trace],
    direction_key: str,
    source_modality: str,
    target_modality: str,
    *,
    cue_fraction: float,
    settle_ticks: int,
    probe_ticks: int,
) -> dict[str, object]:
    source_trace = traces_by_modality[source_modality]
    target_trace = traces_by_modality[target_modality]
    cue_injections, cue_summary = _build_trace_cue_injections(
        source_trace,
        cue_fraction,
        salt=f"crossmodal_recall:{direction_key}",
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
        source_hit, source_rank, source_score = _trace_match_metrics(
            tick_loop.last_active_traces,
            source_trace.id,
        )
        partner_hit, partner_rank, partner_score = _trace_match_metrics(
            tick_loop.last_active_traces,
            target_trace.id,
        )
        top_trace_id = tick_loop.last_active_traces[0][0] if tick_loop.last_active_traces else None
        top_trace_score = float(tick_loop.last_active_traces[0][1]) if tick_loop.last_active_traces else 0.0

        tick_rows.append(
            {
                "tick": int(result["tick"]),
                "active_traces": int(result["active_traces"]),
                "total_active": int(result["total_active"]),
                "total_bindings": int(result["total_bindings"]),
                "sparsity_fraction": round(float(result["total_active"]) / TOTAL_NEURONS, 6),
                "step_internal_ms": round(float(result.get("step_internal_ms", 0.0) or 0.0), 4),
                "source_trace_hit": source_hit,
                "source_trace_rank": source_rank,
                "source_trace_score": round(float(source_score), 4),
                "partner_trace_hit": partner_hit,
                "partner_trace_rank": partner_rank,
                "partner_trace_score": round(float(partner_score), 4),
                "partner_trace_active_ratio": _trace_active_ratio(tick_loop, target_trace),
                "top_trace_id": top_trace_id,
                "top_trace_score": round(top_trace_score, 4),
            }
        )

    partner_ranks = [
        int(row["partner_trace_rank"])
        for row in tick_rows
        if row["partner_trace_rank"] is not None
    ]
    partner_active_ratios = [float(row["partner_trace_active_ratio"]) for row in tick_rows]
    partner_hit_rate = round(
        sum(1 for row in tick_rows if row["partner_trace_hit"]) / max(1, len(tick_rows)),
        4,
    )

    return {
        "direction_key": direction_key,
        "source_modality": source_modality,
        "target_modality": target_modality,
        "executed": True,
        "cue_summary": cue_summary,
        "source_trace_id": source_trace.id,
        "partner_trace_id": target_trace.id,
        "source_trace_hit_rate": round(
            sum(1 for row in tick_rows if row["source_trace_hit"]) / max(1, len(tick_rows)),
            4,
        ),
        "partner_trace_hit_rate": partner_hit_rate,
        "partner_trace_rank_avg": round(sum(partner_ranks) / len(partner_ranks), 4) if partner_ranks else None,
        "partner_trace_score_max": round(
            max(float(row["partner_trace_score"]) for row in tick_rows),
            4,
        ) if tick_rows else 0.0,
        "partner_trace_active_ratio_avg": round(
            sum(partner_active_ratios) / len(partner_active_ratios),
            4,
        ) if partner_active_ratios else 0.0,
        "partner_trace_active_ratio_max": round(max(partner_active_ratios), 4) if partner_active_ratios else 0.0,
        "sparsity_max": round(
            max(float(row["sparsity_fraction"]) for row in tick_rows),
            4,
        ) if tick_rows else 0.0,
        "step_internal_ms_total": round(
            sum(float(row["step_internal_ms"]) for row in tick_rows),
            4,
        ),
        "probe_wall_ms": round((time.perf_counter() - started) * 1000, 4),
        "passed": partner_hit_rate > 0.0,
        "tick_rows": tick_rows,
    }


def _run_training_samples(
    tick_loop: TickLoop,
    traces_by_modality: dict[str, Trace],
    modality_by_trace_id: dict[str, str],
    *,
    train_samples: int,
    ticks_per_sample: int,
    rest_ticks: int,
    sparsity_limit: float,
) -> dict[str, object]:
    training_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    training_started = time.perf_counter()
    training_stopped_early = False
    stop_reason: str | None = None
    stop_sample_index: int | None = None
    stop_tick_index: int | None = None

    for sample_index in range(train_samples):
        sample_tick_rows: list[dict[str, object]] = []
        sample_started = time.perf_counter()

        for tick_index in range(ticks_per_sample):
            cue_summary = _inject_trace_cues(
                traces_by_modality,
                1.0,
                salt=f"crossmodal_train:{sample_index}:{tick_index}",
            )
            result = tick_loop.step(allow_trace_formation=False, allow_binding_formation=True)
            active_patterns = _active_primary_patterns(
                trace_store=tick_loop.trace_store,
                active_traces=tick_loop.last_active_traces,
            )
            diagnostics = _coactive_pair_diagnostics(active_patterns, modality_by_trace_id)
            row = {
                "sample_index": sample_index,
                "sample_tick_index": tick_index,
                "tick": int(result["tick"]),
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
            training_rows.append(row)
            sample_tick_rows.append(row)

            if float(row["sparsity_fraction"]) > sparsity_limit:
                training_stopped_early = True
                stop_reason = "sparsity_limit_exceeded"
                stop_sample_index = sample_index
                stop_tick_index = tick_index
                break

        sample_rows.append(
            {
                "sample_index": sample_index,
                "tick_count": len(sample_tick_rows),
                "bindings_formed_total": sum(int(row["bindings_formed"]) for row in sample_tick_rows),
                "binding_candidates_avg": round(
                    sum(float(row["binding_candidates"]) for row in sample_tick_rows) / len(sample_tick_rows),
                    4,
                ) if sample_tick_rows else 0.0,
                "active_traces_avg": round(
                    sum(float(row["active_traces"]) for row in sample_tick_rows) / len(sample_tick_rows),
                    4,
                ) if sample_tick_rows else 0.0,
                "sparsity_max": round(
                    max(float(row["sparsity_fraction"]) for row in sample_tick_rows),
                    4,
                ) if sample_tick_rows else 0.0,
                "stopped_early": training_stopped_early and stop_sample_index == sample_index,
                "wall_ms": round((time.perf_counter() - sample_started) * 1000, 4),
            }
        )

        if training_stopped_early:
            break

        for _ in range(rest_ticks):
            tick_loop.step(learn=False, allow_trace_formation=False, allow_binding_formation=False)
        if sample_index + 1 < train_samples:
            tick_loop.reset_runtime_boundary(preserve_binding_state=True)

    training_summary = _summarize_training_rows(training_rows)
    training_summary.update(
        {
            "train_samples_requested": int(train_samples),
            "train_samples_completed": len(sample_rows),
            "training_stopped_early": training_stopped_early,
            "stop_reason": stop_reason,
            "stop_sample_index": stop_sample_index,
            "stop_tick_index": stop_tick_index,
            "sample_rows": sample_rows,
            "training_wall_ms": round((time.perf_counter() - training_started) * 1000, 4),
            "step_internal_ms_total": round(
                sum(float(row["step_internal_ms"]) for row in training_rows),
                4,
            ),
        }
    )
    return {
        "summary": training_summary,
        "tick_rows": training_rows,
    }


def _summarize_crossmodal_recall_probe(
    *,
    all_modalities_harvested: bool,
    binding_summary: dict[str, object],
    training_summary: dict[str, object],
    cue_direction_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    direction_rows = {
        direction_key: {
            "source_modality": str(direction_result.get("source_modality", "unknown")),
            "target_modality": str(direction_result.get("target_modality", "unknown")),
            "partner_trace_hit_rate": round(
                float(direction_result.get("partner_trace_hit_rate", 0.0) or 0.0),
                4,
            ),
            "passed": bool(direction_result.get("passed", False)),
        }
        for direction_key, direction_result in sorted(cue_direction_results.items())
    }
    passed_direction_count = sum(1 for row in direction_rows.values() if row["passed"])
    training_sparsity_max = round(float(training_summary.get("sparsity_max", 0.0) or 0.0), 4)
    completed_training_samples = int(training_summary.get("train_samples_completed", 0) or 0)
    requested_training_samples = int(training_summary.get("train_samples_requested", 0) or 0)
    training_stopped_early = bool(training_summary.get("training_stopped_early", False))
    probe_sparsity_max = round(
        max(
            [float(direction_result.get("sparsity_max", 0.0) or 0.0) for direction_result in cue_direction_results.values()]
            or [0.0]
        ),
        4,
    )
    cross_modal_binding_count = int(binding_summary.get("cross_modal_count", 0) or 0)

    validations = {
        "harvested_all_modalities": all_modalities_harvested,
        "completed_training_samples": completed_training_samples == requested_training_samples,
        "cross_modal_binding_formed": cross_modal_binding_count >= 1,
        "training_sparsity_under_8_percent": training_sparsity_max <= DEFAULT_CROSSMODAL_RECALL_PROBE_SPARSITY_LIMIT,
        "two_of_three_cue_directions": passed_direction_count >= 2,
    }
    validations["passes_probe"] = all(validations.values()) and not training_stopped_early

    return {
        "train_samples_requested": requested_training_samples,
        "train_samples_completed": completed_training_samples,
        "training_stopped_early": training_stopped_early,
        "training_stop_reason": training_summary.get("stop_reason"),
        "bindings_formed_total": int(training_summary.get("bindings_formed_total", 0) or 0),
        "cross_modal_binding_count": cross_modal_binding_count,
        "training_sparsity_max": training_sparsity_max,
        "probe_sparsity_max": probe_sparsity_max,
        "passed_direction_count": passed_direction_count,
        "required_direction_pass_count": 2,
        "cue_direction_results": direction_rows,
        "validations": validations,
    }


def run_crossmodal_recall_probe(
    ticks_per_sample: int,
    threads: int,
    output_path: str,
    *,
    train_samples: int = DEFAULT_CROSSMODAL_RECALL_PROBE_TRAIN_SAMPLES,
    n_traces: int = 5500,
    seed_chunks: int | None = 1,
    rest_ticks: int = 1,
    settle_ticks: int = 3,
    probe_ticks: int = 4,
    cue_fraction: float = 1.0,
    sparsity_limit: float = DEFAULT_CROSSMODAL_RECALL_PROBE_SPARSITY_LIMIT,
    spec: dict[str, object] | None = None,
    initial_state_dir: str | None = None,
) -> dict[str, object]:
    """Run the stable-regime cue-only recall probe for a single multimodal concept."""
    if train_samples < 1:
        raise ValueError("crossmodal_recall_probe requires at least one training sample")
    if train_samples > MAX_CROSSMODAL_RECALL_PROBE_TRAIN_SAMPLES:
        raise ValueError(
            f"crossmodal_recall_probe refuses to run beyond {MAX_CROSSMODAL_RECALL_PROBE_TRAIN_SAMPLES} training samples"
        )
    if threads > 0:
        try:
            brain_core.set_num_threads(threads)
        except Exception:
            pass
    actual_threads = brain_core.get_num_threads()

    spec = copy.deepcopy(spec or DEFAULT_CROSSMODAL_RECALL_PROBE_SPEC)
    concept_key = str(spec["concept_key"])
    started = time.perf_counter()

    text_sample = _select_text_sample(dict(spec["text"]))
    visual_sample = _select_visual_sample(str(dict(spec["visual"])["label_name"]))
    audio_sample, audio_trace, audio_harvest = _harvest_first_trainable_audio_trace(
        str(dict(spec["audio"])["label_name"]),
        ticks_per_sample=ticks_per_sample,
        train_repeats=train_samples,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        rest_ticks=rest_ticks,
        initial_state_dir=initial_state_dir,
    )

    text_trace, text_harvest = _harvest_text_primary_trace(
        text_sample,
        ticks_per_sample=ticks_per_sample,
        train_repeats=train_samples,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        rest_ticks=rest_ticks,
        initial_state_dir=initial_state_dir,
    )
    visual_trace, visual_harvest = _harvest_visual_primary_trace(
        visual_sample,
        ticks_per_sample=ticks_per_sample,
        train_repeats=train_samples,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        rest_ticks=rest_ticks,
        initial_state_dir=initial_state_dir,
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
        initial_state_dir=initial_state_dir,
    )
    tick_loop = TickLoop(trace_store)
    all_modalities_harvested = len(traces_by_modality) == 3

    training_rows: list[dict[str, object]] = []
    training_summary: dict[str, object] = {
        "train_samples_requested": int(train_samples),
        "train_samples_completed": 0,
        "training_stopped_early": False,
        "stop_reason": "missing_modalities" if not all_modalities_harvested else None,
        "stop_sample_index": None,
        "stop_tick_index": None,
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
        "sample_rows": [],
        "training_wall_ms": 0.0,
        "step_internal_ms_total": 0.0,
    }
    binding_summary: dict[str, object] = {"count": 0, "cross_modal_count": 0, "rows": []}
    cue_direction_results = {
        direction_key: _empty_direction_result(
            direction_key,
            source_modality,
            target_modality,
            reason="training_not_run",
        )
        for direction_key, source_modality, target_modality in DEFAULT_CROSSMODAL_RECALL_PROBE_DIRECTIONS
    }

    if all_modalities_harvested:
        training = _run_training_samples(
            tick_loop,
            traces_by_modality,
            modality_by_trace_id,
            train_samples=train_samples,
            ticks_per_sample=ticks_per_sample,
            rest_ticks=rest_ticks,
            sparsity_limit=sparsity_limit,
        )
        training_rows = training["tick_rows"]
        training_summary = dict(training["summary"])
        binding_summary = _binding_rows(tick_loop, modality_by_trace_id)

        if not bool(training_summary.get("training_stopped_early", False)):
            cue_direction_results = {}
            for direction_key, source_modality, target_modality in DEFAULT_CROSSMODAL_RECALL_PROBE_DIRECTIONS:
                cue_direction_results[direction_key] = _run_direction_probe(
                    tick_loop,
                    traces_by_modality,
                    direction_key,
                    source_modality,
                    target_modality,
                    cue_fraction=cue_fraction,
                    settle_ticks=settle_ticks,
                    probe_ticks=probe_ticks,
                )
        else:
            cue_direction_results = {
                direction_key: _empty_direction_result(
                    direction_key,
                    source_modality,
                    target_modality,
                    reason="training_stopped_early",
                )
                for direction_key, source_modality, target_modality in DEFAULT_CROSSMODAL_RECALL_PROBE_DIRECTIONS
            }

    summary = _summarize_crossmodal_recall_probe(
        all_modalities_harvested=all_modalities_harvested,
        binding_summary=binding_summary,
        training_summary=training_summary,
        cue_direction_results=cue_direction_results,
    )
    performance = {
        "text_harvest_wall_ms": round(float(text_harvest.get("training_wall_ms", 0.0) or 0.0), 4),
        "visual_harvest_wall_ms": round(float(visual_harvest.get("training_wall_ms", 0.0) or 0.0), 4),
        "audio_harvest_wall_ms": round(float(audio_harvest.get("training_wall_ms", 0.0) or 0.0), 4),
        "binding_training_wall_ms": round(float(training_summary.get("training_wall_ms", 0.0) or 0.0), 4),
        "binding_training_step_internal_ms_total": round(float(training_summary.get("step_internal_ms_total", 0.0) or 0.0), 4),
        "cue_probe_wall_ms_total": round(
            sum(float(result.get("probe_wall_ms", 0.0) or 0.0) for result in cue_direction_results.values()),
            4,
        ),
        "cue_probe_step_internal_ms_total": round(
            sum(float(result.get("step_internal_ms_total", 0.0) or 0.0) for result in cue_direction_results.values()),
            4,
        ),
        "total_wall_ms": round((time.perf_counter() - started) * 1000, 4),
    }

    result = {
        "benchmark": "crossmodal_recall_probe",
        "concept_key": concept_key,
        "threads": actual_threads,
        "config": {
            "train_samples": int(train_samples),
            "max_train_samples": int(MAX_CROSSMODAL_RECALL_PROBE_TRAIN_SAMPLES),
            "ticks_per_sample": ticks_per_sample,
            "rest_ticks": rest_ticks,
            "settle_ticks": settle_ticks,
            "probe_ticks": probe_ticks,
            "cue_fraction": cue_fraction,
            "sparsity_limit": sparsity_limit,
            "n_traces": n_traces,
            "seed_chunks": seed_chunks,
            "initial_state_dir": initial_state_dir,
            "cue_directions": [
                {
                    "direction_key": direction_key,
                    "source_modality": source_modality,
                    "target_modality": target_modality,
                }
                for direction_key, source_modality, target_modality in DEFAULT_CROSSMODAL_RECALL_PROBE_DIRECTIONS
            ],
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
        "cue_directions": cue_direction_results,
        "performance": performance,
        "summary": summary,
    }

    Path(output_path).write_text(json.dumps(result, indent=2))
    return result


__all__ = [
    "DEFAULT_CROSSMODAL_RECALL_PROBE_DIRECTIONS",
    "DEFAULT_CROSSMODAL_RECALL_PROBE_SPEC",
    "DEFAULT_CROSSMODAL_RECALL_PROBE_SPARSITY_LIMIT",
    "DEFAULT_CROSSMODAL_RECALL_PROBE_TRAIN_SAMPLES",
    "MAX_CROSSMODAL_RECALL_PROBE_TRAIN_SAMPLES",
    "run_crossmodal_recall_probe",
]