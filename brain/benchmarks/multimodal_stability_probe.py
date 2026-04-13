"""Long-run multimodal stability probe with compressed lifecycle validation."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Iterator

import brain_core

from brain.datasets.downloader import (
    load_audio_dataset,
    load_image_dataset,
    load_text_dataset,
)
from brain.input.audio_input import AudioInput
from brain.input.multimodal import MultimodalInput
from brain.input.sensory_input import SensoryInput
from brain.input.text_input import TextInput
from brain.input.visual_input import VisualInput
from brain.learning.tick_loop import TickLoop
from brain.seed.seed_runner import seed_brain, seed_brain_fast
from brain.serialize.runtime_bundle import load_runtime_bundle
from brain.structures.trace_store import TraceStore
from brain.utils.config import TOTAL_NEURONS


DEFAULT_MULTIMODAL_STABILITY_TEXT_DATASET = "ag_news"
DEFAULT_MULTIMODAL_STABILITY_IMAGE_DATASET = "cifar10"
DEFAULT_MULTIMODAL_STABILITY_AUDIO_DATASET = "esc50"
DEFAULT_MULTIMODAL_STABILITY_TICK_BATCH_SIZE = 1
_DEFAULT_PRUNE_PASS_TARGET = 6


@contextmanager
def _compressed_lifecycle(schedule: dict[str, int]) -> Iterator[None]:
    import brain.learning.anti_hebbian as anti_hebbian
    import brain.learning.hebbian as hebbian
    import brain.learning.pruning as pruning

    originals = {
        "hebbian_bloom": hebbian.BLOOM_END_TICK,
        "hebbian_critical": hebbian.CRITICAL_END_TICK,
        "anti_bloom": anti_hebbian.BLOOM_END_TICK,
        "anti_critical": anti_hebbian.CRITICAL_END_TICK,
        "pruning_bloom": pruning.BLOOM_END_TICK,
        "pruning_critical": pruning.CRITICAL_END_TICK,
        "critical_dormant": pruning.CRITICAL_PRUNE_DORMANT_TICKS,
        "mature_dormant": pruning.MATURE_PRUNE_DORMANT_TICKS,
    }

    hebbian.BLOOM_END_TICK = schedule["bloom_end_tick"]
    hebbian.CRITICAL_END_TICK = schedule["critical_end_tick"]
    anti_hebbian.BLOOM_END_TICK = schedule["bloom_end_tick"]
    anti_hebbian.CRITICAL_END_TICK = schedule["critical_end_tick"]
    pruning.BLOOM_END_TICK = schedule["bloom_end_tick"]
    pruning.CRITICAL_END_TICK = schedule["critical_end_tick"]
    pruning.CRITICAL_PRUNE_DORMANT_TICKS = schedule["critical_dormant_ticks"]
    pruning.MATURE_PRUNE_DORMANT_TICKS = schedule["mature_dormant_ticks"]

    try:
        yield
    finally:
        hebbian.BLOOM_END_TICK = originals["hebbian_bloom"]
        hebbian.CRITICAL_END_TICK = originals["hebbian_critical"]
        anti_hebbian.BLOOM_END_TICK = originals["anti_bloom"]
        anti_hebbian.CRITICAL_END_TICK = originals["anti_critical"]
        pruning.BLOOM_END_TICK = originals["pruning_bloom"]
        pruning.CRITICAL_END_TICK = originals["pruning_critical"]
        pruning.CRITICAL_PRUNE_DORMANT_TICKS = originals["critical_dormant"]
        pruning.MATURE_PRUNE_DORMANT_TICKS = originals["mature_dormant"]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _executed_ticks(row: dict[str, Any]) -> int:
    return max(1, int(row.get("executed_ticks", 1) or 1))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_pickle_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    tmp_path.replace(path)


def _read_pickle(path: Path) -> Any:
    return pickle.loads(path.read_bytes())


def _write_brain_checkpoint_atomic(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    brain_core.save_brain_checkpoint(str(tmp_path))
    tmp_path.replace(path)


def _checkpoint_run_id(config: dict[str, Any]) -> str:
    digest = hashlib.sha1(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"multimodal_{digest[:12]}"


def _checkpoint_paths(run_id: str) -> dict[str, Path]:
    checkpoint_dir = Path("results") / "checkpoints" / run_id
    return {
        "dir": checkpoint_dir,
        "progress": checkpoint_dir / "progress.json",
        "brain": checkpoint_dir / "brain_state.bin",
        "trace_store": checkpoint_dir / "trace_store.json",
        "python_state": checkpoint_dir / "python_state.pkl",
        "result": checkpoint_dir / "result.json",
    }


def _save_checkpoint(
    paths: dict[str, Path],
    *,
    config: dict[str, Any],
    schedule: dict[str, int],
    tick_loop: TickLoop,
    multimodal: MultimodalInput,
    trace_store,
    tick_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    next_sample_index: int,
    accumulated_wall_ms: float,
) -> None:
    _write_brain_checkpoint_atomic(paths["brain"])
    trace_store.save(str(paths["trace_store"]))
    _write_pickle_atomic(
        paths["python_state"],
        {
            "tick_loop": tick_loop.export_checkpoint_state(),
            "multimodal": multimodal.checkpoint_state(),
        },
    )
    _write_json_atomic(
        paths["progress"],
        {
            "version": 1,
            "status": "running",
            "config": dict(config),
            "schedule": dict(schedule),
            "next_sample_index": int(next_sample_index),
            "completed_samples": int(len(sample_rows)),
            "accumulated_wall_ms": _round(accumulated_wall_ms, 4),
            "tick_rows": tick_rows,
            "sample_rows": sample_rows,
        },
    )


def _load_checkpoint(
    paths: dict[str, Path],
    *,
    expected_config: dict[str, Any],
    schedule: dict[str, int],
) -> dict[str, Any] | None:
    progress_path = paths["progress"]
    if not progress_path.exists():
        return None

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if dict(progress.get("config", {})) != dict(expected_config):
        return None
    if dict(progress.get("schedule", {})) != dict(schedule):
        return None

    required_paths = (paths["brain"], paths["trace_store"], paths["python_state"])
    if not all(path.exists() for path in required_paths):
        return None

    return progress


def _build_multimodal_samples(
    max_samples: int,
    *,
    text_name: str = DEFAULT_MULTIMODAL_STABILITY_TEXT_DATASET,
    image_name: str = DEFAULT_MULTIMODAL_STABILITY_IMAGE_DATASET,
    audio_name: str = DEFAULT_MULTIMODAL_STABILITY_AUDIO_DATASET,
) -> list[dict[str, Any]]:
    texts = load_text_dataset(text_name, max_samples=max_samples)
    images = load_image_dataset(image_name, max_samples=max_samples)
    audios = load_audio_dataset(audio_name, max_samples=max_samples)

    count = min(len(texts), len(images), len(audios), max_samples)
    samples: list[dict[str, Any]] = []
    for index in range(count):
        samples.append(
            {
                "text": texts[index]["text"],
                "text_label": texts[index]["label_name"],
                "image": images[index]["image"],
                "image_label": images[index]["label_name"],
                "audio": audios[index]["audio"],
                "audio_sample_rate": audios[index]["sample_rate"],
                "audio_label": audios[index]["label_name"],
            }
        )
    return samples


def _build_lifecycle_schedule(total_run_ticks: int) -> dict[str, int]:
    total_run_ticks = max(3, int(total_run_ticks))

    bloom_end_tick = max(1, total_run_ticks // 3)
    critical_end_tick = max(bloom_end_tick + 1, (2 * total_run_ticks) // 3)
    critical_end_tick = min(critical_end_tick, total_run_ticks - 1)

    prune_interval = max(1, total_run_ticks // _DEFAULT_PRUNE_PASS_TARGET)
    estimated_prune_pass_count = max(1, math.ceil(total_run_ticks / prune_interval))
    prune_batch_size = max(1, math.ceil(TOTAL_NEURONS / estimated_prune_pass_count))

    binding_maintenance_interval = max(1, total_run_ticks // 3)
    rebuild_interval = max(binding_maintenance_interval, total_run_ticks // 2)

    critical_dormant_ticks = max(4, total_run_ticks // 12)
    mature_dormant_ticks = max(critical_dormant_ticks + 1, total_run_ticks // 6)

    return {
        "total_run_ticks": total_run_ticks,
        "bloom_end_tick": bloom_end_tick,
        "critical_end_tick": critical_end_tick,
        "critical_dormant_ticks": critical_dormant_ticks,
        "mature_dormant_ticks": mature_dormant_ticks,
        "prune_interval": prune_interval,
        "estimated_prune_pass_count": estimated_prune_pass_count,
        "prune_batch_size": prune_batch_size,
        "binding_maintenance_interval": binding_maintenance_interval,
        "rebuild_interval": rebuild_interval,
    }


def _tick_row(
    result: dict[str, Any],
    *,
    sample_index: int,
    sample_label: str,
    tick_kind: str,
    local_tick_index: int,
    synapse_count: int,
) -> dict[str, Any]:
    return {
        "tick": int(result["tick"]),
        "executed_ticks": int(result.get("executed_ticks", 1) or 1),
        "sample_index": sample_index,
        "sample_label": sample_label,
        "tick_kind": tick_kind,
        "local_tick_index": local_tick_index,
        "phase": str(result.get("phase", "unknown")),
        "total_active": int(result.get("total_active", 0)),
        "sparsity_fraction": _round(float(result.get("total_active", 0)) / TOTAL_NEURONS, 6),
        "active_traces": int(result.get("active_traces", 0)),
        "trace_candidates": int(result.get("trace_candidates", 0)),
        "traces_formed": int(result.get("traces_formed", 0)),
        "binding_candidates": int(result.get("binding_candidates", 0)),
        "bindings_formed": int(result.get("bindings_formed", 0)),
        "total_bindings": int(result.get("total_bindings", 0)),
        "visual_activation": _round(float(result.get("visual_activation", 0.0)), 6),
        "audio_activation": _round(float(result.get("audio_activation", 0.0)), 6),
        "language_activation": _round(float(result.get("language_activation", 0.0)), 6),
        "step_internal_ms": _round(float(result.get("step_internal_ms", 0.0)), 4),
        "synapse_count": int(synapse_count),
        "synapse_pruned_count": int(result.get("synapse_pruned_count", 0) or 0),
        "binding_pruned_count": int(result.get("binding_pruned_count", 0) or 0),
        "coactivation_cleanup_ran": int(result.get("coactivation_cleanup_ran", 0) or 0),
        "synapse_rebuild_ran": int(result.get("synapse_rebuild_ran", 0) or 0),
        "synapse_update_pending_count": int(result.get("synapse_update_pending_count", 0) or 0),
        "synapse_update_applied_count": int(result.get("synapse_update_applied_count", 0) or 0),
    }


def _sample_summary(
    rows: list[dict[str, Any]],
    *,
    sample_index: int,
    sample_label: str,
    synapse_count_start: int,
    synapse_count_end: int,
) -> dict[str, Any]:
    active_rows = [row for row in rows if row["tick_kind"] == "train"]
    sparsity_values = [float(row["sparsity_fraction"]) for row in active_rows]
    total_active_values = [float(row["total_active"]) for row in active_rows]

    return {
        "sample_index": sample_index,
        "sample_label": sample_label,
        "tick_count": sum(_executed_ticks(row) for row in rows),
        "train_tick_count": sum(_executed_ticks(row) for row in active_rows),
        "phase_start": rows[0]["phase"],
        "phase_end": rows[-1]["phase"],
        "sparsity_avg": _round(_mean(sparsity_values), 6),
        "sparsity_max": _round(max(sparsity_values) if sparsity_values else 0.0, 6),
        "total_active_avg": _round(_mean(total_active_values), 2),
        "synapse_count_start": int(synapse_count_start),
        "synapse_count_end": int(synapse_count_end),
        "synapse_count_delta": int(synapse_count_end - synapse_count_start),
        "synapse_pruned_total": int(sum(int(row["synapse_pruned_count"]) for row in rows)),
    }


def _summarize_multimodal_stability_probe(
    tick_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    schedule: dict[str, int],
) -> dict[str, Any]:
    active_rows = [row for row in tick_rows if row["tick_kind"] == "train"]
    window_size = max(1, math.ceil(len(active_rows) / 3)) if active_rows else 1
    early_rows = active_rows[:window_size]
    late_rows = active_rows[-window_size:]

    phase_tick_counts: Counter[str] = Counter()
    for row in tick_rows:
        phase_tick_counts[str(row["phase"])] += _executed_ticks(row)
    phase_transition_ticks: dict[str, int] = {}
    for phase_name in ("bloom", "critical", "mature"):
        for row in tick_rows:
            if row["phase"] == phase_name:
                phase_transition_ticks[phase_name] = int(row["tick"])
                break

    pruning_rows = [row for row in tick_rows if int(row["synapse_pruned_count"]) > 0]
    critical_pruning_rows = [row for row in pruning_rows if row["phase"] == "critical"]
    mature_pruning_rows = [row for row in pruning_rows if row["phase"] == "mature"]
    binding_pruning_rows = [row for row in tick_rows if int(row["binding_pruned_count"]) > 0]
    rebuild_rows = [row for row in tick_rows if int(row["synapse_rebuild_ran"]) > 0]

    def row_mean(rows: list[dict[str, Any]], key: str, digits: int = 4) -> float:
        weighted_vals = [
            (float(row.get(key, 0.0) or 0.0), _executed_ticks(row))
            for row in rows
        ]
        if not weighted_vals:
            return 0.0
        return _round(
            sum(value * weight for value, weight in weighted_vals)
            / sum(weight for _value, weight in weighted_vals),
            digits,
        )

    def row_max(rows: list[dict[str, Any]], key: str, digits: int = 4) -> float:
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        return _round(max(values) if values else 0.0, digits)

    late_modality_activation_avg = {
        "visual": row_mean(late_rows, "visual_activation", 6),
        "audio": row_mean(late_rows, "audio_activation", 6),
        "language": row_mean(late_rows, "language_activation", 6),
    }

    stability = {
        "sparsity_avg": row_mean(active_rows, "sparsity_fraction", 6),
        "sparsity_max": row_max(active_rows, "sparsity_fraction", 6),
        "early_run_sparsity_avg": row_mean(early_rows, "sparsity_fraction", 6),
        "late_run_sparsity_avg": row_mean(late_rows, "sparsity_fraction", 6),
        "late_run_sparsity_max": row_max(late_rows, "sparsity_fraction", 6),
        "early_run_total_active_avg": row_mean(early_rows, "total_active", 2),
        "late_run_total_active_avg": row_mean(late_rows, "total_active", 2),
        "early_run_trace_candidates_avg": row_mean(early_rows, "trace_candidates", 2),
        "late_run_trace_candidates_avg": row_mean(late_rows, "trace_candidates", 2),
        "early_run_total_bindings_avg": row_mean(early_rows, "total_bindings", 2),
        "late_run_total_bindings_avg": row_mean(late_rows, "total_bindings", 2),
        "early_run_pending_updates_avg": row_mean(early_rows, "synapse_update_pending_count", 2),
        "late_run_pending_updates_avg": row_mean(late_rows, "synapse_update_pending_count", 2),
        "late_run_modality_activation_avg": late_modality_activation_avg,
    }

    pruning = {
        "synapse_count_start": int(tick_rows[0]["synapse_count"]) if tick_rows else 0,
        "synapse_count_end": int(tick_rows[-1]["synapse_count"]) if tick_rows else 0,
        "synapse_count_min": min((int(row["synapse_count"]) for row in tick_rows), default=0),
        "synapse_count_max": max((int(row["synapse_count"]) for row in tick_rows), default=0),
        "synapse_count_delta": (
            int(tick_rows[-1]["synapse_count"]) - int(tick_rows[0]["synapse_count"])
            if tick_rows
            else 0
        ),
        "synapse_pruned_total": int(sum(int(row["synapse_pruned_count"]) for row in tick_rows)),
        "pruning_event_count": len(pruning_rows),
        "critical_phase_pruned_total": int(sum(int(row["synapse_pruned_count"]) for row in critical_pruning_rows)),
        "critical_phase_pruning_event_count": len(critical_pruning_rows),
        "mature_phase_pruned_total": int(sum(int(row["synapse_pruned_count"]) for row in mature_pruning_rows)),
        "mature_phase_pruning_event_count": len(mature_pruning_rows),
        "binding_pruned_total": int(sum(int(row["binding_pruned_count"]) for row in tick_rows)),
        "binding_pruning_event_count": len(binding_pruning_rows),
        "rebuild_event_count": len(rebuild_rows),
    }

    validations = {
        "phases_observed": len(phase_transition_ticks) == 3,
        "synapse_pruning_observed": pruning["synapse_pruned_total"] > 0,
        "critical_phase_pruning_observed": pruning["critical_phase_pruned_total"] > 0,
        "mature_phase_pruning_observed": pruning["mature_phase_pruned_total"] > 0,
        "late_run_modalities_alive": all(value > 0.0 for value in late_modality_activation_avg.values()),
        "late_run_activity_nonzero": stability["late_run_total_active_avg"] > 0.0,
    }

    warnings: list[str] = []
    if not validations["phases_observed"]:
        warnings.append("Not all lifecycle phases were observed during the probe run.")
    if not validations["synapse_pruning_observed"]:
        warnings.append("No synapse pruning was observed under the compressed lifecycle schedule.")
    if phase_tick_counts.get("critical", 0) > 0 and not validations["critical_phase_pruning_observed"]:
        warnings.append("Critical phase was observed but no critical-phase synapse pruning occurred.")
    if phase_tick_counts.get("mature", 0) > 0 and not validations["mature_phase_pruning_observed"]:
        warnings.append("Mature phase was observed but no mature-phase synapse pruning occurred.")
    if not validations["late_run_modalities_alive"]:
        warnings.append("Late-run modality activation collapsed in at least one core modality.")
    if stability["late_run_sparsity_max"] > 0.12:
        warnings.append(
            f"Late-run sparsity exceeded 12% ({stability['late_run_sparsity_max']:.4f})."
        )

    early_total_active = stability["early_run_total_active_avg"]
    late_total_active = stability["late_run_total_active_avg"]
    if early_total_active > 0.0 and late_total_active > early_total_active * 2.5:
        warnings.append(
            "Late-run total activity grew by more than 2.5x relative to the early run window."
        )

    return {
        "sample_count": len(sample_rows),
        "tick_count": sum(_executed_ticks(row) for row in tick_rows),
        "train_tick_count": sum(_executed_ticks(row) for row in active_rows),
        "rest_tick_count": sum(
            _executed_ticks(row) for row in tick_rows if row["tick_kind"] != "train"
        ),
        "lifecycle_schedule": dict(schedule),
        "phase_tick_counts": {
            phase_name: int(count)
            for phase_name, count in sorted(phase_tick_counts.items())
        },
        "phase_transition_ticks": phase_transition_ticks,
        "pruning": pruning,
        "stability": stability,
        "validations": validations,
        "warnings": warnings,
    }


def run_multimodal_stability_probe(
    max_samples: int,
    ticks_per_sample: int,
    threads: int,
    output_path: str,
    *,
    n_traces: int = 5500,
    seed_chunks: int | None = 1,
    full_seed: bool = False,
    rest_ticks: int = 1,
    text_dataset: str = DEFAULT_MULTIMODAL_STABILITY_TEXT_DATASET,
    image_dataset: str = DEFAULT_MULTIMODAL_STABILITY_IMAGE_DATASET,
    audio_dataset: str = DEFAULT_MULTIMODAL_STABILITY_AUDIO_DATASET,
    checkpoint_stop_after_samples: int | None = None,
    initial_state_dir: str | None = None,
) -> dict[str, Any]:
    if threads > 0:
        try:
            brain_core.set_num_threads(threads)
        except Exception:
            pass
    actual_threads = brain_core.get_num_threads()
    run_started = time.perf_counter()

    samples = _build_multimodal_samples(
        max_samples,
        text_name=text_dataset,
        image_name=image_dataset,
        audio_name=audio_dataset,
    )

    config = {
        "max_samples": len(samples),
        "ticks_per_sample": int(ticks_per_sample),
        "rest_ticks": int(rest_ticks),
        "n_traces": int(n_traces),
        "seed_chunks": seed_chunks,
        "seed_mode": "full" if full_seed else "fast",
        "text_dataset": text_dataset,
        "image_dataset": image_dataset,
        "audio_dataset": audio_dataset,
        "threads": int(actual_threads),
        "initial_state_dir": initial_state_dir,
    }
    total_run_ticks = len(samples) * max(1, ticks_per_sample + rest_ticks)
    schedule = _build_lifecycle_schedule(total_run_ticks)
    run_id = _checkpoint_run_id(config)
    checkpoint_paths = _checkpoint_paths(run_id)

    checkpoint_progress = _load_checkpoint(
        checkpoint_paths,
        expected_config=config,
        schedule=schedule,
    )
    if checkpoint_progress is not None and checkpoint_progress.get("status") == "completed":
        if checkpoint_paths["result"].exists():
            completed_result = json.loads(
                checkpoint_paths["result"].read_text(encoding="utf-8")
            )
            _write_json_atomic(Path(output_path), completed_result)
            return completed_result

    resumed_from_sample = 0
    prior_processing_wall_ms = 0.0

    if checkpoint_progress is not None:
        trace_store = TraceStore()
        trace_store.load(str(checkpoint_paths["trace_store"]))
        brain_core.load_brain_checkpoint(str(checkpoint_paths["brain"]))
        tick_loop = TickLoop(
            trace_store,
            rust_tick_batch_size=DEFAULT_MULTIMODAL_STABILITY_TICK_BATCH_SIZE,
            prune_interval=schedule["prune_interval"],
            prune_batch_size=schedule["prune_batch_size"],
            binding_maintenance_interval=schedule["binding_maintenance_interval"],
            rebuild_interval=schedule["rebuild_interval"],
        )
        tick_loop.restore_checkpoint_state(
            dict(_read_pickle(checkpoint_paths["python_state"])["tick_loop"])
        )
        multimodal = MultimodalInput(
            text_encoder=TextInput(trace_store),
            visual_encoder=VisualInput(),
            audio_encoder=AudioInput(),
            sensory_encoder=SensoryInput(),
        )
        multimodal.restore_checkpoint_state(
            dict(_read_pickle(checkpoint_paths["python_state"])["multimodal"])
        )
        tick_rows = list(checkpoint_progress.get("tick_rows", []))
        sample_rows = list(checkpoint_progress.get("sample_rows", []))
        resumed_from_sample = int(checkpoint_progress.get("next_sample_index", 0))
        prior_processing_wall_ms = float(
            checkpoint_progress.get("accumulated_wall_ms", 0.0) or 0.0
        )
    else:
        if initial_state_dir is not None:
            trace_store, tick_loop, _python_state, _metadata = load_runtime_bundle(
                initial_state_dir
            )
            tick_loop.rust_tick_batch_size = DEFAULT_MULTIMODAL_STABILITY_TICK_BATCH_SIZE
            tick_loop.reset_runtime_boundary(preserve_binding_state=True)
            tick_loop._prune_interval = int(schedule["prune_interval"])
            tick_loop._prune_batch_size = int(schedule["prune_batch_size"])
            tick_loop._binding_maintenance_interval = int(
                schedule["binding_maintenance_interval"]
            )
            tick_loop._rebuild_interval = int(schedule["rebuild_interval"])
        else:
            if full_seed:
                _, trace_store = seed_brain(
                    verbose=False,
                    chunk_count=seed_chunks,
                )
            else:
                _, trace_store = seed_brain_fast(
                    n_traces=n_traces,
                    verbose=False,
                    chunk_count=seed_chunks,
                )

            tick_loop = TickLoop(
                trace_store,
                rust_tick_batch_size=DEFAULT_MULTIMODAL_STABILITY_TICK_BATCH_SIZE,
                prune_interval=schedule["prune_interval"],
                prune_batch_size=schedule["prune_batch_size"],
                binding_maintenance_interval=schedule["binding_maintenance_interval"],
                rebuild_interval=schedule["rebuild_interval"],
            )
        multimodal = MultimodalInput(
            text_encoder=TextInput(trace_store),
            visual_encoder=VisualInput(),
            audio_encoder=AudioInput(),
            sensory_encoder=SensoryInput(),
        )
        tick_rows = []
        sample_rows = []

    current_session_processing_ms = 0.0

    with _compressed_lifecycle(schedule):
        for sample_index, sample in enumerate(samples[resumed_from_sample:], start=resumed_from_sample):
            tick_loop.reset_sample_boundary()
            sample_started = time.perf_counter()

            sample_label = (
                f"{sample.get('text_label', '?')}+"
                f"{sample.get('image_label', '?')}+"
                f"{sample.get('audio_label', '?')}"
            )
            audio_array = sample["audio"]
            if hasattr(audio_array, "tolist"):
                audio_list = audio_array.tolist()
            else:
                audio_list = list(audio_array)

            inputs = {
                "text": sample["text"],
                "visual": sample["image"],
                "audio": (audio_list, sample.get("audio_sample_rate", 16000)),
            }

            sample_tick_rows: list[dict[str, Any]] = []
            synapse_count_start = int(brain_core.get_synapse_count())

            for local_tick_index, result in enumerate(
                tick_loop.iter_steps(ticks_per_sample, preserve_first_tick=True)
            ):
                if local_tick_index == 0:
                    multimodal.process(inputs, tick=tick_loop.last_tick_number)
                synapse_count = int(brain_core.get_synapse_count())
                row = _tick_row(
                    result,
                    sample_index=sample_index,
                    sample_label=sample_label,
                    tick_kind="train",
                    local_tick_index=local_tick_index,
                    synapse_count=synapse_count,
                )
                tick_rows.append(row)
                sample_tick_rows.append(row)

            for rest_tick_index, result in enumerate(
                tick_loop.iter_steps(
                    rest_ticks,
                    learn=False,
                    allow_trace_formation=False,
                    allow_binding_formation=False,
                )
            ):
                synapse_count = int(brain_core.get_synapse_count())
                row = _tick_row(
                    result,
                    sample_index=sample_index,
                    sample_label=sample_label,
                    tick_kind="rest",
                    local_tick_index=rest_tick_index,
                    synapse_count=synapse_count,
                )
                tick_rows.append(row)
                sample_tick_rows.append(row)

            synapse_count_end = int(brain_core.get_synapse_count())
            sample_rows.append(
                _sample_summary(
                    sample_tick_rows,
                    sample_index=sample_index,
                    sample_label=sample_label,
                    synapse_count_start=synapse_count_start,
                    synapse_count_end=synapse_count_end,
                )
            )

            current_session_processing_ms += (time.perf_counter() - sample_started) * 1000.0
            _save_checkpoint(
                checkpoint_paths,
                config=config,
                schedule=schedule,
                tick_loop=tick_loop,
                multimodal=multimodal,
                trace_store=trace_store,
                tick_rows=tick_rows,
                sample_rows=sample_rows,
                next_sample_index=sample_index + 1,
                accumulated_wall_ms=prior_processing_wall_ms + current_session_processing_ms,
            )

            if (
                checkpoint_stop_after_samples is not None
                and len(sample_rows) >= checkpoint_stop_after_samples
                and len(sample_rows) < len(samples)
            ):
                raise RuntimeError(
                    f"Checkpoint stop requested after {len(sample_rows)} completed samples"
                )

    summary = _summarize_multimodal_stability_probe(
        tick_rows,
        sample_rows,
        schedule,
    )
    result = {
        "benchmark": "multimodal_stability_probe",
        "threads": actual_threads,
        "config": dict(config),
        "sample_rows": sample_rows,
        "tick_rows": tick_rows,
        "summary": summary,
        "checkpoint": {
            "run_id": run_id,
            "checkpoint_dir": str(checkpoint_paths["dir"]),
            "completed_samples": len(sample_rows),
            "status": "completed",
        },
        "performance": {
            "processing_wall_ms": _round(
                prior_processing_wall_ms + current_session_processing_ms,
                4,
            ),
            "total_wall_ms": _round(
                prior_processing_wall_ms + (time.perf_counter() - run_started) * 1000.0,
                4,
            ),
        },
    }

    _write_json_atomic(checkpoint_paths["result"], result)
    _write_json_atomic(
        checkpoint_paths["progress"],
        {
            "version": 1,
            "status": "completed",
            "config": dict(config),
            "schedule": dict(schedule),
            "next_sample_index": len(samples),
            "completed_samples": len(sample_rows),
            "accumulated_wall_ms": _round(
                prior_processing_wall_ms + current_session_processing_ms,
                4,
            ),
            "tick_rows": tick_rows,
            "sample_rows": sample_rows,
        },
    )
    _write_json_atomic(Path(output_path), result)
    return result