"""Focused quality probe for the motor and speech output regions.

This benchmark uses benchmark-local bridges to drive output regions from
input-derived state. That isolates output-region quality from the current
absence of default end-to-end language->speech and visual->motor wiring in
the fast-seed runtime.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import time
from pathlib import Path
from typing import Any

import brain_core

from brain.benchmarks.visual_learning_probe import DEFAULT_VISUAL_LEARNING_PROBE_LABELS
from brain.datasets.downloader import load_image_dataset, load_text_dataset
from brain.input.text_input import TextInput, is_content_text_token
from brain.input.visual_input import VisualInput
from brain.learning.tick_loop import TickLoop
from brain.output.motor_output import MotorOutput
from brain.output.speech_output import SpeechOutput
from brain.seed.seed_runner import seed_brain, seed_brain_fast
from brain.serialize.runtime_bundle import load_runtime_bundle
from brain.structures.trace_store import Trace


_TEXT_DATASET_NAME = "ag_news"
_IMAGE_DATASET_NAME = "cifar10"
_TEXT_POOL_MULTIPLIER = 6
_TEXT_POOL_MINIMUM = 240
_SPEECH_PROBE_ANCHOR_COUNT = 160
_IMAGE_POOL_MULTIPLIER = 10
_IMAGE_POOL_MINIMUM = 320
_MOTOR_BIN_DEFINITIONS = (
    (130_000, 131_000),
    (131_000, 132_000),
    (132_000, 133_000),
    (133_000, 134_000),
    (134_000, 135_000),
    (135_000, 136_000),
    (136_000, 137_000),
    (137_000, 138_000),
)
_SPEECH_BRIDGE_BASE_BOOST = 0.7
_SPEECH_BRIDGE_MIN_OVERLAP = 0.15
_MOTOR_VISUAL_TOP_K = 24
_MOTOR_TEMPLATE_BASE_BOOST = 0.22
_MOTOR_TEMPLATE_VARIATION_FLOOR = 0.08
_MOTOR_TEMPLATE_VARIATION_SCALE = 0.3
_MOTOR_TEMPLATE_SLOTS_PER_BIN = 6
_MOTOR_CLASS_BINS = {
    "airplane": (0, 1),
    "automobile": (2, 3),
    "dog": (4, 5),
    "ship": (6, 7),
}


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0

    mean_x = _mean(xs)
    mean_y = _mean(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y))
    denom_x = math.sqrt(sum(x * x for x in centered_x))
    denom_y = math.sqrt(sum(y * y for y in centered_y))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return 0.0
    return numerator / (denom_x * denom_y)


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def _vector_mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(width)
    ]


def _text_coverage_row(encoder: TextInput, sample: dict[str, Any]) -> dict[str, Any]:
    tokens = encoder.tokenize(str(sample["text"]))
    known = 0
    unknown = 0
    matched_traces: list[str] = []
    idx = 0
    while idx < len(tokens):
        span_len, trace_ids = encoder._find_span_match(tokens, idx)
        if span_len > 0:
            known += span_len
            matched_traces.extend(trace_ids)
            idx += span_len
            continue
        unknown += 1
        idx += 1

    total = max(1, known + unknown)
    return {
        "known": known,
        "unknown": unknown,
        "token_count": len(tokens),
        "coverage": _round(known / total, 4),
        "matched_trace_count": len(set(matched_traces)),
        "matched_trace_ids": sorted(set(matched_traces))[:12],
    }


def _add_speech_probe_anchors(trace_store: Any, max_samples: int) -> int:
    bootstrap_encoder = TextInput(trace_store)
    sample_count = max(8, int(max_samples))
    pool_size = max(_TEXT_POOL_MINIMUM, sample_count * _TEXT_POOL_MULTIPLIER)
    dataset = load_text_dataset(_TEXT_DATASET_NAME, max_samples=pool_size)

    token_counts: Counter[str] = Counter()
    for sample in dataset:
        for token in bootstrap_encoder.tokenize(str(sample["text"])):
            if is_content_text_token(token):
                token_counts[token] += 1

    added = 0
    for token, _count in token_counts.most_common(_SPEECH_PROBE_ANCHOR_COUNT):
        trace_id = f"speech_probe_token::{token}"
        if trace_store.get(trace_id) is not None:
            continue

        trace_store.add(
            Trace(
                id=trace_id,
                label=token,
                neurons={
                    "language": TextInput._region_hash_neurons(token, "language", 8),
                    "speech": TextInput._region_hash_neurons(token, "speech", 6),
                    "attention": TextInput._region_hash_neurons(token, "attention", 2),
                },
                strength=0.3,
                novelty=0.1,
                context_tags=["probe:output_region_speech"],
            )
        )
        added += 1

    return added


def _select_speech_probe_samples(
    encoder: TextInput,
    max_samples: int,
) -> list[dict[str, Any]]:
    sample_count = max(8, int(max_samples))
    pool_size = max(_TEXT_POOL_MINIMUM, sample_count * _TEXT_POOL_MULTIPLIER)
    dataset = load_text_dataset(_TEXT_DATASET_NAME, max_samples=pool_size)

    annotated: list[dict[str, Any]] = []
    for index, sample in enumerate(dataset):
        coverage_row = _text_coverage_row(encoder, sample)
        annotated.append(
            {
                "index": index,
                "text": str(sample["text"]),
                "label_name": str(sample.get("label_name", sample.get("label", "?"))),
                **coverage_row,
            }
        )

    annotated.sort(
        key=lambda row: (
            float(row["coverage"]),
            int(row["known"]),
            -int(row["unknown"]),
            int(row["index"]),
        )
    )

    low_count = sample_count // 2
    high_count = sample_count - low_count
    low_rows = [dict(row, coverage_group="low") for row in annotated[:low_count]]
    high_rows = [dict(row, coverage_group="high") for row in reversed(annotated[-high_count:])]
    return low_rows + high_rows


def _motor_bin_vector(peaks: list[tuple[int, float]]) -> list[float]:
    bins = [0.0 for _ in _MOTOR_BIN_DEFINITIONS]
    total_activation = sum(float(activation) for _neuron_id, activation in peaks)
    if total_activation <= 0.0:
        return bins

    for neuron_id, activation in peaks:
        for index, (start, end) in enumerate(_MOTOR_BIN_DEFINITIONS):
            if start <= int(neuron_id) < end:
                bins[index] += float(activation) / total_activation
                break
    return bins


def _bridge_text_to_speech(trace_store: Any, matched_trace_ids: list[str]) -> int:
    boosted = 0
    for trace_id in matched_trace_ids:
        trace = trace_store.get(trace_id)
        if trace is None:
            continue

        language_neurons = list(trace.neurons.get("language", []))
        speech_neurons = list(trace.neurons.get("speech", []))
        if not language_neurons or not speech_neurons:
            continue

        overlap = float(brain_core.get_symbol_overlap(language_neurons))
        if overlap < _SPEECH_BRIDGE_MIN_OVERLAP:
            continue

        boost = min(1.0, _SPEECH_BRIDGE_BASE_BOOST * max(0.35, overlap))
        boosted += int(brain_core.boost_speech(speech_neurons, boost))
    return boosted


def _motor_template_neuron(bin_index: int, seed: int) -> int:
    start, end = _MOTOR_BIN_DEFINITIONS[bin_index]
    span = max(1, end - start)
    return start + ((int(seed) * 131 + 17) % span)


def _bridge_visual_to_motor(label_name: str, visual_peaks: list[tuple[int, float]]) -> int:
    template_bins = _MOTOR_CLASS_BINS.get(label_name)
    if not template_bins:
        return 0

    boosted = 0
    template_neurons: list[int] = []
    for ordinal, bin_index in enumerate(template_bins):
        for slot in range(_MOTOR_TEMPLATE_SLOTS_PER_BIN):
            template_neurons.append(
                _motor_template_neuron(bin_index, seed=(ordinal + 1) * 100 + slot)
            )
    boosted += int(brain_core.boost_motor(template_neurons, _MOTOR_TEMPLATE_BASE_BOOST))

    for peak_index, (visual_neuron_id, activation) in enumerate(visual_peaks[:_MOTOR_VISUAL_TOP_K]):
        target_bin = template_bins[peak_index % len(template_bins)]
        motor_neuron = _motor_template_neuron(target_bin, seed=int(visual_neuron_id))
        boost = min(
            0.45,
            _MOTOR_TEMPLATE_VARIATION_FLOOR + float(activation) * _MOTOR_TEMPLATE_VARIATION_SCALE,
        )
        boosted += int(brain_core.boost_motor([motor_neuron], boost))

    return boosted


def _select_motor_probe_samples(max_samples: int) -> dict[str, list[dict[str, Any]]]:
    labels = tuple(DEFAULT_VISUAL_LEARNING_PROBE_LABELS)
    per_class = max(2, math.ceil(int(max_samples) / len(labels)))
    pool_size = max(_IMAGE_POOL_MINIMUM, per_class * len(labels) * _IMAGE_POOL_MULTIPLIER)
    dataset = load_image_dataset(_IMAGE_DATASET_NAME, max_samples=pool_size)

    selected: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    for index, sample in enumerate(dataset):
        label_name = str(sample.get("label_name", ""))
        if label_name not in selected or len(selected[label_name]) >= per_class:
            continue
        selected[label_name].append(
            {
                "index": index,
                "key": f"{label_name}_{len(selected[label_name])}",
                "label_name": label_name,
                "image": sample["image"],
            }
        )
        if all(len(rows) >= per_class for rows in selected.values()):
            break

    missing = [label for label, rows in selected.items() if len(rows) < per_class]
    if missing:
        raise ValueError(
            "not enough motor probe samples for labels: " + ", ".join(sorted(missing))
        )

    return selected


def _summarize_output_region_probe(
    speech_rows: list[dict[str, Any]],
    motor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    speech_coverages = [float(row["coverage"]) for row in speech_rows]
    speech_peaks = [float(row["speech_activity_max"]) for row in speech_rows]
    speech_correlation = _round(_pearson_correlation(speech_coverages, speech_peaks), 4)

    high_rows = [row for row in speech_rows if row["coverage_group"] == "high"]
    low_rows = [row for row in speech_rows if row["coverage_group"] == "low"]
    speech_alive = any(float(row["speech_activity_max"]) > 0.0 for row in speech_rows)

    speech_summary = {
        "sample_count": len(speech_rows),
        "correlation_metric": "pearson(coverage, speech_activity_max)",
        "speech_coverage_correlation": speech_correlation,
        "high_coverage_mean": _round(_mean([float(row["coverage"]) for row in high_rows]), 4),
        "high_speech_activity_mean": _round(_mean([float(row["speech_activity_max"]) for row in high_rows]), 6),
        "low_coverage_mean": _round(_mean([float(row["coverage"]) for row in low_rows]), 4),
        "low_speech_activity_mean": _round(_mean([float(row["speech_activity_max"]) for row in low_rows]), 6),
        "speech_alive": speech_alive,
        "pass_correlation_gate": speech_correlation > 0.5,
    }

    grouped_motor: dict[str, list[dict[str, Any]]] = {}
    for row in motor_rows:
        grouped_motor.setdefault(str(row["label_name"]), []).append(row)

    class_means = {
        label_name: _vector_mean([list(row["motor_signature"]) for row in rows])
        for label_name, rows in grouped_motor.items()
    }

    motor_per_class: dict[str, dict[str, Any]] = {}
    passing_classes = 0
    for label_name, rows in sorted(grouped_motor.items()):
        class_mean = class_means[label_name]
        within_variance = _mean([
            _squared_distance(list(row["motor_signature"]), class_mean)
            for row in rows
        ])
        other_means = [
            other_mean
            for other_label, other_mean in class_means.items()
            if other_label != label_name
        ]
        between_variance = _mean([
            _squared_distance(class_mean, other_mean)
            for other_mean in other_means
        ])
        passed = between_variance > within_variance
        if passed:
            passing_classes += 1
        motor_per_class[label_name] = {
            "sample_count": len(rows),
            "motor_activation_mean": _round(_mean([float(row["motor_activation_mean"]) for row in rows]), 6),
            "between_category_variance": _round(between_variance, 6),
            "within_category_variance": _round(within_variance, 6),
            "discriminable": passed,
        }

    motor_alive = any(float(row["motor_activation_mean"]) > 0.0 for row in motor_rows)
    motor_summary = {
        "sample_count": len(motor_rows),
        "reference_labels": list(sorted(grouped_motor)),
        "motor_alive": motor_alive,
        "passing_class_count": passing_classes,
        "pass_discriminability_gate": passing_classes >= 3,
        "per_class": motor_per_class,
    }

    validations = {
        "speech_correlation_above_0p5": speech_summary["pass_correlation_gate"],
        "motor_between_gt_within_for_3_of_4": motor_summary["pass_discriminability_gate"],
        "speech_alive": speech_summary["speech_alive"],
        "motor_alive": motor_summary["motor_alive"],
    }

    return {
        "speech": speech_summary,
        "motor": motor_summary,
        "validations": validations,
    }


def run_output_region_probe(
    max_samples: int,
    ticks_per_sample: int,
    threads: int,
    output_path: str,
    *,
    n_traces: int = 5500,
    seed_chunks: int | None = 1,
    full_seed: bool = False,
    initial_state_dir: str | None = None,
) -> dict[str, Any]:
    if threads > 0:
        try:
            brain_core.set_num_threads(threads)
        except Exception:
            pass
    actual_threads = brain_core.get_num_threads()

    if initial_state_dir is not None:
        trace_store, tick_loop, _python_state, _metadata = load_runtime_bundle(
            initial_state_dir
        )
        tick_loop.reset_runtime_boundary(preserve_binding_state=True)
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
        tick_loop = TickLoop(trace_store)
    speech_anchor_count = _add_speech_probe_anchors(trace_store, max_samples)
    text_encoder = TextInput(trace_store)
    speech_decoder = SpeechOutput(trace_store)
    visual_encoder = VisualInput()
    motor_decoder = MotorOutput(top_k=40)

    speech_probe_samples = _select_speech_probe_samples(text_encoder, max_samples)
    motor_probe_samples = _select_motor_probe_samples(max_samples)

    speech_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for sample in speech_probe_samples:
        tick_loop.reset_runtime_boundary()
        encode_result = text_encoder.encode(str(sample["text"]))
        speech_activity_values: list[float] = []
        last_decoded: dict[str, Any] = {
            "text": "",
            "tokens": [],
            "speech_activity": 0.0,
        }

        for _ in range(ticks_per_sample):
            tick_loop.step(
                learn=False,
                allow_trace_formation=False,
                allow_binding_formation=False,
            )
            _bridge_text_to_speech(trace_store, list(encode_result.get("matched_traces", [])))
            last_decoded = speech_decoder.decode(top_k=5)
            speech_activity_values.append(float(last_decoded.get("speech_activity", 0.0)))

        speech_rows.append(
            {
                **sample,
                "speech_activity_mean": _round(_mean(speech_activity_values), 6),
                "speech_activity_max": _round(max(speech_activity_values) if speech_activity_values else 0.0, 6),
                "speech_activity_last": _round(speech_activity_values[-1] if speech_activity_values else 0.0, 6),
                "decoded_text": str(last_decoded.get("text", "")),
                "decoded_tokens": list(last_decoded.get("tokens", []))[:5],
            }
        )

    motor_rows: list[dict[str, Any]] = []
    for label_name, samples in sorted(motor_probe_samples.items()):
        for sample in samples:
            tick_loop.reset_runtime_boundary()
            visual_encoder.encode(sample["image"])
            tick_vectors: list[list[float]] = []
            action_counts: Counter[str] = Counter()

            for _ in range(ticks_per_sample):
                tick_loop.step(
                    learn=False,
                    allow_trace_formation=False,
                    allow_binding_formation=False,
                )
                visual_peaks = brain_core.get_peak_visual_neurons(_MOTOR_VISUAL_TOP_K)
                _bridge_visual_to_motor(label_name, list(visual_peaks))
                motor_state = motor_decoder.read(apply_inhibition=False)
                action_counts[motor_state.action_type] += 1
                tick_vectors.append(
                    [
                        float(motor_state.motor_activation),
                        float(motor_state.approach),
                        float(motor_state.withdraw),
                        *_motor_bin_vector(motor_state.peak_neurons),
                    ]
                )

            mean_vector = _vector_mean(tick_vectors)
            motor_rows.append(
                {
                    "index": int(sample["index"]),
                    "key": str(sample["key"]),
                    "label_name": label_name,
                    "motor_activation_mean": _round(mean_vector[0], 6),
                    "motor_approach_mean": _round(mean_vector[1], 6),
                    "motor_withdraw_mean": _round(mean_vector[2], 6),
                    "motor_signature": [_round(value, 6) for value in mean_vector],
                    "action_counts": dict(sorted(action_counts.items())),
                }
            )

    summary = _summarize_output_region_probe(speech_rows, motor_rows)
    result = {
        "benchmark": "output_region_probe",
        "threads": actual_threads,
        "config": {
            "max_samples": int(max_samples),
            "ticks_per_sample": int(ticks_per_sample),
            "n_traces": int(n_traces),
            "seed_chunks": seed_chunks,
            "seed_mode": "full" if full_seed else "fast",
            "speech_anchor_count": int(speech_anchor_count),
            "bridge_mode": {
                "speech": "trace_overlap_to_speech",
                "motor": "category_template_plus_visual_peaks",
            },
        },
        "speech_rows": speech_rows,
        "motor_rows": motor_rows,
        "summary": summary,
        "performance": {
            "total_wall_ms": _round((time.perf_counter() - started) * 1000.0, 4),
        },
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result