"""Validation probe for numbers and executive regions on canonical topology."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import brain_core

from brain.input.text_input import TextInput
from brain.learning.tick_loop import TickLoop
from brain.seed.seed_runner import seed_brain


_DIGIT_SAMPLES = (
    "mission 7 launched",
    "signal 12 repeated",
    "market gained 15 points",
    "room 19 opened",
)
_WORD_NUMBER_SAMPLES = (
    "mission seven launched",
    "signal twelve repeated",
    "market fifteen points",
    "room nineteen opened",
)
_TEXT_ONLY_SAMPLES = (
    "mission launched",
    "signal repeated",
    "market rallied",
    "room opened",
)


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _text_coverage_row(encoder: TextInput, text: str) -> dict[str, Any]:
    tokens = encoder.tokenize(text)
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
    unique_matches = sorted(set(matched_traces))
    number_matches = [trace_id for trace_id in unique_matches if trace_id.startswith("number_")]
    return {
        "tokens": tokens,
        "known": known,
        "unknown": unknown,
        "coverage": _round(known / total, 4),
        "matched_trace_count": len(unique_matches),
        "matched_number_trace_count": len(number_matches),
        "matched_trace_ids": unique_matches[:12],
        "matched_number_trace_ids": number_matches[:12],
    }


def _sample_metrics(
    tick_loop: TickLoop,
    encoder: TextInput,
    text: str,
    *,
    sample_group: str,
    sample_kind: str,
    ticks_per_sample: int,
) -> dict[str, Any]:
    tick_loop.reset_runtime_boundary()
    encode_result = encoder.encode(text)

    numbers_counts: list[float] = []
    numbers_strengths: list[float] = []
    pattern_counts: list[float] = []
    executive_values: list[float] = []
    planning_values: list[float] = []
    language_values: list[float] = []

    for _ in range(ticks_per_sample):
        result = tick_loop.step(
            learn=False,
            allow_trace_formation=False,
            allow_binding_formation=False,
        )
        numbers_acts = brain_core.get_activations("numbers", 0.01)
        pattern_acts = brain_core.get_activations("pattern", 0.01)
        numbers_counts.append(float(len(numbers_acts)))
        numbers_strengths.append(sum(float(activation) for _nid, activation in numbers_acts))
        pattern_counts.append(float(len(pattern_acts)))
        executive_values.append(float(result.get("executive_engagement", 0.0)))
        planning_values.append(float(result.get("planning_signal", 0.0)))
        language_values.append(float(result.get("language_activation", 0.0)))

    coverage = _text_coverage_row(encoder, text)
    return {
        "group": sample_group,
        "kind": sample_kind,
        "text": text,
        **coverage,
        "neurons_activated": int(encode_result.get("neurons_activated", 0)),
        "numbers_active_count_mean": _round(_mean(numbers_counts), 4),
        "numbers_active_count_max": _round(max(numbers_counts) if numbers_counts else 0.0, 4),
        "numbers_activation_sum_mean": _round(_mean(numbers_strengths), 6),
        "numbers_activation_sum_max": _round(max(numbers_strengths) if numbers_strengths else 0.0, 6),
        "pattern_active_count_mean": _round(_mean(pattern_counts), 4),
        "pattern_active_count_max": _round(max(pattern_counts) if pattern_counts else 0.0, 4),
        "language_activation_mean": _round(_mean(language_values), 6),
        "language_activation_max": _round(max(language_values) if language_values else 0.0, 6),
        "executive_engagement_mean": _round(_mean(executive_values), 6),
        "executive_engagement_max": _round(max(executive_values) if executive_values else 0.0, 6),
        "planning_signal_mean": _round(_mean(planning_values), 6),
        "planning_signal_max": _round(max(planning_values) if planning_values else 0.0, 6),
    }


def _select_executive_phrases(encoder: TextInput) -> list[str]:
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()
    for trace in encoder.trace_store.traces.values():
        if not trace.label or not trace.neurons.get("executive"):
            continue
        aliases = encoder._label_aliases(trace.label)
        phrase = next(
            (
                " ".join(alias)
                for alias in aliases
                if alias and all(len(part) >= 3 and not part.isdigit() for part in alias)
            ),
            None,
        )
        if not phrase or phrase in seen:
            continue
        coverage = _text_coverage_row(encoder, phrase)
        if trace.id not in coverage["matched_trace_ids"]:
            continue
        candidates.append((int(coverage["matched_trace_count"]), phrase))
        seen.add(phrase)
        if len(candidates) >= 12:
            break

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [phrase for _matched_count, phrase in candidates[:8]]


def _build_executive_sample_sets(encoder: TextInput) -> tuple[list[str], list[str]]:
    phrases = _select_executive_phrases(encoder)
    if len(phrases) < 4:
        raise ValueError("not enough executive-bearing labeled phrases available for the probe")

    single_samples = phrases[:4]
    multi_samples: list[str] = []
    for index in range(4):
        parts = [phrases[index], phrases[(index + 1) % len(phrases)], phrases[(index + 2) % len(phrases)]]
        multi_samples.append(" ".join(parts))
    return single_samples, multi_samples


def _group_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    return {
        key: _round(_mean([float(row.get(key, 0.0)) for row in rows]), 6)
        for key in keys
    }


def run_executive_numbers_probe(
    ticks_per_sample: int,
    threads: int,
    output_path: str,
    *,
    seed_chunks: int | None = 1,
) -> dict[str, Any]:
    if threads > 0:
        try:
            brain_core.set_num_threads(threads)
        except Exception:
            pass
    actual_threads = brain_core.get_num_threads()

    _, trace_store = seed_brain(
        verbose=False,
        chunk_count=seed_chunks,
    )
    tick_loop = TickLoop(trace_store)
    encoder = TextInput(trace_store)

    number_rows: list[dict[str, Any]] = []
    for text in _DIGIT_SAMPLES:
        number_rows.append(
            _sample_metrics(
                tick_loop,
                encoder,
                text,
                sample_group="numbers",
                sample_kind="digit",
                ticks_per_sample=ticks_per_sample,
            )
        )
    for text in _WORD_NUMBER_SAMPLES:
        number_rows.append(
            _sample_metrics(
                tick_loop,
                encoder,
                text,
                sample_group="numbers",
                sample_kind="word_number",
                ticks_per_sample=ticks_per_sample,
            )
        )
    for text in _TEXT_ONLY_SAMPLES:
        number_rows.append(
            _sample_metrics(
                tick_loop,
                encoder,
                text,
                sample_group="numbers",
                sample_kind="text_only",
                ticks_per_sample=ticks_per_sample,
            )
        )

    executive_single, executive_multi = _build_executive_sample_sets(encoder)
    executive_rows: list[dict[str, Any]] = []
    for text in executive_single:
        executive_rows.append(
            _sample_metrics(
                tick_loop,
                encoder,
                text,
                sample_group="executive",
                sample_kind="single_concept",
                ticks_per_sample=ticks_per_sample,
            )
        )
    for text in executive_multi:
        executive_rows.append(
            _sample_metrics(
                tick_loop,
                encoder,
                text,
                sample_group="executive",
                sample_kind="multi_concept",
                ticks_per_sample=ticks_per_sample,
            )
        )

    digit_rows = [row for row in number_rows if row["kind"] == "digit"]
    word_rows = [row for row in number_rows if row["kind"] == "word_number"]
    text_rows = [row for row in number_rows if row["kind"] == "text_only"]

    number_keys = (
        "matched_number_trace_count",
        "numbers_active_count_max",
        "numbers_activation_sum_max",
        "language_activation_max",
        "pattern_active_count_max",
    )
    numbers_summary = {
        "digit": _group_summary(digit_rows, number_keys),
        "word_number": _group_summary(word_rows, number_keys),
        "text_only": _group_summary(text_rows, number_keys),
    }
    numbers_summary["digit_vs_text_delta"] = {
        key: _round(numbers_summary["digit"][key] - numbers_summary["text_only"][key], 6)
        for key in number_keys
    }
    numbers_summary["word_vs_text_delta"] = {
        key: _round(numbers_summary["word_number"][key] - numbers_summary["text_only"][key], 6)
        for key in number_keys
    }
    numbers_summary["validations"] = {
        "digit_numbers_alive": numbers_summary["digit"]["numbers_active_count_max"] > 0.0,
        "digit_numbers_selective_vs_text_only": numbers_summary["digit"]["numbers_active_count_max"] > numbers_summary["text_only"]["numbers_active_count_max"],
        "digit_language_delta_positive": numbers_summary["digit_vs_text_delta"]["language_activation_max"] > 0.0,
        "digit_pattern_delta_positive": numbers_summary["digit_vs_text_delta"]["pattern_active_count_max"] > 0.0,
        "word_control_numbers_alive": numbers_summary["word_number"]["numbers_active_count_max"] > 0.0,
    }

    single_rows = [row for row in executive_rows if row["kind"] == "single_concept"]
    multi_rows = [row for row in executive_rows if row["kind"] == "multi_concept"]
    executive_summary = {
        "single_concept": _group_summary(
            single_rows,
            (
                "matched_trace_count",
                "executive_engagement_max",
                "planning_signal_max",
                "language_activation_max",
            ),
        ),
        "multi_concept": _group_summary(
            multi_rows,
            (
                "matched_trace_count",
                "executive_engagement_max",
                "planning_signal_max",
                "language_activation_max",
            ),
        ),
    }
    executive_summary["delta"] = {
        key: _round(
            executive_summary["multi_concept"][key] - executive_summary["single_concept"][key],
            6,
        )
        for key in executive_summary["single_concept"]
    }
    single_exec = executive_summary["single_concept"]["executive_engagement_max"]
    multi_exec = executive_summary["multi_concept"]["executive_engagement_max"]
    executive_summary["validations"] = {
        "executive_alive": max(single_exec, multi_exec) > 0.0,
        "multi_concept_gt_single": multi_exec > single_exec,
        "multi_concept_delta_measurable": (multi_exec - single_exec) > 0.01,
    }

    aggregate = {
        "benchmark": "executive_numbers_probe",
        "seed_mode": "full",
        "threads": actual_threads,
        "config": {
            "ticks_per_sample": int(ticks_per_sample),
            "seed_chunks": seed_chunks,
        },
        "numbers_rows": number_rows,
        "executive_rows": executive_rows,
        "summary": {
            "numbers": numbers_summary,
            "executive": executive_summary,
        },
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate