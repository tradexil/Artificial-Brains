"""Dataset-driven labeled vocabulary overlay for text capability experiments."""

from __future__ import annotations

import re
from collections import Counter

from brain.datasets.downloader import load_text_dataset
from brain.input.text_input import TextInput, is_content_text_token
from brain.structures.trace_store import Trace, TraceStore


# Match the region mix observed when capped overlay harvest actually forms learned traces.
TEXT_VOCAB_OVERLAY_REGION_COUNTS = {
    "pattern": 8,
    "language": 8,
    "integration": 8,
    "memory_long": 4,
}


def _sanitize_token_for_id(token: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")
    return sanitized or "token"


def collect_unknown_content_token_counts(
    trace_store: TraceStore,
    dataset_name: str,
    *,
    max_samples: int,
) -> tuple[Counter[str], int]:
    encoder = TextInput(trace_store)
    samples = load_text_dataset(dataset_name, max_samples=max_samples)
    counter: Counter[str] = Counter()
    content_token_total = 0

    for sample in samples:
        tokens = encoder.tokenize(str(sample["text"]))
        content_token_total += sum(1 for token in tokens if is_content_text_token(token))

        idx = 0
        while idx < len(tokens):
            span_len, _trace_ids = encoder._find_span_match(tokens, idx)
            if span_len > 0:
                idx += span_len
                continue

            token = tokens[idx]
            if is_content_text_token(token):
                counter[token] += 1
            idx += 1

    return counter, content_token_total


def select_text_vocab_overlay_terms(
    trace_store: TraceStore,
    dataset_name: str,
    *,
    max_terms: int,
    max_samples: int,
) -> dict[str, object]:
    unknown_counter, content_token_total = collect_unknown_content_token_counts(
        trace_store,
        dataset_name,
        max_samples=max_samples,
    )
    selected_rows = [
        {"token": token, "frequency": frequency}
        for token, frequency in unknown_counter.most_common(max_terms)
    ]
    selected_frequency_total = sum(int(row["frequency"]) for row in selected_rows)
    return {
        "dataset": dataset_name,
        "max_samples": max_samples,
        "content_token_total": content_token_total,
        "selected_term_count": len(selected_rows),
        "selected_term_frequency_total": selected_frequency_total,
        "selected_term_content_coverage": round(
            selected_frequency_total / max(1, content_token_total),
            4,
        ),
        "rows": selected_rows,
    }


def build_text_vocab_overlay_trace(
    token: str,
    *,
    dataset_name: str,
    ordinal: int,
    frequency: int,
) -> Trace:
    neurons = {
        region_name: TextInput._region_hash_neurons(token, region_name, count)
        for region_name, count in TEXT_VOCAB_OVERLAY_REGION_COUNTS.items()
    }
    return Trace(
        id=f"overlay_{dataset_name}_{ordinal:04d}_{_sanitize_token_for_id(token)}",
        label=token,
        neurons=neurons,
        strength=0.2,
        abstraction=0.15,
        novelty=0.2,
        context_tags=[
            "overlay:text_vocab",
            f"dataset:{dataset_name}",
            f"frequency:{frequency}",
        ],
    )


def apply_text_vocab_overlay(
    trace_store: TraceStore,
    dataset_name: str,
    *,
    max_terms: int,
    max_samples: int,
) -> dict[str, object]:
    selection = select_text_vocab_overlay_terms(
        trace_store,
        dataset_name,
        max_terms=max_terms,
        max_samples=max_samples,
    )

    added_rows: list[dict[str, object]] = []
    for ordinal, row in enumerate(selection["rows"], start=1):
        token = str(row["token"])
        frequency = int(row["frequency"])
        trace = build_text_vocab_overlay_trace(
            token,
            dataset_name=dataset_name,
            ordinal=ordinal,
            frequency=frequency,
        )
        trace_store.add(trace)
        added_rows.append(
            {
                "trace_id": trace.id,
                "token": token,
                "frequency": frequency,
                "regions": {
                    region_name: len(neuron_ids)
                    for region_name, neuron_ids in trace.neurons.items()
                },
            }
        )

    return {
        **selection,
        "applied": True,
        "added_trace_count": len(added_rows),
        "added_rows": added_rows,
    }