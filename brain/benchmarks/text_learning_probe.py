"""Reproducible text-learning probe for trace selectivity and recall quality."""

from __future__ import annotations

import json
from pathlib import Path

import brain_core

from brain.datasets.downloader import load_text_dataset
from brain.input.text_input import TextInput
from brain.learning.tick_loop import TickLoop
from brain.seed.seed_runner import seed_brain_fast
from brain.structures.trace_store import Trace, TraceStore


DEFAULT_TEXT_LEARNING_PROBE_SPECS = (
    {
        "index": 27,
        "key": "hp_profit",
        "anchors": (
            ("hp", "pattern"),
            ("profit", "integration"),
            ("quarter", "memory_long"),
            ("expectations", "executive"),
        ),
        "cue_text": "hp profit quarter",
    },
    {
        "index": 16,
        "key": "kids_school",
        "anchors": (
            ("kids", "pattern"),
            ("purchasing power", "integration"),
            ("school", "memory_long"),
            ("marketing", "executive"),
        ),
        "cue_text": "kids purchasing power school",
    },
    {
        "index": 191,
        "key": "broadband_watchdog",
        "anchors": (
            ("watchdog", "pattern"),
            ("broadband", "integration"),
            ("full speed", "memory_long"),
            ("connections", "executive"),
        ),
        "cue_text": "watchdog broadband full speed",
    },
    {
        "index": 107,
        "key": "hubble_space",
        "anchors": (
            ("hubble", "pattern"),
            ("space telescope", "integration"),
            ("instruments", "memory_long"),
            ("engineers", "executive"),
        ),
        "cue_text": "hubble space telescope instruments",
    },
)


def _add_anchor_trace(
    trace_store: TraceStore,
    sample_key: str,
    label: str,
    primary_region: str,
    ordinal: int,
) -> str:
    seed_key = f"{sample_key}:{label}"
    regions = {
        "language": 8,
        primary_region: 16,
        "attention": 4,
    }
    if primary_region != "memory_long":
        regions["memory_long"] = 6
    else:
        regions["pattern"] = 6

    neurons = {
        region_name: TextInput._region_hash_neurons(seed_key, region_name, count)
        for region_name, count in regions.items()
    }
    trace = Trace(
        id=f"probe_anchor_{sample_key}_{ordinal:02d}",
        label=label,
        neurons=neurons,
        strength=0.25,
        novelty=0.2,
        context_tags=[f"probe:{sample_key}", f"primary:{primary_region}"],
    )
    trace_store.add(trace)
    return trace.id


def _dominant_region(trace: Trace) -> str | None:
    if not trace.neurons:
        return None
    return max(trace.neurons.items(), key=lambda item: len(item[1]))[0]


def _coverage_row(encoder: TextInput, sample: dict[str, object]) -> dict[str, object]:
    encoded = encoder.encode(str(sample["text"]))
    brain_core.reset_brain()
    known = int(encoded["known_count"])
    unknown = int(encoded["unknown_count"])
    total = max(1, known + unknown)
    return {
        "known": known,
        "unknown": unknown,
        "coverage": round(known / total, 4),
        "matched_traces": len(encoded["matched_traces"]),
        "matched_trace_ids": list(encoded["matched_traces"][:12]),
    }


def _summarize_learned_traces(
    trace_store: TraceStore,
    learned_trace_ids: set[str],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for trace_id in sorted(learned_trace_ids):
        trace = trace_store.get(trace_id)
        if trace is None:
            continue
        rows.append(
            {
                "trace_id": trace_id,
                "dominant_region": _dominant_region(trace),
                "total_neurons": trace.total_neurons(),
                "regions": {
                    region_name: len(neurons)
                    for region_name, neurons in sorted(trace.neurons.items())
                },
                "co_traces": list(trace.co_traces),
            }
        )

    sizes = [int(row["total_neurons"]) for row in rows]
    dominant_regions: dict[str, int] = {}
    for row in rows:
        dominant = row["dominant_region"]
        if dominant is None:
            continue
        dominant_regions[dominant] = dominant_regions.get(dominant, 0) + 1

    return {
        "count": len(rows),
        "size_min": min(sizes) if sizes else 0,
        "size_max": max(sizes) if sizes else 0,
        "size_avg": round(sum(sizes) / len(sizes), 3) if sizes else 0.0,
        "dominant_regions": dominant_regions,
        "rows": rows,
    }


def _region_overlap_counts(trace: Trace, active_set: set[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for region_name, neurons in trace.neurons.items():
        overlap = len(active_set & set(neurons))
        if overlap > 0:
            counts[region_name] = overlap
    return counts


def _prepare_probe_samples(max_samples: int) -> list[dict[str, object]]:
    specs = list(DEFAULT_TEXT_LEARNING_PROBE_SPECS[:max_samples])
    dataset = load_text_dataset(
        "ag_news",
        max_samples=max(spec["index"] for spec in specs) + 1,
    )
    samples: list[dict[str, object]] = []
    for spec in specs:
        sample = dict(dataset[int(spec["index"])])
        sample["index"] = spec["index"]
        sample["key"] = spec["key"]
        sample["anchors"] = list(spec["anchors"])
        sample["cue_text"] = spec["cue_text"]
        samples.append(sample)
    return samples


def run_text_learning_probe(
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
) -> dict[str, object]:
    """Train one curated text sample at a time and probe recall selectivity."""
    if threads > 0:
        try:
            brain_core.set_num_threads(threads)
        except Exception:
            pass
    actual_threads = brain_core.get_num_threads()

    probe_samples = _prepare_probe_samples(max_samples)
    results: dict[str, object] = {}

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

        encoder = TextInput(trace_store)
        coverage = {
            str(sample["key"]): _coverage_row(encoder, sample)
            for sample in probe_samples
        }

        tick_loop = TickLoop(trace_store)
        learned_trace_ids: set[str] = set()
        learned_binding_ids: set[int] = set()

        for _ in range(train_repeats):
            tick_loop.reset_sample_boundary()
            encoder.encode(str(trained_sample["text"]))
            for _ in range(ticks_per_sample):
                tick_loop.step()
                learned_trace_ids.update(tick_loop.trace_formation.recently_formed)
                learned_binding_ids.update(tick_loop.binding_formation.recently_formed)
            for _ in range(rest_ticks):
                tick_loop.step()

        for _ in range(settle_ticks):
            tick_loop.step()

        cue_results: dict[str, object] = {}
        for cue_sample in probe_samples:
            tick_loop.reset_probe_boundary()
            for _ in range(settle_ticks):
                tick_loop.step(learn=False)

            baseline_hits = sum(
                1
                for trace_id, _score in tick_loop.last_active_traces
                if trace_id in learned_trace_ids
            )

            cue_encoded = encoder.encode(str(cue_sample["cue_text"]))
            tick_rows: list[dict[str, object]] = []
            dominant_overlap_region_counts: dict[str, int] = {}
            for _ in range(probe_ticks):
                tick_loop.step(learn=False)
                ranks = {
                    trace_id: (rank, score)
                    for rank, (trace_id, score) in enumerate(tick_loop.last_active_traces, start=1)
                }
                active_set = tick_loop.history.current.active_set() if tick_loop.history.current is not None else set()
                hit_ranks = [
                    ranks[trace_id][0]
                    for trace_id in learned_trace_ids
                    if trace_id in ranks
                ]
                partial_bindings = set(brain_core.find_partial_bindings(0.01))
                active_bindings = {
                    binding_id
                    for binding_id, _score in brain_core.evaluate_bindings(0.01)
                }
                dominant_overlap_region = None
                dominant_overlap_count = 0
                region_overlap_counts: dict[str, int] = {}
                if learned_trace_ids:
                    trace_id = sorted(learned_trace_ids)[0]
                    trace = trace_store.get(trace_id)
                    if trace is not None:
                        region_overlap_counts = _region_overlap_counts(trace, active_set)
                        if region_overlap_counts:
                            dominant_overlap_region, dominant_overlap_count = max(
                                region_overlap_counts.items(),
                                key=lambda item: (item[1], item[0]),
                            )
                            dominant_overlap_region_counts[dominant_overlap_region] = (
                                dominant_overlap_region_counts.get(dominant_overlap_region, 0) + 1
                            )
                tick_rows.append(
                    {
                        "trace_hit_count": len(hit_ranks),
                        "best_rank": min(hit_ranks) if hit_ranks else None,
                        "active_bindings": len(active_bindings & learned_binding_ids),
                        "partial_bindings": len(partial_bindings & learned_binding_ids),
                        "region_overlap_counts": region_overlap_counts,
                        "dominant_overlap_region": dominant_overlap_region,
                        "dominant_overlap_count": dominant_overlap_count,
                    }
                )

            ranked_hits = [
                row["best_rank"]
                for row in tick_rows
                if row["best_rank"] is not None
            ]
            cue_results[str(cue_sample["key"])] = {
                "cue_known": int(cue_encoded["known_count"]),
                "cue_unknown": int(cue_encoded["unknown_count"]),
                "baseline_trace_hits": baseline_hits,
                "trace_hit_rate": round(
                    sum(1 for row in tick_rows if row["trace_hit_count"] > 0)
                    / len(tick_rows),
                    4,
                ),
                "best_rank_avg": round(sum(ranked_hits) / len(ranked_hits), 3)
                if ranked_hits
                else None,
                "active_bindings_avg": round(
                    sum(int(row["active_bindings"]) for row in tick_rows)
                    / len(tick_rows),
                    3,
                ),
                "partial_bindings_avg": round(
                    sum(int(row["partial_bindings"]) for row in tick_rows)
                    / len(tick_rows),
                    3,
                ),
                "dominant_overlap_region_counts": dominant_overlap_region_counts,
                "ticks": tick_rows,
            }

            for _ in range(settle_ticks):
                tick_loop.step()

        results[str(trained_sample["key"])] = {
            "trained_label": trained_sample["label_name"],
            "coverage": coverage[str(trained_sample["key"])],
            "learned_trace_summary": _summarize_learned_traces(
                trace_store,
                learned_trace_ids,
            ),
            "learned_binding_count": len(learned_binding_ids),
            "cue_results": cue_results,
        }

    output = {
        "benchmark": "text_learning_probe",
        "dataset": "ag_news",
        "threads": actual_threads,
        "seed_traces": n_traces,
        "seed_chunks": seed_chunks,
        "ticks_per_sample": ticks_per_sample,
        "train_repeats": train_repeats,
        "rest_ticks": rest_ticks,
        "settle_ticks": settle_ticks,
        "probe_ticks": probe_ticks,
        "probe_mode": "isolated_single_sample",
        "trace_store_mode": "handcrafted_only",
        "results": results,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print("\nTEXT LEARNING PROBE SUMMARY")
    for sample_key, sample_result in results.items():
        trace_summary = sample_result["learned_trace_summary"]
        target_cue = sample_result["cue_results"].get(sample_key, {})
        print(
            f"  {sample_key}: traces={trace_summary['count']}, "
            f"size_avg={trace_summary['size_avg']}, size_max={trace_summary['size_max']}, "
            f"target_hit={target_cue.get('trace_hit_rate', 0.0):.2f}, "
            f"target_rank={target_cue.get('best_rank_avg')}, "
            f"target_overlap={target_cue.get('dominant_overlap_region_counts', {})}"
        )

    print(f"Probe metrics saved to: {path}")
    return output