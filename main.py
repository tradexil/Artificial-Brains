"""AI Brain — Real Learning with HuggingFace Datasets.

Entry point for Phase 11: downloads real datasets, feeds them through
the full tick loop, collects rich JSON metrics, and compares sequential
vs separate learning runs.

Usage:
    python main.py --dataset ag_news --samples 100 --ticks 10
    python main.py --dataset cifar10 --samples 50 --ticks 15
    python main.py --dataset speech_commands --samples 50 --ticks 10
    python main.py --dataset all --samples 30 --ticks 10 --mode sequential
    python main.py --dataset all --samples 30 --ticks 10 --mode separate
    python main.py --compare --samples 30 --ticks 10
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

import brain_core

from brain.benchmarks import (
    DEFAULT_AUDIO_LEARNING_PROBE_LABELS,
    DEFAULT_ASYNC_MULTI_BRAIN_CORES_PER_WORKER,
    DEFAULT_ASYNC_MULTI_BRAIN_MERGE_EVERY_SAMPLES,
    DEFAULT_ASYNC_MULTI_BRAIN_WORKER_COUNT,
    DEFAULT_CROSSMODAL_RECALL_PROBE_SPEC,
    DEFAULT_FIXED_GRAPH_OVERLAY_DELAYS,
    DEFAULT_FIXED_GRAPH_OVERLAY_REGIONS,
    DEFAULT_MULTIMODAL_BINDING_PROBE_CATALOG,
    DEFAULT_TEXT_LEARNING_PROBE_SPECS,
    DEFAULT_VISUAL_LEARNING_PROBE_LABELS,
    build_fixed_graph_spec,
    run_audio_learning_probe,
    run_async_multi_brain_text,
    run_crossmodal_recall_probe,
    run_executive_numbers_probe,
    run_multimodal_binding_probe,
    run_multimodal_stability_probe,
    run_output_region_probe,
    run_phase11_operational_baseline,
    run_text_binding_probe,
    run_text_learning_probe,
    run_text_vocab_profile,
    run_visual_learning_probe,
    run_coding_assistant_probe,
    run_end_to_end_demo,
)
from brain.seed.seed_runner import seed_brain, seed_brain_fast
from brain.learning.tick_loop import TickLoop
from brain.input.text_input import TextInput
from brain.input.visual_input import VisualInput
from brain.input.audio_input import AudioInput
from brain.input.sensory_input import SensoryInput
from brain.input.multimodal import MultimodalInput
from brain.output.motor_output import MotorOutput
from brain.output.speech_output import SpeechOutput
from brain.metrics.collector import MetricsCollector
from brain.serialize.brain_saver import save_brain
from brain.utils.config import (
    BINDING_RECALL_MIN_RELATIVE_WEIGHT,
    CUE_COLLISION_STRONG_REINFORCEMENTS,
    CUE_COLLISION_WEAK_MISSES,
    NEURONS_PER_TRACE,
    REGIONS,
    REGION_CONFIG,
    TOTAL_NEURONS,
    TRACE_ACTIVATION_THRESHOLD,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label_speech_neurons(label: str) -> list[int]:
    """Deterministic speech neurons for a label (same hash as trace_formation)."""
    import hashlib
    speech_start, speech_end = REGIONS["speech"]
    speech_count = speech_end - speech_start + 1
    inhib_pct = REGION_CONFIG["speech"]["inhibitory_pct"]
    exc_end = speech_start + int(speech_count * (1.0 - inhib_pct)) - 1
    n_speech = NEURONS_PER_TRACE.get("speech", 2)
    h = int(hashlib.sha256(label.encode()).hexdigest(), 16)
    exc_range = exc_end - speech_start + 1
    return [speech_start + (h + j * 2654435761) % exc_range for j in range(n_speech)]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_threads(n: int) -> int:
    """Set rayon thread count. Returns actual thread count."""
    if n > 0:
        try:
            brain_core.set_num_threads(n)
        except Exception:
            pass  # already set (rayon build_global can only be called once)
    return brain_core.get_num_threads()


def _seed_and_build(
    verbose: bool = True,
    fast: bool = False,
    n_traces: int = 5000,
    seed_chunks: int | None = None,
):
    """Seed the brain. Returns (brain_core, TraceStore)."""
    if verbose:
        mode_str = f"fast ({n_traces} traces)" if fast else "full (100k traces)"
        print(f"Seeding brain ({mode_str})...")
        t0 = time.perf_counter()
    if fast:
        _, ts = seed_brain_fast(
            n_traces=n_traces,
            verbose=verbose,
            chunk_count=seed_chunks,
        )
    else:
        _, ts = seed_brain(verbose=verbose, chunk_count=seed_chunks)
    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"  Seed complete: {brain_core.get_neuron_count()} neurons, "
              f"{brain_core.get_synapse_count():,} synapses, "
              f"{len(ts)} traces in {elapsed:.1f}s")
    return ts


def _rest_ticks(tick_loop: TickLoop, n: int = 3):
    """Run N ticks with no input to let activity decay. Returns last result."""
    result = None
    for _ in range(n):
        result = tick_loop.step()
    return result


def _begin_sample(tick_loop: TickLoop) -> None:
    """Reset sample-scoped learning state before injecting a new input."""
    tick_loop.reset_sample_boundary()


def _warmup_tick(tick_loop: TickLoop) -> float:
    """Run a single warmup tick and return elapsed time in milliseconds."""
    print("Warming up (1 tick)...")
    t0 = time.perf_counter()
    tick_loop.step()
    warmup_ms = (time.perf_counter() - t0) * 1000
    print(f"  Warmup: {warmup_ms:.0f}ms")
    return warmup_ms


def _warmup_session(tick_loop: TickLoop) -> float:
    """Pay one-time tick setup cost without shifting the real session cadence.

    Multi-rate scheduling is tick-parity-sensitive for one-shot text injection.
    Reset runtime state after warmup so the first real sample still starts on
    tick 0 while retaining any one-time initialization benefits from the warmup.
    """
    warmup_ms = _warmup_tick(tick_loop)
    tick_loop.reset_runtime_boundary()
    return warmup_ms


_DEFAULT_ABLATION_REGIONS = ["visual", "language", "executive", "memory_long"]
_DEFAULT_ABLATION_DELAYS = [1, 2]
_FIXED_GRAPH_TEXT_DATASETS = ["ag_news", "imdb"]
_DEFAULT_EXECUTIVE_NUMBERS_PROBE_TICKS = 5
_FIXED_GRAPH_TRIPLET = (
    ("structural", False, False, False),
    ("overlay_full_use", True, False, False),
    ("overlay_propagation_ablation", True, True, False),
)


def _cli_option_provided(option_name: str) -> bool:
    """Return True when a CLI option was explicitly provided by the caller."""
    return any(
        arg == option_name or arg.startswith(f"{option_name}=")
        for arg in sys.argv[1:]
    )
_FIXED_GRAPH_SUMMARY_KEYS = (
    "ticks_per_sec",
    "total_active_avg",
    "trace_candidates_avg",
    "binding_candidates_avg",
    "binding_candidates_per_trace_candidate_avg",
    "bindings_formed_total",
)


def _parse_csv_strings(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_csv_ints(raw: str | None) -> list[int]:
    if raw is None:
        return []
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def _configure_same_region_delay_ablation(
    enabled: bool,
    regions: list[str] | None = None,
    delays: list[int] | None = None,
) -> tuple[list[str], list[int]]:
    if not enabled:
        brain_core.clear_same_region_delay_ablation()
        return [], []

    region_names = list(regions or _DEFAULT_ABLATION_REGIONS)
    delay_values = list(delays or _DEFAULT_ABLATION_DELAYS)
    brain_core.set_same_region_delay_ablation(region_names, delay_values)
    print(
        "Ablation: same-region delays "
        f"{', '.join(str(delay) for delay in delay_values)} disabled in "
        f"{', '.join(region_names)}"
    )
    return region_names, delay_values


_CUE_BENCHMARK_SEED = 12345
_COLLISION_DISFAVORED_REGIONS = {"numbers"}
_REGION_PRIORITY = {
    "visual": 0,
    "audio": 1,
    "language": 2,
    "pattern": 3,
    "sensory": 4,
    "emotion": 5,
    "motor": 6,
    "speech": 7,
    "numbers": 8,
    "integration": 9,
    "executive": 10,
    "attention": 11,
    "memory_short": 12,
    "memory_long": 13,
}
_MODALITY_SIGNATURE_PRIORITY = ("visual", "audio", "sensory", "language", "speech")
_EXTERNAL_MODALITY_SIGNATURES = {"visual", "audio", "sensory", "language"}


def _trace_match_metrics(active_traces: list[tuple[str, float]], trace_id: str) -> tuple[bool, int | None, float]:
    """Return (hit, rank, score) for a trace in the current active set."""
    for rank, (active_trace_id, score) in enumerate(active_traces, start=1):
        if active_trace_id == trace_id:
            return True, rank, score
    return False, None, 0.0


def _top_non_excluded_trace(
    active_traces: list[tuple[str, float]],
    excluded_ids: set[str],
) -> tuple[str | None, int | None, float]:
    """Return the highest-ranked active trace outside the excluded set."""
    for rank, (trace_id, score) in enumerate(active_traces, start=1):
        if trace_id not in excluded_ids:
            return trace_id, rank, score
    return None, None, 0.0


def _dominant_trace_region(trace) -> str | None:
    """Pick the dominant region for cue pairing, preferring external modalities on ties."""
    candidates = [
        (region, len(neuron_ids))
        for region, neuron_ids in trace.neurons.items()
        if neuron_ids
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (item[1], -_REGION_PRIORITY.get(item[0], 999)),
    )[0]


def _trace_modality_signature(trace) -> str:
    """Assign a coarse modality family for cross-modality benchmark selection."""
    present_regions = {
        region_name
        for region_name, neuron_ids in trace.neurons.items()
        if neuron_ids
    }
    for region_name in _MODALITY_SIGNATURE_PRIORITY:
        if region_name in present_regions:
            return region_name
    return _dominant_trace_region(trace) or "unknown"


def _build_trace_cue(
    trace,
    cue_fraction: float,
    cue_noise_fraction: float,
    rng: random.Random,
    cue_mode: str = "global",
    focus_region: str | None = None,
) -> tuple[list[tuple[int, float]], int, int]:
    """Build a partial/noisy cue from a trace definition."""
    cue_fraction = min(1.0, max(0.0, cue_fraction))
    cue_noise_fraction = max(0.0, cue_noise_fraction)
    cue_mode = cue_mode.lower()

    trace_neurons = sorted({
        neuron_id
        for neuron_ids in trace.neurons.values()
        for neuron_id in neuron_ids
    })
    if not trace_neurons:
        return [], 0, 0

    signal_neurons: list[int] = []
    if cue_mode == "dominant-region":
        selected_region = focus_region or _dominant_trace_region(trace)
        region_neurons = sorted(trace.neurons.get(selected_region or "", []))
        if not region_neurons:
            region_neurons = trace_neurons
        keep_count = len(region_neurons)
        if cue_fraction < 1.0:
            keep_count = max(1, min(len(region_neurons), math.ceil(len(region_neurons) * cue_fraction)))
        signal_neurons = rng.sample(region_neurons, keep_count)
    elif cue_mode == "per-region":
        for region_name, region_neurons in trace.neurons.items():
            if not region_neurons:
                continue
            keep_count = len(region_neurons)
            if cue_fraction < 1.0:
                keep_count = max(1, min(len(region_neurons), math.ceil(len(region_neurons) * cue_fraction)))
            signal_neurons.extend(rng.sample(list(region_neurons), keep_count))
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


def _select_cue_pairs(traces: list, max_pairs: int) -> list[tuple[object, str, object, str]]:
    """Select deterministic cue→partner trace pairs with different dominant regions."""
    annotated = []
    for trace in traces:
        primary_region = _dominant_trace_region(trace)
        if primary_region is None:
            continue
        if primary_region in _COLLISION_DISFAVORED_REGIONS:
            continue
        if not trace.neurons.get(primary_region):
            continue
        annotated.append((trace, primary_region))

    pairs: list[tuple[object, str, object, str]] = []
    used_trace_ids: set[str] = set()
    for idx, (cue_trace, cue_region) in enumerate(annotated):
        if cue_trace.id in used_trace_ids:
            continue
        for partner_trace, partner_region in annotated[idx + 1:]:
            if partner_trace.id in used_trace_ids:
                continue
            if cue_region == partner_region:
                continue
            pairs.append((cue_trace, cue_region, partner_trace, partner_region))
            used_trace_ids.add(cue_trace.id)
            used_trace_ids.add(partner_trace.id)
            break
        if len(pairs) >= max_pairs:
            break
    return pairs


def _select_cue_collision_cases(
    traces: list,
    max_cases: int,
    selection: str = "overlap-min",
) -> list[tuple[object, str, object, str, object, str, float]]:
    """Select cue cases with one strong and one weak competitor sharing the same source pattern."""
    selection = selection.lower()
    annotated = []
    for trace in traces:
        primary_region = _dominant_trace_region(trace)
        if primary_region is None:
            continue
        if primary_region in _COLLISION_DISFAVORED_REGIONS:
            continue
        if not trace.neurons.get(primary_region):
            continue
        annotated.append((trace, primary_region, _trace_modality_signature(trace)))

    cases: list[tuple[object, str, object, str, object, str, float]] = []
    used_trace_ids: set[str] = set()
    for cue_trace, cue_region, _cue_signature in annotated:
        if cue_trace.id in used_trace_ids:
            continue

        partner_candidates = [
            (partner_trace, partner_region, partner_signature)
            for partner_trace, partner_region, partner_signature in annotated
            if partner_trace.id not in used_trace_ids
            and partner_trace.id != cue_trace.id
            and (selection == "cross-modality" or partner_region != cue_region)
        ]
        preferred_candidates = [
            candidate
            for candidate in partner_candidates
            if candidate[1] not in _COLLISION_DISFAVORED_REGIONS
        ]
        working_candidates = (
            preferred_candidates if len(preferred_candidates) >= 2 else partner_candidates
        )

        if len(working_candidates) < 2:
            continue

        selected: tuple[object, str, object, str, float] | None = None
        selected_score: tuple[int | float, ...] | None = None
        if selection == "cross-modality":
            signature_groups: dict[str, list[tuple[object, str, str]]] = {}
            for candidate in working_candidates:
                signature_groups.setdefault(candidate[2], []).append(candidate)
            if len(signature_groups) < 2:
                continue

            signature_items = sorted(signature_groups.items(), key=lambda item: item[0])
            for strong_group_index, (strong_signature, strong_candidates) in enumerate(signature_items):
                for weak_signature, weak_candidates in signature_items[strong_group_index + 1:]:
                    external_cross_modality = (
                        strong_signature in _EXTERNAL_MODALITY_SIGNATURES
                        and weak_signature in _EXTERNAL_MODALITY_SIGNATURES
                    )
                    cross_modality_penalty = 0 if external_cross_modality else 1
                    for strong_trace, strong_region, _ in strong_candidates:
                        strong_target_neurons = strong_trace.neurons.get(strong_region, [])
                        for weak_trace, weak_region, _ in weak_candidates:
                            weak_target_neurons = weak_trace.neurons.get(weak_region, [])
                            overlap_ratio = _region_overlap_ratio(
                                strong_trace,
                                strong_region,
                                weak_trace,
                                weak_region,
                            )
                            score = (
                                cross_modality_penalty,
                                overlap_ratio,
                                0 if strong_region == weak_region else 1,
                                abs(len(strong_target_neurons) - len(weak_target_neurons)),
                                -min(len(strong_target_neurons), len(weak_target_neurons)),
                                strong_trace.id,
                                weak_trace.id,
                            )
                            if selected_score is None or score < selected_score:
                                selected = (
                                    strong_trace,
                                    strong_region,
                                    weak_trace,
                                    weak_region,
                                    overlap_ratio,
                                )
                                selected_score = score
        else:
            for strong_index, (strong_trace, strong_region, strong_signature) in enumerate(working_candidates):
                strong_target_neurons = strong_trace.neurons.get(strong_region, [])
                for weak_trace, weak_region, weak_signature in working_candidates[strong_index + 1:]:
                    weak_target_neurons = weak_trace.neurons.get(weak_region, [])
                    overlap_ratio = _region_overlap_ratio(
                        strong_trace,
                        strong_region,
                        weak_trace,
                        weak_region,
                    )
                    family_penalty = 0 if strong_signature != weak_signature else 1
                    score = (
                        overlap_ratio,
                        family_penalty,
                        0 if strong_region == weak_region else 1,
                        abs(len(strong_target_neurons) - len(weak_target_neurons)),
                        -min(len(strong_target_neurons), len(weak_target_neurons)),
                        strong_trace.id,
                        weak_trace.id,
                    )
                    if selected_score is None or score < selected_score:
                        selected = (
                            strong_trace,
                            strong_region,
                            weak_trace,
                            weak_region,
                            overlap_ratio,
                        )
                        selected_score = score

        if selected is None:
            continue

        strong_trace, strong_region, weak_trace, weak_region, overlap_ratio = selected
        cases.append(
            (
                cue_trace,
                cue_region,
                strong_trace,
                strong_region,
                weak_trace,
                weak_region,
                overlap_ratio,
            )
        )
        used_trace_ids.update({cue_trace.id, strong_trace.id, weak_trace.id})
        if len(cases) >= max_cases:
            break

    return cases


def _binding_state(binding_id: int) -> dict[str, float | int | bool]:
    """Read binding state and pattern activation ratios for a benchmark binding."""
    if binding_id is None:
        return {
            "exists": False,
            "weight": 0.0,
            "fires": 0,
            "confidence": 0.0,
            "last_fired": 0,
            "source_pattern_ratio": 0.0,
            "source_pattern_active": False,
            "target_pattern_ratio": 0.0,
            "target_pattern_active": False,
        }

    binding_info = brain_core.get_binding_info(binding_id)
    binding_activation = brain_core.get_binding_activation(binding_id, 0.01)

    weight = 0.0
    fires = 0
    confidence = 0.0
    last_fired = 0
    source_pattern_ratio = 0.0
    target_pattern_ratio = 0.0
    if binding_info is not None:
        weight, fires, confidence, last_fired = binding_info
    if binding_activation is not None:
        source_pattern_ratio, target_pattern_ratio = binding_activation

    return {
        "exists": binding_info is not None,
        "weight": weight,
        "fires": fires,
        "confidence": confidence,
        "last_fired": last_fired,
        "source_pattern_ratio": source_pattern_ratio,
        "source_pattern_active": source_pattern_ratio >= TRACE_ACTIVATION_THRESHOLD,
        "target_pattern_ratio": target_pattern_ratio,
        "target_pattern_active": target_pattern_ratio >= TRACE_ACTIVATION_THRESHOLD,
    }


def _trace_region_activity(trace, active_set: set[int]) -> dict[str, object]:
    """Measure which trace region is carrying the most activation this tick."""
    region_ratios: dict[str, float] = {}
    region_active_counts: dict[str, int] = {}
    peak_region: str | None = None
    peak_ratio = 0.0
    peak_active_count = 0
    peak_total_neurons = 0

    for region_name, neuron_ids in trace.neurons.items():
        if not neuron_ids:
            continue
        active_count = sum(1 for neuron_id in neuron_ids if neuron_id in active_set)
        ratio = active_count / len(neuron_ids)
        region_ratios[region_name] = ratio
        region_active_counts[region_name] = active_count

        if peak_region is None or (
            ratio,
            active_count,
            -_REGION_PRIORITY.get(region_name, 999),
        ) > (
            peak_ratio,
            peak_active_count,
            -_REGION_PRIORITY.get(peak_region, 999),
        ):
            peak_region = region_name
            peak_ratio = ratio
            peak_active_count = active_count
            peak_total_neurons = len(neuron_ids)

    return {
        "region_ratios": region_ratios,
        "region_active_counts": region_active_counts,
        "peak_region": peak_region,
        "peak_ratio": peak_ratio,
        "peak_active_count": peak_active_count,
        "peak_total_neurons": peak_total_neurons,
    }

def _region_overlap_ratio(
    trace_a,
    region_a: str,
    trace_b,
    region_b: str,
) -> float:
    """Return overlap normalized by the smaller target pattern size."""
    neurons_a = set(trace_a.neurons.get(region_a, []))
    neurons_b = set(trace_b.neurons.get(region_b, []))
    if not neurons_a or not neurons_b:
        return 1.0
    return len(neurons_a & neurons_b) / max(1, min(len(neurons_a), len(neurons_b)))


# ── Run functions per dataset type ────────────────────────────────────────────

def run_text(
    ts,
    tick_loop: TickLoop,
    samples: list[dict],
    collector: MetricsCollector,
    ticks_per_sample: int,
    speech_decoder: SpeechOutput | None = None,
    rest_ticks: int = 3,
):
    """Run text learning: encode text → tick N times → collect metrics."""
    text_input = TextInput(ts)

    for i, sample in enumerate(samples):
        _begin_sample(tick_loop)
        text = sample["text"]
        label = sample.get("label_name", str(sample.get("label", "")))
        sm = collector.begin_sample(i, label=label, modality="text")
        enc = {"tokens": [], "known_count": 0, "unknown_count": 0}

        # Set label so newly formed traces get labeled for speech decode
        tick_loop.current_label = label

        # Clear residual speech activations from previous sample
        brain_core.zero_speech_activations()

        # Run ticks
        for t in range(ticks_per_sample):
            if t == 0:
                enc = text_input.encode(text)

            t0 = time.perf_counter()
            result = tick_loop.step()
            elapsed = time.perf_counter() - t0
            collector.record_tick_time(elapsed)

            # Add input context to tick result
            result["input_tokens"] = enc.get("tokens", [])[:10]
            result["input_known"] = enc.get("known_count", 0)
            result["input_unknown"] = enc.get("unknown_count", 0)

            # Decode speech output on last tick
            if t == ticks_per_sample - 1 and speech_decoder:
                # Teacher signal: ensure current label's speech neurons are
                # strongly activated so the decoder sees the correct label.
                brain_core.boost_speech(_label_speech_neurons(label), 1.0)
                speech_decoder.refresh_index()
                speech = speech_decoder.decode(top_k=5)
                result["speech_output"] = speech.get("text", "")
                result["speech_tokens"] = [tok for tok, _ in speech.get("tokens", [])]

            sm.add_tick(result)

        collector.end_sample()

        if (i + 1) % 10 == 0 or i == 0:
            speech_text = result.get('speech_output', '')
            speech_top1 = speech_text.split()[0] if speech_text else ''
            print(f"  [{i+1}/{len(samples)}] label={label}, "
                  f"active={result.get('total_active', 0)}, "
                  f"hebb={result.get('hebbian_updates', 0)}, "
                  f"traces_new={result.get('traces_formed', 0)}, "
                  f"speech='{speech_top1}'")

        # Let activity decay between samples
        if rest_ticks > 0:
            _rest_ticks(tick_loop, rest_ticks)


def run_images(
    ts,
    tick_loop: TickLoop,
    collector: MetricsCollector,
    samples: list[dict],
    ticks_per_sample: int,
    rest_ticks: int = 3,
):
    """Run image learning: encode image → tick N times → collect metrics."""
    vis_input = VisualInput()

    for i, sample in enumerate(samples):
        _begin_sample(tick_loop)
        img = sample["image"]
        label = sample.get("label_name", str(sample.get("label", "")))
        sm = collector.begin_sample(i, label=label, modality="image")

        enc = vis_input.encode(img)

        for t in range(ticks_per_sample):
            if t == 0:
                vis_input.encode(img)

            t0 = time.perf_counter()
            result = tick_loop.step()
            elapsed = time.perf_counter() - t0
            collector.record_tick_time(elapsed)
            result["input_neurons_activated"] = enc.get("neurons_activated", 0)
            sm.add_tick(result)

        collector.end_sample()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(samples)}] label={label}, "
                  f"visual={result.get('visual_activation', 0):.3f}, "
                  f"active={result.get('total_active', 0)}")

        if rest_ticks > 0:
            _rest_ticks(tick_loop, rest_ticks)


def run_audio(
    ts,
    tick_loop: TickLoop,
    collector: MetricsCollector,
    samples: list[dict],
    ticks_per_sample: int,
    rest_ticks: int = 3,
):
    """Run audio learning: encode audio → tick N times → collect metrics."""
    audio_input = AudioInput()

    for i, sample in enumerate(samples):
        _begin_sample(tick_loop)
        audio_arr = sample["audio"]
        sr = sample.get("sample_rate", 16000)
        label = sample.get("label_name", str(sample.get("label", "")))
        sm = collector.begin_sample(i, label=label, modality="audio")

        # Audio may need conversion to list of floats
        if hasattr(audio_arr, 'tolist'):
            audio_list = audio_arr.tolist()
        else:
            audio_list = list(audio_arr)

        enc = audio_input.encode(audio_list, sr)

        for t in range(ticks_per_sample):
            if t == 0:
                audio_input.encode(audio_list, sr)

            t0 = time.perf_counter()
            result = tick_loop.step()
            elapsed = time.perf_counter() - t0
            collector.record_tick_time(elapsed)
            result["input_neurons_activated"] = enc.get("neurons_activated", 0)
            sm.add_tick(result)

        collector.end_sample()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(samples)}] label={label}, "
                  f"audio={result.get('audio_activation', 0):.3f}, "
                  f"active={result.get('total_active', 0)}")

        if rest_ticks > 0:
            _rest_ticks(tick_loop, rest_ticks)


def run_multimodal(
    ts,
    tick_loop: TickLoop,
    collector: MetricsCollector,
    samples: list[dict],
    ticks_per_sample: int,
    rest_ticks: int = 3,
):
    """Run multimodal learning: text+image+audio simultaneously."""
    text_input = TextInput(ts)
    mm = MultimodalInput(
        text_encoder=text_input,
        visual_encoder=VisualInput(),
        audio_encoder=AudioInput(),
        sensory_encoder=SensoryInput(),
    )

    for i, sample in enumerate(samples):
        _begin_sample(tick_loop)
        label = f"{sample.get('text_label', '?')}+{sample.get('image_label', '?')}+{sample.get('audio_label', '?')}"
        sm = collector.begin_sample(i, label=label, modality="multimodal")

        # Prepare multimodal input dict
        audio_arr = sample["audio"]
        if hasattr(audio_arr, 'tolist'):
            audio_list = audio_arr.tolist()
        else:
            audio_list = list(audio_arr)

        inputs = {
            "text": sample["text"],
            "visual": sample["image"],
            "audio": (audio_list, sample.get("audio_sample_rate", 16000)),
        }

        for t in range(ticks_per_sample):
            if t == 0:
                enc = mm.process(inputs, tick=tick_loop.last_tick_number)

            t0 = time.perf_counter()
            result = tick_loop.step()
            elapsed = time.perf_counter() - t0
            collector.record_tick_time(elapsed)

            if t == 0:
                result["modalities_active"] = enc.get("_summary", {}).get("modality_count", 0)
                result["total_neurons_injected"] = enc.get("_summary", {}).get("total_neurons_activated", 0)

            sm.add_tick(result)

        collector.end_sample()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(samples)}] {label}, "
                  f"active={result.get('total_active', 0)}, "
                  f"hebb={result.get('hebbian_updates', 0)}")

        if rest_ticks > 0:
            _rest_ticks(tick_loop, rest_ticks)


def run_matched_traces(
    ts,
    tick_loop: TickLoop,
    collector: MetricsCollector,
    max_samples: int,
    ticks_per_sample: int,
    cue_fraction: float = 1.0,
    cue_noise_fraction: float = 0.0,
    cue_mode: str = "global",
    rest_ticks: int = 3,
):
    """Run a synthetic matched-trace workload using seeded traces as direct inputs."""
    rng = random.Random(_CUE_BENCHMARK_SEED)
    traces = [trace for trace in ts.traces.values() if trace.total_neurons() > 0]
    traces.sort(key=lambda trace: (trace.total_neurons(), trace.id), reverse=True)
    traces = traces[:max_samples]

    for i, trace in enumerate(traces):
        _begin_sample(tick_loop)
        label = trace.label or trace.id
        sm = collector.begin_sample(i, label=label, modality="matched_trace")
        injected, cue_signal_neurons, cue_noise_neurons = _build_trace_cue(
            trace,
            cue_fraction,
            cue_noise_fraction,
            rng,
            cue_mode=cue_mode,
        )

        for t in range(ticks_per_sample):
            if t == 0:
                brain_core.inject_activations(injected)

            t0 = time.perf_counter()
            result = tick_loop.step()
            elapsed = time.perf_counter() - t0
            collector.record_tick_time(elapsed)

            matched_rank = None
            matched_score = 0.0
            for rank, (active_trace_id, score) in enumerate(
                tick_loop.last_active_traces,
                start=1,
            ):
                if active_trace_id == trace.id:
                    matched_rank = rank
                    matched_score = score
                    break

            result["matched_trace_id"] = trace.id
            result["matched_trace_regions"] = sorted(trace.neurons.keys())
            result["matched_trace_neurons"] = trace.total_neurons()
            result["cue_fraction"] = cue_fraction
            result["cue_noise_fraction"] = cue_noise_fraction
            result["cue_mode"] = cue_mode
            result["cue_signal_neurons"] = cue_signal_neurons
            result["cue_noise_neurons"] = cue_noise_neurons
            result["matched_trace_hit"] = matched_rank is not None
            result["matched_trace_score"] = matched_score
            if matched_rank is not None:
                result["matched_trace_rank"] = matched_rank
            sm.add_tick(result)

        collector.end_sample()

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(traces)}] trace={trace.id}, "
                f"active_traces={result.get('active_traces', 0)}, "
                f"working_memory={result.get('working_memory', 0)}, "
                f"eval_ms={result.get('evaluation_ms', 0):.3f}"
            )

        if rest_ticks > 0:
            _rest_ticks(tick_loop, rest_ticks)


def run_cue_pairs(
    ts,
    tick_loop: TickLoop,
    collector: MetricsCollector,
    max_samples: int,
    ticks_per_sample: int,
    cue_fraction: float = 1.0,
    cue_noise_fraction: float = 0.0,
    cue_mode: str = "global",
    rest_ticks: int = 3,
):
    """Run a cue→partner probe with synthetic bindings between seeded traces."""
    rng = random.Random(_CUE_BENCHMARK_SEED)
    traces = [trace for trace in ts.traces.values() if trace.total_neurons() > 0]
    traces.sort(key=lambda trace: (trace.total_neurons(), trace.id), reverse=True)
    pairs = _select_cue_pairs(traces, max_samples)

    if not pairs:
        print("  No cue pairs available from current seed.")
        return

    for i, (cue_trace, cue_region, partner_trace, partner_region) in enumerate(pairs):
        _begin_sample(tick_loop)
        label = f"{cue_trace.label or cue_trace.id}->{partner_trace.label or partner_trace.id}"
        sm = collector.begin_sample(i, label=label, modality="cue_pair")
        binding_id = brain_core.create_binding(
            cue_region,
            list(cue_trace.neurons[cue_region]),
            TRACE_ACTIVATION_THRESHOLD,
            partner_region,
            list(partner_trace.neurons[partner_region]),
            TRACE_ACTIVATION_THRESHOLD,
            0.0,
        )
        injected, cue_signal_neurons, cue_noise_neurons = _build_trace_cue(
            cue_trace,
            cue_fraction,
            cue_noise_fraction,
            rng,
            cue_mode=cue_mode,
            focus_region=cue_region,
        )

        for t in range(ticks_per_sample):
            if t == 0:
                brain_core.inject_activations(injected)

            t0 = time.perf_counter()
            result = tick_loop.step()
            elapsed = time.perf_counter() - t0
            collector.record_tick_time(elapsed)

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
            partial_bindings = set(brain_core.find_partial_bindings(0.01))
            active_bindings = {
                active_binding_id
                for active_binding_id, _ in brain_core.evaluate_bindings(0.01)
            }
            binding_state = _binding_state(binding_id)

            false_partner_activation = false_trace_id is not None and (
                partner_rank is None
                or (false_trace_rank is not None and false_trace_rank < partner_rank)
            )

            result["cue_trace_id"] = cue_trace.id
            result["cue_trace_label"] = cue_trace.label or cue_trace.id
            result["cue_trace_region"] = cue_region
            result["cue_trace_neurons"] = cue_trace.total_neurons()
            result["cue_trace_hit"] = cue_hit
            result["cue_trace_score"] = cue_score
            result["partner_trace_id"] = partner_trace.id
            result["partner_trace_label"] = partner_trace.label or partner_trace.id
            result["partner_trace_region"] = partner_region
            result["partner_trace_neurons"] = partner_trace.total_neurons()
            result["partner_trace_hit"] = partner_hit
            result["partner_trace_score"] = partner_score
            result["binding_id"] = binding_id
            result["binding_exists"] = binding_state["exists"]
            result["binding_partial"] = binding_id in partial_bindings
            result["binding_active"] = binding_id in active_bindings
            result["binding_weight"] = binding_state["weight"]
            result["binding_fires"] = binding_state["fires"]
            result["binding_confidence"] = binding_state["confidence"]
            result["binding_last_fired"] = binding_state["last_fired"]
            result["cue_pattern_ratio"] = binding_state["source_pattern_ratio"]
            result["cue_pattern_active"] = binding_state["source_pattern_active"]
            result["partner_pattern_ratio"] = binding_state["target_pattern_ratio"]
            result["partner_pattern_active"] = binding_state["target_pattern_active"]
            result["cue_fraction"] = cue_fraction
            result["cue_noise_fraction"] = cue_noise_fraction
            result["cue_mode"] = cue_mode
            result["cue_signal_neurons"] = cue_signal_neurons
            result["cue_noise_neurons"] = cue_noise_neurons
            result["false_partner_activation"] = false_partner_activation
            result["false_trace_id"] = false_trace_id
            result["false_trace_score"] = false_trace_score
            if cue_rank is not None:
                result["cue_trace_rank"] = cue_rank
            if partner_rank is not None:
                result["partner_trace_rank"] = partner_rank
            if false_trace_rank is not None:
                result["false_trace_rank"] = false_trace_rank
            sm.add_tick(result)

        collector.end_sample()

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(pairs)}] cue={cue_trace.id}->{partner_trace.id}, "
                f"partial={result.get('binding_partial', False)}, "
                f"partner_pattern={result.get('partner_pattern_ratio', 0):.2f}, "
                f"partner_hit={result.get('partner_trace_hit', False)}, "
                f"binding_w={result.get('binding_weight', 0):.3f}, "
                f"eval_ms={result.get('evaluation_ms', 0):.3f}"
            )

        if rest_ticks > 0:
            _rest_ticks(tick_loop, rest_ticks)


def run_cue_collisions(
    ts,
    tick_loop: TickLoop,
    collector: MetricsCollector,
    max_samples: int,
    ticks_per_sample: int,
    cue_fraction: float = 1.0,
    cue_noise_fraction: float = 0.0,
    cue_mode: str = "global",
    selection: str = "overlap-min",
    weak_binding_enabled: bool = True,
    rest_ticks: int = 3,
):
    """Run a cue→multi-partner collision probe with one strong and one weak binding."""
    rng = random.Random(_CUE_BENCHMARK_SEED)
    traces = [trace for trace in ts.traces.values() if trace.total_neurons() > 0]
    traces.sort(key=lambda trace: (trace.total_neurons(), trace.id), reverse=True)
    cases = _select_cue_collision_cases(traces, max_samples, selection=selection)

    if not cases:
        print("  No cue collision cases available from current seed.")
        return

    for i, (
        cue_trace,
        cue_region,
        strong_trace,
        strong_region,
        weak_trace,
        weak_region,
        overlap_ratio,
    ) in enumerate(cases):
        _begin_sample(tick_loop)
        strong_signature = _trace_modality_signature(strong_trace)
        weak_signature = _trace_modality_signature(weak_trace)
        label = (
            f"{cue_trace.label or cue_trace.id}->"
            f"{strong_trace.label or strong_trace.id}|"
            f"{('observe:' if not weak_binding_enabled else '')}{weak_trace.label or weak_trace.id}"
        )
        sm = collector.begin_sample(
            i,
            label=label,
            modality="cue_collision" if weak_binding_enabled else "cue_isolation",
        )

        strong_binding_id = brain_core.create_binding(
            cue_region,
            list(cue_trace.neurons[cue_region]),
            TRACE_ACTIVATION_THRESHOLD,
            strong_region,
            list(strong_trace.neurons[strong_region]),
            TRACE_ACTIVATION_THRESHOLD,
            0.0,
        )
        weak_binding_id = None
        if weak_binding_enabled:
            weak_binding_id = brain_core.create_binding(
                cue_region,
                list(cue_trace.neurons[cue_region]),
                TRACE_ACTIVATION_THRESHOLD,
                weak_region,
                list(weak_trace.neurons[weak_region]),
                TRACE_ACTIVATION_THRESHOLD,
                0.0,
            )
        for _ in range(CUE_COLLISION_STRONG_REINFORCEMENTS):
            brain_core.strengthen_binding(strong_binding_id, tick_loop.last_tick_number)
        if weak_binding_enabled:
            for _ in range(CUE_COLLISION_WEAK_MISSES):
                brain_core.record_binding_miss(weak_binding_id)

        injected, cue_signal_neurons, cue_noise_neurons = _build_trace_cue(
            cue_trace,
            cue_fraction,
            cue_noise_fraction,
            rng,
            cue_mode=cue_mode,
            focus_region=cue_region,
        )
        scoring_window_open = False
        prev_strong_recall_candidate = False

        for t in range(ticks_per_sample):
            if t == 0:
                brain_core.inject_activations(injected)

            t0 = time.perf_counter()
            result = tick_loop.step()
            elapsed = time.perf_counter() - t0
            collector.record_tick_time(elapsed)

            cue_hit, cue_rank, cue_score = _trace_match_metrics(
                tick_loop.last_active_traces,
                cue_trace.id,
            )
            strong_hit, strong_rank, strong_score = _trace_match_metrics(
                tick_loop.last_active_traces,
                strong_trace.id,
            )
            weak_hit, weak_rank, weak_score = _trace_match_metrics(
                tick_loop.last_active_traces,
                weak_trace.id,
            )

            partial_bindings = set(brain_core.find_partial_bindings(0.01))
            active_bindings = {
                active_binding_id
                for active_binding_id, _ in brain_core.evaluate_bindings(0.01)
            }
            recall_candidates = {
                binding_id: (relative_weight, source_ratio)
                for binding_id, relative_weight, source_ratio in tick_loop.last_binding_recall_candidates
            }
            strong_state = _binding_state(strong_binding_id)
            weak_state = _binding_state(weak_binding_id)
            current_snapshot = tick_loop.history.current
            active_set = current_snapshot.active_set() if current_snapshot is not None else set()
            strong_region_activity = _trace_region_activity(strong_trace, active_set)
            weak_region_activity = _trace_region_activity(weak_trace, active_set)
            strong_recall_candidate = recall_candidates.get(strong_binding_id)
            weak_recall_candidate = (
                recall_candidates.get(weak_binding_id)
                if weak_binding_id is not None
                else None
            )

            max_binding_weight = max(
                float(strong_state["weight"]),
                float(weak_state["weight"]),
                1e-9,
            )
            strong_relative_weight = float(strong_state["weight"]) / max_binding_weight
            weak_relative_weight = float(weak_state["weight"]) / max_binding_weight
            competitor_pattern_active = bool(weak_state["target_pattern_active"])
            selective_recall = bool(strong_state["target_pattern_active"]) and not competitor_pattern_active
            scoring_window_eligible = scoring_window_open or prev_strong_recall_candidate
            selective_recall_scored = scoring_window_eligible and selective_recall
            competitor_outcompetes_partner = weak_rank is not None and (
                strong_rank is None or weak_rank < strong_rank
            )

            result["cue_trace_id"] = cue_trace.id
            result["cue_trace_label"] = cue_trace.label or cue_trace.id
            result["cue_trace_region"] = cue_region
            result["cue_trace_neurons"] = cue_trace.total_neurons()
            result["cue_trace_hit"] = cue_hit
            result["cue_trace_score"] = cue_score

            result["partner_trace_id"] = strong_trace.id
            result["partner_trace_label"] = strong_trace.label or strong_trace.id
            result["partner_trace_region"] = strong_region
            result["partner_trace_signature"] = strong_signature
            result["partner_trace_neurons"] = strong_trace.total_neurons()
            result["partner_trace_hit"] = strong_hit
            result["partner_trace_score"] = strong_score
            result["partner_binding_id"] = strong_binding_id
            result["binding_id"] = strong_binding_id
            result["binding_exists"] = strong_state["exists"]
            result["binding_partial"] = strong_binding_id in partial_bindings
            result["binding_active"] = strong_binding_id in active_bindings
            result["binding_weight"] = strong_state["weight"]
            result["binding_fires"] = strong_state["fires"]
            result["binding_confidence"] = strong_state["confidence"]
            result["binding_last_fired"] = strong_state["last_fired"]
            result["strong_recall_candidate"] = strong_recall_candidate is not None
            result["strong_recall_candidate_relative_weight"] = (
                strong_recall_candidate[0] if strong_recall_candidate is not None else 0.0
            )
            result["strong_recall_candidate_source_ratio"] = (
                strong_recall_candidate[1] if strong_recall_candidate is not None else 0.0
            )
            result["cue_pattern_ratio"] = strong_state["source_pattern_ratio"]
            result["cue_pattern_active"] = strong_state["source_pattern_active"]
            result["strong_source_pattern_ratio"] = strong_state["source_pattern_ratio"]
            result["strong_source_pattern_active"] = strong_state["source_pattern_active"]
            result["partner_pattern_ratio"] = strong_state["target_pattern_ratio"]
            result["partner_pattern_active"] = strong_state["target_pattern_active"]
            result["partner_peak_region"] = strong_region_activity["peak_region"]
            result["partner_peak_region_ratio"] = strong_region_activity["peak_ratio"]
            result["partner_peak_region_active_neurons"] = strong_region_activity["peak_active_count"]
            result["partner_peak_region_total_neurons"] = strong_region_activity["peak_total_neurons"]

            result["competitor_trace_id"] = weak_trace.id
            result["competitor_trace_label"] = weak_trace.label or weak_trace.id
            result["competitor_trace_region"] = weak_region
            result["competitor_trace_signature"] = weak_signature
            result["competitor_trace_neurons"] = weak_trace.total_neurons()
            result["competitor_trace_hit"] = weak_hit
            result["competitor_trace_score"] = weak_score
            result["competitor_binding_id"] = weak_binding_id
            result["competitor_binding_present"] = weak_binding_enabled
            result["competitor_binding_exists"] = weak_state["exists"]
            result["competitor_binding_partial"] = (
                weak_binding_id in partial_bindings if weak_binding_id is not None else False
            )
            result["competitor_binding_active"] = (
                weak_binding_id in active_bindings if weak_binding_id is not None else False
            )
            result["competitor_binding_weight"] = weak_state["weight"]
            result["competitor_binding_fires"] = weak_state["fires"]
            result["competitor_binding_confidence"] = weak_state["confidence"]
            result["competitor_binding_last_fired"] = weak_state["last_fired"]
            result["weak_recall_candidate"] = weak_recall_candidate is not None
            result["weak_recall_candidate_relative_weight"] = (
                weak_recall_candidate[0] if weak_recall_candidate is not None else 0.0
            )
            result["weak_recall_candidate_source_ratio"] = (
                weak_recall_candidate[1] if weak_recall_candidate is not None else 0.0
            )
            result["binding_recall_cutoff"] = BINDING_RECALL_MIN_RELATIVE_WEIGHT
            result["weak_source_pattern_ratio"] = weak_state["source_pattern_ratio"]
            result["weak_source_pattern_active"] = weak_state["source_pattern_active"]
            result["competitor_pattern_ratio"] = weak_state["target_pattern_ratio"]
            result["competitor_pattern_active"] = weak_state["target_pattern_active"]
            result["competitor_peak_region"] = weak_region_activity["peak_region"]
            result["competitor_peak_region_ratio"] = weak_region_activity["peak_ratio"]
            result["competitor_peak_region_active_neurons"] = weak_region_activity["peak_active_count"]
            result["competitor_peak_region_total_neurons"] = weak_region_activity["peak_total_neurons"]
            result["competitor_region_ratios"] = weak_region_activity["region_ratios"]

            result["strong_binding_relative_weight"] = strong_relative_weight
            result["weak_binding_relative_weight"] = weak_relative_weight
            result["binding_weight_margin"] = (
                float(strong_state["weight"]) - float(weak_state["weight"])
            )
            result["pattern_ratio_margin"] = (
                float(strong_state["target_pattern_ratio"])
                - float(weak_state["target_pattern_ratio"])
            )
            result["trace_score_margin"] = strong_score - weak_score
            result["selective_recall"] = selective_recall
            result["selective_recall_window_eligible"] = scoring_window_eligible
            result["selective_recall_scored"] = selective_recall_scored
            result["competitor_leak"] = competitor_pattern_active or weak_hit
            result["competitor_outcompetes_partner"] = competitor_outcompetes_partner
            result["collision_strong_reinforcements"] = CUE_COLLISION_STRONG_REINFORCEMENTS
            result["collision_weak_misses"] = CUE_COLLISION_WEAK_MISSES
            result["collision_selection"] = selection
            result["collision_target_overlap_ratio"] = overlap_ratio
            result["collision_same_target_region"] = strong_region == weak_region
            result["cue_fraction"] = cue_fraction
            result["cue_noise_fraction"] = cue_noise_fraction
            result["cue_mode"] = cue_mode
            result["cue_signal_neurons"] = cue_signal_neurons
            result["cue_noise_neurons"] = cue_noise_neurons
            if cue_rank is not None:
                result["cue_trace_rank"] = cue_rank
            if strong_rank is not None:
                result["partner_trace_rank"] = strong_rank
            if weak_rank is not None:
                result["competitor_trace_rank"] = weak_rank
            sm.add_tick(result)
            scoring_window_open = scoring_window_eligible
            prev_strong_recall_candidate = strong_recall_candidate is not None

        collector.end_sample()

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(cases)}] cue={cue_trace.id}, "
                f"strong_w={result.get('binding_weight', 0):.3f}, "
                f"weak_w={result.get('competitor_binding_weight', 0):.3f}, "
                f"strong_p={result.get('partner_pattern_ratio', 0):.2f}, "
                f"weak_p={result.get('competitor_pattern_ratio', 0):.2f}, "
                f"weak_bind={weak_binding_enabled}, "
                f"weak_reg={result.get('competitor_peak_region')}, "
                f"selective={result.get('selective_recall', False)}"
            )

        if rest_ticks > 0:
            _rest_ticks(tick_loop, rest_ticks)


# ── Main session runner ──────────────────────────────────────────────────────

def run_session(
    dataset: str,
    max_samples: int,
    ticks_per_sample: int,
    threads: int,
    mode: str,
    output_path: str,
    save_dir: str | None = None,
    compact: bool = False,
    fast: bool = True,
    n_traces: int = 5000,
    seed_chunks: int | None = None,
    cue_fraction: float = 1.0,
    cue_noise_fraction: float = 0.0,
    cue_mode: str = "global",
    collision_selection: str = "overlap-min",
    weak_binding_enabled: bool = True,
    ablate_short_same_region: bool = False,
    ablation_regions: list[str] | None = None,
    ablation_delays: list[int] | None = None,
    rest_ticks: int = 3,
) -> MetricsCollector:
    """Run a full learning session."""

    actual_threads = _set_threads(threads)
    print(f"Threads: {actual_threads}")

    # Seed the brain
    ts = _seed_and_build(
        verbose=True,
        fast=fast,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
    )
    applied_ablation_regions, applied_ablation_delays = _configure_same_region_delay_ablation(
        ablate_short_same_region,
        regions=ablation_regions,
        delays=ablation_delays,
    )
    tick_loop = TickLoop(ts)
    speech_decoder = SpeechOutput(ts)

    # Warmup tick — first tick is slow (rayon thread pool init, CSR cache load)
    warmup_ms = _warmup_session(tick_loop)

    collector = MetricsCollector(
        dataset=dataset,
        mode=mode,
        threads=actual_threads,
        ticks_per_sample=ticks_per_sample,
        extra_config={
            "max_samples": max_samples,
            "neuron_count": brain_core.get_neuron_count(),
            "synapse_count": brain_core.get_synapse_count(),
            "trace_count": len(ts),
            "seed_chunks": seed_chunks,
            "cue_fraction": cue_fraction,
            "cue_noise_fraction": cue_noise_fraction,
            "cue_mode": cue_mode,
            "collision_selection": collision_selection,
            "weak_binding_enabled": weak_binding_enabled,
            "ablate_short_same_region": ablate_short_same_region,
            "ablation_regions": applied_ablation_regions,
            "ablation_delays": applied_ablation_delays,
            "collision_strong_reinforcements": CUE_COLLISION_STRONG_REINFORCEMENTS,
            "collision_weak_misses": CUE_COLLISION_WEAK_MISSES,
            "warmup_ms": round(warmup_ms, 3),
        },
    )
    collector.start()

    if dataset == "all":
        _run_all_datasets(ts, tick_loop, collector, max_samples, ticks_per_sample,
                          speech_decoder, mode, fast=fast, n_traces=n_traces,
                          seed_chunks=seed_chunks,
                          ablate_short_same_region=ablate_short_same_region,
                          ablation_regions=applied_ablation_regions,
                          ablation_delays=applied_ablation_delays,
                          rest_ticks=rest_ticks)
    elif dataset == "ag_news" or dataset == "imdb":
        print(f"\nLoading {dataset}...")
        from brain.datasets.downloader import load_text_dataset
        data = load_text_dataset(dataset, max_samples=max_samples)
        print(f"  Got {len(data)} samples")
        print(f"\nRunning text learning ({len(data)} samples, {ticks_per_sample} ticks each)...")
        run_text(ts, tick_loop, data, collector, ticks_per_sample, speech_decoder,
            rest_ticks=rest_ticks)

    elif dataset == "cifar10":
        print(f"\nLoading {dataset}...")
        from brain.datasets.downloader import load_image_dataset
        data = load_image_dataset(dataset, max_samples=max_samples)
        print(f"  Got {len(data)} samples")
        print(f"\nRunning image learning ({len(data)} samples, {ticks_per_sample} ticks each)...")
        run_images(ts, tick_loop, collector, data, ticks_per_sample,
                  rest_ticks=rest_ticks)

    elif dataset == "speech_commands":
        print(f"\nLoading {dataset}...")
        from brain.datasets.downloader import load_audio_dataset
        data = load_audio_dataset(dataset, max_samples=max_samples)
        print(f"  Got {len(data)} samples")
        print(f"\nRunning audio learning ({len(data)} samples, {ticks_per_sample} ticks each)...")
        run_audio(ts, tick_loop, collector, data, ticks_per_sample,
                 rest_ticks=rest_ticks)

    elif dataset == "multimodal":
        print("\nLoading multimodal datasets (text+image+audio)...")
        from brain.datasets.downloader import load_multimodal_batch
        data = load_multimodal_batch(max_samples=max_samples)
        print(f"  Got {len(data)} aligned samples")
        print(f"\nRunning multimodal learning ({len(data)} samples, {ticks_per_sample} ticks each)...")
        run_multimodal(ts, tick_loop, collector, data, ticks_per_sample,
                      rest_ticks=rest_ticks)

    elif dataset == "matched_traces":
        print(f"\nRunning matched-trace workload ({max_samples} traces, {ticks_per_sample} ticks each)...")
        run_matched_traces(
            ts,
            tick_loop,
            collector,
            max_samples=max_samples,
            ticks_per_sample=ticks_per_sample,
            cue_fraction=cue_fraction,
            cue_noise_fraction=cue_noise_fraction,
            cue_mode=cue_mode,
            rest_ticks=rest_ticks,
        )

    elif dataset == "cue_pairs":
        print(f"\nRunning cue-pair workload ({max_samples} pairs, {ticks_per_sample} ticks each)...")
        run_cue_pairs(
            ts,
            tick_loop,
            collector,
            max_samples=max_samples,
            ticks_per_sample=ticks_per_sample,
            cue_fraction=cue_fraction,
            cue_noise_fraction=cue_noise_fraction,
            cue_mode=cue_mode,
            rest_ticks=rest_ticks,
        )

    elif dataset == "cue_collisions":
        print(f"\nRunning cue-collision workload ({max_samples} cases, {ticks_per_sample} ticks each)...")
        run_cue_collisions(
            ts,
            tick_loop,
            collector,
            max_samples=max_samples,
            ticks_per_sample=ticks_per_sample,
            cue_fraction=cue_fraction,
            cue_noise_fraction=cue_noise_fraction,
            cue_mode=cue_mode,
            selection=collision_selection,
            weak_binding_enabled=weak_binding_enabled,
            rest_ticks=rest_ticks,
        )

    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)

    collector.finish()
    collector.print_summary()
    collector.save(output_path, compact=compact)
    print(f"Metrics saved to: {output_path}")

    # Save brain state if requested
    if save_dir:
        meta = save_brain(ts, save_dir, extra_metadata={
            "dataset": dataset,
            "mode": mode,
            "samples_processed": len(collector.samples),
        })
        print(f"Brain state saved to: {save_dir} (tick {meta['tick']})")

    return collector


def _run_all_datasets(
    ts,
    tick_loop: TickLoop,
    collector: MetricsCollector,
    max_samples: int,
    ticks_per_sample: int,
    speech_decoder: SpeechOutput,
    mode: str,
    fast: bool = True,
    n_traces: int = 5000,
    seed_chunks: int | None = None,
    ablate_short_same_region: bool = False,
    ablation_regions: list[str] | None = None,
    ablation_delays: list[int] | None = None,
    rest_ticks: int = 3,
):
    """Run all dataset types in sequence (or reset between each for 'separate' mode)."""
    from brain.datasets.downloader import (
        load_text_dataset, load_image_dataset, load_audio_dataset,
    )

    datasets_to_run = [
        ("text", "ag_news", load_text_dataset, run_text),
        ("image", "cifar10", load_image_dataset, run_images),
        ("audio", "speech_commands", load_audio_dataset, run_audio),
    ]

    for dtype, dname, loader, runner in datasets_to_run:
        print(f"\n{'='*50}")
        print(f"Dataset: {dname} ({dtype})")
        print(f"{'='*50}")

        if mode == "separate":
            # Reset brain between datasets
            print("  Resetting brain for separate mode...")
            brain_core.reset_brain()
            ts_new = _seed_and_build(
                verbose=False,
                fast=fast,
                n_traces=n_traces,
                seed_chunks=seed_chunks,
            )
            _configure_same_region_delay_ablation(
                ablate_short_same_region,
                regions=ablation_regions,
                delays=ablation_delays,
            )
            # Re-create tick loop with fresh state
            tick_loop_new = TickLoop(ts_new)
            speech_new = SpeechOutput(ts_new) if dtype == "text" else None
        else:
            ts_new = ts
            tick_loop_new = tick_loop
            speech_new = speech_decoder if dtype == "text" else None

        data = loader(dname, max_samples=max_samples)
        print(f"  Loaded {len(data)} samples")

        if dtype == "text":
                 runner(ts_new, tick_loop_new, data, collector, ticks_per_sample, speech_new,
                     rest_ticks=rest_ticks)
        else:
            runner(ts_new, tick_loop_new, collector, data, ticks_per_sample,
                   rest_ticks=rest_ticks)


# ── Compare sequential vs separate ──────────────────────────────────────────

def run_comparison(
    max_samples: int,
    ticks_per_sample: int,
    threads: int,
    output_dir: str,
    fast: bool = True,
    n_traces: int = 5000,
    seed_chunks: int | None = None,
    ablate_short_same_region: bool = False,
    ablation_regions: list[str] | None = None,
    ablation_delays: list[int] | None = None,
    rest_ticks: int = 3,
):
    """Run the same datasets in both sequential and separate modes, then compare."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("COMPARISON: Sequential vs Separate learning")
    print("="*70)

    # Sequential run
    print("\n>>> SEQUENTIAL RUN (brain accumulates across datasets)")
    seq = run_session(
        dataset="all",
        max_samples=max_samples,
        ticks_per_sample=ticks_per_sample,
        threads=threads,
        mode="sequential",
        output_path=str(out / "metrics_sequential.json"),
        fast=fast,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        ablate_short_same_region=ablate_short_same_region,
        ablation_regions=ablation_regions,
        ablation_delays=ablation_delays,
        rest_ticks=rest_ticks,
    )

    # Separate run
    print("\n>>> SEPARATE RUN (brain resets between datasets)")
    sep = run_session(
        dataset="all",
        max_samples=max_samples,
        ticks_per_sample=ticks_per_sample,
        threads=threads,
        mode="separate",
        output_path=str(out / "metrics_separate.json"),
        fast=fast,
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        ablate_short_same_region=ablate_short_same_region,
        ablation_regions=ablation_regions,
        ablation_delays=ablation_delays,
        rest_ticks=rest_ticks,
    )

    # Compare
    seq_g = seq.global_summary()
    sep_g = sep.global_summary()

    comparison = {
        "sequential": seq_g,
        "separate": sep_g,
        "differences": {},
    }

    for key in seq_g:
        if isinstance(seq_g.get(key), (int, float)) and isinstance(sep_g.get(key), (int, float)):
            diff = seq_g[key] - sep_g[key]
            if seq_g[key] != 0:
                pct = round(100 * diff / abs(seq_g[key]), 1)
            else:
                pct = 0.0
            comparison["differences"][key] = {
                "sequential": seq_g[key],
                "separate": sep_g[key],
                "diff": diff,
                "pct_change": pct,
            }

    with open(out / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    for key in ("ticks_per_sec", "hebbian_updates_total", "traces_formed_total",
                "total_active_avg", "novelty_avg", "duration_sec"):
        d = comparison["differences"].get(key)
        if d:
            print(f"  {key:30s}  seq={d['sequential']:>10}  sep={d['separate']:>10}  "
                  f"diff={d['pct_change']:+.1f}%")

    print(f"\nComparison saved to: {out / 'comparison.json'}")


def run_fixed_overlay_triplet(
    base_dataset: str,
    max_samples: int,
    ticks_per_sample: int,
    threads: int,
    output_path: str,
    compact: bool = False,
    fast: bool = True,
    n_traces: int = 5000,
    seed_chunks: int | None = None,
    rest_ticks: int = 3,
) -> dict[str, object]:
    """Run the canonical fixed-graph triplet on one structural graph."""

    if base_dataset not in _FIXED_GRAPH_TEXT_DATASETS:
        raise ValueError(
            "fixed_overlay_triplet currently supports only text datasets: "
            f"{', '.join(_FIXED_GRAPH_TEXT_DATASETS)}"
        )

    actual_threads = _set_threads(threads)
    print(f"Threads: {actual_threads}")

    print("\nLoading text benchmark dataset...")
    from brain.datasets.downloader import load_text_dataset

    samples = load_text_dataset(base_dataset, max_samples=max_samples)
    print(f"  Got {len(samples)} samples from {base_dataset}")

    spec = build_fixed_graph_spec(
        fast=fast,
        n_traces=n_traces,
        chunk_count=seed_chunks,
        overlay_regions=DEFAULT_FIXED_GRAPH_OVERLAY_REGIONS,
        verbose=True,
    )
    overlay_graph_synapses = spec.structural_synapses + spec.overlay_synapses

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = output.parent / f"{output.stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    condition_results: dict[str, dict[str, object]] = {}
    structural_summary: dict[str, object] | None = None

    for condition_name, use_overlay, propagation_ablation, learning_ablation in _FIXED_GRAPH_TRIPLET:
        print("\n" + "=" * 70)
        print(f"FIXED GRAPH CONDITION: {condition_name}")
        print("=" * 70)

        brain_core.init_brain_with_synapses(
            overlay_graph_synapses if use_overlay else spec.structural_synapses
        )
        brain_core.clear_same_region_delay_ablation()
        brain_core.clear_same_region_delay_learning_ablation()
        if propagation_ablation:
            brain_core.set_same_region_delay_ablation(
                list(spec.overlay_regions),
                list(DEFAULT_FIXED_GRAPH_OVERLAY_DELAYS),
            )
        if learning_ablation:
            brain_core.set_same_region_delay_learning_ablation(
                list(spec.overlay_regions),
                list(DEFAULT_FIXED_GRAPH_OVERLAY_DELAYS),
            )

        trace_store = spec.clone_trace_store()
        tick_loop = TickLoop(trace_store)
        speech_decoder = SpeechOutput(trace_store)
        warmup_ms = _warmup_session(tick_loop)

        collector = MetricsCollector(
            dataset="fixed_overlay_triplet",
            mode=condition_name,
            threads=actual_threads,
            ticks_per_sample=ticks_per_sample,
            extra_config={
                "benchmark_dataset": base_dataset,
                "max_samples": len(samples),
                "rest_ticks": rest_ticks,
                "fast": fast,
                "seed_traces": n_traces,
                "seed_chunks": spec.chunk_count,
                "structural_synapse_count": spec.structural_synapse_count,
                "overlay_synapse_count": spec.overlay_synapse_count,
                "overlay_graph_synapse_count": spec.overlay_graph_synapse_count,
                "overlay_regions": list(spec.overlay_regions),
                "overlay_delays": list(DEFAULT_FIXED_GRAPH_OVERLAY_DELAYS),
                "overlay_delay_histogram": spec.overlay_delay_histogram,
                "overlay_weight_values": spec.overlay_weight_values,
                "use_overlay": use_overlay,
                "propagation_ablation": propagation_ablation,
                "learning_ablation": learning_ablation,
                "warmup_ms": round(warmup_ms, 3),
            },
        )
        collector.start()
        run_text(
            trace_store,
            tick_loop,
            samples,
            collector,
            ticks_per_sample,
            speech_decoder,
            rest_ticks=rest_ticks,
        )
        collector.finish()

        condition_output = artifact_dir / f"{condition_name}.json"
        collector.save(condition_output, compact=compact)
        summary = collector.global_summary()
        condition_results[condition_name] = {
            "summary": summary,
            "artifact": str(condition_output),
            "config": {
                "use_overlay": use_overlay,
                "propagation_ablation": propagation_ablation,
                "learning_ablation": learning_ablation,
            },
        }
        if structural_summary is None:
            structural_summary = summary

        print(
            "  Summary: "
            f"active={summary.get('total_active_avg', 0):.3f}, "
            f"trace_candidates={summary.get('trace_candidates_avg', 0):.3f}, "
            f"binding_candidates={summary.get('binding_candidates_avg', 0):.3f}, "
            f"binding_per_trace={summary.get('binding_candidates_per_trace_candidate_avg', 0):.5f}, "
            f"bindings_formed={summary.get('bindings_formed_total', 0)}, "
            f"ticks_per_sec={summary.get('ticks_per_sec', 0)}"
        )

        del speech_decoder
        del tick_loop
        del trace_store
        del collector
        brain_core.clear_same_region_delay_ablation()
        brain_core.clear_same_region_delay_learning_ablation()

    deltas_vs_structural: dict[str, dict[str, float]] = {}
    if structural_summary is not None:
        for condition_name, result in condition_results.items():
            summary = result["summary"]
            metric_deltas: dict[str, float] = {}
            for key in _FIXED_GRAPH_SUMMARY_KEYS:
                ref = structural_summary.get(key)
                cur = summary.get(key)
                if isinstance(ref, (int, float)) and isinstance(cur, (int, float)):
                    metric_deltas[key] = round(cur - ref, 5)
            deltas_vs_structural[condition_name] = metric_deltas

    aggregate = {
        "benchmark": "fixed_overlay_triplet",
        "benchmark_dataset": base_dataset,
        "threads": actual_threads,
        "samples": len(samples),
        "ticks_per_sample": ticks_per_sample,
        "rest_ticks": rest_ticks,
        "fast": fast,
        "seed_traces": n_traces,
        "seed_chunks": spec.chunk_count,
        "overlay_spec": {
            "regions": list(spec.overlay_regions),
            "delays": list(DEFAULT_FIXED_GRAPH_OVERLAY_DELAYS),
            "delay_histogram": spec.overlay_delay_histogram,
            "weight_values": spec.overlay_weight_values,
            "structural_synapse_count": spec.structural_synapse_count,
            "overlay_synapse_count": spec.overlay_synapse_count,
            "overlay_graph_synapse_count": spec.overlay_graph_synapse_count,
        },
        "conditions": condition_results,
        "deltas_vs_structural": deltas_vs_structural,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("FIXED GRAPH TRIPLET RESULTS")
    print("=" * 70)
    for condition_name, result in condition_results.items():
        summary = result["summary"]
        print(
            f"  {condition_name:28s} "
            f"active={summary.get('total_active_avg', 0):>10.3f} "
            f"trace={summary.get('trace_candidates_avg', 0):>10.3f} "
            f"bind={summary.get('binding_candidates_avg', 0):>10.3f} "
            f"ratio={summary.get('binding_candidates_per_trace_candidate_avg', 0):>8.5f} "
            f"formed={summary.get('bindings_formed_total', 0):>8}"
        )

    print(f"\nAggregate metrics saved to: {output}")
    print(f"Per-condition metrics saved to: {artifact_dir}")
    return aggregate


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Brain — Real Learning with HuggingFace Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset ag_news --samples 100 --ticks 10
  python main.py --dataset cifar10 --samples 50 --ticks 15
  python main.py --dataset speech_commands --samples 50
  python main.py --dataset multimodal --samples 30
        python main.py --dataset async_multi_brain --samples 50 --ticks 5 --threads 2 --async-workers 4 --async-cores-per-worker 2 --merge-every-samples 5 --fast --seed-traces 5000 --seed-chunks 1 --rest-ticks 0
    python main.py --dataset fixed_overlay_triplet --fixed-graph-base-dataset ag_news --samples 50 --ticks 5 --threads 1 --compact --fast --seed-traces 5000 --seed-chunks 1 --rest-ticks 0
        python main.py --dataset text_learning_probe --samples 4 --ticks 12 --probe-train-repeats 6 --threads 1 --fast --seed-traces 5500 --seed-chunks 1 --rest-ticks 1
        python main.py --dataset audio_learning_probe --samples 4 --ticks 6 --probe-train-repeats 6 --probe-settle-ticks 3 --probe-ticks 4 --cue-fraction 0.75 --threads 1 --fast --seed-traces 5500 --seed-chunks 1 --rest-ticks 1
        python main.py --dataset visual_learning_probe --samples 4 --ticks 12 --probe-train-repeats 6 --probe-settle-ticks 3 --probe-ticks 4 --cue-fraction 0.75 --threads 1 --fast --seed-traces 5500 --seed-chunks 1 --rest-ticks 1
    python main.py --dataset multimodal_binding_probe --ticks 12 --probe-train-repeats 6 --probe-settle-ticks 3 --probe-ticks 4 --cue-fraction 1.0 --threads 1 --fast --seed-traces 5500 --seed-chunks 1 --rest-ticks 1
    python main.py --dataset multimodal_stability_probe --samples 120 --ticks 10 --threads 1 --fast --seed-traces 5500 --seed-chunks 1 --rest-ticks 1
    python main.py --dataset output_region_probe --samples 32 --ticks 10 --threads 1 --fast --seed-traces 5500 --seed-chunks 1
    python main.py --dataset phase11_operational_baseline --samples 120 --ticks 10 --threads 1 --full-seed --seed-chunks 1 --rest-ticks 1
    python main.py --dataset executive_numbers_probe --ticks 5 --threads 1 --full-seed --seed-chunks 1
    python main.py --dataset crossmodal_recall_probe --ticks 12 --probe-settle-ticks 3 --probe-ticks 4 --cue-fraction 1.0 --threads 1 --fast --seed-traces 5500 --seed-chunks 1 --rest-ticks 1
                python main.py --dataset text_binding_probe --samples 4 --ticks 12 --probe-train-repeats 6 --probe-settle-ticks 3 --probe-ticks 4 --cue-fraction 0.75 --cue-mode dominant-region --threads 1 --fast --seed-traces 5500 --seed-chunks 1 --rest-ticks 1
        python main.py --dataset text_vocab_profile --text-profile-dataset ag_news --samples 200 --profile-seed-traces 5000,20000 --seed-chunks 1 --output results/text_vocab_profile_ag_news.json
    python main.py --dataset matched_traces --samples 30 --ticks 5 --cue-fraction 0.5 --cue-noise-fraction 0.25 --cue-mode per-region
    python main.py --dataset cue_pairs --samples 20 --ticks 5 --cue-fraction 0.75 --cue-mode dominant-region
    python main.py --dataset cue_collisions --samples 10 --ticks 5 --cue-fraction 0.75 --cue-mode dominant-region
      python main.py --dataset ag_news --samples 20 --ticks 5 --threads 1 --compact --fast --seed-traces 5000 --seed-chunks 1 --rest-ticks 0 --ablate-short-same-region
        python main.py --dataset cue_collisions --samples 1 --ticks 3 --cue-mode dominant-region --disable-weak-binding
        python main.py --dataset cue_collisions --samples 1 --ticks 3 --cue-mode dominant-region --collision-selection cross-modality
  python main.py --dataset all --samples 30 --mode sequential
  python main.py --dataset all --samples 30 --mode separate
  python main.py --compare --samples 30 --ticks 10
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=["ag_news", "imdb", "cifar10", "speech_commands", "multimodal", "matched_traces", "cue_pairs", "cue_collisions", "async_multi_brain", "fixed_overlay_triplet", "text_learning_probe", "audio_learning_probe", "visual_learning_probe", "multimodal_binding_probe", "multimodal_stability_probe", "output_region_probe", "phase11_operational_baseline", "executive_numbers_probe", "crossmodal_recall_probe", "text_binding_probe", "text_vocab_profile", "coding_assistant_probe", "end_to_end_demo", "opus_reasoning", "all"],
        default="ag_news",
        help="Dataset to learn from (default: ag_news)",
    )
    parser.add_argument("--fixed-graph-base-dataset", choices=_FIXED_GRAPH_TEXT_DATASETS, default="ag_news",
                        help="Text dataset to use inside the fixed_overlay_triplet benchmark (default: ag_news)")
    parser.add_argument("--async-text-dataset", choices=_FIXED_GRAPH_TEXT_DATASETS, default="ag_news",
                        help="Text dataset to use inside async_multi_brain (default: ag_news)")
    parser.add_argument("--text-profile-dataset", choices=_FIXED_GRAPH_TEXT_DATASETS, default="ag_news",
                        help="Text dataset to use inside text_vocab_profile (default: ag_news)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to process (default: 100)")
    parser.add_argument("--ticks", type=int, default=10,
                        help="Ticks per sample (default: 10)")
    parser.add_argument("--threads", type=int, default=0,
                        help="CPU threads, 0=auto (default: 0)")
    parser.add_argument("--mode", choices=["sequential", "separate"], default="sequential",
                        help="Learning mode for 'all' dataset (default: sequential)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output metrics JSON path (default: results/<dataset>_metrics.json)")
    parser.add_argument("--save-brain", type=str, default=None,
                        help="Save brain state to this directory after learning")
    parser.add_argument("--compare", action="store_true",
                        help="Run sequential vs separate comparison")
    parser.add_argument("--compact", action="store_true",
                        help="Save compact metrics (summaries only, no per-tick data)")
    parser.add_argument("--fast", action="store_true", default=True,
                        help="Use fast seed (5k traces, default: on)")
    parser.add_argument("--full-seed", action="store_true",
                        help="Use full seed (100k traces, slow but complete)")
    parser.add_argument("--seed-traces", type=int, default=5000,
                        help="Number of seed traces for fast mode (default: 5000)")
    parser.add_argument("--seed-chunks", type=int, default=None,
                        help="Chunk count to use during seeding, decoupled from execution threads (default: match thread count)")
    parser.add_argument("--cue-fraction", type=float, default=1.0,
                        help="Fraction of cue trace neurons to inject for matched/cue benchmark workloads (default: 1.0)")
    parser.add_argument("--cue-noise-fraction", type=float, default=0.0,
                        help="Noise neurons injected relative to cue signal count for matched/cue benchmark workloads (default: 0.0)")
    parser.add_argument("--cue-mode", choices=["global", "per-region", "dominant-region"], default="global",
                        help="How to subsample cues for matched/cue benchmark workloads (default: global)")
    parser.add_argument("--collision-selection", choices=["overlap-min", "cross-modality"], default="overlap-min",
                        help="How to choose strong/weak competitor targets for cue-collision workloads (default: overlap-min)")
    parser.add_argument("--disable-weak-binding", action="store_true",
                        help="Observe the weak target trace without creating its competitor binding in cue-collision workloads")
    parser.add_argument("--ablate-short-same-region", action="store_true",
                        help="Suppress same-region synapses with the configured short delays at runtime for the selected regions")
    parser.add_argument("--ablation-regions", type=str, default=None,
                        help="Comma-separated region names for short-delay ablation (default: visual,language,executive,memory_long)")
    parser.add_argument("--ablation-delays", type=str, default=None,
                        help="Comma-separated raw synapse delays to suppress for the selected regions (default: 1,2)")
    parser.add_argument("--rest-ticks", type=int, default=3,
                        help="Idle ticks between samples to let activity decay (default: 3)")
    parser.add_argument("--probe-train-repeats", type=int, default=6,
                        help="Repeat count for learning-probe training before cue probes (default: 6)")
    parser.add_argument("--probe-settle-ticks", type=int, default=3,
                        help="Idle ticks before and after each cue in learning probes (default: 3)")
    parser.add_argument("--probe-ticks", type=int, default=4,
                        help="Ticks to observe each cue in learning probes (default: 4)")
    parser.add_argument("--profile-seed-traces", type=str, default="5000,20000",
                        help="Comma-separated fast-seed trace counts to compare for text_vocab_profile (default: 5000,20000)")
    parser.add_argument("--text-overlay-terms", type=int, default=0,
                        help="Number of ranked unknown content tokens to add as labeled overlay traces for text benchmarks (default: 0)")
    parser.add_argument("--text-overlay-samples", type=int, default=500,
                        help="Dataset sample count used to rank unknown content tokens for overlay selection (default: 500)")
    parser.add_argument("--text-overlay-working-memory-cap", type=int, default=None,
                        help="Optional cap on how many overlay traces may occupy working memory simultaneously during text binding-probe harvest (default: derive from working-memory capacity when overlay is enabled)")
    parser.add_argument("--async-workers", type=int, default=DEFAULT_ASYNC_MULTI_BRAIN_WORKER_COUNT,
                        help="Worker process count for async_multi_brain (default: 4)")
    parser.add_argument("--async-cores-per-worker", type=int, default=DEFAULT_ASYNC_MULTI_BRAIN_CORES_PER_WORKER,
                        help="Rayon thread count per async worker process (default: 2)")
    parser.add_argument("--merge-every-samples", type=int, default=DEFAULT_ASYNC_MULTI_BRAIN_MERGE_EVERY_SAMPLES,
                        help="Local sample count per worker between async merge rounds (default: 5)")
    parser.add_argument("--async-skip-validation", action="store_true",
                        help="Skip the post-training Phase 11 and crossmodal validation probes for async_multi_brain")
    parser.add_argument("--async-parallelism-diagnostic", action="store_true",
                        help="Compatibility flag for async_multi_brain parallelism benchmark runs")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Target words per chunk (default: auto-adaptive based on text length)")
    parser.add_argument("--chunk-rest-ticks", type=int, default=2,
                        help="Idle ticks between text chunks for chunking pipeline (default: 2)")

    args = parser.parse_args()

    try:
        ablation_regions = _parse_csv_strings(args.ablation_regions)
        ablation_delays = _parse_csv_ints(args.ablation_delays)
        profile_seed_traces = tuple(_parse_csv_ints(args.profile_seed_traces))
    except ValueError as exc:
        parser.error(f"Invalid ablation configuration: {exc}")

    if any(value <= 0 for value in profile_seed_traces):
        parser.error("--profile-seed-traces values must all be positive integers")
    if args.text_overlay_terms < 0:
        parser.error("--text-overlay-terms must be non-negative")
    if args.text_overlay_terms > 0 and args.text_overlay_samples <= 0:
        parser.error("--text-overlay-samples must be positive when --text-overlay-terms is set")
    if args.text_overlay_working_memory_cap is not None and args.text_overlay_working_memory_cap < 0:
        parser.error("--text-overlay-working-memory-cap must be non-negative")
    if args.async_workers <= 0:
        parser.error("--async-workers must be positive")
    if args.async_cores_per_worker <= 0:
        parser.error("--async-cores-per-worker must be positive")
    if args.merge_every_samples <= 0:
        parser.error("--merge-every-samples must be positive")

    ablation_requested = (
        args.ablate_short_same_region
        or args.ablation_regions is not None
        or args.ablation_delays is not None
    )

    # Default output path
    if args.output is None:
        args.output = f"results/{args.dataset}_metrics.json"

    fast = not args.full_seed
    n_traces = args.seed_traces

    if args.compare:
        run_comparison(
            max_samples=args.samples,
            ticks_per_sample=args.ticks,
            threads=args.threads,
            output_dir="results/comparison",
            fast=fast,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            ablate_short_same_region=ablation_requested,
            ablation_regions=ablation_regions,
            ablation_delays=ablation_delays,
            rest_ticks=args.rest_ticks,
        )
    elif args.dataset == "fixed_overlay_triplet":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset fixed_overlay_triplet")
        if ablation_requested:
            parser.error(
                "fixed_overlay_triplet manages propagation/learning ablation internally; "
                "do not combine it with --ablate-short-same-region, --ablation-regions, or --ablation-delays"
            )
        run_fixed_overlay_triplet(
            base_dataset=args.fixed_graph_base_dataset,
            max_samples=args.samples,
            ticks_per_sample=args.ticks,
            threads=args.threads,
            output_path=args.output,
            compact=args.compact,
            fast=fast,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            rest_ticks=args.rest_ticks,
        )
    elif args.dataset == "async_multi_brain":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset async_multi_brain")
        if ablation_requested:
            parser.error(
                "async_multi_brain does not currently support runtime ablation flags; "
                "run it on the canonical topology first"
            )
        result = run_async_multi_brain_text(
            dataset=args.async_text_dataset,
            max_samples=args.samples,
            ticks_per_sample=args.ticks,
            output_path=args.output,
            worker_count=args.async_workers,
            cores_per_worker=args.async_cores_per_worker,
            merge_every_samples=args.merge_every_samples,
            rest_ticks=args.rest_ticks,
            fast=fast,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            validation_threads=(args.threads if args.threads > 0 else None),
            run_validations=not args.async_skip_validation,
        )
        training = result["training"]
        gates = result["gate_results"]
        print("\nASYNC MULTI-BRAIN SUMMARY")
        print(
            f"  training ticks/sec={float(training.get('ticks_per_sec', 0.0)):.4f}, "
            f"ms/tick={float(training.get('tick_time_avg_ms', 0.0)):.4f}, "
            f"merge_cost={100.0 * float(training.get('merge_cost_fraction', 0.0)):.2f}%"
        )
        if training.get("baseline_ticks_per_sec") is not None:
            print(
                f"  checkpoint2 baseline={float(training['baseline_ticks_per_sec']):.4f}, "
                f"scaling_efficiency={float(training.get('scaling_efficiency_per_instance', 0.0) or 0.0):.4f}"
            )
        print(
            f"  throughput>=3x={bool(gates.get('throughput_min_3x_checkpoint2', False))}, "
            f"throughput>=4x={bool(gates.get('throughput_target_4x_checkpoint2', False))}, "
            f"efficiency>=0.75={bool(gates.get('scaling_efficiency_per_instance_gte_0_75', False))}"
        )
        if args.async_skip_validation:
            print("  validation=skipped")
        else:
            print(
                f"  phase11_pass={bool(gates.get('phase11_fast_pass', False))}, "
                f"crossmodal_pass={bool(gates.get('crossmodal_3_of_3_pass', False))}"
            )
        print(f"Async multi-brain benchmark saved to: {args.output}")
    elif args.dataset == "text_learning_probe":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset text_learning_probe")
        if ablation_requested:
            parser.error(
                "text_learning_probe does not currently support runtime ablation flags; "
                "run the probe on the canonical topology first"
            )
        run_text_learning_probe(
            max_samples=min(args.samples, len(DEFAULT_TEXT_LEARNING_PROBE_SPECS)),
            ticks_per_sample=args.ticks,
            train_repeats=args.probe_train_repeats,
            threads=args.threads,
            output_path=args.output,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            rest_ticks=args.rest_ticks,
            settle_ticks=args.probe_settle_ticks,
            probe_ticks=args.probe_ticks,
        )
    elif args.dataset == "visual_learning_probe":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset visual_learning_probe")
        if ablation_requested:
            parser.error(
                "visual_learning_probe does not currently support runtime ablation flags; "
                "run the probe on the canonical topology first"
            )
        run_visual_learning_probe(
            max_samples=min(args.samples, len(DEFAULT_VISUAL_LEARNING_PROBE_LABELS)),
            ticks_per_sample=args.ticks,
            train_repeats=args.probe_train_repeats,
            threads=args.threads,
            output_path=args.output,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            rest_ticks=args.rest_ticks,
            settle_ticks=args.probe_settle_ticks,
            probe_ticks=args.probe_ticks,
            cue_fraction=args.cue_fraction,
        )
    elif args.dataset == "audio_learning_probe":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset audio_learning_probe")
        if ablation_requested:
            parser.error(
                "audio_learning_probe does not currently support runtime ablation flags; "
                "run the probe on the canonical topology first"
            )
        run_audio_learning_probe(
            max_samples=min(args.samples, len(DEFAULT_AUDIO_LEARNING_PROBE_LABELS)),
            ticks_per_sample=args.ticks,
            train_repeats=args.probe_train_repeats,
            threads=args.threads,
            output_path=args.output,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            rest_ticks=args.rest_ticks,
            settle_ticks=args.probe_settle_ticks,
            probe_ticks=args.probe_ticks,
            cue_fraction=args.cue_fraction,
        )
    elif args.dataset == "multimodal_binding_probe":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset multimodal_binding_probe")
        if ablation_requested:
            parser.error(
                "multimodal_binding_probe does not currently support runtime ablation flags; "
                "run the probe on the canonical topology first"
            )
        result = run_multimodal_binding_probe(
            ticks_per_sample=args.ticks,
            train_repeats=args.probe_train_repeats,
            threads=args.threads,
            output_path=args.output,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            rest_ticks=args.rest_ticks,
            settle_ticks=args.probe_settle_ticks,
            probe_ticks=args.probe_ticks,
            cue_fraction=args.cue_fraction,
            catalog=DEFAULT_MULTIMODAL_BINDING_PROBE_CATALOG,
        )
        aggregate = result["aggregate"]
        print("\nMULTIMODAL BINDING PROBE SUMMARY")
        print("  concept      formed   cross   recall   sparsity   pass")
        for row in aggregate.get("per_concept_results", []):
            concept_key = str(row.get("concept_key", "?"))
            print(
                f"  {concept_key:10s} "
                f"{int(row.get('bindings_formed_total', 0)):>6d} "
                f"{int(row.get('cross_modal_binding_count', 0)):>7d} "
                f"{int(row.get('successful_partner_recall_count', 0)):>8d} "
                f"{float(row.get('global_sparsity_max', 0.0)):>10.4f} "
                f"{bool(row.get('passed', False))}"
            )
        print(
            f"  overall: {aggregate.get('passed_concept_count', 0)}/"
            f"{aggregate.get('concept_count', 0)} passed, "
            f"required={aggregate.get('required_pass_count', 0)}, "
            f"pass_rate={float(aggregate.get('pass_rate', 0.0)):.2%}, "
            f"catalog_gate={bool(aggregate.get('passes_catalog_gate', False))}"
        )
        print(f"Multimodal binding probe saved to: {args.output}")
    elif args.dataset == "multimodal_stability_probe":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset multimodal_stability_probe")
        if ablation_requested:
            parser.error(
                "multimodal_stability_probe does not currently support runtime ablation flags; "
                "run the probe on the canonical topology first"
            )
        result = run_multimodal_stability_probe(
            max_samples=args.samples,
            ticks_per_sample=args.ticks,
            threads=args.threads,
            output_path=args.output,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            full_seed=args.full_seed,
            rest_ticks=args.rest_ticks,
        )
        summary = result["summary"]
        pruning = summary["pruning"]
        stability = summary["stability"]
        print("\nMULTIMODAL STABILITY PROBE SUMMARY")
        print(
            f"  samples={summary['sample_count']}, train_ticks={summary['train_tick_count']}, "
            f"rest_ticks={summary['rest_tick_count']}"
        )
        print(
            "  phases: "
            + ", ".join(
                f"{phase}={summary['phase_tick_counts'].get(phase, 0)}"
                for phase in ("bloom", "critical", "mature")
            )
        )
        print(
            f"  synapses: start={pruning['synapse_count_start']:,}, "
            f"end={pruning['synapse_count_end']:,}, "
            f"pruned={pruning['synapse_pruned_total']:,}"
        )
        print(
            f"  late sparsity max={stability['late_run_sparsity_max']:.4f}, "
            f"late total_active avg={stability['late_run_total_active_avg']:.1f}, "
            f"late bindings avg={stability['late_run_total_bindings_avg']:.1f}"
        )
        print(
            "  late modality activation avg: "
            f"visual={stability['late_run_modality_activation_avg']['visual']:.4f}, "
            f"audio={stability['late_run_modality_activation_avg']['audio']:.4f}, "
            f"language={stability['late_run_modality_activation_avg']['language']:.4f}"
        )
        if summary["warnings"]:
            print("  warnings:")
            for warning in summary["warnings"]:
                print(f"    - {warning}")
        print(f"Multimodal stability probe saved to: {args.output}")
    elif args.dataset == "output_region_probe":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset output_region_probe")
        if ablation_requested:
            parser.error(
                "output_region_probe does not currently support runtime ablation flags; "
                "run the probe on the canonical topology first"
            )
        result = run_output_region_probe(
            max_samples=args.samples,
            ticks_per_sample=args.ticks,
            threads=args.threads,
            output_path=args.output,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            full_seed=args.full_seed,
        )
        summary = result["summary"]
        speech = summary["speech"]
        motor = summary["motor"]
        print("\nOUTPUT REGION PROBE SUMMARY")
        print(
            f"  speech correlation={speech['speech_coverage_correlation']:.4f}, "
            f"high_mean={speech['high_speech_activity_mean']:.6f}, "
            f"low_mean={speech['low_speech_activity_mean']:.6f}, "
            f"alive={speech['speech_alive']}"
        )
        print("  motor discriminability:")
        for label_name, row in motor["per_class"].items():
            print(
                f"    {label_name}: between={row['between_category_variance']:.6f}, "
                f"within={row['within_category_variance']:.6f}, "
                f"activation_mean={row['motor_activation_mean']:.6f}, "
                f"pass={row['discriminable']}"
            )
        print(
            f"  motor pass count={motor['passing_class_count']}/"
            f"{len(motor['per_class'])}, alive={motor['motor_alive']}"
        )
        print(f"Output region probe saved to: {args.output}")
    elif args.dataset == "phase11_operational_baseline":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset phase11_operational_baseline")
        if ablation_requested:
            parser.error(
                "phase11_operational_baseline does not currently support runtime ablation flags; "
                "run the canonical topology first"
            )
        result = run_phase11_operational_baseline(
            threads=args.threads,
            output_path=args.output,
            stability_samples=args.samples,
            output_probe_samples=min(args.samples, 32),
            ticks_per_sample=args.ticks,
            rest_ticks=args.rest_ticks,
            seed_chunks=args.seed_chunks,
            fast_mode=not args.full_seed,
            n_traces=args.seed_traces,
        )
        references = result["reference_numbers"]
        performance = result.get("performance", {})
        print("\nPHASE 11 OPERATIONAL BASELINE")
        print(
            f"  mode={result['seed_mode']}, "
            f"stability pass={result['validations']['multimodal_stability_passes']}, "
            f"output pass={result['validations']['output_probe_passes']}"
        )
        print(
            f"  multimodal samples={result['config']['stability_samples']}, "
            f"multimodal wall={float(performance.get('multimodal_total_wall_ms', 0.0)):.1f}ms, "
            f"output wall={float(performance.get('output_probe_total_wall_ms', 0.0)):.1f}ms"
        )
        print(
            f"  speech correlation={references['speech_coverage_correlation']:.4f}, "
            f"high_mean={references['high_speech_activity_mean']:.6f}, "
            f"low_mean={references['low_speech_activity_mean']:.6f}"
        )
        print("  motor discriminability:")
        for label_name, row in references["motor_per_class"].items():
            print(
                f"    {label_name}: between={row['between_category_variance']:.6f}, "
                f"within={row['within_category_variance']:.6f}, "
                f"activation_mean={row['motor_activation_mean']:.6f}, "
                f"pass={row['discriminable']}"
            )
        print(f"Phase 11 operational baseline saved to: {args.output}")
    elif args.dataset == "executive_numbers_probe":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset executive_numbers_probe")
        if ablation_requested:
            parser.error(
                "executive_numbers_probe does not currently support runtime ablation flags; "
                "run the canonical topology first"
            )
        executive_probe_ticks = (
            args.ticks
            if _cli_option_provided("--ticks")
            else _DEFAULT_EXECUTIVE_NUMBERS_PROBE_TICKS
        )
        result = run_executive_numbers_probe(
            ticks_per_sample=executive_probe_ticks,
            threads=args.threads,
            output_path=args.output,
            seed_chunks=args.seed_chunks,
        )
        numbers = result["summary"]["numbers"]
        executive = result["summary"]["executive"]
        print("\nEXECUTIVE + NUMBERS PROBE SUMMARY")
        print(
            f"  numbers digit active={numbers['digit']['numbers_active_count_max']:.4f}, "
            f"text-only active={numbers['text_only']['numbers_active_count_max']:.4f}, "
            f"digit language delta={numbers['digit_vs_text_delta']['language_activation_max']:.6f}, "
            f"digit pattern delta={numbers['digit_vs_text_delta']['pattern_active_count_max']:.6f}"
        )
        print(
            f"  executive single={executive['single_concept']['executive_engagement_max']:.6f}, "
            f"multi={executive['multi_concept']['executive_engagement_max']:.6f}, "
            f"delta={executive['delta']['executive_engagement_max']:.6f}"
        )
        print(f"Executive + numbers probe saved to: {args.output}")
    elif args.dataset == "crossmodal_recall_probe":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset crossmodal_recall_probe")
        if ablation_requested:
            parser.error(
                "crossmodal_recall_probe does not currently support runtime ablation flags; "
                "run the probe on the canonical topology first"
            )
        result = run_crossmodal_recall_probe(
            ticks_per_sample=args.ticks,
            threads=args.threads,
            output_path=args.output,
            train_samples=6,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            rest_ticks=args.rest_ticks,
            settle_ticks=args.probe_settle_ticks,
            probe_ticks=args.probe_ticks,
            cue_fraction=args.cue_fraction,
            spec=DEFAULT_CROSSMODAL_RECALL_PROBE_SPEC,
        )
        summary = result["summary"]
        print("\nCROSSMODAL RECALL PROBE SUMMARY")
        print("  direction        source   target   hit_rate   pass")
        for direction_key, row in summary.get("cue_direction_results", {}).items():
            print(
                f"  {direction_key:16s} "
                f"{str(row.get('source_modality', '?')):7s} "
                f"{str(row.get('target_modality', '?')):7s} "
                f"{float(row.get('partner_trace_hit_rate', 0.0)):>8.4f} "
                f"{bool(row.get('passed', False))}"
            )
        print(
            f"  training samples: {int(summary.get('train_samples_completed', 0))}/"
            f"{int(summary.get('train_samples_requested', 0))}, "
            f"bindings={int(summary.get('cross_modal_binding_count', 0))}, "
            f"training_sparsity_max={float(summary.get('training_sparsity_max', 0.0)):.4f}, "
            f"pass={bool(dict(summary.get('validations', {})).get('passes_probe', False))}"
        )
        print(f"Crossmodal recall probe saved to: {args.output}")
    elif args.dataset == "text_binding_probe":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset text_binding_probe")
        if ablation_requested:
            parser.error(
                "text_binding_probe does not currently support runtime ablation flags; "
                "run the probe on the canonical topology first"
            )
        run_text_binding_probe(
            max_samples=min(args.samples, len(DEFAULT_TEXT_LEARNING_PROBE_SPECS)),
            ticks_per_sample=args.ticks,
            train_repeats=args.probe_train_repeats,
            threads=args.threads,
            output_path=args.output,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            rest_ticks=args.rest_ticks,
            settle_ticks=args.probe_settle_ticks,
            probe_ticks=args.probe_ticks,
            cue_fraction=args.cue_fraction,
            cue_noise_fraction=args.cue_noise_fraction,
            cue_mode=args.cue_mode,
            overlay_terms=args.text_overlay_terms,
            overlay_samples=args.text_overlay_samples,
            overlay_working_memory_cap=args.text_overlay_working_memory_cap,
        )
    elif args.dataset == "text_vocab_profile":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset text_vocab_profile")
        if ablation_requested:
            parser.error(
                "text_vocab_profile measures canonical text coverage; "
                "do not combine it with runtime ablation flags"
            )
        result = run_text_vocab_profile(
            dataset=args.text_profile_dataset,
            max_samples=args.samples,
            output_path=args.output,
            seed_trace_counts=profile_seed_traces,
            seed_chunks=args.seed_chunks,
            overlay_terms=args.text_overlay_terms,
            overlay_samples=args.text_overlay_samples,
        )
        summary = result["summary"]
        profile = result["baseline_profile"]
        overlay_profile = result.get("overlay_profile")
        overlay_selection = result.get("overlay_selection")
        print("\nTEXT VOCAB PROFILE SUMMARY")
        print(
            f"  baseline coverage={profile['baseline']['coverage']:.2%}, "
            f"unknown={summary['baseline_unknown_rate']:.2%}, "
            f"content coverage={profile['baseline']['content_coverage']:.2%}"
        )
        print(
            f"  normalized coverage={profile['normalized']['coverage']:.2%}, "
            f"unknown={summary['normalized_unknown_rate']:.2%}, "
            f"recovered={profile['normalized']['recovered_tokens']}, "
            f"content coverage={profile['normalized']['content_coverage']:.2%}"
        )
        print(
            f"  lexical-neighborhood coverage={profile['lexical_neighborhood']['coverage']:.2%}, "
            f"unknown={summary['lexical_neighborhood_unknown_rate']:.2%}, "
            f"recovered={profile['lexical_neighborhood']['recovered_tokens']}, "
            f"content coverage={profile['lexical_neighborhood']['content_coverage']:.2%}"
        )
        print(
            f"  seed-size moves baseline coverage: {summary['seed_trace_count_changes_baseline_coverage']}"
        )
        if overlay_profile is not None and overlay_selection is not None:
            print(
                f"  overlay coverage={overlay_profile['baseline']['coverage']:.2%}, "
                f"content coverage={overlay_profile['baseline']['content_coverage']:.2%}, "
                f"selected content share={overlay_selection['selected_term_content_coverage']:.2%}, "
                f"overlay traces={overlay_selection['added_trace_count']}"
            )
        print(f"Text vocab profile saved to: {args.output}")
    elif args.dataset == "coding_assistant_probe":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset coding_assistant_probe")
        if ablation_requested:
            parser.error(
                "coding_assistant_probe does not currently support runtime ablation flags; "
                "run the probe on the canonical topology first"
            )
        result = run_coding_assistant_probe(
            n_samples=args.samples,
            chunk_size=args.chunk_size,
            ticks_per_chunk=args.ticks,
            rest_ticks_between_chunks=args.chunk_rest_ticks,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            output_path=args.output,
        )
        gates = result["gates"]
        training = result["training"]
        print("\nCODING ASSISTANT PROBE SUMMARY")
        print(
            f"  samples={training['n_samples']}, chunks={training['total_chunks']}, "
            f"ticks={training['total_ticks']}, tps={training['ticks_per_sec']:.0f}"
        )
        print(
            f"  traces={training['traces_formed']}, bindings={training['bindings_formed']}, "
            f"schemas={training['schemas_formed']}"
        )
        print(f"  Gates: {gates}")
        print(f"Coding assistant probe saved to: {args.output}")
    elif args.dataset == "end_to_end_demo":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset end_to_end_demo")
        if ablation_requested:
            parser.error(
                "end_to_end_demo does not currently support runtime ablation flags; "
                "run the demo on the canonical topology first"
            )
        result = run_end_to_end_demo(
            ticks_per_chunk=args.ticks,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            output_path=args.output,
        )
        gates = result["gates"]
        print("\nEND-TO-END DEMO SUMMARY")
        for k, v in gates.items():
            status = "PASS" if v else "FAIL"
            print(f"  {status}: {k}")
        print(f"End-to-end demo saved to: {args.output}")

    elif args.dataset == "opus_reasoning":
        if args.save_brain is not None:
            parser.error("--save-brain is not supported with --dataset opus_reasoning")
        if ablation_requested:
            parser.error(
                "opus_reasoning does not currently support runtime ablation flags"
            )
        result = run_coding_assistant_probe(
            n_samples=args.samples,
            ticks_per_chunk=args.ticks,
            chunk_size=args.chunk_size,
            rest_ticks_between_chunks=args.chunk_rest_ticks,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            output_path=args.output,
            hf_dataset="opus_reasoning",
        )
        gates = result["gates"]
        print("\nOPUS REASONING PROBE SUMMARY")
        print(f"  samples={result['training']['n_samples']}, "
              f"ticks={result['training']['total_ticks']}, "
              f"tps={result['training']['ticks_per_sec']:.0f}")
        print(f"  traces={result['training']['traces_formed']}, "
              f"bindings={result['training']['bindings_formed']}, "
              f"schemas={result['training']['schemas_formed']}")
        print(f"  Gates: {gates}")
        print(f"Opus reasoning probe saved to: {args.output}")
    else:
        run_session(
            dataset=args.dataset,
            max_samples=args.samples,
            ticks_per_sample=args.ticks,
            threads=args.threads,
            mode=args.mode,
            output_path=args.output,
            save_dir=args.save_brain,
            compact=args.compact,
            fast=fast,
            n_traces=n_traces,
            seed_chunks=args.seed_chunks,
            cue_fraction=args.cue_fraction,
            cue_noise_fraction=args.cue_noise_fraction,
            cue_mode=args.cue_mode,
            collision_selection=args.collision_selection,
            weak_binding_enabled=not args.disable_weak_binding,
            ablate_short_same_region=ablation_requested,
            ablation_regions=ablation_regions,
            ablation_delays=ablation_delays,
            rest_ticks=args.rest_ticks,
        )


if __name__ == "__main__":
    main()
