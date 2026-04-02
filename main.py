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
import sys
import time
from pathlib import Path

import numpy as np

import brain_core

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_threads(n: int) -> int:
    """Set rayon thread count. Returns actual thread count."""
    if n > 0:
        try:
            brain_core.set_num_threads(n)
        except Exception:
            pass  # already set (rayon build_global can only be called once)
    return brain_core.get_num_threads()


def _seed_and_build(verbose: bool = True, fast: bool = False, n_traces: int = 5000):
    """Seed the brain. Returns (brain_core, TraceStore)."""
    if verbose:
        mode_str = f"fast ({n_traces} traces)" if fast else "full (100k traces)"
        print(f"Seeding brain ({mode_str})...")
        t0 = time.perf_counter()
    if fast:
        _, ts = seed_brain_fast(n_traces=n_traces, verbose=verbose)
    else:
        _, ts = seed_brain(verbose=verbose)
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


# ── Run functions per dataset type ────────────────────────────────────────────

def run_text(
    ts,
    tick_loop: TickLoop,
    collector: MetricsCollector,
    samples: list[dict],
    ticks_per_sample: int,
    speech_decoder: SpeechOutput | None = None,
    rest_ticks: int = 3,
):
    """Run text learning: encode text → tick N times → collect metrics."""
    text_input = TextInput(ts)

    for i, sample in enumerate(samples):
        text = sample["text"]
        label = sample.get("label_name", str(sample.get("label", "")))
        sm = collector.begin_sample(i, label=label, modality="text")

        # Encode text into language region
        enc = text_input.encode(text)

        # Run ticks
        for t in range(ticks_per_sample):
            # Re-inject on first tick only (signals propagate on their own)
            if t == 0:
                text_input.encode(text)

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
                speech = speech_decoder.decode(top_k=5)
                result["speech_output"] = speech.get("text", "")
                result["speech_tokens"] = [tok for tok, _ in speech.get("tokens", [])]

            sm.add_tick(result)

        collector.end_sample()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(samples)}] label={label}, "
                  f"active={result.get('total_active', 0)}, "
                  f"hebb={result.get('hebbian_updates', 0)}, "
                  f"traces_new={result.get('traces_formed', 0)}")

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
    rest_ticks: int = 3,
) -> MetricsCollector:
    """Run a full learning session."""

    actual_threads = _set_threads(threads)
    print(f"Threads: {actual_threads}")

    # Seed the brain
    ts = _seed_and_build(verbose=True, fast=fast, n_traces=n_traces)
    tick_loop = TickLoop(ts)
    speech_decoder = SpeechOutput(ts)

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
        },
    )
    collector.start()

    # Warmup tick — first tick is slow (rayon thread pool init, CSR cache load)
    print("Warming up (1 tick)...")
    t0 = time.perf_counter()
    tick_loop.step()
    warmup_ms = (time.perf_counter() - t0) * 1000
    print(f"  Warmup: {warmup_ms:.0f}ms")

    if dataset == "all":
        _run_all_datasets(ts, tick_loop, collector, max_samples, ticks_per_sample,
                          speech_decoder, mode, fast=fast, n_traces=n_traces,
                          rest_ticks=rest_ticks)
    elif dataset == "ag_news" or dataset == "imdb":
        print(f"\nLoading {dataset}...")
        from brain.datasets.downloader import load_text_dataset
        data = load_text_dataset(dataset, max_samples=max_samples)
        print(f"  Got {len(data)} samples")
        print(f"\nRunning text learning ({len(data)} samples, {ticks_per_sample} ticks each)...")
        run_text(ts, tick_loop, collector, data, ticks_per_sample, speech_decoder,
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
            ts_new = _seed_and_build(verbose=False, fast=fast, n_traces=n_traces)
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
            runner(ts_new, tick_loop_new, collector, data, ticks_per_sample, speech_new,
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
  python main.py --dataset all --samples 30 --mode sequential
  python main.py --dataset all --samples 30 --mode separate
  python main.py --compare --samples 30 --ticks 10
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=["ag_news", "imdb", "cifar10", "speech_commands", "multimodal", "all"],
        default="ag_news",
        help="Dataset to learn from (default: ag_news)",
    )
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
    parser.add_argument("--rest-ticks", type=int, default=3,
                        help="Idle ticks between samples to let activity decay (default: 3)")

    args = parser.parse_args()

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
            rest_ticks=args.rest_ticks,
        )
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
            rest_ticks=args.rest_ticks,
        )


if __name__ == "__main__":
    main()
