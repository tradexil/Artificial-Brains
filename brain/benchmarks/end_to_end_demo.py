"""End-to-end demo: chunking → training → schema formation → speech decode → world model.

Combines all new components into a single pipeline demonstration:
1. Seed brain
2. Train on coding samples via chunking pipeline
3. Monitor schema formation during training
4. Probe with cues → speech decoder
5. Run world model simulation
6. Report full pipeline metrics
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import brain_core

from brain.input.text_chunker import chunk_text, process_chunked_document
from brain.input.text_input import TextInput
from brain.learning.schema_formation import SchemaFormationEngine
from brain.learning.tick_loop import TickLoop
from brain.output.speech_decoder import SpeechDecoder
from brain.seed.seed_runner import seed_brain_fast
from brain.structures.schema import SchemaStore
from brain.structures.trace_store import Trace, TraceStore
from brain.structures.world_model import WorldModel
from brain.utils.config import REGIONS


# ---------------------------------------------------------------------------
# Small sample set for the demo
# ---------------------------------------------------------------------------

_DEMO_SAMPLES = [
    "Binary search works by repeatedly dividing the sorted array in half. "
    "Compare the target value to the middle element. "
    "If the target is smaller go left, if larger go right.",

    "A stack uses last in first out ordering. Push adds an element to the top. "
    "Pop removes the top element. Check if empty before popping.",

    "Hash tables provide constant time lookup on average. "
    "Each key maps to a bucket via a hash function. "
    "Handle collisions with chaining or open addressing.",

    "Merge sort divides the array into halves recursively. "
    "Then merge the sorted halves back together. "
    "The time complexity is O(n log n) in all cases.",

    "Linked lists have nodes pointing to the next node. "
    "Insert at head runs in constant time. "
    "Traversal requires visiting each node sequentially.",

    "Graph traversal with depth first search uses a stack. "
    "Mark each node visited to avoid cycles. "
    "Breadth first search uses a queue instead.",

    "Dynamic programming solves overlapping subproblems. "
    "Store results in a table to avoid recomputation. "
    "Build solutions bottom up from base cases.",

    "Recursion breaks a problem into smaller instances of itself. "
    "Every recursive function needs a base case. "
    "Without a base case the function calls itself forever.",

    "Binary trees have at most two children per node. "
    "Inorder traversal visits left subtree then root then right. "
    "Height balanced trees keep operations logarithmic.",

    "Greedy algorithms make locally optimal choices at each step. "
    "Activity selection picks the earliest finishing task. "
    "Not all problems have optimal greedy solutions.",
]

_PROBE_CUES = [
    "binary search sorted array target",
    "stack push pop element",
    "hash table key value lookup",
    "merge sort divide array",
    "linked list node traverse",
]


def run_end_to_end_demo(
    *,
    chunk_size: int | None = None,
    ticks_per_chunk: int = 5,
    rest_ticks_between: int = 2,
    n_traces: int = 5000,
    seed_chunks: int | None = None,
    output_path: str = "results/end_to_end_demo.json",
    verbose: bool = True,
) -> dict:
    """Run the full end-to-end demonstration pipeline.

    Returns dict with per-phase metrics and gate checks.
    """
    t0 = time.perf_counter()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    phases: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # Phase 1: Seed
    # -----------------------------------------------------------------------
    if verbose:
        print("=== Phase 1: Seed brain ===")
    _, trace_store = seed_brain_fast(n_traces=n_traces, verbose=False, chunk_count=seed_chunks)

    # Seed demo vocabulary as labeled traces with speech neurons
    rng = random.Random(42)
    demo_vocab: set[str] = set()
    for sample in _DEMO_SAMPLES:
        for w in sample.lower().split():
            w = w.strip(".,!?;:'\"()[]{}").lower()
            if len(w) >= 3:
                demo_vocab.add(w)
    for cue in _PROBE_CUES:
        for w in cue.split():
            if len(w) >= 3:
                demo_vocab.add(w.lower())

    vocab_created = 0
    for i, word in enumerate(sorted(demo_vocab)):
        neurons: dict[str, list[int]] = {}
        l_start = REGIONS["language"][0]
        neurons["language"] = rng.sample(range(l_start, l_start + 9000), 3)
        p_start, p_end = REGIONS["pattern"]
        neurons["pattern"] = rng.sample(range(p_start, p_end + 1), 3)
        sp_start = REGIONS["speech"][0]
        neurons["speech"] = rng.sample(range(sp_start, sp_start + 8000), 3)
        int_start, int_end = REGIONS["integration"]
        neurons["integration"] = rng.sample(range(int_start, int_end + 1), 2)
        ml_start, ml_end = REGIONS["memory_long"]
        neurons["memory_long"] = rng.sample(range(ml_start, ml_end + 1), 2)
        trace = Trace(
            id=f"demo_vocab_{i:04d}",
            neurons=neurons,
            strength=rng.uniform(0.3, 0.5),
            decay=1.0, polarity=0.0,
            abstraction=rng.uniform(0.2, 0.5),
            novelty=0.5, formation_tick=0, label=word,
        )
        trace_store.add(trace)
        vocab_created += 1

    tick_loop = TickLoop(trace_store)
    tick_loop.collect_full_metrics = False  # speed
    tick_loop.step()  # warmup

    speech_decoder = SpeechDecoder(trace_store)
    schema_store = SchemaStore()
    schema_engine = SchemaFormationEngine(schema_store)
    world_model = WorldModel()
    text_input = TextInput(trace_store)

    phases["seed"] = {
        "neurons": brain_core.get_neuron_count(),
        "synapses": brain_core.get_synapse_count(),
        "traces": len(trace_store),
    }
    if verbose:
        print(f"  Neurons: {phases['seed']['neurons']}, "
              f"Synapses: {phases['seed']['synapses']:,}, "
              f"Traces: {phases['seed']['traces']}")

    # -----------------------------------------------------------------------
    # Phase 2: Train via chunking
    # -----------------------------------------------------------------------
    if verbose:
        print("\n=== Phase 2: Chunked training ===")

    train_start = time.perf_counter()
    total_ticks = 0
    total_traces_formed = 0
    total_bindings_formed = 0
    total_chunks = 0

    for si, sample_text in enumerate(_DEMO_SAMPLES):
        tick_loop.reset_sample_boundary()
        brain_core.reset_runtime_state()  # prevent neuron accumulation
        chunks = chunk_text(sample_text, chunk_size=chunk_size)  # None = adaptive

        for ci, chunk in enumerate(chunks):
            text_input.encode(chunk)

            for t in range(ticks_per_chunk):
                result = tick_loop.step(learn=True)
                total_traces_formed += result.get("traces_formed", 0)
                total_bindings_formed += result.get("bindings_formed", 0)
                total_ticks += 1

                # Schema engine on last tick of each chunk
                if t == ticks_per_chunk - 1:
                    active_traces = tick_loop.last_active_traces
                    schema_engine.step(active_traces, tick_loop.last_tick_number)
                    world_model.predict_from_schemas(active_traces, tick_loop.last_tick_number, schema_store)
                    world_model.check_divergence(active_traces, tick_loop.last_tick_number)

            # Rest between chunks
            for _ in range(rest_ticks_between):
                tick_loop.step(learn=False)
                total_ticks += 1
            total_chunks += 1

    train_elapsed = time.perf_counter() - train_start
    train_tps = total_ticks / max(train_elapsed, 0.001)

    phases["training"] = {
        "samples": len(_DEMO_SAMPLES),
        "total_chunks": total_chunks,
        "total_ticks": total_ticks,
        "elapsed_sec": round(train_elapsed, 2),
        "ticks_per_sec": round(train_tps, 1),
        "traces_formed": total_traces_formed,
        "bindings_formed": total_bindings_formed,
        "schemas_formed": len(schema_store),
    }
    if verbose:
        print(f"  {total_chunks} chunks, {total_ticks} ticks in {train_elapsed:.1f}s "
              f"({train_tps:.0f} tps)")
        print(f"  Traces formed: {total_traces_formed}, Bindings: {total_bindings_formed}, "
              f"Schemas: {len(schema_store)}")

    # -----------------------------------------------------------------------
    # Phase 3: Speech decoder probes
    # -----------------------------------------------------------------------
    if verbose:
        print("\n=== Phase 3: Speech decode probes ===")

    speech_decoder.refresh_index()
    cue_results: list[dict] = []

    for cue in _PROBE_CUES:
        tick_loop.reset_sample_boundary()
        brain_core.reset_runtime_state()  # clean slate for each probe
        text_input.encode(cue)
        speech_decoder.reset_window()

        for _ in range(ticks_per_chunk):
            tick_loop.step(learn=False)
            speech_decoder.accumulate_tick()

        window = speech_decoder.decode_window()
        output_words = set()
        for label, _ in window.get("tokens", []):
            for w in label.lower().split():
                output_words.add(w)

        cue_results.append({
            "cue": cue,
            "speech_text": window.get("text", ""),
            "speech_tokens": window.get("tokens", []),
            "output_words": sorted(output_words),
        })

        if verbose:
            print(f"  Cue: {cue!r}")
            print(f"    → {window.get('text', '')!r}")

    phases["speech_probes"] = {
        "n_probes": len(cue_results),
        "any_output": any(cr["speech_text"] for cr in cue_results),
        "details": cue_results,
    }

    # -----------------------------------------------------------------------
    # Phase 4: Schema summary
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n=== Phase 4: Schema summary ===")

    schema_details: list[dict] = []
    for schema in list(schema_store)[:10]:
        schema_details.append({
            "id": schema.id,
            "label": schema.label,
            "trace_count": len(schema.traces),
            "edge_count": len(schema.causal_edges),
            "strength": schema.strength,
        })

    phases["schemas"] = {
        "total": len(schema_store),
        "sample": schema_details,
    }
    if verbose:
        print(f"  Total schemas: {len(schema_store)}")
        for sd in schema_details[:5]:
            print(f"    {sd['id']}: {sd['trace_count']} traces, "
                  f"{sd['edge_count']} edges, str={sd['strength']:.2f}")

    # -----------------------------------------------------------------------
    # Phase 5: World model simulation
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n=== Phase 5: World model simulation ===")

    wm_summary = world_model.get_summary()
    simulation_result: dict | None = None

    if len(schema_store) > 0:
        # Pick first schema and simulate
        first_schema_id = list(schema_store.schemas.keys())[0]
        simulation_result = world_model.simulate_schema_chain(
            schema_store, first_schema_id, tick_loop, max_ticks=10,
        )
        if verbose:
            print(f"  Simulated schema {first_schema_id}:")
            print(f"    Ticks: {simulation_result['ticks_run']}, "
                  f"Coverage: {simulation_result['chain_coverage']:.2f}, "
                  f"Confidence: {simulation_result['confidence']:.2f}")
    else:
        if verbose:
            print("  No schemas formed — skipping simulation")

    phases["world_model"] = {
        "summary": wm_summary,
        "simulation": simulation_result,
        "divergence_entries": len(world_model.divergence_log),
    }

    # -----------------------------------------------------------------------
    # Gate checks
    # -----------------------------------------------------------------------
    gates = {
        "training_tps_above_150": train_tps > 150,
        "any_speech_output": phases["speech_probes"]["any_output"],
        "schemas_formed": len(schema_store) > 0,
        "world_model_ran": simulation_result is not None,
        "divergence_log_populated": len(world_model.divergence_log) > 0,
        "traces_formed": total_traces_formed > 0,
        "bindings_formed": total_bindings_formed > 0,
    }

    result = {
        "phases": phases,
        "gates": gates,
        "elapsed_total_sec": round(time.perf_counter() - t0, 2),
    }

    output_file.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if verbose:
        print(f"\n=== Gates ===")
        for k, v in gates.items():
            status = "PASS" if v else "FAIL"
            print(f"  {status}: {k}")
        print(f"\nResults saved to {output_path}")

    return result
