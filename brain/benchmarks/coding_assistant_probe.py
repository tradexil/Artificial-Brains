"""Learning probe: train on text samples, verify speech output quality.

Works with ANY text dataset — coding Q&A, conversations, articles.
Generates synthetic coding samples as a fallback, or loads from HuggingFace.

Key design principles:
  - Full runtime reset between samples (including novelty EMA) so every
    sample feels "new" and can trigger trace formation.
  - Enough ticks per chunk (10) for signal to propagate the full
    language → pattern → integration → speech chain.
  - Vocabulary seeded as labeled traces so speech decode can map back
    to human-readable words.
  - Probing with enough accumulation ticks for meaningful speech output.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import brain_core

from brain.input.text_chunker import chunk_text
from brain.input.text_input import TextInput, is_content_text_token
from brain.learning.schema_formation import SchemaFormationEngine
from brain.learning.tick_loop import TickLoop
from brain.output.speech_decoder import SpeechDecoder
from brain.seed.seed_runner import seed_brain_fast
from brain.structures.schema import SchemaStore
from brain.structures.trace_store import Trace, TraceStore
from brain.utils.config import REGIONS


# ---------------------------------------------------------------------------
# Synthetic coding-assistant sample generator
# ---------------------------------------------------------------------------

_CODING_TOPICS = [
    ("binary search", "sorted array", "algorithm", "function", "index", "target", "return", "left right"),
    ("linked list", "node pointer", "data structure", "traverse", "insert", "delete", "next"),
    ("dynamic programming", "subproblem", "optimal", "memoize", "table", "bottom up", "recurrence"),
    ("graph traversal", "depth first", "breadth first", "visited", "queue stack", "adjacency"),
    ("sorting algorithm", "comparison", "merge sort", "quick sort", "partition", "divide conquer"),
    ("hash table", "collision", "load factor", "bucket", "key value", "constant time"),
    ("tree structure", "binary tree", "leaf node", "height balance", "inorder preorder"),
    ("string manipulation", "substring", "palindrome", "character", "reverse", "pattern match"),
    ("recursion", "base case", "recursive call", "stack overflow", "factorial", "fibonacci"),
    ("stack queue", "push pop", "first last", "priority", "deque", "circular buffer"),
    ("array matrix", "two dimensional", "row column", "diagonal", "rotate", "transpose"),
    ("bit manipulation", "bitwise and or", "shift left right", "mask", "power two"),
    ("greedy algorithm", "local optimal", "activity selection", "interval", "schedule"),
    ("backtracking", "constraint", "permutation", "combination", "pruning", "solution space"),
    ("database query", "select where", "join table", "index optimize", "normalize"),
    ("api design", "rest endpoint", "request response", "authentication", "rate limit"),
    ("testing strategy", "unit test", "mock stub", "coverage", "edge case", "assertion"),
    ("error handling", "exception try catch", "graceful failure", "retry", "logging"),
    ("file processing", "read write", "parse format", "csv json", "stream buffer"),
    ("concurrency", "thread lock", "race condition", "mutex", "parallel", "async await"),
    ("memory management", "allocation", "garbage collection", "reference count", "leak"),
]

_RESPONSE_TEMPLATES = [
    "To solve this problem you need to {verb} the {noun}. First define the function signature with input parameters. Then implement the core logic using {technique}. Handle edge cases like empty input and return the result.",
    "The approach uses {technique} to process the {noun}. Start by initializing variables. Iterate through the input applying {verb} at each step. The time complexity is efficient for the given constraints.",
    "Write a function that takes the {noun} as input. Use {technique} as the main strategy. {verb} the elements according to the problem requirements. Test with sample input output pairs to verify correctness.",
    "Consider the {noun} structure carefully. Apply {technique} to optimize the solution. The key insight is to {verb} in the right order. Return the final answer after processing all elements.",
    "Break the problem into smaller parts. For the {noun} component use {technique}. Remember to {verb} before moving to the next step. Edge cases include null input and boundary values.",
]


def _generate_coding_samples(count: int = 2100) -> list[dict]:
    """Generate synthetic coding assistant samples.

    Each sample has system/user/assistant roles with coding-related content.
    """
    samples = []
    for i in range(count):
        topic_idx = i % len(_CODING_TOPICS)
        template_idx = i % len(_RESPONSE_TEMPLATES)
        topic_words = _CODING_TOPICS[topic_idx]
        topic_name = topic_words[0]

        # Vary content per sample
        word_offset = (i // len(_CODING_TOPICS)) % max(1, len(topic_words) - 2)
        noun = topic_words[min(1 + word_offset, len(topic_words) - 1)]
        verb = topic_words[min(2 + word_offset, len(topic_words) - 1)]
        technique = topic_words[min(3 + word_offset, len(topic_words) - 1)]

        system_text = f"You are a coding assistant helping with {topic_name} problems."
        user_text = (
            f"Write a function to {verb} in a {noun}. "
            f"The input is a {noun} and the output should be the result. "
            f"Explain the approach using {technique}."
        )
        assistant_text = _RESPONSE_TEMPLATES[template_idx].format(
            noun=noun, verb=verb, technique=technique,
        )

        # Full sample text combines all three roles
        full_text = f"{system_text} {user_text} {assistant_text}"

        samples.append({
            "index": i,
            "topic": topic_name,
            "system": system_text,
            "user": user_text,
            "assistant": assistant_text,
            "text": full_text,
            "key_concepts": list(topic_words[:4]),
        })

    return samples


# ---------------------------------------------------------------------------
# Cue probes
# ---------------------------------------------------------------------------

def _seed_vocabulary_traces(
    trace_store: TraceStore,
    words: set[str],
    rng: random.Random,
    prefix: str = "vocab",
) -> int:
    """Create labeled traces for a set of words so speech decode works.

    Each word gets a trace with neurons in language, pattern, speech,
    integration, and memory_long regions, plus a label matching the word.
    Skips words that already have a labeled trace in the store.

    Returns number of new vocabulary traces created.
    """
    # Build existing label set to avoid duplicates
    existing_labels = {
        t.label.lower()
        for t in trace_store.traces.values()
        if t.label is not None
    }

    created = 0
    for i, word in enumerate(sorted(words)):
        if word in existing_labels:
            continue
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
            id=f"{prefix}_{created:05d}",
            neurons=neurons,
            strength=rng.uniform(0.3, 0.5),
            decay=1.0,
            polarity=0.0,
            abstraction=rng.uniform(0.2, 0.5),
            novelty=0.5,
            formation_tick=0,
            label=word,
        )
        trace_store.add(trace)
        created += 1

    return created


def _extract_vocabulary(texts: list[str]) -> set[str]:
    """Extract content words from a list of text samples."""
    vocab: set[str] = set()
    for text in texts:
        for w in text.lower().split():
            w = w.strip(".,!?;:'\"()[]{}").lower()
            if is_content_text_token(w):
                vocab.add(w)
    return vocab


_CUE_PROBES = [
    {
        "cue": "problem statement input output",
        "expected_concepts": ["function", "input", "output", "return", "result"],
        "description": "Problem framing cue",
    },
    {
        "cue": "write a function algorithm",
        "expected_concepts": ["function", "algorithm", "implement", "define", "return"],
        "description": "Function writing cue",
    },
    {
        "cue": "input output format example",
        "expected_concepts": ["input", "output", "test", "sample", "format", "result"],
        "description": "I/O format cue",
    },
    {
        "cue": "sort array compare elements",
        "expected_concepts": ["sort", "array", "compare", "merge", "partition"],
        "description": "Sorting domain cue",
    },
    {
        "cue": "binary search target index",
        "expected_concepts": ["binary", "search", "sorted", "index", "target"],
        "description": "Binary search cue",
    },
]


# ---------------------------------------------------------------------------
# Main probe runner
# ---------------------------------------------------------------------------

def run_coding_assistant_probe(
    *,
    n_samples: int = 2100,
    chunk_size: int | None = None,
    ticks_per_chunk: int = 10,
    rest_ticks_between_chunks: int = 3,
    n_traces: int = 5000,
    seed_chunks: int | None = None,
    output_path: str = "results/coding_assistant_probe.json",
    verbose: bool = True,
    hf_dataset: str | None = None,
) -> dict:
    """Train on text samples, probe speech output for learned associations.

    Works with any dataset — coding samples, conversations, articles.
    Returns dict with training stats, cue probe results, and gate checks.
    """
    t0 = time.perf_counter()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Seed brain
    # ------------------------------------------------------------------
    if verbose:
        print(f"Seeding brain (fast, {n_traces} traces)...")
    _, trace_store = seed_brain_fast(n_traces=n_traces, verbose=False, chunk_count=seed_chunks)

    # ------------------------------------------------------------------
    # 2. Generate or load samples
    # ------------------------------------------------------------------
    if hf_dataset is not None:
        if verbose:
            print(f"Loading HuggingFace dataset '{hf_dataset}' ({n_samples} samples)...")

        # Try text dataset first, fall back to conversation dataset
        from brain.datasets.downloader import load_text_dataset, load_conversation_dataset
        _TEXT_DATASETS = {"ag_news", "imdb"}
        if hf_dataset in _TEXT_DATASETS:
            hf_data = load_text_dataset(hf_dataset, max_samples=n_samples)
        else:
            hf_data = load_conversation_dataset(hf_dataset, max_samples=n_samples)

        samples = [
            {
                "index": i,
                "topic": hf_dataset,
                "system": row.get("system", ""),
                "user": row.get("user", ""),
                "assistant": row.get("assistant", ""),
                "text": row["text"],
                "key_concepts": [],
            }
            for i, row in enumerate(hf_data)
        ]
    else:
        if verbose:
            print(f"Generating {n_samples} coding assistant samples...")
        samples = _generate_coding_samples(n_samples)

    # ------------------------------------------------------------------
    # 3. Seed vocabulary from ALL sample text (dataset-agnostic)
    # ------------------------------------------------------------------
    all_texts = [s["text"] for s in samples]
    dataset_vocab = _extract_vocabulary(all_texts)

    # Also add cue probe words so they can be decoded in speech
    for probe in _CUE_PROBES:
        for w in probe["cue"].split():
            if is_content_text_token(w):
                dataset_vocab.add(w.lower())
        for w in probe["expected_concepts"]:
            dataset_vocab.add(w.lower())

    rng = random.Random(42)
    vocab_count = _seed_vocabulary_traces(trace_store, dataset_vocab, rng, prefix="dv")
    if verbose:
        print(f"  {brain_core.get_neuron_count()} neurons, "
              f"{brain_core.get_synapse_count():,} synapses, "
              f"{len(trace_store)} traces ({vocab_count} dataset vocab seeded)")

    # ------------------------------------------------------------------
    # 4. Init learning infrastructure
    # ------------------------------------------------------------------
    tick_loop = TickLoop(trace_store)
    tick_loop.collect_full_metrics = False
    speech_decoder = SpeechDecoder(trace_store)
    schema_store = SchemaStore()
    schema_engine = SchemaFormationEngine(schema_store)
    text_input = TextInput(trace_store)

    # Warmup tick
    tick_loop.step()

    # Quick vocab coverage check
    if verbose and samples:
        test_tokens = text_input.tokenize(samples[0]["text"])
        test_matched = sum(
            1 for tok in test_tokens
            if tok in text_input._token_cache or tok in text_input._normalized_token_cache
        )
        print(f"  Vocab coverage (sample 0): {test_matched}/{len(test_tokens)} "
              f"({100*test_matched/max(len(test_tokens),1):.0f}%)")

    # ------------------------------------------------------------------
    # 5. Train via chunking pipeline
    # ------------------------------------------------------------------
    if verbose:
        cs_label = f"{chunk_size}" if chunk_size is not None else "auto"
        print(f"Training on {len(samples)} samples (chunk_size={cs_label}, "
              f"ticks_per_chunk={ticks_per_chunk}, rest={rest_ticks_between_chunks})...")

    total_traces_formed = 0
    total_bindings_formed = 0
    total_chunks_processed = 0
    total_ticks = 0
    all_distinct_trace_ids: set[str] = set()
    novelty_sum = 0.0
    novelty_count = 0
    skip_reason_counts: dict[str, int] = {}
    train_start = time.perf_counter()

    for si, sample in enumerate(samples):
        # Full runtime reset between samples — crucially resets the novelty
        # EMA so each new sample starts as "novel" and can trigger trace formation.
        tick_loop.reset_runtime_boundary(preserve_binding_state=True)

        full_text = sample["text"]
        chunks = chunk_text(full_text, chunk_size=chunk_size)

        for chunk in chunks:
            text_input.encode(chunk)
            for t in range(ticks_per_chunk):
                result = tick_loop.step(learn=True)
                total_traces_formed += result.get("traces_formed", 0)
                total_bindings_formed += result.get("bindings_formed", 0)

                # Track novelty
                novelty_sum += tick_loop.last_novelty
                novelty_count += 1

                # Track trace formation skip reasons
                skip = tick_loop.trace_formation.last_step_debug.get("failure_stage", "")
                if skip and skip != "none":
                    skip_reason_counts[skip] = skip_reason_counts.get(skip, 0) + 1

                # Feed schema engine on last tick of each chunk
                if t == ticks_per_chunk - 1:
                    schema_engine.step(tick_loop.last_active_traces, tick_loop.last_tick_number)

                # Collect new trace IDs
                for tid in getattr(tick_loop.trace_formation, "_recently_formed", []):
                    all_distinct_trace_ids.add(tid)

            # Rest between chunks
            for _ in range(rest_ticks_between_chunks):
                tick_loop.step(learn=False)
            total_chunks_processed += 1
            total_ticks += ticks_per_chunk + rest_ticks_between_chunks

        if verbose and ((si + 1) % 10 == 0 or si == 0):
            elapsed = time.perf_counter() - train_start
            tps = total_ticks / max(elapsed, 0.001)
            avg_novelty = novelty_sum / max(novelty_count, 1)
            print(f"  [{si+1}/{len(samples)}] ticks={total_ticks}, "
                  f"traces={total_traces_formed}, "
                  f"bindings={total_bindings_formed}, "
                  f"schemas={len(schema_store)}, "
                  f"avg_novelty={avg_novelty:.3f}, "
                  f"tps={tps:.0f}")

    train_elapsed = time.perf_counter() - train_start
    train_tps = total_ticks / max(train_elapsed, 0.001)

    if verbose:
        print(f"  Training complete: {total_ticks} ticks in {train_elapsed:.1f}s "
              f"({train_tps:.0f} tps)")
        print(f"  Traces formed: {total_traces_formed}, Bindings: {total_bindings_formed}, "
              f"Schemas: {len(schema_store)}")
        if skip_reason_counts:
            top_reasons = sorted(skip_reason_counts.items(), key=lambda x: -x[1])[:5]
            print(f"  Trace formation skip reasons: {dict(top_reasons)}")

    # ------------------------------------------------------------------
    # 6. Refresh speech decoder index (new traces may have speech neurons)
    # ------------------------------------------------------------------
    speech_decoder.refresh_index()

    # ------------------------------------------------------------------
    # 7. Probe with cues
    # ------------------------------------------------------------------
    probe_ticks = max(ticks_per_chunk, 15)  # enough for full chain propagation

    if verbose:
        print(f"\nRunning {len(_CUE_PROBES)} cue probes ({probe_ticks} ticks each)...")

    cue_results: list[dict] = []
    for probe in _CUE_PROBES:
        tick_loop.reset_runtime_boundary(preserve_binding_state=True)
        text_input.encode(probe["cue"])

        # Accumulate speech over probe ticks
        speech_decoder.reset_window()
        for t in range(probe_ticks):
            tick_loop.step(learn=False)
            speech_decoder.accumulate_tick()

        window_result = speech_decoder.decode_window()
        output_words = set()
        for label, score in window_result.get("tokens", []):
            for w in label.lower().split():
                output_words.add(w)

        # Check overlap with expected concepts
        expected = set(w.lower() for w in probe["expected_concepts"])
        hits = output_words & expected
        key_concept_count = len(hits)

        cue_results.append({
            "cue": probe["cue"],
            "description": probe["description"],
            "expected_concepts": list(expected),
            "speech_output": window_result.get("text", ""),
            "speech_tokens": window_result.get("tokens", []),
            "output_words": sorted(output_words),
            "concept_hits": sorted(hits),
            "key_concept_count": key_concept_count,
        })

        if verbose:
            print(f"  Cue: {probe['cue']!r}")
            print(f"    Speech: {window_result.get('text', '')!r}")
            print(f"    Concept hits: {sorted(hits)} ({key_concept_count}/{len(expected)})")

    # ------------------------------------------------------------------
    # 8. Compute gates
    # ------------------------------------------------------------------
    min_concept_hits_per_cue = min(cr["key_concept_count"] for cr in cue_results) if cue_results else 0
    any_speech_output = any(cr["speech_output"] for cr in cue_results)
    schemas_formed = len(schema_store)
    avg_novelty = novelty_sum / max(novelty_count, 1)

    result = {
        "training": {
            "n_samples": len(samples),
            "total_chunks": total_chunks_processed,
            "total_ticks": total_ticks,
            "elapsed_sec": train_elapsed,
            "ticks_per_sec": train_tps,
            "traces_formed": total_traces_formed,
            "bindings_formed": total_bindings_formed,
            "distinct_trace_ids": len(all_distinct_trace_ids),
            "schemas_formed": schemas_formed,
            "avg_novelty": avg_novelty,
            "skip_reasons": skip_reason_counts,
        },
        "cue_results": cue_results,
        "gates": {
            "any_speech_output": any_speech_output,
            "min_concept_hits_per_cue": min_concept_hits_per_cue,
            "at_least_one_schema": schemas_formed >= 1,
            "ticks_per_sec_above_5": train_tps > 5,
        },
        "schema_summary": {
            "total": schemas_formed,
            "schema_ids": [s.id for s in schema_store][:20],
        },
        "elapsed_total_sec": time.perf_counter() - t0,
    }

    output_file.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if verbose:
        print(f"\nResults saved to {output_path}")
        print(f"Gates: {result['gates']}")

    return result
