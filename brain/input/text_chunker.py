"""Text chunking pipeline: dynamically split text into brain-digestible chunks.

Adaptive chunking that works at any document size — from a single sentence
to 100k tokens.  Respects natural language boundaries in priority order:
paragraph > sentence > clause > word.  Chunk target scales with document
length so short texts stay whole and long documents produce manageable
chunk counts.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

from brain.input.text_input import TextInput
from brain.learning.tick_loop import TickLoop
from brain.output.speech_output import SpeechOutput
from brain.structures.trace_store import TraceStore
from brain.utils.config import (
    CHUNK_MIN_WORDS,
    CHUNK_MAX_WORDS,
)


# ---------------------------------------------------------------------------
# Boundary patterns (ordered from strongest to weakest break)
# ---------------------------------------------------------------------------
_PARA_SPLIT = re.compile(r'\n\s*\n')                        # paragraph
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')              # sentence
_CLAUSE_SPLIT = re.compile(r'(?<=[;:,])\s+|(?<=\s[-–—])\s') # clause


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""
    chunk_index: int
    text: str
    word_count: int
    ticks_run: int
    rest_ticks_run: int
    traces_formed: int
    bindings_formed: int
    active_traces: list[tuple[str, float]]
    speech_output: str
    speech_tokens: list[tuple[str, float]]
    encoding: dict
    tick_results: list[dict] = field(default_factory=list)


@dataclass
class ChunkedDocumentResult:
    """Result from processing an entire document through chunking."""
    total_chunks: int
    total_words: int
    total_ticks: int
    total_traces_formed: int
    total_bindings_formed: int
    elapsed_sec: float
    chunks: list[ChunkResult] = field(default_factory=list)
    distinct_trace_ids: set = field(default_factory=set)


# ---------------------------------------------------------------------------
# Adaptive target computation
# ---------------------------------------------------------------------------

def _adaptive_target(total_words: int) -> int:
    """Compute a chunk target (in words) that scales with document length.

    Short texts  (<= 25 words)  → single chunk (no split)
    Medium       (25 – 150)     → ~15 words/chunk
    Long         (150 – 800)    → ~25 words/chunk
    Very long    (800 – 3000)   → ~40 words/chunk
    Huge         (3000+)        → ~55 words/chunk  (capped by CHUNK_MAX_WORDS)

    The formula uses a smooth log-scale so there are no hard jumps.
    """
    if total_words <= 25:
        return total_words  # keep as one chunk
    # log-scaled ramp: ~15 at 30 words, ~55 at 5000+
    target = int(10 + 8 * math.log2(max(total_words, 30) / 15))
    return max(CHUNK_MIN_WORDS, min(target, CHUNK_MAX_WORDS))


# ---------------------------------------------------------------------------
# Core splitter
# ---------------------------------------------------------------------------

def _split_segments(text: str, pattern: re.Pattern) -> list[str]:
    """Split *text* by *pattern*, returning non-empty stripped segments."""
    parts = pattern.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _greedy_merge(segments: list[str], target: int, max_words: int) -> list[str]:
    """Merge adjacent segments until the running word count reaches *target*.

    Never produces a chunk > *max_words* unless a single segment already
    exceeds it (handled by the caller via deeper splitting).
    """
    chunks: list[str] = []
    buf: list[str] = []
    buf_wc = 0

    for seg in segments:
        seg_wc = len(seg.split())
        # Would adding this segment blow past max_words?  Flush first.
        if buf and buf_wc + seg_wc > max_words:
            chunks.append(" ".join(buf))
            buf, buf_wc = [], 0

        buf.append(seg)
        buf_wc += seg_wc

        if buf_wc >= target:
            chunks.append(" ".join(buf))
            buf, buf_wc = [], 0

    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _split_deep(chunk: str, target: int, max_words: int) -> list[str]:
    """Break a chunk at clause → word boundaries when it exceeds *target*.

    Sentence-level splitting is handled by the caller (chunk_text processes
    each paragraph independently at the sentence level first).
    """
    words = chunk.split()
    slack = target + target // 2  # allow 150 % of target before splitting

    if len(words) <= slack:
        return [chunk]

    # Try clause split (semicolons, colons, commas, dashes)
    clauses = _split_segments(chunk, _CLAUSE_SPLIT)
    if len(clauses) > 1:
        merged = _greedy_merge(clauses, target, max_words)
        out: list[str] = []
        for m in merged:
            mw = len(m.split())
            if mw <= slack:
                out.append(m)
            else:
                # Force word-boundary split
                ws = m.split()
                for i in range(0, len(ws), target):
                    seg = " ".join(ws[i : i + target])
                    if seg:
                        out.append(seg)
        return out

    # Last resort: hard word-boundary split
    result: list[str] = []
    for i in range(0, len(words), target):
        seg = " ".join(words[i : i + target])
        if seg:
            result.append(seg)
    return result


def chunk_text(
    text: str,
    chunk_size: int | None = None,
) -> list[str]:
    """Split *text* into chunks that respect natural language boundaries.

    Parameters
    ----------
    text : str
        The input text, any length.
    chunk_size : int or None
        Explicit word-count target per chunk.  When *None* (the default)
        the target is computed adaptively based on document length.

    Returns a list of non-empty text chunks.
    """
    stripped = text.strip()
    if not stripped:
        return []

    total_words = len(stripped.split())
    target = chunk_size if chunk_size is not None else _adaptive_target(total_words)
    max_words = max(target * 2, CHUNK_MAX_WORDS)
    slack = target + target // 2  # 150 % tolerance

    # If the whole text fits in one chunk, return immediately
    if total_words <= target:
        return [stripped]

    # 1. Paragraph split — each paragraph processed independently
    paragraphs = _split_segments(stripped, _PARA_SPLIT)
    if not paragraphs:
        paragraphs = [stripped]

    raw_chunks: list[str] = []
    for para in paragraphs:
        para_wc = len(para.split())

        if para_wc <= slack:
            # Paragraph fits within tolerance — keep whole
            raw_chunks.append(para)
        else:
            # Split paragraph at sentence boundaries, then greedy-merge
            sents = _split_segments(para, _SENTENCE_SPLIT)
            if len(sents) <= 1:
                # No sentence boundaries — go straight to clause/word
                raw_chunks.extend(_split_deep(para, target, max_words))
            else:
                merged = _greedy_merge(sents, target, max_words)
                for m in merged:
                    raw_chunks.extend(_split_deep(m, target, max_words))

    # 2. Merge tiny fragments with their predecessor
    final: list[str] = []
    for chunk in raw_chunks:
        wc = len(chunk.split())
        if final and wc < CHUNK_MIN_WORDS:
            final[-1] = final[-1] + " " + chunk
        else:
            final.append(chunk)

    return final if final else [stripped]


def process_chunked_document(
    text: str,
    trace_store: TraceStore,
    tick_loop: TickLoop,
    *,
    chunk_size: int | None = None,
    ticks_per_chunk: int = 5,
    rest_ticks: int = 2,
    speech_decoder: SpeechOutput | None = None,
    learn: bool = True,
) -> ChunkedDocumentResult:
    """Process a long document through the chunking pipeline.

    Each chunk is encoded via TextInput, ticked for ticks_per_chunk,
    then rested for rest_ticks to allow trace consolidation.
    Cross-chunk binding happens naturally through the existing binding tracker
    when chunks share vocabulary.
    """
    t0 = time.perf_counter()
    chunks = chunk_text(text, chunk_size=chunk_size)
    text_input = TextInput(trace_store)

    result = ChunkedDocumentResult(
        total_chunks=len(chunks),
        total_words=len(text.split()),
        total_ticks=0,
        total_traces_formed=0,
        total_bindings_formed=0,
        elapsed_sec=0.0,
    )

    for ci, chunk in enumerate(chunks):
        enc = text_input.encode(chunk)

        chunk_traces_formed = 0
        chunk_bindings_formed = 0
        chunk_tick_results: list[dict] = []
        last_active_traces: list[tuple[str, float]] = []

        # Active ticks
        for t in range(ticks_per_chunk):
            tick_result = tick_loop.step(learn=learn)
            chunk_tick_results.append(tick_result)
            chunk_traces_formed += tick_result.get("traces_formed", 0)
            chunk_bindings_formed += tick_result.get("bindings_formed", 0)
            last_active_traces = tick_loop.last_active_traces

            # Collect newly formed trace IDs
            for tid in getattr(tick_loop.trace_formation, "_recently_formed", []):
                result.distinct_trace_ids.add(tid)

        # Decode speech on last active tick
        speech_text = ""
        speech_tokens: list[tuple[str, float]] = []
        if speech_decoder is not None:
            speech = speech_decoder.decode(top_k=10)
            speech_text = speech.get("text", "")
            speech_tokens = speech.get("tokens", [])

        # Rest ticks for consolidation
        for _ in range(rest_ticks):
            tick_loop.step(learn=False)

        cr = ChunkResult(
            chunk_index=ci,
            text=chunk,
            word_count=len(chunk.split()),
            ticks_run=ticks_per_chunk,
            rest_ticks_run=rest_ticks,
            traces_formed=chunk_traces_formed,
            bindings_formed=chunk_bindings_formed,
            active_traces=last_active_traces,
            speech_output=speech_text,
            speech_tokens=speech_tokens,
            encoding=enc,
            tick_results=chunk_tick_results,
        )
        result.chunks.append(cr)
        result.total_ticks += ticks_per_chunk + rest_ticks
        result.total_traces_formed += chunk_traces_formed
        result.total_bindings_formed += chunk_bindings_formed

    result.elapsed_sec = time.perf_counter() - t0
    return result
