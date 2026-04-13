"""Enhanced speech decoder: accumulate speech activations across ticks.

Wraps the existing SpeechOutput with windowed accumulation and
configurable thresholds for chunk-level decoding.
"""

from __future__ import annotations

from collections import defaultdict

from brain.output.speech_output import SpeechOutput
from brain.structures.trace_store import TraceStore
from brain.utils.config import (
    SPEECH_DECODE_TOP_K,
    SPEECH_OUTPUT_THRESHOLD,
)


class SpeechDecoder:
    """Accumulates speech activations across ticks within a chunk window.

    Usage:
        decoder = SpeechDecoder(trace_store)
        # Per tick:
        decoder.accumulate_tick()
        # At chunk boundary:
        result = decoder.decode_window()
        decoder.reset_window()
    """

    def __init__(
        self,
        trace_store: TraceStore,
        threshold: float = SPEECH_OUTPUT_THRESHOLD,
        top_k: int = SPEECH_DECODE_TOP_K,
    ):
        self._speech_output = SpeechOutput(trace_store)
        self.threshold = threshold
        self.top_k = top_k
        # Accumulated votes: label → total_activation across ticks
        self._window_votes: dict[str, float] = defaultdict(float)
        self._window_tick_count: int = 0

    @property
    def speech_output(self) -> SpeechOutput:
        """Access underlying SpeechOutput for raw decoding."""
        return self._speech_output

    def refresh_index(self) -> None:
        """Rebuild the reverse index after adding new traces."""
        self._speech_output.refresh_index()

    def accumulate_tick(self, suppression: float = 0.7) -> dict:
        """Decode current tick's speech output and accumulate votes.

        Returns the raw per-tick decode result.
        """
        result = self._speech_output.decode(top_k=self.top_k * 2, suppression=suppression)
        for label, vote in result.get("tokens", []):
            self._window_votes[label] += vote
        self._window_tick_count += 1
        return result

    def decode_window(self) -> dict:
        """Return accumulated top-K tokens above threshold for the current window.

        Returns dict with:
            tokens: [(label, score), ...] sorted by descending score
            tick_count: number of ticks accumulated
            text: space-joined top labels
        """
        # Filter by threshold and sort
        filtered = [
            (label, score)
            for label, score in self._window_votes.items()
            if score >= self.threshold
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)
        tokens = filtered[: self.top_k]
        text = " ".join(label for label, _ in tokens)

        return {
            "tokens": tokens,
            "tick_count": self._window_tick_count,
            "text": text,
        }

    def reset_window(self) -> None:
        """Clear accumulated votes for the next chunk window."""
        self._window_votes.clear()
        self._window_tick_count = 0

    def decode_single(self, top_k: int | None = None, suppression: float = 0.7) -> dict:
        """One-shot decode without accumulation (delegates to SpeechOutput)."""
        return self._speech_output.decode(
            top_k=top_k or self.top_k,
            suppression=suppression,
        )
