"""Speech output pipeline: convert speech region activations into text.

Reads the most active speech neurons, maps them back to tokens via
a neuron→token reverse index built from TraceStore, and assembles output.
"""

from __future__ import annotations

from collections import defaultdict

import brain_core

from brain.structures.trace_store import TraceStore
from brain.utils.config import REGIONS


_SPEECH_START = REGIONS["speech"][0]  # 140000
_SPEECH_EXCITATORY_END = 148_000     # first 8k are excitatory (phoneme carriers)


class SpeechOutput:
    """Decodes speech region activations into text tokens.

    Usage:
        decoder = SpeechOutput(trace_store)
        result = decoder.decode(top_k=10)
    """

    def __init__(self, trace_store: TraceStore, min_activation: float = 0.01):
        self.trace_store = trace_store
        self.min_activation = min_activation
        # Reverse index: speech neuron global ID → list of (trace_id, label)
        self._neuron_to_tokens: dict[int, list[tuple[str, str]]] = defaultdict(list)
        self._build_reverse_index()

    def _build_reverse_index(self) -> None:
        """Build speech neuron → token reverse mapping from trace store."""
        for trace in self.trace_store.traces.values():
            if trace.label is None:
                continue
            speech_neurons = trace.neurons.get("speech", [])
            for nid in speech_neurons:
                if _SPEECH_START <= nid < _SPEECH_EXCITATORY_END:
                    self._neuron_to_tokens[nid].append(
                        (trace.id, trace.label)
                    )

    def refresh_index(self) -> None:
        """Rebuild the reverse index (call after adding new traces)."""
        self._neuron_to_tokens.clear()
        self._build_reverse_index()

    def decode(self, top_k: int = 10, suppression: float = 0.7) -> dict:
        """Decode current speech region activations into text.

        Steps:
          1. Apply lateral inhibition to sharpen output
          2. Read top-K active speech neurons
          3. Map neurons → tokens via reverse index (vote-based)
          4. Return ranked token list

        Returns:
            dict with keys: tokens, raw_neurons, speech_activity, text
        """
        # Apply lateral inhibition for winner-take-all
        brain_core.speech_lateral_inhibition(suppression)

        # Read peak speech neurons from Rust
        peaks = brain_core.get_peak_speech_neurons(top_k)
        speech_activity = brain_core.get_speech_activity()

        if not peaks:
            return {
                "tokens": [],
                "raw_neurons": [],
                "speech_activity": speech_activity,
                "text": "",
            }

        # Vote: each active speech neuron votes for the tokens it maps to
        token_votes: dict[str, float] = defaultdict(float)
        token_labels: dict[str, str] = {}  # trace_id → label

        for neuron_id, activation in peaks:
            for trace_id, label in self._neuron_to_tokens.get(neuron_id, []):
                token_votes[trace_id] += activation
                token_labels[trace_id] = label

        # Sort by vote strength
        ranked = sorted(token_votes.items(), key=lambda x: x[1], reverse=True)
        tokens = []
        for trace_id, vote in ranked:
            label = token_labels.get(trace_id, "?")
            tokens.append((label, vote))

        # Assemble text from top-voted tokens
        text = " ".join(label for label, _ in tokens)

        return {
            "tokens": tokens,
            "raw_neurons": peaks,
            "speech_activity": speech_activity,
            "text": text,
        }

    def read_raw(self, top_k: int = 20) -> list[tuple[int, float]]:
        """Read raw peak speech neuron activations without decoding."""
        return brain_core.get_peak_speech_neurons(top_k)
