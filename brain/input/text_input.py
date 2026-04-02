"""Text input pipeline: convert text into language region activations.

Tokenizes input text, maps tokens to traces via TraceStore labels,
and activates the corresponding language neurons in the Rust brain.
Unknown tokens get a hash-based transient activation pattern.
"""

from __future__ import annotations

import hashlib

import brain_core

from brain.structures.trace_store import TraceStore
from brain.utils.config import REGIONS


# Language region token neuron sub-population
_LANG_START = REGIONS["language"][0]  # 105000
_TOKEN_COUNT = 9000  # token neurons: 105000–113999


class TextInput:
    """Encodes text into language region neuron activations.

    Usage:
        encoder = TextInput(trace_store)
        activated = encoder.encode("the cat sat")
    """

    def __init__(self, trace_store: TraceStore, boost: float = 0.8):
        self.trace_store = trace_store
        self.boost = boost
        # Cache: token string → list of language neuron IDs
        self._token_cache: dict[str, list[int]] = {}
        self._build_token_cache()

    def _build_token_cache(self) -> None:
        """Build a mapping from trace labels to their language neurons."""
        for trace in self.trace_store.traces.values():
            if trace.label is not None:
                lang_neurons = trace.neurons.get("language", [])
                if lang_neurons:
                    key = trace.label.lower().strip()
                    self._token_cache[key] = lang_neurons

    def refresh_cache(self) -> None:
        """Rebuild the token cache (call after adding new traces)."""
        self._token_cache.clear()
        self._build_token_cache()

    def tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenizer with lowercasing and punctuation stripping."""
        tokens = []
        for word in text.lower().split():
            # Strip common punctuation
            cleaned = word.strip(".,!?;:'\"()-")
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def _hash_neurons(self, token: str, count: int = 3) -> list[int]:
        """Generate deterministic pseudo-random language neuron IDs for unknown tokens."""
        h = hashlib.sha256(token.encode("utf-8")).digest()
        neurons = []
        for i in range(count):
            # Use successive bytes to pick neuron offsets within token sub-population
            offset = (h[i * 2] << 8 | h[i * 2 + 1]) % _TOKEN_COUNT
            neurons.append(_LANG_START + offset)
        return neurons

    def encode(self, text: str) -> dict:
        """Encode text into brain activations.

        Steps:
          1. Tokenize the text
          2. For each token, look up trace language neurons (or hash-generate)
          3. Activate those neurons via Rust

        Returns:
            dict with keys: tokens, known_count, unknown_count, neurons_activated
        """
        tokens = self.tokenize(text)
        known_count = 0
        unknown_count = 0
        all_neurons: list[int] = []
        matched_traces: list[str] = []

        for token in tokens:
            if token in self._token_cache:
                neurons = self._token_cache[token]
                known_count += 1
                # Find the trace ID for this token
                for trace in self.trace_store.traces.values():
                    if trace.label and trace.label.lower().strip() == token:
                        matched_traces.append(trace.id)
                        break
            else:
                neurons = self._hash_neurons(token)
                unknown_count += 1

            all_neurons.extend(neurons)

        # Activate language neurons in Rust
        neurons_activated = 0
        if all_neurons:
            neurons_activated = brain_core.boost_language(all_neurons, self.boost)

        return {
            "tokens": tokens,
            "known_count": known_count,
            "unknown_count": unknown_count,
            "neurons_activated": neurons_activated,
            "matched_traces": matched_traces,
        }

    def encode_token(self, token: str) -> list[int]:
        """Encode a single token, returning the neuron IDs used (without activating)."""
        key = token.lower().strip()
        if key in self._token_cache:
            return self._token_cache[key]
        return self._hash_neurons(key)
