"""Text input pipeline: convert text into queued language-region inputs.

Tokenizes input text, maps tokens to traces via TraceStore labels,
and injects the corresponding semantic signals into the Rust brain input path.
Those signals become observable as active neurons on the next tick.
Unknown tokens get a hash-based transient activation pattern.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import brain_core

from brain.structures.trace_store import Trace, TraceStore
from brain.utils.config import (
    REGIONS,
    TEXT_INPUT_KNOWN_REGION_SCALES,
    TEXT_INPUT_MAX_MATCHED_TRACES_PER_SPAN,
    TEXT_INPUT_UNKNOWN_REGION_BOOSTS,
    TEXT_INPUT_UNKNOWN_REGION_COUNTS,
)


# Language region token neuron sub-population
_LANG_START = REGIONS["language"][0]  # 105000
_TOKEN_COUNT = 9000  # token neurons: 105000–113999
_OUTPUT_REGIONS = {"motor", "speech"}
_TOKEN_ALIAS_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "if", "in", "of", "on",
    "or", "the", "to", "with",
}
TEXT_INPUT_CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "them", "they", "this", "to", "was", "were", "with",
    "will", "would", "after", "about", "not", "new", "ap", "reuters",
}
_NORMALIZATION_SUFFIX_MIN_LENGTH = 5
_NORMALIZATION_SPLIT_DELIMITERS = ("-", "/", ".")
_DOUBLE_CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


def is_content_text_token(token: str) -> bool:
    return len(token) >= 3 and token not in TEXT_INPUT_CONTENT_STOPWORDS


def normalize_text_token_variants(token: str) -> list[str]:
    """Return exact-first normalization variants for a token.

    This is intentionally limited to low-risk normalization only:
    punctuation cleanup and common suffix reduction. It does not attempt any
    semantic aliasing or lexical-neighborhood mapping.
    """
    base = token.lower().replace("’", "'").strip()
    variants: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = value.strip(".,!?;:'\"()[]{}")
        if not value or value in seen:
            return
        seen.add(value)
        variants.append(value)

    def add_root_variants(root: str, *, allow_silent_e: bool = True) -> None:
        if len(root) < 3:
            return
        add(root)
        if len(root) >= 2 and root[-1] == root[-2] and root[-1] in _DOUBLE_CONSONANTS:
            add(root[:-1])
        if allow_silent_e and not root.endswith("e"):
            add(root + "e")

    add(base)
    add(base.replace("'", ""))
    add(base.replace(".", ""))
    add(base.replace("-", ""))
    add(base.replace("/", ""))

    for delimiter in _NORMALIZATION_SPLIT_DELIMITERS:
        if delimiter not in base:
            continue
        for piece in base.split(delimiter):
            add(piece)

    if base.endswith("'s") and len(base) > 2:
        add(base[:-2])
    if base.endswith("s'") and len(base) > 2:
        add(base[:-1])

    index = 0
    while index < len(variants):
        value = variants[index]
        index += 1
        if len(value) < _NORMALIZATION_SUFFIX_MIN_LENGTH:
            continue
        if value.endswith("ies"):
            add(value[:-3] + "y")
        elif value.endswith("es"):
            add(value[:-2])
        if value.endswith("s") and not value.endswith("ss"):
            add(value[:-1])
        if value.endswith("ing") and len(value) > 6:
            add_root_variants(value[:-3])
        if value.endswith("ed") and len(value) > 5:
            add_root_variants(value[:-2])
        if value.endswith("ier") and len(value) > 5:
            add(value[:-3] + "y")
        elif value.endswith("er") and len(value) > 5 and not value.endswith("eer"):
            add_root_variants(value[:-2])

    return variants


def split_text_token_phrase_variants(token: str) -> list[tuple[str, ...]]:
    """Split punctuation-joined tokens into exact-normalized phrase candidates."""
    token = token.lower().replace("’", "'")
    variants: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for delimiter in _NORMALIZATION_SPLIT_DELIMITERS:
        if delimiter not in token:
            continue
        pieces = []
        for part in token.split(delimiter):
            normalized = normalize_text_token_variants(part)
            if normalized:
                pieces.append(normalized[0])
        phrase = tuple(piece for piece in pieces if piece)
        if len(phrase) > 1 and phrase not in seen:
            seen.add(phrase)
            variants.append(phrase)
    return variants


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
        self._normalized_token_cache: dict[str, list[int]] = {}
        self._token_traces: dict[str, list[str]] = defaultdict(list)
        self._normalized_token_traces: dict[str, list[str]] = defaultdict(list)
        self._phrase_traces: dict[tuple[str, ...], list[str]] = defaultdict(list)
        self._max_phrase_len = 1
        self._build_token_cache()

    @staticmethod
    def _extend_token_lookup(
        token_traces: dict[str, list[str]],
        token_cache: dict[str, list[int]],
        key: str,
        trace_id: str,
        lang_neurons: list[int],
    ) -> None:
        token_traces[key].append(trace_id)
        token_cache.setdefault(key, [])
        token_cache[key].extend(lang_neurons)

    def _build_token_cache(self) -> None:
        """Build token and phrase caches from trace labels."""
        self._token_cache.clear()
        self._normalized_token_cache.clear()
        self._token_traces.clear()
        self._normalized_token_traces.clear()
        self._phrase_traces.clear()
        self._max_phrase_len = 1

        for trace in self.trace_store.traces.values():
            if trace.label is not None:
                lang_neurons = trace.neurons.get("language", [])
                if not lang_neurons:
                    continue

                for alias in self._label_aliases(trace.label):
                    if not alias:
                        continue
                    if len(alias) == 1:
                        key = alias[0]
                        self._extend_token_lookup(
                            self._token_traces,
                            self._token_cache,
                            key,
                            trace.id,
                            lang_neurons,
                        )
                        for normalized_key in normalize_text_token_variants(key):
                            self._extend_token_lookup(
                                self._normalized_token_traces,
                                self._normalized_token_cache,
                                normalized_key,
                                trace.id,
                                lang_neurons,
                            )
                    else:
                        self._phrase_traces[alias].append(trace.id)
                        self._max_phrase_len = max(self._max_phrase_len, len(alias))

                for digit_alias in self._digit_aliases(trace):
                    self._extend_token_lookup(
                        self._token_traces,
                        self._token_cache,
                        digit_alias,
                        trace.id,
                        lang_neurons,
                    )

        for key, neurons in self._token_cache.items():
            self._token_cache[key] = sorted(set(neurons))
        for key, neurons in self._normalized_token_cache.items():
            self._normalized_token_cache[key] = sorted(set(neurons))

    @staticmethod
    def _digit_aliases(trace: Trace) -> list[str]:
        if not trace.id.startswith("number_") and "numbers" not in trace.neurons:
            return []

        if trace.id.startswith("number_"):
            suffix = trace.id.removeprefix("number_")
            if suffix.isdigit():
                return [str(int(suffix))]

        return []

    @staticmethod
    def _normalize_fragment(fragment: str) -> str:
        return fragment.strip(".,!?;:'\"()[]{}")

    def _label_aliases(self, label: str) -> list[tuple[str, ...]]:
        raw_parts = [
            self._normalize_fragment(part)
            for part in re.split(r"[_\s-]+", label.lower().strip())
        ]
        parts = tuple(part for part in raw_parts if part)
        if not parts:
            return []

        aliases: list[tuple[str, ...]] = [parts]
        if len(parts) > 1:
            for part in parts:
                if len(part) >= 3 and part not in _TOKEN_ALIAS_STOPWORDS:
                    aliases.append((part,))
        return aliases

    def refresh_cache(self) -> None:
        """Rebuild the token cache (call after adding new traces)."""
        self._build_token_cache()

    def tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenizer with lowercasing and punctuation stripping."""
        tokens = []
        for word in text.lower().split():
            # Strip common punctuation
            cleaned = word.replace("’", "'").strip(".,!?;:'\"()[]{}")
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

    @staticmethod
    def _region_hash_neurons(token: str, region_name: str, count: int) -> list[int]:
        start, end = REGIONS[region_name]
        span = end - start + 1
        digest = hashlib.sha256(f"{region_name}:{token}".encode("utf-8")).digest()
        neurons: list[int] = []
        cursor = 0
        while len(neurons) < count:
            offset = (digest[cursor % len(digest)] << 8 | digest[(cursor + 1) % len(digest)]) % span
            neuron_id = start + offset
            if neuron_id not in neurons:
                neurons.append(neuron_id)
            cursor += 2
        return neurons

    @staticmethod
    def _accumulate_signals(signal_map: dict[int, float], neurons: list[int], boost: float) -> None:
        if boost <= 0.0:
            return
        for neuron_id in neurons:
            signal_map[neuron_id] = min(1.0, signal_map.get(neuron_id, 0.0) + boost)

    def _resolve_span_traces(self, span: tuple[str, ...]) -> list[str]:
        if len(span) > 1:
            trace_ids = self._phrase_traces.get(span, [])
        else:
            trace_ids = self._token_traces.get(span[0], [])
        return self._resolve_trace_ids(trace_ids)

    def _resolve_normalized_token_traces(self, token: str) -> list[str]:
        trace_ids = self._normalized_token_traces.get(token, [])
        return self._resolve_trace_ids(trace_ids)

    def _resolve_trace_ids(self, trace_ids: list[str]) -> list[str]:
        if not trace_ids:
            return []

        ranked = sorted(
            {
                trace_id
                for trace_id in trace_ids
                if self.trace_store.get(trace_id) is not None
            },
            key=lambda trace_id: self.trace_store.get(trace_id).total_neurons(),
            reverse=True,
        )
        return ranked[:TEXT_INPUT_MAX_MATCHED_TRACES_PER_SPAN]

    def _inject_known_trace_semantics(
        self,
        trace_ids: list[str],
        signal_map: dict[int, float],
        span_text: str,
    ) -> None:
        for trace_id in trace_ids:
            trace = self.trace_store.get(trace_id)
            if trace is None:
                continue
            for region_name, neurons in trace.neurons.items():
                if not neurons or region_name in _OUTPUT_REGIONS:
                    continue
                scale = TEXT_INPUT_KNOWN_REGION_SCALES.get(region_name)
                if scale is None:
                    continue
                self._accumulate_signals(signal_map, neurons, self.boost * scale)

        attention_neurons = self._region_hash_neurons(span_text, "attention", 2)
        self._accumulate_signals(signal_map, attention_neurons, self.boost * 0.3)

    def _inject_unknown_token(self, token: str, signal_map: dict[int, float]) -> None:
        for region_name, boost in TEXT_INPUT_UNKNOWN_REGION_BOOSTS.items():
            count = TEXT_INPUT_UNKNOWN_REGION_COUNTS.get(region_name, 0)
            if count <= 0:
                continue
            neurons = self._region_hash_neurons(token, region_name, count)
            self._accumulate_signals(signal_map, neurons, boost)

    def _find_exact_span_match(self, tokens: list[str], start_idx: int) -> tuple[int, list[str]]:
        max_len = min(self._max_phrase_len, len(tokens) - start_idx)
        for span_len in range(max_len, 1, -1):
            span = tuple(tokens[start_idx:start_idx + span_len])
            trace_ids = self._resolve_span_traces(span)
            if trace_ids:
                return span_len, trace_ids

        span = tuple(tokens[start_idx:start_idx + 1])
        trace_ids = self._resolve_span_traces(span)
        if trace_ids:
            return 1, trace_ids
        return 0, []

    def _find_normalized_token_match(self, token: str) -> tuple[int, list[str]]:
        for phrase_variant in split_text_token_phrase_variants(token):
            trace_ids = self._resolve_span_traces(phrase_variant)
            if trace_ids:
                return 1, trace_ids

        for variant in normalize_text_token_variants(token):
            trace_ids = self._resolve_normalized_token_traces(variant)
            if trace_ids:
                return 1, trace_ids

        return 0, []

    def _find_span_match(self, tokens: list[str], start_idx: int) -> tuple[int, list[str]]:
        span_len, trace_ids = self._find_exact_span_match(tokens, start_idx)
        if span_len > 0:
            return span_len, trace_ids
        return self._find_normalized_token_match(tokens[start_idx])

    def encode(self, text: str) -> dict:
        """Encode text into brain activations.

        Steps:
          1. Tokenize the text
          2. For each token, look up trace language neurons (or hash-generate)
                    3. Inject those signals into the Rust brain for the next tick

        Returns:
            dict with keys: tokens, known_count, unknown_count, neurons_activated
        """
        tokens = self.tokenize(text)
        known_count = 0
        unknown_count = 0
        matched_traces: list[str] = []
        signal_map: dict[int, float] = {}

        idx = 0
        while idx < len(tokens):
            span_len, trace_ids = self._find_span_match(tokens, idx)
            if span_len > 0:
                span_tokens = tokens[idx:idx + span_len]
                known_count += span_len
                matched_traces.extend(trace_ids)
                self._inject_known_trace_semantics(trace_ids, signal_map, " ".join(span_tokens))
                idx += span_len
                continue

            token = tokens[idx]
            self._inject_unknown_token(token, signal_map)
            unknown_count += 1
            idx += 1

        neurons_activated = len(signal_map)
        if signal_map:
            brain_core.inject_activations(list(signal_map.items()))

        return {
            "tokens": tokens,
            "known_count": known_count,
            "unknown_count": unknown_count,
            "neurons_activated": neurons_activated,
            "matched_traces": sorted(set(matched_traces)),
        }

    def encode_token(self, token: str) -> list[int]:
        """Encode a single token, returning the neuron IDs used (without activating)."""
        key = token.lower().replace("’", "'").strip()
        if key in self._token_cache:
            return self._token_cache[key]
        for variant in normalize_text_token_variants(key):
            if variant in self._normalized_token_cache:
                return self._normalized_token_cache[variant]
        return self._hash_neurons(key)
