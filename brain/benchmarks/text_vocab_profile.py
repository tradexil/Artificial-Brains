"""Profile text vocabulary coverage strategies against real datasets.

This module intentionally does not modify the live TextInput path. It measures
how much additional token coverage is plausibly available from three levers:

1. Larger fast-seed trace counts
2. Better token normalization / alias matching
3. Lexical-neighborhood fallback for still-unknown single tokens

The goal is to rank likely next steps before investing in live-path changes.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from brain.datasets.downloader import load_text_dataset
from brain.input.text_input import TextInput, is_content_text_token, normalize_text_token_variants
from brain.seed.text_vocab_overlay import apply_text_vocab_overlay
from brain.seed.seed_runner import seed_brain_fast


_NEIGHBORHOOD_MIN_TOKEN_LENGTH = 5
_NEIGHBORHOOD_MIN_TRIGRAM_JACCARD = 0.5
_DEFAULT_PROFILE_SEED_TRACE_COUNTS = (5000, 20000)
@dataclass(frozen=True)
class CoverageStrategy:
    name: str
    description: str
    effort: str


def _trigrams(text: str) -> set[str]:
    if len(text) < 3:
        return {text}
    return {text[i:i + 3] for i in range(len(text) - 2)}


def _trigram_jaccard(left: str, right: str) -> float:
    left_grams = _trigrams(left)
    right_grams = _trigrams(right)
    union = left_grams | right_grams
    if not union:
        return 0.0
    return len(left_grams & right_grams) / len(union)


def _is_content_token(token: str) -> bool:
    return is_content_text_token(token)


class TextVocabularyProfile:
    def __init__(self, encoder: TextInput):
        self.encoder = encoder
        self._baseline_tokens = set(encoder._token_traces.keys())
        self._normalized_tokens: set[str] = set(encoder._normalized_token_traces.keys())
        self._token_neighbors: dict[str, set[str]] = defaultdict(set)
        self._build_indices()

    def _build_indices(self) -> None:
        for token in self._normalized_tokens:
            self._token_neighbors[token[0] if token else ""].add(token)

    def _baseline_span_match(self, tokens: list[str], start_idx: int) -> int:
        span_len, _trace_ids = self.encoder._find_exact_span_match(tokens, start_idx)
        return span_len

    def _normalized_span_match(self, tokens: list[str], start_idx: int) -> int:
        span_len, _trace_ids = self.encoder._find_span_match(tokens, start_idx)
        return span_len

    def _neighborhood_token_match(self, token: str) -> str | None:
        if not _is_content_token(token) or len(token) < _NEIGHBORHOOD_MIN_TOKEN_LENGTH:
            return None

        candidate_variants = normalize_text_token_variants(token)
        if not candidate_variants:
            return None

        best_token = None
        best_score = 0.0
        first_char = candidate_variants[0][0] if candidate_variants[0] else ""
        candidates = self._token_neighbors.get(first_char, set())
        for variant in candidate_variants:
            for candidate in candidates:
                if not _is_content_token(candidate):
                    continue
                if abs(len(candidate) - len(variant)) > 2:
                    continue
                if len(candidate) < _NEIGHBORHOOD_MIN_TOKEN_LENGTH:
                    continue
                if variant[:2] != candidate[:2]:
                    continue
                score = _trigram_jaccard(variant, candidate)
                if score < _NEIGHBORHOOD_MIN_TRIGRAM_JACCARD:
                    continue
                if score > best_score:
                    best_token = candidate
                    best_score = score
        return best_token

    def analyze_samples(self, samples: list[dict[str, object]]) -> dict[str, object]:
        baseline_known = 0
        normalized_known = 0
        neighborhood_known = 0
        total_tokens = 0
        content_total_tokens = 0
        baseline_content_known = 0
        normalized_content_known = 0
        neighborhood_content_known = 0

        baseline_unknown_counter: Counter[str] = Counter()
        normalized_remaining_unknown_counter: Counter[str] = Counter()
        normalized_recovered_counter: Counter[str] = Counter()
        neighborhood_recovered_counter: Counter[str] = Counter()
        neighborhood_match_map: Counter[str] = Counter()
        unique_tokens: Counter[str] = Counter()

        for sample in samples:
            tokens = self.encoder.tokenize(str(sample["text"]))
            total_tokens += len(tokens)
            unique_tokens.update(tokens)
            content_total_tokens += sum(1 for token in tokens if _is_content_token(token))

            idx = 0
            while idx < len(tokens):
                baseline_span_len = self._baseline_span_match(tokens, idx)
                if baseline_span_len > 0:
                    span_tokens = tokens[idx:idx + baseline_span_len]
                    content_span_count = sum(1 for token in span_tokens if _is_content_token(token))
                    baseline_known += baseline_span_len
                    normalized_known += baseline_span_len
                    neighborhood_known += baseline_span_len
                    baseline_content_known += content_span_count
                    normalized_content_known += content_span_count
                    neighborhood_content_known += content_span_count
                    idx += baseline_span_len
                    continue

                baseline_unknown_counter[tokens[idx]] += 1
                normalized_span_len = self._normalized_span_match(tokens, idx)
                token = tokens[idx]
                if normalized_span_len > 0:
                    span_tokens = tokens[idx:idx + normalized_span_len]
                    content_span_count = sum(1 for item in span_tokens if _is_content_token(item))
                    normalized_known += normalized_span_len
                    neighborhood_known += normalized_span_len
                    normalized_content_known += content_span_count
                    neighborhood_content_known += content_span_count
                    normalized_recovered_counter[token] += normalized_span_len
                    idx += normalized_span_len
                    continue

                neighbor = self._neighborhood_token_match(token)
                if neighbor is not None:
                    neighborhood_known += 1
                    if _is_content_token(token):
                        neighborhood_content_known += 1
                    neighborhood_recovered_counter[token] += 1
                    neighborhood_match_map[f"{token}->{neighbor}"] += 1
                else:
                    normalized_remaining_unknown_counter[token] += 1
                idx += 1

        baseline_unknown = max(0, total_tokens - baseline_known)
        normalized_unknown = max(0, total_tokens - normalized_known)
        neighborhood_unknown = max(0, total_tokens - neighborhood_known)

        return {
            "total_tokens": total_tokens,
            "unique_tokens": len(unique_tokens),
            "baseline": {
                "known": baseline_known,
                "unknown": baseline_unknown,
                "coverage": round(baseline_known / max(1, total_tokens), 4),
                "content_known": baseline_content_known,
                "content_unknown": max(0, content_total_tokens - baseline_content_known),
                "content_coverage": round(baseline_content_known / max(1, content_total_tokens), 4),
            },
            "normalized": {
                "known": normalized_known,
                "unknown": normalized_unknown,
                "coverage": round(normalized_known / max(1, total_tokens), 4),
                "recovered_tokens": normalized_known - baseline_known,
                "content_known": normalized_content_known,
                "content_unknown": max(0, content_total_tokens - normalized_content_known),
                "content_coverage": round(normalized_content_known / max(1, content_total_tokens), 4),
                "content_recovered_tokens": normalized_content_known - baseline_content_known,
                "top_recovered_tokens": normalized_recovered_counter.most_common(25),
            },
            "lexical_neighborhood": {
                "known": neighborhood_known,
                "unknown": neighborhood_unknown,
                "coverage": round(neighborhood_known / max(1, total_tokens), 4),
                "recovered_tokens": neighborhood_known - normalized_known,
                "content_known": neighborhood_content_known,
                "content_unknown": max(0, content_total_tokens - neighborhood_content_known),
                "content_coverage": round(neighborhood_content_known / max(1, content_total_tokens), 4),
                "content_recovered_tokens": neighborhood_content_known - normalized_content_known,
                "top_recovered_tokens": neighborhood_recovered_counter.most_common(25),
                "top_neighbor_matches": neighborhood_match_map.most_common(25),
            },
            "top_baseline_unknown_tokens": baseline_unknown_counter.most_common(40),
            "top_remaining_unknown_tokens_after_profiled_recovery": normalized_remaining_unknown_counter.most_common(40),
            "inventory": {
                "labeled_trace_count": sum(
                    1 for trace in self.encoder.trace_store.traces.values() if trace.label is not None
                ),
                "unlabeled_trace_count": sum(
                    1 for trace in self.encoder.trace_store.traces.values() if trace.label is None
                ),
                "baseline_single_token_aliases": len(self._baseline_tokens),
                "baseline_phrase_aliases": len(self.encoder._phrase_traces),
                "normalized_single_token_aliases": len(self._normalized_tokens),
                "content_token_count": content_total_tokens,
            },
        }


def run_text_vocab_profile(
    *,
    dataset: str,
    max_samples: int,
    output_path: str,
    seed_trace_counts: tuple[int, ...] = _DEFAULT_PROFILE_SEED_TRACE_COUNTS,
    seed_chunks: int | None = 1,
    overlay_terms: int = 0,
    overlay_samples: int = 500,
) -> dict[str, object]:
    if dataset not in {"ag_news", "imdb"}:
        raise ValueError(f"Unsupported text vocabulary profile dataset: {dataset}")

    samples = load_text_dataset(dataset, max_samples=max_samples)
    strategies = [
        CoverageStrategy(
            name="seed_trace_count",
            description="Increase fast-seed random trace count without changing label lookup semantics.",
            effort="low",
        ),
        CoverageStrategy(
            name="normalized_lookup",
            description="Add stronger token normalization and alias matching before falling back to unknown hashes.",
            effort="low-medium",
        ),
        CoverageStrategy(
            name="lexical_neighborhood",
            description="Map still-unknown single tokens into nearby known label aliases instead of raw hashes only.",
            effort="medium",
        ),
        CoverageStrategy(
            name="labeled_vocabulary_overlay",
            description="Add dataset-ranked unknown content terms as labeled cross-region traces.",
            effort="medium",
        ),
    ]

    seed_rows: list[dict[str, object]] = []
    baseline_seed_profile: dict[str, object] | None = None
    overlay_seed_profile: dict[str, object] | None = None
    overlay_selection_summary: dict[str, object] | None = None
    for seed_trace_count in seed_trace_counts:
        _, trace_store = seed_brain_fast(
            n_traces=seed_trace_count,
            verbose=False,
            chunk_count=seed_chunks,
        )
        profile = TextVocabularyProfile(TextInput(trace_store))
        row = profile.analyze_samples(samples)
        row["seed_traces"] = seed_trace_count

        overlay_row = None
        overlay_selection = None
        if overlay_terms > 0:
            overlay_selection = apply_text_vocab_overlay(
                trace_store,
                dataset,
                max_terms=overlay_terms,
                max_samples=overlay_samples,
            )
            overlay_profile = TextVocabularyProfile(TextInput(trace_store))
            overlay_row = overlay_profile.analyze_samples(samples)

        seed_rows.append(row)
        if baseline_seed_profile is None:
            baseline_seed_profile = row
        if overlay_row is not None:
            row["overlay_profile"] = overlay_row
            row["overlay_selection"] = overlay_selection
            if overlay_seed_profile is None:
                overlay_seed_profile = overlay_row
                overlay_selection_summary = overlay_selection

    if baseline_seed_profile is None:
        raise RuntimeError("No seed profile rows were generated")

    coverage_by_seed = [
        {
            "seed_traces": row["seed_traces"],
            "baseline_coverage": row["baseline"]["coverage"],
            "baseline_content_coverage": row["baseline"]["content_coverage"],
            "normalized_coverage": row["normalized"]["coverage"],
            "normalized_content_coverage": row["normalized"]["content_coverage"],
            "lexical_neighborhood_coverage": row["lexical_neighborhood"]["coverage"],
            "lexical_neighborhood_content_coverage": row["lexical_neighborhood"]["content_coverage"],
            "overlay_coverage": row.get("overlay_profile", {}).get("baseline", {}).get("coverage"),
            "overlay_content_coverage": row.get("overlay_profile", {}).get("baseline", {}).get("content_coverage"),
            "overlay_selected_content_occurrence_share": row.get("overlay_selection", {}).get("selected_term_content_coverage"),
            "labeled_trace_count": row["inventory"]["labeled_trace_count"],
            "unlabeled_trace_count": row["inventory"]["unlabeled_trace_count"],
            "overlay_added_trace_count": row.get("overlay_selection", {}).get("added_trace_count"),
        }
        for row in seed_rows
    ]

    result = {
        "dataset": dataset,
        "sample_count": len(samples),
        "seed_trace_counts": list(seed_trace_counts),
        "overlay_terms": overlay_terms,
        "overlay_samples": overlay_samples,
        "strategies": [strategy.__dict__ for strategy in strategies],
        "coverage_by_seed": coverage_by_seed,
        "baseline_seed": int(seed_rows[0]["seed_traces"]),
        "baseline_profile": baseline_seed_profile,
        "overlay_profile": overlay_seed_profile,
        "overlay_selection": overlay_selection_summary,
        "summary": {
            "baseline_unknown_rate": round(
                baseline_seed_profile["baseline"]["unknown"] / max(1, baseline_seed_profile["total_tokens"]),
                4,
            ),
            "baseline_content_unknown_rate": round(
                baseline_seed_profile["baseline"]["content_unknown"] / max(1, baseline_seed_profile["inventory"]["content_token_count"]),
                4,
            ),
            "normalized_unknown_rate": round(
                baseline_seed_profile["normalized"]["unknown"] / max(1, baseline_seed_profile["total_tokens"]),
                4,
            ),
            "normalized_content_unknown_rate": round(
                baseline_seed_profile["normalized"]["content_unknown"] / max(1, baseline_seed_profile["inventory"]["content_token_count"]),
                4,
            ),
            "lexical_neighborhood_unknown_rate": round(
                baseline_seed_profile["lexical_neighborhood"]["unknown"] / max(1, baseline_seed_profile["total_tokens"]),
                4,
            ),
            "lexical_neighborhood_content_unknown_rate": round(
                baseline_seed_profile["lexical_neighborhood"]["content_unknown"] / max(1, baseline_seed_profile["inventory"]["content_token_count"]),
                4,
            ),
            "overlay_unknown_rate": round(
                overlay_seed_profile["baseline"]["unknown"] / max(1, overlay_seed_profile["total_tokens"]),
                4,
            ) if overlay_seed_profile is not None else None,
            "overlay_content_unknown_rate": round(
                overlay_seed_profile["baseline"]["content_unknown"] / max(1, overlay_seed_profile["inventory"]["content_token_count"]),
                4,
            ) if overlay_seed_profile is not None else None,
            "overlay_selected_content_occurrence_share": overlay_selection_summary.get("selected_term_content_coverage")
            if overlay_selection_summary is not None
            else None,
            "seed_trace_count_changes_baseline_coverage": len(
                {row["baseline_coverage"] for row in coverage_by_seed}
            ) > 1,
        },
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result