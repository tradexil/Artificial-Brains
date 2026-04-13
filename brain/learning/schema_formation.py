"""Schema formation: detect repeated causal trace sequences and form schemas.

Monitors trace firing order across ticks. When a sequence of traces fires
in the same causal order N times (SCHEMA_FORMATION_COUNT), a schema is
formed capturing that temporal pattern.

Schema prediction: when a schema's first trace fires, predict the next
trace within the expected delay window. Hits strengthen the schema;
misses generate a surprise signal that feeds back into novelty.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from brain.structures.schema import CausalEdge, Schema, SchemaStore
from brain.utils.config import (
    SCHEMA_CAUSAL_WINDOW_TICKS,
    SCHEMA_FORMATION_COUNT,
    SCHEMA_MAX_SEQUENCE_LENGTH,
    SCHEMA_MIN_SEQUENCE_LENGTH,
    SCHEMA_PREDICTION_BONUS,
    SCHEMA_SURPRISE_PENALTY,
)


@dataclass
class _PendingPrediction:
    """A prediction made by a schema that awaits confirmation."""
    schema_id: str
    expected_trace: str
    deadline_tick: int  # must fire by this tick
    edge: CausalEdge


class SchemaFormationEngine:
    """Detects causal trace sequences and forms schemas.

    Usage:
        engine = SchemaFormationEngine(schema_store)
        # Per tick:
        surprise = engine.step(active_traces, tick)
    """

    def __init__(self, schema_store: SchemaStore):
        self.schema_store = schema_store

        # Recent trace firing history: [(trace_id, tick), ...]
        self._firing_history: list[tuple[str, int]] = []
        self._max_history = 200

        # Candidate sequences: tuple of trace_ids → observation count
        self._candidates: dict[tuple[str, ...], int] = defaultdict(int)
        # Track the ticks at which candidates were last seen
        self._candidate_last_tick: dict[tuple[str, ...], int] = {}

        # Active predictions awaiting confirmation
        self._pending: list[_PendingPrediction] = []

        # Existing schema sequence keys (to avoid duplicates)
        self._known_sequences: set[tuple[str, ...]] = set()
        for schema in schema_store:
            self._known_sequences.add(tuple(schema.traces))

        # Stats from last step
        self.last_step_debug: dict[str, object] = {}

    def step(
        self,
        active_traces: list[tuple[str, float]],
        tick: int,
    ) -> float:
        """Process one tick of schema formation and prediction.

        Args:
            active_traces: [(trace_id, activation_score), ...] currently active
            tick: current tick number

        Returns:
            surprise_signal: float in [0, 1]. 0 = all predictions confirmed,
            >0 = missed predictions (feeds into novelty calculation).
        """
        active_ids = {tid for tid, score in active_traces if score >= 0.5}
        debug: dict[str, object] = {
            "active_trace_count": len(active_ids),
            "pending_predictions": len(self._pending),
            "schemas_formed": 0,
            "predictions_hit": 0,
            "predictions_missed": 0,
            "surprise_signal": 0.0,
        }

        # 1. Record firings
        for tid in active_ids:
            self._firing_history.append((tid, tick))
        # Trim history
        if len(self._firing_history) > self._max_history:
            self._firing_history = self._firing_history[-self._max_history:]

        # 2. Check pending predictions
        surprise_signal = self._check_predictions(active_ids, tick, debug)

        # 3. Make new predictions from active schemas
        self._make_predictions(active_ids, tick)

        # 4. Detect candidate sequences in recent history
        self._detect_sequences(tick)

        # 5. Promote candidates to schemas
        schemas_formed = self._promote_candidates(tick)
        debug["schemas_formed"] = schemas_formed

        debug["surprise_signal"] = surprise_signal
        debug["total_schemas"] = len(self.schema_store)
        debug["candidate_count"] = len(self._candidates)
        self.last_step_debug = debug
        return surprise_signal

    def _check_predictions(
        self,
        active_ids: set[str],
        tick: int,
        debug: dict,
    ) -> float:
        """Check pending predictions against current activations.

        Returns surprise signal [0, 1].
        """
        still_pending: list[_PendingPrediction] = []
        hits = 0
        misses = 0

        for pred in self._pending:
            if pred.expected_trace in active_ids:
                # Prediction confirmed!
                hits += 1
                pred.edge.prediction_hits += 1
                pred.edge.strength = min(1.0, pred.edge.strength + SCHEMA_PREDICTION_BONUS * 0.1)
                schema = self.schema_store.get(pred.schema_id)
                if schema is not None:
                    schema.strength = min(1.0, schema.strength + SCHEMA_PREDICTION_BONUS * 0.05)
                    schema.fire_count += 1
                    schema.last_activated = tick
            elif tick >= pred.deadline_tick:
                # Prediction expired — surprise!
                misses += 1
                pred.edge.prediction_misses += 1
                pred.edge.strength = max(0.0, pred.edge.strength - SCHEMA_SURPRISE_PENALTY * 0.1)
            else:
                still_pending.append(pred)

        self._pending = still_pending
        debug["predictions_hit"] = hits
        debug["predictions_missed"] = misses

        if hits + misses == 0:
            return 0.0
        return min(1.0, misses / max(hits + misses, 1) * SCHEMA_SURPRISE_PENALTY)

    def _make_predictions(self, active_ids: set[str], tick: int) -> None:
        """For each active trace that starts a schema, predict next trace."""
        for tid in active_ids:
            for schema in self.schema_store.schemas_for_trace(tid):
                result = schema.next_trace(tid)
                if result is None:
                    continue
                next_tid, delay = result
                # Don't duplicate predictions
                already_pending = any(
                    p.schema_id == schema.id and p.expected_trace == next_tid
                    for p in self._pending
                )
                if already_pending:
                    continue
                edge = schema.edge_map.get(tid)
                if edge is None:
                    continue
                self._pending.append(_PendingPrediction(
                    schema_id=schema.id,
                    expected_trace=next_tid,
                    deadline_tick=tick + delay + 2,  # small grace window
                    edge=edge,
                ))

    def _detect_sequences(self, tick: int) -> None:
        """Scan recent history for repeated causal subsequences."""
        if len(self._firing_history) < SCHEMA_MIN_SEQUENCE_LENGTH:
            return

        # Extract unique trace IDs in temporal order (deduplicated per tick)
        seen_per_tick: dict[int, set[str]] = defaultdict(set)
        ordered: list[tuple[str, int]] = []
        for tid, t in self._firing_history:
            if tid not in seen_per_tick[t]:
                seen_per_tick[t].add(tid)
                ordered.append((tid, t))

        # Look for subsequences of length 2..MAX in the recent window
        n = len(ordered)
        for length in range(SCHEMA_MIN_SEQUENCE_LENGTH, min(SCHEMA_MAX_SEQUENCE_LENGTH + 1, n + 1)):
            for start in range(max(0, n - 50), n - length + 1):
                seq_items = ordered[start:start + length]
                # Check causality: each step within SCHEMA_CAUSAL_WINDOW_TICKS
                valid = True
                for i in range(len(seq_items) - 1):
                    delta = seq_items[i + 1][1] - seq_items[i][1]
                    if delta <= 0 or delta > SCHEMA_CAUSAL_WINDOW_TICKS:
                        valid = False
                        break
                if not valid:
                    continue

                seq_key = tuple(tid for tid, _ in seq_items)
                if seq_key in self._known_sequences:
                    continue
                # Don't count sequences with duplicate traces
                if len(set(seq_key)) < len(seq_key):
                    continue

                self._candidates[seq_key] += 1
                self._candidate_last_tick[seq_key] = tick

    def _promote_candidates(self, tick: int) -> int:
        """Promote candidates that have been observed enough times."""
        formed = 0
        to_remove: list[tuple[str, ...]] = []

        for seq_key, count in self._candidates.items():
            if count < SCHEMA_FORMATION_COUNT:
                continue
            if seq_key in self._known_sequences:
                to_remove.append(seq_key)
                continue

            # Build schema
            schema_id = f"schema_{uuid.uuid4().hex[:8]}"
            traces = list(seq_key)
            edges: list[CausalEdge] = []

            # Estimate delays from history
            for i in range(len(traces) - 1):
                from_tid = traces[i]
                to_tid = traces[i + 1]
                delay = self._estimate_delay(from_tid, to_tid)
                edges.append(CausalEdge(
                    from_trace=from_tid,
                    to_trace=to_tid,
                    delay_ticks=delay,
                    strength=0.5,
                    observations=count,
                ))

            schema = Schema(
                id=schema_id,
                label=f"seq_{len(traces)}",
                traces=traces,
                causal_edges=edges,
                strength=min(1.0, count * 0.1),
                formation_tick=tick,
            )
            self.schema_store.add(schema)
            self._known_sequences.add(seq_key)
            to_remove.append(seq_key)
            formed += 1

        for key in to_remove:
            self._candidates.pop(key, None)
            self._candidate_last_tick.pop(key, None)

        # Clean old candidates (not seen in 200 ticks)
        stale = [
            key for key, last in self._candidate_last_tick.items()
            if tick - last > 200
        ]
        for key in stale:
            self._candidates.pop(key, None)
            self._candidate_last_tick.pop(key, None)

        return formed

    def _estimate_delay(self, from_tid: str, to_tid: str) -> int:
        """Estimate typical delay between two traces from firing history."""
        delays: list[int] = []
        last_from_tick: int | None = None

        for tid, t in self._firing_history:
            if tid == from_tid:
                last_from_tick = t
            elif tid == to_tid and last_from_tick is not None:
                delta = t - last_from_tick
                if 0 < delta <= SCHEMA_CAUSAL_WINDOW_TICKS:
                    delays.append(delta)
                last_from_tick = None

        if not delays:
            return 3  # default delay
        return max(1, sum(delays) // len(delays))

    @property
    def recently_formed(self) -> list[str]:
        """Schema IDs formed in the last step."""
        return [
            sid for sid in self.schema_store.schemas
            if self.schema_store.schemas[sid].formation_tick == self.last_step_debug.get("formation_tick", -1)
        ]
