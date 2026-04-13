"""Minimal World Model — simulation mode with divergence tracking.

Provides a boundary between real and imagined activation:
- Simulation mode: suppress real input, fire schemas from internal state
- Divergence tracking: compare predicted vs actual state per tick
- Planning: run schemas in simulation, evaluate outcome

This is the first iteration — intentionally minimal and extensible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import brain_core

from brain.structures.schema import SchemaStore


@dataclass
class DivergenceEntry:
    """Log entry for a single tick's prediction vs actual mismatch."""
    tick: int
    predicted_traces: list[str]
    actual_traces: list[str]
    overlap_count: int
    divergence_score: float  # 0 = perfect prediction, 1 = complete miss
    schema_id: str | None = None


@dataclass
class WorldModel:
    """Minimal world model for internal simulation and divergence tracking."""

    simulation_mode: bool = False
    active_schema_ids: list[str] = field(default_factory=list)
    prediction_horizon: int = 12
    confidence: float = 0.5
    divergence_log: list[DivergenceEntry] = field(default_factory=list)
    _max_log_size: int = 500

    # Running predictions: trace_id → expected_by_tick
    _predictions: dict[str, int] = field(default_factory=dict)
    _prediction_source: dict[str, str] = field(default_factory=dict)  # trace_id → schema_id

    # Stats
    total_predictions: int = 0
    total_hits: int = 0
    total_misses: int = 0

    def enter_simulation(self, schema_store: SchemaStore, schema_ids: list[str] | None = None) -> None:
        """Enter simulation mode — suppress external input, run from schemas."""
        self.simulation_mode = True
        if schema_ids is not None:
            self.active_schema_ids = list(schema_ids)
        else:
            # Use all available schemas
            self.active_schema_ids = list(schema_store.schemas.keys())

    def exit_simulation(self) -> None:
        """Exit simulation mode — resume normal input processing."""
        self.simulation_mode = False
        self.active_schema_ids = []
        self._predictions.clear()
        self._prediction_source.clear()

    def predict_from_schemas(
        self,
        active_traces: list[tuple[str, float]],
        tick: int,
        schema_store: SchemaStore,
    ) -> list[str]:
        """Generate predictions based on active schemas and current traces.

        Returns list of predicted trace IDs.
        """
        active_ids = {tid for tid, score in active_traces if score >= 0.4}
        predicted: list[str] = []

        for tid in active_ids:
            for schema in schema_store.schemas_for_trace(tid):
                if self.active_schema_ids and schema.id not in self.active_schema_ids:
                    continue
                result = schema.next_trace(tid)
                if result is None:
                    continue
                next_tid, delay = result
                deadline = tick + delay + 2
                if next_tid not in self._predictions:
                    self._predictions[next_tid] = deadline
                    self._prediction_source[next_tid] = schema.id
                    predicted.append(next_tid)
                    self.total_predictions += 1

        return predicted

    def check_divergence(
        self,
        active_traces: list[tuple[str, float]],
        tick: int,
    ) -> float:
        """Compare predictions against actual activations, log divergence.

        Returns divergence score for this tick (0 = perfect, 1 = all missed).
        """
        actual_ids = {tid for tid, score in active_traces if score >= 0.4}

        # Check expired predictions
        expired: list[str] = []
        confirmed: list[str] = []
        still_pending: dict[str, int] = {}

        for pred_tid, deadline in self._predictions.items():
            if pred_tid in actual_ids:
                confirmed.append(pred_tid)
                self.total_hits += 1
            elif tick >= deadline:
                expired.append(pred_tid)
                self.total_misses += 1
            else:
                still_pending[pred_tid] = deadline

        # Clean up
        for tid in confirmed + expired:
            self._prediction_source.pop(tid, None)
        self._predictions = still_pending

        # Compute divergence
        total = len(confirmed) + len(expired)
        if total == 0:
            return 0.0

        divergence = len(expired) / total

        # Log
        schema_id = None
        if expired:
            schema_id = self._prediction_source.get(expired[0])

        entry = DivergenceEntry(
            tick=tick,
            predicted_traces=expired,
            actual_traces=list(actual_ids),
            overlap_count=len(confirmed),
            divergence_score=divergence,
            schema_id=schema_id,
        )
        self.divergence_log.append(entry)
        if len(self.divergence_log) > self._max_log_size:
            self.divergence_log = self.divergence_log[-self._max_log_size:]

        # Update confidence
        self.confidence = max(0.0, min(1.0,
            self.confidence + (0.05 if divergence < 0.3 else -0.05)
        ))

        return divergence

    def simulate_schema_chain(
        self,
        schema_store: SchemaStore,
        schema_id: str,
        tick_loop,
        max_ticks: int | None = None,
    ) -> dict:
        """Run a schema in simulation mode and return outcome.

        Suppresses external input, fires the schema's first trace,
        and ticks forward, checking if the rest of the chain activates.

        Returns dict with simulation results.
        """
        schema = schema_store.get(schema_id)
        if schema is None:
            return {"error": f"Schema {schema_id!r} not found", "ticks_run": 0}

        max_ticks = max_ticks or self.prediction_horizon
        self.enter_simulation(schema_store, [schema_id])

        ticks_run = 0
        activated_traces: list[str] = []
        divergence_scores: list[float] = []

        try:
            for _ in range(max_ticks):
                result = tick_loop.step(learn=False)
                ticks_run += 1
                active = result.get("active_traces", [])
                active_ids = [tid for tid, _ in active]
                activated_traces.extend(active_ids)

                div = self.check_divergence(active, tick_loop.last_tick_number)
                divergence_scores.append(div)

                # Check if full chain activated
                chain_complete = all(
                    tid in activated_traces for tid in schema.traces
                )
                if chain_complete:
                    break
        finally:
            self.exit_simulation()

        chain_coverage = sum(
            1 for tid in schema.traces if tid in activated_traces
        ) / max(len(schema.traces), 1)

        return {
            "schema_id": schema_id,
            "ticks_run": ticks_run,
            "chain_coverage": chain_coverage,
            "activated_traces": list(set(activated_traces)),
            "mean_divergence": (
                sum(divergence_scores) / len(divergence_scores)
                if divergence_scores else 0.0
            ),
            "confidence": self.confidence,
        }

    def get_summary(self) -> dict:
        """Return a summary of world model state."""
        return {
            "simulation_mode": self.simulation_mode,
            "active_schemas": list(self.active_schema_ids),
            "prediction_horizon": self.prediction_horizon,
            "confidence": self.confidence,
            "total_predictions": self.total_predictions,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "divergence_log_size": len(self.divergence_log),
            "recent_divergence": (
                [e.divergence_score for e in self.divergence_log[-10:]]
                if self.divergence_log else []
            ),
        }
