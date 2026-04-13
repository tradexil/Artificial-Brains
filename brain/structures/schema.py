"""Level 5 Schemas — causal sequence structures above traces and bindings.

A Schema captures a temporal causal chain of traces:
  trace_A (delay d1) → trace_B (delay d2) → trace_C

Schemas enable prediction: when trace_A fires, the schema predicts
trace_B will fire within d1 ticks. If it does → confirmation (learning
consolidation). If not → surprise signal (increased novelty/learning).

Hierarchy:
  Neurons → Synapses → Bindings → Traces → **Schemas**
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CausalEdge:
    """A directed temporal link between two traces in a schema."""
    from_trace: str          # source trace ID
    to_trace: str            # target trace ID
    delay_ticks: int         # expected delay (ticks between firings)
    strength: float = 0.5    # confidence in this edge (0–1)
    observations: int = 0    # how many times this transition was observed
    prediction_hits: int = 0   # confirmed predictions
    prediction_misses: int = 0  # missed predictions


@dataclass
class Schema:
    """A causal sequence structure linking traces in temporal order."""
    id: str
    label: str | None = None
    traces: list[str] = field(default_factory=list)         # ordered trace IDs
    causal_edges: list[CausalEdge] = field(default_factory=list)
    polarity: float = 0.0     # overall valence (-1 to +1)
    novelty: float = 1.0      # decays as schema becomes familiar
    strength: float = 0.1     # accumulated evidence
    fire_count: int = 0       # number of full-chain activations
    formation_tick: int = 0   # when schema was created
    last_activated: int = 0   # last tick the schema was triggered

    @property
    def edge_map(self) -> dict[str, CausalEdge]:
        """Map from source trace_id to outgoing edge."""
        return {e.from_trace: e for e in self.causal_edges}

    def next_trace(self, current_trace_id: str) -> tuple[str, int] | None:
        """Given a current trace, return (next_trace_id, expected_delay) or None."""
        for edge in self.causal_edges:
            if edge.from_trace == current_trace_id:
                return edge.to_trace, edge.delay_ticks
        return None

    def trace_index(self, trace_id: str) -> int | None:
        """Return position of trace in the sequence, or None."""
        try:
            return self.traces.index(trace_id)
        except ValueError:
            return None


@dataclass
class SchemaStore:
    """Collection of schemas with lookup helpers."""
    schemas: dict[str, Schema] = field(default_factory=dict)
    # Inverted index: trace_id → set of schema IDs containing that trace
    _trace_to_schemas: dict[str, set[str]] = field(default_factory=dict)

    def add(self, schema: Schema) -> None:
        self.schemas[schema.id] = schema
        for tid in schema.traces:
            self._trace_to_schemas.setdefault(tid, set()).add(schema.id)

    def get(self, schema_id: str) -> Schema | None:
        return self.schemas.get(schema_id)

    def remove(self, schema_id: str) -> Schema | None:
        schema = self.schemas.pop(schema_id, None)
        if schema is not None:
            for tid in schema.traces:
                s = self._trace_to_schemas.get(tid)
                if s is not None:
                    s.discard(schema_id)
        return schema

    def schemas_for_trace(self, trace_id: str) -> list[Schema]:
        """Return all schemas that contain the given trace."""
        schema_ids = self._trace_to_schemas.get(trace_id, set())
        return [self.schemas[sid] for sid in schema_ids if sid in self.schemas]

    def __len__(self) -> int:
        return len(self.schemas)

    def __iter__(self):
        return iter(self.schemas.values())
