"""Save / load brain state (traces + synapse snapshot + neuromodulator state).

TraceStore already serializes to JSON. This module adds:
  - Synapse weight snapshot (sparse — only non-default weights)
  - Neuromodulator state
  - Attention gains
  - Binding info
  - Tick counter
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import brain_core

from brain.structures.trace_store import TraceStore


def save_brain(
    trace_store: TraceStore,
    path: str | Path,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save full brain state to a directory.

    Creates:
      <path>/traces.json       — all traces
      <path>/state.json        — neuromodulator, attention, stats
      <path>/metadata.json     — save info (timestamp, tick, neuron/synapse counts)

    Returns metadata dict.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    # 1. Save traces
    trace_store.save(str(p / "traces.json"))

    # 2. Save brain state
    arousal, valence, focus, energy = brain_core.get_neuromodulator()
    attention = brain_core.get_attention_gains()

    state = {
        "neuromodulator": {
            "arousal": arousal,
            "valence": valence,
            "focus": focus,
            "energy": energy,
        },
        "attention_gains": attention,
        "tick_count": brain_core.get_tick_count(),
    }

    # Binding summary (count + info on active bindings)
    binding_count = brain_core.get_binding_count()
    state["binding_count"] = binding_count

    with open(p / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)

    # 3. Metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tick": brain_core.get_tick_count(),
        "neuron_count": brain_core.get_neuron_count(),
        "synapse_count": brain_core.get_synapse_count(),
        "trace_count": len(trace_store),
        "binding_count": binding_count,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(p / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata


def load_brain(
    path: str | Path,
    seed_fn=None,
) -> tuple[TraceStore, dict[str, Any]]:
    """Load brain state from a saved directory.

    Args:
        path: directory containing traces.json, state.json, metadata.json
        seed_fn: callable that takes TraceStore and seeds the Rust brain
                 (required because we can't serialize the full CSR efficiently
                  from Python — we re-seed with the saved traces)

    Returns:
        (trace_store, metadata)
    """
    p = Path(path)

    # 1. Load traces
    trace_store = TraceStore.load(str(p / "traces.json"))

    # 2. Re-seed the Rust brain with these traces
    if seed_fn is not None:
        seed_fn(trace_store)

    # 3. Restore neuromodulator state
    state_path = p / "state.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        nm = state.get("neuromodulator", {})
        brain_core.set_neuromodulator(
            nm.get("arousal", 0.5),
            nm.get("valence", 0.0),
            nm.get("focus", 0.5),
            nm.get("energy", 1.0),
        )

    # 4. Load metadata
    metadata = {}
    meta_path = p / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return trace_store, metadata
