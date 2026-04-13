"""Save and load runtime bundles for multi-process training and evaluation."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import brain_core

from brain.learning.tick_loop import TickLoop
from brain.structures.trace_store import Trace, TraceStore


_BRAIN_FILENAME = "brain_state.bin"
_TRACE_STORE_FILENAME = "trace_store.json"
_PYTHON_STATE_FILENAME = "python_state.pkl"
_METADATA_FILENAME = "metadata.json"


def _validate_brain_checkpoint_kind(brain_checkpoint_kind: str) -> None:
    if brain_checkpoint_kind not in {"full", "runtime"}:
        raise ValueError(
            f"Unsupported brain_checkpoint_kind: {brain_checkpoint_kind}"
        )


def _build_python_state(
    tick_loop: TickLoop,
    extra_python_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    python_state = {
        "tick_loop": tick_loop.export_checkpoint_state(),
    }
    if extra_python_state:
        python_state.update(extra_python_state)
    return python_state


def _build_bundle_metadata(
    trace_store: TraceStore,
    *,
    extra_metadata: dict[str, Any] | None = None,
    pending_before: int,
    pending_profile: dict[str, Any] | None,
    brain_checkpoint_kind: str,
) -> dict[str, Any]:
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tick_count": int(brain_core.get_tick_count()),
        "neuron_count": int(brain_core.get_neuron_count()),
        "synapse_count": int(brain_core.get_synapse_count()),
        "trace_count": int(len(trace_store)),
        "binding_count": int(brain_core.get_binding_count()),
        "pending_synapse_updates_before_flush": pending_before,
        "pending_synapse_updates_after_flush": int(brain_core.get_pending_synapse_update_count()),
        "brain_checkpoint_kind": brain_checkpoint_kind,
    }
    if pending_profile:
        metadata["pending_synapse_update_profile"] = pending_profile
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def _trace_rows(trace_store: TraceStore) -> list[dict[str, Any]]:
    trace_store.sync_runtime_state()
    return [trace.to_dict() for trace in trace_store.traces.values()]


def _trace_store_from_rows(trace_rows: list[dict[str, Any]]) -> TraceStore:
    trace_store = TraceStore()
    for row in trace_rows:
        trace_store.add(Trace.from_dict(dict(row)))
    return trace_store


def _brain_checkpoint_bytes(brain_checkpoint_kind: str) -> bytes:
    if brain_checkpoint_kind == "full":
        return bytes(brain_core.dump_brain_checkpoint_bytes())
    if brain_checkpoint_kind == "runtime":
        return bytes(brain_core.dump_brain_runtime_checkpoint_bytes())
    raise ValueError(
        f"Unsupported brain_checkpoint_kind: {brain_checkpoint_kind}"
    )


def build_runtime_bundle_payload(
    trace_store: TraceStore,
    tick_loop: TickLoop,
    *,
    extra_metadata: dict[str, Any] | None = None,
    extra_python_state: dict[str, Any] | None = None,
    flush_pending_synapse_updates: bool = False,
    brain_checkpoint_kind: str = "full",
) -> dict[str, Any]:
    _validate_brain_checkpoint_kind(brain_checkpoint_kind)

    pending_profile: dict[str, Any] | None = None
    pending_before = int(brain_core.get_pending_synapse_update_count())
    if flush_pending_synapse_updates and pending_before > 0:
        pending_profile = dict(brain_core.apply_synapse_updates_profiled())

    metadata = _build_bundle_metadata(
        trace_store,
        extra_metadata=extra_metadata,
        pending_before=pending_before,
        pending_profile=pending_profile,
        brain_checkpoint_kind=brain_checkpoint_kind,
    )
    python_state = _build_python_state(tick_loop, extra_python_state)
    return {
        "brain_checkpoint_kind": brain_checkpoint_kind,
        "brain_bytes": _brain_checkpoint_bytes(brain_checkpoint_kind),
        "trace_store": _trace_rows(trace_store),
        "python_state": python_state,
        "metadata": metadata,
    }


def serialize_runtime_bundle(
    trace_store: TraceStore,
    tick_loop: TickLoop,
    *,
    extra_metadata: dict[str, Any] | None = None,
    extra_python_state: dict[str, Any] | None = None,
    flush_pending_synapse_updates: bool = False,
    brain_checkpoint_kind: str = "full",
) -> tuple[bytes, dict[str, Any]]:
    payload = build_runtime_bundle_payload(
        trace_store,
        tick_loop,
        extra_metadata=extra_metadata,
        extra_python_state=extra_python_state,
        flush_pending_synapse_updates=flush_pending_synapse_updates,
        brain_checkpoint_kind=brain_checkpoint_kind,
    )
    bundle_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    return bundle_bytes, dict(payload["metadata"])


def deserialize_runtime_bundle(
    bundle_bytes: bytes,
) -> dict[str, Any]:
    return dict(pickle.loads(bundle_bytes))


def load_runtime_bundle_payload(
    payload: dict[str, Any],
) -> tuple[TraceStore, TickLoop, dict[str, Any], dict[str, Any]]:
    metadata = dict(payload.get("metadata", {}))
    trace_store = _trace_store_from_rows(list(payload.get("trace_store", [])))

    brain_checkpoint_kind = str(
        payload.get("brain_checkpoint_kind", metadata.get("brain_checkpoint_kind", "full"))
    )
    _validate_brain_checkpoint_kind(brain_checkpoint_kind)
    brain_bytes = bytes(payload.get("brain_bytes", b""))
    if brain_checkpoint_kind == "full":
        brain_core.load_brain_checkpoint_bytes(brain_bytes)
    else:
        try:
            brain_core.load_brain_runtime_checkpoint_bytes(brain_bytes)
        except Exception:
            bootstrap_bundle_dir = metadata.get("runtime_bootstrap_bundle_dir")
            if not bootstrap_bundle_dir:
                raise
            bootstrap_paths = bundle_paths(str(bootstrap_bundle_dir))
            brain_core.load_brain_checkpoint(str(bootstrap_paths["brain"]))
            brain_core.load_brain_runtime_checkpoint_bytes(brain_bytes)

    python_state = dict(payload.get("python_state", {}))
    tick_loop = TickLoop(trace_store)
    tick_state = python_state.get("tick_loop")
    if isinstance(tick_state, dict):
        tick_loop.restore_checkpoint_state(dict(tick_state))

    return trace_store, tick_loop, python_state, metadata


def load_runtime_bundle_bytes(
    bundle_bytes: bytes,
) -> tuple[TraceStore, TickLoop, dict[str, Any], dict[str, Any]]:
    return load_runtime_bundle_payload(deserialize_runtime_bundle(bundle_bytes))


def bundle_paths(path: str | Path) -> dict[str, Path]:
    bundle_dir = Path(path)
    return {
        "dir": bundle_dir,
        "brain": bundle_dir / _BRAIN_FILENAME,
        "trace_store": bundle_dir / _TRACE_STORE_FILENAME,
        "python_state": bundle_dir / _PYTHON_STATE_FILENAME,
        "metadata": bundle_dir / _METADATA_FILENAME,
    }


def save_runtime_bundle(
    trace_store: TraceStore,
    tick_loop: TickLoop,
    path: str | Path,
    *,
    extra_metadata: dict[str, Any] | None = None,
    extra_python_state: dict[str, Any] | None = None,
    flush_pending_synapse_updates: bool = False,
    brain_checkpoint_kind: str = "full",
) -> dict[str, Any]:
    paths = bundle_paths(path)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    _validate_brain_checkpoint_kind(brain_checkpoint_kind)

    pending_profile: dict[str, Any] | None = None
    pending_before = int(brain_core.get_pending_synapse_update_count())
    if flush_pending_synapse_updates and pending_before > 0:
        pending_profile = dict(brain_core.apply_synapse_updates_profiled())

    if brain_checkpoint_kind == "full":
        brain_core.save_brain_checkpoint(str(paths["brain"]))
    else:
        brain_core.save_brain_runtime_checkpoint(str(paths["brain"]))
    trace_store.save(str(paths["trace_store"]))

    python_state = _build_python_state(tick_loop, extra_python_state)
    with open(paths["python_state"], "wb") as handle:
        pickle.dump(python_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = _build_bundle_metadata(
        trace_store,
        extra_metadata=extra_metadata,
        pending_before=pending_before,
        pending_profile=pending_profile,
        brain_checkpoint_kind=brain_checkpoint_kind,
    )

    with open(paths["metadata"], "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    return metadata


def load_runtime_bundle(
    path: str | Path,
) -> tuple[TraceStore, TickLoop, dict[str, Any], dict[str, Any]]:
    paths = bundle_paths(path)

    metadata: dict[str, Any] = {}
    if paths["metadata"].exists():
        with open(paths["metadata"], "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    trace_store = TraceStore()
    trace_store.load(str(paths["trace_store"]))
    brain_checkpoint_kind = str(metadata.get("brain_checkpoint_kind", "full"))
    if brain_checkpoint_kind == "full":
        brain_core.load_brain_checkpoint(str(paths["brain"]))
    elif brain_checkpoint_kind == "runtime":
        try:
            brain_core.load_brain_runtime_checkpoint(str(paths["brain"]))
        except Exception:
            bootstrap_bundle_dir = metadata.get("runtime_bootstrap_bundle_dir")
            if not bootstrap_bundle_dir:
                raise
            bootstrap_paths = bundle_paths(str(bootstrap_bundle_dir))
            brain_core.load_brain_checkpoint(str(bootstrap_paths["brain"]))
            brain_core.load_brain_runtime_checkpoint(str(paths["brain"]))
    else:
        raise ValueError(
            f"Unsupported brain_checkpoint_kind in metadata: {brain_checkpoint_kind}"
        )

    python_state: dict[str, Any] = {}
    if paths["python_state"].exists():
        with open(paths["python_state"], "rb") as handle:
            python_state = pickle.load(handle)

    tick_loop = TickLoop(trace_store)
    tick_state = python_state.get("tick_loop")
    if isinstance(tick_state, dict):
        tick_loop.restore_checkpoint_state(dict(tick_state))

    return trace_store, tick_loop, python_state, metadata


def load_runtime_bundle_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = bundle_paths(path)["metadata"]
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)