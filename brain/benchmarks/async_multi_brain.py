"""Process-based async multi-brain training for text workloads."""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import queue
import struct
import time
from collections import defaultdict
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import brain_core
import numpy as np

from brain.input.text_input import TextInput
from brain.metrics.collector import MetricsCollector
from brain.seed.seed_runner import seed_brain, seed_brain_fast
from brain.serialize.runtime_bundle import (
    bundle_paths,
    load_runtime_bundle_bytes,
    save_runtime_bundle,
    serialize_runtime_bundle,
)
from brain.structures.trace_store import Trace, TraceStore
from brain.learning.tick_loop import TickLoop
from brain.utils.config import TRACE_MERGE_OVERLAP


DEFAULT_ASYNC_MULTI_BRAIN_WORKER_COUNT = 4
DEFAULT_ASYNC_MULTI_BRAIN_CORES_PER_WORKER = 2
DEFAULT_ASYNC_MULTI_BRAIN_MERGE_EVERY_SAMPLES = 5
DEFAULT_ASYNC_MULTI_BRAIN_LEADER_WORKER_ID = 0
DEFAULT_ASYNC_MULTI_BRAIN_TICK_BATCH_SIZE = 2
_ASYNC_MULTI_BRAIN_TRACE_BINDING_SYNC_INTERVAL = 10
_CHECKPOINT2_BASELINE_PATH = Path("results") / "checkpoint2_multirate_ag_news_50x5.json"
_WEIGHT_DTYPE = np.float32
_SYNAPSE_DELTA_DTYPE = np.dtype(
    [("synapse_index", np.uint32), ("delta", np.float32)]
)


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _copy_trace(trace: Trace) -> Trace:
    return Trace.from_dict(trace.to_dict())


def _flatten_trace_neurons(trace: Trace) -> set[int]:
    neurons: set[int] = set()
    for neuron_ids in trace.neurons.values():
        neurons.update(int(neuron_id) for neuron_id in neuron_ids)
    return neurons


def _trace_overlap_ratio(left: Trace, right: Trace) -> float:
    left_neurons = _flatten_trace_neurons(left)
    right_neurons = _flatten_trace_neurons(right)
    if not left_neurons or not right_neurons:
        return 0.0
    overlap = len(left_neurons & right_neurons)
    union = len(left_neurons | right_neurons)
    if union <= 0:
        return 0.0
    return overlap / union


def _merge_trace_into(into_trace: Trace, from_trace: Trace) -> None:
    for region_name, neuron_ids in from_trace.neurons.items():
        merged_ids = set(into_trace.neurons.get(region_name, []))
        merged_ids.update(int(neuron_id) for neuron_id in neuron_ids)
        if merged_ids:
            into_trace.neurons[region_name] = sorted(merged_ids)

    into_trace.binding_ids = sorted(
        set(into_trace.binding_ids) | set(from_trace.binding_ids)
    )
    into_trace.strength = max(float(into_trace.strength), float(from_trace.strength))
    into_trace.decay = max(float(into_trace.decay), float(from_trace.decay))
    into_trace.polarity = _round(
        (float(into_trace.polarity) + float(from_trace.polarity)) / 2.0,
        6,
    )
    into_trace.abstraction = max(
        float(into_trace.abstraction),
        float(from_trace.abstraction),
    )
    into_trace.novelty = min(float(into_trace.novelty), float(from_trace.novelty))
    into_trace.co_traces = sorted(set(into_trace.co_traces) | set(from_trace.co_traces))
    into_trace.context_tags = sorted(
        set(into_trace.context_tags) | set(from_trace.context_tags)
    )
    into_trace.fire_count += int(from_trace.fire_count)
    into_trace.last_fired = max(int(into_trace.last_fired), int(from_trace.last_fired))
    if int(into_trace.formation_tick) <= 0:
        into_trace.formation_tick = int(from_trace.formation_tick)
    elif int(from_trace.formation_tick) > 0:
        into_trace.formation_tick = min(
            int(into_trace.formation_tick),
            int(from_trace.formation_tick),
        )


def _merge_worker_trace_iterable(
    merged_store: TraceStore,
    worker_traces: Any,
) -> dict[str, int]:
    added = 0
    deduped = 0
    skipped_existing_ids = 0

    for worker_trace in worker_traces:
        if worker_trace.id in merged_store:
            skipped_existing_ids += 1
            continue

        worker_neurons = _flatten_trace_neurons(worker_trace)
        candidate_ids: set[str] = set()
        for neuron_id in worker_neurons:
            candidate_ids.update(merged_store.traces_for_neuron(neuron_id))

        best_candidate_id = None
        best_ratio = 0.0
        for candidate_id in candidate_ids:
            candidate_trace = merged_store.get(candidate_id)
            if candidate_trace is None:
                continue
            ratio = _trace_overlap_ratio(candidate_trace, worker_trace)
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate_id = candidate_id

        if best_candidate_id is not None and best_ratio >= TRACE_MERGE_OVERLAP:
            merged_trace = merged_store.get(best_candidate_id)
            if merged_trace is not None:
                _merge_trace_into(merged_trace, worker_trace)
                merged_store.sync_trace(merged_trace.id)
                deduped += 1
                continue

        merged_store.add(_copy_trace(worker_trace))
        added += 1

    return {
        "traces_added": added,
        "traces_deduped": deduped,
        "traces_skipped_existing_id": skipped_existing_ids,
    }


def _merge_worker_traces(
    merged_store: TraceStore,
    worker_store: TraceStore,
) -> dict[str, int]:
    return _merge_worker_trace_iterable(merged_store, worker_store.traces.values())


def _merge_worker_trace_rows(
    merged_store: TraceStore,
    worker_trace_rows: list[dict[str, Any]],
) -> dict[str, int]:
    worker_traces = (Trace.from_dict(dict(row)) for row in worker_trace_rows)
    return _merge_worker_trace_iterable(merged_store, worker_traces)


def _merge_worker_trace_iterable_incremental(
    merged_store: TraceStore,
    worker_traces: Any,
) -> dict[str, int]:
    added = 0
    deduped = 0
    merged_existing_ids = 0
    added_trace_ids: list[str] = []
    updated_trace_ids: set[str] = set()
    dropped_trace_ids: set[str] = set()
    trace_id_redirects: dict[str, str] = {}

    for worker_trace in worker_traces:
        existing_trace = merged_store.get(worker_trace.id)
        if existing_trace is not None:
            _merge_trace_into(existing_trace, worker_trace)
            merged_store.sync_trace(existing_trace.id)
            updated_trace_ids.add(existing_trace.id)
            merged_existing_ids += 1
            continue

        worker_neurons = _flatten_trace_neurons(worker_trace)
        candidate_ids: set[str] = set()
        for neuron_id in worker_neurons:
            candidate_ids.update(merged_store.traces_for_neuron(neuron_id))

        best_candidate_id = None
        best_ratio = 0.0
        for candidate_id in candidate_ids:
            candidate_trace = merged_store.get(candidate_id)
            if candidate_trace is None:
                continue
            ratio = _trace_overlap_ratio(candidate_trace, worker_trace)
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate_id = candidate_id

        if best_candidate_id is not None and best_ratio >= TRACE_MERGE_OVERLAP:
            merged_trace = merged_store.get(best_candidate_id)
            if merged_trace is not None:
                _merge_trace_into(merged_trace, worker_trace)
                merged_store.sync_trace(merged_trace.id)
                updated_trace_ids.add(merged_trace.id)
                dropped_trace_ids.add(worker_trace.id)
                trace_id_redirects[str(worker_trace.id)] = str(merged_trace.id)
                deduped += 1
                continue

        merged_store.add(_copy_trace(worker_trace))
        added_trace_ids.append(str(worker_trace.id))
        added += 1

    return {
        "traces_added": added,
        "traces_deduped": deduped,
        "traces_merged_existing_id": merged_existing_ids,
        "added_trace_ids": sorted(added_trace_ids),
        "updated_trace_ids": sorted(updated_trace_ids),
        "dropped_trace_ids": sorted(dropped_trace_ids),
        "trace_id_redirects": dict(sorted(trace_id_redirects.items())),
    }


def _merge_worker_trace_rows_incremental(
    merged_store: TraceStore,
    worker_trace_rows: list[dict[str, Any]],
) -> dict[str, int]:
    worker_traces = (Trace.from_dict(dict(row)) for row in worker_trace_rows)
    return _merge_worker_trace_iterable_incremental(merged_store, worker_traces)


def _trace_runtime_cache_entry(trace: Trace) -> dict[str, Any]:
    return {
        "strength": float(trace.strength),
        "decay": float(trace.decay),
        "novelty": float(trace.novelty),
        "polarity": float(trace.polarity),
        "fire_count": int(trace.fire_count),
        "last_fired": int(trace.last_fired),
    }


def _trace_runtime_cache_from_store(
    trace_store: TraceStore,
) -> dict[str, dict[str, Any]]:
    trace_store.sync_runtime_state()
    return {
        trace_id: _trace_runtime_cache_entry(trace)
        for trace_id, trace in trace_store.traces.items()
    }


def _trace_runtime_update_needs_sync(
    previous: dict[str, Any],
    current: dict[str, Any],
    fire_count_delta: int,
) -> bool:
    if fire_count_delta > 0:
        return True
    if int(current["last_fired"]) > int(previous["last_fired"]):
        return True

    # Decay snapshots include deterministic freshness drift from current_tick.
    # If we sync on decay alone, every inactive trace looks dirty on slow-sync
    # rounds even when no worker actually changed its runtime state.
    for key in ("strength", "novelty", "polarity"):
        if abs(float(current[key]) - float(previous[key])) > 1e-9:
            return True
    return False


def _build_worker_trace_delta(
    trace_store: TraceStore,
    *,
    new_trace_ids: set[str] | None = None,
    touched_trace_ids: set[str] | None = None,
    last_synced_trace_runtime: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_trace_ids = set(new_trace_ids or set()) | set(touched_trace_ids or set())
    if candidate_trace_ids:
        sync_ids = sorted(
            trace_id
            for trace_id in candidate_trace_ids
            if trace_store.get(trace_id) is not None
        )
    else:
        sync_ids = sorted(trace_store.traces)
    if sync_ids:
        trace_store.sync_runtime_state(sync_ids)

    known_new_trace_ids = {
        trace_id
        for trace_id in set(new_trace_ids or set())
        if trace_store.get(trace_id) is not None
    }
    new_traces = [
        trace_store.get(trace_id).to_dict()
        for trace_id in sorted(known_new_trace_ids)
        if trace_store.get(trace_id) is not None
    ]

    runtime_updates: list[dict[str, Any]] = []
    for trace_id in sorted(set(sync_ids) - known_new_trace_ids):
        trace = trace_store.get(trace_id)
        if trace is None:
            continue

        previous = last_synced_trace_runtime.get(trace_id)
        if previous is None:
            new_traces.append(trace.to_dict())
            continue

        current = _trace_runtime_cache_entry(trace)
        fire_count_delta = max(
            0,
            int(current["fire_count"]) - int(previous["fire_count"]),
        )
        if not _trace_runtime_update_needs_sync(previous, current, fire_count_delta):
            continue

        runtime_updates.append(
            {
                "id": str(trace_id),
                "strength": float(current["strength"]),
                "decay": float(current["decay"]),
                "novelty": float(current["novelty"]),
                "polarity": float(current["polarity"]),
                "fire_count_delta": int(fire_count_delta),
                "last_fired": int(current["last_fired"]),
            }
        )

    return {
        "new_traces": new_traces,
        "runtime_updates": runtime_updates,
    }


def _aggregate_trace_runtime_updates(
    worker_update_groups: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}

    for update_group in worker_update_groups:
        for update in update_group:
            trace_id = str(update["id"])
            entry = aggregated.setdefault(
                trace_id,
                {
                    "id": trace_id,
                    "strength": float(update["strength"]),
                    "decay": float(update["decay"]),
                    "novelty": float(update["novelty"]),
                    "polarity_sum": 0.0,
                    "polarity_count": 0,
                    "fire_count_delta": 0,
                    "last_fired": 0,
                },
            )
            entry["strength"] = max(float(entry["strength"]), float(update["strength"]))
            entry["decay"] = max(float(entry["decay"]), float(update["decay"]))
            entry["novelty"] = min(float(entry["novelty"]), float(update["novelty"]))
            entry["polarity_sum"] += float(update["polarity"])
            entry["polarity_count"] += 1
            entry["fire_count_delta"] += int(update.get("fire_count_delta", 0) or 0)
            entry["last_fired"] = max(
                int(entry["last_fired"]),
                int(update.get("last_fired", 0) or 0),
            )

    merged_updates: list[dict[str, Any]] = []
    for trace_id in sorted(aggregated):
        entry = aggregated[trace_id]
        merged_updates.append(
            {
                "id": trace_id,
                "strength": float(entry["strength"]),
                "decay": float(entry["decay"]),
                "novelty": float(entry["novelty"]),
                "polarity": (
                    float(entry["polarity_sum"]) / max(int(entry["polarity_count"]), 1)
                ),
                "fire_count_delta": int(entry["fire_count_delta"]),
                "last_fired": int(entry["last_fired"]),
            }
        )
    return merged_updates


def _apply_trace_runtime_update(trace: Trace, update: dict[str, Any]) -> None:
    trace.strength = max(float(trace.strength), float(update["strength"]))
    trace.decay = max(float(trace.decay), float(update["decay"]))
    trace.novelty = min(float(trace.novelty), float(update["novelty"]))
    trace.polarity = float(update["polarity"])
    trace.fire_count += int(update.get("fire_count_delta", 0) or 0)
    trace.last_fired = max(
        int(trace.last_fired),
        int(update.get("last_fired", 0) or 0),
    )


def _apply_trace_sync_payload(
    trace_store: TraceStore,
    trace_sync: dict[str, Any],
    last_synced_trace_runtime: dict[str, dict[str, Any]] | None = None,
) -> None:
    for trace_id in list(trace_sync.get("remove_trace_ids", [])):
        trace_store.remove(str(trace_id))
        if last_synced_trace_runtime is not None:
            last_synced_trace_runtime.pop(str(trace_id), None)

    for key in ("new_traces", "full_updates"):
        for row in list(trace_sync.get(key, [])):
            trace = Trace.from_dict(dict(row))
            if trace.id in trace_store:
                trace_store.remove(trace.id)
            trace_store.add(trace)
            if last_synced_trace_runtime is not None:
                last_synced_trace_runtime[trace.id] = _trace_runtime_cache_entry(trace)

    for update in list(trace_sync.get("runtime_updates", [])):
        trace_id = str(update.get("id", ""))
        trace = trace_store.get(trace_id)
        if trace is None:
            continue
        _apply_trace_runtime_update(trace, dict(update))
        trace_store.sync_trace(trace.id)
        if last_synced_trace_runtime is not None:
            last_synced_trace_runtime[trace.id] = _trace_runtime_cache_entry(trace)


def _bound_pair_key(
    trace_id_a: str,
    region_a: str,
    trace_id_b: str,
    region_b: str,
) -> tuple[str, str, str, str]:
    left = (trace_id_a, region_a)
    right = (trace_id_b, region_b)
    if left <= right:
        return trace_id_a, region_a, trace_id_b, region_b
    return trace_id_b, region_b, trace_id_a, region_a


def _pattern_snapshot_key(pattern: tuple[Any, ...]) -> str:
    region_name, neuron_ids, threshold, _trace_id = pattern
    threshold_bits = struct.unpack("<I", struct.pack("<f", float(threshold)))[0]
    neuron_key = ",".join(str(int(neuron_id)) for neuron_id in neuron_ids)
    return f"{region_name}:{threshold_bits:08x}:{neuron_key}"


def _binding_snapshot_pair_key(snapshot: tuple[Any, ...]) -> str:
    left = _pattern_snapshot_key(snapshot[1])
    right = _pattern_snapshot_key(snapshot[2])
    if left <= right:
        return f"{left}|{right}"
    return f"{right}|{left}"


def _remap_binding_snapshots_trace_ids(
    binding_snapshots: list[tuple[Any, ...]],
    trace_id_redirects: dict[str, str],
) -> list[tuple[Any, ...]]:
    if not trace_id_redirects:
        return list(binding_snapshots)

    remapped_snapshots: list[tuple[Any, ...]] = []
    for snapshot in binding_snapshots:
        (
            binding_id,
            pattern_a,
            pattern_b,
            weight,
            fires,
            time_delta,
            last_fired,
            confidence,
            opportunities,
        ) = snapshot
        region_a, neuron_ids_a, threshold_a, trace_id_a = pattern_a
        region_b, neuron_ids_b, threshold_b, trace_id_b = pattern_b
        mapped_trace_id_a = (
            trace_id_redirects.get(str(trace_id_a), trace_id_a)
            if trace_id_a is not None
            else None
        )
        mapped_trace_id_b = (
            trace_id_redirects.get(str(trace_id_b), trace_id_b)
            if trace_id_b is not None
            else None
        )
        if mapped_trace_id_a is not None and mapped_trace_id_a == mapped_trace_id_b:
            continue
        remapped_snapshots.append(
            (
                binding_id,
                (region_a, neuron_ids_a, threshold_a, mapped_trace_id_a),
                (region_b, neuron_ids_b, threshold_b, mapped_trace_id_b),
                weight,
                fires,
                time_delta,
                last_fired,
                confidence,
                opportunities,
            )
        )
    return remapped_snapshots


def _merge_binding_snapshots(
    binding_snapshot_groups: list[list[tuple[Any, ...]]],
    leader_index: int,
) -> tuple[list[tuple[Any, ...]], dict[str, int]]:
    if not binding_snapshot_groups:
        return [], {
            "binding_count": 0,
            "bindings_added": 0,
            "bindings_deduped": 0,
        }

    leader_snapshots = list(binding_snapshot_groups[leader_index])
    next_binding_id = max((int(snapshot[0]) for snapshot in leader_snapshots), default=-1) + 1
    merged_snapshots: list[tuple[Any, ...]] = []
    seen_binding_keys: set[str] = set()
    added_bindings = 0
    duplicate_bindings = 0

    for group_index, snapshots in enumerate(binding_snapshot_groups):
        for snapshot in snapshots:
            binding_key = _binding_snapshot_pair_key(snapshot)
            if binding_key in seen_binding_keys:
                duplicate_bindings += 1
                continue
            seen_binding_keys.add(binding_key)

            merged_snapshot = snapshot
            if group_index != leader_index:
                merged_snapshot = (
                    next_binding_id,
                    snapshot[1],
                    snapshot[2],
                    snapshot[3],
                    snapshot[4],
                    snapshot[5],
                    snapshot[6],
                    snapshot[7],
                    snapshot[8],
                )
                next_binding_id += 1
                added_bindings += 1

            merged_snapshots.append(merged_snapshot)

    return merged_snapshots, {
        "binding_count": len(merged_snapshots),
        "bindings_added": added_bindings,
        "bindings_deduped": duplicate_bindings,
    }


def _binding_state_from_snapshots(
    binding_snapshots: list[tuple[Any, ...]],
) -> tuple[set[tuple[str, str, str, str]], dict[int, dict[str, object]]]:
    bound_pairs: set[tuple[str, str, str, str]] = set()
    binding_details: dict[int, dict[str, object]] = {}

    for snapshot in binding_snapshots:
        (
            binding_id,
            pattern_a,
            pattern_b,
            _weight,
            _fires,
            time_delta,
            _last_fired,
            _confidence,
            _opportunities,
        ) = snapshot
        region_a, _neurons_a, _threshold_a, trace_id_a = pattern_a
        region_b, _neurons_b, _threshold_b, trace_id_b = pattern_b

        if trace_id_a is None or trace_id_b is None:
            continue

        detail = {
            "binding_id": int(binding_id),
            "trace_id_a": str(trace_id_a),
            "region_a": str(region_a),
            "trace_id_b": str(trace_id_b),
            "region_b": str(region_b),
            "avg_delta": float(time_delta),
        }
        binding_details[int(binding_id)] = detail
        bound_pairs.add(
            _bound_pair_key(
                str(trace_id_a),
                str(region_a),
                str(trace_id_b),
                str(region_b),
            )
        )

    return bound_pairs, binding_details


def _trace_binding_ids_from_snapshots(
    binding_snapshots: list[tuple[Any, ...]],
) -> dict[str, list[int]]:
    binding_ids_by_trace: dict[str, set[int]] = defaultdict(set)
    for snapshot in binding_snapshots:
        binding_id = int(snapshot[0])
        trace_id_a = snapshot[1][3]
        trace_id_b = snapshot[2][3]
        if trace_id_a is not None:
            binding_ids_by_trace[str(trace_id_a)].add(binding_id)
        if trace_id_b is not None:
            binding_ids_by_trace[str(trace_id_b)].add(binding_id)
    return {
        trace_id: sorted(binding_ids)
        for trace_id, binding_ids in binding_ids_by_trace.items()
    }


def _apply_binding_ids_from_snapshots(
    trace_store: TraceStore,
    binding_snapshots: list[tuple[Any, ...]],
) -> set[str]:
    binding_ids_by_trace = _trace_binding_ids_from_snapshots(binding_snapshots)
    changed_trace_ids: set[str] = set()
    for trace_id, trace in trace_store.traces.items():
        next_binding_ids = list(binding_ids_by_trace.get(trace_id, []))
        if trace.binding_ids == next_binding_ids:
            continue
        trace.binding_ids = next_binding_ids
        changed_trace_ids.add(trace_id)
    return changed_trace_ids


def _shared_weight_array(
    shared_block: shared_memory.SharedMemory,
    synapse_count: int,
) -> np.ndarray:
    return np.ndarray((synapse_count,), dtype=_WEIGHT_DTYPE, buffer=shared_block.buf)


def _encode_synapse_delta_rows(delta_rows: list[tuple[int, float]]) -> bytes:
    if not delta_rows:
        return b""

    payload = np.empty(len(delta_rows), dtype=_SYNAPSE_DELTA_DTYPE)
    payload["synapse_index"] = np.fromiter(
        (int(synapse_index) for synapse_index, _ in delta_rows),
        dtype=np.uint32,
        count=len(delta_rows),
    )
    payload["delta"] = np.fromiter(
        (float(delta) for _, delta in delta_rows),
        dtype=np.float32,
        count=len(delta_rows),
    )
    return payload.tobytes()


def _decode_synapse_delta_rows(delta_bytes: bytes) -> list[tuple[int, float]]:
    if not delta_bytes:
        return []
    if len(delta_bytes) % _SYNAPSE_DELTA_DTYPE.itemsize != 0:
        raise ValueError(
            "Sparse synapse delta payload length does not align to the expected record size"
        )

    payload = np.frombuffer(delta_bytes, dtype=_SYNAPSE_DELTA_DTYPE)
    return [
        (int(row["synapse_index"]), float(row["delta"]))
        for row in payload
    ]


def _average_sparse_synapse_deltas(
    worker_delta_groups: list[list[tuple[int, float]]],
) -> tuple[list[tuple[int, float]], dict[str, float]]:
    delta_sums: dict[int, float] = defaultdict(float)
    contributor_counts: dict[int, int] = defaultdict(int)
    contributor_updates = 0

    for delta_rows in worker_delta_groups:
        contributor_updates += len(delta_rows)
        for synapse_index, delta in delta_rows:
            delta_sums[int(synapse_index)] += float(delta)
            contributor_counts[int(synapse_index)] += 1

    averaged_rows: list[tuple[int, float]] = []
    max_contributors = 0
    for synapse_index in sorted(delta_sums):
        contributors = contributor_counts[synapse_index]
        max_contributors = max(max_contributors, contributors)
        averaged_delta = delta_sums[synapse_index] / float(contributors)
        if abs(averaged_delta) <= 1e-12:
            continue
        averaged_rows.append((int(synapse_index), float(averaged_delta)))

    return averaged_rows, {
        "dirty_synapse_count": float(len(averaged_rows)),
        "contributor_update_count": float(contributor_updates),
        "max_workers_per_synapse": float(max_contributors),
    }


def _shared_bytes_block(data: bytes) -> shared_memory.SharedMemory:
    shared_block = shared_memory.SharedMemory(create=True, size=max(len(data), 1))
    if data:
        shared_block.buf[: len(data)] = data
    return shared_block


def _read_shared_bytes(
    shared_block_name: str,
    data_size: int,
) -> bytes:
    shared_block = shared_memory.SharedMemory(name=shared_block_name)
    try:
        if data_size <= 0:
            return b""
        return bytes(shared_block.buf[:data_size])
    finally:
        _close_shared_memory_block(shared_block)


def _flush_pending_synapse_updates() -> dict[str, Any]:
    pending_before = int(brain_core.get_pending_synapse_update_count())
    pending_profile = None
    if pending_before > 0:
        pending_profile = dict(brain_core.apply_synapse_updates_profiled())
    return {
        "pending_synapse_updates_before_flush": pending_before,
        "pending_synapse_updates_after_flush": int(brain_core.get_pending_synapse_update_count()),
        "pending_synapse_update_profile": pending_profile,
    }


def _serialize_worker_round_state(
    trace_store: TraceStore,
    tick_loop: TickLoop,
    *,
    extra_metadata: dict[str, Any] | None = None,
    include_leader_state: bool,
    last_synced_trace_runtime: dict[str, dict[str, Any]],
) -> tuple[bytes, dict[str, Any]]:
    trace_delta = _build_worker_trace_delta(
        trace_store,
        last_synced_trace_runtime=last_synced_trace_runtime,
    )
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tick_count": int(brain_core.get_tick_count()),
        "neuron_count": int(brain_core.get_neuron_count()),
        "synapse_count": int(brain_core.get_synapse_count()),
        "trace_count": int(len(trace_store)),
        "binding_count": int(brain_core.get_binding_count()),
        "saved_leader_state": bool(include_leader_state),
        "trace_delta_new_count": int(len(trace_delta["new_traces"])),
        "trace_delta_runtime_update_count": int(len(trace_delta["runtime_updates"])),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    payload = {
        "trace_delta": trace_delta,
        "leader_state": (
            tick_loop.export_async_sync_state() if include_leader_state else {}
        ),
        "metadata": metadata,
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), metadata


def _close_shared_memory_block(shared_block: shared_memory.SharedMemory) -> None:
    try:
        shared_block.close()
    except FileNotFoundError:
        pass


def _unlink_shared_memory_block(shared_block: shared_memory.SharedMemory) -> None:
    try:
        shared_block.unlink()
    except FileNotFoundError:
        pass


def _load_baseline_ticks_per_sec(dataset: str) -> float | None:
    if dataset != "ag_news" or not _CHECKPOINT2_BASELINE_PATH.exists():
        return None
    try:
        data = json.loads(_CHECKPOINT2_BASELINE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data.get("global_summary"), dict):
        value = data["global_summary"].get("ticks_per_sec")
    else:
        value = data.get("ticks_per_sec")
    return float(value) if isinstance(value, (int, float)) else None


def _build_round_batches(
    samples: list[dict[str, Any]],
    worker_count: int,
    merge_every_samples: int,
) -> list[list[list[dict[str, Any]]]]:
    rounds: list[list[list[dict[str, Any]]]] = []
    cursor = 0
    round_capacity = worker_count * merge_every_samples
    while cursor < len(samples):
        round_window = list(samples[cursor : cursor + round_capacity])
        cursor += len(round_window)
        round_batches = [[] for _worker_id in range(worker_count)]
        for sample_index, sample in enumerate(round_window):
            round_batches[sample_index % worker_count].append(sample)
        rounds.append(round_batches)
    return rounds


def _reset_worker_round_runtime(
    tick_loop: TickLoop,
    *,
    round_index: int,
    reused_local_runtime: bool,
) -> None:
    if round_index <= 1 or not reused_local_runtime:
        return

    # Async rounds are scheduler artifacts. Preserve learned structure while
    # clearing transient neuron/delay state so one batch does not heat the next.
    tick_loop.reset_runtime_boundary(preserve_binding_state=True)


def _run_text_batch(
    trace_store: TraceStore,
    tick_loop: TickLoop,
    samples: list[dict[str, Any]],
    *,
    ticks_per_sample: int,
    rest_ticks: int,
    worker_id: int,
    round_index: int,
) -> dict[str, Any]:
    text_input = TextInput(trace_store)
    collector = MetricsCollector(
        dataset="async_multi_brain",
        mode=f"worker_{worker_id}",
        threads=int(brain_core.get_num_threads()),
        ticks_per_sample=ticks_per_sample,
        extra_config={
            "worker_id": int(worker_id),
            "round_index": int(round_index),
            "sample_count": int(len(samples)),
            "rest_ticks": int(rest_ticks),
        },
    )
    collector.start()

    for sample_index, sample in enumerate(samples):
        tick_loop.reset_sample_boundary()
        text = str(sample["text"])
        sample_metrics = collector.begin_sample(
            sample_index,
            label=str(sample.get("label_name", sample.get("label", ""))),
            modality="text",
        )
        encoded: dict[str, Any] = {"tokens": [], "known_count": 0, "unknown_count": 0}

        if ticks_per_sample > 0:
            encoded = text_input.encode(text)
        for result in tick_loop.iter_steps(ticks_per_sample, preserve_first_tick=True):
            executed_ticks = max(1, int(result.get("executed_ticks", 1) or 1))
            collector.record_tick_time(
                float(result.get("step_internal_ms", 0.0) or 0.0)
                / 1000.0
                / executed_ticks
            )
            result["input_known"] = int(encoded.get("known_count", 0) or 0)
            result["input_unknown"] = int(encoded.get("unknown_count", 0) or 0)
            sample_metrics.add_tick(result)

        collector.end_sample()

        for _result in tick_loop.iter_steps(max(0, rest_ticks)):
            pass

    collector.finish()
    summary = collector.global_summary()
    summary["duration_sec_raw"] = float(collector.duration)
    return summary


def _worker_main(
    worker_id: int,
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    *,
    cores_per_worker: int,
    weight_shm_name: str,
    weight_shm_size: int,
) -> None:
    if cores_per_worker > 0:
        try:
            brain_core.set_num_threads(int(cores_per_worker))
        except Exception:
            pass

    bootstrapped = False
    trace_store: TraceStore | None = None
    tick_loop: TickLoop | None = None
    retained_bundle_blocks: list[tuple[int, shared_memory.SharedMemory]] = []
    last_synced_trace_runtime: dict[str, dict[str, Any]] = {}

    try:
        while True:
            command = command_queue.get()
            command_type = command.get("type")

            if command_type == "stop":
                return

            if command_type != "train_text_batch":
                result_queue.put(
                    {
                        "worker_id": int(worker_id),
                        "status": "error",
                        "error": f"Unknown worker command: {command_type}",
                    }
                )
                continue

            try:
                round_index = int(command["round_index"])
                retained_next: list[tuple[int, shared_memory.SharedMemory]] = []
                for retained_round, retained_block in retained_bundle_blocks:
                    if retained_round <= round_index - 2:
                        _close_shared_memory_block(retained_block)
                        _unlink_shared_memory_block(retained_block)
                    else:
                        retained_next.append((retained_round, retained_block))
                retained_bundle_blocks = retained_next
                samples = list(command.get("samples", []))
                ticks_per_sample = int(command["ticks_per_sample"])
                rest_ticks = int(command.get("rest_ticks", 0))
                runtime_bundle_name = str(command.get("runtime_bundle_name", ""))
                runtime_bundle_size = int(command.get("runtime_bundle_size", 0) or 0)
                sync_payload_name = str(command.get("sync_payload_name", ""))
                sync_payload_size = int(command.get("sync_payload_size", 0) or 0)
                sync_trace_binding = bool(command.get("sync_trace_binding", False))

                if not bootstrapped:
                    bootstrap_paths = bundle_paths(str(command["bootstrap_bundle_dir"]))
                    brain_core.load_brain_checkpoint(str(bootstrap_paths["brain"]))
                    brain_core.attach_shared_weight_buffer(
                        str(weight_shm_name),
                        int(weight_shm_size),
                    )
                    if not runtime_bundle_name:
                        raise RuntimeError(
                            "Initial runtime bundle is required to bootstrap async worker state"
                        )
                    runtime_bundle_bytes = _read_shared_bytes(
                        runtime_bundle_name,
                        runtime_bundle_size,
                    )
                    trace_store, tick_loop, _python_state, _metadata = load_runtime_bundle_bytes(
                        runtime_bundle_bytes
                    )
                    tick_loop.rust_tick_batch_size = DEFAULT_ASYNC_MULTI_BRAIN_TICK_BATCH_SIZE
                    tick_loop.collect_full_metrics = False
                    last_synced_trace_runtime = _trace_runtime_cache_from_store(trace_store)
                    bootstrapped = True
                elif trace_store is None or tick_loop is None:
                    raise RuntimeError("Async worker lost local runtime state")
                elif runtime_bundle_name:
                    runtime_bundle_bytes = _read_shared_bytes(
                        runtime_bundle_name,
                        runtime_bundle_size,
                    )
                    trace_store, tick_loop, _python_state, _metadata = load_runtime_bundle_bytes(
                        runtime_bundle_bytes
                    )
                    tick_loop.rust_tick_batch_size = DEFAULT_ASYNC_MULTI_BRAIN_TICK_BATCH_SIZE
                    tick_loop.collect_full_metrics = False
                    last_synced_trace_runtime = _trace_runtime_cache_from_store(trace_store)

                if trace_store is not None and tick_loop is not None:
                    _reset_worker_round_runtime(
                        tick_loop,
                        round_index=round_index,
                        reused_local_runtime=bootstrapped and not bool(runtime_bundle_name),
                    )

                if sync_payload_name:
                    if trace_store is None or tick_loop is None:
                        raise RuntimeError(
                            "Async worker cannot apply sync payload before bootstrap"
                        )
                    sync_payload_bytes = _read_shared_bytes(
                        sync_payload_name,
                        sync_payload_size,
                    )
                    sync_payload = dict(pickle.loads(sync_payload_bytes))
                    _apply_trace_sync_payload(
                        trace_store,
                        dict(sync_payload.get("trace_sync", {})),
                        last_synced_trace_runtime,
                    )
                    merged_binding_snapshots = list(sync_payload.get("binding_snapshots", []))
                    brain_core.replace_bindings(merged_binding_snapshots)
                    _apply_binding_ids_from_snapshots(trace_store, merged_binding_snapshots)
                    bound_pairs, binding_details = _binding_state_from_snapshots(
                        merged_binding_snapshots
                    )
                    tick_loop.binding_formation._bound_pairs = set(bound_pairs)
                    tick_loop.binding_formation._binding_details = dict(binding_details)
                    leader_state = dict(sync_payload.get("leader_state", {}))
                    if leader_state:
                        tick_loop.apply_async_sync_state(leader_state)

                summary = _run_text_batch(
                    trace_store,
                    tick_loop,
                    samples,
                    ticks_per_sample=ticks_per_sample,
                    rest_ticks=rest_ticks,
                    worker_id=worker_id,
                    round_index=round_index,
                )
                flush_summary = _flush_pending_synapse_updates()
                current_synapse_count = int(brain_core.get_synapse_count())
                topology_signature = int(brain_core.get_synapse_topology_signature())
                binding_snapshots = (
                    list(brain_core.export_bindings()) if sync_trace_binding else []
                )
                result_bundle_block: shared_memory.SharedMemory | None = None
                try:
                    bundle_bytes = b""
                    metadata = {
                        "round_index": int(round_index),
                        "sample_count": int(len(samples)),
                        "worker_threads": int(brain_core.get_num_threads()),
                        "runtime_bootstrap_bundle_dir": str(
                            command.get("bootstrap_bundle_dir", "")
                        ),
                        **flush_summary,
                        "topology_signature": topology_signature,
                        "trace_binding_sync": bool(sync_trace_binding),
                    }
                    if sync_trace_binding:
                        bundle_bytes, metadata = _serialize_worker_round_state(
                            trace_store,
                            tick_loop,
                            extra_metadata=dict(metadata),
                            include_leader_state=(
                                worker_id == DEFAULT_ASYNC_MULTI_BRAIN_LEADER_WORKER_ID
                            ),
                            last_synced_trace_runtime=last_synced_trace_runtime,
                        )
                        metadata["trace_binding_sync"] = True
                        result_bundle_block = _shared_bytes_block(bundle_bytes)

                    result_queue.put(
                        {
                            "worker_id": int(worker_id),
                            "round_index": int(round_index),
                            "status": "ok",
                            "summary": summary,
                            "metadata": metadata,
                            "binding_snapshots": binding_snapshots,
                            "bundle_shm_name": (
                                result_bundle_block.name if result_bundle_block is not None else ""
                            ),
                            "bundle_size": int(len(bundle_bytes)),
                            "topology_signature": topology_signature,
                            "synapse_count": current_synapse_count,
                            "trace_binding_sync": bool(sync_trace_binding),
                        }
                    )
                    if result_bundle_block is not None:
                        retained_bundle_blocks.append((round_index, result_bundle_block))
                        result_bundle_block = None
                except Exception:
                    if result_bundle_block is not None:
                        _close_shared_memory_block(result_bundle_block)
                        _unlink_shared_memory_block(result_bundle_block)
                    raise
            except Exception as exc:
                result_queue.put(
                    {
                        "worker_id": int(worker_id),
                        "status": "error",
                        "error": str(exc),
                    }
                )
    finally:
        for _retained_round, retained_block in retained_bundle_blocks:
            _close_shared_memory_block(retained_block)
            _unlink_shared_memory_block(retained_block)


def _start_train_round(
    command_queues: list[mp.Queue],
    *,
    round_index: int,
    round_batches: list[list[dict[str, Any]]],
    bootstrap_bundle_dir: str,
    ticks_per_sample: int,
    rest_ticks: int,
    runtime_bundle_bytes: bytes | None = None,
    sync_payload_bytes: bytes | None = None,
    sync_trace_binding: bool = False,
) -> dict[str, Any]:
    dispatch_started = time.perf_counter()
    runtime_bundle_block = (
        _shared_bytes_block(runtime_bundle_bytes)
        if runtime_bundle_bytes is not None
        else None
    )
    sync_payload_block = (
        _shared_bytes_block(sync_payload_bytes)
        if sync_payload_bytes is not None
        else None
    )

    for worker_id, batch in enumerate(round_batches):
        command: dict[str, Any] = {
            "type": "train_text_batch",
            "round_index": int(round_index),
            "bootstrap_bundle_dir": bootstrap_bundle_dir,
            "samples": batch,
            "ticks_per_sample": int(ticks_per_sample),
            "rest_ticks": int(rest_ticks),
            "sync_trace_binding": bool(sync_trace_binding),
        }
        if runtime_bundle_block is not None:
            command["runtime_bundle_name"] = runtime_bundle_block.name
            command["runtime_bundle_size"] = int(len(runtime_bundle_bytes or b""))
        if sync_payload_block is not None:
            command["sync_payload_name"] = sync_payload_block.name
            command["sync_payload_size"] = int(len(sync_payload_bytes or b""))
        command_queues[worker_id].put(command)

    return {
        "round_index": int(round_index),
        "dispatch_started": dispatch_started,
        "runtime_bundle_block": runtime_bundle_block,
        "sync_payload_block": sync_payload_block,
    }


def _finish_train_round(
    round_handle: dict[str, Any],
    result_queue: mp.Queue,
    worker_count: int,
) -> tuple[list[dict[str, Any]], float]:
    try:
        worker_results = _collect_worker_results(result_queue, worker_count)
    finally:
        runtime_bundle_block = round_handle.get("runtime_bundle_block")
        if runtime_bundle_block is not None:
            _close_shared_memory_block(runtime_bundle_block)
            _unlink_shared_memory_block(runtime_bundle_block)
        sync_payload_block = round_handle.get("sync_payload_block")
        if sync_payload_block is not None:
            _close_shared_memory_block(sync_payload_block)
            _unlink_shared_memory_block(sync_payload_block)

    worker_wall_ms = (time.perf_counter() - float(round_handle["dispatch_started"])) * 1000
    return worker_results, worker_wall_ms


def _collect_worker_results(
    result_queue: mp.Queue,
    worker_count: int,
) -> list[dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    while len(results) < worker_count:
        try:
            result = result_queue.get(timeout=300)
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for async worker results") from exc
        worker_id = int(result.get("worker_id", -1))
        results[worker_id] = result
    return [results[index] for index in sorted(results)]


def _materialize_worker_result_bundle(worker_result: dict[str, Any]) -> bytes:
    bundle_name = str(worker_result.get("bundle_shm_name", ""))
    bundle_size = int(worker_result.get("bundle_size", 0) or 0)
    result_bundle = shared_memory.SharedMemory(name=bundle_name)
    try:
        if bundle_size <= 0:
            return b""
        return bytes(result_bundle.buf[:bundle_size])
    finally:
        _close_shared_memory_block(result_bundle)
        _unlink_shared_memory_block(result_bundle)


def _stop_workers(
    workers: list[mp.Process],
    command_queues: list[mp.Queue],
) -> None:
    for command_queue in command_queues:
        try:
            command_queue.put({"type": "stop"})
        except Exception:
            pass

    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)


def run_async_multi_brain_text(
    dataset: str,
    max_samples: int,
    ticks_per_sample: int,
    output_path: str,
    *,
    worker_count: int = DEFAULT_ASYNC_MULTI_BRAIN_WORKER_COUNT,
    cores_per_worker: int = DEFAULT_ASYNC_MULTI_BRAIN_CORES_PER_WORKER,
    merge_every_samples: int = DEFAULT_ASYNC_MULTI_BRAIN_MERGE_EVERY_SAMPLES,
    rest_ticks: int = 3,
    fast: bool = True,
    n_traces: int = 5000,
    seed_chunks: int | None = None,
    validation_threads: int | None = None,
    run_validations: bool = True,
) -> dict[str, Any]:
    if dataset not in {"ag_news", "imdb"}:
        raise ValueError("async_multi_brain currently supports only ag_news and imdb")
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    if merge_every_samples < 1:
        raise ValueError("merge_every_samples must be at least 1")

    from brain.datasets.downloader import load_text_dataset

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_file.parent / f"{output_file.stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if cores_per_worker > 0:
        try:
            brain_core.set_num_threads(int(cores_per_worker))
        except Exception:
            pass
    actual_threads = int(brain_core.get_num_threads())
    validation_threads = int(validation_threads or actual_threads)

    samples = load_text_dataset(dataset, max_samples=max_samples)
    rounds = _build_round_batches(samples, worker_count, merge_every_samples)
    leader_worker_id = DEFAULT_ASYNC_MULTI_BRAIN_LEADER_WORKER_ID
    base_bundle_dir = artifact_dir / "base_bundle"

    if fast:
        _, trace_store = seed_brain_fast(
            n_traces=n_traces,
            verbose=False,
            chunk_count=seed_chunks,
        )
    else:
        _, trace_store = seed_brain(
            verbose=False,
            chunk_count=seed_chunks,
        )
    tick_loop = TickLoop(
        trace_store,
        rust_tick_batch_size=DEFAULT_ASYNC_MULTI_BRAIN_TICK_BATCH_SIZE,
        collect_full_metrics=False,
    )
    save_runtime_bundle(
        trace_store,
        tick_loop,
        base_bundle_dir,
        extra_metadata={
            "dataset": dataset,
            "seed_mode": "fast" if fast else "full",
            "seed_traces": int(n_traces),
            "seed_chunks": seed_chunks,
            "worker_threads": actual_threads,
        },
    )
    bootstrap_bundle_dir = str(base_bundle_dir)
    initial_bundle_bytes, _initial_bundle_metadata = serialize_runtime_bundle(
        trace_store,
        tick_loop,
        brain_checkpoint_kind="runtime",
        extra_metadata={
            "dataset": dataset,
            "seed_mode": "fast" if fast else "full",
            "seed_traces": int(n_traces),
            "seed_chunks": seed_chunks,
            "worker_threads": actual_threads,
            "runtime_bootstrap_bundle_dir": bootstrap_bundle_dir,
        },
    )
    current_trace_store = trace_store
    current_tick_loop = tick_loop
    current_binding_snapshots = list(brain_core.export_bindings())
    synapse_count = int(brain_core.get_synapse_count())
    topology_signature = int(brain_core.get_synapse_topology_signature())
    dense_weight_snapshot_bytes_per_worker = synapse_count * np.dtype(_WEIGHT_DTYPE).itemsize
    dense_weight_snapshot_total_bytes = dense_weight_snapshot_bytes_per_worker * worker_count
    shared_weight_allocation_bytes = max(
        dense_weight_snapshot_bytes_per_worker,
        np.dtype(_WEIGHT_DTYPE).itemsize,
    )
    shared_weight_block = shared_memory.SharedMemory(
        create=True,
        size=shared_weight_allocation_bytes,
    )
    shared_weight_view = _shared_weight_array(shared_weight_block, synapse_count)
    brain_core.write_synapse_weights_to_buffer(shared_weight_view)
    brain_core.attach_shared_weight_buffer(
        shared_weight_block.name,
        dense_weight_snapshot_bytes_per_worker,
    )

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    command_queues: list[mp.Queue] = []
    workers: list[mp.Process] = []
    try:
        for worker_id in range(worker_count):
            command_queue: mp.Queue = ctx.Queue()
            worker = ctx.Process(
                target=_worker_main,
                args=(worker_id, command_queue, result_queue),
                kwargs={
                    "cores_per_worker": cores_per_worker,
                    "weight_shm_name": shared_weight_block.name,
                    "weight_shm_size": dense_weight_snapshot_bytes_per_worker,
                },
            )
            worker.start()
            command_queues.append(command_queue)
            workers.append(worker)

        worker_totals: dict[int, dict[str, float]] = defaultdict(
            lambda: {"samples": 0.0, "ticks": 0.0, "duration_sec": 0.0}
        )
        round_rows: list[dict[str, Any]] = []
        trace_binding_sync_rows: list[dict[str, Any]] = []
        total_merge_ms = 0.0
        total_trace_merge_ms = 0.0
        total_binding_merge_ms = 0.0
        total_state_load_ms = 0.0
        total_tick_restore_ms = 0.0
        total_state_serialize_ms = 0.0
        active_traces_weighted_sum = 0.0
        active_traces_weight = 0.0
        bindings_formed_total = 0.0
        max_weight_round_merge_fraction = 0.0
        training_started = time.perf_counter()
        bootstrap_runtime_bundle_bytes: bytes | None = initial_bundle_bytes
        sync_payload_bytes: bytes | None = None

        try:
            for round_index, round_batches in enumerate(rounds, start=1):
                sync_trace_binding = (
                    (
                        _ASYNC_MULTI_BRAIN_TRACE_BINDING_SYNC_INTERVAL > 0
                        and round_index % _ASYNC_MULTI_BRAIN_TRACE_BINDING_SYNC_INTERVAL == 0
                    )
                    or round_index == len(rounds)
                )

                round_handle = _start_train_round(
                    command_queues,
                    round_index=round_index,
                    round_batches=round_batches,
                    bootstrap_bundle_dir=bootstrap_bundle_dir,
                    ticks_per_sample=ticks_per_sample,
                    rest_ticks=rest_ticks,
                    runtime_bundle_bytes=bootstrap_runtime_bundle_bytes,
                    sync_payload_bytes=sync_payload_bytes,
                    sync_trace_binding=sync_trace_binding,
                )
                bootstrap_runtime_bundle_bytes = None
                sync_payload_bytes = None
                worker_results, worker_wall_ms = _finish_train_round(
                    round_handle,
                    result_queue,
                    worker_count,
                )

                for worker_result in worker_results:
                    if worker_result.get("status") != "ok":
                        raise RuntimeError(
                            f"Async worker {worker_result.get('worker_id')} failed: {worker_result.get('error')}"
                        )
                    if int(worker_result.get("synapse_count", -1)) != synapse_count:
                        raise RuntimeError(
                            "Worker synapse count diverged during shared-weight async execution: "
                            f"expected {synapse_count}, got {worker_result.get('synapse_count')}"
                        )
                    if int(worker_result.get("topology_signature", -1)) != topology_signature:
                        raise RuntimeError(
                            "Worker checkpoint topology diverged from the fixed shared-weight topology"
                        )
                    if sync_trace_binding:
                        worker_result["bundle_bytes"] = _materialize_worker_result_bundle(worker_result)
                        if worker_result["bundle_bytes"]:
                            worker_result["bundle_payload"] = dict(
                                pickle.loads(worker_result["bundle_bytes"])
                            )

                trace_merge_summary = {
                    "traces_added": 0,
                    "traces_deduped": 0,
                    "traces_merged_existing_id": 0,
                }
                binding_merge_counts = {
                    "binding_count": len(current_binding_snapshots),
                    "bindings_added": 0,
                    "bindings_deduped": 0,
                }
                state_load_ms = 0.0
                trace_merge_ms = 0.0
                weight_merge_ms = 0.0
                binding_merge_ms = 0.0
                tick_restore_ms = 0.0
                state_serialize_ms = 0.0
                merge_ms = 0.0

                if sync_trace_binding:
                    merge_started = time.perf_counter()

                    state_load_started = time.perf_counter()
                    leader_payload = dict(
                        worker_results[leader_worker_id].get("bundle_payload", {})
                    )
                    if not leader_payload:
                        raise RuntimeError(
                            "Leader worker did not emit a sync payload on a trace/binding sync round"
                        )
                    leader_state = dict(leader_payload.get("leader_state", {}))
                    state_load_ms = (time.perf_counter() - state_load_started) * 1000
                    total_state_load_ms += state_load_ms

                    trace_merge_started = time.perf_counter()
                    trace_added_ids: set[str] = set()
                    trace_full_update_ids: set[str] = set()
                    trace_remove_ids: set[str] = set()
                    worker_runtime_update_groups: list[list[dict[str, Any]]] = []
                    for worker_result in worker_results:
                        worker_payload = dict(worker_result.get("bundle_payload", {}))
                        worker_trace_delta = dict(worker_payload.get("trace_delta", {}))
                        worker_summary = _merge_worker_trace_rows_incremental(
                            current_trace_store,
                            list(worker_trace_delta.get("new_traces", [])),
                        )
                        for key in (
                            "traces_added",
                            "traces_deduped",
                            "traces_merged_existing_id",
                        ):
                            trace_merge_summary[key] += int(worker_summary.get(key, 0) or 0)
                        trace_added_ids.update(
                            str(trace_id)
                            for trace_id in worker_summary.get("added_trace_ids", [])
                        )
                        trace_full_update_ids.update(
                            str(trace_id)
                            for trace_id in worker_summary.get("updated_trace_ids", [])
                        )
                        trace_remove_ids.update(
                            str(trace_id)
                            for trace_id in worker_summary.get("dropped_trace_ids", [])
                        )
                        worker_result["binding_snapshots"] = _remap_binding_snapshots_trace_ids(
                            list(worker_result.get("binding_snapshots", [])),
                            dict(worker_summary.get("trace_id_redirects", {})),
                        )
                        worker_runtime_update_groups.append(
                            list(worker_trace_delta.get("runtime_updates", []))
                        )

                    aggregated_runtime_updates = _aggregate_trace_runtime_updates(
                        worker_runtime_update_groups
                    )
                    for update in aggregated_runtime_updates:
                        trace = current_trace_store.get(str(update["id"]))
                        if trace is None:
                            continue
                        _apply_trace_runtime_update(trace, update)
                        current_trace_store.sync_trace(trace.id)
                    trace_merge_ms = (time.perf_counter() - trace_merge_started) * 1000
                    total_trace_merge_ms += trace_merge_ms

                    binding_merge_started = time.perf_counter()
                    binding_snapshot_groups: list[list[tuple[Any, ...]]] = [
                        list(current_binding_snapshots)
                    ]
                    for worker_result in worker_results:
                        binding_snapshot_groups.append(
                            list(worker_result.get("binding_snapshots", []))
                        )
                    merged_binding_snapshots, binding_merge_counts = _merge_binding_snapshots(
                        binding_snapshot_groups,
                        0,
                    )
                    current_binding_snapshots = list(merged_binding_snapshots)
                    brain_core.replace_bindings(current_binding_snapshots)
                    binding_trace_ids_changed = _apply_binding_ids_from_snapshots(
                        current_trace_store,
                        current_binding_snapshots,
                    )
                    bound_pairs, binding_details = _binding_state_from_snapshots(
                        merged_binding_snapshots
                    )
                    binding_merge_ms = (time.perf_counter() - binding_merge_started) * 1000
                    total_binding_merge_ms += binding_merge_ms

                    tick_restore_started = time.perf_counter()
                    current_tick_loop.apply_async_sync_state(leader_state)
                    current_tick_loop.binding_formation._bound_pairs = set(bound_pairs)
                    current_tick_loop.binding_formation._binding_details = dict(binding_details)
                    tick_restore_ms = (time.perf_counter() - tick_restore_started) * 1000
                    total_tick_restore_ms += tick_restore_ms

                    trace_full_update_ids.update(binding_trace_ids_changed)
                    trace_full_update_ids.difference_update(trace_added_ids)
                    sync_runtime_updates = [
                        dict(update)
                        for update in aggregated_runtime_updates
                        if str(update["id"]) not in trace_added_ids
                        and str(update["id"]) not in trace_full_update_ids
                    ]
                    coordinator_trace_sync = {
                        "new_traces": [
                            current_trace_store.get(trace_id).to_dict()
                            for trace_id in sorted(trace_added_ids)
                            if current_trace_store.get(trace_id) is not None
                        ],
                        "full_updates": [
                            current_trace_store.get(trace_id).to_dict()
                            for trace_id in sorted(trace_full_update_ids)
                            if current_trace_store.get(trace_id) is not None
                        ],
                        "runtime_updates": sync_runtime_updates,
                        "remove_trace_ids": sorted(trace_remove_ids),
                    }

                    if round_index < len(rounds):
                        state_serialize_started = time.perf_counter()
                        sync_payload_bytes = pickle.dumps(
                            {
                                "trace_sync": coordinator_trace_sync,
                                "leader_state": leader_state,
                                "binding_snapshots": current_binding_snapshots,
                                "metadata": {
                                    "dataset": dataset,
                                    "round_index": int(round_index),
                                    "worker_count": int(worker_count),
                                    "merge_every_samples": int(merge_every_samples),
                                    "runtime_bootstrap_bundle_dir": bootstrap_bundle_dir,
                                    "trace_merge_summary": trace_merge_summary,
                                    "merge_strategy": "lock_free_shared_weights_trace_delta_sync",
                                },
                            },
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                        state_serialize_ms = (
                            time.perf_counter() - state_serialize_started
                        ) * 1000
                        total_state_serialize_ms += state_serialize_ms

                    merge_ms = (time.perf_counter() - merge_started) * 1000
                    total_merge_ms += merge_ms
                    trace_binding_sync_rows.append(
                        {
                            "round_index": int(round_index),
                            "trace_count": len(current_trace_store),
                            "binding_count": len(current_binding_snapshots),
                            "state_load_ms": _round(state_load_ms, 4),
                            "trace_merge_ms": _round(trace_merge_ms, 4),
                            "binding_merge_ms": _round(binding_merge_ms, 4),
                            "tick_restore_ms": _round(tick_restore_ms, 4),
                            "state_serialize_ms": _round(state_serialize_ms, 4),
                            "merge_ms": _round(merge_ms, 4),
                            "next_sync_payload_bytes": int(len(sync_payload_bytes or b"")),
                        }
                    )
                else:
                    max_weight_round_merge_fraction = max(
                        max_weight_round_merge_fraction,
                        merge_ms / max(worker_wall_ms, 1.0),
                    )

                brain_merge_summary = {
                    "merge_strategy": "lock_free_shared_weights_trace_delta_sync",
                    "worker_count": float(worker_count),
                    "leader_index": float(leader_worker_id),
                    "synapse_count": float(synapse_count),
                    "binding_count": float(binding_merge_counts["binding_count"]),
                    "bindings_added": float(binding_merge_counts["bindings_added"]),
                    "bindings_deduped": float(binding_merge_counts["bindings_deduped"]),
                    "trace_binding_sync": bool(sync_trace_binding),
                    "trace_binding_sync_interval_rounds": float(
                        _ASYNC_MULTI_BRAIN_TRACE_BINDING_SYNC_INTERVAL
                    ),
                    "state_load_ms": _round(state_load_ms, 4),
                    "trace_merge_ms": _round(trace_merge_ms, 4),
                    "weight_merge_ms": _round(weight_merge_ms, 4),
                    "binding_merge_ms": _round(binding_merge_ms, 4),
                    "tick_restore_ms": _round(tick_restore_ms, 4),
                    "state_serialize_ms": _round(state_serialize_ms, 4),
                    "merge_ms": _round(merge_ms, 4),
                    "dense_weight_snapshot_bytes_per_worker_before": float(
                        dense_weight_snapshot_bytes_per_worker
                    ),
                    "dense_weight_snapshot_total_bytes_before": float(
                        dense_weight_snapshot_total_bytes
                    ),
                    "shared_weight_buffer_bytes": float(
                        dense_weight_snapshot_bytes_per_worker
                    ),
                    "weight_transfer_total_bytes": 0.0,
                    "worker_state_bytes_total": float(
                        sum(
                            int(worker_result.get("bundle_size", 0) or 0)
                            for worker_result in worker_results
                        )
                    ),
                    "binding_snapshot_total": float(
                        sum(
                            len(worker_result.get("binding_snapshots", []))
                            for worker_result in worker_results
                        )
                    ),
                }

                round_worker_rows: list[dict[str, Any]] = []
                for worker_result in worker_results:
                    worker_summary = dict(worker_result["summary"])
                    worker_id = int(worker_result["worker_id"])
                    worker_totals[worker_id]["samples"] += float(
                        worker_summary.get("total_samples", 0)
                    )
                    worker_totals[worker_id]["ticks"] += float(
                        worker_summary.get("total_ticks", 0)
                    )
                    worker_totals[worker_id]["duration_sec"] += float(
                        worker_summary.get(
                            "duration_sec_raw",
                            worker_summary.get("duration_sec", 0.0) or 0.0,
                        )
                    )

                    active_traces_avg = worker_summary.get("active_traces_avg")
                    worker_ticks = float(worker_summary.get("total_ticks", 0.0) or 0.0)
                    if isinstance(active_traces_avg, (int, float)) and worker_ticks > 0.0:
                        active_traces_weighted_sum += float(active_traces_avg) * worker_ticks
                        active_traces_weight += worker_ticks

                    bindings_formed_value = worker_summary.get("bindings_formed_total")
                    if isinstance(bindings_formed_value, (int, float)):
                        bindings_formed_total += float(bindings_formed_value)

                    round_worker_rows.append(
                        {
                            "worker_id": worker_id,
                            "sample_count": len(round_batches[worker_id]),
                            "bundle_size": int(worker_result.get("bundle_size", 0) or 0),
                            "trace_binding_sync": bool(
                                worker_result.get("trace_binding_sync", False)
                            ),
                            "summary": worker_summary,
                            "metadata": dict(worker_result.get("metadata", {})),
                        }
                    )

                round_rows.append(
                    {
                        "round_index": int(round_index),
                        "worker_wall_ms": _round(worker_wall_ms, 4),
                        "merge_ms": _round(merge_ms, 4),
                        "workers": round_worker_rows,
                        "trace_merge_summary": trace_merge_summary,
                        "brain_merge_summary": dict(brain_merge_summary),
                        "merged_state_trace_count": len(current_trace_store),
                        "merged_state_binding_count": len(current_binding_snapshots),
                        "trace_binding_sync": (
                            trace_binding_sync_rows[-1]
                            if sync_trace_binding and trace_binding_sync_rows
                            else None
                        ),
                    }
                )
        finally:
            _stop_workers(workers, command_queues)

        training_duration_sec = time.perf_counter() - training_started
        total_ticks = len(samples) * int(ticks_per_sample)
        ticks_per_sec = total_ticks / max(training_duration_sec, 0.001)
        baseline_ticks_per_sec = _load_baseline_ticks_per_sec(dataset)
        scaling_efficiency = None
        if baseline_ticks_per_sec and baseline_ticks_per_sec > 0.0:
            scaling_efficiency = ticks_per_sec / baseline_ticks_per_sec / max(worker_count, 1)

        final_bundle_dir = artifact_dir / "final_merged_bundle"
        shared_weight_snapshot = np.array(shared_weight_view, copy=True)
        rust_weight_snapshot = np.zeros(synapse_count, dtype=_WEIGHT_DTYPE)
        brain_core.write_synapse_weights_to_buffer(rust_weight_snapshot)
        weight_consistency_max_abs_diff = float(
            np.max(np.abs(shared_weight_snapshot - rust_weight_snapshot))
        ) if synapse_count > 0 else 0.0
        final_bundle_metadata = save_runtime_bundle(
            current_trace_store,
            current_tick_loop,
            final_bundle_dir,
            brain_checkpoint_kind="full",
            extra_metadata={
                "dataset": dataset,
                "worker_count": int(worker_count),
                "merge_every_samples": int(merge_every_samples),
                "runtime_bootstrap_bundle_dir": bootstrap_bundle_dir,
                "merge_strategy": "lock_free_shared_weights_trace_delta_sync",
                "trace_binding_sync_interval_rounds": int(
                    _ASYNC_MULTI_BRAIN_TRACE_BINDING_SYNC_INTERVAL
                ),
            },
        )
        final_bundle_file_bytes = sum(
            path.stat().st_size
            for name, path in bundle_paths(final_bundle_dir).items()
            if name != "dir" and path.exists()
        )
        weight_transfer_totals = [
            float(row["brain_merge_summary"].get("weight_transfer_total_bytes", 0.0))
            for row in round_rows
        ]
        max_weight_transfer_bytes = max(weight_transfer_totals, default=0.0)
        avg_weight_transfer_bytes = (
            sum(weight_transfer_totals) / len(weight_transfer_totals)
            if weight_transfer_totals
            else 0.0
        )

        phase11_path = artifact_dir / "phase11_operational_baseline.json"
        crossmodal_path = artifact_dir / "crossmodal_recall_probe.json"
        phase11_result: dict[str, Any] | None = None
        crossmodal_result: dict[str, Any] | None = None
        if run_validations:
            from brain.benchmarks.crossmodal_recall_probe import (
                run_crossmodal_recall_probe,
            )
            from brain.benchmarks.phase11_operational_baseline import (
                run_phase11_operational_baseline,
            )

            phase11_result = run_phase11_operational_baseline(
                threads=validation_threads,
                output_path=str(phase11_path),
                stability_samples=120,
                output_probe_samples=32,
                ticks_per_sample=10,
                rest_ticks=1,
                seed_chunks=seed_chunks,
                fast_mode=True,
                n_traces=max(int(n_traces), 5500),
                initial_state_dir=str(final_bundle_dir),
            )

            crossmodal_result = run_crossmodal_recall_probe(
                ticks_per_sample=max(12, int(ticks_per_sample)),
                threads=validation_threads,
                output_path=str(crossmodal_path),
                train_samples=6,
                n_traces=max(int(n_traces), 5500),
                seed_chunks=seed_chunks,
                rest_ticks=1,
                settle_ticks=3,
                probe_ticks=4,
                cue_fraction=1.0,
                initial_state_dir=str(final_bundle_dir),
            )

        worker_summaries: dict[str, dict[str, float]] = {}
        for worker_id, totals in sorted(worker_totals.items()):
            duration_sec = float(totals["duration_sec"])
            total_worker_ticks = float(totals["ticks"])
            worker_summaries[str(worker_id)] = {
                "samples": float(totals["samples"]),
                "total_ticks": total_worker_ticks,
                "duration_sec": _round(duration_sec, 6),
                "ticks_per_sec": _round(
                    total_worker_ticks / max(duration_sec, 0.001),
                    4,
                ),
            }

        training_summary = {
            "dataset": dataset,
            "total_samples": len(samples),
            "total_ticks": total_ticks,
            "worker_count": int(worker_count),
            "cores_per_worker": int(cores_per_worker),
            "actual_worker_threads": int(actual_threads),
            "merge_every_samples": int(merge_every_samples),
            "trace_binding_sync_interval_rounds": int(
                _ASYNC_MULTI_BRAIN_TRACE_BINDING_SYNC_INTERVAL
            ),
            "duration_sec": _round(training_duration_sec, 6),
            "ticks_per_sec": _round(ticks_per_sec, 4),
            "tick_time_avg_ms": _round(
                (training_duration_sec * 1000.0) / max(total_ticks, 1),
                4,
            ),
            "merge_total_ms": _round(total_merge_ms, 4),
            "weight_merge_total_ms": 0.0,
            "trace_merge_total_ms": _round(total_trace_merge_ms, 4),
            "binding_merge_total_ms": _round(total_binding_merge_ms, 4),
            "state_load_total_ms": _round(total_state_load_ms, 4),
            "tick_restore_total_ms": _round(total_tick_restore_ms, 4),
            "state_serialize_total_ms": _round(total_state_serialize_ms, 4),
            "merge_cost_fraction": _round(
                total_merge_ms / max(training_duration_sec * 1000.0, 1.0),
                6,
            ),
            "weight_round_merge_cost_max_fraction": _round(
                max_weight_round_merge_fraction,
                6,
            ),
            "baseline_ticks_per_sec": baseline_ticks_per_sec,
            "dense_weight_snapshot_bytes_per_worker_before": int(
                dense_weight_snapshot_bytes_per_worker
            ),
            "dense_weight_snapshot_total_bytes_before": int(
                dense_weight_snapshot_total_bytes
            ),
            "shared_weight_buffer_bytes": int(dense_weight_snapshot_bytes_per_worker),
            "weight_transfer_max_bytes": int(max_weight_transfer_bytes),
            "weight_transfer_avg_bytes": _round(avg_weight_transfer_bytes, 4),
            "weight_consistency_max_abs_diff": _round(
                weight_consistency_max_abs_diff,
                8,
            ),
            "final_state_bundle_bytes": int(final_bundle_file_bytes),
            "scaling_efficiency_per_instance": (
                _round(scaling_efficiency, 6) if scaling_efficiency is not None else None
            ),
            "active_traces_avg": (
                _round(active_traces_weighted_sum / active_traces_weight, 4)
                if active_traces_weight > 0.0
                else None
            ),
            "bindings_formed_total": int(bindings_formed_total),
            "worker_summaries": worker_summaries,
        }

        validation_summary = {
            "skipped": not run_validations,
            "phase11_operational_baseline": phase11_result,
            "crossmodal_recall_probe": crossmodal_result,
        }

        phase11_fast_pass = None
        crossmodal_pass = None
        if run_validations:
            phase11_fast_pass = all(
                bool(value)
                for value in dict((phase11_result or {}).get("validations", {})).values()
            )
            crossmodal_pass = bool(
                (crossmodal_result or {}).get("summary", {})
                .get("validations", {})
                .get("passes_probe", False)
            )

        gate_results = {
            "throughput_min_3x_checkpoint2": (
                baseline_ticks_per_sec is not None
                and ticks_per_sec >= baseline_ticks_per_sec * 3.0
            ),
            "throughput_target_4x_checkpoint2": (
                baseline_ticks_per_sec is not None
                and ticks_per_sec >= baseline_ticks_per_sec * 4.0
            ),
            "merge_cost_below_10_percent_on_weight_rounds": (
                max_weight_round_merge_fraction < 0.10
            ),
            "trace_merge_below_50_ms_per_sync": all(
                float(row.get("trace_merge_ms", 0.0) or 0.0) < 50.0
                for row in trace_binding_sync_rows
            ),
            "leader_load_below_20_ms_per_sync": all(
                float(row.get("state_load_ms", 0.0) or 0.0) < 20.0
                for row in trace_binding_sync_rows
            ),
            "merge_cost_below_10_percent_overall": (
                (total_merge_ms / max(training_duration_sec * 1000.0, 1.0)) < 0.10
            ),
            "ticks_per_sec_above_50": ticks_per_sec > 50.0,
            "ticks_per_sec_above_100": ticks_per_sec > 100.0,
            "weight_transfer_zero": max_weight_transfer_bytes == 0.0,
            "no_weight_corruption_detected": weight_consistency_max_abs_diff <= 1e-7,
            "scaling_efficiency_per_instance_gte_0_75": (
                scaling_efficiency is not None and scaling_efficiency >= 0.75
            ),
            "phase11_fast_pass": phase11_fast_pass,
            "crossmodal_3_of_3_pass": crossmodal_pass,
        }

        aggregate = {
            "benchmark": "async_multi_brain",
            "training": training_summary,
            "rounds": round_rows,
            "validation": validation_summary,
            "gate_results": gate_results,
            "artifacts": {
                "base_bundle": str(base_bundle_dir),
                "final_merged_bundle": str(final_bundle_dir),
                "phase11": str(phase11_path) if run_validations else None,
                "crossmodal": str(crossmodal_path) if run_validations else None,
                "trace_binding_sync": trace_binding_sync_rows,
            },
            "final_bundle_metadata": final_bundle_metadata,
        }

        output_file.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        return aggregate
    finally:
        _close_shared_memory_block(shared_weight_block)
        _unlink_shared_memory_block(shared_weight_block)