"""Brain state: activation snapshots and neuromodulator state.

BrainState captures per-tick activation data from Rust and provides
it to Python-side learning and evaluation modules. NeuromodulatorState
tracks global modulators (arousal, valence, focus, energy).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NeuromodulatorState:
    """Global modulator state affecting all regions."""

    arousal: float = 0.5    # 0.0 (asleep) - 1.0 (panic)
    valence: float = 0.0    # -1.0 (negative) to 1.0 (positive)
    focus: float = 0.5      # 0.0 (scattered) to 1.0 (laser focused)
    energy: float = 1.0     # 0.0 (depleted) to 1.0 (full)

    def clamp(self) -> None:
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.valence = max(-1.0, min(1.0, self.valence))
        self.focus = max(0.0, min(1.0, self.focus))
        self.energy = max(0.0, min(1.0, self.energy))

    def to_dict(self) -> dict[str, float]:
        return {
            "arousal": float(self.arousal),
            "valence": float(self.valence),
            "focus": float(self.focus),
            "energy": float(self.energy),
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "NeuromodulatorState":
        return cls(
            arousal=float(data.get("arousal", 0.5)),
            valence=float(data.get("valence", 0.0)),
            focus=float(data.get("focus", 0.5)),
            energy=float(data.get("energy", 1.0)),
        )


@dataclass
class ActivationSnapshot:
    """Snapshot of neuron activations for one tick, per region.

    Used by learning modules to detect co-activation patterns.
    Stores only active neurons (sparse) to keep memory low.
    """

    tick: int
    # region_name → list of (global_neuron_id, activation_value)
    active_neurons: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    active_values: list[tuple[int, float]] = field(default_factory=list)
    total_active: int = 0
    active_ids: list[int] = field(default_factory=list)
    region_active_counts: dict[str, int] = field(default_factory=dict)

    def all_active_ids(self) -> list[int]:
        """Flat list of all active neuron IDs across regions."""
        if self.active_ids:
            return list(self.active_ids)
        if self.active_values:
            return [nid for nid, _ in self.active_values]
        ids = []
        for neurons in self.active_neurons.values():
            ids.extend(nid for nid, _ in neurons)
        return ids

    def active_set(self) -> set[int]:
        """Set of all active neuron IDs (for O(1) membership check)."""
        if self.active_ids:
            return set(self.active_ids)
        if self.active_values:
            return {nid for nid, _ in self.active_values}
        s = set()
        for neurons in self.active_neurons.values():
            for nid, _ in neurons:
                s.add(nid)
        return s

    def activation_of(self, neuron_id: int) -> float:
        """Get activation for a specific neuron, 0.0 if not active."""
        for nid, act in self.active_values:
            if nid == neuron_id:
                return act
        for neurons in self.active_neurons.values():
            for nid, act in neurons:
                if nid == neuron_id:
                    return act
        return 0.0

    def to_dict(self) -> dict:
        return {
            "tick": int(self.tick),
            "active_neurons": {
                region_name: [(int(nid), float(act)) for nid, act in neurons]
                for region_name, neurons in self.active_neurons.items()
            },
            "active_values": [(int(nid), float(act)) for nid, act in self.active_values],
            "total_active": int(self.total_active),
            "active_ids": [int(nid) for nid in self.active_ids],
            "region_active_counts": {
                region_name: int(count)
                for region_name, count in self.region_active_counts.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActivationSnapshot":
        return cls(
            tick=int(data.get("tick", 0)),
            active_neurons={
                str(region_name): [
                    (int(nid), float(act))
                    for nid, act in neurons
                ]
                for region_name, neurons in dict(data.get("active_neurons", {})).items()
            },
            active_values=[
                (int(nid), float(act))
                for nid, act in list(data.get("active_values", []))
            ],
            total_active=int(data.get("total_active", 0)),
            active_ids=[int(nid) for nid in list(data.get("active_ids", []))],
            region_active_counts={
                str(region_name): int(count)
                for region_name, count in dict(data.get("region_active_counts", {})).items()
            },
        )


class ActivationHistory:
    """Rolling window of recent activation snapshots.

    Keeps the last N ticks for Hebbian co-activation detection.
    """

    def __init__(self, window: int = 3):
        self.window = window
        self._history: list[ActivationSnapshot] = []
        self._baseline_totals: dict[int, float] = {}
        self._baseline_window_count: int = 0

    @staticmethod
    def _snapshot_values(snapshot: ActivationSnapshot) -> list[tuple[int, float]]:
        if snapshot.active_values:
            return list(snapshot.active_values)
        values: list[tuple[int, float]] = []
        for neurons in snapshot.active_neurons.values():
            values.extend(neurons)
        return values

    def _accumulate_baseline_snapshot(
        self,
        snapshot: ActivationSnapshot,
        multiplier: float,
    ) -> None:
        for neuron_id, activation in self._snapshot_values(snapshot):
            next_total = self._baseline_totals.get(neuron_id, 0.0) + (activation * multiplier)
            if abs(next_total) <= 1e-12:
                self._baseline_totals.pop(neuron_id, None)
            else:
                self._baseline_totals[neuron_id] = next_total

    def _rebuild_baseline_cache(self) -> None:
        self._baseline_totals = {}
        previous = self._history[:-1]
        for snapshot in previous:
            self._accumulate_baseline_snapshot(snapshot, 1.0)
        self._baseline_window_count = len(previous)

    def to_dict(self) -> dict:
        return {
            "window": int(self.window),
            "history": [snapshot.to_dict() for snapshot in self._history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActivationHistory":
        history = cls(window=int(data.get("window", 3)))
        history._history = [
            ActivationSnapshot.from_dict(snapshot)
            for snapshot in list(data.get("history", []))
        ]
        history._rebuild_baseline_cache()
        return history

    def push(self, snapshot: ActivationSnapshot) -> None:
        if self._history:
            self._accumulate_baseline_snapshot(self._history[-1], 1.0)
        self._history.append(snapshot)
        if len(self._history) > self.window:
            removed = self._history.pop(0)
            self._accumulate_baseline_snapshot(removed, -1.0)
        self._baseline_window_count = max(0, len(self._history) - 1)

    def __len__(self) -> int:
        return len(self._history)

    @property
    def current(self) -> ActivationSnapshot | None:
        return self._history[-1] if self._history else None

    @property
    def previous(self) -> ActivationSnapshot | None:
        return self._history[-2] if len(self._history) >= 2 else None

    def neurons_active_in_window(self) -> dict[int, float]:
        """Return all neurons that fired in the window, with max activation.

        Returns: {neuron_id: max_activation_in_window}
        """
        result: dict[int, float] = {}
        for snap in self._history:
            # Prefer flat arrays (avoids tuple overhead)
            flat_ids = getattr(snap, '_flat_ids', None)
            flat_vals = getattr(snap, '_flat_vals', None)
            if flat_ids is not None and flat_vals is not None:
                for i in range(len(flat_ids)):
                    nid = flat_ids[i]
                    act = flat_vals[i]
                    if nid not in result or act > result[nid]:
                        result[nid] = act
                continue
            if snap.active_values:
                for nid, act in snap.active_values:
                    if nid not in result or act > result[nid]:
                        result[nid] = act
                continue
            for neurons in snap.active_neurons.values():
                for nid, act in neurons:
                    if nid not in result or act > result[nid]:
                        result[nid] = act
        return result

    def neurons_active_in_window_flat(self) -> tuple[list, list]:
        """Return all neurons that fired in the window as flat parallel arrays.

        Returns: (ids: list[int], vals: list[float])
        Merges duplicates by keeping max activation.
        """
        result: dict[int, float] = {}
        for snap in self._history:
            flat_ids = getattr(snap, '_flat_ids', None)
            flat_vals = getattr(snap, '_flat_vals', None)
            if flat_ids is not None and flat_vals is not None:
                for i in range(len(flat_ids)):
                    nid = flat_ids[i]
                    act = flat_vals[i]
                    if nid not in result or act > result[nid]:
                        result[nid] = act
                continue
            if snap.active_values:
                for nid, act in snap.active_values:
                    if nid not in result or act > result[nid]:
                        result[nid] = act
                continue
            for neurons in snap.active_neurons.values():
                for nid, act in neurons:
                    if nid not in result or act > result[nid]:
                        result[nid] = act
        ids = list(result.keys())
        vals = list(result.values())
        return ids, vals

    def rolling_activation_baseline(
        self,
        window: int | None = None,
    ) -> tuple[dict[int, float], int]:
        """Average recent per-neuron activation, excluding the current tick.

        Returns a sparse {neuron_id: mean_activation} map plus the number of
        historical ticks included in the baseline. Neurons absent from the map
        have an implicit baseline of 0.0 across that window.
        """
        if len(self._history) <= 1:
            return {}, 0

        previous = self._history[:-1]
        if window is None or window <= 0 or window >= len(previous):
            baseline_window = self._baseline_window_count
            if baseline_window == 0:
                return {}, 0
            return (
                {
                    nid: total / baseline_window
                    for nid, total in self._baseline_totals.items()
                },
                baseline_window,
            )

        if window is not None and window > 0:
            previous = previous[-window:]

        baseline_window = len(previous)
        if baseline_window == 0:
            return {}, 0

        totals: dict[int, float] = {}
        for snap in previous:
            if snap.active_values:
                for nid, act in snap.active_values:
                    totals[nid] = totals.get(nid, 0.0) + act
                continue

            for neurons in snap.active_neurons.values():
                for nid, act in neurons:
                    totals[nid] = totals.get(nid, 0.0) + act

        return (
            {
                nid: total / baseline_window
                for nid, total in totals.items()
            },
            baseline_window,
        )

    def push_snapshot(
        self,
        tick: int,
        active_neurons: dict[str, list[tuple[int, float]]],
        total_active: int | None = None,
        region_active_counts: dict[str, int] | None = None,
    ) -> ActivationSnapshot:
        """Create a snapshot from already-collected activation data and store it."""
        active_ids: list[int] = []
        active_values: list[tuple[int, float]] = []
        counts = dict(region_active_counts or {})
        computed_total = 0

        for region_name, neurons in active_neurons.items():
            if not neurons:
                continue
            if region_name not in counts:
                counts[region_name] = len(neurons)
            active_values.extend(neurons)
            active_ids.extend(nid for nid, _ in neurons)
            computed_total += len(neurons)

        snap = ActivationSnapshot(
            tick=tick,
            active_neurons=active_neurons,
            active_values=active_values,
            total_active=total_active if total_active is not None else computed_total,
            active_ids=active_ids,
            region_active_counts=counts,
        )
        self.push(snap)
        return snap

    def push_compact_snapshot(
        self,
        tick: int,
        active_values: list[tuple[int, float]],
        total_active: int | None = None,
        region_active_counts: dict[str, int] | None = None,
    ) -> ActivationSnapshot:
        """Create a snapshot from flat sparse activations and store it."""
        active_ids = [nid for nid, _ in active_values]
        snap = ActivationSnapshot(
            tick=tick,
            active_values=list(active_values),
            total_active=total_active if total_active is not None else len(active_values),
            active_ids=active_ids,
            region_active_counts=dict(region_active_counts or {}),
        )
        self.push(snap)
        return snap

    def push_flat_snapshot(
        self,
        tick: int,
        active_ids: list[int],
        active_vals: list[float],
        total_active: int | None = None,
        region_active_counts: dict[str, int] | None = None,
    ) -> ActivationSnapshot:
        """Create a snapshot from separate flat ID/value arrays (avoids tuple allocation)."""
        snap = ActivationSnapshot(
            tick=tick,
            active_values=list(zip(active_ids, active_vals)) if len(active_ids) <= 500 else [],
            total_active=total_active if total_active is not None else len(active_ids),
            active_ids=list(active_ids),
            region_active_counts=dict(region_active_counts or {}),
        )
        # Store flat arrays for efficient access without tuple allocation
        snap._flat_ids = active_ids
        snap._flat_vals = active_vals
        self.push(snap)
        return snap

    def take_snapshot(self, brain_core) -> ActivationSnapshot:
        """Take a snapshot from the Rust brain and push to history.

        Args:
            brain_core: the Rust brain_core module.

        Returns:
            The new snapshot.
        """
        tick_num = brain_core.get_tick_count()
        all_acts = brain_core.get_all_activations(0.01)
        return self.push_snapshot(tick_num, all_acts)
