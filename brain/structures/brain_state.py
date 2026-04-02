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


@dataclass
class ActivationSnapshot:
    """Snapshot of neuron activations for one tick, per region.

    Used by learning modules to detect co-activation patterns.
    Stores only active neurons (sparse) to keep memory low.
    """

    tick: int
    # region_name → list of (global_neuron_id, activation_value)
    active_neurons: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    total_active: int = 0

    def all_active_ids(self) -> list[int]:
        """Flat list of all active neuron IDs across regions."""
        ids = []
        for neurons in self.active_neurons.values():
            ids.extend(nid for nid, _ in neurons)
        return ids

    def active_set(self) -> set[int]:
        """Set of all active neuron IDs (for O(1) membership check)."""
        s = set()
        for neurons in self.active_neurons.values():
            for nid, _ in neurons:
                s.add(nid)
        return s

    def activation_of(self, neuron_id: int) -> float:
        """Get activation for a specific neuron, 0.0 if not active."""
        for neurons in self.active_neurons.values():
            for nid, act in neurons:
                if nid == neuron_id:
                    return act
        return 0.0


class ActivationHistory:
    """Rolling window of recent activation snapshots.

    Keeps the last N ticks for Hebbian co-activation detection.
    """

    def __init__(self, window: int = 3):
        self.window = window
        self._history: list[ActivationSnapshot] = []

    def push(self, snapshot: ActivationSnapshot) -> None:
        self._history.append(snapshot)
        if len(self._history) > self.window:
            self._history.pop(0)

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
            for neurons in snap.active_neurons.values():
                for nid, act in neurons:
                    if nid not in result or act > result[nid]:
                        result[nid] = act
        return result

    def take_snapshot(self, brain_core) -> ActivationSnapshot:
        """Take a snapshot from the Rust brain and push to history.

        Args:
            brain_core: the Rust brain_core module.

        Returns:
            The new snapshot.
        """
        tick_num = brain_core.get_tick_count()
        all_acts = brain_core.get_all_activations(0.01)
        total = sum(len(v) for v in all_acts.values())
        snap = ActivationSnapshot(
            tick=tick_num,
            active_neurons=all_acts,
            total_active=total,
        )
        self.push(snap)
        return snap
