"""Novelty scoring and prediction error.

Novelty drives attention and learning rate. The system predicts expected
activation per region, compares to actual, and produces:
  - prediction_error per region
  - global novelty signal
  - attention/learning modulation effects

Effects:
  error > 0.8  → ALARM       (arousal spike, hypervigilance)
  error > 0.5  → SURPRISE    (attention spike, learning rate x2)
  error 0.1-0.5 → INTERESTING (moderate attention, normal learning)
  error < 0.1  → EXPECTED    (attention drops, minimal learning)
"""

from __future__ import annotations

from brain.structures.brain_state import ActivationSnapshot, NeuromodulatorState
from brain.structures.neuron_map import all_region_names, region_size
from brain.utils.config import (
    PREDICTION_ALARM_THRESHOLD,
    PREDICTION_BORING_THRESHOLD,
    PREDICTION_SURPRISE_THRESHOLD,
)


class NoveltyTracker:
    """Tracks expected activation and computes prediction error."""

    def __init__(self):
        # Exponential moving average of active counts per region
        self._ema_active: dict[str, float] = {r: 0.0 for r in all_region_names()}
        self._alpha = 0.05  # EMA smoothing factor
        self._tick_count = 0

    def update(self, snapshot: ActivationSnapshot) -> dict[str, float]:
        """Update predictions with actual data, return prediction errors.

        Args:
            snapshot: current tick's activation snapshot.

        Returns:
            Dict of region_name → prediction_error (0.0 - 1.0).
        """
        errors: dict[str, float] = {}
        self._tick_count += 1

        for region in all_region_names():
            actual = len(snapshot.active_neurons.get(region, []))
            size = region_size(region)
            actual_rate = actual / size if size > 0 else 0.0

            expected_rate = self._ema_active[region]

            # Prediction error = absolute difference in firing rate
            error = abs(actual_rate - expected_rate)
            # Normalize to 0-1 range (cap at 1.0)
            error = min(error * 20.0, 1.0)  # Scale since rates are very small
            errors[region] = error

            # Update EMA
            self._ema_active[region] = (
                self._alpha * actual_rate + (1 - self._alpha) * expected_rate
            )

        return errors

    def global_novelty(self, errors: dict[str, float]) -> float:
        """Compute a single global novelty score from per-region errors."""
        if not errors:
            return 0.0
        return sum(errors.values()) / len(errors)

    def classify_error(self, error: float) -> str:
        """Classify an error level into a category."""
        if error > PREDICTION_ALARM_THRESHOLD:
            return "alarm"
        elif error > PREDICTION_SURPRISE_THRESHOLD:
            return "surprise"
        elif error > PREDICTION_BORING_THRESHOLD:
            return "interesting"
        else:
            return "expected"

    def modulate_neuromodulators(
        self,
        novelty: float,
        neuromod: NeuromodulatorState,
    ) -> None:
        """Adjust neuromodulators based on novelty signal.

        Mutates the neuromod state in-place.
        """
        classification = self.classify_error(novelty)

        if classification == "alarm":
            neuromod.arousal = min(1.0, neuromod.arousal + 0.3)
            neuromod.focus = min(1.0, neuromod.focus + 0.2)
        elif classification == "surprise":
            neuromod.arousal = min(1.0, neuromod.arousal + 0.1)
            neuromod.focus = min(1.0, neuromod.focus + 0.1)
        elif classification == "expected":
            neuromod.arousal = max(0.0, neuromod.arousal - 0.05)
            neuromod.focus = max(0.0, neuromod.focus - 0.03)
        # "interesting" → no modulation change (steady state)

        neuromod.clamp()
