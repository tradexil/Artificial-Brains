"""Trace-based prediction engine and prediction error effects.

Goes beyond simple EMA prediction (novelty.py) by using active traces
and working memory context to predict what SHOULD happen next.
When prediction is wrong, that drives attention and learning.

Prediction error effects:
  ALARM (>0.8):       arousal spike, threshold drop, hypervigilance 200 ticks
  SURPRISE (>0.5):    attention spike, learning rate ×2 for 50 ticks
  INTERESTING (0.1-0.5): moderate attention, normal learning
  EXPECTED (<0.1):    attention drops, minimal learning
"""

from __future__ import annotations

from collections import defaultdict

from brain.structures.brain_state import (
    ActivationHistory,
    ActivationSnapshot,
    NeuromodulatorState,
)
from brain.structures.trace_store import TraceStore
from brain.structures.neuron_map import all_region_names, region_size
from brain.utils.config import (
    PREDICTION_ALARM_THRESHOLD,
    PREDICTION_BORING_THRESHOLD,
    PREDICTION_SURPRISE_THRESHOLD,
    PREDICTION_SURPRISE_DURATION,
    PREDICTION_ALARM_DURATION,
)


class PredictionEngine:
    """Trace-based next-tick prediction and error computation.

    Instead of just tracking firing rate EMA, this engine looks at
    which traces are active and predicts what regions should activate
    next based on trace co-occurrence history.
    """

    def __init__(self, trace_store: TraceStore):
        self.trace_store = trace_store
        # Per-region predicted activation counts (from trace analysis)
        self._predicted: dict[str, float] = {r: 0.0 for r in all_region_names()}
        # EMA of actual rates per region (for baseline prediction)
        self._ema_rates: dict[str, float] = {r: 0.0 for r in all_region_names()}
        self._alpha = 0.05
        # Active effect timers
        self._surprise_remaining: int = 0
        self._alarm_remaining: int = 0

    def predict(
        self,
        active_traces: list[tuple[str, float]],
        working_memory_ids: list[str],
    ) -> dict[str, float]:
        """Predict next-tick activation per region.

        Uses currently active traces to predict which regions should
        fire next, based on co-trace relationships and trace neuron
        distributions.

        Args:
            active_traces: list of (trace_id, match_score)
            working_memory_ids: trace IDs currently in working memory

        Returns:
            Dict of region_name → predicted_activation_rate (0.0–1.0)
        """
        # Accumulate predicted neuron counts per region
        region_predicted: dict[str, float] = defaultdict(float)

        # Traces that are active contribute prediction for their co-traces
        seen_traces: set[str] = set()
        for tid, score in active_traces:
            seen_traces.add(tid)
            trace = self.trace_store.get(tid)
            if trace is None:
                continue

            # Active trace predicts activation in its own regions
            for region, neurons in trace.neurons.items():
                size = region_size(region)
                if size > 0:
                    region_predicted[region] += (len(neurons) / size) * score

            # Co-traces predict activation too (weaker signal)
            for co_id in trace.co_traces:
                if co_id in seen_traces:
                    continue
                co_trace = self.trace_store.get(co_id)
                if co_trace is None:
                    continue
                for region, neurons in co_trace.neurons.items():
                    size = region_size(region)
                    if size > 0:
                        region_predicted[region] += (len(neurons) / size) * score * 0.3

        # Working memory traces also contribute prediction (weaker)
        for wm_id in working_memory_ids:
            if wm_id in seen_traces:
                continue
            wm_trace = self.trace_store.get(wm_id)
            if wm_trace is None:
                continue
            for region, neurons in wm_trace.neurons.items():
                size = region_size(region)
                if size > 0:
                    region_predicted[region] += (len(neurons) / size) * 0.2

        # Combine trace-based prediction with EMA baseline
        predicted: dict[str, float] = {}
        for region in all_region_names():
            trace_pred = min(region_predicted.get(region, 0.0), 1.0)
            ema_pred = self._ema_rates.get(region, 0.0)
            # Weighted: 60% trace-based, 40% EMA baseline
            predicted[region] = 0.6 * trace_pred + 0.4 * ema_pred

        self._predicted = predicted
        return predicted

    def compute_errors(
        self, snapshot: ActivationSnapshot
    ) -> dict[str, float]:
        """Compute prediction error per region.

        Args:
            snapshot: actual activation after tick

        Returns:
            Dict of region_name → error (0.0–1.0)
        """
        errors: dict[str, float] = {}

        for region in all_region_names():
            actual_count = len(snapshot.active_neurons.get(region, []))
            size = region_size(region)
            actual_rate = actual_count / size if size > 0 else 0.0

            predicted = self._predicted.get(region, 0.0)

            # Error: absolute difference, scaled
            error = min(abs(actual_rate - predicted) * 20.0, 1.0)
            errors[region] = error

            # Update EMA
            self._ema_rates[region] = (
                self._alpha * actual_rate
                + (1.0 - self._alpha) * self._ema_rates.get(region, 0.0)
            )

        return errors

    def global_error(self, errors: dict[str, float]) -> float:
        """Mean prediction error across all regions."""
        if not errors:
            return 0.0
        return sum(errors.values()) / len(errors)

    def classify(self, error: float) -> str:
        """Classify a prediction error into a category."""
        if error > PREDICTION_ALARM_THRESHOLD:
            return "alarm"
        elif error > PREDICTION_SURPRISE_THRESHOLD:
            return "surprise"
        elif error > PREDICTION_BORING_THRESHOLD:
            return "interesting"
        else:
            return "expected"

    def apply_effects(
        self,
        global_error: float,
        neuromod: NeuromodulatorState,
    ) -> None:
        """Apply prediction error effects to neuromodulator state.

        Also manages surprise/alarm duration timers.
        """
        classification = self.classify(global_error)

        if classification == "alarm":
            neuromod.arousal = min(1.0, neuromod.arousal + 0.3)
            neuromod.focus = min(1.0, neuromod.focus + 0.2)
            neuromod.valence = max(-1.0, neuromod.valence - 0.1)
            self._alarm_remaining = PREDICTION_ALARM_DURATION
            self._surprise_remaining = PREDICTION_SURPRISE_DURATION
        elif classification == "surprise":
            neuromod.arousal = min(1.0, neuromod.arousal + 0.15)
            neuromod.focus = min(1.0, neuromod.focus + 0.1)
            self._surprise_remaining = PREDICTION_SURPRISE_DURATION
        elif classification == "interesting":
            # Proportional arousal for interesting stimuli
            neuromod.arousal = min(1.0, neuromod.arousal + 0.05 * global_error)
        elif classification == "expected":
            neuromod.arousal = max(0.0, neuromod.arousal - 0.05)
            neuromod.focus = max(0.0, neuromod.focus - 0.03)

        # Decay timers
        if self._surprise_remaining > 0:
            self._surprise_remaining -= 1
        if self._alarm_remaining > 0:
            self._alarm_remaining -= 1

        neuromod.clamp()

    @property
    def in_surprise(self) -> bool:
        """Is the system currently in a surprise state (boosted learning)?"""
        return self._surprise_remaining > 0

    @property
    def in_alarm(self) -> bool:
        """Is the system in alarm/hypervigilance?"""
        return self._alarm_remaining > 0

    @property
    def learning_rate_multiplier(self) -> float:
        """Current learning rate multiplier from prediction effects."""
        if self._alarm_remaining > 0:
            return 3.0
        elif self._surprise_remaining > 0:
            return 2.0
        else:
            return 1.0
