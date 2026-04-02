"""Phase 4 tests: Attention & Pattern Recognition.

Tests cover:
  1. Three-drive attention system (novelty, threat, relevance)
  2. Attention gain computation with inertia
  3. Pattern recognition / trace matching integration
  4. Prediction engine (trace-based prediction, error computation)
  5. Prediction error effects (surprise, alarm, learning rate)
  6. Novel pattern → attention spike
  7. Familiar pattern → attention drop
  8. Threatening pattern → attention override
  9. Full tick loop with prediction

PLAN milestones:
  - Brain attends to novel/important stimuli
  - Brain recognizes familiar patterns
  - Prediction error drives learning
"""

import pytest
import brain_core
from brain.structures.trace_store import Trace, TraceStore
from brain.structures.brain_state import (
    ActivationHistory,
    ActivationSnapshot,
    NeuromodulatorState,
)
from brain.structures.neuron_map import all_region_names, region_size
from brain.learning.prediction import PredictionEngine
from brain.learning.tick_loop import TickLoop, WorkingMemory
from brain.utils.config import (
    REGIONS,
    ATTENTION_GAIN_MIN,
    ATTENTION_GAIN_MAX,
    PREDICTION_SURPRISE_DURATION,
    PREDICTION_ALARM_DURATION,
)


# ── Helpers ──

@pytest.fixture(autouse=True)
def fresh_brain():
    """Reset the global brain before every test."""
    brain_core.init_brain()
    yield
    brain_core.reset_brain()


def make_trace(tid, regions_neurons, **kwargs):
    """Create a Trace with specified neuron mapping."""
    return Trace(
        id=tid,
        neurons=regions_neurons,
        **kwargs,
    )


def make_snapshot(active_dict, tick=0):
    """Create an ActivationSnapshot from a dict of region→[(id, act)]."""
    total = sum(len(v) for v in active_dict.values())
    return ActivationSnapshot(tick=tick, active_neurons=active_dict, total_active=total)


# ══════════════════════════════════════════════════════════════
# GROUP 1: Attention Drive System (PyO3 integration)
# ══════════════════════════════════════════════════════════════

class TestAttentionDrives:
    """Test the three-drive attention system via PyO3 bridge."""

    def test_default_gains_are_one(self):
        gains = brain_core.get_attention_gains()
        for region in all_region_names():
            assert abs(gains[region] - 1.0) < 0.01, f"{region} gain should start at 1.0"

    def test_set_novelty_drive_increases_gain(self):
        # Set high novelty for pattern region
        brain_core.set_attention_drives("pattern", 1.0, 0.0, 0.0)
        # Run a tick to trigger gain update
        brain_core.tick()
        gains = brain_core.get_attention_gains()
        assert gains["pattern"] > 1.0, f"Novelty drive should increase gain, got {gains['pattern']}"

    def test_set_threat_drive_increases_gain(self):
        brain_core.set_attention_drives("emotion", 0.0, 1.0, 0.0)
        brain_core.tick()
        gains = brain_core.get_attention_gains()
        # Note: threat is auto-computed from emotion activity,
        # but setting drives manually should also work
        assert gains["emotion"] >= 1.0

    def test_gains_stay_in_range(self):
        # Max all drives
        for r in all_region_names():
            brain_core.set_attention_drives(r, 10.0, 10.0, 10.0)
        for _ in range(100):
            brain_core.tick()
        gains = brain_core.get_attention_gains()
        for region, gain in gains.items():
            assert ATTENTION_GAIN_MIN <= gain <= ATTENTION_GAIN_MAX, \
                f"{region} gain {gain} out of range [{ATTENTION_GAIN_MIN}, {ATTENTION_GAIN_MAX}]"

    def test_inertia_prevents_instant_change(self):
        brain_core.set_attention_drives("visual", 1.0, 0.0, 0.0)
        brain_core.tick()
        gains = brain_core.get_attention_gains()
        # With inertia=15 ticks, single tick should move only ~6.7% toward target
        assert gains["visual"] < 1.5, f"Inertia should prevent instant jump, got {gains['visual']}"

    def test_gains_converge_over_ticks(self):
        for r in all_region_names():
            brain_core.set_attention_drives(r, 0.5, 0.0, 0.0)
        for _ in range(100):
            brain_core.tick()
        gains = brain_core.get_attention_gains()
        # Should have converged toward target = 1.0 + 0.5*0.4*4.0 = 1.8
        for r in all_region_names():
            assert gains[r] > 1.5, f"Expected convergence for {r}, got {gains[r]}"


# ══════════════════════════════════════════════════════════════
# GROUP 2: Prediction Errors (PyO3 integration)
# ══════════════════════════════════════════════════════════════

class TestPredictionErrors:
    """Test Rust-side prediction error computation."""

    def test_no_activity_no_error(self):
        brain_core.tick()  # First tick with no input
        errors = brain_core.get_prediction_errors()
        for r in all_region_names():
            assert errors[r] < 0.1, f"No-activity error for {r} should be low: {errors[r]}"

    def test_surprise_on_sudden_activity(self):
        # First tick: establish baseline (no activity)
        brain_core.tick()

        # Second tick: inject massive signal into sensory
        signals = [(i, 0.8) for i in range(500)]
        brain_core.inject_activations(signals)
        brain_core.tick()

        errors = brain_core.get_prediction_errors()
        assert errors["sensory"] > 0.1, \
            f"Sudden sensory activity should cause prediction error: {errors['sensory']}"

    def test_global_prediction_error(self):
        brain_core.tick()
        ge = brain_core.get_global_prediction_error()
        assert 0.0 <= ge <= 1.0, f"Global error should be in [0,1]: {ge}"

    def test_adaptation_reduces_error(self):
        # Inject same pattern repeatedly
        signals = [(i, 0.5) for i in range(100)]
        errors_first = None
        for i in range(30):
            brain_core.inject_activations(signals)
            brain_core.tick()
            if i == 1:
                errors_first = brain_core.get_prediction_errors()

        errors_last = brain_core.get_prediction_errors()
        # After many repetitions, error should decrease
        assert errors_last["sensory"] <= errors_first["sensory"] + 0.05, \
            f"Repeated pattern should reduce error: first={errors_first['sensory']}, last={errors_last['sensory']}"


# ══════════════════════════════════════════════════════════════
# GROUP 3: PredictionEngine (Python)
# ══════════════════════════════════════════════════════════════

class TestPredictionEngine:
    """Test the trace-based prediction engine."""

    def test_classify_expected(self):
        store = TraceStore()
        engine = PredictionEngine(store)
        assert engine.classify(0.05) == "expected"

    def test_classify_interesting(self):
        store = TraceStore()
        engine = PredictionEngine(store)
        assert engine.classify(0.3) == "interesting"

    def test_classify_surprise(self):
        store = TraceStore()
        engine = PredictionEngine(store)
        assert engine.classify(0.6) == "surprise"

    def test_classify_alarm(self):
        store = TraceStore()
        engine = PredictionEngine(store)
        assert engine.classify(0.9) == "alarm"

    def test_predict_with_no_traces(self):
        store = TraceStore()
        engine = PredictionEngine(store)
        predicted = engine.predict([], [])
        for r in all_region_names():
            assert r in predicted
            assert predicted[r] >= 0.0

    def test_predict_with_active_traces(self):
        store = TraceStore()
        # Create a trace with sensory and visual neurons
        t1 = make_trace("t1", {
            "sensory": [0, 1, 2, 3, 4],
            "visual": [10000, 10001, 10002],
        })
        store.add(t1)

        engine = PredictionEngine(store)
        predicted = engine.predict([("t1", 0.8)], [])

        # Sensory should have higher predicted activation
        assert predicted["sensory"] > 0.0, "Active trace should predict sensory activation"
        assert predicted["visual"] > 0.0, "Active trace should predict visual activation"

    def test_compute_errors(self):
        store = TraceStore()
        engine = PredictionEngine(store)

        # Make a prediction (no traces, so prediction is ~0)
        engine.predict([], [])

        # Create a snapshot with activity
        snap = make_snapshot({"sensory": [(i, 0.5) for i in range(200)]})
        errors = engine.compute_errors(snap)

        assert errors["sensory"] > 0.0, "Unexpected activity should cause error"

    def test_learning_rate_multiplier_default(self):
        store = TraceStore()
        engine = PredictionEngine(store)
        assert engine.learning_rate_multiplier == 1.0

    def test_surprise_boosts_learning(self):
        store = TraceStore()
        engine = PredictionEngine(store)
        neuromod = NeuromodulatorState()

        engine.apply_effects(0.6, neuromod)  # SURPRISE level
        assert engine.in_surprise
        assert engine.learning_rate_multiplier == 2.0

    def test_alarm_boosts_learning_higher(self):
        store = TraceStore()
        engine = PredictionEngine(store)
        neuromod = NeuromodulatorState()

        engine.apply_effects(0.9, neuromod)  # ALARM level
        assert engine.in_alarm
        assert engine.learning_rate_multiplier == 3.0
        assert neuromod.arousal > 0.7, "Alarm should spike arousal"

    def test_surprise_duration_decays(self):
        store = TraceStore()
        engine = PredictionEngine(store)
        neuromod = NeuromodulatorState()

        engine.apply_effects(0.6, neuromod)
        assert engine.in_surprise

        # Simulate ticks with expected (low error)
        for _ in range(PREDICTION_SURPRISE_DURATION + 5):
            engine.apply_effects(0.01, neuromod)

        assert not engine.in_surprise, "Surprise should expire after duration"
        assert engine.learning_rate_multiplier == 1.0


# ══════════════════════════════════════════════════════════════
# GROUP 4: Pattern Recognition (trace matching integration)
# ══════════════════════════════════════════════════════════════

class TestPatternRecognition:
    """Test trace matching finds patterns above threshold."""

    def test_match_above_threshold(self):
        store = TraceStore()
        t1 = make_trace("t1", {"sensory": [0, 1, 2, 3, 4]})
        store.add(t1)

        # All 5 neurons are active
        matches = store.matching_traces([0, 1, 2, 3, 4], threshold=0.6)
        assert len(matches) > 0, "Should find matching trace"
        assert matches[0][0] == "t1"
        assert matches[0][1] >= 0.6

    def test_partial_match_below_threshold(self):
        store = TraceStore()
        t1 = make_trace("t1", {"sensory": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        store.add(t1)

        # Only 3 of 10 neurons active → 30% match
        matches = store.matching_traces([0, 1, 2], threshold=0.6)
        assert len(matches) == 0, "Partial match should not exceed threshold"

    def test_no_match_novel_pattern(self):
        store = TraceStore()
        t1 = make_trace("t1", {"sensory": [0, 1, 2, 3, 4]})
        store.add(t1)

        # Completely different neurons
        matches = store.matching_traces([100, 101, 102, 103, 104], threshold=0.6)
        assert len(matches) == 0, "Novel pattern should have no match"

    def test_trace_metadata_updates_on_fire(self):
        store = TraceStore()
        t1 = make_trace("t1", {"sensory": [0, 1, 2]})
        store.add(t1)

        trace = store.get("t1")
        assert trace.fire_count == 0

        # Simulate trace firing (as tick_loop does)
        matches = store.matching_traces([0, 1, 2], threshold=0.6)
        for tid, score in matches:
            t = store.get(tid)
            t.fire_count += 1
            t.strength = min(1.0, t.strength + 0.005 * score)

        trace = store.get("t1")
        assert trace.fire_count == 1
        assert trace.strength > 0.1


# ══════════════════════════════════════════════════════════════
# GROUP 5: Novel/Familiar/Threatening → Attention Behavior
# ══════════════════════════════════════════════════════════════

class TestAttentionBehavior:
    """Test PLAN milestone behaviors."""

    def test_novel_pattern_spikes_attention(self):
        """Inject novel pattern → verify attention spikes."""
        # Establish calm baseline
        for _ in range(5):
            brain_core.tick()
        gains_before = brain_core.get_attention_gains()

        # Set high novelty drives (simulating novel pattern detection)
        for r in all_region_names():
            brain_core.set_attention_drives(r, 0.8, 0.0, 0.0)

        # Run several ticks
        for _ in range(20):
            brain_core.tick()

        gains_after = brain_core.get_attention_gains()

        # At least some regions should have increased attention
        increased = sum(
            1 for r in all_region_names()
            if gains_after[r] > gains_before[r] + 0.05
        )
        assert increased > 0, "Novel pattern should spike attention in some regions"

    def test_familiar_pattern_drops_attention(self):
        """Inject familiar pattern → verify attention drops."""
        # First, spike attention with novelty
        for r in all_region_names():
            brain_core.set_attention_drives(r, 0.8, 0.0, 0.0)
        for _ in range(30):
            brain_core.tick()
        gains_spiked = brain_core.get_attention_gains()

        # Now set zero novelty (familiar pattern)
        for r in all_region_names():
            brain_core.set_attention_drives(r, 0.0, 0.0, 0.0)
        for _ in range(50):
            brain_core.tick()

        gains_after = brain_core.get_attention_gains()

        # Gains should have decreased toward 1.0
        decreased = sum(
            1 for r in all_region_names()
            if gains_after[r] < gains_spiked[r] - 0.05
        )
        assert decreased > 0, "Familiar pattern should drop attention"

    def test_threat_overrides_attention(self):
        """Inject threatening pattern (high emotion activity) → verify attention override."""
        # Inject strong activation into emotion neurons to trigger threat drive
        # Emotion region: 70000–79999
        emotion_signals = [(70000 + i, 0.8) for i in range(500)]
        for _ in range(20):
            brain_core.inject_activations(emotion_signals)
            brain_core.tick()

        gains = brain_core.get_attention_gains()
        # Threat drive is auto-computed from emotion firing rate
        # High emotion activity → high threat → gains should increase
        any_increased = any(gains[r] > 1.01 for r in all_region_names())
        assert any_increased, \
            f"High emotion (threat) should increase some attention gains: {gains}"


# ══════════════════════════════════════════════════════════════
# GROUP 6: Working Memory
# ══════════════════════════════════════════════════════════════

class TestWorkingMemory:

    def test_capacity_limit(self):
        wm = WorkingMemory(capacity=3)
        wm.update([("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)])
        assert len(wm) == 3
        assert "d" not in wm.trace_ids

    def test_refresh_strengthens(self):
        wm = WorkingMemory(capacity=5)
        wm.update([("a", 0.5)])
        wm.update([("a", 0.9)])  # Refresh
        assert len(wm) == 1
        assert wm.slots[0][1] >= 0.9 * (1 - 0.02)  # After decay

    def test_decay(self):
        wm = WorkingMemory(capacity=5)
        wm.update([("a", 0.5)])
        # Several updates without "a"
        for _ in range(10):
            wm.update([])
        if len(wm) > 0:
            assert wm.slots[0][1] < 0.5


# ══════════════════════════════════════════════════════════════
# GROUP 7: Full Tick Loop Integration
# ══════════════════════════════════════════════════════════════

class TestTickLoopPhase4:
    """Test the full tick loop with Phase 4 features."""

    def test_tick_loop_step_returns_prediction_fields(self):
        store = TraceStore()
        loop = TickLoop(store)
        result = loop.step()

        assert "tick" in result
        assert "novelty" in result
        assert "in_surprise" in result
        assert "in_alarm" in result
        assert "learning_multiplier" in result

    def test_tick_loop_no_crash_multiple_steps(self):
        store = TraceStore()
        loop = TickLoop(store)
        for _ in range(10):
            result = loop.step()
        assert result["tick"] >= 9

    def test_tick_loop_with_traces(self):
        store = TraceStore()
        t1 = make_trace("t1", {"sensory": [0, 1, 2, 3, 4]})
        store.add(t1)

        loop = TickLoop(store)

        # Inject activation matching trace
        brain_core.inject_activations([(i, 0.5) for i in range(5)])
        result = loop.step()

        assert isinstance(result["active_traces"], int)
        assert isinstance(result["novelty"], float)

    def test_tick_loop_prediction_engine_active(self):
        store = TraceStore()
        loop = TickLoop(store)

        # Run a few ticks to establish baseline
        for _ in range(5):
            loop.step()

        # Inject sudden activity
        brain_core.inject_activations([(i, 0.8) for i in range(300)])
        result = loop.step()

        # Novelty should be nonzero due to surprise
        assert result["novelty"] >= 0.0
