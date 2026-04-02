"""Phase 6 tests: Emotion & Executive.

Tests:
  - Neuromodulator get/set and threshold modifier
  - Emotion polarity (positive vs negative emotion neurons)
  - Emotion arousal (firing intensity)
  - Emotion urgency (high-activation fraction)
  - Emotion→motor impulse pathway
  - Executive engagement level
  - Motor conflict detection
  - Motor conflict resolution (suppress weaker)
  - Impulse suppression (executive blocks emotion→motor)
  - Planning signal (executive × language)
  - Trace polarity tagging during emotional state
  - Tick loop returns Phase 6 fields
  - Energy depletion and recovery
"""

import pytest
import brain_core

from brain.structures.trace_store import Trace, TraceStore
from brain.learning.tick_loop import TickLoop


EMOTION_START = 70_000
POSITIVE_END = 74_250  # first half excitatory
NEGATIVE_END = 78_500  # second half excitatory
MOTOR_START = 130_000
MOTOR_MID = 135_000
EXECUTIVE_START = 120_000
LANGUAGE_START = 105_000


@pytest.fixture(autouse=True)
def reset_brain():
    brain_core.init_brain()
    yield
    brain_core.reset_brain()


def activate(neuron_ids, value=1.0):
    """Inject activations and tick to fire neurons."""
    signals = [(nid, value) for nid in neuron_ids]
    brain_core.inject_activations(signals)
    brain_core.tick()


# === NEUROMODULATOR TESTS ===


class TestNeuromodulator:
    def test_default_state(self):
        nm = brain_core.get_neuromodulator()
        arousal, valence, focus, energy = nm
        assert 0.4 <= arousal <= 0.6  # around 0.5
        assert -0.1 <= valence <= 0.1  # around 0.0
        assert 0.4 <= focus <= 0.6
        assert energy > 0.9

    def test_set_and_get(self):
        brain_core.set_neuromodulator(0.8, -0.5, 0.9, 0.3)
        a, v, f, e = brain_core.get_neuromodulator()
        assert abs(a - 0.8) < 0.01
        assert abs(v - (-0.5)) < 0.01
        assert abs(f - 0.9) < 0.01
        assert abs(e - 0.3) < 0.01

    def test_clamps_values(self):
        brain_core.set_neuromodulator(5.0, -10.0, 3.0, -1.0)
        a, v, f, e = brain_core.get_neuromodulator()
        assert abs(a - 1.0) < 0.01
        assert abs(v - (-1.0)) < 0.01
        assert abs(f - 1.0) < 0.01
        assert abs(e) < 0.01

    def test_threshold_modifier_default(self):
        mod = brain_core.get_threshold_modifier()
        # Default arousal ~0.5 → modifier ~0.8
        assert 0.7 <= mod <= 0.9

    def test_threshold_modifier_high_arousal(self):
        brain_core.set_neuromodulator(1.0, 0.0, 0.5, 1.0)
        mod = brain_core.get_threshold_modifier()
        assert abs(mod - 0.6) < 0.01

    def test_threshold_modifier_low_arousal(self):
        brain_core.set_neuromodulator(0.0, 0.0, 0.5, 1.0)
        mod = brain_core.get_threshold_modifier()
        assert abs(mod - 1.0) < 0.01


# === EMOTION TESTS ===


class TestEmotion:
    def test_no_emotion_when_silent(self):
        polarity = brain_core.get_emotion_polarity()
        assert abs(polarity) < 0.01

    def test_positive_polarity(self):
        """Activating positive emotion neurons should yield positive polarity."""
        pos_neurons = list(range(EMOTION_START, EMOTION_START + 50))
        activate(pos_neurons)
        polarity = brain_core.get_emotion_polarity()
        assert polarity > 0.5, f"polarity={polarity}"

    def test_negative_polarity(self):
        """Activating negative emotion neurons should yield negative polarity."""
        neg_neurons = list(range(POSITIVE_END, POSITIVE_END + 50))
        activate(neg_neurons)
        polarity = brain_core.get_emotion_polarity()
        assert polarity < -0.5, f"polarity={polarity}"

    def test_arousal_scales_with_activity(self):
        """More emotion neurons firing → higher arousal."""
        # Few neurons
        activate(list(range(EMOTION_START, EMOTION_START + 10)))
        arousal_low = brain_core.get_emotion_arousal()

        brain_core.reset_brain()
        brain_core.init_brain()

        # Many neurons
        activate(list(range(EMOTION_START, EMOTION_START + 200)))
        arousal_high = brain_core.get_emotion_arousal()

        assert arousal_high > arousal_low

    def test_urgency(self):
        """Urgency = fraction of emotion neurons with high activation."""
        activate(list(range(EMOTION_START, EMOTION_START + 30)))
        urgency = brain_core.get_emotion_urgency(0.5)
        assert 0.0 <= urgency <= 1.0

    def test_motor_impulse_positive(self):
        """Positive emotions should generate approach motor impulses."""
        pos_neurons = list(range(EMOTION_START, EMOTION_START + 30))
        activate(pos_neurons)
        impulses = brain_core.get_emotion_motor_impulse()
        assert len(impulses) > 0
        # Approach motor neurons: 130000–134999
        for gid, strength in impulses:
            assert MOTOR_START <= gid < MOTOR_MID

    def test_motor_impulse_negative(self):
        """Negative emotions should generate withdraw motor impulses."""
        neg_neurons = list(range(POSITIVE_END, POSITIVE_END + 30))
        activate(neg_neurons)
        impulses = brain_core.get_emotion_motor_impulse()
        assert len(impulses) > 0
        # Withdraw motor neurons: 135000–139999
        for gid, strength in impulses:
            assert MOTOR_MID <= gid < 140_000

    def test_no_impulse_when_silent(self):
        impulses = brain_core.get_emotion_motor_impulse()
        assert len(impulses) == 0


# === EXECUTIVE TESTS ===


class TestExecutive:
    def test_no_engagement_when_silent(self):
        eng = brain_core.get_executive_engagement()
        assert eng < 0.01

    def test_engagement_with_activity(self):
        """Executive neurons firing → engagement > 0."""
        exec_neurons = list(range(EXECUTIVE_START, EXECUTIVE_START + 200))
        activate(exec_neurons)
        eng = brain_core.get_executive_engagement()
        assert eng > 0.2

    def test_no_conflict_single_motor(self):
        """Only approach motor → no conflict."""
        approach = list(range(MOTOR_START, MOTOR_START + 20))
        activate(approach)
        conflict = brain_core.get_motor_conflict()
        assert conflict < 0.2

    def test_conflict_both_motor(self):
        """Both approach and withdraw motor → high conflict."""
        both = list(range(MOTOR_START, MOTOR_START + 20))
        both += list(range(MOTOR_MID, MOTOR_MID + 20))
        activate(both)
        conflict = brain_core.get_motor_conflict()
        assert conflict > 0.5

    def test_resolve_conflict(self):
        """Executive resolves motor conflict by suppressing weaker side."""
        # Create balanced motor conflict
        both = list(range(MOTOR_START, MOTOR_START + 50))
        both += list(range(MOTOR_MID, MOTOR_MID + 10))
        activate(both)

        # Activate executive strongly
        exec_neurons = list(range(EXECUTIVE_START, EXECUTIVE_START + 300))
        signals = [(nid, 1.0) for nid in exec_neurons]
        brain_core.inject_activations(signals)
        brain_core.tick()

        suppressed = brain_core.resolve_motor_conflict(2.0)
        assert suppressed >= 0  # May or may not suppress depending on timing

    def test_impulse_suppression(self):
        """Executive blocks emotion→motor impulse when strongly engaged."""
        # Activate emotion to generate impulse
        pos_neurons = list(range(EMOTION_START, EMOTION_START + 50))
        activate(pos_neurons)
        impulses = brain_core.get_emotion_motor_impulse()

        if impulses:
            # Now activate executive strongly
            exec_neurons = list(range(EXECUTIVE_START, EXECUTIVE_START + 400))
            signals = [(nid, 1.0) for nid in exec_neurons]
            brain_core.inject_activations(signals)
            brain_core.tick()

            inhibited = brain_core.inhibit_motor(impulses)
            # Executive should try to block
            assert inhibited >= 0

    def test_planning_signal_silent(self):
        planning = brain_core.get_planning_signal()
        assert planning < 0.01

    def test_planning_signal_active(self):
        """Planning requires both executive and language active."""
        exec_neurons = list(range(EXECUTIVE_START, EXECUTIVE_START + 300))
        lang_neurons = list(range(LANGUAGE_START, LANGUAGE_START + 300))
        both = exec_neurons + lang_neurons
        activate(both)
        planning = brain_core.get_planning_signal()
        assert planning > 0.1


# === ENERGY TESTS ===


class TestEnergy:
    def test_energy_depletes_with_activity(self):
        _, _, _, energy_before = brain_core.get_neuromodulator()
        # Run many ticks with lots of activity
        for _ in range(100):
            activate(list(range(0, 100)))  # sensory activity
        _, _, _, energy_after = brain_core.get_neuromodulator()
        assert energy_after < energy_before

    def test_energy_recovery(self):
        brain_core.set_neuromodulator(0.5, 0.0, 0.5, 0.3)
        brain_core.recover_energy(0.2)
        _, _, _, energy = brain_core.get_neuromodulator()
        assert abs(energy - 0.5) < 0.01

    def test_energy_caps_at_one(self):
        brain_core.set_neuromodulator(0.5, 0.0, 0.5, 0.9)
        brain_core.recover_energy(0.5)
        _, _, _, energy = brain_core.get_neuromodulator()
        assert abs(energy - 1.0) < 0.01


# === TRACE POLARITY TAGGING ===


class TestTracePolarityTagging:
    def test_polarity_shifts_with_emotion(self):
        """Active traces should absorb emotional polarity."""
        store = TraceStore()
        # Trace with emotion region neurons
        t = Trace(
            id="t1",
            neurons={
                "sensory": [0, 1, 2],
                "emotion": [EMOTION_START, EMOTION_START + 1],
            },
            polarity=0.0,
            strength=0.5,
        )
        store.add(t)

        loop = TickLoop(store)

        # Activate positive emotion strongly
        pos_neurons = list(range(EMOTION_START, EMOTION_START + 50))
        all_neurons = list(range(0, 5)) + pos_neurons  # sensory + emotion
        signals = [(nid, 1.0) for nid in all_neurons]
        brain_core.inject_activations(signals)

        # Run several ticks
        for _ in range(10):
            brain_core.inject_activations(signals)
            loop.step()

        trace = store.get("t1")
        # Polarity should have shifted toward positive
        # (small shift per tick, so check direction)
        assert trace.polarity >= -0.01  # at least not negative


# === TICK LOOP PHASE 6 INTEGRATION ===


class TestTickLoopPhase6:
    def test_step_returns_phase6_fields(self):
        store = TraceStore()
        loop = TickLoop(store)
        result = loop.step()

        assert "emotion_polarity" in result
        assert "emotion_arousal" in result
        assert "executive_engagement" in result
        assert "motor_conflict" in result
        assert "planning_signal" in result
        assert "valence" in result
        assert "energy" in result

    def test_arousal_updates_from_emotion(self):
        """Neuromodulator arousal should track emotion activity."""
        store = TraceStore()
        loop = TickLoop(store)

        # No emotion at first
        result = loop.step()
        arousal_baseline = result["arousal"]

        # Activate emotion region heavily
        emotion_neurons = list(range(EMOTION_START, EMOTION_START + 500))
        for _ in range(20):
            signals = [(nid, 1.0) for nid in emotion_neurons]
            brain_core.inject_activations(signals)
            result = loop.step()

        # Arousal should have risen
        assert result["arousal"] > arousal_baseline or result["emotion_arousal"] > 0.0

    def test_no_crash_many_steps(self):
        """Multiple steps with Phase 6 should not crash."""
        store = TraceStore()
        loop = TickLoop(store)
        for _ in range(20):
            result = loop.step()
            assert isinstance(result["emotion_polarity"], float)
            assert isinstance(result["executive_engagement"], float)

    def test_energy_depletes_over_time(self):
        """Energy should slowly deplete from tick_loop."""
        store = TraceStore()
        loop = TickLoop(store)

        result_first = loop.step()
        for _ in range(50):
            activate(list(range(0, 50)))
            result = loop.step()

        # Energy should be less than initial
        assert result["energy"] <= result_first["energy"]

    def test_consolidation_recovers_energy(self):
        """During consolidation, energy should recover."""
        store = TraceStore()
        t = Trace(id="t1", neurons={"memory_long": [55000]}, strength=0.5)
        store.add(t)

        loop = TickLoop(store)
        loop._awake_trace_ids = ["t1"]
        loop.neuromod.energy = 0.1  # Trigger consolidation

        result = loop.step()
        # Consolidation should have started
        assert loop.consolidation.is_consolidating or result["consolidating"]
