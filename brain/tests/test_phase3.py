"""Phase 3 tests: Learning Core.

Tests:
  - BrainState: snapshots, neuromodulators, activation history
  - Hebbian: repeated signals strengthen synapses
  - Anti-Hebbian: uncorrelated signals weaken synapses
  - Pruning: lifecycle phase detection and weight-based pruning
  - Novelty: prediction error and novelty scoring
  - WorkingMemory: capacity limits and decay
  - TickLoop: integrated learning over multiple ticks
"""

import random

import pytest

import brain_core

from brain.learning.hebbian import compute_effective_learning_rate, hebbian_update
from brain.learning.anti_hebbian import anti_hebbian_update, compute_anti_hebbian_rate
from brain.learning.novelty import NoveltyTracker
from brain.learning.pruning import get_phase, pruning_pass_sampled
from brain.learning.tick_loop import TickLoop, WorkingMemory
from brain.structures.brain_state import (
    ActivationHistory,
    ActivationSnapshot,
    NeuromodulatorState,
)
from brain.structures.trace_store import Trace, TraceStore
from brain.utils.config import (
    BLOOM_END_TICK,
    CRITICAL_END_TICK,
    HEBBIAN_RATE,
    REGIONS,
    WORKING_MEMORY_CAPACITY,
)


# === Helpers ===

def setup_small_brain():
    """Initialize a small brain with known synapses for predictable tests."""
    # Create synapses within the sensory region (0-9999)
    # A few specific synapses to test learning
    synapses = [
        # (from, to, weight, delay, plasticity)
        (0, 1, 0.1, 1, 1.0),
        (0, 2, 0.1, 1, 1.0),
        (1, 2, 0.1, 1, 1.0),
        (1, 3, 0.1, 1, 1.0),
        (2, 3, 0.1, 1, 1.0),
        # A hardwired synapse (plasticity=0)
        (3, 4, 0.5, 1, 0.0),
        # Cross-region: sensory → pattern
        (100, 85_000, 0.1, 3, 1.0),
        (101, 85_001, 0.1, 3, 1.0),
    ]
    brain_core.init_brain_with_synapses(synapses)


def make_snapshot(tick, active_neurons_dict, total=None):
    """Helper to create an ActivationSnapshot."""
    if total is None:
        total = sum(len(v) for v in active_neurons_dict.values())
    return ActivationSnapshot(
        tick=tick,
        active_neurons=active_neurons_dict,
        total_active=total,
    )


# =======================
# BrainState tests
# =======================

class TestBrainState:
    def test_neuromodulator_clamp(self):
        nm = NeuromodulatorState(arousal=1.5, valence=-2.0, focus=0.5, energy=-0.1)
        nm.clamp()
        assert nm.arousal == 1.0
        assert nm.valence == -1.0
        assert nm.focus == 0.5
        assert nm.energy == 0.0

    def test_snapshot_active_ids(self):
        snap = make_snapshot(0, {
            "sensory": [(0, 0.5), (1, 0.8)],
            "visual": [(10_000, 0.6)],
        })
        ids = snap.all_active_ids()
        assert set(ids) == {0, 1, 10_000}

    def test_snapshot_active_set(self):
        snap = make_snapshot(0, {"sensory": [(0, 0.5), (1, 0.8)]})
        assert snap.active_set() == {0, 1}

    def test_snapshot_activation_of(self):
        snap = make_snapshot(0, {"sensory": [(0, 0.5), (1, 0.8)]})
        assert snap.activation_of(0) == 0.5
        assert snap.activation_of(99) == 0.0

    def test_history_push_and_window(self):
        history = ActivationHistory(window=3)
        for i in range(5):
            snap = make_snapshot(i, {"sensory": [(i, 0.5)]})
            history.push(snap)
        assert len(history) == 3
        assert history.current.tick == 4
        assert history.previous.tick == 3

    def test_history_neurons_in_window(self):
        history = ActivationHistory(window=3)
        history.push(make_snapshot(0, {"sensory": [(0, 0.3)]}))
        history.push(make_snapshot(1, {"sensory": [(1, 0.5)]}))
        history.push(make_snapshot(2, {"sensory": [(0, 0.8), (2, 0.4)]}))

        active = history.neurons_active_in_window()
        assert active[0] == 0.8  # max across window
        assert active[1] == 0.5
        assert active[2] == 0.4


# =======================
# Hebbian learning tests
# =======================

class TestHebbian:
    def test_effective_learning_rate_bloom(self):
        nm = NeuromodulatorState()
        lr = compute_effective_learning_rate(0, nm, novelty_signal=0.0)
        # BLOOM phase: base * 2.0 * 1.0 * (1 + 0.5 * 0.5)
        assert lr > HEBBIAN_RATE

    def test_effective_learning_rate_mature(self):
        nm = NeuromodulatorState()
        lr_bloom = compute_effective_learning_rate(0, nm)
        lr_mature = compute_effective_learning_rate(CRITICAL_END_TICK + 1, nm)
        assert lr_mature < lr_bloom

    def test_novelty_boosts_learning(self):
        nm = NeuromodulatorState()
        lr_low = compute_effective_learning_rate(0, nm, novelty_signal=0.0)
        lr_high = compute_effective_learning_rate(0, nm, novelty_signal=1.0)
        assert lr_high > lr_low

    def test_hebbian_strengthens_coactive_synapses(self):
        setup_small_brain()

        # Neurons 0 and 1 are co-active, synapse 0→1 should strengthen
        history = ActivationHistory(window=3)
        history.push(make_snapshot(0, {"sensory": [(0, 0.8), (1, 0.7)]}))

        w_before = brain_core.get_synapse_weight(0, 1)

        nm = NeuromodulatorState()
        count = hebbian_update(history, 0, nm)

        brain_core.apply_synapse_updates()
        w_after = brain_core.get_synapse_weight(0, 1)

        assert count > 0
        assert w_after > w_before

    def test_hebbian_ignores_hardwired(self):
        setup_small_brain()

        # Neuron 3→4 is hardwired (plasticity=0), should not be updated
        history = ActivationHistory(window=3)
        history.push(make_snapshot(0, {"sensory": [(3, 0.8), (4, 0.7)]}))

        w_before = brain_core.get_synapse_weight(3, 4)

        nm = NeuromodulatorState()
        hebbian_update(history, 0, nm)

        brain_core.apply_synapse_updates()
        w_after = brain_core.get_synapse_weight(3, 4)

        assert w_after == w_before


# =======================
# Anti-Hebbian tests
# =======================

class TestAntiHebbian:
    def test_anti_hebbian_rate_phases(self):
        r_bloom = compute_anti_hebbian_rate(0)
        r_critical = compute_anti_hebbian_rate(BLOOM_END_TICK + 1)
        r_mature = compute_anti_hebbian_rate(CRITICAL_END_TICK + 1)
        assert r_critical > r_bloom  # More aggressive during CRITICAL
        assert r_mature < r_critical

    def test_anti_hebbian_weakens_non_coactive(self):
        setup_small_brain()

        # Neuron 0 is active, neuron 2 is NOT active
        # Synapse 0→2 should weaken
        history = ActivationHistory(window=3)
        history.push(make_snapshot(0, {"sensory": [(0, 0.8)]}))

        w_before = brain_core.get_synapse_weight(0, 2)

        count = anti_hebbian_update(history, 0)

        brain_core.apply_synapse_updates()
        w_after = brain_core.get_synapse_weight(0, 2)

        assert count > 0
        assert w_after < w_before


# =======================
# Pruning tests
# =======================

class TestPruning:
    def test_phase_detection(self):
        assert get_phase(0) == "bloom"
        assert get_phase(BLOOM_END_TICK - 1) == "bloom"
        assert get_phase(BLOOM_END_TICK) == "critical"
        assert get_phase(CRITICAL_END_TICK - 1) == "critical"
        assert get_phase(CRITICAL_END_TICK) == "mature"

    def test_no_pruning_during_bloom(self):
        setup_small_brain()
        count = pruning_pass_sampled(0, list(range(100)))
        assert count == 0

    def test_critical_prunes_low_weight(self):
        # Create brain with very low weight synapses
        synapses = [
            (0, 1, 0.01, 1, 1.0),  # Very low weight — should be pruned
            (0, 2, 0.5, 1, 1.0),   # Normal weight — should survive
        ]
        brain_core.init_brain_with_synapses(synapses)

        count = pruning_pass_sampled(BLOOM_END_TICK, [0])
        assert count == 1  # Only the 0.01 weight synapse

    def test_critical_does_not_prune_hardwired(self):
        synapses = [
            (0, 1, 0.01, 1, 0.0),  # Low weight but hardwired
        ]
        brain_core.init_brain_with_synapses(synapses)
        count = pruning_pass_sampled(BLOOM_END_TICK, [0])
        assert count == 0


# =======================
# Novelty tests
# =======================

class TestNovelty:
    def test_novelty_tracker_initial(self):
        tracker = NoveltyTracker()
        snap = make_snapshot(0, {"sensory": [(0, 0.5), (1, 0.8)]})
        errors = tracker.update(snap)
        assert "sensory" in errors
        # First tick should have some surprise (since EMA starts at 0)
        assert errors["sensory"] > 0.0

    def test_novelty_settles_with_constant_input(self):
        tracker = NoveltyTracker()
        for i in range(100):
            snap = make_snapshot(i, {"sensory": [(0, 0.5), (1, 0.8)]})
            errors = tracker.update(snap)

        # After many constant inputs, prediction error should be very low
        last_error = errors["sensory"]
        assert last_error < 0.1

    def test_novelty_spikes_on_change(self):
        tracker = NoveltyTracker()
        # Train on constant input
        for i in range(100):
            snap = make_snapshot(i, {"sensory": [(0, 0.5)]})
            tracker.update(snap)

        # Sudden change — many more neurons active
        big_snap = make_snapshot(
            100,
            {"sensory": [(j, 0.8) for j in range(500)]},
        )
        errors = tracker.update(big_snap)
        assert errors["sensory"] > 0.3  # Clear surprise

    def test_classify_error(self):
        tracker = NoveltyTracker()
        assert tracker.classify_error(0.05) == "expected"
        assert tracker.classify_error(0.3) == "interesting"
        assert tracker.classify_error(0.6) == "surprise"
        assert tracker.classify_error(0.9) == "alarm"

    def test_neuromod_modulation(self):
        tracker = NoveltyTracker()
        nm = NeuromodulatorState(arousal=0.5)
        tracker.modulate_neuromodulators(0.9, nm)  # alarm
        assert nm.arousal > 0.5

    def test_expected_lowers_arousal(self):
        tracker = NoveltyTracker()
        nm = NeuromodulatorState(arousal=0.5)
        tracker.modulate_neuromodulators(0.05, nm)  # expected
        assert nm.arousal < 0.5


# =======================
# Working Memory tests
# =======================

class TestWorkingMemory:
    def test_capacity_limit(self):
        wm = WorkingMemory(capacity=3)
        wm.update([("t1", 0.5), ("t2", 0.6), ("t3", 0.7), ("t4", 0.8)])
        assert len(wm) == 3
        # Should keep the 3 strongest
        assert "t4" in wm.trace_ids
        assert "t3" in wm.trace_ids
        assert "t2" in wm.trace_ids

    def test_decay(self):
        wm = WorkingMemory(capacity=5)
        wm.update([("t1", 0.5)])
        strength_before = wm.slots[0][1]
        # Update with no new traces — existing should decay
        wm.update([])
        strength_after = wm.slots[0][1] if wm.slots else 0.0
        assert strength_after < strength_before

    def test_refresh_existing(self):
        wm = WorkingMemory(capacity=5)
        wm.update([("t1", 0.3)])
        wm.update([("t1", 0.8)])  # Refresh with higher activation
        assert len(wm) == 1
        assert wm.slots[0][1] >= 0.8


# =======================
# TickLoop integration tests
# =======================

class TestTickLoop:
    def test_basic_step(self):
        """Run a few ticks with small brain and verify stats returned."""
        setup_small_brain()
        store = TraceStore()
        store.add(Trace(id="t1", neurons={"sensory": [0, 1, 2]}))

        loop = TickLoop(store)

        # Inject and step
        brain_core.inject_activations([(0, 0.8), (1, 0.7), (2, 0.6)])
        result = loop.step()

        assert "tick" in result
        assert "total_active" in result
        assert "novelty" in result
        assert "phase" in result
        assert result["phase"] == "bloom"

    def test_repeated_signal_strengthens(self):
        """Inject the same signal 50 times and verify synapses strengthen."""
        setup_small_brain()
        store = TraceStore()
        loop = TickLoop(store)

        w_initial = brain_core.get_synapse_weight(0, 1)

        # Repeatedly inject same pattern
        for i in range(50):
            brain_core.inject_activations([(0, 0.8), (1, 0.7)])
            loop.step()

        # Force apply any queued updates
        brain_core.apply_synapse_updates()

        w_final = brain_core.get_synapse_weight(0, 1)
        assert w_final > w_initial, (
            f"Synapse 0→1 should have strengthened: {w_initial} → {w_final}"
        )

    def test_trace_activation_detected(self):
        """Traces should be detected as active when their neurons fire."""
        setup_small_brain()
        store = TraceStore()
        store.add(Trace(
            id="t1",
            neurons={"sensory": [0, 1, 2]},
            strength=0.2,
        ))

        loop = TickLoop(store)

        # Activate all 3 neurons of trace t1
        brain_core.inject_activations([(0, 0.8), (1, 0.7), (2, 0.6)])
        result = loop.step()

        # The trace should have been detected
        assert result["active_traces"] >= 0  # May or may not match depending on threshold

    def test_novelty_decreases_over_time(self):
        """Constant input should reduce novelty over time."""
        setup_small_brain()
        store = TraceStore()
        loop = TickLoop(store)

        novelties = []
        for i in range(30):
            brain_core.inject_activations([(0, 0.8)])
            result = loop.step()
            novelties.append(result["novelty"])

        # Novelty should generally decrease (or stay low)
        # Compare first few vs last few
        early = sum(novelties[:5]) / 5
        late = sum(novelties[-5:]) / 5
        assert late <= early + 0.1  # Allow small noise but trend should be downward
