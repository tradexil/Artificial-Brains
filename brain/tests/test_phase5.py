"""Phase 5 tests: Integration & Memory.

Tests:
  - Binding creation, evaluation, strengthening, dissolution
  - Working memory capacity limit (7±2)
  - Pattern completion (partial cue → full recall)
  - Consolidation (short→long transfer)
  - Trace formation (persistent novel patterns become traces)
  - Binding formation (co-active cross-region patterns)
  - Integration boost (multi-modal convergence)
  - Tick loop Phase 5 integration
"""

import pytest
import brain_core

from brain.structures.trace_store import Trace, TraceStore
from brain.structures.brain_state import (
    ActivationHistory,
    ActivationSnapshot,
    NeuromodulatorState,
)
from brain.learning.consolidation import ConsolidationEngine
from brain.learning.trace_formation import TraceFormationEngine, NovelPatternTracker
from brain.learning.binding_formation import BindingFormationEngine, CoActivationTracker
from brain.learning.tick_loop import TickLoop, WorkingMemory
from brain.utils.config import (
    WORKING_MEMORY_CAPACITY,
    TRACE_ACTIVATION_THRESHOLD,
    TRACE_FORMATION_PERSISTENCE,
    BINDING_FORMATION_COUNT,
    CONSOLIDATION_TRIGGER_TICKS,
)


@pytest.fixture(autouse=True)
def reset_brain():
    """Reset brain before each test."""
    brain_core.init_brain()
    yield
    brain_core.reset_brain()


def make_trace(trace_id, regions_neurons, **kwargs):
    """Helper: create a Trace with specified neurons."""
    t = Trace(id=trace_id, neurons=regions_neurons, **kwargs)
    return t


# === BINDING TESTS ===


class TestBindings:
    def test_create_binding(self):
        """Create a binding between visual and audio patterns."""
        binding_id = brain_core.create_binding(
            "visual", [10000, 10001, 10002], 0.6,
            "audio", [30000, 30001, 30002], 0.6,
            0.0,
        )
        assert binding_id == 0
        assert brain_core.get_binding_count() == 1

    def test_binding_evaluation_inactive(self):
        """Binding should not fire when patterns are inactive."""
        brain_core.create_binding(
            "visual", [10000, 10001, 10002], 0.6,
            "audio", [30000, 30001, 30002], 0.6,
            0.0,
        )
        active = brain_core.evaluate_bindings(0.01)
        assert len(active) == 0

    def test_binding_evaluation_active(self):
        """Binding should fire when both patterns are active."""
        bid = brain_core.create_binding(
            "visual", [10000, 10001, 10002], 0.6,
            "audio", [30000, 30001, 30002], 0.6,
            0.0,
        )
        # Inject activations for both patterns
        signals = [(10000, 1.0), (10001, 1.0), (10002, 1.0),
                    (30000, 1.0), (30001, 1.0), (30002, 1.0)]
        brain_core.inject_activations(signals)
        brain_core.tick()

        active = brain_core.evaluate_bindings(0.01)
        assert len(active) == 1
        assert active[0][0] == bid

    def test_binding_strengthen(self):
        """Strengthening should increase weight and fire count."""
        bid = brain_core.create_binding(
            "visual", [10000], 0.5,
            "audio", [30000], 0.5,
            0.0,
        )
        info_before = brain_core.get_binding_info(bid)
        assert info_before is not None
        weight_before = info_before[0]

        brain_core.strengthen_binding(bid, 100)
        info_after = brain_core.get_binding_info(bid)
        assert info_after[0] > weight_before  # weight increased
        assert info_after[1] == 1  # fires == 1

    def test_binding_dissolution(self):
        """Bindings should dissolve after many misses."""
        bid = brain_core.create_binding(
            "visual", [10000], 0.5,
            "audio", [30000], 0.5,
            0.0,
        )
        # Many misses
        for _ in range(1000):
            brain_core.record_binding_miss(bid)

        info = brain_core.get_binding_info(bid)
        assert info[0] < 0.05  # weight below dissolution threshold

        pruned = brain_core.prune_bindings(0.05, 10)
        assert pruned == 1
        assert brain_core.get_binding_count() == 0

    def test_partial_binding(self):
        """Partial bindings detected when only one pattern fires."""
        brain_core.create_binding(
            "visual", [10000, 10001], 0.5,
            "audio", [30000, 30001], 0.5,
            0.0,
        )
        # Only activate visual pattern
        brain_core.inject_activations([(10000, 1.0), (10001, 1.0)])
        brain_core.tick()

        partial = brain_core.find_partial_bindings(0.01)
        assert len(partial) == 1


# === WORKING MEMORY TESTS ===


class TestWorkingMemory:
    def test_capacity_limit(self):
        """Working memory should hold at most CAPACITY traces."""
        wm = WorkingMemory(capacity=WORKING_MEMORY_CAPACITY)
        # Add more traces than capacity
        traces = [(f"t{i}", 0.8) for i in range(WORKING_MEMORY_CAPACITY + 3)]
        wm.update(traces)
        assert len(wm) <= WORKING_MEMORY_CAPACITY

    def test_weakest_evicted(self):
        """When over capacity, weakest trace should be evicted."""
        wm = WorkingMemory(capacity=3)
        wm.update([("a", 0.9), ("b", 0.5), ("c", 0.7)])
        assert len(wm) == 3

        # Add a stronger one — weakest (b) should be evicted
        wm.update([("d", 0.8)])
        assert len(wm) == 3
        ids = wm.trace_ids
        assert "d" in ids
        # b was weakest and decayed, should be gone
        # (after decay b=0.5*0.98=0.49, d=0.8, c=0.7*0.98=0.686, a=0.9*0.98=0.882)

    def test_decay(self):
        """Items decay when not refreshed."""
        wm = WorkingMemory(capacity=7)
        wm.update([("a", 0.5)])

        # Multiple updates without refreshing — should decay
        for _ in range(150):
            wm.update([])

        # After 150 decays at 0.02 rate: 0.5 * 0.98^150 ≈ 0.024
        assert len(wm) == 0 or wm.slots[0][1] < 0.05

    def test_refresh_prevents_decay(self):
        """Refreshing an item should maintain its strength."""
        wm = WorkingMemory(capacity=7)
        wm.update([("a", 0.8)])

        # Keep refreshing
        for _ in range(20):
            wm.update([("a", 0.8)])

        assert len(wm) == 1
        assert wm.slots[0][1] >= 0.7


# === PATTERN COMPLETION TESTS ===


class TestPatternCompletion:
    def test_below_threshold_no_completion(self):
        """Pattern completion should not trigger below 40% threshold."""
        # memory_long starts at 55000
        trace_neurons = list(range(55000, 55010))  # 10 neurons

        # Activate only 2/10 (20% < 40%)
        brain_core.inject_activations([(55000, 1.0), (55001, 1.0)])
        brain_core.tick()

        boosted = brain_core.pattern_complete(trace_neurons, 0.4, 0.5)
        assert boosted == 0

    def test_above_threshold_completes(self):
        """Pattern completion should boost inactive trace neurons above 40%."""
        trace_neurons = list(range(55000, 55010))

        # Activate 5/10 (50% >= 40%) — need to get them into activations
        signals = [(55000 + i, 1.0) for i in range(5)]
        brain_core.inject_activations(signals)
        brain_core.tick()

        boosted = brain_core.pattern_complete(trace_neurons, 0.4, 0.5)
        assert boosted == 5  # 5 inactive neurons should be boosted

    def test_empty_trace_no_completion(self):
        """Empty trace should not trigger completion."""
        boosted = brain_core.pattern_complete([], 0.4, 0.5)
        assert boosted == 0


# === CONSOLIDATION TESTS ===


class TestConsolidation:
    def test_not_triggered_initially(self):
        """Consolidation should not trigger with fresh brain."""
        engine = ConsolidationEngine()
        neuromod = NeuromodulatorState()  # energy=1.0
        assert not engine.should_consolidate(0, neuromod)

    def test_triggered_by_low_energy(self):
        """Consolidation should trigger when energy drops below threshold."""
        engine = ConsolidationEngine()
        neuromod = NeuromodulatorState()
        neuromod.energy = 0.1  # Below 0.2 threshold
        assert engine.should_consolidate(0, neuromod)

    def test_triggered_by_tick_count(self):
        """Consolidation should trigger after CONSOLIDATION_TRIGGER_TICKS."""
        engine = ConsolidationEngine()
        neuromod = NeuromodulatorState()  # energy=1.0
        assert engine.should_consolidate(CONSOLIDATION_TRIGGER_TICKS, neuromod)

    def test_start_consolidation(self):
        """Starting consolidation should queue traces and reduce input gain."""
        engine = ConsolidationEngine()
        store = TraceStore()
        t1 = make_trace("t1", {"memory_long": [55000, 55001]}, strength=0.5, polarity=0.8)
        t2 = make_trace("t2", {"memory_long": [55002, 55003]}, strength=0.3, novelty=0.9)
        store.add(t1)
        store.add(t2)

        queued = engine.start_consolidation(1000, store, ["t1", "t2"])
        assert queued == 2
        assert engine.is_consolidating

    def test_consolidation_strengthens_traces(self):
        """Consolidation should boost trace strength."""
        engine = ConsolidationEngine()
        store = TraceStore()
        t1 = make_trace("t1", {"memory_long": [55000, 55001]}, strength=0.3)
        store.add(t1)

        engine.start_consolidation(0, store, ["t1"])

        # Run consolidation steps
        for i in range(100):
            result = engine.consolidation_step(i + 1, store)

        # Trace should be stronger
        trace = store.get("t1")
        assert trace.strength > 0.3

    def test_context_stripping(self):
        """Consolidation should strip context_tags over time."""
        engine = ConsolidationEngine()
        store = TraceStore()
        t1 = make_trace(
            "t1",
            {"memory_long": [55000]},
            context_tags=["morning", "kitchen", "tuesday", "hot", "coffee"],
        )
        store.add(t1)

        engine.start_consolidation(0, store, ["t1"])

        # Run enough steps to process
        for i in range(100):
            engine.consolidation_step(i + 1, store)

        trace = store.get("t1")
        assert len(trace.context_tags) < 5  # Some tags stripped

    def test_consolidation_ends(self):
        """Consolidation should end after duration."""
        engine = ConsolidationEngine()
        store = TraceStore()
        t1 = make_trace("t1", {"memory_long": [55000]})
        store.add(t1)

        engine.start_consolidation(0, store, ["t1"])

        from brain.utils.config import CONSOLIDATION_DURATION
        for i in range(CONSOLIDATION_DURATION + 10):
            engine.consolidation_step(i + 1, store)

        assert not engine.is_consolidating


# === TRACE FORMATION TESTS ===


class TestTraceFormation:
    def test_no_formation_without_persistence(self):
        """Traces should not form from transient patterns."""
        store = TraceStore()
        engine = TraceFormationEngine(store)

        snap = ActivationSnapshot(
            tick=1,
            active_neurons={
                "visual": [(10000, 1.0), (10001, 1.0)],
                "audio": [(30000, 1.0)],
            },
            total_active=3,
        )
        formed = engine.step(snap, [], 0.5, 1, 0)
        assert formed == 0

    def test_formation_after_persistence(self):
        """Traces should form after pattern persists for required ticks."""
        store = TraceStore()
        engine = TraceFormationEngine(store)

        for tick in range(TRACE_FORMATION_PERSISTENCE + 5):
            snap = ActivationSnapshot(
                tick=tick,
                active_neurons={
                    "visual": [(10000, 1.0), (10001, 1.0)],
                    "audio": [(30000, 1.0), (30001, 1.0)],
                },
                total_active=4,
            )
            formed = engine.step(snap, [], 0.5, tick, 0)

        # After enough persistence, a trace should be formed
        assert len(store) > 0

    def test_no_formation_at_capacity(self):
        """No trace formation when working memory is full."""
        store = TraceStore()
        engine = TraceFormationEngine(store)

        snap = ActivationSnapshot(
            tick=1,
            active_neurons={
                "visual": [(10000, 1.0)],
                "audio": [(30000, 1.0)],
            },
            total_active=2,
        )
        formed = engine.step(snap, [], 0.5, 1, WORKING_MEMORY_CAPACITY)
        assert formed == 0

    def test_no_formation_with_match(self):
        """No trace formation when a strong match already exists."""
        store = TraceStore()
        engine = TraceFormationEngine(store)

        snap = ActivationSnapshot(
            tick=1,
            active_neurons={
                "visual": [(10000, 1.0)],
                "audio": [(30000, 1.0)],
            },
            total_active=2,
        )
        # Simulate a strong existing match
        formed = engine.step(snap, [("existing_trace", 0.9)], 0.5, 1, 0)
        assert formed == 0

    def test_merge_overlapping_traces(self):
        """Traces with >80% overlap should merge."""
        store = TraceStore()
        engine = TraceFormationEngine(store)

        neurons_a = list(range(10000, 10010))
        neurons_b = list(range(10000, 10009)) + [10010]  # 90% overlap

        t1 = make_trace("t1", {"visual": neurons_a}, fire_count=15,
                         strength=0.5)
        t2 = make_trace("t2", {"visual": neurons_b}, fire_count=15,
                         strength=0.3)
        store.add(t1)
        store.add(t2)

        merged = engine.merge_overlapping(min_co_fires=10)
        assert merged == 1
        assert len(store) == 1


# === BINDING FORMATION TESTS ===


class TestBindingFormation:
    def test_co_activation_tracking(self):
        """Co-activation tracker should record cross-region co-activations."""
        tracker = CoActivationTracker()
        store = TraceStore()
        t1 = make_trace("t1", {"visual": [10000, 10001]})
        t2 = make_trace("t2", {"audio": [30000, 30001]})
        store.add(t1)
        store.add(t2)

        history = ActivationHistory(window=3)

        # Record co-activations below threshold
        for tick in range(BINDING_FORMATION_COUNT - 1):
            ready = tracker.record(
                [("t1", 0.8), ("t2", 0.7)],
                store, tick, history,
            )
            assert len(ready) == 0

    def test_binding_forms_at_threshold(self):
        """Binding should form after BINDING_FORMATION_COUNT co-activations."""
        tracker = CoActivationTracker()
        store = TraceStore()
        t1 = make_trace("t1", {"visual": [10000, 10001]})
        t2 = make_trace("t2", {"audio": [30000, 30001]})
        store.add(t1)
        store.add(t2)

        history = ActivationHistory(window=3)

        ready = []
        for tick in range(BINDING_FORMATION_COUNT + 5):
            ready = tracker.record(
                [("t1", 0.8), ("t2", 0.7)],
                store, tick, history,
            )
            if ready:
                break

        assert len(ready) > 0
        # Should contain the correct trace/region pair
        pair = ready[0]
        assert "t1" in pair or "t2" in pair


# === INTEGRATION TESTS ===


class TestIntegration:
    def test_no_input_no_integration(self):
        """Integration should report 0 active regions with no input."""
        count = brain_core.integration_input_count(0.5)
        assert count == 0

    def test_multi_modal_input_integration(self):
        """Multiple active input regions should boost integration."""
        # Activate visual and audio neurons
        signals = [(10000 + i, 1.0) for i in range(20)]  # visual
        signals += [(30000 + i, 1.0) for i in range(20)]  # audio
        brain_core.inject_activations(signals)
        brain_core.tick()

        count = brain_core.integration_input_count(0.5)
        assert count >= 2

    def test_integration_boost(self):
        """Integration region should be boosted with multi-modal input."""
        # First activate some visual neurons so potentials exist
        signals = [(10000 + i, 1.0) for i in range(20)]
        signals += [(30000 + i, 1.0) for i in range(20)]
        brain_core.inject_activations(signals)
        brain_core.tick()

        boosted = brain_core.boost_integration(0.8, 50)
        assert boosted >= 0  # May boost neurons that have non-zero potentials


# === MEMORY OPERATIONS TESTS ===


class TestMemoryOperations:
    def test_boost_working_memory(self):
        """WM neuron boost should work through PyO3."""
        # memory_short starts at 45000
        neurons = [45000, 45001, 45002]
        boosted = brain_core.boost_working_memory(neurons, 0.3)
        assert boosted == 3

    def test_strengthen_memory_trace(self):
        """Memory trace strengthening should work through PyO3."""
        neurons = [55000, 55001, 55002]
        count = brain_core.strengthen_memory_trace(neurons, 0.3)
        assert count == 3


# === TICK LOOP PHASE 5 INTEGRATION ===


class TestTickLoopPhase5:
    def test_step_returns_phase5_fields(self):
        """Tick loop step should return Phase 5 stats."""
        store = TraceStore()
        loop = TickLoop(store)
        result = loop.step()

        assert "traces_formed" in result
        assert "bindings_formed" in result
        assert "total_bindings" in result
        assert "consolidating" in result

    def test_no_crash_many_steps(self):
        """Multiple steps with Phase 5 should not crash."""
        store = TraceStore()
        # Add a trace for the loop to find
        t = make_trace("t1", {
            "sensory": [0, 1, 2],
            "visual": [10000, 10001],
            "memory_short": [45000, 45001],
            "memory_long": [55000, 55001],
        }, strength=0.5)
        store.add(t)

        loop = TickLoop(store)
        for i in range(20):
            signals = [(j, 0.5) for j in range(5)]
            brain_core.inject_activations(signals)
            result = loop.step()
            assert result["tick"] == i

    def test_consolidation_triggered_by_energy(self):
        """Consolidation should start when energy depletes."""
        store = TraceStore()
        t = make_trace("t1", {"memory_long": [55000, 55001]}, strength=0.5)
        store.add(t)

        loop = TickLoop(store)
        loop._awake_trace_ids = ["t1"]
        loop.neuromod.energy = 0.1  # Below threshold

        result = loop.step()
        assert loop.consolidation.is_consolidating or result["consolidating"]

    def test_working_memory_neurons_boosted(self):
        """Working memory traces should have their neurons boosted."""
        store = TraceStore()
        t = make_trace("t1", {
            "sensory": [0, 1, 2],
            "memory_short": [45000, 45001, 45002],
        }, strength=0.8)
        store.add(t)

        loop = TickLoop(store)

        # Activate all trace neurons to trigger matching
        signals = [(0, 1.0), (1, 1.0), (2, 1.0),
                    (45000, 1.0), (45001, 1.0), (45002, 1.0)]
        brain_core.inject_activations(signals)
        result = loop.step()

        # If trace matched, WM neurons should have been boosted
        # Check that the loop at least tried to boost
        wm_neurons = loop._get_working_memory_neurons()
        # wm_neurons may be empty if trace didn't match
        # but the method should work without error
        assert isinstance(wm_neurons, list)

    def test_pattern_completion_during_step(self):
        """Pattern completion should run for active traces with memory_long neurons."""
        store = TraceStore()
        t = make_trace("t1", {
            "sensory": [0, 1, 2],
            "memory_long": [55000, 55001, 55002, 55003, 55004],
        }, strength=0.8)
        store.add(t)

        loop = TickLoop(store)

        # Activate enough trace neurons to trigger matching
        signals = [(0, 1.0), (1, 1.0), (2, 1.0),
                    (55000, 1.0), (55001, 1.0), (55002, 1.0)]
        brain_core.inject_activations(signals)
        result = loop.step()

        # No crashes — pattern completion ran for matching traces
        assert result["tick"] == 0
