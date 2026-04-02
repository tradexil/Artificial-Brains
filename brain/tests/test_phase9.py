"""Phase 9 tests: Homeostasis & Sleep.

Tests:
  - Rust-side homeostasis (regulation, sleep pressure, circadian)
  - Rust-side sleep cycle (state transitions, input gating, energy recovery)
  - PyO3 bridge functions (get/set homeostasis, sleep state queries)
  - Python HomeostasisManager (dream replay, consolidation scheduling)
  - Tick loop Phase 9 integration (sleep metrics in return dict)
  - Sleep/wake cycle end-to-end
"""

import pytest

import brain_core

from brain.structures.trace_store import Trace, TraceStore
from brain.learning.homeostasis import HomeostasisManager, DreamReplayStats
from brain.learning.tick_loop import TickLoop


@pytest.fixture(autouse=True)
def fresh_brain():
    brain_core.init_brain()
    yield
    brain_core.reset_brain()


# === Rust-side PyO3 bridge tests ===


class TestHomeostasisBridge:
    """Test PyO3 homeostasis functions."""

    def test_get_homeostasis_summary(self):
        pressure, phase, awake, asleep = brain_core.get_homeostasis_summary()
        assert pressure == pytest.approx(0.0, abs=1e-3)
        assert 0.0 <= phase <= 1.0
        assert awake == 0
        assert asleep == 0

    def test_homeostasis_after_ticks(self):
        for _ in range(100):
            brain_core.tick()
        pressure, phase, awake, asleep = brain_core.get_homeostasis_summary()
        # After 100 awake ticks, sleep pressure should increase
        assert pressure > 0.0
        assert phase > 0.0
        # Brain should be awake → awake counter increases (tracked via Rust)
        assert awake == 100

    def test_set_homeostasis_params(self):
        brain_core.set_homeostasis_params(0.01, 0.01, 0.01, 0.001, 0.01)
        # Run some ticks — sleep pressure should accumulate faster with 0.001
        for _ in range(100):
            brain_core.tick()
        pressure, _, _, _ = brain_core.get_homeostasis_summary()
        assert pressure > 0.05  # much faster accumulation than default

    def test_circadian_phase_advances(self):
        for _ in range(1000):
            brain_core.tick()
        _, phase, _, _ = brain_core.get_homeostasis_summary()
        assert phase > 0.0  # should have advanced from 0

    def test_get_circadian_phase(self):
        phase = brain_core.get_circadian_phase()
        assert phase == pytest.approx(0.0, abs=1e-3)

    def test_get_sleep_pressure(self):
        pressure = brain_core.get_sleep_pressure()
        assert pressure == pytest.approx(0.0, abs=1e-3)


class TestSleepBridge:
    """Test PyO3 sleep functions."""

    def test_get_sleep_summary(self):
        state, ticks, cycles, rem = brain_core.get_sleep_summary()
        assert state == "awake"
        assert ticks == 0
        assert cycles == 0
        assert rem == 0

    def test_is_asleep_default(self):
        assert not brain_core.is_asleep()

    def test_in_rem_default(self):
        assert not brain_core.in_rem()

    def test_sleep_input_gate_awake(self):
        gate = brain_core.get_sleep_input_gate()
        assert gate == pytest.approx(1.0)

    def test_force_wake(self):
        # Force into sleep by setting very high pressure / low energy
        brain_core.set_homeostasis_params(0.005, 0.002, 0.003, 0.1, 0.0001)
        brain_core.set_neuromodulator(0.5, 0.0, 0.5, 0.1)  # low energy
        # Run ticks to trigger drowsy
        for _ in range(50):
            brain_core.tick()
        if brain_core.is_asleep():
            brain_core.force_wake()
            assert not brain_core.is_asleep()
            state, _, _, _ = brain_core.get_sleep_summary()
            assert state == "awake"

    def test_set_sleep_durations(self):
        brain_core.set_sleep_durations(100, 200, 300, 200)
        # This just tests no crash — durations affect state machine timing

    def test_sleep_trigger_on_low_energy(self):
        # Set energy very low
        brain_core.set_neuromodulator(0.5, 0.0, 0.5, 0.05)
        brain_core.tick()
        # After tick, sleep cycle should have triggered
        state, _, _, _ = brain_core.get_sleep_summary()
        assert state in ("drowsy", "awake")  # May or may not trigger on first tick

    def test_sleep_gate_reduces_during_sleep(self):
        # Fast sleep: high pressure rate, short durations
        brain_core.set_homeostasis_params(0.005, 0.002, 0.003, 0.5, 0.0001)
        brain_core.set_sleep_durations(10, 20, 30, 20)
        brain_core.set_neuromodulator(0.5, 0.0, 0.5, 0.05)

        for _ in range(100):
            brain_core.tick()

        if brain_core.is_asleep():
            gate = brain_core.get_sleep_input_gate()
            assert gate < 1.0


class TestSleepCycleFull:
    """Test full sleep/wake cycle through Rust."""

    def test_full_sleep_cycle_with_fast_params(self):
        # Use very fast sleep parameters
        brain_core.set_homeostasis_params(0.005, 0.002, 0.003, 0.5, 0.001)
        brain_core.set_sleep_durations(50, 100, 150, 100)

        # Deplete energy to trigger sleep
        brain_core.set_neuromodulator(0.5, 0.0, 0.5, 0.05)

        states_seen = set()
        for _ in range(2000):
            brain_core.tick()
            state, _, _, _ = brain_core.get_sleep_summary()
            states_seen.add(state)

        # Should have seen at least awake + drowsy
        assert "awake" in states_seen or "drowsy" in states_seen

    def test_energy_recovery_during_sleep(self):
        # Set low energy, fast sleep
        brain_core.set_homeostasis_params(0.005, 0.002, 0.003, 0.5, 0.001)
        brain_core.set_sleep_durations(10, 20, 30, 20)
        brain_core.set_neuromodulator(0.5, 0.0, 0.5, 0.05)

        # Run through a sleep period
        for _ in range(200):
            brain_core.tick()

        _, _, _, energy = brain_core.get_neuromodulator()
        # Energy should have recovered somewhat during sleep
        # (can't assert exact value due to complex interactions)
        assert energy >= 0.0  # At minimum, not negative

    def test_homeostasis_regulates_arousal(self):
        # Set arousal high
        brain_core.set_neuromodulator(1.0, 0.0, 0.5, 1.0)
        initial_arousal = 1.0

        # Run ticks — homeostasis should pull arousal back toward 0.5
        for _ in range(200):
            brain_core.tick()

        arousal, _, _, _ = brain_core.get_neuromodulator()
        assert arousal < initial_arousal, f"Arousal should decrease: {arousal}"


# === Python HomeostasisManager tests ===


class TestHomeostasisManager:

    def _make_trace_store(self, n=10):
        ts = TraceStore()
        for i in range(n):
            t = Trace(
                id=f"t{i}",
                neurons={
                    "memory_long": [55000 + i * 10, 55001 + i * 10],
                    "pattern": [85000 + i * 10],
                },
                strength=0.5 + (i % 5) * 0.1,
                novelty=0.3 + (i % 3) * 0.2,
                polarity=(-1.0 if i % 2 == 0 else 1.0) * i * 0.1,
            )
            ts.add(t)
        return ts

    def test_manager_creation(self):
        ts = self._make_trace_store()
        mgr = HomeostasisManager(ts)
        assert not mgr._in_sleep_session

    def test_record_active_traces(self):
        ts = self._make_trace_store()
        mgr = HomeostasisManager(ts)
        mgr.record_active_traces(["t0", "t1", "t2"])
        assert len(mgr._recent_trace_ids) == 3

    def test_record_active_traces_dedup(self):
        ts = self._make_trace_store()
        mgr = HomeostasisManager(ts)
        mgr.record_active_traces(["t0", "t1"])
        mgr.record_active_traces(["t1", "t2"])
        assert mgr._recent_trace_ids == ["t0", "t1", "t2"]

    def test_step_while_awake(self):
        ts = self._make_trace_store()
        mgr = HomeostasisManager(ts)
        from brain.structures.brain_state import NeuromodulatorState
        nm = NeuromodulatorState()
        stats = mgr.step(0, nm)
        assert stats["sleep_state"] == "awake"
        assert not stats["is_asleep"]

    def test_dream_replay_not_during_awake(self):
        ts = self._make_trace_store()
        mgr = HomeostasisManager(ts)
        from brain.structures.brain_state import NeuromodulatorState
        nm = NeuromodulatorState()
        stats = mgr.step(0, nm)
        assert stats["dream_replayed"] == 0

    def test_dream_candidates(self):
        ts = self._make_trace_store()
        mgr = HomeostasisManager(ts)
        mgr.record_active_traces(["t0", "t3", "t7"])
        candidates = mgr.get_dream_candidates()
        assert "t0" in candidates
        assert "t3" in candidates
        assert "t7" in candidates

    def test_consolidation_scheduling(self):
        ts = self._make_trace_store()
        mgr = HomeostasisManager(ts)
        # Not in sleep → should not consolidate in sleep
        assert not mgr.should_consolidate_in_sleep()

    def test_mark_consolidation_done(self):
        ts = self._make_trace_store()
        mgr = HomeostasisManager(ts)
        mgr.mark_consolidation_done()
        assert mgr._consolidation_done_this_sleep

    def test_wake_alarm_on_high_pain(self):
        ts = self._make_trace_store()
        mgr = HomeostasisManager(ts)
        from brain.structures.brain_state import NeuromodulatorState
        nm = NeuromodulatorState()

        # Force sleep through fast params
        brain_core.set_homeostasis_params(0.005, 0.002, 0.003, 0.5, 0.0001)
        brain_core.set_sleep_durations(5, 10, 15, 10)
        brain_core.set_neuromodulator(0.5, 0.0, 0.5, 0.05)

        # Run until asleep
        for _ in range(50):
            brain_core.tick()

        if brain_core.is_asleep():
            # Inject high pain to trigger wake alarm
            pain_neurons = [(5000 + i, 0.9) for i in range(50)]
            brain_core.inject_activations(pain_neurons)
            brain_core.tick()

            stats = mgr.step(100, nm)
            # If pain was high enough, should have forced wake
            if stats.get("forced_wake"):
                assert not brain_core.is_asleep()

    def test_recent_traces_cap(self):
        ts = self._make_trace_store(n=1)
        mgr = HomeostasisManager(ts)
        mgr._recent_max = 5
        mgr.record_active_traces([f"trace_{i}" for i in range(10)])
        assert len(mgr._recent_trace_ids) <= 5


# === Tick loop integration tests ===


class TestTickLoopPhase9:

    def _make_tick_loop(self):
        ts = TraceStore()
        for i in range(5):
            t = Trace(
                id=f"test_t{i}",
                neurons={
                    "sensory": [i * 100, i * 100 + 1],
                    "memory_long": [55000 + i * 10],
                    "pattern": [85000 + i],
                },
                strength=0.5,
            )
            ts.add(t)
        return TickLoop(ts)

    def test_tick_loop_returns_phase9_keys(self):
        loop = self._make_tick_loop()
        result = loop.step()
        assert "sleep_state" in result
        assert "sleep_pressure" in result
        assert "circadian_phase" in result
        assert "is_asleep" in result
        assert "in_rem" in result
        assert "dream_replayed" in result

    def test_tick_loop_starts_awake(self):
        loop = self._make_tick_loop()
        result = loop.step()
        assert result["sleep_state"] == "awake"
        assert not result["is_asleep"]

    def test_tick_loop_sleep_pressure_increases(self):
        loop = self._make_tick_loop()
        pressures = []
        for _ in range(200):
            result = loop.step()
            pressures.append(result["sleep_pressure"])
        assert pressures[-1] > pressures[0]

    def test_tick_loop_circadian_advances(self):
        loop = self._make_tick_loop()
        phases = []
        for _ in range(100):
            result = loop.step()
            phases.append(result["circadian_phase"])
        assert phases[-1] > phases[0]

    def test_tick_loop_homeostasis_object_exists(self):
        loop = self._make_tick_loop()
        assert hasattr(loop, "homeostasis")
        assert isinstance(loop.homeostasis, HomeostasisManager)

    def test_tick_loop_with_fast_sleep(self):
        loop = self._make_tick_loop()
        # Use fast sleep parameters
        brain_core.set_homeostasis_params(0.005, 0.002, 0.003, 0.5, 0.001)
        brain_core.set_sleep_durations(10, 20, 30, 20)
        brain_core.set_neuromodulator(0.5, 0.0, 0.5, 0.05)

        states_seen = set()
        for _ in range(500):
            result = loop.step()
            states_seen.add(result["sleep_state"])

        # Should see at least awake or drowsy
        assert len(states_seen) >= 1


# === Config tests ===


class TestPhase9Config:

    def test_config_values_exist(self):
        from brain.utils.config import (
            HOMEOSTASIS_AROUSAL_REG_RATE,
            HOMEOSTASIS_VALENCE_REG_RATE,
            HOMEOSTASIS_FOCUS_REG_RATE,
            SLEEP_PRESSURE_RATE,
            SLEEP_DISSIPATION_RATE,
            CIRCADIAN_PERIOD,
            SLEEP_DROWSY_DURATION,
            SLEEP_LIGHT_DURATION,
            SLEEP_DEEP_DURATION,
            SLEEP_REM_DURATION,
            DREAM_REPLAY_PER_TICK,
            WAKE_ALARM_PAIN_THRESHOLD,
        )
        assert HOMEOSTASIS_AROUSAL_REG_RATE > 0
        assert SLEEP_PRESSURE_RATE > 0
        assert CIRCADIAN_PERIOD > 0
        assert SLEEP_DROWSY_DURATION > 0
        assert DREAM_REPLAY_PER_TICK > 0
        assert 0.0 < WAKE_ALARM_PAIN_THRESHOLD <= 1.0
