"""Homeostasis manager: Python-side orchestration of homeostatic regulation.

Manages:
  - Dream replay during REM sleep (trace reactivation)
  - Autonomous consolidation scheduling tied to sleep phases
  - Arousal/valence/focus temperature regulation
  - Sleep/wake transition callbacks

Works with the Rust-side HomeostasisSystem and SleepCycleManager
which handle the per-tick mechanics. This module handles the higher-level
policy decisions that require access to trace stores and learning state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import brain_core

from brain.structures.trace_store import Trace, TraceStore
from brain.structures.brain_state import NeuromodulatorState


@dataclass
class DreamReplayStats:
    """Stats from one dream replay episode."""

    traces_replayed: int = 0
    neurons_activated: int = 0
    synapses_strengthened: int = 0


@dataclass
class SleepSessionStats:
    """Accumulated stats for a complete sleep session."""

    total_ticks: int = 0
    rem_episodes: int = 0
    traces_replayed: int = 0
    cycles_completed: int = 0
    energy_start: float = 0.0
    energy_end: float = 0.0


class HomeostasisManager:
    """Manages homeostatic processes: dream replay, consolidation scheduling,
    arousal regulation.

    Usage:
        manager = HomeostasisManager(trace_store)
        # Each tick in the main loop:
        stats = manager.step(tick, neuromod)
    """

    def __init__(self, trace_store: TraceStore):
        self.trace_store = trace_store

        # Dream replay state
        self._dream_queue: list[str] = []  # trace IDs to replay in REM
        self._dream_idx: int = 0
        self._dreams_per_tick: int = 3  # traces replayed per tick during REM

        # Recent trace activity (for dream content selection)
        self._recent_trace_ids: list[str] = []
        self._recent_max: int = 500  # max traces to track for dream content

        # Sleep session tracking
        self._in_sleep_session: bool = False
        self._sleep_session: SleepSessionStats = SleepSessionStats()

        # Consolidation scheduling
        self._consolidation_done_this_sleep: bool = False

        # Wake alarm: if high-importance stimulus arrives during sleep
        self._wake_alarm_threshold: float = 0.8  # pain > this forces wake

    def record_active_traces(self, trace_ids: list[str]) -> None:
        """Record recently active traces (call each tick while awake).

        These become candidates for dream replay during REM.
        """
        for tid in trace_ids:
            if tid not in self._recent_trace_ids:
                self._recent_trace_ids.append(tid)
        # Trim to max
        if len(self._recent_trace_ids) > self._recent_max:
            self._recent_trace_ids = self._recent_trace_ids[-self._recent_max:]

    def step(self, tick: int, neuromod: NeuromodulatorState) -> dict:
        """Run one homeostasis step. Call each tick.

        Returns dict with homeostasis metrics.
        """
        is_asleep = brain_core.is_asleep()
        in_rem = brain_core.in_rem()
        sleep_state, ticks_in_state, cycles_completed, rem_episodes = brain_core.get_sleep_summary()
        sleep_pressure, circadian_phase, _, _ = brain_core.get_homeostasis_summary()

        stats = {
            "sleep_state": sleep_state,
            "sleep_pressure": sleep_pressure,
            "circadian_phase": circadian_phase,
            "is_asleep": is_asleep,
            "in_rem": in_rem,
            "dream_replayed": 0,
            "dream_neurons_activated": 0,
            "sleep_session_active": self._in_sleep_session,
        }

        # === Sleep session start ===
        if is_asleep and not self._in_sleep_session:
            self._start_sleep_session(neuromod.energy)

        # === Sleep session end ===
        if not is_asleep and self._in_sleep_session:
            self._end_sleep_session(neuromod.energy)

        # === During sleep ===
        if is_asleep:
            self._sleep_session.total_ticks += 1

            # Dream replay during REM
            if in_rem:
                replay_stats = self._dream_replay_step()
                stats["dream_replayed"] = replay_stats.traces_replayed
                stats["dream_neurons_activated"] = replay_stats.neurons_activated

        # === Check for wake alarm (pain during sleep) ===
        if is_asleep:
            pain = brain_core.get_pain_level()
            if pain > self._wake_alarm_threshold:
                brain_core.force_wake()
                stats["forced_wake"] = True

        return stats

    def should_consolidate_in_sleep(self) -> bool:
        """Whether consolidation should run during current sleep.

        Policy: consolidate once per sleep session, during deep sleep.
        """
        if self._consolidation_done_this_sleep:
            return False
        sleep_state, _, _, _ = brain_core.get_sleep_summary()
        return sleep_state == "deep"

    def mark_consolidation_done(self) -> None:
        """Mark that consolidation ran this sleep session."""
        self._consolidation_done_this_sleep = True

    def get_dream_candidates(self) -> list[str]:
        """Get trace IDs that are candidates for dream replay."""
        return list(self._recent_trace_ids)

    def _start_sleep_session(self, energy: float) -> None:
        """Begin a new sleep session."""
        self._in_sleep_session = True
        self._consolidation_done_this_sleep = False
        self._sleep_session = SleepSessionStats(energy_start=energy)

        self.trace_store.sync_runtime_state(self._recent_trace_ids)

        # Build dream queue: sort by emotional importance + novelty
        candidates: list[Trace] = []
        for tid in self._recent_trace_ids:
            trace = self.trace_store.get(tid)
            if trace is not None:
                candidates.append(trace)

        # Prioritize: high |polarity|, then high novelty, then recent
        candidates.sort(
            key=lambda t: (abs(t.polarity), t.novelty, t.fire_count),
            reverse=True,
        )

        self._dream_queue = [t.id for t in candidates]
        self._dream_idx = 0

    def _end_sleep_session(self, energy: float) -> None:
        """End the current sleep session."""
        self._in_sleep_session = False
        self._sleep_session.energy_end = energy
        # Clear recent traces — they've been processed
        self._recent_trace_ids = []

    def _dream_replay_step(self) -> DreamReplayStats:
        """Replay traces during REM: reactivate stored patterns.

        This strengthens memory_long neurons and associated synapses,
        similar to consolidation but lighter-weight and focused on
        pattern reactivation rather than transfer.
        """
        stats = DreamReplayStats()

        for _ in range(self._dreams_per_tick):
            if self._dream_idx >= len(self._dream_queue):
                # Wrap around with some randomization
                if self._dream_queue:
                    random.shuffle(self._dream_queue)
                    self._dream_idx = 0
                else:
                    break

            tid = self._dream_queue[self._dream_idx]
            self._dream_idx += 1

            trace = self.trace_store.get(tid)
            if trace is None:
                continue

            # Reactivate trace neurons at reduced strength (dream-level)
            all_neurons = []
            for region_name, neuron_ids in trace.neurons.items():
                for nid in neuron_ids:
                    all_neurons.append((nid, 0.3))  # 30% activation = dream strength

            if all_neurons:
                brain_core.inject_activations(all_neurons)
                stats.traces_replayed += 1
                stats.neurons_activated += len(all_neurons)

            # Strengthen memory_long trace neurons (mild reinforcement)
            ml_neurons = trace.neurons.get("memory_long", [])
            if ml_neurons:
                brain_core.strengthen_memory_trace(ml_neurons, 0.1)
                stats.synapses_strengthened += len(ml_neurons)

            # Slight strength boost from dream replay
            trace.strength = min(1.0, trace.strength + 0.005)
            self.trace_store.sync_trace(trace.id)

        self._sleep_session.traces_replayed += stats.traces_replayed
        return stats
