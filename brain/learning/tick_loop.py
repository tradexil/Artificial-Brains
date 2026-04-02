"""Python-side tick loop orchestrator.

Wraps the Rust tick (Phase 1: propagate + Phase 2: update) with
Python-driven Phase 2 (evaluate) + Phase 3 (learn) + Phase 4 (maintain).

Tick phases:
  Phase 1 (Rust): propagate signals, deliver delayed, update neurons,
                   update attention gains (three-drive), compute prediction errors
  Phase 2 (Python): evaluate — trace matching, prediction, attention drives,
                     binding evaluation, integration boost
  Phase 3 (Python): learn — Hebbian, anti-Hebbian, trace formation, binding formation
  Phase 4 (Python): maintain — apply weight updates, rebuild CSR, prune,
                     consolidation
"""

from __future__ import annotations

import brain_core

from brain.learning.hebbian import compute_effective_learning_rate
from brain.learning.anti_hebbian import compute_anti_hebbian_rate
from brain.learning.prediction import PredictionEngine
from brain.learning.pruning import get_phase, pruning_pass_sampled
from brain.learning.consolidation import ConsolidationEngine
from brain.learning.trace_formation import TraceFormationEngine
from brain.learning.binding_formation import BindingFormationEngine
from brain.learning.homeostasis import HomeostasisManager
from brain.structures.brain_state import (
    ActivationHistory,
    ActivationSnapshot,
    NeuromodulatorState,
)
from brain.structures.trace_store import TraceStore
from brain.utils.config import (
    HEBBIAN_WINDOW,
    TRACE_ACTIVATION_THRESHOLD,
    WORKING_MEMORY_CAPACITY,
    WORKING_MEMORY_DECAY_RATE,
)
from brain.structures.neuron_map import all_region_names


class WorkingMemory:
    """Capacity-limited buffer of active trace IDs."""

    def __init__(self, capacity: int = WORKING_MEMORY_CAPACITY):
        self.capacity = capacity
        # (trace_id, activation_strength) — most recent is strongest
        self.slots: list[tuple[str, float]] = []

    def update(self, active_traces: list[tuple[str, float]]) -> None:
        """Update working memory with currently active traces.

        Active traces refresh/enter slots. All slots decay slightly.
        If over capacity, weakest is evicted.
        """
        # Decay existing slots
        self.slots = [
            (tid, strength * (1.0 - WORKING_MEMORY_DECAY_RATE))
            for tid, strength in self.slots
        ]

        # Refresh or add active traces
        active_map = dict(active_traces)
        existing_ids = {tid for tid, _ in self.slots}

        for tid, score in active_traces:
            if tid in existing_ids:
                # Refresh — update strength
                self.slots = [
                    (t, max(s, score)) if t == tid else (t, s)
                    for t, s in self.slots
                ]
            else:
                self.slots.append((tid, score))

        # Sort by strength descending
        self.slots.sort(key=lambda x: x[1], reverse=True)

        # Enforce capacity
        if len(self.slots) > self.capacity:
            self.slots = self.slots[: self.capacity]

        # Remove dead entries
        self.slots = [(t, s) for t, s in self.slots if s > 0.01]

    @property
    def trace_ids(self) -> list[str]:
        return [tid for tid, _ in self.slots]

    def __len__(self) -> int:
        return len(self.slots)


class TickLoop:
    """Orchestrates the full tick cycle: Rust tick + Python learning.

    Usage:
        loop = TickLoop(trace_store)
        for i in range(N):
            brain_core.inject_activations(signals)
            result = loop.step()
    """

    def __init__(self, trace_store: TraceStore):
        self.trace_store = trace_store
        self.history = ActivationHistory(window=HEBBIAN_WINDOW)
        self.neuromod = NeuromodulatorState()
        self.prediction = PredictionEngine(trace_store)
        self.working_memory = WorkingMemory()

        # Phase 5 engines
        self.consolidation = ConsolidationEngine()
        self.trace_formation = TraceFormationEngine(trace_store)
        self.binding_formation = BindingFormationEngine(trace_store)

        # Phase 9: homeostasis manager
        self.homeostasis = HomeostasisManager(trace_store)

        # Track all traces that fired during this awake period (for consolidation)
        self._awake_trace_ids: list[str] = []

        # Stats per step
        self.last_tick_number: int = 0
        self.last_total_active: int = 0
        self.last_novelty: float = 0.0
        self.last_hebbian_updates: int = 0
        self.last_anti_hebbian_updates: int = 0
        self.last_active_traces: list[tuple[str, float]] = []

        # Synapse fire tracking for pruning (sparse — only updated during learning)
        self._synapse_last_fired: dict[tuple[int, int], int] = {}

        # Pruning rotation counter
        self._prune_offset: int = 0
        self._prune_batch_size: int = 10_000  # neurons per pruning pass

    def step(self) -> dict:
        """Execute one full tick cycle.

        Returns dict with tick stats.
        """
        # === Pre-tick: Predict what should happen ===
        self.prediction.predict(
            self.last_active_traces,
            self.working_memory.trace_ids,
        )

        # === Phase 1: Rust tick ===
        # (propagate + update + attention gain update + prediction error)
        tick_num, active_counts, total_active = brain_core.tick()
        self.last_tick_number = tick_num
        self.last_total_active = total_active

        # === Phase 2: Evaluate (snapshot, trace matching, prediction, drives) ===
        snapshot = self.history.take_snapshot(brain_core)

        # Read all brain state in one FFI call (used by Phase 5, 6, 7, 8)
        state = brain_core.batch_read_state()

        # Trace activation detection
        active_ids = snapshot.all_active_ids()
        active_traces = self.trace_store.matching_traces(
            active_ids, threshold=TRACE_ACTIVATION_THRESHOLD
        )
        self.last_active_traces = active_traces

        # Update trace metadata for firing traces
        for tid, score in active_traces:
            trace = self.trace_store.get(tid)
            if trace is not None:
                trace.fire_count += 1
                trace.last_fired = tick_num
                trace.strength = min(1.0, trace.strength + 0.005 * score)
                trace.novelty = max(0.0, trace.novelty - 0.01)

        # Working memory
        self.working_memory.update(active_traces)

        # Prediction error (trace-based)
        errors = self.prediction.compute_errors(snapshot)
        novelty = self.prediction.global_error(errors)
        self.last_novelty = novelty
        self.prediction.apply_effects(novelty, self.neuromod)

        # Feed novelty drives to Rust attention system (batch — single FFI call)
        drives = {}
        for region in all_region_names():
            error = errors.get(region, 0.0)
            drives[region] = (error, 0.0, 0.0)
        brain_core.batch_set_attention_drives(drives)

        # === Phase 3: Learn (Hebbian + anti-Hebbian + coactive tracking — single FFI call) ===
        current = self.history.current
        if current is not None:
            # Compute effective learning rates
            hebbian_lr = compute_effective_learning_rate(
                tick_num, self.neuromod, novelty
            )
            hebbian_lr *= self.prediction.learning_rate_multiplier
            anti_hebbian_rate = compute_anti_hebbian_rate(tick_num)

            # Flatten all active neurons from current snapshot
            active_neurons: list[tuple[int, float]] = []
            for neurons in current.active_neurons.values():
                active_neurons.extend(neurons)

            if active_neurons:
                # Collect window-active neurons
                window_active = self.history.neurons_active_in_window()

                # Single Rust FFI call: hebbian + anti-hebbian + coactive tracking
                hebb_count, anti_count, coactive_pairs = brain_core.batch_learn_step(
                    active_neurons, window_active, hebbian_lr, anti_hebbian_rate,
                )

                # Update synapse fire tracking for pruning
                for src_id, tgt_id in coactive_pairs:
                    self._synapse_last_fired[(src_id, tgt_id)] = tick_num
            else:
                hebb_count = 0
                anti_count = 0
        else:
            hebb_count = 0
            anti_count = 0

        self.last_hebbian_updates = hebb_count
        self.last_anti_hebbian_updates = anti_count

        # === Phase 5: Integration & Memory ===

        # Integration boost based on multi-modal convergence (from batch_read_state)
        n_input_regions = int(state.get("integration_input_count", 0.0))
        if n_input_regions >= 2:
            from brain.utils.config import REGIONS
            integration_size = REGIONS["integration"][1] - REGIONS["integration"][0] + 1
            strength = min(1.0, n_input_regions / 6.0)
            brain_core.boost_integration(strength, min(100, integration_size))

        # Working memory neuron boost (Rust-side)
        wm_neurons = self._get_working_memory_neurons()
        if wm_neurons:
            brain_core.boost_working_memory(wm_neurons, 0.15)

        # Pattern completion in memory_long for active traces
        for tid, score in active_traces:
            trace = self.trace_store.get(tid)
            if trace is not None:
                ml_neurons = trace.neurons.get("memory_long", [])
                if ml_neurons:
                    brain_core.pattern_complete(ml_neurons, 0.4, 0.3)

        # Track awake trace IDs for consolidation
        for tid, _ in active_traces:
            if tid not in self._awake_trace_ids:
                self._awake_trace_ids.append(tid)

        # Trace formation: detect persistent novel patterns
        traces_formed = self.trace_formation.step(
            snapshot,
            active_traces,
            novelty,
            tick_num,
            len(self.working_memory),
            co_trace_ids=self.working_memory.trace_ids,
        )

        # Binding formation: detect co-active cross-region patterns
        binding_stats = self.binding_formation.step(
            active_traces, tick_num, self.history,
        )

        # === Phase 6: Emotion & Executive ===

        # Save Python-computed arousal (from prediction engine) before Rust sync
        python_arousal = self.neuromod.arousal

        # Sync Python neuromodulator state TO Rust
        brain_core.set_neuromodulator(
            self.neuromod.arousal,
            self.neuromod.valence,
            self.neuromod.focus,
            self.neuromod.energy,
        )

        # Read back Rust-computed state (emotion-driven arousal, energy depletion)
        rust_arousal, rust_valence, rust_focus, rust_energy = brain_core.get_neuromodulator()

        # Combine: Python arousal (prediction-based) + Rust arousal (emotion-based)
        # Use max to preserve signal from either source
        self.neuromod.arousal = max(python_arousal, rust_arousal)
        self.neuromod.valence = rust_valence
        self.neuromod.energy = rust_energy

        emotion_polarity = state.get("emotion_polarity", 0.0)
        emotion_arousal = state.get("emotion_arousal", 0.0)

        # Tag active traces with emotional polarity
        if abs(emotion_polarity) > 0.1:
            for tid, score in active_traces:
                trace = self.trace_store.get(tid)
                if trace is not None:
                    # Slowly shift trace polarity toward current emotion
                    trace.polarity = trace.polarity * 0.95 + emotion_polarity * 0.05

        # Read executive state (from batch_read_state)
        exec_engagement = state.get("executive_engagement", 0.0)
        motor_conflict = state.get("motor_conflict", 0.0)
        planning = state.get("planning_signal", 0.0)

        # === Phase 7: Language & Speech ===

        # Read language activation and inner monologue signal (from batch_read_state)
        language_activation = state.get("language_activation", 0.0)
        inner_monologue = state.get("inner_monologue", 0.0)
        speech_activity = state.get("speech_activity", 0.0)

        # If language is active, boost speech neurons for active traces
        # (language → speech pathway)
        if language_activation > 0.1:
            for tid, score in active_traces:
                trace = self.trace_store.get(tid)
                if trace is not None:
                    speech_neurons = trace.neurons.get("speech", [])
                    if speech_neurons:
                        brain_core.boost_speech(
                            speech_neurons, 0.2 * score * language_activation
                        )

        # === Phase 8: Full I/O ===

        # Read sensory, visual, audio, motor activations (from batch_read_state)
        sensory_activation = state.get("sensory_activation", 0.0)
        visual_activation = state.get("visual_activation", 0.0)
        audio_activation = state.get("audio_activation", 0.0)
        motor_activation = state.get("motor_activation", 0.0)

        # Read motor action from batch state
        motor_approach = state.get("motor_approach", 0.0)
        motor_withdraw = state.get("motor_withdraw", 0.0)
        if motor_approach > 0 and motor_withdraw > 0:
            motor_action_str = "conflict"
        elif motor_approach > 0:
            motor_action_str = "approach"
        elif motor_withdraw > 0:
            motor_action_str = "withdraw"
        else:
            motor_action_str = "idle"

        # Pain detection
        pain_level = state.get("pain_level", 0.0)

        # === Phase 9: Homeostasis & Sleep ===

        # Record active traces for dream replay content
        if not brain_core.is_asleep():
            self.homeostasis.record_active_traces(
                [tid for tid, _ in active_traces]
            )

        # Homeostasis step (dream replay during REM, wake alarm, etc.)
        homeo_stats = self.homeostasis.step(tick_num, self.neuromod)

        # Consolidation: integrate with sleep cycle
        # During sleep: consolidation can be triggered by deep sleep phase
        # While awake: original energy/tick triggers still apply
        if self.consolidation.is_consolidating:
            consol_stats = self.consolidation.consolidation_step(tick_num, self.trace_store)
            # Recover energy during consolidation
            brain_core.recover_energy(0.001)
            self.neuromod.energy = min(1.0, self.neuromod.energy + 0.001)
            if consol_stats.get("finished"):
                self._awake_trace_ids = []  # Reset awake tracking
                self.homeostasis.mark_consolidation_done()
        elif brain_core.is_asleep() and self.homeostasis.should_consolidate_in_sleep():
            # Sleep-triggered consolidation (during deep sleep)
            self.consolidation.start_consolidation(
                tick_num, self.trace_store,
                self.homeostasis.get_dream_candidates(),
            )
        elif not brain_core.is_asleep() and self.consolidation.should_consolidate(tick_num, self.neuromod):
            # Normal awake-triggered consolidation (energy/tick based)
            self.consolidation.start_consolidation(
                tick_num, self.trace_store, self._awake_trace_ids,
            )

        # === Phase 4: Maintain (periodic) ===
        self._maintain(tick_num)

        return {
            "tick": tick_num,
            "total_active": total_active,
            "active_traces": len(active_traces),
            "working_memory": len(self.working_memory),
            "novelty": novelty,
            "hebbian_updates": hebb_count,
            "anti_hebbian_updates": anti_count,
            "phase": get_phase(tick_num),
            "arousal": self.neuromod.arousal,
            "valence": self.neuromod.valence,
            "energy": self.neuromod.energy,
            "in_surprise": self.prediction.in_surprise,
            "in_alarm": self.prediction.in_alarm,
            "learning_multiplier": self.prediction.learning_rate_multiplier,
            "traces_formed": traces_formed,
            "bindings_formed": binding_stats.get("formed", 0),
            "total_bindings": binding_stats.get("total_bindings", 0),
            "consolidating": self.consolidation.is_consolidating,
            # Phase 6
            "emotion_polarity": emotion_polarity,
            "emotion_arousal": emotion_arousal,
            "executive_engagement": exec_engagement,
            "motor_conflict": motor_conflict,
            "planning_signal": planning,
            # Phase 7
            "language_activation": language_activation,
            "inner_monologue": inner_monologue,
            "speech_activity": speech_activity,
            # Phase 8
            "sensory_activation": sensory_activation,
            "visual_activation": visual_activation,
            "audio_activation": audio_activation,
            "motor_activation": motor_activation,
            "motor_action": motor_action_str,
            "motor_approach": motor_approach,
            "motor_withdraw": motor_withdraw,
            "pain_level": pain_level,
            # Phase 9
            "sleep_state": homeo_stats.get("sleep_state", "awake"),
            "sleep_pressure": homeo_stats.get("sleep_pressure", 0.0),
            "circadian_phase": homeo_stats.get("circadian_phase", 0.0),
            "is_asleep": homeo_stats.get("is_asleep", False),
            "in_rem": homeo_stats.get("in_rem", False),
            "dream_replayed": homeo_stats.get("dream_replayed", 0),
        }

    def _track_synapse_fires(self, snapshot: ActivationSnapshot, tick: int) -> None:
        """Track last-fired tick for synapses between active neurons.

        Uses Rust batch_track_coactive for parallelized synapse traversal.
        """
        active_ids = snapshot.all_active_ids()
        if not active_ids:
            return
        active_set = set(active_ids)
        pairs = brain_core.batch_track_coactive(active_ids, active_set)
        for src_id, tgt_id in pairs:
            self._synapse_last_fired[(src_id, tgt_id)] = tick

    def _get_working_memory_neurons(self) -> list[int]:
        """Get all neurons belonging to working memory traces."""
        neurons = []
        for tid in self.working_memory.trace_ids:
            trace = self.trace_store.get(tid)
            if trace is not None:
                ms_neurons = trace.neurons.get("memory_short", [])
                neurons.extend(ms_neurons)
        return neurons

    def _maintain(self, tick: int) -> None:
        """Periodic maintenance: apply updates, prune, rebuild."""
        # Every 100 ticks: apply queued weight updates
        if tick % 100 == 0:
            brain_core.apply_synapse_updates()

        # Every 1000 ticks: incremental pruning pass
        if tick % 1000 == 0:
            from brain.utils.config import TOTAL_NEURONS
            start = self._prune_offset
            end = min(start + self._prune_batch_size, TOTAL_NEURONS)
            sample = list(range(start, end))
            pruning_pass_sampled(tick, sample, self._synapse_last_fired)
            self._prune_offset = end if end < TOTAL_NEURONS else 0

        # Every 5000 ticks: prune dissolved bindings + cleanup stale co-activation data
        if tick % 5000 == 0:
            self.binding_formation.periodic_prune()
            self.binding_formation.periodic_cleanup(tick)

        # Every 10000 ticks: full CSR rebuild + trace merge check
        if tick % 10000 == 0:
            brain_core.rebuild_synapse_index()
            self.trace_formation.merge_overlapping()

        # Slow energy decay
        self.neuromod.energy = max(0.0, self.neuromod.energy - 0.0001)
        self.neuromod.clamp()
