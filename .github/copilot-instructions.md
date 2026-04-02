# AI_BRAINS — Artificial Brain

## What This Is

A biologically-inspired artificial brain: 152k neurons across 14 regions, connected by synapses, bindings, and traces. Rust core for real-time signal propagation, Python for learning policy and I/O. NOT a neural network — no backprop, no loss function, no training loop.

## Architecture

```
4-Level Hierarchy:
  Neurons → Synapses → Bindings → Traces

14 Regions (152k neurons total):
  Input:      sensory, visual, audio
  Processing: memory_short, memory_long, emotion, attention,
              pattern, integration, language, executive
  Output:     motor, speech
  Special:    numbers (hardwired)

Signal Flow:
  Inputs → Attention (gain control) → Pattern Recognition
  → Integration → Memory/Emotion/Language → Executive → Motor/Speech
```

## Tech Stack

- **Rust** (`brain/src/core/`, `brain/src/regions/`): neuron firing, synapse propagation, tick cycle. Uses PyO3 for Python bindings (lib name: `brain_core`), rayon for parallelism.
- **Python** (`brain/structures/`, `brain/learning/`, `brain/input/`, `brain/output/`, `brain/seed/`): trace management, learning rules, I/O pipelines, seeding.
- `brain/__init__.py` re-exports from the Rust `brain_core` module so Python code can `import brain` or `import brain_core`.
- Synapse storage: CSR (Compressed Sparse Row) format.
- Neuron model: Leaky Integrate-and-Fire (LIF).
- All hyperparameters live in `brain/utils/config.py`.

## Key Concepts

- **Trace**: a concept = set of neuron IDs across regions + metadata (strength, polarity, novelty, etc.)
- **Binding**: pattern↔pattern link across regions (not neuron-level, pattern-level)
- **Pruning lifecycle**: BLOOM (grow) → CRITICAL (aggressive prune) → MATURE (slow growth)
- **Sparsity**: <5% of neurons fire per tick. Use active lists, never iterate all neurons.

## Conventions

- Rust code prioritizes performance: SoA layout, SIMD-friendly, sparse iteration.
- Python code prioritizes readability and iteration speed.
- Config values are never hardcoded in Rust or Python modules — import from `config.py` or pass as constructor args from Python.
- Tests go in `#[cfg(test)]` modules in Rust, and `brain/tests/` for Python.
- Full architecture details: see `PLAN.md` at project root.

## Build & Test

```bash
# Rust
cd brain && cargo build --release && cargo test

# Python (after Rust builds)
cd brain && maturin develop --release
cd .. && python -m pytest brain/tests/
```

## Current Progress

- **Phase 1 (Foundation)**: COMPLETE — 32 Rust tests pass
  - `brain/src/core/`: neuron.rs, region.rs, synapse.rs, propagate.rs, tick.rs, brain.rs
  - `brain/src/lib.rs`: PyO3 bridge (27 functions exposed)

- **Phase 2 (Seed & Structure)**: COMPLETE — 26 Python tests pass
  - `brain/utils/config.py`: all hyperparameters
  - `brain/structures/neuron_map.py`: region ranges, type assignments, global↔local conversion
  - `brain/structures/trace_store.py`: Trace dataclass, TraceStore with inverted index
  - `brain/seed/spawn_neurons.py`: neuron count verification
  - `brain/seed/spawn_synapses.py`: within-region + cross-region synapse generation
  - `brain/seed/spawn_traces.py`: 100k random traces
  - `brain/seed/seed_runner.py`: orchestrates full seed procedure

- **Phase 3 (Learning Core)**: COMPLETE — 30 Python tests pass
  - `brain/structures/brain_state.py`: ActivationSnapshot, ActivationHistory, NeuromodulatorState
  - `brain/learning/hebbian.py`: co-activation detection, synapse strengthening
  - `brain/learning/anti_hebbian.py`: decorrelation, synapse weakening
  - `brain/learning/pruning.py`: phase-aware pruning (BLOOM/CRITICAL/MATURE)
  - `brain/learning/novelty.py`: prediction error, novelty scoring, neuromod modulation
  - `brain/learning/tick_loop.py`: full tick orchestrator (Rust tick + Python learn/maintain)

- **Phase 4 (Attention & Pattern Recognition)**: COMPLETE — 48 Rust tests, 35 Python tests
  - `brain/src/core/attention.rs`: AttentionSystem with three-drive gain map (novelty/threat/relevance), inertia, clamping
  - `brain/src/regions/attention.rs`: compute_threat_drive (from emotion), compute_relevance_drive (from executive)
  - `brain/src/regions/pattern.rs`: PredictionState (EMA-based), ErrorClass enum, classify_error
  - `brain/src/core/brain.rs`: integrated AttentionSystem + PredictionState into Brain, auto-computes drives per tick
  - `brain/src/lib.rs`: 4 new PyO3 functions (set_attention_drives, get_attention_gains, get_prediction_errors, get_global_prediction_error)
  - `brain/learning/prediction.py`: PredictionEngine (trace-based prediction, error effects, surprise/alarm timers, learning rate multiplier)
  - `brain/learning/tick_loop.py`: updated — uses PredictionEngine, three-drive attention, prediction_multiplier for Hebbian learning

- **Phase 5 (Integration & Memory)**: COMPLETE — 65 Rust tests, 128 Python tests (193 total)
  - `brain/src/core/binding.rs`: BindingStore, PatternRef, Binding — cross-region pattern links with activation, strengthening, dissolution
  - `brain/src/regions/integration.rs`: count_active_input_regions, integration_strength, boost_integration_neurons
  - `brain/src/regions/memory_short.rs`: boost_working_memory_neurons, suppress_non_trace_neurons, active_neuron_count
  - `brain/src/regions/memory_long.rs`: pattern_completion (40% threshold), strengthen_trace_neurons
  - `brain/src/core/brain.rs`: binding + memory methods (create/evaluate/strengthen/prune bindings, pattern_complete, boost_working_memory, strengthen_memory_trace, boost_integration)
  - `brain/src/lib.rs`: 13 new PyO3 functions for bindings, memory, and integration
  - `brain/learning/consolidation.py`: ConsolidationEngine — short→long memory transfer (energy/tick triggers, trace replay, context stripping)
  - `brain/learning/trace_formation.py`: TraceFormationEngine, NovelPatternTracker — persistent novel patterns → new traces
  - `brain/learning/binding_formation.py`: BindingFormationEngine, CoActivationTracker — co-active cross-region patterns → bindings
  - `brain/learning/tick_loop.py`: updated — WorkingMemory class (capacity=7, decay), integration boost, pattern completion, trace/binding formation, consolidation

- **Phase 6 (Emotion & Executive)**: COMPLETE — 92 Rust tests, 159 Python tests (251 total)
  - `brain/src/core/neuromodulator.rs`: NeuromodulatorSystem — global arousal/valence/focus/energy, threshold modifier, EMA updates from emotion, energy depletion/recovery
  - `brain/src/regions/emotion.rs`: compute_polarity (positive/negative neuron populations), compute_arousal, compute_urgency, emotion_motor_impulse (direct emotion→motor bypass)
  - `brain/src/regions/executive.rs`: executive_engagement, detect_motor_conflict, resolve_motor_conflict (suppress weaker side), inhibit_motor_neurons (impulse suppression), planning_signal (exec×language)
  - `brain/src/core/brain.rs`: integrated NeuromodulatorSystem, auto-computes emotion/executive state per tick, conflict resolution and impulse suppression in tick loop
  - `brain/src/lib.rs`: 14 new PyO3 functions (set/get_neuromodulator, get_threshold_modifier, get_emotion_polarity/arousal/urgency/motor_impulse, get_executive_engagement, get/resolve_motor_conflict, inhibit_motor, get_planning_signal, recover_energy)
  - `brain/learning/tick_loop.py`: updated — syncs Python↔Rust neuromodulator state, reads emotion polarity/arousal per tick, trace polarity tagging, energy recovery during consolidation, returns Phase 6 metrics

- **Phase 7 (Language & Speech)**: COMPLETE — 112 Rust tests, 197 Python tests (309 total)

- **Phase 8 (Full Input/Output)**: COMPLETE — 149 Rust tests, 228 Python tests (377 total)
  - `brain/src/regions/sensory.rs`: population_code (gaussian bump encoding), encode_sensory (temperature/pressure/pain/texture), sensory_activation_strength, boost_sensory_neurons, peak_sensory_neurons, detect_pain_level — sub-ranges: TEMP 0–2499, PRESSURE 2500–4999, PAIN 5000–7499, TEXTURE 7500–9999
  - `brain/src/regions/visual.rs`: VisualSubRegion enum (Low/Mid/High/Spatial), visual_activation_strength, sub_region_activation, boost_visual_neurons, peak_visual_neurons (with optional sub_region filter), read_visual_activations (for imagination) — sub-regions: LOW 10000–14999, MID 15000–19999, HIGH 20000–24999, SPATIAL 25000–29999
  - `brain/src/regions/audio.rs`: AudioSubRegion enum (Freq/Temporal/Complex), audio_activation_strength, sub_region_activation, boost_audio_neurons, peak_audio_neurons, frequency_to_neurons (log-scale Hz mapping) — sub-regions: FREQ 30000–34999, TEMPORAL 35000–39999, COMPLEX 40000–44999
  - `brain/src/regions/motor.rs`: MotorAction enum (Approach/Withdraw/Idle/Conflict), motor_activation_strength, approach_vs_withdraw, decode_motor_action, peak_motor_neurons, motor_lateral_inhibition, boost_motor_neurons — sub-pops: APPROACH 130000–134999, WITHDRAW 135000–137999, INHIBITORY 138000–139999
  - `brain/src/core/brain.rs`: ~30 new methods wrapping all 4 new region modules
  - `brain/src/lib.rs`: 19 new PyO3 functions (encode_sensory, get_sensory_activation, boost_sensory, get_peak_sensory_neurons, get_pain_level, get_visual_activation, boost_visual, get_peak_visual_neurons, read_visual_activations, get_audio_activation, boost_audio, get_peak_audio_neurons, frequency_to_neurons, get_motor_activation, get_approach_vs_withdraw, decode_motor_action, get_peak_motor_neurons, motor_lateral_inhibition, boost_motor)
  - `brain/input/sensory_input.py`: SensoryInput — Rust population coding for temperature/pressure/pain/texture, inject + boost
  - `brain/input/visual_input.py`: VisualInput — image frame → edge/shape/spatial extraction, hierarchical sub-region encoding, 32×32 target resolution
  - `brain/input/audio_input.py`: AudioInput — Goertzel DFT for frequency decomposition, onset detection, zero-crossing rate for timbre
  - `brain/input/multimodal.py`: MultimodalInput — synchronize text/visual/audio/sensory streams, auto-boost integration when ≥2 modalities active
  - `brain/output/motor_output.py`: MotorOutput — lateral inhibition → decode_motor_action → peak neurons, MotorAction dataclass
  - `brain/output/imagination.py`: ImaginationOutput — read visual activations, reconstruct 32×32 grayscale image, extract spatial focus, report sub-region levels
  - `brain/seed/physics_traces.py`: ~200 physics seed traces (gravity, collision, momentum, temperature, etc.) across sensory/visual/pattern/language/memory_long
  - `brain/seed/relational_traces.py`: ~200 relational seed traces (cause/effect, part/whole, sequence, hierarchy) across language-relational/pattern/integration/executive/memory_long
  - `brain/seed/numbers_wiring.py`: Hardwired numbers region — 100 number clusters (0–99), 20 neurons each, within-cluster recurrent + successor/predecessor connections, number traces with labels
  - `brain/seed/reflex_wiring.py`: Hardwired reflex pathways — pain→withdraw (100 synapses), hot→withdraw (50), pressure→approach (30), delay=2, plasticity=0.05
  - `brain/seed/seed_runner.py`: updated — orchestrates physics/relational traces, number traces, number wiring, reflex wiring in seed procedure
  - `brain/learning/tick_loop.py`: updated — reads sensory/visual/audio/motor activations per tick, decodes motor action, reads pain level, returns Phase 8 metrics

- **Phase 9 (Homeostasis & Sleep)**: COMPLETE — 179 Rust tests, 262 Python tests (441 total)
  - `brain/src/core/homeostasis.rs`: HomeostasisSystem — arousal/valence/focus regulation toward baselines (EMA), sleep pressure accumulation/dissipation, circadian phase cycle (100k ticks), circadian energy modifier, should_sleep/should_wake thresholds
  - `brain/src/core/sleep.rs`: SleepCycleManager, SleepState enum (Awake/Drowsy/Light/Deep/Rem) — state machine with duration-based transitions, input gating per state (Awake=1.0, Drowsy=0.5, Light=0.2, Deep=0.05, Rem=0.1), energy recovery rates per state, cycle counting, REM episode tracking, force_wake
  - `brain/src/core/brain.rs`: integrated HomeostasisSystem + SleepCycleManager into Brain struct and tick() — circadian advance, sleep state update, homeostatic regulation of neuromodulators, sleep pressure accumulate/dissipate, sleep input gating on sensory/visual/audio attention gains, circadian-modulated energy depletion
  - `brain/src/lib.rs`: 10 new PyO3 functions (get_homeostasis_summary, get_sleep_summary, is_asleep, in_rem, force_wake, get_sleep_input_gate, get_sleep_pressure, get_circadian_phase, set_homeostasis_params, set_sleep_durations)
  - `brain/learning/homeostasis.py`: HomeostasisManager — dream replay during REM (trace reactivation at 30% strength, memory_long reinforcement), autonomous consolidation scheduling tied to deep sleep phase, wake alarm on high pain, sleep session tracking, recent trace recording for dream content
  - `brain/utils/config.py`: 12 new constants (HOMEOSTASIS_AROUSAL/VALENCE/FOCUS_REG_RATE, SLEEP_PRESSURE/DISSIPATION_RATE, CIRCADIAN_PERIOD, SLEEP_DROWSY/LIGHT/DEEP/REM_DURATION, DREAM_REPLAY_PER_TICK, WAKE_ALARM_PAIN_THRESHOLD)
  - `brain/learning/tick_loop.py`: updated — integrates HomeostasisManager, records active traces while awake, dream replay during REM, sleep-triggered consolidation during deep sleep, returns Phase 9 metrics (sleep_state, sleep_pressure, circadian_phase, is_asleep, in_rem, dream_replayed)

- **Phase 10 (Parallelism & Deep Integration)**: COMPLETE — 179 Rust tests, 312 Python tests (491 total)
  - `brain/src/core/propagate.rs`: rayon parallelism — `par_iter()` over regions for signal collection, parallel CSR traversal per active neuron
  - `brain/src/core/tick.rs`: rayon parallelism — `par_iter_mut()` for pre_tick/update_neurons, parallel reduce for stats collection
  - `brain/src/core/synapse.rs`: added `num_neurons()` public getter for CSR bounds checking
  - `brain/src/lib.rs`: 5 new PyO3 functions:
    - `set_num_threads(n)` / `get_num_threads()` — rayon thread pool control (must call before first tick)
    - `batch_hebbian(active_neurons, window_active, learning_rate)` — parallel Hebbian synapse traversal via rayon, replaces per-neuron Python FFI loop
    - `batch_anti_hebbian(active_neurons, window_active, rate)` — parallel anti-Hebbian synapse traversal
    - `batch_track_coactive(active_neurons, active_set)` — parallel co-active synapse pair detection for pruning
  - `brain/learning/hebbian.py`: updated — delegates to `brain_core.batch_hebbian()` (single FFI call, parallel Rust loop)
  - `brain/learning/anti_hebbian.py`: updated — delegates to `brain_core.batch_anti_hebbian()` (single FFI call, parallel Rust loop)
  - `brain/learning/tick_loop.py`: updated — `_track_synapse_fires()` uses `brain_core.batch_track_coactive()` instead of per-neuron Python loop
  - `brain/tests/test_phase10.py`: 50 deep integration tests across 10 categories:
    - Parallelism (5): thread count, parallel tick, high-activity multi-region parallel
    - Text pipeline (5): encoding, tick integration, speech output, unknown token hashing
    - Image pipeline (7): gradient/checkerboard/circle encoding, imagination roundtrip, activation differences
    - Audio pipeline (7): sine wave, chord, noise, onset detection, silence handling
    - Sensory-motor reflex (4): population coding, pain detection, reflex wiring, temperature gradient
    - Image+caption multimodal (4): integration boost, language+visual co-activation, triple modality
    - Video pipeline (4): frame sequence, moving object, tick loop integration, long video stability
    - Video+audio multimodal (4): synchronized processing, integration boost, triple+quad modality
    - Full lifecycle (6): fast seed smoke, seeded tick, multi-step TickLoop, Hebbian learning, full I/O, full seed_brain
    - Performance (4): tick throughput (empty/active), thread count query, multimodal throughput

## Next Phase

- **Phase 11 (Polish & Observe)**: IN PROGRESS — 179 Rust tests, 312 Python tests (491 total)
  - `main.py`: CLI entry point for real learning with HuggingFace datasets. Supports --dataset (ag_news/imdb/cifar10/speech_commands/multimodal/all), --samples, --ticks, --threads, --mode (sequential/separate), --compare, --fast, --full-seed, --seed-traces, --rest-ticks, --compact, --save-brain, --output
  - `brain/metrics/collector.py`: MetricsCollector — per-tick/per-sample/global aggregation, JSON save, compact mode
  - `brain/datasets/downloader.py`: load_text_dataset (ag_news, imdb), load_image_dataset (cifar10), load_audio_dataset (speech_commands), load_multimodal_batch, images_to_video_frames
  - `brain/serialize/brain_saver.py`: save_brain/load_brain (traces.json + state.json + metadata.json)
  - `brain/seed/seed_runner.py`: added seed_brain_fast(n_traces=5000) — lightweight seed for interactive learning (4.5s vs 23s)
  - `brain/src/lib.rs`: 3 new PyO3 functions:
    - `batch_learn_step(active_neurons, window_active, hebbian_rate, anti_hebbian_rate)` — combined hebbian + anti-hebbian + coactive tracking in single FFI call with rayon parallelism
    - `batch_set_attention_drives(drives)` — set all region drives in one call (replaces 14× per-region calls)
    - `batch_read_state()` — read all brain state (activations, emotion, executive, motor, pain, language) in one call (replaces 15+ individual FFI calls)
  - `brain/learning/tick_loop.py`: updated — uses batch_learn_step (single FFI call for all learning), batch_set_attention_drives, batch_read_state; fixed arousal sync bug (Python arousal no longer overwritten by Rust); computes effective learning rates inline
  - `brain/learning/prediction.py`: updated — "interesting" classification now boosts arousal proportionally (was no-op)
  - `brain/learning/trace_formation.py`: updated — persistence lowered to 3 ticks (was 20), removed strong-match blocking (small known traces no longer prevent learning), low novelty skips tracking without clearing persistence
  - `brain/utils/config.py`: TRACE_FORMATION_PERSISTENCE = 3 (was 20)
  - Verified: 50-sample ag_news learning — 5 traces formed, arousal=1.0, 203M hebbian updates, 330ms/tick avg
  - Remaining: image/audio/multimodal learning validation, sequential vs separate comparison, long-run stability, metrics dashboard
