/// PyO3 bindings: expose the Rust brain core to Python.
///
/// Python calls these functions to:
///   - Create and manage the brain
///   - Inject signals, run ticks, read activations
///   - Manage synapses (create, update, prune, rebuild)
///   - Set attention gains

pub mod core;
pub mod regions;

use core::brain::Brain;
use core::region::RegionId;
use core::synapse::SynapseData;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

// Global brain instance, protected by mutex.
// Python holds no Rust pointers — all access goes through these functions.
static BRAIN: Mutex<Option<Brain>> = Mutex::new(None);

// === THREAD POOL CONTROL ===

/// Set the number of threads rayon uses for parallelism.
/// Must be called before the first tick() — once set, cannot be changed.
/// If not called, rayon defaults to all available CPU cores.
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| PyValueError::new_err(format!("Failed to set thread count: {}", e)))
}

/// Get the current number of threads in the rayon thread pool.
#[pyfunction]
fn get_num_threads() -> PyResult<usize> {
    Ok(rayon::current_num_threads())
}

// === BATCH LEARNING (Rust-side Hebbian / Anti-Hebbian) ===

use rayon::prelude::*;

/// Batch Hebbian update: for each active neuron, traverse outgoing synapses
/// and strengthen those whose target was also active in the window.
///
/// Uses rayon for parallelism — active neurons processed in parallel.
///
/// Returns: number of synapse updates queued.
#[pyfunction]
fn batch_hebbian(
    active_neurons: Vec<(u32, f32)>,
    window_active: HashMap<u32, f32>,
    learning_rate: f64,
) -> PyResult<usize> {
    let lr = learning_rate as f32;
    with_brain(|brain| {
        let nn = brain.synapse_pool.num_neurons();
        // Parallel phase: each active neuron independently collects updates
        let all_updates: Vec<Vec<(u32, u32, f32)>> = active_neurons
            .par_iter()
            .map(|&(src_id, src_act)| {
                let mut updates = Vec::new();
                if src_id >= nn {
                    return updates;
                }
                let start = brain.synapse_pool.offsets[src_id as usize] as usize;
                let end = brain.synapse_pool.offsets[src_id as usize + 1] as usize;

                for i in start..end {
                    let plasticity = brain.synapse_pool.plasticity[i];
                    if plasticity < 0.01 {
                        continue;
                    }
                    let tgt_id = brain.synapse_pool.targets[i];
                    if let Some(&tgt_act) = window_active.get(&tgt_id) {
                        if tgt_act > 0.0 {
                            let delta = lr * src_act * tgt_act * plasticity;
                            if delta > 0.0001 {
                                updates.push((src_id, tgt_id, delta));
                            }
                        }
                    }
                }
                updates
            })
            .collect();

        // Sequential phase: queue all updates
        let mut total = 0usize;
        for updates in all_updates {
            total += updates.len();
            for (from, to, delta) in updates {
                brain.synapse_pool.queue_update(from, to, delta);
            }
        }
        total
    })
}

/// Batch anti-Hebbian update: weaken synapses where source fires but target does not.
///
/// delta = -rate × src_act × (1 - tgt_act) × plasticity
///
/// Returns: number of synapse updates queued.
#[pyfunction]
fn batch_anti_hebbian(
    active_neurons: Vec<(u32, f32)>,
    window_active: HashMap<u32, f32>,
    rate: f64,
) -> PyResult<usize> {
    let r = rate as f32;
    with_brain(|brain| {
        let nn = brain.synapse_pool.num_neurons();
        let all_updates: Vec<Vec<(u32, u32, f32)>> = active_neurons
            .par_iter()
            .map(|&(src_id, src_act)| {
                let mut updates = Vec::new();
                if src_id >= nn {
                    return updates;
                }
                let start = brain.synapse_pool.offsets[src_id as usize] as usize;
                let end = brain.synapse_pool.offsets[src_id as usize + 1] as usize;

                for i in start..end {
                    let plasticity = brain.synapse_pool.plasticity[i];
                    if plasticity < 0.01 {
                        continue;
                    }
                    let tgt_id = brain.synapse_pool.targets[i];
                    let tgt_act = window_active.get(&tgt_id).copied().unwrap_or(0.0);
                    if tgt_act > 0.0 {
                        continue; // Target active — Hebbian handles this
                    }
                    let delta = -r * src_act * (1.0 - tgt_act) * plasticity;
                    if delta < -0.0001 {
                        updates.push((src_id, tgt_id, delta));
                    }
                }
                updates
            })
            .collect();

        let mut total = 0usize;
        for updates in all_updates {
            total += updates.len();
            for (from, to, delta) in updates {
                brain.synapse_pool.queue_update(from, to, delta);
            }
        }
        total
    })
}

/// Batch track co-active synapses: returns (src, tgt) pairs where both
/// source and target are in the active set. Used for pruning dormancy tracking.
#[pyfunction]
fn batch_track_coactive(
    active_neurons: Vec<u32>,
    active_set: std::collections::HashSet<u32>,
) -> PyResult<Vec<(u32, u32)>> {
    with_brain_ref(|brain| {
        let nn = brain.synapse_pool.num_neurons();
        let pairs: Vec<Vec<(u32, u32)>> = active_neurons
            .par_iter()
            .map(|&src_id| {
                let mut local = Vec::new();
                if src_id >= nn {
                    return local;
                }
                let start = brain.synapse_pool.offsets[src_id as usize] as usize;
                let end = brain.synapse_pool.offsets[src_id as usize + 1] as usize;

                for i in start..end {
                    let tgt_id = brain.synapse_pool.targets[i];
                    if active_set.contains(&tgt_id) {
                        local.push((src_id, tgt_id));
                    }
                }
                local
            })
            .collect();

        pairs.into_iter().flatten().collect()
    })
}

// === Phase 11: Combined learning step (single FFI call) ===

/// Combined learning step: hebbian + anti-hebbian + coactive tracking in one call.
/// Avoids 3 separate FFI round-trips, keeping rayon hot throughout.
///
/// Returns: (hebbian_count, anti_hebbian_count, coactive_pairs)
#[pyfunction]
fn batch_learn_step(
    active_neurons: Vec<(u32, f32)>,
    window_active: HashMap<u32, f32>,
    hebbian_rate: f64,
    anti_hebbian_rate: f64,
) -> PyResult<(usize, usize, Vec<(u32, u32)>)> {
    let h_lr = hebbian_rate as f32;
    let a_lr = anti_hebbian_rate as f32;
    with_brain(|brain| {
        let nn = brain.synapse_pool.num_neurons();
        let window_set: std::collections::HashSet<u32> = window_active.keys().copied().collect();

        // Parallel phase: each active neuron does hebbian + anti-hebbian + coactive
        let results: Vec<(Vec<(u32, u32, f32)>, Vec<(u32, u32, f32)>, Vec<(u32, u32)>)> = 
            active_neurons
                .par_iter()
                .map(|&(src_id, src_act)| {
                    let mut hebb_updates = Vec::new();
                    let mut anti_updates = Vec::new();
                    let mut coactive = Vec::new();
                    
                    if src_id >= nn {
                        return (hebb_updates, anti_updates, coactive);
                    }
                    
                    let start = brain.synapse_pool.offsets[src_id as usize] as usize;
                    let end = brain.synapse_pool.offsets[src_id as usize + 1] as usize;

                    for i in start..end {
                        let tgt_id = brain.synapse_pool.targets[i];
                        let plasticity = brain.synapse_pool.plasticity[i];
                        let tgt_act = window_active.get(&tgt_id).copied().unwrap_or(0.0);
                        
                        if tgt_act > 0.0 {
                            // Hebbian: both fire → strengthen
                            if plasticity >= 0.01 {
                                let delta = h_lr * src_act * tgt_act * plasticity;
                                if delta > 0.0001 {
                                    hebb_updates.push((src_id, tgt_id, delta));
                                }
                            }
                            // Coactive tracking
                            if window_set.contains(&tgt_id) {
                                coactive.push((src_id, tgt_id));
                            }
                        } else {
                            // Anti-Hebbian: src fires, tgt doesn't → weaken
                            if plasticity >= 0.01 {
                                let delta = -a_lr * src_act * plasticity;
                                if delta < -0.0001 {
                                    anti_updates.push((src_id, tgt_id, delta));
                                }
                            }
                        }
                    }
                    (hebb_updates, anti_updates, coactive)
                })
                .collect();

        // Sequential phase: queue all updates and collect coactive pairs
        let mut hebb_total = 0usize;
        let mut anti_total = 0usize;
        let mut all_coactive = Vec::new();
        
        for (hebb, anti, coact) in results {
            hebb_total += hebb.len();
            for (from, to, delta) in hebb {
                brain.synapse_pool.queue_update(from, to, delta);
            }
            anti_total += anti.len();
            for (from, to, delta) in anti {
                brain.synapse_pool.queue_update(from, to, delta);
            }
            all_coactive.extend(coact);
        }
        
        (hebb_total, anti_total, all_coactive)
    })
}

/// Set attention drives for all regions at once (batch version).
/// drives: HashMap<region_name, (novelty, threat, relevance)>
#[pyfunction]
fn batch_set_attention_drives(
    drives: HashMap<String, (f32, f32, f32)>,
) -> PyResult<()> {
    with_brain(|brain| {
        for (region_name, (novelty, threat, relevance)) in &drives {
            if let Some(region_id) = RegionId::from_name(region_name) {
                brain.set_attention_drives(region_id, *novelty, *threat, *relevance);
            }
        }
    })
}

/// Read all brain state needed for Python learning in one call.
/// Returns a dict with all activation levels, emotion, executive, motor, etc.
#[pyfunction]
fn batch_read_state() -> PyResult<HashMap<String, f64>> {
    with_brain_ref(|brain| {
        let mut state = HashMap::new();
        
        // Region activation levels
        state.insert("sensory_activation".to_string(), brain.sensory_activation() as f64);
        state.insert("visual_activation".to_string(), brain.visual_activation() as f64);
        state.insert("audio_activation".to_string(), brain.audio_activation() as f64);
        state.insert("motor_activation".to_string(), brain.motor_activation() as f64);
        state.insert("language_activation".to_string(), brain.language_activation() as f64);
        state.insert("speech_activity".to_string(), brain.speech_activity() as f64);
        
        // Emotion
        state.insert("emotion_polarity".to_string(), brain.emotion_polarity() as f64);
        state.insert("emotion_arousal".to_string(), brain.emotion_arousal() as f64);
        
        // Executive
        state.insert("executive_engagement".to_string(), brain.executive_engagement() as f64);
        state.insert("motor_conflict".to_string(), brain.motor_conflict() as f64);
        state.insert("planning_signal".to_string(), brain.planning_signal() as f64);
        
        // Motor decode
        match brain.decode_motor_action() {
            crate::regions::motor::MotorAction::Idle => {
                state.insert("motor_approach".to_string(), 0.0);
                state.insert("motor_withdraw".to_string(), 0.0);
            }
            crate::regions::motor::MotorAction::Approach { strength } => {
                state.insert("motor_approach".to_string(), strength as f64);
                state.insert("motor_withdraw".to_string(), 0.0);
            }
            crate::regions::motor::MotorAction::Withdraw { strength } => {
                state.insert("motor_approach".to_string(), 0.0);
                state.insert("motor_withdraw".to_string(), strength as f64);
            }
            crate::regions::motor::MotorAction::Conflict { approach, withdraw } => {
                state.insert("motor_approach".to_string(), approach as f64);
                state.insert("motor_withdraw".to_string(), withdraw as f64);
            }
        }
        
        // Inner monologue  
        state.insert("inner_monologue".to_string(), brain.inner_monologue_signal() as f64);
        
        // Pain
        state.insert("pain_level".to_string(), brain.detect_pain() as f64);
        
        // Integration
        state.insert("integration_input_count".to_string(), brain.integration_input_count(0.5) as f64);
        
        state
    })
}

fn with_brain<F, R>(f: F) -> PyResult<R>
where
    F: FnOnce(&mut Brain) -> R,
{
    let mut guard = BRAIN.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
    match guard.as_mut() {
        Some(brain) => Ok(f(brain)),
        None => Err(PyValueError::new_err("Brain not initialized. Call init_brain() first.")),
    }
}

fn with_brain_ref<F, R>(f: F) -> PyResult<R>
where
    F: FnOnce(&Brain) -> R,
{
    let guard = BRAIN.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
    match guard.as_ref() {
        Some(brain) => Ok(f(brain)),
        None => Err(PyValueError::new_err("Brain not initialized. Call init_brain() first.")),
    }
}

// === INITIALIZATION ===

#[pyfunction]
fn init_brain() -> PyResult<()> {
    let mut guard = BRAIN.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
    *guard = Some(Brain::new());
    Ok(())
}

#[pyfunction]
fn init_brain_with_synapses(synapses: Vec<(u32, u32, f32, u8, f32)>) -> PyResult<()> {
    let synapse_data: Vec<SynapseData> = synapses
        .into_iter()
        .map(|(from, to, weight, delay, plasticity)| SynapseData {
            from, to, weight, delay, plasticity,
        })
        .collect();

    let mut guard = BRAIN.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
    *guard = Some(Brain::with_synapses(synapse_data));
    Ok(())
}

#[pyfunction]
fn reset_brain() -> PyResult<()> {
    with_brain(|brain| brain.reset())
}

// === TICK CONTROL ===

#[pyfunction]
fn tick() -> PyResult<(u64, HashMap<String, u32>, u32)> {
    with_brain(|brain| {
        let result = brain.tick();
        let named_counts: HashMap<String, u32> = result
            .active_counts
            .into_iter()
            .map(|(id, count)| (id.name().to_string(), count))
            .collect();
        (result.tick_number, named_counts, result.total_active)
    })
}

#[pyfunction]
fn get_tick_count() -> PyResult<u64> {
    with_brain_ref(|brain| brain.tick_count)
}

// === INPUT ===

#[pyfunction]
fn inject_activations(signals: Vec<(u32, f32)>) -> PyResult<()> {
    with_brain(|brain| brain.inject(&signals))
}

#[pyfunction]
fn set_attention_gain(region_name: &str, gain: f32) -> PyResult<()> {
    let region_id = RegionId::from_name(region_name)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown region: {}", region_name)))?;
    with_brain(|brain| brain.set_attention_gain(region_id, gain))
}

// === READ STATE ===

#[pyfunction]
fn get_activations(region_name: &str, min_activation: f32) -> PyResult<Vec<(u32, f32)>> {
    let region_id = RegionId::from_name(region_name)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown region: {}", region_name)))?;
    with_brain_ref(|brain| brain.get_activations(region_id, min_activation))
}

#[pyfunction]
fn get_all_activations(min_activation: f32) -> PyResult<HashMap<String, Vec<(u32, f32)>>> {
    with_brain_ref(|brain| {
        brain
            .get_all_activations(min_activation)
            .into_iter()
            .map(|(id, acts)| (id.name().to_string(), acts))
            .collect()
    })
}

#[pyfunction]
fn get_neuron_potential(neuron_id: u32) -> PyResult<f32> {
    with_brain_ref(|brain| brain.get_neuron_potential(neuron_id).unwrap_or(0.0))
}

#[pyfunction]
fn get_active_count(region_name: &str) -> PyResult<u32> {
    let region_id = RegionId::from_name(region_name)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown region: {}", region_name)))?;
    with_brain_ref(|brain| {
        brain.region(region_id)
            .map(|r| r.active_global_ids(0.5).len() as u32)
            .unwrap_or(0)
    })
}

#[pyfunction]
fn get_region_firing_rate(region_name: &str) -> PyResult<f32> {
    let region_id = RegionId::from_name(region_name)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown region: {}", region_name)))?;
    with_brain_ref(|brain| brain.firing_rate(region_id))
}

// === SYNAPSE OPS ===

#[pyfunction]
fn get_synapse_weight(from: u32, to: u32) -> PyResult<Option<f32>> {
    with_brain_ref(|brain| brain.synapse_pool.get_weight(from, to))
}

#[pyfunction]
fn update_synapse(from: u32, to: u32, delta: f32) -> PyResult<()> {
    with_brain(|brain| brain.update_synapse(from, to, delta))
}

#[pyfunction]
fn create_synapse(from: u32, to: u32, weight: f32, delay: u8, plasticity: f32) -> PyResult<()> {
    with_brain(|brain| brain.create_synapse(from, to, weight, delay, plasticity))
}

#[pyfunction]
fn prune_synapse(from: u32, to: u32) -> PyResult<()> {
    with_brain(|brain| brain.prune_synapse(from, to))
}

#[pyfunction]
fn apply_synapse_updates() -> PyResult<()> {
    with_brain(|brain| brain.apply_synapse_updates())
}

#[pyfunction]
fn rebuild_synapse_index() -> PyResult<()> {
    with_brain(|brain| brain.rebuild_synapses())
}

#[pyfunction]
fn get_neuron_count() -> PyResult<u32> {
    with_brain_ref(|brain| brain.neuron_count())
}

#[pyfunction]
fn get_synapse_count() -> PyResult<u64> {
    with_brain_ref(|brain| brain.synapse_count())
}

/// Get all outgoing synapses for a given source neuron.
/// Returns list of (target_id, weight, delay, plasticity).
#[pyfunction]
fn get_outgoing_synapses(from: u32) -> PyResult<Vec<(u32, f32, u8, f32)>> {
    with_brain_ref(|brain| {
        let start = brain.synapse_pool.offsets[from as usize] as usize;
        let end = brain.synapse_pool.offsets[from as usize + 1] as usize;
        (start..end)
            .map(|i| (
                brain.synapse_pool.targets[i],
                brain.synapse_pool.weights[i],
                brain.synapse_pool.delays[i],
                brain.synapse_pool.plasticity[i],
            ))
            .collect()
    })
}

/// Batch update synapse weights. Takes list of (from, to, delta).
/// More efficient than calling update_synapse one at a time.
#[pyfunction]
fn batch_update_synapses(updates: Vec<(u32, u32, f32)>) -> PyResult<()> {
    with_brain(|brain| {
        for (from, to, delta) in updates {
            brain.synapse_pool.queue_update(from, to, delta);
        }
    })
}

/// Batch prune synapses. Takes list of (from, to) pairs.
#[pyfunction]
fn batch_prune_synapses(pairs: Vec<(u32, u32)>) -> PyResult<()> {
    with_brain(|brain| {
        for (from, to) in pairs {
            brain.synapse_pool.queue_prune(from, to);
        }
    })
}

/// Get activation value for a specific neuron.
#[pyfunction]
fn get_neuron_activation(neuron_id: u32) -> PyResult<f32> {
    with_brain_ref(|brain| {
        for region in &brain.regions {
            if let Some(local) = region.global_to_local(neuron_id) {
                return region.neurons.activations[local as usize];
            }
        }
        0.0
    })
}

// === PHASE 4: ATTENTION & PREDICTION ===

/// Set attention drives for a region: novelty, threat, relevance (all 0.0–1.0).
#[pyfunction]
fn set_attention_drives(region_name: &str, novelty: f32, threat: f32, relevance: f32) -> PyResult<()> {
    let region_id = RegionId::from_name(region_name)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown region: {}", region_name)))?;
    with_brain(|brain| brain.set_attention_drives(region_id, novelty, threat, relevance))
}

/// Get current attention gains for all regions.
#[pyfunction]
fn get_attention_gains() -> PyResult<HashMap<String, f32>> {
    with_brain_ref(|brain| {
        brain.get_attention_gains()
            .into_iter()
            .map(|(id, g)| (id.name().to_string(), g))
            .collect()
    })
}

/// Get per-region prediction errors from the last tick.
#[pyfunction]
fn get_prediction_errors() -> PyResult<HashMap<String, f32>> {
    with_brain_ref(|brain| {
        brain.get_prediction_errors()
            .into_iter()
            .map(|(id, e)| (id.name().to_string(), e))
            .collect()
    })
}

/// Get global prediction error (mean across all regions).
#[pyfunction]
fn get_global_prediction_error() -> PyResult<f32> {
    with_brain_ref(|brain| brain.get_global_prediction_error())
}

// === PHASE 5: BINDING & MEMORY ===

/// Create a binding between two patterns in different regions.
/// Returns the binding ID.
#[pyfunction]
fn create_binding(
    region_a: &str, neurons_a: Vec<u32>, threshold_a: f32,
    region_b: &str, neurons_b: Vec<u32>, threshold_b: f32,
    time_delta: f32,
) -> PyResult<u32> {
    let ra = RegionId::from_name(region_a)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown region: {}", region_a)))?;
    let rb = RegionId::from_name(region_b)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown region: {}", region_b)))?;
    with_brain(|brain| brain.create_binding(ra, neurons_a, threshold_a, rb, neurons_b, threshold_b, time_delta))
}

/// Evaluate which bindings are fully active.
/// Returns list of (binding_id, weight).
#[pyfunction]
fn evaluate_bindings(min_activation: f32) -> PyResult<Vec<(u32, f32)>> {
    with_brain_ref(|brain| brain.evaluate_bindings(min_activation))
}

/// Find bindings where only one pattern is active.
#[pyfunction]
fn find_partial_bindings(min_activation: f32) -> PyResult<Vec<u32>> {
    with_brain_ref(|brain| brain.find_partial_bindings(min_activation))
}

/// Strengthen a binding (co-activation detected).
#[pyfunction]
fn strengthen_binding(binding_id: u32, tick: u64) -> PyResult<()> {
    with_brain(|brain| brain.strengthen_binding(binding_id, tick))
}

/// Record a missed opportunity for a binding.
#[pyfunction]
fn record_binding_miss(binding_id: u32) -> PyResult<()> {
    with_brain(|brain| brain.record_binding_miss(binding_id))
}

/// Prune dissolved bindings. Returns count pruned.
#[pyfunction]
fn prune_bindings(weight_threshold: f32, min_fires: u32) -> PyResult<u32> {
    with_brain(|brain| brain.prune_bindings(weight_threshold, min_fires))
}

/// Get binding count.
#[pyfunction]
fn get_binding_count() -> PyResult<usize> {
    with_brain_ref(|brain| brain.binding_count())
}

/// Get binding info: (weight, fires, confidence, last_fired).
#[pyfunction]
fn get_binding_info(binding_id: u32) -> PyResult<Option<(f32, u32, f32, u64)>> {
    with_brain_ref(|brain| brain.get_binding_info(binding_id))
}

/// Pattern completion in memory_long. Returns number of neurons boosted.
#[pyfunction]
fn pattern_complete(trace_neurons: Vec<u32>, threshold: f32, boost: f32) -> PyResult<u32> {
    with_brain(|brain| brain.pattern_complete(&trace_neurons, threshold, boost))
}

/// Strengthen memory_long trace neurons (for consolidation). Returns count.
#[pyfunction]
fn strengthen_memory_trace(trace_neurons: Vec<u32>, boost: f32) -> PyResult<u32> {
    with_brain(|brain| brain.strengthen_memory_trace(&trace_neurons, boost))
}

/// Boost working memory neurons for active traces. Returns count.
#[pyfunction]
fn boost_working_memory(trace_neurons: Vec<u32>, boost: f32) -> PyResult<u32> {
    with_brain(|brain| brain.boost_working_memory(&trace_neurons, boost))
}

/// Count active input regions for integration.
#[pyfunction]
fn integration_input_count(min_activation: f32) -> PyResult<u32> {
    with_brain_ref(|brain| brain.integration_input_count(min_activation))
}

/// Boost integration region based on multi-modal convergence.
#[pyfunction]
fn boost_integration(strength: f32, max_neurons: u32) -> PyResult<u32> {
    with_brain(|brain| brain.boost_integration(strength, max_neurons as usize))
}

// === PHASE 6: EMOTION & EXECUTIVE ===

#[pyfunction]
fn set_neuromodulator(arousal: f32, valence: f32, focus: f32, energy: f32) -> PyResult<()> {
    with_brain(|brain| brain.set_neuromodulator(arousal, valence, focus, energy))
}

#[pyfunction]
fn get_neuromodulator() -> PyResult<(f32, f32, f32, f32)> {
    with_brain_ref(|brain| brain.get_neuromodulator())
}

#[pyfunction]
fn get_threshold_modifier() -> PyResult<f32> {
    with_brain_ref(|brain| brain.neuromod_threshold_modifier())
}

#[pyfunction]
fn get_emotion_polarity() -> PyResult<f32> {
    with_brain_ref(|brain| brain.emotion_polarity())
}

#[pyfunction]
fn get_emotion_arousal() -> PyResult<f32> {
    with_brain_ref(|brain| brain.emotion_arousal())
}

#[pyfunction]
fn get_emotion_urgency(urgency_threshold: f32) -> PyResult<f32> {
    with_brain_ref(|brain| brain.emotion_urgency(urgency_threshold))
}

#[pyfunction]
fn get_emotion_motor_impulse() -> PyResult<Vec<(u32, f32)>> {
    with_brain_ref(|brain| brain.emotion_motor_impulse())
}

#[pyfunction]
fn get_executive_engagement() -> PyResult<f32> {
    with_brain_ref(|brain| brain.executive_engagement())
}

#[pyfunction]
fn get_motor_conflict() -> PyResult<f32> {
    with_brain_ref(|brain| brain.motor_conflict())
}

#[pyfunction]
fn resolve_motor_conflict(suppress_strength: f32) -> PyResult<u32> {
    with_brain(|brain| brain.resolve_motor_conflict(suppress_strength))
}

#[pyfunction]
fn inhibit_motor(impulse_neurons: Vec<(u32, f32)>) -> PyResult<u32> {
    with_brain(|brain| brain.inhibit_motor(&impulse_neurons))
}

#[pyfunction]
fn get_planning_signal() -> PyResult<f32> {
    with_brain_ref(|brain| brain.planning_signal())
}

#[pyfunction]
fn recover_energy(amount: f32) -> PyResult<()> {
    with_brain(|brain| brain.recover_energy(amount))
}

// === PHASE 7: LANGUAGE & SPEECH ===

/// Get language region activation strength (0.0–1.0).
#[pyfunction]
fn get_language_activation() -> PyResult<f32> {
    with_brain_ref(|brain| brain.language_activation())
}

/// Compute symbol overlap between active language neurons and given trace neurons.
#[pyfunction]
fn get_symbol_overlap(trace_lang_neurons: Vec<u32>) -> PyResult<f32> {
    with_brain_ref(|brain| brain.symbol_overlap(&trace_lang_neurons))
}

/// Get inner monologue signal (language↔executive loop strength).
#[pyfunction]
fn get_inner_monologue_signal() -> PyResult<f32> {
    with_brain_ref(|brain| brain.inner_monologue_signal())
}

/// Boost language neurons for token activation. Returns count boosted.
#[pyfunction]
fn boost_language(neurons: Vec<u32>, boost: f32) -> PyResult<u32> {
    with_brain(|brain| brain.boost_language(&neurons, boost))
}

/// Get top-K active token neurons in language region.
#[pyfunction]
fn get_peak_language_neurons(top_k: usize) -> PyResult<Vec<(u32, f32)>> {
    with_brain_ref(|brain| brain.peak_language_neurons(top_k))
}

/// Get speech region activity level (0.0–1.0).
#[pyfunction]
fn get_speech_activity() -> PyResult<f32> {
    with_brain_ref(|brain| brain.speech_activity())
}

/// Get top-K active speech neurons for output decoding.
#[pyfunction]
fn get_peak_speech_neurons(top_k: usize) -> PyResult<Vec<(u32, f32)>> {
    with_brain_ref(|brain| brain.peak_speech_neurons(top_k))
}

/// Apply lateral inhibition in speech region for winner-take-all. Returns count suppressed.
#[pyfunction]
fn speech_lateral_inhibition(suppression_factor: f32) -> PyResult<u32> {
    with_brain(|brain| brain.speech_lateral_inhibition(suppression_factor))
}

/// Boost speech neurons for output generation. Returns count boosted.
#[pyfunction]
fn boost_speech(neurons: Vec<u32>, boost: f32) -> PyResult<u32> {
    with_brain(|brain| brain.boost_speech(&neurons, boost))
}

// === PHASE 8: SENSORY, VISUAL, AUDIO, MOTOR ===

/// Encode sensory values into population-coded neuron activations.
/// Returns list of (global_id, activation).
#[pyfunction]
fn encode_sensory(temperature: f32, pressure: f32, pain: f32, texture: f32, spread: u32) -> PyResult<Vec<(u32, f32)>> {
    with_brain_ref(|brain| brain.encode_sensory(temperature, pressure, pain, texture, spread))
}

/// Get sensory activation strength (0.0–1.0).
#[pyfunction]
fn get_sensory_activation() -> PyResult<f32> {
    with_brain_ref(|brain| brain.sensory_activation())
}

/// Boost sensory neurons. Returns count boosted.
#[pyfunction]
fn boost_sensory(neurons: Vec<u32>, boost: f32) -> PyResult<u32> {
    with_brain(|brain| brain.boost_sensory(&neurons, boost))
}

/// Get top-K active sensory neurons.
#[pyfunction]
fn get_peak_sensory_neurons(top_k: usize) -> PyResult<Vec<(u32, f32)>> {
    with_brain_ref(|brain| brain.peak_sensory_neurons(top_k))
}

/// Detect pain level from sensory activations (0.0–1.0).
#[pyfunction]
fn get_pain_level() -> PyResult<f32> {
    with_brain_ref(|brain| brain.detect_pain())
}

/// Get visual activation strength (0.0–1.0).
#[pyfunction]
fn get_visual_activation() -> PyResult<f32> {
    with_brain_ref(|brain| brain.visual_activation())
}

/// Boost visual neurons. Returns count boosted.
#[pyfunction]
fn boost_visual(neurons: Vec<u32>, boost: f32) -> PyResult<u32> {
    with_brain(|brain| brain.boost_visual(&neurons, boost))
}

/// Get top-K active visual neurons. Optional sub_region: "low", "mid", "high", "spatial".
#[pyfunction]
#[pyo3(signature = (top_k, sub_region=None))]
fn get_peak_visual_neurons(top_k: usize, sub_region: Option<&str>) -> PyResult<Vec<(u32, f32)>> {
    with_brain_ref(|brain| brain.peak_visual_neurons(top_k, sub_region))
}

/// Read all visual activations above threshold (for imagination output).
#[pyfunction]
fn read_visual_activations() -> PyResult<Vec<(u32, f32)>> {
    with_brain_ref(|brain| brain.read_visual_activations())
}

/// Get audio activation strength (0.0–1.0).
#[pyfunction]
fn get_audio_activation() -> PyResult<f32> {
    with_brain_ref(|brain| brain.audio_activation())
}

/// Boost audio neurons. Returns count boosted.
#[pyfunction]
fn boost_audio(neurons: Vec<u32>, boost: f32) -> PyResult<u32> {
    with_brain(|brain| brain.boost_audio(&neurons, boost))
}

/// Get top-K active audio neurons.
#[pyfunction]
fn get_peak_audio_neurons(top_k: usize) -> PyResult<Vec<(u32, f32)>> {
    with_brain_ref(|brain| brain.peak_audio_neurons(top_k))
}

/// Map a frequency (Hz) to audio neurons. Returns (global_id, activation).
#[pyfunction]
fn frequency_to_neurons(freq_hz: f32, spread: u32) -> PyResult<Vec<(u32, f32)>> {
    with_brain_ref(|brain| brain.frequency_to_neurons(freq_hz, spread))
}

/// Get motor activation strength (0.0–1.0).
#[pyfunction]
fn get_motor_activation() -> PyResult<f32> {
    with_brain_ref(|brain| brain.motor_activation())
}

/// Get approach vs withdraw strengths as (approach, withdraw).
#[pyfunction]
fn get_approach_vs_withdraw() -> PyResult<(f32, f32)> {
    with_brain_ref(|brain| brain.approach_vs_withdraw())
}

/// Decode motor action: returns (action_type, strength).
/// action_type: "idle", "approach", "withdraw", "conflict"
#[pyfunction]
fn decode_motor_action() -> PyResult<(String, f32, f32)> {
    with_brain_ref(|brain| {
        match brain.decode_motor_action() {
            crate::regions::motor::MotorAction::Idle => ("idle".to_string(), 0.0, 0.0),
            crate::regions::motor::MotorAction::Approach { strength } => ("approach".to_string(), strength, 0.0),
            crate::regions::motor::MotorAction::Withdraw { strength } => ("withdraw".to_string(), 0.0, strength),
            crate::regions::motor::MotorAction::Conflict { approach, withdraw } => ("conflict".to_string(), approach, withdraw),
        }
    })
}

/// Get top-K active motor neurons.
#[pyfunction]
fn get_peak_motor_neurons(top_k: usize) -> PyResult<Vec<(u32, f32)>> {
    with_brain_ref(|brain| brain.peak_motor_neurons(top_k))
}

/// Apply motor lateral inhibition. Returns count suppressed.
#[pyfunction]
fn motor_lateral_inhibition(suppression_factor: f32) -> PyResult<u32> {
    with_brain(|brain| brain.motor_lateral_inhibition(suppression_factor))
}

/// Boost motor neurons. Returns count boosted.
#[pyfunction]
fn boost_motor(neurons: Vec<u32>, boost: f32) -> PyResult<u32> {
    with_brain(|brain| brain.boost_motor(&neurons, boost))
}

// === Phase 9: Homeostasis & Sleep ===

/// Get homeostasis summary: (sleep_pressure, circadian_phase, ticks_awake, ticks_asleep).
#[pyfunction]
fn get_homeostasis_summary() -> PyResult<(f32, f32, u64, u64)> {
    with_brain_ref(|brain| brain.homeostasis_summary())
}

/// Get sleep state: (state_name, ticks_in_state, cycles_completed, rem_episodes).
#[pyfunction]
fn get_sleep_summary() -> PyResult<(String, u64, u32, u32)> {
    with_brain_ref(|brain| {
        let (name, ticks, cycles, rem) = brain.sleep_summary();
        (name.to_string(), ticks, cycles, rem)
    })
}

/// Is the brain asleep?
#[pyfunction]
fn is_asleep() -> PyResult<bool> {
    with_brain_ref(|brain| brain.is_asleep())
}

/// Is the brain in REM sleep?
#[pyfunction]
fn in_rem() -> PyResult<bool> {
    with_brain_ref(|brain| brain.in_rem())
}

/// Force wake-up (e.g. strong stimulus).
#[pyfunction]
fn force_wake() -> PyResult<()> {
    with_brain(|brain| brain.force_wake())
}

/// Get current sleep input gate factor.
#[pyfunction]
fn get_sleep_input_gate() -> PyResult<f32> {
    with_brain_ref(|brain| brain.sleep_input_gate())
}

/// Get sleep pressure.
#[pyfunction]
fn get_sleep_pressure() -> PyResult<f32> {
    with_brain_ref(|brain| brain.sleep_pressure())
}

/// Get circadian phase (0.0–1.0).
#[pyfunction]
fn get_circadian_phase() -> PyResult<f32> {
    with_brain_ref(|brain| brain.circadian_phase())
}

/// Set homeostasis regulation parameters.
#[pyfunction]
fn set_homeostasis_params(
    arousal_reg_rate: f32,
    valence_reg_rate: f32,
    focus_reg_rate: f32,
    sleep_pressure_rate: f32,
    sleep_dissipation_rate: f32,
) -> PyResult<()> {
    with_brain(|brain| {
        brain.set_homeostasis_params(
            arousal_reg_rate, valence_reg_rate, focus_reg_rate,
            sleep_pressure_rate, sleep_dissipation_rate,
        )
    })
}

/// Set sleep cycle phase durations (ticks).
#[pyfunction]
fn set_sleep_durations(drowsy: u64, light: u64, deep: u64, rem: u64) -> PyResult<()> {
    with_brain(|brain| brain.set_sleep_durations(drowsy, light, deep, rem))
}

// === MODULE ===

#[pymodule]
fn brain_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_brain, m)?)?;
    m.add_function(wrap_pyfunction!(init_brain_with_synapses, m)?)?;
    m.add_function(wrap_pyfunction!(reset_brain, m)?)?;

    m.add_function(wrap_pyfunction!(tick, m)?)?;
    m.add_function(wrap_pyfunction!(get_tick_count, m)?)?;

    m.add_function(wrap_pyfunction!(inject_activations, m)?)?;
    m.add_function(wrap_pyfunction!(set_attention_gain, m)?)?;

    m.add_function(wrap_pyfunction!(get_activations, m)?)?;
    m.add_function(wrap_pyfunction!(get_all_activations, m)?)?;
    m.add_function(wrap_pyfunction!(get_neuron_potential, m)?)?;
    m.add_function(wrap_pyfunction!(get_active_count, m)?)?;
    m.add_function(wrap_pyfunction!(get_region_firing_rate, m)?)?;

    m.add_function(wrap_pyfunction!(get_synapse_weight, m)?)?;
    m.add_function(wrap_pyfunction!(update_synapse, m)?)?;
    m.add_function(wrap_pyfunction!(create_synapse, m)?)?;
    m.add_function(wrap_pyfunction!(prune_synapse, m)?)?;
    m.add_function(wrap_pyfunction!(apply_synapse_updates, m)?)?;
    m.add_function(wrap_pyfunction!(rebuild_synapse_index, m)?)?;
    m.add_function(wrap_pyfunction!(get_neuron_count, m)?)?;
    m.add_function(wrap_pyfunction!(get_synapse_count, m)?)?;
    m.add_function(wrap_pyfunction!(get_outgoing_synapses, m)?)?;
    m.add_function(wrap_pyfunction!(batch_update_synapses, m)?)?;
    m.add_function(wrap_pyfunction!(batch_prune_synapses, m)?)?;
    m.add_function(wrap_pyfunction!(get_neuron_activation, m)?)?;

    // Phase 4: Attention & Prediction
    m.add_function(wrap_pyfunction!(set_attention_drives, m)?)?;
    m.add_function(wrap_pyfunction!(get_attention_gains, m)?)?;
    m.add_function(wrap_pyfunction!(get_prediction_errors, m)?)?;
    m.add_function(wrap_pyfunction!(get_global_prediction_error, m)?)?;

    // Phase 5: Binding & Memory
    m.add_function(wrap_pyfunction!(create_binding, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_bindings, m)?)?;
    m.add_function(wrap_pyfunction!(find_partial_bindings, m)?)?;
    m.add_function(wrap_pyfunction!(strengthen_binding, m)?)?;
    m.add_function(wrap_pyfunction!(record_binding_miss, m)?)?;
    m.add_function(wrap_pyfunction!(prune_bindings, m)?)?;
    m.add_function(wrap_pyfunction!(get_binding_count, m)?)?;
    m.add_function(wrap_pyfunction!(get_binding_info, m)?)?;
    m.add_function(wrap_pyfunction!(pattern_complete, m)?)?;
    m.add_function(wrap_pyfunction!(strengthen_memory_trace, m)?)?;
    m.add_function(wrap_pyfunction!(boost_working_memory, m)?)?;
    m.add_function(wrap_pyfunction!(integration_input_count, m)?)?;
    m.add_function(wrap_pyfunction!(boost_integration, m)?)?;

    // Phase 6: Emotion & Executive
    m.add_function(wrap_pyfunction!(set_neuromodulator, m)?)?;
    m.add_function(wrap_pyfunction!(get_neuromodulator, m)?)?;
    m.add_function(wrap_pyfunction!(get_threshold_modifier, m)?)?;
    m.add_function(wrap_pyfunction!(get_emotion_polarity, m)?)?;
    m.add_function(wrap_pyfunction!(get_emotion_arousal, m)?)?;
    m.add_function(wrap_pyfunction!(get_emotion_urgency, m)?)?;
    m.add_function(wrap_pyfunction!(get_emotion_motor_impulse, m)?)?;
    m.add_function(wrap_pyfunction!(get_executive_engagement, m)?)?;
    m.add_function(wrap_pyfunction!(get_motor_conflict, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_motor_conflict, m)?)?;
    m.add_function(wrap_pyfunction!(inhibit_motor, m)?)?;
    m.add_function(wrap_pyfunction!(get_planning_signal, m)?)?;
    m.add_function(wrap_pyfunction!(recover_energy, m)?)?;

    // Phase 7: Language & Speech
    m.add_function(wrap_pyfunction!(get_language_activation, m)?)?;
    m.add_function(wrap_pyfunction!(get_symbol_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(get_inner_monologue_signal, m)?)?;
    m.add_function(wrap_pyfunction!(boost_language, m)?)?;
    m.add_function(wrap_pyfunction!(get_peak_language_neurons, m)?)?;
    m.add_function(wrap_pyfunction!(get_speech_activity, m)?)?;
    m.add_function(wrap_pyfunction!(get_peak_speech_neurons, m)?)?;
    m.add_function(wrap_pyfunction!(speech_lateral_inhibition, m)?)?;
    m.add_function(wrap_pyfunction!(boost_speech, m)?)?;

    // Phase 8: Sensory, Visual, Audio, Motor
    m.add_function(wrap_pyfunction!(encode_sensory, m)?)?;
    m.add_function(wrap_pyfunction!(get_sensory_activation, m)?)?;
    m.add_function(wrap_pyfunction!(boost_sensory, m)?)?;
    m.add_function(wrap_pyfunction!(get_peak_sensory_neurons, m)?)?;
    m.add_function(wrap_pyfunction!(get_pain_level, m)?)?;
    m.add_function(wrap_pyfunction!(get_visual_activation, m)?)?;
    m.add_function(wrap_pyfunction!(boost_visual, m)?)?;
    m.add_function(wrap_pyfunction!(get_peak_visual_neurons, m)?)?;
    m.add_function(wrap_pyfunction!(read_visual_activations, m)?)?;
    m.add_function(wrap_pyfunction!(get_audio_activation, m)?)?;
    m.add_function(wrap_pyfunction!(boost_audio, m)?)?;
    m.add_function(wrap_pyfunction!(get_peak_audio_neurons, m)?)?;
    m.add_function(wrap_pyfunction!(frequency_to_neurons, m)?)?;
    m.add_function(wrap_pyfunction!(get_motor_activation, m)?)?;
    m.add_function(wrap_pyfunction!(get_approach_vs_withdraw, m)?)?;
    m.add_function(wrap_pyfunction!(decode_motor_action, m)?)?;
    m.add_function(wrap_pyfunction!(get_peak_motor_neurons, m)?)?;
    m.add_function(wrap_pyfunction!(motor_lateral_inhibition, m)?)?;
    m.add_function(wrap_pyfunction!(boost_motor, m)?)?;

    // Phase 9: Homeostasis & Sleep
    m.add_function(wrap_pyfunction!(get_homeostasis_summary, m)?)?;
    m.add_function(wrap_pyfunction!(get_sleep_summary, m)?)?;
    m.add_function(wrap_pyfunction!(is_asleep, m)?)?;
    m.add_function(wrap_pyfunction!(in_rem, m)?)?;
    m.add_function(wrap_pyfunction!(force_wake, m)?)?;
    m.add_function(wrap_pyfunction!(get_sleep_input_gate, m)?)?;
    m.add_function(wrap_pyfunction!(get_sleep_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(get_circadian_phase, m)?)?;
    m.add_function(wrap_pyfunction!(set_homeostasis_params, m)?)?;
    m.add_function(wrap_pyfunction!(set_sleep_durations, m)?)?;

    // Phase 10: Parallelism control
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;

    // Phase 10: Batch learning (Rust-side Hebbian/Anti-Hebbian)
    m.add_function(wrap_pyfunction!(batch_hebbian, m)?)?;
    m.add_function(wrap_pyfunction!(batch_anti_hebbian, m)?)?;
    m.add_function(wrap_pyfunction!(batch_track_coactive, m)?)?;

    // Phase 11: Combined learning step + batch state reading
    m.add_function(wrap_pyfunction!(batch_learn_step, m)?)?;
    m.add_function(wrap_pyfunction!(batch_set_attention_drives, m)?)?;
    m.add_function(wrap_pyfunction!(batch_read_state, m)?)?;

    Ok(())
}
