/// PyO3 bindings: expose the Rust brain core to Python.
///
/// Python calls these functions to:
///   - Create and manage the brain
///   - Inject signals, run ticks, read activations
///   - Manage synapses (create, update, prune, rebuild)
///   - Set attention gains

pub mod core;
pub mod regions;

use core::activity::ActivityCache;
use core::attention::AttentionSystem;
use core::brain::Brain;
use core::binding::{Binding, BindingStore, PatternRef};
use core::formation::{BindingTrackerRegistry, NovelPatternRegistry};
use core::homeostasis::HomeostasisSystem;
use core::neuromodulator::NeuromodulatorSystem;
use core::propagate::{DelayBuffer, SameRegionDelayAblation};
use core::region::{Region, RegionId};
use core::sleep::SleepCycleManager;
use core::synapse::{ApplyWeightUpdatesProfile, SynapseData, SynapseRuntimeState};
use core::tick::TickProfile;
use core::trace_match::TraceMatcherRegistry;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyIOError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use regions::pattern::PredictionState;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

// Global brain instance, protected by mutex.
// Python holds no Rust pointers — all access goes through these functions.
static BRAIN: Mutex<Option<Brain>> = Mutex::new(None);
static TRACE_MATCHERS: LazyLock<Mutex<TraceMatcherRegistry>> =
    LazyLock::new(|| Mutex::new(TraceMatcherRegistry::new()));
static NOVEL_TRACKERS: LazyLock<Mutex<NovelPatternRegistry>> =
    LazyLock::new(|| Mutex::new(NovelPatternRegistry::new()));
static BINDING_TRACKERS: LazyLock<Mutex<BindingTrackerRegistry>> =
    LazyLock::new(|| Mutex::new(BindingTrackerRegistry::new()));

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

const LEARNING_CHUNK_SIZE: usize = 256;

type NamedActivations = HashMap<String, Vec<(u32, f32)>>;
type NamedActiveCounts = HashMap<String, u32>;
type NamedState = HashMap<String, f64>;
type NamedProfile = HashMap<String, f64>;
type CompactState = Vec<f64>;
type BindingActivations = Vec<(u32, f64, f64)>;
type BindingRecallCandidates = Vec<(u32, f64, f64)>;
type NamedTraceScores = Vec<(String, f64)>;
type BindingSnapshot = (
    u32,
    (String, Vec<u32>, f64, Option<String>),
    (String, Vec<u32>, f64, Option<String>),
    f64,
    u32,
    f64,
    u64,
    f64,
    u32,
);

const COMPACT_STATE_LANGUAGE_ACTIVATION_IDX: usize = 4;
const COMPACT_STATE_EMOTION_POLARITY_IDX: usize = 6;

fn region_id_for_neuron(neuron_id: u32) -> Option<RegionId> {
    RegionId::from_neuron_id(neuron_id)
}

fn binding_snapshot(binding: &Binding) -> BindingSnapshot {
    (
        binding.id,
        (
            binding.pattern_a.region.name().to_string(),
            binding.pattern_a.neurons.clone(),
            binding.pattern_a.threshold as f64,
            binding.trace_id_a.clone(),
        ),
        (
            binding.pattern_b.region.name().to_string(),
            binding.pattern_b.neurons.clone(),
            binding.pattern_b.threshold as f64,
            binding.trace_id_b.clone(),
        ),
        binding.weight as f64,
        binding.fires,
        binding.time_delta as f64,
        binding.last_fired,
        binding.confidence as f64,
        binding.opportunities,
    )
}

fn binding_from_snapshot(binding: BindingSnapshot) -> PyResult<Binding> {
    let (
        id,
        (region_a, neurons_a, threshold_a, trace_id_a),
        (region_b, neurons_b, threshold_b, trace_id_b),
        weight,
        fires,
        time_delta,
        last_fired,
        confidence,
        opportunities,
    ) = binding;

    let region_a = RegionId::from_name(&region_a).ok_or_else(|| {
        PyValueError::new_err(format!("Unknown binding region: {}", region_a))
    })?;
    let region_b = RegionId::from_name(&region_b).ok_or_else(|| {
        PyValueError::new_err(format!("Unknown binding region: {}", region_b))
    })?;

    Ok(Binding {
        id,
        pattern_a: PatternRef::new(region_a, neurons_a, threshold_a as f32),
        pattern_b: PatternRef::new(region_b, neurons_b, threshold_b as f32),
        trace_id_a,
        trace_id_b,
        weight: weight as f32,
        fires,
        time_delta: time_delta as f32,
        last_fired,
        confidence: confidence as f32,
        opportunities,
    })
}

#[derive(Clone, Serialize, Deserialize)]
struct BrainRuntimeCheckpoint {
    synapse_topology_signature: u64,
    synapse_runtime_state: SynapseRuntimeState,
    regions: Vec<Region>,
    delay_buffer: DelayBuffer,
    attention_system: AttentionSystem,
    prediction_state: PredictionState,
    binding_store: BindingStore,
    neuromodulator: NeuromodulatorSystem,
    homeostasis: HomeostasisSystem,
    sleep_cycle: SleepCycleManager,
    activity_cache: ActivityCache,
    same_region_delay_ablation: SameRegionDelayAblation,
    same_region_delay_learning_ablation: SameRegionDelayAblation,
    tick_count: u64,
}

impl BrainRuntimeCheckpoint {
    fn from_brain(brain: &Brain) -> Self {
        Self {
            synapse_topology_signature: synapse_topology_signature_for(brain),
            synapse_runtime_state: brain.synapse_pool.export_runtime_state(),
            regions: brain.regions.clone(),
            delay_buffer: brain.delay_buffer.clone(),
            attention_system: brain.attention_system.clone(),
            prediction_state: brain.prediction_state.clone(),
            binding_store: brain.binding_store.clone(),
            neuromodulator: brain.neuromodulator.clone(),
            homeostasis: brain.homeostasis.clone(),
            sleep_cycle: brain.sleep_cycle.clone(),
            activity_cache: brain.activity_cache.clone(),
            same_region_delay_ablation: brain.same_region_delay_ablation.clone(),
            same_region_delay_learning_ablation: brain
                .same_region_delay_learning_ablation
                .clone(),
            tick_count: brain.tick_count,
        }
    }

    fn apply_to(&self, brain: &mut Brain) -> PyResult<()> {
        let current_signature = synapse_topology_signature_for(brain);
        if self.synapse_topology_signature != current_signature {
            return Err(PyValueError::new_err(format!(
                "Runtime checkpoint topology mismatch: expected {}, found {}",
                self.synapse_topology_signature, current_signature,
            )));
        }

        brain
            .synapse_pool
            .apply_runtime_state(self.synapse_runtime_state.clone());
        brain.regions = self.regions.clone();
        brain.delay_buffer = self.delay_buffer.clone();
        brain.attention_system = self.attention_system.clone();
        brain.prediction_state = self.prediction_state.clone();
        brain.binding_store = self.binding_store.clone();
        brain.neuromodulator = self.neuromodulator.clone();
        brain.homeostasis = self.homeostasis.clone();
        brain.sleep_cycle = self.sleep_cycle.clone();
        brain.activity_cache = self.activity_cache.clone();
        brain.same_region_delay_ablation = self.same_region_delay_ablation.clone();
        brain.same_region_delay_learning_ablation =
            self.same_region_delay_learning_ablation.clone();
        brain.tick_count = self.tick_count;
        Ok(())
    }
}

fn synapse_topology_signature_for(brain: &Brain) -> u64 {
    let mut hasher = DefaultHasher::new();
    brain.synapse_pool.num_neurons().hash(&mut hasher);
    brain.synapse_pool.chunk_count().hash(&mut hasher);
    brain.synapse_pool.chunk_size().hash(&mut hasher);
    brain.synapse_pool.offsets.hash(&mut hasher);
    brain.synapse_pool.targets.hash(&mut hasher);
    brain.synapse_pool.delays.hash(&mut hasher);
    for plasticity in &brain.synapse_pool.plasticity {
        plasticity.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn read_brain_checkpoint(path: &str) -> PyResult<Brain> {
    let bytes = fs::read(path)
        .map_err(|e| PyIOError::new_err(format!("Failed to read brain checkpoint: {}", e)))?;
    bincode::deserialize(&bytes).map_err(|e| {
        PyValueError::new_err(format!("Failed to deserialize brain checkpoint: {}", e))
    })
}

fn write_brain_checkpoint(path: &str, brain: &Brain) -> PyResult<()> {
    let bytes = bincode::serialize(brain).map_err(|e| {
        PyValueError::new_err(format!("Failed to serialize brain checkpoint: {}", e))
    })?;

    let path_ref = Path::new(path);
    if let Some(parent) = path_ref.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                PyIOError::new_err(format!("Failed to create checkpoint directory: {}", e))
            })?;
        }
    }

    fs::write(path_ref, bytes)
        .map_err(|e| PyIOError::new_err(format!("Failed to write brain checkpoint: {}", e)))?;
    Ok(())
}

fn serialize_brain_checkpoint(brain: &Brain) -> PyResult<Vec<u8>> {
    bincode::serialize(brain).map_err(|e| {
        PyValueError::new_err(format!("Failed to serialize brain checkpoint: {}", e))
    })
}

fn deserialize_brain_checkpoint(bytes: &[u8]) -> PyResult<Brain> {
    bincode::deserialize(bytes).map_err(|e| {
        PyValueError::new_err(format!("Failed to deserialize brain checkpoint: {}", e))
    })
}

fn read_brain_runtime_checkpoint(path: &str) -> PyResult<BrainRuntimeCheckpoint> {
    let bytes = fs::read(path).map_err(|e| {
        PyIOError::new_err(format!("Failed to read brain runtime checkpoint: {}", e))
    })?;
    bincode::deserialize(&bytes).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to deserialize brain runtime checkpoint: {}",
            e
        ))
    })
}

fn write_brain_runtime_checkpoint(path: &str, brain: &Brain) -> PyResult<()> {
    let checkpoint = BrainRuntimeCheckpoint::from_brain(brain);
    let bytes = bincode::serialize(&checkpoint).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to serialize brain runtime checkpoint: {}",
            e
        ))
    })?;

    let path_ref = Path::new(path);
    if let Some(parent) = path_ref.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                PyIOError::new_err(format!(
                    "Failed to create runtime checkpoint directory: {}",
                    e
                ))
            })?;
        }
    }

    fs::write(path_ref, bytes).map_err(|e| {
        PyIOError::new_err(format!("Failed to write brain runtime checkpoint: {}", e))
    })?;
    Ok(())
}

fn serialize_brain_runtime_checkpoint(brain: &Brain) -> PyResult<Vec<u8>> {
    let checkpoint = BrainRuntimeCheckpoint::from_brain(brain);
    bincode::serialize(&checkpoint).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to serialize brain runtime checkpoint: {}",
            e
        ))
    })
}

fn deserialize_brain_runtime_checkpoint(bytes: &[u8]) -> PyResult<BrainRuntimeCheckpoint> {
    bincode::deserialize(bytes).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to deserialize brain runtime checkpoint: {}",
            e
        ))
    })
}

fn synapse_topology_matches(reference: &Brain, candidate: &Brain) -> bool {
    reference.synapse_pool.offsets == candidate.synapse_pool.offsets
        && reference.synapse_pool.targets == candidate.synapse_pool.targets
        && reference.synapse_pool.delays == candidate.synapse_pool.delays
        && reference.synapse_pool.plasticity == candidate.synapse_pool.plasticity
}

fn pattern_ref_key(pattern: &PatternRef) -> String {
    let neuron_key = pattern
        .neurons
        .iter()
        .map(|neuron_id| neuron_id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{}:{:08x}:{}",
        pattern.region.name(),
        pattern.threshold.to_bits(),
        neuron_key,
    )
}

fn binding_pair_key(binding: &Binding) -> String {
    let left = pattern_ref_key(&binding.pattern_a);
    let right = pattern_ref_key(&binding.pattern_b);
    if left <= right {
        format!("{}|{}", left, right)
    } else {
        format!("{}|{}", right, left)
    }
}

#[inline]
fn same_region_delay_mask_suppresses_learning(
    ablation: &SameRegionDelayAblation,
    source_region_idx: Option<usize>,
    target_id: u32,
    raw_delay: u8,
) -> bool {
    if !ablation.is_enabled() {
        return false;
    }

    let Some(source_region_idx) = source_region_idx else {
        return false;
    };
    let Some(target_region_id) = region_id_for_neuron(target_id) else {
        return false;
    };

    ablation.suppresses(source_region_idx, target_region_id.index(), raw_delay)
}

fn build_apply_weight_updates_profile_map(profile: &ApplyWeightUpdatesProfile) -> NamedProfile {
    let mut data = HashMap::new();
    data.insert("pending_update_count".to_string(), profile.pending_update_count as f64);
    data.insert("deferred_update_count".to_string(), profile.deferred_update_count as f64);
    data.insert("applied_update_count".to_string(), profile.applied_update_count as f64);
    data.insert("unmatched_update_count".to_string(), profile.unmatched_update_count as f64);
    data.insert("positive_update_count".to_string(), profile.positive_update_count as f64);
    data.insert("negative_update_count".to_string(), profile.negative_update_count as f64);
    data.insert("delta_sum".to_string(), profile.delta_sum);
    data.insert("delta_abs_sum".to_string(), profile.delta_abs_sum);
    data.insert("delta_min".to_string(), profile.delta_min as f64);
    data.insert("delta_max".to_string(), profile.delta_max as f64);
    data.insert(
        "before_weight_avg".to_string(),
        if profile.applied_update_count > 0 {
            profile.before_weight_sum / profile.applied_update_count as f64
        } else {
            0.0
        },
    );
    data.insert(
        "after_weight_avg".to_string(),
        if profile.applied_update_count > 0 {
            profile.after_weight_sum / profile.applied_update_count as f64
        } else {
            0.0
        },
    );
    data.insert("crossed_up_0p05_count".to_string(), profile.crossed_up_0p05_count as f64);
    data.insert("crossed_up_0p10_count".to_string(), profile.crossed_up_0p10_count as f64);
    data.insert("crossed_up_0p20_count".to_string(), profile.crossed_up_0p20_count as f64);
    data.insert("crossed_down_0p05_count".to_string(), profile.crossed_down_0p05_count as f64);
    data.insert("crossed_down_0p10_count".to_string(), profile.crossed_down_0p10_count as f64);
    data.insert("crossed_down_0p20_count".to_string(), profile.crossed_down_0p20_count as f64);

    for &source_region in RegionId::ALL.iter() {
        let source_idx = source_region.index();
        let source_name = source_region.name();
        for &target_region in RegionId::ALL.iter() {
            let target_idx = target_region.index();
            let target_name = target_region.name();
            let count = profile.region_pair_update_counts[source_idx][target_idx];
            if count == 0 {
                continue;
            }

            let key_prefix = format!("region_pair_{}_to_{}", source_name, target_name);
            data.insert(format!("{}_count", key_prefix), count as f64);
            data.insert(
                format!("{}_delta_abs_sum", key_prefix),
                profile.region_pair_delta_abs_sums[source_idx][target_idx],
            );
            data.insert(
                format!("{}_before_weight_avg", key_prefix),
                profile.region_pair_before_weight_sums[source_idx][target_idx] / count as f64,
            );
            data.insert(
                format!("{}_after_weight_avg", key_prefix),
                profile.region_pair_after_weight_sums[source_idx][target_idx] / count as f64,
            );
        }
    }

    for delay in 0..profile.delay_update_counts.len() {
        let count = profile.delay_update_counts[delay];
        if count == 0 {
            continue;
        }
        data.insert(format!("delay_{}_count", delay), count as f64);
        data.insert(
            format!("delay_{}_delta_abs_sum", delay),
            profile.delay_delta_abs_sums[delay],
        );
    }

    data
}

fn region_counts_for_neurons(neurons: &[u32]) -> [u32; RegionId::ALL.len()] {
    let mut counts = [0u32; RegionId::ALL.len()];
    for &neuron_id in neurons {
        if let Some(region_id) = region_id_for_neuron(neuron_id) {
            counts[region_id.index()] = counts[region_id.index()].saturating_add(1);
        }
    }
    counts
}

fn unique_neurons(neurons: &[u32]) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for &neuron_id in neurons {
        if seen.insert(neuron_id) {
            unique.push(neuron_id);
        }
    }
    unique
}

fn insert_region_profile_counts(profile: &mut NamedProfile, prefix: &str, counts: &[u32; RegionId::ALL.len()]) {
    for (idx, region_id) in RegionId::ALL.iter().enumerate() {
        profile.insert(
            format!("{}_region_{}_neurons", prefix, region_id.name()),
            counts[idx] as f64,
        );
    }
}

fn parse_region_ids(region_names: &[String]) -> PyResult<Vec<RegionId>> {
    region_names
        .iter()
        .map(|region_name| {
            RegionId::from_name(region_name).ok_or_else(|| {
                PyValueError::new_err(format!("Unknown region name: {}", region_name))
            })
        })
        .collect()
}

fn normalize_delays(delays: &[u8]) -> PyResult<Vec<u8>> {
    let mut normalized = Vec::new();
    let mut seen = HashSet::new();
    for &delay in delays {
        if delay == 0 || delay as usize > core::propagate::MAX_DELAY {
            return Err(PyValueError::new_err(format!(
                "Delay must be between 1 and {} inclusive, got {}",
                core::propagate::MAX_DELAY,
                delay,
            )));
        }
        if seen.insert(delay) {
            normalized.push(delay);
        }
    }
    normalized.sort_unstable();
    Ok(normalized)
}

fn dense_window_activity(window_active: &HashMap<u32, f32>, num_neurons: u32) -> Vec<f32> {
    let mut dense = vec![0.0f32; num_neurons as usize];
    for (&neuron_id, &activation) in window_active {
        if neuron_id < num_neurons {
            dense[neuron_id as usize] = activation;
        }
    }
    dense
}

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
        let window_activity = dense_window_activity(&window_active, nn);
        let synapse_pool = &brain.synapse_pool;
        let shared_weights_attached = synapse_pool.shared_weights_attached();
        let learning_ablation = &brain.same_region_delay_learning_ablation;
        let learning_ablation_enabled = learning_ablation.is_enabled();

        if shared_weights_attached {
            return active_neurons
                .par_chunks(LEARNING_CHUNK_SIZE)
                .map(|chunk| {
                    let mut update_count = 0usize;
                    for &(src_id, src_act) in chunk {
                        if src_id >= nn {
                            continue;
                        }

                        let source_region_idx = if learning_ablation_enabled {
                            region_id_for_neuron(src_id).map(|region_id| region_id.index())
                        } else {
                            None
                        };

                        let start = synapse_pool.offsets[src_id as usize] as usize;
                        let end = synapse_pool.offsets[src_id as usize + 1] as usize;

                        for i in start..end {
                            let plasticity = synapse_pool.plasticity[i];
                            if plasticity < 0.01 {
                                continue;
                            }
                            let tgt_id = synapse_pool.targets[i];
                            let raw_delay = synapse_pool.delays[i];
                            if same_region_delay_mask_suppresses_learning(
                                learning_ablation,
                                source_region_idx,
                                tgt_id,
                                raw_delay,
                            ) {
                                continue;
                            }
                            let tgt_act = window_activity[tgt_id as usize];
                            if tgt_act > 0.0 {
                                let delta = lr * src_act * tgt_act * plasticity;
                                if delta > 0.0001 {
                                    synapse_pool.atomic_add_weight_at_index(i, delta);
                                    update_count += 1;
                                }
                            }
                        }
                    }
                    update_count
                })
                .sum();
        }

        // Parallel phase: each active neuron independently collects updates
        let all_updates: Vec<Vec<(u32, u32, u32, f32)>> = active_neurons
            .par_chunks(LEARNING_CHUNK_SIZE)
            .map(|chunk| {
                let mut updates = Vec::new();
                for &(src_id, src_act) in chunk {
                    if src_id >= nn {
                        continue;
                    }

                    let source_region_idx = if learning_ablation_enabled {
                        region_id_for_neuron(src_id).map(|region_id| region_id.index())
                    } else {
                        None
                    };

                    let start = brain.synapse_pool.offsets[src_id as usize] as usize;
                    let end = brain.synapse_pool.offsets[src_id as usize + 1] as usize;

                    for i in start..end {
                        let plasticity = brain.synapse_pool.plasticity[i];
                        if plasticity < 0.01 {
                            continue;
                        }
                        let tgt_id = brain.synapse_pool.targets[i];
                        let raw_delay = brain.synapse_pool.delays[i];
                        if same_region_delay_mask_suppresses_learning(
                            learning_ablation,
                            source_region_idx,
                            tgt_id,
                            raw_delay,
                        ) {
                            continue;
                        }
                        let tgt_act = window_activity[tgt_id as usize];
                        if tgt_act > 0.0 {
                            let delta = lr * src_act * tgt_act * plasticity;
                            if delta > 0.0001 {
                                updates.push((src_id, tgt_id, i as u32, delta));
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
            for (from, to, synapse_index, delta) in updates {
                brain
                    .synapse_pool
                    .queue_indexed_update(from, to, synapse_index, delta);
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
        let window_activity = dense_window_activity(&window_active, nn);
        let synapse_pool = &brain.synapse_pool;
        let shared_weights_attached = synapse_pool.shared_weights_attached();
        let learning_ablation = &brain.same_region_delay_learning_ablation;
        let learning_ablation_enabled = learning_ablation.is_enabled();

        if shared_weights_attached {
            return active_neurons
                .par_chunks(LEARNING_CHUNK_SIZE)
                .map(|chunk| {
                    let mut update_count = 0usize;
                    for &(src_id, src_act) in chunk {
                        if src_id >= nn {
                            continue;
                        }

                        let source_region_idx = if learning_ablation_enabled {
                            region_id_for_neuron(src_id).map(|region_id| region_id.index())
                        } else {
                            None
                        };

                        let start = synapse_pool.offsets[src_id as usize] as usize;
                        let end = synapse_pool.offsets[src_id as usize + 1] as usize;

                        for i in start..end {
                            let plasticity = synapse_pool.plasticity[i];
                            if plasticity < 0.01 {
                                continue;
                            }
                            let tgt_id = synapse_pool.targets[i];
                            let raw_delay = synapse_pool.delays[i];
                            if same_region_delay_mask_suppresses_learning(
                                learning_ablation,
                                source_region_idx,
                                tgt_id,
                                raw_delay,
                            ) {
                                continue;
                            }
                            let tgt_act = window_activity[tgt_id as usize];
                            if tgt_act > 0.0 {
                                continue;
                            }
                            let delta = -r * src_act * (1.0 - tgt_act) * plasticity;
                            if delta < -0.0001 {
                                synapse_pool.atomic_add_weight_at_index(i, delta);
                                update_count += 1;
                            }
                        }
                    }
                    update_count
                })
                .sum();
        }

        let all_updates: Vec<Vec<(u32, u32, u32, f32)>> = active_neurons
            .par_chunks(LEARNING_CHUNK_SIZE)
            .map(|chunk| {
                let mut updates = Vec::new();
                for &(src_id, src_act) in chunk {
                    if src_id >= nn {
                        continue;
                    }

                    let source_region_idx = if learning_ablation_enabled {
                        region_id_for_neuron(src_id).map(|region_id| region_id.index())
                    } else {
                        None
                    };

                    let start = brain.synapse_pool.offsets[src_id as usize] as usize;
                    let end = brain.synapse_pool.offsets[src_id as usize + 1] as usize;

                    for i in start..end {
                        let plasticity = brain.synapse_pool.plasticity[i];
                        if plasticity < 0.01 {
                            continue;
                        }
                        let tgt_id = brain.synapse_pool.targets[i];
                        let raw_delay = brain.synapse_pool.delays[i];
                        if same_region_delay_mask_suppresses_learning(
                            learning_ablation,
                            source_region_idx,
                            tgt_id,
                            raw_delay,
                        ) {
                            continue;
                        }
                        let tgt_act = window_activity[tgt_id as usize];
                        if tgt_act > 0.0 {
                            continue; // Target active — Hebbian handles this
                        }
                        let delta = -r * src_act * (1.0 - tgt_act) * plasticity;
                        if delta < -0.0001 {
                            updates.push((src_id, tgt_id, i as u32, delta));
                        }
                    }
                }
                updates
            })
            .collect();

        let mut total = 0usize;
        for updates in all_updates {
            total += updates.len();
            for (from, to, synapse_index, delta) in updates {
                brain
                    .synapse_pool
                    .queue_indexed_update(from, to, synapse_index, delta);
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
    batch_learn_step_inner(
        active_neurons,
        window_active,
        hebbian_rate,
        anti_hebbian_rate,
        true,
    )
}

fn batch_learn_step_inner(
    active_neurons: Vec<(u32, f32)>,
    window_active: HashMap<u32, f32>,
    hebbian_rate: f64,
    anti_hebbian_rate: f64,
    track_coactive: bool,
) -> PyResult<(usize, usize, Vec<(u32, u32)>)> {
    let h_lr = hebbian_rate as f32;
    let a_lr = anti_hebbian_rate as f32;
    with_brain(|brain| {
        let nn = brain.synapse_pool.num_neurons();
        let window_activity = dense_window_activity(&window_active, nn);
        let synapse_pool = &brain.synapse_pool;
        let shared_weights_attached = synapse_pool.shared_weights_attached();
        let learning_ablation = &brain.same_region_delay_learning_ablation;
        let learning_ablation_enabled = learning_ablation.is_enabled();

        if shared_weights_attached {
            let results: Vec<(usize, usize, Vec<(u32, u32)>)> = active_neurons
                .par_chunks(LEARNING_CHUNK_SIZE)
                .map(|chunk| {
                    let mut hebb_total = 0usize;
                    let mut anti_total = 0usize;
                    let mut coactive = if track_coactive {
                        Vec::new()
                    } else {
                        Vec::with_capacity(0)
                    };

                    for &(src_id, src_act) in chunk {
                        if src_id >= nn {
                            continue;
                        }

                        let source_region_idx = if learning_ablation_enabled {
                            region_id_for_neuron(src_id).map(|region_id| region_id.index())
                        } else {
                            None
                        };

                        let start = synapse_pool.offsets[src_id as usize] as usize;
                        let end = synapse_pool.offsets[src_id as usize + 1] as usize;

                        for i in start..end {
                            let tgt_id = synapse_pool.targets[i];
                            let raw_delay = synapse_pool.delays[i];
                            if same_region_delay_mask_suppresses_learning(
                                learning_ablation,
                                source_region_idx,
                                tgt_id,
                                raw_delay,
                            ) {
                                continue;
                            }
                            let plasticity = synapse_pool.plasticity[i];
                            let tgt_act = window_activity[tgt_id as usize];

                            if tgt_act > 0.0 {
                                if plasticity >= 0.01 {
                                    let delta = h_lr * src_act * tgt_act * plasticity;
                                    if delta > 0.0001 {
                                        synapse_pool.atomic_add_weight_at_index(i, delta);
                                        hebb_total += 1;
                                    }
                                }
                                if track_coactive {
                                    coactive.push((src_id, tgt_id));
                                }
                            } else if plasticity >= 0.01 {
                                let delta = -a_lr * src_act * plasticity;
                                if delta < -0.0001 {
                                    synapse_pool.atomic_add_weight_at_index(i, delta);
                                    anti_total += 1;
                                }
                            }
                        }
                    }

                    (hebb_total, anti_total, coactive)
                })
                .collect();

            let mut hebb_total = 0usize;
            let mut anti_total = 0usize;
            let mut all_coactive = if track_coactive {
                Vec::new()
            } else {
                Vec::with_capacity(0)
            };

            for (chunk_hebb, chunk_anti, chunk_coactive) in results {
                hebb_total += chunk_hebb;
                anti_total += chunk_anti;
                if track_coactive {
                    all_coactive.extend(chunk_coactive);
                }
            }

            return (hebb_total, anti_total, all_coactive);
        }

        // Parallel phase: each active neuron does hebbian + anti-hebbian + coactive
        let results: Vec<(
            Vec<(u32, u32, u32, f32)>,
            Vec<(u32, u32, u32, f32)>,
            Vec<(u32, u32)>,
        )> =
            active_neurons
                .par_chunks(LEARNING_CHUNK_SIZE)
                .map(|chunk| {
                    let mut hebb_updates = Vec::new();
                    let mut anti_updates = Vec::new();
                    let mut coactive = Vec::new();

                    for &(src_id, src_act) in chunk {
                        if src_id >= nn {
                            continue;
                        }

                        let source_region_idx = if learning_ablation_enabled {
                            region_id_for_neuron(src_id).map(|region_id| region_id.index())
                        } else {
                            None
                        };

                        let start = brain.synapse_pool.offsets[src_id as usize] as usize;
                        let end = brain.synapse_pool.offsets[src_id as usize + 1] as usize;

                        for i in start..end {
                            let tgt_id = brain.synapse_pool.targets[i];
                            let raw_delay = brain.synapse_pool.delays[i];
                            if same_region_delay_mask_suppresses_learning(
                                learning_ablation,
                                source_region_idx,
                                tgt_id,
                                raw_delay,
                            ) {
                                continue;
                            }
                            let plasticity = brain.synapse_pool.plasticity[i];
                            let tgt_act = window_activity[tgt_id as usize];

                            if tgt_act > 0.0 {
                                // Hebbian: both fire → strengthen
                                if plasticity >= 0.01 {
                                    let delta = h_lr * src_act * tgt_act * plasticity;
                                    if delta > 0.0001 {
                                        hebb_updates.push((src_id, tgt_id, i as u32, delta));
                                    }
                                }
                                if track_coactive {
                                    coactive.push((src_id, tgt_id));
                                }
                            } else {
                                // Anti-Hebbian: src fires, tgt doesn't → weaken
                                if plasticity >= 0.01 {
                                    let delta = -a_lr * src_act * plasticity;
                                    if delta < -0.0001 {
                                        anti_updates.push((src_id, tgt_id, i as u32, delta));
                                    }
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
        let mut all_coactive = if track_coactive { Vec::new() } else { Vec::with_capacity(0) };

        for (hebb, anti, coact) in results {
            hebb_total += hebb.len();
            for (from, to, synapse_index, delta) in hebb {
                brain
                    .synapse_pool
                    .queue_indexed_update(from, to, synapse_index, delta);
            }
            anti_total += anti.len();
            for (from, to, synapse_index, delta) in anti {
                brain
                    .synapse_pool
                    .queue_indexed_update(from, to, synapse_index, delta);
            }
            if track_coactive {
                all_coactive.extend(coact);
            }
        }

        (hebb_total, anti_total, all_coactive)
    })
}

/// Combined learning step with optional coactive synapse export.
/// When `track_coactive` is false, Hebbian/anti-Hebbian still run but Python
/// avoids paying to receive and process every co-active edge on that tick.
#[pyfunction]
fn batch_learn_step_configurable(
    active_neurons: Vec<(u32, f32)>,
    window_active: HashMap<u32, f32>,
    hebbian_rate: f64,
    anti_hebbian_rate: f64,
    track_coactive: bool,
) -> PyResult<(usize, usize, Vec<(u32, u32)>)> {
    batch_learn_step_inner(
        active_neurons,
        window_active,
        hebbian_rate,
        anti_hebbian_rate,
        track_coactive,
    )
}

/// Like batch_learn_step_configurable but accepts flat parallel arrays for
/// active neurons (avoids Python→Rust tuple conversion overhead).
/// window_ids/window_vals are parallel arrays for the window-active neurons.
#[pyfunction]
fn batch_learn_step_flat(
    active_ids: Vec<u32>,
    active_vals: Vec<f32>,
    window_ids: Vec<u32>,
    window_vals: Vec<f32>,
    hebbian_rate: f64,
    anti_hebbian_rate: f64,
    track_coactive: bool,
) -> PyResult<(usize, usize, Vec<(u32, u32)>)> {
    let active_neurons: Vec<(u32, f32)> = active_ids
        .into_iter()
        .zip(active_vals)
        .collect();
    let window_active: HashMap<u32, f32> = window_ids
        .into_iter()
        .zip(window_vals)
        .collect();
    batch_learn_step_inner(
        active_neurons,
        window_active,
        hebbian_rate,
        anti_hebbian_rate,
        track_coactive,
    )
}

/// Accepts concatenated raw window snapshot arrays (all snapshots' flat_ids/vals
/// concatenated together). Merges them in Rust with max-aggregation, avoiding
/// the expensive Python-side dict-building loop.
#[pyfunction]
fn batch_learn_step_from_snapshots(
    active_ids: Vec<u32>,
    active_vals: Vec<f32>,
    raw_window_ids: Vec<u32>,
    raw_window_vals: Vec<f32>,
    hebbian_rate: f64,
    anti_hebbian_rate: f64,
    track_coactive: bool,
) -> PyResult<(usize, usize, Vec<(u32, u32)>)> {
    let active_neurons: Vec<(u32, f32)> = active_ids
        .into_iter()
        .zip(active_vals)
        .collect();
    // Merge raw window snapshots with max-aggregation
    let mut window_active: HashMap<u32, f32> = HashMap::with_capacity(raw_window_ids.len());
    for (id, val) in raw_window_ids.into_iter().zip(raw_window_vals) {
        let entry = window_active.entry(id).or_insert(0.0f32);
        if val > *entry {
            *entry = val;
        }
    }
    batch_learn_step_inner(
        active_neurons,
        window_active,
        hebbian_rate,
        anti_hebbian_rate,
        track_coactive,
    )
}

/// Push activation snapshot into the Rust-side cache (called after each eval).
#[pyfunction]
fn push_activation_snapshot(ids: Vec<u32>, vals: Vec<f32>) -> PyResult<()> {
    with_brain(|brain| {
        brain.push_activation_snapshot(&ids, &vals);
    })
}

/// Learn from the Rust-side activation snapshot cache.
/// No activation data crosses FFI — everything stays in Rust.
/// Returns (hebb_count, anti_count, coactive_pairs).
#[pyfunction]
fn learn_from_snapshot_cache(
    hebbian_rate: f64,
    anti_hebbian_rate: f64,
    track_coactive: bool,
) -> PyResult<(usize, usize, Vec<(u32, u32)>)> {
    let h_lr = hebbian_rate as f32;
    let a_lr = anti_hebbian_rate as f32;
    with_brain(|brain| {
        // Get active neurons from the latest snapshot
        let active_neurons = brain.latest_snapshot_active();
        if active_neurons.is_empty() {
            return (0, 0, Vec::new());
        }

        // Compute window activity directly from the cache
        let window_activity = brain.compute_window_activity();

        let nn = brain.synapse_pool.num_neurons();
        let synapse_pool = &brain.synapse_pool;
        let shared_weights_attached = synapse_pool.shared_weights_attached();
        let learning_ablation = &brain.same_region_delay_learning_ablation;
        let learning_ablation_enabled = learning_ablation.is_enabled();

        if shared_weights_attached {
            let results: Vec<(usize, usize, Vec<(u32, u32)>)> = active_neurons
                .par_chunks(LEARNING_CHUNK_SIZE)
                .map(|chunk| {
                    let mut hebb_total = 0usize;
                    let mut anti_total = 0usize;
                    let mut coactive = if track_coactive {
                        Vec::new()
                    } else {
                        Vec::with_capacity(0)
                    };

                    for &(src_id, src_act) in chunk {
                        if src_id >= nn {
                            continue;
                        }
                        let source_region_idx = if learning_ablation_enabled {
                            region_id_for_neuron(src_id).map(|r| r.index())
                        } else {
                            None
                        };
                        let start = synapse_pool.offsets[src_id as usize] as usize;
                        let end = synapse_pool.offsets[src_id as usize + 1] as usize;
                        for i in start..end {
                            let tgt_id = synapse_pool.targets[i];
                            let raw_delay = synapse_pool.delays[i];
                            if same_region_delay_mask_suppresses_learning(
                                learning_ablation,
                                source_region_idx,
                                tgt_id,
                                raw_delay,
                            ) {
                                continue;
                            }
                            let plasticity = synapse_pool.plasticity[i];
                            let tgt_act = window_activity[tgt_id as usize];
                            if tgt_act > 0.0 {
                                if plasticity >= 0.01 {
                                    let delta = h_lr * src_act * tgt_act * plasticity;
                                    if delta > 0.0001 {
                                        synapse_pool.atomic_add_weight_at_index(i, delta);
                                        hebb_total += 1;
                                    }
                                }
                                if track_coactive {
                                    coactive.push((src_id, tgt_id));
                                }
                            } else if plasticity >= 0.01 {
                                let delta = -a_lr * src_act * plasticity;
                                if delta < -0.0001 {
                                    synapse_pool.atomic_add_weight_at_index(i, delta);
                                    anti_total += 1;
                                }
                            }
                        }
                    }
                    (hebb_total, anti_total, coactive)
                })
                .collect();

            let mut hebb_total = 0usize;
            let mut anti_total = 0usize;
            let mut all_coactive = if track_coactive { Vec::new() } else { Vec::with_capacity(0) };
            for (h, a, c) in results {
                hebb_total += h;
                anti_total += a;
                if track_coactive {
                    all_coactive.extend(c);
                }
            }
            return (hebb_total, anti_total, all_coactive);
        }

        // Non-shared-weights path: queue updates
        let results: Vec<(Vec<(u32, u32, u32, f32)>, Vec<(u32, u32, u32, f32)>, Vec<(u32, u32)>)> =
            active_neurons
                .par_chunks(LEARNING_CHUNK_SIZE)
                .map(|chunk| {
                    let mut hebb_updates = Vec::new();
                    let mut anti_updates = Vec::new();
                    let mut coactive = Vec::new();
                    for &(src_id, src_act) in chunk {
                        if src_id >= nn {
                            continue;
                        }
                        let source_region_idx = if learning_ablation_enabled {
                            region_id_for_neuron(src_id).map(|r| r.index())
                        } else {
                            None
                        };
                        let start = brain.synapse_pool.offsets[src_id as usize] as usize;
                        let end = brain.synapse_pool.offsets[src_id as usize + 1] as usize;
                        for i in start..end {
                            let tgt_id = brain.synapse_pool.targets[i];
                            let raw_delay = brain.synapse_pool.delays[i];
                            if same_region_delay_mask_suppresses_learning(
                                learning_ablation,
                                source_region_idx,
                                tgt_id,
                                raw_delay,
                            ) {
                                continue;
                            }
                            let plasticity = brain.synapse_pool.plasticity[i];
                            let tgt_act = window_activity[tgt_id as usize];
                            if tgt_act > 0.0 {
                                if plasticity >= 0.01 {
                                    let delta = h_lr * src_act * tgt_act * plasticity;
                                    if delta > 0.0001 {
                                        hebb_updates.push((src_id, tgt_id, i as u32, delta));
                                    }
                                }
                                if track_coactive {
                                    coactive.push((src_id, tgt_id));
                                }
                            } else if plasticity >= 0.01 {
                                let delta = -a_lr * src_act * plasticity;
                                if delta < -0.0001 {
                                    anti_updates.push((src_id, tgt_id, i as u32, delta));
                                }
                            }
                        }
                    }
                    (hebb_updates, anti_updates, coactive)
                })
                .collect();

        let mut hebb_total = 0usize;
        let mut anti_total = 0usize;
        let mut all_coactive = if track_coactive { Vec::new() } else { Vec::with_capacity(0) };
        for (hebb, anti, coact) in results {
            hebb_total += hebb.len();
            for (from, to, synapse_index, delta) in hebb {
                brain.synapse_pool.queue_indexed_update(from, to, synapse_index, delta);
            }
            anti_total += anti.len();
            for (from, to, synapse_index, delta) in anti {
                brain.synapse_pool.queue_indexed_update(from, to, synapse_index, delta);
            }
            if track_coactive {
                all_coactive.extend(coact);
            }
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

fn read_state_from_brain(brain: &Brain) -> NamedState {
    let mut state = HashMap::new();

    // Region activation levels
    state.insert("sensory_activation".to_string(), brain.cached_sensory_activation() as f64);
    state.insert("visual_activation".to_string(), brain.cached_visual_activation() as f64);
    state.insert("audio_activation".to_string(), brain.cached_audio_activation() as f64);
    state.insert("motor_activation".to_string(), brain.cached_motor_activation() as f64);
    state.insert("language_activation".to_string(), brain.cached_language_activation() as f64);
    state.insert("speech_activity".to_string(), brain.cached_speech_activity() as f64);

    // Emotion
    state.insert("emotion_polarity".to_string(), brain.emotion_polarity() as f64);
    state.insert("emotion_arousal".to_string(), brain.cached_emotion_arousal() as f64);

    // Executive
    state.insert(
        "executive_engagement".to_string(),
        brain.cached_executive_engagement() as f64,
    );
    state.insert("motor_conflict".to_string(), brain.motor_conflict() as f64);
    state.insert("planning_signal".to_string(), brain.cached_planning_signal() as f64);

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
    state.insert(
        "integration_input_count".to_string(),
        brain.integration_input_count(0.5) as f64,
    );

    state
}

fn read_state_compact_from_brain(brain: &Brain) -> CompactState {
    let (motor_approach, motor_withdraw) = match brain.decode_motor_action() {
        crate::regions::motor::MotorAction::Idle => (0.0, 0.0),
        crate::regions::motor::MotorAction::Approach { strength } => (strength as f64, 0.0),
        crate::regions::motor::MotorAction::Withdraw { strength } => (0.0, strength as f64),
        crate::regions::motor::MotorAction::Conflict { approach, withdraw } => {
            (approach as f64, withdraw as f64)
        }
    };

    vec![
        brain.cached_sensory_activation() as f64,
        brain.cached_visual_activation() as f64,
        brain.cached_audio_activation() as f64,
        brain.cached_motor_activation() as f64,
        brain.cached_language_activation() as f64,
        brain.cached_speech_activity() as f64,
        brain.emotion_polarity() as f64,
        brain.cached_emotion_arousal() as f64,
        brain.cached_executive_engagement() as f64,
        brain.motor_conflict() as f64,
        brain.cached_planning_signal() as f64,
        motor_approach,
        motor_withdraw,
        brain.inner_monologue_signal() as f64,
        brain.detect_pain() as f64,
        brain.integration_input_count(0.5) as f64,
    ]
}

fn collect_named_activations(
    brain: &Brain,
    min_activation: f32,
) -> (NamedActivations, Vec<u32>, NamedActiveCounts, u32) {
    let mut all_activations = HashMap::new();
    let mut active_ids = Vec::new();
    let mut active_counts = HashMap::new();
    let mut total_active = 0u32;

    for region in &brain.regions {
        let mut region_acts = Vec::new();
        for (local_idx, &activation) in region.neurons.activations.iter().enumerate() {
            if activation > min_activation {
                let global_id = region.local_to_global(local_idx as u32);
                region_acts.push((global_id, activation));
                active_ids.push(global_id);
            }
        }

        let count = region_acts.len() as u32;
        active_counts.insert(region.id.name().to_string(), count);
        total_active += count;
        if !region_acts.is_empty() {
            all_activations.insert(region.id.name().to_string(), region_acts);
        }
    }

    (all_activations, active_ids, active_counts, total_active)
}

fn collect_flat_activations(
    brain: &Brain,
    min_activation: f32,
) -> (Vec<(u32, f32)>, Vec<u32>, NamedActiveCounts, u32) {
    let mut activations = Vec::new();
    let mut active_ids = Vec::new();
    let mut active_counts = HashMap::new();
    let mut total_active = 0u32;

    for region in &brain.regions {
        let mut count = 0u32;
        for (local_idx, &activation) in region.neurons.activations.iter().enumerate() {
            if activation > min_activation {
                let global_id = region.local_to_global(local_idx as u32);
                activations.push((global_id, activation));
                active_ids.push(global_id);
                count += 1;
            }
        }

        active_counts.insert(region.id.name().to_string(), count);
        total_active += count;
    }

    (activations, active_ids, active_counts, total_active)
}

fn collect_region_neuron_ids(
    brain: &Brain,
    min_activation: f32,
) -> HashMap<String, Vec<u32>> {
    let mut region_neurons = HashMap::new();

    for region in &brain.regions {
        let mut neuron_ids = Vec::new();
        for (local_idx, &activation) in region.neurons.activations.iter().enumerate() {
            if activation > min_activation {
                neuron_ids.push(region.local_to_global(local_idx as u32));
            }
        }
        if !neuron_ids.is_empty() {
            region_neurons.insert(region.id.name().to_string(), neuron_ids);
        }
    }

    region_neurons
}

fn build_tick_profile_map(profile: &TickProfile, include_detailed: bool) -> NamedProfile {
    let mut data = HashMap::new();
    data.insert("prepare_ms".to_string(), profile.prepare_ms);
    data.insert("delayed_delivery_ms".to_string(), profile.delayed_delivery_ms);
    data.insert("propagate_ms".to_string(), profile.propagate_ms);
    data.insert("update_ms".to_string(), profile.update_ms);
    data.insert(
        "incoming_signal_count".to_string(),
        profile.incoming_signal_count as f64,
    );
    data.insert(
        "incoming_signal_abs_sum".to_string(),
        profile.incoming_signal_abs_sum,
    );
    data.insert(
        "immediate_signal_count".to_string(),
        profile.immediate_signal_count as f64,
    );
    data.insert(
        "immediate_signal_abs_sum".to_string(),
        profile.immediate_signal_abs_sum,
    );
    data.insert(
        "delayed_delivery_signal_count".to_string(),
        profile.delayed_delivery_signal_count as f64,
    );
    data.insert(
        "delayed_delivery_signal_abs_sum".to_string(),
        profile.delayed_delivery_signal_abs_sum,
    );
    data.insert(
        "scheduled_delayed_signal_count".to_string(),
        profile.scheduled_delayed_signal_count as f64,
    );
    data.insert(
        "scheduled_delayed_signal_abs_sum".to_string(),
        profile.scheduled_delayed_signal_abs_sum,
    );
    data.insert("total_fired".to_string(), profile.total_fired as f64);
    let total_refractory_ignored_abs_sum: f64 =
        profile.region_refractory_ignored_abs_sums.iter().sum();
    let total_fire_interval_sum: f64 = profile.region_fire_interval_sums.iter().sum();
    let total_fire_interval_count: u64 = profile.region_fire_interval_counts.iter().sum();
    data.insert(
        "refractory_ignored_abs_sum".to_string(),
        total_refractory_ignored_abs_sum,
    );
    data.insert("fire_interval_sum".to_string(), total_fire_interval_sum);
    data.insert(
        "fire_interval_count".to_string(),
        total_fire_interval_count as f64,
    );
    if !include_detailed {
        data.insert("tick_profile_ms".to_string(), profile.total_ms());
        return data;
    }
    for &region_id in RegionId::ALL.iter() {
        let idx = region_id.index();
        let region_name = region_id.name();
        data.insert(
            format!("incoming_region_{}_signals", region_name),
            profile.incoming_region_signal_counts[idx] as f64,
        );
        data.insert(
            format!("incoming_region_{}_abs_sum", region_name),
            profile.incoming_region_signal_abs_sums[idx],
        );
        data.insert(
            format!("fired_region_{}_neurons", region_name),
            profile.fired_region_counts[idx] as f64,
        );
        data.insert(
            format!("potential_region_{}_pre_leak_sum", region_name),
            profile.region_positive_pre_leak_sums[idx],
        );
        data.insert(
            format!("potential_region_{}_leak_loss_sum", region_name),
            profile.region_positive_leak_loss_sums[idx],
        );
        data.insert(
            format!("potential_region_{}_reset_sum", region_name),
            profile.region_positive_reset_sums[idx],
        );
        data.insert(
            format!("potential_region_{}_carried_sum", region_name),
            profile.region_positive_carried_sums[idx],
        );
        data.insert(
            format!("refractory_ignored_region_{}_abs_sum", region_name),
            profile.region_refractory_ignored_abs_sums[idx],
        );
        data.insert(
            format!("refractory_ignored_region_{}_immediate_same_abs_sum", region_name),
            profile.region_refractory_ignored_immediate_same_abs_sums[idx],
        );
        data.insert(
            format!("refractory_ignored_region_{}_immediate_cross_abs_sum", region_name),
            profile.region_refractory_ignored_immediate_cross_abs_sums[idx],
        );
        data.insert(
            format!("refractory_ignored_region_{}_delayed_same_abs_sum", region_name),
            profile.region_refractory_ignored_delayed_same_abs_sums[idx],
        );
        data.insert(
            format!("refractory_ignored_region_{}_delayed_cross_abs_sum", region_name),
            profile.region_refractory_ignored_delayed_cross_abs_sums[idx],
        );
        data.insert(
            format!("fire_interval_region_{}_sum", region_name),
            profile.region_fire_interval_sums[idx],
        );
        data.insert(
            format!("fire_interval_region_{}_count", region_name),
            profile.region_fire_interval_counts[idx] as f64,
        );
    }
    for &source_region in RegionId::ALL.iter() {
        let source_idx = source_region.index();
        let source_name = source_region.name();
        for &target_region in RegionId::ALL.iter() {
            let target_idx = target_region.index();
            let target_name = target_region.name();
            data.insert(
                format!("delayed_flow_{}_to_{}_signals", source_name, target_name),
                profile.delayed_flow_signal_counts[source_idx][target_idx] as f64,
            );
            data.insert(
                format!("delayed_flow_{}_to_{}_abs_sum", source_name, target_name),
                profile.delayed_flow_signal_abs_sums[source_idx][target_idx],
            );
        }
    }
    data.insert("tick_profile_ms".to_string(), profile.total_ms());
    data
}

fn merge_tick_profile(into: &mut TickProfile, from: &TickProfile) {
    into.prepare_ms += from.prepare_ms;
    into.delayed_delivery_ms += from.delayed_delivery_ms;
    into.propagate_ms += from.propagate_ms;
    into.update_ms += from.update_ms;
    into.incoming_signal_count += from.incoming_signal_count;
    into.incoming_signal_abs_sum += from.incoming_signal_abs_sum;
    into.immediate_signal_count += from.immediate_signal_count;
    into.immediate_signal_abs_sum += from.immediate_signal_abs_sum;
    into.delayed_delivery_signal_count += from.delayed_delivery_signal_count;
    into.delayed_delivery_signal_abs_sum += from.delayed_delivery_signal_abs_sum;
    into.scheduled_delayed_signal_count += from.scheduled_delayed_signal_count;
    into.scheduled_delayed_signal_abs_sum += from.scheduled_delayed_signal_abs_sum;
    into.total_fired += from.total_fired;

    for idx in 0..RegionId::ALL.len() {
        into.incoming_region_signal_counts[idx] += from.incoming_region_signal_counts[idx];
        into.incoming_region_signal_abs_sums[idx] += from.incoming_region_signal_abs_sums[idx];
        into.fired_region_counts[idx] += from.fired_region_counts[idx];
        into.region_positive_pre_leak_sums[idx] += from.region_positive_pre_leak_sums[idx];
        into.region_positive_leak_loss_sums[idx] += from.region_positive_leak_loss_sums[idx];
        into.region_positive_reset_sums[idx] += from.region_positive_reset_sums[idx];
        into.region_positive_carried_sums[idx] += from.region_positive_carried_sums[idx];
        into.region_refractory_ignored_abs_sums[idx] += from.region_refractory_ignored_abs_sums[idx];
        into.region_refractory_ignored_immediate_same_abs_sums[idx] +=
            from.region_refractory_ignored_immediate_same_abs_sums[idx];
        into.region_refractory_ignored_immediate_cross_abs_sums[idx] +=
            from.region_refractory_ignored_immediate_cross_abs_sums[idx];
        into.region_refractory_ignored_delayed_same_abs_sums[idx] +=
            from.region_refractory_ignored_delayed_same_abs_sums[idx];
        into.region_refractory_ignored_delayed_cross_abs_sums[idx] +=
            from.region_refractory_ignored_delayed_cross_abs_sums[idx];
        into.region_fire_interval_sums[idx] += from.region_fire_interval_sums[idx];
        into.region_fire_interval_counts[idx] += from.region_fire_interval_counts[idx];

        for target_idx in 0..RegionId::ALL.len() {
            into.delayed_flow_signal_counts[idx][target_idx] +=
                from.delayed_flow_signal_counts[idx][target_idx];
            into.delayed_flow_signal_abs_sums[idx][target_idx] +=
                from.delayed_flow_signal_abs_sums[idx][target_idx];
        }
    }
}

/// Read all brain state needed for Python learning in one call.
/// Returns a dict with all activation levels, emotion, executive, motor, etc.
#[pyfunction]
fn batch_read_state() -> PyResult<HashMap<String, f64>> {
    with_brain_ref(|brain| read_state_from_brain(brain))
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

fn with_trace_matchers<F, R>(f: F) -> PyResult<R>
where
    F: FnOnce(&mut TraceMatcherRegistry) -> R,
{
    let mut guard = TRACE_MATCHERS
        .lock()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(f(&mut guard))
}

fn with_novel_trackers<F, R>(f: F) -> PyResult<R>
where
    F: FnOnce(&mut NovelPatternRegistry) -> R,
{
    let mut guard = NOVEL_TRACKERS
        .lock()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(f(&mut guard))
}

fn with_binding_trackers<F, R>(f: F) -> PyResult<R>
where
    F: FnOnce(&mut BindingTrackerRegistry) -> R,
{
    let mut guard = BINDING_TRACKERS
        .lock()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(f(&mut guard))
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

#[pyfunction]
fn reset_runtime_state() -> PyResult<()> {
    with_brain(|brain| {
        brain.reset_runtime_state();
        brain.tick_count = 0;
    })
}

#[pyfunction]
fn save_brain_checkpoint(path: String) -> PyResult<()> {
    let mut guard = BRAIN.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let brain = guard
        .as_mut()
        .ok_or_else(|| PyValueError::new_err("Brain not initialized"))?;
    brain.synapse_pool.refresh_owned_weights_from_shared();
    write_brain_checkpoint(&path, brain)
}

#[pyfunction]
fn dump_brain_checkpoint_bytes(py: Python<'_>) -> PyResult<Py<PyBytes>> {
    let mut guard = BRAIN.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let brain = guard
        .as_mut()
        .ok_or_else(|| PyValueError::new_err("Brain not initialized"))?;
    brain.synapse_pool.refresh_owned_weights_from_shared();
    let bytes = serialize_brain_checkpoint(brain)?;
    Ok(PyBytes::new_bound(py, &bytes).unbind())
}

#[pyfunction]
fn save_brain_runtime_checkpoint(path: String) -> PyResult<()> {
    let guard = BRAIN.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let brain = guard
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("Brain not initialized"))?;
    write_brain_runtime_checkpoint(&path, brain)
}

#[pyfunction]
fn dump_brain_runtime_checkpoint_bytes(py: Python<'_>) -> PyResult<Py<PyBytes>> {
    let guard = BRAIN.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let brain = guard
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("Brain not initialized"))?;
    let bytes = serialize_brain_runtime_checkpoint(brain)?;
    Ok(PyBytes::new_bound(py, &bytes).unbind())
}

#[pyfunction]
fn load_brain_checkpoint(path: String) -> PyResult<()> {
    let brain = read_brain_checkpoint(&path)?;

    let mut guard = BRAIN.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
    *guard = Some(brain);
    Ok(())
}

#[pyfunction]
fn load_brain_checkpoint_bytes(bytes: Vec<u8>) -> PyResult<()> {
    let brain = deserialize_brain_checkpoint(&bytes)?;

    let mut guard = BRAIN.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
    *guard = Some(brain);
    Ok(())
}

#[pyfunction]
fn load_brain_runtime_checkpoint(path: String) -> PyResult<()> {
    let checkpoint = read_brain_runtime_checkpoint(&path)?;

    let mut guard = BRAIN.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let brain = guard
        .as_mut()
        .ok_or_else(|| PyValueError::new_err("Brain not initialized"))?;
    checkpoint.apply_to(brain)
}

#[pyfunction]
fn load_brain_runtime_checkpoint_bytes(bytes: Vec<u8>) -> PyResult<()> {
    let checkpoint = deserialize_brain_runtime_checkpoint(&bytes)?;

    let mut guard = BRAIN.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let brain = guard
        .as_mut()
        .ok_or_else(|| PyValueError::new_err("Brain not initialized"))?;
    checkpoint.apply_to(brain)
}

#[pyfunction]
fn merge_brain_checkpoints(
    checkpoint_paths: Vec<String>,
    output_path: String,
    leader_index: usize,
) -> PyResult<HashMap<String, f64>> {
    if checkpoint_paths.is_empty() {
        return Err(PyValueError::new_err(
            "merge_brain_checkpoints requires at least one checkpoint path",
        ));
    }
    if leader_index >= checkpoint_paths.len() {
        return Err(PyValueError::new_err(format!(
            "Leader index {} is out of range for {} checkpoints",
            leader_index,
            checkpoint_paths.len()
        )));
    }

    let merge_started = Instant::now();
    let worker_count = checkpoint_paths.len();
    let mut leader_brain = read_brain_checkpoint(&checkpoint_paths[leader_index])?;
    let leader_weight_len = leader_brain.synapse_pool.weights.len();
    let mut weight_sums: Vec<f64> = leader_brain
        .synapse_pool
        .weights
        .iter()
        .map(|weight| *weight as f64)
        .collect();
    let mut merged_bindings: Vec<Binding> = leader_brain.binding_store.iter().cloned().collect();
    let mut seen_binding_keys: HashSet<String> = merged_bindings
        .iter()
        .map(binding_pair_key)
        .collect();
    let mut next_binding_id = merged_bindings
        .iter()
        .map(|binding| binding.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let mut added_bindings = 0u64;
    let mut duplicate_bindings = 0u64;

    for (path_index, checkpoint_path) in checkpoint_paths.iter().enumerate() {
        if path_index == leader_index {
            continue;
        }

        let worker_brain = read_brain_checkpoint(checkpoint_path)?;
        if !synapse_topology_matches(&leader_brain, &worker_brain) {
            return Err(PyValueError::new_err(format!(
                "Worker checkpoint topology diverged from leader: {}",
                checkpoint_path,
            )));
        }
        if worker_brain.synapse_pool.weights.len() != leader_weight_len {
            return Err(PyValueError::new_err(format!(
                "Worker checkpoint synapse count diverged from leader: {}",
                checkpoint_path,
            )));
        }

        for (idx, weight) in worker_brain.synapse_pool.weights.iter().enumerate() {
            weight_sums[idx] += *weight as f64;
        }

        for binding in worker_brain.binding_store.iter() {
            let binding_key = binding_pair_key(binding);
            if !seen_binding_keys.insert(binding_key) {
                duplicate_bindings += 1;
                continue;
            }

            let mut cloned = binding.clone();
            cloned.id = next_binding_id;
            next_binding_id = next_binding_id.saturating_add(1);
            merged_bindings.push(cloned);
            added_bindings += 1;
        }
    }

    let divisor = worker_count as f64;
    for (idx, weight_sum) in weight_sums.into_iter().enumerate() {
        leader_brain.synapse_pool.weights[idx] = (weight_sum / divisor) as f32;
    }
    leader_brain.binding_store = BindingStore::from_bindings(merged_bindings);

    write_brain_checkpoint(&output_path, &leader_brain)?;

    let mut summary = HashMap::new();
    summary.insert("worker_count".to_string(), worker_count as f64);
    summary.insert("leader_index".to_string(), leader_index as f64);
    summary.insert("synapse_count".to_string(), leader_weight_len as f64);
    summary.insert(
        "binding_count".to_string(),
        leader_brain.binding_store.len() as f64,
    );
    summary.insert("bindings_added".to_string(), added_bindings as f64);
    summary.insert(
        "bindings_deduped".to_string(),
        duplicate_bindings as f64,
    );
    summary.insert(
        "merge_ms".to_string(),
        merge_started.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(summary)
}

#[pyfunction]
fn set_same_region_delay_ablation(regions: Vec<String>, delays: Vec<u8>) -> PyResult<()> {
    let region_ids = parse_region_ids(&regions)?;
    let delay_values = normalize_delays(&delays)?;
    with_brain(|brain| brain.configure_same_region_delay_ablation(&region_ids, &delay_values))
}

#[pyfunction]
fn clear_same_region_delay_ablation() -> PyResult<()> {
    with_brain(|brain| brain.clear_same_region_delay_ablation())
}

#[pyfunction]
fn set_same_region_delay_learning_ablation(regions: Vec<String>, delays: Vec<u8>) -> PyResult<()> {
    let region_ids = parse_region_ids(&regions)?;
    let delay_values = normalize_delays(&delays)?;
    with_brain(|brain| {
        brain.configure_same_region_delay_learning_ablation(&region_ids, &delay_values)
    })
}

#[pyfunction]
fn clear_same_region_delay_learning_ablation() -> PyResult<()> {
    with_brain(|brain| brain.clear_same_region_delay_learning_ablation())
}

// === TRACE MATCHING INDEX ===

#[pyfunction]
fn trace_index_create(store_id: u64) -> PyResult<()> {
    with_trace_matchers(|registry| registry.create_store(store_id))
}

#[pyfunction]
fn trace_index_clear(store_id: u64) -> PyResult<()> {
    with_trace_matchers(|registry| registry.clear_store(store_id))
}

#[pyfunction]
fn trace_index_clear_working_memory(store_id: u64) -> PyResult<()> {
    with_trace_matchers(|registry| registry.clear_working_memory(store_id))
}

#[pyfunction]
fn trace_index_set_working_memory(store_id: u64, slots: Vec<(String, f32)>) -> PyResult<()> {
    with_trace_matchers(|registry| registry.set_working_memory(store_id, slots))
}

#[pyfunction]
fn trace_index_drop(store_id: u64) -> PyResult<()> {
    with_trace_matchers(|registry| registry.drop_store(store_id))
}

#[pyfunction]
fn trace_index_upsert_trace(store_id: u64, trace_id: String, neurons: Vec<u32>) -> PyResult<()> {
    with_trace_matchers(|registry| {
        registry.upsert_trace(
            store_id,
            trace_id,
            neurons,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            0.1,
            1.0,
            1.0,
            0.0,
            0,
            0,
        )
    })
}

#[pyfunction]
fn trace_index_upsert_trace_full(
    store_id: u64,
    trace_id: String,
    neurons: Vec<u32>,
    memory_short_neurons: Vec<u32>,
    memory_long_neurons: Vec<u32>,
    speech_neurons: Vec<u32>,
    co_trace_ids: Vec<String>,
    strength: f64,
    decay: f64,
    novelty: f64,
    polarity: f64,
    fire_count: u32,
    last_fired: u64,
) -> PyResult<()> {
    with_trace_matchers(|registry| {
        registry.upsert_trace(
            store_id,
            trace_id,
            neurons,
            memory_short_neurons,
            memory_long_neurons,
            speech_neurons,
            co_trace_ids,
            strength as f32,
            decay as f32,
            novelty as f32,
            polarity as f32,
            fire_count,
            last_fired,
        )
    })
}

#[pyfunction]
fn trace_index_remove_trace(store_id: u64, trace_id: &str) -> PyResult<()> {
    with_trace_matchers(|registry| registry.remove_trace(store_id, trace_id))
}

#[pyfunction]
fn trace_index_matching_traces(
    store_id: u64,
    active_neurons: Vec<u32>,
    threshold: f64,
) -> PyResult<Vec<(String, f64)>> {
    with_trace_matchers(|registry| {
        registry
            .matching_traces(store_id, &active_neurons, threshold as f32)
            .into_iter()
            .map(|(trace_id, score)| (trace_id, score as f64))
            .collect::<Vec<_>>()
    })
}

#[pyfunction(signature = (store_id, trace_ids=None))]
fn trace_index_runtime_snapshots(
    store_id: u64,
    trace_ids: Option<Vec<String>>,
) -> PyResult<Vec<(String, f64, f64, f64, f64, u32, u64)>> {
    with_trace_matchers(|registry| {
        registry
            .runtime_snapshots(store_id, trace_ids.as_deref())
            .into_iter()
            .map(|snapshot| {
                (
                    snapshot.id,
                    snapshot.strength as f64,
                    snapshot.decay as f64,
                    snapshot.novelty as f64,
                    snapshot.polarity as f64,
                    snapshot.fire_count,
                    snapshot.last_fired,
                )
            })
            .collect::<Vec<_>>()
    })
}

#[pyfunction]
fn trace_index_predict_regions(
    store_id: u64,
    active_traces: Vec<(String, f64)>,
    working_memory_ids: Vec<String>,
) -> PyResult<HashMap<String, f64>> {
    with_trace_matchers(|registry| {
        registry
            .predict_region_activity(
                store_id,
                &active_traces
                    .into_iter()
                    .map(|(trace_id, score)| (trace_id, score as f32))
                    .collect::<Vec<_>>(),
                &working_memory_ids,
            )
            .into_iter()
            .map(|(region, score)| (region, score as f64))
            .collect::<HashMap<_, _>>()
    })
}

#[pyfunction]
fn trace_index_active_primary_patterns(
    store_id: u64,
    active_traces: Vec<(String, f64)>,
) -> PyResult<Vec<(String, String)>> {
    with_trace_matchers(|registry| {
        registry.active_primary_patterns(
            store_id,
            &active_traces
                .into_iter()
                .map(|(trace_id, score)| (trace_id, score as f32))
                .collect::<Vec<_>>(),
        )
    })
}

// === FORMATION TRACKERS ===

#[pyfunction]
fn novel_tracker_create(tracker_id: u64) -> PyResult<()> {
    with_novel_trackers(|registry| registry.create_tracker(tracker_id))
}

#[pyfunction]
fn novel_tracker_clear(tracker_id: u64) -> PyResult<()> {
    with_novel_trackers(|registry| registry.clear_tracker(tracker_id))
}

#[pyfunction]
fn novel_tracker_drop(tracker_id: u64) -> PyResult<()> {
    with_novel_trackers(|registry| registry.drop_tracker(tracker_id))
}

#[pyfunction]
fn novel_tracker_update(
    tracker_id: u64,
    region_neurons: HashMap<String, Vec<u32>>,
    novelty: f64,
    min_regions: usize,
    persistence_required: usize,
    jaccard_threshold: Option<f32>,
) -> PyResult<Vec<HashMap<String, Vec<u32>>>> {
    with_novel_trackers(|registry| {
        registry.update(
            tracker_id,
            region_neurons,
            novelty as f32,
            min_regions,
            persistence_required,
            jaccard_threshold.unwrap_or(0.4),
        )
    })
}

#[pyfunction]
fn novel_tracker_update_from_brain(
    tracker_id: u64,
    novelty: f64,
    min_regions: usize,
    persistence_required: usize,
    min_activation: f32,
    jaccard_threshold: Option<f32>,
) -> PyResult<Vec<HashMap<String, Vec<u32>>>> {
    let region_neurons = with_brain_ref(|brain| collect_region_neuron_ids(brain, min_activation))?;
    with_novel_trackers(|registry| {
        registry.update(
            tracker_id,
            region_neurons,
            novelty as f32,
            min_regions,
            persistence_required,
            jaccard_threshold.unwrap_or(0.4),
        )
    })
}

#[pyfunction]
fn binding_tracker_create(tracker_id: u64) -> PyResult<()> {
    with_binding_trackers(|registry| registry.create_tracker(tracker_id))
}

#[pyfunction]
fn binding_tracker_clear(tracker_id: u64) -> PyResult<()> {
    with_binding_trackers(|registry| registry.clear_tracker(tracker_id))
}

#[pyfunction]
fn binding_tracker_drop(tracker_id: u64) -> PyResult<()> {
    with_binding_trackers(|registry| registry.drop_tracker(tracker_id))
}

#[pyfunction]
fn binding_tracker_consume(
    tracker_id: u64,
    ready_pairs: Vec<(String, String, String, String)>,
) -> PyResult<()> {
    with_binding_trackers(|registry| registry.consume(tracker_id, ready_pairs))
}

#[pyfunction]
fn binding_tracker_record(
    tracker_id: u64,
    active_patterns: Vec<(String, String)>,
    tick: u64,
    formation_count: usize,
    temporal_window: u64,
) -> PyResult<Vec<(String, String, String, String, f64)>> {
    with_binding_trackers(|registry| {
        registry
            .record(tracker_id, active_patterns, tick, formation_count, temporal_window)
            .into_iter()
            .map(|(a, ar, b, br, delta)| (a, ar, b, br, delta as f64))
            .collect::<Vec<_>>()
    })
}

#[pyfunction]
fn binding_tracker_record_detailed(
    tracker_id: u64,
    active_patterns: Vec<(String, String)>,
    tick: u64,
    formation_count: usize,
    temporal_window: u64,
) -> PyResult<Vec<(String, String, String, String, f64, usize, u64, u64, u64)>> {
    with_binding_trackers(|registry| {
        registry
            .record_detailed(tracker_id, active_patterns, tick, formation_count, temporal_window)
            .into_iter()
            .map(|pair| {
                (
                    pair.trace_id_a,
                    pair.region_a,
                    pair.trace_id_b,
                    pair.region_b,
                    pair.avg_delta as f64,
                    pair.support_count,
                    pair.span_ticks,
                    pair.first_tick,
                    pair.last_tick,
                )
            })
            .collect::<Vec<_>>()
    })
}

#[pyfunction]
fn binding_tracker_record_from_active_traces(
    tracker_id: u64,
    store_id: u64,
    active_traces: Vec<(String, f64)>,
    tick: u64,
    formation_count: usize,
    temporal_window: u64,
) -> PyResult<Vec<(String, String, String, String, f64)>> {
    let active_patterns = with_trace_matchers(|registry| {
        registry.active_primary_patterns(
            store_id,
            &active_traces
                .into_iter()
                .map(|(trace_id, score)| (trace_id, score as f32))
                .collect::<Vec<_>>(),
        )
    })?;

    with_binding_trackers(|registry| {
        registry
            .record(tracker_id, active_patterns, tick, formation_count, temporal_window)
            .into_iter()
            .map(|(a, ar, b, br, delta)| (a, ar, b, br, delta as f64))
            .collect::<Vec<_>>()
    })
}

#[pyfunction]
fn binding_tracker_record_detailed_from_active_traces(
    tracker_id: u64,
    store_id: u64,
    active_traces: Vec<(String, f64)>,
    tick: u64,
    formation_count: usize,
    temporal_window: u64,
) -> PyResult<Vec<(String, String, String, String, f64, usize, u64, u64, u64)>> {
    let active_patterns = with_trace_matchers(|registry| {
        registry.active_primary_patterns(
            store_id,
            &active_traces
                .into_iter()
                .map(|(trace_id, score)| (trace_id, score as f32))
                .collect::<Vec<_>>(),
        )
    })?;

    with_binding_trackers(|registry| {
        registry
            .record_detailed(tracker_id, active_patterns, tick, formation_count, temporal_window)
            .into_iter()
            .map(|pair| {
                (
                    pair.trace_id_a,
                    pair.region_a,
                    pair.trace_id_b,
                    pair.region_b,
                    pair.avg_delta as f64,
                    pair.support_count,
                    pair.span_ticks,
                    pair.first_tick,
                    pair.last_tick,
                )
            })
            .collect::<Vec<_>>()
    })
}

#[pyfunction]
fn binding_tracker_cleanup(tracker_id: u64, current_tick: u64, max_age: u64) -> PyResult<()> {
    with_binding_trackers(|registry| registry.cleanup(tracker_id, current_tick, max_age))
}

#[pyfunction]
fn binding_tracker_mark_bound(
    tracker_id: u64,
    keys: Vec<(String, String, String, String)>,
) -> PyResult<()> {
    with_binding_trackers(|registry| registry.mark_bound(tracker_id, keys))
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
fn tick_profiled() -> PyResult<(u64, HashMap<String, u32>, u32, NamedProfile)> {
    with_brain(|brain| {
        let result = brain.tick();
        let named_counts: HashMap<String, u32> = result
            .active_counts
            .into_iter()
            .map(|(id, count)| (id.name().to_string(), count))
            .collect();
        let profile = build_tick_profile_map(&result.profile, true);
        (result.tick_number, named_counts, result.total_active, profile)
    })
}

#[pyfunction]
fn tick_batch(
    batch_size: usize,
) -> PyResult<(u64, HashMap<String, u32>, u32, NamedProfile, usize)> {
    let effective_batch_size = batch_size.max(1);
    with_brain(|brain| {
        let mut last_result = brain.tick();
        let mut combined_profile = TickProfile::default();
        merge_tick_profile(&mut combined_profile, &last_result.profile);

        for _ in 1..effective_batch_size {
            let result = brain.tick();
            merge_tick_profile(&mut combined_profile, &result.profile);
            last_result = result;
        }

        let named_counts: HashMap<String, u32> = last_result
            .active_counts
            .into_iter()
            .map(|(id, count)| (id.name().to_string(), count))
            .collect();
        let profile = build_tick_profile_map(&combined_profile, true);
        (
            last_result.tick_number,
            named_counts,
            last_result.total_active,
            profile,
            effective_batch_size,
        )
    })
}

#[pyfunction]
fn tick_batch_compact(
    batch_size: usize,
) -> PyResult<(u64, HashMap<String, u32>, u32, NamedProfile, usize)> {
    let effective_batch_size = batch_size.max(1);
    with_brain(|brain| {
        let mut last_result = brain.tick();
        let mut combined_profile = TickProfile::default();
        merge_tick_profile(&mut combined_profile, &last_result.profile);

        for _ in 1..effective_batch_size {
            let result = brain.tick();
            merge_tick_profile(&mut combined_profile, &result.profile);
            last_result = result;
        }

        let named_counts: HashMap<String, u32> = last_result
            .active_counts
            .into_iter()
            .map(|(id, count)| (id.name().to_string(), count))
            .collect();
        let profile = build_tick_profile_map(&combined_profile, false);
        (
            last_result.tick_number,
            named_counts,
            last_result.total_active,
            profile,
            effective_batch_size,
        )
    })
}

#[pyfunction]
fn evaluate_tick(
    store_id: u64,
    min_activation: f32,
    threshold: f64,
) -> PyResult<(NamedActivations, Vec<(String, f64)>, NamedState, NamedActiveCounts, u32, NamedProfile)> {
    let (
        activations,
        active_ids,
        active_counts,
        total_active,
        snapshot_ms,
        state,
        batch_state_ms,
    ) = with_brain_ref(|brain| {
        let snapshot_started = Instant::now();
        let (activations, active_ids, active_counts, total_active) =
            collect_named_activations(brain, min_activation);
        let snapshot_ms = snapshot_started.elapsed().as_secs_f64() * 1000.0;

        let batch_state_started = Instant::now();
        let state = read_state_from_brain(brain);
        let batch_state_ms = batch_state_started.elapsed().as_secs_f64() * 1000.0;

        (
            activations,
            active_ids,
            active_counts,
            total_active,
            snapshot_ms,
            state,
            batch_state_ms,
        )
    })?;

    let trace_match_started = Instant::now();
    let active_traces = if active_ids.is_empty() {
        Vec::new()
    } else {
        with_trace_matchers(|registry| {
            registry
                .matching_traces(store_id, &active_ids, threshold as f32)
                .into_iter()
                .map(|(trace_id, score)| (trace_id, score as f64))
                .collect::<Vec<_>>()
        })?
    };
    let trace_match_ms = trace_match_started.elapsed().as_secs_f64() * 1000.0;

    let mut profile = HashMap::new();
    profile.insert("snapshot_ms".to_string(), snapshot_ms);
    profile.insert("batch_state_ms".to_string(), batch_state_ms);
    profile.insert("trace_match_ms".to_string(), trace_match_ms);
    profile.insert(
        "evaluation_rust_ms".to_string(),
        snapshot_ms + batch_state_ms + trace_match_ms,
    );

    Ok((
        activations,
        active_traces,
        state,
        active_counts,
        total_active,
        profile,
    ))
}

fn evaluate_tick_compact_impl(
    store_id: u64,
    min_activation: f32,
    threshold: f64,
    tick_num: u64,
    working_memory_decay: f64,
    working_memory_capacity: usize,
    working_memory_overlay_cap: usize,
    trace_active_neuron_budget: usize,
    trace_freshness_retention: f64,
    trace_freshness_floor: f64,
    trace_freshness_min_score: f64,
    trace_refresh_max_boost: f64,
    trace_age_decay_window: u64,
    trace_age_floor_ceiling: f64,
    binding_recall_min_relative_weight: f64,
    binding_recall_boost_scale: f64,
    speech_boost_multiplier: f64,
    include_detailed_profile: bool,
) -> PyResult<(
    Vec<(u32, f32)>,
    Vec<(String, f64)>,
    NamedState,
    NamedActiveCounts,
    u32,
    Vec<(String, f64)>,
    BindingRecallCandidates,
    NamedProfile,
)> {
    let (
        activations,
        active_ids,
        active_counts,
        total_active,
        snapshot_ms,
        state,
        batch_state_ms,
    ) = with_brain_ref(|brain| {
        let snapshot_started = Instant::now();
        let (activations, active_ids, active_counts, total_active) =
            collect_flat_activations(brain, min_activation);
        let snapshot_ms = snapshot_started.elapsed().as_secs_f64() * 1000.0;

        let batch_state_started = Instant::now();
        let state = read_state_from_brain(brain);
        let batch_state_ms = batch_state_started.elapsed().as_secs_f64() * 1000.0;

        (
            activations,
            active_ids,
            active_counts,
            total_active,
            snapshot_ms,
            state,
            batch_state_ms,
        )
    })?;

    let language_activation = state.get("language_activation").copied().unwrap_or(0.0) as f32;
    let emotion_polarity = state.get("emotion_polarity").copied().unwrap_or(0.0) as f32;

    let trace_match_started = Instant::now();
    let trace_evaluation = with_trace_matchers(|registry| {
        registry.evaluate_active_traces(
            store_id,
            &active_ids,
            threshold as f32,
            tick_num,
            emotion_polarity,
            language_activation,
            trace_active_neuron_budget,
            trace_freshness_retention as f32,
            trace_freshness_floor as f32,
            trace_freshness_min_score as f32,
            trace_refresh_max_boost as f32,
            trace_age_decay_window,
            trace_age_floor_ceiling as f32,
            working_memory_decay as f32,
            working_memory_capacity,
            working_memory_overlay_cap,
            speech_boost_multiplier as f32,
        )
    })?;
    let trace_match_ms = trace_match_started.elapsed().as_secs_f64() * 1000.0;

    let trace_side_effects_started = Instant::now();
    let unique_working_memory_neurons = unique_neurons(&trace_evaluation.working_memory_neurons);
    let working_memory_boost_neurons = unique_working_memory_neurons.len() as u32;
    let working_memory_boost_region_counts =
        region_counts_for_neurons(&unique_working_memory_neurons);
    let mut all_pattern_completion_neurons = Vec::new();
    for neurons in &trace_evaluation.memory_long_patterns {
        all_pattern_completion_neurons.extend(neurons.iter().copied());
    }
    let unique_pattern_completion_neurons = unique_neurons(&all_pattern_completion_neurons);
    let pattern_completion_neurons = unique_pattern_completion_neurons.len() as u32;
    let pattern_completion_region_counts =
        region_counts_for_neurons(&unique_pattern_completion_neurons);
    let mut all_speech_boost_neurons = Vec::new();
    for (neurons, boost) in &trace_evaluation.speech_boosts {
        if *boost <= 0.0 {
            continue;
        }
        all_speech_boost_neurons.extend(neurons.iter().copied());
    }
    let unique_speech_boost_neurons = unique_neurons(&all_speech_boost_neurons);
    let speech_boost_neurons = unique_speech_boost_neurons.len() as u32;
    let speech_boost_region_counts = region_counts_for_neurons(&unique_speech_boost_neurons);
    with_brain(|brain| {
        if !trace_evaluation.working_memory_neurons.is_empty() {
            brain.boost_working_memory(&trace_evaluation.working_memory_neurons, 0.15);
        }
        for neurons in &trace_evaluation.memory_long_patterns {
            if !neurons.is_empty() {
                brain.pattern_complete(neurons, 0.4, 0.3);
            }
        }
        for (neurons, boost) in &trace_evaluation.speech_boosts {
            if !neurons.is_empty() && *boost > 0.0 {
                brain.boost_speech(neurons, *boost);
            }
        }
    })?;
    let trace_side_effects_ms = trace_side_effects_started.elapsed().as_secs_f64() * 1000.0;

    let binding_recall_candidates = with_brain_ref(|brain| {
        brain
            .get_binding_recall_candidates(
                min_activation,
                binding_recall_min_relative_weight as f32,
            )
            .into_iter()
            .map(|(binding_id, relative_weight, source_activation_ratio)| {
                (
                    binding_id,
                    relative_weight as f64,
                    source_activation_ratio as f64,
                )
            })
            .collect::<Vec<_>>()
    })?;

    let binding_recall_started = Instant::now();
    let (binding_recall_bindings, binding_recall_neurons, binding_recall_region_counts, binding_recall_max_relative_weight, binding_recall_max_boost) =
        with_brain(|brain| {
            brain.binding_recall(
                min_activation,
                binding_recall_min_relative_weight as f32,
                binding_recall_boost_scale as f32,
            )
        })?;
    let binding_recall_ms = binding_recall_started.elapsed().as_secs_f64() * 1000.0;

    let mut profile = HashMap::new();
    profile.insert("snapshot_ms".to_string(), snapshot_ms);
    profile.insert("batch_state_ms".to_string(), batch_state_ms);
    profile.insert("trace_match_ms".to_string(), trace_match_ms);
    profile.insert(
        "trace_candidates".to_string(),
        trace_evaluation.candidate_traces as f64,
    );
    profile.insert(
        "working_memory_boost_neurons".to_string(),
        working_memory_boost_neurons as f64,
    );
    profile.insert(
        "pattern_completion_neurons".to_string(),
        pattern_completion_neurons as f64,
    );
    profile.insert(
        "speech_boost_neurons".to_string(),
        speech_boost_neurons as f64,
    );
    profile.insert("trace_side_effects_ms".to_string(), trace_side_effects_ms);
    profile.insert("binding_recall_ms".to_string(), binding_recall_ms);
    profile.insert(
        "binding_recall_bindings".to_string(),
        binding_recall_bindings as f64,
    );
    profile.insert(
        "binding_recall_neurons".to_string(),
        binding_recall_neurons as f64,
    );
    profile.insert(
        "binding_recall_max_relative_weight".to_string(),
        binding_recall_max_relative_weight as f64,
    );
    profile.insert(
        "binding_recall_max_boost".to_string(),
        binding_recall_max_boost as f64,
    );
    if include_detailed_profile {
        insert_region_profile_counts(
            &mut profile,
            "working_memory_boost",
            &working_memory_boost_region_counts,
        );
        insert_region_profile_counts(
            &mut profile,
            "pattern_completion",
            &pattern_completion_region_counts,
        );
        insert_region_profile_counts(
            &mut profile,
            "speech_boost",
            &speech_boost_region_counts,
        );
        insert_region_profile_counts(
            &mut profile,
            "binding_recall",
            &binding_recall_region_counts,
        );
    }
    profile.insert(
        "evaluation_rust_ms".to_string(),
        snapshot_ms
            + batch_state_ms
            + trace_match_ms
            + trace_side_effects_ms
            + binding_recall_ms,
    );

    Ok((
        activations,
        trace_evaluation
            .active_traces
            .into_iter()
            .map(|(trace_id, score)| (trace_id, score as f64))
            .collect(),
        state,
        active_counts,
        total_active,
        trace_evaluation
            .working_memory
            .into_iter()
            .map(|(trace_id, score)| (trace_id, score as f64))
            .collect(),
        binding_recall_candidates,
        profile,
    ))
}

#[pyfunction]
fn evaluate_tick_compact(
    store_id: u64,
    min_activation: f32,
    threshold: f64,
    tick_num: u64,
    working_memory_decay: f64,
    working_memory_capacity: usize,
    working_memory_overlay_cap: usize,
    trace_active_neuron_budget: usize,
    trace_freshness_retention: f64,
    trace_freshness_floor: f64,
    trace_freshness_min_score: f64,
    trace_refresh_max_boost: f64,
    trace_age_decay_window: u64,
    trace_age_floor_ceiling: f64,
    binding_recall_min_relative_weight: f64,
    binding_recall_boost_scale: f64,
    speech_boost_multiplier: f64,
) -> PyResult<(
    Vec<(u32, f32)>,
    Vec<(String, f64)>,
    NamedState,
    NamedActiveCounts,
    u32,
    Vec<(String, f64)>,
    BindingRecallCandidates,
    NamedProfile,
)> {
    evaluate_tick_compact_impl(
        store_id,
        min_activation,
        threshold,
        tick_num,
        working_memory_decay,
        working_memory_capacity,
        working_memory_overlay_cap,
        trace_active_neuron_budget,
        trace_freshness_retention,
        trace_freshness_floor,
        trace_freshness_min_score,
        trace_refresh_max_boost,
        trace_age_decay_window,
        trace_age_floor_ceiling,
        binding_recall_min_relative_weight,
        binding_recall_boost_scale,
        speech_boost_multiplier,
        true,
    )
}

#[pyfunction]
fn evaluate_tick_compact_minimal(
    store_id: u64,
    min_activation: f32,
    threshold: f64,
    tick_num: u64,
    working_memory_decay: f64,
    working_memory_capacity: usize,
    working_memory_overlay_cap: usize,
    trace_active_neuron_budget: usize,
    trace_freshness_retention: f64,
    trace_freshness_floor: f64,
    trace_freshness_min_score: f64,
    trace_refresh_max_boost: f64,
    trace_age_decay_window: u64,
    trace_age_floor_ceiling: f64,
    binding_recall_min_relative_weight: f64,
    binding_recall_boost_scale: f64,
    speech_boost_multiplier: f64,
) -> PyResult<(
    Vec<u32>,           // active neuron IDs (flat)
    Vec<f32>,           // active neuron activations (flat, parallel to IDs)
    Vec<(String, f64)>, // active traces
    CompactState,       // brain state as flat Vec<f64>
    NamedActiveCounts,  // per-region active counts
    u32,                // total active
    Vec<(String, f64)>, // working memory slots
    NamedProfile,       // evaluation profile
)> {
    let (
        activations,
        active_ids,
        active_counts,
        total_active,
        snapshot_ms,
        state,
        batch_state_ms,
    ) = with_brain_ref(|brain| {
        let snapshot_started = Instant::now();
        let (activations, active_ids, active_counts, total_active) =
            collect_flat_activations(brain, min_activation);
        let snapshot_ms = snapshot_started.elapsed().as_secs_f64() * 1000.0;

        let batch_state_started = Instant::now();
        let state = read_state_compact_from_brain(brain);
        let batch_state_ms = batch_state_started.elapsed().as_secs_f64() * 1000.0;

        (
            activations,
            active_ids,
            active_counts,
            total_active,
            snapshot_ms,
            state,
            batch_state_ms,
        )
    })?;

    let language_activation = state
        .get(COMPACT_STATE_LANGUAGE_ACTIVATION_IDX)
        .copied()
        .unwrap_or(0.0) as f32;
    let emotion_polarity = state
        .get(COMPACT_STATE_EMOTION_POLARITY_IDX)
        .copied()
        .unwrap_or(0.0) as f32;

    let trace_match_started = Instant::now();
    let trace_evaluation = with_trace_matchers(|registry| {
        registry.evaluate_active_traces(
            store_id,
            &active_ids,
            threshold as f32,
            tick_num,
            emotion_polarity,
            language_activation,
            trace_active_neuron_budget,
            trace_freshness_retention as f32,
            trace_freshness_floor as f32,
            trace_freshness_min_score as f32,
            trace_refresh_max_boost as f32,
            trace_age_decay_window,
            trace_age_floor_ceiling as f32,
            working_memory_decay as f32,
            working_memory_capacity,
            working_memory_overlay_cap,
            speech_boost_multiplier as f32,
        )
    })?;
    let trace_match_ms = trace_match_started.elapsed().as_secs_f64() * 1000.0;

    let trace_side_effects_started = Instant::now();
    with_brain(|brain| {
        if !trace_evaluation.working_memory_neurons.is_empty() {
            brain.boost_working_memory(&trace_evaluation.working_memory_neurons, 0.15);
        }
        for neurons in &trace_evaluation.memory_long_patterns {
            if !neurons.is_empty() {
                brain.pattern_complete(neurons, 0.4, 0.3);
            }
        }
        for (neurons, boost) in &trace_evaluation.speech_boosts {
            if !neurons.is_empty() && *boost > 0.0 {
                brain.boost_speech(neurons, *boost);
            }
        }
    })?;
    let trace_side_effects_ms = trace_side_effects_started.elapsed().as_secs_f64() * 1000.0;

    let binding_recall_started = Instant::now();
    let (binding_recall_bindings, binding_recall_neurons, _binding_recall_region_counts, binding_recall_max_relative_weight, binding_recall_max_boost) =
        with_brain(|brain| {
            brain.binding_recall(
                min_activation,
                binding_recall_min_relative_weight as f32,
                binding_recall_boost_scale as f32,
            )
        })?;
    let binding_recall_ms = binding_recall_started.elapsed().as_secs_f64() * 1000.0;

    // Push activation snapshot into Rust-side cache for zero-FFI learning
    with_brain(|brain| {
        let nn = brain.synapse_pool.num_neurons() as usize;
        let mut dense = vec![0.0f32; nn];
        for &(id, val) in &activations {
            if (id as usize) < nn {
                dense[id as usize] = val;
            }
        }
        if brain.activation_snapshot_cache.len() >= brain.activation_snapshot_window {
            brain.activation_snapshot_cache.remove(0);
        }
        brain.activation_snapshot_cache.push(dense);
    })?;

    // Split activations into parallel flat arrays for efficient PyO3 conversion
    let (act_ids, act_vals): (Vec<u32>, Vec<f32>) = activations.into_iter().unzip();

    let mut profile = HashMap::new();
    profile.insert("snapshot_ms".to_string(), snapshot_ms);
    profile.insert("batch_state_ms".to_string(), batch_state_ms);
    profile.insert("trace_match_ms".to_string(), trace_match_ms);
    profile.insert(
        "trace_candidates".to_string(),
        trace_evaluation.candidate_traces as f64,
    );
    profile.insert("trace_side_effects_ms".to_string(), trace_side_effects_ms);
    profile.insert("binding_recall_ms".to_string(), binding_recall_ms);
    profile.insert(
        "binding_recall_bindings".to_string(),
        binding_recall_bindings as f64,
    );
    profile.insert(
        "binding_recall_neurons".to_string(),
        binding_recall_neurons as f64,
    );
    profile.insert(
        "binding_recall_max_relative_weight".to_string(),
        binding_recall_max_relative_weight as f64,
    );
    profile.insert(
        "binding_recall_max_boost".to_string(),
        binding_recall_max_boost as f64,
    );
    profile.insert(
        "evaluation_rust_ms".to_string(),
        snapshot_ms + batch_state_ms + trace_match_ms + trace_side_effects_ms + binding_recall_ms,
    );

    Ok((
        act_ids,
        act_vals,
        trace_evaluation
            .active_traces
            .into_iter()
            .map(|(trace_id, score)| (trace_id, score as f64))
            .collect(),
        state,
        active_counts,
        total_active,
        trace_evaluation
            .working_memory
            .into_iter()
            .map(|(trace_id, score)| (trace_id, score as f64))
            .collect(),
        profile,
    ))
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
    with_brain_ref(|brain| brain.active_count(region_id))
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
fn attach_shared_weight_buffer(shm_name: String, size: usize) -> PyResult<()> {
    with_brain(|brain| {
        brain
            .synapse_pool
            .attach_shared_weight_buffer(&shm_name, size)
            .map_err(PyValueError::new_err)
    })?
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
fn apply_synapse_updates_profiled() -> PyResult<NamedProfile> {
    with_brain(|brain| {
        let profile = brain.apply_synapse_updates_profiled();
        build_apply_weight_updates_profile_map(&profile)
    })
}

#[pyfunction]
fn apply_synapse_updates_profiled_bounded(max_updates: usize) -> PyResult<NamedProfile> {
    with_brain(|brain| {
        let profile = brain.apply_synapse_updates_profiled_bounded(max_updates);
        build_apply_weight_updates_profile_map(&profile)
    })
}

#[pyfunction]
fn rebuild_synapse_index() -> PyResult<()> {
    with_brain(|brain| brain.rebuild_synapses())
}

#[pyfunction]
fn decay_synapse_weights(factor: f32, min_weight: f32) -> PyResult<u64> {
    with_brain(|brain| Ok(brain.decay_synapse_weights(factor, min_weight)))?
}

#[pyfunction]
fn get_neuron_count() -> PyResult<u32> {
    with_brain_ref(|brain| brain.neuron_count())
}

#[pyfunction]
fn get_synapse_count() -> PyResult<u64> {
    with_brain_ref(|brain| brain.synapse_count())
}

#[pyfunction]
fn get_synapse_topology_signature() -> PyResult<u64> {
    with_brain_ref(|brain| Ok(synapse_topology_signature_for(brain)))?
}

#[pyfunction]
fn write_synapse_weights_to_buffer(py: Python<'_>, buffer: &Bound<'_, PyAny>) -> PyResult<()> {
    let weight_buffer = PyBuffer::<f32>::get_bound(buffer)?;
    let writable = weight_buffer.as_mut_slice(py).ok_or_else(|| {
        PyValueError::new_err(
            "Expected a writable, C-contiguous float32 buffer for synapse weights",
        )
    })?;

    with_brain_ref(|brain| {
        let mut dense_weights = vec![0.0f32; writable.len()];
        brain
            .synapse_pool
            .copy_weights_to_slice(&mut dense_weights)
            .map_err(PyValueError::new_err)?;

        for (target, weight) in writable.iter().zip(dense_weights.iter()) {
            target.set(*weight);
        }
        Ok(())
    })?
}

#[pyfunction]
fn read_synapse_weights_from_buffer(py: Python<'_>, buffer: &Bound<'_, PyAny>) -> PyResult<()> {
    let weight_buffer = PyBuffer::<f32>::get_bound(buffer)?;
    let readable = weight_buffer.as_slice(py).ok_or_else(|| {
        PyValueError::new_err(
            "Expected a C-contiguous float32 buffer for synapse weights",
        )
    })?;

    with_brain(|brain| {
        let source_weights = readable.iter().map(|weight| weight.get()).collect::<Vec<_>>();
        brain
            .synapse_pool
            .overwrite_weights_from_slice(&source_weights)
            .map_err(PyValueError::new_err)?;
        Ok(())
    })?
}

#[pyfunction]
fn reset_round_synapse_deltas() -> PyResult<()> {
    with_brain(|brain| brain.synapse_pool.reset_round_delta_tracker())
}

#[pyfunction]
fn take_round_synapse_deltas() -> PyResult<Vec<(u32, f64)>> {
    with_brain(|brain| {
        brain
            .synapse_pool
            .take_round_deltas()
            .into_iter()
            .map(|(synapse_index, delta)| (synapse_index, delta as f64))
            .collect::<Vec<_>>()
    })
}

#[pyfunction]
fn apply_synapse_deltas_by_index(deltas: Vec<(u32, f64)>) -> PyResult<usize> {
    with_brain(|brain| {
        let sparse_deltas = deltas
            .into_iter()
            .map(|(synapse_index, delta)| (synapse_index, delta as f32))
            .collect::<Vec<_>>();
        brain.synapse_pool.apply_sparse_deltas_by_index(&sparse_deltas)
    })
}

#[pyfunction]
fn get_pending_synapse_update_count() -> PyResult<usize> {
    with_brain_ref(|brain| brain.pending_synapse_update_count())
}

#[pyfunction]
fn get_synapse_chunk_stats() -> PyResult<HashMap<String, f64>> {
    with_brain_ref(|brain| {
        let (chunk_count, chunk_size, local_count, cross_count, cross_fraction) =
            brain.synapse_pool.chunk_stats();
        let mut stats = HashMap::new();
        stats.insert("chunk_count".to_string(), chunk_count as f64);
        stats.insert("chunk_size".to_string(), chunk_size as f64);
        stats.insert("local_synapses".to_string(), local_count as f64);
        stats.insert("cross_synapses".to_string(), cross_count as f64);
        stats.insert("cross_fraction".to_string(), cross_fraction);
        stats.insert("local_fraction".to_string(), 1.0 - cross_fraction);
        stats
    })
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
                brain.synapse_pool.weight_at_index(i).unwrap_or(0.0),
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

/// Evaluate and update binding activity in one Rust pass.
/// Returns (strengthened_count, missed_count, total_bindings).
#[pyfunction]
fn process_bindings(min_activation: f32, tick: u64) -> PyResult<(u32, u32, usize)> {
    with_brain(|brain| {
        let (strengthened, missed) = brain.process_bindings(min_activation, tick);
        (strengthened, missed, brain.binding_count())
    })
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

#[pyfunction]
fn export_bindings() -> PyResult<Vec<BindingSnapshot>> {
    with_brain_ref(|brain| {
        brain
            .binding_store
            .iter()
            .map(binding_snapshot)
            .collect()
    })
}

#[pyfunction]
fn replace_bindings(bindings: Vec<BindingSnapshot>) -> PyResult<usize> {
    let mut decoded = Vec::with_capacity(bindings.len());
    for binding in bindings {
        decoded.push(binding_from_snapshot(binding)?);
    }

    with_brain(|brain| {
        brain.binding_store = BindingStore::from_bindings(decoded);
        brain.binding_count()
    })
}

/// Get binding pattern activation ratios: (pattern_a_ratio, pattern_b_ratio).
#[pyfunction]
fn get_binding_activation(binding_id: u32, min_activation: f32) -> PyResult<Option<(f32, f32)>> {
    with_brain_ref(|brain| brain.get_binding_activation(binding_id, min_activation))
}

/// Get binding pattern activation ratios for multiple bindings in one pass.
#[pyfunction]
fn get_binding_activations(
    binding_ids: Vec<u32>,
    min_activation: f32,
) -> PyResult<BindingActivations> {
    with_brain_ref(|brain| {
        brain
            .get_binding_activations(&binding_ids, min_activation)
            .into_iter()
            .map(|(binding_id, ratio_a, ratio_b)| {
                (binding_id, ratio_a as f64, ratio_b as f64)
            })
            .collect()
    })
}

/// Get pre-injection binding recall candidates for the current state.
/// Returns (binding_id, relative_weight, source_activation_ratio).
#[pyfunction]
fn get_binding_recall_candidates(
    min_activation: f32,
    min_relative_weight: f32,
) -> PyResult<BindingRecallCandidates> {
    with_brain_ref(|brain| {
        brain
            .get_binding_recall_candidates(min_activation, min_relative_weight)
            .into_iter()
            .map(|(binding_id, relative_weight, source_activation_ratio)| {
                (
                    binding_id,
                    relative_weight as f64,
                    source_activation_ratio as f64,
                )
            })
            .collect()
    })
}

#[pyfunction]
fn annotate_binding_traces(binding_id: u32, trace_id_a: String, trace_id_b: String) -> PyResult<()> {
    with_brain(|brain| brain.annotate_binding_traces(binding_id, trace_id_a, trace_id_b))
}

#[pyfunction]
fn complete_binding_recall(
    store_id: u64,
    active_neurons: Vec<u32>,
    active_traces: Vec<(String, f64)>,
    min_relative_weight: f32,
    trace_match_threshold: f32,
    trace_activation_threshold: f32,
    _pattern_completion_threshold: f32,
    pattern_completion_boost: f32,
) -> PyResult<(NamedTraceScores, NamedTraceScores, u32, u32, NamedProfile)> {
    let binding_input_started = Instant::now();
    let binding_inputs = with_brain_ref(|brain| {
        brain.get_binding_recall_trace_inputs_from_active(
            &active_neurons,
            min_relative_weight,
        )
    })?;
    let binding_input_ms = binding_input_started.elapsed().as_secs_f64() * 1000.0;

    let active_traces_f32 = active_traces
        .into_iter()
        .map(|(trace_id, score)| (trace_id, score as f32))
        .collect::<Vec<_>>();

    let trace_completion_started = Instant::now();
    let completion = with_trace_matchers(|registry| {
        registry.complete_binding_recall(
            store_id,
            &active_neurons,
            &active_traces_f32,
            &binding_inputs,
            trace_match_threshold,
            trace_activation_threshold,
            pattern_completion_boost,
        )
    })?;
    let trace_completion_ms = trace_completion_started.elapsed().as_secs_f64() * 1000.0;

    let pattern_apply_started = Instant::now();
    let mut completed_traces = 0u32;
    let mut completion_signal_by_neuron: HashMap<u32, f32> = HashMap::new();
    for (trace_neurons, boost) in &completion.pattern_completions {
        if trace_neurons.is_empty() || *boost <= 0.0 {
            continue;
        }
        completed_traces += 1;
        for &neuron_id in trace_neurons {
            completion_signal_by_neuron
                .entry(neuron_id)
                .and_modify(|existing| {
                    if *boost > *existing {
                        *existing = *boost;
                    }
                })
                .or_insert(*boost);
        }
    }
    let completed_neurons = completion_signal_by_neuron.len() as u32;
    if !completion_signal_by_neuron.is_empty() {
        let completion_signals = completion_signal_by_neuron
            .into_iter()
            .collect::<Vec<_>>();
        with_brain(|brain| brain.inject(&completion_signals))?;
    }
    let pattern_apply_ms = pattern_apply_started.elapsed().as_secs_f64() * 1000.0;

    let mut profile = HashMap::new();
    profile.insert(
        "binding_recall_completion_binding_input_ms".to_string(),
        binding_input_ms,
    );
    profile.insert(
        "binding_recall_completion_trace_completion_ms".to_string(),
        trace_completion_ms,
    );
    profile.insert(
        "binding_recall_completion_pattern_apply_ms".to_string(),
        pattern_apply_ms,
    );
    profile.insert(
        "binding_recall_completion_candidate_bindings".to_string(),
        completion.candidate_bindings as f64,
    );
    profile.insert(
        "binding_recall_completion_trace_checks".to_string(),
        completion.trace_checks as f64,
    );
    profile.insert(
        "binding_recall_completion_pattern_endpoint_checks".to_string(),
        completion.pattern_endpoint_checks as f64,
    );
    profile.insert(
        "binding_recall_completion_pattern_completion_candidates".to_string(),
        completion.pattern_completions.len() as f64,
    );
    profile.insert(
        "binding_recall_completion_reactivated_traces".to_string(),
        completion.reactivated_traces.len() as f64,
    );
    profile.insert(
        "binding_recall_completion_internal_ms".to_string(),
        binding_input_ms + trace_completion_ms + pattern_apply_ms,
    );

    Ok((
        completion
            .augmented_active_traces
            .into_iter()
            .map(|(trace_id, score)| (trace_id, score as f64))
            .collect(),
        completion
            .reactivated_traces
            .into_iter()
            .map(|(trace_id, score)| (trace_id, score as f64))
            .collect(),
        completed_traces,
        completed_neurons,
        profile,
    ))
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

/// Zero all activations in the speech region.
#[pyfunction]
fn zero_speech_activations() -> PyResult<()> {
    with_brain(|brain| brain.zero_speech_activations())
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
    m.add_function(wrap_pyfunction!(reset_runtime_state, m)?)?;
    m.add_function(wrap_pyfunction!(save_brain_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(dump_brain_checkpoint_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(save_brain_runtime_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(dump_brain_runtime_checkpoint_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(load_brain_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(load_brain_checkpoint_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(load_brain_runtime_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(load_brain_runtime_checkpoint_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(merge_brain_checkpoints, m)?)?;
    m.add_function(wrap_pyfunction!(set_same_region_delay_ablation, m)?)?;
    m.add_function(wrap_pyfunction!(clear_same_region_delay_ablation, m)?)?;
    m.add_function(wrap_pyfunction!(set_same_region_delay_learning_ablation, m)?)?;
    m.add_function(wrap_pyfunction!(clear_same_region_delay_learning_ablation, m)?)?;

    m.add_function(wrap_pyfunction!(tick, m)?)?;
    m.add_function(wrap_pyfunction!(tick_profiled, m)?)?;
    m.add_function(wrap_pyfunction!(tick_batch, m)?)?;
    m.add_function(wrap_pyfunction!(tick_batch_compact, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_tick, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_tick_compact, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_tick_compact_minimal, m)?)?;
    m.add_function(wrap_pyfunction!(get_tick_count, m)?)?;
    m.add_function(wrap_pyfunction!(trace_index_create, m)?)?;
    m.add_function(wrap_pyfunction!(trace_index_clear, m)?)?;
    m.add_function(wrap_pyfunction!(trace_index_clear_working_memory, m)?)?;
    m.add_function(wrap_pyfunction!(trace_index_set_working_memory, m)?)?;
    m.add_function(wrap_pyfunction!(trace_index_drop, m)?)?;
    m.add_function(wrap_pyfunction!(trace_index_upsert_trace, m)?)?;
    m.add_function(wrap_pyfunction!(trace_index_upsert_trace_full, m)?)?;
    m.add_function(wrap_pyfunction!(trace_index_remove_trace, m)?)?;
    m.add_function(wrap_pyfunction!(trace_index_matching_traces, m)?)?;
    m.add_function(wrap_pyfunction!(trace_index_runtime_snapshots, m)?)?;
    m.add_function(wrap_pyfunction!(trace_index_predict_regions, m)?)?;
    m.add_function(wrap_pyfunction!(trace_index_active_primary_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(novel_tracker_create, m)?)?;
    m.add_function(wrap_pyfunction!(novel_tracker_clear, m)?)?;
    m.add_function(wrap_pyfunction!(novel_tracker_drop, m)?)?;
    m.add_function(wrap_pyfunction!(novel_tracker_update, m)?)?;
    m.add_function(wrap_pyfunction!(novel_tracker_update_from_brain, m)?)?;
    m.add_function(wrap_pyfunction!(binding_tracker_create, m)?)?;
    m.add_function(wrap_pyfunction!(binding_tracker_clear, m)?)?;
    m.add_function(wrap_pyfunction!(binding_tracker_drop, m)?)?;
    m.add_function(wrap_pyfunction!(binding_tracker_consume, m)?)?;
    m.add_function(wrap_pyfunction!(binding_tracker_record, m)?)?;
    m.add_function(wrap_pyfunction!(binding_tracker_record_detailed, m)?)?;
    m.add_function(wrap_pyfunction!(binding_tracker_record_from_active_traces, m)?)?;
    m.add_function(wrap_pyfunction!(binding_tracker_record_detailed_from_active_traces, m)?)?;
    m.add_function(wrap_pyfunction!(binding_tracker_cleanup, m)?)?;
    m.add_function(wrap_pyfunction!(binding_tracker_mark_bound, m)?)?;

    m.add_function(wrap_pyfunction!(inject_activations, m)?)?;
    m.add_function(wrap_pyfunction!(set_attention_gain, m)?)?;

    m.add_function(wrap_pyfunction!(get_activations, m)?)?;
    m.add_function(wrap_pyfunction!(get_all_activations, m)?)?;
    m.add_function(wrap_pyfunction!(get_neuron_potential, m)?)?;
    m.add_function(wrap_pyfunction!(get_active_count, m)?)?;
    m.add_function(wrap_pyfunction!(get_region_firing_rate, m)?)?;

    m.add_function(wrap_pyfunction!(get_synapse_weight, m)?)?;
    m.add_function(wrap_pyfunction!(attach_shared_weight_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(update_synapse, m)?)?;
    m.add_function(wrap_pyfunction!(create_synapse, m)?)?;
    m.add_function(wrap_pyfunction!(prune_synapse, m)?)?;
    m.add_function(wrap_pyfunction!(apply_synapse_updates, m)?)?;
    m.add_function(wrap_pyfunction!(apply_synapse_updates_profiled, m)?)?;
    m.add_function(wrap_pyfunction!(apply_synapse_updates_profiled_bounded, m)?)?;
    m.add_function(wrap_pyfunction!(rebuild_synapse_index, m)?)?;
    m.add_function(wrap_pyfunction!(decay_synapse_weights, m)?)?;
    m.add_function(wrap_pyfunction!(get_neuron_count, m)?)?;
    m.add_function(wrap_pyfunction!(get_synapse_count, m)?)?;
    m.add_function(wrap_pyfunction!(get_synapse_topology_signature, m)?)?;
    m.add_function(wrap_pyfunction!(write_synapse_weights_to_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(read_synapse_weights_from_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(reset_round_synapse_deltas, m)?)?;
    m.add_function(wrap_pyfunction!(take_round_synapse_deltas, m)?)?;
    m.add_function(wrap_pyfunction!(apply_synapse_deltas_by_index, m)?)?;
    m.add_function(wrap_pyfunction!(get_pending_synapse_update_count, m)?)?;
    m.add_function(wrap_pyfunction!(get_synapse_chunk_stats, m)?)?;
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
    m.add_function(wrap_pyfunction!(process_bindings, m)?)?;
    m.add_function(wrap_pyfunction!(strengthen_binding, m)?)?;
    m.add_function(wrap_pyfunction!(record_binding_miss, m)?)?;
    m.add_function(wrap_pyfunction!(prune_bindings, m)?)?;
    m.add_function(wrap_pyfunction!(get_binding_count, m)?)?;
    m.add_function(wrap_pyfunction!(get_binding_info, m)?)?;
    m.add_function(wrap_pyfunction!(export_bindings, m)?)?;
    m.add_function(wrap_pyfunction!(replace_bindings, m)?)?;
    m.add_function(wrap_pyfunction!(get_binding_activation, m)?)?;
    m.add_function(wrap_pyfunction!(get_binding_activations, m)?)?;
    m.add_function(wrap_pyfunction!(get_binding_recall_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(annotate_binding_traces, m)?)?;
    m.add_function(wrap_pyfunction!(complete_binding_recall, m)?)?;
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
    m.add_function(wrap_pyfunction!(zero_speech_activations, m)?)?;

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
    m.add_function(wrap_pyfunction!(batch_learn_step_configurable, m)?)?;
    m.add_function(wrap_pyfunction!(batch_learn_step_flat, m)?)?;
    m.add_function(wrap_pyfunction!(batch_learn_step_from_snapshots, m)?)?;
    m.add_function(wrap_pyfunction!(push_activation_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(learn_from_snapshot_cache, m)?)?;



    Ok(())
}
