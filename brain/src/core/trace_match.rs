use crate::core::binding::BindingRecallTraceInput;
use crate::core::region::RegionId;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

const BINDING_RECALL_PARALLEL_MIN_INPUTS: usize = 128;
const SALIENT_REGION_MIN_RATIO_NUMERATOR: u16 = 2;
const SALIENT_REGION_MIN_RATIO_DENOMINATOR: u16 = 3;

#[derive(Clone, Debug)]
pub struct TraceRuntimeSnapshot {
    pub id: String,
    pub strength: f32,
    pub decay: f32,
    pub novelty: f32,
    pub polarity: f32,
    pub fire_count: u32,
    pub last_fired: u64,
}

#[derive(Clone, Debug, Default)]
pub struct TraceEvaluation {
    pub candidate_traces: usize,
    pub active_traces: Vec<(String, f32)>,
    pub working_memory: Vec<(String, f32)>,
    pub working_memory_neurons: Vec<u32>,
    pub memory_long_patterns: Vec<Vec<u32>>,
    pub speech_boosts: Vec<(Vec<u32>, f32)>,
}

#[derive(Clone, Debug, Default)]
pub struct BindingRecallCompletion {
    pub augmented_active_traces: Vec<(String, f32)>,
    pub reactivated_traces: Vec<(String, f32)>,
    pub pattern_completions: Vec<(Vec<u32>, f32)>,
    pub candidate_bindings: usize,
    pub trace_checks: usize,
    pub pattern_endpoint_checks: usize,
}

#[derive(Clone, Debug)]
struct TraceRuntime {
    strength: f32,
    decay: f32,
    novelty: f32,
    polarity: f32,
    fire_count: u32,
    last_fired: u64,
}

impl TraceRuntime {
    fn freshness(&self, tick: u64, retention: f32, floor: f32) -> f32 {
        let retention = retention.clamp(0.0, 1.0);
        let floor = floor.clamp(0.0, 1.0);
        let current = self.decay.clamp(0.0, 1.0);

        if tick <= self.last_fired {
            return current.max(floor);
        }

        let inactive_ticks = tick.saturating_sub(self.last_fired) as i32;
        let decayed = current * retention.powi(inactive_ticks);
        decayed.clamp(floor, 1.0)
    }

    fn freshness_weight(&self, tick: u64, retention: f32, floor: f32) -> f32 {
        let floor = floor.clamp(0.0, 1.0);
        let freshness = self.freshness(tick, retention, floor);
        if floor >= 1.0 {
            return 0.0;
        }

        ((freshness - floor).max(0.0) / (1.0 - floor)).clamp(0.0, 1.0)
    }

    fn refresh_ceiling(&self, tick: u64, age_decay_window: u64, age_floor_ceiling: f32) -> f32 {
        let age_floor_ceiling = age_floor_ceiling.clamp(0.0, 1.0);
        if tick <= self.last_fired {
            return 1.0;
        }
        if age_decay_window == 0 {
            return age_floor_ceiling;
        }

        let inactive_ticks = tick.saturating_sub(self.last_fired) as f32;
        let age_decay_window = age_decay_window as f32;
        let ceiling = 1.0 - (inactive_ticks / (age_decay_window * 2.0));
        ceiling.clamp(age_floor_ceiling, 1.0)
    }

    fn refreshed_decay(
        &self,
        tick: u64,
        retention: f32,
        floor: f32,
        overlap_score: f32,
        activation_threshold: f32,
        refresh_max_boost: f32,
        age_decay_window: u64,
        age_floor_ceiling: f32,
    ) -> f32 {
        let floor = floor.clamp(0.0, 1.0);
        let current_freshness = self.freshness(tick, retention, floor);
        let refresh_ceiling = self.refresh_ceiling(tick, age_decay_window, age_floor_ceiling);
        let activation_threshold = activation_threshold.clamp(0.0, 1.0);
        let overlap_score = overlap_score.clamp(0.0, 1.0);
        let normalized_overlap = if activation_threshold >= 1.0 {
            if overlap_score >= 1.0 { 1.0 } else { 0.0 }
        } else {
            ((overlap_score - activation_threshold).max(0.0) / (1.0 - activation_threshold))
                .clamp(0.0, 1.0)
        };
        let effective_boost = 1.0 + (refresh_max_boost.max(1.0) - 1.0) * normalized_overlap;
        let boosted_freshness = current_freshness * effective_boost;

        boosted_freshness
            .max(current_freshness)
            .min(refresh_ceiling)
            .clamp(floor, 1.0)
    }

    fn snapshot(
        &self,
        id: &str,
        tick: u64,
        retention: f32,
        floor: f32,
    ) -> TraceRuntimeSnapshot {
        TraceRuntimeSnapshot {
            id: id.to_string(),
            strength: self.strength,
            decay: self.freshness(tick, retention, floor),
            novelty: self.novelty,
            polarity: self.polarity,
            fire_count: self.fire_count,
            last_fired: self.last_fired,
        }
    }
}

#[derive(Clone, Debug)]
struct WorkingMemoryEntry {
    id: String,
    strength: f32,
}

#[derive(Default)]
pub struct TraceMatcherRegistry {
    stores: HashMap<u64, TraceMatcher>,
}

impl TraceMatcherRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_store(&mut self, store_id: u64) {
        self.stores.insert(store_id, TraceMatcher::new());
    }

    pub fn clear_store(&mut self, store_id: u64) {
        self.stores
            .entry(store_id)
            .or_insert_with(TraceMatcher::new)
            .clear();
    }

    pub fn clear_working_memory(&mut self, store_id: u64) {
        self.stores
            .entry(store_id)
            .or_insert_with(TraceMatcher::new)
            .clear_working_memory();
    }

    pub fn set_working_memory(&mut self, store_id: u64, slots: Vec<(String, f32)>) {
        self.stores
            .entry(store_id)
            .or_insert_with(TraceMatcher::new)
            .set_working_memory(slots);
    }

    pub fn drop_store(&mut self, store_id: u64) {
        self.stores.remove(&store_id);
    }

    pub fn upsert_trace(
        &mut self,
        store_id: u64,
        trace_id: String,
        neurons: Vec<u32>,
        memory_short_neurons: Vec<u32>,
        memory_long_neurons: Vec<u32>,
        speech_neurons: Vec<u32>,
        co_trace_ids: Vec<String>,
        strength: f32,
        decay: f32,
        novelty: f32,
        polarity: f32,
        fire_count: u32,
        last_fired: u64,
    ) {
        self.stores
            .entry(store_id)
            .or_insert_with(TraceMatcher::new)
            .upsert_trace(
                trace_id,
                neurons,
                memory_short_neurons,
                memory_long_neurons,
                speech_neurons,
                co_trace_ids,
                strength,
                decay,
                novelty,
                polarity,
                fire_count,
                last_fired,
            );
    }

    pub fn remove_trace(&mut self, store_id: u64, trace_id: &str) {
        if let Some(store) = self.stores.get_mut(&store_id) {
            store.remove_trace(trace_id);
        }
    }

    pub fn matching_traces(
        &self,
        store_id: u64,
        active_neurons: &[u32],
        threshold: f32,
    ) -> Vec<(String, f32)> {
        self.stores
            .get(&store_id)
            .map(|store| store.matching_traces(active_neurons, threshold))
            .unwrap_or_default()
    }

    pub fn evaluate_active_traces(
        &mut self,
        store_id: u64,
        active_neurons: &[u32],
        threshold: f32,
        tick: u64,
        emotion_polarity: f32,
        language_activation: f32,
        trace_active_neuron_budget: usize,
        trace_freshness_retention: f32,
        trace_freshness_floor: f32,
        trace_freshness_min_score: f32,
        trace_refresh_max_boost: f32,
        trace_age_decay_window: u64,
        trace_age_floor_ceiling: f32,
        working_memory_decay: f32,
        working_memory_capacity: usize,
        working_memory_overlay_cap: usize,
    ) -> TraceEvaluation {
        self.stores
            .entry(store_id)
            .or_insert_with(TraceMatcher::new)
            .evaluate_active_traces(
                active_neurons,
                threshold,
                tick,
                emotion_polarity,
                language_activation,
                trace_active_neuron_budget,
                trace_freshness_retention,
                trace_freshness_floor,
                trace_freshness_min_score,
                trace_refresh_max_boost,
                trace_age_decay_window,
                trace_age_floor_ceiling,
                working_memory_decay,
                working_memory_capacity,
                working_memory_overlay_cap,
            )
    }

    pub fn runtime_snapshots(
        &self,
        store_id: u64,
        trace_ids: Option<&[String]>,
    ) -> Vec<TraceRuntimeSnapshot> {
        self.stores
            .get(&store_id)
            .map(|store| store.runtime_snapshots(trace_ids))
            .unwrap_or_default()
    }

    pub fn predict_region_activity(
        &self,
        store_id: u64,
        active_traces: &[(String, f32)],
        working_memory_ids: &[String],
    ) -> HashMap<String, f32> {
        self.stores
            .get(&store_id)
            .map(|store| store.predict_region_activity(active_traces, working_memory_ids))
            .unwrap_or_else(|| {
                RegionId::ALL
                    .iter()
                    .map(|region_id| (region_id.name().to_string(), 0.0))
                    .collect()
            })
    }

    pub fn active_primary_patterns(
        &self,
        store_id: u64,
        active_traces: &[(String, f32)],
    ) -> Vec<(String, String)> {
        self.stores
            .get(&store_id)
            .map(|store| store.active_salient_patterns(active_traces))
            .unwrap_or_default()
    }

    pub fn complete_binding_recall(
        &self,
        store_id: u64,
        active_neurons: &[u32],
        active_traces: &[(String, f32)],
        binding_inputs: &[BindingRecallTraceInput],
        trace_match_threshold: f32,
        trace_activation_threshold: f32,
        pattern_completion_boost: f32,
    ) -> BindingRecallCompletion {
        self.stores
            .get(&store_id)
            .map(|store| {
                store.complete_binding_recall(
                    active_neurons,
                    active_traces,
                    binding_inputs,
                    trace_match_threshold,
                    trace_activation_threshold,
                    pattern_completion_boost,
                )
            })
            .unwrap_or_default()
    }
}

struct TraceEntry {
    id: String,
    neurons: Vec<u32>,
    total_neurons: u32,
    memory_short_neurons: Vec<u32>,
    memory_long_neurons: Vec<u32>,
    speech_neurons: Vec<u32>,
    region_counts: [u16; RegionId::ALL.len()],
    salient_region_mask: u16,
    co_trace_ids: Vec<String>,
    runtime: TraceRuntime,
}

struct TraceMatcher {
    traces: Vec<Option<TraceEntry>>,
    id_to_index: HashMap<String, usize>,
    neuron_to_traces: HashMap<u32, Vec<usize>>,
    free_indices: Vec<usize>,
    working_memory: Vec<WorkingMemoryEntry>,
    current_tick: u64,
    trace_freshness_retention: f32,
    trace_freshness_floor: f32,
    trace_freshness_min_score: f32,
}

fn completion_seed_neurons(
    entry: &TraceEntry,
    active_set: &HashSet<u32>,
    missing_for_trace_match: usize,
) -> Vec<u32> {
    if missing_for_trace_match == 0 {
        return Vec::new();
    }

    let mut seeds = entry
        .memory_long_neurons
        .iter()
        .copied()
        .filter(|neuron_id| !active_set.contains(neuron_id))
        .collect::<Vec<_>>();

    if seeds.len() < missing_for_trace_match {
        for &neuron_id in &entry.neurons {
            if seeds.len() >= missing_for_trace_match {
                break;
            }
            if active_set.contains(&neuron_id) || seeds.contains(&neuron_id) {
                continue;
            }
            seeds.push(neuron_id);
        }
    }

    seeds.truncate(missing_for_trace_match);
    seeds
}

impl TraceMatcher {
    fn new() -> Self {
        Self {
            traces: Vec::new(),
            id_to_index: HashMap::new(),
            neuron_to_traces: HashMap::new(),
            free_indices: Vec::new(),
            working_memory: Vec::new(),
            current_tick: 0,
            trace_freshness_retention: 1.0,
            trace_freshness_floor: 0.0,
            trace_freshness_min_score: 0.0,
        }
    }

    fn clear(&mut self) {
        self.traces.clear();
        self.id_to_index.clear();
        self.neuron_to_traces.clear();
        self.free_indices.clear();
        self.working_memory.clear();
        self.current_tick = 0;
        self.trace_freshness_retention = 1.0;
        self.trace_freshness_floor = 0.0;
        self.trace_freshness_min_score = 0.0;
    }

    fn clear_working_memory(&mut self) {
        self.working_memory.clear();
    }

    fn set_working_memory(&mut self, slots: Vec<(String, f32)>) {
        self.working_memory = slots
            .into_iter()
            .filter(|(trace_id, strength)| {
                *strength > 0.01 && self.id_to_index.contains_key(trace_id)
            })
            .map(|(id, strength)| WorkingMemoryEntry { id, strength })
            .collect();
        self.working_memory.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    fn upsert_trace(
        &mut self,
        trace_id: String,
        mut neurons: Vec<u32>,
        mut memory_short_neurons: Vec<u32>,
        mut memory_long_neurons: Vec<u32>,
        mut speech_neurons: Vec<u32>,
        co_trace_ids: Vec<String>,
        strength: f32,
        decay: f32,
        novelty: f32,
        polarity: f32,
        fire_count: u32,
        last_fired: u64,
    ) {
        neurons.sort_unstable();
        neurons.dedup();
        memory_short_neurons.sort_unstable();
        memory_short_neurons.dedup();
        memory_long_neurons.sort_unstable();
        memory_long_neurons.dedup();
        speech_neurons.sort_unstable();
        speech_neurons.dedup();

        if neurons.is_empty() {
            self.remove_trace(&trace_id);
            return;
        }

        self.remove_trace(&trace_id);

        let idx = self.free_indices.pop().unwrap_or_else(|| {
            self.traces.push(None);
            self.traces.len() - 1
        });

        let region_counts = compute_region_counts(&neurons);
        let entry = TraceEntry {
            id: trace_id.clone(),
            total_neurons: neurons.len() as u32,
            neurons: neurons.clone(),
            memory_short_neurons,
            memory_long_neurons,
            speech_neurons,
            salient_region_mask: salient_region_mask(&region_counts),
            region_counts,
            co_trace_ids,
            runtime: TraceRuntime {
                strength,
                decay,
                novelty,
                polarity,
                fire_count,
                last_fired,
            },
        };

        for neuron_id in &neurons {
            self.neuron_to_traces
                .entry(*neuron_id)
                .or_default()
                .push(idx);
        }

        self.id_to_index.insert(trace_id, idx);
        self.traces[idx] = Some(entry);
    }

    fn remove_trace(&mut self, trace_id: &str) {
        let Some(idx) = self.id_to_index.remove(trace_id) else {
            return;
        };

        let Some(entry) = self.traces.get_mut(idx).and_then(Option::take) else {
            return;
        };

        for neuron_id in entry.neurons {
            let should_remove_key = if let Some(indices) = self.neuron_to_traces.get_mut(&neuron_id)
            {
                indices.retain(|&existing_idx| existing_idx != idx);
                indices.is_empty()
            } else {
                false
            };

            if should_remove_key {
                self.neuron_to_traces.remove(&neuron_id);
            }
        }

        self.free_indices.push(idx);
        self.working_memory.retain(|entry| entry.id != trace_id);
    }

    fn matching_traces(&self, active_neurons: &[u32], threshold: f32) -> Vec<(String, f32)> {
        let (matches, _) = self.matching_trace_indices(
            active_neurons,
            threshold,
            self.current_tick,
            0,
            self.trace_freshness_retention,
            self.trace_freshness_floor,
            self.trace_freshness_min_score,
        );
        matches
            .into_iter()
            .filter_map(|(trace_idx, score, _)| {
                self.traces
                    .get(trace_idx)
                    .and_then(|entry| entry.as_ref())
                    .map(|entry| (entry.id.clone(), score))
            })
            .collect()
    }

    fn predict_region_activity(
        &self,
        active_traces: &[(String, f32)],
        working_memory_ids: &[String],
    ) -> HashMap<String, f32> {
        let mut region_predicted = [0.0f32; RegionId::ALL.len()];
        let mut seen_active = HashSet::new();

        for (trace_id, score) in active_traces {
            let Some(entry) = self.entry_for_id(trace_id) else {
                continue;
            };
            if let Some(trace_idx) = self.id_to_index.get(trace_id) {
                seen_active.insert(*trace_idx);
            }

            accumulate_region_counts(&mut region_predicted, &entry.region_counts, *score);

            for co_trace_id in &entry.co_trace_ids {
                let Some(co_idx) = self.id_to_index.get(co_trace_id) else {
                    continue;
                };
                if seen_active.contains(co_idx) {
                    continue;
                }
                if let Some(co_entry) = self.entry_for_id(co_trace_id) {
                    accumulate_region_counts(
                        &mut region_predicted,
                        &co_entry.region_counts,
                        *score * 0.3,
                    );
                }
            }
        }

        for wm_id in working_memory_ids {
            let Some(wm_idx) = self.id_to_index.get(wm_id) else {
                continue;
            };
            if seen_active.contains(wm_idx) {
                continue;
            }
            if let Some(entry) = self.entry_for_id(wm_id) {
                accumulate_region_counts(&mut region_predicted, &entry.region_counts, 0.2);
            }
        }

        RegionId::ALL
            .iter()
            .enumerate()
            .map(|(idx, region_id)| (region_id.name().to_string(), region_predicted[idx].min(1.0)))
            .collect()
    }

    fn active_salient_patterns(&self, active_traces: &[(String, f32)]) -> Vec<(String, String)> {
        let mut patterns = Vec::new();

        for (trace_id, _) in active_traces {
            let Some(entry) = self.entry_for_id(trace_id) else {
                continue;
            };

            for (idx, region_id) in RegionId::ALL.iter().enumerate() {
                if entry.salient_region_mask & (1u16 << idx) != 0 {
                    patterns.push((trace_id.clone(), region_id.name().to_string()));
                }
            }
        }

        patterns
    }

    fn complete_binding_recall(
        &self,
        active_neurons: &[u32],
        active_traces: &[(String, f32)],
        binding_inputs: &[BindingRecallTraceInput],
        trace_match_threshold: f32,
        trace_activation_threshold: f32,
        pattern_completion_boost: f32,
    ) -> BindingRecallCompletion {
        if binding_inputs.is_empty() {
            return BindingRecallCompletion {
                augmented_active_traces: active_traces.to_vec(),
                ..BindingRecallCompletion::default()
            };
        }

        let active_set: HashSet<u32> = active_neurons.iter().copied().collect();
        let mut active_trace_scores: HashMap<String, f32> = active_traces
            .iter()
            .map(|(trace_id, score)| (trace_id.clone(), *score))
            .collect();
        let use_parallel = rayon::current_num_threads() > 1
            && binding_inputs.len() >= BINDING_RECALL_PARALLEL_MIN_INPUTS;
        let initial_active_trace_ids: HashSet<String> = active_trace_scores.keys().cloned().collect();
        let trace_checks = binding_inputs.len().saturating_mul(2);

        let reactivation_candidates: Vec<(usize, String, f32)> = if use_parallel {
            binding_inputs
                .par_iter()
                .enumerate()
                .map(|(binding_idx, binding_input)| {
                    let mut local = Vec::with_capacity(2);
                    for (endpoint_idx, trace_id) in
                        [&binding_input.trace_id_a, &binding_input.trace_id_b]
                            .into_iter()
                            .enumerate()
                    {
                        if initial_active_trace_ids.contains(trace_id) {
                            continue;
                        }

                        let Some(entry) = self.entry_for_id(trace_id) else {
                            continue;
                        };
                        if entry.total_neurons == 0 {
                            continue;
                        }

                        let active_count = entry
                            .neurons
                            .iter()
                            .filter(|neuron_id| active_set.contains(neuron_id))
                            .count();
                        let overlap_ratio = active_count as f32 / entry.total_neurons as f32;
                        if overlap_ratio < trace_match_threshold {
                            continue;
                        }

                        local.push((binding_idx * 2 + endpoint_idx, trace_id.clone(), overlap_ratio));
                    }
                    local
                })
                .reduce(Vec::new, |mut acc, mut local| {
                    acc.append(&mut local);
                    acc
                })
        } else {
            let mut local = Vec::new();
            for (binding_idx, binding_input) in binding_inputs.iter().enumerate() {
                for (endpoint_idx, trace_id) in
                    [&binding_input.trace_id_a, &binding_input.trace_id_b]
                        .into_iter()
                        .enumerate()
                {
                    if initial_active_trace_ids.contains(trace_id) {
                        continue;
                    }

                    let Some(entry) = self.entry_for_id(trace_id) else {
                        continue;
                    };
                    if entry.total_neurons == 0 {
                        continue;
                    }

                    let active_count = entry
                        .neurons
                        .iter()
                        .filter(|neuron_id| active_set.contains(neuron_id))
                        .count();
                    let overlap_ratio = active_count as f32 / entry.total_neurons as f32;
                    if overlap_ratio < trace_match_threshold {
                        continue;
                    }

                    local.push((binding_idx * 2 + endpoint_idx, trace_id.clone(), overlap_ratio));
                }
            }
            local
        };

        let mut reactivated_by_trace: HashMap<String, (usize, f32)> = HashMap::new();
        for (order, trace_id, score) in reactivation_candidates {
            match reactivated_by_trace.get_mut(&trace_id) {
                Some((existing_order, existing_score)) => {
                    if order < *existing_order {
                        *existing_order = order;
                        *existing_score = score;
                    }
                }
                None => {
                    reactivated_by_trace.insert(trace_id, (order, score));
                }
            }
        }

        let mut reactivated_traces: Vec<(usize, String, f32)> = reactivated_by_trace
            .into_iter()
            .map(|(trace_id, (order, score))| (order, trace_id, score))
            .collect();
        reactivated_traces.sort_by_key(|(order, _, _)| *order);
        for (_, trace_id, score) in &reactivated_traces {
            active_trace_scores.insert(trace_id.clone(), *score);
        }
        let reactivated_traces = reactivated_traces
            .into_iter()
            .map(|(_, trace_id, score)| (trace_id, score))
            .collect::<Vec<_>>();

        let active_trace_ids: HashSet<String> = active_trace_scores.keys().cloned().collect();
        let pattern_endpoint_checks = binding_inputs.len().saturating_mul(2);
        let pattern_completion_candidates: Vec<(usize, String, Vec<u32>, f32)> = if use_parallel {
            binding_inputs
                .par_iter()
                .enumerate()
                .map(|(binding_idx, binding_input)| {
                    let mut local = Vec::with_capacity(2);
                    for (endpoint_idx, (trace_id, endpoint_ratio)) in [
                        (&binding_input.trace_id_a, binding_input.ratio_a),
                        (&binding_input.trace_id_b, binding_input.ratio_b),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        if endpoint_ratio < trace_activation_threshold {
                            continue;
                        }
                        if active_trace_ids.contains(trace_id) {
                            continue;
                        }

                        let Some(entry) = self.entry_for_id(trace_id) else {
                            continue;
                        };
                        if entry.total_neurons == 0 {
                            continue;
                        }

                        let required_active_count =
                            ((trace_match_threshold * entry.total_neurons as f32).ceil() as usize)
                                .max(1);
                        let active_count = entry
                            .neurons
                            .iter()
                            .filter(|neuron_id| active_set.contains(neuron_id))
                            .count();
                        let missing_for_trace_match =
                            required_active_count.saturating_sub(active_count);
                        if missing_for_trace_match == 0 {
                            continue;
                        }

                        let completion_seeds =
                            completion_seed_neurons(entry, &active_set, missing_for_trace_match);
                        if completion_seeds.is_empty() {
                            continue;
                        }

                        let boost = pattern_completion_boost
                            * binding_input.recall_weight.max(0.0)
                            * endpoint_ratio.max(0.0);
                        if boost <= 0.0 {
                            continue;
                        }

                        local.push((
                            binding_idx * 2 + endpoint_idx,
                            trace_id.clone(),
                            completion_seeds,
                            boost,
                        ));
                    }
                    local
                })
                .reduce(Vec::new, |mut acc, mut local| {
                    acc.append(&mut local);
                    acc
                })
        } else {
            let mut local = Vec::new();
            for (binding_idx, binding_input) in binding_inputs.iter().enumerate() {
                for (endpoint_idx, (trace_id, endpoint_ratio)) in [
                    (&binding_input.trace_id_a, binding_input.ratio_a),
                    (&binding_input.trace_id_b, binding_input.ratio_b),
                ]
                .into_iter()
                .enumerate()
                {
                    if endpoint_ratio < trace_activation_threshold {
                        continue;
                    }
                    if active_trace_ids.contains(trace_id) {
                        continue;
                    }

                    let Some(entry) = self.entry_for_id(trace_id) else {
                        continue;
                    };
                    if entry.total_neurons == 0 {
                        continue;
                    }

                    let required_active_count =
                        ((trace_match_threshold * entry.total_neurons as f32).ceil() as usize)
                            .max(1);
                    let active_count = entry
                        .neurons
                        .iter()
                        .filter(|neuron_id| active_set.contains(neuron_id))
                        .count();
                    let missing_for_trace_match = required_active_count.saturating_sub(active_count);
                    if missing_for_trace_match == 0 {
                        continue;
                    }

                    let completion_seeds =
                        completion_seed_neurons(entry, &active_set, missing_for_trace_match);
                    if completion_seeds.is_empty() {
                        continue;
                    }

                    let boost = pattern_completion_boost
                        * binding_input.recall_weight.max(0.0)
                        * endpoint_ratio.max(0.0);
                    if boost <= 0.0 {
                        continue;
                    }

                    local.push((
                        binding_idx * 2 + endpoint_idx,
                        trace_id.clone(),
                        completion_seeds,
                        boost,
                    ));
                }
            }
            local
        };

        let mut pattern_completion_by_trace: HashMap<String, (usize, Vec<u32>, f32)> =
            HashMap::new();
        for (order, trace_id, memory_long_neurons, boost) in pattern_completion_candidates {
            match pattern_completion_by_trace.get_mut(&trace_id) {
                Some((existing_order, existing_neurons, existing_boost)) => {
                    if order < *existing_order {
                        *existing_order = order;
                        *existing_neurons = memory_long_neurons;
                        *existing_boost = boost;
                    }
                }
                None => {
                    pattern_completion_by_trace
                        .insert(trace_id, (order, memory_long_neurons, boost));
                }
            }
        }

        let mut pattern_completion_ordered: Vec<(usize, Vec<u32>, f32)> =
            pattern_completion_by_trace
                .into_iter()
                .map(|(_, (order, memory_long_neurons, boost))| {
                    (order, memory_long_neurons, boost)
                })
                .collect();
        pattern_completion_ordered.sort_by_key(|(order, _, _)| *order);
        let pattern_completions = pattern_completion_ordered
            .into_iter()
            .map(|(_, memory_long_neurons, boost)| (memory_long_neurons, boost))
            .collect();

        let mut augmented_active_traces: Vec<(String, f32)> = active_trace_scores.into_iter().collect();
        augmented_active_traces.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        BindingRecallCompletion {
            augmented_active_traces,
            reactivated_traces,
            pattern_completions,
            candidate_bindings: binding_inputs.len(),
            trace_checks,
            pattern_endpoint_checks,
        }
    }

    fn evaluate_active_traces(
        &mut self,
        active_neurons: &[u32],
        threshold: f32,
        tick: u64,
        emotion_polarity: f32,
        language_activation: f32,
        trace_active_neuron_budget: usize,
        trace_freshness_retention: f32,
        trace_freshness_floor: f32,
        trace_freshness_min_score: f32,
        trace_refresh_max_boost: f32,
        trace_age_decay_window: u64,
        trace_age_floor_ceiling: f32,
        working_memory_decay: f32,
        working_memory_capacity: usize,
        working_memory_overlay_cap: usize,
    ) -> TraceEvaluation {
        self.current_tick = tick;
        self.trace_freshness_retention = trace_freshness_retention.clamp(0.0, 1.0);
        self.trace_freshness_floor = trace_freshness_floor.clamp(0.0, 1.0);
        self.trace_freshness_min_score = trace_freshness_min_score.clamp(0.0, 1.0);
        let (matches, candidate_traces) = self.matching_trace_indices(
            active_neurons,
            threshold,
            tick,
            trace_active_neuron_budget,
            self.trace_freshness_retention,
            self.trace_freshness_floor,
            self.trace_freshness_min_score,
        );
        let mut active_traces = Vec::with_capacity(matches.len());
        let mut memory_long_patterns = Vec::new();
        let mut speech_boosts = Vec::new();
        let should_update_polarity = emotion_polarity.abs() > 0.1;

        for (trace_idx, score, overlap_score) in matches {
            let Some(Some(entry)) = self.traces.get_mut(trace_idx) else {
                continue;
            };

            let refreshed_decay = entry.runtime.refreshed_decay(
                tick,
                self.trace_freshness_retention,
                self.trace_freshness_floor,
                overlap_score,
                threshold,
                trace_refresh_max_boost,
                trace_age_decay_window,
                trace_age_floor_ceiling,
            );

            entry.runtime.fire_count += 1;
            entry.runtime.last_fired = tick;
            entry.runtime.decay = refreshed_decay;
            entry.runtime.strength = (entry.runtime.strength + 0.005 * score).min(1.0);
            entry.runtime.novelty = (entry.runtime.novelty - 0.01).max(0.0);
            if should_update_polarity {
                entry.runtime.polarity = entry.runtime.polarity * 0.95 + emotion_polarity * 0.05;
            }

            if !entry.memory_long_neurons.is_empty() {
                memory_long_patterns.push(entry.memory_long_neurons.clone());
            }
            if language_activation > 0.1 && !entry.speech_neurons.is_empty() {
                speech_boosts.push((
                    entry.speech_neurons.clone(),
                    0.2 * score * language_activation,
                ));
            }
            active_traces.push((entry.id.clone(), score));
        }

        let working_memory = self.update_working_memory(
            &active_traces,
            working_memory_decay,
            working_memory_capacity,
            working_memory_overlay_cap,
        );
        let mut working_memory_neurons = Vec::new();
        for (trace_id, _) in &working_memory {
            if let Some(entry) = self.entry_for_id(trace_id) {
                working_memory_neurons.extend(entry.memory_short_neurons.iter().copied());
            }
        }

        TraceEvaluation {
            candidate_traces,
            active_traces,
            working_memory,
            working_memory_neurons,
            memory_long_patterns,
            speech_boosts,
        }
    }

    fn runtime_snapshots(&self, trace_ids: Option<&[String]>) -> Vec<TraceRuntimeSnapshot> {
        match trace_ids {
            Some(trace_ids) => trace_ids
                .iter()
                .filter_map(|trace_id| self.entry_for_id(trace_id))
                .map(|entry| {
                    entry.runtime.snapshot(
                        &entry.id,
                        self.current_tick,
                        self.trace_freshness_retention,
                        self.trace_freshness_floor,
                    )
                })
                .collect(),
            None => self
                .traces
                .iter()
                .filter_map(|entry| entry.as_ref())
                .map(|entry| {
                    entry.runtime.snapshot(
                        &entry.id,
                        self.current_tick,
                        self.trace_freshness_retention,
                        self.trace_freshness_floor,
                    )
                })
                .collect(),
        }
    }

    fn matching_trace_indices(
        &self,
        active_neurons: &[u32],
        threshold: f32,
        tick: u64,
        trace_active_neuron_budget: usize,
        trace_freshness_retention: f32,
        trace_freshness_floor: f32,
        trace_freshness_min_score: f32,
    ) -> (Vec<(usize, f32, f32)>, usize) {
        if active_neurons.is_empty() {
            return (Vec::new(), 0);
        }

        let mut counts: HashMap<usize, u32> = HashMap::new();
        for neuron_id in active_neurons {
            if let Some(indices) = self.neuron_to_traces.get(neuron_id) {
                for &trace_idx in indices {
                    *counts.entry(trace_idx).or_insert(0) += 1;
                }
            }
        }

        let mut matches = Vec::new();
        for (trace_idx, active_count) in counts {
            let Some(Some(entry)) = self.traces.get(trace_idx) else {
                continue;
            };
            if entry.total_neurons == 0 {
                continue;
            }

            let overlap_score = active_count as f32 / entry.total_neurons as f32;
            if overlap_score < threshold {
                continue;
            }

            let freshness_weight = entry.runtime.freshness_weight(
                tick,
                trace_freshness_retention,
                trace_freshness_floor,
            );
            let score = overlap_score * freshness_weight;
            if score >= trace_freshness_min_score {
                matches.push((trace_idx, score, overlap_score));
            }
        }

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let candidate_count = matches.len();
        if trace_active_neuron_budget == 0 || candidate_count <= 1 {
            return (matches, candidate_count);
        }

        let mut selected = Vec::with_capacity(matches.len());
        let mut used_neurons = 0usize;
        for (trace_idx, score, overlap_score) in matches {
            let Some(Some(entry)) = self.traces.get(trace_idx) else {
                continue;
            };
            let trace_neurons = entry.total_neurons.max(1) as usize;
            if !selected.is_empty()
                && used_neurons.saturating_add(trace_neurons) > trace_active_neuron_budget
            {
                break;
            }
            used_neurons = used_neurons.saturating_add(trace_neurons);
            selected.push((trace_idx, score, overlap_score));
        }

        (selected, candidate_count)
    }

    fn entry_for_id(&self, trace_id: &str) -> Option<&TraceEntry> {
        let idx = *self.id_to_index.get(trace_id)?;
        self.traces.get(idx).and_then(|entry| entry.as_ref())
    }

    fn update_working_memory(
        &mut self,
        active_traces: &[(String, f32)],
        decay_rate: f32,
        capacity: usize,
        overlay_cap: usize,
    ) -> Vec<(String, f32)> {
        let retained_decay = (1.0 - decay_rate).max(0.0);
        for slot in &mut self.working_memory {
            slot.strength *= retained_decay;
        }

        for (trace_id, score) in active_traces {
            if let Some(existing) = self
                .working_memory
                .iter_mut()
                .find(|entry| entry.id == *trace_id)
            {
                existing.strength = existing.strength.max(*score);
            } else {
                self.working_memory.push(WorkingMemoryEntry {
                    id: trace_id.clone(),
                    strength: *score,
                });
            }
        }

        self.working_memory.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if overlay_cap > 0 {
            let mut kept_overlay = 0usize;
            self.working_memory.retain(|entry| {
                if !entry.id.starts_with("overlay_") {
                    return true;
                }
                if kept_overlay < overlay_cap {
                    kept_overlay += 1;
                    true
                } else {
                    false
                }
            });
        }
        if self.working_memory.len() > capacity {
            self.working_memory.truncate(capacity);
        }
        self.working_memory.retain(|entry| entry.strength > 0.01);

        self.working_memory
            .iter()
            .map(|entry| (entry.id.clone(), entry.strength))
            .collect()
    }
}

fn compute_region_counts(neurons: &[u32]) -> [u16; RegionId::ALL.len()] {
    let mut counts = [0u16; RegionId::ALL.len()];
    for &neuron_id in neurons {
        if let Some((region_idx, _)) = region_index_local(neuron_id) {
            counts[region_idx] = counts[region_idx].saturating_add(1);
        }
    }
    counts
}

fn accumulate_region_counts(
    region_predicted: &mut [f32; RegionId::ALL.len()],
    region_counts: &[u16; RegionId::ALL.len()],
    weight: f32,
) {
    for (idx, &count) in region_counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let size = RegionId::ALL[idx].neuron_count() as f32;
        if size > 0.0 {
            region_predicted[idx] += (count as f32 / size) * weight;
        }
    }
}

fn salient_region_mask(region_counts: &[u16; RegionId::ALL.len()]) -> u16 {
    let mut best_count = 0u16;
    for &count in region_counts {
        if count > best_count {
            best_count = count;
        }
    }

    if best_count == 0 {
        return 0;
    }

    let mut mask = 0u16;
    for (idx, &count) in region_counts.iter().enumerate() {
        if count.saturating_mul(SALIENT_REGION_MIN_RATIO_DENOMINATOR)
            >= best_count.saturating_mul(SALIENT_REGION_MIN_RATIO_NUMERATOR)
        {
            mask |= 1u16 << idx;
        }
    }

    mask
}

fn region_index_local(global_id: u32) -> Option<(usize, u32)> {
    match global_id {
        0..=9_999 => Some((0, global_id)),
        10_000..=29_999 => Some((1, global_id - 10_000)),
        30_000..=44_999 => Some((2, global_id - 30_000)),
        45_000..=54_999 => Some((3, global_id - 45_000)),
        55_000..=69_999 => Some((4, global_id - 55_000)),
        70_000..=79_999 => Some((5, global_id - 70_000)),
        80_000..=84_999 => Some((6, global_id - 80_000)),
        85_000..=94_999 => Some((7, global_id - 85_000)),
        95_000..=104_999 => Some((8, global_id - 95_000)),
        105_000..=119_999 => Some((9, global_id - 105_000)),
        120_000..=129_999 => Some((10, global_id - 120_000)),
        130_000..=139_999 => Some((11, global_id - 130_000)),
        140_000..=149_999 => Some((12, global_id - 140_000)),
        150_000..=151_999 => Some((13, global_id - 150_000)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_evaluation_updates_runtime_and_working_memory() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "t1".to_string(),
            vec![0, 1, 2],
            vec![45_000, 45_001],
            vec![55_000, 55_001],
            vec![140_000, 140_001],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            7,
        );

        let result = registry.evaluate_active_traces(
            1,
            &[0, 1, 2],
            0.6,
            7,
            0.4,
            0.5,
            10,
            0.85,
            0.15,
            0.12,
            1.2,
            20,
            0.4,
            0.1,
            7,
            0,
        );
        assert_eq!(result.candidate_traces, 1);
        assert_eq!(result.active_traces, vec![("t1".to_string(), 1.0)]);
        assert_eq!(result.working_memory, vec![("t1".to_string(), 1.0)]);
        assert_eq!(result.working_memory_neurons, vec![45_000, 45_001]);
        assert_eq!(result.memory_long_patterns, vec![vec![55_000, 55_001]]);
        assert_eq!(result.speech_boosts, vec![(vec![140_000, 140_001], 0.1)]);

        let snapshots = registry.runtime_snapshots(1, Some(&["t1".to_string()]));
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].fire_count, 1);
        assert_eq!(snapshots[0].last_fired, 7);
        assert!((snapshots[0].decay - 1.0).abs() < 0.001);
        assert!(snapshots[0].strength > 0.2);
        assert!(snapshots[0].novelty < 1.0);
        assert!(snapshots[0].polarity > 0.0);
    }

    #[test]
    fn test_working_memory_decays_without_activity() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "t1".to_string(),
            vec![0, 1, 2],
            vec![45_000],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            1,
        );

        let first = registry.evaluate_active_traces(
            1,
            &[0, 1, 2],
            0.6,
            1,
            0.0,
            0.0,
            10,
            0.85,
            0.15,
            0.12,
            1.2,
            20,
            0.4,
            0.1,
            7,
            0,
        );
        assert_eq!(first.working_memory[0].1, 1.0);

        let second = registry.evaluate_active_traces(
            1,
            &[],
            0.6,
            2,
            0.0,
            0.0,
            10,
            0.85,
            0.15,
            0.12,
            1.2,
            20,
            0.4,
            0.1,
            7,
            0,
        );
        assert_eq!(second.working_memory.len(), 1);
        assert!(second.working_memory[0].1 < 1.0);

        let snapshots = registry.runtime_snapshots(1, Some(&["t1".to_string()]));
        assert_eq!(snapshots.len(), 1);
        assert!((snapshots[0].decay - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_floor_freshness_exact_match_is_filtered_out() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "fresh".to_string(),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            9,
        );
        registry.upsert_trace(
            1,
            "stale_partial".to_string(),
            vec![20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            0.15,
            1.0,
            0.0,
            0,
            0,
        );
        registry.upsert_trace(
            1,
            "stale_exact".to_string(),
            vec![40, 41, 42, 43, 44],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            0.15,
            1.0,
            0.0,
            0,
            0,
        );

        let result = registry.evaluate_active_traces(
            1,
            &[0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 25, 26, 40, 41, 42, 43, 44],
            0.6,
            10,
            0.0,
            0.0,
            100,
            0.85,
            0.15,
            0.12,
            1.2,
            20,
            0.4,
            0.1,
            7,
            0,
        );

        assert_eq!(result.candidate_traces, 1);
        assert_eq!(result.active_traces.len(), 1);
        assert_eq!(result.active_traces[0].0, "fresh");
        assert!(result.active_traces.iter().all(|(trace_id, _)| trace_id != "stale_partial"));
        assert!(result.active_traces.iter().all(|(trace_id, _)| trace_id != "stale_exact"));
    }

    #[test]
    fn test_exact_match_above_floor_still_survives_gate() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "survivor".to_string(),
            vec![0, 1, 2, 3, 4],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            0.45,
            1.0,
            0.0,
            0,
            8,
        );

        let result = registry.evaluate_active_traces(
            1,
            &[0, 1, 2, 3, 4],
            0.6,
            10,
            0.0,
            0.0,
            100,
            0.85,
            0.15,
            0.12,
            1.2,
            20,
            0.4,
            0.1,
            7,
            0,
        );

        assert_eq!(result.candidate_traces, 1);
        assert_eq!(result.active_traces.len(), 1);
        assert_eq!(result.active_traces[0].0, "survivor");
    }

    #[test]
    fn test_stale_trace_refresh_is_capped_by_age() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "stale".to_string(),
            vec![0, 1, 2],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            0.45,
            1.0,
            0.0,
            0,
            0,
        );

        let result = registry.evaluate_active_traces(
            1,
            &[0, 1, 2],
            0.6,
            20,
            0.0,
            0.0,
            10,
            1.0,
            0.15,
            0.12,
            1.2,
            20,
            0.4,
            0.1,
            7,
            0,
        );

        assert_eq!(result.candidate_traces, 1);
        let snapshots = registry.runtime_snapshots(1, Some(&["stale".to_string()]));
        assert_eq!(snapshots.len(), 1);
        assert!((snapshots[0].decay - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_partial_reactivation_cannot_hold_trace_near_full_freshness() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "partial".to_string(),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );

        for tick in [5_u64, 10, 15] {
            registry.evaluate_active_traces(
                1,
                &[0, 1, 2, 3, 4, 5, 6],
                0.6,
                tick,
                0.0,
                0.0,
                10,
                0.85,
                0.15,
                0.12,
                1.2,
                20,
                0.4,
                0.1,
                7,
                0,
            );
        }

        let snapshots = registry.runtime_snapshots(1, Some(&["partial".to_string()]));
        assert_eq!(snapshots.len(), 1);
        assert!(snapshots[0].decay < 0.25);
    }

    #[test]
    fn test_evaluate_active_traces_caps_matches_to_neuron_budget() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "t1".to_string(),
            vec![0, 1, 2],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            1,
        );
        registry.upsert_trace(
            1,
            "t2".to_string(),
            vec![10, 11, 12],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            1,
        );
        registry.upsert_trace(
            1,
            "t3".to_string(),
            vec![20, 21, 22],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );

        let result = registry.evaluate_active_traces(
            1,
            &[0, 1, 2, 10, 11, 12, 20, 21],
            0.6,
            1,
            0.0,
            0.0,
            6,
            0.85,
            0.15,
            0.12,
            1.2,
            20,
            0.4,
            0.1,
            7,
            0,
        );

        assert_eq!(result.candidate_traces, 3);
        assert_eq!(result.active_traces.len(), 2);
        let selected_ids: HashSet<String> = result
            .active_traces
            .iter()
            .map(|(trace_id, _)| trace_id.clone())
            .collect();
        assert_eq!(selected_ids, HashSet::from(["t1".to_string(), "t2".to_string()]));
    }

    #[test]
    fn test_working_memory_overlay_cap_keeps_highest_scoring_overlay_entries() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "overlay_a".to_string(),
            vec![0, 1, 2, 3],
            vec![45_000],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );
        registry.upsert_trace(
            1,
            "overlay_b".to_string(),
            vec![10, 11, 12, 13, 14, 15],
            vec![45_001],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );
        registry.upsert_trace(
            1,
            "overlay_c".to_string(),
            vec![20, 21, 22, 23, 24, 25, 26, 27],
            vec![45_002],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );
        registry.upsert_trace(
            1,
            "learned_main".to_string(),
            vec![30, 31, 32, 33, 34],
            vec![45_003],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );

        let result = registry.evaluate_active_traces(
            1,
            &[0, 1, 2, 3, 10, 11, 12, 13, 30, 31, 32, 33],
            0.5,
            1,
            0.0,
            0.0,
            100,
            0.85,
            0.15,
            0.12,
            1.2,
            20,
            0.4,
            0.1,
            7,
            2,
        );

        assert_eq!(result.working_memory.len(), 3);
        assert_eq!(result.working_memory[0].0, "overlay_a");
        assert!((result.working_memory[0].1 - 0.8235294).abs() < 0.0001);
        assert_eq!(result.working_memory[1].0, "learned_main");
        assert!((result.working_memory[1].1 - 0.65882355).abs() < 0.0001);
        assert_eq!(result.working_memory[2].0, "overlay_b");
        assert!((result.working_memory[2].1 - 0.54901963).abs() < 0.0001);
    }

    #[test]
    fn test_predict_region_activity_uses_trace_and_cotrace_regions() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "t1".to_string(),
            vec![0, 1, 10_000],
            vec![],
            vec![],
            vec![],
            vec!["t2".to_string()],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );
        registry.upsert_trace(
            1,
            "t2".to_string(),
            vec![30_000, 30_001],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );

        let predicted = registry.predict_region_activity(1, &[("t1".to_string(), 1.0)], &[]);
        assert!(predicted.get("sensory").copied().unwrap_or(0.0) > 0.0);
        assert!(predicted.get("visual").copied().unwrap_or(0.0) > 0.0);
        assert!(predicted.get("audio").copied().unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn test_active_primary_patterns_uses_largest_region() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "t1".to_string(),
            vec![0, 10_000, 10_001],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );

        let patterns = registry.active_primary_patterns(1, &[("t1".to_string(), 1.0)]);
        assert_eq!(patterns, vec![("t1".to_string(), "visual".to_string())]);
    }

    #[test]
    fn test_active_primary_patterns_include_all_tied_regions() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "t1".to_string(),
            vec![55_000, 55_001, 85_000, 85_001],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );

        let patterns = registry.active_primary_patterns(1, &[("t1".to_string(), 1.0)]);
        assert_eq!(
            patterns,
            vec![
                ("t1".to_string(), "memory_long".to_string()),
                ("t1".to_string(), "pattern".to_string()),
            ]
        );
    }

    #[test]
    fn test_active_primary_patterns_include_strong_secondary_regions() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "t1".to_string(),
            vec![10_000, 10_001, 10_002, 85_000, 85_001],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );

        let patterns = registry.active_primary_patterns(1, &[("t1".to_string(), 1.0)]);
        assert_eq!(
            patterns,
            vec![
                ("t1".to_string(), "visual".to_string()),
                ("t1".to_string(), "pattern".to_string()),
            ]
        );
    }

    #[test]
    fn test_complete_binding_recall_reactivates_overlap_and_skips_pattern_completion_for_active_trace() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "cue".to_string(),
            vec![85000, 85001],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );
        registry.upsert_trace(
            1,
            "partner".to_string(),
            vec![86000, 86001, 56000, 56001],
            vec![],
            vec![56000, 56001],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );

        let completion = registry.complete_binding_recall(
            1,
            &[86000, 86001, 56000, 56001],
            &[("cue".to_string(), 0.9)],
            &[BindingRecallTraceInput {
                binding_id: 7,
                trace_id_a: "cue".to_string(),
                trace_id_b: "partner".to_string(),
                ratio_a: 0.0,
                ratio_b: 0.75,
                recall_weight: 1.0,
            }],
            0.4,
            0.6,
            0.8,
        );

        assert!(completion
            .reactivated_traces
            .iter()
            .any(|(trace_id, score)| trace_id == "partner" && *score >= 0.5));
        assert!(completion
            .augmented_active_traces
            .iter()
            .any(|(trace_id, score)| trace_id == "partner" && *score >= 0.5));
        assert!(completion.pattern_completions.is_empty());
        assert_eq!(completion.candidate_bindings, 1);
        assert_eq!(completion.trace_checks, 2);
        assert_eq!(completion.pattern_endpoint_checks, 2);
    }

    #[test]
    fn test_complete_binding_recall_keeps_first_pattern_completion_for_duplicate_trace() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "shared".to_string(),
            vec![85_000, 85_001, 56_000, 56_001],
            vec![],
            vec![56_000, 56_001],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );

        let mut binding_inputs = Vec::new();
        binding_inputs.push(BindingRecallTraceInput {
            binding_id: 1,
            trace_id_a: "shared".to_string(),
            trace_id_b: "missing".to_string(),
            ratio_a: 0.8,
            ratio_b: 0.0,
            recall_weight: 0.25,
        });
        for binding_id in 2..=130 {
            binding_inputs.push(BindingRecallTraceInput {
                binding_id,
                trace_id_a: "shared".to_string(),
                trace_id_b: "missing".to_string(),
                ratio_a: 0.8,
                ratio_b: 0.0,
                recall_weight: 1.0,
            });
        }

        let completion = registry.complete_binding_recall(
            1,
            &[],
            &[],
            &binding_inputs,
            0.4,
            0.6,
            0.8,
        );

        assert_eq!(completion.pattern_completions.len(), 1);
        assert_eq!(completion.pattern_completions[0].0, vec![56_000, 56_001]);
        assert!((completion.pattern_completions[0].1 - 0.16).abs() < 1e-6);
        assert_eq!(completion.candidate_bindings, 130);
        assert_eq!(completion.trace_checks, 260);
        assert_eq!(completion.pattern_endpoint_checks, 260);
    }

    #[test]
    fn test_complete_binding_recall_seeds_missing_memory_long_to_cross_trace_threshold() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "partner".to_string(),
            vec![
                85_000, 85_001, 85_002, 85_003, 85_004, 85_005, 85_006, 85_007,
                56_000, 56_001, 56_002, 56_003, 56_004, 56_005,
                105_000, 105_001, 105_002, 105_003, 105_004, 105_005, 105_006, 105_007,
            ],
            vec![],
            vec![56_000, 56_001, 56_002, 56_003, 56_004, 56_005],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );

        let completion = registry.complete_binding_recall(
            1,
            &[85_000, 85_001, 85_002, 85_003, 85_004, 85_005, 85_006, 85_007],
            &[],
            &[BindingRecallTraceInput {
                binding_id: 9,
                trace_id_a: "cue".to_string(),
                trace_id_b: "partner".to_string(),
                ratio_a: 1.0,
                ratio_b: 1.0,
                recall_weight: 1.0,
            }],
            0.4,
            0.6,
            0.8,
        );

        assert!(completion.reactivated_traces.is_empty());
        assert_eq!(completion.pattern_completions.len(), 1);
        assert_eq!(completion.pattern_completions[0].0.len(), 1);
        assert_eq!(completion.pattern_completions[0].0[0], 56_000);
        assert!((completion.pattern_completions[0].1 - 0.8).abs() < 1e-6);
        assert_eq!(completion.candidate_bindings, 1);
        assert_eq!(completion.trace_checks, 2);
        assert_eq!(completion.pattern_endpoint_checks, 2);
    }

    #[test]
    fn test_complete_binding_recall_falls_back_to_other_trace_neurons_when_memory_long_is_absent() {
        let mut registry = TraceMatcherRegistry::new();
        registry.create_store(1);
        registry.upsert_trace(
            1,
            "partner".to_string(),
            vec![
                95_000, 95_001, 95_002, 95_003, 95_004, 95_005, 95_006, 95_007,
                105_000, 105_001, 105_002, 105_003, 105_004, 105_005, 105_006, 105_007,
                85_000, 85_001, 85_002, 85_003, 85_004, 85_005, 85_006, 85_007,
            ],
            vec![],
            vec![],
            vec![],
            vec![],
            0.2,
            1.0,
            1.0,
            0.0,
            0,
            0,
        );

        let completion = registry.complete_binding_recall(
            1,
            &[105_000, 105_001, 105_002, 105_003, 105_004, 105_005, 105_006, 105_007],
            &[],
            &[BindingRecallTraceInput {
                binding_id: 11,
                trace_id_a: "cue".to_string(),
                trace_id_b: "partner".to_string(),
                ratio_a: 1.0,
                ratio_b: 1.0,
                recall_weight: 1.0,
            }],
            0.4,
            0.6,
            0.8,
        );

        assert!(completion.reactivated_traces.is_empty());
        assert_eq!(completion.pattern_completions.len(), 1);
        assert_eq!(completion.pattern_completions[0].0.len(), 2);
        assert!(completion.pattern_completions[0]
            .0
            .iter()
            .all(|neuron_id| *neuron_id != 105_000
                && *neuron_id != 105_001
                && *neuron_id != 105_002
                && *neuron_id != 105_003
                && *neuron_id != 105_004
                && *neuron_id != 105_005
                && *neuron_id != 105_006
                && *neuron_id != 105_007));
        assert!((completion.pattern_completions[0].1 - 0.8).abs() < 1e-6);
        assert_eq!(completion.candidate_bindings, 1);
        assert_eq!(completion.trace_checks, 2);
        assert_eq!(completion.pattern_endpoint_checks, 2);
    }
}