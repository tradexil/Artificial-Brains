/// Binding: cross-region pattern links.
///
/// A Binding connects two PatternRefs in different regions.
/// When both patterns co-activate, the binding fires and strengthens.
/// Bindings are how the brain links "red" (visual) with "ball" (pattern)
/// into one unified concept.
///
/// Lifecycle: Formation → Strengthening → Weakening → Dissolution
///   - Formation: 5+ co-activations within ±5 tick window
///   - Strengthening: weight += 0.05 × (1 - weight)
///   - Weakening: weight *= 0.998 per missed opportunity
///   - Dissolution: weight < 0.05 AND fires < 10

use crate::core::region::RegionId;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const RECALL_COMPLETION_PARALLEL_MIN_BINDINGS: usize = 128;
const RECALL_COMPLETION_PARALLEL_CHUNK_SIZE: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PatternKey {
    region: RegionId,
    threshold_bits: u32,
    neurons: Vec<u32>,
}

impl From<&PatternRef> for PatternKey {
    fn from(pattern: &PatternRef) -> Self {
        Self {
            region: pattern.region,
            threshold_bits: pattern.threshold.to_bits(),
            neurons: pattern.neurons.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRef {
    pub region: RegionId,
    pub neurons: Vec<u32>,
    pub threshold: f32,
}

impl PatternRef {
    pub fn new(region: RegionId, neurons: Vec<u32>, threshold: f32) -> Self {
        Self { region, neurons, threshold }
    }

    /// Check if this pattern is active given a set of active global neuron IDs.
    pub fn is_active(&self, active_set: &HashSet<u32>) -> bool {
        if self.neurons.is_empty() {
            return false;
        }
        let active_count = self.neurons.iter().filter(|n| active_set.contains(n)).count();
        (active_count as f32 / self.neurons.len() as f32) >= self.threshold
    }

    /// Fraction of neurons currently active.
    pub fn activation_ratio(&self, active_set: &HashSet<u32>) -> f32 {
        if self.neurons.is_empty() {
            return 0.0;
        }
        let active_count = self.neurons.iter().filter(|n| active_set.contains(n)).count();
        active_count as f32 / self.neurons.len() as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Binding {
    pub id: u32,
    pub pattern_a: PatternRef,
    pub pattern_b: PatternRef,
    pub trace_id_a: Option<String>,
    pub trace_id_b: Option<String>,
    pub weight: f32,
    pub fires: u32,
    pub time_delta: f32,
    pub last_fired: u64,
    pub confidence: f32,
    pub opportunities: u32,
}

#[derive(Debug, Clone)]
pub struct BindingRecallTraceInput {
    pub binding_id: u32,
    pub trace_id_a: String,
    pub trace_id_b: String,
    pub ratio_a: f32,
    pub ratio_b: f32,
    pub recall_weight: f32,
}

#[derive(Debug, Clone)]
pub struct BindingRecallSignal {
    pub binding_id: u32,
    pub target_region: RegionId,
    pub target_neurons: Vec<u32>,
    pub relative_weight: f32,
    pub source_activation_ratio: f32,
}

#[derive(Debug, Clone, Default)]
pub struct BindingRecallPlan {
    pub signals: Vec<BindingRecallSignal>,
    pub max_relative_weight: f32,
    pub max_source_activation_ratio: f32,
}

impl BindingRecallPlan {
    pub fn triggered_bindings(&self) -> u32 {
        self.signals.len() as u32
    }

    pub fn injected_neurons(&self) -> u32 {
        self.signals
            .iter()
            .map(|signal| signal.target_neurons.len() as u32)
            .sum()
    }

}

/// Manages all bindings with fast lookup by region pair.
#[derive(Clone, Serialize, Deserialize)]
pub struct BindingStore {
    bindings: Vec<Binding>,
    next_id: u32,
    region_pair_index: HashMap<(RegionId, RegionId), Vec<usize>>,
}

fn accumulate_max_weight_by_source(
    max_weight_by_source: &mut HashMap<PatternKey, f32>,
    binding: &Binding,
) {
    for pattern in [&binding.pattern_a, &binding.pattern_b] {
        let pattern_key = PatternKey::from(pattern);
        max_weight_by_source
            .entry(pattern_key)
            .and_modify(|max_weight| *max_weight = (*max_weight).max(binding.weight))
            .or_insert(binding.weight);
    }
}

fn merge_max_weight_maps(
    mut acc: HashMap<PatternKey, f32>,
    local: HashMap<PatternKey, f32>,
) -> HashMap<PatternKey, f32> {
    for (pattern_key, weight) in local {
        acc.entry(pattern_key)
            .and_modify(|max_weight| *max_weight = (*max_weight).max(weight))
            .or_insert(weight);
    }
    acc
}

impl BindingStore {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            next_id: 0,
            region_pair_index: HashMap::new(),
        }
    }

    pub fn from_bindings(bindings: Vec<Binding>) -> Self {
        let next_id = bindings
            .iter()
            .map(|binding| binding.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let mut store = Self {
            bindings,
            next_id,
            region_pair_index: HashMap::new(),
        };
        store.rebuild_index();
        store
    }

    /// Create a new binding between two patterns.
    pub fn add(
        &mut self,
        pattern_a: PatternRef,
        pattern_b: PatternRef,
        time_delta: f32,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let idx = self.bindings.len();
        let pair = (pattern_a.region, pattern_b.region);

        self.bindings.push(Binding {
            id,
            pattern_a,
            pattern_b,
            trace_id_a: None,
            trace_id_b: None,
            weight: 0.2,
            fires: 0,
            time_delta,
            last_fired: 0,
            confidence: 0.0,
            opportunities: 0,
        });

        self.region_pair_index.entry(pair).or_default().push(idx);
        id
    }

    /// Evaluate all bindings against current active neurons.
    /// Returns (binding_id, weight) for bindings where BOTH patterns are active.
    pub fn evaluate(&self, active_set: &HashSet<u32>) -> Vec<(u32, f32)> {
        let mut results = Vec::new();
        for binding in &self.bindings {
            if binding.pattern_a.is_active(active_set) && binding.pattern_b.is_active(active_set) {
                results.push((binding.id, binding.weight));
            }
        }
        results
    }

    pub fn annotate_traces(&mut self, binding_id: u32, trace_id_a: String, trace_id_b: String) {
        if let Some(binding) = self.bindings.iter_mut().find(|binding| binding.id == binding_id) {
            binding.trace_id_a = Some(trace_id_a);
            binding.trace_id_b = Some(trace_id_b);
        }
    }

    /// Find bindings where only ONE pattern is active (opportunity without co-activation).
    pub fn find_partial(&self, active_set: &HashSet<u32>) -> Vec<u32> {
        let mut results = Vec::new();
        for binding in &self.bindings {
            let a = binding.pattern_a.is_active(active_set);
            let b = binding.pattern_b.is_active(active_set);
            if a ^ b {
                results.push(binding.id);
            }
        }
        results
    }

    /// Get the current activation ratios for both sides of a binding.
    pub fn activation_ratios(&self, binding_id: u32, active_set: &HashSet<u32>) -> Option<(f32, f32)> {
        self.get(binding_id).map(|binding| {
            (
                binding.pattern_a.activation_ratio(active_set),
                binding.pattern_b.activation_ratio(active_set),
            )
        })
    }

    /// Get current activation ratios for multiple bindings in one pass.
    pub fn activation_ratios_for_bindings(
        &self,
        binding_ids: &[u32],
        active_set: &HashSet<u32>,
    ) -> Vec<(u32, f32, f32)> {
        if binding_ids.is_empty() {
            return Vec::new();
        }

        let requested: HashSet<u32> = binding_ids.iter().copied().collect();
        let mut ratios_by_id: HashMap<u32, (f32, f32)> =
            HashMap::with_capacity(requested.len());

        for binding in &self.bindings {
            if !requested.contains(&binding.id) {
                continue;
            }

            ratios_by_id.insert(
                binding.id,
                (
                    binding.pattern_a.activation_ratio(active_set),
                    binding.pattern_b.activation_ratio(active_set),
                ),
            );

            if ratios_by_id.len() == requested.len() {
                break;
            }
        }

        binding_ids
            .iter()
            .filter_map(|binding_id| {
                ratios_by_id
                    .get(binding_id)
                    .map(|(ratio_a, ratio_b)| (*binding_id, *ratio_a, *ratio_b))
            })
            .collect()
    }

    /// Plan partner-side recall for partially active bindings.
    ///
    /// Invariant: relative weight is always computed against the current-tick
    /// strongest binding that contains the same source pattern, regardless of
    /// whether that stronger binding is partial, fully active, or otherwise not
    /// itself eligible as a recall candidate this tick.
    pub fn recall_candidates(
        &self,
        active_set: &HashSet<u32>,
        min_relative_weight: f32,
    ) -> BindingRecallPlan {
        let min_relative_weight = min_relative_weight.clamp(0.0, 1.0);
        let use_parallel = rayon::current_num_threads() > 1
            && self.bindings.len() >= RECALL_COMPLETION_PARALLEL_MIN_BINDINGS;
        let max_weight_by_source = if use_parallel {
            self.bindings
                .par_chunks(RECALL_COMPLETION_PARALLEL_CHUNK_SIZE)
                .map(|chunk| {
                    let mut local = HashMap::new();
                    for binding in chunk {
                        accumulate_max_weight_by_source(&mut local, binding);
                    }
                    local
                })
                .reduce(HashMap::new, merge_max_weight_maps)
        } else {
            let mut local = HashMap::new();
            for binding in &self.bindings {
                accumulate_max_weight_by_source(&mut local, binding);
            }
            local
        };

        let candidates: Vec<(u32, PatternKey, RegionId, Vec<u32>, f32, f32)> = if use_parallel {
            self.bindings
                .par_iter()
                .filter_map(|binding| {
                    let a_ratio = binding.pattern_a.activation_ratio(active_set);
                    let b_ratio = binding.pattern_b.activation_ratio(active_set);
                    let a_active = a_ratio >= binding.pattern_a.threshold;
                    let b_active = b_ratio >= binding.pattern_b.threshold;

                    if a_active == b_active {
                        return None;
                    }

                    let (source_pattern, target_pattern, source_ratio) = if a_active {
                        (&binding.pattern_a, &binding.pattern_b, a_ratio)
                    } else {
                        (&binding.pattern_b, &binding.pattern_a, b_ratio)
                    };

                    Some((
                        binding.id,
                        PatternKey::from(source_pattern),
                        target_pattern.region,
                        target_pattern.neurons.clone(),
                        binding.weight,
                        source_ratio,
                    ))
                })
                .collect()
        } else {
            let mut local = Vec::new();
            for binding in &self.bindings {
                let a_ratio = binding.pattern_a.activation_ratio(active_set);
                let b_ratio = binding.pattern_b.activation_ratio(active_set);
                let a_active = a_ratio >= binding.pattern_a.threshold;
                let b_active = b_ratio >= binding.pattern_b.threshold;

                if a_active == b_active {
                    continue;
                }

                let (source_pattern, target_pattern, source_ratio) = if a_active {
                    (&binding.pattern_a, &binding.pattern_b, a_ratio)
                } else {
                    (&binding.pattern_b, &binding.pattern_a, b_ratio)
                };

                local.push((
                    binding.id,
                    PatternKey::from(source_pattern),
                    target_pattern.region,
                    target_pattern.neurons.clone(),
                    binding.weight,
                    source_ratio,
                ));
            }
            local
        };

        let mut plan = BindingRecallPlan::default();
        for (binding_id, source_key, target_region, target_neurons, weight, source_ratio) in candidates {
            let Some(max_weight) = max_weight_by_source.get(&source_key).copied() else {
                continue;
            };
            if max_weight <= 0.0 {
                continue;
            }

            let relative_weight = (weight / max_weight).clamp(0.0, 1.0);
            if relative_weight < min_relative_weight {
                continue;
            }

            plan.max_relative_weight = plan.max_relative_weight.max(relative_weight);
            plan.max_source_activation_ratio =
                plan.max_source_activation_ratio.max(source_ratio);
            plan.signals.push(BindingRecallSignal {
                binding_id,
                target_region,
                target_neurons,
                relative_weight,
                source_activation_ratio: source_ratio,
            });
        }

        plan
    }

    pub fn recall_completion_inputs(
        &self,
        active_set: &HashSet<u32>,
        min_relative_weight: f32,
    ) -> Vec<BindingRecallTraceInput> {
        let min_relative_weight = min_relative_weight.clamp(0.0, 1.0);
        let use_parallel = rayon::current_num_threads() > 1
            && self.bindings.len() >= RECALL_COMPLETION_PARALLEL_MIN_BINDINGS;
        let max_weight_by_source = if use_parallel {
            self.bindings
                .par_chunks(RECALL_COMPLETION_PARALLEL_CHUNK_SIZE)
                .map(|chunk| {
                    let mut local = HashMap::new();
                    for binding in chunk {
                        accumulate_max_weight_by_source(&mut local, binding);
                    }
                    local
                })
                .reduce(HashMap::new, merge_max_weight_maps)
        } else {
            let mut local = HashMap::new();
            for binding in &self.bindings {
                accumulate_max_weight_by_source(&mut local, binding);
            }
            local
        };

        let build_input = |binding: &Binding| {
            let (Some(trace_id_a), Some(trace_id_b)) = (
                binding.trace_id_a.as_ref(),
                binding.trace_id_b.as_ref(),
            ) else {
                return None;
            };

            let ratio_a = binding.pattern_a.activation_ratio(active_set);
            let ratio_b = binding.pattern_b.activation_ratio(active_set);
            let a_active = ratio_a >= binding.pattern_a.threshold;
            let b_active = ratio_b >= binding.pattern_b.threshold;

            let recall_weight = if a_active && b_active {
                1.0
            } else if a_active ^ b_active {
                let source_pattern = if a_active {
                    &binding.pattern_a
                } else {
                    &binding.pattern_b
                };
                let source_key = PatternKey::from(source_pattern);
                let max_weight = max_weight_by_source.get(&source_key).copied()?;
                if max_weight <= 0.0 {
                    return None;
                }
                let relative_weight = (binding.weight / max_weight).clamp(0.0, 1.0);
                if relative_weight < min_relative_weight {
                    return None;
                }
                relative_weight
            } else {
                return None;
            };

            Some(BindingRecallTraceInput {
                binding_id: binding.id,
                trace_id_a: trace_id_a.clone(),
                trace_id_b: trace_id_b.clone(),
                ratio_a,
                ratio_b,
                recall_weight,
            })
        };

        if use_parallel {
            self.bindings.par_iter().filter_map(build_input).collect()
        } else {
            self.bindings.iter().filter_map(build_input).collect()
        }
    }

    /// Evaluate and update bindings in one pass.
    /// Returns (strengthened_count, missed_count).
    pub fn process_activity(&mut self, active_set: &HashSet<u32>, tick: u64) -> (u32, u32) {
        let mut strengthened = 0u32;
        let mut missed = 0u32;

        for binding in &mut self.bindings {
            let a = binding.pattern_a.is_active(active_set);
            let b = binding.pattern_b.is_active(active_set);

            if a && b {
                binding.fires += 1;
                binding.opportunities += 1;
                binding.weight = (binding.weight + 0.05 * (1.0 - binding.weight)).min(1.0);
                binding.last_fired = tick;
                binding.confidence = binding.fires as f32 / binding.opportunities.max(1) as f32;
                strengthened += 1;
            } else if a ^ b {
                binding.opportunities += 1;
                binding.weight *= 0.998;
                binding.confidence = binding.fires as f32 / binding.opportunities.max(1) as f32;
                missed += 1;
            }
        }

        (strengthened, missed)
    }

    /// Strengthen a binding (co-activation detected).
    pub fn strengthen(&mut self, binding_id: u32, tick: u64) {
        if let Some(b) = self.bindings.iter_mut().find(|b| b.id == binding_id) {
            b.fires += 1;
            b.opportunities += 1;
            b.weight = (b.weight + 0.05 * (1.0 - b.weight)).min(1.0);
            b.last_fired = tick;
            b.confidence = b.fires as f32 / b.opportunities.max(1) as f32;
        }
    }

    /// Record a missed opportunity (one pattern active, other not).
    pub fn record_miss(&mut self, binding_id: u32) {
        if let Some(b) = self.bindings.iter_mut().find(|b| b.id == binding_id) {
            b.opportunities += 1;
            b.weight *= 0.998;
            b.confidence = b.fires as f32 / b.opportunities.max(1) as f32;
        }
    }

    /// Remove dissolved bindings.
    pub fn prune(&mut self, weight_threshold: f32, min_fires: u32) -> u32 {
        let before = self.bindings.len();
        self.bindings.retain(|b| b.weight >= weight_threshold || b.fires >= min_fires);
        let pruned = (before - self.bindings.len()) as u32;
        if pruned > 0 {
            self.rebuild_index();
        }
        pruned
    }

    fn rebuild_index(&mut self) {
        self.region_pair_index.clear();
        for (idx, binding) in self.bindings.iter().enumerate() {
            let pair = (binding.pattern_a.region, binding.pattern_b.region);
            self.region_pair_index.entry(pair).or_default().push(idx);
        }
    }

    pub fn get(&self, binding_id: u32) -> Option<&Binding> {
        self.bindings.iter().find(|b| b.id == binding_id)
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Get bindings involving a specific region pair (either direction).
    pub fn bindings_for_pair(&self, a: RegionId, b: RegionId) -> Vec<&Binding> {
        let mut result = Vec::new();
        if let Some(indices) = self.region_pair_index.get(&(a, b)) {
            for &idx in indices {
                if idx < self.bindings.len() {
                    result.push(&self.bindings[idx]);
                }
            }
        }
        if a != b {
            if let Some(indices) = self.region_pair_index.get(&(b, a)) {
                for &idx in indices {
                    if idx < self.bindings.len() {
                        result.push(&self.bindings[idx]);
                    }
                }
            }
        }
        result
    }

    /// Iterate all bindings.
    pub fn iter(&self) -> impl Iterator<Item = &Binding> {
        self.bindings.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pattern(region: RegionId, neurons: Vec<u32>) -> PatternRef {
        PatternRef::new(region, neurons, 0.7)
    }

    #[test]
    fn test_pattern_ref_active() {
        let p = make_pattern(RegionId::Visual, vec![10000, 10001, 10002]);
        let mut active = HashSet::new();
        assert!(!p.is_active(&active));

        // 2/3 = 0.67 < 0.7 threshold
        active.insert(10000);
        active.insert(10001);
        assert!(!p.is_active(&active));

        // 3/3 = 1.0 >= 0.7
        active.insert(10002);
        assert!(p.is_active(&active));
    }

    #[test]
    fn test_binding_creation_and_eval() {
        let mut store = BindingStore::new();
        let pa = make_pattern(RegionId::Visual, vec![10000, 10001, 10002]);
        let pb = make_pattern(RegionId::Audio, vec![30000, 30001, 30002]);
        let id = store.add(pa, pb, 0.0);
        assert_eq!(store.len(), 1);

        // Nothing active
        let active = HashSet::new();
        assert!(store.evaluate(&active).is_empty());

        // Both patterns active
        let active: HashSet<u32> = [10000, 10001, 10002, 30000, 30001, 30002].into();
        let results = store.evaluate(&active);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        assert!((results[0].1 - 0.2).abs() < 0.01); // initial weight
    }

    #[test]
    fn test_binding_strengthen() {
        let mut store = BindingStore::new();
        let pa = make_pattern(RegionId::Visual, vec![10000]);
        let pb = make_pattern(RegionId::Audio, vec![30000]);
        let id = store.add(pa, pb, 0.0);

        store.strengthen(id, 100);
        let b = store.get(id).unwrap();
        assert_eq!(b.fires, 1);
        assert!(b.weight > 0.2);
    }

    #[test]
    fn test_process_activity_updates_active_and_partial_bindings() {
        let mut store = BindingStore::new();
        let active_id = store.add(
            make_pattern(RegionId::Visual, vec![10000]),
            make_pattern(RegionId::Audio, vec![30000]),
            0.0,
        );
        let partial_id = store.add(
            make_pattern(RegionId::Visual, vec![10001]),
            make_pattern(RegionId::Audio, vec![30001]),
            0.0,
        );

        let active: HashSet<u32> = [10000, 30000, 10001].into();
        let (strengthened, missed) = store.process_activity(&active, 42);

        assert_eq!(strengthened, 1);
        assert_eq!(missed, 1);
        assert_eq!(store.get(active_id).unwrap().last_fired, 42);
        assert!(store.get(active_id).unwrap().weight > 0.2);
        assert!(store.get(partial_id).unwrap().weight < 0.2);
    }

    #[test]
    fn test_binding_weaken_and_prune() {
        let mut store = BindingStore::new();
        let pa = make_pattern(RegionId::Visual, vec![10000]);
        let pb = make_pattern(RegionId::Audio, vec![30000]);
        let id = store.add(pa, pb, 0.0);

        // Weaken by many missed opportunities
        // 0.2 * 0.998^1000 ≈ 0.027 → below 0.05 threshold
        for _ in 0..1000 {
            store.record_miss(id);
        }
        let b = store.get(id).unwrap();
        assert!(b.weight < 0.05, "Many misses should weaken: {}", b.weight);

        // Prune: weight < 0.05 and fires < 10
        let pruned = store.prune(0.05, 10);
        assert_eq!(pruned, 1);
        assert!(store.is_empty());
    }

    #[test]
    fn test_find_partial() {
        let mut store = BindingStore::new();
        let pa = make_pattern(RegionId::Visual, vec![10000]);
        let pb = make_pattern(RegionId::Audio, vec![30000]);
        let id = store.add(pa, pb, 0.0);

        // Only pattern A active
        let active: HashSet<u32> = [10000].into();
        let partial = store.find_partial(&active);
        assert_eq!(partial.len(), 1);
        assert_eq!(partial[0], id);
    }

    #[test]
    fn test_region_pair_lookup() {
        let mut store = BindingStore::new();
        let pa = make_pattern(RegionId::Visual, vec![10000]);
        let pb = make_pattern(RegionId::Audio, vec![30000]);
        store.add(pa, pb, 0.0);

        assert_eq!(store.bindings_for_pair(RegionId::Visual, RegionId::Audio).len(), 1);
        assert_eq!(store.bindings_for_pair(RegionId::Audio, RegionId::Visual).len(), 1);
        assert!(store.bindings_for_pair(RegionId::Sensory, RegionId::Motor).is_empty());
    }

    #[test]
    fn test_binding_activation_ratios() {
        let mut store = BindingStore::new();
        let id = store.add(
            PatternRef::new(RegionId::Visual, vec![10000, 10001], 0.5),
            PatternRef::new(RegionId::Audio, vec![30000, 30001], 0.5),
            0.0,
        );

        let active: HashSet<u32> = [10000, 30000].into();
        let (a_ratio, b_ratio) = store.activation_ratios(id, &active).unwrap();

        assert!((a_ratio - 0.5).abs() < 0.001);
        assert!((b_ratio - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_recall_completion_inputs_include_annotated_partial_and_active_bindings() {
        let mut store = BindingStore::new();
        let partial_id = store.add(
            PatternRef::new(RegionId::Visual, vec![10000], 1.0),
            PatternRef::new(RegionId::Audio, vec![30000], 1.0),
            0.0,
        );
        let active_id = store.add(
            PatternRef::new(RegionId::Pattern, vec![80000], 1.0),
            PatternRef::new(RegionId::MemoryLong, vec![55000], 1.0),
            0.0,
        );
        let unannotated_id = store.add(
            PatternRef::new(RegionId::Emotion, vec![70000], 1.0),
            PatternRef::new(RegionId::Executive, vec![120000], 1.0),
            0.0,
        );
        store.annotate_traces(partial_id, "cue".to_string(), "partner".to_string());
        store.annotate_traces(active_id, "pattern".to_string(), "memory".to_string());
        store.bindings[0].weight = 0.2;
        store.bindings[1].weight = 0.1;
        store.bindings[2].weight = 0.3;

        let active: HashSet<u32> = [10000, 80000, 55000].into();
        let inputs = store.recall_completion_inputs(&active, 0.5);

        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].binding_id, partial_id);
        assert_eq!(inputs[0].trace_id_a, "cue");
        assert_eq!(inputs[0].trace_id_b, "partner");
        assert!((inputs[0].ratio_a - 1.0).abs() < 0.001);
        assert!((inputs[0].ratio_b - 0.0).abs() < 0.001);
        assert!((inputs[0].recall_weight - 1.0).abs() < 0.001);
        assert_eq!(inputs[1].binding_id, active_id);
        assert_eq!(inputs[1].trace_id_a, "pattern");
        assert_eq!(inputs[1].trace_id_b, "memory");
        assert!((inputs[1].ratio_a - 1.0).abs() < 0.001);
        assert!((inputs[1].ratio_b - 1.0).abs() < 0.001);
        assert!((inputs[1].recall_weight - 1.0).abs() < 0.001);
        assert!(inputs.iter().all(|input| input.binding_id != unannotated_id));
    }

    #[test]
    fn test_batch_binding_activation_ratios_preserve_request_order() {
        let mut store = BindingStore::new();
        let first_id = store.add(
            PatternRef::new(RegionId::Visual, vec![10000, 10001], 0.5),
            PatternRef::new(RegionId::Audio, vec![30000, 30001], 0.5),
            0.0,
        );
        let second_id = store.add(
            PatternRef::new(RegionId::Pattern, vec![80000, 80001], 0.5),
            PatternRef::new(RegionId::MemoryLong, vec![50000, 50001], 0.5),
            0.0,
        );

        let active: HashSet<u32> = [10000, 30000, 80000, 80001].into();
        let ratios =
            store.activation_ratios_for_bindings(&[second_id, 9999, first_id], &active);

        assert_eq!(ratios.len(), 2);
        assert_eq!(ratios[0].0, second_id);
        assert!((ratios[0].1 - 1.0).abs() < 0.001);
        assert!((ratios[0].2 - 0.0).abs() < 0.001);
        assert_eq!(ratios[1].0, first_id);
        assert!((ratios[1].1 - 0.5).abs() < 0.001);
        assert!((ratios[1].2 - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_recall_candidates_use_relative_weight() {
        let mut store = BindingStore::new();
        let shared_source = PatternRef::new(RegionId::Visual, vec![10000], 1.0);
        let strong_target = PatternRef::new(RegionId::Audio, vec![30000], 1.0);
        let weak_target = PatternRef::new(RegionId::Audio, vec![30001], 1.0);

        let strong_id = store.add(shared_source.clone(), strong_target, 0.0);
        let _weak_id = store.add(shared_source, weak_target, 0.0);
        store.bindings[0].weight = 0.2;
        store.bindings[1].weight = 0.1;

        let active: HashSet<u32> = [10000].into();
        let plan = store.recall_candidates(&active, 0.75);

        assert_eq!(plan.triggered_bindings(), 1);
        assert_eq!(plan.injected_neurons(), 1);
        assert_eq!(plan.signals[0].binding_id, strong_id);
        assert!((plan.max_relative_weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_recall_candidates_use_global_source_max_even_when_strong_binding_is_active() {
        let mut store = BindingStore::new();
        let shared_source = PatternRef::new(RegionId::Visual, vec![10000], 1.0);
        let strong_target = PatternRef::new(RegionId::Audio, vec![30000], 1.0);
        let weak_target = PatternRef::new(RegionId::Audio, vec![30001], 1.0);

        let strong_id = store.add(shared_source.clone(), strong_target, 0.0);
        let weak_id = store.add(shared_source, weak_target, 0.0);
        store.bindings[0].weight = 0.2;
        store.bindings[1].weight = 0.1;

        // Strong binding is already fully active, weak binding remains partial.
        let active: HashSet<u32> = [10000, 30000].into();
        let plan = store.recall_candidates(&active, 0.75);

        assert_eq!(plan.triggered_bindings(), 0);
        assert_eq!(plan.injected_neurons(), 0);
        assert!(plan.signals.iter().all(|signal| signal.binding_id != weak_id));
        assert!(store.evaluate(&active).iter().any(|(binding_id, _)| *binding_id == strong_id));
    }
}
