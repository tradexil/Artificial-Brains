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
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct Binding {
    pub id: u32,
    pub pattern_a: PatternRef,
    pub pattern_b: PatternRef,
    pub weight: f32,
    pub fires: u32,
    pub time_delta: f32,
    pub last_fired: u64,
    pub confidence: f32,
    pub opportunities: u32,
}

/// Manages all bindings with fast lookup by region pair.
pub struct BindingStore {
    bindings: Vec<Binding>,
    next_id: u32,
    region_pair_index: HashMap<(RegionId, RegionId), Vec<usize>>,
}

impl BindingStore {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            next_id: 0,
            region_pair_index: HashMap::new(),
        }
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
}
