use std::collections::HashMap;

#[derive(Default)]
pub struct NovelPatternRegistry {
    trackers: HashMap<u64, NovelPatternTracker>,
}

impl NovelPatternRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_tracker(&mut self, tracker_id: u64) {
        self.trackers.insert(tracker_id, NovelPatternTracker::new());
    }

    pub fn clear_tracker(&mut self, tracker_id: u64) {
        self.trackers
            .entry(tracker_id)
            .or_insert_with(NovelPatternTracker::new)
            .clear();
    }

    pub fn drop_tracker(&mut self, tracker_id: u64) {
        self.trackers.remove(&tracker_id);
    }

    pub fn update(
        &mut self,
        tracker_id: u64,
        region_neurons: HashMap<String, Vec<u32>>,
        novelty: f32,
        min_regions: usize,
        persistence_required: usize,
    ) -> Vec<HashMap<String, Vec<u32>>> {
        self.trackers
            .entry(tracker_id)
            .or_insert_with(NovelPatternTracker::new)
            .update(region_neurons, novelty, min_regions, persistence_required)
    }
}

#[derive(Default)]
pub struct BindingTrackerRegistry {
    trackers: HashMap<u64, BindingTracker>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BindingReadyPair {
    pub trace_id_a: String,
    pub region_a: String,
    pub trace_id_b: String,
    pub region_b: String,
    pub avg_delta: f32,
    pub support_count: usize,
    pub span_ticks: u64,
    pub first_tick: u64,
    pub last_tick: u64,
}

impl BindingReadyPair {
    fn key(&self) -> (String, String, String, String) {
        (
            self.trace_id_a.clone(),
            self.region_a.clone(),
            self.trace_id_b.clone(),
            self.region_b.clone(),
        )
    }
}

impl BindingTrackerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_tracker(&mut self, tracker_id: u64) {
        self.trackers.insert(tracker_id, BindingTracker::new());
    }

    pub fn clear_tracker(&mut self, tracker_id: u64) {
        self.trackers
            .entry(tracker_id)
            .or_insert_with(BindingTracker::new)
            .clear();
    }

    pub fn drop_tracker(&mut self, tracker_id: u64) {
        self.trackers.remove(&tracker_id);
    }

    pub fn record(
        &mut self,
        tracker_id: u64,
        active_patterns: Vec<(String, String)>,
        tick: u64,
        formation_count: usize,
        temporal_window: u64,
    ) -> Vec<(String, String, String, String, f32)> {
        let tracker = self
            .trackers
            .entry(tracker_id)
            .or_insert_with(BindingTracker::new);
        let ready_pairs = tracker.record_detailed(active_patterns, tick, formation_count, temporal_window);
        let ready_keys = ready_pairs
            .iter()
            .map(BindingReadyPair::key)
            .collect::<Vec<_>>();
        tracker.consume(&ready_keys);
        ready_pairs
            .into_iter()
            .map(|pair| {
                (
                    pair.trace_id_a,
                    pair.region_a,
                    pair.trace_id_b,
                    pair.region_b,
                    pair.avg_delta,
                )
            })
            .collect()
    }

    pub fn record_detailed(
        &mut self,
        tracker_id: u64,
        active_patterns: Vec<(String, String)>,
        tick: u64,
        formation_count: usize,
        temporal_window: u64,
    ) -> Vec<BindingReadyPair> {
        self.trackers
            .entry(tracker_id)
            .or_insert_with(BindingTracker::new)
            .record_detailed(active_patterns, tick, formation_count, temporal_window)
    }

    pub fn consume(
        &mut self,
        tracker_id: u64,
        keys: Vec<(String, String, String, String)>,
    ) {
        if let Some(tracker) = self.trackers.get_mut(&tracker_id) {
            tracker.consume(&keys);
        }
    }

    pub fn cleanup(&mut self, tracker_id: u64, current_tick: u64, max_age: u64) {
        if let Some(tracker) = self.trackers.get_mut(&tracker_id) {
            tracker.cleanup(current_tick, max_age);
        }
    }
}

#[derive(Clone)]
struct TrackedNovelPattern {
    fingerprint: Vec<u32>,
    region_neurons: HashMap<String, Vec<u32>>,
    persistence: usize,
}

struct NovelPatternTracker {
    patterns: Vec<TrackedNovelPattern>,
}

impl NovelPatternTracker {
    fn new() -> Self {
        Self { patterns: Vec::new() }
    }

    fn clear(&mut self) {
        self.patterns.clear();
    }

    fn update(
        &mut self,
        region_neurons: HashMap<String, Vec<u32>>,
        novelty: f32,
        min_regions: usize,
        persistence_required: usize,
    ) -> Vec<HashMap<String, Vec<u32>>> {
        if novelty < 0.01 {
            return Vec::new();
        }

        let normalized = normalize_region_neurons(region_neurons);
        if normalized.len() < min_regions {
            return Vec::new();
        }

        let fingerprint = flattened_fingerprint(&normalized);
        if fingerprint.is_empty() {
            return Vec::new();
        }

        if let Some(existing) = self
            .patterns
            .iter_mut()
            .find(|pattern| jaccard_ratio(&pattern.fingerprint, &fingerprint) > 0.5)
        {
            existing.persistence += 1;
            existing.fingerprint = fingerprint;
            existing.region_neurons = normalized;
        } else {
            self.patterns.push(TrackedNovelPattern {
                fingerprint,
                region_neurons: normalized,
                persistence: 1,
            });
        }

        let mut ready = Vec::new();
        let mut survivors = Vec::with_capacity(self.patterns.len());
        for pattern in self.patterns.drain(..) {
            if pattern.persistence >= persistence_required {
                ready.push(pattern.region_neurons);
            } else {
                survivors.push(pattern);
            }
        }
        self.patterns = survivors;
        ready
    }
}

struct BindingTracker {
    co_activations: HashMap<(String, String, String, String), Vec<u64>>,
}

impl BindingTracker {
    fn new() -> Self {
        Self {
            co_activations: HashMap::new(),
        }
    }

    fn clear(&mut self) {
        self.co_activations.clear();
    }

    fn record_detailed(
        &mut self,
        active_patterns: Vec<(String, String)>,
        tick: u64,
        formation_count: usize,
        temporal_window: u64,
    ) -> Vec<BindingReadyPair> {
        if active_patterns.len() < 2 {
            return Vec::new();
        }

        let mut ready = Vec::new();

        for i in 0..active_patterns.len() {
            let (tid_a, region_a) = &active_patterns[i];
            for (tid_b, region_b) in active_patterns.iter().skip(i + 1) {
                if tid_a == tid_b {
                    continue;
                }
                if region_a == region_b {
                    continue;
                }

                let key = canonical_binding_key(tid_a, region_a, tid_b, region_b);
                let activations = self.co_activations.entry(key.clone()).or_default();
                activations.push(tick);
                if activations.len() > formation_count {
                    let drop_count = activations.len() - formation_count;
                    activations.drain(0..drop_count);
                }

                if activations.len() >= formation_count {
                    let first = activations[activations.len() - formation_count];
                    let last = *activations.last().unwrap_or(&tick);
                    if last.saturating_sub(first)
                        <= temporal_window.saturating_mul(formation_count as u64)
                    {
                        let support_count = activations.len();
                        let span_ticks = last.saturating_sub(first);
                        let avg_delta = if support_count > 1 {
                            span_ticks as f32 / (support_count as f32 - 1.0)
                        } else {
                            0.0
                        };
                        ready.push(BindingReadyPair {
                            trace_id_a: key.0.clone(),
                            region_a: key.1.clone(),
                            trace_id_b: key.2.clone(),
                            region_b: key.3.clone(),
                            avg_delta,
                            support_count,
                            span_ticks,
                            first_tick: first,
                            last_tick: last,
                        });
                    }
                }
            }
        }

        ready
    }

    fn consume(&mut self, keys: &[(String, String, String, String)]) {
        for key in keys {
            self.co_activations.remove(key);
        }
    }

    fn cleanup(&mut self, current_tick: u64, max_age: u64) {
        self.co_activations.retain(|_, ticks| {
            ticks.retain(|tick| current_tick.saturating_sub(*tick) < max_age);
            !ticks.is_empty()
        });
    }
}

fn normalize_region_neurons(
    region_neurons: HashMap<String, Vec<u32>>,
) -> HashMap<String, Vec<u32>> {
    let mut normalized = HashMap::new();
    for (region, mut neurons) in region_neurons {
        neurons.sort_unstable();
        neurons.dedup();
        if !neurons.is_empty() {
            normalized.insert(region, neurons);
        }
    }
    normalized
}

fn flattened_fingerprint(region_neurons: &HashMap<String, Vec<u32>>) -> Vec<u32> {
    let mut fingerprint = Vec::new();
    for neurons in region_neurons.values() {
        fingerprint.extend(neurons.iter().copied());
    }
    fingerprint.sort_unstable();
    fingerprint.dedup();
    fingerprint
}

fn jaccard_ratio(a: &[u32], b: &[u32]) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut overlap = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                overlap += 1;
                i += 1;
                j += 1;
            }
        }
    }

    let union = a.len() + b.len() - overlap;
    if union == 0 {
        0.0
    } else {
        overlap as f32 / union as f32
    }
}

fn canonical_binding_key(
    tid_a: &str,
    region_a: &str,
    tid_b: &str,
    region_b: &str,
) -> (String, String, String, String) {
    if (tid_a, region_a) > (tid_b, region_b) {
        (
            tid_b.to_string(),
            region_b.to_string(),
            tid_a.to_string(),
            region_a.to_string(),
        )
    } else {
        (
            tid_a.to_string(),
            region_a.to_string(),
            tid_b.to_string(),
            region_b.to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_novel_tracker_forms_after_persistence() {
        let mut registry = NovelPatternRegistry::new();
        registry.create_tracker(1);
        let pattern = HashMap::from([
            ("visual".to_string(), vec![10_000, 10_001]),
            ("audio".to_string(), vec![30_000, 30_001]),
        ]);

        assert!(registry.update(1, pattern.clone(), 0.5, 2, 3).is_empty());
        assert!(registry.update(1, pattern.clone(), 0.5, 2, 3).is_empty());
        let ready = registry.update(1, pattern, 0.5, 2, 3);

        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].len(), 2);
    }

    #[test]
    fn test_binding_tracker_forms_after_threshold() {
        let mut registry = BindingTrackerRegistry::new();
        registry.create_tracker(1);

        for tick in 0..4 {
            let ready = registry.record(
                1,
                vec![
                    ("t1".to_string(), "visual".to_string()),
                    ("t2".to_string(), "audio".to_string()),
                ],
                tick,
                5,
                5,
            );
            assert!(ready.is_empty());
        }

        let ready = registry.record(
            1,
            vec![
                ("t1".to_string(), "visual".to_string()),
                ("t2".to_string(), "audio".to_string()),
            ],
            4,
            5,
            5,
        );

        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].0, "t1");
        assert_eq!(ready[0].2, "t2");
    }

    #[test]
    fn test_binding_tracker_requires_repeats_within_effective_horizon() {
        let mut registry = BindingTrackerRegistry::new();
        registry.create_tracker(1);

        for tick in [0_u64, 3, 6, 9] {
            let ready = registry.record(
                1,
                vec![
                    ("t1".to_string(), "visual".to_string()),
                    ("t2".to_string(), "audio".to_string()),
                ],
                tick,
                5,
                2,
            );
            assert!(ready.is_empty());
        }

        let ready = registry.record(
            1,
            vec![
                ("t1".to_string(), "visual".to_string()),
                ("t2".to_string(), "audio".to_string()),
            ],
            12,
            5,
            2,
        );

        assert!(ready.is_empty());
    }

    #[test]
    fn test_binding_tracker_ignores_same_trace_multi_region_pairs() {
        let mut registry = BindingTrackerRegistry::new();
        registry.create_tracker(1);

        let ready = registry.record(
            1,
            vec![
                ("t1".to_string(), "visual".to_string()),
                ("t1".to_string(), "audio".to_string()),
                ("t2".to_string(), "audio".to_string()),
            ],
            0,
            1,
            1,
        );

        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].0, "t1");
        assert_eq!(ready[0].1, "visual");
        assert_eq!(ready[0].2, "t2");
        assert_eq!(ready[0].3, "audio");
    }

    #[test]
    fn test_binding_tracker_detailed_ready_pairs_persist_until_consumed() {
        let mut registry = BindingTrackerRegistry::new();
        registry.create_tracker(1);

        for tick in 0..4 {
            let ready = registry.record_detailed(
                1,
                vec![
                    ("t1".to_string(), "visual".to_string()),
                    ("t2".to_string(), "audio".to_string()),
                ],
                tick,
                5,
                5,
            );
            assert!(ready.is_empty());
        }

        let ready = registry.record_detailed(
            1,
            vec![
                ("t1".to_string(), "visual".to_string()),
                ("t2".to_string(), "audio".to_string()),
            ],
            4,
            5,
            5,
        );
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].support_count, 5);
        assert_eq!(ready[0].first_tick, 0);
        assert_eq!(ready[0].last_tick, 4);

        let ready_again = registry.record_detailed(
            1,
            vec![
                ("t1".to_string(), "visual".to_string()),
                ("t2".to_string(), "audio".to_string()),
            ],
            5,
            5,
            5,
        );
        assert_eq!(ready_again.len(), 1);
        assert_eq!(ready_again[0].last_tick, 5);

        registry.consume(
            1,
            vec![(
                "t1".to_string(),
                "visual".to_string(),
                "t2".to_string(),
                "audio".to_string(),
            )],
        );

        let after_consume = registry.record_detailed(
            1,
            vec![
                ("t1".to_string(), "visual".to_string()),
                ("t2".to_string(), "audio".to_string()),
            ],
            6,
            5,
            5,
        );
        assert!(after_consume.is_empty());
    }
}