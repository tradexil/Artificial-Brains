use std::collections::HashMap;

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

    pub fn drop_store(&mut self, store_id: u64) {
        self.stores.remove(&store_id);
    }

    pub fn upsert_trace(&mut self, store_id: u64, trace_id: String, neurons: Vec<u32>) {
        self.stores
            .entry(store_id)
            .or_insert_with(TraceMatcher::new)
            .upsert_trace(trace_id, neurons);
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
}

struct TraceEntry {
    id: String,
    neurons: Vec<u32>,
    total_neurons: u32,
}

struct TraceMatcher {
    traces: Vec<Option<TraceEntry>>,
    id_to_index: HashMap<String, usize>,
    neuron_to_traces: HashMap<u32, Vec<usize>>,
    free_indices: Vec<usize>,
}

impl TraceMatcher {
    fn new() -> Self {
        Self {
            traces: Vec::new(),
            id_to_index: HashMap::new(),
            neuron_to_traces: HashMap::new(),
            free_indices: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.traces.clear();
        self.id_to_index.clear();
        self.neuron_to_traces.clear();
        self.free_indices.clear();
    }

    fn upsert_trace(&mut self, trace_id: String, mut neurons: Vec<u32>) {
        neurons.sort_unstable();
        neurons.dedup();

        if neurons.is_empty() {
            self.remove_trace(&trace_id);
            return;
        }

        self.remove_trace(&trace_id);

        let idx = self.free_indices.pop().unwrap_or_else(|| {
            self.traces.push(None);
            self.traces.len() - 1
        });

        let entry = TraceEntry {
            id: trace_id.clone(),
            total_neurons: neurons.len() as u32,
            neurons: neurons.clone(),
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
    }

    fn matching_traces(&self, active_neurons: &[u32], threshold: f32) -> Vec<(String, f32)> {
        if active_neurons.is_empty() {
            return Vec::new();
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

            let score = active_count as f32 / entry.total_neurons as f32;
            if score >= threshold {
                matches.push((entry.id.clone(), score));
            }
        }

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }
}