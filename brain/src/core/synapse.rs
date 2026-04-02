/// Synapse storage in Compressed Sparse Row (CSR) format.
///
/// CSR gives cache-friendly contiguous memory for 5M+ synapses.
/// Iteration over a neuron's outgoing connections is O(degree).
///
/// Modifications are batched: weight updates, creates, and prunes
/// go into pending buffers, then applied during rebuild.

use crate::core::neuron::NeuronType;
use crate::core::region::RegionId;

const CANONICAL_TOTAL_NEURONS: u32 = 152_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkLayout {
    Contiguous,
    CanonicalRegionBuckets,
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkSegment {
    pub region_id: RegionId,
    pub global_start: u32,
    pub local_start: u32,
    pub len: u32,
}

fn select_chunk_layout(num_neurons: u32, chunk_count: usize) -> ChunkLayout {
    if chunk_count > 1 && num_neurons == CANONICAL_TOTAL_NEURONS {
        ChunkLayout::CanonicalRegionBuckets
    } else {
        ChunkLayout::Contiguous
    }
}

fn normalized_chunk_count(num_neurons: u32, chunk_count: usize) -> usize {
    (chunk_count.max(1)).min(num_neurons.max(1) as usize)
}

fn chunk_size_for(num_neurons: u32, chunk_count: usize) -> u32 {
    ((num_neurons as usize + chunk_count - 1) / chunk_count) as u32
}

fn bucket_bounds(size: u32, bucket_idx: usize, bucket_count: usize) -> Option<(u32, u32)> {
    let bucket_size = ((size as usize + bucket_count - 1) / bucket_count) as u32;
    let bucket_start = bucket_idx as u32 * bucket_size;
    if bucket_start >= size {
        return None;
    }

    Some((bucket_start, (bucket_start + bucket_size).min(size)))
}

fn canonical_bucket_for_neuron(global_id: u32, chunk_count: usize) -> Option<usize> {
    for &region_id in RegionId::ALL.iter() {
        let (region_start, region_end) = region_id.neuron_range();
        if global_id < region_start || global_id > region_end {
            continue;
        }

        let local_idx = global_id - region_start;
        let bucket_size = ((region_id.neuron_count() as usize + chunk_count - 1) / chunk_count) as u32;
        return Some(((local_idx / bucket_size) as usize).min(chunk_count - 1));
    }

    None
}

fn chunk_for_neuron_with(
    global_id: u32,
    num_neurons: u32,
    chunk_size: u32,
    chunk_count: usize,
    chunk_layout: ChunkLayout,
) -> usize {
    match chunk_layout {
        ChunkLayout::Contiguous => ((global_id / chunk_size) as usize).min(chunk_count - 1),
        ChunkLayout::CanonicalRegionBuckets => canonical_bucket_for_neuron(global_id, chunk_count)
            .unwrap_or_else(|| ((global_id.min(num_neurons.saturating_sub(1)) / chunk_size) as usize).min(chunk_count - 1)),
    }
}

/// A single synapse's data (used during construction / pending ops).
#[derive(Debug, Clone, Copy)]
pub struct SynapseData {
    pub from: u32,
    pub to: u32,
    pub weight: f32,
    pub delay: u8,
    pub plasticity: f32,
}

/// Queued weight modification.
#[derive(Debug, Clone, Copy)]
pub struct SynapseUpdate {
    pub from: u32,
    pub to: u32,
    pub delta: f32,
}

/// CSR synapse pool — the main synapse storage.
pub struct SynapsePool {
    /// Total number of "source" slots = max global neuron ID + 1.
    num_neurons: u32,

    chunk_count: usize,
    chunk_size: u32,
    chunk_layout: ChunkLayout,

    // CSR arrays — contiguous, sorted by source neuron
    pub targets: Vec<u32>,
    pub weights: Vec<f32>,
    pub delays: Vec<u8>,
    pub plasticity: Vec<f32>,
    target_chunks: Vec<u16>,

    /// offsets[i] = index into targets/weights/delays where neuron i's
    /// outgoing synapses start. offsets[num_neurons] = total synapse count.
    pub offsets: Vec<u32>,

    // Pending modification buffers
    pending_updates: Vec<SynapseUpdate>,
    pending_creates: Vec<SynapseData>,
    pending_prunes: Vec<(u32, u32)>, // (from, to)

    // Stats
    pub total_count: u64,
    pub create_count: u64,
    pub prune_count: u64,
    local_chunk_synapses: u64,
    cross_chunk_synapses: u64,
}

impl SynapsePool {
    /// Create an empty synapse pool for `num_neurons` total neurons.
    pub fn new(num_neurons: u32) -> Self {
        Self::new_with_chunks(num_neurons, rayon::current_num_threads())
    }

    /// Create an empty synapse pool with an explicit chunk count.
    pub fn new_with_chunks(num_neurons: u32, chunk_count: usize) -> Self {
        let n = num_neurons as usize;
        let chunk_count = normalized_chunk_count(num_neurons, chunk_count);
        let chunk_size = chunk_size_for(num_neurons, chunk_count);
        let chunk_layout = select_chunk_layout(num_neurons, chunk_count);
        Self {
            num_neurons,
            chunk_count,
            chunk_size,
            chunk_layout,
            targets: Vec::new(),
            weights: Vec::new(),
            delays: Vec::new(),
            plasticity: Vec::new(),
            target_chunks: Vec::new(),
            offsets: vec![0; n + 1],
            pending_updates: Vec::new(),
            pending_creates: Vec::new(),
            pending_prunes: Vec::new(),
            total_count: 0,
            create_count: 0,
            prune_count: 0,
            local_chunk_synapses: 0,
            cross_chunk_synapses: 0,
        }
    }

    /// Build CSR from a list of synapses. Synapses do NOT need to be sorted.
    pub fn from_synapses(num_neurons: u32, mut synapses: Vec<SynapseData>) -> Self {
        Self::from_synapses_with_chunks(num_neurons, synapses, rayon::current_num_threads())
    }

    /// Build CSR from a list of synapses with an explicit chunk count.
    pub fn from_synapses_with_chunks(
        num_neurons: u32,
        mut synapses: Vec<SynapseData>,
        chunk_count: usize,
    ) -> Self {
        // Sort by source neuron for CSR layout
        synapses.sort_by_key(|s| s.from);

        let n = num_neurons as usize;
        let total = synapses.len();
        let chunk_count = normalized_chunk_count(num_neurons, chunk_count);
        let chunk_size = chunk_size_for(num_neurons, chunk_count);
        let chunk_layout = select_chunk_layout(num_neurons, chunk_count);

        let mut targets = Vec::with_capacity(total);
        let mut weights = Vec::with_capacity(total);
        let mut delays = Vec::with_capacity(total);
        let mut plasticity = Vec::with_capacity(total);
        let mut target_chunks = Vec::with_capacity(total);
        let mut offsets = vec![0u32; n + 1];
        let mut local_chunk_synapses = 0u64;
        let mut cross_chunk_synapses = 0u64;

        // Count synapses per source neuron
        for s in &synapses {
            if (s.from as usize) < n {
                offsets[s.from as usize + 1] += 1;
            }
        }

        // Prefix sum to get offsets
        for i in 1..=n {
            offsets[i] += offsets[i - 1];
        }

        // Fill arrays
        for s in &synapses {
            targets.push(s.to);
            weights.push(s.weight);
            delays.push(s.delay);
            plasticity.push(s.plasticity);

            let src_chunk = chunk_for_neuron_with(
                s.from,
                num_neurons,
                chunk_size,
                chunk_count,
                chunk_layout,
            );
            let tgt_chunk = chunk_for_neuron_with(
                s.to,
                num_neurons,
                chunk_size,
                chunk_count,
                chunk_layout,
            );
            target_chunks.push(tgt_chunk as u16);
            if src_chunk == tgt_chunk {
                local_chunk_synapses += 1;
            } else {
                cross_chunk_synapses += 1;
            }
        }

        Self {
            num_neurons,
            chunk_count,
            chunk_size,
            chunk_layout,
            targets,
            weights,
            delays,
            plasticity,
            target_chunks,
            offsets,
            pending_updates: Vec::new(),
            pending_creates: Vec::new(),
            pending_prunes: Vec::new(),
            total_count: total as u64,
            create_count: 0,
            prune_count: 0,
            local_chunk_synapses,
            cross_chunk_synapses,
        }
    }

    /// Number of synapses currently in the pool.
    #[inline]
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Number of neuron slots (max global_id + 1).
    #[inline]
    pub fn num_neurons(&self) -> u32 {
        self.num_neurons
    }

    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    #[inline]
    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    #[inline]
    pub fn chunk_for_neuron(&self, global_id: u32) -> usize {
        chunk_for_neuron_with(
            global_id,
            self.num_neurons,
            self.chunk_size,
            self.chunk_count,
            self.chunk_layout,
        )
    }

    pub fn chunk_segments(&self, chunk_id: usize) -> Vec<ChunkSegment> {
        if chunk_id >= self.chunk_count {
            return Vec::new();
        }

        match self.chunk_layout {
            ChunkLayout::Contiguous => {
                let start_gid = chunk_id as u32 * self.chunk_size;
                let end_gid = (start_gid + self.chunk_size).min(self.num_neurons);
                let mut segments = Vec::new();

                for &region_id in RegionId::ALL.iter() {
                    let (region_start, region_end_inclusive) = region_id.neuron_range();
                    let region_end = (region_end_inclusive + 1).min(self.num_neurons);
                    let segment_start = start_gid.max(region_start);
                    let segment_end = end_gid.min(region_end);

                    if segment_start < segment_end {
                        segments.push(ChunkSegment {
                            region_id,
                            global_start: segment_start,
                            local_start: segment_start - region_start,
                            len: segment_end - segment_start,
                        });
                    }
                }

                segments
            }
            ChunkLayout::CanonicalRegionBuckets => {
                let mut segments = Vec::with_capacity(RegionId::ALL.len());

                for &region_id in RegionId::ALL.iter() {
                    let (region_start, _) = region_id.neuron_range();
                    let region_size = region_id.neuron_count();
                    let Some((bucket_start, bucket_end)) =
                        bucket_bounds(region_size, chunk_id, self.chunk_count)
                    else {
                        continue;
                    };

                    segments.push(ChunkSegment {
                        region_id,
                        global_start: region_start + bucket_start,
                        local_start: bucket_start,
                        len: bucket_end - bucket_start,
                    });
                }

                segments
            }
        }
    }

    /// Iterate over outgoing synapses for a given source neuron.
    /// Returns (target_id, weight, delay) tuples.
    #[inline]
    pub fn outgoing(&self, from: u32) -> &[u32] {
        if from >= self.num_neurons {
            return &[];
        }
        let start = self.offsets[from as usize] as usize;
        let end = self.offsets[from as usize + 1] as usize;
        &self.targets[start..end]
    }

    /// Get weight and delay for all outgoing synapses of a source neuron.
    /// Returns slices into (targets, weights, delays) for the given source.
    #[inline]
    pub fn outgoing_full(&self, from: u32) -> (&[u32], &[f32], &[u8]) {
        if from >= self.num_neurons {
            return (&[], &[], &[]);
        }
        let start = self.offsets[from as usize] as usize;
        let end = self.offsets[from as usize + 1] as usize;
        (
            &self.targets[start..end],
            &self.weights[start..end],
            &self.delays[start..end],
        )
    }

    /// Get target chunk IDs alongside the normal outgoing slices.
    #[inline]
    pub fn outgoing_full_with_chunks(&self, from: u32) -> (&[u32], &[f32], &[u8], &[u16]) {
        if from >= self.num_neurons {
            return (&[], &[], &[], &[]);
        }
        let start = self.offsets[from as usize] as usize;
        let end = self.offsets[from as usize + 1] as usize;
        (
            &self.targets[start..end],
            &self.weights[start..end],
            &self.delays[start..end],
            &self.target_chunks[start..end],
        )
    }

    /// Get weight of a specific synapse. O(degree of `from` neuron).
    pub fn get_weight(&self, from: u32, to: u32) -> Option<f32> {
        if from >= self.num_neurons {
            return None;
        }
        let start = self.offsets[from as usize] as usize;
        let end = self.offsets[from as usize + 1] as usize;
        for i in start..end {
            if self.targets[i] == to {
                return Some(self.weights[i]);
            }
        }
        None
    }

    /// Queue a weight update (applied during rebuild/apply).
    pub fn queue_update(&mut self, from: u32, to: u32, delta: f32) {
        self.pending_updates.push(SynapseUpdate { from, to, delta });
    }

    /// Queue a new synapse creation.
    pub fn queue_create(&mut self, synapse: SynapseData) {
        self.pending_creates.push(synapse);
    }

    /// Queue a synapse for pruning.
    pub fn queue_prune(&mut self, from: u32, to: u32) {
        self.pending_prunes.push((from, to));
    }

    /// Apply all pending weight updates without rebuilding CSR.
    /// This is cheap — just walks the update buffer.
    pub fn apply_weight_updates(&mut self) {
        for update in self.pending_updates.drain(..) {
            if update.from >= self.num_neurons {
                continue;
            }
            let start = self.offsets[update.from as usize] as usize;
            let end = self.offsets[update.from as usize + 1] as usize;
            for i in start..end {
                if self.targets[i] == update.to {
                    self.weights[i] = (self.weights[i] + update.delta).clamp(0.0, 1.0);
                    break;
                }
            }
        }
    }

    /// Full rebuild: apply creates, prunes, then reconstruct CSR.
    /// Expensive — call periodically (every 1000 ticks or when buffers are large).
    pub fn rebuild(&mut self, neuron_types: &dyn Fn(u32) -> NeuronType) {
        // First apply any remaining weight updates
        self.apply_weight_updates();

        // Collect all current synapses
        let mut all_synapses: Vec<SynapseData> = Vec::with_capacity(self.len());

        // Build prune set for O(1) lookup
        let prune_set: std::collections::HashSet<(u32, u32)> =
            self.pending_prunes.drain(..).collect();

        // Copy existing (minus pruned)
        for from in 0..self.num_neurons {
            let start = self.offsets[from as usize] as usize;
            let end = self.offsets[from as usize + 1] as usize;
            for i in start..end {
                let to = self.targets[i];
                if !prune_set.contains(&(from, to)) {
                    all_synapses.push(SynapseData {
                        from,
                        to,
                        weight: self.weights[i],
                        delay: self.delays[i],
                        plasticity: self.plasticity[i],
                    });
                } else {
                    self.prune_count += 1;
                }
            }
        }

        // Add new synapses
        for s in self.pending_creates.drain(..) {
            all_synapses.push(s);
            self.create_count += 1;
        }

        // Rebuild from scratch
        let _ = neuron_types; // for future Dale's Law enforcement
        let rebuilt = SynapsePool::from_synapses_with_chunks(
            self.num_neurons,
            all_synapses,
            self.chunk_count,
        );

        self.targets = rebuilt.targets;
        self.weights = rebuilt.weights;
        self.delays = rebuilt.delays;
        self.plasticity = rebuilt.plasticity;
        self.target_chunks = rebuilt.target_chunks;
        self.offsets = rebuilt.offsets;
        self.total_count = self.len() as u64;
        self.local_chunk_synapses = rebuilt.local_chunk_synapses;
        self.cross_chunk_synapses = rebuilt.cross_chunk_synapses;
    }

    /// How many pending modifications are queued.
    pub fn pending_count(&self) -> usize {
        self.pending_updates.len() + self.pending_creates.len() + self.pending_prunes.len()
    }

    /// Summary of chunk-local vs cross-chunk synapse ownership.
    pub fn chunk_stats(&self) -> (usize, u32, u64, u64, f64) {
        let total = self.local_chunk_synapses + self.cross_chunk_synapses;
        let cross_fraction = if total == 0 {
            0.0
        } else {
            self.cross_chunk_synapses as f64 / total as f64
        };
        (
            self.chunk_count,
            self.chunk_size,
            self.local_chunk_synapses,
            self.cross_chunk_synapses,
            cross_fraction,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool() -> SynapsePool {
        let synapses = vec![
            SynapseData { from: 0, to: 1, weight: 0.5, delay: 1, plasticity: 1.0 },
            SynapseData { from: 0, to: 2, weight: 0.3, delay: 1, plasticity: 1.0 },
            SynapseData { from: 1, to: 2, weight: 0.7, delay: 2, plasticity: 0.5 },
            SynapseData { from: 2, to: 0, weight: 0.2, delay: 1, plasticity: 1.0 },
        ];
        SynapsePool::from_synapses_with_chunks(10, synapses, 2)
    }

    #[test]
    fn test_csr_construction() {
        let pool = make_pool();
        assert_eq!(pool.len(), 4);

        // Neuron 0 has 2 outgoing
        let out0 = pool.outgoing(0);
        assert_eq!(out0.len(), 2);
        assert!(out0.contains(&1));
        assert!(out0.contains(&2));

        // Neuron 1 has 1 outgoing
        assert_eq!(pool.outgoing(1).len(), 1);
        assert_eq!(pool.outgoing(1)[0], 2);

        // Neuron 3 has 0 outgoing
        assert_eq!(pool.outgoing(3).len(), 0);
    }

    #[test]
    fn test_get_weight() {
        let pool = make_pool();
        assert_eq!(pool.get_weight(0, 1), Some(0.5));
        assert_eq!(pool.get_weight(0, 2), Some(0.3));
        assert_eq!(pool.get_weight(1, 2), Some(0.7));
        assert_eq!(pool.get_weight(0, 3), None); // doesn't exist
        assert_eq!(pool.get_weight(5, 0), None); // no synapses from 5
    }

    #[test]
    fn test_weight_update() {
        let mut pool = make_pool();
        pool.queue_update(0, 1, 0.1);
        pool.apply_weight_updates();
        assert!((pool.get_weight(0, 1).unwrap() - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_weight_clamp() {
        let mut pool = make_pool();
        pool.queue_update(0, 1, 0.8); // 0.5 + 0.8 = 1.3 → clamped to 1.0
        pool.apply_weight_updates();
        assert!((pool.get_weight(0, 1).unwrap() - 1.0).abs() < 0.001);

        pool.queue_update(2, 0, -0.5); // 0.2 - 0.5 = -0.3 → clamped to 0.0
        pool.apply_weight_updates();
        assert!((pool.get_weight(2, 0).unwrap() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_rebuild_with_create_and_prune() {
        let mut pool = make_pool();
        assert_eq!(pool.len(), 4);

        // Prune synapse 0→1
        pool.queue_prune(0, 1);

        // Add synapse 3→4
        pool.queue_create(SynapseData {
            from: 3, to: 4, weight: 0.4, delay: 1, plasticity: 1.0,
        });

        pool.rebuild(&|_| NeuronType::Excitatory);

        assert_eq!(pool.len(), 4); // 4 - 1 + 1 = 4
        assert_eq!(pool.get_weight(0, 1), None); // pruned
        assert_eq!(pool.get_weight(3, 4), Some(0.4)); // created
        assert_eq!(pool.get_weight(0, 2), Some(0.3)); // survived
    }

    #[test]
    fn test_outgoing_full() {
        let pool = make_pool();
        let (targets, weights, delays) = pool.outgoing_full(0);
        assert_eq!(targets.len(), 2);
        assert_eq!(weights.len(), 2);
        assert_eq!(delays.len(), 2);
    }

    #[test]
    fn test_chunk_classification_stats() {
        let synapses = vec![
            SynapseData { from: 0, to: 1, weight: 0.5, delay: 1, plasticity: 1.0 },
            SynapseData { from: 0, to: 9, weight: 0.3, delay: 1, plasticity: 1.0 },
            SynapseData { from: 9, to: 8, weight: 0.7, delay: 2, plasticity: 0.5 },
        ];
        let pool = SynapsePool::from_synapses_with_chunks(10, synapses, 2);
        let (chunk_count, chunk_size, local_count, cross_count, cross_fraction) = pool.chunk_stats();

        assert_eq!(chunk_count, 2);
        assert_eq!(chunk_size, 5);
        assert_eq!(local_count, 2);
        assert_eq!(cross_count, 1);
        assert!((cross_fraction - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_canonical_region_bucket_chunk_stats() {
        let synapses = vec![
            SynapseData { from: 100, to: 10_100, weight: 0.5, delay: 1, plasticity: 1.0 },
            SynapseData { from: 250, to: 11_600, weight: 0.3, delay: 1, plasticity: 1.0 },
            SynapseData { from: 70_100, to: 120_200, weight: 0.7, delay: 2, plasticity: 0.5 },
        ];
        let pool = SynapsePool::from_synapses_with_chunks(152_000, synapses, 16);
        let (_, _, local_count, cross_count, cross_fraction) = pool.chunk_stats();

        assert_eq!(pool.chunk_for_neuron(100), pool.chunk_for_neuron(10_100));
        assert_ne!(pool.chunk_for_neuron(250), pool.chunk_for_neuron(11_600));
        assert_eq!(local_count, 2);
        assert_eq!(cross_count, 1);
        assert!((cross_fraction - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_empty_pool() {
        let pool = SynapsePool::new(100);
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        assert_eq!(pool.outgoing(0).len(), 0);
        assert_eq!(pool.get_weight(0, 1), None);
    }
}
