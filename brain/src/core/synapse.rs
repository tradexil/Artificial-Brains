/// Synapse storage in Compressed Sparse Row (CSR) format.
///
/// CSR gives cache-friendly contiguous memory for 5M+ synapses.
/// Iteration over a neuron's outgoing connections is O(degree).
///
/// Modifications are batched: weight updates, creates, and prunes
/// go into pending buffers, then applied during rebuild.

use crate::core::neuron::NeuronType;
use crate::core::region::RegionId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(target_os = "windows")]
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
#[cfg(target_os = "windows")]
use windows_sys::Win32::System::Memory::{
    MapViewOfFile,
    MEMORY_MAPPED_VIEW_ADDRESS,
    OpenFileMappingW,
    UnmapViewOfFile,
    FILE_MAP_ALL_ACCESS,
};

const CANONICAL_TOTAL_NEURONS: u32 = 152_000;
const REGION_COUNT: usize = 14;
const DELAY_BUCKETS: usize = 256;
const WEIGHT_BYTES_PER_ENTRY: usize = std::mem::size_of::<f32>();

struct SharedWeightBuffer {
    len: usize,
    #[cfg(target_os = "windows")]
    handle: HANDLE,
    #[cfg(target_os = "windows")]
    view: MEMORY_MAPPED_VIEW_ADDRESS,
}

unsafe impl Send for SharedWeightBuffer {}
unsafe impl Sync for SharedWeightBuffer {}

impl SharedWeightBuffer {
    fn open(name: &str, len: usize, size_bytes: usize) -> Result<Self, String> {
        let expected_bytes = len.saturating_mul(WEIGHT_BYTES_PER_ENTRY);
        if size_bytes != expected_bytes {
            return Err(format!(
                "Shared weight buffer size {} does not match expected {} bytes",
                size_bytes, expected_bytes,
            ));
        }

        #[cfg(target_os = "windows")]
        {
            let wide_name = name
                .encode_utf16()
                .chain(std::iter::once(0))
                .collect::<Vec<u16>>();
            unsafe {
                let handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, wide_name.as_ptr());
                if handle.is_null() {
                    return Err(format!(
                        "Failed to open shared weight buffer '{}'",
                        name,
                    ));
                }

                let view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size_bytes);
                if view.Value.is_null() {
                    CloseHandle(handle);
                    return Err(format!(
                        "Failed to map shared weight buffer '{}'",
                        name,
                    ));
                }

                Ok(Self { len, handle, view })
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            let _ = (name, len, size_bytes);
            Err("Shared weight buffers are only supported on Windows".to_string())
        }
    }

    #[cfg(target_os = "windows")]
    #[inline]
    fn atomics(&self) -> &[AtomicU32] {
        unsafe { std::slice::from_raw_parts(self.view.Value.cast::<AtomicU32>(), self.len) }
    }

    #[inline]
    fn load(&self, index: usize) -> f32 {
        #[cfg(target_os = "windows")]
        {
            return f32::from_bits(self.atomics()[index].load(Ordering::Relaxed));
        }

        #[cfg(not(target_os = "windows"))]
        {
            let _ = index;
            unreachable!("shared weights are unsupported on this platform")
        }
    }

    #[inline]
    fn store(&self, index: usize, value: f32) {
        #[cfg(target_os = "windows")]
        {
            self.atomics()[index].store(value.to_bits(), Ordering::Relaxed);
            return;
        }

        #[cfg(not(target_os = "windows"))]
        {
            let _ = (index, value);
            unreachable!("shared weights are unsupported on this platform")
        }
    }

    #[inline]
    fn fetch_add_clamped(&self, index: usize, delta: f32) -> (f32, f32) {
        #[cfg(target_os = "windows")]
        {
            let atomic = &self.atomics()[index];
            let mut observed = atomic.load(Ordering::Relaxed);
            loop {
                let before = f32::from_bits(observed);
                let after = (before + delta).clamp(0.0, 1.0);
                match atomic.compare_exchange_weak(
                    observed,
                    after.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return (before, after),
                    Err(next_observed) => observed = next_observed,
                }
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            let _ = (index, delta);
            unreachable!("shared weights are unsupported on this platform")
        }
    }
}

impl Drop for SharedWeightBuffer {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        unsafe {
            if !self.view.Value.is_null() {
                UnmapViewOfFile(self.view);
            }
            if !self.handle.is_null() {
                CloseHandle(self.handle);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApplyWeightUpdatesProfile {
    pub pending_update_count: u64,
    pub deferred_update_count: u64,
    pub applied_update_count: u64,
    pub unmatched_update_count: u64,
    pub positive_update_count: u64,
    pub negative_update_count: u64,
    pub delta_sum: f64,
    pub delta_abs_sum: f64,
    pub delta_min: f32,
    pub delta_max: f32,
    pub before_weight_sum: f64,
    pub after_weight_sum: f64,
    pub crossed_up_0p05_count: u64,
    pub crossed_up_0p10_count: u64,
    pub crossed_up_0p20_count: u64,
    pub crossed_down_0p05_count: u64,
    pub crossed_down_0p10_count: u64,
    pub crossed_down_0p20_count: u64,
    pub region_pair_update_counts: [[u64; REGION_COUNT]; REGION_COUNT],
    pub region_pair_delta_abs_sums: [[f64; REGION_COUNT]; REGION_COUNT],
    pub region_pair_before_weight_sums: [[f64; REGION_COUNT]; REGION_COUNT],
    pub region_pair_after_weight_sums: [[f64; REGION_COUNT]; REGION_COUNT],
    pub delay_update_counts: [u64; DELAY_BUCKETS],
    pub delay_delta_abs_sums: [f64; DELAY_BUCKETS],
}

impl Default for ApplyWeightUpdatesProfile {
    fn default() -> Self {
        Self {
            pending_update_count: 0,
            deferred_update_count: 0,
            applied_update_count: 0,
            unmatched_update_count: 0,
            positive_update_count: 0,
            negative_update_count: 0,
            delta_sum: 0.0,
            delta_abs_sum: 0.0,
            delta_min: f32::INFINITY,
            delta_max: f32::NEG_INFINITY,
            before_weight_sum: 0.0,
            after_weight_sum: 0.0,
            crossed_up_0p05_count: 0,
            crossed_up_0p10_count: 0,
            crossed_up_0p20_count: 0,
            crossed_down_0p05_count: 0,
            crossed_down_0p10_count: 0,
            crossed_down_0p20_count: 0,
            region_pair_update_counts: [[0u64; REGION_COUNT]; REGION_COUNT],
            region_pair_delta_abs_sums: [[0.0f64; REGION_COUNT]; REGION_COUNT],
            region_pair_before_weight_sums: [[0.0f64; REGION_COUNT]; REGION_COUNT],
            region_pair_after_weight_sums: [[0.0f64; REGION_COUNT]; REGION_COUNT],
            delay_update_counts: [0u64; DELAY_BUCKETS],
            delay_delta_abs_sums: [0.0f64; DELAY_BUCKETS],
        }
    }
}

impl ApplyWeightUpdatesProfile {
    fn finalize(&mut self) {
        if self.applied_update_count == 0 {
            self.delta_min = 0.0;
            self.delta_max = 0.0;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SynapseData {
    pub from: u32,
    pub to: u32,
    pub weight: f32,
    pub delay: u8,
    pub plasticity: f32,
}

/// Queued weight modification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SynapseUpdate {
    pub from: u32,
    pub to: u32,
    pub delta: f32,
    pub synapse_index: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SynapseRuntimeState {
    pending_updates: Vec<SynapseUpdate>,
    pending_creates: Vec<SynapseData>,
    pending_prunes: Vec<(u32, u32)>,
    total_count: u64,
    create_count: u64,
    prune_count: u64,
    local_chunk_synapses: u64,
    cross_chunk_synapses: u64,
}

/// CSR synapse pool — the main synapse storage.
#[derive(Serialize, Deserialize)]
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

    #[serde(skip)]
    round_delta_baselines: HashMap<u32, f32>,
    #[serde(skip)]
    round_delta_effective: HashMap<u32, f32>,
    #[serde(skip)]
    shared_weights: Option<SharedWeightBuffer>,

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
            round_delta_baselines: HashMap::new(),
            round_delta_effective: HashMap::new(),
            shared_weights: None,
            total_count: 0,
            create_count: 0,
            prune_count: 0,
            local_chunk_synapses: 0,
            cross_chunk_synapses: 0,
        }
    }

    /// Build CSR from a list of synapses. Synapses do NOT need to be sorted.
    pub fn from_synapses(num_neurons: u32, synapses: Vec<SynapseData>) -> Self {
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
            round_delta_baselines: HashMap::new(),
            round_delta_effective: HashMap::new(),
            shared_weights: None,
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

    #[inline]
    pub fn shared_weights_attached(&self) -> bool {
        self.shared_weights.is_some()
    }

    pub fn attach_shared_weight_buffer(&mut self, name: &str, size_bytes: usize) -> Result<(), String> {
        self.shared_weights = Some(SharedWeightBuffer::open(name, self.weights.len(), size_bytes)?);
        Ok(())
    }

    pub fn refresh_owned_weights_from_shared(&mut self) {
        if let Some(shared_weights) = &self.shared_weights {
            for (index, weight) in self.weights.iter_mut().enumerate() {
                *weight = shared_weights.load(index);
            }
        }
    }

    pub fn copy_weights_to_slice(&self, target: &mut [f32]) -> Result<(), String> {
        if target.len() != self.weights.len() {
            return Err(format!(
                "Synapse weight buffer length {} does not match synapse count {}",
                target.len(),
                self.weights.len(),
            ));
        }

        if let Some(shared_weights) = &self.shared_weights {
            for (index, slot) in target.iter_mut().enumerate() {
                *slot = shared_weights.load(index);
            }
        } else {
            target.copy_from_slice(&self.weights);
        }

        Ok(())
    }

    pub fn overwrite_weights_from_slice(&mut self, source: &[f32]) -> Result<(), String> {
        if source.len() != self.weights.len() {
            return Err(format!(
                "Synapse weight buffer length {} does not match synapse count {}",
                source.len(),
                self.weights.len(),
            ));
        }

        self.weights.copy_from_slice(source);
        if let Some(shared_weights) = &self.shared_weights {
            for (index, weight) in source.iter().enumerate() {
                shared_weights.store(index, *weight);
            }
        }
        Ok(())
    }

    /// Multiplicatively decay all synapse weights towards a floor.
    ///
    /// Each weight becomes `max(weight * factor, floor_weight)`.
    /// Factor should be in (0, 1), e.g. 0.95 for a 5% per-call decay.
    /// Weights are clamped at `floor_weight` to preserve structural
    /// connectivity from initial wiring.
    pub fn decay_all_weights(&mut self, factor: f32, floor_weight: f32) -> u64 {
        let mut clamped: u64 = 0;
        if let Some(shared_weights) = &self.shared_weights {
            for i in 0..self.weights.len() {
                let old_w = shared_weights.load(i);
                if old_w <= floor_weight {
                    continue; // Already at or below floor — skip
                }
                let new_w = (old_w * factor).max(floor_weight);
                if new_w <= floor_weight {
                    clamped += 1;
                }
                shared_weights.store(i, new_w);
                self.weights[i] = new_w;
            }
        } else {
            for w in self.weights.iter_mut() {
                if *w <= floor_weight {
                    continue;
                }
                *w = (*w * factor).max(floor_weight);
                if *w <= floor_weight {
                    clamped += 1;
                }
            }
        }
        clamped
    }

    #[inline]
    pub fn weight_at_index(&self, synapse_index: usize) -> Option<f32> {
        if synapse_index >= self.weights.len() {
            return None;
        }
        if let Some(shared_weights) = &self.shared_weights {
            Some(shared_weights.load(synapse_index))
        } else {
            Some(self.weights[synapse_index])
        }
    }

    #[inline]
    pub fn atomic_add_weight_at_index(&self, synapse_index: usize, delta: f32) -> Option<(f32, f32)> {
        if synapse_index >= self.weights.len() || delta.abs() <= f32::EPSILON {
            return None;
        }
        self.shared_weights
            .as_ref()
            .map(|shared_weights| shared_weights.fetch_add_clamped(synapse_index, delta))
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
                return self.weight_at_index(i);
            }
        }
        None
    }

    /// Queue a weight update (applied during rebuild/apply).
    pub fn queue_update(&mut self, from: u32, to: u32, delta: f32) {
        self.pending_updates.push(SynapseUpdate {
            from,
            to,
            delta,
            synapse_index: u32::MAX,
        });
    }

    /// Queue a weight update for a known CSR synapse index.
    pub fn queue_indexed_update(&mut self, from: u32, to: u32, synapse_index: u32, delta: f32) {
        self.pending_updates.push(SynapseUpdate {
            from,
            to,
            delta,
            synapse_index,
        });
    }

    pub fn pending_update_count(&self) -> usize {
        self.pending_updates.len()
    }

    pub fn reset_round_delta_tracker(&mut self) {
        self.round_delta_baselines.clear();
        self.round_delta_effective.clear();
    }

    pub fn take_round_deltas(&mut self) -> Vec<(u32, f32)> {
        let mut deltas = self
            .round_delta_effective
            .iter()
            .map(|(&synapse_index, &delta)| (synapse_index, delta))
            .collect::<Vec<_>>();
        deltas.sort_unstable_by_key(|(synapse_index, _)| *synapse_index);
        self.reset_round_delta_tracker();
        deltas
    }

    pub fn apply_sparse_deltas_by_index(&mut self, deltas: &[(u32, f32)]) -> usize {
        let mut applied = 0usize;
        for &(synapse_index, delta) in deltas {
            let idx = synapse_index as usize;
            if idx >= self.weights.len() {
                continue;
            }
            if delta.abs() <= f32::EPSILON {
                continue;
            }
            if let Some(shared_weights) = &self.shared_weights {
                let _ = shared_weights.fetch_add_clamped(idx, delta);
            } else {
                self.weights[idx] = (self.weights[idx] + delta).clamp(0.0, 1.0);
            }
            applied += 1;
        }
        applied
    }

    fn record_round_delta(&mut self, synapse_index: usize, before_weight: f32, after_weight: f32) {
        let synapse_index = synapse_index as u32;
        let baseline = *self
            .round_delta_baselines
            .entry(synapse_index)
            .or_insert(before_weight);
        let effective_delta = after_weight - baseline;
        if effective_delta.abs() <= f32::EPSILON {
            self.round_delta_baselines.remove(&synapse_index);
            self.round_delta_effective.remove(&synapse_index);
        } else {
            self.round_delta_effective
                .insert(synapse_index, effective_delta);
        }
    }

    fn apply_weight_update_at_index(
        &mut self,
        synapse_index: usize,
        from: u32,
        delta: f32,
        profile: &mut ApplyWeightUpdatesProfile,
    ) {
        profile.applied_update_count += 1;

        if delta > 0.0 {
            profile.positive_update_count += 1;
        } else if delta < 0.0 {
            profile.negative_update_count += 1;
        }

        let (before_weight, after_weight) = if let Some(shared_weights) = &self.shared_weights {
            shared_weights.fetch_add_clamped(synapse_index, delta)
        } else {
            let before_weight = self.weights[synapse_index];
            let after_weight = (before_weight + delta).clamp(0.0, 1.0);
            self.weights[synapse_index] = after_weight;
            (before_weight, after_weight)
        };
        let delta_abs = delta.abs() as f64;

        profile.delta_sum += delta as f64;
        profile.delta_abs_sum += delta_abs;
        profile.delta_min = profile.delta_min.min(delta);
        profile.delta_max = profile.delta_max.max(delta);
        profile.before_weight_sum += before_weight as f64;
        profile.after_weight_sum += after_weight as f64;

        if before_weight < 0.05 && after_weight >= 0.05 {
            profile.crossed_up_0p05_count += 1;
        }
        if before_weight < 0.10 && after_weight >= 0.10 {
            profile.crossed_up_0p10_count += 1;
        }
        if before_weight < 0.20 && after_weight >= 0.20 {
            profile.crossed_up_0p20_count += 1;
        }
        if before_weight >= 0.05 && after_weight < 0.05 {
            profile.crossed_down_0p05_count += 1;
        }
        if before_weight >= 0.10 && after_weight < 0.10 {
            profile.crossed_down_0p10_count += 1;
        }
        if before_weight >= 0.20 && after_weight < 0.20 {
            profile.crossed_down_0p20_count += 1;
        }

        let target = self.targets[synapse_index];
        if let (Some(src_region), Some(tgt_region)) = (
            RegionId::from_neuron_id(from),
            RegionId::from_neuron_id(target),
        ) {
            let src_idx = src_region.index();
            let tgt_idx = tgt_region.index();
            profile.region_pair_update_counts[src_idx][tgt_idx] += 1;
            profile.region_pair_delta_abs_sums[src_idx][tgt_idx] += delta_abs;
            profile.region_pair_before_weight_sums[src_idx][tgt_idx] += before_weight as f64;
            profile.region_pair_after_weight_sums[src_idx][tgt_idx] += after_weight as f64;
        }

        let delay_idx = self.delays[synapse_index] as usize;
        profile.delay_update_counts[delay_idx] += 1;
        profile.delay_delta_abs_sums[delay_idx] += delta_abs;

        self.record_round_delta(synapse_index, before_weight, after_weight);
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
        let _ = self.apply_weight_updates_profiled();
    }

    pub fn apply_weight_updates_profiled(&mut self) -> ApplyWeightUpdatesProfile {
        self.apply_weight_updates_profiled_bounded(usize::MAX)
    }

    pub fn apply_weight_updates_profiled_bounded(
        &mut self,
        max_updates: usize,
    ) -> ApplyWeightUpdatesProfile {
        let mut profile = ApplyWeightUpdatesProfile::default();
        let total_pending = self.pending_updates.len();
        profile.pending_update_count = total_pending as u64;

        if total_pending == 0 {
            profile.finalize();
            return profile;
        }

        if max_updates == 0 {
            profile.deferred_update_count = total_pending as u64;
            profile.finalize();
            return profile;
        }

        let apply_count = total_pending.min(max_updates);
        let deferred_updates = if apply_count < total_pending {
            self.pending_updates.split_off(apply_count)
        } else {
            Vec::new()
        };
        let pending_updates = std::mem::take(&mut self.pending_updates);
        self.pending_updates = deferred_updates;
        profile.deferred_update_count = self.pending_updates.len() as u64;

        for update in pending_updates {
            if update.from >= self.num_neurons {
                profile.unmatched_update_count += 1;
                continue;
            }

            let start = self.offsets[update.from as usize] as usize;
            let end = self.offsets[update.from as usize + 1] as usize;
            let mut matched = false;

            if update.synapse_index != u32::MAX {
                let synapse_index = update.synapse_index as usize;
                if synapse_index >= start
                    && synapse_index < end
                    && synapse_index < self.targets.len()
                    && self.targets[synapse_index] == update.to
                {
                    matched = true;
                    self.apply_weight_update_at_index(
                        synapse_index,
                        update.from,
                        update.delta,
                        &mut profile,
                    );
                }
            }

            if matched {
                continue;
            }

            for i in start..end {
                if self.targets[i] != update.to {
                    continue;
                }

                matched = true;
                self.apply_weight_update_at_index(i, update.from, update.delta, &mut profile);
                break;
            }

            if !matched {
                profile.unmatched_update_count += 1;
            }
        }

        profile.finalize();
        profile
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
                        weight: self.weight_at_index(i).unwrap_or(0.0),
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
    self.shared_weights = None;
        self.total_count = self.len() as u64;
        self.local_chunk_synapses = rebuilt.local_chunk_synapses;
        self.cross_chunk_synapses = rebuilt.cross_chunk_synapses;
    }

    /// How many pending modifications are queued.
    pub fn pending_count(&self) -> usize {
        self.pending_updates.len() + self.pending_creates.len() + self.pending_prunes.len()
    }

    pub fn export_runtime_state(&self) -> SynapseRuntimeState {
        SynapseRuntimeState {
            pending_updates: self.pending_updates.clone(),
            pending_creates: self.pending_creates.clone(),
            pending_prunes: self.pending_prunes.clone(),
            total_count: self.total_count,
            create_count: self.create_count,
            prune_count: self.prune_count,
            local_chunk_synapses: self.local_chunk_synapses,
            cross_chunk_synapses: self.cross_chunk_synapses,
        }
    }

    pub fn apply_runtime_state(&mut self, runtime_state: SynapseRuntimeState) {
        self.pending_updates = runtime_state.pending_updates;
        self.pending_creates = runtime_state.pending_creates;
        self.pending_prunes = runtime_state.pending_prunes;
        self.reset_round_delta_tracker();
        self.total_count = runtime_state.total_count;
        self.create_count = runtime_state.create_count;
        self.prune_count = runtime_state.prune_count;
        self.local_chunk_synapses = runtime_state.local_chunk_synapses;
        self.cross_chunk_synapses = runtime_state.cross_chunk_synapses;
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
    fn test_apply_weight_updates_profiled_reports_region_pairs_and_threshold_crossings() {
        let synapses = vec![
            SynapseData { from: 55_000, to: 55_001, weight: 0.04, delay: 9, plasticity: 1.0 },
            SynapseData { from: 70_000, to: 70_001, weight: 0.20, delay: 1, plasticity: 1.0 },
        ];
        let mut pool = SynapsePool::from_synapses_with_chunks(152_000, synapses, 1);

        pool.queue_update(55_000, 55_001, 0.02);
        pool.queue_update(70_000, 70_001, -0.16);

        let profile = pool.apply_weight_updates_profiled();

        assert_eq!(profile.pending_update_count, 2);
        assert_eq!(profile.deferred_update_count, 0);
        assert_eq!(profile.applied_update_count, 2);
        assert_eq!(profile.unmatched_update_count, 0);
        assert_eq!(profile.crossed_up_0p05_count, 1);
        assert_eq!(profile.crossed_down_0p10_count, 1);
        assert_eq!(profile.delay_update_counts[9], 1);
        assert_eq!(
            profile.region_pair_update_counts[RegionId::MemoryLong.index()][RegionId::MemoryLong.index()],
            1,
        );
        assert_eq!(
            profile.region_pair_update_counts[RegionId::Emotion.index()][RegionId::Emotion.index()],
            1,
        );
        assert!((pool.get_weight(55_000, 55_001).unwrap() - 0.06).abs() < 0.001);
        assert!((pool.get_weight(70_000, 70_001).unwrap() - 0.04).abs() < 0.001);
    }

    #[test]
    fn test_apply_weight_updates_profiled_bounded_defers_tail_updates() {
        let synapses = vec![
            SynapseData { from: 0, to: 1, weight: 0.10, delay: 1, plasticity: 1.0 },
            SynapseData { from: 0, to: 2, weight: 0.10, delay: 1, plasticity: 1.0 },
            SynapseData { from: 1, to: 2, weight: 0.10, delay: 1, plasticity: 1.0 },
        ];
        let mut pool = SynapsePool::from_synapses_with_chunks(10, synapses, 1);

        pool.queue_update(0, 1, 0.05);
        pool.queue_update(0, 2, 0.05);
        pool.queue_update(1, 2, 0.05);

        let profile = pool.apply_weight_updates_profiled_bounded(2);

        assert_eq!(profile.pending_update_count, 3);
        assert_eq!(profile.applied_update_count, 2);
        assert_eq!(profile.deferred_update_count, 1);
        assert!((pool.get_weight(0, 1).unwrap() - 0.15).abs() < 0.001);
        assert!((pool.get_weight(0, 2).unwrap() - 0.15).abs() < 0.001);
        assert!((pool.get_weight(1, 2).unwrap() - 0.10).abs() < 0.001);

        let second = pool.apply_weight_updates_profiled_bounded(10);
        assert_eq!(second.pending_update_count, 1);
        assert_eq!(second.applied_update_count, 1);
        assert_eq!(second.deferred_update_count, 0);
        assert!((pool.get_weight(1, 2).unwrap() - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_round_delta_tracker_exports_effective_sparse_deltas() {
        let mut pool = make_pool();

        pool.reset_round_delta_tracker();
        pool.queue_indexed_update(0, 1, 0, 0.1);
        pool.queue_indexed_update(0, 1, 0, 0.1);
        pool.queue_indexed_update(2, 0, 3, -0.05);
        pool.apply_weight_updates();

        let deltas = pool.take_round_deltas();

        assert_eq!(deltas.len(), 2);
        assert_eq!(deltas[0].0, 0);
        assert!((deltas[0].1 - 0.2).abs() < 0.001);
        assert_eq!(deltas[1].0, 3);
        assert!((deltas[1].1 + 0.05).abs() < 0.001);
        assert!(pool.take_round_deltas().is_empty());
    }

    #[test]
    fn test_apply_sparse_deltas_by_index_updates_weights() {
        let mut pool = make_pool();

        let applied = pool.apply_sparse_deltas_by_index(&[(0, 0.1), (3, -0.3), (99, 0.2)]);

        assert_eq!(applied, 2);
        assert!((pool.get_weight(0, 1).unwrap() - 0.6).abs() < 0.001);
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
