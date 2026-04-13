/// Signal propagation: the hot loop of the brain.
///
/// For each active neuron, traverse its outgoing synapses and
/// deposit weighted signal into target neurons' incoming buffers.
/// Respects synapse delay via a delay buffer ring.
///
/// This is the most performance-critical code — runs every tick.

use crate::core::neuron::NeuronType;
use crate::core::region::{LaneId, Region, RegionId};
use crate::core::synapse::{ChunkSegment, SynapsePool};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
struct RoutedSignal {
    source_region_idx: u8,
    region_idx: u8,
    local_idx: u32,
    value: f32,
    delayed: bool,
}

struct LocalIncomingBuffer {
    region_idx: usize,
    local_start: u32,
    incoming: Vec<f32>,
    immediate_same: Vec<f32>,
    immediate_cross: Vec<f32>,
}

struct ChunkOutput {
    local_buffers: Vec<LocalIncomingBuffer>,
    immediate_mailbox: Vec<RoutedSignal>,
    delayed_mailboxes: Vec<Vec<RoutedSignal>>,
    stats: PropagationStats,
}

struct WaveChunkOutput {
    local_buffers: Vec<LocalIncomingBuffer>,
    immediate_mailbox: Vec<RoutedSignal>,
    scheduled_mailboxes: Vec<Vec<Vec<RoutedSignal>>>,
    stats: PropagationStats,
}

#[derive(Debug, Clone, Default)]
pub struct DeliveryStats {
    pub signal_count: u64,
    pub signal_abs_sum: f64,
    pub target_signal_counts: [u64; 14],
    pub target_signal_abs_sums: [f64; 14],
}

impl DeliveryStats {
    #[inline]
    fn record(&mut self, region_idx: usize, value: f32) {
        let signal_abs = value.abs() as f64;
        self.signal_count += 1;
        self.signal_abs_sum += signal_abs;
        self.target_signal_counts[region_idx] += 1;
        self.target_signal_abs_sums[region_idx] += signal_abs;
    }
}

#[derive(Debug, Clone, Default)]
pub struct PropagationStats {
    pub immediate_signal_count: u64,
    pub immediate_signal_abs_sum: f64,
    pub scheduled_delayed_signal_count: u64,
    pub scheduled_delayed_signal_abs_sum: f64,
    pub target_signal_counts: [u64; 14],
    pub target_signal_abs_sums: [f64; 14],
    pub delayed_flow_signal_counts: [[u64; 14]; 14],
    pub delayed_flow_signal_abs_sums: [[f64; 14]; 14],
}

impl PropagationStats {
    #[inline]
    fn record_immediate(&mut self, region_idx: usize, value: f32) {
        let signal_abs = value.abs() as f64;
        self.immediate_signal_count += 1;
        self.immediate_signal_abs_sum += signal_abs;
        self.target_signal_counts[region_idx] += 1;
        self.target_signal_abs_sums[region_idx] += signal_abs;
    }

    #[inline]
    fn record_delayed_schedule(&mut self, value: f32) {
        self.scheduled_delayed_signal_count += 1;
        self.scheduled_delayed_signal_abs_sum += value.abs() as f64;
    }

    #[inline]
    fn record_delayed_flow(&mut self, source_region_idx: usize, target_region_idx: usize, value: f32) {
        let signal_abs = value.abs() as f64;
        self.delayed_flow_signal_counts[source_region_idx][target_region_idx] += 1;
        self.delayed_flow_signal_abs_sums[source_region_idx][target_region_idx] += signal_abs;
    }

    fn merge(&mut self, other: &PropagationStats) {
        self.immediate_signal_count += other.immediate_signal_count;
        self.immediate_signal_abs_sum += other.immediate_signal_abs_sum;
        self.scheduled_delayed_signal_count += other.scheduled_delayed_signal_count;
        self.scheduled_delayed_signal_abs_sum += other.scheduled_delayed_signal_abs_sum;
        for idx in 0..RegionId::ALL.len() {
            self.target_signal_counts[idx] += other.target_signal_counts[idx];
            self.target_signal_abs_sums[idx] += other.target_signal_abs_sums[idx];
            for target_idx in 0..RegionId::ALL.len() {
                self.delayed_flow_signal_counts[idx][target_idx] +=
                    other.delayed_flow_signal_counts[idx][target_idx];
                self.delayed_flow_signal_abs_sums[idx][target_idx] +=
                    other.delayed_flow_signal_abs_sums[idx][target_idx];
            }
        }
    }
}

/// Maximum synapse delay in ticks.
pub const MAX_DELAY: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SameRegionDelayAblation {
    region_mask: [bool; RegionId::ALL.len()],
    delay_mask: [bool; MAX_DELAY + 1],
}

impl Default for SameRegionDelayAblation {
    fn default() -> Self {
        Self {
            region_mask: [false; RegionId::ALL.len()],
            delay_mask: [false; MAX_DELAY + 1],
        }
    }
}

impl SameRegionDelayAblation {
    pub fn clear(&mut self) {
        self.region_mask.fill(false);
        self.delay_mask.fill(false);
    }

    pub fn configure(&mut self, regions: &[RegionId], delays: &[u8]) {
        self.clear();
        for &region_id in regions {
            self.region_mask[region_id.index()] = true;
        }
        for &delay in delays {
            if let Some(enabled) = self.delay_mask.get_mut(delay as usize) {
                *enabled = true;
            }
        }
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.region_mask.iter().any(|&enabled| enabled)
            && self.delay_mask.iter().any(|&enabled| enabled)
    }

    #[inline]
    pub fn suppresses(&self, source_region_idx: usize, target_region_idx: usize, raw_delay: u8) -> bool {
        source_region_idx == target_region_idx
            && self
                .region_mask
                .get(source_region_idx)
                .copied()
                .unwrap_or(false)
            && self
                .delay_mask
                .get(raw_delay as usize)
                .copied()
                .unwrap_or(false)
    }
}

/// Ring buffer for delayed signal delivery.
/// Stores (target_global_id, signal_value) for each future tick.
#[derive(Clone, Serialize, Deserialize)]
pub struct DelayBuffer {
    /// Ring of per-base-tick signal buffers. Each slot contains one mailbox per wave.
    slots: Vec<Vec<Vec<RoutedSignal>>>,
    current_slot: usize,
}

impl DelayBuffer {
    pub fn new() -> Self {
        Self {
            slots: (0..=MAX_DELAY)
                .map(|_| {
                    (0..LaneId::ALL.len())
                        .map(|_| Vec::new())
                        .collect::<Vec<_>>()
                })
                .collect(),
            current_slot: 0,
        }
    }

    #[inline]
    fn clamp_offset(&self, offset: usize) -> usize {
        offset.min(self.slots.len().saturating_sub(1))
    }

    /// Schedule a signal to be delivered `delay` ticks from now.
    #[inline]
    pub fn schedule(&mut self, target: u32, value: f32, delay: u8) {
        let Some((region_idx, local_idx)) = region_index_local(target) else {
            return;
        };
        self.schedule_local_wave(
            region_idx,
            region_idx,
            local_idx,
            value,
            delay as usize,
            LaneId::Perception,
            true,
        );
    }

    pub fn schedule_batch(&mut self, entries: &[(u32, f32, u8)]) {
        if entries.is_empty() {
            return;
        }

        for &(target, value, delay) in entries {
            self.schedule(target, value, delay);
        }
    }

    #[inline]
    fn schedule_local(
        &mut self,
        source_region_idx: usize,
        region_idx: usize,
        local_idx: u32,
        value: f32,
        delay: u8,
    ) {
        self.schedule_local_wave(
            source_region_idx,
            region_idx,
            local_idx,
            value,
            delay as usize,
            LaneId::Perception,
            true,
        );
    }

    #[inline]
    fn schedule_local_wave(
        &mut self,
        source_region_idx: usize,
        region_idx: usize,
        local_idx: u32,
        value: f32,
        base_tick_offset: usize,
        ready_wave: LaneId,
        delayed: bool,
    ) {
        let slot = (self.current_slot + self.clamp_offset(base_tick_offset)) % self.slots.len();
        self.slots[slot][ready_wave.index()].push(RoutedSignal {
            source_region_idx: source_region_idx as u8,
            region_idx: region_idx as u8,
            local_idx,
            value,
            delayed,
        });
    }

    fn extend_relative_slot(&mut self, delay: usize, entries: Vec<RoutedSignal>) {
        self.extend_relative_slot_wave(delay, LaneId::Perception, entries);
    }

    fn extend_relative_slot_wave(
        &mut self,
        delay: usize,
        ready_wave: LaneId,
        mut entries: Vec<RoutedSignal>,
    ) {
        if entries.is_empty() {
            return;
        }
        let slot = (self.current_slot + self.clamp_offset(delay)) % self.slots.len();
        self.slots[slot][ready_wave.index()].append(&mut entries);
    }

    fn drain_current_local(&mut self) -> Vec<RoutedSignal> {
        self.drain_current_wave_local(LaneId::Perception)
    }

    fn drain_current_wave_local(&mut self, wave: LaneId) -> Vec<RoutedSignal> {
        std::mem::take(&mut self.slots[self.current_slot][wave.index()])
    }

    pub fn carry_current_wave_to_next(&mut self, wave: LaneId) {
        let mut entries = self.drain_current_wave_local(wave);
        if entries.is_empty() {
            return;
        }
        mark_signals_delayed(&mut entries);
        self.extend_relative_slot_wave(1, wave, entries);
    }

    /// Get all signals due for delivery this tick, and clear the slot.
    pub fn drain_current(&mut self) -> Vec<(u32, f32)> {
        self.drain_current_local()
            .into_iter()
            .map(|signal| {
                (
                    RegionId::ALL[signal.region_idx as usize].neuron_range().0 + signal.local_idx,
                    signal.value,
                )
            })
            .collect()
    }

    /// Advance to next tick.
    pub fn advance(&mut self) {
        self.current_slot = (self.current_slot + 1) % self.slots.len();
    }
}

/// Determine which region a global neuron ID belongs to.
/// Uses the known region boundaries for O(1) lookup.
#[inline]
pub fn region_for_neuron(global_id: u32) -> Option<RegionId> {
    region_index_local(global_id).map(|(idx, _)| RegionId::ALL[idx])
}

#[inline]
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

#[inline]
fn region_index_local_in_slice(regions: &[Region], global_id: u32) -> Option<(usize, u32)> {
    let (canonical_idx, local_idx) = region_index_local(global_id)?;

    if regions.len() == RegionId::ALL.len()
        && regions
            .get(canonical_idx)
            .map(|region| region.id == RegionId::ALL[canonical_idx])
            .unwrap_or(false)
    {
        return Some((canonical_idx, local_idx));
    }

    let region_id = RegionId::ALL[canonical_idx];
    regions
        .iter()
        .position(|region| region.id == region_id)
        .map(|region_idx| (region_idx, local_idx))
}

#[inline]
fn regions_in_canonical_order(regions: &[Region]) -> bool {
    regions.len() == RegionId::ALL.len()
        && regions
            .iter()
            .enumerate()
            .all(|(idx, region)| region.id == RegionId::ALL[idx])
}

#[inline]
fn add_signal_to_regions(
    regions: &mut [Region],
    global_id: u32,
    source_region_idx: usize,
    value: f32,
    delayed: bool,
) {
    if let Some((region_idx, local_idx)) = region_index_local_in_slice(regions, global_id) {
        regions[region_idx].add_incoming_classified_local(
            local_idx,
            value,
            source_region_idx,
            delayed,
        );
    }
}

#[inline]
fn segment_outgoing_capacity(synapse_pool: &SynapsePool, segment: &ChunkSegment) -> usize {
    let start = segment.global_start as usize;
    let end = (segment.global_start + segment.len) as usize;
    (synapse_pool.offsets[end] - synapse_pool.offsets[start]) as usize
}

#[inline]
fn propagate_from_local(
    region: &Region,
    local_idx: u32,
    chunk_id: usize,
    synapse_pool: &SynapsePool,
    same_region_delay_ablation: &SameRegionDelayAblation,
    local_segment_lookup: &[Option<usize>; RegionId::ALL.len()],
    gains_by_region: &[f32; RegionId::ALL.len()],
    local_buffers: &mut [LocalIncomingBuffer],
    immediate_mailbox: &mut Vec<RoutedSignal>,
    delayed_mailboxes: &mut [Vec<RoutedSignal>],
    stats: &mut PropagationStats,
) {
    let activation = region.neurons.prev_activations[local_idx as usize];
    let source_region_idx = region.id.index();

    let sign = match region.neurons.neuron_types[local_idx as usize] {
        NeuronType::Excitatory => 1.0f32,
        NeuronType::Inhibitory => -1.0f32,
    };

    let global_id = region.local_to_global(local_idx);
    let start = synapse_pool.offsets[global_id as usize] as usize;
    let (targets, _weights, delays, target_chunks) =
        synapse_pool.outgoing_full_with_chunks(global_id);

    for i in 0..targets.len() {
        let target = targets[i];
        let weight = synapse_pool.weight_at_index(start + i).unwrap_or(0.0);
        let delay = delays[i];
        let normalized_delay = if delay <= 1 { 0 } else { delay };
        let target_chunk = target_chunks[i] as usize;

        let Some((target_region_idx, target_local_idx)) = region_index_local(target) else {
            continue;
        };
        let target_gain = gains_by_region[target_region_idx];

        let signal = activation * weight * sign * target_gain;
        if signal.abs() <= f32::EPSILON {
            continue;
        }

        if same_region_delay_ablation.suppresses(source_region_idx, target_region_idx, delay) {
            continue;
        }

        if target_chunk == chunk_id && normalized_delay == 0 {
            stats.record_immediate(target_region_idx, signal);
            if let Some(buffer_idx) = local_segment_lookup[target_region_idx] {
                let target_buffer = &mut local_buffers[buffer_idx];
                let local_offset = target_local_idx.saturating_sub(target_buffer.local_start);
                if local_offset < target_buffer.incoming.len() as u32 {
                    let local_offset = local_offset as usize;
                    target_buffer.incoming[local_offset] += signal;
                    if source_region_idx == target_region_idx {
                        target_buffer.immediate_same[local_offset] += signal;
                    } else {
                        target_buffer.immediate_cross[local_offset] += signal;
                    }
                    continue;
                }
            }
        }

        if normalized_delay == 0 {
            stats.record_immediate(target_region_idx, signal);
            immediate_mailbox.push(RoutedSignal {
                source_region_idx: source_region_idx as u8,
                region_idx: target_region_idx as u8,
                local_idx: target_local_idx,
                value: signal,
                delayed: false,
            });
        } else {
            stats.record_delayed_schedule(signal);
            stats.record_delayed_flow(source_region_idx, target_region_idx, signal);
            let delayed_index = normalized_delay.min(MAX_DELAY as u8) as usize;
            delayed_mailboxes[delayed_index].push(RoutedSignal {
                source_region_idx: source_region_idx as u8,
                region_idx: target_region_idx as u8,
                local_idx: target_local_idx,
                value: signal,
                delayed: true,
            });
        }
    }
}

#[inline]
fn empty_wave_mailboxes() -> Vec<Vec<Vec<RoutedSignal>>> {
    (0..=MAX_DELAY)
        .map(|_| {
            (0..LaneId::ALL.len())
                .map(|_| Vec::new())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[inline]
fn mark_signals_delayed(entries: &mut [RoutedSignal]) {
    for entry in entries {
        entry.delayed = true;
    }
}

#[inline]
fn lane_schedule_for_signal(
    source_lane: LaneId,
    target_lane: LaneId,
    normalized_delay: u8,
    target_available_same_tick: bool,
) -> (usize, LaneId, bool) {
    if normalized_delay == 0 {
        if target_available_same_tick && source_lane.index() < target_lane.index() {
            (0, target_lane, false)
        } else {
            (1, target_lane, true)
        }
    } else {
        (normalized_delay as usize, target_lane, true)
    }
}

#[inline]
fn propagate_from_local_wave(
    region: &Region,
    local_idx: u32,
    chunk_id: usize,
    lane: LaneId,
    active_regions: &[bool; RegionId::ALL.len()],
    synapse_pool: &SynapsePool,
    same_region_delay_ablation: &SameRegionDelayAblation,
    local_segment_lookup: &[Option<usize>; RegionId::ALL.len()],
    gains_by_region: &[f32; RegionId::ALL.len()],
    local_buffers: &mut [LocalIncomingBuffer],
    immediate_mailbox: &mut Vec<RoutedSignal>,
    scheduled_mailboxes: &mut [Vec<Vec<RoutedSignal>>],
    stats: &mut PropagationStats,
) {
    let activation = region.neurons.prev_activations[local_idx as usize];
    let source_region_idx = region.id.index();
    let source_lane = region.id.lane();
    debug_assert_eq!(source_lane, lane);

    let sign = match region.neurons.neuron_types[local_idx as usize] {
        NeuronType::Excitatory => 1.0f32,
        NeuronType::Inhibitory => -1.0f32,
    };

    let global_id = region.local_to_global(local_idx);
    let start = synapse_pool.offsets[global_id as usize] as usize;
    let (targets, _weights, delays, target_chunks) =
        synapse_pool.outgoing_full_with_chunks(global_id);

    for i in 0..targets.len() {
        let target = targets[i];
        let weight = synapse_pool.weight_at_index(start + i).unwrap_or(0.0);
        let delay = delays[i];
        let normalized_delay = if delay <= 1 { 0 } else { delay };
        let target_chunk = target_chunks[i] as usize;

        let Some((target_region_idx, target_local_idx)) = region_index_local(target) else {
            continue;
        };
        let target_gain = gains_by_region[target_region_idx];
        let target_lane = RegionId::ALL[target_region_idx].lane();

        let signal = activation * weight * sign * target_gain;
        if signal.abs() <= f32::EPSILON {
            continue;
        }

        if same_region_delay_ablation.suppresses(source_region_idx, target_region_idx, delay) {
            continue;
        }

        if target_lane == lane && active_regions[target_region_idx] && normalized_delay == 0 {
            stats.record_immediate(target_region_idx, signal);
            if target_chunk == chunk_id {
                if let Some(buffer_idx) = local_segment_lookup[target_region_idx] {
                    let target_buffer = &mut local_buffers[buffer_idx];
                    let local_offset = target_local_idx.saturating_sub(target_buffer.local_start);
                    if local_offset < target_buffer.incoming.len() as u32 {
                        let local_offset = local_offset as usize;
                        target_buffer.incoming[local_offset] += signal;
                        if source_region_idx == target_region_idx {
                            target_buffer.immediate_same[local_offset] += signal;
                        } else {
                            target_buffer.immediate_cross[local_offset] += signal;
                        }
                        continue;
                    }
                }
            }

            immediate_mailbox.push(RoutedSignal {
                source_region_idx: source_region_idx as u8,
                region_idx: target_region_idx as u8,
                local_idx: target_local_idx,
                value: signal,
                delayed: false,
            });
            continue;
        }

        let target_available_same_tick =
            active_regions[target_region_idx] && source_lane.index() < target_lane.index();
        let (base_tick_offset, ready_wave, delayed) =
            lane_schedule_for_signal(
                source_lane,
                target_lane,
                normalized_delay,
                target_available_same_tick,
            );
        if delayed {
            stats.record_delayed_schedule(signal);
            stats.record_delayed_flow(source_region_idx, target_region_idx, signal);
        } else {
            stats.record_immediate(target_region_idx, signal);
        }
        scheduled_mailboxes[base_tick_offset.min(MAX_DELAY)][ready_wave.index()].push(
            RoutedSignal {
                source_region_idx: source_region_idx as u8,
                region_idx: target_region_idx as u8,
                local_idx: target_local_idx,
                value: signal,
                delayed,
            },
        );
    }
}

#[inline]
fn active_locals_in_segment<'a>(region: &'a Region, segment: &ChunkSegment) -> Option<&'a [u32]> {
    if !region.neurons.prev_active_cache_valid {
        return None;
    }

    let active = &region.neurons.prev_active_local_ids;
    let start = segment.local_start;
    let end = segment.local_start + segment.len;
    let start_idx = active.partition_point(|&local_idx| local_idx < start);
    let end_idx = active.partition_point(|&local_idx| local_idx < end);
    Some(&active[start_idx..end_idx])
}

pub fn propagate_wave(
    regions: &mut [Region],
    synapse_pool: &SynapsePool,
    delay_buffer: &mut DelayBuffer,
    attention_gains: &std::collections::HashMap<RegionId, f32>,
    same_region_delay_ablation: &SameRegionDelayAblation,
    lane: LaneId,
    active_regions: &[bool; RegionId::ALL.len()],
) -> PropagationStats {
    let default_gain = 1.0f32;
    let chunk_count = synapse_pool.chunk_count();
    let mut gains_by_region = [default_gain; RegionId::ALL.len()];
    for &region_id in RegionId::ALL.iter() {
        gains_by_region[region_id.index()] =
            *attention_gains.get(&region_id).unwrap_or(&default_gain);
    }

    let outputs: Vec<WaveChunkOutput> = {
        let read_regions: &[Region] = &regions[..];
        let mut region_lookup = [None; RegionId::ALL.len()];
        for (idx, region) in read_regions.iter().enumerate() {
            region_lookup[region.id.index()] = Some(idx);
        }

        (0..chunk_count)
            .into_par_iter()
            .map(|chunk_id| {
                let segments = synapse_pool.chunk_segments(chunk_id);
                let lane_segments: Vec<&ChunkSegment> = segments
                    .iter()
                    .filter(|segment| segment.region_id.lane() == lane)
                    .collect();
                let mailbox_capacity: usize = lane_segments
                    .iter()
                    .map(|segment| segment_outgoing_capacity(synapse_pool, segment))
                    .sum();
                let mut local_buffers = Vec::with_capacity(lane_segments.len());
                let mut local_segment_lookup = [None; RegionId::ALL.len()];
                for segment in &lane_segments {
                    let Some(region_idx) = region_lookup[segment.region_id.index()] else {
                        continue;
                    };
                    let buffer_idx = local_buffers.len();
                    local_segment_lookup[segment.region_id.index()] = Some(buffer_idx);
                    local_buffers.push(LocalIncomingBuffer {
                        region_idx,
                        local_start: segment.local_start,
                        incoming: vec![0.0f32; segment.len as usize],
                        immediate_same: vec![0.0f32; segment.len as usize],
                        immediate_cross: vec![0.0f32; segment.len as usize],
                    });
                }
                let mut immediate_mailbox = Vec::with_capacity(mailbox_capacity / 4);
                let mut scheduled_mailboxes = empty_wave_mailboxes();
                let mut stats = PropagationStats::default();

                for segment in lane_segments {
                    let Some(region_idx) = region_lookup[segment.region_id.index()] else {
                        continue;
                    };

                    let region = &read_regions[region_idx];
                    if let Some(active_locals) = active_locals_in_segment(region, segment) {
                        for &local_idx in active_locals {
                            propagate_from_local_wave(
                                region,
                                local_idx,
                                chunk_id,
                                lane,
                                active_regions,
                                synapse_pool,
                                same_region_delay_ablation,
                                &local_segment_lookup,
                                &gains_by_region,
                                &mut local_buffers,
                                &mut immediate_mailbox,
                                &mut scheduled_mailboxes,
                                &mut stats,
                            );
                        }
                    } else {
                        for offset in 0..segment.len {
                            let local_idx = segment.local_start + offset;
                            if region.neurons.prev_activations[local_idx as usize] <= 0.0 {
                                continue;
                            }
                            propagate_from_local_wave(
                                region,
                                local_idx,
                                chunk_id,
                                lane,
                                active_regions,
                                synapse_pool,
                                same_region_delay_ablation,
                                &local_segment_lookup,
                                &gains_by_region,
                                &mut local_buffers,
                                &mut immediate_mailbox,
                                &mut scheduled_mailboxes,
                                &mut stats,
                            );
                        }
                    }
                }

                WaveChunkOutput {
                    local_buffers,
                    immediate_mailbox,
                    scheduled_mailboxes,
                    stats,
                }
            })
            .collect()
    };

    let mut propagation_stats = PropagationStats::default();
    for output in outputs {
        for local_buffer in output.local_buffers {
            let region = &mut regions[local_buffer.region_idx];
            for (offset, value) in local_buffer.incoming.into_iter().enumerate() {
                if value != 0.0 {
                    region.incoming[(local_buffer.local_start + offset as u32) as usize] += value;
                }
            }
            for (offset, value) in local_buffer.immediate_same.into_iter().enumerate() {
                if value != 0.0 {
                    region.incoming_immediate_same
                        [(local_buffer.local_start + offset as u32) as usize] += value;
                }
            }
            for (offset, value) in local_buffer.immediate_cross.into_iter().enumerate() {
                if value != 0.0 {
                    region.incoming_immediate_cross
                        [(local_buffer.local_start + offset as u32) as usize] += value;
                }
            }
        }

        for signal in output.immediate_mailbox {
            regions[signal.region_idx as usize].add_incoming_classified_local(
                signal.local_idx,
                signal.value,
                signal.source_region_idx as usize,
                false,
            );
        }

        for (delay, wave_mailboxes) in output.scheduled_mailboxes.into_iter().enumerate() {
            for (wave_idx, entries) in wave_mailboxes.into_iter().enumerate() {
                if entries.is_empty() {
                    continue;
                }
                delay_buffer.extend_relative_slot_wave(delay, LaneId::ALL[wave_idx], entries);
            }
        }

        propagation_stats.merge(&output.stats);
    }

    propagation_stats
}

pub fn deliver_wave_signals(
    regions: &mut [Region],
    delay_buffer: &mut DelayBuffer,
    lane: LaneId,
    active_regions: &[bool; RegionId::ALL.len()],
) -> DeliveryStats {
    let signals = delay_buffer.drain_current_wave_local(lane);
    let mut stats = DeliveryStats::default();
    let mut deferred = Vec::new();

    if regions_in_canonical_order(regions) {
        for mut signal in signals {
            let target_region_idx = signal.region_idx as usize;
            if !active_regions[target_region_idx] {
                signal.delayed = true;
                deferred.push(signal);
                continue;
            }
            if signal.delayed {
                stats.record(target_region_idx, signal.value);
            }
            regions[target_region_idx].add_incoming_classified_local(
                signal.local_idx,
                signal.value,
                signal.source_region_idx as usize,
                signal.delayed,
            );
        }
        if !deferred.is_empty() {
            delay_buffer.extend_relative_slot_wave(1, lane, deferred);
        }
        return stats;
    }

    for mut signal in signals {
        let target_region_idx = signal.region_idx as usize;
        if !active_regions[target_region_idx] {
            signal.delayed = true;
            deferred.push(signal);
            continue;
        }
        if signal.delayed {
            stats.record(target_region_idx, signal.value);
        }
        let global_id = RegionId::ALL[target_region_idx].neuron_range().0 + signal.local_idx;
        add_signal_to_regions(
            regions,
            global_id,
            signal.source_region_idx as usize,
            signal.value,
            signal.delayed,
        );
    }

    if !deferred.is_empty() {
        delay_buffer.extend_relative_slot_wave(1, lane, deferred);
    }

    stats
}

/// Propagate signals from all active neurons through synapses.
///
/// Uses rayon to parallelize across regions. Each region collects its
/// outgoing signals independently, then signals are merged into the
/// delay buffer sequentially.
///
/// `attention_gains` maps RegionId to a gain multiplier (default 1.0).
/// Signals arriving at a target region are multiplied by that region's gain.
pub fn propagate(
    regions: &mut [Region],
    synapse_pool: &SynapsePool,
    delay_buffer: &mut DelayBuffer,
    attention_gains: &std::collections::HashMap<RegionId, f32>,
    same_region_delay_ablation: &SameRegionDelayAblation,
) -> PropagationStats {
    let default_gain = 1.0f32;
    let chunk_count = synapse_pool.chunk_count();
    let mut gains_by_region = [default_gain; RegionId::ALL.len()];
    for &region_id in RegionId::ALL.iter() {
        gains_by_region[region_id.index()] =
            *attention_gains.get(&region_id).unwrap_or(&default_gain);
    }

    // Parallel phase: each chunk owns immediate local delivery and emits
    // only delayed or cross-chunk traffic into mailboxes.
    let outputs: Vec<ChunkOutput> = {
        let read_regions: &[Region] = &regions[..];
        let mut region_lookup = [None; RegionId::ALL.len()];
        for (idx, region) in read_regions.iter().enumerate() {
            region_lookup[region.id.index()] = Some(idx);
        }

        (0..chunk_count)
            .into_par_iter()
            .map(|chunk_id| {
                let segments = synapse_pool.chunk_segments(chunk_id);
                let mailbox_capacity: usize = segments
                    .iter()
                    .map(|segment| segment_outgoing_capacity(synapse_pool, segment))
                    .sum();
                let mut local_buffers = Vec::with_capacity(segments.len());
                let mut local_segment_lookup = [None; RegionId::ALL.len()];
                for segment in &segments {
                    let Some(region_idx) = region_lookup[segment.region_id.index()] else {
                        continue;
                    };
                    let buffer_idx = local_buffers.len();
                    local_segment_lookup[segment.region_id.index()] = Some(buffer_idx);
                    local_buffers.push(LocalIncomingBuffer {
                        region_idx,
                        local_start: segment.local_start,
                        incoming: vec![0.0f32; segment.len as usize],
                        immediate_same: vec![0.0f32; segment.len as usize],
                        immediate_cross: vec![0.0f32; segment.len as usize],
                    });
                }
                let mut immediate_mailbox = Vec::with_capacity(mailbox_capacity / 4);
                let mut delayed_mailboxes = (0..=MAX_DELAY)
                    .map(|_| Vec::new())
                    .collect::<Vec<_>>();
                let mut stats = PropagationStats::default();

                for segment in &segments {
                    let Some(region_idx) = region_lookup[segment.region_id.index()] else {
                        continue;
                    };

                    let region = &read_regions[region_idx];
                    if let Some(active_locals) = active_locals_in_segment(region, segment) {
                        for &local_idx in active_locals {
                            propagate_from_local(
                                region,
                                local_idx,
                                chunk_id,
                                synapse_pool,
                                same_region_delay_ablation,
                                &local_segment_lookup,
                                &gains_by_region,
                                &mut local_buffers,
                                &mut immediate_mailbox,
                                &mut delayed_mailboxes,
                                &mut stats,
                            );
                        }
                    } else {
                        for offset in 0..segment.len {
                            let local_idx = segment.local_start + offset;
                            if region.neurons.prev_activations[local_idx as usize] <= 0.0 {
                                continue;
                            }
                            propagate_from_local(
                                region,
                                local_idx,
                                chunk_id,
                                synapse_pool,
                                same_region_delay_ablation,
                                &local_segment_lookup,
                                &gains_by_region,
                                &mut local_buffers,
                                &mut immediate_mailbox,
                                &mut delayed_mailboxes,
                                &mut stats,
                            );
                        }
                    }
                }

                ChunkOutput {
                    local_buffers,
                    immediate_mailbox,
                    delayed_mailboxes,
                    stats,
                }
            })
            .collect()
    };

    // Sequential phase: apply owned local contributions, then schedule only
    // delayed or cross-chunk traffic into the delay buffer.
    let mut propagation_stats = PropagationStats::default();
    for output in outputs {
        for local_buffer in output.local_buffers {
            let region = &mut regions[local_buffer.region_idx];
            for (offset, value) in local_buffer.incoming.into_iter().enumerate() {
                if value != 0.0 {
                    region.incoming[(local_buffer.local_start + offset as u32) as usize] += value;
                }
            }
            for (offset, value) in local_buffer.immediate_same.into_iter().enumerate() {
                if value != 0.0 {
                    region.incoming_immediate_same[(local_buffer.local_start + offset as u32) as usize] += value;
                }
            }
            for (offset, value) in local_buffer.immediate_cross.into_iter().enumerate() {
                if value != 0.0 {
                    region.incoming_immediate_cross[(local_buffer.local_start + offset as u32) as usize] += value;
                }
            }
        }

        for signal in output.immediate_mailbox {
            regions[signal.region_idx as usize].add_incoming_classified_local(
                signal.local_idx,
                signal.value,
                signal.source_region_idx as usize,
                false,
            );
        }

        for (delay, entries) in output.delayed_mailboxes.into_iter().enumerate() {
            if delay == 0 {
                continue;
            }
            delay_buffer.extend_relative_slot(delay, entries);
        }

        propagation_stats.merge(&output.stats);
    }

    propagation_stats
}

/// Deliver signals from the delay buffer into region incoming buffers.
pub fn deliver_delayed_signals(
    regions: &mut [Region],
    delay_buffer: &mut DelayBuffer,
) -> DeliveryStats {
    let signals = delay_buffer.drain_current_local();
    let mut stats = DeliveryStats::default();

    if regions_in_canonical_order(regions) {
        for signal in signals {
            if signal.delayed {
                stats.record(signal.region_idx as usize, signal.value);
            }
            regions[signal.region_idx as usize].add_incoming_classified_local(
                signal.local_idx,
                signal.value,
                signal.source_region_idx as usize,
                signal.delayed,
            );
        }
        delay_buffer.advance();
        return stats;
    }

    for signal in signals {
        if signal.delayed {
            stats.record(signal.region_idx as usize, signal.value);
        }
        let global_id = RegionId::ALL[signal.region_idx as usize].neuron_range().0 + signal.local_idx;
        add_signal_to_regions(
            regions,
            global_id,
            signal.source_region_idx as usize,
            signal.value,
            signal.delayed,
        );
    }

    delay_buffer.advance();
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::synapse::SynapseData;

    #[test]
    fn test_region_for_neuron() {
        assert_eq!(region_for_neuron(0), Some(RegionId::Sensory));
        assert_eq!(region_for_neuron(9_999), Some(RegionId::Sensory));
        assert_eq!(region_for_neuron(10_000), Some(RegionId::Visual));
        assert_eq!(region_for_neuron(151_999), Some(RegionId::Numbers));
        assert_eq!(region_for_neuron(152_000), None);
    }

    #[test]
    fn test_delay_buffer() {
        let mut buf = DelayBuffer::new();

        // Schedule a signal 3 ticks from now
        buf.schedule(42, 0.5, 3);

        // Nothing now
        assert!(buf.drain_current().is_empty());
        buf.advance();

        // Nothing at tick 1
        assert!(buf.drain_current().is_empty());
        buf.advance();

        // Nothing at tick 2
        assert!(buf.drain_current().is_empty());
        buf.advance();

        // Should arrive at tick 3
        let signals = buf.drain_current();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0], (42, 0.5));
    }

    #[test]
    fn test_propagate_basic() {
        // Create two regions: sensory (source) and emotion (target)
        let mut sensory = Region::new(RegionId::Sensory);
        let emotion = Region::new(RegionId::Emotion);

        // Manually activate sensory neuron 100 (global: 100)
        sensory.neurons.prev_activations[100] = 1.0;

        // Create synapse: sensory neuron 100 → emotion neuron 70000 (local 0)
        let synapses = vec![SynapseData {
            from: 100,
            to: 70_000,
            weight: 0.5,
            delay: 0,
            plasticity: 1.0,
        }];
        let pool = SynapsePool::from_synapses(152_000, synapses);

        let mut delay_buffer = DelayBuffer::new();
        let gains = std::collections::HashMap::new();

        let regions = vec![sensory, emotion];
        let mut regions = regions;
        let stats = propagate(
            &mut regions,
            &pool,
            &mut delay_buffer,
            &gains,
            &SameRegionDelayAblation::default(),
        );

        // Emotion neuron 0 (global 70000) should have received signal
        assert!(regions[1].incoming[0].abs() > 0.01,
            "Expected emotion neuron to receive signal, got {}",
            regions[1].incoming[0]);
        assert_eq!(stats.immediate_signal_count, 1);
        assert_eq!(stats.scheduled_delayed_signal_count, 0);
        assert_eq!(stats.target_signal_counts[RegionId::Emotion.index()], 1);
    }

    #[test]
    fn test_local_chunk_immediate_delivery_skips_mailbox() {
        let mut sensory = Region::new(RegionId::Sensory);
        sensory.neurons.prev_activations[1] = 1.0;

        let synapses = vec![SynapseData {
            from: 1,
            to: 2,
            weight: 0.5,
            delay: 0,
            plasticity: 1.0,
        }];
        let pool = SynapsePool::from_synapses_with_chunks(10, synapses, 2);
        let mut delay_buffer = DelayBuffer::new();
        let gains = std::collections::HashMap::new();
        let mut regions = vec![sensory];

        let stats = propagate(
            &mut regions,
            &pool,
            &mut delay_buffer,
            &gains,
            &SameRegionDelayAblation::default(),
        );

        assert!(regions[0].incoming[2] > 0.01);
        assert!(delay_buffer.drain_current().is_empty());
        assert_eq!(stats.immediate_signal_count, 1);
        assert_eq!(stats.target_signal_counts[RegionId::Sensory.index()], 1);
    }

    #[test]
    fn test_region_bucket_immediate_delivery_skips_mailbox() {
        let mut sensory = Region::new(RegionId::Sensory);
        let visual = Region::new(RegionId::Visual);
        sensory.neurons.prev_activations[100] = 1.0;

        let synapses = vec![SynapseData {
            from: 100,
            to: 10_100,
            weight: 0.5,
            delay: 0,
            plasticity: 1.0,
        }];
        let pool = SynapsePool::from_synapses_with_chunks(152_000, synapses, 16);
        let mut delay_buffer = DelayBuffer::new();
        let gains = std::collections::HashMap::new();
        let mut regions = vec![sensory, visual];

        let stats = propagate(
            &mut regions,
            &pool,
            &mut delay_buffer,
            &gains,
            &SameRegionDelayAblation::default(),
        );

        assert!(regions[1].incoming[100] > 0.01);
        assert!(delay_buffer.drain_current().is_empty());
        assert_eq!(stats.immediate_signal_count, 1);
        assert_eq!(stats.target_signal_counts[RegionId::Visual.index()], 1);
    }

    #[test]
    fn test_deliver_delayed_signals_returns_stats() {
        let mut regions = vec![Region::new(RegionId::Emotion)];
        let mut delay_buffer = DelayBuffer::new();

        delay_buffer.schedule(70_000, 0.5, 0);
        let stats = deliver_delayed_signals(&mut regions, &mut delay_buffer);

        assert_eq!(stats.signal_count, 1);
        assert_eq!(stats.target_signal_counts[RegionId::Emotion.index()], 1);
        assert!((stats.signal_abs_sum - 0.5).abs() < 0.0001);
        assert!((regions[0].incoming[0] - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_propagate_tracks_delayed_scheduling() {
        let mut sensory = Region::new(RegionId::Sensory);
        let emotion = Region::new(RegionId::Emotion);
        sensory.neurons.prev_activations[5] = 1.0;

        let synapses = vec![SynapseData {
            from: 5,
            to: 70_000,
            weight: 0.25,
            delay: 2,
            plasticity: 1.0,
        }];
        let pool = SynapsePool::from_synapses(152_000, synapses);
        let mut delay_buffer = DelayBuffer::new();
        let gains = std::collections::HashMap::new();
        let mut regions = vec![sensory, emotion];

        let stats = propagate(
            &mut regions,
            &pool,
            &mut delay_buffer,
            &gains,
            &SameRegionDelayAblation::default(),
        );

        assert_eq!(stats.immediate_signal_count, 0);
        assert_eq!(stats.scheduled_delayed_signal_count, 1);
        assert!((stats.scheduled_delayed_signal_abs_sum - 0.25).abs() < 0.0001);
        assert_eq!(
            stats.delayed_flow_signal_counts[RegionId::Sensory.index()][RegionId::Emotion.index()],
            1
        );
    }

    #[test]
    fn test_propagate_ablation_suppresses_same_region_delay_1_and_2_only() {
        let mut sensory = Region::new(RegionId::Sensory);
        let visual = Region::new(RegionId::Visual);
        sensory.neurons.prev_activations[0] = 1.0;

        let synapses = vec![
            SynapseData {
                from: 0,
                to: 1,
                weight: 0.5,
                delay: 1,
                plasticity: 1.0,
            },
            SynapseData {
                from: 0,
                to: 2,
                weight: 0.25,
                delay: 2,
                plasticity: 1.0,
            },
            SynapseData {
                from: 0,
                to: 3,
                weight: 0.125,
                delay: 3,
                plasticity: 1.0,
            },
            SynapseData {
                from: 0,
                to: 10_000,
                weight: 0.75,
                delay: 2,
                plasticity: 1.0,
            },
        ];
        let pool = SynapsePool::from_synapses(152_000, synapses);
        let mut delay_buffer = DelayBuffer::new();
        let gains = std::collections::HashMap::new();
        let mut ablation = SameRegionDelayAblation::default();
        ablation.configure(&[RegionId::Sensory], &[1, 2]);
        let mut regions = vec![sensory, visual];

        let stats = propagate(&mut regions, &pool, &mut delay_buffer, &gains, &ablation);

        assert_eq!(stats.immediate_signal_count, 0);
        assert_eq!(stats.scheduled_delayed_signal_count, 2);
        assert!((regions[0].incoming[1] - 0.0).abs() < 0.0001);
        assert!((regions[0].incoming_immediate_same[1] - 0.0).abs() < 0.0001);

        delay_buffer.advance();
        delay_buffer.advance();
        let delay_two_stats = deliver_delayed_signals(&mut regions, &mut delay_buffer);
        assert_eq!(delay_two_stats.signal_count, 1);
        assert!((regions[1].incoming[0] - 0.75).abs() < 0.0001);

        let delay_three_stats = deliver_delayed_signals(&mut regions, &mut delay_buffer);
        assert_eq!(delay_three_stats.signal_count, 1);
        assert!((regions[0].incoming[3] - 0.125).abs() < 0.0001);
        assert!((regions[0].incoming_delayed_same[3] - 0.125).abs() < 0.0001);
        assert!((regions[0].incoming[2] - 0.0).abs() < 0.0001);
    }
}
