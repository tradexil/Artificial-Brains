/// Signal propagation: the hot loop of the brain.
///
/// For each active neuron, traverse its outgoing synapses and
/// deposit weighted signal into target neurons' incoming buffers.
/// Respects synapse delay via a delay buffer ring.
///
/// This is the most performance-critical code — runs every tick.

use crate::core::neuron::NeuronType;
use crate::core::region::{Region, RegionId};
use crate::core::synapse::{ChunkSegment, SynapsePool};
use rayon::prelude::*;

type MailboxEntry = (u32, f32, u8);

struct LocalIncomingBuffer {
    region_idx: usize,
    local_start: u32,
    incoming: Vec<f32>,
}

struct ChunkOutput {
    local_buffers: Vec<LocalIncomingBuffer>,
    mailbox: Vec<MailboxEntry>,
}

/// Maximum synapse delay in ticks.
pub const MAX_DELAY: usize = 10;

/// Ring buffer for delayed signal delivery.
/// Stores (target_global_id, signal_value) for each future tick.
pub struct DelayBuffer {
    /// Ring of per-tick signal buffers. Index = tick % MAX_DELAY.
    slots: Vec<Vec<(u32, f32)>>,
    current_slot: usize,
}

impl DelayBuffer {
    pub fn new() -> Self {
        Self {
            slots: (0..MAX_DELAY).map(|_| Vec::new()).collect(),
            current_slot: 0,
        }
    }

    /// Schedule a signal to be delivered `delay` ticks from now.
    #[inline]
    pub fn schedule(&mut self, target: u32, value: f32, delay: u8) {
        let slot = (self.current_slot + delay as usize) % MAX_DELAY;
        self.slots[slot].push((target, value));
    }

    pub fn schedule_batch(&mut self, entries: &[(u32, f32, u8)]) {
        if entries.is_empty() {
            return;
        }

        let mut slot_counts = [0usize; MAX_DELAY];
        for &(_, _, delay) in entries {
            let slot = (self.current_slot + delay as usize) % MAX_DELAY;
            slot_counts[slot] += 1;
        }

        for (slot, count) in slot_counts.into_iter().enumerate() {
            if count > 0 {
                self.slots[slot].reserve(count);
            }
        }

        for &(target, value, delay) in entries {
            let slot = (self.current_slot + delay as usize) % MAX_DELAY;
            self.slots[slot].push((target, value));
        }
    }

    /// Get all signals due for delivery this tick, and clear the slot.
    pub fn drain_current(&mut self) -> Vec<(u32, f32)> {
        let signals = std::mem::take(&mut self.slots[self.current_slot]);
        signals
    }

    /// Advance to next tick.
    pub fn advance(&mut self) {
        self.current_slot = (self.current_slot + 1) % MAX_DELAY;
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
fn add_signal_to_regions(regions: &mut [Region], global_id: u32, value: f32) {
    if let Some((region_idx, local_idx)) = region_index_local_in_slice(regions, global_id) {
        regions[region_idx].incoming[local_idx as usize] += value;
    }
}

#[inline]
fn segment_outgoing_capacity(synapse_pool: &SynapsePool, segment: &ChunkSegment) -> usize {
    let start = segment.global_start as usize;
    let end = (segment.global_start + segment.len) as usize;
    (synapse_pool.offsets[end] - synapse_pool.offsets[start]) as usize
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
) {
    let default_gain = 1.0f32;
    let chunk_count = synapse_pool.chunk_count();

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
                let mailbox_capacity = segments
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
                    });
                }
                let mut mailbox = Vec::with_capacity(mailbox_capacity);

                for segment in &segments {
                    let Some(region_idx) = region_lookup[segment.region_id.index()] else {
                        continue;
                    };

                    let region = &read_regions[region_idx];
                    for offset in 0..segment.len {
                        let local_idx = segment.local_start + offset;
                        let activation = region.neurons.prev_activations[local_idx as usize];
                        if activation <= 0.0 {
                            continue;
                        }

                        let sign = match region.neurons.neuron_types[local_idx as usize] {
                            NeuronType::Excitatory => 1.0f32,
                            NeuronType::Inhibitory => -1.0f32,
                        };

                        let global_id = segment.global_start + offset;
                        let (targets, weights, delays, target_chunks) =
                            synapse_pool.outgoing_full_with_chunks(global_id);

                        for i in 0..targets.len() {
                            let target = targets[i];
                            let weight = weights[i];
                            let delay = delays[i];
                            let normalized_delay = if delay <= 1 { 0 } else { delay };
                            let target_chunk = target_chunks[i] as usize;

                            let target_gain = if let Some(target_region) = region_for_neuron(target) {
                                *attention_gains.get(&target_region).unwrap_or(&default_gain)
                            } else {
                                default_gain
                            };

                            let signal = activation * weight * sign * target_gain;

                            if target_chunk == chunk_id && normalized_delay == 0 {
                                if let Some((canonical_idx, target_local_idx)) = region_index_local(target) {
                                    if let Some(buffer_idx) = local_segment_lookup[canonical_idx] {
                                        let target_buffer = &mut local_buffers[buffer_idx];
                                        let local_offset = target_local_idx.saturating_sub(target_buffer.local_start);
                                        if local_offset < target_buffer.incoming.len() as u32 {
                                            target_buffer.incoming[local_offset as usize] += signal;
                                            continue;
                                        }
                                    }
                                }
                            }

                            mailbox.push((target, signal, normalized_delay));
                        }
                    }
                }

                ChunkOutput {
                    local_buffers,
                    mailbox,
                }
            })
            .collect()
    };

    // Sequential phase: apply owned local contributions, then schedule only
    // delayed or cross-chunk traffic into the delay buffer.
    for output in outputs {
        for local_buffer in output.local_buffers {
            let region = &mut regions[local_buffer.region_idx];
            for (offset, value) in local_buffer.incoming.into_iter().enumerate() {
                if value != 0.0 {
                    region.incoming[(local_buffer.local_start + offset as u32) as usize] += value;
                }
            }
        }

        delay_buffer.schedule_batch(&output.mailbox);
    }
}

/// Deliver signals from the delay buffer into region incoming buffers.
pub fn deliver_delayed_signals(
    regions: &mut [Region],
    delay_buffer: &mut DelayBuffer,
) {
    let signals = delay_buffer.drain_current();

    for (global_id, value) in signals {
        add_signal_to_regions(regions, global_id, value);
    }

    delay_buffer.advance();
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
        propagate(&mut regions, &pool, &mut delay_buffer, &gains);

        // Emotion neuron 0 (global 70000) should have received signal
        assert!(regions[1].incoming[0].abs() > 0.01,
            "Expected emotion neuron to receive signal, got {}",
            regions[1].incoming[0]);
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

        propagate(&mut regions, &pool, &mut delay_buffer, &gains);

        assert!(regions[0].incoming[2] > 0.01);
        assert!(delay_buffer.drain_current().is_empty());
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

        propagate(&mut regions, &pool, &mut delay_buffer, &gains);

        assert!(regions[1].incoming[100] > 0.01);
        assert!(delay_buffer.drain_current().is_empty());
    }
}
