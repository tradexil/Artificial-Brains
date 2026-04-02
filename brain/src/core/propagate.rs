/// Signal propagation: the hot loop of the brain.
///
/// For each active neuron, traverse its outgoing synapses and
/// deposit weighted signal into target neurons' incoming buffers.
/// Respects synapse delay via a delay buffer ring.
///
/// This is the most performance-critical code — runs every tick.

use crate::core::neuron::NeuronType;
use crate::core::region::{Region, RegionId};
use crate::core::synapse::SynapsePool;
use rayon::prelude::*;

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
    match global_id {
        0..=9_999       => Some(RegionId::Sensory),
        10_000..=29_999 => Some(RegionId::Visual),
        30_000..=44_999 => Some(RegionId::Audio),
        45_000..=54_999 => Some(RegionId::MemoryShort),
        55_000..=69_999 => Some(RegionId::MemoryLong),
        70_000..=79_999 => Some(RegionId::Emotion),
        80_000..=84_999 => Some(RegionId::Attention),
        85_000..=94_999 => Some(RegionId::Pattern),
        95_000..=104_999  => Some(RegionId::Integration),
        105_000..=119_999 => Some(RegionId::Language),
        120_000..=129_999 => Some(RegionId::Executive),
        130_000..=139_999 => Some(RegionId::Motor),
        140_000..=149_999 => Some(RegionId::Speech),
        150_000..=151_999 => Some(RegionId::Numbers),
        _ => None,
    }
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
    regions: &[Region],
    synapse_pool: &SynapsePool,
    delay_buffer: &mut DelayBuffer,
    attention_gains: &std::collections::HashMap<RegionId, f32>,
) {
    let default_gain = 1.0f32;

    // Parallel phase: each region collects (target, signal, delay) tuples
    let all_signals: Vec<Vec<(u32, f32, u8)>> = regions
        .par_iter()
        .map(|region| {
            let mut signals = Vec::new();

            for local_idx in 0..region.neurons.count {
                let activation = region.neurons.prev_activations[local_idx as usize];
                if activation <= 0.0 {
                    continue;
                }

                let global_id = region.local_to_global(local_idx);
                let neuron_type = region.neurons.neuron_types[local_idx as usize];

                let sign = match neuron_type {
                    NeuronType::Excitatory => 1.0f32,
                    NeuronType::Inhibitory => -1.0f32,
                };

                let (targets, weights, delays) = synapse_pool.outgoing_full(global_id);

                for i in 0..targets.len() {
                    let target = targets[i];
                    let weight = weights[i];
                    let delay = delays[i];

                    let target_gain = if let Some(target_region) = region_for_neuron(target) {
                        *attention_gains.get(&target_region).unwrap_or(&default_gain)
                    } else {
                        default_gain
                    };

                    let signal = activation * weight * sign * target_gain;
                    signals.push((target, signal, delay));
                }
            }

            signals
        })
        .collect();

    // Sequential phase: merge signals into delay buffer
    for signals in all_signals {
        for (target, signal, delay) in signals {
            if delay <= 1 {
                delay_buffer.schedule(target, signal, 0);
            } else {
                delay_buffer.schedule(target, signal, delay);
            }
        }
    }
}

/// Deliver signals from the delay buffer into region incoming buffers.
pub fn deliver_delayed_signals(
    regions: &mut [Region],
    delay_buffer: &mut DelayBuffer,
) {
    let signals = delay_buffer.drain_current();

    for (global_id, value) in signals {
        // Find which region this target belongs to and add to incoming
        if let Some(region_id) = region_for_neuron(global_id) {
            for region in regions.iter_mut() {
                if region.id == region_id {
                    region.add_incoming_global(global_id, value);
                    break;
                }
            }
        }
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
        let mut emotion = Region::new(RegionId::Emotion);

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
        propagate(&regions, &pool, &mut delay_buffer, &gains);

        // Deliver the scheduled signals
        let mut sensory = Region::new(RegionId::Sensory);
        let emotion = Region::new(RegionId::Emotion);
        sensory.neurons.prev_activations[100] = 1.0;

        let mut mutable_regions = [sensory, emotion];
        deliver_delayed_signals(&mut mutable_regions, &mut delay_buffer);

        // Emotion neuron 0 (global 70000) should have received signal
        assert!(mutable_regions[1].incoming[0].abs() > 0.01,
            "Expected emotion neuron to receive signal, got {}",
            mutable_regions[1].incoming[0]);
    }
}
