/// Tick cycle orchestration.
///
/// Each tick has phases:
///   Phase 1: Propagate — signals flow through synapses (rayon-parallel)
///   Phase 2: Update — neurons integrate and fire (rayon-parallel)
///
/// Phase 3 (Learn) and Phase 4 (Maintain) are driven by Python.

use crate::core::propagate::{propagate, deliver_delayed_signals, DelayBuffer};
use crate::core::region::{Region, RegionId};
use crate::core::synapse::SynapsePool;
use rayon::prelude::*;
use std::collections::HashMap;

/// Result of a single tick — stats for Python to consume.
#[derive(Debug, Clone)]
pub struct TickResult {
    pub tick_number: u64,
    pub active_counts: HashMap<RegionId, u32>,
    pub total_active: u32,
}

/// Execute one complete tick cycle.
///
/// 1. Swap activation buffers (prev ← current)
/// 2. Deliver delayed signals from previous ticks
/// 3. Propagate: active neurons push signal through synapses
/// 4. Deliver immediate signals
/// 5. Update: each region integrates incoming and fires
pub fn tick(
    regions: &mut Vec<Region>,
    synapse_pool: &SynapsePool,
    delay_buffer: &mut DelayBuffer,
    attention_gains: &HashMap<RegionId, f32>,
    tick_number: u64,
) -> TickResult {
    // Phase 0: Prepare — swap buffers, clear incoming (parallel)
    regions.par_iter_mut().for_each(|region| {
        region.pre_tick();
    });

    // Phase 1a: Deliver signals that were delayed from previous ticks
    deliver_delayed_signals(regions, delay_buffer);

    // Phase 1b: Propagate — active neurons push signal through synapses (parallel)
    // Propagation uses rayon internally to parallelize across regions.
    propagate(regions, synapse_pool, delay_buffer, attention_gains);

    // Phase 1c: Deliver immediate signals (delay=0 scheduled this tick)
    deliver_delayed_signals(regions, delay_buffer);

    // Phase 2: Update — each region integrates incoming signals (parallel)
    regions.par_iter_mut().for_each(|region| {
        region.update_neurons();
    });

    // Collect stats (parallel reduce)
    let (active_counts, total_active) = regions
        .par_iter()
        .map(|region| {
            let count = region.active_global_ids(0.5).len() as u32;
            (vec![(region.id, count)], count)
        })
        .reduce(
            || (Vec::new(), 0u32),
            |(mut acc_map, acc_total), (map, total)| {
                acc_map.extend(map);
                (acc_map, acc_total + total)
            },
        );
    let active_counts: HashMap<RegionId, u32> = active_counts.into_iter().collect();

    TickResult {
        tick_number,
        active_counts,
        total_active,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::synapse::SynapseData;

    #[test]
    fn test_tick_no_activity() {
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        let pool = SynapsePool::new(152_000);
        let mut delay_buf = DelayBuffer::new();
        let gains = HashMap::new();

        let result = tick(&mut regions, &pool, &mut delay_buf, &gains, 0);
        assert_eq!(result.tick_number, 0);
        assert_eq!(result.total_active, 0);
    }

    #[test]
    fn test_tick_inject_and_fire() {
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        let pool = SynapsePool::new(152_000);
        let mut delay_buf = DelayBuffer::new();
        let gains = HashMap::new();

        // Inject strong signal into sensory neurons
        // Sensory threshold = 0.3, so 0.5 should fire after leak (0.5 * 0.85 = 0.425 > 0.3)
        regions[0].neurons.inject(&[(0, 0.5), (1, 0.5), (2, 0.5)]);

        let result = tick(&mut regions, &pool, &mut delay_buf, &gains, 0);

        // Should have some active neurons in sensory
        let sensory_active = result.active_counts.get(&RegionId::Sensory).copied().unwrap_or(0);
        assert!(sensory_active > 0, "Expected sensory neurons to fire");
    }

    #[test]
    fn test_tick_signal_propagation() {
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();

        // Create synapse from sensory neuron 0 → emotion neuron 70000
        let synapses = vec![SynapseData {
            from: 0,
            to: 70_000,
            weight: 0.8,
            delay: 1,
            plasticity: 1.0,
        }];
        let pool = SynapsePool::from_synapses(152_000, synapses);
        let mut delay_buf = DelayBuffer::new();
        let gains = HashMap::new();

        // Tick 0: Inject and fire sensory neuron 0
        regions[0].neurons.inject(&[(0, 0.5)]);
        let _r0 = tick(&mut regions, &pool, &mut delay_buf, &gains, 0);

        // Tick 1: Signal should propagate to emotion via delay buffer
        let _r1 = tick(&mut regions, &pool, &mut delay_buf, &gains, 1);

        // Emotion neuron should have received something
        // (may or may not fire depending on accumulated potential + threshold)
        // At minimum, the incoming buffer should have been written to
        // Check the emotion region (index 5) has non-zero potential
        let emotion_potential = regions[5].neurons.potentials[0];
        // The signal may have decayed, but there should be some trace
        // Given: activation=1.0 * weight=0.8 * sign=1.0 * gain=1.0 = 0.8
        // After leak in emotion (0.92): 0.8 * 0.92 = 0.736 > threshold 0.3
        // So it should have fired!
        let emotion_activation = regions[5].neurons.activations[0];
        assert!(
            emotion_activation > 0.0 || emotion_potential > 0.0,
            "Expected emotion neuron to receive signal. activation={}, potential={}",
            emotion_activation, emotion_potential
        );
    }
}
