/// Tick cycle orchestration.
///
/// Each tick has phases:
///   Phase 1: Propagate — signals flow through synapses (rayon-parallel)
///   Phase 2: Update — neurons integrate and fire (rayon-parallel)
///
/// Phase 3 (Learn) and Phase 4 (Maintain) are driven by Python.

use crate::core::propagate::{
    deliver_wave_signals, propagate_wave, DelayBuffer, DeliveryStats, PropagationStats,
    SameRegionDelayAblation,
};
use crate::core::region::{LaneId, Region, RegionId};
use crate::core::synapse::SynapsePool;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

const DEFAULT_ACTIVE_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone, Default)]
pub struct TickProfile {
    pub prepare_ms: f64,
    pub delayed_delivery_ms: f64,
    pub propagate_ms: f64,
    pub update_ms: f64,
    pub incoming_signal_count: u64,
    pub incoming_signal_abs_sum: f64,
    pub immediate_signal_count: u64,
    pub immediate_signal_abs_sum: f64,
    pub delayed_delivery_signal_count: u64,
    pub delayed_delivery_signal_abs_sum: f64,
    pub scheduled_delayed_signal_count: u64,
    pub scheduled_delayed_signal_abs_sum: f64,
    pub total_fired: u32,
    pub incoming_region_signal_counts: [u64; 14],
    pub incoming_region_signal_abs_sums: [f64; 14],
    pub fired_region_counts: [u32; 14],
    pub delayed_flow_signal_counts: [[u64; 14]; 14],
    pub delayed_flow_signal_abs_sums: [[f64; 14]; 14],
    pub region_positive_pre_leak_sums: [f64; 14],
    pub region_positive_leak_loss_sums: [f64; 14],
    pub region_positive_reset_sums: [f64; 14],
    pub region_positive_carried_sums: [f64; 14],
    pub region_refractory_ignored_abs_sums: [f64; 14],
    pub region_refractory_ignored_immediate_same_abs_sums: [f64; 14],
    pub region_refractory_ignored_immediate_cross_abs_sums: [f64; 14],
    pub region_refractory_ignored_delayed_same_abs_sums: [f64; 14],
    pub region_refractory_ignored_delayed_cross_abs_sums: [f64; 14],
    pub region_fire_interval_sums: [f64; 14],
    pub region_fire_interval_counts: [u64; 14],
}

impl TickProfile {
    pub fn total_ms(&self) -> f64 {
        self.prepare_ms + self.delayed_delivery_ms + self.propagate_ms + self.update_ms
    }

    fn record_delayed_delivery(&mut self, stats: &DeliveryStats) {
        self.incoming_signal_count += stats.signal_count;
        self.incoming_signal_abs_sum += stats.signal_abs_sum;
        self.delayed_delivery_signal_count += stats.signal_count;
        self.delayed_delivery_signal_abs_sum += stats.signal_abs_sum;
        for idx in 0..RegionId::ALL.len() {
            self.incoming_region_signal_counts[idx] += stats.target_signal_counts[idx];
            self.incoming_region_signal_abs_sums[idx] += stats.target_signal_abs_sums[idx];
        }
    }

    fn record_propagation(&mut self, stats: &PropagationStats) {
        self.incoming_signal_count += stats.immediate_signal_count;
        self.incoming_signal_abs_sum += stats.immediate_signal_abs_sum;
        self.immediate_signal_count += stats.immediate_signal_count;
        self.immediate_signal_abs_sum += stats.immediate_signal_abs_sum;
        self.scheduled_delayed_signal_count += stats.scheduled_delayed_signal_count;
        self.scheduled_delayed_signal_abs_sum += stats.scheduled_delayed_signal_abs_sum;
        for idx in 0..RegionId::ALL.len() {
            self.incoming_region_signal_counts[idx] += stats.target_signal_counts[idx];
            self.incoming_region_signal_abs_sums[idx] += stats.target_signal_abs_sums[idx];
            for target_idx in 0..RegionId::ALL.len() {
                self.delayed_flow_signal_counts[idx][target_idx] +=
                    stats.delayed_flow_signal_counts[idx][target_idx];
                self.delayed_flow_signal_abs_sums[idx][target_idx] +=
                    stats.delayed_flow_signal_abs_sums[idx][target_idx];
            }
        }
    }
}

/// Result of a single tick — stats for Python to consume.
#[derive(Debug, Clone)]
pub struct TickResult {
    pub tick_number: u64,
    pub active_counts: HashMap<RegionId, u32>,
    pub total_active: u32,
    pub profile: TickProfile,
}

#[inline]
fn lane_activity_mask(tick_number: u64) -> [bool; LaneId::ALL.len()] {
    let mut active = [false; LaneId::ALL.len()];
    for lane in LaneId::ALL {
        active[lane.index()] = lane.is_scheduled_on(tick_number);
    }
    active
}

#[inline]
fn region_activity_mask(tick_number: u64) -> [bool; RegionId::ALL.len()] {
    let mut active = [false; RegionId::ALL.len()];
    for region_id in RegionId::ALL {
        active[region_id.index()] = region_id.is_scheduled_on(tick_number);
    }
    active
}

/// Execute one complete tick cycle.
///
/// 1. Prepare one lane at a time (swap activation buffers for that lane only)
/// 2. Deliver any mailbox signals ready for that lane
/// 3. Propagate only from that lane's previous-tick activations
/// 4. Update only that lane's neurons
/// 5. Advance the delayed ring once after all waves complete
pub fn tick(
    regions: &mut Vec<Region>,
    synapse_pool: &SynapsePool,
    delay_buffer: &mut DelayBuffer,
    attention_gains: &HashMap<RegionId, f32>,
    same_region_delay_ablation: &SameRegionDelayAblation,
    tick_number: u64,
) -> TickResult {
    let mut profile = TickProfile::default();
    let mut active_counts = HashMap::new();
    let lane_active = lane_activity_mask(tick_number);
    let region_active = region_activity_mask(tick_number);

    for lane in LaneId::ALL {
        if !lane_active[lane.index()] {
            delay_buffer.carry_current_wave_to_next(lane);
            continue;
        }

        let phase_started = Instant::now();
        regions
            .par_iter_mut()
            .filter(|region| region.id.lane() == lane && region_active[region.id.index()])
            .for_each(|region| {
                region.pre_tick();
            });
        profile.prepare_ms += phase_started.elapsed().as_secs_f64() * 1000.0;

        let phase_started = Instant::now();
        let delayed_stats = deliver_wave_signals(regions, delay_buffer, lane, &region_active);
        profile.record_delayed_delivery(&delayed_stats);
        profile.delayed_delivery_ms += phase_started.elapsed().as_secs_f64() * 1000.0;

        let phase_started = Instant::now();
        let propagation_stats = propagate_wave(
            regions,
            synapse_pool,
            delay_buffer,
            attention_gains,
            same_region_delay_ablation,
            lane,
            &region_active,
        );
        profile.record_propagation(&propagation_stats);
        profile.propagate_ms += phase_started.elapsed().as_secs_f64() * 1000.0;

        let phase_started = Instant::now();
        let update_stats_vec: Vec<(RegionId, crate::core::neuron::UpdateStats)> = regions
            .par_iter_mut()
            .filter(|region| region.id.lane() == lane && region_active[region.id.index()])
            .map(|region| {
                let stats = region.update_neurons_at(tick_number);
                (region.id, stats)
            })
            .collect();
        profile.update_ms += phase_started.elapsed().as_secs_f64() * 1000.0;

        for (region_id, stats) in &update_stats_vec {
            active_counts.insert(*region_id, stats.active_count);
            profile.total_fired += stats.fired_count;
            profile.fired_region_counts[region_id.index()] = stats.fired_count;
            profile.region_positive_pre_leak_sums[region_id.index()] = stats.positive_pre_leak_sum;
            profile.region_positive_leak_loss_sums[region_id.index()] = stats.positive_leak_loss_sum;
            profile.region_positive_reset_sums[region_id.index()] = stats.positive_reset_sum;
            profile.region_positive_carried_sums[region_id.index()] = stats.positive_carried_sum;
            profile.region_refractory_ignored_abs_sums[region_id.index()] =
                stats.refractory_ignored_abs_sum;
            profile.region_refractory_ignored_immediate_same_abs_sums[region_id.index()] =
                stats.refractory_ignored_immediate_same_abs_sum;
            profile.region_refractory_ignored_immediate_cross_abs_sums[region_id.index()] =
                stats.refractory_ignored_immediate_cross_abs_sum;
            profile.region_refractory_ignored_delayed_same_abs_sums[region_id.index()] =
                stats.refractory_ignored_delayed_same_abs_sum;
            profile.region_refractory_ignored_delayed_cross_abs_sums[region_id.index()] =
                stats.refractory_ignored_delayed_cross_abs_sum;
            profile.region_fire_interval_sums[region_id.index()] = stats.fire_interval_sum;
            profile.region_fire_interval_counts[region_id.index()] = stats.fire_interval_count;
        }
    }

    for region in regions.iter() {
        active_counts
            .entry(region.id)
            .or_insert_with(|| region.active_count(DEFAULT_ACTIVE_THRESHOLD));
    }

    let total_active = active_counts.values().copied().sum();

    delay_buffer.advance();

    TickResult {
        tick_number,
        active_counts,
        total_active,
        profile,
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

        let result = tick(
            &mut regions,
            &pool,
            &mut delay_buf,
            &gains,
            &SameRegionDelayAblation::default(),
            0,
        );
        assert_eq!(result.tick_number, 0);
        assert_eq!(result.total_active, 0);
        assert_eq!(result.profile.incoming_signal_count, 0);
        assert_eq!(result.profile.total_fired, 0);
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

        let result = tick(
            &mut regions,
            &pool,
            &mut delay_buf,
            &gains,
            &SameRegionDelayAblation::default(),
            0,
        );

        // Should have some active neurons in sensory
        let sensory_active = result.active_counts.get(&RegionId::Sensory).copied().unwrap_or(0);
        assert!(sensory_active > 0, "Expected sensory neurons to fire");
        assert!(result.profile.total_fired > 0);
        assert!(result.profile.fired_region_counts[RegionId::Sensory.index()] > 0);
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
        let _r0 = tick(
            &mut regions,
            &pool,
            &mut delay_buf,
            &gains,
            &SameRegionDelayAblation::default(),
            0,
        );

        // Tick 1: cognition is off-cadence, so the ready signal is carried forward.
        let r1 = tick(
            &mut regions,
            &pool,
            &mut delay_buf,
            &gains,
            &SameRegionDelayAblation::default(),
            1,
        );

        assert_eq!(r1.profile.incoming_signal_count, 0);

        // Tick 2: cognition runs again, so the delayed signal should arrive.
        let r2 = tick(
            &mut regions,
            &pool,
            &mut delay_buf,
            &gains,
            &SameRegionDelayAblation::default(),
            2,
        );

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
        assert!(r2.profile.incoming_signal_count > 0);
        assert!(r2.profile.incoming_region_signal_counts[RegionId::Emotion.index()] > 0);
    }
}
