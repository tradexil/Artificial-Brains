/// Neuron model: Leaky Integrate-and-Fire (LIF)
///
/// Each neuron accumulates potential from incoming synapses,
/// leaks toward zero each tick, and fires when threshold is exceeded.

use serde::{Deserialize, Serialize};

const DEFAULT_ACTIVE_THRESHOLD: f32 = 0.5;

/// Excitatory neurons produce positive-weight synapses only.
/// Inhibitory neurons produce negative-weight synapses only (Dale's Law).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronType {
    Excitatory,
    Inhibitory,
}

/// Per-region parameters that apply to all neurons in that region.
/// Passed once at region construction, not stored per-neuron.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NeuronParams {
    pub threshold: f32,
    pub leak_rate: f32,
    pub refractory_period: u16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct UpdateStats {
    pub active_count: u32,
    pub fired_count: u32,
    pub positive_pre_leak_sum: f64,
    pub positive_leak_loss_sum: f64,
    pub positive_reset_sum: f64,
    pub positive_carried_sum: f64,
    pub refractory_ignored_abs_sum: f64,
    pub refractory_ignored_immediate_same_abs_sum: f64,
    pub refractory_ignored_immediate_cross_abs_sum: f64,
    pub refractory_ignored_delayed_same_abs_sum: f64,
    pub refractory_ignored_delayed_cross_abs_sum: f64,
    pub fire_interval_sum: f64,
    pub fire_interval_count: u64,
}

pub struct IncomingSourceSlices<'a> {
    pub immediate_same: &'a [f32],
    pub immediate_cross: &'a [f32],
    pub delayed_same: &'a [f32],
    pub delayed_cross: &'a [f32],
}

/// SoA (Structure-of-Arrays) storage for all neurons in one region.
/// This layout is cache-friendly and SIMD-friendly — we iterate
/// one field at a time across all neurons, not all fields for one neuron.
#[derive(Clone, Serialize, Deserialize)]
pub struct NeuronStorage {
    pub count: u32,
    pub neuron_types: Vec<NeuronType>,

    // Dynamics — change every tick
    pub potentials: Vec<f32>,
    pub activations: Vec<f32>,
    pub prev_activations: Vec<f32>, // double-buffered for cross-region reads
    pub prev_active_local_ids: Vec<u32>,
    pub prev_active_cache_valid: bool,
    pub refractory: Vec<u16>,
    pub last_fire_ticks: Vec<u64>,
    pub has_fired_before: Vec<bool>,

    // Parameters — uniform per region
    pub params: NeuronParams,
}

impl NeuronStorage {
    /// Create neuron storage for a region.
    /// `inhibitory_pct` controls what fraction is inhibitory (e.g. 0.20 = 20%).
    pub fn new(count: u32, params: NeuronParams, inhibitory_pct: f32) -> Self {
        let n = count as usize;
        let inhib_count = (n as f32 * inhibitory_pct) as usize;

        let mut neuron_types = vec![NeuronType::Excitatory; n];
        // Place inhibitory neurons at the end of the range
        for i in (n - inhib_count)..n {
            neuron_types[i] = NeuronType::Inhibitory;
        }

        Self {
            count,
            neuron_types,
            potentials: vec![0.0; n],
            activations: vec![0.0; n],
            prev_activations: vec![0.0; n],
            prev_active_local_ids: Vec::new(),
            prev_active_cache_valid: false,
            refractory: vec![0; n],
            last_fire_ticks: vec![0; n],
            has_fired_before: vec![false; n],
            params,
        }
    }

    /// Swap activations into prev_activations buffer.
    /// Called once at the start of each tick so that cross-region reads
    /// see the PREVIOUS tick's state while we compute the new one.
    #[inline]
    pub fn swap_activation_buffers(&mut self) {
        std::mem::swap(&mut self.activations, &mut self.prev_activations);
        self.prev_active_local_ids.clear();
        self.prev_active_local_ids.reserve(self.count_active_in_prev_hint());
        for (local_idx, &activation) in self.prev_activations.iter().enumerate() {
            if activation > 0.0 {
                self.prev_active_local_ids.push(local_idx as u32);
            }
        }
        self.prev_active_cache_valid = true;
        // Clear current activations — they'll be recomputed this tick
        self.activations.fill(0.0);
    }

    /// Apply the LIF update to all neurons in this region.
    /// `incoming` is a slice of accumulated input for each neuron
    /// (already multiplied by synapse weights and attention gain).
    /// Length must equal self.count.
    pub fn update(&mut self, incoming: &[f32]) -> UpdateStats {
        self.update_with_sources_at(incoming, None, 0)
    }

    pub fn update_with_sources_at(
        &mut self,
        incoming: &[f32],
        incoming_sources: Option<IncomingSourceSlices<'_>>,
        tick_number: u64,
    ) -> UpdateStats {
        debug_assert_eq!(incoming.len(), self.count as usize);
        if let Some(source_slices) = &incoming_sources {
            debug_assert_eq!(source_slices.immediate_same.len(), incoming.len());
            debug_assert_eq!(source_slices.immediate_cross.len(), incoming.len());
            debug_assert_eq!(source_slices.delayed_same.len(), incoming.len());
            debug_assert_eq!(source_slices.delayed_cross.len(), incoming.len());
        }

        let threshold = self.params.threshold;
        let leak = self.params.leak_rate;
        let refr_period = self.params.refractory_period;

        let mut active_count = 0u32;
        let mut fired_count = 0u32;
        let mut positive_pre_leak_sum = 0.0f64;
        let mut positive_leak_loss_sum = 0.0f64;
        let mut positive_reset_sum = 0.0f64;
        let mut positive_carried_sum = 0.0f64;
        let mut refractory_ignored_abs_sum = 0.0f64;
        let mut refractory_ignored_immediate_same_abs_sum = 0.0f64;
        let mut refractory_ignored_immediate_cross_abs_sum = 0.0f64;
        let mut refractory_ignored_delayed_same_abs_sum = 0.0f64;
        let mut refractory_ignored_delayed_cross_abs_sum = 0.0f64;
        let mut fire_interval_sum = 0.0f64;
        let mut fire_interval_count = 0u64;
        for i in 0..self.count as usize {
            // Refractory period — can't fire, activation decays
            if self.refractory[i] > 0 {
                if let Some(source_slices) = &incoming_sources {
                    let ignored_immediate_same = source_slices.immediate_same[i].abs() as f64;
                    let ignored_immediate_cross = source_slices.immediate_cross[i].abs() as f64;
                    let ignored_delayed_same = source_slices.delayed_same[i].abs() as f64;
                    let ignored_delayed_cross = source_slices.delayed_cross[i].abs() as f64;
                    refractory_ignored_immediate_same_abs_sum += ignored_immediate_same;
                    refractory_ignored_immediate_cross_abs_sum += ignored_immediate_cross;
                    refractory_ignored_delayed_same_abs_sum += ignored_delayed_same;
                    refractory_ignored_delayed_cross_abs_sum += ignored_delayed_cross;
                    refractory_ignored_abs_sum += ignored_immediate_same
                        + ignored_immediate_cross
                        + ignored_delayed_same
                        + ignored_delayed_cross;
                }
                self.refractory[i] -= 1;
                self.activations[i] = self.prev_activations[i] * 0.5;
                if self.potentials[i] > 0.0 {
                    positive_carried_sum += self.potentials[i] as f64;
                }
            } else {
                // Accumulate incoming signal
                self.potentials[i] += incoming[i];
                let pre_leak_potential = self.potentials[i];
                if pre_leak_potential > 0.0 {
                    positive_pre_leak_sum += pre_leak_potential as f64;
                }

                // Leak toward zero
                self.potentials[i] *= leak;
                let post_leak_potential = self.potentials[i];
                if pre_leak_potential > 0.0 {
                    positive_leak_loss_sum +=
                        ((pre_leak_potential as f64) - (post_leak_potential.max(0.0) as f64))
                            .max(0.0);
                }

                // Fire if above threshold
                if self.potentials[i] >= threshold {
                    self.activations[i] = 1.0;
                    if post_leak_potential > 0.0 {
                        positive_reset_sum += post_leak_potential as f64;
                    }
                    if self.has_fired_before[i] {
                        fire_interval_sum +=
                            tick_number.saturating_sub(self.last_fire_ticks[i]) as f64;
                        fire_interval_count += 1;
                    }
                    self.last_fire_ticks[i] = tick_number;
                    self.has_fired_before[i] = true;
                    self.potentials[i] = 0.0; // reset
                    self.refractory[i] = refr_period;
                    fired_count += 1;
                } else {
                    // Decay activation if didn't fire
                    self.activations[i] = self.prev_activations[i] * 0.8;
                    if self.potentials[i] > 0.0 {
                        positive_carried_sum += self.potentials[i] as f64;
                    }
                }
            }

            if self.activations[i] > DEFAULT_ACTIVE_THRESHOLD {
                active_count += 1;
            }
        }

        UpdateStats {
            active_count,
            fired_count,
            positive_pre_leak_sum,
            positive_leak_loss_sum,
            positive_reset_sum,
            positive_carried_sum,
            refractory_ignored_abs_sum,
            refractory_ignored_immediate_same_abs_sum,
            refractory_ignored_immediate_cross_abs_sum,
            refractory_ignored_delayed_same_abs_sum,
            refractory_ignored_delayed_cross_abs_sum,
            fire_interval_sum,
            fire_interval_count,
        }
    }

    /// Inject external activation into specific neurons (for input regions).
    /// `signals` is a list of (local_index, activation_value).
    pub fn inject(&mut self, signals: &[(u32, f32)]) {
        for &(idx, val) in signals {
            if (idx as usize) < self.count as usize {
                self.potentials[idx as usize] += val;
            }
        }
    }

    /// Return indices of neurons with activation > min_activation.
    /// This is the "active list" for sparse iteration.
    pub fn active_neurons(&self, min_activation: f32) -> Vec<u32> {
        self.activations
            .iter()
            .enumerate()
            .filter(|(_, &a)| a > min_activation)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Count neurons with activation > min_activation without allocating.
    pub fn count_active(&self, min_activation: f32) -> u32 {
        self.activations
            .iter()
            .filter(|&&activation| activation > min_activation)
            .count() as u32
    }

    /// Reset all dynamics to zero (for testing / reinitialization).
    pub fn reset(&mut self) {
        self.potentials.fill(0.0);
        self.activations.fill(0.0);
        self.prev_activations.fill(0.0);
        self.prev_active_local_ids.clear();
        self.prev_active_cache_valid = false;
        self.refractory.fill(0);
        self.last_fire_ticks.fill(0);
        self.has_fired_before.fill(false);
    }

    fn count_active_in_prev_hint(&self) -> usize {
        self.prev_activations.iter().filter(|&&activation| activation > 0.0).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> NeuronParams {
        NeuronParams {
            threshold: 0.5,
            leak_rate: 0.9,
            refractory_period: 3,
        }
    }

    #[test]
    fn test_creation() {
        let storage = NeuronStorage::new(100, default_params(), 0.2);
        assert_eq!(storage.count, 100);
        // 20% inhibitory = 20 neurons at the end
        assert_eq!(storage.neuron_types[79], NeuronType::Excitatory);
        assert_eq!(storage.neuron_types[80], NeuronType::Inhibitory);
        assert_eq!(storage.neuron_types[99], NeuronType::Inhibitory);
    }

    #[test]
    fn test_firing() {
        let mut storage = NeuronStorage::new(10, default_params(), 0.0);
        let mut incoming = vec![0.0; 10];

        // Push neuron 0 above threshold (0.5)
        // After leak: 0.6 * 0.9 = 0.54, still above 0.5 → fires
        incoming[0] = 0.6;
        let stats = storage.update(&incoming);

        assert_eq!(storage.activations[0], 1.0);
        assert_eq!(storage.potentials[0], 0.0); // reset after fire
        assert_eq!(storage.refractory[0], 3);   // refractory set
        assert_eq!(stats.fired_count, 1);
        assert_eq!(stats.active_count, 1);
        assert!((stats.positive_pre_leak_sum - 0.6).abs() < 0.0001);
        assert!((stats.positive_leak_loss_sum - 0.06).abs() < 0.0001);
        assert!((stats.positive_reset_sum - 0.54).abs() < 0.0001);
        assert!((stats.positive_carried_sum - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_subthreshold() {
        let mut storage = NeuronStorage::new(10, default_params(), 0.0);
        let mut incoming = vec![0.0; 10];

        // Below threshold after leak: 0.4 * 0.9 = 0.36 < 0.5
        incoming[0] = 0.4;
        let stats = storage.update(&incoming);

        assert!(storage.activations[0] < 0.01); // didn't fire (prev was 0)
        assert!((storage.potentials[0] - 0.36).abs() < 0.01);
        assert_eq!(stats.fired_count, 0);
        assert!((stats.positive_pre_leak_sum - 0.4).abs() < 0.0001);
        assert!((stats.positive_leak_loss_sum - 0.04).abs() < 0.0001);
        assert!((stats.positive_reset_sum - 0.0).abs() < 0.0001);
        assert!((stats.positive_carried_sum - 0.36).abs() < 0.0001);
    }

    #[test]
    fn test_refractory() {
        let mut storage = NeuronStorage::new(10, default_params(), 0.0);
        let mut incoming = vec![0.0; 10];

        // Fire neuron 0
        incoming[0] = 0.6;
        storage.update(&incoming);
        assert_eq!(storage.activations[0], 1.0);

        // Swap buffers for next tick
        storage.swap_activation_buffers();

        // Next tick — still refractory, even with strong input
        incoming[0] = 1.0;
        let stats = storage.update(&incoming);
        assert!(storage.activations[0] < 1.0); // decayed, not freshly fired
        assert_eq!(storage.refractory[0], 2);  // decremented from 3
        assert_eq!(stats.fired_count, 0);
        assert!((stats.positive_pre_leak_sum - 0.0).abs() < 0.0001);
        assert!((stats.positive_carried_sum - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_potential_accumulation() {
        let mut storage = NeuronStorage::new(10, default_params(), 0.0);
        let mut incoming = vec![0.0; 10];

        // Two sub-threshold ticks accumulate
        incoming[0] = 0.3; // 0.3 * 0.9 = 0.27
        let stats = storage.update(&incoming);
        assert!(storage.activations[0] < 0.01);
        assert_eq!(stats.fired_count, 0);

        incoming[0] = 0.3; // (0.27 + 0.3) * 0.9 = 0.513 > 0.5 → fire!
        let stats = storage.update(&incoming);
        assert_eq!(storage.activations[0], 1.0);
        assert_eq!(stats.fired_count, 1);
    }

    #[test]
    fn test_inject() {
        let mut storage = NeuronStorage::new(10, default_params(), 0.0);
        storage.inject(&[(3, 0.7)]);
        assert!((storage.potentials[3] - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_active_neurons() {
        let mut storage = NeuronStorage::new(10, default_params(), 0.0);
        let mut incoming = vec![0.0; 10];
        incoming[2] = 0.6;
        incoming[7] = 0.8;
        storage.update(&incoming);

        let active = storage.active_neurons(0.5);
        assert!(active.contains(&2));
        assert!(active.contains(&7));
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_update_returns_active_count() {
        let mut storage = NeuronStorage::new(10, default_params(), 0.0);
        let mut incoming = vec![0.0; 10];
        incoming[1] = 0.6;
        incoming[4] = 0.8;

        let stats = storage.update(&incoming);

        assert_eq!(stats.active_count, 2);
        assert_eq!(stats.fired_count, 2);
        assert_eq!(storage.count_active(0.5), 2);
        assert!(stats.positive_pre_leak_sum > 0.0);
        assert!(stats.positive_reset_sum > 0.0);
    }

    #[test]
    fn test_refractory_ignored_input_breakdown() {
        let mut storage = NeuronStorage::new(1, default_params(), 0.0);
        let incoming = vec![0.6];
        storage.update_with_sources_at(&incoming, None, 0);
        storage.swap_activation_buffers();

        let source_slices = IncomingSourceSlices {
            immediate_same: &[0.0],
            immediate_cross: &[0.25],
            delayed_same: &[0.75],
            delayed_cross: &[0.0],
        };
        let ignored = storage.update_with_sources_at(&[1.0], Some(source_slices), 1);

        assert!((ignored.refractory_ignored_abs_sum - 1.0).abs() < 0.0001);
        assert!((ignored.refractory_ignored_immediate_cross_abs_sum - 0.25).abs() < 0.0001);
        assert!((ignored.refractory_ignored_delayed_same_abs_sum - 0.75).abs() < 0.0001);
        assert_eq!(ignored.fire_interval_count, 0);
    }

    #[test]
    fn test_fire_interval_accumulates() {
        let mut storage = NeuronStorage::new(1, default_params(), 0.0);

        storage.update_with_sources_at(&[0.6], None, 0);
        for tick in 1..=4 {
            storage.swap_activation_buffers();
            let incoming = if tick == 4 { [0.6] } else { [0.0] };
            let stats = storage.update_with_sources_at(&incoming, None, tick);
            if tick == 4 {
                assert_eq!(stats.fire_interval_count, 1);
                assert!((stats.fire_interval_sum - 4.0).abs() < 0.0001);
            }
        }
    }
}
