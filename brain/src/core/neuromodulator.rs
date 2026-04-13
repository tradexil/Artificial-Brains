// core/neuromodulator.rs — Global neuromodulator state modulation.
//
// Four global signals that modulate all brain activity:
//   arousal  (0.0 = asleep, 1.0 = panic)
//   valence  (-1.0 = negative, 1.0 = positive)
//   focus    (0.0 = scattered, 1.0 = laser-focused)
//   energy   (0.0 = depleted, 1.0 = full)
//
// Arousal lowers firing thresholds. Focus narrows attention.
// Energy depletes with activity and gates consolidation.

use serde::{Deserialize, Serialize};

/// Global neuromodulator system.
#[derive(Clone, Serialize, Deserialize)]
pub struct NeuromodulatorSystem {
    pub arousal: f32,
    pub valence: f32,
    pub focus: f32,
    pub energy: f32,
    /// Energy cost per active neuron per tick.
    energy_cost_per_active: f32,
}

impl NeuromodulatorSystem {
    pub fn new() -> Self {
        Self {
            arousal: 0.5,
            valence: 0.0,
            focus: 0.5,
            energy: 1.0,
            energy_cost_per_active: 1e-7,
        }
    }

    pub fn set(&mut self, arousal: f32, valence: f32, focus: f32, energy: f32) {
        self.arousal = arousal.clamp(0.0, 1.0);
        self.valence = valence.clamp(-1.0, 1.0);
        self.focus = focus.clamp(0.0, 1.0);
        self.energy = energy.clamp(0.0, 1.0);
    }

    pub fn get(&self) -> (f32, f32, f32, f32) {
        (self.arousal, self.valence, self.focus, self.energy)
    }

    /// Compute a threshold modifier based on arousal.
    /// High arousal => lower thresholds (more sensitive, more firing).
    /// Returns a multiplier for the base threshold: 0.6–1.0 range.
    pub fn threshold_modifier(&self) -> f32 {
        // arousal 0.0 → modifier 1.0 (no change)
        // arousal 1.0 → modifier 0.6 (40% lower thresholds)
        1.0 - 0.4 * self.arousal
    }

    /// Update arousal based on emotion region firing rate.
    /// High emotion firing → arousal rises; low → decays toward baseline.
    pub fn update_arousal_from_emotion(&mut self, emotion_firing_rate: f32) {
        let target = emotion_firing_rate.clamp(0.0, 1.0);
        // EMA with alpha=0.1 toward target
        self.arousal = self.arousal * 0.9 + target * 0.1;
        self.arousal = self.arousal.clamp(0.0, 1.0);
    }

    /// Update valence from emotion polarity.
    pub fn update_valence_from_emotion(&mut self, polarity: f32) {
        // Slow EMA toward polarity
        self.valence = self.valence * 0.95 + polarity * 0.05;
        self.valence = self.valence.clamp(-1.0, 1.0);
    }

    /// Deplete energy proportional to number of active neurons.
    pub fn deplete_energy(&mut self, active_neuron_count: u32) {
        self.energy -= active_neuron_count as f32 * self.energy_cost_per_active;
        if self.energy < 0.0 {
            self.energy = 0.0;
        }
    }

    /// Slowly recover energy (rest/consolidation).
    pub fn recover_energy(&mut self, amount: f32) {
        self.energy = (self.energy + amount).min(1.0);
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromodulator_default() {
        let nm = NeuromodulatorSystem::new();
        assert!((nm.arousal - 0.5).abs() < 1e-6);
        assert!((nm.valence).abs() < 1e-6);
        assert!((nm.focus - 0.5).abs() < 1e-6);
        assert!((nm.energy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_clamps() {
        let mut nm = NeuromodulatorSystem::new();
        nm.set(2.0, -5.0, 3.0, -1.0);
        assert!((nm.arousal - 1.0).abs() < 1e-6);
        assert!((nm.valence - (-1.0)).abs() < 1e-6);
        assert!((nm.focus - 1.0).abs() < 1e-6);
        assert!((nm.energy).abs() < 1e-6);
    }

    #[test]
    fn test_threshold_modifier_low_arousal() {
        let mut nm = NeuromodulatorSystem::new();
        nm.arousal = 0.0;
        assert!((nm.threshold_modifier() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_threshold_modifier_high_arousal() {
        let mut nm = NeuromodulatorSystem::new();
        nm.arousal = 1.0;
        assert!((nm.threshold_modifier() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_arousal_update_from_emotion() {
        let mut nm = NeuromodulatorSystem::new();
        nm.arousal = 0.5;
        // High emotion firing
        for _ in 0..50 {
            nm.update_arousal_from_emotion(1.0);
        }
        assert!(nm.arousal > 0.9);
    }

    #[test]
    fn test_valence_update() {
        let mut nm = NeuromodulatorSystem::new();
        nm.valence = 0.0;
        for _ in 0..100 {
            nm.update_valence_from_emotion(0.8);
        }
        assert!(nm.valence > 0.5);
    }

    #[test]
    fn test_energy_depletion() {
        let mut nm = NeuromodulatorSystem::new();
        // 10000 active neurons per tick, 100 ticks
        for _ in 0..100 {
            nm.deplete_energy(10000);
        }
        assert!(nm.energy < 1.0);
        assert!(nm.energy >= 0.0);
    }

    #[test]
    fn test_energy_recovery() {
        let mut nm = NeuromodulatorSystem::new();
        nm.energy = 0.5;
        nm.recover_energy(0.3);
        assert!((nm.energy - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_energy_recovery_cap() {
        let mut nm = NeuromodulatorSystem::new();
        nm.energy = 0.9;
        nm.recover_energy(0.5);
        assert!((nm.energy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reset() {
        let mut nm = NeuromodulatorSystem::new();
        nm.set(0.8, -0.5, 0.9, 0.3);
        nm.reset();
        assert!((nm.arousal - 0.5).abs() < 1e-6);
        assert!((nm.energy - 1.0).abs() < 1e-6);
    }
}
