/// Attention system: per-region gain map with three-drive computation and inertia.
///
/// Manages per-region attention gains that multiply incoming signals during
/// propagation. Python feeds three drives per region:
///   - Novelty: high prediction error → boost attention
///   - Threat: emotion-flagged urgency → force attention
///   - Relevance: executive top-down targets → bias attention
///
/// Each tick, gains move toward drive-computed targets with inertia,
/// simulating the cost of context-switching.

use crate::core::region::RegionId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize)]
pub struct AttentionSystem {
    /// Current gain per region (applied during propagation).
    gains: HashMap<RegionId, f32>,
    /// Target gain per region (what gains converge toward).
    targets: HashMap<RegionId, f32>,

    /// Per-region drive values (set by Python each tick).
    novelty_drives: HashMap<RegionId, f32>,
    threat_drives: HashMap<RegionId, f32>,
    relevance_drives: HashMap<RegionId, f32>,

    /// Inertia smoothing factor: 1.0 / inertia_ticks.
    inertia_alpha: f32,
    /// Minimum gain (maximum suppression).
    gain_min: f32,
    /// Maximum gain (hyper-focus).
    gain_max: f32,
    /// Drive combination weights (must sum to ~1.0).
    novelty_weight: f32,
    threat_weight: f32,
    relevance_weight: f32,
}

impl AttentionSystem {
    pub fn new(
        inertia_ticks: u32,
        gain_min: f32,
        gain_max: f32,
        novelty_weight: f32,
        threat_weight: f32,
        relevance_weight: f32,
    ) -> Self {
        let gains: HashMap<RegionId, f32> =
            RegionId::ALL.iter().map(|&id| (id, 1.0)).collect();
        let targets = gains.clone();
        let zeros: HashMap<RegionId, f32> =
            RegionId::ALL.iter().map(|&id| (id, 0.0)).collect();

        Self {
            gains,
            targets,
            novelty_drives: zeros.clone(),
            threat_drives: zeros.clone(),
            relevance_drives: zeros,
            inertia_alpha: 1.0 / inertia_ticks.max(1) as f32,
            gain_min,
            gain_max,
            novelty_weight,
            threat_weight,
            relevance_weight,
        }
    }

    /// Set all three drive values for a specific region.
    pub fn set_drives(&mut self, region: RegionId, novelty: f32, threat: f32, relevance: f32) {
        self.novelty_drives.insert(region, novelty);
        self.threat_drives.insert(region, threat);
        self.relevance_drives.insert(region, relevance);
    }

    /// Set a single gain directly (backward compat with set_attention_gain).
    pub fn set_gain_direct(&mut self, region: RegionId, gain: f32) {
        let clamped = gain.clamp(self.gain_min, self.gain_max);
        self.gains.insert(region, clamped);
        self.targets.insert(region, clamped);
    }

    /// Compute target gains from drives and move current gains toward targets.
    /// Call once per tick before propagation.
    pub fn update_gains(&mut self) {
        for &region_id in RegionId::ALL.iter() {
            let novelty = *self.novelty_drives.get(&region_id).unwrap_or(&0.0);
            let threat = *self.threat_drives.get(&region_id).unwrap_or(&0.0);
            let relevance = *self.relevance_drives.get(&region_id).unwrap_or(&0.0);

            // Weighted combination of drives
            let combined = novelty * self.novelty_weight
                + threat * self.threat_weight
                + relevance * self.relevance_weight;

            // Target: base gain (1.0) + drive-scaled boost (×4 for range coverage)
            let target = (1.0 + combined * 4.0).clamp(self.gain_min, self.gain_max);
            self.targets.insert(region_id, target);

            // Move current gain toward target with inertia
            let current = *self.gains.get(&region_id).unwrap_or(&1.0);
            let new_gain =
                (current + self.inertia_alpha * (target - current)).clamp(self.gain_min, self.gain_max);
            self.gains.insert(region_id, new_gain);
        }
    }

    /// Get the full gains map (for propagation).
    pub fn gains(&self) -> &HashMap<RegionId, f32> {
        &self.gains
    }

    /// Get gain for a specific region.
    pub fn gain_for(&self, region: RegionId) -> f32 {
        *self.gains.get(&region).unwrap_or(&1.0)
    }

    /// Get target gain for a region (for diagnostics).
    pub fn target_for(&self, region: RegionId) -> f32 {
        *self.targets.get(&region).unwrap_or(&1.0)
    }

    /// Get all three drive values for a region (for diagnostics).
    pub fn drives_for(&self, region: RegionId) -> (f32, f32, f32) {
        (
            *self.novelty_drives.get(&region).unwrap_or(&0.0),
            *self.threat_drives.get(&region).unwrap_or(&0.0),
            *self.relevance_drives.get(&region).unwrap_or(&0.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_gains_are_one() {
        let sys = AttentionSystem::new(15, 0.1, 5.0, 0.4, 0.4, 0.2);
        for &id in RegionId::ALL.iter() {
            assert!((sys.gain_for(id) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_novelty_drive_increases_gain() {
        let mut sys = AttentionSystem::new(1, 0.1, 5.0, 0.4, 0.4, 0.2);
        // Set high novelty drive for pattern region
        sys.set_drives(RegionId::Pattern, 1.0, 0.0, 0.0);
        sys.update_gains();
        // With inertia_ticks=1, alpha=1.0 so gain should jump to target
        // target = 1.0 + (1.0 * 0.4) * 4.0 = 2.6
        let gain = sys.gain_for(RegionId::Pattern);
        assert!(gain > 2.0, "Expected gain > 2.0, got {}", gain);
    }

    #[test]
    fn test_threat_drive_increases_gain() {
        let mut sys = AttentionSystem::new(1, 0.1, 5.0, 0.4, 0.4, 0.2);
        sys.set_drives(RegionId::Emotion, 0.0, 1.0, 0.0);
        sys.update_gains();
        let gain = sys.gain_for(RegionId::Emotion);
        assert!(gain > 2.0, "Expected gain > 2.0 from threat, got {}", gain);
    }

    #[test]
    fn test_inertia_slows_gain_change() {
        let mut sys = AttentionSystem::new(15, 0.1, 5.0, 0.4, 0.4, 0.2);
        sys.set_drives(RegionId::Visual, 1.0, 0.0, 0.0);
        // Single update with inertia_ticks=15 → alpha = 1/15 ≈ 0.067
        sys.update_gains();
        let gain = sys.gain_for(RegionId::Visual);
        // Should move only ~6.7% toward target (2.6)
        // new = 1.0 + 0.067 * (2.6 - 1.0) = 1.107
        assert!(gain > 1.0 && gain < 1.5, "Expected gradual change, got {}", gain);
    }

    #[test]
    fn test_gain_clamped_to_range() {
        let mut sys = AttentionSystem::new(1, 0.1, 5.0, 0.4, 0.4, 0.2);
        // Max all drives
        sys.set_drives(RegionId::Sensory, 10.0, 10.0, 10.0);
        sys.update_gains();
        let gain = sys.gain_for(RegionId::Sensory);
        assert!(gain <= 5.0, "Gain should be clamped to max 5.0, got {}", gain);

        // Set gain directly below range
        sys.set_gain_direct(RegionId::Sensory, -1.0);
        assert!((sys.gain_for(RegionId::Sensory) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_combined_drives() {
        let mut sys = AttentionSystem::new(1, 0.1, 5.0, 0.4, 0.4, 0.2);
        // All three drives at 0.5
        sys.set_drives(RegionId::Integration, 0.5, 0.5, 0.5);
        sys.update_gains();
        // combined = 0.5*0.4 + 0.5*0.4 + 0.5*0.2 = 0.5
        // target = 1.0 + 0.5 * 4.0 = 3.0
        let gain = sys.gain_for(RegionId::Integration);
        assert!((gain - 3.0).abs() < 0.1, "Expected ~3.0, got {}", gain);
    }

    #[test]
    fn test_convergence_over_ticks() {
        let mut sys = AttentionSystem::new(10, 0.1, 5.0, 0.4, 0.4, 0.2);
        sys.set_drives(RegionId::MemoryShort, 1.0, 0.0, 0.0);
        // target = 1.0 + 0.4 * 4.0 = 2.6
        for _ in 0..50 {
            sys.update_gains();
        }
        let gain = sys.gain_for(RegionId::MemoryShort);
        assert!((gain - 2.6).abs() < 0.1, "Expected convergence to ~2.6, got {}", gain);
    }
}
