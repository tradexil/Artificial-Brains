/// Attention region helpers: extract drive signals from brain activity.
///
/// Computes threat and relevance drives from region activation patterns.
/// Novelty drive comes from prediction error (computed separately).
///
/// - Threat drive: derived from emotion region firing rate.
///   High emotion firing → high threat → attention override.
/// - Relevance drive: derived from executive region activity.
///   Executive focus → top-down attention bias.

use crate::core::region::{Region, RegionId};

/// Compute threat drive from emotion region firing rate.
///
/// When emotion region fires heavily, threat is high — forcing attention
/// globally (emotional override). Returns 0.0–1.0.
pub fn compute_threat_drive(regions: &[Region]) -> f32 {
    for region in regions {
        if region.id == RegionId::Emotion {
            let active = region.active_global_ids(0.5).len() as f32;
            let rate = active / region.neurons.count.max(1) as f32;
            // Scale: emotion firing rate > 5% → escalating threat
            return (rate * 10.0).min(1.0);
        }
    }
    0.0
}

/// Compute relevance drive from executive region activity.
///
/// When executive region is active, it signals top-down attention targets.
/// Returns 0.0–1.0.
pub fn compute_relevance_drive(regions: &[Region]) -> f32 {
    for region in regions {
        if region.id == RegionId::Executive {
            let active = region.active_global_ids(0.5).len() as f32;
            let rate = active / region.neurons.count.max(1) as f32;
            return (rate * 10.0).min(1.0);
        }
    }
    0.0
}

/// Compute per-region novelty from prediction errors.
///
/// Each region's novelty drive = its prediction error (already 0.0–1.0).
/// This is a passthrough for clarity — the actual error computation
/// happens in PredictionState.
pub fn novelty_from_errors(
    errors: &std::collections::HashMap<RegionId, f32>,
    region: RegionId,
) -> f32 {
    *errors.get(&region).unwrap_or(&0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threat_drive_no_activity() {
        let regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        let threat = compute_threat_drive(&regions);
        assert!(threat.abs() < 1e-6, "No activity → zero threat");
    }

    #[test]
    fn test_threat_drive_with_emotion_activity() {
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        // Activate 10% of emotion neurons (indices 0..999 out of 10000)
        for region in &mut regions {
            if region.id == RegionId::Emotion {
                for i in 0..1000 {
                    region.neurons.activations[i] = 1.0;
                }
            }
        }
        let threat = compute_threat_drive(&regions);
        // 1000/10000 = 10% → 0.1 * 10 = 1.0
        assert!(threat > 0.5, "High emotion activity → high threat: {}", threat);
    }

    #[test]
    fn test_relevance_drive_no_activity() {
        let regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        let rel = compute_relevance_drive(&regions);
        assert!(rel.abs() < 1e-6, "No activity → zero relevance");
    }

    #[test]
    fn test_novelty_from_errors() {
        let mut errors = std::collections::HashMap::new();
        errors.insert(RegionId::Visual, 0.7);
        assert!((novelty_from_errors(&errors, RegionId::Visual) - 0.7).abs() < 1e-6);
        assert!(novelty_from_errors(&errors, RegionId::Audio).abs() < 1e-6);
    }
}
