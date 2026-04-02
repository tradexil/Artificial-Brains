/// Integration region helpers: multi-region merging and binding evaluation.
///
/// The integration region (95000–104999) binds multi-modal input into
/// unified experiences. It fires when multiple regions are co-active,
/// with strength proportional to the number of active regions.
///
/// Temporal binding window: patterns within ±5 ticks are bound together.

use crate::core::region::{Region, RegionId};

/// Input regions that feed into integration.
const INPUT_REGIONS: [RegionId; 6] = [
    RegionId::Sensory,
    RegionId::Visual,
    RegionId::Audio,
    RegionId::Pattern,
    RegionId::Emotion,
    RegionId::Language,
];

/// Count how many input regions have active neurons.
pub fn count_active_input_regions(regions: &[Region], min_activation: f32) -> u32 {
    let mut count = 0;
    for region in regions {
        if INPUT_REGIONS.contains(&region.id)
            && !region.active_global_ids(min_activation).is_empty()
        {
            count += 1;
        }
    }
    count
}

/// Compute integration strength based on number of co-active input regions.
/// More co-active regions → stronger integration → richer binding.
pub fn integration_strength(n_active_regions: u32) -> f32 {
    match n_active_regions {
        0 => 0.0,
        1 => 0.1,
        2 => 0.3,
        3 => 0.6,
        4..=5 => 0.8,
        _ => 1.0,
    }
}

/// Boost integration region neurons proportionally to integration strength.
/// This creates a natural multi-modal convergence signal.
pub fn boost_integration_neurons(
    regions: &mut [Region],
    strength: f32,
    max_neurons: usize,
) -> u32 {
    let mut boosted = 0;
    for region in regions.iter_mut() {
        if region.id == RegionId::Integration {
            // Boost first N neurons proportional to strength
            let n = max_neurons.min(region.neurons.count as usize);
            for i in 0..n {
                if region.neurons.potentials[i] > 0.0 || strength > 0.3 {
                    region.neurons.potentials[i] += strength * 0.3;
                    boosted += 1;
                }
            }
            break;
        }
    }
    boosted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_activity_zero_strength() {
        let regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        assert_eq!(count_active_input_regions(&regions, 0.5), 0);
        assert_eq!(integration_strength(0), 0.0);
    }

    #[test]
    fn test_single_region_weak_integration() {
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        // Activate one visual neuron
        for r in &mut regions {
            if r.id == RegionId::Visual {
                r.neurons.activations[0] = 1.0;
            }
        }
        let count = count_active_input_regions(&regions, 0.5);
        assert_eq!(count, 1);
        assert!((integration_strength(count) - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_multi_region_strong_integration() {
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        for r in &mut regions {
            match r.id {
                RegionId::Visual => r.neurons.activations[0] = 1.0,
                RegionId::Audio => r.neurons.activations[0] = 1.0,
                RegionId::Sensory => r.neurons.activations[0] = 1.0,
                _ => {}
            }
        }
        let count = count_active_input_regions(&regions, 0.5);
        assert_eq!(count, 3);
        assert!((integration_strength(count) - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_integration_strength_range() {
        for n in 0..=10 {
            let s = integration_strength(n);
            assert!(s >= 0.0 && s <= 1.0, "n={} → strength={}", n, s);
        }
    }
}
