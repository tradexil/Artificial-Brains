// regions/sensory.rs — Sensory region logic.
//
// Sensory region (0–9999, 10k neurons, 15% inhibitory):
//   Receives raw sensor values (temperature, pressure, pain, texture).
//   Uses population coding: each value activates a gaussian bump of neurons.
//
// Sub-ranges:
//   0–2499:    temperature (cold → hot gradient)
//   2500–4999: pressure (light → heavy)
//   5000–7499: pain (none → severe)
//   7500–9999: texture (smooth → rough)

use crate::core::region::{Region, RegionId};

const SENSORY_START: u32 = 0;
const SENSORY_END: u32 = 10_000;
const SENSORY_COUNT: u32 = 10_000;

// Sub-range boundaries
const TEMP_START: u32 = 0;
const TEMP_END: u32 = 2_500;
const PRESSURE_START: u32 = 2_500;
const PRESSURE_END: u32 = 5_000;
const PAIN_START: u32 = 5_000;
const PAIN_END: u32 = 7_500;
const TEXTURE_START: u32 = 7_500;
const TEXTURE_END: u32 = 10_000;

/// Generate population-coded neuron activations for a value in [0.0, 1.0].
/// Returns (global_id, activation) pairs — a gaussian bump centered on value.
pub fn population_code(
    value: f32,
    sub_start: u32,
    sub_end: u32,
    spread: u32,
) -> Vec<(u32, f32)> {
    let range = sub_end - sub_start;
    if range == 0 {
        return Vec::new();
    }

    let value = value.clamp(0.0, 1.0);
    let center = sub_start as f32 + value * (range - 1) as f32;
    let sigma = spread as f32;
    let sigma_sq = sigma * sigma;

    let start = if center as u32 > sub_start + spread * 3 {
        center as u32 - spread * 3
    } else {
        sub_start
    };
    let end = (center as u32 + spread * 3 + 1).min(sub_end);

    let mut result = Vec::new();
    for gid in start..end {
        let dist = (gid as f32 - center).abs();
        let activation = (-0.5 * dist * dist / sigma_sq).exp();
        if activation > 0.01 {
            result.push((gid, activation));
        }
    }
    result
}

/// Generate population codes for all four sensory modalities.
/// Each value in [0.0, 1.0]. spread is the gaussian width in neurons.
pub fn encode_sensory(
    temperature: f32,
    pressure: f32,
    pain: f32,
    texture: f32,
    spread: u32,
) -> Vec<(u32, f32)> {
    let mut result = Vec::new();
    result.extend(population_code(temperature, TEMP_START, TEMP_END, spread));
    result.extend(population_code(pressure, PRESSURE_START, PRESSURE_END, spread));
    result.extend(population_code(pain, PAIN_START, PAIN_END, spread));
    result.extend(population_code(texture, TEXTURE_START, TEXTURE_END, spread));
    result
}

/// Compute overall sensory activation strength (0.0–1.0).
pub fn sensory_activation_strength(regions: &[Region], min_activation: f32) -> f32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Sensory) {
        Some(i) => i,
        None => return 0.0,
    };

    let active = regions[idx].active_global_ids(min_activation).len() as f32;
    (active / (SENSORY_COUNT as f32 * 0.05)).min(1.0)
}

/// Boost specific sensory neurons. Returns count boosted.
pub fn boost_sensory_neurons(
    regions: &mut [Region],
    neurons: &[u32],
    boost: f32,
) -> u32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Sensory) {
        Some(i) => i,
        None => return 0,
    };

    let mut count = 0u32;
    for &gid in neurons {
        if gid >= SENSORY_START && gid < SENSORY_END {
            if let Some(local) = regions[idx].global_to_local(gid) {
                regions[idx].neurons.activations[local as usize] += boost;
                count += 1;
            }
        }
    }
    count
}

/// Get top-K active sensory neurons.
pub fn peak_sensory_neurons(
    regions: &[Region],
    min_activation: f32,
    top_k: usize,
) -> Vec<(u32, f32)> {
    let idx = match regions.iter().position(|r| r.id == RegionId::Sensory) {
        Some(i) => i,
        None => return Vec::new(),
    };

    let region = &regions[idx];
    let excit_end = (SENSORY_COUNT as f32 * 0.85) as usize; // 85% excitatory
    let mut active: Vec<(u32, f32)> = Vec::new();

    for i in 0..excit_end.min(region.neurons.count as usize) {
        let act = region.neurons.activations[i];
        if act > min_activation {
            active.push((region.local_to_global(i as u32), act));
        }
    }

    active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    active.truncate(top_k);
    active
}

/// Detect pain level from sensory activations (0.0–1.0).
/// Reads the pain sub-range (5000–7499) firing rate.
pub fn detect_pain_level(regions: &[Region], min_activation: f32) -> f32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Sensory) {
        Some(i) => i,
        None => return 0.0,
    };

    let region = &regions[idx];
    let pain_local_start = (PAIN_START - SENSORY_START) as usize;
    let pain_local_end = (PAIN_END - SENSORY_START) as usize;
    let pain_range = pain_local_end - pain_local_start;

    let mut active_count = 0u32;
    for i in pain_local_start..pain_local_end.min(region.neurons.count as usize) {
        if region.neurons.activations[i] > min_activation {
            active_count += 1;
        }
    }

    // Scale: 10% of pain neurons active = full pain
    (active_count as f32 / (pain_range as f32 * 0.10)).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::region::Region;

    fn make_all_regions() -> Vec<Region> {
        RegionId::ALL.iter().map(|&id| Region::new(id)).collect()
    }

    fn inject(regions: &mut [Region], region_id: RegionId, global_ids: &[u32], value: f32) {
        let idx = regions.iter().position(|r| r.id == region_id).unwrap();
        regions[idx].pre_tick();
        for &gid in global_ids {
            regions[idx].add_incoming_global(gid, value);
        }
        regions[idx].update_neurons();
    }

    #[test]
    fn test_population_code_center() {
        let codes = population_code(0.5, 0, 2500, 30);
        assert!(!codes.is_empty());
        // Center should be around neuron 1250
        let max_pair = codes.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        assert!((max_pair.0 as i32 - 1250).unsigned_abs() < 2, "center={}", max_pair.0);
        assert!((max_pair.1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_population_code_edges() {
        let low = population_code(0.0, 0, 2500, 30);
        let high = population_code(1.0, 0, 2500, 30);
        let low_center = low.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        let high_center = high.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        assert!(low_center.0 < 50, "low center={}", low_center.0);
        assert!(high_center.0 > 2450, "high center={}", high_center.0);
    }

    #[test]
    fn test_population_code_spread() {
        let narrow = population_code(0.5, 0, 2500, 10);
        let wide = population_code(0.5, 0, 2500, 50);
        assert!(wide.len() > narrow.len(), "wide={} narrow={}", wide.len(), narrow.len());
    }

    #[test]
    fn test_encode_sensory_all_modalities() {
        let signals = encode_sensory(0.5, 0.5, 0.0, 0.5, 30);
        assert!(!signals.is_empty());
        // Should have neurons from temperature, pressure, and texture ranges
        let has_temp = signals.iter().any(|(gid, _)| *gid >= TEMP_START && *gid < TEMP_END);
        let has_press = signals.iter().any(|(gid, _)| *gid >= PRESSURE_START && *gid < PRESSURE_END);
        let has_texture = signals.iter().any(|(gid, _)| *gid >= TEXTURE_START && *gid < TEXTURE_END);
        assert!(has_temp, "missing temperature");
        assert!(has_press, "missing pressure");
        assert!(has_texture, "missing texture");
    }

    #[test]
    fn test_sensory_activation_silent() {
        let regions = make_all_regions();
        let s = sensory_activation_strength(&regions, 0.01);
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn test_sensory_activation_active() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (0..200).collect();
        inject(&mut regions, RegionId::Sensory, &neurons, 1.0);
        let s = sensory_activation_strength(&regions, 0.01);
        assert!(s > 0.2, "strength={s}");
    }

    #[test]
    fn test_boost_sensory() {
        let mut regions = make_all_regions();
        let count = boost_sensory_neurons(&mut regions, &[0, 1, 2, 100000], 0.5);
        assert_eq!(count, 3); // 100000 is out of range
    }

    #[test]
    fn test_peak_sensory_neurons() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (0..50).collect();
        inject(&mut regions, RegionId::Sensory, &neurons, 1.0);
        let peaks = peak_sensory_neurons(&regions, 0.01, 10);
        assert!(!peaks.is_empty());
        assert!(peaks.len() <= 10);
    }

    #[test]
    fn test_detect_pain_none() {
        let regions = make_all_regions();
        let pain = detect_pain_level(&regions, 0.01);
        assert!(pain.abs() < 1e-6);
    }

    #[test]
    fn test_detect_pain_active() {
        let mut regions = make_all_regions();
        // Activate pain neurons (5000–7499)
        let neurons: Vec<u32> = (5000..5250).collect();
        inject(&mut regions, RegionId::Sensory, &neurons, 1.0);
        let pain = detect_pain_level(&regions, 0.01);
        assert!(pain > 0.5, "pain={pain}");
    }
}
