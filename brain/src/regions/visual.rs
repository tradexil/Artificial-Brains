// regions/visual.rs — Visual region logic.
//
// Visual region (10000–29999, 20k neurons, 20% inhibitory):
//   Largest input region. Processes visual features in a hierarchy.
//
// Sub-regions:
//   10000–14999: low-level (edges, colors, orientation)
//   15000–19999: mid-level (shapes, textures, contours)
//   20000–24999: high-level (objects, faces, scenes)
//   25000–29999: spatial (position, movement, depth)

use crate::core::region::{Region, RegionId};

const VISUAL_START: u32 = 10_000;
const VISUAL_END: u32 = 30_000;
const VISUAL_COUNT: u32 = 20_000;

// Sub-region boundaries
const LOW_START: u32 = 10_000;
const LOW_END: u32 = 15_000;
const MID_START: u32 = 15_000;
const MID_END: u32 = 20_000;
const HIGH_START: u32 = 20_000;
const HIGH_END: u32 = 25_000;
const SPATIAL_START: u32 = 25_000;
const SPATIAL_END: u32 = 30_000;

/// Visual sub-region identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualSubRegion {
    Low,
    Mid,
    High,
    Spatial,
}

impl VisualSubRegion {
    pub fn range(self) -> (u32, u32) {
        match self {
            VisualSubRegion::Low => (LOW_START, LOW_END),
            VisualSubRegion::Mid => (MID_START, MID_END),
            VisualSubRegion::High => (HIGH_START, HIGH_END),
            VisualSubRegion::Spatial => (SPATIAL_START, SPATIAL_END),
        }
    }
}

/// Compute overall visual activation strength (0.0–1.0).
pub fn visual_activation_strength(regions: &[Region], min_activation: f32) -> f32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Visual) {
        Some(i) => i,
        None => return 0.0,
    };

    let active = regions[idx].active_global_ids(min_activation).len() as f32;
    (active / (VISUAL_COUNT as f32 * 0.05)).min(1.0)
}

/// Compute activation strength within a specific visual sub-region.
pub fn sub_region_activation(
    regions: &[Region],
    sub: VisualSubRegion,
    min_activation: f32,
) -> f32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Visual) {
        Some(i) => i,
        None => return 0.0,
    };

    let region = &regions[idx];
    let (sub_start, sub_end) = sub.range();
    let sub_count = sub_end - sub_start;

    let local_start = (sub_start - VISUAL_START) as usize;
    let local_end = (sub_end - VISUAL_START) as usize;

    let mut active = 0u32;
    for i in local_start..local_end.min(region.neurons.count as usize) {
        if region.neurons.activations[i] > min_activation {
            active += 1;
        }
    }

    (active as f32 / (sub_count as f32 * 0.05)).min(1.0)
}

/// Boost specific visual neurons. Returns count boosted.
pub fn boost_visual_neurons(
    regions: &mut [Region],
    neurons: &[u32],
    boost: f32,
) -> u32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Visual) {
        Some(i) => i,
        None => return 0,
    };

    let mut count = 0u32;
    for &gid in neurons {
        if gid >= VISUAL_START && gid < VISUAL_END {
            if let Some(local) = regions[idx].global_to_local(gid) {
                regions[idx].neurons.activations[local as usize] += boost;
                count += 1;
            }
        }
    }
    count
}

/// Get top-K active visual neurons (optionally within a sub-region).
pub fn peak_visual_neurons(
    regions: &[Region],
    min_activation: f32,
    top_k: usize,
    sub_region: Option<VisualSubRegion>,
) -> Vec<(u32, f32)> {
    let idx = match regions.iter().position(|r| r.id == RegionId::Visual) {
        Some(i) => i,
        None => return Vec::new(),
    };

    let region = &regions[idx];
    let excit_end = (VISUAL_COUNT as f32 * 0.80) as u32; // 80% excitatory

    let (filter_start, filter_end) = match sub_region {
        Some(sub) => sub.range(),
        None => (VISUAL_START, VISUAL_START + excit_end),
    };

    let mut active: Vec<(u32, f32)> = Vec::new();
    let local_start = (filter_start - VISUAL_START) as usize;
    let local_end = (filter_end - VISUAL_START) as usize;

    for i in local_start..local_end.min(region.neurons.count as usize) {
        let act = region.neurons.activations[i];
        if act > min_activation {
            active.push((region.local_to_global(i as u32), act));
        }
    }

    active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    active.truncate(top_k);
    active
}

/// Read all visual activations above threshold (for imagination output).
/// Returns (global_id, activation) for all active excitatory visual neurons.
pub fn read_visual_activations(
    regions: &[Region],
    min_activation: f32,
) -> Vec<(u32, f32)> {
    let idx = match regions.iter().position(|r| r.id == RegionId::Visual) {
        Some(i) => i,
        None => return Vec::new(),
    };

    let region = &regions[idx];
    let excit_end = (VISUAL_COUNT as f32 * 0.80) as usize;
    let mut result = Vec::new();

    for i in 0..excit_end.min(region.neurons.count as usize) {
        let act = region.neurons.activations[i];
        if act > min_activation {
            result.push((region.local_to_global(i as u32), act));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_visual_activation_silent() {
        let regions = make_all_regions();
        let s = visual_activation_strength(&regions, 0.01);
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn test_visual_activation_active() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (10000..10400).collect();
        inject(&mut regions, RegionId::Visual, &neurons, 1.0);
        let s = visual_activation_strength(&regions, 0.01);
        assert!(s > 0.2, "strength={s}");
    }

    #[test]
    fn test_sub_region_activation() {
        let mut regions = make_all_regions();
        // Activate low-level only
        let neurons: Vec<u32> = (10000..10200).collect();
        inject(&mut regions, RegionId::Visual, &neurons, 1.0);

        let low = sub_region_activation(&regions, VisualSubRegion::Low, 0.01);
        let high = sub_region_activation(&regions, VisualSubRegion::High, 0.01);
        assert!(low > 0.0, "low sub-region should be active");
        assert!(high.abs() < 1e-6, "high sub-region should be silent");
    }

    #[test]
    fn test_boost_visual() {
        let mut regions = make_all_regions();
        let count = boost_visual_neurons(&mut regions, &[10000, 10001, 50000], 0.5);
        assert_eq!(count, 2); // 50000 out of range
    }

    #[test]
    fn test_peak_visual_neurons() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (10000..10050).collect();
        inject(&mut regions, RegionId::Visual, &neurons, 1.0);
        let peaks = peak_visual_neurons(&regions, 0.01, 10, None);
        assert!(!peaks.is_empty());
        assert!(peaks.len() <= 10);
    }

    #[test]
    fn test_peak_visual_sub_region() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (20000..20050).collect(); // high-level
        inject(&mut regions, RegionId::Visual, &neurons, 1.0);

        let high_peaks = peak_visual_neurons(&regions, 0.01, 10, Some(VisualSubRegion::High));
        let low_peaks = peak_visual_neurons(&regions, 0.01, 10, Some(VisualSubRegion::Low));
        assert!(!high_peaks.is_empty());
        assert!(low_peaks.is_empty());
    }

    #[test]
    fn test_read_visual_activations() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (10000..10020).collect();
        inject(&mut regions, RegionId::Visual, &neurons, 1.0);
        let acts = read_visual_activations(&regions, 0.01);
        assert!(!acts.is_empty());
        assert!(acts.len() >= 10);
    }

    #[test]
    fn test_read_visual_empty() {
        let regions = make_all_regions();
        let acts = read_visual_activations(&regions, 0.01);
        assert!(acts.is_empty());
    }
}
