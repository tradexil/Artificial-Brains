// regions/speech.rs — Speech region logic.
//
// Speech region (140000–149999, 10k neurons, 20% inhibitory):
//   Output region that converts language patterns into phoneme sequences.
//   Receives from language and executive regions.
//
// Sub-populations:
//   Phoneme neurons (140000–147999): 8k excitatory, each subset = a phoneme/token
//   Inhibitory neurons (148000–149999): 2k, lateral inhibition for winner-take-all output

use crate::core::region::{Region, RegionId};

const SPEECH_START: u32 = 140_000;
const SPEECH_END: u32 = 150_000;
const SPEECH_COUNT: u32 = 10_000;
const SPEECH_EXCITATORY_END: u32 = 148_000; // first 8k are excitatory

/// Compute overall speech region activity level (0.0–1.0).
pub fn speech_activity_level(regions: &[Region], min_activation: f32) -> f32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Speech) {
        Some(i) => i,
        None => return 0.0,
    };

    let active = regions[idx].active_count(min_activation) as f32;
    // Scale: 5% firing = full activity
    (active / (SPEECH_COUNT as f32 * 0.05)).min(1.0)
}

/// Get the top-K most active excitatory speech neurons.
/// Returns (global_id, activation) sorted descending by activation.
pub fn peak_speech_neurons(
    regions: &[Region],
    min_activation: f32,
    top_k: usize,
) -> Vec<(u32, f32)> {
    let idx = match regions.iter().position(|r| r.id == RegionId::Speech) {
        Some(i) => i,
        None => return Vec::new(),
    };

    let region = &regions[idx];
    let mut active: Vec<(u32, f32)> = Vec::new();

    for i in 0..region.neurons.count as usize {
        let act = region.neurons.activations[i];
        if act > min_activation {
            let gid = region.local_to_global(i as u32);
            // Only excitatory neurons carry phoneme signal
            if gid >= SPEECH_START && gid < SPEECH_EXCITATORY_END {
                active.push((gid, act));
            }
        }
    }

    active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    active.truncate(top_k);
    active
}

/// Apply lateral inhibition in speech region.
/// Suppresses weaker neurons to sharpen output (winner-take-all).
/// Returns number of neurons suppressed.
pub fn speech_lateral_inhibition(
    regions: &mut [Region],
    min_activation: f32,
    suppression_factor: f32,
) -> u32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Speech) {
        Some(i) => i,
        None => return 0,
    };

    // Find the peak activation among excitatory speech neurons
    let mut peak = 0.0f32;
    let region = &regions[idx];
    let excit_local_end = (SPEECH_EXCITATORY_END - SPEECH_START) as usize;

    for i in 0..excit_local_end.min(region.neurons.count as usize) {
        if region.neurons.activations[i] > peak {
            peak = region.neurons.activations[i];
        }
    }

    if peak < min_activation {
        return 0;
    }

    // Suppress neurons that are below half of peak
    let threshold = peak * 0.5;
    let mut suppressed = 0u32;

    for i in 0..excit_local_end.min(regions[idx].neurons.count as usize) {
        let act = regions[idx].neurons.activations[i];
        if act > min_activation && act < threshold {
            regions[idx].neurons.activations[i] *= 1.0 - suppression_factor;
            suppressed += 1;
        }
    }
    suppressed
}

/// Boost specific speech neurons (for driving output from language patterns).
/// Returns count of neurons boosted.
pub fn boost_speech_neurons(
    regions: &mut [Region],
    neurons: &[u32],
    boost: f32,
) -> u32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Speech) {
        Some(i) => i,
        None => return 0,
    };

    let mut count = 0u32;
    for &gid in neurons {
        if gid >= SPEECH_START && gid < SPEECH_END {
            if let Some(local) = regions[idx].global_to_local(gid) {
                regions[idx].neurons.activations[local as usize] += boost;
                count += 1;
            }
        }
    }
    count
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
    fn test_speech_activity_silent() {
        let regions = make_all_regions();
        let a = speech_activity_level(&regions, 0.01);
        assert!(a.abs() < 1e-6);
    }

    #[test]
    fn test_speech_activity_active() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (140000..140200).collect();
        inject(&mut regions, RegionId::Speech, &neurons, 1.0);
        let a = speech_activity_level(&regions, 0.01);
        assert!(a > 0.2, "activity={a}");
    }

    #[test]
    fn test_peak_speech_neurons_empty() {
        let regions = make_all_regions();
        let peaks = peak_speech_neurons(&regions, 0.01, 5);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_peak_speech_neurons_returns_top_k() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (140000..140050).collect();
        inject(&mut regions, RegionId::Speech, &neurons, 1.0);
        let peaks = peak_speech_neurons(&regions, 0.01, 10);
        assert!(!peaks.is_empty());
        assert!(peaks.len() <= 10);
        // Sorted descending
        for i in 1..peaks.len() {
            assert!(peaks[i].1 <= peaks[i - 1].1);
        }
    }

    #[test]
    fn test_peak_excludes_inhibitory() {
        let mut regions = make_all_regions();
        // Inject into inhibitory range (148000+)
        let neurons: Vec<u32> = (148000..148050).collect();
        inject(&mut regions, RegionId::Speech, &neurons, 1.0);
        let peaks = peak_speech_neurons(&regions, 0.01, 10);
        // Should be empty — inhibitory neurons excluded from peak
        assert!(peaks.is_empty(), "inhibitory neurons should not appear in peaks");
    }

    #[test]
    fn test_lateral_inhibition() {
        let mut regions = make_all_regions();
        let idx = regions.iter().position(|r| r.id == RegionId::Speech).unwrap();
        // Manually set activations: one strong, several weak
        regions[idx].neurons.activations[0] = 1.0; // strong: 140000
        regions[idx].neurons.activations[1] = 0.3; // weak: 140001
        regions[idx].neurons.activations[2] = 0.2; // weak: 140002

        let suppressed = speech_lateral_inhibition(&mut regions, 0.01, 0.8);
        assert!(suppressed >= 2, "suppressed={suppressed}");
        // Strong neuron should be untouched
        assert!((regions[idx].neurons.activations[0] - 1.0).abs() < 1e-6);
        // Weak neurons should be reduced
        assert!(regions[idx].neurons.activations[1] < 0.3);
    }

    #[test]
    fn test_boost_speech_neurons() {
        let mut regions = make_all_regions();
        let neurons = vec![140000u32, 140001, 140002];
        let count = boost_speech_neurons(&mut regions, &neurons, 0.5);
        assert_eq!(count, 3);
        let idx = regions.iter().position(|r| r.id == RegionId::Speech).unwrap();
        assert!((regions[idx].neurons.activations[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_boost_speech_out_of_range() {
        let mut regions = make_all_regions();
        let count = boost_speech_neurons(&mut regions, &[0, 200000], 0.5);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_lateral_inhibition_no_activity() {
        let mut regions = make_all_regions();
        let suppressed = speech_lateral_inhibition(&mut regions, 0.01, 0.8);
        assert_eq!(suppressed, 0);
    }
}
