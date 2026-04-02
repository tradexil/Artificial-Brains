// regions/audio.rs — Audio region logic.
//
// Audio region (30000–44999, 15k neurons, 20% inhibitory):
//   Processes auditory input: frequencies, temporal patterns, complex sounds.
//
// Sub-regions:
//   30000–34999: frequency decomposition (pitch)
//   35000–39999: temporal patterns (rhythm, onset, duration)
//   40000–44999: complex (timbre, melody, speech phonemes)

use crate::core::region::{Region, RegionId};

const AUDIO_START: u32 = 30_000;
const AUDIO_END: u32 = 45_000;
const AUDIO_COUNT: u32 = 15_000;

// Sub-region boundaries
const FREQ_START: u32 = 30_000;
const FREQ_END: u32 = 35_000;
const TEMPORAL_START: u32 = 35_000;
const TEMPORAL_END: u32 = 40_000;
const COMPLEX_START: u32 = 40_000;
const COMPLEX_END: u32 = 45_000;

/// Audio sub-region identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSubRegion {
    Frequency,
    Temporal,
    Complex,
}

impl AudioSubRegion {
    pub fn range(self) -> (u32, u32) {
        match self {
            AudioSubRegion::Frequency => (FREQ_START, FREQ_END),
            AudioSubRegion::Temporal => (TEMPORAL_START, TEMPORAL_END),
            AudioSubRegion::Complex => (COMPLEX_START, COMPLEX_END),
        }
    }
}

/// Compute overall audio activation strength (0.0–1.0).
pub fn audio_activation_strength(regions: &[Region], min_activation: f32) -> f32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Audio) {
        Some(i) => i,
        None => return 0.0,
    };

    let active = regions[idx].active_global_ids(min_activation).len() as f32;
    (active / (AUDIO_COUNT as f32 * 0.05)).min(1.0)
}

/// Compute activation within a specific audio sub-region.
pub fn sub_region_activation(
    regions: &[Region],
    sub: AudioSubRegion,
    min_activation: f32,
) -> f32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Audio) {
        Some(i) => i,
        None => return 0.0,
    };

    let region = &regions[idx];
    let (sub_start, sub_end) = sub.range();
    let sub_count = sub_end - sub_start;

    let local_start = (sub_start - AUDIO_START) as usize;
    let local_end = (sub_end - AUDIO_START) as usize;

    let mut active = 0u32;
    for i in local_start..local_end.min(region.neurons.count as usize) {
        if region.neurons.activations[i] > min_activation {
            active += 1;
        }
    }

    (active as f32 / (sub_count as f32 * 0.05)).min(1.0)
}

/// Boost specific audio neurons. Returns count boosted.
pub fn boost_audio_neurons(
    regions: &mut [Region],
    neurons: &[u32],
    boost: f32,
) -> u32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Audio) {
        Some(i) => i,
        None => return 0,
    };

    let mut count = 0u32;
    for &gid in neurons {
        if gid >= AUDIO_START && gid < AUDIO_END {
            if let Some(local) = regions[idx].global_to_local(gid) {
                regions[idx].neurons.activations[local as usize] += boost;
                count += 1;
            }
        }
    }
    count
}

/// Get top-K active audio neurons.
pub fn peak_audio_neurons(
    regions: &[Region],
    min_activation: f32,
    top_k: usize,
) -> Vec<(u32, f32)> {
    let idx = match regions.iter().position(|r| r.id == RegionId::Audio) {
        Some(i) => i,
        None => return Vec::new(),
    };

    let region = &regions[idx];
    let excit_end = (AUDIO_COUNT as f32 * 0.80) as usize; // 80% excitatory
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

/// Map a frequency (Hz) to the frequency sub-region neuron range.
/// Returns (global_id, activation). frequency in [20, 20000] Hz.
pub fn frequency_to_neurons(
    freq_hz: f32,
    spread: u32,
) -> Vec<(u32, f32)> {
    // Log scale: 20 Hz → neuron 30000, 20000 Hz → neuron 34999
    let freq_clamped = freq_hz.clamp(20.0, 20000.0);
    let log_min = 20.0f32.ln();
    let log_max = 20000.0f32.ln();
    let normalized = (freq_clamped.ln() - log_min) / (log_max - log_min);

    let range = FREQ_END - FREQ_START;
    let center = FREQ_START as f32 + normalized * (range - 1) as f32;
    let sigma = spread as f32;
    let sigma_sq = sigma * sigma;

    let start = if center as u32 > FREQ_START + spread * 3 {
        center as u32 - spread * 3
    } else {
        FREQ_START
    };
    let end = (center as u32 + spread * 3 + 1).min(FREQ_END);

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
    fn test_audio_activation_silent() {
        let regions = make_all_regions();
        let s = audio_activation_strength(&regions, 0.01);
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn test_audio_activation_active() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (30000..30300).collect();
        inject(&mut regions, RegionId::Audio, &neurons, 1.0);
        let s = audio_activation_strength(&regions, 0.01);
        assert!(s > 0.2, "strength={s}");
    }

    #[test]
    fn test_sub_region_activation() {
        let mut regions = make_all_regions();
        // Activate frequency sub-region only
        let neurons: Vec<u32> = (30000..30200).collect();
        inject(&mut regions, RegionId::Audio, &neurons, 1.0);

        let freq = sub_region_activation(&regions, AudioSubRegion::Frequency, 0.01);
        let temporal = sub_region_activation(&regions, AudioSubRegion::Temporal, 0.01);
        assert!(freq > 0.0, "frequency should be active");
        assert!(temporal.abs() < 1e-6, "temporal should be silent");
    }

    #[test]
    fn test_boost_audio() {
        let mut regions = make_all_regions();
        let count = boost_audio_neurons(&mut regions, &[30000, 30001, 200000], 0.5);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_peak_audio_neurons() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (30000..30050).collect();
        inject(&mut regions, RegionId::Audio, &neurons, 1.0);
        let peaks = peak_audio_neurons(&regions, 0.01, 10);
        assert!(!peaks.is_empty());
        assert!(peaks.len() <= 10);
    }

    #[test]
    fn test_frequency_to_neurons_low() {
        let neurons = frequency_to_neurons(20.0, 20);
        assert!(!neurons.is_empty());
        let center = neurons.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        assert!(center.0 >= FREQ_START && center.0 < FREQ_START + 50,
            "low freq center={}", center.0);
    }

    #[test]
    fn test_frequency_to_neurons_high() {
        let neurons = frequency_to_neurons(20000.0, 20);
        assert!(!neurons.is_empty());
        let center = neurons.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        assert!(center.0 > FREQ_END - 50 && center.0 < FREQ_END,
            "high freq center={}", center.0);
    }

    #[test]
    fn test_frequency_mid() {
        // 440 Hz (A4) should be roughly in the middle-ish (log scale)
        let neurons = frequency_to_neurons(440.0, 20);
        assert!(!neurons.is_empty());
        let center = neurons.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        // 440 Hz is about 45% through log scale, so ~32250
        assert!(center.0 > 31500 && center.0 < 33500,
            "440 Hz center={}", center.0);
    }
}
