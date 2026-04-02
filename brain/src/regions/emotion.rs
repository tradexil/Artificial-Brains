// regions/emotion.rs — Emotion region logic.
//
// Emotion region (70000–79999, 10k neurons, 15% inhibitory):
//   - First half excitatory (70000–74249): positive-valence population
//   - Second half excitatory (74250–78499): negative-valence population  [accounting for 8500 excit total]
//   - Inhibitory (78500–79999): regulation
//
// Provides:
//   - polarity: balance of positive vs negative firing
//   - arousal: overall firing intensity
//   - urgency: fraction of high-activation neurons (strong emotional signal)
//   - motor impulse: direct emotional response wanting to bypass executive

use crate::core::region::{Region, RegionId};

/// Emotion region global ID range.
const EMOTION_START: u32 = 70_000;
const EMOTION_END: u32 = 80_000; // exclusive
const EMOTION_COUNT: u32 = 10_000;

/// Excitatory neuron split (85% of 10k = 8500 excitatory).
const EXCITATORY_COUNT: u32 = 8_500;
const POSITIVE_END: u32 = EMOTION_START + EXCITATORY_COUNT / 2; // 74250
const NEGATIVE_END: u32 = EMOTION_START + EXCITATORY_COUNT;     // 78500

/// Compute emotional polarity from region firing.
/// Returns -1.0 (fully negative) to +1.0 (fully positive).
pub fn compute_polarity(regions: &[Region], min_activation: f32) -> f32 {
    let emotion_idx = regions
        .iter()
        .position(|r| r.id == RegionId::Emotion);
    let region = match emotion_idx {
        Some(i) => &regions[i],
        None => return 0.0,
    };

    let active = region.active_global_ids(min_activation);
    if active.is_empty() {
        return 0.0;
    }

    let mut pos_count = 0u32;
    let mut neg_count = 0u32;

    for &gid in &active {
        if gid >= EMOTION_START && gid < POSITIVE_END {
            pos_count += 1;
        } else if gid >= POSITIVE_END && gid < NEGATIVE_END {
            neg_count += 1;
        }
        // inhibitory neurons (>= NEGATIVE_END) don't count toward polarity
    }

    let total = pos_count + neg_count;
    if total == 0 {
        return 0.0;
    }
    (pos_count as f32 - neg_count as f32) / total as f32
}

/// Compute emotional arousal (overall firing intensity).
/// Returns 0.0–1.0 scaled by max plausible firing rate.
pub fn compute_arousal(regions: &[Region], min_activation: f32) -> f32 {
    let emotion_idx = regions
        .iter()
        .position(|r| r.id == RegionId::Emotion);
    let region = match emotion_idx {
        Some(i) => &regions[i],
        None => return 0.0,
    };

    let active_count = region.active_global_ids(min_activation).len() as f32;
    // Scale: 10% firing = high arousal
    let rate = active_count / (EMOTION_COUNT as f32 * 0.10);
    rate.min(1.0)
}

/// Compute urgency: fraction of emotion neurons with high activation.
/// High urgency means strong emotional signal demanding immediate response.
pub fn compute_urgency(
    regions: &[Region],
    min_activation: f32,
    urgency_threshold: f32,
) -> f32 {
    let emotion_idx = regions
        .iter()
        .position(|r| r.id == RegionId::Emotion);
    let region = match emotion_idx {
        Some(i) => &regions[i],
        None => return 0.0,
    };

    let all_active = region.active_global_ids(min_activation);
    if all_active.is_empty() {
        return 0.0;
    }

    let high_active = region.active_global_ids(urgency_threshold);
    high_active.len() as f32 / all_active.len() as f32
}

/// Get emotion→motor impulse neurons.
/// Returns list of (motor_neuron_global_id, impulse_strength) pairs.
/// These are direct emotional responses that bypasse executive control.
/// Maps active positive emotion → approach motor neurons,
///       active negative emotion → withdraw motor neurons.
pub fn emotion_motor_impulse(
    regions: &[Region],
    min_activation: f32,
) -> Vec<(u32, f32)> {
    let emotion_idx = regions
        .iter()
        .position(|r| r.id == RegionId::Emotion);
    let region = match emotion_idx {
        Some(i) => &regions[i],
        None => return Vec::new(),
    };

    let active = region.active_global_ids(min_activation);
    if active.is_empty() {
        return Vec::new();
    }

    let mut impulses = Vec::new();
    let motor_start: u32 = 130_000;
    let motor_mid: u32 = 135_000; // approach vs withdraw split

    let mut pos_strength = 0.0f32;
    let mut neg_strength = 0.0f32;
    let mut pos_count = 0u32;
    let mut neg_count = 0u32;

    for &gid in &active {
        if gid >= EMOTION_START && gid < POSITIVE_END {
            pos_count += 1;
            pos_strength += region.neurons.activations
                [region.global_to_local(gid).unwrap() as usize];
        } else if gid >= POSITIVE_END && gid < NEGATIVE_END {
            neg_count += 1;
            neg_strength += region.neurons.activations
                [region.global_to_local(gid).unwrap() as usize];
        }
    }

    // Positive emotion → approach motor (first half of motor, small sample)
    if pos_count > 0 {
        let avg = pos_strength / pos_count as f32;
        let impulse = avg * 0.3; // scaled down — executive should gate this
        for i in 0..5u32 {
            impulses.push((motor_start + i, impulse));
        }
    }

    // Negative emotion → withdraw motor (second half of motor, small sample)
    if neg_count > 0 {
        let avg = neg_strength / neg_count as f32;
        let impulse = avg * 0.3;
        for i in 0..5u32 {
            impulses.push((motor_mid + i, impulse));
        }
    }

    impulses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::region::Region;

    fn make_emotion_region() -> Vec<Region> {
        // Build all regions up to and including emotion
        let all_ids = [
            RegionId::Sensory,
            RegionId::Visual,
            RegionId::Audio,
            RegionId::MemoryShort,
            RegionId::MemoryLong,
            RegionId::Emotion,
            RegionId::Attention,
            RegionId::Pattern,
            RegionId::Integration,
            RegionId::Language,
            RegionId::Executive,
            RegionId::Motor,
            RegionId::Speech,
            RegionId::Numbers,
        ];
        all_ids.iter().map(|&id| Region::new(id)).collect()
    }

    fn inject_emotion(regions: &mut [Region], global_ids: &[u32], value: f32) {
        let eidx = regions
            .iter()
            .position(|r| r.id == RegionId::Emotion)
            .unwrap();
        regions[eidx].pre_tick();
        for &gid in global_ids {
            regions[eidx].add_incoming_global(gid, value);
        }
        regions[eidx].update_neurons();
    }

    #[test]
    fn test_polarity_no_activity() {
        let regions = make_emotion_region();
        let p = compute_polarity(&regions, 0.01);
        assert!((p).abs() < 1e-6);
    }

    #[test]
    fn test_polarity_positive() {
        let mut regions = make_emotion_region();
        // Activate positive neurons (70000–74249)
        let pos_neurons: Vec<u32> = (70000..70050).collect();
        inject_emotion(&mut regions, &pos_neurons, 1.0);
        let p = compute_polarity(&regions, 0.01);
        assert!(p > 0.5, "polarity={p} should be strongly positive");
    }

    #[test]
    fn test_polarity_negative() {
        let mut regions = make_emotion_region();
        // Activate negative neurons (74250–78499)
        let neg_neurons: Vec<u32> = (74250..74300).collect();
        inject_emotion(&mut regions, &neg_neurons, 1.0);
        let p = compute_polarity(&regions, 0.01);
        assert!(p < -0.5, "polarity={p} should be strongly negative");
    }

    #[test]
    fn test_arousal_scales() {
        let mut regions = make_emotion_region();
        // Zero activity → zero arousal
        let a0 = compute_arousal(&regions, 0.01);
        assert!((a0).abs() < 1e-6);

        // Activate many neurons → high arousal
        let neurons: Vec<u32> = (70000..70500).collect();
        inject_emotion(&mut regions, &neurons, 1.0);
        let a1 = compute_arousal(&regions, 0.01);
        assert!(a1 > 0.3, "arousal={a1} should be elevated");
    }

    #[test]
    fn test_urgency_with_threshold() {
        let mut regions = make_emotion_region();
        let eidx = regions.iter().position(|r| r.id == RegionId::Emotion).unwrap();
        regions[eidx].pre_tick();
        // Mix of weak and strong activations
        let weak: Vec<u32> = (70000..70020).collect();
        let strong: Vec<u32> = (70020..70030).collect();
        for &gid in &weak {
            regions[eidx].add_incoming_global(gid, 0.4);
        }
        for &gid in &strong {
            regions[eidx].add_incoming_global(gid, 1.5);
        }
        regions[eidx].update_neurons();

        let u = compute_urgency(&regions, 0.01, 0.5);
        // Only the strong-activated neurons should be above urgency threshold
        assert!(u > 0.0 && u <= 1.0, "urgency={u}");
    }

    #[test]
    fn test_motor_impulse_positive() {
        let mut regions = make_emotion_region();
        let pos_neurons: Vec<u32> = (70000..70020).collect();
        inject_emotion(&mut regions, &pos_neurons, 1.0);
        let impulses = emotion_motor_impulse(&regions, 0.01);
        // Should produce approach motor impulses
        assert!(!impulses.is_empty());
        // Motor approach neurons start at 130000
        assert!(impulses.iter().all(|&(gid, _)| gid >= 130000 && gid < 135000));
    }

    #[test]
    fn test_motor_impulse_negative() {
        let mut regions = make_emotion_region();
        let neg_neurons: Vec<u32> = (74250..74270).collect();
        inject_emotion(&mut regions, &neg_neurons, 1.0);
        let impulses = emotion_motor_impulse(&regions, 0.01);
        assert!(!impulses.is_empty());
        // Motor withdraw neurons start at 135000
        assert!(impulses.iter().all(|&(gid, _)| gid >= 135000 && gid < 140000));
    }

    #[test]
    fn test_no_impulse_when_silent() {
        let regions = make_emotion_region();
        let impulses = emotion_motor_impulse(&regions, 0.01);
        assert!(impulses.is_empty());
    }
}
