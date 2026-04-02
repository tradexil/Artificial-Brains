// regions/motor.rs — Motor region logic.
//
// Motor region (130000–139999, 10k neurons, 20% inhibitory):
//   Receives commands from executive (deliberate, 20-50 tick delay)
//   and sensory/emotion (reflexes, 2-3 tick delay).
//
// Sub-populations:
//   130000–134999: approach motor neurons (move toward, grab, engage)
//   135000–139999: withdraw motor neurons (retreat, release, avoid)
//   Note: last 20% = inhibitory (shared between sub-populations)
//   Excitatory: 130000–137999 (8000 neurons)
//     Approach excitatory: 130000–134999 (5000)
//     Withdraw excitatory: 135000–137999 (3000)
//   Inhibitory: 138000–139999 (2000)

use crate::core::region::{Region, RegionId};

const MOTOR_START: u32 = 130_000;
const MOTOR_END: u32 = 140_000;
const MOTOR_COUNT: u32 = 10_000;

const APPROACH_START: u32 = 130_000;
const APPROACH_END: u32 = 135_000;
const WITHDRAW_START: u32 = 135_000;
const WITHDRAW_END: u32 = 138_000; // excitatory withdraw only
const EXCITATORY_END: u32 = 138_000;

/// Motor action types decoded from motor neuron activity.
#[derive(Debug, Clone, PartialEq)]
pub enum MotorAction {
    Approach { strength: f32 },
    Withdraw { strength: f32 },
    Idle,
    Conflict { approach: f32, withdraw: f32 },
}

/// Compute overall motor activation strength (0.0–1.0).
pub fn motor_activation_strength(regions: &[Region], min_activation: f32) -> f32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Motor) {
        Some(i) => i,
        None => return 0.0,
    };

    let active = regions[idx].active_global_ids(min_activation).len() as f32;
    (active / (MOTOR_COUNT as f32 * 0.05)).min(1.0)
}

/// Compute approach vs withdraw strengths from motor neuron firing.
/// Returns (approach_strength, withdraw_strength) each 0.0–1.0.
pub fn approach_vs_withdraw(regions: &[Region], min_activation: f32) -> (f32, f32) {
    let idx = match regions.iter().position(|r| r.id == RegionId::Motor) {
        Some(i) => i,
        None => return (0.0, 0.0),
    };

    let region = &regions[idx];
    let approach_local_end = (APPROACH_END - MOTOR_START) as usize;
    let withdraw_local_start = (WITHDRAW_START - MOTOR_START) as usize;
    let withdraw_local_end = (WITHDRAW_END - MOTOR_START) as usize;

    let mut approach_sum = 0.0f32;
    let mut approach_count = 0u32;
    for i in 0..approach_local_end.min(region.neurons.count as usize) {
        if region.neurons.activations[i] > min_activation {
            approach_sum += region.neurons.activations[i];
            approach_count += 1;
        }
    }

    let mut withdraw_sum = 0.0f32;
    let mut withdraw_count = 0u32;
    for i in withdraw_local_start..withdraw_local_end.min(region.neurons.count as usize) {
        if region.neurons.activations[i] > min_activation {
            withdraw_sum += region.neurons.activations[i];
            withdraw_count += 1;
        }
    }

    // Normalize by population size (5% active = 1.0)
    let approach_norm = (approach_count as f32 / ((APPROACH_END - APPROACH_START) as f32 * 0.05)).min(1.0);
    let withdraw_norm = (withdraw_count as f32 / ((WITHDRAW_END - WITHDRAW_START) as f32 * 0.05)).min(1.0);

    (approach_norm, withdraw_norm)
}

/// Decode the dominant motor action from current activations.
pub fn decode_motor_action(regions: &[Region], min_activation: f32) -> MotorAction {
    let (approach, withdraw) = approach_vs_withdraw(regions, min_activation);

    if approach < 0.05 && withdraw < 0.05 {
        return MotorAction::Idle;
    }

    // Conflict if both are significant
    if approach > 0.15 && withdraw > 0.15 {
        return MotorAction::Conflict { approach, withdraw };
    }

    if approach >= withdraw {
        MotorAction::Approach { strength: approach }
    } else {
        MotorAction::Withdraw { strength: withdraw }
    }
}

/// Get top-K active motor neurons (excitatory only).
pub fn peak_motor_neurons(
    regions: &[Region],
    min_activation: f32,
    top_k: usize,
) -> Vec<(u32, f32)> {
    let idx = match regions.iter().position(|r| r.id == RegionId::Motor) {
        Some(i) => i,
        None => return Vec::new(),
    };

    let region = &regions[idx];
    let excit_local_end = (EXCITATORY_END - MOTOR_START) as usize;
    let mut active: Vec<(u32, f32)> = Vec::new();

    for i in 0..excit_local_end.min(region.neurons.count as usize) {
        let act = region.neurons.activations[i];
        if act > min_activation {
            active.push((region.local_to_global(i as u32), act));
        }
    }

    active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    active.truncate(top_k);
    active
}

/// Apply lateral inhibition in motor region (winner-take-all between approach/withdraw).
/// Returns number of neurons suppressed.
pub fn motor_lateral_inhibition(
    regions: &mut [Region],
    min_activation: f32,
    suppression_factor: f32,
) -> u32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Motor) {
        Some(i) => i,
        None => return 0,
    };

    // Use raw activation sums for comparison (not normalized)
    let approach_local_end = (APPROACH_END - MOTOR_START) as usize;
    let withdraw_local_start = (WITHDRAW_START - MOTOR_START) as usize;
    let withdraw_local_end = (WITHDRAW_END - MOTOR_START) as usize;

    let mut approach_sum = 0.0f32;
    let mut withdraw_sum = 0.0f32;

    for i in 0..approach_local_end.min(regions[idx].neurons.count as usize) {
        if regions[idx].neurons.activations[i] > min_activation {
            approach_sum += regions[idx].neurons.activations[i];
        }
    }
    for i in withdraw_local_start..withdraw_local_end.min(regions[idx].neurons.count as usize) {
        if regions[idx].neurons.activations[i] > min_activation {
            withdraw_sum += regions[idx].neurons.activations[i];
        }
    }

    let mut suppressed = 0u32;

    // Suppress the weaker population (based on raw sum)
    if approach_sum > withdraw_sum && withdraw_sum > 0.0 {
        for i in withdraw_local_start..withdraw_local_end.min(regions[idx].neurons.count as usize) {
            if regions[idx].neurons.activations[i] > min_activation {
                regions[idx].neurons.activations[i] *= 1.0 - suppression_factor;
                suppressed += 1;
            }
        }
    } else if withdraw_sum > approach_sum && approach_sum > 0.0 {
        for i in 0..approach_local_end.min(regions[idx].neurons.count as usize) {
            if regions[idx].neurons.activations[i] > min_activation {
                regions[idx].neurons.activations[i] *= 1.0 - suppression_factor;
                suppressed += 1;
            }
        }
    }

    suppressed
}

/// Boost specific motor neurons. Returns count boosted.
pub fn boost_motor_neurons(
    regions: &mut [Region],
    neurons: &[u32],
    boost: f32,
) -> u32 {
    let idx = match regions.iter().position(|r| r.id == RegionId::Motor) {
        Some(i) => i,
        None => return 0,
    };

    let mut count = 0u32;
    for &gid in neurons {
        if gid >= MOTOR_START && gid < MOTOR_END {
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
    fn test_motor_activation_silent() {
        let regions = make_all_regions();
        let s = motor_activation_strength(&regions, 0.01);
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn test_motor_activation_active() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (130000..130200).collect();
        inject(&mut regions, RegionId::Motor, &neurons, 1.0);
        let s = motor_activation_strength(&regions, 0.01);
        assert!(s > 0.2, "strength={s}");
    }

    #[test]
    fn test_approach_vs_withdraw_idle() {
        let regions = make_all_regions();
        let (app, wd) = approach_vs_withdraw(&regions, 0.01);
        assert!(app.abs() < 1e-6);
        assert!(wd.abs() < 1e-6);
    }

    #[test]
    fn test_approach_dominant() {
        let mut regions = make_all_regions();
        // Activate approach neurons
        let neurons: Vec<u32> = (130000..130300).collect();
        inject(&mut regions, RegionId::Motor, &neurons, 1.0);
        let (app, wd) = approach_vs_withdraw(&regions, 0.01);
        assert!(app > 0.5, "approach={app}");
        assert!(wd < 0.1, "withdraw={wd}");
    }

    #[test]
    fn test_withdraw_dominant() {
        let mut regions = make_all_regions();
        // Activate withdraw neurons
        let neurons: Vec<u32> = (135000..135200).collect();
        inject(&mut regions, RegionId::Motor, &neurons, 1.0);
        let (app, wd) = approach_vs_withdraw(&regions, 0.01);
        assert!(app < 0.1, "approach={app}");
        assert!(wd > 0.5, "withdraw={wd}");
    }

    #[test]
    fn test_decode_idle() {
        let regions = make_all_regions();
        let action = decode_motor_action(&regions, 0.01);
        assert_eq!(action, MotorAction::Idle);
    }

    #[test]
    fn test_decode_approach() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (130000..130300).collect();
        inject(&mut regions, RegionId::Motor, &neurons, 1.0);
        let action = decode_motor_action(&regions, 0.01);
        match action {
            MotorAction::Approach { strength } => assert!(strength > 0.5),
            other => panic!("expected approach, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_conflict() {
        let mut regions = make_all_regions();
        // Activate both approach and withdraw
        let approach: Vec<u32> = (130000..130300).collect();
        let withdraw: Vec<u32> = (135000..135200).collect();
        let mut all = approach;
        all.extend(withdraw);
        inject(&mut regions, RegionId::Motor, &all, 1.0);
        let action = decode_motor_action(&regions, 0.01);
        match action {
            MotorAction::Conflict { approach, withdraw } => {
                assert!(approach > 0.15, "approach={approach}");
                assert!(withdraw > 0.15, "withdraw={withdraw}");
            }
            other => panic!("expected conflict, got {:?}", other),
        }
    }

    #[test]
    fn test_peak_motor_neurons() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (130000..130050).collect();
        inject(&mut regions, RegionId::Motor, &neurons, 1.0);
        let peaks = peak_motor_neurons(&regions, 0.01, 10);
        assert!(!peaks.is_empty());
        assert!(peaks.len() <= 10);
    }

    #[test]
    fn test_motor_lateral_inhibition() {
        let mut regions = make_all_regions();
        // Create conflict: both approach and withdraw active
        let approach: Vec<u32> = (130000..130300).collect();
        let withdraw: Vec<u32> = (135000..135200).collect();
        let mut all = approach;
        all.extend(withdraw);
        inject(&mut regions, RegionId::Motor, &all, 1.0);

        let suppressed = motor_lateral_inhibition(&mut regions, 0.01, 0.7);
        assert!(suppressed > 0, "should suppress weaker pop");

        // After inhibition, the weaker population's activations should be reduced
        // Approach had more neurons (300 vs 200), so withdraw should be suppressed
        let idx = regions.iter().position(|r| r.id == RegionId::Motor).unwrap();
        // Check a withdraw neuron's activation was reduced (should be ~0.3 of original)
        let withdraw_local = (135000 - 130000) as usize;
        let withdraw_act = regions[idx].neurons.activations[withdraw_local];
        assert!(withdraw_act < 0.5, "withdraw should be suppressed, got {withdraw_act}");
    }

    #[test]
    fn test_boost_motor() {
        let mut regions = make_all_regions();
        let count = boost_motor_neurons(&mut regions, &[130000, 130001, 200000], 0.5);
        assert_eq!(count, 2);
    }
}
