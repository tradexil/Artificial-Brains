// regions/executive.rs — Executive control region logic.
//
// Executive region (120000–129999, 10k neurons, 30% inhibitory):
//   High inhibitory % enables conflict resolution and impulse suppression.
//
// Provides:
//   - conflict detection: multiple competing motor patterns active simultaneously
//   - impulse suppression: block emotion→motor pathways when executive is active
//   - planning signal: overall executive engagement level

use crate::core::region::{Region, RegionId};

const MOTOR_START: u32 = 130_000;
const MOTOR_END: u32 = 140_000;
const MOTOR_MID: u32 = 135_000; // approach vs withdraw boundary

/// Compute executive engagement level (0.0–1.0).
/// Based on raw firing rate of executive region.
pub fn executive_engagement(regions: &[Region], min_activation: f32) -> f32 {
    let exec_idx = regions
        .iter()
        .position(|r| r.id == RegionId::Executive);
    let region = match exec_idx {
        Some(i) => &regions[i],
        None => return 0.0,
    };

    let active = region.active_count(min_activation) as f32;
    let total = region.neurons.count as f32;
    // Scale: 5% firing = full engagement
    (active / (total * 0.05)).min(1.0)
}

/// Detect motor conflict: are both approach AND withdraw motor populations active?
/// Returns conflict score 0.0 (one clear winner) to 1.0 (50/50 tie).
pub fn detect_motor_conflict(regions: &[Region], min_activation: f32) -> f32 {
    let motor_idx = regions
        .iter()
        .position(|r| r.id == RegionId::Motor);
    let region = match motor_idx {
        Some(i) => &regions[i],
        None => return 0.0,
    };

    let active = region.active_global_ids(min_activation);
    if active.is_empty() {
        return 0.0;
    }

    let mut approach = 0u32;
    let mut withdraw = 0u32;

    for &gid in &active {
        if gid >= MOTOR_START && gid < MOTOR_MID {
            approach += 1;
        } else if gid >= MOTOR_MID && gid < MOTOR_END {
            withdraw += 1;
        }
    }

    let total = approach + withdraw;
    if total == 0 {
        return 0.0;
    }

    // Conflict = 1.0 when 50/50, 0.0 when one side dominates
    let ratio = approach.min(withdraw) as f32 / total as f32;
    ratio * 2.0 // scale so 50/50 → 1.0
}

/// Suppress motor neurons from the losing side.
/// If executive engagement is high enough and conflict exists,
/// suppress the weaker motor population.
/// Returns number of motor neurons suppressed.
pub fn resolve_motor_conflict(
    regions: &mut [Region],
    min_activation: f32,
    suppress_strength: f32,
) -> u32 {
    let exec_engagement = executive_engagement(regions, min_activation);
    if exec_engagement < 0.3 {
        return 0; // Executive not engaged enough to resolve
    }

    // Find which motor population is stronger
    let motor_idx = match regions.iter().position(|r| r.id == RegionId::Motor) {
        Some(i) => i,
        None => return 0,
    };

    let active = regions[motor_idx].active_global_ids(min_activation);
    let mut approach_sum = 0.0f32;
    let mut withdraw_sum = 0.0f32;
    let mut approach_ids = Vec::new();
    let mut withdraw_ids = Vec::new();

    for &gid in &active {
        let local = regions[motor_idx].global_to_local(gid).unwrap() as usize;
        let act = regions[motor_idx].neurons.activations[local];
        if gid >= MOTOR_START && gid < MOTOR_MID {
            approach_sum += act;
            approach_ids.push(gid);
        } else if gid >= MOTOR_MID && gid < MOTOR_END {
            withdraw_sum += act;
            withdraw_ids.push(gid);
        }
    }

    // Suppress the weaker side
    let suppress_ids = if approach_sum >= withdraw_sum {
        &withdraw_ids
    } else {
        &approach_ids
    };

    let mut suppressed = 0u32;
    for &gid in suppress_ids {
        let local = regions[motor_idx].global_to_local(gid).unwrap() as usize;
        let old = regions[motor_idx].neurons.activations[local];
        let new_val = (old - suppress_strength * exec_engagement).max(0.0);
        regions[motor_idx].neurons.activations[local] = new_val;
        if new_val < min_activation && old >= min_activation {
            suppressed += 1;
        }
    }
    suppressed
}

/// Inhibit motor neurons completely when executive blocks an impulse.
/// Returns number of neurons fully suppressed (set to 0).
pub fn inhibit_motor_neurons(
    regions: &mut [Region],
    min_activation: f32,
    impulse_neurons: &[(u32, f32)],
) -> u32 {
    let exec_engagement = executive_engagement(regions, min_activation);
    if exec_engagement < 0.5 {
        return 0; // Need strong executive to block impulse
    }

    let motor_idx = match regions.iter().position(|r| r.id == RegionId::Motor) {
        Some(i) => i,
        None => return 0,
    };

    let mut inhibited = 0u32;
    for &(gid, _strength) in impulse_neurons {
        if gid >= MOTOR_START && gid < MOTOR_END {
            if let Some(local) = regions[motor_idx].global_to_local(gid) {
                let old = regions[motor_idx].neurons.activations[local as usize];
                // Scale inhibition by executive engagement
                let suppress = old * exec_engagement;
                regions[motor_idx].neurons.activations[local as usize] =
                    (old - suppress).max(0.0);
                if old >= min_activation {
                    inhibited += 1;
                }
            }
        }
    }
    inhibited
}

/// Compute planning signal: how strongly is executive engaging
/// with language/memory regions (indicating deliberation).
pub fn planning_signal(regions: &[Region], min_activation: f32) -> f32 {
    let exec_eng = executive_engagement(regions, min_activation);

    // Check if language is also active (deliberation = exec + language)
    let lang_idx = regions
        .iter()
        .position(|r| r.id == RegionId::Language);
    let lang_rate = match lang_idx {
        Some(i) => {
            let active = regions[i].active_count(min_activation) as f32;
            (active / (regions[i].neurons.count as f32 * 0.05)).min(1.0)
        }
        None => 0.0,
    };

    // Planning = geometric mean of executive and language engagement
    (exec_eng * lang_rate).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::region::Region;

    fn make_all_regions() -> Vec<Region> {
        let ids = [
            RegionId::Sensory, RegionId::Visual, RegionId::Audio,
            RegionId::MemoryShort, RegionId::MemoryLong, RegionId::Emotion,
            RegionId::Attention, RegionId::Pattern, RegionId::Integration,
            RegionId::Language, RegionId::Executive, RegionId::Motor,
            RegionId::Speech, RegionId::Numbers,
        ];
        ids.iter().map(|&id| Region::new(id)).collect()
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
    fn test_executive_engagement_silent() {
        let regions = make_all_regions();
        let eng = executive_engagement(&regions, 0.01);
        assert!((eng).abs() < 1e-6);
    }

    #[test]
    fn test_executive_engagement_active() {
        let mut regions = make_all_regions();
        let exec_neurons: Vec<u32> = (120000..120200).collect();
        inject(&mut regions, RegionId::Executive, &exec_neurons, 1.0);
        let eng = executive_engagement(&regions, 0.01);
        assert!(eng > 0.2, "engagement={eng}");
    }

    #[test]
    fn test_no_conflict_single_population() {
        let mut regions = make_all_regions();
        // Only approach motor neurons
        let approach: Vec<u32> = (130000..130020).collect();
        inject(&mut regions, RegionId::Motor, &approach, 1.0);
        let conflict = detect_motor_conflict(&regions, 0.01);
        assert!(conflict < 0.1, "conflict={conflict} should be low");
    }

    #[test]
    fn test_conflict_both_populations() {
        let mut regions = make_all_regions();
        // Both approach and withdraw active
        let approach: Vec<u32> = (130000..130020).collect();
        let withdraw: Vec<u32> = (135000..135020).collect();
        let both: Vec<u32> = approach.iter().chain(withdraw.iter()).copied().collect();
        inject(&mut regions, RegionId::Motor, &both, 1.0);
        let conflict = detect_motor_conflict(&regions, 0.01);
        assert!(conflict > 0.5, "conflict={conflict} should be high when balanced");
    }

    #[test]
    fn test_resolve_conflict_suppresses_weaker() {
        let mut regions = make_all_regions();
        // Strong approach, weak withdraw
        let approach: Vec<u32> = (130000..130050).collect();
        let withdraw: Vec<u32> = (135000..135010).collect();
        let motor_all: Vec<u32> = approach.iter().chain(withdraw.iter()).copied().collect();
        inject(&mut regions, RegionId::Motor, &motor_all, 1.0);

        // Active executive
        let exec: Vec<u32> = (120000..120300).collect();
        inject(&mut regions, RegionId::Executive, &exec, 1.0);

        let suppressed = resolve_motor_conflict(&mut regions, 0.01, 2.0);
        assert!(suppressed > 0, "should suppress some withdraw neurons");
    }

    #[test]
    fn test_inhibit_motor_needs_executive() {
        let mut regions = make_all_regions();
        // No executive active → can't inhibit
        let impulses = vec![(130000u32, 0.5f32), (130001, 0.5)];
        let inhibited = inhibit_motor_neurons(&mut regions, 0.01, &impulses);
        assert_eq!(inhibited, 0);
    }

    #[test]
    fn test_inhibit_motor_with_executive() {
        let mut regions = make_all_regions();
        // Activate motor neurons first
        let motor_neurons: Vec<u32> = (130000..130010).collect();
        inject(&mut regions, RegionId::Motor, &motor_neurons, 1.0);

        // Strong executive
        let exec: Vec<u32> = (120000..120400).collect();
        inject(&mut regions, RegionId::Executive, &exec, 1.0);

        let impulses: Vec<(u32, f32)> = motor_neurons.iter().map(|&g| (g, 0.5)).collect();
        let inhibited = inhibit_motor_neurons(&mut regions, 0.01, &impulses);
        assert!(inhibited > 0, "executive should inhibit motor impulse");
    }

    #[test]
    fn test_planning_signal_no_activity() {
        let regions = make_all_regions();
        let plan = planning_signal(&regions, 0.01);
        assert!((plan).abs() < 1e-6);
    }

    #[test]
    fn test_planning_signal_exec_and_language() {
        let mut regions = make_all_regions();
        let exec: Vec<u32> = (120000..120300).collect();
        let lang: Vec<u32> = (105000..105300).collect();
        inject(&mut regions, RegionId::Executive, &exec, 1.0);
        inject(&mut regions, RegionId::Language, &lang, 1.0);
        let plan = planning_signal(&regions, 0.01);
        assert!(plan > 0.2, "planning={plan} should be elevated");
    }
}
