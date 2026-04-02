// regions/language.rs — Language region logic.
//
// Language region (105000–119999, 15k neurons, 20% inhibitory):
//   Processes symbolic representations, supports relational logic,
//   and enables the language↔executive inner monologue loop.
//
// Sub-populations:
//   Token neurons (105000–113999): activated by specific words/tokens
//   Relational neurons (114000–116999): encode relationships (A→B)
//   Compositional neurons (117000–119999): combine tokens into phrases (includes inhibitory at tail)

use crate::core::region::{Region, RegionId};

const LANG_START: u32 = 105_000;
const LANG_END: u32 = 120_000;
const LANG_COUNT: u32 = 15_000;

// Sub-population boundaries
const TOKEN_START: u32 = 105_000;
const TOKEN_END: u32 = 114_000;    // 9k token neurons
const RELATIONAL_START: u32 = 114_000;
const RELATIONAL_END: u32 = 117_000; // 3k relational neurons
const COMPOSITIONAL_START: u32 = 117_000;
const COMPOSITIONAL_END: u32 = 120_000; // 3k (includes inhibitory at tail)

/// Compute language region activation strength (0.0–1.0).
pub fn language_activation_strength(regions: &[Region], min_activation: f32) -> f32 {
    let lang_idx = regions.iter().position(|r| r.id == RegionId::Language);
    let region = match lang_idx {
        Some(i) => &regions[i],
        None => return 0.0,
    };

    let active = region.active_global_ids(min_activation).len() as f32;
    // Scale: 5% firing = strong activation
    (active / (LANG_COUNT as f32 * 0.05)).min(1.0)
}

/// Compute overlap between active language neurons and a set of token neurons.
/// Returns activation ratio (0.0–1.0).
pub fn symbol_overlap(
    regions: &[Region],
    trace_lang_neurons: &[u32],
    min_activation: f32,
) -> f32 {
    if trace_lang_neurons.is_empty() {
        return 0.0;
    }

    let lang_idx = match regions.iter().position(|r| r.id == RegionId::Language) {
        Some(i) => i,
        None => return 0.0,
    };

    let active: std::collections::HashSet<u32> = regions[lang_idx]
        .active_global_ids(min_activation)
        .into_iter()
        .collect();

    let matched = trace_lang_neurons
        .iter()
        .filter(|n| active.contains(n))
        .count();

    matched as f32 / trace_lang_neurons.len() as f32
}

/// Compute inner monologue signal strength.
/// Requires both language and executive to be active (bidirectional loop).
pub fn inner_monologue_signal(regions: &[Region], min_activation: f32) -> f32 {
    let lang_strength = language_activation_strength(regions, min_activation);

    let exec_idx = regions.iter().position(|r| r.id == RegionId::Executive);
    let exec_strength = match exec_idx {
        Some(i) => {
            let active = regions[i].active_global_ids(min_activation).len() as f32;
            (active / (regions[i].neurons.count as f32 * 0.05)).min(1.0)
        }
        None => 0.0,
    };

    // Inner monologue = both language and executive engaged
    // Geometric mean ensures both must be active
    (lang_strength * exec_strength).sqrt()
}

/// Boost specific language neurons (for token activation).
/// Returns count of neurons boosted.
pub fn boost_language_neurons(
    regions: &mut [Region],
    neurons: &[u32],
    boost: f32,
) -> u32 {
    let lang_idx = match regions.iter().position(|r| r.id == RegionId::Language) {
        Some(i) => i,
        None => return 0,
    };

    let mut count = 0u32;
    for &gid in neurons {
        if gid >= LANG_START && gid < LANG_END {
            if let Some(local) = regions[lang_idx].global_to_local(gid) {
                regions[lang_idx].neurons.activations[local as usize] += boost;
                count += 1;
            }
        }
    }
    count
}

/// Get the top-K most active token neurons (for reading language output).
pub fn peak_language_neurons(
    regions: &[Region],
    min_activation: f32,
    top_k: usize,
) -> Vec<(u32, f32)> {
    let lang_idx = match regions.iter().position(|r| r.id == RegionId::Language) {
        Some(i) => i,
        None => return Vec::new(),
    };

    let mut active: Vec<(u32, f32)> = Vec::new();
    let region = &regions[lang_idx];

    for i in 0..region.neurons.count as usize {
        let act = region.neurons.activations[i];
        if act > min_activation {
            let gid = region.local_to_global(i as u32);
            // Only token neurons for output
            if gid >= TOKEN_START && gid < TOKEN_END {
                active.push((gid, act));
            }
        }
    }

    active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    active.truncate(top_k);
    active
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
    fn test_language_activation_silent() {
        let regions = make_all_regions();
        let s = language_activation_strength(&regions, 0.01);
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn test_language_activation_active() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (105000..105300).collect();
        inject(&mut regions, RegionId::Language, &neurons, 1.0);
        let s = language_activation_strength(&regions, 0.01);
        assert!(s > 0.2, "strength={s}");
    }

    #[test]
    fn test_symbol_overlap_none() {
        let regions = make_all_regions();
        let overlap = symbol_overlap(&regions, &[105000, 105001, 105002], 0.01);
        assert!(overlap.abs() < 1e-6);
    }

    #[test]
    fn test_symbol_overlap_full() {
        let mut regions = make_all_regions();
        let trace_neurons = vec![105000u32, 105001, 105002];
        inject(&mut regions, RegionId::Language, &trace_neurons, 1.0);
        let overlap = symbol_overlap(&regions, &trace_neurons, 0.01);
        assert!((overlap - 1.0).abs() < 0.01, "overlap={overlap}");
    }

    #[test]
    fn test_symbol_overlap_partial() {
        let mut regions = make_all_regions();
        let trace_neurons = vec![105000u32, 105001, 105002, 105003];
        // Only activate half
        inject(&mut regions, RegionId::Language, &[105000, 105001], 1.0);
        let overlap = symbol_overlap(&regions, &trace_neurons, 0.01);
        assert!((overlap - 0.5).abs() < 0.01, "overlap={overlap}");
    }

    #[test]
    fn test_inner_monologue_needs_both() {
        let mut regions = make_all_regions();
        // Only language → weak monologue
        let lang: Vec<u32> = (105000..105300).collect();
        inject(&mut regions, RegionId::Language, &lang, 1.0);
        let m1 = inner_monologue_signal(&regions, 0.01);

        // Add executive
        let exec: Vec<u32> = (120000..120300).collect();
        inject(&mut regions, RegionId::Executive, &exec, 1.0);
        let m2 = inner_monologue_signal(&regions, 0.01);

        assert!(m2 > m1, "monologue with exec ({m2}) should be > without ({m1})");
    }

    #[test]
    fn test_inner_monologue_zero_when_silent() {
        let regions = make_all_regions();
        let m = inner_monologue_signal(&regions, 0.01);
        assert!(m.abs() < 1e-6);
    }

    #[test]
    fn test_boost_language_neurons() {
        let mut regions = make_all_regions();
        let neurons = vec![105000u32, 105001, 105002];
        let count = boost_language_neurons(&mut regions, &neurons, 0.5);
        assert_eq!(count, 3);
        let idx = regions.iter().position(|r| r.id == RegionId::Language).unwrap();
        assert!((regions[idx].neurons.activations[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_peak_language_neurons() {
        let mut regions = make_all_regions();
        let neurons: Vec<u32> = (105000..105020).collect();
        inject(&mut regions, RegionId::Language, &neurons, 1.0);
        let peaks = peak_language_neurons(&regions, 0.01, 5);
        assert!(!peaks.is_empty());
        assert!(peaks.len() <= 5);
        // Should be sorted descending
        for i in 1..peaks.len() {
            assert!(peaks[i].1 <= peaks[i - 1].1);
        }
    }

    #[test]
    fn test_boost_out_of_range_ignored() {
        let mut regions = make_all_regions();
        let count = boost_language_neurons(&mut regions, &[0, 1, 200000], 0.5);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_peak_only_token_neurons() {
        let mut regions = make_all_regions();
        // Activate relational neurons (114000+) — should NOT appear in peak_language_neurons
        let rel_neurons: Vec<u32> = (114000..114020).collect();
        inject(&mut regions, RegionId::Language, &rel_neurons, 1.0);
        let peaks = peak_language_neurons(&regions, 0.01, 10);
        // Relational neurons are outside TOKEN_START..TOKEN_END
        for &(gid, _) in &peaks {
            assert!(gid >= TOKEN_START && gid < TOKEN_END,
                "non-token neuron {gid} found in peaks");
        }
    }
}
