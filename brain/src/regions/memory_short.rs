/// Working memory (memory_short) region helpers.
///
/// Memory_short (45000–54999) has capacity-limited working memory.
/// At most 7 ± 2 traces can be simultaneously active, enforced by
/// strong lateral inhibition.
///
/// Items in working memory get boosted attention gain (2x).
/// Items decay rapidly if not refreshed (rehearsal).

use crate::core::region::{Region, RegionId};
use std::collections::HashSet;

/// Boost neurons that belong to active working memory traces.
/// Returns number of neurons boosted.
pub fn boost_working_memory_neurons(
    regions: &mut [Region],
    trace_neurons: &[u32],
    boost: f32,
) -> u32 {
    let mut boosted = 0;
    for region in regions.iter_mut() {
        if region.id == RegionId::MemoryShort {
            for &global_id in trace_neurons {
                if let Some(local) = region.global_to_local(global_id) {
                    region.neurons.potentials[local as usize] += boost;
                    boosted += 1;
                }
            }
            break;
        }
    }
    boosted
}

/// Suppress neurons not part of active traces (lateral inhibition).
/// This enforces the capacity limit at the neuron level.
/// Returns number of neurons suppressed.
pub fn suppress_non_trace_neurons(
    regions: &mut [Region],
    active_trace_neurons: &HashSet<u32>,
    suppression: f32,
) -> u32 {
    let mut suppressed = 0;
    for region in regions.iter_mut() {
        if region.id == RegionId::MemoryShort {
            for i in 0..region.neurons.count as usize {
                let global_id = region.local_to_global(i as u32);
                if region.neurons.activations[i] > 0.0
                    && !active_trace_neurons.contains(&global_id)
                {
                    region.neurons.potentials[i] -= suppression;
                    if region.neurons.potentials[i] < 0.0 {
                        region.neurons.potentials[i] = 0.0;
                    }
                    suppressed += 1;
                }
            }
            break;
        }
    }
    suppressed
}

/// Count active neurons in working memory region.
pub fn active_neuron_count(regions: &[Region], min_activation: f32) -> u32 {
    for region in regions {
        if region.id == RegionId::MemoryShort {
            return region.active_count(min_activation);
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boost_neurons() {
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        // Memory_short starts at 45000
        let trace_neurons = vec![45000, 45001, 45002];
        let boosted = boost_working_memory_neurons(&mut regions, &trace_neurons, 0.3);
        assert_eq!(boosted, 3);

        for r in &regions {
            if r.id == RegionId::MemoryShort {
                assert!((r.neurons.potentials[0] - 0.3).abs() < 0.01);
                assert!((r.neurons.potentials[1] - 0.3).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_suppress_non_trace() {
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();

        // Activate some neurons
        for r in &mut regions {
            if r.id == RegionId::MemoryShort {
                r.neurons.activations[0] = 1.0; // in trace
                r.neurons.activations[1] = 1.0; // NOT in trace
                r.neurons.potentials[1] = 0.5;
            }
        }

        let mut trace_set = HashSet::new();
        trace_set.insert(45000u32);  // neuron 0 is in trace

        let suppressed = suppress_non_trace_neurons(&mut regions, &trace_set, 0.3);
        assert_eq!(suppressed, 1);

        for r in &regions {
            if r.id == RegionId::MemoryShort {
                assert!((r.neurons.potentials[1] - 0.2).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_active_neuron_count() {
        let regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        assert_eq!(active_neuron_count(&regions, 0.5), 0);
    }
}
