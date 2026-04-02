/// Long-term memory (memory_long) region helpers.
///
/// Memory_long (55000–69999) stores persistent patterns with pattern
/// completion: if 40%+ of a trace's memory_long neurons fire, the
/// remaining neurons get boosted toward threshold. This IS memory recall.
///
/// Cue-based retrieval: activation from other regions (language,
/// sensory, emotion) triggers pattern completion.

use crate::core::region::{Region, RegionId};

/// Pattern completion: if enough trace neurons are active, boost the rest.
///
/// This is the core memory recall mechanism. A partial cue activates
/// some neurons of a stored pattern, and pattern completion fills in
/// the rest, reconstructing the full memory.
///
/// Returns number of neurons boosted (0 if below threshold).
pub fn pattern_completion(
    regions: &mut [Region],
    trace_memory_long_neurons: &[u32],
    activation_threshold: f32,
    boost_amount: f32,
) -> u32 {
    if trace_memory_long_neurons.is_empty() {
        return 0;
    }

    for region in regions.iter_mut() {
        if region.id == RegionId::MemoryLong {
            // Count how many trace neurons are currently active
            let mut active_count = 0;
            for &global_id in trace_memory_long_neurons {
                if let Some(local) = region.global_to_local(global_id) {
                    if region.neurons.activations[local as usize] > 0.0 {
                        active_count += 1;
                    }
                }
            }

            let ratio = active_count as f32 / trace_memory_long_neurons.len() as f32;
            if ratio < activation_threshold {
                return 0;
            }

            // Boost inactive neurons in the trace
            let mut boosted = 0;
            for &global_id in trace_memory_long_neurons {
                if let Some(local) = region.global_to_local(global_id) {
                    if region.neurons.activations[local as usize] <= 0.0 {
                        region.neurons.potentials[local as usize] += boost_amount;
                        boosted += 1;
                    }
                }
            }
            return boosted;
        }
    }
    0
}

/// Strengthen memory_long neuron synapses for a trace (during consolidation).
/// Boosts potentials of all trace neurons in memory_long.
/// Returns number of neurons strengthened.
pub fn strengthen_trace_neurons(
    regions: &mut [Region],
    trace_memory_long_neurons: &[u32],
    boost: f32,
) -> u32 {
    let mut strengthened = 0;
    for region in regions.iter_mut() {
        if region.id == RegionId::MemoryLong {
            for &global_id in trace_memory_long_neurons {
                if let Some(local) = region.global_to_local(global_id) {
                    region.neurons.potentials[local as usize] += boost;
                    strengthened += 1;
                }
            }
            break;
        }
    }
    strengthened
}

/// Get active neurons in memory_long region.
pub fn active_memory_long_ids(regions: &[Region], min_activation: f32) -> Vec<u32> {
    for region in regions {
        if region.id == RegionId::MemoryLong {
            return region.active_global_ids(min_activation);
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_completion_below_threshold() {
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        // Trace has 10 neurons in memory_long (55000-55009)
        let trace_neurons: Vec<u32> = (55000..55010).collect();

        // Only 2/10 active (20% < 40% threshold)
        for r in &mut regions {
            if r.id == RegionId::MemoryLong {
                r.neurons.activations[0] = 1.0; // 55000
                r.neurons.activations[1] = 1.0; // 55001
            }
        }

        let boosted = pattern_completion(&mut regions, &trace_neurons, 0.4, 0.5);
        assert_eq!(boosted, 0, "Below threshold should not boost");
    }

    #[test]
    fn test_pattern_completion_above_threshold() {
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        let trace_neurons: Vec<u32> = (55000..55010).collect();

        // 5/10 active (50% >= 40% threshold)
        for r in &mut regions {
            if r.id == RegionId::MemoryLong {
                for i in 0..5 {
                    r.neurons.activations[i] = 1.0;
                }
            }
        }

        let boosted = pattern_completion(&mut regions, &trace_neurons, 0.4, 0.5);
        assert_eq!(boosted, 5, "Should boost 5 inactive neurons");

        // Verify boost applied
        for r in &regions {
            if r.id == RegionId::MemoryLong {
                for i in 5..10 {
                    assert!(
                        (r.neurons.potentials[i] - 0.5).abs() < 0.01,
                        "Neuron {} should be boosted",
                        i
                    );
                }
            }
        }
    }

    #[test]
    fn test_pattern_completion_empty_trace() {
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        assert_eq!(pattern_completion(&mut regions, &[], 0.4, 0.5), 0);
    }

    #[test]
    fn test_strengthen_trace_neurons() {
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        let trace_neurons = vec![55000, 55001, 55002];
        let count = strengthen_trace_neurons(&mut regions, &trace_neurons, 0.3);
        assert_eq!(count, 3);

        for r in &regions {
            if r.id == RegionId::MemoryLong {
                assert!((r.neurons.potentials[0] - 0.3).abs() < 0.01);
            }
        }
    }
}
