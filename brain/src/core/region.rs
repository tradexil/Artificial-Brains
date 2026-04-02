/// Region: a named group of neurons with shared parameters.
///
/// Each region owns its NeuronStorage and knows its global neuron ID offset.
/// Local index 0 in visual region = global neuron ID 10,000.

use crate::core::neuron::{NeuronParams, NeuronStorage};

/// Region identifiers — maps 1:1 with the 14 regions in PLAN.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionId {
    Sensory,
    Visual,
    Audio,
    MemoryShort,
    MemoryLong,
    Emotion,
    Attention,
    Pattern,
    Integration,
    Language,
    Executive,
    Motor,
    Speech,
    Numbers,
}

impl RegionId {
    /// All region IDs in canonical order.
    pub const ALL: [RegionId; 14] = [
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

    /// Return (start, end) global neuron IDs for this region (inclusive).
    pub fn neuron_range(self) -> (u32, u32) {
        match self {
            RegionId::Sensory     => (0,       9_999),
            RegionId::Visual      => (10_000,  29_999),
            RegionId::Audio       => (30_000,  44_999),
            RegionId::MemoryShort => (45_000,  54_999),
            RegionId::MemoryLong  => (55_000,  69_999),
            RegionId::Emotion     => (70_000,  79_999),
            RegionId::Attention   => (80_000,  84_999),
            RegionId::Pattern     => (85_000,  94_999),
            RegionId::Integration => (95_000,  104_999),
            RegionId::Language    => (105_000, 119_999),
            RegionId::Executive   => (120_000, 129_999),
            RegionId::Motor       => (130_000, 139_999),
            RegionId::Speech      => (140_000, 149_999),
            RegionId::Numbers     => (150_000, 151_999),
        }
    }

    /// Number of neurons in this region.
    pub fn neuron_count(self) -> u32 {
        let (start, end) = self.neuron_range();
        end - start + 1
    }

    /// Default neuron parameters for this region (from PLAN.md config).
    pub fn default_params(self) -> NeuronParams {
        match self {
            RegionId::Sensory     => NeuronParams { threshold: 0.30, leak_rate: 0.85, refractory_period: 2 },
            RegionId::Visual      => NeuronParams { threshold: 0.40, leak_rate: 0.90, refractory_period: 3 },
            RegionId::Audio       => NeuronParams { threshold: 0.35, leak_rate: 0.88, refractory_period: 2 },
            RegionId::MemoryShort => NeuronParams { threshold: 0.50, leak_rate: 0.95, refractory_period: 5 },
            RegionId::MemoryLong  => NeuronParams { threshold: 0.60, leak_rate: 0.99, refractory_period: 8 },
            RegionId::Emotion     => NeuronParams { threshold: 0.30, leak_rate: 0.92, refractory_period: 4 },
            RegionId::Attention   => NeuronParams { threshold: 0.40, leak_rate: 0.93, refractory_period: 3 },
            RegionId::Pattern     => NeuronParams { threshold: 0.50, leak_rate: 0.90, refractory_period: 4 },
            RegionId::Integration => NeuronParams { threshold: 0.55, leak_rate: 0.92, refractory_period: 5 },
            RegionId::Language    => NeuronParams { threshold: 0.50, leak_rate: 0.94, refractory_period: 4 },
            RegionId::Executive   => NeuronParams { threshold: 0.60, leak_rate: 0.93, refractory_period: 6 },
            RegionId::Motor       => NeuronParams { threshold: 0.55, leak_rate: 0.85, refractory_period: 3 },
            RegionId::Speech      => NeuronParams { threshold: 0.50, leak_rate: 0.87, refractory_period: 3 },
            RegionId::Numbers     => NeuronParams { threshold: 0.50, leak_rate: 0.96, refractory_period: 4 },
        }
    }

    /// Default inhibitory neuron percentage for this region.
    pub fn inhibitory_pct(self) -> f32 {
        match self {
            RegionId::Sensory     => 0.15,
            RegionId::Visual      => 0.20,
            RegionId::Audio       => 0.20,
            RegionId::MemoryShort => 0.25,
            RegionId::MemoryLong  => 0.20,
            RegionId::Emotion     => 0.15,
            RegionId::Attention   => 0.40,
            RegionId::Pattern     => 0.25,
            RegionId::Integration => 0.20,
            RegionId::Language    => 0.20,
            RegionId::Executive   => 0.30,
            RegionId::Motor       => 0.20,
            RegionId::Speech      => 0.20,
            RegionId::Numbers     => 0.10,
        }
    }

    /// String name for Python interop.
    pub fn name(self) -> &'static str {
        match self {
            RegionId::Sensory     => "sensory",
            RegionId::Visual      => "visual",
            RegionId::Audio       => "audio",
            RegionId::MemoryShort => "memory_short",
            RegionId::MemoryLong  => "memory_long",
            RegionId::Emotion     => "emotion",
            RegionId::Attention   => "attention",
            RegionId::Pattern     => "pattern",
            RegionId::Integration => "integration",
            RegionId::Language    => "language",
            RegionId::Executive   => "executive",
            RegionId::Motor       => "motor",
            RegionId::Speech      => "speech",
            RegionId::Numbers     => "numbers",
        }
    }

    /// Look up region by name string (for Python interop).
    pub fn from_name(name: &str) -> Option<RegionId> {
        match name {
            "sensory"      => Some(RegionId::Sensory),
            "visual"       => Some(RegionId::Visual),
            "audio"        => Some(RegionId::Audio),
            "memory_short" => Some(RegionId::MemoryShort),
            "memory_long"  => Some(RegionId::MemoryLong),
            "emotion"      => Some(RegionId::Emotion),
            "attention"    => Some(RegionId::Attention),
            "pattern"      => Some(RegionId::Pattern),
            "integration"  => Some(RegionId::Integration),
            "language"     => Some(RegionId::Language),
            "executive"    => Some(RegionId::Executive),
            "motor"        => Some(RegionId::Motor),
            "speech"       => Some(RegionId::Speech),
            "numbers"      => Some(RegionId::Numbers),
            _              => None,
        }
    }
}

/// A Region owns its neurons and knows its position in global neuron space.
pub struct Region {
    pub id: RegionId,
    pub neurons: NeuronStorage,
    pub global_offset: u32, // first global neuron ID in this region

    /// Per-tick incoming signal buffer. Accumulated during propagation,
    /// consumed during neuron update, then zeroed.
    pub incoming: Vec<f32>,
}

impl Region {
    pub fn new(id: RegionId) -> Self {
        let (start, _end) = id.neuron_range();
        let count = id.neuron_count();
        let params = id.default_params();
        let inhibitory_pct = id.inhibitory_pct();

        Region {
            id,
            neurons: NeuronStorage::new(count, params, inhibitory_pct),
            global_offset: start,
            incoming: vec![0.0; count as usize],
        }
    }

    /// Convert global neuron ID to local index within this region.
    /// Returns None if the ID doesn't belong to this region.
    #[inline]
    pub fn global_to_local(&self, global_id: u32) -> Option<u32> {
        if global_id >= self.global_offset
            && global_id < self.global_offset + self.neurons.count
        {
            Some(global_id - self.global_offset)
        } else {
            None
        }
    }

    /// Convert local index to global neuron ID.
    #[inline]
    pub fn local_to_global(&self, local_idx: u32) -> u32 {
        self.global_offset + local_idx
    }

    /// Add signal to a neuron's incoming buffer (by global ID).
    /// Returns false if the neuron doesn't belong to this region.
    #[inline]
    pub fn add_incoming_global(&mut self, global_id: u32, value: f32) -> bool {
        if let Some(local) = self.global_to_local(global_id) {
            self.incoming[local as usize] += value;
            true
        } else {
            false
        }
    }

    /// Prepare for new tick: swap activation buffers.
    pub fn pre_tick(&mut self) {
        self.neurons.swap_activation_buffers();
        self.incoming.fill(0.0);
    }

    /// Update all neurons using accumulated incoming signals.
    pub fn update_neurons(&mut self) {
        self.neurons.update(&self.incoming);
    }

    /// Get list of currently active neurons as global IDs.
    pub fn active_global_ids(&self, min_activation: f32) -> Vec<u32> {
        self.neurons
            .active_neurons(min_activation)
            .into_iter()
            .map(|local| self.local_to_global(local))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_neuron_counts() {
        let total: u32 = RegionId::ALL.iter().map(|r| r.neuron_count()).sum();
        assert_eq!(total, 152_000);
    }

    #[test]
    fn test_region_ranges_contiguous() {
        // Verify no gaps or overlaps between regions
        let mut ranges: Vec<(u32, u32)> = RegionId::ALL.iter().map(|r| r.neuron_range()).collect();
        ranges.sort_by_key(|r| r.0);
        for i in 1..ranges.len() {
            assert_eq!(ranges[i].0, ranges[i - 1].1 + 1,
                "Gap or overlap between regions at boundary {} - {}",
                ranges[i - 1].1, ranges[i].0);
        }
    }

    #[test]
    fn test_global_local_conversion() {
        let region = Region::new(RegionId::Visual);
        assert_eq!(region.global_to_local(10_000), Some(0));
        assert_eq!(region.global_to_local(29_999), Some(19_999));
        assert_eq!(region.global_to_local(30_000), None);
        assert_eq!(region.global_to_local(9_999), None);
        assert_eq!(region.local_to_global(0), 10_000);
    }

    #[test]
    fn test_region_incoming() {
        let mut region = Region::new(RegionId::Sensory);
        assert!(region.add_incoming_global(500, 0.5));
        assert!(!region.add_incoming_global(10_000, 0.5)); // belongs to visual
        assert!((region.incoming[500] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_region_from_name_roundtrip() {
        for id in RegionId::ALL {
            assert_eq!(RegionId::from_name(id.name()), Some(id));
        }
    }
}
