use crate::core::region::{Region, RegionId};
use crate::core::tick::TickResult;

#[derive(Debug, Clone)]
pub struct ActivityCache {
    active_counts: [u32; RegionId::ALL.len()],
    active_rates: [f32; RegionId::ALL.len()],
    total_active: u32,
}

impl Default for ActivityCache {
    fn default() -> Self {
        Self {
            active_counts: [0; RegionId::ALL.len()],
            active_rates: [0.0; RegionId::ALL.len()],
            total_active: 0,
        }
    }
}

impl ActivityCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_from_tick_result(&mut self, result: &TickResult, regions: &[Region]) {
        self.total_active = result.total_active;
        for region in regions {
            let index = region.id.index();
            let count = result.active_counts.get(&region.id).copied().unwrap_or(0);
            self.active_counts[index] = count;
            self.active_rates[index] = count as f32 / region.neurons.count.max(1) as f32;
        }
    }

    pub fn refresh_region_count(&mut self, region_id: RegionId, neuron_count: u32, count: u32) {
        let index = region_id.index();
        let previous = self.active_counts[index];
        self.active_counts[index] = count;
        self.active_rates[index] = count as f32 / neuron_count.max(1) as f32;
        self.total_active = self.total_active + count - previous;
    }

    pub fn refresh_region(&mut self, region: &Region, min_activation: f32) {
        let count = region.active_count(min_activation);
        self.refresh_region_count(region.id, region.neurons.count, count);
    }

    pub fn active_count(&self, region_id: RegionId) -> u32 {
        self.active_counts[region_id.index()]
    }

    pub fn active_rate(&self, region_id: RegionId) -> f32 {
        self.active_rates[region_id.index()]
    }

    pub fn total_active(&self) -> u32 {
        self.total_active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::region::Region;

    #[test]
    fn test_refresh_region_updates_counts() {
        let mut cache = ActivityCache::new();
        let mut region = Region::new(RegionId::Sensory);
        region.neurons.activations[0] = 1.0;
        region.neurons.activations[1] = 1.0;

        cache.refresh_region(&region, 0.5);

        assert_eq!(cache.active_count(RegionId::Sensory), 2);
        assert!(cache.active_rate(RegionId::Sensory) > 0.0);
        assert_eq!(cache.total_active(), 2);
    }
}