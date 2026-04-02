/// Pattern recognition helpers: activation prediction and error computation.
///
/// PredictionState tracks expected activation rates per region using an
/// exponential moving average. Each tick it compares predicted vs actual
/// firing rates and produces per-region prediction errors.
///
/// Prediction error is the core learning signal:
///   error > 0.8  → ALARM       (hypervigilance, arousal spike)
///   error > 0.5  → SURPRISE    (attention spike, learning rate ×2)
///   error 0.1-0.5 → INTERESTING (moderate attention, normal learning)
///   error < 0.1  → EXPECTED    (attention drops, minimal learning)

use crate::core::region::{Region, RegionId};
use std::collections::HashMap;

/// Error classification thresholds (mirror config.py values).
pub const ALARM_THRESHOLD: f32 = 0.8;
pub const SURPRISE_THRESHOLD: f32 = 0.5;
pub const BORING_THRESHOLD: f32 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    Expected,
    Interesting,
    Surprise,
    Alarm,
}

impl ErrorClass {
    pub fn name(&self) -> &'static str {
        match self {
            ErrorClass::Expected => "expected",
            ErrorClass::Interesting => "interesting",
            ErrorClass::Surprise => "surprise",
            ErrorClass::Alarm => "alarm",
        }
    }
}

/// Classify a prediction error value.
pub fn classify_error(error: f32) -> ErrorClass {
    if error > ALARM_THRESHOLD {
        ErrorClass::Alarm
    } else if error > SURPRISE_THRESHOLD {
        ErrorClass::Surprise
    } else if error > BORING_THRESHOLD {
        ErrorClass::Interesting
    } else {
        ErrorClass::Expected
    }
}

/// Tracks predicted activation and computes prediction errors per region.
pub struct PredictionState {
    /// EMA of activation rate per region.
    predicted_rates: HashMap<RegionId, f32>,
    /// Last computed prediction errors.
    last_errors: HashMap<RegionId, f32>,
    /// EMA smoothing factor (higher = faster adaptation).
    alpha: f32,
    /// Tick counter.
    tick_count: u64,
}

impl PredictionState {
    pub fn new(alpha: f32) -> Self {
        let predicted: HashMap<RegionId, f32> =
            RegionId::ALL.iter().map(|&id| (id, 0.0)).collect();

        Self {
            predicted_rates: predicted,
            last_errors: HashMap::new(),
            alpha,
            tick_count: 0,
        }
    }

    /// Update predictions with actual region data, return per-region errors.
    ///
    /// For each region:
    ///   1. Compute actual firing rate
    ///   2. Compare to predicted → error
    ///   3. Update predicted (EMA)
    pub fn update(&mut self, regions: &[Region]) -> HashMap<RegionId, f32> {
        let mut errors = HashMap::new();
        self.tick_count += 1;

        for region in regions {
            let active = region.active_global_ids(0.5).len() as f32;
            let actual_rate = active / region.neurons.count.max(1) as f32;

            let predicted = *self.predicted_rates.get(&region.id).unwrap_or(&0.0);

            // Prediction error: absolute difference, scaled to 0–1
            // Scale ×20 because firing rates are typically very small (<5%)
            let error = ((actual_rate - predicted).abs() * 20.0).min(1.0);
            errors.insert(region.id, error);

            // Update EMA
            let new_predicted = self.alpha * actual_rate + (1.0 - self.alpha) * predicted;
            self.predicted_rates.insert(region.id, new_predicted);
        }

        self.last_errors = errors.clone();
        errors
    }

    /// Compute global prediction error (mean of per-region errors).
    pub fn global_error(&self) -> f32 {
        if self.last_errors.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.last_errors.values().sum();
        sum / self.last_errors.len() as f32
    }

    /// Get last errors map.
    pub fn last_errors(&self) -> &HashMap<RegionId, f32> {
        &self.last_errors
    }

    /// Get predicted rate for a specific region.
    pub fn predicted_rate(&self, region: RegionId) -> f32 {
        *self.predicted_rates.get(&region).unwrap_or(&0.0)
    }

    /// Get error for a specific region from last update.
    pub fn error_for(&self, region: RegionId) -> f32 {
        *self.last_errors.get(&region).unwrap_or(&0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_error() {
        assert_eq!(classify_error(0.05), ErrorClass::Expected);
        assert_eq!(classify_error(0.3), ErrorClass::Interesting);
        assert_eq!(classify_error(0.6), ErrorClass::Surprise);
        assert_eq!(classify_error(0.9), ErrorClass::Alarm);
    }

    #[test]
    fn test_prediction_state_no_activity() {
        let mut ps = PredictionState::new(0.05);
        let regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        let errors = ps.update(&regions);
        // No activity, predicted 0 → error should be 0
        for &e in errors.values() {
            assert!(e < 0.01, "No activity should mean no error");
        }
    }

    #[test]
    fn test_prediction_error_on_surprise() {
        let mut ps = PredictionState::new(0.05);
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();

        // First tick: no activity, predicted stays at 0
        ps.update(&regions);

        // Second tick: sudden activity in sensory (lots of neurons fire)
        for region in &mut regions {
            if region.id == RegionId::Sensory {
                // Fire 20% of neurons → actual_rate = 0.2
                for i in 0..2000 {
                    region.neurons.activations[i] = 1.0;
                }
            }
        }
        let errors = ps.update(&regions);
        let sensory_error = *errors.get(&RegionId::Sensory).unwrap();
        // predicted ~0, actual 0.2, error = 0.2 * 20 = 4.0 → clamped to 1.0
        assert!(sensory_error > 0.5, "Surprise should cause high error: {}", sensory_error);
    }

    #[test]
    fn test_prediction_adapts_over_time() {
        let mut ps = PredictionState::new(0.1);
        let mut regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();

        // Sustained activity in visual
        for region in &mut regions {
            if region.id == RegionId::Visual {
                for i in 0..100 {
                    region.neurons.activations[i] = 1.0;
                }
            }
        }

        // Run many ticks with same activity
        let mut last_error = 1.0;
        for _ in 0..50 {
            let errors = ps.update(&regions);
            let e = *errors.get(&RegionId::Visual).unwrap();
            assert!(e <= last_error + 0.01, "Error should decrease or stay stable");
            last_error = e;
        }
        // After adaptation, error should be low
        assert!(last_error < 0.3, "Adapted prediction should have low error: {}", last_error);
    }

    #[test]
    fn test_global_error() {
        let mut ps = PredictionState::new(0.05);
        let regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();
        ps.update(&regions);
        let ge = ps.global_error();
        assert!(ge >= 0.0 && ge <= 1.0, "Global error should be in range");
    }
}
