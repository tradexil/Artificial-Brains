// core/homeostasis.rs — Homeostatic regulation system.
//
// Maintains set-points for arousal, valence, and focus via exponential
// decay toward baselines. Tracks sleep pressure that accumulates while
// awake and dissipates during sleep.
//
// This is NOT a separate region — it modifies the NeuromodulatorSystem
// each tick, pulling modulators back toward equilibrium.

/// Homeostatic regulation system.
pub struct HomeostasisSystem {
    // Set-points (baselines)
    pub arousal_baseline: f32,
    pub valence_baseline: f32,
    pub focus_baseline: f32,

    // Regulation rates (per tick, 0..1 — higher = faster return to baseline)
    pub arousal_reg_rate: f32,
    pub valence_reg_rate: f32,
    pub focus_reg_rate: f32,

    // Sleep pressure: accumulates while awake, dissipates during sleep
    pub sleep_pressure: f32,
    /// Sleep pressure accumulation rate per tick while awake.
    pub sleep_pressure_rate: f32,
    /// Sleep pressure dissipation rate per tick while sleeping.
    pub sleep_dissipation_rate: f32,

    // Circadian-like cycle
    /// Current phase in the circadian cycle (0.0 to 1.0).
    pub circadian_phase: f32,
    /// How many ticks per full circadian cycle.
    pub circadian_period: u64,

    // Tracking
    pub ticks_awake: u64,
    pub ticks_asleep: u64,
}

impl HomeostasisSystem {
    pub fn new() -> Self {
        Self {
            arousal_baseline: 0.5,
            valence_baseline: 0.0,
            focus_baseline: 0.5,
            arousal_reg_rate: 0.005,
            valence_reg_rate: 0.002,
            focus_reg_rate: 0.003,
            sleep_pressure: 0.0,
            sleep_pressure_rate: 0.00002,
            sleep_dissipation_rate: 0.0001,
            circadian_phase: 0.0,
            circadian_period: 100_000,
            ticks_awake: 0,
            ticks_asleep: 0,
        }
    }

    /// Regulate arousal/valence/focus toward baselines.
    /// Returns (arousal_delta, valence_delta, focus_delta) — the corrections applied.
    pub fn regulate(
        &self,
        arousal: f32,
        valence: f32,
        focus: f32,
    ) -> (f32, f32, f32) {
        let arousal_delta = (self.arousal_baseline - arousal) * self.arousal_reg_rate;
        let valence_delta = (self.valence_baseline - valence) * self.valence_reg_rate;
        let focus_delta = (self.focus_baseline - focus) * self.focus_reg_rate;
        (arousal_delta, valence_delta, focus_delta)
    }

    /// Accumulate sleep pressure (called every tick while awake).
    pub fn accumulate_pressure(&mut self) {
        // Circadian modulation: pressure builds faster in the "night" half
        let circadian_factor = 1.0 + 0.5 * (self.circadian_phase * std::f32::consts::TAU).sin().max(0.0);
        self.sleep_pressure += self.sleep_pressure_rate * circadian_factor;
        if self.sleep_pressure > 1.0 {
            self.sleep_pressure = 1.0;
        }
        self.ticks_awake += 1;
    }

    /// Dissipate sleep pressure (called every tick while sleeping).
    pub fn dissipate_pressure(&mut self) {
        self.sleep_pressure -= self.sleep_dissipation_rate;
        if self.sleep_pressure < 0.0 {
            self.sleep_pressure = 0.0;
        }
        self.ticks_asleep += 1;
    }

    /// Advance the circadian phase by one tick.
    pub fn advance_circadian(&mut self) {
        self.circadian_phase += 1.0 / self.circadian_period as f32;
        if self.circadian_phase >= 1.0 {
            self.circadian_phase -= 1.0;
        }
    }

    /// Energy depletion rate multiplier from circadian cycle.
    /// During "night" phase (0.5–1.0), energy depletes faster.
    pub fn circadian_energy_modifier(&self) -> f32 {
        // Day phase (0.0–0.5): modifier ~1.0
        // Night phase (0.5–1.0): modifier up to 1.5
        1.0 + 0.5 * (self.circadian_phase * std::f32::consts::TAU).sin().max(0.0)
    }

    /// Should the brain enter sleep? Based on sleep_pressure + energy.
    pub fn should_sleep(&self, energy: f32) -> bool {
        self.sleep_pressure > 0.7 || energy < 0.15
    }

    /// Should the brain wake up? Based on sleep_pressure + energy.
    pub fn should_wake(&self, energy: f32) -> bool {
        self.sleep_pressure < 0.1 && energy > 0.7
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Get a summary: (sleep_pressure, circadian_phase, ticks_awake, ticks_asleep).
    pub fn summary(&self) -> (f32, f32, u64, u64) {
        (self.sleep_pressure, self.circadian_phase, self.ticks_awake, self.ticks_asleep)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homeostasis_default() {
        let h = HomeostasisSystem::new();
        assert!((h.arousal_baseline - 0.5).abs() < 1e-6);
        assert!((h.sleep_pressure).abs() < 1e-6);
        assert_eq!(h.ticks_awake, 0);
        assert_eq!(h.ticks_asleep, 0);
    }

    #[test]
    fn test_regulate_toward_baseline() {
        let h = HomeostasisSystem::new();
        // Arousal at 1.0 should push down toward 0.5
        let (ad, vd, fd) = h.regulate(1.0, 0.0, 0.5);
        assert!(ad < 0.0, "Arousal delta should be negative: {}", ad);
        assert!((vd).abs() < 1e-6, "Valence at baseline should have ~0 delta");
        assert!((fd).abs() < 1e-6, "Focus at baseline should have ~0 delta");
    }

    #[test]
    fn test_regulate_low_arousal() {
        let h = HomeostasisSystem::new();
        let (ad, _, _) = h.regulate(0.0, 0.0, 0.5);
        assert!(ad > 0.0, "Low arousal should push up: {}", ad);
    }

    #[test]
    fn test_sleep_pressure_accumulation() {
        let mut h = HomeostasisSystem::new();
        for _ in 0..10000 {
            h.accumulate_pressure();
        }
        assert!(h.sleep_pressure > 0.1, "Sleep pressure should accumulate: {}", h.sleep_pressure);
        assert_eq!(h.ticks_awake, 10000);
    }

    #[test]
    fn test_sleep_pressure_dissipation() {
        let mut h = HomeostasisSystem::new();
        h.sleep_pressure = 0.5;
        for _ in 0..2000 {
            h.dissipate_pressure();
        }
        assert!(h.sleep_pressure < 0.5, "Sleep pressure should dissipate: {}", h.sleep_pressure);
        assert_eq!(h.ticks_asleep, 2000);
    }

    #[test]
    fn test_sleep_pressure_capped() {
        let mut h = HomeostasisSystem::new();
        h.sleep_pressure = 0.99;
        h.accumulate_pressure();
        assert!(h.sleep_pressure <= 1.0);
    }

    #[test]
    fn test_sleep_pressure_floor() {
        let mut h = HomeostasisSystem::new();
        h.sleep_pressure = 0.001;
        for _ in 0..100 {
            h.dissipate_pressure();
        }
        assert!(h.sleep_pressure >= 0.0);
    }

    #[test]
    fn test_circadian_advance() {
        let mut h = HomeostasisSystem::new();
        for _ in 0..h.circadian_period {
            h.advance_circadian();
        }
        assert!((h.circadian_phase).abs() < 0.01, "Should complete one cycle: {}", h.circadian_phase);
    }

    #[test]
    fn test_circadian_energy_modifier() {
        let mut h = HomeostasisSystem::new();
        h.circadian_phase = 0.0;
        let m0 = h.circadian_energy_modifier();
        h.circadian_phase = 0.25; // peak "night"
        let m25 = h.circadian_energy_modifier();
        assert!(m25 > m0, "Night modifier should be higher: {} vs {}", m25, m0);
    }

    #[test]
    fn test_should_sleep() {
        let h = HomeostasisSystem::new();
        assert!(!h.should_sleep(1.0), "Should not sleep at full energy with no pressure");
        assert!(h.should_sleep(0.1), "Should sleep at very low energy");
    }

    #[test]
    fn test_should_wake() {
        let mut h = HomeostasisSystem::new();
        h.sleep_pressure = 0.0;
        assert!(h.should_wake(0.8), "Should wake with low pressure + high energy");
        h.sleep_pressure = 0.5;
        assert!(!h.should_wake(0.8), "Should NOT wake with high pressure");
    }

    #[test]
    fn test_full_cycle() {
        let mut h = HomeostasisSystem::new();
        // Accumulate pressure
        for _ in 0..50_000 {
            h.accumulate_pressure();
            h.advance_circadian();
        }
        assert!(h.sleep_pressure > 0.5);

        // Dissipate during sleep
        for _ in 0..10_000 {
            h.dissipate_pressure();
            h.advance_circadian();
        }
        let post_sleep = h.sleep_pressure;
        assert!(post_sleep < h.sleep_pressure + 0.5); // Some reduction
        assert!(h.ticks_awake == 50_000);
        assert!(h.ticks_asleep == 10_000);
    }

    #[test]
    fn test_reset() {
        let mut h = HomeostasisSystem::new();
        h.sleep_pressure = 0.9;
        h.ticks_awake = 100000;
        h.reset();
        assert!((h.sleep_pressure).abs() < 1e-6);
        assert_eq!(h.ticks_awake, 0);
    }
}
