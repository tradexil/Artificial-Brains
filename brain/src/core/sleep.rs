// core/sleep.rs — Sleep/wake cycle state machine.
//
// Five states: Awake → Drowsy → Light → Deep → REM → Light (cycles).
// Each state has an input gating factor that multiplies attention gains,
// reducing external input processing during sleep.
//
// Transitions are driven by energy, sleep pressure, and time-in-state.
// During REM, a flag signals Python to perform dream replay (trace re-activation).

use serde::{Deserialize, Serialize};

/// Sleep cycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SleepState {
    Awake,
    Drowsy,
    Light,
    Deep,
    Rem,
}

impl SleepState {
    pub fn name(&self) -> &'static str {
        match self {
            SleepState::Awake => "awake",
            SleepState::Drowsy => "drowsy",
            SleepState::Light => "light",
            SleepState::Deep => "deep",
            SleepState::Rem => "rem",
        }
    }

    /// Input gating multiplier for this sleep state.
    /// Awake=1.0, Drowsy=0.5, Light=0.2, Deep=0.05, REM=0.1
    pub fn input_gate(&self) -> f32 {
        match self {
            SleepState::Awake => 1.0,
            SleepState::Drowsy => 0.5,
            SleepState::Light => 0.2,
            SleepState::Deep => 0.05,
            SleepState::Rem => 0.1,
        }
    }

    /// Energy recovery rate multiplier per state.
    /// Sleep recovers energy; deeper sleep recovers more.
    pub fn energy_recovery_rate(&self) -> f32 {
        match self {
            SleepState::Awake => 0.0,
            SleepState::Drowsy => 0.0002,
            SleepState::Light => 0.0005,
            SleepState::Deep => 0.001,
            SleepState::Rem => 0.0003,
        }
    }
}

/// Sleep cycle manager.
#[derive(Clone, Serialize, Deserialize)]
pub struct SleepCycleManager {
    pub state: SleepState,
    /// Ticks spent in current state.
    pub ticks_in_state: u64,
    /// Total full sleep cycles completed (Light→Deep→REM→Light = 1 cycle).
    pub cycles_completed: u32,
    /// Number of REM episodes this sleep session.
    pub rem_episodes: u32,

    // Duration parameters (ticks)
    pub drowsy_duration: u64,
    pub light_duration: u64,
    pub deep_duration: u64,
    pub rem_duration: u64,

    // Transition thresholds
    pub sleep_pressure_threshold: f32, // pressure above which sleep starts
    pub wake_energy_threshold: f32,    // energy above which wake-up allowed
    pub wake_pressure_threshold: f32,  // pressure below which wake-up allowed
}

impl SleepCycleManager {
    pub fn new() -> Self {
        Self {
            state: SleepState::Awake,
            ticks_in_state: 0,
            cycles_completed: 0,
            rem_episodes: 0,
            drowsy_duration: 2_000,
            light_duration: 5_000,
            deep_duration: 8_000,
            rem_duration: 5_000,
            sleep_pressure_threshold: 0.7,
            wake_energy_threshold: 0.7,
            wake_pressure_threshold: 0.1,
        }
    }

    /// Update sleep state machine. Call once per tick.
    /// Returns true if state changed this tick.
    pub fn update(&mut self, energy: f32, sleep_pressure: f32) -> bool {
        self.ticks_in_state += 1;
        let old_state = self.state;

        match self.state {
            SleepState::Awake => {
                // Transition to Drowsy if sleep pressure or energy triggers it
                if sleep_pressure > self.sleep_pressure_threshold || energy < 0.15 {
                    self.transition(SleepState::Drowsy);
                }
            }
            SleepState::Drowsy => {
                // Can wake up if external stimulation happens (handled by Python)
                // Otherwise progress to light sleep after duration
                if self.ticks_in_state >= self.drowsy_duration {
                    self.transition(SleepState::Light);
                }
                // Emergency wake if somehow energy spikes (e.g. external boost)
                if energy > 0.9 && sleep_pressure < 0.3 {
                    self.transition(SleepState::Awake);
                }
            }
            SleepState::Light => {
                if self.ticks_in_state >= self.light_duration {
                    self.transition(SleepState::Deep);
                }
            }
            SleepState::Deep => {
                if self.ticks_in_state >= self.deep_duration {
                    self.transition(SleepState::Rem);
                    self.rem_episodes += 1;
                }
            }
            SleepState::Rem => {
                if self.ticks_in_state >= self.rem_duration {
                    // Check if we should wake or cycle back to light
                    if energy > self.wake_energy_threshold
                        && sleep_pressure < self.wake_pressure_threshold
                    {
                        self.transition(SleepState::Awake);
                    } else {
                        // Another sleep cycle
                        self.cycles_completed += 1;
                        self.transition(SleepState::Light);
                    }
                }
            }
        }

        old_state != self.state
    }

    fn transition(&mut self, new_state: SleepState) {
        self.state = new_state;
        self.ticks_in_state = 0;
    }

    /// Is the brain currently asleep (any non-Awake state)?
    pub fn is_asleep(&self) -> bool {
        self.state != SleepState::Awake
    }

    /// Is the brain in REM? (Python should do dream replay during this.)
    pub fn in_rem(&self) -> bool {
        self.state == SleepState::Rem
    }

    /// Force wake-up (e.g. strong external stimulus).
    pub fn force_wake(&mut self) {
        if self.state != SleepState::Awake {
            self.transition(SleepState::Awake);
            self.rem_episodes = 0;
        }
    }

    /// Get summary: (state_name, ticks_in_state, cycles_completed, rem_episodes).
    pub fn summary(&self) -> (&str, u64, u32, u32) {
        (
            self.state.name(),
            self.ticks_in_state,
            self.cycles_completed,
            self.rem_episodes,
        )
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        let sc = SleepCycleManager::new();
        assert_eq!(sc.state, SleepState::Awake);
        assert_eq!(sc.ticks_in_state, 0);
        assert_eq!(sc.cycles_completed, 0);
    }

    #[test]
    fn test_input_gates() {
        assert!((SleepState::Awake.input_gate() - 1.0).abs() < 1e-6);
        assert!((SleepState::Drowsy.input_gate() - 0.5).abs() < 1e-6);
        assert!((SleepState::Light.input_gate() - 0.2).abs() < 1e-6);
        assert!((SleepState::Deep.input_gate() - 0.05).abs() < 1e-6);
        assert!((SleepState::Rem.input_gate() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_energy_recovery_rates() {
        assert!((SleepState::Awake.energy_recovery_rate()).abs() < 1e-6);
        assert!(SleepState::Deep.energy_recovery_rate() > SleepState::Light.energy_recovery_rate());
        assert!(SleepState::Light.energy_recovery_rate() > SleepState::Drowsy.energy_recovery_rate());
    }

    #[test]
    fn test_awake_to_drowsy_on_pressure() {
        let mut sc = SleepCycleManager::new();
        let changed = sc.update(1.0, 0.8); // high pressure
        assert!(changed);
        assert_eq!(sc.state, SleepState::Drowsy);
    }

    #[test]
    fn test_awake_to_drowsy_on_low_energy() {
        let mut sc = SleepCycleManager::new();
        let changed = sc.update(0.1, 0.0); // very low energy
        assert!(changed);
        assert_eq!(sc.state, SleepState::Drowsy);
    }

    #[test]
    fn test_stays_awake_when_fine() {
        let mut sc = SleepCycleManager::new();
        for _ in 0..100 {
            let changed = sc.update(0.8, 0.3);
            assert!(!changed);
        }
        assert_eq!(sc.state, SleepState::Awake);
    }

    #[test]
    fn test_drowsy_to_light() {
        let mut sc = SleepCycleManager::new();
        sc.update(0.1, 0.8); // → Drowsy
        assert_eq!(sc.state, SleepState::Drowsy);

        // Run through drowsy duration
        for _ in 0..sc.drowsy_duration {
            sc.update(0.3, 0.6);
        }
        assert_eq!(sc.state, SleepState::Light);
    }

    #[test]
    fn test_full_sleep_cycle() {
        let mut sc = SleepCycleManager::new();
        // Enter sleep
        sc.update(0.1, 0.8); // → Drowsy
        assert_eq!(sc.state, SleepState::Drowsy);

        // Drowsy → Light
        for _ in 0..sc.drowsy_duration {
            sc.update(0.3, 0.6);
        }
        assert_eq!(sc.state, SleepState::Light);

        // Light → Deep
        for _ in 0..sc.light_duration {
            sc.update(0.3, 0.5);
        }
        assert_eq!(sc.state, SleepState::Deep);

        // Deep → REM
        for _ in 0..sc.deep_duration {
            sc.update(0.4, 0.4);
        }
        assert_eq!(sc.state, SleepState::Rem);
        assert_eq!(sc.rem_episodes, 1);
    }

    #[test]
    fn test_rem_to_wake() {
        let mut sc = SleepCycleManager::new();
        sc.state = SleepState::Rem;
        sc.ticks_in_state = 0;

        // Run through REM with recovered energy + low pressure → wake
        for _ in 0..sc.rem_duration {
            sc.update(0.9, 0.05);
        }
        assert_eq!(sc.state, SleepState::Awake);
    }

    #[test]
    fn test_rem_cycles_back_to_light() {
        let mut sc = SleepCycleManager::new();
        sc.state = SleepState::Rem;
        sc.ticks_in_state = 0;

        // Run through REM with low energy + pressure → cycles back
        for _ in 0..sc.rem_duration {
            sc.update(0.3, 0.5);
        }
        assert_eq!(sc.state, SleepState::Light);
        assert_eq!(sc.cycles_completed, 1);
    }

    #[test]
    fn test_force_wake() {
        let mut sc = SleepCycleManager::new();
        sc.state = SleepState::Deep;
        sc.ticks_in_state = 100;
        sc.force_wake();
        assert_eq!(sc.state, SleepState::Awake);
        assert_eq!(sc.ticks_in_state, 0);
    }

    #[test]
    fn test_is_asleep() {
        let mut sc = SleepCycleManager::new();
        assert!(!sc.is_asleep());
        sc.state = SleepState::Light;
        assert!(sc.is_asleep());
    }

    #[test]
    fn test_in_rem() {
        let mut sc = SleepCycleManager::new();
        assert!(!sc.in_rem());
        sc.state = SleepState::Rem;
        assert!(sc.in_rem());
    }

    #[test]
    fn test_drowsy_emergency_wake() {
        let mut sc = SleepCycleManager::new();
        sc.state = SleepState::Drowsy;
        sc.ticks_in_state = 100;
        let changed = sc.update(0.95, 0.2); // high energy, low pressure
        assert!(changed);
        assert_eq!(sc.state, SleepState::Awake);
    }

    #[test]
    fn test_summary() {
        let sc = SleepCycleManager::new();
        let (name, ticks, cycles, rem) = sc.summary();
        assert_eq!(name, "awake");
        assert_eq!(ticks, 0);
        assert_eq!(cycles, 0);
        assert_eq!(rem, 0);
    }

    #[test]
    fn test_reset() {
        let mut sc = SleepCycleManager::new();
        sc.state = SleepState::Deep;
        sc.cycles_completed = 5;
        sc.rem_episodes = 3;
        sc.reset();
        assert_eq!(sc.state, SleepState::Awake);
        assert_eq!(sc.cycles_completed, 0);
        assert_eq!(sc.rem_episodes, 0);
    }

    #[test]
    fn test_multiple_full_cycles() {
        let mut sc = SleepCycleManager::new();
        // Force into sleep
        sc.update(0.05, 0.9); // → Drowsy
        assert_eq!(sc.state, SleepState::Drowsy);

        // Run 3 full cycles at low energy
        for _cycle in 0..3 {
            // Drowsy → Light
            while sc.state == SleepState::Drowsy {
                sc.update(0.2, 0.5);
            }
            // Light → Deep
            while sc.state == SleepState::Light {
                sc.update(0.3, 0.4);
            }
            // Deep → REM
            while sc.state == SleepState::Deep {
                sc.update(0.3, 0.3);
            }
            // REM → cycle back (low energy)
            while sc.state == SleepState::Rem {
                sc.update(0.3, 0.3);
            }
        }
        assert!(sc.cycles_completed >= 2, "Expected 2+ cycles, got {}", sc.cycles_completed);
        assert!(sc.rem_episodes >= 3, "Expected 3+ REM episodes, got {}", sc.rem_episodes);
    }
}
