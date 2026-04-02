/// Brain: top-level struct that holds all regions, synapse pool,
/// delay buffer, attention system, prediction state, binding store,
/// and orchestrates the tick cycle.

use crate::core::attention::AttentionSystem;
use crate::core::activity::ActivityCache;
use crate::core::binding::{BindingStore, PatternRef};
use crate::core::homeostasis::HomeostasisSystem;
use crate::core::neuromodulator::NeuromodulatorSystem;
use crate::core::propagate::DelayBuffer;
use crate::core::region::{Region, RegionId};
use crate::core::sleep::SleepCycleManager;
use crate::core::synapse::{SynapseData, SynapsePool};
use crate::core::tick::{self, TickResult};
use crate::regions::emotion as region_emotion;
use crate::regions::executive as region_executive;
use crate::regions::language as region_language;
use crate::regions::speech as region_speech;
use crate::regions::sensory as region_sensory;
use crate::regions::visual as region_visual;
use crate::regions::audio as region_audio;
use crate::regions::motor as region_motor;
use crate::regions::pattern::PredictionState;
use std::collections::HashMap;

pub struct Brain {
    pub regions: Vec<Region>,
    pub synapse_pool: SynapsePool,
    pub delay_buffer: DelayBuffer,
    pub attention_system: AttentionSystem,
    pub prediction_state: PredictionState,
    pub binding_store: BindingStore,
    pub neuromodulator: NeuromodulatorSystem,
    pub homeostasis: HomeostasisSystem,
    pub sleep_cycle: SleepCycleManager,
    pub activity_cache: ActivityCache,
    pub tick_count: u64,
}

impl Brain {
    /// Create a new brain with all 14 regions and an empty synapse pool.
    pub fn new() -> Self {
        let regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();

        Brain {
            regions,
            synapse_pool: SynapsePool::new(152_000),
            delay_buffer: DelayBuffer::new(),
            attention_system: AttentionSystem::new(15, 0.1, 5.0, 0.4, 0.4, 0.2),
            prediction_state: PredictionState::new(0.05),
            binding_store: BindingStore::new(),
            neuromodulator: NeuromodulatorSystem::new(),
            homeostasis: HomeostasisSystem::new(),
            sleep_cycle: SleepCycleManager::new(),
            activity_cache: ActivityCache::new(),
            tick_count: 0,
        }
    }

    /// Create a brain with a pre-built synapse pool.
    pub fn with_synapses(synapses: Vec<SynapseData>) -> Self {
        let regions: Vec<Region> = RegionId::ALL.iter().map(|&id| Region::new(id)).collect();

        Brain {
            regions,
            synapse_pool: SynapsePool::from_synapses(152_000, synapses),
            delay_buffer: DelayBuffer::new(),
            attention_system: AttentionSystem::new(15, 0.1, 5.0, 0.4, 0.4, 0.2),
            prediction_state: PredictionState::new(0.05),
            binding_store: BindingStore::new(),
            neuromodulator: NeuromodulatorSystem::new(),
            homeostasis: HomeostasisSystem::new(),
            sleep_cycle: SleepCycleManager::new(),
            activity_cache: ActivityCache::new(),
            tick_count: 0,
        }
    }

    /// Advance one tick.
    ///
    /// Phase 4: Compute drives, update gains, run tick, compute predictions.
    /// Phase 6: Compute emotion state, update neuromodulators, executive conflict.
    /// Phase 9: Homeostatic regulation, sleep cycle update, input gating.
    pub fn tick(&mut self) -> TickResult {
        // Phase 9: advance circadian clock
        self.homeostasis.advance_circadian();

        // Phase 9: sleep cycle state machine update
        self.sleep_cycle.update(
            self.neuromodulator.energy,
            self.homeostasis.sleep_pressure,
        );

        // Phase 9: homeostatic regulation of neuromodulators
        let (ad, vd, fd) = self.homeostasis.regulate(
            self.neuromodulator.arousal,
            self.neuromodulator.valence,
            self.neuromodulator.focus,
        );
        self.neuromodulator.arousal = (self.neuromodulator.arousal + ad).clamp(0.0, 1.0);
        self.neuromodulator.valence = (self.neuromodulator.valence + vd).clamp(-1.0, 1.0);
        self.neuromodulator.focus = (self.neuromodulator.focus + fd).clamp(0.0, 1.0);

        // Phase 9: accumulate or dissipate sleep pressure
        if self.sleep_cycle.is_asleep() {
            self.homeostasis.dissipate_pressure();
            // Energy recovery during sleep
            let recovery = self.sleep_cycle.state.energy_recovery_rate();
            self.neuromodulator.recover_energy(recovery);
        } else {
            self.homeostasis.accumulate_pressure();
        }

        // Phase 4: auto-compute threat and relevance drives from brain activity
        let threat = (self.activity_cache.active_rate(RegionId::Emotion) * 10.0).min(1.0);
        let relevance = (self.activity_cache.active_rate(RegionId::Executive) * 10.0).min(1.0);
        for &rid in RegionId::ALL.iter() {
            let (existing_novelty, _, _) = self.attention_system.drives_for(rid);
            self.attention_system.set_drives(rid, existing_novelty, threat, relevance);
        }

        // Phase 6: compute emotion state and update neuromodulators
        let emotion_arousal = (self.activity_cache.active_rate(RegionId::Emotion) / 0.10).min(1.0);
        let emotion_polarity = region_emotion::compute_polarity(&self.regions, 0.01);
        self.neuromodulator.update_arousal_from_emotion(emotion_arousal);
        self.neuromodulator.update_valence_from_emotion(emotion_polarity);

        // Update attention gains with inertia
        self.attention_system.update_gains();

        // Phase 9: apply sleep input gating to attention gains
        let sleep_gate = self.sleep_cycle.state.input_gate();
        if sleep_gate < 1.0 {
            // Gate input regions: sensory, visual, audio
            for &rid in &[RegionId::Sensory, RegionId::Visual, RegionId::Audio] {
                let current = self.attention_system.gain_for(rid);
                self.attention_system.set_gain_direct(rid, current * sleep_gate);
            }
        }

        // Run tick cycle with current attention gains
        let result = tick::tick(
            &mut self.regions,
            &self.synapse_pool,
            &mut self.delay_buffer,
            self.attention_system.gains(),
            self.tick_count,
        );
        self.activity_cache.update_from_tick_result(&result, &self.regions);

        // Phase 6: deplete energy based on activity (modulated by circadian)
        let circadian_mod = self.homeostasis.circadian_energy_modifier();
        let effective_active = (result.total_active as f32 * circadian_mod) as u32;
        self.neuromodulator.deplete_energy(effective_active);

        // Phase 6: executive conflict resolution (if conflict detected)
        let conflict = region_executive::detect_motor_conflict(&self.regions, 0.01);
        if conflict > 0.3 {
            region_executive::resolve_motor_conflict(&mut self.regions, 0.01, 0.5);
            if let Some(region) = self.region(RegionId::Motor) {
                let active_count = region.active_count(0.5);
                self.activity_cache
                    .refresh_region_count(RegionId::Motor, region.neurons.count, active_count);
            }
        }

        // Phase 6: impulse suppression — executive blocks emotion→motor impulse
        let impulses = region_emotion::emotion_motor_impulse(&self.regions, 0.01);
        if !impulses.is_empty() {
            region_executive::inhibit_motor_neurons(&mut self.regions, 0.01, &impulses);
            if let Some(region) = self.region(RegionId::Motor) {
                let active_count = region.active_count(0.5);
                self.activity_cache
                    .refresh_region_count(RegionId::Motor, region.neurons.count, active_count);
            }
        }

        // Compute prediction errors from post-tick activations
        self.prediction_state.update_from_cache(&self.activity_cache);

        self.tick_count += 1;
        result
    }

    /// Inject external signals into neurons (by global ID).
    /// Signals: list of (global_neuron_id, activation_value).
    pub fn inject(&mut self, signals: &[(u32, f32)]) {
        for &(global_id, value) in signals {
            for region in &mut self.regions {
                if let Some(local) = region.global_to_local(global_id) {
                    region.neurons.inject(&[(local, value)]);
                    break;
                }
            }
        }
    }

    /// Set attention gain for a specific region (backward compat).
    pub fn set_attention_gain(&mut self, region: RegionId, gain: f32) {
        self.attention_system.set_gain_direct(region, gain);
    }

    /// Set attention drives for a region.
    pub fn set_attention_drives(&mut self, region: RegionId, novelty: f32, threat: f32, relevance: f32) {
        self.attention_system.set_drives(region, novelty, threat, relevance);
    }

    /// Get attention gains for all regions.
    pub fn get_attention_gains(&self) -> HashMap<RegionId, f32> {
        self.attention_system.gains().clone()
    }

    /// Get prediction errors from last tick.
    pub fn get_prediction_errors(&self) -> HashMap<RegionId, f32> {
        self.prediction_state.last_errors().clone()
    }

    /// Get global prediction error (mean across regions).
    pub fn get_global_prediction_error(&self) -> f32 {
        self.prediction_state.global_error()
    }

    /// Get activations from a specific region.
    /// Returns (global_neuron_id, activation) for neurons above `min_activation`.
    pub fn get_activations(&self, region_id: RegionId, min_activation: f32) -> Vec<(u32, f32)> {
        for region in &self.regions {
            if region.id == region_id {
                let mut result = Vec::new();
                for i in 0..region.neurons.count as usize {
                    if region.neurons.activations[i] > min_activation {
                        result.push((region.local_to_global(i as u32), region.neurons.activations[i]));
                    }
                }
                return result;
            }
        }
        Vec::new()
    }

    /// Get all activations across all regions above threshold.
    pub fn get_all_activations(&self, min_activation: f32) -> HashMap<RegionId, Vec<(u32, f32)>> {
        let mut all = HashMap::new();
        for region in &self.regions {
            let acts: Vec<(u32, f32)> = (0..region.neurons.count as usize)
                .filter(|&i| region.neurons.activations[i] > min_activation)
                .map(|i| (region.local_to_global(i as u32), region.neurons.activations[i]))
                .collect();
            if !acts.is_empty() {
                all.insert(region.id, acts);
            }
        }
        all
    }

    /// Get a specific neuron's potential (by global ID).
    pub fn get_neuron_potential(&self, global_id: u32) -> Option<f32> {
        for region in &self.regions {
            if let Some(local) = region.global_to_local(global_id) {
                return Some(region.neurons.potentials[local as usize]);
            }
        }
        None
    }

    /// Get region by ID.
    pub fn region(&self, id: RegionId) -> Option<&Region> {
        self.regions.iter().find(|r| r.id == id)
    }

    /// Get mutable region by ID.
    pub fn region_mut(&mut self, id: RegionId) -> Option<&mut Region> {
        self.regions.iter_mut().find(|r| r.id == id)
    }

    /// Queue a synapse weight update.
    pub fn update_synapse(&mut self, from: u32, to: u32, delta: f32) {
        self.synapse_pool.queue_update(from, to, delta);
    }

    /// Queue a new synapse.
    pub fn create_synapse(&mut self, from: u32, to: u32, weight: f32, delay: u8, plasticity: f32) {
        self.synapse_pool.queue_create(SynapseData {
            from, to, weight, delay, plasticity,
        });
    }

    /// Queue a synapse for removal.
    pub fn prune_synapse(&mut self, from: u32, to: u32) {
        self.synapse_pool.queue_prune(from, to);
    }

    /// Apply pending weight updates (cheap, do frequently).
    pub fn apply_synapse_updates(&mut self) {
        self.synapse_pool.apply_weight_updates();
    }

    /// Full rebuild of synapse CSR (expensive, do periodically).
    pub fn rebuild_synapses(&mut self) {
        // Build a closure that looks up neuron type by global ID
        let regions = &self.regions;
        self.synapse_pool.rebuild(&|global_id: u32| {
            for region in regions {
                if let Some(local) = region.global_to_local(global_id) {
                    return region.neurons.neuron_types[local as usize];
                }
            }
            crate::core::neuron::NeuronType::Excitatory // fallback
        });
    }

    /// Get total neuron count across all regions.
    pub fn neuron_count(&self) -> u32 {
        self.regions.iter().map(|r| r.neurons.count).sum()
    }

    /// Get synapse pool stats.
    pub fn synapse_count(&self) -> u64 {
        self.synapse_pool.total_count
    }

    /// Get firing rate for a region (fraction of neurons active).
    pub fn firing_rate(&self, region_id: RegionId) -> f32 {
        self.activity_cache.active_rate(region_id)
    }

    pub fn active_count(&self, region_id: RegionId) -> u32 {
        self.activity_cache.active_count(region_id)
    }

    pub fn cached_emotion_arousal(&self) -> f32 {
        (self.activity_cache.active_rate(RegionId::Emotion) / 0.10).min(1.0)
    }

    pub fn cached_executive_engagement(&self) -> f32 {
        (self.activity_cache.active_rate(RegionId::Executive) / 0.05).min(1.0)
    }

    pub fn cached_language_activation(&self) -> f32 {
        (self.activity_cache.active_rate(RegionId::Language) / 0.05).min(1.0)
    }

    pub fn cached_speech_activity(&self) -> f32 {
        (self.activity_cache.active_rate(RegionId::Speech) / 0.05).min(1.0)
    }

    pub fn cached_sensory_activation(&self) -> f32 {
        (self.activity_cache.active_rate(RegionId::Sensory) / 0.05).min(1.0)
    }

    pub fn cached_visual_activation(&self) -> f32 {
        (self.activity_cache.active_rate(RegionId::Visual) / 0.05).min(1.0)
    }

    pub fn cached_audio_activation(&self) -> f32 {
        (self.activity_cache.active_rate(RegionId::Audio) / 0.05).min(1.0)
    }

    pub fn cached_motor_activation(&self) -> f32 {
        (self.activity_cache.active_rate(RegionId::Motor) / 0.05).min(1.0)
    }

    pub fn cached_planning_signal(&self) -> f32 {
        (self.cached_executive_engagement() * self.cached_language_activation()).sqrt()
    }

    // === BINDING OPERATIONS ===

    /// Create a new binding between two patterns.
    pub fn create_binding(
        &mut self,
        region_a: RegionId, neurons_a: Vec<u32>, threshold_a: f32,
        region_b: RegionId, neurons_b: Vec<u32>, threshold_b: f32,
        time_delta: f32,
    ) -> u32 {
        let pa = PatternRef::new(region_a, neurons_a, threshold_a);
        let pb = PatternRef::new(region_b, neurons_b, threshold_b);
        self.binding_store.add(pa, pb, time_delta)
    }

    /// Evaluate all bindings against current active neurons.
    /// Returns (binding_id, weight) for bindings where BOTH patterns active.
    pub fn evaluate_bindings(&self, min_activation: f32) -> Vec<(u32, f32)> {
        let active_set = self.active_neuron_set(min_activation);
        self.binding_store.evaluate(&active_set)
    }

    /// Find bindings where only one pattern is active.
    pub fn find_partial_bindings(&self, min_activation: f32) -> Vec<u32> {
        let active_set = self.active_neuron_set(min_activation);
        self.binding_store.find_partial(&active_set)
    }

    /// Strengthen a binding.
    pub fn strengthen_binding(&mut self, binding_id: u32, tick: u64) {
        self.binding_store.strengthen(binding_id, tick);
    }

    /// Record a missed opportunity for a binding.
    pub fn record_binding_miss(&mut self, binding_id: u32) {
        self.binding_store.record_miss(binding_id);
    }

    /// Prune dissolved bindings.
    pub fn prune_bindings(&mut self, weight_threshold: f32, min_fires: u32) -> u32 {
        self.binding_store.prune(weight_threshold, min_fires)
    }

    /// Get binding count.
    pub fn binding_count(&self) -> usize {
        self.binding_store.len()
    }

    /// Get binding info: (weight, fires, confidence, last_fired).
    pub fn get_binding_info(&self, binding_id: u32) -> Option<(f32, u32, f32, u64)> {
        self.binding_store.get(binding_id).map(|b| {
            (b.weight, b.fires, b.confidence, b.last_fired)
        })
    }

    // === MEMORY OPERATIONS ===

    /// Pattern completion in memory_long: if enough trace neurons are active, boost the rest.
    pub fn pattern_complete(&mut self, trace_neurons: &[u32], threshold: f32, boost: f32) -> u32 {
        crate::regions::memory_long::pattern_completion(
            &mut self.regions, trace_neurons, threshold, boost,
        )
    }

    /// Strengthen memory_long trace neurons (for consolidation).
    pub fn strengthen_memory_trace(&mut self, trace_neurons: &[u32], boost: f32) -> u32 {
        crate::regions::memory_long::strengthen_trace_neurons(
            &mut self.regions, trace_neurons, boost,
        )
    }

    /// Boost working memory neurons for active traces.
    pub fn boost_working_memory(&mut self, trace_neurons: &[u32], boost: f32) -> u32 {
        crate::regions::memory_short::boost_working_memory_neurons(
            &mut self.regions, trace_neurons, boost,
        )
    }

    /// Suppress non-trace neurons in working memory (lateral inhibition).
    pub fn suppress_working_memory(&mut self, active_trace_neurons: &std::collections::HashSet<u32>, suppression: f32) -> u32 {
        crate::regions::memory_short::suppress_non_trace_neurons(
            &mut self.regions, active_trace_neurons, suppression,
        )
    }

    /// Count active input regions for integration.
    pub fn integration_input_count(&self, min_activation: f32) -> u32 {
        let _ = min_activation;
        [
            RegionId::Sensory,
            RegionId::Visual,
            RegionId::Audio,
            RegionId::Pattern,
            RegionId::Emotion,
            RegionId::Language,
        ]
        .into_iter()
        .filter(|&region_id| self.activity_cache.active_count(region_id) > 0)
        .count() as u32
    }

    /// Boost integration region based on multi-modal convergence.
    pub fn boost_integration(&mut self, strength: f32, max_neurons: usize) -> u32 {
        crate::regions::integration::boost_integration_neurons(
            &mut self.regions, strength, max_neurons,
        )
    }

    // === EMOTION & EXECUTIVE OPERATIONS (Phase 6) ===

    /// Set neuromodulator state directly.
    pub fn set_neuromodulator(&mut self, arousal: f32, valence: f32, focus: f32, energy: f32) {
        self.neuromodulator.set(arousal, valence, focus, energy);
    }

    /// Get neuromodulator state.
    pub fn get_neuromodulator(&self) -> (f32, f32, f32, f32) {
        self.neuromodulator.get()
    }

    /// Get threshold modifier from current arousal level.
    pub fn neuromod_threshold_modifier(&self) -> f32 {
        self.neuromodulator.threshold_modifier()
    }

    /// Get emotion polarity from current region state.
    pub fn emotion_polarity(&self) -> f32 {
        region_emotion::compute_polarity(&self.regions, 0.01)
    }

    /// Get emotion arousal from current region state.
    pub fn emotion_arousal(&self) -> f32 {
        let active = self
            .region(RegionId::Emotion)
            .map(|region| region.active_count(0.01) as f32)
            .unwrap_or(0.0);
        (active / (RegionId::Emotion.neuron_count() as f32 * 0.10)).min(1.0)
    }

    /// Get emotion urgency.
    pub fn emotion_urgency(&self, urgency_threshold: f32) -> f32 {
        region_emotion::compute_urgency(&self.regions, 0.01, urgency_threshold)
    }

    /// Get emotion→motor impulses (before executive gating).
    pub fn emotion_motor_impulse(&self) -> Vec<(u32, f32)> {
        region_emotion::emotion_motor_impulse(&self.regions, 0.01)
    }

    /// Get executive engagement level (0.0–1.0).
    pub fn executive_engagement(&self) -> f32 {
        region_executive::executive_engagement(&self.regions, 0.01)
    }

    /// Detect motor conflict (0.0–1.0).
    pub fn motor_conflict(&self) -> f32 {
        region_executive::detect_motor_conflict(&self.regions, 0.01)
    }

    /// Executive resolves motor conflict by suppressing weaker motor population.
    pub fn resolve_motor_conflict(&mut self, suppress_strength: f32) -> u32 {
        region_executive::resolve_motor_conflict(&mut self.regions, 0.01, suppress_strength)
    }

    /// Executive inhibits specific motor neurons (impulse suppression).
    pub fn inhibit_motor(&mut self, impulse_neurons: &[(u32, f32)]) -> u32 {
        region_executive::inhibit_motor_neurons(&mut self.regions, 0.01, impulse_neurons)
    }

    /// Get planning signal (executive×language engagement).
    pub fn planning_signal(&self) -> f32 {
        let exec = self.executive_engagement();
        let lang = self.language_activation();
        (exec * lang).sqrt()
    }

    /// Recover energy (e.g. during consolidation/rest).
    pub fn recover_energy(&mut self, amount: f32) {
        self.neuromodulator.recover_energy(amount);
    }

    // === LANGUAGE & SPEECH OPERATIONS (Phase 7) ===

    /// Get language region activation strength (0.0–1.0).
    pub fn language_activation(&self) -> f32 {
        region_language::language_activation_strength(&self.regions, 0.01)
    }

    /// Compute symbol overlap between active language neurons and trace neurons.
    pub fn symbol_overlap(&self, trace_lang_neurons: &[u32]) -> f32 {
        region_language::symbol_overlap(&self.regions, trace_lang_neurons, 0.01)
    }

    /// Get inner monologue signal (language↔executive loop strength).
    pub fn inner_monologue_signal(&self) -> f32 {
        region_language::inner_monologue_signal(&self.regions, 0.01)
    }

    /// Boost language neurons for token activation.
    pub fn boost_language(&mut self, neurons: &[u32], boost: f32) -> u32 {
        region_language::boost_language_neurons(&mut self.regions, neurons, boost)
    }

    /// Get top-K active token neurons in language region.
    pub fn peak_language_neurons(&self, top_k: usize) -> Vec<(u32, f32)> {
        region_language::peak_language_neurons(&self.regions, 0.01, top_k)
    }

    /// Get speech region activity level (0.0–1.0).
    pub fn speech_activity(&self) -> f32 {
        region_speech::speech_activity_level(&self.regions, 0.01)
    }

    /// Get top-K active speech neurons (for output decoding).
    pub fn peak_speech_neurons(&self, top_k: usize) -> Vec<(u32, f32)> {
        region_speech::peak_speech_neurons(&self.regions, 0.01, top_k)
    }

    /// Apply lateral inhibition in speech region.
    pub fn speech_lateral_inhibition(&mut self, suppression_factor: f32) -> u32 {
        region_speech::speech_lateral_inhibition(&mut self.regions, 0.01, suppression_factor)
    }

    /// Boost speech neurons for output generation.
    pub fn boost_speech(&mut self, neurons: &[u32], boost: f32) -> u32 {
        region_speech::boost_speech_neurons(&mut self.regions, neurons, boost)
    }

    // === SENSORY OPERATIONS (Phase 8) ===

    /// Encode sensory values into population-coded neuron activations.
    pub fn encode_sensory(&self, temperature: f32, pressure: f32, pain: f32, texture: f32, spread: u32) -> Vec<(u32, f32)> {
        region_sensory::encode_sensory(temperature, pressure, pain, texture, spread)
    }

    /// Get sensory activation strength (0.0–1.0).
    pub fn sensory_activation(&self) -> f32 {
        region_sensory::sensory_activation_strength(&self.regions, 0.01)
    }

    /// Boost sensory neurons. Returns count boosted.
    pub fn boost_sensory(&mut self, neurons: &[u32], boost: f32) -> u32 {
        region_sensory::boost_sensory_neurons(&mut self.regions, neurons, boost)
    }

    /// Get top-K active sensory neurons.
    pub fn peak_sensory_neurons(&self, top_k: usize) -> Vec<(u32, f32)> {
        region_sensory::peak_sensory_neurons(&self.regions, 0.01, top_k)
    }

    /// Detect pain level from sensory region (0.0–1.0).
    pub fn detect_pain(&self) -> f32 {
        region_sensory::detect_pain_level(&self.regions, 0.01)
    }

    // === VISUAL OPERATIONS (Phase 8) ===

    /// Get visual activation strength (0.0–1.0).
    pub fn visual_activation(&self) -> f32 {
        region_visual::visual_activation_strength(&self.regions, 0.01)
    }

    /// Boost visual neurons. Returns count boosted.
    pub fn boost_visual(&mut self, neurons: &[u32], boost: f32) -> u32 {
        region_visual::boost_visual_neurons(&mut self.regions, neurons, boost)
    }

    /// Get top-K active visual neurons.
    pub fn peak_visual_neurons(&self, top_k: usize, sub_region: Option<&str>) -> Vec<(u32, f32)> {
        let sub = sub_region.and_then(|s| match s {
            "low" => Some(region_visual::VisualSubRegion::Low),
            "mid" => Some(region_visual::VisualSubRegion::Mid),
            "high" => Some(region_visual::VisualSubRegion::High),
            "spatial" => Some(region_visual::VisualSubRegion::Spatial),
            _ => None,
        });
        region_visual::peak_visual_neurons(&self.regions, 0.01, top_k, sub)
    }

    /// Read all visual activations (for imagination output).
    pub fn read_visual_activations(&self) -> Vec<(u32, f32)> {
        region_visual::read_visual_activations(&self.regions, 0.01)
    }

    // === AUDIO OPERATIONS (Phase 8) ===

    /// Get audio activation strength (0.0–1.0).
    pub fn audio_activation(&self) -> f32 {
        region_audio::audio_activation_strength(&self.regions, 0.01)
    }

    /// Boost audio neurons. Returns count boosted.
    pub fn boost_audio(&mut self, neurons: &[u32], boost: f32) -> u32 {
        region_audio::boost_audio_neurons(&mut self.regions, neurons, boost)
    }

    /// Get top-K active audio neurons.
    pub fn peak_audio_neurons(&self, top_k: usize) -> Vec<(u32, f32)> {
        region_audio::peak_audio_neurons(&self.regions, 0.01, top_k)
    }

    /// Map a frequency (Hz) to audio neurons.
    pub fn frequency_to_neurons(&self, freq_hz: f32, spread: u32) -> Vec<(u32, f32)> {
        region_audio::frequency_to_neurons(freq_hz, spread)
    }

    // === MOTOR OPERATIONS (Phase 8) ===

    /// Get motor activation strength (0.0–1.0).
    pub fn motor_activation(&self) -> f32 {
        region_motor::motor_activation_strength(&self.regions, 0.01)
    }

    /// Get approach vs withdraw strengths.
    pub fn approach_vs_withdraw(&self) -> (f32, f32) {
        region_motor::approach_vs_withdraw(&self.regions, 0.01)
    }

    /// Decode motor action from current state.
    pub fn decode_motor_action(&self) -> region_motor::MotorAction {
        region_motor::decode_motor_action(&self.regions, 0.01)
    }

    /// Get top-K active motor neurons.
    pub fn peak_motor_neurons(&self, top_k: usize) -> Vec<(u32, f32)> {
        region_motor::peak_motor_neurons(&self.regions, 0.01, top_k)
    }

    /// Apply motor lateral inhibition.
    pub fn motor_lateral_inhibition(&mut self, suppression_factor: f32) -> u32 {
        region_motor::motor_lateral_inhibition(&mut self.regions, 0.01, suppression_factor)
    }

    /// Boost motor neurons. Returns count boosted.
    pub fn boost_motor(&mut self, neurons: &[u32], boost: f32) -> u32 {
        region_motor::boost_motor_neurons(&mut self.regions, neurons, boost)
    }

    /// Get a set of all active neuron global IDs above threshold.
    fn active_neuron_set(&self, min_activation: f32) -> std::collections::HashSet<u32> {
        let mut set = std::collections::HashSet::new();
        for region in &self.regions {
            for i in 0..region.neurons.count as usize {
                if region.neurons.activations[i] > min_activation {
                    set.insert(region.local_to_global(i as u32));
                }
            }
        }
        set
    }

    /// Reset the entire brain state (keeps structure, clears dynamics).
    pub fn reset(&mut self) {
        for region in &mut self.regions {
            region.neurons.reset();
            region.incoming.fill(0.0);
        }
        self.delay_buffer = DelayBuffer::new();
        self.attention_system = AttentionSystem::new(15, 0.1, 5.0, 0.4, 0.4, 0.2);
        self.prediction_state = PredictionState::new(0.05);
        self.binding_store = BindingStore::new();
        self.neuromodulator.reset();
        self.homeostasis.reset();
        self.sleep_cycle.reset();
        self.activity_cache = ActivityCache::new();
        self.tick_count = 0;
    }

    // === Phase 9: Homeostasis & Sleep ===

    /// Get homeostasis summary: (sleep_pressure, circadian_phase, ticks_awake, ticks_asleep).
    pub fn homeostasis_summary(&self) -> (f32, f32, u64, u64) {
        self.homeostasis.summary()
    }

    /// Get sleep state: (state_name, ticks_in_state, cycles_completed, rem_episodes).
    pub fn sleep_summary(&self) -> (&str, u64, u32, u32) {
        self.sleep_cycle.summary()
    }

    /// Is the brain currently asleep?
    pub fn is_asleep(&self) -> bool {
        self.sleep_cycle.is_asleep()
    }

    /// Is the brain in REM sleep? (Python should perform dream replay.)
    pub fn in_rem(&self) -> bool {
        self.sleep_cycle.in_rem()
    }

    /// Force wake-up (e.g. strong external stimulus).
    pub fn force_wake(&mut self) {
        self.sleep_cycle.force_wake();
    }

    /// Get current sleep input gate multiplier.
    pub fn sleep_input_gate(&self) -> f32 {
        self.sleep_cycle.state.input_gate()
    }

    /// Get current sleep pressure.
    pub fn sleep_pressure(&self) -> f32 {
        self.homeostasis.sleep_pressure
    }

    /// Get current circadian phase (0.0 to 1.0).
    pub fn circadian_phase(&self) -> f32 {
        self.homeostasis.circadian_phase
    }

    /// Set homeostasis parameters.
    pub fn set_homeostasis_params(
        &mut self,
        arousal_reg_rate: f32,
        valence_reg_rate: f32,
        focus_reg_rate: f32,
        sleep_pressure_rate: f32,
        sleep_dissipation_rate: f32,
    ) {
        self.homeostasis.arousal_reg_rate = arousal_reg_rate;
        self.homeostasis.valence_reg_rate = valence_reg_rate;
        self.homeostasis.focus_reg_rate = focus_reg_rate;
        self.homeostasis.sleep_pressure_rate = sleep_pressure_rate;
        self.homeostasis.sleep_dissipation_rate = sleep_dissipation_rate;
    }

    /// Set sleep cycle durations.
    pub fn set_sleep_durations(
        &mut self,
        drowsy: u64,
        light: u64,
        deep: u64,
        rem: u64,
    ) {
        self.sleep_cycle.drowsy_duration = drowsy;
        self.sleep_cycle.light_duration = light;
        self.sleep_cycle.deep_duration = deep;
        self.sleep_cycle.rem_duration = rem;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_creation() {
        let brain = Brain::new();
        assert_eq!(brain.regions.len(), 14);
        assert_eq!(brain.tick_count, 0);

        let total_neurons: u32 = brain.regions.iter().map(|r| r.neurons.count).sum();
        assert_eq!(total_neurons, 152_000);
    }

    #[test]
    fn test_brain_inject_and_tick() {
        let mut brain = Brain::new();

        // Inject into sensory (threshold 0.3)
        brain.inject(&[(0, 0.5), (1, 0.5), (2, 0.5)]);
        let result = brain.tick();

        assert_eq!(result.tick_number, 0);
        assert_eq!(brain.tick_count, 1);

        let sensory_acts = brain.get_activations(RegionId::Sensory, 0.5);
        assert!(!sensory_acts.is_empty(), "Expected sensory neurons to fire");
    }

    #[test]
    fn test_brain_cross_region_propagation() {
        // Create brain with a synapse from sensory→emotion
        let synapses = vec![SynapseData {
            from: 0,
            to: 70_000,
            weight: 0.8,
            delay: 1,
            plasticity: 1.0,
        }];
        let mut brain = Brain::with_synapses(synapses);

        // Tick 0: fire sensory neuron 0
        brain.inject(&[(0, 0.5)]);
        brain.tick();

        // Tick 1: signal should reach emotion
        brain.tick();

        let emotion = brain.region(RegionId::Emotion).unwrap();
        let has_activity = emotion.neurons.activations[0] > 0.0
            || emotion.neurons.potentials[0] > 0.0;
        assert!(has_activity, "Expected emotion neuron to receive signal");
    }

    #[test]
    fn test_brain_attention_gain() {
        let synapses = vec![SynapseData {
            from: 0,
            to: 70_000,
            weight: 0.4,
            delay: 1,
            plasticity: 1.0,
        }];
        let mut brain = Brain::with_synapses(synapses);

        // Boost emotion attention gain
        brain.set_attention_gain(RegionId::Emotion, 3.0);

        // Fire sensory → emotion with boosted gain
        brain.inject(&[(0, 0.5)]);
        brain.tick();
        brain.tick();

        let emotion = brain.region(RegionId::Emotion).unwrap();
        let potential = emotion.neurons.potentials[0];
        let activation = emotion.neurons.activations[0];

        // With 3x gain, signal = 1.0 * 0.4 * 3.0 = 1.2 → should fire
        assert!(
            activation > 0.0 || potential > 0.0,
            "Expected boosted signal to reach emotion. activation={}, potential={}",
            activation, potential
        );
    }

    #[test]
    fn test_brain_reset() {
        let mut brain = Brain::new();
        brain.inject(&[(0, 0.5)]);
        brain.tick();
        assert_eq!(brain.tick_count, 1);

        brain.reset();
        assert_eq!(brain.tick_count, 0);

        let acts = brain.get_all_activations(0.01);
        assert!(acts.is_empty(), "Expected no activity after reset");
    }

    #[test]
    fn test_multiple_ticks_stability() {
        let mut brain = Brain::new();

        // Run 100 ticks with no input — should stay silent
        for i in 0..100 {
            let result = brain.tick();
            assert_eq!(result.total_active, 0,
                "Expected no activity at tick {} without input", i);
        }
    }

    #[test]
    fn test_sparsity() {
        let mut brain = Brain::new();

        // Inject signal into 50 sensory neurons (0.5% of sensory region)
        let signals: Vec<(u32, f32)> = (0..50).map(|i| (i, 0.5)).collect();
        brain.inject(&signals);
        brain.tick();

        let rate = brain.firing_rate(RegionId::Sensory);
        assert!(rate < 0.05, "Sparsity violated: {:.1}% active", rate * 100.0);
    }
}
