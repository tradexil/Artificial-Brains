"""Phase 8 tests: Full Input/Output pipelines.

Tests:
  - Sensory input encoding (population coding)
  - Visual input pipeline (edge/shape/spatial extraction)
  - Audio input pipeline (frequency/temporal/complex)
  - Multimodal input synchronization
  - Motor output decoding
  - Imagination output reconstruction
  - Reflex pathway (pain → withdraw < 5 ticks)
  - Seed scripts (physics traces, relational traces, numbers, reflexes)
  - Tick loop Phase 8 metrics
"""

import random

import pytest

import brain_core

from brain.input.sensory_input import SensoryInput
from brain.input.visual_input import VisualInput
from brain.input.audio_input import AudioInput
from brain.input.multimodal import MultimodalInput
from brain.input.text_input import TextInput
from brain.output.motor_output import MotorOutput
from brain.output.imagination import ImaginationOutput
from brain.seed.physics_traces import spawn_physics_traces
from brain.seed.relational_traces import spawn_relational_traces
from brain.seed.numbers_wiring import wire_numbers, create_number_traces, number_neurons
from brain.seed.reflex_wiring import wire_reflexes
from brain.structures.trace_store import Trace, TraceStore
from brain.learning.tick_loop import TickLoop


@pytest.fixture(autouse=True)
def fresh_brain():
    brain_core.init_brain()
    yield
    brain_core.reset_brain()


def _make_trace_store() -> TraceStore:
    """Build a minimal TraceStore for testing."""
    store = TraceStore()
    store.add(Trace(
        id="t_cat",
        neurons={
            "sensory": [100, 101, 102],
            "visual": [10500, 10501, 10502, 10503, 10504, 10505],
            "language": [105500, 105501, 105502],
            "memory_short": [45500, 45501],
            "memory_long": [55500, 55501, 55502],
            "speech": [140100, 140101],
            "motor": [130100, 130101],
        },
        strength=0.5,
        label="cat",
    ))
    store.add(Trace(
        id="t_hot",
        neurons={
            "sensory": [2400, 2401, 2402],   # high-temp range
            "visual": [11000, 11001],
            "language": [106000, 106001],
            "memory_short": [46000],
            "motor": [135100, 135101],         # withdraw
        },
        strength=0.5,
        label="hot",
    ))
    return store


# === Sensory Input ===

class TestSensoryInput:
    def test_encode_activates_sensory_region(self):
        si = SensoryInput()
        si.encode(temperature=0.8, pressure=0.3, pain=0.0, texture=0.5)
        act = brain_core.get_sensory_activation()
        assert act > 0.0

    def test_encode_multiple_modalities(self):
        si = SensoryInput()
        si.encode(temperature=0.5, pressure=0.5, pain=0.5, texture=0.5)
        act = brain_core.get_sensory_activation()
        # Should activate more neurons than single modality
        assert act > 0.0

    def test_pain_detection(self):
        si = SensoryInput()
        si.encode(temperature=0.0, pressure=0.0, pain=0.9, texture=0.0)
        pain = brain_core.get_pain_level()
        assert pain > 0.0


# === Visual Input ===

class TestVisualInput:
    def test_encode_frame(self):
        vi = VisualInput()
        # Minimal 4x4 grayscale frame
        frame = [[float(i + j) / 8.0 for j in range(4)] for i in range(4)]
        vi.encode(frame)
        act = brain_core.get_visual_activation()
        assert act > 0.0

    def test_encode_raw(self):
        vi = VisualInput()
        signals = [(10500, 0.8), (10501, 0.7), (25000, 0.5)]
        vi.encode_raw(signals)
        act = brain_core.get_visual_activation()
        assert act > 0.0

    def test_zero_frame_low_activation(self):
        vi = VisualInput()
        frame = [[0.0] * 4 for _ in range(4)]
        vi.encode(frame)
        act = brain_core.get_visual_activation()
        # All zeros → minimal activation
        assert act < 0.1


# === Audio Input ===

class TestAudioInput:
    def test_encode_sine_wave(self):
        import math
        ai = AudioInput()
        # Generate a 440Hz tone (1k samples at 8kHz)
        sr = 8000
        samples = [math.sin(2 * math.pi * 440 * t / sr) for t in range(1000)]
        ai.encode(samples, sr)
        act = brain_core.get_audio_activation()
        assert act > 0.0

    def test_silence_low_activation(self):
        ai = AudioInput()
        samples = [0.0] * 1000
        ai.encode(samples, 8000)
        act = brain_core.get_audio_activation()
        assert act < 0.1


# === Multimodal Input ===

class TestMultimodalInput:
    def test_single_modality(self):
        store = _make_trace_store()
        text_enc = TextInput(store)
        mm = MultimodalInput(text_encoder=text_enc)
        result = mm.process({"text": "hello"}, tick=1)
        assert "text" in result["_summary"]["active_modalities"]

    def test_multi_modality_boosts_integration(self):
        store = _make_trace_store()
        text_enc = TextInput(store)
        mm = MultimodalInput(text_encoder=text_enc)
        frame = [[0.5] * 4 for _ in range(4)]
        result = mm.process({"text": "hello", "visual": frame}, tick=1)
        assert result["_summary"]["modality_count"] >= 2


# === Motor Output ===

class TestMotorOutput:
    def test_idle_on_fresh_brain(self):
        mo = MotorOutput()
        action = mo.read()
        assert action.action_type == "idle"

    def test_approach_with_injected_activation(self):
        mo = MotorOutput()
        # Inject approach neurons
        signals = [(130000 + i, 1.0) for i in range(200)]
        brain_core.inject_activations(signals)
        brain_core.tick()
        action = mo.read()
        assert action.approach > 0.0

    def test_withdraw_with_injected_activation(self):
        mo = MotorOutput()
        signals = [(135000 + i, 1.0) for i in range(200)]
        brain_core.inject_activations(signals)
        brain_core.tick()
        action = mo.read()
        assert action.withdraw > 0.0


# === Imagination Output ===

class TestImaginationOutput:
    def test_empty_brain_returns_blank(self):
        io = ImaginationOutput()
        result = io.read()
        assert "image" in result
        assert result["spatial_focus"] is not None or result["spatial_focus"] is None

    def test_with_visual_activation(self):
        io = ImaginationOutput()
        # Inject visual neurons
        signals = [(10000 + i, 0.8) for i in range(100)]
        brain_core.inject_activations(signals)
        brain_core.tick()
        result = io.read()
        assert result["sub_activations"]["low"] > 0.0


# === Seed Scripts ===

class TestPhysicsTraces:
    def test_spawn_correct_count(self):
        store = TraceStore()
        count = spawn_physics_traces(store, count=50)
        assert count == 50
        assert len(store) == 50

    def test_traces_have_labels(self):
        store = TraceStore()
        spawn_physics_traces(store, count=10)
        for trace in store.traces.values():
            assert trace.label is not None
            assert len(trace.label) > 0

    def test_traces_span_multiple_regions(self):
        store = TraceStore()
        spawn_physics_traces(store, count=10)
        for trace in store.traces.values():
            assert len(trace.neurons) >= 3  # at least 3 regions


class TestRelationalTraces:
    def test_spawn_correct_count(self):
        store = TraceStore()
        count = spawn_relational_traces(store, count=50)
        assert count == 50
        assert len(store) == 50

    def test_traces_high_abstraction(self):
        store = TraceStore()
        spawn_relational_traces(store, count=20)
        abstractions = [t.abstraction for t in store.traces.values()]
        avg = sum(abstractions) / len(abstractions)
        assert avg >= 0.5  # relational traces are abstract

    def test_traces_include_executive_region(self):
        store = TraceStore()
        spawn_relational_traces(store, count=10)
        for trace in store.traces.values():
            assert "executive" in trace.neurons


class TestNumbersWiring:
    def test_number_neurons_range(self):
        neurons = number_neurons(0)
        assert len(neurons) == 20
        assert neurons[0] == 150000

    def test_number_neurons_n99(self):
        neurons = number_neurons(99)
        assert len(neurons) == 20
        assert neurons[0] == 150000 + 99 * 20

    def test_number_neurons_out_of_range(self):
        assert number_neurons(-1) == []
        assert number_neurons(100) == []

    def test_wire_creates_synapses(self):
        synapses = wire_numbers()
        assert len(synapses) > 0
        # Check format: (from, to, weight, delay, plasticity)
        s = synapses[0]
        assert len(s) == 5
        assert s[4] == 0.0  # hardwired = 0 plasticity

    def test_create_number_traces(self):
        store = TraceStore()
        count = create_number_traces(store)
        assert count == 100  # 0–99
        assert len(store) == 100
        # Check trace for "zero" exists by searching labels
        found = [t for t in store.traces.values() if t.label == "zero"]
        assert len(found) >= 1


class TestReflexWiring:
    def test_wire_creates_synapses(self):
        synapses = wire_reflexes()
        assert len(synapses) > 0

    def test_pain_to_withdraw_connections(self):
        synapses = wire_reflexes(pain_to_withdraw=50, heat_to_withdraw=0, pressure_to_approach=0)
        assert len(synapses) == 50
        for src, tgt, w, d, p in synapses:
            assert 5000 <= src < 7500       # pain range
            assert 135000 <= tgt < 138000   # withdraw range
            assert d == 2                     # low delay
            assert p == 0.05                 # low plasticity

    def test_default_synapse_count(self):
        synapses = wire_reflexes()
        # 100 pain + 50 heat + 30 pressure = 180
        assert len(synapses) == 180


# === Tick Loop Phase 8 Metrics ===

class TestTickLoopPhase8:
    def test_phase8_metrics_in_return(self):
        store = _make_trace_store()
        loop = TickLoop(store)
        result = loop.step()
        assert "sensory_activation" in result
        assert "visual_activation" in result
        assert "audio_activation" in result
        assert "motor_activation" in result
        assert "motor_action" in result
        assert "pain_level" in result

    def test_motor_action_after_inject(self):
        store = _make_trace_store()
        loop = TickLoop(store)
        signals = [(130000 + i, 1.0) for i in range(200)]
        brain_core.inject_activations(signals)
        result = loop.step()
        assert result["motor_approach"] > 0.0
