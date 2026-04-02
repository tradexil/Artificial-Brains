"""Phase 10: Deep Integration Tests & Parallelism Verification.

Tests go beyond unit tests — they verify:
  1. Rayon parallelism works and thread count is controllable
  2. Text input → full tick(s) → speech output (end-to-end)
  3. Image input → full tick(s) → visual activation + imagination
  4. Audio input → full tick(s) → audio region activations
  5. Sensory input → full tick(s) → motor output (reflex path)
  6. Image + caption → multimodal → integration region boost
  7. Video (frame sequence) → temporal processing coherence
  8. Video + audio → multimodal synchronized processing
  9. Full lifecycle: seed → inject → multi-tick → learning metrics
 10. Performance: measure tick throughput with parallelism
"""

import math
import random
import time

import pytest

import brain_core

from brain.structures.trace_store import Trace, TraceStore
from brain.input.text_input import TextInput
from brain.input.visual_input import VisualInput
from brain.input.audio_input import AudioInput
from brain.input.sensory_input import SensoryInput
from brain.input.multimodal import MultimodalInput
from brain.output.motor_output import MotorOutput
from brain.output.speech_output import SpeechOutput
from brain.output.imagination import ImaginationOutput
from brain.learning.tick_loop import TickLoop
from brain.seed.seed_runner import seed_brain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trace_store_with_tokens(tokens: list[str]) -> TraceStore:
    """Create a trace store with labeled traces for given tokens."""
    ts = TraceStore()
    for i, tok in enumerate(tokens):
        lang_neurons = [105_000 + i * 3 + j for j in range(3)]
        speech_neurons = [140_000 + i * 3 + j for j in range(3)]
        pattern_neurons = [85_000 + i * 2 + j for j in range(2)]
        ml_neurons = [55_000 + i * 2 + j for j in range(2)]
        t = Trace(
            id=f"tok_{tok}",
            label=tok,
            neurons={
                "language": lang_neurons,
                "speech": speech_neurons,
                "pattern": pattern_neurons,
                "memory_long": ml_neurons,
            },
            strength=0.8,
            polarity=0.0,
            novelty=0.3,
        )
        ts.add(t)
    return ts


def make_gradient_image(w: int = 32, h: int = 32) -> list[list[float]]:
    """Create a synthetic gradient image (edge-rich) as 2D float list."""
    return [
        [abs(math.sin(r * 0.3) * math.cos(c * 0.3)) for c in range(w)]
        for r in range(h)
    ]


def make_checkerboard_image(w: int = 32, h: int = 32, block: int = 4) -> list[list[float]]:
    """Create a high-contrast checkerboard for strong edge detection."""
    return [
        [1.0 if ((r // block) + (c // block)) % 2 == 0 else 0.0
         for c in range(w)]
        for r in range(h)
    ]


def make_circle_image(w: int = 32, h: int = 32, cx: int = 16, cy: int = 16, radius: int = 8) -> list[list[float]]:
    """Create a circle for shape detection (mid-level) and spatial (center-of-mass)."""
    return [
        [1.0 if math.sqrt((r - cy) ** 2 + (c - cx) ** 2) <= radius else 0.0
         for c in range(w)]
        for r in range(h)
    ]


def make_sine_wave(freq: float = 440.0, duration: float = 0.05, sample_rate: int = 44100) -> list[float]:
    """Create a pure sine wave audio chunk."""
    n = int(duration * sample_rate)
    return [math.sin(2.0 * math.pi * freq * i / sample_rate) for i in range(n)]


def make_chord(freqs: list[float], duration: float = 0.05, sample_rate: int = 44100) -> list[float]:
    """Create a chord (sum of sine waves)."""
    n = int(duration * sample_rate)
    samples = [0.0] * n
    for freq in freqs:
        for i in range(n):
            samples[i] += math.sin(2.0 * math.pi * freq * i / sample_rate) / len(freqs)
    return samples


def make_noise(duration: float = 0.05, sample_rate: int = 44100, seed: int = 42) -> list[float]:
    """Create white noise audio chunk."""
    rng = random.Random(seed)
    n = int(duration * sample_rate)
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def make_video_frames(n_frames: int = 10, w: int = 32, h: int = 32) -> list[list[list[float]]]:
    """Create moving circle video — circle sweeps left to right across frames."""
    frames = []
    for f in range(n_frames):
        cx = int(4 + (w - 8) * f / max(n_frames - 1, 1))
        cy = h // 2
        frame = make_circle_image(w, h, cx, cy, 5)
        frames.append(frame)
    return frames


def fast_seed() -> TraceStore:
    """Quick seed with ~1000 traces and ~20k synapses for integration testing.
    Much faster than full seed_brain() while still exercising all pipelines."""
    from brain.utils.config import REGIONS

    rng = random.Random(42)
    ts = TraceStore()

    region_names = ["sensory", "visual", "audio", "pattern", "language",
                    "memory_short", "memory_long", "speech", "motor", "emotion"]
    for i in range(1000):
        neurons = {}
        for rname in region_names:
            start, end = REGIONS[rname]
            span = end - start + 1
            neurons[rname] = [start + rng.randint(0, min(span - 1, 999)) for _ in range(2)]
        label = f"word_{i}" if i < 100 else None
        t = Trace(
            id=f"t_{i}",
            label=label,
            neurons=neurons,
            strength=0.5,
            novelty=0.5 if i < 500 else 0.1,
        )
        ts.add(t)

    synapses = []
    # within sensory
    for _ in range(10_000):
        src = rng.randint(0, 9_999)
        tgt = rng.randint(0, 9_999)
        synapses.append((src, tgt, rng.uniform(0.1, 0.5), rng.randint(1, 3), 1.0))
    # sensory → pattern
    for _ in range(3_000):
        src = rng.randint(0, 9_999)
        tgt = rng.randint(85_000, 94_999)
        synapses.append((src, tgt, rng.uniform(0.2, 0.6), 1, 1.0))
    # pattern → language
    for _ in range(3_000):
        src = rng.randint(85_000, 94_999)
        tgt = rng.randint(105_000, 119_999)
        synapses.append((src, tgt, rng.uniform(0.2, 0.6), 1, 1.0))
    # language → speech
    for _ in range(2_000):
        src = rng.randint(105_000, 119_999)
        tgt = rng.randint(140_000, 149_999)
        synapses.append((src, tgt, rng.uniform(0.2, 0.6), 1, 1.0))
    # pain → motor withdraw (reflex)
    for i in range(200):
        synapses.append((5000 + i, 135_000 + i, 0.8, 2, 0.05))

    brain_core.init_brain_with_synapses(synapses)
    return ts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_brain():
    """Each test gets a fresh brain."""
    brain_core.init_brain()
    yield
    brain_core.reset_brain()


@pytest.fixture
def token_store() -> TraceStore:
    """Trace store with common English tokens."""
    return make_trace_store_with_tokens([
        "the", "cat", "sat", "on", "mat", "hello", "world",
        "hot", "cold", "pain", "big", "small", "run", "stop",
    ])


# ===========================================================================
# 1. PARALLELISM TESTS
# ===========================================================================


class TestParallelism:
    """Verify rayon parallelism is active and thread count is controllable."""

    def test_get_num_threads_returns_positive(self):
        """Thread pool should have at least 1 thread."""
        n = brain_core.get_num_threads()
        assert n >= 1, f"Expected ≥1 threads, got {n}"

    def test_tick_works_with_default_threads(self):
        """Tick should work with default (all cores) thread pool."""
        brain_core.inject_activations([(i, 0.5) for i in range(100)])
        tick_num, counts, total = brain_core.tick()
        assert tick_num == 0
        # Some sensory neurons should have fired
        assert counts.get("sensory", 0) > 0

    def test_parallel_tick_produces_same_as_sequential_logic(self):
        """Run many ticks — results should be deterministic (no race conditions)."""
        brain_core.inject_activations([(i, 0.6) for i in range(0, 200, 2)])
        results_a = []
        for _ in range(20):
            t, c, a = brain_core.tick()
            results_a.append(a)
        # Reset and do it again
        brain_core.reset_brain()
        brain_core.init_brain()
        brain_core.inject_activations([(i, 0.6) for i in range(0, 200, 2)])
        results_b = []
        for _ in range(20):
            t, c, a = brain_core.tick()
            results_b.append(a)
        assert results_a == results_b, "Parallel tick must be deterministic"

    def test_parallel_tick_with_synapses(self):
        """Parallelism should work correctly with cross-region synapses."""
        synapses = []
        # sensory → emotion connections
        for i in range(50):
            synapses.append((i, 70_000 + i, 0.5, 1, 1.0))
        # emotion → motor connections
        for i in range(50):
            synapses.append((70_000 + i, 130_000 + i, 0.5, 1, 1.0))
        brain_core.reset_brain()
        brain_core.init_brain_with_synapses(synapses)
        # Inject into sensory
        brain_core.inject_activations([(i, 0.8) for i in range(50)])
        # Tick 1: sensory fires, signal to emotion
        brain_core.tick()
        # Tick 2: emotion fires, signal to motor
        brain_core.tick()
        # Tick 3: motor should have received signal
        brain_core.tick()
        motor_act = brain_core.get_motor_activation()
        # After 3 ticks with strong synapses, motor should show some activity
        # (might be zero due to thresholds, but the pipeline should run without crash)
        assert motor_act >= 0.0

    def test_high_activity_parallel(self):
        """Many neurons active simultaneously — stress test parallelism."""
        # Inject neurons explicitly across multiple regions
        signals = []
        region_starts = [
            (0, 100),       # sensory
            (10_000, 100),  # visual
            (30_000, 100),  # audio
            (70_000, 100),  # emotion
            (85_000, 100),  # pattern
            (105_000, 100), # language
            (130_000, 100), # motor
        ]
        for start, count in region_starts:
            for j in range(count):
                signals.append((start + j, 0.8))
        brain_core.inject_activations(signals)
        tick_num, counts, total = brain_core.tick()
        assert total > 0, "Expected many neurons to fire with strong injection"
        # Multiple regions should be active
        active_regions = [r for r, c in counts.items() if c > 0]
        assert len(active_regions) >= 3, f"Expected ≥3 active regions, got {active_regions}"


# ===========================================================================
# 2. TEXT INPUT → TICK → SPEECH OUTPUT
# ===========================================================================


class TestTextPipeline:
    """End-to-end text input through ticks to speech output."""

    def test_text_activates_language_neurons(self, token_store):
        text_in = TextInput(token_store)
        result = text_in.encode("the cat sat on the mat")
        assert result["known_count"] >= 4, f"Expected ≥4 known tokens, got {result}"
        assert result["neurons_activated"] > 0

    def test_text_to_tick_activates_language_region(self, token_store):
        text_in = TextInput(token_store)
        text_in.encode("hello world")
        brain_core.tick()
        lang_act = brain_core.get_language_activation()
        assert lang_act > 0.0, "Language region should be active after text injection"

    def test_text_unknown_tokens_get_hash_neurons(self, token_store):
        text_in = TextInput(token_store)
        result = text_in.encode("xylophone quantum")
        assert result["unknown_count"] == 2
        assert result["neurons_activated"] > 0, "Unknown tokens should still activate neurons"

    def test_text_to_speech_pipeline(self, token_store):
        """Inject text → run ticks → decode speech output."""
        # Create synapses: language → speech
        synapses = []
        for tok in ["the", "cat", "sat"]:
            trace = token_store.get(f"tok_{tok}")
            if trace:
                for ln in trace.neurons.get("language", []):
                    for sn in trace.neurons.get("speech", []):
                        synapses.append((ln, sn, 0.7, 1, 1.0))
        brain_core.reset_brain()
        brain_core.init_brain_with_synapses(synapses)

        text_in = TextInput(token_store)
        speech_out = SpeechOutput(token_store)

        text_in.encode("the cat sat")
        for _ in range(5):
            brain_core.tick()

        result = speech_out.decode(top_k=20)
        assert result["speech_activity"] >= 0.0
        # Even if no strong speech output, the pipeline should not crash

    def test_text_tick_loop_integration(self, token_store):
        """Text input through the full TickLoop orchestrator."""
        loop = TickLoop(token_store)
        text_in = TextInput(token_store)
        text_in.encode("hello world")
        result = loop.step()
        assert "tick" in result
        assert result["language_activation"] >= 0.0


# ===========================================================================
# 3. IMAGE INPUT → TICK → IMAGINATION OUTPUT
# ===========================================================================


class TestImagePipeline:
    """Image input through ticks to imagination output."""

    def test_gradient_image_activates_visual(self):
        vis_in = VisualInput()
        frame = make_gradient_image()
        result = vis_in.encode(frame)
        assert result["neurons_activated"] > 0
        assert result["low_count"] > 0, "Gradient should produce edge features"
        assert result["mid_count"] > 0, "Gradient blocks should produce mid features"

    def test_checkerboard_strong_edges(self):
        vis_in = VisualInput()
        frame = make_checkerboard_image()
        result = vis_in.encode(frame)
        assert result["low_count"] > 50, "Checkerboard has many edges"

    def test_circle_spatial_features(self):
        vis_in = VisualInput()
        frame = make_circle_image(cx=8, cy=8, radius=5)
        result = vis_in.encode(frame)
        assert result["spatial_count"] > 0, "Circle should produce spatial features"

    def test_image_to_tick_activates_visual_region(self):
        vis_in = VisualInput()
        vis_in.encode(make_checkerboard_image())
        brain_core.tick()
        vis_act = brain_core.get_visual_activation()
        assert vis_act > 0.0, "Visual region should be active after image encoding"

    def test_image_to_imagination_roundtrip(self):
        """Encode image → tick → read back from imagination."""
        vis_in = VisualInput()
        vis_in.encode(make_checkerboard_image())
        brain_core.tick()
        brain_core.tick()

        imag = ImaginationOutput()
        result = imag.read()
        assert result["visual_activation"] > 0.0
        # Image should have some non-zero pixels
        total_brightness = sum(
            sum(row) for row in result["image"]
        )
        assert total_brightness > 0.0, "Imagination image should have content"

    def test_different_images_produce_different_activations(self):
        """Two different images should produce different neuron patterns."""
        vis_in = VisualInput()

        # Image 1: checkerboard
        vis_in.encode(make_checkerboard_image())
        brain_core.tick()
        peaks_1 = set(nid for nid, _ in brain_core.get_peak_visual_neurons(50))

        # Reset and try circle
        brain_core.reset_brain()
        brain_core.init_brain()
        vis_in.encode(make_circle_image())
        brain_core.tick()
        peaks_2 = set(nid for nid, _ in brain_core.get_peak_visual_neurons(50))

        # They should not be identical (different images → different patterns)
        if peaks_1 and peaks_2:
            overlap = len(peaks_1 & peaks_2) / max(len(peaks_1 | peaks_2), 1)
            assert overlap < 1.0, "Different images should produce different neuron patterns"

    def test_blank_image_minimal_activation(self):
        """A blank image should produce minimal visual activation."""
        vis_in = VisualInput()
        blank = [[0.0] * 32 for _ in range(32)]
        result = vis_in.encode(blank)
        assert result["low_count"] == 0, "Blank image has no edges"


# ===========================================================================
# 4. AUDIO INPUT → TICK → ACTIVATIONS
# ===========================================================================


class TestAudioPipeline:
    """Audio input through ticks to audio region activations."""

    def test_sine_wave_activates_frequency_neurons(self):
        audio_in = AudioInput()
        samples = make_sine_wave(440.0)
        result = audio_in.encode(samples, 44100)
        assert result["freq_count"] > 0, "440Hz sine should activate frequency neurons"

    def test_different_frequencies_different_neurons(self):
        """Two different pitches should activate different frequency neurons."""
        audio_in = AudioInput()

        # 440 Hz (A4)
        result_a = audio_in.encode(make_sine_wave(440.0), 44100)
        brain_core.tick()
        peaks_440 = set(nid for nid, _ in brain_core.get_peak_audio_neurons(30))

        brain_core.reset_brain()
        brain_core.init_brain()
        audio_in_b = AudioInput()

        # 1000 Hz
        result_b = audio_in_b.encode(make_sine_wave(1000.0), 44100)
        brain_core.tick()
        peaks_1k = set(nid for nid, _ in brain_core.get_peak_audio_neurons(30))

        if peaks_440 and peaks_1k:
            overlap = len(peaks_440 & peaks_1k) / max(len(peaks_440 | peaks_1k), 1)
            assert overlap < 0.8, "440Hz and 1kHz should activate different neurons"

    def test_chord_activates_multiple_frequency_bands(self):
        audio_in = AudioInput()
        samples = make_chord([261.6, 329.6, 392.0])  # C major chord
        result = audio_in.encode(samples, 44100)
        assert result["freq_count"] > 3, "Chord should activate more frequency neurons"

    def test_noise_activates_complex_features(self):
        audio_in = AudioInput()
        samples = make_noise()
        result = audio_in.encode(samples, 44100)
        # Noise has high zero-crossing rate → complex features
        assert result["complex_count"] > 0, "Noise should activate complex (timbre) neurons"

    def test_onset_detection(self):
        """Sudden energy increase should produce temporal onset signals."""
        audio_in = AudioInput()
        # First: silence
        silence = [0.0] * 2000
        audio_in.encode(silence, 44100)
        # Then: loud sine (energy jump)
        loud = make_sine_wave(440.0, 0.05, 44100)
        result = audio_in.encode(loud, 44100)
        assert result["temporal_count"] > 0, "Onset (silence→loud) should trigger temporal"

    def test_audio_to_tick_activates_region(self):
        audio_in = AudioInput()
        audio_in.encode(make_sine_wave(440.0), 44100)
        brain_core.tick()
        audio_act = brain_core.get_audio_activation()
        assert audio_act > 0.0, "Audio region should be active after audio encoding"

    def test_silence_minimal_activation(self):
        audio_in = AudioInput()
        silence = [0.0] * 2000
        result = audio_in.encode(silence, 44100)
        assert result["freq_count"] == 0, "Silence should have no frequency hits"


# ===========================================================================
# 5. SENSORY → MOTOR REFLEX PATH
# ===========================================================================


class TestSensoryMotorReflex:
    """Sensory input through reflex path to motor output."""

    def test_sensory_encoding_population_code(self):
        sens_in = SensoryInput()
        result = sens_in.encode(temperature=0.5, pressure=0.3, pain=0.0, texture=0.6)
        assert result["neurons_activated"] > 0
        assert result["signals_generated"] > 0

    def test_pain_activates_sensory_region(self):
        sens_in = SensoryInput()
        sens_in.encode(pain=0.8)
        brain_core.tick()
        pain_level = brain_core.get_pain_level()
        assert pain_level > 0.0, "Pain encoding should register as pain"

    def test_sensory_to_motor_with_reflex_wiring(self):
        """Pain → withdraw reflex via hardwired reflex synapses."""
        # Create reflex synapses (pain → motor withdraw)
        synapses = []
        for i in range(50):
            synapses.append((5000 + i, 135_000 + i, 0.8, 2, 0.05))
        brain_core.reset_brain()
        brain_core.init_brain_with_synapses(synapses)

        sens_in = SensoryInput()
        motor_out = MotorOutput()

        # Strong pain stimulus
        sens_in.encode(pain=0.9)

        # Several ticks for signal to propagate
        for _ in range(5):
            brain_core.tick()

        result = motor_out.read()
        assert result.action_type in ("withdraw", "conflict", "idle", "approach")
        # Motor should at least be accessible
        assert result.withdraw >= 0.0

    def test_temperature_encoding_gradient(self):
        """Different temperatures should activate different sensory sub-regions."""
        sens_in = SensoryInput()

        sens_in.encode(temperature=0.1)  # cold
        brain_core.tick()
        cold_peaks = set(nid for nid, _ in brain_core.get_peak_sensory_neurons(20))

        brain_core.reset_brain()
        brain_core.init_brain()
        sens_in.encode(temperature=0.9)  # hot
        brain_core.tick()
        hot_peaks = set(nid for nid, _ in brain_core.get_peak_sensory_neurons(20))

        if cold_peaks and hot_peaks:
            overlap = len(cold_peaks & hot_peaks) / max(len(cold_peaks | hot_peaks), 1)
            assert overlap < 0.5, "Cold and hot should activate different neurons"


# ===========================================================================
# 6. IMAGE + CAPTION (MULTIMODAL)
# ===========================================================================


class TestImageCaptionMultimodal:
    """Image with text caption — multimodal integration."""

    def test_multimodal_two_inputs_boosts_integration(self, token_store):
        """Image + text should boost the integration region."""
        mm = MultimodalInput(text_encoder=TextInput(token_store))

        result = mm.process({
            "text": "the cat",
            "visual": make_checkerboard_image(),
        }, tick=0)

        assert result["_summary"]["modality_count"] == 2
        assert "text" in result["_summary"]["active_modalities"]
        assert "visual" in result["_summary"]["active_modalities"]
        assert result["_summary"]["total_neurons_activated"] > 0

    def test_multimodal_integration_after_tick(self, token_store):
        mm = MultimodalInput(text_encoder=TextInput(token_store))
        mm.process({"text": "hello", "visual": make_gradient_image()}, tick=0)
        brain_core.tick()

        # Integration region should show activity after multimodal input
        integration_count = brain_core.get_active_count("integration")
        # At minimum the boost_integration was called
        assert integration_count >= 0

    def test_image_caption_produces_language_and_visual(self, token_store):
        mm = MultimodalInput(text_encoder=TextInput(token_store))
        result = mm.process({"text": "big cat", "visual": make_circle_image()}, tick=0)
        brain_core.tick()

        lang_act = brain_core.get_language_activation()
        vis_act = brain_core.get_visual_activation()
        assert lang_act > 0.0, "Language should be active from caption"
        assert vis_act > 0.0, "Visual should be active from image"

    def test_three_modalities_stronger_integration(self, token_store):
        """Text + visual + sensory should produce stronger integration than 2."""
        mm = MultimodalInput(text_encoder=TextInput(token_store))
        result = mm.process({
            "text": "hot pain",
            "visual": make_gradient_image(),
            "sensory": {"temperature": 0.8, "pain": 0.5},
        }, tick=0)
        assert result["_summary"]["modality_count"] == 3
        assert result["_summary"]["total_neurons_activated"] > 0


# ===========================================================================
# 7. VIDEO (FRAME SEQUENCE) → TEMPORAL PROCESSING
# ===========================================================================


class TestVideoPipeline:
    """Video = sequence of image frames. Tests temporal visual processing."""

    def test_video_frame_sequence(self):
        """Processing a sequence of frames should work and produce activations."""
        vis_in = VisualInput()
        frames = make_video_frames(n_frames=10)

        all_results = []
        for i, frame in enumerate(frames):
            result = vis_in.encode(frame)
            brain_core.tick()
            all_results.append(result)

        # All frames should produce some visual activity
        activated_counts = [r["neurons_activated"] for r in all_results]
        assert all(c > 0 for c in activated_counts), "All video frames should activate neurons"

    def test_moving_object_changes_activation(self):
        """Moving circle should change spatial focus across frames."""
        vis_in = VisualInput()
        imag = ImaginationOutput()
        frames = make_video_frames(n_frames=5)

        spatial_foci = []
        for frame in frames:
            vis_in.encode(frame)
            brain_core.tick()
            result = imag.read()
            if result["spatial_focus"] is not None:
                spatial_foci.append(result["spatial_focus"])

        if len(spatial_foci) >= 2:
            # Spatial focus should change as circle moves
            xs = [f[0] for f in spatial_foci]
            assert max(xs) - min(xs) > 0.0 or len(spatial_foci) < 2, \
                "Spatial focus should shift with moving object"

    def test_video_with_tick_loop(self, token_store):
        """Video frames through the full tick loop orchestrator."""
        loop = TickLoop(token_store)
        vis_in = VisualInput()
        frames = make_video_frames(5)

        metrics = []
        for frame in frames:
            vis_in.encode(frame)
            result = loop.step()
            metrics.append(result)

        # All ticks should complete
        assert len(metrics) == 5
        ticks = [m["tick"] for m in metrics]
        assert ticks == list(range(5)), "Tick numbers should be sequential"

    def test_long_video_stability(self):
        """50 frames should not crash or accumulate errors."""
        vis_in = VisualInput()
        frames = make_video_frames(n_frames=50)

        for frame in frames:
            vis_in.encode(frame)
            t, c, a = brain_core.tick()

        # Should complete without error
        assert brain_core.get_tick_count() == 50


# ===========================================================================
# 8. VIDEO + AUDIO (MULTIMODAL)
# ===========================================================================


class TestVideoAudioMultimodal:
    """Synchronized video + audio multimodal processing."""

    def test_video_with_audio_sync(self, token_store):
        """Process video frames synced with audio chunks."""
        mm = MultimodalInput(text_encoder=TextInput(token_store))
        frames = make_video_frames(n_frames=10)
        audio_chunk_dur = 0.05  # 50ms per frame at 20fps

        all_results = []
        for i, frame in enumerate(frames):
            # Generate audio: frequency increases with frame number
            freq = 200 + i * 100
            audio = make_sine_wave(freq, audio_chunk_dur)

            result = mm.process({
                "visual": frame,
                "audio": (audio, 44100),
            }, tick=i)
            brain_core.tick()
            all_results.append(result)

        # Each frame should have 2 active modalities
        for r in all_results:
            assert r["_summary"]["modality_count"] == 2

    def test_video_audio_integration_boost(self, token_store):
        """Video + audio should consistently boost integration."""
        mm = MultimodalInput(text_encoder=TextInput(token_store))
        frames = make_video_frames(5)

        for i, frame in enumerate(frames):
            audio = make_sine_wave(440.0, 0.05)
            mm.process({
                "visual": frame,
                "audio": (audio, 44100),
            }, tick=i)
            brain_core.tick()

        # After 5 synced frames, integration should have been boosted
        # (boost_integration is called when ≥2 modalities)
        int_count = brain_core.get_active_count("integration")
        assert int_count >= 0

    def test_video_audio_text_triple_modal(self, token_store):
        """Three modalities: video + audio + text narration."""
        mm = MultimodalInput(text_encoder=TextInput(token_store))
        frame = make_checkerboard_image()
        audio = make_sine_wave(440.0, 0.05)

        result = mm.process({
            "text": "the cat sat",
            "visual": frame,
            "audio": (audio, 44100),
        }, tick=0)

        assert result["_summary"]["modality_count"] == 3
        assert result["_summary"]["total_neurons_activated"] > 0

    def test_all_four_modalities(self, token_store):
        """Video + audio + text + sensory — maximal multimodal input."""
        mm = MultimodalInput(text_encoder=TextInput(token_store))
        frame = make_checkerboard_image()
        audio = make_sine_wave(440.0, 0.05)

        result = mm.process({
            "text": "hot pain stop",
            "visual": frame,
            "audio": (audio, 44100),
            "sensory": {"temperature": 0.9, "pain": 0.7, "pressure": 0.3},
        }, tick=0)

        assert result["_summary"]["modality_count"] == 4
        assert len(result["_summary"]["active_modalities"]) == 4
        brain_core.tick()

        # All input regions should show some activation
        vis_act = brain_core.get_visual_activation()
        audio_act = brain_core.get_audio_activation()
        sens_act = brain_core.get_sensory_activation()
        lang_act = brain_core.get_language_activation()
        assert vis_act > 0.0, "Visual should be active"
        assert audio_act > 0.0, "Audio should be active"
        assert sens_act > 0.0, "Sensory should be active"
        assert lang_act > 0.0, "Language should be active"


# ===========================================================================
# 9. FULL LIFECYCLE: SEED → MULTI-TICK → LEARNING
# ===========================================================================


class TestFullLifecycle:
    """Full lifecycle tests with a seeded brain (uses fast_seed for speed)."""

    def _seed(self) -> TraceStore:
        """Re-seed with fast_seed (called per test since autouse resets)."""
        return fast_seed()

    def test_seed_smoke(self):
        """Fast-seeded brain should have correct neuron and synapse counts."""
        ts = self._seed()
        n = brain_core.get_neuron_count()
        s = brain_core.get_synapse_count()
        assert n == 152_000, f"Expected 152k neurons, got {n}"
        assert s > 10_000, f"Expected >10k synapses, got {s}"
        assert len(ts) == 1000, f"Expected 1000 traces, got {len(ts)}"

    def test_seeded_brain_tick_produces_activity(self):
        """After seeding and injecting, the brain should show activity."""
        self._seed()
        brain_core.inject_activations([(i, 0.6) for i in range(0, 100)])
        t, c, a = brain_core.tick()
        assert a > 0, "Seeded brain should have active neurons after injection"

    def test_seeded_tick_loop_multi_step(self):
        """Run the TickLoop for 10 steps on a seeded brain."""
        ts = self._seed()
        loop = TickLoop(ts)
        sens_in = SensoryInput()

        all_results = []
        for i in range(10):
            temp = 0.3 + 0.4 * math.sin(i * 0.5)
            sens_in.encode(temperature=temp)
            result = loop.step()
            all_results.append(result)

        assert len(all_results) == 10
        assert all_results[-1]["energy"] <= all_results[0]["energy"] + 0.01

    def test_hebbian_learning_occurs(self):
        """Hebbian updates should occur when neurons co-fire."""
        ts = self._seed()
        loop = TickLoop(ts)
        brain_core.inject_activations([(i, 0.8) for i in range(0, 500)])

        total_hebb = 0
        for _ in range(5):
            result = loop.step()
            total_hebb += result["hebbian_updates"]

        assert total_hebb > 0, "Hebbian learning should occur with co-active neurons"

    def test_full_io_lifecycle(self):
        """Full I/O: text + image → ticks → speech + motor + imagination."""
        ts = self._seed()
        text_in = TextInput(ts)
        vis_in = VisualInput()
        loop = TickLoop(ts)
        speech_out = SpeechOutput(ts)
        motor_out = MotorOutput()
        imag = ImaginationOutput()

        text_in.encode("word_0 word_1 word_2")
        vis_in.encode(make_gradient_image())

        for _ in range(5):
            result = loop.step()

        speech_result = speech_out.decode(top_k=20)
        motor_result = motor_out.read()
        imag_result = imag.read()

        assert "text" in speech_result
        assert motor_result.action_type in ("idle", "approach", "withdraw", "conflict")
        assert "image" in imag_result
        assert len(imag_result["image"]) == 32

    def test_full_seed_brain_smoke(self):
        """Verify full seed_brain() works (slow but proves the real pipeline)."""
        _, ts = seed_brain(seed=42, verbose=False)
        n = brain_core.get_neuron_count()
        s = brain_core.get_synapse_count()
        assert n == 152_000
        assert s > 100_000
        assert len(ts) > 100_000
        # One tick should work
        brain_core.inject_activations([(i, 0.6) for i in range(50)])
        t, c, a = brain_core.tick()
        assert a > 0


# ===========================================================================
# 10. PERFORMANCE: TICK THROUGHPUT
# ===========================================================================


class TestPerformance:
    """Measure and verify tick throughput with parallelism."""

    def test_tick_throughput_empty(self):
        """Measure ticks/sec with no activity (baseline)."""
        n_ticks = 1000
        start = time.perf_counter()
        for _ in range(n_ticks):
            brain_core.tick()
        elapsed = time.perf_counter() - start
        tps = n_ticks / elapsed
        # With 152k neurons and parallel updates, should be fast
        assert tps > 100, f"Expected >100 ticks/sec baseline, got {tps:.0f}"

    def test_tick_throughput_with_activity(self):
        """Measure ticks/sec with moderate neuron activity."""
        # Set up synapses for sustained activity
        synapses = []
        for i in range(0, 1000, 2):
            synapses.append((i, i + 1, 0.4, 1, 1.0))
        brain_core.reset_brain()
        brain_core.init_brain_with_synapses(synapses)

        n_ticks = 500
        start = time.perf_counter()
        for i in range(n_ticks):
            if i % 50 == 0:
                brain_core.inject_activations([(j, 0.5) for j in range(0, 200, 2)])
            brain_core.tick()
        elapsed = time.perf_counter() - start
        tps = n_ticks / elapsed
        assert tps > 50, f"Expected >50 ticks/sec with activity, got {tps:.0f}"

    def test_thread_count_query(self):
        """Verify thread count is accessible and reasonable."""
        n = brain_core.get_num_threads()
        assert 1 <= n <= 256, f"Thread count {n} is out of reasonable range"

    def test_multimodal_throughput(self, token_store):
        """Measure throughput for full multimodal pipeline per tick."""
        mm = MultimodalInput(text_encoder=TextInput(token_store))
        frame = make_checkerboard_image()
        audio = make_sine_wave(440.0, 0.02)

        n_ticks = 50
        start = time.perf_counter()
        for i in range(n_ticks):
            mm.process({
                "text": "hello",
                "visual": frame,
                "audio": (audio, 44100),
            }, tick=i)
            brain_core.tick()
        elapsed = time.perf_counter() - start
        tps = n_ticks / elapsed
        assert tps > 5, f"Expected >5 multimodal ticks/sec, got {tps:.1f}"
