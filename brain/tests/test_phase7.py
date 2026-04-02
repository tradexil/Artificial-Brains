"""Phase 7 tests: Language & Speech.

Tests:
  - Rust PyO3 language functions (activation, overlap, monologue, boost, peaks)
  - Rust PyO3 speech functions (activity, peaks, lateral inhibition, boost)
  - TextInput encoder (tokenize, encode, hash neurons)
  - SpeechOutput decoder (decode, reverse index, raw read)
  - Tick loop Phase 7 metrics (language_activation, inner_monologue, speech_activity)
  - End-to-end: text → language → speech → text
"""

import pytest
import brain_core

from brain.structures.trace_store import Trace, TraceStore
from brain.input.text_input import TextInput
from brain.output.speech_output import SpeechOutput


@pytest.fixture(autouse=True)
def fresh_brain():
    """Reset the brain before every test."""
    brain_core.init_brain()
    yield
    brain_core.reset_brain()


# === Helper ===

def make_trace_store_with_words():
    """Create a TraceStore with some labeled traces that have language+speech neurons."""
    ts = TraceStore()
    words = [
        ("cat",  [105000, 105001, 105002], [140000, 140001]),
        ("dog",  [105010, 105011, 105012], [140010, 140011]),
        ("bird", [105020, 105021, 105022], [140020, 140021]),
        ("tree", [105030, 105031, 105032], [140030, 140031]),
        ("run",  [105040, 105041, 105042], [140040, 140041]),
    ]
    for i, (word, lang_n, speech_n) in enumerate(words):
        t = Trace(
            id=f"trace_{i}",
            label=word,
            neurons={
                "language": lang_n,
                "speech": speech_n,
                "sensory": [i * 10, i * 10 + 1],
                "memory_short": [45000 + i],
            },
        )
        ts.add(t)
    return ts


# ===================== RUST LANGUAGE FUNCTIONS =====================

class TestLanguageRust:

    def test_language_activation_zero_initially(self):
        act = brain_core.get_language_activation()
        assert act == pytest.approx(0.0, abs=1e-6)

    def test_language_activation_after_boost(self):
        neurons = list(range(105000, 105300))
        brain_core.boost_language(neurons, 0.8)
        act = brain_core.get_language_activation()
        assert act > 0.2

    def test_symbol_overlap_none(self):
        overlap = brain_core.get_symbol_overlap([105000, 105001, 105002])
        assert overlap == pytest.approx(0.0, abs=1e-6)

    def test_symbol_overlap_after_boost(self):
        neurons = [105000, 105001, 105002]
        brain_core.boost_language(neurons, 0.8)
        overlap = brain_core.get_symbol_overlap(neurons)
        assert overlap > 0.9

    def test_symbol_overlap_partial(self):
        trace_neurons = [105000, 105001, 105002, 105003]
        brain_core.boost_language([105000, 105001], 0.8)
        overlap = brain_core.get_symbol_overlap(trace_neurons)
        assert 0.4 <= overlap <= 0.6

    def test_inner_monologue_zero_when_silent(self):
        mono = brain_core.get_inner_monologue_signal()
        assert mono == pytest.approx(0.0, abs=1e-6)

    def test_inner_monologue_needs_both_regions(self):
        # Only language
        brain_core.boost_language(list(range(105000, 105300)), 0.8)
        mono_lang_only = brain_core.get_inner_monologue_signal()

        # Add executive activity
        brain_core.inject_activations([(n, 1.5) for n in range(120000, 120300)])
        brain_core.tick()
        mono_both = brain_core.get_inner_monologue_signal()

        assert mono_both >= mono_lang_only

    def test_boost_language_returns_count(self):
        count = brain_core.boost_language([105000, 105001, 105002], 0.5)
        assert count == 3

    def test_boost_language_out_of_range(self):
        count = brain_core.boost_language([0, 200000], 0.5)
        assert count == 0

    def test_peak_language_neurons_empty(self):
        peaks = brain_core.get_peak_language_neurons(5)
        assert peaks == []

    def test_peak_language_neurons_after_boost(self):
        brain_core.boost_language(list(range(105000, 105020)), 0.8)
        peaks = brain_core.get_peak_language_neurons(5)
        assert len(peaks) > 0
        assert len(peaks) <= 5
        # Each element is (neuron_id, activation)
        for nid, act in peaks:
            assert 105000 <= nid < 114000
            assert act > 0.0
        # Sorted descending
        for i in range(1, len(peaks)):
            assert peaks[i][1] <= peaks[i - 1][1]


# ===================== RUST SPEECH FUNCTIONS =====================

class TestSpeechRust:

    def test_speech_activity_zero_initially(self):
        act = brain_core.get_speech_activity()
        assert act == pytest.approx(0.0, abs=1e-6)

    def test_speech_activity_after_boost(self):
        neurons = list(range(140000, 140200))
        brain_core.boost_speech(neurons, 0.8)
        act = brain_core.get_speech_activity()
        assert act > 0.2

    def test_peak_speech_neurons_empty(self):
        peaks = brain_core.get_peak_speech_neurons(5)
        assert peaks == []

    def test_peak_speech_neurons_after_boost(self):
        brain_core.boost_speech(list(range(140000, 140050)), 0.8)
        peaks = brain_core.get_peak_speech_neurons(10)
        assert len(peaks) > 0
        for nid, act in peaks:
            assert 140000 <= nid < 148000
            assert act > 0.0

    def test_lateral_inhibition_no_activity(self):
        suppressed = brain_core.speech_lateral_inhibition(0.8)
        assert suppressed == 0

    def test_lateral_inhibition_suppresses_weak(self):
        # One strong, several weak
        brain_core.boost_speech([140000], 1.0)
        brain_core.boost_speech([140001, 140002, 140003], 0.3)
        suppressed = brain_core.speech_lateral_inhibition(0.8)
        assert suppressed >= 3

    def test_boost_speech_returns_count(self):
        count = brain_core.boost_speech([140000, 140001], 0.5)
        assert count == 2

    def test_boost_speech_out_of_range(self):
        count = brain_core.boost_speech([0, 200000], 0.5)
        assert count == 0


# ===================== TEXT INPUT =====================

class TestTextInput:

    def test_tokenize(self):
        ts = TraceStore()
        encoder = TextInput(ts)
        tokens = encoder.tokenize("The Cat sat, on the mat!")
        assert tokens == ["the", "cat", "sat", "on", "the", "mat"]

    def test_encode_known_tokens(self):
        ts = make_trace_store_with_words()
        encoder = TextInput(ts)
        result = encoder.encode("cat dog")
        assert result["known_count"] == 2
        assert result["unknown_count"] == 0
        assert result["neurons_activated"] > 0
        assert len(result["matched_traces"]) == 2

    def test_encode_unknown_tokens(self):
        ts = make_trace_store_with_words()
        encoder = TextInput(ts)
        result = encoder.encode("xyz abc")
        assert result["known_count"] == 0
        assert result["unknown_count"] == 2
        assert result["neurons_activated"] > 0

    def test_encode_mixed(self):
        ts = make_trace_store_with_words()
        encoder = TextInput(ts)
        result = encoder.encode("cat xyz dog")
        assert result["known_count"] == 2
        assert result["unknown_count"] == 1

    def test_hash_deterministic(self):
        ts = TraceStore()
        encoder = TextInput(ts)
        n1 = encoder._hash_neurons("hello")
        n2 = encoder._hash_neurons("hello")
        assert n1 == n2

    def test_hash_different_tokens(self):
        ts = TraceStore()
        encoder = TextInput(ts)
        n1 = encoder._hash_neurons("hello")
        n2 = encoder._hash_neurons("world")
        assert n1 != n2

    def test_encode_token_single(self):
        ts = make_trace_store_with_words()
        encoder = TextInput(ts)
        neurons = encoder.encode_token("cat")
        assert neurons == [105000, 105001, 105002]

    def test_encode_activates_language_region(self):
        ts = make_trace_store_with_words()
        encoder = TextInput(ts)
        encoder.encode("cat dog bird")
        act = brain_core.get_language_activation()
        assert act > 0.0

    def test_refresh_cache(self):
        ts = make_trace_store_with_words()
        encoder = TextInput(ts)
        # Add a new trace
        new_trace = Trace(
            id="new_trace",
            label="fish",
            neurons={"language": [105050, 105051]},
        )
        ts.add(new_trace)
        # Before refresh, fish is unknown
        result1 = encoder.encode("fish")
        assert result1["known_count"] == 0
        # After refresh, fish is known
        encoder.refresh_cache()
        result2 = encoder.encode("fish")
        assert result2["known_count"] == 1


# ===================== SPEECH OUTPUT =====================

class TestSpeechOutput:

    def test_decode_no_activity(self):
        ts = make_trace_store_with_words()
        decoder = SpeechOutput(ts)
        result = decoder.decode()
        assert result["text"] == ""
        assert result["tokens"] == []

    def test_decode_with_boosted_speech(self):
        ts = make_trace_store_with_words()
        decoder = SpeechOutput(ts)
        # Boost speech neurons for "cat" trace (140000, 140001)
        brain_core.boost_speech([140000, 140001], 1.0)
        result = decoder.decode(top_k=10)
        assert result["speech_activity"] > 0.0
        # Should find "cat" in decoded tokens
        labels = [label for label, _ in result["tokens"]]
        assert "cat" in labels

    def test_decode_multiple_words(self):
        ts = make_trace_store_with_words()
        decoder = SpeechOutput(ts)
        # Boost both cat and dog speech neurons
        brain_core.boost_speech([140000, 140001], 1.0)
        brain_core.boost_speech([140010, 140011], 0.8)
        result = decoder.decode(top_k=10)
        labels = [label for label, _ in result["tokens"]]
        assert "cat" in labels
        assert "dog" in labels

    def test_read_raw(self):
        ts = make_trace_store_with_words()
        decoder = SpeechOutput(ts)
        brain_core.boost_speech([140000], 0.5)
        raw = decoder.read_raw(5)
        assert len(raw) >= 1
        assert raw[0][0] == 140000

    def test_refresh_index(self):
        ts = make_trace_store_with_words()
        decoder = SpeechOutput(ts)
        # Add new trace with speech neurons
        new_trace = Trace(
            id="new_trace",
            label="fish",
            neurons={"speech": [140050, 140051]},
        )
        ts.add(new_trace)
        decoder.refresh_index()
        brain_core.boost_speech([140050, 140051], 1.0)
        result = decoder.decode(top_k=10)
        labels = [label for label, _ in result["tokens"]]
        assert "fish" in labels


# ===================== TICK LOOP INTEGRATION =====================

class TestTickLoopPhase7:

    def test_tick_loop_returns_phase7_metrics(self):
        from brain.learning.tick_loop import TickLoop
        ts = make_trace_store_with_words()
        loop = TickLoop(ts)
        result = loop.step()
        assert "language_activation" in result
        assert "inner_monologue" in result
        assert "speech_activity" in result

    def test_tick_loop_language_activation_from_boost(self):
        from brain.learning.tick_loop import TickLoop
        ts = make_trace_store_with_words()
        loop = TickLoop(ts)
        # Boost language neurons
        brain_core.boost_language(list(range(105000, 105300)), 0.8)
        result = loop.step()
        assert result["language_activation"] > 0.0

    def test_tick_loop_silent_phase7(self):
        from brain.learning.tick_loop import TickLoop
        ts = TraceStore()
        loop = TickLoop(ts)
        result = loop.step()
        assert result["language_activation"] == pytest.approx(0.0, abs=1e-6)
        assert result["inner_monologue"] == pytest.approx(0.0, abs=1e-6)
        assert result["speech_activity"] == pytest.approx(0.0, abs=1e-6)


# ===================== END-TO-END =====================

class TestEndToEnd:

    def test_text_to_language_to_speech(self):
        """Full pipeline: text → language neurons → speech neurons → text."""
        ts = make_trace_store_with_words()
        encoder = TextInput(ts)
        decoder = SpeechOutput(ts)

        # Step 1: Encode text → activates language neurons
        encode_result = encoder.encode("cat")
        assert encode_result["neurons_activated"] > 0

        # Step 2: Drive language → speech via trace neuron mapping
        # (In full brain this happens via synapses, here we simulate)
        for trace in ts.traces.values():
            if trace.label == "cat":
                speech_neurons = trace.neurons.get("speech", [])
                lang_neurons = trace.neurons.get("language", [])
                # Check language neurons are active
                overlap = brain_core.get_symbol_overlap(lang_neurons)
                if overlap > 0.5:
                    brain_core.boost_speech(speech_neurons, 0.8)

        # Step 3: Decode speech → text
        decode_result = decoder.decode(top_k=10)
        labels = [label for label, _ in decode_result["tokens"]]
        assert "cat" in labels

    def test_multiple_words_pipeline(self):
        """Encode multiple words, verify language activation, decode speech."""
        ts = make_trace_store_with_words()
        encoder = TextInput(ts, boost=1.0)
        decoder = SpeechOutput(ts)

        # Encode
        result = encoder.encode("cat run tree")
        assert result["known_count"] == 3

        # Check language is active
        act = brain_core.get_language_activation()
        assert act > 0.0

        # Manually bridge language → speech for matching traces
        for tid in result["matched_traces"]:
            trace = ts.get(tid)
            if trace:
                speech_n = trace.neurons.get("speech", [])
                if speech_n:
                    brain_core.boost_speech(speech_n, 0.8)

        # Decode
        decode_result = decoder.decode(top_k=10)
        labels = [label for label, _ in decode_result["tokens"]]
        assert len(labels) >= 2  # At least some words decoded
