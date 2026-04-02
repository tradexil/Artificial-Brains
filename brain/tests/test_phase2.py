"""Phase 2 tests: Seed & Structure.

Tests:
  - Neuron counts per region match spec
  - Within-region synapse generation
  - Trace generation and inverted index
  - Cross-region synapse generation
  - TraceStore save/load round-trip
  - Full seed runner integration
"""

import os
import random
import tempfile

import pytest

from brain.structures.neuron_map import (
    all_region_names,
    global_to_local,
    inhibitory_range,
    is_inhibitory,
    local_to_global,
    region_for_neuron,
    region_size,
)
from brain.structures.trace_store import Trace, TraceStore
from brain.seed.spawn_synapses import spawn_within_region_synapses, spawn_cross_region_synapses
from brain.seed.spawn_traces import spawn_traces
from brain.utils.config import (
    INITIAL_TRACES,
    INITIAL_WITHIN_REGION_SYNAPSES_PER_NEURON,
    NEURONS_PER_TRACE,
    REGIONS,
    SIGNAL_FLOW_CONNECTIONS,
    TOTAL_NEURONS,
)


# =======================
# neuron_map tests
# =======================

class TestNeuronMap:
    def test_total_neurons(self):
        total = sum(region_size(r) for r in all_region_names())
        assert total == TOTAL_NEURONS

    def test_region_ranges_contiguous(self):
        names = all_region_names()
        for i in range(1, len(names)):
            _, prev_end = REGIONS[names[i - 1]]
            curr_start, _ = REGIONS[names[i]]
            assert curr_start == prev_end + 1, (
                f"Gap between {names[i-1]} and {names[i]}: "
                f"{prev_end} → {curr_start}"
            )

    def test_region_lookup(self):
        assert region_for_neuron(0) == "sensory"
        assert region_for_neuron(9_999) == "sensory"
        assert region_for_neuron(10_000) == "visual"
        assert region_for_neuron(151_999) == "numbers"
        assert region_for_neuron(152_000) is None

    def test_local_global_roundtrip(self):
        for name in all_region_names():
            start, end = REGIONS[name]
            # first neuron
            g = local_to_global(name, 0)
            assert g == start
            assert global_to_local(name, g) == 0
            # last neuron
            last_local = region_size(name) - 1
            g2 = local_to_global(name, last_local)
            assert g2 == end
            assert global_to_local(name, g2) == last_local

    def test_inhibitory_neurons(self):
        for name in all_region_names():
            start, end = REGIONS[name]
            inhib_start, inhib_end = inhibitory_range(name)
            assert inhib_start > start  # not all inhibitory
            assert inhib_end == end
            assert is_inhibitory(inhib_end)
            assert not is_inhibitory(start)


# =======================
# trace_store tests
# =======================

class TestTraceStore:
    def test_add_and_get(self):
        store = TraceStore()
        t = Trace(id="t1", neurons={"sensory": [0, 1, 2]})
        store.add(t)
        assert "t1" in store
        assert store.get("t1") is t
        assert len(store) == 1

    def test_remove(self):
        store = TraceStore()
        t = Trace(id="t1", neurons={"sensory": [0, 1]})
        store.add(t)
        removed = store.remove("t1")
        assert removed is t
        assert "t1" not in store
        assert store.traces_for_neuron(0) == set()

    def test_inverted_index(self):
        store = TraceStore()
        t1 = Trace(id="t1", neurons={"sensory": [0, 1, 2]})
        t2 = Trace(id="t2", neurons={"sensory": [1, 2, 3]})
        store.add(t1)
        store.add(t2)

        assert store.traces_for_neuron(0) == {"t1"}
        assert store.traces_for_neuron(1) == {"t1", "t2"}
        assert store.traces_for_neuron(3) == {"t2"}
        assert store.traces_for_neuron(99) == set()

    def test_candidate_traces(self):
        store = TraceStore()
        t1 = Trace(id="t1", neurons={"sensory": [0, 1, 2]})
        t2 = Trace(id="t2", neurons={"visual": [10_000, 10_001]})
        store.add(t1)
        store.add(t2)

        candidates = store.candidate_traces([0, 1])
        assert candidates == {"t1": 2}

    def test_matching_traces(self):
        store = TraceStore()
        t = Trace(id="t1", neurons={"sensory": [0, 1, 2, 3, 4]})
        store.add(t)

        # 3/5 = 0.6 → matches at threshold 0.6
        matches = store.matching_traces([0, 1, 2], threshold=0.6)
        assert len(matches) == 1
        assert matches[0][0] == "t1"
        assert abs(matches[0][1] - 0.6) < 0.01

        # 2/5 = 0.4 → does not match at threshold 0.6
        matches = store.matching_traces([0, 1], threshold=0.6)
        assert len(matches) == 0

    def test_save_load_roundtrip(self):
        store = TraceStore()
        t1 = Trace(
            id="t1",
            label="test",
            neurons={"sensory": [0, 1], "visual": [10_000]},
            strength=0.5,
            polarity=-0.3,
            novelty=0.8,
        )
        t2 = Trace(id="t2", neurons={"audio": [30_000]}, strength=0.2)
        store.add(t1)
        store.add(t2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "traces.json")
            store.save(path)

            loaded = TraceStore()
            loaded.load(path)

        assert len(loaded) == 2
        assert loaded.get("t1").label == "test"
        assert abs(loaded.get("t1").strength - 0.5) < 0.001
        assert loaded.get("t1").neurons["sensory"] == [0, 1]
        assert loaded.traces_for_neuron(0) == {"t1"}

    def test_trace_serialization(self):
        t = Trace(
            id="t1",
            label="hello",
            neurons={"sensory": [5, 6]},
            binding_ids=[10, 20],
            strength=0.7,
            decay=0.9,
            polarity=0.5,
            abstraction=0.3,
            novelty=0.2,
            co_traces=["t2"],
            context_tags=["tag1"],
            fire_count=10,
            last_fired=100,
            formation_tick=50,
        )
        d = t.to_dict()
        t2 = Trace.from_dict(d)
        assert t2.id == t.id
        assert t2.label == t.label
        assert t2.neurons == t.neurons
        assert t2.binding_ids == t.binding_ids
        assert abs(t2.strength - t.strength) < 0.001
        assert t2.fire_count == t.fire_count


# =======================
# spawn_synapses tests
# =======================

class TestSpawnSynapses:
    def test_within_region_synapse_count(self):
        """Within-region synapses should be approximately 20 per neuron."""
        rng = random.Random(42)
        synapses = spawn_within_region_synapses(rng=rng)

        # Each non-numbers neuron gets ~20 synapses
        # numbers region is skipped
        non_numbers = TOTAL_NEURONS - region_size("numbers")
        expected = non_numbers * INITIAL_WITHIN_REGION_SYNAPSES_PER_NEURON
        # Allow 1% tolerance (some neurons may not find enough unique targets)
        assert abs(len(synapses) - expected) < expected * 0.01, (
            f"Expected ~{expected:,}, got {len(synapses):,}"
        )

    def test_within_region_stays_in_region(self):
        """All synapses should connect neurons in the same region."""
        rng = random.Random(42)
        synapses = spawn_within_region_synapses(rng=rng)

        # Check a sample (all 3M would be slow in a test)
        sample = synapses[:10_000]
        for (src, dst, w, d, p) in sample:
            src_region = region_for_neuron(src)
            dst_region = region_for_neuron(dst)
            assert src_region == dst_region, (
                f"Cross-region synapse found: {src}({src_region}) → {dst}({dst_region})"
            )

    def test_within_region_no_self_connections(self):
        rng = random.Random(42)
        synapses = spawn_within_region_synapses(rng=rng)
        for (src, dst, w, d, p) in synapses[:10_000]:
            assert src != dst

    def test_within_region_weight_range(self):
        rng = random.Random(42)
        synapses = spawn_within_region_synapses(rng=rng)
        for (src, dst, w, d, p) in synapses[:10_000]:
            assert 0.01 <= w <= 0.15
            assert d in (1, 2)
            assert p == 1.0

    def test_cross_region_synapses(self):
        """Cross-region synapses follow signal flow."""
        rng = random.Random(42)
        traces_neurons = [
            {"sensory": [100, 101], "pattern": [85_000, 85_001], "visual": [10_000]},
            {"sensory": [200], "pattern": [85_100]},
        ]
        synapses = spawn_cross_region_synapses(
            traces_neurons, SIGNAL_FLOW_CONNECTIONS, rng=rng,
        )
        assert len(synapses) > 0

        for (src, dst, w, d, p) in synapses:
            src_region = region_for_neuron(src)
            dst_region = region_for_neuron(dst)
            assert (src_region, dst_region) in SIGNAL_FLOW_CONNECTIONS, (
                f"Unexpected connection: {src_region} → {dst_region}"
            )
            assert 0.05 <= w <= 0.2
            assert 3 <= d <= 8

    def test_cross_region_empty_when_no_matching_regions(self):
        rng = random.Random(42)
        traces_neurons = [
            {"sensory": [100]},  # No pattern neurons → no sensory→pattern synapses
        ]
        synapses = spawn_cross_region_synapses(
            traces_neurons, SIGNAL_FLOW_CONNECTIONS, rng=rng,
        )
        assert len(synapses) == 0


# =======================
# spawn_traces tests
# =======================

class TestSpawnTraces:
    def test_trace_count(self):
        store = spawn_traces(count=100, rng=random.Random(1))
        assert len(store) == 100

    def test_default_count(self):
        store = spawn_traces(count=INITIAL_TRACES, rng=random.Random(1))
        assert len(store) == INITIAL_TRACES

    def test_trace_neuron_structure(self):
        store = spawn_traces(count=10, rng=random.Random(1))
        for trace in store.traces.values():
            for region, expected_count in NEURONS_PER_TRACE.items():
                if expected_count == 0:
                    assert region not in trace.neurons or len(trace.neurons.get(region, [])) == 0
                else:
                    assert len(trace.neurons[region]) == expected_count, (
                        f"Trace {trace.id} region {region}: "
                        f"expected {expected_count}, got {len(trace.neurons.get(region, []))}"
                    )

    def test_trace_neurons_in_correct_region(self):
        store = spawn_traces(count=100, rng=random.Random(1))
        for trace in store.traces.values():
            for region, neuron_ids in trace.neurons.items():
                start, end = REGIONS[region]
                for nid in neuron_ids:
                    assert start <= nid <= end, (
                        f"Neuron {nid} not in {region} [{start}, {end}]"
                    )

    def test_trace_no_duplicate_neurons(self):
        store = spawn_traces(count=100, rng=random.Random(1))
        for trace in store.traces.values():
            for region, neuron_ids in trace.neurons.items():
                assert len(neuron_ids) == len(set(neuron_ids)), (
                    f"Duplicate neurons in {trace.id} {region}"
                )

    def test_inverted_index_populated(self):
        store = spawn_traces(count=100, rng=random.Random(1))
        # At least some neurons should appear in the index
        all_indexed_neurons = set()
        for trace in store.traces.values():
            for neuron_ids in trace.neurons.values():
                all_indexed_neurons.update(neuron_ids)

        for nid in list(all_indexed_neurons)[:50]:
            assert len(store.traces_for_neuron(nid)) > 0

    def test_trace_metadata_ranges(self):
        store = spawn_traces(count=100, rng=random.Random(1))
        for trace in store.traces.values():
            assert 0.1 <= trace.strength <= 0.3
            assert trace.decay == 1.0
            assert trace.polarity == 0.0
            assert 0.0 <= trace.abstraction <= 0.4
            assert trace.novelty == 1.0

    def test_total_neurons_per_trace(self):
        expected_total = sum(v for v in NEURONS_PER_TRACE.values())
        store = spawn_traces(count=10, rng=random.Random(1))
        for trace in store.traces.values():
            assert trace.total_neurons() == expected_total
