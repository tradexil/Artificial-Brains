"""Benchmark helpers for controlled graph and workload comparisons."""

from brain.benchmarks.fixed_graph_overlay import (
    DEFAULT_FIXED_GRAPH_OVERLAY_DELAYS,
    DEFAULT_FIXED_GRAPH_OVERLAY_REGIONS,
    FixedGraphSpec,
    build_fixed_graph_spec,
)
from brain.benchmarks.text_learning_probe import (
    DEFAULT_TEXT_LEARNING_PROBE_SPECS,
    run_text_learning_probe,
)
from brain.benchmarks.audio_learning_probe import (
    DEFAULT_AUDIO_LEARNING_PROBE_LABELS,
    run_audio_learning_probe,
)
from brain.benchmarks.async_multi_brain import (
    DEFAULT_ASYNC_MULTI_BRAIN_CORES_PER_WORKER,
    DEFAULT_ASYNC_MULTI_BRAIN_MERGE_EVERY_SAMPLES,
    DEFAULT_ASYNC_MULTI_BRAIN_WORKER_COUNT,
    run_async_multi_brain_text,
)
from brain.benchmarks.crossmodal_recall_probe import (
    DEFAULT_CROSSMODAL_RECALL_PROBE_SPEC,
    run_crossmodal_recall_probe,
)
from brain.benchmarks.executive_numbers_probe import run_executive_numbers_probe
from brain.benchmarks.multimodal_binding_probe import (
    DEFAULT_MULTIMODAL_BINDING_PROBE_CATALOG,
    DEFAULT_MULTIMODAL_BINDING_PROBE_SPEC,
    run_multimodal_binding_probe,
)
from brain.benchmarks.multimodal_stability_probe import run_multimodal_stability_probe
from brain.benchmarks.output_region_probe import run_output_region_probe
from brain.benchmarks.phase11_operational_baseline import run_phase11_operational_baseline
from brain.benchmarks.text_binding_probe import run_text_binding_probe
from brain.benchmarks.text_vocab_profile import run_text_vocab_profile
from brain.benchmarks.visual_learning_probe import (
    DEFAULT_VISUAL_LEARNING_PROBE_LABELS,
    run_visual_learning_probe,
)
from brain.benchmarks.coding_assistant_probe import run_coding_assistant_probe
from brain.benchmarks.end_to_end_demo import run_end_to_end_demo

__all__ = [
    "DEFAULT_FIXED_GRAPH_OVERLAY_DELAYS",
    "DEFAULT_FIXED_GRAPH_OVERLAY_REGIONS",
    "DEFAULT_AUDIO_LEARNING_PROBE_LABELS",
    "DEFAULT_ASYNC_MULTI_BRAIN_CORES_PER_WORKER",
    "DEFAULT_ASYNC_MULTI_BRAIN_MERGE_EVERY_SAMPLES",
    "DEFAULT_ASYNC_MULTI_BRAIN_WORKER_COUNT",
    "DEFAULT_CROSSMODAL_RECALL_PROBE_SPEC",
    "DEFAULT_MULTIMODAL_BINDING_PROBE_CATALOG",
    "DEFAULT_MULTIMODAL_BINDING_PROBE_SPEC",
    "DEFAULT_TEXT_LEARNING_PROBE_SPECS",
    "DEFAULT_VISUAL_LEARNING_PROBE_LABELS",
    "FixedGraphSpec",
    "build_fixed_graph_spec",
    "run_audio_learning_probe",
    "run_async_multi_brain_text",
    "run_crossmodal_recall_probe",
    "run_executive_numbers_probe",
    "run_multimodal_binding_probe",
    "run_multimodal_stability_probe",
    "run_output_region_probe",
    "run_phase11_operational_baseline",
    "run_text_binding_probe",
    "run_text_learning_probe",
    "run_text_vocab_profile",
    "run_visual_learning_probe",
    "run_coding_assistant_probe",
    "run_end_to_end_demo",
]