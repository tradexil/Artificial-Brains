"""Phase 11 operational baseline: stability plus output-region quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brain.benchmarks.multimodal_stability_probe import run_multimodal_stability_probe
from brain.benchmarks.output_region_probe import run_output_region_probe


def run_phase11_operational_baseline(
    threads: int,
    output_path: str,
    *,
    stability_samples: int = 120,
    output_probe_samples: int = 32,
    ticks_per_sample: int = 10,
    rest_ticks: int = 1,
    seed_chunks: int | None = 1,
    fast_mode: bool = True,
    n_traces: int = 5500,
    initial_state_dir: str | None = None,
) -> dict[str, Any]:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_file.parent / f"{output_file.stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    seed_mode = "fast" if fast_mode else "full"
    stability_sample_count = min(int(stability_samples), 30) if fast_mode else int(stability_samples)
    output_probe_sample_count = min(int(output_probe_samples), 32)

    stability_path = artifact_dir / f"multimodal_stability_{seed_mode}.json"
    output_probe_path = artifact_dir / f"output_region_{seed_mode}.json"

    stability = run_multimodal_stability_probe(
        max_samples=stability_sample_count,
        ticks_per_sample=ticks_per_sample,
        threads=threads,
        output_path=str(stability_path),
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        full_seed=not fast_mode,
        rest_ticks=rest_ticks,
        initial_state_dir=initial_state_dir,
    )
    output_probe = run_output_region_probe(
        max_samples=output_probe_sample_count,
        ticks_per_sample=ticks_per_sample,
        threads=threads,
        output_path=str(output_probe_path),
        n_traces=n_traces,
        seed_chunks=seed_chunks,
        full_seed=not fast_mode,
        initial_state_dir=initial_state_dir,
    )

    output_summary = output_probe["summary"]
    motor_summary = output_summary["motor"]
    aggregate = {
        "benchmark": "phase11_operational_baseline",
        "seed_mode": seed_mode,
        "config": {
            "threads": int(threads),
            "stability_samples": int(stability_sample_count),
            "output_probe_samples": int(output_probe_sample_count),
            "ticks_per_sample": int(ticks_per_sample),
            "rest_ticks": int(rest_ticks),
            "seed_chunks": seed_chunks,
            "n_traces": int(n_traces),
            "fast_mode": bool(fast_mode),
            "initial_state_dir": initial_state_dir,
        },
        "artifacts": {
            "multimodal_stability": str(stability_path),
            "output_region": str(output_probe_path),
        },
        "reference_numbers": {
            "speech_coverage_correlation": output_summary["speech"]["speech_coverage_correlation"],
            "high_speech_activity_mean": output_summary["speech"]["high_speech_activity_mean"],
            "low_speech_activity_mean": output_summary["speech"]["low_speech_activity_mean"],
            "motor_passing_class_count": motor_summary["passing_class_count"],
            "motor_per_class": {
                label_name: {
                    "between_category_variance": row["between_category_variance"],
                    "within_category_variance": row["within_category_variance"],
                    "motor_activation_mean": row["motor_activation_mean"],
                    "discriminable": row["discriminable"],
                }
                for label_name, row in motor_summary["per_class"].items()
            },
        },
        "validations": {
            "multimodal_stability_passes": all(
                bool(value)
                for value in stability["summary"]["validations"].values()
            ),
            "output_probe_passes": all(
                bool(value)
                for value in output_summary["validations"].values()
            ),
        },
        "multimodal_stability": stability["summary"],
        "output_region": output_summary,
        "performance": {
            "multimodal_processing_wall_ms": float(
                dict(stability.get("performance", {})).get("processing_wall_ms", 0.0) or 0.0
            ),
            "multimodal_total_wall_ms": float(
                dict(stability.get("performance", {})).get("total_wall_ms", 0.0) or 0.0
            ),
            "output_probe_total_wall_ms": float(
                dict(output_probe.get("performance", {})).get("total_wall_ms", 0.0) or 0.0
            ),
        },
    }

    output_file.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate