"""Rich JSON metrics collector for brain learning sessions.

Collects per-tick, per-sample, and global session metrics.
All data stored in a single JSON structure for analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SampleMetrics:
    """Metrics for a single input sample."""

    index: int
    label: str | None = None
    modality: str = ""
    ticks: list[dict[str, Any]] = field(default_factory=list)

    def add_tick(self, tick_result: dict[str, Any]) -> None:
        self.ticks.append(tick_result)

    def summary(self) -> dict[str, Any]:
        if not self.ticks:
            return {}
        n = len(self.ticks)
        # Aggregate numeric fields
        numeric_keys = [
            "total_active", "active_traces", "working_memory", "novelty",
            "hebbian_updates", "anti_hebbian_updates", "learning_multiplier",
            "traces_formed", "bindings_formed",
            "arousal", "valence", "energy",
            "emotion_polarity", "emotion_arousal", "executive_engagement",
            "language_activation", "speech_activity",
            "sensory_activation", "visual_activation", "audio_activation",
            "motor_activation", "motor_approach", "motor_withdraw", "pain_level",
            "sleep_pressure",
        ]
        agg: dict[str, Any] = {}
        for key in numeric_keys:
            vals = [t.get(key, 0) for t in self.ticks if isinstance(t.get(key), (int, float))]
            if vals:
                agg[f"{key}_avg"] = sum(vals) / len(vals)
                agg[f"{key}_max"] = max(vals)
                agg[f"{key}_min"] = min(vals)
                if key in ("hebbian_updates", "anti_hebbian_updates", "traces_formed", "bindings_formed"):
                    agg[f"{key}_total"] = sum(vals)

        # Count-based aggregations
        agg["surprise_ticks"] = sum(1 for t in self.ticks if t.get("in_surprise"))
        agg["alarm_ticks"] = sum(1 for t in self.ticks if t.get("in_alarm"))
        agg["asleep_ticks"] = sum(1 for t in self.ticks if t.get("is_asleep"))
        agg["rem_ticks"] = sum(1 for t in self.ticks if t.get("in_rem"))
        agg["consolidating_ticks"] = sum(1 for t in self.ticks if t.get("consolidating"))

        # Motor action distribution
        actions = [t.get("motor_action", "idle") for t in self.ticks]
        agg["motor_actions"] = {a: actions.count(a) for a in set(actions)}

        # Phase info from last tick
        agg["final_phase"] = self.ticks[-1].get("phase", "unknown")
        agg["tick_count"] = n

        return agg

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "modality": self.modality,
            "ticks": self.ticks,
            "summary": self.summary(),
        }


class MetricsCollector:
    """Collects and aggregates metrics across an entire learning session.

    Usage:
        mc = MetricsCollector(dataset="ag_news", mode="sequential")
        mc.start()
        sample = mc.begin_sample(0, label="World", modality="text")
        for tick_result in tick_results:
            sample.add_tick(tick_result)
        mc.end_sample()
        mc.finish()
        mc.save("metrics.json")
    """

    def __init__(
        self,
        dataset: str = "",
        mode: str = "sequential",
        threads: int = 0,
        ticks_per_sample: int = 10,
        extra_config: dict[str, Any] | None = None,
    ):
        self.session_config = {
            "dataset": dataset,
            "mode": mode,
            "threads": threads,
            "ticks_per_sample": ticks_per_sample,
        }
        if extra_config:
            self.session_config.update(extra_config)

        self.samples: list[SampleMetrics] = []
        self._current_sample: SampleMetrics | None = None
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._tick_times: list[float] = []
        self._sample_times: list[float] = []
        self._sample_start_time: float = 0.0

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def finish(self) -> None:
        self._end_time = time.perf_counter()

    def begin_sample(self, index: int, label: str | None = None, modality: str = "") -> SampleMetrics:
        self._current_sample = SampleMetrics(index=index, label=label, modality=modality)
        self._sample_start_time = time.perf_counter()
        return self._current_sample

    def end_sample(self) -> None:
        if self._current_sample is not None:
            self.samples.append(self._current_sample)
            elapsed = time.perf_counter() - self._sample_start_time
            self._sample_times.append(elapsed)
            self._current_sample = None

    def record_tick_time(self, elapsed: float) -> None:
        self._tick_times.append(elapsed)

    @property
    def duration(self) -> float:
        return self._end_time - self._start_time

    def global_summary(self) -> dict[str, Any]:
        """Compute global metrics across all samples."""
        total_ticks = sum(len(s.ticks) for s in self.samples)
        all_ticks = [t for s in self.samples for t in s.ticks]

        summary: dict[str, Any] = {
            "total_samples": len(self.samples),
            "total_ticks": total_ticks,
            "duration_sec": round(self.duration, 3),
        }

        if total_ticks > 0:
            summary["ticks_per_sec"] = round(total_ticks / max(self.duration, 0.001), 1)

        # Aggregate key metrics across ALL ticks
        for key in ("total_active", "hebbian_updates", "anti_hebbian_updates",
                     "traces_formed", "bindings_formed"):
            vals = [t.get(key, 0) for t in all_ticks if isinstance(t.get(key), (int, float))]
            if vals:
                summary[f"{key}_total"] = sum(vals)
                summary[f"{key}_avg"] = round(sum(vals) / len(vals), 3)

        for key in ("novelty", "arousal", "valence", "energy",
                     "emotion_polarity", "emotion_arousal",
                     "language_activation", "speech_activity",
                     "sensory_activation", "visual_activation", "audio_activation",
                     "motor_activation", "learning_multiplier"):
            vals = [t.get(key, 0) for t in all_ticks if isinstance(t.get(key), (int, float))]
            if vals:
                summary[f"{key}_avg"] = round(sum(vals) / len(vals), 5)
                summary[f"{key}_max"] = round(max(vals), 5)

        # Timing stats
        if self._tick_times:
            summary["tick_time_avg_ms"] = round(1000 * sum(self._tick_times) / len(self._tick_times), 2)
            summary["tick_time_max_ms"] = round(1000 * max(self._tick_times), 2)
            summary["tick_time_min_ms"] = round(1000 * min(self._tick_times), 2)

        if self._sample_times:
            summary["sample_time_avg_ms"] = round(1000 * sum(self._sample_times) / len(self._sample_times), 2)
            summary["sample_time_max_ms"] = round(1000 * max(self._sample_times), 2)

        # State at end
        if all_ticks:
            last = all_ticks[-1]
            summary["final_energy"] = last.get("energy", 0)
            summary["final_arousal"] = last.get("arousal", 0)
            summary["final_valence"] = last.get("valence", 0)
            summary["final_sleep_state"] = last.get("sleep_state", "awake")
            summary["final_sleep_pressure"] = last.get("sleep_pressure", 0)

        return summary

    def to_dict(self) -> dict[str, Any]:
        """Full session data as nested dict."""
        return {
            "session": self.session_config,
            "global_summary": self.global_summary(),
            "samples": [s.to_dict() for s in self.samples],
        }

    def to_dict_compact(self) -> dict[str, Any]:
        """Compact version — sample summaries only, no per-tick data."""
        return {
            "session": self.session_config,
            "global_summary": self.global_summary(),
            "sample_summaries": [
                {"index": s.index, "label": s.label, "modality": s.modality, **s.summary()}
                for s in self.samples
            ],
        }

    def save(self, path: str | Path, compact: bool = False) -> None:
        """Save metrics to JSON file."""
        data = self.to_dict_compact() if compact else self.to_dict()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def print_summary(self) -> None:
        """Print a concise summary to stdout."""
        gs = self.global_summary()
        print(f"\n{'='*60}")
        print(f"SESSION SUMMARY: {self.session_config.get('dataset', '?')} "
              f"({self.session_config.get('mode', '?')})")
        print(f"{'='*60}")
        print(f"  Samples:          {gs.get('total_samples', 0)}")
        print(f"  Total ticks:      {gs.get('total_ticks', 0)}")
        print(f"  Duration:         {gs.get('duration_sec', 0):.1f}s")
        print(f"  Ticks/sec:        {gs.get('ticks_per_sec', 0):.0f}")
        print(f"  Tick avg:         {gs.get('tick_time_avg_ms', 0):.2f}ms")
        print(f"  Hebbian total:    {gs.get('hebbian_updates_total', 0)}")
        print(f"  Traces formed:    {gs.get('traces_formed_total', 0)}")
        print(f"  Bindings formed:  {gs.get('bindings_formed_total', 0)}")
        print(f"  Activity avg:     {gs.get('total_active_avg', 0):.1f}")
        print(f"  Novelty avg:      {gs.get('novelty_avg', 0):.4f}")
        print(f"  Final energy:     {gs.get('final_energy', 0):.3f}")
        print(f"  Final arousal:    {gs.get('final_arousal', 0):.3f}")
        print(f"{'='*60}\n")
