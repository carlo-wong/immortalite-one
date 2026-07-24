"""Opt-in, process-local aggregate profiling for self-play benchmarks."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileCounters:
    """Cheap aggregate counters; callers pass ``None`` when profiling is off."""

    schema_version: int = 1
    seconds: Counter[str] = field(default_factory=Counter)
    counts: Counter[str] = field(default_factory=Counter)
    bytes: Counter[str] = field(default_factory=Counter)
    histograms: dict[str, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def add_seconds(self, name: str, value: float) -> None:
        self.seconds[name] += float(value)

    def add_count(self, name: str, value: int = 1) -> None:
        self.counts[name] += int(value)

    def add_bytes(self, name: str, value: int) -> None:
        self.bytes[name] += int(value)

    def observe(self, name: str, value: int, weight: int = 1) -> None:
        self.histograms[name][int(value)] += int(weight)

    def merge(self, other: ProfileCounters | dict[str, Any]) -> None:
        data = other.snapshot() if isinstance(other, ProfileCounters) else other
        self.seconds.update({k: float(v) for k, v in data.get("seconds", {}).items()})
        self.counts.update({k: int(v) for k, v in data.get("counts", {}).items()})
        self.bytes.update({k: int(v) for k, v in data.get("bytes", {}).items()})
        for name, values in data.get("histograms", {}).items():
            self.histograms[name].update({int(k): int(v) for k, v in values.items()})

    def snapshot(self) -> dict[str, Any]:
        def percentile(values: Counter[int], fraction: float) -> int:
            target = max(1, int(sum(values.values()) * fraction + 0.999999))
            cumulative = 0
            for value, count in sorted(values.items()):
                cumulative += count
                if cumulative >= target:
                    return value
            return max(values)

        return {
            "schema_version": self.schema_version,
            "seconds": dict(sorted(self.seconds.items())),
            "counts": dict(sorted(self.counts.items())),
            "bytes": dict(sorted(self.bytes.items())),
            "histograms": {
                name: {str(k): v for k, v in sorted(values.items())}
                for name, values in sorted(self.histograms.items())
            },
            "histogram_percentiles": {
                name: {
                    "p10": percentile(values, 0.10),
                    "p50": percentile(values, 0.50),
                    "p90": percentile(values, 0.90),
                }
                for name, values in sorted(self.histograms.items())
                if values
            },
        }
