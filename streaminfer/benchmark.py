"""Small helpers for summarizing local serving load tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil


@dataclass(frozen=True)
class LoadTestSummary:
    total_requests: int
    errors_total: int
    elapsed_seconds: float
    throughput_rps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def percentile(values: list[float], percentile_rank: float) -> float:
    if not values:
        return 0.0
    if percentile_rank <= 0:
        return round(min(values), 2)
    if percentile_rank >= 100:
        return round(max(values), 2)

    ordered = sorted(values)
    index = max(0, ceil(len(ordered) * percentile_rank / 100) - 1)
    return round(ordered[index], 2)


def summarize_load_test(
    *,
    latencies_ms: list[float],
    elapsed_seconds: float,
    errors_total: int,
) -> LoadTestSummary:
    total_requests = len(latencies_ms)
    throughput = total_requests / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return LoadTestSummary(
        total_requests=total_requests,
        errors_total=errors_total,
        elapsed_seconds=round(elapsed_seconds, 3),
        throughput_rps=round(throughput, 2),
        latency_p50_ms=percentile(latencies_ms, 50),
        latency_p95_ms=percentile(latencies_ms, 95),
        latency_p99_ms=percentile(latencies_ms, 99),
    )
