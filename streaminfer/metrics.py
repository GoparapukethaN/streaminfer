"""Simple in-memory metrics.

No Prometheus, no StatsD, no external deps. Just counters and a /metrics
endpoint that returns JSON. Good enough for most deployments — if you need
more, pipe the JSON to your monitoring system.
"""

from __future__ import annotations

import time
import threading


class Metrics:
    """Thread-safe in-memory counters."""

    def __init__(self):
        self._lock = threading.Lock()
        self._started_at = time.time()
        self.requests_total = 0
        self.requests_rejected = 0
        self.batches_total = 0
        self.items_processed = 0
        self.errors_total = 0
        self.connections_active = 0
        self.connections_total = 0

        # latency tracking (last 1000 requests)
        self._latencies: list[float] = []
        self._max_latencies = 1000

    def record_request(self):
        with self._lock:
            self.requests_total += 1

    def record_rejection(self):
        with self._lock:
            self.requests_rejected += 1

    def record_batch(self, size: int):
        with self._lock:
            self.batches_total += 1
            self.items_processed += size

    def record_error(self):
        with self._lock:
            self.errors_total += 1

    def record_latency(self, ms: float):
        with self._lock:
            self._latencies.append(ms)
            if len(self._latencies) > self._max_latencies:
                self._latencies = self._latencies[-self._max_latencies:]

    def record_connect(self):
        with self._lock:
            self.connections_active += 1
            self.connections_total += 1

    def record_disconnect(self):
        with self._lock:
            self.connections_active = max(0, self.connections_active - 1)

    def snapshot(self) -> dict:
        """Return current metrics as a dict."""
        with self._lock:
            latencies = sorted(self._latencies) if self._latencies else [0]
            n = len(latencies)
            return {
                "uptime_seconds": round(time.time() - self._started_at, 1),
                "requests_total": self.requests_total,
                "requests_rejected": self.requests_rejected,
                "batches_total": self.batches_total,
                "items_processed": self.items_processed,
                "errors_total": self.errors_total,
                "connections_active": self.connections_active,
                "connections_total": self.connections_total,
                "latency_p50_ms": round(latencies[n // 2], 2) if n else 0,
                "latency_p95_ms": round(latencies[int(n * 0.95)], 2) if n else 0,
                "latency_p99_ms": round(latencies[int(n * 0.99)], 2) if n else 0,
                "avg_batch_size": (
                    round(self.items_processed / self.batches_total, 1)
                    if self.batches_total
                    else 0
                ),
            }
