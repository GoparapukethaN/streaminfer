"""Backpressure control for HTTP and WebSocket clients.

Two mechanisms:
  1. Token bucket rate limiter — caps per-client request rate
  2. Pending request guard — rejects clients with too many in-flight requests

Token bucket is standard (see https://en.wikipedia.org/wiki/Token_bucket).
We add tokens at a fixed rate and consume one per request. If the bucket
is empty, the request is rejected.
"""

from __future__ import annotations

import time


class TokenBucket:
    """Classic token bucket rate limiter.

    Tokens refill at `rate` per second, up to `capacity`.
    Call consume() before processing a request — returns False if rate exceeded.
    """

    def __init__(self, rate: float, capacity: float | None = None):
        self.rate = rate
        self.capacity = capacity or rate  # default: 1 second burst
        self.tokens = self.capacity
        self._last_refill = time.monotonic()

    def consume(self, n: int = 1) -> bool:
        """Try to consume n tokens. Returns True if allowed, False if rate-limited."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        # refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def wait_time(self) -> float:
        """How long until 1 token is available (seconds)."""
        if self.tokens >= 1:
            return 0.0
        return (1 - self.tokens) / self.rate


class ClientState:
    """Per-client state tracking for backpressure decisions."""

    def __init__(self, rate_limit: float, max_queue: int):
        self.bucket = TokenBucket(rate=rate_limit)
        self.max_queue = max_queue
        self.pending_count = 0
        self.total_requests = 0
        self.total_rejected = 0

    def can_accept(self) -> bool:
        """Check if we should accept a new request from this client."""
        if self.pending_count >= self.max_queue:
            self.total_rejected += 1
            return False
        if not self.bucket.consume():
            self.total_rejected += 1
            return False
        self.total_requests += 1
        return True

    def on_request_start(self):
        self.pending_count += 1

    def on_request_done(self):
        self.pending_count = max(0, self.pending_count - 1)

    @property
    def is_slow(self) -> bool:
        """Detect slow consumers — if queue is >80% full, they're falling behind."""
        return self.pending_count > self.max_queue * 0.8
