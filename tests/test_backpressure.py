"""Tests for backpressure control."""

import time

from streaminfer.backpressure import ClientState, TokenBucket


class TestTokenBucket:
    def test_allows_burst_up_to_capacity(self):
        bucket = TokenBucket(rate=10, capacity=5)
        # should allow 5 in a row
        results = [bucket.consume() for _ in range(5)]
        assert all(results)

    def test_rejects_when_empty(self):
        bucket = TokenBucket(rate=10, capacity=2)
        bucket.consume()
        bucket.consume()
        assert not bucket.consume()

    def test_refills_over_time(self):
        bucket = TokenBucket(rate=1000, capacity=1)
        bucket.consume()  # empty the bucket
        assert not bucket.consume()

        # manually advance time by faking the refill
        bucket._last_refill = time.monotonic() - 0.01  # 10ms ago, rate=1000/s = 10 tokens
        assert bucket.consume()

    def test_wait_time_when_empty(self):
        bucket = TokenBucket(rate=10, capacity=1)
        bucket.consume()
        wait = bucket.wait_time()
        assert wait > 0
        assert wait <= 0.2  # should be ~0.1s at rate=10

    def test_wait_time_when_available(self):
        bucket = TokenBucket(rate=10, capacity=5)
        assert bucket.wait_time() == 0.0


class TestClientState:
    def test_accepts_within_limits(self):
        client = ClientState(rate_limit=100, max_queue=10)
        assert client.can_accept()
        assert client.total_requests == 1

    def test_rejects_when_queue_full(self):
        client = ClientState(rate_limit=10000, max_queue=2)
        client.on_request_start()
        client.on_request_start()
        # queue is now at 2/2 = full
        assert not client.can_accept()
        assert client.total_rejected == 1

    def test_detects_slow_consumer(self):
        client = ClientState(rate_limit=10000, max_queue=10)
        # fill queue to >80%
        for _ in range(9):
            client.on_request_start()
        assert client.is_slow

    def test_not_slow_when_queue_low(self):
        client = ClientState(rate_limit=10000, max_queue=10)
        client.on_request_start()
        assert not client.is_slow

    def test_request_done_decrements(self):
        client = ClientState(rate_limit=10000, max_queue=10)
        client.on_request_start()
        client.on_request_start()
        assert client.pending_count == 2
        client.on_request_done()
        assert client.pending_count == 1
