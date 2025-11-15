"""Tests for adaptive batching logic."""

import asyncio

import pytest

from streaminfer.batcher import AdaptiveBatcher


async def echo_process(batch):
    """Simple process function that returns inputs as-is."""
    return [{"echo": item["text"]} for item in batch]


async def slow_process(batch):
    """Simulates a slow model — 50ms per batch."""
    await asyncio.sleep(0.05)
    return [{"slow": item["text"]} for item in batch]


class TestAdaptiveBatcher:
    @pytest.mark.asyncio
    async def test_single_item_flushes_on_timeout(self):
        batcher = AdaptiveBatcher(echo_process, batch_size=10, timeout_ms=50)
        await batcher.start()

        result = await batcher.submit({"text": "hello"})
        assert result == {"echo": "hello"}
        assert batcher.total_timeouts >= 1  # flushed by timeout, not by size

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_full_batch_flushes_immediately(self):
        batcher = AdaptiveBatcher(echo_process, batch_size=3, timeout_ms=5000)
        await batcher.start()

        # send 3 items concurrently — should flush as soon as batch is full
        results = await asyncio.gather(
            batcher.submit({"text": "a"}),
            batcher.submit({"text": "b"}),
            batcher.submit({"text": "c"}),
        )

        assert len(results) == 3
        assert all("echo" in r for r in results)

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_stats_are_tracked(self):
        batcher = AdaptiveBatcher(echo_process, batch_size=2, timeout_ms=50)
        await batcher.start()

        await asyncio.gather(
            batcher.submit({"text": "a"}),
            batcher.submit({"text": "b"}),
        )

        await batcher.stop()

        assert batcher.total_items >= 2
        assert batcher.total_batches >= 1

    @pytest.mark.asyncio
    async def test_process_error_propagates(self):
        async def failing_process(batch):
            raise ValueError("model crashed")

        batcher = AdaptiveBatcher(failing_process, batch_size=1, timeout_ms=50)
        await batcher.start()

        with pytest.raises(ValueError, match="model crashed"):
            await batcher.submit({"text": "boom"})

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_concurrent_submissions(self):
        batcher = AdaptiveBatcher(slow_process, batch_size=5, timeout_ms=100)
        await batcher.start()

        # fire 10 requests — should be processed in 2 batches
        tasks = [batcher.submit({"text": f"item-{i}"}) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all("slow" in r for r in results)

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_stop_flushes_remaining(self):
        batcher = AdaptiveBatcher(echo_process, batch_size=100, timeout_ms=5000)
        await batcher.start()

        # submit without waiting for full batch
        task = asyncio.create_task(batcher.submit({"text": "leftover"}))

        # small delay to let it enter the buffer
        await asyncio.sleep(0.01)
        await batcher.stop()

        result = await task
        assert result == {"echo": "leftover"}
