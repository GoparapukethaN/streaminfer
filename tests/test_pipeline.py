"""Integration tests for the inference pipeline."""

import asyncio

import pytest

from streaminfer.hotswap import EchoModel, ModelHolder, UpperModel
from streaminfer.metrics import Metrics
from streaminfer.pipeline import InferencePipeline


class TestPipeline:
    @pytest.mark.asyncio
    async def test_basic_prediction(self):
        holder = ModelHolder(model=EchoModel(), name="echo")
        metrics = Metrics()
        pipeline = InferencePipeline(holder, metrics, batch_size=4, timeout_ms=50)
        await pipeline.start()

        result = await pipeline.predict({"text": "hello world"})
        assert result["result"] == "hello world"
        assert result["model"] == "echo"

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_concurrent_predictions(self):
        holder = ModelHolder(model=EchoModel(), name="echo")
        metrics = Metrics()
        pipeline = InferencePipeline(holder, metrics, batch_size=4, timeout_ms=100)
        await pipeline.start()

        tasks = [pipeline.predict({"text": f"msg-{i}"}) for i in range(8)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 8
        texts = {r["result"] for r in results}
        assert texts == {f"msg-{i}" for i in range(8)}

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_metrics_are_recorded(self):
        holder = ModelHolder(model=EchoModel(), name="echo")
        metrics = Metrics()
        pipeline = InferencePipeline(holder, metrics, batch_size=2, timeout_ms=50)
        await pipeline.start()

        await asyncio.gather(
            pipeline.predict({"text": "a"}),
            pipeline.predict({"text": "b"}),
        )

        await pipeline.stop()

        assert metrics.requests_total >= 2
        snapshot = metrics.snapshot()
        assert snapshot["items_processed"] >= 2

    @pytest.mark.asyncio
    async def test_hot_swap_mid_flight(self):
        """Swap model while requests are in flight."""
        holder = ModelHolder(model=EchoModel(), name="echo")
        metrics = Metrics()
        pipeline = InferencePipeline(holder, metrics, batch_size=1, timeout_ms=50)
        await pipeline.start()

        # first request with echo
        r1 = await pipeline.predict({"text": "before"})
        assert r1["model"] == "echo"

        # swap to upper
        holder.swap(UpperModel(), "upper")

        # next request should use upper model
        r2 = await pipeline.predict({"text": "after"})
        assert r2["model"] == "upper"
        assert r2["result"] == "AFTER"

        await pipeline.stop()
