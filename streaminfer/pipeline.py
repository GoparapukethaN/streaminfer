"""Inference pipeline — chains model inference with the batcher.

The pipeline is the glue between the WebSocket handler and the batcher.
It handles: model lookup, batch processing, and metrics recording.
"""

from __future__ import annotations

import time

from .batcher import AdaptiveBatcher
from .hotswap import ModelHolder
from .metrics import Metrics


class InferencePipeline:
    """Manages the model + batcher lifecycle."""

    def __init__(self, model_holder: ModelHolder, metrics: Metrics, **batcher_kwargs):
        self.model_holder = model_holder
        self.metrics = metrics
        self.batcher = AdaptiveBatcher(
            process_fn=self._run_inference,
            **batcher_kwargs,
        )

    async def start(self):
        await self.batcher.start()

    async def stop(self):
        await self.batcher.stop()

    async def predict(self, data: dict) -> dict:
        """Submit a single prediction request through the batcher."""
        self.metrics.record_request()
        t0 = time.monotonic()

        try:
            result = await self.batcher.submit(data)
            latency_ms = (time.monotonic() - t0) * 1000
            self.metrics.record_latency(latency_ms)
            return result
        except Exception:
            self.metrics.record_error()
            raise

    async def _run_inference(self, batch: list[dict]) -> list[dict]:
        """Process a batch through the current model."""
        model = self.model_holder.model
        self.metrics.record_batch(len(batch))

        # model.predict can be sync or async — handle both
        result = model.predict(batch)
        return result
