"""Adaptive batcher for inference requests.

Accumulates items and flushes on whichever comes first:
  1. batch_size items collected
  2. timeout_ms elapsed since first item in current batch

This is the same strategy used in TensorFlow Serving and Triton — it balances
throughput (bigger batches) against latency (don't wait forever).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class BatchItem:
    """A single inference request waiting to be batched."""

    data: dict
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.monotonic)


class AdaptiveBatcher:
    """Collects items and flushes as complete batches.

    The process_fn receives a list of data dicts and must return
    a list of results in the same order.
    """

    def __init__(
        self,
        process_fn,
        batch_size: int = 16,
        timeout_ms: int = 50,
    ):
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms

        self._buffer: list[BatchItem] = []
        self._lock = asyncio.Lock()
        self._flush_event = asyncio.Event()
        self._running = False
        self._task: asyncio.Task | None = None

        # stats
        self.total_batches = 0
        self.total_items = 0
        self.total_timeouts = 0  # batches flushed by timeout (not full)

    async def start(self):
        """Start the background flush loop."""
        if self._running:
            return  # already started
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        """Stop the batcher and flush remaining items."""
        self._running = False
        self._flush_event.set()
        if self._task:
            await self._task

        # flush anything left
        async with self._lock:
            if self._buffer:
                await self._process_batch(self._buffer[:])
                self._buffer.clear()

    async def submit(self, data: dict) -> dict:
        """Submit a single item for batched processing.

        Returns the result when the batch containing this item is processed.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = BatchItem(data=data, future=future)

        async with self._lock:
            self._buffer.append(item)
            if len(self._buffer) >= self.batch_size:
                self._flush_event.set()

        return await future

    async def _flush_loop(self):
        """Background loop that flushes batches."""
        while self._running:
            # wait for either: batch is full, or timeout
            try:
                await asyncio.wait_for(
                    self._flush_event.wait(),
                    timeout=self.timeout_ms / 1000.0,
                )
            except asyncio.TimeoutError:
                pass

            self._flush_event.clear()

            async with self._lock:
                if not self._buffer:
                    continue

                batch = self._buffer[: self.batch_size]
                self._buffer = self._buffer[self.batch_size :]

                was_timeout = len(batch) < self.batch_size
                if was_timeout:
                    self.total_timeouts += 1

            await self._process_batch(batch)

    async def _process_batch(self, batch: list[BatchItem]):
        """Run the process function on a batch and resolve futures."""
        data_list = [item.data for item in batch]

        try:
            results = await self.process_fn(data_list)
            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)
        except Exception as e:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)

        self.total_batches += 1
        self.total_items += len(batch)
