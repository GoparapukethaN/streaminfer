"""Example WebSocket client that streams text to the server.

Usage:
    # start the server first:
    python -m streaminfer.server

    # then run this client:
    python examples/client.py
"""

import asyncio
import json
import time

import websockets


async def main():
    uri = "ws://localhost:8000/ws"
    messages = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming how we build software",
        "Real-time inference requires careful attention to latency",
        "Adaptive batching balances throughput and response time",
        "Backpressure prevents slow consumers from overwhelming the server",
    ]

    async with websockets.connect(uri) as ws:
        print(f"connected to {uri}\n")

        for i, text in enumerate(messages):
            payload = json.dumps({"text": text, "id": i})
            t0 = time.monotonic()

            await ws.send(payload)
            response = await ws.recv()
            result = json.loads(response)

            latency = (time.monotonic() - t0) * 1000
            print(f"[{latency:6.1f}ms] {text[:50]}...")
            print(f"         → {result}\n")

            # simulate streaming delay
            await asyncio.sleep(0.1)

    print("done")


if __name__ == "__main__":
    asyncio.run(main())
