"""Load test — fires N concurrent WebSocket connections at the server.

Usage:
    python examples/load_test.py --connections 100 --requests 50

Prints throughput and latency percentiles when done.
"""

import argparse
import asyncio
import json
import time

import websockets


async def run_client(client_id: int, n_requests: int, results: list):
    """Single client that sends n_requests and records latencies."""
    uri = "ws://localhost:8000/ws"
    latencies = []

    try:
        async with websockets.connect(uri) as ws:
            for i in range(n_requests):
                payload = json.dumps({"text": f"client {client_id} request {i}", "id": i})
                t0 = time.monotonic()
                await ws.send(payload)
                await ws.recv()
                latencies.append((time.monotonic() - t0) * 1000)
    except Exception as e:
        print(f"client {client_id} error: {e}")

    results.extend(latencies)


async def main(n_connections: int, n_requests: int):
    results: list[float] = []
    t0 = time.monotonic()

    tasks = [
        run_client(i, n_requests, results) for i in range(n_connections)
    ]
    await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t0
    total = len(results)

    if not results:
        print("no results — is the server running?")
        return

    results.sort()
    rps = total / elapsed

    print(f"\n{'='*50}")
    print(f"connections:  {n_connections}")
    print(f"requests:     {n_requests} per connection")
    print(f"total:        {total} requests in {elapsed:.1f}s")
    print(f"throughput:   {rps:.0f} req/s")
    print(f"latency p50:  {results[len(results)//2]:.1f}ms")
    print(f"latency p95:  {results[int(len(results)*0.95)]:.1f}ms")
    print(f"latency p99:  {results[int(len(results)*0.99)]:.1f}ms")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--connections", type=int, default=50)
    parser.add_argument("--requests", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args.connections, args.requests))
