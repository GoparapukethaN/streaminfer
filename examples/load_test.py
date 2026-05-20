"""Load test that fires concurrent WebSocket requests at the server.

Usage:
    python examples/load_test.py --connections 100 --requests 50 --output artifacts/load-test.json

Prints throughput and latency percentiles and can write a JSON report.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import websockets

from streaminfer.benchmark import summarize_load_test


async def run_client(
    *,
    client_id: int,
    n_requests: int,
    uri: str,
    latencies_ms: list[float],
    errors: list[str],
) -> None:
    """Single client that sends n_requests and records latencies."""
    try:
        async with websockets.connect(uri) as ws:
            for i in range(n_requests):
                payload = json.dumps({"text": f"client {client_id} request {i}", "id": i})
                t0 = time.monotonic()
                await ws.send(payload)
                await ws.recv()
                latencies_ms.append((time.monotonic() - t0) * 1000)
    except Exception as e:
        errors.append(f"client {client_id}: {e}")


async def main(
    *,
    n_connections: int,
    n_requests: int,
    uri: str,
    output: Path | None,
) -> None:
    latencies_ms: list[float] = []
    errors: list[str] = []
    started_at = time.monotonic()

    tasks = [
        run_client(
            client_id=i,
            n_requests=n_requests,
            uri=uri,
            latencies_ms=latencies_ms,
            errors=errors,
        )
        for i in range(n_connections)
    ]
    await asyncio.gather(*tasks)

    elapsed_seconds = time.monotonic() - started_at
    summary = summarize_load_test(
        latencies_ms=latencies_ms,
        elapsed_seconds=elapsed_seconds,
        errors_total=len(errors),
    )

    if not latencies_ms:
        print("no successful requests -- is the server running?")
        if errors:
            print("\n".join(errors[:5]))
        return

    report = {
        "config": {
            "uri": uri,
            "connections": n_connections,
            "requests_per_connection": n_requests,
        },
        "summary": summary.to_dict(),
        "sample_errors": errors[:10],
    }

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'=' * 50}")
    print(f"connections:  {n_connections}")
    print(f"requests:     {n_requests} per connection")
    print(f"total:        {summary.total_requests} requests in {summary.elapsed_seconds:.1f}s")
    print(f"errors:       {summary.errors_total}")
    print(f"throughput:   {summary.throughput_rps:.1f} req/s")
    print(f"latency p50:  {summary.latency_p50_ms:.1f}ms")
    print(f"latency p95:  {summary.latency_p95_ms:.1f}ms")
    print(f"latency p99:  {summary.latency_p99_ms:.1f}ms")
    if output:
        print(f"report:       {output}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--connections", type=int, default=50)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--uri", default="ws://localhost:8000/ws")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    asyncio.run(
        main(
            n_connections=args.connections,
            n_requests=args.requests,
            uri=args.uri,
            output=args.output,
        )
    )
