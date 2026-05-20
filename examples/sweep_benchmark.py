"""Run a deterministic local benchmark sweep for StreamInfer batching settings."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from streaminfer.benchmark import (
    render_markdown_report,
    run_inference_sweep,
    write_sweep_report,
)

DEFAULT_PROMPTS = [
    "summarize the current serving latency trend",
    "explain why batching can increase throughput",
    "classify this request as latency sensitive",
    "write a short incident summary for a model timeout",
    "compare p50 and p95 latency for an inference endpoint",
    "describe the rollback plan for a slow model version",
    "extract the key metric from this monitoring note",
    "draft a response for an on-call model alert",
]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--timeouts-ms", default="5,25,50")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/inference-sweep.json"))
    parser.add_argument("--output-md", type=Path, default=Path("artifacts/inference-sweep.md"))
    args = parser.parse_args()

    report = await run_inference_sweep(
        prompts=DEFAULT_PROMPTS,
        batch_sizes=parse_int_list(args.batch_sizes),
        timeout_ms_values=parse_int_list(args.timeouts_ms),
        concurrency=args.concurrency,
    )
    write_sweep_report(report, args.output_json)
    write_sweep_report(report, args.output_md)

    recommendation = report["recommendation"]
    print(render_markdown_report(report))
    print(
        "recommended: batch_size={batch_size}, timeout_ms={timeout_ms}, "
        "throughput={throughput_rps} req/s, p95={latency_p95_ms} ms".format(
            **recommendation
        )
    )
    print(f"json: {args.output_json}")
    print(f"markdown: {args.output_md}")


if __name__ == "__main__":
    asyncio.run(main())
