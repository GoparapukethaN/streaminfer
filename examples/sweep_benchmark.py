"""Run a deterministic local benchmark sweep for StreamInfer batching settings."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from streaminfer.benchmark import (
    BenchmarkGateThresholds,
    evaluate_benchmark_gate,
    render_markdown_report,
    run_inference_sweep,
    write_benchmark_gate,
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
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--gate-json", type=Path, default=Path("artifacts/inference-gate.json"))
    parser.add_argument("--gate-md", type=Path, default=Path("artifacts/inference-gate.md"))
    parser.add_argument("--max-throughput-drop-pct", type=float, default=10.0)
    parser.add_argument("--max-p95-latency-increase-pct", type=float, default=20.0)
    parser.add_argument("--max-errors-total", type=int, default=0)
    args = parser.parse_args()

    report = await run_inference_sweep(
        prompts=DEFAULT_PROMPTS,
        batch_sizes=parse_int_list(args.batch_sizes),
        timeout_ms_values=parse_int_list(args.timeouts_ms),
        concurrency=args.concurrency,
    )
    baseline = (
        json.loads(args.baseline_json.read_text(encoding="utf-8"))
        if args.baseline_json
        else None
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

    if baseline is not None:
        gate = evaluate_benchmark_gate(
            baseline=baseline,
            current=report,
            thresholds=BenchmarkGateThresholds(
                max_throughput_drop_pct=args.max_throughput_drop_pct,
                max_p95_latency_increase_pct=args.max_p95_latency_increase_pct,
                max_errors_total=args.max_errors_total,
            ),
        )
        write_benchmark_gate(gate, args.gate_json)
        write_benchmark_gate(gate, args.gate_md)
        verdict = gate["verdict"]
        if verdict == "pass":
            print("Benchmark gate passed.")
        elif verdict == "warn":
            print("Benchmark gate passed with warnings.")
        else:
            print("Benchmark gate failed.")
        print(f"gate json: {args.gate_json}")
        print(f"gate markdown: {args.gate_md}")
        if verdict == "fail":
            raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
