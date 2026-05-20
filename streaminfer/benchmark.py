"""Small helpers for summarizing local serving and benchmark runs."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

from .hotswap import ModelHolder
from .metrics import Metrics
from .pipeline import InferencePipeline
from .synthetic_llm import SyntheticLLMModel


@dataclass(frozen=True)
class LoadTestSummary:
    total_requests: int
    errors_total: int
    elapsed_seconds: float
    throughput_rps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def percentile(values: list[float], percentile_rank: float) -> float:
    if not values:
        return 0.0
    if percentile_rank <= 0:
        return round(min(values), 2)
    if percentile_rank >= 100:
        return round(max(values), 2)

    ordered = sorted(values)
    index = max(0, ceil(len(ordered) * percentile_rank / 100) - 1)
    return round(ordered[index], 2)


def summarize_load_test(
    *,
    latencies_ms: list[float],
    elapsed_seconds: float,
    errors_total: int,
) -> LoadTestSummary:
    total_requests = len(latencies_ms)
    throughput = total_requests / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return LoadTestSummary(
        total_requests=total_requests,
        errors_total=errors_total,
        elapsed_seconds=round(elapsed_seconds, 3),
        throughput_rps=round(throughput, 2),
        latency_p50_ms=percentile(latencies_ms, 50),
        latency_p95_ms=percentile(latencies_ms, 95),
        latency_p99_ms=percentile(latencies_ms, 99),
    )


async def run_inference_sweep(
    *,
    prompts: list[str],
    batch_sizes: list[int],
    timeout_ms_values: list[int],
    concurrency: int = 8,
    model_latency_ms: float = 8,
    per_token_latency_ms: float = 0.25,
    target_p95_ms: float | None = None,
) -> dict:
    """Compare batching configurations with a deterministic local LLM-style model."""
    results = []
    for batch_size in batch_sizes:
        for timeout_ms in timeout_ms_values:
            result = await _run_single_config(
                prompts=prompts,
                batch_size=batch_size,
                timeout_ms=timeout_ms,
                concurrency=concurrency,
                model_latency_ms=model_latency_ms,
                per_token_latency_ms=per_token_latency_ms,
            )
            results.append(result)

    recommendation = _choose_recommendation(results, target_p95_ms=target_p95_ms)
    return {
        "config": {
            "total_prompts": len(prompts),
            "batch_sizes": batch_sizes,
            "timeout_ms_values": timeout_ms_values,
            "concurrency": concurrency,
            "model": "synthetic-llm",
            "model_latency_ms": model_latency_ms,
            "per_token_latency_ms": per_token_latency_ms,
            "target_p95_ms": target_p95_ms,
        },
        "recommendation": recommendation,
        "results": results,
    }


def render_markdown_report(report: dict) -> str:
    """Render a benchmark sweep as a compact Markdown artifact."""
    recommendation = report["recommendation"]
    lines = [
        "# StreamInfer Benchmark Sweep",
        "",
        (
            "This report compares local batching settings with the deterministic "
            "`synthetic-llm` profile. The numbers are useful for regression checks "
            "and relative tradeoffs on this machine, not as universal serving "
            "benchmarks."
        ),
        "",
        "## Recommendation",
        "",
        (
            f"- Batch size `{recommendation['batch_size']}` with timeout "
            f"`{recommendation['timeout_ms']}ms`."
        ),
        f"- Throughput: `{recommendation['throughput_rps']}` req/s.",
        f"- p95 latency: `{recommendation['latency_p95_ms']}` ms.",
        "",
        "## Results",
        "",
        "| batch_size | timeout_ms | throughput_rps | latency_p95_ms | avg_batch_size |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        summary = result["summary"]
        batcher = result["batcher"]
        lines.append(
            "| {batch_size} | {timeout_ms} | {throughput} | {p95} | {avg_batch} |".format(
                batch_size=result["batch_size"],
                timeout_ms=result["timeout_ms"],
                throughput=summary["throughput_rps"],
                p95=summary["latency_p95_ms"],
                avg_batch=batcher["avg_batch_size"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_sweep_report(report: dict, output: Path) -> None:
    """Write a benchmark sweep report as JSON or Markdown based on file suffix."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".md":
        output.write_text(render_markdown_report(report), encoding="utf-8")
        return
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


async def _run_single_config(
    *,
    prompts: list[str],
    batch_size: int,
    timeout_ms: int,
    concurrency: int,
    model_latency_ms: float,
    per_token_latency_ms: float,
) -> dict:
    model = SyntheticLLMModel(
        base_latency_ms=model_latency_ms,
        per_token_latency_ms=per_token_latency_ms,
    )
    metrics = Metrics()
    pipeline = InferencePipeline(
        ModelHolder(model=model, name="synthetic-llm"),
        metrics,
        batch_size=batch_size,
        timeout_ms=timeout_ms,
    )
    semaphore = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []
    errors: list[str] = []

    await pipeline.start()
    started_at = time.monotonic()
    try:
        await asyncio.gather(
            *[
                _run_prompt(
                    pipeline=pipeline,
                    prompt=prompt,
                    semaphore=semaphore,
                    latencies_ms=latencies_ms,
                    errors=errors,
                )
                for prompt in prompts
            ]
        )
    finally:
        elapsed_seconds = time.monotonic() - started_at
        await pipeline.stop()

    summary = summarize_load_test(
        latencies_ms=latencies_ms,
        elapsed_seconds=elapsed_seconds,
        errors_total=len(errors),
    )
    avg_batch_size = (
        round(pipeline.batcher.total_items / pipeline.batcher.total_batches, 2)
        if pipeline.batcher.total_batches
        else 0
    )
    return {
        "batch_size": batch_size,
        "timeout_ms": timeout_ms,
        "summary": summary.to_dict(),
        "batcher": {
            "total_batches": pipeline.batcher.total_batches,
            "total_items": pipeline.batcher.total_items,
            "total_timeouts": pipeline.batcher.total_timeouts,
            "avg_batch_size": avg_batch_size,
        },
        "sample_errors": errors[:5],
    }


async def _run_prompt(
    *,
    pipeline: InferencePipeline,
    prompt: str,
    semaphore: asyncio.Semaphore,
    latencies_ms: list[float],
    errors: list[str],
) -> None:
    async with semaphore:
        started_at = time.monotonic()
        try:
            await pipeline.predict({"prompt": prompt})
            latencies_ms.append((time.monotonic() - started_at) * 1000)
        except Exception as exc:
            errors.append(str(exc))


def _choose_recommendation(results: list[dict], *, target_p95_ms: float | None) -> dict:
    candidates = [result for result in results if result["summary"]["errors_total"] == 0]
    if target_p95_ms is not None:
        under_target = [
            result
            for result in candidates
            if result["summary"]["latency_p95_ms"] <= target_p95_ms
        ]
        if under_target:
            candidates = under_target
    if not candidates:
        candidates = results

    best = sorted(
        candidates,
        key=lambda result: (
            -result["summary"]["throughput_rps"],
            result["summary"]["latency_p95_ms"],
            -result["batcher"]["avg_batch_size"],
        ),
    )[0]
    return {
        "batch_size": best["batch_size"],
        "timeout_ms": best["timeout_ms"],
        "throughput_rps": best["summary"]["throughput_rps"],
        "latency_p95_ms": best["summary"]["latency_p95_ms"],
        "avg_batch_size": best["batcher"]["avg_batch_size"],
    }
