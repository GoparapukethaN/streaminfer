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


@dataclass(frozen=True)
class BenchmarkGateThresholds:
    max_throughput_drop_pct: float = 10.0
    max_p95_latency_increase_pct: float = 20.0
    max_errors_total: int = 0


def evaluate_benchmark_gate(
    *,
    baseline: dict,
    current: dict,
    thresholds: BenchmarkGateThresholds = BenchmarkGateThresholds(),
) -> dict:
    baseline_rec = baseline.get("recommendation") or {}
    current_rec = current.get("recommendation") or {}
    checks = [
        _percent_drop_check(
            name="throughput_drop_pct",
            baseline_value=baseline_rec.get("throughput_rps"),
            current_value=current_rec.get("throughput_rps"),
            threshold=thresholds.max_throughput_drop_pct,
        ),
        _percent_increase_check(
            name="p95_latency_increase_pct",
            baseline_value=baseline_rec.get("latency_p95_ms"),
            current_value=current_rec.get("latency_p95_ms"),
            threshold=thresholds.max_p95_latency_increase_pct,
        ),
        _current_errors_check(current, thresholds.max_errors_total),
        _recommendation_changed_check(
            _recommendation_id(baseline_rec),
            _recommendation_id(current_rec),
        ),
        _result_count_changed_check(
            len(baseline.get("results") or []),
            len(current.get("results") or []),
        ),
    ]
    return {
        "verdict": _gate_verdict(checks),
        "thresholds": {
            "max_throughput_drop_pct": thresholds.max_throughput_drop_pct,
            "max_p95_latency_increase_pct": thresholds.max_p95_latency_increase_pct,
            "max_errors_total": thresholds.max_errors_total,
        },
        "baseline": _gate_summary(baseline),
        "current": _gate_summary(current),
        "checks": checks,
    }


def render_benchmark_gate_markdown(gate: dict) -> str:
    baseline = gate["baseline"]
    current = gate["current"]
    lines = [
        "# StreamInfer Benchmark Gate",
        "",
        f"**Verdict:** `{gate['verdict']}`",
        "",
        "## Recommendation",
        "",
        "| Field | Baseline | Current |",
        "| --- | ---: | ---: |",
        f"| Config | `{baseline['config_id']}` | `{current['config_id']}` |",
        f"| Throughput | {_format_gate_value(baseline['throughput_rps'])} req/s | "
        f"{_format_gate_value(current['throughput_rps'])} req/s |",
        f"| p95 latency | {_format_gate_value(baseline['latency_p95_ms'])} ms | "
        f"{_format_gate_value(current['latency_p95_ms'])} ms |",
        f"| Total errors | {baseline['errors_total']} | {current['errors_total']} |",
        f"| Result count | {baseline['result_count']} | {current['result_count']} |",
        "",
        "## Checks",
        "",
        "| Check | Status | Observed | Threshold |",
        "| --- | --- | ---: | ---: |",
    ]
    for check in gate["checks"]:
        lines.append(
            f"| `{check['name']}` | `{check['status']}` | "
            f"{_format_gate_value(check.get('observed'))} | "
            f"{_format_gate_value(check.get('threshold'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_benchmark_gate(gate: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".md":
        output.write_text(render_benchmark_gate_markdown(gate), encoding="utf-8")
        return
    output.write_text(json.dumps(gate, indent=2), encoding="utf-8")


def _percent_drop_check(
    *,
    name: str,
    baseline_value: float | None,
    current_value: float | None,
    threshold: float,
) -> dict:
    if baseline_value is None or current_value is None:
        observed = None
        status = "fail"
    else:
        baseline_float = float(baseline_value)
        current_float = float(current_value)
        if baseline_float <= 0:
            observed = 0.0 if current_float >= baseline_float else 100.0
        else:
            observed = round(((baseline_float - current_float) / baseline_float) * 100, 2)
        status = "pass" if observed <= threshold else "fail"
    return {
        "name": name,
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "baseline": baseline_value,
        "current": current_value,
    }


def _percent_increase_check(
    *,
    name: str,
    baseline_value: float | None,
    current_value: float | None,
    threshold: float,
) -> dict:
    if baseline_value is None or current_value is None:
        observed = None
        status = "fail"
    else:
        baseline_float = float(baseline_value)
        current_float = float(current_value)
        if baseline_float <= 0:
            observed = 0.0 if current_float <= baseline_float else 100.0
        else:
            observed = round(((current_float - baseline_float) / baseline_float) * 100, 2)
        status = "pass" if observed <= threshold else "fail"
    return {
        "name": name,
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "baseline": baseline_value,
        "current": current_value,
    }


def _current_errors_check(current: dict, threshold: int) -> dict:
    observed = _errors_total(current)
    return {
        "name": "current_errors_total",
        "status": "pass" if observed <= threshold else "fail",
        "observed": observed,
        "threshold": threshold,
    }


def _recommendation_changed_check(baseline_config_id: str, current_config_id: str) -> dict:
    status = "pass" if baseline_config_id == current_config_id else "warn"
    if baseline_config_id == "n/a" or current_config_id == "n/a":
        status = "fail"
    return {
        "name": "recommendation_changed",
        "status": status,
        "observed": baseline_config_id != current_config_id,
        "threshold": "same recommended config",
        "baseline": baseline_config_id,
        "current": current_config_id,
    }


def _result_count_changed_check(baseline_count: int, current_count: int) -> dict:
    return {
        "name": "result_count_changed",
        "status": "pass" if baseline_count == current_count else "warn",
        "observed": current_count - baseline_count,
        "threshold": 0,
        "baseline": baseline_count,
        "current": current_count,
    }


def _gate_verdict(checks: list[dict]) -> str:
    statuses = {check["status"] for check in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _gate_summary(report: dict) -> dict:
    recommendation = report.get("recommendation") or {}
    return {
        "config_id": _recommendation_id(recommendation),
        "throughput_rps": recommendation.get("throughput_rps"),
        "latency_p95_ms": recommendation.get("latency_p95_ms"),
        "avg_batch_size": recommendation.get("avg_batch_size"),
        "errors_total": _errors_total(report),
        "result_count": len(report.get("results") or []),
    }


def _recommendation_id(recommendation: dict) -> str:
    batch_size = recommendation.get("batch_size")
    timeout_ms = recommendation.get("timeout_ms")
    if batch_size is None or timeout_ms is None:
        return "n/a"
    return f"batch_size={batch_size},timeout_ms={timeout_ms}"


def _errors_total(report: dict) -> int:
    return sum(
        int((result.get("summary") or {}).get("errors_total") or 0)
        for result in report.get("results") or []
    )


def _format_gate_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"`{value}`"


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
