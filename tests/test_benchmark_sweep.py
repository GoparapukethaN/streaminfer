import pytest

from streaminfer.benchmark import render_markdown_report, run_inference_sweep


@pytest.mark.asyncio
async def test_inference_sweep_compares_batching_configs() -> None:
    report = await run_inference_sweep(
        prompts=[
            "small prompt",
            "explain batch size",
            "why does p95 matter",
            "how should I compare serving configs",
            "another request",
            "latency budget",
        ],
        batch_sizes=[1, 3],
        timeout_ms_values=[1, 20],
        concurrency=3,
        model_latency_ms=0,
        per_token_latency_ms=0,
    )

    assert report["config"]["total_prompts"] == 6
    assert len(report["results"]) == 4
    assert report["recommendation"]["batch_size"] in {1, 3}
    assert report["recommendation"]["timeout_ms"] in {1, 20}
    assert all(result["summary"]["errors_total"] == 0 for result in report["results"])
    assert all(result["summary"]["total_requests"] == 6 for result in report["results"])
    assert all("avg_batch_size" in result["batcher"] for result in report["results"])


@pytest.mark.asyncio
async def test_markdown_report_includes_recommendation_and_result_table() -> None:
    report = await run_inference_sweep(
        prompts=["one", "two", "three"],
        batch_sizes=[1],
        timeout_ms_values=[1],
        concurrency=2,
        model_latency_ms=0,
        per_token_latency_ms=0,
    )

    markdown = render_markdown_report(report)

    assert "# StreamInfer Benchmark Sweep" in markdown
    assert "## Recommendation" in markdown
    assert "| batch_size | timeout_ms | throughput_rps | latency_p95_ms | avg_batch_size |" in markdown
