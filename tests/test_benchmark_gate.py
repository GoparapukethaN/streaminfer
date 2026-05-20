from pathlib import Path

from streaminfer.benchmark import (
    BenchmarkGateThresholds,
    evaluate_benchmark_gate,
    render_benchmark_gate_markdown,
    write_benchmark_gate,
)


def _report(
    *,
    batch_size: int = 8,
    timeout_ms: int = 5,
    throughput_rps: float = 400.0,
    latency_p95_ms: float = 20.0,
    errors_total: int = 0,
    result_count: int = 2,
) -> dict:
    return {
        "config": {
            "total_prompts": 8,
            "batch_sizes": [1, batch_size],
            "timeout_ms_values": [timeout_ms],
            "concurrency": 8,
            "model": "synthetic-llm",
        },
        "recommendation": {
            "batch_size": batch_size,
            "timeout_ms": timeout_ms,
            "throughput_rps": throughput_rps,
            "latency_p95_ms": latency_p95_ms,
            "avg_batch_size": float(batch_size),
        },
        "results": [
            {
                "batch_size": batch_size,
                "timeout_ms": timeout_ms,
                "summary": {
                    "total_requests": 8,
                    "errors_total": errors_total if index == 0 else 0,
                    "throughput_rps": throughput_rps,
                    "latency_p95_ms": latency_p95_ms,
                },
                "batcher": {"avg_batch_size": float(batch_size)},
                "sample_errors": ["timeout"] if errors_total and index == 0 else [],
            }
            for index in range(result_count)
        ],
    }


def test_benchmark_gate_passes_when_current_stays_inside_thresholds() -> None:
    gate = evaluate_benchmark_gate(
        baseline=_report(throughput_rps=400.0, latency_p95_ms=20.0),
        current=_report(throughput_rps=380.0, latency_p95_ms=23.0),
        thresholds=BenchmarkGateThresholds(
            max_throughput_drop_pct=10.0,
            max_p95_latency_increase_pct=20.0,
            max_errors_total=0,
        ),
    )

    assert gate["verdict"] == "pass"
    assert {check["name"]: check["status"] for check in gate["checks"]} == {
        "throughput_drop_pct": "pass",
        "p95_latency_increase_pct": "pass",
        "current_errors_total": "pass",
        "recommendation_changed": "pass",
        "result_count_changed": "pass",
    }


def test_benchmark_gate_fails_on_throughput_latency_or_errors() -> None:
    gate = evaluate_benchmark_gate(
        baseline=_report(throughput_rps=400.0, latency_p95_ms=20.0),
        current=_report(throughput_rps=300.0, latency_p95_ms=30.0, errors_total=1),
        thresholds=BenchmarkGateThresholds(
            max_throughput_drop_pct=10.0,
            max_p95_latency_increase_pct=20.0,
            max_errors_total=0,
        ),
    )

    checks = {check["name"]: check for check in gate["checks"]}
    assert gate["verdict"] == "fail"
    assert checks["throughput_drop_pct"]["status"] == "fail"
    assert checks["p95_latency_increase_pct"]["status"] == "fail"
    assert checks["current_errors_total"]["status"] == "fail"
    assert checks["throughput_drop_pct"]["observed"] == 25.0
    assert checks["p95_latency_increase_pct"]["observed"] == 50.0


def test_benchmark_gate_warns_when_recommended_config_changes_without_regression() -> None:
    gate = evaluate_benchmark_gate(
        baseline=_report(batch_size=8, timeout_ms=5, throughput_rps=400.0, latency_p95_ms=20.0),
        current=_report(batch_size=4, timeout_ms=25, throughput_rps=405.0, latency_p95_ms=19.0),
    )

    checks = {check["name"]: check for check in gate["checks"]}
    assert gate["verdict"] == "warn"
    assert checks["recommendation_changed"]["status"] == "warn"
    assert checks["recommendation_changed"]["baseline"] == "batch_size=8,timeout_ms=5"
    assert checks["recommendation_changed"]["current"] == "batch_size=4,timeout_ms=25"


def test_benchmark_gate_markdown_and_json_outputs(tmp_path: Path) -> None:
    gate = evaluate_benchmark_gate(
        baseline=_report(throughput_rps=400.0, latency_p95_ms=20.0),
        current=_report(throughput_rps=300.0, latency_p95_ms=30.0),
    )
    json_path = tmp_path / "gate.json"
    markdown_path = tmp_path / "gate.md"

    write_benchmark_gate(gate, json_path)
    write_benchmark_gate(gate, markdown_path)

    markdown = render_benchmark_gate_markdown(gate)
    assert "# StreamInfer Benchmark Gate" in markdown
    assert "**Verdict:** `fail`" in markdown
    assert "| Check | Status | Observed | Threshold |" in markdown
    assert "`throughput_drop_pct`" in markdown_path.read_text(encoding="utf-8")
    assert "\"verdict\": \"fail\"" in json_path.read_text(encoding="utf-8")
