from streaminfer.benchmark import percentile, summarize_load_test


def test_percentile_returns_zero_for_empty_values() -> None:
    assert percentile([], 95) == 0.0


def test_percentile_uses_sorted_values() -> None:
    assert percentile([30, 10, 20, 40], 50) == 20
    assert percentile([30, 10, 20, 40], 100) == 40


def test_summarize_load_test_reports_serving_metrics() -> None:
    summary = summarize_load_test(
        latencies_ms=[5, 10, 15, 20],
        elapsed_seconds=0.5,
        errors_total=1,
    )

    assert summary.total_requests == 4
    assert summary.errors_total == 1
    assert summary.throughput_rps == 8
    assert summary.latency_p95_ms == 20
