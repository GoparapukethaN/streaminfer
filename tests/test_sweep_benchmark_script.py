import json
import subprocess
import sys
from pathlib import Path


def test_sweep_script_can_write_gate_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    baseline_path = tmp_path / "baseline.json"
    output_json = tmp_path / "sweep.json"
    output_md = tmp_path / "sweep.md"
    gate_json = tmp_path / "gate.json"
    gate_md = tmp_path / "gate.md"
    baseline_path.write_text(
        json.dumps(
            {
                "config": {
                    "total_prompts": 8,
                    "batch_sizes": [1],
                    "timeout_ms_values": [1],
                    "concurrency": 1,
                    "model": "synthetic-llm",
                },
                "recommendation": {
                    "batch_size": 1,
                    "timeout_ms": 1,
                    "throughput_rps": 1.0,
                    "latency_p95_ms": 10000.0,
                    "avg_batch_size": 1.0,
                },
                "results": [
                    {
                        "batch_size": 1,
                        "timeout_ms": 1,
                        "summary": {
                            "total_requests": 8,
                            "errors_total": 0,
                            "throughput_rps": 1.0,
                            "latency_p95_ms": 10000.0,
                        },
                        "batcher": {"avg_batch_size": 1.0},
                        "sample_errors": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "examples/sweep_benchmark.py",
            "--batch-sizes",
            "1",
            "--timeouts-ms",
            "1",
            "--concurrency",
            "1",
            "--baseline-json",
            str(baseline_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--gate-json",
            str(gate_json),
            "--gate-md",
            str(gate_md),
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Benchmark gate passed" in result.stdout
    assert json.loads(gate_json.read_text(encoding="utf-8"))["verdict"] == "pass"
    assert "# StreamInfer Benchmark Gate" in gate_md.read_text(encoding="utf-8")


def test_sweep_script_loads_baseline_before_overwriting_output(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "sweep.json"
    output_md = tmp_path / "sweep.md"
    gate_json = tmp_path / "gate.json"
    gate_md = tmp_path / "gate.md"
    output_json.write_text(
        json.dumps(
            {
                "config": {
                    "total_prompts": 8,
                    "batch_sizes": [9],
                    "timeout_ms_values": [9],
                    "concurrency": 1,
                    "model": "synthetic-llm",
                },
                "recommendation": {
                    "batch_size": 9,
                    "timeout_ms": 9,
                    "throughput_rps": 1.0,
                    "latency_p95_ms": 10000.0,
                    "avg_batch_size": 9.0,
                },
                "results": [
                    {
                        "batch_size": 9,
                        "timeout_ms": 9,
                        "summary": {
                            "total_requests": 8,
                            "errors_total": 0,
                            "throughput_rps": 1.0,
                            "latency_p95_ms": 10000.0,
                        },
                        "batcher": {"avg_batch_size": 9.0},
                        "sample_errors": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "examples/sweep_benchmark.py",
            "--batch-sizes",
            "1",
            "--timeouts-ms",
            "1",
            "--concurrency",
            "1",
            "--baseline-json",
            str(output_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--gate-json",
            str(gate_json),
            "--gate-md",
            str(gate_md),
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Benchmark gate passed with warnings" in result.stdout
    gate = json.loads(gate_json.read_text(encoding="utf-8"))
    assert gate["verdict"] == "warn"
    assert gate["checks"][3]["baseline"] == "batch_size=9,timeout_ms=9"
