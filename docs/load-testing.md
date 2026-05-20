# Load Testing

StreamInfer includes a small WebSocket load-test client for checking local serving
behavior after batching, backpressure, or model reload changes.

Start the server:

```bash
python -m streaminfer.server
```

Run a short load test:

```bash
python examples/load_test.py \
  --connections 10 \
  --requests 20 \
  --output artifacts/load-test.json
```

The JSON report records the run configuration, successful request count, error count,
throughput, and p50/p95/p99 latency. I use it as a local regression artifact rather than
as a permanent benchmark, because latency depends on the machine running the server.

The report shape is:

```json
{
  "config": {
    "uri": "ws://localhost:8000/ws",
    "connections": 10,
    "requests_per_connection": 20
  },
  "summary": {
    "total_requests": 200,
    "errors_total": 0,
    "elapsed_seconds": 1.25,
    "throughput_rps": 160.0,
    "latency_p50_ms": 12.4,
    "latency_p95_ms": 30.8,
    "latency_p99_ms": 45.1
  },
  "sample_errors": []
}
```

Before quoting numbers in a public note, rerun the command on the target machine and keep
the raw JSON artifact with the run.

Latest local report-path check: a small 4-connection, 20-request run completed with `0`
errors and wrote `/tmp/streaminfer-load-test.json`. The exact latency numbers are not
treated as benchmark claims.
