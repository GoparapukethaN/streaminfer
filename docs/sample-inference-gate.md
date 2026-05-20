# StreamInfer Benchmark Gate

**Verdict:** `pass`

## Recommendation

| Field | Baseline | Current |
| --- | ---: | ---: |
| Config | `batch_size=8,timeout_ms=5` | `batch_size=8,timeout_ms=5` |
| Throughput | 423.47 req/s | 423.47 req/s |
| p95 latency | 18.79 ms | 18.79 ms |
| Total errors | 0 | 0 |
| Result count | 6 | 6 |

## Checks

| Check | Status | Observed | Threshold |
| --- | --- | ---: | ---: |
| `throughput_drop_pct` | `pass` | 0 | 10 |
| `p95_latency_increase_pct` | `pass` | 0 | 20 |
| `current_errors_total` | `pass` | 0 | 0 |
| `recommendation_changed` | `pass` | false | `same recommended config` |
| `result_count_changed` | `pass` | 0 | 0 |
