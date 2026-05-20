# StreamInfer Benchmark Sweep

This report compares local batching settings with the deterministic `synthetic-llm` profile. The numbers are useful for regression checks and relative tradeoffs on this machine, not as universal serving benchmarks.

## Recommendation

- Batch size `8` with timeout `5ms`.
- Throughput: `411.26` req/s.
- p95 latency: `19.37` ms.

## Results

| batch_size | timeout_ms | throughput_rps | latency_p95_ms | avg_batch_size |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5 | 41.93 | 190.56 | 1.0 |
| 1 | 25 | 24.0 | 333.2 | 1.0 |
| 4 | 5 | 180.04 | 44.35 | 4.0 |
| 4 | 25 | 123.74 | 64.52 | 4.0 |
| 8 | 5 | 411.26 | 19.37 | 8.0 |
| 8 | 25 | 411.38 | 19.37 | 8.0 |
