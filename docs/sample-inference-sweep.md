# StreamInfer Benchmark Sweep

This report compares local batching settings with the deterministic `synthetic-llm` profile. The numbers are useful for regression checks and relative tradeoffs on this machine, not as universal serving benchmarks.

## Recommendation

- Batch size `8` with timeout `5ms`.
- Throughput: `407.46` req/s.
- p95 latency: `19.48` ms.

## Results

| batch_size | timeout_ms | throughput_rps | latency_p95_ms | avg_batch_size |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5 | 41.68 | 191.79 | 1.0 |
| 1 | 25 | 23.95 | 333.89 | 1.0 |
| 4 | 5 | 179.41 | 44.47 | 4.0 |
| 4 | 25 | 122.96 | 64.9 | 4.0 |
| 8 | 5 | 407.46 | 19.48 | 8.0 |
| 8 | 25 | 407.06 | 19.49 | 8.0 |
