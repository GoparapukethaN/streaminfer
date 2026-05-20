# StreamInfer Benchmark Sweep

This report compares local batching settings with the deterministic `synthetic-llm` profile. The numbers are useful for regression checks and relative tradeoffs on this machine, not as universal serving benchmarks.

## Recommendation

- Batch size `8` with timeout `5ms`.
- Throughput: `423.47` req/s.
- p95 latency: `18.79` ms.

## Results

| batch_size | timeout_ms | throughput_rps | latency_p95_ms | avg_batch_size |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5 | 41.75 | 191.49 | 1.0 |
| 1 | 25 | 23.73 | 336.87 | 1.0 |
| 4 | 5 | 179.49 | 44.43 | 4.0 |
| 4 | 25 | 123.26 | 64.82 | 4.0 |
| 8 | 5 | 423.47 | 18.79 | 8.0 |
| 8 | 25 | 409.16 | 19.44 | 8.0 |
