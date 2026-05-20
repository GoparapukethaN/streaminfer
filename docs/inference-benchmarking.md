# Inference Benchmarking

StreamInfer includes an in-process benchmark sweep for comparing batching configurations
with a deterministic LLM-style model profile.

The goal is to make the serving tradeoff visible:

- smaller batches usually reduce wait time but leave throughput on the table
- larger batches can improve throughput when requests arrive together
- timeout settings decide how long a partial batch can wait before it runs

## Run A Sweep

```bash
python examples/sweep_benchmark.py \
  --batch-sizes 1,4,8 \
  --timeouts-ms 5,25 \
  --concurrency 8 \
  --output-json artifacts/inference-sweep.json \
  --output-md artifacts/inference-sweep.md
```

The script writes both JSON and Markdown reports. The JSON report is easier to compare in
automation; the Markdown report is easier to review during an interview or design walk.

## Run A Gate

Use the previous JSON report as a baseline when checking a new sweep:

```bash
python examples/sweep_benchmark.py \
  --batch-sizes 1,4,8 \
  --timeouts-ms 5,25 \
  --concurrency 8 \
  --baseline-json docs/sample-inference-sweep.json \
  --output-json artifacts/inference-sweep.json \
  --output-md artifacts/inference-sweep.md \
  --gate-json artifacts/inference-gate.json \
  --gate-md artifacts/inference-gate.md
```

The gate compares the recommended configuration from the baseline and current report.
It fails when throughput drops beyond the configured threshold, p95 latency increases too
much, or the current sweep has errors. It warns when the recommended batch/timeout pair
or the benchmark grid changes.

## What The Synthetic Profile Does

The built-in `synthetic-llm` model estimates prompt tokens with a simple whitespace
tokenizer, adds a fixed completion-token count, and waits for a deterministic amount of
time before returning a result. That keeps the benchmark keyless and repeatable while
still exercising the async model path, batcher, latency counters, and report generation.

This is not a replacement for benchmarking a real model on target hardware. I use it as a
local regression tool and as a way to reason about serving configuration before adding a
heavier backend.

## Report Fields

- `throughput_rps`: successful requests divided by elapsed wall time
- `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`: request latency percentiles
- `avg_batch_size`: average processed batch size for the configuration
- `total_timeouts`: batches flushed by timeout instead of full batch size
- `recommendation`: highest-throughput zero-error configuration, optionally constrained
  by a p95 target
- `gate`: pass/warn/fail comparison between a baseline and current sweep report

## Current Sample

The current sample report is tracked at
[sample-inference-sweep.md](sample-inference-sweep.md), with the raw JSON beside it. The
sample gate is tracked at [sample-inference-gate.md](sample-inference-gate.md). The
numbers should be rerun on the target machine before being quoted as performance claims.
