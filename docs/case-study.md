# StreamInfer Case Study

StreamInfer is my small inference-serving lab for studying the operational mechanics
around model serving: batching, backpressure, model reloads, smoke checks, and benchmark
gates.

It is not trying to replace Triton, Ray Serve, vLLM, or a managed inference platform. I
built it to make the serving tradeoffs visible in plain Python before reaching for a
larger system.

## Problem

Inference services have a few tensions that are easy to hand-wave in a demo:

- Small batches keep latency low but can waste throughput.
- Large batches improve throughput but make requests wait.
- Clients need backpressure before they fill the server queue.
- Model reloads need a visible health and metrics path.
- Benchmark numbers are only useful if the same gate can be rerun later.

I wanted a compact project where those behaviors are explicit and testable.

## What I Built

The service exposes:

- `POST /predict` for request/response inference through the same batcher path as
  WebSocket traffic
- `/ws` for WebSocket request/response inference
- `/api/reload` for switching demo models
- `/metrics` for request, batch, latency, and active-model counters
- local load-test and benchmark-sweep scripts that write JSON/Markdown artifacts
- a benchmark gate that compares a baseline sweep against a current sweep

The benchmark path uses a deterministic `synthetic-llm` profile, so it runs without API
keys, GPUs, or model downloads.

## Current Sample Result

The tracked sample sweep compares `6` batch/timeout configurations.

| Batch Size | Timeout | Throughput | p95 Latency | Avg Batch Size |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5ms | 41.93 req/s | 190.56 ms | 1.0 |
| 4 | 5ms | 180.04 req/s | 44.35 ms | 4.0 |
| 8 | 5ms | 411.26 req/s | 19.37 ms | 8.0 |

The sample gate currently passes:

- throughput drop: `2.88%` against a `10%` threshold
- p95 latency increase: `3.09%` against a `20%` threshold
- errors: `0`
- recommendation: unchanged at `batch_size=8,timeout_ms=5`

These are local regression-check numbers, not universal serving benchmarks. I keep that
boundary explicit because latency and throughput depend on the machine and runtime load.

## Design Choices

### Use one batcher path

HTTP and WebSocket requests both enter the same adaptive batcher. That keeps the
important serving behavior in one place instead of splitting logic by transport.

### Put backpressure before the batcher

Per-client rate limits and pending-request checks happen before work enters the batcher.
That makes overload behavior easier to reason about and easier to test.

### Keep benchmark artifacts reviewable

The sweep writes JSON for automation and Markdown for humans. The gate makes regression
criteria explicit: throughput drop, p95 latency increase, current errors, recommendation
change, and result-count change.

### Keep the model simple

The demo models and synthetic profile are intentionally small. The point is to test
serving behavior, not claim model quality.

## Verification

Local verification currently includes:

- `40` pytest tests
- Ruff checks
- live smoke checks for `/health`, `/predict`, `/api/reload`, and `/metrics`
- Docker smoke checks for image build, health, prediction, reload, and metrics
- sample benchmark sweep and sample gate artifacts

Commands:

```bash
PYTHON=.venv/bin/python make verify
PYTHON=.venv/bin/python make docker-check
```

The repeatable proof is tracked in [verification.md](verification.md),
[sample-inference-sweep.md](sample-inference-sweep.md), and
[sample-inference-gate.md](sample-inference-gate.md).

## What I Would Improve Next

- Add a real model backend behind the same batcher interface.
- Add a small dashboard for comparing benchmark runs over time.
- Persist metrics outside process memory.
- Add graceful model-rollout states instead of a direct hot-swap.
- Add a load profile that better matches streaming-token workloads.

