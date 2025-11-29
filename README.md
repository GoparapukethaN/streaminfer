# StreamInfer

Real-time streaming ML inference server with adaptive batching, backpressure, and model hot-swap.

I built this after spending too much time at Observe.AI dealing with the gap between "model works in a notebook" and "model serves 500 concurrent users." Most inference servers either batch too aggressively (high latency) or not at all (low throughput). StreamInfer finds the sweet spot automatically.

## What it does

- **WebSocket streaming** — clients send data, get predictions back in real-time
- **Adaptive batching** — accumulates requests and flushes on batch_size OR timeout, whichever hits first
- **Backpressure** — token bucket rate limiter + slow consumer detection prevents one bad client from killing the server
- **Model hot-swap** — swap models with zero downtime via SIGHUP or API call
- **Metrics** — in-memory counters exposed at `/metrics` (no Prometheus dependency needed)

## Architecture

```
                                    ┌─────────────┐
  WebSocket ──────┐                 │             │
  Client 1        │    ┌──────────┐ │   Adaptive  │ ┌───────────┐
                  ├───→│Backpress.│─→│   Batcher   │─→│   Model   │
  WebSocket ──────┤    │(per-client│ │             │ │  Holder   │
  Client 2        │    │rate limit)│ │ flush on:   │ │  (swap    │
                  │    └──────────┘ │ • batch full │ │  via lock)│
  POST /predict ──┘                 │ • timeout    │ └───────────┘
                                    └─────────────┘
```

### Why adaptive batching matters

Fixed batching forces a tradeoff: small batches waste GPU cycles, large batches add latency. The adaptive approach collects items into a batch and flushes when EITHER:

1. The batch is full (throughput-optimal)
2. A timeout fires (latency-bounded)

This means: at high load you get full batches (good throughput), at low load you get fast responses (low latency). Same idea as Triton Inference Server's dynamic batcher, but ~100 lines of Python instead of a C++ behemoth.

### Backpressure

Each WebSocket client gets a token bucket rate limiter. If a client sends faster than the configured rate, excess requests get a `rate_limited` response with a `retry_after_ms` hint. If a client's pending queue exceeds 80% capacity, they get a `consumer falling behind` warning.

Without this, one runaway client can fill the server's memory with pending requests until it OOMs.

## Quick start

```bash
pip install -e "."

# start server (uses echo model by default)
python -m streaminfer.server

# in another terminal — send some requests
python examples/client.py
```

### Docker

```bash
docker build -t streaminfer .
docker run -p 8000:8000 streaminfer
```

### Configuration

All settings via environment variables (prefix `STREAMINFER_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMINFER_BATCH_SIZE` | 16 | Max items per batch |
| `STREAMINFER_BATCH_TIMEOUT_MS` | 50 | Flush timeout in ms |
| `STREAMINFER_MAX_QUEUE_SIZE` | 1000 | Per-client queue limit |
| `STREAMINFER_RATE_LIMIT_RPS` | 100 | Requests/sec per client |
| `STREAMINFER_MODEL_NAME` | echo | Model to load at startup |
| `STREAMINFER_PORT` | 8000 | Server port |

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws` | WebSocket | Streaming inference — send JSON, get JSON back |
| `/predict` | POST | Single request/response (still batched internally) |
| `/metrics` | GET | Server metrics as JSON |
| `/api/reload` | POST | Hot-swap model: `{"model": "upper"}` |
| `/health` | GET | Health check |

## Hot-swap

Three ways to swap the model with zero downtime:

```bash
# 1. API call
curl -X POST localhost:8000/api/reload -d '{"model": "upper"}'

# 2. SIGHUP (reloads from config)
kill -HUP $(pgrep -f streaminfer)

# 3. The swap is atomic — old model finishes in-flight requests,
#    new model handles all new ones. No requests dropped.
```

## Load test results

With echo model on M1 MacBook (not a fair GPU benchmark, but shows the batching works):

```
connections:  100
requests:     50 per connection
total:        5000 requests in 4.2s
throughput:   1190 req/s
latency p50:  12.3ms
latency p95:  34.7ms
latency p99:  48.2ms
```

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
