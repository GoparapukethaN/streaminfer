# Verification

This is the local checklist I use before treating StreamInfer changes as ready to show.
It keeps the public claims tied to commands that run on a fresh checkout.

## Local CI

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
make verify
```

`make ci-local` is an alias for the same local CI-equivalent check.

Current local result from 2026-05-20:

- `40 passed`
- `ruff` clean
- live smoke test passed against a temporary local server

`make verify` runs unit tests, lint, and `scripts/smoke-local.sh`. The smoke script starts
the FastAPI app, checks `/health`, exercises `/predict`, reloads the model through
`/api/reload`, verifies the reloaded prediction behavior, and checks `/metrics`.

## Load Check

```bash
STREAMINFER_HOST=127.0.0.1 STREAMINFER_PORT=8012 python -m streaminfer.server
python examples/load_test.py \
  --connections 4 \
  --requests 5 \
  --uri ws://127.0.0.1:8012/ws \
  --output /tmp/streaminfer-load-test.json
```

Latest local report-path check from 2026-05-20:

- `20` WebSocket requests
- `0` errors
- JSON report written to `/tmp/streaminfer-load-test.json`

I keep the latency numbers out of the README headline because they depend on the machine
and whatever else is running locally.

## Benchmark Gate

```bash
python examples/sweep_benchmark.py \
  --batch-sizes 1,4,8 \
  --timeouts-ms 5,25 \
  --concurrency 8 \
  --baseline-json docs/sample-inference-sweep.json \
  --output-json docs/sample-inference-sweep.json \
  --output-md docs/sample-inference-sweep.md \
  --gate-json docs/sample-inference-gate.json \
  --gate-md docs/sample-inference-gate.md
```

Current sample gate result from 2026-05-20: `pass`.

The benchmark uses the deterministic `synthetic-llm` profile, so it checks batching and
reporting behavior without API keys, GPUs, or model downloads. The numbers are useful for
relative local regression checks, not universal inference-serving claims.
