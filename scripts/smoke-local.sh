#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PORT="${STREAMINFER_SMOKE_PORT:-8010}"
BASE_URL="http://127.0.0.1:${PORT}"
LOG_FILE="$(mktemp "${TMPDIR:-/tmp}/streaminfer-smoke.XXXXXX")"

STREAMINFER_HOST=127.0.0.1 STREAMINFER_PORT="$PORT" \
  "$PYTHON" -m streaminfer.server >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in {1..40}; do
  if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "StreamInfer server exited during startup" >&2
    cat "$LOG_FILE" >&2
    exit 1
  fi
  sleep 0.25
done

curl -fsS "$BASE_URL/health" | "$PYTHON" -c \
  'import json,sys; data=json.load(sys.stdin); assert data["status"] == "ok"'

curl -fsS -X POST "$BASE_URL/predict" \
  -H 'Content-Type: application/json' \
  -d '{"text":"mlops"}' | "$PYTHON" -c \
  'import json,sys; data=json.load(sys.stdin); assert data["result"] == "mlops"'

curl -fsS -X POST "$BASE_URL/api/reload" \
  -H 'Content-Type: application/json' \
  -d '{"model":"upper"}' | "$PYTHON" -c \
  'import json,sys; data=json.load(sys.stdin); assert data["status"] == "ok"; assert data["model"] == "upper"'

curl -fsS -X POST "$BASE_URL/predict" \
  -H 'Content-Type: application/json' \
  -d '{"text":"mlops"}' | "$PYTHON" -c \
  'import json,sys; data=json.load(sys.stdin); assert data["result"] == "MLOPS"'

curl -fsS "$BASE_URL/metrics" | "$PYTHON" -c \
  'import json,sys; data=json.load(sys.stdin); assert "requests_total" in data; assert data["model_name"] == "upper"'

echo "StreamInfer smoke test passed on ${BASE_URL}"
