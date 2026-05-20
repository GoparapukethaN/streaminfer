#!/usr/bin/env bash
set -euo pipefail

IMAGE="${STREAMINFER_DOCKER_IMAGE:-streaminfer:local-smoke}"
PORT="${STREAMINFER_DOCKER_PORT:-8011}"
CONTAINER="streaminfer-smoke-$$"
BASE_URL="http://127.0.0.1:${PORT}"
PYTHON="${PYTHON:-python}"

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build -t "$IMAGE" .
docker run -d --name "$CONTAINER" -p "${PORT}:8000" "$IMAGE" >/dev/null

for _ in {1..60}; do
  if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "StreamInfer container exited during startup" >&2
    docker logs "$CONTAINER" >&2 || true
    exit 1
  fi
  sleep 0.5
done

curl -fsS "$BASE_URL/health" | "$PYTHON" -c \
  'import json,sys; data=json.load(sys.stdin); assert data["status"] == "ok"; assert data["model"] == "echo"'

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
  'import json,sys; data=json.load(sys.stdin); assert data["requests_total"] >= 2; assert data["model_name"] == "upper"'

for _ in {1..30}; do
  status="$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER")"
  if [[ "$status" == "healthy" ]]; then
    break
  fi
  sleep 0.5
done
test "$status" = "healthy"

echo "StreamInfer Docker smoke test passed on ${BASE_URL}"
