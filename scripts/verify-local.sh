#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

"$PYTHON" -m pytest tests -q
"$PYTHON" -m ruff check streaminfer tests
./scripts/smoke-local.sh
