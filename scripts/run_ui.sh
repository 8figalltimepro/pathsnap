#!/usr/bin/env bash
# PathSnap UI launcher - run this on LAPTOP 2 (evaluator machine).
# Serves the static Leaflet UI on port 8501; point it at laptop 1's API.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"  # laptop 2 only needs the Python standard library
fi
exec "$PY" ui/serve.py --port 8501
