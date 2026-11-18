#!/usr/bin/env bash
# PathSnap API launcher - run this on LAPTOP 1 (model + API machine).
# Serves the FastAPI backend on port 8000 for the evaluator UI on laptop 2.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
