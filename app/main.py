"""PathSnap FastAPI backend (laptop 1).

This process owns the trained classifier / distance baseline, the road-network
data and every ground-truth comparison.  Responses are built exclusively from
pathsnap.pipeline.to_api_payload(), so reference routes (.route files) never
leave the machine: only predictions, map geometry and server-side metrics do.

Run (laptop 1, from the project root):

    .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET /health                service status + model mode + evaluation partition
    GET /trips                 trips listed by the evaluator UI
    GET /evaluate/{trip_id}    run the pipeline for one trip (cached)
    GET /                      small JSON index
    GET /docs                  interactive API documentation
"""
from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the workspace src/ directory importable regardless of how uvicorn
# was started (project root, --app-dir, etc.).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402

from pathsnap import config  # noqa: E402
from pathsnap.io import partition_roots  # noqa: E402
from pathsnap.pipeline import (MODELS_SUBDIR, PathSnapPipeline, available_trips,  # noqa: E402
                               model_registry)

# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PathSnap API",
    description=(
        "Distributed map-matching service: predicts road arcs for every GPS fix, "
        "reconstructs the driven route with an HMM/Viterbi decoder and scores the "
        "result server side. Ground-truth reference routes are never returned."
    ),
    version="1.0.0",
)

# The evaluator UI (laptop 2) calls this API from a different host/port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Payloads with thousands of GPS fixes compress very well.
app.add_middleware(GZipMiddleware, minimum_size=1024)

# ---------------------------------------------------------------------------
# Pipeline / result caches
#
# The trained model is reloaded automatically when artifacts/model.joblib
# changes on disk (e.g. the training script has just exported a new model),
# so the API does not need a restart to pick it up.  A missing model file is
# not an error: the pipeline falls back to its distance-based baseline.
# ---------------------------------------------------------------------------
_ARTIFACTS = Path(config.ARTIFACTS_DIR)
_DEFAULT_PATH = _ARTIFACTS / "model.joblib"

_lock = threading.Lock()
_pipelines: dict[str, PathSnapPipeline] = {}
_pipeline_stamps: dict[str, object] = {}
_results: dict[tuple[str, str, str, object], dict] = {}
_registry_cache: dict = {"mtime": None, "data": None}


def _artifact_path(model_id: str) -> Path:
    """Default model lives at artifacts/model.joblib, picker models in models/."""
    if model_id == default_model_id():
        return _DEFAULT_PATH
    return _ARTIFACTS / MODELS_SUBDIR / f"{model_id}.joblib"


def _model_stamp(model_id: str) -> float | None:
    """Last-modified time of one model artifact, or None when absent."""
    try:
        return _artifact_path(model_id).stat().st_mtime
    except OSError:
        return None


def get_registry() -> dict:
    """Deployable-model registry (re-read when the file changes)."""
    path = _ARTIFACTS / "results" / "model_registry.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    if _registry_cache["data"] is None or _registry_cache["mtime"] != mtime:
        _registry_cache["data"] = model_registry(config.ARTIFACTS_DIR)
        _registry_cache["mtime"] = mtime
    return _registry_cache["data"]


def default_model_id() -> str:
    return str(get_registry().get("default", "lightgbm_tuned"))


def model_ids() -> list[str]:
    return [str(m["id"]) for m in get_registry().get("models", [])]


def get_pipeline(model_id: str | None = None) -> PathSnapPipeline:
    """Return the cached pipeline for a model, reloading when its file changes.

    A half-written or corrupt artifact must not take the API down: keep serving
    with the previously loaded model (or the distance baseline on a cold start)
    and log a warning instead.
    """
    model_id = model_id or default_model_id()
    stamp = _model_stamp(model_id)
    with _lock:
        cached = _pipelines.get(model_id)
        if cached is not None and stamp == _pipeline_stamps.get(model_id):
            return cached
        try:
            loaded = PathSnapPipeline.load(model_id=model_id)
        except Exception as exc:  # corrupt/partial joblib file
            print(
                "WARNING: could not load model %s (%s: %s); keeping the previous model"
                % (model_id, type(exc).__name__, exc),
                flush=True,
            )
            if cached is not None:
                return cached
            loaded = PathSnapPipeline(None, config.ARTIFACTS_DIR)
            loaded.model_id = model_id
        _pipelines[model_id] = loaded
        _pipeline_stamps[model_id] = stamp
        # predictions scored with the replaced model are stale
        for key in [k for k in _results if k[2] == model_id]:
            _results.pop(key, None)
        return loaded


def _eval_set() -> tuple[str, list[str]]:
    """Evaluation partition + trip ids (evaluator_test when present)."""
    partition, trip_ids = available_trips()
    if partition is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trip data found. Expected "
                "'{}/evaluator_test' or '{}/student_test'.".format(
                    config.DATA_ROOT, config.DATA_ROOT
                )
            ),
        )
    return partition, trip_ids


def resolved_track(partition: str, trip_id: str) -> Path:
    """Path of a trip's .track file, checking both the course and evaluator roots."""
    for root in partition_roots(partition, config.DATA_ROOT):
        path = root / trip_id / f"{trip_id}.track"
        if path.exists():
            return path
    return Path(config.DATA_ROOT) / partition / trip_id / f"{trip_id}.track"


def _count_lines(path: Path) -> int:
    """Number of GPS fixes in a .track file."""
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _json_safe(value):
    """Recursively replace NaN/inf with null so responses are strict JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/models")
def models() -> dict:
    """Deployable candidate classifiers the UI can switch between."""
    registry = get_registry()
    available = [m["id"] for m in registry.get("models", [])
                 if _artifact_path(str(m["id"])).exists()]
    return {
        "default": default_model_id(),
        "models": [{**m, "available": m["id"] in available}
                   for m in registry.get("models", [])],
    }


@app.get("/health")
def health() -> dict:
    """Service status used by the UI to confirm the laptop 2 -> laptop 1 link."""
    partition, trip_ids = _eval_set()
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "mode": pipeline.mode,  # "trained" or "distance-baseline"
        # True when artifacts/model.joblib was loaded; the UI shows a warning
        # banner when it is False (distance-baseline fallback).
        "model_loaded": pipeline.model is not None,
        "eval_partition": partition,  # "evaluator_test" or "student_test"
        "n_trips": len(trip_ids),
        "n_models": len([m for m in get_registry().get("models", [])
                         if _artifact_path(str(m["id"])).exists()]),
        "default_model": default_model_id(),
    }


@app.get("/trips")
def trips() -> dict:
    """Trips the evaluator UI can request."""
    partition, trip_ids = _eval_set()
    return {
        "partition": partition,
        # "evaluation" once the hidden evaluator partition is present,
        # "demo" while the service falls back to the student_test partition.
        "mode": "evaluation" if partition == config.EVAL_PARTITION else "demo",
        "trips": [
            {
                "trip_id": trip_id,
                "n_obs": _count_lines(resolved_track(partition, trip_id)),
            }
            for trip_id in trip_ids
        ],
    }


@app.get("/evaluate/{trip_id}")
def evaluate(trip_id: str, model: str | None = None) -> dict:
    """Run map-matching for one trip (and one model) returning predictions + metrics."""
    partition, trip_ids = _eval_set()
    if trip_id not in trip_ids:
        raise HTTPException(
            status_code=404,
            detail=f"trip '{trip_id}' not found in partition '{partition}'",
        )
    model_id = model or default_model_id()
    if model_id not in model_ids():
        raise HTTPException(
            status_code=404,
            detail=f"model '{model_id}' not found; available: {model_ids()}",
        )
    if not _artifact_path(model_id).exists():
        raise HTTPException(status_code=503,
                            detail=f"model '{model_id}' artifact missing on the server")

    pipeline = get_pipeline(model_id)
    stamp = _model_stamp(model_id)
    cache_key = (partition, trip_id, model_id, stamp)
    with _lock:
        cached = _results.get(cache_key)
    if cached is not None:
        return cached

    try:
        pred = pipeline.predict_trip(trip_id, partition)
        metrics = pipeline.score_prediction(pred)
        payload = pipeline.to_api_payload(pred, metrics)
    except FileNotFoundError as exc:  # trip folder vanished between listing and loading
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # surface a clean error instead of a raw traceback
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed for trip {trip_id}: {exc}",
        ) from exc

    payload = _json_safe(payload)
    with _lock:
        _results[cache_key] = payload
    return payload


@app.get("/")
def index() -> dict:
    return {
        "service": "PathSnap API",
        "status": "ok",
        "endpoints": {
            "GET /health": "model mode, evaluation partition, trip count",
            "GET /trips": "evaluation trips available to the UI",
            "GET /models": "deployable candidate classifiers for the model picker",
            "GET /evaluate/{trip_id}?model=<id>": "predictions + server-side metrics for one trip and model",
            "GET /docs": "interactive API documentation",
        },
        "note": "Ground-truth reference routes are never included in any response.",
    }
