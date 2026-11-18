"""End-to-end PathSnap prediction pipeline.

predict_trip():   raw trip -> candidates -> classifier -> HMM/Viterbi route
score_prediction(): prediction -> observation-level metrics (server side only)
to_api_payload(): prediction -> JSON for the evaluator UI (no ground truth)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import config
from .candidates import CandidateSet, TripNetwork, track_points_xy
from .features import FEATURE_NAMES, build_features
from .geo import to_lonlat
from .hmm import viterbi_decode
from .io import Trip, load_trip, pick_eval_partition, list_trips
from .labels import label_observations
from .metrics import compute_metrics

MODEL_FILE = "model.joblib"
MODELS_SUBDIR = "models"
REGISTRY_FILE = "model_registry.json"


def model_registry(artifacts_dir: Path | str = config.ARTIFACTS_DIR) -> dict:
    """Deployable model registry written by scripts/export_models.py."""
    path = Path(artifacts_dir) / "results" / REGISTRY_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            pass
    return {"default": "lightgbm_tuned",
            "models": [{"id": "lightgbm_tuned", "name": "LightGBM (tuned)",
                        "family": "Boosting"}]}


def _probs_from_distance(cand: CandidateSet, scale: float = 12.0) -> np.ndarray:
    p = np.exp(-cand.dist / scale)
    out = np.empty(len(p))
    if len(p) == 0:
        return out
    counts = np.bincount(cand.obs_idx, minlength=cand.obs_idx.max() + 1)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for s, c in zip(starts, counts):
        if c:
            seg = p[s:s + c]
            out[s:s + c] = seg / seg.sum()
    return out


def _normalise_probs(cand: CandidateSet, probs: np.ndarray) -> np.ndarray:
    """Normalise candidate scores per GPS fix (so sampling ratios cancel)."""
    out = np.empty(len(probs), dtype=np.float64)
    if len(probs) == 0:
        return out
    counts = np.bincount(cand.obs_idx, minlength=int(cand.obs_idx.max()) + 1)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for s, c in zip(starts, counts):
        if c:
            seg = probs[s:s + c].astype(np.float64)
            tot = seg.sum()
            out[s:s + c] = seg / tot if tot > 0 else 1.0 / c
    return out


def _argmax_per_obs(probs: np.ndarray, cand: CandidateSet, n_obs: int) -> np.ndarray:
    pred = np.full(n_obs, -1, dtype=np.int64)
    if len(cand) == 0:
        return pred
    counts = np.bincount(cand.obs_idx, minlength=n_obs)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for i in range(n_obs):
        s = starts[i]; c = counts[i]
        if c:
            pred[i] = int(cand.arc[s + int(np.argmax(probs[s:s + c]))])
    return pred


def _chain_strokes(pred_arc: np.ndarray, trip: Trip, net: TripNetwork):
    """Convert a predicted arc sequence into a list of connected lon/lat polylines."""
    arcs = trip.arcs
    chains: list[list[int]] = []
    cur: list[int] = []
    for a in pred_arc:
        a = int(a)
        if a < 0:
            continue
        if cur:
            prev = cur[-1]
            if a != prev and int(arcs[a, 0]) != int(arcs[prev, 1]):
                chains.append(cur)
                cur = []
            elif a == prev:
                continue
        cur.append(a)
    if cur:
        chains.append(cur)
    strokes = []
    for chain in chains:
        node_ids = [int(arcs[chain[0], 0])] + [int(arcs[a, 1]) for a in chain]
        lon, lat = to_lonlat(net.node_x[node_ids], net.node_y[node_ids], net.frame)
        pts = [[round(float(x), 6), round(float(y), 6)] for x, y in zip(lon, lat)]
        dedup = [pts[0]]
        for p in pts[1:]:
            if p != dedup[-1]:
                dedup.append(p)
        if len(dedup) >= 2:
            strokes.append(dedup)
    return strokes


def _network_corridor(trip: Trip, net: TripNetwork, obs_xy: np.ndarray,
                      buffer_m: float = 120.0):
    """Road-network geometry near the trajectory (for drawing on the UI map)."""
    from scipy.spatial import cKDTree
    node_xy = np.column_stack([net.node_x, net.node_y])
    tree = cKDTree(node_xy)
    if len(obs_xy) > 4000:
        sample = obs_xy[:: max(1, len(obs_xy) // 4000)]
    else:
        sample = obs_xy
    hits = tree.query_ball_point(sample, r=buffer_m)
    keep = np.zeros(len(node_xy), dtype=bool)
    for lst in hits:
        keep[np.asarray(lst, dtype=np.int64)] = True
    arcs = trip.arcs
    # both endpoints must be inside the corridor: a remapped arc can never
    # reference a missing node
    arc_keep = keep[arcs[:, 0]] & keep[arcs[:, 1]]
    node_ids = np.where(keep)[0]
    if len(node_ids) == 0:
        return {"nodes": [], "arcs": []}
    remap = np.full(len(node_xy), -1, dtype=np.int64)
    remap[node_ids] = np.arange(len(node_ids))
    sub_arcs = arcs[arc_keep]
    sub_arcs = remap[sub_arcs]
    lon, lat = to_lonlat(node_xy[node_ids, 0], node_xy[node_ids, 1], net.frame)
    nodes = [[round(float(x), 6), round(float(y), 6)] for x, y in zip(lon, lat)]
    return {"nodes": nodes, "arcs": sub_arcs.astype(int).tolist()}


class PathSnapPipeline:
    """Loads trained artifacts and runs prediction + scoring."""

    def __init__(self, artifacts: dict | None, artifacts_dir: Path | str):
        self.artifacts_dir = Path(artifacts_dir)
        self.model = (artifacts or {}).get("model")
        self.feature_names = (artifacts or {}).get("feature_names", FEATURE_NAMES)
        self.hmm_params = (artifacts or {}).get("hmm", {}) or {}
        self.metadata = (artifacts or {}).get("metadata", {}) or {}
        self.mode = "trained" if self.model is not None else "distance-baseline"
        self.model_id = self.metadata.get("id", "lightgbm_tuned")

    @classmethod
    def load(cls, artifacts_dir: Path | str = config.ARTIFACTS_DIR,
             model_id: str | None = None):
        """Load the default artifact or one of the exported picker models."""
        artifacts_dir = Path(artifacts_dir)
        registry = model_registry(artifacts_dir)
        chosen = model_id or registry.get("default", "lightgbm_tuned")
        if model_id and model_id != registry.get("default", "lightgbm_tuned"):
            path = artifacts_dir / MODELS_SUBDIR / f"{model_id}.joblib"
        else:
            path = artifacts_dir / MODEL_FILE
            if not path.exists():
                path = artifacts_dir / MODELS_SUBDIR / f"{chosen}.joblib"
        artifacts = joblib.load(path) if path.exists() else None
        return cls(artifacts, artifacts_dir)

    # ------------------------------------------------------------------
    def predict_trip(self, trip_id: str, partition: str,
                     data_root: Path | str = config.DATA_ROOT,
                     use_hmm: bool = True) -> dict:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        trip = load_trip(trip_id, partition, data_root)
        net = TripNetwork(trip)
        obs_xy = track_points_xy(net, trip)
        t1 = time.perf_counter()
        timings["load_network_ms"] = (t1 - t0) * 1000.0
        cand = net.candidates(obs_xy)
        t2 = time.perf_counter()
        timings["candidates_ms"] = (t2 - t1) * 1000.0
        n = len(trip.track)
        if len(cand) == 0:
            probs = np.empty(0)
        elif self.model is None:
            probs = _probs_from_distance(cand)
        else:
            feat = build_features(net, trip, cand)
            cols = [self.feature_names.index(c) for c in self.feature_names]
            raw = self.model.predict_proba(feat["X"][:, cols])
            classes = list(getattr(self.model, "classes_", [0, 1]))
            col = classes.index(1) if 1 in classes else raw.shape[1] - 1
            probs = raw[:, col]
        t3 = time.perf_counter()
        timings["features_model_ms"] = (t3 - t2) * 1000.0
        pred_clf = _argmax_per_obs(probs, cand, n)
        if use_hmm and len(cand) > 0:
            pred_hmm = viterbi_decode(
                net, cand, probs, obs_xy, trip.arcs,
                beta=float(self.hmm_params.get("beta", config.HMM_BETA_M)),
                emission_alpha=float(self.hmm_params.get("emission_alpha",
                                                         config.HMM_EMISSION_ALPHA)),
                max_states=int(self.hmm_params.get("max_states", config.HMM_MAX_STATES)),
                gap_free_s=float(self.hmm_params.get("gap_free_s", config.HMM_GAP_FREE_S)),
                times=trip.track[:, 2], n_obs=n)
        else:
            pred_hmm = pred_clf.copy()
        t4 = time.perf_counter()
        timings["hmm_ms"] = (t4 - t3) * 1000.0
        timings["total_ms"] = (t4 - t0) * 1000.0
        return {"trip": trip, "net": net, "cand": cand, "probs": probs,
                "pred_clf": pred_clf, "pred_hmm": pred_hmm, "obs_xy": obs_xy,
                "timings": timings, "mode": self.mode, "partition": partition}

    # ------------------------------------------------------------------
    def score_prediction(self, pred: dict) -> dict:
        trip = pred["trip"]
        if not trip.has_route:
            return {"reference_available": False}
        labels = label_observations(pred["net"], trip, pred["obs_xy"])
        # candidate-level metrics always use per-fix normalised probabilities so
        # they are identical to the numbers recorded in the model comparison
        probs = _normalise_probs(pred["cand"], pred["probs"])
        metrics = compute_metrics(pred["net"], pred["cand"], probs,
                                  pred["pred_clf"], pred["pred_hmm"],
                                  labels.arc, labels.valid, trip.route)
        metrics["reference_available"] = True
        metrics["label_method"] = "route-projection-dp"
        return metrics

    def _display_name(self) -> str:
        """Human-readable model name from the registry (fallback: metadata)."""
        for entry in model_registry(self.artifacts_dir).get("models", []):
            if entry.get("id") == self.model_id:
                return str(entry.get("name") or self.model_id)
        return str(self.metadata.get("name") or self.model_id)

    # ------------------------------------------------------------------
    def to_api_payload(self, pred: dict, metrics: dict | None,
                       include_network: bool = True) -> dict:
        trip = pred["trip"]
        net = pred["net"]
        gps_lon, gps_lat = to_lonlat(pred["obs_xy"][:, 0], pred["obs_xy"][:, 1], net.frame)
        payload = {
            "trip_id": trip.trip_id,
            "partition": pred["partition"],
            "mode": pred["mode"],
            "model": self.model_id,
            "model_name": self._display_name(),
            "n_obs": int(len(trip.track)),
            "timings_ms": {k: round(float(v), 1) for k, v in pred["timings"].items()},
            "gps": [[round(float(a), 6), round(float(b), 6)]
                    for a, b in zip(gps_lon, gps_lat)],
            "predicted_strokes_hmm": _chain_strokes(pred["pred_hmm"], trip, net),
            "predicted_strokes_clf": _chain_strokes(pred["pred_clf"], trip, net),
            "metrics": metrics,
        }
        if include_network:
            payload["network"] = _network_corridor(trip, net, pred["obs_xy"])
        return payload


def available_trips(partition: str | None = None,
                    data_root: Path | str = config.DATA_ROOT):
    """Trip ids for the evaluator UI: evaluator partition if present, else dev set."""
    if partition is None:
        partition = pick_eval_partition(data_root)
    if partition is None:
        return None, []
    return partition, list_trips(partition, data_root)


_PIPELINE_CACHE: PathSnapPipeline | None = None


def get_pipeline() -> PathSnapPipeline:
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is None:
        _PIPELINE_CACHE = PathSnapPipeline.load()
    return _PIPELINE_CACHE
