"""Integrate the Kaggle PyTorch MLP result into the model comparison.

The Kaggle kernel exports student-test candidate probabilities for its MLP; this
script recomputes the full metric bundle locally (including HMM decoding with the
same default decoder parameters used for every other model in the comparison)
and appends the model as "mlp_deep" in artifacts/results/model_comparison.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import config
from pathsnap.candidates import CandidateSet, TripNetwork, track_points_xy
from pathsnap.hmm import viterbi_decode
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import (candidate_probability_metrics, route_continuity,
                              sequence_f1, top1_accuracy, true_candidate_log_loss)

CACHE = config.ARTIFACTS_DIR / "cache"
# Kaggle rebuilt the candidate cache with a slightly different numpy version, which
# breaks ties between opposite-direction arcs differently.  Use the Kaggle cache
# for its exported probabilities so rows stay aligned.
KG_CACHE = config.ARTIFACTS_DIR / "kaggle" / "artifacts" / "cache"
RESULTS = config.ARTIFACTS_DIR / "results"


def cand_from_cache(tp):
    n = len(tp["obs_idx"])
    return CandidateSet(obs_idx=tp["obs_idx"].astype(np.int64),
                        arc=tp["arc"].astype(np.int64),
                        dist=tp["dist"].astype(np.float64),
                        t=tp["t"].astype(np.float64),
                        foot_x=np.zeros(n), foot_y=np.zeros(n),
                        rank=tp["rank"].astype(np.int64),
                        n_cand=tp["n_cand"].astype(np.int64))


def normalised(cand, probs):
    out = np.empty(len(probs), dtype=np.float64)
    counts = np.bincount(cand.obs_idx, minlength=int(cand.obs_idx.max()) + 1)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for s, c in zip(starts, counts):
        if c:
            seg = probs[s:s + c].astype(np.float64)
            tot = seg.sum()
            out[s:s + c] = seg / tot if tot > 0 else 1.0 / c
    return out


def main():
    mlp_path = RESULTS / "kaggle_mlp.json"
    if not mlp_path.exists():
        print("no Kaggle MLP results found at", mlp_path, "- skipping integration")
        return
    preds_path = RESULTS / "kaggle_mlp_preds.npz"
    if not preds_path.exists():
        print("no Kaggle MLP probability cache found - skipping integration")
        return
    mlp = json.loads(mlp_path.read_text())
    preds = np.load(RESULTS / "kaggle_mlp_preds.npz")
    trips = {}
    per_trip = []
    for tid in list_trips("student_test"):
        cache_file = KG_CACHE / f"student_test_{tid}.npz"
        if not cache_file.exists():
            cache_file = CACHE / f"student_test_{tid}.npz"
        tp = dict(np.load(cache_file))
        trip = load_trip(tid, "student_test")
        net = TripNetwork(trip)
        obs_xy = track_points_xy(net, trip)
        probs = preds[tid].astype(np.float64)
        cand = cand_from_cache(tp)
        assert len(probs) == len(cand), (tid, len(probs), len(cand))
        pn = normalised(cand, probs)
        valid = tp["label_valid"].astype(bool)
        m = candidate_probability_metrics(cand, pn, tp["label_arc"], valid)
        n = int(tp["n_obs"])
        pred = np.full(n, -1, dtype=np.int64)
        counts = np.bincount(cand.obs_idx, minlength=n)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for i in range(n):
            s, c = starts[i], counts[i]
            if c:
                pred[i] = cand.arc[s + int(np.argmax(probs[s:s + c]))]
        pred_h = viterbi_decode(net, cand, pn, obs_xy, trip.arcs,
                                times=trip.track[:, 2], n_obs=n)
        per_trip.append({
            "trip_id": tid,
            "top1_accuracy": top1_accuracy(pred, tp["label_arc"], valid),
            "top1_accuracy_hmm": top1_accuracy(pred_h, tp["label_arc"], valid),
            "macro_f1": m["macro_f1"], "log_loss": m["log_loss"], "f1": m["f1"],
            "precision": m["precision"], "recall": m["recall"],
            "true_candidate_log_loss": true_candidate_log_loss(cand, pn, tp["label_arc"], valid),
            "route_continuity": route_continuity(net, pred),
            "route_continuity_hmm": route_continuity(net, pred_h),
            "seq_f1_clf": sequence_f1(pred, trip.route)["seq_f1"],
            "seq_f1_hmm": sequence_f1(pred_h, trip.route)["seq_f1"],
        })
        print(f"  mlp_deep {tid}: top1 {per_trip[-1]['top1_accuracy']:.4f} "
              f"hmm {per_trip[-1]['top1_accuracy_hmm']:.4f}", flush=True)
    keys = [k for k in per_trip[0] if k != "trip_id"]
    aggregate = {k: float(np.mean([t[k] for t in per_trip])) for k in keys}
    aggregate["n_obs"] = float("nan")  # filled by report from caches if needed

    comp = json.loads((RESULTS / "model_comparison.json").read_text())
    cv = mlp["cv"]
    comp["models"]["mlp_deep"] = {
        "model": {"name": "mlp_deep", "family": "Neural network",
                  "config": mlp.get("config", "PyTorch MLP 26-128-64-1, Adam, early stopping"),
                  "cap": 1_000_000, "neg_per_obs": 4},
        "cv": {
            "macro_f1_mean": cv["macro_f1_mean"], "macro_f1_std": cv["macro_f1_std"],
            "top1_accuracy_mean": cv["top1_accuracy_mean"],
            "top1_accuracy_std": cv["top1_accuracy_std"],
            "log_loss_mean": cv["log_loss_mean"],
            "f1_mean": float("nan"), "precision_mean": float("nan"), "recall_mean": float("nan"),
            "fit_seconds_mean": float("nan"), "predict_seconds_mean": float("nan"),
            "folds": cv.get("folds", []),
        },
        "student_test": {
            "top1_accuracy": aggregate["top1_accuracy"],
            "macro_f1": aggregate["macro_f1"],
            "log_loss": aggregate["log_loss"],
            "f1": aggregate["f1"], "precision": aggregate["precision"],
            "recall": aggregate["recall"],
            "route_continuity": aggregate["route_continuity"],
            "seq_f1": aggregate["seq_f1_clf"],
            "true_candidate_log_loss": aggregate["true_candidate_log_loss"],
            "hmm_top1_accuracy": aggregate["top1_accuracy_hmm"],
            "hmm_route_continuity": aggregate["route_continuity_hmm"],
            "hmm_seq_f1": aggregate["seq_f1_hmm"],
            "fit_seconds": float("nan"), "inference_seconds": float("nan"),
            "inference_ms_per_obs": float("nan"),
        },
        "total_seconds": float("nan"),
        "trained_on": "kaggle",
        "per_trip": per_trip,
    }
    (RESULTS / "model_comparison.json").write_text(json.dumps(comp, indent=2))
    (RESULTS / "kaggle_mlp_eval.json").write_text(json.dumps(
        {"aggregate": aggregate, "per_trip": per_trip}, indent=2))
    print("aggregate:", {k: round(v, 4) for k, v in aggregate.items() if isinstance(v, float)})
    print("models:", sorted(comp["models"]))


if __name__ == "__main__":
    main()
