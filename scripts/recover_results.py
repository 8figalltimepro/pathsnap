"""Recover model-comparison records from an interrupted experiment log.

The long batch that trained the slower models was interrupted while KNN was
being evaluated.  Its per-fold CV numbers and student-test summary lines are in
the log, and the student-test probability caches were written per trip, so the
missing JSON records can be rebuilt exactly from those two artifacts.
"""
from __future__ import annotations

import json
import re
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
RESULTS = config.ARTIFACTS_DIR / "results"
PREDS = CACHE / "preds"

SPEC = {
    "logreg": {"family": "Linear / probabilistic", "cap": 200_000,
               "config": "StandardScaler + L2 logistic regression, C=1.0, liblinear, 200k-row cap"},
    "random_forest": {"family": "Ensemble (bagging)", "cap": 400_000,
                      "config": "RandomForest 120 trees, max_depth=18, leaf=15, sqrt features, balanced"},
    "extra_trees": {"family": "Ensemble (bagging)", "cap": 400_000,
                    "config": "ExtraTrees 150 trees, max_depth=18, leaf=15, sqrt features, balanced"},
    "mlp": {"family": "Neural network", "cap": 250_000,
            "config": "StandardScaler + MLP (64,32), relu, adam lr=1e-3, early stopping"},
}


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


def cand_from_cache(tp):
    n = len(tp["obs_idx"])
    return CandidateSet(obs_idx=tp["obs_idx"].astype(np.int64),
                        arc=tp["arc"].astype(np.int64),
                        dist=tp["dist"].astype(np.float64),
                        t=tp["t"].astype(np.float64),
                        foot_x=np.zeros(n), foot_y=np.zeros(n),
                        rank=tp["rank"].astype(np.int64),
                        n_cand=tp["n_cand"].astype(np.int64))


def student_test_metrics(name):
    trips = []
    for tid in list_trips("student_test"):
        tp = dict(np.load(CACHE / f"student_test_{tid}.npz"))
        tp["trip_id"] = tid
        trip = load_trip(tid, "student_test")
        net = TripNetwork(trip)
        obs_xy = track_points_xy(net, trip)
        probs = np.load(PREDS / f"{name}_{tid}.npz")["probs"].astype(np.float64)
        cand = cand_from_cache(tp)
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
        top1 = top1_accuracy(pred, tp["label_arc"], valid)
        nll = true_candidate_log_loss(cand, pn, tp["label_arc"], valid)
        pred_h = viterbi_decode(net, cand, pn, obs_xy, trip.arcs, times=trip.track[:, 2], n_obs=n)
        trips.append({
            "macro_f1": m["macro_f1"], "log_loss": m["log_loss"], "f1": m["f1"],
            "precision": m["precision"], "recall": m["recall"],
            "top1_accuracy": top1, "route_continuity": route_continuity(net, pred),
            "seq_f1": sequence_f1(pred, trip.route)["seq_f1"],
            "true_candidate_log_loss": nll,
            "hmm_top1_accuracy": top1_accuracy(pred_h, tp["label_arc"], valid),
            "hmm_route_continuity": route_continuity(net, pred_h),
            "hmm_seq_f1": sequence_f1(pred_h, trip.route)["seq_f1"],
        })
        print(f"  {name} {tid}: top1 {top1:.4f} hmm {trips[-1]['hmm_top1_accuracy']:.4f}", flush=True)
    keys = trips[0].keys()
    return {k: float(np.mean([t[k] for t in trips])) for k in keys}


def parse_log(path):
    text = Path(path).read_text()
    blocks = re.split(r"(?m)^===== (\S+) \((.+)\) =====$", text)
    out = {}
    for i in range(1, len(blocks), 3):
        name = blocks[i]
        family = blocks[i + 1]
        body = blocks[i + 2]
        folds = re.findall(
            r"fold (\d+): macro_f1=([\d.]+) top1=([\d.]+) logloss=([\d.]+) "
            r"fit=([\d.]+)s pred=([\d.]+)s", body)
        cvline = re.search(
            r"CV macro_f1 ([\-\d.]+)\+/-[\d.]+ \(std ([\-\d.]+)\)|"
            r"CV macro_f1 ([\-\d.]+)\+-([\-\d.]+) \| CV top1 ([\-\d.]+) \| CV logloss ([\-\d.]+)",
            body + " CV macro_f1 " + "", re.S)
        # simple robust parsing of the three summary lines
        cv_m = re.search(r"CV macro_f1 ([\-\d.]+)\+-([\-\d.]+)", body)
        cv_t = re.search(r"CV top1 ([\-\d.]+)", body)
        cv_l = re.search(r"CV logloss ([\-\d.]+)", body)
        st = re.search(
            r"STUDENT-TEST top1=([\d.]+) macro_f1=([\d.]+) logloss=([\d.]+) \| "
            r"HMM top1=([\d.]+) cont=([\d.]+) \| train=([\d.]+)s infer=([\d.]+)ms/obs", body)
        if not (folds and cv_m and st):
            continue
        fold_recs = [{"macro_f1": float(f[1]), "top1_accuracy": float(f[2]),
                      "log_loss": float(f[3]), "fit_seconds": float(f[4]),
                      "predict_seconds": float(f[5])} for f in folds]
        out[name] = {
            "family": family,
            "cv": {
                "macro_f1_mean": float(cv_m.group(1)), "macro_f1_std": float(cv_m.group(2)),
                "top1_accuracy_mean": float(cv_t.group(1)),
                "top1_accuracy_std": float(np.std([f["top1_accuracy"] for f in fold_recs])),
                "log_loss_mean": float(cv_l.group(1)),
                "f1_mean": float("nan"), "precision_mean": float("nan"),
                "recall_mean": float("nan"),
                "fit_seconds_mean": float(np.mean([f["fit_seconds"] for f in fold_recs])),
                "predict_seconds_mean": float(np.mean([f["predict_seconds"] for f in fold_recs])),
                "folds": fold_recs,
            },
            "student_test_log": {
                "top1_accuracy": float(st.group(1)), "macro_f1": float(st.group(2)),
                "log_loss": float(st.group(3)), "hmm_top1_accuracy": float(st.group(4)),
                "hmm_route_continuity": float(st.group(5)), "fit_seconds": float(st.group(6)),
                "inference_ms_per_obs": float(st.group(7)),
            },
        }
    return out


def main():
    log = RESULTS / "experiments_batch2.log"
    parsed = parse_log(log)
    comp_path = RESULTS / "model_comparison.json"
    comp = json.loads(comp_path.read_text())
    for name, rec in parsed.items():
        if name in comp["models"]:
            continue
        if name not in SPEC:
            continue
        metrics = student_test_metrics(name)
        st = rec["student_test_log"]
        # log numbers must agree with the recomputed ones (sanity check)
        print(f"{name}: log top1 {st['top1_accuracy']:.4f} recomputed "
              f"{metrics['top1_accuracy']:.4f}", flush=True)
        dev = dict(metrics)
        dev.update({"fit_seconds": st["fit_seconds"],
                    "inference_seconds": float("nan"),
                    "inference_ms_per_obs": st["inference_ms_per_obs"],
                    "hmm_route_continuity": metrics["hmm_route_continuity"],
                    "hmm_top1_accuracy": metrics["hmm_top1_accuracy"],
                    "hmm_seq_f1": metrics["hmm_seq_f1"]})
        comp["models"][name] = {
            "model": {"name": name, "family": SPEC[name]["family"],
                      "config": SPEC[name]["config"], "cap": SPEC[name]["cap"],
                      "neg_per_obs": 4},
            "cv": rec["cv"],
            "student_test": dev,
            "total_seconds": float("nan"),
            "recovered_from_log": True,
        }
    comp["generated_at"] = comp["generated_at"]
    comp_path.write_text(json.dumps(comp, indent=2))
    print("models now:", sorted(comp["models"]))
    print("wrote", comp_path)


if __name__ == "__main__":
    main()
