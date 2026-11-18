"""Tune the HMM / Viterbi decoder on the student-testing partition.

The decoder parameters (transition scale beta, emission sharpening alpha,
number of candidate states) are optimised for observation-level top-1 accuracy
on the 10 development trips.  Student-testing data is permitted for decoder
tuning by the hackathon rules; the evaluator partition is never touched here.

Usage:
    python scripts/tune_hmm.py --probs artifacts/cache/preds/lightgbm_00000000.npz ...
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import config
from pathsnap.candidates import CandidateSet, TripNetwork, track_points_xy
from pathsnap.hmm import viterbi_decode
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import route_continuity, sequence_f1, top1_accuracy

CACHE = config.ARTIFACTS_DIR / "cache"
RESULTS = config.ARTIFACTS_DIR / "results"


def cand_from_cache(tp):
    n = len(tp["obs_idx"])
    return CandidateSet(
        obs_idx=tp["obs_idx"].astype(np.int64),
        arc=tp["arc"].astype(np.int64),
        dist=tp["dist"].astype(np.float64),
        t=tp["t"].astype(np.float64),
        foot_x=np.zeros(n), foot_y=np.zeros(n),
        rank=tp["rank"].astype(np.int64),
        n_cand=tp["n_cand"].astype(np.int64),
    )


def _read_probs_file(path: Path, trip_ids: list[str]) -> dict[str, np.ndarray]:
    z = np.load(path)
    files = set(z.files)
    if all(t in files for t in trip_ids):
        return {t: z[t] for t in trip_ids}
    if "probs" in files:
        tid = path.stem.split("_")[-1]
        return {tid: z["probs"]} if tid in trip_ids else {}
    raise ValueError(f"unrecognised probability file {path}: keys {z.files}")


def load_probs(pattern: str, trip_ids: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "{trip}" in pattern:
        for tid in trip_ids:
            path = Path(pattern.replace("{trip}", tid))
            if not path.exists():
                raise FileNotFoundError(f"no probability cache for trip {tid}: {path}")
            out.update(_read_probs_file(path, trip_ids))
        return out
    path = Path(pattern)
    if path.is_dir():
        for p in sorted(path.glob("*.npz")):
            out.update(_read_probs_file(p, trip_ids))
    elif path.is_file():
        out.update(_read_probs_file(path, trip_ids))
    else:
        raise FileNotFoundError(f"probability source not found: {pattern}")
    missing = [t for t in trip_ids if t not in out]
    if missing:
        raise FileNotFoundError(f"missing probabilities for {missing}")
    return out


def load_dev_trips():
    trips = []
    for tid in list_trips("student_test"):
        tp = dict(np.load(CACHE / f"student_test_{tid}.npz"))
        tp["trip_id"] = tid
        trip = load_trip(tid, "student_test")
        net = TripNetwork(trip)
        obs_xy = track_points_xy(net, trip)
        trips.append({"tp": tp, "trip": trip, "net": net, "obs_xy": obs_xy,
                      "cand": cand_from_cache(tp)})
    return trips


def evaluate(trips, probs_by_trip, params):
    top1, cont, seq = [], [], []
    for tr in trips:
        tid = tr["tp"]["trip_id"]
        n = int(tr["tp"]["n_obs"])
        pred = viterbi_decode(tr["net"], tr["cand"], probs_by_trip[tid], tr["obs_xy"],
                              tr["trip"].arcs, beta=params["beta"],
                              emission_alpha=params["alpha"],
                              max_states=params["max_states"],
                              gap_free_s=params["gap_free_s"],
                              times=tr["trip"].track[:, 2], n_obs=n)
        valid = tr["tp"]["label_valid"].astype(bool)
        top1.append(top1_accuracy(pred, tr["tp"]["label_arc"], valid))
        cont.append(route_continuity(tr["net"], pred))
        seq.append(sequence_f1(pred, tr["trip"].route)["seq_f1"])
    return {"top1_accuracy": float(np.mean(top1)),
            "route_continuity": float(np.mean(cont)),
            "seq_f1": float(np.mean(seq))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True,
                    help="npz with per-trip probabilities (keys = trip ids) or glob pattern")
    ap.add_argument("--models", action="store_true", help="run the full grid, otherwise the quick grid")
    ap.add_argument("--out", default="hmm_tuning.json")
    args = ap.parse_args()
    t0 = time.perf_counter()
    trips = load_dev_trips()
    trip_ids = [tr["tp"]["trip_id"] for tr in trips]
    probs = load_probs(args.probs, trip_ids)
    # classifier-only reference
    clf_top1, clf_cont, clf_seq = [], [], []
    for tr in trips:
        tid = tr["tp"]["trip_id"]
        cand = tr["cand"]
        pr = probs[tid]
        n = int(tr["tp"]["n_obs"])
        pred = np.full(n, -1, dtype=np.int64)
        counts = np.bincount(cand.obs_idx, minlength=n)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for i in range(n):
            s = starts[i]; c = counts[i]
            if c:
                pred[i] = cand.arc[s + int(np.argmax(pr[s:s + c]))]
        valid = tr["tp"]["label_valid"].astype(bool)
        clf_top1.append(top1_accuracy(pred, tr["tp"]["label_arc"], valid))
        clf_cont.append(route_continuity(tr["net"], pred))
        clf_seq.append(sequence_f1(pred, tr["trip"].route)["seq_f1"])
    classifier_only = {"top1_accuracy": float(np.mean(clf_top1)),
                       "route_continuity": float(np.mean(clf_cont)),
                       "seq_f1": float(np.mean(clf_seq))}
    print("classifier-only:", {k: round(v, 4) for k, v in classifier_only.items()}, flush=True)

    if args.models:
        betas = [8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0, 30.0]
        alphas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
        states = [8, 10]
    else:
        betas = [12.0, 20.0, 30.0]
        alphas = [0.5, 1.0, 2.0]
        states = [8]
    gap = 30.0
    rows = []
    best = None
    for beta in betas:
        for alpha in alphas:
            for k in states:
                params = {"beta": beta, "alpha": alpha, "max_states": k, "gap_free_s": gap}
                m = evaluate(trips, probs, params)
                row = {**params, **m}
                rows.append(row)
                print(f"  beta={beta:5.1f} alpha={alpha:4.1f} states={k:2d} "
                      f"top1={m['top1_accuracy']:.4f} cont={m['route_continuity']:.4f} "
                      f"seqF1={m['seq_f1']:.4f}", flush=True)
                if best is None or m["top1_accuracy"] > best["top1_accuracy"]:
                    best = {**params, **m}
    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "probs": str(args.probs),
        "objective": "mean student-test observation-level top-1 accuracy",
        "classifier_only": classifier_only,
        "best": best,
        "grid": rows,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(out, indent=2))
    print("\nBEST:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()})
    print("wrote", RESULTS / args.out, f"({out['seconds']}s)")


if __name__ == "__main__":
    main()
