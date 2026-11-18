"""Build the cached candidate-classification dataset for all trips.

For every GPS fix we generate nearby road-arc candidates, extract features and
derive the reference label (nearest arc of the reference route, with a soft
monotonic constraint).  The output cache is what the model experiments read.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap import config
from pathsnap.candidates import TripNetwork, track_points_xy
from pathsnap.features import build_features
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import label_diagnostics, label_observations

CACHE_DIR = config.ARTIFACTS_DIR / "cache"


def build_one(trip_id: str, partition: str) -> dict:
    t0 = time.perf_counter()
    trip = load_trip(trip_id, partition)
    net = TripNetwork(trip)
    obs_xy = track_points_xy(net, trip)
    cand = net.candidates(obs_xy)
    if len(cand) == 0:
        raise RuntimeError(f"{partition}/{trip_id}: no candidates generated")
    labels = label_observations(net, trip, obs_xy)
    feat = build_features(net, trip, cand)
    y = np.full(len(cand), -1, dtype=np.int8)
    valid_rows = labels.valid[cand.obs_idx]
    is_true = cand.arc == labels.arc[cand.obs_idx]
    y[valid_rows] = is_true[valid_rows].astype(np.int8)
    out = CACHE_DIR / f"{partition}_{trip_id}.npz"
    np.savez_compressed(
        out,
        X=feat["X"],
        y=y,
        obs_idx=cand.obs_idx.astype(np.int32),
        arc=cand.arc.astype(np.int32),
        dist=cand.dist.astype(np.float32),
        t=cand.t.astype(np.float32),
        rank=cand.rank.astype(np.int16),
        n_cand=cand.n_cand.astype(np.int16),
        n_obs=np.int64(len(trip.track)),
        label_arc=labels.arc.astype(np.int32),
        label_valid=labels.valid,
        frame=np.asarray(net.frame, dtype=np.float64),
    )
    diag = label_diagnostics(labels, trip)
    n_labelled = int(labels.valid.sum())
    n_pos = int((y == 1).sum())
    diag.update({
        "partition": partition, "trip_id": trip_id,
        "n_candidate_rows": int(len(cand)),
        "n_positive_rows": n_pos,
        "candidate_miss_rate": (1.0 - n_pos / n_labelled) if n_labelled else float("nan"),
        "route_arcs": int(len(trip.route)) if trip.has_route else 0,
        "seconds": round(time.perf_counter() - t0, 3),
    })
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    t_all = time.perf_counter()
    for partition in args.partitions:
        trips = list_trips(partition)
        print(f"[{partition}] {len(trips)} trips")
        for i, tid in enumerate(trips):
            out = CACHE_DIR / f"{partition}_{tid}.npz"
            if out.exists() and not args.force:
                print(f"  [{i+1}/{len(trips)}] {tid} cached", flush=True)
                continue
            diag = build_one(tid, partition)
            report.append(diag)
            print(f"  [{i+1}/{len(trips)}] {tid} obs={diag['n_obs']} rows={diag['n_candidate_rows']} "
                  f"labelled={diag['labelled_fraction']:.3f} miss={diag['candidate_miss_rate']:.3f} "
                  f"{diag['seconds']}s", flush=True)
    results_path = config.ARTIFACTS_DIR / "results" / "dataset_build.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if results_path.exists():
        prev = json.loads(results_path.read_text())
    by_key = {(d["partition"], d["trip_id"]): d for d in prev.get("trips", [])}
    for d in report:
        by_key[(d["partition"], d["trip_id"])] = d
    trips = sorted(by_key.values(), key=lambda d: (d["partition"], d["trip_id"]))
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_trips": len(trips),
        "n_obs": int(sum(d["n_obs"] for d in trips)),
        "n_candidate_rows": int(sum(d["n_candidate_rows"] for d in trips)),
        "n_positive_rows": int(sum(d["n_positive_rows"] for d in trips)),
        "mean_labelled_fraction": float(np.mean([d["labelled_fraction"] for d in trips])),
        "mean_candidate_miss_rate": float(np.mean([d["candidate_miss_rate"] for d in trips])),
        "candidate_radius_m": config.CANDIDATE_RADIUS_M,
        "max_candidates": config.MAX_CANDIDATES,
        "total_seconds": round(time.perf_counter() - t_all, 1),
        "trips": trips,
    }
    results_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "trips"}, indent=2))
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
