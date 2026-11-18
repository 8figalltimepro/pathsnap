"""Evaluate the exported artifact through the production inference pipeline.

Runs every student-testing trip through predict_trip() -> score_prediction(),
which is the exact code path the FastAPI backend uses.  Writes the numbers used
in the report plus per-trip predictions for the maps.

Also evaluates the distance-only baseline (no learned classifier) with the same
HMM parameters so the report can show the value of learning.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import config
from pathsnap.io import list_trips
from pathsnap.pipeline import PathSnapPipeline

RESULTS = config.ARTIFACTS_DIR / "results"
ARTIFACTS = config.ARTIFACTS_DIR
METRIC_KEYS = ["top1_accuracy", "top1_accuracy_hmm", "macro_f1", "log_loss",
               "f1", "precision", "recall", "route_continuity",
               "route_continuity_hmm", "seq_f1_clf", "seq_f1_hmm",
               "candidate_miss_rate", "scored_fraction", "n_obs", "n_scored"]


def run(pipe, partition, trip_ids, tag):
    per_trip = {}
    timings = []
    preds = {}
    for tid in trip_ids:
        pred = pipe.predict_trip(tid, partition)
        metrics = pipe.score_prediction(pred)
        metrics["timings_ms"] = {k: round(float(v), 1) for k, v in pred["timings"].items()}
        per_trip[tid] = metrics
        timings.append(pred["timings"]["total_ms"])
        preds[f"{tid}_clf"] = pred["pred_clf"].astype(np.int32)
        preds[f"{tid}_hmm"] = pred["pred_hmm"].astype(np.int32)
        print(f"  {tag} {tid}: top1={metrics.get('top1_accuracy', float('nan')):.4f} "
              f"hmm={metrics.get('top1_accuracy_hmm', float('nan')):.4f} "
              f"cont={metrics.get('route_continuity_hmm', float('nan')):.4f} "
              f"seqF1={metrics.get('seq_f1_hmm', float('nan')):.4f} "
              f"total={pred['timings']['total_ms']:.0f}ms", flush=True)

    def agg(key):
        vals = [per_trip[t][key] for t in trip_ids
                if isinstance(per_trip[t].get(key), (int, float))
                and not (isinstance(per_trip[t].get(key), float) and np.isnan(per_trip[t][key]))]
        return float(np.mean(vals)) if vals else float("nan")

    aggregate = {k: agg(k) for k in METRIC_KEYS}
    aggregate["sum_scored"] = int(sum(per_trip[t]["n_scored"] for t in trip_ids))
    aggregate["mean_total_ms"] = float(np.mean(timings))
    return {"aggregate": aggregate, "per_trip": per_trip, "predictions": preds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--baseline", action="store_true", help="also evaluate the distance baseline")
    ap.add_argument("--out", default="final_eval.json", help="results file name in artifacts/results")
    args = ap.parse_args()
    t0 = time.perf_counter()
    trip_ids = list_trips(args.partition)
    if not trip_ids:
        raise SystemExit(f"no trips in partition {args.partition}")

    pipe = PathSnapPipeline.load()
    print(f"mode={pipe.mode} hmm={pipe.hmm_params} trips={len(trip_ids)}", flush=True)
    trained = run(pipe, args.partition, trip_ids, "trained")
    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "partition": args.partition,
        "mode": pipe.mode,
        "artifact_metadata": pipe.metadata,
        "hmm_params": pipe.hmm_params,
        "aggregate": trained["aggregate"],
        "per_trip": trained["per_trip"],
        "evaluation_seconds": round(time.perf_counter() - t0, 1),
    }
    stem = Path(args.out).stem
    np.savez_compressed(RESULTS / f"{stem}_predictions.npz", **trained["predictions"])

    if args.baseline:
        baseline_pipe = PathSnapPipeline(
            {"model": None, "feature_names": [], "hmm": pipe.hmm_params,
             "metadata": {"model_name": "distance_baseline"}}, ARTIFACTS)
        base = run(baseline_pipe, args.partition, trip_ids, "baseline")
        out["baseline"] = {"aggregate": base["aggregate"], "per_trip": base["per_trip"]}
        np.savez_compressed(RESULTS / f"{stem}_baseline_predictions.npz", **base["predictions"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(out, indent=2))
    print("\nFINAL AGGREGATE:", json.dumps({
        k: (round(v, 4) if isinstance(v, float) else v)
        for k, v in trained["aggregate"].items()}, indent=2))
    if args.baseline:
        print("BASELINE AGGREGATE:", json.dumps({
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in out["baseline"]["aggregate"].items()}, indent=2))
    print(f"wrote {RESULTS / args.out} ({out['evaluation_seconds']}s)")


if __name__ == "__main__":
    main()
