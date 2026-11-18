"""Export a curated set of trained candidate classifiers for the live UI picker.

The model-comparison experiment trained and evaluated twelve models, but only the
final LightGBM artifact was kept.  This script refits the deployable models with
the identical training protocol, evaluates each on the development partition
(same metrics the API computes live) and saves one joblib per model under
artifacts/models/.  The API then lets the evaluator switch models.

RandomForest, ExtraTrees and KNN stay in the report's comparison table only:
their artifacts are large or their inference is slow, so they are not shipped to
the live picker.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_experiments as R  # noqa: E402
from pathsnap import config  # noqa: E402
from pathsnap.candidates import TripNetwork, track_points_xy  # noqa: E402
from pathsnap.hmm import viterbi_decode  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.metrics import (candidate_miss_rate, candidate_probability_metrics,  # noqa: E402
                              route_continuity, sequence_f1, top1_accuracy,
                              true_candidate_log_loss)

RESULTS = config.ARTIFACTS_DIR / "results"
MODELS = config.ARTIFACTS_DIR / "models"

DISPLAY = {
    "lightgbm_tuned": ("LightGBM (tuned)", "Boosting"),
    "lightgbm": ("LightGBM (default)", "Boosting"),
    "xgboost": ("XGBoost", "Boosting"),
    "hist_gb": ("HistGradientBoosting", "Boosting"),
    "decision_tree": ("Decision Tree", "Tree"),
    "mlp": ("MLP (scikit-learn)", "Neural network"),
    "logreg": ("Logistic Regression", "Linear"),
    "gaussian_nb": ("Gaussian Naive Bayes", "Probabilistic"),
}
DEPLOYABLE = list(DISPLAY)


def build_training_data():
    R.load_feature_names()
    Xs, ys = [], []
    for tid in list_trips("train"):
        tp = R.load_cache("train", tid)
        rng = np.random.default_rng(config.RANDOM_SEED + int(tid))
        sel = R.subsample_trip(tp, 4, rng)
        Xs.append(tp["X"][sel])
        ys.append(tp["y"][sel])
    return np.concatenate(Xs), np.concatenate(ys).astype(np.int64)


def load_dev_trips():
    out = []
    for tid in list_trips("student_test"):
        tp = R.load_cache("student_test", tid)
        trip = load_trip(tid, "student_test")
        net = TripNetwork(trip)
        out.append({"tp": tp, "trip": trip, "net": net,
                    "obs_xy": track_points_xy(net, trip),
                    "cand": R.cand_from_cache(tp)})
    return out


def evaluate_model(model, dev, hmm_params):
    per = []
    t_inf = time.perf_counter()
    for d in dev:
        tp = d["tp"]
        probs = R.predict_probs(model, tp["X"])
        pn = R.normalised_probs(d["cand"], probs)
        valid = tp["label_valid"].astype(bool)
        m = candidate_probability_metrics(d["cand"], pn, tp["label_arc"], valid)
        n = int(tp["n_obs"])
        pred = np.full(n, -1, dtype=np.int64)
        counts = np.bincount(d["cand"].obs_idx, minlength=n)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for i in range(n):
            s, c = starts[i], counts[i]
            if c:
                pred[i] = d["cand"].arc[s + int(np.argmax(probs[s:s + c]))]
        pred_h = viterbi_decode(d["net"], d["cand"], pn, d["obs_xy"], d["trip"].arcs,
                                beta=float(hmm_params["beta"]),
                                emission_alpha=float(hmm_params["emission_alpha"]),
                                max_states=int(hmm_params["max_states"]),
                                gap_free_s=float(hmm_params["gap_free_s"]),
                                times=d["trip"].track[:, 2], n_obs=n)
        per.append({
            "top1_accuracy": top1_accuracy(pred, tp["label_arc"], valid),
            "top1_accuracy_hmm": top1_accuracy(pred_h, tp["label_arc"], valid),
            "macro_f1": m["macro_f1"], "log_loss": m["log_loss"],
            "route_continuity": route_continuity(d["net"], pred),
            "route_continuity_hmm": route_continuity(d["net"], pred_h),
            "seq_f1_clf": sequence_f1(pred, d["trip"].route)["seq_f1"],
            "seq_f1_hmm": sequence_f1(pred_h, d["trip"].route)["seq_f1"],
            "candidate_miss_rate": candidate_miss_rate(d["cand"], tp["label_arc"], valid),
            "n_obs": n, "n_scored": int(valid.sum()),
            "true_candidate_log_loss": true_candidate_log_loss(d["cand"], pn, tp["label_arc"], valid),
        })
    inf_s = time.perf_counter() - t_inf
    n_obs = sum(p["n_obs"] for p in per)
    agg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    agg["inference_ms_per_obs"] = 1000.0 * inf_s / max(n_obs, 1)
    return agg, per


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    comp = json.loads((RESULTS / "model_comparison.json").read_text())
    hmm_best = json.loads((RESULTS / "hmm_tuning_best.json").read_text())["best"]
    hmm = {"beta": float(hmm_best["beta"]),
           "emission_alpha": float(hmm_best["alpha"]),
           "max_states": int(hmm_best["max_states"]),
           "gap_free_s": float(hmm_best["gap_free_s"])}
    dev = load_dev_trips()
    X, y = build_training_data()
    print(f"training rows: {len(X)} ({int(y.sum())} positives)", flush=True)
    zoo = R.model_zoo()
    registry = []
    for name in DEPLOYABLE:
        t0 = time.perf_counter()
        out_path = MODELS / f"{name}.joblib"
        if name == "lightgbm_tuned":
            art = joblib.load(config.ARTIFACTS_DIR / "model.joblib")
            model = art["model"]
        else:
            spec = zoo[name]
            rng = np.random.default_rng(config.RANDOM_SEED + 7)
            sel = R.cap_rows(X, y, spec["cap"], rng)
            model = spec["build"]()
            model.fit(X[sel], y[sel])
        agg, _ = evaluate_model(model, dev, hmm)
        cv = comp["models"].get(name, {}).get("cv", {})
        display, family = DISPLAY[name]
        metadata = {
            "id": name, "name": display, "family": family,
            "config": comp["models"].get(name, {}).get("model", {}).get("config", ""),
            "cv_macro_f1": float(cv.get("macro_f1_mean", float("nan"))),
            "cv_macro_f1_std": float(cv.get("macro_f1_std", float("nan"))),
            "test_top1": agg["top1_accuracy"], "test_top1_hmm": agg["top1_accuracy_hmm"],
            "test_macro_f1": agg["macro_f1"], "test_log_loss": agg["log_loss"],
            "test_route_continuity": agg["route_continuity"],
            "test_route_continuity_hmm": agg["route_continuity_hmm"],
            "inference_ms_per_obs": agg["inference_ms_per_obs"],
            "trained_on": "80 training trips",
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if name == "lightgbm_tuned":
            art["metadata"].update(metadata)
            art["metadata"]["trained_on"] = "final artifact (80 training trips)"
            art["hmm"] = hmm
            joblib.dump(art, out_path)
        else:
            joblib.dump({"model": model, "feature_names": R.FEATURE_NAMES, "hmm": hmm,
                         "metadata": metadata}, out_path)
        size_mb = out_path.stat().st_size / 1e6
        registry.append({"id": name, "name": display, "family": family,
                         "cv_macro_f1": metadata["cv_macro_f1"],
                         "cv_macro_f1_std": metadata["cv_macro_f1_std"],
                         "test_top1": metadata["test_top1"],
                         "test_top1_hmm": metadata["test_top1_hmm"],
                         "test_macro_f1": metadata["test_macro_f1"],
                         "test_log_loss": metadata["test_log_loss"],
                         "inference_ms_per_obs": metadata["inference_ms_per_obs"],
                         "size_mb": round(size_mb, 2),
                         "description": metadata["config"]})
        print(f"{name:16s} cv_macro={metadata['cv_macro_f1']:.4f} "
              f"test_top1={metadata['test_top1']:.4f} hmm={metadata['test_top1_hmm']:.4f} "
              f"size={size_mb:.1f}MB ({time.perf_counter()-t0:.1f}s)", flush=True)

    registry.sort(key=lambda r: r["cv_macro_f1"], reverse=True)
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "default": "lightgbm_tuned", "hmm": hmm, "models": registry}
    (RESULTS / "model_registry.json").write_text(json.dumps(payload, indent=2))
    print("wrote", RESULTS / "model_registry.json")


if __name__ == "__main__":
    main()
