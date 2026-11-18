"""Fit the final model and export the inference artifact.

The final model is the Bayesian-tuned LightGBM candidate classifier.  It is
trained with the same sampling protocol as the model-comparison experiment
(4 negatives per positive row), refit on the 80 training trips, and exported
with its feature list and tuned HMM parameters into artifacts/model.joblib.

Also records the tuned-versus-untuned comparison required by the report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import config
from pathsnap.features import FEATURE_NAMES
from pathsnap.io import list_trips
from pathsnap.pipeline import PathSnapPipeline

CACHE = config.ARTIFACTS_DIR / "cache"
RESULTS = config.ARTIFACTS_DIR / "results"
ARTIFACTS = config.ARTIFACTS_DIR
PREDS = CACHE / "preds"


def subsample_trip(tp, neg_per_obs, rng):
    """Identical negative sampling to scripts/run_experiments.py (the protocol
    every compared model uses), so the exported artifact is the same model that
    was evaluated in the comparison table."""
    y = tp["y"]; obs = tp["obs_idx"]
    keep_pos = np.where(y == 1)[0]
    neg_rows = np.where(y == 0)[0]
    if len(neg_rows) == 0:
        return keep_pos
    obs_of_neg = obs[neg_rows]
    order = np.argsort(obs_of_neg, kind="stable")
    neg_rows = neg_rows[order]; obs_of_neg = obs_of_neg[order]
    _, starts, counts = np.unique(obs_of_neg, return_index=True, return_counts=True)
    picks = []
    for s, c in zip(starts, counts):
        k = min(c, neg_per_obs)
        if k == c:
            picks.append(neg_rows[s:s + c])
        else:
            picks.append(neg_rows[s + rng.choice(c, size=k, replace=False)])
    return np.concatenate([keep_pos] + picks)


def cap_rows(X, y, max_rows, rng):
    if max_rows is None or len(X) <= max_rows:
        return np.arange(len(X))
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_neg = max(max_rows - len(pos), 0)
    if n_neg >= len(neg):
        return np.arange(len(X))
    sel = rng.choice(neg, size=n_neg, replace=False)
    return np.concatenate([pos, sel])


def load_train_rows(neg_per_obs=4, seed=config.RANDOM_SEED, cap=900_000):
    Xs, ys = [], []
    for tid in list_trips("train"):
        z = np.load(CACHE / f"train_{tid}.npz")
        X, y, obs = z["X"], z["y"], z["obs_idx"]
        usable = y >= 0
        tp = {"X": X[usable], "y": y[usable], "obs_idx": obs[usable]}
        rng = np.random.default_rng(seed + int(tid))
        sel = subsample_trip(tp, neg_per_obs, rng)
        Xs.append(tp["X"][sel]); ys.append(tp["y"][sel])
    X = np.concatenate(Xs)
    y = np.concatenate(ys).astype(np.int64)
    rng = np.random.default_rng(seed + 7)
    sel = cap_rows(X, y, cap, rng)
    return X[sel], y[sel]


def load_tuned_params():
    for name in ("local_optuna.json", "kaggle_optuna.json"):
        p = RESULTS / name
        if p.exists():
            data = json.loads(p.read_text())
            params = data.get("best_params") or (data.get("best") or {}).get("params")
            if params:
                return params, name
    return None, None


def load_hmm_params():
    for name in ("hmm_tuning_best.json", "hmm_tuning_quick.json"):
        p = RESULTS / name
        if p.exists():
            data = json.loads(p.read_text())
            best = data.get("best") or {}
            if "beta" in best:
                return {k: float(best[k]) if isinstance(best[k], (int, float)) else best[k]
                        for k in ("beta", "emission_alpha" if "emission_alpha" in best else "alpha",
                                  "max_states", "gap_free_s") if k in best}
    return {"beta": config.HMM_BETA_M, "emission_alpha": config.HMM_EMISSION_ALPHA,
            "max_states": config.HMM_MAX_STATES, "gap_free_s": config.HMM_GAP_FREE_S}


def hmm_params_from_file():
    p = RESULTS / "hmm_tuning_best.json"
    if not p.exists():
        p = RESULTS / "hmm_tuning_quick.json"
    if not p.exists():
        return {}
    best = json.loads(p.read_text()).get("best", {})
    out = {}
    if "beta" in best:
        out["beta"] = float(best["beta"])
    if "alpha" in best:
        out["emission_alpha"] = float(best["alpha"])
    if "emission_alpha" in best:
        out["emission_alpha"] = float(best["emission_alpha"])
    if "max_states" in best:
        out["max_states"] = int(best["max_states"])
    if "gap_free_s" in best:
        out["gap_free_s"] = float(best["gap_free_s"])
    return out


def evaluate_model(model, hmm_params, name, dev_ids):
    pipe = PathSnapPipeline({"model": model, "feature_names": FEATURE_NAMES,
                             "hmm": hmm_params, "metadata": {"name": name}}, ARTIFACTS)
    per_trip = {}
    for tid in dev_ids:
        pred = pipe.predict_trip(tid, "student_test")
        metrics = pipe.score_prediction(pred)
        metrics["timings_ms"] = {k: round(float(v), 1) for k, v in pred["timings"].items()}
        per_trip[tid] = metrics
    def agg(key):
        vals = [per_trip[t][key] for t in dev_ids if isinstance(per_trip[t].get(key), float)]
        return float(np.mean(vals)) if vals else float("nan")
    keys = ["top1_accuracy", "top1_accuracy_hmm", "macro_f1", "log_loss",
            "route_continuity", "route_continuity_hmm", "seq_f1_clf", "seq_f1_hmm",
            "candidate_miss_rate", "scored_fraction"]
    return {k: agg(k) for k in keys}, per_trip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-per-obs", type=int, default=4)
    ap.add_argument("--skip-untuned", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    import lightgbm as lgb

    tuned_params, params_source = load_tuned_params()
    if tuned_params is None:
        raise SystemExit("no tuned parameters found (kaggle_optuna.json / local_optuna.json)")
    tuned_params = dict(tuned_params)
    tuned_params.update({"n_jobs": -1, "random_state": config.RANDOM_SEED, "verbosity": -1})
    print("tuned params from", params_source, ":", tuned_params, flush=True)

    X, y = load_train_rows(args.neg_per_obs)
    print(f"training rows: {len(X)} ({int(y.sum())} positives)", flush=True)

    untuned = lgb.LGBMClassifier(
        n_estimators=350, learning_rate=0.08, num_leaves=63, min_child_samples=20,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        n_jobs=-1, random_state=config.RANDOM_SEED, verbosity=-1)
    t1 = time.perf_counter()
    untuned.fit(X, y)
    untuned_fit = time.perf_counter() - t1
    print(f"untuned LightGBM fit {untuned_fit:.1f}s", flush=True)

    tuned = lgb.LGBMClassifier(**tuned_params)
    t1 = time.perf_counter()
    tuned.fit(X, y)
    tuned_fit = time.perf_counter() - t1
    print(f"tuned LightGBM fit {tuned_fit:.1f}s", flush=True)

    hmm_params = hmm_params_from_file()
    if not hmm_params:
        hmm_params = {"beta": config.HMM_BETA_M, "emission_alpha": config.HMM_EMISSION_ALPHA,
                      "max_states": config.HMM_MAX_STATES, "gap_free_s": config.HMM_GAP_FREE_S}
    print("HMM params:", hmm_params, flush=True)

    dev_ids = list_trips("student_test")
    comparison = {}
    models = {"lightgbm_untuned": (untuned, untuned_fit)}
    if not args.skip_untuned:
        pass
    models["lightgbm_tuned"] = (tuned, tuned_fit)
    per_trip_all = {}
    for name, (model, fit_s) in models.items():
        agg, per_trip = evaluate_model(model, hmm_params, name, dev_ids)
        comparison[name] = {"aggregate": agg, "fit_seconds": round(fit_s, 1),
                            "params": (tuned_params if name == "lightgbm_tuned" else
                                       {"n_estimators": 350, "learning_rate": 0.08,
                                        "num_leaves": 63, "min_child_samples": 20,
                                        "subsample": 0.9, "colsample_bytree": 0.9,
                                        "reg_lambda": 1.0})}
        per_trip_all[name] = per_trip
        print(f"{name}: top1={agg['top1_accuracy']:.4f} hmm_top1={agg['top1_accuracy_hmm']:.4f} "
              f"macro_f1={agg['macro_f1']:.4f} logloss={agg['log_loss']:.4f}", flush=True)
    winner = "lightgbm_tuned"
    tuned_hmm = comparison["lightgbm_tuned"]["aggregate"]["top1_accuracy_hmm"]
    untuned_hmm = comparison["lightgbm_untuned"]["aggregate"]["top1_accuracy_hmm"]
    tuned_cv = None
    # keep the tuned model unless the untuned one wins by a meaningful margin;
    # the two are expected to be close, and the tuned model is the deliverable
    # of the Bayesian optimisation step
    if untuned_hmm - tuned_hmm > 0.005:
        winner = "lightgbm_untuned"
    print("selected final model:", winner, flush=True)
    final_model = models[winner][0]

    # probabilities on the development partition (used for HMM tuning)
    PREDS.mkdir(parents=True, exist_ok=True)
    for tid in dev_ids:
        z = np.load(CACHE / f"student_test_{tid}.npz")
        probs = final_model.predict_proba(z["X"])[:, 1]
        np.savez_compressed(PREDS / f"final_{tid}.npz", probs=probs.astype(np.float32))

    metadata = {
        "model_name": winner,
        "model_family": "Boosting (LightGBM, Optuna-tuned)" if winner == "lightgbm_tuned"
        else "Boosting (LightGBM, untuned)",
        "tuned_params": tuned_params,
        "hmm": hmm_params,
        "feature_names": FEATURE_NAMES,
        "neg_per_obs": args.neg_per_obs,
        "train_trips": len(list_trips("train")),
        "train_rows": int(len(X)),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": comparison[winner]["aggregate"],
    }
    artifact = {"model": final_model, "feature_names": FEATURE_NAMES,
                "hmm": hmm_params, "metadata": metadata}
    joblib.dump(artifact, ARTIFACTS / "model.joblib")
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selected": winner,
        "hmm_params": hmm_params,
        "comparison": comparison,
        "per_trip": per_trip_all,
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    (RESULTS / "final_model_selection.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {ARTIFACTS / 'model.joblib'} and {RESULTS / 'final_model_selection.json'} "
          f"({out['total_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
