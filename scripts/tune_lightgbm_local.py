"""Local Bayesian (TPE) hyperparameter search for LightGBM.

This is the fallback twin of the Kaggle Optuna study: same objective and search
space, run on the local machine so the repository can reproduce the tuning even
without Kaggle.  Results go to artifacts/results/local_optuna.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import config
from pathsnap.io import list_trips

CACHE = config.ARTIFACTS_DIR / "cache"
RESULTS = config.ARTIFACTS_DIR / "results"


def load_trip(tid):
    z = np.load(CACHE / f"train_{tid}.npz")
    tp = {k: z[k] for k in z.files}
    tp["trip_id"] = tid
    return tp


def subsample_trip(tp, neg_per_obs, rng):
    y, obs = tp["y"], tp["obs_idx"]
    keep_pos = np.where(y == 1)[0]
    rows = np.where(y == 0)[0]
    obs_neg = obs[rows]
    order = np.argsort(obs_neg, kind="stable")
    rows, obs_neg = rows[order], obs_neg[order]
    _, starts, counts = np.unique(obs_neg, return_index=True, return_counts=True)
    picks = []
    for s, c in zip(starts, counts):
        k = min(c, neg_per_obs)
        picks.append(rows[s:s + c] if k == c else rows[s + rng.choice(c, size=k, replace=False)])
    return np.concatenate([keep_pos] + picks)


def normalised_probs(cand_obs, probs):
    n = int(cand_obs.max()) + 1
    counts = np.bincount(cand_obs, minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out = np.empty(len(probs))
    for s, c in zip(starts, counts):
        seg = probs[s:s + c]
        tot = seg.sum()
        out[s:s + c] = seg / tot if tot > 0 else 1.0 / max(c, 1)
    return out


def fold_score(tp, probs):
    from sklearn.metrics import f1_score
    cand_arc = tp["arc"]
    valid = tp["label_valid"].astype(bool)
    mask = valid[tp["obs_idx"]]
    y = (cand_arc == tp["label_arc"][tp["obs_idx"]])[mask].astype(np.int64)
    pn = normalised_probs(tp["obs_idx"], probs)[mask]
    yhat = (pn >= 0.5).astype(np.int64)
    return float(f1_score(y, yhat, average="macro", zero_division=0))



def fold_score_subset(tp, probs_sub, rows):
    """Macro F1 on a pre-selected subset of candidate rows (fast tuning estimate)."""
    from sklearn.metrics import f1_score
    obs = tp["obs_idx"][rows]
    valid = tp["label_valid"].astype(bool)[obs]
    y = (tp["arc"][rows] == tp["label_arc"][obs]).astype(np.int64)
    pn = normalised_probs(obs, probs_sub)
    yhat = (pn >= 0.5).astype(np.int64)
    return float(f1_score(y[valid], yhat[valid], average="macro", zero_division=0))


def main():
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.perf_counter()
    trips = [load_trip(t) for t in list_trips("train")]
    rng = np.random.default_rng(config.RANDOM_SEED)
    order = rng.permutation(len(trips))
    fold_of = {}
    for pos, idx in enumerate(order):
        fold_of[trips[idx]["trip_id"]] = pos % 3
    data = []
    for tp in trips:
        r = np.random.default_rng(config.RANDOM_SEED + int(tp["trip_id"]))
        sel = subsample_trip(tp, 4, r)
        data.append((tp["X"][sel], tp["y"][sel].astype(np.int64)))
    val_idx = [[i for i, tp in enumerate(trips) if fold_of[tp["trip_id"]] == f]
               for f in range(3)]
    # evaluate on every third observation to keep each Optuna trial cheap; the
    # full 5-fold evaluation is rerun later with the winning configuration
    row_subsets = {}
    for tp in trips:
        n_obs = int(tp["n_obs"])
        keep_obs = np.arange(0, n_obs, 3)
        row_subsets[tp["trip_id"]] = np.where(np.isin(tp["obs_idx"], keep_obs))[0]
    rng_cap = np.random.default_rng(7)

    def cap(X, y, n=300_000):
        if len(X) <= n:
            return X, y
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        sel = np.concatenate([pos, rng_cap.choice(neg, size=max(n - len(pos), 0), replace=False)])
        return X[sel], y[sel]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.18, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 96),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_jobs": 4, "random_state": config.RANDOM_SEED, "verbosity": -1,
        }
        fold_scores = []
        for f in range(3):
            tr = [i for i, tp in enumerate(trips) if fold_of[tp["trip_id"]] != f]
            X = np.concatenate([data[i][0] for i in tr])
            y = np.concatenate([data[i][1] for i in tr])
            X, y = cap(X, y)
            model = lgb.LGBMClassifier(**params)
            model.fit(X, y)
            sc = []
            for i in val_idx[f]:
                tp = trips[i]
                rows = row_subsets[tp["trip_id"]]
                probs = model.predict_proba(tp["X"][rows])[:, 1]
                sc.append(fold_score_subset(tp, probs, rows))
            fold_scores.append(float(np.mean(sc)))
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
    trials_path = RESULTS / "local_optuna_trials.json"

    def _save(study, trial):
        dump = [{"number": t.number, "value": t.value, "params": t.params,
                 "state": str(t.state)} for t in study.trials]
        trials_path.write_text(json.dumps(dump, indent=2))
        print(f"trial {trial.number}: value={trial.value:.4f} best={study.best_value:.4f}", flush=True)

    study.optimize(objective, n_trials=14, timeout=540, callbacks=[_save])
    trials = [{"number": t.number, "value": t.value, "params": t.params,
               "state": str(t.state)} for t in study.trials]
    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objective": "mean 3-fold group-CV candidate macro F1",
        "n_trials": len(study.trials),
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "trials": trials,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "local_optuna.json").write_text(json.dumps(out, indent=2))
    print("best:", study.best_params, "value:", study.best_value)
    print("wrote", RESULTS / "local_optuna.json")


if __name__ == "__main__":
    main()
