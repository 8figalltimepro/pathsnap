"""Train, cross-validate and compare candidate-classification models.

Protocol
--------
* Training data: candidate rows from the 80 training trips.  Negative rows are
  subsampled to 4 per positive row to keep the slower models tractable; the
  sample is fixed per trip so all models see comparable data.
* Cross-validation: 5-fold GroupKFold on trip ids (no trip is split).
* Development evaluation: all 10 student_test trips, full candidate sets.
* Every model is also run through the HMM decoder so classifier-only and
  classifier+HMM results can be compared.

Writes artifacts/results/model_comparison.json and per-model student-test
probability caches used by the tuning/report scripts.
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
from pathsnap.candidates import CandidateSet, TripNetwork, track_points_xy
from pathsnap.hmm import viterbi_decode
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import (candidate_probability_metrics, route_continuity,
                              sequence_f1, top1_accuracy, true_candidate_log_loss)

CACHE = config.ARTIFACTS_DIR / "cache"
RESULTS = config.ARTIFACTS_DIR / "results"
PREDS = CACHE / "preds"
FEATURE_NAMES = None


def load_feature_names():
    global FEATURE_NAMES
    from pathsnap.features import FEATURE_NAMES as FN
    FEATURE_NAMES = FN
    return FN


def load_cache(partition: str, trip_id: str) -> dict:
    z = np.load(CACHE / f"{partition}_{trip_id}.npz")
    tp = {k: z[k] for k in z.files}
    tp["trip_id"] = trip_id
    tp["partition"] = partition
    return tp


def cand_from_cache(tp: dict) -> CandidateSet:
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


def normalised_probs(cand: CandidateSet, probs: np.ndarray) -> np.ndarray:
    out = np.empty_like(probs, dtype=np.float64)
    n_obs = int(cand.obs_idx.max()) + 1 if len(cand) else 0
    counts = np.bincount(cand.obs_idx, minlength=n_obs)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for s, c in zip(starts, counts):
        if c:
            seg = probs[s:s + c].astype(np.float64)
            total = seg.sum()
            out[s:s + c] = seg / total if total > 0 else 1.0 / c
    return out


def subsample_trip(tp: dict, neg_per_obs: int, rng: np.random.Generator):
    """Row indices to keep: all labelled positives plus a fixed negative sample."""
    y = tp["y"]
    obs = tp["obs_idx"]
    keep_pos = np.where(y == 1)[0]
    neg_rows = np.where(y == 0)[0]
    if len(neg_rows) == 0:
        return keep_pos
    obs_of_neg = obs[neg_rows]
    # group negative rows by observation
    order = np.argsort(obs_of_neg, kind="stable")
    neg_rows = neg_rows[order]
    obs_of_neg = obs_of_neg[order]
    uniq, starts, counts = np.unique(obs_of_neg, return_index=True, return_counts=True)
    picks = []
    for s, c in zip(starts, counts):
        k = min(c, neg_per_obs)
        if k == c:
            picks.append(neg_rows[s:s + c])
        else:
            picks.append(neg_rows[s + rng.choice(c, size=k, replace=False)])
    sel = np.concatenate(picks) if picks else np.empty(0, dtype=np.int64)
    return np.concatenate([keep_pos, sel])


def cap_rows(X: np.ndarray, y: np.ndarray, max_rows: int,
             rng: np.random.Generator):
    if max_rows is None or len(X) <= max_rows:
        return np.arange(len(X))
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_neg = max(max_rows - len(pos), 0)
    if n_neg >= len(neg):
        return np.arange(len(X))
    sel = rng.choice(neg, size=n_neg, replace=False)
    return np.concatenate([pos, sel])


def eval_fold(val_trips: list[dict], probs_by_trip: dict) -> dict:
    """Candidate-level metrics and top-1 accuracy on the held-out trips."""
    ys, ps, top1 = [], [], []
    for tp in val_trips:
        probs = probs_by_trip[tp["trip_id"]]
        cand = cand_from_cache(tp)
        pn = normalised_probs(cand, probs)
        valid = tp["label_valid"].astype(bool)
        mask = valid[cand.obs_idx]
        y = (cand.arc == tp["label_arc"][cand.obs_idx])[mask].astype(np.int64)
        p = np.clip(pn[mask], 1e-7, 1 - 1e-7)
        ys.append(y); ps.append(p)
        # observation-level top-1
        pred = np.full(len(valid), -1, dtype=np.int64)
        counts = np.bincount(cand.obs_idx, minlength=len(valid))
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for i in range(len(valid)):
            s = starts[i]; c = counts[i]
            if c:
                pred[i] = cand.arc[s + int(np.argmax(probs[s:s + c]))]
        top1.append(top1_accuracy(pred, tp["label_arc"], valid))
    from sklearn.metrics import f1_score, log_loss
    y = np.concatenate(ys); p = np.concatenate(ps)
    yhat = (p >= 0.5).astype(np.int64)
    return {
        "macro_f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "f1": float(f1_score(y, yhat, average="binary", zero_division=0)),
        "precision": float(((y == 1) & (yhat == 1)).sum() / max((yhat == 1).sum(), 1)),
        "recall": float(((y == 1) & (yhat == 1)).sum() / max((y == 1).sum(), 1)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "top1_accuracy": float(np.mean(top1)),
        "n_rows": int(len(y)),
    }


def predict_probs(model, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    col = classes.index(1) if 1 in classes else raw.shape[1] - 1
    return raw[:, col]


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def _build_tuned_lgbm():
    import lightgbm as lgb
    params = load_tuned_params()
    if params is None:
        raise RuntimeError("no tuned LightGBM parameters found; run the Optuna search first")
    return lgb.LGBMClassifier(**params)



def load_tuned_params():
    """Best LightGBM parameters from the Bayesian search (Kaggle or local)."""
    import json
    candidates = [
        RESULTS / "local_optuna.json",
        RESULTS / "kaggle_optuna.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            params = data.get("best_params") or (data.get("best") or {}).get("params")
            if params:
                out = dict(params)
                out.setdefault("n_jobs", -1)
                out.setdefault("random_state", config.RANDOM_SEED)
                out.setdefault("verbosity", -1)
                return out
    return None


def model_zoo():
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                                  ExtraTreesClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    import lightgbm as lgb
    import xgboost as xgb

    zoo = {
        "logreg": {
            "family": "Linear / probabilistic",
            "cap": 200_000,
            "build": lambda: Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(solver="liblinear", C=1.0, max_iter=500,
                                           random_state=config.RANDOM_SEED)),
            ]),
            "config": "StandardScaler + L2 logistic regression, C=1.0, liblinear, 200k-row cap",
        },
        "gaussian_nb": {
            "family": "Probabilistic (generative)",
            "cap": 900_000,
            "build": lambda: GaussianNB(var_smoothing=1e-9),
            "config": "GaussianNB, var_smoothing=1e-9",
        },
        "decision_tree": {
            "family": "Tree-based",
            "cap": 900_000,
            "build": lambda: DecisionTreeClassifier(max_depth=16, min_samples_leaf=20,
                                                    class_weight="balanced",
                                                    random_state=config.RANDOM_SEED),
            "config": "DecisionTreeClassifier max_depth=16, min_samples_leaf=20, balanced",
        },
        "random_forest": {
            "family": "Ensemble (bagging)",
            "cap": 400_000,
            "build": lambda: RandomForestClassifier(n_estimators=120, max_depth=18,
                                                    min_samples_leaf=15,
                                                    max_features="sqrt", n_jobs=-1,
                                                    class_weight="balanced_subsample",
                                                    random_state=config.RANDOM_SEED),
            "config": "RandomForest 120 trees, max_depth=18, leaf=15, sqrt features, balanced",
        },
        "extra_trees": {
            "family": "Ensemble (bagging)",
            "cap": 400_000,
            "build": lambda: ExtraTreesClassifier(n_estimators=150, max_depth=18,
                                                  min_samples_leaf=15, max_features="sqrt",
                                                  n_jobs=-1, class_weight="balanced",
                                                  random_state=config.RANDOM_SEED),
            "config": "ExtraTrees 150 trees, max_depth=18, leaf=15, sqrt features, balanced",
        },
        "hist_gb": {
            "family": "Boosting",
            "cap": 600_000,
            "build": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                            max_leaf_nodes=31, min_samples_leaf=25,
                                                            l2_regularization=1.0,
                                                            random_state=config.RANDOM_SEED),
            "config": "HistGradientBoosting 200 iters, lr=0.1, 31 leaves, l2=1.0",
        },
        "lightgbm": {
            "family": "Boosting",
            "cap": 900_000,
            "build": lambda: lgb.LGBMClassifier(n_estimators=350, learning_rate=0.08,
                                               num_leaves=63, min_child_samples=20,
                                               subsample=0.9, colsample_bytree=0.9,
                                               reg_lambda=1.0, n_jobs=-1,
                                               random_state=config.RANDOM_SEED,
                                               verbosity=-1),
            "config": "LightGBM 350 trees, lr=0.08, 63 leaves, subsample 0.9",
        },
        "xgboost": {
            "family": "Boosting",
            "cap": 900_000,
            "build": lambda: xgb.XGBClassifier(n_estimators=350, learning_rate=0.08, max_depth=7,
                                               subsample=0.9, colsample_bytree=0.9,
                                               reg_lambda=1.0, tree_method="hist",
                                               n_jobs=-1, random_state=config.RANDOM_SEED,
                                               eval_metric="logloss"),
            "config": "XGBoost 350 trees, lr=0.08, depth=7, hist, subsample 0.9",
        },
        "lightgbm_tuned": {
            "family": "Boosting (Bayesian-tuned)",
            "cap": 900_000,
            "build": lambda: _build_tuned_lgbm(),
            "config": "LightGBM, Optuna TPE best params (see results JSON)",
        },
        "mlp": {
            "family": "Neural network",
            "cap": 250_000,
            "build": lambda: Pipeline([
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                      batch_size=1024, learning_rate_init=1e-3,
                                      max_iter=30, early_stopping=True,
                                      validation_fraction=0.1, n_iter_no_change=4,
                                      random_state=config.RANDOM_SEED)),
            ]),
            "config": "StandardScaler + MLP (64,32), relu, adam lr=1e-3, early stopping",
        },
        "knn": {
            "family": "Instance-based",
            "cap": 20_000,
            "build": lambda: Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance",
                                             algorithm="brute", n_jobs=-1)),
            ]),
            "config": "StandardScaler + distance-weighted KNN k=15, 20k-row cap "
                      "(inference on full candidate tables is expensive)",
        },
    }
    return zoo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--folds-quick", type=int, default=None,
                    help="optional smaller fold count for the slow models")
    ap.add_argument("--neg-per-obs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    load_feature_names()
    RESULTS.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(parents=True, exist_ok=True)
    zoo = model_zoo()
    names = list(zoo) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    for n in names:
        if n not in zoo:
            raise SystemExit(f"unknown model {n}; choose from {list(zoo)}")

    t_start = time.perf_counter()
    train_ids = list_trips("train")
    dev_ids = list_trips("student_test")
    print(f"train trips {len(train_ids)}, student_test trips {len(dev_ids)}", flush=True)

    train_trips = []
    for tid in train_ids:
        tp = load_cache("train", tid)
        rng = np.random.default_rng(args.seed + int(tid))
        sel = subsample_trip(tp, args.neg_per_obs, rng)
        tp["X_sub"] = tp["X"][sel]
        tp["y_sub"] = tp["y"][sel].astype(np.int64)
        train_trips.append(tp)
    dev_trips = [load_cache("student_test", tid) for tid in dev_ids]

    # pre-compute fold assignment
    rng = np.random.default_rng(args.seed)
    fold_of = {}
    order = rng.permutation(len(train_trips))
    for pos, idx in enumerate(order):
        fold_of[train_trips[idx]["trip_id"]] = pos % args.folds

    results = {}
    for name in names:
        spec = zoo[name]
        print(f"\n===== {name} ({spec['family']}) =====", flush=True)
        model_cfg = {"name": name, "family": spec["family"], "config": spec["config"],
                     "cap": spec["cap"], "neg_per_obs": args.neg_per_obs}
        t_model = time.perf_counter()
        fold_results = []
        confusions = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for fold in range(args.folds):
            rng = np.random.default_rng(args.seed + 1000 * fold)
            tr_idx = [i for i, tp in enumerate(train_trips)
                      if fold_of[tp["trip_id"]] != fold]
            va_idx = [i for i, tp in enumerate(train_trips)
                      if fold_of[tp["trip_id"]] == fold]
            X = np.concatenate([train_trips[i]["X_sub"] for i in tr_idx])
            y = np.concatenate([train_trips[i]["y_sub"] for i in tr_idx])
            cap_sel = cap_rows(X, y, spec["cap"], rng)
            Xtr, ytr = X[cap_sel], y[cap_sel]
            t_fit = time.perf_counter()
            model = spec["build"]()
            model.fit(Xtr, ytr)
            fit_s = time.perf_counter() - t_fit
            t_pred = time.perf_counter()
            probs_by_trip = {}
            for i in va_idx:
                tp = train_trips[i]
                probs_by_trip[tp["trip_id"]] = predict_probs(model, tp["X"])
            pred_s = time.perf_counter() - t_pred
            m = eval_fold([train_trips[i] for i in va_idx], probs_by_trip)
            m["fit_seconds"] = round(fit_s, 2)
            m["predict_seconds"] = round(pred_s, 2)
            fold_results.append(m)
            print(f"  fold {fold}: macro_f1={m['macro_f1']:.4f} top1={m['top1_accuracy']:.4f} "
                  f"logloss={m['log_loss']:.4f} fit={fit_s:.1f}s pred={pred_s:.1f}s", flush=True)
            del X, y, Xtr, ytr, model
        cv = {
            "macro_f1_mean": float(np.mean([f["macro_f1"] for f in fold_results])),
            "macro_f1_std": float(np.std([f["macro_f1"] for f in fold_results])),
            "top1_accuracy_mean": float(np.mean([f["top1_accuracy"] for f in fold_results])),
            "top1_accuracy_std": float(np.std([f["top1_accuracy"] for f in fold_results])),
            "log_loss_mean": float(np.mean([f["log_loss"] for f in fold_results])),
            "f1_mean": float(np.mean([f["f1"] for f in fold_results])),
            "precision_mean": float(np.mean([f["precision"] for f in fold_results])),
            "recall_mean": float(np.mean([f["recall"] for f in fold_results])),
            "fit_seconds_mean": float(np.mean([f["fit_seconds"] for f in fold_results])),
            "predict_seconds_mean": float(np.mean([f["predict_seconds"] for f in fold_results])),
            "folds": fold_results,
        }
        print(f"  CV macro_f1 {cv['macro_f1_mean']:.4f}+-{cv['macro_f1_std']:.4f} | "
              f"CV top1 {cv['top1_accuracy_mean']:.4f} | CV logloss {cv['log_loss_mean']:.4f}", flush=True)

        # ---- refit on all training trips, evaluate on student_test ----
        rng = np.random.default_rng(args.seed + 7)
        X = np.concatenate([tp["X_sub"] for tp in train_trips])
        y = np.concatenate([tp["y_sub"] for tp in train_trips])
        cap_sel = cap_rows(X, y, spec["cap"], rng)
        Xtr, ytr = X[cap_sel], y[cap_sel]
        model = spec["build"]()
        t_fit = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_all_s = time.perf_counter() - t_fit
        t_inf = time.perf_counter()
        dev_probs = {}
        for tp in dev_trips:
            dev_probs[tp["trip_id"]] = predict_probs(model, tp["X"])
        inf_s = time.perf_counter() - t_inf
        # classifier-only metrics
        from pathsnap.metrics import candidate_probability_metrics
        clf_metrics = {"macro_f1": [], "log_loss": [], "top1": [], "continuity": [],
                       "f1": [], "precision": [], "recall": [], "seq_f1": [],
                       "true_nll": []}
        hmm_metrics = {"top1": [], "continuity": [], "seq_f1": []}
        preds_store = {}
        for tp in dev_trips:
            cand = cand_from_cache(tp)
            probs = dev_probs[tp["trip_id"]]
            pn = normalised_probs(cand, probs)
            valid = tp["label_valid"].astype(bool)
            m = candidate_probability_metrics(cand, pn, tp["label_arc"], valid)
            clf_metrics["macro_f1"].append(m["macro_f1"])
            clf_metrics["log_loss"].append(m["log_loss"])
            clf_metrics["f1"].append(m["f1"])
            n = int(tp["n_obs"])
            pred = np.full(n, -1, dtype=np.int64)
            counts = np.bincount(cand.obs_idx, minlength=n)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            for i in range(n):
                s = starts[i]; c = counts[i]
                if c:
                    pred[i] = cand.arc[s + int(np.argmax(probs[s:s + c]))]
            clf_metrics["top1"].append(top1_accuracy(pred, tp["label_arc"], valid))
            clf_metrics["true_nll"].append(true_candidate_log_loss(cand, pn, tp["label_arc"], valid))
            # HMM decode needs the real network
            trip = load_trip(tp["trip_id"], "student_test")
            net = TripNetwork(trip)
            obs_xy = track_points_xy(net, trip)
            pred_h = viterbi_decode(net, cand, pn, obs_xy, trip.arcs,
                                    times=trip.track[:, 2], n_obs=n)
            clf_metrics["continuity"].append(route_continuity(net, pred))
            clf_metrics["seq_f1"].append(sequence_f1(pred, trip.route)["seq_f1"])
            hmm_metrics["top1"].append(top1_accuracy(pred_h, tp["label_arc"], valid))
            hmm_metrics["continuity"].append(route_continuity(net, pred_h))
            hmm_metrics["seq_f1"].append(sequence_f1(pred_h, trip.route)["seq_f1"])
            preds_store[tp["trip_id"]] = np.column_stack([pred, pred_h]).astype(np.int32)
            np.savez_compressed(PREDS / f"{name}_{tp['trip_id']}.npz", probs=probs.astype(np.float32))
        dev = {
            "top1_accuracy": float(np.mean(clf_metrics["top1"])),
            "macro_f1": float(np.mean(clf_metrics["macro_f1"])),
            "log_loss": float(np.mean(clf_metrics["log_loss"])),
            "f1": float(np.mean(clf_metrics["f1"])),
            "precision": float(np.mean(clf_metrics["precision"])),
            "recall": float(np.mean(clf_metrics["recall"])),
            "route_continuity": float(np.mean(clf_metrics["continuity"])),
            "seq_f1": float(np.mean(clf_metrics["seq_f1"])),
            "true_candidate_log_loss": float(np.mean(clf_metrics["true_nll"])),
            "hmm_top1_accuracy": float(np.mean(hmm_metrics["top1"])),
            "hmm_route_continuity": float(np.mean(hmm_metrics["continuity"])),
            "hmm_seq_f1": float(np.mean(hmm_metrics["seq_f1"])),
            "fit_seconds": round(fit_all_s, 2),
            "inference_seconds": round(inf_s, 3),
        }
        n_obs_dev = sum(int(tp["n_obs"]) for tp in dev_trips)
        dev["inference_ms_per_obs"] = round(1000.0 * inf_s / n_obs_dev, 4)
        total_s = time.perf_counter() - t_model
        rec = {"model": model_cfg, "cv": cv, "student_test": dev,
               "total_seconds": round(total_s, 1)}
        results[name] = rec
        # save after every model so a long run cannot lose finished work
        partial = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "protocol": {"train_trips": len(train_ids),
                                "student_test_trips": len(dev_ids),
                                "neg_per_obs": args.neg_per_obs, "folds": args.folds,
                                "seed": args.seed, "tag": args.tag},
                   "models": results}
        actual_out = RESULTS / f"model_comparison{('_' + args.tag) if args.tag else ''}.json"
        if actual_out.exists():
            try:
                prev = json.loads(actual_out.read_text())
                merged = dict(prev.get("models", {}))
                merged.update(results)
                partial["models"] = merged
            except Exception:
                pass
        actual_out.write_text(json.dumps(partial, indent=2))
        print(f"  STUDENT-TEST top1={dev['top1_accuracy']:.4f} macro_f1={dev['macro_f1']:.4f} "
              f"logloss={dev['log_loss']:.4f} | HMM top1={dev['hmm_top1_accuracy']:.4f} "
              f"cont={dev['hmm_route_continuity']:.4f} | train={fit_all_s:.1f}s "
              f"infer={dev['inference_ms_per_obs']:.3f}ms/obs", flush=True)
        # keep the globally best trained model per name? only final is kept separately
        del X, y, Xtr, ytr, model

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "train_trips": len(train_ids), "student_test_trips": len(dev_ids),
            "neg_per_obs": args.neg_per_obs, "folds": args.folds,
            "seed": args.seed, "tag": args.tag,
        },
        "models": results,
        "total_seconds": round(time.perf_counter() - t_start, 1),
    }
    out_path = RESULTS / f"model_comparison{('_' + args.tag) if args.tag else ''}.json"
    # merge with existing record if present
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        prev["models"].update(results)
        prev["generated_at"] = out["generated_at"]
        out = prev
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({out.get('total_seconds', '?')}s total)")


if __name__ == "__main__":
    main()
