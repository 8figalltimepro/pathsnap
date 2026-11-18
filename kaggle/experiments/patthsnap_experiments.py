"""PathSnap Kaggle experiments (CPU kernel).

1. Rebuild the candidate dataset on Kaggle from the raw dataset.
2. Train a deep MLP candidate classifier with PyTorch (5-fold group CV) and
   export its student-test probabilities + epoch-wise training history.
3. Run a Bayesian (TPE) hyperparameter search for the LightGBM candidate
   classifier on Kaggle CPUs and export the trials + best configuration.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

T0 = time.perf_counter()
DATA_ROOT = Path("/kaggle/input/datasets/figalltimepro8/patthsnap-data")
CODE_ROOT = Path("/kaggle/input/datasets/figalltimepro8/patthsnap-code")
WORK = Path("/kaggle/working")
ARTIFACTS = WORK / "artifacts"
RESULTS = ARTIFACTS / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

os.environ["PATHSNAP_DATA_DIR"] = str(DATA_ROOT)
os.environ["PATHSNAP_ARTIFACTS_DIR"] = str(ARTIFACTS)
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "scripts"))

print("data root:", DATA_ROOT, "exists:", DATA_ROOT.exists())
print("code root:", CODE_ROOT, "exists:", CODE_ROOT.exists())

from pathsnap import config
from pathsnap.candidates import CandidateSet
from pathsnap.io import list_trips

CACHE = ARTIFACTS / "cache"


def build_dataset():
    print("\n=== building candidate dataset ===", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CODE_ROOT / "src")
    env["PATHSNAP_DATA_DIR"] = str(DATA_ROOT)
    env["PATHSNAP_ARTIFACTS_DIR"] = str(ARTIFACTS)
    subprocess.run([sys.executable, str(CODE_ROOT / "scripts" / "build_dataset.py")],
                   check=True, env=env)


def load_cache(partition, trip_id):
    z = np.load(CACHE / f"{partition}_{trip_id}.npz")
    tp = {k: z[k] for k in z.files}
    tp["trip_id"] = trip_id
    tp["partition"] = partition
    return tp


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


def normalised_probs(cand, probs):
    out = np.empty(len(probs), dtype=np.float64)
    n_obs = int(cand.obs_idx.max()) + 1
    counts = np.bincount(cand.obs_idx, minlength=n_obs)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for s, c in zip(starts, counts):
        if c:
            seg = probs[s:s + c].astype(np.float64)
            total = seg.sum()
            out[s:s + c] = seg / total if total > 0 else 1.0 / c
    return out


def subsample_trip(tp, neg_per_obs, rng):
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


def fold_metrics(val_trips, probs_by_trip):
    from sklearn.metrics import f1_score, log_loss
    ys, ps, top1 = [], [], []
    for tp in val_trips:
        cand = cand_from_cache(tp)
        pn = normalised_probs(cand, probs_by_trip[tp["trip_id"]])
        valid = tp["label_valid"].astype(bool)
        mask = valid[cand.obs_idx]
        y = (cand.arc == tp["label_arc"][cand.obs_idx])[mask].astype(np.int64)
        ys.append(y); ps.append(np.clip(pn[mask], 1e-7, 1 - 1e-7))
        n = int(tp["n_obs"])
        pred = np.full(n, -1, dtype=np.int64)
        counts = np.bincount(cand.obs_idx, minlength=n)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for i in range(n):
            s = starts[i]; c = counts[i]
            if c:
                pred[i] = cand.arc[s + int(np.argmax(probs_by_trip[tp["trip_id"]][s:s + c]))]
        top1.append(float((pred[valid] == tp["label_arc"][valid]).mean()))
    y = np.concatenate(ys); p = np.concatenate(ps)
    yhat = (p >= 0.5).astype(np.int64)
    return {
        "macro_f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "f1": float(f1_score(y, yhat, average="binary", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "top1_accuracy": float(np.mean(top1)),
        "n_rows": int(len(y)),
    }


def train_mlp():
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== MLP on {device} ===", flush=True)

    train_ids = list_trips("train")
    dev_ids = list_trips("student_test")
    trips = [load_cache("train", t) for t in train_ids]
    dev_trips = [load_cache("student_test", t) for t in dev_ids]
    subs = []
    for tp in trips:
        rng = np.random.default_rng(42 + int(tp["trip_id"]))
        sel = subsample_trip(tp, 4, rng)
        subs.append((tp["X"][sel].astype(np.float32), tp["y"][sel].astype(np.float32)))

    rng = np.random.default_rng(42)
    order = rng.permutation(len(trips))
    fold_of = {}
    for pos, idx in enumerate(order):
        fold_of[trips[idx]["trip_id"]] = pos % 5
    n_feat = 26

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_feat, 128), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def fit_mlp(Xtr, ytr, Xva, yva, epochs=18, patience=3, batch=4096, lr=1e-3):
        model = MLP().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        lossf = nn.BCEWithLogitsLoss()
        Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
        Xva_t = torch.from_numpy(Xva).to(device); yva_t = torch.from_numpy(yva).to(device)
        n = len(Xtr_t)
        history = []
        best_val = float("inf"); best_state = None; bad = 0
        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(n)
            tot = 0.0
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                xb = Xtr_t[idx].to(device); yb = ytr_t[idx].to(device)
                opt.zero_grad()
                out = model(xb)
                loss = lossf(out, yb)
                loss.backward()
                opt.step()
                tot += float(loss.item()) * len(idx)
            train_loss = tot / n
            model.eval()
            with torch.no_grad():
                va_out = model(Xva_t)
                val_loss = float(lossf(va_out, yva_t).item())
            history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})
            print(f"  epoch {ep:02d} train {train_loss:.4f} val {val_loss:.4f}", flush=True)
            if val_loss < best_val - 1e-5:
                best_val = val_loss; bad = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history

    def predict(model, X):
        model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(X), 8192):
                xb = torch.from_numpy(X[s:s + 8192].astype(np.float32)).to(device)
                outs.append(torch.sigmoid(model(xb)).cpu().numpy())
        return np.concatenate(outs)

    fold_results = []
    for fold in range(5):
        tr_idx = [i for i, tp in enumerate(trips) if fold_of[tp["trip_id"]] != fold]
        va_idx = [i for i, tp in enumerate(trips) if fold_of[tp["trip_id"]] == fold]
        X = np.concatenate([subs[i][0] for i in tr_idx])
        y = np.concatenate([subs[i][1] for i in tr_idx])
        mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
        Xn = np.clip((X - mu) / sd, -8, 8).astype(np.float32)
        va_X, va_y = [], []
        for i in va_idx:
            tp = trips[i]
            va_X.append(np.clip((tp["X"] - mu) / sd, -8, 8).astype(np.float32))
            va_y.append(np.where(tp["y"] >= 0, tp["y"], 0).astype(np.float32))
        Xva = np.concatenate(va_X); yva = np.concatenate(va_y)
        model, hist = fit_mlp(Xn, y, Xva, yva)
        probs_by_trip = {}
        for i in va_idx:
            tp = trips[i]
            Xp = np.clip((tp["X"] - mu) / sd, -8, 8).astype(np.float32)
            probs_by_trip[tp["trip_id"]] = predict(model, Xp)
        m = fold_metrics([trips[i] for i in va_idx], probs_by_trip)
        m["fold"] = fold
        m["history"] = hist
        fold_results.append(m)
        print(f"  fold {fold}: macro_f1={m['macro_f1']:.4f} top1={m['top1_accuracy']:.4f} "
              f"logloss={m['log_loss']:.4f}", flush=True)

    X = np.concatenate([s[0] for s in subs]); y = np.concatenate([s[1] for s in subs])
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    Xn = np.clip((X - mu) / sd, -8, 8).astype(np.float32)
    va_idx = [i for i, tp in enumerate(trips) if fold_of[tp["trip_id"]] == 4]
    va_X = np.concatenate([np.clip((trips[i]["X"] - mu) / sd, -8, 8).astype(np.float32) for i in va_idx])
    va_y = np.concatenate([np.where(trips[i]["y"] >= 0, trips[i]["y"], 0).astype(np.float32) for i in va_idx])
    model, hist = fit_mlp(Xn, y, va_X, va_y, epochs=25)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd}, ARTIFACTS / "mlp_model.pt")
    dev_probs = {}
    for tp in dev_trips:
        Xp = np.clip((tp["X"] - mu) / sd, -8, 8).astype(np.float32)
        dev_probs[tp["trip_id"]] = predict(model, Xp)
    np.savez_compressed(RESULTS / "kaggle_mlp_preds.npz",
                        **{k: v.astype(np.float32) for k, v in dev_probs.items()})
    dev_metrics = {}
    for tp in dev_trips:
        probs = dev_probs[tp["trip_id"]]
        m = fold_metrics([tp], {tp["trip_id"]: probs})
        dev_metrics[tp["trip_id"]] = m
    agg = {k: float(np.mean([m[k] for m in dev_metrics.values()]))
           for k in ["macro_f1", "f1", "log_loss", "top1_accuracy"]}
    out = {
        "model": "torch_mlp",
        "family": "Neural network",
        "config": "MLP 26-128-64-1, ReLU, dropout 0.1, Adam lr=1e-3, batch 4096, early stopping",
        "device": str(device),
        "cv": {
            "macro_f1_mean": float(np.mean([m["macro_f1"] for m in fold_results])),
            "macro_f1_std": float(np.std([m["macro_f1"] for m in fold_results])),
            "top1_accuracy_mean": float(np.mean([m["top1_accuracy"] for m in fold_results])),
            "top1_accuracy_std": float(np.std([m["top1_accuracy"] for m in fold_results])),
            "log_loss_mean": float(np.mean([m["log_loss"] for m in fold_results])),
            "folds": [{k: v for k, v in m.items() if k != "history"} for m in fold_results],
        },
        "student_test": agg,
        "per_trip": dev_metrics,
        "final_history": hist,
    }
    (RESULTS / "kaggle_mlp.json").write_text(json.dumps(out, indent=2))
    import csv
    with open(RESULTS / "kaggle_mlp_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(hist)
    print("MLP done:", json.dumps({k: round(v, 4) for k, v in agg.items()}), flush=True)
    return out


def tune_lightgbm(n_trials=40):
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("\n=== Optuna LightGBM study ===", flush=True)
    train_ids = list_trips("train")
    trips = [load_cache("train", t) for t in train_ids]
    rng = np.random.default_rng(42)
    order = rng.permutation(len(trips))
    fold_of = {}
    for pos, idx in enumerate(order):
        fold_of[trips[idx]["trip_id"]] = pos % 3
    data = []
    for tp in trips:
        r = np.random.default_rng(42 + int(tp["trip_id"]))
        sel = subsample_trip(tp, 4, r)
        data.append((tp["X"][sel].astype(np.float32), tp["y"][sel].astype(np.int64)))
    val_tables = []
    for fold in range(3):
        val_tables.append([i for i, tp in enumerate(trips) if fold_of[tp["trip_id"]] == fold])
    trap = np.random.default_rng(7)

    def cap(X, y, n=350_000):
        if len(X) <= n:
            return X, y
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        keep_neg = trap.choice(neg, size=max(n - len(pos), 0), replace=False)
        sel = np.concatenate([pos, keep_neg])
        return X[sel], y[sel]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 650, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 96),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        scores = []
        for fold in range(3):
            tr_idx = [i for i, tp in enumerate(trips) if fold_of[tp["trip_id"]] != fold]
            X = np.concatenate([data[i][0] for i in tr_idx])
            y = np.concatenate([data[i][1] for i in tr_idx])
            X, y = cap(X, y)
            model = lgb.LGBMClassifier(**params, n_jobs=4, random_state=42, verbosity=-1)
            model.fit(X, y)
            probs_by_trip = {trips[i]["trip_id"]: model.predict_proba(trips[i]["X"])[:, 1]
                             for i in val_tables[fold]}
            m = fold_metrics([trips[i] for i in val_tables[fold]], probs_by_trip)
            scores.append(m["macro_f1"])
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=1500)
    trials = [{"number": t.number, "value": t.value, "params": t.params,
               "state": str(t.state)} for t in study.trials]
    best = study.best_params
    X = np.concatenate([d[0] for d in data]); y = np.concatenate([d[1] for d in data])
    X, y = cap(X, y, 600_000)
    model = lgb.LGBMClassifier(**best, n_jobs=4, random_state=42, verbosity=-1)
    model.fit(X, y)
    model.booster_.save_model(str(ARTIFACTS / "lightgbm_tuned.txt"))
    dev_ids = list_trips("student_test")
    dev_probs = {}
    for tid in dev_ids:
        tp = load_cache("student_test", tid)
        dev_probs[tid] = model.predict_proba(tp["X"])[:, 1]
    np.savez_compressed(RESULTS / "kaggle_lightgbm_preds.npz",
                        **{k: v.astype(np.float32) for k, v in dev_probs.items()})
    dev_results = {}
    for tid in dev_ids:
        tp = load_cache("student_test", tid)
        dev_results[tid] = fold_metrics([tp], {tid: dev_probs[tid]})
    agg = {k: float(np.mean([m[k] for m in dev_results.values()]))
           for k in ["macro_f1", "f1", "log_loss", "top1_accuracy"]}
    out = {
        "model": "lightgbm_tuned",
        "objective": "mean 3-fold group-CV candidate macro F1 (threshold 0.5)",
        "n_trials": n_trials,
        "best_params": best,
        "best_cv_macro_f1": float(study.best_value),
        "cv_macro_f1": float(study.best_value),
        "student_test": agg,
        "per_trip": dev_results,
        "trials": trials,
        "seconds": round(time.perf_counter() - T0, 1),
    }
    (RESULTS / "kaggle_optuna.json").write_text(json.dumps(out, indent=2))
    import csv
    with open(RESULTS / "kaggle_optuna_trials.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "state", "value"] + sorted(best.keys()))
        for t in trials:
            w.writerow([t["number"], t["state"], t["value"]] +
                       [t["params"].get(k) for k in sorted(best.keys())])
    print("Optuna best:", best, "cv macro F1:", study.best_value, flush=True)
    print("Tuned LightGBM student-test:", json.dumps({k: round(v, 4) for k, v in agg.items()}), flush=True)
    return out


if __name__ == "__main__":
    build_dataset()
    build_info = json.loads((RESULTS / "dataset_build.json").read_text())
    print("dataset build:", {k: v for k, v in build_info.items() if k != "trips"}, flush=True)
    mlp_out = train_mlp()
    tune_out = tune_lightgbm()
    final = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_seconds": round(time.perf_counter() - T0, 1),
        "mlp": {"cv": mlp_out["cv"], "student_test": mlp_out["student_test"]},
        "lightgbm_tuned": {"best_params": tune_out["best_params"],
                           "cv_macro_f1": tune_out["cv_macro_f1"],
                           "student_test": tune_out["student_test"]},
    }
    (RESULTS / "kaggle_summary.json").write_text(json.dumps(final, indent=2))
    print("\n=== KAGGLE SUMMARY ===")
    print(json.dumps(final, indent=2))
    print("total seconds:", final["total_seconds"])
