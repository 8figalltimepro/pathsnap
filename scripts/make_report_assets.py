"""Build report figures and LaTeX tables from recorded experiment results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
FIG = ROOT / "reports" / "figures" / "models"
TAB = ROOT / "reports" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}
plt.rcParams.update(STYLE)
COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b",
          "#17becf", "#bcbd22", "#e377c2", "#7f7f7f", "#aec7e8"]


def load_json(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def model_rows(comp):
    rows = []
    for name, rec in comp["models"].items():
        rows.append({
            "name": name,
            "family": rec["model"]["family"],
            "config": rec["model"]["config"],
            "cv_macro_f1": rec["cv"]["macro_f1_mean"],
            "cv_macro_f1_std": rec["cv"]["macro_f1_std"],
            "cv_top1": rec["cv"]["top1_accuracy_mean"],
            "cv_logloss": rec["cv"]["log_loss_mean"],
            "test_top1": rec["student_test"]["top1_accuracy"],
            "test_macro_f1": rec["student_test"]["macro_f1"],
            "test_logloss": rec["student_test"]["log_loss"],
            "test_precision": rec["student_test"]["precision"],
            "test_recall": rec["student_test"]["recall"],
            "test_continuity": rec["student_test"]["route_continuity"],
            "hmm_top1": rec["student_test"]["hmm_top1_accuracy"],
            "hmm_continuity": rec["student_test"]["hmm_route_continuity"],
            "hmm_seq_f1": rec["student_test"]["hmm_seq_f1"],
            "train_s": rec["student_test"]["fit_seconds"],
            "infer_ms": rec["student_test"]["inference_ms_per_obs"],
        })
    rows.sort(key=lambda r: r["cv_macro_f1"], reverse=True)
    return rows


def fig_cv(rows):
    names = [r["name"] for r in rows]
    mf = [r["cv_macro_f1"] for r in rows]
    mf_err = [r["cv_macro_f1_std"] for r in rows]
    top1 = [r["cv_top1"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(x - 0.2, mf, 0.4, yerr=mf_err, capsize=3, label="CV macro F1", color=COLORS[0])
    ax.bar(x + 0.2, top1, 0.4, label="CV top-1 accuracy", color=COLORS[1])
    for i, (a, b) in enumerate(zip(mf, top1)):
        ax.text(i - 0.2, a + 0.008, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, b + 0.008, f"{b:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("score")
    ax.set_title("5-fold group CV on the 80 training trips (candidate classifier)")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "cv_comparison.png")
    plt.close(fig)


def fig_test(rows):
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.bar(x - 0.2, [r["test_top1"] for r in rows], 0.4, label="top-1 accuracy", color=COLORS[0])
    ax.bar(x + 0.2, [r["test_macro_f1"] for r in rows], 0.4, label="macro F1", color=COLORS[2])
    for i, r in enumerate(rows):
        ax.text(i - 0.2, r["test_top1"] + 0.008, f"{r['test_top1']:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, r["test_macro_f1"] + 0.008, f"{r['test_macro_f1']:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.75, 1.0); ax.set_ylabel("score")
    ax.set_title("Student-test candidate quality (classifier only)")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.bar(x, [r["test_logloss"] for r in rows], color=COLORS[3])
    for i, r in enumerate(rows):
        ax.text(i, r["test_logloss"] + 0.004, f"{r['test_logloss']:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("log loss")
    ax.set_title("Student-test candidate log loss")
    fig.tight_layout()
    fig.savefig(FIG / "test_comparison.png")
    plt.close(fig)


def fig_hmm(rows):
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.bar(x - 0.2, [r["test_top1"] for r in rows], 0.4, label="classifier only", color=COLORS[0])
    ax.bar(x + 0.2, [r["hmm_top1"] for r in rows], 0.4, label="classifier + HMM", color=COLORS[1])
    for i, r in enumerate(rows):
        ax.text(i - 0.2, r["test_top1"] + 0.006, f"{r['test_top1']:.3f}", ha="center", fontsize=6.5)
        ax.text(i + 0.2, r["hmm_top1"] + 0.006, f"{r['hmm_top1']:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("top-1 accuracy")
    ax.set_title("Observation-level top-1 accuracy: decoding gain")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.bar(x - 0.2, [r["test_continuity"] for r in rows], 0.4, label="classifier only", color=COLORS[0])
    ax.bar(x + 0.2, [r["hmm_continuity"] for r in rows], 0.4, label="classifier + HMM", color=COLORS[1])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.85, 1.005); ax.set_ylabel("route continuity")
    ax.set_title("Route continuity on student-test trips")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "hmm_comparison.png")
    plt.close(fig)


def fig_time(rows):
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(x - 0.2, [r["train_s"] for r in rows], 0.4, label="fit time (s)", color=COLORS[4])
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [r["infer_ms"] for r in rows], 0.4, label="inference (ms/obs)", color=COLORS[5])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("fit time (s)")
    ax2.set_ylabel("inference (ms/obs)")
    ax.set_title("Computational cost (full training tripless fit vs student-test inference)")
    ax.grid(False); ax2.grid(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "computational_cost.png")
    plt.close(fig)


def fig_per_trip(final):
    per = final.get("per_trip", {})
    if not per:
        return
    trips = sorted(per)
    clf = [per[t]["top1_accuracy"] for t in trips]
    hmm = [per[t]["top1_accuracy_hmm"] for t in trips]
    x = np.arange(len(trips))
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.bar(x - 0.2, clf, 0.4, label="classifier only", color=COLORS[0])
    ax.bar(x + 0.2, hmm, 0.4, label="classifier + HMM", color=COLORS[1])
    ax.set_xticks(x); ax.set_xticklabels(trips, rotation=0, fontsize=8)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("top-1 accuracy")
    ax.set_xlabel("student-test trip id")
    ax.set_title(f"Final model per-trip accuracy ({final.get('model_name', 'final')})")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "final_per_trip.png")
    plt.close(fig)


def fig_hmm_heatmap(tuning):
    grid = tuning.get("grid", [])
    if not grid:
        return
    states = tuning["best"].get("max_states")
    rows = sorted({g["beta"] for g in grid if g["max_states"] == states})
    cols = sorted({g["alpha"] for g in grid if g["max_states"] == states})
    M = np.full((len(rows), len(cols)), np.nan)
    for g in grid:
        if g["max_states"] != states:
            continue
        i = rows.index(g["beta"]); j = cols.index(g["alpha"])
        M[i, j] = g["top1_accuracy"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{c:g}" for c in cols])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r:g}" for r in rows])
    ax.set_xlabel("emission alpha"); ax.set_ylabel("transition beta (m)")
    ax.set_title(f"HMM top-1 accuracy (max_states={states})")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        color="white", fontsize=7.5)
    fig.colorbar(im, ax=ax, label="top-1 accuracy")
    ax.grid(False)
    fig.savefig(FIG / "hmm_heatmap.png")
    plt.close(fig)


def fig_mlp_curves():
    p = RESULTS / "kaggle_mlp_history.csv"
    if not p.exists():
        return
    import csv
    rows = list(csv.DictReader(p.open()))
    ep = [int(r["epoch"]) for r in rows]
    tr = [float(r["train_loss"]) for r in rows]
    va = [float(r["val_loss"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ep, tr, "-o", ms=3, label="train loss")
    ax.plot(ep, va, "-o", ms=3, label="validation loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("binary cross-entropy")
    ax.set_title("MLP (PyTorch, trained on Kaggle) training curve")
    ax.legend()
    fig.savefig(FIG / "mlp_training_curve.png")
    plt.close(fig)


def fig_optuna():
    p = RESULTS / "kaggle_optuna_trials.csv"
    if not p.exists():
        return
    import csv
    rows = list(csv.DictReader(p.open()))
    vals = []
    for r in rows:
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            v = np.nan
        vals.append(v)
    vals = np.array(vals)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.scatter(np.arange(len(vals)), vals, s=14, color=COLORS[0], label="trial macro F1")
    finite = np.isfinite(vals)
    best = np.maximum.accumulate(np.where(finite, vals, -np.inf))
    best[~finite] = np.nan
    ax.plot(np.arange(len(vals)), best, color=COLORS[3], label="best so far")
    ax.axhline(float(np.nanmax(vals)), color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Optuna trial"); ax.set_ylabel("3-fold CV macro F1")
    ax.set_title("Bayesian (TPE) hyperparameter search for LightGBM")
    ax.legend()
    fig.savefig(FIG / "optuna_history.png")
    plt.close(fig)


def write_model_table(rows):
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Model comparison. CV scores are means over 5 group folds on the 80 training trips; student-test scores are means over the 10 development trips.}")
    lines.append(r"\label{tab:model_comparison}")
    lines.append(r"\begin{tabular}{lccccccccc}")
    lines.append(r"\hline")
    lines.append(r"Model & Family & CV macro F1 & Top-1 & Macro F1 & Log loss & Precision & Recall & Continuity & HMM Top-1 \\")
    lines.append(r"\hline")
    for r in rows:
        lines.append(
            f"{r['name']} & {r['family']} & {r['cv_macro_f1']:.3f} $\\pm$ {r['cv_macro_f1_std']:.3f} & "
            f"{r['test_top1']:.3f} & {r['test_macro_f1']:.3f} & {r['test_logloss']:.3f} & "
            f"{r['test_precision']:.3f} & {r['test_recall']:.3f} & {r['test_continuity']:.3f} & "
            f"{r['hmm_top1']:.3f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (TAB / "model_comparison.tex").write_text("\n".join(lines) + "\n")
    md = ["| Model | Family | CV macro F1 | CV top-1 | Test top-1 | Test macro F1 | Log loss | HMM top-1 | Continuity (clf / HMM) | Train s | Infer ms/obs |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['family']} | {r['cv_macro_f1']:.3f} ± {r['cv_macro_f1_std']:.3f} | "
                  f"{r['cv_top1']:.3f} | {r['test_top1']:.3f} | {r['test_macro_f1']:.3f} | "
                  f"{r['test_logloss']:.3f} | {r['hmm_top1']:.3f} | "
                  f"{r['test_continuity']:.3f} / {r['hmm_continuity']:.3f} | "
                  f"{r['train_s']:.1f} | {r['infer_ms']:.3f} |")
    (TAB / "model_comparison.md").write_text("\n".join(md) + "\n")


def write_config_table(rows):
    lines = [r"\begin{table}[h]", r"\centering", r"\footnotesize",
             r"\caption{Hyperparameter configurations of the compared models.}",
             r"\label{tab:model_configs}", r"\begin{tabular}{llp{8.2cm}}", r"\hline",
             r"Model & Family & Configuration \\", r"\hline"]
    for r in rows:
        conf = r["config"].replace("_", r"\_")
        lines.append(f"{r['name']} & {r['family']} & {conf} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (TAB / "model_configs.tex").write_text("\n".join(lines) + "\n")


def main():
    comp = load_json("model_comparison.json")
    if comp is None:
        raise SystemExit("run scripts/run_experiments.py first")
    rows = model_rows(comp)
    fig_cv(rows)
    fig_test(rows)
    fig_hmm(rows)
    fig_time(rows)
    write_model_table(rows)
    write_config_table(rows)
    final = load_json("final_eval.json")
    if final:
        fig_per_trip(final)
    tuning = load_json("hmm_tuning_best.json") or load_json("hmm_tuning_quick.json")
    if tuning:
        fig_hmm_heatmap(tuning)
    fig_mlp_curves()
    fig_optuna()
    print("figures written to", FIG)
    print("tables written to", TAB)
    for r in rows:
        print(f"{r['name']:15s} cv_macro {r['cv_macro_f1']:.4f} test_top1 {r['test_top1']:.4f} "
              f"hmm_top1 {r['hmm_top1']:.4f}")


if __name__ == "__main__":
    main()
