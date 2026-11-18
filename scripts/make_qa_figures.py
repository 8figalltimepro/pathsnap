"""Build the single compact comparison figure for the Q&A report from the
actual model records: CV macro F1, student-test top-1 / macro F1, log loss.
Dot plots are used instead of bars so no axis is truncated.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "results"
OUT = ROOT / "reports" / "figures" / "qa"
OUT.mkdir(parents=True, exist_ok=True)

MC = json.loads((ART / "model_comparison.json").read_text())["models"]
LABEL = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "lightgbm_tuned": "LightGBM tuned",
    "hist_gb": "HistGB", "mlp_deep": "Deep MLP", "decision_tree": "Decision tree",
    "random_forest": "Random forest", "mlp": "MLP", "extra_trees": "Extra trees",
    "logreg": "LogReg", "gaussian_nb": "Gaussian NB", "knn": "KNN",
}
ORDER = list(LABEL)
INK, GREY, MID = "#111111", "#b8b8b8", "#6b6b6b"
PICK = "lightgbm_tuned"

plt.rcParams.update({
    "font.size": 6.0, "axes.edgecolor": "#d4d4d4", "axes.linewidth": 0.5,
    "xtick.color": MID, "ytick.color": MID, "axes.labelcolor": INK,
    "xtick.labelsize": 5.8, "ytick.labelsize": 5.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#ececec", linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    cv = [MC[k]["cv"]["macro_f1_mean"] for k in ORDER]
    top1 = [MC[k]["student_test"]["top1_accuracy"] for k in ORDER]
    mf1 = [MC[k]["student_test"]["macro_f1"] for k in ORDER]
    ll = [MC[k]["student_test"]["log_loss"] for k in ORDER]
    names = [LABEL[k] for k in ORDER]
    y = list(range(len(ORDER)))[::-1]
    ip = ORDER.index(PICK)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.35),
                             gridspec_kw={"wspace": 0.42, "left": 0.115, "right": 0.985,
                                          "top": 0.86, "bottom": 0.14})

    ax = axes[0]
    ax.scatter(cv, y, s=13, color=GREY, edgecolor="none", zorder=3)
    ax.scatter([cv[ip]], [y[ip]], s=17, color=INK, edgecolor="none", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0.56, 0.94)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9])
    clean(ax)
    ax.set_title("CV macro F1 (80 training trips)", fontsize=6.3, color=INK, loc="left", pad=4)

    ax = axes[1]
    ax.scatter(top1, [v + 0.12 for v in y], s=13, color=GREY, edgecolor="none", zorder=3, label="Top-1")
    ax.scatter(mf1, [v - 0.12 for v in y], s=13, color=INK, edgecolor="none", zorder=3, label="Macro F1")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0.5, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    clean(ax)
    ax.legend(frameon=False, fontsize=5.6, loc="lower right", handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("Student-test top-1 / macro F1", fontsize=6.3, color=INK, loc="left", pad=4)

    ax = axes[2]
    ax.scatter(ll, y, s=13, color=GREY, edgecolor="none", zorder=3)
    ax.scatter([ll[ip]], [y[ip]], s=17, color=INK, edgecolor="none", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0.0, 0.36)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    clean(ax)
    ax.set_title("Student-test log loss (lower is better)", fontsize=6.3, color=INK, loc="left", pad=4)

    fig.savefig(OUT / "model_compare.png", dpi=300, facecolor="white")
    print("wrote", OUT / "model_compare.png")


if __name__ == "__main__":
    main()
