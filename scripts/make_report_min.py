"""Generate the minimal report assets: one model-comparison figure and two tables.

The report is intentionally short (about 6-7 pages), so this replaces the large
set of figures from the full report with a single clean chart plus compact tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
FIG = ROOT / "reports" / "figures" / "min"
GEN = ROOT / "reports" / "generated"
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

DISPLAY = {
    "xgboost": "XGBoost", "lightgbm_tuned": "LightGBM (tuned)",
    "lightgbm": "LightGBM (default)", "hist_gb": "HistGradientBoosting",
    "mlp_deep": "Deep MLP (Kaggle)", "decision_tree": "Decision Tree",
    "random_forest": "Random Forest", "mlp": "MLP (scikit-learn)",
    "extra_trees": "Extra Trees", "logreg": "Logistic Regression",
    "gaussian_nb": "Gaussian Naive Bayes", "knn": "K-Nearest Neighbours",
}
SHORT_CONFIG = {
    "xgboost": "350 trees, lr .08, depth 7",
    "lightgbm_tuned": "TPE: 500 trees, 96 leaves",
    "lightgbm": "350 trees, 63 leaves",
    "hist_gb": "200 iters, 31 leaves",
    "mlp_deep": "26-128-64-1, Adam",
    "decision_tree": "depth 16, leaf 20",
    "random_forest": "120 trees, depth 18",
    "mlp": "(64,32), Adam, early stop",
    "extra_trees": "150 trees, depth 18",
    "logreg": "C=1.0, liblinear",
    "gaussian_nb": "var_smoothing 1e-9",
    "knn": "k=15, 20k cap",
}

FAMILY = {
    "xgboost": "Boosting", "lightgbm_tuned": "Boosting", "lightgbm": "Boosting",
    "hist_gb": "Boosting", "mlp_deep": "Neural", "decision_tree": "Tree",
    "random_forest": "Bagging", "mlp": "Neural", "extra_trees": "Bagging",
    "logreg": "Linear", "gaussian_nb": "Probabilistic", "knn": "Instance",
}


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_scores():
    comp = load("model_comparison.json")
    rows = []
    for key, rec in comp["models"].items():
        st = rec["student_test"]
        rows.append({"key": key, "clf": st["top1_accuracy"],
                     "hmm": st["hmm_top1_accuracy"]})
    rows.sort(key=lambda r: r["hmm"])
    labels = [DISPLAY.get(r["key"], r["key"]) for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, r in enumerate(rows):
        ax.plot([r["clf"], r["hmm"]], [i, i], color="#b0b0b0", lw=2.4, zorder=1)
    ax.scatter([r["clf"] for r in rows], y, s=26, color="#737373", zorder=2,
               label="classifier only")
    ax.scatter([r["hmm"] for r in rows], y, s=32, color="#111111", zorder=3,
               label="classifier + HMM")
    for i, r in enumerate(rows):
        ax.text(r["hmm"] + 0.008, i, f"{r['hmm']:.3f}", va="center", ha="left",
                fontsize=8.0, color="#111111")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == "LightGBM (tuned)":
            lbl.set_fontweight("bold")
    ax.set_xlim(0.5, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("observation-level top-1 accuracy (10 student-test trips)")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2,
              frameon=False, handletextpad=0.3, columnspacing=1.6)
    fig.tight_layout()
    fig.savefig(FIG / "model_scores.png")
    plt.close(fig)
    print("wrote", FIG / "model_scores.png")
    return rows


def tex_escape(s):
    return (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#"))


def fmt_ms(v):
    return "N/A" if v != v else f"{v:.3f}"


def table_models():
    comp = load("model_comparison.json")
    rows = []
    for key, rec in comp["models"].items():
        cv = rec["cv"]
        st = rec["student_test"]
        rows.append({
            "name": DISPLAY.get(key, key), "family": FAMILY.get(key, ""),
            "config": SHORT_CONFIG.get(key, rec["model"]["config"]),
            "cv": cv["macro_f1_mean"], "cv_sd": cv["macro_f1_std"],
            "top1": st["top1_accuracy"], "hmm": st["hmm_top1_accuracy"],
            "ll": st["log_loss"], "ms": st["inference_ms_per_obs"],
        })
    rows.sort(key=lambda r: r["cv"], reverse=True)
    lines = [r"\begin{table}[H]", r"\centering", r"\scriptsize",
             r"\caption{Twelve models compared on 80 training trips (5-fold trip-grouped "
             r"CV) and 10 student-testing trips. HMM top-1 uses the shared default decoder "
             r"(beta=20 m, alpha=1.0); the tuned decoder (alpha=0.1, Table~\ref{tab:hmm}) "
             r"gives 93.5\% for the selected model.}", r"\label{tab:models}",
             r"\begin{tabular}{llp{4.4cm}cccccc}", r"\hline",
             r"Model & Family & Configuration & CV mF1 & Top-1 & HMM & Log loss & ms/obs \\",
             r"\hline"]
    for r in rows:
        conf = tex_escape(r["config"])
        lines.append(f"{tex_escape(r['name'])} & {r['family']} & {conf} & "
                     f"{r['cv']:.3f} $\\pm$ {r['cv_sd']:.3f} & {r['top1']:.3f} & "
                     f"{r['hmm']:.3f} & {r['ll']:.3f} & "
                     f"{fmt_ms(r['ms'])} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (GEN / "min_model_table.tex").write_text("\n".join(lines) + "\n")
    print("wrote", GEN / "min_model_table.tex")


def table_hmm():
    best = load("hmm_tuning_best.json")["best"]
    tuned = load("local_optuna.json")["best_params"]
    kg = load("kaggle_optuna.json")
    rows = [
        (r"transition scale $\beta$", f"{best['beta']:.0f} m"),
        (r"emission sharpening $\alpha$", f"{best['alpha']:.2f}"),
        (r"candidate states per fix", f"{best['max_states']}"),
        (r"gap-free threshold", f"{best['gap_free_s']:.0f} s"),
        (r"local TPE study", "14 trials, best CV mF1 0.919"),
        (r"Kaggle TPE study", f"{kg['n_trials']} trials, best CV mF1 {kg['best_cv_macro_f1']:.3f}"),
        (r"final tuned LightGBM", f"{tuned['n_estimators']} trees, lr {tuned['learning_rate']:.3f}, "
                                  f"{tuned['num_leaves']} leaves"),
    ]
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{Decoder and tuning configuration of the exported system.}",
             r"\label{tab:hmm}", r"\begin{tabular}{ll}", r"\hline",
             r"Parameter & Value \\", r"\hline"]
    for a, b in rows:
        lines.append(f"{a} & {b} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (GEN / "min_hmm_table.tex").write_text("\n".join(lines) + "\n")
    print("wrote", GEN / "min_hmm_table.tex")


def table_deploy():
    lines = [r"\begin{tabular}{llp{7.6cm}}", r"\hline",
             r"Endpoint & Method & Purpose \\", r"\hline",
             r"\code{/health} & GET & status, trained/baseline mode, partition, model count \\",
             r"\code{/trips} & GET & evaluation trip ids (evaluator partition when present) \\",
             r"\code{/models} & GET & deployable classifiers with recorded scores \\",
             r"\code{/evaluate/\{trip\}?model=<id>} & GET & predictions, metrics and timings for one trip and model \\",
             r"\code{/docs} & GET & OpenAPI documentation \\",
             r"\hline", r"\end{tabular}"]
    (GEN / "min_api_table.tex").write_text("\n".join(lines) + "\n")
    print("wrote", GEN / "min_api_table.tex")


def final_numbers():
    fe = load("final_eval.json")
    agg = fe["aggregate"]
    b = fe["baseline"]["aggregate"]
    print("final:", {k: round(agg[k], 4) for k in
                     ["top1_accuracy", "top1_accuracy_hmm", "macro_f1", "log_loss",
                      "route_continuity", "route_continuity_hmm", "seq_f1_clf",
                      "seq_f1_hmm", "candidate_miss_rate"]})
    print("baseline:", {k: round(b[k], 4) for k in
                        ["top1_accuracy", "top1_accuracy_hmm", "route_continuity",
                         "seq_f1_hmm"]})
    print("mean total ms:", round(agg["mean_total_ms"], 1))


if __name__ == "__main__":
    fig_scores()
    table_models()
    table_hmm()
    table_deploy()
    final_numbers()
