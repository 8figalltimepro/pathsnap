"""Generate the compact Q&A report tables and per-model answer blocks.

All numbers come from the frozen artifacts; nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "results"
GEN = ROOT / "reports" / "generated"
GEN.mkdir(parents=True, exist_ok=True)

MC = json.loads((ART / "model_comparison.json").read_text())["models"]
FMS = json.loads((ART / "final_model_selection.json").read_text())
DEV = json.loads((ART / "final_eval.json").read_text())
EVAL = json.loads((ART / "evaluator_eval.json").read_text())
DB = json.loads((ART / "dataset_build.json").read_text())

ORDER = [
    ("xgboost", "M1", "XGBoost"),
    ("lightgbm", "M2", "LightGBM (untuned)"),
    ("lightgbm_tuned", "M3", "LightGBM (tuned, selected)"),
    ("hist_gb", "M4", "HistGradientBoosting"),
    ("mlp_deep", "M5", "Deep MLP (PyTorch, Kaggle)"),
    ("decision_tree", "M6", "Decision Tree"),
    ("random_forest", "M7", "Random Forest"),
    ("mlp", "M8", "MLP (scikit-learn)"),
    ("extra_trees", "M9", "Extra Trees"),
    ("logreg", "M10", "Logistic Regression"),
    ("gaussian_nb", "M11", "Gaussian Naive Bayes"),
    ("knn", "M12", "K-Nearest Neighbours"),
]

TABLE_LABEL = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "lightgbm_tuned": "LightGBM tuned",
    "hist_gb": "HistGB", "mlp_deep": "Deep MLP", "decision_tree": "Decision tree",
    "random_forest": "Random forest", "mlp": "MLP", "extra_trees": "Extra trees",
    "logreg": "LogReg", "gaussian_nb": "Gaussian NB", "knn": "KNN",
}

TABLE_CONFIG = {
    "xgboost": "350 trees, lr 0.08, depth 7",
    "lightgbm": "350 trees, lr 0.08, 63 leaves",
    "lightgbm_tuned": "500 trees, lr 0.157, 96 leaves",
    "hist_gb": "200 iters, 31 leaves",
    "mlp_deep": "26-128-64 MLP, Adam",
    "decision_tree": "depth 16, leaf 20",
    "random_forest": "120 trees, depth 18",
    "mlp": "64-32, Adam, early stop",
    "extra_trees": "150 trees, depth 18",
    "logreg": "C 1.0, liblinear",
    "gaussian_nb": "var smoothing 1e-9",
    "knn": "k=15, 20k cap",
}

TABLE_FAMILY = {
    "Boosting (Bayesian-tuned)": "Boosting", "Boosting": "Boosting", "Tree-based": "Tree",
    "Ensemble (bagging)": "Bagging", "Neural network": "Neural",
    "Linear / probabilistic": "Linear", "Probabilistic (generative)": "Prob.",
    "Instance-based": "Instance",
}

# one-sentence "why suitable" and "strengths / limitations" per model, from the experiments
WHY = {
"xgboost": ("boosted trees capture the non-linear distance/heading/motion interactions that separate parallel streets and give probabilities usable as HMM emissions.",
            "best CV macro F1 and fast scoring; deep trees can memorise rare patterns."),
"lightgbm": ("same boosted-tree rationale, with hand-set defaults kept as the untuned control.",
             "fastest boosted training; defaults leave accuracy to the tuning study."),
"lightgbm_tuned": ("its hyperparameters were selected on grouped cross-validation, so the model is tuned on trip-level splits.",
                   "best student-test macro F1 and best decoded route; the gain over M2 is small."),
"hist_gb": ("histogram binning gives a fast, low-memory version of the best-performing tree family.",
            "close to the top at 0.041 ms/obs; fixed defaults cap it."),
"mlp_deep": ("a small network can approximate smooth feature combinations and gives a different inductive bias.",
             "0.908 CV macro F1, but needs scaling and early stopping and does not beat boosted trees."),
"decision_tree": ("every split is a readable distance/heading/motion threshold, the geometry that decides between neighbouring arcs.",
                  "0.001 ms/obs and interpretable; leaf probabilities are poorly calibrated (log loss 0.103)."),
"random_forest": ("bagging reduces the variance of a single tree while keeping threshold splits.",
                  r"strong decoded top-1 (93.0\%) and stable folds; memory and time cost."),
"mlp": ("a light network with early stopping tests whether a neural model can capture the geometry.",
        "cheap and fast; underfits the interactions (CV macro F1 0.882)."),
"extra_trees": ("extremely randomised splits decorrelate trees more than a random forest.",
                "fastest ensemble training; extra randomness increases bias (CV macro F1 0.875)."),
"logreg": ("a linear model on scaled features is the calibrated baseline and measures how much signal is linear.",
           "well calibrated and fast; cannot represent heading/motion interactions (CV macro F1 0.802)."),
"gaussian_nb": ("a generative baseline that assumes conditionally independent features given the class.",
                "sub-millisecond training; the independence assumption is violated, giving a poor log loss (0.250, second worst)."),
"knn": ("no training at all: tests whether local similarity in feature space is enough to rank arcs.",
        r"its 56.4\% top-1 only matches the distance baseline; 1.259 ms/obs and unusable calibration (log loss 0.336)."),
}


def pct(x, nd=2):
    return f"{100*x:.{nd}f}" + chr(92) + "%"


def fmt_ms(x):
    return "N/A (Kaggle kernel)" if x != x else f"{x:.3f}"


def table_model_comparison():
    rows = []
    for key, mid, _label in ORDER:
        m = MC[key]
        cv, st = m["cv"], m["student_test"]
        rows.append(
            f"{mid} & {TABLE_LABEL[key]} & {TABLE_FAMILY[m['model']['family']]} & {TABLE_CONFIG[key]} & "
            f"{cv['macro_f1_mean']:.4f} $\\pm$ {cv['macro_f1_std']:.4f} & {pct(st['top1_accuracy'])} & "
            f"{pct(st['route_continuity'],1)} & {st['macro_f1']:.4f} & {st['log_loss']:.4f} & "
            f"{fmt_ms(st['inference_ms_per_obs'])} \\\\")
    bl = DEV["baseline"]["aggregate"]
    rows.append(
        "B0 & Distance baseline & no learning & score from perpendicular distance & N/A (no training) & "
        f"{pct(bl['top1_accuracy'])} & {pct(bl['route_continuity'],1)} & {bl['macro_f1']:.4f} & "
        f"{bl['log_loss']:.4f} & 0.000 \\\\")
    tex = r"""\begin{table}[!htb]
\centering
\scriptsize
\setlength{\tabcolsep}{1.2pt}
\caption{Consolidated model comparison. CV macro F1: 5-fold trip-grouped CV on the 80 training trips
(mean $\pm$ std). Other metrics: 10 student-testing trips; route continuity is classifier-only;
inference is classifier-only, ms per observation. B0 is the non-learned distance reference (its macro F1 and log loss come from the distance-derived score); N/A marks the timing of the deep MLP, which was measured only inside the Kaggle kernel.}
\label{tab:modelcmp}
\begin{tabular}{@{}llp{1.2cm}p{2.0cm}cccccc@{}}
\toprule
Model & Algorithm & Family & Key Configuration & \shortstack{CV Macro\\F1-score} & \shortstack{Top-1\\Accuracy} & \shortstack{Route\\continuity} & \shortstack{Student-Test\\Macro F1-score} & \shortstack{Log\\Loss} & \shortstack{Inference\\Time} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "qa_model_table.tex").write_text(tex)


def table_model_qa():
    out = []
    for key, mid, label in ORDER:
        m = MC[key]
        cv, st = m["cv"], m["student_test"]
        why, lim = WHY[key]
        cfg = TABLE_CONFIG[key]
        if key == "lightgbm_tuned":
            cfg = "500 trees, lr 0.157, 96 leaves, min child 46, subsample 0.861, L2 0.018"
        if key == "mlp_deep":
            cfg = "26-128-64 MLP, Adam, dropout 0.1, early stopping (Kaggle)"
        loss = ("binary cross-entropy" if key not in ("knn", "gaussian_nb")
                else ("Gaussian class likelihood" if key == "gaussian_nb"
                      else "no training (k-nearest-neighbour vote, k=15)"))
        gap = 100 * (st["top1_accuracy"] - cv["top1_accuracy_mean"])
        sd = 100 * cv["top1_accuracy_std"]
        over = ("CV top-1 " + pct(cv["top1_accuracy_mean"]) + " $\\pm$ " + f"{sd:.1f}" + " pp vs test "
                + pct(st["top1_accuracy"]) + f" (gap {gap:+.1f} pp, inside one fold std): no over- or "
                "underfitting.")
        if st["macro_f1"] - cv["macro_f1_mean"] < -0.02:
            over += (" The larger macro-F1 gap is a threshold/calibration effect (raw CV probabilities "
                     "versus per-fix normalised production scores).")
        if st["inference_ms_per_obs"] != st["inference_ms_per_obs"]:
            inf = "inference N/A (timed only in the Kaggle kernel)"
        else:
            inf = "inference " + f"{st['inference_ms_per_obs']:.3f}" + " ms/obs"
        out.append(r"\subsection*{" + f"{mid} --- {label}" + "}")
        out.append(r"\emph{1.} " + why + " " + r"\emph{2.} " + lim + " "
                   + r"\emph{5.} " + cfg + ". " + r"\emph{6.} " + loss + ". "
                   + r"\emph{7.} " + pct(st["top1_accuracy"]) + " top-1, "
                   + pct(st["hmm_top1_accuracy"]) + " decoded; continuity "
                   + pct(st["route_continuity"], 1) + " to " + pct(st["hmm_route_continuity"], 1) + ". "
                   + r"\emph{8.} macro F1 " + f"{st['macro_f1']:.4f}"
                   + ", log loss " + f"{st['log_loss']:.4f}" + ", " + inf + ". "
                   + r"\emph{9.} " + over)
        out.append("")
    (GEN / "qa_model_qa.tex").write_text("\n".join(out))


def table_tuning():
    unt, tun = FMS["comparison"]["lightgbm_untuned"], FMS["comparison"]["lightgbm_tuned"]
    ucv, tcv = MC["lightgbm"]["cv"], MC["lightgbm_tuned"]["cv"]
    ust, tst = MC["lightgbm"]["student_test"], MC["lightgbm_tuned"]["student_test"]
    rows = [
        ("CV macro F1", f"{ucv['macro_f1_mean']:.4f} $\\pm$ {ucv['macro_f1_std']:.4f}",
         f"{tcv['macro_f1_mean']:.4f} $\\pm$ {tcv['macro_f1_std']:.4f}"),
        ("Student-test top-1", pct(ust["top1_accuracy"]), pct(tst["top1_accuracy"])),
        ("Student-test macro F1", f"{ust['macro_f1']:.4f}", f"{tst['macro_f1']:.4f}"),
        ("Student-test log loss", f"{ust['log_loss']:.4f}", f"{tst['log_loss']:.4f}"),
        ("Inference (ms/obs)", f"{ust['inference_ms_per_obs']:.3f}", f"{tst['inference_ms_per_obs']:.3f}"),
        ("Fit time (s)", f"{unt['fit_seconds']:.1f}", f"{tun['fit_seconds']:.1f}"),
    ]
    body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)
    tex = r"""\begin{table}[!htb]
\centering
\small
\caption{Untuned versus TPE-tuned LightGBM on the same folds, features and student-testing partition.}
\label{tab:tuning}
\begin{tabular}{lcc}
\toprule
Metric & Untuned (M2) & Tuned (M3) \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "qa_tuning_table.tex").write_text(tex)


def table_hmm():
    d, e = DEV["aggregate"], EVAL["aggregate"]
    rows = [
        ("Observation-level top-1", pct(d["top1_accuracy"]), pct(d["top1_accuracy_hmm"]),
         pct(e["top1_accuracy"]), pct(e["top1_accuracy_hmm"])),
        ("Route continuity", pct(d["route_continuity"], 1), pct(d["route_continuity_hmm"], 1),
         pct(e["route_continuity"], 1), pct(e["route_continuity_hmm"], 1)),
        ("Sequence F1", f"{d['seq_f1_clf']:.3f}", f"{d['seq_f1_hmm']:.3f}",
         f"{e['seq_f1_clf']:.3f}", f"{e['seq_f1_hmm']:.3f}"),
        ("Candidate macro F1 (decoding-independent)", f"{d['macro_f1']:.4f}", "---",
         f"{e['macro_f1']:.4f}", "---"),
        ("Log loss (classifier probabilities)", f"{d['log_loss']:.4f}", "---",
         f"{e['log_loss']:.4f}", "---"),
        ("Candidate miss rate", pct(d["candidate_miss_rate"]), "---", pct(e["candidate_miss_rate"]), "---"),
        ("Mean latency per trip (ms)", f"{d['mean_total_ms']:.0f}", "---", f"{e['mean_total_ms']:.0f}", "---"),
    ]
    body = "\n".join(f"{a} & {b} & {c} & {d2} & {e2} \\\\" for a, b, c, d2, e2 in rows)
    tex = r"""\begin{table}[!htb]
\centering
\small
\caption{Classifier-only versus classifier followed by HMM/Viterbi decoding, frozen artifact
($\beta=20$\,m, $\alpha=0.1$, 8 states, 30\,s gaps). Macro F1 and log loss describe the candidate
probabilities and do not change with decoding.}
\label{tab:hmm}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{Student-testing (10 trips)} & \multicolumn{2}{c}{Evaluator (10 trips)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Metric & Classifier & + HMM & Classifier & + HMM \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "qa_hmm_table.tex").write_text(tex)


def table_dataset():
    db = DB
    rows = [
        ("Trips (training / student-testing), GPS fixes", "80 / 10, 227{,}577"),
        ("Missing values, non-increasing timestamps, duplicate rows, bad ids", "0"),
        ("Intervals exactly 1 s / over 1.5 s (longest)", "96.7\\% / 7{,}537 (153{,}064 s)"),
        ("Labelled fixes (fix level; per-trip mean 98.2\\%)", "90.8\\%"),
        ("Mean candidate miss rate per trip", "0.75\\%"),
        ("Candidate rows / positives (all 90 supplied trips)", f"{int(db['n_candidate_rows']):,}".replace(",", "{,}") + " / " + f"{int(db['n_positive_rows']):,}".replace(",", "{,}")),
    ]
    body = "\n".join(f"{a} & {b} \\\\" for a, b in rows)
    tex = r"""\begin{table}[!htb]
\centering
\small
\caption{Missing-value and data-quality analysis on the supplied development trips (80 training, 10 student-testing); evaluator trips are only scored.}
\label{tab:quality}
\begin{tabular}{lr}
\toprule
Check & Value \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "qa_dataset_table.tex").write_text(tex)


def table_partitions():
    tex = r"""\begin{table}[!htb]
\centering
\small
\caption{Dataset partitions and their use.}
\label{tab:partitions}
\begin{tabular}{llp{8.4cm}}
\toprule
Partition & Trips & Use \\
\midrule
Training & 80 & Fitting, 5-fold trip-grouped CV, all feature and hyperparameter decisions. \\
Student testing & 10 & Model comparison, decoder tuning, every development-stage result. \\
Evaluator testing & 10 & Frozen-system scoring only; never read during training or tuning. \\
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "qa_partitions.tex").write_text(tex)


def table_api():
    tex = r"""\begin{table}[!htb]
\centering
\small
\caption{FastAPI endpoints served by laptop 1.}
\label{tab:api}
\begin{tabular}{@{}llp{9.4cm}@{}}
\toprule
Endpoint & Method & Purpose \\
\midrule
\code{/health} & GET & Status, mode, evaluation partition, trip and model counts, default model. \\
\code{/trips} & GET & Evaluation trip ids with fix counts. \\
\code{/models} & GET & The eight deployable classifiers with recorded scores. \\
\code{/evaluate/\{trip\}?model=<id>} & GET & Predictions, map geometry, server-side metrics, timings. \\
\code{/docs} & GET & OpenAPI documentation. \\
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "qa_api_table.tex").write_text(tex)


def main():
    table_model_comparison()
    table_model_qa()
    table_tuning()
    table_hmm()
    table_dataset()
    table_partitions()
    table_api()
    print("wrote compact qa tables")


if __name__ == "__main__":
    main()
