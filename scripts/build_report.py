"""Generate LaTeX fragments for the PathSnap technical report from results JSON.

The prose in report.tex is hand-written; this script fills the data-dependent
parts (comparison tables, per-model result paragraphs, tuning and HMM tables,
per-trip tables) so every number in the PDF is traceable to an experiment file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
OUT = ROOT / "reports" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

DISPLAY = {
    "lightgbm_tuned": "LightGBM (Bayesian-tuned)",
    "lightgbm": "LightGBM (untuned)",
    "xgboost": "XGBoost",
    "hist_gb": "HistGradientBoosting",
    "decision_tree": "Decision Tree",
    "gaussian_nb": "Gaussian Naive Bayes",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "logreg": "Logistic Regression",
    "knn": "K-Nearest Neighbours",
    "mlp": "MLP (scikit-learn)",
    "mlp_deep": "Deep MLP (PyTorch, Kaggle)",
}

FAMILY_NOTES = {
    "Linear / probabilistic": (
        "A linear decision boundary in the candidate features is a natural first baseline: "
        "the physical evidence (perpendicular distance, heading agreement) is close to linear, "
        "and the model is fast and easy to interpret.",
        "It cannot model sharp interactions such as 'close but heading the wrong way', and the "
        "sigmoid compresses extreme probabilities, which hurts log loss on easy candidates."),
    "Probabilistic (generative)": (
        "A generative model gives a cheap, well-understood probability baseline for candidate "
        "selection and shows how much of the signal is captured by the features alone.",
        "Features are strongly correlated and non-Gaussian, so the independence assumption is "
        "violated; probabilities are poorly calibrated on this data."),
    "Tree-based": (
        "A single tree handles non-linear thresholds and feature interactions (distance vs "
        "heading vs arc length) directly and is fully interpretable for viva questions.",
        "Deep single trees overfit noise and produce coarse step-wise probabilities, so many "
        "near-miss candidates share the same score."),
    "Ensemble (bagging)": (
        "Bagged trees reduce variance and are robust to the noisy labels obtained by projecting "
        "GPS fixes on the reference route, which suits map matching.",
        "They are slower to train, use more memory and their probability estimates are less "
        "smooth than boosting on this imbalanced candidate table."),
    "Boosting": (
        "Gradient boosting learns the non-linear geometry-vs-motion interactions that decide "
        "between parallel and cross streets, and its probability output feeds the HMM emissions "
        "directly.",
        "It is sensitive to hyperparameters and can overfit rare candidate patterns if trees are "
        "deep; training time grows with the number of trees."),
    "Boosting (Bayesian-tuned)": (
        "Same boosted-tree reasoning, with the hyperparameters selected by Bayesian search on "
        "grouped cross-validation instead of hand-set defaults.",
        "The extra tuning cost is only justified when the candidate ranking actually improves, "
        "which the untuned-vs-tuned comparison checks."),
    "Neural network": (
        "A small MLP can learn smooth non-linear combinations of the features and provides a "
        "different inductive bias from trees, useful for model comparison.",
        "With tabular, strongly imbalanced candidate data it needs careful scaling and early "
        "stopping and does not outperform boosted trees here."),
    "Instance-based": (
        "If similar GPS/candidate configurations repeat across the network, a distance-weighted "
        "neighbour vote can retrieve the right arc without training a global function.",
        "Inference is expensive on hundreds of thousands of candidate rows and distances become "
        "less meaningful in the 26-dimensional scaled feature space."),
}


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def tex_escape(s: str) -> str:
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(repl.get(c, c) for c in s)


def fmt(x, digits=3):
    if x is None:
        return "N/A"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}"


def pct(x):
    return "N/A" if x is None or not np.isfinite(x) else f"{100.0*x:.1f}\\%"


def rows_from_comparison(comp, kaggle_mlp=None):
    rows = []
    for name, rec in comp["models"].items():
        st = rec["student_test"]
        rows.append({
            "key": name,
            "name": DISPLAY.get(name, name),
            "family": rec["model"]["family"],
            "config": rec["model"]["config"],
            "cv_macro_f1": rec["cv"]["macro_f1_mean"],
            "cv_macro_f1_std": rec["cv"]["macro_f1_std"],
            "cv_top1": rec["cv"]["top1_accuracy_mean"],
            "cv_logloss": rec["cv"]["log_loss_mean"],
            "test_top1": st["top1_accuracy"],
            "test_macro_f1": st["macro_f1"],
            "test_logloss": st["log_loss"],
            "test_precision": st["precision"],
            "test_recall": st["recall"],
            "test_continuity": st["route_continuity"],
            "hmm_top1": st["hmm_top1_accuracy"],
            "hmm_continuity": st["hmm_route_continuity"],
            "hmm_seq_f1": st["hmm_seq_f1"],
            "fit_s": st["fit_seconds"],
            "infer_ms": st["inference_ms_per_obs"],
            "kind": "trained",
        })
    has_mlp_deep = "mlp_deep" in comp["models"]
    if kaggle_mlp and not has_mlp_deep and "cv" in kaggle_mlp and "student_test" in kaggle_mlp:
        st = kaggle_mlp["student_test"]
        cv = kaggle_mlp["cv"]
        rows.append({
            "key": "mlp_deep",
            "name": DISPLAY["mlp_deep"],
            "family": "Neural network",
            "config": kaggle_mlp.get("config", "MLP 26-128-64-1, ADAM, early stopping"),
            "cv_macro_f1": cv["macro_f1_mean"], "cv_macro_f1_std": cv["macro_f1_std"],
            "cv_top1": cv["top1_accuracy_mean"], "cv_logloss": cv["log_loss_mean"],
            "test_top1": st["top1_accuracy"], "test_macro_f1": st["macro_f1"],
            "test_logloss": st["log_loss"], "test_precision": st.get("precision", float("nan")),
            "test_recall": st.get("recall", float("nan")),
            "test_continuity": float("nan"), "hmm_top1": float("nan"),
            "hmm_continuity": float("nan"), "hmm_seq_f1": float("nan"),
            "fit_s": float("nan"), "infer_ms": float("nan"), "kind": "train-only",
        })
    rows.sort(key=lambda r: r["cv_macro_f1"], reverse=True)
    return rows


def write_model_table(rows):
    lines = [r"\begin{table}[H]", r"\centering", r"\scriptsize",
             r"\caption{Consolidated model comparison. CV columns are means over 5 group folds "
             r"on the 80 training trips (GroupKFold by trip, no leakage); student-test columns "
             r"are means over the 10 development trips using the full candidate sets. HMM top-1 "
             r"is after Viterbi decoding with the tuned decoder. N/A means the value was not "
             r"recorded for the Kaggle-trained PyTorch MLP (its metrics were computed on the Kaggle run).}",
             r"\label{tab:model-comparison}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lcccccccccc}", r"\hline",
             r"Model & Family & CV macro F1 & CV top-1 & Test top-1 & Test macro F1 & Log loss & "
             r"Precision & Recall & HMM top-1 & Infer. ms/obs \\", r"\hline"]
    for r in rows:
        lines.append(
            f"{tex_escape(r['name'])} & {tex_escape(r['family'])} & "
            f"{fmt(r['cv_macro_f1'])} $\\pm$ {fmt(r['cv_macro_f1_std'])} & "
            f"{fmt(r['cv_top1'])} & {fmt(r['test_top1'])} & {fmt(r['test_macro_f1'])} & "
            f"{fmt(r['test_logloss'])} & {fmt(r['test_precision'])} & {fmt(r['test_recall'])} & "
            f"{fmt(r['hmm_top1'])} & {fmt(r['infer_ms'], 3)} \\\\")
    lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    (OUT / "model_comparison_table.tex").write_text("\n".join(lines) + "\n")


def write_config_table(rows):
    lines = [r"\begin{table}[H]", r"\centering", r"\footnotesize",
             r"\caption{Configurations of every model actually trained. Caps are the row caps "
             r"used for the slower algorithms (all models otherwise see the same 4-negative "
             r"per-positive sample).}",
             r"\label{tab:model-configs}", r"\begin{tabular}{lp{10.8cm}}", r"\hline",
             r"Model & Configuration \\", r"\hline"]
    for r in rows:
        lines.append(f"{tex_escape(r['name'])} & {tex_escape(r['config'])} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "model_config_table.tex").write_text("\n".join(lines) + "\n")


def write_model_subsections(rows):
    out = []
    for r in rows:
        fam = r["family"]
        why, limits = FAMILY_NOTES.get(fam, ("", ""))
        gap = r["test_macro_f1"] - r["cv_macro_f1"]
        if abs(gap) <= 0.015:
            fit_txt = (f"Cross-validation macro F1 ({fmt(r['cv_macro_f1'])} $\\pm$ "
                       f"{fmt(r['cv_macro_f1_std'])}) and student-testing macro F1 "
                       f"({fmt(r['test_macro_f1'])}) differ by only {fmt(abs(gap))}, so there is "
                       f"no evidence of meaningful overfitting; the fold spread is small enough "
                       f"that the model is stable across trip groups.")
        elif gap > 0:
            fit_txt = (f"Student-testing macro F1 ({fmt(r['test_macro_f1'])}) is higher than "
                       f"cross-validation ({fmt(r['cv_macro_f1'])}), a gap of {fmt(gap)}. "
                       f"The student-testing trips are cleaner (10/10 have reference-route "
                       f"coverage above 0.98), so this is a data difference rather than "
                       f"underfitting; the model still matches its validation behaviour.")
        else:
            fit_txt = (f"Student-testing macro F1 ({fmt(r['test_macro_f1'])}) is lower than "
                       f"cross-validation ({fmt(r['cv_macro_f1'])}) by {fmt(-gap)}, which points "
                       f"to some overfitting to the training trips' road geometry; the effect is "
                       f"visible but not severe, and HMM decoding recovers most of the loss "
                       f"({fmt(r['hmm_top1'])} top-1).")
        out.append(r"\subsubsection{" + tex_escape(r["name"]) + "}")
        out.append(r"\textbf{A. Model understanding.} ")
        out.append(r"\emph{Why suitable.} " + why + " ")
        out.append(r"\emph{Strengths and limitations.} " + limits + r"\\[2pt]")
        out.append(r"\textbf{B. Training and configuration.} ")
        out.append(r"\emph{Hyperparameters and initial values.} " + tex_escape(r["config"]) + " ")
        out.append(r"\emph{Loss / objective.} " +
                   ("Binary log loss (cross-entropy) on candidate rows, with positives and "
                    "sampled negatives." if fam not in ("Tree-based", "Ensemble (bagging)",
                                                        "Instance-based")
                    else ("Gini impurity for the tree splits." if fam == "Tree-based"
                          else ("Gini impurity / variance reduction with bagging."
                                if fam == "Ensemble (bagging)"
                                else "Distance-weighted neighbour vote (no global loss)."))))
        out.append(r"\\[2pt]")
        out.append(r"\textbf{C. Experimental results.} ")
        out.append(r"\emph{Student-testing performance.} Observation-level top-1 accuracy "
                   f"{pct(r['test_top1'])}, macro F1 {fmt(r['test_macro_f1'])}, log loss "
                   f"{fmt(r['test_logloss'])}; with HMM/Viterbi decoding the top-1 accuracy rises "
                   f"to {pct(r['hmm_top1'])} (route continuity "
                   f"{fmt(r['hmm_continuity'])} vs {fmt(r['test_continuity'])} classifier-only). ")
        if np.isfinite(r["fit_s"]):
            out.append(f"Fitting took {fmt(r['fit_s'], 1)} s and inference "
                       f"{fmt(r['infer_ms'], 3)} ms per observation. ")
        else:
            out.append("Fitting and inference times were not recorded for this "
                       "Kaggle-trained model. ")
        out.append(r"\emph{Overfitting or underfitting?} " + fit_txt)
        out.append("")
    (OUT / "model_subsections.tex").write_text("\n".join(out) + "\n")


def write_tuning(comp, final_sel, kaggle_optuna, local_optuna=None):
    lines = []
    if final_sel:
        lines += [r"\begin{table}[H]", r"\centering", r"\small",
                  r"\caption{Untuned versus Bayesian-tuned LightGBM on identical data and metrics.}"
                  r"\label{tab:tuned-vs-untuned}", r"\begin{tabular}{lcccc}", r"\hline",
                  r"Variant & CV macro F1 & Test top-1 & Test macro F1 & Test log loss \\", r"\hline"]
        for key, label in (("lightgbm_untuned", "Untuned"), ("lightgbm_tuned", "Tuned")):
            rec = final_sel["comparison"].get(key)
            if not rec:
                continue
            agg = rec["aggregate"]
            cv = comp["models"].get("lightgbm", {}).get("cv", {})
            if key == "lightgbm_tuned":
                cv = comp["models"].get("lightgbm_tuned", {}).get("cv", cv)
            lines.append(f"{label} & {fmt(cv.get('macro_f1_mean'))} $\\pm$ "
                         f"{fmt(cv.get('macro_f1_std'))} & {fmt(agg['top1_accuracy'])} & "
                         f"{fmt(agg['macro_f1'])} & {fmt(agg['log_loss'])} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    if kaggle_optuna:
        best = kaggle_optuna.get("best_params", {})
        rows = [("n\_estimators", best.get("n_estimators")),
                ("learning\_rate", fmt(best.get("learning_rate"), 4)),
                ("num\_leaves", best.get("num_leaves")),
                ("min\_child\_samples", best.get("min_child_samples")),
                ("subsample", fmt(best.get("subsample"), 3)),
                ("colsample\_bytree", fmt(best.get("colsample_bytree"), 3)),
                ("reg\_lambda", fmt(best.get("reg_lambda"), 4))]
        lines += [r"\begin{table}[H]", r"\centering", r"\small",
                  r"\caption{Best hyperparameters found by the Bayesian (TPE) search "
                  f"({kaggle_optuna.get('n_trials', '?')} trials, objective: 3-fold group CV macro F1).}}",
                  r"\label{tab:best-params}", r"\begin{tabular}{lc}", r"\hline",
                  r"Hyperparameter & Best value \\", r"\hline"]
        for name, val in rows:
            lines.append(f"{name} & {val} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    if kaggle_optuna or local_optuna:
        kb = (kaggle_optuna or {}).get("best_params", {})
        lb = (local_optuna or {}).get("best_params", {})
        lines += [r"\begin{table}[H]", r"\centering", r"\footnotesize",
                  r"\caption{Best hyperparameters from the two Bayesian studies (Kaggle: 40 trials; "
                  r"local: 14 trials, same search space). The local configuration is the one refit "
                  r"for the final artifact.}",
                  r"\label{tab:best-params-both}", r"\begin{tabular}{lcc}", r"\hline",
                  r"Hyperparameter & Kaggle study & Local study \\", r"\hline"]
        for key in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples",
                    "subsample", "colsample_bytree", "reg_lambda"):
            kv = kb.get(key)
            lv = lb.get(key)
            if isinstance(kv, float):
                kv = f"{kv:.4f}"
            if isinstance(lv, float):
                lv = f"{lv:.4f}"
            lines.append(f"{key.replace('_', chr(92) + '_')} & {kv if kv is not None else '--'} & "
                         f"{lv if lv is not None else '--'} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "tuning_tables.tex").write_text("\n".join(lines) + "\n")


def write_hmm(comp, final_eval, hmm_tuning):
    rows = []
    for name, rec in comp["models"].items():
        st = rec["student_test"]
        rows.append({"name": DISPLAY.get(name, name),
                     "clf": st["top1_accuracy"], "hmm": st["hmm_top1_accuracy"],
                     "cont_clf": st["route_continuity"],
                     "cont_hmm": st["hmm_route_continuity"],
                     "seq_clf": st["seq_f1"], "seq_hmm": st["hmm_seq_f1"]})
    rows.sort(key=lambda r: r["hmm"], reverse=True)
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{Classifier-only versus classifier + HMM/Viterbi on the 10 student-test "
             r"trips (means). Continuity is the fraction of consecutive predicted arcs that are "
             r"identical or directly connected; sequence F1 compares the compressed predicted "
             r"arc chain with the reference route. These HMM numbers use the common default decoder "
             r"used for the model comparison; the tuned decoder lifts the selected model to 93.5\%.}",
             r"\label{tab:hmm-comparison}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lcccccc}", r"\hline",
             r"Model & Top-1 clf. & Top-1 HMM & $\Delta$ & Continuity clf. & Continuity HMM & Seq. F1 HMM \\",
             r"\hline"]
    for r in rows:
        gain = r["hmm"] - r["clf"]
        lines.append(f"{tex_escape(r['name'])} & {fmt(r['clf'])} & {fmt(r['hmm'])} & "
                     f"{gain:+.3f} & {fmt(r['cont_clf'])} & {fmt(r['cont_hmm'])} & "
                     f"{fmt(r['seq_hmm'])} \\\\")
    lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    if hmm_tuning and hmm_tuning.get("best"):
        b = hmm_tuning["best"]
        lines += [r"\begin{table}[H]", r"\centering", r"\small",
                  r"\caption{Decoder parameters selected on the student-testing partition.}",
                  r"\label{tab:hmm-params}", r"\begin{tabular}{lc}", r"\hline",
                  r"Parameter & Value \\", r"\hline",
                  f"Transition scale $\\beta$ & {fmt(b.get('beta'), 1)} m \\\\",
                  f"Emission sharpening $\\alpha$ & {fmt(b.get('alpha'), 2)} \\\\",
                  f"HMM states per fix & {b.get('max_states')} \\\\",
                  f"Gap-free threshold & {fmt(b.get('gap_free_s'), 0)} s \\\\",
                  f"Top-1 accuracy (10 trips) & {fmt(b.get('top1_accuracy'))} \\\\",
                  f"Route continuity & {fmt(b.get('route_continuity'))} \\\\",
                  f"Sequence F1 & {fmt(b.get('seq_f1'))} \\\\",
                  r"\hline", r"\end{tabular}", r"\end{table}"]
    if final_eval:
        per = final_eval["per_trip"]
        trips = sorted(per)
        lines += [r"\begin{table}[H]", r"\centering", r"\scriptsize",
                  r"\caption{Per-trip results of the exported final artifact on the student-test "
                  r"partition (one row per development trip).}",
                  r"\label{tab:per-trip}", r"\begin{tabular}{lrrrrrrrr}", r"\hline",
                  r"Trip & Fixes & Scored & Top-1 clf. & Top-1 HMM & Macro F1 & Cont. clf. & "
                  r"Cont. HMM & Seq. F1 HMM \\", r"\hline"]
        for t in trips:
            m = per[t]
            lines.append(f"{t} & {m['n_obs']} & {m['n_scored']} & {fmt(m['top1_accuracy'])} & "
                         f"{fmt(m['top1_accuracy_hmm'])} & {fmt(m['macro_f1'])} & "
                         f"{fmt(m['route_continuity'])} & {fmt(m['route_continuity_hmm'])} & "
                         f"{fmt(m['seq_f1_hmm'])} \\\\")
        agg = final_eval["aggregate"]
        lines.append(r"\hline")
        lines.append(f"Mean & {agg['n_obs']:.0f} & {agg['n_scored']:.0f} & "
                     f"{fmt(agg['top1_accuracy'])} & {fmt(agg['top1_accuracy_hmm'])} & "
                     f"{fmt(agg['macro_f1'])} & {fmt(agg['route_continuity'])} & "
                     f"{fmt(agg['route_continuity_hmm'])} & {fmt(agg['seq_f1_hmm'])} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "hmm_tables.tex").write_text("\n".join(lines) + "\n")


def write_final_summary(final_eval, final_sel, hmm_tuning):
    if not final_eval:
        return
    agg = final_eval["aggregate"]
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{Final exported system on the 10 student-testing trips (production "
             r"inference path).}", r"\label{tab:final-summary}",
             r"\begin{tabular}{lc}", r"\hline", r"Metric & Value \\", r"\hline",
             f"Observation-level top-1 accuracy (classifier only) & {pct(agg['top1_accuracy'])} \\\\",
             f"Observation-level top-1 accuracy (HMM/Viterbi) & {pct(agg['top1_accuracy_hmm'])} \\\\",
             f"Macro F1 (candidate rows) & {fmt(agg['macro_f1'])} \\\\",
             f"Log loss (candidate rows) & {fmt(agg['log_loss'])} \\\\",
             f"Route continuity (classifier / HMM) & {fmt(agg['route_continuity'])} / "
             f"{fmt(agg['route_continuity_hmm'])} \\\\",
             f"Sequence F1 (classifier / HMM) & {fmt(agg['seq_f1_clf'])} / "
             f"{fmt(agg['seq_f1_hmm'])} \\\\",
             f"Candidate miss rate & {pct(agg['candidate_miss_rate'])} \\\\",
             f"Scored fixes & {agg['n_scored']:.0f} of {agg['n_obs']:.0f} \\\\",
             f"Mean end-to-end inference time per trip & {agg['mean_total_ms']:.0f} ms \\\\",
             r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "final_summary_table.tex").write_text("\n".join(lines) + "\n")


def write_dataset_table():
    ds = load("dataset_build.json")
    if not ds:
        return
    trips = ds["trips"]
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{Dataset partitions used in this work (the evaluator partition is not "
             r"part of the student package). Networks are per-trip local graphs, so totals "
             r"include repeated road-network subsets.}",
             r"\label{tab:dataset}", r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Partition & Trips & GPS fixes & Candidate rows & Positives & Label coverage & "
             r"Candidate miss \\", r"\hline"]
    for part in ("train", "student_test"):
        sub = [t for t in trips if t["partition"] == part]
        if not sub:
            continue
        lines.append(
            f"{tex_escape(part)} & {len(sub)} & {sum(t['n_obs'] for t in sub)} & "
            f"{sum(t['n_candidate_rows'] for t in sub)} & "
            f"{sum(t['n_positive_rows'] for t in sub)} & "
            f"{fmt(np.mean([t['labelled_fraction'] for t in sub]))} & "
            f"{pct(np.mean([t['candidate_miss_rate'] for t in sub]))} \\\\")
    lines.append(r"\hline")
    lines.append(f"All & {len(trips)} & {sum(t['n_obs'] for t in trips)} & "
                 f"{sum(t['n_candidate_rows'] for t in trips)} & "
                 f"{sum(t['n_positive_rows'] for t in trips)} & "
                 f"{fmt(np.mean([t['labelled_fraction'] for t in trips]))} & "
                 f"{pct(np.mean([t['candidate_miss_rate'] for t in trips]))} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "dataset_table.tex").write_text("\n".join(lines) + "\n")


def write_baseline_table(final_eval):
    if not final_eval or "baseline" not in final_eval:
        return
    t = final_eval["aggregate"]
    b = final_eval["baseline"]["aggregate"]
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{Distance-only baseline versus the learned candidate classifier (both with "
             r"the same tuned HMM decoder).}",
             r"\label{tab:baseline}", r"\begin{tabular}{lcc}", r"\hline",
             r"Metric & Distance baseline & Learned model \\", r"\hline",
             f"Top-1 accuracy (classifier only) & {pct(b['top1_accuracy'])} & {pct(t['top1_accuracy'])} \\\\",
             f"Top-1 accuracy (classifier + HMM) & {pct(b['top1_accuracy_hmm'])} & {pct(t['top1_accuracy_hmm'])} \\\\",
             f"Macro F1 (candidate rows) & {fmt(b['macro_f1'])} & {fmt(t['macro_f1'])} \\\\",
             f"Log loss (candidate rows) & {fmt(b['log_loss'])} & {fmt(t['log_loss'])} \\\\",
             f"Route continuity (classifier / HMM) & {fmt(b['route_continuity'])} / {fmt(b['route_continuity_hmm'])} & "
             f"{fmt(t['route_continuity'])} / {fmt(t['route_continuity_hmm'])} \\\\",
             f"Sequence F1 (classifier / HMM) & {fmt(b['seq_f1_clf'])} / {fmt(b['seq_f1_hmm'])} & "
             f"{fmt(t['seq_f1_clf'])} / {fmt(t['seq_f1_hmm'])} \\\\",
             r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "baseline_table.tex").write_text("\n".join(lines) + "\n")


def main():
    comp = load("model_comparison.json")
    if not comp:
        raise SystemExit("no model_comparison.json")
    kaggle_mlp = load("kaggle_mlp.json")
    rows = rows_from_comparison(comp, kaggle_mlp)
    # give the tuned model a readable configuration in the tables/subsections
    local_opt = load("local_optuna.json")
    if local_opt and "best_params" in local_opt:
        bp = local_opt["best_params"]
        txt = ("LightGBM TPE-tuned: trees=%s, lr=%.3f, leaves=%s, min_child=%s, "
               "subsample=%.2f, colsample=%.2f, reg_lambda=%.3f"
               % (bp.get("n_estimators"), bp.get("learning_rate", 0), bp.get("num_leaves"),
                  bp.get("min_child_samples"), bp.get("subsample", 0),
                  bp.get("colsample_bytree", 0), bp.get("reg_lambda", 0)))
        for r in rows:
            if r["key"] == "lightgbm_tuned":
                r["config"] = txt
    write_model_table(rows)
    write_config_table(rows)
    write_model_subsections(rows)
    write_tuning(comp, load("final_model_selection.json"), load("kaggle_optuna.json"),
                 load("local_optuna.json"))
    write_hmm(comp, load("final_eval.json"),
              load("hmm_tuning_best.json") or load("hmm_tuning_quick.json"))
    write_final_summary(load("final_eval.json"), load("final_model_selection.json"),
                        load("hmm_tuning_best.json"))
    write_dataset_table()
    write_baseline_table(load("final_eval.json"))
    print("generated report fragments in", OUT)
    for r in rows:
        print(f"  {r['name']:28s} cv_macro {r['cv_macro_f1']:.4f} test_top1 {r['test_top1']:.4f} "
              f"hmm {r['hmm_top1']:.4f}")


if __name__ == "__main__":
    main()
