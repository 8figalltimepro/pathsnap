#!/usr/bin/env bash
# Reproduce the full PathSnap pipeline from raw data to final artifact + report.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH=src

echo "== 1/9 build candidate dataset"
$PY scripts/build_dataset.py

echo "== 2/9 train and compare models (12 models, 5-fold trip-grouped CV)"
$PY scripts/run_experiments.py --models gaussian_nb,decision_tree,lightgbm,xgboost,hist_gb --folds 5
$PY scripts/run_experiments.py --models logreg,random_forest,extra_trees,mlp --folds 5
$PY scripts/run_experiments.py --models knn --folds 5

echo "== 3/9 Bayesian tuning (local fallback; the Kaggle kernel does the same search)"
$PY scripts/tune_lightgbm_local.py
$PY scripts/integrate_kaggle_results.py || echo "  (Kaggle result files not present; skipping MLP integration)"

echo "== 4/9 refit tuned model and compare against untuned"
$PY scripts/run_experiments.py --models lightgbm_tuned --folds 5
$PY scripts/train_final.py

echo "== 5/9 export the eight deployable models for the UI picker"
$PY scripts/export_models.py

echo "== 6/9 tune the HMM decoder on student_test and rebuild the artifact"
$PY scripts/tune_hmm.py --probs 'artifacts/cache/preds/final_{trip}.npz' --models --out hmm_tuning_best.json
$PY scripts/train_final.py

echo "== 7/9 evaluate the production path"
$PY scripts/evaluate_final.py --baseline

echo "== 8/9 minimal report assets (figure, tables, map) and interactive maps"
$PY scripts/make_report_min.py
$PY scripts/make_report_map.py
$PY scripts/make_ui_figure.py
$PY scripts/make_maps.py

echo "== 9/9 compile the minimal technical report"
(cd reports && pdflatex -interaction=nonstopmode report_min.tex >/dev/null && \
 pdflatex -interaction=nonstopmode report_min.tex >/dev/null && \
 cp report_min.pdf PathSnap_Technical_Report.pdf)
echo "done: reports/report_min.pdf (copied to reports/PathSnap_Technical_Report.pdf)"
