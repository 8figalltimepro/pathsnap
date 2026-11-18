# PathSnap - Distributed Map-Matching and Route Reconstruction

PathSnap matches noisy vehicle GPS traces to a road network and reconstructs the driven route.
Each GPS fix gets a set of nearby road-arc candidates, a trained classifier scores those
candidates, and an HMM/Viterbi decoder turns the independent scores into a connected route.

Student project for **UE24CS352A Machine Learning (PES University, RR Campus)** hackathon.

## Results (10 student-testing trips)

| Metric | Classifier only | Classifier + HMM |
|---|---|---|
| Observation-level top-1 accuracy | 87.8% | **93.5%** |
| Route continuity | 93.4% | **97.8%** |
| Sequence F1 | 0.69 | **0.91** |

Distance-only baseline with the same decoder: 56.4% classifier-only, 92.6% with HMM - the learned
classifier contributes ~31 points before decoding and ~0.9 points after it.

**Reserved evaluator trips (frozen model):** 88.7% classifier-only, **94.9%** decoded top-1,
macro F1 0.935, route continuity 92.8% -> 96.4%, sequence F1 0.70 -> 0.87, 0.38 s/trip; baseline
54.8%/92.1%. Results in artifacts/results/evaluator_eval.json; reproduce with
`scripts/evaluate_final.py --partition evaluator_test --out evaluator_eval.json --baseline`.

Final model: LightGBM candidate classifier (Bayesian-tuned), 26 features, 4-negative sampling,
5-fold trip-grouped CV; HMM decoder tuned on the development partition (beta=20, alpha=0.1, 8 states).
The technical report (7 pages, question-and-answer format following the brief) is submitted
separately as the report deliverable and is not part of this code repository.

## Repository layout

    src/pathsnap/          core library
      io.py                raw .track/.nodes/.arcs/.route loading
      geo.py               local metric frame, haversine, projections
      candidates.py        spatial-index candidate generation per trip
      labels.py            reference-route labelling (monotone DP)
      features.py          26 candidate features
      hmm.py               HMM transitions + Viterbi decoder
      metrics.py           candidate / route metrics
      pipeline.py          end-to-end prediction, scoring, API payload, model registry
    scripts/               reproducible experiment scripts (see below)
    app/main.py            FastAPI backend (laptop 1)
    ui/                    Vercel-style Leaflet UI (laptop 2) with live model picker
    kaggle/experiments/    Kaggle kernel used for the MLP + Optuna CPU run
    artifacts/model.joblib final artifact (model + features + HMM params)
    artifacts/models/      eight deployable models used by the UI picker
    artifacts/results/     experiment logs, metrics, tuning trials, model registry
    configs/               pipeline configuration snapshot

## Deliverables map (hackathon brief)

| Required component | Where |
|---|---|
| Data processing | src/pathsnap/io.py, candidates.py, labels.py, scripts/build_dataset.py |
| Feature engineering | src/pathsnap/features.py (26 features) |
| Model training (5+ models) | scripts/run_experiments.py, artifacts/results/model_comparison.json |
| Model evaluation / plots | scripts/make_qa_figures.py, scripts/make_report_qa.py, reports/figures/qa/, Table 3 of the report |
| Bayesian optimisation | scripts/tune_lightgbm_local.py, kaggle/experiments/patthsnap_experiments.py, artifacts/results/local_optuna.json, kaggle_optuna.json |
| Deployable models | scripts/export_models.py, artifacts/models/*.joblib, artifacts/results/model_registry.json |
| HMM and Viterbi | src/pathsnap/hmm.py, scripts/tune_hmm.py, artifacts/results/hmm_tuning_best.json |
| API (laptop 1) | app/main.py, scripts/run_api.sh |
| UI (laptop 2) | ui/, scripts/run_ui.sh - live model picker over 8 deployable models |
| Model + config artifacts | artifacts/model.joblib, artifacts/model_metadata.json, configs/pipeline_config.json |
| Visualisations | scripts/make_report_map.py, scripts/make_maps.py, scripts/eda_analysis.py (Folium maps and figures) |
| Supporting results | artifacts/results/*.json and *.csv, experiment logs |
| Technical report | submitted separately as the report PDF (Deliverable 2) |
| Reproducible pipeline | scripts/run_all.sh or the step-by-step commands below |

## Data

The course package is not redistributed. Put it next to the code as:

    Student folder/train/<trip>/<trip>.{track,nodes,arcs,route}        80 trips
    Student folder/student_test/<trip>/<trip>.{track,nodes,arcs,route} 10 trips

For final evaluation, put the reserved trips in evaluator_test/ next to the repository root (same
layout). The API discovers that partition automatically and lists it in the UI; the student-testing
partition is used as a demo fallback. Ground-truth .route files stay server side and are never sent
to the UI.

## Setup

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt

Python 3.12 is used locally; a Kaggle CPU session was used for the PyTorch MLP and one Optuna study.

## Reproduce everything

    # 1. candidates, labels and features for all 90 trips (~1 min)
    PYTHONPATH=src .venv/bin/python scripts/build_dataset.py

    # 2. train + group-CV + student-test comparison (the ten base models)
    PYTHONPATH=src .venv/bin/python scripts/run_experiments.py \
        --models gaussian_nb,decision_tree,lightgbm,xgboost,hist_gb,logreg,random_forest,extra_trees,mlp --folds 5
    PYTHONPATH=src .venv/bin/python scripts/run_experiments.py --models knn --folds 5

    # 3. Bayesian (TPE) tuning: Kaggle kernel (40 trials) and local fallback (14 trials)
    PYTHONPATH=src .venv/bin/python scripts/tune_lightgbm_local.py
    PYTHONPATH=src .venv/bin/python scripts/integrate_kaggle_results.py   # if Kaggle results present

    # 4. evaluate the tuned configuration, refit and export; export picker models
    PYTHONPATH=src .venv/bin/python scripts/run_experiments.py --models lightgbm_tuned --folds 5
    PYTHONPATH=src .venv/bin/python scripts/train_final.py
    PYTHONPATH=src .venv/bin/python scripts/export_models.py

    # 5. tune the HMM decoder, then rebuild the final artifact
    PYTHONPATH=src .venv/bin/python scripts/tune_hmm.py --probs 'artifacts/cache/preds/final_{trip}.npz' --models --out hmm_tuning_best.json
    PYTHONPATH=src .venv/bin/python scripts/train_final.py

    # 6. final evaluation through the production path
    PYTHONPATH=src .venv/bin/python scripts/evaluate_final.py --baseline

    # 7. report tables, comparison figure, map and PDF (7 pages)
    .venv/bin/python scripts/make_report_qa.py
    .venv/bin/python scripts/make_qa_figures.py
    PYTHONPATH=src .venv/bin/python scripts/make_report_map.py
    .venv/bin/python scripts/capture_api_evidence.py   # with the API running
    # The report PDF itself is a separate deliverable; its LaTeX sources travel
    # with the report, not with this code repository.

Random seeds are fixed (config.RANDOM_SEED = 42). The evaluator-testing partition is never read.

## Figures and screenshots

The report uses seven figures, all from actual results: three EDA/geographical figures
(scripts/eda_analysis.py, scripts/make_maps.py), one model comparison figure
(scripts/make_qa_figures.py), one epoch-wise MLP loss curve, one Folium map
(scripts/make_report_map.py) and one UI + API evidence pair
(scripts/make_ui_figure.py, scripts/capture_api_evidence.py). The map and UI PNGs are
generated by the committed scripts, so the PDF rebuilds without a browser.

## Two-laptop deployment

Laptop 1 (model + API):  bash scripts/run_api.sh          # uvicorn on 0.0.0.0:8000
Laptop 2 (UI):           bash scripts/run_ui.sh           # static UI on :8501
Open http://localhost:8501, set the API URL to http://<laptop1-ip>:8000. The header model picker
switches between the eight deployed models; a compact table compares their recorded scores and the
map/metrics panel updates live through /evaluate/{trip}?model=<id>. Responses contain predictions,
road-network context, server-side metrics and timings - never ground-truth geometry.

## Kaggle usage

Two private datasets (raw data + code) back the Kaggle CPU kernel in kaggle/experiments/, which
rebuilds the candidate dataset, trains a PyTorch MLP candidate classifier (5-fold group CV) and runs a
40-trial Optuna TPE search for LightGBM. Outputs are committed in artifacts/results/ (kaggle_*.json,
kaggle_*_trials.csv, kaggle_mlp_history.csv, prediction caches) and artifacts/kaggle_results/.
The local fallback performs the same search space and was used for the final artifact.

## Notes

- Ground-truth routes can be partial; fixes farther than 75 m from the reference route are treated
  as unlabelled (98.2% of fixes are labelled on average).
- Candidate miss rate is about 0.75% - that is the ceiling on observation-level accuracy.
- No hardcoded predictions, no evaluator labels, no secrets in the repository.
- AI assistance was used during development and writing; all reported experiments were actually run
  with this code (see the report's disclosure section).
- Team workflow: parts were authored separately and consolidated into this repository at the end of
  the day; commit authors record who wrote each part, and all pushes were made by one member.
