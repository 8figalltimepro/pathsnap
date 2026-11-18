"""Project-wide configuration constants and paths.

Everything that controls the pipeline lives here so the API, the experiment
scripts and the report can stay in sync.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_DIR", PROJECT_ROOT / "Student folder"))
ARTIFACTS_DIR = Path(os.environ.get("PATHSNAP_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))

TRAIN_PARTITION = "train"
DEV_PARTITION = "student_test"
# The evaluator partition is dropped in at evaluation time.  The API looks for
# this folder first and falls back to the development partition for local demos.
EVAL_PARTITION = os.environ.get("PATHSNAP_EVAL_PARTITION", "evaluator_test")
# The evaluator trips are dropped in next to the repository (not inside the course
# data folder), so they get their own configurable root.
EVAL_DATA_ROOT = Path(os.environ.get("PATHSNAP_EVAL_DATA_DIR", PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
CANDIDATE_RADIUS_M = 50.0       # nominal search radius around each GPS fix
CANDIDATE_FALLBACK_RADII_M = (150.0, 400.0)
MAX_CANDIDATES = 16             # candidates kept per observation (closest first)
CANDIDATE_CELL_M = 100.0        # spatial hash cell size

# ---------------------------------------------------------------------------
# Labelling: project GPS fixes on the reference route polyline
# ---------------------------------------------------------------------------
LABEL_MAX_DIST_M = 75.0         # fixes farther than this are treated as unlabelled
LABEL_SIGMA_M = 15.0

# ---------------------------------------------------------------------------
# HMM / Viterbi
# ---------------------------------------------------------------------------
HMM_MAX_STATES = 8              # top candidate states kept by the decoder
HMM_BETA_M = 20.0               # transition scale (metres)
HMM_EMISSION_ALPHA = 1.0        # sharpening of classifier probabilities
HMM_GAP_FREE_S = 30.0           # time gaps longer than this relax connectivity
HMM_MAX_SEARCH_M = 3000.0       # cap for bounded shortest-path search

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
