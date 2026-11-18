"""Fill missing candidate precision/recall aggregates in model_comparison.json
from the saved student-test probability caches (the HMM metrics are unaffected)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import config
from pathsnap.candidates import CandidateSet
from pathsnap.metrics import candidate_probability_metrics
from pathsnap.io import list_trips

CACHE = config.ARTIFACTS_DIR / "cache"
PREDS = CACHE / "preds"
RESULTS = config.ARTIFACTS_DIR / "results"


def normalised(cand, probs):
    out = np.empty(len(probs))
    counts = np.bincount(cand.obs_idx, minlength=int(cand.obs_idx.max()) + 1)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for s, c in zip(starts, counts):
        if c:
            seg = probs[s:s + c].astype(np.float64)
            tot = seg.sum()
            out[s:s + c] = seg / tot if tot > 0 else 1.0 / c
    return out


def main():
    comp = json.loads((RESULTS / "model_comparison.json").read_text())
    for name, rec in comp["models"].items():
        if name == "mlp_deep":
            continue
        precs, recs = [], []
        ok = True
        for tid in list_trips("student_test"):
            p = PREDS / f"{name}_{tid}.npz"
            if not p.exists():
                ok = False
                break
            z = np.load(CACHE / f"student_test_{tid}.npz")
            cand = CandidateSet(obs_idx=z["obs_idx"].astype(np.int64), arc=z["arc"].astype(np.int64),
                                dist=z["dist"].astype(np.float64), t=z["t"].astype(np.float64),
                                foot_x=np.zeros(len(z["arc"])), foot_y=np.zeros(len(z["arc"])),
                                rank=z["rank"].astype(np.int64), n_cand=z["n_cand"].astype(np.int64))
            probs = normalised(cand, np.load(p)["probs"].astype(np.float64))
            m = candidate_probability_metrics(cand, probs, z["label_arc"], z["label_valid"].astype(bool))
            precs.append(m["precision"]); recs.append(m["recall"])
        if ok and precs:
            rec["student_test"]["precision"] = float(np.mean(precs))
            rec["student_test"]["recall"] = float(np.mean(recs))
            print(name, "precision", round(float(np.mean(precs)), 4),
                  "recall", round(float(np.mean(recs)), 4))
    (RESULTS / "model_comparison.json").write_text(json.dumps(comp, indent=2))


if __name__ == "__main__":
    main()
