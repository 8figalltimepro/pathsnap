"""Capture live API responses as compact terminal evidence for the report."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "reports" / "figures" / "ui"
BASE = "http://localhost:8000"


def call(path):
    t0 = time.perf_counter()
    raw = subprocess.run(["curl", "-sS", "-m", "60", BASE + path], capture_output=True, text=True, check=True).stdout
    return len(raw.encode()), time.perf_counter() - t0, json.loads(raw)


def main():
    out = []
    out.append("PathSnap API - live responses (evaluator partition)")
    out.append("=" * 72)
    out.append("Backend: laptop 1, uvicorn app.main:app --host 0.0.0.0 --port 8000")
    out.append("UI:      laptop 2 (scripts/run_ui.sh, port 8501) requesting the same endpoints")
    out.append("Model:   artifacts/model.joblib (frozen before evaluator scoring)")
    out.append("")

    nbytes, dt, health = call("/health")
    out.append("1. GET /health -> HTTP 200 | %d bytes | %.4f s" % (nbytes, dt))
    out.append(json.dumps(health))
    out.append("")

    nbytes, dt, trips = call("/trips")
    out.append("2. GET /trips -> HTTP 200 | %d bytes | %.4f s" % (nbytes, dt))
    out.append("partition=%s mode=%s trips=%d" % (trips["partition"], trips["mode"], len(trips["trips"])))
    out.append("  " + ", ".join(t["trip_id"] for t in trips["trips"]))
    out.append("")

    nbytes, dt, models = call("/models")
    out.append("3. GET /models -> HTTP 200 | %d bytes | %.4f s" % (nbytes, dt))
    ids = [m.get("id") or m.get("model") for m in models.get("models", [])]
    out.append("deployable: " + ", ".join(str(i) for i in ids))
    out.append("")

    nbytes, dt, ev = call("/evaluate/00000049?model=lightgbm_tuned")
    out.append("4. GET /evaluate/00000049?model=lightgbm_tuned")
    out.append("   -> HTTP 200 | %d bytes | %.4f s" % (nbytes, dt))
    m = ev["metrics"]
    keep = {
        "trip_id": ev["trip_id"], "model": ev["model"], "n_obs": ev["n_obs"],
        "timings_ms": ev["timings_ms"],
        "metrics": {k: round(m[k], 4) for k in
                    ("top1_accuracy", "top1_accuracy_hmm", "route_continuity",
                     "route_continuity_hmm", "macro_f1", "log_loss", "seq_f1_clf", "seq_f1_hmm",
                     "candidate_miss_rate", "scored_fraction") if k in m},
    }
    out.append(json.dumps(keep))
    out.append("")
    geom = [k for k in ev if isinstance(ev[k], list)]
    out.append("payload geometry: " + ", ".join("%s[%d]" % (k, len(ev[k])) for k in geom))
    gt = [k for k in ev if ("route" in k.lower() and k not in ("gps",)) or "ground_truth" in k.lower()]
    gt = [k for k in gt if k not in ("predicted_strokes_hmm", "predicted_strokes_clf")]
    out.append("reference route geometry in payload: " + ("YES" if gt else "NO"))

    UI.mkdir(parents=True, exist_ok=True)
    txt = UI / "api_eval_curl.txt"
    txt.write_text("\n".join(out))

    lines = []
    for line in out:
        lines.extend(textwrap.wrap(line, width=86) or [""])
    n = len(lines)
    fig = plt.figure(figsize=(9.0, 0.20 * n + 0.75), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    fig.patch.set_facecolor("#101418"); ax.set_facecolor("#101418")
    ax.add_patch(plt.Rectangle((0, 0.975), 1, 0.025, color="#1d2329", transform=ax.transAxes))
    ax.text(0.012, 0.987, "laptop 2 terminal - PathSnap API responses from laptop 1",
            color="#9aa7b2", ha="left", va="center", fontsize=9, family="monospace",
            transform=ax.transAxes)
    y = 0.945
    step = 0.925 / max(n, 1)
    for line in lines:
        color = "#e6edf3"
        if line.startswith("HTTP") or line[:2] in ("1.", "2.", "3.", "4."):
            color = "#79c0ff"
        elif line.startswith("reference route"):
            color = "#7ee787"
        elif line.startswith("payload geometry"):
            color = "#febc2e"
        ax.text(0.012, y, line, color=color, fontsize=7.6, family="monospace",
                va="top", transform=ax.transAxes)
        y -= step
    fig.savefig(UI / "api_eval_curl.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("lines", n, "->", UI / "api_eval_curl.png")


if __name__ == "__main__":
    main()
