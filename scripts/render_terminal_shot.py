"""Render captured terminal output (curl evidence) as a terminal-style PNG.

The backend writes the real API responses to reports/figures/ui/api_curl.txt;
this script turns that capture into an image for the PDF report without any
manual editing of the content.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "reports" / "figures" / "ui"


def render(src: Path, dst: Path, title: str, width: int = 118, max_lines: int = 62):
    lines = src.read_text(errors="replace").splitlines()
    wrapped = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(line)
        else:
            wrapped.extend(textwrap.wrap(line, width=width) or [""])
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines] + ["... (truncated in the PDF; full capture in api_curl.txt)"]
    n = len(wrapped)
    fig = plt.figure(figsize=(11.5, 8.4), dpi=150)
    fig.patch.set_facecolor("#101418")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#101418")
    ax.axis("off")
    # window chrome
    ax.add_patch(plt.Rectangle((0, 0.955), 1, 0.045, color="#1d2329", transform=ax.transAxes))
    for i, color in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        ax.add_patch(plt.Circle((0.014 + i * 0.018, 0.977), 0.006, color=color,
                                transform=ax.transAxes))
    ax.text(0.5, 0.977, title, color="#9aa7b2", ha="center", va="center",
            fontsize=9, family="monospace", transform=ax.transAxes)
    y = 0.935
    step = 0.9 / max_lines
    for line in wrapped:
        color = "#e6edf3"
        if line.startswith("$"):
            color = "#7ee787"
        elif line.startswith("HTTP") or line.startswith("=="):
            color = "#79c0ff"
        ax.text(0.012, y, line, color=color, fontsize=6.1, family="monospace",
                va="top", transform=ax.transAxes)
        y -= step
    fig.savefig(dst, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", dst)


def main():
    src = UI / "api_curl.txt"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    render(src, UI / "api_curl.png", "laptop 1 terminal - PathSnap API (curl)")
    # health/trips-only quick capture, if available
    alt = UI / "api_health_trips.txt"
    if alt.exists():
        render(alt, UI / "api_health_trips.png", "laptop 1 terminal - /health and /trips")


if __name__ == "__main__":
    main()
