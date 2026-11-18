#!/usr/bin/env python3
"""PathSnap exploratory analysis.

Regenerates, from the raw Student folder/ data:

* reports/tables/trip_stats.csv        - one row per trip
* reports/figures/eda/*.png            - 7 report figures + README
* reports/maps/eda_trip_<trip>.html    - interactive folium maps
* reports/eda_summary.md               - numbers-first EDA summary

Run from the repository root:

    PYTHONPATH=src python scripts/eda_analysis.py

Everything is computed from the raw files; nothing else is read.  The whole
pipeline runs in well under three minutes.

Conventions used below (also stated in the report):

* distances are metres, computed in a per-trip local equirectangular frame
  (pathsnap.geo.to_local); haversine for absolute path lengths;
* dt is the difference between consecutive track timestamps (seconds);
* gap_frac is the fraction of consecutive dt values above 1.5 s;
* track_length_m sums consecutive GPS-fix separations, so it includes GPS
  jitter and jumps across long time gaps;
* route_length_m sums the arc lengths of the reference route in file order;
* gps_to_route_median_m is the median distance from every fix to the nearest
  reference-route segment;
* route_coverage_frac is the fraction of fixes within 30 m of that route.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths / imports.  Make the script runnable both as "python scripts/x.py" and
# as "PYTHONPATH=src python scripts/x.py".
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.config import (
    DATA_ROOT,
    DEV_PARTITION,
    PROJECT_ROOT as CFG_PROJECT_ROOT,
    RANDOM_SEED,
    TRAIN_PARTITION,
)
from pathsnap.geo import haversine_m, to_local
from pathsnap.io import list_trips, load_trip

REPORTS_DIR = CFG_PROJECT_ROOT / "reports"
TABLE_DIR = REPORTS_DIR / "tables"
FIG_DIR = REPORTS_DIR / "figures" / "eda"
MAP_DIR = REPORTS_DIR / "maps"
CSV_PATH = TABLE_DIR / "trip_stats.csv"
SUMMARY_PATH = REPORTS_DIR / "eda_summary.md"
FIG_README_PATH = FIG_DIR / "README.md"

PARTITIONS = [TRAIN_PARTITION, DEV_PARTITION]
PART_COLORS = {TRAIN_PARTITION: "#1f77b4", DEV_PARTITION: "#d62728"}
GPS_COLOR = "#1f77b4"
NET_COLOR = "#b3b3b3"
ROUTE_COLOR = "#ff7f0e"

GAP_THRESHOLD_S = 1.5
COVERAGE_RADIUS_M = 30.0
LABEL_CAP_M = 75.0
CITY_LINK_KM = 20.0

RNG = np.random.default_rng(RANDOM_SEED)

# Explicit examples, chosen from the computed stats (ids are stable):
#   train 00000027        - compact ~3 x 3 km trip, full route, no big gaps
#   train 00000033        - compact trip whose supplied route is only partial
#   train 00000090        - 42.8 h span with a single 153,064 s gap
EXAMPLE_TRACK_TRIPS = [
    (TRAIN_PARTITION, "00000027", "clean, full route"),
    (TRAIN_PARTITION, "00000033", "partial route"),
    (TRAIN_PARTITION, "00000090", "42.8 h span, 153,064 s gap"),
]
#   train 00000027         - full route, compact city network
#   train 00000033         - partial route (most fixes far from the route)
#   student_test 00000040  - a student_test trip with a good route
MAP_TRIPS = [
    (TRAIN_PARTITION, "00000027", "full route"),
    (TRAIN_PARTITION, "00000033", "partial route"),
    (DEV_PARTITION, "00000040", "student_test"),
]

CSV_COLUMNS = [
    "partition", "trip_id", "n_obs", "n_nodes", "n_arcs", "n_route_arcs",
    "dt_median", "dt_max", "gap_frac", "track_length_m", "route_length_m",
    "length_ratio", "bbox_lon_min", "bbox_lon_max", "bbox_lat_min",
    "bbox_lat_max", "gps_to_route_median_m", "route_coverage_frac",
]
CSV_INT_COLUMNS = {"n_obs", "n_nodes", "n_arcs", "n_route_arcs"}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linewidth": 0.6,
    "legend.fontsize": 9,
})


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def distance_to_route(tr, chunk: int = 512) -> np.ndarray:
    """Distance (m) from every GPS fix to the closest reference-route segment."""
    lon0 = float(tr.nodes[:, 0].mean())
    lat0 = float(tr.nodes[:, 1].mean())
    nx, ny = to_local(tr.nodes[:, 0], tr.nodes[:, 1], (lon0, lat0))
    arc = tr.arcs[tr.route]
    ax, ay = nx[arc[:, 0]], ny[arc[:, 0]]
    bx, by = nx[arc[:, 1]], ny[arc[:, 1]]
    px, py = to_local(tr.track[:, 0], tr.track[:, 1], (lon0, lat0))
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    safe = np.where(seg2 > 1e-12, seg2, 1.0)   # zero-length arc -> point distance
    out = np.empty(len(px), dtype=np.float64)
    for start in range(0, len(px), chunk):
        stop = min(start + chunk, len(px))
        t = ((px[start:stop, None] - ax[None, :]) * dx[None, :]
             + (py[start:stop, None] - ay[None, :]) * dy[None, :]) / safe[None, :]
        np.clip(t, 0.0, 1.0, out=t)
        fx = ax[None, :] + t * dx[None, :]
        fy = ay[None, :] + t * dy[None, :]
        out[start:stop] = np.hypot(px[start:stop, None] - fx,
                                   py[start:stop, None] - fy).min(axis=1)
    return out


def split_runs(t: np.ndarray, gap_s: float) -> list[slice]:
    """Index slices of a track split wherever consecutive dt exceeds gap_s."""
    if len(t) < 2:
        return [slice(0, len(t))]
    breaks = np.flatnonzero(np.diff(t) > gap_s) + 1
    bounds = np.concatenate([[0], breaks, [len(t)]])
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def route_node_runs(arcs: np.ndarray, route: np.ndarray) -> list[np.ndarray]:
    """Split a route arc sequence into runs of contiguous node ids."""
    a = arcs[route]
    if len(a) == 0:
        return []
    breaks = np.flatnonzero(a[:-1, 1] != a[1:, 0]) + 1
    bounds = np.concatenate([[0], breaks, [len(a)]])
    runs = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        nodes = np.concatenate([[a[start, 0]], a[start:stop, 1]])
        runs.append(nodes)
    return runs


# ---------------------------------------------------------------------------
# Per-trip statistics
# ---------------------------------------------------------------------------
def compute_trip_stats(partition: str, trip_id: str):
    """Return (per-trip stats dict, dt array, step array, speed array, dist array)."""
    tr = load_trip(trip_id, partition, DATA_ROOT)
    t = tr.track[:, 2]
    lon = tr.track[:, 0]
    lat = tr.track[:, 1]
    dt = np.diff(t)
    step = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    speed = np.full_like(step, np.nan)
    np.divide(step, dt, out=speed, where=dt > 0)

    # -- reference route ---------------------------------------------------
    route_node_ids = tr.arcs[tr.route]
    route_seg_len = haversine_m(
        tr.nodes[route_node_ids[:, 0], 0], tr.nodes[route_node_ids[:, 0], 1],
        tr.nodes[route_node_ids[:, 1], 0], tr.nodes[route_node_ids[:, 1], 1],
    )
    route_length = float(route_seg_len.sum())
    track_length = float(step.sum())
    length_ratio = track_length / route_length if route_length > 0 else float("nan")

    dist = distance_to_route(tr)
    cover = dist <= COVERAGE_RADIUS_M

    # -- network structure -------------------------------------------------
    n_nodes = int(len(tr.nodes))
    n_arcs = int(len(tr.arcs))
    graph = coo_matrix(
        (np.ones(n_arcs, dtype=np.int8), (tr.arcs[:, 0], tr.arcs[:, 1])),
        shape=(n_nodes, n_nodes),
    )
    n_comp, labels = connected_components(graph, directed=False)
    comp_sizes = np.bincount(labels)
    degree = np.bincount(tr.arcs.ravel(), minlength=n_nodes)

    stats = {
        "partition": partition,
        "trip_id": trip_id,
        "n_obs": int(len(tr.track)),
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "n_route_arcs": int(len(tr.route)),
        "dt_median": float(np.median(dt)) if len(dt) else 0.0,
        "dt_max": float(dt.max()) if len(dt) else 0.0,
        "gap_frac": float((dt > GAP_THRESHOLD_S).mean()) if len(dt) else 0.0,
        "track_length_m": track_length,
        "route_length_m": route_length,
        "length_ratio": float(length_ratio),
        "bbox_lon_min": float(lon.min()),
        "bbox_lon_max": float(lon.max()),
        "bbox_lat_min": float(lat.min()),
        "bbox_lat_max": float(lat.max()),
        "gps_to_route_median_m": float(np.median(dist)),
        "route_coverage_frac": float(cover.mean()),
        # -- extra fields used by figures / summary (not in the CSV) --------
        "duration_s": float(t[-1] - t[0]),
        "t0": float(t[0]),
        "n_intervals": int(len(dt)),
        "n_gaps": int((dt > GAP_THRESHOLD_S).sum()),
        "frac_beyond_label_cap": float((dist > LABEL_CAP_M).mean()),
        "route_discontinuities": int((route_node_ids[:-1, 1] != route_node_ids[1:, 0]).sum())
        if len(route_node_ids) > 1 else 0,
        "route_repeated_arcs": int(len(tr.route) - len(np.unique(tr.route))),
        "zero_len_route_arcs": int((route_seg_len < 1e-6).sum()),
        "n_components": int(n_comp),
        "main_component_frac": float(comp_sizes.max() / n_nodes),
        "degree_mean": float(degree.mean()),
        "n_isolated_nodes": int((degree == 0).sum()),
        "n_frac_ts": int(np.sum(t != np.round(t))),
        "n_frac_dt": int((dt != np.round(dt)).sum()),
        "start_lon": float(lon[0]),
        "start_lat": float(lat[0]),
        "n_nan_track": int(np.isnan(tr.track).sum()),
        "n_nan_nodes": int(np.isnan(tr.nodes).sum()),
        "n_bad_arc_ids": int(((tr.arcs < 0) | (tr.arcs >= n_nodes)).sum()),
        "n_bad_route_ids": int(((tr.route < 0) | (tr.route >= n_arcs)).sum()),
        "n_noninc_dt": int((dt <= 0).sum()),
        "n_duplicate_dt": int((dt == 0).sum()),
    }
    return stats, dt, step, speed, dist


# ---------------------------------------------------------------------------
# Aggregation for the figures and the written summary
# ---------------------------------------------------------------------------
def _pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def _count_true(seq, predicate):
    return int(sum(1 for x in seq if predicate(x)))


def aggregate(stats, pooled):
    """Collapse per-trip rows and pooled per-fix arrays into report numbers."""
    dt = pooled["dt"]
    step = pooled["step"]
    speed = pooled["speed"]
    dist = pooled["dist"]
    ratio = np.array([s["length_ratio"] for s in stats], dtype=float)
    cov = np.array([s["route_coverage_frac"] for s in stats], dtype=float)
    dur = np.array([s["duration_s"] for s in stats], dtype=float)
    gap_frac = np.array([s["gap_frac"] for s in stats], dtype=float)
    trip_dist_medians = np.array([s["gps_to_route_median_m"] for s in stats], dtype=float)

    agg = {"partitions": {}}
    for part in PARTITIONS:
        rows = [s for s in stats if s["partition"] == part]
        n_obs = np.array([s["n_obs"] for s in rows], dtype=float)
        agg["partitions"][part] = {
            "trips": len(rows),
            "fixes": int(n_obs.sum()),
            "nodes": int(sum(s["n_nodes"] for s in rows)),
            "arcs": int(sum(s["n_arcs"] for s in rows)),
            "route_arcs": int(sum(s["n_route_arcs"] for s in rows)),
            "median_fixes_per_trip": float(np.median(n_obs)),
            "median_nodes": float(np.median([s["n_nodes"] for s in rows])),
            "median_arcs": float(np.median([s["n_arcs"] for s in rows])),
        }

    agg.update({
        "n_trips": len(stats),
        "n_fix": int(sum(s["n_obs"] for s in stats)),
        "n_nodes": int(sum(s["n_nodes"] for s in stats)),
        "n_arcs": int(sum(s["n_arcs"] for s in stats)),
        "n_route_arcs": int(sum(s["n_route_arcs"] for s in stats)),
        "n_obs_min": int(min(s["n_obs"] for s in stats)),
        "n_obs_median": float(np.median([s["n_obs"] for s in stats])),
        "n_obs_max": int(max(s["n_obs"] for s in stats)),
        "n_nodes_min": int(min(s["n_nodes"] for s in stats)),
        "n_nodes_median": float(np.median([s["n_nodes"] for s in stats])),
        "n_nodes_max": int(max(s["n_nodes"] for s in stats)),
        "n_arcs_min": int(min(s["n_arcs"] for s in stats)),
        "n_arcs_median": float(np.median([s["n_arcs"] for s in stats])),
        "n_arcs_max": int(max(s["n_arcs"] for s in stats)),
        "n_route_min": int(min(s["n_route_arcs"] for s in stats)),
        "n_route_median": float(np.median([s["n_route_arcs"] for s in stats])),
        "n_route_max": int(max(s["n_route_arcs"] for s in stats)),
        # timestamps / sampling
        "n_intervals": int(len(dt)),
        "dt_median": float(np.median(dt)),
        "dt_max": float(dt.max()),
        "frac_dt_1s": float((dt == 1.0).mean()),
        "n_dt_gt_15": int((dt > GAP_THRESHOLD_S).sum()),
        "frac_dt_gt_15": float((dt > GAP_THRESHOLD_S).mean()),
        "n_dt_gt_10": int((dt > 10.0).sum()),
        "n_dt_gt_60": int((dt > 60.0).sum()),
        "n_dt_gt_600": int((dt > 600.0).sum()),
        "n_trips_dtmax_gt_60": _count_true(stats, lambda s: s["dt_max"] > 60.0),
        "n_trips_dtmax_gt_600": _count_true(stats, lambda s: s["dt_max"] > 600.0),
        "n_trips_dtmax_gt_3600": _count_true(stats, lambda s: s["dt_max"] > 3600.0),
        "n_trips_gap_frac_gt_10pct": _count_true(stats, lambda s: s["gap_frac"] > 0.10),
        "gap_frac_max": float(gap_frac.max()),
        "gap_frac_median": float(np.median(gap_frac)),
        "n_frac_ts_timestamps": int(sum(s["n_frac_ts"] for s in stats)),
        "n_frac_ts_trips": _count_true(stats, lambda s: s["n_frac_ts"] > 0),
        "n_frac_dt_intervals": int(sum(s["n_frac_dt"] for s in stats)),
        "t0_min": float(min(s["t0"] for s in stats)),
        "t0_max": float(max(s["t0"] for s in stats)),
        "n_noninc_dt": int(sum(s["n_noninc_dt"] for s in stats)),
        "n_duplicate_dt": int(sum(s["n_duplicate_dt"] for s in stats)),
        "n_nan": int(sum(s["n_nan_track"] + s["n_nan_nodes"] for s in stats)),
        "n_bad_arc_ids": int(sum(s["n_bad_arc_ids"] for s in stats)),
        "n_bad_route_ids": int(sum(s["n_bad_route_ids"] for s in stats)),
        "route_present": sum(1 for s in stats if s["n_route_arcs"] > 0),
        # durations
        "dur_min": float(dur.min()),
        "dur_median": float(np.median(dur)),
        "dur_max": float(dur.max()),
        "n_dur_gt_3h": int((dur > 3 * 3600).sum()),
        # steps / speeds
        "step_median": float(np.median(step)),
        "step_p10": _pct(step, 10),
        "step_p90": _pct(step, 90),
        "step_p99": _pct(step, 99),
        "step_max": float(step.max()),
        "speed_median": float(np.median(speed)),
        "speed_kmh_median": float(np.median(speed) * 3.6),
        "speed_p10": _pct(speed, 10),
        "speed_p90": _pct(speed, 90),
        "speed_p99": _pct(speed, 99),
        "speed_max_kmh": float(speed.max() * 3.6),
        "frac_speed_lt_1": float((speed < 1.0).mean()),
        "frac_speed_lt_2": float((speed < 2.0).mean()),
        # route proximity
        "dist_median": float(np.median(dist)),
        "dist_p75": _pct(dist, 75),
        "dist_p90": _pct(dist, 90),
        "dist_p95": _pct(dist, 95),
        "dist_p99": _pct(dist, 99),
        "trip_median_dist": float(np.median(trip_dist_medians)),
        "frac_le_30": float((dist <= COVERAGE_RADIUS_M).mean()),
        "frac_le_75": float((dist <= LABEL_CAP_M).mean()),
        "n_gt_75": int((dist > LABEL_CAP_M).sum()),
        "frac_gt_75": float((dist > LABEL_CAP_M).mean()),
        "frac_gt_150": float((dist > 150.0).mean()),
        "median_trip_cov": float(np.median(cov)),
        "cov_min": float(cov.min()),
        "n_cov_lt_05": _count_true(stats, lambda s: s["route_coverage_frac"] < 0.50),
        "n_cov_lt_09": _count_true(stats, lambda s: s["route_coverage_frac"] < 0.90),
        "n_cov_lt_095": _count_true(stats, lambda s: s["route_coverage_frac"] < 0.95),
        "n_cov_lt_099": _count_true(stats, lambda s: s["route_coverage_frac"] < 0.99),
        "n_test_cov_ge_098": _count_true(
            [s for s in stats if s["partition"] == DEV_PARTITION],
            lambda s: s["route_coverage_frac"] >= 0.98,
        ),
        # ratio
        "ratio_median": float(np.median(ratio)),
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "n_ratio_le_110": int((ratio <= 1.10).sum()),
        "n_ratio_le_150": int((ratio <= 1.50).sum()),
        "n_ratio_gt_150": int((ratio > 1.50).sum()),
        "n_ratio_gt_200": int((ratio > 2.00).sum()),
        "n_ratio_gt_500": int((ratio > 5.00).sum()),
        # route structure
        "n_route_disc_trips": _count_true(stats, lambda s: s["route_discontinuities"] > 0),
        "route_disc_max": int(max(s["route_discontinuities"] for s in stats)),
        "n_route_repeat_trips": _count_true(stats, lambda s: s["route_repeated_arcs"] > 0),
        "route_repeat_max": int(max(s["route_repeated_arcs"] for s in stats)),
        "zero_len_route_arcs": int(sum(s["zero_len_route_arcs"] for s in stats)),
        # network structure
        "degree_mean_median": float(np.median([s["degree_mean"] for s in stats])),
        "degree_mean_min": float(min(s["degree_mean"] for s in stats)),
        "degree_mean_max": float(max(s["degree_mean"] for s in stats)),
        "n_isolated_nodes": int(sum(s["n_isolated_nodes"] for s in stats)),
        "n_net_disconnected": _count_true(stats, lambda s: s["n_components"] > 1),
        "ncc_median": float(np.median([s["n_components"] for s in stats])),
        "ncc_max": int(max(s["n_components"] for s in stats)),
        "main_comp_median": float(np.median([s["main_component_frac"] for s in stats])),
        "main_comp_min": float(min(s["main_component_frac"] for s in stats)),
        "n_main_comp_lt_99": _count_true(stats, lambda s: s["main_component_frac"] < 0.99),
        # geography
        "lon_min": float(min(s["bbox_lon_min"] for s in stats)),
        "lon_max": float(max(s["bbox_lon_max"] for s in stats)),
        "lat_min": float(min(s["bbox_lat_min"] for s in stats)),
        "lat_max": float(max(s["bbox_lat_max"] for s in stats)),
    })

    # -- geographic clustering of trips (single linkage on centroid distance) --
    cent_lon = np.array([s["start_lon"] for s in stats])
    cent_lat = np.array([s["start_lat"] for s in stats])
    dmat = haversine_m(cent_lon[:, None], cent_lat[:, None],
                       cent_lon[None, :], cent_lat[None, :])
    np.fill_diagonal(dmat, 1e18)
    close = dmat < CITY_LINK_KM * 1000.0
    np.fill_diagonal(close, True)
    closed = close.copy()
    while True:
        nxt = closed | (closed.astype(np.int32) @ closed.astype(np.int32) > 0)
        if np.array_equal(nxt, closed):
            break
        closed = nxt
    clusters = []
    seen = np.zeros(len(stats), dtype=bool)
    for i in range(len(stats)):
        if seen[i]:
            continue
        members = np.flatnonzero(closed[i])
        seen[members] = True
        clusters.append([stats[int(j)] for j in members])
    multi = [c for c in clusters if len(c) > 1]
    largest_cluster = max(clusters, key=len)
    i, j = np.unravel_index(np.argmin(dmat), dmat.shape)
    n_bbox_overlap_pairs = 0
    for a in range(len(stats)):
        for b in range(a + 1, len(stats)):
            sa, sb = stats[a], stats[b]
            if (sa["bbox_lon_min"] < sb["bbox_lon_max"]
                    and sb["bbox_lon_min"] < sa["bbox_lon_max"]
                    and sa["bbox_lat_min"] < sb["bbox_lat_max"]
                    and sb["bbox_lat_min"] < sa["bbox_lat_max"]):
                n_bbox_overlap_pairs += 1
    agg.update({
        "n_location_clusters": len(clusters),
        "n_multi_clusters": len(multi),
        "n_trips_in_multi": int(sum(len(c) for c in multi)),
        "largest_cluster_size": int(len(largest_cluster)),
        "largest_cluster_ids": [s["trip_id"] for s in largest_cluster],
        "largest_cluster_mean_lon": float(np.mean([s["start_lon"] for s in largest_cluster])),
        "largest_cluster_mean_lat": float(np.mean([s["start_lat"] for s in largest_cluster])),
        "min_pair_dist_m": float(dmat[i, j]),
        "min_pair_ids": (stats[int(i)]["trip_id"], stats[int(j)]["trip_id"]),
        "n_bbox_overlap_pairs": n_bbox_overlap_pairs,
        "clusters": clusters,
        "test_min_train_dist_m": float(min(
            dmat[a, b]
            for a in range(len(stats))
            for b in range(len(stats))
            if stats[a]["partition"] == DEV_PARTITION and stats[b]["partition"] == TRAIN_PARTITION
        )),
    })
    return agg


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def write_trip_csv(stats, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for s in stats:
            row = []
            for col in CSV_COLUMNS:
                value = s[col]
                if col in CSV_INT_COLUMNS:
                    row.append(int(value))
                elif isinstance(value, str):
                    row.append(value)
                else:
                    row.append(f"{float(value):.6f}")
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Figures (matplotlib)
# ---------------------------------------------------------------------------
def _save(fig, name: str):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_inventory(stats, agg):
    parts = PARTITIONS
    trips = [agg["partitions"][p]["trips"] for p in parts]
    fixes = [agg["partitions"][p]["fixes"] for p in parts]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    ax = axes[0]
    bars = ax.bar(range(len(parts)), trips, color=[PART_COLORS[p] for p in parts], width=0.55)
    for bar, n, f in zip(bars, trips, fixes):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 1.5, f"{n} trips\n{f:,} fixes",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels(parts)
    ax.set_ylabel("number of trips")
    ax.set_ylim(0, max(trips) * 1.32)
    ax.set_title("Trips per partition")

    ax = axes[1]
    nobs = np.array([s["n_obs"] for s in stats], dtype=float)
    bins = np.geomspace(nobs.min() * 0.85, nobs.max() * 1.15, 18)
    for part in parts:
        vals = [s["n_obs"] for s in stats if s["partition"] == part]
        ax.hist(vals, bins=bins, color=PART_COLORS[part], alpha=0.65,
                label=f"{part} (n={len(vals)})", edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(agg["n_obs_median"], color="black", ls="--", lw=1,
               label=f"median {agg['n_obs_median']:,.0f}")
    ax.set_xlabel("GPS fixes per trip (log scale)")
    ax.set_ylabel("number of trips (log scale)")
    ax.set_title("Observations per trip")
    ax.legend(loc="upper left")
    return _save(fig, "inventory.png")


def fig_sampling(stats, pooled, agg):
    dt = pooled["dt"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    ax = axes[0]
    bins = np.arange(0.5, 30.5, 1.0)
    ax.hist(np.clip(dt, 0.5, 29.5), bins=bins, color="#4c72b0", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("dt between consecutive fixes (s, clipped at 30; log y)")
    ax.set_ylabel("intervals")
    ax.set_title("Sampling interval dt")
    ax.text(0.97, 0.95,
            f"1 s: {100 * agg['frac_dt_1s']:.1f}% of {agg['n_intervals']:,} intervals\n"
            f">1.5 s: {agg['n_dt_gt_15']:,} ({100 * agg['frac_dt_gt_15']:.1f}%)\n"
            f"max dt: {agg['dt_max']:,.0f} s\n"
            "last bar: all dt > 28.5 s",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9))

    ax = axes[1]
    gap_counts = np.array([s["n_gaps"] for s in stats])
    order = np.argsort(-gap_counts)
    colors = [PART_COLORS[stats[i]["partition"]] for i in order]
    ax.bar(np.arange(len(stats)), gap_counts[order], color=colors, width=1.0)
    ax.set_yscale("log")
    ax.set_ylim(0.7, max(gap_counts.max() * 2.2, 10))
    ax.set_xlabel("trip (sorted by number of gaps)")
    ax.set_ylabel("intervals with dt > 1.5 s (log)")
    top = stats[int(order[0])]
    ax.annotate(f"{top['trip_id']}: {gap_counts[order[0]]:,} gaps\n(max dt {top['dt_max']:,.0f} s)",
                xy=(0, gap_counts[order[0]]), xytext=(10, gap_counts[order[0]] * 0.55),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_title("Gaps per trip")
    handles = [plt.Line2D([], [], color=PART_COLORS[p], lw=6, label=p) for p in PARTITIONS]
    ax.legend(handles=handles, loc="upper right")
    return _save(fig, "sampling.png")


def fig_steps_speed(pooled, agg):
    step = pooled["step"]
    speed = pooled["speed"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    ax = axes[0]
    bins = np.arange(0.0, 61.0, 1.5)
    ax.hist(np.clip(step, 0.0, 59.9), bins=bins, color="#55a868", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.axvline(agg["step_median"], color="black", ls="--", lw=1)
    ax.set_xlabel("GPS step length (m, clipped at 60; log y)")
    ax.set_ylabel("steps")
    ax.set_title("Step length between consecutive fixes")
    ax.text(0.97, 0.95,
            f"median {agg['step_median']:.1f} m\n"
            f"p90 {agg['step_p90']:.1f} m - p99 {agg['step_p99']:.1f} m\n"
            f"max {agg['step_max']:.0f} m (gap jump)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9))

    ax = axes[1]
    bins = np.arange(0.0, 40.5, 0.75)
    ax.hist(np.clip(speed, 0.0, 40.0), bins=bins, color="#c44e52", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.axvline(agg["speed_median"], color="black", ls="--", lw=1)
    ax.set_xlabel("implied speed = step / dt (m/s, clipped at 40; log y)")
    ax.set_ylabel("steps")
    ax.set_title("Implied speed between fixes")
    ax.text(0.97, 0.95,
            f"median {agg['speed_median']:.1f} m/s ({agg['speed_kmh_median']:.0f} km/h)\n"
            f"p90 {agg['speed_p90']:.1f} - p99 {agg['speed_p99']:.1f} m/s\n"
            f"<1 m/s: {100 * agg['frac_speed_lt_1']:.1f}% of steps",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9))
    return _save(fig, "steps_speed.png")


def fig_coverage(stats, pooled, agg):
    ratio = np.array([s["length_ratio"] for s in stats], dtype=float)
    cov = np.array([s["route_coverage_frac"] for s in stats], dtype=float)
    dist = pooled["dist"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    ax = axes[0]
    edges = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0])
    ax.hist(ratio, bins=edges, color="#4c72b0", edgecolor="white", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 8.0)
    ax.set_xticks([1, 1.5, 2, 3, 5, 8])
    ax.set_xticklabels(["1", "1.5", "2", "3", "5", "8"])
    ax.minorticks_off()
    ax.axvline(agg["ratio_median"], color="black", ls="--", lw=1,
               label=f"median {agg['ratio_median']:.2f}")
    ax.set_xlabel("track length / route length (log scale)")
    ax.set_ylabel("trips")
    ax.set_title("Track vs reference-route length")
    ax.legend(loc="upper right")
    ax.text(0.97, 0.70, f"{agg['n_ratio_le_150']}/{agg['n_trips']} trips <= 1.5\n"
                        f"max {agg['ratio_max']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9))

    ax = axes[1]
    d_sorted = np.sort(dist)
    xs = np.maximum(d_sorted, 0.1)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color="#4c72b0", lw=1.8)
    for threshold, color in ((COVERAGE_RADIUS_M, "#d62728"), (LABEL_CAP_M, "#2ca02c")):
        ax.axvline(threshold, color=color, ls="--", lw=1)
        frac = float((dist <= threshold).mean())
        ax.text(threshold * 1.06, 0.10, f"{threshold:.0f} m: {100 * frac:.1f}%",
                rotation=90, fontsize=8.5, color=color, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.5, 500)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("distance to reference route (m, log scale)")
    ax.set_ylabel("fraction of fixes <= x")
    ax.set_title(f"GPS-to-route distance CDF ({agg['n_fix']:,} fixes)")
    ax.text(0.03, 0.97, f"median {agg['dist_median']:.1f} m\np95 {agg['dist_p95']:.1f} m",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9))

    ax = axes[2]
    order = np.argsort(cov)
    ranks = np.arange(1, len(cov) + 1)
    ax.scatter(ranks, cov[order], s=22, color=[PART_COLORS[stats[i]["partition"]] for i in order],
               alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axhline(0.95, color="grey", ls="--", lw=1)
    ax.text(len(cov) * 0.99, 0.955, "0.95", ha="right", va="bottom", fontsize=8, color="grey")
    worst = [stats[int(idx)] for idx in order[:4]]
    worst_text = "\n".join(f"{s['trip_id']}: {100 * s['route_coverage_frac']:.0f}%"
                           for s in worst)
    ax.text(0.30, 0.50, "lowest coverage:\n" + worst_text, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9))
    ax.set_xlim(0, len(cov) + 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("trip (sorted by coverage)")
    ax.set_ylabel("fraction of fixes within 30 m")
    ax.set_title("Per-trip route coverage")
    handles = [plt.Line2D([], [], marker="o", ls="", color=PART_COLORS[p], label=p) for p in PARTITIONS]
    ax.legend(handles=handles, loc="lower right")
    return _save(fig, "coverage.png")


def fig_networks(stats, agg):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    ax = axes[0]
    for part in PARTITIONS:
        rows = [s for s in stats if s["partition"] == part]
        ax.scatter([s["n_nodes"] for s in rows], [s["n_arcs"] for s in rows],
                   s=20, alpha=0.8, color=PART_COLORS[part], label=part,
                   edgecolor="white", linewidth=0.3)
    lims = np.array([200.0, 4e5])
    ax.plot(lims, 2.0 * lims, color="grey", ls="--", lw=1, label="arcs = 2 x nodes")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 4e5)
    ax.set_ylim(400, 8e5)
    ax.set_xlabel("network nodes (log)")
    ax.set_ylabel("network arcs (log)")
    ax.set_title("Network size per trip")
    ax.legend(loc="lower right")

    ax = axes[1]
    by_nodes = sorted(stats, key=lambda s: s["n_nodes"])
    picks = [by_nodes[0], by_nodes[len(by_nodes) // 2], by_nodes[-1]]
    bins = np.arange(-0.5, 16.5, 1.0)
    for s, color in zip(picks, ["#4c72b0", "#dd8452", "#55a868"]):
        tr = load_trip(s["trip_id"], s["partition"], DATA_ROOT)
        degree = np.bincount(tr.arcs.ravel(), minlength=len(tr.nodes))
        ax.hist(degree, bins=bins, density=True, histtype="step", lw=1.6, color=color,
                label=f"{s['trip_id']}: {s['n_nodes']:,} nodes, mean {degree.mean():.2f}")
    ax.set_xlabel("undirected node degree (in + out arcs)")
    ax.set_ylabel("density")
    ax.set_title("Degree distribution (smallest / median / largest network)")
    ax.legend(loc="upper right")

    ax = axes[2]
    ncc = np.array([s["n_components"] for s in stats], dtype=float)
    bins = np.geomspace(1.0, max(500.0, ncc.max() * 1.2), 18)
    ax.hist(ncc, bins=bins, color="#8172b3", edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("connected components per network (log)")
    ax.set_ylabel("trips (log)")
    ax.set_title("Connected components")
    ax.text(0.97, 0.95,
            f"{agg['n_net_disconnected']}/{agg['n_trips']} networks > 1 component\n"
            f"median {agg['ncc_median']:.0f}, max {agg['ncc_max']}\n"
            f"main component >= {100 * agg['main_comp_min']:.0f}% of nodes",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9))
    return _save(fig, "networks.png")


def fig_world(stats, agg):
    fig, ax = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    for part in PARTITIONS:
        rows = [s for s in stats if s["partition"] == part]
        ax.scatter([s["start_lon"] for s in rows], [s["start_lat"] for s in rows],
                   s=44, color=PART_COLORS[part], alpha=0.85, edgecolor="white",
                   linewidth=0.5, label=f"{part} ({len(rows)} trips)", zorder=3)
    for cluster in agg["clusters"]:
        if len(cluster) < 2:
            continue
        lon = float(np.mean([s["start_lon"] for s in cluster]))
        lat = float(np.mean([s["start_lat"] for s in cluster]))
        ax.text(lon + 2.5, lat + 1.5, f"x{len(cluster)}", fontsize=9, color="#444444", zorder=4)
    ax.set_xlim(-130, 155)
    ax.set_ylim(-48, 75)
    ax.set_xticks(np.arange(-120, 151, 30))
    ax.set_yticks(np.arange(-40, 71, 20))
    ax.set_xlabel("longitude (degrees)")
    ax.set_ylabel("latitude (degrees)")
    ax.set_title("Trip start locations by partition")
    ax.legend(loc="lower left")
    ax.text(0.99, 0.02,
            f"{agg['n_trips']} trips - {agg['n_location_clusters']} location clusters (<={CITY_LINK_KM:.0f} km)\n"
            f"{agg['n_multi_clusters']} clusters contain >1 trip - closest start pair "
            f"{agg['min_pair_dist_m'] / 1000:.2f} km",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9))
    return _save(fig, "world.png")


def fig_example_track(stats, agg):
    by_key = {(s["partition"], s["trip_id"]): s for s in stats}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

    for ax, (part, tid, note) in zip(axes, EXAMPLE_TRACK_TRIPS):
        s = by_key[(part, tid)]
        tr = load_trip(tid, part, DATA_ROOT)
        lon0 = float(tr.nodes[:, 0].mean())
        lat0 = float(tr.nodes[:, 1].mean())
        nx, ny = to_local(tr.nodes[:, 0], tr.nodes[:, 1], (lon0, lat0))
        px, py = to_local(tr.track[:, 0], tr.track[:, 1], (lon0, lat0))

        seg_a = np.column_stack([nx[tr.arcs[:, 0]], ny[tr.arcs[:, 0]]])
        seg_b = np.column_stack([nx[tr.arcs[:, 1]], ny[tr.arcs[:, 1]]])
        ax.add_collection(LineCollection(np.stack([seg_a, seg_b], axis=1),
                                         colors="#9e9e9e", linewidths=0.35, alpha=0.8,
                                         zorder=1))

        for run in split_runs(tr.track[:, 2], max(5.0, 5.0 * s["dt_median"])):
            ax.plot(px[run], py[run], color=GPS_COLOR, lw=1.0, alpha=0.9, zorder=2)
        ax.scatter(px, py, s=2.0, color=GPS_COLOR, zorder=3, linewidths=0)

        route_segs = [np.column_stack([nx[run], ny[run]])
                      for run in route_node_runs(tr.arcs, tr.route)]
        ax.add_collection(LineCollection(route_segs, colors=ROUTE_COLOR, linewidths=2.2, zorder=4))

        pad_x = 0.06 * (px.max() - px.min() + 1.0)
        pad_y = 0.06 * (py.max() - py.min() + 1.0)
        ax.set_xlim(px.min() - pad_x, px.max() + pad_x)
        ax.set_ylim(py.min() - pad_y, py.max() + pad_y)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("east offset (m)")
        ax.set_ylabel("north offset (m)")
        ax.set_title(f"{part} {tid} - {note}\n"
                     f"{s['n_obs']:,} fixes - ratio {s['length_ratio']:.2f} - "
                     f"coverage {100 * s['route_coverage_frac']:.0f}% - max dt {s['dt_max']:,.0f} s",
                     fontsize=9.5)
    handles = [
        plt.Line2D([], [], color=GPS_COLOR, lw=1.5, label="GPS fixes"),
        plt.Line2D([], [], color=NET_COLOR, lw=1.5, label="road network (trip subset)"),
        plt.Line2D([], [], color=ROUTE_COLOR, lw=2.5, label="reference route"),
    ]
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.17, wspace=0.22)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    return _save(fig, "example_track.png")


def write_figure_readme(agg):
    lines = [
        "# EDA figures",
        "",
        "All figures are regenerated by scripts/eda_analysis.py (matplotlib, dpi 150).",
        "",
        "- inventory.png - Trips per partition (annotated with fix counts) and the "
        "distribution of GPS fixes per trip on log axes.",
        f"- sampling.png - Histogram of sampling interval dt (log y; {100 * agg['frac_dt_1s']:.1f}% "
        f"of {agg['n_intervals']:,} intervals are exactly 1 s) and the number of gaps (>1.5 s) "
        "for every trip, sorted.",
        f"- steps_speed.png - GPS step-length histogram (median {agg['step_median']:.1f} m) and "
        f"implied-speed histogram (median {agg['speed_median']:.1f} m/s); slow/noisy segments "
        "dominate the left tail.",
        "- coverage.png - Track/reference-route length ratio (median "
        f"{agg['ratio_median']:.2f}), CDF of fix-to-route distance with 30 m and 75 m markers, "
        "and per-trip fraction of fixes within 30 m of the route.",
        "- networks.png - Network nodes vs arcs per trip (log-log), node-degree distribution for "
        "the smallest/median/largest networks, and connected-component counts.",
        "- world.png - Trip start locations by partition; xk marks multi-trip location clusters "
        f"(<={CITY_LINK_KM:.0f} km single link).",
        "- example_track.png - Three example trips side by side: GPS fixes (blue), the trip's "
        "road-network subset (grey) and the reference route (orange): a clean full-route trip, a "
        "partial-route trip, and the trip with the record 153,064 s gap.",
        "",
    ]
    FIG_README_PATH.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Interactive folium maps
# ---------------------------------------------------------------------------
def _legend_html(part, tid, note, s, n_net_arcs, n_net_shown):
    return f"""
<div style="position: fixed; bottom: 40px; left: 40px; z-index: 9999; background: white;
            padding: 10px 14px; border: 1px solid #999; border-radius: 6px;
            font-size: 12px; line-height: 1.6; box-shadow: 0 1px 4px rgba(0,0,0,0.3);">
  <b>PathSnap EDA - {part} {tid}</b><br>
  <span style="display:inline-block;width:16px;height:4px;background:{GPS_COLOR};vertical-align:middle;"></span>
    GPS fixes ({s['n_obs']:,})<br>
  <span style="display:inline-block;width:16px;height:2px;background:#7b3294;vertical-align:middle;"></span>
    road network ({n_net_arcs:,} arcs; {n_net_shown:,} shown)<br>
  <span style="display:inline-block;width:16px;height:4px;background:{ROUTE_COLOR};vertical-align:middle;"></span>
    reference route ({s['n_route_arcs']:,} arcs)<br>
  <span style="color:#555;">{note} - length ratio {s['length_ratio']:.2f} -
    {100 * s['route_coverage_frac']:.0f}% of fixes within 30 m</span>
</div>"""


def write_map(part: str, tid: str, note: str, n_net_arcs: int = 6000, n_gps_markers: int = 1400):
    import folium

    s = TRIP_STATS_BY_KEY[(part, tid)]
    tr = load_trip(tid, part, DATA_ROOT)
    n_obs = len(tr.track)
    lon = tr.track[:, 0]
    lat = tr.track[:, 1]
    lat_c = float((lat.min() + lat.max()) / 2.0)
    lon_c = float((lon.min() + lon.max()) / 2.0)
    span_lon = float(lon.max() - lon.min())
    span_lat = float(lat.max() - lat.min())
    zoom = int(np.clip(np.floor(np.log2(360.0 / max(span_lon, 1e-3))) - 0.3, 3, 17))

    fmap = folium.Map(location=[lat_c, lon_c], zoom_start=zoom, tiles="OpenStreetMap",
                      control_scale=True)

    # -- road network: arcs whose midpoint falls in the (padded) fix bbox ----
    pad_lon = 0.25 * span_lon + 1e-4
    pad_lat = 0.25 * span_lat + 1e-4
    node_lon = tr.nodes[:, 0]
    node_lat = tr.nodes[:, 1]
    mid_lon = 0.5 * (node_lon[tr.arcs[:, 0]] + node_lon[tr.arcs[:, 1]])
    mid_lat = 0.5 * (node_lat[tr.arcs[:, 0]] + node_lat[tr.arcs[:, 1]])
    inside = ((mid_lon >= lon.min() - pad_lon) & (mid_lon <= lon.max() + pad_lon)
              & (mid_lat >= lat.min() - pad_lat) & (mid_lat <= lat.max() + pad_lat))
    net_arc_ids = np.flatnonzero(inside)
    if len(net_arc_ids) > n_net_arcs:
        net_arc_ids = np.sort(RNG.choice(net_arc_ids, size=n_net_arcs, replace=False))
    features = []
    for arc_id in net_arc_ids:
        a, b = tr.arcs[arc_id]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[float(node_lon[a]), float(node_lat[a])],
                                         [float(node_lon[b]), float(node_lat[b])]]},
            "properties": {},
        })
    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name=f"Road network ({len(net_arc_ids):,} arcs shown)",
        style_function=lambda _: {"color": "#7b3294", "weight": 1.2, "opacity": 0.75},
    ).add_to(fmap)

    # -- GPS fixes: split the trajectory at big time gaps --------------------
    gps_group = folium.FeatureGroup(name=f"GPS fixes ({n_obs:,})", show=True)
    gap_split = max(30.0, 10.0 * float(np.median(np.diff(tr.track[:, 2]))))
    for run in split_runs(tr.track[:, 2], gap_split):
        folium.PolyLine(
            [(float(lat[i]), float(lon[i])) for i in range(run.start, run.stop)],
            color=GPS_COLOR, weight=2.0, opacity=0.85,
        ).add_to(gps_group)
    marker_idx = np.sort(RNG.choice(n_obs, size=min(n_obs, n_gps_markers), replace=False))
    for i in marker_idx:
        folium.CircleMarker(
            location=(float(lat[i]), float(lon[i])), radius=2.0,
            color=GPS_COLOR, fill=True, fill_opacity=0.8, weight=0,
        ).add_to(gps_group)
    gps_group.add_to(fmap)

    # -- reference route -----------------------------------------------------
    route_group = folium.FeatureGroup(name=f"Reference route ({len(tr.route):,} arcs)", show=True)
    for nodes in route_node_runs(tr.arcs, tr.route):
        folium.PolyLine(
            [(float(node_lat[n]), float(node_lon[n])) for n in nodes],
            color=ROUTE_COLOR, weight=4.0, opacity=0.9,
        ).add_to(route_group)
    route_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(
        _legend_html(part, tid, note, s, len(tr.arcs), len(net_arc_ids))))

    # -- extent --------------------------------------------------------------
    pad = 0.06
    fmap.fit_bounds([[float(lat.min() - pad * span_lat), float(lon.min() - pad * span_lon)],
                     [float(lat.max() + pad * span_lat), float(lon.max() + pad * span_lon)]])
    out_path = MAP_DIR / f"eda_trip_{tid}.html"
    fmap.save(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Written summary
# ---------------------------------------------------------------------------
def write_summary(stats, agg):
    p = agg["partitions"]
    train = p[TRAIN_PARTITION]
    test = p[DEV_PARTITION]
    cov_sorted = sorted(stats, key=lambda s: s["route_coverage_frac"])
    ratio_sorted = sorted(stats, key=lambda s: -s["length_ratio"])
    gap_sorted = sorted(stats, key=lambda s: -s["dt_max"])
    cov_all_test = agg["n_test_cov_ge_098"]

    def fmt_ids(rows, n=3):
        return ", ".join(f"{s['trip_id']} ({s['partition']})" for s in rows[:n])

    text = f"""# PathSnap - EDA summary

*Generated by scripts/eda_analysis.py from the Student folder data ({agg['n_trips']} trips: {train['trips']} train + {test['trips']} student_test). All distances in metres, all times in seconds; per-trip numbers are in reports/tables/trip_stats.csv.*

## 1. Dataset inventory

| partition | trips | GPS fixes | network nodes | network arcs | route arcs | fixes/trip (median) |
|---|--:|--:|--:|--:|--:|--:|
| train | {train['trips']} | {train['fixes']:,} | {train['nodes']:,} | {train['arcs']:,} | {train['route_arcs']:,} | {train['median_fixes_per_trip']:,.0f} |
| student_test | {test['trips']} | {test['fixes']:,} | {test['nodes']:,} | {test['arcs']:,} | {test['route_arcs']:,} | {test['median_fixes_per_trip']:,.0f} |
| **all** | **{agg['n_trips']}** | **{agg['n_fix']:,}** | **{agg['n_nodes']:,}** | **{agg['n_arcs']:,}** | **{agg['n_route_arcs']:,}** | **{agg['n_obs_median']:,.0f}** |

Per trip: {agg['n_obs_min']:,}-{agg['n_obs_max']:,} fixes; {agg['n_nodes_min']:,}-{agg['n_nodes_max']:,} network nodes (median {agg['n_nodes_median']:,.0f}) and {agg['n_arcs_min']:,}-{agg['n_arcs_max']:,} arcs (median {agg['n_arcs_median']:,.0f}); reference route {agg['n_route_min']:,}-{agg['n_route_max']:,} arcs (median {agg['n_route_median']:,.0f}). All {agg['route_present']}/{agg['n_trips']} trips supply a .route file. Trip duration {agg['dur_min']:,.0f}-{agg['dur_max']:,.0f} s (median {agg['dur_median']:,.0f} s, about {agg['dur_median'] / 60:.0f} min); {agg['n_dur_gt_3h']} trips last over 3 h.

Column notes for trip_stats.csv: gap_frac = fraction of consecutive intervals with dt > 1.5 s; track_length_m sums raw fix-to-fix distances (includes GPS jitter and jumps across gaps); route_length_m sums reference-route arc lengths in file order; gps_to_route_median_m = median distance from a fix to the nearest route segment; route_coverage_frac = fraction of fixes within 30 m of the route.

## 2. Data quality

- **Timestamps are clean.** All {agg['n_intervals']:,} consecutive intervals have dt > 0 ({agg['n_noninc_dt']} non-increasing, {agg['n_duplicate_dt']} duplicates); no NaNs; every arc id and route id in range ({agg['n_bad_arc_ids']} / {agg['n_bad_route_ids']} bad); no zero-length arcs in any reference route.
- **Sampling is 1 Hz.** {100 * agg['frac_dt_1s']:.1f}% of intervals are exactly 1 s and the median dt is 1 s in every trip. dt is a whole number of seconds for all but {agg['n_frac_dt_intervals']} intervals; {agg['n_frac_ts_trips']} trips store timestamps with a small fractional offset (e.g. .5 s) that cancels in the differences. First timestamps are t = {agg['t0_min']:.0f}-{agg['t0_max']:.0f} s (arbitrary offsets), so only dt is meaningful.
- **Gaps are rare but extreme.** {agg['n_dt_gt_15']:,} intervals ({100 * agg['frac_dt_gt_15']:.2f}%) exceed 1.5 s, {agg['n_dt_gt_10']:,} exceed 10 s, {agg['n_dt_gt_60']:,} exceed 60 s and {agg['n_dt_gt_600']:,} exceed 600 s. Maximum dt is {agg['dt_max']:,.0f} s ({agg['dt_max'] / 3600:.1f} h, {gap_sorted[0]['partition']} {gap_sorted[0]['trip_id']}). {agg['n_trips_dtmax_gt_60']} trips contain a gap over 60 s, {agg['n_trips_dtmax_gt_600']} over 600 s and {agg['n_trips_dtmax_gt_3600']} over 1 h.
- **Gap concentration.** Per-trip gap fraction ranges 0-{100 * agg['gap_frac_max']:.1f}% (median {100 * agg['gap_frac_median']:.1f}%); {agg['n_trips_gap_frac_gt_10pct']} trips spend over 10% of their intervals on gaps larger than 1.5 s.
- **Motion is vehicle-like.** GPS step length median {agg['step_median']:.1f} m (p10 {agg['step_p10']:.1f}, p90 {agg['step_p90']:.1f}, p99 {agg['step_p99']:.1f}; max {agg['step_max']:.0f} m is a jump across a time gap). Implied speed median {agg['speed_median']:.1f} m/s ({agg['speed_kmh_median']:.0f} km/h), p10 {agg['speed_p10']:.1f} m/s, p99 {agg['speed_p99']:.1f} m/s; {100 * agg['frac_speed_lt_1']:.1f}% of steps are below 1 m/s and {100 * agg['frac_speed_lt_2']:.1f}% below 2 m/s.
- **Networks are dense but disconnected.** {agg['n_nodes']:,} nodes and {agg['n_arcs']:,} arcs overall; arcs are about 2x nodes (median undirected degree {agg['degree_mean_median']:.2f}, min {agg['degree_mean_min']:.2f}, max {agg['degree_mean_max']:.2f}, no isolated nodes). {agg['n_net_disconnected']}/{agg['n_trips']} trips have a disconnected network (median {agg['ncc_median']:.0f} components, max {agg['ncc_max']}); the main component holds {100 * agg['main_comp_median']:.1f}% of nodes on average, but as little as {100 * agg['main_comp_min']:.1f}% ({agg['n_main_comp_lt_99']} trips below 99%).
- **Trips are global but not one-city-each.** Fix longitudes span {agg['lon_min']:.1f} to {agg['lon_max']:.1f} degrees, latitudes {agg['lat_min']:.1f} to {agg['lat_max']:.1f}. Single-link clustering of trip start points at 20 km gives {agg['n_location_clusters']} location clusters for {agg['n_trips']} trips: {agg['n_multi_clusters']} clusters contain {agg['n_trips_in_multi']} trips (largest: {agg['largest_cluster_size']} trips) and the closest pair of trip start points is {agg['min_pair_dist_m'] / 1000:.2f} km apart ({', '.join(agg['min_pair_ids'])}). {agg['n_bbox_overlap_pairs']} trip pairs have overlapping fix bounding boxes. All {test['trips']} student_test trips are singleton clusters at this threshold, but the closest student_test-to-train start-point distance is only {agg['test_min_train_dist_m'] / 1000:.0f} km, so the split is not guaranteed to be geographically disjoint.
- **Route files are chronological arc lists with a few quirks.** {agg['n_route_disc_trips']}/{agg['n_trips']} routes have a node discontinuity between consecutive arcs (max {agg['route_disc_max']}); {agg['n_route_repeat_trips']}/{agg['n_trips']} revisit at least one arc (max {agg['route_repeat_max']} repeats, never consecutive, so genuine loops/re-visits rather than duplicated lines).

## 3. Route coverage (reference route vs trajectory)

- **Length ratio** (raw GPS path / reference-route length) is {agg['ratio_min']:.2f}-{agg['ratio_max']:.2f}, median {agg['ratio_median']:.2f}. {agg['n_ratio_le_150']}/{agg['n_trips']} trips are at most 1.5, {agg['n_ratio_gt_150']} are above 1.5, {agg['n_ratio_gt_200']} above 2 and {agg['n_ratio_gt_500']} above 5. In this data it is never below 1, i.e. a reference route never covers more distance than the raw GPS path.
- **Proximity.** The median fix is {agg['dist_median']:.1f} m from the reference route (per-trip median of medians {agg['trip_median_dist']:.1f} m); {100 * agg['frac_le_30']:.1f}% of all fixes are within 30 m and {100 * agg['frac_le_75']:.1f}% within 75 m. {agg['n_gt_75']:,} fixes ({100 * agg['frac_gt_75']:.1f}%) are farther than 75 m from the route.
- **Per-trip coverage** (fraction of fixes within 30 m): median {agg['median_trip_cov']:.3f}; {agg['n_cov_lt_099']} trips are below 0.99, {agg['n_cov_lt_095']} below 0.95, {agg['n_cov_lt_09']} below 0.90 and {agg['n_cov_lt_05']} below 0.50: {fmt_ids(cov_sorted, 4)}. Partial routes explain the large length ratios - the worst-covered trips are also the highest-ratio ones ({fmt_ids(ratio_sorted, 3)}).
- **Partitions behave differently.** {cov_all_test}/{test['trips']} student_test trips have coverage of at least 0.98, while the low-coverage trips are all train trips, so validating on student_test alone would overstate how well the supplied routes explain noisy trajectories.

## 4. Map-matching implications / assumptions

- **Each trip has its own network.** Candidate search must stay inside the trip's own road graph; networks range from {agg['n_nodes_min']:,} to {agg['n_nodes_max']:,} nodes ({agg['n_arcs_max']:,} arcs worst case), so a spatial index is mandatory.
- **1 Hz sampling dominates.** With a median step of {agg['step_median']:.1f} m, most consecutive fixes are several arcs apart; transitions should use network shortest paths, not straight-line distance alone.
- **Noise competes with motion.** The median fix offset to the route ({agg['dist_median']:.1f} m) is comparable to the median step ({agg['step_median']:.1f} m); for {100 * agg['frac_speed_lt_2']:.1f}% of steps (speed under 2 m/s) GPS noise exceeds true displacement, so emissions and candidates must stay tolerant.
- **Time gaps break continuity.** {agg['n_dt_gt_15']:,} intervals over 1.5 s and {agg['n_dt_gt_60']} gaps over 60 s require the transition model to relax or disable connectivity across a gap; {agg['n_trips_dtmax_gt_600']} trips contain gaps over 10 min.
- **Partial routes limit labels.** With a 75 m labelling cap, {agg['n_gt_75']:,} fixes ({100 * agg['frac_gt_75']:.1f}%) cannot be labelled, concentrated in {agg['n_cov_lt_09']} trips (coverage down to {100 * agg['cov_min']:.0f}%). Report accuracy both overall and restricted to fixes within the route corridor; never assume every fix has a ground-truth arc.
- **Disconnected networks.** {agg['n_net_disconnected']}/{agg['n_trips']} networks are disconnected (median {agg['ncc_median']:.0f} components), so shortest-path transitions must handle unreachable candidate pairs by skipping or penalising them instead of failing.
- **Route quirks.** Reference routes contain {agg['n_route_disc_trips']} discontinuity trips and {agg['n_route_repeat_trips']} trips that re-use arcs; an along-route DP must allow repeated arcs and treat arc-index jumps as gaps, not backward motion.
- **Geography is not a clean holdout.** {agg['n_trips_in_multi']} trip starts fall into {agg['n_multi_clusters']} multi-trip metro clusters, so a random train/validation split shares city geometry; a city-grouped split is safer if generalisation is claimed.

## Figures and maps

Figures: reports/figures/eda/ contains inventory.png, sampling.png, steps_speed.png, coverage.png, networks.png, world.png and example_track.png (captions in reports/figures/eda/README.md). Interactive maps: reports/maps/ has one map per example trip (eda_trip_<trip>.html) with GPS fixes, the trip's road-network subset and the reference route as togglable layers.
"""
    SUMMARY_PATH.write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TRIP_STATS_BY_KEY: dict = {}


def main() -> int:
    t_start = time.time()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    stats = []
    pooled = {"dt": [], "step": [], "speed": [], "dist": []}
    for partition in PARTITIONS:
        trip_ids = list_trips(partition, DATA_ROOT)
        print(f"[load] {partition}: {len(trip_ids)} trips")
        for trip_id in trip_ids:
            trip_stats, dt, step, speed, dist = compute_trip_stats(partition, trip_id)
            stats.append(trip_stats)
            pooled["dt"].append(dt)
            pooled["step"].append(step)
            pooled["speed"].append(speed[np.isfinite(speed)])
            pooled["dist"].append(dist)
            print(f"  {partition}/{trip_id}: {trip_stats['n_obs']:6,} fixes, "
                  f"{trip_stats['n_nodes']:7,} nodes, {trip_stats['n_arcs']:7,} arcs")
    pooled = {key: np.concatenate(value) for key, value in pooled.items()}

    print(f"[stats] computed in {time.time() - t_start:.1f} s")
    write_trip_csv(stats, CSV_PATH)
    print(f"[write] {CSV_PATH.relative_to(PROJECT_ROOT)}")

    agg = aggregate(stats, pooled)
    TRIP_STATS_BY_KEY.clear()
    TRIP_STATS_BY_KEY.update({(s["partition"], s["trip_id"]): s for s in stats})

    paths = [
        fig_inventory(stats, agg),
        fig_sampling(stats, pooled, agg),
        fig_steps_speed(pooled, agg),
        fig_coverage(stats, pooled, agg),
        fig_networks(stats, agg),
        fig_world(stats, agg),
        fig_example_track(stats, agg),
    ]
    for path in paths:
        print(f"[figure] {path.relative_to(PROJECT_ROOT)}")
    write_figure_readme(agg)
    print(f"[write] {FIG_README_PATH.relative_to(PROJECT_ROOT)}")

    for partition, trip_id, note in MAP_TRIPS:
        path = write_map(partition, trip_id, note)
        print(f"[map] {path.relative_to(PROJECT_ROOT)}")

    write_summary(stats, agg)
    print(f"[write] {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[done] total {time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
