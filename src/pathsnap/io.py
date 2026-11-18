"""Raw dataset loading for PathSnap.

File layout (one folder per trip)::

    <trip_id>.track   longitude latitude timestamp_seconds
    <trip_id>.nodes   longitude latitude                       (node id = row)
    <trip_id>.arcs    from_node_id to_node_id                  (arc id = row)
    <trip_id>.route   arc_id, chronological ground-truth route (may be absent)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import (DATA_ROOT, DEV_PARTITION, EVAL_DATA_ROOT, EVAL_PARTITION,
                     TRAIN_PARTITION)


@dataclass
class Trip:
    trip_id: str
    partition: str
    directory: Path
    track: np.ndarray                 # (n, 3) lon, lat, t
    nodes: np.ndarray                 # (m, 2) lon, lat
    arcs: np.ndarray                  # (k, 2) int node ids
    route: Optional[np.ndarray] = None  # (r,) int arc ids
    issues: dict = field(default_factory=dict)

    @property
    def has_route(self) -> bool:
        return self.route is not None and len(self.route) > 0


def partition_roots(partition: str, data_root: Path | str = DATA_ROOT) -> list[Path]:
    """Candidate folders for a partition (evaluator root first)."""
    roots: list[Path] = []
    if partition == EVAL_PARTITION:
        roots.append(Path(EVAL_DATA_ROOT) / partition)
    roots.append(Path(data_root) / partition)
    return roots


def list_trips(partition: str, data_root: Path | str = DATA_ROOT):
    """Sorted trip ids present in a partition folder."""
    root = None
    for candidate in partition_roots(partition, data_root):
        if candidate.is_dir():
            root = candidate
            break
    if root is None:
        return []
    out = []
    for p in sorted(root.iterdir()):
        # accept any folder that carries a <name>.track file: the evaluator
        # partition may use non-numeric trip ids
        if p.is_dir() and (p / f"{p.name}.track").exists():
            out.append(p.name)
    return out


def load_trip(trip_id: str, partition: str, data_root: Path | str = DATA_ROOT,
              with_route: bool = True) -> Trip:
    d = None
    for candidate in partition_roots(partition, data_root):
        if (candidate / trip_id).is_dir():
            d = candidate / trip_id
            break
    if d is None:
        raise FileNotFoundError(
            f"trip {trip_id} not found in partition {partition} "
            f"({[str(c) for c in partition_roots(partition, data_root)]})")
    track = np.loadtxt(d / f"{trip_id}.track", dtype=np.float64)
    nodes = np.loadtxt(d / f"{trip_id}.nodes", dtype=np.float64)
    arcs = np.loadtxt(d / f"{trip_id}.arcs", dtype=np.int64)
    track = np.atleast_2d(track)
    nodes = np.atleast_2d(nodes)
    arcs = np.atleast_2d(arcs)
    if track.shape[1] != 3:
        raise ValueError(f"{trip_id}: track must have 3 columns, got {track.shape}")
    if nodes.shape[1] != 2:
        raise ValueError(f"{trip_id}: nodes must have 2 columns, got {nodes.shape}")
    if arcs.shape[1] != 2:
        raise ValueError(f"{trip_id}: arcs must have 2 columns, got {arcs.shape}")
    route = None
    route_path = d / f"{trip_id}.route"
    if with_route and route_path.exists():
        route = np.atleast_1d(np.loadtxt(route_path, dtype=np.int64)).ravel()
    return Trip(trip_id=trip_id, partition=partition, directory=d,
                track=track, nodes=nodes, arcs=arcs, route=route)


def validate_trip(trip: Trip) -> dict:
    """Basic data-quality report used by the EDA and by the API at load time."""
    tr = trip.track
    issues = {
        "n_obs": int(len(tr)),
        "n_nodes": int(len(trip.nodes)),
        "n_arcs": int(len(trip.arcs)),
        "n_route_arcs": int(len(trip.route)) if trip.has_route else 0,
        "nan_track": int(np.isnan(tr).any(axis=1).sum()),
        "non_increasing_time": int((np.diff(tr[:, 2]) <= 0).sum()),
        "out_of_range_arc_nodes": int(((trip.arcs < 0).any(axis=1) |
                                       (trip.arcs >= len(trip.nodes)).any(axis=1)).sum()),
        "lon_range": [float(np.nanmin(tr[:, 0])), float(np.nanmax(tr[:, 0]))],
        "lat_range": [float(np.nanmin(tr[:, 1])), float(np.nanmax(tr[:, 1]))],
    }
    dt = np.diff(tr[:, 2])
    issues["dt_median_s"] = float(np.median(dt)) if len(dt) else 0.0
    issues["dt_max_s"] = float(dt.max()) if len(dt) else 0.0
    issues["short_steps"] = int((dt < 1.0).sum())
    issues["large_gaps_gt_10s"] = int((dt > 10.0).sum())
    if trip.has_route:
        r = trip.route
        issues["route_out_of_range"] = int(((r < 0) | (r >= len(trip.arcs))).sum())
        a = trip.arcs[r]
        issues["route_discontinuities"] = int((a[:-1, 1] != a[1:, 0]).sum())
        issues["route_repeated_arcs"] = int(len(r) - len(np.unique(r)))
    return issues


def discover_partitions(data_root: Path | str = DATA_ROOT) -> dict:
    """Return {partition: [trip ids]} for everything available locally."""
    return {p: list_trips(p, data_root) for p in (TRAIN_PARTITION, DEV_PARTITION, EVAL_PARTITION)}


def pick_eval_partition(data_root: Path | str = DATA_ROOT):
    """Prefer the evaluator partition, fall back to the development partition."""
    for p in (EVAL_PARTITION, DEV_PARTITION):
        if list_trips(p, data_root):
            return p
    return None
