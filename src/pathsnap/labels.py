"""Ground-truth labelling of GPS fixes against the reference route.

The raw .route file is a chronological list of arc ids, not one label per GPS
fix.  We project every fix onto the reference-route segments and pick the arc
that best explains it.  A small dynamic program enforces the physical fact
that the along-route coordinate is non-decreasing (vehicles do not teleport
backwards along their own route).  Fixes farther than LABEL_MAX_DIST_M from
the reference route are left unlabelled: some supplied routes only cover part
of the trajectory, which the dataset guidelines warn about.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidates import TripNetwork
from .config import LABEL_MAX_DIST_M, LABEL_SIGMA_M
from .io import Trip

INVALID_COST = 1e6
ALLOWED_BACK_M = 10.0   # GPS jitter can move the projected point slightly backwards
BACK_SCALE_M = 20.0     # soft penalty scale for real backwards jumps


@dataclass
class LabelResult:
    arc: np.ndarray            # (n,) reference arc id or -1
    dist: np.ndarray           # (n,) distance to the reference route, metres
    along: np.ndarray          # (n,) distance along the reference route, metres
    route_idx: np.ndarray      # (n,) index in the .route sequence (-1 = unlabelled)
    valid: np.ndarray          # (n,) bool


def _route_segments(net: TripNetwork, trip: Trip):
    route = trip.route
    ax = net.ax[route]; ay = net.ay[route]
    bx = net.bx[route]; by = net.by[route]
    seg_len = np.hypot(bx - ax, by - ay)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    return ax, ay, bx, by, seg_len, cum


def _project_chunk(px, py, ax, ay, bx, by, seg_len, cum, k):
    dx = bx - ax; dy = by - ay
    seg2 = dx * dx + dy * dy
    safe = np.where(seg2 > 1e-12, seg2, 1.0)
    t = ((px[:, None] - ax[None, :]) * dx[None, :]
         + (py[:, None] - ay[None, :]) * dy[None, :]) / safe[None, :]
    t = np.clip(t, 0.0, 1.0)
    fx = ax[None, :] + t * dx[None, :]
    fy = ay[None, :] + t * dy[None, :]
    d = np.hypot(px[:, None] - fx, py[:, None] - fy)
    kk = min(k, d.shape[1])
    idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    dsel = np.take_along_axis(d, idx, axis=1)
    order = np.argsort(dsel, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    dsel = np.take_along_axis(dsel, order, axis=1)
    tsel = np.take_along_axis(t, idx, axis=1)
    along = cum[idx] + tsel * seg_len[idx]
    return idx, dsel, along


def label_observations(net: TripNetwork, trip: Trip, obs_xy: np.ndarray,
                       top_k: int = 6, sigma: float = LABEL_SIGMA_M,
                       max_dist: float = LABEL_MAX_DIST_M) -> LabelResult:
    n = len(obs_xy)
    if not trip.has_route or n == 0:
        return LabelResult(arc=np.full(n, -1, np.int64), dist=np.full(n, np.inf),
                           along=np.full(n, np.nan), route_idx=np.full(n, -1, np.int64),
                           valid=np.zeros(n, bool))
    ax, ay, bx, by, seg_len, cum = _route_segments(net, trip)
    best_idx = np.empty((n, top_k), np.int64)
    best_d = np.empty((n, top_k))
    best_along = np.empty((n, top_k))
    chunk = 512
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        idx, dsel, along = _project_chunk(obs_xy[s:e, 0], obs_xy[s:e, 1],
                                          ax, ay, bx, by, seg_len, cum, top_k)
        best_idx[s:e] = idx
        best_d[s:e] = dsel
        best_along[s:e] = along

    k = top_k
    owns = (best_d ** 2) / (sigma * sigma)
    owns = np.where(best_d <= max_dist, owns, INVALID_COST)
    cand_valid = best_d[:, 0] <= max_dist      # obs close enough to the route

    back = np.zeros((n, k), np.int64)
    prev_cost = owns[0].copy()
    prev_along = best_along[0].copy()
    prev_ok = bool(cand_valid[0])
    cols = np.arange(k)
    for i in range(1, n):
        if not prev_ok:
            j = int(np.argmin(prev_cost))
            back[i] = j
            if not cand_valid[i]:
                prev_cost = np.full(k, prev_cost[j])
                prev_along = np.full(k, -np.inf)
                prev_ok = False
            else:
                prev_cost = owns[i] + prev_cost[j]
                prev_along = best_along[i].copy()
                prev_ok = True
            continue
        if not cand_valid[i]:
            j = int(np.argmin(prev_cost))
            back[i] = j
            prev_cost = np.full(k, prev_cost[j])
            prev_along = np.full(k, -np.inf)
            prev_ok = False
            continue
        deficit = np.maximum(prev_along[:, None] - best_along[i][None, :] - ALLOWED_BACK_M, 0.0)
        mat = prev_cost[:, None] + (deficit / BACK_SCALE_M) ** 2
        jbest = np.argmin(mat, axis=0)
        back[i] = jbest
        prev_cost = owns[i] + mat[jbest, cols]
        prev_along = best_along[i].copy()
        prev_ok = True

    route_idx = np.full(n, -1, np.int64)
    sel = int(np.argmin(prev_cost))
    route_idx[-1] = sel if cand_valid[-1] else -1
    for i in range(n - 1, 0, -1):
        sel = int(back[i, sel])
        if cand_valid[i - 1]:
            route_idx[i - 1] = sel
    # rebuild labels from the chosen states
    dist = np.full(n, np.inf)
    along = np.full(n, np.nan)
    arc = np.full(n, -1, np.int64)
    rows = np.where(route_idx >= 0)[0]
    if len(rows):
        seg = np.empty(len(rows), np.int64)
        cur = int(np.argmin(prev_cost))
        # route_idx already holds state indices for all valid observations
        sel_states = route_idx[rows]
        dist[rows] = best_d[rows, sel_states]
        along[rows] = best_along[rows, sel_states]
        seg[:] = best_idx[rows, sel_states]
        arc[rows] = trip.route[seg]
    valid = (route_idx >= 0) & (dist <= max_dist) & (arc >= 0)
    arc[~valid] = -1
    return LabelResult(arc=arc, dist=dist, along=along, route_idx=route_idx, valid=valid)


def label_diagnostics(labels: LabelResult, trip: Trip) -> dict:
    """"Summary of how cleanly the reference route explains the GPS fixes."""
    n = len(labels.arc)
    d = labels.dist
    idx = labels.route_idx
    ok = labels.valid
    return {
        "n_obs": int(n),
        "labelled_fraction": float(ok.mean()) if n else 0.0,
        "median_dist_m": float(np.median(d)) if n else 0.0,
        "p95_dist_m": float(np.percentile(d, 95)) if n else 0.0,
        "max_dist_m": float(np.max(d)) if n else 0.0,
        "backward_steps": int((np.diff(labels.along[ok]) < -ALLOWED_BACK_M).sum()) if ok.sum() > 1 else 0,
        "unique_arcs_referenced": int(len(np.unique(labels.arc[ok]))) if n else 0,
        "route_arcs": int(len(trip.route)) if trip.has_route else 0,
    }
